# agent/reviews_weekly_report_agent.py
from __future__ import annotations

import os
import io
import re
import csv
import ssl
import json
import math
import time
import base64
import smtplib
import logging
from email.message import EmailMessage
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# --- наши модули (не трогаем surveys) ---
try:
    from agent import reviews_io
    from agent import reviews_core
except ModuleNotFoundError:
    import reviews_io  # type: ignore
    import reviews_core  # type: ignore

# --- периодизация (общая с surveys) ---
try:
    from agent.metrics_core import iso_week_monday, period_ranges_for_week
except ModuleNotFoundError:
    from metrics_core import iso_week_monday, period_ranges_for_week  # type: ignore

# --- Google API ---
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# -----------------------------------------------------------------------------
# Константы / логгер
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("reviews_weekly_agent")

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
SHEETS_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

HISTORY_SHEET_NAME = "reviews_history"  # отдельная вкладка в общем SHEETS_HISTORY_ID

# шаблон имён файлов в Google Drive:
# Reviews_DD-MM-YYYY.xls  ИЛИ  reviews_YYYY-MM-DD.xls — поддерживаем оба
_RE_FNAME_DMY = re.compile(r"(?i)\breviews?_?(\d{2})-(\d{2})-(\d{4})\b")
_RE_FNAME_YMD = re.compile(r"(?i)\breviews?_?(\d{4})-(\d{2})-(\d{2})\b")


# -----------------------------------------------------------------------------
# Утилиты
# -----------------------------------------------------------------------------
def _today() -> date:
    # Агент запускается по понедельникам утром по Мск; «последняя завершившаяся неделя» = неделя, заканчивающаяся вчера
    return datetime.utcnow().date()

def _week_key_from_date(d: date) -> str:
    iso_year, iso_week, _ = d.isocalendar()
    return f"{iso_year}-W{iso_week:02d}"

def _last_completed_week_key(anchor: Optional[str] = None) -> str:
    if anchor:
        return anchor
    # Берём «вчера» — это безопасно для запуска по понедельникам утром
    y = _today() - timedelta(days=1)
    return _week_key_from_date(y)

def _build_credentials(sa_json_path: str):
    return service_account.Credentials.from_service_account_file(sa_json_path, scopes=DRIVE_SCOPES + SHEETS_SCOPES)

def _b64_to_sa_json_path(b64_env: str) -> str:
    """
    Декодирует GOOGLE_SERVICE_ACCOUNT_JSON_B64 в /tmp/sa.json и возвращает путь.
    """
    content_b64 = os.environ.get(b64_env) or ""
    if not content_b64:
        raise RuntimeError(f"{b64_env} не задан.")
    raw = base64.b64decode(content_b64)
    out_path = "/tmp/sa.json"
    with open(out_path, "wb") as f:
        f.write(raw)
    return out_path

def _build_credentials_from_b64():
    sa_path = _b64_to_sa_json_path("GOOGLE_SERVICE_ACCOUNT_JSON_B64")
    return service_account.Credentials.from_service_account_file(sa_path, scopes=DRIVE_SCOPES + SHEETS_SCOPES)

def _build_drive(creds):
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def _build_sheets(creds):
    return build("sheets", "v4", credentials=creds, cache_discovery=False)

def _parse_date_from_name(name: str) -> Optional[date]:
    """
    Поддерживаем оба паттерна:
    - Reviews_DD-MM-YYYY.xls
    - reviews_YYYY-MM-DD.xls
    """
    if not name:
        return None
    m = _RE_FNAME_DMY.search(name)
    if m:
        dd, mm, yyyy = m.groups()
        try:
            return date(int(yyyy), int(mm), int(dd))
        except Exception:
            pass
    m = _RE_FNAME_YMD.search(name)
    if m:
        yyyy, mm, dd = m.groups()
        try:
            return date(int(yyyy), int(mm), int(dd))
        except Exception:
            pass
    return None

def _drive_list_files_in_folder(drive, folder_id: str) -> List[Dict[str, Any]]:
    q = f"'{folder_id}' in parents and trashed = false"
    fields = "nextPageToken, files(id, name, mimeType, modifiedTime, size)"
    files: List[Dict[str, Any]] = []
    page_token = None
    while True:
        resp = drive.files().list(q=q, fields=fields, pageToken=page_token, pageSize=1000, orderBy="modifiedTime desc").execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files

def _drive_download_file_bytes(drive, file_id: str) -> bytes:
    req = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, req)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    return fh.getvalue()

def _pick_best_reviews_file(files: List[Dict[str, Any]], week_end: date) -> Dict[str, Any]:
    """
    Берём файл с максимальной датой в имени, не позже week_end.
    Если ни у кого дата не парсится — берём самый свежий по modifiedTime.
    """
    candidates: List[Tuple[date, Dict[str, Any]]] = []
    for f in files:
        d = _parse_date_from_name(f.get("name", ""))
        if d is not None and d <= week_end:
            candidates.append((d, f))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # fallback: свежий modifiedTime
    files_sorted = sorted(files, key=lambda x: x.get("modifiedTime", ""), reverse=True)
    if not files_sorted:
        raise FileNotFoundError("В папке нет файлов с отзывами.")
    return files_sorted[0]

def _trim_text(s: str, n: int = 280) -> str:
    s = (s or "").strip()
    return (s[: n - 1] + "…") if len(s) > n else s

def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _ensure_sheet_exists(sheets, spreadsheet_id: str, title: str) -> None:
    meta = sheets.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    for sh in meta.get("sheets", []):
        if sh.get("properties", {}).get("title") == title:
            return
    # создаём лист
    body = {"requests": [{"addSheet": {"properties": {"title": title}}}]}
    sheets.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body=body).execute()

def _read_sheet_as_df(sheets, spreadsheet_id: str, title: str) -> pd.DataFrame:
    try:
        resp = sheets.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range=f"'{title}'!A:Z").execute()
        values = resp.get("values", [])
        if not values:
            return pd.DataFrame()
        header = values[0]
        rows = values[1:]
        return pd.DataFrame(rows, columns=header)
    except Exception:
        return pd.DataFrame()

def _parse_history_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводим types и базовые поля. Ожидаемые колонки:
    date, iso_week, source, lang, rating10, sentiment_score, sentiment_overall,
    aspects, topics, has_response, review_key, text_trimmed, ingested_at
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "review_id","source","created_at","week_key","rating10",
            "sentiment_overall","sentiment_score","lang","topics","aspects","raw_text",
        ])
    d = df.copy()
    # Канонизируем имена
    cols = {c.lower(): c for c in d.columns}
    def col(name: str) -> str:
        return cols.get(name, name)
    # Типы
    d[col("date")] = pd.to_datetime(d[col("date")], errors="coerce").dt.date
    d[col("rating10")] = pd.to_numeric(d.get(col("rating10")), errors="coerce")
    d[col("sentiment_score")] = pd.to_numeric(d.get(col("sentiment_score")), errors="coerce")
    d[col("iso_week")] = d[col("iso_week")].astype(str)
    d[col("source")] = d[col("source")].astype(str)
    d[col("lang")] = d[col("lang")].astype(str)
    d[col("sentiment_overall")] = d[col("sentiment_overall")].astype(str)
    # Сборка в формат df_reviews
    out = pd.DataFrame({
        "review_id": d.get(col("review_key")).astype(str),
        "source": d.get(col("source")).astype(str),
        "created_at": pd.to_datetime(d.get(col("date")), errors="coerce"),
        "week_key": d.get(col("iso_week")).astype(str),
        "rating10": d.get(col("rating10")),
        "sentiment_overall": d.get(col("sentiment_overall")).astype(str),
        "sentiment_score": d.get(col("sentiment_score")),
        "lang": d.get(col("lang")).astype(str),
        "topics": d.get(col("topics")).fillna(""),
        "aspects": d.get(col("aspects")).fillna(""),
        "raw_text": d.get(col("text_trimmed")).fillna(""),
    })
    # фильтр валидных дат
    out = out[~out["created_at"].isna()].copy()
    return out

def _append_rows_to_sheet(sheets, spreadsheet_id: str, title: str, rows: List[List[Any]]) -> None:
    if not rows:
        return
    body = {"values": rows}
    sheets.spreadsheets().values().append(
        spreadsheetId=spreadsheet_id,
        range=f"'{title}'!A1",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body=body,
    ).execute()

def _upsert_reviews_history_week(
    sheets, spreadsheet_id: str, df_reviews: pd.DataFrame, df_raw_with_has_response: pd.DataFrame, week_key: str
) -> int:
    """
    История = канонический слой. Пишем строки только за заданную неделю, не дублируя review_key.
    Схема:
      date, iso_week, source, lang, rating10,
      sentiment_score, sentiment_overall,
      aspects, topics, has_response,
      review_key, text_trimmed, ingested_at
    """
    _ensure_sheet_exists(sheets, spreadsheet_id, HISTORY_SHEET_NAME)
    df_sheet = _read_sheet_as_df(sheets, spreadsheet_id, HISTORY_SHEET_NAME)

    # текущие ключи недели (если лист пуст — просто пишем)
    existing_keys: set = set()
    if not df_sheet.empty and "iso_week" in df_sheet.columns and "review_key" in df_sheet.columns:
        existing_keys = set(
            df_sheet.loc[df_sheet["iso_week"] == week_key, "review_key"].astype(str).tolist()
        )

    # добавим has_response из сырой таблицы по review_id
    # df_raw_with_has_response: columns: review_id, has_response
    raw_map = {}
    if not df_raw_with_has_response.empty:
        raw_map = dict(zip(df_raw_with_has_response["review_id"], df_raw_with_has_response["has_response"]))

    to_append: List[List[Any]] = []
    cols = [
        "date","iso_week","source","lang","rating10",
        "sentiment_score","sentiment_overall",
        "aspects","topics","has_response",
        "review_key","text_trimmed","ingested_at"
    ]

    now = _now_iso()
    for _, row in df_reviews.iterrows():
        if row.get("week_key") != week_key:
            continue
        review_id = str(row.get("review_id"))
        review_key = review_id  # в нашем ядре review_id уже стабилен; если отдельный key нужен — можно прокинуть
        if review_key in existing_keys:
            continue

        aspects = row.get("aspects") or ""
        topics  = row.get("topics") or ""
        # подтянем has_response
        has_resp = raw_map.get(review_id, "")
        text_trimmed = _trim_text(str(row.get("raw_text") or ""), 280)

        vals = [
            str(row.get("created_at") or ""),
            week_key,
            str(row.get("source") or ""),
            str(row.get("lang") or ""),
            "" if pd.isna(row.get("rating10")) else float(row.get("rating10")),
            "" if pd.isna(row.get("sentiment_score")) else float(row.get("sentiment_score")),
            str(row.get("sentiment_overall") or ""),
            aspects,
            topics,
            has_resp,
            review_key,
            text_trimmed,
            now,
        ]
        to_append.append(vals)

    if to_append:
        _append_rows_to_sheet(sheets, spreadsheet_id, HISTORY_SHEET_NAME, [cols] if df_sheet.empty else [])
        _append_rows_to_sheet(sheets, spreadsheet_id, HISTORY_SHEET_NAME, to_append)
    return len(to_append)

def _render_sources_block_html(period_to_df: Dict[str, pd.DataFrame]) -> str:
    """
    Рендерим таблицу «Источник × Период».
    Ожидает: {'week':df, 'mtd':df, 'qtd':df, 'ytd':df, 'all':df}
    В каждой ячейке показываем: средняя /10 и нативная (для five-star источников), кол-во, %позитив/%негатив.
    """
    labels = {
        "week": "Неделя",
        "mtd": "Месяц-to-date",
        "qtd": "Квартал-to-date",
        "ytd": "Год-to-date",
        "all": "Исторический фон",
    }
    # Собираем множество всех источников из всех периодов
    all_sources = set()
    for df in period_to_df.values():
        if df is not None and len(df) > 0:
            all_sources.update(df["source"].astype(str).unique().tolist())
    if not all_sources:
        return "<p>Нет данных по источникам.</p>"

    # Вспомогательный индекс: period->code->row
    px: Dict[str, Dict[str, pd.Series]] = {}
    for pkey, df in period_to_df.items():
        pm = {}
        if df is not None and len(df) > 0:
            for _, r in df.iterrows():
                pm[str(r["source"])] = r
        px[pkey] = pm

    # Строим таблицу с блоками по периодам
    rows = []
    for code in sorted(all_sources):
        name = reviews_io.source_display_name(code)
        cell_html = []
        for pkey in ["week","mtd","qtd","ytd","all"]:
            r = px.get(pkey, {}).get(code)
            if r is None:
                cell_html.append("<td></td><td></td><td></td><td></td>")
                continue
            avg10 = "" if pd.isna(r["avg10"]) else f"{float(r['avg10']):.2f}"
            native = reviews_io.to_native_for_sources_block(
                rating10=float(r["avg10"]) if not pd.isna(r["avg10"]) else None,
                source_code=code
            )
            native_str = "" if native is None else f"{native:.2f}"
            cell_html.append(
                f"<td style='text-align:right'>{avg10}</td>"
                f"<td style='text-align:right'>{native_str}</td>"
                f"<td style='text-align:right'>{int(r['reviews'])}</td>"
                f"<td style='text-align:right'>{float(r['neg_pct'])*100:.1f}%</td>"
            )
        rows.append(f"<tr><td>{name}</td>{''.join(cell_html)}</tr>")

    head_cols = "".join(
        f"<th colspan='4'>{labels[k]}</th>" for k in ["week","mtd","qtd","ytd","all"]
    )
    subhead = ("<tr><th>Источник</th>" +
               "<th>/10</th><th>Нативная</th><th>Кол-во</th><th>% негатив</th>" * 5 +
               "</tr>")

    table = (
        "<table border='1' cellspacing='0' cellpadding='6'>"
        f"<thead><tr><th></th>{head_cols}</tr>{subhead}</thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )
    return table

def _send_email(
    smtp_host: str,
    smtp_user: str,
    smtp_pass: str,
    smtp_from: str,
    recipients: List[str],
    subject: str,
    html_body: str,
    attachments: List[Tuple[str, bytes]] | None = None,
) -> None:
    if not recipients:
        LOG.warning("RECIPIENTS пуст — письмо не будет отправлено.")
        return

    msg = EmailMessage()
    msg["From"] = smtp_from
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.set_content("Ваш почтовый клиент не поддерживает HTML.")
    msg.add_alternative(html_body, subtype="html")

    for fname, data in (attachments or []):
        msg.add_attachment(data, maintype="application", subtype="octet-stream", filename=fname)

    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_host, 465, context=ctx) as server:
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)

def _fmt_pct(x: float) -> str:
    try:
        return f"{x*100:.1f}%"
    except Exception:
        return ""

def _section_A_summary(week_df: pd.DataFrame, mtd_df: pd.DataFrame, qtd_df: pd.DataFrame, ytd_df: pd.DataFrame, all_df: pd.DataFrame) -> str:
    def _basic(df):
        if df is None or len(df) == 0:
            return (0, float("nan"), float("nan"), float("nan"))
        total = int(df["review_id"].nunique())
        avg10 = float(df["rating10"].mean()) if "rating10" in df.columns else float("nan")
        pos = ( (df["sentiment_overall"]=="positive") | (df["rating10"]>=9) ).mean()
        neg = ( (df["sentiment_overall"]=="negative") | (df["rating10"]<=6) ).mean()
        return (total, avg10, pos, neg)

    t_w, a_w, p_w, n_w = _basic(week_df)
    t_m, a_m, p_m, n_m = _basic(mtd_df)
    t_q, a_q, p_q, n_q = _basic(qtd_df)
    t_y, a_y, p_y, n_y = _basic(ytd_df)
    t_a, a_a, p_a, n_a = _basic(all_df)

    def _line(lbl, total, avg, pos, neg):
        avg_s = "" if (avg!=avg) else f"{avg:.2f}"
        return f"<p><b>{lbl}:</b> средняя {avg_s} /10 · позитив {_fmt_pct(pos)} · негатив {_fmt_pct(neg)} · отзывов {total}</p>"

    return (
        _line("Неделя", t_w, a_w, p_w, n_w) +
        _line("Месяц-to-date", t_m, a_m, p_m, n_m) +
        _line("Квартал-to-date", t_q, a_q, p_q, n_q) +
        _line("Год-to-date", t_y, a_y, p_y, n_y) +
        _line("Исторический фон", t_a, a_a, p_a, n_a)
    )

def _section_B_drivers_and_risks(aspects_week: pd.DataFrame) -> Tuple[str, str]:
    """
    Формируем bullets для B1/B2:
    - драйверы: top по positive_impact_index
    - риски:   top по negative_impact_index
    Порог отбора — минимум 2 упоминания (reviews_with_aspect>=2).
    """
    if aspects_week is None or len(aspects_week) == 0:
        return ("<p>На этой неделе не выделяется единого фактора, который тянет оценку вверх.</p>",
                "<p>Системных жалоб, которые тянут оценки вниз, на этой неделе не зафиксировано.</p>")
    pos = aspects_week[aspects_week["reviews_with_aspect"]>=2].sort_values("positive_impact_index", ascending=False).head(5)
    neg = aspects_week[aspects_week["reviews_with_aspect"]>=2].sort_values("negative_impact_index", ascending=False).head(5)

    def _bullets(df, sign: str) -> str:
        if df is None or len(df) == 0:
            return ""
        items = []
        for _, r in df.iterrows():
            name = r["display_short"] or r["aspect_code"]
            cnt = int(r["reviews_with_aspect"])
            if sign == "pos":
                txt = f"<li><b>{name}</b> — {cnt} упомин.; удерживает высокие оценки (индекс {r['positive_impact_index']:.2f}).</li>"
            else:
                txt = f"<li><b>{name}</b> — {cnt} упомин.; тянет оценки вниз (индекс {r['negative_impact_index']:.2f}).</li>"
            items.append(txt)
        return "<ul>" + "".join(items) + "</ul>"

    pos_html = _bullets(pos, "pos") or "<p>На этой неделе не выделяется единого фактора, который тянет оценку вверх.</p>"
    neg_html = _bullets(neg, "neg") or "<p>Системных жалоб, которые тянут оценки вниз, на этой неделе не зафиксировано.</p>"
    return pos_html, neg_html

def _section_B0_dynamics(week_df: pd.DataFrame, df_hist_all: pd.DataFrame, anchor_week_key: str) -> str:
    """
    Сравнение с предыдущими 4 неделями.
    """
    if df_hist_all is None or len(df_hist_all)==0 or week_df is None:
        return ""
    # определим 4 предыдущие недели по ключу
    this_year, this_w = anchor_week_key.split("-W")
    this_year = int(this_year); this_w = int(this_w)
    # соберём ключи W-1..W-4 (в рамках одного года этого достаточно для текста; robust-вариант можно расширить)
    prev_keys = [f"{this_year}-W{w:02d}" for w in range(this_w-4, this_w) if w>0]
    prev_df = df_hist_all[df_hist_all["week_key"].isin(prev_keys)].copy()
    if prev_df.empty:
        return "<p>Средняя оценка текущей недели — недостаточно данных для сравнения с предыдущими неделями.</p>"
    def _avg(df):
        return float(df["rating10"].mean()) if len(df)>0 else float("nan")
    def _share(df, cond):
        if len(df)==0: return float("nan")
        return float(cond(df).mean())
    a_cur = _avg(week_df)
    a_prev = _avg(prev_df)
    p_cur = _share(week_df, lambda d: ( (d["sentiment_overall"]=="positive") | (d["rating10"]>=9) ))
    p_prev= _share(prev_df, lambda d: ( (d["sentiment_overall"]=="positive") | (d["rating10"]>=9) ))
    n_cur = _share(week_df, lambda d: ( (d["sentiment_overall"]=="negative") | (d["rating10"]<=6) ))
    n_prev= _share(prev_df, lambda d: ( (d["sentiment_overall"]=="negative") | (d["rating10"]<=6) ))
    delta = a_cur - a_prev if (a_cur==a_cur and a_prev==a_prev) else float("nan")
    if delta==delta and abs(delta) >= 0.05:
        trend = "выше" if delta>0 else "ниже"
        return (f"<p>Средняя оценка текущей недели — {a_cur:.2f}/10, что {trend} среднего предыдущих четырёх недель "
                f"({a_prev:.2f}, Δ {delta:+.2f}). Доля позитивных — {_fmt_pct(p_cur)} (было {_fmt_pct(p_prev)}), "
                f"доля негативных — {_fmt_pct(n_cur)} (было {_fmt_pct(n_prev)}).</p>")
    else:
        return (f"<p>Средняя оценка текущей недели — {'' if a_cur!=a_cur else f'{a_cur:.2f}/10'}. "
                f"Показатели позитивных/негативных отзывов близки к уровню последних четырёх недель.</p>")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main() -> None:
    # --- ENV ---
    drive_folder_id = os.environ.get("DRIVE_FOLDER_ID") or ""
    sheets_id = os.environ.get("SHEETS_HISTORY_ID") or ""
    recipients_env = os.environ.get("RECIPIENTS") or ""
    smtp_from = os.environ.get("SMTP_FROM") or ""
    smtp_user = os.environ.get("SMTP_USER") or ""
    smtp_pass = os.environ.get("SMTP_PASS") or ""
    week_key_env = os.environ.get("WEEK_KEY") or ""
    dry_run = (os.environ.get("DRY_RUN") or "false").strip().lower() == "true"

    if not (os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON_B64") or "").strip():
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON_B64 не задан.")
    if not drive_folder_id:
        raise RuntimeError("DRIVE_FOLDER_ID не задан.")
    if not sheets_id:
        raise RuntimeError("SHEETS_HISTORY_ID не задан.")

    recipients = [r.strip() for r in recipients_env.split(",") if r.strip()]

    # --- Период ---
    anchor_week_key = _last_completed_week_key(week_key_env if week_key_env else None)
    wk_monday = iso_week_monday(anchor_week_key)
    ranges = period_ranges_for_week(wk_monday)
    week_start, week_end = ranges["week"]["start"], ranges["week"]["end"]
    LOG.info(f"Anchor week: {anchor_week_key} ({week_start}..{week_end})")

    # --- Google clients ---
    creds = _build_credentials_from_b64()
    drive = _build_drive(creds)
    sheets = _build_sheets(creds)

    # --- Файл из Drive ---
    all_files = _drive_list_files_in_folder(drive, drive_folder_id)
    best = _pick_best_reviews_file(all_files, week_end)
    LOG.info(f"Выбран файл: {best.get('name')}  (id={best.get('id')})")
    blob = _drive_download_file_bytes(drive, best["id"])

    # --- Парсинг XLS ---
    df_raw = reviews_io.read_reviews_xls(blob)
    # сохраним has_response для истории
    df_raw_map = pd.DataFrame({
        "review_id": [], "has_response": []
    })
    try:
        # превратим в inputs (даёт review_id)
        inputs = reviews_io.df_to_inputs(df_raw)
        # соотнесём has_response по review_id
        tmp = []
        for r in inputs:
            has_resp = df_raw.loc[df_raw["text"].astype(str).str.strip() == str(r.text).strip(), "has_response"]
            tmp.append((r.review_id, (None if has_resp.empty else has_resp.iloc[0])))
        df_raw_map = pd.DataFrame(tmp, columns=["review_id","has_response"])
    except Exception as e:
        LOG.warning(f"Не удалось собрать map has_response: {e}")

    # --- Анализ ---
    # В реальном запуске сюда передаётся ваш Lexicon() из lexicon_module.
    # Чтобы каркас не падал на пустом лексиконе, поддержим lazy import.
    try:
        from agent.lexicon_module import Lexicon
    except ModuleNotFoundError:
        from lexicon_module import Lexicon  # type: ignore

    lexicon = Lexicon()  # ВАЖНО: предполагается, что в модуле реализованы compiled_topics/topic_schema и т.д.
    analyzed = reviews_core.analyze_reviews_bulk(inputs, lexicon)

    df_reviews = reviews_core.build_reviews_dataframe(analyzed)
    df_aspects = reviews_core.build_aspects_dataframe(analyzed)

        # --- История из Google Sheets + объединение с текущей неделей ---
    hist_df_raw = _read_sheet_as_df(sheets, sheets_id, HISTORY_SHEET_NAME)
    df_hist = _parse_history_df(hist_df_raw)

    # объединяем: history ∪ текущая неделя (без дублей по review_id)
    if not df_reviews.empty:
        cur = df_reviews.copy()
        cur["created_at"] = pd.to_datetime(cur["created_at"])
        # у нас review_id — уже стабильный ключ
        if not df_hist.empty:
            known = set(df_hist["review_id"].astype(str).tolist())
            cur = cur[~cur["review_id"].astype(str).isin(known)].copy()
            df_hist_all = pd.concat([df_hist, cur], ignore_index=True)
        else:
            df_hist_all = cur
    else:
        df_hist_all = df_hist.copy()

    # срезы периодов относительно anchor_week_key
    periods = reviews_core.slice_periods(df_hist_all, anchor_week_key)
    week_df = periods["week"]
    mtd_df  = periods["mtd"]
    qtd_df  = periods["qtd"]
    ytd_df  = periods["ytd"]
    all_df  = periods["all"]

    # базовая сводка по источникам по всем периодам (ядро блока C1)
    src_week = reviews_core.build_source_pivot(week_df)
    src_mtd  = reviews_core.build_source_pivot(mtd_df)
    src_qtd  = reviews_core.build_source_pivot(qtd_df)
    src_ytd  = reviews_core.build_source_pivot(ytd_df)
    src_all  = reviews_core.build_source_pivot(all_df)

    # impact по аспектам для недели/месяца/квартала/года/истории
    aspects_week = reviews_core.compute_aspect_impacts(
        df_reviews_period=week_df,
        df_aspects_period=df_aspects[df_aspects["week_key"] == anchor_week_key] if not df_aspects.empty else df_aspects
    )
    # Для остальных периодов у нас нет разметки аспектов в истории; пока используем week-грануляцию (позже добавим накопление)

    # --- Сводка по источникам за неделю (ядро блока C1) ---
    df_week = df_reviews.loc[(df_reviews["created_at"] >= pd.to_datetime(week_start)) &
                             (df_reviews["created_at"] <= pd.to_datetime(week_end))].copy()
    try:
        # если добавите в reviews_core build_source_pivot — можно заменить этим вызовом
        # df_sources = reviews_core.build_source_pivot(df_week)
        # временный вариант здесь, чтобы каркас работал до внедрения функции в ядро:
        lab = df_week.apply(lambda r: ("positive" if ((r.get("sentiment_overall") == "positive") or ((r.get("rating10") or 0) >= 9))
                                       else "negative" if ((r.get("sentiment_overall") == "negative") or ((r.get("rating10") or 11) <= 6))
                                       else "neutral"), axis=1)
        df_week = df_week.assign(__label__=lab)
        g = df_week.groupby("source", dropna=False)
        df_sources = g.agg(
            reviews=("review_id", "nunique"),
            avg10=("rating10", "mean"),
            pos_cnt=("__label__", lambda s: (s == "positive").sum()),
            neg_cnt=("__label__", lambda s: (s == "negative").sum()),
        ).reset_index()
        df_sources["pos_pct"] = (df_sources["pos_cnt"] / df_sources["reviews"]).fillna(0.0)
        df_sources["neg_pct"] = (df_sources["neg_cnt"] / df_sources["reviews"]).fillna(0.0)
        df_sources["avg10"] = df_sources["avg10"].round(2)
        df_sources = df_sources[["source","reviews","avg10","pos_pct","neg_pct","pos_cnt","neg_cnt"]].sort_values("reviews", ascending=False).reset_index(drop=True)
    except Exception as e:
        LOG.warning(f"Ошибка сводки по источникам: {e}")
        df_sources = pd.DataFrame(columns=["source","reviews","avg10","pos_pct","neg_pct","pos_cnt","neg_cnt"])

    # --- История в Google Sheets (идемпотентно) ---
    appended = _upsert_reviews_history_week(
        sheets=sheets,
        spreadsheet_id=sheets_id,
        df_reviews=df_reviews,
        df_raw_with_has_response=df_raw_map,
        week_key=anchor_week_key,
    )
    LOG.info(f"В историю добавлено строк: {appended}")

    # --- E-mail (A–C) ---
    subject = f"ARTSTUDIO | Отчёт по отзывам — неделя {week_start.strftime('%d %b')}–{week_end.strftime('%d %b %Y')}"

    a_block = _section_A_summary(week_df, mtd_df, qtd_df, ytd_df, all_df)
    b0_line = _section_B0_dynamics(week_df, df_hist_all, anchor_week_key)
    b1_html, b2_html = _section_B_drivers_and_risks(aspects_week)

    sources_html = _render_sources_block_html({
        "week": src_week, "mtd": src_mtd, "qtd": src_qtd, "ytd": src_ytd, "all": src_all
    })

    html = f"""
    <html><body>
      <h3>Итоги недели {week_start:%d %b} — {week_end:%d %b %Y}</h3>

      <h4>A. Итоги периода</h4>
      {a_block}

      <h4>B0. Краткая динамическая сводка</h4>
      {b0_line}

      <h4>B1. Что создаёт высокий балл сейчас (драйверы)</h4>
      {b1_html}

      <h4>B2. Что тянет оценки вниз (зоны риска)</h4>
      {b2_html}

      <h4>C1. Источники × Период</h4>
      {sources_html}
      <p style="color:#666;margin-top:8px">
        Примечание: для TL: Marketing, Trip.com, Яндекс, 2GIS, Google, TripAdvisor в колонке «Нативная» отображается 5-балльная шкала; все расчёты ведутся в 10-балльной.
      </p>
    </body></html>
    """
dy></html>
    """

    attachments: List[Tuple[str, bytes]] = []
    # CSV с обзорной таблицей по отзывам недели
    try:
        buf_reviews = io.StringIO()
        df_week.to_csv(buf_reviews, index=False, quoting=csv.QUOTE_MINIMAL)
        attachments.append((f"reviews_week_{anchor_week_key}.csv", buf_reviews.getvalue().encode("utf-8-sig")))
    except Exception as e:
        LOG.warning(f"Не удалось подготовить CSV с отзывами недели: {e}")

    # CSV по аспектам (вся неделя)
    try:
        df_aspects_week = df_aspects[df_aspects["week_key"] == anchor_week_key].copy()
        buf_aspects = io.StringIO()
        df_aspects_week.to_csv(buf_aspects, index=False, quoting=csv.QUOTE_MINIMAL)
        attachments.append((f"reviews_aspects_week_{anchor_week_key}.csv", buf_aspects.getvalue().encode("utf-8-sig")))
    except Exception as e:
        LOG.warning(f"Не удалось подготовить CSV по аспектам: {e}")

    if dry_run:
        LOG.info("DRY_RUN=true — письмо не отправляем.")
        return

    # SMTP-хост. По аналогии с анкетами предполагаем внешний хост (если другой — поправим в секрете/окружении)
    smtp_host = "smtp.yandex.com" if "yandex" in (smtp_user or "").lower() else "smtp.gmail.com"
    try:
        _send_email(
            smtp_host=smtp_host,
            smtp_user=smtp_user,
            smtp_pass=smtp_pass,
            smtp_from=smtp_from,
            recipients=recipients,
            subject=subject,
            html_body=html,
            attachments=attachments,
        )
        LOG.info(f"Письмо отправлено на: {', '.join(recipients) if recipients else '(нет адресатов)'}")
    except Exception as e:
        LOG.error(f"Ошибка отправки письма: {e}")


if __name__ == "__main__":
    main()
