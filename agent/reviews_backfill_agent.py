# agent/reviews_backfill_agent.py
from __future__ import annotations

import os
import io
import re
import base64
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import date, datetime, timedelta

import pandas as pd

# --- наши модули (пакетные импорты) ---
from . import reviews_io, reviews_core
from .metrics_core import iso_week_monday, period_ranges_for_week

# --- Google API ---
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# -----------------------------------------------------------------------------
# Логгер
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("reviews_backfill_agent")

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
SHEETS_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

HISTORY_SHEET_NAME = "reviews_history"  # отдельная вкладка в общем SHEETS_HISTORY_ID

# Поддерживаемые паттерны имён файлов (даты в именах)
_RE_FNAME_DMY = re.compile(r"(?i)\breviews?_?(\d{2})-(\d{2})-(\d{4})\b")
_RE_FNAME_YMD = re.compile(r"(?i)\breviews?_?(\d{4})-(\d{2})-(\d{2})\b")
# Годовой агрегат: reviews_2019-25.xls (с 2019 по 2025)
_RE_FNAME_YEAR_RANGE = re.compile(r"(?i)\breviews?_?(\d{4})-(\d{2})\b")


# -----------------------------------------------------------------------------
# Утилиты
# -----------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _parse_date_from_name(name: str) -> Optional[date]:
    """
    Reviews_DD-MM-YYYY.xls  или  reviews_YYYY-MM-DD.xls
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

def _date_from_env(var_name: str) -> date:
    val = (os.environ.get(var_name) or "").strip()
    if not val:
        raise RuntimeError(f"{var_name} не задан (ожидается YYYY-MM-DD).")
    try:
        return datetime.strptime(val, "%Y-%m-%d").date()
    except Exception:
        raise RuntimeError(f"{var_name} имеет неверный формат: {val} (ожидается YYYY-MM-DD).")

def _b64_to_sa_json_path(b64_env: str) -> str:
    """
    Декодирует GOOGLE_SERVICE_ACCOUNT_JSON_B64 в временный файл и возвращает путь.
    """
    content_b64 = os.environ.get(b64_env) or ""
    if not content_b64:
        raise RuntimeError(f"{b64_env} не задан.")
    raw = base64.b64decode(content_b64)
    out_path = "/tmp/sa.json"
    with open(out_path, "wb") as f:
        f.write(raw)
    return out_path

def _build_credentials_from_b64() -> "service_account.Credentials":
    sa_path = _b64_to_sa_json_path("GOOGLE_SERVICE_ACCOUNT_JSON_B64")
    return service_account.Credentials.from_service_account_file(
        sa_path, scopes=DRIVE_SCOPES + SHEETS_SCOPES
    )

def _build_drive(creds):
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def _build_sheets(creds):
    return build("sheets", "v4", credentials=creds, cache_discovery=False)

def _drive_list_files_in_folder(drive, folder_id: str) -> List[Dict[str, Any]]:
    q = f"'{folder_id}' in parents and trashed = false"
    fields = "nextPageToken, files(id, name, mimeType, modifiedTime, size)"
    files: List[Dict[str, Any]] = []
    page_token = None
    while True:
        resp = drive.files().list(
            q=q, fields=fields, pageToken=page_token, pageSize=1000, orderBy="modifiedTime desc"
        ).execute()
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

def _ensure_sheet_exists(sheets, spreadsheet_id: str, title: str) -> None:
    meta = sheets.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    for sh in meta.get("sheets", []):
        if sh.get("properties", {}).get("title") == title:
            return
    body = {"requests": [{"addSheet": {"properties": {"title": title}}}]}
    sheets.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body=body).execute()

def _read_sheet_as_df(sheets, spreadsheet_id: str, title: str) -> pd.DataFrame:
    try:
        resp = sheets.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id, range=f"'{title}'!A:Z"
        ).execute()
        values = resp.get("values", [])
        if not values:
            return pd.DataFrame()
        header = values[0]
        rows = values[1:]
        return pd.DataFrame(rows, columns=header)
    except Exception:
        return pd.DataFrame()

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

def _trim_text(s: str, n: int = 280) -> str:
    s = (s or "").strip()
    return (s[: n - 1] + "…") if len(s) > n else s

def _serialize_aspects_for_sheet(value: Any) -> str:
    """
    Превращает список аспектов (['wifi', 'noise', ...]) в строку "wifi;noise".
    Если уже строка — возвращаем как есть.
    """
    if value is None:
        return ""
    # если уже str — не трогаем
    if isinstance(value, str):
        return value.strip()
    # list / tuple / set со строками
    if isinstance(value, (list, tuple, set)):
        parts = []
        for v in value:
            if v is None:
                continue
            parts.append(str(v).strip())
        return ";".join(p for p in parts if p)
    # на всякий случай fallback
    return str(value).strip()

def _serialize_topics_for_sheet(value: Any) -> str:
    """
    Превращает список пар (topic, subtopic) в строку:
      [("breakfast","service_dining_staff"), ("checkin_stay","stay_support")]
      -> "breakfast:service_dining_staff;checkin_stay:stay_support"
    Если уже строка — возвращаем как есть.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple, set)):
        parts = []
        for v in value:
            # ожидаем кортеж/список длины >=2
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                topic_key = str(v[0]).strip()
                sub_key = str(v[1]).strip()
                if topic_key or sub_key:
                    parts.append(f"{topic_key}:{sub_key}")
            else:
                # если вдруг прилетел одиночный код
                parts.append(str(v).strip())
        return ";".join(p for p in parts if p)
    return str(value).strip()

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

    existing_keys: set = set()
    if not df_sheet.empty and "iso_week" in df_sheet.columns and "review_key" in df_sheet.columns:
        existing_keys = set(
            df_sheet.loc[df_sheet["iso_week"] == week_key, "review_key"].astype(str).tolist()
        )

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
        review_key = review_id  # наш review_id уже стабилен и уникален
        if review_key in existing_keys:
            continue

        aspects = row.get("aspects") or ""
        topics  = row.get("topics") or ""
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


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main() -> None:
    # --- ENV ---
    drive_folder_id = os.environ.get("DRIVE_FOLDER_ID") or ""
    sheets_id = os.environ.get("SHEETS_HISTORY_ID") or ""
    dry_run = (os.environ.get("DRY_RUN") or "false").strip().lower() == "true"

    if not drive_folder_id:
        raise RuntimeError("DRIVE_FOLDER_ID не задан.")
    if not sheets_id:
        raise RuntimeError("SHEETS_HISTORY_ID не задан.")

    # --- Google clients (через B64 секрет) ---
    creds = _build_credentials_from_b64()
    drive = _build_drive(creds)
    sheets = _build_sheets(creds)

    # --- Выбор файла: BACKFILL_FILE -> reviews_YYYY-YY.* -> ошибка ---
    backfill_file = (os.environ.get("BACKFILL_FILE") or "").strip()
    files = _drive_list_files_in_folder(drive, drive_folder_id)

    selected: List[Dict[str, Any]] = []
    if backfill_file:
        for f in files:
            name = f.get("name", "")
            if f["id"] == backfill_file or name.lower() == backfill_file.lower():
                selected = [f]
                LOG.info(f"BACKFILL_FILE: выбран файл '{name}' (id={f['id']}).")
                break
        if not selected:
            raise RuntimeError(f"BACKFILL_FILE='{backfill_file}' не найден в папке.")
    else:
        yr_file = None
        for f in files:
            name = f.get("name", "")
            if _RE_FNAME_YEAR_RANGE.search(name):
                yr_file = f
                break  # берём самый свежий
        if yr_file:
            selected = [yr_file]
            LOG.info(f"Обнаружен агрегированный файл: {yr_file.get('name')}")
        else:
            raise RuntimeError(
                "Не найден агрегированный файл reviews_YYYY-YY.* и не задан BACKFILL_FILE."
            )

    if not selected:
        LOG.warning("Не найдено входных файлов.")
        return

    LOG.info(f"Файлов к обработке: {len(selected)}")

    # --- Чтение/парсинг выбранных файлов ---
    all_inputs: List[reviews_core.ReviewRecordInput] = []
    raw_has_response_pairs: List[Tuple[str, Any]] = []

    for f in selected:
        fid = f["id"]
        fname = f.get("name", "")
        try:
            blob = _drive_download_file_bytes(drive, fid)
            df_raw = reviews_io.read_reviews_xls(blob)

            LOG.info(f"OK: {fname} (raw rows: {len(df_raw)})")
            try:
                LOG.info(f"Колонки: {list(df_raw.columns)}")
                if "date" in df_raw.columns:
                    LOG.info(f"Пример дат: {df_raw['date'].astype(str).head(5).tolist()}")
            except Exception:
                pass

            inputs = reviews_io.df_to_inputs(df_raw)
            LOG.info(f"→ inputs: {len(inputs)}")

            if not inputs:
                LOG.warning(f"Файл {fname}: после нормализации записей нет.")
                continue

            all_inputs.extend(inputs)

            # map review_id -> has_response (по тексту как стабильному ключу в рамках файла)
            tmp: List[Tuple[str, Any]] = []
            for r in inputs:
                mask = df_raw["text"].astype(str).str.strip() == str(r.text).strip()
                has_resp = df_raw.loc[mask, "has_response"]
                tmp.append((r.review_id, (None if has_resp.empty else has_resp.iloc[0])))
            raw_has_response_pairs.extend(tmp)

        except Exception as e:
            LOG.error(f"Ошибка чтения {fname}: {e}")

    if not all_inputs:
        LOG.warning("Нет валидных записей отзывов (inputs пуст). Проверяй парсинг даты/колонки.")
        return

    # --- Анализ через лексикон ---
    from .lexicon_module import Lexicon
    lexicon = Lexicon()

    analyzed = reviews_core.analyze_reviews_bulk(all_inputs, lexicon)
    LOG.info(f"Анализировано записей: {len(analyzed)}")
    if not analyzed:
        LOG.warning("После анализа записей нет (analyzed=0).")
        return

    df_reviews = reviews_core.build_reviews_dataframe(analyzed)
    df_raw_map = (
        pd.DataFrame(raw_has_response_pairs, columns=["review_id", "has_response"])
        .drop_duplicates("review_id")
    )

    # --- Берём ВСЕ записи файла (без фильтров по датам) ---
    df_reviews_period = df_reviews.copy()
    if df_reviews_period.empty:
        LOG.warning("После отбора записей — пусто (ничего писать в историю).")
        return

    LOG.info(
        f"Готовим к записи строк: {len(df_reviews_period)}; "
        f"диапазон дат: {df_reviews_period['created_at'].min()} — {df_reviews_period['created_at'].max()}"
    )

    weeks = sorted(set(df_reviews_period["week_key"].tolist()))
    LOG.info(f"Недели к upsert: {', '.join(weeks) if weeks else '(нет)'}")

    total_appended = 0
    if dry_run:
        LOG.info("DRY_RUN=true — запись в Google Sheets не выполняется.")
    else:
        for wk in weeks:
            appended = _upsert_reviews_history_week(
                sheets=sheets,
                spreadsheet_id=sheets_id,
                df_reviews=df_reviews_period,
                df_raw_with_has_response=df_raw_map,
                week_key=wk,
            )
            LOG.info(f"Неделя {wk}: добавлено строк {appended}")
            total_appended += appended

    LOG.info(f"Готово. Всего добавлено: {total_appended}")


if __name__ == "__main__":
    main()
