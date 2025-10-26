# agent/surveys_weekly_report_agent.py
#
# Еженедельный отчёт по анкетам TL: Marketing
#
# Чистая версия без avg10:
# - формат листа surveys_history теперь:
#   week_key, param, surveys_total, answered, avg5,
#   promoters, detractors, nps_answers, nps_value
#
# - upsert_week() пишет/обновляет в этом формате
# - surveys_aggregate_period() и письмо работают только с /5 и NPS
#
# Предпосылка: лист surveys_history в Google Sheets ты очистил до шапки
# и готов принять новые данные.

import os, io, re, sys, json, math, datetime as dt
from datetime import date
import pandas as pd
import numpy as np

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import smtplib, mimetypes

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Headless charts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- ядро анкет (нормализация и недельная агрегация) ---
try:
    from agent.surveys_core import (
        parse_and_aggregate_weekly,
        SURVEYS_TAB,
        PARAM_ORDER,
    )
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(__file__))
    from surveys_core import (
        parse_and_aggregate_weekly,
        SURVEYS_TAB,
        PARAM_ORDER,
    )

# --- утилиты периодов/дат/подписей ---
try:
    from agent.metrics_core import (
        iso_week_monday,
        period_ranges_for_week,
        week_label,
        month_label,
        quarter_label,
        year_label,
    )
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(__file__))
    from metrics_core import (
        iso_week_monday,
        period_ranges_for_week,
        week_label,
        month_label,
        quarter_label,
        year_label,
    )

# =========================
# ENV & Google API clients
# =========================
SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/spreadsheets",
]

SA_PATH    = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
SA_CONTENT = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON_CONTENT")

if SA_CONTENT and SA_CONTENT.strip().startswith("{"):
    CREDS = Credentials.from_service_account_info(json.loads(SA_CONTENT), scopes=SCOPES)
else:
    if not SA_PATH:
        raise RuntimeError("No GOOGLE_SERVICE_ACCOUNT_JSON or GOOGLE_SERVICE_ACCOUNT_JSON_CONTENT provided.")
    CREDS = Credentials.from_service_account_file(SA_PATH, scopes=SCOPES)

DRIVE  = build("drive",  "v3", credentials=CREDS)
SHEETS = build("sheets", "v4", credentials=CREDS).spreadsheets()

DRIVE_FOLDER_ID   = os.environ["DRIVE_FOLDER_ID"]
HISTORY_SHEET_ID  = os.environ["SHEETS_HISTORY_ID"]

RECIPIENTS        = [e.strip() for e in os.environ.get("RECIPIENTS","").split(",") if e.strip()]
SMTP_USER         = os.environ.get("SMTP_USER")
SMTP_PASS         = os.environ.get("SMTP_PASS")
SMTP_FROM         = os.environ.get("SMTP_FROM", SMTP_USER or "")
SMTP_HOST         = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT         = int(os.environ.get("SMTP_PORT", 587))

# =========================
# Drive helpers
# =========================
WEEKLY_RE = re.compile(r"^Report_(\d{2})-(\d{2})-(\d{4})\.xlsx$", re.IGNORECASE)

def latest_report_from_drive():
    """
    Берём самый свежий файл вида Report_DD-MM-YYYY.xlsx из папки DRIVE_FOLDER_ID.
    """
    res = DRIVE.files().list(
        q=f"'{DRIVE_FOLDER_ID}' in parents and mimeType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' and trashed=false",
        fields="files(id,name,modifiedTime)",
    ).execute()
    items = []
    for f in res.get("files", []):
        m = WEEKLY_RE.match(f["name"])
        if not m:
            continue
        dd, mm, yyyy = m.groups()
        try:
            d = dt.date(int(yyyy), int(mm), int(dd))
        except Exception:
            d = dt.date.min
        items.append((f["id"], f["name"], d))
    if not items:
        raise RuntimeError("В папке нет файлов вида Report_dd-mm-yyyy.xlsx.")
    items.sort(key=lambda t: t[2], reverse=True)
    return items[0]  # (file_id, filename, date_obj)

def drive_download(file_id: str) -> bytes:
    req = DRIVE.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
    return buf.getvalue()

# =========================
# Sheets helpers
# =========================
def ensure_tab(spreadsheet_id: str, tab_name: str, header: list[str]):
    """
    Гарантируем, что лист существует и в первой строке правильный заголовок.
    Если лист создаётся заново, сразу пишем шапку.
    """
    meta = SHEETS.get(spreadsheetId=spreadsheet_id).execute()
    tabs = [s["properties"]["title"] for s in meta.get("sheets", [])]
    if tab_name not in tabs:
        SHEETS.batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"requests":[{"addSheet":{"properties":{"title":tab_name}}}]}
        ).execute()
        SHEETS.values().update(
            spreadsheetId=spreadsheet_id,
            range=f"{tab_name}!A1:{chr(64+len(header))}1",
            valueInputOption="RAW",
            body={"values":[header]},
        ).execute()
    else:
        # Если лист уже есть, мы не трогаем шапку автоматически.
        # Но если хочешь, можешь раскомментить этот блок, чтобы на каждом запуске
        # гарантированно восстанавливать шапку в A1:I1.
        #
        # SHEETS.values().update(
        #     spreadsheetId=spreadsheet_id,
        #     range=f"{tab_name}!A1:{chr(64+len(header))}1",
        #     valueInputOption="RAW",
        #     body={"values":[header]},
        # ).execute()
        pass

def gs_get_df(tab: str, a1: str) -> pd.DataFrame:
    """
    Считать диапазон из Google Sheets в pandas.DataFrame.
    Возвращает пустой df, если нет данных под шапкой.
    """
    try:
        res = SHEETS.values().get(
            spreadsheetId=HISTORY_SHEET_ID,
            range=f"{tab}!{a1}"
        ).execute()
        vals = res.get("values", [])
        return pd.DataFrame(vals[1:], columns=vals[0]) if len(vals) > 1 else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# Новая целевая шапка surveys_history — без avg10
SURVEYS_HEADER = [
    "week_key",
    "param",
    "surveys_total",
    "answered",
    "avg5",
    "promoters",
    "detractors",
    "nps_answers",
    "nps_value",
]

def rows_from_agg(df: pd.DataFrame) -> list[list]:
    """
    Преобразуем недельную агрегацию (agg_week из surveys_core) к строкам для шита.
    df ожидается с колонками:
      week_key, param, surveys_total, answered, avg5,
      promoters, detractors, nps_answers, nps_value
    """
    out = []
    for _, r in df.iterrows():
        out.append([
            str(r["week_key"]),
            str(r["param"]),
            (int(r["surveys_total"]) if not pd.isna(r["surveys_total"]) else 0),
            (int(r["answered"])      if not pd.isna(r["answered"])      else 0),
            (None if pd.isna(r["avg5"])          else float(r["avg5"])),
            (None if "promoters"    not in r or pd.isna(r["promoters"])    else int(r["promoters"])),
            (None if "detractors"   not in r or pd.isna(r["detractors"])   else int(r["detractors"])),
            (None if "nps_answers"  not in r or pd.isna(r["nps_answers"])  else int(r["nps_answers"])),
            (None if "nps_value"    not in r or pd.isna(r["nps_value"])    else float(r["nps_value"])),
        ])
    return out

def upsert_week(agg_week_df: pd.DataFrame) -> int:
    """
    Обновляем в шите только одну неделю:
    - читаем текущую историю
    - выкидываем строки с той же week_key
    - чистим всё тело листа (кроме заголовка)
    - пишем обратно старые недели + новую неделю
    Формат без avg10 (A:I).
    """
    if agg_week_df.empty:
        return 0

    ensure_tab(HISTORY_SHEET_ID, SURVEYS_TAB, SURVEYS_HEADER)

    wk = str(agg_week_df["week_key"].iloc[0])

    hist = gs_get_df(SURVEYS_TAB, "A:I")
    keep = (
        hist[hist.get("week_key", "") != wk]
        if not hist.empty
        else pd.DataFrame(columns=SURVEYS_HEADER)
    )

    # подчистить старые значения под шапкой
    SHEETS.values().clear(
        spreadsheetId=HISTORY_SHEET_ID,
        range=f"{SURVEYS_TAB}!A2:I",
    ).execute()

    rows_keep = keep[SURVEYS_HEADER].values.tolist() if not keep.empty else []
    rows_new  = rows_from_agg(agg_week_df)
    rows_all  = rows_keep + rows_new

    if rows_all:
        SHEETS.values().append(
            spreadsheetId=HISTORY_SHEET_ID,
            range=f"{SURVEYS_TAB}!A2",
            valueInputOption="RAW",
            body={"values": rows_all},
        ).execute()
    return len(rows_new)

# =========================
# Aggregation over periods
# =========================

def _to_num_series(s: pd.Series) -> pd.Series:
    """
    Приводим данные из Sheets к числам:
    - '4,71' → 4.71
    - '5'    → 5.0
    - всё некорректное → NaN
    """
    if s is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(
        s.astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )

def _weighted_avg(values: pd.Series, weights: pd.Series) -> float | None:
    """
    Взвешенная средняя по /5:
    - values  = средние /5 по неделям (как строки из Sheets)
    - weights = answered по неделям (как строки из Sheets)
    """
    v = _to_num_series(values)
    w = _to_num_series(weights)

    m = (~v.isna()) & (~w.isna()) & (w > 0)
    if not m.any():
        return None

    s = (v[m] * w[m]).sum()
    W = w[m].sum()
    if W <= 0:
        return None

    return round(float(s / W), 2)

def surveys_aggregate_period(history_df: pd.DataFrame, start: date, end: date) -> dict:
    """
    Считаем агрегат за произвольный период (неделя / месяц / квартал / год / итого).
    Возвращаем:
    {
      "by_param": DataFrame с колонками
          param, surveys_total, answered, avg5,
          promoters, detractors, nps_answers, nps_value
      "totals": {
          "surveys_total": int,
          "overall5": float | None,
          "nps": float | None
      }
    }

    Логика:
    - средняя avg5 взвешивается по answered;
    - NPS считается из промоутеров и детракторов.
    """

    empty_bp_cols = [
        "param",
        "surveys_total",
        "answered",
        "avg5",
        "promoters",
        "detractors",
        "nps_answers",
        "nps_value",
    ]
    if history_df is None or history_df.empty:
        return {
            "by_param": pd.DataFrame(columns=empty_bp_cols),
            "totals": {"surveys_total": 0, "overall5": None, "nps": None},
        }

    df = history_df.copy()

    # приведение всех числовых столбцов
    numeric_cols = [
        "surveys_total",
        "answered",
        "avg5",
        "promoters",
        "detractors",
        "nps_answers",
        "nps_value",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = _to_num_series(df[c])

    # добавляем понедельник ISO-недели, чтобы фильтровать по диапазону
    df["mon"] = df["week_key"].map(lambda k: iso_week_monday(str(k)))

    df = df[(df["mon"] >= start) & (df["mon"] <= end)].copy()
    if df.empty:
        return {
            "by_param": pd.DataFrame(columns=empty_bp_cols),
            "totals": {"surveys_total": 0, "overall5": None, "nps": None},
        }

    rows = []
    for param, g in df.groupby("param"):
        surveys_total_sum = int(_to_num_series(g["surveys_total"]).fillna(0).sum())
        answered_sum      = int(_to_num_series(g["answered"]).fillna(0).sum())

        avg5_weighted  = _weighted_avg(g["avg5"], g["answered"])

        promoters_sum   = None
        detractors_sum  = None
        nps_answers_sum = None
        nps_val         = None
        if param == "nps":
            promoters_sum   = int(_to_num_series(g.get("promoters",   pd.Series())).fillna(0).sum())
            detractors_sum  = int(_to_num_series(g.get("detractors",  pd.Series())).fillna(0).sum())
            nps_answers_sum = int(_to_num_series(g.get("nps_answers", pd.Series())).fillna(0).sum())

            if nps_answers_sum > 0:
                nps_val = round(
                    float(
                        (promoters_sum / nps_answers_sum - detractors_sum / nps_answers_sum)
                        * 100.0
                    ),
                    2,
                )

        rows.append({
            "param":         param,
            "surveys_total": surveys_total_sum,
            "answered":      answered_sum,
            "avg5":          avg5_weighted,
            "promoters":     promoters_sum,
            "detractors":    detractors_sum,
            "nps_answers":   nps_answers_sum,
            "nps_value":     nps_val,
        })

    by_param = pd.DataFrame(rows)

    overall_row = by_param[by_param["param"] == "overall"]
    nps_row     = by_param[by_param["param"] == "nps"]

    totals = {
        "surveys_total": int(overall_row["surveys_total"].iloc[0]) if not overall_row.empty else 0,
        "overall5":      None if overall_row.empty else overall_row["avg5"].iloc[0],
        "nps":           None if nps_row.empty     else nps_row["nps_value"].iloc[0],
    }

    return {"by_param": by_param, "totals": totals}

# =========================
# Текст / HTML
# =========================
PARAM_TITLES = {
    "overall":        "Итоговая оценка",
    "spir_checkin":   "СПиР при заезде",
    "clean_checkin":  "Чистота при заезде",
    "comfort":        "Комфорт и оснащение",
    "spir_stay":      "СПиР в проживании",
    "tech_service":   "ИТС (техслужба)",
    "housekeeping":   "Уборка в проживании",
    "breakfast":      "Завтраки",
    "atmosphere":     "Атмосфера",
    "location":       "Расположение",
    "value":          "Цена/качество",
    "return_intent":  "Готовность вернуться",
    "nps":            "NPS (1–5)",
}

def fmt_avg5(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{float(x):.2f} /5"

def fmt_int(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return str(int(x))

def fmt_nps(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{float(x):.1f}"

def header_block(week_start: date, week_end: date, W: dict, M: dict, Q: dict, Y: dict, T: dict):
    wl = week_label(week_start, week_end)

    parts = []
    parts.append(f"<h2>ARTSTUDIO Nevsky — Анкеты за неделю {wl}</h2>")

    parts.append(
        "<p><b>Неделя:</b> "
        f"{wl}; анкет: <b>{fmt_int(W['totals']['surveys_total'])}</b>; "
        f"итоговая: <b>{fmt_avg5(W['totals']['overall5'])}</b>; "
        f"NPS: <b>{fmt_nps(W['totals']['nps'])}</b>.</p>"
    )

    def one_line(name, D):
        return (
            f"<b>{name}:</b> анкет {fmt_int(D['totals']['surveys_total'])}, "
            f"итоговая {fmt_avg5(D['totals']['overall5'])}, "
            f"NPS {fmt_nps(D['totals']['nps'])}"
        )

    parts.append(
        "<p>"
        + one_line(f"Текущий месяц ({month_label(week_start)})", M) + ";<br>"
        + one_line(f"Текущий квартал ({quarter_label(week_start)})", Q) + ";<br>"
        + one_line(f"Текущий год ({year_label(week_start)})", Y) + ";<br>"
        + one_line("Итого (вся история)", T)
        + ".</p>"
    )

    return "\n".join(parts)

def table_params_block(W_df: pd.DataFrame, M_df: pd.DataFrame, Q_df: pd.DataFrame, Y_df: pd.DataFrame, T_df: pd.DataFrame):
    def df_to_map(df):
        mp = {}
        for _, r in df.iterrows():
            mp[str(r["param"])] = r.to_dict()
        return mp

    W = df_to_map(W_df)
    M = df_to_map(M_df)
    Q = df_to_map(Q_df)
    Y = df_to_map(Y_df)
    T = df_to_map(T_df)

    order = [
        "overall",
        "spir_checkin",
        "clean_checkin",
        "comfort",
        "spir_stay",
        "tech_service",
        "housekeeping",
        "breakfast",
        "atmosphere",
        "location",
        "value",
        "return_intent",
        "nps",
    ]

    def cell(mp: dict, param: str) -> str:
        r = mp.get(param)
        if r is None:
            return "<td>—</td><td>—</td>"
        if param == "nps":
            return (
                f"<td>{fmt_nps(r.get('nps_value'))}</td>"
                f"<td>{fmt_int(r.get('nps_answers'))}</td>"
            )
        return (
            f"<td>{fmt_avg5(r.get('avg5'))}</td>"
            f"<td>{fmt_int(r.get('answered'))}</td>"
        )

    rows_html = []
    for p in order:
        title = PARAM_TITLES.get(p, p)
        rows_html.append(
            "<tr><td><b>"
            + title
            + "</b></td>"
            + cell(W, p)
            + cell(M, p)
            + cell(Q, p)
            + cell(Y, p)
            + cell(T, p)
            + "</tr>"
        )

    html = f"""
    <h3>Параметры (Неделя / Текущий месяц / Текущий квартал / Текущий год / Итого)</h3>
    <table border='1' cellspacing='0' cellpadding='6'>
      <tr>
        <th rowspan="2">Параметр</th>
        <th colspan="2">Неделя</th>
        <th colspan="2">Месяц</th>
        <th colspan="2">Квартал</th>
        <th colspan="2">Год</th>
        <th colspan="2">Итого</th>
      </tr>
      <tr>
        <th>Ср.</th><th>Ответы</th>
        <th>Ср.</th><th>Ответы</th>
        <th>Ср.</th><th>Ответы</th>
        <th>Ср.</th><th>Ответы</th>
        <th>Ср.</th><th>Ответы</th>
      </tr>
      {''.join(rows_html)}
    </table>
    """
    return html

def footnote_block():
    return """
    <hr>
    <p><i>* Все значения показаны в шкале /5 (1 = плохо, 5 = отлично). Средняя по каждому параметру считается ТОЛЬКО среди гостей, которые ответили на этот вопрос; их количество показано как «Ответы».
    <br>
    NPS считается по вопросу «вероятность порекомендовать»: 1–2 — детракторы, 3–4 — нейтралы, 5 — промоутеры. Пустые ответы не учитываются. NPS = %промоутеров − %детракторов (в п.п.).
    <br>
    «Итого» — накопленные данные с начала сбора анкет.</i></p>
    """

def trends_block(W, Prev, L4):
    """
    Формируем аналитический блок "Динамика и точки внимания".

    Аргументы:
      W    - агрегат текущей недели (dict с ключами "totals", "by_param")
      Prev - агрегат прошлой недели
      L4   - агрегат по окну последних 4 недель ДО текущей
             (то есть baseline, без текущей недели)

    Возвращает:
      HTML-строку <h3>...</h3><p>...</p> готовую для встраивания в письмо.
    """

    # --- Вспомогательные форматтеры внутри блока ---

    def fmt_delta(curr, prev, unit):
        """
        Возвращает строку вида '+0.12 /5' или '-4.0 п.п.'.
        Если нет данных, вернёт '—'.
        unit: '/5' или 'п.п.'
        """
        if curr is None or prev is None:
            return "—"
        try:
            d = float(curr) - float(prev)
        except Exception:
            return "—"
        sign = "+" if d >= 0 else ""
        if unit == "/5":
            return f"{sign}{d:.2f} {unit}"
        else:
            return f"{sign}{d:.1f} {unit}"

    def df_to_map(df):
        """
        Превращает DataFrame с by_param в словарь:
        {
          "comfort": { "avg5": ..., "answered": ..., ... },
          ...
        }
        Удобно для сравнений по параметрам.
        """
        mp = {}
        if df is None or len(df) == 0:
            return mp
        for _, r in df.iterrows():
            mp[str(r["param"])] = r.to_dict()
        return mp

    # --- 1. Краткосрочная динамика (неделя vs прошлая неделя) ---

    cur_overall  = W["totals"]["overall5"]
    prev_overall = Prev["totals"]["overall5"]

    cur_nps   = W["totals"]["nps"]
    prev_nps  = Prev["totals"]["nps"]

    short_overall_line = (
        f"Итоговая оценка за неделю: {fmt_avg5(cur_overall)} "
        f"({fmt_delta(cur_overall, prev_overall, '/5')} к прошлой неделе "
        f"{fmt_avg5(prev_overall)})."
    )

    short_nps_line = (
        f"NPS: {fmt_nps(cur_nps)} "
        f"({fmt_delta(cur_nps, prev_nps, 'п.п.')} к прошлой неделе "
        f"{fmt_nps(prev_nps)})."
    )

    # --- 2. Долгосрочная динамика (неделя vs baseline последних 4 недель) ---

    base_overall = L4["totals"]["overall5"]
    base_nps     = L4["totals"]["nps"]

    long_overall_line = (
        "Итоговая оценка сейчас "
        f"{fmt_avg5(cur_overall)} "
        f"({fmt_delta(cur_overall, base_overall, '/5')} к среднему за последние 4 недели "
        f"{fmt_avg5(base_overall)})."
        if base_overall is not None
        else "Недостаточно данных для долгосрочного сравнения по итоговой оценке."
    )

    long_nps_line = (
        "NPS сейчас "
        f"{fmt_nps(cur_nps)} "
        f"({fmt_delta(cur_nps, base_nps, 'п.п.')} к среднему за последние 4 недели "
        f"{fmt_nps(base_nps)})."
        if base_nps is not None
        else "Недостаточно данных для долгосрочного сравнения по NPS."
    )

    # --- 3. Сильные стороны и зоны риска (по параметрам: Неделя vs baseline L4) ---

    Wmap  = df_to_map(W["by_param"])
    L4map = df_to_map(L4["by_param"])

    deltas = []
    for p, wrow in Wmap.items():
        if p == "nps":
            continue  # NPS не сравниваем как /5 параметр
        w_avg = wrow.get("avg5")
        l4_avg = L4map.get(p, {}).get("avg5")
        if (
            w_avg is None or l4_avg is None
            or (isinstance(w_avg, float) and np.isnan(w_avg))
            or (isinstance(l4_avg, float) and np.isnan(l4_avg))
        ):
            continue
        try:
            w_avg_f = float(w_avg)
            l4_avg_f = float(l4_avg)
        except Exception:
            continue
        delta = w_avg_f - l4_avg_f
        deltas.append({
            "param": p,
            "week_avg": w_avg_f,
            "base_avg": l4_avg_f,
            "delta": delta,
        })

    strengths_html = ""
    risks_html = ""

    if not deltas:
        strengths_html = (
            "<p><b>Сильные стороны недели:</b><br>"
            "Недостаточно данных для оценки параметров относительно последних 4 недель.</p>"
        )
        risks_html = (
            "<p><b>Зоны риска:</b><br>"
            "Недостаточно данных для оценки параметров относительно последних 4 недель.</p>"
        )
    else:
        IMPROVE_THRESHOLD = 0.20   # значимое улучшение
        DECLINE_THRESHOLD = -0.20  # значимое ухудшение

        improved = [d for d in deltas if d["delta"] >= IMPROVE_THRESHOLD]
        declined = [d for d in deltas if d["delta"] <= DECLINE_THRESHOLD]

        # самые сильные апгрейды вверху
        improved.sort(key=lambda x: x["delta"], reverse=True)
        # самые проблемные просадки вверху
        declined.sort(key=lambda x: x["delta"])

        def fmt_param_line(d):
            title = PARAM_TITLES.get(d["param"], d["param"])
            sign = "+" if d["delta"] >= 0 else ""
            return (
                f"- {title}: {d['week_avg']:.2f} /5, "
                f"{sign}{d['delta']:.2f} к среднему за последние 4 недели "
                f"({d['base_avg']:.2f} /5)"
            )

        if improved:
            top_imp = improved[:2]
            strengths_lines = "<br>".join(fmt_param_line(x) for x in top_imp)
            strengths_html = (
                "<p><b>Сильные стороны недели:</b><br>"
                f"{strengths_lines}</p>"
            )
        else:
            strengths_html = (
                "<p><b>Сильные стороны недели:</b><br>"
                "Существенных улучшений относительно среднего за последние 4 недели не зафиксировано.</p>"
            )

        if declined:
            top_decl = declined[:2]
            risks_lines = "<br>".join(fmt_param_line(x) for x in top_decl)
            risks_html = (
                "<p><b>Зоны риска:</b><br>"
                f"{risks_lines}</p>"
            )
        else:
            risks_html = (
                "<p><b>Зоны риска:</b><br>"
                "Нет параметров со значимым ухудшением относительно последних 4 недель.</p>"
            )

    # --- Собираем HTML-блок целиком ---
    html = f"""
    <h3>Динамика и точки внимания</h3>

    <p><b>Краткосрочно (текущая неделя vs прошлая неделя):</b><br>
    {short_overall_line}<br>
    {short_nps_line}
    </p>

    <p><b>Долгосрочно (текущая неделя vs среднее за последние 4 недели):</b><br>
    {long_overall_line}<br>
    {long_nps_line}
    </p>

    {strengths_html}
    {risks_html}
    """
    return html


# =========================
# Charts
# =========================
def _week_order_key(k):
    try:
        y, w = str(k).split("-W")
        return int(y) * 100 + int(w)
    except Exception:
        return 0

def plot_radar_params(week_df: pd.DataFrame, month_df: pd.DataFrame, path_png: str):
    """
    Радар-диаграмма (Неделя vs Месяц) по средним /5 по ключевым параметрам.
    """
    if week_df is None or week_df.empty or month_df is None or month_df.empty:
        return None

    def to_map(df):
        mp = {}
        for _, r in df.iterrows():
            p = str(r["param"])
            if p == "nps":
                continue
            mp[p] = float(r["avg5"]) if pd.notna(r["avg5"]) else np.nan
        return mp

    W = to_map(week_df)
    M = to_map(month_df)

    order = [
        "overall",
        "spir_checkin",
        "clean_checkin",
        "comfort",
        "spir_stay",
        "tech_service",
        "housekeeping",
        "breakfast",
        "atmosphere",
        "location",
        "value",
        "return_intent",
    ]
    order = [p for p in order if (p in W or p in M)]
    if len(order) < 3:
        return None  # без минимум 3 осей радар выглядит странно

    labels = [PARAM_TITLES.get(p, p) for p in order]
    wvals  = [W.get(p, np.nan) for p in order]
    mvals  = [M.get(p, np.nan) for p in order]

    N = len(order)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)

    angles_closed = np.concatenate([angles, [angles[0]]])
    w_closed = np.array(wvals + [wvals[0]])
    m_closed = np.array(mvals + [mvals[0]])

    fig = plt.figure(figsize=(7.5, 6.5))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids(np.degrees(angles), labels, fontsize=9)
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 5)

    ax.plot(angles_closed, w_closed, marker="o", linewidth=1, label="Неделя")
    ax.fill(angles_closed, w_closed, alpha=0.1)
    ax.plot(angles_closed, m_closed, marker="o", linewidth=1, linestyle="--", label="Текущий месяц")
    ax.fill(angles_closed, m_closed, alpha=0.1)

    ax.set_title("Анкеты: параметры (Неделя vs Текущий месяц), шкала /5")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

    plt.tight_layout()
    plt.savefig(path_png)
    plt.close()
    return path_png

def plot_params_heatmap(history_df: pd.DataFrame, path_png: str):
    """
    Теплокарта по последним 8 неделям:
    строки — параметры с наибольшим числом ответов,
    колонки — недели,
    значение цвета — средняя /5.
    """
    if history_df is None or history_df.empty:
        return None

    df = history_df.copy()
    # числа с запятой → float
    df["avg5"] = _to_num_series(df.get("avg5", pd.Series(dtype=str)))
    df["answered"] = _to_num_series(df.get("answered", pd.Series(dtype=str)))

    # NPS не кладём в теплокарту - это не /5
    df = df[df["param"] != "nps"].copy()
    if df.empty:
        return None

    weeks = sorted(df["week_key"].unique(), key=_week_order_key)[-8:]
    df = df[df["week_key"].isin(weeks)].copy()
    if df.empty:
        return None

    top_params = (
        df.groupby("param")["answered"]
          .sum()
          .sort_values(ascending=False)
          .head(10)
          .index
          .tolist()
    )
    df = df[df["param"].isin(top_params)]

    pv = (
        df.pivot_table(
            index="param",
            columns="week_key",
            values="avg5",
            aggfunc="mean",
        )
        .reindex(index=top_params, columns=weeks)
    )
    if pv.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(pv.values, aspect="auto")

    ax.set_yticks(range(len(pv.index)))
    ax.set_yticklabels([PARAM_TITLES.get(p, p) for p in pv.index])

    ax.set_xticks(range(len(pv.columns)))
    ax.set_xticklabels(list(pv.columns), rotation=45)

    ax.set_title("Анкеты: средняя /5 по параметрам (8 последних недель)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(path_png)
    plt.close()
    return path_png

def plot_overall_nps_trends(history_df: pd.DataFrame, path_png: str):
    """
    Линейный график за 12 последних недель:
    - итоговая оценка /5,
    - NPS,
    + 4-недельное скользящее.
    """
    if history_df is None or history_df.empty:
        return None

    df = history_df.copy()

    df["avg5"] = _to_num_series(df.get("avg5", pd.Series(dtype=str)))
    df["nps_value"] = _to_num_series(df.get("nps_value", pd.Series(dtype=str)))

    ov  = df[df["param"] == "overall"][["week_key","avg5"]]
    npv = df[df["param"] == "nps"][["week_key","nps_value"]]

    weeks = sorted(
        set(ov["week_key"]).union(set(npv["week_key"])),
        key=_week_order_key
    )[-12:]
    if not weeks:
        return None

    ov  = ov.set_index("week_key").reindex(weeks)
    npv = npv.set_index("week_key").reindex(weeks)

    ov_roll  = ov["avg5"].rolling(window=4, min_periods=2).mean()
    nps_roll = npv["nps_value"].rolling(window=4, min_periods=2).mean()

    fig, ax1 = plt.subplots(figsize=(10, 5.0))

    ax1.plot(weeks, ov["avg5"].values, marker="o", label="Итоговая /5")
    ax1.plot(weeks, ov_roll.values, linestyle="--", label="Итоговая /5 (скользящее)")
    ax1.set_ylim(0, 5)
    ax1.set_ylabel("Итоговая /5")

    ax2 = ax1.twinx()
    ax2.plot(weeks, npv["nps_value"].values, marker="s", label="NPS", alpha=0.8)
    ax2.plot(weeks, nps_roll.values, linestyle="--", label="NPS (скользящее)", alpha=0.8)
    ax2.set_ylim(-100, 100)
    ax2.set_ylabel("NPS, п.п.")

    ax1.set_title("Анкеты: тренды Итоговой /5 и NPS (12 недель)")
    ax1.set_xticks(range(len(weeks)))
    ax1.set_xticklabels(weeks, rotation=45)

    lines, labels = [], []
    for ax in (ax1, ax2):
        l, lab = ax.get_legend_handles_labels()
        lines += l
        labels += lab
    ax1.legend(lines, labels, loc="upper left")

    plt.tight_layout()
    plt.savefig(path_png)
    plt.close()
    return path_png

# =========================
# Email helpers
# =========================
def attach_file(msg, path):
    if not path or not os.path.exists(path):
        return
    ctype, encoding = mimetypes.guess_type(path)
    if ctype is None or encoding is not None:
        ctype = "application/octet-stream"
    maintype, subtype = ctype.split("/", 1)
    with open(path, "rb") as fp:
        part = MIMEBase(maintype, subtype)
        part.set_payload(fp.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            "attachment",
            filename=os.path.basename(path),
        )
        msg.attach(part)

def send_email(subject, html_body, attachments=None):
    if not RECIPIENTS:
        print("[WARN] RECIPIENTS is empty; skip email")
        return
    msg = MIMEMultipart("mixed")
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM
    msg["To"] = ", ".join(RECIPIENTS)

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(html_body, "html", "utf-8"))
    msg.attach(alt)

    for p in attachments or []:
        attach_file(msg, p)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        if SMTP_USER and SMTP_PASS:
            server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_FROM, RECIPIENTS, msg.as_string())

# =========================
# Main
# =========================
def main():
    # 1) Берём свежий отчёт (Report_DD-MM-YYYY.xlsx) из Drive
    file_id, fname, fdate = latest_report_from_drive()
    data = drive_download(file_id)

    xls = pd.ExcelFile(io.BytesIO(data))
    if "Оценки гостей" in xls.sheet_names:
        raw = pd.read_excel(xls, sheet_name="Оценки гостей")
    else:
        # fallback на старую структуру
        raw = pd.read_excel(xls, sheet_name="Reviews")

    # нормализация и недельная агрегация по новой логике из surveys_core
    norm_df, agg_week = parse_and_aggregate_weekly(raw)

    # 2) обновляем лист surveys_history:
    #    - без дублей по неделе
    added = upsert_week(agg_week)
    print(f"[INFO] upsert_week(): добавлено строк недели: {added}")

    # 3) читаем всю историю обратно
    hist = gs_get_df(SURVEYS_TAB, "A:I")
    if hist.empty:
        raise RuntimeError("surveys_history пуст — нечего анализировать.")

    # определяем дату текущей недели
    wk_key = str(agg_week["week_key"].iloc[0])
    w_start = iso_week_monday(wk_key)
    w_end   = w_start + dt.timedelta(days=6)

    prev_start = w_start - dt.timedelta(days=7)
    prev_end   = prev_start + dt.timedelta(days=6)
    Prev = surveys_aggregate_period(hist, prev_start, prev_end)

    # предшествующее окно "последние 4 недели до этой"
    l4_start = w_start - dt.timedelta(days=28)
    l4_end   = w_start - dt.timedelta(days=1)
    L4 = surveys_aggregate_period(hist, l4_start, l4_end)

    # периоды: неделя / месяц-to-date / квартал-to-date / год-to-date
    ranges = period_ranges_for_week(w_start)

    # Итого за всё время: от первой недели в истории до последней
    hist_mondays = hist["week_key"].map(lambda k: iso_week_monday(str(k)))
    all_start = hist_mondays.min()
    all_end   = hist_mondays.max() + dt.timedelta(days=6)

    # агрегаты по периодам
    W = surveys_aggregate_period(hist, w_start, w_end)
    M = surveys_aggregate_period(hist, ranges["mtd"]["start"], ranges["mtd"]["end"])
    Q = surveys_aggregate_period(hist, ranges["qtd"]["start"], ranges["qtd"]["end"])
    Y = surveys_aggregate_period(hist, ranges["ytd"]["start"], ranges["ytd"]["end"])
    T = surveys_aggregate_period(hist, all_start, all_end)

    # Тело письма
    head  = header_block(w_start, w_end, W, M, Q, Y, T)
    table = table_params_block(W["by_param"], M["by_param"], Q["by_param"], Y["by_param"], T["by_param"])
    trends  = trends_block(W, Prev, L4)
    html = head + table + trends + footnote_block()

    # Построение графиков
    charts = []
    p1 = "/tmp/surveys_radar.png"
    p2 = "/tmp/surveys_heatmap.png"
    p3 = "/tmp/surveys_trends.png"

    if plot_radar_params(W["by_param"], M["by_param"], p1): charts.append(p1)
    if plot_params_heatmap(hist, p2):                      charts.append(p2)
    if plot_overall_nps_trends(hist, p3):                  charts.append(p3)

    subject = f"ARTSTUDIO Nevsky. Анкеты TL: Marketing — неделя {w_start:%d.%m}–{w_end:%d.%m}"
    send_email(subject, html, attachments=charts)

if __name__ == "__main__":
    main()
