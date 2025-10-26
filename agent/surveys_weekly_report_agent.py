# agent/surveys_weekly_report_agent.py
#
# ARTSTUDIO Nevsky — еженедельный отчёт по анкетам TL: Marketing
#
# В этой версии:
# - расчёты только по 5-балльной шкале (+ NPS), без avg10
# - upsert_week() пишет недели в Sheets в формате без avg10
# - письма формируются в бренд-стиле и с управленческой аналитикой:
#     • Шапка с периодами (Неделя / Месяц / Квартал / Год / Итого)
#     • Таблица показателей качества
#     • Динамика и точки внимания (краткосрочно, тренд, сильные стороны, зоны риска, тревоги)
#     • Сравнение с прошлым годом (неделя, месяц-to-date, квартал-to-date, год-to-date)
#     • Сноска с методикой
# - subject письма в формате:
#     "ARTSTUDIO Nevsky. Анкеты TL: Marketing — неделя 13–19 окт 2025 г."
#
# Цветовая палитра:
#   Основные:
#     RGB 196 154 95   -> #C49A5F  (границы / акценты)
#     RGB 38 45 54     -> #262D36  (основной текст)
#     RGB 255 255 255  -> #FFFFFF  (белый фон)
#   Дополнительные:
#     RGB 255 246 229  -> #FFF6E5  (светлый бежевый фон таблиц)
#     RGB 239 125 23   -> #EF7D17  (отрицательная дельта / риск)
#     RGB 255 204 0    -> #FFCC00  (положительная дельта / улучшение)
#
# Требуемые Secrets:
#   DRIVE_FOLDER_ID        — ID папки в Drive с файлами Report_DD-MM-YYYY.xlsx
#   SHEETS_HISTORY_ID      — ID Google Sheets с листом surveys_history
#   GOOGLE_SERVICE_ACCOUNT_JSON       или GOOGLE_SERVICE_ACCOUNT_JSON_CONTENT
#   RECIPIENTS, SMTP_USER, SMTP_PASS, SMTP_FROM
#
# Примечание:
#   Лист surveys_history должен иметь шапку:
#     week_key | param | surveys_total | answered | avg5 |
#     promoters | detractors | nps_answers | nps_value
#
#   И не содержать дубликатов одной и той же недели.
#

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

# --- утилиты периодов/дат/подписей из metrics_core ---
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


# =====================================================
# ENV & Google API clients
# =====================================================

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


# =====================================================
# Цвета и визуальные константы
# =====================================================

COLOR_TEXT_MAIN   = "#262D36"  # основной текст
COLOR_BG_HEADER   = "#FFF6E5"  # фон таблиц/шапок
COLOR_BORDER      = "#C49A5F"  # границы
COLOR_POSITIVE    = "#FFCC00"  # улучшение
COLOR_NEGATIVE    = "#EF7D17"  # ухудшение / риск


# =====================================================
# Drive helpers
# =====================================================

WEEKLY_RE = re.compile(r"^Report_(\d{2})-(\d{2})-(\d{4})\.xlsx$", re.IGNORECASE)

def latest_report_from_drive():
    """
    Находим самый свежий Report_DD-MM-YYYY.xlsx в папке DRIVE_FOLDER_ID.
    Возвращаем (file_id, filename, date_obj).
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


# =====================================================
# Sheets helpers
# =====================================================

def ensure_tab(spreadsheet_id: str, tab_name: str, header: list[str]):
    """
    Гарантируем, что лист существует и (если создаётся) получает шапку.
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

def gs_get_df(tab: str, a1: str) -> pd.DataFrame:
    """
    Считываем диапазон из Google Sheets → DataFrame.
    Если в листе только шапка, вернёт пустой df.
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


# Шапка для листа surveys_history
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
    Конверсия недельной агрегации (agg_week из surveys_core)
    к строкам для Sheets (формат без avg10).
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
    Обновляем/перезаписываем данные по конкретной неделе в листе surveys_history:
    - читаем историю
    - удаляем строки с той же week_key
    - чистим тело листа (кроме шапки)
    - пишем обратно остальные недели + текущую
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

    # очистка старого тела листа
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


# =====================================================
# Даты и подписи периодов
# =====================================================

RU_MONTH_SHORT = {
    1: "янв", 2: "фев", 3: "мар", 4: "апр", 5: "май", 6: "июн",
    7: "июл", 8: "авг", 9: "сен", 10: "окт", 11: "ноя", 12: "дек",
}

def human_date(d: date) -> str:
    # Пример: 15 янв 2024 г.
    return f"{d.day} {RU_MONTH_SHORT[d.month]} {d.year} g.".replace(" g.", " г.")

def range_label_simple(start: date, end: date) -> str:
    """
    Диапазон дат:
    - '13–19 окт 2025'
    - '28 сен – 5 окт 2025'
    - '30 дек 2025 – 5 янв 2026'
    (без 'г.')
    """
    def _one(d: date, show_year: bool) -> str:
        m = RU_MONTH_SHORT[d.month]
        if show_year:
            return f"{d.day} {m} {d.year}"
        else:
            return f"{d.day} {m}"

    same_year  = (start.year == end.year)
    same_month = (start.month == end.month) and same_year

    if same_month:
        return f"{start.day}–{end.day} {RU_MONTH_SHORT[start.month]} {start.year}"
    else:
        left  = _one(start, show_year=not same_year)
        right = _one(end,   show_year=True)
        return f"{left} – {right}"

def add_year_suffix(label_no_g: str) -> str:
    # '13–19 окт 2025' → '13–19 окт 2025 г.'
    return f"{label_no_g} г."

def pretty_month_label(d: date) -> str:
    # month_label(d) → 'октябрь 2025'
    ml = month_label(d)
    ml_cap = ml[0].upper() + ml[1:]
    return f"{ml_cap} г."

def pretty_quarter_label(d: date) -> str:
    # quarter_label(d) → 'IV кв. 2025' → 'IV квартал 2025 г.'
    ql = quarter_label(d).replace("кв.", "квартал")
    return f"{ql} г."

def pretty_year_label(d: date) -> str:
    # year_label(d) → '2025' → '2025 г.'
    return f"{year_label(d)} г."

def shift_year_back_safe(d: date) -> date:
    """
    Берём ту же календарную дату за прошлый год.
    Если дата "ломается" (например 29 февраля), fallback = минус 365 дней.
    """
    try:
        return d.replace(year=d.year - 1)
    except ValueError:
        return d - dt.timedelta(days=365)


# =====================================================
# Агрегация по периодам
# =====================================================

def _to_num_series(s: pd.Series) -> pd.Series:
    """
    Перевод столбцов из Sheets в числа:
    '4,71' → 4.71, '5' → 5.0, иначе NaN.
    """
    if s is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(
        s.astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )

def _weighted_avg(values: pd.Series, weights: pd.Series) -> float | None:
    """
    Взвешенная средняя по шкале /5.
    values  = средние по неделям
    weights = сколько реально ответили на этот вопрос в каждую неделю
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

    return float(s / W)

def surveys_aggregate_period(history_df: pd.DataFrame, start: date, end: date) -> dict:
    """
    Агрегация за произвольный период (неделя / MTD / QTD / YTD / Итого / бэйзлайн L4 и т.д).
    Возвращает:
    {
      "by_param": DataFrame с колонками:
          param, surveys_total, answered, avg5,
          promoters, detractors, nps_answers, nps_value
      "totals": {
          "surveys_total": int,
          "overall5": float|None,
          "nps": float|None
      }
    }
    """

    empty_bp_cols = [
        "param","surveys_total","answered","avg5",
        "promoters","detractors","nps_answers","nps_value",
    ]
    if history_df is None or history_df.empty:
        return {
            "by_param": pd.DataFrame(columns=empty_bp_cols),
            "totals": {"surveys_total": 0, "overall5": None, "nps": None},
        }

    df = history_df.copy()

    numeric_cols = [
        "surveys_total","answered","avg5",
        "promoters","detractors","nps_answers","nps_value",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = _to_num_series(df[c])

    # Привязываем week_key к фактическому понедельнику
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

        avg5_weighted = _weighted_avg(g["avg5"], g["answered"])
        if avg5_weighted is not None:
            avg5_weighted = round(avg5_weighted, 2)

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
                        (promoters_sum / nps_answers_sum - detractors_sum / nps_answers_sum) * 100.0
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


# =====================================================
# Форматирование чисел и дельт
# =====================================================

def _to_float_or_none_local(x):
    try:
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None

def fmt_avg5(x):
    """ '4.7 /5' или '—' """
    val = _to_float_or_none_local(x)
    if val is None:
        return "—"
    return f"{val:.1f} /5"

def fmt_int(x):
    """ '5' или '—' """
    val = _to_float_or_none_local(x)
    if val is None:
        return "—"
    return str(int(round(val)))

def fmt_nps(x):
    """ '60.0%' или '—' """
    val = _to_float_or_none_local(x)
    if val is None:
        return "—"
    return f"{val:.1f}%"

def fmt_delta_arrow(curr, prev, suffix=""):
    """
    Возвращает HTML с цветной стрелкой:
    ▲ +0.3 / ▼ -0.4
    Если почти ноль → 0.0
    suffix = "" для оценки, " п.п." для NPS
    """
    c = _to_float_or_none_local(curr)
    p = _to_float_or_none_local(prev)
    if c is None or p is None:
        return "—"

    d = round(c - p, 1)

    if abs(d) < 0.05:
        # нейтрально
        return f"0.0{suffix}"

    if d > 0:
        return (
            f"<span style='color:{COLOR_POSITIVE};font-weight:bold;'>"
            f"▲ +{d:.1f}{suffix}"
            f"</span>"
        )
    else:
        return (
            f"<span style='color:{COLOR_NEGATIVE};font-weight:bold;'>"
            f"▼ {d:.1f}{suffix}"
            f"</span>"
        )


# =====================================================
# Тексты, названия параметров
# =====================================================

PARAM_TITLES = {
    "overall":        "Итоговая оценка",
    "spir_checkin":   "Работа СПиР при заезде",
    "clean_checkin":  "Чистота номера при заезде",
    "comfort":        "Комфорт и оснащение номера",
    "spir_stay":      "Работа СПиР во время проживания",
    "tech_service":   "Работа ИТС",
    "housekeeping":   "Чистота номера во время проживания",
    "breakfast":      "Завтраки",
    "atmosphere":     "Атмосфера",
    "location":       "Расположение",
    "value":          "Цена/качество",
    "return_intent":  "Готовность вернуться",
    "nps":            "NPS",
}


# =====================================================
# Блок: шапка письма
# =====================================================

def _period_card_html(period_label: str, agg: dict) -> str:
    return (
        f"<div style='margin-bottom:12px;color:{COLOR_TEXT_MAIN};'>"
        f"<div style='font-weight:bold;'>{period_label}</div>"
        f"<div>Анкет: {fmt_int(agg['totals']['surveys_total'])}</div>"
        f"<div>Итоговая оценка: {fmt_avg5(agg['totals']['overall5'])}</div>"
        f"<div>NPS: {fmt_nps(agg['totals']['nps'])}</div>"
        f"</div>"
    )

def header_block(
    obj_name: str,
    week_label_full: str,
    month_label_full: str,
    quarter_label_full: str,
    year_label_full: str,
    total_label_full: str,
    W: dict,
    M: dict,
    Q: dict,
    Y: dict,
    T: dict,
):
    """
    Верхняя часть письма — карточки по периодам.
    Пример заголовков:
      "Неделя 13–19 окт 2025 г."
      "Октябрь 2025 г."
      "IV квартал 2025 г."
      "2025 г."
      "Итого"
    """
    title_html = (
        f"<h2 style='margin-bottom:16px;color:{COLOR_TEXT_MAIN};'>"
        f"{obj_name} — анкеты гостей TL: Marketing"
        f"</h2>"
    )

    cards_html = (
        _period_card_html(week_label_full,    W)
        + _period_card_html(month_label_full,   M)
        + _period_card_html(quarter_label_full, Q)
        + _period_card_html(year_label_full,    Y)
        + _period_card_html(total_label_full,   T)
    )

    return title_html + cards_html


# =====================================================
# Блок: основная таблица параметров
# =====================================================

def table_params_block(
    W_df: pd.DataFrame,
    M_df: pd.DataFrame,
    Q_df: pd.DataFrame,
    Y_df: pd.DataFrame,
    T_df: pd.DataFrame,
    week_col_label: str,
    month_col_label: str,
    quarter_col_label: str,
    year_col_label: str,
    total_col_label: str = "Итого",
):
    """
    Таблица параметров качества:
    - Средняя оценка (/5, 1 знак после запятой)
    - Ответы (сколько гостей реально ответили)
    NPS показываем как процент и количество ответивших.
    """

    def df_to_map(df):
        mp = {}
        for _, r in df.iterrows():
            mp[str(r["param"])] = r.to_dict()
        return mp

    Wm = df_to_map(W_df)
    Mm = df_to_map(M_df)
    Qm = df_to_map(Q_df)
    Ym = df_to_map(Y_df)
    Tm = df_to_map(T_df)

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
            return (
                f"<td style='text-align:right;'>—</td>"
                f"<td style='text-align:right;'>—</td>"
            )

        if param == "nps":
            return (
                f"<td style='text-align:right;'>{fmt_nps(r.get('nps_value'))}</td>"
                f"<td style='text-align:right;'>{fmt_int(r.get('nps_answers'))}</td>"
            )

        return (
            f"<td style='text-align:right;'>{fmt_avg5(r.get('avg5'))}</td>"
            f"<td style='text-align:right;'>{fmt_int(r.get('answered'))}</td>"
        )

    rows_html = []
    for p in order:
        title = PARAM_TITLES.get(p, p)
        rows_html.append(
            "<tr>"
            f"<td style='white-space:nowrap;color:{COLOR_TEXT_MAIN};'><b>{title}</b></td>"
            + cell(Wm, p)
            + cell(Mm, p)
            + cell(Qm, p)
            + cell(Ym, p)
            + cell(Tm, p)
            + "</tr>"
        )

    html = f"""
    <h3 style="color:{COLOR_TEXT_MAIN};margin-top:16px;">Ключевые показатели по параметрам качества</h3>
    <table border='1' cellspacing='0' cellpadding='6'
           style="border-collapse:collapse;border-color:{COLOR_BORDER};color:{COLOR_TEXT_MAIN};font-size:14px;">
      <tr style="background-color:{COLOR_BG_HEADER};">
        <th rowspan="2" style="border:1px solid {COLOR_BORDER};text-align:left;">Параметр</th>
        <th colspan="2" style="border:1px solid {COLOR_BORDER};text-align:center;">{week_col_label}</th>
        <th colspan="2" style="border:1px solid {COLOR_BORDER};text-align:center;">{month_col_label}</th>
        <th colspan="2" style="border:1px solid {COLOR_BORDER};text-align:center;">{quarter_col_label}</th>
        <th colspan="2" style="border:1px solid {COLOR_BORDER};text-align:center;">{year_col_label}</th>
        <th colspan="2" style="border:1px solid {COLOR_BORDER};text-align:center;">{total_col_label}</th>
      </tr>
      <tr style="background-color:{COLOR_BG_HEADER};">
        <th style="border:1px solid {COLOR_BORDER};text-align:right;">Ср. оценка</th>
        <th style="border:1px solid {COLOR_BORDER};text-align:right;">Ответы</th>
        <th style="border:1px solid {COLOR_BORDER};text-align:right;">Ср. оценка</th>
        <th style="border:1px solid {COLOR_BORDER};text-align:right;">Ответы</th>
        <th style="border:1px solid {COLOR_BORDER};text-align:right;">Ср. оценка</th>
        <th style="border:1px solid {COLOR_BORDER};text-align:right;">Ответы</th>
        <th style="border:1px solid {COLOR_BORDER};text-align:right;">Ср. оценка</th>
        <th style="border:1px solid {COLOR_BORDER};text-align:right;">Ответы</th>
        <th style="border:1px solid {COLOR_BORDER};text-align:right;">Ср. оценка</th>
        <th style="border:1px solid {COLOR_BORDER};text-align:right;">Ответы</th>
      </tr>
      {''.join(rows_html)}
    </table>
    """
    return html


# =====================================================
# Блок: Динамика и точки внимания
# =====================================================

def trends_block(W, Prev, L4):
    """
    Аналитика качества:
    - Краткосрочно: эта неделя vs предыдущая неделя
    - Тенденция: эта неделя vs средний уровень за последние 4 недели
    - Сильные стороны недели (рост >=0.2 при >=2 ответах)
    - Зоны риска недели (падение <=-0.2 при >=2 ответах)
    - Предупреждения (показатели <4.0 /5 при >=3 ответах)
    """

    def df_to_map(df):
        mp = {}
        if df is None or len(df) == 0:
            return mp
        for _, r in df.iterrows():
            mp[str(r["param"])] = r.to_dict()
        return mp

    # 1. Краткосрочно (vs предыдущая неделя)
    cur_overall  = W["totals"]["overall5"]
    prev_overall = Prev["totals"]["overall5"]

    cur_nps   = W["totals"]["nps"]
    prev_nps  = Prev["totals"]["nps"]

    short_block = (
        "<p style='margin-top:16px;color:{color};'>"
        "<b>Краткосрочно (по сравнению с предыдущей неделей):</b><br>"
        "Итоговая оценка: {cur_over} ({delta_over} к прошлой неделе).<br>"
        "NPS: {cur_nps} ({delta_nps} к прошлой неделе)."
        "</p>"
    ).format(
        color=COLOR_TEXT_MAIN,
        cur_over=fmt_avg5(cur_overall),
        delta_over=fmt_delta_arrow(cur_overall, prev_overall, suffix=""),
        cur_nps=fmt_nps(cur_nps),
        delta_nps=fmt_delta_arrow(cur_nps, prev_nps, suffix=" п.п."),
    )

    # 2. Тенденция (vs последние 4 недели)
    base_overall = L4["totals"]["overall5"]
    base_nps     = L4["totals"]["nps"]

    if base_overall is not None:
        long_overall_line = (
            f"Итоговая оценка сейчас: {fmt_avg5(cur_overall)} "
            f"({fmt_delta_arrow(cur_overall, base_overall, suffix='')} "
            f"к среднему последних недель)."
        )
    else:
        long_overall_line = (
            "Недостаточно данных для анализа итоговой оценки по тренду последних недель."
        )

    if base_nps is not None:
        long_nps_line = (
            f"NPS сейчас: {fmt_nps(cur_nps)} "
            f"({fmt_delta_arrow(cur_nps, base_nps, suffix=' п.п.')} "
            f"к среднему последних недель)."
        )
    else:
        long_nps_line = (
            "Недостаточно данных для анализа NPS по тренду последних недель."
        )

    long_block = (
        "<p style='margin-top:12px;color:{color};'>"
        "<b>Тенденция (по сравнению со средним уровнем последних 4 недель):</b><br>"
        "{o}<br>"
        "{n}"
        "</p>"
    ).format(
        color=COLOR_TEXT_MAIN,
        o=long_overall_line,
        n=long_nps_line,
    )

    # 3. Сильные стороны / зоны риска по параметрам
    Wmap  = df_to_map(W["by_param"])
    L4map = df_to_map(L4["by_param"])

    deltas = []
    for p, wrow in Wmap.items():
        if p == "nps":
            continue

        w_avg = _to_float_or_none_local(wrow.get("avg5"))
        w_ans = _to_float_or_none_local(wrow.get("answered"))

        base_avg = _to_float_or_none_local(L4map.get(p, {}).get("avg5"))
        base_ans = _to_float_or_none_local(L4map.get(p, {}).get("answered"))

        if w_avg is None or base_avg is None:
            continue
        if w_ans is None or base_ans is None:
            continue

        # фильтр репрезентативности: хотя бы 2 ответа и в этой неделе, и в baseline
        if w_ans < 2 or base_ans < 2:
            continue

        delta_val = round(w_avg - base_avg, 1)
        deltas.append({
            "param": p,
            "week_avg": w_avg,
            "delta": delta_val,
        })

    IMPROVE_THRESHOLD = 0.2
    DECLINE_THRESHOLD = -0.2

    improved = [d for d in deltas if d["delta"] >= IMPROVE_THRESHOLD]
    declined = [d for d in deltas if d["delta"] <= DECLINE_THRESHOLD]

    improved.sort(key=lambda x: x["delta"], reverse=True)
    declined.sort(key=lambda x: x["delta"])

    def fmt_delta_span(d):
        if abs(d) < 0.05:
            return "0.0"
        if d > 0:
            return (
                f"<span style='color:{COLOR_POSITIVE};font-weight:bold;'>▲ +{d:.1f}</span>"
            )
        else:
            return (
                f"<span style='color:{COLOR_NEGATIVE};font-weight:bold;'>▼ {d:.1f}</span>"
            )

    # Сильные стороны
    if improved:
        imp_lines = []
        for d in improved[:2]:
            title = PARAM_TITLES.get(d["param"], d["param"])
            imp_lines.append(
                f"- {title}: {d['week_avg']:.1f} /5 ({fmt_delta_span(d['delta'])})."
            )
        strengths_html = (
            "<p style='margin-top:12px;color:{color};'>"
            "<b>Сильные стороны недели:</b><br>"
            "{lines}"
            "</p>"
        ).format(
            color=COLOR_TEXT_MAIN,
            lines="<br>".join(imp_lines),
        )
    else:
        strengths_html = (
            "<p style='margin-top:12px;color:{color};'>"
            "<b>Сильные стороны недели:</b><br>"
            "Существенного роста показателей относительно базового уровня не зафиксировано."
            "</p>"
        ).format(color=COLOR_TEXT_MAIN)

    # Зоны риска
    if declined:
        dec_lines = []
        for d in declined[:2]:
            title = PARAM_TITLES.get(d["param"], d["param"])
            dec_lines.append(
                f"- {title}: {d['week_avg']:.1f} /5 ({fmt_delta_span(d['delta'])})."
            )
        risks_html = (
            "<p style='margin-top:12px;color:{color};'>"
            "<b>Зоны риска недели:</b><br>"
            "{lines}"
            "</p>"
        ).format(
            color=COLOR_TEXT_MAIN,
            lines="<br>".join(dec_lines),
        )
    else:
        risks_html = (
            "<p style='margin-top:12px;color:{color};'>"
            "<b>Зоны риска недели:</b><br>"
            "Значимого снижения показателей относительно базового уровня не зафиксировано."
            "</p>"
        ).format(color=COLOR_TEXT_MAIN)

    # 4. Тревоги (<4.0 /5 и >=3 ответов)
    alerts = []
    for p, wrow in Wmap.items():
        if p == "nps":
            continue
        w_avg = _to_float_or_none_local(wrow.get("avg5"))
        w_ans = _to_float_or_none_local(wrow.get("answered"))
        if w_avg is None or w_ans is None:
            continue
        if w_ans >= 3 and w_avg < 4.0:
            title = PARAM_TITLES.get(p, p)
            alerts.append(
                f"{title} — {w_avg:.1f} /5 ({int(round(w_ans))} ответов)"
            )

    if alerts:
        alert_block = (
            "<p style='margin-top:12px;color:{neg};font-weight:bold;'>"
            "⚠ Обратите внимание: {items}. "
            "Показатели опустились ниже 4.0 /5 на этой неделе."
            "</p>"
        ).format(
            neg=COLOR_NEGATIVE,
            items="; ".join(alerts),
        )
    else:
        alert_block = ""

    html = (
        f"<h3 style='color:{COLOR_TEXT_MAIN};margin-top:24px;'>Динамика и точки внимания</h3>"
        + short_block
        + long_block
        + strengths_html
        + risks_html
        + alert_block
    )
    return html


# =====================================================
# Блок: сравнение с прошлым годом
# =====================================================

def yoy_block(period_rows: list[dict]):
    """
    Таблица "Сравнение с прошлым годом".
    Ожидает список строк:
    {
      "label": "Неделя 13–19 окт 2025 г.",
      "cur":  {"surveys_total":..., "overall5":..., "nps":...},
      "prev": {"surveys_total":..., "overall5":..., "nps":...},
    }
    """

    row_html = []
    for row in period_rows:
        label = row["label"]
        cur   = row.get("cur",  {})
        prev  = row.get("prev", {})

        cur_cnt  = cur.get("surveys_total")
        cur_over = cur.get("overall5")
        cur_nps  = cur.get("nps")

        prev_over = prev.get("overall5")
        prev_nps  = prev.get("nps")

        delta_over = fmt_delta_arrow(cur_over, prev_over, suffix="")
        delta_nps  = fmt_delta_arrow(cur_nps,  prev_nps,  suffix=" п.п.")

        row_html.append(
            "<tr>"
            f"<td style='border:1px solid {COLOR_BORDER};text-align:left;color:{COLOR_TEXT_MAIN};white-space:nowrap;'>{label}</td>"
            f"<td style='border:1px solid {COLOR_BORDER};text-align:right;color:{COLOR_TEXT_MAIN};'>{fmt_int(cur_cnt)}</td>"
            f"<td style='border:1px solid {COLOR_BORDER};text-align:right;color:{COLOR_TEXT_MAIN};'>{fmt_avg5(cur_over)}</td>"
            f"<td style='border:1px solid {COLOR_BORDER};text-align:right;color:{COLOR_TEXT_MAIN};'>{delta_over}</td>"
            f"<td style='border:1px solid {COLOR_BORDER};text-align:right;color:{COLOR_TEXT_MAIN};'>{fmt_nps(cur_nps)}</td>"
            f"<td style='border:1px solid {COLOR_BORDER};text-align:right;color:{COLOR_TEXT_MAIN};'>{delta_nps}</td>"
            "</tr>"
        )

    html = f"""
    <h3 style="color:{COLOR_TEXT_MAIN};margin-top:24px;">Сравнение с прошлым годом</h3>
    <table border='1' cellspacing='0' cellpadding='6'
           style="border-collapse:collapse;border-color:{COLOR_BORDER};color:{COLOR_TEXT_MAIN};font-size:14px;">
      <tr style="background-color:{COLOR_BG_HEADER};color:{COLOR_TEXT_MAIN};">
        <th style='border:1px solid {COLOR_BORDER};text-align:left;'>Период</th>
        <th style='border:1px solid {COLOR_BORDER};text-align:right;'>Анкет</th>
        <th style='border:1px solid {COLOR_BORDER};text-align:right;'>Итоговая оценка</th>
        <th style='border:1px solid {COLOR_BORDER};text-align:right;'>Δ к прошлому году</th>
        <th style='border:1px solid {COLOR_BORDER};text-align:right;'>NPS</th>
        <th style='border:1px solid {COLOR_BORDER};text-align:right;'>Δ к прошлому году</th>
      </tr>
      {''.join(row_html)}
    </table>
    """
    return html


# =====================================================
# Сноска (методика)
# =====================================================

def footnote_block(all_start: date, w_end: date):
    return f"""
    <hr style="margin-top:24px;border:0;border-top:1px solid {COLOR_BORDER};">
    <p style="font-size:12px;color:{COLOR_TEXT_MAIN};line-height:1.5;margin-top:12px;">
    <b>Пояснения.</b><br>
    • «Итого» — накопленная статистика с начала сбора анкет (с {human_date(all_start)}) по {human_date(w_end)}.<br>
    • Все оценки даны по шкале 1–5, где 1 — плохо, 5 — отлично. В таблицах приводится среднее значение с округлением до 0,1. В отчёте «Итоговая оценка» — общее впечатление гостя о проживании.<br>
    • «Ответы» — количество гостей, которые поставили оценку по конкретному вопросу. Если гость пропустил вопрос, он не влияет на среднюю.<br>
    • NPS рассчитывается по вопросу о готовности рекомендовать отель: 1–2 — детракторы, 3–4 — нейтральные, 5 — промоутеры. Пустые ответы не учитываются. NPS показан как разница между долей промоутеров и долей детракторов, в процентах.<br>
    • Δ в разделе «Динамика» и в блоке «Сравнение с прошлым годом» показывает изменение показателя. Рост выделяется как
      <span style="color:{COLOR_POSITIVE};font-weight:bold;">▲ положительное отклонение</span>,
      снижение — как
      <span style="color:{COLOR_NEGATIVE};font-weight:bold;">▼ отрицательное отклонение</span>.
      Для NPS дельта указана в п.п. (процентных пунктах).<br>
    • «Сравнение с прошлым годом» сопоставляет текущие результаты с той же точкой прошлого года: неделя к той же неделе, месяц на текущую дату к месяцу на ту же дату прошлого года, квартал на текущую дату квартала, год с начала года. Это позволяет отслеживать динамику качества по сезонам, а не только по календарю.
    </p>
    """


# =====================================================
# Графики (прикладываются во вложениях)
# =====================================================

def _week_order_key(k):
    try:
        y, w = str(k).split("-W")
        return int(y) * 100 + int(w)
    except Exception:
        return 0

def plot_radar_params(week_df: pd.DataFrame, month_df: pd.DataFrame, path_png: str):
    """
    Радар-диаграмма (Неделя vs Месяц) по средним /5
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

    Wm = to_map(week_df)
    Mm = to_map(month_df)

    param_order = [
        "overall","spir_checkin","clean_checkin","comfort",
        "spir_stay","tech_service","housekeeping","breakfast",
        "atmosphere","location","value","return_intent",
    ]
    order = [p for p in param_order if (p in Wm or p in Mm)]
    if len(order) < 3:
        return None

    labels = [PARAM_TITLES.get(p, p) for p in order]
    wvals  = [Wm.get(p, np.nan) for p in order]
    mvals  = [Mm.get(p, np.nan) for p in order]

    N = len(order)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)

    angles_closed = np.concatenate([angles, [angles[0]]])
    w_closed = np.array(wvals + [wvals[0]])
    m_closed = np.array(mvals + [mvals[0]])

    fig = plt.figure(figsize=(7.5, 6.5))
    fig.patch.set_facecolor("#FFFFFF")
    ax = plt.subplot(111, polar=True)
    ax.set_facecolor("#FFF6E5")  # лёгкий брендовый фон

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids(np.degrees(angles), labels, fontsize=9)
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 5)

    ax.plot(angles_closed, w_closed, marker="o", linewidth=1, label="Неделя")
    ax.fill(angles_closed, w_closed, alpha=0.1)
    ax.plot(angles_closed, m_closed, marker="o", linewidth=1, linestyle="--", label="Месяц (на сегодня)")
    ax.fill(angles_closed, m_closed, alpha=0.1)

    ax.set_title("Анкеты: параметры качества (/5)", color="#262D36", fontsize=11)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=8)

    plt.tight_layout()
    plt.savefig(path_png)
    plt.close()
    return path_png

def plot_params_heatmap(history_df: pd.DataFrame, path_png: str):
    """
    Теплокарта средних /5 за последние 8 недель по ключевым параметрам.
    """
    if history_df is None or history_df.empty:
        return None

    df = history_df.copy()
    df["avg5"] = _to_num_series(df.get("avg5", pd.Series(dtype=str)))
    df["answered"] = _to_num_series(df.get("answered", pd.Series(dtype=str)))

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
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFF6E5")

    im = ax.imshow(pv.values, aspect="auto")

    ax.set_yticks(range(len(pv.index)))
    ax.set_yticklabels([PARAM_TITLES.get(p, p) for p in pv.index], fontsize=8, color="#262D36")

    ax.set_xticks(range(len(pv.columns)))
    ax.set_xticklabels(list(pv.columns), rotation=45, fontsize=8, color="#262D36")

    ax.set_title("Анкеты: средняя оценка (/5) по параметрам, 8 последних недель", color="#262D36", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(path_png)
    plt.close()
    return path_png

def plot_overall_nps_trends(history_df: pd.DataFrame, path_png: str, as_of: date):
    """
    Линейный график за 12 последних недель:
    - Итоговая оценка (/5)
    - NPS (%)
    + 4-недельные скользящие средние
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
    fig.patch.set_facecolor("#FFFFFF")
    ax1.set_facecolor("#FFF6E5")

    ax1.plot(weeks, ov["avg5"].values, marker="o", linewidth=1, label="Итоговая /5")
    ax1.plot(weeks, ov_roll.values, linestyle="--", linewidth=1, label="Итоговая /5 (скользящее)")
    ax1.set_ylim(0, 5)
    ax1.set_ylabel("Итоговая /5", color="#262D36")
    ax1.tick_params(axis='y', labelcolor="#262D36")
    ax1.tick_params(axis='x', labelrotation=45, labelcolor="#262D36")

    ax2 = ax1.twinx()
    ax2.plot(weeks, npv["nps_value"].values, marker="s", linewidth=1, label="NPS, %", alpha=0.8)
    ax2.plot(weeks, nps_roll.values, linestyle="--", linewidth=1, label="NPS, % (скользящее)", alpha=0.8)
    ax2.set_ylim(-100, 100)
    ax2.set_ylabel("NPS, %", color="#262D36")
    ax2.tick_params(axis='y', labelcolor="#262D36")

    ax1.set_title(
        f"Анкеты: динамика итоговой оценки и NPS (последние 12 недель, по состоянию на {human_date(as_of)})",
        color="#262D36",
        fontsize=10,
    )

    lines, labels = [], []
    for ax in (ax1, ax2):
        l, lab = ax.get_legend_handles_labels()
        lines += l
        labels += lab
    ax1.legend(lines, labels, loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(path_png)
    plt.close()
    return path_png


# =====================================================
# Email helpers
# =====================================================

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


# =====================================================
# Main
# =====================================================

def main():
    # 1) Забираем последний Report_*.xlsx с Диска
    file_id, fname, fdate = latest_report_from_drive()
    data = drive_download(file_id)

    # 2) Читаем excel с анкетами
    xls = pd.ExcelFile(io.BytesIO(data))
    if "Оценки гостей" in xls.sheet_names:
        raw = pd.read_excel(xls, sheet_name="Оценки гостей")
    else:
        raw = pd.read_excel(xls, sheet_name="Reviews")

    # 3) Парсим и агрегируем неделю (новая логика из surveys_core)
    norm_df, agg_week = parse_and_aggregate_weekly(raw)

    # 4) Записываем неделю в history (без дублей)
    added = upsert_week(agg_week)
    print(f"[INFO] upsert_week(): добавлено строк недели: {added}")

    # 5) Читаем всю историю обратно
    hist = gs_get_df(SURVEYS_TAB, "A:I")
    if hist.empty:
        raise RuntimeError("surveys_history пуст — нечего анализировать.")

    # 6) Определяем ключевые даты
    wk_key   = str(agg_week["week_key"].iloc[0])
    w_start  = iso_week_monday(wk_key)
    w_end    = w_start + dt.timedelta(days=6)

    prev_start = w_start - dt.timedelta(days=7)
    prev_end   = prev_start + dt.timedelta(days=6)

    # Окно последних 4 недель ДО этой
    l4_start = w_start - dt.timedelta(days=28)
    l4_end   = w_start - dt.timedelta(days=1)

    # Периоды (Неделя / Месяц-to-date / Квартал-to-date / Год-to-date)
    ranges = period_ranges_for_week(w_start)

    # Исторические границы "Итого"
    hist_mondays = hist["week_key"].map(lambda k: iso_week_monday(str(k)))
    all_start = hist_mondays.min()
    all_end   = hist_mondays.max() + dt.timedelta(days=6)

    # 7) Считаем агрегаты по периодам (текущие)
    W = surveys_aggregate_period(hist, w_start, w_end)
    M = surveys_aggregate_period(hist, ranges["mtd"]["start"], ranges["mtd"]["end"])
    Q = surveys_aggregate_period(hist, ranges["qtd"]["start"], ranges["qtd"]["end"])
    Y = surveys_aggregate_period(hist, ranges["ytd"]["start"], ranges["ytd"]["end"])
    T = surveys_aggregate_period(hist, all_start, all_end)

    Prev = surveys_aggregate_period(hist, prev_start, prev_end)
    L4   = surveys_aggregate_period(hist, l4_start, l4_end)

    # 8) Подготовка подписей периодов (в человеко-читаемом виде)

    # неделя "13–19 окт 2025" → колоночный вариант: "13–19 окт 2025 г."
    week_lbl_no_g   = week_label(w_start, w_end)
    week_col_label  = add_year_suffix(week_lbl_no_g)
    week_label_full = "Неделя " + week_col_label  # "Неделя 13–19 окт 2025 г."

    # месяц → "Октябрь 2025 г."
    month_label_full   = pretty_month_label(w_start)
    month_col_label    = month_label_full

    # квартал → "IV квартал 2025 г."
    quarter_label_full = pretty_quarter_label(w_start)
    quarter_col_label  = quarter_label_full

    # год → "2025 г."
    year_label_full    = pretty_year_label(w_start)
    year_col_label     = year_label_full

    total_label_full   = "Итого"
    total_col_label    = "Итого"

    # 9) Готовим окна для сравнения с прошлым годом
    # текущие интервалы:
    mtd_start = ranges["mtd"]["start"]
    mtd_end   = ranges["mtd"]["end"]

    qtd_start = ranges["qtd"]["start"]
    qtd_end   = ranges["qtd"]["end"]

    ytd_start = ranges["ytd"]["start"]
    ytd_end   = ranges["ytd"]["end"]

    # интервалы за прошлый год (та же "точка во времени", но -1 год):
    py_w_start     = shift_year_back_safe(w_start)
    py_w_end       = shift_year_back_safe(w_end)
    py_mtd_start   = shift_year_back_safe(mtd_start)
    py_mtd_end     = shift_year_back_safe(mtd_end)
    py_qtd_start   = shift_year_back_safe(qtd_start)
    py_qtd_end     = shift_year_back_safe(qtd_end)
    py_ytd_start   = shift_year_back_safe(ytd_start)
    py_ytd_end     = shift_year_back_safe(ytd_end)

    W_prevY = surveys_aggregate_period(hist, py_w_start,   py_w_end)
    M_prevY = surveys_aggregate_period(hist, py_mtd_start, py_mtd_end)
    Q_prevY = surveys_aggregate_period(hist, py_qtd_start, py_qtd_end)
    Y_prevY = surveys_aggregate_period(hist, py_ytd_start, py_ytd_end)

    def _totals_row(agg: dict) -> dict:
        return {
            "surveys_total": agg["totals"]["surveys_total"],
            "overall5":      agg["totals"]["overall5"],
            "nps":           agg["totals"]["nps"],
        }

    yoy_periods = [
        {
            "label": week_label_full,
            "cur":  _totals_row(W),
            "prev": _totals_row(W_prevY),
        },
        {
            "label": month_label_full,
            "cur":  _totals_row(M),
            "prev": _totals_row(M_prevY),
        },
        {
            "label": quarter_label_full,
            "cur":  _totals_row(Q),
            "prev": _totals_row(Q_prevY),
        },
        {
            "label": year_label_full,
            "cur":  _totals_row(Y),
            "prev": _totals_row(Y_prevY),
        },
    ]

    # 10) Формируем HTML письма

    head_html = header_block(
        obj_name="ARTSTUDIO Nevsky",
        week_label_full=week_label_full,
        month_label_full=month_label_full,
        quarter_label_full=quarter_label_full,
        year_label_full=year_label_full,
        total_label_full=total_label_full,
        W=W, M=M, Q=Q, Y=Y, T=T,
    )

    table_html = table_params_block(
        W["by_param"], M["by_param"], Q["by_param"], Y["by_param"], T["by_param"],
        week_col_label=week_col_label,
        month_col_label=month_col_label,
        quarter_col_label=quarter_col_label,
        year_col_label=year_col_label,
        total_col_label=total_col_label,
    )

    trends_html = trends_block(W, Prev, L4)

    yoy_html = yoy_block(yoy_periods)

    footer_html = footnote_block(all_start, w_end)

    html_body = (
        head_html
        + table_html
        + trends_html
        + yoy_html
        + footer_html
    )

    # 11) Готовим графики
    charts = []
    p1 = "/tmp/surveys_radar.png"
    p2 = "/tmp/surveys_heatmap.png"
    p3 = "/tmp/surveys_trends.png"

    if plot_radar_params(W["by_param"], M["by_param"], p1): charts.append(p1)
    if plot_params_heatmap(hist, p2):                       charts.append(p2)
    if plot_overall_nps_trends(hist, p3, as_of=w_end):      charts.append(p3)

    # 12) Subject письма
    subject = f"ARTSTUDIO Nevsky. Анкеты TL: Marketing — неделя {week_col_label}"

    # 13) Отправляем письмо
    send_email(subject, html_body, attachments=charts)


if __name__ == "__main__":
    main()
