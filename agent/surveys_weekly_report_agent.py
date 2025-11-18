# agent/surveys_weekly_report_agent.py

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
from .connectors import build_credentials_from_env, get_drive_client, get_sheets_client

# headless matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- imports из соседних модулей ---
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


# =========================================
# ENV, creds, Google API clients
# =========================================

CREDS = build_credentials_from_env()
DRIVE = get_drive_client(CREDS)
SHEETS = get_sheets_client(CREDS).spreadsheets()

DRIVE_FOLDER_ID   = os.environ["DRIVE_FOLDER_ID"]
HISTORY_SHEET_ID  = os.environ["SHEETS_HISTORY_ID"]

RECIPIENTS        = [e.strip() for e in os.environ.get("RECIPIENTS","").split(",") if e.strip()]
SMTP_USER         = os.environ.get("SMTP_USER")
SMTP_PASS         = os.environ.get("SMTP_PASS")
SMTP_FROM         = os.environ.get("SMTP_FROM", SMTP_USER or "")
SMTP_HOST         = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT         = int(os.environ.get("SMTP_PORT", 587))


# =========================================
# Цвета и шрифты
# =========================================

COLOR_TEXT_MAIN   = "#262D36"  # основной тёмный
COLOR_BG_HEADER   = "#FFF6E5"  # мягкий фон
COLOR_BORDER      = "#C49A5F"  # границы/рамки
COLOR_POSITIVE    = "#FFCC00"  # позитив ▲
COLOR_NEGATIVE    = "#EF7D17"  # негатив ▼

FONT_STACK = "'Helvetica Neue','Segoe UI',Arial,sans-serif"


# =========================================
# Drive helpers
# =========================================

WEEKLY_RE = re.compile(r"^Report_(\d{2})-(\d{2})-(\d{4})\.xlsx$", re.IGNORECASE)

def latest_report_from_drive():
    """
    Находим самый свежий Report_DD-MM-YYYY.xlsx
    Возвращаем (file_id, filename, date_obj).
    """
    res = DRIVE.files().list(
        q=(
            f"'{DRIVE_FOLDER_ID}' in parents and "
            "mimeType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' "
            "and trashed=false"
        ),
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
        raise RuntimeError("В папке нет файлов формата Report_dd-mm-yyyy.xlsx.")

    items.sort(key=lambda t: t[2], reverse=True)
    return items[0]


def drive_download(file_id: str) -> bytes:
    req = DRIVE.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
    return buf.getvalue()


# =========================================
# Sheets helpers
# =========================================

def ensure_tab(spreadsheet_id: str, tab_name: str, header: list[str]):
    """
    Убедиться, что лист есть, если нет — создать и задать шапку.
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
    Считать диапазон с Google Sheets → DataFrame
    """
    try:
        res = SHEETS.values().get(
            spreadsheetId=HISTORY_SHEET_ID,
            range=f"{tab}!{a1}",
        ).execute()
        vals = res.get("values", [])
        return pd.DataFrame(vals[1:], columns=vals[0]) if len(vals) > 1 else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


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
    agg_week → строки для Sheets
    """
    out = []
    for _, r in df.iterrows():
        out.append([
            str(r["week_key"]),
            str(r["param"]),
            (int(r["surveys_total"]) if not pd.isna(r["surveys_total"]) else 0),
            (int(r["answered"])      if not pd.isna(r["answered"])      else 0),
            (None if pd.isna(r["avg5"]) else float(r["avg5"])),
            (None if "promoters"    not in r or pd.isna(r["promoters"])    else int(r["promoters"])),
            (None if "detractors"   not in r or pd.isna(r["detractors"])   else int(r["detractors"])),
            (None if "nps_answers"  not in r or pd.isna(r["nps_answers"])  else int(r["nps_answers"])),
            (None if "nps_value"    not in r or pd.isna(r["nps_value"])    else float(r["nps_value"])),
        ])
    return out

def upsert_week(agg_week_df: pd.DataFrame) -> int:
    """
    Обновить данные за неделю в листе surveys_history:
    - удалить старые строки с этим week_key
    - записать всё остальное + новые строки
    """
    if agg_week_df.empty:
        return 0

    ensure_tab(HISTORY_SHEET_ID, SURVEYS_TAB, SURVEYS_HEADER)

    wk = str(agg_week_df["week_key"].iloc[0])

    hist = gs_get_df(SURVEYS_TAB, "A:I")
    keep = (
        hist[hist.get("week_key", "") != wk]
        if not hist.empty else
        pd.DataFrame(columns=SURVEYS_HEADER)
    )

    # очистим тело
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


# =========================================
# Даты, подписи
# =========================================

RU_MONTH_SHORT = {
    1:"янв",2:"фев",3:"мар",4:"апр",5:"май",6:"июн",
    7:"июл",8:"авг",9:"сен",10:"окт",11:"ноя",12:"дек",
}

def human_date(d: date) -> str:
    # "15 янв 2025 г."
    return f"{d.day} {RU_MONTH_SHORT[d.month]} {d.year} г."

def add_year_suffix(label_no_g: str) -> str:
    # "13–19 окт 2025" -> "13–19 окт 2025 г."
    return f"{label_no_g} г."

def pretty_month_label(d: date) -> str:
    ml = month_label(d)  # "октябрь 2025"
    ml_cap = ml[0].upper() + ml[1:]
    return f"{ml_cap} г."

def pretty_quarter_label(d: date) -> str:
    ql = quarter_label(d).replace("кв.", "квартал")  # "IV квартал 2025"
    return f"{ql} г."

def pretty_year_label(d: date) -> str:
    return f"{year_label(d)} г."

def shift_year_back_safe(d: date) -> date:
    try:
        return d.replace(year=d.year-1)
    except ValueError:
        return d - dt.timedelta(days=365)

def week_short_label_for_key(week_key: str, ref_year: int|None=None) -> str:
    """
    Для теплокарты: короткая подпись недели вида '13–19 окт' или '13–19 окт 2025'
    """
    start = iso_week_monday(str(week_key))
    end = start + dt.timedelta(days=6)

    same_year = (start.year == end.year)
    show_year = (not same_year) or (ref_year is not None and ref_year != start.year)

    if start.month == end.month and start.year == end.year:
        label_core = f"{start.day}–{end.day} {RU_MONTH_SHORT[start.month]}"
    else:
        left = f"{start.day} {RU_MONTH_SHORT[start.month]}"
        right = f"{end.day} {RU_MONTH_SHORT[end.month]}"
        label_core = f"{left}–{right}"

    if show_year:
        label_core += f" {end.year}"

    return label_core


# =========================================
# Агрегация периодов из истории
# =========================================

def _to_num_series(s: pd.Series|None) -> pd.Series:
    if s is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(
        s.astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )

def _weighted_avg(values: pd.Series, weights: pd.Series) -> float|None:
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
    Возвращает:
    {
      "by_param": df[param, surveys_total, answered, avg5,
                     promoters, detractors, nps_answers, nps_value],
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
                    (promoters_sum / nps_answers_sum - detractors_sum / nps_answers_sum) * 100.0,
                    2
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


# =========================================
# Форматирование чисел и дельт
# =========================================

def _to_float_or_none(x):
    try:
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None

def fmt_avg5(x):
    v = _to_float_or_none(x)
    if v is None:
        return "—"
    return f"{v:.1f}"

def fmt_int(x):
    v = _to_float_or_none(x)
    if v is None:
        return "—"
    return str(int(round(v)))

def fmt_nps(x):
    v = _to_float_or_none(x)
    if v is None:
        return "—"
    return f"{v:.1f}%"

def signed_delta(val_diff, suffix=""):
    """
    "+0.3", "−0.4 п.п.", "0.0"
    """
    if val_diff is None:
        return "0.0" + suffix
    d = round(val_diff, 1)
    if abs(d) < 0.05:
        return "0.0" + suffix
    if d > 0:
        return f"+{d:.1f}{suffix}"
    else:
        # U+2212 minus
        return f"−{abs(d):.1f}{suffix}"

def delta_text_relative(cur, base, score_suffix="", nps_suffix=" п.п.", for_nps=False):
    """
    Возвращает кортеж (phrase, arrow_html_or_None).
    phrase, например:
      "выше среднего уровня (+0.3)"
      "ниже среднего уровня (−0.4)"
      "на уровне"
    """
    c = _to_float_or_none(cur)
    b = _to_float_or_none(base)
    if c is None or b is None:
        return None, None

    diff = round(c - b, 1)
    thr = 2.0 if for_nps else 0.1
    if abs(diff) < thr:
        return "на уровне", None

    if diff > 0:
        txt = "выше среднего уровня"
        arrow = f"<span style='color:{COLOR_POSITIVE};font-weight:bold;'>▲</span>"
        suff = nps_suffix if for_nps else score_suffix
        delta_part = signed_delta(diff, suff)
        return f"{txt} ({delta_part})", arrow
    else:
        txt = "ниже среднего уровня"
        arrow = f"<span style='color:{COLOR_NEGATIVE};font-weight:bold;'>▼</span>"
        suff = nps_suffix if for_nps else score_suffix
        delta_part = signed_delta(diff, suff)
        return f"{txt} ({delta_part})", arrow


# =========================================
# Названия параметров
# =========================================

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


# =========================================
# Вспомогательные выборки по параметрам
# =========================================

def df_param_map(df: pd.DataFrame) -> dict[str, dict]:
    """
    превращает by_param df → {param: row_as_dict}
    """
    mp = {}
    if df is None or df.empty:
        return mp
    for _, r in df.iterrows():
        mp[str(r["param"])] = r.to_dict()
    return mp

def extract_param_deltas(cur_map, base_map,
                         rep_min_cur:int,
                         rep_min_base:int,
                         delta_thr:float):
    """
    Сравниваем параметры cur_map vs base_map.
    Возвращаем:
      ups   = [(param_name, cur_avg, diff), ...]  diff>0
      downs = [(param_name, cur_avg, diff), ...] diff<0
    Фильтр:
      answered_cur >= rep_min_cur
      answered_base >= rep_min_base
      abs(diff) >= delta_thr
      param != 'nps'
    """
    ups, downs = [], []
    for p, wrow in cur_map.items():
        if p == "nps":
            continue
        cur_avg = _to_float_or_none(wrow.get("avg5"))
        cur_ans = _to_float_or_none(wrow.get("answered"))

        brow  = base_map.get(p, {})
        base_avg = _to_float_or_none(brow.get("avg5"))
        base_ans = _to_float_or_none(brow.get("answered"))

        if cur_avg is None or base_avg is None:
            continue
        if cur_ans is None or base_ans is None:
            continue
        if cur_ans < rep_min_cur or base_ans < rep_min_base:
            continue

        diff = round(cur_avg - base_avg, 1)
        if abs(diff) < delta_thr:
            continue

        if diff > 0:
            ups.append((p, cur_avg, diff))
        else:
            downs.append((p, cur_avg, diff))

    ups.sort(key=lambda x: x[2], reverse=True)
    downs.sort(key=lambda x: x[2])
    return ups, downs

def extract_problem_params_for_alert(cur_map, base_map):
    """
    Для тревожного блока (⚠):
    Возвращает список [(param_name, cur_avg, cur_ans, diff_vs_base)]
    Параметр попадает, если:
      - cur_ans >=3
      И ( cur_avg <4.0
          ИЛИ (base_avg - cur_avg)>=0.3 )
    """
    out = []
    for p, wrow in cur_map.items():
        if p == "nps":
            continue
        cur_avg = _to_float_or_none(wrow.get("avg5"))
        cur_ans = _to_float_or_none(wrow.get("answered"))
        if cur_avg is None or cur_ans is None:
            continue
        if cur_ans < 3:
            continue

        base_avg = _to_float_or_none(base_map.get(p,{}).get("avg5"))

        flag_low_abs = (cur_avg < 4.0)
        flag_drop = False
        if base_avg is not None:
            if (base_avg - cur_avg) >= 0.3:
                flag_drop = True

        if flag_low_abs or flag_drop:
            diff_vs_base = None
            if base_avg is not None:
                diff_vs_base = round(cur_avg - base_avg, 1)
            out.append((p, cur_avg, int(round(cur_ans)), diff_vs_base))

    out.sort(key=lambda x: x[1])
    return out


# =========================================
# Шапка письма
# =========================================

def summarize_period_influence_text(
    W,
    P,
    period_word_gen: str,
    period_word_acc: str,
    is_total: bool
):
    """
    Возвращает:
    - contrib_line_html
    - compare_html
    - params_block_html
    - (Pavg, Wavg) для стрелки в заголовке

    period_word_gen: "месяца" / "квартала" / "года" / "периода"
    period_word_acc: "месяц" / "квартал" / "год" / "период" (винительный падеж)
    is_total: True для Итого (используем "общего исторического уровня")
    """

    Wtot = _to_float_or_none(W["totals"]["surveys_total"]) or 0
    Ptot = _to_float_or_none(P["totals"]["surveys_total"]) or 0

    Wavg = _to_float_or_none(W["totals"]["overall5"])
    Pavg = _to_float_or_none(P["totals"]["overall5"])

    Wnps = _to_float_or_none(W["totals"]["nps"])
    Pnps = _to_float_or_none(P["totals"]["nps"])

    # ----- 1. вклад недели -----
    if Ptot <= 0:
        contrib_line_html = (
            "Данных по периоду пока недостаточно для сравнения."
        )
    else:
        share = 100.0 * (Wtot / Ptot) if Ptot > 0 else 0.0
        share_rounded = int(round(share))
        if Ptot == Wtot and Ptot > 0:
            if is_total:
                contrib_line_html = (
                    "Показатели этого сводного периода сейчас полностью сформированы "
                    "данными текущей недели."
                )
            else:
                contrib_line_html = (
                    f"Показатели {period_word_gen} сейчас полностью определяются текущей неделей "
                    f"(<b>100%</b> анкет {period_word_gen} приходятся на текущую неделю)."
                )
        elif share < 5:
            contrib_line_html = (
                "Текущая неделя дала менее <b>5%</b> всех анкет за "
                f"{period_word_acc}; влияние на общий уровень метрик минимальное."
            )
        else:
            contrib_line_html = (
                "Текущая неделя дала около "
                f"<b>{share_rounded}%</b> всех анкет за {period_word_acc}."
            )

    # ----- 2. сравнение средней оценки и NPS -----
    def compare_line_html():
        score_phrase, _score_arrow = delta_text_relative(
            Wavg, Pavg, score_suffix="", for_nps=False
        )
        nps_phrase, _nps_arrow = delta_text_relative(
            Wnps, Pnps, nps_suffix=" п.п.", for_nps=True
        )

        base_word = period_word_gen if not is_total else "периода"

        if score_phrase is None and nps_phrase is None:
            return (
                f"<b>Средняя оценка и NPS этой недели</b> соответствуют текущему уровню {base_word}."
            )

        parts = []

        if score_phrase is not None:
            if "на уровне" in score_phrase:
                parts.append(
                    f"<b>Средняя оценка этой недели</b> на уровне {base_word}."
                )
            else:
                # "выше среднего уровня (+0.3)" → "...выше среднего уровня месяца (+0.3)"
                parts.append(
                    "<b>Средняя оценка этой недели</b> " +
                    score_phrase.replace("среднего уровня",
                                         f"среднего уровня {base_word}") +
                    "."
                )

        if nps_phrase is not None:
            if "на уровне" in nps_phrase:
                parts.append(
                    f"<b>NPS этой недели</b> на уровне {base_word}."
                )
            else:
                parts.append(
                    "<b>NPS этой недели</b> " +
                    nps_phrase.replace("среднего уровня",
                                       f"среднего уровня {base_word}") +
                    "."
                )

        if not parts:
            return (
                f"<b>Средняя оценка и NPS этой недели</b> соответствуют текущему уровню {base_word}."
            )
        return " ".join(parts)

    compare_html = compare_line_html()

    # ----- 3. существенные отклонения по параметрам -----
    Wmap = df_param_map(W["by_param"])
    Pmap = df_param_map(P["by_param"])

    ups, downs = extract_param_deltas(
        cur_map=Wmap,
        base_map=Pmap,
        rep_min_cur=2,
        rep_min_base=4,
        delta_thr=0.3,
    )

    downs_txt = []
    for p, cur_avg, diff in downs:
        title = PARAM_TITLES.get(p, p)
        # diff отрицателен → signed_delta(diff) уже с минусом
        downs_txt.append(f"«{title}» ({signed_delta(diff)})")

    ups_txt = []
    for p, cur_avg, diff in ups:
        title = PARAM_TITLES.get(p, p)
        ups_txt.append(f"«{title}» ({signed_delta(diff)})")

    if is_total:
        base_low  = "Ниже общего исторического уровня"
        base_high = "Выше общего исторического уровня"
    else:
        base_low  = f"Ниже текущего уровня {period_word_gen}"
        base_high = f"Выше текущего уровня {period_word_gen}"

    if not downs_txt and not ups_txt:
        params_block_html = (
            "По отдельным параметрам существенных отклонений за неделю не зафиксировано."
        )
    else:
        lines = [ "<b>Существенные отклонения этой недели по параметрам:</b>" ]
        if downs_txt:
            lines.append(
                f"{base_low} — " + ", ".join(downs_txt) + "."
            )
        if ups_txt:
            lines.append(
                f"{base_high} — " + ", ".join(ups_txt) + "."
            )
        params_block_html = "<br>".join(lines)

    return contrib_line_html, compare_html, params_block_html, Pavg, Wavg


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
    Шапка письма.
    """

    def impact_arrow(period_avg, week_avg):
        """
        Возвращает '' или стрелку ▲/▼ цветом бренда,
        если неделя тянет период вверх или вниз по итоговой оценке.
        """
        pa = _to_float_or_none(period_avg)
        wa = _to_float_or_none(week_avg)
        if pa is None or wa is None:
            return ""
        diff = round(wa - pa, 1)
        thr = 0.1
        if abs(diff) < thr:
            return ""
        if diff > 0:
            return f"<span style='color:{COLOR_POSITIVE};font-weight:bold;'>▲</span> "
        else:
            return f"<span style='color:{COLOR_NEGATIVE};font-weight:bold;'>▼</span> "

    # блок для недели (без сравнений, просто факты)
    week_block = (
        f"<div style='margin-bottom:16px;color:{COLOR_TEXT_MAIN};font-family:{FONT_STACK};'>"
        f"<div style='font-weight:bold;'>{'Итоги ' + week_label_full}</div>"
        f"<div>"
        f"<b>Анкет:</b> {fmt_int(W['totals']['surveys_total'])}. "
        f"<b>Итоговая оценка:</b> {fmt_avg5(W['totals']['overall5'])}. "
        f"<b>NPS:</b> {fmt_nps(W['totals']['nps'])}"
        f"</div>"
        f"</div>"
    )

    # блоки для периода: месяц / квартал / год / итого
    def period_block(period_title, P, period_word_gen, period_word_acc, is_total=False):
        contrib_line_html, compare_html, params_block_html, Pavg, Wavg = summarize_period_influence_text(
            W, P,
            period_word_gen=period_word_gen,
            period_word_acc=period_word_acc,
            is_total=is_total
        )
        arrow = impact_arrow(Pavg, Wavg)

        return (
            f"<div style='margin-bottom:16px;color:{COLOR_TEXT_MAIN};font-family:{FONT_STACK};'>"
            f"<div style='font-weight:bold;'>{arrow}{period_title}</div>"
            f"<div>"
            f"<b>Анкет:</b> {fmt_int(P['totals']['surveys_total'])}. "
            f"<b>Итоговая оценка:</b> {fmt_avg5(P['totals']['overall5'])}. "
            f"<b>NPS:</b> {fmt_nps(P['totals']['nps'])}"
            f"</div>"
            f"<div>{contrib_line_html}</div>"
            f"<div>{compare_html}</div>"
            f"<div>{params_block_html}</div>"
            f"</div>"
        )

    # генитив / винительный для периодов
    # месяц:   "месяца"   / "месяц"
    # квартал: "квартала" / "квартал"
    # год:     "года"     / "год"
    # итого:   "периода"  / "период"
    month_block   = period_block(month_label_full,   M, "месяца",   "месяц",   is_total=False)
    quarter_block = period_block(quarter_label_full, Q, "квартала", "квартал", is_total=False)
    year_block    = period_block(year_label_full,    Y, "года",     "год",     is_total=False)
    total_block   = period_block(total_label_full,   T, "периода",  "период",  is_total=True)

    title_html = (
        f"<h2 style='margin-bottom:16px;color:{COLOR_TEXT_MAIN};font-family:{FONT_STACK};'>"
        f"{obj_name} — анкеты гостей TL: Marketing"
        f"</h2>"
    )

    return title_html + week_block + month_block + quarter_block + year_block + total_block


# =========================================
# Таблица параметров качества
# =========================================

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
    Wm = df_param_map(W_df)
    Mm = df_param_map(M_df)
    Qm = df_param_map(Q_df)
    Ym = df_param_map(Y_df)
    Tm = df_param_map(T_df)

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
                "<td style='text-align:right;'>—</td>"
                "<td style='text-align:right;'>—</td>"
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
            f"<td style='white-space:nowrap;color:{COLOR_TEXT_MAIN};font-family:{FONT_STACK};'><b>{title}</b></td>"
            + cell(Wm, p)
            + cell(Mm, p)
            + cell(Qm, p)
            + cell(Ym, p)
            + cell(Tm, p)
            + "</tr>"
        )

    html = f"""
    <h3 style="color:{COLOR_TEXT_MAIN};margin-top:24px;font-family:{FONT_STACK};">
      Ключевые показатели по параметрам качества
    </h3>
    <table border='1' cellspacing='0' cellpadding='6'
           style="border-collapse:collapse;border-color:{COLOR_BORDER};color:{COLOR_TEXT_MAIN};font-size:14px;font-family:{FONT_STACK};">
      <tr style="background-color:{COLOR_BG_HEADER};">
        <th rowspan="2" style="border:1px solid {COLOR_BORDER};text-align:left;">Параметр</th>
        <th colspan="2" style="border:1px solid {COLOR_BORDER};text-align:center;">{week_col_label}</th>
        <th colspan="2" style="border:1px solid {COLOR_BORDER};text-align:center;">{month_col_label}</th>
        <th colspan="2" style="border:1px solid {COLOR_BORDER};text-align:center;">{quarter_col_label}</th>
        <th colspan="2" style="border:1px solid {COLOR_BORDER};text-align:center;">{year_col_label}</th>
        <th colspan="2" style="border:1px solid {COLOR_BORDER};text-align:center;">{total_col_label}</th>
      </tr>
      <tr style="background-color:{COLOR_BG_HEADER};">
        <th style="border:1px solid {COLOR_BORDER};text-align:right;">Средняя оценка</th>
        <th style="border:1px solid {COLOR_BORDER};text-align:right;">Ответы</th>
        <th style="border:1px solid {COLOR_BORDER};text-align:right;">Средняя оценка</th>
        <th style="border:1px solid {COLOR_BORDER};text-align:right;">Ответы</th>
        <th style="border:1px solid {COLOR_BORDER};text-align:right;">Средняя оценка</th>
        <th style="border:1px solid {COLOR_BORDER};text-align:right;">Ответы</th>
        <th style="border:1px solid {COLOR_BORDER};text-align:right;">Средняя оценка</th>
        <th style="border:1px solid {COLOR_BORDER};text-align:right;">Ответы</th>
        <th style="border:1px solid {COLOR_BORDER};text-align:right;">Средняя оценка</th>
        <th style="border:1px solid {COLOR_BORDER};text-align:right;">Ответы</th>
      </tr>
      {''.join(rows_html)}
    </table>
    """
    return html


# =========================================
# Блок динамики и точек внимания
# =========================================

def trends_block(W, Prev, L4):
    """
    Формируем 4 подпункта:
    1. Текущая неделя по сравнению с предыдущей
    2. Текущая неделя относительно среднего уровня последних недель
    3. Ключевые сигналы этой недели
    4. ⚠ Тревожные точки
    """

    def to_map(agg):
        return df_param_map(agg["by_param"])

    Wmap    = to_map(W)
    Prevmap = to_map(Prev)
    L4map   = to_map(L4)

    # ------- 1. сравнение с предыдущей неделей -------
    cur_overall  = W["totals"]["overall5"]
    prev_overall = Prev["totals"]["overall5"]
    cur_nps      = W["totals"]["nps"]
    prev_nps     = Prev["totals"]["nps"]

    diff_over_prev = None
    if _to_float_or_none(cur_overall) is not None and _to_float_or_none(prev_overall) is not None:
        diff_over_prev = round(_to_float_or_none(cur_overall) - _to_float_or_none(prev_overall), 1)

    diff_nps_prev = None
    if _to_float_or_none(cur_nps) is not None and _to_float_or_none(prev_nps) is not None:
        diff_nps_prev = round(_to_float_or_none(cur_nps) - _to_float_or_none(prev_nps), 1)

    ups_prev, downs_prev = extract_param_deltas(
        cur_map=Wmap,
        base_map=Prevmap,
        rep_min_cur=2,
        rep_min_base=2,
        delta_thr=0.3,
    )

    def list_params_diff(items):
        out = []
        for p, cur_avg, diff in items:
            title = PARAM_TITLES.get(p, p)
            out.append(f"«{title}» ({signed_delta(diff)})")
        return out

    ups_prev_txt   = list_params_diff(ups_prev)
    downs_prev_txt = list_params_diff(downs_prev)

    if (_to_float_or_none(cur_overall) is None) and (_to_float_or_none(cur_nps) is None):
        line_prev_header = (
            "Данных для сравнения с предыдущей неделей пока недостаточно."
        )
    else:
        parts1 = []
        if _to_float_or_none(cur_overall) is not None:
            if diff_over_prev is None or abs(diff_over_prev) < 0.05:
                parts1.append(
                    f"<b>Итоговая оценка:</b> {fmt_avg5(cur_overall)} (на уровне прошлой недели)."
                )
            else:
                parts1.append(
                    f"<b>Итоговая оценка:</b> {fmt_avg5(cur_overall)} "
                    f"({signed_delta(diff_over_prev)} к прошлой неделе)."
                )
        if _to_float_or_none(cur_nps) is not None:
            if diff_nps_prev is None or abs(diff_nps_prev) < 2.0:
                parts1.append(
                    f"<b>NPS:</b> {fmt_nps(cur_nps)} (на уровне прошлой недели)."
                )
            else:
                parts1.append(
                    f"<b>NPS:</b> {fmt_nps(cur_nps)} "
                    f"({signed_delta(diff_nps_prev,' п.п.')} к прошлой неделе)."
                )
        line_prev_header = " ".join(parts1)

    detail_prev_lines = []
    if ups_prev_txt:
        detail_prev_lines.append(
            "<b>Улучшилось по сравнению с прошлой неделей:</b> " +
            ", ".join(ups_prev_txt) + "."
        )
    if downs_prev_txt:
        detail_prev_lines.append(
            "<b>Ухудшилось по сравнению с прошлой неделей:</b> " +
            ", ".join(downs_prev_txt) + "."
        )
    if not detail_prev_lines:
        detail_prev_lines.append(
            "Существенных изменений по отдельным параметрам относительно прошлой недели не отмечено."
        )

    block_prev = (
        f"<p style='margin-top:16px;color:{COLOR_TEXT_MAIN};font-family:{FONT_STACK};'>"
        f"<b>Текущая неделя по сравнению с предыдущей:</b><br>"
        f"{line_prev_header}<br>"
        + "<br>".join(detail_prev_lines) +
        "</p>"
    )

    # ------- 2. сравнение с последними неделями (L4) -------
    cur_overall_L4  = W["totals"]["overall5"]
    base_overall_L4 = L4["totals"]["overall5"]
    cur_nps_L4      = W["totals"]["nps"]
    base_nps_L4     = L4["totals"]["nps"]

    diff_over_L4 = None
    if _to_float_or_none(cur_overall_L4) is not None and _to_float_or_none(base_overall_L4) is not None:
        diff_over_L4 = round(_to_float_or_none(cur_overall_L4) - _to_float_or_none(base_overall_L4), 1)

    diff_nps_L4 = None
    if _to_float_or_none(cur_nps_L4) is not None and _to_float_or_none(base_nps_L4) is not None:
        diff_nps_L4 = round(_to_float_or_none(cur_nps_L4) - _to_float_or_none(base_nps_L4), 1)

    ups_L4, downs_L4 = extract_param_deltas(
        cur_map=Wmap,
        base_map=L4map,
        rep_min_cur=2,
        rep_min_base=4,
        delta_thr=0.3,
    )

    ups_L4_txt   = list_params_diff(ups_L4)
    downs_L4_txt = list_params_diff(downs_L4)

    if (_to_float_or_none(cur_overall_L4) is None) and (_to_float_or_none(cur_nps_L4) is None):
        line_L4_header = (
            "Данных для оценки относительно среднего уровня последних недель пока недостаточно."
        )
    else:
        parts2 = []
        if _to_float_or_none(cur_overall_L4) is not None:
            if diff_over_L4 is None or abs(diff_over_L4) < 0.1:
                parts2.append(
                    f"<b>Итоговая оценка этой недели:</b> {fmt_avg5(cur_overall_L4)} "
                    "(на уровне средних последних недель)."
                )
            else:
                parts2.append(
                    f"<b>Итоговая оценка этой недели:</b> {fmt_avg5(cur_overall_L4)} "
                    f"({signed_delta(diff_over_L4)} к среднему последних недель)."
                )
        if _to_float_or_none(cur_nps_L4) is not None:
            if diff_nps_L4 is None or abs(diff_nps_L4) < 2.0:
                parts2.append(
                    f"<b>NPS этой недели:</b> {fmt_nps(cur_nps_L4)} "
                    "(на уровне средних последних недель)."
                )
            else:
                parts2.append(
                    f"<b>NPS этой недели:</b> {fmt_nps(cur_nps_L4)} "
                    f"({signed_delta(diff_nps_L4,' п.п.')} к среднему последних недель)."
                )
        line_L4_header = " ".join(parts2)

    detail_L4_lines = []
    if ups_L4_txt:
        detail_L4_lines.append(
            "<b>Выше обычного уровня:</b> " + ", ".join(ups_L4_txt) + "."
        )
    if downs_L4_txt:
        detail_L4_lines.append(
            "<b>Ниже обычного уровня:</b> " + ", ".join(downs_L4_txt) + "."
        )
    if not detail_L4_lines:
        detail_L4_lines.append(
            "По ключевым параметрам отклонений от обычного уровня не зафиксировано."
        )

    block_L4 = (
        f"<p style='margin-top:12px;color:{COLOR_TEXT_MAIN};font-family:{FONT_STACK};'>"
        f"<b>Текущая неделя относительно среднего уровня последних недель:</b><br>"
        f"{line_L4_header}<br>"
        + "<br>".join(detail_L4_lines) +
        "</p>"
    )

    # ------- 3. ключевые сигналы недели -------
    good_lines = []
    for p, cur_avg, diff in ups_L4:
        title = PARAM_TITLES.get(p, p)
        good_lines.append(f"«{title}» ({cur_avg:.1f})")

    risk_lines = []
    for p, cur_avg, diff in downs_L4:
        title = PARAM_TITLES.get(p, p)
        risk_lines.append(f"«{title}» ({cur_avg:.1f})")

    key_signal_lines = []
    if good_lines:
        key_signal_lines.append(
            "<b>На хорошем уровне:</b> " + ", ".join(good_lines) + "."
        )
    if risk_lines:
        key_signal_lines.append(
            "<b>Требует внимания:</b> " + ", ".join(risk_lines) + "."
        )
    if not key_signal_lines:
        key_signal_lines.append(
            "Существенных позитивных или проблемных изменений по отдельным параметрам не выделено."
        )

    block_keys = (
        f"<p style='margin-top:12px;color:{COLOR_TEXT_MAIN};font-family:{FONT_STACK};'>"
        f"<b>Ключевые сигналы этой недели:</b><br>"
        + "<br>".join(key_signal_lines) +
        "</p>"
    )

    # ------- 4. тревоги (⚠) -------
    alerts = extract_problem_params_for_alert(Wmap, L4map)
    if alerts:
        alert_items = []
        for p, cur_avg, cur_ans, diff_vs_base in alerts:
            title = PARAM_TITLES.get(p, p)
            if diff_vs_base is not None and diff_vs_base < -0.3:
                alert_items.append(
                    f"«{title}»: {cur_avg:.1f} ({cur_ans} ответов), ниже обычного уровня."
                )
            else:
                alert_items.append(
                    f"«{title}»: {cur_avg:.1f} ({cur_ans} ответов), показатель ниже ожидаемого."
                )

        alert_block = (
            f"<p style='margin-top:12px;color:{COLOR_NEGATIVE};font-weight:bold;font-family:{FONT_STACK};'>"
            "⚠ Обратить внимание:<br>"
            + "<br>".join(alert_items) +
            "</p>"
        )
    else:
        alert_block = ""

    html = (
        f"<h3 style='color:{COLOR_TEXT_MAIN};margin-top:24px;font-family:{FONT_STACK};'>"
        "Динамика и точки внимания"
        "</h3>"
        + block_prev
        + block_L4
        + block_keys
        + alert_block
    )
    return html


# =========================================
# Сравнение с прошлым годом
# =========================================

def yoy_block_table(period_rows: list[dict]):
    """
    Таблица:
    Период | Анкет | Итоговая оценка | Δ к прошлому году | NPS | Δ к прошлому году
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

        diff_over = None
        if _to_float_or_none(cur_over) is not None and _to_float_or_none(prev_over) is not None:
            diff_over = round(_to_float_or_none(cur_over) - _to_float_or_none(prev_over), 1)
        diff_nps = None
        if _to_float_or_none(cur_nps) is not None and _to_float_or_none(prev_nps) is not None:
            diff_nps = round(_to_float_or_none(cur_nps) - _to_float_or_none(prev_nps), 1)

        def delta_arrow_html(d, is_nps=False):
            if d is None or abs(d) < (2.0 if is_nps else 0.1):
                return "0.0" + (" п.п." if is_nps else "")
            if d > 0:
                return (
                    f"<span style='color:{COLOR_POSITIVE};font-weight:bold;'>▲ {signed_delta(d, ' п.п.' if is_nps else '')}</span>"
                )
            else:
                return (
                    f"<span style='color:{COLOR_NEGATIVE};font-weight:bold;'>▼ {signed_delta(d, ' п.п.' if is_nps else '')}</span>"
                )

        row_html.append(
            "<tr>"
            f"<td style='border:1px solid {COLOR_BORDER};text-align:left;color:{COLOR_TEXT_MAIN};white-space:nowrap;font-family:{FONT_STACK};'>{label}</td>"
            f"<td style='border:1px solid {COLOR_BORDER};text-align:right;color:{COLOR_TEXT_MAIN};font-family:{FONT_STACK};'>{fmt_int(cur_cnt)}</td>"
            f"<td style='border:1px solid {COLOR_BORDER};text-align:right;color:{COLOR_TEXT_MAIN};font-family:{FONT_STACK};'>{fmt_avg5(cur_over)}</td>"
            f"<td style='border:1px solid {COLOR_BORDER};text-align:right;color:{COLOR_TEXT_MAIN};font-family:{FONT_STACK};'>{delta_arrow_html(diff_over, is_nps=False)}</td>"
            f"<td style='border:1px solid {COLOR_BORDER};text-align:right;color:{COLOR_TEXT_MAIN};font-family:{FONT_STACK};'>{fmt_nps(cur_nps)}</td>"
            f"<td style='border:1px solid {COLOR_BORDER};text-align:right;color:{COLOR_TEXT_MAIN};font-family:{FONT_STACK};'>{delta_arrow_html(diff_nps, is_nps=True)}</td>"
            "</tr>"
        )

    html = f"""
    <h3 style="color:{COLOR_TEXT_MAIN};margin-top:24px;font-family:{FONT_STACK};">
      Сравнение с прошлым годом
    </h3>
    <table border='1' cellspacing='0' cellpadding='6'
           style="border-collapse:collapse;border-color:{COLOR_BORDER};color:{COLOR_TEXT_MAIN};font-size:14px;font-family:{FONT_STACK};">
      <tr style="background-color:{COLOR_BG_HEADER};color:{COLOR_TEXT_MAIN};font-family:{FONT_STACK};">
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

def yoy_comment_block(W, W_prevY):
    """
    После таблицы "Сравнение с прошлым годом":
    - общая оценка и NPS
    - параметры, ставшие лучше или хуже
    """

    Wmap      = df_param_map(W["by_param"])
    Wprev_map = df_param_map(W_prevY["by_param"])

    cur_over = _to_float_or_none(W["totals"]["overall5"])
    prv_over = _to_float_or_none(W_prevY["totals"]["overall5"])
    cur_nps  = _to_float_or_none(W["totals"]["nps"])
    prv_nps  = _to_float_or_none(W_prevY["totals"]["nps"])

    lines = []

    # итоговая оценка
    if cur_over is not None and prv_over is not None:
        d_over = round(cur_over - prv_over, 1)
        if abs(d_over) < 0.1:
            lines.append("<b>Итоговая оценка</b> на уровне прошлого года.")
        elif d_over > 0:
            lines.append(f"<b>Итоговая оценка</b> выше прошлого года ({signed_delta(d_over)}).")
        else:
            lines.append(f"<b>Итоговая оценка</b> ниже прошлого года ({signed_delta(d_over)}).")

    # nps
    if cur_nps is not None and prv_nps is not None:
        d_nps = round(cur_nps - prv_nps, 1)
        if abs(d_nps) < 2.0:
            lines.append("<b>NPS</b> на уровне прошлого года.")
        elif d_nps > 0:
            lines.append(f"<b>NPS</b> выше прошлого года ({signed_delta(d_nps,' п.п.')}).")
        else:
            lines.append(f"<b>NPS</b> ниже прошлого года ({signed_delta(d_nps,' п.п.')}).")

    ups_prevY, downs_prevY = extract_param_deltas(
        cur_map=Wmap,
        base_map=Wprev_map,
        rep_min_cur=2,
        rep_min_base=2,
        delta_thr=0.3,
    )

    downs_txt = []
    for p, cur_avg, diff in downs_prevY:
        title = PARAM_TITLES.get(p, p)
        downs_txt.append(f"«{title}» ({signed_delta(diff)})")

    ups_txt = []
    for p, cur_avg, diff in ups_prevY:
        title = PARAM_TITLES.get(p, p)
        ups_txt.append(f"«{title}» ({signed_delta(diff)})")

    if downs_txt:
        lines.append(
            "Ниже прошлого года: " + ", ".join(downs_txt) + "."
        )
    if ups_txt:
        lines.append(
            "Выше прошлого года: " + ", ".join(ups_txt) + "."
        )

    if not lines:
        lines = ["Существенных отличий от аналогичной недели прошлого года не зафиксировано."]

    html = (
        f"<p style='margin-top:12px;color:{COLOR_TEXT_MAIN};font-family:{FONT_STACK};'>"
        f"<b>Сравнение с аналогичной неделей прошлого года:</b><br>"
        + " ".join(lines) +
        "</p>"
    )
    return html


# =========================================
# Сноска с методикой
# =========================================

def footnote_block(all_start: date, w_end: date):
    return f"""
    <hr style="margin-top:24px;border:0;border-top:1px solid {COLOR_BORDER};">
    <p style="font-size:12px;color:{COLOR_TEXT_MAIN};line-height:1.5;margin-top:12px;font-family:{FONT_STACK};">
    <b>Пояснения.</b><br>
    • «Итого» — накопленная статистика с начала сбора анкет (с {human_date(all_start)}) по {human_date(w_end)}.<br>
    • Все оценки даются по шкале 1–5 (1 — плохо, 5 — отлично). В отчёте мы показываем средние значения по этой шкале, округлённые до 0,1.<br>
    • «Ответы» — число гостей, которые поставили оценку по конкретному вопросу. Если гость пропустил вопрос, он не влияет на среднюю.<br>
    • NPS рассчитывается по вопросу о готовности рекомендовать отель: 1–2 — детракторы, 3–4 — нейтральные, 5 — промоутеры. Пустые ответы не учитываются.
      NPS — разница между долей промоутеров и долей детракторов, в процентных пунктах.<br>
    • В блоках анализа:
      – «на уровне» означает отсутствие значимых отклонений (менее 0,1 балла по общей оценке и менее 2,0 п.п. по NPS);<br>
      – стрелка <span style="color:{COLOR_POSITIVE};font-weight:bold;">▲</span> и золотой цвет указывают на рост показателя;
        стрелка <span style="color:{COLOR_NEGATIVE};font-weight:bold;">▼</span> и оранжевый цвет — на снижение показателя;<br>
      – «ниже текущего уровня месяца / квартала / года» означает, что текущая неделя даёт заметно более низкую оценку по сравнению с усреднённым уровнем периода (падение ≥0,3).<br>
    </p>
    """


# =========================================
# Графики
# =========================================

def _week_order_key(k):
    try:
        y, w = str(k).split("-W")
        return int(y)*100 + int(w)
    except Exception:
        return 0

def plot_radar_params(W_df: pd.DataFrame, L4_df: pd.DataFrame, w_label_short: str, path_png: str):
    """
    Радар: текущая неделя vs средний уровень последних недель.
    Фильтр: неделя ≥2 ответов, L4 ≥4 ответов, исключаем NPS.
    """
    if W_df is None or W_df.empty or L4_df is None or L4_df.empty:
        return None

    Wm = df_param_map(W_df)
    Bm = df_param_map(L4_df)

    labels = []
    week_vals = []
    base_vals = []

    for p in PARAM_ORDER:
        if p == "nps":
            continue
        wrow = Wm.get(p)
        brow = Bm.get(p)
        if not wrow or not brow:
            continue
        w_ans = _to_float_or_none(wrow.get("answered"))
        b_ans = _to_float_or_none(brow.get("answered"))
        if w_ans is None or b_ans is None:
            continue
        if w_ans < 2 or b_ans < 4:
            continue
        w_avg = _to_float_or_none(wrow.get("avg5"))
        b_avg = _to_float_or_none(brow.get("avg5"))
        if w_avg is None or b_avg is None:
            continue

        labels.append(PARAM_TITLES.get(p, p))
        week_vals.append(w_avg)
        base_vals.append(b_avg)

    if len(labels) < 3:
        return None

    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    angles_closed = np.concatenate([angles, [angles[0]]])

    w_closed = np.array(week_vals + [week_vals[0]])
    b_closed = np.array(base_vals + [base_vals[0]])

    fig = plt.figure(figsize=(7.5, 6.5))
    fig.patch.set_facecolor("#FFFFFF")
    ax = plt.subplot(111, polar=True)
    ax.set_facecolor("#FFF6E5")

    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids(np.degrees(angles), labels, fontsize=8, color="#262D36")
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 5)

    ax.plot(angles_closed, w_closed, marker="o", linewidth=1, label="Текущая неделя")
    ax.fill(angles_closed, w_closed, alpha=0.1)

    ax.plot(angles_closed, b_closed, marker="o", linewidth=1, linestyle="--", label="Средний уровень последних недель")
    ax.fill(angles_closed, b_closed, alpha=0.1)

    ax.set_title(
        f"Ключевые параметры качества ({w_label_short})",
        color="#262D36",
        fontsize=10,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=8)

    plt.tight_layout()
    plt.savefig(path_png)
    plt.close()
    return path_png

def plot_params_heatmap(history_df: pd.DataFrame, path_png: str):
    """
    Теплокарта средних значений параметров за последние 8 недель.
    Ось X подписываем человеко-читаемыми неделями ('13–19 окт', ...).
    """
    if history_df is None or history_df.empty:
        return None

    df = history_df.copy()
    df["avg5"] = _to_num_series(df.get("avg5", pd.Series(dtype=str)))
    df["answered"] = _to_num_series(df.get("answered", pd.Series(dtype=str)))

    df = df[df["param"] != "nps"].copy()
    if df.empty:
        return None

    weeks_sorted = sorted(df["week_key"].unique(), key=_week_order_key)[-8:]
    df = df[df["week_key"].isin(weeks_sorted)].copy()
    if df.empty:
        return None

    ref_year = iso_week_monday(str(weeks_sorted[-1])).year
    week_labels_short = {wk: week_short_label_for_key(wk, ref_year=ref_year) for wk in weeks_sorted}

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
        .reindex(index=top_params, columns=weeks_sorted)
    )
    if pv.empty:
        return None

    fig, ax = plt.subplots(figsize=(10,5.5))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFF6E5")

    im = ax.imshow(pv.values, aspect="auto")

    ax.set_yticks(range(len(pv.index)))
    ax.set_yticklabels([PARAM_TITLES.get(p, p) for p in pv.index],
                       fontsize=8, color="#262D36")

    ax.set_xticks(range(len(pv.columns)))
    ax.set_xticklabels(
        [week_labels_short[wk] for wk in pv.columns],
        rotation=45,
        fontsize=8,
        color="#262D36",
        ha="right"
    )

    ax.set_title(
        "Оценки гостей по параметрам, последние 8 недель",
        color="#262D36",
        fontsize=10,
    )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path_png)
    plt.close()
    return path_png

def plot_overall_nps_trends(history_df: pd.DataFrame, path_png: str, as_of: date):
    """
    Линейный график за последние 12 недель:
    - Итоговая оценка
    - NPS (%)
    + их 4-недельные скользящие средние
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

    fig, ax1 = plt.subplots(figsize=(10,5.0))
    fig.patch.set_facecolor("#FFFFFF")
    ax1.set_facecolor("#FFF6E5")

    ax1.plot(weeks, ov["avg5"].values, marker="o", linewidth=1, label="Итоговая оценка")
    ax1.plot(weeks, ov_roll.values, linestyle="--", linewidth=1, label="Итоговая оценка (скользящее)")
    ax1.set_ylim(0,5)
    ax1.set_ylabel("Итоговая оценка", color="#262D36")
    ax1.tick_params(axis='y', labelcolor="#262D36")
    ax1.tick_params(axis='x', labelrotation=45, labelcolor="#262D36")

    ax2 = ax1.twinx()
    ax2.plot(weeks, npv["nps_value"].values, marker="s", linewidth=1, label="NPS, %", alpha=0.8)
    ax2.plot(weeks, nps_roll.values, linestyle="--", linewidth=1, label="NPS, % (скользящее)", alpha=0.8)
    ax2.set_ylim(-100,100)
    ax2.set_ylabel("NPS, %", color="#262D36")
    ax2.tick_params(axis='y', labelcolor="#262D36")

    ax1.set_title(
        f"Итоговая оценка и NPS: последние 12 недель (по состоянию на {human_date(as_of)})",
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


# =========================================
# Email helpers
# =========================================

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
        part.add_header("Content-Disposition", "attachment", filename=os.path.basename(path))
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


# =========================================
# Main
# =========================================

def main():
    # 1) последний Report_*.xlsx из Диска
    file_id, fname, fdate = latest_report_from_drive()
    data = drive_download(file_id)

    # 2) читаем Excel
    xls = pd.ExcelFile(io.BytesIO(data))
    if "Оценки гостей" in xls.sheet_names:
        raw = pd.read_excel(xls, sheet_name="Оценки гостей")
    else:
        raw = pd.read_excel(xls, sheet_name="Reviews")

    # 3) считаем неделю
    norm_df, agg_week = parse_and_aggregate_weekly(raw)

    # 4) пишем неделю в историю
    added = upsert_week(agg_week)
    print(f"[INFO] upsert_week(): добавлено строк недели: {added}")

    # 5) история
    hist = gs_get_df(SURVEYS_TAB, "A:I")
    if hist.empty:
        raise RuntimeError("surveys_history пуст — нечего анализировать.")

    # 6) временные границы
    wk_key   = str(agg_week["week_key"].iloc[0])
    w_start  = iso_week_monday(wk_key)
    w_end    = w_start + dt.timedelta(days=6)

    prev_start = w_start - dt.timedelta(days=7)
    prev_end   = prev_start + dt.timedelta(days=6)

    l4_start = w_start - dt.timedelta(days=28)
    l4_end   = w_start - dt.timedelta(days=1)

    ranges = period_ranges_for_week(w_start)

    # диапазон всей истории
    hist_mondays = hist["week_key"].map(lambda k: iso_week_monday(str(k)))
    all_start = hist_mondays.min()
    all_end   = hist_mondays.max() + dt.timedelta(days=6)

    # 7) агрегаты
    W = surveys_aggregate_period(hist, w_start, w_end)
    M = surveys_aggregate_period(hist, ranges["mtd"]["start"], ranges["mtd"]["end"])
    Q = surveys_aggregate_period(hist, ranges["qtd"]["start"], ranges["qtd"]["end"])
    Y = surveys_aggregate_period(hist, ranges["ytd"]["start"], ranges["ytd"]["end"])
    T = surveys_aggregate_period(hist, all_start, all_end)

    Prev = surveys_aggregate_period(hist, prev_start, prev_end)
    L4   = surveys_aggregate_period(hist, l4_start, l4_end)

    # Лог-сводка по текущей неделе
    try:
        totals = W.get("totals", {}) if isinstance(W, dict) else {}
        surveys_total = int(totals.get("surveys_total") or 0)
        overall5 = totals.get("overall5")
        nps = totals.get("nps")

        overall_txt = "n/a"
        if isinstance(overall5, (int, float)):
            try:
                if not math.isnan(overall5):
                    overall_txt = f"{overall5:.2f}"
            except Exception:
                pass

        nps_txt = "n/a"
        if isinstance(nps, (int, float)):
            try:
                if not math.isnan(nps):
                    nps_txt = f"{nps:.1f}"
            except Exception:
                pass

        print(
            f"[INFO] Неделя {wk_key} ({w_start}..{w_end}): "
            f"анкет {surveys_total}, средняя оценка {overall_txt}/5, NPS {nps_txt}"
        )
    except Exception as e:
        print(f"[WARN] Не удалось сформировать краткую сводку по неделе {wk_key}: {e}")

    # 8) подписи периодов
    week_lbl_no_g   = week_label(w_start, w_end)          # "13–19 окт 2025"
    week_col_label  = add_year_suffix(week_lbl_no_g)      # "13–19 окт 2025 г."
    week_label_full = "недели " + week_col_label          # "недели 13–19 окт 2025 г."

    month_label_full   = pretty_month_label(w_start)      # "Октябрь 2025 г."
    month_col_label    = month_label_full

    quarter_label_full = pretty_quarter_label(w_start)    # "IV квартал 2025 г."
    quarter_col_label  = quarter_label_full

    year_label_full    = pretty_year_label(w_start)       # "2025 г."
    year_col_label     = year_label_full

    total_label_full   = "Итого"
    total_col_label    = "Итого"

    # 9) сравнение с прошлым годом
    mtd_start = ranges["mtd"]["start"];   mtd_end = ranges["mtd"]["end"]
    qtd_start = ranges["qtd"]["start"];   qtd_end = ranges["qtd"]["end"]
    ytd_start = ranges["ytd"]["start"];   ytd_end = ranges["ytd"]["end"]

    py_w_start   = shift_year_back_safe(w_start)
    py_w_end     = shift_year_back_safe(w_end)
    py_mtd_start = shift_year_back_safe(mtd_start)
    py_mtd_end   = shift_year_back_safe(mtd_end)
    py_qtd_start = shift_year_back_safe(qtd_start)
    py_qtd_end   = shift_year_back_safe(qtd_end)
    py_ytd_start = shift_year_back_safe(ytd_start)
    py_ytd_end   = shift_year_back_safe(ytd_end)

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
            "label": "Неделя " + week_col_label,
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

    # 10) HTML секции письма
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

    yoy_table_html   = yoy_block_table(yoy_periods)
    yoy_comment_html = yoy_comment_block(W, W_prevY)

    footer_html = footnote_block(all_start, w_end)

    html_body = (
        head_html
        + table_html
        + trends_html
        + yoy_table_html
        + yoy_comment_html
        + footer_html
    )

    # 11) графики
    charts = []
    p1 = "/tmp/surveys_radar.png"
    p2 = "/tmp/surveys_heatmap.png"
    p3 = "/tmp/surveys_trends.png"

    w_label_short = week_short_label_for_key(wk_key, ref_year=w_start.year)

    if plot_radar_params(W["by_param"], L4["by_param"], w_label_short, p1): charts.append(p1)
    if plot_params_heatmap(hist, p2):                                      charts.append(p2)
    if plot_overall_nps_trends(hist, p3, as_of=w_end):                      charts.append(p3)

    # 12) тема письма
    subject = f"ARTSTUDIO Nevsky. Анкеты TL: Marketing — неделя {week_col_label}"

    # 13) отправка
    send_email(subject, html_body, attachments=charts)


if __name__ == "__main__":
    main()
