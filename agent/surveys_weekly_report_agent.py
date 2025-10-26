# agent/surveys_weekly_report_agent.py
# Еженедельный отчёт по анкетам TL: Marketing
#
# Версия с исправлением средних значений:
# - корректно читаем числа с запятой ("4,71") из Google Sheets
# - корректно считаем средние /5 и показываем их в письме
#
# avg10 пока остаётся в истории (в шите), но не используется в письме.
# На следующем шаге уберём avg10 из пайплайна полностью.

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
    return items[0]  # (file_id, filename, date)

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
    try:
        res = SHEETS.values().get(
            spreadsheetId=HISTORY_SHEET_ID,
            range=f"{tab}!{a1}"
        ).execute()
        vals = res.get("values", [])
        return pd.DataFrame(vals[1:], columns=vals[0]) if len(vals) > 1 else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# структура листа surveys_history (пока в ней avg10 остаётся для совместимости с текущими данными)
SURVEYS_HEADER = [
    "week_key",
    "param",
    "surveys_total",
    "answered",
    "avg5",
    "avg10",
    "promoters",
    "detractors",
    "nps_answers",
    "nps_value",
]

def rows_from_agg(df: pd.DataFrame) -> list[list]:
    out = []
    for _, r in df.iterrows():
        out.append([
            str(r["week_key"]),
            str(r["param"]),
            (int(r["surveys_total"]) if not pd.isna(r["surveys_total"]) else 0),
            (int(r["answered"])      if not pd.isna(r["answered"])      else 0),
            (None if pd.isna(r["avg5"])        else float(r["avg5"])),
            (None if pd.isna(r["avg10"])       else float(r["avg10"])),
            (None if "promoters"    not in r or pd.isna(r["promoters"])    else int(r["promoters"])),
            (None if "detractors"   not in r or pd.isna(r["detractors"])   else int(r["detractors"])),
            (None if "nps_answers"  not in r or pd.isna(r["nps_answers"])  else int(r["nps_answers"])),
            (None if "nps_value"    not in r or pd.isna(r["nps_value"])    else float(r["nps_value"])),
        ])
    return out

def upsert_week(agg_week_df: pd.DataFrame) -> int:
    """
    Перезаписываем ТОЛЬКО эту неделю:
     - читаем текущую историю
     - выкидываем строки той же week_key
     - чистим строковые данные ниже заголовка
     - записываем обратно (старые недели + новая неделя)
    """
    if agg_week_df.empty:
        return 0

    ensure_tab(HISTORY_SHEET_ID, SURVEYS_TAB, SURVEYS_HEADER)

    wk = str(agg_week_df["week_key"].iloc[0])

    hist = gs_get_df(SURVEYS_TAB, "A:J")
    keep = (
        hist[hist.get("week_key", "") != wk]
        if not hist.empty
        else pd.DataFrame(columns=SURVEYS_HEADER)
    )

    # очистка листа со второй строки
    SHEETS.values().clear(
        spreadsheetId=HISTORY_SHEET_ID,
        range=f"{SURVEYS_TAB}!A2:J",
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
    Приводим серию значений из Google Sheets к числам:
    - заменяем запятую на точку
    - конвертим в float
    Всё, что не удаётся, становится NaN.
    """
    if s is None:
        return pd.Series(dtype="float64")

    # в таблице могут быть уже числа (float/int), тогда astype(str) им не навредит
    return pd.to_numeric(
        s.astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )

def _weighted_avg(values: pd.Series, weights: pd.Series) -> float | None:
    """
    Взвешенная средняя по /5.
    values  = средние за недели (могут быть строками "4,71")
    weights = answered за недели (могут быть строками "5")
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
    Собираем агрегаты за период [start; end] по всем параметрам.

    Возвращает:
    {
      "by_param": DataFrame с колонками:
          param, surveys_total, answered, avg5, avg10,
          promoters, detractors, nps_answers, nps_value
      "totals": {
          "surveys_total": int,
          "overall5": float | None,
          "nps": float | None
      }
    }

    Логика:
    - Средняя /5 считается взвешенно по числу ответивших.
    - NPS = (%промоутеров - %детракторов)*100 по тем, кто ответил на NPS.
    """

    empty_bp_cols = [
        "param",
        "surveys_total",
        "answered",
        "avg5",
        "avg10",
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

    # Грубо нормализуем числовые столбцы сразу после чтения из Google Sheets
    numeric_cols = [
        "surveys_total",
        "answered",
        "avg5",
        "avg10",
        "promoters",
        "detractors",
        "nps_answers",
        "nps_value",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = _to_num_series(df[c])

    # Вычисляем понедельник недели из week_key и фильтруем интервал периода
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
        # avg10 взвешенно для совместимости со структурой, хотя в письме его не показываем
        if "avg10" in g.columns:
            avg10_weighted = _weighted_avg(g["avg10"], g["answered"])
        else:
            avg10_weighted = None

        # NPS только для param == "nps"
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
            "param":        param,
            "surveys_total": surveys_total_sum,
            "answered":      answered_sum,
            "avg5":          avg5_weighted,
            "avg10":         avg10_weighted,
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
    # одна цифра после запятой/точки ок (в письме ты видел "60.0")
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
        return None

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
    if history_df is None or history_df.empty:
        return None

    df = history_df.copy()
    for c in ["avg5", "answered"]:
        if c in df.columns:
            df[c] = _to_num_series(df[c])

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
    if history_df is None or history_df.empty:
        return None

    df = history_df.copy()
    df["avg5"] = np.where(
        df["avg5"].notna(),
        _to_num_series(df["avg5"]),
        np.nan
    )
    df["nps_value"] = np.where(
        df["nps_value"].notna(),
        _to_num_series(df["nps_value"]),
        np.nan
    )

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
    # 1) берём последний отчёт Report_DD-MM-YYYY.xlsx из Drive
    file_id, fname, fdate = latest_report_from_drive()
    data = drive_download(file_id)

    xls = pd.ExcelFile(io.BytesIO(data))
    if "Оценки гостей" in xls.sheet_names:
        raw = pd.read_excel(xls, sheet_name="Оценки гостей")
    else:
        raw = pd.read_excel(xls, sheet_name="Reviews")

    # нормализация и недельная агрегация
    norm_df, agg_week = parse_and_aggregate_weekly(raw)

    # 2) обновляем лист surveys_history без дублей по неделе
    added = upsert_week(agg_week)
    print(f"[INFO] upsert_week(): добавлено строк недели: {added}")

    # перечитываем всю историю (уже с новой неделей)
    hist = gs_get_df(SURVEYS_TAB, "A:J")
    if hist.empty:
        raise RuntimeError("surveys_history пуст — нечего анализировать.")

    wk_key = str(agg_week["week_key"].iloc[0])
    w_start = iso_week_monday(wk_key)
    w_end   = w_start + dt.timedelta(days=6)

    ranges = period_ranges_for_week(w_start)

    # "Итого (вся история)" — от первой недели в шите до последней
    hist_mondays = hist["week_key"].map(lambda k: iso_week_monday(str(k)))
    all_start = hist_mondays.min()
    all_end   = hist_mondays.max() + dt.timedelta(days=6)

    # агрегаты по периодам
    W = surveys_aggregate_period(hist, w_start, w_end)
    M = surveys_aggregate_period(hist, ranges["mtd"]["start"], ranges["mtd"]["end"])
    Q = surveys_aggregate_period(hist, ranges["qtd"]["start"], ranges["qtd"]["end"])
    Y = surveys_aggregate_period(hist, ranges["ytd"]["start"], ranges["ytd"]["end"])
    T = surveys_aggregate_period(hist, all_start, all_end)

    head  = header_block(w_start, w_end, W, M, Q, Y, T)
    table = table_params_block(W["by_param"], M["by_param"], Q["by_param"], Y["by_param"], T["by_param"])
    html  = head + table + footnote_block()

    # графики
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
