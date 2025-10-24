# Еженедельный отчёт по анкетам TL: Marketing (отдельный от отзывов)
# Требует: surveys_core.py и metrics_core.py (форматы периодов/лейблы и iso_week_monday)

import os, io, re, sys, json, math, time, datetime as dt
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
from googleapiclient.errors import HttpError

# Charts (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- ядро анкет ---
try:
    from agent.surveys_core import (
        parse_and_aggregate_weekly, SURVEYS_TAB,
        dedupe_surveys_history, surveys_aggregate_period
    )
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(__file__))
    from surveys_core import (
        parse_and_aggregate_weekly, SURVEYS_TAB,
        dedupe_surveys_history, surveys_aggregate_period
    )

# --- периоды/лейблы ---
try:
    from agent.metrics_core import (
        iso_week_monday, period_ranges_for_week,
        week_label, month_label, quarter_label, year_label
    )
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(__file__))
    from metrics_core import (
        iso_week_monday, period_ranges_for_week,
        week_label, month_label, quarter_label, year_label
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
# Helpers (Drive/Sheets) с retry
# =========================
WEEKLY_RE = re.compile(r"^Report_(\d{2})-(\d{2})-(\d{4})\.xlsx$", re.IGNORECASE)

def _retry(fn, tries=3, delay=2, backoff=2):
    err = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            err = e
            time.sleep(delay)
            delay *= backoff
    raise err

def latest_report_from_drive():
    def _list():
        return DRIVE.files().list(
            q=f"'{DRIVE_FOLDER_ID}' in parents and trashed=false and name contains 'Report_'",
            fields="files(id,name,modifiedTime)",
            pageSize=1000
        ).execute()
    res = _retry(_list)
    items = res.get("files", [])
    def parse_date_from_name(name: str):
        m = WEEKLY_RE.match(name)
        if not m: return None
        dd, mm, yyyy = map(int, m.groups())
        return date(yyyy, mm, dd)
    items = [(i["id"], i["name"], parse_date_from_name(i["name"])) for i in items if WEEKLY_RE.match(i["name"])]
    if not items:
        raise RuntimeError("В папке нет файлов вида Report_dd-mm-yyyy.xlsx.")
    items.sort(key=lambda t: t[2], reverse=True)
    return items[0]

def drive_download(file_id: str) -> bytes:
    def _dl():
        req = DRIVE.files().get_media(fileId=file_id)
        buf = io.BytesIO()
        dl = MediaIoBaseDownload(buf, req)
        done = False
        while not done:
            _, done = dl.next_chunk()
        return buf.getvalue()
    return _retry(_dl)

def ensure_tab(spreadsheet_id: str, tab_name: str, header: list[str]):
    def _get(): return SHEETS.get(spreadsheetId=spreadsheet_id).execute()
    meta = _retry(_get)
    tabs = [s["properties"]["title"] for s in meta.get("sheets", [])]
    if tab_name not in tabs:
        _retry(lambda: SHEETS.batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"requests":[{"addSheet":{"properties":{"title":tab_name}}}]}
        ).execute())
        _retry(lambda: SHEETS.values().update(
            spreadsheetId=spreadsheet_id,
            range=f"{tab_name}!A1:{chr(64+len(header))}1",
            valueInputOption="RAW",
            body={"values":[header]}
        ).execute())

def gs_get_df(tab: str, a1: str) -> pd.DataFrame:
    try:
        res = _retry(lambda: SHEETS.values().get(spreadsheetId=HISTORY_SHEET_ID, range=f"{tab}!{a1}").execute())
        vals = res.get("values", [])
        return pd.DataFrame(vals[1:], columns=vals[0]) if len(vals)>1 else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def gs_clear(a1: str):
    return _retry(lambda: SHEETS.values().clear(spreadsheetId=HISTORY_SHEET_ID, range=a1).execute())

def gs_append(tab: str, a1: str, rows: list[list]):
    if not rows: return
    _retry(lambda: SHEETS.values().append(
        spreadsheetId=HISTORY_SHEET_ID, range=f"{tab}!{a1}",
        valueInputOption="RAW", body={"values": rows}
    ).execute())

SURVEYS_HEADER = ["week_key","param","responses","avg5","avg10","promoters","detractors","nps"]

def rows_from_agg(df: pd.DataFrame) -> list[list]:
    out=[]
    for _, r in df.iterrows():
        out.append([
            str(r["week_key"]), str(r["param"]),
            int(r["responses"]) if pd.notna(r["responses"]) else 0,
            (None if pd.isna(r["avg5"])  else float(r["avg5"])),
            (None if pd.isna(r["avg10"]) else float(r["avg10"])),
            (None if "promoters"  not in r or pd.isna(r["promoters"])  else int(r["promoters"])),
            (None if "detractors" not in r or pd.isna(r["detractors"]) else int(r["detractors"])),
            (None if "nps"        not in r or pd.isna(r["nps"])        else float(r["nps"])),
        ])
    return out

def upsert_week(agg_week_df: pd.DataFrame) -> int:
    if agg_week_df.empty: return 0
    ensure_tab(HISTORY_SHEET_ID, SURVEYS_TAB, SURVEYS_HEADER)
    wk = str(agg_week_df["week_key"].iloc[0])

    hist = gs_get_df(SURVEYS_TAB, "A:H")
    keep = hist[hist.get("week_key","") != wk] if not hist.empty else pd.DataFrame(columns=SURVEYS_HEADER)

    # очистка и запись (с retry)
    gs_clear(f"{SURVEYS_TAB}!A2:H")
    rows_keep = keep[SURVEYS_HEADER].values.tolist() if not keep.empty else []
    rows_new  = rows_from_agg(agg_week_df)
    rows_all  = rows_keep + rows_new
    if rows_all:
        gs_append(SURVEYS_TAB, "A2", rows_all)
    return len(rows_new)

# =========================
# Парсинг Excel (Report_*.xlsx)
# =========================
def choose_surveys_sheet(xls_file: pd.ExcelFile) -> pd.DataFrame:
    preferred = {"оценки гостей","оценки","ответы","анкеты","responses"}
    for name in xls_file.sheet_names:
        if name.strip().lower() in preferred:
            return pd.read_excel(xls_file, name)
    # эвристика по колонкам
    best_name, best_score = None, -1
    probes = {"дата", "комментар", "оцен", "№ 1", "№ 2", "№ 3"}
    for name in xls_file.sheet_names:
        try:
            probe = pd.read_excel(xls_file, name, nrows=1)
            cols = [str(c).lower() for c in probe.columns]
            score = sum(any(p in c for p in probes) for c in cols)
            if score > best_score:
                best_name, best_score = name, score
        except Exception:
            continue
    return pd.read_excel(xls_file, best_name or xls_file.sheet_names[0])

# =========================
# Текст/HTML блоки
# =========================
PARAM_TITLES = {
    "overall": "Итоговая оценка",
    "fo_checkin": "СПиР при заезде",
    "clean_checkin": "Чистота при заезде",
    "room_comfort": "Комфорт и оснащение",
    "fo_stay": "СПиР в проживании",
    "its_service": "ИТС (техслужба)",
    "hsk_stay": "Уборка в проживании",
    "breakfast": "Завтраки",
    "atmosphere": "Атмосфера",
    "location": "Расположение",
    "value": "Цена/качество",
    "would_return": "Готовность вернуться",
    "nps": "NPS (1–5)",
}
def fmt_avg5(x):   return "—" if x is None or (isinstance(x,float) and np.isnan(x)) else f"{x:.2f} /5"
def fmt_int(x):    return "—" if x is None or (isinstance(x,float) and np.isnan(x)) else str(int(x))
def fmt_plain(x):  return "—" if x is None or (isinstance(x,float) and np.isnan(x)) else f"{x:.2f}"

def header_block(week_start: date, week_end: date, W: dict, M: dict, Q: dict, Y: dict):
    wl = week_label(week_start, week_end)
    def one_line(name, D):
        return (f"<b>{name}:</b> анкет {fmt_int(D['totals']['responses'])}, "
                f"итоговая {fmt_avg5(D['totals']['overall5'])}, "
                f"NPS {fmt_plain(D['totals']['nps'])}")
    return "\n".join([
        f"<h2>ARTSTUDIO Nevsky — Анкеты за неделю {wl}</h2>",
        f"<p><b>Неделя:</b> {wl}; анкет: <b>{fmt_int(W['totals']['responses'])}</b>; "
        f"итоговая: <b>{fmt_avg5(W['totals']['overall5'])}</b>; NPS: <b>{fmt_plain(W['totals']['nps'])}</b>.</p>",
        "<p>" +
        one_line(f"Текущий месяц ({month_label(week_start)})", M) + ";<br>" +
        one_line(f"Текущий квартал ({quarter_label(week_start)})", Q) + ";<br>" +
        one_line(f"Текущий год ({year_label(week_start)})", Y) + ".</p>"
    ])

def table_params_block(W_df: pd.DataFrame, M_df: pd.DataFrame, Q_df: pd.DataFrame, Y_df: pd.DataFrame):
    order = [
        "overall","fo_checkin","clean_checkin","room_comfort","fo_stay","its_service","hsk_stay","breakfast",
        "atmosphere","location","value","would_return","nps"
    ]
    def to_map(df: pd.DataFrame) -> dict[str, dict]:
        mp={}
        if df is None or df.empty: return mp
        for _, r in df.iterrows():
            p = str(r["param"])
            mp[p] = {
                "avg5": (None if "avg5" not in r or pd.isna(r["avg5"]) else float(r["avg5"])),
                "responses": (None if "responses" not in r or pd.isna(r["responses"]) else int(r["responses"])),
                "nps": (None if "nps" not in r or pd.isna(r["nps"]) else float(r["nps"])),
            }
        return mp
    W = to_map(W_df); M = to_map(M_df); Q = to_map(Q_df); Y = to_map(Y_df)

    def cell(mp: dict, param: str) -> str:
        r = mp.get(param)
        if r is None: return "<td>—</td><td>—</td>"
        if param == "nps":
            return f"<td>{fmt_plain(r['nps'])}</td><td>{fmt_int(r['responses'])}</td>"
        return f"<td>{fmt_avg5(r['avg5'])}</td><td>{fmt_int(r['responses'])}</td>"

    rows=[]
    for p in order:
        title = PARAM_TITLES.get(p, p)
        rows.append(f"<tr><td><b>{title}</b></td>{cell(W,p)}{cell(M,p)}{cell(Q,p)}{cell(Y,p)}</tr>")

    return f"""
    <h3>Параметры (неделя / Текущий месяц / Текущий квартал / Текущий год)</h3>
    <table border='1' cellspacing='0' cellpadding='6'>
      <tr>
        <th rowspan="2">Параметр</th>
        <th colspan="2">Неделя</th>
        <th colspan="2">Месяц</th>
        <th colspan="2">Квартал</th>
        <th colspan="2">Год</th>
      </tr>
      <tr>
        <th>Ср.</th><th>Ответы</th>
        <th>Ср.</th><th>Ответы</th>
        <th>Ср.</th><th>Ответы</th>
        <th>Ср.</th><th>Ответы</th>
      </tr>
      {''.join(rows)}
    </table>
    """

def footnote_block():
    return """
    <hr>
    <p><i>* Все значения по анкетам в шкале <b>/5</b> (средние взвешены по числу ответов).
    NPS: 1–2 — детракторы, 3–4 — нейтралы, 5 — промоутеры; NPS = %промоутеров − %детракторов.</i></p>
    """

# =========================
# Charts (с защитой от дублей)
# =========================
def _week_order_key(k):
    try:
        y, w = str(k).split("-W")
        return int(y) * 100 + int(w)
    except:
        return 0

def plot_radar_params(week_df: pd.DataFrame, month_df: pd.DataFrame, path_png: str):
    """
    Радар-диаграмма параметров: Неделя vs Текущий месяц (по /5).
    """
    if week_df is None or week_df.empty or month_df is None or month_df.empty:
        return None

    _nan = float("nan")

    # собираем значения по параметрам (без NPS)
    def to_map(df):
        mp={}
        for _, r in df.iterrows():
            p = str(r["param"])
            if p == "nps":
                continue
            mp[p] = (float(r["avg5"]) if pd.notna(r["avg5"]) else _nan)
        return mp

    W = to_map(week_df)
    M = to_map(month_df)

    # общий упорядоченный список параметров
    order = [p for p in [
        "overall","fo_checkin","clean_checkin","room_comfort","fo_stay","its_service","hsk_stay","breakfast",
        "atmosphere","location","value","would_return"
    ] if (p in W or p in M)]

    if len(order) < 3:
        return None

    labels = [PARAM_TITLES.get(p, p) for p in order]
    wvals  = [W.get(p, _nan) for p in order]
    mvals  = [M.get(p, _nan) for p in order]

    import numpy as np  # локально ОК, но теперь переменных внешнего замыкания нет
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
    plt.tight_layout(); plt.savefig(path_png); plt.close(); return path_png


def plot_params_heatmap(history_df: pd.DataFrame, path_png: str):
    if history_df is None or history_df.empty: return None
    df = history_df.copy()
    for c in ["avg5","responses"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["param"]!="nps"].copy()
    if df.empty: return None
    weeks = sorted(df["week_key"].unique(), key=_week_order_key)[-8:]
    df = df[df["week_key"].isin(weeks)]
    if df.empty: return None
    top_params = (df.groupby("param")["responses"].sum().sort_values(ascending=False).head(10).index.tolist())
    df = df[df["param"].isin(top_params)]
    pv = (df.pivot_table(index="param", columns="week_key", values="avg5", aggfunc="mean")
            .reindex(index=top_params, columns=weeks))
    if pv.empty: return None
    fig, ax = plt.subplots(figsize=(10,5.5))
    im = ax.imshow(pv.values, aspect="auto")
    ax.set_yticks(range(len(pv.index))); ax.set_yticklabels([PARAM_TITLES.get(p,p) for p in pv.index])
    ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels(list(pv.columns), rotation=45)
    ax.set_title("Анкеты: средняя /5 по параметрам (8 последних недель)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(path_png); plt.close(); return path_png

def plot_overall_nps_trends(history_df: pd.DataFrame, path_png: str):
    if history_df is None or history_df.empty: return None
    df = history_df.copy()
    for c in ["avg5","nps"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    # сгладим дубли недель (берём среднюю/первую)
    ov = (df[df["param"]=="overall"][["week_key","avg5"]]
            .groupby("week_key", as_index=False).agg(avg5=("avg5","mean")))
    npv= (df[df["param"]=="nps"][["week_key","nps"]]
            .groupby("week_key", as_index=False).agg(nps=("nps","mean")))
    weeks = sorted(set(ov["week_key"]).union(set(npv["week_key"])), key=_week_order_key)[-12:]
    if not weeks: return None
    ov = ov.set_index("week_key").reindex(weeks)
    npv= npv.set_index("week_key").reindex(weeks)
    ov_roll  = ov["avg5"].rolling(window=4, min_periods=2).mean()
    nps_roll = npv["nps"].rolling(window=4, min_periods=2).mean()

    fig, ax1 = plt.subplots(figsize=(10,5.0))
    ax1.plot(weeks, ov["avg5"].values, marker="o", label="Итоговая /5")
    ax1.plot(weeks, ov_roll.values, linestyle="--", label="Итоговая /5 (скользящее)")
    ax1.set_ylim(0,5); ax1.set_ylabel("Итоговая /5")

    ax2 = ax1.twinx()
    ax2.plot(weeks, npv["nps"].values, marker="s", label="NPS", alpha=0.8)
    ax2.plot(weeks, nps_roll.values, linestyle="--", label="NPS (скользящее)", alpha=0.8)
    ax2.set_ylim(-100,100); ax2.set_ylabel("NPS, п.п.")
    ax1.set_title("Анкеты: тренды Итоговой /5 и NPS (12 недель)")
    ax1.set_xticks(range(len(weeks))); ax1.set_xticklabels(weeks, rotation=45)
    lines, labels = [], []
    for ax in (ax1, ax2):
        l, lab = ax.get_legend_handles_labels(); lines += l; labels += lab
    ax1.legend(lines, labels, loc="upper left")
    plt.tight_layout(); plt.savefig(path_png); plt.close(); return path_png

# =========================
# Email
# =========================
def attach_file(msg, path):
    if not path or not os.path.exists(path): return
    ctype, encoding = mimetypes.guess_type(path)
    if ctype is None or encoding is not None: ctype='application/octet-stream'
    maintype, subtype = ctype.split('/',1)
    with open(path,"rb") as fp:
        part = MIMEBase(maintype, subtype); part.set_payload(fp.read()); encoders.encode_base64(part)
        part.add_header('Content-Disposition','attachment', filename=os.path.basename(path)); msg.attach(part)

def send_email(subject, html_body, attachments=None):
    if not RECIPIENTS:
        print("[WARN] RECIPIENTS is empty; skip email"); return
    msg = MIMEMultipart("mixed")
    msg["Subject"] = subject; msg["From"] = SMTP_FROM; msg["To"] = ", ".join(RECIPIENTS)
    alt = MIMEMultipart("alternative"); alt.attach(MIMEText(html_body, "html", "utf-8")); msg.attach(alt)
    for p in (attachments or []): attach_file(msg, p)
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        if SMTP_USER and SMTP_PASS:
            s.login(SMTP_USER, SMTP_PASS)
        s.sendmail(msg["From"], RECIPIENTS, msg.as_string())

# =========================
# MAIN
# =========================
def main():
    # 1) подтянули последний Report_*.xlsx
    fid, name, fdate = latest_report_from_drive()
    blob = drive_download(fid)
    xls = pd.ExcelFile(io.BytesIO(blob))
    df_raw = choose_surveys_sheet(xls)

    # 2) нормализация + недельная агрегация и апсерт в surveys_history
    norm, agg_week = parse_and_aggregate_weekly(df_raw)
    upserted = upsert_week(agg_week)
    print(f"[INFO] surveys_weekly: upserted {upserted} rows into '{SURVEYS_TAB}'")

    # 3) история (с дедупом) + периоды
    hist = gs_get_df(SURVEYS_TAB, "A:H")
    if hist.empty:
        raise RuntimeError("surveys_history пуст — нечего анализировать.")
    hist = dedupe_surveys_history(hist)

    wk_key = str(agg_week["week_key"].iloc[0])
    w_start = iso_week_monday(wk_key); w_end = w_start + dt.timedelta(days=6)
    ranges = period_ranges_for_week(w_start)  # mtd/qtd/ytd: {start,end,label}

    W = surveys_aggregate_period(hist, w_start, w_end, iso_week_monday)
    M = surveys_aggregate_period(hist, ranges["mtd"]["start"], ranges["mtd"]["end"], iso_week_monday)
    Q = surveys_aggregate_period(hist, ranges["qtd"]["start"], ranges["qtd"]["end"], iso_week_monday)
    Y = surveys_aggregate_period(hist, ranges["ytd"]["start"], ranges["ytd"]["end"], iso_week_monday)

    # 4) HTML
    head = header_block(w_start, w_end, W, M, Q, Y)
    table = table_params_block(W["by_param"], M["by_param"], Q["by_param"], Y["by_param"])
    html = head + table + footnote_block()

    # 5) Графики
    charts = []
    p1 = "/tmp/surveys_radar.png";   plot_radar_params(W["by_param"], M["by_param"], p1); charts.append(p1)
    p2 = "/tmp/surveys_heatmap.png"; plot_params_heatmap(hist, p2);                      charts.append(p2)
    p3 = "/tmp/surveys_trends.png";  plot_overall_nps_trends(hist, p3);                 charts.append(p3)
    charts = [p for p in charts if p and os.path.exists(p)]

    # 6) Письмо
    subject = f"ARTSTUDIO Nevsky. Анкеты TL: Marketing — неделя {w_start:%d.%m}–{w_end:%d.%m}"
    send_email(subject, html, attachments=charts)

if __name__ == "__main__":
    main()
