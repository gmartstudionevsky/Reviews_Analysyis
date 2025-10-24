# agent/surveys_weekly_report_agent.py
# Еженедельный отчёт по анкетам TL: Marketing (отдельно от отзывов)

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

# --- ядро анкет (нормализация и недельная агрегация) ---
try:
    from agent.surveys_core import parse_and_aggregate_weekly, SURVEYS_TAB
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(__file__))
    from surveys_core import parse_and_aggregate_weekly, SURVEYS_TAB

# --- утилиты периодов и форматирования (берём из metrics_core) ---
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
# Drive helpers
# =========================
WEEKLY_RE = re.compile(r"^Report_(\d{2})-(\d{2})-(\d{4})\.xlsx$", re.IGNORECASE)

def latest_report_from_drive():
    res = DRIVE.files().list(
        q=f"'{DRIVE_FOLDER_ID}' in parents and trashed=false and name contains 'Report_'",
        fields="files(id,name,modifiedTime)",
        pageSize=1000
    ).execute()
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
    return items[0]  # id, name, date

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
        SHEETS.batchUpdate(spreadsheetId=spreadsheet_id,
                           body={"requests":[{"addSheet":{"properties":{"title":tab_name}}}]}).execute()
        SHEETS.values().update(spreadsheetId=spreadsheet_id,
                               range=f"{tab_name}!A1:{chr(64+len(header))}1",
                               valueInputOption="RAW",
                               body={"values":[header]}).execute()

def gs_get_df(tab: str, a1: str) -> pd.DataFrame:
    try:
        res = SHEETS.values().get(spreadsheetId=HISTORY_SHEET_ID, range=f"{tab}!{a1}").execute()
        vals = res.get("values", [])
        return pd.DataFrame(vals[1:], columns=vals[0]) if len(vals)>1 else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def gs_append(tab: str, a1: str, rows: list[list]):
    if not rows: return
    SHEETS.values().append(spreadsheetId=HISTORY_SHEET_ID,
                           range=f"{tab}!{a1}",
                           valueInputOption="RAW",
                           body={"values": rows}).execute()

SURVEYS_HEADER = ["week_key","param","responses","avg5","avg10","promoters","detractors","nps"]

def load_existing_keys() -> set[tuple[str,str]]:
    df = gs_get_df(SURVEYS_TAB, "A:H")
    if df.empty: return set()
    return set(zip(df.get("week_key",[]), df.get("param",[])))

def append_week_if_needed(agg_week_df: pd.DataFrame):
    """Добавляет строки недели в surveys_history, если их ещё нет."""
    if agg_week_df.empty: return 0
    ensure_tab(HISTORY_SHEET_ID, SURVEYS_TAB, SURVEYS_HEADER)
    exists = load_existing_keys()
    need=[]
    for _, r in agg_week_df.iterrows():
        k=(str(r["week_key"]), str(r["param"]))
        if k not in exists:
            need.append([
                str(r["week_key"]), str(r["param"]),
                int(r["responses"]) if pd.notna(r["responses"]) else 0,
                (None if pd.isna(r["avg5"]) else float(r["avg5"])),
                (None if pd.isna(r["avg10"]) else float(r["avg10"])),
                (None if "promoters" not in r or pd.isna(r["promoters"]) else int(r["promoters"])),
                (None if "detractors" not in r or pd.isna(r["detractors"]) else int(r["detractors"])),
                (None if "nps" not in r or pd.isna(r["nps"]) else float(r["nps"])),
            ])
            exists.add(k)
    if need:
        gs_append(SURVEYS_TAB, "A:H", need)
    return len(need)

# =========================
# Aggregation over periods
# =========================
def week_monday_from_key(week_key: str) -> date:
    # week_key = 'YYYY-W##'
    return iso_week_monday(week_key)

def surveys_aggregate_period(history_df: pd.DataFrame, start: date, end: date) -> dict:
    """
    Взвешенная агрегация по параметрам: суммы ответов, ср./5 (и /10) и NPS (по суммам P и D).
    Возвращает dict:
      {
        'by_param': DataFrame[param,responses,avg5,avg10,promoters,detractors,nps],
        'totals': {'responses': int, 'overall5': float|None, 'nps': float|None}
      }
    """
    if history_df.empty:
        return {"by_param": pd.DataFrame(columns=["param","responses","avg5","avg10","promoters","detractors","nps"]),
                "totals": {"responses": 0, "overall5": None, "nps": None}}

    df = history_df.copy()
    # привести числа
    for c in ["responses","avg5","avg10","promoters","detractors","nps"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # фильтр по календарному интервалу
    df["monday"] = df["week_key"].map(week_monday_from_key)
    df = df[(df["monday"]>=start) & (df["monday"]<=end)].copy()
    if df.empty:
        return {"by_param": pd.DataFrame(columns=["param","responses","avg5","avg10","promoters","detractors","nps"]),
                "totals": {"responses": 0, "overall5": None, "nps": None}}

    # взвешенная средняя по avg5/avg10: вес = responses
    rows=[]
    for param, grp in df.groupby("param"):
        resp = int(grp["responses"].fillna(0).sum())
        if param == "nps":
            prom = int(grp["promoters"].fillna(0).sum())
            detr = int(grp["detractors"].fillna(0).sum())
            nps = ( (prom/(resp or 1) - detr/(resp or 1)) * 100.0 ) if resp>0 else np.nan
            rows.append([param, resp, np.nan, np.nan, prom, detr, (None if np.isnan(nps) else round(float(nps),1))])
        else:
            # взвешенная avg5/avg10
            w = grp["responses"].fillna(0).values
            a5 = grp["avg5"].values
            a10= grp["avg10"].values
            avg5  = float(np.nansum(a5*w)/np.nansum(w)) if np.nansum(w)>0 else np.nan
            avg10 = float(np.nansum(a10*w)/np.nansum(w)) if np.nansum(w)>0 else np.nan
            rows.append([param, resp,
                         (None if np.isnan(avg5) else round(avg5,2)),
                         (None if np.isnan(avg10) else round(avg10,2)),
                         None, None, None])

    by_param = pd.DataFrame(rows, columns=["param","responses","avg5","avg10","promoters","detractors","nps"])

    # итого по overall и nps
    totals = {"responses": int(by_param["responses"].fillna(0).sum())}
    ov = by_param[by_param["param"]=="overall"]
    totals["overall5"] = (None if ov.empty or pd.isna(ov.iloc[0]["avg5"]) else float(ov.iloc[0]["avg5"]))
    nps_row = by_param[by_param["param"]=="nps"]
    totals["nps"] = (None if nps_row.empty or pd.isna(nps_row.iloc[0]["nps"]) else float(nps_row.iloc[0]["nps"]))
    return {"by_param": by_param.sort_values("param"), "totals": totals}

# =========================
# Текст/HTML
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
def fmt_nps(x):    return "—" if x is None or (isinstance(x,float) and np.isnan(x)) else f"{x:+.1f} п.п."
def fmt_plain(x):  return "—" if x is None or (isinstance(x,float) and np.isnan(x)) else f"{x:.2f}"

def header_block(week_start: date, week_end: date, W: dict, M: dict, Q: dict, Y: dict):
    wl = week_label(week_start, week_end)
    parts = []
    parts.append(f"<h2>ARTSTUDIO Nevsky — Анкеты за неделю {wl}</h2>")
    parts.append(
        "<p><b>Неделя:</b> "
        f"{wl}; анкет: <b>{fmt_int(W['totals']['responses'])}</b>; "
        f"итоговая: <b>{fmt_avg5(W['totals']['overall5'])}</b>; "
        f"NPS: <b>{fmt_plain(W['totals']['nps'])}</b>.</p>"
    )
    def one_line(name, D):
        return (f"<b>{name}:</b> анкет {fmt_int(D['totals']['responses'])}, "
                f"итоговая {fmt_avg5(D['totals']['overall5'])}, "
                f"NPS {fmt_plain(D['totals']['nps'])}")
    parts.append("<p>" +
        one_line(f"Текущий месяц ({month_label(week_start)})", M) + ";<br>" +
        one_line(f"Текущий квартал ({quarter_label(week_start)})", Q) + ";<br>" +
        one_line(f"Текущий год ({year_label(week_start)})", Y) + ".</p>"
    )
    return "\n".join(parts)

def table_params_block(W_df: pd.DataFrame, M_df: pd.DataFrame, Q_df: pd.DataFrame, Y_df: pd.DataFrame):
    # делаем объединённый список параметров в нужном порядке
    order = [p for p in [
        "overall","fo_checkin","clean_checkin","room_comfort","fo_stay","its_service","hsk_stay","breakfast",
        "atmosphere","location","value","would_return","nps"
    ]]
    def to_map(df):
        mp={}
        if df is None or df.empty: return mp
        for _, r in df.iterrows():
            mp[str(r["param"])] = r
        return mp
    W, M, Q, Y = map(to_map, [W_df, M_df, Q_df, Y_df])

    rows=[]
    for p in order:
        t = PARAM_TITLES.get(p, p)
        def cell(mp):
            r = mp.get(p)
            if not r: 
                return "<td>—</td><td>—</td>"
            if p == "nps":
                return f"<td>{fmt_plain(r['nps'])}</td><td>{fmt_int(r['responses'])}</td>"
            return f"<td>{fmt_avg5(r['avg5'])}</td><td>{fmt_int(r['responses'])}</td>"
        rows.append(f"<tr><td><b>{t}</b></td>{cell(W)}{cell(M)}{cell(Q)}{cell(Y)}</tr>")

    html = f"""
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
    return html

def footnote_block():
    return """
    <hr>
    <p><i>* Все значения по анкетам отображаются в шкале <b>/5</b>. Внутри расчётов применяется взвешивание по количеству ответов.
    NPS считается по шкале 1–5: 1–2 — детракторы, 3 — нейтрал, 4–5 — промоутеры; NPS = %промоутеров − %детракторов.</i></p>
    """

# =========================
# Email helpers
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
        print("[WARN] RECIPIENTS is empty; skip email")
        return
    msg = MIMEMultipart("mixed")
    msg["Subject"] = subject; msg["From"] = SMTP_FROM; msg["To"] = ", ".join(RECIPIENTS)
    alt = MIMEMultipart("alternative"); alt.attach(MIMEText(html_body, "html", "utf-8")); msg.attach(alt)
    for p in (attachments or []): attach_file(msg, p)
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls(); 
        if SMTP_USER and SMTP_PASS:
            s.login(SMTP_USER, SMTP_PASS)
        s.sendmail(msg["From"], RECIPIENTS, msg.as_string())

# =========================
# MAIN
# =========================
def main():
    # 1) Подтянем последний Report_*.xlsx и при необходимости допишем неделю в surveys_history
    fid, name, fdate = latest_report_from_drive()
    blob = drive_download(fid)
    df_raw = pd.read_excel(io.BytesIO(blob))
    norm, agg_week = parse_and_aggregate_weekly(df_raw)  # agg_week: week_key|param|responses|avg5|avg10|promoters|detractors|nps
    added = append_week_if_needed(agg_week)
    print(f"[INFO] surveys_weekly: appended {added} new rows into {SURVEYS_TAB}")

    # 2) История + периоды
    hist = gs_get_df(SURVEYS_TAB, "A:H")
    if hist.empty:
        raise RuntimeError("surveys_history пуст — нечего анализировать.")
    # определить неделю из agg_week
    wk_key = str(agg_week["week_key"].iloc[0])
    w_start = iso_week_monday(wk_key); w_end = w_start + dt.timedelta(days=6)

    ranges = period_ranges_for_week(w_start)  # mtd/qtd/ytd {start,end,label}

    # Агрегации
    W = surveys_aggregate_period(hist, w_start, w_end)
    M = surveys_aggregate_period(hist, ranges["mtd"]["start"], ranges["mtd"]["end"])
    Q = surveys_aggregate_period(hist, ranges["qtd"]["start"], ranges["qtd"]["end"])
    Y = surveys_aggregate_period(hist, ranges["ytd"]["start"], ranges["ytd"]["end"])

    # 3) HTML
    head = header_block(w_start, w_end, W, M, Q, Y)
    table = table_params_block(W["by_param"], M["by_param"], Q["by_param"], Y["by_param"])
    html = head + table + footnote_block()

    # 4) Письмо
    subject = f"ARTSTUDIO Nevsky. Анкеты TL: Marketing — неделя {w_start:%d.%m}–{w_end:%d.%m}"
    send_email(subject, html, attachments=None)

if __name__ == "__main__":
    main()
