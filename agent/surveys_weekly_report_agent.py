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

# Charts (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- ядро анкет (нормализация и недельная агрегация) ---
try:
    from agent.surveys_core import parse_and_aggregate_weekly, SURVEYS_TAB, iso_week_key
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(__file__))
    from surveys_core import parse_and_aggregate_weekly, SURVEYS_TAB, iso_week_key

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
            body={"requests":[{"addSheet":{"properties":{"title":tab_name}}}]}
        ).execute()
        SHEETS.values().update(
            spreadsheetId=spreadsheet_id,
            range=f"{tab_name}!A1:{chr(64+len(header))}1",
            valueInputOption="RAW",
            body={"values":[header]}
        ).execute()

def gs_get_df(tab: str, a1: str) -> pd.DataFrame:
    try:
        res = SHEETS.values().get(spreadsheetId=HISTORY_SHEET_ID, range=f"{tab}!{a1}").execute()
        vals = res.get("values", [])
        return pd.DataFrame(vals[1:], columns=vals[0]) if len(vals)>1 else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def gs_append(tab: str, a1: str, rows: list[list]):
    if not rows: return
    SHEETS.values().append(
        spreadsheetId=HISTORY_SHEET_ID,
        range=f"{tab}!{a1}",
        valueInputOption="RAW",
        body={"values": rows}
    ).execute()

# ======================
# History upsert
# ======================
HEADER = ["week_key","param","responses","avg5","avg10","promoters","detractors","nps"]

def load_existing_keys() -> set[tuple[str,str]]:
    df = gs_get_df(SURVEYS_TAB, "A:H")
    if df.empty: return set()
    return set(zip(df.get("week_key",[]), df.get("param",[])))

def upsert_week(agg_week: pd.DataFrame) -> int:
    existing = load_existing_keys()
    need = []
    for _, r in agg_week.iterrows():
        k = (str(r["week_key"]), str(r["param"]))
        if k not in existing:
            need.append(r)
            existing.add(k)
    if not need:
        return 0
    rows = rows_from_agg(pd.DataFrame(need))
    gs_append(SURVEYS_TAB, "A:H", rows)
    return len(rows)

def rows_from_agg(df: pd.DataFrame) -> list[list]:
    rows=[]
    for _, r in df.iterrows():
        rows.append([
            str(r["week_key"]),
            str(r["param"]),
            int(r["responses"]) if pd.notna(r["responses"]) else 0,
            (None if pd.isna(r["avg5"]) else float(r["avg5"])),
            (None if pd.isna(r["avg10"]) else float(r["avg10"])),
            int(r["promoters"]) if "promoters" in r and pd.notna(r["promoters"]) else None,
            int(r["detractors"]) if "detractors" in r and pd.notna(r["detractors"]) else None,
            (None if ("nps" not in r or pd.isna(r["nps"])) else float(r["nps"])),
        ])
    return rows

# ======================
# Period aggregate
# ======================
def surveys_aggregate_period(hist: pd.DataFrame, start: date, end: date) -> dict:
    wk_start = iso_week_key(start)
    wk_end = iso_week_key(end)
    period_df = hist[(hist["week_key"] >= wk_start) & (hist["week_key"] <= wk_end)].copy()

    by_param = []
    for p in PARAM_ORDER if p != "nps_1_5":
        pdf = period_df[period_df["param"] == p]
        valid = pd.notna(pdf["avg5"])
        if not valid.any():
            avg = np.nan
            total_responses = 0
        else:
            values = pdf["avg5"][valid].astype(float)
            weights = pdf["responses"][valid].astype(int)
            avg = np.average(values, weights=weights)
            total_responses = int(weights.sum())
        by_param.append({"param":p, "avg5":round(avg,2) if not np.isnan(avg) else None, "avg10":round(avg*2,2) if not np.isnan(avg) else None, "responses":total_responses, "nps":None})

    # NPS total
    pdf = period_df[period_df["param"] == "nps"]
    if not pdf.empty:
        prom = pdf["promoters"].sum()
        det = pdf["detractors"].sum()
        tot = pdf["responses"].sum()
        nps = round(100 * (prom - det) / tot, 2) if tot > 0 else np.nan
    else:
        nps, tot = np.nan, 0
    by_param.append({"param":"nps", "avg5":None, "avg10":None, "responses":int(tot), "nps":nps if not np.isnan(nps) else None})

    overall = next((bp for bp in by_param if bp["param"] == "overall"), {"responses":0, "avg5":None, "nps":None})
    return {"total_surveys": overall["responses"], "overall_avg5": overall["avg5"], "nps": by_param[-1]["nps"], "by_param": by_param}

# ======================
# HTML blocks
# ======================
PARAM_NAMES = {
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
    "nps": "NPS (1-5)",
}

def header_block(w_start: date, w_end: date, W: dict, M: dict, Q: dict, Y: dict) -> str:
    w_overall = f"{W['overall_avg5']:.2f} /5" if pd.notna(W['overall_avg5']) else "—"
    m_overall = f"{M['overall_avg5']:.2f} /5" if pd.notna(M['overall_avg5']) else "—"
    q_overall = f"{Q['overall_avg5']:.2f} /5" if pd.notna(Q['overall_avg5']) else "—"
    y_overall = f"{Y['overall_avg5']:.2f} /5" if pd.notna(Y['overall_avg5']) else "—"
    w_nps = f"{W['nps']:.2f}" if pd.notna(W['nps']) else "—"
    m_nps = f"{M['nps']:.2f}" if pd.notna(M['nps']) else "—"
    q_nps = f"{Q['nps']:.2f}" if pd.notna(Q['nps']) else "—"
    y_nps = f"{Y['nps']:.2f}" if pd.notna(Y['nps']) else "—"
    return f"""
ARTSTUDIO Nevsky — Анкеты за неделю {w_start:%d}–{w_end:%d} окт {w_start.year}

Неделя: {w_start:%d.%m}–{w_end:%d.%m}; анкет: {W['total_surveys']}; итоговая: {w_overall}; NPS: {w_nps}.

Текущий месяц ({month_label(w_start)}): анкет {M['total_surveys']}, итоговая {m_overall}, NPS {m_nps};
Текущий квартал ({quarter_label(w_start)}): анкет {Q['total_surveys']}, итоговая {q_overall}, NPS {q_nps};
Текущий год ({year_label(w_start)}): анкет {Y['total_surveys']}, итоговая {y_overall}, NPS {y_nps}.

Параметры (неделя / Текущий месяц / Текущий квартал / Текущий год)
"""

def table_params_block(W_bp: list, M_bp: list, Q_bp: list, Y_bp: list) -> str:
    thead = "<tr><th>Параметр</th><th colspan=2>Неделя</th><th colspan=2>Месяц</th><th colspan=2>Квартал</th><th colspan=2>Год</th></tr>"
    thead += "<tr><th></th><th>Ср.</th><th>Ответы</th><th>Ср.</th><th>Ответы</th><th>Ср.</th><th>Ответы</th><th>Ср.</th><th>Ответы</th></tr>"
    tbody = ""
    for p in [d["param"] for d in W_bp]:
        w = next(bp for bp in W_bp if bp["param"] == p)
        m = next(bp for bp in M_bp if bp["param"] == p)
        q = next(bp for bp in Q_bp if bp["param"] == p)
        y = next(bp for bp in Y_bp if bp["param"] == p)
        name = PARAM_NAMES.get(p, p)
        w_avg = w["avg5"] if p != "nps" else w["nps"]
        m_avg = m["avg5"] if p != "nps" else m["nps"]
        q_avg = q["avg5"] if p != "nps" else q["nps"]
        y_avg = y["avg5"] if p != "nps" else y["nps"]
        w_str = f"{w_avg:.2f}" if pd.notna(w_avg) and p == "nps" else f"{w_avg:.2f} /5" if pd.notna(w_avg) else "—"
        m_str = f"{m_avg:.2f}" if pd.notna(m_avg) and p == "nps" else f"{m_avg:.2f} /5" if pd.notna(m_avg) else "—"
        q_str = f"{q_avg:.2f}" if pd.notna(q_avg) and p == "nps" else f"{q_avg:.2f} /5" if pd.notna(q_avg) else "—"
        y_str = f"{y_avg:.2f}" if pd.notna(y_avg) and p == "nps" else f"{y_avg:.2f} /5" if pd.notna(y_avg) else "—"
        tbody += f"<tr><td>{name}</td><td>{w_str}</td><td>{w['responses']}</td><td>{m_str}</td><td>{m['responses']}</td><td>{q_str}</td><td>{q['responses']}</td><td>{y_str}</td><td>{y['responses']}</td></tr>"
    return f"<table>{thead}{tbody}</table>"

def footnote_block() -> str:
    return """
* Все значения по анкетам отображаются в шкале /5. Внутри расчётов применяется взвешивание по количеству ответов. NPS считается по шкале 1–
5: 1–2 — детракторы, 3–4 — нейтрал, 5 — промоутеры; NPS = %промоутеров − %детракторов.
"""

# ======================
# Charts
# ======================
def plot_radar_params(w_bp: list, m_bp: list, path: str):
    # Implement or leave as is (truncated in original)
    pass

def plot_params_heatmap(hist: pd.DataFrame, path: str):
    pass

def plot_overall_nps_trends(hist: pd.DataFrame, path: str):
    pass

# ======================
# Email
# ======================
def attach_file(msg: MIMEMultipart, path: str):
    if not os.path.exists(path): return
    maintype, subtype = mimetypes.guess_type(path)[0].split('/', 1)
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
    # выберем правильный лист с ответами анкет
    xls = pd.ExcelFile(io.BytesIO(blob))
    
    def choose_surveys_sheet(xls_file: pd.ExcelFile) -> pd.DataFrame:
        # 1) приоритет по имени
        preferred = ["Оценки гостей", "Оценки", "Ответы", "Анкеты", "Responses"]
        for name in xls_file.sheet_names:
            if name.strip().lower() in {p.lower() for p in preferred}:
                return pd.read_excel(xls_file, name)
    
        # 2) эвристика по колонкам
        candidates = ["Дата", "Дата анкетирования", "Комментарий", "Средняя оценка", "№ 1", "№ 2", "№ 3"]
        best_name, best_score = None, -1
        for name in xls_file.sheet_names:
            try:
                probe = pd.read_excel(xls_file, name, nrows=1)
                cols = [str(c) for c in probe.columns]
                score = sum(any(cand.lower() in c.lower() for cand in candidates) for c in cols)
                if score > best_score:
                    best_name, best_score = name, score
            except Exception:
                continue
        # 3) fallback — первый лист
        pick = best_name or xls_file.sheet_names[0]
        return pd.read_excel(xls_file, pick)
    
    df_raw = choose_surveys_sheet(xls)

    norm, agg_week = parse_and_aggregate_weekly(df_raw)  # agg_week: week_key|param|responses|avg5|avg10|promoters|detractors|nps
    upserted = upsert_week(agg_week)
    print(f"[INFO] surveys_weekly: upserted {upserted} rows for {SURVEYS_TAB}")

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

    # 4) Графики
    charts = []
    p1 = "/tmp/surveys_radar.png";     plot_radar_params(W["by_param"], M["by_param"], p1); charts.append(p1)
    p2 = "/tmp/surveys_heatmap.png";   plot_params_heatmap(hist, p2);                      charts.append(p2)
    p3 = "/tmp/surveys_trends.png";    plot_overall_nps_trends(hist, p3);                 charts.append(p3)
    charts = [p for p in charts if p and os.path.exists(p)]

    # 5) Письмо
    subject = f"ARTSTUDIO Nevsky. Анкеты TL: Marketing — неделя {w_start:%d.%m}–{w_end:%d.%m}"
    send_email(subject, html, attachments=charts)


if __name__ == "__main__":
    main()
