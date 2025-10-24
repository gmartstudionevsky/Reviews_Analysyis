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

# ================
# Sheets helpers
# ================
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
            body={"values":[header]}
        ).execute()

def gs_get_df(tab: str, a1: str) -> pd.DataFrame:
    try:
        res = SHEETS.values().get(spreadsheetId=HISTORY_SHEET_ID, range=f"{tab}!{a1}").execute()
        vals = res.get("values", [])
        if len(vals) <= 1:
            return pd.DataFrame()
        df = pd.DataFrame(vals[1:], columns=vals[0])
        
        # Конвертация: ',' → '.', '' → NaN, to_numeric
        numeric_cols = ['responses', 'avg5', 'avg10', 'promoters', 'detractors', 'nps']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '.').replace('', np.nan).replace('nan', np.nan, case=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
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

# ==================
# Core functions
# ==================
HEADER = ["week_key","param","responses","avg5","avg10","promoters","detractors","nps"]

def load_existing_keys() -> set[tuple[str,str]]:
    df = gs_get_df(SURVEYS_TAB, "A:H")
    if df.empty: return set()
    return set(zip(df.get("week_key",[]), df.get("param",[])))

def rows_from_agg(df: pd.DataFrame) -> list[list]:
    rows = []
    for _, r in df.iterrows():
        def get_float(val):
            if pd.isna(val) or (isinstance(val, str) and val.strip() == ''):
                return None
            try:
                str_val = str(val).replace(',', '.').lower()
                if 'nan' in str_val:
                    return None
                f = float(str_val)
                if math.isnan(f):
                    return None
                return f
            except ValueError:
                return None

        def get_int(val):
            if pd.isna(val) or (isinstance(val, str) and val.strip() == ''):
                return None
            try:
                str_val = str(val).replace(',', '.').lower()
                if 'nan' in str_val:
                    return None
                return int(float(str_val))
            except ValueError:
                return None

        row = [
            str(r["week_key"]),
            str(r["param"]),
            get_int(r["responses"]) or 0,
            get_float(r["avg5"]),
            get_float(r["avg10"]),
            get_int(r["promoters"]),
            get_int(r["detractors"]),
            get_float(r["nps"]),
        ]
        rows.append(row)
    return rows

def append_week_if_needed(agg_week: pd.DataFrame) -> int:
    ensure_tab(HISTORY_SHEET_ID, SURVEYS_TAB, HEADER)
    existing = load_existing_keys()
    need = []
    for _, r in agg_week.iterrows():
        k = (str(r["week_key"]), str(r["param"]))
        if k not in existing:
            need.append(r)
    if not need:
        return 0
    rows = rows_from_agg(pd.DataFrame(need))
    gs_append(SURVEYS_TAB, "A2", rows)
    return len(rows)

def surveys_aggregate_period(df: pd.DataFrame, start: date, end: date) -> dict:
    """
    Aggregate survey data for a given period from historical DataFrame.
    
    Filters weeks where the Monday is between start and end (inclusive).
    Then groups by param and computes:
    - sum(responses)
    - weighted avg for avg5 and avg10 (weighted by responses)
    - For NPS: sum(promoters), sum(detractors), compute nps = (prom - det) / total_resp * 100
    
    Returns dict with:
    - 'by_param': pd.DataFrame with columns ['param', 'responses', 'avg5', 'avg10', 'promoters', 'detractors', 'nps']
    - 'totals': dict with 'responses' (from overall), 'avg5', 'avg10', 'nps'
    """
    if df.empty:
        empty_df = pd.DataFrame(columns=['param', 'responses', 'avg5', 'avg10', 'promoters', 'detractors', 'nps'])
        return {
            'by_param': empty_df,
            'totals': {
                'responses': 0,
                'avg5': np.nan,
                'avg10': np.nan,
                'nps': np.nan
            }
        }
    
    # Copy and ensure numeric types
    df = df.copy()
    numeric_cols = ['responses', 'avg5', 'avg10', 'promoters', 'detractors', 'nps']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.').replace('', np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Filter weeks in period
    df['monday'] = df['week_key'].map(iso_week_monday)
    filtered = df[(df['monday'] >= start) & (df['monday'] <= end)].drop(columns=['monday'])
    
    if filtered.empty:
        empty_df = pd.DataFrame(columns=['param', 'responses', 'avg5', 'avg10', 'promoters', 'detractors', 'nps'])
        return {
            'by_param': empty_df,
            'totals': {
                'responses': 0,
                'avg5': np.nan,
                'avg10': np.nan,
                'nps': np.nan
            }
        }
    
    # Separate nps and other params
    nps_df = filtered[filtered['param'] == 'nps']
    params_df = filtered[filtered['param'] != 'nps']
    
    # Aggregate params
    if not params_df.empty:
        params_agg = params_df.groupby('param', group_keys=False).apply(
            lambda g: pd.Series({
                'responses': g['responses'].sum(),
                'avg5': np.average(g['avg5'], weights=g['responses']) if g['responses'].sum() > 0 else np.nan,
                'avg10': np.average(g['avg10'], weights=g['responses']) if g['responses'].sum() > 0 else np.nan,
                'promoters': np.nan,
                'detractors': np.nan,
                'nps': np.nan
            }),
            include_groups=False
        ).reset_index()
    else:
        params_agg = pd.DataFrame(columns=['param', 'responses', 'avg5', 'avg10', 'promoters', 'detractors', 'nps'])
    
    # Aggregate nps
    if not nps_df.empty:
        total_resp = nps_df['responses'].sum()
        promoters = nps_df['promoters'].sum()
        detractors = nps_df['detractors'].sum()
        nps = ((promoters - detractors) / total_resp * 100) if total_resp > 0 else np.nan
        nps_row = pd.DataFrame([{
            'param': 'nps',
            'responses': total_resp,
            'avg5': np.nan,
            'avg10': np.nan,
            'promoters': promoters,
            'detractors': detractors,
            'nps': nps
        }])
    else:
        nps_row = pd.DataFrame([{
            'param': 'nps',
            'responses': 0,
            'avg5': np.nan,
            'avg10': np.nan,
            'promoters': np.nan,
            'detractors': np.nan,
            'nps': np.nan
        }])
    
    # Combine
    by_param = pd.concat([params_agg, nps_row], ignore_index=True)
    
    # Extract totals
    overall_row = by_param[by_param['param'] == 'overall']
    responses = int(overall_row['responses'].iloc[0]) if not overall_row.empty else 0
    avg5 = overall_row['avg5'].iloc[0] if not overall_row.empty else np.nan
    avg10 = overall_row['avg10'].iloc[0] if not overall_row.empty else np.nan
    nps_value = nps_row['nps'].iloc[0]
    
    # Order params as per PARAM_ORDER + 'nps'
    from agent.surveys_core import PARAM_ORDER  # Assuming available
    order = [p for p in PARAM_ORDER if p != 'nps_1_5'] + ['nps']
    by_param['param'] = pd.Categorical(by_param['param'], categories=order, ordered=True)
    by_param = by_param.sort_values('param').reset_index(drop=True)
    
    return {
        'by_param': by_param,
        'totals': {
            'responses': responses,
            'avg5': avg5,
            'avg10': avg10,
            'nps': nps_value
        }
    }

# ==================
# Report blocks
# ==================
def fmt_int(x: int) -> str:
    return f"{x:,}".replace(",", " ")

def fmt_avg5(x: float) -> str:
    if math.isnan(x):
        return "—"
    return f"{x:.2f} /5"

def fmt_nps(x: float) -> str:
    if math.isnan(x):
        return "—"
    return f"{x:.2f}"

def header_block(w_start: date, w_end: date, W: dict, M: dict, Q: dict, Y: dict) -> str:
    wl = week_label(w_start)
    ml = month_label(w_start)
    ql = quarter_label(w_start)
    yl = year_label(w_start)
    return (
        f"<h3>ARTSTUDIO Nevsky — Анкеты за неделю {w_start:%d–%m.%Y}</h3>"
        f"<p>Неделя: <b>{wl}</b>; анкет: <b>{fmt_int(W['totals']['responses'])}</b>; "
        f"итоговая: <b>{fmt_avg5(W['totals']['avg5'])}</b>; NPS: <b>{fmt_nps(W['totals']['nps'])}</b>.</p>"
        f"<p>Текущий месяц ({ml}): анкет {fmt_int(M['totals']['responses'])}, итоговая {fmt_avg5(M['totals']['avg5'])}, NPS {fmt_nps(M['totals']['nps'])};</p>"
        f"<p>Текущий квартал ({ql}): анкет {fmt_int(Q['totals']['responses'])}, итоговая {fmt_avg5(Q['totals']['avg5'])}, NPS {fmt_nps(Q['totals']['nps'])};</p>"
        f"<p>Текущий год ({yl}): анкет {fmt_int(Y['totals']['responses'])}, итоговая {fmt_avg5(Y['totals']['avg5'])}, NPS {fmt_nps(Y['totals']['nps'])}.</p>"
    )

def table_params_block(W: pd.DataFrame, M: pd.DataFrame, Q: pd.DataFrame, Y: pd.DataFrame) -> str:
    # Assume PARAM_ORDER from surveys_core
    from agent.surveys_core import PARAM_ORDER
    params = [p for p in PARAM_ORDER if p != 'nps_1_5'] + ['nps']
    html = "<table><tr><th>Параметр</th><th>Неделя</th><th>Месяц</th><th>Квартал</th><th>Год</th></tr>"
    html += "<tr><th></th><th>Ср. / Ответы</th><th>Ср. / Ответы</th><th>Ср. / Ответы</th><th>Ср. / Ответы</th></tr>"
    for p in params:
        w_row = W[W['param'] == p].iloc[0] if not W[W['param'] == p].empty else None
        m_row = M[M['param'] == p].iloc[0] if not M[M['param'] == p].empty else None
        q_row = Q[Q['param'] == p].iloc[0] if not Q[Q['param'] == p].empty else None
        y_row = Y[Y['param'] == p].iloc[0] if not Y[Y['param'] == p].empty else None
        
        if p == 'nps':
            w_val = fmt_nps(w_row['nps']) if w_row is not None else '—'
            m_val = fmt_nps(m_row['nps']) if m_row is not None else '—'
            q_val = fmt_nps(q_row['nps']) if q_row is not None else '—'
            y_val = fmt_nps(y_row['nps']) if y_row is not None else '—'
        else:
            w_val = fmt_avg5(w_row['avg5']) if w_row is not None else '—'
            m_val = fmt_avg5(m_row['avg5']) if m_row is not None else '—'
            q_val = fmt_avg5(q_row['avg5']) if q_row is not None else '—'
            y_val = fmt_avg5(y_row['avg5']) if y_row is not None else '—'
        
        w_resp = fmt_int(w_row['responses']) if w_row is not None else '0'
        m_resp = fmt_int(m_row['responses']) if m_row is not None else '0'
        q_resp = fmt_int(q_row['responses']) if q_row is not None else '0'
        y_resp = fmt_int(y_row['responses']) if y_row is not None else '0'
        
        param_name = p.capitalize()  # Customize as needed
        html += f"<tr><td>{param_name}</td><td>{w_val} / {w_resp}</td><td>{m_val} / {m_resp}</td><td>{q_val} / {q_resp}</td><td>{y_val} / {y_resp}</td></tr>"
    html += "</table>"
    return html

def footnote_block() -> str:
    return "<p><small>Данные из анкет TL: Marketing. NPS рассчитан по шкале 1-5 (5: promoters, 1-2: detractors).</small></p>"

# ==================
# Plot functions (stubbed for completeness; customize as needed)
# ==================
def plot_radar_params(w_df: pd.DataFrame, m_df: pd.DataFrame, path: str):
    # Stub: Implement radar chart
    plt.figure()
    plt.savefig(path)
    plt.close()

def plot_params_heatmap(hist: pd.DataFrame, path: str):
    # Stub: Implement heatmap
    plt.figure()
    plt.savefig(path)
    plt.close()

def plot_overall_nps_trends(hist: pd.DataFrame, path: str):
    # Stub: Implement trends
    plt.figure()
    plt.savefig(path)
    plt.close()

def attach_file(msg: MIMEMultipart, path: str):
    if not os.path.exists(path):
        return
    ctype, encoding = mimetypes.guess_type(path)
    if ctype is None or encoding is not None:
        ctype = 'application/octet-stream'
    maintype, subtype = ctype.split('/', 1)
    with open(path, 'rb') as fp:
        att = MIMEBase(maintype, subtype)
        att.set_payload(fp.read())
    encoders.encode_base64(att)
    att.add_header('Content-Disposition', 'attachment', filename=os.path.basename(path))
    msg.attach(att)

def send_email(subject: str, html_body: str, attachments: list[str] = None):
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
