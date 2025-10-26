# agent/surveys_weekly_report_agent.py
import os
import io
import re
import json
import math
import datetime as dt
from datetime import date
import pandas as pd
import numpy as np

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

import smtplib
import mimetypes
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from surveys_core import (
    parse_and_aggregate_weekly, SURVEYS_TAB, PARAM_ORDER, PARAM_NAMES,
    iso_week_key, iso_week_monday, period_ranges_for_week, aggregate_period
)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly", "https://www.googleapis.com/auth/spreadsheets"]
SA_PATH = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
SA_CONTENT = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON_CONTENT")

if SA_CONTENT and SA_CONTENT.strip().startswith("{"):
    CREDS = Credentials.from_service_account_info(json.loads(SA_CONTENT), scopes=SCOPES)
else:
    CREDS = Credentials.from_service_account_file(SA_PATH, scopes=SCOPES)

DRIVE = build("drive", "v3", credentials=CREDS)
SHEETS = build("sheets", "v4", credentials=CREDS).spreadsheets()

DRIVE_FOLDER_ID = os.environ["DRIVE_FOLDER_ID"]
HISTORY_SHEET_ID = os.environ["SHEETS_HISTORY_ID"]
RECIPIENTS = [e.strip() for e in os.environ.get("RECIPIENTS", "").split(",") if e.strip()]
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASS = os.environ.get("SMTP_PASS")
SMTP_FROM = os.environ.get("SMTP_FROM", SMTP_USER)
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))

WEEKLY_RE = re.compile(r"^Report_(\d{2})-(\d{2})-(\d{4})\.xlsx$", re.IGNORECASE)

def latest_report_from_drive():
    res = DRIVE.files().list(
        q=f"'{DRIVE_FOLDER_ID}' in parents and trashed=false and name contains 'Report_'",
        fields="files(id,name,modifiedTime)",
        pageSize=1000
    ).execute()
    items = res.get("files", [])
    items = sorted(items, key=lambda i: i["modifiedTime"], reverse=True)
    for item in items:
        if WEEKLY_RE.match(item["name"]):
            return item["id"], item["name"]
    raise ValueError("No recent Report file found")

def drive_download(file_id: str) -> bytes:
    req = DRIVE.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
    return buf.getvalue()

def gs_get_df(tab: str) -> pd.DataFrame:
    res = SHEETS.values().get(spreadsheetId=HISTORY_SHEET_ID, range=tab).execute()
    vals = res.get("values", [])
    if not vals:
        return pd.DataFrame()
    return pd.DataFrame(vals[1:], columns=vals[0])

def gs_append(tab: str, rows: List[List]):
    if not rows:
        return
    SHEETS.values().append(
        spreadsheetId=HISTORY_SHEET_ID,
        range=tab,
        valueInputOption="RAW",
        body={"values": rows}
    ).execute()

def upsert_week(agg_week: pd.DataFrame) -> int:
    hist = gs_get_df(SURVEYS_TAB)
    existing = set(zip(hist["week_key"], hist["param"]))
    rows = []
    for _, r in agg_week.iterrows():
        key = (r["week_key"], r["param"])
        if key not in existing:
            row = [
                r["week_key"], r["param"], r["responses"], r["avg5"],
                r.get("promoters"), r.get("detractors"), r.get("nps")
            ]
            rows.append([None if pd.isna(v) else v for v in row])
    if rows:
        gs_append(SURVEYS_TAB, rows)
    return len(rows)

def header_block(w_start: date, w_end: date, W: Dict, M: Dict, Q: Dict, Y: Dict) -> str:
    w_overall = f"{W['overall_avg5']:.2f}" if W['overall_avg5'] is not None else "-"
    m_overall = f"{M['overall_avg5']:.2f}" if M['overall_avg5'] is not None else "-"
    q_overall = f"{Q['overall_avg5']:.2f}" if Q['overall_avg5'] is not None else "-"
    y_overall = f"{Y['overall_avg5']:.2f}" if Y['overall_avg5'] is not None else "-"
    w_nps = f"{W['nps']:.2f}" if W['nps'] is not None else "-"
    m_nps = f"{M['nps']:.2f}" if M['nps'] is not None else "-"
    q_nps = f"{Q['nps']:.2f}" if Q['nps'] is not None else "-"
    y_nps = f"{Y['nps']:.2f}" if Y['nps'] is not None else "-"
    return f"""
ARTSTUDIO Nevsky — Анкеты за неделю {w_start.day}-{w_end.day} окт {w_start.year}

Неделя: {w_start.day}-{w_end.day} окт {w_start.year}; анкет: {W['total_surveys']}; итоговая: {w_overall}; NPS: {w_nps}.

Текущий месяц (октябрь {w_start.year}): анкет {M['total_surveys']}, итоговая {m_overall}, NPS {m_nps};
Текущий квартал (IV кв. {w_start.year}): анкет {Q['total_surveys']}, итоговая {q_overall}, NPS {q_nps};
Текущий год ({w_start.year}): анкет {Y['total_surveys']}, итоговая {y_overall}, NPS {y_nps}.
"""

def table_block(W: Dict, M: Dict, Q: Dict, Y: Dict) -> str:
    html = "<table><tr><th>Параметр</th><th>Ср.</th><th>Ответы</th><th>Ср.</th><th>Ответы</th><th>Ср.</th><th>Ответы</th><th>Ср.</th><th>Ответы</th></tr>"
    for p in PARAM_ORDER + ["nps_1_5"]:
        if p == "nps_1_5":
            p = "nps"
            w_val = W["by_param"][p]["nps"]
            m_val = M["by_param"][p]["nps"]
            q_val = Q["by_param"][p]["nps"]
            y_val = Y["by_param"][p]["nps"]
            fmt = ".2f"
        else:
            w_val = W["by_param"].get(p, {})["avg5"]
            m_val = M["by_param"].get(p, {})["avg5"]
            q_val = Q["by_param"].get(p, {})["avg5"]
            y_val = Y["by_param"].get(p, {})["avg5"]
            fmt = ".2f"
        w_str = f"{w_val:{fmt}}" if w_val is not None else "-"
        m_str = f"{m_val:{fmt}}" if m_val is not None else "-"
        q_str = f"{q_val:{fmt}}" if q_val is not None else "-"
        y_str = f"{y_val:{fmt}}" if y_val is not None else "-"
        w_resp = W["by_param"].get(p, {})["responses"]
        m_resp = M["by_param"].get(p, {})["responses"]
        q_resp = Q["by_param"].get(p, {})["responses"]
        y_resp = Y["by_param"].get(p, {})["responses"]
        name = PARAM_NAMES.get(p, p)
        html += f"<tr><td>{name}</td><td>{w_str}</td><td>{w_resp}</td><td>{m_str}</td><td>{m_resp}</td><td>{q_str}</td><td>{q_resp}</td><td>{y_str}</td><td>{y_resp}</td></tr>"
    html += "</table>"
    return html

def footnote_block() -> str:
    return "* Все значения по анкетам отображаются в шкале /5. NPS считается по шкале 1-5: 1-2 — детракторы, 3-4 — нейтрал, 5 — промоутеры; NPS = %промоутеров - %детракторов."

def plot_radar_params(W_bp: Dict, M_bp: Dict, path: str):
    params = [p for p in PARAM_ORDER if p != "nps_1_5"]
    w_avgs = [W_bp.get(p, {})["avg5"] or 0 for p in params]
    m_avgs = [M_bp.get(p, {})["avg5"] or 0 for p in params]
    labels = [PARAM_NAMES[p] for p in params]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    w_avgs += w_avgs[:1]
    m_avgs += m_avgs[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, w_avgs, color='blue', alpha=0.25)
    ax.plot(angles, w_avgs, color='blue', linewidth=2, label='Week')
    ax.fill(angles, m_avgs, color='green', alpha=0.25)
    ax.plot(angles, m_avgs, color='green', linewidth=2, label='Month')
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.legend()
    plt.savefig(path)
    plt.close()

def plot_heatmap(hist: pd.DataFrame, path: str):
    params = [p for p in PARAM_ORDER if p != "nps_1_5"]
    pivot = hist[hist["param"].isin(params)].pivot(index="week_key", columns="param", values="avg5")
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = cm.get_cmap('YlGn')
    norm = Normalize(vmin=1, vmax=5)
    im = ax.imshow(pivot, cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([PARAM_NAMES[p] for p in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    plt.colorbar(im)
    plt.savefig(path)
    plt.close()

def plot_trends(hist: pd.DataFrame, path: str):
    overall = hist[hist["param"] == "overall"]
    nps = hist[hist["param"] == "nps"]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(overall["week_key"], overall["avg5"], color='blue', label='Overall Avg')
    ax1.set_ylabel('Average /5')
    ax2 = ax1.twinx()
    ax2.plot(nps["week_key"], nps["nps"], color='green', label='NPS')
    ax2.set_ylabel('NPS')
    fig.legend()
    plt.savefig(path)
    plt.close()

def send_email(subject: str, html: str, attachments: List[str]):
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM
    msg["To"] = ", ".join(RECIPIENTS)
    msg.attach(MIMEText(html, "html"))
    for path in attachments:
        if os.path.exists(path):
            with open(path, "rb") as f:
                part = MIMEImage(f.read())
                part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(path)}")
                msg.attach(part)
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_FROM, RECIPIENTS, msg.as_string())

def main():
    fid, name = latest_report_from_drive()
    blob = drive_download(fid)
    xls = pd.ExcelFile(io.BytesIO(blob))
    sheet_names = xls.sheet_names
    preferred = ["Оценки гостей", "Оценки", "Ответы", "Анкеты", "Responses"]
    sheet = next((s for s in sheet_names if s.lower() in [p.lower() for p in preferred]), sheet_names[0])
    df_raw = pd.read_excel(xls, sheet)
    _, agg_week = parse_and_aggregate_weekly(df_raw)
    upsert_week(agg_week)
    hist = gs_get_df(SURVEYS_TAB)
    wk_key = agg_week["week_key"].iloc[0]
    w_start = iso_week_monday(wk_key)
    w_end = w_start + timedelta(days=6)
    ranges = period_ranges_for_week(w_start)
    W = aggregate_period(hist, ranges["week"]["start"], ranges["week"]["end"])
    M = aggregate_period(hist, ranges["mtd"]["start"], ranges["mtd"]["end"])
    Q = aggregate_period(hist, ranges["qtd"]["start"], ranges["qtd"]["end"])
    Y = aggregate_period(hist, ranges["ytd"]["start"], ranges["ytd"]["end"])
    head = header_block(w_start, w_end, W, M, Q, Y)
    table = table_block(W, M, Q, Y)
    footnote = footnote_block()
    html = head + table + footnote
    charts = []
    p1 = "/tmp/surveys_radar.png"
    plot_radar_params(W["by_param"], M["by_param"], p1)
    charts.append(p1)
    p2 = "/tmp/surveys_heatmap.png"
    plot_heatmap(hist, p2)
    charts.append(p2)
    p3 = "/tmp/surveys_trends.png"
    plot_trends(hist, p3)
    charts.append(p3)
    subject = f"ARTSTUDIO Nevsky. Анкеты TL: Marketing — неделя {w_start.day}.{w_start.month}-{w_end.day}.{w_end.month}"
    send_email(subject, html, charts)

if __name__ == "__main__":
    main()
