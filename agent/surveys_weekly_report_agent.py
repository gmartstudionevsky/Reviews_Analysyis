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
# Charts
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
    import numpy as np
    if week_df is None or week_df.empty or month_df is None or month_df.empty:
        return None

    # собираем значения по параметрам (без NPS)
    def to_map(df):
        mp={}
        for _, r in df.iterrows():
            p = str(r["param"])
            if p == "nps": 
                continue
            mp[p] = float(r["avg5"]) if pd.notna(r["avg5"]) else np.nan
        return mp

    W = to_map(week_df)
    M = to_map(month_df)

    # общий упорядоченный список параметров
    order = [p for p in [
        "overall","fo_checkin","clean_checkin","room_comfort","fo_stay","its_service","hsk_stay","breakfast",
        "atmosphere","location","value","would_return"
    ] if (p in W or p in M)]

    if not order:
        return None

    labels = [PARAM_TITLES.get(p, p) for p in order]
    wvals  = [W.get(p, np.nan) for p in order]
    mvals  = [M.get(p, np.nan) for p in order]

    # замыкаем круг
    labels += labels[:1]
    wvals  += wvals[:1]
    mvals  += mvals[:1]

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7.5, 6.5))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 5)

    ax.plot(angles, wvals, marker="o", linewidth=1, label="Неделя")
    ax.fill(angles, wvals, alpha=0.1)
    ax.plot(angles, mvals, marker="o", linewidth=1, linestyle="--", label="Текущий месяц")
    ax.fill(angles, mvals, alpha=0.1)
    ax.set_title("Анкеты: параметры (Неделя vs Текущий месяц), шкала /5")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout(); plt.savefig(path_png); plt.close(); return path_png

def plot_params_heatmap(history_df: pd.DataFrame, path_png: str):
    """
    Теплокарта: средняя /5 по параметрам за 8 последних недель.
    """
    if history_df is None or history_df.empty:
        return None

    df = history_df.copy()
    for c in ["avg5","responses"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df["param"]!="nps"].copy()
    if df.empty: return None

    # последние 8 недель
    weeks = sorted(df["week_key"].unique(), key=_week_order_key)[-8:]
    df = df[df["week_key"].isin(weeks)].copy()
    if df.empty: return None

    # топ-10 параметров по числу ответов
    top_params = (df.groupby("param")["responses"].sum()
                    .sort_values(ascending=False).head(10).index.tolist())
    df = df[df["param"].isin(top_params)]

    pv = (df.pivot_table(index="param", columns="week_key", values="avg5", aggfunc="mean")
            .reindex(index=top_params, columns=weeks))
    if pv.empty: return None

    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(pv.values, aspect="auto")
    ax.set_yticks(range(len(pv.index))); ax.set_yticklabels([PARAM_TITLES.get(p,p) for p in pv.index])
    ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels(list(pv.columns), rotation=45)
    ax.set_title("Анкеты: средняя /5 по параметрам (8 последних недель)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(path_png); plt.close(); return path_png

def plot_overall_nps_trends(history_df: pd.DataFrame, path_png: str):
    """
    Тренды: Итоговая /5 и NPS по неделям (12 последних) + 4-нед. скользящее среднее.
    """
    if history_df is None or history_df.empty:
        return None

    df = history_df.copy()
    for c in ["avg5","nps"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # фильтруем нужные параметры
    ov = df[df["param"]=="overall"][["week_key","avg5"]]
    npv= df[df["param"]=="nps"][["week_key","nps"]]

    # объединяем уникальные недели
    weeks = sorted(set(ov["week_key"]).union(set(npv["week_key"])), key=_week_order_key)[-12:]
    if not weeks:
        return None

    ov = ov.set_index("week_key").reindex(weeks)
    npv= npv.set_index("week_key").reindex(weeks)

    # скользящие
    ov_roll  = ov["avg5"].rolling(window=4, min_periods=2).mean()
    nps_roll = npv["nps"].rolling(window=4, min_periods=2).mean()

    fig, ax1 = plt.subplots(figsize=(10, 5.0))
    ax1.plot(weeks, ov["avg5"].values, marker="o", label="Итоговая /5")
    ax1.plot(weeks, ov_roll.values, linestyle="--", label="Итоговая /5 (скользящее)")
    ax1.set_ylim(0, 5); ax1.set_ylabel("Итоговая /5")

    ax2 = ax1.twinx()
    ax2.plot(weeks, npv["nps"].values, marker="s", label="NPS", alpha=0.8)
    ax2.plot(weeks, nps_roll.values, linestyle="--", label="NPS (скользящее)", alpha=0.8)
    ax2.set_ylim(-100, 100); ax2.set_ylabel("NPS, п.п.")

    ax1.set_title("Анкеты: тренды Итоговой /5 и NPS (12 недель)")
    ax1.set_xticks(range(len(weeks))); ax1.set_xticklabels(weeks, rotation=45)

    # общая легенда
    lines, labels = [], []
    for ax in (ax1, ax2):
        l, lab = ax.get_legend_handles_labels()
        lines += l; labels += lab
    ax1.legend(lines, labels, loc="upper left")

    plt.tight_layout(); plt.savefig(path_png); plt.close(); return path_png


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
