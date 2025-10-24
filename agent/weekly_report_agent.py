# agent/weekly_report_agent.py
# Еженедельный агент: отчёт «Неделя vs MTD/QTD/YTD» + источники + темы + графики
import os, io, re, json, math, datetime as dt
from datetime import date
import pandas as pd
import numpy as np

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

import smtplib, mimetypes
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Графики (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- наше ядро метрик / периодов ---
from agent.metrics_core import (
    HISTORY_TAB, TOPICS_TAB, SOURCES_TAB, SURVEYS_TAB,
    iso_week_monday, week_range_for_monday,
    period_ranges_for_week, prev_period_ranges,
    aggregate_weeks_from_history, deltas_week_vs_period,
    role_of_week_in_period, sources_summary_for_periods,
    week_label, month_label, quarter_label, year_label
)

# ======================================
# ENV / Google API clients / constants
# ======================================
SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/spreadsheets",
]
CREDS = Credentials.from_service_account_file(
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"], scopes=SCOPES
)
DRIVE  = build("drive",  "v3", credentials=CREDS)
SHEETS = build("sheets", "v4", credentials=CREDS).spreadsheets()

DRIVE_FOLDER_ID   = os.environ["DRIVE_FOLDER_ID"]
HISTORY_SHEET_ID  = os.environ["SHEETS_HISTORY_ID"]
RECIPIENTS        = [e.strip() for e in os.environ["RECIPIENTS"].split(",") if e.strip()]
SMTP_USER         = os.environ["SMTP_USER"]
SMTP_PASS         = os.environ["SMTP_PASS"]
SMTP_FROM         = os.environ.get("SMTP_FROM", SMTP_USER)
SMTP_HOST         = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT         = int(os.environ.get("SMTP_PORT", 587))

# Ожидаемая схема файла отзывов
EXPECTED = ["Дата","Рейтинг","Источник","Автор","Код языка","Текст отзыва","Наличие ответа"]
RENAMES  = {"Дата":"Дата","Рейтинг":"Рейтинг","Источник":"Источник","Автор":"Автор",
            "Код языка":"Код языка","Текст отзыва":"Текст","Наличие ответа":"Ответ"}

# ====================
# Utils: Drive/Sheets
# ====================
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

def latest_reviews_file_from_drive() -> tuple[str,str,date]:
    regex = re.compile(r"^Reviews_\d{2}-\d{2}-\d{4}\.xls(x)?$", re.IGNORECASE)
    res = DRIVE.files().list(
        q=f"'{DRIVE_FOLDER_ID}' in parents and trashed=false",
        fields="files(id,name,modifiedTime)",
        pageSize=1000
    ).execute()
    items = res.get("files", [])
    def parse_date_from_name(name: str):
        m = re.search(r"^Reviews_(\d{2})-(\d{2})-(\d{4})\.xls(x)?$", name, re.IGNORECASE)
        if not m: return None
        dd, mm, yyyy = map(int, m.groups()[:3])
        return date(yyyy, mm, dd)
    items = [(i["id"], i["name"], parse_date_from_name(i["name"])) for i in items if regex.match(i["name"])]
    if not items:
        raise RuntimeError("В папке нет файлов вида Reviews_dd-mm-yyyy.xls(x).")
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

# ===============
# IO: Reviews XLS
# ===============
def normalize_to10(series: pd.Series) -> pd.Series:
    num = pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False).str.extract(r"([-+]?\d*\.?\d+)")[0],
        errors="coerce"
    )
    vmax, vmin = num.max(skipna=True), num.min(skipna=True)
    if pd.isna(vmax): return num
    if vmax <= 5: return num*2
    if vmax <= 10: return num
    if vmax <= 100 and vmin >= 0: return num/10
    if vmax <= 1: return num*10
    return num.clip(upper=10)

def load_reviews_df(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)  # .xls/.xlsx
    low = {str(c).strip().lower(): c for c in df.columns}
    ren = {}
    for k in EXPECTED:
        hit = low.get(k.lower())
        if hit: ren[hit] = RENAMES[k]
        else:
            cand = next((c for lc,c in low.items() if k.split()[0].lower() in lc), None)
            if cand: ren[cand] = RENAMES[k]
    df = df.rename(columns=ren)
    missing = [v for v in RENAMES.values() if v not in df.columns]
    if missing: raise RuntimeError(f"Нет ожидаемых колонок: {missing}")
    return df[[*RENAMES.values()]]

# ===============
# Sentiment/topics
# ===============
NEG_CUES = re.compile(
    r"\bне\b|\bнет\b|проблем|ошиб|задерж|долг|ждал|ждали|"
    r"сложн|плохо|неудоб|путан|непонят|неясн|не приш|не сработ|не открыл|не откры|скромн|бедн|мало",
    re.IGNORECASE
)
POS_LEX = re.compile(r"отличн|прекрасн|замечат|идеальн|классн|любезн|комфортн|удобн|чисто|тихо|вкусн|быстро", re.IGNORECASE)
NEG_LEX = re.compile(r"ужасн|плох|грязн|громк|холодн|жарко|слаб|плохо|долго|скучн|бедн|мало|дорог", re.IGNORECASE)

TOPIC_PATTERNS = {
    "Локация":        [r"расположен|расположение|рядом|вокзал|невск"],
    "Персонал":       [r"персонал|сотрудник|администрат|ресепшен|ресепшн|менеджер"],
    "Чистота":        [r"чисто|уборк|бель[её]|пятн|полотенц|постель"],
    "Номер/оснащение":[r"номер|кухн|посудомо|стирал|плита|чайник|фен|сейф"],
    "Сантехника/вода":[r"душ|слив|кран|смесител|бойлер|водонагрев|давлен|температур|канализац|засор"],
    "AC/шум":         [r"кондицион|вентиляц|шум|тихо|громк"],
    "Завтраки":       [r"завтрак|ланч-?бокс|ресторан"],
    "Чек-ин/вход":    [r"заселен|заезд|чек-?ин|вход|инструкц|селф|код|ключ|карта"],
    "Wi-Fi/интернет": [r"wi-?fi|wifi|вай-?фай|интернет|парол"],
    "Цена/ценность":  [r"цена|стоимост|дорог|дешев|соотношен.*цена"],
}

def sentiment_sign(text: str, rating10: float):
    t = str(text)
    pos = len(POS_LEX.findall(t))
    neg = len(NEG_LEX.findall(t)) + len(NEG_CUES.findall(t))
    if neg - pos >= 2: return "negative"
    if pos - neg >= 1 and neg == 0: return "positive"
    if rating10 is not None and not (isinstance(rating10, float) and math.isnan(rating10)):
        if rating10 < 6: return "negative"
        if rating10 >= 8: return "positive"
        return "neutral"
    return "neutral"

def sentence_split(text: str):
    parts = re.split(r"(?<=[\.\!\?\n])\s+", str(text).strip())
    return [p.strip() for p in parts if len(p.strip())>=12]

def classify_topics(text: str):
    tl = str(text).lower()
    hits = set()
    for tp, pats in TOPIC_PATTERNS.items():
        for p in pats:
            if re.search(p, tl, re.IGNORECASE):
                hits.add(tp); break
    return hits

def summarize_topics(wk_df: pd.DataFrame):
    if wk_df.empty:
        return pd.DataFrame(columns=["topic","mentions","share","avg10","neg_text_share"]), {}
    rows=[]; quotes={}
    total = len(wk_df)
    for idx, r in wk_df.iterrows():
        t = str(r["Текст"])
        found = classify_topics(t)
        if not found: continue
        for tp in found:
            rows.append({"topic": tp, "idx": idx, "rating10": r["_rating10"], "sent": sentiment_sign(t, r["_rating10"])})
            for s in sentence_split(t):
                if any(re.search(p, s, re.IGNORECASE) for p in TOPIC_PATTERNS[tp]):
                    qm = quotes.setdefault(tp, {"pos": None, "neg": None})
                    sg = sentiment_sign(s, r["_rating10"])
                    if sg=="positive" and not qm["pos"]: qm["pos"]=s
                    if sg=="negative" and not qm["neg"]: qm["neg"]=s
    if not rows:
        return pd.DataFrame(columns=["topic","mentions","share","avg10","neg_text_share"]), quotes
    tmp = pd.DataFrame(rows)
    agg = (tmp.groupby("topic")
             .agg(mentions=("topic","size"),
                  avg10=("rating10","mean"),
                  neg_text_share=("sent", lambda s: (s=="negative").mean()))
             .reset_index())
    agg["share"]          = (agg["mentions"]/total*100).round(1)
    agg["avg10"]          = agg["avg10"].round(2)
    agg["neg_text_share"] = (agg["neg_text_share"]*100).round(1)
    return agg.sort_values("mentions", ascending=False), quotes

# ===================
# Week calc & append
# ===================
def last_week_range(today: date):
    monday = today - dt.timedelta(days=today.weekday())
    end = monday - dt.timedelta(days=1)
    start = end - dt.timedelta(days=6)
    return start, end

def analyze_week(df: pd.DataFrame, start: date, end: date):
    df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce").dt.date
    wk = df[(df["Дата"]>=start) & (df["Дата"]<=end)].copy()
    wk["_rating10"] = normalize_to10(wk["Рейтинг"])
    wk["_sent"] = wk.apply(lambda r: sentiment_sign(r["Текст"], r["_rating10"]), axis=1)
    agg = {
        "reviews": len(wk),
        "avg10": round(wk["_rating10"].mean(),2) if len(wk) else None,
        "pos": int((wk["_sent"]=="positive").sum()),
        "neu": int((wk["_sent"]=="neutral").sum()),
        "neg": int((wk["_sent"]=="negative").sum()),
    }
    return wk, agg

def ensure_storage_tabs():
    ensure_tab(HISTORY_SHEET_ID, HISTORY_TAB, ["period_type","period_key","reviews","avg10","pos","neu","neg"])
    ensure_tab(HISTORY_SHEET_ID, SOURCES_TAB, ["week_key","source","reviews","avg10","pos","neu","neg"])
    ensure_tab(HISTORY_SHEET_ID, TOPICS_TAB,   ["week_key","topic","mentions","share_pct","avg10","neg_text_share_pct","by_channel_json"])

def append_history_week(week_start: date, agg: dict):
    iso_y, iso_w, _ = week_start.isocalendar()
    week_key = f"{iso_y}-W{iso_w}"
    hist = gs_get_df(HISTORY_TAB, "A:G")
    if len(hist[(hist["period_type"]=="week") & (hist["period_key"]==week_key)])==0:
        rows = [["week", week_key, agg["reviews"], agg["avg10"], agg["pos"], agg["neu"], agg["neg"]]]
        gs_append(HISTORY_TAB, "A:G", rows)
    return week_key

def append_sources_week(week_key: str, wk_df: pd.DataFrame):
    exist = gs_get_df(SOURCES_TAB, "A:G")
    have = set(zip(exist.get("week_key",[]), exist.get("source",[]))) if not exist.empty else set()
    rows=[]
    for src, sub in wk_df.groupby("Источник"):
        if (week_key, src) in have: continue
        n = int(len(sub))
        avg10 = round(float(sub["_rating10"].mean()), 2) if n else None
        pos = int((sub["_sent"]=="positive").sum())
        neu = int((sub["_sent"]=="neutral").sum())
        neg = int((sub["_sent"]=="negative").sum())
        rows.append([week_key, src, n, avg10, pos, neu, neg])
    gs_append(SOURCES_TAB, "A:G", rows)

# =============
# HTML builders
# =============
def fmt_pct(x):   return "—" if x is None else f"{x:.1f}%"
def fmt_avg(x):   return "—" if x is None else f"{x:.2f}"
def fmt_pp(x):    return "—" if x is None else f"{x:+.2f} п.п."
def fmt_int(x):   return "—" if x is None else str(int(x))

def header_block(week_start: date, labels: dict, week_agg: dict, mtd_agg: dict, qtd_agg: dict, ytd_agg: dict, deltas: dict):
    s, e = week_start, week_start + dt.timedelta(days=6)
    return f"""
    <h2>ARTSTUDIO Nevsky — отзывы за неделю {week_label(s, e)}</h2>
    <p><b>Неделя:</b> {week_label(s, e)} — {week_agg['reviews']} отзывов; средняя <b>{fmt_avg(week_agg['avg10'])}/10</b>;
       позитив {fmt_pct(100.0*week_agg['pos']/week_agg['reviews']) if week_agg['reviews'] else '—'},
       негатив {fmt_pct(100.0*week_agg['neg']/week_agg['reviews']) if week_agg['reviews'] else '—'}.</p>
    <p><b>Сравнение с периодами (на конец недели):</b><br>
       MTD ({labels['mtd']}): ср. {fmt_avg(mtd_agg['avg10'])}/10, позитив {fmt_pct(mtd_agg['pos_share'])}, негатив {fmt_pct(mtd_agg['neg_share'])}; 
       Δнедели → MTD: ср. {fmt_avg(deltas['mtd']['avg10_delta'])}, Δ+ {fmt_pp(deltas['mtd']['pos_delta_pp'])}, Δ− {fmt_pp(deltas['mtd']['neg_delta_pp'])}, вклад недели {fmt_pct(deltas['mtd']['week_share_pct'])}.<br>
       QTD ({labels['qtd']}): ср. {fmt_avg(qtd_agg['avg10'])}/10, позитив {fmt_pct(qtd_agg['pos_share'])}, негатив {fmt_pct(qtd_agg['neg_share'])};
       Δнедели → QTD: ср. {fmt_avg(deltas['qtd']['avg10_delta'])}, Δ+ {fmt_pp(deltas['qtd']['pos_delta_pp'])}, Δ− {fmt_pp(deltas['qtd']['neg_delta_pp'])}, вклад {fmt_pct(deltas['qtd']['week_share_pct'])}.<br>
       YTD ({labels['ytd']}): ср. {fmt_avg(ytd_agg['avg10'])}/10, позитив {fmt_pct(ytd_agg['pos_share'])}, негатив {fmt_pct(ytd_agg['neg_share'])};
       Δнедели → YTD: ср. {fmt_avg(deltas['ytd']['avg10_delta'])}, Δ+ {fmt_pp(deltas['ytd']['pos_delta_pp'])}, Δ− {fmt_pp(deltas['ytd']['neg_delta_pp'])}, вклад {fmt_pct(deltas['ytd']['week_share_pct'])}.
    </p>
    """

def topics_table_html(topics_df: pd.DataFrame, prev_topics_df: pd.DataFrame | None):
    prev = prev_topics_df.set_index("topic") if (prev_topics_df is not None and not prev_topics_df.empty) else None
    rows=[]
    if topics_df.empty:
        return "<h3>Темы недели</h3><p>Нет данных</p>"
    for _, r in topics_df.head(10).iterrows():
        d_m = d_n = ""
        if prev is not None and r["topic"] in prev.index:
            dm = int(r["mentions"] - prev.loc[r["topic"], "mentions"])
            dn = float(r["neg_text_share"] - prev.loc[r["topic"], "neg_text_share"])
            d_m = f" ({'+' if dm>=0 else ''}{dm})"
            d_n = f" ({'+' if dn>=0 else ''}{round(dn,1)} п.п.)"
        rows.append(
            f"<tr><td>{r['topic']}</td><td>{int(r['mentions'])}{d_m}</td>"
            f"<td>{r['share']}%</td><td>{fmt_avg(r['avg10'])}</td><td>{r['neg_text_share']}%{d_n}</td></tr>"
        )
    return ("<h3>Темы недели</h3>"
            "<table border='1' cellspacing='0' cellpadding='6'>"
            "<tr><th>Тема</th><th>Упоминания</th><th>Доля</th><th>Средняя /10</th><th>Негатив по тексту</th></tr>"
            + "".join(rows) + "</table>")

def quotes_block(quotes: dict):
    parts=[]
    for tp, q in quotes.items():
        line=[]
        if q.get("pos"): line.append(f"<b>+</b> «{q['pos']}»")
        if q.get("neg"): line.append(f"<b>–</b> «{q['neg']}»")
        if line: parts.append(f"<p><b>{tp}:</b> " + " / ".join(line) + "</p>")
    return ("<h3>Цитаты</h3>" + "".join(parts)) if parts else ""

def sources_table_block(summ: dict):
    """Основная таблица по источникам: Неделя/MTD/QTD/YTD (avg/rev/pos%/neg%)."""
    def to_map(df):
        return {r["source"]:{
            "reviews": int(r["reviews"]) if pd.notna(r["reviews"]) else 0,
            "avg10": r["avg10"],
            "pos": r.get("pos_share"), "neg": r.get("neg_share")
        } for _, r in (df if df is not None else pd.DataFrame(columns=["source"])).iterrows()}
    W = to_map(summ["week"]); M = to_map(summ["mtd"]); Q = to_map(summ["qtd"]); Y = to_map(summ["ytd"])
    all_sources = sorted(set(list(W.keys())|set(M.keys())|set(Q.keys())|set(Y.keys())))
    rows=[]
    for s in all_sources:
        def cell(d):
            if s not in d: return "<td>—</td><td>—</td><td>—</td><td>—</td>"
            v = d[s]; return f"<td>{fmt_avg(v['avg10'])}</td><td>{fmt_int(v['reviews'])}</td><td>{fmt_pct(v['pos'])}</td><td>{fmt_pct(v['neg'])}</td>"
        rows.append(
            f"<tr><td><b>{s}</b></td>{cell(W)}{cell(M)}{cell(Q)}{cell(Y)}</tr>"
        )
    return f"""
    <h3>Источники (по периодам)</h3>
    <p>Неделя: {summ['labels']['week']} • Месяц (MTD): {summ['labels']['mtd']} • Квартал (QTD): {summ['labels']['qtd']} • Год (YTD): {summ['labels']['ytd']}</p>
    <table border='1' cellspacing='0' cellpadding='6'>
      <tr>
        <th rowspan="2">Источник</th>
        <th colspan="4">Неделя</th>
        <th colspan="4">Месяц (MTD)</th>
        <th colspan="4">Квартал (QTD)</th>
        <th colspan="4">Год (YTD)</th>
      </tr>
      <tr>
        <th>Ср./10</th><th>Отз.</th><th>Позитив</th><th>Негатив</th>
        <th>Ср./10</th><th>Отз.</th><th>Позитив</th><th>Негатив</th>
        <th>Ср./10</th><th>Отз.</th><th>Позитив</th><th>Негатив</th>
        <th>Ср./10</th><th>Отз.</th><th>Позитив</th><th>Негатив</th>
      </tr>
      {''.join(rows) if rows else '<tr><td colspan="17">Нет данных</td></tr>'}
    </table>
    """

def sources_deltas_block(summ: dict):
    """Отдельная компактная таблица дельт по средним: MTD vs prev_month, QTD vs prev_quarter, YTD vs prev_year."""
    prevM = summ["prev_month"]; prevQ = summ["prev_quarter"]; prevY = summ["prev_year"]
    def to_map(df):
        return {r["source"]: r["avg10"] for _, r in (df if df is not None else pd.DataFrame(columns=["source"])).iterrows()}
    M, PM = to_map(summ["mtd"]), to_map(prevM)
    Q, PQ = to_map(summ["qtd"]), to_map(prevQ)
    Y, PY = to_map(summ["ytd"]), to_map(prevY)
    all_sources = sorted(set(list(M.keys())|set(PM.keys())|list(Q.keys())|set(PQ.keys())|list(Y.keys())|set(PY.keys())))
    def dd(a,b):
        if a is None or (isinstance(a,float) and math.isnan(a)) or b is None or (isinstance(b,float) and math.isnan(b)):
            return "—"
        d = float(a) - float(b)
        return f"{d:+.2f}"
    rows=[]
    for s in all_sources:
        rows.append(f"<tr><td><b>{s}</b></td><td>{dd(M.get(s), PM.get(s))}</td><td>{dd(Q.get(s), PQ.get(s))}</td><td>{dd(Y.get(s), PY.get(s))}</td></tr>")
    return f"""
    <h3>Дельты по источникам</h3>
    <p>Сравнение средних /10: MTD vs {summ['labels']['prev_month']}, QTD vs {summ['labels']['prev_quarter']}, YTD vs {summ['labels']['prev_year']}.</p>
    <table border='1' cellspacing='0' cellpadding='6'>
      <tr><th>Источник</th><th>Δ MTD</th><th>Δ QTD</th><th>Δ YTD</th></tr>
      {''.join(rows) if rows else '<tr><td colspan="4">Нет данных</td></tr>'}
    </table>
    """

# ===========
# Charts
# ===========
def plot_ratings_reviews_trend(history_df, path_png):
    wk = history_df[history_df["period_type"]=="week"].copy()
    if wk.empty: return None
    for c in ["reviews","avg10"]:
        wk[c] = pd.to_numeric(wk[c], errors="coerce")
    def worder(k):
        try:
            y,w = str(k).split("-W"); return int(y)*100+int(w)
        except: return 0
    wk = wk.sort_values(by="period_key", key=lambda s: s.map(worder)).tail(8)
    if wk.empty: return None
    x = wk["period_key"]; y1=wk["avg10"]; y2=wk["reviews"]
    fig, ax1 = plt.subplots(figsize=(9,4.5))
    ax1.plot(x, y1, marker="o")
    ax1.set_ylabel("Средняя /10")
    ax1.set_title("Динамика: средняя /10 и объём отзывов (8 недель)")
    ax2 = ax1.twinx()
    ax2.bar(x, y2, alpha=0.25)
    ax2.set_ylabel("Отзывы")
    plt.xticks(rotation=45); fig.tight_layout()
    plt.savefig(path_png); plt.close(); return path_png

def plot_topics_trend(topics_hist, path_png):
    if topics_hist.empty: return None
    def worder(k):
        try: y,w=str(k).split("-W"); return int(y)*100+int(w)
        except: return 0
    recent = sorted(topics_hist["week_key"].unique(), key=worder)[-8:]
    th = topics_hist[topics_hist["week_key"].isin(recent)].copy()
    if th.empty: return None
    top_topics = th.groupby("topic")["mentions"].sum().sort_values(ascending=False).head(5).index.tolist()
    pv = th[th["topic"].isin(top_topics)].pivot_table(index="week_key", columns="topic", values="mentions", aggfunc="sum").fillna(0)
    plt.figure(figsize=(9,5))
    for col in pv.columns:
        plt.plot(pv.index, pv[col], marker="o", label=col)
    plt.legend(); plt.title("Тренды тем (8 недель)"); plt.ylabel("Упоминания"); plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(path_png); plt.close(); return path_png

def plot_sources_trends(sources_hist, path_avg_png, path_vol_png):
    if sources_hist.empty: return None, None
    def worder(k):
        try: y,w=str(k).split("-W"); return int(y)*100+int(w)
        except: return 0
    sh = sources_hist.copy()
    for c in ["reviews","avg10"]:
        sh[c] = pd.to_numeric(sh[c], errors="coerce")
    top_src = (sh.groupby("source")["reviews"].sum().sort_values(ascending=False).head(5).index.tolist())
    recent = sorted(sh["week_key"].unique(), key=worder)[-8:]
    sh = sh[(sh["source"].isin(top_src)) & (sh["week_key"].isin(recent))]

    if sh.empty: return None, None
    pv_avg = sh.pivot_table(index="week_key", columns="source", values="avg10", aggfunc="mean")
    pv_cnt = sh.pivot_table(index="week_key", columns="source", values="reviews", aggfunc="sum")
    # avg
    plt.figure(figsize=(9,5))
    for col in pv_avg.columns:
        plt.plot(pv_avg.index, pv_avg[col], marker="o", label=col)
    plt.legend(); plt.title("Средняя /10 по источникам (8 недель)"); plt.ylabel("Средняя /10"); plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(path_avg_png); plt.close()
    # volumes
    plt.figure(figsize=(9,5))
    bottom = np.zeros(len(pv_cnt.index))
    for col in pv_cnt.columns:
        plt.bar(pv_cnt.index, pv_cnt[col].values, bottom=bottom, label=col)
        bottom += pv_cnt[col].fillna(0).values
    plt.legend(); plt.title("Объём отзывов по источникам (8 недель, stacked)"); plt.ylabel("Отзывы"); plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(path_vol_png); plt.close()
    return path_avg_png, path_vol_png

# ===========
# Email
# ===========
def attach_file(msg, path):
    if not path or not os.path.exists(path): return
    ctype, encoding = mimetypes.guess_type(path)
    if ctype is None or encoding is not None: ctype='application/octet-stream'
    maintype, subtype = ctype.split('/',1)
    with open(path,"rb") as fp:
        part = MIMEBase(maintype, subtype); part.set_payload(fp.read()); encoders.encode_base64(part)
        part.add_header('Content-Disposition','attachment', filename=os.path.basename(path)); msg.attach(part)

def send_email(subject, html_body, attachments=None):
    msg = MIMEMultipart("mixed")
    msg["Subject"] = subject; msg["From"] = SMTP_FROM; msg["To"] = ", ".join(RECIPIENTS)
    alt = MIMEMultipart("alternative"); alt.attach(MIMEText(html_body, "html", "utf-8")); msg.attach(alt)
    for p in (attachments or []): attach_file(msg, p)
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls(); s.login(SMTP_USER, SMTP_PASS); s.sendmail(msg["From"], RECIPIENTS, msg.as_string())

# ===========
# MAIN
# ===========
def main():
    # 0) подготовка
    ensure_storage_tabs()

    # 1) забираем последний файл отзывов с диска и парсим прошлую неделю
    fid, fname, fdate = latest_reviews_file_from_drive()
    blob = drive_download(fid); tmp = "/tmp/reviews.xls"; open(tmp,"wb").write(blob)

    today = dt.date.today()
    w_start, w_end = last_week_range(today)

    df = load_reviews_df(tmp)
    wk_df, week_agg = analyze_week(df, w_start, w_end)
    week_key = append_history_week(w_start, week_agg)
    append_sources_week(week_key, wk_df)  # обновляем sources_history для недели

    # 2) читаем историю и строим агрегаты периода «на конец недели»
    hist_df     = gs_get_df(HISTORY_TAB, "A:G")
    topics_hist = gs_get_df(TOPICS_TAB,   "A:G")
    sources_hist= gs_get_df(SOURCES_TAB,  "A:G")

    ranges = period_ranges_for_week(w_start)
    labels = {k: v["label"] for k,v in ranges.items()}

    # агрегаты из недель
    mtd_agg = aggregate_weeks_from_history(hist_df, ranges["mtd"]["start"], ranges["mtd"]["end"])
    qtd_agg = aggregate_weeks_from_history(hist_df, ranges["qtd"]["start"], ranges["qtd"]["end"])
    ytd_agg = aggregate_weeks_from_history(hist_df, ranges["ytd"]["start"], ranges["ytd"]["end"])

    deltas = {
        "mtd": deltas_week_vs_period(week_agg, mtd_agg),
        "qtd": deltas_week_vs_period(week_agg, qtd_agg),
        "ytd": deltas_week_vs_period(week_agg, ytd_agg),
    }

    # 3) темы недели
    topics_df, quotes = summarize_topics(wk_df)
    # prev week topics для дельт
    prev_week_key = f"{w_start.isocalendar()[0]}-W{w_start.isocalendar()[1]-1}" if w_start.isocalendar()[1]>1 else None
    prev_topics_df = pd.DataFrame()
    if prev_week_key and not topics_hist.empty:
        th = topics_hist.rename(columns={"share_pct":"share","neg_text_share_pct":"neg_text_share"})
        prev_topics_df = th[th["week_key"]==prev_week_key][["topic","mentions","share","avg10","neg_text_share"]]

    # 4) блоки по источникам
    sources_summ = sources_summary_for_periods(sources_hist, w_start)

    # 5) собираем HTML
    head_html   = header_block(w_start, labels, week_agg, mtd_agg, qtd_agg, ytd_agg, deltas)
    topics_html = topics_table_html(topics_df, prev_topics_df)
    quotes_html = quotes_block(quotes)
    src_tbl     = sources_table_block(sources_summ)
    src_deltas  = sources_deltas_block(sources_summ)

    html = head_html + topics_html + quotes_html + src_tbl + src_deltas
    html += "<p><i>История и сравнения: листы <b>history</b>, <b>sources_history</b>, <b>topics_history</b>. Графики — во вложениях.</i></p>"

    # 6) графики
    charts=[]
    p1="/tmp/ratings_trend.png"; plot_ratings_reviews_trend(hist_df, p1); charts.append(p1)
    p2="/tmp/topics_trends.png"; plot_topics_trend(topics_hist, p2);  charts.append(p2)
    a="/tmp/sources_avg.png"; v="/tmp/sources_vol.png"; 
    a,v = plot_sources_trends(sources_hist, a, v); 
    if a: charts.append(a)
    if v: charts.append(v)
    charts = [p for p in charts if p and os.path.exists(p)]

    # 7) письмо
    subject = f"ARTSTUDIO Nevsky. Анализ отзывов за неделю {w_start:%d.%m}–{w_end:%d.%m}"
    send_email(subject, html, attachments=charts)

if __name__ == "__main__":
    main()
