# agent/weekly_reviews_agent.py
import os, re, io, json, math, datetime as dt
from datetime import date
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import smtplib, mimetypes
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# ---- Matplotlib (headless for CI)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================
# Google API clients / ENV
# =========================
SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/spreadsheets",
]
CREDS = Credentials.from_service_account_file(
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"], scopes=SCOPES
)
DRIVE = build("drive", "v3", credentials=CREDS)
SHEETS = build("sheets", "v4", credentials=CREDS).spreadsheets()

DRIVE_FOLDER_ID = os.environ["DRIVE_FOLDER_ID"]
HISTORY_SHEET_ID = os.environ["SHEETS_HISTORY_ID"]
RECIPIENTS = [e.strip() for e in os.environ["RECIPIENTS"].split(",") if e.strip()]
HISTORY_TAB = os.environ.get("HISTORY_TAB", "history")
TOPICS_TAB  = os.environ.get("TOPICS_TAB", "topics_history")

# =========================
# Config: columns & patterns
# =========================
EXPECTED = ["Дата","Рейтинг","Источник","Автор","Код языка","Текст отзыва","Наличие ответа"]
RENAMES  = {
    "Дата":"Дата","Рейтинг":"Рейтинг","Источник":"Источник","Автор":"Автор",
    "Код языка":"Код языка","Текст отзыва":"Текст","Наличие ответа":"Ответ"
}

TOPIC_PATTERNS = {
    "Локация":        [r"расположен", r"расположение", r"рядом", r"вокзал", r"невск"],
    "Персонал":       [r"персонал", r"сотрудник", r"администрат", r"ресепшен|ресепшн", r"менеджер"],
    "Чистота":        [r"чисто|уборк", r"бель[её]", r"пятн", r"полотенц", r"постель"],
    "Номер/оснащение":[r"номер", r"кухн", r"посудомо", r"стирал", r"плита", r"чайник", r"фен", r"сейф"],
    "Сантехника/вода":[r"душ", r"слив", r"кран|смесител", r"бойлер|водонагрев", r"давлен", r"температур", r"канализац|засор"],
    "AC/шум":         [r"кондицион", r"вентиляц", r"шум", r"тихо", r"громк"],
    "Завтраки":       [r"завтрак", r"ланч-?бокс", r"ресторан"],
    "Чек-ин/вход":    [r"заселен|заезд|чек-?ин|вход|инструкц", r"селф|код|ключ|карта"],
    "Wi-Fi/интернет": [r"wi-?fi|wifi|вай-?фай", r"интернет", r"парол"],
    "Цена/ценность":  [r"цена|стоимост", r"дорог|дешев", r"соотношен.*цена"],
}

# Лексиконы/триггеры для правил сентимента
NEG_CUES = re.compile(
    r"\bне\b|\bнет\b|проблем|ошиб|задерж|долг|ждал|ждали|"
    r"сложн|плохо|неудоб|путан|непонят|неясн|не приш|не сработ|не открыл|не откры|скромн|бедн|мало",
    re.IGNORECASE
)
POS_LEX = re.compile(r"отличн|прекрасн|замечат|идеальн|классн|любезн|комфортн|удобн|чисто|тихо|вкусн|быстро", re.IGNORECASE)
NEG_LEX = re.compile(r"ужасн|плох|грязн|громк|холодн|жарко|слаб|плохо|долго|скучн|бедн|мало|дорог", re.IGNORECASE)

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9\-]{3,}")
RU_STOP = set('и или для что это эта этот эти также там тут как при по на из от до во со бы ли был была были уже еще есть нет да мы вы они она он в за к с у о а но же то тд др очень всего более менее'.split())

# =========================
# Drive helpers
# =========================
def parse_date_from_name(name: str):
    m = re.search(r"^Reviews_(\d{2})-(\d{2})-(\d{4})\.xls(x)?$", name, re.IGNORECASE)
    if not m: return None
    dd, mm, yyyy = map(int, m.groups()[:3])
    return date(yyyy, mm, dd)

def latest_reviews_file():
    # Берём все файлы в папке и фильтруем по regex — надёжно, без "ends with" в q
    regex = re.compile(r"^Reviews_\d{2}-\d{2}-\d{4}\.xls(x)?$", re.IGNORECASE)
    res = DRIVE.files().list(
        q=f"'{DRIVE_FOLDER_ID}' in parents and trashed=false",
        fields="files(id,name,modifiedTime)",
        pageSize=1000
    ).execute()
    items = res.get("files", [])
    items = [(i["id"], i["name"], parse_date_from_name(i["name"])) for i in items if regex.match(i["name"])]
    if not items:
        raise RuntimeError("В папке нет файлов вида Reviews_dd-mm-yyyy.xls(x).")
    items.sort(key=lambda t: t[2], reverse=True)
    return items[0]  # id, name, date

def download_file(file_id: str) -> bytes:
    request = DRIVE.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = dl.next_chunk()
    return buf.getvalue()

# =========================
# Sheets helpers
# =========================
def ensure_tab(spreadsheet_id: str, tab_name: str, header: list):
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

def ensure_history_sheet():
    ensure_tab(HISTORY_SHEET_ID, HISTORY_TAB,
               ["period_type","period_key","reviews","avg10","pos","neu","neg"])

def history_read():
    try:
        res = SHEETS.values().get(
            spreadsheetId=HISTORY_SHEET_ID,
            range=f"{HISTORY_TAB}!A:G"
        ).execute()
        vals = res.get("values", [])
        if not vals: return pd.DataFrame(columns=["period_type","period_key","reviews","avg10","pos","neu","neg"])
        df = pd.DataFrame(vals[1:], columns=vals[0])
        return df
    except Exception:
        return pd.DataFrame(columns=["period_type","period_key","reviews","avg10","pos","neu","neg"])

def history_append(rows: pd.DataFrame):
    if rows.empty: return
    SHEETS.values().append(
        spreadsheetId=HISTORY_SHEET_ID,
        range=f"{HISTORY_TAB}!A:G",
        valueInputOption="RAW",
        body={"values": rows.values.tolist()}
    ).execute()

def load_topics_history():
    ensure_tab(HISTORY_SHEET_ID, TOPICS_TAB,
               ["week_key","topic","mentions","share_pct","avg10","neg_text_share_pct","by_channel_json"])
    try:
        res = SHEETS.values().get(
            spreadsheetId=HISTORY_SHEET_ID,
            range=f"{TOPICS_TAB}!A:G"
        ).execute()
        vals = res.get("values", [])
        if len(vals) <= 1:
            return pd.DataFrame(columns=["week_key","topic","mentions","share_pct","avg10","neg_text_share_pct","by_channel_json"])
        th = pd.DataFrame(vals[1:], columns=vals[0])
        # numeric cast
        for c in ["mentions","share_pct","avg10","neg_text_share_pct"]:
            th[c] = pd.to_numeric(th[c], errors="coerce")
        return th
    except Exception:
        return pd.DataFrame(columns=["week_key","topic","mentions","share_pct","avg10","neg_text_share_pct","by_channel_json"])

def topics_append(week_key: str, df_topics: pd.DataFrame):
    ensure_tab(HISTORY_SHEET_ID, TOPICS_TAB,
               ["week_key","topic","mentions","share_pct","avg10","neg_text_share_pct","by_channel_json"])
    if df_topics.empty:
        return
    rows = df_topics.assign(week_key=week_key)[
        ["week_key","topic","mentions","share","avg10","neg_text_share","by_channel_json"]
    ].values.tolist()
    SHEETS.values().append(
        spreadsheetId=HISTORY_SHEET_ID,
        range=f"{TOPICS_TAB}!A:G",
        valueInputOption="RAW",
        body={"values": rows}
    ).execute()

# =========================
# IO & transforms
# =========================
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
    df = pd.read_excel(path)  # .xls/.xlsx (xlrd/openpyxl)
    # мягкий маппинг по русским заголовкам
    low = {str(c).strip().lower(): c for c in df.columns}
    rename = {}
    for k in EXPECTED:
        hit = low.get(k.lower())
        if hit:
            rename[hit] = RENAMES[k]
        else:
            cand = next((c for lc, c in low.items() if k.split()[0].lower() in lc), None)
            if cand: rename[cand] = RENAMES[k]
    df = df.rename(columns=rename)
    missing = [v for v in RENAMES.values() if v not in df.columns]
    if missing:
        raise RuntimeError(f"В файле не найдены ожидаемые колонки: {missing}")
    return df[[*RENAMES.values()]]

# =========================
# Sentiment / topics
# =========================
def sentence_split(text: str):
    parts = re.split(r"(?<=[\.\!\?\n])\s+", str(text).strip())
    return [p.strip() for p in parts if len(p.strip()) >= 12]

def sentiment_sign(text: str, rating10: float):
    """
    Правила:
    - если NEG-сигналов намного больше POS → negative
    - если есть POS и нет NEG → positive
    - иначе fallback по рейтингу: <6 neg, 6–8 neutral, >=8 pos
    """
    t = str(text)
    pos = len(POS_LEX.findall(t))
    neg = len(NEG_LEX.findall(t)) + len(NEG_CUES.findall(t))
    if neg - pos >= 2:
        return "negative"
    if pos - neg >= 1 and neg == 0:
        return "positive"
    if rating10 is not None and not (isinstance(rating10, float) and math.isnan(rating10)):
        if rating10 < 6: return "negative"
        if rating10 >= 8: return "positive"
        return "neutral"
    return "neutral"

def classify_topics(text: str):
    tl = str(text).lower()
    hits = set()
    for topic, pats in TOPIC_PATTERNS.items():
        for p in pats:
            if re.search(p, tl, re.IGNORECASE):
                hits.add(topic); break
    return hits

def extract_keywords(texts, topn=20):
    tokens = []
    for t in texts:
        for w in TOKEN_RE.findall(str(t)):
            wl = w.lower()
            if wl not in RU_STOP and not wl.isdigit():
                tokens.append(wl)
    from collections import Counter
    return Counter(tokens).most_common(topn)

def summarize_topics(wk_df: pd.DataFrame):
    """
    Возвращает:
      - df тем недели: topic | mentions | share | avg10 | neg_text_share | by_channel_json
      - quotes: {topic: {"pos": str|None, "neg": str|None}}
      - keywords: top tokens
    """
    if wk_df.empty:
        return pd.DataFrame(columns=["topic","mentions","share","avg10","neg_text_share","by_channel_json"]), {}, []
    rows = []; quotes_map = {}; kw_texts=[]
    for idx, r in wk_df.iterrows():
        text = str(r["Текст"])
        found = classify_topics(text)
        if not found: 
            continue
        s = sentiment_sign(text, r["_rating10"])
        kw_texts.append(text)
        for tp in found:
            rows.append({"topic": tp, "idx": idx, "sent": s, "rating10": r["_rating10"], "channel": r["Источник"]})
            for sent in sentence_split(text):
                if any(re.search(p, sent, re.IGNORECASE) for p in TOPIC_PATTERNS[tp]):
                    qm = quotes_map.setdefault(tp, {"pos": None, "neg": None})
                    ss = sentiment_sign(sent, r["_rating10"])
                    if ss == "negative" and not qm["neg"]:
                        qm["neg"] = sent
                    if ss == "positive" and not qm["pos"]:
                        qm["pos"] = sent
    if not rows:
        return pd.DataFrame(columns=["topic","mentions","share","avg10","neg_text_share","by_channel_json"]), quotes_map, []
    tmp = pd.DataFrame(rows)
    total_reviews = len(wk_df)
    # агрегаты по темам
    agg = (tmp.groupby("topic")
             .agg(mentions=("topic","size"),
                  avg10=("rating10","mean"),
                  neg_text_share=("sent", lambda s: (s=="negative").mean()))
             .reset_index())
    agg["share"] = (agg["mentions"]/total_reviews*100).round(1)
    agg["avg10"] = agg["avg10"].round(2)
    agg["neg_text_share"] = (agg["neg_text_share"]*100).round(1)
    # по каналам
    bych = (tmp.groupby(["topic","channel"])
              .agg(cnt=("topic","size"), avg10=("rating10","mean"))
              .reset_index())
    by_channel_json=[]
    for tp, sub in bych.groupby("topic"):
        d = {row["channel"]: {"count": int(row["cnt"]), "avg10": round(float(row["avg10"]),2)}
             for _, row in sub.iterrows()}
        by_channel_json.append({"topic": tp, "by_channel_json": json.dumps(d, ensure_ascii=False)})
    agg = agg.merge(pd.DataFrame(by_channel_json), on="topic", how="left")
    keywords = extract_keywords(kw_texts, topn=20)
    return agg.sort_values("mentions", ascending=False), quotes_map, keywords

# =========================
# Week-level analysis
# =========================
def analyze_week(df: pd.DataFrame, start: date, end: date):
    df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce", dayfirst=True).dt.date
    wk = df[(df["Дата"]>=start) & (df["Дата"]<=end)].copy()
    wk["_rating10"] = normalize_to10(wk["Рейтинг"])
    wk["_sent_rule"] = wk.apply(lambda r: sentiment_sign(r["Текст"], r["_rating10"]), axis=1)
    agg = {
        "start": start, "end": end,
        "reviews": len(wk),
        "avg10": round(wk["_rating10"].mean(),2) if len(wk) else None,
        "pos": int((wk["_sent_rule"]=="positive").sum()),
        "neu": int((wk["_sent_rule"]=="neutral").sum()),
        "neg": int((wk["_sent_rule"]=="negative").sum()),
        "by_channel": wk.groupby("Источник")["_rating10"].agg(["count","mean"]).round(2)
                      .sort_values("count", ascending=False).reset_index().to_dict("records")
    }
    return wk, agg

def upsert_history(agg, week_start: date):
    iso_y, iso_w, _ = week_start.isocalendar()
    rows = pd.DataFrame([
        ["week",   f"{iso_y}-W{iso_w}",                   agg["reviews"], agg["avg10"], agg["pos"], agg["neu"], agg["neg"]],
        ["month",  f"{week_start:%Y-%m}",                 agg["reviews"], agg["avg10"], agg["pos"], agg["neu"], agg["neg"]],
        ["quarter",f"{week_start.year}-Q{(week_start.month-1)//3 + 1}", agg["reviews"], agg["avg10"], agg["pos"], agg["neu"], agg["neg"]],
        ["year",   f"{week_start.year}",                  agg["reviews"], agg["avg10"], agg["pos"], agg["neu"], agg["neg"]],
    ], columns=["period_type","period_key","reviews","avg10","pos","neu","neg"])
    hist = history_read()
    if not len(hist[(hist["period_type"]=="week") & (hist["period_key"]==f"{iso_y}-W{iso_w}")]):
        history_append(rows)

# =========================
# Trends & period aggregations (from weeks)
# =========================
def week_key_to_monday(week_key: str) -> date:
    y, w = week_key.split("-W")
    # ISO week Monday
    return date(int(y), 1, 4).fromisocalendar(int(y), int(w), 1)

def _num(x):
    try: return float(x)
    except: return float("nan")

def aggregate_weeks(hist_df: pd.DataFrame, start_d: date, end_d: date):
    """
    Собираем агрегаты (reviews, avg10, pos, neu, neg) по week-строкам в history
    с понедельниками в диапазоне [start_d, end_d]. avg10 — взвешенная.
    """
    if hist_df.empty:
        return {"reviews": 0, "avg10": None, "pos": 0, "neu": 0, "neg": 0,
                "pos_share": None, "neg_share": None}
    wk = hist_df[hist_df["period_type"]=="week"].copy()
    if wk.empty:
        return {"reviews": 0, "avg10": None, "pos": 0, "neu": 0, "neg": 0,
                "pos_share": None, "neg_share": None}
    for c in ["reviews","avg10","pos","neu","neg"]:
        wk[c] = wk[c].apply(_num)
    def in_range(k):
        try:
            md = week_key_to_monday(k)
            return (md >= start_d) and (md <= end_d)
        except:
            return False
    wk = wk[wk["period_key"].apply(in_range)]
    if wk.empty:
        return {"reviews": 0, "avg10": None, "pos": 0, "neu": 0, "neg": 0,
                "pos_share": None, "neg_share": None}
    n = wk["reviews"].sum()
    if n <= 0:
        return {"reviews": 0, "avg10": None, "pos": 0, "neu": 0, "neg": 0,
                "pos_share": None, "neg_share": None}
    avg10 = (wk["avg10"] * wk["reviews"]).sum() / n
    pos = wk["pos"].sum(); neu = wk["neu"].sum(); neg = wk["neg"].sum()
    return {
        "reviews": int(n),
        "avg10": round(float(avg10),2),
        "pos": int(pos), "neu": int(neu), "neg": int(neg),
        "pos_share": round(float(100.0*pos/n),1),
        "neg_share": round(float(100.0*neg/n),1),
    }

def month_range_for_week(week_start: date):
    start = week_start.replace(day=1)
    end = week_start + dt.timedelta(days=6)
    return start, end

def quarter_of(d: date):
    return (d.month - 1)//3 + 1

def quarter_range_for_week(week_start: date):
    q = quarter_of(week_start)
    q_start_month = (q-1)*3 + 1
    start = date(week_start.year, q_start_month, 1)
    end = week_start + dt.timedelta(days=6)
    return start, end

def ytd_range_for_week(week_start: date):
    start = date(week_start.year, 1, 1)
    end = week_start + dt.timedelta(days=6)
    return start, end

def build_period_trends_from_weeks(hist_df: pd.DataFrame, week_start: date, week_agg: dict):
    m_start, m_end = month_range_for_week(week_start)
    q_start, q_end = quarter_range_for_week(week_start)
    y_start, y_end = ytd_range_for_week(week_start)

    mtd = aggregate_weeks(hist_df, m_start, m_end)
    qtd = aggregate_weeks(hist_df, q_start, q_end)
    ytd = aggregate_weeks(hist_df, y_start, y_end)

    def dd(cur, ref):
        if cur is None or ref is None: return None
        return round(float(cur) - float(ref), 2)

    week_pos_share = (100.0*week_agg["pos"]/week_agg["reviews"]) if week_agg["reviews"] else None
    week_neg_share = (100.0*week_agg["neg"]/week_agg["reviews"]) if week_agg["reviews"] else None

    return {
        "mtd": {"agg": mtd, "delta": {
            "avg10_delta": dd(week_agg["avg10"], mtd.get("avg10")),
            "pos_delta_pp": dd(week_pos_share, mtd.get("pos_share")),
            "neg_delta_pp": dd(week_neg_share, mtd.get("neg_share")),
        }},
        "qtd": {"agg": qtd, "delta": {
            "avg10_delta": dd(week_agg["avg10"], qtd.get("avg10")),
            "pos_delta_pp": dd(week_pos_share, qtd.get("pos_share")),
            "neg_delta_pp": dd(week_neg_share, qtd.get("neg_share")),
        }},
        "ytd": {"agg": ytd, "delta": {
            "avg10_delta": dd(week_agg["avg10"], ytd.get("avg10")),
            "pos_delta_pp": dd(week_pos_share, ytd.get("pos_share")),
            "neg_delta_pp": dd(week_neg_share, ytd.get("neg_share")),
        }},
    }

def summarize_trends(hist_df: pd.DataFrame, week_start: date):
    iso_y, iso_w, _ = week_start.isocalendar()
    def pick(ptype, pkey):
        if not pkey: return None
        m = hist_df[(hist_df["period_type"]==ptype) & (hist_df["period_key"]==pkey)]
        return m.iloc[0].to_dict() if len(m) else None
    prev_week_key = f"{iso_y}-W{iso_w-1}" if iso_w>1 else None
    month_key = f"{week_start:%Y-%m}"
    q = (week_start.month-1)//3 + 1
    quarter_key = f"{week_start.year}-Q{q}"
    year_key = str(week_start.year)
    return {
        "prev_week": pick("week", prev_week_key),
        "month": pick("month", month_key),
        "quarter": pick("quarter", quarter_key),
        "year": pick("year", year_key),
    }

# =========================
# Charts
# =========================
def plot_topics_bar(df_topics, out_path):
    if df_topics.empty:
        return None
    top = df_topics.sort_values("mentions", ascending=False).head(6)
    plt.figure(figsize=(8,4.5))
    plt.bar(top["topic"], top["mentions"])
    plt.title("Топ-темы недели (по числу упоминаний)")
    plt.ylabel("Упоминания")
    plt.tight_layout()
    plt.savefig(out_path); plt.close()
    return out_path

def plot_trends_topics(topics_hist, out_path):
    if topics_hist.empty:
        return None
    def week_order(k):
        try:
            y, w = k.split("-W"); return int(y)*100 + int(w)
        except: return 0
    recent_weeks = sorted(topics_hist["week_key"].unique(), key=week_order)[-8:]
    th = topics_hist[topics_hist["week_key"].isin(recent_weeks)].copy()
    if th.empty:
        return None
    top_topics = (th.groupby("topic")["mentions"].sum().sort_values(ascending=False).head(5).index.tolist())
    pv = th[th["topic"].isin(top_topics)].pivot_table(index="week_key", columns="topic", values="mentions", aggfunc="sum").fillna(0)
    plt.figure(figsize=(9,5))
    for col in pv.columns:
        plt.plot(pv.index, pv[col], marker="o", label=col)
    plt.xticks(rotation=45)
    plt.title("Тренды тем (последние 8 недель)")
    plt.ylabel("Упоминания")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path); plt.close()
    return out_path

def plot_ratings_reviews_trend(history_df, out_path):
    if history_df.empty:
        return None
    wk = history_df[history_df["period_type"]=="week"].copy()
    if wk.empty:
        return None
    # numeric
    for c in ["reviews","avg10"]:
        wk[c] = wk[c].apply(_num)
    # sort last 8 weeks
    def week_order(k):
        try:
            y, w = k.split("-W"); return int(y)*100 + int(w)
        except: return 0
    wk = wk.sort_values(key=lambda s: s.map(week_order), by="period_key")
    wk = wk.tail(8)
    if wk.empty:
        return None
    x = wk["period_key"].tolist()
    y1 = wk["avg10"].tolist()
    y2 = wk["reviews"].tolist()
    fig, ax1 = plt.subplots(figsize=(9,4.5))
    ax1.plot(x, y1, marker="o")
    ax1.set_ylabel("Средняя /10")
    ax1.set_ylim(bottom=min(7.0, min(y1) if all(pd.notna(y1)) else 0))
    ax1.set_title("Динамика: средняя /10 и объём отзывов (последние 8 недель)")
    ax2 = ax1.twinx()
    ax2.bar(x, y2, alpha=0.25)
    ax2.set_ylabel("Отзывы")
    plt.xticks(rotation=45)
    fig.tight_layout()
    plt.savefig(out_path); plt.close()
    return out_path

# =========================
# HTML helpers
# =========================
def fmt_pp(x):
    if x is None or (isinstance(x, float) and math.isnan(x)): return "—"
    s = f"{x:+.2f} п.п."
    return s.replace("+-", "−")

def fmt_avg(x):
    return "—" if x is None else f"{x:.2f}"

def topics_table_html(topics_df, prev_df=None):
    if topics_df.empty:
        return "<h3>Темы недели</h3><p>Нет данных</p>"
    prev = None
    if prev_df is not None and not prev_df.empty:
        prev = prev_df.set_index("topic")
    rows=[]
    for _, r in topics_df.head(10).iterrows():
        d_m = ""; d_n = ""
        if prev is not None and r["topic"] in prev.index:
            dm = int(r["mentions"] - prev.loc[r["topic"], "mentions"])
            dn = float(r["neg_text_share"] - prev.loc[r["topic"], "neg_text_share"])
            d_m = f" ({'+' if dm>=0 else ''}{dm})"
            d_n = f" ({'+' if dn>=0 else ''}{round(dn,1)} п.п.)"
        rows.append(
            f"<tr><td>{r['topic']}</td>"
            f"<td>{int(r['mentions'])}{d_m}</td>"
            f"<td>{r['share']}%</td>"
            f"<td>{r['avg10']}/10</td>"
            f"<td>{r['neg_text_share']}%{d_n}</td></tr>"
        )
    return (
        "<h3>Темы недели</h3>"
        "<table border='1' cellspacing='0' cellpadding='6'>"
        "<tr><th>Тема</th><th>Упоминания</th><th>Доля</th><th>Средняя /10</th><th>Негатив по тексту</th></tr>"
        + "".join(rows) + "</table>"
    )

def periods_table(periods):
    rows = []
    for title, key in [("Месяц (MTD)", "mtd"), ("Квартал (QTD)", "qtd"), ("Год (YTD)", "ytd")]:
        a = periods[key]["agg"]; d = periods[key]["delta"]
        rows.append(
            f"<tr><td>{title}</td>"
            f"<td>{a['reviews']}</td>"
            f"<td>{fmt_avg(a['avg10'])}</td>"
            f"<td>{a['pos_share'] if a['pos_share'] is not None else '—'}%</td>"
            f"<td>{a['neg_share'] if a['neg_share'] is not None else '—'}%</td>"
            f"<td>{fmt_pp(d['avg10_delta'])}</td>"
            f"<td>{fmt_pp(d['pos_delta_pp'])}</td>"
            f"<td>{fmt_pp(d['neg_delta_pp'])}</td></tr>"
        )
    return (
        "<h3>Динамика к MTD/QTD/YTD</h3>"
        "<table border='1' cellspacing='0' cellpadding='6'>"
        "<tr><th>Период</th><th>Отзывы</th><th>Средняя /10</th>"
        "<th>Позитив, %</th><th>Негатив, %</th>"
        "<th>Δ Ср./10 к неделе</th><th>Δ Позитив, п.п.</th><th>Δ Негатив, п.п.</th></tr>"
        + "".join(rows) + "</table>"
    )

def html_header_block(agg, trends):
    s, e = agg["start"], agg["end"]
    def pct(x, tot): return f"{x} ({round(100*x/tot,1)}%)" if tot else "—"
    def delta(cur, ref):
        try:
            d = float(cur) - float(ref); return f"{'+' if d>=0 else ''}{round(d,2)} п.п."
        except Exception: return "—"
    body = f"""
    <h2>Еженедельный отчёт по отзывам ({s:%d.%m.%Y}–{e:%d.%m.%Y})</h2>
    <p><b>Итого за неделю:</b> {agg['reviews']} отзывов; средняя <b>{agg['avg10']}/10</b>;
    позитив {pct(agg['pos'], agg['reviews'])}, нейтраль {pct(agg['neu'], agg['reviews'])}, негатив {pct(agg['neg'], agg['reviews'])}.</p>
    """
    if trends.get("prev_week"): body += f"<p>К прошлой неделе: {delta(agg['avg10'], trends['prev_week']['avg10'])}; объём {agg['reviews']} vs {trends['prev_week']['reviews']}.</p>"
    if trends.get("month"):     body += f"<p>К среднему по месяцу: {delta(agg['avg10'], trends['month']['avg10'])}.</p>"
    if trends.get("quarter"):   body += f"<p>К среднему по кварталу: {delta(agg['avg10'], trends['quarter']['avg10'])}.</p>"
    if trends.get("year"):      body += f"<p>К среднему по году (YTD): {delta(agg['avg10'], trends['year']['avg10'])}.</p>"
    return body

def html_full(agg, trends, topics_tbl_html, quotes_html, keywords, periods_html):
    body = html_header_block(agg, trends)
    body += topics_tbl_html
    if keywords:
        kw = ", ".join([w for w,_ in keywords[:20]])
        body += f"<p><b>Ключевые слова недели:</b> {kw}</p>"
    body += quotes_html
    body += periods_html
    body += "<p><i>История и сравнения ведутся в Google Sheet (листы <b>history</b> и <b>topics_history</b>). PNG-графики — во вложениях.</i></p>"
    return body

# =========================
# Mail
# =========================
def attach_file(msg, path):
    if not path or not os.path.exists(path):
        return
    ctype, encoding = mimetypes.guess_type(path)
    if ctype is None or encoding is not None:
        ctype = 'application/octet-stream'
    maintype, subtype = ctype.split('/', 1)
    with open(path, "rb") as fp:
        part = MIMEBase(maintype, subtype)
        part.set_payload(fp.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment', filename=os.path.basename(path))
        msg.attach(part)

def send_email(subject, html_body, attachments=None):
    msg = MIMEMultipart("mixed")
    msg["Subject"] = subject
    msg["From"] = os.environ.get("SMTP_FROM", os.environ["SMTP_USER"])
    msg["To"] = ", ".join(RECIPIENTS)
    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(html_body, "html", "utf-8"))
    msg.attach(alt)
    if attachments:
        for p in attachments:
            attach_file(msg, p)
    with smtplib.SMTP(os.environ.get("SMTP_HOST","smtp.gmail.com"), int(os.environ.get("SMTP_PORT",587))) as s:
        s.starttls()
        s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
        s.sendmail(msg["From"], RECIPIENTS, msg.as_string())

# =========================
# Main
# =========================
def last_week_range(ref: date):
    monday = ref - dt.timedelta(days=ref.weekday())  # текущий понедельник
    end = monday - dt.timedelta(days=1)              # прошлая неделя (вс)
    start = end - dt.timedelta(days=6)               # прошлая неделя (пн)
    return start, end

def main():
    # 1) Забираем последний файл с Drive
    file_id, name, fdate = latest_reviews_file()
    blob = download_file(file_id)
    tmp = "/tmp/reviews.xls"
    with open(tmp, "wb") as f: f.write(blob)

    # 2) Отчётная неделя — прошлая (скрипт по расписанию запускается по пн)
    today = dt.date.today()
    w_start, w_end = last_week_range(today)

    # 3) Читаем данные, анализируем неделю
    ensure_history_sheet()
    df = load_reviews_df(tmp)
    wk, agg = analyze_week(df, w_start, w_end)

    # 4) Пишем агрегаты в history (week/month/quarter/year, неделя — idempotent)
    upsert_history(agg, w_start)
    hist = history_read()
    trends = summarize_trends(hist, w_start)

    # 5) MTD/QTD/YTD из недель (включая backfill)
    periods = build_period_trends_from_weeks(hist, w_start, agg)

    # 6) Темы/цитаты/ключевые слова; запись истории тем + by_channel_json
    topics_df, quotes, keywords = summarize_topics(wk)
    wk_key = f"{w_start.isocalendar()[0]}-W{w_start.isocalendar()[1]}"
    topics_append(wk_key, topics_df)

    # 7) Для дельт в таблице тем: предыдущая неделя из topics_history
    prev_topics = pd.DataFrame()
    try:
        th = load_topics_history()
        iso_y, iso_w, _ = w_start.isocalendar()
        prev_week_key = f"{iso_y}-W{iso_w-1}" if iso_w>1 else None
        if prev_week_key and not th.empty:
            prev_topics = (th[th["week_key"]==prev_week_key]
                           .rename(columns={"share_pct":"share","neg_text_share_pct":"neg_text_share"})
                           [["topic","mentions","share","avg10","neg_text_share"]])
    except Exception:
        pass

    # 8) HTML: темы (с дельтами), цитаты, блок MTD/QTD/YTD
    topics_html = topics_table_html(topics_df, prev_topics)
    quotes_html_parts=[]
    for tp, q in quotes.items():
        parts=[]
        if q.get("pos"): parts.append(f"<b>+</b> «{q['pos']}»")
        if q.get("neg"): parts.append(f"<b>–</b> «{q['neg']}»")
        if parts:
            quotes_html_parts.append(f"<p><b>{tp}:</b> " + " / ".join(parts) + "</p>")
    quotes_html = ("<h3>Цитаты</h3>" + "".join(quotes_html_parts)) if quotes_html_parts else ""
    periods_html = periods_table(periods)

    # 9) Графики (PNG вложения)
    charts = []
    topics_hist = load_topics_history()
    p1 = "/tmp/topics_week.png";     plot_topics_bar(topics_df, p1); charts.append(p1)
    p2 = "/tmp/topics_trends.png";   plot_trends_topics(topics_hist, p2); charts.append(p2)
    p3 = "/tmp/ratings_trend.png";   plot_ratings_reviews_trend(hist, p3); charts.append(p3)
    charts = [p for p in charts if p and os.path.exists(p)]

    # 10) Письмо
    subject = f"ARTSTUDIO Nevsky. Анализ отзывов за неделю {w_start:%d.%m}–{w_end:%d.%m}"
    html = html_full(agg, trends, topics_html, quotes_html, keywords, periods_html)
    send_email(subject, html, attachments=charts)

if __name__ == "__main__":
    main()
