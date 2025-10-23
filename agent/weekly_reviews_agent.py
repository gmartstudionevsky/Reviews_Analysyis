import os, re, io, json, math, datetime as dt
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

# ---- Matplotlib (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Google clients ---
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

# --- Patterns/lexicons ---
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
NEG_CUES = re.compile(
    r"\bне\b|\bнет\b|проблем|ошиб|задерж|долг|ждал|ждали|"
    r"сложн|плохо|неудоб|путан|непонят|неясн|не приш|не сработ|не открыл|не откры|скромн|бедн|мало",
    re.IGNORECASE
)
POS_LEX = re.compile(r"отличн|прекрасн|замечат|идеальн|классн|любезн|комфортн|удобн|чисто|тихо|вкусн|быстро", re.IGNORECASE)
NEG_LEX = re.compile(r"ужасн|плох|грязн|громк|холодн|жарко|слаб|плохо|долго|скучн|бедн|мало|дорог", re.IGNORECASE)

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9\-]{3,}")
RU_STOP = set('и или для что это эта этот эти также там тут как при по на из от до во со бы ли был была были уже еще есть нет да мы вы они она он в за к с у о а но же то тд др очень всего более менее'.split())

EXPECTED = ["Дата","Рейтинг","Источник","Автор","Код языка","Текст отзыва","Наличие ответа"]
RENAMES = {
    "Дата": "Дата", "Рейтинг":"Рейтинг", "Источник":"Источник", "Автор":"Автор",
    "Код языка":"Код языка", "Текст отзыва":"Текст", "Наличие ответа":"Ответ"
}

# --- Helpers ---
def parse_date_from_name(name: str):
    m = re.search(r"^Reviews_(\d{2})-(\d{2})-(\d{4})\.xls(x)?$", name, re.IGNORECASE)
    if not m: return None
    dd, mm, yyyy = map(int, m.groups()[:3])
    return dt.date(yyyy, mm, dd)

def latest_reviews_file():
    regex = re.compile(r"^Reviews_\d{2}-\d{2}-\d{4}\.xls(x)?$", re.IGNORECASE)
    results = DRIVE.files().list(
        q=f"'{DRIVE_FOLDER_ID}' in parents and trashed=false",
        fields="files(id,name,modifiedTime)",
        pageSize=1000
    ).execute()
    items = results.get("files", [])
    items = [(i["id"], i["name"], parse_date_from_name(i["name"])) for i in items if regex.match(i["name"])]
    if not items:
        raise RuntimeError("В папке нет файлов вида Reviews_dd-mm-yyyy.xls(x).")
    items.sort(key=lambda t: t[2], reverse=True)
    return items[0]  # id, name, date

def download_file(file_id: str) -> bytes:
    request = DRIVE.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()

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

def last_week_range(ref: dt.date):
    monday = ref - dt.timedelta(days=ref.weekday())
    end = monday - dt.timedelta(days=1)
    start = end - dt.timedelta(days=6)
    return start, end

# --- Sheets utils ---
def ensure_tab(spreadsheet_id: str, tab_name: str, header: list):
    meta = SHEETS.get(spreadsheetId=spreadsheet_id).execute()
    tabs = [s["properties"]["title"] for s in meta.get("sheets", [])]
    if tab_name not in tabs:
        SHEETS.batchUpdate(spreadsheetId=spreadsheet_id,
                           body={"requests":[{"addSheet":{"properties":{"title":tab_name}}}]}).execute()
        SHEETS.values().update(spreadsheetId=spreadsheet_id,
                               range=f"{tab_name}!A1:{chr(64+len(header))}1",
                               valueInputOption="RAW",
                               body={"values":[header]}).execute()

def ensure_history_sheet():
    ensure_tab(HISTORY_SHEET_ID, HISTORY_TAB, ["period_type","period_key","reviews","avg10","pos","neu","neg"])

def history_read():
    try:
        res = SHEETS.values().get(spreadsheetId=HISTORY_SHEET_ID, range=f"{HISTORY_TAB}!A:G").execute()
        vals = res.get("values", [])
        if not vals: return pd.DataFrame(columns=["period_type","period_key","reviews","avg10","pos","neu","neg"])
        return pd.DataFrame(vals[1:], columns=vals[0])
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

def load_reviews_df(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)  # .xls/.xlsx
    lowmap = {str(c).strip().lower(): c for c in df.columns}
    rename = {}
    for k in EXPECTED:
        hit = lowmap.get(k.lower())
        if hit:
            rename[hit] = RENAMES[k]
        else:
            cand = next((c for lc, c in lowmap.items() if k.split()[0].lower() in lc), None)
            if cand: rename[cand] = RENAMES[k]
    df = df.rename(columns=rename)
    missing = [v for v in RENAMES.values() if v not in df.columns]
    if missing:
        raise RuntimeError(f"В файле не найдены ожидаемые колонки: {missing}")
    return df[[*RENAMES.values()]]

# --- Sentiment / topics ---
def sentence_split(text: str):
    parts = re.split(r"(?<=[\.\!\?\n])\s+", str(text).strip())
    return [p.strip() for p in parts if len(p.strip())>=12]

def classify_topics(text: str):
    tl = str(text).lower()
    hits = set()
    for topic, pats in TOPIC_PATTERNS.items():
        for p in pats:
            if re.search(p, tl, flags=re.IGNORECASE):
                hits.add(topic); break
    return hits

def sentiment_sign(text: str, rating10: float):
    """
    Правила:
    - если много NEG_CUES или NEG_LEX и мало POS → negative
    - если POS_LEX и нет NEG → positive
    - иначе fallback по рейтингу: <6 neg, 6–8 neutral, >=8 pos
    """
    t = str(text)
    pos = len(POS_LEX.findall(t))
    neg = len(NEG_LEX.findall(t)) + len(NEG_CUES.findall(t))
    if neg - pos >= 2:
        return "negative"
    if pos - neg >= 1 and neg == 0:
        return "positive"
    if rating10 is not None:
        if rating10 < 6: return "negative"
        if rating10 >= 8: return "positive"
        return "neutral"
    return "neutral"

def extract_keywords(texts, topn=20):
    tokens=[]
    for t in texts:
        for w in TOKEN_RE.findall(str(t)):
            wl = w.lower()
            if wl not in RU_STOP and not wl.isdigit():
                tokens.append(wl)
    from collections import Counter
    return Counter(tokens).most_common(topn)

def summarize_topics(wk_df: pd.DataFrame):
    if wk_df.empty:
        return pd.DataFrame(columns=["topic","mentions","share","avg10","neg_text_share","by_channel_json"]), {}, []
    topics_rows=[]; quotes_map={}; kw_texts=[]
    for idx, row in wk_df.iterrows():
        t = str(row["Текст"])
        found = classify_topics(t)
        if not found: 
            continue
        s_sign = sentiment_sign(t, row["_rating10"])
        kw_texts.append(t)
        for tp in found:
            topics_rows.append({
                "topic": tp, "idx": idx, "sent": s_sign, 
                "rating10": row["_rating10"], "channel": row["Источник"]
            })
            for s in sentence_split(t):
                if any(re.search(p, s, re.IGNORECASE) for p in TOPIC_PATTERNS[tp]):
                    qm = quotes_map.setdefault(tp, {"pos": None, "neg": None})
                    if sentiment_sign(s, row["_rating10"]) == "negative" and not qm["neg"]:
                        qm["neg"] = s
                    if sentiment_sign(s, row["_rating10"]) == "positive" and not qm["pos"]:
                        qm["pos"] = s
    if not topics_rows:
        return pd.DataFrame(columns=["topic","mentions","share","avg10","neg_text_share","by_channel_json"]), quotes_map, []
    tmp = pd.DataFrame(topics_rows)
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
    # по каналам (json)
    bych = (tmp.groupby(["topic","channel"])
              .agg(cnt=("topic","size"), avg10=("rating10","mean"))
              .reset_index())
    by_channel_json=[]
    for tp, sub in bych.groupby("topic"):
        d = {r["channel"]: {"count": int(r["cnt"]), "avg10": round(float(r["avg10"]),2)} for _, r in sub.iterrows()}
        by_channel_json.append({"topic": tp, "by_channel_json": json.dumps(d, ensure_ascii=False)})
    agg = agg.merge(pd.DataFrame(by_channel_json), on="topic", how="left")
    # keywords
    keywords = extract_keywords(kw_texts, topn=20)
    return agg.sort_values("mentions", ascending=False), quotes_map, keywords

# --- Core analysis ---
def analyze_week(df: pd.DataFrame, start: dt.date, end: dt.date):
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
        "by_channel": wk.groupby("Источник")["_rating10"].agg(["count","mean"]).round(2).sort_values("count", ascending=False).reset_index().to_dict("records")
    }
    return wk, agg

def summarize_trends(hist: pd.DataFrame, week_start: dt.date):
    iso_y, iso_w, _ = week_start.isocalendar()
    def pick(ptype, pkey):
        if not pkey: return None
        m = hist[(hist["period_type"]==ptype) & (hist["period_key"]==pkey)]
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

def upsert_history(agg, week_start: dt.date):
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

# --- Topics history I/O ---
def topics_append(week_key: str, df_topics: pd.DataFrame):
    ensure_tab(HISTORY_SHEET_ID, TOPICS_TAB, ["week_key","topic","mentions","share_pct","avg10","neg_text_share_pct","by_channel_json"])
    if df_topics.empty: 
        return
    rows = df_topics.assign(week_key=week_key)\
                    [["week_key","topic","mentions","share","avg10","neg_text_share","by_channel_json"]].values.tolist()
    SHEETS.values().append(spreadsheetId=HISTORY_SHEET_ID,
                           range=f"{TOPICS_TAB}!A:G",
                           valueInputOption="RAW",
                           body={"values": rows}).execute()

def load_topics_history():
    try:
        res = SHEETS.values().get(spreadsheetId=HISTORY_SHEET_ID, range=f"{TOPICS_TAB}!A:G").execute()
        vals = res.get("values", [])
        if len(vals) <= 1: 
            return pd.DataFrame(columns=["week_key","topic","mentions","share_pct","avg10","neg_text_share_pct","by_channel_json"])
        th = pd.DataFrame(vals[1:], columns=vals[0])
        # numeric
        for c in ["mentions","share_pct","avg10","neg_text_share_pct"]:
            th[c] = pd.to_numeric(th[c], errors="coerce")
        return th
    except Exception:
        return pd.DataFrame(columns=["week_key","topic","mentions","share_pct","avg10","neg_text_share_pct","by_channel_json"])

# --- Charts ---
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

def plot_trends(topics_hist, history_df, out_path):
    # последние 8 недель, топ-5 тем по сумме упоминаний
    if topics_hist.empty:
        return None
    # week_key вида YYYY-Wxx → сортируем
    def week_order(k):
        try:
            y, w = k.split("-W"); return int(y)*100 + int(w)
        except: return 0
    recent_weeks = sorted(topics_hist["week_key"].unique(), key=week_order)[-8:]
    th = topics_hist[topics_hist["week_key"].isin(recent_weeks)].copy()
    top_topics = (th.groupby("topic")["mentions"].sum().sort_values(ascending=False).head(5).index.tolist())
    pv = th[th["topic"].isin(top_topics)].pivot_table(index="week_key", columns="topic", values="mentions", aggfunc="sum").fillna(0)
    # строим
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

# --- HTML ---
def html_report(agg, trends, topics_tbl_html, quotes_html, keywords):
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
    body += topics_tbl_html
    if keywords:
        kw = ", ".join([w for w,_ in keywords[:20]])
        body += f"<p><b>Ключевые слова недели:</b> {kw}</p>"
    body += quotes_html
    body += "<p><i>История и сравнения ведутся в Google Sheet (листы <b>history</b> и <b>topics_history</b>). PNG-графики во вложениях.</i></p>"
    return body

def topics_table_html(topics_df, prev_df=None):
    if topics_df.empty:
        return "<h3>Темы недели</h3><p>Нет данных</p>"
    # дельты к предыдущей неделе (если есть)
    prev = None
    if prev_df is not None and not prev_df.empty:
        prev = prev_df.set_index("topic")
    rows=[]
    for _, r in topics_df.head(10).iterrows():
        d_m = ""
        d_n = ""
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

# --- Mail ---
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

# --- Main ---
def main():
    file_id, name, fdate = latest_reviews_file()
    blob = download_file(file_id)
    tmp = "/tmp/reviews.xls"
    with open(tmp, "wb") as f: f.write(blob)

    today = dt.date.today()
    w_start, w_end = last_week_range(today)
    ensure_history_sheet()

    df = load_reviews_df(tmp)
    wk, agg = analyze_week(df, w_start, w_end)
    upsert_history(agg, w_start)
    hist = history_read()
    trends = summarize_trends(hist, w_start)

    # Темы/цитаты/ключевые слова
    topics_df, quotes, keywords = summarize_topics(wk)

    # предыдущая неделя для дельт в таблице
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

    # сохраняем историю тем + by_channel_json
    wk_key = f"{w_start.isocalendar()[0]}-W{w_start.isocalendar()[1]}"
    topics_append(wk_key, topics_df)

    # таблица по темам + цитаты
    topics_html = topics_table_html(topics_df, prev_topics)
    quotes_html_parts=[]
    for tp, q in quotes.items():
        parts=[]
        if q.get("pos"): parts.append(f"<b>+</b> «{q['pos']}»")
        if q.get("neg"): parts.append(f"<b>–</b> «{q['neg']}»")
        if parts:
            quotes_html_parts.append(f"<p><b>{tp}:</b> " + " / ".join(parts) + "</p>")
    quotes_html = ("<h3>Цитаты</h3>" + "".join(quotes_html_parts)) if quotes_html_parts else ""

    # графики
    charts = []
    topics_hist = load_topics_history()
    p1 = "/tmp/topics_week.png";  plot_topics_bar(topics_df, p1); charts.append(p1)
    p2 = "/tmp/topics_trends.png"; plot_trends(topics_hist, hist, p2); charts.append(p2)
    charts = [p for p in charts if os.path.exists(p)]

    subject = f"ARTSTUDIO Nevsky. Анализ отзывов за неделю {w_start:%d.%m}–{w_end:%d.%m}"
    html = html_report(agg, trends, topics_html, quotes_html, keywords)
    send_email(subject, html, attachments=charts)

if __name__ == "__main__":
    main()
