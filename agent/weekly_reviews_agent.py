import os, re, io, datetime as dt
import pandas as pd
from dateutil.relativedelta import relativedelta
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

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

# --- Helpers ---
def parse_date_from_name(name: str):
    m = re.search(r"Reviews_(\d{2})-(\d{2})-(\d{4})\.xls(x)?$", name)
    if not m: return None
    dd, mm, yyyy = map(int, m.groups()[:3])
    return dt.date(yyyy, mm, dd)

def latest_reviews_file():
    # Берём все прямые файлы в папке и фильтруем по шаблону Reviews_dd-mm-yyyy.xls(x)
    regex = re.compile(r"^Reviews_\d{2}-\d{2}-\d{4}\.xls(x)?$", re.IGNORECASE)
    results = DRIVE.files().list(
        q=f"'{DRIVE_FOLDER_ID}' in parents and trashed=false",
        fields="files(id,name,modifiedTime)",
        pageSize=1000
    ).execute()
    items = results.get("files", [])
    items = [(i["id"], i["name"], parse_date_from_name(i["name"])) for i in items if regex.match(i["name"])]
    if not items:
        raise RuntimeError("В папке нет файлов вида Reviews_dd-mm-yyyy.xls(x). Проверьте имя и расположение файла.")
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
    monday = ref - dt.timedelta(days=ref.weekday())  # текущий понедельник
    end = monday - dt.timedelta(days=1)              # прошлая неделя (вс)
    start = end - dt.timedelta(days=6)               # прошлая неделя (пн)
    return start, end

# --- History in Google Sheets ---
def history_read():
    try:
        res = SHEETS.values().get(spreadsheetId=HISTORY_SHEET_ID, range="history!A:G").execute()
        vals = res.get("values", [])
        if not vals: return pd.DataFrame(columns=["period_type","period_key","reviews","avg10","pos","neu","neg"])
        return pd.DataFrame(vals[1:], columns=vals[0])
    except Exception:
        return pd.DataFrame(columns=["period_type","period_key","reviews","avg10","pos","neu","neg"])

def history_append(rows: pd.DataFrame):
    if rows.empty: return
    SHEETS.values().append(
        spreadsheetId=HISTORY_SHEET_ID,
        range="history!A:G",
        valueInputOption="RAW",
        body={"values": rows.values.tolist()}
    ).execute()

# --- Columns (RU) ---
EXPECTED = ["Дата","Рейтинг","Источник","Автор","Код языка","Текст отзыва","Наличие ответа"]
RENAMES = {
    "Дата": "Дата", "Рейтинг":"Рейтинг", "Источник":"Источник", "Автор":"Автор",
    "Код языка":"Код языка", "Текст отзыва":"Текст", "Наличие ответа":"Ответ"
}

def load_reviews_df(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)  # поддерживает .xls/.xlsx при наличии xlrd/openpyxl
    # мягкий маппинг по русским заголовкам
    lowmap = {str(c).strip().lower(): c for c in df.columns}
    rename = {}
    for k in EXPECTED:
        hit = lowmap.get(k.lower())
        if hit:
            rename[hit] = RENAMES[k]
        else:
            # попытка частичного совпадения
            cand = next((c for lc, c in lowmap.items() if k.split()[0].lower() in lc), None)
            if cand: rename[cand] = RENAMES[k]
    df = df.rename(columns=rename)
    missing = [v for v in RENAMES.values() if v not in df.columns]
    if missing:
        raise RuntimeError(f"В файле не найдены ожидаемые колонки: {missing}")
    return df[[*RENAMES.values()]]

def analyze_week(df: pd.DataFrame, start: dt.date, end: dt.date):
    df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce", dayfirst=True).dt.date
    wk = df[(df["Дата"]>=start) & (df["Дата"]<=end)].copy()
    wk["_rating10"] = normalize_to10(wk["Рейтинг"])
    wk["_sent"] = wk["_rating10"].apply(lambda r: "positive" if r>=8 else ("neutral" if r>=6 else "negative"))
    agg = {
        "start": start, "end": end,
        "reviews": len(wk),
        "avg10": round(wk["_rating10"].mean(),2) if len(wk) else None,
        "pos": int((wk["_sent"]=="positive").sum()),
        "neu": int((wk["_sent"]=="neutral").sum()),
        "neg": int((wk["_sent"]=="negative").sum()),
        "by_channel": wk.groupby("Источник")["_rating10"].agg(["count","mean"]).round(2)
                        .sort_values("count", ascending=False).reset_index().to_dict("records")
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

def html_report(agg, trends):
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

    body += "<h3>Каналы</h3><ul>"
    for r in agg["by_channel"][:8]:
        body += f"<li>{r['Источник']}: {int(r['count'])} шт., {r['mean']}/10</li>"
    body += "</ul>"
    body += "<p><i>История и сравнения ведутся в Google Sheet (лист <b>history</b>).</i></p>"
    return body

def send_email(subject, html_body):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = os.environ.get("SMTP_FROM", os.environ["SMTP_USER"])
    msg["To"] = ", ".join(RECIPIENTS)
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    with smtplib.SMTP(os.environ.get("SMTP_HOST","smtp.gmail.com"), int(os.environ.get("SMTP_PORT",587))) as s:
        s.starttls()
        s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
        s.sendmail(msg["From"], RECIPIENTS, msg.as_string())

def main():
    file_id, name, fdate = latest_reviews_file()
    blob = download_file(file_id)
    tmp = "/tmp/reviews.xls"
    with open(tmp, "wb") as f: f.write(blob)

    # Прошлая неделя относительно сегодняшнего дня (скрипт запускается по понедельникам)
    today = dt.date.today()
    w_start, w_end = last_week_range(today)

    df = load_reviews_df(tmp)
    _, agg = analyze_week(df, w_start, w_end)
    upsert_history(agg, w_start)
    trends = summarize_trends(history_read(), w_start)

    # Тема письма по вашей идентике:
    subject = f"ARTSTUDIO Nevsky. Анализ отзывов за неделю {w_start:%d.%m}–{w_end:%d.%m}"
    html = html_report(agg, trends)
    send_email(subject, html)

if __name__ == "__main__":
    main()
