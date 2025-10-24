# agent/backfill_full_history.py
# Бэкфилл "весь период": history(week/month/quarter/year) + sources_history(week, source)
import os, io, re, json, datetime as dt
import pandas as pd
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# === ENV ===
SCOPES = ["https://www.googleapis.com/auth/drive.readonly",
          "https://www.googleapis.com/auth/spreadsheets"]
CREDS = Credentials.from_service_account_file(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"], scopes=SCOPES)
DRIVE = build("drive", "v3", credentials=CREDS)
SHEETS = build("sheets", "v4", credentials=CREDS).spreadsheets()

DRIVE_FOLDER_ID   = os.environ["DRIVE_FOLDER_ID"]
SHEETS_HISTORY_ID = os.environ["SHEETS_HISTORY_ID"]
FILE_NAME         = os.environ.get("FILE_NAME", "Reviews_alltime.xls")
START_DATE        = os.environ.get("START_DATE")  # "YYYY-MM-DD" (опционально)
END_DATE          = os.environ.get("END_DATE")    # "YYYY-MM-DD" (опционально)

# === Tabs / Columns ===
HISTORY_TAB  = "history"
SOURCES_TAB  = "sources_history"

EXPECTED = ["Дата","Рейтинг","Источник","Автор","Код языка","Текст отзыва","Наличие ответа"]
RENAMES  = {"Дата":"Дата","Рейтинг":"Рейтинг","Источник":"Источник","Автор":"Автор",
            "Код языка":"Код языка","Текст отзыва":"Текст","Наличие ответа":"Ответ"}

# === Sentiment правила (те же, что в недельном агенте)
import math, re
NEG_CUES = re.compile(
    r"\bне\b|\bнет\b|проблем|ошиб|задерж|долг|ждал|ждали|"
    r"сложн|плохо|неудоб|путан|непонят|неясн|не приш|не сработ|не открыл|не откры|скромн|бедн|мало",
    re.IGNORECASE
)
POS_LEX = re.compile(r"отличн|прекрасн|замечат|идеальн|классн|любезн|комфортн|удобн|чисто|тихо|вкусн|быстро", re.IGNORECASE)
NEG_LEX = re.compile(r"ужасн|плох|грязн|громк|холодн|жарко|слаб|плохо|долго|скучн|бедн|мало|дорог", re.IGNORECASE)

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

# === Utils ===
def ensure_tab(tab: str, header: list[str]):
    meta = SHEETS.get(spreadsheetId=SHEETS_HISTORY_ID).execute()
    tabs = [s["properties"]["title"] for s in meta.get("sheets", [])]
    if tab not in tabs:
        SHEETS.batchUpdate(spreadsheetId=SHEETS_HISTORY_ID,
                           body={"requests":[{"addSheet":{"properties":{"title":tab}}}]}).execute()
        SHEETS.values().update(spreadsheetId=SHEETS_HISTORY_ID,
                               range=f"{tab}!A1:{chr(64+len(header))}1",
                               valueInputOption="RAW",
                               body={"values":[header]}).execute()

def append_rows(tab: str, a1_range: str, rows: list[list]):
    if not rows: return
    SHEETS.values().append(spreadsheetId=SHEETS_HISTORY_ID,
                           range=f"{tab}!{a1_range}",
                           valueInputOption="RAW",
                           body={"values": rows}).execute()

def drive_find_file_by_name(name: str):
    q = f"'{DRIVE_FOLDER_ID}' in parents and name = '{name}' and trashed=false"
    items = DRIVE.files().list(q=q, fields="files(id,name)").execute().get("files", [])
    if not items:
        raise RuntimeError(f"Файл '{name}' не найден в папке.")
    return items[0]["id"]

def drive_download(file_id: str) -> bytes:
    req = DRIVE.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
    return buf.getvalue()

def normalize_to10(s: pd.Series) -> pd.Series:
    num = pd.to_numeric(
        s.astype(str).str.replace(",", ".", regex=False).str.extract(r"([-+]?\d*\.?\d+)")[0],
        errors="coerce"
    )
    vmax, vmin = num.max(skipna=True), num.min(skipna=True)
    if pd.isna(vmax): return num
    if vmax <= 5: return num*2
    if vmax <= 10: return num
    if vmax <= 100 and vmin >= 0: return num/10
    if vmax <= 1: return num*10
    return num.clip(upper=10)

def load_df(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
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
    if missing:
        raise RuntimeError(f"Нет ожидаемых колонок: {missing}")
    return df[[*RENAMES.values()]]

# === History existing
def history_existing():
    ensure_tab(HISTORY_TAB, ["period_type","period_key","reviews","avg10","pos","neu","neg"])
    try:
        vals = SHEETS.values().get(spreadsheetId=SHEETS_HISTORY_ID, range=f"{HISTORY_TAB}!A:G").execute().get("values", [])
        return pd.DataFrame(vals[1:], columns=vals[0]) if len(vals)>1 else pd.DataFrame(columns=["period_type","period_key"])
    except:
        return pd.DataFrame(columns=["period_type","period_key"])

def sources_existing():
    ensure_tab(SOURCES_TAB, ["week_key","source","reviews","avg10","pos","neu","neg"])
    try:
        vals = SHEETS.values().get(spreadsheetId=SHEETS_HISTORY_ID, range=f"{SOURCES_TAB}!A:G").execute().get("values", [])
        return pd.DataFrame(vals[1:], columns=vals[0]) if len(vals)>1 else pd.DataFrame(columns=["week_key","source"])
    except:
        return pd.DataFrame(columns=["week_key","source"])

# === Backfill ===
def main():
    # 1) Download
    file_id = drive_find_file_by_name(FILE_NAME)
    blob = drive_download(file_id)
    tmp = "/tmp/reviews_all.xls"
    with open(tmp, "wb") as f: f.write(blob)

    # 2) Load & clean
    df = load_df(tmp)
    df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce")
    if START_DATE:
        start = pd.to_datetime(START_DATE)
    else:
        start = df["Дата"].min()
    if END_DATE:
        end = pd.to_datetime(END_DATE)
    else:
        end = df["Дата"].max()
    df = df[df["Дата"].between(start, end)].copy()
    if df.empty:
        raise RuntimeError("В указанном диапазоне дат нет строк.")

    df["Рейтинг10"] = normalize_to10(df["Рейтинг"])
    df["Источник"] = df["Источник"].astype(str).str.strip()
    # sentiment
    df["_sent"] = df.apply(lambda r: sentiment_sign(r["Текст"], r["Рейтинг10"]), axis=1)

    # week key
    iso = df["Дата"].dt.isocalendar()
    df["_iso_y"] = iso["year"].astype(int)
    df["_iso_w"] = iso["week"].astype(int)
    df["_week_key"] = df["_iso_y"].astype(str) + "-W" + df["_iso_w"].astype(str)

    # month/quarter/year keys
    df["_month_key"]   = df["Дата"].dt.strftime("%Y-%m")
    q = (df["Дата"].dt.month.sub(1)//3 + 1).astype(int)
    df["_quarter_key"] = df["Дата"].dt.year.astype(str) + "-Q" + q.astype(str)
    df["_year_key"]    = df["Дата"].dt.year.astype(str)

    # 3) Build weekly history rows
    hist = history_existing()
    have_weeks = set(hist.loc[hist["period_type"]=="week","period_key"]) if not hist.empty else set()
    week_rows = []
    for (y,w), sub in df.groupby(["_iso_y","_iso_w"], sort=True):
        week_key = f"{y}-W{int(w)}"
        if week_key in have_weeks: 
            continue
        n = int(len(sub))
        avg10 = round(float((sub["Рейтинг10"]).mean()), 2) if n else None  # не взвешиваем — все по 1
        pos = int((sub["_sent"]=="positive").sum())
        neu = int((sub["_sent"]=="neutral").sum())
        neg = int((sub["_sent"]=="negative").sum())
        week_rows.append(["week", week_key, n, avg10, pos, neu, neg])

    append_rows(HISTORY_TAB, "A:G", week_rows)

    # 4) Build weekly by source → sources_history
    src_exist = sources_existing()
    have_src_pairs = set(zip(src_exist.get("week_key",[]), src_exist.get("source",[]))) if not src_exist.empty else set()
    src_rows = []
    for (wk, src), sub in df.groupby(["_week_key","Источник"], sort=True):
        if (wk, src) in have_src_pairs:
            continue
        n = int(len(sub))
        avg10 = round(float(sub["Рейтинг10"].mean()), 2) if n else None
        pos = int((sub["_sent"]=="positive").sum())
        neu = int((sub["_sent"]=="neutral").sum())
        neg = int((sub["_sent"]=="negative").sum())
        src_rows.append([wk, src, n, avg10, pos, neu, neg])

    append_rows(SOURCES_TAB, "A:G", src_rows)

    # 5) Month / Quarter / Year in history (append if missing) — взвешенно по неделям файла
    hist = history_existing()  # refresh
    have_months   = set(hist.loc[hist["period_type"]=="month","period_key"])
    have_quarters = set(hist.loc[hist["period_type"]=="quarter","period_key"])
    have_years    = set(hist.loc[hist["period_type"]=="year","period_key"])

    def pack_period(ptype: str, key_col: str):
        rows=[]
        for key, sub in df.groupby(key_col):
            if (ptype=="month" and key in have_months) or \
               (ptype=="quarter" and key in have_quarters) or \
               (ptype=="year" and key in have_years):
                continue
            n = int(len(sub))
            avg10 = round(float(sub["Рейтинг10"].mean()), 2) if n else None
            pos = int((sub["_sent"]=="positive").sum())
            neu = int((sub["_sent"]=="neutral").sum())
            neg = int((sub["_sent"]=="negative").sum())
            rows.append([ptype, key, n, avg10, pos, neu, neg])
        return rows

    month_rows   = pack_period("month",   "_month_key")
    quarter_rows = pack_period("quarter", "_quarter_key")
    year_rows    = pack_period("year",    "_year_key")

    append_rows(HISTORY_TAB, "A:G", month_rows + quarter_rows + year_rows)

    print(f"Backfill done: weeks + sources; appended: weeks={len(week_rows)}, source_pairs={len(src_rows)}, "
          f"months={len(month_rows)}, quarters={len(quarter_rows)}, years={len(year_rows)}")

if __name__ == "__main__":
    main()
