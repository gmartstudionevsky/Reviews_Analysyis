import os, re, io, json, datetime as dt
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ====== AUTH / CLIENTS ======
SCOPES = ["https://www.googleapis.com/auth/drive.readonly",
          "https://www.googleapis.com/auth/spreadsheets"]
CREDS = Credentials.from_service_account_file(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"], scopes=SCOPES)
DRIVE = build("drive", "v3", credentials=CREDS)
SHEETS = build("sheets", "v4", credentials=CREDS).spreadsheets()
DRIVE_FOLDER_ID = os.environ["DRIVE_FOLDER_ID"]
HISTORY_SHEET_ID = os.environ["SHEETS_HISTORY_ID"]
HISTORY_TAB = os.environ.get("HISTORY_TAB","history")
TOPICS_TAB  = os.environ.get("TOPICS_TAB","topics_history")
FILE_NAME   = os.environ.get("FILE_NAME","Reviews_2024-25.xls")
START_DATE  = os.environ.get("START_DATE","2024-10-19")
END_DATE    = os.environ.get("END_DATE","2025-10-19")

# ====== SETTINGS (совпадает с weekly) ======
EXPECTED = ["Дата","Рейтинг","Источник","Автор","Код языка","Текст отзыва","Наличие ответа"]
RENAMES  = {"Дата":"Дата","Рейтинг":"Рейтинг","Источник":"Источник","Автор":"Автор",
            "Код языка":"Код языка","Текст отзыва":"Текст","Наличие ответа":"Ответ"}

TOPIC_PATTERNS = {
    "Локация":[r"расположен|расположение|рядом|вокзал|невск"],
    "Персонал":[r"персонал|сотрудник|администрат|ресепшен|ресепшн|менеджер"],
    "Чистота":[r"чисто|уборк|бель[её]|пятн|полотенц|постель"],
    "Номер/оснащение":[r"номер|кухн|посудомо|стирал|плита|чайник|фен|сейф"],
    "Сантехника/вода":[r"душ|слив|кран|смесител|бойлер|водонагрев|давлен|температур|канализац|засор"],
    "AC/шум":[r"кондицион|вентиляц|шум|тихо|громк"],
    "Завтраки":[r"завтрак|ланч-?бокс|ресторан"],
    "Чек-ин/вход":[r"заселен|заезд|чек-?ин|вход|инструкц|селф|код|ключ|карта"],
    "Wi-Fi/интернет":[r"wi-?fi|wifi|вай-?фай|интернет|парол"],
    "Цена/ценность":[r"цена|стоимост|дорог|дешев|соотношен.*цена"],
}
POS_LEX = re.compile(r"отличн|прекрасн|замечат|идеальн|классн|любезн|комфортн|удобн|чисто|тихо|вкусн|быстро", re.IGNORECASE)
NEG_LEX = re.compile(r"ужасн|плох|грязн|громк|холодн|жарко|слаб|плохо|долго|скучн|бедн|мало|дорог|проблем|не ", re.IGNORECASE)

def ensure_tab(tab, header):
    meta = SHEETS.get(spreadsheetId=HISTORY_SHEET_ID).execute()
    tabs = [s["properties"]["title"] for s in meta.get("sheets", [])]
    if tab not in tabs:
        SHEETS.batchUpdate(spreadsheetId=HISTORY_SHEET_ID,
                           body={"requests":[{"addSheet":{"properties":{"title":tab}}}]}).execute()
        SHEETS.values().update(spreadsheetId=HISTORY_SHEET_ID,
                               range=f"{tab}!A1:{chr(64+len(header))}1",
                               valueInputOption="RAW",
                               body={"values":[header]}).execute()

def drive_find_file_by_name(name):
    q = f"'{DRIVE_FOLDER_ID}' in parents and name = '{name}' and trashed=false"
    items = DRIVE.files().list(q=q, fields="files(id,name)").execute().get("files", [])
    if not items: raise RuntimeError(f"Файл '{name}' не найден в папке.")
    return items[0]["id"], items[0]["name"]

def drive_download(file_id):
    req = DRIVE.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
    return buf.getvalue()

def normalize_to10(x: pd.Series):
    s = pd.to_numeric(x.astype(str).str.replace(",", ".", regex=False).str.extract(r"([-+]?\d*\.?\d+)")[0], errors="coerce")
    vmax, vmin = s.max(skipna=True), s.min(skipna=True)
    if pd.isna(vmax): return s
    if vmax <= 5: return s*2
    if vmax <= 10: return s
    if vmax <= 100 and vmin >= 0: return s/10
    if vmax <= 1: return s*10
    return s.clip(upper=10)

def load_df(path):
    df = pd.read_excel(path)  # .xls/.xlsx
    low = {str(c).strip().lower():c for c in df.columns}
    ren = {}
    for k in EXPECTED:
        hit = low.get(k.lower())
        if hit: ren[hit] = RENAMES[k]
        else:
            cand = next((c for lc,c in low.items() if k.split()[0].lower() in lc), None)
            if cand: ren[cand] = RENAMES[k]
    df = df.rename(columns=ren)
    miss = [v for v in RENAMES.values() if v not in df.columns]
    if miss: raise RuntimeError(f"Нет ожидаемых колонок: {miss}")
    return df[[*RENAMES.values()]]

def sentiment_sign(text, rating10):
    pos = len(POS_LEX.findall(str(text)))
    neg = len(NEG_LEX.findall(str(text)))
    if neg - pos >= 2: return "negative"
    if pos - neg >= 1 and neg == 0: return "positive"
    if pd.notna(rating10):
        if rating10 < 6: return "negative"
        if rating10 >= 8: return "positive"
        return "neutral"
    return "neutral"

def classify_topics(text):
    tl = str(text).lower()
    hits=set()
    for tp, pats in TOPIC_PATTERNS.items():
        for p in pats:
            if re.search(p, tl, re.IGNORECASE): hits.add(tp); break
    return hits

def append_rows(tab, range_a1, rows):
    if not rows: return
    SHEETS.values().append(spreadsheetId=HISTORY_SHEET_ID, range=f"{tab}!{range_a1}",
                           valueInputOption="RAW", body={"values": rows}).execute()

def history_existing():
    ensure_tab(HISTORY_TAB, ["period_type","period_key","reviews","avg10","pos","neu","neg"])
    try:
        v = SHEETS.values().get(spreadsheetId=HISTORY_SHEET_ID, range=f"{HISTORY_TAB}!A:G").execute().get("values", [])
        return pd.DataFrame(v[1:], columns=v[0]) if len(v)>1 else pd.DataFrame(columns=["period_type","period_key"])
    except: return pd.DataFrame(columns=["period_type","period_key"])

def topics_existing():
    ensure_tab(TOPICS_TAB, ["week_key","topic","mentions","share_pct","avg10","neg_text_share_pct","by_channel_json"])
    try:
        v = SHEETS.values().get(spreadsheetId=HISTORY_SHEET_ID, range=f"{TOPICS_TAB}!A:G").execute().get("values", [])
        return pd.DataFrame(v[1:], columns=v[0]) if len(v)>1 else pd.DataFrame(columns=["week_key","topic"])
    except: return pd.DataFrame(columns=["week_key","topic"])

def main():
    file_id, _ = drive_find_file_by_name(FILE_NAME)
    blob = drive_download(file_id)
    tmp = "/tmp/backfill.xls"
    with open(tmp,"wb") as f: f.write(blob)

    df = load_df(tmp)
    df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce", dayfirst=True)
    start = pd.to_datetime(START_DATE)
    end   = pd.to_datetime(END_DATE) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df = df[df["Дата"].between(start, end)].copy()
    df["_rating10"] = normalize_to10(df["Рейтинг"])
    df["_sent_rule"] = df.apply(lambda r: sentiment_sign(r["Текст"], r["_rating10"]), axis=1)

    # ISO week keys + helpers
    iso = df["Дата"].dt.isocalendar()
    df["_iso_y"] = iso["year"]
    df["_iso_w"] = iso["week"]
    df["_week_key"] = df["_iso_y"].astype(str) + "-W" + iso["week"].astype(str)

    # ===== AGG: weeks =====
    hist = history_existing()
    existing_weeks = set(hist.loc[hist["period_type"]=="week","period_key"])
    week_rows = []
    topics_rows = []
    topics_exist = topics_existing()
    existing_topics_keys = set(zip(topics_exist.get("week_key",[]), topics_exist.get("topic",[])))

    for (y,w), sub in df.groupby(["_iso_y","_iso_w"]):
        week_key = f"{y}-W{int(w)}"
        # метрики недели
        avg10 = round(sub["_rating10"].mean(),2) if len(sub) else None
        pos = int((sub["_sent_rule"]=="positive").sum())
        neu = int((sub["_sent_rule"]=="neutral").sum())
        neg = int((sub["_sent_rule"]=="negative").sum())
        row = ["week", week_key, len(sub), avg10, pos, neu, neg]
        # upsert недельной строки (и MONTH/QTR/YEAR ниже отдельно)
        if week_key not in existing_weeks:
            week_rows.append(row)

        # темы/каналы недели
        # классификация
        topics_list=[]
        for i,r in sub.iterrows():
            ts = classify_topics(r["Текст"])
            if not ts: continue
            for tp in ts:
                topics_list.append({"topic":tp, "ch": r["Источник"], "rating": r["_rating10"],
                                    "sent": sentiment_sign(r["Текст"], r["_rating10"])})
        if topics_list:
            tdf = pd.DataFrame(topics_list)
            agg = (tdf.groupby("topic")
                     .agg(mentions=("topic","size"),
                          avg10=("rating","mean"),
                          neg_text_share=("sent", lambda s: (s=="negative").mean()))
                     .reset_index())
            agg["share_pct"] = (agg["mentions"]/len(sub)*100).round(1)
            agg["avg10"] = agg["avg10"].round(2)
            agg["neg_text_share_pct"] = (agg["neg_text_share"]*100).round(1)
            # by channel
            ch = (tdf.groupby(["topic","ch"])
                    .agg(cnt=("topic","size"), avg10=("rating","mean")).reset_index())
            for _, r in agg.iterrows():
                tp = r["topic"]
                if (week_key, tp) in existing_topics_keys:
                    continue
                subch = ch[ch["topic"]==tp]
                bych = {row["ch"]: {"count": int(row["cnt"]), "avg10": round(float(row["avg10"]),2)} 
                        for _,row in subch.iterrows()}
                topics_rows.append([week_key, tp, int(r["mentions"]), float(r["share_pct"]),
                                    float(r["avg10"]), float(r["neg_text_share_pct"]),
                                    json.dumps(bych, ensure_ascii=False)])

    # записываем разом
    ensure_tab(HISTORY_TAB, ["period_type","period_key","reviews","avg10","pos","neu","neg"])
    if week_rows:
        append_rows(HISTORY_TAB, "A:G", week_rows)
        existing_weeks |= {r[1] for r in week_rows}

    ensure_tab(TOPICS_TAB, ["week_key","topic","mentions","share_pct","avg10","neg_text_share_pct","by_channel_json"])
    if topics_rows:
        append_rows(TOPICS_TAB, "A:G", topics_rows)

    # ===== AGG: months / quarters / years (idempotent append если ещё нет) =====
    def upsert_period(ptype, keys, frame):
        already = set(hist.loc[hist["period_type"]==ptype, "period_key"])
        rows=[]
        for k, grp in frame.items():
            if k in already: continue
            r = [ptype, k, int(grp["count"]), round(grp["avg10"],2), int(grp["pos"]), int(grp["neu"]), int(grp["neg"])]
            rows.append(r)
        if rows:
            append_rows(HISTORY_TAB, "A:G", rows)

    # подготовка
    df["_month_key"] = df["Дата"].dt.strftime("%Y-%m")
    q = (df["Дата"].dt.month.sub(1)//3 + 1).astype(int)
    df["_quarter_key"] = df["Дата"].dt.year.astype(str) + "-Q" + q.astype(str)
    df["_year_key"] = df["Дата"].dt.year.astype(str)

    def rollup(group_col):
        g = df.groupby(group_col).agg(
            count=("Рейтинг","size"),
            avg10=("_rating10","mean"),
            pos=("_sent_rule", lambda s: (s=="positive").sum()),
            neu=("_sent_rule", lambda s: (s=="neutral").sum()),
            neg=("_sent_rule", lambda s: (s=="negative").sum()),
        )
        return g.to_dict(orient="index")

    hist = history_existing()  # refresh после записи недель
    upsert_period("month",   df["_month_key"].unique(),   rollup("_month_key"))
    upsert_period("quarter", df["_quarter_key"].unique(), rollup("_quarter_key"))
    upsert_period("year",    df["_year_key"].unique(),    rollup("_year_key"))

if __name__ == "__main__":
    main()
