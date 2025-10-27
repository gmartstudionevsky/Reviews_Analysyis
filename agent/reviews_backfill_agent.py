# agent/reviews_backfill_agent.py

import os, io, json
from typing import Any, List, Dict, Tuple
from datetime import datetime

import pandas as pd
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# импорт фабрики строки семантики
try:
    from agent.text_analytics_core import build_semantic_row
except Exception:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from text_analytics_core import build_semantic_row

SEMANTIC_TAB = "reviews_semantic_raw"
KPI_TAB      = "history"
ASPECTS_TAB  = "semantic_agg_aspects_period"
PAIRS_TAB    = "semantic_agg_pairs_period"
SOURCES_TAB  = "sources_history"

PERIOD_LEVELS: List[Tuple[str, str]] = [
    ("week",    "week_key"),
    ("month",   "month_key"),
    ("quarter", "quarter_key"),
    ("year",    "year_key"),
]

SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/spreadsheets",
]

def get_google_clients():
    sa_path    = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    sa_content = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON_CONTENT")
    if sa_content and sa_content.strip().startswith("{"):
        creds = Credentials.from_service_account_info(json.loads(sa_content), scopes=SCOPES)
    else:
        if not sa_path:
            raise RuntimeError("No GOOGLE_SERVICE_ACCOUNT_JSON(_CONTENT) provided.")
        creds = Credentials.from_service_account_file(sa_path, scopes=SCOPES)
    drive  = build("drive",  "v3", credentials=creds)
    sheets = build("sheets", "v4", credentials=creds).spreadsheets()
    return drive, sheets

def _download_drive_file(drive, file_id: str) -> bytes:
    req = drive.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()

# === A) IO: чтение Excel из Google Drive (лист 'Отзывы') ===
def drive_download_reviews_file(drive, folder_id: str, filename_base: str) -> pd.DataFrame:
    for ext in (".xls", ".xlsx"):
        q = (
            f"'{folder_id}' in parents and "
            f"name = '{filename_base}{ext}' and "
            f"mimeType != 'application/vnd.google-apps.folder' and trashed = false"
        )
        resp = drive.files().list(
            q=q, fields="files(id,name,modifiedTime,size)", pageSize=1, orderBy="modifiedTime desc"
        ).execute()
        items = resp.get("files", [])
        if not items:
            continue
        file_id = items[0]["id"]
        content = _download_drive_file(drive, file_id)
        bio = io.BytesIO(content)
        try:
            xls = pd.ExcelFile(bio)
        except Exception:
            xls = pd.ExcelFile(bio, engine="xlrd")  # .xls
        sheet = "Отзывы" if "Отзывы" in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(io.BytesIO(content), sheet_name=sheet)
        # строгая проверка колонок
        need = {"Дата","Рейтинг","Источник","Текст отзыва"}
        miss = [c for c in need if c not in set(df.columns)]
        if miss:
            raise RuntimeError(f"Нет колонок {miss} на листе '{sheet}'")
        return df
    raise RuntimeError(f"Файл {filename_base}.xls(x) не найден в папке {folder_id}")

def ensure_tab_exists(sheets, spreadsheet_id: str, tab_name: str, header: List[str]):
    meta = sheets.get(spreadsheetId=spreadsheet_id).execute()
    existing = [s["properties"]["title"] for s in meta.get("sheets", [])]
    if tab_name not in existing:
        sheets.batchUpdate(spreadsheetId=spreadsheet_id,
                           body={"requests":[{"addSheet":{"properties":{"title":tab_name}}}]}).execute()
    end_col = chr(ord("A") + len(header) - 1)
    sheets.values().update(
        spreadsheetId=spreadsheet_id,
        range=f"{tab_name}!A1:{end_col}1",
        valueInputOption="RAW",
        body={"values":[header]},
    ).execute()

def clear_tab_data(sheets, spreadsheet_id: str, tab_name: str, start_cell: str = "A2"):
    sheets.values().clear(
        spreadsheetId=spreadsheet_id,
        range=f"{tab_name}!{start_cell}:ZZ",
        body={},
    ).execute()

def append_df_to_sheet(sheets, spreadsheet_id: str, tab_name: str, df: pd.DataFrame, cols: List[str]):
    if df.empty:
        return
    vals = []
    for _, r in df.iterrows():
        vals.append([("" if pd.isna(r.get(c, "")) else r.get(c, "")) for c in cols])
    sheets.values().append(
        spreadsheetId=spreadsheet_id,
        range=f"{tab_name}!A2",
        valueInputOption="RAW",
        body={"values": vals},
    ).execute()

# ---------- RAW (1 строка = 1 отзыв)

# === B) RAW: одна строка = один отзыв ===
def build_reviews_semantic_raw_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in raw_df.columns}
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        return None
    col_date   = pick("дата","date")
    col_rate   = pick("рейтинг","rating","оценка","score")
    col_source = pick("источник","source","площадка")
    col_lang   = pick("код языка","lang","язык","language","language code")
    col_text   = pick("текст отзыва","text","отзыв","review text")

    if not all([col_date, col_rate, col_source, col_text]):
        raise RuntimeError("Нет обязательных столбцов: дата/рейтинг/источник/текст")

    today = pd.Timestamp.utcnow().normalize()
    rows, skipped = [], 0

    for idx, rr in raw_df.iterrows():
        payload = {
            "date": rr[col_date],
            "source": rr[col_source],
            "rating_raw": rr[col_rate],
            "text": (rr[col_text] if pd.notna(rr[col_text]) else ""),
            "lang_hint": (rr[col_lang] if col_lang else ""),
        }
        try:
            row = build_semantic_row(payload)
            # фильтр «странных» будущих дат: > сегодня + 7 дней => скип
            dt = pd.to_datetime(row["date"], errors="coerce")
            if pd.isna(dt) or dt > (today + pd.Timedelta(days=7)):
                skipped += 1
                continue
            rows.append(row)
        except Exception as e:
            skipped += 1
            print(f"[WARN] skip row #{idx}: {e}")

    print(f"[INFO] built {len(rows)} rows; skipped={skipped}")
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=[
            "review_id","date","week_key","month_key","quarter_key","year_key",
            "source","rating10","sentiment_overall",
            "topics_pos","topics_neg","topics_all","pair_tags","quote_candidates"
        ])
    return df.sort_values(by=["date","source","review_id"], ignore_index=True)


# ---------- Aggregations

def compute_history_from_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ptype, col in PERIOD_LEVELS:
        if col not in df_raw.columns: continue
        sub = df_raw.dropna(subset=[col])
        if sub.empty: continue
        for pkey, g in sub.groupby(col, dropna=True):
            rows.append({
                "period_type": ptype,
                "period_key":  pkey,
                "reviews":     int(g["review_id"].nunique()),
                "avg10":       (round(float(g["rating10"].dropna().mean()),2) if g["rating10"].notna().any() else ""),
                "pos":         int((g["sentiment_overall"]=="pos").sum()),
                "neu":         int((g["sentiment_overall"]=="neu").sum()),
                "neg":         int((g["sentiment_overall"]=="neg").sum()),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["period_type","period_key","reviews","avg10","pos","neu","neg"])
    order = {p:i for i,(p,_) in enumerate(PERIOD_LEVELS)}
    out["__o"] = out["period_type"].map(order).fillna(999)
    return out.sort_values(by=["__o","period_key"], ascending=[True,True], ignore_index=True).drop(columns="__o")

def _parse_list(cell):
    if isinstance(cell, list): return cell
    if pd.isna(cell): return []
    try: return json.loads(cell)
    except Exception: return []

def compute_sources_history_weekly_from_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    sub = df_raw.dropna(subset=["week_key","source"])
    if sub.empty:
        return pd.DataFrame(columns=["week_key","source","reviews","avg10","pos","neu","neg"])
    rows = []
    for (wk, src), g in sub.groupby(["week_key","source"], dropna=True):
        rows.append({
            "week_key": wk,
            "source":   src,
            "reviews":  int(g["review_id"].nunique()),
            "avg10":    (round(float(g["rating10"].dropna().mean()),2) if g["rating10"].notna().any() else ""),
            "pos":      int((g["sentiment_overall"]=="pos").sum()),
            "neu":      int((g["sentiment_overall"]=="neu").sum()),
            "neg":      int((g["sentiment_overall"]=="neg").sum()),
        })
    return pd.DataFrame(rows).sort_values(by=["week_key","source"], ascending=[True,True], ignore_index=True)

def compute_semantic_agg_aspects_period_from_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sources = sorted([s for s in df_raw["source"].dropna().unique().tolist() if s])
    for ptype, col in PERIOD_LEVELS:
        subp = df_raw.dropna(subset=[col])
        if subp.empty: continue
        for scope in ["all"] + sources:
            subs = subp if scope=="all" else subp[subp["source"]==scope]
            if subs.empty: continue
            for pkey, g in subs.groupby(col, dropna=True):
                pos_ids = set(g.loc[g["sentiment_overall"]=="pos","review_id"])
                neg_ids = set(g.loc[g["sentiment_overall"]=="neg","review_id"])
                pos_total, neg_total = len(pos_ids), len(neg_ids)
                stats: Dict[str, Dict[str,set]] = {}
                for _, r in g.iterrows():
                    rid = r["review_id"]
                    a_all = set(_parse_list(r.get("topics_all")))
                    a_pos = set(_parse_list(r.get("topics_pos")))
                    a_neg = set(_parse_list(r.get("topics_neg")))
                    a_neu = a_all - a_pos - a_neg
                    for a in a_all: stats.setdefault(a, {"m":set(),"p":set(),"n":set(),"u":set()})["m"].add(rid)
                    for a in a_pos: stats.setdefault(a, {"m":set(),"p":set(),"n":set(),"u":set()})["p"].add(rid)
                    for a in a_neg: stats.setdefault(a, {"m":set(),"p":set(),"n":set(),"u":set()})["n"].add(rid)
                    for a in a_neu: stats.setdefault(a, {"m":set(),"p":set(),"n":set(),"u":set()})["u"].add(rid)
                for a, d in stats.items():
                    mentions = len(d["m"])
                    p_m = len(d["p"]); n_m = len(d["n"]); u_m = len(d["u"])
                    rows.append({
                        "period_type": ptype, "period_key": pkey, "source_scope": scope,
                        "aspect_key": a,
                        "mentions_total": mentions,
                        "pos_mentions": p_m, "neg_mentions": n_m, "neu_mentions": u_m,
                        "pos_share": round(p_m/mentions,4) if mentions else 0.0,
                        "neg_share": round(n_m/mentions,4) if mentions else 0.0,
                        "pos_weight": round(len(d["p"].intersection(pos_ids))/pos_total,4) if pos_total else 0.0,
                        "neg_weight": round(len(d["n"].intersection(neg_ids))/neg_total,4) if neg_total else 0.0,
                    })
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=[
            "period_type","period_key","source_scope","aspect_key",
            "mentions_total","pos_mentions","neg_mentions","neu_mentions",
            "pos_share","neg_share","pos_weight","neg_weight",
        ])
    order = {p:i for i,(p,_) in enumerate(PERIOD_LEVELS)}
    out["__o"] = out["period_type"].map(order).fillna(999)
    out["__s"] = out["source_scope"].apply(lambda s: 0 if s=="all" else 1)
    return out.sort_values(
        by=["__o","period_key","__s","mentions_total"],
        ascending=[True,True,True,False],
        ignore_index=True
    ).drop(columns=["__o","__s"])

def _parse_pairs(cell):
    if isinstance(cell, list): return cell
    if pd.isna(cell): return []
    try: return json.loads(cell)
    except Exception: return []

def compute_semantic_agg_pairs_period_from_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sources = sorted([s for s in df_raw["source"].dropna().unique().tolist() if s])
    for ptype, col in PERIOD_LEVELS:
        subp = df_raw.dropna(subset=[col])
        if subp.empty: continue
        for scope in ["all"] + sources:
            subs = subp if scope=="all" else subp[subp["source"]==scope]
            if subs.empty: continue
            for pkey, g in subs.groupby(col, dropna=True):
                pair2ids: Dict[Tuple[str,str], set] = {}
                pair2q:   Dict[Tuple[str,str], str] = {}
                for _, r in g.iterrows():
                    rid = r["review_id"]
                    pairs = _parse_pairs(r.get("pair_tags"))
                    quotes = _parse_list(r.get("quote_candidates"))
                    q = quotes[0] if quotes else ""
                    for p in pairs:
                        a = (p.get("a") or "").strip(); b=(p.get("b") or "").strip(); cat=(p.get("cat") or "").strip()
                        if not a or not b or not cat: continue
                        key = ("|".join(sorted([a,b])), cat)
                        pair2ids.setdefault(key,set()).add(rid)
                        if key not in pair2q and q: pair2q[key] = q
                for (pair_key, cat), ids in pair2ids.items():
                    rows.append({
                        "period_type": ptype, "period_key": pkey, "source_scope": scope,
                        "pair_key": pair_key, "category": cat,
                        "distinct_reviews": len(ids),
                        "example_quote": pair2q.get((pair_key, cat), ""),
                    })
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=[
            "period_type","period_key","source_scope","pair_key","category","distinct_reviews","example_quote"
        ])
    order = {p:i for i,(p,_) in enumerate(PERIOD_LEVELS)}
    out["__o"] = out["period_type"].map(order).fillna(999)
    out["__s"] = out["source_scope"].apply(lambda s: 0 if s=="all" else 1)
    return out.sort_values(
        by=["__o","period_key","__s","distinct_reviews"],
        ascending=[True,True,True,False],
        ignore_index=True
    ).drop(columns=["__o","__s"])

def run_backfill():
    DRIVE_FOLDER_ID   = os.environ["DRIVE_FOLDER_ID"]
    SHEETS_HISTORY_ID = os.environ["SHEETS_HISTORY_ID"]
    drive, sheets = get_google_clients()

    # 1) чтение источника
    raw = drive_download_reviews_file(drive, DRIVE_FOLDER_ID, filename_base="Reviews_2019-25")

    # sanity-print: сколько всего отзывов и с непустым текстом
    total_rows = len(raw)
    nonempty_text = int(raw["Текст отзыва"].notna().sum()) if "Текст отзыва" in raw.columns else total_rows
    print(f"[INFO] Raw rows: {total_rows}, with text: {nonempty_text}")

    # 2) сырые семантические строки
    df_raw = build_reviews_semantic_raw_df(raw)
    print(f"[INFO] semantic_raw rows: {len(df_raw)}  date_range: {df_raw['date'].min()} .. {df_raw['date'].max()}")

    # 3) агрегаты
    kpi_df     = compute_history_from_raw(df_raw)
    aspects_df = compute_semantic_agg_aspects_period_from_raw(df_raw)
    pairs_df   = compute_semantic_agg_pairs_period_from_raw(df_raw)
    src_df     = compute_sources_history_weekly_from_raw(df_raw)

    # 4) запись в Sheets
    semantic_header = [
        "review_id","date","week_key","month_key","quarter_key","year_key",
        "source","rating10","sentiment_overall",
        "topics_pos","topics_neg","topics_all","pair_tags","quote_candidates",
    ]
    ensure_tab_exists(sheets, SHEETS_HISTORY_ID, SEMANTIC_TAB, semantic_header)
    clear_tab_data(sheets, SHEETS_HISTORY_ID, SEMANTIC_TAB, start_cell="A2")
    append_df_to_sheet(sheets, SHEETS_HISTORY_ID, SEMANTIC_TAB, df_raw, semantic_header)

    kpi_header = ["period_type","period_key","reviews","avg10","pos","neu","neg"]
    ensure_tab_exists(sheets, SHEETS_HISTORY_ID, KPI_TAB, kpi_header)
    clear_tab_data(sheets, SHEETS_HISTORY_ID, KPI_TAB, start_cell="A2")
    append_df_to_sheet(sheets, SHEETS_HISTORY_ID, KPI_TAB, kpi_df, kpi_header)

    aspects_header = [
        "period_type","period_key","source_scope","aspect_key",
        "mentions_total","pos_mentions","neg_mentions","neu_mentions",
        "pos_share","neg_share","pos_weight","neg_weight",
    ]
    ensure_tab_exists(sheets, SHEETS_HISTORY_ID, ASPECTS_TAB, aspects_header)
    clear_tab_data(sheets, SHEETS_HISTORY_ID, ASPECTS_TAB, start_cell="A2")
    append_df_to_sheet(sheets, SHEETS_HISTORY_ID, ASPECTS_TAB, aspects_df, aspects_header)

    pairs_header = ["period_type","period_key","source_scope","pair_key","category","distinct_reviews","example_quote"]
    ensure_tab_exists(sheets, SHEETS_HISTORY_ID, PAIRS_TAB, pairs_header)
    clear_tab_data(sheets, SHEETS_HISTORY_ID, PAIRS_TAB, start_cell="A2")
    append_df_to_sheet(sheets, SHEETS_HISTORY_ID, PAIRS_TAB, pairs_df, pairs_header)

    sources_header = ["week_key","source","reviews","avg10","pos","neu","neg"]
    ensure_tab_exists(sheets, SHEETS_HISTORY_ID, SOURCES_TAB, sources_header)
    clear_tab_data(sheets, SHEETS_HISTORY_ID, SOURCES_TAB, start_cell="A2")
    append_df_to_sheet(sheets, SHEETS_HISTORY_ID, SOURCES_TAB, src_df, sources_header)

    print("[INFO] Backfill done.")

if __name__ == "__main__":
    run_backfill()
