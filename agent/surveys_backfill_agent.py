# agent/surveys_backfill_agent.py
import json
import os, io, re, sys, datetime as dt
from datetime import date
import pandas as pd

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# наш модуль ядра анкет
try:
    from agent.surveys_core import parse_and_aggregate_weekly, SURVEYS_TAB
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(__file__))
    from surveys_core import parse_and_aggregate_weekly, SURVEYS_TAB

# =========================
# ENV & Google API clients
# =========================
SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/spreadsheets",
]

# поддержка двух способов: файл (GOOGLE_SERVICE_ACCOUNT_JSON) или прямой контент (GOOGLE_SERVICE_ACCOUNT_JSON_CONTENT)
SA_PATH    = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
SA_CONTENT = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON_CONTENT")

if SA_CONTENT and SA_CONTENT.strip().startswith("{"):
    CREDS = Credentials.from_service_account_info(json.loads(SA_CONTENT), scopes=SCOPES)
else:
    CREDS = Credentials.from_service_account_file(SA_PATH, scopes=SCOPES)

DRIVE  = build("drive",  "v3", credentials=CREDS)
SHEETS = build("sheets", "v4", credentials=CREDS).spreadsheets()

DRIVE_FOLDER_ID  = os.environ["DRIVE_FOLDER_ID"]
HISTORY_SHEET_ID = os.environ["SHEETS_HISTORY_ID"]

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
        return pd.DataFrame(vals[1:], columns=vals[0]) if len(vals)>1 else pd.DataFrame()
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

# ==================
# Drive helpers
# ==================
WEEKLY_RE = re.compile(r"^Report_(\d{2})-(\d{2})-(\d{4})\.xlsx$", re.IGNORECASE)
HIST_HINT = re.compile(r"(history|истор)", re.IGNORECASE)

def list_reports_in_folder(folder_id: str):
    """Возвращает список (id, name, modifiedTime) всех xlsx в папке."""
    items=[]
    pageToken=None
    while True:
        res = DRIVE.files().list(
            q=f"'{folder_id}' in parents and trashed=false and mimeType contains 'spreadsheet'",
            fields="nextPageToken, files(id,name,modifiedTime,mimeType)",
            pageSize=1000,
            pageToken=pageToken
        ).execute()
        items.extend(res.get("files", []))
        pageToken = res.get("nextPageToken")
        if not pageToken: break
    # Google считает «Google Sheets» тоже spreadsheet — нам нужны XLSX
    out=[]
    for it in items:
        name = it["name"]
        if name.lower().endswith(".xlsx") or name.lower().endswith(".xls"):
            out.append((it["id"], name, it.get("modifiedTime")))
    return out

def drive_download(file_id: str) -> bytes:
    req = DRIVE.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
    return buf.getvalue()

# ==================
# Core backfill
# ==================
HEADER = ["week_key","param","responses","avg5","avg10","promoters","detractors","nps"]

def load_existing_keys() -> set[tuple[str,str]]:
    df = gs_get_df(SURVEYS_TAB, "A:H")
    if df.empty: return set()
    return set(zip(df.get("week_key",[]), df.get("param",[])))

def rows_from_agg(df: pd.DataFrame) -> list[list]:
    rows=[]
    for _, r in df.iterrows():
        rows.append([
            str(r["week_key"]),
            str(r["param"]),
            int(r["responses"]) if pd.notna(r["responses"]) else 0,
            (None if pd.isna(r["avg5"]) else float(r["avg5"])),
            (None if pd.isna(r["avg10"]) else float(r["avg10"])),
            int(r["promoters"]) if "promoters" in r and pd.notna(r["promoters"]) else None,
            int(r["detractors"]) if "detractors" in r and pd.notna(r["detractors"]) else None,
            (None if ("nps" not in r or pd.isna(r["nps"])) else float(r["nps"])),
        ])
    return rows

def process_excel_bytes(blob: bytes) -> pd.DataFrame:
    xls = pd.ExcelFile(io.BytesIO(blob))

    def choose_surveys_sheet(xls_file: pd.ExcelFile) -> pd.DataFrame:
        preferred = ["Оcenки гостей", "Оценки гостей", "Оценки", "Ответы", "Анкеты", "Responses"]
        for name in xls_file.sheet_names:
            if name.strip().lower() in {p.lower() for p in preferred}:
                return pd.read_excel(xls_file, name)
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
        pick = best_name or xls_file.sheet_names[0]
        return pd.read_excel(xls_file, pick)

    df_raw = choose_surveys_sheet(xls)
    _, agg = parse_and_aggregate_weekly(df_raw)
    return agg


def main():
    ensure_tab(HISTORY_SHEET_ID, SURVEYS_TAB, HEADER)
    existing = load_existing_keys()

    files = list_reports_in_folder(DRIVE_FOLDER_ID)
    weekly_files, history_files = [], []
    for fid, name, mtime in files:
        if WEEKLY_RE.match(name):
            weekly_files.append((fid, name, mtime))
        elif HIST_HINT.search(name):
            history_files.append((fid, name, mtime))
    # сначала прогоняем исторический файл(ы), затем недельные
    ordered = history_files + sorted(weekly_files, key=lambda t: t[1])

    total_new = 0
    for fid, name, _ in ordered:
        try:
            blob = drive_download(fid)
            agg = process_excel_bytes(blob)
            if agg.empty:
                continue
            # фильтр дублей по (week_key,param)
            need = []
            for _, r in agg.iterrows():
                k = (str(r["week_key"]), str(r["param"]))
                if k not in existing:
                    need.append(r)
                    existing.add(k)
            if not need:
                continue
            rows = rows_from_agg(pd.DataFrame(need))
            gs_append(SURVEYS_TAB, "A:H", rows)
            total_new += len(rows)
        except Exception as e:
            # просто продолжаем следующий файл
            print(f"[WARN] Failed {name}: {e}")

    print(f"[OK] Surveys backfill: appended {total_new} new rows into '{SURVEYS_TAB}'.")

if __name__ == "__main__":
    main()
