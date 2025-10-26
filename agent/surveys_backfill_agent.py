# agent/surveys_backfill_agent.py
import json
import os
import re
import sys
import datetime as dt
from datetime import date
import pandas as pd

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from surveys_core import parse_and_aggregate_weekly, SURVEYS_TAB, PARAM_ORDER

SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/spreadsheets",
]

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

def ensure_tab(spreadsheet_id: str, tab_name: str, header: list[str]):
    meta = SHEETS.get(spreadsheetId=spreadsheet_id).execute()
    tabs = [s["properties"]["title"] for s in meta.get("sheets", [])]
    if tab_name not in tabs:
        SHEETS.batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"requests": [{"addSheet": {"properties": {"title": tab_name}}}]}
        ).execute()
        SHEETS.values().update(
            spreadsheetId=spreadsheet_id,
            range=f"{tab_name}!A1:{chr(64 + len(header))}1",
            valueInputOption="RAW",
            body={"values": [header]}
        ).execute()

def gs_get_df(tab: str, a1: str) -> pd.DataFrame:
    try:
        res = SHEETS.values().get(spreadsheetId=HISTORY_SHEET_ID, range=f"{tab}!{a1}").execute()
        vals = res.get("values", [])
        return pd.DataFrame(vals[1:], columns=vals[0]) if len(vals) > 1 else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def gs_append(tab: str, a1: str, rows: list[list]):
    if not rows:
        return
    SHEETS.values().append(
        spreadsheetId=HISTORY_SHEET_ID,
        range=f"{tab}!{a1}",
        valueInputOption="RAW",
        body={"values": rows}
    ).execute()

WEEKLY_RE = re.compile(r"^Report_(\d{2})-(\d{2})-(\d{4})\.xlsx$", re.IGNORECASE)
HIST_HINT = re.compile(r"(history|истор)", re.IGNORECASE)

def list_reports_in_folder(folder_id: str):
    res = DRIVE.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id,name,modifiedTime)",
        pageSize=1000
    ).execute()
    files = [(f["id"], f["name"], f["modifiedTime"]) for f in res.get("files", [])]
    return sorted(files, key=lambda x: x[2])

def load_existing_keys() -> set[tuple[str, str]]:
    df = gs_get_df(SURVEYS_TAB, "A:H")
    if df.empty:
        return set()
    return set(zip(df["week_key"].astype(str), df["param"].astype(str)))

HEADER = ["week_key", "param", "responses", "avg5", "promoters", "detractors", "nps"]

def rows_from_agg(df: pd.DataFrame) -> list[list]:
    rows = []
    for _, r in df.iterrows():
        rows.append([
            str(r["week_key"]),
            str(r["param"]),
            int(r["responses"]) if pd.notna(r["responses"]) else 0,
            float(r["avg5"]) if pd.notna(r["avg5"]) else None,
            int(r["promoters"]) if pd.notna(r.get("promoters")) else None,
            int(r["detractors"]) if pd.notna(r.get("detractors")) else None,
            float(r["nps"]) if pd.notna(r.get("nps")) else None,
        ])
    return rows

def process_excel_bytes(blob: bytes) -> pd.DataFrame:
    xls = pd.ExcelFile(io.BytesIO(blob))
    preferred = ["Оценки гостей", "Оценки гостей", "Оценки", "Ответы", "Анкеты", "Responses"]
    sheet = next((s for s in xls.sheet_names if s.strip().lower() in [p.lower() for p in preferred]), None)
    if not sheet:
        candidates = ["Дата", "Дата анкетирования", "Комментарий", "Средняя оценка", "№ 1", "№ 2", "№ 3"]
        best_name, best_score = None, -1
        for name in xls.sheet_names:
            probe = pd.read_excel(xls, name, nrows=1)
            cols = [str(c) for c in probe.columns]
            score = sum(any(cand.lower() in c.lower() for cand in candidates) for c in cols)
            if score > best_score:
                best_name, best_score = name, score
        sheet = best_name or xls.sheet_names[0]
    df_raw = pd.read_excel(xls, sheet)
    _, agg = parse_and_aggregate_weekly(df_raw)
    return agg

def main():
    ensure_tab(HISTORY_SHEET_ID, SURVEYS_TAB, HEADER)
    existing = load_existing_keys()
    files = list_reports_in_folder(DRIVE_FOLDER_ID)
    weekly_files = [f for f in files if WEEKLY_RE.match(f[1])]
    history_files = [f for f in files if HIST_HINT.search(f[1])]
    ordered = history_files + sorted(weekly_files, key=lambda t: t[1])
    total_new = 0
    for fid, name, _ in ordered:
        try:
            blob = drive_download(fid)
            agg = process_excel_bytes(blob)
            if agg.empty:
                continue
            need = []
            for _, r in agg.iterrows():
                k = (str(r["week_key"]), str(r["param"]))
                if k not in existing:
                    need.append(r)
                    existing.add(k)
            if not need:
                continue
            rows = rows_from_agg(pd.DataFrame(need))
            gs_append(SURVEYS_TAB, "A:G", rows)
            total_new += len(rows)
        except Exception as e:
            print(f"[WARN] Failed {name}: {e}")
    print(f"[OK] Surveys backfill: appended {total_new} new rows into '{SURVEYS_TAB}'.")

if __name__ == "__main__":
    main()
