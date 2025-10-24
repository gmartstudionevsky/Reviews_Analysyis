import os, json
import pandas as pd
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SA_PATH    = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
SA_CONTENT = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON_CONTENT")

if SA_CONTENT and SA_CONTENT.strip().startswith("{"):
    CREDS = Credentials.from_service_account_info(json.loads(SA_CONTENT), scopes=SCOPES)
else:
    CREDS = Credentials.from_service_account_file(SA_PATH, scopes=SCOPES)

SHEETS = build("sheets","v4",credentials=CREDS).spreadsheets()
SPREAD = os.environ["SHEETS_HISTORY_ID"]
TAB    = "surveys_history"
HEADER = ["week_key","param","responses","avg5","avg10","promoters","detractors","nps"]

def get_df():
    res = SHEETS.values().get(spreadsheetId=SPREAD, range=f"{TAB}!A:H").execute()
    vals = res.get("values", [])
    return pd.DataFrame(vals[1:], columns=vals[0]) if len(vals)>1 else pd.DataFrame(columns=HEADER)

def put_df(df):
    SHEETS.values().clear(spreadsheetId=SPREAD, range=f"{TAB}!A2:H").execute()
    if not df.empty:
        SHEETS.values().append(
            spreadsheetId=SPREAD, range=f"{TAB}!A2", valueInputOption="RAW",
            body={"values": df[HEADER].values.tolist()}
        ).execute()

def main():
    df = get_df()
    if df.empty: return
    for c in ["responses","avg5","avg10","promoters","detractors","nps"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

    # coalesce по (week_key,param): выбрать строку с max responses
    def pick(g):
        idx = g["responses"].fillna(0).astype(float).idxmax()
        return g.loc[idx, HEADER]
    clean = (df.groupby(["week_key","param"], as_index=False, group_keys=False)
               .apply(pick)
               .reset_index(drop=True))

    put_df(clean)
    print(f"[OK] Cleaned: {len(df)} -> {len(clean)} rows")

if __name__ == "__main__":
    main()
