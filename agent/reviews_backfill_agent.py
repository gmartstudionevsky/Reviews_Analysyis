# agent/reviews_backfill_agent.py
from __future__ import annotations
from typing import Any, Dict, List
import os, io, json
from datetime import datetime
import pandas as pd

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# наши модули
from agent.text_analytics_core import build_semantic_row, parse_date_any
from agent.metrics_core import (
    build_history,
    build_sources_history,
    build_semantic_agg_aspects_period,
    build_semantic_agg_pairs_period,
)

# названия вкладок в Sheets
TAB_RAW            = "reviews_semantic_raw"
TAB_HISTORY        = "history"
TAB_SOURCES        = "sources_history"
TAB_ASPECTS        = "semantic_agg_aspects_period"
TAB_PAIRS          = "semantic_agg_pairs_period"

# Google API scopes
SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/spreadsheets",
]

def get_google_clients():
    """
    Берём сервисный аккаунт:
    - либо через GOOGLE_SERVICE_ACCOUNT_JSON_CONTENT (прямой JSON),
    - либо через GOOGLE_SERVICE_ACCOUNT_JSON (путь к sa.json).
    """
    sa_path = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    sa_content = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON_CONTENT")

    if sa_content and sa_content.strip().startswith("{"):
        creds = Credentials.from_service_account_info(json.loads(sa_content), scopes=SCOPES)
    else:
        if not sa_path:
            raise RuntimeError("No GOOGLE_SERVICE_ACCOUNT_JSON(_CONTENT) provided for service account auth.")
        creds = Credentials.from_service_account_file(sa_path, scopes=SCOPES)

    drive  = build("drive", "v3", credentials=creds)
    sheets = build("sheets", "v4", credentials=creds).spreadsheets()
    return drive, sheets


def download_reviews_excel(drive, folder_id: str, filename_base: str) -> pd.DataFrame:
    """
    Находит файл в Google Drive по имени filename_base + .xls / .xlsx,
    скачивает, читает лист 'Отзывы'.
    Если листа 'Отзывы' нет, берёт первый лист.
    Проверяет обязательные столбцы.
    """
    for ext in (".xls", ".xlsx"):
        q = (
            f"'{folder_id}' in parents and name = '{filename_base}{ext}' and "
            "mimeType != 'application/vnd.google-apps.folder' and trashed = false"
        )
        resp = drive.files().list(
            q=q,
            fields="files(id,name,modifiedTime,size)",
            pageSize=1,
            orderBy="modifiedTime desc",
        ).execute()
        items = resp.get("files", [])
        if not items:
            continue

        file_id = items[0]["id"]

        # скачали двоично
        req = drive.files().get_media(fileId=file_id)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, req)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        content = buf.getvalue()

        # читаем Excel
        bio = io.BytesIO(content)
        try:
            xl = pd.ExcelFile(bio)
        except Exception:
            xl = pd.ExcelFile(bio, engine="xlrd")

        sheet_name = "Отзывы" if "Отзывы" in xl.sheet_names else xl.sheet_names[0]
        df = pd.read_excel(io.BytesIO(content), sheet_name=sheet_name)

        required_cols = ["Дата", "Рейтинг", "Источник", "Текст отзыва"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"Missing required columns {missing} in sheet '{sheet_name}'")

        return df

    raise RuntimeError(f"File {filename_base}.xls(x) not found in Drive folder {folder_id}")


def build_semantic_raw_table(src_df: pd.DataFrame) -> pd.DataFrame:
    """
    src_df — то, что мы прочитали из Excel (все отзывы за всю историю).
    Возвращает DataFrame для вкладки reviews_semantic_raw.
    Параллельно выкидываем строки без текста, и явно фильтруем странные будущие даты.
    """
    out_rows: List[Dict[str, Any]] = []
    skipped = 0

    today = pd.Timestamp.utcnow().normalize()

    for idx, row in src_df.iterrows():
        text_val = row.get("Текст отзыва", "")
        if not isinstance(text_val, str) or not text_val.strip():
            skipped += 1
            continue

        payload = {
            "date":       row.get("Дата"),
            "source":     row.get("Источник"),
            "rating_raw": row.get("Рейтинг"),
            "text":       text_val,
            "lang_hint":  row.get("Код языка", ""),
        }

        sem_row = build_semantic_row(payload)

        # дата sanity: если дата сильно из будущего (> сегодня + 7 дней) — дропаём
        dtt = parse_date_any(payload["date"])
        if dtt > (today + pd.Timedelta(days=7)):
            skipped += 1
            continue

        out_rows.append(sem_row)

    print(f"[INFO] semantic rows built = {len(out_rows)}, skipped = {skipped}")
    if not out_rows:
        # вернём пустую структуру, чтобы агент не падал
        return pd.DataFrame(columns=[
            "review_id","date","week_key","month_key","quarter_key","year_key",
            "source","rating10","sentiment_overall",
            "topics_pos","topics_neg","topics_all","pair_tags","quote_candidates",
        ])

    df = pd.DataFrame(out_rows)
    df = df.sort_values(by=["date","source","review_id"], ignore_index=True)
    return df


def ensure_sheet_tab(sheets, spreadsheet_id: str, tab_name: str, header: List[str]):
    """
    Создаёт таб, если его нет. Обновляет шапку (строка 1).
    """
    meta = sheets.get(spreadsheetId=spreadsheet_id).execute()
    have_tabs = [s["properties"]["title"] for s in meta.get("sheets", [])]

    if tab_name not in have_tabs:
        sheets.batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"requests":[{"addSheet":{"properties":{"title":tab_name}}}]}
        ).execute()

    end_col_letter = chr(ord("A") + len(header) - 1)
    sheets.values().update(
        spreadsheetId=spreadsheet_id,
        range=f"{tab_name}!A1:{end_col_letter}1",
        valueInputOption="RAW",
        body={"values":[header]},
    ).execute()


def clear_sheet_data(sheets, spreadsheet_id: str, tab_name: str):
    """
    Чистим всё, начиная со второй строки.
    """
    sheets.values().clear(
        spreadsheetId=spreadsheet_id,
        range=f"{tab_name}!A2:ZZ",
        body={},
    ).execute()


def append_df(sheets, spreadsheet_id: str, tab_name: str, df: pd.DataFrame, cols: List[str]):
    """
    Грузим df (в нужном порядке колонок) построчно через append.
    """
    if df.empty:
        print(f"[WARN] {tab_name}: df is empty, skip append.")
        return

    values = []
    for _, r in df.iterrows():
        row_vals = []
        for c in cols:
            v = r.get(c, "")
            if pd.isna(v):
                v = ""
            row_vals.append(v)
        values.append(row_vals)

    sheets.values().append(
        spreadsheetId=spreadsheet_id,
        range=f"{tab_name}!A2",
        valueInputOption="RAW",
        body={"values": values},
    ).execute()


def run_backfill():
    """
    Главный раннер для GitHub Action.
    Ожидаемые переменные окружения:
        DRIVE_FOLDER_ID        - папка в Google Drive, где лежит Reviews_2019-25.xls(x)
        SHEETS_HISTORY_ID      - ID итоговой гугл-таблицы с хранилищем
        GOOGLE_SERVICE_ACCOUNT_JSON / GOOGLE_SERVICE_ACCOUNT_JSON_CONTENT
    """
    folder_id = os.environ["DRIVE_FOLDER_ID"]
    sheet_id  = os.environ["SHEETS_HISTORY_ID"]

    drive, sheets = get_google_clients()

    # 1. стягиваем историю отзывов
    src_df = download_reviews_excel(drive, folder_id, filename_base="Reviews_2019-25")
    print(f"[INFO] downloaded source rows: {len(src_df)}")

    # 2. строим сырую семантику
    raw_df = build_semantic_raw_table(src_df)
    print(f"[INFO] final semantic_raw rows: {len(raw_df)} "
          f"date range: {raw_df['date'].min() if not raw_df.empty else 'n/a'} .. "
          f"{raw_df['date'].max() if not raw_df.empty else 'n/a'}")

    # 3. строим агрегаты
    hist_df    = build_history(raw_df)
    src_hist   = build_sources_history(raw_df)
    aspects_df = build_semantic_agg_aspects_period(raw_df)
    pairs_df   = build_semantic_agg_pairs_period(raw_df)

    # 4. пишем в Google Sheets
    # 4.1 reviews_semantic_raw
    raw_header = [
        "review_id","date","week_key","month_key","quarter_key","year_key",
        "source","rating10","sentiment_overall",
        "topics_pos","topics_neg","topics_all","pair_tags","quote_candidates",
    ]
    ensure_sheet_tab(sheets, sheet_id, TAB_RAW, raw_header)
    clear_sheet_data(sheets, sheet_id, TAB_RAW)
    append_df(sheets, sheet_id, TAB_RAW, raw_df, raw_header)

    # 4.2 history
    hist_header = ["period_type","period_key","reviews","avg10","pos","neu","neg"]
    ensure_sheet_tab(sheets, sheet_id, TAB_HISTORY, hist_header)
    clear_sheet_data(sheets, sheet_id, TAB_HISTORY)
    append_df(sheets, sheet_id, TAB_HISTORY, hist_df, hist_header)

    # 4.3 sources_history
    src_header = ["week_key","source","reviews","avg10","pos","neu","neg"]
    ensure_sheet_tab(sheets, sheet_id, TAB_SOURCES, src_header)
    clear_sheet_data(sheets, sheet_id, TAB_SOURCES)
    append_df(sheets, sheet_id, TAB_SOURCES, src_hist, src_header)

    # 4.4 semantic_agg_aspects_period
    asp_header = [
        "period_type","period_key","source_scope","aspect_key",
        "mentions_total","pos_mentions","neg_mentions","neu_mentions",
        "pos_share","neg_share","pos_weight","neg_weight",
    ]
    ensure_sheet_tab(sheets, sheet_id, TAB_ASPECTS, asp_header)
    clear_sheet_data(sheets, sheet_id, TAB_ASPECTS)
    append_df(sheets, sheet_id, TAB_ASPECTS, aspects_df, asp_header)

    # 4.5 semantic_agg_pairs_period
    pair_header = [
        "period_type","period_key","source_scope",
        "pair_key","category","distinct_reviews","example_quote",
    ]
    ensure_sheet_tab(sheets, sheet_id, TAB_PAIRS, pair_header)
    clear_sheet_data(sheets, sheet_id, TAB_PAIRS)
    append_df(sheets, sheet_id, TAB_PAIRS, pairs_df, pair_header)

    print("[INFO] backfill completed.")


if __name__ == "__main__":
    run_backfill()
