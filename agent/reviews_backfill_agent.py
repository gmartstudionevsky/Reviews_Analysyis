# agent/reviews_backfill_agent.py
#
# Бэкфилл истории отзывов (2019-2025) в Google Sheets.
#
# Что делает:
#   1. Берёт файл Reviews_2019-25.xls из папки DRIVE_FOLDER_ID в Google Drive.
#   2. Читает лист "Отзывы".
#   3. Прогоняет через build_exploded_reviews_df() из text_analytics_core:
#        - нормализация языка
#        - тональность (pos/neg/neu)
#        - аспекты (темы)
#        - приведение рейтинга к шкале /5
#        - week_key, month_key, quarter_key, year_key
#   4. Дедупит (review_id + aspect_code).
#   5. Строит агрегаты:
#        a) period_kpi_history: KPI по периодам (неделя / месяц / квартал / год)
#        b) period_aspects_history: частота аспектов и их тональность по периодам
#        c) source_history: репутация по источникам (недельная)
#   6. Полностью ПЕРЕЗАПИСЫВАЕТ в таблице SHEETS_HISTORY_ID вкладки:
#        - reviews_semantic_history
#        - period_kpi_history
#        - period_aspects_history
#        - source_history
#
# Этот скрипт запускается вручную (через отдельный workflow_dispatch или локально).
# В еженедельном отчёте мы бэкфилл не дергаем.

import os
import io
import math
from datetime import datetime
from typing import Any, List, Dict, Tuple

import pandas as pd

import json
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


# =========================
# Импорт ядра семантики отзывов
# =========================
# Нас интересует только build_exploded_reviews_df (и TOPIC_SCHEMA она тянет).
try:
    from agent.text_analytics_core import (
        TOPIC_SCHEMA,
        build_exploded_reviews_df,
    )
except ModuleNotFoundError:
    # fallback если запускаем модуль локально как script без пакета agent
    import sys
    sys.path.append(os.path.dirname(__file__))
    from text_analytics_core import (
        TOPIC_SCHEMA,
        build_exploded_reviews_df,
    )


# =========================
# Настройки листов и колонок
# =========================

SEMANTIC_TAB = "reviews_semantic_history"
KPI_TAB      = "period_kpi_history"
ASPECTS_TAB  = "period_aspects_history"
SOURCES_TAB  = "source_history"

PERIOD_LEVELS = [
    ("week",    "week_key"),
    ("month",   "month_key"),
    ("quarter", "quarter_key"),
    ("year",    "year_key"),
]

SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/spreadsheets",
]

# =========================
# Авторизация в Google API
# =========================

def get_google_clients() -> tuple:
    """
    Создаёт клиенты DRIVE и SHEETS на основе сервисного аккаунта.
    Поддерживает два варианта:
    1. GOOGLE_SERVICE_ACCOUNT_JSON_CONTENT = '{"type":"service_account",...}'
    2. GOOGLE_SERVICE_ACCOUNT_JSON = путь к sa.json
    Возвращает (DRIVE, SHEETS).
    """
    sa_path    = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    sa_content = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON_CONTENT")

    if sa_content and sa_content.strip().startswith("{"):
        creds = Credentials.from_service_account_info(json.loads(sa_content), scopes=SCOPES)
    else:
        if not sa_path:
            raise RuntimeError("No GOOGLE_SERVICE_ACCOUNT_JSON or GOOGLE_SERVICE_ACCOUNT_JSON_CONTENT provided.")
        creds = Credentials.from_service_account_file(sa_path, scopes=SCOPES)

    drive  = build("drive",  "v3", credentials=creds)
    sheets = build("sheets", "v4", credentials=creds).spreadsheets()
    return drive, sheets


# =========================
# Drive helpers
# =========================

def drive_download_reviews_file(drive, folder_id: str, filename: str) -> pd.DataFrame:
    """
    Находит в папке folder_id файл с именем filename (точное совпадение),
    скачивает и читает лист 'Отзывы'.
    Ожидаемый формат столбцов:
      'Дата', 'Рейтинг', 'Источник', 'Автор', 'Код языка', 'Текст отзыва', 'Наличие ответа'
    """
    query = (
        f"'{folder_id}' in parents and "
        f"name = '{filename}' and "
        f"mimeType != 'application/vnd.google-apps.folder' and trashed = false"
    )
    resp = drive.files().list(
        q=query,
        fields="files(id,name,modifiedTime,size)",
        pageSize=5,
        orderBy="modifiedTime desc",
    ).execute()
    files = resp.get("files", [])
    if not files:
        raise RuntimeError(f"Файл {filename} не найден в папке {folder_id}.")

    file_id = files[0]["id"]

    req = drive.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
    buf.seek(0)

    # .xls может требовать движок xlrd → убедимся, что xlrd стоит в requirements
    df = pd.read_excel(buf, sheet_name="Отзывы")
    return df


# =========================
# Sheets helpers
# =========================

def ensure_tab_exists(sheets, spreadsheet_id: str, tab_name: str, header: List[str]):
    """
    Убедиться, что вкладка tab_name есть в таблице spreadsheet_id,
    и записать туда шапку в A1:... .
    Если вкладки нет — создаём её.
    """
    meta = sheets.get(spreadsheetId=spreadsheet_id).execute()
    existing_tabs = [s["properties"]["title"] for s in meta.get("sheets", [])]

    if tab_name not in existing_tabs:
        sheets.batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"requests":[{"addSheet":{"properties":{"title":tab_name}}}]}
        ).execute()

    # теперь пишем шапку
    end_col_letter = chr(ord("A") + len(header) - 1)
    sheets.values().update(
        spreadsheetId=spreadsheet_id,
        range=f"{tab_name}!A1:{end_col_letter}1",
        valueInputOption="RAW",
        body={"values":[header]},
    ).execute()


def clear_tab_data(sheets, spreadsheet_id: str, tab_name: str, start_cell: str = "A2"):
    """
    Чистим всё ниже шапки, чтобы залить историю с нуля.
    """
    sheets.values().clear(
        spreadsheetId=spreadsheet_id,
        range=f"{tab_name}!{start_cell}:ZZ",
        body={},
    ).execute()


def append_df_to_sheet(sheets, spreadsheet_id: str, tab_name: str, df: pd.DataFrame, columns: List[str]):
    """
    Добавляет df в конец вкладки tab_name,
    значения колонок идут в указанном порядке.
    Преобразует NaN → "".
    """
    if df.empty:
        return

    rows = []
    for _, r in df.iterrows():
        row_out = []
        for col in columns:
            val = r.get(col, "")
            if pd.isna(val):
                row_out.append("")
            else:
                row_out.append(val)
        rows.append(row_out)

    sheets.values().append(
        spreadsheetId=spreadsheet_id,
        range=f"{tab_name}!A2",
        valueInputOption="RAW",
        body={"values": rows},
    ).execute()


# =========================
# Агрегация для KPI
# =========================

def build_per_review_df(exploded_df: pd.DataFrame) -> pd.DataFrame:
    """
    Одна строка на отзыв.
    Возвращает DataFrame с колонками:
        review_id, date, week_key, month_key, quarter_key, year_key,
        source, rating_5, sentiment
    """
    if exploded_df.empty:
        return pd.DataFrame(columns=[
            "review_id","date","week_key","month_key","quarter_key","year_key",
            "source","rating_5","sentiment"
        ])

    per_review = (
        exploded_df.groupby("review_id")
        .agg({
            "date": "first",
            "week_key": "first",
            "month_key": "first",
            "quarter_key": "first",
            "year_key": "first",
            "source": "first",
            "rating_5": "mean",
            "sentiment": "first",
        })
        .reset_index()
    )

    return per_review


def compute_period_kpi_history(exploded_df: pd.DataFrame) -> pd.DataFrame:
    """
    KPI по каждому периодическому ключу:
      period_type ('week'/'month'/'quarter'/'year'),
      period_key  ('2025-W42', '2025-10', ...),
      reviews_count,
      avg_rating_5,
      share_pos,
      share_neg
    """
    per_review = build_per_review_df(exploded_df)

    rows = []
    for period_type, col in PERIOD_LEVELS:
        if col not in per_review.columns:
            continue

        tmp = per_review.dropna(subset=[col]).copy()
        if tmp.empty:
            continue

        g = tmp.groupby(col)
        for period_key, gdf in g:
            reviews_count = len(gdf)
            avg_rating_5 = gdf["rating_5"].dropna().mean()

            pos_cnt = (gdf["sentiment"] == "pos").sum()
            neg_cnt = (gdf["sentiment"] == "neg").sum()

            share_pos = (pos_cnt / reviews_count) if reviews_count else 0.0
            share_neg = (neg_cnt / reviews_count) if reviews_count else 0.0

            rows.append({
                "period_type": period_type,
                "period_key": period_key,
                "reviews_count": reviews_count,
                "avg_rating_5": round(avg_rating_5, 4) if avg_rating_5 == avg_rating_5 else "",
                "share_pos": share_pos,
                "share_neg": share_neg,
            })

    out_df = pd.DataFrame(rows, columns=[
        "period_type",
        "period_key",
        "reviews_count",
        "avg_rating_5",
        "share_pos",
        "share_neg",
    ]).sort_values(
        by=["period_type","period_key"],
        ascending=[True, True],
        ignore_index=True,
    )

    return out_df


def compute_period_aspects_history(exploded_df: pd.DataFrame) -> pd.DataFrame:
    """
    По каждому периоду и аспекту считаем:
        mentions_reviews  — сколько уникальных отзывов упомянули аспект
        pos_share / neg_share — доля позитивных / негативных упоминаний
    Возвращает столбцы:
        period_type, period_key,
        aspect_code,
        mentions_reviews,
        pos_share,
        neg_share
    """
    rows = []

    for period_type, col in PERIOD_LEVELS:
        if col not in exploded_df.columns:
            continue

        tmp = exploded_df.dropna(subset=[col]).copy()
        if tmp.empty:
            continue

        # (period_key, aspect_code, sentiment, review_id)
        grp = (
            tmp.groupby([col, "aspect_code", "sentiment", "review_id"])
               .size()
               .reset_index(name="cnt")
        )

        # агрегируем по (period_key, aspect_code, sentiment)
        agg_mentions = (
            grp.groupby([col, "aspect_code", "sentiment"])["review_id"]
               .nunique()
               .reset_index(name="review_mentions")
        )

        # сводная по sentiment
        pivot = agg_mentions.pivot_table(
            index=[col, "aspect_code"],
            columns="sentiment",
            values="review_mentions",
            fill_value=0,
            aggfunc="sum",
        )

        for need in ["pos","neg","neu"]:
            if need not in pivot.columns:
                pivot[need] = 0

        pivot["mentions_total"] = (
            pivot["pos"] + pivot["neg"] + pivot["neu"]
        ).astype(float)

        pivot["pos_share"] = pivot["pos"] / pivot["mentions_total"].replace(0, 1)
        pivot["neg_share"] = pivot["neg"] / pivot["mentions_total"].replace(0, 1)

        pivot = pivot.reset_index()

        for _, r in pivot.iterrows():
            period_key = r[col]
            rows.append({
                "period_type": period_type,
                "period_key": period_key,
                "aspect_code": r["aspect_code"],
                "mentions_reviews": int(r["mentions_total"]),
                "pos_share": r["pos_share"],
                "neg_share": r["neg_share"],
            })

    out_df = pd.DataFrame(rows, columns=[
        "period_type",
        "period_key",
        "aspect_code",
        "mentions_reviews",
        "pos_share",
        "neg_share",
    ]).sort_values(
        by=["period_type","period_key","mentions_reviews"],
        ascending=[True, True, False],
        ignore_index=True,
    )

    return out_df


def compute_source_history_weekly(exploded_df: pd.DataFrame) -> pd.DataFrame:
    """
    История по источникам (недели):
        week_key,
        source,
        reviews_count,
        avg_rating_5,
        share_pos,
        share_neg
    """
    per_review = build_per_review_df(exploded_df)
    tmp = per_review.dropna(subset=["week_key"]).copy()
    if tmp.empty:
        return pd.DataFrame(columns=[
            "week_key","source","reviews_count","avg_rating_5","share_pos","share_neg"
        ])

    rows = []
    for (week_key, source), gdf in tmp.groupby(["week_key","source"]):
        cnt = len(gdf)
        avg_rating_5 = gdf["rating_5"].dropna().mean()

        pos_cnt = (gdf["sentiment"] == "pos").sum()
        neg_cnt = (gdf["sentiment"] == "neg").sum()

        share_pos = (pos_cnt / cnt) if cnt else 0.0
        share_neg = (neg_cnt / cnt) if cnt else 0.0

        rows.append({
            "week_key": week_key,
            "source": source,
            "reviews_count": cnt,
            "avg_rating_5": round(avg_rating_5,4) if avg_rating_5 == avg_rating_5 else "",
            "share_pos": share_pos,
            "share_neg": share_neg,
        })

    out_df = pd.DataFrame(rows, columns=[
        "week_key","source","reviews_count","avg_rating_5","share_pos","share_neg"
    ]).sort_values(
        by=["week_key","reviews_count"],
        ascending=[True, False],
        ignore_index=True,
    )

    return out_df


# =========================
# Основной сценарий backfill
# =========================

def run_backfill():
    """
    1. Скачиваем Reviews_2019-25.xls.
    2. Читаем лист "Отзывы".
    3. Парсим в exploded_df (отзыв × аспект).
    4. Дедупим.
    5. Считаем агрегаты.
    6. Полностью перезаписываем 4 вкладки в SHEETS_HISTORY_ID.
    """
    DRIVE_FOLDER_ID   = os.environ["DRIVE_FOLDER_ID"]
    SHEETS_HISTORY_ID = os.environ["SHEETS_HISTORY_ID"]

    drive, sheets = get_google_clients()

    # 1. выкачиваем xls из Google Drive
    raw_reviews_df = drive_download_reviews_file(
        drive=drive,
        folder_id=DRIVE_FOLDER_ID,
        filename="Reviews_2019-25.xls",
    )

    # 2. семантический разбор
    exploded_df = build_exploded_reviews_df(
        raw_reviews_df=raw_reviews_df,
        topic_schema=TOPIC_SCHEMA,
        add_rating_5scale=True,
    )

    # Если пусто - тоже должны залить пустые вкладки с шапками
    if exploded_df.empty:
        print("[WARN] exploded_df пустой, но всё равно создаём вкладки с пустыми данными.")
        semantic_out = pd.DataFrame(columns=[
            "review_id","date","week_key","month_key","quarter_key","year_key",
            "source","rating_5","sentiment","aspect_code","text",
        ])
        kpi_df = pd.DataFrame(columns=[
            "period_type","period_key","reviews_count","avg_rating_5","share_pos","share_neg",
        ])
        aspects_df = pd.DataFrame(columns=[
            "period_type","period_key","aspect_code","mentions_reviews","pos_share","neg_share",
        ])
        sources_df = pd.DataFrame(columns=[
            "week_key","source","reviews_count","avg_rating_5","share_pos","share_neg",
        ])
    else:
        # 3. дедуп: один отзыв × аспект только один раз
        exploded_df = exploded_df.drop_duplicates(
            subset=["review_id","aspect_code"]
        ).reset_index(drop=True)

        exploded_df = exploded_df.sort_values(
            by=["date","review_id","aspect_code"],
            ascending=[True,True,True],
        ).reset_index(drop=True)

        # 4. данные для SEMANTIC_TAB
        semantic_out = exploded_df[[
            "review_id",
            "date",
            "week_key",
            "month_key",
            "quarter_key",
            "year_key",
            "source",
            "rating_5",
            "sentiment",
            "aspect_code",
            "text",
        ]].copy()

        # дату приводим к строке, чтобы гугл не наигрался с таймзонами
        semantic_out["date"] = semantic_out["date"].dt.strftime("%Y-%m-%d %H:%M:%S")

        # 5. агрегаты
        kpi_df     = compute_period_kpi_history(exploded_df)
        aspects_df = compute_period_aspects_history(exploded_df)
        sources_df = compute_source_history_weekly(exploded_df)

    # 6. запись во вкладки гугл-таблицы
    # 6.1 SEMANTIC_TAB
    semantic_header = [
        "review_id",
        "date",
        "week_key",
        "month_key",
        "quarter_key",
        "year_key",
        "source",
        "rating_5",
        "sentiment",
        "aspect_code",
        "text",
    ]
    ensure_tab_exists(sheets, SHEETS_HISTORY_ID, SEMANTIC_TAB, semantic_header)
    clear_tab_data(sheets, SHEETS_HISTORY_ID, SEMANTIC_TAB, start_cell="A2")
    append_df_to_sheet(sheets, SHEETS_HISTORY_ID, SEMANTIC_TAB, semantic_out, semantic_header)

    # 6.2 KPI_TAB
    kpi_header = [
        "period_type",
        "period_key",
        "reviews_count",
        "avg_rating_5",
        "share_pos",
        "share_neg",
    ]
    ensure_tab_exists(sheets, SHEETS_HISTORY_ID, KPI_TAB, kpi_header)
    clear_tab_data(sheets, SHEETS_HISTORY_ID, KPI_TAB, start_cell="A2")
    append_df_to_sheet(sheets, SHEETS_HISTORY_ID, KPI_TAB, kpi_df, kpi_header)

    # 6.3 ASPECTS_TAB
    aspects_header = [
        "period_type",
        "period_key",
        "aspect_code",
        "mentions_reviews",
        "pos_share",
        "neg_share",
    ]
    ensure_tab_exists(sheets, SHEETS_HISTORY_ID, ASPECTS_TAB, aspects_header)
    clear_tab_data(sheets, SHEETS_HISTORY_ID, ASPECTS_TAB, start_cell="A2")
    append_df_to_sheet(sheets, SHEETS_HISTORY_ID, ASPECTS_TAB, aspects_df, aspects_header)

    # 6.4 SOURCES_TAB
    sources_header = [
        "week_key",
        "source",
        "reviews_count",
        "avg_rating_5",
        "share_pos",
        "share_neg",
    ]
    ensure_tab_exists(sheets, SHEETS_HISTORY_ID, SOURCES_TAB, sources_header)
    clear_tab_data(sheets, SHEETS_HISTORY_ID, SOURCES_TAB, start_cell="A2")
    append_df_to_sheet(sheets, SHEETS_HISTORY_ID, SOURCES_TAB, sources_df, sources_header)

    print("[INFO] Backfill по отзывам завершён успешно.")


if __name__ == "__main__":
    run_backfill()
