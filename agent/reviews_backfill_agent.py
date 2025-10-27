import os
import io
import math
from datetime import datetime
from typing import Dict, Any, List, Tuple

import pandas as pd

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Импортируем ядро аналитики текста, которое мы уже разработали
from agent.text_analytics_core import (
    TOPIC_SCHEMA,
    build_exploded_reviews_df,
)


# ---------------------------
# Константы / настройки
# ---------------------------

SEMANTIC_TAB = "reviews_semantic_history"
KPI_TAB = "period_kpi_history"
ASPECTS_TAB = "period_aspects_history"
SOURCES_TAB = "source_history"

# Периодические ключи
PERIOD_LEVELS = [
    ("week", "week_key"),
    ("month", "month_key"),
    ("quarter", "quarter_key"),
    ("year", "year_key"),
]


# ---------------------------
# Google auth / helpers
# ---------------------------

def get_google_services(sa_json_path: str):
    """
    Возвращает клиенты Drive и Sheets по сервисному аккаунту.
    Ожидается, что сервисный аккаунт имеет доступ
    к папке DRIVE_FOLDER_ID (для чтения) и к таблице SHEETS_HISTORY_ID (для записи).
    """
    creds = service_account.Credentials.from_service_account_file(
        sa_json_path,
        scopes=[
            "https://www.googleapis.com/auth/drive.readonly",
            "https://www.googleapis.com/auth/spreadsheets",
        ],
    )
    drive_svc = build("drive", "v3", credentials=creds)
    sheets_svc = build("sheets", "v4", credentials=creds)
    return drive_svc, sheets_svc


def drive_download_file_by_name(drive_svc, folder_id: str, target_name: str) -> pd.DataFrame:
    """
    Ищем в указанной папке Drive файл с именем target_name (точное совпадение),
    скачиваем, читаем лист 'Отзывы' в pandas DataFrame.
    Формат ожидается как наш History xls.
    """
    q = (
        f"'{folder_id}' in parents and "
        f"name = '{target_name}' and "
        f"mimeType != 'application/vnd.google-apps.folder'"
    )
    resp = drive_svc.files().list(
        q=q,
        fields="files(id, name, mimeType, modifiedTime, size)",
        orderBy="modifiedTime desc",
        pageSize=5,
    ).execute()
    files = resp.get("files", [])
    if not files:
        raise RuntimeError(f"Не найден файл {target_name} в папке {folder_id}")

    file_id = files[0]["id"]

    # скачиваем содержимое
    request = drive_svc.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    buf.seek(0)

    # читаем лист "Отзывы"
    df = pd.read_excel(buf, sheet_name="Отзывы")
    return df


def df_to_values(df: pd.DataFrame) -> List[List[Any]]:
    """
    Превращаем DataFrame в values[][] для Sheets API:
    первая строка — заголовки.
    NaN превращаем в "".
    """
    out = [list(df.columns)]
    for _, row in df.iterrows():
        line = []
        for val in row.tolist():
            if pd.isna(val):
                line.append("")
            else:
                line.append(val)
        out.append(line)
    return out


def ensure_sheet_exists(sheets_svc, spreadsheet_id: str, tab_name: str):
    """
    Проверяет, что в таблице есть лист с именем tab_name.
    Если нет — добавляет.
    Возвращает sheetId.
    """
    meta = sheets_svc.spreadsheets().get(
        spreadsheetId=spreadsheet_id
    ).execute()

    sheets = meta.get("sheets", [])
    for s in sheets:
        title = s["properties"]["title"]
        if title == tab_name:
            return s["properties"]["sheetId"]

    # если не нашли — создаём
    reqs = [
        {
            "addSheet": {
                "properties": {
                    "title": tab_name,
                    "gridProperties": {
                        "rowCount": 1000,
                        "columnCount": 50,
                    },
                }
            }
        }
    ]
    body = {"requests": reqs}
    resp = sheets_svc.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body=body,
    ).execute()

    replies = resp.get("replies", [])
    if not replies:
        raise RuntimeError(f"Не удалось создать лист {tab_name}")
    sheet_id = replies[0]["addSheet"]["properties"]["sheetId"]
    return sheet_id


def clear_and_write_sheet(sheets_svc,
                          spreadsheet_id: str,
                          tab_name: str,
                          df: pd.DataFrame):
    """
    Полностью очищает вкладку tab_name и заливает df заново.
    """
    ensure_sheet_exists(sheets_svc, spreadsheet_id, tab_name)

    # чистим
    sheets_svc.spreadsheets().values().clear(
        spreadsheetId=spreadsheet_id,
        range=f"{tab_name}!A:ZZ",
        body={},
    ).execute()

    # пишем
    values = df_to_values(df)
    sheets_svc.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=f"{tab_name}!A1",
        valueInputOption="RAW",
        body={"values": values},
    ).execute()


# ---------------------------
# Агрегации
# ---------------------------

def build_per_review_df(exploded_df: pd.DataFrame) -> pd.DataFrame:
    """
    Делаем сводку: одна строка = один отзыв.
    Нам нужно для kpi-шек по периодам и источникам.

    Возвращает:
    review_id, date, week_key, month_key, quarter_key, year_key,
    source, rating_5, sentiment
    """
    if exploded_df.empty:
        cols = [
            "review_id", "date", "week_key", "month_key",
            "quarter_key", "year_key", "source",
            "rating_5", "sentiment"
        ]
        return pd.DataFrame(columns=cols)

    # sentiment у нас один на отзыв (по нашему пайплайну),
    # rating_5 — усредним, если вдруг есть микс значений для одного отзыва.
    per_review = (
        exploded_df.groupby("review_id").agg({
            "date": "first",
            "week_key": "first",
            "month_key": "first",
            "quarter_key": "first",
            "year_key": "first",
            "source": "first",
            "rating_5": "mean",
            "sentiment": "first",
        }).reset_index()
    )

    return per_review


def compute_period_kpi_history(exploded_df: pd.DataFrame) -> pd.DataFrame:
    """
    Строим KPI по периодам на всех уровнях:
    - week_key
    - month_key
    - quarter_key
    - year_key

    Возвращаем поля:
    period_type, period_key,
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

    kpi_df = pd.DataFrame(rows, columns=[
        "period_type",
        "period_key",
        "reviews_count",
        "avg_rating_5",
        "share_pos",
        "share_neg",
    ])

    # сортируем по period_type, а потом по ключу
    # ключи могут быть вида '2025-W42' / '2025-10' / '2025-Q4' / '2025'
    # сортируем просто лексикографически внутри типа
    kpi_df = kpi_df.sort_values(
        by=["period_type", "period_key"],
        ascending=[True, True],
        ignore_index=True,
    )

    return kpi_df


def compute_period_aspects_history(exploded_df: pd.DataFrame) -> pd.DataFrame:
    """
    По каждому периоду и аспекту считаем:
    - сколько уникальных отзывов упомянули аспект,
    - долю позитивных упоминаний,
    - долю негативных упоминаний.

    Возвращаем:
    period_type,
    period_key,
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

        # считаем уникальные упоминания аспектов в разрезе (aspect_code, sentiment, period_key)
        grp = (
            tmp.groupby([col, "aspect_code", "sentiment", "review_id"])
               .size()
               .reset_index(name="cnt")
        )

        # теперь сворачиваем по периоду+аспекту+sentiment → уникальных отзывов
        agg_mentions = (
            grp.groupby([col, "aspect_code", "sentiment"])["review_id"]
               .nunique()
               .reset_index(name="review_mentions")
        )

        # сделаем сводную по sentiment
        pivot = agg_mentions.pivot_table(
            index=[col, "aspect_code"],
            columns="sentiment",
            values="review_mentions",
            fill_value=0,
            aggfunc="sum",
        )

        # гарантируем столбцы
        for need_col in ["pos", "neg", "neu"]:
            if need_col not in pivot.columns:
                pivot[need_col] = 0

        pivot["mentions_total"] = (
            pivot["pos"] + pivot["neg"] + pivot["neu"]
        ).astype(float)

        pivot["pos_share"] = pivot["pos"] / pivot["mentions_total"].replace(0, 1)
        pivot["neg_share"] = pivot["neg"] / pivot["mentions_total"].replace(0, 1)

        pivot = pivot.reset_index()

        for _, row in pivot.iterrows():
            period_key = row[col]
            aspect_code = row["aspect_code"]
            mentions_reviews = row["mentions_total"]
            pos_share = row["pos_share"]
            neg_share = row["neg_share"]

            rows.append({
                "period_type": period_type,
                "period_key": period_key,
                "aspect_code": aspect_code,
                "mentions_reviews": int(mentions_reviews),
                "pos_share": pos_share,
                "neg_share": neg_share,
            })

    out_df = pd.DataFrame(rows, columns=[
        "period_type",
        "period_key",
        "aspect_code",
        "mentions_reviews",
        "pos_share",
        "neg_share",
    ])

    out_df = out_df.sort_values(
        by=["period_type", "period_key", "mentions_reviews"],
        ascending=[True, True, False],
        ignore_index=True,
    )

    return out_df


def compute_source_history_weekly(exploded_df: pd.DataFrame) -> pd.DataFrame:
    """
    Собираем статистику по источникам в недельном разрезе.
    (Недели нам нужны и для трендов, и для динамки каналов/площадок)

    Возвращаем:
    week_key,
    source,
    reviews_count,
    avg_rating_5,
    share_pos,
    share_neg
    """

    if exploded_df.empty:
        return pd.DataFrame(columns=[
            "week_key",
            "source",
            "reviews_count",
            "avg_rating_5",
            "share_pos",
            "share_neg",
        ])

    per_review = build_per_review_df(exploded_df)

    # если у каких-то отзывов нет week_key (нет даты) — пропустим
    tmp = per_review.dropna(subset=["week_key"]).copy()
    if tmp.empty:
        return pd.DataFrame(columns=[
            "week_key",
            "source",
            "reviews_count",
            "avg_rating_5",
            "share_pos",
            "share_neg",
        ])

    rows = []
    for (week_key, source), gdf in tmp.groupby(["week_key", "source"]):
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
            "avg_rating_5": round(avg_rating_5, 4) if avg_rating_5 == avg_rating_5 else "",
            "share_pos": share_pos,
            "share_neg": share_neg,
        })

    out_df = pd.DataFrame(rows, columns=[
        "week_key",
        "source",
        "reviews_count",
        "avg_rating_5",
        "share_pos",
        "share_neg",
    ])

    out_df = out_df.sort_values(
        by=["week_key", "reviews_count"],
        ascending=[True, False],
        ignore_index=True,
    )

    return out_df


# ---------------------------
# Основной пайплайн backfill
# ---------------------------

def run_reviews_backfill():
    """
    Основной сценарий бэкфилла:
    1. Получаем Reviews_2019-25.xls с Google Drive.
    2. Читаем лист "Отзывы".
    3. Прогоняем через build_exploded_reviews_df -> получаем exploded_df.
    4. Дедуп по (review_id, aspect_code) и сортировка по дате.
    5. Считаем периодические агрегаты:
       - KPI по периодам (неделя/месяц/квартал/год)
       - Частоту аспектов по периодам
       - Источники по неделям
    6. Всё заливаем в гугл-таблицу SHEETS_HISTORY_ID
       в разные вкладки.
    """

    drive_folder_id = os.environ["DRIVE_FOLDER_ID"]
    sheets_history_id = os.environ["SHEETS_HISTORY_ID"]
    sa_json_path = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]

    drive_svc, sheets_svc = get_google_services(sa_json_path)

    # 1. скачиваем историческую выгрузку
    raw_reviews_df = drive_download_file_by_name(
        drive_svc=drive_svc,
        folder_id=drive_folder_id,
        target_name="Reviews_2019-25.xls",
    )

    # 2. строим exploded_df через наш текстовый пайплайн
    exploded_df = build_exploded_reviews_df(
        raw_reviews_df=raw_reviews_df,
        topic_schema=TOPIC_SCHEMA,
        add_rating_5scale=True,
    )

    if exploded_df.empty:
        # Если вообще ничего не распарсили — всё равно создаём пустые вкладки
        empty_semantic = pd.DataFrame(columns=[
            "review_id","date","week_key","month_key","quarter_key","year_key",
            "source","rating_5","sentiment","aspect_code","text",
        ])
        clear_and_write_sheet(sheets_svc, sheets_history_id, SEMANTIC_TAB, empty_semantic)
        clear_and_write_sheet(sheets_svc, sheets_history_id, KPI_TAB, pd.DataFrame(columns=[
            "period_type","period_key","reviews_count","avg_rating_5","share_pos","share_neg",
        ]))
        clear_and_write_sheet(sheets_svc, sheets_history_id, ASPECTS_TAB, pd.DataFrame(columns=[
            "period_type","period_key","aspect_code","mentions_reviews","pos_share","neg_share",
        ]))
        clear_and_write_sheet(sheets_svc, sheets_history_id, SOURCES_TAB, pd.DataFrame(columns=[
            "week_key","source","reviews_count","avg_rating_5","share_pos","share_neg",
        ]))
        return

    # 3. дедуп (review_id, aspect_code), чтобы один отзыв × аспект был один раз
    exploded_df = exploded_df.drop_duplicates(
        subset=["review_id", "aspect_code"]
    ).reset_index(drop=True)

    # сортировка для стабильности
    exploded_df = exploded_df.sort_values(
        by=["date","review_id","aspect_code"],
        ascending=[True, True, True],
        ignore_index=True,
    )

    # 4. готовим фрейм для выгрузки в SEMANTIC_TAB
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

    # Приведём дату к строке, иначе Google Sheets может странно съесть таймзону из tz-naive
    semantic_out["date"] = semantic_out["date"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # 5. KPI по периодам
    kpi_df = compute_period_kpi_history(exploded_df)

    # 6. Аспекты по периодам
    aspects_df = compute_period_aspects_history(exploded_df)

    # 7. Источники по неделям
    sources_df = compute_source_history_weekly(exploded_df)

    # 8. Заливка в Sheets
    clear_and_write_sheet(sheets_svc, sheets_history_id, SEMANTIC_TAB, semantic_out)
    clear_and_write_sheet(sheets_svc, sheets_history_id, KPI_TAB, kpi_df)
    clear_and_write_sheet(sheets_svc, sheets_history_id, ASPECTS_TAB, aspects_df)
    clear_and_write_sheet(sheets_svc, sheets_history_id, SOURCES_TAB, sources_df)


if __name__ == "__main__":
    run_reviews_backfill()
