# reviews_backfill_agent.py
#
# Полный бэкфилл по отзывам в Google Sheets.
#
# Что делает:
#   1. Берёт файл Reviews_2019-25.xls из папки DRIVE_FOLDER_ID в Google Drive.
#   2. Читает лист "Отзывы".
#   3. Прогоняет каждый отзыв через text_analytics_core.build_semantic_row():
#        - нормализация языка / текста
#        - рейтинг в шкале /10
#        - общая тональность (pos/neg/neu)
#        - аспекты (темы/подтемы/аспекты) + знак
#        - пары аспектов (systemic_risk / expectations_conflict / loyalty_driver)
#        - цитаты
#        - ключи периодов (week_key, month_key и т.д.)
#        - источник
#   4. На базе этого строит агрегаты:
#        a) history:
#            KPI по периодам (week / month / quarter / year)
#            avg10, pos/neu/neg, количество отзывов
#        b) semantic_agg_aspects_period:
#            аспекты × период × source_scope ('all' или конкретный источник)
#            + pos_weight / neg_weight
#        c) semantic_agg_pairs_period:
#            связки аспектов × период × source_scope
#        d) sources_history:
#            неделя × источник:
#            средняя оценка /10, pos/neu/neg
#   5. Полностью ПЕРЕЗАПИСЫВАЕТ в таблице SHEETS_HISTORY_ID вкладки:
#        - reviews_semantic_raw
#        - history
#        - semantic_agg_aspects_period
#        - semantic_agg_pairs_period
#        - sources_history
#
# После этого таблица в Google Sheets становится "истиной", из которой
# будет собираться недельный отчёт.
#
# Важные требования окружения:
#   - DRIVE_FOLDER_ID          (ID папки в Google Drive с историческими файлами)
#   - SHEETS_HISTORY_ID        (ID Google Sheets с историей)
#   - GOOGLE_SERVICE_ACCOUNT_JSON   или GOOGLE_SERVICE_ACCOUNT_JSON_CONTENT
#
# Зависит от:
#   - pandas
#   - google-api-python-client
#   - google-auth
#   - xlrd (чтение .xls)
#
# Зависит также от text_analytics_core.build_semantic_row, который ты уже обновил.


import os
import io
import json
from datetime import datetime
from typing import Any, List, Dict, Tuple

import pandas as pd
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


# =========================
# Импорт ядра семантики отзывов
# =========================
# Берём фабрику одной строки семантики (1 отзыв -> 1 запись)
try:
    from agent.text_analytics_core import build_semantic_row
except Exception:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from text_analytics_core import build_semantic_row


# =========================
# Константы / настройки
# =========================

# Названия вкладок в Google Sheets
SEMANTIC_TAB = "reviews_semantic_raw"            # 1 строка = 1 отзыв (сырой слой)
KPI_TAB      = "history"                         # KPI по периодам (/10 + pos/neu/neg)
ASPECTS_TAB  = "semantic_agg_aspects_period"     # аспекты × период × источник/all (+веса)
PAIRS_TAB    = "semantic_agg_pairs_period"       # связки факторов × период × источник/all
SOURCES_TAB  = "sources_history"                 # разбивка по источникам (недели)

# Периодические уровни, которые мы считаем
# порядок важен и будет использоваться и для сортировки
PERIOD_LEVELS: List[Tuple[str, str]] = [
    ("week",    "week_key"),
    ("month",   "month_key"),
    ("quarter", "quarter_key"),
    ("year",    "year_key"),
]

# Права для сервисного аккаунта
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

    Возвращает (drive, sheets).
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
    Находит в папке folder_id файл с именем filename (строгое совпадение),
    скачивает его и читает лист 'Отзывы'.

    Ожидаемые столбцы в файле:
      'Дата', 'Рейтинг', 'Источник', 'Автор', 'Код языка', 'Текст отзыва', 'Наличие ответа'
    Не все важны для нас, но 'Дата', 'Рейтинг', 'Источник', 'Текст отзыва', 'Код языка'
    должны быть найдены.
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

    # читаем .xls
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
    Добавляет df в конец вкладки tab_name (начиная с A2),
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
# Вспомогательные утилиты для агрегаций
# =========================

def _first_non_empty(row, *names):
    for n in names:
        if n in row and pd.notna(row[n]) and str(row[n]).strip():
            return row[n]
    return None


def _parse_json_list(cell) -> list:
    """
    Безопасно превратить ячейку (строку JSON или уже list) в list[str]
    """
    if isinstance(cell, list):
        return cell
    if pd.isna(cell):
        return []
    try:
        return json.loads(cell)
    except Exception:
        return []


def _parse_json_pairs(cell) -> list:
    """
    Безопасно превратить ячейку (строку JSON или уже list[dict]) в list[dict]
    """
    if isinstance(cell, list):
        return cell
    if pd.isna(cell):
        return []
    try:
        return json.loads(cell)
    except Exception:
        return []


# =========================
# 0. Строим reviews_semantic_raw (строка = один отзыв)
# =========================

def build_reviews_semantic_raw_df(raw_reviews_df: pd.DataFrame) -> pd.DataFrame:
    """
    Преобразует лист "Отзывы" в DataFrame с 1 строкой на 1 отзыв,
    используя text_analytics_core.build_semantic_row().

    Ожидаемые названия исходных столбцов (учитываем рус/англ варианты):
      - 'Дата' / 'Date'
      - 'Рейтинг' / 'Rating' / 'Оценка'
      - 'Источник' / 'Source'
      - 'Код языка' / 'Lang' / 'Language'
      - 'Текст отзыва' / 'Text' / 'Отзыв'
    """

    cols_lower = {c.lower(): c for c in raw_reviews_df.columns}

    def pick(*options):
        for opt in options:
            if opt in cols_lower:
                return cols_lower[opt]
        return None

    col_date   = pick("дата","date")
    col_rate   = pick("рейтинг","rating","оценка","score")
    col_source = pick("источник","source","площадка")
    col_lang   = pick("код языка","lang","язык","language code","language")
    col_text   = pick("текст отзыва","text","отзыв","review text")

    if not all([col_date, col_rate, col_source, col_text]):
        raise RuntimeError("Не найдены обязательные столбцы (дата/рейтинг/источник/текст).")

    out_rows = []
    for _, r in raw_reviews_df.iterrows():
        review_raw = {
            "date":       r[col_date],
            "source":     r[col_source],
            "rating_raw": r[col_rate],
            "text":       r[col_text] if pd.notna(r[col_text]) else "",
            "lang_hint":  r[col_lang] if col_lang else "",
        }
        try:
            row_norm = build_semantic_row(review_raw)
            out_rows.append(row_norm)
        except Exception as e:
            # Не роняем весь бэкфилл из-за одной битой строки
            print(f"[WARN] Ошибка обработки отзыва: {e}")
            continue

    if not out_rows:
        return pd.DataFrame(columns=[
            "review_id","date","week_key","month_key","quarter_key","year_key",
            "source","rating10","sentiment_overall",
            "topics_pos","topics_neg","topics_all","pair_tags","quote_candidates"
        ])

    df = pd.DataFrame(out_rows)

    # сортируем для удобства анализа "сырых" данных:
    # сначала по дате, потом по источнику, потом по review_id
    # (date у нас уже строка ISO через build_semantic_row → YYYY-MM-DD, т.е. сортируем как строку)
    df = df.sort_values(
        by=["date","source","review_id"],
        ascending=[True, True, True],
        ignore_index=True,
    )

    return df


# =========================
# 1. KPI history (period_type / period_key)
# =========================

def compute_history_from_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    KPI по периодам для history:
      period_type ('week'/'month'/'quarter'/'year')
      period_key
      reviews
      avg10
      pos / neu / neg (кол-во отзывов)
    """
    rows = []

    for period_type, col in PERIOD_LEVELS:
        if col not in df_raw.columns:
            continue

        tmp = df_raw.dropna(subset=[col]).copy()
        if tmp.empty:
            continue

        for period_key, gdf in tmp.groupby(col, dropna=True):
            reviews = int(gdf["review_id"].nunique())
            avg10   = float(gdf["rating10"].dropna().mean()) if reviews else None
            pos_cnt = int((gdf["sentiment_overall"] == "pos").sum())
            neu_cnt = int((gdf["sentiment_overall"] == "neu").sum())
            neg_cnt = int((gdf["sentiment_overall"] == "neg").sum())

            rows.append({
                "period_type": period_type,
                "period_key":  period_key,
                "reviews":     reviews,
                "avg10":       round(avg10, 2) if avg10 == avg10 else "",
                "pos":         pos_cnt,
                "neu":         neu_cnt,
                "neg":         neg_cnt,
            })

    if not rows:
        return pd.DataFrame(columns=[
            "period_type","period_key","reviews","avg10","pos","neu","neg"
        ])

    # "умная" сортировка:
    # - сначала 'week', потом 'month', потом 'quarter', потом 'year'
    # - внутри каждого типа период_key по возрастанию
    period_order = {p_type: idx for idx, (p_type, _) in enumerate(PERIOD_LEVELS)}
    out_df = pd.DataFrame(rows)
    out_df["__ptype_rank"] = out_df["period_type"].map(period_order).fillna(999)

    out_df = out_df.sort_values(
        by=["__ptype_rank","period_key"],
        ascending=[True, True],
        ignore_index=True,
    ).drop(columns=["__ptype_rank"])

    return out_df


# =========================
# 2. Источники по неделям (sources_history)
# =========================

def compute_sources_history_weekly_from_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    История по источникам (недельно):
      week_key
      source
      reviews
      avg10
      pos / neu / neg
    """
    tmp = df_raw.dropna(subset=["week_key","source"]).copy()
    if tmp.empty:
        return pd.DataFrame(columns=[
            "week_key","source","reviews","avg10","pos","neu","neg"
        ])

    rows = []
    for (wk, src), gdf in tmp.groupby(["week_key","source"], dropna=True):
        reviews = int(gdf["review_id"].nunique())
        avg10   = float(gdf["rating10"].dropna().mean()) if reviews else None
        pos_cnt = int((gdf["sentiment_overall"] == "pos").sum())
        neu_cnt = int((gdf["sentiment_overall"] == "neu").sum())
        neg_cnt = int((gdf["sentiment_overall"] == "neg").sum())

        rows.append({
            "week_key": wk,
            "source":   src,
            "reviews":  reviews,
            "avg10":    round(avg10, 2) if avg10 == avg10 else "",
            "pos":      pos_cnt,
            "neu":      neu_cnt,
            "neg":      neg_cnt,
        })

    out_df = pd.DataFrame(rows)

    # "умная" сортировка:
    # - сначала по week_key (возрастание, то есть хронология)
    # - внутри одной недели по source (алфавит)
    out_df = out_df.sort_values(
        by=["week_key","source"],
        ascending=[True, True],
        ignore_index=True,
    )

    return out_df


# =========================
# 3. Агрегат по аспектам (semantic_agg_aspects_period)
# =========================

def compute_semantic_agg_aspects_period_from_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Для каждой комбинации:
      (period_type, period_key, source_scope ∈ {'all' | конкретный source}, aspect_key)
    считаем:
      - mentions_total  (сколько отзывов вообще упомянули аспект)
      - pos_mentions / neg_mentions / neu_mentions
      - pos_share / neg_share
      - pos_weight / neg_weight

    pos_weight = доля позитивных отзывов периода, где аспект упомянут позитивно
    neg_weight = доля негативных отзывов периода, где аспект упомянут негативно
    """
    rows = []

    # список источников, пригодится для source_scope
    sources = sorted([s for s in df_raw["source"].dropna().unique().tolist() if s])

    for period_type, col in PERIOD_LEVELS:
        if col not in df_raw.columns:
            continue

        df_p = df_raw.dropna(subset=[col]).copy()
        if df_p.empty:
            continue

        scopes = ["all"] + sources
        for scope in scopes:
            df_s = df_p if scope == "all" else df_p[df_p["source"] == scope]
            if df_s.empty:
                continue

            for period_key, gdf in df_s.groupby(col, dropna=True):
                # деноминаторы для весов
                pos_reviews = set(gdf.loc[gdf["sentiment_overall"] == "pos","review_id"])
                neg_reviews = set(gdf.loc[gdf["sentiment_overall"] == "neg","review_id"])
                pos_total = len(pos_reviews)
                neg_total = len(neg_reviews)

                stats: Dict[str, Dict[str, set]] = {}
                # stats[aspect] = {
                #   "mentions": set([...]),
                #   "pos": set([...]),
                #   "neg": set([...]),
                #   "neu": set([...])
                # }

                for _, r in gdf.iterrows():
                    rid = r["review_id"]

                    a_all = set(_parse_json_list(r.get("topics_all")))
                    a_pos = set(_parse_json_list(r.get("topics_pos")))
                    a_neg = set(_parse_json_list(r.get("topics_neg")))
                    a_neu = a_all - a_pos - a_neg

                    for a in a_all:
                        stats.setdefault(a, {"mentions": set(), "pos": set(), "neg": set(), "neu": set()})
                        stats[a]["mentions"].add(rid)
                    for a in a_pos:
                        stats.setdefault(a, {"mentions": set(), "pos": set(), "neg": set(), "neu": set()})
                        stats[a]["pos"].add(rid)
                    for a in a_neg:
                        stats.setdefault(a, {"mentions": set(), "pos": set(), "neg": set(), "neu": set()})
                        stats[a]["neg"].add(rid)
                    for a in a_neu:
                        stats.setdefault(a, {"mentions": set(), "pos": set(), "neg": set(), "neu": set()})
                        stats[a]["neu"].add(rid)

                for a_key, d in stats.items():
                    mentions_total = len(d["mentions"])
                    pos_mentions   = len(d["pos"])
                    neg_mentions   = len(d["neg"])
                    neu_mentions   = len(d["neu"])

                    if mentions_total == 0:
                        pos_share = 0.0
                        neg_share = 0.0
                    else:
                        pos_share = round(pos_mentions / mentions_total, 4)
                        neg_share = round(neg_mentions / mentions_total, 4)

                    # веса (вклад аспекта в позитив/негатив периода)
                    if pos_total:
                        pos_weight = round(len(d["pos"].intersection(pos_reviews)) / pos_total, 4)
                    else:
                        pos_weight = 0.0
                    if neg_total:
                        neg_weight = round(len(d["neg"].intersection(neg_reviews)) / neg_total, 4)
                    else:
                        neg_weight = 0.0

                    rows.append({
                        "period_type":    period_type,
                        "period_key":     period_key,
                        "source_scope":   scope,
                        "aspect_key":     a_key,
                        "mentions_total": mentions_total,
                        "pos_mentions":   pos_mentions,
                        "neg_mentions":   neg_mentions,
                        "neu_mentions":   neu_mentions,
                        "pos_share":      pos_share,
                        "neg_share":      neg_share,
                        "pos_weight":     pos_weight,
                        "neg_weight":     neg_weight,
                    })

    if not rows:
        return pd.DataFrame(columns=[
            "period_type","period_key","source_scope","aspect_key",
            "mentions_total","pos_mentions","neg_mentions","neu_mentions",
            "pos_share","neg_share","pos_weight","neg_weight",
        ])

    out_df = pd.DataFrame(rows)

    # "умная" сортировка:
    #   - по period_type в порядке week -> month -> quarter -> year
    #   - по period_key возрастание
    #   - по source_scope ('all' идёт первой)
    #   - внутри — по убыванию mentions_total (чтобы крупные темы были сверху)
    period_order = {p_type: idx for idx, (p_type, _) in enumerate(PERIOD_LEVELS)}
    out_df["__ptype_rank"] = out_df["period_type"].map(period_order).fillna(999)
    out_df["__scope_rank"] = out_df["source_scope"].apply(lambda s: 0 if s == "all" else 1)

    out_df = out_df.sort_values(
        by=["__ptype_rank","period_key","__scope_rank","mentions_total"],
        ascending=[True, True, True, False],
        ignore_index=True,
    ).drop(columns=["__ptype_rank","__scope_rank"])

    return out_df


# =========================
# 4. Агрегат по связкам факторов (semantic_agg_pairs_period)
# =========================

def compute_semantic_agg_pairs_period_from_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Для каждой комбинации:
      (period_type, period_key, source_scope ∈ {'all' | конкретный source},
       pair_key, category)
    считаем:
      - distinct_reviews (сколько отзывов содержат эту связку)
      - example_quote (характерная цитата)
    """

    rows = []

    sources = sorted([s for s in df_raw["source"].dropna().unique().tolist() if s])

    for period_type, col in PERIOD_LEVELS:
        if col not in df_raw.columns:
            continue

        df_p = df_raw.dropna(subset=[col]).copy()
        if df_p.empty:
            continue

        scopes = ["all"] + sources
        for scope in scopes:
            df_s = df_p if scope == "all" else df_p[df_p["source"] == scope]
            if df_s.empty:
                continue

            for period_key, gdf in df_s.groupby(col, dropna=True):

                pair_to_reviews: Dict[Tuple[str,str], set] = {}
                pair_to_quote:   Dict[Tuple[str,str], str] = {}

                for _, r in gdf.iterrows():
                    rid    = r["review_id"]
                    pairs  = _parse_json_pairs(r.get("pair_tags"))
                    quotes = _parse_json_list(r.get("quote_candidates"))
                    quote0 = quotes[0] if quotes else ""

                    for p in pairs:
                        a   = (p.get("a") or "").strip()
                        b   = (p.get("b") or "").strip()
                        cat = (p.get("cat") or "").strip()
                        if not a or not b or not cat:
                            continue
                        pair_key = "|".join(sorted([a, b]))
                        key      = (pair_key, cat)
                        pair_to_reviews.setdefault(key, set()).add(rid)
                        # сохраняем первую подходящую цитату как example_quote
                        if key not in pair_to_quote and quote0:
                            pair_to_quote[key] = quote0

                for (pair_key, cat), rset in pair_to_reviews.items():
                    rows.append({
                        "period_type":      period_type,
                        "period_key":       period_key,
                        "source_scope":     scope,
                        "pair_key":         pair_key,
                        "category":         cat,
                        "distinct_reviews": len(rset),
                        "example_quote":    pair_to_quote.get((pair_key, cat), ""),
                    })

    if not rows:
        return pd.DataFrame(columns=[
            "period_type","period_key","source_scope",
            "pair_key","category","distinct_reviews","example_quote",
        ])

    out_df = pd.DataFrame(rows)

    # сортировка:
    #   - period_type в фиксированном порядке
    #   - period_key по возрастанию
    #   - source_scope ('all' первой)
    #   - distinct_reviews по убыванию (топовые связки - наверх)
    period_order = {p_type: idx for idx, (p_type, _) in enumerate(PERIOD_LEVELS)}
    out_df["__ptype_rank"] = out_df["period_type"].map(period_order).fillna(999)
    out_df["__scope_rank"] = out_df["source_scope"].apply(lambda s: 0 if s == "all" else 1)

    out_df = out_df.sort_values(
        by=["__ptype_rank","period_key","__scope_rank","distinct_reviews"],
        ascending=[True, True, True, False],
        ignore_index=True,
    ).drop(columns=["__ptype_rank","__scope_rank"])

    return out_df


# =========================
# Основной сценарий backfill
# =========================

def run_backfill():
    """
    Полный бэкфилл истории отзывов в Google Sheets.

    Шаги:
      1. Скачиваем Reviews_2019-25.xls
      2. Читаем лист "Отзывы"
      3. Готовим reviews_semantic_raw (1 отзыв = 1 строка)
      4. Считаем агрегаты:
         - history
         - semantic_agg_aspects_period
         - semantic_agg_pairs_period
         - sources_history
      5. Полностью перезаписываем соответствующие вкладки
         в SHEETS_HISTORY_ID.
    """
    DRIVE_FOLDER_ID   = os.environ["DRIVE_FOLDER_ID"]
    SHEETS_HISTORY_ID = os.environ["SHEETS_HISTORY_ID"]

    drive, sheets = get_google_clients()

    # 1. Вытягиваем исторический XLS из Google Drive
    raw_reviews_df = drive_download_reviews_file(
        drive=drive,
        folder_id=DRIVE_FOLDER_ID,
        filename="Reviews_2019-25.xls",
    )

    # 2. Готовим "сырые" семантические строки
    df_raw = build_reviews_semantic_raw_df(raw_reviews_df)

    # 3. Считаем агрегаты
    kpi_df     = compute_history_from_raw(df_raw)
    aspects_df = compute_semantic_agg_aspects_period_from_raw(df_raw)
    pairs_df   = compute_semantic_agg_pairs_period_from_raw(df_raw)
    src_df     = compute_sources_history_weekly_from_raw(df_raw)

    # 4. Запись во вкладки Google Sheets
    # 4.1 reviews_semantic_raw
    semantic_header = [
        "review_id","date","week_key","month_key","quarter_key","year_key",
        "source","rating10","sentiment_overall",
        "topics_pos","topics_neg","topics_all","pair_tags","quote_candidates",
    ]
    ensure_tab_exists(sheets, SHEETS_HISTORY_ID, SEMANTIC_TAB, semantic_header)
    clear_tab_data(sheets, SHEETS_HISTORY_ID, SEMANTIC_TAB, start_cell="A2")
    append_df_to_sheet(sheets, SHEETS_HISTORY_ID, SEMANTIC_TAB, df_raw, semantic_header)

    # 4.2 history
    kpi_header = [
        "period_type","period_key","reviews","avg10","pos","neu","neg",
    ]
    ensure_tab_exists(sheets, SHEETS_HISTORY_ID, KPI_TAB, kpi_header)
    clear_tab_data(sheets, SHEETS_HISTORY_ID, KPI_TAB, start_cell="A2")
    append_df_to_sheet(sheets, SHEETS_HISTORY_ID, KPI_TAB, kpi_df, kpi_header)

    # 4.3 semantic_agg_aspects_period
    aspects_header = [
        "period_type","period_key","source_scope","aspect_key",
        "mentions_total","pos_mentions","neg_mentions","neu_mentions",
        "pos_share","neg_share","pos_weight","neg_weight",
    ]
    ensure_tab_exists(sheets, SHEETS_HISTORY_ID, ASPECTS_TAB, aspects_header)
    clear_tab_data(sheets, SHEETS_HISTORY_ID, ASPECTS_TAB, start_cell="A2")
    append_df_to_sheet(sheets, SHEETS_HISTORY_ID, ASPECTS_TAB, aspects_df, aspects_header)

    # 4.4 semantic_agg_pairs_period
    pairs_header = [
        "period_type","period_key","source_scope",
        "pair_key","category","distinct_reviews","example_quote",
    ]
    ensure_tab_exists(sheets, SHEETS_HISTORY_ID, PAIRS_TAB, pairs_header)
    clear_tab_data(sheets, SHEETS_HISTORY_ID, PAIRS_TAB, start_cell="A2")
    append_df_to_sheet(sheets, SHEETS_HISTORY_ID, PAIRS_TAB, pairs_df, pairs_header)

    # 4.5 sources_history
    sources_header = [
        "week_key","source","reviews","avg10","pos","neu","neg",
    ]
    ensure_tab_exists(sheets, SHEETS_HISTORY_ID, SOURCES_TAB, sources_header)
    clear_tab_data(sheets, SHEETS_HISTORY_ID, SOURCES_TAB, start_cell="A2")
    append_df_to_sheet(sheets, SHEETS_HISTORY_ID, SOURCES_TAB, src_df, sources_header)

    print("[INFO] Backfill по отзывам завершён успешно.")


if __name__ == "__main__":
    run_backfill()
