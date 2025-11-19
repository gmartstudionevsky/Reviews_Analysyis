# agent/reviews_backfill_agent.py
from __future__ import annotations

import os
import io
import re
import base64
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import date, datetime, timedelta

import pandas as pd

# --- наши модули (пакетные импорты) ---
from . import reviews_io, reviews_core
from .metrics_core import iso_week_monday, period_ranges_for_week
from .connectors import build_credentials_from_b64, get_drive_client, get_sheets_client

def _require_env(name: str) -> str:
    """
    Берёт переменную окружения name или падает с понятной ошибкой.
    Используем для критичных настроек, без которых агент работать не может.
    """
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is required but not set.")
    return value

# --- Google API ---
from googleapiclient.http import MediaIoBaseDownload

# -----------------------------------------------------------------------------
# Логгер
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("reviews_backfill_agent")

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
SHEETS_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

HISTORY_SHEET_NAME = "reviews_history"  # отдельная вкладка в общем SHEETS_HISTORY_ID

# Поддерживаемые паттерны имён файлов (даты в именах)
_RE_FNAME_DMY = re.compile(r"(?i)\breviews?_?(\d{2})-(\d{2})-(\d{4})\b")
_RE_FNAME_YMD = re.compile(r"(?i)\breviews?_?(\d{4})-(\d{2})-(\d{2})\b")
# Годовой агрегат: reviews_2019-25.xls (с 2019 по 2025)
_RE_FNAME_YEAR_RANGE = re.compile(r"(?i)\breviews?_?(\d{4})-(\d{2})\b")


# -----------------------------------------------------------------------------
# Утилиты
# -----------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _parse_date_from_name(name: str) -> Optional[date]:
    """
    Reviews_DD-MM-YYYY.xls  или  reviews_YYYY-MM-DD.xls
    """
    if not name:
        return None
    m = _RE_FNAME_DMY.search(name)
    if m:
        dd, mm, yyyy = m.groups()
        try:
            return date(int(yyyy), int(mm), int(dd))
        except Exception:
            pass
    m = _RE_FNAME_YMD.search(name)
    if m:
        yyyy, mm, dd = m.groups()
        try:
            return date(int(yyyy), int(mm), int(dd))
        except Exception:
            pass
    return None

def _date_from_env(var_name: str) -> date:
    val = (os.environ.get(var_name) or "").strip()
    if not val:
        raise RuntimeError(f"{var_name} не задан (ожидается YYYY-MM-DD).")
    try:
        return datetime.strptime(val, "%Y-%m-%d").date()
    except Exception:
        raise RuntimeError(f"{var_name} имеет неверный формат: {val} (ожидается YYYY-MM-DD).")

def _drive_list_files_in_folder(drive, folder_id: str) -> List[Dict[str, Any]]:
    q = f"'{folder_id}' in parents and trashed = false"
    fields = "nextPageToken, files(id, name, mimeType, modifiedTime, size)"
    files: List[Dict[str, Any]] = []
    page_token = None
    while True:
        resp = drive.files().list(
            q=q, fields=fields, pageToken=page_token, pageSize=1000, orderBy="modifiedTime desc"
        ).execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files

def _drive_download_file_bytes(drive, file_id: str) -> bytes:
    req = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, req)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    return fh.getvalue()

def _ensure_sheet_exists(sheets, spreadsheet_id: str, title: str) -> None:
    meta = sheets.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    for sh in meta.get("sheets", []):
        if sh.get("properties", {}).get("title") == title:
            return
    body = {"requests": [{"addSheet": {"properties": {"title": title}}}]}
    sheets.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body=body).execute()

def _read_sheet_as_df(sheets, spreadsheet_id: str, title: str) -> pd.DataFrame:
    try:
        resp = sheets.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id, range=f"'{title}'!A:Z"
        ).execute()
        values = resp.get("values", [])
        if not values:
            return pd.DataFrame()
        header = values[0]
        rows = values[1:]
        return pd.DataFrame(rows, columns=header)
    except Exception:
        return pd.DataFrame()

def _append_rows_to_sheet(sheets, spreadsheet_id: str, title: str, rows: List[List[Any]]) -> None:
    if not rows:
        return
    body = {"values": rows}
    sheets.spreadsheets().values().append(
        spreadsheetId=spreadsheet_id,
        range=f"'{title}'!A1",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body=body,
    ).execute()

def _trim_text(s: str, n: int = 280) -> str:
    s = (s or "").strip()
    return (s[: n - 1] + "…") if len(s) > n else s

def _serialize_aspects_for_sheet(value: Any) -> str:
    """
    Превращает список аспектов (['wifi', 'noise', ...]) в строку "wifi;noise".
    Если уже строка — возвращаем как есть.
    """
    if value is None:
        return ""
    # если уже str — не трогаем
    if isinstance(value, str):
        return value.strip()
    # list / tuple / set со строками
    if isinstance(value, (list, tuple, set)):
        parts = []
        for v in value:
            if v is None:
                continue
            parts.append(str(v).strip())
        return ";".join(p for p in parts if p)
    # на всякий случай fallback
    return str(value).strip()

def _serialize_topics_for_sheet(value: Any) -> str:
    """
    Превращает список пар (topic, subtopic) в строку:
      [("breakfast","service_dining_staff"), ("checkin_stay","stay_support")]
      -> "breakfast:service_dining_staff;checkin_stay:stay_support"
    Если уже строка — возвращаем как есть.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple, set)):
        parts = []
        for v in value:
            # ожидаем кортеж/список длины >=2
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                topic_key = str(v[0]).strip()
                sub_key = str(v[1]).strip()
                if topic_key or sub_key:
                    parts.append(f"{topic_key}:{sub_key}")
            else:
                # если вдруг прилетел одиночный код
                parts.append(str(v).strip())
        return ";".join(p for p in parts if p)
    return str(value).strip()

def _upsert_reviews_history_week(
    sheets,
    spreadsheet_id: str,
    df_reviews: pd.DataFrame,
    df_raw_with_has_response: pd.DataFrame,
    week_key: str,
    existing_keys: set[str],
) -> int:
    """
    Записывает в историю строки за указанную неделю week_key.
    Для бэкфилла:
    - не делает дополнительных чтений из Sheets;
    - использует общий набор existing_keys, который передаётся снаружи;
    - пополняет existing_keys новыми ключами по мере добавления строк.
    """
    # review_id -> has_response
    raw_map = (
        df_raw_with_has_response.set_index("review_id")["has_response"].to_dict()
        if not df_raw_with_has_response.empty
        else {}
    )

    to_append: List[List[Any]] = []
    now = _now_iso()

    for _, row in df_reviews.iterrows():
        if row.get("week_key") != week_key:
            continue

        review_id = str(row.get("review_id"))
        review_key = review_id  # наш стабильный идентификатор

        if review_key in existing_keys:
            # уже есть в истории — пропускаем (идемпотентность)
            continue

        aspects = _serialize_aspects_for_sheet(row.get("aspects"))
        topics = _serialize_topics_for_sheet(row.get("topics"))
        has_resp = raw_map.get(review_id, "")
        text_trimmed = _trim_text(str(row.get("raw_text") or ""), 280)

        vals = [
            str(row.get("created_at") or ""),
            week_key,
            str(row.get("source") or ""),
            str(row.get("lang") or ""),
            "" if pd.isna(row.get("rating10")) else float(row.get("rating10")),
            "" if pd.isna(row.get("sentiment_score")) else float(row.get("sentiment_score")),
            str(row.get("sentiment_overall") or ""),
            aspects,
            topics,
            has_resp,
            review_key,
            text_trimmed,
            now,
        ]
        to_append.append(vals)
        existing_keys.add(review_key)  # пополняем набор, чтобы не словить дубликат в этом же запуске

    if not to_append:
        return 0

    _append_rows_to_sheet(sheets, spreadsheet_id, HISTORY_SHEET_NAME, to_append)
    return len(to_append)

def _upsert_reviews_history_bulk(
    sheets,
    spreadsheet_id: str,
    df_reviews: pd.DataFrame,
    df_raw_with_has_response: pd.DataFrame,
    existing_keys: set[str],
) -> int:
    """
    Бэкфилл-режим: собираем все новые строки по всему периоду и
    отправляем одним append-запросом, чтобы не упираться в лимиты
    write_requests per minute.
    """
    # review_id -> has_response
    raw_map = (
        df_raw_with_has_response.set_index("review_id")["has_response"].to_dict()
        if not df_raw_with_has_response.empty
        else {}
    )

    to_append: List[List[Any]] = []
    now = _now_iso()

    for _, row in df_reviews.iterrows():
        review_id = str(row.get("review_id"))
        if not review_id:
            continue
        review_key = review_id  # стабильный идентификатор (см. reviews_core)

        if review_key in existing_keys:
            # уже есть в истории — пропускаем (идемпотентность)
            continue

        aspects = _serialize_aspects_for_sheet(row.get("aspects"))
        topics = _serialize_topics_for_sheet(row.get("topics"))
        has_resp = raw_map.get(review_id, "")
        text_trimmed = _trim_text(str(row.get("raw_text") or ""), 280)

        vals = [
            str(row.get("created_at") or ""),
            str(row.get("week_key") or ""),
            str(row.get("source") or ""),
            str(row.get("lang") or ""),
            "" if pd.isna(row.get("rating10")) else float(row.get("rating10")),
            "" if pd.isna(row.get("sentiment_score")) else float(row.get("sentiment_score")),
            str(row.get("sentiment_overall") or ""),
            aspects,
            topics,
            has_resp,
            review_key,
            text_trimmed,
            now,
        ]
        to_append.append(vals)
        existing_keys.add(review_key)

    if not to_append:
        return 0

    _append_rows_to_sheet(sheets, spreadsheet_id, HISTORY_SHEET_NAME, to_append)
    return len(to_append)

def _read_existing_review_keys_all(
    sheets,
    spreadsheet_id: str,
    sheet_name: str,
) -> set[str]:
    """
    Считает все уже записанные review_key из истории (колонка с ключами).
    Предполагаем, что review_key лежит в 11-й колонке (K), начиная со строки 2.
    """
    try:
        resp = (
            sheets.spreadsheets()
            .values()
            .get(
                spreadsheetId=spreadsheet_id,
                range=f"'{sheet_name}'!K2:K",
            )
            .execute()
        )
    except Exception as e:
        LOG.warning(
            "Не удалось прочитать существующие review_key из листа %s: %s",
            sheet_name,
            e,
        )
        return set()

    values = resp.get("values", [])
    existing: set[str] = set()
    for row in values:
        if not row:
            continue
        key = str(row[0]).strip()
        if key:
            existing.add(key)

    LOG.info("В истории уже есть %d review_key", len(existing))
    return existing



# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main() -> None:
    # --- ENV ---
    drive_folder_id = _require_env("DRIVE_FOLDER_ID")
    sheets_id = _require_env("SHEETS_HISTORY_ID")
    dry_run = (os.environ.get("DRY_RUN") or "false").strip().lower() == "true"

    # --- Google clients (через B64 секрет) ---
    creds = build_credentials_from_b64()
    drive = get_drive_client(creds)
    sheets = get_sheets_client(creds)

    # --- Выбор файла: BACKFILL_FILE -> reviews_YYYY-YY.* -> ошибка ---
    backfill_file = (os.environ.get("BACKFILL_FILE") or "").strip()
    files = _drive_list_files_in_folder(drive, drive_folder_id)

    selected: List[Dict[str, Any]] = []
    if backfill_file:
        for f in files:
            name = f.get("name", "")
            if f["id"] == backfill_file or name.lower() == backfill_file.lower():
                selected = [f]
                LOG.info(f"BACKFILL_FILE: выбран файл '{name}' (id={f['id']}).")
                break
        if not selected:
            raise RuntimeError(f"BACKFILL_FILE='{backfill_file}' не найден в папке.")
    else:
        yr_file = None
        for f in files:
            name = f.get("name", "")
            if _RE_FNAME_YEAR_RANGE.search(name):
                yr_file = f
                break  # берём самый свежий
        if yr_file:
            selected = [yr_file]
            LOG.info(f"Обнаружен агрегированный файл: {yr_file.get('name')}")
        else:
            raise RuntimeError(
                "Не найден агрегированный файл reviews_YYYY-YY.* и не задан BACKFILL_FILE."
            )

    if not selected:
        LOG.warning("Не найдено входных файлов.")
        return

    LOG.info(f"Файлов к обработке: {len(selected)}")

    # --- Чтение/парсинг выбранных файлов ---
    all_inputs: List[reviews_core.ReviewRecordInput] = []
    raw_has_response_pairs: List[Tuple[str, Any]] = []

    for f in selected:
        fid = f["id"]
        fname = f.get("name", "")
        try:
            blob = _drive_download_file_bytes(drive, fid)
            df_raw = reviews_io.read_reviews_xls(blob)

            LOG.info(f"OK: {fname} (raw rows: {len(df_raw)})")
            try:
                LOG.info(f"Колонки: {list(df_raw.columns)}")
                if "date" in df_raw.columns:
                    LOG.info(f"Пример дат: {df_raw['date'].astype(str).head(5).tolist()}")
            except Exception:
                pass

            inputs = reviews_io.df_to_inputs(df_raw)
            LOG.info(f"→ inputs: {len(inputs)}")

            if not inputs:
                LOG.warning(f"Файл {fname}: после нормализации записей нет.")
                continue

            all_inputs.extend(inputs)

            # map review_id -> has_response (по тексту как стабильному ключу в рамках файла)
            tmp: List[Tuple[str, Any]] = []
            for r in inputs:
                mask = df_raw["text"].astype(str).str.strip() == str(r.text).strip()
                has_resp = df_raw.loc[mask, "has_response"]
                tmp.append((r.review_id, (None if has_resp.empty else has_resp.iloc[0])))
            raw_has_response_pairs.extend(tmp)

        except Exception as e:
            LOG.error(f"Ошибка чтения {fname}: {e}")

    if not all_inputs:
        LOG.warning("Нет валидных записей отзывов (inputs пуст). Проверяй парсинг даты/колонки.")
        return

    # --- Анализ через лексикон ---
    from .lexicon_module import Lexicon
    lexicon = Lexicon()

    analyzed = reviews_core.analyze_reviews_bulk(all_inputs, lexicon)
    LOG.info(f"Анализировано записей: {len(analyzed)}")
    if not analyzed:
        LOG.warning("После анализа записей нет (analyzed=0).")
        return

    df_reviews = reviews_core.build_reviews_dataframe(analyzed)
    df_raw_map = (
        pd.DataFrame(raw_has_response_pairs, columns=["review_id", "has_response"])
        .drop_duplicates("review_id")
    )

    # --- Берём ВСЕ записи файла (без фильтров по датам) ---
    df_reviews_period = df_reviews.copy()
    if df_reviews_period.empty:
        LOG.warning("После отбора записей — пусто (ничего писать в историю).")
        return

    LOG.info(
        f"Готовим к записи строк: {len(df_reviews_period)}; "
        f"диапазон дат: {df_reviews_period['created_at'].min()} — {df_reviews_period['created_at'].max()}"
    )

    weeks = sorted(set(df_reviews_period["week_key"].tolist()))
    LOG.info(f"Недели к upsert: {', '.join(weeks) if weeks else '(нет)'}")

    total_appended = 0
    if dry_run:
        LOG.info("DRY_RUN=true — запись в Google Sheets не выполняется.")
    else:
        # 1) гарантируем, что лист существует (один read-запрос)
        _ensure_sheet_exists(sheets, sheets_id, HISTORY_SHEET_NAME)

        # 2) читаем все уже существующие review_key (один read-запрос)
        existing_keys = _read_existing_review_keys_all(
            sheets,
            sheets_id,
            HISTORY_SHEET_NAME,
        )

        # 3) одним заходом дописываем все новые записи за весь период
        total_appended = _upsert_reviews_history_bulk(
            sheets=sheets,
            spreadsheet_id=sheets_id,
            df_reviews=df_reviews_period,
            df_raw_with_has_response=df_raw_map,
            existing_keys=existing_keys,
        )
        LOG.info("В бэкфилл добавлено строк: %d", total_appended)

    LOG.info(f"Готово. Всего добавлено: {total_appended}")
    
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        try:
            with open(summary_path, "a", encoding="utf-8") as fh:
                fh.write("### Reviews backfill\n\n")
                fh.write(f"- Файлов к обработке: {len(selected)}\n")
                fh.write(f"- Входных записей (inputs): {len(all_inputs)}\n")
                fh.write(f"- DRY_RUN: {'true' if dry_run else 'false'}\n")
                fh.write(f"- Новых строк добавлено: {total_appended}\n\n")
        except Exception as e:
            LOG.debug("Не удалось записать summary для backfill: %s", e)

if __name__ == "__main__":
    main()
