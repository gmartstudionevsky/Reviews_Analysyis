# agent/reviews_io.py
from __future__ import annotations

import io
import re
import hashlib
from datetime import datetime, date
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

# Берём датакласс входа из ядра отзывов
try:
    from agent.reviews_core import ReviewRecordInput
except ModuleNotFoundError:
    # на случай локального запуска без пакета
    from reviews_core import ReviewRecordInput  # type: ignore


# =========================================
# Канонизация источников и языков
# =========================================

# Алиасы → каноническое имя источника
# Алиасы → внутренний канон-код источника (латиницей), display-имя зададим ниже отдельно.
SOURCE_ALIASES: Dict[str, str] = {
    # TL: Marketing
    "tl: marketing": "tlmarketing",
    "tl marketing": "tlmarketing",
    "tl": "tlmarketing",

    # Trip.com
    "trip.com": "tripcom",
    "tripcom": "tripcom",

    # Яндекс (объединяем 'Яндекс' и 'Яндекс Путешествия' в один канон)
    "yandex": "yandex",
    "yandex travel": "yandex",
    "яндекс": "yandex",
    "яндекс путешествия": "yandex",

    # Ostrovok.ru (Emerging Travel Group)
    "ostrovok": "ostrovok",
    "ostrovok.ru": "ostrovok",

    # 2GIS
    "2gis": "2gis",

    # Sutochno → «Суточно.ру»
    "sutochno": "sutochno",
    "sutochno.ru": "sutochno",
    "суточно": "sutochno",
    "суточно.ру": "sutochno",

    # Google
    "google": "google",
    "google maps": "google",
    "gmaps": "google",

    # TripAdvisor
    "tripadvisor": "tripadvisor",

    # OneTwoTrip
    "onetwotrip": "onetwotrip",
    "one two trip": "onetwotrip",

    # 101Hotels.com
    "101hotels": "101hotels",
    "101hotels.com": "101hotels",

    # Tvil.ru
    "tvil": "tvil",
    "tvil.ru": "tvil",

    # TopHotels
    "tophotels": "tophotels",

    # Booking.com
    "booking": "booking",
    "booking.com": "booking",
}

# Полный список допустимых канон-кодов + 'other'
CANON_SOURCES: set = {
    "tlmarketing",
    "tripcom",
    "yandex",
    "ostrovok",
    "2gis",
    "sutochno",
    "google",
    "tripadvisor",
    "onetwotrip",
    "101hotels",
    "tvil",
    "tophotels",
    "booking",
    "other",
}

# Отображаемое имя источника для отчётов/таблиц
SOURCE_DISPLAY_NAME: Dict[str, str] = {
    "tlmarketing": "TL: Marketing",
    "tripcom": "Trip.com",
    "yandex": "Яндекс",
    "ostrovok": "Ostrovok.ru",
    "2gis": "2GIS",
    "sutochno": "Суточно.ру",
    "google": "Google",
    "tripadvisor": "TripAdvisor",
    "onetwotrip": "OneTwoTrip",
    "101hotels": "101Hotels.com",
    "tvil": "Tvil.ru",
    "tophotels": "TopHotels",
    "booking": "Booking.com",
    "other": "Other",
}

def source_display_name(source_code: str) -> str:
    return SOURCE_DISPLAY_NAME.get(source_code, source_code)

# Источники, где нативная внешняя шкала — 1..5 (для блока C «Источник×Период»)
FIVE_STAR_SOURCES: set = {
    "tlmarketing",
    "tripcom",
    "yandex",
    "2gis",
    "google",
    "tripadvisor",
}

def source_native_scale(source_code: str) -> str:
    """
    'five' → нативная 1..5; 'ten' → нативная 1..10 (или нефиксированная, показываем 10-балльную норму).
    Используем позже при формировании таблиц по источникам.
    """
    return "five" if source_code in FIVE_STAR_SOURCES else "ten"

def to_native_for_sources_block(rating10: Optional[float], source_code: str) -> Optional[float]:
    """
    Возвращает значение для визуального вывода в таблице C1 «Источник×Период».
    - Для источников с нативной 5-балльной шкалой (см. FIVE_STAR_SOURCES)
      конвертируем 10→5 делением на 2.
    - Для остальных источников оставляем 10-балльное значение как есть.
    В остальных расчётах (агрегации, метрики) продолжаем использовать 10-балльную шкалу.
    """
    if rating10 is None:
        return None
    try:
        r = float(rating10)
    except Exception:
        return None
    if not (1.0 <= r <= 10.0):
        return None
    return round(r / 2.0, 2) if source_code in FIVE_STAR_SOURCES else round(r, 2)

def normalize_source(raw: str) -> str:
    if not raw:
        return "other"
    s = str(raw).strip().lower()
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s)
    return SOURCE_ALIASES.get(s, s if s in CANON_SOURCES else "other")


def normalize_lang_code(raw: str) -> str:
    """
    'ru-RU' → 'ru', 'EN' → 'en'; пустое → 'en' (дефолт).
    """
    if not raw:
        return "en"
    s = str(raw).strip().lower()
    if "-" in s:
        s = s.split("-")[0]
    return s or "en"


# =========================================
# Дата / рейтинг / ключ
# =========================================

_DATE_REPLACERS = [
    (r"\.", "-"),
    (r"/", "-"),
    (r"\s+", " "),
]

def _parse_date_any(x) -> date:
    """
    Принимаем date/datetime/str. Поддерживаем форматы:
    - YYYY-MM-DD
    - DD-MM-YYYY
    - DD.MM.YYYY / DD/MM/YYYY
    Возвращаем date (UTC-naive).
    """
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    s = str(x).strip()
    if not s:
        raise ValueError("empty date")
    for pat, repl in _DATE_REPLACERS:
        s = re.sub(pat, repl, s)
    # пробуем ISO
    for fmt in ("%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    # последний шанс — pandas
    return pd.to_datetime(s, errors="coerce").date()


def normalize_rating10(source: str, rating_value) -> Optional[float]:
    """
    В сырых данных все рейтинги уже в 10-балльной шкале.
    Задача: принять число в диапазоне 1..10 и вернуть float.
    Любые значения вне диапазона или нечисловые → None.
    """
    if rating_value is None:
        return None
    try:
        x = float(rating_value)
    except Exception:
        return None
    if not (1.0 <= x <= 10.0):
        return None
    return round(x, 2)

def _text_norm_for_key(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t[:500]  # ограничим длину, чтобы ключ был стабильным


def make_review_key(source: str, author: str, dt: date, text: str) -> str:
    """
    Стабильный ключ, если нет явного ID площадки.
    Хешируем (source, author, date_iso, text_norm) → sha1, берём укороченный префикс.
    """
    base = f"{normalize_source(source)}|{(author or '').strip()}|{dt.isoformat()}|{_text_norm_for_key(text)}"
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
    return f"{normalize_source(source)}:{h}"


# =========================================
# Чтение XLS и приведение колонок
# =========================================

# Ожидаемая схема листа:
# Дата | Рейтинг | Источник | Автор | Код языка | Текст отзыва | Наличие ответа
COL_SYNONYMS: Dict[str, str] = {
    # канонические имена
    "date": "date",
    "rating": "rating",
    "source": "source",
    "author": "author",
    "lang": "lang",
    "text": "text",
    "has_response": "has_response",

    # русские заголовки
    "дата": "date",
    "рейтинг": "rating",
    "источник": "source",
    "автор": "author",
    "код языка": "lang",
    "текст отзыва": "text",
    "наличие ответа": "has_response",
}

CANON_ORDER = ["date", "rating", "source", "author", "lang", "text", "has_response"]

def _canon_col(name: str) -> Optional[str]:
    if not name:
        return None
    n = str(name).strip().lower()
    n = n.replace(" ", " ")  # NBSP
    n = re.sub(r"\s+", " ", n)
    return COL_SYNONYMS.get(n)


def read_reviews_xls(blob: bytes, sheet: Optional[str] = None) -> pd.DataFrame:
    """
    Прочитать XLS/XLSX байты → DataFrame с каноническими колонками.
    Если лист не указан — берём первый.
    Лишние колонки сохраняем (не мешают), но основные — нормализуем.
    """
    xls = pd.ExcelFile(io.BytesIO(blob))
    sheet_name = sheet or xls.sheet_names[0]
    raw = pd.read_excel(xls, sheet_name=sheet_name)

    # построим маппинг реальных колонок → канон-имён
    rename_map: Dict[str, str] = {}
    for c in raw.columns:
        canon = _canon_col(c)
        if canon:
            rename_map[c] = canon

    df = raw.rename(columns=rename_map).copy()

    # убедимся, что ключевые колонки присутствуют
    for need in ["date", "source", "text"]:
        if need not in df.columns:
            raise KeyError(f"В входной таблице отсутствует обязательная колонка: {need}")

    # приведём порядок (если каких-то нет — добавим пустые)
    for col in CANON_ORDER:
        if col not in df.columns:
            df[col] = None
    df = df[CANON_ORDER + [c for c in df.columns if c not in CANON_ORDER]]

    return df


# =========================================
# Преобразование таблицы → ReviewRecordInput[]
# =========================================

def df_to_inputs(df: pd.DataFrame) -> List[ReviewRecordInput]:
    """
    На вход — DataFrame с каноническими колонками.
    На выход — список ReviewRecordInput для ядра.
    """
    out: List[ReviewRecordInput] = []

    for idx, row in df.iterrows():
        try:
            dt = _parse_date_any(row.get("date"))
        except Exception:
            # пропускаем строки без корректной даты
            continue

        src = normalize_source(row.get("source"))
        lang = normalize_lang_code(row.get("lang") or "")
        rating10 = normalize_rating10(src, row.get("rating"))

        author = row.get("author") or ""
        text = str(row.get("text") or "").strip()
        if not text:
            # пустые записи нам не нужны
            continue

        review_id = make_review_key(src, author, dt, text)

        out.append(ReviewRecordInput(
            review_id=review_id,
            source=src,
            created_at=dt,       # reviews_core сам сделает week_key (см. analyze_single_review)
            rating10=rating10,
            lang=lang,
            text=text,
        ))
    return out
