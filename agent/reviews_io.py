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
SOURCE_ALIASES: Dict[str, str] = {
    # Google
    "google": "google",
    "google maps": "google",
    "gmaps": "google",

    # Booking (и близкие)
    "booking": "booking",
    "booking.com": "booking",

    # Яндекс
    "yandex": "yandex",
    "yandex travel": "yandex",
    "яндекс": "yandex",
    "яндекс путешествия": "yandex",

    # 2GIS
    "2gis": "2gis",

    # TripAdvisor
    "tripadvisor": "tripadvisor",

    # Trip.com
    "trip.com": "tripcom",
    "tripcom": "tripcom",

    # Соцсети и прочее
    "instagram": "instagram",
    "vk": "vk",
    "facebook": "facebook",
    "fb": "facebook",
}

CANON_SOURCES: set = set(SOURCE_ALIASES.values()) | {
    "google", "booking", "yandex", "2gis", "tripadvisor", "tripcom",
    "instagram", "vk", "facebook", "other"
}

def normalize_source(raw: str) -> str:
    if not raw:
        return "other"
    s = str(raw).strip().lower()
    s = s.replace(" ", " ")  # NBSP → space
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
    Унификация в шкалу 1..10:
    - если пусто → None
    - если число ≤5 → умножаем на 2 (нативная «звёздная» шкала)
    - если число ≤10 → оставляем
    - иначе → None (странное значение)
    """
    if rating_value is None:
        return None
    try:
        x = float(rating_value)
    except Exception:
        return None
    if x <= 0:
        return None
    if x <= 5:
        return round(x * 2.0, 2)
    if x <= 10:
        return round(x, 2)
    return None


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
