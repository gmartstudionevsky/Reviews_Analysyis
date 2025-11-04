# agent/reviews_io.py
from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

# Пакетный импорт из agent
from .reviews_core import ReviewRecordInput



# --------------------------------------------------------------------------------------
# Канонизация источников
# --------------------------------------------------------------------------------------

# ВНИМАНИЕ: Яндекс и Яндекс Путешествия объединяем в один код: "yandex".
_SOURCE_CANON_MAP: Dict[str, str] = {
    # TL Marketing
    "tl: marketing": "tl_marketing",
    "tl marketing": "tl_marketing",
    "tl-marketing": "tl_marketing",
    "tl - marketing": "tl_marketing",
    # Trip.com
    "trip.com": "trip_com",
    "tripcom": "trip_com",
    # Yandex (объединяем)
    "yandex": "yandex",
    "яндекс": "yandex",
    "яндекс путешествия": "yandex",
    "yandex travel": "yandex",
    "yandex.travel": "yandex",
    # Ostrovok.ru
    "ostrovok.ru": "ostrovok",
    "ostrovok": "ostrovok",
    "emerging travel group": "ostrovok",
    "ostrovok (emerging travel group)": "ostrovok",
    # 2GIS
    "2gis": "2gis",
    "2 гис": "2gis",
    # Sutochno / Суточно.ру
    "sutochno": "sutochno",
    "суточно": "sutochno",
    "суточно.ру": "sutochno",
    # Google
    "google": "google",
    "google maps": "google",
    "google reviews": "google",
    # TripAdvisor
    "tripadvisor": "tripadvisor",
    "trip advisor": "tripadvisor",
    # OneTwoTrip
    "onetwotrip": "onetwotrip",
    "one two trip": "onetwotrip",
    "one-two-trip": "onetwotrip",
    # 101hotels
    "101hotels.com": "101hotels",
    "101hotels": "101hotels",
    # Tvil.ru
    "tvil.ru": "tvil",
    "tvil": "tvil",
    # TopHotels
    "tophotels": "tophotels",
    "top hotels": "tophotels",
    # Booking.com
    "booking": "booking",
    "booking.com": "booking",
}

# Отображаемые имена (в письме/отчёте)
_SOURCE_DISPLAY: Dict[str, str] = {
    "tl_marketing": "TL: Marketing",
    "trip_com": "Trip.com",
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
}

# Источники, для которых в блоке C1 «Источник × Период» нужно визуально показывать нативную /5
FIVE_STAR_SOURCES = {
    "tl_marketing",
    "trip_com",
    "yandex",
    "2gis",
    "google",
    "tripadvisor",
}


def _clean_nbsp(s: str) -> str:
    return (s or "").replace("\u00A0", " ").strip()


def normalize_source(s: Any) -> str:
    """
    Приводим произвольную строку источника к каноническому коду.
    """
    if s is None:
        return ""
    raw = _clean_nbsp(str(s)).lower()
    raw = re.sub(r"\s+", " ", raw)
    # специальные варианты (уберём лишние символы, точки)
    raw_key = raw.replace(".", "").replace("-", "-").replace("—", "-")
    raw_key = re.sub(r"\s*-\s*", "-", raw_key)
    if raw_key in _SOURCE_CANON_MAP:
        return _SOURCE_CANON_MAP[raw_key]
    # второй шанс: сырой вариант
    if raw in _SOURCE_CANON_MAP:
        return _SOURCE_CANON_MAP[raw]
    # fallback: heuristic
    if "yandex" in raw or "яндекс" in raw:
        return "yandex"
    if "booking" in raw:
        return "booking"
    if "tripadvisor" in raw or "trip advisor" in raw:
        return "tripadvisor"
    if "google" in raw:
        return "google"
    if "2gis" in raw:
        return "2gis"
    if "ostrovok" in raw:
        return "ostrovok"
    if "onetwotrip" in raw:
        return "onetwotrip"
    if "101hotels" in raw:
        return "101hotels"
    if "tvil" in raw:
        return "tvil"
    if "tophotels" in raw:
        return "tophotels"
    if "tl" in raw and "marketing" in raw:
        return "tl_marketing"
    if "sutochno" in raw or "суточно" in raw:
        return "sutochno"
    return raw  # как есть — пусть попадёт в отчёт, но без красивого имени


def source_display_name(code: str) -> str:
    return _SOURCE_DISPLAY.get(code, code or "")


def source_is_five_star(code: str) -> bool:
    return code in FIVE_STAR_SOURCES


def to_native_for_sources_block(rating10: Optional[float], source_code: str) -> Optional[float]:
    """
    Для блока C1 в письме: нативная шкала 1–5 для ряда источников.
    ВАЖНО: во всех расчётах по пайплайну сохраняем /10, это только отображение.
    """
    if rating10 is None:
        return None
    if source_is_five_star(source_code):
        return round(float(rating10) / 2.0, 2)
    return None


# --------------------------------------------------------------------------------------
# Чтение XLS и нормализация колонок
# --------------------------------------------------------------------------------------

# Синонимы колонок → стандартные имена
_COLMAP = {
    # дата
    "дата": "date",
    "date": "date",
    # рейтинг (всегда приходит /10 по вашему соглашению)
    "рейтинг": "rating10",
    "rating": "rating10",
    "оценка": "rating10",
    # источник
    "источник": "source",
    "source": "source",
    # автор
    "автор": "author",
    "author": "author",
    "пользователь": "author",
    # код языка
    "код языка": "lang",
    "язык": "lang",
    "lang": "lang",
    "language": "lang",
    # текст
    "текст отзыва": "text",
    "текст": "text",
    "отзыв": "text",
    "review": "text",
    "comment": "text",
    # наличие ответа
    "наличие ответа": "has_response",
    "ответ": "has_response",
    "has_response": "has_response",
}


def _read_excel_bytes(xls_bytes: bytes) -> pd.DataFrame:
    """
    Пытаемся прочитать .xls/.xlsx с несколькими fallback'ами движка.
    """
    if not xls_bytes:
        return pd.DataFrame()
    bio = io.BytesIO(xls_bytes)
    try:
        return pd.read_excel(bio)  # пусть pandas сам подберёт движок
    except Exception:
        pass
    # Попытка xlrd (для .xls)
    bio.seek(0)
    try:
        return pd.read_excel(bio, engine="xlrd")
    except Exception:
        pass
    # Попытка openpyxl (для .xlsx)
    bio.seek(0)
    try:
        return pd.read_excel(bio, engine="openpyxl")
    except Exception:
        pass
    return pd.DataFrame()


_LANG_MAP = {
    "ru": "ru", "ru-ru": "ru", "rus": "ru", "russian": "ru", "ru_RU": "ru",
    "en": "en", "en-us": "en", "en_gb": "en", "eng": "en", "english": "en",
    "tr": "tr", "turkish": "tr",
    "ar": "ar", "arabic": "ar",
    "zh": "zh", "zh-cn": "zh", "zh-hans": "zh", "chinese": "zh",
}


def _norm_lang(x: Any) -> str:
    if x is None:
        return "other"
    s = str(x).strip().lower().replace("_", "-")
    return _LANG_MAP.get(s, "other")


def read_reviews_xls(xls_bytes: bytes) -> pd.DataFrame:
    """
    Возвращает нормализованный DataFrame со стандартными колонками:
    date (datetime.date), rating10 (float), source (canon-code),
    author (str), lang (ISO-639-1), text (str), has_response (str/bool/None)
    """
    df = _read_excel_bytes(xls_bytes)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "rating10", "source", "author", "lang", "text", "has_response"])

    # нормализуем заголовки
    norm_cols: Dict[str, str] = {}
    for c in df.columns:
        key = _clean_nbsp(str(c)).lower()
        norm_cols[c] = _COLMAP.get(key, key)
    df = df.rename(columns=norm_cols)

    # минимальная валидация
    for need in ("date", "source", "text"):
        if need not in df.columns:
            df[need] = None

    # типы
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    if "rating10" in df.columns:
        df["rating10"] = pd.to_numeric(df["rating10"], errors="coerce")
    else:
        df["rating10"] = pd.Series(dtype="float64")

    df["source"] = df["source"].map(normalize_source).fillna("")
    if "author" not in df.columns:
        df["author"] = ""
    df["author"] = df["author"].astype(str)

    if "lang" not in df.columns:
        df["lang"] = ""
    df["lang"] = df["lang"].map(_norm_lang)

    if "text" not in df.columns:
        df["text"] = ""
    df["text"] = df["text"].astype(str).map(_clean_nbsp)

    # has_response — оставляем «как есть», мягко нормализуем для истории
    if "has_response" not in df.columns:
        df["has_response"] = ""
    else:
        hr = df["has_response"].astype(str).str.strip().str.lower()
        df["has_response"] = hr.replace(
            {
                "да": "yes", "есть": "yes", "y": "yes", "yes": "yes", "true": "yes", "1": "yes",
                "нет": "no", "n": "no", "no": "no", "false": "no", "0": "no",
            }
        )

    # фильтр пустых текстов
    df = df[df["text"].astype(str).str.strip().ne("")].copy()

    return df.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Ключ/ID отзыва и сборка входов для ядра
# --------------------------------------------------------------------------------------

def _normalize_for_key(s: str) -> str:
    s = _clean_nbsp(s).lower()
    s = re.sub(r"https?://\S+", "", s)          # убираем URL
    s = re.sub(r"\S+@\S+", "", s)               # убираем e-mail
    s = re.sub(r"\s+", " ", s).strip()
    return s


def make_review_id(source_code: str, author: str, dt: Optional[date], text: str) -> str:
    """
    Стабильный ключ для отзыва. Если в исходнике нет review_id — генерируем.
    """
    base = f"{source_code}|{_clean_nbsp(author)}|{dt.isoformat() if dt else ''}|{_normalize_for_key(text)}"
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
    return f"{source_code}:{digest}"


def df_to_inputs(df: pd.DataFrame) -> List[ReviewRecordInput]:
    """
    Преобразует нормализованный DataFrame (read_reviews_xls) в список ReviewRecordInput для ядра.
    """
    out: List[ReviewRecordInput] = []
    if df is None or df.empty:
        return out
    for _, row in df.iterrows():
        dt = row.get("date", None)
        if pd.isna(dt) or not dt:
            # пропускаем, без даты неделя/период не посчитается
            continue
        dt = row["date"] if isinstance(row["date"], date) else pd.to_datetime(row["date"]).date()

        source_code = str(row.get("source") or "")
        author = str(row.get("author") or "")
        text = str(row.get("text") or "")
        lang = str(row.get("lang") or "other")
        rating10 = row.get("rating10", None)
        rating10 = None if pd.isna(rating10) else float(rating10)

        review_id = make_review_id(source_code, author, dt, text)

        out.append(
            ReviewRecordInput(
                review_id=review_id,
                source=source_code,
                created_at=dt,
                rating10=rating10,
                lang=lang,
                text=text,
            )
        )
    return out
