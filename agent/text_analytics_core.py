# agent/text_analytics_core.py
# =============================================================================
# ВАЖНО:
# 1. Прямо сюда, СВЕРХУ этого файла, вставь словарь из dictionary_text_analytics_core.py.
#    Он должен определять:
#       POSITIVE_WORDS_STRONG: Dict[str, List[str]]
#       POSITIVE_WORDS_SOFT:   Dict[str, List[str]]
#       NEGATIVE_WORDS_STRONG: Dict[str, List[str]]
#       NEGATIVE_WORDS_SOFT:   Dict[str, List[str]]
#       NEUTRAL_WORDS:         Dict[str, List[str]]
#       TOPIC_SCHEMA:          Dict[str, Any]
#
#    TOPIC_SCHEMA ожидается такого вида:
#    {
#        "hospitality": {
#            "subtopics": {
#                "staff_friendliness": {
#                    "patterns": { "ru": [...regex...], "en": [...regex...] },
#                    "aspects": ["Персонал / Забота"]
#                },
#                ...
#            }
#        },
#        ...
#    }
#
# 2. Всё остальное в этом файле — ядро логики. Его не меняем.
# =============================================================================

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from datetime import datetime
import pandas as pd
import re
import unicodedata
import hashlib
import json


# =============================================================================
# ТЕКСТОВЫЕ ХЕЛПЕРЫ / НОРМАЛИЗАЦИЯ
# =============================================================================

_RE_MULTI_WS  = re.compile(r"\s+")
_RE_URL       = re.compile(r"https?://\S+|www\.\S+")
_RE_EMAIL     = re.compile(r"\b[\w.\-+%]+@[\w.\-]+\.[A-Za-z]{2,}\b")
_RE_PHONE     = re.compile(r"\+?\d[\d\-\s()]{6,}\d")
_RE_HTML      = re.compile(r"<[^>]+>")
_RE_WIFI      = re.compile(r"\b(wi[\s\-]?fi|вай[\s\-]?[фо]й)\b", re.IGNORECASE)

def clean_text(raw: str) -> str:
    """
    Унифицируем отзыв для детерминированного анализа:
    - вырезаем теги, ссылки, телефоны, email
    - приводим в нижний регистр
    - сводим 'wi fi'/'вай фай' -> 'wi-fi'
    - схлопываем множественные пробелы
    """
    if not raw:
        return ""
    t = unicodedata.normalize("NFKC", str(raw))
    t = _RE_HTML.sub(" ", t)
    t = _RE_URL.sub(" ", t)
    t = _RE_EMAIL.sub(" ", t)
    t = _RE_PHONE.sub(" ", t)
    t = t.lower()
    t = _RE_WIFI.sub("wi-fi", t)
    t = _RE_MULTI_WS.sub(" ", t).strip()
    return t


# =============================================================================
# ДАТА / ПЕРИОДЫ / РЕЙТИНГ
# =============================================================================

def parse_date_any(x: Any) -> datetime:
    """
    Мы должны надёжно получить datetime.
    Поддерживаем:
    - уже готовый datetime
    - dd.mm.yyyy
    - yyyy-mm-dd
    - Excel-serial (число дней от 1899-12-30)
    Если дата не парсится => возвращаем текущую (лучше не падать).
    """
    if isinstance(x, datetime):
        return x.replace(tzinfo=None)

    # Excel serial number (float/int)
    if isinstance(x, (int, float)) and not pd.isna(x):
        try:
            base = pd.Timestamp("1899-12-30")
            ts = base + pd.to_timedelta(float(x), unit="D")
            return ts.to_pydatetime().replace(tzinfo=None)
        except Exception:
            pass

    # String / other
    try:
        return (
            pd.to_datetime(x, errors="raise", dayfirst=True)
            .to_pydatetime()
            .replace(tzinfo=None)
        )
    except Exception:
        # fallback, чтобы не падать
        return datetime.utcnow().replace(tzinfo=None)


def derive_period_keys(dt: datetime) -> Dict[str, str]:
    """
    Строим ключи периодов для последующей агрегации.
    week_key:    YYYY-Www (ISO-неделя)
    month_key:   YYYY-MM
    quarter_key: YYYY-QX
    year_key:    YYYY
    """
    iso_y, iso_w, _ = dt.isocalendar()
    q = (dt.month - 1)//3 + 1
    return {
        "week_key":    f"{iso_y:04d}-W{iso_w:02d}",
        "month_key":   f"{dt.year:04d}-{dt.month:02d}",
        "quarter_key": f"{dt.year:04d}-Q{q}",
        "year_key":    f"{dt.year:04d}",
    }


def normalize_rating_to_10(rating_raw: Any) -> float | None:
    """
    Приводим рейтинг к /10.
    Если рейтинг 1..5 -> умножаем на 2.
    Если 0..10 -> оставляем.
    Иначе -> None.
    """
    if rating_raw is None or (isinstance(rating_raw, float) and pd.isna(rating_raw)):
        return None
    try:
        val = float(str(rating_raw).replace(",", "."))
    except Exception:
        return None

    if 1.0 <= val <= 5.0:
        return round(val * 2.0, 2)
    if 0.0 <= val <= 10.0:
        return round(val, 2)

    return None


# =============================================================================
# ЯЗЫК
# =============================================================================

def normalize_lang(lang_hint: str, text_norm: str) -> str:
    """
    Минималистичная эвристика.
    Если lang_hint есть — используем его (в нижнем регистре).
    Если нет, смотрим алфавит, чтобы не путать 'ru'/'en'/'tr'/'zh'/'ar'.
    Остальное — 'en'.
    """
    if lang_hint:
        return lang_hint.strip().lower()

    if re.search(r"[а-яё]", text_norm, flags=re.IGNORECASE):
        return "ru"
    if re.search(r"[\u4e00-\u9fff]", text_norm):
        return "zh"
    if re.search(r"[\u0600-\u06FF]", text_norm):
        return "ar"
    if re.search(r"[ğüşöıçİĞÜŞÖÇ]", text_norm):
        return "tr"

    return "en"


def translate_to_ru(text_norm: str, lang: str) -> str:
    """
    Сейчас НЕ делаем реального перевода.
    Возвращаем text_norm как есть.
    Это ок, потому что:
    - если отзыв на русском, он уже на русском;
    - если нет, мы всё равно матчим по EN/TR/... паттернам из твоего словаря.
    """
    return text_norm


# =============================================================================
# ОБЩАЯ ТОНАЛЬНОСТЬ ОТЗЫВА
# =============================================================================

def _any_match(patterns: List[str], chunk: str) -> bool:
    for pat in patterns:
        if re.search(pat, chunk, flags=re.IGNORECASE):
            return True
    return False


def detect_sentiment_label(full_text: str, lang: str) -> str:
    """
    Грубая итоговая тональность всего отзыва (pos/neg/neu).
    Приоритет:
    1) сильный негатив
    2) сильный позитив
    3) мягкий негатив
    4) мягкий позитив
    5) нейтральные слова
    иначе "neu"
    """
    lang_key = lang if lang in NEGATIVE_WORDS_STRONG else "en"

    if _any_match(NEGATIVE_WORDS_STRONG.get(lang_key, []), full_text):
        return "neg"
    if _any_match(POSITIVE_WORDS_STRONG.get(lang_key, []), full_text):
        return "pos"
    if _any_match(NEGATIVE_WORDS_SOFT.get(lang_key, []), full_text):
        return "neg"
    if _any_match(POSITIVE_WORDS_SOFT.get(lang_key, []), full_text):
        return "pos"
    if _any_match(NEUTRAL_WORDS.get(lang_key, []), full_text):
        return "neu"

    return "neu"


# =============================================================================
# ТОНАЛЬНОСТЬ КОНКРЕТНОГО АСПЕКТА (ПОДТЕМЫ)
# =============================================================================

def detect_subtopic_sentiment(full_text: str, lang: str, subtopic_patterns: Dict[str, List[str]]) -> str:
    """
    Оцениваем настроение по конкретной подтеме.
    Алгоритм:
    - находим все совпадения regex подтемы,
    - для каждого совпадения берём окно ±40 символов
    - по каждому окну проверяем приоритет (neg strong -> pos strong -> neg soft -> pos soft -> neutral)
    - если ничего не нашлось в окнах, считаем "neu"

    Возвращает "pos" / "neg" / "neu".
    """
    lang_key = lang if lang in NEGATIVE_WORDS_STRONG else "en"

    # Собираем окна вокруг матчей подтемы
    windows: List[str] = []
    pats = (
        subtopic_patterns.get(lang, []) or
        subtopic_patterns.get("ru", []) or
        subtopic_patterns.get("en", []) or
        []
    )
    for pat in pats:
        for m in re.finditer(pat, full_text, flags=re.IGNORECASE):
            s, e = m.span()
            left = max(0, s - 40)
            right = min(len(full_text), e + 40)
            windows.append(full_text[left:right])

    if not windows:
        # если нет конкретных матчей (но нас вообще вызвали) — просто оцениваем весь текст
        windows = [full_text]

    def classify_window(chunk: str) -> str | None:
        # приоритет ШАГА
        if _any_match(NEGATIVE_WORDS_STRONG.get(lang_key, []), chunk):
            return "neg"
        if _any_match(POSITIVE_WORDS_STRONG.get(lang_key, []), chunk):
            return "pos"
        if _any_match(NEGATIVE_WORDS_SOFT.get(lang_key, []), chunk):
            return "neg"
        if _any_match(POSITIVE_WORDS_SOFT.get(lang_key, []), chunk):
            return "pos"
        if _any_match(NEUTRAL_WORDS.get(lang_key, []), chunk):
            return "neu"
        return None

    # Пробуем классифицировать каждое окно по приоритету.
    # Берём первое окно, где получили явный pos/neg/neu.
    for w in windows:
        label = classify_window(w)
        if label is not None:
            return label

    return "neu"


# =============================================================================
# ИЗВЛЕЧЕНИЕ АСПЕКТОВ (topics_all / topics_pos / topics_neg) + ПАРЫ
# =============================================================================

def extract_aspects_and_pairs(full_text: str, lang: str) -> Tuple[List[str], List[str], List[str], List[Dict[str, str]]]:
    """
    1) Проходим по TOPIC_SCHEMA:
       - topic -> subtopic -> {patterns, aspects}
       - если паттерн сматчился, считаем подтему упомянутой
       - определяем sentiment этой подтемы
       - маппим подтему к аспекту (sub_def["aspects"][0])
    2) Составляем множества:
       topics_all, topics_pos, topics_neg
    3) Строим пары аспектов (ко-упоминаний):
       для каждой пары (a,b) -> категория:
         - оба neg => 'systemic_risk'
         - один neg + один pos => 'expectations_conflict'
         - оба pos => 'loyalty_driver'
    """
    topics_all: List[str] = []
    topics_pos: List[str] = []
    topics_neg: List[str] = []

    for topic_key, topic_def in TOPIC_SCHEMA.items():
        subtopics = topic_def.get("subtopics", {})
        for sub_key, sub_def in subtopics.items():
            patterns = (sub_def or {}).get("patterns", {})

            # Проверяем, встречается ли подтема в тексте:
            pats = (
                patterns.get(lang, []) or
                patterns.get("ru", []) or
                patterns.get("en", []) or
                []
            )
            matched = False
            for pat in pats:
                if re.search(pat, full_text, flags=re.IGNORECASE):
                    matched = True
                    break
            if not matched:
                continue

            # Определяем тональность этой подтемы
            local_sent = detect_subtopic_sentiment(full_text, lang, patterns)

            # Название аспекта (человекочитаемое)
            aspects = (sub_def or {}).get("aspects", [])
            if aspects:
                aspect_key = aspects[0]
            else:
                # fallback, если аспект не указан
                aspect_key = f"{topic_key}.{sub_key}"

            topics_all.append(aspect_key)
            if local_sent == "pos":
                topics_pos.append(aspect_key)
            elif local_sent == "neg":
                topics_neg.append(aspect_key)

    # Уникализируем / сортируем
    topics_all = sorted(list(set(topics_all)))
    topics_pos = sorted(list(set(topics_pos)))
    topics_neg = sorted(list(set(topics_neg)))

    # Кластерные пары (для блока B4)
    pair_tags: List[Dict[str, str]] = []
    for i in range(len(topics_all)):
        for j in range(i+1, len(topics_all)):
            a = topics_all[i]
            b = topics_all[j]

            a_pos = a in topics_pos
            a_neg = a in topics_neg
            b_pos = b in topics_pos
            b_neg = b in topics_neg

            if a_neg and b_neg:
                cat = "systemic_risk"  # массовая операционная проблема
            elif (a_pos and b_neg) or (b_pos and a_neg):
                cat = "expectations_conflict"  # конфликт ожиданий (цена/качество и т.д.)
            elif a_pos and b_pos:
                cat = "loyalty_driver"        # фактор привязанности/возврата
            else:
                # пара не даёт чёткой категории — пропускаем
                continue

            pair_tags.append({"a": a, "b": b, "cat": cat})

    return topics_all, topics_pos, topics_neg, pair_tags


# =============================================================================
# ЦИТАТЫ (для блока B5)
# =============================================================================

def extract_quote_candidates(full_text: str, limit: int = 3) -> List[str]:
    """
    Берём 1-3 характерных предложения без персональных данных,
    длиной от 15 до 300 символов.
    Это будут «цитаты гостей» в блоке B5.
    """
    quotes: List[str] = []
    # Разбивка по . ! ? … + переходы строки
    parts = re.split(r"[.!?…]+[\s\n]+", full_text)
    for part in parts:
        p = part.strip()
        if 15 <= len(p) <= 300:
            quotes.append(p)
        if len(quotes) >= limit:
            break
    return quotes


# =============================================================================
# ID отзыва (стабильный хэш)
# =============================================================================

def make_review_id(dt: datetime, source: str, raw_text: str) -> str:
    """
    Дет-идентификатор отзыва:
    дата + источник + первые ~200 символов текста → SHA-1.
    """
    base = f"{dt.date().isoformat()}|{source}|{(raw_text or '')[:200]}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


# =============================================================================
# ПУБЛИЧНАЯ ФУНКЦИЯ: 1 отзыв -> 1 нормализованная строка (reviews_semantic_raw)
# =============================================================================

def build_semantic_row(review_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    На входе:
        {
            "date": <исходная дата или excel-serial>,
            "source": "Booking"/"Google"/...,
            "rating_raw": 9.2 / 4.5 / ...,
            "text": "исходный текст отзыва",
            "lang_hint": "ru" / "en" / ...
        }

    На выходе одна готовая строка для вкладки reviews_semantic_raw:
        {
            "review_id": "...",
            "date": "2025-10-19",
            "week_key": "2025-W42",
            "month_key": "2025-10",
            "quarter_key": "2025-Q4",
            "year_key": "2025",
            "source": "Booking",
            "rating10": 9.2,
            "sentiment_overall": "pos"|"neg"|"neu",
            "topics_pos": "[...]",
            "topics_neg": "[...]",
            "topics_all": "[...]",
            "pair_tags": "[{'a':..,'b':..,'cat':..}, ...]",
            "quote_candidates": "[...]"
        }
    """
    # 1. Исходные значения
    raw_text  = (review_input.get("text") or "").strip()
    lang_hint = (review_input.get("lang_hint") or "").strip()
    source    = (review_input.get("source") or "").strip()

    # 2. Нормализация текста и языка
    text_norm = clean_text(raw_text)
    lang      = normalize_lang(lang_hint, text_norm)
    text_for_sem = translate_to_ru(text_norm, lang)  # сейчас это identity

    # 3. Дата и периоды
    dt       = parse_date_any(review_input.get("date"))
    periods  = derive_period_keys(dt)

    # 4. Рейтинг и общая тональность
    rating10 = normalize_rating_to_10(review_input.get("rating_raw"))
    sentiment_overall = detect_sentiment_label(text_for_sem, lang)

    # 5. Аспекты, пары, цитаты
    topics_all, topics_pos, topics_neg, pair_tags = extract_aspects_and_pairs(text_for_sem, lang)
    quotes = extract_quote_candidates(text_for_sem, limit=3)

    # 6. Устойчивый review_id
    rid = make_review_id(dt, source, raw_text)

    # 7. Строим финальный словарь
    return {
        "review_id": rid,
        "date": dt.date().isoformat(),
        "week_key": periods["week_key"],
        "month_key": periods["month_key"],
        "quarter_key": periods["quarter_key"],
        "year_key": periods["year_key"],
        "source": source,
        "rating10": rating10,
        "sentiment_overall": sentiment_overall,
        "topics_pos": json.dumps(sorted(list(set(topics_pos))), ensure_ascii=False),
        "topics_neg": json.dumps(sorted(list(set(topics_neg))), ensure_ascii=False),
        "topics_all": json.dumps(sorted(list(set(topics_all))), ensure_ascii=False),
        "pair_tags": json.dumps(pair_tags, ensure_ascii=False),
        "quote_candidates": json.dumps(quotes, ensure_ascii=False),
    }
