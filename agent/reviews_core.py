from __future__ import annotations

"""
reviews_core.py

Задача модуля:
1. Принять сырые отзывы из внешних источников.
2. Прогнать через лексический модуль (lexicon_module) для:
   - общей тональности отзыва,
   - тем / подтем (topic -> subtopic),
   - аспектов (aspect_code + человекочитаемые подписи).
3. Вернуть:
   a) нормализованный список разобранных отзывов (для метрик и хронологии),
   b) плоский список "хитов аспектов" (для топ-проблем / топ-драйверов).

Важно:
- Здесь мы НЕ делаем ML, только regex-правила.
- Аспект считается валидным ТОЛЬКО если:
    (1) аспект сработал в предложении,
    (2) в этом же предложении есть хотя бы одна подходящая (topic, subtopic),
    (3) этот (topic, subtopic) указан в aspect_to_subtopics для данного аспекта.
- Тональность агрегируется на уровне всего отзыва.

Ожидания по lexicon_module:
- Есть dataclass AspectRule (с display_short и long_hint).
- Есть класс LexiconModule (или совместимый объект), у которого:
    compiled_sentiment: Dict[str, Dict[str, List[Pattern]]]
        # ключи sentiment buckets:
        #   'positive_strong', 'positive_soft',
        #   'negative_soft', 'negative_strong',
        #   'neutral'
        # внутри каждого: lang -> [compiled_regex,...]

    topic_schema: Dict[str, Dict[str, Any]]
        # topic_schema[topic_key]["subtopics"][subtopic_key] = {
        #     "display": str,
        #     "patterns": {lang: [raw_regex,...]},  # исходно
        #     ... }
        # нам важны ключи тем/подтем, чтобы репортить

    compiled_topics: Dict[str, Dict[str, Dict[str, List[Pattern]]]]
        # compiled_topics[topic_key][subtopic_key][lang] -> [Pattern,...]

    aspect_rules: Dict[str, AspectRule]
        # AspectRule включает:
        #   aspect_code: str
        #   polarity_hint: str ("positive"/"negative"/"neutral")
        #   patterns_by_lang: Dict[str, List[str]]   # сырой
        #   display_short: str
        #   long_hint: str

    compiled_aspects: Dict[str, Dict[str, List[Pattern]]]
        # compiled_aspects[aspect_code][lang] -> [Pattern,...]

    aspect_to_subtopics: Dict[str, List[Tuple[str, str]]]
        # пример:
        #   "spir_friendly": [("staff_spir", "staff_attitude"), ...]

Всё это уже есть (или у тебя почти есть) в lexicon_module после шагов 1-3.
Мы просто используем.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import (
    Dict,
    List,
    Tuple,
    Any,
    Iterable,
    Optional,
    Set,
    Protocol,
)
import re
import pandas as pd

# Мы хотим заюзать AspectRule напрямую
from lexicon_module import AspectRule  # не создаёт циклов, lexicon_module не импортирует reviews_core


# -----------------------------------------------------------------------------
# 1. Доп. типы / протоколы
# -----------------------------------------------------------------------------

class LexiconProtocol(Protocol):
    """
    Неформальный контракт для lexicon_module.LexiconModule.
    Это нужно только для type hints; в рантайме не проверяется жёстко.
    """

    compiled_sentiment: Dict[str, Dict[str, List[re.Pattern]]]
    # {
    #   "positive_strong": {"ru":[...RegExp...], "en":[...], ...},
    #   "positive_soft": {...},
    #   "negative_soft": {...},
    #   "negative_strong": {...},
    #   "neutral": {...},
    # }

    topic_schema: Dict[str, Dict[str, Any]]
    # {
    #   "staff_spir": {
    #       "display": "...",
    #       "subtopics": {
    #           "staff_attitude": {
    #               "display": "...",
    #               "patterns": {"ru":[...raw...], ...},
    #               "aspects": [...],
    #           },
    #           ...
    #       }
    #   },
    #   ...
    # }

    compiled_topics: Dict[str, Dict[str, Dict[str, List[re.Pattern]]]]
    # compiled_topics[topic_key][subtopic_key][lang] -> [Pattern,...]

    aspect_rules: Dict[str, AspectRule]
    # "spir_friendly" -> AspectRule(...)

    compiled_aspects: Dict[str, Dict[str, List[re.Pattern]]]
    # compiled_aspects["spir_friendly"]["ru"] -> [Pattern,...]

    aspect_to_subtopics: Dict[str, List[Tuple[str, str]]]
    # "spir_friendly": [("staff_spir","staff_attitude"), ...]


@dataclass(frozen=True)
class ReviewRecordInput:
    """
    Входной сырый отзыв до классификации.
    Это то, что мы принимаем из пайплайна загрузки отзывов (backfill / актуальная неделя).

    rating10:
        ожидаем числовой скоринг в шкале 1..10 (float или int),
        или None, если источник его не даёт.
    created_at:
        может прийти datetime/date/str -> мы нормализуем ниже.
    lang:
        код языка ('ru','en','tr','ar','zh', ...), должен
        совпадать с тем, на что учился лексикон. Если язык не поддержан,
        мы просто ничего не матчим.
    """
    review_id: str
    source: str
    created_at: Any     # приведём к date позже
    rating10: Optional[float]
    lang: str
    text: str


@dataclass(frozen=True)
class AspectHit:
    """
    Один зафиксированный аспект внутри отзыва.
    Мы сразу связываем его с (topic, subtopic), чтобы в отчёте можно было сказать:
    'Аспект X относится к теме Y / подтеме Z'.
    """
    review_id: str
    aspect_code: str
    topic_key: str
    subtopic_key: str
    display_short: str
    long_hint: str
    polarity_hint: str  # "positive"/"negative"/"neutral"
    created_at: date
    week_key: str
    source: str
    rating10: Optional[float]
    sentiment_overall: str
    lang: str


@dataclass
class ReviewAnalysisResult:
    """
    То, что мы получаем после классификации одного отзыва.

    topic_hits:
        множество (topic_key, subtopic_key), найденных где-то в отзыве.
        Используем для аналитики по темам.

    aspects:
        список AspectHit по конкретным предложениям.

    sentiment_overall:
        итоговая тональность отзыва:
        'positive' / 'negative' / 'neutral' / 'mixed'
        (mixed — одновременно позитив и негатив).

    sentiment_detail:
        флаги срабатываний по корзинам правил
        {
            "positive_strong": bool,
            "positive_soft": bool,
            "negative_soft": bool,
            "negative_strong": bool,
            "neutral": bool
        }
        пригодится, если надо будет потом делать более тонкую агрегацию.
    """
    review_id: str
    source: str
    created_at: date
    week_key: str
    rating10: Optional[float]
    lang: str

    sentiment_overall: str
    sentiment_detail: Dict[str, bool]

    topic_hits: Set[Tuple[str, str]]
    aspects: List[AspectHit] = field(default_factory=list)

    raw_text: str = ""


# -----------------------------------------------------------------------------
# 2. Утилиты
# -----------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"[.!?…]+|\n+")


def _split_into_sentences(text: str) -> List[str]:
    """
    Очень простой сплиттер на "предложения" / смысловые фрагменты.
    Нам не нужна идеальная лингвистика, нам важно локализовать паттерны.
    """
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    # подчистим пустые
    return [p.strip() for p in parts if p and p.strip()]


def _safe_to_date(d: Any) -> date:
    """
    Нормализуем created_at -> date.
    Принимаем:
    - date
    - datetime
    - str (ISO / что pandas.to_datetime поймает)
    """
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, str):
        # на всякий: pandas более терпелив к форматам
        return pd.to_datetime(d).date()
    # крайний случай — пусть pandas попробует
    return pd.to_datetime(d).date()


def _week_key_for_date(d: date) -> str:
    """
    ISO-неделя в формате 'YYYY-W##'.
    Совместимо с metrics_core.iso_week_monday().
    """
    iso_year, iso_week, _ = d.isocalendar()  # (year, week, weekday)
    return f"{iso_year}-W{iso_week:02d}"


def _match_any(patterns: List[re.Pattern], s: str) -> bool:
    for rx in patterns:
        if rx.search(s):
            return True
    return False


# -----------------------------------------------------------------------------
# 3. Поиск тональности на уровне всего отзыва
# -----------------------------------------------------------------------------

def detect_sentiment_for_review(
    review_text: str,
    lang: str,
    lexicon: LexiconProtocol,
) -> Tuple[str, Dict[str, bool]]:
    """
    Возвращает:
      (sentiment_overall, sentiment_detail_flags)

    sentiment_detail_flags:
        {
            "positive_strong": bool,
            "positive_soft": bool,
            "negative_soft": bool,
            "negative_strong": bool,
            "neutral": bool
        }

    sentiment_overall:
        - 'negative' если есть жёсткий негатив,
        - 'positive' если есть позитив без негатива,
        - 'mixed'   если есть и позитив, и негатив,
        - иначе 'neutral'.

    Логика может эволюционировать, но это минимально рабочая версия.
    """

    buckets = [
        "positive_strong",
        "positive_soft",
        "negative_soft",
        "negative_strong",
        "neutral",
    ]

    flags: Dict[str, bool] = {b: False for b in buckets}

    # если язык не поддержан вообще, всё False -> упадём в neutral
    if not review_text:
        return "neutral", flags

    text = review_text

    for b in buckets:
        lang_map = lexicon.compiled_sentiment.get(b, {})
        pats = lang_map.get(lang, [])
        if pats and _match_any(pats, text):
            flags[b] = True

    # теперь решим общий лейбл
    any_pos = flags["positive_strong"] or flags["positive_soft"]
    any_neg = flags["negative_strong"] or flags["negative_soft"]

    if any_neg and not any_pos:
        overall = "negative"
    elif any_pos and not any_neg:
        overall = "positive"
    elif any_pos and any_neg:
        overall = "mixed"
    else:
        # ни позитива, ни негатива -> либо neutral, либо "neutral" (сигналы)
        overall = "neutral"

    return overall, flags


# -----------------------------------------------------------------------------
# 4. Поиск тем/подтем и аспектов в пределах одного предложения
# -----------------------------------------------------------------------------

def _topics_in_sentence(
    sent: str,
    lang: str,
    lexicon: LexiconProtocol,
) -> List[Tuple[str, str]]:
    """
    Вернёт список (topic_key, subtopic_key), которые встречаются в тексте sent.
    """
    hits: List[Tuple[str, str]] = []

    # бежим по всем темам
    for topic_key, topic_data in lexicon.topic_schema.items():
        subtopics = topic_data.get("subtopics", {})
        for subtopic_key, subtopic_data in subtopics.items():
            compiled_per_lang = (
                lexicon.compiled_topics
                .get(topic_key, {})
                .get(subtopic_key, {})
                .get(lang, [])
            )
            if not compiled_per_lang:
                continue
            if _match_any(compiled_per_lang, sent):
                hits.append((topic_key, subtopic_key))

    return hits


def _aspects_in_sentence(
    sent: str,
    lang: str,
    lexicon: LexiconProtocol,
    sentence_topics: List[Tuple[str, str]],
    base_review_meta: Dict[str, Any],
) -> List[AspectHit]:
    """
    sentence_topics:
        список (topic_key, subtopic_key), найденных в ЭТОМ ЖЕ предложении.

    Возвращаем список AspectHit.
    Мы считаем аспект валидным только если он "привязан" хотя бы к одной
    из найденных подтем (topic_key, subtopic_key) через lexicon.aspect_to_subtopics.
    """

    if not sentence_topics:
        return []

    sentence_topic_set = set(sentence_topics)
    out: List[AspectHit] = []

    for aspect_code, rule in lexicon.aspect_rules.items():
        compiled_lang_list = lexicon.compiled_aspects.get(aspect_code, {}).get(lang, [])
        if not compiled_lang_list:
            continue

        if not _match_any(compiled_lang_list, sent):
            continue

        # есть лексическое совпадение по аспекту. Совместим ли он с подтемами,
        # которые мы нашли в этом предложении?
        allowed_pairs = set(lexicon.aspect_to_subtopics.get(aspect_code, []))
        common_pairs = sentence_topic_set.intersection(allowed_pairs)
        if not common_pairs:
            # аспект не "привязан" ни к одной подтеме, найденной в этом предложении,
            # значит, мы не считаем его валидным.
            continue

        # Если несколько пар совпали, задокументируем первую.
        topic_key, subtopic_key = next(iter(common_pairs))

        hit = AspectHit(
            review_id=base_review_meta["review_id"],
            aspect_code=aspect_code,
            topic_key=topic_key,
            subtopic_key=subtopic_key,
            display_short=getattr(rule, "display_short", aspect_code),
            long_hint=getattr(rule, "long_hint", ""),
            polarity_hint=rule.polarity_hint,
            created_at=base_review_meta["created_at"],
            week_key=base_review_meta["week_key"],
            source=base_review_meta["source"],
            rating10=base_review_meta["rating10"],
            sentiment_overall=base_review_meta["sentiment_overall"],
            lang=base_review_meta["lang"],
        )
        out.append(hit)

    return out


# -----------------------------------------------------------------------------
# 5. Анализ одного отзыва целиком
# -----------------------------------------------------------------------------

def analyze_single_review(
    raw: ReviewRecordInput,
    lexicon: LexiconProtocol,
) -> ReviewAnalysisResult:
    """
    Основная функция для одного отзыва.
    Делает:
      - нормализацию даты,
      - вычисление week_key,
      - определение тональности,
      - разбор по предложениям, поиск подтем и аспектов,
      - сбор всего результата в ReviewAnalysisResult.
    """

    created_at_date = _safe_to_date(raw.created_at)
    week_key = _week_key_for_date(created_at_date)

    sentiment_overall, sentiment_detail = detect_sentiment_for_review(
        review_text=raw.text,
        lang=raw.lang,
        lexicon=lexicon,
    )

    # метаданные, которые нужны аспектам:
    base_meta = {
        "review_id": raw.review_id,
        "created_at": created_at_date,
        "week_key": week_key,
        "source": raw.source,
        "rating10": raw.rating10,
        "sentiment_overall": sentiment_overall,
        "lang": raw.lang,
    }

    all_topic_hits: Set[Tuple[str, str]] = set()
    all_aspect_hits: List[AspectHit] = []

    # нарежем на куски (условно "предложения") и пройдемся
    for sent in _split_into_sentences(raw.text):
        # на уровне предложения находим темы/подтемы
        st_topics = _topics_in_sentence(sent, raw.lang, lexicon)
        if st_topics:
            all_topic_hits.update(st_topics)

        # на уровне предложения находим аспекты,
        # разрешая только те, которые "подвязаны" к найденным здесь подтемам.
        st_aspects = _aspects_in_sentence(
            sent=sent,
            lang=raw.lang,
            lexicon=lexicon,
            sentence_topics=st_topics,
            base_review_meta=base_meta,
        )
        if st_aspects:
            all_aspect_hits.extend(st_aspects)

    result = ReviewAnalysisResult(
        review_id=raw.review_id,
        source=raw.source,
        created_at=created_at_date,
        week_key=week_key,
        rating10=raw.rating10,
        lang=raw.lang,
        sentiment_overall=sentiment_overall,
        sentiment_detail=sentiment_detail,
        topic_hits=all_topic_hits,
        aspects=all_aspect_hits,
        raw_text=raw.text,
    )
    return result


# -----------------------------------------------------------------------------
# 6. Анализ пачки отзывов и подготовка DataFrame'ов
# -----------------------------------------------------------------------------

def analyze_reviews_bulk(
    raw_reviews: Iterable[ReviewRecordInput],
    lexicon: LexiconProtocol,
) -> List[ReviewAnalysisResult]:
    """
    Удобная обёртка: берёт генератор/список ReviewRecordInput,
    возвращает список ReviewAnalysisResult.
    """
    results: List[ReviewAnalysisResult] = []
    for r in raw_reviews:
        try:
            res = analyze_single_review(r, lexicon)
            results.append(res)
        except Exception as e:
            # продакшен-версия может логировать/репортить.
            # Здесь мы просто пропускаем падающие отзывы,
            # но важно не обрушать весь отчёт.
            # Можно также сохранить "пустой" с neutral.
            # Пока делаем soft-fail.
            continue
    return results


def build_reviews_dataframe(
    analyzed_reviews: Iterable[ReviewAnalysisResult],
) -> pd.DataFrame:
    """
    Превращает результаты анализа отзывов в табличку,
    с которой потом будет работать metrics_core.build_history().

    ВАЖНО:
    build_history() ожидает:
        ['review_id','source','created_at','week_key','rating10','sentiment_overall']
    + мы можем добавить вспомогательные поля (lang, topics, aspects), они не помешают.

    topics:
        список всех (topic_key, subtopic_key) по отзыву.
        Храним как list[Tuple[str,str]] -> pandas сделает dtype=object;
        downstream (отчёт) сможет распарсить.

    aspects:
        список кодов аспектов, сработавших в отзыве.
        (уникализируем коды на уровне отзыва, чтобы не плодить дубликаты).
    """
    rows: List[Dict[str, Any]] = []

    for r in analyzed_reviews:
        topics_list = sorted(list(r.topic_hits))
        aspects_list = sorted({a.aspect_code for a in r.aspects})

        rows.append({
            "review_id": r.review_id,
            "source": r.source,
            "created_at": r.created_at,
            "week_key": r.week_key,
            "rating10": r.rating10,
            "sentiment_overall": r.sentiment_overall,
            "lang": r.lang,
            "topics": topics_list,
            "aspects": aspects_list,
            "raw_text": r.raw_text,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "review_id","source","created_at","week_key","rating10",
            "sentiment_overall","lang","topics","aspects","raw_text",
        ])

    df = pd.DataFrame(rows)
    return df


def build_aspects_dataframe(
    analyzed_reviews: Iterable[ReviewAnalysisResult],
) -> pd.DataFrame:
    """
    Возвращает "взрыв" по аспектам:
    каждая строка = один AspectHit.

    Это нужно для:
      - топ проблем (негативные аспекты),
      - топ драйверов (позитивные аспекты),
      - drilldown в отчёте (пример: "шум от техники" или "дружелюбный персонал").

    Колонки:
        review_id
        aspect_code
        topic_key
        subtopic_key
        display_short
        long_hint
        polarity_hint    ("positive"/"negative"/"neutral")
        created_at
        week_key
        source
        rating10
        sentiment_overall (тональность целого отзыва)
        lang
    """
    rows: List[Dict[str, Any]] = []

    for r in analyzed_reviews:
        for a in r.aspects:
            rows.append({
                "review_id": a.review_id,
                "aspect_code": a.aspect_code,
                "topic_key": a.topic_key,
                "subtopic_key": a.subtopic_key,
                "display_short": a.display_short,
                "long_hint": a.long_hint,
                "polarity_hint": a.polarity_hint,
                "created_at": a.created_at,
                "week_key": a.week_key,
                "source": a.source,
                "rating10": a.rating10,
                "sentiment_overall": a.sentiment_overall,
                "lang": a.lang,
            })

    if not rows:
        return pd.DataFrame(columns=[
            "review_id","aspect_code","topic_key","subtopic_key",
            "display_short","long_hint","polarity_hint",
            "created_at","week_key","source","rating10",
            "sentiment_overall","lang",
        ])

    df = pd.DataFrame(rows)
    return df


# -----------------------------------------------------------------------------
# 7. Публичный API этого модуля
# -----------------------------------------------------------------------------

__all__ = [
    # датаклассы
    "ReviewRecordInput",
    "AspectHit",
    "ReviewAnalysisResult",
    # функции анализа
    "analyze_single_review",
    "analyze_reviews_bulk",
    # функции агрегации в датафреймы
    "build_reviews_dataframe",
    "build_aspects_dataframe",
]
