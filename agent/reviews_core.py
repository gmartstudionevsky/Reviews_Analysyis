from __future__ import annotations

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

# --- пакетные импорты внутри agent ---
from .metrics_core import iso_week_monday, period_ranges_for_week
from .lexicon_module import AspectRule


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
    sentiment_score: float

    topic_hits: Set[Tuple[str, str]]
    aspects: List[AspectHit] = field(default_factory=list)

    raw_text: str = ""


# -----------------------------------------------------------------------------
# 2. Утилиты
# -----------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"[.!?…]+|\n+")

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)

def _normalize_text(text: str) -> str:
    """
    Мягкая нормализация: убираем URL/Email, схлопываем пробелы.
    Ничего принципиального не выкидываем, чтобы не ломать матчи.
    """
    if not text:
        return ""
    s = _URL_RE.sub(" ", text)
    s = _EMAIL_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s, flags=re.UNICODE).strip()
    return s

def _split_into_sentences(text: str) -> List[str]:
    """
    Очень простой сплиттер на "предложения" / смысловые фрагменты.
    Нормализуем мягко и режем по знакам конца фразы / переводу строки.
    """
    if not text:
        return []
    text = _normalize_text(text)
    parts = _SENTENCE_SPLIT_RE.split(text)
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

def _candidate_langs(lang: str) -> List[str]:
    """
    "ru-RU" -> ["ru-ru", "ru", "en"]; пустой -> ["en"].
    """
    lang = (lang or "").strip().lower()
    cands: List[str] = []
    if lang:
        cands.append(lang)
        if "-" in lang:
            short = lang.split("-")[0]
            if short not in cands:
                cands.append(short)
    if "en" not in cands:
        cands.append("en")
    return cands

def _label_pos_neg_neu(sentiment_overall: str, rating10: Optional[float]) -> str:
    """
    Классификация отзыва недели:
      - Позитив: (тональность overall == 'positive') ИЛИ (rating10 >= 9)
      - Негатив: (тональность overall == 'negative') ИЛИ (rating10 <= 6)
      - Иначе: neutral
    """
    is_pos = (sentiment_overall == "positive") or (rating10 is not None and rating10 >= 9.0)
    is_neg = (sentiment_overall == "negative") or (rating10 is not None and rating10 <= 6.0)
    if is_pos and not is_neg:
        return "positive"
    if is_neg and not is_pos:
        return "negative"
    return "neutral"

def slice_periods(df_reviews: "pd.DataFrame", anchor_week_key: str) -> Dict[str, "pd.DataFrame"]:
    """
    Возвращает словарь { 'week','mtd','qtd','ytd','all' } → DataFrame.
    Ожидается, что df_reviews содержит 'created_at' (date) и 'week_key'.
    """
    if df_reviews is None or len(df_reviews) == 0:
        return {"week": df_reviews.copy(), "mtd": df_reviews.copy(),
                "qtd": df_reviews.copy(), "ytd": df_reviews.copy(), "all": df_reviews.copy()}

    wk_monday = iso_week_monday(anchor_week_key)
    ranges = period_ranges_for_week(wk_monday)  # {'week':{'start','end'}, ...}

    def _cut(df, start, end):
        m = (df["created_at"] >= start) & (df["created_at"] <= end)
        return df.loc[m].copy()

    out = {
        "week": _cut(df_reviews, ranges["week"]["start"], ranges["week"]["end"]),
        "mtd":  _cut(df_reviews, ranges["mtd"]["start"],  ranges["mtd"]["end"]),
        "qtd":  _cut(df_reviews, ranges["qtd"]["start"],  ranges["qtd"]["end"]),
        "ytd":  _cut(df_reviews, ranges["ytd"]["start"],  ranges["ytd"]["end"]),
        "all":  df_reviews.copy(),
    }
    return out

def build_source_pivot(df_period: "pd.DataFrame") -> "pd.DataFrame":
    """
    Сводка по источникам за выбранный период.
    На выходе: source, reviews, avg10, pos_pct, neg_pct, pos_cnt, neg_cnt.
    (Визуальная нативная шкала /5 делается на уровне отчёта, не здесь.)
    """
    import pandas as pd  # локальный импорт, чтобы не ломать ранние импорты

    if df_period is None or len(df_period) == 0:
        return pd.DataFrame(columns=["source","reviews","avg10","pos_pct","neg_pct","pos_cnt","neg_cnt"])

    df = df_period.copy()
    df["__label__"] = df.apply(lambda r: _label_pos_neg_neu(r.get("sentiment_overall"), r.get("rating10")), axis=1)

    grp = df.groupby("source", dropna=False)
    agg = grp.agg(
        reviews=("review_id", "nunique"),
        avg10=("rating10", "mean"),
        pos_cnt=("__label__", lambda s: (s == "positive").sum()),
        neg_cnt=("__label__", lambda s: (s == "negative").sum()),
    ).reset_index()

    agg["pos_pct"] = (agg["pos_cnt"] / agg["reviews"]).fillna(0.0).round(4)
    agg["neg_pct"] = (agg["neg_cnt"] / agg["reviews"]).fillna(0.0).round(4)
    agg["avg10"] = agg["avg10"].round(2)

    return agg[["source","reviews","avg10","pos_pct","neg_pct","pos_cnt","neg_cnt"]].sort_values("reviews", ascending=False).reset_index(drop=True)

def _score_from_flags_and_rating(flags: Dict[str, bool], rating10: Optional[float]) -> float:
    """
    Возвращает sentiment_score в диапазоне [-1.0 .. 1.0].
    Комбинация: текст (60%) + оценка гостя (40%, если есть).
    Текстовая сила:
      pos_strong=+1.0, pos_soft=+0.6, neg_soft=-0.6, neg_strong=-1.0; суммируем и клипуем.
    Нормализация оценки 1..10 -> [-1..1]: (x - 5.5)/4.5.
    """
    pos = (1.0 if flags.get("positive_strong") else 0.0) + (0.6 if flags.get("positive_soft") else 0.0)
    neg = (1.0 if flags.get("negative_strong") else 0.0) + (0.6 if flags.get("negative_soft") else 0.0)
    text_score = pos - neg
    text_score = max(-1.0, min(1.0, text_score))

    if rating10 is None:
        return float(text_score)

    rating_norm = (float(rating10) - 5.5) / 4.5
    rating_norm = max(-1.0, min(1.0, rating_norm))
    return float(round(0.6 * text_score + 0.4 * rating_norm, 4))

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
    Логика:
      - ищем ключевые корзины тональностей по списку кандидатов языков,
      - учитываем мягкую нормализацию текста,
      - итог: 'negative' / 'positive' / 'mixed' / 'neutral'.
    """
    buckets = [
        "positive_strong",
        "positive_soft",
        "negative_soft",
        "negative_strong",
        "neutral",
    ]
    flags: Dict[str, bool] = {b: False for b in buckets}
    if not review_text:
        return "neutral", flags

    text = _normalize_text(review_text)
    for b in buckets:
        lang_map = lexicon.compiled_sentiment.get(b, {})
        pats: List[re.Pattern] = []
        for cand in _candidate_langs(lang):
            pats.extend(lang_map.get(cand, []))
        if pats and _match_any(pats, text):
            flags[b] = True

    any_pos = flags["positive_strong"] or flags["positive_soft"]
    any_neg = flags["negative_strong"] or flags["negative_soft"]

    if any_neg and not any_pos:
        overall = "negative"
    elif any_pos and not any_neg:
        overall = "positive"
    elif any_pos and any_neg:
        overall = "mixed"
    else:
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
    Учитываем кандидатов языка: lang, short-lang, en.
    """
    hits: List[Tuple[str, str]] = []
    if not sent:
        return hits

    for topic_key, topic_data in lexicon.topic_schema.items():
        subtopics = topic_data.get("subtopics", {})
        for subtopic_key, _sub_def in subtopics.items():
            pats: List[re.Pattern] = []
            compiled_map = lexicon.compiled_topics.get(topic_key, {}).get(subtopic_key, {})
            for cand in _candidate_langs(lang):
                pats.extend(compiled_map.get(cand, []))
            if pats and _match_any(pats, sent):
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
    Валидируем аспект только если он "подвязан" к найденным в предложении подтемам.
    """
    if not sentence_topics or not sent:
        return []

    sentence_topic_set = set(sentence_topics)
    out: List[AspectHit] = []

    for aspect_code, rule in lexicon.aspect_rules.items():
        pats: List[re.Pattern] = []
        compiled_lang_map = lexicon.compiled_aspects.get(aspect_code, {})
        for cand in _candidate_langs(lang):
            pats.extend(compiled_lang_map.get(cand, []))
        if not pats or not _match_any(pats, sent):
            continue

        allowed_pairs = set(lexicon.aspect_to_subtopics.get(aspect_code, []))
        common_pairs = sentence_topic_set.intersection(allowed_pairs)
        if not common_pairs:
            continue

        topic_key, subtopic_key = next(iter(common_pairs))
        out.append(AspectHit(
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
        ))

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
    # числовой скоринг тональности
    sentiment_score = _score_from_flags_and_rating(sentiment_detail, raw.rating10)


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
        sentiment_score=sentiment_score,
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
            "sentiment_score": r.sentiment_score,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "review_id","source","created_at","week_key","rating10",
            "sentiment_overall","sentiment_score","lang","topics","aspects","raw_text",
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

def compute_aspect_impacts(
    df_reviews_period: "pd.DataFrame",
    df_aspects_period: "pd.DataFrame",
) -> "pd.DataFrame":
    """
    Возвращает таблицу по аспектам с метриками частоты, интенсивности и связи с экстремальными оценками.
    Формулы (см. ТЗ 0.4):
      - freq = уникальные отзывы с аспектом / все отзывы периода
      - intensity_pos: средняя "сила" для положительных аспектов (1.0 при rating10>=9, 0.6 при 7..8, иначе 0; если rating10 NaN, учитываем sentiment_overall)
      - intensity_neg: аналогично для отрицательных аспектов (1.0 при rating10<=6, 0.6 при 7..8, иначе 0)
      - share_hi = доля упоминаний аспекта в отзывах с rating10>=9
      - share_lo = доля упоминаний аспекта в отзывах с rating10<=6
      - positive_impact_index  = 0.50*freq + 0.30*intensity_pos + 0.20*share_hi
      - negative_impact_index  = 0.50*freq + 0.30*intensity_neg + 0.20*share_lo
    """
    import numpy as np
    import pandas as pd

    # Пустые входы → пустая сводка
    if df_reviews_period is None or len(df_reviews_period) == 0:
        return pd.DataFrame(columns=[
            "aspect_code","topic_key","subtopic_key","display_short","long_hint",
            "reviews_with_aspect","freq","pos_hits","neg_hits",
            "intensity_pos","intensity_neg","share_hi","share_lo",
            "positive_impact_index","negative_impact_index",
        ])
    if df_aspects_period is None or len(df_aspects_period) == 0:
        # аспектов нет — вернём пустую, чтобы отчёт корректно отработал "в пределах фона"
        return pd.DataFrame(columns=[
            "aspect_code","topic_key","subtopic_key","display_short","long_hint",
            "reviews_with_aspect","freq","pos_hits","neg_hits",
            "intensity_pos","intensity_neg","share_hi","share_lo",
            "positive_impact_index","negative_impact_index",
        ])

    # Берём только нужные колонки
    rev = df_reviews_period[["review_id","rating10","sentiment_overall"]].drop_duplicates("review_id").copy()
    asp = df_aspects_period[[
        "aspect_code","review_id","polarity_hint","topic_key","subtopic_key","display_short","long_hint"
    ]].copy()

    # Дедуп: один отзыв считается один раз на аспект
    asp = asp.drop_duplicates(subset=["aspect_code","review_id"]).copy()

    # Join для оценок/тональности отзыва
    m = asp.merge(rev, on="review_id", how="left")

    # Бинарные признаки
    m["is_pos_hit"] = (m["polarity_hint"] == "positive")
    m["is_neg_hit"] = (m["polarity_hint"] == "negative")

    # Бины оценок
    r = m["rating10"]
    m["hi"]  = r.ge(9.0).fillna(False)
    m["mid"] = r.between(7.0, 8.0, inclusive="both").fillna(False)
    m["lo"]  = r.le(6.0).fillna(False)

    # Весовые contribution для интенсивности
    so = m["sentiment_overall"].fillna("neutral")
    m["w_pos"] = np.where(m["is_pos_hit"] & m["hi"], 1.0,
                   np.where(m["is_pos_hit"] & m["mid"], 0.6,
                   np.where(m["is_pos_hit"] & (so == "positive"), 0.6, 0.0)))
    m["w_neg"] = np.where(m["is_neg_hit"] & m["lo"], 1.0,
                   np.where(m["is_neg_hit"] & m["mid"], 0.6,
                   np.where(m["is_neg_hit"] & (so == "negative"), 0.6, 0.0)))

    total_reviews = max(1, rev["review_id"].nunique())

    def _agg(group: "pd.DataFrame") -> "pd.Series":
        reviews_with_aspect = group["review_id"].nunique()
        freq = reviews_with_aspect / total_reviews

        pos_hits = int((group["is_pos_hit"]).sum())
        neg_hits = int((group["is_neg_hit"]).sum())

        # интенсивности считаем только по соответствующим знакам; если нет — 0
        intensity_pos = float(group.loc[group["is_pos_hit"], "w_pos"].mean()) if pos_hits > 0 else 0.0
        intensity_neg = float(group.loc[group["is_neg_hit"], "w_neg"].mean()) if neg_hits > 0 else 0.0

        total_mentions = max(1, len(group))
        share_hi = float(group["hi"].sum()) / total_mentions
        share_lo = float(group["lo"].sum()) / total_mentions

        positive_impact_index = 0.50*freq + 0.30*intensity_pos + 0.20*share_hi
        negative_impact_index = 0.50*freq + 0.30*intensity_neg + 0.20*share_lo

        return pd.Series({
            "reviews_with_aspect": reviews_with_aspect,
            "freq": round(freq, 4),
            "pos_hits": pos_hits,
            "neg_hits": neg_hits,
            "intensity_pos": round(float(intensity_pos), 4),
            "intensity_neg": round(float(intensity_neg), 4),
            "share_hi": round(float(share_hi), 4),
            "share_lo": round(float(share_lo), 4),
            "positive_impact_index": round(float(positive_impact_index), 4),
            "negative_impact_index": round(float(negative_impact_index), 4),
        })

    agg = (m.groupby(["aspect_code","topic_key","subtopic_key","display_short","long_hint"], dropna=False)
             .apply(_agg)
             .reset_index())

    # Сортировка: сначала сильные риски, затем драйверы (для удобства отбора)
    agg = agg.sort_values(
        by=["negative_impact_index","positive_impact_index","reviews_with_aspect"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return agg

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
    "slice_periods",
    "build_source_pivot",
    "compute_aspect_impacts",
]
