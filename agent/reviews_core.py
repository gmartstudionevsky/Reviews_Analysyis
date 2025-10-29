"""
reviews_core.py

Назначение:
- преобразовать сырой текст отзыва (review) в структурированные сущности:
  * список предложений с тональностью
  * список аспект-упоминаний с категорией, подтемой, тональностью и т.д.

Это ядро пайплайна:
  raw_review -> ParsedReviewResult -> (metrics_core агрегирует -> weekly_report_agent рисует инсайты)

Основные сущности:
- RawReview: всё, что пришло "как есть" из источника (отель, дата, текст...)
- SentenceInfo: отдельное предложение с попыткой определить тональность
- AspectMention: одно конкретное "гость пожаловался на шум кондиционера"
- ParsedReviewResult: всё вместе, уже готовое к агрегации

"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
from datetime import datetime, date

# Предполагается, что lexicon_module.py лежит рядом и содержит класс Lexicon
# с методами, на которые мы опираемся ниже.
#
# В частности мы используем следующие публичные методы Lexicon:
#   - match_sentiment_group(text: str, lang: str) -> Optional[str]
#       Возвращает тональность предложения, например:
#           "positive_strong", "positive_soft",
#           "negative_strong", "negative_soft",
#           "neutral", "mixed", None
#
#   - match_aspects(text: str, lang: str) -> List[str]
#       Возвращает list[aspect_code], например ["spir_friendly", "hallway_smell_smoke"]
#
#   - get_aspect_rule(aspect_code: str) -> Optional[AspectRule]
#       Возвращает объект AspectRule с .polarity_hint, .display_short, .long_hint
#
#   - get_aspect_topics(aspect_code: str) -> List[Tuple[str, str]]
#       Возвращает список пар (category_key, subtopic_key),
#       например [("staff_spir", "staff_attitude")]
#
#   - get_aspect_display_short(aspect_code: str) -> Optional[str]
#       Возвращает короткое человекочитаемое имя аспекта для репорта
#
#   - get_aspect_long_hint(aspect_code: str) -> Optional[str]
#       Возвращает пояснение аспекта на человеческом языке
#
#   - get_polarity_hint(aspect_code: str) -> Optional[str]
#       Возвращает встроенный "базовый" знак аспекта, например "negative" для
#       запаха сигарет, на случай если в предложении не нашли явного
#       лексического маркера тональности.
#
# Если у тебя названия методов отличаются — просто адаптируй в одном месте.
#
from lexicon_module import Lexicon  # type: ignore


###############################################################################
# Константы и утилиты
###############################################################################

# маппинг детальной тональности предложения -> укрупнённой корзины
# Эти ярлыки пойдут в метрики и отчёты.
_SENTIMENT_COLLAPSE_MAP: Dict[str, str] = {
    "positive_strong": "positive",
    "positive_soft": "positive",
    "negative_strong": "negative",
    "negative_soft": "negative",
    "neutral": "neutral",
    "mixed": "neutral",  # спорно, но безопасно: "и плюс, и минус"
}


def collapse_sentiment_group(sent_group: Optional[str]) -> str:
    """
    Превращаем детальную группу тональности в укрупнённую корзину:
        "positive" / "negative" / "neutral"
    Если не уверены — считаем neutral.
    """
    if not sent_group:
        return "neutral"
    return _SENTIMENT_COLLAPSE_MAP.get(sent_group, "neutral")


def _clean_text_basic(text: str) -> str:
    """
    Базовая нормализация текста перед regex-поиском:
    - приведение к нижнему регистру
    - схлопывание множественных пробелов
    - трим
    """
    lowered = text.lower()
    # убираем повторяющиеся пробелы/табуляции/переводы строк -> один пробел
    cleaned = re.sub(r"\s+", " ", lowered).strip()
    return cleaned


def _simple_sentence_split(text: str) -> List[str]:
    """
    Очень простая нарезка на "предложения".
    Мы НЕ используем внешние NLP-библиотеки.
    Это лучше, чем ничего, и работает приемлемо для отзывов.

    Логика:
    - сплитим по [.!?…] и переносам строки
    - чистим пробелы
    - отсекаем пустые

    ВНИМАНИЕ: для китайского / арабского без точек
    мы всё равно дадим всю реплику как одно "предложение",
    что нас устраивает на данном этапе.
    """
    # заменяем переводы строк на точки, чтобы их тоже порезать
    tmp = text.replace("\n", ". ").replace("...", ". ")
    # сплит по . ! ? … (многоточие мы уже схлопнули)
    chunks = re.split(r"[\.!?…]+", tmp)
    out: List[str] = []
    for ch in chunks:
        sent = ch.strip()
        if sent:
            out.append(sent)
    if not out:
        # fallback чтобы точно было хотя бы одно "предложение"
        out = [text.strip()]
    return out


def _contains_cyrillic(s: str) -> bool:
    return bool(re.search(r"[а-яё]", s.lower()))


def _contains_arabic(s: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", s))


def _contains_cjk(s: str) -> bool:
    # Диапазон CJK Unified Ideographs (китайский, японский, корейский ханы)
    return bool(re.search(r"[\u4e00-\u9fff]", s))


def _detect_lang_simple(text: str) -> str:
    """
    Очень простая эвристика для языка.
    Если у тебя уже есть нормальный lang_id в апстриме — просто передавай его
    в RawReview.lang, и тогда этот детектор не будет использоваться.
    """
    t = text.strip()
    if not t:
        return "en"

    if _contains_cyrillic(t):
        return "ru"
    if _contains_arabic(t):
        return "ar"
    if _contains_cjk(t):
        # упрощенно считаем zh
        return "zh"

    # можно ещё чуть-чуть турецкий по специфичным буквам
    if re.search(r"[çğıİöşüÇĞİÖŞÜ]", t):
        return "tr"

    # дефолтимся в английский
    return "en"


###############################################################################
# Основные датаклассы
###############################################################################

@dataclass
class RawReview:
    """
    Исходные метаданные и текст отзыва.

    review_id: уникальный ID отзыва в источнике
    hotel_id:  к какому отелю относится отзыв
    text:      сырой текст отзыва (как написал гость)
    created_at: когда отзыв был оставлен на платформе (UTC или локаль — неважно для нас здесь)
    stay_date:  дата/месяц проживания гостя, если известна
    rating:     числовой рейтинг (например, 7.0/10), если есть
    lang:       язык отзыва, если уже определён упстримом. Если None — делаем эвристику.
    """
    review_id: str
    hotel_id: str
    text: str
    created_at: Optional[datetime] = None
    stay_date: Optional[date] = None
    rating: Optional[float] = None
    lang: Optional[str] = None


@dataclass
class SentenceInfo:
    """
    Отдельное предложение после нарезки.

    sentence_id:
        стабильный ID внутри отзыва: f"{review_id}__{idx}"
        Нужен, чтобы потом связать аспект с конкретной репликой.

    text:
        оригинальный текст предложения (без lowercase),
        чтобы в отчёте можно было процитировать.

    lang:
        язык, на котором мы считаем regex-ы.

    sentiment_group:
        детальная оценка тональности этого предложения:
        "positive_strong", "positive_soft", "negative_strong", "negative_soft",
        "neutral", "mixed", или None.
    """
    sentence_id: str
    text: str
    lang: str
    sentiment_group: Optional[str]


@dataclass
class AspectMention:
    """
    Упоминание аспекта в конкретном предложении.

    Это то, что потом агрегируется в метриках.

    review_id / hotel_id:
        чтобы потом группировать по отелю, по периоду, по конкретному отзыву

    sentence_id / sentence_text:
        чтобы уметь показать "первичный источник жалобы/похвалы" в отчёте

    aspect_code:
        внутренний код аспекта (например, "hallway_smell_smoke")

    category_key / subtopic_key:
        к какой теме и подтеме это относится ("cleanliness" / "smell_common_areas")

    sentiment_group:
        детальная тональность ("negative_strong" и т.д.),
        финальная для этого упоминания

    sentiment_bucket:
        укрупнённая корзина "negative" / "positive" / "neutral"
        -> будет использоваться в метриках и графиках

    polarity_source:
        "explicit" - если в предложении нашли явный маркер тональности
        "default_aspect_hint" - если не нашли явной тональности и упали
                                 на polarity_hint самого аспекта

    aspect_display_short:
        человечное короткое имя аспекта,
        например "запах сигарет в общих зонах"
        Используется в отчёте/дашборде.

    aspect_hint:
        длинная пояснялка (long_hint),
        например "Гости пишут, что в коридорах пахнет сигаретами/дымом."
        Используем в аналитике и комментариях к слайдам.
    """
    review_id: str
    hotel_id: str

    sentence_id: str
    sentence_text: str

    aspect_code: str
    category_key: str
    subtopic_key: str

    sentiment_group: str
    sentiment_bucket: str
    polarity_source: str

    aspect_display_short: Optional[str]
    aspect_hint: Optional[str]

    created_at: Optional[datetime]
    stay_date: Optional[date]
    rating: Optional[float]


@dataclass
class ParsedReviewResult:
    """
    Полный результат парсинга одного отзыва.

    language:
        язык отзыва (либо из RawReview.lang, либо автоопределение).
        Он же прокинут в каждый SentenceInfo и AspectMention.

    sentences:
        список предложений с тональностью.

    mentions:
        список аспектных упоминаний (каждое - потенциальная метрика).
    """
    review: RawReview
    language: str
    sentences: List[SentenceInfo]
    mentions: List[AspectMention]


###############################################################################
# Основной парсер
###############################################################################

class ReviewParser:
    """
    Главный класс, который превращает RawReview -> ParsedReviewResult.

    Вкратце:
        1. определить язык
        2. разбить отзыв на предложения
        3. для каждого предложения:
            - определить тональность (match_sentiment_group)
            - найти аспекты (match_aspects)
            - для каждого аспекта сделать AspectMention
              (с категорией, подтемой, тональностью и т.д.)

    Важный смысловой момент:
    Если явно-тональная лексика (из словарей тональности) не найдена,
    мы fallback-аемся на polarity_hint аспекта (например, "смрад в коридоре"
    почти точно негатив, даже если человек не сказал "ужасно").
    """

    def __init__(
        self,
        lexicon: Lexicon,
        lang_detector: Optional[Callable[[str], str]] = None,
    ):
        """
        lexicon:
            Экземпляр Lexicon (тот самый модуль словарей).
        lang_detector:
            Опционально можно передать свой детектор языка.
            Если None -> используется наш _detect_lang_simple.
        """
        self.lexicon = lexicon
        self._lang_detector = lang_detector or _detect_lang_simple

    ###########################################################################
    # Публичный метод
    ###########################################################################

    def parse_review(self, review: RawReview) -> ParsedReviewResult:
        """
        Главная точка входа.
        Возвращает ParsedReviewResult, содержащий:
        - список предложений
        - список аспектных упоминаний
        """
        lang = review.lang or self._lang_detector(review.text or "")

        # 1. Режем отзыв на предложения
        raw_sentences = _simple_sentence_split(review.text)

        sentences_info: List[SentenceInfo] = []
        mentions: List[AspectMention] = []

        for idx, sent_original in enumerate(raw_sentences):
            sentence_id = f"{review.review_id}__{idx}"

            # Чистим, чтобы матчить регексы
            sent_clean = _clean_text_basic(sent_original)

            # 2.1 Тональность предложения по словарям
            sent_sentiment_group = self.lexicon.match_sentiment_group(
                text=sent_clean,
                lang=lang,
            )

            # 2.2 Регистрируем предложение как SentenceInfo
            sentence_info = SentenceInfo(
                sentence_id=sentence_id,
                text=sent_original.strip(),
                lang=lang,
                sentiment_group=sent_sentiment_group,
            )
            sentences_info.append(sentence_info)

            # 2.3 Найти аспекты в предложении
            aspect_codes = self.lexicon.match_aspects(
                text=sent_clean,
                lang=lang,
            )

            # 2.4 Для каждого аспекта построим AspectMention
            for aspect_code in aspect_codes:
                aspect_mentions_for_this_aspect = self._build_mentions_for_aspect(
                    review=review,
                    sentence_info=sentence_info,
                    aspect_code=aspect_code,
                )
                mentions.extend(aspect_mentions_for_this_aspect)

        return ParsedReviewResult(
            review=review,
            language=lang,
            sentences=sentences_info,
            mentions=mentions,
        )

    ###########################################################################
    # Внутренняя логика
    ###########################################################################

    def _build_mentions_for_aspect(
        self,
        review: RawReview,
        sentence_info: SentenceInfo,
        aspect_code: str,
    ) -> List[AspectMention]:
        """
        Из одного аспекта в одном предложении может получиться несколько записей,
        потому что аспект может быть привязан к нескольким (category, subtopic).

        Например:
            aspect_code="spir_friendly"
            -> [("staff_spir", "staff_attitude")]

        или в теории один аспект мог бы маппиться на несколько субтопиков.

        Возвращает список AspectMention.
        """

        # Получаем маппинг (категория, подтема)
        cats_subs = self.lexicon.get_aspect_topics(aspect_code)
        # На всякий случай защитимся
        if not cats_subs:
            cats_subs = [("unknown_category", "unknown_subtopic")]

        # Итоговая тональность для этого конкретного аспекта в этом предложении
        final_sentiment_group, sentiment_bucket, polarity_source = \
            self._decide_aspect_sentiment(
                aspect_code=aspect_code,
                sentence_group=sentence_info.sentiment_group,
            )

        # Человеческие тексты для отчёта
        display_short = self.lexicon.get_aspect_display_short(aspect_code)
        long_hint = self.lexicon.get_aspect_long_hint(aspect_code)

        aspect_mentions: List[AspectMention] = []

        for category_key, subtopic_key in cats_subs:
            aspect_mentions.append(
                AspectMention(
                    review_id=review.review_id,
                    hotel_id=review.hotel_id,
                    sentence_id=sentence_info.sentence_id,
                    sentence_text=sentence_info.text,

                    aspect_code=aspect_code,
                    category_key=category_key,
                    subtopic_key=subtopic_key,

                    sentiment_group=final_sentiment_group,
                    sentiment_bucket=sentiment_bucket,
                    polarity_source=polarity_source,

                    aspect_display_short=display_short,
                    aspect_hint=long_hint,

                    created_at=review.created_at,
                    stay_date=review.stay_date,
                    rating=review.rating,
                )
            )

        return aspect_mentions

    def _decide_aspect_sentiment(
        self,
        aspect_code: str,
        sentence_group: Optional[str],
    ) -> Tuple[str, str, str]:
        """
        Определяем финальную тональность для данного аспект-упоминания.

        Алгоритм:
        1. Если по самому предложению найдена тональность (sentence_group != None):
            - используем её.
            - sentiment_bucket = collapse_sentiment_group(sentence_group)
            - polarity_source = "explicit"
        2. Иначе: fallback к polarity_hint аспекта.
            - берём lexicon.get_polarity_hint(aspect_code)
            - маппим это к условному fine-grain ("positive_soft"/"negative_soft"/"neutral")
              чтобы metrics_core потом мог это агрегировать без провалов.
            - polarity_source = "default_aspect_hint"

        Возвращает кортеж:
            (final_sentiment_group, sentiment_bucket, polarity_source)
        """

        # 1. Есть явный маркер тональности в предложении?
        if sentence_group:
            bucket = collapse_sentiment_group(sentence_group)
            return sentence_group, bucket, "explicit"

        # 2. Нет явной лексики -> fallback на polarity_hint аспекта
        aspect_rule = self.lexicon.get_aspect_rule(aspect_code)
        if aspect_rule and aspect_rule.polarity_hint:
            polarity_hint = aspect_rule.polarity_hint
        else:
            polarity_hint = "neutral"  # максимально безопасно, если ничего не знаем

        # Приводим polarity_hint к детальной группе:
        #   "positive" -> "positive_soft"
        #   "negative" -> "negative_soft"
        #   "neutral"  -> "neutral"
        if polarity_hint == "positive":
            final_group = "positive_soft"
        elif polarity_hint == "negative":
            final_group = "negative_soft"
        else:
            final_group = "neutral"

        bucket = collapse_sentiment_group(final_group)
        return final_group, bucket, "default_aspect_hint"
