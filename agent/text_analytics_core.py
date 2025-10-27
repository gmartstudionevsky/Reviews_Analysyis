# agent/text_analytics_core.py
#
# Назначение:
# - Нормализовать отзывы (язык, текст, оценка)
# - Разметить каждый отзыв по темам/подтемам/аспектам
# - Определить тональность внутри этих тем
#
# Это ядро семантики. На нем будут строиться:
# - Картина качества и тем (сводная таблица по категориям)
# - Динамика и вклад недели
# - Причины проблем с заселением
# - Цитаты гостей
#
# Совместимо с Python 3.11

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import re
import math
import json
import hashlib
import unicodedata
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass


###############################################################################
# 0. Нормализация языка и текста
###############################################################################

# Мы не можем полагаться на "Код языка" из таблицы,
# потому что он может быть нестабильным (ru vs uk vs mix, en-US vs en, и т.д.).
# Подход:
# - нормализуем код до базового вида ('ru','en','tr','ar','zh','other')
# - если не уверены, пытаемся угадать по алфавиту/символам
# - дальше работаем с text_ru (будет русская версия текста)
#   Для не-русских языков пока текст не переводим реально, но оставляем интерфейс,
#   чтобы потом встроить перевод и улучшить качество классификации.
#

CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
CJK_RE = re.compile(r"[\u4E00-\u9FFF]")  # базовый китайский диапазон
TURKISH_HINTS = re.compile(r"[ıİğĞşŞçÇöÖüÜ]")

def normalize_lang(raw_lang: Optional[str], text: str) -> str:
    """
    Приводим код языка к одной из категорий: ru, en, tr, ar, zh, other.
    Используем эвристику по коду и по тексту.
    """
    if raw_lang is None:
        raw_lang = ""
    raw_lang_low = raw_lang.lower()

    # явные матчеры
    if raw_lang_low.startswith("ru") or raw_lang_low.startswith("uk") or raw_lang_low.startswith("be"):
        return "ru"
    if raw_lang_low.startswith("en"):
        return "en"
    if raw_lang_low.startswith("tr"):
        return "tr"
    if raw_lang_low.startswith("ar"):
        return "ar"
    if raw_lang_low.startswith("zh") or raw_lang_low.startswith("cn") or raw_lang_low.startswith("ch"):
        return "zh"

    # эвристика по символам
    if CYRILLIC_RE.search(text):
        return "ru"
    if ARABIC_RE.search(text):
        return "ar"
    if CJK_RE.search(text):
        return "zh"
    if TURKISH_HINTS.search(text):
        return "tr"

    # латиница по умолчанию -> en (лучше чем other)
    if re.search(r"[A-Za-z]", text):
        return "en"

    return "other"


def translate_to_ru(text: str, lang: str) -> str:
    """
    Псевдо-перевод частых сервисных слов в «условно русский словарь».
    Это НЕ машинный перевод и он специально очень узкий.

    Логика:
    - Если в тексте уже есть кириллица — возвращаем как есть.
    - Если язык не ru — делаем лексическую подмену типичных отельных слов
      (wifi → вайфай, breakfast → завтрак, noisy → шумно и т.д.).
    Это повышает шанс, что русские паттерны сработают одинаково для
    англ/тур/прочих отзывов.
    """
    if not text:
        return ""

    # если кириллица уже есть — просто вернуть
    if lang == "ru" or CYRILLIC_RE.search(text):
        return text

    # словарь частых соответствий
    lex_map = {
        # английский
        r"\bwifi\b": "вайфай",
        r"\bwi[- ]?fi\b": "вайфай",
        r"\binternet\b": "интернет",
        r"\bbreakfast\b": "завтрак",
        r"\bstaff\b": "персонал",
        r"\breception\b": "ресепшен",
        r"\bcheck[- ]?in\b": "заселение",
        r"\bcheckin\b": "заселение",
        r"\bcheck[- ]?out\b": "выезд",
        r"\broom\b": "номер",
        r"\bclean\b": "чисто",
        r"\bdirty\b": "грязно",
        r"\bnoise[y]?\b": "шумно",
        r"\bnoisy\b": "шумно",
        r"\blocation\b": "расположение",
        r"\bexpensive\b": "дорого",
        r"\bprice\b": "цена",
        r"\bvalue for money\b": "соотношение цена качество",
        r"\bbad value\b": "не соответствует цене",
        r"\bair ?conditioning\b": "кондиционер",
        r"\bheating\b": "отопление",

        # турецкий
        r"\bwifi\b": "вайфай",
        r"\binternet\b": "интернет",
        r"\bkahvalt[ıi]\b": "завтрак",
        r"\bpersonel\b": "персонал",
        r"\btemiz\b": "чисто",
        r"\bkirli\b": "грязно",
        r"\bgürült[üu]\b": "шумно",
        r"\bkonum\b": "расположение",
        r"\bpahalı\b": "дорого",
    }

    out = text
    for pat, repl in lex_map.items():
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)

    return out

# Нормализация текста

WI_FI_UNIFY = re.compile(r"\b(wi[\s\-]?[fi]|вай[\s\-]?[фа]й)\b", re.IGNORECASE)
MULTISPACE = re.compile(r"\s+")
URL = re.compile(r"https?://\S+|www\.\S+")
EMAIL = re.compile(r"\b[\w.\-+%]+@[\w.\-]+\.[A-Za-z]{2,}\b")
PHONE = re.compile(r"\+?\d[\d\-\s()]{6,}\d")
HTML = re.compile(r"<[^>]+>")

def normalize_text_basic(text: str, lang: str) -> str:
    """
    ОЧИСТКА:
    - убираем html-теги, url, e-mail, телефоны;
    - нормализуем юникод (NFKC);
    - приводим к lower;
    - унифицируем написание 'wi-fi'/'wifi'/'вай фай' -> 'wi-fi';
    - схлопываем лишние пробелы.
    Это делается ДО анализа тональности и тем.
    """
    if not text:
        return ""
    t = str(text)
    t = unicodedata.normalize("NFKC", t)
    t = HTML.sub(" ", t)
    t = URL.sub(" ", t)
    t = EMAIL.sub(" ", t)
    t = PHONE.sub(" ", t)
    t = t.lower()
    t = WI_FI_UNIFY.sub("wi-fi", t)
    t = MULTISPACE.sub(" ", t).strip()
    return t


###############################################################################
# 1. Нормализация рейтинга и базовая тональность
###############################################################################

def normalize_rating_to_10(raw_rating: Any) -> Optional[float]:
    """
    Приводим рейтинг отзыва к 10-балльной шкале.
    Предполагаем, что:
    - если оценка <= 5 -> это пятибалльная шкала -> умножаем на 2
    - если оценка <= 10 -> уже десятибалльная -> оставляем
    - если оценка > 10 и <= 100 -> проценты, делим на 10
    - иначе None

    Возвращает float или None.
    """
    if raw_rating is None:
        return None
    try:
        r = float(str(raw_rating).replace(",", "."))
    except ValueError:
        return None

    if r <= 0:
        return None
    if r <= 5:      # предположим 1..5
        return r * 2.0
    if r <= 10:     # предположим 1..10
        return r
    if r <= 100:    # проценты
        return r / 10.0

    # что-то странное
    return None


# Словари слов для тональности.
# Это упрощённый эвристический sentiment. Мы будем использовать его
# в качестве дополнительного сигнала при оценке негатива внутри подтем.
#
# В перспективе это место, куда можно воткнуть ML-модель:
# - например, дообучить классификатор тональности по размеченным отзывам.
#
###############################################################################
# Тональность: расширенные словари
###############################################################################

# Ярко выраженная похвала
POSITIVE_WORDS_STRONG = {
    "ru": [
        r"\bидеальн", r"\bпревосходн", r"\bпотрясающе\b", r"\bвеликолепн", r"\bшикарн",
        r"\bсупер\b", r"\bлучший опыт\b", r"\bлучшее место\b", r"\bочень понравил", r"\bв восторге\b",
        r"\bобожаю\b", r"\bверн[уё]мся обязательно\b", r"\bоднозначно рекомендую\b",
        r"\bвсё просто отлично\b", r"\bбезупречн", r"\bчисто идеально\b",
    ],
    "en": [
        r"\bamazing\b", r"\bawesome\b", r"\bperfect\b", r"\bflawless\b", r"\bexceptional\b",
        r"\bexcellent\b", r"\boutstanding\b", r"\bloved it\b", r"\bi loved\b",
        r"\bhighly recommend\b", r"\bdefinitely recommend\b", r"\bwe will come back\b",
        r"\bspotless\b", r"\bimmaculate\b", r"\bfantastic\b",
    ],
    "tr": [
        r"\bharika\b", r"\bmükemmel\b", r"\bkusursuz\b", r"\bşahane\b",
        r"\bçok beğendik\b", r"\bçok memnun kaldık\b", r"\bkesinlikle tavsiye ederim\b",
        r"\btekrar geleceğiz\b",
    ],
    "ar": [
        r"\bرائع\b", r"\bممتاز\b", r"\bمذهل\b", r"\bمثالي\b", r"\bافضل تجربة\b",
        r"\bسأعود بالتأكيد\b", r"\bأنصح بشدة\b",
    ],
    "zh": [
        r"非常好", r"太棒了", r"完美", r"极好", r"超赞", r"特别满意", r"非常满意", r"强烈推荐",
        r"一定会再来", r"无可挑剔",
    ],
}

# Мягкий позитив, спокойное одобрение
POSITIVE_WORDS_SOFT = {
    "ru": [
        r"\bхорошо\b", r"\bочень хорошо\b", r"\bдоволен\b", r"\bдовольн", r"\bприятно\b",
        r"\bвсё ок\b", r"\bвсе ок\b", r"\bвсё было ок\b", r"\bв целом понравил", r"\bприятный опыт\b",
        r"\bчисто\b", r"\bуютн", r"\bкомфортн", r"\bудобн", r"\bвежлив", r"\bдоброжелательн",
        r"\bрадушн", r"\bдружелюб", r"\bгостеприимн", r"\bприняли хорошо\b",
        r"\bбыстро заселили\b", r"\bбыстро поселили\b",
    ],
    "en": [
        r"\bgood\b", r"\bvery good\b", r"\bnice\b", r"\bpleasant\b", r"\bcomfortable\b",
        r"\bcozy\b", r"\bclean\b", r"\bfriendly staff\b", r"\bpolite staff\b", r"\bhelpful staff\b",
        r"\bwelcoming\b", r"\bquick check[- ]?in\b", r"\bfast check[- ]?in\b",
        r"\bno issues\b", r"\bno problems\b",
    ],
    "tr": [
        r"\bi̇yi\b", r"\bçok iyi\b", r"\bgayet iyi\b", r"\brahat\b", r"\btemiz\b",
        r"\bgüler yüzlü\b", r"\byardımsever\b", r"\bmisafirperver\b", r"\bhızlı check[- ]?in\b",
        r"\bsorun yoktu\b",
    ],
    "ar": [
        r"\bجيد\b", r"\bجيد جدًا\b", r"\bمرتاح\b", r"\bنظيف\b", r"\bمريح\b",
        r"\bخدمة لطيفة\b", r"\bاستقبال جيد\b", r"\bلا توجد مشكلة\b",
    ],
    "zh": [
        r"很好", r"不错", r"满意", r"挺好", r"干净", r"舒适", r"服务很好", r"员工很友好", r"入住很快",
        r"没问题", r"一切都可以", r"还可以", r"可以接受",
    ],
}

# Умеренный негатив / претензии, без жёсткой эмоциональной лексики
NEGATIVE_WORDS_SOFT = {
    "ru": [
        r"\bне очень\b", r"\bмогло бы быть лучше\b", r"\bсредне\b", r"\bтак себе\b",
        r"\bожидал(и)? лучше\b", r"\bразочаров", r"\bне впечатлил", r"\bесть недочеты\b",
        r"\bнемного грязн", r"\bчуть грязн", r"\bшумновато\b", r"\bслегка шумно\b",
        r"\bнекомфортно\b", r"\bнеудобно\b", r"\bнеудобн(ая|ый|о)\b",
        r"\bждали д(олг|олго)\b", r"\bдолго ждали\b", r"\bподождать пришлось\b",
        r"\bпроблемы с заселением\b", r"\bне сразу заселили\b", r"\bкомната ещё не была готова\b",
    ],
    "en": [
        r"\bnot great\b", r"\bnot very good\b", r"\bcould be better\b", r"\baverage\b",
        r"\bdisappoint", r"\bunderwhelming\b", r"\ba bit dirty\b", r"\ba little dirty\b",
        r"\bnoisy\b", r"\bquite noisy\b", r"\ba bit noisy\b",
        r"\buncomfortable\b", r"\binconvenient\b",
        r"\bhad to wait\b", r"\bwaited a while\b", r"\broom not ready\b",
    ],
    "tr": [
        r"\bok değildi\b", r"\bo kadar iyi değil\b", r"\bordinerd(i|i)\b", r"\bortalama\b",
        r"\bbiraz kirli\b", r"\bbiraz gürültülü\b", r"\bbiraz rahatsız\b",
        r"\bbeklemek zorunda kaldık\b", r"\boda hazır değildi\b",
        r"\bhayal kırıklığı\b",
    ],
    "ar": [
        r"\bليس رائع\b", r"\bعادي\b", r"\bمتوسط\b", r"\bمخيب\b",
        r"\bقليل الوسخ\b", r"\bقليل الضجيج\b", r"\bغير مريح\b",
        r"\bانتظرنا\b", r"\bلم تكن الغرفة جاهزة\b",
    ],
    "zh": [
        r"一般", r"有点失望", r"不算好", r"有点脏", r"有点吵", r"有点不舒服", r"等了很久", r"房间还没准备好",
        r"有点麻烦", r"不是很方便",
    ],
}

# Сильный негатив, жёсткая эмоциональная лексика
NEGATIVE_WORDS_STRONG = {
    "ru": [
        r"\bужасн", r"\bкошмар", r"\bкатастроф", r"\bотвратител", r"\bмерзко\b", r"\bгрязь\b",
        r"\bгрязно\b", r"\bвонял", r"\bвонь\b", r"\bплесень\b", r"\bплесн[ью]\b",
        r"\bгромко\b", r"\bочень шумно\b", r"\bневыносимо\b", r"\bневозможно спать\b",
        r"\bобман\b", r"\bскрыт(ые|ые) платеж", r"\bнадули\b", r"\bунизительно\b",
        r"\bникому не советую\b", r"\bникому не рекомендую\b",
        r"\bникогда больше\b", r"\bбольше не приеду\b",
        r"\bперсонал хамил\b", r"\bгрубый персонал\b", r"\bхамство\b", r"\bгрубость\b",
    ],
    "en": [
        r"\bterrible\b", r"\bawful\b", r"\bdisgusting\b", r"\bfilthy\b", r"\bdirty\b",
        r"\bsmelled bad\b", r"\bstinky\b", r"\bmold\b", r"\bmould\b",
        r"\bscam\b", r"\brip[- ]?off\b", r"\bfraud\b", r"\bhidden fees\b",
        r"\bnever again\b", r"\bwill not come back\b", r"\bnot recommend\b",
        r"\brude staff\b", r"\bvery rude\b", r"\bextremely rude\b", r"\binsulting\b",
        r"\bimpossible to sleep\b", r"\bno sleep\b", r"\bso noisy\b", r"\bextremely noisy\b",
    ],
    "tr": [
        r"\bberbat\b", r"\brezalet\b", r"\biğrenç\b", r"\bçok pis\b", r"\bleş gibi kokuyordu\b",
        r"\bküf\b", r"\baldatıldık\b", r"\bdolandırıcılık\b", r"\bgizli ücretler\b",
        r"\bbir daha asla\b", r"\btavsiye etmiyorum\b",
        r"\bçok kaba\b", r"\bhaşin\b", r"\başağılayıcı\b",
        r"\buyuyamadık\b", r"\büstümüzde gürültü\b", r"\bçok gürültülü\b",
    ],
    "ar": [
        r"\bسيئ جدًا\b", r"\bفظيع\b", r"\bقذر جدًا\b", r"\bمقرف\b", r"\bرائحة كريهة\b",
        r"\bاحتيال\b", r"\bنصب\b", r"\bرسوم مخفية\b",
        r"\bأبدًا مرة أخرى\b", r"\bلا أنصح\b",
        r"\bوقحين جدًا\b", r"\bغير محترمين\b", r"\bأهانونا\b",
        r"\bمستحيل النوم\b", r"\bضجيج لا يحتمل\b",
    ],
    "zh": [
        r"太糟糕", r"很糟", r"恶心", r"肮脏", r"非常脏", r"有霉味", r"发霉", r"特别臭",
        r"被骗", r"坑钱", r"乱收费", r"隐形消费",
        r"绝不会再来", r"不推荐", r"服务员很粗鲁", r"态度很差",
        r"吵得没法睡", r"完全睡不着", r"太吵了", r"受不了",
    ],
}

# Нейтральный / смешанный фидбек:
# "нормально", "в целом окей", "приемлемо", "ок для одной ночи", "сойдёт", "acceptable", "fine", "okish".
# Часто в таких отзывах рейтинг 7-8/10, тональность ближе к нейтральной/слегка положительной.
NEUTRAL_WORDS = {
    "ru": [
        r"\bнормально\b", r"\bнорм\b", r"\bнормал(ьно|ьный)\b", r"\bв целом норм\b",
        r"\bтерпимо\b", r"\bсойдёт\b", r"\bсойдет\b", r"\bприемлемо\b", r"\bвполне сносно\b",
        r"\bбез особых проблем\b", r"\bничего страшного\b", r"\bничего критичного\b",
        r"\bдля одной ночи ок\b", r"\bдля одной ночи нормально\b",
    ],
    "en": [
        r"\bok\b", r"\bokay\b", r"\bfine\b", r"\ball right\b", r"\bacceptable\b", r"\bdecent\b",
        r"\bit was ok\b", r"\bit was fine\b", r"\bnothing special\b", r"\bnothing crazy\b",
        r"\bgood for one night\b", r"\bfor one night it's fine\b",
    ],
    "tr": [
        r"\bidare eder\b", r"\bfena değil\b", r"\bkötü değil\b",
        r"\btamamdır\b", r"\bokeydi\b", r"\baynen\b",
        r"\bbir gece için yeterli\b",
    ],
    "ar": [
        r"\bلا بأس\b", r"\bمقبول\b", r"\bعلى ما يرام\b", r"\bجيد بشكل عام\b",
        r"\bكافي لليلة واحدة\b",
    ],
    "zh": [
        r"还行", r"可以", r"凑合", r"勉强可以", r"没什么大问题", r"总体可以", r"一般般", r"还好",
        r"住一晚还行",
    ],
}

def _get_lexicons_for_lang(lang: str) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    """
    Возвращает кортеж списков regex:
    (strong_neg, soft_neg, strong_pos, soft_pos, neutrals)
    с fallback на английский, если языка нет явно.
    """
    key = lang if lang in NEGATIVE_WORDS_STRONG else "en"

    strong_neg = NEGATIVE_WORDS_STRONG.get(key, [])
    soft_neg   = NEGATIVE_WORDS_SOFT.get(key, [])
    strong_pos = POSITIVE_WORDS_STRONG.get(key, [])
    soft_pos   = POSITIVE_WORDS_SOFT.get(key, [])
    neutrals   = NEUTRAL_WORDS.get(key, [])

    return (strong_neg, soft_neg, strong_pos, soft_pos, neutrals)


# Негаторы (языковые частицы отрицания)
RU_NEG = r"(не|без)\s+"
EN_NEG = r"(not|no|without|never)\s+"
TR_NEG = r"(değil|yok|olm[aı]yor)\s+"

def _negated(text: str, lang: str, term_regex: str) -> bool:
    """
    True, если перед term_regex стоит отрицание.
    Т.е. 'не грязно' не считаем грязью, 'not dirty' не считаем грязью.
    """
    if lang == "ru":
        return re.search(rf"{RU_NEG}\w{{0,3}}(?:{term_regex})", text, flags=re.IGNORECASE) is not None
    if lang == "tr":
        return re.search(rf"{TR_NEG}\w{{0,3}}(?:{term_regex})", text, flags=re.IGNORECASE) is not None
    # fallback → английский
    return re.search(rf"{EN_NEG}\w{{0,3}}(?:{term_regex})", text, flags=re.IGNORECASE) is not None


def _match_any(patterns: List[str], text: str) -> bool:
    for pat in patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True
    return False


def detect_sentiment_label(text: str, lang: str, fallback_rating10: Optional[float]) -> str:
    """
    Общая тональность отзыва.
    Приоритет:
    1. сильный негатив (учитывая negation — 'не грязно' не считается)
    2. мягкий негатив
    3. сильный позитив (если 'не хорошо' → трактуем как мягкий негатив)
    4. мягкий позитив
    5. нейтральные маркеры
    6. fallback по рейтингу /10
    """
    (strong_neg, soft_neg, strong_pos, soft_pos, neutrals) = _get_lexicons_for_lang(lang)

    # 1. сильный негатив
    for rgx in strong_neg:
        if re.search(rgx, text, flags=re.IGNORECASE) and not _negated(text, lang, rgx):
            return "neg"

    # 2. мягкий негатив
    for rgx in soft_neg:
        if re.search(rgx, text, flags=re.IGNORECASE) and not _negated(text, lang, rgx):
            return "neg"

    # 3. сильный позитив
    for rgx in strong_pos:
        if re.search(rgx, text, flags=re.IGNORECASE):
            # если "не отлично" -> это жалоба
            if _negated(text, lang, rgx):
                return "neg"
            return "pos"

    # 4. мягкий позитив
    for rgx in soft_pos:
        if re.search(rgx, text, flags=re.IGNORECASE):
            if _negated(text, lang, rgx):
                return "neg"
            return "pos"

    # 5. явная нейтральная лексика
    for rgx in neutrals:
        if re.search(rgx, text, flags=re.IGNORECASE):
            return "neu"

    # 6. fallback по рейтингу
    if fallback_rating10 is not None:
        if fallback_rating10 >= 9.0:
            return "pos"
        if fallback_rating10 <= 6.0:
            return "neg"

    return "neu"


def detect_subtopic_sentiment(text: str, lang: str, subtopic_patterns: Optional[List[str]] = None) -> str:
    """
    Тональность по конкретной подтеме.
    Если subtopic_patterns даны, смотрим окно ±40 символов вокруг каждого совпадения,
    иначе смотрим весь текст.
    """
    if not text:
        return "neu"

    # соберём кандидаты-фрагменты
    candidates: List[str] = [text]
    if subtopic_patterns:
        spans: List[str] = []
        for pat in subtopic_patterns:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                s, e = m.span()
                spans.append(text[max(0, s - 40): min(len(text), e + 40)])
        if spans:
            candidates = spans

    (strong_neg, soft_neg, strong_pos, soft_pos, neutrals) = _get_lexicons_for_lang(lang)

    for chunk in candidates:
        # сильный негатив
        for rgx in strong_neg:
            if re.search(rgx, chunk, flags=re.IGNORECASE) and not _negated(chunk, lang, rgx):
                return "neg"
        # мягкий негатив
        for rgx in soft_neg:
            if re.search(rgx, chunk, flags=re.IGNORECASE) and not _negated(chunk, lang, rgx):
                return "neg"
        # сильный позитив
        for rgx in strong_pos:
            if re.search(rgx, chunk, flags=re.IGNORECASE):
                if _negated(chunk, lang, rgx):
                    return "neg"
                return "pos"
        # мягкий позитив
        for rgx in soft_pos:
            if re.search(rgx, chunk, flags=re.IGNORECASE):
                if _negated(chunk, lang, rgx):
                    return "neg"
                return "pos"
        # нейтрально
        for rgx in neutrals:
            if re.search(rgx, chunk, flags=re.IGNORECASE):
                return "neu"

    return "neu"

###############################################################################
# 2. Схема тем (Категория -> Подтемы -> Аспекты)
###############################################################################

# Объяснение:
# TOPIC_SCHEMA описывает:
# - ключ категории (например, "cleanliness")
#   - "display": Человеческое название категории для отчёта
#   - "subtopics": словарь подтем
#       - каждая подтема имеет:
#           "display": человекочитаемое имя
#           "patterns": словарь {lang_code: [regex,...]} для выявления упоминаний
#           "aspects": список ярлыков причин/контекстов (необязательно)
#
# Пример:
#   "checkin_speed": гость долго ждал заселения
#       aspects: ["staff_slow", "room_not_ready", "tech_access_issue"]
#
# В дальнейшем aspects помогут нам текстом объяснить:
# "заселение страдало не из-за персонала, а из-за того, что номер был не готов"
#
# Сейчас дадим базовые паттерны. Потом мы можем расширять массивы regex,
# не ломая остальной код.

TOPIC_SCHEMA: Dict[str, Dict[str, Any]] = {
    "staff_spir": {
        "display": "Персонал",
        "subtopics": {
            "staff_attitude": {
                "display": "Отношение и вежливость",
                "patterns": {
                    "ru": [
                        r"\bвежлив", r"\bдоброжелательн", r"\bдружелюб", r"\bприветлив",
                        r"\bрадушн", r"\bтепло встретил", r"\bотзывчив", r"\bс улыбкой",
                        r"\bхамил", r"\bхамство", r"\bнагруб", r"\bгруб(о|ые)",
                        r"\bнеприветлив", r"\bнедружелюб", r"\bразговаривал[аи]? свысока",
                    ],
                    "en": [
                        r"\bfriendly staff\b", r"\bvery friendly\b", r"\bwelcoming\b",
                        r"\bpolite\b", r"\bkind\b",
                        r"\brude staff\b", r"\bunfriendly\b", r"\bimpolite\b",
                        r"\bdisrespectful\b", r"\btreated us badly\b",
                    ],
                    "tr": [
                        r"\bgüler yüzlü\b", r"\bnazik\b", r"\bkibar\b", r"\bsıcak karşıladılar\b",
                        r"\bçok kaba\b", r"\bsaygısız\b", r"\bters davrandılar\b",
                    ],
                    "ar": [
                        r"\bموظفين لطيفين\b", r"\bاستقبال دافئ\b", r"\bتعامل محترم\b", r"\bابتسامة\b",
                        r"\bموظفين وقحين\b", r"\bسيء التعامل\b", r"\bغير محترمين\b",
                    ],
                    "zh": [
                        r"服务很友好", r"前台很热情", r"态度很好", r"很有礼貌",
                        r"态度很差", r"服务很差", r"很不耐烦", r"不礼貌", r"很凶",
                    ],
                },
                "aspects": [
                    "spir_friendly", "spir_polite",
                    "spir_rude", "spir_unrespectful",
                ],
            },

            "staff_helpfulness": {
                "display": "Помощь и решение вопросов",
                "patterns": {
                    "ru": [
                        r"\bпомог(ли|ли нам)\b", r"\bрешил[аи]? вопрос\b", r"\bрешили проблему\b",
                        r"\bвсё объяснил[аи]?\b", r"\bвсё подсказал[аи]?\b", r"\bподсказали куда\b",
                        r"\bдали рекомендации\b", r"\bбыстро отреагировал[аи]?\b",
                        r"\bне помогли\b", r"\bничего не сделали\b", r"\bне решили\b",
                        r"\bсказали это не наша проблема\b", r"\bпроигнорировали\b",
                    ],
                    "en": [
                        r"\bhelpful\b", r"\bvery helpful\b", r"\bsolved the issue\b",
                        r"\bfixed it quickly\b", r"\bassisted us\b", r"\bgave recommendations\b",
                        r"\bexplained everything\b",
                        r"\bnot helpful\b", r"\bunhelpful\b", r"\bignored us\b",
                        r"\bdidn't solve\b", r"\bno assistance\b", r"\bnot their problem\b",
                    ],
                    "tr": [
                        r"\byardımcı oldular\b", r"\bhemen çözdüler\b", r"\bbize anlattılar\b", r"\byönlendirdiler\b",
                        r"\byardımcı olmadılar\b", r"\bilgilenmediler\b", r"\bsorunu çözmediler\b",
                    ],
                    "ar": [
                        r"\bساعدونا\b", r"\bحلّوا المشكلة\b", r"\bشرحوا كل شيء\b", r"\bاستجابوا بسرعة\b",
                        r"\bلم يساعدونا\b", r"\bتجاهلونا\b", r"\bلم يحلوا المشكلة\b", r"\bقالوا ليست مشكلتنا\b",
                    ],
                    "zh": [
                        r"很帮忙", r"很乐于助人", r"马上处理", r"马上解决", r"给我们解释", r"给了建议",
                        r"不帮忙", r"不理我们", r"没解决", r"让我们自己处理",
                    ],
                },
                "aspects": [
                    "spir_helpful_fast", "spir_problem_solved", "spir_info_clear",
                    "spir_unhelpful", "spir_problem_ignored", "spir_info_confusing",
                ],
            },

            "staff_speed": {
                "display": "Оперативность и скорость реакции",
                "patterns": {
                    "ru": [
                        r"\bбыстро заселили\b", r"\bмоментально заселили\b", r"\bоформили быстро\b",
                        r"\bреагируют быстро\b", r"\bпришли сразу\b", r"\bоперативно\b",
                        r"\bждали долго\b", r"\bпришлось долго ждать\b", r"\bникого не было на ресепшен",
                        r"\bне могли дозвониться\b", r"\bне брали трубку\b", r"\bдолго оформляли\b",
                    ],
                    "en": [
                        r"\bquick check[- ]?in\b", r"\bfast check[- ]?in\b", r"\bresponded immediately\b",
                        r"\bthey came right away\b", r"\bhandled it quickly\b",
                        r"\bhad to wait a long time\b", r"\bno one answered\b",
                        r"\bnobody at the desk\b", r"\bslow check[- ]?in\b", r"\btook too long\b",
                    ],
                    "tr": [
                        r"\bhızlı check[- ]?in\b", r"\bçok hızlı ilgilendiler\b", r"\bhemen geldiler\b", r"\banında yardımcı oldular\b",
                        r"\bçok bekledik\b", r"\bresepsiyonda kimse yoktu\b", r"\btelefon açmadılar\b", r"\bgeç cevap verdiler\b",
                    ],
                    "ar": [
                        r"\bتسجيل دخول سريع\b", r"\bاستجابوا فورًا\b", r"\bجاءوا مباشرة\b", r"\bسريع جدًا\b",
                        r"\bانتظرنا كثيرًا\b", r"\bلم يرد أحد\b", r"\bلا أحد في الاستقبال\b", r"\bبطيء جدًا\b",
                    ],
                    "zh": [
                        r"办理入住很快", r"马上处理", r"很快就来了", r"反应很快",
                        r"等了很久", r"前台没人", r"没人接电话", r"太慢了", r"入住很慢",
                    ],
                },
                "aspects": [
                    "spir_helpful_fast", "spir_fast_response",
                    "spir_slow_response", "spir_absent", "spir_no_answer",
                ],
            },

            "staff_professionalism": {
                "display": "Профессионализм и компетентность",
                "patterns": {
                    "ru": [
                        r"\bпрофессионал", r"\bкомпетентн", r"\bвсё чётко объяснил", r"\bвсё грамотно объяснила",
                        r"\bвсё прозрачно\b", r"\bоформили документы\b", r"\bдали все чеки\b",
                        r"\bнекомпетентн", r"\bне знают\b", r"\bбардак с документами\b",
                        r"\bне смогли объяснить оплату\b", r"\bошиблись в брон[иь]\b",
                        r"\bошибка в сч(е|ё)те\b", r"\bпутаница с оплатой\b", r"\bнепрозрачно\b",
                    ],
                    "en": [
                        r"\bprofessional\b", r"\bvery professional\b", r"\bknowledgeable\b", r"\bclear explanation\b",
                        r"\btransparent\b", r"\bsorted all paperwork\b", r"\bgave invoice\b",
                        r"\bunprofessional\b", r"\bdidn't know\b", r"\bconfused about payment\b",
                        r"\bmessed up reservation\b", r"\bwrong charge\b", r"\bbilling mistake\b",
                    ],
                    "tr": [
                        r"\bprofesyonel\b", r"\bçok profesyonel\b", r"\bişini biliyor\b", r"\baçıkça anlattı\b",
                        r"\bfaturayı düzgün verdiler\b",
                        r"\bprofesyonel değildi\b", r"\bbilmiyorlardı\b", r"\bödeme konusunda karışıklık\b",
                        r"\byanlış ücret\b", r"\brezervasyonu karıştırdılar\b",
                    ],
                    "ar": [
                        r"\bمحترفين\b", r"\bيعرفون شغلهم\b", r"\bشرح واضح\b", r"\bكل شيء كان واضح بالدفع\b",
                        r"\bأعطونا كل الفواتير\b",
                        r"\bغير محترفين\b", r"\bمش فاهمين الإجراءات\b", r"\bخطأ في الحجز\b",
                        r"\bخطأ في الفاتورة\b", r"\bمش واضح بالدفع\b",
                    ],
                    "zh": [
                        r"很专业", r"非常专业", r"解释很清楚", r"流程很清楚", r"收费很透明", r"单据都给了",
                        r"不专业", r"搞不清楚", r"解释不清楚", r"收费不明", r"账单有问题", r"搞错预订",
                    ],
                },
                "aspects": [
                    "spir_professional", "spir_info_clear", "spir_payment_clear",
                    "spir_unprofessional", "spir_info_confusing", "spir_payment_issue", "spir_booking_mistake",
                ],
            },

            "staff_availability": {
                "display": "Доступность персонала",
                "patterns": {
                    "ru": [
                        r"\bна связи 24\b", r"\bкруглосуточно помогали\b", r"\bдаже ночью помогли\b",
                        r"\bответили ночью\b", r"\bбыли всегда доступны\b",
                        r"\bна ресепшен[е]? никого\b", r"\bникого не было на стойке\b", r"\bне дозвониться ночью\b",
                        r"\bникто не приш[её]л\b", r"\bресепшн закрыт ночью\b",
                    ],
                    "en": [
                        r"\b24/7\b", r"\balways available\b", r"\beven at night they helped\b",
                        r"\bnight staff was helpful\b", r"\banswered phone at night\b",
                        r"\bno one at the desk\b", r"\bnobody at reception\b", r"\bno answer at night\b",
                        r"\bcouldn't reach anyone\b", r"\breception closed at night\b",
                    ],
                    "tr": [
                        r"\b24 saat ulaşılabilir\b", r"\bgecede bile yardımcı oldular\b", r"\bgece personeli çok yardımcı\b",
                        r"\bresepsiyonda kimse yoktu\b", r"\bgece kimse yoktu\b", r"\bgece kimse cevap vermedi\b",
                    ],
                    "ar": [
                        r"\bمتوفرين طول الوقت\b", r"\bحتى بالليل ساعدونا\b", r"\bردوا علينا في الليل\b",
                        r"\bمافي أحد بالاستقبال\b", r"\bبالليل ما حد يرد\b", r"\bمغلق بالليل\b", r"\bمافي دعم ليلي\b",
                    ],
                    "zh": [
                        r"24小时有人", r"半夜也有人帮忙", r"晚上也能联系到", r"夜班也很负责",
                        r"前台没人", r"晚上没人", r"打电话没人接", r"夜里没人管",
                    ],
                },
                "aspects": [
                    "spir_available", "spir_24h_support",
                    "spir_absent", "spir_no_answer", "spir_no_night_support",
                ],
            },

            "staff_communication": {
                "display": "Коммуникация и понятность объяснений",
                "patterns": {
                    "ru": [
                        r"\bвсё понятн[оы] объяснил", r"\bподробно рассказал", r"\bинструкции понятные\b",
                        r"\bобъяснили как зайти\b", r"\bобъяснили куда идти\b", r"\bвсё разжевали\b",
                        r"\bничего не объяснили\b", r"\bнепонятные инструкции\b", r"\bобъясняли как попало\b",
                        r"\bне смогли объяснить\b", r"\bя не понял\b", r"\bя не поняла\b", r"\bпутались\b",
                    ],
                    "en": [
                        r"\bclear instructions\b", r"\bexplained everything clearly\b", r"\beasy to understand\b",
                        r"\bcommunicated clearly\b", r"\bgood English\b", r"\bspoke English well\b",
                        r"\bhard to understand\b", r"\bunclear instructions\b", r"\bpoor communication\b",
                        r"\bnobody speaks English\b", r"\blanguage barrier\b", r"\bcouldn't explain\b",
                    ],
                    "tr": [
                        r"\bher şeyi açıkladılar\b", r"\btalimatlar çok netti\b", r"\bingilizce konuşabiliyorlardı\b",
                        r"\banlaşılması zordu\b", r"\btalimatlar net değildi\b", r"\bingilizce konuşmuyorlar\b",
                    ],
                    "ar": [
                        r"\bشرح واضح\b", r"\bفسروا كل شيء\b", r"\bالتعليمات كانت واضحة\b",
                        r"\bتعليمات غير واضحة\b", r"\bصعب نفهم\b", r"\bما يحكوا انجليزي\b", r"\bحاجز لغة\b",
                    ],
                    "zh": [
                        r"解释得很清楚", r"指示很清晰", r"英文很好", r"沟通很顺",
                        r"听不懂", r"解释不清楚", r"沟通不好", r"没有说明清楚", r"语言有问题",
                    ],
                },
                "aspects": [
                    "spir_info_clear", "spir_language_ok",
                    "spir_info_confusing", "spir_language_barrier",
                ],
            },
        },
    },

    "checkin_stay": {
        "display": "Заселение и проживание",
        "subtopics": {
            "checkin_speed": {
                "display": "Скорость заселения",
                "patterns": {
                    "ru": [
                        r"\bбыстро заселили\b", r"\bмоментально заселили\b", r"\bоформили быстро\b",
                        r"\bзаселили без задержек\b", r"\bчек-?ин занял (пару минут|минуту)\b",
                        r"\bждали долго\b", r"\bпришлось долго ждать\b", r"\bждали пока подготовят номер\b",
                        r"\bдолго оформляли\b", r"\bочередь на заселение\b",
                        r"\bне могли заселиться сразу\b",
                    ],
                    "en": [
                        r"\bquick check[- ]?in\b", r"\bfast check[- ]?in\b", r"\bcheck ?in was fast\b",
                        r"\bno wait\b", r"\bgot our room immediately\b",
                        r"\bhad to wait a long time\b", r"\blong wait to check in\b",
                        r"\bslow check[- ]?in\b", r"\bcheck[- ]?in took too long\b",
                        r"\bwaited forever\b", r"\broom wasn't ready so we had to wait\b",
                    ],
                    "tr": [
                        r"\bhızlı check[- ]?in\b", r"\bhemen yerleştirdiler\b", r"\bbeklemeden oda verdiler\b",
                        r"\bçok bekledik\b", r"\bcheck[- ]?in çok yavaştı\b",
                        r"\boda hazır değildi, beklemek zorunda kaldık\b",
                    ],
                    "ar": [
                        r"\bتسجيل دخول سريع\b", r"\bدخلنا فورًا\b", r"\bما انتظرنا\b",
                        r"\bانتظرنا كثيرًا\b", r"\bتسجيل الدخول كان بطيء\b",
                        r"\bاضطرينا ننتظر حتى يجهزوا الغرفة\b",
                    ],
                    "zh": [
                        r"办理入住很快", r"马上就办好", r"几乎不用等", r"很快给了房间",
                        r"等了很久才入住", r"入住很慢", r"房间还没准备好我们只能等", r"排队很久",
                    ],
                },
                "aspects": [
                    "checkin_fast", "no_wait_checkin",
                    "checkin_wait_long", "room_not_ready_delay",
                ],
            },
    
            "room_ready": {
                "display": "Готовность номера к заселению",
                "patterns": {
                    "ru": [
                        r"\bномер был готов\b", r"\bвсё готово к нашему приезду\b",
                        r"\bчисто при заселении\b", r"\bидеально чисто при заезде\b",
                        r"\bномер не был готов\b", r"\bв номере не убрано\b", r"\bгрязно при заселении\b",
                        r"\bпостель не сменили\b", r"\bмусор от прошлых гостей\b",
                        r"\bпопросили подождать пока уберут\b", r"\bк заселению номер не подготовили\b",
                    ],
                    "en": [
                        r"\broom was ready\b", r"\broom ready on arrival\b",
                        r"\bclean on arrival\b", r"\bspotless when we arrived\b",
                        r"\broom was not ready\b", r"\broom not prepared\b",
                        r"\bstill dirty when we arrived\b", r"\btrash from previous guest\b",
                        r"\bbed not changed\b", r"\bhad to wait for cleaning\b",
                    ],
                    "tr": [
                        r"\boda hazırdı\b", r"\bgirdiğimizde tertemizdi\b", r"\bhemen hazır odayı verdiler\b",
                        r"\boda hazır değildi\b", r"\boda temizlenmemişti\b",
                        r"\bönce temizlemeleri gerekti\b", r"\bönceki misafirin izleri vardı\b",
                    ],
                    "ar": [
                        r"\bالغرفة جاهزة\b", r"\bكل شي كان جاهز\b", r"\bنظيف وقت ما وصلنا\b",
                        r"\bالغرفة ما كانت جاهزة\b", r"\bلسا ما نظفوا\b",
                        r"\bبقي وسخ من الضيف السابق\b", r"\bاضطرينا نستنى لين ينظفوا\b",
                    ],
                    "zh": [
                        r"房间一开始就准备好了", r"一进来就很干净", r"房间很干净刚入住", r"床铺都弄好了",
                        r"房间还没准备好", r"房间没打扫", r"还有上个客人的垃圾", r"让我们等他们打扫",
                    ],
                },
                "aspects": [
                    "room_ready_on_arrival", "clean_on_arrival",
                    "room_not_ready", "dirty_on_arrival", "leftover_trash_previous_guest",
                ],
            },
    
            "access": {
                "display": "Доступ и вход в отель / номер",
                "patterns": {
                    "ru": [
                        r"\bлегко нашли вход\b", r"\bкод от двери сработал\b", r"\bдоступ в номер без проблем\b",
                        r"\bсложно найти вход\b", r"\bнепонятно куда заходить\b",
                        r"\bкод не сработал\b", r"\bзамок не открывался\b", r"\bкарта не работала\b",
                        r"\bне могли попасть внутрь\b",
                        r"\bлифт не работал\b", r"\bбез лифта очень тяжело\b", r"\bтащить чемоданы\b",
                    ],
                    "en": [
                        r"\beasy to get in\b", r"\baccess was simple\b", r"\bdoor code worked\b", r"\bentrance was clear\b",
                        r"\bhard to find the entrance\b", r"\bcouldn't get in\b", r"\bdoor code didn't work\b",
                        r"\bkey card didn't work\b", r"\block didn't open\b",
                        r"\blift not working\b", r"\bno elevator and we had luggage\b",
                    ],
                    "tr": [
                        r"\bgirişi bulmak kolaydı\b", r"\bkodu çalıştı\b", r"\bodaya girmek sorunsuzdu\b",
                        r"\bgirişi bulmak zordu\b", r"\bkapı kodu çalışmadı\b", r"\bkart çalışmadı\b",
                        r"\biçeri giremedik\b", r"\basansör yoktu\b", r"\bbavullarla çok zor oldu\b",
                    ],
                    "ar": [
                        r"\bالدخول سهل\b", r"\bالكود اشتغل من أول مرة\b", r"\bدخلنا بدون مشكلة\b",
                        r"\bصعب نلاقي المدخل\b", r"\bالكود ما اشتغل\b", r"\bالباب ما يفتح\b",
                        r"\bالكرت ما يشتغل\b", r"\bما قدرنا ندخل بسهولة\b",
                        r"\bما في مصعد\b", r"\bشنطنا ثقيلة\b",
                    ],
                    "zh": [
                        r"很容易进来", r"门码直接能用", r"进房间没问题",
                        r"入口很难找", r"门码不好用", r"门打不开", r"卡打不开门",
                        r"没有电梯", r"拿行李很麻烦",
                    ],
                },
                "aspects": [
                    "access_smooth", "door_code_worked",
                    "tech_access_issue", "entrance_hard_to_find", "no_elevator_baggage_issue",
                ],
            },
    
            "docs_payment": {
                "display": "Оплата, депозиты и документы",
                "patterns": {
                    "ru": [
                        r"\bвсё прозрачно по оплате\b", r"\bвсё объяснили по оплате\b",
                        r"\bдали чеки\b", r"\bдали отчетные документы\b", r"\bдепозит объяснили\b",
                        r"\bникаких скрытых платежей\b",
                        r"\bпопросили неожиданный депозит\b", r"\bзаблокировали деньги без объяснения\b",
                        r"\bне объяснили налоги\b", r"\bпутаница с оплатой\b", r"\bнепонятные доплаты\b",
                        r"\bошибка в сч(е|ё)те\b", r"\bнас попытались взять больше\b",
                    ],
                    "en": [
                        r"\btransparent payment\b", r"\bclear about charges\b", r"\bexplained the deposit\b",
                        r"\bgave invoice\b", r"\bno hidden fees\b",
                        r"\bhidden fees\b", r"\bextra charge\b", r"\bunexpected deposit\b",
                        r"\bblocked money without explaining\b", r"\bconfusing payment\b", r"\bwrong bill\b", r"\bovercharged\b",
                    ],
                    "tr": [
                        r"\bödeme şeffaftı\b", r"\bher şeyi anlattılar\b", r"\bfatura verdiler\b", r"\bekstra ücret yoktu\b",
                        r"\bgizli ücret\b", r"\bek depozito istediler\b", r"\bpara bloke edildi\b",
                        r"\bödeme karışıktı\b", r"\byanlış ücret\b",
                    ],
                    "ar": [
                        r"\bالدفع كان واضح\b", r"\bشرحوا الرسوم\b", r"\bأعطونا الفاتورة\b", r"\bمافي رسوم مخفية\b",
                        r"\bرسوم مخفية\b", r"\bطلبوا عربون بدون توضيح\b", r"\bسحبوا مبلغ إضافي\b",
                        r"\bفاتورة غلط\b", r"\bدفع غير واضح\b",
                    ],
                    "zh": [
                        r"收费很透明", r"提前说清楚押金", r"给了发票", r"没有乱收费",
                        r"乱收费", r"多收钱", r"要额外押金没说明", r"账单有问题", r"付款不清楚",
                    ],
                },
                "aspects": [
                    "payment_clear", "deposit_clear", "docs_provided", "no_hidden_fees",
                    "payment_confusing", "unexpected_charge", "hidden_fees",
                    "deposit_problematic", "billing_mistake", "overcharge",
                ],
            },
    
            "instructions": {
                "display": "Инструкции по заселению и проживанию",
                "patterns": {
                    "ru": [
                        r"\bвсё подробно объяснили\b", r"\bполучили понятные инструкции\b",
                        r"\bвсе инструкции заранее\b", r"\bпароль от ?wi[- ]?fi сразу дали\b",
                        r"\bнам всё разжевали\b",
                        r"\bникаких инструкций\b", r"\bнепонятно куда идти\b", r"\bнепонятно куда заходить\b",
                        r"\bкод прислали поздно\b", r"\bне сказали пароль от ?wi[- ]?fi\b",
                        r"\bразбираться пришлось самим\b",
                    ],
                    "en": [
                        r"\bclear instructions\b", r"\bthey sent all instructions\b",
                        r"\bself check[- ]?in was easy\b", r"\bthey explained how to enter\b",
                        r"\bwifi password (was|given) (right away|immediately)\b",
                        r"\bno instructions\b", r"\binstructions unclear\b", r"\bconfusing self check[- ]?in\b",
                        r"\bcode (arrived|sent) late\b", r"\bdidn't tell us the wifi password\b",
                        r"\bwe had to figure it out ourselves\b",
                    ],
                    "tr": [
                        r"\btalimatlar çok netti\b", r"\bself check[- ]?in kolaydı\b",
                        r"\bwifi şifresini hemen verdiler\b", r"\bnereden gireceğimizi anlattılar\b",
                        r"\btalimat yoktu\b", r"\btalimatlar net değildi\b",
                        r"\bkodu geç gönderdiler\b", r"\bwifi şifresini söylemediler\b",
                        r"\bkendimiz anlamak zorunda kaldık\b",
                    ],
                    "ar": [
                        r"\bالتعليمات كانت واضحة\b", r"\bرسلوا كل التعليمات\b",
                        r"\bدخول ذاتي كان سهل\b", r"\bعطونا كلمة سر الواي فاي مباشرة\b",
                        r"\bما في تعليمات واضحة\b", r"\bتعليمات مربكة\b", r"\bالكود تأخر\b",
                        r"\bما عطونا كلمة سر الواي فاي\b", r"\bاضطرينا نكتشف لوحدنا\b",
                    ],
                    "zh": [
                        r"指示很清楚", r"自助入住很简单", r"一开始就给了wifi密码", r"告诉我们怎么进门",
                        r"没有说明", r"指示不清楚", r"自助入住很复杂", r"门码很晚才发",
                        r"wifi密码没人说", r"只能自己摸索",
                    ],
                },
                "aspects": [
                    "instructions_clear", "self_checkin_easy", "wifi_info_given",
                    "instructions_confusing", "late_access_code", "wifi_info_missing", "had_to_figure_out",
                ],
            },
    
            "stay_support": {
                "display": "Поддержка во время проживания",
                "patterns": {
                    "ru": [
                        r"\bпринесли сразу\b", r"\bпринесли дополнительно\b", r"\bотреагировали за пару минут\b",
                        r"\bрешили сразу\b", r"\bмгновенно помогли\b", r"\bсразу поменяли\b",
                        r"\bникто не приш[её]л\b", r"\bпришлось просить несколько раз\b",
                        r"\bникакой реакции\b", r"\bигнорировали просьбы\b",
                        r"\bобещали и не сделали\b",
                    ],
                    "en": [
                        r"\bbrought it right away\b", r"\bfixed it immediately\b", r"\breplaced immediately\b",
                        r"\bgave us extra towels instantly\b", r"\bvery responsive during stay\b",
                        r"\bnobody came\b", r"\bwe had to ask multiple times\b",
                        r"\bno response during stay\b", r"\bignored our request\b", r"\bpromised but never did\b",
                    ],
                    "tr": [
                        r"\bhemen getirdiler\b", r"\banında hallettiler\b", r"\bhemen değiştirdiler\b",
                        r"\bisteğimizi hemen yaptılar\b",
                        r"\bkimse gelmedi\b", r"\bdefalarca istemek zorunda kaldık\b",
                        r"\btepki yoktu\b", r"\bsöz verdiler ama yapmadılar\b",
                    ],
                    "ar": [
                        r"\bجابوه فورًا\b", r"\bصلّحوه فورًا\b", r"\bبدلوا مباشرة\b",
                        r"\bاستجابوا بسرعة خلال الإقامة\b",
                        r"\bما حدا إجا\b", r"\bطلبنا أكتر من مرة\b",
                        r"\bما في استجابة\b", r"\bوعدوا وما عملوا شي\b",
                    ],
                    "zh": [
                        r"马上送来了", r"马上修好了", r"立刻换了", r"服务响应很快", r"我们要的东西很快就拿来了",
                        r"没有人来", r"说了好几次", r"没人理", r"他们答应了但没做",
                    ],
                },
                "aspects": [
                    "support_during_stay_good", "issue_fixed_immediately",
                    "support_during_stay_slow", "support_ignored", "promised_not_done",
                ],
            },
    
            "checkout": {
                "display": "Выезд",
                "patterns": {
                    "ru": [
                        r"\bвыезд удобный\b", r"\bвыписали быстро\b", r"\bчек-?аут занял минуту\b",
                        r"\bбез проблем с выездом\b",
                        r"\bпроблемы с выездом\b", r"\bвыписывали очень долго\b",
                        r"\bнам не вернули депозит сразу\b",
                        r"\bникого не было на ресепшен[е] когда выезжали\b",
                        r"\bне могли сдать ключ\b",
                    ],
                    "en": [
                        r"\beasy checkout\b", r"\bcheckout was fast\b", r"\bsmooth checkout\b",
                        r"\bleft with no problem\b",
                        r"\bcheckout was slow\b", r"\bproblem with checkout\b",
                        r"\bthey didn't return the deposit\b",
                        r"\bnobody at reception when we left\b", r"\bwe couldn't give the key back\b",
                    ],
                    "tr": [
                        r"\bçıkış kolaydı\b", r"\bcheck[- ]?out çok hızlıydı\b", r"\bsorunsuz ayrıldık\b",
                        r"\bcheck[- ]?out yavaştı\b", r"\bdepozitoyu geri vermediler hemen\b",
                        r"\bayrılırken resepsiyonda kimse yoktu\b",
                    ],
                    "ar": [
                        r"\bالخروج كان سهل\b", r"\bطلّعونا بسرعة\b", r"\bما في مشاكل عند الخروج\b",
                        r"\bتأخير بالخروج\b", r"\bما رجعوا العربون بسرعة\b",
                        r"\bما لقينا حدا بالاستقبال وقت رحنا\b", r"\bتشيك آوت معقد\b",
                    ],
                    "zh": [
                        r"退房很方便", r"退房很快", r"走得很顺利", r"没什么麻烦",
                        r"退房很慢", r"退押金拖很久", r"退房的时候前台没人", r"交钥匙很麻烦",
                    ],
                },
                "aspects": [
                    "checkout_easy", "checkout_fast",
                    "checkout_slow", "deposit_return_issue", "checkout_no_staff",
                ],
            },
        },
    },

    "cleanliness": {
        "display": "Чистота",
        "subtopics": {
    
            "arrival_clean": {
                "display": "Чистота номера при заезде",
                "patterns": {
                    "ru": [
                        r"\bчисто при заселении\b", r"\bномер был чистый\b", r"\bвсё убрано\b",
                        r"\bидеально чисто\b", r"\bочень чисто\b",
                        r"\bсвежее бель[её]\b", r"\bсвежая постель\b", r"\bпостель чистая\b",
                        r"\bполы чистые\b", r"\bникакой пыли\b",
    
                        r"\bгрязно при заселении\b", r"\bв номере не убрано\b", r"\bгрязный номер\b",
                        r"\bпыль на поверхностях\b", r"\bгрязный пол\b", r"\bлипкий пол\b", r"\bлипкий стол\b",
                        r"\bпятна на постел[еи]\b", r"\bгрязная постель\b",
                        r"\bволосы на кроват[еию]\b", r"\bволосы на подушк[е]\b",
                        r"\bмусор от прошлых гостей\b", r"\bбутылки остались\b", r"\bгрязные полотенца от предыдущих\b",
                        r"\bкрошки на столе\b", r"\bкрошки везде\b",
                    ],
                    "en": [
                        r"\broom was clean on arrival\b", r"\bvery clean\b", r"\bspotless\b",
                        r"\beverything was cleaned\b", r"\bfresh bedding\b", r"\bclean sheets\b",
                        r"\bno dust\b", r"\bfloors were clean\b",
    
                        r"\broom was dirty\b", r"\bdirty when we arrived\b", r"\bnot cleaned before we arrived\b",
                        r"\bdust on surfaces\b", r"\bdusty surfaces\b", r"\bsticky floor\b", r"\bsticky table\b",
                        r"\bstains on the bed\b", r"\bstained sheets\b",
                        r"\bhair on the bed\b", r"\bhair on the pillow\b",
                        r"\btrash from previous guest\b", r"\bprevious guest's garbage\b", r"\bold towels left\b",
                        r"\bcrumbs everywhere\b",
                    ],
                    "tr": [
                        r"\boda temizdi\b", r"\bgeldiğimizde tertemizdi\b", r"\bher yer temizdi\b",
                        r"\byeni çarşaf\b", r"\btemiz çarşaf\b", r"\btoz yoktu\b",
    
                        r"\boda kirliydi\b", r"\bgeldiğimizde kirliydi\b", r"\btemizlenmemişti\b",
                        r"\btoz vardı\b", r"\byapış yapış masa\b", r"\byapış yapış zemin\b",
                        r"\bçarşafta leke vardı\b", r"\bçarşafta saç vardı\b", r"\byastıkta saç vardı\b",
                        r"\bönceki misafirin çöpleri\b", r"\beski havlular duruyordu\b",
                        r"\bher yerde kırıntı vardı\b",
                    ],
                    "ar": [
                        r"\bالغرفة نظيفة عند الوصول\b", r"\bنظيف جدًا\b", r"\bكل شي كان نظيف\b",
                        r"\bمفارش نظيفة\b", r"\bالسرير نظيف\b", r"\bمافي غبار\b",
    
                        r"\bالغرفة كانت وسخة عند الوصول\b", r"\bمكان ما كان منظف\b",
                        r"\bغبار على السطوح\b", r"\bالأرض وسخة\b", r"\bالطاولة لزجة\b",
                        r"\bبقع على الشرشف\b", r"\bشعر على السرير\b", r"\bشعر على المخدة\b",
                        r"\bزبالة من الضيف السابق\b", r"\bمناشف وسخة من قبل\b",
                        r"\bفتات على الطاولة\b",
                    ],
                    "zh": [
                        r"一进来就很干净", r"房间很干净", r"打扫得很干净",
                        r"床单很干净", r"新的床单", r"没有灰尘", r"地板很干净",
    
                        r"刚入住的时候就很脏", r"房间没打扫",
                        r"桌子黏黏的", r"地板黏", r"到处是灰",
                        r"床单有污渍", r"床上有头发", r"枕头上有头发",
                        r"上个客人的垃圾还在", r"旧的毛巾还在",
                        r"桌上有碎屑",
                    ],
                },
                "aspects": [
                    "clean_on_arrival", "fresh_bedding", "no_dust_surfaces",
                    "floor_clean",
                    "dirty_on_arrival", "dusty_surfaces", "sticky_surfaces",
                    "stained_bedding", "hair_on_bed",
                    "leftover_trash_previous_guest", "used_towels_left", "crumbs_left",
                ],
            },
    
            "bathroom_state": {
                "display": "Санузел при заезде",
                "patterns": {
                    "ru": [
                        r"\bванная чистая\b", r"\bсанузел чистый\b", r"\bдуш чистый\b", r"\bвсё блестит\b",
                        r"\bчистая раковина\b", r"\bникакой плесени\b",
    
                        r"\bгрязный санузел\b", r"\bгрязный унитаз\b", r"\bгрязный туалет\b",
                        r"\bгрязная раковина\b", r"\bгрязный слив\b",
                        r"\bволосы в душе\b", r"\bволосы в раковине\b",
                        r"\bплесень в душе\b", r"\bч(е|ё)рная плесень\b", r"\bплесень в швах\b",
                        r"\bнал[её]т\b", r"\bизвестковый нал[её]т\b", r"\bржавчина\b",
                        r"\bвоняет из туалета\b", r"\bзапах канализации в ванной\b",
                    ],
                    "en": [
                        r"\bbathroom was clean\b", r"\bbathroom spotless\b", r"\bshower was very clean\b",
                        r"\bsink was clean\b", r"\bno mold\b", r"\bno mildew\b",
    
                        r"\bdirty bathroom\b", r"\bdirty toilet\b", r"\bdirty sink\b",
                        r"\bhair in the shower\b", r"\bhair in the sink\b", r"\bhair in the drain\b",
                        r"\bmold in the shower\b", r"\bblack mold\b", r"\bmildew\b",
                        r"\blimescale\b", r"\brust stains\b",
                        r"\bsewage smell in the bathroom\b", r"\bbad smell from the toilet\b",
                    ],
                    "tr": [
                        r"\bbanyo temizdi\b", r"\bduş çok temizdi\b", r"\blavabo tertemizdi\b",
                        r"\bküf yoktu\b",
    
                        r"\bbanyo kirliydi\b", r"\btuvalet kirliydi\b", r"\blavabo kirliydi\b",
                        r"\bduşta saç vardı\b", r"\blavaboda saç vardı\b",
                        r"\bduşta küf vardı\b", r"\bfayanslarda küf\b",
                        r"\bkireç lekeleri\b", r"\bpas lekeleri\b",
                        r"\btuvaletten koku geliyordu\b",
                    ],
                    "ar": [
                        r"\bالحمام نظيف\b", r"\bالدوش نظيف\b", r"\bمغسلة نظيفة\b", r"\bمافي عفن\b",
    
                        r"\bحمام وسخ\b", r"\bتواليت وسخ\b", r"\bمغسلة وسخة\b",
                        r"\bشعر في الدوش\b", r"\bشعر في المغسلة\b",
                        r"\bعفن في الدوش\b", r"\bعفن أسود\b",
                        r"\bرواسب كلس\b", r"\bبقع صدأ\b",
                        r"\bريحة مجاري من الحمام\b", r"\bريحة تواليت\b",
                    ],
                    "zh": [
                        r"卫生间很干净", r"浴室很干净", r"淋浴很干净", r"洗手池很干净", r"没有霉",
                        r"没有霉斑",
    
                        r"卫生间很脏", r"厕所很脏", r"洗手池很脏",
                        r"淋浴间有头发", r"下水口有头发",
                        r"有霉", r"黑色霉斑", r"瓷砖发霉",
                        r"有水垢", r"有锈迹",
                        r"卫生间有下水道味", r"厕所味很重",
                    ],
                },
                "aspects": [
                    "bathroom_clean_on_arrival", "no_mold_visible", "sink_clean", "shower_clean",
                    "bathroom_dirty_on_arrival", "hair_in_shower", "hair_in_sink",
                    "mold_in_shower", "limescale_stains", "sewage_smell_bathroom",
                ],
            },
    
            "stay_cleaning": {
                "display": "Уборка во время проживания",
                "patterns": {
                    "ru": [
                        r"\bубирали каждый день\b", r"\bуборка ежедневно\b", r"\bприходили убирать\b",
                        r"\bвыносили мусор\b", r"\bвынесли мусор\b",
                        r"\bзастилали кровать\b", r"\bкровать заправляли\b",
    
                        r"\bне убирали\b", r"\bуборки не было\b", r"\bникто не убирался\b",
                        r"\bмусор не выносили\b", r"\bмусор так и остался\b",
                        r"\bкровать не заправили\b",
                        r"\bпришлось просить уборку\b", r"\bприходилось напоминать\b",
                        r"\bгрязь копилась\b", r"\bгрязно оставалось\b",
                    ],
                    "en": [
                        r"\bthey cleaned every day\b", r"\bdaily cleaning\b", r"\bthey came to clean\b",
                        r"\bthey took out the trash\b", r"\btrash was taken\b",
                        r"\bthey made the bed\b", r"\bbed was made every day\b",
    
                        r"\bnobody cleaned\b", r"\bno cleaning during stay\b",
                        r"\btrash was not taken\b", r"\btrash piled up\b",
                        r"\bbed was never made\b",
                        r"\bwe had to ask for cleaning\b", r"\bwe had to remind them\b",
                        r"\bit stayed dirty\b", r"\bit got dirty and stayed that way\b",
                    ],
                    "tr": [
                        r"\bher gün temizlediler\b", r"\bgünlük temizlik yapıldı\b", r"\bgeldiler temizlediler\b",
                        r"\bçöpü aldılar\b", r"\bçöpü topladılar\b",
                        r"\byatağı düzelttiler\b",
    
                        r"\btemizlik yapılmadı\b", r"\bhiç temizlenmedi\b",
                        r"\bçöpü almadılar\b", r"\bçöp birikti\b",
                        r"\byatağı düzeltmediler\b",
                        r"\btemizlik için istemek zorunda kaldık\b", r"\bhatırlatmak zorunda kaldık\b",
                        r"\bkir kaldı\b", r"\bkir birikti\b",
                    ],
                    "ar": [
                        r"\bنظفوا كل يوم\b", r"\bيجوا ينظفوا\b",
                        r"\bشالوا الزبالة\b", r"\bسووا السرير\b",
    
                        r"\bما حد نظف\b", r"\bما في تنظيف طول الإقامة\b",
                        r"\bالزبالة ظلت\b", r"\bالزبالة تراكمت\b",
                        r"\bما سووا السرير\b",
                        r"\bاضطرينا نطلب تنظيف\b", r"\bاضطرينا نذكرهم\b",
                        r"\bظل وسخ\b", r"\bصار وسخ وما نظفوه\b",
                    ],
                    "zh": [
                        r"每天都会打扫", r"每天有人来打扫", r"有人来打扫房间",
                        r"垃圾每天都拿走", r"垃圾有拿走",
                        r"床每天都有整理", r"床整理好了",
    
                        r"没人打扫", r"住着期间没有打扫", r"从来没人来打扫",
                        r"垃圾没人倒", r"垃圾越积越多",
                        r"床也不整理",
                        r"我们得自己要求打扫", r"我们得提醒他们",
                        r"一直很脏", r"越住越脏",
                    ],
                },
                "aspects": [
                    "housekeeping_regular", "trash_taken_out", "bed_made",
                    "housekeeping_missed", "trash_not_taken", "bed_not_made",
                    "had_to_request_cleaning", "dirt_accumulated",
                ],
            },
    
            "linen_towels": {
                "display": "Полотенца, бельё и принадлежности",
                "patterns": {
                    "ru": [
                        r"\bменяли полотенца\b", r"\bполотенца меняли регулярно\b",
                        r"\bпринесли чистые полотенца сразу\b", r"\bпринесли новые полотенца\b",
                        r"\bсменили постельное бель[её]\b", r"\bновое бель[её]\b",
                        r"\bпополняли воду\b", r"\bпринесли воду\b",
                        r"\bпополняли туалетную бумагу\b", r"\bпринесли туалетную бумагу\b",
                        r"\bпринесли мыло\b", r"\bпополняли шампунь\b",
    
                        r"\bполотенца грязные\b", r"\bгрязные полотенца\b", r"\bпятна на полотенцах\b",
                        r"\bполотенца пахли\b", r"\bнеприятный запах от полотенец\b",
                        r"\bне меняли полотенца\b", r"\bполотенца не меняли\b",
                        r"\bбелье не поменяли\b", r"\bне поменяли белье\b",
                        r"\bне пополняли воду\b", r"\bне принесли туалетную бумагу\b", r"\bне пополняли мыло\b",
                    ],
                    "en": [
                        r"\bthey changed the towels\b", r"\bfresh towels\b", r"\bclean towels\b",
                        r"\bbrought new towels right away\b",
                        r"\bchanged the sheets\b", r"\bfresh bedding\b",
                        r"\brestocked toiletries\b", r"\bgave us toilet paper\b", r"\bbrought water\b",
    
                        r"\bdirty towels\b", r"\btowels were dirty\b", r"\bstains on the towels\b",
                        r"\btowels smelled bad\b", r"\btowels smelled\b",
                        r"\bthey never changed the towels\b", r"\bsheets not changed\b",
                        r"\bno toilet paper\b", r"\bthey didn't restock soap\b", r"\bno water refill\b",
                    ],
                    "tr": [
                        r"\bhavluları değiştirdiler\b", r"\btemiz havlu getirdiler\b",
                        r"\bhemen yeni havlu getirdiler\b",
                        r"\bçarşafları değiştirdiler\b", r"\btemiz çarşaf\b",
                        r"\btuvalet kağıdı getirdiler\b", r"\bşampuan yenilediler\b", r"\bsu bıraktılar\b",
    
                        r"\bhavlular kirliydi\b", r"\bhavluda leke vardı\b", r"\bhavlu kötü kokuyordu\b",
                        r"\bhavlu değiştirmediler\b", r"\bçarşaf değiştirmediler\b",
                        r"\btuvalet kağıdı yoktu\b", r"\byenilemediler\b", r"\bsu getirmediler\b",
                    ],
                    "ar": [
                        r"\bمناشف نظيفة\b", r"\bجابوا مناشف جديدة\b",
                        r"\bغيروا الشراشف\b", r"\bشرشف نظيف\b",
                        r"\bجابوا ورق تواليت\b", r"\bجابوا صابون\b", r"\bجابوا مي\b",
    
                        r"\bمناشف وسخة\b", r"\bبقع على المناشف\b", r"\bريحة المناشف سيئة\b",
                        r"\bما غيروا المناشف\b", r"\bما غيروا الشراشف\b",
                        r"\bما رجعوا ورق تواليت\b", r"\bما زودونا بالصابون\b", r"\bما جابوا مي\b",
                    ],
                    "zh": [
                        r"给了新的毛巾", r"毛巾很干净", r"马上送了干净的毛巾",
                        r"换了床单", r"床单是干净的",
                        r"补了卫生纸", r"补了洗浴用品", r"补了水",
    
                        r"毛巾很脏", r"毛巾有污渍", r"毛巾有味道",
                        r"毛巾一直没换", r"床单没换",
                        r"没有卫生纸", r"不补洗漱用品", r"没有水补充",
                    ],
                },
                "aspects": [
                    "towels_changed", "fresh_towels_fast", "linen_changed",
                    "amenities_restocked",
                    "towels_dirty", "towels_stained", "towels_smell",
                    "towels_not_changed", "linen_not_changed", "no_restock",
                ],
            },
    
            "smell": {
                "display": "Запахи",
                "patterns": {
                    "ru": [
                        r"\bзапах сигарет\b", r"\bпахло табаком\b", r"\bпахло дымом\b",
                        r"\bзапах канализации\b", r"\bвоняет из канализации\b",
                        r"\bзапах плесени\b", r"\bзапах сырости\b", r"\bзатхлый запах\b", r"\bсырой запах\b",
                        r"\bвоняло хлоркой\b", r"\bсильный запах химии\b",
    
                        r"\bникакого неприятного запаха\b", r"\bничем не пахло\b",
                        r"\bсвежий запах\b", r"\bприятно пахнет\b",
                    ],
                    "en": [
                        r"\bsmelled like smoke\b", r"\bcigarette smell\b", r"\bsmelled of tobacco\b",
                        r"\bsewage smell\b", r"\bsewer smell\b", r"\bsmelled like sewage\b",
                        r"\bmoldy smell\b", r"\bmusty smell\b", r"\bdamp smell\b",
                        r"\bstrong bleach smell\b", r"\bsmelled like chemicals\b",
    
                        r"\bno bad smell\b", r"\bno smell at all\b", r"\bfresh smell\b", r"\bsmelled clean\b",
                    ],
                    "tr": [
                        r"\bsigara kokusu\b", r"\btütün kokuyordu\b",
                        r"\bkanalizasyon kokusu\b", r"\blağım kokusu\b",
                        r"\bnem kokusu\b", r"\bküf kokusu\b", r"\brutubet kokusu\b",
                        r"\başırı çamaşır suyu kokusu\b", r"\bkimyasal kokuyordu\b",
    
                        r"\bkötü koku yoktu\b", r"\bhiç koku yoktu\b", r"\btemiz kokuyordu\b",
                    ],
                    "ar": [
                        r"\bريحة دخان\b", r"\bريحة سجاير\b",
                        r"\bريحة مجاري\b", r"\bريحة صرف\b",
                        r"\bريحة رطوبة\b", r"\bريحة عفن\b", r"\bريحة عفن رطوبة\b",
                        r"\bريحة كلور قوية\b", r"\bريحة مواد تنظيف قوية\b",
    
                        r"\bما في ريحة مزعجة\b", r"\bما في ريحة\b", r"\bريحة نظيفة\b", r"\bريحة حلوة\b",
                    ],
                    "zh": [
                        r"有烟味", r"有香烟味",
                        r"下水道味", r"下水道的味道",
                        r"霉味", r"潮味", r"发霉的味道", r"很潮很闷的味道",
                        r"一股消毒水味", r"消毒水味太重", r"化学品的味道很重",
    
                        r"没有异味", r"没有味道", r"闻起来很干净", r"味道很清新",
                    ],
                },
                "aspects": [
                    "smell_of_smoke", "sewage_smell", "musty_smell",
                    "chemical_smell_strong",
                    "no_bad_smell", "fresh_smell",
                ],
            },
    
            "public_areas": {
                "display": "Общие зоны и входные группы",
                "patterns": {
                    "ru": [
                        r"\bчистый подъезд\b", r"\bчистая лестница\b", r"\bчисто в коридоре\b",
                        r"\bчистая общая зона\b", r"\bчисто в холле\b", r"\bаккуратный коридор\b",
    
                        r"\bгрязный подъезд\b", r"\bстарый грязный подъезд\b", r"\bподъезд в ужасном состоянии\b",
                        r"\bгрязный вход\b", r"\bвход грязный\b", r"\bлестница грязная\b",
                        r"\bгрязные коридоры\b", r"\bгрязно в коридоре\b", r"\bпыльно в коридоре\b",
                        r"\bгрязный лифт\b", r"\bлифт грязный\b",
                        r"\bнеприятный запах в подъезде\b", r"\bвоняет в коридоре\b",
                        r"\bмрачно в подъезде\b", r"\bвыглядит небезопасно\b", r"\bпугающий вход\b",
                    ],
                    "en": [
                        r"\bclean hallway\b", r"\bclean corridor\b", r"\bcommon areas were clean\b",
                        r"\bentrance was clean\b", r"\bstairwell was clean\b", r"\blobby was clean\b",
    
                        r"\bdirty entrance\b", r"\bdirty hallway\b", r"\bhallway looked terrible\b",
                        r"\bstairwell was dirty\b", r"\bdirty corridor\b", r"\bdusty corridor\b",
                        r"\bdirty elevator\b", r"\belevator was dirty\b",
                        r"\bsmelled bad in the hallway\b",
                        r"\bthe entrance felt sketchy\b", r"\bnot safe looking entrance\b",
                    ],
                    "tr": [
                        r"\bkoridor temizdi\b", r"\bortak alanlar temizdi\b", r"\bgiriş temizdi\b", r"\blobi temizdi\b",
    
                        r"\bkirli giriş\b", r"\bmerdivenler kirliydi\b", r"\bkoridor kirliydi\b", r"\btozluydu\b",
                        r"\basansör kirliydi\b",
                        r"\bkoridorda kötü koku vardı\b",
                        r"\bgiriş güvensiz görünüyordu\b", r"\bgiriş korkutucuydu\b",
                    ],
                    "ar": [
                        r"\bالمدخل نظيف\b", r"\bالدرج نظيف\b", r"\bالممر نظيف\b",
                        r"\bالمناطق المشتركة نظيفة\b", r"\bاللوبي نظيف\b",
    
                        r"\bمدخل وسخ\b", r"\bالدرج وسخ\b", r"\bالممر وسخ\b",
                        r"\bالممر ريحته سيئة\b", r"\bالمصعد وسخ\b",
                        r"\bالمدخل شكله يخوف\b", r"\bالمدخل مش مريح\b", r"\bمبين مو آمن\b",
                    ],
                    "zh": [
                        r"走廊很干净", r"公共区域很干净", r"入口很干净", r"楼道很干净", r"大堂很干净",
    
                        r"入口很脏", r"楼道很脏", r"走廊很脏", r"电梯很脏",
                        r"走廊有异味",
                        r"入口让人不舒服", r"入口看起来不安全", r"感觉很吓人",
                    ],
                },
                "aspects": [
                    "entrance_clean", "hallway_clean", "common_areas_clean",
                    "entrance_dirty", "hallway_dirty", "elevator_dirty",
                    "hallway_bad_smell", "entrance_feels_unsafe",
                ],
            },
        },
    },

    "comfort": {
        "display": "Комфорт проживания",
        "subtopics": {
    
            "room_equipment": {
                "display": "Оснащение и удобство номера",
                "patterns": {
                    "ru": [
                        r"\bвсё продумано\b", r"\bочень удобно\b", r"\bвсё необходимое есть\b", r"\bесть всё необходимое\b",
                        r"\bв номере есть чайник\b", r"\bесть чайник и посуда\b", r"\bесть холодильник\b",
                        r"\bмного розеток\b", r"\bрозетки рядом с кроватью\b",
                        r"\bесть фен\b", r"\bесть утюг\b", r"\bудобный рабочий стол\b",
                        r"\bесть где разложить чемоданы\b", r"\bесть куда разложить вещи\b",
    
                        r"\bне хватало посуды\b", r"\bне хватает посуды\b", r"\bне хватает чайника\b",
                        r"\bнет чайника\b", r"\bнет холодильника\b", r"\bнет фена\b",
                        r"\bнеудобно разложить вещи\b", r"\bнекуда разложить вещи\b", r"\bне было места для чемодана\b",
                        r"\bмало розеток\b", r"\bрозеток не хватает\b",
                        r"\bнет нормального стола\b", r"\bнеудобно работать\b", r"\bнегде поработать\b",
                    ],
                    "en": [
                        r"\bwell equipped room\b", r"\bthe room had everything we needed\b",
                        r"\bthere was a kettle\b", r"\bthere was a fridge\b", r"\bthere was a hairdryer\b",
                        r"\benough outlets\b", r"\bsockets next to the bed\b",
                        r"\bgood desk to work\b", r"\bspace for luggage\b", r"\bplace to unpack\b",
    
                        r"\bno kettle\b", r"\bno fridge\b", r"\bno hairdryer\b",
                        r"\bnot enough outlets\b", r"\bno sockets near the bed\b",
                        r"\bno place for luggage\b", r"\bno space to unpack\b",
                        r"\bnowhere to work\b", r"\bno proper desk\b",
                    ],
                    "tr": [
                        r"\bodada her şey vardı\b", r"\bihtiyacımız olan her şey vardı\b",
                        r"\bsu ısıtıcısı vardı\b", r"\bbuzdolabı vardı\b", r"\bsaç kurutma makinesi vardı\b",
                        r"\byeterince priz vardı\b", r"\byatağın yanında priz vardı\b",
                        r"\bçalışmak için masa vardı\b", r"\bbavulu açacak yer vardı\b",
    
                        r"\bsu ısıtıcısı yoktu\b", r"\bbuzdolabı yoktu\b", r"\bsaç kurutma yoktu\b",
                        r"\bpriz azdı\b", r"\byatağın yanında priz yoktu\b",
                        r"\bbavulu açacak yer yoktu\b", r"\beşyaları koyacak yer yoktu\b",
                        r"\bçalışacak masa yoktu\b",
                    ],
                    "ar": [
                        r"\bكل شي موجود بالغرفة\b", r"\bكل الأشياء المهمة موجودة\b",
                        r"\bفي غلاية ماء\b", r"\bفي براد\b", r"\bفي سيشوار\b",
                        r"\bفي فيش جنب التخت\b", r"\bفي مكاتب للشغل\b",
                        r"\bفي مكان للشنط\b", r"\bفي مساحة نرتب أغراضنا\b",
    
                        r"\bما في غلاية\b", r"\bما في براد\b", r"\bما في سيشوار\b",
                        r"\bمافي فيش قريب من السرير\b",
                        r"\bما في مكان للشنط\b", r"\bما في مساحة نحط الأغراض\b",
                        r"\bما في طاولة نشتغل عليها\b",
                    ],
                    "zh": [
                        r"房间设备齐全", r"需要的东西都有",
                        r"有水壶", r"有烧水壶", r"有冰箱", r"有吹风机",
                        r"插座很多", r"床边有插座",
                        r"有书桌可以办公", r"有地方放行李", r"行李可以打开",
    
                        r"没有水壶", r"没有热水壶", r"没有冰箱", r"没有吹风机",
                        r"插座不够", r"床边没有插座",
                        r"没地方放行李", r"行李没法打开",
                        r"没有桌子可以办公", r"没有书桌",
                    ],
                },
                "aspects": [
                    "room_well_equipped", "kettle_available", "fridge_available",
                    "hairdryer_available", "sockets_enough", "workspace_available",
                    "luggage_space_ok",
                    "kettle_missing", "fridge_missing", "hairdryer_missing",
                    "sockets_not_enough", "no_workspace", "no_luggage_space",
                ],
            },
    
            "sleep_quality": {
                "display": "Сон и кровать",
                "patterns": {
                    "ru": [
                        r"\bкровать удобная\b", r"\bочень удобная кровать\b",
                        r"\bудобный матрас\b", r"\bматрас удобный\b",
                        r"\bудобные подушки\b", r"\bподушки удобные\b",
                        r"\bспать было комфортно\b", r"\bспалось отлично\b", r"\bвыспались отлично\b",
    
                        r"\bкровать неудобная\b", r"\bнеудобная кровать\b",
                        r"\bматрас слишком мягк(ий|ий)\b", r"\bслишком мягкий матрас\b",
                        r"\bматрас слишком ж(ё|е)сткий\b", r"\bслишком жесткий матрас\b",
                        r"\bпродавленный матрас\b", r"\bматрас проваливался\b",
                        r"\bкровать скрипела\b", r"\bскрипучая кровать\b",
                        r"\bподушки неудобные\b", r"\bжесткие подушки\b", r"\bслишком высокие подушки\b",
                    ],
                    "en": [
                        r"\bthe bed was very comfortable\b", r"\bcomfortable bed\b",
                        r"\bcomfortable mattress\b", r"\bgood mattress\b",
                        r"\bpillows were comfortable\b", r"\bslept really well\b", r"\bslept great\b",
    
                        r"\buncomfortable bed\b",
                        r"\bmattress too soft\b", r"\bmattress too hard\b",
                        r"\bmattress was sagging\b", r"\bsaggy mattress\b",
                        r"\bcreaky bed\b", r"\bthe bed was creaking\b",
                        r"\bpillows were uncomfortable\b", r"\bpillows too hard\b", r"\bpillows too high\b",
                    ],
                    "tr": [
                        r"\byatak rahattı\b", r"\bçok rahat yatak\b",
                        r"\bşilte rahattı\b", r"\byastıklar rahattı\b",
                        r"\bçok iyi uyuduk\b", r"\biyı dinlendik\b",
    
                        r"\byatak rahatsızdı\b", r"\brahat değildi\b",
                        r"\bşilte çok yumuşaktı\b", r"\bşilte çok sertti\b",
                        r"\bşilte çökmüştü\b", r"\byatağın yayları hissediliyordu\b",
                        r"\byatak gıcırdıyordu\b",
                        r"\byastıklar rahatsızdı\b", r"\byastık çok sertti\b", r"\byastık çok yüksekti\b",
                    ],
                    "ar": [
                        r"\bالسرير مريح\b", r"\bالماترس مريح\b", r"\bالمخدات مريحة\b",
                        r"\bنمنا منيح\b", r"\bنمنا كتير منيح\b",
    
                        r"\bالسرير مو مريح\b", r"\bمش مريح\b",
                        r"\bالماترس لين كتير\b", r"\bالماترس قاسي كتير\b",
                        r"\bالماترس غاطس\b", r"\bالماترس خربان\b",
                        r"\bالسرير بيوزن\b", r"\bالسرير بيصرّف\b", r"\bالسرير بيصرخ\b",
                        r"\bالمخدات مو مريحة\b", r"\bالمخدة عالية كتير\b", r"\bقاسية كتير\b",
                    ],
                    "zh": [
                        r"床很舒服", r"床垫很舒服", r"枕头很舒服",
                        r"睡得很好", r"睡得很香", r"睡得很棒",
    
                        r"床不舒服", r"床垫不舒服",
                        r"床垫太软", r"床垫太硬",
                        r"床垫塌了", r"床垫塌陷",
                        r"床会吱吱响", r"床一直响",
                        r"枕头不舒服", r"枕头太硬", r"枕头太高",
                    ],
                },
                "aspects": [
                    "bed_comfy", "mattress_comfy", "pillow_comfy", "slept_well",
                    "bed_uncomfortable", "mattress_too_soft", "mattress_too_hard",
                    "mattress_sagging", "bed_creaks",
                    "pillow_uncomfortable", "pillow_too_hard", "pillow_too_high",
                ],
            },
    
            "noise": {
                "display": "Шум и звукоизоляция",
                "patterns": {
                    "ru": [
                        r"\bтихо\b", r"\bочень тихо\b", r"\bспокойно ночью\b",
                        r"\bхорошая звукоизоляция\b", r"\bничего не слышно\b", r"\bсоседей не слышно\b", r"\bулицу не слышно\b",
    
                        r"\bшумно\b", r"\bочень шумно\b", r"\bшум с улицы\b", r"\bгромко с улицы\b",
                        r"\bтонкие стены\b", r"\bслышно соседей\b", r"\bслышно всё из коридора\b",
                        r"\bслышно ресепшен\b", r"\bслышно лифт\b",
                        r"\bмузыка ночью\b", r"\bкрики ночью\b", r"\bне могли уснуть из-за шума\b",
                    ],
                    "en": [
                        r"\bvery quiet\b", r"\bnice and quiet\b", r"\bquiet at night\b", r"\bgood soundproofing\b",
                        r"\bwe couldn't hear the neighbors\b", r"\bno street noise\b",
    
                        r"\bnoisy\b", r"\bvery noisy\b", r"\bstreet noise\b", r"\btraffic noise\b",
                        r"\bthin walls\b", r"\byou can hear everything\b", r"\bwe could hear the neighbors\b",
                        r"\bnoise from reception\b", r"\bnoise from the hallway\b", r"\bnoise from the elevator\b",
                        r"\bloud music at night\b", r"\bpeople shouting at night\b", r"\bhard to sleep because of noise\b",
                    ],
                    "tr": [
                        r"\bçok sessizdi\b", r"\bgece çok sakindi\b", r"\biyı ses yalıtımı vardı\b",
                        r"\bkomşuları duymuyorduk\b", r"\bsokağın sesi yoktu\b",
    
                        r"\bçok gürültülüydü\b", r"\bgece gürültülüydü\b",
                        r"\bsokak çok gürültülüydü\b", r"\btrafik sesi vardı\b",
                        r"\bduvarlar inceydi\b", r"\bher şeyi duyabiliyorduk\b", r"\byan odanın sesini duyuyorduk\b",
                        r"\bresepsiyondan ses geliyordu\b", r"\bkoridordan ses geliyordu\b", r"\basansör sesi geliyordu\b",
                        r"\bgece müzik vardı\b", r"\bgece bağırış vardı\b", r"\buyuyamadık gürültüden\b",
                    ],
                    "ar": [
                        r"\bهادئ\b", r"\bكتير هادي\b", r"\bبالليل هادي\b", r"\bالعزل منيح\b",
                        r"\bما بنسمع حدا\b", r"\bما في صوت شارع\b",
    
                        r"\bفي ازعاج\b", r"\bصوت عالي\b", r"\bصوت الشارع عالي\b",
                        r"\bالجدران رفيعة\b", r"\bعم نسمع الجيران\b", r"\bعم نسمع كل شي من الممر\b",
                        r"\bصوت الريسيبشن\b", r"\bصوت الأسانسير\b",
                        r"\bموسيقى بالليل\b", r"\bصراخ بالليل\b", r"\bما قدرنا ننام من الصوت\b",
                    ],
                    "zh": [
                        r"很安静", r"晚上很安静", r"隔音很好", r"听不到邻居", r"没有街上的噪音",
    
                        r"很吵", r"晚上很吵", r"街上很吵", r"马路太吵",
                        r"隔音很差", r"墙很薄", r"能听到隔壁",
                        r"能听到走廊的声音", r"能听到前台", r"能听到电梯",
                        r"晚上有人大声讲话", r"晚上有音乐", r"吵得睡不着",
                    ],
                },
                "aspects": [
                    "quiet_room", "good_soundproofing", "no_street_noise",
                    "noisy_room", "street_noise", "thin_walls",
                    "hallway_noise", "night_noise_trouble_sleep",
                ],
            },
    
            "climate": {
                "display": "Температура и воздух",
                "patterns": {
                    "ru": [
                        r"\bтемпература комфортная\b", r"\bтемпература идеальная\b",
                        r"\bне жарко\b", r"\bне холодно\b",
                        r"\bможно проветрить\b", r"\bхорошо проветривается\b",
                        r"\bкондиционер работает\b", r"\bкондиционер отлично работал\b",
                        r"\bотопление хорошее\b", r"\bв номере тепло\b",
    
                        r"\bжарко\b", r"\bочень жарко\b", r"\bспать жарко\b",
                        r"\bдушно\b", r"\bнечем дышать\b", r"\bне проветривается\b",
                        r"\bслишком холодно\b", r"\bв номере холодно\b",
                        r"\bкондиционер не работал\b", r"\bкондиционера нет\b", r"\bкондиционер не спасал\b",
                        r"\bотопление не работало\b", r"\bхолодные батареи\b",
                        r"\bсквозняк\b", r"\bдует из окна\b",
                    ],
                    "en": [
                        r"\btemperature was comfortable\b", r"\bperfect temperature\b",
                        r"\bnot too hot\b", r"\bnot too cold\b",
                        r"\beasy to air the room\b", r"\bgood ventilation\b",
                        r"\bAC worked well\b", r"\bair conditioning worked well\b",
                        r"\bheating worked\b", r"\bthe room was warm enough\b",
    
                        r"\btoo hot\b", r"\bvery hot in the room\b", r"\bhard to sleep because it was hot\b",
                        r"\bstuffy\b", r"\bno air\b", r"\bno ventilation\b",
                        r"\btoo cold\b", r"\bcold in the room\b",
                        r"\bAC didn't work\b", r"\bno AC\b", r"\bAC was not working\b",
                        r"\bheating didn't work\b", r"\bno heating\b",
                        r"\bdraft from the window\b", r"\bcold draft\b",
                    ],
                    "tr": [
                        r"\boda sıcaklığı rahattı\b", r"\bne çok sıcak ne çok soğuk\b",
                        r"\boda havalanabiliyordu\b", r"\bhava sirkülasyonu iyiydi\b",
                        r"\bklima çalışıyordu\b", r"\bısıtma çalışıyordu\b", r"\bodada yeterince sıcaktı\b",
    
                        r"\boda çok sıcaktı\b", r"\buyuyamayacak kadar sıcaktı\b",
                        r"\bhava boğucuydu\b", r"\bhava alamadık\b", r"\bhavalandırma yoktu\b",
                        r"\boda soğuktu\b", r"\bçok soğuktu içerisi\b",
                        r"\bklima çalışmıyordu\b", r"\bklima yoktu\b",
                        r"\bısıtma çalışmıyordu\b",
                        r"\bcamdan rüzgar geliyordu\b", r"\bcereyan vardı\b",
                    ],
                    "ar": [
                        r"\bالحرارة مريحة\b", r"\bالجو بالغرفة كان تمام\b",
                        r"\bمش حار ومش برد\b",
                        r"\bفي تهوية كويسة\b", r"\bقدرنا نهوّي\b",
                        r"\bالمكيف شغال منيح\b", r"\bالتدفئة شغالة\b",
    
                        r"\bحر كتير\b", r"\bحر ما منقدر ننام\b",
                        r"\bمخنوقين\b", r"\bما في هوا\b", r"\bما في تهوية\b",
                        r"\bبرد بالغرفة\b", r"\bالغرفة باردة\b",
                        r"\bالمكيف ما بيشتغل\b", r"\bما في مكيف\b",
                        r"\bالتدفئة ما اشتغلت\b",
                        r"\bفي هوا بارد من الشباك\b", r"\bفي هوا داخل من الشباك\b",
                    ],
                    "zh": [
                        r"房间温度很舒服", r"温度正好", r"不热不冷",
                        r"可以通风", r"通风很好",
                        r"空调很好用", r"空调很给力",
                        r"暖气很好", r"房间很暖和",
    
                        r"房间太热", r"太热了睡不着",
                        r"很闷", r"空气很闷", r"没有空气流通", r"没有通风",
                        r"房间很冷", r"很冷",
                        r"空调不好用", r"空调不工作", r"没有空调",
                        r"暖气不工作",
                        r"窗户漏风", r"有冷风进来",
                    ],
                },
                "aspects": [
                    "temp_comfortable", "ventilation_ok", "ac_working", "heating_working",
                    "too_hot_sleep_issue", "too_cold", "stuffy_no_air",
                    "no_ventilation", "ac_not_working", "no_ac",
                    "heating_not_working", "draft_window",
                ],
            },
    
            "space_light": {
                "display": "Пространство и освещённость",
                "patterns": {
                    "ru": [
                        r"\bпросторный номер\b", r"\bмного места\b", r"\bномер большой\b",
                        r"\bудобная планировка\b", r"\bвсё удобно расположено\b",
                        r"\bуютный номер\b", r"\bочень уютно\b",
                        r"\bсветлый номер\b", r"\bмного света\b", r"\bмного дневного света\b",
                        r"\bбольшие окна\b",
    
                        r"\bтесный номер\b", r"\bномер очень маленький\b", r"\bочень тесно\b",
                        r"\bне развернуться\b", r"\bнекуда поставить чемодан\b",
                        r"\bтемно в номере\b", r"\bмало света\b", r"\bпочти нет окна\b",
                        r"\bокно маленькое\b", r"\bмрачно\b", r"\bдавит\b",
                    ],
                    "en": [
                        r"\bspacious room\b", r"\ba lot of space\b", r"\broom was big\b",
                        r"\bgood layout\b", r"\bwell organized\b",
                        r"\bcozy\b", r"\bvery cozy\b", r"\bfelt cozy\b",
                        r"\bbright room\b", r"\blots of natural light\b", r"\bbig windows\b",
    
                        r"\bsmall room\b", r"\bvery small\b", r"\bcramped\b",
                        r"\bno space for luggage\b", r"\bhard to move around\b",
                        r"\bdark room\b", r"\bnot enough light\b", r"\bno natural light\b",
                        r"\bno window\b", r"\btiny window\b", r"\bfelt gloomy\b",
                    ],
                    "tr": [
                        r"\boda genişti\b", r"\bferah oda\b", r"\bçok boş alan vardı\b",
                        r"\bdüzeni iyiydi\b", r"\bdüzen çok kullanışlıydı\b",
                        r"\boda çok rahat bir his veriyor\b", r"\brahat/ev gibi hissettiriyordu\b",
                        r"\baydınlık odaydı\b", r"\bdoğal ışık çoktu\b", r"\bbüyük pencere vardı\b",
    
                        r"\boda küçüktü\b", r"\bçok küçüktü\b", r"\bsıkışıktı\b",
                        r"\bbavulu koyacak yer yoktu\b", r"\bhareket etmek zor\b",
                        r"\boda karanlıktı\b", r"\byeterince ışık yoktu\b",
                        r"\bpencere çok küçüktü\b", r"\bpencere yok gibi\b", r"\biçerisi kasvetliydi\b",
                    ],
                    "ar": [
                        r"\bالغرفة واسعة\b", r"\bمساحة واسعة\b", r"\bفيها مجال\b",
                        r"\bمرتبة بشكل مريح\b", r"\bكل شي بمكانه\b",
                        r"\bالغرفة مريحة ودافئة\b", r"\bبتحسها مريحة\b",
                        r"\bفيها ضو طبيعي\b", r"\bفيه شبابيك كبيرة\b", r"\bالغرفة منوّرة\b",
    
                        r"\bالغرفة صغيرة\b", r"\bكتير صغيرة\b", r"\bمخانقة\b",
                        r"\bمافي محل للشنط\b", r"\bصعب نتحرك\b",
                        r"\bالغرفة معتمة\b", r"\bما فيها ضو\b", r"\bما فيها ضو طبيعي\b",
                        r"\bشباك صغير\b", r"\bالغرفة كئيبة\b",
                    ],
                    "zh": [
                        r"房间很大", r"很宽敞", r"空间很大",
                        r"布局很合理", r"很方便摆放东西",
                        r"很温馨", r"很舒适", r"很有家的感觉",
                        r"房间很亮", r"光线很好", r"有自然光", r"窗户很大",
    
                        r"房间很小", r"很挤", r"空间很小",
                        r"行李没地方放", r"走不开",
                        r"房间很暗", r"灯光不够", r"没有自然光",
                        r"没有窗户", r"窗户很小", r"房间感觉有点压抑",
                    ],
                },
                "aspects": [
                    "room_spacious", "good_layout", "cozy_feel",
                    "bright_room", "big_windows",
                    "room_small", "no_space_for_luggage",
                    "dark_room", "no_natural_light", "gloomy_feel",
                ],
            },
    
        },
    },

    "tech_state": {
        "display": "Техническое состояние и инфраструктура",
        "subtopics": {
    
            "plumbing_water": {
                "display": "Вода и сантехника",
                "patterns": {
                    "ru": [
                        r"\bгорячая вода сразу\b", r"\bгорячая вода без перебоев\b",
                        r"\bнормальное давление воды\b", r"\bхорошее давление\b",
                        r"\bдуш работал отлично\b", r"\bвода текла равномерно\b",
                        r"\bничего не текло\b", r"\bничего не капало\b",
    
                        r"\bне было горячей воды\b", r"\bбез горячей воды\b", r"\bнет горячей воды утром\b",
                        r"\bгорячая вода пропадала\b", r"\bгорячая вода только на пару минут\b",
                        r"\bслабый напор\b", r"\bслабое давление\b", r"\bеле теч(е|ё)т\b",
                        r"\bдуш сломан\b", r"\bсломанный душ\b", r"\bдуш не держится\b", r"\bлейка не держится\b",
                        r"\bкран теч(е|ё)т\b", r"\bвода капает\b", r"\bпротечка\b", r"\bподтекает\b",
                        r"\bвода на полу после душа\b",
                        r"\bзасор в раковине\b", r"\bзасор в душе\b", r"\bслив не работает\b",
                        r"\bвонял[ao]? из слива\b", r"\bзапах из труб\b",
                    ],
                    "en": [
                        r"\bhot water right away\b", r"\bgood hot water\b", r"\bhot water all the time\b",
                        r"\bgood water pressure\b", r"\bstrong water pressure\b",
                        r"\bshower worked fine\b", r"\bshower worked perfectly\b",
                        r"\bno leaks\b", r"\bno dripping\b",
    
                        r"\bno hot water\b", r"\bno hot water in the morning\b",
                        r"\bhot water cuts off\b", r"\bhot water stops after a minute\b",
                        r"\blow water pressure\b", r"\bweak water pressure\b",
                        r"\bshower was broken\b", r"\bshower head broken\b", r"\bshower holder broken\b",
                        r"\bleaking tap\b", r"\bthe tap was leaking\b", r"\bwater was leaking everywhere\b",
                        r"\bflooded bathroom\b", r"\bwater all over the floor\b",
                        r"\bclogged drain\b", r"\bdrain was clogged\b", r"\bsink clogged\b",
                        r"\bbad smell from the drain\b", r"\bsewage smell from the shower\b",
                    ],
                    "tr": [
                        r"\bsıcak su hemen vardı\b", r"\bsıcak su sürekli vardı\b",
                        r"\bsu basıncı iyiydi\b", r"\bbasınç güçlüydü\b",
                        r"\bduş sorunsuz çalışıyordu\b",
                        r"\bhiç sızıntı yoktu\b",
    
                        r"\bsıcak su yoktu\b", r"\bsabah sıcak su yoktu\b",
                        r"\bsıcak su gidip geliyordu\b", r"\bsıcak su birden kesiliyordu\b",
                        r"\bsu basıncı çok düşüktü\b", r"\bbasınç zayıftı\b",
                        r"\bduş bozuktu\b", r"\bduş kafası kırıktı\b", r"\bduş askısı kırıktı\b",
                        r"\bmusluk akıtıyordu\b", r"\blavabo akıtıyordu\b", r"\bsu sızdırıyordu\b",
                        r"\bduştan sonra her yer su oldu\b", r"\byer hep ıslaktı\b",
                        r"\blavabo tıkalıydı\b", r"\bduş gideri tıkalıydı\b",
                        r"\bgidermekten kötü koku geliyordu\b", r"\blağım kokusu\b",
                    ],
                    "ar": [
                        r"\bفي مي سخنة ع طول\b", r"\bفي مي سخنة بدون انقطاع\b",
                        r"\bضغط المي منيح\b", r"\bالضغط قوي\b",
                        r"\bالدوش شغال تمام\b",
                        r"\bما في تسريب\b",
    
                        r"\bما في مي سخنة\b", r"\bما كان في مي سخنة الصبح\b",
                        r"\bالمي السخنة بتقطع\b", r"\bالمي السخنة وقفت\b",
                        r"\bالضغط ضعيف\b", r"\bالضغط كتير ضعيف\b",
                        r"\bالدوش خربان\b", r"\bرأس الدش مكسور\b", r"\bما بيثبت\b",
                        r"\bالحنفية عم تسرّب\b", r"\bفي تسريب مي\b", r"\bمي بكل الحمام\b",
                        r"\bالمصرف مسدود\b", r"\bالمجلى مسدود\b",
                        r"\bريحة مجاري من البالوعة\b",
                    ],
                    "zh": [
                        r"热水马上就有", r"热水很稳定",
                        r"水压很好", r"水压很大",
                        r"淋浴正常", r"淋浴很好用",
                        r"没有漏水",
    
                        r"没有热水", r"早上没有热水", r"热水用一下就没了",
                        r"水压很低", r"水压很小",
                        r"淋浴坏了", r"花洒坏了", r"花洒架坏了",
                        r"水龙头漏水", r"一直滴水",
                        r"洗完澡地上都是水", r"卫生间全是水",
                        r"下水道堵了", r"排水不通", r"洗手池堵了",
                        r"下水道有臭味", r"排水口有臭味",
                    ],
                },
                "aspects": [
                    "hot_water_ok", "water_pressure_ok", "shower_ok", "no_leak",
                    "no_hot_water", "weak_pressure", "shower_broken",
                    "leak_water", "bathroom_flooding",
                    "drain_clogged", "drain_smell",
                ],
            },
    
            "appliances_equipment": {
                "display": "Оборудование и состояние номера",
                "patterns": {
                    "ru": [
                        r"\bкондиционер работал\b", r"\bкондиционер отлично работал\b",
                        r"\bотопление работало\b",
                        r"\bвсё оборудование исправно\b", r"\bвсё работало\b",
                        r"\bтелевизор работает\b", r"\bтелевизор показывал нормально\b",
                        r"\bчайник работает\b", r"\bхолодильник работает\b",
                        r"\bдверь закрывается плотно\b", r"\bзамок нормальный\b",
                        r"\bничего не скрипит\b",
    
                        r"\bкондиционер не работал\b", r"\bкондиционер сломан\b",
                        r"\bотопление не работало\b", r"\bобогрев не работал\b",
                        r"\bтелевизор не работал\b", r"\bтелевизор не показывал\b",
                        r"\bхолодильник не работал\b", r"\bхолодильник еле холодил\b",
                        r"\bчайник не работал\b", r"\bчайник сломан\b",
                        r"\bрозетка искрит\b", r"\bрозетка болтается\b",
                        r"\bдверь плохо закрывается\b", r"\bдверь не закрывалась до конца\b",
                        r"\bзамок заедал\b", r"\bзамок не закрывался\b", r"\bне чувствовали себя в безопасности\b",
                        r"\bсломанный шкаф\b", r"\bдверца шкафа отваливается\b", r"\bшатается стол\b",
                        r"\bпошарпанные стены\b", r"\bоблезлые стены\b", r"\bтребует ремонта\b",
                    ],
                    "en": [
                        r"\bAC worked fine\b", r"\bair conditioning worked well\b",
                        r"\bheating worked\b", r"\bheater worked\b",
                        r"\beverything worked\b", r"\beverything in the room was working\b",
                        r"\bTV worked\b", r"\bTV channels were fine\b",
                        r"\bkettle worked\b", r"\bfridge worked\b",
                        r"\bdoor closed properly\b", r"\bfelt secure\b",
    
                        r"\bAC didn't work\b", r"\bAC was broken\b", r"\bair conditioner not working\b",
                        r"\bheating didn't work\b", r"\bheater was not working\b",
                        r"\bTV didn't work\b", r"\bTV had no channels\b",
                        r"\bfridge didn't work\b", r"\bfridge barely cooled\b",
                        r"\bkettle was broken\b",
                        r"\bsocket was loose\b", r"\bsocket was sparking\b",
                        r"\bdoor didn't close properly\b", r"\bdoor wouldn't lock\b",
                        r"\block was broken\b", r"\block was sticking\b", r"\bdidn't feel safe\b",
                        r"\bwardrobe was broken\b", r"\bcloset door falling off\b",
                        r"\bfurniture damaged\b", r"\bwalls were damaged\b", r"\broom looks worn out\b",
                    ],
                    "tr": [
                        r"\bklima çalışıyordu\b", r"\bklima sorunsuzdu\b",
                        r"\bısıtma çalışıyordu\b", r"\boda ısısı cihaz olarak iyiydi\b",
                        r"\bher şey çalışıyordu\b",
                        r"\bTV çalışıyordu\b",
                        r"\bsu ısıtıcısı çalışıyordu\b", r"\bbuzdolabı çalışıyordu\b",
                        r"\bkapı düzgün kapanıyordu\b", r"\bkendimizi güvende hissettik\b",
    
                        r"\bklima çalışmıyordu\b", r"\bklima bozuktu\b",
                        r"\bısıtma çalışmıyordu\b",
                        r"\bTV çalışmıyordu\b", r"\bkanal yoktu\b",
                        r"\bbuzdolabı çalışmıyordu\b", r"\bsoğutmuyordu\b",
                        r"\bsu ısıtıcısı bozuktu\b",
                        r"\bpriz gevşekti\b", r"\bpriz kıvılcım atıyordu\b",
                        r"\bkapı tam kapanmıyordu\b", r"\bkapı kilitlenmiyordu\b",
                        r"\bkilit bozuktu\b", r"\bkilit takılıyordu\b", r"\bkendimizi güvende hissetmedik\b",
                        r"\bdolap kırılmıştı\b", r"\bmasa sallanıyordu\b",
                        r"\bduvarlar çok yıpranmıştı\b", r"\boda yorgun görünüyordu\b",
                    ],
                    "ar": [
                        r"\bالمكيف شغال\b", r"\bالتكييف شغال تمام\b",
                        r"\bالتدفئة شغالة\b",
                        r"\bكل الأجهزة شغالة\b",
                        r"\bالتلفزيون شغال\b",
                        r"\bالبراد شغال\b",
                        r"\bالباب يسكّر منيح\b", r"\bحاسين بأمان\b",
    
                        r"\bالمكيف ما بيشتغل\b", r"\bالمكيف خربان\b",
                        r"\bالتدفئة ما اشتغلت\b",
                        r"\bالتلفزيون ما اشتغل\b", r"\bما في قنوات\b",
                        r"\bالبراد ما بيبرد\b", r"\bالبراد خربان\b",
                        r"\bالكاتل خربان\b",
                        r"\bفيش الكهرباء عم يشرّر\b", r"\bفيش مرتخي\b",
                        r"\bالباب ما بيسكّر منيح\b", r"\bالقفل بيعلق\b", r"\bالقفل خربان\b",
                        r"\bما حسّينا بأمان\b",
                        r"\bالخزانة مكسورة\b", r"\bالطاولة مكسورة\b",
                        r"\bالجدران باين عليها تعبانة\b", r"\bشكلها مستهلَك\b",
                    ],
                    "zh": [
                        r"空调很好用", r"空调正常工作",
                        r"暖气正常", r"暖气很好",
                        r"一切设备都能用", r"所有东西都正常",
                        r"电视能看", r"电视没问题",
                        r"冰箱正常", r"烧水壶能用",
                        r"门关得很严", r"门锁很安全", r"感觉很安全",
    
                        r"空调不好用", r"空调不工作", r"空调坏了",
                        r"暖气不工作",
                        r"电视看不了", r"电视没有频道",
                        r"冰箱不制冷", r"冰箱坏了",
                        r"水壶坏了", r"烧水壶不能用",
                        r"插座松", r"插座打火",
                        r"门关不严", r"门锁不上", r"锁坏了",
                        r"衣柜坏了", r"衣柜门要掉了",
                        r"家具很破", r"墙很旧", r"房间看起来很旧",
                    ],
                },
                "aspects": [
                    "ac_working_device", "heating_working_device",
                    "appliances_ok", "tv_working", "fridge_working", "kettle_working",
                    "door_secure",
                    "ac_broken", "heating_broken", "tv_broken", "fridge_broken",
                    "kettle_broken", "socket_danger",
                    "door_not_closing", "lock_broken", "furniture_broken",
                    "room_worn_out",
                ],
            },
    
            "wifi_internet": {
                "display": "Wi-Fi и интернет",
                "patterns": {
                    "ru": [
                        r"\bбыстрый wi[- ]?fi\b", r"\bотличный wi[- ]?fi\b", r"\bwi[- ]?fi работал хорошо\b",
                        r"\bинтернет стабильный\b", r"\bхороший интернет\b", r"\bможно работать удалённо\b",
    
                        r"\bwi[- ]?fi не работал\b", r"\bwi[- ]?fi не ловил\b", r"\bwi[- ]?fi вообще не было\b",
                        r"\bочень медленный интернет\b", r"\bинтернет ужасно медленный\b",
                        r"\bwi[- ]?fi постоянно отваливался\b", r"\bобрывался интернет\b",
                        r"\bсложно подключиться к wi[- ]?fi\b", r"\bпароль от wi[- ]?fi не работал\b",
                        r"\bневозможно было работать\b", r"\bне могли работать удалённо\b",
                    ],
                    "en": [
                        r"\bwifi was fast\b", r"\bwifi was very fast\b",
                        r"\bgood wifi\b", r"\breliable wifi\b", r"\bstable connection\b",
                        r"\binternet was great for work\b",
    
                        r"\bwifi didn't work\b", r"\bwifi was not working\b",
                        r"\bvery slow wifi\b", r"\bunusable wifi\b",
                        r"\bkept disconnecting\b", r"\bkept dropping\b",
                        r"\bhard to connect to wifi\b", r"\bwifi password didn't work\b",
                        r"\bcouldn't work remotely because of the internet\b",
                    ],
                    "tr": [
                        r"\bwifi hızlıydı\b", r"\binternet çok iyiydi\b",
                        r"\bbağlantı stabildi\b", r"\bçalışmak için yeterince iyiydi\b",
    
                        r"\bwifi çalışmıyordu\b", r"\bwifi yoktu\b",
                        r"\binternet çok yavaştı\b",
                        r"\bbağlantı sürekli koptu\b", r"\bsürekli düşüyordu\b",
                        r"\bbağlanmak çok zordu\b", r"\bşifre çalışmadı\b",
                        r"\buzaktan çalışmak imkansızdı\b",
                    ],
                    "ar": [
                        r"\bالواي فاي سريع\b", r"\bالانترنت ممتاز\b",
                        r"\bالاتصال ثابت\b", r"\bفيك تشتغل أونلاين عادي\b",
    
                        r"\bالواي فاي ما اشتغل\b", r"\bما في واي فاي\b",
                        r"\bالانترنت بطيء كتير\b", r"\bمستحيل تستعمله\b",
                        r"\bالواي فاي عم يقطع\b",
                        r"\bما عرفنا ندخل عالواي فاي\b", r"\bالباسورد ما اشتغل\b",
                        r"\bما قدرنا نشتغل عن بُعد\b",
                    ],
                    "zh": [
                        r"wifi很快", r"网速很快",
                        r"网络很稳定", r"上网很稳定",
                        r"可以正常远程工作",
    
                        r"wifi不好用", r"wifi不能用",
                        r"网速很慢", r"基本没网",
                        r"老是掉线", r"一直断线",
                        r"连不上wifi", r"密码连不上",
                        r"没法远程办公",
                    ],
                },
                "aspects": [
                    "wifi_fast", "internet_stable", "good_for_work",
                    "wifi_down", "wifi_slow", "wifi_unstable",
                    "wifi_hard_to_connect", "internet_not_suitable_for_work",
                ],
            },
    
            "tech_noise": {
                "display": "Шум оборудования и инженерки",
                "patterns": {
                    "ru": [
                        r"\bкондиционер очень шумный\b", r"\bгромко гудел кондиционер\b",
                        r"\bшумел холодильник\b", r"\bгромко жужжал холодильник\b", r"\bхолодильник трещит\b",
                        r"\bгул труб\b", r"\bшум в трубах\b", r"\bгул стояка\b",
                        r"\bшумит вентиляция\b", r"\bгудит вентилятор\b",
                        r"\bночью что-то гудело\b", r"\bкакой-то агрегат жужжал всю ночь\b",
                        r"\bне могли уснуть из-за шума техники\b",
    
                        r"\bкондиционер тихий\b", r"\bтихий кондиционер\b",
                        r"\bтихий холодильник\b",
                        r"\bничего не шумело ночью\b",
                    ],
                    "en": [
                        r"\bAC was very loud\b", r"\bair conditioner was noisy\b",
                        r"\bfridge was noisy\b", r"\bfridge was humming loudly\b", r"\bfridge was rattling\b",
                        r"\bpipes were making noise\b", r"\bloud pipes\b", r"\bnoise from the pipes\b",
                        r"\bventilation was loud\b", r"\bfan was loud\b",
                        r"\bsomething was buzzing all night\b", r"\bconstant humming at night\b",
                        r"\bhard to sleep because of the noise from the unit\b",
    
                        r"\bAC was quiet\b", r"\bvery quiet AC\b",
                        r"\bfridge was quiet\b",
                        r"\bno mechanical noise at night\b",
                    ],
                    "tr": [
                        r"\bklima çok gürültülüydü\b", r"\bklima ses yapıyordu\b",
                        r"\bbuzdolabı çok ses yapıyordu\b", r"\bbuzdolabı sürekli uğulduyordu\b",
                        r"\bborulardan ses geliyordu\b", r"\btesisattan ses geliyordu\b",
                        r"\bhavalandırma çok sesliydi\b", r"\bfan çok ses çıkartıyordu\b",
                        r"\bgece boyunca bir şey uğulduyordu\b", r"\bsürekli bir uğultu vardı\b",
                        r"\bbu sesten uyumak zordu\b",
    
                        r"\bklima sessizdi\b", r"\bbuzdolabı sessizdi\b",
                        r"\bgece hiçbir cihaz ses çıkarmıyordu\b",
                    ],
                    "ar": [
                        r"\bالمكيف صوته عالي\b", r"\bالمكيف مزعج\b",
                        r"\bالبراد عم يطن بصوت عالي\b", r"\bالبراد عم يقطع صوت\b",
                        r"\bصوت مواسير\b", r"\bصوت المواسير عالي\b",
                        r"\bالشفاط صوته عالي\b", r"\bالتهوية صوتها عالي\b",
                        r"\bصوت أزيز طول الليل\b", r"\bفي أزاز طول الليل\b",
                        r"\bما قدرنا ننام من صوت الأجهزة\b",
    
                        r"\bالمكيف هادي\b", r"\bالبراد هادي\b",
                        r"\bما في أي صوت بالليل\b",
                    ],
                    "zh": [
                        r"空调声音很大", r"空调很吵",
                        r"冰箱很吵", r"冰箱一直嗡嗡响", r"冰箱一直在响",
                        r"水管有声音", r"管道一直响",
                        r"排风很吵", r"风扇很吵",
                        r"半夜一直有嗡嗡声", r"一晚上都在响",
                        r"吵得睡不着",
    
                        r"空调很安静", r"冰箱很安静",
                        r"晚上没有机器的噪音",
                    ],
                },
                "aspects": [
                    "ac_noisy", "fridge_noisy", "pipes_noise",
                    "ventilation_noisy", "night_mechanical_hum",
                    "tech_noise_sleep_issue",
                    "ac_quiet", "fridge_quiet", "no_tech_noise_night",
                ],
            },
    
            "elevator_infrastructure": {
                "display": "Лифт и доступ с багажом",
                "patterns": {
                    "ru": [
                        r"\bлифт работал\b", r"\bлифт исправен\b", r"\bлифт всегда работал\b",
                        r"\bудобно с чемоданами\b", r"\bлегко подняться с багажом\b",
    
                        r"\bлифт не работал\b", r"\bлифт сломан\b", r"\bлифт отключали\b", r"\bлифт выключен\b",
                        r"\bзастряли в лифте\b", r"\bзависли в лифте\b",
                        r"\bбез лифта очень тяжело с багажом\b", r"\bтащить чемоданы по лестнице\b",
                    ],
                    "en": [
                        r"\bthe elevator was working\b", r"\belevator worked fine\b",
                        r"\beasy with luggage\b", r"\beasy to bring luggage up\b",
    
                        r"\bthe elevator was not working\b", r"\belevator was broken\b",
                        r"\belevator was out of service\b",
                        r"\bwe got stuck in the elevator\b", r"\bwe were stuck in the elevator\b",
                        r"\bno elevator\b", r"\bno working elevator\b",
                        r"\bhad to carry luggage up the stairs\b", r"\bcarrying suitcases upstairs was hard\b",
                    ],
                    "tr": [
                        r"\basansör çalışıyordu\b", r"\basansör sorunsuzdu\b",
                        r"\bbavullarla çıkmak kolaydı\b",
    
                        r"\basansör çalışmıyordu\b", r"\basansör bozuktu\b",
                        r"\basansör kapalıydı\b",
                        r"\basansörde kaldık\b", r"\basansörde sıkıştık\b",
                        r"\basansör yoktu\b",
                        r"\bmerdivenle bavul taşımak çok zordu\b",
                    ],
                    "ar": [
                        r"\bالمصعد شغال\b", r"\bالمصعد تمام\b",
                        r"\bسهل تطلع مع الشنط\b",
    
                        r"\bالمصعد معطل\b", r"\bالمصعد خربان\b", r"\bما في مصعد شغال\b",
                        r"\bعلقنا بالمصعد\b", r"\bحبسنا بالمصعد\b",
                        r"\bاضطرينا نطلع الدرج مع الشنط\b", r"\bصعب كتير مع الشنط\b",
                    ],
                    "zh": [
                        r"电梯正常", r"电梯可以用",
                        r"带行李上去很方便",
    
                        r"电梯坏了", r"电梯不能用", r"电梯停用",
                        r"我们被困在电梯里",
                        r"没有电梯",
                        r"只能扛行李上楼", r"拿行李走楼梯很辛苦",
                    ],
                },
                "aspects": [
                    "elevator_working", "luggage_easy",
                    "elevator_broken", "elevator_stuck",
                    "no_elevator_heavy_bags",
                ],
            },
    
            "lock_security": {
                "display": "Двери и безопасность",
                "patterns": {
                    "ru": [
                        r"\bдверь закрывалась плотно\b", r"\bнормальный замок\b",
                        r"\bчувствовали себя в безопасности\b", r"\bбезопасно хранить вещи\b",
    
                        r"\bдверь не закрывалась нормально\b", r"\bдверь плохо закрывается\b",
                        r"\bзамок заедал\b", r"\bзамок не работал\b", r"\bзамок не закрывался\b",
                        r"\bне чувствовали себя в безопасности\b", r"\bлюбому можно войти\b",
                    ],
                    "en": [
                        r"\bdoor closed properly\b", r"\bthe lock felt secure\b",
                        r"\bwe felt safe\b", r"\bfelt safe leaving our stuff\b",
    
                        r"\bdoor wouldn't close properly\b", r"\bdoor didn't lock\b",
                        r"\block was broken\b", r"\block was sticking\b",
                        r"\bdidn't feel safe\b", r"\bfelt unsafe leaving our belongings\b",
                    ],
                    "tr": [
                        r"\bkapı düzgün kilitleniyordu\b", r"\bkilit sağlamdı\b",
                        r"\bkendimizi güvende hissettik\b",
    
                        r"\bkapı tam kapanmıyordu\b", r"\bkapı kilitlenmiyordu\b",
                        r"\bkilit bozuktu\b", r"\bkilit takılıyordu\b",
                        r"\bkendimizi güvende hissetmedik\b",
                    ],
                    "ar": [
                        r"\bالباب بسكّر منيح\b", r"\bالقفل منيح\b",
                        r"\bحاسين بأمان\b", r"\bمأمنين على أغراضنا\b",
    
                        r"\bالباب ما بيسكّر منيح\b", r"\bالقفل خربان\b", r"\bالقفل بيعلق\b",
                        r"\bما حسّينا بأمان\b", r"\bحاسين إنه أي حدا بفوت\b",
                    ],
                    "zh": [
                        r"门关得很严", r"门锁很安全", r"感觉很安全", r"放心把东西放房间",
    
                        r"门关不严", r"门锁不上", r"锁坏了", r"锁老卡",
                        r"感觉不安全", r"不敢把行李放里面",
                    ],
                },
                "aspects": [
                    "door_secure", "felt_safe",
                    "door_not_closing", "lock_broken", "felt_unsafe",
                ],
            },
    
        },
    },

    "breakfast": {
        "display": "Завтрак и питание",
        "subtopics": {
    
            "food_quality": {
                "display": "Качество и вкус блюд",
                "patterns": {
                    "ru": [
                        r"\bвкусный завтрак\b", r"\bочень вкусный завтрак\b", r"\bзавтрак был вкусн\w*\b",
                        r"\bвсё было свежим\b", r"\bсвежие продукты\b", r"\bсвежее\b",
                        r"\bгорячие блюда\b.*\bгоряч\w*\b", r"\bпода(л|в)и горячее горячим\b",
                        r"\bвкусный кофе\b", r"\bкофе хороший\b", r"\bхороший кофе\b",
    
                        r"\bневкусный завтрак\b", r"\bзавтрак был не очень\b", r"\bзавтрак отвратительный\b",
                        r"\bневкусная еда\b", r"\bпересолено\b", r"\bпережарен\w*\b", r"\bсырая яичниц\w*\b",
                        r"\bнесвежие продукты\b", r"\bиспорченный вкус\b",
                        r"\bвсё холодное\b", r"\bхолодные блюда\b", r"\bхолодная яичниц\w*\b",
                        r"\bкофе ужасный\b", r"\bмерзкий кофе\b", r"\bкофе растворимый и невкусный\b",
                    ],
                    "en": [
                        r"\bbreakfast was delicious\b", r"\bvery tasty breakfast\b", r"\bfood was tasty\b",
                        r"\beverything was fresh\b", r"\bfresh ingredients\b",
                        r"\bhot dishes were actually hot\b",
                        r"\bgood coffee\b", r"\bcoffee was good\b",
    
                        r"\bbreakfast was not good\b", r"\bbreakfast was bad\b", r"\bterrible breakfast\b",
                        r"\bfood tasted bad\b", r"\bovercooked\b", r"\btoo salty\b", r"\bundercooked eggs\b",
                        r"\bnot fresh\b", r"\bfelt old\b", r"\bstale\b",
                        r"\beverything was cold\b", r"\bcold eggs\b",
                        r"\bcoffee was terrible\b", r"\bbad coffee\b", r"\binstant coffee only\b",
                    ],
                    "tr": [
                        r"\bkahvaltı çok lezzetliydi\b", r"\bkahvaltı lezzetliydi\b",
                        r"\bher şey tazeydi\b", r"\btaze ürünler vardı\b",
                        r"\bsıcak yemekler sıcaktı\b",
                        r"\bkahve iyiydi\b", r"\bkahve güzeldi\b",
    
                        r"\bkahvaltı iyi değildi\b", r"\bkahvaltı berbattı\b",
                        r"\byemeklerin tadı kötüydü\b", r"\başırı tuzluydu\b", r"\bçok pişmişti\b", r"\baz pişmişti\b",
                        r"\btaze değildi\b", r"\bbayat gibiydi\b",
                        r"\bher şey soğuktu\b", r"\byumurta soğuktu\b",
                        r"\bkahve çok kötüydü\b", r"\bsadece hazır kahve vardı\b",
                    ],
                    "ar": [
                        r"\bالفطور طيب\b", r"\bالفطور كتير طيب\b", r"\bالأكل طعمو طيب\b",
                        r"\bكلو طازة\b", r"\bأكل طازة\b",
                        r"\bالأكل السخن كان سخن\b",
                        r"\bالقهوة طيبة\b", r"\bالقهوة منيحة\b",
    
                        r"\bالفطور مو طيب\b", r"\bالفطور سيء\b", r"\bفطور بشع\b",
                        r"\bمبين الأكل مو طازة\b", r"\bالأكل بايت\b", r"\bالأكل قديم\b",
                        r"\bالأكل كان بارد\b", r"\bالبيض بارد\b",
                        r"\bالقهوة سيئة\b", r"\bالقهوة زبالة\b", r"\bبس قهوة فورية\b",
                    ],
                    "zh": [
                        r"早餐很好吃", r"早餐很棒", r"东西很好吃",
                        r"食材很新鲜", r"都很新鲜",
                        r"热的菜是热的",
                        r"咖啡很好喝", r"咖啡不错",
    
                        r"早餐不好吃", r"早餐很差", r"早餐太糟了",
                        r"东西不好吃", r"太咸", r"没熟", r"煎蛋没熟",
                        r"不新鲜", r"有点不新鲜", r"像放很久了",
                        r"都是凉的", r"都是冷的",
                        r"咖啡很难喝", r"只有速溶咖啡",
                    ],
                },
                "aspects": [
                    "breakfast_tasty", "food_fresh", "food_hot_served_hot", "coffee_good",
                    "breakfast_bad_taste", "food_not_fresh", "food_cold", "coffee_bad",
                ],
            },
    
            "variety_offering": {
                "display": "Разнообразие и выбор",
                "patterns": {
                    "ru": [
                        r"\bбольшой выбор\b", r"\bогромный выбор\b", r"\bмного всего\b",
                        r"\bразнообразный завтрак\b", r"\bразнообразие блюд\b",
                        r"\bшведский стол отличный\b",
                        r"\bфрукты\b", r"\bовощи\b", r"\bсыры\b", r"\bколбасы\b", r"\bвыпечка\b",
                        r"\bесть и сладкое и несладкое\b",
    
                        r"\bвыбор маленький\b", r"\bразнообразия нет\b",
                        r"\bочень скудный завтрак\b", r"\bскудный выбор\b",
                        r"\bкаждый день одно и то же\b",
                        r"\bничего нормального поесть\b", r"\bесть особо нечего\b",
                    ],
                    "en": [
                        r"\ba lot of choice\b", r"\bgreat selection\b", r"\bbig variety\b",
                        r"\bbuffet was great\b", r"\bgood buffet\b",
                        r"\bfresh fruit\b", r"\bcheese\b", r"\bcold cuts\b", r"\bpastries available\b",
                        r"\bsweet and savory options\b",
    
                        r"\bvery limited choice\b", r"\bpoor selection\b",
                        r"\bsame food every day\b", r"\brepetitive breakfast\b",
                        r"\bnot much to choose from\b",
                        r"\bhard to find anything to eat\b",
                    ],
                    "tr": [
                        r"\bçeşit çok fazlaydı\b", r"\bseçenek çoktu\b",
                        r"\baçık büfe çok iyiydi\b", r"\bbüfe zengindi\b",
                        r"\bmeyve vardı\b", r"\bpeynir çeşitleri vardı\b", r"\bhamur işi vardı\b",
    
                        r"\bseçenek azdı\b", r"\bçeşit azdı\b",
                        r"\bher gün aynı şeyler\b", r"\bkahvaltı çok tekdüze\b",
                        r"\byiyebileceğimiz bir şey bulmak zordu\b",
                    ],
                    "ar": [
                        r"\bفي كتير خيارات\b", r"\bخيارات متنوعة\b",
                        r"\bالبوفيه ممتاز\b", r"\bبوفيه غني\b",
                        r"\bفي فواكه\b", r"\bفي أجبان\b", r"\bفي معجنات\b",
    
                        r"\bما في خيارات\b", r"\bخيارات قليلة\b",
                        r"\bكل يوم نفس الأكل\b",
                        r"\bما في شي ناكله\b", r"\bصعب تلاقي شي تاكله الصبح\b",
                    ],
                    "zh": [
                        r"选择很多", r"种类很多", r"早餐很丰富",
                        r"自助很不错", r"自助很丰盛",
                        r"有水果", r"有奶酪", r"有面包", r"有糕点",
    
                        r"选择很少", r"种类不多",
                        r"每天都一样", r"每天都是同样的东西",
                        r"没什么可以吃的",
                    ],
                },
                "aspects": [
                    "breakfast_variety_good", "buffet_rich", "fresh_fruit_available", "pastries_available",
                    "breakfast_variety_poor", "breakfast_repetitive", "hard_to_find_food",
                ],
            },
    
            "service_dining_staff": {
                "display": "Сервис завтрака (персонал)",
                "patterns": {
                    "ru": [
                        r"\bприветлив(ый|ые) персонал на завтраке\b", r"\bперсонал завтрака очень дружелюбн\w*\b",
                        r"\bперсонал вежливый\b", r"\bперсонал заботливый\b",
                        r"\bбыстро приносили\b", r"\bбыстро пополняли блюда\b",
                        r"\bубирали со стола сразу\b", r"\bчистили стол сразу\b",
    
                        r"\bперсонал неприветливый\b", r"\bгрубо общал\w*\b",
                        r"\bникто не пополнял\b", r"\bничего не добавляли\b", r"\bпустые лотки стояли\b",
                        r"\bперсонал не следит\b", r"\bникто не убирал со стола\b",
                        r"\bпришлось просить несколько раз\b", r"\bигнорировал\w* просьбы\b",
                    ],
                    "en": [
                        r"\bstaff at breakfast were very friendly\b", r"\bbreakfast staff were nice\b",
                        r"\bpolite staff\b", r"\battentive staff\b",
                        r"\bthey refilled everything quickly\b",
                        r"\btable was cleaned immediately\b",
    
                        r"\bunfriendly staff\b", r"\brude staff\b",
                        r"\bthey didn't refill anything\b", r"\bempty trays not refilled\b",
                        r"\bnobody cleaned the tables\b", r"\btables were left dirty\b",
                        r"\bignored us\b", r"\bhad to ask several times\b",
                    ],
                    "tr": [
                        r"\bkahvaltı personeli çok nazikti\b", r"\bpersonel çok güler yüzlüydü\b",
                        r"\bçok yardımcıydılar\b", r"\bhemen tazeliyorlardı\b",
                        r"\bmasaları hemen temizlediler\b",
    
                        r"\bpersonel kaba davrandı\b", r"\bpersonel ilgisizdi\b",
                        r"\bboşalan ürünleri doldurmadılar\b", r"\btepsiler boş kaldı\b",
                        r"\bmasalar temizlenmedi\b", r"\bmasa kirli bırakıldı\b",
                        r"\bdefalarca istemek zorunda kaldık\b",
                    ],
                    "ar": [
                        r"\bالموظفين تبع الفطور كتير لطيفين\b", r"\bالموظفين محترمين\b",
                        r"\bبينتبهوا بسرعة\b", r"\bبيعبيوا الأكل بسرعة\b",
                        r"\bبينضفوا الطاولة دغري\b",
    
                        r"\bالموظفين مو لطيفين\b", r"\bتعامل سيء\b",
                        r"\bما عبّوا الأكل\b", r"\bكل شي فاضي وتركوه\b",
                        r"\bالطاولات وسخة وما حدا عم ينضف\b",
                        r"\bتجاهلونا\b", r"\bبدنا نطلب أكتر من مرة\b",
                    ],
                    "zh": [
                        r"早餐的服务员很友好", r"服务员很热情",
                        r"服务很周到", r"很照顾我们",
                        r"很快就补菜", r"马上补上吃的",
                        r"很快就把桌子收拾好了",
    
                        r"服务员态度不好", r"服务员很不客气",
                        r"没人补菜", r"菜都空了也没人管",
                        r"桌子没人收", r"桌子很脏没人擦",
                        r"我们说了好几次才有人理",
                    ],
                },
                "aspects": [
                    "breakfast_staff_friendly", "breakfast_staff_attentive",
                    "buffet_refilled_quickly", "tables_cleared_fast",
                    "breakfast_staff_rude", "no_refill_food",
                    "tables_left_dirty", "ignored_requests",
                ],
            },
    
            "availability_flow": {
                "display": "Наличие еды и организация завтрака",
                "patterns": {
                    "ru": [
                        r"\bеды хватало всем\b", r"\bвсем хватило\b",
                        r"\bвсё постоянно подносили\b",
                        r"\bместо всегда было\b", r"\bнашли стол без проблем\b",
                        r"\bбез очередей\b", r"\bбез толпы\b",
                        r"\bорганизовано удобно\b",
    
                        r"\bничего не осталось\b", r"\bк \d+.* уже ничего не было\b", r"\bк \d+ утра уже ничего не было\b",
                        r"\bпустые лотки\b", r"\bвсё съели и не обновляли\b",
                        r"\bпришлось ждать еду\b", r"\bждали пока что-то вынесут\b",
                        r"\bнегде сесть\b", r"\bне было свободных столов\b",
                        r"\bбольшая очередь\b", r"\bпришлось стоять в очереди\b",
                    ],
                    "en": [
                        r"\bthere was enough food for everyone\b",
                        r"\bthey kept bringing more food\b",
                        r"\beasy to find a table\b", r"\balways found a table\b",
                        r"\bno line\b", r"\bno long line\b",
                        r"\bbreakfast was well organized\b",
    
                        r"\bnothing left by\b", r"\balmost nothing left\b",
                        r"\bempty trays\b", r"\bbuffet not restocked\b",
                        r"\bwe had to wait for food\b",
                        r"\bno free tables\b", r"\bhard to find a table\b",
                        r"\blong line\b", r"\bhad to queue for breakfast\b",
                    ],
                    "tr": [
                        r"\byemek herkese yetiyordu\b", r"\bherkese yeterince vardı\b",
                        r"\bsürekli tazeliyorlardı\b", r"\byeniden getiriyorlardı\b",
                        r"\bhemen masa bulduk\b", r"\bmasa bulmak kolaydı\b",
                        r"\bsıra yoktu\b", r"\bkuyruk yoktu\b",
                        r"\borganizasyon iyiydi\b",
    
                        r"\b9'da neredeyse hiçbir şey kalmamıştı\b",
                        r"\btepsiler boştu\b", r"\byenilemediler\b",
                        r"\byemek beklemek zorunda kaldık\b",
                        r"\bboş masa yoktu\b", r"\byer bulmak çok zordu\b",
                        r"\buzun kuyruk vardı\b", r"\bkahvaltı için sıra bekledik\b",
                    ],
                    "ar": [
                        r"\bالأكل كان مكفي الكل\b",
                        r"\bكل شوي عم يجيبوا أكل جديد\b",
                        r"\bلقينا طاولة بسرعة\b",
                        r"\bما كان في طوابير\b",
                        r"\bالتنظيم منيح\b",
    
                        r"\bما بقي شي عالبوفيه\b", r"\bكلو مخلص\b",
                        r"\bالصواني فاضية\b", r"\bما عبّوا الأكل\b",
                        r"\bاستنينا ليجيبوا أكل\b",
                        r"\bما في طاولة فاضية\b",
                        r"\bانتظرنا بالدور لحتى نفطر\b",
                    ],
                    "zh": [
                        r"食物够大家吃", r"一直有补菜",
                        r"很容易找到位子", r"很容易有桌子",
                        r"不用排队", r"几乎不用排队",
                        r"早餐安排得很好",
    
                        r"九点多几乎没东西了", r"到九点什么都没了",
                        r"盘子都是空的", r"没人补菜",
                        r"我们还得等他们再拿出来",
                        r"没有空位", r"很难找座位",
                        r"要排很长的队",
                    ],
                },
                "aspects": [
                    "food_enough_for_all", "kept_restocking",
                    "tables_available", "no_queue", "breakfast_flow_ok",
                    "food_ran_out", "not_restocked",
                    "had_to_wait_food", "no_tables_available", "long_queue",
                ],
            },
    
            "cleanliness_breakfast": {
                "display": "Чистота на завтраке",
                "patterns": {
                    "ru": [
                        r"\bчистый зал\b", r"\bв столовой чисто\b", r"\bвсё аккуратно\b",
                        r"\bстолы быстро протирали\b", r"\bсразу убирали посуду\b",
    
                        r"\bгрязные столы\b", r"\bстолы не убирают\b",
                        r"\bгрязная посуда стоит\b", r"\bпосуду не уносят\b",
                        r"\bлипкий стол\b", r"\bвсё в крошках\b",
                        r"\bгрязно возле еды\b", r"\bгрязно у раздачи\b",
                    ],
                    "en": [
                        r"\bdining area was clean\b", r"\beverything was clean and tidy\b",
                        r"\bthey cleaned the tables quickly\b", r"\bthey cleared tables fast\b",
    
                        r"\bdirty tables\b", r"\btables not cleaned\b",
                        r"\bused dishes left everywhere\b",
                        r"\bsticky tables\b", r"\bcrumbs everywhere\b",
                        r"\barea around the buffet was messy\b", r"\bmessy around the food\b",
                    ],
                    "tr": [
                        r"\bkahvaltı alanı temizdi\b", r"\bher yer çok düzenliydi\b",
                        r"\bmasaları hemen temizliyorlardı\b",
    
                        r"\bmasalar kirliydi\b", r"\bmasalar temizlenmiyordu\b",
                        r"\bkirli tabaklar kaldı masada\b",
                        r"\byapış yapış masa\b", r"\bher yerde kırıntı\b",
                        r"\bbüfe etrafı dağınıktı\b", r"\byemek kısmı dağınıktı\b",
                    ],
                    "ar": [
                        r"\bالمكان نظيف\b", r"\bكلشي نضيف\b",
                        r"\bبينضفوا الطاولات بسرعة\b", r"\bبيشيلوا الصحون بسرعة\b",
    
                        r"\bالطاولات وسخة\b", r"\bالطاولة ما نضفوها\b",
                        r"\bصحون وسخة ضلت عالطاولة\b",
                        r"\bالطاولة لزقة\b", r"\bفتافيت بكل مكان\b",
                        r"\bالمنطقة تبع الفطور وسخة\b",
                    ],
                    "zh": [
                        r"用餐区很干净", r"环境很干净",
                        r"很快就收桌子", r"很快就把桌子擦干净",
    
                        r"桌子很脏", r"桌子没人擦",
                        r"盘子都没收", r"桌上都是脏盘子",
                        r"桌子黏黏的", r"到处都是碎屑",
                        r"自助台那边很乱", r"自助台那边很脏",
                    ],
                },
                "aspects": [
                    "breakfast_area_clean", "tables_cleaned_quickly",
                    "dirty_tables", "dirty_dishes_left",
                    "buffet_area_messy",
                ],
            },
    
        },
    },

    "value": {
        "display": "Цена и ценность",
        "subtopics": {
    
            "value_for_money": {
                "display": "Соотношение цена/качество",
                "patterns": {
                    "ru": [
                        r"\bотличное соотношение цена и качеств\w*\b",
                        r"\bза такие деньги просто супер\b", r"\bочень хорошее качество за эти деньги\b",
                        r"\bнедорого для такого уровня\b",
                        r"\bцена оправдана\b", r"\bцена полностью оправдана\b",
    
                        r"\bслишком дорого\b", r"\bдорого для такого уровня\b",
                        r"\bзавышенная цена\b", r"\bне стоит этих денег\b",
                        r"\bне оправдывает цену\b",
                        r"\bза такие деньги ожидаешь лучше\b",
                    ],
                    "en": [
                        r"\bgreat value for money\b", r"\bexcellent value\b",
                        r"\bworth the price\b", r"\bworth the money\b",
                        r"\bgood quality for the price\b", r"\baffordable for this level\b",
    
                        r"\btoo expensive\b", r"\boverpriced\b",
                        r"\bnot worth the price\b", r"\bnot worth the money\b",
                        r"\bpoor value\b", r"\bbad value for money\b",
                        r"\bfor that price we expected more\b",
                    ],
                    "tr": [
                        r"\bfiyat performansı çok iyiydi\b", r"\bfiyatına göre harika\b",
                        r"\bbu fiyata gayet iyi\b", r"\bparasına değer\b",
    
                        r"\bçok pahalıydı\b", r"\bfiyat fazla yüksekti\b",
                        r"\bparasına değmez\b", r"\bbu paraya değmez\b",
                        r"\bbu fiyata daha iyisini beklersin\b",
                    ],
                    "ar": [
                        r"\bالسعر مناسب\b", r"\bالقيمة مقابل السعر ممتازة\b",
                        r"\bعنجد بيسوى هالمصاري\b", r"\bبهاد السعر كتير منيح\b",
    
                        r"\bغالي\b", r"\bغالي عالفاضي\b",
                        r"\bما بيسوى هالمصاري\b", r"\bما بيستاهل السعر\b",
                        r"\bبهاد السعر كنا متوقعين أحسن\b",
                    ],
                    "zh": [
                        r"性价比很高", r"很值这个价", r"这个价位很不错",
                        r"这个价格很合理", r"物有所值",
    
                        r"太贵了", r"价格太高",
                        r"不值这个价", r"性价比太低",
                        r"这个价格应该更好",
                    ],
                },
                "aspects": [
                    "good_value", "worth_the_price", "affordable_for_level",
                    "overpriced", "not_worth_price", "expected_better_for_price",
                ],
            },
    
            "expectations_vs_price": {
                "display": "Ожидания vs цена",
                "patterns": {
                    "ru": [
                        r"\bна фото выглядело лучше\b", r"\bна фото номер лучше\b", r"\bв реальности хуже\b",
                        r"\bожидали выше уровень\b", r"\bожидали уровень повыше\b",
                        r"\bпо описанию думали будет лучше\b",
                        r"\bэто не тянет на такой ценник\b",
                    ],
                    "en": [
                        r"\blooked better in the pictures\b",
                        r"\broom looked better in the photos\b",
                        r"\bwe expected more from the description\b",
                        r"\bnot the level we expected for this price\b",
                    ],
                    "tr": [
                        r"\bfotoğraflarda daha iyi görünüyordu\b",
                        r"\bgerçekte fotoğraflardaki gibi değildi\b",
                        r"\bilanına göre daha düşük seviyede\b",
                        r"\bbu fiyata böyle olmasını beklemezdik\b",
                    ],
                    "ar": [
                        r"\bبالصور أرتب\b", r"\bبالصور شكله أحسن\b",
                        r"\bمش متل الصور\b",
                        r"\bحسب الوصف توقعنا مستوى أعلى\b",
                        r"\bبهالسعر كنا متوقعين مستوى أعلى\b",
                    ],
                    "zh": [
                        r"照片上看起来更好", r"实物没有照片好",
                        r"跟介绍的不一样",
                        r"这个价位我们以为会更好",
                    ],
                },
                "aspects": [
                    "photos_misleading", "quality_below_expectation",
                ],
            },
    
        },
    },
    "location": {
        "display": "Локация и окружение",
        "subtopics": {
    
            "proximity_area": {
                "display": "Расположение и окружение",
                "patterns": {
                    "ru": [
                        r"\bотличное расположение\b", r"\bрасположение супер\b",
                        r"\bцентр рядом\b", r"\bвсё рядом\b",
                        r"\bблизко к метро\b", r"\bрядом метро\b",
                        r"\bрядом магазины\b", r"\bрядом кафе\b", r"\bмного ресторанов рядом\b",
                        r"\bудобно добираться до центра\b", r"\bудобно гулять\b",
    
                        r"\bдалеко от центра\b",
                        r"\bнеудобно добираться\b",
                        r"\bничего нет рядом\b", r"\bнет магазинов рядом\b",
                        r"\bне очень удобное расположение\b",
                    ],
                    "en": [
                        r"\bgreat location\b", r"\bexcellent location\b", r"\bperfect location\b",
                        r"\bvery central\b", r"\bclose to everything\b",
                        r"\bclose to the metro\b", r"\bnear public transport\b",
                        r"\blots of restaurants nearby\b", r"\bshops nearby\b",
    
                        r"\bnot a good location\b", r"\blocation was not convenient\b",
                        r"\bfar from the center\b", r"\bfar from everything\b",
                        r"\bnothing around\b", r"\bnothing nearby\b",
                    ],
                    "tr": [
                        r"\bkonum harikaydı\b", r"\blokasyon mükemmeldi\b",
                        r"\bher yere yakın\b", r"\bmerkeze çok yakın\b",
                        r"\bmetroya yakın\b", r"\btoplu taşımaya yakın\b",
                        r"\byakında market vardı\b", r"\byakında restoranlar vardı\b",
    
                        r"\bkonum pek iyi değildi\b", r"\bkonum uygun değildi\b",
                        r"\bmerkeze uzak\b", r"\bher şeye uzak\b",
                        r"\byakında hiçbir şey yoktu\b",
                    ],
                    "ar": [
                        r"\bالموقع ممتاز\b", r"\bالمكان كتير منيح\b",
                        r"\bقريب من كل شي\b", r"\bقريب من السنتر\b",
                        r"\bقريب عالـ مترو\b", r"\bسهل توصل بالمواصلات\b",
                        r"\bفي مطاعم وسوبرماركت حدّك\b",
    
                        r"\bالموقع مش منيح\b", r"\bالموقع مو مريح\b",
                        r"\bبعيد عن المركز\b",
                        r"\bما في شي حوالي\b", r"\bما في شي قريب\b",
                    ],
                    "zh": [
                        r"位置很好", r"地段很好", r"位置非常方便",
                        r"离市中心很近", r"去哪儿都方便",
                        r"离地铁很近", r"交通方便",
                        r"附近有超市", r"附近有餐厅",
    
                        r"位置不太好", r"位置不方便",
                        r"离市中心很远",
                        r"周围什么都没有", r"附近没什么",
                    ],
                },
                "aspects": [
                    "great_location", "central_convenient", "near_transport", "area_has_food_shops",
                    "location_inconvenient", "far_from_center", "nothing_around",
                ],
            },
    
            "safety_environment": {
                "display": "Ощущение района и безопасность",
                "patterns": {
                    "ru": [
                        r"\bчувствовал\w* себя в безопасности\b", r"\bчувствовали себя в безопасности\b",
                        r"\bспокойный район\b", r"\bтихий район\b",
                        r"\bнормальный подъезд\b", r"\bчистый подъезд\b",
    
                        r"\bрайон стр(е|ё)мный\b", r"\bнебезопасно\b", r"\bнеуютно выходить вечером\b",
                        r"\bподъезд грязный\b", r"\bподъезд ужасный\b",
                        r"\bподозрительные люди\b", r"\bмного пьяных\b",
                        r"\bночью страшно выходить\b",
                    ],
                    "en": [
                        r"\bfelt safe in the area\b", r"\bthe area felt safe\b",
                        r"\bquiet area at night\b",
                        r"\bentrance was clean\b",
    
                        r"\barea felt unsafe\b", r"\bwe didn't feel safe outside\b",
                        r"\bsketchy area\b", r"\bdodgy area\b",
                        r"\bdrunk people outside\b", r"\bpeople hanging around the entrance\b",
                        r"\bentrance was dirty\b",
                    ],
                    "tr": [
                        r"\bbölge güvenliydi\b", r"\bkendimizi güvende hissettik\b",
                        r"\bgece de sakin\b", r"\bgeceleri sessizdi\b",
                        r"\bgiriş temizdi\b",
    
                        r"\bbölge pek güvenli değildi\b", r"\bpek güvenli hissettirmedi\b",
                        r"\bgece dışarı çıkmak pek rahat değildi\b",
                        r"\bgiriş kirliydi\b", r"\bmerdivenler kirliydi\b",
                        r"\betrafta tuhaf tipler vardı\b",
                    ],
                    "ar": [
                        r"\bالمنطقة أمان\b", r"\bحسينا بأمان\b",
                        r"\bالمنطقة هادية بالليل\b",
                        r"\bالمدخل نضيف\b",
    
                        r"\bالمنطقة مو آمنة\b", r"\bما حسّينا بأمان برا\b",
                        r"\bالمنطقة بتخوف\b", r"\bالمنطقة بتخوف شوي\b",
                        r"\bفي ناس مزعجين عالباب\b",
                        r"\bالمدخل مو مريح\b", r"\bالمدخل وسخ\b",
                    ],
                    "zh": [
                        r"附近很安全", r"感觉很安全",
                        r"晚上也很安静",
                        r"入口很干净",
    
                        r"附近不太安全", r"感觉不安全",
                        r"晚上不敢出门",
                        r"门口有人喝酒", r"门口有人闹",
                        r"入口很脏", r"楼道很脏",
                    ],
                },
                "aspects": [
                    "area_safe", "area_quiet_at_night", "entrance_clean",
                    "area_unsafe", "uncomfortable_at_night", "entrance_dirty", "people_loitering",
                ],
            },
    
            "access_navigation": {
                "display": "Доступ и навигация",
                "patterns": {
                    "ru": [
                        r"\bлегко найти\b", r"\bадрес найти легко\b",
                        r"\bинструкции по заселению понятные\b", r"\bпонятные инструкции\b",
                        r"\bнашли вход без проблем\b", r"\bпонятно как зайти в здание\b",
                        r"\bудобно добраться с чемоданом\b",
    
                        r"\bсложно найти вход\b", r"\bтрудно найти вход\b",
                        r"\bне могли найти подъезд\b", r"\bне могли найти домофон\b",
                        r"\bнеочевидный вход\b", r"\bзапутанный вход\b",
                        r"\bочень сложно попасть внутрь\b",
                        r"\bс чемоданами тяжело\b", r"\bнеудобно с багажом\b",
                        r"\bнет нормальной вывески\b",
                    ],
                    "en": [
                        r"\beasy to find\b", r"\beasy to find the entrance\b",
                        r"\bcheck-?in instructions were clear\b",
                        r"\beasy access with luggage\b",
    
                        r"\bhard to find the entrance\b", r"\bdifficult to find the building\b",
                        r"\bconfusing entrance\b", r"\bconfusing access\b",
                        r"\bwe couldn't figure out how to get in\b",
                        r"\bno sign\b", r"\bno signage\b",
                        r"\bnot easy with luggage\b", r"\bhard with suitcases\b",
                    ],
                    "tr": [
                        r"\bbulması kolaydı\b", r"\bgirişi bulmak kolaydı\b",
                        r"\btalimatlar çok açıktı\b",
                        r"\bvalizle girmek rahattı\b",
    
                        r"\bgirişi bulmak zor\b", r"\bbina girişi karışıktı\b",
                        r"\biçeri girmek zor oldu\b", r"\bkapıyı anlamak zordu\b",
                        r"\btabela yoktu\b",
                        r"\bvalizle girmek çok zordu\b",
                    ],
                    "ar": [
                        r"\bسهل نلاقي المدخل\b", r"\bالدخول سهل\b",
                        r"\bالتعليمات واضحة\b",
                        r"\bسهل مع الشنط\b",
    
                        r"\bصعب تلاقي المدخل\b", r"\bما عرفنا من وين نفوت\b",
                        r"\bالدخول معقّد\b", r"\bالمدخل معقّد\b",
                        r"\bما في أي اشارة\b", r"\bما في علامة\b",
                        r"\bصعب مع الشنط\b",
                    ],
                    "zh": [
                        r"很容易找到入口", r"很容易找到地址",
                        r"进楼的指引很清楚",
                        r"带行李进去也还可以",
    
                        r"入口很难找", r"很难找到门",
                        r"不知道怎么进楼", r"进门很麻烦",
                        r"没有指示牌", r"没有标识",
                        r"拿着行李很不方便",
                    ],
                },
                "aspects": [
                    "easy_to_find", "clear_instructions", "luggage_access_ok",
                    "hard_to_find_entrance", "confusing_access", "no_signage", "luggage_access_hard",
                ],
            },
    
        },
    },
    "atmosphere": {
        "display": "Атмосфера и общее впечатление",
        "subtopics": {
    
            "style_feel": {
                "display": "Атмосфера и уют",
                "patterns": {
                    "ru": [
                        r"\bочень уютно\b", r"\bуютная атмосфера\b", r"\bприятная атмосфера\b",
                        r"\bдомашняя атмосфера\b", r"\bкак дома\b",
                        r"\bкрасивый интерьер\b", r"\bстильно\b", r"\bдизайн очень красивый\b",
                        r"\bприятное место\b", r"\bхотелось остаться дольше\b",
    
                        r"\bнеуютно\b", r"\bнеуютная атмосфера\b",
                        r"\bатмосфера .*холодная\b", r"\bхолодная атмосфера\b",
                        r"\bмрачно\b", r"\bугнетающе\b", r"\bдавит\b",
                        r"\bсоветский ремонт\b", r"\bстарый ремонт\b", r"\bвсё уставшее\b",
                        r"\bнет ощущения уюта\b", r"\bне чувствуется уют\b",
                    ],
                    "en": [
                        r"\bvery cozy\b", r"\bsuper cozy\b",
                        r"\bnice atmosphere\b", r"\bpleasant atmosphere\b",
                        r"\bfelt like home\b", r"\bfelt homelike\b",
                        r"\bstylish interior\b", r"\bbeautiful design\b",
                        r"\bgreat vibe\b", r"\bwe loved the vibe\b", r"\bdidn't want to leave\b",
    
                        r"\bnot cozy\b", r"\bnot very cozy\b",
                        r"\bcold atmosphere\b", r"\bdidn't feel welcoming\b",
                        r"\bfelt depressing\b", r"\bfelt gloomy\b",
                        r"\btired looking\b", r"\bdated decor\b", r"\bfelt cheap\b", r"\bfelt soulless\b",
                    ],
                    "tr": [
                        r"\bçok samimi bir atmosfer vardı\b", r"\bçok sıcak bir ortam vardı\b",
                        r"\bçok rahat bir his veriyor\b", r"\bev gibi hissettirdi\b",
                        r"\btasarım çok şıktı\b", r"\bdekorasyon çok güzeldi\b",
                        r"\bortamın havasını çok sevdik\b",
    
                        r"\batmosfer pek sıcak değildi\b", r"\bsoğuk bir his vardı\b",
                        r"\bortam biraz kasvetliydi\b", r"\bdekorasyon eskiydi\b",
                        r"\brahat hissettirmedi\b", r"\bev gibi hissettirmedi\b",
                    ],
                    "ar": [
                        r"\bجو كتير دافئ\b", r"\bجو كتير مريح\b",
                        r"\bبتحس كأنك ببيتك\b",
                        r"\bالديكور حلو\b", r"\bالمكان شكله حلو\b", r"\bستايل حلو\b",
                        r"\bحبّينا الجو\b", r"\bعنجد الجو حلو\b",
    
                        r"\bالجو بارد\b", r"\bما في راحة بالمكان\b", r"\bمش مريح\b",
                        r"\bالشكل قديم\b", r"\bالديكور قديم\b",
                        r"\bالمكان كئيب\b", r"\bشوي كئيب\b",
                    ],
                    "zh": [
                        r"很温馨", r"很舒适", r"很有家的感觉",
                        r"气氛很好", r"氛围很好",
                        r"装修很好看", r"装修很有设计感", r"很有风格",
                        r"很喜欢这里的感觉",
    
                        r"不太温馨", r"没有家的感觉",
                        r"氛围有点冷淡", r"感觉不太舒服",
                        r"装修很旧", r"显得很旧", r"看起来很老旧",
                        r"有点压抑", r"感觉有点压抑",
                    ],
                },
                "aspects": [
                    "cozy_atmosphere", "nice_design", "good_vibe",
                    "not_cozy", "gloomy_feel", "dated_look", "soulless_feel",
                ],
            },
    
            "smell_common_areas": {
                "display": "Запах и ощущение общих зон",
                "patterns": {
                    "ru": [
                        r"\bв коридоре приятно пахнет\b", r"\bприятный запах\b", r"\bсвежо в коридоре\b",
                        r"\bникаких запахов\b",
    
                        r"\bв коридоре воняет\b", r"\bвонь в коридоре\b",
                        r"\bнеприятный запах\b", r"\bзапах канализации\b",
                        r"\bзапах сигарет\b", r"\bпахло сигаретами\b",
                        r"\bпахло сырым\b", r"\bзатхлый запах\b", r"\bзапах плесени\b",
                    ],
                    "en": [
                        r"\bhallway smelled fresh\b", r"\bnice smell in the hallway\b",
                        r"\bno smell\b", r"\bno bad smell\b",
    
                        r"\bhallway smelled bad\b", r"\bbad smell in the hallway\b",
                        r"\bsmelled like cigarettes\b", r"\bcigarette smell everywhere\b",
                        r"\bsewage smell\b", r"\bsmelled like sewage\b",
                        r"\bmusty smell\b", r"\bdamp smell\b",
                    ],
                    "tr": [
                        r"\bkoridor temiz kokuyordu\b", r"\bgüzel kokuyordu\b",
                        r"\bkötü koku yoktu\b",
    
                        r"\bkoridorda kötü koku vardı\b",
                        r"\bsigara kokuyordu\b",
                        r"\blağım gibi kokuyordu\b",
                        r"\bnem kokuyordu\b", r"\brutubet kokuyordu\b",
                    ],
                    "ar": [
                        r"\bريحة حلوة بالممر\b", r"\bالريحة حلوة\b",
                        r"\bما في ريحة خلتنا نضايق\b",
    
                        r"\bريحة مش منيحة بالممر\b", r"\bريحة بشعة\b",
                        r"\bريحة سيجارة\b", r"\bريحة دخان\b",
                        r"\bريحة مجاري\b",
                        r"\bريحة رطوبة\b", r"\bريحة عفن\b",
                    ],
                    "zh": [
                        r"走廊味道很好", r"走廊很清新",
                        r"没有异味",
    
                        r"走廊有味道", r"走廊有臭味",
                        r"都是烟味", r"有烟味",
                        r"下水道的味道",
                        r"霉味", r"潮味",
                    ],
                },
                "aspects": [
                    "fresh_smell_common", "no_bad_smell",
                    "bad_smell_common", "cigarette_smell", "sewage_smell", "musty_smell",
                ],
            },
    
        },
    }
}

# ========= [EXPORT] semantic row factory (вставить НИЖЕ словаря тем) =================
import json, re, hashlib, unicodedata
from datetime import datetime
import pandas as pd
from typing import Any, Dict, List

# --- Базовая очистка текста (без изменения твоих словарей)
_MULTI_WS  = re.compile(r"\s+")
_URL       = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE  = re.compile(r"\b[\w.\-+%]+@[\w.\-]+\.[A-Za-z]{2,}\b")
_PHONE_RE  = re.compile(r"\+?\d[\d\-\s()]{6,}\d")
_HTML_RE   = re.compile(r"<[^>]+>")
_WIFI_UNI  = re.compile(r"\b(wi[\s\-]?fi|вай[\s\-]?[фа]й)\b", re.IGNORECASE)

def _clean_text(x: str) -> str:
    if not x:
        return ""
    t = unicodedata.normalize("NFKC", str(x))
    t = _HTML_RE.sub(" ", t)
    t = _URL.sub(" ", t)
    t = _EMAIL_RE.sub(" ", t)
    t = _PHONE_RE.sub(" ", t)
    t = t.lower()
    t = _WIFI_UNI.sub("wi-fi", t)
    t = _MULTI_WS.sub(" ", t).strip()
    return t

# --- Дата: поддержка dd.mm.yyyy, yyyy-mm-dd, datetime и Excel-serial
def _parse_date_any(x: Any) -> datetime:
    if isinstance(x, datetime):
        return x.replace(tzinfo=None)
    # Excel serial?
    if isinstance(x, (int, float)) and not pd.isna(x):
        try:
            # Excel day 0 = 1899-12-30 (правильный origin для pandas)
            ts = pd.Timestamp("1899-12-30") + pd.to_timedelta(float(x), unit="D")
            return ts.to_pydatetime().replace(tzinfo=None)
        except Exception:
            pass
    # Общий путь
    try:
        return pd.to_datetime(x, errors="raise", dayfirst=True).to_pydatetime().replace(tzinfo=None)
    except Exception:
        # в крайнем случае — сегодня (но лучше логировать в агенте)
        return datetime.utcnow().replace(tzinfo=None)

def _derive_period_keys(dt: datetime) -> Dict[str, str]:
    iso_y, iso_w, _ = dt.isocalendar()
    q = (dt.month - 1)//3 + 1
    return {
        "week_key":    f"{iso_y:04d}-W{iso_w:02d}",
        "month_key":   f"{dt.year:04d}-{dt.month:02d}",
        "quarter_key": f"{dt.year:04d}-Q{q}",
        "year_key":    f"{dt.year:04d}",
    }

def _review_id(dt: datetime, source: str, raw_text: str) -> str:
    base = f"{dt.date().isoformat()}|{source}|{(raw_text or '')[:200]}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

# --- Локальная тональность по подтеме (приоритет: neg strong -> neg soft -> pos strong -> pos soft -> neu)
def _subtopic_sentiment(text: str, lang: str, sub_patterns: Dict[str, List[str]]) -> str:
    lang_key = lang if lang in NEGATIVE_WORDS_STRONG else "en"

    def _has_any(rgx_list, s: str) -> bool:
        for rgx in rgx_list:
            if re.search(rgx, s, flags=re.IGNORECASE):
                return True
        return False

    # окно ±40 символов вокруг совпадений подтемы
    windows: List[str] = []
    pats = sub_patterns.get(lang, []) or sub_patterns.get("ru", []) or sub_patterns.get("en", [])
    for p in pats:
        for m in re.finditer(p, text, flags=re.IGNORECASE):
            s, e = m.span()
            windows.append(text[max(0, s-40): min(len(text), e+40)])
    if not windows:
        windows = [text]

    for chunk in windows:
        if _has_any(NEGATIVE_WORDS_STRONG.get(lang_key, []), chunk): return "neg"
        if _has_any(NEGATIVE_WORDS_SOFT.get(lang_key, []),   chunk): return "neg"
        if _has_any(POSITIVE_WORDS_STRONG.get(lang_key, []), chunk): return "pos"
        if _has_any(POSITIVE_WORDS_SOFT.get(lang_key, []),   chunk): return "pos"
    return "neu"

# --- Главная функция экспорта одной строки семантики
def build_semantic_row(review_raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    На входе: {'date','source','rating_raw','text','lang_hint'}
    На выходе: одна строка для reviews_semantic_raw.
    """
    raw_text = (review_raw.get("text") or "").strip()
    lang_hint = (review_raw.get("lang_hint") or "").strip()
    source    = (review_raw.get("source") or "").strip()

    # язык -> легкая русификация
    lang = normalize_lang(lang_hint, raw_text)
    text_basic = _clean_text(raw_text)
    text_ru = translate_to_ru(text_basic, lang)  # не меняет кириллицу

    rating10 = normalize_rating_to_10(review_raw.get("rating_raw"))
    try:
        overall = detect_sentiment_label(text_ru, lang)  # твоя функция из словарей
    except Exception:
        overall = "neu"

    # темы/аспекты
    topics_all, topics_pos, topics_neg = set(), set(), set()
    # защитимся, если TOPIC_SCHEMA не импортирован
    schema = globals().get("TOPIC_SCHEMA", {}) or {}
    for t_key, t_def in schema.items():
        subs = (t_def or {}).get("subtopics", {}) or {}
        for s_key, s_def in subs.items():
            pats = (s_def or {}).get("patterns", {}) or {}
            # есть совпадение?
            lang_pats = pats.get(lang) or pats.get("ru") or pats.get("en") or []
            if not lang_pats:
                continue
            matched = any(re.search(p, text_ru, flags=re.IGNORECASE) for p in lang_pats)
            if not matched:
                continue
            # локальная тональность подтемы
            sent_local = _subtopic_sentiment(text_ru, lang, pats)
            aspects = (s_def or {}).get("aspects") or [f"{t_key}.{s_key}"]
            aspect = aspects[0]
            topics_all.add(aspect)
            if sent_local == "pos":
                topics_pos.add(aspect)
            elif sent_local == "neg":
                topics_neg.add(aspect)

    # сопряжённые пары (ко-упоминания)
    pair_tags: List[Dict[str, str]] = []
    if topics_all:
        pairs = build_cooccurrence_pairs(sorted(list(topics_all)))
        for a, b in pairs:
            a_pos, a_neg = (a in topics_pos), (a in topics_neg)
            b_pos, b_neg = (b in topics_pos), (b in topics_neg)
            if a_neg and b_neg:
                cat = "systemic_risk"
            elif (a_pos and b_neg) or (b_pos and a_neg):
                cat = "expectations_conflict"
            elif a_pos and b_pos:
                cat = "loyalty_driver"
            else:
                continue
            pair_tags.append({"a": a, "b": b, "cat": cat})

    # цитаты (1–3 предложения)
    quotes = []
    for part in re.split(r"[.!?…]+[\s\n]+", text_ru):
        p = part.strip()
        if 15 <= len(p) <= 300:
            quotes.append(p)
        if len(quotes) >= 3:
            break

    dt = _parse_date_any(review_raw.get("date"))
    per = _derive_period_keys(dt)
    rid = _review_id(dt, source, raw_text)

    return {
        "review_id": rid,
        "date": dt.date().isoformat(),
        "week_key": per["week_key"],
        "month_key": per["month_key"],
        "quarter_key": per["quarter_key"],
        "year_key": per["year_key"],
        "source": source,
        "rating10": rating10,
        "sentiment_overall": overall,
        "topics_pos": json.dumps(sorted(list(topics_pos)), ensure_ascii=False),
        "topics_neg": json.dumps(sorted(list(topics_neg)), ensure_ascii=False),
        "topics_all": json.dumps(sorted(list(topics_all)), ensure_ascii=False),
        "pair_tags": json.dumps(pair_tags, ensure_ascii=False),
        "quote_candidates": json.dumps(quotes, ensure_ascii=False),
    }
# ========= [/EXPORT] =================================================================



import re
from typing import Dict, Any, List

# 1. Человекочитаемые подписи для аспектов.
#    Всё, что мы точно хотим красиво выводить в отчёте.
#    Если какого-то аспекта здесь нет, мы сгенерируем fallback из subtopic_display.
ASPECT_LABEL_OVERRIDES: Dict[str, str] = {
    # Техническое состояние / вода
    "hot_water_ok": "Стабильная горячая вода",
    "no_hot_water": "Не было горячей воды / перебои с горячей водой",
    "water_pressure_ok": "Хорошее давление воды",
    "weak_pressure": "Слабый напор воды",
    "shower_ok": "Душ исправен",
    "shower_broken": "Проблемы с душем",
    "leak_water": "Течи воды / подтекает",
    "bathroom_flooding": "Вода на полу после душа",
    "drain_clogged": "Плохой слив / засор",
    "drain_smell": "Неприятный запах из слива/канализации",

    # Оборудование номера
    "ac_working_device": "Кондиционер работает",
    "ac_broken": "Кондиционер не работает / сломан",
    "heating_working_device": "Отопление работает",
    "heating_broken": "Нет отопления / не работает обогрев",
    "tv_working": "Телевизор исправен",
    "tv_broken": "Телевизор не работает",
    "fridge_working": "Холодильник работает",
    "fridge_broken": "Холодильник не работает / не холодит",
    "kettle_working": "Чайник исправен",
    "kettle_broken": "Чайник не работает",
    "appliances_ok": "Техника и оснащение исправны",
    "socket_danger": "Опасные розетки (люфт, искрит)",
    "door_secure": "Дверь закрывается надёжно",
    "door_not_closing": "Дверь плохо закрывается / не фиксируется",
    "lock_broken": "Проблемы с замком / не закрывается",
    "felt_safe": "Гости чувствуют себя в безопасности",
    "felt_unsafe": "Гости не чувствуют себя в безопасности",
    "furniture_broken": "Сломанная мебель / износ",
    "room_worn_out": "Номер визуально уставший / требует ремонта",

    # Wi-Fi
    "wifi_fast": "Быстрый Wi-Fi",
    "internet_stable": "Стабильный интернет",
    "good_for_work": "Интернет подходит для удалённой работы",
    "wifi_down": "Wi-Fi не работал",
    "wifi_slow": "Медленный Wi-Fi",
    "wifi_unstable": "Нестабильный Wi-Fi / обрывы",
    "wifi_hard_to_connect": "Сложно подключиться к Wi-Fi / пароль не работает",
    "internet_not_suitable_for_work": "Интернет не подходит для удалённой работы",

    # Шум инженерки
    "ac_quiet": "Кондиционер работает тихо",
    "ac_noisy": "Шумный кондиционер",
    "fridge_quiet": "Холодильник не шумит",
    "fridge_noisy": "Шумный холодильник",
    "pipes_noise": "Шум в трубах / стояке",
    "ventilation_noisy": "Шумная вентиляция/вентилятор",
    "night_mechanical_hum": "Гул оборудования ночью",
    "tech_noise_sleep_issue": "Шум техники мешал спать",
    "no_tech_noise_night": "Ночью тихо (нет шума техники)",

    # Лифт
    "elevator_working": "Лифт работает",
    "elevator_broken": "Лифт не работает",
    "elevator_stuck": "Гости застревали в лифте",
    "luggage_easy": "Удобно с багажом (лифт/доступ)",
    "no_elevator_heavy_bags": "Тяжело с багажом из-за лифта/лестниц",

    # Завтрак — вкус и качество
    "breakfast_tasty": "Завтрак вкусный",
    "breakfast_bad_taste": "Невкусный завтрак / проблемы со вкусом",
    "food_fresh": "Свежие продукты",
    "food_not_fresh": "Несвежие продукты / вкус «вчерашний»",
    "food_hot_served_hot": "Горячее подано горячим",
    "food_cold": "Горячее подано холодным",
    "coffee_good": "Кофе нормального качества",
    "coffee_bad": "Плохой кофе",

    # Завтрак — выбор
    "breakfast_variety_good": "Хороший выбор на завтраке",
    "buffet_rich": "Полноценный/богатый шведский стол",
    "fresh_fruit_available": "Есть свежие фрукты",
    "pastries_available": "Есть выпечка/сладкое",
    "breakfast_variety_poor": "Слабый выбор на завтраке",
    "breakfast_repetitive": "Одно и то же каждый день",
    "hard_to_find_food": "Сложно найти что поесть",

    # Завтрак — сервис персонала
    "breakfast_staff_friendly": "Персонал завтрака дружелюбный",
    "breakfast_staff_attentive": "Персонал завтрака внимательный/заботливый",
    "buffet_refilled_quickly": "Блюда оперативно пополняют",
    "tables_cleared_fast": "Столы быстро убирают",
    "breakfast_staff_rude": "Персонал завтрака неприветливый/грубый",
    "no_refill_food": "Еду не пополняют, подносы пустые",
    "tables_left_dirty": "Столы остаются грязными",
    "ignored_requests": "Просьбы гостей игнорировали",

    # Завтрак — организация потока
    "food_enough_for_all": "Еды хватает всем",
    "kept_restocking": "Постоянно пополняют еду",
    "tables_available": "Есть свободные столы",
    "no_queue": "Без очередей/толпы",
    "breakfast_flow_ok": "Завтрак организован удобно",
    "food_ran_out": "К моменту прихода еды почти не осталось",
    "not_restocked": "Пустые лотки не пополняли",
    "had_to_wait_food": "Приходилось ждать, пока вынесут еду",
    "no_tables_available": "Некуда сесть на завтраке",
    "long_queue": "Очередь/давка на завтраке",

    # Завтрак — чистота
    "breakfast_area_clean": "Зона завтрака чистая/аккуратная",
    "tables_cleaned_quickly": "Столы на завтраке быстро протирают",
    "dirty_tables": "Грязные/липкие столы",
    "dirty_dishes_left": "Грязная посуда не убрана",
    "buffet_area_messy": "Грязно/неаккуратно у раздачи",

    # Цена
    "good_value": "Хорошее соотношение цена/качество",
    "worth_the_price": "Стоит своих денег",
    "affordable_for_level": "Недорого для такого уровня",
    "overpriced": "Слишком дорого для такого уровня",
    "not_worth_price": "Не стоит своих денег",
    "expected_better_for_price": "Ожидали выше уровень за эту цену",

    # Ожидания vs цена
    "photos_misleading": "Фото/описание обещали больше, чем в реальности",
    "quality_below_expectation": "Уровень ниже ожидаемого за эти деньги",

    # Локация / окружение
    "great_location": "Отличное расположение",
    "central_convenient": "Близко к центру / всё рядом",
    "near_transport": "Удобно по транспорту (метро/транспорт рядом)",
    "area_has_food_shops": "Рядом кафе и магазины",
    "location_inconvenient": "Неудобная локация",
    "far_from_center": "Далеко от центра/основных точек",
    "nothing_around": "Вокруг почти ничего нет",

    # Безопасность района / подъезд
    "area_safe": "В районе безопасно / спокойно",
    "area_quiet_at_night": "Тихо в районе ночью",
    "entrance_clean": "Чистый/нормальный вход",
    "area_unsafe": "Небезопасный район / стрёмно",
    "uncomfortable_at_night": "Неуютно выходить вечером / страшно ночью",
    "entrance_dirty": "Грязный/неприятный вход/подъезд",
    "people_loitering": "Неприятный контингент у входа",

    # Навигация/доступ
    "easy_to_find": "Легко найти адрес и вход",
    "clear_instructions": "Понятные инструкции по доступу и заселению",
    "luggage_access_ok": "Удобно заходить с багажом с улицы",
    "hard_to_find_entrance": "Сложно найти вход / адрес / подъезд",
    "confusing_access": "Неочевидно, как попасть внутрь",
    "no_signage": "Нет нормальной навигации/вывесок",
    "luggage_access_hard": "Тяжело с багажом с улицы/по лестнице",

    # Атмосфера / вайб
    "cozy_atmosphere": "Уютно, домашняя атмосфера",
    "nice_design": "Красивый / стильный интерьер",
    "good_vibe": "Приятная атмосфера, «хочется остаться»",
    "not_cozy": "Неуютно / холодная атмосфера",
    "gloomy_feel": "Мрачно / давит / депрессивно",
    "dated_look": "Уставший / старый визуально вид",
    "soulless_feel": "Без души / нет ощущения «добро пожаловать»",

    # Запахи общих зон
    "fresh_smell_common": "Приятный запах / свежо в коридорах",
    "no_bad_smell": "Нет неприятного запаха в общих зонах",
    "bad_smell_common": "Неприятный запах в коридорах",
    "cigarette_smell": "Запах сигарет в общих зонах",
    "sewage_smell": "Запах канализации в общих зонах",
    "musty_smell": "Запах сырости / плесени / затхлости",
}


# 2. Эвристика полярности аспекта.
#    Возвращает "pos" / "neg" / "neutral".
def infer_polarity(aspect_code: str) -> str:
    positive_markers = [
        "ok", "good", "fast", "friendly", "attentive",
        "clean", "cleaned", "refilled", "enough", "available",
        "quiet", "safe", "secure", "working", "tasty",
        "fresh", "hot_served_hot", "great", "worth", "value",
        "easy", "cozy", "nice", "flow_ok", "no_queue",
        "no_bad_smell", "kept_restocking", "tables_available",
        "luggage_access_ok", "area_safe", "entrance_clean",
        "central", "near_transport", "good_vibe",
    ]
    negative_markers = [
        "no_",  # e.g. no_hot_water / no_refill_food / no_tables_available
        "not_", "slow", "dirty", "broken", "unsafe",
        "hard", "hard_to", "unsafe", "unfriendly", "rude",
        "smell", "smelly", "bad", "overpriced", "ran_out",
        "repetitive", "poor", "cold", "leak", "clog",
        "flood", "noisy", "stuck", "queue", "long_queue",
        "uncomfortable", "confusing", "gloomy", "dated",
        "soulless", "people_loitering", "luggage_access_hard",
        "area_unsafe", "entrance_dirty", "far_from_center",
        "nothing_around", "not_worth_price", "expected_better",
    ]

    a = aspect_code.lower()

    # если начинается с no_ или contains obvious negative first -> считаем отрицательно
    if any(marker in a for marker in negative_markers):
        return "neg"

    # если содержит позитивные маркеры -> считаем позитивно
    if any(marker in a for marker in positive_markers):
        return "pos"

    # по умолчанию
    return "neutral"


# 3. Генерация реестра аспектов на основе TOPIC_SCHEMA и ASPECT_LABEL_OVERRIDES
def build_aspect_registry(topic_schema: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Превращает TOPIC_SCHEMA в плоский словарь аспектов.
    Ключ словаря = код аспекта (например "wifi_unstable").
    Значение = метаданные: тема, подтема, красивая подпись, полярность.
    """
    registry: Dict[str, Dict[str, Any]] = {}

    for topic_key, topic_block in topic_schema.items():
        topic_display = topic_block.get("display", topic_key)

        subtopics: Dict[str, Any] = topic_block.get("subtopics", {})
        for sub_key, sub_block in subtopics.items():
            sub_display = sub_block.get("display", sub_key)

            aspects: List[str] = sub_block.get("aspects", [])
            for aspect_code in aspects:
                label = ASPECT_LABEL_OVERRIDES.get(
                    aspect_code,
                    f"{sub_display}: {aspect_code}"
                )
                registry[aspect_code] = {
                    "topic_key": topic_key,
                    "topic_display": topic_display,
                    "subtopic_key": sub_key,
                    "subtopic_display": sub_display,
                    "label": label,
                    "polarity": infer_polarity(aspect_code),
                }

    return registry


# 4. Вспомогательная функция — сгруппировать аспекты в управленческие группы
#    Это пригодится, чтобы потом в отчёте выводить не просто список тегов,
#    а блоками по зонам ответственности (Завтрак, Wi-Fi, Вода, Локация и т.д.).
def group_aspects_by_topic(registry: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Возвращает структуру вида:
    {
        "breakfast": {
            "topic_display": "Завтрак и питание",
            "aspects": ["breakfast_tasty", "food_not_fresh", ...]
        },
        "tech_state": {
            "topic_display": "Техническое состояние и инфраструктура",
            "aspects": ["wifi_unstable", "no_hot_water", ...]
        },
        ...
    }
    """
    grouped: Dict[str, Dict[str, Any]] = {}
    for aspect_code, meta in registry.items():
        k = meta["topic_key"]
        if k not in grouped:
            grouped[k] = {
                "topic_display": meta["topic_display"],
                "aspects": [],
            }
        grouped[k]["aspects"].append(aspect_code)
    return grouped


# 5. Агрегация по аспектам для заданного периода.
#    Предполагаем, что у нас уже есть "взорванный" датафрейм отзывов,
#    где каждая строка = один (review_id, aspect_code, sentiment_label, period_key).
#
#    sentiment_label должен быть 'pos'/'neg'/'neu' после тонального анализа.
#
#    Возвращаем сводку:
#    - сколько упоминаний аспекта,
#    - доля негатива и позитива,
#    - чтобы потом ранжировать и говорить
#      «Топ факторов негатива недели» и «Ключевые плюсы недели».
#

# ====== [ADD BELOW TOPIC_SCHEMA] ==================================================
# Лёгкая нормализация текста и ID/даты + фабрика строки семантики

import json, hashlib, unicodedata
from datetime import datetime

WI_FI_UNIFY = re.compile(r"\b(wi[\s\-]?fi|вай[\s\-]?фай)\b", re.IGNORECASE)
MULTISPACE = re.compile(r"\s+")
URL = re.compile(r"https?://\S+|www\.\S+")
EMAIL = re.compile(r"\b[\w.\-+%]+@[\w.\-]+\.[A-Za-z]{2,}\b")
PHONE = re.compile(r"\+?\d[\d\-\s()]{6,}\d")
HTML = re.compile(r"<[^>]+>")

def _normalize_text_basic_v2(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = HTML.sub(" ", t)
    t = URL.sub(" ", t)
    t = EMAIL.sub(" ", t)
    t = PHONE.sub(" ", t)
    t = t.lower()
    t = WI_FI_UNIFY.sub("wi-fi", t)
    t = MULTISPACE.sub(" ", t).strip()
    return t

def _parse_date_to_datetime_v2(date_raw: Any) -> datetime:
    """
    Парсим dd.mm.yyyy / yyyy-mm-dd / dd/mm/yyyy и Excel-serial (float/int).
    Всегда dayfirst=True.
    """
    if isinstance(date_raw, datetime):
        return date_raw.replace(tzinfo=None)
    # Excel serial number?
    if isinstance(date_raw, (int, float)) and not pd.isna(date_raw):
        try:
            # Excel считает от 1899-12-30
            return (pd.Timestamp("1899-12-30") + pd.to_timedelta(float(date_raw), unit="D")).to_pydatetime()
        except Exception:
            pass
    try:
        return pd.to_datetime(date_raw, dayfirst=True, errors="raise").to_pydatetime().replace(tzinfo=None)
    except Exception:
        return datetime.utcnow().replace(tzinfo=None)

def _derive_period_keys_v2(dt: datetime) -> Dict[str, str]:
    iso_year, iso_week, _ = dt.isocalendar()
    q = (dt.month - 1) // 3 + 1
    return {
        "week_key": f"{iso_year:04d}-W{iso_week:02d}",
        "month_key": f"{dt.year:04d}-{dt.month:02d}",
        "quarter_key": f"{dt.year:04d}-Q{q}",
        "year_key": f"{dt.year:04d}",
    }

def _build_review_id_v2(dt: datetime, source: str, raw_text: str) -> str:
    base = f"{dt.date().isoformat()}|{source}|{(raw_text or '')[:200]}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

# Негаторы для локальной оценки подтем
_RU_NEG = r"(не|без)\s+"
_EN_NEG = r"(not|no|without|never)\s+"
_TR_NEG = r"(değil|yok|olm[aeı]yor)\s+"

def _negated_v2(text: str, lang: str, term_regex: str) -> bool:
    if lang == "ru":
        return re.search(rf"{_RU_NEG}\w{{0,3}}(?:{term_regex})", text, flags=re.IGNORECASE) is not None
    if lang == "tr":
        return re.search(rf"{_TR_NEG}\w{{0,3}}(?:{term_regex})", text, flags=re.IGNORECASE) is not None
    return re.search(rf"{_EN_NEG}\w{{0,3}}(?:{term_regex})", text, flags=re.IGNORECASE) is not None

def _detect_subtopic_sentiment_v2(text: str, lang: str, patterns: Dict[str, List[str]]) -> str:
    """
    Смотрим окно ±40 символов вокруг совпадений подтемы и применяем приоритет:
    strong-neg -> soft-neg -> strong-pos -> soft-pos -> neu (с учётом negation).
    Используем уже имеющиеся лексиконы из файла (POSITIVE_WORDS_*, NEGATIVE_WORDS_*).
    """
    lang_key = lang if lang in NEGATIVE_WORDS_STRONG else "en"
    spans: List[str] = []
    pats = patterns.get(lang, []) or patterns.get("en", [])
    for pat in pats:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            s, e = m.span()
            spans.append(text[max(0, s-40): min(len(text), e+40)])
    if not spans:
        spans = [text]

    def has(rgx_list, chunk) -> bool:
        for rgx in rgx_list:
            if re.search(rgx, chunk, flags=re.IGNORECASE):
                if _negated_v2(chunk, lang, rgx):
                    # «не хорошо» трактуем как негатив
                    return False
                return True
        return False

    for ch in spans:
        if has(NEGATIVE_WORDS_STRONG.get(lang_key, []), ch): return "neg"
        if has(NEGATIVE_WORDS_SOFT.get(lang_key, []), ch):   return "neg"
        if has(POSITIVE_WORDS_STRONG.get(lang_key, []), ch): return "pos"
        if has(POSITIVE_WORDS_SOFT.get(lang_key, []), ch):   return "pos"
        for rgx in NEUTRAL_WORDS.get(lang_key, []):
            if re.search(rgx, ch, flags=re.IGNORECASE): return "neu"
    return "neu"

def build_semantic_row(review_raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    1 отзыв -> 1 нормализованная строка для reviews_semantic_raw:
    { review_id, date, week_key, month_key, quarter_key, year_key,
      source, rating10, sentiment_overall,
      topics_pos, topics_neg, topics_all, pair_tags, quote_candidates }
    """
    raw_text = (review_raw.get("text") or "").strip()
    lang_hint = (review_raw.get("lang_hint") or "").strip()
    source    = (review_raw.get("source") or "").strip()

    lang_norm = normalize_lang(lang_hint, raw_text)
    text_clean = _normalize_text_basic_v2(raw_text)
    # псевдо-перевод оставляем как есть (в модуле он безопасный)
    _ = translate_to_ru(text_clean, lang_norm)

    rating10 = normalize_rating_to_10(review_raw.get("rating_raw"))
    # общая тональность как была (без падения на None)
    try:
        sentiment_overall = detect_sentiment_label(text_clean, lang_norm)
    except Exception:
        sentiment_overall = "neu"

    # темы/аспекты
    topics_all: set[str] = set()
    topics_pos: set[str] = set()
    topics_neg: set[str] = set()

    # соберём флаги для каждой подтемы по схеме
    for topic_key, tdef in TOPIC_SCHEMA.items():
        for sub_key, sdef in tdef.get("subtopics", {}).items():
            patterns = sdef.get("patterns", {})
            pats = patterns.get(lang_norm) or patterns.get("en") or []
            if not pats:
                continue
            # совпадение есть?
            matched = any(re.search(p, text_clean, flags=re.IGNORECASE) for p in pats)
            if not matched:
                continue

            sent_local = _detect_subtopic_sentiment_v2(text_clean, lang_norm, patterns)
            aspects = sdef.get("aspects") or [f"{topic_key}.{sub_key}"]
            # нормализуем ключ аспекта
            aspect = aspects[0]
            topics_all.add(aspect)
            if sent_local == "pos":
                topics_pos.add(aspect)
            elif sent_local == "neg":
                topics_neg.add(aspect)

    # пары (ко-упоминания) на основе уже имеющейся функции
    pair_tags: List[Dict[str, str]] = []
    if topics_all:
        pairs = build_cooccurrence_pairs(sorted(list(topics_all)))
        for a, b in pairs:
            # классификация связок
            a_pos, a_neg = (a in topics_pos), (a in topics_neg)
            b_pos, b_neg = (b in topics_pos), (b in topics_neg)
            if a_neg and b_neg:
                cat = "systemic_risk"
            elif (a_pos and b_neg) or (b_pos and a_neg):
                cat = "expectations_conflict"
            elif a_pos and b_pos:
                cat = "loyalty_driver"
            else:
                continue
            pair_tags.append({"a": a, "b": b, "cat": cat})

    # цитаты — 1–3 предложения из очищенного текста
    quotes = []
    for part in re.split(r"[.!?…]+[\s\n]+", text_clean):
        p = part.strip()
        if 15 <= len(p) <= 300:
            quotes.append(p)
        if len(quotes) >= 3:
            break

    # дата/периоды/id
    dt = _parse_date_to_datetime_v2(review_raw.get("date"))
    per = _derive_period_keys_v2(dt)
    review_id = _build_review_id_v2(dt, source, raw_text)

    return {
        "review_id": review_id,
        "date": dt.date().isoformat(),
        "week_key": per["week_key"],
        "month_key": per["month_key"],
        "quarter_key": per["quarter_key"],
        "year_key": per["year_key"],
        "source": source,
        "rating10": rating10,
        "sentiment_overall": sentiment_overall,
        "topics_pos": json.dumps(sorted(list(topics_pos)), ensure_ascii=False),
        "topics_neg": json.dumps(sorted(list(topics_neg)), ensure_ascii=False),
        "topics_all": json.dumps(sorted(list(topics_all)), ensure_ascii=False),
        "pair_tags": json.dumps(pair_tags, ensure_ascii=False),
        "quote_candidates": json.dumps(quotes, ensure_ascii=False),
    }
# ====== [/ADD] ====================================================================


def aggregate_aspects_for_period(df_exploded,
                                 period_key_col: str,
                                 current_period_key: str,
                                 registry: Dict[str, Dict[str, Any]]):
    """
    df_exploded: DataFrame со столбцами:
        - aspect_code : str
        - sentiment   : {'pos','neg','neu'}
        - {period_key_col} : например 'week_key' / '2025-W42'
        - review_id   : id отзыва (для подсчёта уникальных отзывов по аспекту)

    period_key_col: имя столбца периода для группировки, например 'week_key'
    current_period_key: значение периода, например '2025-W42'
    registry: выход из build_aspect_registry()

    Возвращает DataFrame с колонками:
        aspect_code
        mentions_reviews  - уникальных отзывов, где аспект встречался
        pos_share         - доля позитивных употреблений
        neg_share         - доля негативных употреблений
        topic_display
        subtopic_display
        label
        polarity_hint     - polarity из реестра ('pos'/'neg'/'neutral')
    """

    import pandas as pd

    cur = df_exploded[df_exploded[period_key_col] == current_period_key].copy()

    # считаем по отзывам (уникальным), а не по количеству предложений
    grp = (
        cur.groupby(["aspect_code", "sentiment", "review_id"])
           .size()
           .reset_index()
    )

    # теперь считаем, сколько отзывов упомянули аспект хотя бы один раз
    mentions = (
        grp.groupby(["aspect_code", "sentiment"])["review_id"]
           .nunique()
           .reset_index(name="review_mentions")
    )

    # сводим в wide: pos / neg / neu отдельно
    pivot = mentions.pivot_table(
        index="aspect_code",
        columns="sentiment",
        values="review_mentions",
        fill_value=0,
        aggfunc="sum",
    )

    pivot = pivot.rename(columns={
        "pos": "mentions_pos",
        "neg": "mentions_neg",
        "neu": "mentions_neu",
    })

    pivot["mentions_total"] = (
        pivot.get("mentions_pos", 0)
        + pivot.get("mentions_neg", 0)
        + pivot.get("mentions_neu", 0)
    )

    # доля позитива / негатива от всех упоминаний аспекта
    pivot["pos_share"] = pivot["mentions_pos"] / pivot["mentions_total"].replace(0, 1)
    pivot["neg_share"] = pivot["mentions_neg"] / pivot["mentions_total"].replace(0, 1)

    pivot = pivot.reset_index()

    # присоединяем человекочитаемые подписи и метаданные
    meta_df = []
    for aspect_code, meta in registry.items():
        meta_df.append({
            "aspect_code": aspect_code,
            "topic_display": meta["topic_display"],
            "subtopic_display": meta["subtopic_display"],
            "label": meta["label"],
            "polarity_hint": meta["polarity"],
        })
    meta_df = pd.DataFrame(meta_df)

    out = pivot.merge(meta_df, on="aspect_code", how="left")

    # сортируем по полной значимости аспекта в этом периоде
    out = out.sort_values(
        by=["mentions_total", "mentions_neg", "mentions_pos"],
        ascending=[False, False, False],
        ignore_index=True,
    )

    return out

import re
import uuid
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Tuple


# -------------------------------------------------
# 1. Нормализация языка
# -------------------------------------------------

LANG_MAP_EXPLICIT = {
    "ru": "ru", "ru-ru": "ru", "russian": "ru",
    "ua": "ru", "uk": "ru", "ukrainian": "ru",  # частая путаница кириллицы как "украинский"
    "en": "en", "en-us": "en", "en-gb": "en", "english": "en",
    "tr": "tr", "turkish": "tr",
    "ar": "ar", "arabic": "ar",
    "zh": "zh", "cn": "zh", "zh-cn": "zh", "chinese": "zh", "zh-hans": "zh",
    "zh-hant": "zh", "zh-tw": "zh",
}

CYRILLIC_RE = re.compile(r"[а-яё]", re.IGNORECASE)
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
HANZI_RE = re.compile(r"[\u4e00-\u9fff]")
TURKISH_HINT_RE = re.compile(r"[çğıöşü]", re.IGNORECASE)

def normalize_lang(raw_code: str, text: str) -> str:
    """
    Приводим язык к одному из {ru,en,tr,ar,zh,other}
    Используем сначала код из источника, потом эвристику по алфавиту.
    """
    if raw_code:
        c = raw_code.strip().lower()
        if c in LANG_MAP_EXPLICIT:
            return LANG_MAP_EXPLICIT[c]

    t = text or ""
    # эвристики
    if CYRILLIC_RE.search(t):
        return "ru"
    if ARABIC_RE.search(t):
        return "ar"
    if HANZI_RE.search(t):
        return "zh"
    if TURKISH_HINT_RE.search(t):
        return "tr"

    # если есть явные английские слова типа "the", "and", "was", "great"
    if re.search(r"\b(the|and|was|were|is|are|very|really|great|good|bad|price)\b", t.lower()):
        return "en"

    return "other"


# -------------------------------------------------
# 2. Тональность (sentiment)
# -------------------------------------------------

# Регэкспы для позитивной / негативной окраски.
# Мы не пытаемся грамматически распарсить фразу,
# просто считаем "кол-во попаданий сильных позитивных и негативных маркеров".
SENTIMENT_LEXICON: Dict[str, Dict[str, List[re.Pattern]]] = {
    "ru": {
        "pos": [
            re.compile(r"\bотличн\w+\b", re.I),
            re.compile(r"\bпрекрасн\w+\b", re.I),
            re.compile(r"\bсупер\b", re.I),
            re.compile(r"\bклассн\w+\b", re.I),
            re.compile(r"\bвеликолепн\w+\b", re.I),
            re.compile(r"\bпонравил\w+\b", re.I),
            re.compile(r"\bочень вкусн\w+\b", re.I),
            re.compile(r"\bочень уютн\w+\b", re.I),
            re.compile(r"\bдружелюбн\w+\b", re.I),
            re.compile(r"\bвежлив\w+\b", re.I),
            re.compile(r"\bчисто\b", re.I),
            re.compile(r"\bсвежо\b", re.I),
            re.compile(r"\bуютн\w+\b", re.I),
            re.compile(r"\bстильн\w+\b", re.I),
            re.compile(r"\bкрасив\w+\b", re.I),
            re.compile(r"\bвсё было хорошо\b", re.I),
            re.compile(r"\bвсё понравил\w+\b", re.I),
            re.compile(r"\bрекомендую\b", re.I),
            re.compile(r"\bвернусь\b", re.I),
        ],
        "neg": [
            re.compile(r"\bужасн\w+\b", re.I),
            re.compile(r"\bотвратительн\w+\b", re.I),
            re.compile(r"\bстр(е|ё)мн\w+\b", re.I),
            re.compile(r"\bгрязн\w+\b", re.I),
            re.compile(r"\bвоняет\b", re.I),
            re.compile(r"\bвонь\b", re.I),
            re.compile(r"\bгрязный подъезд\b", re.I),
            re.compile(r"\bслишком дорог\w+\b", re.I),
            re.compile(r"\bдорого\b", re.I),
            re.compile(r"\bне стоит (этих|своих) денег\b", re.I),
            re.compile(r"\bне оправдывает цен\w+\b", re.I),
            re.compile(r"\bпроблемы с\b", re.I),
            re.compile(r"\bне работал\w*\b", re.I),
            re.compile(r"\bсломан\w+\b", re.I),
            re.compile(r"\bхолодн(о|ый|ая)\b", re.I),
            re.compile(r"\bшумн\w+\b", re.I),
            re.compile(r"\bнеуютн\w+\b", re.I),
            re.compile(r"\bне понравил\w+\b", re.I),
            re.compile(r"\bразочарован\w*\b", re.I),
        ],
    },
    "en": {
        "pos": [
            re.compile(r"\bgreat\b", re.I),
            re.compile(r"\bexcellent\b", re.I),
            re.compile(r"\bwonderful\b", re.I),
            re.compile(r"\bamazing\b", re.I),
            re.compile(r"\bdelicious\b", re.I),
            re.compile(r"\btasty\b", re.I),
            re.compile(r"\bcozy\b", re.I),
            re.compile(r"\bclean\b", re.I),
            re.compile(r"\bfriendly staff\b", re.I),
            re.compile(r"\bhelpful\b", re.I),
            re.compile(r"\bworth the price\b", re.I),
            re.compile(r"\bgood value\b", re.I),
            re.compile(r"\bperfect location\b", re.I),
            re.compile(r"\bwould stay again\b", re.I),
            re.compile(r"\bwe loved\b", re.I),
        ],
        "neg": [
            re.compile(r"\bterrible\b", re.I),
            re.compile(r"\bawful\b", re.I),
            re.compile(r"\bhorrible\b", re.I),
            re.compile(r"\bdisgusting\b", re.I),
            re.compile(r"\bdirty\b", re.I),
            re.compile(r"\bsmelled bad\b", re.I),
            re.compile(r"\bbad smell\b", re.I),
            re.compile(r"\boverpriced\b", re.I),
            re.compile(r"\btoo expensive\b", re.I),
            re.compile(r"\bnot worth\b", re.I),
            re.compile(r"\bno hot water\b", re.I),
            re.compile(r"\bnoisy\b", re.I),
            re.compile(r"\bunsafe\b", re.I),
            re.compile(r"\bwe will not come back\b", re.I),
        ],
    },
    "tr": {
        "pos": [
            re.compile(r"\bharika\b", re.I),
            re.compile(r"\bçok iyi\b", re.I),
            re.compile(r"\bmükemmel\b", re.I),
            re.compile(r"\blezzetli\b", re.I),
            re.compile(r"\btemizdi\b", re.I),
            re.compile(r"\btemiz\b", re.I),
            re.compile(r"\bgüvenliydi\b", re.I),
            re.compile(r"\bgüzel konum\b", re.I),
            re.compile(r"\bpersonel çok nazik\b", re.I),
            re.compile(r"\bparasına değer\b", re.I),
        ],
        "neg": [
            re.compile(r"\bberbat\b", re.I),
            re.compile(r"\bkorkunç\b", re.I),
            re.compile(r"\bkirli\b", re.I),
            re.compile(r"\bpis kokuyordu\b", re.I),
            re.compile(r"\bpahalı\b", re.I),
            re.compile(r"\bçok pahalı\b", re.I),
            re.compile(r"\bdeğmez\b", re.I),
            re.compile(r"\bgürültülü\b", re.I),
            re.compile(r"\bgüvensiz\b", re.I),
            re.compile(r"\byavaş internet\b", re.I),
        ],
    },
    "ar": {
        "pos": [
            re.compile(r"كتير حلو", re.I),
            re.compile(r"كتير نظيف", re.I),
            re.compile(r"نضيف", re.I),
            re.compile(r"مرتاح", re.I),
            re.compile(r"مرتاحين", re.I),
            re.compile(r"أمان", re.I),
            re.compile(r"طَيّب", re.I),
            re.compile(r"زاكي", re.I),
            re.compile(r"منيح", re.I),
        ],
        "neg": [
            re.compile(r"وسخ", re.I),
            re.compile(r"ريحة بشعة", re.I),
            re.compile(r"ريحة مجاري", re.I),
            re.compile(r"غالي", re.I),
            re.compile(r"غالي عالفاضي", re.I),
            re.compile(r"مو آمن", re.I),
            re.compile(r"مو مريح", re.I),
            re.compile(r"زحمة", re.I),
            re.compile(r"كنا مستائين", re.I),
        ],
    },
    "zh": {
        "pos": [
            re.compile(r"很好", re.I),
            re.compile(r"很棒", re.I),
            re.compile(r"很舒服", re.I),
            re.compile(r"很干净", re.I),
            re.compile(r"干净", re.I),
            re.compile(r"安全", re.I),
            re.compile(r"方便", re.I),
            re.compile(r"位置很好", re.I),
            re.compile(r"很好吃", re.I),
        ],
        "neg": [
            re.compile(r"很脏", re.I),
            re.compile(r"脏", re.I),
            re.compile(r"臭", re.I),
            re.compile(r"味道很差", re.I),
            re.compile(r"太贵", re.I),
            re.compile(r"不值", re.I),
            re.compile(r"吵", re.I),
            re.compile(r"很吵", re.I),
            re.compile(r"不安全", re.I),
        ],
    },
    # если язык нераспознан → базовая англо-ориентированная схема
    "other": {
        "pos": [
            re.compile(r"\bgreat\b", re.I),
            re.compile(r"\bexcellent\b", re.I),
            re.compile(r"\bgood\b", re.I),
            re.compile(r"\bclean\b", re.I),
            re.compile(r"\bcozy\b", re.I),
            re.compile(r"\bworth\b", re.I),
        ],
        "neg": [
            re.compile(r"\bbad\b", re.I),
            re.compile(r"\bdirty\b", re.I),
            re.compile(r"\bsmell\b", re.I),
            re.compile(r"\bexpensive\b", re.I),
            re.compile(r"\bnoisy\b", re.I),
            re.compile(r"\bnot worth\b", re.I),
        ],
    },
}


def detect_sentiment_label(text: str, lang: str, rating10: float | None = None) -> str:
    """
    Общая тональность отзыва.
    Негаторы гасят "ложные" срабатывания (не грязно → не негатив),
    а "не хорошо" интерпретируется как мягкий негатив.
    """
    if not text:
        return "neu"

    strong_neg, soft_neg, strong_pos, soft_pos, neutrals = _get_lexicons_for_lang(lang)

    # strong neg
    for rgx in strong_neg:
        if re.search(rgx, text, re.I) and not _negated(text, lang, rgx):
            return "neg"

    # soft neg
    for rgx in soft_neg:
        if re.search(rgx, text, re.I) and not _negated(text, lang, rgx):
            return "neg"

    # strong pos (с проверкой negation)
    for rgx in strong_pos:
        if re.search(rgx, text, re.I):
            if _negated(text, lang, rgx):
                return "neg"
            return "pos"

    # soft pos
    for rgx in soft_pos:
        if re.search(rgx, text, re.I):
            if _negated(text, lang, rgx):
                return "neg"
            return "pos"

    # neutrals
    for rgx in neutrals:
        if re.search(rgx, text, re.I):
            return "neu"

    # fallback по оценке
    if rating10 is not None:
        if rating10 >= 9.0:
            return "pos"
        if rating10 <= 6.0:
            return "neg"

    return "neu"



# -------------------------------------------------
# 3. Вспомогательные функции для периодов
# -------------------------------------------------

def get_period_keys(dt: datetime) -> Dict[str, str]:
    """
    Возвращает словарь с ключами периода для отзыва:
    - week_key: 'YYYY-Www'
    - month_key: 'YYYY-MM'
    - quarter_key: 'YYYY-Qn'
    - year_key: 'YYYY'
    """
    iso_year, iso_week, _ = dt.isocalendar()  # (year, week, weekday)
    month_key = f"{dt.year:04d}-{dt.month:02d}"

    # квартал по месяцу
    q = (dt.month - 1) // 3 + 1
    quarter_key = f"{dt.year:04d}-Q{q}"

    return {
        "week_key": f"{iso_year:04d}-W{iso_week:02d}",
        "month_key": month_key,
        "quarter_key": quarter_key,
        "year_key": f"{dt.year:04d}",
    }


# -------------------------------------------------
# 4. Матчинг подтем (subtopics) и маппинг в аспекты
# -------------------------------------------------

def match_subtopics_for_review(
    text: str,
    lang: str,
    topic_schema: Dict[str, Dict[str, Any]]
) -> List[Tuple[str, str]]:
    """
    Возвращает список (topic_key, subtopic_key) которые сработали в этом отзыве.
    Если по языку нет паттернов -> не матчим эту подтему.
    """
    hits: List[Tuple[str, str]] = []
    if not text:
        return hits

    for topic_key, topic_block in topic_schema.items():
        subtopics = topic_block.get("subtopics", {})
        for sub_key, sub_block in subtopics.items():
            patterns = sub_block.get("patterns", {})
            lang_patterns = patterns.get(lang, [])
            if not lang_patterns:
                # если нет паттернов на этом языке - пропускаем
                continue
            for pat in lang_patterns:
                # pat в схеме у нас как raw string/regex source
                # нужно скомпилить
                if isinstance(pat, str):
                    rgx = re.compile(pat, re.IGNORECASE | re.UNICODE | re.MULTILINE)
                else:
                    # если вдруг кто-то положит ужеcompiled re.Pattern
                    rgx = pat
                if rgx.search(text):
                    hits.append((topic_key, sub_key))
                    break  # достаточно одного попадания для данной подтемы
    return hits


def aspects_from_subtopic_by_sentiment(
    topic_schema: Dict[str, Dict[str, Any]],
    topic_key: str,
    subtopic_key: str,
    sentiment_label: str,
) -> List[str]:
    """
    У подтемы есть список aspects (включающий и позитивные, и негативные формулировки).
    Мы фильтруем эти аспекты по полярности (infer_polarity) так, чтобы
    не приписывать позитивный тег при явно негативном отзыве и наоборот.
    Нейтральные аспекты считаем для всех типов sent (если они вообще появятся).
    """
    sub_block = topic_schema[topic_key]["subtopics"][subtopic_key]
    aspects_all = sub_block.get("aspects", [])

    filtered: List[str] = []
    for a in aspects_all:
        pol = infer_polarity(a)  # 'pos' / 'neg' / 'neutral'
        if pol == "neutral":
            filtered.append(a)
        elif pol == sentiment_label:
            filtered.append(a)
        # иначе не добавляем
    return filtered


# -------------------------------------------------
# 5. Основная функция построения exploded-DF
# -------------------------------------------------

def build_exploded_reviews_df(
    raw_reviews_df: pd.DataFrame,
    topic_schema: Dict[str, Dict[str, Any]],
    add_rating_5scale: bool = True,
) -> pd.DataFrame:
    """
    Превращает сырые отзывы (как в выгрузке Reviews_dd-mm-yyyy.xls)
    в "взорванный" датафрейм аспектов.

    Ожидаемые столбцы raw_reviews_df:
        - 'Дата' (или 'date') : строка даты
        - 'Рейтинг' (или 'rating') : float/int
        - 'Источник' (или 'source')
        - 'Автор' (или 'author')
        - 'Код языка' (или 'lang_code')
        - 'Текст отзыва' (или 'text')
        - (опционально) 'Наличие ответа' (ignored here)

    Возвращаем DataFrame со столбцами:
        review_id          уникальный id отзыва (str)
        source             площадка
        date               datetime
        week_key, month_key, quarter_key, year_key
        rating_raw         исходная оценка
        rating_5           оценка приведённая к шкале 1–5 (если add_rating_5scale=True)
        lang               нормализованный язык ('ru','en','tr','ar','zh','other')
        aspect_code        код аспекта (wifi_unstable, breakfast_tasty, ...)
        sentiment          'pos'/'neg'/'neu' (тональность отзыва в целом)
        text               полный текст отзыва (для референса анализа)

    Важно: одна строка = один (review_id, aspect_code) для данного периода.
    Если один и тот же аспект ловится много раз в одном отзыве, мы оставляем его один раз.
    """

    # нормализуем имена столбцов к удобным
    cols_map = {
        'Дата': 'date',
        'Рейтинг': 'rating_raw',
        'Источник': 'source',
        'Автор': 'author',
        'Код языка': 'lang_code',
        'Текст отзыва': 'text',
        'Наличие ответа': 'responded',
        'rating': 'rating_raw',
        'date': 'date',
        'source': 'source',
        'author': 'author',
        'lang_code': 'lang_code',
        'text': 'text',
    }
    df = raw_reviews_df.rename(columns={c: cols_map.get(c, c) for c in raw_reviews_df.columns})

    # убедимся, что обязательные столбцы есть
    required_cols = ["date", "rating_raw", "source", "author", "lang_code", "text"]
    for rc in required_cols:
        if rc not in df.columns:
            # если чего-то нет (например lang_code), добавим пустой
            df[rc] = df.get(rc, "")

    # приводим дату к datetime
    def parse_date_safe(x):
        if isinstance(x, datetime):
            return x
        if pd.isna(x):
            return None
        # пробуем несколько форматов
        for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y", "%d.%m.%y"):
            try:
                return datetime.strptime(str(x), fmt)
            except ValueError:
                continue
        # last resort: let pandas try
        try:
            return pd.to_datetime(x)
        except Exception:
            return None

    df["date"] = df["date"].apply(parse_date_safe)

    # фильтруем пустые тексты
    df = df[~(df["text"].isna() | (df["text"].astype(str).str.strip() == ""))].copy()

    # создаём review_id, если в исходниках его нет
    # (делаем детерминированно через hash конкатенации текста+даты+автора+source)
    if "review_id" not in df.columns:
        def make_id(row):
            base = f"{row.get('source','')}|{row.get('author','')}|{row.get('date','')}|{row.get('text','')}"
            # короткий uuid5-like (без crypto)
            return str(uuid.uuid5(uuid.NAMESPACE_URL, base))
        df["review_id"] = df.apply(make_id, axis=1)

    # нормализуем язык
    df["lang"] = df.apply(
        lambda r: normalize_lang(str(r.get("lang_code", "")), str(r.get("text", ""))),
        axis=1
    )

    # определяем тональность ВСЕГО отзыва
    df["sentiment"] = df.apply(
        lambda r: detect_sentiment_label(str(r.get("text", "")), r.get("lang", "other")),
        axis=1
    )

    # периодические ключи
    period_keys = df["date"].apply(lambda d: get_period_keys(d) if pd.notna(d) else {
        "week_key": None,
        "month_key": None,
        "quarter_key": None,
        "year_key": None,
    })
    df = pd.concat([df, period_keys.apply(pd.Series)], axis=1)

    # опционально нормализуем рейтинг в 5-балльную шкалу
    # Логика:
    # - если rating_raw <=5 → считаем, что это уже 5-балльная
    # - если rating_raw <=10 → делим на 2
    def to_5scale(x):
        try:
            val = float(x)
        except Exception:
            return None
        if val <= 5:
            return val
        if val <= 10:
            return val / 2.0
        # если внезапно другая шкала — приводить не будем
        return None

    if add_rating_5scale:
        df["rating_5"] = df["rating_raw"].apply(to_5scale)
    else:
        df["rating_5"] = None

    # теперь извлекаем подтемы → аспекты
    exploded_rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        rid = row["review_id"]
        txt = str(row["text"])
        lang = row["lang"]
        sent_label = row["sentiment"]

        wk = row["week_key"]
        mk = row["month_key"]
        qk = row["quarter_key"]
        yk = row["year_key"]

        src = row["source"]
        dt  = row["date"]
        rating_raw = row["rating_raw"]
        rating_5 = row["rating_5"]

        # какие подтемы сработали
        subtopic_hits = match_subtopics_for_review(
            text=txt,
            lang=lang,
            topic_schema=topic_schema
        )

        # для каждой подтемы — берём аспекты соответствующей полярности
        aspects_all: List[str] = []
        for (topic_key, sub_key) in subtopic_hits:
            a_list = aspects_from_subtopic_by_sentiment(
                topic_schema=topic_schema,
                topic_key=topic_key,
                subtopic_key=sub_key,
                sentiment_label=sent_label,
            )
            aspects_all.extend(a_list)

        # дедуп по аспектам
        aspects_all = sorted(set(aspects_all))

        # теперь формируем финальные строки
        for aspect_code in aspects_all:
            exploded_rows.append({
                "review_id": rid,
                "source": src,
                "date": dt,
                "week_key": wk,
                "month_key": mk,
                "quarter_key": qk,
                "year_key": yk,
                "rating_raw": rating_raw,
                "rating_5": rating_5,
                "lang": lang,
                "aspect_code": aspect_code,
                "sentiment": sent_label,
                "text": txt,
            })

    exploded_df = pd.DataFrame(exploded_rows)

    # если не нашли ни одной темы в отзыве, он не попадает в exploded_df.
    # это норм, но для прозрачности можно (по желанию) включить "generic" аспект
    # вроде "unclassified_positive"/"unclassified_negative", если нужно покрыть всё 100%.
    # Пока не добавляем.

    return exploded_df


###############################################################################
# 3. Классификация тем для одного отзыва
###############################################################################

@dataclass
class TopicHit:
    category: str           # "cleanliness"
    category_display: str   # "Чистота"
    subtopic: str           # "arrival_clean"
    subtopic_display: str   # "Чистота номера при заезде"
    aspects: List[str]      # ["room_not_ready", ...] - может быть []
    sentiment: str          # 'pos'/'neg'/'neu' для этой подтемы (эвристика)
    rating10: Optional[float]  # нормализованная оценка гостя
    lang: str               # нормализованный язык ("ru","en","tr","ar","zh","other")


def _match_any_pattern(patterns: List[str], text: str) -> bool:
    for pat in patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True
    return False


def detect_subtopic_sentiment(text: str, lang: str, subtopic_patterns: Dict[str, List[str]]) -> str:
    """
    Эвристика тональности по конкретной подтеме:
    - если явно есть негатив-слова => neg
    - иначе если есть позитив-слова => pos
    - иначе neu
    (на практике можно доработать весами, но пока достаточно)
    """
    lang_key = lang if lang in NEGATIVE_WORDS else "en"

    # негатив приоритетнее
    for pat in NEGATIVE_WORDS.get(lang_key, []):
        if re.search(pat, text, flags=re.IGNORECASE):
            return "neg"

    for pat in POSITIVE_WORDS.get(lang_key, []):
        if re.search(pat, text, flags=re.IGNORECASE):
            return "pos"

    return "neu"


def classify_review_topics(
    text_original: str,
    lang_norm: str,
    rating10: Optional[float],
) -> List[TopicHit]:
    """
    Находит все (category, subtopic), которые встречаются в отзыве.
    Возвращает список TopicHit.
    1. По каждому subtopic проверяем регэкспы данного языка (если их нет - fallback en)
    2. Для срабатывающих подтем определяем тональность по этой подтеме.
    """

    hits: List[TopicHit] = []

    for cat_key, cat_obj in TOPIC_SCHEMA.items():
        cat_disp = cat_obj["display"]
        for sub_key, sub_obj in cat_obj["subtopics"].items():
            sub_disp = sub_obj["display"]
            patterns_by_lang = sub_obj.get("patterns", {})

            # набор паттернов для языка, либо fallback en, либо пусто
            pats = patterns_by_lang.get(lang_norm) \
                or patterns_by_lang.get("en") \
                or []

            if not pats:
                continue

            if _match_any_pattern(pats, text_original):
                sub_sent = detect_subtopic_sentiment(
                    text_original,
                    lang_norm,
                    patterns_by_lang
                )
                aspects_list = sub_obj.get("aspects", [])
                hits.append(
                    TopicHit(
                        category=cat_key,
                        category_display=cat_disp,
                        subtopic=sub_key,
                        subtopic_display=sub_disp,
                        aspects=aspects_list,
                        sentiment=sub_sent,
                        rating10=rating10,
                        lang=lang_norm,
                    )
                )

    return hits


###############################################################################
# 4. Предобработка датафрейма отзывов
###############################################################################

def preprocess_reviews_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Приводим исходную выгрузку отзывов к нормализованному датафрейму,
    готовому для дальнейшей агрегации.

    Ожидаемые входные столбцы (текущая логика выгрузки):
    - "Дата"
    - "Рейтинг"
    - "Источник"
    - "Автор"
    - "Код языка"
    - "Текст отзыва"
    - "Наличие ответа"

    Возвращаем датафрейм со столбцами:
    - date                (datetime64[ns])
    - source              (str)
    - author              (str)
    - lang_raw            (str)
    - lang_norm           (str in ["ru","en","tr","ar","zh","other"])
    - text_orig           (str)
    - text_ru             (str)  # сейчас = text_orig или "переведенный" текст
    - rating_raw          (float|None)
    - rating10            (float|None)
    - sentiment_overall   ('pos'/'neg'/'neu')
    - reply_present       (bool)  # Наличие ответа
    - topics              (list[TopicHit]) -- пока пустой, заполним дальше отдельно
    """

    # Копия, не мутируем вход
    df = df_raw.copy()

    # нормализуем названия столбцов, чтобы не зависеть от регистра и пробелов
    rename_map = {}
    for col in df.columns:
        norm_col = col.strip().lower()
        if norm_col.startswith("дата"):
            rename_map[col] = "date"
        elif "рейтинг" in norm_col:
            rename_map[col] = "rating_raw"
        elif "источник" in norm_col:
            rename_map[col] = "source"
        elif "автор" in norm_col:
            rename_map[col] = "author"
        elif "код языка" in norm_col or "язык" in norm_col:
            rename_map[col] = "lang_raw"
        elif "текст" in norm_col:
            rename_map[col] = "text_orig"
        elif "налич" in norm_col and "ответ" in norm_col:
            rename_map[col] = "reply_present"
    df = df.rename(columns=rename_map)

    # Пробуем привести date к datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # reply_present -> bool
    if "reply_present" in df.columns:
        df["reply_present"] = df["reply_present"].astype(str).str.lower().isin(["1", "true", "yes", "да", "y"])

    # Язык
    if "lang_raw" not in df.columns:
        df["lang_raw"] = ""
    df["lang_raw"] = df["lang_raw"].astype(str)
    df["text_orig"] = df.get("text_orig", "").astype(str)

    df["lang_norm"] = [
        normalize_lang(lr, txt) for lr, txt in zip(df["lang_raw"], df["text_orig"])
    ]

    # Нормализация рейтинга
    df["rating10"] = df.get("rating_raw", None).apply(normalize_rating_to_10)

    # Нормализация текста
    text_clean = normalize_text_basic(text, lang_norm)

    # text_ru (потом сюда можно будет подвесить реальный перевод)
    df["text_ru"] = [
        translate_to_ru(txt, lg) for txt, lg in zip(df["text_clean"], df["lang_norm"])
    ]

    # sentiment_overall
    df["sentiment_overall"] = [
        detect_sentiment_label(txt, lg, rt)
        for txt, lg, rt in zip(df["text_ru"], df["lang_norm"], df["rating10"])
    ]

    # заглушка для topics - заполним во второй фазе (агрегация)
    df["topics"] = [[] for _ in range(len(df))]

    return df


def annotate_topics(df: pd.DataFrame) -> pd.DataFrame:
    """
    На входе df после preprocess_reviews_df.
    Для каждой строки вызываем classify_review_topics и заполняем df["topics"].
    """
    if "text_ru" not in df.columns or "lang_norm" not in df.columns:
        raise ValueError("DataFrame must be preprocessed before annotate_topics")

    topics_all = []
    for txt, lang, r10 in zip(df["text_ru"], df["lang_norm"], df["rating10"]):
        hits = classify_review_topics(txt, lang, r10)
        topics_all.append(hits)

    df = df.copy()
    df["topics"] = topics_all
    return df

###############################################################################
# 5. Что дальше
###############################################################################

# На следующем шаге мы будем реализовывать:
#
# - агрегатор по неделе / месяцу / кварталу / году:
#   build_period_aggregates(df)
#   -> метрики по категориям и подтемам:
#       mentions, доля от всех отзывов, средний rating10 по упоминавшим,
#       доля негатива, и т.д.
#
# - baseline за последние 8 недель:
#   чтобы понимать "обычный" уровень частоты темы и качества,
#   чтобы находить отклонения недели.
#
# - функцию select_hot_topics(...) и prepare_trend_narrative(...),
#   которая будет строить "Существенные отклонения этой недели по темам"
#   в том формате, который мы уже обсудили:
#
#   Ниже исторического уровня — «Чистота номера при заезде» (−0.4),
#   «Заселение и проживание» (−0.3).
#
#   Выше исторического уровня — «Персонал (СПиР)» (+0.5), «Локация и окружение» (+0.4).
#
# - генератор объяснений причин по заселению:
#   типа "Задержки при заселении в основном вызваны неготовностью номера
#   (претензии к чистоте перед заездом) и техническими сложностями с доступом".
#
# - выбор цитат гостей (позитив/негатив по ключевым темам)
#   для блока «Голос гостей», с меткой языка и источника.
#
# Всё это будет использовать df["topics"] и TOPIC_SCHEMA выше.

###############################################################################
# 6. Вспомогательные метки периодов (неделя, месяц, квартал, год)
###############################################################################

def iso_year_week(dt: pd.Timestamp) -> str:
    """
    Вернёт ключ недели формата '2025-W42'.
    week = ISO week number (понедельник - первый день недели).
    """
    if pd.isna(dt):
        return ""
    iso = dt.isocalendar()  # pandas >= 1.1: (year, week, weekday)
    return f"{iso.year}-W{iso.week:02d}"


def quarter_label(dt: pd.Timestamp) -> str:
    """
    Вернёт метку квартала для человека и для аналитики:
    Q1, Q2, Q3, Q4
    """
    if pd.isna(dt):
        return ""
    q = (dt.month - 1) // 3 + 1
    return f"Q{q}"


def enrich_time_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет вспомогательные колонки period_key на уровне строки отзыва:
    - week_key       : '2025-W42'
    - month_key      : '2025-10'
    - quarter_key    : '2025-Q4'
    - year_key       : '2025'
    Это пригодится и для baseline, и для агрегации периодов.
    """
    out = df.copy()
    out["week_key"] = out["date"].apply(lambda d: iso_year_week(d) if pd.notna(d) else "")
    out["month_key"] = out["date"].dt.strftime("%Y-%m")
    out["year_key"] = out["date"].dt.strftime("%Y")
    out["quarter_key"] = out["date"].apply(
        lambda d: f"{d.year}-{quarter_label(d)}" if pd.notna(d) else ""
    )
    return out


###############################################################################
# 7. Взрыв тем (explode topics)
###############################################################################

def explode_topics(df: pd.DataFrame) -> pd.DataFrame:
    """
    На входе df после annotate_topics (у каждой строки отзывов есть list[TopicHit] в df["topics"]).
    Разворачиваем в формат "одна строка = одна (отзыв, подтема)".

    Возвращаем датафрейм со столбцами:
    - idx (индекс исходного отзыва) - чтобы потом джойнить при необходимости
    - date, week_key, month_key, quarter_key, year_key
    - source
    - rating10
    - sentiment_overall
    - category, category_display
    - subtopic, subtopic_display
    - subtopic_sentiment ('pos'/'neg'/'neu' внутри этой подтемы)
    - aspects (список аспектов влияния)
    """
    rows = []

    for idx, row in df.iterrows():
        topics: List[TopicHit] = row.get("topics", [])
        if not topics:
            continue
        for hit in topics:
            rows.append({
                "idx": idx,
                "date": row["date"],
                "week_key": row["week_key"],
                "month_key": row["month_key"],
                "quarter_key": row["quarter_key"],
                "year_key": row["year_key"],
                "source": row.get("source", ""),
                "rating10": row.get("rating10", None),
                "sentiment_overall": row.get("sentiment_overall", "neu"),

                "category": hit.category,
                "category_display": hit.category_display,
                "subtopic": hit.subtopic,
                "subtopic_display": hit.subtopic_display,
                "subtopic_sentiment": hit.sentiment,   # эвристика по подтеме
                "aspects": hit.aspects,               # список причинных аспектов
            })

    if not rows:
        # Нет попаданий тем вообще
        return pd.DataFrame(columns=[
            "idx","date","week_key","month_key","quarter_key","year_key","source","rating10","sentiment_overall",
            "category","category_display","subtopic","subtopic_display","subtopic_sentiment","aspects"
        ])

    exploded = pd.DataFrame(rows)

    return exploded


###############################################################################
# 8. Агрегация тем за период
###############################################################################

@dataclass
class TopicAggregate:
    # Это агрегат для одной темы (категории или подтемы) в рамках одного периода.
    mentions: int                       # сколько отзывов упоминали эту тему
    share_pct: float                    # какая доля отзывов периода упомянула эту тему, %
    avg_rating10: Optional[float]       # средний рейтинг10 среди упомянувших (NaN -> None)
    neg_share_pct: float                # % негативных упоминаний этой темы
    aspects_counter: Dict[str, int]     # какие аспекты фигурировали и сколько раз (для причин заселения и т.п.)

    # поля для сравнения с baseline
    trend_importance_pp: Optional[float]    # дельта share_pct vs baseline (в п.п.)
    trend_quality_diff: Optional[float]     # дельта avg_rating10 vs baseline (в баллах)


def _calc_topic_aggregate(
    df_period: pd.DataFrame,
    df_total_reviews_in_period: pd.DataFrame,
    group_cols: List[str],
) -> pd.DataFrame:
    """
    Универсальная вспомогательная функция.

    df_period: exploded topics, но уже отфильтрованный на интересующий период
    df_total_reviews_in_period: просто df отзывов за период (после preprocess+annotate+enrich_time_keys)

    group_cols: ["category","category_display"] или
                ["category","category_display","subtopic","subtopic_display"]

    Возвращает агрегацию по этим ключам с полями:
    - mentions
    - share_pct
    - avg_rating10
    - neg_share_pct
    - aspects_counter (dict)
    """
    if df_period.empty:
        # вернём пустой df с ожидаемыми колонками
        cols = group_cols + ["mentions","share_pct","avg_rating10","neg_share_pct","aspects_counter"]
        return pd.DataFrame(columns=cols)

    # Сколько всего отзывов в периоде (не строк в df_period, а реальных отзывов)
    total_reviews = df_total_reviews_in_period["idx"].nunique()

    # Группируем по теме
    grp = df_period.groupby(group_cols, dropna=False)

    out_rows = []
    for gkey, subdf in grp:
        # gkey может быть либо str, либо tuple
        if not isinstance(gkey, tuple):
            gkey = (gkey,)

        # сколько отзывов упомянули эту тему (уникальных idx)
        mentions = subdf["idx"].nunique()

        # доля отзывов периода, где эта тема всплыла
        share_pct = (mentions / total_reviews * 100.0) if total_reviews > 0 else 0.0

        # средняя оценка среди отзывов, где тема упомянута
        avg_rating10 = subdf.drop_duplicates("idx")["rating10"].astype(float).replace({None: np.nan}).mean()
        if math.isnan(avg_rating10):
            avg_rating10 = None

        # негативные упоминания подтемы:
        # считаем долю подстрок, где subtopic_sentiment == "neg"
        neg_mentions = (subdf["subtopic_sentiment"] == "neg").sum()
        total_mentions = len(subdf)
        neg_share_pct = (neg_mentions / total_mentions * 100.0) if total_mentions else 0.0

        # собрать аспекты влияния
        all_aspects = []
        for arr in subdf["aspects"]:
            if isinstance(arr, list):
                all_aspects.extend(arr)
        # посчитаем сколько каких аспектов
        aspects_counter = {}
        for a in all_aspects:
            aspects_counter[a] = aspects_counter.get(a, 0) + 1

        out_row = dict(zip(group_cols, gkey))
        out_row.update({
            "mentions": mentions,
            "share_pct": share_pct,
            "avg_rating10": avg_rating10,
            "neg_share_pct": neg_share_pct,
            "aspects_counter": aspects_counter,
        })
        out_rows.append(out_row)

    res = pd.DataFrame(out_rows)
    return res


def aggregate_period_topics(
    df_reviews: pd.DataFrame,
    df_topics_exploded: pd.DataFrame,
    period_filter: pd.Series,
    period_level: str,
) -> Dict[str, pd.DataFrame]:
    """
    Строит агрегаты по категориям и подтемам для выбранного периода.

    Вход:
      df_reviews          - датафрейм отзывов после preprocess+annotate+enrich_time_keys,
                            со столбцами (idx,date,rating10,week_key,...,topics)
      df_topics_exploded  - explode_topics(df_reviews)
      period_filter       - булева маска по df_reviews (True = отзыв входит в период)
      period_level        - строка-индикатор для отладки, типа "week" / "month" / "year"

    Выход:
      {
        "by_category": DataFrame[
           category, category_display,
           mentions, share_pct, avg_rating10, neg_share_pct, aspects_counter
        ],
        "by_subtopic": DataFrame[
           category, category_display, subtopic, subtopic_display,
           mentions, share_pct, avg_rating10, neg_share_pct, aspects_counter
        ],
        "total_reviews": int
      }

    """
    # Оставляем только нужные отзывы
    df_rev_period = df_reviews[period_filter].copy()
    if df_rev_period.empty:
        return {
            "by_category": pd.DataFrame(columns=[
                "category","category_display",
                "mentions","share_pct","avg_rating10","neg_share_pct","aspects_counter"
            ]),
            "by_subtopic": pd.DataFrame(columns=[
                "category","category_display","subtopic","subtopic_display",
                "mentions","share_pct","avg_rating10","neg_share_pct","aspects_counter"
            ]),
            "total_reviews": 0,
        }

    idx_set = set(df_rev_period.index.tolist())

    # отфильтруем explode по тем же отзывам
    df_topics_period = df_topics_exploded[df_topics_exploded["idx"].isin(idx_set)].copy()

    # агрегат по категориям
    by_cat = _calc_topic_aggregate(
        df_topics_period,
        df_rev_period.reset_index(names="idx"),
        ["category","category_display"],
    )

    # агрегат по подтемам
    by_sub = _calc_topic_aggregate(
        df_topics_period,
        df_rev_period.reset_index(names="idx"),
        ["category","category_display","subtopic","subtopic_display"],
    )

    return {
        "by_category": by_cat,
        "by_subtopic": by_sub,
        "total_reviews": df_rev_period["idx"].nunique(),
    }


###############################################################################
# 9. Базовая линия (baseline) по последним N неделям
###############################################################################

def get_recent_weeks_baseline(
    df_reviews: pd.DataFrame,
    df_topics_exploded: pd.DataFrame,
    current_week_key: str,
    n_weeks: int = 8,
) -> Dict[str, pd.DataFrame]:
    """
    Строим baseline по последним n_weeks ПОЛНЫХ недель ДО текущей недели.
    То есть если текущая неделя = 2025-W42,
    то baseline = [W34..W41] (8 предыдущих).

    Мы считаем ту же агрегацию по категориям и подтемам, но усредняем
    доли/оценки по этим неделям.

    Результат:
      {
        "by_category": DataFrame[
            category, category_display,
            base_share_pct, base_avg_rating10
        ],
        "by_subtopic": DataFrame[
            category, category_display, subtopic, subtopic_display,
            base_share_pct, base_avg_rating10
        ]
      }
    """

    # 1. Определим список предыдущих недель
    #    Преобразуем week_key вида "2025-W42" в (year, week_int)
    def parse_week_key(wk: str) -> Tuple[int,int]:
        # "2025-W42" -> (2025,42)
        m = re.match(r"(\d{4})-W(\d{2})$", wk)
        if not m:
            return (0,0)
        return (int(m.group(1)), int(m.group(2)))

    cur_y, cur_w = parse_week_key(current_week_key)
    if cur_y == 0:
        # нет валидной недели -> пусто
        return {
            "by_category": pd.DataFrame(columns=[
                "category","category_display","base_share_pct","base_avg_rating10"
            ]),
            "by_subtopic": pd.DataFrame(columns=[
                "category","category_display","subtopic","subtopic_display",
                "base_share_pct","base_avg_rating10"
            ])
        }

    # Генерируем список n_weeks предыдущих ключей.
    # Учитываем переход года назад (W01 предыдущего года и т.д.).
    prev_weeks = []
    y, w = cur_y, cur_w
    for _ in range(n_weeks):
        # шаг назад на неделю
        w -= 1
        if w <= 0:
            y -= 1
            # сколько недель в том годе? ISO: обычно 52 или 53
            # небольшая утилита:
            last_week = pd.Timestamp(f"{y}-12-28").isocalendar().week
            w = last_week
        prev_weeks.append(f"{y}-W{w:02d}")

    # теперь фильтруем df_reviews по этим неделям
    mask_baseline = df_reviews["week_key"].isin(prev_weeks)
    df_reviews_base = df_reviews[mask_baseline].copy()

    if df_reviews_base.empty:
        return {
            "by_category": pd.DataFrame(columns=[
                "category","category_display","base_share_pct","base_avg_rating10"
            ]),
            "by_subtopic": pd.DataFrame(columns=[
                "category","category_display","subtopic","subtopic_display",
                "base_share_pct","base_avg_rating10"
            ])
        }

    df_topics_base = df_topics_exploded[df_topics_exploded["week_key"].isin(prev_weeks)].copy()

    # baseline мы считаем так:
    #   для каждой недели считаем агрегацию категорий и подтем,
    #   потом усредняем эти метрики по неделям.
    #
    # То есть мы не просто свалили все недели в одну кучу (иначе 1 огромная неделя убьёт структуру),
    # а усреднили по неделям, чтобы получить "типичную неделю".

    def agg_one_week(week_id: str):
        mask_w = df_reviews["week_key"] == week_id
        return aggregate_period_topics(
            df_reviews=df_reviews,
            df_topics_exploded=df_topics_exploded,
            period_filter=mask_w,
            period_level="week",
        )

    per_week_cats = []
    per_week_subs = []

    for wk in prev_weeks:
        wk_res = agg_one_week(wk)
        cat_df = wk_res["by_category"].copy()
        cat_df["week_key"] = wk
        per_week_cats.append(cat_df)

        sub_df = wk_res["by_subtopic"].copy()
        sub_df["week_key"] = wk
        per_week_subs.append(sub_df)

    if per_week_cats:
        cats_all = pd.concat(per_week_cats, ignore_index=True)
    else:
        cats_all = pd.DataFrame(columns=[
            "category","category_display",
            "mentions","share_pct","avg_rating10","neg_share_pct","aspects_counter","week_key"
        ])

    if per_week_subs:
        subs_all = pd.concat(per_week_subs, ignore_index=True)
    else:
        subs_all = pd.DataFrame(columns=[
            "category","category_display","subtopic","subtopic_display",
            "mentions","share_pct","avg_rating10","neg_share_pct","aspects_counter","week_key"
        ])

    # усредняем по неделям (groupby category / subtopic)
    def avg_over_weeks(df_in: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        if df_in.empty:
            cols = group_cols + ["base_share_pct","base_avg_rating10"]
            return pd.DataFrame(columns=cols)

        # среднее share_pct по неделям и среднее avg_rating10 по неделям
        grouped = df_in.groupby(group_cols, dropna=False).agg({
            "share_pct": "mean",
            "avg_rating10": "mean",
        }).reset_index()

        grouped = grouped.rename(columns={
            "share_pct": "base_share_pct",
            "avg_rating10": "base_avg_rating10",
        })
        return grouped

    base_cat = avg_over_weeks(
        cats_all,
        ["category","category_display"],
    )

    base_sub = avg_over_weeks(
        subs_all,
        ["category","category_display","subtopic","subtopic_display"],
    )

    return {
        "by_category": base_cat,
        "by_subtopic": base_sub,
    }


###############################################################################
# 10. Обогащение текущей недели baseline-метриками
###############################################################################

def merge_with_baseline(
    cur_df: pd.DataFrame,
    base_df: pd.DataFrame,
    key_cols: List[str],
) -> pd.DataFrame:
    """
    Присоединяет baseline к текущей агрегации,
    и считает тренды:
      trend_importance_pp = share_pct - base_share_pct
      trend_quality_diff  = avg_rating10 - base_avg_rating10
    """
    if cur_df.empty:
        cols = key_cols + [
            "mentions","share_pct","avg_rating10","neg_share_pct","aspects_counter",
            "base_share_pct","base_avg_rating10",
            "trend_importance_pp","trend_quality_diff",
        ]
        return pd.DataFrame(columns=cols)

    out = cur_df.merge(
        base_df,
        how="left",
        on=key_cols,
        suffixes=("", "_base")
    )

    out["trend_importance_pp"] = out["share_pct"] - out["base_share_pct"]
    out["trend_quality_diff"] = out["avg_rating10"] - out["base_avg_rating10"]

    return out


###############################################################################
# 11. Главный хелпер для периода с baseline
###############################################################################

def build_period_analysis(
    df_reviews: pd.DataFrame,
    df_topics_exploded: pd.DataFrame,
    period_filter: pd.Series,
    period_level: str,
    current_week_key: str,
    baseline_dict: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """
    Возвращает полную картину по периоду:
      - агрегаты категорий и подтем за период
      - baseline-сравнение (только для недели — но мы можем применять и к другим периодам при желании)
      - общее число отзывов

    Структура результата:
      {
        "total_reviews": int,
        "by_category": DataFrame[
            category, category_display,
            mentions, share_pct, avg_rating10, neg_share_pct, aspects_counter,
            base_share_pct, base_avg_rating10,
            trend_importance_pp, trend_quality_diff
        ],
        "by_subtopic": DataFrame[
            ... то же самое но с subtopic ...
        ]
      }

    Пояснение:
    - Для недельного периода мы будем заполнять baseline
    - Для месяца/квартала/года baseline можем не присоединять,
      или присоединить ту же baseline (это допустимо, но не критично в тексте).
    """
    # 1. Агрегация по периоду
    agg_dict = aggregate_period_topics(
        df_reviews=df_reviews,
        df_topics_exploded=df_topics_exploded,
        period_filter=period_filter,
        period_level=period_level,
    )

    by_cat_cur = agg_dict["by_category"]
    by_sub_cur = agg_dict["by_subtopic"]
    total_reviews = agg_dict["total_reviews"]

    # 2. Склеиваем baseline с текущей метрикой,
    #    чтобы получить тренды (importances / quality diffs).
    # baseline_dict:
    #   {
    #     "by_category": ...,
    #     "by_subtopic": ...
    #   }
    #
    # Если baseline пустой (например, первая неделя сбора данных),
    # merge_with_baseline аккуратно отдаст NaN -> trend_importance_pp/quality_diff станут NaN.
    by_cat_full = merge_with_baseline(
        by_cat_cur,
        baseline_dict.get("by_category", pd.DataFrame()),
        ["category","category_display"],
    )

    by_sub_full = merge_with_baseline(
        by_sub_cur,
        baseline_dict.get("by_subtopic", pd.DataFrame()),
        ["category","category_display","subtopic","subtopic_display"],
    )

    return {
        "total_reviews": total_reviews,
        "by_category": by_cat_full,
        "by_subtopic": by_sub_full,
    }


###############################################################################
# 12. Обёртка: готовим все периоды разом
###############################################################################

def prepare_all_period_summaries(
    df_raw_reviews: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Главная точка входа для weekly_report_agent в будущем.

    На вход:
      df_raw_reviews — это исходный датафрейм с колонками выгрузки
      (Дата, Рейтинг, Источник, Код языка, Текст отзыва, ...)

    Шаги:
      1. preprocess_reviews_df -> нормализация текста, оценок, языка
      2. enrich_time_keys -> добавляем week_key, month_key, quarter_key, year_key
      3. annotate_topics -> вытягиваем topics (List[TopicHit]) по каждой строке
      4. explode_topics -> раздуваем по подтемам
      5. считаем baseline по последним 8 неделям
      6. для пяти периодов строим сводки:
         - this_week        (по текущей неделе = последняя неделя в данных)
         - month_to_date    (все отзывы с начала месяца этого конца недели)
         - quarter_to_date
         - year_to_date
         - all_time         (всё)

    Возвращаем структуру:
      {
        "ref": {
            "week_key": ...,
            "week_range": (start_date, end_date),
            "month_label": "Октябрь 2025 г.",
            "quarter_label": "IV кв. 2025 г.",
            "year_label": "2025 г.",
            "alltime_label": "Итого",
        },
        "this_week": { ... build_period_analysis(...) ... },
        "mtd":       { ... },
        "qtd":       { ... },
        "ytd":       { ... },
        "alltime":   { ... },
        "df_reviews": df_reviews,               # нормализованный df
        "df_topics_exploded": df_topics_exploded
      }

    ВАЖНО:
    - Здесь мы не делаем HTML и не пишем письмо.
    - Это чистый аналитический слой, чтобы агент мог красиво отрисовать.
    """

    # 1. preprocess
    df_reviews = preprocess_reviews_df(df_raw_reviews)

    # индексация для дальнейшего удобства (уникальный idx)
    df_reviews = df_reviews.reset_index(drop=True)
    df_reviews["idx"] = df_reviews.index

    # 2. time keys
    df_reviews = enrich_time_keys(df_reviews)

    # 3. topics
    df_reviews = annotate_topics(df_reviews)

    # 4. explode
    df_topics_exploded = explode_topics(df_reviews)

    if df_reviews.empty:
        # Пустые данные — вернём заготовку
        return {
            "ref": {},
            "this_week": {},
            "mtd": {},
            "qtd": {},
            "ytd": {},
            "alltime": {},
            "df_reviews": df_reviews,
            "df_topics_exploded": df_topics_exploded,
        }

    # Опорная "текущая неделя" — это последняя неделя, которая есть в данных.
    # Берем по максимальной дате.
    last_date = df_reviews["date"].max()
    cur_week_key = iso_year_week(last_date)
    cur_month_key = last_date.strftime("%Y-%m")          # "2025-10"
    cur_year_key = last_date.strftime("%Y")              # "2025"
    cur_quarter_key = f"{last_date.year}-{quarter_label(last_date)}"

    # Границы недели (понедельник-воскресенье) по last_date
    # NOTE: pandas weekday(): Monday=0, Sunday=6
    wd = last_date.weekday()
    week_start = (last_date - pd.Timedelta(days=wd)).normalize()
    week_end = (week_start + pd.Timedelta(days=6)).normalize()

    # Аналогично month-start
    month_start = last_date.replace(day=1).normalize()
    # quarter-start
    q = (last_date.month - 1)//3 + 1
    q_start_month = 3*(q-1)+1
    quarter_start = last_date.replace(month=q_start_month, day=1).normalize()
    # year-start
    year_start = last_date.replace(month=1, day=1).normalize()

    # Метки для подписи периодов в письме
    # Неделя: "13–19 окт 2025 г."
    # Месяц: "Октябрь 2025 г."
    # Квартал: "IV кв. 2025 г."
    # Год: "2025 г."
    # Итого: "Итого"
    MONTH_NAMES_RU = {
        1:"январь",2:"февраль",3:"март",4:"апрель",5:"май",6:"июнь",
        7:"июль",8:"август",9:"сентябрь",10:"октябрь",11:"ноябрь",12:"декабрь"
    }
    MONTH_NAMES_RU_GEN = {
        1:"января",2:"февраля",3:"марта",4:"апреля",5:"мая",6:"июня",
        7:"июля",8:"августа",9:"сентября",10:"октября",11:"ноября",12:"декабря"
    }
    # квартал в человекочитаемой форме
    QUARTER_NAME_RU = {
        1:"I кв.", 2:"II кв.", 3:"III кв.", 4:"IV кв."
    }

    # Неделя: "13–19 окт 2025 г."
    def format_week_range(d1: pd.Timestamp, d2: pd.Timestamp) -> str:
        # пример: 13–19 окт 2025 г.
        # берём день начала, день конца, месяц конца в родительном ("октября") и год
        m = MONTH_NAMES_RU_GEN[d2.month]
        return f"{d1.day}–{d2.day} {m} {d2.year} г."

    # Месяц: "Октябрь 2025 г."
    def format_month_label(d: pd.Timestamp) -> str:
        m = MONTH_NAMES_RU[d.month].capitalize()
        return f"{m} {d.year} г."

    # Квартал: "IV кв. 2025 г."
    def format_quarter_label(d: pd.Timestamp) -> str:
        qq = (d.month - 1)//3 + 1
        return f"{QUARTER_NAME_RU[qq]} {d.year} г."

    # Год: "2025 г."
    def format_year_label(d: pd.Timestamp) -> str:
        return f"{d.year} г."

    # 5. baseline: последние 8 недель ДО текущей
    baseline = get_recent_weeks_baseline(
        df_reviews=df_reviews,
        df_topics_exploded=df_topics_exploded,
        current_week_key=cur_week_key,
        n_weeks=8,
    )

    # 6. Создаём маски для периодов

    # Текущая неделя = все отзывы с week_key == cur_week_key
    mask_week = (df_reviews["week_key"] == cur_week_key)

    # Месяц-to-date = с month_start по last_date
    mask_mtd = (df_reviews["date"] >= month_start) & (df_reviews["date"] <= last_date)

    # Квартал-to-date
    mask_qtd = (df_reviews["date"] >= quarter_start) & (df_reviews["date"] <= last_date)

    # Год-to-date
    mask_ytd = (df_reviews["date"] >= year_start) & (df_reviews["date"] <= last_date)

    # Вся история
    mask_all = pd.Series(True, index=df_reviews.index)

    # 7. Сводки по периодам
    this_week = build_period_analysis(
        df_reviews=df_reviews,
        df_topics_exploded=df_topics_exploded,
        period_filter=mask_week,
        period_level="week",
        current_week_key=cur_week_key,
        baseline_dict=baseline,
    )
    mtd = build_period_analysis(
        df_reviews=df_reviews,
        df_topics_exploded=df_topics_exploded,
        period_filter=mask_mtd,
        period_level="mtd",
        current_week_key=cur_week_key,
        baseline_dict=baseline,
    )
    qtd = build_period_analysis(
        df_reviews=df_reviews,
        df_topics_exploded=df_topics_exploded,
        period_filter=mask_qtd,
        period_level="qtd",
        current_week_key=cur_week_key,
        baseline_dict=baseline,
    )
    ytd = build_period_analysis(
        df_reviews=df_reviews,
        df_topics_exploded=df_topics_exploded,
        period_filter=mask_ytd,
        period_level="ytd",
        current_week_key=cur_week_key,
        baseline_dict=baseline,
    )
    alltime = build_period_analysis(
        df_reviews=df_reviews,
        df_topics_exploded=df_topics_exploded,
        period_filter=mask_all,
        period_level="alltime",
        current_week_key=cur_week_key,
        baseline_dict=baseline,
    )

    result = {
        "ref": {
            "week_key": cur_week_key,
            "week_range": (week_start, week_end),
            "week_label": f"Итоги недели {format_week_range(week_start, week_end)}",
            "month_label": format_month_label(last_date),
            "quarter_label": format_quarter_label(last_date),
            "year_label": format_year_label(last_date),
            "alltime_label": "Итого",

            # Эти метки пойдут потом во врезки текста
            "week_start": week_start,
            "week_end": week_end,
            "month_start": month_start,
            "quarter_start": quarter_start,
            "year_start": year_start,
            "last_date": last_date,
        },
        "this_week": this_week,
        "mtd": mtd,
        "qtd": qtd,
        "ytd": ytd,
        "alltime": alltime,
        "df_reviews": df_reviews,
        "df_topics_exploded": df_topics_exploded,
    }

    return result

###############################################################################
# 13. Форматирование чисел для отчёта
###############################################################################

def fmt_score(v: Optional[float]) -> str:
    """
    Возвращает строку с округлением до десятых или '—', если нет данных.
    Пример: 9.06 -> '9.1'
    """
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    try:
        return f"{float(v):.1f}"
    except Exception:
        return "—"

def fmt_pct(v: Optional[float]) -> str:
    """
    Округляет до десятых и добавляет %.
    12.345 -> '12.3 %'
    """
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    try:
        return f"{float(v):.1f} %"
    except Exception:
        return "—"

def safe_get(series: pd.Series, default=None):
    """
    Утилита, чтобы безопасно доставать значения из Series.loc[...] без KeyError.
    """
    try:
        return series.iloc[0]
    except Exception:
        return default
###############################################################################
# 14. Выбор "ключевых" тем недели и формирование отклонений
###############################################################################

def pick_key_topics_for_week(
    by_cat_week: pd.DataFrame,
    by_sub_week: pd.DataFrame,
    min_mentions: int = 2,
    quality_drop_thr: float = -0.3,
    quality_rise_thr: float = 0.3,
    importance_rise_thr_pp: float = 5.0,
    importance_fall_thr_pp: float = -10.0,
) -> Dict[str, Any]:
    """
    Анализирует сводку текущей недели (by_category и by_subtopic)
    и выбирает:
    - темы, которые просели по качеству
    - темы, которые выросли по качеству
    - темы, которые резко чаще упоминаются (гостей волнует)
    - темы, которые сейчас почти не волнуют (молчание)
    
    Возвращает словарь с полями:
      {
        "weaker": [ { "name": "...", "delta_quality": -0.4 }, ... ],
        "stronger": [ { "name": "...", "delta_quality": +0.5 }, ... ],
        "hot_attention": [ { "name": "...", "delta_importance_pp": +7.5 }, ... ],
        "calm_down": [ { "name": "...", "delta_importance_pp": -12.0 }, ... ]
      }

    Здесь "name" — человекочитаемое имя (category_display).
    """
    weaker = []
    stronger = []
    hot_attention = []
    calm_down = []

    # Мы смотрим на уровень КАТЕГОРИЙ, потому что это управленческий уровень.
    dfc = by_cat_week.copy()

    if dfc.empty:
        return {
            "weaker": weaker,
            "stronger": stronger,
            "hot_attention": hot_attention,
            "calm_down": calm_down,
        }

    # оставим только значимые темы
    dfc = dfc[dfc["mentions"] >= min_mentions].copy()

    for _, row in dfc.iterrows():
        name = row["category_display"]
        dq = row.get("trend_quality_diff", np.nan)
        di = row.get("trend_importance_pp", np.nan)
        neg_share = row.get("neg_share_pct", 0.0)

        # просадка качества
        if not pd.isna(dq) and dq <= quality_drop_thr:
            weaker.append({
                "name": name,
                "delta_quality": dq,
            })

        # рост качества
        if not pd.isna(dq) and dq >= quality_rise_thr:
            stronger.append({
                "name": name,
                "delta_quality": dq,
            })

        # рост важности темы (гостей волнует чаще обычного).
        # Если важность растёт и при этом негатив высокий — это "горит".
        if not pd.isna(di) and di >= importance_rise_thr_pp:
            hot_attention.append({
                "name": name,
                "delta_importance_pp": di,
                "neg_share_pct": neg_share,
            })

        # падение важности темы + нет негатива -> спокойная зона.
        if (
            not pd.isna(di)
            and di <= importance_fall_thr_pp
            and (neg_share is not None)
            and (neg_share < 10.0)  # низкий негатив
        ):
            calm_down.append({
                "name": name,
                "delta_importance_pp": di,
            })

    return {
        "weaker": weaker,
        "stronger": stronger,
        "hot_attention": hot_attention,
        "calm_down": calm_down,
    }
###############################################################################
# 15. Анализ причинно-следственных связей
###############################################################################

import itertools
import math
import pandas as pd
from typing import Dict, Any, List, Tuple


def build_cooccurrence_pairs(
    df_exploded: pd.DataFrame,
    period_key_col: str,
    current_period_key: str,
    min_pair_reviews: int = 2,
    min_pair_share: float = 0.05,
) -> pd.DataFrame:
    """
    Строит ко-упоминания аспектов (пары аспектов, которые часто встречаются вместе в одном отзыве)
    за конкретный период.

    df_exploded: DataFrame со столбцами:
        - review_id: str / int
        - aspect_code: str
        - sentiment: {'pos','neg','neu'}
        - {period_key_col}: например 'week_key', 'month_key', ...

    period_key_col: имя столбца с ключом периода ('week_key' и т.д.)
    current_period_key: значение периода (например '2025-W42' или '2025-10')
    min_pair_reviews: минимальное число уникальных отзывов,
                      в которых пара встретилась, чтобы считать её значимой
    min_pair_share: минимальная доля от общего числа отзывов периода,
                    в которых эта пара встретилась (например, 0.05 = 5%)

    Возвращает DataFrame с колонками:
        left_aspect
        right_aspect
        pair_reviews         — в скольких отзывах встречалась такая пара
        pair_share           — доля отзывов периода, где пара упомянута
        left_sentiment_neg_share / _pos_share — характер левой темы
        right_sentiment_neg_share / _pos_share — характер правой темы
    """

    # ограничиваемся текущим периодом
    cur = df_exploded[df_exploded[period_key_col] == current_period_key].copy()

    # собираем для каждого review_id список аспектов, уникальных внутри этого отзыва
    review_aspects = (
        cur.groupby("review_id")["aspect_code"]
           .apply(lambda s: sorted(set(s)))
           .reset_index()
    )

    # считаем общее число отзывов в периоде
    total_reviews_period = review_aspects["review_id"].nunique()
    if total_reviews_period == 0:
        return pd.DataFrame(columns=[
            "left_aspect", "right_aspect",
            "pair_reviews", "pair_share",
            "left_sentiment_pos_share", "left_sentiment_neg_share",
            "right_sentiment_pos_share", "right_sentiment_neg_share",
        ])

    # генерим пары аспектов внутри каждого отзыва
    # пример: ['no_hot_water','overpriced','hard_to_find_entrance']
    #   -> ('no_hot_water','overpriced'), ('no_hot_water','hard_to_find_entrance'), ...
    pairs_list: List[Tuple[str, str]] = []
    for _, row in review_aspects.iterrows():
        aspects_for_review = row["aspect_code"]
        if len(aspects_for_review) < 2:
            continue
        for a, b in itertools.combinations(aspects_for_review, 2):
            left, right = sorted((a, b))
            pairs_list.append((left, right))

    if not pairs_list:
        return pd.DataFrame(columns=[
            "left_aspect", "right_aspect",
            "pair_reviews", "pair_share",
            "left_sentiment_pos_share", "left_sentiment_neg_share",
            "right_sentiment_pos_share", "right_sentiment_neg_share",
        ])

    pairs_df = pd.DataFrame(pairs_list, columns=["left_aspect", "right_aspect"])

    # сколько уникальных отзывов упоминали ту или иную пару
    # нам нужно посчитать не количество строк в pairs_list,
    # а количество review_id, где пара встречалась хотя бы раз.
    # Для этого сначала построим mapping от review_id к сету пар:
    review_pairs = []
    for _, row in review_aspects.iterrows():
        aspects_for_review = row["aspect_code"]
        rid = row["review_id"]
        for a, b in itertools.combinations(aspects_for_review, 2):
            left, right = sorted((a, b))
            review_pairs.append((rid, left, right))
    review_pairs_df = pd.DataFrame(review_pairs, columns=["review_id", "left_aspect", "right_aspect"])

    pair_reviews = (
        review_pairs_df
        .groupby(["left_aspect", "right_aspect"])["review_id"]
        .nunique()
        .reset_index(name="pair_reviews")
    )

    # добавляем долю отзывов
    pair_reviews["pair_share"] = pair_reviews["pair_reviews"] / float(total_reviews_period)

    # фильтруем только значимые по частоте связи
    pair_reviews = pair_reviews[
        (pair_reviews["pair_reviews"] >= min_pair_reviews) &
        (pair_reviews["pair_share"]  >= min_pair_share)
    ].copy()

    if pair_reviews.empty:
        return pair_reviews.assign(
            left_sentiment_pos_share=pd.Series(dtype=float),
            left_sentiment_neg_share=pd.Series(dtype=float),
            right_sentiment_pos_share=pd.Series(dtype=float),
            right_sentiment_neg_share=pd.Series(dtype=float),
        )

    # посчитаем характер каждой отдельной темы (аспекта)
    # для этого соберём сводку по sentiment внутри периода
    aspect_sent = (
        cur.groupby(["aspect_code", "sentiment", "review_id"])
           .size()
           .reset_index()
           .groupby(["aspect_code", "sentiment"])["review_id"]
           .nunique()
           .reset_index(name="mentions_reviews")
    )

    aspect_pivot = aspect_sent.pivot_table(
        index="aspect_code",
        columns="sentiment",
        values="mentions_reviews",
        fill_value=0,
        aggfunc="sum",
    ).rename(columns={
        "pos": "mentions_pos",
        "neg": "mentions_neg",
        "neu": "mentions_neu"
    })

    aspect_pivot["total_mentions"] = (
        aspect_pivot.get("mentions_pos", 0)
        + aspect_pivot.get("mentions_neg", 0)
        + aspect_pivot.get("mentions_neu", 0)
    ).replace(0, 1)

    aspect_pivot["pos_share"] = aspect_pivot["mentions_pos"] / aspect_pivot["total_mentions"]
    aspect_pivot["neg_share"] = aspect_pivot["mentions_neg"] / aspect_pivot["total_mentions"]

    aspect_pivot = aspect_pivot[["pos_share", "neg_share"]].reset_index()

    # теперь присоединим эту сводку (для left/right отдельно)
    pair_reviews = pair_reviews.merge(
        aspect_pivot.add_prefix("left_").rename(columns={"left_aspect_code": "left_aspect", "left_index": "left_aspect"}),
        on="left_aspect",
        how="left"
    )
    pair_reviews = pair_reviews.merge(
        aspect_pivot.add_prefix("right_").rename(columns={"right_aspect_code": "right_aspect", "right_index": "right_aspect"}),
        on="right_aspect",
        how="left"
    )

    # переименуем для наглядности
    pair_reviews = pair_reviews.rename(columns={
        "left_pos_share": "left_sentiment_pos_share",
        "left_neg_share": "left_sentiment_neg_share",
        "right_pos_share": "right_sentiment_pos_share",
        "right_neg_share": "right_sentiment_neg_share",
    })

    # отсортируем по значимости (сначала самые частые связки)
    pair_reviews = pair_reviews.sort_values(
        by=["pair_reviews", "pair_share"],
        ascending=[False, False],
        ignore_index=True
    )

    return pair_reviews

def classify_pair_tone(row: pd.Series) -> str:
    """
    row — строка из результата build_cooccurrence_pairs()
    Возвращает:
      "neg_cluster"     — обе стороны в основном негатив,
      "pos_cluster"     — обе стороны в основном позитив,
      "mixed_cluster"   — смешано (одна скорее негативная, другая скорее позитивная),
      "unclear"
    """
    left_neg = row["left_sentiment_neg_share"]
    right_neg = row["right_sentiment_neg_share"]
    left_pos = row["left_sentiment_pos_share"]
    right_pos = row["right_sentiment_pos_share"]

    # считаем "негативной" если neg_share >= 0.5
    left_is_neg = (left_neg >= 0.5)
    right_is_neg = (right_neg >= 0.5)

    # считаем "позитивной" если pos_share >= 0.5
    left_is_pos = (left_pos >= 0.5)
    right_is_pos = (right_pos >= 0.5)

    if left_is_neg and right_is_neg:
        return "neg_cluster"
    if left_is_pos and right_is_pos:
        return "pos_cluster"
    if (left_is_neg and right_is_pos) or (left_is_pos and right_is_neg):
        return "mixed_cluster"
    return "unclear"


def describe_top_pairs(
    pairs_df: pd.DataFrame,
    aspect_registry: Dict[str, Dict[str, Any]],
    top_n: int = 5
) -> Dict[str, List[str]]:
    """
    Превращает топ-связки (по важности) в текстовые тезисы для отчёта.

    pairs_df — результат build_cooccurrence_pairs(), отсортированный по значимости.
    aspect_registry — из build_aspect_registry(TOPIC_SCHEMA).
    top_n — сколько связок максимум выводить в каждую категорию.

    Возвращает словарь:
    {
        "neg_cluster": [
            "...",
            "..."
        ],
        "mixed_cluster": [
            "...",
        ],
        "pos_cluster": [
            "...",
        ]
    }
    """

    results = {
        "neg_cluster": [],
        "mixed_cluster": [],
        "pos_cluster": [],
    }

    if pairs_df.empty:
        return results

    # Добавим колонку типа связки
    df = pairs_df.copy()
    df["cluster_type"] = df.apply(classify_pair_tone, axis=1)

    # пройдёмся по строкам по значимости
    for _, row in df.iterrows():
        ctype = row["cluster_type"]
        if ctype not in results:
            continue

        left_code = row["left_aspect"]
        right_code = row["right_aspect"]

        left_label  = aspect_registry.get(left_code,  {}).get("label", left_code)
        right_label = aspect_registry.get(right_code, {}).get("label", right_code)

        pair_reviews = row["pair_reviews"]
        pair_share   = row["pair_share"]

        # аккуратная человеческая формулировка
        if ctype == "neg_cluster":
            text = (
                f"Часто одновременно жалуются на «{left_label}» и «{right_label}» — "
                f"эту связку отметили примерно {pair_reviews} гост(я/ей), "
                f"что составляет около {pair_share:.0%} всех отзывов за период. "
                f"Это выглядит как системная зона риска."
            )

        elif ctype == "pos_cluster":
            text = (
                f"Гости часто одновременно хвалят «{left_label}» и «{right_label}». "
                f"Эта позитивная связка упоминалась примерно в {pair_reviews} отзыв(ах), "
                f"~{pair_share:.0%} от всех упоминаний недели. "
                f"Это ядро сильных сторон, влияющее на лояльность."
            )

        elif ctype == "mixed_cluster":
            text = (
                f"Часто встречается контраст: положительно оценивают «{left_label}», "
                f"но при этом жалуются на «{right_label}». "
                f"Такой разрыв ожиданий отметили ~{pair_reviews} гост(я/ей), "
                f"что около {pair_share:.0%} отзывов недели. "
                f"Это критичная точка восприятия ценности."
            )

        else:
            continue

        # не пихаем бесконечно
        if len(results[ctype]) < top_n:
            results[ctype].append(text)

    return results
###############################################################################
# 16. Формирование текстовых блоков для письма
###############################################################################

def summarize_overview_for_header(
    ref_info: Dict[str, Any],
    this_week: Dict[str, Any],
    mtd: Dict[str, Any],
    qtd: Dict[str, Any],
    ytd: Dict[str, Any],
    alltime: Dict[str, Any],
) -> str:
    """
    Возвращает текстовый блок (HTML-параграфы) для шапки письма.
    Он описывает текущую неделю и роль этой недели относительно месяца/квартала/года/итого.

    Логика:
    - Число отзывов недели и средняя оценка недели.
    - Доля позитива/негатива недели.
    - Какой вклад эта неделя даёт в MTD/QTD/YTD (в процентах отзывов).
    - Ничего про "/10" в тексте.
    """

    # полезные краткие метрики: средний рейтинг недели, доля позитива и негатива недели
    def get_basic_week_stats(block: Dict[str, Any]) -> Dict[str, Any]:
        df_by_cat = block["by_category"]
        total_reviews = block["total_reviews"]

        # средняя оценка недели в целом:
        # просто среднее rating10 по всем отзывам периода
        # (не только темам). Это нужно достать из df_reviews, но у нас
        # здесь нет df_reviews. Поэтому чуть ниже мы отметим, что
        # weekly_report_agent при финальной сборке должен передать эти числа явно.
        #
        # Для каркаса пока поставим None -> "—".
        avg_rating_overall = None

        # посчитаем % позитива / негатива по sentiment_overall,
        # но это тоже требует df_reviews периода. Аналогично — агент может это посчитать
        # и передать как параметры. Здесь формально оставим "—".
        pos_pct = None
        neg_pct = None

        return {
            "total_reviews": total_reviews,
            "avg_rating_overall": avg_rating_overall,
            "pos_pct": pos_pct,
            "neg_pct": neg_pct,
        }

    wk_stats = get_basic_week_stats(this_week)

    # вклад недели в larger periods (% отзывов)
    def share_in(b_small: Dict[str,Any], b_big: Dict[str,Any]) -> Optional[float]:
        num_small = b_small["total_reviews"]
        num_big = b_big["total_reviews"]
        if not num_big:
            return None
        return (num_small / num_big) * 100.0

    share_in_mtd = share_in(this_week, mtd)
    share_in_qtd = share_in(this_week, qtd)
    share_in_ytd = share_in(this_week, ytd)
    share_in_all = share_in(this_week, alltime)

    # формируем фразы
    # Пример тона:
    # "<b>Текущая неделя:</b> 14 отзывов, средняя оценка 9.1. Позитивные упоминания — 78.6 %, негативные — 7.1 %."
    # "Эта неделя дала около 62 % всех отзывов за месяц и идёт на уровне общего тона месяца."
    p_week = (
        "<p>"
        f"<b>Текущая неделя:</b> {wk_stats['total_reviews']} отзыв(ов). "
        f"Средняя оценка — {fmt_score(wk_stats['avg_rating_overall'])}. "
        f"Позитивные упоминания — {fmt_pct(wk_stats['pos_pct'])}, "
        f"негативные — {fmt_pct(wk_stats['neg_pct'])}."
        "</p>"
    )

    # второе предложение: вклад недели
    parts_role = []
    if share_in_mtd is not None:
        parts_role.append(
            f"Эта неделя дала примерно {fmt_pct(share_in_mtd)[:-2]} всех отзывов за месяц"
        )
    if share_in_qtd is not None:
        parts_role.append(
            f"{fmt_pct(share_in_qtd)[:-2]} за квартал"
        )
    if share_in_ytd is not None:
        parts_role.append(
            f"{fmt_pct(share_in_ytd)[:-2]} за год"
        )
    if share_in_all is not None:
        parts_role.append(
            f"{fmt_pct(share_in_all)[:-2]} за всю совокупность отзывов"
        )

    if parts_role:
        role_sentence = (
            "<p><b>Роль недели в общей статистике:</b> "
            + "; ".join(parts_role)
            + ".</p>"
        )
    else:
        role_sentence = ""

    return p_week + role_sentence


def summarize_trends_block(
    by_cat_week: pd.DataFrame,
    picked: Dict[str, Any],
) -> str:
    """
    Формирует текст блока "Динамика и вклад текущей недели":
    - ниже исторического уровня,
    - выше исторического уровня,
    - что особенно волнует гостей,
    - что сейчас не вызывает жалоб.
    
    Формат:
    <h3>Динамика и вклад текущей недели</h3>
    <p><b>Существенные отклонения этой недели по темам:</b><br>
    <span style="color:#c0392b;"><b>Ниже исторического уровня</b></span> — ...<br>
    <span style="color:#2e8b57;"><b>Выше исторического уровня</b></span> — ...<br>
    ...
    </p>
    """

    weaker_lines = []
    for w in picked["weaker"]:
        delta_q = fmt_score(w["delta_quality"])
        # delta_q уже со знаком (напр. -0.4), fmt_score оставит минус
        weaker_lines.append(f"«{w['name']}» ({delta_q})")

    stronger_lines = []
    for s in picked["stronger"]:
        delta_q = fmt_score(s["delta_quality"])
        stronger_lines.append(f"«{s['name']}» (+{delta_q.lstrip('-')})")

    hot_lines = []
    for h in picked["hot_attention"]:
        di = fmt_score(h["delta_importance_pp"])  # рост важности, в п.п.
        # fmt_score не добавляет %, добавим позже текстом
        hot_lines.append(
            f"«{h['name']}» (+{di.lstrip('-')} п.п. к типичной частоте обсуждения)"
        )

    calm_lines = []
    for c in picked["calm_down"]:
        di = fmt_score(c["delta_importance_pp"])
        calm_lines.append(
            f"«{c['name']}» (−{di.lstrip('-')} п.п. к обычной частоте, жалоб практически нет)"
        )

    # собираем абзац
    lines = []

    lines.append("<p><b>Существенные отклонения этой недели по темам:</b><br>")

    # Ниже исторического уровня
    if weaker_lines:
        lines.append(
            '<span style="color:#c0392b;"><b>Ниже исторического уровня</b></span> — '
            + ", ".join(weaker_lines)
            + ".<br>"
        )

    # Выше исторического уровня
    if stronger_lines:
        lines.append(
            '<span style="color:#2e8b57;"><b>Выше исторического уровня</b></span> — '
            + ", ".join(stronger_lines)
            + ".<br>"
        )

    # Часто волнует
    if hot_lines:
        lines.append(
            "<b>Особенно часто упоминалось</b> — "
            + ", ".join(hot_lines)
            + ".<br>"
        )

    # Спокойно
    if calm_lines:
        lines.append(
            "Отдельно отметим спокойные зоны — "
            + ", ".join(calm_lines)
            + "."
        )

    lines.append("</p>")

    return "\n".join(lines)


def summarize_checkin_line(root_cause_text: Optional[str]) -> str:
    """
    Возвращает фразу о причинах заселения, если есть.
    Если нет причин — возвращает "".
    """
    if not root_cause_text:
        return ""
    return (
        "<p><b>Заселение и проживание.</b> "
        + root_cause_text +
        "</p>"
    )
def parse_date_to_datetime(date_raw: Any) -> datetime:
    """
    Унификация даты отзыва из XLS. Принимаем datetime, pandas.Timestamp или строку.
    Возвращаем datetime без таймзоны.
    """
    if isinstance(date_raw, datetime):
        return date_raw
    try:
        return pd.to_datetime(date_raw).to_pydatetime().replace(tzinfo=None)
    except Exception:
        # если совсем мусор — ставим сегодняшнюю дату
        return datetime.utcnow().replace(tzinfo=None)


def derive_period_keys(dt: datetime) -> Tuple[str, str, str, str]:
    """
    На базе даты делаем ключи периодов:
    - неделя формата YYYY-W## (ISO неделя)
    - месяц YYYY-MM
    - квартал YYYY-Q#
    - год YYYY
    """
    iso_year, iso_week, _ = dt.isocalendar()  # (год ISO, номер недели, день)
    week_key = f"{iso_year}-W{iso_week:02d}"

    month_key = f"{dt.year}-{dt.month:02d}"

    quarter = (dt.month - 1) // 3 + 1
    quarter_key = f"{dt.year}-Q{quarter}"

    year_key = f"{dt.year}"

    return week_key, month_key, quarter_key, year_key


def build_review_id(dt: datetime, source: str, raw_text: str) -> str:
    """
    Делаем стабильный ID отзыва:
    sha1(<yyyy-mm-dd>|<source>|<первые 200 символов текста>)
    """
    base = f"{dt.date().isoformat()}|{source}|{raw_text[:200]}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


SENTENCE_SPLIT_RE = re.compile(r"[.!?…]+[\s\n]+")

def extract_clean_quotes(text_clean: str) -> List[str]:
    """
    Берём 1-3 характерных предложения из текста отзыва
    - уже без URL/e-mail/телефонов (normalize_text_basic это сделал)
    - отбрасываем слишком короткие (<15 симв) и слишком длинные (>300 симв)
    """
    if not text_clean:
        return []
    # режем по предложениям
    parts = SENTENCE_SPLIT_RE.split(text_clean)
    quotes = []
    for p in parts:
        p2 = p.strip()
        if 15 <= len(p2) <= 300:
            quotes.append(p2)
        if len(quotes) >= 3:
            break
    return quotes


def build_semantic_row(review_raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Главная функция для backfill/weekly:
    превращает одну сырую строку отзыва (как в XLS)
    в нормализованную семантическую запись.

    Ожидается, что review_raw содержит минимум:
    - 'date'         -> дата отзыва (str/datetime)
    - 'source'       -> источник (Booking/Google/...)
    - 'rating_raw'   -> оригинальная оценка (строка/цифра)
    - 'text'         -> текст отзыва
    - 'lang_hint'    -> код языка источника (может быть None)
    """

    raw_text = review_raw.get("text", "") or ""
    lang_hint = review_raw.get("lang_hint", "")

    # язык
    lang_norm = normalize_lang(lang_hint, raw_text)

    # очистили текст
    text_clean = normalize_text_basic(raw_text, lang_norm)

    # псевдо-перевод в русскую лексику
    text_ru = translate_to_ru(text_clean, lang_norm)

    # рейтинг в /10
    rating10 = normalize_rating_to_10(review_raw.get("rating_raw"))

    # общая тональность
    sentiment_overall = detect_sentiment_label(
        text_clean,
        lang_norm,
        rating10
    )

    # извлечение аспектов
    topics_pos: set[str] = set()
    topics_neg: set[str] = set()
    topics_all: set[str] = set()

    # вспомогательное: храним инфо для последующей классификации пар
    # (aspect_key, is_pos, is_neg)
    aspect_sentiment_flags: List[Tuple[str, bool, bool]] = []

    for topic_name, topic_def in TOPIC_SCHEMA.items():
        for subtopic_name, sub_def in topic_def["subtopics"].items():
            patterns_lang = sub_def.get("patterns", {})
            # выбираем паттерны для текущего языка, либо fallback 'en'
            pats = patterns_lang.get(lang_norm) or patterns_lang.get("en") or []
            if not pats:
                continue

            # проверяем, встречается ли подтема
            matched = False
            for pat in pats:
                if re.search(pat, text_clean, flags=re.IGNORECASE):
                    matched = True
                    break
            if not matched:
                continue

            # локальная тональность по подтеме
            sent_local = detect_subtopic_sentiment(
                text_clean,
                lang_norm,
                subtopic_patterns=pats
            )

            # нормализованный аспект
            aspects_list = sub_def.get("aspects", [])
            if aspects_list:
                aspect_key = aspects_list[0]
            else:
                aspect_key = f"{topic_name}.{subtopic_name}"

            topics_all.add(aspect_key)

            if sent_local == "pos":
                topics_pos.add(aspect_key)
                aspect_sentiment_flags.append((aspect_key, True, False))
            elif sent_local == "neg":
                topics_neg.add(aspect_key)
                aspect_sentiment_flags.append((aspect_key, False, True))
            else:
                aspect_sentiment_flags.append((aspect_key, False, False))

    # построение связок аспектов (co-occurrence)
    pair_tags: List[Dict[str, str]] = []
    seen_pairs: set[Tuple[str, str]] = set()
    aspects_list_all = list(topics_all)

    for i in range(len(aspects_list_all)):
        for j in range(i + 1, len(aspects_list_all)):
            a = aspects_list_all[i]
            b = aspects_list_all[j]
            pair_sorted = tuple(sorted([a, b]))
            if pair_sorted in seen_pairs:
                continue
            seen_pairs.add(pair_sorted)

            # определяем категорию пары:
            a_pos = any(x[0] == a and x[1] for x in aspect_sentiment_flags)
            a_neg = any(x[0] == a and x[2] for x in aspect_sentiment_flags)
            b_pos = any(x[0] == b and x[1] for x in aspect_sentiment_flags)
            b_neg = any(x[0] == b and x[2] for x in aspect_sentiment_flags)

            if a_neg and b_neg:
                cat = "systemic_risk"
            elif (a_pos and b_neg) or (b_pos and a_neg):
                cat = "expectations_conflict"
            elif a_pos and b_pos:
                cat = "loyalty_driver"
            else:
                # пара не несёт смысла для отчёта — пропускаем
                continue

            pair_tags.append({
                "a": a,
                "b": b,
                "cat": cat,
            })

    # цитаты-кандидаты
    quote_candidates = extract_clean_quotes(text_clean)

    # периоды
    dt = parse_date_to_datetime(review_raw.get("date"))
    week_key, month_key, quarter_key, year_key = derive_period_keys(dt)

    # стабильный id
    review_id = build_review_id(dt, review_raw.get("source", ""), raw_text)

    return {
        "review_id": review_id,
        "date": dt.date().isoformat(),
        "week_key": week_key,
        "month_key": month_key,
        "quarter_key": quarter_key,
        "year_key": year_key,
        "source": review_raw.get("source", ""),
        "rating10": rating10,
        "sentiment_overall": sentiment_overall,
        "topics_pos": json.dumps(sorted(list(topics_pos)), ensure_ascii=False),
        "topics_neg": json.dumps(sorted(list(topics_neg)), ensure_ascii=False),
        "topics_all": json.dumps(sorted(list(topics_all)), ensure_ascii=False),
        "pair_tags": json.dumps(pair_tags, ensure_ascii=False),
        "quote_candidates": json.dumps(quote_candidates, ensure_ascii=False),
    }
