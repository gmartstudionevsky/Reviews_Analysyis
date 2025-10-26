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
# ВАЖНО:
# - Здесь нет форматирования HTML. Это чистая аналитика.
# - Здесь нет отправки писем.
#
# Совместимо с Python 3.11

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import re
import math
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
    Заглушка. В перспективе:
    - сюда можно встроить реальный переводчик (вызов API или локальная модель)
    - для арабского, турецкого, китайского, английского.

    Сейчас:
    - Если это русский (или кириллица), возвращаем как есть
    - Иначе возвращаем оригинал без перевода (чтобы пайплайн не ломался)
    """
    if lang == "ru":
        return text
    if CYRILLIC_RE.search(text):
        return text
    # TODO: сюда потом вставим фактический перевод
    return text


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
POSITIVE_WORDS = {
    "ru": [
        r"\bотличн", r"\bпрекрасн", r"\bзамечательн", r"\bчисто\b", r"\bчистый\b",
        r"\bудобн", r"\bкомфорт", r"\bвежлив", r"\bдружелюб", r"\bпонравил", r"\bрекомендую",
        r"\bспасибо", r"\bбыстро заселили", r"\bбыстро поселили",
    ],
    "en": [
        r"\bgreat\b", r"\bexcellent\b", r"\bperfect\b", r"\bclean\b",
        r"\bfriendly\b", r"\bhelpful\b", r"\bcomfortable\b", r"\brecommend",
        r"\bthank you", r"\bfast check[- ]?in\b", r"\bquick check[- ]?in\b",
    ],
    "tr": [
        r"\bharika\b", r"\btemiz\b", r"\bmükemmel\b", r"\bçok iyi\b", r"\biyiydi\b",
        r"\bpersonel yardımsever\b", r"\bkonforlu\b",
    ],
    "ar": [
        r"ممتاز", r"نظيف", r"رائع", r"مريح", r"ودود", r"لطيف", r"سريع",
    ],
    "zh": [
        r"很好", r"非常好", r"干净", r"友好", r"热情", r"舒服", r"满意",
    ],
}

NEGATIVE_WORDS = {
    "ru": [
        r"\bгрязн", r"\bгрязно\b", r"\bвонял", r"\bзапах\b", r"\bшумно\b",
        r"\bшумит", r"\bхолодно\b", r"\bжарко\b", r"\bплесен", r"\bгрязная\b",
        r"\bужасн", r"\bотврат", r"\bхамил", r"\bгруб", r"\bдолго ждали\b",
        r"\bне работал", r"\bне работала", r"\bслом", r"\bпротек", r"\bне убирали",
        r"\bзадержка заселения", r"\bдолго заселяли",
    ],
    "en": [
        r"\bdirty\b", r"\bsmell", r"\bstinky\b", r"\bnoisy\b", r"\bnoise\b",
        r"\bcold\b", r"\bhot\b", r"\bmold\b", r"\brude\b", r"\bunfriendly\b",
        r"\bslow check[- ]?in\b", r"\bwaited too long\b", r"\bnot working\b",
        r"\bbroken\b", r"\bleaking\b", r"\bno cleaning\b",
    ],
    "tr": [
        r"\bpis\b", r"\bkoku\b", r"\bgürültülü\b", r"\bgürültü\b", r"\bçok soğuk\b", r"\bçok sıcak\b",
        r"\bbozuk\b", r"\bçalışmıyor\b", r"\btemizlenmedi\b", r"\bgecikti\b", r"\bbekledik\b",
    ],
    "ar": [
        r"قذر", r"رائحة كريهة", r"صاخب", r"إزعاج", r"بارد جدًا", r"حار جدًا",
        r"مكسور", r"لا يعمل", r"تأخير", r"انتظرنا طويلًا", r"لم ينظفوا",
    ],
    "zh": [
        r"脏", r"味道", r"臭", r"吵", r"很吵", r"太冷", r"太热", r"坏了", r"不工作", r"等了很久", r"没人打扫",
    ],
}


def detect_sentiment_label(text: str, lang: str, fallback_rating10: Optional[float]) -> str:
    """
    Возвращает 'pos', 'neg' или 'neu' для отзыва в целом.
    Примитивная эвристика:
    1. ищем негативные паттерны -> если нашли что-то "сильное", сразу neg
    2. ищем позитивные паттерны -> pos
    3. fallback по оценке:
        - >=9.0 -> pos
        - <=6.0 -> neg
        - иначе neu
    Это не ML. Но:
    - мы заложили сюда язык,
    - мы можем в будущем воткнуть модель вместо regexp.
    """
    lang_key = lang if lang in NEGATIVE_WORDS else "en"  # дефолт к англ набору

    # негатив в приоритете
    for pat in NEGATIVE_WORDS.get(lang_key, []):
        if re.search(pat, text, flags=re.IGNORECASE):
            return "neg"

    for pat in POSITIVE_WORDS.get(lang_key, []):
        if re.search(pat, text, flags=re.IGNORECASE):
            return "pos"

    if fallback_rating10 is not None:
        if fallback_rating10 >= 9.0:
            return "pos"
        if fallback_rating10 <= 6.0:
            return "neg"

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
    "staff": {
        "display": "Персонал (СПиР)",
        "subtopics": {
            "politeness": {
                "display": "Вежливость и отношение",
                "patterns": {
                    "ru": [r"вежлив", r"доброжелательн", r"приветлив", r"дружелюб"],
                    "en": [r"friendly", r"polite", r"kind", r"welcom", r"helpful"],
                    "tr": [r"güleryüz", r"nazik", r"yardımsever"],
                    "ar": [r"ودود", r"لطيف", r"مهذب"],
                    "zh": [r"友好", r"热情", r"礼貌"],
                },
            },
            "competence": {
                "display": "Решили вопрос / помогли",
                "patterns": {
                    "ru": [r"помог", r"решил(и)? вопрос", r"разрешил(и)? проблему"],
                    "en": [r"solved.*problem", r"helped us", r"assisted", r"took care of"],
                    "tr": [r"çözdü", r"yardım etti", r"ilgilendiler"],
                    "ar": [r"حلوا المشكلة", r"ساعدونا", r"اعتنى"],
                    "zh": [r"帮忙", r"帮助我们", r"处理了", r"解决了问题"],
                },
            },
            "speed_response": {
                "display": "Скорость реакции",
                "patterns": {
                    "ru": [r"быстро ответил", r"оперативно", r"сразу помогли", r"мгновенно"],
                    "en": [r"quick response", r"responded fast", r"immediately helped"],
                    "tr": [r"hemen cevap", r"çok hızlı yardımcı"],
                    "ar": [r"بسرعة", r"فورًا", r"استجاب.*سريع"],
                    "zh": [r"很快回复", r"马上帮忙", r"立即处理"],
                },
            },
            "rude": {
                "display": "Некорректное поведение",
                "patterns": {
                    "ru": [r"груб", r"хамил", r"не.*помогать", r"плевать", r"не хочу.*слушать"],
                    "en": [r"rude", r"unfriendly", r"impolite", r"didn't care"],
                    "tr": [r"kabaca", r"saygısız", r"umursamadı"],
                    "ar": [r"وقح", r"غير ودود", r"لم يهتم"],
                    "zh": [r"态度.*差", r"不友好", r"粗鲁"],
                },
            },
            "communication_before_arrival": {
                "display": "Коммуникация до заезда",
                "patterns": {
                    "ru": [r"заранее предупредил", r"заранее связались", r"связались заранее", r"объяснил(и)? как заехать", r"инструкци[яи] до заезда"],
                    "en": [r"before arrival", r"prior to arrival", r"instructions before check[- ]?in", r"contacted us in advance"],
                    "tr": [r"gelmeden önce bilgi", r"önceden yazdılar", r"talimat verdi"],
                    "ar": [r"قبل الوصول", r"أرسلوا التعليمات", r"أخبرونا مسبقًا"],
                    "zh": [r"到达前联系", r"提前告知", r"入住说明提前给"],
                },
            },
        },
    },

    "checkin_stay": {
        "display": "Заселение и проживание",
        "subtopics": {
            "checkin_speed": {
                "display": "Скорость заселения / ожидание",
                "patterns": {
                    "ru": [r"долго заселяли", r"ждал(и)? заселения", r"очередь на заселение", r"ждать ключ", r"ждали ключ", r"пока подготовят номер"],
                    "en": [r"slow check[- ]?in", r"wait(ed)? too long", r"had to wait for the room", r"wait for the key"],
                    "tr": [r"check[- ]?in.*gecikti", r"çok bekledik", r"oda hazır değildi"],
                    "ar": [r"تسجيل الدخول.*طويل", r"انتظرنا.*الغرفة", r"انتظرنا المفتاح"],
                    "zh": [r"办理入住.*久", r"等了很久", r"房间还没准备好", r"等钥匙"],
                },
                "aspects": ["staff_slow", "room_not_ready", "tech_access_issue"],
            },
            "access": {
                "display": "Доступ на этаж / в номер",
                "patterns": {
                    "ru": [r"код.*не подход", r"не открыть дверь", r"карта.*не работал", r"не пустил(о)? в подъезд", r"не мог(ли)? попасть на этаж"],
                    "en": [r"code didn't work", r"key.*card.*not working", r"couldn't open the door", r"access to the floor"],
                    "tr": [r"kod çalışmadı", r"kart çalışmıyordu", r"kapı açılmadı", r"kata çıkamadık"],
                    "ar": [r"لم يفتح الكود", r"بطاقة الدخول لا تعمل", r"لا أستطيع فتح الباب", r"لا أقدر الصعود للطابق"],
                    "zh": [r"门卡.*不好用", r"进不去房间", r"进不去楼层", r"门打不开", r"密码不对"],
                },
                "aspects": ["tech_access_issue"],
            },
            "docs_payment": {
                "display": "Документы и оплата",
                "patterns": {
                    "ru": [r"чек", r"отчетн(ые|ые документы)", r"закрывающ", r"депозит", r"залог", r"оплатить доп", r"заблокировали сумму"],
                    "en": [r"invoice", r"receipt", r"deposit", r"blocked my card", r"extra charge", r"no receipt"],
                    "tr": [r"fatura", r"fiş vermedi", r"depozito", r"karttan blokaj"],
                    "ar": [r"إيصال", r"فاتورة", r"تأمين", r"حجزوا مبلغًا", r"دفع إضافي"],
                    "zh": [r"押金", r"发票", r"收据", r"额外收费", r"预授权"],
                },
            },
            "instructions": {
                "display": "Инструкции по проживанию / доступу / Wi-Fi",
                "patterns": {
                    "ru": [r"инструкци", r"не объяснил.*как попасть", r"не сказали пароль.*wi-?fi", r"никто не объяснил куда идти", r"как пользоваться", r"не рассказали правила"],
                    "en": [r"instructions", r"nobody explained", r"no info how to enter", r"no wifi password", r"no wi[- ]?fi code", r"didn't tell us the rules"],
                    "tr": [r"talimat yoktu", r"bize anlatmadılar", r"wifi şifresi yok", r"nasıl gireceğimizi söylemediler"],
                    "ar": [r"لم يشرحوا", r"لا تعليمات", r"لم يخبرونا بكلمة سر الواي فاي", r"لا نعرف كيف ندخل"],
                    "zh": [r"没有说明", r"没人告诉我们怎么进", r"没说wifi密码", r"没讲规则"],
                },
            },
            "checkout": {
                "display": "Удобство выезда",
                "patterns": {
                    "ru": [r"выселени[ея].*удобн", r"просто сдать ключ", r"легко уехать", r"чек[- ]?аут.*удобно"],
                    "en": [r"easy check[- ]?out", r"checkout was easy", r"just left the key", r"left the keys"],
                    "tr": [r"çıkış kolaydı", r"check[- ]?out kolaydı"],
                    "ar": [r"الخروج كان سهل", r"سهل المغادرة"],
                    "zh": [r"退房.*方便", r"离开很方便", r"直接把钥匙留"],
                },
            },
        },
    },

    "cleanliness": {
        "display": "Чистота",
        "subtopics": {
            "arrival_clean": {
                "display": "Чистота номера при заезде",
                "patterns": {
                    "ru": [r"грязн.*при заезд", r"грязный номер", r"не убран", r"постель.*грязн", r"волос[ы] на постел", r"грязная ванн", r"неприбранно"],
                    "en": [r"dirty room", r"not cleaned", r"hair.*bed", r"bathroom was dirty", r"smell of smoke"],
                    "tr": [r"oda kirliydi", r"temizlenmemiş", r"yatakta saç", r"banyo kirli"],
                    "ar": [r"الغرفة متسخة", r"غير نظيف", r"رائحة دخان", r"لم يتم تنظيفها"],
                    "zh": [r"房间很脏", r"没打扫", r"床上有头发", r"卫生间很脏", r"烟味"],
                },
                "aspects": ["room_not_ready"],
            },
            "stay_cleaning": {
                "display": "Уборка во время проживания",
                "patterns": {
                    "ru": [r"не убирали", r"уборка.*редко", r"полотенц[а]? не поменял", r"не выносили мусор"],
                    "en": [r"no cleaning", r"didn't clean", r"didn't change towels", r"trash not taken"],
                    "tr": [r"temizlik yapılmadı", r"havlu değişmedi", r"çöp alınmadı"],
                    "ar": [r"لم ينظفوا", r"لم يغيروا المناشف", r"لم يرموا القمامة"],
                    "zh": [r"没人打扫", r"没换毛巾", r"垃圾没倒"],
                },
            },
            "smell": {
                "display": "Запахи",
                "patterns": {
                    "ru": [r"запах", r"вонял", r"вонь", r"пахло сигарет", r"пахло канализаци"],
                    "en": [r"smell", r"stinky", r"cigarette smell", r"sewage smell"],
                    "tr": [r"koku", r"sigara kokusu", r"kötü kokuyordu"],
                    "ar": [r"رائحة سيئة", r"رائحة دخان", r"رائحة مجاري"],
                    "zh": [r"味道很重", r"臭味", r"烟味", r"下水道味"],
                },
            },
            "public_areas": {
                "display": "Чистота общих зон",
                "patterns": {
                    "ru": [r"грязн.*коридор", r"грязн.*лобби", r"грязн.*лифте", r"пыль.*лестниц"],
                    "en": [r"dirty hallway", r"dirty lobby", r"dirty corridor", r"dusty stairs"],
                    "tr": [r"koridor kirli", r"lobide kirli", r"merdiven tozlu"],
                    "ar": [r"الممر متسخ", r"المدخل متسخ", r"اللوبي متسخ"],
                    "zh": [r"走廊很脏", r"大堂很脏", r"楼道很脏"],
                },
            },
        },
    },

    "comfort": {
        "display": "Комфорт проживания",
        "subtopics": {
            "equipment": {
                "display": "Оснащение номера и удобства",
                "patterns": {
                    "ru": [r"кровать удобн", r"удобная кровать", r"удобный матрас", r"подушк.*удобн", r"есть фен", r"есть чайник", r"кофе в номере", r"есть все необходимое", r"не хватало розеток", r"не было фена"],
                    "en": [r"comfortable bed", r"good mattress", r"pillows were good", r"had a kettle", r"coffee in the room", r"no sockets", r"no hairdryer"],
                    "tr": [r"yatak rahattı", r"yatak konforlu", r"yastık iyiydi", r"su ısıtıcısı vardı", r"odalarda kahve", r"priz yoktu"],
                    "ar": [r"السرير مريح", r"المرتبة مريحة", r"المخدة مريحة", r"يوجد غلاية", r"لا يوجد مجفف شعر"],
                    "zh": [r"床很舒服", r"床垫舒服", r"枕头很好", r"有水壶", r"没有吹风机", r"没有插座"],
                },
            },
            "noise": {
                "display": "Шумоизоляция / тишина",
                "patterns": {
                    "ru": [r"шумно", r"тонкие стен", r"слышно сосед", r"слышно улиц", r"не выспаться из-за шума"],
                    "en": [r"noisy", r"thin walls", r"hear neighbors", r"street noise", r"hard to sleep because of noise"],
                    "tr": [r"çok gürültülü", r"duvarlar ince", r"komşular duyuluyor"],
                    "ar": [r"صاخب", r"جدران رقيقة", r"نسمع الجيران", r"ضجيج الشارع"],
                    "zh": [r"很吵", r"墙很薄", r"听到邻居", r"街上太吵"],
                },
            },
            "climate": {
                "display": "Температура / вентиляция / кондиционер",
                "patterns": {
                    "ru": [r"жарко", r"холодно", r"душно", r"кондиционер.*не работ", r"кондиционер шумит", r"вентилятор шумит", r"плохая вентиляци"],
                    "en": [r"too hot", r"too cold", r"stuffy", r"AC.*not working", r"aircon.*not working", r"AC noisy", r"ventilation noisy"],
                    "tr": [r"çok sıcak", r"çok soğuk", r"klima çalışmıyordu", r"klima çok gürültülü", r"hava almıyordu"],
                    "ar": [r"حار جدًا", r"بارد جدًا", r"مكيف لا يعمل", r"المكيف مزعج", r"تهوية سيئة"],
                    "zh": [r"太热", r"太冷", r"闷", r"空调不好用", r"空调太吵", r"通风不好"],
                },
            },
            "space_coziness": {
                "display": "Пространство и уют",
                "patterns": {
                    "ru": [r"уютн(ый|о)", r"красивый номер", r"стильно", r"просторно", r"тесно", r"маленькая комнат", r"очень маленький"],
                    "en": [r"cozy", r"stylish", r"nice design", r"spacious", r"too small", r"tiny room"],
                    "tr": [r"çok şık", r"çok güzel tasarım", r"çok küçük oda", r"ferah", r"dar"],
                    "ar": [r"مريح", r"جميل الديكور", r"صغير جدًا", r"ضيقة"],
                    "zh": [r"很温馨", r"房间很漂亮", r"房间很小", r"很挤", r"空间大"],
                },
            },
            "sleep_quality": {
                "display": "Качество сна",
                "patterns": {
                    "ru": [r"хорошо выспал", r"плохо спал", r"не выспал", r"сон плох", r"не смог уснуть"],
                    "en": [r"slept well", r"good sleep", r"couldn't sleep", r"hard to sleep"],
                    "tr": [r"iyi uyuduk", r"uyuyamadık", r"uyku rahatsızdı"],
                    "ar": [r"نمت جيدًا", r"لم أنم جيدًا"],
                    "zh": [r"睡得很好", r"没睡好", r"很难睡觉"],
                },
            },
        },
    },

    "tech_state": {
        "display": "Техническое состояние и инфраструктура",
        "subtopics": {
            "broken_things": {
                "display": "Поломки и неисправности",
                "patterns": {
                    "ru": [r"сломано", r"не работал", r"не работала", r"протекал", r"протечк", r"душ не работал", r"розетка не работал", r"лампа не горела"],
                    "en": [r"broken", r"not working", r"leaking", r"leak", r"shower didn't work", r"socket didn't work", r"light didn't work"],
                    "tr": [r"bozuk", r"çalışmıyordu", r"su sızdırıyor", r"duş çalışmıyordu"],
                    "ar": [r"مكسور", r"لا يعمل", r"تسريب", r"الدش لا يعمل"],
                    "zh": [r"坏了", r"不工作", r"漏水", r"淋浴坏了", r"插座不工作"],
                },
            },
            "water_heating": {
                "display": "Вода / сантехника / горячая вода",
                "patterns": {
                    "ru": [r"не было горячей воды", r"слабый напор", r"сантехник", r"текла вода", r"проблемы с душем"],
                    "en": [r"no hot water", r"low pressure", r"water pressure", r"plumbing issue"],
                    "tr": [r"sıcak su yoktu", r"su basıncı düşük", r"tesisat sorunu"],
                    "ar": [r"لا يوجد ماء ساخن", r"ضغط ماء ضعيف", r"مشكلة السباكة"],
                    "zh": [r"没有热水", r"水压很低", r"水管问题"],
                },
            },
            "wifi_quality": {
                "display": "Wi-Fi / интернет (качество)",
                "patterns": {
                    "ru": [r"wi[- ]?fi.*плох", r"интернет.*медлен", r"интернет не работал", r"вай[- ]?фай.*отваливался"],
                    "en": [r"wifi.*slow", r"wifi.*bad", r"internet.*slow", r"no internet", r"internet didn't work"],
                    "tr": [r"internet yavaştı", r"wifi kötüydü", r"wifi çalışmıyordu"],
                    "ar": [r"الواي فاي بطيء", r"انترنت سيء", r"لا يوجد انترنت"],
                    "zh": [r"网速很慢", r"无线网不好", r"wifi不好用", r"没有网络"],
                },
            },
            "building_access_tech": {
                "display": "Инженерные и технические вопросы доступа (замок, карта, лифт)",
                "patterns": {
                    "ru": [r"замок.*не работал", r"карта.*размагнит", r"лифт.*не работал", r"дверь.*не открывалась"],
                    "en": [r"lock.*not working", r"key card demagnetized", r"elevator.*not working", r"door wouldn't open"],
                    "tr": [r"kilit çalışmıyordu", r"kart bozuldu", r"asansör çalışmıyordu"],
                    "ar": [r"القفل لا يعمل", r"البطاقة لا تعمل", r"المصعد لا يعمل"],
                    "zh": [r"锁不好用", r"门卡坏了", r"电梯坏了", r"门打不开"],
                },
            },
        },
    },

    "breakfast": {
        "display": "Завтраки",
        "subtopics": {
            "food_quality": {
                "display": "Вкус / качество блюд",
                "patterns": {
                    "ru": [r"вкусн", r"невкусн", r"отвратител.*завтрак", r"потрясающ.*завтрак", r"завтрак отличный"],
                    "en": [r"breakfast.*delicious", r"breakfast.*tasty", r"breakfast.*awful", r"breakfast was terrible"],
                    "tr": [r"kahvaltı harikaydı", r"kahvaltı lezzetliydi", r"kahvaltı berbattı"],
                    "ar": [r"فطور رائع", r"فطور سيء", r"الإفطار كان لذيذ", r"الإفطار كان سيئًا"],
                    "zh": [r"早餐很好吃", r"早餐很差", r"早餐不好吃"],
                },
            },
            "variety": {
                "display": "Ассортимент / выбор",
                "patterns": {
                    "ru": [r"небольшой выбор", r"мало выбора", r"однообразн", r"разнообразн.*завтрак"],
                    "en": [r"little choice", r"not much choice", r"variety.*good", r"variety.*poor"],
                    "tr": [r"çeşit azdı", r"çeşit yoktu", r"çeşit çok iyiydi"],
                    "ar": [r"خيارات قليلة", r"تنوع ضعيف", r"تنوع جيد"],
                    "zh": [r"选择很少", r"种类不多", r"种类很多"],
                },
            },
            "availability_freshness": {
                "display": "Наличие и свежесть (температура, пополнение)",
                "patterns": {
                    "ru": [r"всё холодное", r"было холодно подано", r"ничего не осталось", r"не пополняли", r"не доложили еду"],
                    "en": [r"food was cold", r"everything cold", r"nothing left", r"not refilled"],
                    "tr": [r"yemek soğuktu", r"hiçbir şey kalmamıştı", r"yenilenmedi"],
                    "ar": [r"الطعام كان بارد", r"لم يبق شيء", r"لم يضيفوا المزيد"],
                    "zh": [r"都是冷的", r"什么都不剩", r"没有补菜"],
                },
            },
            "breakfast_staff": {
                "display": "Персонал завтрака (сервис в зоне питания)",
                "patterns": {
                    "ru": [r"девушки.*на завтраке.*вежлив", r"персонал.*завтрак.*невнимател", r"официант.*груб"],
                    "en": [r"breakfast staff.*friendly", r"breakfast staff.*rude", r"staff.*during breakfast.*helpful"],
                    "tr": [r"kahvaltı personeli güler yüzlüydü", r"kahvaltı personeli kaba"],
                    "ar": [r"الموظفين في الإفطار ودودين", r"طاقم الإفطار فظ"],
                    "zh": [r"早餐服务员很热情", r"早餐工作人员很差"],
                },
            },
            "area_cleanliness": {
                "display": "Чистота и организация зоны завтрака",
                "patterns": {
                    "ru": [r"грязн.*стол", r"грязн.*зал", r"грязн.*столов", r"грязная посуда", r"не убирают столы"],
                    "en": [r"dirty tables", r"dirty dishes", r"messy breakfast area"],
                    "tr": [r"masalar kirliydi", r"tabaklar kirliydi", r"kahvaltı alanı dağınıktı"],
                    "ar": [r"الطاولات متسخة", r"الأطباق متسخة"],
                    "zh": [r"早餐区很脏", r"桌子很脏", r"碗盘没收", r"没人收拾桌子"],
                },
            },
        },
    },

    "value": {
        "display": "Цена / ценность",
        "subtopics": {
            "value_for_money": {
                "display": "Цена соответствует качеству",
                "patterns": {
                    "ru": [r"цена.*соответств", r"стоил своих денег", r"дешев.*для такого уровня", r"дорог.*для такого уровня", r"слишком дорого", r"не стоит таких денег"],
                    "en": [r"worth the price", r"good value", r"overpriced", r"too expensive", r"not worth the money"],
                    "tr": [r"fiyatına değer", r"çok pahalı", r"paranın karşılığını vermiyor"],
                    "ar": [r"يسوى المال", r"غالي جدًا", r"لا يستحق المبلغ"],
                    "zh": [r"值这个价", r"太贵", r"性价比高", r"性价比低"],
                },
            },
            "hidden_charges": {
                "display": "Доплаты / скрытые платежи",
                "patterns": {
                    "ru": [r"доплат", r"скрыт.*платеж", r"дополнительн.*сбор", r"залог", r"депозит", r"списал(и)?.*без предупреждения"],
                    "en": [r"extra charge", r"hidden fee", r"charged.*without notice", r"deposit", r"they blocked my card"],
                    "tr": [r"ek ücret", r"gizli ücret", r"habersiz para çekildi", r"depozito aldı"],
                    "ar": [r"رسوم إضافية", r"رسوم مخفية", r"سحبوا مبلغ بدون علمي"],
                    "zh": [r"额外收费", r"隐藏收费", r"未提前告知就扣费", r"押金"],
                },
            },
        },
    },

    "location": {
        "display": "Локация и окружение",
        "subtopics": {
            "proximity": {
                "display": "Расположение и доступность",
                "patterns": {
                    "ru": [r"расположен.*удобн", r"в центр[е]?", r"близко к метро", r"рядом с достопримечательност", r"локац(ия|и[яй]) отличн"],
                    "en": [r"great location", r"close to metro", r"close to the center", r"walking distance", r"near attractions"],
                    "tr": [r"lokasyon harika", r"merkeze yakın", r"yürüyerek ulaşılabilir"],
                    "ar": [r"موقع ممتاز", r"قريب من المركز", r"بالقرب من المترو", r"قريب من المعالم"],
                    "zh": [r"位置很好", r"离市中心很近", r"离景点很近", r"交通方便"],
                },
            },
            "surroundings": {
                "display": "Окружение / подъездность / ощущение района",
                "patterns": {
                    "ru": [r"район.*небезопасн", r"странный подъезд", r"сложно найти вход", r"трудно найти вход", r"во двор", r"страшно ночью", r"подъезд грязный"],
                    "en": [r"hard to find entrance", r"entrance.*strange", r"area felt unsafe", r"sketchy area", r"scary at night", r"dirty entrance"],
                    "tr": [r"girşi zor bulundu", r"bina girişi zor", r"gece güvenli değil", r"bina girişi kirli"],
                    "ar": [r"صعب إيجاد المدخل", r"الحي غير آمن", r"مخيف ليلًا", r"المدخل متسخ"],
                    "zh": [r"入口不好找", r"晚上不安全", r"小区有点吓人", r"门口很脏"],
                },
            },
            "street_noise": {
                "display": "Шум с улицы / двор",
                "patterns": {
                    "ru": [r"шум с улиц", r"шум во двор", r"машины мешают", r"ночью шумно на улице"],
                    "en": [r"street noise", r"traffic noise", r"noisy outside at night"],
                    "tr": [r"sokak gürültüsü", r"gece dışarıdan gürültü"],
                    "ar": [r"ضجيج الشارع", r"إزعاج من الخارج ليلًا"],
                    "zh": [r"街上很吵", r"晚上外面很吵", r"车声很大"],
                },
            },
        },
    },
}


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

    # text_ru (потом сюда можно будет подвесить реальный перевод)
    df["text_ru"] = [
        translate_to_ru(txt, lg) for txt, lg in zip(df["text_orig"], df["lang_norm"])
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


