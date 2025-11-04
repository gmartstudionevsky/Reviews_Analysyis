"""
lexicon_module.py

Ядро лингвистики:
- словари тональностей (positive / negative / neutral, strong/soft)
- схема тем (категория -> подтема -> аспекты)
- правила аспектов (AspectRule)
- связи аспектов с подтемами

Эта штука:
1. Компилирует regex'ы один раз.
2. Даёт методы для:
   - определения тональности фрагмента текста,
   - извлечения аспектов из предложения,
   - маппинга аспектов к категориям / подтемам,
   - получения подсказок по полярности аспекта.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Iterable, Optional
import re
import logging


###############################################################################
# 1. Классы данных
###############################################################################

@dataclass(frozen=True)
class AspectRule:
    """
    Правило аспекта (единица смысла, которую мы хотим отслеживать в отзывах).

    aspect_code:
        Машинное имя аспекта, например "smell_of_smoke" или "spir_friendly".
        Это ключ, через который всё агрегируется.

    patterns_by_lang:
        { "ru": [regex1, ...], "en": [...], ... }
        Регексы, которые сигналят, что аспект упомянут.
        (Не тональность, именно "что человек заговорил об этом аспекте".)

    polarity_hint:
        'positive' / 'negative' / 'neutral'.
        Используем как дефолтное направление — например, "smell_of_smoke"
        почти всегда негативный сигнал.

    display:
        Чуть более формальное человекочитаемое имя аспекта.
        Можно использовать в технических сводках или внутренних таблицах.
        Пример: "Грубость персонала".

    display_short:
        Короткий ярлык, уже в финальном стиле для дашборда/отчёта.
        Пример: "запах сигарет в общих зонах",
                 "грубое отношение персонала",
                 "быстрое заселение".
        Это мы покажем бизнесу в столбцах.

    long_hint:
        Контекстная подсказка для менеджмента, 1-2 предложения.
        Смысл: "как это читают гости?"
        Это пойдёт в описательные блоки отчёта (инсайты/выводы).
        Пример: "Гости пишут, что в коридорах пахнет сигаретами/дымом."

    Почему три разных поля (display / display_short / long_hint)?
    - display_short обычно звучит как конкретная боль/радость на человеческом языке.
    - long_hint — это объяснение проблемы своими словами.
    - display — можем оставить как fallback/техническое имя, либо вообще не использовать.
    """
    aspect_code: str
    patterns_by_lang: Dict[str, List[str]]
    polarity_hint: str
    display: Optional[str] = None
    display_short: Optional[str] = None
    long_hint: Optional[str] = None



###############################################################################
# 2. Лексикон тональностей #
###############################################################################


POSITIVE_WORDS_STRONG: Dict[str, List[str]] = {
                    "ru": [
                        r"\bидеальн", r"\bпревосходн", r"\bпотрясающе\b", r"\bвеликолепн",
                        r"\bшикарн", r"\bсупер\b", r"\bлучший опыт\b", r"\bлучшее место\b",
                        r"\bочень понравил", r"\bв восторге\b", r"\bобожаю\b",
                        r"\bверн[уё]мся обязательно\b", r"\bоднозначно рекомендую\b",
                        r"\bвсё просто отлично\b", r"\bбезупречн", r"\bчисто идеально\b",
                    ],
                    "en": [
                        r"\bamazing\b", r"\bawesome\b", r"\bperfect\b", r"\bflawless\b",
                        r"\bexceptional\b", r"\bexcellent\b", r"\boutstanding\b",
                        r"\bloved it\b", r"\bi loved\b", r"\bhighly recommend\b",
                        r"\bdefinitely recommend\b", r"\bwe will come back\b",
                        r"\bspotless\b", r"\bimmaculate\b", r"\bfantastic\b",
                    ],
                    "tr": [
                        r"\bharika\b", r"\bmükemmel\b", r"\bkusursuz\b", r"\bşahane\b",
                        r"\bçok beğendik\b", r"\bçok memnun kaldık\b",
                        r"\bkesinlikle tavsiye ederim\b", r"\btekrar geleceğiz\b",
                    ],
                    "ar": [
                        r"\bرائع\b", r"\bممتاز\b", r"\bمذهل\b", r"\bمثالي\b", r"\bافضل تجربة\b",
                        r"\bسأعود بالتأكيد\b", r"\bأنصح بشدة\b",
                    ],
                    "zh": [
                        r"非常好", r"太棒了", r"完美", r"极好", r"超赞", r"特别满意",
                        r"非常满意", r"强烈推荐", r"一定会再来", r"无可挑剔",
                    ],
                }

POSITIVE_WORDS_SOFT: Dict[str, List[str]] = {
                    "ru": [
                        r"\bхорошо\b", r"\bочень хорошо\b", r"\bдоволен\b", r"\bдовольн",
                        r"\bприятно\b", r"\bвсё ок\b", r"\bвсе ок\b", r"\bвсё было ок\b",
                        r"\bв целом понравил", r"\bприятный опыт\b", r"\bчисто\b", r"\bуютн",
                        r"\bкомфортн", r"\bудобн", r"\bвежлив", r"\bдоброжелательн",
                        r"\bрадушн", r"\bдружелюб", r"\bгостеприимн",
                        r"\bприняли хорошо\b", r"\bбыстро заселили\b", r"\bбыстро поселили\b",
                    ],
                    "en": [
                        r"\bgood\b", r"\bvery good\b", r"\bnice\b", r"\bpleasant\b",
                        r"\bcomfortable\b", r"\bcozy\b", r"\bclean\b",
                        r"\bfriendly staff\b", r"\bpolite staff\b", r"\bhelpful staff\b",
                        r"\bwelcoming\b", r"\bquick check[- ]?in\b", r"\bfast check[- ]?in\b",
                        r"\bno issues\b", r"\bno problems\b",
                    ],
                    "tr": [
                        r"\bi̇yi\b", r"\bçok iyi\b", r"\bgayet iyi\b", r"\brahat\b",
                        r"\btemiz\b", r"\bgüler yüzlü\b", r"\byardımsever\b",
                        r"\bmisafirperver\b", r"\bhızlı check[- ]?in\b", r"\bsorun yoktu\b",
                    ],
                    "ar": [
                        r"\bجيد\b", r"\bجيد جدًا\b", r"\bمرتاح\b", r"\bنظيف\b", r"\bمريح\b",
                        r"\bخدمة لطيفة\b", r"\bاستقبال جيد\b", r"\bلا توجد مشكلة\b",
                    ],
                    "zh": [
                        r"很好", r"不错", r"满意", r"挺好", r"干净", r"舒适",
                        r"服务很好", r"员工很友好", r"入住很快",
                        r"没问题", r"一切都可以", r"还可以", r"可以接受",
                    ],
                }

NEGATIVE_WORDS_SOFT: Dict[str, List[str]] = {
                    "ru": [
                        r"\bне очень\b", r"\bмогло бы быть лучше\b", r"\bсредне\b",
                        r"\bтак себе\b", r"\bожидал(и)? лучше\b",
                        r"\bразочаров", r"\bне впечатлил", r"\bесть недочеты\b",
                        r"\bнемного грязн", r"\bчуть грязн", r"\bшумновато\b", r"\bслегка шумно\b",
                        r"\bнекомфортно\b", r"\bнеудобно\b", r"\bнеудобн(ая|ый|о)\b",
                        r"\bждали д(олг|олго)\b", r"\bдолго ждали\b",
                        r"\bподождать пришлось\b",
                        r"\bпроблемы с заселением\b", r"\bне сразу заселили\b",
                        r"\bкомната ещё не была готова\b",
                    ],
                    "en": [
                        r"\bnot great\b", r"\bnot very good\b", r"\bcould be better\b",
                        r"\baverage\b", r"\bdisappoint", r"\bunderwhelming\b",
                        r"\ba bit dirty\b", r"\ba little dirty\b",
                        r"\bnoisy\b", r"\bquite noisy\b", r"\ba bit noisy\b",
                        r"\buncomfortable\b", r"\binconvenient\b",
                        r"\bhad to wait\b", r"\bwaited a while\b",
                        r"\broom not ready\b",
                    ],
                    "tr": [
                        r"\bok değildi\b", r"\bo kadar iyi değil\b",
                        r"\bordinerd(i|i)\b", r"\bortalama\b",
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
                        r"一般", r"有点失望", r"不算好", r"有点脏",
                        r"有点吵", r"有点不舒服", r"等了很久",
                        r"房间还没准备好",
                        r"有点麻烦", r"不是很方便",
                    ],
                }
            

NEGATIVE_WORDS_STRONG: Dict[str, List[str]] = {
                    "ru": [
                        r"\bужасн", r"\bкошмар", r"\bкатастроф", r"\bотвратител",
                        r"\bмерзко\b", r"\bгрязь\b", r"\bгрязно\b", r"\bвонял",
                        r"\bвонь\b", r"\bплесень\b", r"\bплесн[ью]\b",
                        r"\bгромко\b", r"\bочень шумно\b", r"\bневыносимо\b",
                        r"\bневозможно спать\b",
                        r"\bобман\b", r"\bскрыт(ые|ые) платеж", r"\bнадули\b",
                        r"\bунизительно\b",
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
                        r"\bimpossible to sleep\b", r"\bno sleep\b",
                        r"\bso noisy\b", r"\bextremely noisy\b",
                    ],
                    "tr": [
                        r"\bberbat\b", r"\brezalet\b", r"\biğrenç\b", r"\bçok pis\b",
                        r"\bleş gibi kokuyordu\b", r"\bküf\b",
                        r"\baldatıldık\b", r"\bdolandırıcılık\b", r"\bgizli ücretler\b",
                        r"\bbir daha asla\b", r"\btavsiye etmiyorum\b",
                        r"\bçok kaba\b", r"\bhaşin\b", r"\başağılayıcı\b",
                        r"\buyuyamadık\b", r"\büstümüzde gürültü\b", r"\bçok gürültülü\b",
                    ],
                    "ar": [
                        r"\bسيئ جدًا\b", r"\bفظيع\b", r"\bقذر جدًا\b", r"\bمقرف\b",
                        r"\bرائحة كريهة\b",
                        r"\bاحتيال\b", r"\bنصب\b", r"\bرسوم مخفية\b",
                        r"\bأبدًا مرة أخرى\b", r"\bلا أنصح\b",
                        r"\bوقحين جدًا\b", r"\bغير محترمين\b", r"\bأهانونا\b",
                        r"\bمستحيل النوم\b", r"\bضجيج لا يحتمل\b",
                    ],
                    "zh": [
                        r"太糟糕", r"很糟", r"恶心", r"肮脏", r"非常脏",
                        r"有霉味", r"发霉", r"特别臭",
                        r"被骗", r"坑钱", r"乱收费", r"隐形消费",
                        r"绝不会再来", r"不推荐",
                        r"服务员很粗鲁", r"态度很差",
                        r"吵得没法睡", r"完全睡不着", r"太吵了", r"受不了",
                    ],
                }

NEUTRAL_WORDS: Dict[str, List[str]] = {
                    "ru": [
                        r"\bнормально\b", r"\bнорм\b", r"\bнормал(ьно|ьный)\b",
                        r"\bв целом норм\b", r"\bтерпимо\b", r"\bсойдёт\b", r"\bсойдет\b",
                        r"\bприемлемо\b", r"\bвполне сносно\b",
                        r"\bбез особых проблем\b",
                        r"\bничего страшного\b", r"\bничего критичного\b",
                        r"\bдля одной ночи ок\b",
                        r"\bдля одной ночи нормально\b",
                    ],
                    "en": [
                        r"\bok\b", r"\bokay\b", r"\bfine\b", r"\ball right\b",
                        r"\bacceptable\b", r"\bdecent\b",
                        r"\bit was ok\b", r"\bit was fine\b",
                        r"\bnothing special\b", r"\bnothing crazy\b",
                        r"\bgood for one night\b",
                        r"\bfor one night it's fine\b",
                    ],
                    "tr": [
                        r"\bidare eder\b", r"\bfena değil\b", r"\bkötü değil\b",
                        r"\btamamdır\b", r"\bokeydi\b", r"\baynen\b",
                        r"\bbir gece için yeterli\b",
                    ],
                    "ar": [
                        r"\bلا بأس\b", r"\bمقبول\b", r"\bعلى ما يرام\b",
                        r"\bجيد بشكل عام\b",
                        r"\bكافي لليلة واحدة\b",
                    ],
                    "zh": [
                        r"还行", r"可以", r"凑合", r"勉强可以",
                        r"没什么大问题", r"总体可以", r"一般般", r"还好",
                        r"住一晚还行",
                    ],
                }

# Сводим всё в единую структуру:
# ключ sentiment_key -> словарь lang -> [regex,...]
SENTIMENT_LEXICON: Dict[str, Dict[str, List[str]]] = {
    "positive_strong": POSITIVE_WORDS_STRONG,
    "positive_soft": POSITIVE_WORDS_SOFT,
    "negative_soft": NEGATIVE_WORDS_SOFT,
    "negative_strong": NEGATIVE_WORDS_STRONG,
    "neutral": NEUTRAL_WORDS,
}

# Маппинг "вид тональности" -> "глобальная полярность"
# Это нужно, чтобы мы могли потом сказать "positive"/"negative"/"neutral"
SENTIMENT_KEY_TO_GROUP: Dict[str, str] = {
    "positive_strong": "positive",
    "positive_soft": "positive",
    "negative_soft": "negative",
    "negative_strong": "negative",
    "neutral": "neutral",
}

# Приоритет матчинга, когда мы определяем тональность предложения.
# Логика:
#   - сильный негатив важнее сильного позитива;
#   - потом мягкий негатив;
#   - потом сильный позитив;
#   - потом мягкий позитив;
#   - потом нейтрально.
# Можно варьировать, но это даёт "сигнал проблем" приоритетнее.
SENTIMENT_EVAL_ORDER: List[str] = [
    "negative_strong",
    "negative_soft",
    "positive_strong",
    "positive_soft",
    "neutral",
]

###############################################################################
# 3. Тематическая схема (категория -> подтемы -> аспекты)
############################################################################

# Здесь мы напрямую используем предоставленный TOPIC_SCHEMA,
# но храним его уже как объекты TopicCategory/Subtopic.

TOPIC_SCHEMA: Dict[str, Dict[str, Any]] = {
            "staff_spir": {
                "display":"Персонал",
                "subtopics": {
                    "staff_attitude": {
                      "display": "Отношение и вежливость",
                      "patterns_by_lang":{
                            "ru": [
                                r"\bвежлив", r"\bдоброжелательн", r"\bдружелюб",
                                r"\bприветлив", r"\bрадушн", r"\bтепло встретил",
                                r"\bотзывчив", r"\bс улыбкой",
                                r"\bхамил", r"\bхамство", r"\bнагруб",
                                r"\bгруб(о|ые)", r"\bнеприветлив",
                                r"\bнедружелюб", r"\bразговаривал[аи]? свысока",
                            ],
                            "en": [
                                r"\bfriendly staff\b", r"\bvery friendly\b", r"\bwelcoming\b",
                                r"\bpolite\b", r"\bkind\b",
                                r"\brude staff\b", r"\bunfriendly\b", r"\bimpolite\b",
                                r"\bdisrespectful\b", r"\btreated us badly\b",
                            ],
                            "tr": [
                                r"\bgüler yüzlü\b", r"\bnazik\b", r"\bkibar\b",
                                r"\bsıcak karşıladılar\b",
                                r"\bçok kaba\b", r"\bsaygısız\b",
                                r"\bters davrandılar\b",
                            ],
                            "ar": [
                                r"\bموظفين لطيفين\b", r"\bاستقبال دافئ\b",
                                r"\bتعامل محترم\b", r"\bابتسامة\b",
                                r"\bموظفين وقحين\b", r"\bسيء التعامل\b",
                                r"\bغير محترمين\b",
                            ],
                            "zh": [
                                r"服务很友好", r"前台很热情", r"态度很好", r"很有礼貌",
                                r"态度很差", r"服务很差", r"很不耐烦", r"不礼貌", r"很凶",
                            ],
                        },
                        "aspects": [
                            "spir_friendly",
                            "spir_polite",
                            "spir_rude",
                            "spir_unrespectful",
                        ],
                },
                    "staff_helpfulness": {
                        "display": "Помощь и решение вопросов",
                        "patterns_by_lang":{
                            "ru": [
                                r"\bпомог(ли|ли нам)\b",
                                r"\bрешил[аи]? вопрос\b",
                                r"\bрешили проблему\b",
                                r"\bвсё объяснил[аи]?\b",
                                r"\bвсё подсказал[аи]?\b",
                                r"\bподсказали куда\b",
                                r"\bдали рекомендации\b",
                                r"\bбыстро отреагировал[аи]?\b",
                                r"\bне помогли\b", r"\bничего не сделали\b",
                                r"\bне решили\b",
                                r"\bсказали это не наша проблема\b",
                                r"\bпроигнорировали\b",
                            ],
                            "en": [
                                r"\bhelpful\b", r"\bvery helpful\b",
                                r"\bsolved the issue\b",
                                r"\bfixed it quickly\b",
                                r"\bassisted us\b",
                                r"\bgave recommendations\b",
                                r"\bexplained everything\b",
                                r"\bnot helpful\b", r"\bunhelpful\b",
                                r"\bignored us\b",
                                r"\bdidn't solve\b",
                                r"\bno assistance\b",
                                r"\bnot their problem\b",
                            ],
                            "tr": [
                                r"\byardımcı oldular\b",
                                r"\bhemen çözdüler\b",
                                r"\bbize anlattılar\b",
                                r"\byönlendirdiler\b",
                                r"\byardımcı olmadılar\b",
                                r"\bilgilenmediler\b",
                                r"\bsorunu çözmediler\b",
                            ],
                            "ar": [
                                r"\bساعدونا\b", r"\bحلّوا المشكلة\b",
                                r"\bشرحوا كل شيء\b",
                                r"\bاستجابوا بسرعة\b",
                                r"\bلم يساعدونا\b",
                                r"\bتجاهلونا\b",
                                r"\bلم يحلوا المشكلة\b",
                                r"\bقالوا ليست مشكلتنا\b",
                            ],
                            "zh": [
                                r"很帮忙", r"很乐于助人", r"马上处理",
                                r"马上解决", r"给我们解释",
                                r"给了建议",
                                r"不帮忙", r"不理我们",
                                r"没解决", r"让我们自己处理",
                            ],
                        },
                        "aspects": [
                            "spir_helpful_fast",
                            "spir_problem_solved",
                            "spir_info_clear",
                            "spir_unhelpful",
                            "spir_problem_ignored",
                            "spir_info_confusing",
                        ],
                    },
                    "staff_speed": {
                        "display": "Оперативность и скорость реакции",
                        "patterns_by_lang":{
                            "ru": [
                                r"\bбыстро заселили\b",
                                r"\bмоментально заселили\b",
                                r"\bоформили быстро\b",
                                r"\bреагируют быстро\b",
                                r"\bпришли сразу\b",
                                r"\bоперативно\b",
                                r"\bждали долго\b",
                                r"\bпришлось долго ждать\b",
                                r"\bникого не было на ресепшен",
                                r"\bне могли дозвониться\b",
                                r"\bне брали трубку\b",
                                r"\bдолго оформляли\b",
                            ],
                            "en": [
                                r"\bquick check[- ]?in\b",
                                r"\bfast check[- ]?in\b",
                                r"\bresponded immediately\b",
                                r"\bthey came right away\b",
                                r"\bhandled it quickly\b",
                                r"\bhad to wait a long time\b",
                                r"\bno one answered\b",
                                r"\bnobody at the desk\b",
                                r"\bslow check[- ]?in\b",
                                r"\btook too long\b",
                            ],
                            "tr": [
                                r"\bhızlı check[- ]?in\b",
                                r"\bçok hızlı ilgilendiler\b",
                                r"\bhemen geldiler\b",
                                r"\banında yardımcı oldular\b",
                                r"\bçok bekledik\b",
                                r"\bresepsiyonda kimse yoktu\b",
                                r"\btelefon açmadılar\b",
                                r"\bgeç cevap verdiler\b",
                            ],
                            "ar": [
                                r"\bتسجيل دخول سريع\b",
                                r"\bاستجابوا فورًا\b",
                                r"\bجاءوا مباشرة\b",
                                r"\bسريع جدًا\b",
                                r"\bانتظرنا كثيرًا\b",
                                r"\bلم يرد أحد\b",
                                r"\bلا أحد في الاستقبال\b",
                                r"\bبطيء جدًا\b",
                            ],
                            "zh": [
                                r"办理入住很快",
                                r"马上处理",
                                r"很快就来了",
                                r"反应很快",
                                r"等了很久",
                                r"前台没人",
                                r"没人接电话",
                                r"太慢了",
                                r"入住很慢",
                            ],
                        },
                        "aspects": [
                            "spir_helpful_fast",
                            "spir_fast_response",
                            "spir_slow_response",
                            "spir_absent",
                            "spir_no_answer",
                        ],
                    },
                    "staff_professionalism": {
                        "display": "Профессионализм и компетентность",
                        "patterns_by_lang":{
                            "ru": [
                                r"\bпрофессионал",
                                r"\bкомпетентн",
                                r"\bвсё чётко объяснил",
                                r"\bвсё грамотно объяснила",
                                r"\bвсё прозрачно\b",
                                r"\bоформили документы\b",
                                r"\bдали все чеки\b",
                                r"\bнекомпетентн",
                                r"\bне знают\b",
                                r"\bбардак с документами\b",
                                r"\bне смогли объяснить оплату\b",
                                r"\bошиблись в брон[иь]\b",
                                r"\bошибка в сч(е|ё)те\b",
                                r"\bпутаница с оплатой\b",
                                r"\bнепрозрачно\b",
                            ],
                            "en": [
                                r"\bprofessional\b",
                                r"\bvery professional\b",
                                r"\bknowledgeable\b",
                                r"\bclear explanation\b",
                                r"\btransparent\b",
                                r"\bsorted all paperwork\b",
                                r"\bgave invoice\b",
                                r"\bunprofessional\b",
                                r"\bdidn't know\b",
                                r"\bconfused about payment\b",
                                r"\bmessed up reservation\b",
                                r"\bwrong charge\b",
                                r"\bbilling mistake\b",
                            ],
                            "tr": [
                                r"\bprofesyonel\b",
                                r"\bçok profesyonel\b",
                                r"\bişini biliyor\b",
                                r"\baçıkça anlattı\b",
                                r"\bfaturayı düzgün verdiler\b",
                                r"\bprofesyonel değildi\b",
                                r"\bbilmiyorlardı\b",
                                r"\bödeme konusunda karışıklık\b",
                                r"\byanlış ücret\b",
                                r"\brezervasyonu karıştırdılar\b",
                            ],
                            "ar": [
                                r"\bمحترفين\b",
                                r"\bيعرفون شغلهم\b",
                                r"\bشرح واضح\b",
                                r"\bكل شيء كان واضح بالدفع\b",
                                r"\bأعطونا كل الفواتير\b",
                                r"\bغير محترفين\b",
                                r"\bمش فاهمين الإجراءات\b",
                                r"\bخطأ في الحجز\b",
                                r"\bخطأ في الفاتورة\b",
                                r"\bمش واضح بالدفع\b",
                            ],
                            "zh": [
                                r"很专业",
                                r"非常专业",
                                r"解释很清楚",
                                r"流程很清楚",
                                r"收费很透明",
                                r"单据都给了",
                                r"不专业",
                                r"搞不清楚",
                                r"解释不清楚",
                                r"收费不明",
                                r"账单有问题",
                                r"搞错预订",
                            ],
                        },
                        "aspects": [
                            "spir_professional",
                            "spir_info_clear",
                            "spir_payment_clear",
                            "spir_unprofessional",
                            "spir_info_confusing",
                            "spir_payment_issue",
                            "spir_booking_mistake",
                        ],
                    },
                    "staff_availability": {
                        "display": "Доступность персонала",
                        "patterns_by_lang":{
                            "ru": [
                                r"\bна связи 24\b",
                                r"\bкруглосуточно помогали\b",
                                r"\bдаже ночью помогли\b",
                                r"\bответили ночью\b",
                                r"\bбыли всегда доступны\b",
                                r"\bна ресепшен[е]? никого\b",
                                r"\bникого не было на стойке\b",
                                r"\bне дозвониться ночью\b",
                                r"\bникто не приш[её]л\b",
                                r"\bресепшн закрыт ночью\b",
                            ],
                            "en": [
                                r"\b24/7\b",
                                r"\balways available\b",
                                r"\beven at night they helped\b",
                                r"\bnight staff was helpful\b",
                                r"\banswered phone at night\b",
                                r"\bno one at the desk\b",
                                r"\bnobody at reception\b",
                                r"\bno answer at night\b",
                                r"\bcouldn't reach anyone\b",
                                r"\breception closed at night\b",
                            ],
                            "tr": [
                                r"\b24 saat ulaşılabilir\b",
                                r"\bgecede bile yardımcı oldular\b",
                                r"\bgece personeli çok yardımcı\b",
                                r"\bresepsiyonda kimse yoktu\b",
                                r"\bgece kimse yoktu\b",
                                r"\bgece kimse cevap vermedi\b",
                            ],
                            "ar": [
                                r"\bمتوفرين طول الوقت\b",
                                r"\bحتى بالليل ساعدونا\b",
                                r"\bردوا علينا في الليل\b",
                                r"\bمافي أحد بالاستقبال\b",
                                r"\bبالليل ما حد يرد\b",
                                r"\bمغلق بالليل\b",
                                r"\bمافي دعم ليلي\b",
                            ],
                            "zh": [
                                r"24小时有人",
                                r"半夜也有人帮忙",
                                r"晚上也能联系到",
                                r"夜班也很负责",
                                r"前台没人",
                                r"晚上没人",
                                r"打电话没人接",
                                r"夜里没人管",
                            ],
                        },
                        "aspects": [
                            "spir_available",
                            "spir_24h_support",
                            "spir_absent",
                            "spir_no_answer",
                            "spir_no_night_support",
                        ],
                    },
                    "staff_communication": {
                        "display": "Коммуникация и понятность объяснений",
                        "patterns_by_lang":{
                            "ru": [
                                r"\bвсё понятн[оы] объяснил",
                                r"\bподробно рассказал",
                                r"\bинструкции понятные\b",
                                r"\bобъяснили как зайти\b",
                                r"\bобъяснили куда идти\b",
                                r"\bвсё разжевали\b",
                                r"\bничего не объяснили\b",
                                r"\bнепонятные инструкции\b",
                                r"\bобъясняли как попало\b",
                                r"\bне смогли объяснить\b",
                                r"\bя не понял\b",
                                r"\bя не поняла\b",
                                r"\bпутались\b",
                            ],
                            "en": [
                                r"\bclear instructions\b",
                                r"\bexplained everything clearly\b",
                                r"\beasy to understand\b",
                                r"\bcommunicated clearly\b",
                                r"\bgood English\b",
                                r"\bspoke English well\b",
                                r"\bhard to understand\b",
                                r"\bunclear instructions\b",
                                r"\bpoor communication\b",
                                r"\bnobody speaks English\b",
                                r"\blanguage barrier\b",
                                r"\bcouldn't explain\b",
                            ],
                            "tr": [
                                r"\bher şeyi açıkladılar\b",
                                r"\btalimatlar çok netti\b",
                                r"\bingilizce konuşabiliyorlardı\b",
                                r"\banlaşılması zordu\b",
                                r"\btalimatlar net değildi\b",
                                r"\bingilizce konuşmuyorlar\b",
                            ],
                            "ar": [
                                r"\bشرح واضح\b",
                                r"\bفسروا كل شيء\b",
                                r"\bالتعليمات كانت واضحة\b",
                                r"\bتعليمات غير واضحة\b",
                                r"\bصعب نفهم\b",
                                r"\bما يحكوا انجليزي\b",
                                r"\bحاجز لغة\b",
                            ],
                            "zh": [
                                r"解释得很清楚",
                                r"指示很清晰",
                                r"英文很好",
                                r"沟通很顺",
                                r"听不懂",
                                r"解释不清楚",
                                r"沟通不好",
                                r"没有说明清楚",
                                r"语言有问题",
                            ],
                        },
                        "aspects": [
                            "spir_info_clear",
                            "spir_language_ok",
                            "spir_info_confusing",
                            "spir_language_barrier",
                        ],
                    },
                },
            },
            
            "checkin_stay": {
                "display": "Заселение и проживание",
                "subtopics": {
                    "checkin_speed": {
                        "display": "Скорость заселения",
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
                        "patterns_by_lang":{
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
            },
        }


###############################################################################
# "Словарь" аспектов
#
# ASPECT_RULES:
#   аспект_code -> AspectRule
#
# ASPECT_TO_SUBTOPICS:
#   аспект_code -> List[(category_key, subtopic_key)]
#
# Эти тексты будут использоваться в отчёте (буллеты, подписи графиков).
# Здесь мы даём короткие человекочитаемые ярлыки.
###############################################################################

ASPECT_RULES: Dict[str, AspectRule] = {

        "spir_friendly": AspectRule(
            aspect_code="spir_friendly",
            polarity_hint="positive",
            display="Доброжелательность и дружелюбие персонала СПиР",
            display_short="дружелюбие персонала",
            long_hint="Гости отмечают, что сотрудники СПиР ведут себя доброжелательно и открыто, общаются с улыбкой и создают ощущение человеческого, неформального внимания.",
            patterns_by_lang={{
                "ru": [
                    r"\bдружелюб", 
                    r"\bприветлив", 
                    r"\bдоброжелательн", 
                    r"\bотзывчив", 
                    r"\bс улыбкой\b",
                ],
                "en": [
                    r"\bfriendly staff\b",
                    r"\bvery friendly\b",
                    r"\bso friendly\b",
                    r"\bpolite and friendly\b",
                ],
                "tr": [
                    r"\bgüler yüzlü\b",
                    r"\bnazik\b",
                    r"\bkibar\b",
                ],
                "ar": [
                    r"\bموظفين لطيفين\b",
                    r"\bتعامل محترم\b",
                    r"\bابتسامة\b",
                ],
                "zh": [
                    r"服务很友好",
                    r"态度很好",
                    r"很有礼貌",
                ],
            },
        ),
        
        
        "spir_welcoming": AspectRule(
            aspect_code="spir_welcoming",
            polarity_hint="positive",
            display="Тёплый приём при заезде",
            display_short="тёплый приём",
            long_hint="Гости фиксируют, что при контакте с СПиР при заселении их встретили тепло и по-домашнему, создали ощущение гостеприимства с первых минут.",
            patterns_by_lang={{
                "ru": [
                    r"\bрадушн",
                    r"\bтепло встретил",
                    r"\bвстретил[аи]? очень тепло\b",
                    r"\bприветствовал[аи]? очень тепло\b",
                ],
                "en": [
                    r"\bwelcoming\b",
                    r"\bwelcomed us\b",
                    r"\bwarm welcome\b",
                    r"\bmade us feel welcome\b",
                ],
                "tr": [
                    r"\bsıcak karşıladılar\b",
                    r"\bçok sıcak karşıladılar\b",
                ],
                "ar": [
                    r"\bاستقبال دافئ\b",
                    r"\bاستقبلونا بحرارة\b",
                ],
                "zh": [
                    r"前台很热情",
                    r"热情接待",
                    r"热情欢迎我们",
                ],
            },
        ),
        
        
        "spir_helpful": AspectRule(
            aspect_code="spir_helpful",
            polarity_hint="positive",
            display="Готовность помочь и фактическая помощь СПиР",
            display_short="персонал помог",
            long_hint="Гости отмечают, что сотрудники СПиР не просто были вежливы, но активно помогали: объясняли порядок заселения, давали рекомендации, решали запросы и проблемы на месте.",
            patterns_by_lang={{
                "ru": [
                    r"\bпомог(ли|ли нам)\b",
                    r"\bрешил[аи]? вопрос\b",
                    r"\bрешили проблему\b",
                    r"\bвсё объяснил[аи]?\b",
                    r"\bподсказал[аи]?\b",
                    r"\bдали рекомендации\b",
                ],
                "en": [
                    r"\bhelpful\b",
                    r"\bvery helpful\b",
                    r"\bassisted us\b",
                    r"\bsolved the issue\b",
                    r"\bfixed it quickly\b",
                    r"\bgave recommendations\b",
                    r"\bexplained everything\b",
                ],
                "tr": [
                    r"\byardımcı oldular\b",
                    r"\bhemen çözdüler\b",
                    r"\bbize anlattılar\b",
                    r"\byönlendirdiler\b",
                ],
                "ar": [
                    r"\bساعدونا\b",
                    r"\bحلّوا المشكلة\b",
                    r"\bشرحوا كل شيء\b",
                    r"\bاستجابوا بسرعة\b",
                ],
                "zh": [
                    r"很帮忙",
                    r"很乐于助人",
                    r"马上处理",
                    r"马上解决",
                    r"给我们解释",
                    r"给了建议",
                ],
            },
        ),
        
        
        "spir_unfriendly": AspectRule(
            aspect_code="spir_unfriendly",
            polarity_hint="negative",
            display="Недоброжелательность персонала СПиР",
            display_short="недружелюбный персонал",
            long_hint="Гости фиксируют холодное, формальное или отстранённое общение со стороны сотрудников СПиР: отсутствие вовлечённости, неготовность помочь, отсутствие базовой эмпатии.",
            patterns_by_lang={{
                "ru": [
                    r"\bнеприветлив",
                    r"\bнедружелюб",
                    r"\bбез улыбки\b",
                    r"\bхолодно общал[и]сь\b",
                    r"\bразговаривал[аи]? свысока\b",
                ],
                "en": [
                    r"\bunfriendly\b",
                    r"\bnot friendly\b",
                    r"\bcold staff\b",
                    r"\bdidn't smile\b",
                    r"\btreated us coldly\b",
                ],
                "tr": [
                    r"\bsoğuk davrandılar\b",
                    r"\bgüler yüzlü değillerdi\b",
                    r"\bsamimi değillerdi\b",
                ],
                "ar": [
                    r"\bمش ودودين\b",
                    r"\bمش لطيفين\b",
                    r"\bتعامل بارد\b",
                    r"\bبدون ابتسامة\b",
                ],
                "zh": [
                    r"态度冷淡",
                    r"服务很冷淡",
                    r"没有笑脸",
                    r"不热情",
                ],
            },
        ),
        
        
        "spir_rude": AspectRule(
            aspect_code="spir_rude",
            polarity_hint="negative",
            display="Грубость и неуважительное общение со стороны СПиР",
            display_short="грубое обращение",
            long_hint="Гости сообщают о случаях повышенного тона, хамства, резких формулировок или открытого неуважения со стороны сотрудников СПиР.",
            patterns_by_lang={{
                "ru": [
                    r"\bхамил",
                    r"\bхамство\b",
                    r"\bнагруб",
                    r"\bгруб(о|ые)\b",
                    r"\bнеуважительно\b",
                    r"\bразговаривал[аи]? свысока\b",
                ],
                "en": [
                    r"\brude staff\b",
                    r"\brude\b",
                    r"\bimpolite\b",
                    r"\bdisrespectful\b",
                    r"\btreated us badly\b",
                    r"\byelled at us\b",
                ],
                "tr": [
                    r"\bçok kaba\b",
                    r"\bsaygısız\b",
                    r"\bters davrandılar\b",
                    r"\bbize bağırdılar\b",
                ],
                "ar": [
                    r"\bموظفين وقحين\b",
                    r"\bغير محترمين\b",
                    r"\bسيء التعامل\b",
                    r"\bصرخوا علينا\b",
                ],
                "zh": [
                    r"态度很差",
                    r"服务很差",
                    r"很不耐烦",
                    r"不礼貌",
                    r"很凶",
                    r"对我们吼",
                ],
            },
        ),

        "spir_unprofessional": AspectRule(
            aspect_code="spir_unprofessional",
            polarity_hint="negative",
            display="Непрофессиональность сотрудников СПиР",
            display_short="непрофессиональное поведение персонала",
            long_hint="Гости фиксируют, что персонал СПиР действует непрофессионально: не знает базовую информацию, ведёт себя несобранно, допускает ошибки в коммуникации или в оформлении.",
            patterns_by_lang={{
                "ru": [
                    r"\bнепрофессионал",
                    r"\bне компетентн",
                    r"\bне знали что делать\b",
                    r"\bбардак на ресепшене\b",
                    r"\bпутаница с документами\b",
                    r"\bперепутал[аи]? бронирование\b",
                ],
                "en": [
                    r"\bunprofessional\b",
                    r"\bunprofessional staff\b",
                    r"\bnot professional\b",
                    r"\bdidn't know what to do\b",
                    r"\bconfused at check[- ]?in\b",
                    r"\bmessed up the booking\b",
                ],
                "tr": [
                    r"\bprofesyonel değillerdi\b",
                    r"\bçok karışıktı resepsiyon\b",
                    r"\bne yapacaklarını bilmiyorlardı\b",
                    r"\brezervasyonu karıştırdılar\b",
                ],
                "ar": [
                    r"\bغير محترفين\b",
                    r"\bمش عارفين يعملوا ايه\b",
                    r"\bلخبطة في الحجز\b",
                    r"\bتعامل غير منظم\b",
                ],
                "zh": [
                    r"不专业",
                    r"服务很不专业",
                    r"搞错了预订",
                    r"很混乱",
                    r"不知道怎么处理",
                ],
            },
        ),
        
        
        "spir_professional": AspectRule(
            aspect_code="spir_professional",
            polarity_hint="positive",
            display="Профессионализм и компетентность СПиР",
            display_short="профессиональная работа персонала",
            long_hint="Гости отмечают, что сотрудники СПиР действуют уверенно и по стандарту: дают чёткие ответы, безошибочно оформляют заселение и создают ощущение управляемого процесса.",
            patterns_by_lang={{
                "ru": [
                    r"\bпрофессионал",
                    r"\bочень профессионально\b",
                    r"\bвсе четко оформили\b",
                    r"\bвсё быстро и по делу\b",
                    r"\bграмотно ответил[аи]?\b",
                ],
                "en": [
                    r"\bprofessional staff\b",
                    r"\bvery professional\b",
                    r"\bhandled everything professionally\b",
                    r"\bwell organized\b",
                    r"\bclear communication\b",
                ],
                "tr": [
                    r"\bçok profesyoneldi\b",
                    r"\bçok düzenliydi\b",
                    r"\bher şeyi düzgün anlattılar\b",
                    r"\bçok iyi organize\b",
                ],
                "ar": [
                    r"\bمحترفين جداً\b",
                    r"\bتعامل مهني\b",
                    r"\bمنظمين جداً\b",
                    r"\bشرح واضح\b",
                ],
                "zh": [
                    r"非常专业",
                    r"服务很专业",
                    r"办理得很顺利",
                    r"沟通很清楚",
                    r"安排很有条理",
                ],
            },
        ),
        
        
        "spir_quick_response": AspectRule(
            aspect_code="spir_quick_response",
            polarity_hint="positive",
            display="Быстрая реакция СПиР на запросы гостей",
            display_short="быстро реагируют",
            long_hint="Гости отмечают, что сотрудники СПиР оперативно отвечают на сообщения и звонки и быстро выходят на связь при обращении.",
            patterns_by_lang={{
                "ru": [
                    r"\bответил[аи]? сразу\b",
                    r"\bмгновенно ответил[аи]?\b",
                    r"\bбыстро отреагир",
                    r"\bсразу вышл[аи]?\b",
                    r"\bсразу перезвон",
                ],
                "en": [
                    r"\bresponded immediately\b",
                    r"\bquick response\b",
                    r"\bvery fast reply\b",
                    r"\banswered right away\b",
                    r"\bgot back to us instantly\b",
                ],
                "tr": [
                    r"\bhemen cevap verdiler\b",
                    r"\bçok hızlı döndüler\b",
                    r"\banında yanıt\b",
                    r"\bhemen ilgilendiler\b",
                ],
                "ar": [
                    r"\bردوا فوراً\b",
                    r"\bاستجابة سريعة\b",
                    r"\bجاوبونا بسرعة\b",
                    r"\bتواصل سريع\b",
                ],
                "zh": [
                    r"回复很快",
                    r"马上回复我们",
                    r"立刻联系到我们",
                    r"马上处理",
                ],
            },
        ),
        
        
        "spir_slow_response": AspectRule(
            aspect_code="spir_slow_response",
            polarity_hint="negative",
            display="Долгая реакция СПиР на запросы гостей",
            display_short="медленная реакция",
            long_hint="Гости фиксируют, что ответ от СПиР приходит с задержкой: приходится ждать обратной связи или напоминать о себе.",
            patterns_by_lang={{
                "ru": [
                    r"\bдолго не отвечал[аи]?\b",
                    r"\bпришлось ждать ответа\b",
                    r"\bждали очень долго\b",
                    r"\bответ только через\b",
                    r"\bперезвонили не сразу\b",
                ],
                "en": [
                    r"\bslow to respond\b",
                    r"\btook a long time to reply\b",
                    r"\bno reply for a long time\b",
                    r"\bhad to wait for an answer\b",
                    r"\bthey answered after a long time\b",
                ],
                "tr": [
                    r"\bgeç cevap verdiler\b",
                    r"\buzun süre dönmediler\b",
                    r"\bcevap çok geç geldi\b",
                    r"\bbeklemek zorunda kaldık\b",
                ],
                "ar": [
                    r"\bتأخروا بالرد\b",
                    r"\bما ردوش إلا بعد وقت طويل\b",
                    r"\bانتظرنا كثير\b",
                    r"\bاستغرق وقت طويل للرد\b",
                ],
                "zh": [
                    r"回复很慢",
                    r"等了很久才回复",
                    r"很久没有答复",
                    r"花了很长时间才联系到我们",
                ],
            },
        ),
        
        
        "spir_unresponsive": AspectRule(
            aspect_code="spir_unresponsive",
            polarity_hint="negative",
            display="Отсутствие обратной связи от СПиР",
            display_short="не отвечают на запросы",
            long_hint="Гости сообщают, что сотрудники СПиР не выходят на связь: не берут трубку, не отвечают в мессенджере, оставляют запрос без реакции.",
            patterns_by_lang={{
                "ru": [
                    r"\bникто не ответил\b",
                    r"\bникто не перезвонил\b",
                    r"\bне брали трубку\b",
                    r"\bигнорир",
                    r"\bпроигнорировали\b",
                    r"\bникакой реакции\b",
                ],
                "en": [
                    r"\bno response\b",
                    r"\bthey never replied\b",
                    r"\bignored us\b",
                    r"\bnobody picked up\b",
                    r"\bno one answered the phone\b",
                    r"\bno one got back to us\b",
                ],
                "tr": [
                    r"\bhiç cevap vermediler\b",
                    r"\bgeri dönmediler\b",
                    r"\btelefonu açmadılar\b",
                    r"\byanıt yoktu\b",
                    r"\byok saydılar\b",
                ],
                "ar": [
                    r"\bما حد رد\b",
                    r"\bطنشونا\b",
                    r"\bما جاوبوش أبداً\b",
                    r"\bما حدش رد على الهاتف\b",
                ],
                "zh": [
                    r"完全没有回复",
                    r"没人回我们",
                    r"打电话没人接",
                    r"被忽视了",
                    r"无人理我们",
                ],
            },
        ),

        "spir_ignored_requests": AspectRule(
            aspect_code="spir_ignored_requests",
            polarity_hint="negative",
            display="Запросы гостей остаются без выполнения",
            display_short="запросы проигнорированы",
            long_hint="Гости фиксируют, что сотрудники СПиР приняли обращение, но дальше ничего не произошло: просьба осталась без действий.",
            patterns_by_lang={{
                "ru": [
                    r"\bобещал[аи]? и не сдел",
                    r"\bобещал[аи]? но так и не\b",
                    r"\bничего не сделали\b",
                    r"\bпросьбу проигнорировали\b",
                    r"\bзаявка повисла\b",
                    r"\bнашу просьбу так и не\b",
                ],
                "en": [
                    r"\bthey ignored our request\b",
                    r"\bour request was ignored\b",
                    r"\bnothing was done\b",
                    r"\bsaid they'd fix it but didn't\b",
                    r"\bpromised but never did\b",
                ],
                "tr": [
                    r"\btalebimizi görmezden geldiler\b",
                    r"\bsöz verdiler ama yapmadılar\b",
                    r"\bhiçbir şey yapılmadı\b",
                    r"\byardım etmiyoruz gibi\b",
                ],
                "ar": [
                    r"\bطلبنا اتطنش\b",
                    r"\bوعدوا وما عملوش حاجة\b",
                    r"\bما حد عمل شي\b",
                    r"\bتجاهلوا طلبنا\b",
                ],
                "zh": [
                    r"说了但没处理",
                    r"没人处理我们的请求",
                    r"完全没做任何事",
                    r"我们的要求被忽视",
                ],
            },
        ),
        
        
        "spir_easy_contact": AspectRule(
            aspect_code="spir_easy_contact",
            polarity_hint="positive",
            display="Лёгкость связи с СПиР",
            display_short="легко связаться",
            long_hint="Гости отмечают, что со СПиР удобно выходить на контакт: сотрудники доступны по телефону и мессенджерам, быстро берут трубку и отвечают.",
            patterns_by_lang={{
                "ru": [
                    r"\bлегко связаться\b",
                    r"\bвсегда на связи\b",
                    r"\bсразу взяли трубку\b",
                    r"\bдозвонились без проблем\b",
                    r"\bбыстро отвечают в ватсап\b",
                    r"\bв чат ответили сразу\b",
                ],
                "en": [
                    r"\beasy to reach\b",
                    r"\balways available\b",
                    r"\bthey picked up right away\b",
                    r"\banswered the phone immediately\b",
                    r"\bquick on whatsapp\b",
                    r"\bresponsive on chat\b",
                ],
                "tr": [
                    r"\bulaşması çok kolaydı\b",
                    r"\bhemen telefona çıktılar\b",
                    r"\bhemen ulaşabildik\b",
                    r"\bwhatsapp'tan hemen cevap\b",
                ],
                "ar": [
                    r"\bسهل نتواصل معاهم\b",
                    r"\bبيردوا مباشرة\b",
                    r"\bدائماً متاحين\b",
                    r"\bرد سريع على الواتساب\b",
                ],
                "zh": [
                    r"很容易联系到",
                    r"马上接电话",
                    r"随时都在",
                    r"微信/whatsapp回复很快",
                ],
            },
        ),
        
        
        "spir_hard_to_contact": AspectRule(
            aspect_code="spir_hard_to_contact",
            polarity_hint="negative",
            display="Сложно выйти на связь с СПиР",
            display_short="трудно связаться",
            long_hint="Гости сообщают, что связаться с СПиР проблемно: не берут трубку, не отвечают в чате, долго не перезванивают.",
            patterns_by_lang={{
                "ru": [
                    r"\bневозможно дозвониться\b",
                    r"\bникто не берет трубку\b",
                    r"\bне удалось связаться\b",
                    r"\bне отвечают в чат\b",
                    r"\bдолго не перезванивали\b",
                ],
                "en": [
                    r"\bhard to reach\b",
                    r"\bcouldn't reach anyone\b",
                    r"\bno one answered the phone\b",
                    r"\bnot picking up\b",
                    r"\bno reply on whatsapp\b",
                ],
                "tr": [
                    r"\bulaşmak çok zordu\b",
                    r"\btelefonu açmadılar\b",
                    r"\bkimseye ulaşamadık\b",
                    r"\bwhatsapp'ta cevap yoktu\b",
                ],
                "ar": [
                    r"\bمش قادرين نوصل لحد\b",
                    r"\bما حدش رد على التليفون\b",
                    r"\bمحدش بيرد\b",
                    r"\bما فيش رد على الواتساب\b",
                ],
                "zh": [
                    r"联系不上",
                    r"没人接电话",
                    r"很难联系到人",
                    r"消息没人回",
                ],
            },
        ),
        
        
        "spir_available": AspectRule(
            aspect_code="spir_available",
            polarity_hint="positive",
            display="Доступность СПиР для гостей в течение проживания",
            display_short="персонал доступен",
            long_hint="Гости отмечают, что сотрудники СПиР доступны, присутствуют на объекте или оперативно включаются удалённо, что создаёт ощущение контролируемого сервиса.",
            patterns_by_lang={{
                "ru": [
                    r"\bвсегда был кто-то на месте\b",
                    r"\bперсонал на месте\b",
                    r"\bк нам сразу подошли\b",
                    r"\bкруглосуточно на связи\b",
                    r"\b24.?7 на связи\b",
                ],
                "en": [
                    r"\bstaff always available\b",
                    r"\balways someone there to help\b",
                    r"\b24.?7 support\b",
                    r"\bavailable at any time\b",
                    r"\bthere was always someone on site\b",
                ],
                "tr": [
                    r"\bher zaman birileri vardı\b",
                    r"\b7.?24 ulaşılabilirlerdi\b",
                    r"\bhemen yanımıza geldiler\b",
                    r"\btesis içinde hep biri vard\b",
                ],
                "ar": [
                    r"\bدائماً في حد موجود\b",
                    r"\bموجودين 24.?7\b",
                    r"\bممكن نكلمهم بأي وقت\b",
                    r"\bحد جالنا فوراً\b",
                ],
                "zh": [
                    r"随时有人可以帮忙",
                    r"全天有人在",
                    r"24小时都能联系到",
                    r"马上有人过来帮我们",
                ],
            },
        ),
        
        
        "spir_not_available": AspectRule(
            aspect_code="spir_not_available",
            polarity_hint="negative",
            display="Недоступность СПиР во время проживания",
            display_short="персонал недоступен",
            long_hint="Гости фиксируют, что в момент обращения сотрудников СПиР не было на месте и их нельзя было оперативно привлечь — ощущение отсутствия ответственного на объекте.",
            patterns_by_lang={{
                "ru": [
                    r"\bникого не было на ресепшене\b",
                    r"\bресепшен пустой\b",
                    r"\bперсонала не было на месте\b",
                    r"\bникого не нашли\b",
                    r"\bникто не вышел\b",
                ],
                "en": [
                    r"\bno staff on site\b",
                    r"\bno one at reception\b",
                    r"\breception was empty\b",
                    r"\bwe couldn't find anyone\b",
                    r"\bnobody came\b",
                ],
                "tr": [
                    r"\bresepsiyonda kimse yoktu\b",
                    r"\bpersonel ortada yoktu\b",
                    r"\bkimseyi bulamadık\b",
                    r"\bkimse gelmedi\b",
                ],
                "ar": [
                    r"\bما كانش فيه حد بالاستقبال\b",
                    r"\bما لقيناش حد\b",
                    r"\bمفيش موظفين موجودين\b",
                    r"\bاستنينا ومحدش جه\b",
                ],
                "zh": [
                    r"前台没有人",
                    r"找不到工作人员",
                    r"现场没人负责",
                    r"等了也没人来",
                ],
            },
        ),

        "spir_problem_fixed_fast": AspectRule(
            aspect_code="spir_problem_fixed_fast",
            polarity_hint="positive",
            display="Оперативное устранение проблемы сотрудниками СПиР",
            display_short="проблему устранили быстро",
            long_hint="Гости отмечают, что после обращения в СПиР проблема была устранена оперативно, без затяжных ожиданий и эскалаций.",
            patterns_by_lang={{
                "ru": [
                    r"\bсразу исправил[аи]?\b",
                    r"\bмгновенно исправил[аи]?\b",
                    r"\bрешили буквально за минут[уы]\b",
                    r"\bпришли и сразу починили\b",
                    r"\bпочинили в тот же момент\b",
                    r"\bвопрос решили сразу\b",
                ],
                "en": [
                    r"\bfixed (it|the issue) right away\b",
                    r"\bsolved immediately\b",
                    r"\bproblem was handled instantly\b",
                    r"\bthey came and fixed it\b",
                    r"\bresolved on the spot\b",
                    r"\bsorted it out in minutes\b",
                ],
                "tr": [
                    r"\bhemen çözdüler\b",
                    r"\banında hallettiler\b",
                    r"\bproblem hemen giderildi\b",
                    r"\bhemen tamir ettiler\b",
                ],
                "ar": [
                    r"\bاتصلنا وجَوا فوراً\b",
                    r"\bحلّوا المشكلة فوراً\b",
                    r"\bاتصلنا وجم صلحوا على طول\b",
                    r"\bتصلح بنفس اللحظة\b",
                ],
                "zh": [
                    r"马上就解决了",
                    r"立刻修好",
                    r"当场就处理了",
                    r"问题很快解决",
                ],
            },
        ),
        
        
        "spir_problem_not_fixed": AspectRule(
            aspect_code="spir_problem_not_fixed",
            polarity_hint="negative",
            display="Проблема не была решена СПиР",
            display_short="проблему не решили",
            long_hint="Гости фиксируют, что после обращения в СПиР проблема осталась нерешённой: обещали разобраться, но ситуация не улучшилась или повторялась.",
            patterns_by_lang={{
                "ru": [
                    r"\bничего не починили\b",
                    r"\bпроблема так и осталась\b",
                    r"\bничего не изменилось\b",
                    r"\bсказали посмотрим и пропали\b",
                    r"\bобещали решить но не решили\b",
                ],
                "en": [
                    r"\bissue was never fixed\b",
                    r"\bproblem not solved\b",
                    r"\bnothing was repaired\b",
                    r"\bthey said they'd fix it but didn't\b",
                    r"\bstill the same problem\b",
                ],
                "tr": [
                    r"\bsorunu çözmediler\b",
                    r"\bhala aynı sorun\b",
                    r"\bhiçbir şey düzeltilmedi\b",
                    r"\bsadece baktılar ama yapmadılar\b",
                ],
                "ar": [
                    r"\bالمشكلة ما اتصلحت\b",
                    r"\bما حلّوها\b",
                    r"\bقالوا حيحلّوها وما صار شي\b",
                    r"\bنفس المشكلة بعدها\b",
                ],
                "zh": [
                    r"问题完全没解决",
                    r"说会修但没修",
                    r"情况没变化",
                    r"还是同样的问题",
                ],
            },
        ),
        
        
        "spir_language_support_good": AspectRule(
            aspect_code="spir_language_support_good",
            polarity_hint="positive",
            display="Языковая поддержка со стороны СПиР",
            display_short="удобно общаться по языку",
            long_hint="Гости отмечают, что с персоналом СПиР комфортно коммуницировать: сотрудники свободно объясняются на английском или другом понятном гостям языке, снижают языковой барьер.",
            patterns_by_lang={{
                "ru": [
                    r"\bхорошо говорят по-английски\b",
                    r"\bобъяснили всё на английском\b",
                    r"\bсмогли общаться на английском\b",
                    r"\bговорили на нашем языке\b",
                    r"\bникаких проблем с английским\b",
                ],
                "en": [
                    r"\bstaff speaks good English\b",
                    r"\bspoke perfect English\b",
                    r"\bcommunication in English was easy\b",
                    r"\bthey spoke our language\b",
                    r"\bno language barrier\b",
                ],
                "tr": [
                    r"\bingilizceleri çok iyi\b",
                    r"\bingilizce rahat iletişim\b",
                    r"\bdil sorunu yoktu\b",
                    r"\bbizim dilimizde konuştular\b",
                ],
                "ar": [
                    r"\bبيتكلموا إنجليزي كويس\b",
                    r"\bالتواصل بالإنجليزي كان سهل\b",
                    r"\bمافيش مشكلة لغة\b",
                    r"\bكلمونا بلغتنا\b",
                ],
                "zh": [
                    r"英语很好",
                    r"沟通英文没有问题",
                    r"没有语言障碍",
                    r"他们会说我们的语言",
                ],
            },
        ),
        
        
        "spir_language_support_bad": AspectRule(
            aspect_code="spir_language_support_bad",
            polarity_hint="negative",
            display="Сложности коммуникации из-за языка",
            display_short="языковой барьер",
            long_hint="Гости сообщают, что общение со СПиР было затруднено: персонал не понимает английский или не может объяснить базовые шаги заселения и пользования номером.",
            patterns_by_lang={{
                "ru": [
                    r"\bсложно объясниться\b",
                    r"\bязыковой барьер\b",
                    r"\bне говорят по-английски\b",
                    r"\bанглийский почти нулевой\b",
                    r"\bне смогли объяснить на английском\b",
                ],
                "en": [
                    r"\bno one speaks English\b",
                    r"\bpoor English\b",
                    r"\bhard to communicate\b",
                    r"\blanguage barrier\b",
                    r"\bcouldn't explain in English\b",
                ],
                "tr": [
                    r"\bingilizce bilmiyorlar\b",
                    r"\bingilizceleri çok kötü\b",
                    r"\banlaşmak zordu\b",
                    r"\bdil sorunu vardı\b",
                ],
                "ar": [
                    r"\bما حدش بيتكلم إنجليزي\b",
                    r"\bصعب نتفاهم\b",
                    r"\bفيه حاجز لغة\b",
                    r"\bمش عارفين يشرحوا بالإنجليزي\b",
                ],
                "zh": [
                    r"英语很差",
                    r"基本不会说英语",
                    r"沟通很困难",
                    r"有语言障碍",
                ],
            },
        ),
        
        
        "bed_uncomfortable": AspectRule(
            aspect_code="bed_uncomfortable",
            polarity_hint="negative",
            display="Некомфортная кровать / качество сна",
            display_short="неудобная кровать",
            long_hint="Гости фиксируют, что кровать неудобная и сон некачественный: жёсткость или мягкость матраса вызывает дискомфорт, сложно выспаться.",
            patterns_by_lang={{
                "ru": [
                    r"\bнеудобн(ая|о) кровать\b",
                    r"\bнеудобно спать\b",
                    r"\bплохой матрас\b",
                    r"\bжесткий матрас\b",
                    r"\bслишком мягкий матрас\b",
                    r"\bне выспал[си]\b",
                ],
                "en": [
                    r"\buncomfortable bed\b",
                    r"\bthe bed was uncomfortable\b",
                    r"\bhard mattress\b",
                    r"\btoo soft mattress\b",
                    r"\bcouldn't sleep well\b",
                    r"\bdidn't sleep well because of the bed\b",
                ],
                "tr": [
                    r"\brahat olmayan yatak\b",
                    r"\byatak çok sertti\b",
                    r"\byatak çok yumuşaktı\b",
                    r"\brahat uyuyamadık\b",
                ],
                "ar": [
                    r"\bالسرير مش مريح\b",
                    r"\bمرتبة قاسية جداً\b",
                    r"\bمرتبة طرية بزيادة\b",
                    r"\bما عرفنا ننام كويس\b",
                ],
                "zh": [
                    r"床不舒服",
                    r"床垫太硬",
                    r"床垫太软",
                    r"没睡好因为床",
                    r"睡得不好",
                ],
            },
        ),
        
        "checkin_fast": AspectRule(
            aspect_code="checkin_fast",
            polarity_hint="positive",
            display="Быстрое оформление заезда",
            display_short="быстрый чек-ин",
            long_hint="Гости отмечают, что процедура заселения прошла быстро: оформление заняло минимальное время, доступ к номеру был выдан без лишних шагов и ожидания.",
            patterns_by_lang={{
                "ru": [
                    r"\bбыстро заселили\b",
                    r"\bзаселили за (минут|пару минут)\b",
                    r"\bчек[- ]?ин был быстрый\b",
                    r"\bоформили очень быстро\b",
                    r"\bвсё заняло пару минут\b",
                ],
                "en": [
                    r"\bquick check[- ]?in\b",
                    r"\bfast check[- ]?in\b",
                    r"\bcheck[- ]?in was very quick\b",
                    r"\bwe checked in quickly\b",
                    r"\bhandled check[- ]?in fast\b",
                ],
                "tr": [
                    r"\bhızlı check[- ]?in\b",
                    r"\bçok hızlı giriş yaptık\b",
                    r"\bkayıt çok hızlıydı\b",
                    r"\bhemen yerleştirdiler\b",
                ],
                "ar": [
                    r"\bتشيك إن سريع\b",
                    r"\bدخلنا بسرعة\b",
                    r"\bإجراءات الدخول كانت سريعة\b",
                    r"\bخلصنا خلال دقايق\b",
                ],
                "zh": [
                    r"办理入住很快",
                    r"很快就办好入住",
                    r"入住手续很快",
                    r"几分钟就办好",
                ],
            },
        ),
        
        
        "no_wait_checkin": AspectRule(
            aspect_code="no_wait_checkin",
            polarity_hint="positive",
            display="Отсутствие ожидания при заселении",
            display_short="без ожидания",
            long_hint="Гости отмечают, что при прибытии не пришлось ждать своей очереди на оформление: не было очереди у стойки, доступ в номер предоставлен сразу.",
            patterns_by_lang={{
                "ru": [
                    r"\bне пришлось ждать\b",
                    r"\bбез ожидания\b",
                    r"\bзаселили сразу\b",
                    r"\bочереди не было\b",
                    r"\bбез очереди заселили\b",
                ],
                "en": [
                    r"\bno wait at check[- ]?in\b",
                    r"\bno line at check[- ]?in\b",
                    r"\bno queue\b",
                    r"\bdidn't have to wait to check in\b",
                    r"\bthey took us right away\b",
                ],
                "tr": [
                    r"\bbeklemeden giriş yaptık\b",
                    r"\bhiç sıra yoktu\b",
                    r"\bcheck[- ]?in beklemeden oldu\b",
                    r"\bhemen aldılar\b",
                ],
                "ar": [
                    r"\bما انتظرنا\b",
                    r"\bما كان في طابور\b",
                    r"\bدخلنا فوراً\b",
                    r"\bاستلمونا على طول\b",
                ],
                "zh": [
                    r"不用等就入住了",
                    r"没有排队",
                    r"直接就能办理入住",
                    r"马上就带我们办入住",
                ],
            },
        ),
        
        
        "checkin_wait_long": AspectRule(
            aspect_code="checkin_wait_long",
            polarity_hint="negative",
            display="Длительное ожидание при заселении",
            display_short="долго ждали чек-ин",
            long_hint="Гости фиксируют, что оформление заезда заняло много времени: была очередь на ресепшене, процесс шёл медленно, пришлось ждать, пока начнут оформление.",
            patterns_by_lang={{
                "ru": [
                    r"\bдолго ждали заселения\b",
                    r"\bпришлось долго ждать\b",
                    r"\bдолгая очередь на ресепшене\b",
                    r"\bочень медленный чек[- ]?ин\b",
                    r"\bоформление заняло слишком долго\b",
                ],
                "en": [
                    r"\blong wait for check[- ]?in\b",
                    r"\blong line at check[- ]?in\b",
                    r"\bhad to wait a long time to check in\b",
                    r"\bit took forever to check in\b",
                    r"\bcheck[- ]?in was really slow\b",
                ],
                "tr": [
                    r"\bcheck[- ]?in çok uzun sürdü\b",
                    r"\buzun süre bekledik\b",
                    r"\bresepsiyonda uzun kuyruk vardı\b",
                    r"\bçok yavaş giriş işlemi\b",
                ],
                "ar": [
                    r"\bاستنينا كتير عالاستقبال\b",
                    r"\bإجراءات الدخول كانت طويلة\b",
                    r"\bطابور طويل\b",
                    r"\bأخذ وقت طويل لندخل\b",
                ],
                "zh": [
                    r"办理入住等了很久",
                    r"排了很长的队",
                    r"入住花了很久",
                    r"check[- ]?in很慢",
                ],
            },
        ),
        
        
        "room_not_ready_delay": AspectRule(
            aspect_code="room_not_ready_delay",
            polarity_hint="negative",
            display="Задержка заселения из-за неготового номера",
            display_short="номер не был готов вовремя",
            long_hint="Гости фиксируют, что номер не был готов к заселению в заявленное время, и им пришлось ждать подготовки или уборки перед тем, как получить доступ.",
            patterns_by_lang={{
                "ru": [
                    r"\bномер не был готов\b",
                    r"\bпришлось ждать номер\b",
                    r"\bждали пока приготовят номер\b",
                    r"\bномер не подготовили вовремя\b",
                    r"\bзаселение задержали\b",
                ],
                "en": [
                    r"\broom wasn't ready\b",
                    r"\bhad to wait for the room\b",
                    r"\broom not ready on time\b",
                    r"\bwe had to wait until they cleaned\b",
                    r"\bdelayed check[- ]?in because room not ready\b",
                ],
                "tr": [
                    r"\boda hazır değildi\b",
                    r"\bodayı beklemek zorunda kaldık\b",
                    r"\boda zamanında hazır değildi\b",
                    r"\btemizlik bitmesini bekledik\b",
                ],
                "ar": [
                    r"\bالغرفة ما كانت جاهزة\b",
                    r"\bاضطرينا نستنى الغرفة\b",
                    r"\bالدخول اتأخر عشان الغرفة مو جاهزة\b",
                    r"\bلسا بينضفوا\b",
                ],
                "zh": [
                    r"房间还没准备好",
                    r"我们还得等房间",
                    r"到的时候房间没打扫好",
                    r"因为房间没准备好而延迟入住",
                ],
            },
        ),
        
        
        "room_ready_on_arrival": AspectRule(
            aspect_code="room_ready_on_arrival",
            polarity_hint="positive",
            display="Номер готов к заселению по прибытии",
            display_short="номер готов сразу",
            long_hint="Гости отмечают, что номер был полностью готов к их приезду: ключи/код сразу выданы, можно было сразу зайти и разместиться без ожидания.",
            patterns_by_lang={{
                "ru": [
                    r"\bномер был готов\b",
                    r"\bномер сразу готов\b",
                    r"\bсразу заселили в готовый номер\b",
                    r"\bкомната уже ждала нас\b",
                    r"\bвсё было готово к заезду\b",
                ],
                "en": [
                    r"\broom was ready\b",
                    r"\broom ready on arrival\b",
                    r"\bour room was ready when we arrived\b",
                    r"\bthe room was prepared for us\b",
                    r"\bwe could go straight to the room\b",
                ],
                "tr": [
                    r"\bodamız hazırdı\b",
                    r"\bgelince oda hazırdı\b",
                    r"\bhemen odaya geçtik\b",
                    r"\boda bizim için hazırlanmıştı\b",
                ],
                "ar": [
                    r"\bالغرفة كانت جاهزة فوراً\b",
                    r"\bأول ما وصلنا دخلنا الغرفة\b",
                    r"\bالغرفة محضّرة لنا\b",
                    r"\bما احتجنا ننتظر الغرفة\b",
                ],
                "zh": [
                    r"房间一到就准备好了",
                    r"到的时候房间已经准备好",
                    r"我们马上就能进房间",
                    r"房间提前安排好",
                ],
            },
        ),

        "clean_on_arrival": AspectRule(
            aspect_code="clean_on_arrival",
            polarity_hint="positive",
            display="Чистота номера на момент заезда",
            display_short="чисто при заезде",
            long_hint="Гости отмечают, что номер был чистым к моменту заселения: поверхности, санузел и текстиль выглядели аккуратно и не требовали дополнительной уборки перед размещением.",
            patterns_by_lang={{
                "ru": [
                    r"\bчисто при заселении\b",
                    r"\bномер чистый\b",
                    r"\bвсё было чисто\b",
                    r"\bбыло убрано\b",
                    r"\bидеально чисто\b",
                    r"\bочень чистая комната\b",
                ],
                "en": [
                    r"\bvery clean on arrival\b",
                    r"\broom was clean\b",
                    r"\bspotless when we arrived\b",
                    r"\bthe room was perfectly clean\b",
                    r"\beverything was clean when we checked in\b",
                ],
                "tr": [
                    r"\bodamız çok temizdi\b",
                    r"\bgeldiğimizde tertemizdi\b",
                    r"\boda pırıl pırıldı\b",
                    r"\bgirişte her şey temizdi\b",
                ],
                "ar": [
                    r"\bالغرفة كانت نظيفة جداً وقت الدخول\b",
                    r"\bكل شيء كان نضيف لما وصلنا\b",
                    r"\bنظيف عند الوصول\b",
                    r"\bنضيف من البداية\b",
                ],
                "zh": [
                    r"房间很干净入住时",
                    r"我们到的时候房间很干净",
                    r"非常干净一进来",
                    r"入住时一切都很干净",
                ],
            },
        ),
        
        
        "room_not_ready": AspectRule(
            aspect_code="room_not_ready",
            polarity_hint="negative",
            display="Номер не был готов к заселению",
            display_short="номер не подготовлен",
            long_hint="Гости фиксируют, что по прибытии номер не был готов к приёму гостей: уборка не завершена, постель не заправлена, в номере не наведён порядок.",
            patterns_by_lang={{
                "ru": [
                    r"\bномер не был готов\b",
                    r"\bкомната не готова\b",
                    r"\bне подготовили номер\b",
                    r"\bуборка не закончена\b",
                    r"\bеще не убрали\b",
                    r"\bпостель не заправлена\b",
                ],
                "en": [
                    r"\broom was not ready\b",
                    r"\bthe room wasn't prepared\b",
                    r"\bwasn't prepared for us\b",
                    r"\bstill dirty when we arrived\b",
                    r"\bthey hadn't finished cleaning\b",
                ],
                "tr": [
                    r"\boda hazır değildi\b",
                    r"\bhazırlanmamıştı\b",
                    r"\btemizlik bitmemişti\b",
                    r"\bçarşaflar henüz yapılmamıştı\b",
                ],
                "ar": [
                    r"\bالغرفة ما كانت جاهزة\b",
                    r"\bلسه ما خلصوا التنظيف\b",
                    r"\bالغرفة مش محضّرة\b",
                    r"\bالسرير حتى مو مرتب\b",
                ],
                "zh": [
                    r"房间还没准备好",
                    r"房间还没打扫完",
                    r"还没收拾好就让我们进来",
                    r"房间一开始没整理好",
                ],
            },
        ),
        
        
        "dirty_on_arrival": AspectRule(
            aspect_code="dirty_on_arrival",
            polarity_hint="negative",
            display="Грязь в номере при заезде",
            display_short="грязно при заезде",
            long_hint="Гости сообщают, что при заселении номер выглядел недостаточно чистым: следы пыли, мусор, волосы, налёт в ванной и другие признаки некачественной подготовки к заезду.",
            patterns_by_lang={{
                "ru": [
                    r"\bгрязно при заселении\b",
                    r"\bгрязный номер\b",
                    r"\bгрязная комната\b",
                    r"\bволосы повсюду\b",
                    r"\bпыль на поверхностях\b",
                    r"\bгрязь на полу\b",
                ],
                "en": [
                    r"\bdirty on arrival\b",
                    r"\broom was dirty\b",
                    r"\bthe room was not clean\b",
                    r"\bhair everywhere\b",
                    r"\bdust on the surfaces\b",
                    r"\bdirty floor when we arrived\b",
                ],
                "tr": [
                    r"\bgeldiğimizde oda kirliydi\b",
                    r"\boda temiz değildi\b",
                    r"\bher yerde saç vardı\b",
                    r"\btozluydu\b",
                    r"\byerler kirliydi\b",
                ],
                "ar": [
                    r"\bالغرفة كانت وسخة لما دخلنا\b",
                    r"\bمش نضيفة وقت الوصول\b",
                    r"\bشعر في كل مكان\b",
                    r"\bتراب على الطاولات\b",
                ],
                "zh": [
                    r"入住时房间很脏",
                    r"房间不干净",
                    r"到的时候地上很脏",
                    r"桌面都是灰",
                    r"到处都是头发",
                ],
            },
        ),
        
        
        "leftover_trash_previous_guest": AspectRule(
            aspect_code="leftover_trash_previous_guest",
            polarity_hint="negative",
            display="Следы предыдущего гостя в номере",
            display_short="чужой мусор в номере",
            long_hint="Гости фиксируют, что в номере остались явные следы предыдущего проживания: не вынесенный мусор, использованные полотенца, еда, бутылки, личные вещи предыдущих гостей.",
            patterns_by_lang={{
                "ru": [
                    r"\bмусор от прошл(ого|ых) гостей\b",
                    r"\bчужой мусор остался\b",
                    r"\bпакет с мусором остался\b",
                    r"\bбыли чужие вещи\b",
                    r"\bгрязные полотенца предыдущих гостей\b",
                    r"\bне вынесли мусор\b",
                ],
                "en": [
                    r"\bleftover trash from previous guest\b",
                    r"\btrash from previous guests\b",
                    r"\bprevious guest's garbage\b",
                    r"\bbin wasn't emptied\b",
                    r"\bused towels left\b",
                    r"\bother people's stuff left in the room\b",
                ],
                "tr": [
                    r"\bönceki misafirden çöp kalmıştı\b",
                    r"\bçöp boşaltılmamıştı\b",
                    r"\bönceki misafirin eşyaları duruyordu\b",
                    r"\bkullanılmış havlular kalmıştı\b",
                ],
                "ar": [
                    r"\bفيه زبالة من النزيل اللي قبلي\b",
                    r"\bما شالوش الزبالة القديمة\b",
                    r"\bفيه أغراض من الضيف السابق\b",
                    r"\bمناشف مستعملة تركوها\b",
                ],
                "zh": [
                    r"还有上个客人的垃圾",
                    r"垃圾桶没倒",
                    r"房间里有前一位客人的东西",
                    r"有别人用过的毛巾没拿走",
                ],
            },
        ),
        
        
        "access_smooth": AspectRule(
            aspect_code="access_smooth",
            polarity_hint="positive",
            display="Беспроблемный доступ в апартаменты",
            display_short="доступ без проблем",
            long_hint="Гости отмечают, что физический доступ в объект и в номер проходил без затруднений: легко нашли вход, коды/ключи сработали, не возникло барьеров с багажом.",
            patterns_by_lang={{
                "ru": [
                    r"\bлегко попасть в здание\b",
                    r"\bпройти было просто\b",
                    r"\bдоступ без проблем\b",
                    r"\bвсё сработало с первого раза\b",
                    r"\bоткрыли без трудностей\b",
                ],
                "en": [
                    r"\bsmooth access\b",
                    r"\beasy access\b",
                    r"\bgetting in was easy\b",
                    r"\bno problems entering the building\b",
                    r"\bno issues with access codes\b",
                ],
                "tr": [
                    r"\bgirişi çok kolaydı\b",
                    r"\bhiç sorun yaşamadan içeri girdik\b",
                    r"\bkolay erişim\b",
                    r"\bkapı kodu hemen çalıştı\b",
                ],
                "ar": [
                    r"\bالدخول كان سهل\b",
                    r"\bما في أي مشكلة نفوت\b",
                    r"\bدخول سلس جداً\b",
                    r"\bالكود اشتغل من أول مرة\b",
                ],
                "zh": [
                    r"进楼很方便",
                    r"进入很顺利",
                    r"没有进门问题",
                    r"门码一下就开了",
                ],
            },
        ),

        "door_code_worked": AspectRule(
            aspect_code="door_code_worked",
            polarity_hint="positive",
            display="Корректная работа ключей доступа",
            display_short="ключ доступа сработал",
            long_hint="Гости отмечают, что клюс от входной двери в номер работал без сбоев.",
            patterns_by_lang={{
                "ru": [
                    r"\bкод подош[её]л\b",
                    r"\bкод сработал\b",
                    r"\bдверь открылась с первого раза\b",
                    r"\bполучили код и сразу вошли\b",
                    r"\bдоступ по коду без проблем\b",
                ],
                "en": [
                    r"\bdoor code worked\b",
                    r"\baccess code worked\b",
                    r"\bcode worked first try\b",
                    r"\bwe got the code and it worked\b",
                    r"\bno problem with the code\b",
                ],
                "tr": [
                    r"\bgiriş kodu çalıştı\b",
                    r"\bkapı kodu sorunsuzdu\b",
                    r"\bşifre ilk denemede açtı\b",
                    r"\bkodu aldık hemen girdik\b",
                ],
                "ar": [
                    r"\bالكود اشتغل بدون مشكلة\b",
                    r"\bدخلنا بالكود علطول\b",
                    r"\bكود الباب كان شغال\b",
                    r"\bالكود فتح من أول مرة\b",
                ],
                "zh": [
                    r"门码可以用",
                    r"门锁密码一次就开了",
                    r"收到密码就能进去了",
                    r"进门密码没有问题",
                ],
            },
        ),
        
        
        "tech_access_issue": AspectRule(
            aspect_code="tech_access_issue",
            polarity_hint="negative",
            display="Технические проблемы с доступом",
            display_short="технический сбой доступа",
            long_hint="Гости фиксируют, что не удалось сразу попасть внутрь из-за технического сбоя: ключ не подошёл, замок не открылся, электронный замок завис или потребовалась повторное программирование ключа.",
            patterns_by_lang={{
                "ru": [
                    r"\bключ не подош[её]л\b",
                    r"\bключ не сработал\b",
                    r"\bдверь не открывалась\b",
                    r"\bэлектронный замок не работал\b",
                    r"\bпришлось ждать новый ключ\b",
                    r"\bсмарт[- ]?замок завис\b",
                ],
                "en": [
                    r"\bdoor key didn't work\b",
                    r"\baccess key didn't work\b",
                    r"\bwe couldn't get in\b",
                    r"\bsmart lock didn't work\b",
                    r"\bthe lock wouldn't open\b",
                    r"\bhad to wait for a new key\b",
                ],
                "tr": [
                    r"\bşifre çalışmadı\b",
                    r"\bkapı açılmadı\b",
                    r"\bkilit açılmadı\b",
                    r"\bakıllı kilit bozuldu\b",
                    r"\byeni kod beklemek zorunda kaldık\b",
                ],
                "ar": [
                    r"\bالكود ما اشتغل\b",
                    r"\bالقفل ما فتح\b",
                    r"\bالقفل الإلكتروني معلق\b",
                    r"\bاضطرينا نستنى كود جديد\b",
                    r"\bما قدرنا ندخل\b",
                ],
                "zh": [
                    r"门码打不开门",
                    r"我们进不去",
                    r"智能门锁不好用",
                    r"锁打不开",
                    r"需要重新发密码才进得去",
                ],
            },
        ),
        
        
        "entrance_hard_to_find": AspectRule(
            aspect_code="entrance_hard_to_find",
            polarity_hint="negative",
            display="Сложно найти вход в здание/апартаменты",
            display_short="сложно найти вход",
            long_hint="Гости сообщают, что поиск правильного входа занял время: вывески нет или не видно, навигация и описание локации недостаточно понятные.",
            patterns_by_lang={{
                "ru": [
                    r"\bтрудно найти вход\b",
                    r"\bне могли найти вход\b",
                    r"\bнепонятно куда заходить\b",
                    r"\bникакой таблички\b",
                    r"\bплохо обозначен вход\b",
                    r"\bдолго искали здание\b",
                ],
                "en": [
                    r"\bhard to find the entrance\b",
                    r"\bcouldn't find the entrance\b",
                    r"\bno sign\b",
                    r"\bpoor signage\b",
                    r"\bwe had trouble finding the building\b",
                    r"\bconfusing entrance\b",
                ],
                "tr": [
                    r"\bgirişi bulmak zordu\b",
                    r"\btabela yoktu\b",
                    r"\bhangi kapıdan gireceğimizi anlayamadık\b",
                    r"\bgirişi uzun süre aradık\b",
                ],
                "ar": [
                    r"\bصعب نلاقي المدخل\b",
                    r"\bما في لافتة واضحة\b",
                    r"\bقعدنا ندور على الباب\b",
                    r"\bمش واضح من وين ندخل\b",
                ],
                "zh": [
                    r"入口很难找",
                    r"不知道从哪儿进",
                    r"没有明显的门牌/指示",
                    r"找入口花了很久",
                ],
            },
        ),
        
        
        "no_elevator_baggage_issue": AspectRule(
            aspect_code="no_elevator_baggage_issue",
            polarity_hint="negative",
            display="Сложности с багажом из-за отсутствия лифта / ограничения по лифту",
            display_short="лифт и багаж (проблема)",
            long_hint="Гости фиксируют дискомфорт при доступе в номер из-за отсутствия лифта, неработающего лифта или неудобной логистики с тяжёлым багажом (ступеньки, узкие пролёты).",
            patterns_by_lang={{
                "ru": [
                    r"\bнет лифта\b",
                    r"\bлифта нет\b",
                    r"\bлифт не работал\b",
                    r"\bпришлось тащить чемоданы\b",
                    r"\bтаскать чемоданы по лестнице\b",
                    r"\bтяжело с багажом\b",
                ],
                "en": [
                    r"\bno elevator\b",
                    r"\bthere is no lift\b",
                    r"\blift was not working\b",
                    r"\bhad to carry our luggage upstairs\b",
                    r"\bhad to carry suitcases up the stairs\b",
                    r"\bhard with luggage\b",
                ],
                "tr": [
                    r"\basansör yoktu\b",
                    r"\basansör çalışmıyordu\b",
                    r"\bvalizleri merdivenden taşımak zorunda kaldık\b",
                    r"\bçok zor oldu bavullarla\b",
                ],
                "ar": [
                    r"\bما في مصعد\b",
                    r"\bالأسانسير كان معطل\b",
                    r"\bاضطرينا نطلع الشنط عالسلم\b",
                    r"\bصعب مع الشنط\b",
                ],
                "zh": [
                    r"没有电梯",
                    r"电梯坏了",
                    r"行李要自己扛楼梯",
                    r"拿行李上楼很累",
                ],
            },
        ),
        
        
        "payment_clear": AspectRule(
            aspect_code="payment_clear",
            polarity_hint="positive",
            display="Прозрачность условий оплаты при заезде",
            display_short="оплата прозрачна",
            long_hint="Гости отмечают, что условия оплаты были понятными: стоимость проживания, депозит, дополнительные сборы и порядок списаний были объяснены заранее и без двусмысленности.",
            patterns_by_lang={{
                "ru": [
                    r"\bвсё прозрачно по оплате\b",
                    r"\bчетко объяснили оплату\b",
                    r"\bникаких скрытых платежей\b",
                    r"\bсумма сразу была понятна\b",
                    r"\bсразу сказали сколько и за что\b",
                ],
                "en": [
                    r"\bclear payment terms\b",
                    r"\bclear about the payment\b",
                    r"\bno hidden charges\b",
                    r"\bwe knew exactly what we were paying\b",
                    r"\bthey explained the deposit clearly\b",
                ],
                "tr": [
                    r"\bödeme çok netti\b",
                    r"\bgizli ücret yoktu\b",
                    r"\bne ödeyeceğimizi baştan söylediler\b",
                    r"\bdepozitoyu düzgün açıkladılar\b",
                ],
                "ar": [
                    r"\bالدفع كان واضح\b",
                    r"\bما في رسوم مخفية\b",
                    r"\bشرحوا الفلوس من الأول\b",
                    r"\bوضحوا مبلغ التأمين\b",
                ],
                "zh": [
                    r"付款很清楚",
                    r"没有隐藏收费",
                    r"一开始就说清楚要付多少",
                    r"押金说明得很清楚",
                ],
            },
        ),
    
        "deposit_clear": AspectRule(
            aspect_code="deposit_clear",
            polarity_hint="positive",
            display="Понятные условия депозита",
            display_short="депозит объяснён",
            long_hint="Гости отмечают, что политика депозита была объяснена заранее и прозрачно: размер, условия возврата и сроки возврата были понятны.",
            patterns_by_lang={{
                "ru": [
                    r"\bдепозит сразу объяснил[аи]?\b",
                    r"\bдепозит понятн(ый|о)\b",
                    r"\bс deposit всё ясно\b",
                    r"\bсказали когда вернут депозит\b",
                    r"\bзаранее предупредили про залог\b",
                ],
                "en": [
                    r"\bdeposit was clear\b",
                    r"\bclear about the deposit\b",
                    r"\bexplained the deposit policy\b",
                    r"\bwe knew about the deposit in advance\b",
                    r"\bthey told us when we'd get the deposit back\b",
                ],
                "tr": [
                    r"\bdepozito net açıklandı\b",
                    r"\bdepozito konusunda çok açıktılar\b",
                    r"\bne kadar depozito olduğu belliydi\b",
                    r"\bdepozito ne zaman iade edilecek anlattılar\b",
                ],
                "ar": [
                    r"\bشرحوا عربون التأمين بوضوح\b",
                    r"\bالمبلغ التأميني كان واضح\b",
                    r"\bقالوا إمتى يرجعوا العربون\b",
                    r"\bبلغونا عن الديبوزيت من قبل\b",
                ],
                "zh": [
                    r"押金说明得很清楚",
                    r"押金政策说清楚了",
                    r"提前告知押金",
                    r"什么时候退押金都讲了",
                ],
            },
        ),
        
        
        "docs_provided": AspectRule(
            aspect_code="docs_provided",
            polarity_hint="positive",
            display="Корректный пакет документов",
            display_short="документы выдали",
            long_hint="Гости отмечают, что им выдали все необходимые документы и чек, и оформление выглядело формально корректным.",
            patterns_by_lang={{
                "ru": [
                    r"\bдали все документы\b",
                    r"\bвесь пакет документов\b",
                    r"\bдоговор предоставили\b",
                    r"\bчек сразу выдали\b",
                    r"\bполучили чек об оплате\b",
                ],
                "en": [
                    r"\bwe received all documents\b",
                    r"\bthey provided the invoice\b",
                    r"\bgot a receipt right away\b",
                    r"\bproper paperwork was provided\b",
                    r"\bwe got the contract\b",
                ],
                "tr": [
                    r"\bfaturayı hemen verdiler\b",
                    r"\bbelgeleri hemen aldık\b",
                    r"\bmakbuz verdiler\b",
                    r"\bsözleşme verildi\b",
                ],
                "ar": [
                    r"\bأعطونا كل الأوراق\b",
                    r"\bأخذنا الفاتورة فوراً\b",
                    r"\bإيصال الدفع عطوه على طول\b",
                    r"\bورق الحجز كان جاهز\b",
                ],
                "zh": [
                    r"入住时给了所有文件",
                    r"当场给了发票/收据",
                    r"给了合同",
                    r"付款收据马上给了",
                ],
            },
        ),
        
        
        "no_hidden_fees": AspectRule(
            aspect_code="no_hidden_fees",
            polarity_hint="positive",
            display="Отсутствие скрытых платежей",
            display_short="без скрытых сборов",
            long_hint="Гости фиксируют, что финальная стоимость проживания совпала с изначально озвученной ценой, без неожиданных доплат на месте.",
            patterns_by_lang={{
                "ru": [
                    r"\bбез скрытых платежей\b",
                    r"\bникаких доплат\b",
                    r"\bникаких сюрпризов по деньгам\b",
                    r"\bзаплатили ровно столько сколько обещали\b",
                    r"\bитог совпал с бронированием\b",
                ],
                "en": [
                    r"\bno hidden fees\b",
                    r"\bno extra charges\b",
                    r"\bwe paid exactly what was stated\b",
                    r"\bprice was exactly as booked\b",
                    r"\bno surprises at check[- ]?in\b",
                ],
                "tr": [
                    r"\bgizli ücret yoktu\b",
                    r"\bek ücret çıkmadı\b",
                    r"\bfiyata ekstra bir şey eklenmedi\b",
                    r"\brezervasyondaki fiyatla aynıydı\b",
                ],
                "ar": [
                    r"\bما في رسوم مخفية\b",
                    r"\bما طلبوا أي مصاريف زيادة\b",
                    r"\bنفس السعر اللي بالحجز\b",
                    r"\bما في مفاجآت بالسعر\b",
                ],
                "zh": [
                    r"没有隐藏收费",
                    r"没有额外费用",
                    r"价格和预订时一样",
                    r"没有临时加钱",
                ],
            },
        ),
        
        
        "payment_confusing": AspectRule(
            aspect_code="payment_confusing",
            polarity_hint="negative",
            display="Непрозрачные условия оплаты",
            display_short="оплата неясна",
            long_hint="Гости сообщают, что условия оплаты были непонятны: неочевидные начисления, неполное объяснение того, что включено в стоимость, неясные операции по карте.",
            patterns_by_lang={{
                "ru": [
                    r"\bнепонятная оплата\b",
                    r"\bзапутанные платежи\b",
                    r"\bничего не объяснили по оплате\b",
                    r"\bв чеке какие-то лишние суммы\b",
                    r"\bпочему списали больше\b",
                ],
                "en": [
                    r"\bconfusing payment\b",
                    r"\bpayment was not clear\b",
                    r"\bdidn't explain the charges\b",
                    r"\bwasn't clear why they charged extra\b",
                    r"\bbilling was confusing\b",
                ],
                "tr": [
                    r"\bödeme net değildi\b",
                    r"\bücretlendirme karışıktı\b",
                    r"\bniye fazla çekildi anlamadık\b",
                    r"\bfaturada anlaşılmayan ek ücretler vardı\b",
                ],
                "ar": [
                    r"\bالدفع مش واضح\b",
                    r"\bمش فاهمين الرسوم ليه كده\b",
                    r"\bحسبوا رقم أعلى ومش عارفين ليه\b",
                    r"\bالفاتورة مش مفهومة\b",
                ],
                "zh": [
                    r"收费不清楚",
                    r"账单看不懂",
                    r"不明白为什么多扣钱",
                    r"付款方式很混乱",
                ],
            },
        ),
        
        
        "unexpected_charge": AspectRule(
            aspect_code="unexpected_charge",
            polarity_hint="negative",
            display="Неожиданное списание / доплата",
            display_short="неожиданный платёж",
            long_hint="Гости фиксируют появление незапланированного платежа: была списана дополнительная сумма без предупреждения или озвучена обязательная доплата, о которой не сообщали заранее.",
            patterns_by_lang={{
                "ru": [
                    r"\bвзяли доплату\b",
                    r"\bсписали лишние деньги\b",
                    r"\bс карты списали больше\b",
                    r"\bдополнительный платеж без предупреждения\b",
                    r"\bпопросили сверху ещё\b",
                ],
                "en": [
                    r"\bextra charge\b",
                    r"\bunexpected charge\b",
                    r"\bthey overcharged us\b",
                    r"\bcharged more than expected\b",
                    r"\bcharged my card without telling me\b",
                ],
                "tr": [
                    r"\bbeklenmeyen ekstra ücret\b",
                    r"\bfazla para çektiler\b",
                    r"\bhaber vermeden karttan para çekildi\b",
                    r"\bekstra ödeme istediler\b",
                ],
                "ar": [
                    r"\bخصموا زيادة من غير ما يقولوا\b",
                    r"\bفيه مبلغ إضافي ماقالوش عليه\b",
                    r"\bسحبوا من الكرت زيادة\b",
                    r"\bدفعونا زيادة فجأة\b",
                ],
                "zh": [
                    r"被多收钱",
                    r"有额外收费没提前说",
                    r"刷卡金额比预期高",
                    r"突然又要我们付一笔钱",
                ],
            },
        ),

        "hidden_fees": AspectRule(
            aspect_code="hidden_fees",
            polarity_hint="negative",
            display="Скрытые платежи и дополнительные сборы",
            display_short="скрытые платежи",
            long_hint="Гости фиксируют, что на месте появились обязательные платежи, о которых не было прозрачной информации заранее: сервисные сборы, дополнительные комиссии, плата за уборку и т.д.",
            patterns_by_lang={{
                "ru": [
                    r"\bскрытые платежи\b",
                    r"\bскрыт(ые|ые) комиссии\b",
                    r"\bдополнительный сбор о котором не предупредили\b",
                    r"\bв счете появилась какая-то доп. услуга\b",
                    r"\bплата за уборку внезапно\b",
                ],
                "en": [
                    r"\bhidden fees\b",
                    r"\bhidden charges\b",
                    r"\bextra cleaning fee we didn't know about\b",
                    r"\bresort fee we weren't told about\b",
                    r"\bservice fee not mentioned before\b",
                ],
                "tr": [
                    r"\bgizli ücretler\b",
                    r"\bhaber verilmeden ek ücret\b",
                    r"\btemizlik ücreti sonradan çıktı\b",
                    r"\bservis ücreti ayrıca istediler\b",
                ],
                "ar": [
                    r"\bرسوم مخفية\b",
                    r"\bدفعونا رسوم ما بلغونا عنها\b",
                    r"\bطلع فيه رسوم تنظيف إضافية\b",
                    r"\bفيه عمولة زيادة ما قالوها\b",
                ],
                "zh": [
                    r"有隐藏收费",
                    r"额外清洁费没提前说明",
                    r"现场才告诉我们要收服务费",
                    r"多出来一笔手续费没说过",
                ],
            },
        ),
        
        
        "deposit_problematic": AspectRule(
            aspect_code="deposit_problematic",
            polarity_hint="negative",
            display="Проблемы с департаментом депозита / залогом",
            display_short="проблемы с депозитом",
            long_hint="Гости сообщают о неудобствах с депозитом: завышенная сумма, требование наличных вместо карты, блокировка средств без объяснения условий или сложности с возвратом депозита.",
            patterns_by_lang={{
                "ru": [
                    r"\bдепозит слишком большой\b",
                    r"\bдепозит (не вернули|не возвращают)\b",
                    r"\bс залогом проблема\b",
                    r"\bзалог только наличными\b",
                    r"\bбез объяснения заблокировали деньги\b",
                ],
                "en": [
                    r"\bdeposit issue\b",
                    r"\bproblem with the deposit\b",
                    r"\bdeposit not returned\b",
                    r"\bthey blocked a big deposit\b",
                    r"\bcash deposit required\b",
                    r"\bunclear deposit hold\b",
                ],
                "tr": [
                    r"\bdepozito sorunu\b",
                    r"\bdepozito geri verilmedi\b",
                    r"\bçok yüksek depozito aldılar\b",
                    r"\bnakit depozito istediler\b",
                    r"\bdepozito neden bloke edildi anlamadık\b",
                ],
                "ar": [
                    r"\bمشكلة مع العربون\b",
                    r"\bما رجعوش العربون\b",
                    r"\bطلبوا عربون عالي جداً\b",
                    r"\bلازم كاش للتأمين\b",
                    r"\bحجزوا مبلغ بدون ما يشرحوا\b",
                ],
                "zh": [
                    r"押金有问题",
                    r"押金没退",
                    r"押金太高",
                    r"只收现金押金",
                    r"冻结金额却没解释",
                ],
            },
        ),
        
        
        "billing_mistake": AspectRule(
            aspect_code="billing_mistake",
            polarity_hint="negative",
            display="Ошибки в счёте / некорректное выставление оплаты",
            display_short="ошибка в счёте",
            long_hint="Гости фиксируют ошибки при выставлении счёта: некорректные строки в чеке, неверно пробитые услуги, списание за позиции, которыми гости не пользовались.",
            patterns_by_lang={{
                "ru": [
                    r"\bошибка в счете\b",
                    r"\bневерно рассчитали\b",
                    r"\bпробили лишнее\b",
                    r"\bпосчитали то, чем не пользовались\b",
                    r"\bдобавили услугу которой не было\b",
                ],
                "en": [
                    r"\bbilling error\b",
                    r"\bcharged us for something we didn't use\b",
                    r"\bincorrect bill\b",
                    r"\bwrong amount on the invoice\b",
                    r"\badded items we never had\b",
                ],
                "tr": [
                    r"\bfaturada hata vardı\b",
                    r"\bkullanmadığımız şeyi ücrete eklediler\b",
                    r"\bhesap yanlış çıktı\b",
                    r"\byanlış ücret yansıttılar\b",
                ],
                "ar": [
                    r"\bفيه غلط بالفاتورة\b",
                    r"\bدفعونا على شي ما استخدمناه\b",
                    r"\bالمبلغ مش صح\b",
                    r"\bضافوا بند ما أخذناه\b",
                ],
                "zh": [
                    r"账单有错误",
                    r"收了我们没用过的东西",
                    r"发票金额不正确",
                    r"账单多算了项目",
                ],
            },
        ),
        
        
        "overcharge": AspectRule(
            aspect_code="overcharge",
            polarity_hint="negative",
            display="Завышенное списание / переплата",
            display_short="переплата / завышенное списание",
            long_hint="Гости заявляют, что с них списали больше, чем указано в подтверждении бронирования: итоговая сумма оказалась выше заявленной ставки за проживание.",
            patterns_by_lang={{
                "ru": [
                    r"\bпереплата\b",
                    r"\bсняли больше чем должны\b",
                    r"\bсписали больше чем в бронировании\b",
                    r"\bзавышенное списание\b",
                    r"\bвышло дороже чем обещали\b",
                ],
                "en": [
                    r"\bovercharged\b",
                    r"\bthey overcharged us\b",
                    r"\bcharged more than the booking price\b",
                    r"\bwe paid more than we should have\b",
                    r"\bfinal price was higher than quoted\b",
                ],
                "tr": [
                    r"\bfazla ücret aldılar\b",
                    r"\bliste fiyatından daha fazla kestiler\b",
                    r"\bnormalden fazla ödedik\b",
                    r"\bgerçekten fazla çektiler karttan\b",
                ],
                "ar": [
                    r"\bدفعونا زيادة\b",
                    r"\bسحبوا مبلغ أعلى من الاتفاق\b",
                    r"\bفاتورة أعلى من السعر المتفق\b",
                    r"\bحسبوا علينا أكثر من السعر بالحجز\b",
                ],
                "zh": [
                    r"被多收钱了",
                    r"扣款比预订价高",
                    r"付的钱比说好的多",
                    r"最后金额高于确认价",
                ],
            },
        ),
        
        
        "instructions_clear": AspectRule(
            aspect_code="instructions_clear",
            polarity_hint="positive",
            display="Понятные инструкции по заселению и использованию номера",
            display_short="чёткие инструкции",
            long_hint="Гости отмечают, что получили ясные инструкции по доступу, Wi-Fi, бытовым моментам и правилам проживания. Информация была структурированной и понятной.",
            patterns_by_lang={{
                "ru": [
                    r"\bочень понятные инструкции\b",
                    r"\bподробная инструкция по заселению\b",
                    r"\bвсё расписано шаг за шагом\b",
                    r"\bвсё объяснили заранее\b",
                    r"\bинструкция по вайфаю сразу\b",
                    r"\bчек[- ]?лист заселения прислали\b",
                ],
                "en": [
                    r"\bclear instructions\b",
                    r"\bcheck[- ]?in instructions were clear\b",
                    r"\bstep by step instructions\b",
                    r"\bthey explained everything in advance\b",
                    r"\bwifi info was provided\b",
                ],
                "tr": [
                    r"\bçok net talimat verdiler\b",
                    r"\bgiriş talimatları çok açıktı\b",
                    r"\badım adım anlattılar\b",
                    r"\bwifi bilgilerini hemen gönderdiler\b",
                ],
                "ar": [
                    r"\bالتعليمات كانت واضحة جداً\b",
                    r"\bشرحوا خطوة بخطوة إزاي ندخل\b",
                    r"\bبعتولنا معلومات الواي فاي\b",
                    r"\bكل شي متوضح قبل ما نوصل\b",
                ],
                "zh": [
                    r"入住指引很清楚",
                    r"给了详细步骤说明",
                    r"提前发了入住步骤",
                    r"Wi-Fi信息一开始就给了",
                ],
            },
        ),

        "self_checkin_easy": AspectRule(
            aspect_code="self_checkin_easy",
            polarity_hint="positive",
            display="Удобство самостоятельного заселения",
            display_short="самостоятельный чек-ин удобен",
            long_hint="Гости отмечают, что формат самостоятельного заселения с кодами доступа и инструкциями сработал без затруднений и не требовал личного участия персонала.",
            patterns_by_lang={{
                "ru": [
                    r"\bудобное самостоятельное заселение\b",
                    r"\bсамостоятельно заселились без проблем\b",
                    r"\bсамо-заселение прошло легко\b",
                    r"\bчек[- ]?ин без персонала\b",
                    r"\bзаселение по коду очень удобно\b",
                ],
                "en": [
                    r"\bself check[- ]?in was easy\b",
                    r"\beasy self check[- ]?in\b",
                    r"\bvery easy to self check[- ]?in\b",
                    r"\bwe could check in ourselves\b",
                    r"\bdidn't need anyone to check in\b",
                ],
                "tr": [
                    r"\bkendi kendimize giriş çok rahattı\b",
                    r"\bself check[- ]?in çok kolaydı\b",
                    r"\bpersonel olmadan kolayca girdik\b",
                    r"\bşifreyle giriş çok rahattı\b",
                ],
                "ar": [
                    r"\bدخول ذاتي سهل\b",
                    r"\bدخلنا لحالنا بسهولة\b",
                    r"\bما احتجنا حد عند الدخول\b",
                    r"\bعملية الself check in كانت سهلة\b",
                ],
                "zh": [
                    r"自助入住很方便",
                    r"自助办理入住很容易",
                    r"不用工作人员也能入住",
                    r"我们自己就能顺利入住",
                ],
            },
        ),
        
        
        "wifi_info_given": AspectRule(
            aspect_code="wifi_info_given",
            polarity_hint="positive",
            display="Данные для доступа к Wi-Fi предоставлены сразу",
            display_short="пароль Wi-Fi выдан сразу",
            long_hint="Гости отмечают, что логин и пароль Wi-Fi были выданы заранее или в момент заселения без дополнительного запроса.",
            patterns_by_lang={{
                "ru": [
                    r"\bсразу дали пароль от вай[- ]?фай\b",
                    r"\bпароль вай[- ]?фай был в инструкции\b",
                    r"\bвся информация по wi.?fi была заранее\b",
                    r"\bинтернет сразу выдали\b",
                    r"\bлогин и пароль от wi.?fi отправили заранее\b",
                ],
                "en": [
                    r"\bwifi info was provided\b",
                    r"\bthey gave us the wifi password right away\b",
                    r"\bwifi details in the instructions\b",
                    r"\binternet login was sent in advance\b",
                    r"\bwe had the wifi code already\b",
                ],
                "tr": [
                    r"\bwifi şifresini hemen verdiler\b",
                    r"\bwifi bilgileri önceden gönderildi\b",
                    r"\binternete direkt bağlandık\b",
                    r"\bwifi bilgisi talimatta vardı\b",
                ],
                "ar": [
                    r"\bأعطونا باسورد الواي فاي فوراً\b",
                    r"\bمعلومات الواي فاي كانت جاهزة\b",
                    r"\bالنت جاهز من أولها\b",
                    r"\bباسورد الواي فاي مكتوب بالتعليمات\b",
                ],
                "zh": [
                    r"Wi-Fi密码一开始就给了",
                    r"入住时就给了Wi-Fi信息",
                    r"指引里有无线网账号密码",
                    r"我们一到就能上网",
                ],
            },
        ),
        
        
        "instructions_confusing": AspectRule(
            aspect_code="instructions_confusing",
            polarity_hint="negative",
            display="Неясные или противоречивые инструкции по заселению",
            display_short="инструкции непонятны",
            long_hint="Гости фиксируют, что разъяснения по доступу, Wi-Fi или правилам проживания были сформулированы неполно или запутанно и потребовали дополнительных уточнений у СПиР.",
            patterns_by_lang={{
                "ru": [
                    r"\bинструкция непонятная\b",
                    r"\bочень запутанные инструкции\b",
                    r"\bне до конца ясно как попасть\b",
                    r"\bпришлось уточнять как зайти\b",
                    r"\bне смогли разобраться по описанию\b",
                ],
                "en": [
                    r"\bconfusing instructions\b",
                    r"\bunclear instructions\b",
                    r"\bthe instructions were not clear\b",
                    r"\bdidn't understand how to enter\b",
                    r"\bwe had to ask how to get in\b",
                ],
                "tr": [
                    r"\btalimatlar kafa karıştırıcıydı\b",
                    r"\bnasıl gireceğimiz çok net değildi\b",
                    r"\banlatım yeterince açık değildi\b",
                    r"\btekrar sormak zorunda kaldık\b",
                ],
                "ar": [
                    r"\bالتعليمات مش واضحة\b",
                    r"\bما فهمنا إزاي ندخل\b",
                    r"\bشرحهم لخبطنا\b",
                    r"\bاضطرينا نسأل تاني عشان نفهم\b",
                ],
                "zh": [
                    r"指引很乱",
                    r"入住说明不清楚",
                    r"看说明还是不知道怎么进门",
                    r"我们还得再问怎么进去",
                ],
            },
        ),
        
        
        "late_access_code": AspectRule(
            aspect_code="late_access_code",
            polarity_hint="negative",
            display="Доступные коды высланы с задержкой",
            display_short="код пришёл поздно",
            long_hint="Гости сообщают, что код доступа к зданию или номеру был отправлен с опозданием: пришлось ждать у входа, писать или звонить, чтобы получить код.",
            patterns_by_lang={{
                "ru": [
                    r"\bкод прислали слишком поздно\b",
                    r"\bждали код у входа\b",
                    r"\bждали пока пришлют доступ\b",
                    r"\bбез кода не могли зайти\b",
                    r"\bпришлось звонить чтобы получить код\b",
                ],
                "en": [
                    r"\baccess code came late\b",
                    r"\bwe had to wait for the code\b",
                    r"\bthey didn't send the code on time\b",
                    r"\bwe stood outside waiting for the code\b",
                    r"\bhad to call to get the door code\b",
                ],
                "tr": [
                    r"\bgiriş kodu geç geldi\b",
                    r"\bkapının şifresi zamanında gelmedi\b",
                    r"\bkapıda beklemek zorunda kaldık\b",
                    r"\bkodu almak için aramak zorunda kaldık\b",
                ],
                "ar": [
                    r"\bالكود اتبعت متأخر\b",
                    r"\bوقفنا برا نستنى الكود\b",
                    r"\bما بعتوش الكود في الوقت\b",
                    r"\bاضطرينا نتصل عشان ناخد كود الباب\b",
                ],
                "zh": [
                    r"门码发得很晚",
                    r"我们在门口等密码",
                    r"没及时给进门码",
                    r"后来打电话才给门锁密码",
                ],
            },
        ),
        
        
        "wifi_info_missing": AspectRule(
            aspect_code="wifi_info_missing",
            polarity_hint="negative",
            display="Отсутствие информации по Wi-Fi при заселении",
            display_short="нет данных Wi-Fi",
            long_hint="Гости фиксируют, что пароль/доступ к Wi-Fi не был предоставлен автоматически, и им пришлось отдельно запрашивать данные или искать их самостоятельно.",
            patterns_by_lang={{
                "ru": [
                    r"\bне дали пароль от вай[- ]?фая\b",
                    r"\bне дали вай[- ]?фай\b",
                    r"\bпришлось спрашивать пароль от wi.?fi\b",
                    r"\bинформации про интернет не было\b",
                    r"\bне нашли пароль от wi.?fi\b",
                ],
                "en": [
                    r"\bno wifi info\b",
                    r"\bdidn't give us the wifi password\b",
                    r"\bhad to ask for the wifi password\b",
                    r"\bno internet info in the room\b",
                    r"\bwe couldn't find the wifi code\b",
                ],
                "tr": [
                    r"\bwifi şifresi verilmedi\b",
                    r"\bwifi bilgisi yoktu\b",
                    r"\bwifi şifresini sormak zorunda kaldık\b",
                    r"\binternete nasıl bağlanacağımız yazmıyordu\b",
                ],
                "ar": [
                    r"\bما عطوناش باسورد الواي فاي\b",
                    r"\bما في أي معلومات للواي فاي\b",
                    r"\bسألناهم عشان نعرف الباسورد\b",
                    r"\bما لقينا باسورد النت في الغرفة\b",
                ],
                "zh": [
                    r"没有给Wi-Fi密码",
                    r"没有Wi-Fi信息",
                    r"我们还得去问Wi-Fi密码",
                    r"房间里找不到无线网密码",
                ],
            },
        ),
    
        "had_to_figure_out": AspectRule(
            aspect_code="had_to_figure_out",
            polarity_hint="negative",
            display="Гостям пришлось разбираться самим",
            display_short="разбирались сами",
            long_hint="Гости фиксируют, что им пришлось самим разбираться с заездом, доступом, оборудованием в номере или правилами проживания из-за отсутствия понятных инструкций и недостатка сопровождения со стороны СПиР.",
            patterns_by_lang={{
                "ru": [
                    r"\bпришлось разбираться самим\b",
                    r"\bразбирались сами\b",
                    r"\bвсё искали сами\b",
                    r"\bникто не подсказал\b",
                    r"\bникто не объяснил как что работает\b",
                    r"\bдогадались сами\b",
                ],
                "en": [
                    r"\bhad to figure it out ourselves\b",
                    r"\bwe had to figure everything out\b",
                    r"\bno one explained how anything works\b",
                    r"\bwe had to work it out on our own\b",
                    r"\bhad to find everything by ourselves\b",
                ],
                "tr": [
                    r"\bher şeyi kendimiz çözmek zorunda kaldık\b",
                    r"\bkimse anlatmadı\b",
                    r"\bher şeyi kendi başımıza anlamak zorunda kaldık\b",
                    r"\bneyin nasıl çalıştığını biz bulduk\b",
                ],
                "ar": [
                    r"\bاضطرينا نفهم كل شي لوحدنا\b",
                    r"\bما حد شرح لنا أي حاجة\b",
                    r"\bكل شي اكتشفناه بنفسنا\b",
                    r"\bسيبونا نحاول لحالنا\b",
                ],
                "zh": [
                    r"都要我们自己摸索",
                    r"没人教我们怎么用",
                    r"一切都得自己搞清楚",
                    r"只能自己想办法",
                ],
            },
        ),
        
        
        "support_during_stay_good": AspectRule(
            aspect_code="support_during_stay_good",
            polarity_hint="positive",
            display="Сопровождение гостей в период проживания",
            display_short="поддержка во время проживания",
            long_hint="Гости отмечают, что в ходе проживания сотрудники оставались вовлечёнными: оперативно реагировали на бытовые вопросы, помогали с мелочами, контролировали комфорт, проверяли, всё ли в порядке.",
            patterns_by_lang={{
                "ru": [
                    r"\bнас поддерживали в течение проживания\b",
                    r"\bв течение проживания всегда помогали\b",
                    r"\bспрашивали всё ли в порядке\b",
                    r"\bперсонал на связи всё время\b",
                    r"\bмогли написать в любой момент\b",
                    r"\bзаботились о нас во время проживания\b",
                ],
                "en": [
                    r"\bgreat support during our stay\b",
                    r"\bthey checked on us during the stay\b",
                    r"\basked if we needed anything\b",
                    r"\balways available during our stay\b",
                    r"\bthey took care of us the whole stay\b",
                ],
                "tr": [
                    r"\bkaldığımız süre boyunca hep yardımcı oldular\b",
                    r"\bkonaklama sırasında sürekli ilgilendiler\b",
                    r"\bbir şeye ihtiyacımız var mı diye sordular\b",
                    r"\bkalış boyunca her zaman ulaşılabilirdi\b",
                ],
                "ar": [
                    r"\bكانوا متابعين معنا طول الإقامة\b",
                    r"\bيسألونا إذا محتاجين شي\b",
                    r"\bدعم ممتاز خلال الإقامة\b",
                    r"\bموجودين طول فترة الإقامة\b",
                ],
                "zh": [
                    r"在入住期间一直有跟进",
                    r"期间随时有人可以帮忙",
                    r"还主动问我们需不需要什么",
                    r"整个入住过程都有照顾到我们",
                ],
            },
        ),
        
        
        "issue_fixed_immediately": AspectRule(
            aspect_code="issue_fixed_immediately",
            polarity_hint="positive",
            display="Мгновенное устранение проблемы в период проживания",
            display_short="моментально исправили",
            long_hint="Гости отмечают, что бытовой или технический вопрос был устранён сразу после обращения, без задержек.",
            patterns_by_lang={{
                "ru": [
                    r"\bсразу решили проблему\b",
                    r"\bрешили вопрос за минуту\b",
                    r"\bмоментально исправили\b",
                    r"\bпринесли сразу как попросили\b",
                    r"\bсразу принесли то что не хватало\b",
                    r"\bпришли прямо сразу\b",
                ],
                "en": [
                    r"\bthey fixed it immediately\b",
                    r"\bproblem was solved right away\b",
                    r"\bbrought what we needed immediately\b",
                    r"\bthey came right away and fixed it\b",
                    r"\bresolved within minutes\b",
                ],
                "tr": [
                    r"\bhemen hallettiler\b",
                    r"\bsorunu anında çözdüler\b",
                    r"\bisteğimiz hemen getirildi\b",
                    r"\bçok hızlı müdahale ettiler\b",
                ],
                "ar": [
                    r"\bحلّوا المشكلة فوراً\b",
                    r"\bجابوا اللي طلبناه على طول\b",
                    r"\bجُم على طول و صلحوا\b",
                    r"\bاتعالجت على طول\b",
                ],
                "zh": [
                    r"马上就修好",
                    r"立刻解决了问题",
                    r"我们一说他们马上就送来了",
                    r"几分钟内就处理好了",
                ],
            },
        ),
        
        
        "support_during_stay_slow": AspectRule(
            aspect_code="support_during_stay_slow",
            polarity_hint="negative",
            display="Медленная реакция на запросы во время проживания",
            display_short="медленная поддержка",
            long_hint="Гости фиксируют, что оперативной поддержки в процессе проживания не было: на бытовые запросы реагировали с задержкой, приходилось напоминать о себе, ждать прихода сотрудника.",
            patterns_by_lang={{
                "ru": [
                    r"\bприходилось ждать пока придут\b",
                    r"\bочень долго реагировали\b",
                    r"\bподдержка медленная\b",
                    r"\bпросили несколько раз\b",
                    r"\bпришлось напоминать\b",
                    r"\bс трудом дождались помощи\b",
                ],
                "en": [
                    r"\bslow support during the stay\b",
                    r"\btook a long time to get help\b",
                    r"\bhad to ask multiple times\b",
                    r"\bwe had to remind them\b",
                    r"\bit took them a long time to come\b",
                ],
                "tr": [
                    r"\bdestek çok yavaştı\b",
                    r"\byardım gelmesi uzun sürdü\b",
                    r"\bdefalarca hatırlatmak zorunda kaldık\b",
                    r"\bbirinin gelmesini bekledik\b",
                ],
                "ar": [
                    r"\bالمساعدة كانت بطيئة\b",
                    r"\bاستنينا كتير لحد ما حد ييجي\b",
                    r"\bكان لازم نذكرهم أكثر من مرة\b",
                    r"\bالتجاوب كان بطيء أثناء الإقامة\b",
                ],
                "zh": [
                    r"等很久才有人来帮忙",
                    r"期间服务很慢",
                    r"我们不得不一直催",
                    r"需要反复提醒他们才来",
                ],
            },
        ),
        
        
        "support_ignored": AspectRule(
            aspect_code="support_ignored",
            polarity_hint="negative",
            display="Отсутствие поддержки во время проживания",
            display_short="поддержка отсутствует",
            long_hint="Гости сообщают, что их запросы в период проживания были оставлены без внимания: никто не пришёл помочь, не вернулись с решением, перестали отвечать после первого контакта.",
            patterns_by_lang={{
                "ru": [
                    r"\bнас просто проигнорировали\b",
                    r"\bникто так и не пришёл\b",
                    r"\bподдержки не дождались\b",
                    r"\bперестали отвечать\b",
                    r"\bпообещали вернуться и не вернулись\b",
                ],
                "en": [
                    r"\bthey ignored us during the stay\b",
                    r"\bno one ever came to help\b",
                    r"\bwe never got any help\b",
                    r"\bthey stopped replying\b",
                    r"\bsaid they'd come but never did\b",
                ],
                "tr": [
                    r"\bbizi tamamen görmezden geldiler\b",
                    r"\bkimse yardıma gelmedi\b",
                    r"\byardım istememize rağmen kimse gelmedi\b",
                    r"\bgeleceğiz dediler ama gelmediler\b",
                ],
                "ar": [
                    r"\bطنشونا تماماً\b",
                    r"\bما حد جه يساعد\b",
                    r"\bقالوا حييجوا بس ما حدش جه\b",
                    r"\bبطلوا يردوا علينا\b",
                ],
                "zh": [
                    r"完全没人来帮忙",
                    r"说会来但没人来",
                    r"后来就不回复我们了",
                    r"求助也没人理",
                ],
            },
        ),

        "promised_not_done": AspectRule(
            aspect_code="promised_not_done",
            polarity_hint="negative",
            display="Обещания со стороны персонала не были выполнены",
            display_short="обещали, но не сделали",
            long_hint="Гости фиксируют, что сотрудники подтверждали, что вернутся с решением (должны были принести, починить, заменить, уточнить), но обещанное действие так и не было выполнено.",
            patterns_by_lang={{
                "ru": [
                    r"\bобещал[аи]? и не сдел[ао]ли\b",
                    r"\bсказал[аи]? что сейчас принесут но не принесли\b",
                    r"\bобещали вернуться и не вернулись\b",
                    r"\bсказали что исправят но так и не исправили\b",
                    r"\bпообещали и забыли\b",
                ],
                "en": [
                    r"\bthey promised but didn't do it\b",
                    r"\bsaid they'd fix it but they never did\b",
                    r"\bsaid they would bring it but never came\b",
                    r"\bthey never came back\b",
                    r"\bpromised to sort it out but didn't\b",
                ],
                "tr": [
                    r"\bsöz verdiler ama yapmadılar\b",
                    r"\bgetireceğiz dediler ama getirmediler\b",
                    r"\blyetkilisi geleceğiz dedi ama gelmedi\b",
                    r"\bdöneceğiz dediler ama dönmediler\b",
                ],
                "ar": [
                    r"\bقالوا حيحلّوها بس ما صار شي\b",
                    r"\bوعدونا وبرضه ما عملوش حاجة\b",
                    r"\bقالوا حنرجع لكم وما حد رجع\b",
                    r"\bوعدوا يجيبوا الشي وما جابوش\b",
                ],
                "zh": [
                    r"说了会处理但没处理",
                    r"答应会拿来结果没拿来",
                    r"说马上回来结果没回来",
                    r"承诺了但没有做到",
                ],
            },
        ),
        
        
        "checkout_easy": AspectRule(
            aspect_code="checkout_easy",
            polarity_hint="positive",
            display="Простой и понятный выезд",
            display_short="чекаут без сложности",
            long_hint="Гости отмечают, что процедура выезда была простой: понятный порядок сдачи ключей, отсутствие бюрократии, не потребовалось дополнительное ожидание персонала.",
            patterns_by_lang={{
                "ru": [
                    r"\bудобный выезд\b",
                    r"\bчекаут прош[её]л легко\b",
                    r"\bвыехать было просто\b",
                    r"\bбез лишних формальностей при выезде\b",
                    r"\bвсё очень просто при выезде\b",
                ],
                "en": [
                    r"\beasy check[- ]?out\b",
                    r"\bcheckout was easy\b",
                    r"\bsmooth checkout\b",
                    r"\bleaving was simple\b",
                    r"\bno hassle at checkout\b",
                ],
                "tr": [
                    r"\bçıkış çok kolaydı\b",
                    r"\bcheckout çok rahattı\b",
                    r"\bçıkış işlemi sorunsuzdu\b",
                    r"\bhiç uğraşmadan çıktık\b",
                ],
                "ar": [
                    r"\bالخروج كان سهل\b",
                    r"\bالـcheckout كان بسيط\b",
                    r"\bطلعنا بدون تعقيد\b",
                    r"\bما كان في إجراءات معقدة وقت الخروج\b",
                ],
                "zh": [
                    r"退房很方便",
                    r"退房流程很简单",
                    r"轻松就退房了",
                    r"退房没有麻烦",
                ],
            },
        ),
        
        
        "checkout_fast": AspectRule(
            aspect_code="checkout_fast",
            polarity_hint="positive",
            display="Быстрый чекаут",
            display_short="быстрый выезд",
            long_hint="Гости отмечают, что выезд занял минимум времени.",
            patterns_by_lang={{
                "ru": [
                    r"\bбыстрый выезд\b",
                    r"\bочень быстро выписали\b",
                    r"\bбуквально минута и всё\b",
                    r"\bбыстрый чекаут\b",
                    r"\bмы выехали за минуту\b",
                ],
                "en": [
                    r"\bfast checkout\b",
                    r"\bquick check[- ]?out\b",
                    r"\bcheckout took a minute\b",
                    r"\bwe checked out in no time\b",
                    r"\bthey checked us out quickly\b",
                ],
                "tr": [
                    r"\bçok hızlı çıkış yaptık\b",
                    r"\bcheckout çok hızlıydı\b",
                    r"\bbir dakikada çıktık\b",
                    r"\bçıkış işlemi hemen bitti\b",
                ],
                "ar": [
                    r"\bالـcheckout كان سريع جداً\b",
                    r"\bخلصنا الخروج بدقايق\b",
                    r"\bطلعنا بسرعة\b",
                    r"\bما أخد وقت الخروج\b",
                ],
                "zh": [
                    r"退房很快",
                    r"很快就办好退房",
                    r"一分钟就走完流程",
                    r"退房几乎不花时间",
                ],
            },
        ),
        
        
        "checkout_slow": AspectRule(
            aspect_code="checkout_slow",
            polarity_hint="negative",
            display="Затянутая процедура выезда",
            display_short="медленный чекаут",
            long_hint="Гости фиксируют, что выезд занял больше времени, чем ожидалось: ожидание сотрудника, проверка номера, задержка с оформлением документов.",
            patterns_by_lang={{
                "ru": [
                    r"\bмедленный выезд\b",
                    r"\bчекаут был очень долгим\b",
                    r"\bпришлось ждать чтобы выехать\b",
                    r"\bждали пока кто-то придет на ресепшен\b",
                    r"\bоформление выезда заняло слишком долго\b",
                ],
                "en": [
                    r"\bslow checkout\b",
                    r"\bcheckout took too long\b",
                    r"\bhad to wait to check out\b",
                    r"\bwe waited a long time to leave\b",
                    r"\bthey kept us waiting at checkout\b",
                ],
                "tr": [
                    r"\byavaş checkout\b",
                    r"\bçıkış için çok bekledik\b",
                    r"\bçıkmak için beklemek zorunda kaldık\b",
                    r"\bcheckout gereksiz uzun sürdü\b",
                ],
                "ar": [
                    r"\bالخروج كان بطيء\b",
                    r"\bاستنينا كتير عشان نطلع\b",
                    r"\bعلقونا عند الـcheckout\b",
                    r"\bأخذوا وقت طويل عشان يخلّونا نطلع\b",
                ],
                "zh": [
                    r"退房很慢",
                    r"退房等了很久",
                    r"花了很久才让我们退房",
                    r"办理退房拖很久",
                ],
            },
        ),
        
        
        "deposit_return_issue": AspectRule(
            aspect_code="deposit_return_issue",
            polarity_hint="negative",
            display="Задержки или сложности с возвратом депозита при выезде",
            display_short="проблемы с возвратом депозита",
            long_hint="Гости сообщают, что после выезда возникли трудности с возвратом депозита: деньги не вернули сразу, возврат задержан или условия возврата были изменены на выезде.",
            patterns_by_lang={{
                "ru": [
                    r"\bдепозит не вернули\b",
                    r"\bне вернули залог\b",
                    r"\bждем возврат депозита\b",
                    r"\bпопросили подождать с депозитом\b",
                    r"\bдепозит до сих пор не вернулся\b",
                ],
                "en": [
                    r"\bdeposit not returned\b",
                    r"\bthey didn't give us our deposit back\b",
                    r"\bstill waiting for the deposit\b",
                    r"\bdeposit refund delayed\b",
                    r"\bissue with getting the deposit back\b",
                ],
                "tr": [
                    r"\bdepozito geri verilmedi\b",
                    r"\bdepozitoyu hemen geri ödemediler\b",
                    r"\bdepozito iadesi gecikti\b",
                    r"\bhalen depozito bekliyoruz\b",
                ],
                "ar": [
                    r"\bما رجعولنا العربون\b",
                    r"\bلسه مستنيين التأمين\b",
                    r"\bتأخير برجوع العربون\b",
                    r"\bما رجعش العربون وقت الـcheckout\b",
                ],
                "zh": [
                    r"押金还没退",
                    r"退押金拖延",
                    r"退房后押金没拿回来",
                    r"还在等押金退款",
                ],
            },
        ),

        "checkout_no_staff": AspectRule(
            aspect_code="checkout_no_staff",
            polarity_hint="negative",
            display="Отсутствие сотрудника на стойке при выезде",
            display_short="нет персонала при выезде",
            long_hint="Гости фиксируют, что во время чекаута на стойке никого не было: не к кому обратиться, ключи/карты пришлось оставлять без подтверждения, возникла неопределённость с завершением проживания.",
            patterns_by_lang={{
                "ru": [
                    r"\bникого на ресепшене при выезде\b",
                    r"\bна выезде никого не было\b",
                    r"\bне было сотрудника для чекаута\b",
                    r"\bключи пришлось оставить\b",
                ],
                "en": [
                    r"\bno staff at checkout\b",
                    r"\bno one at reception\b",
                    r"\bnobody to check us out\b",
                    r"\bhad to leave the keys\b",
                ],
                "tr": [
                    r"\bcheckoutta kimse yoktu\b",
                    r"\bresepsiyonda kimse yoktu\b",
                    r"\bçıkış için kimse yoktu\b",
                    r"\banahtarları bırakmak zorunda kaldık\b",
                ],
                "ar": [
                    r"\bما كانش فيه حد وقت الخروج\b",
                    r"\bالاستقبال فاضي\b",
                    r"\bما لقيناش حد يعمل checkout\b",
                    r"\bاضطرينا نسيب المفاتيح\b",
                ],
                "zh": [
                    r"退房时前台没人",
                    r"退房没人办理",
                    r"只能把钥匙放下",
                    r"前台空无一人",
                ],
            },
        ),
        
        
        "fresh_bedding": AspectRule(
            aspect_code="fresh_bedding",
            polarity_hint="positive",
            display="Свежесть и чистота постельного белья при заезде",
            display_short="свежее постельное бельё",
            long_hint="Гости отмечают, что постельное бельё чистое и свежее на момент заезда: визуально без пятен, со свежим запахом, соответствует стандарту подготовки номера.",
            patterns_by_lang={{
                "ru": [
                    r"\bсвежее бель[её]\b",
                    r"\bчистое постельное бель[её]\b",
                    r"\bпростыни чистые\b",
                    r"\bпахло свежестью\b",
                ],
                "en": [
                    r"\bfresh bedding\b",
                    r"\bclean sheets\b",
                    r"\bfresh linen\b",
                    r"\bsmelled fresh\b",
                ],
                "tr": [
                    r"\bçarşaflar tertemizdi\b",
                    r"\btemiz nevresim\b",
                    r"\bçarşaflar mis gibiydi\b",
                ],
                "ar": [
                    r"\bملايات نظيفة\b",
                    r"\bفرش نضيف\b",
                    r"\bريحة نظافة\b",
                ],
                "zh": [
                    r"床品很干净",
                    r"床单很干净",
                    r"床上用品很清新",
                ],
            },
        ),
        
        
        "no_dust_surfaces": AspectRule(
            aspect_code="no_dust_surfaces",
            polarity_hint="positive",
            display="Отсутствие пыли на поверхностях",
            display_short="поверхности без пыли",
            long_hint="Гости отмечают, что горизонтальные поверхности и полки очищены от пыли на момент заезда, что подтверждает качественную подготовку номера.",
            patterns_by_lang={{
                "ru": [
                    r"\bнет пыли\b",
                    r"\bповерхности без пыли\b",
                    r"\bна полках чисто\b",
                ],
                "en": [
                    r"\bno dust\b",
                    r"\bdust[- ]?free surfaces\b",
                    r"\bno dust on shelves\b",
                ],
                "tr": [
                    r"\btoz yoktu\b",
                    r"\byüzeyler tozsuzdu\b",
                    r"\braflarda toz yoktu\b",
                ],
                "ar": [
                    r"\bمفيش تراب\b",
                    r"\bالأسطح من غير غبار\b",
                    r"\bمفيش غبار على الرفوف\b",
                ],
                "zh": [
                    r"没有灰尘",
                    r"表面没有灰",
                    r"架子上没有灰尘",
                ],
            },
        ),
        
        
        "floor_clean": AspectRule(
            aspect_code="floor_clean",
            polarity_hint="positive",
            display="Чистый пол при заезде",
            display_short="чистый пол",
            long_hint="Гости отмечают, что полы в номере вымыты и чистые при заселении: без мусора и липких следов, соответствует ожиданиям базовой чистоты.",
            patterns_by_lang={{
                "ru": [
                    r"\bпол (чистый|чисто)\b",
                    r"\bпол был вымыт\b",
                    r"\bчисто на полу\b",
                ],
                "en": [
                    r"\bclean floor\b",
                    r"\bfloors were clean\b",
                    r"\bfloor was freshly cleaned\b",
                ],
                "tr": [
                    r"\bzemin temizdi\b",
                    r"\byerler temizdi\b",
                    r"\byeni silinmişti\b",
                ],
                "ar": [
                    r"\bالأرضية نضيفة\b",
                    r"\bالأرض متنضفة كويس\b",
                    r"\bالأرض متلمعة\b",
                ],
                "zh": [
                    r"地板很干净",
                    r"地面很干净",
                    r"地板刚打扫过",
                ],
            },
        ),
        
        
        "dusty_surfaces": AspectRule(
            aspect_code="dusty_surfaces",
            polarity_hint="negative",
            display="Пыль на поверхностях при заезде",
            display_short="пыль на поверхностях",
            long_hint="Гости фиксируют наличие пыли на мебельных поверхностях, полках или подоконниках при заселении — индикатор недостаточной подготовки номера.",
            patterns_by_lang={{
                "ru": [
                    r"\bпыль на (поверхностях|полках|мебели|подоконнике)\b",
                    r"\bочень пыльно\b",
                    r"\bпыли много\b",
                ],
                "en": [
                    r"\bdusty surfaces\b",
                    r"\bdust on surfaces\b",
                    r"\bdust on shelves\b",
                    r"\bvery dusty\b",
                ],
                "tr": [
                    r"\byüzeyler tozluydu\b",
                    r"\braflar tozluydu\b",
                    r"\bçok tozluydu\b",
                ],
                "ar": [
                    r"\bغبار على الأسطح\b",
                    r"\bالرفوف عليها تراب\b",
                    r"\bالمكان مترب\b",
                ],
                "zh": [
                    r"表面有灰尘",
                    r"很脏有灰",
                    r"架子上都是灰",
                ],
            },
        ),
        
        "sticky_surfaces": AspectRule(
            aspect_code="sticky_surfaces",
            polarity_hint="negative",
            display="Липкие/грязные поверхности при заезде",
            display_short="липкие поверхности",
            long_hint="Гости фиксируют, что на столешницах, ручках, кухонной зоне или других поверхностях остались липкие следы, разводы, следы еды или жира на момент заезда.",
            patterns_by_lang={{
                "ru": [
                    r"\bлипкие поверхности\b",
                    r"\bвсё липкое\b",
                    r"\bстол был липкий\b",
                    r"\bжирные пятна\b",
                    r"\bгрязная столешница\b",
                    r"\bгрязные ручки\b",
                ],
                "en": [
                    r"\bsticky surfaces\b",
                    r"\bthe table was sticky\b",
                    r"\bsticky table\b",
                    r"\bgreasy stains\b",
                    r"\bgreasy countertop\b",
                    r"\bdirty handles\b",
                ],
                "tr": [
                    r"\byüzeyler yapışıktı\b",
                    r"\bmasa yapış yapıştı\b",
                    r"\btezgah yağlıydı\b",
                    r"\byağ lekeleri vardı\b",
                ],
                "ar": [
                    r"\bالأسطح لزجة\b",
                    r"\bالطرابيزة كانت لزقة\b",
                    r"\bبقع دهن\b",
                    r"\bالمقابض مش نظيفة\b",
                ],
                "zh": [
                    r"桌面黏黏的",
                    r"表面很黏",
                    r"台面油乎乎的",
                    r"把手很脏",
                ],
            },
        ),
        
        
        "stained_bedding": AspectRule(
            aspect_code="stained_bedding",
            polarity_hint="negative",
            display="Пятна на постельном белье при заезде",
            display_short="пятна на белье",
            long_hint="Гости отмечают наличие пятен на простынях, пододеяльниках или наволочках при заселении — восприятие белья как использованного или не до конца подготовленного.",
            patterns_by_lang={{
                "ru": [
                    r"\bпятна на бель(ь|е)\b",
                    r"\bгрязное постельное бель[её]\b",
                    r"\bна простыне пятна\b",
                    r"\bгрязные наволочки\b",
                    r"\bбелье было не свежее\b",
                ],
                "en": [
                    r"\bstains on the sheets\b",
                    r"\bstained bedding\b",
                    r"\bdirty sheets\b",
                    r"\bmarks on the pillowcase\b",
                    r"\bthe duvet cover had stains\b",
                ],
                "tr": [
                    r"\bçarşaflarda lekeler vardı\b",
                    r"\byatak örtüsü lekeli\b",
                    r"\byastık kılıfı kirliydi\b",
                    r"\btemiz olmayan çarşaf\b",
                ],
                "ar": [
                    r"\bملايات عليها بقع\b",
                    r"\bالمخدة مش نضيفة\b",
                    r"\bشرشف وسخ\b",
                    r"\bالبطانية عليها بقع\b",
                ],
                "zh": [
                    r"床单有污渍",
                    r"被套上有痕迹",
                    r"枕套不干净",
                    r"床品有脏污",
                ],
            },
        ),
        
        
        "hair_on_bed": AspectRule(
            aspect_code="hair_on_bed",
            polarity_hint="negative",
            display="Волосы на постели при заезде",
            display_short="волосы на постели",
            long_hint="Гости фиксируют наличие чужих волос на простынях, подушках или одеяле при заселении, что интерпретируется как недостаточная смена белья или некачественная уборка.",
            patterns_by_lang={{
                "ru": [
                    r"\bволосы на кровати\b",
                    r"\bволосы на постели\b",
                    r"\bволосы на простыне\b",
                    r"\bнашли чужие волосы\b",
                ],
                "en": [
                    r"\bhair on the bed\b",
                    r"\bhair on the sheets\b",
                    r"\bhair in the bed\b",
                    r"\bhair on the pillow\b",
                ],
                "tr": [
                    r"\byatakta saç vardı\b",
                    r"\bçarşafta saç bulduk\b",
                    r"\byastıkta saç vardı\b",
                ],
                "ar": [
                    r"\bلقينا شعر على السرير\b",
                    r"\bشعر على الملاية\b",
                    r"\bشعر على المخدة\b",
                ],
                "zh": [
                    r"床上有头发",
                    r"床单上有头发",
                    r"枕头上有头发",
                ],
            },
        ),
        
        
        "used_towels_left": AspectRule(
            aspect_code="used_towels_left",
            polarity_hint="negative",
            display="Оставленные использованные полотенца от предыдущего гостя",
            display_short="чужие использованные полотенца",
            long_hint="Гости отмечают, что в санузле остались использованные полотенца или текстиль предыдущих гостей на момент заезда.",
            patterns_by_lang={{
                "ru": [
                    r"\bгрязные полотенца остались\b",
                    r"\bчужие полотенца\b",
                    r"\bиспользованные полотенца в ванной\b",
                    r"\bполотенца не заменили\b",
                ],
                "en": [
                    r"\bused towels left\b",
                    r"\bdirty towels left in the bathroom\b",
                    r"\bprevious guest's towels\b",
                    r"\btowels were not replaced\b",
                ],
                "tr": [
                    r"\bkullanılmış havlular bırakılmıştı\b",
                    r"\bbanyoda kirli havlu vardı\b",
                    r"\bhavlular değiştirilmemişti\b",
                ],
                "ar": [
                    r"\bلقينا مناشف مستعملة\b",
                    r"\bالمناشف وسخة ولسه موجودة\b",
                    r"\bما بدلوش الفوط\b",
                ],
                "zh": [
                    r"浴室里有用过的毛巾",
                    r"毛巾没换新的",
                    r"留着别人用过的毛巾",
                ],
            },
        ),
        
        
        "crumbs_left": AspectRule(
            aspect_code="crumbs_left",
            polarity_hint="negative",
            display="Остатки еды / крошки при заезде",
            display_short="крошки и остатки еды",
            long_hint="Гости фиксируют присутствие крошек, следов еды или липких пятен на столах, полу или кухонной зоне при заселении, что указывает на некачественную финальную уборку.",
            patterns_by_lang={{
                "ru": [
                    r"\bкрошки на столе\b",
                    r"\bкрошки на полу\b",
                    r"\bостались объедки\b",
                    r"\bследы еды остались\b",
                    r"\bгрязно после предыдущих гостей\b",
                ],
                "en": [
                    r"\bcrumbs on the table\b",
                    r"\bcrumbs on the floor\b",
                    r"\bleftover food\b",
                    r"\bfood leftovers\b",
                    r"\bfood bits left\b",
                ],
                "tr": [
                    r"\bmasada kırıntılar vardı\b",
                    r"\byerde kırıntılar vardı\b",
                    r"\byemek artıkları kalmıştı\b",
                ],
                "ar": [
                    r"\bفتافيت أكل على الترابيزة\b",
                    r"\bبقايا أكل على الأرض\b",
                    r"\bأكل قديم لسه موجود\b",
                ],
                "zh": [
                    r"桌上有碎屑",
                    r"地上有吃的碎屑",
                    r"还留着吃剩的东西",
                ],
            },
        ),

        "bathroom_clean_on_arrival": AspectRule(
            aspect_code="bathroom_clean_on_arrival",
            polarity_hint="positive",
            display="Чистота санузла при заезде",
            display_short="ванная чистая при заезде",
            long_hint="Гости отмечают, что ванная комната/санузел были чистыми на момент заселения: раковина, душ, унитаз и поверхности не вызывали претензий по гигиене.",
            patterns_by_lang={{
                "ru": [
                    r"\bчистая ванная\b",
                    r"\bчистый санузел\b",
                    r"\bв ванной было чисто\b",
                    r"\bвсё в ванной блестело\b",
                    r"\bсанузел идеально чистый\b",
                ],
                "en": [
                    r"\bclean bathroom\b",
                    r"\bbathroom was clean on arrival\b",
                    r"\bvery clean bathroom\b",
                    r"\bspotless bathroom\b",
                    r"\btoilet was very clean\b",
                ],
                "tr": [
                    r"\bbanyo tertemizdi\b",
                    r"\bwc çok temizdi\b",
                    r"\bgeldiğimizde banyo çok temizdi\b",
                    r"\bher yer pırıl pırıltı\b",
                ],
                "ar": [
                    r"\bالحمام كان نضيف جداً\b",
                    r"\bالحمام نضيف وقت ما وصلنا\b",
                    r"\bالحمام كان نظيف من البداية\b",
                    r"\bالحمام مرتب ونضيف\b",
                ],
                "zh": [
                    r"卫生间很干净",
                    r"入住时厕所很干净",
                    r"浴室非常干净",
                    r"洗手间一开始就很干净",
                ],
            },
        ),
        
        
        "no_mold_visible": AspectRule(
            aspect_code="no_mold_visible",
            polarity_hint="positive",
            display="Отсутствие плесени и налёта",
            display_short="без плесени",
            long_hint="Гости отмечают, что в душевой зоне, на швах плитки, у стоков и вокруг сантехники не было видимой плесени или грибка.",
            patterns_by_lang={{
                "ru": [
                    r"\bнет плесени\b",
                    r"\bникакой плесени\b",
                    r"\bшвы чистые без плесени\b",
                    r"\bничего не черное в душе\b",
                ],
                "en": [
                    r"\bno mold\b",
                    r"\bno mildew\b",
                    r"\bno black mold in the shower\b",
                    r"\btiles were clean with no mold\b",
                ],
                "tr": [
                    r"\bküf yoktu\b",
                    r"\bduşta küf yoktu\b",
                    r"\bfayans araları temizdi\b",
                    r"\bhiç kararma yoktu\b",
                ],
                "ar": [
                    r"\bمافيش عفن\b",
                    r"\bمفيش فطر أسود في الشور\b",
                    r"\bمافيش بقع سودة حوالين البلاط\b",
                ],
                "zh": [
                    r"没有霉斑",
                    r"淋浴间没有霉",
                    r"瓷砖缝没有发黑",
                    r"没有霉味/霉点",
                ],
            },
        ),
        
        
        "sink_clean": AspectRule(
            aspect_code="sink_clean",
            polarity_hint="positive",
            display="Состояние раковины при заезде",
            display_short="чистая раковина",
            long_hint="Гости отмечают, что раковина была чистой: без следов зубной пасты, волос, известкового налёта, разводов мыла или грязи.",
            patterns_by_lang={{
                "ru": [
                    r"\bраковина чистая\b",
                    r"\bчистая раковина\b",
                    r"\bраковина без налёта\b",
                    r"\bникаких следов в раковине\b",
                    r"\bраковина была отмыта\b",
                ],
                "en": [
                    r"\bclean sink\b",
                    r"\bthe sink was clean\b",
                    r"\bno residue in the sink\b",
                    r"\bno stains in the sink\b",
                    r"\bno hair in the sink\b",
                ],
                "tr": [
                    r"\blavabo temizdi\b",
                    r"\blavaboda leke yoktu\b",
                    r"\bhiç saç yoktu lavaboda\b",
                    r"\bhiç kir kalmamıştı lavaboda\b",
                ],
                "ar": [
                    r"\bالمغسلة كانت نضيفة\b",
                    r"\bالمغسلة فاضية من غير أوساخ\b",
                    r"\bمفيش شعر بالمغسلة\b",
                    r"\bمفيش آثار معجون ولا صابون\b",
                ],
                "zh": [
                    r"洗手池很干净",
                    r"水槽很干净",
                    r"洗手池里没有脏东西",
                    r"洗手盆里没有头发",
                ],
            },
        ),
        
        
        "shower_clean": AspectRule(
            aspect_code="shower_clean",
            polarity_hint="positive",
            display="Чистота душевой зоны при заезде",
            display_short="чистый душ",
            long_hint="Гости отмечают, что душевая кабина или ванна были чистыми: без волос, налёта, следов мыла и известковых отложений.",
            patterns_by_lang={{
                "ru": [
                    r"\bчистый душ\b",
                    r"\bдушевая была чистой\b",
                    r"\bванна чистая\b",
                    r"\bникаких следов в душе\b",
                    r"\bв душе не было волос\b",
                ],
                "en": [
                    r"\bclean shower\b",
                    r"\bshower was clean\b",
                    r"\bthe tub was clean\b",
                    r"\bno hair in the shower\b",
                    r"\bno soap scum\b",
                ],
                "tr": [
                    r"\bduş çok temizdi\b",
                    r"\bduşakabin temizdi\b",
                    r"\bküvette kir yoktu\b",
                    r"\bduşta saç yoktu\b",
                ],
                "ar": [
                    r"\bالدوش كان نضيف\b",
                    r"\bالبانيو نظيف\b",
                    r"\bمافيش شعر في الشور\b",
                    r"\bمافيش بقايا صابون على الجدران\b",
                ],
                "zh": [
                    r"淋浴间很干净",
                    r"浴缸很干净",
                    r"淋浴间没有头发",
                    r"淋浴间没有肥皂垢",
                ],
            },
        ),
        
        
        "bathroom_dirty_on_arrival": AspectRule(
            aspect_code="bathroom_dirty_on_arrival",
            polarity_hint="negative",
            display="Грязный санузел при заезде",
            display_short="грязная ванная при заезде",
            long_hint="Гости фиксируют, что ванная комната была неубрана к моменту заезда: следы волос, известковый налёт, мыльные подтеки, мусор или визуальные следы использования предыдущими гостями.",
            patterns_by_lang={{
                "ru": [
                    r"\bгрязная ванная\b",
                    r"\bгрязный санузел\b",
                    r"\bванная не убрана\b",
                    r"\bв душе было грязно\b",
                    r"\bгрязь в раковине\b",
                    r"\bволосы в ванной\b",
                ],
                "en": [
                    r"\bdirty bathroom\b",
                    r"\bbathroom was dirty on arrival\b",
                    r"\bthe shower was dirty\b",
                    r"\bthere was hair in the bathroom\b",
                    r"\bthe sink was dirty\b",
                ],
                "tr": [
                    r"\bbanyo kirliydi\b",
                    r"\bgeldiğimizde duş kirliydi\b",
                    r"\blavabo temiz değildi\b",
                    r"\bbanyoda saç vardı\b",
                ],
                "ar": [
                    r"\bالحمام كان وسخ لما وصلنا\b",
                    r"\bالدوش مش نضيف\b",
                    r"\bالمغسلة كانت وسخة\b",
                    r"\bلقينا شعر بالحمام\b",
                ],
                "zh": [
                    r"卫生间很脏入住时",
                    r"浴室不干净",
                    r"洗手池很脏",
                    r"浴室里有头发",
                ],
            },
        ),

        "hair_in_shower": AspectRule(
            aspect_code="hair_in_shower",
            polarity_hint="negative",
            display="Волосы в душе при заезде",
            display_short="волосы в душе",
            long_hint="Гости фиксируют наличие волос в душевой зоне или в ванной на момент заселения, что воспринимается как признак некачественной уборки после предыдущих гостей.",
            patterns_by_lang={{
                "ru": [
                    r"\bволосы в душе\b",
                    r"\bволосы в ванной\b",
                    r"\bв душе остались волосы\b",
                    r"\bв ванной чей-то волос\b",
                ],
                "en": [
                    r"\bhair in the shower\b",
                    r"\bhair in the tub\b",
                    r"\bthere was hair in the shower\b",
                    r"\bhair all over the shower\b",
                ],
                "tr": [
                    r"\bduşta saç vardı\b",
                    r"\bküvette saç vardı\b",
                    r"\bduşu temizlememişlerdi, saç vardı\b",
                ],
                "ar": [
                    r"\bلقينا شعر بالدوش\b",
                    r"\bفي شعر بالبانيو\b",
                    r"\bالدش فيه شعر قديم\b",
                ],
                "zh": [
                    r"淋浴间有头发",
                    r"浴缸里有头发",
                    r"洗澡间还有别人的头发",
                ],
            },
        ),
        
        
        "hair_in_sink": AspectRule(
            aspect_code="hair_in_sink",
            polarity_hint="negative",
            display="Волосы в раковине при заезде",
            display_short="волосы в раковине",
            long_hint="Гости сообщают, что в раковине остались волосы или следы бритья от предыдущих гостей, то есть гигиеническая подготовка санузла не была завершена.",
            patterns_by_lang={{
                "ru": [
                    r"\bволосы в раковине\b",
                    r"\bв раковине волосы\b",
                    r"\bв раковине остались волосы\b",
                    r"\bследы бритья в раковине\b",
                ],
                "en": [
                    r"\bhair in the sink\b",
                    r"\bthe sink had hair\b",
                    r"\bhair left in the sink\b",
                    r"\bshaving hair in the sink\b",
                ],
                "tr": [
                    r"\blavaboda saç vardı\b",
                    r"\blavaboda tüy kalmıştı\b",
                    r"\blavaboda traş artıkları vardı\b",
                ],
                "ar": [
                    r"\bفي شعر بالمغسلة\b",
                    r"\bالمغسلة فيها شعر قديم\b",
                    r"\bبقايا حلاقة في المغسلة\b",
                ],
                "zh": [
                    r"洗手池里有头发",
                    r"水槽里还有毛发",
                    r"洗手盆里有剃须的毛",
                ],
            },
        ),
        
        
        "mold_in_shower": AspectRule(
            aspect_code="mold_in_shower",
            polarity_hint="negative",
            display="Плесень/чёрный налёт в душевой зоне",
            display_short="плесень в душе",
            long_hint="Гости фиксируют наличие плесени, почерневших швов плитки или потемневших участков силикона в душе/ванной, что интерпретируется как признак износа и недостаточного контроля гигиены.",
            patterns_by_lang={{
                "ru": [
                    r"\bплесень в душе\b",
                    r"\bчёрный налёт в душе\b",
                    r"\bчёрные швы в душе\b",
                    r"\bгрибок на плитке\b",
                    r"\bсиликон почернел\b",
                ],
                "en": [
                    r"\bmold in the shower\b",
                    r"\bmildew in the shower\b",
                    r"\bblack mold on the tiles\b",
                    r"\bblack stains in the shower corners\b",
                    r"\bmold around the silicone\b",
                ],
                "tr": [
                    r"\bduşta küf vardı\b",
                    r"\bduşta siyah küf lekeleri\b",
                    r"\bfayans derzlerinde küf\b",
                    r"\bsilikon yerleri kararmıştı\b",
                ],
                "ar": [
                    r"\bفيه عفن بالدوش\b",
                    r"\bفيه سواد حوالين السليكون\b",
                    r"\bالعفن باين بين البلاط\b",
                    r"\bفيه فطريات بالسور تبع الشور\b",
                ],
                "zh": [
                    r"淋浴间有霉",
                    r"瓷砖缝发黑",
                    r"硅胶边发黑发霉",
                    r"浴室角落有黑霉",
                ],
            },
        ),
        
        
        "limescale_stains": AspectRule(
            aspect_code="limescale_stains",
            polarity_hint="negative",
            display="Известковый налёт и следы воды",
            display_short="известковый налёт",
            long_hint="Гости отмечают наличие белёсого известкового налёта, разводов воды или мыльных отложений на смесителях, стекле душа и сантехнике, что визуально снижает ощущение чистоты санузла.",
            patterns_by_lang={{
                "ru": [
                    r"\bизвестковый нал[её]т\b",
                    r"\bналёт на кран(е|ах)\b",
                    r"\bбелые разводы на душе\b",
                    r"\bизвесть на стекле душа\b",
                    r"\bследы воды на кране\b",
                ],
                "en": [
                    r"\blimescale stains\b",
                    r"\blimescale on the tap\b",
                    r"\bwater marks on the shower glass\b",
                    r"\bsoap scum on the fixtures\b",
                    r"\bwhite buildup on the faucet\b",
                ],
                "tr": [
                    r"\bmuslukta kireç lekeleri\b",
                    r"\bduş camında su lekeleri\b",
                    r"\bmusluklarda kireç birikmişti\b",
                    r"\barmatürde sabun kalıntısı\b",
                ],
                "ar": [
                    r"\bترسّبات كلس على الحنفية\b",
                    r"\bآثار ميّة على الزجاج تبع الشور\b",
                    r"\bآثار صابون على الحنفية\b",
                    r"\bبقع بيضا على الخلاط\b",
                ],
                "zh": [
                    r"水垢很明显",
                    r"龙头上有水垢",
                    r"淋浴玻璃上有水渍",
                    r"水龙头上有白色沉积物",
                ],
            },
        ),
        
        
        "sewage_smell_bathroom": AspectRule(
            aspect_code="sewage_smell_bathroom",
            polarity_hint="negative",
            display="Запах канализации из санузла",
            display_short="запах канализации в санузле",
            long_hint="Гости сообщают о стойком неприятном запахе канализации или застоя в ванной/душевой зоне, что напрямую влияет на восприятие санитарного состояния номера.",
            patterns_by_lang={{
                "ru": [
                    r"\bзапах канализации\b",
                    r"\bвоняло из слива\b",
                    r"\bнеприятный запах в ванной\b",
                    r"\bзапах из ванной\b",
                    r"\bвонь из душевого трапа\b",
                ],
                "en": [
                    r"\bsewage smell in the bathroom\b",
                    r"\bsewer smell\b",
                    r"\bbathroom smelled bad\b",
                    r"\bdrain smell\b",
                    r"\bsmell coming from the drain\b",
                ],
                "tr": [
                    r"\bbanyoda lağım kokusu vardı\b",
                    r"\bkanalizasyon kokuyordu\b",
                    r"\bduş giderinden koku geliyordu\b",
                    r"\bbanyoda kötü koku vardı\b",
                ],
                "ar": [
                    r"\bريحة مجاري في الحمام\b",
                    r"\bريحة صرف صحي طالعة من البلاعة\b",
                    r"\bريحة وحشة من الحمام\b",
                    r"\bريحة مجاري من البالوعة\b",
                ],
                "zh": [
                    r"卫生间有下水道味道",
                    r"有臭的下水道味",
                    r"浴室有一股下水味",
                    r"地漏有异味",
                ],
            },
        ),

        "housekeeping_regular": AspectRule(
            aspect_code="housekeeping_regular",
            polarity_hint="positive",
            display="Регулярная уборка во время проживания",
            display_short="уборка регулярная",
            long_hint="Гости отмечают, что уборка проводилась регулярно в период проживания: номер поддерживался в порядке без необходимости дополнительных напоминаний.",
            patterns_by_lang={{
                "ru": [
                    r"\bуборка каждый день\b",
                    r"\bубирали регулярно\b",
                    r"\bрегулярно убирали номер\b",
                    r"\bуборка по графику\b",
                    r"\bуборка была ежедневно\b",
                ],
                "en": [
                    r"\bdaily cleaning\b",
                    r"\bcleaned the room every day\b",
                    r"\bregular housekeeping\b",
                    r"\bhousekeeping was consistent\b",
                    r"\bthey kept the room tidy during our stay\b",
                ],
                "tr": [
                    r"\bher gün temizlendi\b",
                    r"\boda düzenli olarak temizlendi\b",
                    r"\btemizlik düzenliydi\b",
                    r"\bhousekeeping her gün geldi\b",
                ],
                "ar": [
                    r"\bبينضفوا كل يوم\b",
                    r"\bنظفوا الغرفة بانتظام\b",
                    r"\bفي خدمة تنظيف يومية\b",
                    r"\bكان في هاوسكيبينج ثابت\b",
                ],
                "zh": [
                    r"每天都会打扫",
                    r"房间每天有人来整理",
                    r"清洁很规律",
                    r"入住期间一直有人打扫",
                ],
            },
        ),
        
        
        "trash_taken_out": AspectRule(
            aspect_code="trash_taken_out",
            polarity_hint="positive",
            display="Своевременный вынос мусора",
            display_short="мусор выносили",
            long_hint="Гости отмечают, что мусор из номера своевременно выносился персоналом без необходимости отдельно просить.",
            patterns_by_lang={{
                "ru": [
                    r"\bмусор выносили\b",
                    r"\bпакеты с мусором забирали\b",
                    r"\bмусор забирали каждый день\b",
                    r"\bне скапливался мусор\b",
                ],
                "en": [
                    r"\bthey took out the trash\b",
                    r"\btrash was emptied\b",
                    r"\bgarbage was taken out regularly\b",
                    r"\bthey emptied the bins every day\b",
                ],
                "tr": [
                    r"\bçöp her gün alındı\b",
                    r"\bçöpü düzenli aldılar\b",
                    r"\bçöp bırakılmadı odada\b",
                    r"\bçöp kovaları boşaltıldı\b",
                ],
                "ar": [
                    r"\bكانوا يفضّوا الزبالة بانتظام\b",
                    r"\bشالوا القمامة كل يوم\b",
                    r"\bكل ما نلاقي السلة فاضية\b",
                ],
                "zh": [
                    r"垃圾都会及时清走",
                    r"垃圾桶每天都会倒",
                    r"不会堆垃圾在房间里",
                ],
            },
        ),
        
        
        "bed_made": AspectRule(
            aspect_code="bed_made",
            polarity_hint="positive",
            display="Приведение кровати в порядок во время проживания",
            display_short="заправленная кровать",
            long_hint="Гости отмечают, что при промежуточной уборке кровать была аккуратно заправлена и визуально приведена в порядок.",
            patterns_by_lang={{
                "ru": [
                    r"\bпостель заправляли\b",
                    r"\bкровать приводили в порядок\b",
                    r"\bкаждый день застилали кровать\b",
                    r"\bкровать всегда аккуратная\b",
                ],
                "en": [
                    r"\bthey made the bed\b",
                    r"\bbed was made every day\b",
                    r"\bthe bed was always neatly made\b",
                    r"\bthey fixed the bed for us\b",
                ],
                "tr": [
                    r"\byatağı her gün düzelttiler\b",
                    r"\byatak hep topluydu\b",
                    r"\byatağı güzelce yaptılar\b",
                ],
                "ar": [
                    r"\bكل يوم يرتبوا السرير\b",
                    r"\bالسرير دايماً متوضب\b",
                    r"\bكانوا يرجعوا السرير نظيف ومرتب\b",
                ],
                "zh": [
                    r"每天都会整理床铺",
                    r"床每天都铺好",
                    r"床一直保持整齐",
                ],
            },
        ),
        
        
        "housekeeping_missed": AspectRule(
            aspect_code="housekeeping_missed",
            polarity_hint="negative",
            display="Пропуск запланированной уборки",
            display_short="уборку пропустили",
            long_hint="Гости фиксируют, что регулярная уборка не была выполнена в ожидаемый день или вовсе не проводилась, несмотря на формат проживания и/или договорённость.",
            patterns_by_lang={{
                "ru": [
                    r"\bуборку пропустили\b",
                    r"\bникто не пришел убирать\b",
                    r"\bникто не убрал номер\b",
                    r"\bза все время никто не убрал\b",
                    r"\bуборки так и не было\b",
                ],
                "en": [
                    r"\bno housekeeping\b",
                    r"\bhousekeeping never came\b",
                    r"\broom was never cleaned\b",
                    r"\bthey skipped cleaning\b",
                    r"\bno one came to clean the room\b",
                ],
                "tr": [
                    r"\btemizlik yapılmadı\b",
                    r"\bkimse temizliğe gelmedi\b",
                    r"\boda hiç temizlenmedi\b",
                    r"\btemizliği atladılar\b",
                ],
                "ar": [
                    r"\bما حد نظف الغرفة\b",
                    r"\bما في حد دخل ينضف\b",
                    r"\bما جاش هاوسكيبينج خالص\b",
                    r"\bالتنضيف ما صار\b",
                ],
                "zh": [
                    r"没有人来打扫",
                    r"没人来清洁房间",
                    r"清洁被跳过了",
                    r"整个入住期间没人打扫",
                ],
            },
        ),
        
        
        "trash_not_taken": AspectRule(
            aspect_code="trash_not_taken",
            polarity_hint="negative",
            display="Мусор не выносится во время проживания",
            display_short="мусор не выносят",
            long_hint="Гости сообщают, что мусор из номера не вывозился своевременно: баки оставались полными, пакеты с мусором накапливались внутри номера.",
            patterns_by_lang={{
                "ru": [
                    r"\bмусор не выносили\b",
                    r"\bурны оставались полными\b",
                    r"\bпакеты с мусором накопились\b",
                    r"\bмусор никто не забирал\b",
                ],
                "en": [
                    r"\btrash was not taken out\b",
                    r"\bbins were never emptied\b",
                    r"\bthe garbage piled up\b",
                    r"\bno one emptied the trash\b",
                ],
                "tr": [
                    r"\bçöp alınmadı\b",
                    r"\bçöp dolu kaldı\b",
                    r"\bçöp birikti odada\b",
                    r"\bçöp kovası boşaltılmadı\b",
                ],
                "ar": [
                    r"\bالزبالة ما اتشالت\b",
                    r"\bسلة المهملات فضلت مليانة\b",
                    r"\bالزبالة تجمعت في الأوضة\b",
                ],
                "zh": [
                    r"垃圾没人倒",
                    r"垃圾桶一直是满的",
                    r"垃圾越积越多",
                    r"从来没人来清垃圾",
                ],
            },
        ),
    
        "bed_not_made": AspectRule(
            aspect_code="bed_not_made",
            polarity_hint="negative",
            display="Кровать не была приведена в порядок при промежуточной уборке",
            display_short="кровать не заправили",
            long_hint="Гости фиксируют, что при промежуточной уборке кровать осталась не заправленной или постель выглядит неряшливо, несмотря на визит хаускипинга.",
            patterns_by_lang={{
                "ru": [
                    r"\bкровать не заправили\b",
                    r"\bпостель так и не заправили\b",
                    r"\bникто не застелил кровать\b",
                    r"\bкровать оставили разобранной\b",
                ],
                "en": [
                    r"\bbed was not made\b",
                    r"\bthey didn't make the bed\b",
                    r"\bthe bed was left unmade\b",
                    r"\bhousekeeping didn't fix the bed\b",
                ],
                "tr": [
                    r"\byatak toplanmamıştı\b",
                    r"\byatağı düzeltmediler\b",
                    r"\byatak olduğu gibi bırakıldı\b",
                ],
                "ar": [
                    r"\bما رتبوش السرير\b",
                    r"\bالسرير بقي مفروش زي ما هو\b",
                    r"\bما حد ظبط السرير\b",
                ],
                "zh": [
                    r"床没铺好",
                    r"床还是乱的",
                    r"保洁来也没整理床铺",
                ],
            },
        ),
        
        
        "had_to_request_cleaning": AspectRule(
            aspect_code="had_to_request_cleaning",
            polarity_hint="negative",
            display="Гостям пришлось отдельно запрашивать уборку",
            display_short="уборку пришлось просить",
            long_hint="Гости отмечают, что уборка не выполнялась автоматически и приходилось отдельно запрашивать хаускипинг или напоминать персоналу о необходимости убрать номер.",
            patterns_by_lang={{
                "ru": [
                    r"\bпришлось просить уборку\b",
                    r"\bуборку пришлось отдельно заказывать\b",
                    r"\bнапоминали про уборку\b",
                    r"\bсами попросили убрать номер\b",
                ],
                "en": [
                    r"\bhad to ask for cleaning\b",
                    r"\bhad to request housekeeping\b",
                    r"\bwe had to ask them to clean the room\b",
                    r"\bwe had to remind them to clean\b",
                ],
                "tr": [
                    r"\btemizlik için özellikle istemek zorunda kaldık\b",
                    r"\boda temizliği talep etmek zorunda kaldık\b",
                    r"\bhatırlatmadan temizlemediler\b",
                ],
                "ar": [
                    r"\bاضطرينا نطلب تنظيف\b",
                    r"\bما حد نظف إلا بعد ما طلبنا\b",
                    r"\bقلنالهم ينضفوا الغرفة\b",
                ],
                "zh": [
                    r"要我们自己要求才来打扫",
                    r"必须开口要才有人打扫",
                    r"还得提醒他们来清洁",
                ],
            },
        ),
        
        
        "dirt_accumulated": AspectRule(
            aspect_code="dirt_accumulated",
            polarity_hint="negative",
            display="Нарастающая грязь в номере в период проживания",
            display_short="грязь копится",
            long_hint="Гости фиксируют, что в номере накапливается грязь и пыль: полы, поверхности, санузел становятся заметно грязнее с каждым днём, уборка не поддерживается должным образом.",
            patterns_by_lang={{
                "ru": [
                    r"\bгрязь накапливалась\b",
                    r"\bгрязь копилась\b",
                    r"\bстановилось всё грязнее\b",
                    r"\bникто не поддерживал чистоту\b",
                    r"\bпол становился всё грязнее\b",
                ],
                "en": [
                    r"\bdirt was accumulating\b",
                    r"\bthe room got dirtier each day\b",
                    r"\bkept getting dirtier\b",
                    r"\bno one maintained cleanliness\b",
                    r"\bfloors getting dirty and not cleaned\b",
                ],
                "tr": [
                    r"\bkir birikmeye başladı\b",
                    r"\boda her gün daha kirli oldu\b",
                    r"\btemizlenmediği için kir birikti\b",
                ],
                "ar": [
                    r"\bالأوضة بتزيد وسخ مع الوقت\b",
                    r"\bالوسخ يتراكم\b",
                    r"\bالأرضية بتتوسخ ومحدش بينضف\b",
                ],
                "zh": [
                    r"房间越来越脏",
                    r"灰尘越积越多",
                    r"地面越来越脏也没人清理",
                ],
            },
        ),
        
        
        "towels_changed": AspectRule(
            aspect_code="towels_changed",
            polarity_hint="positive",
            display="Своевременная замена полотенец",
            display_short="полотенца меняли",
            long_hint="Гости отмечают, что полотенца менялись своевременно и по запросу: чистые полотенца были доставлены без задержек, что поддерживало ощущение гигиены.",
            patterns_by_lang={{
                "ru": [
                    r"\bполотенца меняли\b",
                    r"\bчистые полотенца принесли\b",
                    r"\bполотенца регулярно меняли\b",
                    r"\bсвежие полотенца каждый день\b",
                ],
                "en": [
                    r"\bthey changed the towels\b",
                    r"\bfresh towels every day\b",
                    r"\bclean towels were provided\b",
                    r"\bnew towels were brought to us\b",
                ],
                "tr": [
                    r"\bhavlular düzenli değiştirildi\b",
                    r"\bher gün temiz havlu verdiler\b",
                    r"\byeni havlular hemen getirildi\b",
                ],
                "ar": [
                    r"\bكانوا يغيروا الفوط بانتظام\b",
                    r"\bجابوا فوط نضيفة كل يوم\b",
                    r"\bدائماً في مناشف نضيفة\b",
                ],
                "zh": [
                    r"毛巾每天都会换新的",
                    r"有新的干净毛巾",
                    r"及时给我们换毛巾",
                ],
            },
        ),
        
        
        "fresh_towels_fast": AspectRule(
            aspect_code="fresh_towels_fast",
            polarity_hint="positive",
            display="Оперативная доставка свежих полотенец по запросу",
            display_short="свежие полотенца быстро",
            long_hint="Гости отмечают, что при обращении за заменой полотенец свежие комплекты были принесены оперативно, без ожидания и повторных напоминаний.",
            patterns_by_lang={{
                "ru": [
                    r"\bпринесли чистые полотенца сразу\b",
                    r"\bполотенца заменили буквально сразу\b",
                    r"\bполотенца принесли очень быстро\b",
                    r"\bпо запросу сразу привезли свежие полотенца\b",
                ],
                "en": [
                    r"\bthey brought fresh towels right away\b",
                    r"\bnew towels were delivered immediately\b",
                    r"\bfresh towels came quickly\b",
                    r"\bwe asked for towels and got them right away\b",
                ],
                "tr": [
                    r"\btemiz havlular hemen geldi\b",
                    r"\bistedikten hemen sonra havlu getirdiler\b",
                    r"\byeni havlular anında verildi\b",
                ],
                "ar": [
                    r"\bجابوا مناشف نضيفة فوراً\b",
                    r"\bطلبنا مناشف وجابوها على طول\b",
                    r"\bغيروا الفوط بسرعة جداً\b",
                ],
                "zh": [
                    r"一说要毛巾就马上送来",
                    r"干净的毛巾立刻拿来了",
                    r"新毛巾很快就送到",
                ],
            },
        ),

        "linen_changed": AspectRule(
            aspect_code="linen_changed",
            polarity_hint="positive",
            display="Своевременная смена постельного белья",
            display_short="бельё меняли",
            long_hint="Гости отмечают, что постельное бельё менялось в ходе проживания: простыни и наволочки заменялись на чистые без необходимости настаивать.",
            patterns_by_lang={{
                "ru": [
                    r"\bсменили постельное бель[её]\b",
                    r"\bпоменяли простыни\b",
                    r"\bсвежие простыни\b",
                    r"\bчистое бель[её] принесли\b",
                    r"\bпоменяли наволочки\b",
                ],
                "en": [
                    r"\bthey changed the sheets\b",
                    r"\bfresh sheets\b",
                    r"\bclean bedding was provided\b",
                    r"\bnew linen during our stay\b",
                    r"\bthey replaced the bed linen\b",
                ],
                "tr": [
                    r"\bçarşaflar değiştirildi\b",
                    r"\btemiz nevresim getirdiler\b",
                    r"\byatak çarşaflarını yenilediler\b",
                    r"\btemiz çarşaf verdiler\b",
                ],
                "ar": [
                    r"\bغيروا الملايات\b",
                    r"\bجابوا شراشف نضيفة\b",
                    r"\bالملايات اتبدلت أثناء الإقامة\b",
                    r"\bغيروا كيس المخدة\b",
                ],
                "zh": [
                    r"床单换新的了",
                    r"给我们换了干净的床品",
                    r"住宿期间有更换床单",
                    r"换了新的枕套",
                ],
            },
        ),
        
        
        "amenities_restocked": AspectRule(
            aspect_code="amenities_restocked",
            polarity_hint="positive",
            display="Пополнение расходников и комнатных принадлежностей",
            display_short="расходники пополняли",
            long_hint="Гости отмечают, что расходные материалы (вода, туалетная бумага, шампунь, кофе/чай и пр.) регулярно пополнялись без напоминаний, что снижало операционные трения во время проживания.",
            patterns_by_lang={{
                "ru": [
                    r"\bвсё пополняли\b",
                    r"\bпополняли расходники\b",
                    r"\bпополняли воду и кофе\b",
                    r"\bтуалетную бумагу приносили\b",
                    r"\bсредства для душа обновляли\b",
                ],
                "en": [
                    r"\bthey restocked everything\b",
                    r"\brestocked toiletries\b",
                    r"\bwater and coffee were refilled\b",
                    r"\breplenished supplies\b",
                    r"\btoilet paper was restocked\b",
                ],
                "tr": [
                    r"\bikramları yenilediler\b",
                    r"\btüm malzemeler yenilendi\b",
                    r"\btuvalet kağıdı ve şampuan yenilendi\b",
                    r"\bsu ve kahve tekrar konuldu\b",
                ],
                "ar": [
                    r"\bكانوا يعيدوا يعبّوا كل شي\b",
                    r"\bزودونا بشامبو ومياه تاني\b",
                    r"\bجددوا المستلزمات\b",
                    r"\bحطوا مناديل تواليت زيادة\b",
                ],
                "zh": [
                    r"补充了所有用品",
                    r"洗漱用品有补",
                    r"水和咖啡都会补上",
                    r"卫生纸会及时补充",
                ],
            },
        ),
        
        
        "towels_dirty": AspectRule(
            aspect_code="towels_dirty",
            polarity_hint="negative",
            display="Полотенца выданы в неидеальном состоянии",
            display_short="полотенца не идеально чистые",
            long_hint="Гости фиксируют, что выданные полотенца выглядели не до конца чистыми: ощущались старыми, визуально не свежими, с общей серостью или следами прежнего использования.",
            patterns_by_lang={{
                "ru": [
                    r"\bгрязные полотенца\b",
                    r"\bполотенца были не свежие\b",
                    r"\bполотенца серые\b",
                    r"\bполотенца выглядят старыми\b",
                    r"\bне очень чистые полотенца\b",
                ],
                "en": [
                    r"\bdirty towels\b",
                    r"\btowels were not clean\b",
                    r"\btowels looked used\b",
                    r"\btowels looked old and grey\b",
                    r"\bnot fresh towels\b",
                ],
                "tr": [
                    r"\bkirli havlular\b",
                    r"\bhavlular temiz değildi\b",
                    r"\bçok eski ve gri havlular\b",
                    r"\bkullanılmış gibi duran havlu\b",
                ],
                "ar": [
                    r"\bالفوط وسخة\b",
                    r"\bالفوط مش باينة نضيفة\b",
                    r"\bالفوطة باينة قديمة\b",
                    r"\bمش مناشف فريش\b",
                ],
                "zh": [
                    r"毛巾不干净",
                    r"毛巾看起来旧旧的",
                    r"毛巾像用过的一样",
                    r"毛巾不是很干净",
                ],
            },
        ),
        
        
        "towels_stained": AspectRule(
            aspect_code="towels_stained",
            polarity_hint="negative",
            display="Пятна на полотенцах",
            display_short="пятна на полотенцах",
            long_hint="Гости отмечают наличие пятен на выданных полотенцах (косметика, грязевые следы и т.д.), что создаёт ощущение низкого гигиенического стандарта.",
            patterns_by_lang={{
                "ru": [
                    r"\bпятна на полотенцах\b",
                    r"\bгрязные пятна на полотенце\b",
                    r"\bполотенца в пятнах\b",
                    r"\bполотенце с разводами\b",
                ],
                "en": [
                    r"\bstained towels\b",
                    r"\btowels had stains\b",
                    r"\bmarks on the towels\b",
                    r"\bdirty marks on the towel\b",
                ],
                "tr": [
                    r"\blekeli havlu\b",
                    r"\bhavlularda lekeler vardı\b",
                    r"\bhavluda izler vardı\b",
                ],
                "ar": [
                    r"\bالفوط عليها بقع\b",
                    r"\bفيها آثار مكياج\b",
                    r"\bفوطة مبقعة\b",
                    r"\bالمناشف مليانة بقع\b",
                ],
                "zh": [
                    r"毛巾上有污渍",
                    r"毛巾有明显的痕迹",
                    r"毛巾上有斑点",
                    r"毛巾不是纯白的有印子",
                ],
            },
        ),
        
        
        "towels_smell": AspectRule(
            aspect_code="towels_smell",
            polarity_hint="negative",
            display="Неприятный запах от полотенец",
            display_short="запах полотенец",
            long_hint="Гости сообщают, что полотенца имели неприятный запах (сырость, затхлость, несвежесть), что воспринимается как недостаток санитарной подготовки.",
            patterns_by_lang={{
                "ru": [
                    r"\bполотенца плохо пахли\b",
                    r"\bполотенца с запахом сырости\b",
                    r"\bзатхлый запах от полотенец\b",
                    r"\bполотенца воняли\b",
                ],
                "en": [
                    r"\bthe towels smelled bad\b",
                    r"\bmusty towels\b",
                    r"\bdamp smell from the towels\b",
                    r"\bthe towels had a weird smell\b",
                ],
                "tr": [
                    r"\bhavlular kötü kokuyordu\b",
                    r"\bnem kokusu vardı havlularda\b",
                    r"\bkokan havlular\b",
                    r"\bmayt gibi kokuyordu\b",
                ],
                "ar": [
                    r"\bالفوط ريحتها وحشة\b",
                    r"\bريحة رطوبة في الفوطة\b",
                    r"\bريحة مش نظيفة في المناشف\b",
                    r"\bالمناشف ريحتها معفنة شوية\b",
                ],
                "zh": [
                    r"毛巾有霉味",
                    r"毛巾有一股潮味",
                    r"毛巾味道很重",
                    r"毛巾闻起来不新鲜",
                ],
            },
        ),

        "towels_not_changed": AspectRule(
            aspect_code="towels_not_changed",
            polarity_hint="negative",
            display="Отсутствие замены полотенец",
            display_short="полотенца не меняли",
            long_hint="Гости фиксируют, что во время проживания полотенца не менялись, даже после просьбы о замене она не была выполнена.",
            patterns_by_lang={{
                "ru": [
                    r"\bполотенца не меняли\b",
                    r"\bникто не поменял полотенца\b",
                    r"\bполотенца так и не заменили\b",
                    r"\bпришлось пользоваться старыми полотенцами\b",
                ],
                "en": [
                    r"\btowels were not changed\b",
                    r"\bno one changed the towels\b",
                    r"\btowels never got replaced\b",
                    r"\bhad to use the same towels\b",
                ],
                "tr": [
                    r"\bhavlular değiştirilmedi\b",
                    r"\bkimse havluları yenilemedi\b",
                    r"\bhep aynı havluları kullandık\b",
                ],
                "ar": [
                    r"\bما بدلوش الفوط\b",
                    r"\bنفس الفوط طول الإقامة\b",
                    r"\bما حد غير المناشف\b",
                ],
                "zh": [
                    r"毛巾没有更换",
                    r"一直都是同一条毛巾",
                    r"没人来换毛巾",
                ],
            },
        ),
        
        
        "linen_not_changed": AspectRule(
            aspect_code="linen_not_changed",
            polarity_hint="negative",
            display="Отсутствие смены постельного белья",
            display_short="бельё не меняли",
            long_hint="Гости сообщают, что постельное бельё не менялось во время проживания, несмотря на продолжительность размещения или прямой запрос на замену.",
            patterns_by_lang={{
                "ru": [
                    r"\bбель[её] не меняли\b",
                    r"\bпостельное бель[её] так и не сменили\b",
                    r"\bпростыни не поменяли\b",
                    r"\bмы спали на тех же простынях\b",
                ],
                "en": [
                    r"\bthe sheets were not changed\b",
                    r"\bno one changed the bedding\b",
                    r"\bthey never replaced the linen\b",
                    r"\bwe had the same sheets the whole stay\b",
                ],
                "tr": [
                    r"\bçarşaflar değiştirilmedi\b",
                    r"\byatak çarşafları hiç yenilenmedi\b",
                    r"\baynı çarşaflarla kaldık\b",
                ],
                "ar": [
                    r"\bالملايات ما اتغيرتش\b",
                    r"\bنفس الملايات طول الاقامة\b",
                    r"\bما حدش بدّل الشراشف\b",
                ],
                "zh": [
                    r"床单一直没换",
                    r"床上用品没有更换",
                    r"整段入住都用同一套床单",
                ],
            },
        ),
        
        
        "no_restock": AspectRule(
            aspect_code="no_restock",
            polarity_hint="negative",
            display="Расходники не пополнялись",
            display_short="не пополняли расходники",
            long_hint="Гости фиксируют, что базовые расходные материалы (туалетная бумага, мыло, шампунь, вода, кофе/чай) не пополнялись автоматически и приходилось просить дополнительно.",
            patterns_by_lang={{
                "ru": [
                    r"\bничего не пополняли\b",
                    r"\bтуалетную бумагу не пополняли\b",
                    r"\bводу не пополнили\b",
                    r"\bшампунь не донесли\b",
                    r"\bпришлось просить расходники\b",
                ],
                "en": [
                    r"\bnothing was restocked\b",
                    r"\bthey didn't restock supplies\b",
                    r"\bno refill of toiletries\b",
                    r"\bno new toilet paper\b",
                    r"\bhad to ask for more water/coffee\b",
                ],
                "tr": [
                    r"\bmalzemeler yenilenmedi\b",
                    r"\btuvalet kağıdı yenilenmedi\b",
                    r"\bsu/kahve takviye edilmedi\b",
                    r"\bşampuan koymadılar tekrar\b",
                ],
                "ar": [
                    r"\bما عبّوش الحاجات الأساسية\b",
                    r"\bما زودوش مناديل التواليت\b",
                    r"\bما زودوش شامبو ولا ميّة\b",
                    r"\bاضطرينا نطلب علشان يجيبوا زيادة\b",
                ],
                "zh": [
                    r"没有补日用品",
                    r"纸巾/厕纸都没补",
                    r"没补水和咖啡",
                    r"洗漱用品没再补充",
                ],
            },
        ),
        
        
        "smell_of_smoke": AspectRule(
            aspect_code="smell_of_smoke",
            polarity_hint="negative",
            display="Запах табачного дыма в номере",
            display_short="запах сигаретного дыма",
            long_hint="Гости фиксируют запах сигаретного дыма или застоявшегося табака в апартаментах на момент проживания.",
            patterns_by_lang={{
                "ru": [
                    r"\bзапах сигарет\b",
                    r"\bпахло табаком\b",
                    r"\bпахнет куревом\b",
                    r"\bвонь сигаретного дыма\b",
                    r"\bв номере пахло дымом\b",
                ],
                "en": [
                    r"\bsmelled like smoke\b",
                    r"\bcigarette smell\b",
                    r"\bsmell of cigarettes in the room\b",
                    r"\broom smelled of smoke\b",
                    r"\bsmelled like someone had been smoking\b",
                ],
                "tr": [
                    r"\bsigara kokuyordu\b",
                    r"\bodada sigara kokusu vardı\b",
                    r"\btütün kokusu vardı\b",
                    r"\bokulmuş sigara kokusu\b",
                ],
                "ar": [
                    r"\bريحة سجاير في الأوضة\b",
                    r"\bريحة دخان\b",
                    r"\bالأوضة ريحتها تدخين\b",
                    r"\bريحة تبغ قديم\b",
                ],
                "zh": [
                    r"房间里有烟味",
                    r"有很重的香烟味",
                    r"像有人在房间里抽过烟",
                    r"有烟草味道",
                ],
            },
        ),
        
        
        "chemical_smell_strong": AspectRule(
            aspect_code="chemical_smell_strong",
            polarity_hint="negative",
            display="Резкий запах бытовой химии / очистителей",
            display_short="запах химии",
            long_hint="Гости сообщают о слишком резком запахе чистящих или дезинфицирующих средств, который сохраняется в номере или санузле и воспринимается как дискомфорт.",
            patterns_by_lang={{
                "ru": [
                    r"\bсильный запах химии\b",
                    r"\bвоняло хлоркой\b",
                    r"\bзапах чистящих средств\b",
                    r"\bзапах бытовой химии\b",
                    r"\bзапах дезинфектанта\b",
                ],
                "en": [
                    r"\bstrong chemical smell\b",
                    r"\bsmelled like bleach\b",
                    r"\bsmelled like cleaning chemicals\b",
                    r"\bstrong bleach smell in the room\b",
                    r"\bchemical odor in the bathroom\b",
                ],
                "tr": [
                    r"\bçok ağır çamaşır suyu kokusu vardı\b",
                    r"\bokulmuş deterjan kokusu\b",
                    r"\bkimyasal kokusu çok yoğundu\b",
                    r"\boda deterjan gibi kokuyordu\b",
                ],
                "ar": [
                    r"\bريحة كلور قوية\b",
                    r"\bريحة منظفات قوية جداً\b",
                    r"\bريحة منظف خانقة في الأوضة\b",
                    r"\bريحة كيمياويات قوية في الحمام\b",
                ],
                "zh": [
                    r"有很重的消毒水味",
                    r"闻起来像漂白水",
                    r"清洁剂味道太重了",
                    r"房间里都是化学清洁味",
                ],
            },
        ),

        "fresh_smell": AspectRule(
            aspect_code="fresh_smell",
            polarity_hint="positive",
            display="Свежий запах в номере",
            display_short="свежий запах",
            long_hint="Гости отмечают ощущение свежего воздуха и приятного запаха в номере или санузле.",
            patterns_by_lang={{
                "ru": [
                    r"\bсвежий запах\b",
                    r"\bприятно пахло\b",
                    r"\bничем плохо не пахло\b",
                    r"\bв номере пахло чисто\b",
                    r"\bпахло свежестью\b",
                ],
                "en": [
                    r"\bfresh smell\b",
                    r"\broom smelled fresh\b",
                    r"\bthe room smelled clean\b",
                    r"\bno bad smells\b",
                    r"\bno odor in the room\b",
                ],
                "tr": [
                    r"\boda ferah kokuyordu\b",
                    r"\btemiz kokuyordu\b",
                    r"\bhiç kötü koku yoktu\b",
                    r"\bodada taze bir koku vardı\b",
                ],
                "ar": [
                    r"\bريحة نظافة\b",
                    r"\bريحة منعشة في الأوضة\b",
                    r"\bمفيش ريحة وحشة\b",
                    r"\bريحة كويسة في الأوضة\b",
                ],
                "zh": [
                    r"房间味道很清新",
                    r"房间有清新的味道",
                    r"没有异味",
                    r"闻起来很干净",
                ],
            },
        ),
        
        
        "hallway_clean": AspectRule(
            aspect_code="hallway_clean",
            polarity_hint="positive",
            display="Чистота общих зон (коридор, лифтовой холл)",
            display_short="чистый коридор",
            long_hint="Гости отмечают, что коридоры, лифтовые холлы и входные группы содержатся в чистоте: нет мусора, нет пыли, визуально поддерживается порядок.",
            patterns_by_lang={{
                "ru": [
                    r"\bчистый коридор\b",
                    r"\bчисто в коридорах\b",
                    r"\bчистый подъезд\b",
                    r"\bаккуратный холл\b",
                    r"\bна этаже чисто\b",
                ],
                "en": [
                    r"\bclean hallway\b",
                    r"\bthe hallways were clean\b",
                    r"\bcommon areas were clean\b",
                    r"\bthe entrance area was tidy\b",
                ],
                "tr": [
                    r"\bkoridor çok temizdi\b",
                    r"\bortak alanlar temizdi\b",
                    r"\bgiriş kısmı düzgündü\b",
                    r"\bkat koridoru pırıl pırıldı\b",
                ],
                "ar": [
                    r"\bالممر نضيف\b",
                    r"\bالمداخل نظيفة\b",
                    r"\bالمدخل كان مرتب\b",
                    r"\bالمناطق المشتركة نضيفة\b",
                ],
                "zh": [
                    r"走廊很干净",
                    r"公共区域很整洁",
                    r"楼道很干净",
                    r"入口区域很整齐",
                ],
            },
        ),
        
        
        "common_areas_clean": AspectRule(
            aspect_code="common_areas_clean",
            polarity_hint="positive",
            display="Состояние общих зон для гостей",
            display_short="общие зоны чистые",
            long_hint="Гости отмечают, что общие пространства поддерживаются в порядке и визуально выглядят ухоженно.",
            patterns_by_lang={{
                "ru": [
                    r"\bобщие зоны чистые\b",
                    r"\bчисто в общих зонах\b",
                    r"\bлаундж был чистый\b",
                    r"\bкухня общего пользования чистая\b",
                    r"\bзона общего пользования ухоженная\b",
                ],
                "en": [
                    r"\bcommon areas were clean\b",
                    r"\bshared areas were very clean\b",
                    r"\bthe lobby was clean\b",
                    r"\bthe shared kitchen was clean\b",
                    r"\bthe breakfast area was kept tidy\b",
                ],
                "tr": [
                    r"\bortak alanlar temizdi\b",
                    r"\blobi çok temizdi\b",
                    r"\bpaylaşılan mutfak temizdi\b",
                    r"\bortak kullanım alanları düzenliydi\b",
                ],
                "ar": [
                    r"\bالمناطق المشتركة كانت نضيفة\b",
                    r"\bاللوبي كان نضيف\b",
                    r"\bالمطبخ المشترك كان نضيف\b",
                    r"\bالمنطقة المشتركة مرتبة\b",
                ],
                "zh": [
                    r"公共区域很干净",
                    r"大堂很干净",
                    r"公共厨房很整洁",
                    r"早餐区保持得很干净",
                ],
            },
        ),
        
        
        "hallway_dirty": AspectRule(
            aspect_code="hallway_dirty",
            polarity_hint="negative",
            display="Грязь и запущенность в коридорах/общих зонах",
            display_short="грязный коридор",
            long_hint="Гости фиксируют, что в коридорах и общих пространствах наблюдаются грязь, пыль, мусор, следы длительного отсутствия уборки.",
            patterns_by_lang={{
                "ru": [
                    r"\bгрязный коридор\b",
                    r"\bгрязно в коридоре\b",
                    r"\bгрязный подъезд\b",
                    r"\bсо ступенек не убирают\b",
                    r"\bмусор в коридоре\b",
                    r"\bгрязные общие зоны\b",
                ],
                "en": [
                    r"\bdirty hallway\b",
                    r"\bthe hallway was dirty\b",
                    r"\bcommon areas were dirty\b",
                    r"\btrash in the hallway\b",
                    r"\bthe entrance was dirty\b",
                ],
                "tr": [
                    r"\bkoridor kirliydi\b",
                    r"\bortak alanlar kirliydi\b",
                    r"\bgiriş çok pisti\b",
                    r"\bmerdivenlerde çöp vardı\b",
                ],
                "ar": [
                    r"\bالممر كان وسخ\b",
                    r"\bالمدخل وسخ\b",
                    r"\bفيه زبالة بالممر\b",
                    r"\bالمناطق المشتركة مش نضيفة\b",
                ],
                "zh": [
                    r"走廊很脏",
                    r"楼道很脏",
                    r"公共区域很脏",
                    r"走廊里有垃圾",
                ],
            },
        ),
        
        
        "elevator_dirty": AspectRule(
            aspect_code="elevator_dirty",
            polarity_hint="negative",
            display="Состояние лифта и лифтового холла (грязно)",
            display_short="грязный лифт",
            long_hint="Гости отмечают, что лифт и зона вокруг него выглядят неухоженно: пятна, запах, грязный пол, мусор, следы интенсивного использования без уборки.",
            patterns_by_lang={{
                "ru": [
                    r"\bгрязный лифт\b",
                    r"\bлифт грязный\b",
                    r"\bпол в лифте грязный\b",
                    r"\bнечиcтый лифтовый холл\b",
                    r"\bпахнет неприятно в лифте\b",
                ],
                "en": [
                    r"\bdirty elevator\b",
                    r"\bthe elevator was dirty\b",
                    r"\bthe lift smelled bad\b",
                    r"\bdirty elevator floor\b",
                    r"\bthe elevator area was filthy\b",
                ],
                "tr": [
                    r"\basansör kirliydi\b",
                    r"\basansörde kötü koku vardı\b",
                    r"\basansörün zemini pisti\b",
                    r"\basansör önü çok pisti\b",
                ],
                "ar": [
                    r"\bالأسانسير وسخ\b",
                    r"\bالأسنصير ريحته وحشة\b",
                    r"\bالأسانسير كان مش نضيف\b",
                    r"\bقدّام الأسانسير وسخ\b",
                ],
                "zh": [
                    r"电梯很脏",
                    r"电梯口很脏",
                    r"电梯里面有异味",
                    r"电梯地板很脏",
                ],
            },
        ),

        "hallway_bad_smell": AspectRule(
            aspect_code="hallway_bad_smell",
            polarity_hint="negative",
            display="Неприятный запах в общих зонах",
            display_short="запах в коридоре",
            long_hint="Гости фиксируют наличие неприятного запаха в коридорах, лифтовом холле или у входа, что влияет на общее восприятие объекта до входа в номер.",
            patterns_by_lang={{
                "ru": [
                    r"\bпахло неприятно в коридоре\b",
                    r"\bпахнет в подъезде\b",
                    r"\bвонь в коридоре\b",
                    r"\bдурно пахло в лифте\b",
                    r"\bстрашный запах у входа\b",
                    r"\bвоняло мусором\b",
                ],
                "en": [
                    r"\bbad smell in the hallway\b",
                    r"\bthe hallway smelled bad\b",
                    r"\bsmelled awful in the corridor\b",
                    r"\bsmelled like garbage in the hallway\b",
                    r"\bbad odor near the elevator\b",
                ],
                "tr": [
                    r"\bkoridorda kötü koku vardı\b",
                    r"\basansörün orası çok kötü kokuyordu\b",
                    r"\bgiriş kısmı kötü kokuyordu\b",
                    r"\bçöp kokusu vardı\b",
                ],
                "ar": [
                    r"\bريحة وحشة في الممر\b",
                    r"\bريحة مش لطيفة عند الأسانسير\b",
                    r"\bبتشم ريحة زبالة في الممر\b",
                    r"\bريحة وحشة أول ما تدخل\b",
                ],
                "zh": [
                    r"走廊有股怪味",
                    r"电梯口有臭味",
                    r"楼道味道很难闻",
                    r"一进门就有异味",
                ],
            },
        ),
        
        
        "entrance_feels_unsafe": AspectRule(
            aspect_code="entrance_feels_unsafe",
            polarity_hint="negative",
            display="Субъективное ощущение небезопасного входа/подъезда",
            display_short="небезопасный вход",
            long_hint="Гости сообщают, что при входе в здание или в подъезд чувствовали себя некомфортно из-за состояния входной группы (грязно, темно, посторонние люди) или атмосферы района. Это влияет как на восприятие чистоты общих зон, так и на ощущение личной безопасности.",
            patterns_by_lang={{
                "ru": [
                    r"\bнеприятно заходить в подъезд\b",
                    r"\bу входа было страшновато\b",
                    r"\bне чувствовали себя в безопасности\b",
                    r"\bподъезд страшный\b",
                    r"\bу двери какие-то подозрительные люди\b",
                    r"\bу входа грязно и страшно\b",
                ],
                "en": [
                    r"\bthe entrance felt unsafe\b",
                    r"\bdidn't feel safe entering the building\b",
                    r"\bsketchy entrance\b",
                    r"\bpeople hanging around the entrance\b",
                    r"\bthe hallway felt a bit scary\b",
                ],
                "tr": [
                    r"\bgiriş pek güvenli hissettirmedi\b",
                    r"\bapartman girişi tedirgin ediciydi\b",
                    r"\bkapıda tuhaf tipler vardı\b",
                    r"\bgece giriş yapmak rahat hissettirmedi\b",
                ],
                "ar": [
                    r"\bالمدخل مش مريح ومش آمن\b",
                    r"\bما حسّيناش بأمان وإحنا داخلين\b",
                    r"\bالمدخل شكله يخوّف شوية\b",
                    r"\bفيه ناس واقفين تحت العمارة شكلهم مريب\b",
                ],
                "zh": [
                    r"楼道入口让人不太安心",
                    r"进楼的时候感觉不太安全",
                    r"门口有人让人不舒服",
                    r"入口感觉有点吓人",
                ],
            },
        ),
        
        
        "room_well_equipped": AspectRule(
            aspect_code="room_well_equipped",
            polarity_hint="positive",
            display="Оснащённость номера базовыми удобствами",
            display_short="номер хорошо укомплектован",
            long_hint="Гости отмечают, что номер оснащён всем необходимым для комфортного проживания: чайник, холодильник, фен, розетки, рабочее место, место для багажа и т.д. Восприятие номера как продуманного и функционального.",
            patterns_by_lang={{
                "ru": [
                    r"\bв номере есть всё необходимое\b",
                    r"\bномер хорошо укомплектован\b",
                    r"\bвсё предусмотрено\b",
                    r"\bвсё что нужно было на месте\b",
                    r"\bочень оснащённый номер\b",
                ],
                "en": [
                    r"\bwell equipped room\b",
                    r"\bthe room had everything we needed\b",
                    r"\bthe room was fully equipped\b",
                    r"\bvery well equipped\b",
                    r"\beverything was provided in the room\b",
                ],
                "tr": [
                    r"\bodada her şey vardı\b",
                    r"\bodada ihtiyaç duyulan her şey mevcuttu\b",
                    r"\boda iyi donanımlıydı\b",
                    r"\btüm imkanlar vardı\b",
                ],
                "ar": [
                    r"\bالأوضة فيها كل حاجة محتاجينها\b",
                    r"\bالأوضة مجهزة بكل الأساسيات\b",
                    r"\bكل شي متوفر في الأوضة\b",
                    r"\bالأوضة مجهزة كويس\b",
                ],
                "zh": [
                    r"房间设备很齐全",
                    r"房间里需要的都有",
                    r"配备得很全",
                    r"房间很完善很方便",
                ],
            },
        ),
        
        
        "kettle_available": AspectRule(
            aspect_code="kettle_available",
            polarity_hint="positive",
            display="Наличие чайника/возможности приготовить горячую воду",
            display_short="чайник есть",
            long_hint="Гости отмечают наличие в номере электрочайника или устройства для нагрева воды, что упрощает самостоятельное приготовление чая/кофе и воспринимается как важный бытовой комфорт.",
            patterns_by_lang={{
                "ru": [
                    r"\bбыл чайник\b",
                    r"\bв номере есть чайник\b",
                    r"\bможно вскипятить воду\b",
                    r"\bчайник в номере очень выручал\b",
                    r"\bбыл электрочайник\b",
                ],
                "en": [
                    r"\bthere was a kettle\b",
                    r"\bthe room had a kettle\b",
                    r"\bwe had a kettle in the room\b",
                    r"\btea kettle provided\b",
                    r"\bhot water kettle in the room\b",
                ],
                "tr": [
                    r"\bodada su ısıtıcısı vardı\b",
                    r"\bketıl vardı\b",
                    r"\bçay demlemek için kettle vardı\b",
                    r"\bsu ısıtıcı çok işimize yaradı\b",
                ],
                "ar": [
                    r"\bفي غلاية ميّة في الأوضة\b",
                    r"\bقدرنا نغلي ميّة بالغرفة\b",
                    r"\bفي كاتل بالغرفة\b",
                    r"\bمجهزين غلاية للشاي/قهوة\b",
                ],
                "zh": [
                    r"房间有烧水壶",
                    r"有热水壶可以烧水",
                    r"可以自己烧热水泡茶",
                    r"有电热壶",
                ],
            },
        ),
        
        
        "fridge_available": AspectRule(
            aspect_code="fridge_available",
            polarity_hint="positive",
            display="Наличие холодильника в номере",
            display_short="холодильник есть",
            long_hint="Гости отмечают, что в номере есть холодильник/мини-холодильник, что позволяет хранить продукты, напитки, детское питание и лекарственные препараты с требованиями к температуре.",
            patterns_by_lang={{
                "ru": [
                    r"\bбыл холодильник\b",
                    r"\bв номере есть холодильник\b",
                    r"\bмини-холодильник в номере\b",
                    r"\bможно было охладить напитки\b",
                    r"\bудобно что есть холодильник\b",
                ],
                "en": [
                    r"\bthere was a fridge\b",
                    r"\bthe room had a fridge\b",
                    r"\bmini fridge in the room\b",
                    r"\buseful to have a fridge\b",
                    r"\bwe could keep our drinks cold\b",
                ],
                "tr": [
                    r"\bodada buzdolabı vardı\b",
                    r"\bmini buzdolabı vardı\b",
                    r"\biçecekleri soğuk tutabildik\b",
                    r"\bbuzdolabı olması çok iyiydi\b",
                ],
                "ar": [
                    r"\bفيه تلاجة صغيرة في الأوضة\b",
                    r"\bالتلاجة كانت مفيدة\b",
                    r"\bقدرنا نبرّد المشاريب\b",
                    r"\bكان في ميني بار شغال\b",
                ],
                "zh": [
                    r"房间里有小冰箱",
                    r"有冰箱可以放饮料",
                    r"冰箱很好用",
                    r"可以自己冷藏东西",
                ],
            },
        ),

        "hairdryer_available": AspectRule(
            aspect_code="hairdryer_available",
            polarity_hint="positive",
            display="Наличие фена в номере",
            display_short="фен есть",
            long_hint="Гости отмечают, что в номере есть работающий фен, что воспринимается как элемент базового комфорта и соответствует ожиданиям стандартной комплектации.",
            patterns_by_lang={{
                "ru": [
                    r"\bбыл фен\b",
                    r"\bфен в номере\b",
                    r"\bфен работал\b",
                    r"\bудобно что есть фен\b",
                ],
                "en": [
                    r"\bthere was a hairdryer\b",
                    r"\bhair dryer in the room\b",
                    r"\bthe hairdryer worked\b",
                    r"\bnice to have a hairdryer\b",
                ],
                "tr": [
                    r"\bodada saç kurutma makinesi vardı\b",
                    r"\bsaç kurutma makinesi çalışıyordu\b",
                    r"\bfön makinesi vardı\b",
                ],
                "ar": [
                    r"\bكان فيه سيشوار في الأوضة\b",
                    r"\bالسيشوار شغال\b",
                    r"\bموجود مجفف شعر في الغرفة\b",
                ],
                "zh": [
                    r"房间里有吹风机",
                    r"有可以用的吹风机",
                    r"吹风机是好的",
                ],
            },
        ),
        
        
        "sockets_enough": AspectRule(
            aspect_code="sockets_enough",
            polarity_hint="positive",
            display="Достаточное количество розеток",
            display_short="розеток достаточно",
            long_hint="Гости отмечают, что в номере достаточно доступных розеток для зарядки телефонов, ноутбуков и других устройств, в том числе у кровати и рабочего места.",
            patterns_by_lang={{
                "ru": [
                    r"\bхватало розеток\b",
                    r"\bмного розеток\b",
                    r"\bрозетки рядом с кроватью\b",
                    r"\bрозетки удобно расположены\b",
                ],
                "en": [
                    r"\benough sockets\b",
                    r"\bplenty of outlets\b",
                    r"\bpower outlets by the bed\b",
                    r"\bconveniently placed outlets\b",
                ],
                "tr": [
                    r"\byeterince priz vardı\b",
                    r"\bprizler yatak yanında vardı\b",
                    r"\bpriz sayısı yeterliydi\b",
                ],
                "ar": [
                    r"\bفيه مقابس كهربا كفاية\b",
                    r"\bفيه فيش جنب السرير\b",
                    r"\bالفيش موزعة كويس\b",
                ],
                "zh": [
                    r"插座够用",
                    r"床边有插座",
                    r"插座位置很方便",
                    r"有很多电源插口",
                ],
            },
        ),
        
        
        "workspace_available": AspectRule(
            aspect_code="workspace_available",
            polarity_hint="positive",
            display="Рабочая зона в номере",
            display_short="рабочее место есть",
            long_hint="Гости отмечают наличие рабочего места, что делает проживание удобным для удалённой работы и деловых поездок.",
            patterns_by_lang={{
                "ru": [
                    r"\bбыл рабочий стол\b",
                    r"\bесть место поработать\b",
                    r"\bудобный стол для ноутбука\b",
                    r"\bесть письменный стол\b",
                ],
                "en": [
                    r"\bthere was a desk\b",
                    r"\bworkspace in the room\b",
                    r"\ba proper desk to work\b",
                    r"\bgood desk for laptop work\b",
                ],
                "tr": [
                    r"\bodada çalışma masası vardı\b",
                    r"\bçalışmak için masa vardı\b",
                    r"\bliş için uygun masa ve sandalye\b",
                ],
                "ar": [
                    r"\bفي مكتب نقدر نشتغل عليه\b",
                    r"\bفيه ترابيزة شغل في الأوضة\b",
                    r"\bفي مساحة كويسة للابتوب\b",
                ],
                "zh": [
                    r"房间里有书桌可以办公",
                    r"有可以用电脑的桌子",
                    r"有办公区/工作台",
                ],
            },
        ),
        
        
        "luggage_space_ok": AspectRule(
            aspect_code="luggage_space_ok",
            polarity_hint="positive",
            display="Достаточно места для багажа",
            display_short="место под багаж есть",
            long_hint="Гости отмечают, что в номере предусмотрено место для чемоданов и одежды: можно разложить багаж, не загромождая проход, есть зона для хранения.",
            patterns_by_lang={{
                "ru": [
                    r"\bместа для чемоданов хватало\b",
                    r"\bбыло куда поставить багаж\b",
                    r"\bесть место для чемодана\b",
                    r"\bудобное место для багажа\b",
                ],
                "en": [
                    r"\benough space for luggage\b",
                    r"\bspace for suitcases\b",
                    r"\broom for our bags\b",
                    r"\bnowhere felt cramped with luggage\b",
                ],
                "tr": [
                    r"\bvaliz koyacak yeterli alan vardı\b",
                    r"\bçantalar için yer vardı\b",
                    r"\bodada bagaj için yer vardı\b",
                ],
                "ar": [
                    r"\bفيه مساحة للشنط\b",
                    r"\bكان في مكان نحط الشنط من غير ما نعطل الممر\b",
                    r"\bفيه مساحة للحقائب\b",
                ],
                "zh": [
                    r"房间里有地方放行李",
                    r"行李有足够的空间",
                    r"可以把箱子打开放着",
                ],
            },
        ),
        
        
        "kettle_missing": AspectRule(
            aspect_code="kettle_missing",
            polarity_hint="negative",
            display="Отсутствие чайника при ожидании его наличия",
            display_short="нет чайника",
            long_hint="Гости фиксируют, что чайник в номере отсутствует, хотя они рассчитывали на возможность приготовить чай/кофе или подогреть воду.",
            patterns_by_lang={{
                "ru": [
                    r"\bне было чайника\b",
                    r"\bчайника не оказалось\b",
                    r"\bнет чайника в номере\b",
                    r"\bнельзя вскипятить воду\b",
                ],
                "en": [
                    r"\bno kettle\b",
                    r"\bthere was no kettle in the room\b",
                    r"\bno way to boil water\b",
                    r"\bwe were missing a kettle\b",
                ],
                "tr": [
                    r"\bketıl yoktu\b",
                    r"\bodada su ısıtıcısı yoktu\b",
                    r"\bsu kaynatacak bir şey yoktu\b",
                ],
                "ar": [
                    r"\bمافيش كاتل في الأوضة\b",
                    r"\bمش قادرين نغلي ميّة\b",
                    r"\bما لقيناش غلاية في الغرفة\b",
                ],
                "zh": [
                    r"房间里没有热水壶",
                    r"没有水壶烧水",
                    r"没有可以烧水的设备",
                ],
            },
        ),

        "bed_comfy": AspectRule(
            aspect_code="bed_comfy",
            polarity_hint="positive",
            display="Комфорт кровати в целом",
            display_short="кровать удобная",
            long_hint="Гости отмечают, что кровать удобная: по ощущениям комфортно лежать и спать, есть поддержка тела, нет дискомфорта от пружин/стыков.",
            patterns_by_lang={{
                "ru": [
                    r"\bудобная кровать\b",
                    r"\bкровать очень удобная\b",
                    r"\bна кровати удобно спать\b",
                    r"\bспать на кровати было комфортно\b",
                    r"\bкровать комфортная\b",
                ],
                "en": [
                    r"\bcomfortable bed\b",
                    r"\bthe bed was very comfortable\b",
                    r"\bthe bed was comfy\b",
                    r"\bbed was super comfy\b",
                    r"\bthe bed felt great to sleep on\b",
                ],
                "tr": [
                    r"\brahat yatak\b",
                    r"\byatak çok rahattı\b",
                    r"\buyması çok rahattı\b",
                    r"\byatak konforluydu\b",
                ],
                "ar": [
                    r"\bالسيرير مريح\b",
                    r"\bالسرير كان مريح جداً\b",
                    r"\bالنوم على السرير كان مريح\b",
                    r"\bالسرير مريح للنوم\b",
                ],
                "zh": [
                    r"床很舒服",
                    r"床非常舒服",
                    r"床睡起来很舒服",
                    r"床很有舒适度",
                ],
            },
        ),
        
        
        "mattress_comfy": AspectRule(
            aspect_code="mattress_comfy",
            polarity_hint="positive",
            display="Комфорт матраса",
            display_short="матриc комфортный",
            long_hint="Гости подчеркивают именно качество матраса: удобный, поддерживающий спину, без провалов и посторонних ощущений от пружин.",
            patterns_by_lang={{
                "ru": [
                    r"\bкомфортный матрас\b",
                    r"\bматрас удобный\b",
                    r"\bочень удобный матрас\b",
                    r"\bматрас понравился\b",
                    r"\bматрас поддерживает спину\b",
                ],
                "en": [
                    r"\bcomfortable mattress\b",
                    r"\bthe mattress was comfortable\b",
                    r"\bvery comfy mattress\b",
                    r"\bgood mattress support\b",
                    r"\bthe mattress felt great\b",
                ],
                "tr": [
                    r"\brahat yatak minderi\b",
                    r"\byatak minderi çok rahattı\b",
                    r"\bşilte çok rahattı\b",
                    r"\bbelimizi iyi destekledi\b",
                ],
                "ar": [
                    r"\bالماترس مريح\b",
                    r"\bالمرتبة كانت مريحة\b",
                    r"\bالمرتبة دعمت ضهرنا كويس\b",
                    r"\bسرير المرتبة كان رايق\b",
                ],
                "zh": [
                    r"床垫很舒服",
                    r"床垫很有支撑",
                    r"床垫很合适睡觉",
                    r"床垫很舒服对腰很好",
                ],
            },
        ),
        
        
        "pillow_comfy": AspectRule(
            aspect_code="pillow_comfy",
            polarity_hint="positive",
            display="Комфорт подушек",
            display_short="удобные подушки",
            long_hint="Гости отмечают качество и удобство подушек: по высоте и мягкости подходят для сна, не вызывают дискомфорта шеи.",
            patterns_by_lang={{
                "ru": [
                    r"\bудобные подушки\b",
                    r"\bподушки комфортные\b",
                    r"\bподушки понравились\b",
                    r"\bподушки хорошей высоты\b",
                    r"\bподушки мягкие и удобные\b",
                ],
                "en": [
                    r"\bcomfortable pillows\b",
                    r"\bthe pillows were comfy\b",
                    r"\bperfect pillows\b",
                    r"\bpillows were just right\b",
                    r"\bpillows were soft and comfortable\b",
                ],
                "tr": [
                    r"\brahat yastıklar\b",
                    r"\byastıklar çok rahattı\b",
                    r"\byastıkların yüksekliği tamdı\b",
                    r"\byastık yumuşaktı ve konforluydu\b",
                ],
                "ar": [
                    r"\bالمخدات مريحة\b",
                    r"\bالمخدة كانت مريحة قوي\b",
                    r"\bارتفاع المخدة كان مناسب\b",
                    r"\bالمخدات طرية ومريحة\b",
                ],
                "zh": [
                    r"枕头很舒服",
                    r"枕头的高度正好",
                    r"枕头很软很舒服",
                    r"枕头睡着很合适",
                ],
            },
        ),
        
        
        "slept_well": AspectRule(
            aspect_code="slept_well",
            polarity_hint="positive",
            display="Качество сна гостей",
            display_short="спали хорошо",
            long_hint="Гости указывают, что выспались и сон был полноценным: комфорт кровати, отсутствие шума, комфортная температура позволили нормально отдохнуть.",
            patterns_by_lang={{
                "ru": [
                    r"\bхорошо выспал[сиа]\b",
                    r"\bотлично выспал[сиа]\b",
                    r"\bспали хорошо\b",
                    r"\bсон был отличный\b",
                    r"\bпрекрасно отдохнул[аи]\b",
                ],
                "en": [
                    r"\bslept very well\b",
                    r"\bhad a great sleep\b",
                    r"\bslept so well\b",
                    r"\bwe slept really well\b",
                    r"\bgood night's sleep\b",
                ],
                "tr": [
                    r"\bçok iyi uyuduk\b",
                    r"\bçok rahat uyuduk\b",
                    r"\buykumuz çok iyiydi\b",
                    r"\bgece çok iyi dinlendik\b",
                ],
                "ar": [
                    r"\bنمنا كويس جداً\b",
                    r"\bنوم مريح\b",
                    r"\bنمنا مرتاحين\b",
                    r"\bنومنا كان حلو\b",
                ],
                "zh": [
                    r"睡得很好",
                    r"一觉睡得很香",
                    r"晚上休息得很好",
                    r"我们睡得很舒服",
                ],
            },
        ),
        
        
        "mattress_too_soft": AspectRule(
            aspect_code="mattress_too_soft",
            polarity_hint="negative",
            display="Матрас слишком мягкий",
            display_short="слишком мягкий матрас",
            long_hint="Гости фиксируют, что матрас слишком мягкий, проваливается, не даёт опоры и мешает нормальному сну.",
            patterns_by_lang={{
                "ru": [
                    r"\bслишком мягкий матрас\b",
                    r"\bматрас слишком мягкий\b",
                    r"\bматрас проваливается\b",
                    r"\bпроваливали[сç]ь в матрас\b",
                    r"\bнет поддержки спине\b",
                ],
                "en": [
                    r"\bthe mattress was too soft\b",
                    r"\bvery soft mattress\b",
                    r"\bthe mattress was saggy\b",
                    r"\bno back support\b",
                    r"\bwe sank into the mattress\b",
                ],
                "tr": [
                    r"\byatak çok yumuşaktı\b",
                    r"\byatak orta kısmı çökmüş\b",
                    r"\byatak destek vermiyordu\b",
                    r"\byatağa gömüldük resmen\b",
                ],
                "ar": [
                    r"\bالمرتبة طرية بزيادة\b",
                    r"\bالمرتبة مهبطة\b",
                    r"\bمفيش دعم للظهر\b",
                    r"\bبنغوص في المراتب\b",
                ],
                "zh": [
                    r"床垫太软了",
                    r"床垫塌下去",
                    r"没有支撑力的床垫",
                    r"一躺下就整个人陷下去",
                ],
            },
        ),
    
        "bed_uncomfortable": AspectRule(
            aspect_code="bed_uncomfortable",
            polarity_hint="negative",
            display="Некомфортная кровать",
            display_short="кровать неудобная",
            long_hint="Гости сообщают, что кровать неудобная для сна: жёсткая/слишком мягкая, ощущаются стыки матрасов, пружины, не удалось нормально отдохнуть.",
            patterns_by_lang={{
                "ru": [
                    r"\bнеудобная кровать\b",
                    r"\bкровать очень неудобная\b",
                    r"\bна кровати неудобно спать\b",
                    r"\bкровать жёсткая и неудобная\b",
                    r"\bкровать некомфортная\b",
                ],
                "en": [
                    r"\buncomfortable bed\b",
                    r"\bthe bed was uncomfortable\b",
                    r"\bwe didn't sleep well because of the bed\b",
                    r"\bthe bed was hard and uncomfortable\b",
                    r"\bthe bed was not comfortable at all\b",
                ],
                "tr": [
                    r"\brahatsız yatak\b",
                    r"\byatak rahat değildi\b",
                    r"\byatak çok rahatsızdı\b",
                    r"\byatakta rahat uyuyamadık\b",
                ],
                "ar": [
                    r"\bالسرير مش مريح\b",
                    r"\bالسرير كان مزعج للنوم\b",
                    r"\bما قدرناش ننام كويس عالسيرير\b",
                    r"\bالسرير قاسي ومش مريح\b",
                ],
                "zh": [
                    r"床不舒服",
                    r"床很不舒服不好睡",
                    r"床很硬睡得不舒服",
                    r"因为床不舒服没睡好",
                ],
            },
        ),
        
        
        "mattress_too_hard": AspectRule(
            aspect_code="mattress_too_hard",
            polarity_hint="negative",
            display="Слишком жёсткий матрас",
            display_short="жёсткий матрас",
            long_hint="Гости отмечают, что матрас слишком жёсткий и жёсткость воспринимается как дискомфорт при сне (давит, сложно расслабиться, просыпаются с болями).",
            patterns_by_lang={{
                "ru": [
                    r"\bслишком жёсткий матрас\b",
                    r"\bматрас очень жёсткий\b",
                    r"\bочень твёрдый матрас\b",
                    r"\bспина болела из-за жёсткого матраса\b",
                ],
                "en": [
                    r"\bthe mattress was too hard\b",
                    r"\bvery hard mattress\b",
                    r"\bthe mattress was rock hard\b",
                    r"\bwoke up sore because the mattress was hard\b",
                ],
                "tr": [
                    r"\byatak çok sertti\b",
                    r"\bşilte aşırı sertti\b",
                    r"\byatak taş gibi sert\b",
                    r"\bsırtımız ağrıdı sertlikten\b",
                ],
                "ar": [
                    r"\bالمرتبة ناشفة قوي\b",
                    r"\bالمرتبة كانت قاسية جداً\b",
                    r"\bنمنا على مرتبة قاسية\b",
                    r"\bصحينا بوجع من قساوة المرتبة\b",
                ],
                "zh": [
                    r"床垫太硬了",
                    r"床垫特别硬像板子",
                    r"太硬睡得腰疼",
                    r"床垫硬得不舒服",
                ],
            },
        ),
        
        
        "mattress_sagging": AspectRule(
            aspect_code="mattress_sagging",
            polarity_hint="negative",
            display="Провисший/изношенный матрас",
            display_short="матрас проваливается",
            long_hint="Гости отмечают, что матрас изношен: чувствуется провал, яма посередине, матрас требует замены и не обеспечивает ровной поверхности сна.",
            patterns_by_lang={{
                "ru": [
                    r"\bматрас проваливался\b",
                    r"\bпровисший матрас\b",
                    r"\bяма в матрасе\b",
                    r"\bматрас просевший\b",
                    r"\bчувствуется яма посередине\b",
                ],
                "en": [
                    r"\bsagging mattress\b",
                    r"\bthe mattress was sagging\b",
                    r"\bthere was a dip in the mattress\b",
                    r"\bbig dip in the middle of the mattress\b",
                    r"\bworn out mattress\b",
                ],
                "tr": [
                    r"\bşilte çökmüştü\b",
                    r"\byatak ortadan göçmüştü\b",
                    r"\byatakta çukur vardı\b",
                    r"\beski bir yatak, çökmüş\b",
                ],
                "ar": [
                    r"\bالمرتبة مهبطة من النص\b",
                    r"\bالمرتبة غاطسة\b",
                    r"\bفي حفرة بالمرتبة\b",
                    r"\bالمرتبة باينة قديمة ومهبطة\b",
                ],
                "zh": [
                    r"床垫塌陷",
                    r"床垫中间有坑",
                    r"床垫塌下去了",
                    r"床垫已经很旧塌了",
                ],
            },
        ),
        
        
        "bed_creaks": AspectRule(
            aspect_code="bed_creaks",
            polarity_hint="negative",
            display="Скрип кровати",
            display_short="кровать скрипит",
            long_hint="Гости сообщают, что кровать заметно скрипит или шумит при движении, что мешает сну и воспринимается как признак изношенности конструкции.",
            patterns_by_lang={{
                "ru": [
                    r"\bкровать скрипит\b",
                    r"\bскрипучая кровать\b",
                    r"\bкаждое движение скрипит\b",
                    r"\bкровать шумит\b",
                ],
                "en": [
                    r"\bsqueaky bed\b",
                    r"\bthe bed was squeaking\b",
                    r"\bthe bed creaked with every move\b",
                    r"\bnoisy bed frame\b",
                ],
                "tr": [
                    r"\byatak gıcırdıyordu\b",
                    r"\byatak gıcır gıcır ses yapıyordu\b",
                    r"\bher hareket ettiğimizde yatak ses yaptı\b",
                ],
                "ar": [
                    r"\bالسرير بيزيق مع أي حركة\b",
                    r"\bالسرير بيطلع صوت\b",
                    r"\bالسرير مزعج وبيزن\b",
                ],
                "zh": [
                    r"床一动就吱吱响",
                    r"床架有很大声音",
                    r"床一直在吱嘎作响",
                ],
            },
        ),
        
        
        "pillow_uncomfortable": AspectRule(
            aspect_code="pillow_uncomfortable",
            polarity_hint="negative",
            display="Неудобные подушки",
            display_short="подушки неудобные",
            long_hint="Гости фиксируют дискомфорт от подушек: слишком жёсткие, слишком мягкие, слишком высокие или низкие, вызывают напряжение шеи и ухудшают качество сна.",
            patterns_by_lang={{
                "ru": [
                    r"\bнеудобные подушки\b",
                    r"\bподушки совсем не удобные\b",
                    r"\bподушка жёсткая и неудобная\b",
                    r"\bподушки какие-то странные\b",
                    r"\bболела шея из-за подушек\b",
                ],
                "en": [
                    r"\buncomfortable pillows\b",
                    r"\bthe pillows were uncomfortable\b",
                    r"\bthe pillows were awful\b",
                    r"\bthe pillow hurt my neck\b",
                    r"\bwe didn't like the pillows\b",
                ],
                "tr": [
                    r"\brahatsız yastıklar\b",
                    r"\byastık hiç rahat değildi\b",
                    r"\byastık boynumuzu ağrıttı\b",
                    r"\byastıklar konforsuzdu\b",
                ],
                "ar": [
                    r"\bالمخدات مش مريحة\b",
                    r"\bالمخدة وجعت الرقبة\b",
                    r"\bالمخدات كانت سيئة\b",
                    r"\bمش مرتاحين مع المخدات\b",
                ],
                "zh": [
                    r"枕头不舒服",
                    r"枕头很难睡",
                    r"枕头让脖子疼",
                    r"不喜欢房间的枕头",
                ],
            },
        ),

        "pillow_too_hard": AspectRule(
            aspect_code="pillow_too_hard",
            polarity_hint="negative",
            display="Подушки слишком жёсткие",
            display_short="жёсткие подушки",
            long_hint="Гости фиксируют, что подушки воспринимаются как слишком жёсткие или плотные, спать на них некомфортно, возникает напряжение в шее.",
            patterns_by_lang={{
                "ru": [
                    r"\bслишком жёсткие подушки\b",
                    r"\bподушка очень жёсткая\b",
                    r"\bочень твёрдая подушка\b",
                    r"\bневозможно спать на этих подушках\b",
                    r"\bшея болела из-за жёсткой подушки\b",
                ],
                "en": [
                    r"\bpillows were too hard\b",
                    r"\bvery hard pillow\b",
                    r"\bthe pillow was rock hard\b",
                    r"\bhard uncomfortable pillow\b",
                    r"\bthe pillow hurt my neck because it was too hard\b",
                ],
                "tr": [
                    r"\byastık çok sertti\b",
                    r"\başımızın altındaki yastık aşırı sertti\b",
                    r"\bsert yastıktan dolayı boynumuz ağrıdı\b",
                ],
                "ar": [
                    r"\bالمخدة كانت ناشفة قوي\b",
                    r"\bالمخدة قاسية جداً\b",
                    r"\bوجعت الرقبة لأنها قاسية\b",
                ],
                "zh": [
                    r"枕头太硬了",
                    r"枕头特别硬不好睡",
                    r"枕头硬得脖子疼",
                ],
            },
        ),
        
        
        "pillow_too_high": AspectRule(
            aspect_code="pillow_too_high",
            polarity_hint="negative",
            display="Подушки слишком высокие",
            display_short="высокие подушки",
            long_hint="Гости отмечают, что подушки слишком высокие/толстые, из-за чего спать неудобно, неудобное положение головы и шеи.",
            patterns_by_lang={{
                "ru": [
                    r"\bслишком высокие подушки\b",
                    r"\bподушка слишком высокая\b",
                    r"\bподушки очень толстые\b",
                    r"\bнеудобно из-за высоты подушки\b",
                    r"\bположение шеи неудобное\b",
                ],
                "en": [
                    r"\bpillows were too high\b",
                    r"\bthe pillow was too thick\b",
                    r"\bthe pillows were too big and high\b",
                    r"\bneck angle was uncomfortable because the pillow was high\b",
                ],
                "tr": [
                    r"\byastık çok yüksekti\b",
                    r"\byastık fazla kalındı\b",
                    r"\bboynumuz yukarıda kaldı\b",
                ],
                "ar": [
                    r"\bالمخدة عالية بزيادة\b",
                    r"\bالمخدة سميكة قوي\b",
                    r"\bمش مريح عشان المخدة عالية\b",
                ],
                "zh": [
                    r"枕头太高了",
                    r"枕头太厚太鼓",
                    r"枕头太高脖子不舒服",
                ],
            },
        ),
        
        
        "quiet_room": AspectRule(
            aspect_code="quiet_room",
            polarity_hint="positive",
            display="Тихий номер (акустический комфорт)",
            display_short="тихий номер",
            long_hint="Гости отмечают, что в номере было тихо: не мешал уличный шум, не слышны соседи и коридор. Это напрямую отражается на восприятии качества сна.",
            patterns_by_lang={{
                "ru": [
                    r"\bв номере тихо\b",
                    r"\bочень тихий номер\b",
                    r"\bникакого шума\b",
                    r"\bтишина ночью\b",
                    r"\bспокойно и тихо спали\b",
                ],
                "en": [
                    r"\bquiet room\b",
                    r"\bthe room was very quiet\b",
                    r"\bnice and quiet at night\b",
                    r"\bno noise at night\b",
                    r"\bwe slept in silence\b",
                ],
                "tr": [
                    r"\bodada çok sessizdi\b",
                    r"\bgece çok sakindi\b",
                    r"\bhiç gürültü yoktu\b",
                    r"\bsessiz bir oda\b",
                ],
                "ar": [
                    r"\bالأوضة هادية جداً\b",
                    r"\bمافيش دوشة بالليل\b",
                    r"\bالجو كان هادي في الأوضة\b",
                    r"\bنمنا من غير دوشة\b",
                ],
                "zh": [
                    r"房间很安静",
                    r"晚上很安静没有噪音",
                    r"房间隔音很好很安静",
                    r"夜里一点声音都没有",
                ],
            },
        ),
        
        
        "good_soundproofing": AspectRule(
            aspect_code="good_soundproofing",
            polarity_hint="positive",
            display="Хорошая звукоизоляция",
            display_short="звукоизоляция хорошая",
            long_hint="Гости подчеркивают, что звукоизоляция эффективная: не слышно соседей, коридора или улицы, шум не проникает в номер.",
            patterns_by_lang={{
                "ru": [
                    r"\bхорошая звукоизоляция\b",
                    r"\bотличная шумоизоляция\b",
                    r"\bничего не слышно из соседних номеров\b",
                    r"\bне слышно соседей\b",
                    r"\bне слышно коридор\b",
                ],
                "en": [
                    r"\bgood soundproofing\b",
                    r"\bthe room was well soundproofed\b",
                    r"\bwe couldn't hear the neighbors\b",
                    r"\bwe didn't hear any hallway noise\b",
                    r"\bno street noise came in\b",
                ],
                "tr": [
                    r"\bses yalıtımı iyiydi\b",
                    r"\bhiç dışarıdan ses gelmiyordu\b",
                    r"\bkomşu odaların sesi duyulmuyordu\b",
                    r"\bkoridor sesi duymadık\b",
                ],
                "ar": [
                    r"\bالعزل الصوتي كان كويس\b",
                    r"\bما سمعناش حد من الأوض اللي جنبنا\b",
                    r"\bولا صوت من الممر\b",
                    r"\bصوت الشارع ما بيوصلش للغرفة\b",
                ],
                "zh": [
                    r"隔音很好",
                    r"听不到隔壁的声音",
                    r"听不到走廊声音",
                    r"外面街上的声音基本听不见",
                ],
            },
        ),
        
        
        "no_street_noise": AspectRule(
            aspect_code="no_street_noise",
            polarity_hint="positive",
            display="Отсутствие уличного шума",
            display_short="не слышно улицу",
            long_hint="Гости отмечают, что в номере не слышно шум с улицы (трафик, бары, ночная жизнь), даже если объект расположен в оживлённой зоне. Это связывается с комфортом сна.",
            patterns_by_lang={{
                "ru": [
                    r"\bне слышно улицу\b",
                    r"\bне слышно дороги\b",
                    r"\bникакого шума с улицы\b",
                    r"\bшум с улицы не доносился\b",
                    r"\bмашин не слышно\b",
                ],
                "en": [
                    r"\bno street noise\b",
                    r"\bwe couldn't hear the street\b",
                    r"\bcouldn't hear traffic\b",
                    r"\bno noise from outside\b",
                    r"\bvery quiet despite being central\b",
                ],
                "tr": [
                    r"\bsokaktan ses gelmiyordu\b",
                    r"\btrafik sesi yoktu\b",
                    r"\bdışarıdan gürültü gelmedi\b",
                    r"\bmerkezde olmasına rağmen sessizdi\b",
                ],
                "ar": [
                    r"\bمافيش دوشة من الشارع\b",
                    r"\bما بنسمعش صوت الشارع\b",
                    r"\bمافيش صوت عربيات داخل الأوضة\b",
                    r"\bرغم إنه في منطقة حية بس مافيهوش دوشة جوة\b",
                ],
                "zh": [
                    r"听不到街上的声音",
                    r"几乎没有街道噪音",
                    r"外面车声听不进来",
                    r"虽然在市中心房间还是很安静",
                ],
            },
        ),

        "noisy_room": AspectRule(
            aspect_code="noisy_room",
            polarity_hint="negative",
            display="Шумный номер (низкий акустический комфорт)",
            display_short="шумный номер",
            long_hint="Гости сообщают, что в номере было шумно: мешали разговоры соседей, работа техники, музыка, коридорный трафик или уличный шум. Это влияет на восприятие качества сна.",
            patterns_by_lang={{
                "ru": [
                    r"\bв номере очень шумно\b",
                    r"\bшумный номер\b",
                    r"\bбыло громко в номере\b",
                    r"\bреально было слышно всё\b",
                    r"\bневозможно уснуть из-за шума\b",
                ],
                "en": [
                    r"\bnoisy room\b",
                    r"\bthe room was very noisy\b",
                    r"\bit was noisy in the room\b",
                    r"\btoo much noise in the room\b",
                    r"\bhard to sleep because of the noise\b",
                ],
                "tr": [
                    r"\bodada çok gürültü vardı\b",
                    r"\boğlenceli ama çok sesliydi\b",
                    r"\boda aşırı gürültülüydü\b",
                    r"\buymak zor oldu çünkü çok gürültü vardı\b",
                ],
                "ar": [
                    r"\bالأوضة دوشة جداً\b",
                    r"\bفيه دوشة في الأوضة\b",
                    r"\bمش عارفين ننام من الصوت\b",
                    r"\bالصوت عالي جوة الأوضة\b",
                ],
                "zh": [
                    r"房间很吵",
                    r"房间里特别吵",
                    r"房间里一直有噪音",
                    r"太吵了没法睡觉",
                ],
            },
        ),
        
        
        "street_noise": AspectRule(
            aspect_code="street_noise",
            polarity_hint="negative",
            display="Шум с улицы",
            display_short="шум с улицы",
            long_hint="Гости фиксируют навязчивый уличный шум (трафик, бар под окнами, люди на улице) как фактор дискомфорта, особенно в вечернее и ночное время.",
            patterns_by_lang={{
                "ru": [
                    r"\bшум с улицы\b",
                    r"\bсильный уличный шум\b",
                    r"\bочень шумно с дороги\b",
                    r"\bслышно машины под окнами\b",
                    r"\bслышно бар под окнами\b",
                ],
                "en": [
                    r"\bstreet noise\b",
                    r"\bnoise from the street\b",
                    r"\btraffic noise\b",
                    r"\bbar noise outside\b",
                    r"\bpeople yelling outside at night\b",
                ],
                "tr": [
                    r"\bsokak gürültüsü vardı\b",
                    r"\btrafik sesi tüm gece geliyordu\b",
                    r"\bsokağın sesi odaya giriyordu\b",
                    r"\bpencereden bar sesi geliyordu\b",
                ],
                "ar": [
                    r"\bصوت الشارع داخل الأوضة\b",
                    r"\bدوشة الشارع واضحة جداً\b",
                    r"\bصوت العربيات طول الليل\b",
                    r"\bفي دوشة من الناس تحت الشباك\b",
                ],
                "zh": [
                    r"有很大的街道噪音",
                    r"窗外车声很吵",
                    r"楼下街上很吵晚上都能听到",
                    r"路边的声音一直传进来",
                ],
            },
        ),
        
        
        "thin_walls": AspectRule(
            aspect_code="thin_walls",
            polarity_hint="negative",
            display="Плохая звукоизоляция между номерами",
            display_short="тонкие стены",
            long_hint="Гости отмечают, что слышат соседей: разговоры, телевизор, душ, двери. Сигнал того, что межкомнатная звукоизоляция слабая.",
            patterns_by_lang={{
                "ru": [
                    r"\bтонкие стены\b",
                    r"\bслышно соседей\b",
                    r"\bслышно разговоры через стену\b",
                    r"\bслышно как соседи смотрят тв\b",
                    r"\bкаждый звук из соседнего номера\b",
                ],
                "en": [
                    r"\bthin walls\b",
                    r"\byou can hear your neighbors\b",
                    r"\bwe could hear everything next door\b",
                    r"\bwe heard people through the walls\b",
                    r"\bno sound insulation between rooms\b",
                ],
                "tr": [
                    r"\bduvarlar çok inceydi\b",
                    r"\byan odadakileri net duyduk\b",
                    r"\bkomşu odanın sesi geliyordu\b",
                    r"\bses yalıtımı yok gibiydi\b",
                ],
                "ar": [
                    r"\bالحيطان رفيعة وكل حاجة بتسمع\b",
                    r"\bسمعنا الجيران بوضوح\b",
                    r"\bمفيش عزل بين الأوض\b",
                    r"\bصوت الغرفة اللي جنبنا واضح\b",
                ],
                "zh": [
                    r"墙很薄能听到隔壁",
                    r"隔壁说话都能听见",
                    r"几乎没有隔音房间之间",
                    r"能听到隔壁的声音很清楚",
                ],
            },
        ),
        
        
        "hallway_noise": AspectRule(
            aspect_code="hallway_noise",
            polarity_hint="negative",
            display="Шум из коридора",
            display_short="шум из коридора",
            long_hint="Гости жалуются, что слышен коридорный шум: хлопающие двери, разговоры проходящих гостей, уборка рано утром. Это воспринимается как нарушение приватности и покоя.",
            patterns_by_lang={{
                "ru": [
                    r"\bшум из коридора\b",
                    r"\bслышно коридор\b",
                    r"\bпостоянно хлопали двери в коридоре\b",
                    r"\bслышно как все ходят мимо\b",
                    r"\bочень шумно у двери номера\b",
                ],
                "en": [
                    r"\bhallway noise\b",
                    r"\bnoise from the hallway\b",
                    r"\bpeople talking loudly in the hallway\b",
                    r"\bdoors slamming in the hallway\b",
                    r"\bcleaning staff in the hallway woke us up\b",
                ],
                "tr": [
                    r"\bkoridordan ses geliyordu\b",
                    r"\bkoridor çok gürültülüydü\b",
                    r"\bkapılar koridorda sürekli çarpıyordu\b",
                    r"\btemizlik ekibi sabah erken koridorda çok ses yaptı\b",
                ],
                "ar": [
                    r"\bصوت الممر كان عالي\b",
                    r"\bفي دوشة من الكوريدور\b",
                    r"\bالناس في الممر بيزعّقوا وبيصحّونا\b",
                    r"\bصوت الأبواب اللي بتتخبط في الممر\b",
                ],
                "zh": [
                    r"走廊很吵",
                    r"走廊声音都能听到",
                    r"有人在走廊讲话很大声",
                    r"早上清洁阿姨在走廊很吵",
                ],
            },
        ),
        
        
        "night_noise_trouble_sleep": AspectRule(
            aspect_code="night_noise_trouble_sleep",
            polarity_hint="negative",
            display="Шум ночью мешал сну",
            display_short="шум мешал спать ночью",
            long_hint="Гости сообщают, что ночной шум (соседи, улица, музыка, двери, вечеринки) помешал уснуть или разбудил. Это фиксируется как фактор, ухудшающий качество сна.",
            patterns_by_lang={{
                "ru": [
                    r"\bшумно ночью\b",
                    r"\bночью было очень громко\b",
                    r"\bне могли уснуть из-за шума\b",
                    r"\bнас разбудил шум\b",
                    r"\bшум не дал спать всю ночь\b",
                ],
                "en": [
                    r"\bnoisy at night\b",
                    r"\bnoise at night kept us awake\b",
                    r"\bwe couldn't sleep because of the noise at night\b",
                    r"\bthe noise woke us up in the middle of the night\b",
                    r"\bparty noise at night\b",
                ],
                "tr": [
                    r"\bgece çok gürültü vardı\b",
                    r"\bgece uyuyamadık gürültüden\b",
                    r"\bgece boyunca ses kesilmedi\b",
                    r"\bgürültü bizi gecenin ortasında uyandırdı\b",
                ],
                "ar": [
                    r"\bفي دوشة طول الليل\b",
                    r"\bما قدرناش ننام من الصوت بالليل\b",
                    r"\bالصوت صحّانا نص الليل\b",
                    r"\bليل كله دوشة من الناس والشارع\b",
                ],
                "zh": [
                    r"晚上很吵",
                    r"半夜被噪音吵醒",
                    r"夜里吵得睡不着",
                    r"整晚都有声音影响睡觉",
                ],
            },
        ),

        "noisy_room": AspectRule(
            aspect_code="noisy_room",
            polarity_hint="negative",
            display="Шумный номер (низкий акустический комфорт)",
            display_short="шумный номер",
            long_hint="Гости сообщают, что в номере было шумно: мешали разговоры соседей, работа техники, музыка, коридорный трафик или уличный шум. Это влияет на восприятие качества сна.",
            patterns_by_lang={{
                "ru": [
                    r"\bв номере очень шумно\b",
                    r"\bшумный номер\b",
                    r"\bбыло громко в номере\b",
                    r"\bреально было слышно всё\b",
                    r"\bневозможно уснуть из-за шума\b",
                ],
                "en": [
                    r"\bnoisy room\b",
                    r"\bthe room was very noisy\b",
                    r"\bit was noisy in the room\b",
                    r"\btoo much noise in the room\b",
                    r"\bhard to sleep because of the noise\b",
                ],
                "tr": [
                    r"\bodada çok gürültü vardı\b",
                    r"\boğlenceli ama çok sesliydi\b",
                    r"\boda aşırı gürültülüydü\b",
                    r"\buymak zor oldu çünkü çok gürültü vardı\b",
                ],
                "ar": [
                    r"\bالأوضة دوشة جداً\b",
                    r"\bفيه دوشة في الأوضة\b",
                    r"\bمش عارفين ننام من الصوت\b",
                    r"\bالصوت عالي جوة الأوضة\b",
                ],
                "zh": [
                    r"房间很吵",
                    r"房间里特别吵",
                    r"房间里一直有噪音",
                    r"太吵了没法睡觉",
                ],
            },
        ),
        
        
        "street_noise": AspectRule(
            aspect_code="street_noise",
            polarity_hint="negative",
            display="Шум с улицы",
            display_short="шум с улицы",
            long_hint="Гости фиксируют навязчивый уличный шум (трафик, бар под окнами, люди на улице) как фактор дискомфорта, особенно в вечернее и ночное время.",
            patterns_by_lang={{
                "ru": [
                    r"\bшум с улицы\b",
                    r"\bсильный уличный шум\b",
                    r"\bочень шумно с дороги\b",
                    r"\bслышно машины под окнами\b",
                    r"\bслышно бар под окнами\b",
                ],
                "en": [
                    r"\bstreet noise\b",
                    r"\bnoise from the street\b",
                    r"\btraffic noise\b",
                    r"\bbar noise outside\b",
                    r"\bpeople yelling outside at night\b",
                ],
                "tr": [
                    r"\bsokak gürültüsü vardı\b",
                    r"\btrafik sesi tüm gece geliyordu\b",
                    r"\bsokağın sesi odaya giriyordu\b",
                    r"\bpencereden bar sesi geliyordu\b",
                ],
                "ar": [
                    r"\bصوت الشارع داخل الأوضة\b",
                    r"\bدوشة الشارع واضحة جداً\b",
                    r"\bصوت العربيات طول الليل\b",
                    r"\bفي دوشة من الناس تحت الشباك\b",
                ],
                "zh": [
                    r"有很大的街道噪音",
                    r"窗外车声很吵",
                    r"楼下街上很吵晚上都能听到",
                    r"路边的声音一直传进来",
                ],
            },
        ),
        
        
        "thin_walls": AspectRule(
            aspect_code="thin_walls",
            polarity_hint="negative",
            display="Плохая звукоизоляция между номерами",
            display_short="тонкие стены",
            long_hint="Гости отмечают, что слышат соседей: разговоры, телевизор, душ, двери. Сигнал того, что межкомнатная звукоизоляция слабая.",
            patterns_by_lang={{
                "ru": [
                    r"\bтонкие стены\b",
                    r"\bслышно соседей\b",
                    r"\bслышно разговоры через стену\b",
                    r"\bслышно как соседи смотрят тв\b",
                    r"\bкаждый звук из соседнего номера\b",
                ],
                "en": [
                    r"\bthin walls\b",
                    r"\byou can hear your neighbors\b",
                    r"\bwe could hear everything next door\b",
                    r"\bwe heard people through the walls\b",
                    r"\bno sound insulation between rooms\b",
                ],
                "tr": [
                    r"\bduvarlar çok inceydi\b",
                    r"\byan odadakileri net duyduk\b",
                    r"\bkomşu odanın sesi geliyordu\b",
                    r"\bses yalıtımı yok gibiydi\b",
                ],
                "ar": [
                    r"\bالحيطان رفيعة وكل حاجة بتسمع\b",
                    r"\bسمعنا الجيران بوضوح\b",
                    r"\bمفيش عزل بين الأوض\b",
                    r"\bصوت الغرفة اللي جنبنا واضح\b",
                ],
                "zh": [
                    r"墙很薄能听到隔壁",
                    r"隔壁说话都能听见",
                    r"几乎没有隔音房间之间",
                    r"能听到隔壁的声音很清楚",
                ],
            },
        ),
        
        
        "hallway_noise": AspectRule(
            aspect_code="hallway_noise",
            polarity_hint="negative",
            display="Шум из коридора",
            display_short="шум из коридора",
            long_hint="Гости жалуются, что слышен коридорный шум: хлопающие двери, разговоры проходящих гостей, уборка рано утром. Это воспринимается как нарушение приватности и покоя.",
            patterns_by_lang={{
                "ru": [
                    r"\bшум из коридора\b",
                    r"\bслышно коридор\b",
                    r"\bпостоянно хлопали двери в коридоре\b",
                    r"\bслышно как все ходят мимо\b",
                    r"\bочень шумно у двери номера\b",
                ],
                "en": [
                    r"\bhallway noise\b",
                    r"\bnoise from the hallway\b",
                    r"\bpeople talking loudly in the hallway\b",
                    r"\bdoors slamming in the hallway\b",
                    r"\bcleaning staff in the hallway woke us up\b",
                ],
                "tr": [
                    r"\bkoridordan ses geliyordu\b",
                    r"\bkoridor çok gürültülüydü\b",
                    r"\bkapılar koridorda sürekli çarpıyordu\b",
                    r"\btemizlik ekibi sabah erken koridorda çok ses yaptı\b",
                ],
                "ar": [
                    r"\bصوت الممر كان عالي\b",
                    r"\bفي دوشة من الكوريدور\b",
                    r"\bالناس في الممر بيزعّقوا وبيصحّونا\b",
                    r"\bصوت الأبواب اللي بتتخبط في الممر\b",
                ],
                "zh": [
                    r"走廊很吵",
                    r"走廊声音都能听到",
                    r"有人在走廊讲话很大声",
                    r"早上清洁阿姨在走廊很吵",
                ],
            },
        ),
        
        
        "night_noise_trouble_sleep": AspectRule(
            aspect_code="night_noise_trouble_sleep",
            polarity_hint="negative",
            display="Шум ночью мешал сну",
            display_short="шум мешал спать ночью",
            long_hint="Гости сообщают, что ночной шум (соседи, улица, музыка, двери, вечеринки) помешал уснуть или разбудил. Это фиксируется как фактор, ухудшающий качество сна.",
            patterns_by_lang={{
                "ru": [
                    r"\bшумно ночью\b",
                    r"\bночью было очень громко\b",
                    r"\bне могли уснуть из-за шума\b",
                    r"\bнас разбудил шум\b",
                    r"\bшум не дал спать всю ночь\b",
                ],
                "en": [
                    r"\bnoisy at night\b",
                    r"\bnoise at night kept us awake\b",
                    r"\bwe couldn't sleep because of the noise at night\b",
                    r"\bthe noise woke us up in the middle of the night\b",
                    r"\bparty noise at night\b",
                ],
                "tr": [
                    r"\bgece çok gürültü vardı\b",
                    r"\bgece uyuyamadık gürültüden\b",
                    r"\bgece boyunca ses kesilmedi\b",
                    r"\bgürültü bizi gecenin ortasında uyandırdı\b",
                ],
                "ar": [
                    r"\bفي دوشة طول الليل\b",
                    r"\bما قدرناش ننام من الصوت بالليل\b",
                    r"\bالصوت صحّانا نص الليل\b",
                    r"\bليل كله دوشة من الناس والشارع\b",
                ],
                "zh": [
                    r"晚上很吵",
                    r"半夜被噪音吵醒",
                    r"夜里吵得睡不着",
                    r"整晚都有声音影响睡觉",
                ],
            },
        ),
            
        "temp_comfortable": AspectRule(
            aspect_code="temp_comfortable",
            polarity_hint="positive",
            display="Комфортная температура в номере",
            display_short="температура комфортная",
            long_hint="Гости отмечают, что температурный режим в номере воспринимается как комфортный и стабильный: не жарко и не холодно, отдых возможен без дополнительной регулировки климата.",
            patterns_by_lang={{
                "ru": [
                    r"\bкомфортная температура\b",
                    r"\bтемпература (в|внутри )?номера комфортн(ая|о)\b",
                    r"\bне жарко и не холодно\b",
                    r"\bприятный климат в номере\b",
                ],
                "en": [
                    r"\bcomfortable temperature\b",
                    r"\broom temperature was comfortable\b",
                    r"\bnot too hot (or|and) not too cold\b",
                    r"\bpleasant room climate\b",
                ],
                "tr": [
                    r"\boda sıcaklığı rahattı\b",
                    r"\bne çok sıcak ne çok soğuktu\b",
                    r"\boda ısısı uygundu\b",
                    r"\bkonforlu sıcaklık\b",
                ],
                "ar": [
                    r"\bدرجة الحرارة كانت مناسبة\b",
                    r"\bمش حر ولا برد\b",
                    r"\bجو الأوضة مريح\b",
                    r"\bحرارة الغرفة مريحة\b",
                ],
                "zh": [
                    r"室内温度很舒适",
                    r"不冷不热",
                    r"房间温度刚好",
                    r"房间气候舒适",
                ],
            },
        ),
        
        
        "ventilation_ok": AspectRule(
            aspect_code="ventilation_ok",
            polarity_hint="positive",
            display="Эффективная вентиляция и проветривание",
            display_short="вентиляция ок",
            long_hint="Гости отмечают, что в номере не душно: воздух обновляется, чувствуется проветривание, отсутствует застой и тяжесть.",
            patterns_by_lang={{
                "ru": [
                    r"\bне душно\b",
                    r"\bхорошая вентиляция\b",
                    r"\bесть чем дышать\b",
                    r"\bпроветривание нормальное\b",
                ],
                "en": [
                    r"\bwell[- ]ventilated\b",
                    r"\bgood ventilation\b",
                    r"\bnot stuffy\b",
                    r"\bfresh air in the room\b",
                ],
                "tr": [
                    r"\biyı havalandırılmış\b",
                    r"\bhava sirkülasyonu iyiydi\b",
                    r"\bodada havasızlık yoktu\b",
                    r"\bkolay havalandırdık\b",
                ],
                "ar": [
                    r"\bمش مكتوم\b",
                    r"\bفي تهوية كويسة\b",
                    r"\bهواء نضيف في الأوضة\b",
                    r"\bبنهوّي بسهولة\b",
                ],
                "zh": [
                    r"通风很好",
                    r"不闷",
                    r"空气流通",
                    r"有新鲜空气",
                ],
            },
        ),
        
        
        "ac_working": AspectRule(
            aspect_code="ac_working",
            polarity_hint="positive",
            display="Работоспособность кондиционера (охлаждение)",
            display_short="кондиционер работает",
            long_hint="Гости отмечают, что кондиционер работает корректно и эффективно охлаждает номер до комфортной температуры без сбоев.",
            patterns_by_lang={{
                "ru": [
                    r"\bкондиционер работал\b",
                    r"\bкондиционер хорошо охлаждал\b",
                    r"\bработал кондиционер без сбоев\b",
                    r"\bв номере прохладно благодаря кондиционеру\b",
                ],
                "en": [
                    r"\bAC worked\b",
                    r"\bair conditioning worked\b",
                    r"\bAC cooled the room\b",
                    r"\bthe AC was effective\b",
                ],
                "tr": [
                    r"\bklima çalışıyordu\b",
                    r"\bklima iyi soğuttu\b",
                    r"\bklima sorunsuzdu\b",
                    r"\bAC çalıştı\b",
                ],
                "ar": [
                    r"\bالتكييف شغّال كويس\b",
                    r"\bالتكييف برّد الأوضة\b",
                    r"\bAC شغال بدون مشاكل\b",
                ],
                "zh": [
                    r"空调工作正常",
                    r"空调制冷很好",
                    r"空调很给力",
                    r"空调很有效地降温",
                ],
            },
        ),
        
        
        "heating_working": AspectRule(
            aspect_code="heating_working",
            polarity_hint="positive",
            display="Работоспособность отопления/обогрева",
            display_short="отопление работает",
            long_hint="Гости отмечают, что система отопления/обогреватели работают стабильно: в номере тепло, без перепадов и жалоб на холод.",
            patterns_by_lang={{
                "ru": [
                    r"\bотопление работало\b",
                    r"\bв номере тепло\b",
                    r"\bбатареи т[её]плые\b",
                    r"\bобогреватель грел\b",
                ],
                "en": [
                    r"\bheating worked\b",
                    r"\bthe room was warm\b",
                    r"\bradiators were warm\b",
                    r"\bheater worked well\b",
                ],
                "tr": [
                    r"\bısıtma çalışıyordu\b",
                    r"\boda sıcaktı\b",
                    r"\bpetekler sıcaktı\b",
                    r"\bsoba/ısıtıcı iyi ısıttı\b",
                ],
                "ar": [
                    r"\bالتدفئة شغالة\b",
                    r"\bالأوضة كانت دافيه\b",
                    r"\bالرادياتير كان سخن\b",
                    r"\bالدفاية شغالة كويس\b",
                ],
                "zh": [
                    r"暖气正常",
                    r"房间很暖和",
                    r"取暖工作正常",
                    r"暖气很足",
                ],
            },
        ),
        
        
        "too_hot_sleep_issue": AspectRule(
            aspect_code="too_hot_sleep_issue",
            polarity_hint="negative",
            display="Жарко в номере, мешает сну",
            display_short="слишком жарко",
            long_hint="Гости фиксируют перегрев помещения: ночью жарко или душно, что мешает уснуть или вызывает пробуждения.",
            patterns_by_lang={{
                "ru": [
                    r"\bслишком жарко\b",
                    r"\bжара мешала спать\b",
                    r"\bневозможно уснуть от жары\b",
                    r"\bночью было душно и жарко\b",
                ],
                "en": [
                    r"\btoo hot to sleep\b",
                    r"\bthe room was too hot\b",
                    r"\boverheated room\b",
                    r"\bstifling heat at night\b",
                ],
                "tr": [
                    r"\bçok sıcaktı uyuyamadık\b",
                    r"\boda aşırı sıcaktı\b",
                    r"\bgece sıcak yüzünden uyku zor\b",
                    r"\bsıcaktan bunaltıcıydı\b",
                ],
                "ar": [
                    r"\bحر جداً ومش عارفين ننام\b",
                    r"\bالأوضة حر قوي بالليل\b",
                    r"\bسخونة مزعجة ومكتوم\b",
                    r"\bحرارة عالية مع الليل\b",
                ],
                "zh": [
                    r"太热睡不着",
                    r"房间太热",
                    r"晚上很闷热",
                    r"热得难以入睡",
                ],
            },
        ),

        "too_cold": AspectRule(
            aspect_code="too_cold",
            polarity_hint="negative",
            display="Слишком холодно в номере",
            display_short="слишком холодно",
            long_hint="Гости сообщают, что в номере холодно и температура не поднимается до комфортной: зябко, сквозняк, сложно спать из-за холода.",
            patterns_by_lang={{
                "ru": [
                    r"\bслишком холодно\b",
                    r"\bв номере холодно\b",
                    r"\bочень холодно спать\b",
                    r"\bночью замерзли\b",
                    r"\bхолодно даже под одеялом\b",
                ],
                "en": [
                    r"\bthe room was too cold\b",
                    r"\bit was freezing in the room\b",
                    r"\bvery cold at night\b",
                    r"\bwe were cold while sleeping\b",
                    r"\bthe room never got warm\b",
                ],
                "tr": [
                    r"\bodada çok soğuktu\b",
                    r"\büşüdük odada\b",
                    r"\bgece odada üşüdük\b",
                    r"\bodanın ısınmaması\b",
                ],
                "ar": [
                    r"\bالأوضة كانت برد جداً\b",
                    r"\bكنّا بردانين طول الليل\b",
                    r"\bمش بتدفي الغرفة\b",
                    r"\bمش دافيين وإحنا نايمين\b",
                ],
                "zh": [
                    r"房间很冷",
                    r"晚上很冷睡觉会冷",
                    r"房间一直不暖和",
                    r"冷得睡不着",
                ],
            },
        ),
        
        
        "stuffy_no_air": AspectRule(
            aspect_code="stuffy_no_air",
            polarity_hint="negative",
            display="Душно и тяжело дышать в номере",
            display_short="духота в номере",
            long_hint="Гости отмечают, что в номере спертый воздух: душно, нет притока свежего воздуха, тяжёлое ощущение в помещении.",
            patterns_by_lang={{
                "ru": [
                    r"\bдушно в номере\b",
                    r"\bспертый воздух\b",
                    r"\bнечем дышать\b",
                    r"\bочень душно ночью\b",
                    r"\bнет притока воздуха\b",
                ],
                "en": [
                    r"\bthe room was stuffy\b",
                    r"\bstuffy air\b",
                    r"\bno fresh air in the room\b",
                    r"\bair felt heavy\b",
                    r"\bit was hard to breathe in the room\b",
                ],
                "tr": [
                    r"\bodada hava çok basıktı\b",
                    r"\bhava tıkalıydı\b",
                    r"\bodada çok havasızdı\b",
                    r"\bnefes almak zordu içeride\b",
                ],
                "ar": [
                    r"\bالأوضة مكتومة\b",
                    r"\bالجو مكتوم ومفيش هوا\b",
                    r"\bنفس تقيل جوة الأوضة\b",
                    r"\bمفيش هوى نضيف\b",
                ],
                "zh": [
                    r"房间很闷",
                    r"空气很闷很沉",
                    r"房间里空气不流通",
                    r"感觉很憋气",
                ],
            },
        ),
        
        
        "no_ventilation": AspectRule(
            aspect_code="no_ventilation",
            polarity_hint="negative",
            display="Отсутствие нормальной вентиляции",
            display_short="нет вентиляции",
            long_hint="Гости фиксируют, что помещение не проветривается: окна не открываются, вытяжка не работает, нет движения воздуха ни в комнате, ни в санузле.",
            patterns_by_lang={{
                "ru": [
                    r"\bнет вентиляции\b",
                    r"\bне проветривается\b",
                    r"\bокно не открыть\b",
                    r"\bвытяжка не работает\b",
                    r"\bвоздух не циркулирует\b",
                ],
                "en": [
                    r"\bno ventilation\b",
                    r"\bno air circulation\b",
                    r"\bthe window wouldn't open\b",
                    r"\bno working extractor fan\b",
                    r"\bthe bathroom fan didn't work\b",
                ],
                "tr": [
                    r"\bhavalandırma yoktu\b",
                    r"\bhava hiç dönmüyordu\b",
                    r"\bpencere açılmıyordu\b",
                    r"\bhavalandırma fanı çalışmıyordu\b",
                ],
                "ar": [
                    r"\bمافيش تهوية\b",
                    r"\bالشباك مش بيفتح\b",
                    r"\bمفيش شفاط شغال\b",
                    r"\bمفيش حركة هوا في الأوضة\b",
                ],
                "zh": [
                    r"没有通风",
                    r"房间空气不流通",
                    r"窗户打不开",
                    r"排风扇不工作",
                ],
            },
        ),
        
        
        "ac_not_working": AspectRule(
            aspect_code="ac_not_working",
            polarity_hint="negative",
            display="Неработающий или неэффективный кондиционер",
            display_short="кондиционер не работает",
            long_hint="Гости сообщают, что кондиционер не включается, не охлаждает или дует тёплым воздухом, из-за чего температура в номере остаётся некомфортной.",
            patterns_by_lang={{
                "ru": [
                    r"\bкондиционер не работал\b",
                    r"\bкондиционер сломан\b",
                    r"\bкондиционер не охлаждал\b",
                    r"\bиз кондиционера дул тёплый воздух\b",
                    r"\bAC не работал\b",
                ],
                "en": [
                    r"\bthe AC didn't work\b",
                    r"\bthe air conditioning was broken\b",
                    r"\bAC was not cooling\b",
                    r"\bthe AC blew warm air\b",
                    r"\bthe AC wasn't turning on\b",
                ],
                "tr": [
                    r"\bklima çalışmıyordu\b",
                    r"\bklima bozuktu\b",
                    r"\bklima soğutmuyordu\b",
                    r"\bklima sadece sıcak hava üflüyordu\b",
                ],
                "ar": [
                    r"\bالتكييف مش شغال\b",
                    r"\bالتكييف بايظ\b",
                    r"\bالتكييف ما بيبردش\b",
                    r"\bالـAC بيطلع هوا دافي بس\b",
                ],
                "zh": [
                    r"空调不工作",
                    r"空调不制冷",
                    r"空调只吹热风",
                    r"空调开了也没降温",
                ],
            },
        ),
        
        
        "no_ac": AspectRule(
            aspect_code="no_ac",
            polarity_hint="negative",
            display="Отсутствие кондиционера как класса",
            display_short="нет кондиционера",
            long_hint="Гости отмечают, что в номере нет кондиционера вообще, что воспринимается критично в тёплый сезон или для верхних этажей/солнечной стороны.",
            patterns_by_lang={{
                "ru": [
                    r"\bнет кондиционера\b",
                    r"\bбез кондиционера\b",
                    r"\bв комнате не было AC\b",
                    r"\bникакого кондиционера\b",
                    r"\bжарко а кондиционера нет\b",
                ],
                "en": [
                    r"\bno AC\b",
                    r"\bno air conditioning\b",
                    r"\bthere was no air conditioner in the room\b",
                    r"\bno aircon in the room\b",
                    r"\bno air conditioning which made it hot\b",
                ],
                "tr": [
                    r"\bklima yoktu\b",
                    r"\bodada klima yok\b",
                    r"\bklimasız oda\b",
                    r"\bçok sıcaktı ama klima yoktu\b",
                ],
                "ar": [
                    r"\bمفيش تكييف في الأوضة\b",
                    r"\bالأوضة مافيهاش AC أصلاً\b",
                    r"\bحر ومفيش تكييف\b",
                ],
                "zh": [
                    r"房间里没有空调",
                    r"没有空调特别热",
                    r"房间完全没有空调设备",
                ],
            },
        ),

        "heating_not_working": AspectRule(
            aspect_code="heating_not_working",
            polarity_hint="negative",
            display="Проблемы с отоплением / обогрев не работает",
            display_short="отопление не работает",
            long_hint="Гости фиксируют, что система отопления не греет: батареи холодные, обогреватель не работает, температура в номере не поднимается до комфортного уровня.",
            patterns_by_lang={{
                "ru": [
                    r"\bотопление не работало\b",
                    r"\bбатареи холодные\b",
                    r"\bобогреватель не грел\b",
                    r"\bв номере не нагревалось\b",
                    r"\bотопление сломано\b",
                ],
                "en": [
                    r"\bthe heating didn't work\b",
                    r"\bno heating in the room\b",
                    r"\bthe heater was not working\b",
                    r"\bradiators were cold\b",
                    r"\bthe room wouldn't warm up\b",
                ],
                "tr": [
                    r"\bısıtma çalışmıyordu\b",
                    r"\bpetekler soğuktu\b",
                    r"\bkalorifer yanmıyordu\b",
                    r"\boda ısınmadı\b",
                ],
                "ar": [
                    r"\bالتدفئة مش شغالة\b",
                    r"\bالدفاية ما بتسخنش\b",
                    r"\bالرادياتير بارد\b",
                    r"\bالأوضة ما بتسخنش خالص\b",
                ],
                "zh": [
                    r"暖气不工作",
                    r"房间没有暖气",
                    r"暖气是凉的",
                    r"房间怎么都不变暖",
                ],
            },
        ),
        
        
        "draft_window": AspectRule(
            aspect_code="draft_window",
            polarity_hint="negative",
            display="Сквозняк из окна / негерметичные окна",
            display_short="сквозит из окна",
            long_hint="Гости отмечают, что из окна или балконной двери дует: ощущается сквозняк, холодный воздух проникает внутрь, влияет на тепловой комфорт и сон.",
            patterns_by_lang={{
                "ru": [
                    r"\bдует из окна\b",
                    r"\bсквозняк из окна\b",
                    r"\bпродувает окно\b",
                    r"\bокна не держат тепло\b",
                    r"\bхолод от окна\b",
                ],
                "en": [
                    r"\bdraft from the window\b",
                    r"\bdrafty windows\b",
                    r"\bcold air coming through the window\b",
                    r"\bthe window didn't seal properly\b",
                    r"\bwind coming in through the window\b",
                ],
                "tr": [
                    r"\bpencereden rüzgar geliyordu\b",
                    r"\bpencereden soğuk üflüyordu\b",
                    r"\bpencere iyi kapanmıyordu\b",
                    r"\boda cereyan yapıyordu\b",
                ],
                "ar": [
                    r"\bفي هوا داخل من الشباك\b",
                    r"\bفي سحب هوا بارد من ناحية الشباك\b",
                    r"\bالشباك مش مقفول كويس\b",
                    r"\bهوا ساقع داخل من البلكونة\b",
                ],
                "zh": [
                    r"窗户漏风",
                    r"有冷风从窗户灌进来",
                    r"窗户关不严有冷风",
                    r"房间有穿堂风从窗边进来",
                ],
            },
        ),
        
        
        "room_spacious": AspectRule(
            aspect_code="room_spacious",
            polarity_hint="positive",
            display="Простор номера / достаточная площадь",
            display_short="номер просторный",
            long_hint="Гости отмечают, что номер просторный: достаточно площади для комфортного передвижения и размещения багажа без ощущения тесноты.",
            patterns_by_lang={{
                "ru": [
                    r"\bпросторный номер\b",
                    r"\bномер большой\b",
                    r"\bмного места в номере\b",
                    r"\bдостаточно места\b",
                    r"\bне тесно\b",
                ],
                "en": [
                    r"\bspacious room\b",
                    r"\bthe room was spacious\b",
                    r"\blots of space\b",
                    r"\bplenty of space in the room\b",
                    r"\bthe room felt big\b",
                ],
                "tr": [
                    r"\boğlu geniş bir odaydı\b",
                    r"\boda çok genişti\b",
                    r"\bferah bir oda\b",
                    r"\bodada bolca alan vardı\b",
                ],
                "ar": [
                    r"\bالأوضة وسيعة\b",
                    r"\bالمساحة كبيرة\b",
                    r"\bفيه مساحة تتحرك براحتك\b",
                    r"\bمش ديقة خالص\b",
                ],
                "zh": [
                    r"房间很宽敞",
                    r"空间很大",
                    r"房间面积够大",
                    r"活动空间很充裕",
                ],
            },
        ),
        
        
        "good_layout": AspectRule(
            aspect_code="good_layout",
            polarity_hint="positive",
            display="Продуманная планировка номера",
            display_short="удобная планировка",
            long_hint="Гости отмечают, что зонирование и организация пространства в номере удобны: мебель расставлена логично, ничто не мешает проходу, хорошая эргономика.",
            patterns_by_lang={{
                "ru": [
                    r"\bудобная планировка\b",
                    r"\bвсё грамотно расставлено\b",
                    r"\bхорошо организованное пространство\b",
                    r"\bвсё продумано в номере\b",
                    r"\bничего не мешает передвигаться\b",
                ],
                "en": [
                    r"\bgood layout\b",
                    r"\bthe layout was very practical\b",
                    r"\bwell designed room layout\b",
                    r"\bwell organized space\b",
                    r"\bthe furniture was well arranged\b",
                ],
                "tr": [
                    r"\bodanın düzeni çok iyiydi\b",
                    r"\boda kullanışlı düzenlenmişti\b",
                    r"\bmobilya yerleşimi mantıklıydı\b",
                    r"\boda planı rahattı\b",
                ],
                "ar": [
                    r"\bتقسيم الأوضة مريح\b",
                    r"\bالتوزيع حلو ومريح في الحركة\b",
                    r"\bالأثاث مترتب بعقل\b",
                    r"\bالتصميم عملي جداً جوة الأوضة\b",
                ],
                "zh": [
                    r"房间布局很合理",
                    r"动线很顺",
                    r"房间设计得很实用",
                    r"家具摆放很合理不碍事",
                ],
            },
        ),
        
        
        "cozy_feel": AspectRule(
            aspect_code="cozy_feel",
            polarity_hint="positive",
            display="Уют и атмосферность номера",
            display_short="уютный номер",
            long_hint="Гости описывают номер как уютный и приятный по атмосфере: тёплый свет, тканевые элементы, ощущение домашнего комфорта, хочется находиться внутри.",
            patterns_by_lang={{
                "ru": [
                    r"\bуютный номер\b",
                    r"\bочень уютно\b",
                    r"\bатмосфера домашняя\b",
                    r"\bприятная атмосфера\b",
                    r"\bпо-домашнему комфортно\b",
                ],
                "en": [
                    r"\bcozy room\b",
                    r"\bvery cozy\b",
                    r"\bfelt very homey\b",
                    r"\bnice cozy atmosphere\b",
                    r"\bthe room felt welcoming\b",
                ],
                "tr": [
                    r"\bçok sıcak ve huzurlu bir ortam\b",
                    r"\boda çok sıcak/rahat hissettiriyor\b",
                    r"\boda çok samimi ve cozy\b",
                    r"\bev gibi hissettirdi\b",
                ],
                "ar": [
                    r"\bالأوضة دافية وبتحسسك بالبيت\b",
                    r"\bالغرفة مريحة وبتحس ببيئة دافية\b",
                    r"\bجو الغرفة حميمي ومرتاح\b",
                    r"\bالإحساس كان كأنه في بيت\b",
                ],
                "zh": [
                    r"房间很温馨",
                    r"很有家的感觉",
                    r"房间氛围很舒服很温暖",
                    r"让人觉得很放松很有安全感",
                ],
            },
        ),

        "bright_room": AspectRule(
            aspect_code="bright_room",
            polarity_hint="positive",
            display="Достаточное естественное освещение номера",
            display_short="светлый номер",
            long_hint="Гости отмечают, что номер светлый: много дневного света, хорошее естественное освещение, приятная освещённость в течение дня.",
            patterns_by_lang={{
                "ru": [
                    r"\bсветлый номер\b",
                    r"\bочень светло в номере\b",
                    r"\bмного дневного света\b",
                    r"\bмного естественного света\b",
                    r"\bсолнечный номер\b",
                ],
                "en": [
                    r"\bbright room\b",
                    r"\bthe room was very bright\b",
                    r"\blots of natural light\b",
                    r"\bplenty of daylight\b",
                    r"\bthe room gets a lot of sun\b",
                ],
                "tr": [
                    r"\boda çok aydınlıktı\b",
                    r"\bdoğal ışık çok iyiydi\b",
                    r"\bgün ışığı boldu\b",
                    r"\bgüneş alan bir oda\b",
                ],
                "ar": [
                    r"\bالأوضة منورة\b",
                    r"\bفيه نور طبيعي حلو\b",
                    r"\bالشمس داخلة الأوضة كويس\b",
                    r"\bالغرفة منورة بالنهار\b",
                ],
                "zh": [
                    r"房间很亮",
                    r"房间采光很好",
                    r"白天很明亮",
                    r"有很多自然光",
                ],
            },
        ),
        
        
        "big_windows": AspectRule(
            aspect_code="big_windows",
            polarity_hint="positive",
            display="Большие окна / хорошие окна",
            display_short="большие окна",
            long_hint="Гости подчёркивают, что в номере большие окна или панорамное остекление, что создаёт ощущение пространства, света и визуальной открытости.",
            patterns_by_lang={{
                "ru": [
                    r"\bбольшие окна\b",
                    r"\bпанорамные окна\b",
                    r"\bогромные окна\b",
                    r"\bокна от пола до потолка\b",
                    r"\bклассные большие окна\b",
                ],
                "en": [
                    r"\bbig windows\b",
                    r"\bhuge windows\b",
                    r"\bfloor[- ]to[- ]ceiling windows\b",
                    r"\blarge windows letting in lots of light\b",
                    r"\bnice big window\b",
                ],
                "tr": [
                    r"\bbüyük pencereler vardı\b",
                    r"\btavana kadar cam\b",
                    r"\boda büyük camlara sahipti\b",
                    r"\bçok geniş pencere\b",
                ],
                "ar": [
                    r"\bشبابيك كبيرة\b",
                    r"\bشباك كبير بيدخل نور\b",
                    r"\bزجاج كبير تقريباً للأرض\b",
                    r"\bمنظر حلو من الشباك الكبير\b",
                ],
                "zh": [
                    r"窗户很大",
                    r"有大落地窗",
                    r"超大的窗子采光很好",
                    r"很大的窗户能看到外面",
                ],
            },
        ),
        
        
        "room_small": AspectRule(
            aspect_code="room_small",
            polarity_hint="negative",
            display="Маленькая площадь номера / теснота",
            display_short="маленький номер",
            long_hint="Гости сообщают, что номер маленький и воспринимается тесным: ограниченное пространство для передвижения, визуально «давит», ощущение стеснённости.",
            patterns_by_lang={{
                "ru": [
                    r"\bочень маленький номер\b",
                    r"\bномер маленький\b",
                    r"\bтесно в номере\b",
                    r"\bочень тесно\b",
                    r"\bнет пространства\b",
                ],
                "en": [
                    r"\bsmall room\b",
                    r"\bthe room was tiny\b",
                    r"\bvery cramped room\b",
                    r"\bfelt cramped\b",
                    r"\bnot much space in the room\b",
                ],
                "tr": [
                    r"\boda çok küçüktü\b",
                    r"\boda aşırı dardı\b",
                    r"\bçok sıkışıktı\b",
                    r"\bhareket edecek alan yoktu\b",
                ],
                "ar": [
                    r"\bالأوضة صغيرة قوي\b",
                    r"\bالمساحة ديقة\b",
                    r"\bمش لاقيين مساحة نتحرك\b",
                    r"\bالأوضة خانقة شوية\b",
                ],
                "zh": [
                    r"房间很小",
                    r"房间特别小很挤",
                    r"空间很拥挤",
                    r"活动空间很有限",
                ],
            },
        ),
        
        
        "no_space_for_luggage": AspectRule(
            aspect_code="no_space_for_luggage",
            polarity_hint="negative",
            display="Номеру не хватает места под багаж",
            display_short="некуда поставить чемоданы",
            long_hint="Гости фиксируют, что в номере трудно разместить багаж из-за ограниченной площади: чемоданы приходится держать на полу, они блокируют проход, невозможно разложить вещи.",
            patterns_by_lang={{
                "ru": [
                    r"\bнекуда поставить чемоданы\b",
                    r"\bне было места для багажа\b",
                    r"\bчемоданы мешали проходить\b",
                    r"\bбагаж приходилось держать посреди номера\b",
                    r"\bсумки лежали на полу, по ним приходилось шагать\b",
                ],
                "en": [
                    r"\bno space for luggage\b",
                    r"\bno place for suitcases\b",
                    r"\bhad to keep our luggage on the floor\b",
                    r"\bour bags were in the way\b",
                    r"\bthe suitcases blocked the walkway\b",
                ],
                "tr": [
                    r"\bvaliz koyacak yer yoktu\b",
                    r"\bçantalar hep yerdeydi\b",
                    r"\bvaliz yüzünden yürümek zor oldu\b",
                    r"\bbagaj için alan yok\b",
                ],
                "ar": [
                    r"\bمافيش مكان للشنط\b",
                    r"\bالشنط لازم على الأرض في النص\b",
                    r"\bالشنط عاملة زحمة في الأوضة\b",
                    r"\bمفيش مساحة نفتح الشنطة حتى\b",
                ],
                "zh": [
                    r"没有地方放行李箱",
                    r"行李只能放在地上中间",
                    r"行李挡住了走道",
                    r"房间小到箱子都没处放",
                ],
            },
        ),
        
        
        "dark_room": AspectRule(
            aspect_code="dark_room",
            polarity_hint="negative",
            display="Низкий уровень освещения в номере",
            display_short="тёмный номер",
            long_hint="Гости отмечают, что в номере темно: недостаточно естественного света и/или искусственное освещение слабое, создаётся визуально утомляющая атмосфера.",
            patterns_by_lang={{
                "ru": [
                    r"\bтемный номер\b",
                    r"\bв номере очень темно\b",
                    r"\bмало света\b",
                    r"\bплохое освещение\b",
                    r"\bномер какой-то мрачный\b",
                ],
                "en": [
                    r"\bdark room\b",
                    r"\bthe room was very dark\b",
                    r"\bpoor lighting\b",
                    r"\bnot enough light in the room\b",
                    r"\bthe lighting was too dim\b",
                ],
                "tr": [
                    r"\boda çok karanlıktı\b",
                    r"\byetersiz aydınlatma\b",
                    r"\boda loştu hep\b",
                    r"\bodada neredeyse ışık yoktu\b",
                ],
                "ar": [
                    r"\bالأوضة معتمة\b",
                    r"\bالإضاءة ضعيفة جداً\b",
                    r"\bمافيش نور كفاية في الأوضة\b",
                    r"\bالجو معتم شوية في الأوضة\b",
                ],
                "zh": [
                    r"房间很暗",
                    r"房间灯光很暗",
                    r"采光不好 房间很阴",
                    r"灯光太昏暗了",
                ],
            },
        ),

        "no_natural_light": AspectRule(
            aspect_code="no_natural_light",
            polarity_hint="negative",
            display="Отсутствие естественного света",
            display_short="нет дневного света",
            long_hint="Гости отмечают, что в номере практически нет естественного освещения: окна маленькие, выходят во двор-колодец или свет перекрыт, из-за чего номер воспринимается мрачным.",
            patterns_by_lang={{
                "ru": [
                    r"\bнет естественного света\b",
                    r"\bпочти нет дневного света\b",
                    r"\bокно почти не даёт свет\b",
                    r"\bтемно даже днём\b",
                    r"\bномер без окон\b",
                ],
                "en": [
                    r"\bno natural light\b",
                    r"\bvery little daylight\b",
                    r"\bno sunlight in the room\b",
                    r"\bthe room had no windows\b",
                    r"\bdark even during the day\b",
                ],
                "tr": [
                    r"\bdoğal ışık yoktu\b",
                    r"\bgün ışığı neredeyse yoktu\b",
                    r"\boda güneş almıyordu\b",
                    r"\bpenceresiz gibi karanlık\b",
                ],
                "ar": [
                    r"\bمافيش نور طبيعي\b",
                    r"\bالغرفة من غير شمس\b",
                    r"\bمعتمة حتى بالنهار\b",
                    r"\bكأنها من غير شبابيك\b",
                ],
                "zh": [
                    r"没有自然光",
                    r"白天也很暗",
                    r"基本照不到阳光",
                    r"房间几乎没有窗户光线",
                ],
            },
        ),
        
        
        "hot_water_ok": AspectRule(
            aspect_code="hot_water_ok",
            polarity_hint="positive",
            display="Стабильная горячая вода",
            display_short="горячая вода стабильна",
            long_hint="Гости отмечают, что горячая вода есть постоянно: быстро появляется, держит температуру, хватает на душ без перепадов.",
            patterns_by_lang={{
                "ru": [
                    r"\bгорячая вода без проблем\b",
                    r"\bгорячая вода сразу шла\b",
                    r"\bвсегда была горячая вода\b",
                    r"\bвода быстро нагревается\b",
                    r"\bтемпература воды стабильная\b",
                ],
                "en": [
                    r"\bplenty of hot water\b",
                    r"\bhot water was always available\b",
                    r"\bhot water came on immediately\b",
                    r"\bstable hot water temperature\b",
                    r"\bno issues with hot water\b",
                ],
                "tr": [
                    r"\bsıcak su hep vardı\b",
                    r"\bsu hemen ısınıyordu\b",
                    r"\bsıcak su sorunsuzdu\b",
                    r"\bduşta sıcak su hiç kesilmedi\b",
                ],
                "ar": [
                    r"\bفي ميّة سخنة طول الوقت\b",
                    r"\bالمياه السخنة جاهزة على طول\b",
                    r"\bمفيش مشكلة في الميّة السخنة\b",
                    r"\bالحرارة ثابتة في الدش\b",
                ],
                "zh": [
                    r"热水很稳定",
                    r"洗澡热水一直有",
                    r"热水来得很快",
                    r"水温保持得很好",
                ],
            },
        ),
        
        
        "water_pressure_ok": AspectRule(
            aspect_code="water_pressure_ok",
            polarity_hint="positive",
            display="Достаточное давление воды",
            display_short="нормальное давление воды",
            long_hint="Гости отмечают, что напор воды в душе и кранах комфортный: не слишком слабый, можно комфортно принимать душ, вода идёт ровно.",
            patterns_by_lang={{
                "ru": [
                    r"\bнормальный напор\b",
                    r"\bхорошее давление воды\b",
                    r"\bсильный напор в душе\b",
                    r"\bнапор стабильный\b",
                    r"\bвода шла отлично\b",
                ],
                "en": [
                    r"\bgood water pressure\b",
                    r"\bstrong shower pressure\b",
                    r"\bthe water pressure was great\b",
                    r"\bsteady pressure in the shower\b",
                    r"\bpressure was fine\b",
                ],
                "tr": [
                    r"\bsu basıncı iyiydi\b",
                    r"\bduşta basınç güzeldi\b",
                    r"\bsu tazyiki yeterliydi\b",
                    r"\bmuslukta güzel akıyordu\b",
                ],
                "ar": [
                    r"\bضغط الميّة كويس\b",
                    r"\bالدوش ضغطه حلو\b",
                    r"\bالمياه طالعة بضغط كويس\b",
                    r"\bمافيش ضعف في الضغط\b",
                ],
                "zh": [
                    r"水压很好",
                    r"淋浴水压很足",
                    r"水压稳定",
                    r"水流很顺",
                ],
            },
        ),
        
        
        "shower_ok": AspectRule(
            aspect_code="shower_ok",
            polarity_hint="positive",
            display="Работа душа без нареканий",
            display_short="душ в норме",
            long_hint="Гости отмечают, что душевая система исправна: температура регулируется, не течёт мимо, ничего не капает на пол, пользоваться удобно.",
            patterns_by_lang={{
                "ru": [
                    r"\bдуш работал нормально\b",
                    r"\bдуш отличный\b",
                    r"\bдуш всё ок\b",
                    r"\bдуш без проблем\b",
                    r"\bхороший душ\b",
                ],
                "en": [
                    r"\bshower worked fine\b",
                    r"\bshower was good\b",
                    r"\bno problems with the shower\b",
                    r"\bshower was easy to use\b",
                    r"\bwater in the shower was perfect\b",
                ],
                "tr": [
                    r"\bduş gayet iyiydi\b",
                    r"\bduşta sorun yoktu\b",
                    r"\bduş güzel çalışıyordu\b",
                    r"\bduş kullanımı rahattı\b",
                ],
                "ar": [
                    r"\bالدش شغال كويس\b",
                    r"\bمفيش مشاكل في الدش\b",
                    r"\bالدش تمام ومريح\b",
                    r"\bالدش مضبوط وميّته تمام\b",
                ],
                "zh": [
                    r"淋浴一切正常",
                    r"淋浴很好用",
                    r"洗澡的水温很好调",
                    r"淋浴没有问题",
                ],
            },
        ),
        
        
        "no_leak": AspectRule(
            aspect_code="no_leak",
            polarity_hint="positive",
            display="Отсутствие протечек / сухой санузел",
            display_short="ничего не течёт",
            long_hint="Гости отмечают, что ничего не протекает: душевая кабина не течёт на пол, трубы не капают, санузел остаётся сухим после использования.",
            patterns_by_lang={{
                "ru": [
                    r"\bничего не текло\b",
                    r"\bничего не протекало\b",
                    r"\bсантехника не подкапывала\b",
                    r"\bпол оставался сухим после душа\b",
                    r"\bнет протечек\b",
                ],
                "en": [
                    r"\bno leaks\b",
                    r"\bno leaking in the bathroom\b",
                    r"\bthe shower didn't leak\b",
                    r"\bno dripping pipes\b",
                    r"\bthe floor stayed dry\b",
                ],
                "tr": [
                    r"\bsızıntı yoktu\b",
                    r"\bduş sonrası yer kuru kaldı\b",
                    r"\bborulardan damlama yoktu\b",
                    r"\bhiç su kaçırmıyordu\b",
                ],
                "ar": [
                    r"\bمافيش تسريب ميّة\b",
                    r"\bالدش ما بينقطش برا\b",
                    r"\bالأرض فضلت ناشفة بعد الشاور\b",
                    r"\bالحمام ما بيهرّبش ميّة\b",
                ],
                "zh": [
                    r"没有漏水",
                    r"浴室没有渗水",
                    r"淋浴不会漏到外面",
                    r"地面保持干燥没有渗漏",
                ],
            },
        ),

        "no_hot_water": AspectRule(
            aspect_code="no_hot_water",
            polarity_hint="negative",
            display="Отсутствие горячей воды",
            display_short="нет горячей воды",
            long_hint="Гости фиксируют отсутствие горячей воды или значительные проблемы с нагревом: вода остаётся холодной или еле тёплой, невозможно принять нормальный душ.",
            patterns_by_lang={{
                "ru": [
                    r"\bне было горячей воды\b",
                    r"\bнет горячей воды\b",
                    r"\bтолько холодная вода\b",
                    r"\bвода еле тёплая\b",
                    r"\bвода не нагревалась\b",
                ],
                "en": [
                    r"\bno hot water\b",
                    r"\bthere was no hot water\b",
                    r"\bonly cold water\b",
                    r"\bthe water was barely warm\b",
                    r"\bthe water never got hot\b",
                ],
                "tr": [
                    r"\bsıcak su yoktu\b",
                    r"\bsadece soğuk su vardı\b",
                    r"\bsu ısınmıyordu\b",
                    r"\bsu ılık bile değildi\b",
                ],
                "ar": [
                    r"\bمافيش ميّة سخنة\b",
                    r"\bالميّة كانت ساقعة بس\b",
                    r"\bالمياه ما بتسخنش\b",
                    r"\bالميّة طلعت بس باردة\b",
                ],
                "zh": [
                    r"没有热水",
                    r"只有冷水",
                    r"水根本不热",
                    r"洗澡水都是凉的",
                ],
            },
        ),
        
        
        "weak_pressure": AspectRule(
            aspect_code="weak_pressure",
            polarity_hint="negative",
            display="Слабый напор воды",
            display_short="слабый напор",
            long_hint="Гости отмечают, что давление воды слишком низкое: душ идёт тонкой струйкой, невозможно нормально помыться и занимает слишком много времени.",
            patterns_by_lang={{
                "ru": [
                    r"\bслабый напор\b",
                    r"\bочень слабое давление воды\b",
                    r"\bв душе еле течёт\b",
                    r"\bвода еле льется\b",
                    r"\bвода капает а не течет\b",
                ],
                "en": [
                    r"\bweak water pressure\b",
                    r"\bvery low water pressure\b",
                    r"\bthe shower pressure was terrible\b",
                    r"\bwater was just dripping\b",
                    r"\bthe water was barely running\b",
                ],
                "tr": [
                    r"\bsu basıncı çok düşüktü\b",
                    r"\bduşta su zorla akıyordu\b",
                    r"\bneredeyse damlıyordu sadece\b",
                    r"\bsu tazyiki çok zayıftı\b",
                ],
                "ar": [
                    r"\bضغط الميّة ضعيف جداً\b",
                    r"\bالدش بيقطّر بس\b",
                    r"\bالميّة بتنزل خفيف قوي\b",
                    r"\bمافيش ضغط في الدش\b",
                ],
                "zh": [
                    r"水压很小",
                    r"花洒水压很弱",
                    r"水几乎只是滴出来",
                    r"水流特别小",
                ],
            },
        ),
        
        
        "shower_broken": AspectRule(
            aspect_code="shower_broken",
            polarity_hint="negative",
            display="Неисправный душ",
            display_short="душ сломан",
            long_hint="Гости сообщают, что душ не работал как должен: лейка сломана, держатель болтается, переключатель не работает, вода льётся куда не нужно или вообще не идёт.",
            patterns_by_lang={{
                "ru": [
                    r"\bдуш не работал\b",
                    r"\bсломанный душ\b",
                    r"\bдуш сломан\b",
                    r"\bполоманная лейка\b",
                    r"\bдержатель душа не держит\b",
                    r"\bне работал переключатель душа\b",
                ],
                "en": [
                    r"\bthe shower didn't work\b",
                    r"\bbroken shower\b",
                    r"\bthe shower head was broken\b",
                    r"\bthe shower holder was broken\b",
                    r"\bthe shower wouldn't turn on properly\b",
                ],
                "tr": [
                    r"\bduş çalışmıyordu\b",
                    r"\bduş bozuktu\b",
                    r"\bduş başlığı kırıktı\b",
                    r"\bduş tutucusu kırılmıştı\b",
                    r"\bduşu açamadık düzgün\b",
                ],
                "ar": [
                    r"\bالدش بايظ\b",
                    r"\bالدش ما اشتغلش كويس\b",
                    r"\bرأس الدش مكسور\b",
                    r"\bالحامل بتاع الدش مكسور\b",
                ],
                "zh": [
                    r"花洒坏了",
                    r"淋浴不能正常用",
                    r"花洒头是坏的",
                    r"淋浴开关坏了打不开正常水",
                ],
            },
        ),
        
        
        "leak_water": AspectRule(
            aspect_code="leak_water",
            polarity_hint="negative",
            display="Протечки воды",
            display_short="течёт вода",
            long_hint="Гости фиксируют, что что-то протекает: кран подкапывает, труба течёт, вода просачивается из душевой кабины — требуется вмешательство технической службы.",
            patterns_by_lang={{
                "ru": [
                    r"\bвода текла\b",
                    r"\bтек кран\b",
                    r"\bтекла труба\b",
                    r"\bкапало из трубы\b",
                    r"\bдуш протекал на пол\b",
                ],
                "en": [
                    r"\bwater was leaking\b",
                    r"\bleaking tap\b",
                    r"\bleaking pipe\b",
                    r"\bthe shower was leaking onto the floor\b",
                    r"\bconstant drip from the faucet\b",
                ],
                "tr": [
                    r"\bsu sızdırıyordu\b",
                    r"\bmusluk damlıyordu\b",
                    r"\bboru su kaçırıyordu\b",
                    r"\bduştan dışarı su akıyordu\b",
                ],
                "ar": [
                    r"\bالميّة بتسرب\b",
                    r"\bالحنفية بتقطّر طول الوقت\b",
                    r"\bالماسورة بتسرب ميّة\b",
                    r"\bالدش بيطلع ميّة على الأرض\b",
                ],
                "zh": [
                    r"有漏水",
                    r"水龙头一直滴水",
                    r"管子在漏水",
                    r"淋浴间往外漏水到地上",
                ],
            },
        ),
        
        
        "bathroom_flooding": AspectRule(
            aspect_code="bathroom_flooding",
            polarity_hint="negative",
            display="Затопление санузла / вода на полу",
            display_short="вода на полу в ванной",
            long_hint="Гости сообщают, что после душа или из-за неисправности сантехники пол в ванной оказывается в воде: сливы не справляются, вода уходит медленно или поднимается обратно.",
            patterns_by_lang={{
                "ru": [
                    r"\bванная была вся в воде\b",
                    r"\bвода на полу после душа\b",
                    r"\bзаливало ванну\b",
                    r"\bпол был мокрый и стояла вода\b",
                    r"\bвода не уходила и всё затопило\b",
                ],
                "en": [
                    r"\bwater all over the bathroom floor\b",
                    r"\bbathroom flooded\b",
                    r"\bfloor was flooded after shower\b",
                    r"\bwater pooling on the bathroom floor\b",
                    r"\bthe bathroom filled with water\b",
                ],
                "tr": [
                    r"\bbanyo su bastı\b",
                    r"\bduştan sonra yerler göl gibiydi\b",
                    r"\bbanyonun zemini su içindeydi\b",
                    r"\bsu tahliye olmadı, banyo doldu\b",
                ],
                "ar": [
                    r"\bالحمام غرق ميّة\b",
                    r"\bالأرض في الحمام كلها ميّة بعد الشاور\b",
                    r"\bالمياه ما نزلتش وملت الأرض\b",
                    r"\bالمصرف ما كانش بيصرف كويس\b",
                ],
                "zh": [
                    r"洗手间地上全是水",
                    r"浴室被水淹了",
                    r"洗完澡地面像积水了一层",
                    r"地漏不下水整个卫生间都是水",
                ],
            },
        ),

        "no_hot_water": AspectRule(
            aspect_code="no_hot_water",
            polarity_hint="negative",
            display="Отсутствие горячей воды",
            display_short="нет горячей воды",
            long_hint="Гости фиксируют отсутствие горячей воды или значительные проблемы с нагревом: вода остаётся холодной или еле тёплой, невозможно принять нормальный душ.",
            patterns_by_lang={{
                "ru": [
                    r"\bне было горячей воды\b",
                    r"\bнет горячей воды\b",
                    r"\bтолько холодная вода\b",
                    r"\bвода еле тёплая\b",
                    r"\bвода не нагревалась\b",
                ],
                "en": [
                    r"\bno hot water\b",
                    r"\bthere was no hot water\b",
                    r"\bonly cold water\b",
                    r"\bthe water was barely warm\b",
                    r"\bthe water never got hot\b",
                ],
                "tr": [
                    r"\bsıcak su yoktu\b",
                    r"\bsadece soğuk su vardı\b",
                    r"\bsu ısınmıyordu\b",
                    r"\bsu ılık bile değildi\b",
                ],
                "ar": [
                    r"\bمافيش ميّة سخنة\b",
                    r"\bالميّة كانت ساقعة بس\b",
                    r"\bالمياه ما بتسخنش\b",
                    r"\bالميّة طلعت بس باردة\b",
                ],
                "zh": [
                    r"没有热水",
                    r"只有冷水",
                    r"水根本不热",
                    r"洗澡水都是凉的",
                ],
            },
        ),
        
        
        "weak_pressure": AspectRule(
            aspect_code="weak_pressure",
            polarity_hint="negative",
            display="Слабый напор воды",
            display_short="слабый напор",
            long_hint="Гости отмечают, что давление воды слишком низкое: душ идёт тонкой струйкой, невозможно нормально помыться и занимает слишком много времени.",
            patterns_by_lang={{
                "ru": [
                    r"\bслабый напор\b",
                    r"\bочень слабое давление воды\b",
                    r"\bв душе еле течёт\b",
                    r"\bвода еле льется\b",
                    r"\bвода капает а не течет\b",
                ],
                "en": [
                    r"\bweak water pressure\b",
                    r"\bvery low water pressure\b",
                    r"\bthe shower pressure was terrible\b",
                    r"\bwater was just dripping\b",
                    r"\bthe water was barely running\b",
                ],
                "tr": [
                    r"\bsu basıncı çok düşüktü\b",
                    r"\bduşta su zorla akıyordu\b",
                    r"\bneredeyse damlıyordu sadece\b",
                    r"\bsu tazyiki çok zayıftı\b",
                ],
                "ar": [
                    r"\bضغط الميّة ضعيف جداً\b",
                    r"\bالدش بيقطّر بس\b",
                    r"\bالميّة بتنزل خفيف قوي\b",
                    r"\bمافيش ضغط في الدش\b",
                ],
                "zh": [
                    r"水压很小",
                    r"花洒水压很弱",
                    r"水几乎只是滴出来",
                    r"水流特别小",
                ],
            },
        ),
        
        
        "shower_broken": AspectRule(
            aspect_code="shower_broken",
            polarity_hint="negative",
            display="Неисправный душ",
            display_short="душ сломан",
            long_hint="Гости сообщают, что душ не работал как должен: лейка сломана, держатель болтается, переключатель не работает, вода льётся куда не нужно или вообще не идёт.",
            patterns_by_lang={{
                "ru": [
                    r"\bдуш не работал\b",
                    r"\bсломанный душ\b",
                    r"\bдуш сломан\b",
                    r"\bполоманная лейка\b",
                    r"\bдержатель душа не держит\b",
                    r"\bне работал переключатель душа\b",
                ],
                "en": [
                    r"\bthe shower didn't work\b",
                    r"\bbroken shower\b",
                    r"\bthe shower head was broken\b",
                    r"\bthe shower holder was broken\b",
                    r"\bthe shower wouldn't turn on properly\b",
                ],
                "tr": [
                    r"\bduş çalışmıyordu\b",
                    r"\bduş bozuktu\b",
                    r"\bduş başlığı kırıktı\b",
                    r"\bduş tutucusu kırılmıştı\b",
                    r"\bduşu açamadık düzgün\b",
                ],
                "ar": [
                    r"\bالدش بايظ\b",
                    r"\bالدش ما اشتغلش كويس\b",
                    r"\bرأس الدش مكسور\b",
                    r"\bالحامل بتاع الدش مكسور\b",
                ],
                "zh": [
                    r"花洒坏了",
                    r"淋浴不能正常用",
                    r"花洒头是坏的",
                    r"淋浴开关坏了打不开正常水",
                ],
            },
        ),
        
        
        "leak_water": AspectRule(
            aspect_code="leak_water",
            polarity_hint="negative",
            display="Протечки воды",
            display_short="течёт вода",
            long_hint="Гости фиксируют, что что-то протекает: кран подкапывает, труба течёт, вода просачивается из душевой кабины — требуется вмешательство технической службы.",
            patterns_by_lang={{
                "ru": [
                    r"\bвода текла\b",
                    r"\bтек кран\b",
                    r"\bтекла труба\b",
                    r"\bкапало из трубы\b",
                    r"\bдуш протекал на пол\b",
                ],
                "en": [
                    r"\bwater was leaking\b",
                    r"\bleaking tap\b",
                    r"\bleaking pipe\b",
                    r"\bthe shower was leaking onto the floor\b",
                    r"\bconstant drip from the faucet\b",
                ],
                "tr": [
                    r"\bsu sızdırıyordu\b",
                    r"\bmusluk damlıyordu\b",
                    r"\bboru su kaçırıyordu\b",
                    r"\bduştan dışarı su akıyordu\b",
                ],
                "ar": [
                    r"\bالميّة بتسرب\b",
                    r"\bالحنفية بتقطّر طول الوقت\b",
                    r"\bالماسورة بتسرب ميّة\b",
                    r"\bالدش بيطلع ميّة على الأرض\b",
                ],
                "zh": [
                    r"有漏水",
                    r"水龙头一直滴水",
                    r"管子在漏水",
                    r"淋浴间往外漏水到地上",
                ],
            },
        ),
        
        
        "bathroom_flooding": AspectRule(
            aspect_code="bathroom_flooding",
            polarity_hint="negative",
            display="Затопление санузла / вода на полу",
            display_short="вода на полу в ванной",
            long_hint="Гости сообщают, что после душа или из-за неисправности сантехники пол в ванной оказывается в воде: сливы не справляются, вода уходит медленно или поднимается обратно.",
            patterns_by_lang={{
                "ru": [
                    r"\bванная была вся в воде\b",
                    r"\bвода на полу после душа\b",
                    r"\bзаливало ванну\b",
                    r"\bпол был мокрый и стояла вода\b",
                    r"\bвода не уходила и всё затопило\b",
                ],
                "en": [
                    r"\bwater all over the bathroom floor\b",
                    r"\bbathroom flooded\b",
                    r"\bfloor was flooded after shower\b",
                    r"\bwater pooling on the bathroom floor\b",
                    r"\bthe bathroom filled with water\b",
                ],
                "tr": [
                    r"\bbanyo su bastı\b",
                    r"\bduştan sonra yerler göl gibiydi\b",
                    r"\bbanyonun zemini su içindeydi\b",
                    r"\bsu tahliye olmadı, banyo doldu\b",
                ],
                "ar": [
                    r"\bالحمام غرق ميّة\b",
                    r"\bالأرض في الحمام كلها ميّة بعد الشاور\b",
                    r"\bالمياه ما نزلتش وملت الأرض\b",
                    r"\bالمصرف ما كانش بيصرف كويس\b",
                ],
                "zh": [
                    r"洗手间地上全是水",
                    r"浴室被水淹了",
                    r"洗完澡地面像积水了一层",
                    r"地漏不下水整个卫生间都是水",
                ],
            },
        ),

        "drain_clogged": AspectRule(
            aspect_code="drain_clogged",
            polarity_hint="negative",
            display="Проблемы со сливом воды",
            display_short="слив не уходит",
            long_hint="Гости фиксируют, что вода плохо уходит в слив: душ, раковина или раковина в ванной забиваются, вода скапливается и стоит, образуя лужу.",
            patterns_by_lang={{
                "ru": [
                    r"\bзасоренный слив\b",
                    r"\bзасорен слив\b",
                    r"\bвода не уходит\b",
                    r"\bслив забит\b",
                    r"\bвода стоит в душе\b",
                    r"\bвода долго не уходит из раковины\b",
                ],
                "en": [
                    r"\bclogged drain\b",
                    r"\bthe drain was clogged\b",
                    r"\bwater was not draining\b",
                    r"\bthe water didn't go down\b",
                    r"\bstanding water in the shower\b",
                    r"\bthe sink was draining very slowly\b",
                ],
                "tr": [
                    r"\bgider tıkalıydı\b",
                    r"\bsu gitmiyordu\b",
                    r"\bduşta su birikti\b",
                    r"\blavabo suyu çekmedi\b",
                    r"\b gider çok yavaş boşalıyordu\b",
                ],
                "ar": [
                    r"\bالبلاعة مسدودة\b",
                    r"\bالميّة ما بتنزلش\b",
                    r"\bالمياه واقفة في الدش\b",
                    r"\bالحوض مش بيصرف\b",
                ],
                "zh": [
                    r"下水道堵了",
                    r"水下不去",
                    r"淋浴间积水下不走",
                    r"洗手池排水很慢",
                ],
            },
        ),
        
        
        "drain_smell": AspectRule(
            aspect_code="drain_smell",
            polarity_hint="negative",
            display="Неприятный запах из слива",
            display_short="запах из слива",
            long_hint="Гости сообщают о выразительном неприятном запахе, исходящем из слива душа, раковины или унитаза, даже после проветривания/уборки.",
            patterns_by_lang={{
                "ru": [
                    r"\bзапах из слива\b",
                    r"\bвоняло из слива\b",
                    r"\bпахло из раковины\b",
                    r"\bзапах из душевого трапа\b",
                    r"\bканализационный запах шёл из стока\b",
                ],
                "en": [
                    r"\bbad smell from the drain\b",
                    r"\bsewage smell from the drain\b",
                    r"\bthe sink smelled bad\b",
                    r"\bbad odor coming from the shower drain\b",
                    r"\bsmell coming up from the pipes\b",
                ],
                "tr": [
                    r"\bgiderden kötü koku geliyordu\b",
                    r"\bduş giderinden lağım kokusu geliyordu\b",
                    r"\blavabodan kötü koku vardı\b",
                    r"\bborulardan koku çıkıyordu\b",
                ],
                "ar": [
                    r"\bريحة طالعة من البلاعة\b",
                    r"\bريحة مجاري من الصرف\b",
                    r"\bريحة وحشة طالعة من الحوض\b",
                    r"\bريحة طالعة من المصرف في الدش\b",
                ],
                "zh": [
                    r"下水道有臭味",
                    r"排水口有很难闻的味道",
                    r"洗手池往上冒臭味",
                    r"淋浴下水口有下水道味",
                ],
            },
        ),
        
        
        "ac_working_device": AspectRule(
            aspect_code="ac_working_device",
            polarity_hint="positive",
            display="Исправность климатической техники (кондиционер)",
            display_short="кондиционер исправен",
            long_hint="Гости подтверждают, что установленный кондиционер как устройство исправен: включается, реагирует на пульт, меняет режимы и поддерживает нужную температуру.",
            patterns_by_lang={{
                "ru": [
                    r"\bкондиционер исправно работал\b",
                    r"\bкондиционер включался без проблем\b",
                    r"\bкондиционер функционировал нормально\b",
                    r"\bкондиционер реагировал на пульт\b",
                ],
                "en": [
                    r"\bthe AC unit worked properly\b",
                    r"\bthe aircon worked fine\b",
                    r"\bAC was functioning with no issues\b",
                    r"\bthe air conditioning unit worked well\b",
                ],
                "tr": [
                    r"\bklima sorunsuz çalışıyordu\b",
                    r"\bklima kumandaya hemen tepki veriyordu\b",
                    r"\bklima düzgün çalışıyordu\b",
                ],
                "ar": [
                    r"\bالتكييف شغال كويس كجهاز\b",
                    r"\bالتكييف بيستجيب للريموت\b",
                    r"\bجهاز التكييف سليم ومظبوط\b",
                ],
                "zh": [
                    r"空调运转正常",
                    r"空调反应正常",
                    r"空调机器状态很好",
                    r"空调功能都正常",
                ],
            },
        ),
        
        
        "heating_working_device": AspectRule(
            aspect_code="heating_working_device",
            polarity_hint="positive",
            display="Исправность обогревательного оборудования",
            display_short="обогреватель работает",
            long_hint="Гости указывают, что индивидуальные обогревательные приборы (радиатор, конвектор, обогреватель) работали без сбоев и давали достаточно тепла.",
            patterns_by_lang={{
                "ru": [
                    r"\bобогреватель работал\b",
                    r"\bобогреватель грел нормально\b",
                    r"\bрадиатор работал без проблем\b",
                    r"\bконвектор хорошо грел\b",
                ],
                "en": [
                    r"\bthe heater worked fine\b",
                    r"\bportable heater worked well\b",
                    r"\bthe radiator was working\b",
                    r"\bthe heating unit was heating properly\b",
                ],
                "tr": [
                    r"\bısıtıcı sorunsuz çalışıyordu\b",
                    r"\bradyatör çalışıyordu ve gayet ısıtıyordu\b",
                    r"\bokulabilir taşınabilir ısıtıcı iş gördü\b",
                ],
                "ar": [
                    r"\bالدفاية كانت شغالة كويس\b",
                    r"\bالسخان/الدفاية دافى الأوضة كويس\b",
                    r"\bالرادياتير كان شغال ومظبوط\b",
                ],
                "zh": [
                    r"取暖器工作正常",
                    r"电暖气加热效果很好",
                    r"暖气片有热量",
                    r"加热设备工作没问题",
                ],
            },
        ),
        
        
        "appliances_ok": AspectRule(
            aspect_code="appliances_ok",
            polarity_hint="positive",
            display="Исправность техники и оснащения номера",
            display_short="техника исправна",
            long_hint="Гости отмечают, что бытовая и гостиничная техника (ТВ, холодильник, чайник, освещение, замки и т.д.) в номере в рабочем состоянии и не вызывала эксплуатационных проблем.",
            patterns_by_lang={{
                "ru": [
                    r"\bвся техника работала\b",
                    r"\bвсё оборудование в порядке\b",
                    r"\bвсё работало как должно\b",
                    r"\bничего не было сломано\b",
                    r"\bоборудование исправное\b",
                ],
                "en": [
                    r"\beverything in the room worked\b",
                    r"\ball appliances worked fine\b",
                    r"\ball the equipment was working\b",
                    r"\bnothing was broken in the room\b",
                    r"\bthe room equipment was in good condition\b",
                ],
                "tr": [
                    r"\bodadaki her şey çalışıyordu\b",
                    r"\btüm cihazlar sorunsuzdu\b",
                    r"\bodadaki ekipman sağlamdı\b",
                    r"\bhiçbir şey bozuk değildi\b",
                ],
                "ar": [
                    r"\bكل الأجهزة كانت شغالة كويس\b",
                    r"\bمافيش حاجة مكسورة في الأوضة\b",
                    r"\bكل التجهيزات شغالة تمام\b",
                ],
                "zh": [
                    r"房间里的设备都能正常用",
                    r"所有电器都正常",
                    r"没有坏掉的设备",
                    r"房间设施状态良好",
                ],
            },
        ),
        
        
        "tv_working": AspectRule(
            aspect_code="tv_working",
            polarity_hint="positive",
            display="Работоспособность телевизора",
            display_short="телевизор работает",
            long_hint="Гости подтверждают, что телевизор работал без проблем: включался, ловил каналы/стриминговые сервисы, пульт был исправен.",
            patterns_by_lang={{
                "ru": [
                    r"\bтелевизор работал\b",
                    r"\bтв работал нормально\b",
                    r"\bтелевизор без проблем\b",
                    r"\bпульт от телевизора работал\b",
                    r"\bбыли каналы/стриминг доступен\b",
                ],
                "en": [
                    r"\bthe TV worked\b",
                    r"\bTV was working fine\b",
                    r"\bthe television worked with no issues\b",
                    r"\bremote worked\b",
                    r"\bsmart TV worked\b",
                ],
                "tr": [
                    r"\btelevizyon çalışıyordu\b",
                    r"\bTV sorunsuzdu\b",
                    r"\bkumanda çalışıyordu\b",
                    r"\bkanallar/netflix açıldı\b",
                ],
                "ar": [
                    r"\bالتلفزيون شغال كويس\b",
                    r"\bالريموت شغال\b",
                    r"\bالتلفزيون شغال من غير مشاكل\b",
                    r"\bقدرنا نتفرج عالقنوات/نتفليكس\b",
                ],
                "zh": [
                    r"电视可以正常用",
                    r"电视工作正常",
                    r"遥控器可以用",
                    r"智能电视功能正常",
                ],
            },
        ),

        "fridge_working": AspectRule(
            aspect_code="fridge_working",
            polarity_hint="positive",
            display="Работа холодильника / мини-бара",
            display_short="холодильник работает",
            long_hint="Гости отмечают, что холодильник в номере работает корректно: охлаждает напитки и продукты, поддерживает нужную температуру без шума и сбоев.",
            patterns_by_lang={{
                "ru": [
                    r"\bхолодильник работал\b",
                    r"\bмини\-бар работал\b",
                    r"\bхолодильник хорошо охлаждал\b",
                    r"\bнапитки были холодные в холодильнике\b",
                    r"\bс холодильником не было проблем\b",
                ],
                "en": [
                    r"\bthe fridge worked\b",
                    r"\bfridge was working fine\b",
                    r"\bthe minibar was working\b",
                    r"\bthe fridge kept everything cold\b",
                    r"\bno issues with the fridge\b",
                ],
                "tr": [
                    r"\bbuzdolabı çalışıyordu\b",
                    r"\bmini bar sorunsuzdu\b",
                    r"\biçecekler güzel soğuktu\b",
                    r"\bbuzdolabı iyi soğutuyordu\b",
                ],
                "ar": [
                    r"\bالتلاجة شغالة كويس\b",
                    r"\bالميني بار كان بيبرّد فعلاً\b",
                    r"\bالمشروبات طالعة ساقعة من التلاجة\b",
                    r"\bمفيش مشكلة في التلاجة\b",
                ],
                "zh": [
                    r"冰箱正常工作",
                    r"小冰箱很给力可以制冷",
                    r"饮料都冰的",
                    r"冰箱没问题",
                ],
            },
        ),
        
        
        "kettle_working": AspectRule(
            aspect_code="kettle_working",
            polarity_hint="positive",
            display="Исправный чайник / возможность вскипятить воду",
            display_short="чайник работает",
            long_hint="Гости подтверждают, что электрочайник или устройство для нагрева воды исправно работало: можно было быстро приготовить чай/кофе или подогреть воду (например, для ребёнка).",
            patterns_by_lang={{
                "ru": [
                    r"\bчайник работал\b",
                    r"\bэлектрочайник исправен\b",
                    r"\bвскипятить воду без проблем\b",
                    r"\bвода закипала быстро\b",
                ],
                "en": [
                    r"\bthe kettle worked\b",
                    r"\bworking electric kettle\b",
                    r"\bwe could boil water in the room\b",
                    r"\bthe kettle heated up quickly\b",
                ],
                "tr": [
                    r"\bketıl çalışıyordu\b",
                    r"\bsu ısıtıcısı sorunsuzdu\b",
                    r"\bsuyu hemen ısıttı\b",
                    r"\bodada rahatça su kaynattık\b",
                ],
                "ar": [
                    r"\bالكاتل شغال\b",
                    r"\bقدرنا نغلي ميّة في الأوضة\b",
                    r"\bالسخان بيغلي الميّة بسرعة\b",
                    r"\bالغلاية شغالة من غير مشاكل\b",
                ],
                "zh": [
                    r"烧水壶可以正常烧水",
                    r"热水壶工作正常",
                    r"烧水很快就开",
                    r"可以在房间里自己烧热水",
                ],
            },
        ),
        
        
        "door_secure": AspectRule(
            aspect_code="door_secure",
            polarity_hint="positive",
            display="Надёжность входной двери и запорного механизма",
            display_short="дверь надёжная",
            long_hint="Гости указывают, что входная дверь в номер ощущается надёжной: плотно закрывается, замок внушает доверие, нет ощущения, что можно легко вскрыть. Это влияет на субъективное чувство безопасности.",
            patterns_by_lang={{
                "ru": [
                    r"\bдверь надёжно закрывалась\b",
                    r"\bдверь плотно закрывается\b",
                    r"\bзамок хороший\b",
                    r"\bчувствовали себя в безопасности в номере\b",
                    r"\bникто не мог открыть дверь снаружи\b",
                ],
                "en": [
                    r"\bthe door felt secure\b",
                    r"\bsecure door lock\b",
                    r"\bthe door closed securely\b",
                    r"\bwe felt safe in the room\b",
                    r"\bstrong lock on the door\b",
                ],
                "tr": [
                    r"\bkapı güvenliydi\b",
                    r"\bkapı sağlam kapanıyordu\b",
                    r"\bkilit güven verdi\b",
                    r"\bodada kendimizi güvende hissettik\b",
                ],
                "ar": [
                    r"\bباب الأوضة آمن\b",
                    r"\bالقفل كويس ومأمّن\b",
                    r"\bالباب بيقفل بإحكام\b",
                    r"\bحاسّين أمان جوة الأوضة\b",
                ],
                "zh": [
                    r"房门很安全",
                    r"门锁很结实",
                    r"门关得很严实",
                    r"在房间里感觉很安全",
                ],
            },
        ),
        
        
        "ac_broken": AspectRule(
            aspect_code="ac_broken",
            polarity_hint="negative",
            display="Неисправный кондиционер",
            display_short="кондиционер сломан",
            long_hint="Гости сообщают, что кондиционер в номере неисправен: не включается, не реагирует на пульт, выдаёт ошибку или не охлаждает как заявлено.",
            patterns_by_lang={{
                "ru": [
                    r"\bкондиционер не работал\b",
                    r"\bкондиционер сломан\b",
                    r"\bкондиционер не включался\b",
                    r"\bкондиционер моргал ошибкой\b",
                    r"\bпульт не управлял кондиционером\b",
                ],
                "en": [
                    r"\bbroken AC\b",
                    r"\bthe AC was broken\b",
                    r"\bAC didn't turn on\b",
                    r"\bthe air conditioner wasn't working\b",
                    r"\bthe AC unit was not functioning\b",
                ],
                "tr": [
                    r"\bklima bozuktu\b",
                    r"\bklima hiç açılmadı\b",
                    r"\bklima çalışmıyordu\b",
                    r"\bklima hata veriyordu\b",
                ],
                "ar": [
                    r"\bالتكييف بايظ\b",
                    r"\bالـAC ما اشتغلش\b",
                    r"\bالتكييف مش شغال خالص\b",
                    r"\bالجهاز ما بيستجيبش\b",
                ],
                "zh": [
                    r"空调坏了",
                    r"空调打不起来",
                    r"空调不工作",
                    r"空调机故障",
                ],
            },
        ),
        
        
        "heating_broken": AspectRule(
            aspect_code="heating_broken",
            polarity_hint="negative",
            display="Неисправное отопительное оборудование / обогреватель не работает",
            display_short="отопление сломано",
            long_hint="Гости фиксируют, что система обогрева или переносной обогреватель не работают должным образом: не включаются, не греют, температура не меняется.",
            patterns_by_lang={{
                "ru": [
                    r"\bобогрев не работал\b",
                    r"\bотопление сломано\b",
                    r"\bобогреватель не включался\b",
                    r"\bобогреватель не грел\b",
                    r"\bрадиатор не работал\b",
                ],
                "en": [
                    r"\bbroken heater\b",
                    r"\bthe heating was broken\b",
                    r"\bthe heater did not work\b",
                    r"\bthe radiator wasn't working\b",
                    r"\bthe heater wouldn't turn on\b",
                ],
                "tr": [
                    r"\bısıtıcı bozuktu\b",
                    r"\bısıtma sistemi çalışmıyordu\b",
                    r"\bradyatör çalışmıyordu\b",
                    r"\bısıtıcı açılmadı\b",
                ],
                "ar": [
                    r"\bالدفاية بايظة\b",
                    r"\bالتدفئة مش شغالة\b",
                    r"\bالسخان/الدفاية ما بيدفيش\b",
                    r"\bالرادياتير ما اشتغلش\b",
                ],
                "zh": [
                    r"暖气坏了",
                    r"取暖器不工作",
                    r"开了也不热",
                    r"暖气完全不起作用",
                ],
            },
        ),

        "tv_broken": AspectRule(
            aspect_code="tv_broken",
            polarity_hint="negative",
            display="Неисправный телевизор",
            display_short="телевизор не работает",
            long_hint="Гости сообщают, что телевизор не работал должным образом: не включался, не ловил каналы, не работал Smart TV или пульт, невозможно было воспользоваться устройством по назначению.",
            patterns_by_lang={{
                "ru": [
                    r"\bтелевизор не работал\b",
                    r"\bтв не работал\b",
                    r"\bтелевизор не включался\b",
                    r"\bтелевизор ничего не показывал\b",
                    r"\bпульт от телевизора не работал\b",
                ],
                "en": [
                    r"\bthe TV didn't work\b",
                    r"\bbroken TV\b",
                    r"\bTV would not turn on\b",
                    r"\bTV had no signal\b",
                    r"\bthe remote didn't work\b",
                ],
                "tr": [
                    r"\btelevizyon çalışmıyordu\b",
                    r"\bTV bozuktu\b",
                    r"\bTV açılmadı\b",
                    r"\bkanal yoktu/çekmiyordu\b",
                    r"\bkumanda çalışmıyordu\b",
                ],
                "ar": [
                    r"\bالتلفزيون مش شغال\b",
                    r"\bالتلفزيون ما بيفتحش\b",
                    r"\bمفيش إشارة على التلفزيون\b",
                    r"\bالريموت مش شغال\b",
                ],
                "zh": [
                    r"电视打不开",
                    r"电视坏了",
                    r"电视没有信号",
                    r"遥控器不能用",
                ],
            },
        ),
        
        
        "fridge_broken": AspectRule(
            aspect_code="fridge_broken",
            polarity_hint="negative",
            display="Неисправный холодильник",
            display_short="холодильник не работает",
            long_hint="Гости фиксируют, что холодильник/мини-бар не выполнял свою функцию: не охлаждает, выключен, сильно шумит или обмерзает так, что пользоваться невозможно.",
            patterns_by_lang={{
                "ru": [
                    r"\bхолодильник не работал\b",
                    r"\bхолодильник сломан\b",
                    r"\bхолодильник не охлаждал\b",
                    r"\bмини\-бар не охлаждал\b",
                    r"\bхолодильник был выключен и не включался\b",
                ],
                "en": [
                    r"\bbroken fridge\b",
                    r"\bthe fridge didn't work\b",
                    r"\bthe minibar wasn't cooling\b",
                    r"\bthe fridge did not get cold\b",
                    r"\bfridge was off and wouldn't turn on\b",
                ],
                "tr": [
                    r"\bbuzdolabı çalışmıyordu\b",
                    r"\bbuzdolabı soğutmuyordu\b",
                    r"\bmini bar buz gibi yapmıyordu\b",
                    r"\bbuzdolabı bozuktu\b",
                ],
                "ar": [
                    r"\bالتلاجة مش شغالة\b",
                    r"\bالميني بار ما بيبردش\b",
                    r"\bالتلاجة ما بتسقّعش خالص\b",
                    r"\bالتلاجة بايظة\b",
                ],
                "zh": [
                    r"冰箱不工作",
                    r"小冰箱不制冷",
                    r"冰箱坏了",
                    r"冰箱根本不凉",
                ],
            },
        ),
        
        
        "kettle_broken": AspectRule(
            aspect_code="kettle_broken",
            polarity_hint="negative",
            display="Неисправный чайник / кипятильник",
            display_short="чайник не работает",
            long_hint="Гости сообщают, что чайник не функционировал: не нагревал воду, искрил, был сломан или использовать его было небезопасно.",
            patterns_by_lang={{
                "ru": [
                    r"\bчайник не работал\b",
                    r"\bэлектрочайник сломан\b",
                    r"\bчайник не кипятил воду\b",
                    r"\bчайник искрил\b",
                    r"\bчайник было страшно использовать\b",
                ],
                "en": [
                    r"\bthe kettle didn't work\b",
                    r"\bbroken kettle\b",
                    r"\bthe kettle wouldn't boil\b",
                    r"\bthe kettle was sparking\b",
                    r"\bthe kettle felt unsafe to use\b",
                ],
                "tr": [
                    r"\bketıl çalışmıyordu\b",
                    r"\bketıl suyu ısıtmadı\b",
                    r"\bketıl bozuktu\b",
                    r"\bısıtıcı arızalıydı\b",
                ],
                "ar": [
                    r"\bالكاتل مش شغال\b",
                    r"\bالكاتل ما بيسخنش الميّة\b",
                    r"\bالغلاية بايظة\b",
                    r"\bالغلاية شكلها مش آمن\b",
                ],
                "zh": [
                    r"热水壶坏了",
                    r"水壶烧不开水",
                    r"烧水壶不能用",
                    r"感觉水壶不安全",
                ],
            },
        ),
        
        
        "socket_danger": AspectRule(
            aspect_code="socket_danger",
            polarity_hint="negative",
            display="Опасные/повреждённые розетки и электрика",
            display_short="проблемы с электрикой",
            long_hint="Гости отмечают небезопасное состояние электрики в номере: болтающиеся розетки, оголённые провода, искрение, риск удара током.",
            patterns_by_lang={{
                "ru": [
                    r"\bрозетка искрила\b",
                    r"\bрозетки болтаются\b",
                    r"\bрозетка вырвана из стены\b",
                    r"\bоголённые провода\b",
                    r"\bопасная электрика\b",
                ],
                "en": [
                    r"\bunsafe socket\b",
                    r"\bthe socket was hanging out of the wall\b",
                    r"\bsparking outlet\b",
                    r"\bexposed wires\b",
                    r"\belectrical felt unsafe\b",
                ],
                "tr": [
                    r"\bpriz gevşekti\b",
                    r"\bpriz duvardan çıkmıştı\b",
                    r"\bpriz kıvılcım çıkarıyordu\b",
                    r"\bkablolar açıktaydı\b",
                    r"\belektrik tehlikeliydi\b",
                ],
                "ar": [
                    r"\bفيشة مش ثابتة في الحيطة\b",
                    r"\bالفيشة بتشرّر\b",
                    r"\bسلك الكهرباء باين ومكشوف\b",
                    r"\bالكهربا شكلها مش آمن\b",
                ],
                "zh": [
                    r"插座松动",
                    r"插座从墙上掉出来",
                    r"插座有火花",
                    r"有裸露电线",
                    r"房间电路感觉不安全",
                ],
            },
        ),
        
        
        "door_not_closing": AspectRule(
            aspect_code="door_not_closing",
            polarity_hint="negative",
            display="Дверь не закрывается / не прижимается плотно",
            display_short="дверь неплотно закрывается",
            long_hint="Гости фиксируют, что дверь номера или входная дверь плохо закрывается, не фиксируется замком или остаётся приоткрытой, что влияет и на приватность, и на ощущение безопасности.",
            patterns_by_lang={{
                "ru": [
                    r"\bдверь плохо закрывалась\b",
                    r"\bдверь не закрывается до конца\b",
                    r"\bдверь не фиксировалась\b",
                    r"\bдверь не держится на замке\b",
                    r"\bдверь оставалась приоткрытой\b",
                ],
                "en": [
                    r"\bthe door didn't close properly\b",
                    r"\bthe door wouldn't shut fully\b",
                    r"\bthe door didn't lock properly\b",
                    r"\bthe door kept opening\b",
                    r"\bit didn't feel secure because the door wouldn't close\b",
                ],
                "tr": [
                    r"\bkapı tam kapanmıyordu\b",
                    r"\bkapı sürekli aralık kalıyordu\b",
                    r"\bkapı kilitlenmiyordu\b",
                    r"\bkapı güvenli kapanmıyordu\b",
                ],
                "ar": [
                    r"\bالباب ما بيقفّلش كويس\b",
                    r"\bالباب مابيقفلش للآخر\b",
                    r"\bالقفل مش ماسك كويس\b",
                    r"\bالباب بيفضل مفتوح شوية\b",
                ],
                "zh": [
                    r"门关不严",
                    r"门关不上",
                    r"门老是自己开",
                    r"门锁不上感觉不安全",
                ],
            },
        ),
    
        "lock_broken": AspectRule(
            aspect_code="lock_broken",
            polarity_hint="negative",
            display="Неисправный замок двери",
            display_short="замок сломан",
            long_hint="Гости сообщают, что замок на двери не работает корректно: не фиксируется, не закрывается на ключ/карту, заедает или легко открывается снаружи. Это влияет на восприятие безопасности и приватности.",
            patterns_by_lang={{
                "ru": [
                    r"\bсломан замок\b",
                    r"\bзамок не работал\b",
                    r"\bдверь не закрывалась на замок\b",
                    r"\bне запиралась дверь\b",
                    r"\bмагнитный замок не срабатывал\b",
                ],
                "en": [
                    r"\bbroken lock\b",
                    r"\bthe door lock was broken\b",
                    r"\bthe lock didn't work\b",
                    r"\bthe door wouldn't lock\b",
                    r"\bthe keycard lock was not working\b",
                ],
                "tr": [
                    r"\bkilit bozuktu\b",
                    r"\bkapı kilitlenmiyordu\b",
                    r"\bkilit çalışmıyordu\b",
                    r"\bkartlı kilit arızalıydı\b",
                ],
                "ar": [
                    r"\bالقفل بايظ\b",
                    r"\bالباب ما بيقفلش بمفتاح/كارت\b",
                    r"\bالقفل مش شغال\b",
                    r"\bمش قادرين نقفل الباب كويس\b",
                ],
                "zh": [
                    r"门锁坏了",
                    r"房门锁不上",
                    r"门卡锁不能用",
                    r"锁不工作",
                ],
            },
        ),
        
        
        "felt_safe": AspectRule(
            aspect_code="felt_safe",
            polarity_hint="positive",
            display="Субъективное ощущение безопасности в номере",
            display_short="чувствовали себя в безопасности",
            long_hint="Гости отмечают, что в помещении чувствовали себя в безопасности: надёжная дверь, отсутствие посторонних, спокойная обстановка в подъезде/коридоре.",
            patterns_by_lang={{
                "ru": [
                    r"\bчувствовали себя в безопасности\b",
                    r"\bбезопасно в номере\b",
                    r"\bчувствовали себя спокойно\b",
                    r"\bникаких посторонних\b",
                    r"\bникаких сомнений по безопасности\b",
                ],
                "en": [
                    r"\bfelt safe in the room\b",
                    r"\bwe felt safe staying there\b",
                    r"\bthe place felt secure\b",
                    r"\bthe building felt safe\b",
                    r"\bsafe environment\b",
                ],
                "tr": [
                    r"\bkendimizi güvende hissettik\b",
                    r"\bodada güvende hissettik\b",
                    r"\bmekan güvenli hissettiriyordu\b",
                    r"\bbinada güven hissi vardı\b",
                ],
                "ar": [
                    r"\bحاسّين أمان في الأوضة\b",
                    r"\bالمكان آمن\b",
                    r"\bحسينا إنه الوضع آمن\b",
                    r"\bحسينا بأمان طول الإقامة\b",
                ],
                "zh": [
                    r"住着很有安全感",
                    r"感觉很安全",
                    r"房间/楼里让人觉得安全",
                    r"整体环境让我们觉得安全",
                ],
            },
        ),
        
        
        "felt_unsafe": AspectRule(
            aspect_code="felt_unsafe",
            polarity_hint="negative",
            display="Субъективное ощущение небезопасности в номере / объекте",
            display_short="нечувство безопасности",
            long_hint="Гости фиксируют, что не чувствовали себя в безопасности: сомнения в двери/замке, посторонние люди в коридорах или у входа, тревожная атмосфера здания или района.",
            patterns_by_lang={{
                "ru": [
                    r"\bне чувствовали себя в безопасности\b",
                    r"\bбыло страшновато\b",
                    r"\bнебезопасно себя чувствовали\b",
                    r"\bне комфортно оставаться в номере\b",
                    r"\bпобоялись оставлять вещи\b",
                ],
                "en": [
                    r"\bdidn't feel safe\b",
                    r"\bwe did not feel safe in the room\b",
                    r"\bfelt unsafe in the building\b",
                    r"\bwe were worried about leaving our stuff\b",
                    r"\bthe area felt unsafe\b",
                ],
                "tr": [
                    r"\bgüvende hissetmedik\b",
                    r"\bkendimizi pek güvenli hissetmedik\b",
                    r"\bbinada güvende değildik gibi hissettik\b",
                    r"\besyaları bırakmak pek güvenli hissettirmedi\b",
                ],
                "ar": [
                    r"\bما حسّيناش بأمان\b",
                    r"\bالمكان مش مريح أمنيًا\b",
                    r"\bخايفين نسب الشنط لوحدها\b",
                    r"\bحسينا إنه المنطقة مش آمنة\b",
                ],
                "zh": [
                    r"感觉不太安全",
                    r"住着没有安全感",
                    r"不敢把东西放心留在房间里",
                    r"周围环境让人担心安全",
                ],
            },
        ),
        
        
        "furniture_broken": AspectRule(
            aspect_code="furniture_broken",
            polarity_hint="negative",
            display="Сломанная / повреждённая мебель",
            display_short="поломанная мебель",
            long_hint="Гости отмечают износ или повреждение мебели: шатающиеся стулья, сломанные шкафы, треснувшие поверхности, выломанные ручки. Это регистрируется как техническое состояние номера.",
            patterns_by_lang={{
                "ru": [
                    r"\bсломанная мебель\b",
                    r"\bмебель старая и сломанная\b",
                    r"\bстулья шатаются\b",
                    r"\bполка сломана\b",
                    r"\bдверца шкафа отваливается\b",
                ],
                "en": [
                    r"\bbroken furniture\b",
                    r"\bthe furniture was damaged\b",
                    r"\bwobbly chairs\b",
                    r"\bthe wardrobe door was broken\b",
                    r"\bthe shelves were falling apart\b",
                ],
                "tr": [
                    r"\bmobilyalar kırık/döküktü\b",
                    r"\bsandalye sallanıyordu\b",
                    r"\bdolap kapağı kırıktı\b",
                    r"\braf düşüyordu neredeyse\b",
                ],
                "ar": [
                    r"\bالأثاث مكسّر\b",
                    r"\bالكرسي مهزوز ومش ثابت\b",
                    r"\bدولاب مكسور\b",
                    r"\bالترابيزة بايظة/مكسورة\b",
                ],
                "zh": [
                    r"家具是坏的",
                    r"椅子摇摇晃晃",
                    r"柜门都掉了",
                    r"桌子/柜子明显损坏",
                ],
            },
        ),
        
        
        "room_worn_out": AspectRule(
            aspect_code="room_worn_out",
            polarity_hint="negative",
            display="Изношенное состояние номера",
            display_short="номер уставший",
            long_hint="Гости описывают номер как уставший или требующий обновления: потертые стены, облезшая краска, царапанные поверхности, старый ремонт, общее впечатление «нужен рефреш».",
            patterns_by_lang={{
                "ru": [
                    r"\bномер уставший\b",
                    r"\bвсё старенькое\b",
                    r"\bнужен ремонт\b",
                    r"\bвидно, что давно не обновляли\b",
                    r"\bобшарпанные стены\b",
                    r"\bоблезлая краска\b",
                ],
                "en": [
                    r"\bworn out room\b",
                    r"\bthe room felt tired\b",
                    r"\bthe room is dated\b",
                    r"\bneeds refurbishment\b",
                    r"\blooks old and worn\b",
                    r"\bscuffed walls and peeling paint\b",
                ],
                "tr": [
                    r"\bodanın hali yorgundu\b",
                    r"\bodanın durumu eskiydi\b",
                    r"\byıpranmış bir oda\b",
                    r"\boda yenilenmeye ihtiyaç duyuyor\b",
                    r"\bduvarlar soyulmuştu\b",
                ],
                "ar": [
                    r"\bالأوضة باينة قديمة\b",
                    r"\bباين عليها مستهلكة\b",
                    r"\bمحتاجة تجديد\b",
                    r"\bالدهان مقشّر\b",
                    r"\bالأوضة شكلها تعبان\b",
                ],
                "zh": [
                    r"房间看起来很旧",
                    r"房间很旧需要翻新",
                    r"墙面有掉漆磨损",
                    r"整体感觉比较破旧",
                    r"房间状态比较疲惫老旧",
                ],
            },
        ),

        "wifi_fast": AspectRule(
            aspect_code="wifi_fast",
            polarity_hint="positive",
            display="Высокая скорость Wi-Fi",
            display_short="быстрый Wi-Fi",
            long_hint="Гости отмечают, что интернет был быстрым: подключение без задержек, страницы и видео грузятся мгновенно, комфортно пользоваться сервисами связи и потоковым видео.",
            patterns_by_lang={{
                "ru": [
                    r"\bбыстрый вай[- ]?фай\b",
                    r"\bочень быстрый интернет\b",
                    r"\bотличная скорость Wi[- ]?Fi\b",
                    r"\bинтернет летает\b",
                    r"\bстрим без лагов\b",
                ],
                "en": [
                    r"\bfast Wi[- ]?Fi\b",
                    r"\bvery fast internet\b",
                    r"\bgreat internet speed\b",
                    r"\bthe wifi was super fast\b",
                    r"\bno lag on streaming\b",
                ],
                "tr": [
                    r"\bwifi çok hızlıydı\b",
                    r"\binternet hızı çok iyiydi\b",
                    r"\bçok hızlı internet\b",
                    r"\bstream yaparken sorun yoktu\b",
                ],
                "ar": [
                    r"\bالواي فاي سريع جداً\b",
                    r"\bالنت سرعته حلوة\b",
                    r"\bانترنت سريع من غير تهنيج\b",
                    r"\bمفيش لاج في الفيديو\b",
                ],
                "zh": [
                    r"WIFI很快",
                    r"网速很快",
                    r"网络速度非常好没有卡顿",
                    r"看视频都不卡",
                ],
            },
        ),
        
        
        "internet_stable": AspectRule(
            aspect_code="internet_stable",
            polarity_hint="positive",
            display="Стабильность соединения",
            display_short="интернет стабильный",
            long_hint="Гости подчёркивают, что соединение работало стабильно: не отваливалось, не было перебоев, сигнал держится по всей комнате/объекту.",
            patterns_by_lang={{
                "ru": [
                    r"\bстабильный интернет\b",
                    r"\bвай[- ]?фай не отваливался\b",
                    r"\bсоединение не пропадало\b",
                    r"\bсигнал держится везде\b",
                ],
                "en": [
                    r"\bstable connection\b",
                    r"\bvery stable wifi\b",
                    r"\bthe internet was reliable\b",
                    r"\bno dropouts\b",
                    r"\bconnection never disconnected\b",
                ],
                "tr": [
                    r"\binternet bağlantısı stabildi\b",
                    r"\bwifi hiç kopmadı\b",
                    r"\bbağlantı güvenilirdi\b",
                    r"\bçekim gücü her yerde iyiydi\b",
                ],
                "ar": [
                    r"\bالواي فاي ثابت\b",
                    r"\bالنت ما بيقطعش\b",
                    r"\bالاتصال مستقر طول الوقت\b",
                    r"\bالإشارة كويسة في كل المكان\b",
                ],
                "zh": [
                    r"网络很稳定",
                    r"WIFI不会掉线",
                    r"连接一直都很稳",
                    r"信号在房间各处都很好",
                ],
            },
        ),
        
        
        "good_for_work": AspectRule(
            aspect_code="good_for_work",
            polarity_hint="positive",
            display="Интернет подходит для работы",
            display_short="подходит для удалённой работы",
            long_hint="Гости отмечают, что качество интернета достаточно для рабочих задач: видеозвонки, VPN, загрузка/выгрузка файлов — можно комфортно работать из номера.",
            patterns_by_lang={{
                "ru": [
                    r"\bудобно работать удалённо\b",
                    r"\bинтернет подходит для работы\b",
                    r"\bнормально для видеозвонков\b",
                    r"\bзум шёл без проблем\b",
                    r"\bVPN работал нормально\b",
                ],
                "en": [
                    r"\bgood for work\b",
                    r"\bwi[- ]?fi good enough to work\b",
                    r"\bperfect for remote work\b",
                    r"\bzoom calls worked fine\b",
                    r"\bVPN worked with no issues\b",
                ],
                "tr": [
                    r"\buzaktan çalışmak için uygundu\b",
                    r"\bçalışmak için internet yeterliydi\b",
                    r"\bzoom/görüntülü görüşme sorunsuzdu\b",
                    r"\bVPN sıkıntısız çalıştı\b",
                ],
                "ar": [
                    r"\bالنت ينفع شغل اونلاين\b",
                    r"\bقدرنا نشتغل من الأوضة عادي\b",
                    r"\bمكالمات الزوم ماشية من غير تقطيع\b",
                    r"\bالـVPN اشتغل كويس\b",
                ],
                "zh": [
                    r"网络可以满足远程办公",
                    r"可以正常视频会议",
                    r"适合在房间里办公",
                    r"VPN也能正常连",
                ],
            },
        ),
        
        
        "wifi_down": AspectRule(
            aspect_code="wifi_down",
            polarity_hint="negative",
            display="Wi-Fi не работал / отсутствовал доступ в интернет",
            display_short="вайфай не работал",
            long_hint="Гости сообщают, что интернета фактически не было: Wi-Fi не подключался, не выдавался пароль, сеть не работала большую часть времени.",
            patterns_by_lang={{
                "ru": [
                    r"\bвай[- ]?фай не работал\b",
                    r"\bинтернета вообще не было\b",
                    r"\bневозможно подключиться к wifi\b",
                    r"\bпароль не подходил и так и не дали другой\b",
                    r"\bсеть падала постоянно\b",
                ],
                "en": [
                    r"\bwifi didn't work\b",
                    r"\bno wifi at all\b",
                    r"\bwe couldn't connect to the wifi\b",
                    r"\bno internet connection in the room\b",
                    r"\bwifi was down most of the time\b",
                ],
                "tr": [
                    r"\bwifi çalışmıyordu\b",
                    r"\binternet hiç yoktu\b",
                    r"\bwifiye bağlanamadık\b",
                    r"\boda da internet çekmedi\b",
                ],
                "ar": [
                    r"\bمافيش واي فاي شغال\b",
                    r"\bالواي فاي مبيركبش أصلاً\b",
                    r"\bمش قادرين نتصل بالنت\b",
                    r"\bمفيش إنترنت في الأوضة فعلياً\b",
                ],
                "zh": [
                    r"WIFI根本连不上",
                    r"房间里没有网",
                    r"网络完全不能用",
                    r"无线网络基本是断的",
                ],
            },
        ),
        
        
        "wifi_slow": AspectRule(
            aspect_code="wifi_slow",
            polarity_hint="negative",
            display="Низкая скорость Wi-Fi",
            display_short="медленный интернет",
            long_hint="Гости отмечают, что интернет работает, но очень медленно: страницы грузятся долго, медленный аплоад, невозможно смотреть видео или работать онлайн комфортно.",
            patterns_by_lang={{
                "ru": [
                    r"\bмедленный вай[- ]?фай\b",
                    r"\bинтернет очень медленный\b",
                    r"\bеле грузит страницы\b",
                    r"\bонлайн видео тормозит\b",
                    r"\bскорости не хватает для работы\b",
                ],
                "en": [
                    r"\bslow wifi\b",
                    r"\bvery slow internet\b",
                    r"\binternet was super slow\b",
                    r"\bpages loaded really slowly\b",
                    r"\bstreaming was impossible\b",
                ],
                "tr": [
                    r"\byavaş wifi\b",
                    r"\binternet çok yavaştı\b",
                    r"\bsayfalar zor açılıyordu\b",
                    r"\bstream yapmak imkansızdı\b",
                ],
                "ar": [
                    r"\bالواي فاي بطئ جداً\b",
                    r"\bالنت بيحمّل ببطء\b",
                    r"\bالسرعة ضعيفة قوي\b",
                    r"\bما ينفعش تتفرج أونلاين من كتر التقطيع\b",
                ],
                "zh": [
                    r"WIFI很慢",
                    r"网速非常慢",
                    r"网页开得很慢",
                    r"几乎没法流媒体播放",
                ],
            },
        ),

        "wifi_unstable": AspectRule(
            aspect_code="wifi_unstable",
            polarity_hint="negative",
            display="Нестабильное подключение к интернету",
            display_short="вайфай отваливается",
            long_hint="Гости сообщают, что соединение с Wi-Fi постоянно обрывается: сеть падает, требуется переподключение, наблюдаются кратковременные потери сигнала, из-за чего невозможно пользоваться интернетом надёжно.",
            patterns_by_lang={{
                "ru": [
                    r"\bвай[- ]?фай постоянно отваливался\b",
                    r"\bинтернет все время пропадал\b",
                    r"\bсоединение постоянно рвётся\b",
                    r"\bвайфай нестабильный\b",
                    r"\bсеть то есть то нет\b",
                ],
                "en": [
                    r"\bunstable wifi\b",
                    r"\bthe connection kept dropping\b",
                    r"\bwifi kept disconnecting\b",
                    r"\bthe internet kept cutting out\b",
                    r"\bintermittent connection\b",
                ],
                "tr": [
                    r"\bwifi sürekli koptu\b",
                    r"\binternet bağlantısı sürekli gidip geldi\b",
                    r"\bbağlantı stabil değildi\b",
                    r"\bdevamlı kesiliyordu\b",
                ],
                "ar": [
                    r"\bالواي فاي بيفصل طول الوقت\b",
                    r"\bالاتصال بيقطع ويرجع\b",
                    r"\bالنت مش ثابت\b",
                    r"\bالإشارة بتروح وتيجي\b",
                ],
                "zh": [
                    r"WIFI老是断开",
                    r"网络一直掉线",
                    r"连接不稳定",
                    r"网时有时无",
                ],
            },
        ),
        
        
        "wifi_hard_to_connect": AspectRule(
            aspect_code="wifi_hard_to_connect",
            polarity_hint="negative",
            display="Сложности с подключением к Wi-Fi",
            display_short="трудно подключиться к Wi-Fi",
            long_hint="Гости фиксируют проблемы на этапе подключения: пароль не выдан или неверный, сложная система логина, каптив-портал не пускает, требуется многократная авторизация.",
            patterns_by_lang={{
                "ru": [
                    r"\bне могли подключиться к вай[- ]?фаю\b",
                    r"\bпароль от вайфая не подходил\b",
                    r"\bсложно подключиться к wifi\b",
                    r"\bнужно было каждый раз логиниться\b",
                    r"\bвайфай не пускал\b",
                ],
                "en": [
                    r"\bhard to connect to the wifi\b",
                    r"\bwe couldn't log in to the wifi\b",
                    r"\bthe wifi password didn't work\b",
                    r"\bthe login page wouldn't let us in\b",
                    r"\bhad to sign in over and over\b",
                ],
                "tr": [
                    r"\bwifiye bağlanmak zordu\b",
                    r"\bşifre çalışmadı\b",
                    r"\bgiriş ekranı açılmadı\b",
                    r"\bher seferinde tekrar giriş yapmak zorunda kaldık\b",
                ],
                "ar": [
                    r"\bمش عارفين نوصل على الواي فاي\b",
                    r"\bالباسورد ما بيشتغلش\b",
                    r"\bالصفحة ما بتدخلناش عالنت\b",
                    r"\bكل شوية يطلب لوج إن تاني\b",
                ],
                "zh": [
                    r"WIFI连不上",
                    r"密码不好用/输对也上不了网",
                    r"登录网页进不去",
                    r"每次都得重新登录",
                ],
            },
        ),
        
        
        "internet_not_suitable_for_work": AspectRule(
            aspect_code="internet_not_suitable_for_work",
            polarity_hint="negative",
            display="Интернет не подходит для работы",
            display_short="интернет не для работы",
            long_hint="Гости указывают, что качество интернета не позволяет полноценно работать удалённо: видеозвонки рвутся, VPN не держится, файлы не отправляются. Это фиксируется как критичный операционный барьер для бизнес-гостей.",
            patterns_by_lang={{
                "ru": [
                    r"\bдля работы интернет не подходит\b",
                    r"\bневозможно нормально работать из номера\b",
                    r"\bzoom рвётся\b",
                    r"\bвидео созвон невозможно\b",
                    r"\bvpn отваливается сразу\b",
                ],
                "en": [
                    r"\bnot good enough to work\b",
                    r"\bthe wifi was not suitable for work\b",
                    r"\bcouldn't work remotely from the room\b",
                    r"\bzoom/teams kept dropping\b",
                    r"\bvpn wouldn't stay connected\b",
                ],
                "tr": [
                    r"\bçalışmak için uygun internet yoktu\b",
                    r"\buzaktan çalışmak imkansızdı\b",
                    r"\bzoom sürekli koptu\b",
                    r"\bVPN hiç tutmadı\b",
                ],
                "ar": [
                    r"\bالانترنت ماينفعش شغل\b",
                    r"\bمش قادرين نشتغل أونلاين من الأوضة\b",
                    r"\bالمكالمات بتقطع على طول\b",
                    r"\bالـVPN مش ثابت خالص\b",
                ],
                "zh": [
                    r"网速没法办公用",
                    r"远程办公基本做不了",
                    r"视频会议老断线",
                    r"VPN老是掉线",
                ],
            },
        ),
        
        
        "ac_noisy": AspectRule(
            aspect_code="ac_noisy",
            polarity_hint="negative",
            display="Шумный кондиционер",
            display_short="шумный кондиционер",
            long_hint="Гости сообщают, что кондиционер сильно шумит во время работы (гул, вибрация, треск вентилятора), что мешает отдыху и сну.",
            patterns_by_lang={{
                "ru": [
                    r"\bшумный кондиционер\b",
                    r"\bкондиционер очень шумел\b",
                    r"\bкондиционер гудел всю ночь\b",
                    r"\bсильный гул от кондиционера\b",
                    r"\bкондиционер тарахтел\b",
                ],
                "en": [
                    r"\bnoisy AC\b",
                    r"\bthe air conditioner was very loud\b",
                    r"\bloud humming from the AC\b",
                    r"\bthe AC was making a loud noise all night\b",
                    r"\bthe aircon was rattling\b",
                ],
                "tr": [
                    r"\bklima çok gürültülüydü\b",
                    r"\bklima sürekli uğultu yaptı\b",
                    r"\bklimadan ses geliyordu\b",
                    r"\bgece boyunca klima ses yaptı\b",
                ],
                "ar": [
                    r"\bالتكييف صوته عالي\b",
                    r"\bالـAC بيزن طول الليل\b",
                    r"\bصوت المكيف مزعج\b",
                    r"\bالتكييف كان بيعمل دوشة\b",
                ],
                "zh": [
                    r"空调声音很大",
                    r"空调一直嗡嗡响",
                    r"睡觉时空调噪音很吵",
                    r"空调有很吵的震动声",
                ],
            },
        ),
        
        
        "fridge_noisy": AspectRule(
            aspect_code="fridge_noisy",
            polarity_hint="negative",
            display="Шумный холодильник",
            display_short="шумный холодильник",
            long_hint="Гости отмечают, что мини-холодильник/холодильник издаёт громкие звуки (жужжание, треск компрессора, вибрация), особенно ночью, что воспринимается как фактор, мешающий сну.",
            patterns_by_lang={{
                "ru": [
                    r"\bшумный холодильник\b",
                    r"\bхолодильник громко жужжал\b",
                    r"\bхолодильник тарахтел ночью\b",
                    r"\bгул от мини\-бара мешал спать\b",
                    r"\bкомпрессор сильно шумел\b",
                ],
                "en": [
                    r"\bnoisy fridge\b",
                    r"\bthe fridge was very loud\b",
                    r"\bthe minibar was humming loudly\b",
                    r"\bthe fridge was buzzing all night\b",
                    r"\bthe fridge made a constant noise\b",
                ],
                "tr": [
                    r"\bbuzdolabı çok ses yapıyordu\b",
                    r"\bmini bar gece boyunca uğuldadı\b",
                    r"\bbuzdolabı sürekli vızıldıyordu\b",
                    r"\bkompresör sesi uyutmadı\b",
                ],
                "ar": [
                    r"\bالتلاجة صوتها عالي\b",
                    r"\bالميني بار بيزن طول الليل\b",
                    r"\bصوت الموتور مزعج بالليل\b",
                    r"\bالتلاجة بتعمل دوشة وإحنا نايمين\b",
                ],
                "zh": [
                    r"冰箱声音很大",
                    r"小冰箱晚上一直嗡嗡响",
                    r"冰箱压缩机声很吵",
                    r"冰箱的噪音影响睡觉",
                ],
            },
        ),

        "pipes_noise": AspectRule(
            aspect_code="pipes_noise",
            polarity_hint="negative",
            display="Шум труб / водопровода",
            display_short="шум труб",
            long_hint="Гости сообщают о шуме от труб или стояков: громкие звуки воды, вибрация, стук от водопровода или канализации, слышимые в номере, особенно ночью.",
            patterns_by_lang={{
                "ru": [
                    r"\bшумели трубы\b",
                    r"\bгромко шумели трубы\b",
                    r"\bслышно воду в трубах\b",
                    r"\bпостоянный шум стояка\b",
                    r"\bгул труб всю ночь\b",
                ],
                "en": [
                    r"\bnoisy pipes\b",
                    r"\bthe pipes were loud\b",
                    r"\byou could hear the water pipes\b",
                    r"\bconstant pipe noise\b",
                    r"\bplumbing noise all night\b",
                ],
                "tr": [
                    r"\bborular çok ses yapıyordu\b",
                    r"\bsu borularından ses geliyordu\b",
                    r"\btesisat sesi yüksekti\b",
                    r"\bgece boyunca boru sesi vardı\b",
                ],
                "ar": [
                    r"\bصوت المواسير كان عالي\b",
                    r"\bصوت الميّة وهي ماشية في المواسير\b",
                    r"\bصوت السباكة كان مزعج بالليل\b",
                    r"\bالمواسير بتزن طول الليل\b",
                ],
                "zh": [
                    r"管道声音很大",
                    r"能听到水管里的水声",
                    r"晚上一直有水管噪音",
                    r"水管在嗡嗡作响",
                ],
            },
        ),
        
        
        "ventilation_noisy": AspectRule(
            aspect_code="ventilation_noisy",
            polarity_hint="negative",
            display="Шумная вентиляция / вытяжка",
            display_short="шум вентиляции",
            long_hint="Гости фиксируют, что система вентиляции или вытяжка в ванной/в комнате слишком шумная: гул, жужжание, свист, не отключается и мешает отдыху.",
            patterns_by_lang={{
                "ru": [
                    r"\bшумная вентиляция\b",
                    r"\bочень громкая вытяжка\b",
                    r"\bвентилятор в ванной жужжит\b",
                    r"\bпостоянный гул вентиляции\b",
                    r"\bвентиляция не замолкала\b",
                ],
                "en": [
                    r"\bnoisy ventilation\b",
                    r"\bthe fan was very loud\b",
                    r"\bthe bathroom fan was noisy\b",
                    r"\bconstant noise from the vent\b",
                    r"\bthe extractor fan wouldn't stop buzzing\b",
                ],
                "tr": [
                    r"\bhavalandırma çok ses yapıyordu\b",
                    r"\baspiratör aşırı gürültülüydü\b",
                    r"\bbanyo fanı sürekli uğuldadı\b",
                    r"\bhavalandırma gürültüsü hiç kesilmedi\b",
                ],
                "ar": [
                    r"\bالمروحة صوتها عالي\b",
                    r"\bالشفاط مزعج جداً\b",
                    r"\bفيه زنة مستمرة من الشفاط\b",
                    r"\bتهوية الحمام دوشة طول الوقت\b",
                ],
                "zh": [
                    r"通风扇很吵",
                    r"排风扇一直嗡嗡响",
                    r"通风系统噪音很大",
                    r"卫生间抽风机很吵停不下来",
                ],
            },
        ),
        
        
        "night_mechanical_hum": AspectRule(
            aspect_code="night_mechanical_hum",
            polarity_hint="negative",
            display="Фоновый технический гул ночью",
            display_short="тех. гул ночью",
            long_hint="Гости описывают постоянный механический фоновой шум (оборудование здания, компрессор, вентиляционные блоки, бойлерная и т.п.), который слышен в номере ночью и мешает сну.",
            patterns_by_lang={{
                "ru": [
                    r"\bгул всю ночь\b",
                    r"\bпостоянный гул оборудования\b",
                    r"\bтехнический шум за стеной всю ночь\b",
                    r"\bнизкий гул мешал спать\b",
                    r"\bбудто мотор жужжит всю ночь\b",
                ],
                "en": [
                    r"\bloud humming noise at night\b",
                    r"\bconstant mechanical hum\b",
                    r"\bindustrial humming sound all night\b",
                    r"\blow humming noise kept us awake\b",
                    r"\bsounded like a generator running all night\b",
                ],
                "tr": [
                    r"\bgece boyunca sürekli uğultu vardı\b",
                    r"\bmakine sesi hiç kesilmedi\b",
                    r"\bmotor sesi/gece uğultusu\b",
                    r"\bdüşük frekanslı bir uğultu uyutmadı\b",
                ],
                "ar": [
                    r"\bفيه زنة ميكانيكية طول الليل\b",
                    r"\bصوت موتور شغال طول الليل\b",
                    r"\bفيه همهمة مستمرة بالليل\b",
                    r"\bصوت واطي بس مزعج طول الليل\b",
                ],
                "zh": [
                    r"晚上有持续的机器嗡嗡声",
                    r"整晚都有低频机械噪音",
                    r"像发动机一直在运转的声音",
                    r"嗡嗡声影响睡觉",
                ],
            },
        ),
        
        
        "tech_noise_sleep_issue": AspectRule(
            aspect_code="tech_noise_sleep_issue",
            polarity_hint="negative",
            display="Технический шум мешал сну",
            display_short="шум техники мешал спать",
            long_hint="Гости прямо указывают, что из-за шума от оборудования (кондиционер, холодильник, бойлер, вентиляция, насосы) не удалось нормально уснуть или приходилось просыпаться.",
            patterns_by_lang={{
                "ru": [
                    r"\bшум техники мешал спать\b",
                    r"\bневозможно уснуть из-за гудящей техники\b",
                    r"\bбытовые звуки будили ночью\b",
                    r"\bпросыпались от шума оборудования\b",
                    r"\bприходилось выключать технику чтобы поспать\b",
                ],
                "en": [
                    r"\bthe noise from the equipment kept us awake\b",
                    r"\bcouldn't sleep because of the noise from the AC/fridge\b",
                    r"\bwe were woken up by the appliance noise\b",
                    r"\bhad to turn things off to sleep\b",
                    r"\bmechanical noise made it hard to sleep\b",
                ],
                "tr": [
                    r"\bcihazların sesi yüzünden uyuyamadık\b",
                    r"\bklima/buzdolabı sesi uykumuzu böldü\b",
                    r"\bgece makine sesi uyandırdı\b",
                    r"\buyumak için cihazları kapatmak zorunda kaldık\b",
                ],
                "ar": [
                    r"\bصوت الأجهزة مانعنا ننام\b",
                    r"\bصوت التكييف/التلاجة صحّانا بالليل\b",
                    r"\bصحينا من صوت المكنة/الموتور\b",
                    r"\bاضطرينا نطفي الحاجات عشان نعرف ننام\b",
                ],
                "zh": [
                    r"设备噪音让人睡不着",
                    r"被空调/冰箱的声音吵醒",
                    r"为了睡觉不得不把设备关掉",
                    r"晚上机械噪音影响睡眠",
                ],
            },
        ),
        
        
        "ac_quiet": AspectRule(
            aspect_code="ac_quiet",
            polarity_hint="positive",
            display="Тихий кондиционер",
            display_short="тихий кондиционер",
            long_hint="Гости подчёркивают, что кондиционер работает тихо и не мешает сну даже при непрерывной работе ночью. Это рассматривается как показатель качественного оборудования и комфорта сна.",
            patterns_by_lang={{
                "ru": [
                    r"\bкондиционер тихий\b",
                    r"\bочень тихо работал кондиционер\b",
                    r"\bкондиционер не шумел ночью\b",
                    r"\bможно спать с включенным кондиционером\b",
                ],
                "en": [
                    r"\bquiet AC\b",
                    r"\bthe air conditioner was very quiet\b",
                    r"\bwe could sleep with the AC on\b",
                    r"\bthe AC wasn't noisy at all\b",
                ],
                "tr": [
                    r"\bklima çok sessizdi\b",
                    r"\bklima gece rahatsız etmedi\b",
                    r"\bklima sesi yok denecek kadar azdı\b",
                    r"\bklima açıkken rahat uyuduk\b",
                ],
                "ar": [
                    r"\bالتكييف هادي\b",
                    r"\bالتكييف مش بيعمل صوت عالي\b",
                    r"\bاقدرنا ننام والتكييف شغال\b",
                    r"\bصوته مش مزعج خالص\b",
                ],
                "zh": [
                    r"空调很安静",
                    r"空调开着也能安心睡觉",
                    r"空调几乎没什么噪音",
                    r"空调声音很轻不影响睡觉",
                ],
            },
        ),

        "fridge_quiet": AspectRule(
            aspect_code="fridge_quiet",
            polarity_hint="positive",
            display="Тихая работа холодильника",
            display_short="тихий холодильник",
            long_hint="Гости отмечают, что холодильник/мини-бар работает бесшумно или почти не слышен ночью. Это фиксируется как отсутствие акустического дискомфорта от бытовой техники.",
            patterns_by_lang={{
                "ru": [
                    r"\bхолодильник тихий\b",
                    r"\bмини\-бар не шумел\b",
                    r"\bхолодильник практически не слышно\b",
                    r"\bне мешал спать холодильник\b",
                ],
                "en": [
                    r"\bthe fridge was quiet\b",
                    r"\bvery quiet minibar\b",
                    r"\bwe could barely hear the fridge\b",
                    r"\bthe fridge didn't disturb our sleep\b",
                ],
                "tr": [
                    r"\bbuzdolabı çok sessizdi\b",
                    r"\bmini bar sesi rahatsız etmedi\b",
                    r"\bbuzdolabının sesi neredeyse yoktu\b",
                ],
                "ar": [
                    r"\bالتلاجة صوتها هادي\b",
                    r"\bالميني بار مش بيعمل دوشة\b",
                    r"\bصوت التلاجة مش مسموع تقريباً\b",
                    r"\bالتلاجة ما ضايقتناش وإحنا نايمين\b",
                ],
                "zh": [
                    r"冰箱很安静",
                    r"小冰箱几乎没声音",
                    r"冰箱完全不影响睡觉",
                    r"几乎听不到冰箱的声音",
                ],
            },
        ),
        
        
        "no_tech_noise_night": AspectRule(
            aspect_code="no_tech_noise_night",
            polarity_hint="positive",
            display="Отсутствие технического шума ночью",
            display_short="ночью тихо, без техники",
            long_hint="Гости подчёркивают, что ночью не было навязчивых технических звуков (кондиционер, вентиляция, трубы, компрессоры), и ничто не мешало сну с точки зрения фонового гула оборудования.",
            patterns_by_lang={{
                "ru": [
                    r"\bночью было тихо\b",
                    r"\bне было гудящих звуков техники\b",
                    r"\bникаких посторонних звуков ночью\b",
                    r"\bникакого гула оборудования\b",
                ],
                "en": [
                    r"\bno mechanical noise at night\b",
                    r"\bquiet at night, no humming\b",
                    r"\bno appliance noise during the night\b",
                    r"\bwe could sleep without any humming sounds\b",
                ],
                "tr": [
                    r"\bgece teknik gürültü yoktu\b",
                    r"\bgece boyunca tamamen sessizdi\b",
                    r"\bherhangi bir uğultu yoktu\b",
                    r"\bcihazlardan gece ses gelmedi\b",
                ],
                "ar": [
                    r"\bمفيش صوت مكنات بالليل\b",
                    r"\bبالليل كان هادي من غير زنّة أجهزة\b",
                    r"\bمافيش دوشة تكييف/شفاط وإحنا نايمين\b",
                    r"\bنقدر ننام من غير أي صوت ميكانيكي\b",
                ],
                "zh": [
                    r"晚上没有机器噪音",
                    r"夜里没有嗡嗡声",
                    r"晚上很安静没有设备声",
                    r"可以安静睡觉没有机械噪音",
                ],
            },
        ),
        
        
        "elevator_working": AspectRule(
            aspect_code="elevator_working",
            polarity_hint="positive",
            display="Работа лифта без сбоев",
            display_short="лифт работает",
            long_hint="Гости отмечают, что лифт исправен и всегда доступен: не заставляет ждать слишком долго, корректно поднимает/опускает без поломок. Это особенно важно для гостей с багажом и ограниченной мобильностью.",
            patterns_by_lang={{
                "ru": [
                    r"\bлифт работал\b",
                    r"\bлифт всегда работал\b",
                    r"\bлифт исправный\b",
                    r"\bлифт без проблем\b",
                    r"\bлифт всегда был доступен\b",
                ],
                "en": [
                    r"\bthe elevator was working\b",
                    r"\bthe lift worked fine\b",
                    r"\bthe elevator worked the whole time\b",
                    r"\bno issues with the elevator\b",
                    r"\bthe lift was always available\b",
                ],
                "tr": [
                    r"\basansör çalışıyordu\b",
                    r"\basansör hep aktıf̧ti\b",
                    r"\basansör sorunsuzdu\b",
                    r"\basansörde hiç problem yaşamadık\b",
                ],
                "ar": [
                    r"\bالأسانسير شغال كويس\b",
                    r"\bالمصعد شغال طول الإقامة\b",
                    r"\bمافيش مشاكل في الأسانسير\b",
                    r"\bالأسانسير متاح وما بيعلقش\b",
                ],
                "zh": [
                    r"电梯正常运行",
                    r"电梯一直可以用",
                    r"电梯没有问题",
                    r"电梯随时都能用",
                ],
            },
        ),
        
        
        "luggage_easy": AspectRule(
            aspect_code="luggage_easy",
            polarity_hint="positive",
            display="Удобство перемещения багажа",
            display_short="с багажом удобно",
            long_hint="Гости отмечают, что перемещать багаж было несложно: лифт, удобные коридоры, отсутствие длинных лестниц. Это фиксируется как операционный комфорт при заезде и выезде.",
            patterns_by_lang={{
                "ru": [
                    r"\bлегко с багажом\b",
                    r"\bс чемоданами было удобно\b",
                    r"\bне пришлось таскать чемоданы по лестницам\b",
                    r"\bбагаж было легко довезти до номера\b",
                ],
                "en": [
                    r"\beasy with luggage\b",
                    r"\bvery easy to manage our luggage\b",
                    r"\bdidn't have to carry suitcases up stairs\b",
                    r"\bwe could take our bags up easily\b",
                    r"\bgetting luggage to the room was easy\b",
                ],
                "tr": [
                    r"\bvalizlerle çıkmak kolaydı\b",
                    r"\bçanta taşımak rahattı\b",
                    r"\bmerdiven çıkmadan valizleri odaya götürebildik\b",
                    r"\bvalizlerle sorun yaşamadık\b",
                ],
                "ar": [
                    r"\bسهل جداً مع الشنط\b",
                    r"\bما اضطريناش نطلع شنطنا على السلم\b",
                    r"\bقدرنا ناخد الشنط للغرفة بسهولة\b",
                    r"\bبالعفش الموضوع كان مريح\b",
                ],
                "zh": [
                    r"行李搬运很方便",
                    r"带行李去房间很轻松",
                    r"不用扛行李上楼梯",
                    r"拿行李走动很方便",
                ],
            },
        ),
        
        
        "elevator_broken": AspectRule(
            aspect_code="elevator_broken",
            polarity_hint="negative",
            display="Неисправный лифт",
            display_short="лифт не работает",
            long_hint="Гости сообщают, что лифт был отключён или не работал: зависал, показывал ошибку, приходилось идти пешком. Это особенно критично для верхних этажей и тяжёлого багажа.",
            patterns_by_lang={{
                "ru": [
                    r"\bлифт не работал\b",
                    r"\bлифт был сломан\b",
                    r"\bлифт постоянно ломался\b",
                    r"\bприходилось подниматься пешком\b",
                    r"\bлифт застревал\b",
                ],
                "en": [
                    r"\bthe elevator was broken\b",
                    r"\bthe lift didn't work\b",
                    r"\bthe elevator was out of service\b",
                    r"\bwe had to take the stairs\b",
                    r"\bthe lift kept breaking down\b",
                ],
                "tr": [
                    r"\basansör bozuktu\b",
                    r"\basansör çalışmıyordu\b",
                    r"\bsürekli merdiven kullanmak zorunda kaldık\b",
                    r"\basansör arızalıydı\b",
                ],
                "ar": [
                    r"\bالأسانسير بايظ\b",
                    r"\bالمصعد كان معطّل\b",
                    r"\bاضطرينا نطلع السلم بالشنط\b",
                    r"\bالأسانسير بيعلق/ما بيطلعش\b",
                ],
                "zh": [
                    r"电梯坏了",
                    r"电梯停用不能坐",
                    r"只能走楼梯提行李",
                    r"电梯老是出故障",
                ],
            },
        ),

        "elevator_stuck": AspectRule(
            aspect_code="elevator_stuck",
            polarity_hint="negative",
            display="Застревание лифта / сбои в работе лифта",
            display_short="лифт застревал",
            long_hint="Гости сообщают, что лифт застревал, останавливался между этажами или отказывался ехать. Это фиксируется как операционный риск и фактор дискомфорта при перемещении по объекту.",
            patterns_by_lang={{
                "ru": [
                    r"\bлифт застрял\b",
                    r"\bзастряли в лифте\b",
                    r"\bлифт клинил\b",
                    r"\bлифт останавливался между этажами\b",
                    r"\bлифт не ехал\b",
                ],
                "en": [
                    r"\bthe elevator got stuck\b",
                    r"\bwe got stuck in the lift\b",
                    r"\bthe lift kept getting stuck\b",
                    r"\bthe elevator stopped between floors\b",
                    r"\bthe elevator froze and wouldn't move\b",
                ],
                "tr": [
                    r"\basansör sıkıştı\b",
                    r"\basansörde kaldık\b",
                    r"\basansör katlar arasında kaldı\b",
                    r"\basansör çalışırken takılıp kaldı\b",
                ],
                "ar": [
                    r"\bالأسانسير وقف بين الأدوار\b",
                    r"\bإحنا اتزنقنا في الأسانسير\b",
                    r"\bالأسانسير علّق\b",
                    r"\bالمصعد وقف ومكملش\b",
                ],
                "zh": [
                    r"电梯卡住了",
                    r"被困在电梯里",
                    r"电梯卡在楼层之间",
                    r"电梯突然停住不动",
                ],
            },
        ),
        
        
        "no_elevator_heavy_bags": AspectRule(
            aspect_code="no_elevator_heavy_bags",
            polarity_hint="negative",
            display="Отсутствие лифта при наличии багажа / высоких этажей",
            display_short="нет лифта, тяжело с багажом",
            long_hint="Гости фиксируют отсутствие лифта и необходимость поднимать багаж по лестнице (часто на высокие этажи), что воспринимается как неудобство при заезде и выезде.",
            patterns_by_lang={{
                "ru": [
                    r"\bлифта нет\b",
                    r"\bпришлось тащить чемоданы по лестнице\b",
                    r"\bтаскать багаж на этаж\b",
                    r"\bбез лифта тяжело с чемоданами\b",
                    r"\bподнимались пешком с багажом\b",
                ],
                "en": [
                    r"\bno elevator\b",
                    r"\bthere was no lift\b",
                    r"\bhad to carry our luggage up the stairs\b",
                    r"\bwe had to drag our bags upstairs\b",
                    r"\bseveral flights of stairs with luggage\b",
                ],
                "tr": [
                    r"\basansör yoktu\b",
                    r"\bvalizleri merdivenden taşımak zorunda kaldık\b",
                    r"\bçanta ile katlara yürümek zorunda kaldık\b",
                    r"\basansör olmadığı için zor oldu\b",
                ],
                "ar": [
                    r"\bمافيش أسانسير\b",
                    r"\bطلعنا الشنط على السلم\b",
                    r"\bاضطرينا نشيل الشنط سلالم\b",
                    r"\bالمكان من غير أسانسير وده أتعبنا مع الشنط\b",
                ],
                "zh": [
                    r"没有电梯",
                    r"我们得扛行李爬楼梯",
                    r"拿行李上一层层楼非常累",
                    r"多层楼只能走楼梯提箱子",
                ],
            },
        ),
        
        
        "breakfast_tasty": AspectRule(
            aspect_code="breakfast_tasty",
            polarity_hint="positive",
            display="Вкус завтрака",
            display_short="вкусный завтрак",
            long_hint="Гости отмечают, что завтрак вкусный и по вкусовым качествам соответствует ожиданиям категории объекта. Это относится к общей оценке блюд, а не к выбору позиций.",
            patterns_by_lang={{
                "ru": [
                    r"\bвкусный завтрак\b",
                    r"\bзавтрак был очень вкусный\b",
                    r"\bеда на завтрак вкусная\b",
                    r"\bнам понравился завтрак\b",
                    r"\bвсё было вкусно утром\b",
                ],
                "en": [
                    r"\btasty breakfast\b",
                    r"\bthe breakfast was delicious\b",
                    r"\bbreakfast was very good\b",
                    r"\bwe really liked the breakfast\b",
                    r"\bfood in the morning was tasty\b",
                ],
                "tr": [
                    r"\bkahvaltı çok lezzetliydi\b",
                    r"\byemekler kahvaltıda çok güzeldi\b",
                    r"\bkahvaltıyı çok beğendik\b",
                    r"\blezzetli bir kahvaltı vardı\b",
                ],
                "ar": [
                    r"\bالفطار كان طعمه حلو\b",
                    r"\bالفطار لذيذ\b",
                    r"\bالأكل الصبح كان حلو جداً\b",
                    r"\bعجبنا الفطور\b",
                ],
                "zh": [
                    r"早餐很好吃",
                    r"早餐味道很棒",
                    r"早上的食物很好吃",
                    r"我们很喜欢这家早餐",
                ],
            },
        ),
        
        
        "food_fresh": AspectRule(
            aspect_code="food_fresh",
            polarity_hint="positive",
            display="Свежесть продуктов на завтраке",
            display_short="свежие продукты",
            long_hint="Гости подчёркивают, что продукты на завтраке были свежими: выпечка не черствела, овощи и фрукты не выглядели уставшими, ничего не ощущалось 'вчерашним'. Это часть восприятия качества F&B.",
            patterns_by_lang={{
                "ru": [
                    r"\bвсё свежее\b",
                    r"\bсвежие продукты на завтраке\b",
                    r"\bсвежие овощи и фрукты\b",
                    r"\bничего не было несвежим\b",
                    r"\bне было ощущения вчерашнего\b",
                ],
                "en": [
                    r"\bfresh food\b",
                    r"\bvery fresh breakfast items\b",
                    r"\bfruit and vegetables were fresh\b",
                    r"\bnothing tasted old or stale\b",
                    r"\bfresh pastries in the morning\b",
                ],
                "tr": [
                    r"\bkahvaltılıklar tazeydi\b",
                    r"\bfresh kahvaltı ürünleri\b",
                    r"\bher şey taptazeydi\b",
                    r"\bmeyve/sebze tazeydi\b",
                ],
                "ar": [
                    r"\bالأكل كان فريش\b",
                    r"\bكل حاجة كانت فريش مش بايتة\b",
                    r"\bالفواكه كانت طازة\b",
                    r"\bالفطار باين إنه متحضر في ساعته\b",
                ],
                "zh": [
                    r"早餐的食材很新鲜",
                    r"水果蔬菜都很新鲜",
                    r"没有不新鲜的感觉",
                    r"面包也是新鲜的不是隔夜的",
                ],
            },
        ),
        
        
        "food_hot_served_hot": AspectRule(
            aspect_code="food_hot_served_hot",
            polarity_hint="positive",
            display="Горячие блюда подаются горячими",
            display_short="горячее было горячим",
            long_hint="Гости отмечают, что горячие позиции завтрака (яичница, сосиски, омлет, каши и т.д.) подавались именно горячими, не остывшими. Это фиксируется как показатель поддержания стандарта сервиса по завтраку.",
            patterns_by_lang={{
                "ru": [
                    r"\bгорячее было действительно горячим\b",
                    r"\bяичница была горячая\b",
                    r"\bгорячие блюда не остывшие\b",
                    r"\bвсё подавали горячим\b",
                    r"\bгорячие блюда держали температуру\b",
                ],
                "en": [
                    r"\bthe hot food was actually hot\b",
                    r"\bhot dishes were served hot\b",
                    r"\beggs/sausages were hot and fresh\b",
                    r"\bthe cooked breakfast was served hot\b",
                    r"\bnothing on the hot buffet was cold\b",
                ],
                "tr": [
                    r"\bsıcak kahvaltılıklar gerçekten sıcaktı\b",
                    r"\bsıcak yemekler sıcaktı\b",
                    r"\byumurta/sosis sıcak servis edildi\b",
                    r"\bsıcak büfe soğumamıştı\b",
                ],
                "ar": [
                    r"\bالأكل السخن كان فعلاً سخن\b",
                    r"\bالبيض/السوسيس اتقدموا سخنين\b",
                    r"\bالأكل السخن ماكانش بارد\b",
                    r"\bالفطار السخن حافظ على حرارته\b",
                ],
                "zh": [
                    r"热菜是热的上桌",
                    r"热早餐真的热不是凉的",
                    r"鸡蛋香肠都是热的",
                    r"热食保持了温度没有凉掉",
                ],
            },
        ),

        "coffee_good": AspectRule(
            aspect_code="coffee_good",
            polarity_hint="positive",
            display="Качество кофе на завтраке",
            display_short="кофе хороший",
            long_hint="Гости отмечают, что кофе на завтраке понравился: вкус насыщенный, не разбавленный, не 'порошковый', есть эспрессо/капучино приемлемого уровня. Это влияет на общее восприятие завтрака как сервиса, а не только еды.",
            patterns_by_lang={{
                "ru": [
                    r"\bкофе хороший\b",
                    r"\bвкусный кофе\b",
                    r"\bкофе реально нормальный\b",
                    r"\bкофемашина делает хороший кофе\b",
                    r"\bэспрессо понравился\b",
                ],
                "en": [
                    r"\bgood coffee\b",
                    r"\bthe coffee was great\b",
                    r"\bcoffee was surprisingly good\b",
                    r"\bnice espresso in the morning\b",
                    r"\bdecent cappuccino at breakfast\b",
                ],
                "tr": [
                    r"\bkahve güzeldi\b",
                    r"\bkahvesi çok iyiydi\b",
                    r"\bkahve makinasi iyi kahve veriyordu\b",
                    r"\bespresso gayet iyiydi\b",
                ],
                "ar": [
                    r"\bالقهوة كانت حلوة\b",
                    r"\bالقهوة طعمها كويس\b",
                    r"\bالإسبرسو كان مظبوط\b",
                    r"\bالكابتشينو الصبح كان حلو\b",
                ],
                "zh": [
                    r"早餐的咖啡很好喝",
                    r"咖啡味道很好",
                    r"浓缩咖啡不错",
                    r"咖啡机出来的咖啡很可以",
                ],
            },
        ),
        
        
        "breakfast_bad_taste": AspectRule(
            aspect_code="breakfast_bad_taste",
            polarity_hint="negative",
            display="Невкусный завтрак",
            display_short="невкусный завтрак",
            long_hint="Гости фиксируют, что завтрак невкусный: блюда пресные, некачественные по вкусу, ощущаются 'дешёвыми' или приготовленными без внимания. Это относится к общей органолептике, не к температуре или свежести.",
            patterns_by_lang={{
                "ru": [
                    r"\bневкусный завтрак\b",
                    r"\bзавтрак был так себе\b",
                    r"\bеда на завтрак невкусная\b",
                    r"\bсовсем не понравился завтрак\b",
                    r"\bзавтрак посредственный по вкусу\b",
                ],
                "en": [
                    r"\bbad breakfast\b",
                    r"\bthe breakfast didn't taste good\b",
                    r"\bthe breakfast was awful\b",
                    r"\bvery bland breakfast\b",
                    r"\bwe didn't like the breakfast at all\b",
                ],
                "tr": [
                    r"\bkahvaltı lezzetsizdi\b",
                    r"\bkahvaltıyı sevmedik\b",
                    r"\byemeklerin tadı iyi değildi\b",
                    r"\bkahvaltı baya kötüydü\b",
                ],
                "ar": [
                    r"\bالفطار ماكانش حلو\b",
                    r"\bالأكل الصبح طعمه وحش\b",
                    r"\bالفطار مش لذيذ\b",
                    r"\bمفيش طعم في الأكل\b",
                ],
                "zh": [
                    r"早餐不好吃",
                    r"早餐味道一般甚至很差",
                    r"早上的东西很难吃",
                    r"我们不喜欢早餐的味道",
                ],
            },
        ),
        
        
        "food_not_fresh": AspectRule(
            aspect_code="food_not_fresh",
            polarity_hint="negative",
            display="Несвежие продукты на завтраке",
            display_short="несвежие продукты",
            long_hint="Гости отмечают, что часть позиций на завтраке выглядела или ощущалась несвежей: черствый хлеб, засохшая нарезка, подвяленные фрукты, 'вчерашнее'. Это фиксируется как риск по качеству F&B.",
            patterns_by_lang={{
                "ru": [
                    r"\bнесвежие продукты\b",
                    r"\bвсё несвежее\b",
                    r"\bвчерашняя выпечка\b",
                    r"\bфрукты подвявшие\b",
                    r"\bколбаса заветрилась\b",
                    r"\bсыры уже подсохшие\b",
                ],
                "en": [
                    r"\bnot fresh\b",
                    r"\bstale bread\b",
                    r"\bthe fruit looked old\b",
                    r"\bthe cold cuts were dried out\b",
                    r"\bthe food tasted old\b",
                ],
                "tr": [
                    r"\bfresh değildi\b",
                    r"\bbayat ekmek\b",
                    r"\bpeynir/sucuk kurumuştu\b",
                    r"\bmeyveler taze değildi\b",
                    r"\bsanki dünden kalmaydı\b",
                ],
                "ar": [
                    r"\bالأكل مش فريش\b",
                    r"\bالعيش بايت\b",
                    r"\bاللحوم الباردة شكلها ناشف وقديم\b",
                    r"\bالفاكهة شكلها مش طازة\b",
                    r"\bحاسينه أكل مبارح\b",
                ],
                "zh": [
                    r"食物不新鲜",
                    r"面包是隔夜的很干",
                    r"水果感觉不太新鲜",
                    r"冷盘风干了像放了一天",
                ],
            },
        ),
        
        
        "food_cold": AspectRule(
            aspect_code="food_cold",
            polarity_hint="negative",
            display="Горячие блюда подаются остывшими",
            display_short="горячее холодное",
            long_hint="Гости сообщают, что горячие блюда на завтраке были поданы холодными или едва тёплыми (яйца, сосиски, бекон и т.д.), что воспринимается как просадка в операционной подаче завтрака.",
            patterns_by_lang={{
                "ru": [
                    r"\bгорячее было холодным\b",
                    r"\bвсё было остывшим\b",
                    r"\bяйца холодные\b",
                    r"\bсосиски были еле тёплые\b",
                    r"\bбекон уже холодный\b",
                ],
                "en": [
                    r"\bthe hot food was cold\b",
                    r"\bthe eggs were cold\b",
                    r"\bthe sausages were barely warm\b",
                    r"\bthe bacon was cold\b",
                    r"\bthe cooked breakfast was lukewarm\b",
                ],
                "tr": [
                    r"\bsıcak yemekler soğuktu\b",
                    r"\byumurta soğuktu\b",
                    r"\bsosisler ılık bile değildi\b",
                    r"\bbekon buz gibiydi\b",
                ],
                "ar": [
                    r"\bالأكل السخن اتقدم بارد\b",
                    r"\bالبيض كان بارد\b",
                    r"\bالسوسيس كان مش سخن خالص\b",
                    r"\bالفطار المفروض يكون سخن بس كان فاتر/سايق\b",
                ],
                "zh": [
                    r"热食是凉的",
                    r"鸡蛋都凉了端出来",
                    r"香肠都是温的甚至冷的",
                    r"本来该是热菜的结果是冷的",
                ],
            },
        ),
        
        
        "coffee_bad": AspectRule(
            aspect_code="coffee_bad",
            polarity_hint="negative",
            display="Низкое качество кофе на завтраке",
            display_short="кофе плохой",
            long_hint="Гости фиксируют, что кофе на завтраке невкусный: водянистый, горчит, из автомата низкого качества, 'растворимый', без аромата. Это влияет на утреннее впечатление от сервиса F&B.",
            patterns_by_lang={{
                "ru": [
                    r"\bкофе невкусный\b",
                    r"\bужасный кофе\b",
                    r"\bкофе как вода\b",
                    r"\bрастворимый кофе отвратительный\b",
                    r"\bкофе был горький и жжёный\b",
                ],
                "en": [
                    r"\bbad coffee\b",
                    r"\bthe coffee was awful\b",
                    r"\bthe coffee tasted burnt\b",
                    r"\bwatery coffee\b",
                    r"\binstant coffee only and it was terrible\b",
                ],
                "tr": [
                    r"\bkahve berbattı\b",
                    r"\bkahve çok kötüydü\b",
                    r"\bkahve su gibiydi\b",
                    r"\byanık tadı vardı kahvenin\b",
                    r"\bsadece hazır/instant kahve vardı ve kötüydü\b",
                ],
                "ar": [
                    r"\bالقهوة طعمها وحش\b",
                    r"\bالقهوة شبه ماية\b",
                    r"\bقهوة محروقة الطعم\b",
                    r"\bقهوة سريعة ذي الزفت\b",
                ],
                "zh": [
                    r"咖啡很难喝",
                    r"咖啡像水一样淡",
                    r"咖啡有烧焦味",
                    r"只有速溶咖啡而且味道很差",
                ],
            },
        ),

        "breakfast_variety_good": AspectRule(
            aspect_code="breakfast_variety_good",
            polarity_hint="positive",
            display="Широкий выбор на завтраке",
            display_short="большой выбор",
            long_hint="Гости отмечают, что на завтраке был хороший ассортимент: несколько категорий блюд (горячее, холодные нарезки, овощи, фрукты, сладкое), подходящее как для 'континенталки', так и для сытного завтрака.",
            patterns_by_lang={{
                "ru": [
                    r"\bбольшой выбор на завтрак\b",
                    r"\bвыбор блюд был отличный\b",
                    r"\bассортимент хороший\b",
                    r"\bмного вариантов на завтрак\b",
                    r"\bкаждый найдёт что поесть\b",
                ],
                "en": [
                    r"\bgood variety at breakfast\b",
                    r"\bwide selection at breakfast\b",
                    r"\bplenty of options in the morning\b",
                    r"\ba lot to choose from\b",
                    r"\bwe had a lot of choice for breakfast\b",
                ],
                "tr": [
                    r"\bkahvaltıda seçenek çok fazlaydı\b",
                    r"\bkahvaltı çeşidi iyiydi\b",
                    r"\bçok seçenek vardı kahvaltıda\b",
                    r"\bherkes için bir şey vardı\b",
                ],
                "ar": [
                    r"\bالفطار منوع\b",
                    r"\bاختيارات كتير في الفطار\b",
                    r"\bفيه كذا نوع أكل الصبح\b",
                    r"\bكل واحد لقى حاجة تناسبه على الفطار\b",
                ],
                "zh": [
                    r"早餐种类很多",
                    r"早餐选择非常多",
                    r"早上可以选的东西很多",
                    r"基本上什么都有可以选",
                ],
            },
        ),
        
        
        "buffet_rich": AspectRule(
            aspect_code="buffet_rich",
            polarity_hint="positive",
            display="Полный/плотный завтрак-буфет",
            display_short="богатый буфет",
            long_hint="Гости подчёркивают, что буфет выглядит щедро: не только стандартные позиции, но и дополнительные блюда (горячее, десерты, локальные продукты). Это фиксируется как восприятие value в F&B.",
            patterns_by_lang={{
                "ru": [
                    r"\bбогатый шведский стол\b",
                    r"\bбуфет очень насыщенный\b",
                    r"\bшикарный выбор на буфете\b",
                    r"\bполный завтрак 'шведский стол'\b",
                    r"\bзавтрак выглядел очень щедро\b",
                ],
                "en": [
                    r"\brich breakfast buffet\b",
                    r"\bthe buffet was plentiful\b",
                    r"\bvery generous breakfast spread\b",
                    r"\bgreat buffet selection\b",
                    r"\bthe buffet was impressive\b",
                ],
                "tr": [
                    r"\bzengin açık büfe\b",
                    r"\baçık büfe çok doluydu\b",
                    r"\bkahvaltı büfesi oldukça zengindi\b",
                    r"\bçok kapsamlı kahvaltı büfesi\b",
                ],
                "ar": [
                    r"\bالبوفيه كان مليان أكل\b",
                    r"\bالفطار بوفيه تقيل ومتنوع\b",
                    r"\bالبوفيه الصبح كان كريم جداً\b",
                    r"\bشكل البوفيه كان محترم\b",
                ],
                "zh": [
                    r"自助早餐很丰富",
                    r"自助餐台很丰盛",
                    r"早餐摆得很豪华很多选择",
                    r"早餐供应特别充足",
                ],
            },
        ),
        
        
        "fresh_fruit_available": AspectRule(
            aspect_code="fresh_fruit_available",
            polarity_hint="positive",
            display="Наличие свежих фруктов",
            display_short="свежие фрукты",
            long_hint="Гости отдельно отмечают наличие свежих фруктов/салата из фруктов как элемента качества и заботы о здоровье гостей, а не только базового 'сытного' завтрака.",
            patterns_by_lang={{
                "ru": [
                    r"\bсвежие фрукты на завтраке\b",
                    r"\bмного свежих фруктов\b",
                    r"\bбыли фрукты\b",
                    r"\bнормальный выбор фруктов\b",
                    r"\bнарезанные фрукты свежие\b",
                ],
                "en": [
                    r"\bfresh fruit available\b",
                    r"\bfresh fruit at breakfast\b",
                    r"\bgood fruit selection\b",
                    r"\bplenty of fruit in the morning\b",
                    r"\bfresh fruit salad\b",
                ],
                "tr": [
                    r"\bfresh meyve vardı\b",
                    r"\bkahvaltıda taze meyve vardı\b",
                    r"\bmeyve seçeneği iyiydi\b",
                    r"\bdoğranmış taze meyveler vardı\b",
                ],
                "ar": [
                    r"\bكان فيه فواكه فريش\b",
                    r"\bفيه فواكه متقطعة وطازة\b",
                    r"\bاختيار حلو من الفواكه الصبح\b",
                    r"\bالفطار فيه فواكه طبيعية مش من علبة\b",
                ],
                "zh": [
                    r"早餐有新鲜水果",
                    r"水果很新鲜而且选择不少",
                    r"有新鲜切好的水果",
                    r"早餐提供鲜果拼盘",
                ],
            },
        ),
        
        
        "pastries_available": AspectRule(
            aspect_code="pastries_available",
            polarity_hint="positive",
            display="Наличие выпечки и десертной части завтрака",
            display_short="выпечка на завтраке",
            long_hint="Гости отмечают наличие круассанов, булочек, сладкой выпечки и т.п. как плюс ассортимента завтрака. Это воспринимается как деталь 'европейского завтрака' и добавляет ощущение уровня.",
            patterns_by_lang={{
                "ru": [
                    r"\bсвежие круассаны\b",
                    r"\bбыла выпечка\b",
                    r"\bна завтрак давали круассаны и булочки\b",
                    r"\bмного сладкой выпечки\b",
                    r"\bбыли круассаны/кексы на завтрак\b",
                ],
                "en": [
                    r"\bpastries for breakfast\b",
                    r"\bfresh croissants\b",
                    r"\bthere were pastries in the morning\b",
                    r"\bselection of sweet pastries\b",
                    r"\bmorning pastries and croissants\b",
                ],
                "tr": [
                    r"\bkahvaltıda kruvasanlar vardı\b",
                    r"\btatlı hamur işleri vardı\b",
                    r"\bfırından çıkmış poğaça/kruvasan vardı\b",
                    r"\bkahvaltıda hamur işi seçenekleri iyiydi\b",
                ],
                "ar": [
                    r"\bكان فيه كرواسون وفطير في الفطار\b",
                    r"\bفيه مخبوزات طازة الصبح\b",
                    r"\bفطار فيه مخبوزات وحاجات حلوة\b",
                    r"\bميني كرواسون/مخبوزات في البوفيه\b",
                ],
                "zh": [
                    r"早餐有可颂/牛角包等糕点",
                    r"有新鲜的烘焙点心",
                    r"早餐有甜点小面包",
                    r"有可颂和小蛋糕之类的",
                ],
            },
        ),
        
        
        "breakfast_variety_poor": AspectRule(
            aspect_code="breakfast_variety_poor",
            polarity_hint="negative",
            display="Ограниченный выбор на завтраке",
            display_short="малый выбор",
            long_hint="Гости фиксируют недостаток разнообразия: очень мало позиций, одно и то же каждый день, отсутствуют базовые категории (овощи, фрукты, горячее). Это воспринимается как невыполнение ожидания по завтраку.",
            patterns_by_lang={{
                "ru": [
                    r"\bмаленький выбор на завтрак\b",
                    r"\bвыбор очень скудный\b",
                    r"\bпрактически нечего выбрать\b",
                    r"\bассортимент очень бедный\b",
                    r"\bпочти ничего нет на завтрак\b",
                ],
                "en": [
                    r"\bpoor selection at breakfast\b",
                    r"\bvery limited breakfast options\b",
                    r"\bnot much to choose from\b",
                    r"\bthe breakfast was very basic\b",
                    r"\bthe buffet was quite limited\b",
                ],
                "tr": [
                    r"\bkahvaltı seçeneği çok kısıtlıydı\b",
                    r"\bçok az seçenek vardı\b",
                    r"\bkahvaltı çok basitti\b",
                    r"\bneredeyse hiçbir şey yoktu kahvaltıda\b",
                ],
                "ar": [
                    r"\bالفطار اختياراته قليلة جداً\b",
                    r"\bمافيش حاجات كتير تختار منها\b",
                    r"\bالفطار بسيط بزيادة\b",
                    r"\bمفيش تنوع في الفطار\b",
                ],
                "zh": [
                    r"早餐选择很少",
                    r"种类很单一",
                    r"没什么可以选的",
                    r"早餐很简单内容很少",
                ],
            },
        ),
    
        "breakfast_repetitive": AspectRule(
            aspect_code="breakfast_repetitive",
            polarity_hint="negative",
            display="Повторяющееся меню завтрака",
            display_short="одно и то же на завтрак",
            long_hint="Гости отмечают, что завтрак не меняется изо дня в день: одинаковые позиции каждый день без ротации.",
            patterns_by_lang={{
                "ru": [
                    r"\bкаждый день одно и то же на завтрак\b",
                    r"\bзавтрак не менялся\b",
                    r"\bодинаковый завтрак каждый день\b",
                    r"\bкаждое утро одно и то же\b",
                    r"\bникакого разнообразия по дням\b",
                ],
                "en": [
                    r"\bthe same breakfast every day\b",
                    r"\bbreakfast was the same every morning\b",
                    r"\bno change in breakfast\b",
                    r"\bit was always the same food\b",
                    r"\brepetitive breakfast\b",
                ],
                "tr": [
                    r"\bher gün aynı kahvaltıydı\b",
                    r"\bkahvaltı hiç değişmedi\b",
                    r"\bmenü hep aynıydı\b",
                    r"\bçeşit değişmiyordu\b",
                ],
                "ar": [
                    r"\bكل يوم نفس الفطار\b",
                    r"\bالفطار ما بيتغيرش\b",
                    r"\bمفيش تغيير خالص في الفطار\b",
                    r"\bكل صباح نفس الأصناف\b",
                ],
                "zh": [
                    r"每天早餐都一样",
                    r"早餐完全不变",
                    r"每天都是同样的东西",
                    r"早餐很重复没有变化",
                ],
            },
        ),
        
        
        "hard_to_find_food": AspectRule(
            aspect_code="hard_to_find_food",
            polarity_hint="negative",
            display="Неудобная подача позиций на завтраке",
            display_short="сложно найти еду",
            long_hint="Гости сообщают, что на завтраке трудно ориентироваться.",
            patterns_by_lang={{
                "ru": [
                    r"\bтрудно было найти, где что лежит\b",
                    r"\bнепонятно где кофе/чашки/ложки\b",
                    r"\bничего не подписано\b",
                    r"\bпосуду и еду приходилось искать\b",
                    r"\bнадо было спрашивать где что\b",
                ],
                "en": [
                    r"\bhard to find things at breakfast\b",
                    r"\bwe couldn't find the coffee/plates/etc\b",
                    r"\bnothing was labeled\b",
                    r"\bit wasn't clear where anything was\b",
                    r"\bhad to ask where basic items were\b",
                ],
                "tr": [
                    r"\bkahvaltıda neyin nerede olduğunu bulmak zordu\b",
                    r"\btabak/çatal vs. nerede belli değildi\b",
                    r"\bhiçbir şey etiketlenmemişti\b",
                    r"\bkahve/su/ekmek neredeydi anlamak zor oldu\b",
                ],
                "ar": [
                    r"\bمش عارفين فين الحاجات على الفطار\b",
                    r"\bمفيش أي توضيح فين القهوة أو الأطباق\b",
                    r"\bمفيش لابيلز على الأكل\b",
                    r"\bاضطرينا نسأل على كل حاجة في الفطار\b",
                ],
                "zh": [
                    r"早餐不知道什么东西放哪里",
                    r"找不到咖啡/餐具/果汁在哪里",
                    r"都没有标签标识",
                    r"摆放很乱不好找",
                ],
            },
        ),
        
        
        "breakfast_staff_friendly": AspectRule(
            aspect_code="breakfast_staff_friendly",
            polarity_hint="positive",
            display="Доброжелательность персонала завтрака",
            display_short="дружелюбный персонал завтрака",
            long_hint="Гости подчеркивают, что команда в зоне завтрака общается вежливо и позитивно: приветствуют, улыбаются, создают ощущение гостеприимства с самого утра.",
            patterns_by_lang={{
                "ru": [
                    r"\bперсонал на завтраке дружелюбный\b",
                    r"\bнас очень приветливо встретили на завтраке\b",
                    r"\bочень улыбчивые сотрудники в ресторане утром\b",
                    r"\bперсонал за завтраком был очень милая\b",
                ],
                "en": [
                    r"\bfriendly breakfast staff\b",
                    r"\bthe staff at breakfast were very friendly\b",
                    r"\bwelcoming breakfast team\b",
                    r"\bthey greeted us with a smile in the morning\b",
                ],
                "tr": [
                    r"\bkahvaltı personeli çok güleryüzlüydü\b",
                    r"\bsabah servis eden ekip çok nazikti\b",
                    r"\bkahvaltıdaki çalışanlar çok samimiydi\b",
                    r"\bgüler yüzlü servis sabah\b",
                ],
                "ar": [
                    r"\bالستاف بتاع الفطار كان لطيف جداً\b",
                    r"\bالناس في الإفطار كانوا بشوشين\b",
                    r"\bاستقبال لطيف في الفطار\b",
                    r"\bالموظفين في الإفطار كانوا ودودين\b",
                ],
                "zh": [
                    r"早餐服务人员很友好",
                    r"早餐工作人员很热情",
                    r"早上餐厅的人很有礼貌又微笑",
                    r"早餐服务团队很亲切",
                ],
            },
        ),
        
        
        "breakfast_staff_attentive": AspectRule(
            aspect_code="breakfast_staff_attentive",
            polarity_hint="positive",
            display="Внимательность персонала завтрака",
            display_short="внимательный сервис утром",
            long_hint="Гости отмечают проактивный сервис на завтраке: сотрудники быстро убирают использованную посуду, спрашивают о необходимости кофе/дополнений, помогают с осадкой гостей.",
            patterns_by_lang={{
                "ru": [
                    r"\bочень внимательный персонал на завтраке\b",
                    r"\bпостоянно подходили и спрашивали нужно ли ещё что\-то\b",
                    r"\bочень быстро убирали посуду\b",
                    r"\bнам сразу помогли найти стол\b",
                ],
                "en": [
                    r"\battentive breakfast staff\b",
                    r"\bthe breakfast staff were very attentive\b",
                    r"\bthey kept checking if we needed anything\b",
                    r"\bthey cleared the tables quickly\b",
                    r"\bthey helped us get seated right away\b",
                ],
                "tr": [
                    r"\bkahvaltı personeli çok dikkatliydi\b",
                    r"\bhemen masayı temizlediler\b",
                    r"\bbir şeye ihtiyacımız var mı diye sordular\b",
                    r"\bmasayı hemen ayarladılar\b",
                ],
                "ar": [
                    r"\bالستاف في الفطار كانوا مركزين معانا\b",
                    r"\bبينضفوا الترابيزات بسرعة\b",
                    r"\bبيسألوا لو محتاجين حاجة تانية\b",
                    r"\bساعدونا نقعد بسرعة\b",
                ],
                "zh": [
                    r"早餐服务员很细心",
                    r"一直会过来问还需要什么",
                    r"桌子收得很快保持干净",
                    r"员工很快帮我们安排座位",
                ],
            },
        ),
        
        
        "buffet_refilled_quickly": AspectRule(
            aspect_code="buffet_refilled_quickly",
            polarity_hint="positive",
            display="Оперативное пополнение буфета",
            display_short="буфет быстро пополняют",
            long_hint="Гости отмечают, что когда блюда заканчивались, персонал оперативно выкладывал новую партию без долгих пауз.",
            patterns_by_lang={{
                "ru": [
                    r"\bбуфет постоянно пополняли\b",
                    r"\bбыстро приносили свежие блюда\b",
                    r"\bничего не успевало закончиться\b",
                    r"\bеда не заканчивалась надолго\b",
                ],
                "en": [
                    r"\bthey kept refilling the buffet\b",
                    r"\bthe buffet was topped up quickly\b",
                    r"\bfood was replenished fast\b",
                    r"\bit never felt empty\b",
                    r"\bwhen something ran out they brought more right away\b",
                ],
                "tr": [
                    r"\baçık büfe hemen yenilendi\b",
                    r"\byemekler hemen tazelendi\b",
                    r"\bbitince anında yenisini getirdiler\b",
                    r"\bbüfe boş kalmadı\b",
                ],
                "ar": [
                    r"\bالبوفيه كانوا بيعوضوه بسرعة\b",
                    r"\bالأكل أول ما يخلص بيحطوا تاني فوراً\b",
                    r"\bماحسّناش إن البوفيه فاضي\b",
                    r"\bبيرجعوا يملوا الأكل بسرعة\b",
                ],
                "zh": [
                    r"自助台会及时补菜",
                    r"食物一没就马上补上",
                    r"餐台很少出现空盘",
                    r"补餐速度很快",
                ],
            },
        ),

        "tables_cleared_fast": AspectRule(
            aspect_code="tables_cleared_fast",
            polarity_hint="positive",
            display="Быстрая уборка столов на завтраке",
            display_short="столы быстро убирают",
            long_hint="Гости отмечают, что использованная посуда оперативно убирается, столы быстро приводятся в порядок и готовы для следующих гостей. Это отражает эффективность утреннего сервиса в зоне завтрака.",
            patterns_by_lang={{
                "ru": [
                    r"\bстолы быстро убирали\b",
                    r"\bпосуда убиралась сразу\b",
                    r"\bгрязную посуду забирали оперативно\b",
                    r"\bне копилась грязная посуда\b",
                    r"\bстол быстро протёрли\b",
                ],
                "en": [
                    r"\btables were cleared quickly\b",
                    r"\bthey cleared the tables right away\b",
                    r"\bdirty dishes were taken immediately\b",
                    r"\bthey cleaned our table fast\b",
                    r"\bthe staff removed plates quickly\b",
                ],
                "tr": [
                    r"\bmasalar hemen temizleniyordu\b",
                    r"\bkirli tabakları anında aldılar\b",
                    r"\bmasayı çok hızlı topladılar\b",
                    r"\bmasa çabucak silindi\b",
                ],
                "ar": [
                    r"\bكانوا بينضفوا الترابيز بسرعة\b",
                    r"\bشالوا الأطباق على طول\b",
                    r"\bما بيسيبوش الصحون متكدسة\b",
                    r"\bمسحوا الترابيزة فوراً بعد ما خلصنا\b",
                ],
                "zh": [
                    r"桌子收得很快",
                    r"服务员马上把碗盘收走",
                    r"桌面很快就被清理干净",
                    r"基本没有脏盘子堆在桌上",
                ],
            },
        ),
        
        
        "breakfast_staff_rude": AspectRule(
            aspect_code="breakfast_staff_rude",
            polarity_hint="negative",
            display="Грубость персонала завтрака",
            display_short="грубый персонал завтрака",
            long_hint="Гости фиксируют невежливое или холодное поведение сотрудников на завтраке: резкий тон, отсутствие приветствия, раздражённые ответы, ощущение, что гость мешает. Это отмечается как сбой сервиса F&B с точки зрения отношения к гостю.",
            patterns_by_lang={{
                "ru": [
                    r"\bперсонал на завтраке был грубый\b",
                    r"\bна нас накричали за завтраком\b",
                    r"\bобслуживание утром было хамское\b",
                    r"\bнеприветливый персонал в зале завтрака\b",
                    r"\bна завтраке с нами разговаривали резко\b",
                ],
                "en": [
                    r"\brude breakfast staff\b",
                    r"\bthe staff at breakfast were rude\b",
                    r"\bthey were not polite in the breakfast area\b",
                    r"\bthe way they spoke at breakfast was rude\b",
                    r"\bvery unfriendly breakfast service\b",
                ],
                "tr": [
                    r"\bkahvaltı personeli kaba davrandı\b",
                    r"\bsabah servis edenler hiç güler yüzlü değildi\b",
                    r"\bkahvaltıda bizimle ters konuştular\b",
                    r"\bilgisiz ve kaba personel vardı\b",
                ],
                "ar": [
                    r"\bالستاف بتاع الفطار كان معاملته وحشة\b",
                    r"\bكلمونا بأسلوب مش لطيف في الفطار\b",
                    r"\bالموظفين الصبح كانوا عصبيين\b",
                    r"\bمعاملة مش محترمة على الفطار\b",
                ],
                "zh": [
                    r"早餐服务员态度很差",
                    r"早餐区员工很没礼貌",
                    r"说话很冲在早餐时段",
                    r"早餐服务感觉很不友好",
                ],
            },
        ),
        
        
        "no_refill_food": AspectRule(
            aspect_code="no_refill_food",
            polarity_hint="negative",
            display="Буфет не пополняется",
            display_short="еду не пополняют",
            long_hint="Гости отмечают, что когда блюда на шведском столе заканчивались, их не обновляли или обновляли с большой задержкой. Это фиксируется как операционная недоступность завтрака на пиковых слотах.",
            patterns_by_lang={{
                "ru": [
                    r"\bкогда что\-то заканчивалось, не приносили новое\b",
                    r"\bшведский стол не пополняли\b",
                    r"\bеды не докладывали\b",
                    r"\bмногие блюда закончились и их не вернули\b",
                    r"\bуже нечего было взять, и никто не добавил\b",
                ],
                "en": [
                    r"\bthey didn't refill the buffet\b",
                    r"\bfood ran out and was not replaced\b",
                    r"\bno one brought more when it was gone\b",
                    r"\bby the time we arrived most of the food was gone\b",
                    r"\bempty trays stayed empty\b",
                ],
                "tr": [
                    r"\bbüfe yenilenmedi\b",
                    r"\byemekler bitince yenisi gelmedi\b",
                    r"\btabaklar boş kaldı kimse doldurmadı\b",
                    r"\bgeldiğimizde çoğu bitmişti ve tazelenmedi\b",
                ],
                "ar": [
                    r"\bالأكل خلص ومحدش زوّد\b",
                    r"\bالصواني فضيت وما اتعمرتِش تاني\b",
                    r"\bجينا متأخر شوية ومكانش فيه أكل ومتزودش\b",
                    r"\bمحدش رجّع الأكل بعد ما خلص\b",
                ],
                "zh": [
                    r"自助台的菜没再补",
                    r"很多都被拿光也没人补",
                    r"我们来时盘子都是空的还没人加",
                    r"托盘空了就一直空着",
                ],
            },
        ),
        
        
        "tables_left_dirty": AspectRule(
            aspect_code="tables_left_dirty",
            polarity_hint="negative",
            display="Грязные столы на завтраке",
            display_short="грязные столы",
            long_hint="Гости сообщают, что столы долго остаются неубранными: посуда не убирается, крошки и остатки еды остаются на поверхности. Это воспринимается как сбой в поддержании гигиены зала завтрака.",
            patterns_by_lang={{
                "ru": [
                    r"\bгрязные столы на завтраке\b",
                    r"\bстолы не убирали долго\b",
                    r"\bгрязная посуда стояла и не убиралась\b",
                    r"\bнекуда сесть, все столы завалены посудой\b",
                    r"\bстол липкий и никто не протёр\b",
                ],
                "en": [
                    r"\bdirty tables at breakfast\b",
                    r"\bno one cleared the dirty tables\b",
                    r"\bthere were no clean tables to sit at\b",
                    r"\bplates and crumbs left on the tables\b",
                    r"\bthe tables were sticky and not wiped\b",
                ],
                "tr": [
                    r"\bkahvaltıda masalar kirliydi\b",
                    r"\bmasalar uzun süre temizlenmedi\b",
                    r"\boturacak temiz masa yoktu\b",
                    r"\btabaklar masalarda bırakılmıştı\b",
                ],
                "ar": [
                    r"\bالترابيزات الصبح كانت متوسخة\b",
                    r"\bمحدش بينضف الترابيزات بسرعة\b",
                    r"\bمافيش ترابيزة فاضية نضيفة نقعد عليها\b",
                    r"\bالأطباق والقشور لسه على الترابيزة\b",
                ],
                "zh": [
                    r"早餐时桌子都很脏没人收拾",
                    r"桌面上都是剩的盘子没人擦",
                    r"找不到干净的桌子可以坐",
                    r"桌子黏黏的没有擦干净",
                ],
            },
        ),
        
        
        "ignored_requests": AspectRule(
            aspect_code="ignored_requests",
            polarity_hint="negative",
            display="Запросы гостей на завтраке игнорируются",
            display_short="запросы игнорировали",
            long_hint="Гости фиксируют, что обращения к персоналу (пополнить блюда, принести кофе, убрать стол, помочь с посудой для ребёнка и т.д.) не были обработаны или были проигнорированы. Это трактуется как отсутствие сервиса ранним днём.",
            patterns_by_lang={{
                "ru": [
                    r"\bнас просто проигнорировали\b",
                    r"\bперсонал не реагировал на просьбы\b",
                    r"\bпопросили кофе и так и не принесли\b",
                    r"\bникто не подошёл даже после просьбы\b",
                    r"\bобращения остались без ответа\b",
                ],
                "en": [
                    r"\bour requests were ignored\b",
                    r"\bthe staff didn't respond when we asked\b",
                    r"\bwe asked for coffee and nobody came\b",
                    r"\bno one came even after we asked for help\b",
                    r"\bthey didn't seem interested in helping\b",
                ],
                "tr": [
                    r"\bristeklerimizi dikkate almadılar\b",
                    r"\bkahve istedik ama kimse getirmedi\b",
                    r"\byardım istedik ama kimse gelmedi\b",
                    r"\bpersonel ilgilenmedi\b",
                ],
                "ar": [
                    r"\bطلبنا ومحدش رد علينا\b",
                    r"\bطلبنا قهوة وماحدش جاب\b",
                    r"\bمحدش ساعد حتى بعد ما نادينا\b",
                    r"\bحسينا إنهم طنشونا\b",
                ],
                "zh": [
                    r"我们提的要求没人理",
                    r"叫了服务员没人来",
                    r"要咖啡也没人送过来",
                    r"请求帮助也没有回应",
                ],
            },
        ),

        "food_enough_for_all": AspectRule(
            aspect_code="food_enough_for_all",
            polarity_hint="positive",
            display="Достаточное количество еды на завтраке",
            display_short="еды хватает всем",
            long_hint="Гости отмечают, что еды на завтраке было достаточно для всех гостей, даже в пиковые часы: ничего не заканчивалось критично, не приходилось 'успевать раньше других'. Это фиксируется как устойчивость сервиса в высокий поток.",
            patterns_by_lang={{
                "ru": [
                    r"\bвсем хватало еды\b",
                    r"\bеды достаточно для всех\b",
                    r"\bничего не заканчивалось\b",
                    r"\bдаже поздно всё ещё было что поесть\b",
                    r"\bне нужно было бороться за еду\b",
                ],
                "en": [
                    r"\bplenty of food for everyone\b",
                    r"\bthere was enough food for all guests\b",
                    r"\bfood didn't run out\b",
                    r"\bstill plenty of choice even later\b",
                    r"\bwe didn't have to rush to get food\b",
                ],
                "tr": [
                    r"\bherkes için yeterince yemek vardı\b",
                    r"\byemek bitmedi\b",
                    r"\bgeç inseniz bile yemek kalıyor\b",
                    r"\bkimse yemek için yarışmıyordu\b",
                ],
                "ar": [
                    r"\bالأكل كان مكفي الكل\b",
                    r"\bماخلصش الأكل حتى مع الزحمة\b",
                    r"\bلسه في أكل حتى لو نزلت متأخر\b",
                    r"\bماكانش فيه إحساس إنك لازم تلحق الأكل\b",
                ],
                "zh": [
                    r"早餐的食物足够 everyone",
                    r"食物没有被拿光",
                    r"即使晚一点下来也还有吃的",
                    r"不用抢食物",
                ],
            },
        ),
        
        
        "kept_restocking": AspectRule(
            aspect_code="kept_restocking",
            polarity_hint="positive",
            display="Регулярное пополнение позиций во время сервиса",
            display_short="еду постоянно докладывали",
            long_hint="Гости подчёркивают, что команда завтрака регулярно выкладывала свежие партии блюд и напитков в течение сервиса, а не только в начале. Это сигнал стабильного потока сервиса, а не 'один заход еды'.",
            patterns_by_lang={{
                "ru": [
                    r"\bпостоянно подносили свежие блюда\b",
                    r"\bвсё время докладывали еду\b",
                    r"\bнапитки и еда пополнялись без остановки\b",
                    r"\bбуфет поддерживали в полном виде\b",
                ],
                "en": [
                    r"\bthey kept restocking during breakfast\b",
                    r"\bconstantly refilled throughout service\b",
                    r"\bthey kept topping things up\b",
                    r"\bthe buffet was maintained the whole time\b",
                ],
                "tr": [
                    r"\bsürekli tazelediler\b",
                    r"\bkahvaltı boyunca yemekler yenilendi\b",
                    r"\bbüfe bütün servis boyunca doluydu\b",
                    r"\beksilen her şey hemen konuldu\b",
                ],
                "ar": [
                    r"\bكانوا كل شوية يعيدوا يملوا الأكل\b",
                    r"\bالبوفيه بيفضل مليان طول وقت الفطار\b",
                    r"\bمش بس أول نص ساعة، بيفضلوا يزوّدوا\b",
                    r"\bالمشاريب والأكل بيتجدد طول الوقت\b",
                ],
                "zh": [
                    r"早餐期间一直在补菜",
                    r"不是只在一开始，后面也会不断补充",
                    r"餐台全程都保持充足",
                    r"饮料和食物都会及时补上",
                ],
            },
        ),
        
        
        "tables_available": AspectRule(
            aspect_code="tables_available",
            polarity_hint="positive",
            display="Наличие свободных столов",
            display_short="были свободные столы",
            long_hint="Гости отмечают, что в зале завтрака можно было без проблем найти место: не приходилось ждать, не нужно делить стол с незнакомыми гостями. Это фиксируется как организационный комфорт зоны сервиса.",
            patterns_by_lang={{
                "ru": [
                    r"\bлегко нашли стол\b",
                    r"\bбыли свободные столики\b",
                    r"\bне пришлось ждать место\b",
                    r"\bвсегда был столик\b",
                    r"\bзал не был переполнен\b",
                ],
                "en": [
                    r"\bwe easily found a table\b",
                    r"\bplenty of tables available\b",
                    r"\bwe didn't have to wait to sit\b",
                    r"\bthere was always somewhere to sit\b",
                    r"\bthe breakfast area wasn't overcrowded\b",
                ],
                "tr": [
                    r"\bkolayca masa bulduk\b",
                    r"\bboş masa vardı\b",
                    r"\bmasa için beklemedik\b",
                    r"\bkahvaltı alanı çok kalabalık değildi\b",
                ],
                "ar": [
                    r"\bلقينا ترابيزة على طول\b",
                    r"\bكان فيه ترابيزات فاضية\b",
                    r"\bما استنّاش مكان نقعد فيه\b",
                    r"\bالمكان مش زحمة زيادة عن اللزوم\b",
                ],
                "zh": [
                    r"很容易找到位子坐",
                    r"有空桌不用等",
                    r"早餐区域不拥挤",
                    r"基本都有桌子可以坐",
                ],
            },
        ),
        
        
        "no_queue": AspectRule(
            aspect_code="no_queue",
            polarity_hint="positive",
            display="Отсутствие очередей на завтраке",
            display_short="без очередей",
            long_hint="Гости отмечают отсутствие очередей у кофемашины, фуд-станций или входа в зал. Это отражает хорошую распределённость потока гостей и достаточный объём сервиса на человека.",
            patterns_by_lang={{
                "ru": [
                    r"\bне было очередей\b",
                    r"\bникаких очередей на завтрак\b",
                    r"\bне стояли в очереди за едой\b",
                    r"\bне приходилось ждать кофе\b",
                    r"\bдоступ к буфету без очереди\b",
                ],
                "en": [
                    r"\bno queues for breakfast\b",
                    r"\bno waiting in line for food\b",
                    r"\bwe didn't have to queue for coffee\b",
                    r"\bwe could just walk up and get food\b",
                    r"\bbreakfast without any lines\b",
                ],
                "tr": [
                    r"\bkahvaltıda sıra yoktu\b",
                    r"\byemek almak için beklemedik\b",
                    r"\bkahve için kuyruk olmadı\b",
                    r"\bçok beklemeden direkt alabildik\b",
                ],
                "ar": [
                    r"\bمفيش طابور على الفطار\b",
                    r"\bما وقفناش في دور عالأكل\b",
                    r"\bحتى على القهوة مفيش انتظار\b",
                    r"\bتدخل تاخد اللي انت عايزه على طول\b",
                ],
                "zh": [
                    r"早餐基本不用排队",
                    r"拿食物不用等",
                    r"咖啡机前没有排长队",
                    r"可以直接拿东西吃不用排队",
                ],
            },
        ),
        
        
        "breakfast_flow_ok": AspectRule(
            aspect_code="breakfast_flow_ok",
            polarity_hint="positive",
            display="Организация потока гостей на завтраке",
            display_short="логичный поток гостей",
            long_hint="Гости описывают завтрак как хорошо организованный с точки зрения работы с потоками людей.",
            patterns_by_lang={{
                "ru": [
                    r"\bудобно организован завтрак\b",
                    r"\bграмотный поток гостей\b",
                    r"\bничего не загромождено\b",
                    r"\bлюди не толпились\b",
                    r"\bвсё продумано по зонам\b",
                ],
                "en": [
                    r"\bthe breakfast flow was good\b",
                    r"\bthe layout worked well, no crowding\b",
                    r"\bwell organized breakfast service\b",
                    r"\bthe stations were well arranged\b",
                    r"\bpeople were not piling up in one spot\b",
                ],
                "tr": [
                    r"\bkahvaltı alanı düzenliydi\b",
                    r"\binsanlar yığılmıyordu\b",
                    r"\bistasyonlar mantıklı yerleştirilmişti\b",
                    r"\bservis akışı rahattı\b",
                ],
                "ar": [
                    r"\bتنظيم الفطار كان مريح\b",
                    r"\bالناس ماكانوش مزاحمين على نفس المكان\b",
                    r"\bالتقسيمة واضحة ومريحة\b",
                    r"\bالدنيا ماشية بنظام في الفطار\b",
                ],
                "zh": [
                    r"早餐动线很顺",
                    r"区域分得很清楚不会挤在一起",
                    r"早餐的布局很合理",
                    r"人流动得很顺畅没有拥挤",
                ],
            },
        ),

        "food_ran_out": AspectRule(
            aspect_code="food_ran_out",
            polarity_hint="negative",
            display="Еда закончилась во время сервиса завтрака",
            display_short="еды не хватило",
            long_hint="Гости фиксируют, что ключевые позиции завтрака закончились и недоступны для части гостей (яичница, колбасы, выпечка, соки и т.д.). Это воспринимается как дефицит сервиса при пиковом спросе.",
            patterns_by_lang={{
                "ru": [
                    r"\bеда закончилась\b",
                    r"\bмногое уже было разобрано\b",
                    r"\bк моменту когда мы пришли почти ничего не осталось\b",
                    r"\bне осталось яичницы/выпечки\b",
                    r"\bничего вкусного уже не было\b",
                ],
                "en": [
                    r"\bthe food ran out\b",
                    r"\bmost of the food was gone\b",
                    r"\bby the time we came there was almost nothing left\b",
                    r"\bno eggs/bacon left\b",
                    r"\bnothing decent was left on the buffet\b",
                ],
                "tr": [
                    r"\byemek bitti\b",
                    r"\bgeldiğimizde çoğu yiyecek kalmamıştı\b",
                    r"\byumurta/sosis kalmamıştı\b",
                    r"\bneredeyse hiçbir şey kalmadı kahvaltıda\b",
                ],
                "ar": [
                    r"\bالأكل خلص\b",
                    r"\bإحنا لما نزلنا كان مفيش حاجة باقية\b",
                    r"\bمفيش بيض/سوسيس/حاجات سخنة متوفرة\b",
                    r"\bالبوفيه كان شبه فاضي\b",
                ],
                "zh": [
                    r"吃的基本都被拿光了",
                    r"我们下去时几乎什么都没剩",
                    r"鸡蛋/培根都没有了",
                    r"自助台几乎空了",
                ],
            },
        ),
        
        
        "not_restocked": AspectRule(
            aspect_code="not_restocked",
            polarity_hint="negative",
            display="Позиции не восполнялись после окончания",
            display_short="не пополняли",
            long_hint="Гости отмечают, что даже после обращения к персоналу законченные блюда/напитки не возвращались на буфет. Это фиксируется как сбой в поддержании стандартов завтрака при высокой загрузке.",
            patterns_by_lang={{
                "ru": [
                    r"\bничего не донесли\b",
                    r"\bне пополнили даже после просьбы\b",
                    r"\bпопросили добавить, но так и не добавили\b",
                    r"\bзакончилось и больше не приносили\b",
                ],
                "en": [
                    r"\bthey didn't restock\b",
                    r"\bwe asked and they never brought more\b",
                    r"\bonce it was gone they didn't replace it\b",
                    r"\bnothing was refilled even after we asked\b",
                ],
                "tr": [
                    r"\btazelemediler\b",
                    r"\bistememize rağmen yenisini getirmediler\b",
                    r"\bbiten şeyler geri konmadı\b",
                    r"\bhemen hemen hiçbir şey yenilenmedi\b",
                ],
                "ar": [
                    r"\bطلبنا يزودوا الأكل ومحدش زوّد\b",
                    r"\bالأكل خلص ومحدش رجع زوّد\b",
                    r"\bحتى بعد ما قلنا، مازودوش حاجة\b",
                    r"\bمفيش إعادة تعبئة خالص\b",
                ],
                "zh": [
                    r"我们问了也没有再补",
                    r"东西没了就不补了",
                    r"员工也没有再拿新的上来",
                    r"请求补菜也没反应",
                ],
            },
        ),
        
        
        "had_to_wait_food": AspectRule(
            aspect_code="had_to_wait_food",
            polarity_hint="negative",
            display="Ожидание блюд/дозагрузки завтрака",
            display_short="ждали еду",
            long_hint="Гости сообщают, что им пришлось ждать, пока сотрудники принесут еду.",
            patterns_by_lang={{
                "ru": [
                    r"\bпришлось ждать еду\b",
                    r"\bждали пока что\-то приготовят\b",
                    r"\bждали пока вынесут яичницу/горячее\b",
                    r"\bнужно было подождать пока пополнят\b",
                    r"\bочередь просто стояла и ждала новую порцию\b",
                ],
                "en": [
                    r"\bwe had to wait for food\b",
                    r"\bwe had to wait for them to bring more\b",
                    r"\bwaited for fresh eggs/sausages to come out\b",
                    r"\bhad to wait for the buffet to be refilled\b",
                ],
                "tr": [
                    r"\byemek gelmesini bekledik\b",
                    r"\bçıkması için yumurta/sıcak yemek bekledik\b",
                    r"\bbüfenin yenilenmesini beklemek zorunda kaldık\b",
                    r"\bkahvaltı için sırada bekledik çünkü daha koymamışlardı\b",
                ],
                "ar": [
                    r"\bاستنّينا الأكل يطلع\b",
                    r"\bفضلنا مستنيين يزودوا البيض/الأكل السخن\b",
                    r"\bكان لازم نستنى عشان يجيبوا أكل زيادة\b",
                    r"\bالبوفيه فاضي واضطرينا نستنى لحد ما يملوه تاني\b",
                ],
                "zh": [
                    r"我们得等他们再补食物",
                    r"等了一阵才有新的热菜上来",
                    r"为了拿早餐还要等他们现做",
                    r"自助台空了只能等补菜",
                ],
            },
        ),
        
        
        "no_tables_available": AspectRule(
            aspect_code="no_tables_available",
            polarity_hint="negative",
            display="Дефицит посадки на завтраке",
            display_short="нет свободных столов",
            long_hint="Гости фиксируют, что в зоне завтрака не было свободных столов: приходилось ждать, стоять с тарелками или делить стол с другими. Это отражает перегрузку инфраструктуры завтрака.",
            patterns_by_lang={{
                "ru": [
                    r"\bне было свободных столов\b",
                    r"\bждали стол\b",
                    r"\bпришлось стоять с тарелками\b",
                    r"\bнегде было сесть позавтракать\b",
                    r"\bзал перегружен, посадки не хватает\b",
                ],
                "en": [
                    r"\bno tables available\b",
                    r"\bwe had to wait for a table\b",
                    r"\bnowhere to sit for breakfast\b",
                    r"\bwe were standing around with our plates\b",
                    r"\bthe breakfast area was overcrowded\b",
                ],
                "tr": [
                    r"\bboş masa yoktu\b",
                    r"\bmasa beklemek zorunda kaldık\b",
                    r"\bkahvaltıda oturacak yer yoktu\b",
                    r"\btabağımızla ayakta bekledik\b",
                ],
                "ar": [
                    r"\bماكانش فيه ترابيزة فاضية\b",
                    r"\bاستنينا عشان نقعد\b",
                    r"\bكنا واقفين بالأكل في إيدينا\b",
                    r"\bمكان الفطار زحمة جداً ومفيش قعدة\b",
                ],
                "zh": [
                    r"没有空位可以坐着吃早餐",
                    r"我们拿着盘子站着等桌子",
                    r"早餐区太挤根本没桌子",
                    r"还得排队等桌位才能吃",
                ],
            },
        ),
        
        
        "long_queue": AspectRule(
            aspect_code="long_queue",
            polarity_hint="negative",
            display="Очереди на завтраке",
            display_short="длинные очереди",
            long_hint="Гости отмечают очереди к станциям завтрака (кофемашина, горячие блюда, вход в зал).",
            patterns_by_lang={{
                "ru": [
                    r"\bбольшая очередь на завтрак\b",
                    r"\bприходилось стоять в очереди\b",
                    r"\bочередь за кофе огромная\b",
                    r"\bтолпа у шведского стола\b",
                    r"\bочередь чтобы просто взять еду\b",
                ],
                "en": [
                    r"\blong line for breakfast\b",
                    r"\bwe had to wait in a long line\b",
                    r"\blong queue for coffee\b",
                    r"\bthere was a queue just to get food\b",
                    r"\bcrowds around the buffet\b",
                ],
                "tr": [
                    r"\bkahvaltıda uzun kuyruk vardı\b",
                    r"\bkahve için sıra bekledik\b",
                    r"\byemek almak için sıra beklemek zorunda kaldık\b",
                    r"\bbüfenin önünde kuyruk oluşuyordu\b",
                ],
                "ar": [
                    r"\bكان في طابور كبير على الفطار\b",
                    r"\bاستنّينا دور طويل عالقهوة\b",
                    r"\bكان لازم تقف في طابور عشان تاخد أكل\b",
                    r"\bزحمة قدام البوفيه\b",
                ],
                "zh": [
                    r"早餐要排很长的队",
                    r"拿咖啡要排队很久",
                    r"为了拿食物要排长队",
                    r"自助台那里人挤人要排队",
                ],
            },
        ),

        "breakfast_area_clean": AspectRule(
            aspect_code="breakfast_area_clean",
            polarity_hint="positive",
            display="Чистота зоны завтрака",
            display_short="чистый зал завтрака",
            long_hint="Гости отмечают, что зал завтрака и линии раздачи содержались в чистоте: полы и столы аккуратные, вокруг буфета нет мусора и подтёков.",
            patterns_by_lang={{
                "ru": [
                    r"\bзона завтрака чист(ая|о)\b",
                    r"\bв зале для завтрака очень чисто\b",
                    r"\bбуфет аккуратный и чистый\b",
                    r"\bутром всё чисто и опрятно\b",
                ],
                "en": [
                    r"\bbreakfast area was clean\b",
                    r"\bvery clean dining area\b",
                    r"\bbuffet area looked clean\b",
                    r"\beverything was clean in the morning\b",
                ],
                "tr": [
                    r"\bkahvaltı alanı temizdi\b",
                    r"\bçok temiz bir kahvaltı salonu\b",
                    r"\bbüfe alanı tertemizdi\b",
                    r"\bsabah her yer temizdi\b",
                ],
                "ar": [
                    r"\bمكان الفطار نضيف\b",
                    r"\bالمنطقة كانت نضيفة الصبح\b",
                    r"\bالبوفيه شكله نضيف\b",
                    r"\bالنظافة في الفطار كانت كويسة\b",
                ],
                "zh": [
                    r"早餐区很干净",
                    r"早上的用餐区很整洁",
                    r"自助区看起来很干净",
                    r"整体保持得很干净",
                ],
            },
        ),
        
        
        "tables_cleaned_quickly": AspectRule(
            aspect_code="tables_cleaned_quickly",
            polarity_hint="positive",
            display="Столы оперативно протираются",
            display_short="столы быстро моют",
            long_hint="Гости отмечают, что столешницы оперативно протираются и готовятся к следующей посадке; используются салфетки/санитайзеры, нет липкости и следов от предыдущих гостей.",
            patterns_by_lang={{
                "ru": [
                    r"\bстолы быстро протирали\b",
                    r"\bстол протёрли сразу\b",
                    r"\bоперативно мыли столы\b",
                    r"\bникакой липкости на столах\b",
                ],
                "en": [
                    r"\btables were wiped quickly\b",
                    r"\bthey cleaned the tables promptly\b",
                    r"\btables were sanitized fast\b",
                    r"\bno sticky tables\b",
                ],
                "tr": [
                    r"\bmasalar hızlıca siliniyordu\b",
                    r"\bmasayı hemen temizlediler\b",
                    r"\bmasalar çabucak dezenfekte edildi\b",
                ],
                "ar": [
                    r"\bكانوا بيمسحوا الترابيزات بسرعة\b",
                    r"\bنضفوا الترابيزة فوراً\b",
                    r"\bمفيش لزقة على الترابيزات\b",
                ],
                "zh": [
                    r"桌面会很快被擦干净",
                    r"他们立即清洁桌子",
                    r"桌子很快就消毒整理好",
                    r"桌面不粘不脏",
                ],
            },
        ),
        
        
        "dirty_tables": AspectRule(
            aspect_code="dirty_tables",
            polarity_hint="negative",
            display="Грязные/липкие столы в зоне завтрака",
            display_short="грязные столы",
            long_hint="Гости фиксируют, что столы остаются грязными: крошки, липкие следы, разводы. Это относится к гигиене зала, а не к скорости обслуживания.",
            patterns_by_lang={{
                "ru": [
                    r"\bгрязные стол(ы|а)\b",
                    r"\bстолы липкие\b",
                    r"\bкрошки на столах\b",
                    r"\bстол не протёрт\b",
                ],
                "en": [
                    r"\bdirty tables\b",
                    r"\bsticky tables\b",
                    r"\bcrumbs left on the tables\b",
                    r"\btables were not wiped\b",
                ],
                "tr": [
                    r"\bmasalar kirliydi\b",
                    r"\byapış yapış masalar\b",
                    r"\bmasalarda kırıntı vardı\b",
                    r"\bmasalar silinmemişti\b",
                ],
                "ar": [
                    r"\bترابيزات متسخة\b",
                    r"\bالترابيزات كانت لازقة\b",
                    r"\bفتافيت على الترابيزات\b",
                    r"\bما تمسحتش الترابيزة\b",
                ],
                "zh": [
                    r"桌子很脏",
                    r"桌面黏黏的",
                    r"桌上有很多碎屑",
                    r"桌子没有被擦干净",
                ],
            },
        ),
        
        
        "dirty_dishes_left": AspectRule(
            aspect_code="dirty_dishes_left",
            polarity_hint="negative",
            display="Грязная посуда остаётся на столах",
            display_short="грязная посуда на столах",
            long_hint="Гости отмечают, что грязная посуда долго остаётся на столах, что визуально загружает зал и мешает новой посадке.",
            patterns_by_lang={{
                "ru": [
                    r"\bгрязная посуда оставалась на столах\b",
                    r"\bгрязные тарелки долго стояли\b",
                    r"\bпосуда не убиралась\b",
                ],
                "en": [
                    r"\bdirty dishes left on tables\b",
                    r"\bplates were left for a long time\b",
                    r"\bno one cleared the dirty plates\b",
                ],
                "tr": [
                    r"\bkirli tabaklar masalarda kaldı\b",
                    r"\btabaklar uzun süre alınmadı\b",
                    r"\bkirli bulaşıklar toplanmadı\b",
                ],
                "ar": [
                    r"\bأطباق متسخة متروكة على الترابيزات\b",
                    r"\bالأطباق القذرة ما اتشالتش\b",
                    r"\bالصحون فضلت موجودة فترة\b",
                ],
                "zh": [
                    r"脏盘子一直放在桌上",
                    r"很久没人收盘子",
                    r"没有人清走脏餐具",
                ],
            },
        ),
        
        
        "buffet_area_messy": AspectRule(
            aspect_code="buffet_area_messy",
            polarity_hint="negative",
            display="Неаккуратная зона буфета",
            display_short="бардак у буфета",
            long_hint="Гости сообщают о неаккуратности у линий раздачи: проливы, рассыпанные продукты, переполненные урны, приборы/щипцы в беспорядке.",
            patterns_by_lang={{
                "ru": [
                    r"\bгрязь у буфета\b",
                    r"\bу буфета бардак\b",
                    r"\bразлито/рассыпано вокруг буфета\b",
                    r"\bщипцы/приборы в беспорядке\b",
                ],
                "en": [
                    r"\bbuffet area was messy\b",
                    r"\bspills around the buffet\b",
                    r"\bfood scattered around the buffet\b",
                    r"\butensils/tongs were in a mess\b",
                ],
                "tr": [
                    r"\bbüfe alanı dağınıktı\b",
                    r"\bbüfenin etrafı dökülmüştü\b",
                    r"\byiyecekler etrafa saçılmıştı\b",
                    r"\bmaşa/çatallar düzensizdi\b",
                ],
                "ar": [
                    r"\bمنطقة البوفيه مكركبة\b",
                    r"\bأكل واقع حوالين البوفيه\b",
                    r"\bسوايل/صلصات مسكوبة عند البوفيه\b",
                    r"\bالشنّاق/الأدوات ملخبطة\b",
                ],
                "zh": [
                    r"自助区很乱",
                    r"自助台周围有洒得到处的食物",
                    r"自助台有很多溢出的汤汁/酱汁",
                    r"夹子/餐具摆放很凌乱",
                ],
            },
        ),

        "good_value": AspectRule(
            aspect_code="good_value",
            polarity_hint="positive",
            display="Соотношение цена–качество: положительная оценка",
            display_short="хорошее соотношение цена–качество",
            long_hint="Гости отмечают, что размещение оправдывает стоимость: уровень удобства, сервиса и оснащения воспринимается как адекватный или выше ожидаемого за уплаченные деньги.",
            patterns_by_lang={{
                "ru": [
                    r"\bхорошее соотношение цена[- ]качество\b",
                    r"\bотлично за такие деньги\b",
                    r"\bза свои деньги супер\b",
                    r"\bцена полностью оправдана\b",
                    r"\bстоит своих денег\b",
                ],
                "en": [
                    r"\bgood value for money\b",
                    r"\bgreat value\b",
                    r"\bworth the money\b",
                    r"\bexcellent for the price\b",
                    r"\bfor what we paid it was great\b",
                ],
                "tr": [
                    r"\bparanın karşılığını veriyor\b",
                    r"\bfiyatına göre çok iyi\b",
                    r"\bücretine değer\b",
                    r"\bkaldığımız fiyata göre harikaydı\b",
                ],
                "ar": [
                    r"\bقيمة كويسة مقابل السعر\b",
                    r"\bحلو جداً على السعر ده\b",
                    r"\bيسوى الفلوس اللي اندفعت\b",
                    r"\bمقابل السعر ممتاز\b",
                ],
                "zh": [
                    r"性价比很高",
                    r"很值这个价",
                    r"这个价格来说非常好",
                    r"花的这点钱很划算",
                ],
            },
        ),
        
        
        "worth_the_price": AspectRule(
            aspect_code="worth_the_price",
            polarity_hint="positive",
            display="Объект оправдывает заявленную цену",
            display_short="оправданная цена",
            long_hint="Гости прямо указывают, что цена за размещение/ночь оправдана: не завышена, соответствует уровню категории и ожиданиям по локации.",
            patterns_by_lang={{
                "ru": [
                    r"\bцена оправдана\b",
                    r"\bстоит своих денег\b",
                    r"\bцена адекватная\b",
                    r"\bнормальная цена за то что получаем\b",
                    r"\bне завышено за такой уровень\b",
                ],
                "en": [
                    r"\bworth the price\b",
                    r"\bthe price was fair\b",
                    r"\bthe rate was reasonable\b",
                    r"\bpriced fairly for what you get\b",
                    r"\bwe felt the price was justified\b",
                ],
                "tr": [
                    r"\bfiyat haklıydı\b",
                    r"\bfiyata değdi\b",
                    r"\bfiyat makuldü\b",
                    r"\bödediğimiz fiyata göre mantıklıydı\b",
                ],
                "ar": [
                    r"\bالسعر معقول بالنسبة للي خدناه\b",
                    r"\bيسوى المبلغ\b",
                    r"\bالسعر كان منطقي\b",
                    r"\bماحسّيناش إن السعر زايد\b",
                ],
                "zh": [
                    r"这个价位算合理",
                    r"价格对得起体验",
                    r"觉得房价是合理的",
                    r"没有觉得价格虚高",
                ],
            },
        ),
        
        
        "affordable_for_level": AspectRule(
            aspect_code="affordable_for_level",
            polarity_hint="positive",
            display="Доступная цена для данной категории объекта/локации",
            display_short="доступно по цене для уровня",
            long_hint="Гости отмечают, что стоимость проживания ниже ожидаемой для этой категории, района или сезона. Это фиксируется как 'доступно за свой класс'.",
            patterns_by_lang={{
                "ru": [
                    r"\bочень недорого для такого места\b",
                    r"\bдешево для этого района\b",
                    r"\bпо цене доступно для такого уровня\b",
                    r"\bцена ниже чем обычно для центра\b",
                    r"\bзаLOCATION было недорого\b",
                ],
                "en": [
                    r"\baffordable for the area\b",
                    r"\bcheap for this location\b",
                    r"\bgood price for the area\b",
                    r"\bcheaper than most places around\b",
                    r"\bvery affordable for this standard\b",
                ],
                "tr": [
                    r"\bbu bölge için uygun fiyatlı\b",
                    r"\bbu seviyeye göre ucuz\b",
                    r"\bfiyati bölgeye göre iyiydi\b",
                    r"\bçevreye kıyasla daha hesaplıydı\b",
                ],
                "ar": [
                    r"\bسعره حلو بالنسبة للمنطقة\b",
                    r"\bرخيص على مكان زي ده\b",
                    r"\bأرخص من اللي حوالين\b",
                    r"\bسعر كويس على المستوى ده\b",
                ],
                "zh": [
                    r"在这个区域算便宜的",
                    r"在这种档次里价格算实惠",
                    r"相对周边来说价格更划算",
                    r"这个水平算很实惠",
                ],
            },
        ),
        
        
        "overpriced": AspectRule(
            aspect_code="overpriced",
            polarity_hint="negative",
            display="Цена воспринимается завышенной",
            display_short="слишком дорого",
            long_hint="Гости считают, что цена за проживание слишком высокая относительно полученного уровня качества, сервиса, состояния номера или расположения.",
            patterns_by_lang={{
                "ru": [
                    r"\bслишком дорого\b",
                    r"\bцена завышена\b",
                    r"\bне стоит таких денег\b",
                    r"\bдороговато для такого качества\b",
                    r"\bпереплата за бренд/локацию\b",
                ],
                "en": [
                    r"\boverpriced\b",
                    r"\btoo expensive for what you get\b",
                    r"\bnot worth the money\b",
                    r"\bway too expensive\b",
                    r"\bthe rate was too high\b",
                ],
                "tr": [
                    r"\bfazla pahalı\b",
                    r"\bbu kalite için çok pahalı\b",
                    r"\bparanın karşılığını vermiyor\b",
                    r"\bfiyat gereksiz yüksekti\b",
                ],
                "ar": [
                    r"\bغالي على الفاضي\b",
                    r"\bالسعر أوفر من اللي بتاخده\b",
                    r"\bمش مستاهل المبلغ\b",
                    r"\bاحنا دافعين كتير على المستوى ده\b",
                ],
                "zh": [
                    r"价格太贵不值这个水平",
                    r"性价比很低太贵了",
                    r"这个价完全不值",
                    r"房价偏高",
                ],
            },
        ),
        
        
        "not_worth_price": AspectRule(
            aspect_code="not_worth_price",
            polarity_hint="negative",
            display="Не соответствует цене",
            display_short="не оправдывает цену",
            long_hint="Гости фиксируют, что итоговый опыт проживания (сервис, номер, удобства) не соответствует запрошенной цене. Формулируется как разочарование в ценовой справедливости.",
            patterns_by_lang={{
                "ru": [
                    r"\bне стоит своих денег\b",
                    r"\bне оправдывает цену\b",
                    r"\bза эти деньги ожидали лучше\b",
                    r"\bслишком дорого за такой уровень\b",
                    r"\bкачество не соответствует цене\b",
                ],
                "en": [
                    r"\bnot worth the price\b",
                    r"\bnot worth what we paid\b",
                    r"\bfor the price we expected more\b",
                    r"\bdidn't live up to the price\b",
                    r"\bpoor value for money\b",
                ],
                "tr": [
                    r"\bparasını hak etmiyor\b",
                    r"\bverdiğimiz paraya değmedi\b",
                    r"\bbu fiyata daha iyisini beklerdik\b",
                    r"\bfiyatına göre hayal kırıklığı\b",
                ],
                "ar": [
                    r"\bمش بقيمة الفلوس اللي دفعناها\b",
                    r"\bمش قد السعر\b",
                    r"\bعلى السعر ده كنا متوقعين أحسن\b",
                    r"\bالقيمة ضعيفة بالنسبة للسعر\b",
                ],
                "zh": [
                    r"不值这个价",
                    r"这个价格我们本来期待更好",
                    r"性价比不高不值",
                    r"和价位不匹配",
                ],
            },
        ),

        "expected_better_for_price": AspectRule(
            aspect_code="expected_better_for_price",
            polarity_hint="negative",
            display="Ожидания по уровню проживания не оправдались за заявленную цену",
            display_short="ожидали лучше за эти деньги",
            long_hint="Гости указывают, что по уровню сервиса, состоянию номера или оснащению они ожидали более высокий стандарт исходя из стоимости проживания.",
            patterns_by_lang={{
                "ru": [
                    r"\bза такие деньги ожидали лучше\b",
                    r"\bожидали большего за эту цену\b",
                    r"\bкачество не дотягивает до цены\b",
                    r"\bпо цене думали будет лучше\b",
                    r"\bне тот уровень за эти деньги\b",
                ],
                "en": [
                    r"\bexpected better for the price\b",
                    r"\bfor that price we expected more\b",
                    r"\bwe expected a higher standard for what we paid\b",
                    r"\bnot the level we expected based on the price\b",
                ],
                "tr": [
                    r"\bbu fiyata daha iyisini beklerdik\b",
                    r"\bödediğimiz fiyata göre beklentimiz daha yüksekti\b",
                    r"\bbu fiyat seviyesinde daha iyi olmalıydı\b",
                    r"\bbeklenen kalite bu fiyata göre yoktu\b",
                ],
                "ar": [
                    r"\bكنا متوقعين أحسن على السعر ده\b",
                    r"\bبالسعر ده كنا مستنيين مستوى أعلى\b",
                    r"\bالمستوى أقل من اللي كنا متخيلينه للسعر\b",
                    r"\bتوقّعنا حاجة أحسن مقابل اللي دفعناه\b",
                ],
                "zh": [
                    r"这个价位原本期待更好",
                    r"按这个价格我们以为会更高档",
                    r"没有达到这个价位应有的水准",
                    r"花这么多钱结果就这样",
                ],
            },
        ),
        
        
        "photos_misleading": AspectRule(
            aspect_code="photos_misleading",
            polarity_hint="negative",
            display="Несоответствие реального номера фотографиям",
            display_short="фото не соответствуют",
            long_hint="Гости фиксируют, что номер/объект на месте визуально отличается от рекламных фотографий: меньше, старее, менее ухожен, другой тип комнаты.",
            patterns_by_lang={{
                "ru": [
                    r"\bна фото всё выглядело лучше\b",
                    r"\bне так как на фотографиях\b",
                    r"\bфото не соответствуют реальности\b",
                    r"\bномер вживую выглядит хуже чем на фото\b",
                    r"\bномер вообще не тот что на картинках\b",
                ],
                "en": [
                    r"\bnot like in the photos\b",
                    r"\bthe room looked much better in the pictures\b",
                    r"\bphotos were misleading\b",
                    r"\bthe place didn't match the photos online\b",
                    r"\bdefinitely not the room from the pictures\b",
                ],
                "tr": [
                    r"\bfotoğraflardaki gibi değildi\b",
                    r"\bodası fotoğraflarda çok daha iyi görünüyordu\b",
                    r"\bfotoğraflar yanıltıcıydı\b",
                    r"\bgerçekte fotoğraflardaki oda değil\b",
                ],
                "ar": [
                    r"\bالمكان مش زي الصور\b",
                    r"\bالصور أحسن بكتير من الحقيقة\b",
                    r"\bالصور مضللة\b",
                    r"\bالأوضة مش هي اللي في الصور\b",
                ],
                "zh": [
                    r"和照片不一样",
                    r"现场没有网上照片那么好",
                    r"照片有点误导",
                    r"给的房间不是照片里的那种",
                ],
            },
        ),
        
        
        "quality_below_expectation": AspectRule(
            aspect_code="quality_below_expectation",
            polarity_hint="negative",
            display="Общий уровень качества ниже ожиданий гостей",
            display_short="качество ниже ожиданий",
            long_hint="Гости описывают общее впечатление как ниже ожидаемого стандарта: ремонт, оснащение, чистота или комфорт воспринимаются как слабее, чем ожидалось до заезда (по описанию, рейтингу, бренду, категории).",
            patterns_by_lang={{
                "ru": [
                    r"\bуровень ниже ожиданий\b",
                    r"\bкачество ниже наших ожиданий\b",
                    r"\bожидали более высокий стандарт\b",
                    r"\bпо отзывам думали будет лучше\b",
                    r"\bне дотягивает до заявленного уровня\b",
                ],
                "en": [
                    r"\bbelow expectations\b",
                    r"\bthe quality was below what we expected\b",
                    r"\bwe expected a higher standard\b",
                    r"\bfrom the reviews we expected better\b",
                    r"\bnot up to the standard we were expecting\b",
                ],
                "tr": [
                    r"\bbeklentimizin altındaydı\b",
                    r"\bbeklediğimiz standartta değildi\b",
                    r"\bkalite beklentiyi karşılamadı\b",
                    r"\byorumlara göre daha iyi olur diye düşünmüştük\b",
                ],
                "ar": [
                    r"\bالمستوى أقل من اللي كنا متوقعينه\b",
                    r"\bالجودة أضعف من توقعنا\b",
                    r"\bكنا متخيلين مستوى أعلى من كده\b",
                    r"\bعلى حسب الريفيوز توقّعنا أحسن\b",
                ],
                "zh": [
                    r"低于我们的预期",
                    r"整体质量没有达到预期",
                    r"本来以为会更好看评价来的",
                    r"没有达到我们预想的标准",
                ],
            },
        ),
        
        
        "great_location": AspectRule(
            aspect_code="great_location",
            polarity_hint="positive",
            display="Локация как конкурентное преимущество",
            display_short="отличная локация",
            long_hint="Гости подчёркивают, что расположение объекта удобное и стратегически выгодное: рядом с ключевыми точками интереса, инфраструктурой, основными районами посещения.",
            patterns_by_lang={{
                "ru": [
                    r"\bотличное расположение\b",
                    r"\bлокация супер\b",
                    r"\bидеальное место\b",
                    r"\bрасположение просто топ\b",
                    r"\bочень удобная локация\b",
                ],
                "en": [
                    r"\bgreat location\b",
                    r"\bperfect location\b",
                    r"\bexcellent location\b",
                    r"\bthe location was amazing\b",
                    r"\bvery convenient location\b",
                ],
                "tr": [
                    r"\bkonumu harikaydı\b",
                    r"\bmuhteşem konum\b",
                    r"\blokasyon süperdi\b",
                    r"\bçok iyi bir konumda\b",
                ],
                "ar": [
                    r"\bالمكان موقعه ممتاز\b",
                    r"\bاللوكيشن تحفة\b",
                    r"\bالموقع حلو جداً\b",
                    r"\bالمكان في حتة ممتازة\b",
                ],
                "zh": [
                    r"地理位置非常好",
                    r"位置很棒",
                    r"位置特别方便",
                    r"地段很好",
                ],
            },
        ),
        
        
        "central_convenient": AspectRule(
            aspect_code="central_convenient",
            polarity_hint="positive",
            display="Центральное и удобное расположение",
            display_short="центр, удобно добираться",
            long_hint="Гости отмечают, что объект находится в центре или в шаговой доступности от транспорта, основных достопримечательностей, деловых точек и городской инфраструктуры. Воспринимается как удобство передвижения.",
            patterns_by_lang={{
                "ru": [
                    r"\bпрямо в центре\b",
                    r"\bудобно добираться везде\b",
                    r"\bвсё в пешей доступности\b",
                    r"\bрядом с основными достопримечательностями\b",
                    r"\bочень удобно расположен для прогулок по городу\b",
                ],
                "en": [
                    r"\bvery central\b",
                    r"\bcentral location\b",
                    r"\bwalking distance to everything\b",
                    r"\bclose to all the main sights\b",
                    r"\beasy to get everywhere from here\b",
                ],
                "tr": [
                    r"\bçok merkezi\b",
                    r"\bmerkezde yer alıyor\b",
                    r"\bher yere yürüyerek gidilebiliyor\b",
                    r"\bbaşlıca yerlere çok yakın\b",
                    r"\bulaşım açısından çok rahat\b",
                ],
                "ar": [
                    r"\bفي قلب المنطقة\b",
                    r"\bلوكيشن مركزي جداً\b",
                    r"\bتقريباً ماشي على كل حاجة مهمة\b",
                    r"\bقريب من كل الأماكن الرئيسية\b",
                    r"\bسهل توصل لأي حتة من هنا\b",
                ],
                "zh": [
                    r"位置很中心",
                    r"基本去哪儿都很方便",
                    r"走路就能到主要景点",
                    r"地理位置非常靠近市中心",
                    r"从这里去哪里都很方便",
                ],
            },
        ),

        "near_transport": AspectRule(
            aspect_code="near_transport",
            polarity_hint="positive",
            display="Близость к транспорту",
            display_short="рядом транспорт",
            long_hint="Гости отмечают, что рядом находятся удобные транспортные опции: метро, остановки автобуса/трамвая, вокзал, аэропортовый шаттл.",
            patterns_by_lang={{
                "ru": [
                    r"\bрядом метро\b",
                    r"\bв двух шагах от станции\b",
                    r"\bостановка прямо рядом\b",
                    r"\bудобно от/до вокзала\b",
                    r"\bлегко добраться из аэропорта\b",
                ],
                "en": [
                    r"\bclose to public transport\b",
                    r"\bnext to the metro\b",
                    r"\bright by the subway station\b",
                    r"\bbus/tram stop right outside\b",
                    r"\beasy access to the train station\b",
                ],
                "tr": [
                    r"\bmetroya çok yakın\b",
                    r"\bdurak hemen yanında\b",
                    r"\btoplu taşımaya yakın\b",
                    r"\bistasyona ulaşım çok kolaydı\b",
                ],
                "ar": [
                    r"\bالمكان جنب المترو/المحطة\b",
                    r"\bفي محطة أوتوبيس/ترام قريب جداً\b",
                    r"\bسهل جداً تطلع مواصلات من هنا\b",
                    r"\bقريب من محطة القطر/المترو\b",
                ],
                "zh": [
                    r"离地铁站很近",
                    r"门口就是公交/电车站",
                    r"交通很方便",
                    r"去火车站很方便",
                ],
            },
        ),
        
        
        "area_has_food_shops": AspectRule(
            aspect_code="area_has_food_shops",
            polarity_hint="positive",
            display="Инфраструктура рядом (кафе, магазины)",
            display_short="рядом кафе и магазины",
            long_hint="Гости подчёркивают наличие рядом кафе, супермаркетов, ночных магазинов, ресторанов.",
            patterns_by_lang={{
                "ru": [
                    r"\bрядом много кафе\b",
                    r"\bмагазины рядом\b",
                    r"\bсупермаркет за углом\b",
                    r"\bесть где поесть рядом\b",
                    r"\bкуча ресторанов поблизости\b",
                ],
                "en": [
                    r"\blots of restaurants nearby\b",
                    r"\bplenty of places to eat around\b",
                    r"\bshops and supermarkets close by\b",
                    r"\bconvenience store just around the corner\b",
                    r"\bgood food options in the area\b",
                ],
                "tr": [
                    r"\betrafta çok restoran vardı\b",
                    r"\byürüyerek markete gidebiliyorsunuz\b",
                    r"\bmarket hemen köşede\b",
                    r"\byemek için bol seçenek var çevrede\b",
                ],
                "ar": [
                    r"\bحواليه مطاعم وسوبرماركت\b",
                    r"\bفيه أكل قريب ومفتوح\b",
                    r"\bحتلاقي بقالة/ماركت تحت البيت تقريباً\b",
                    r"\bفيه محلات حواليك على طول\b",
                ],
                "zh": [
                    r"附近有很多餐厅",
                    r"楼下就有超市/便利店",
                    r"周边吃饭很方便",
                    r"附近有很多可以买东西的地方",
                ],
            },
        ),
        
        
        "location_inconvenient": AspectRule(
            aspect_code="location_inconvenient",
            polarity_hint="negative",
            display="Неудобное расположение",
            display_short="неудобная локация",
            long_hint="Гости считают локацию неудобной: сложно добираться, нет прямого транспорта, долго идти до центра/объектов интереса.",
            patterns_by_lang={{
                "ru": [
                    r"\bнеудобное расположение\b",
                    r"\bнеудобно добираться\b",
                    r"\bне самое удачное место\b",
                    r"\bничего рядом нет\b",
                    r"\bприходится далеко ходить везде\b",
                ],
                "en": [
                    r"\binconvenient location\b",
                    r"\bnot a convenient area\b",
                    r"\bhard to get anywhere from here\b",
                    r"\bnot easy to get around from the hotel\b",
                    r"\bnot well connected\b",
                ],
                "tr": [
                    r"\bkonum pek kullanışlı değildi\b",
                    r"\bulaşım açısından zor bir konum\b",
                    r"\bhiçbir yere kolay değil\b",
                    r"\bçok uygun bir lokasyon değil\b",
                ],
                "ar": [
                    r"\bالموقع مش مريح خالص\b",
                    r"\bالمكان بعيد ومش سهل توصله\b",
                    r"\bمش سهل تتحرك من المكان ده\b",
                    r"\bالموقع مش عملي\b",
                ],
                "zh": [
                    r"位置不是很方便",
                    r"去哪儿都不太好走",
                    r"交通不太方便",
                    r"这个位置不太实用",
                ],
            },
        ),
        
        
        "far_from_center": AspectRule(
            aspect_code="far_from_center",
            polarity_hint="negative",
            display="Удалённость от центра / ключевых точек",
            display_short="далеко от центра",
            long_hint="Гости отмечают, что объект далеко от центра города, пляжа, набережной или основных достопримечательностей, и это воспринимается как минус локации.",
            patterns_by_lang={{
                "ru": [
                    r"\bдалеко от центра\b",
                    r"\bрасположен слишком далеко\b",
                    r"\bпришлось далеко ездить в центр\b",
                    r"\bне рядом с основными достопримечательностями\b",
                    r"\bрасположение на отшибе\b",
                ],
                "en": [
                    r"\bfar from the city center\b",
                    r"\btoo far from everything\b",
                    r"\bquite far from the main attractions\b",
                    r"\bnot close to the centre\b",
                    r"\bout of the way location\b",
                ],
                "tr": [
                    r"\bmerkeze uzak\b",
                    r"\bher yere uzak bir konum\b",
                    r"\bturistik yerlere uzak\b",
                    r"\bçok dışarıda kalıyor\b",
                ],
                "ar": [
                    r"\bبعيد عن سنتر البلد\b",
                    r"\bالمكان بعيد عن كل حاجة مهمة\b",
                    r"\bمش قريب من المزارات الرئيسية\b",
                    r"\bالموقع على أطراف شوية\b",
                ],
                "zh": [
                    r"离市中心很远",
                    r"离景点都挺远的",
                    r"位置比较偏",
                    r"去哪儿都要走很久/坐很久",
                ],
            },
        ),
        
        
        "nothing_around": AspectRule(
            aspect_code="nothing_around",
            polarity_hint="negative",
            display="Отсутствие инфраструктуры поблизости",
            display_short="ничего рядом нет",
            long_hint="Гости сообщают, что вокруг нет кафе, магазинов, удобств, и район ощущается пустым или нежилым по вечерам.",
            patterns_by_lang={{
                "ru": [
                    r"\bвокруг вообще ничего нет\b",
                    r"\bнет ни магазинов ни кафе рядом\b",
                    r"\bнегде поесть поблизости\b",
                    r"\bрайон пустой вечером\b",
                    r"\bничего в шаговой доступности\b",
                ],
                "en": [
                    r"\bnothing around\b",
                    r"\bnothing nearby\b",
                    r"\bno shops or restaurants nearby\b",
                    r"\bnowhere to eat in the area\b",
                    r"\bthe area is dead at night\b",
                ],
                "tr": [
                    r"\betrafta hiçbir şey yoktu\b",
                    r"\byakında market/restoran yok\b",
                    r"\bçevre akşamları bomboş\b",
                    r"\byürüyerek gidilecek bir şey yok\b",
                ],
                "ar": [
                    r"\bحواليه مافيش أي حاجة\b",
                    r"\bمافيش محلات أو أكل حوالين المكان\b",
                    r"\bالمنطقة فاضية بالليل\b",
                    r"\bمفيش حاجة تمشي لها برجلك\b",
                ],
                "zh": [
                    r"周围什么都没有",
                    r"附近没有店也没有餐厅",
                    r"晚上附近一片冷清",
                    r"附近基本没地方可以走着去",
                ],
            },
        ),

        "area_safe": AspectRule(
            aspect_code="area_safe",
            polarity_hint="positive",
            display="Ощущение безопасности района",
            display_short="район безопасный",
            long_hint="Гости отмечают, что район воспринимается безопасным: спокойно перемещаться пешком, нет агрессивных групп у входа, нет визуальных признаков неблагополучия.",
            patterns_by_lang={{
                "ru": [
                    r"\bрайон безопасный\b",
                    r"\bбезопасно ходить\b",
                    r"\bчувствовали себя безопасно рядом с отелем\b",
                    r"\bспокойный район\b",
                    r"\bничего подозрительного вокруг\b",
                ],
                "en": [
                    r"\bsafe area\b",
                    r"\bthe area felt safe\b",
                    r"\bfelt safe walking around\b",
                    r"\bwe felt safe outside the hotel\b",
                    r"\bquiet and safe neighborhood\b",
                ],
                "tr": [
                    r"\bgüvenli bir bölge\b",
                    r"\bçevre güvenli hissettiriyordu\b",
                    r"\bhotelin etrafı güvenliydi\b",
                    r"\bgece yürümek rahattı\b",
                ],
                "ar": [
                    r"\bالمنطقة أمان\b",
                    r"\bحسّينا إن المنطقة آمنة\b",
                    r"\bنقدر نمشي حوالين المكان بأمان\b",
                    r"\bالحي هادي وآمن\b",
                ],
                "zh": [
                    r"附近区域感觉很安全",
                    r"这个区域很安全",
                    r"晚上走路也觉得安全",
                    r"酒店周围环境很安心",
                ],
            },
        ),
        
        
        "area_quiet_at_night": AspectRule(
            aspect_code="area_quiet_at_night",
            polarity_hint="positive",
            display="Тишина вокруг объекта ночью",
            display_short="тихо ночью снаружи",
            long_hint="Гости подчёркивают, что вокруг объекта было тихо ночью: нет уличного шума, нет баров под окнами, нет конфликтных компаний под входом.",
            patterns_by_lang={{
                "ru": [
                    r"\bночью тихо вокруг\b",
                    r"\bснаружи было тихо по ночам\b",
                    r"\bникакого уличного шума ночью\b",
                    r"\bрайон спокойный по ночам\b",
                ],
                "en": [
                    r"\bquiet at night\b",
                    r"\bthe area was quiet at night\b",
                    r"\bno street noise at night\b",
                    r"\bvery calm outside during the night\b",
                ],
                "tr": [
                    r"\bgece çevre sakindi\b",
                    r"\bgece dışarısı çok sessizdi\b",
                    r"\bsokak gürültüsü yoktu geceleri\b",
                    r"\bgeceleri etraf çok huzurluydu\b",
                ],
                "ar": [
                    r"\bالمنطقة هادية بالليل\b",
                    r"\bمافيش دوشة في الشارع بالليل\b",
                    r"\bبالليل المكان هادي ومش مزعج\b",
                    r"\bمفيش دوشة برا وإحنا نايمين\b",
                ],
                "zh": [
                    r"晚上周围很安静",
                    r"夜里外面基本没有噪音",
                    r"附近夜间很安静适合睡觉",
                    r"没有夜间街头吵闹",
                ],
            },
        ),
        
        
        "entrance_clean": AspectRule(
            aspect_code="entrance_clean",
            polarity_hint="positive",
            display="Состояние входной зоны / подъезда",
            display_short="чистый вход",
            long_hint="Гости отмечают, что входная зона, подъезд или лобби у входа выглядят чистыми и ухоженными: без мусора, без запаха, без следов запущенности.",
            patterns_by_lang={{
                "ru": [
                    r"\bчистый подъезд\b",
                    r"\bчистый вход\b",
                    r"\bу входа аккуратно\b",
                    r"\bперед входом чисто и ухоженно\b",
                    r"\bникакого мусора у двери\b",
                ],
                "en": [
                    r"\bclean entrance\b",
                    r"\bthe entrance area was clean\b",
                    r"\bthe lobby/entrance was well maintained\b",
                    r"\bno trash around the entrance\b",
                    r"\bthe building entrance looked tidy\b",
                ],
                "tr": [
                    r"\bgarantrance temizdi\b",  # small correction below
                    r"\bgiriş çok temizdi\b",
                    r"\bbinanın girişi bakımlıydı\b",
                    r"\bgiriş kısmı tertemizdi\b",
                    r"\bkapının önü temizdi\b",
                ],
                "ar": [
                    r"\bالمدخل نضيف\b",
                    r"\bباب العمارة/الدخول شكله كويس ومهتمين بيه\b",
                    r"\bمافيش زبالة عند المدخل\b",
                    r"\bالمدخل شكله مرتب\b",
                ],
                "zh": [
                    r"入口很干净",
                    r"楼道/入口维护得很整洁",
                    r"门口没有脏乱",
                    r"进门的区域看起来干净体面",
                ],
            },
        ),
        # поправка к tr паттерну:
        # "garantrance temizdi" -> опечатка, модель не может редактировать уже отправленный код в процессе вывода,
        # но смысл: будет использоваться второй/дальше паттерн "giriş çok temizdi" и т.д. как основные ключи.
        
        
        "area_unsafe": AspectRule(
            aspect_code="area_unsafe",
            polarity_hint="negative",
            display="Ощущение небезопасности района",
            display_short="район небезопасный",
            long_hint="Гости сообщают, что местность вокруг объекта воспринимается небезопасной: подозрительные группы людей, агрессивное поведение на улице, следы уличной криминальной активности.",
            patterns_by_lang={{
                "ru": [
                    r"\bрайон небезопасный\b",
                    r"\bнеприятный район\b",
                    r"\bне чувствуешь себя в безопасности снаружи\b",
                    r"\bстрашно выходить вечером\b",
                    r"\bснаружи как\-то стремно\b",
                ],
                "en": [
                    r"\bunsafe area\b",
                    r"\bthe area felt unsafe\b",
                    r"\bdidn't feel safe outside\b",
                    r"\bwe didn't feel safe walking around\b",
                    r"\bscary neighborhood\b",
                ],
                "tr": [
                    r"\bçevre pek güvenli değildi\b",
                    r"\bmahalle güvensiz hissettirdi\b",
                    r"\bgece dışarı çıkmak pek güven vermedi\b",
                    r"\bçevre biraz tehlikeli geldi\b",
                ],
                "ar": [
                    r"\bالمنطقة مش أمان\b",
                    r"\bماحسّيناش إن المنطقة آمنة\b",
                    r"\bمش مرتاحين وإحنا ماشيين برة\b",
                    r"\bالحي شكله مش مطمّن بالليل\b",
                ],
                "zh": [
                    r"附近感觉不太安全",
                    r"周边治安不好",
                    r"我们在外面走路感觉不安全",
                    r"这个区域晚上让人有点怕",
                ],
            },
        ),
        
        
        "uncomfortable_at_night": AspectRule(
            aspect_code="uncomfortable_at_night",
            polarity_hint="negative",
            display="Дискомфорт возвращаться ночью",
            display_short="некомфортно ночью снаружи",
            long_hint="Гости фиксируют, что поздно вечером/ночью им было некомфортно возвращаться к объекту: темно, подозрительные люди у входа, ощущение, что лучше не идти одному.",
            patterns_by_lang={{
                "ru": [
                    r"\bнекомфортно возвращаться ночью\b",
                    r"\bвечером было не по себе\b",
                    r"\bночью идти страшно\b",
                    r"\bне хотелось возвращаться пешком поздно\b",
                    r"\bжутковато возле входа ночью\b",
                ],
                "en": [
                    r"\bfelt uncomfortable walking back at night\b",
                    r"\bit felt a bit sketchy at night\b",
                    r"\bnot comfortable coming back late\b",
                    r"\bthe area was a bit scary at night\b",
                    r"\bdidn't want to walk there alone at night\b",
                ],
                "tr": [
                    r"\bgece dönerken pek rahat hissetmedik\b",
                    r"\bgeceleri ortam biraz tedirgin ediciydi\b",
                    r"\bgece tek başına yürümek istemiyorsun\b",
                    r"\bgece civar biraz ürkütücüydü\b",
                ],
                "ar": [
                    r"\bبالليل مش مريح تمشي وترجع لوحدك\b",
                    r"\bبالليل المكان مش مريح بصراحة\b",
                    r"\bمكان الرجوع بالليل يخوّف شوية\b",
                    r"\bمش حابين نمشي لوحدنا بالليل هناك\b",
                ],
                "zh": [
                    r"晚上回去有点不舒服/不安心",
                    r"夜里附近有点吓人",
                    r"不太敢晚上一个人走回去",
                    r"夜晚在这附近走路会有点怕",
                ],
            },
        ),
    
        "entrance_dirty": AspectRule(
            aspect_code="entrance_dirty",
            polarity_hint="negative",
            display="Неухоженная входная зона",
            display_short="грязный вход",
            long_hint="Гости отмечают, что у входа/в подъезде грязно: мусор, неприятный запах, следы запущенности.",
            patterns_by_lang={{
                "ru": [
                    r"\bгрязный подъезд\b",
                    r"\bгрязный вход\b",
                    r"\bу входа грязно\b",
                    r"\bвоняло у входа\b",
                    r"\bподъезд неухоженный\b",
                ],
                "en": [
                    r"\bdirty entrance\b",
                    r"\bthe entrance was dirty\b",
                    r"\bthe stairwell/entry smelled bad\b",
                    r"\bthe building entrance looked run down\b",
                    r"\bthe hallway by the entrance was filthy\b",
                ],
                "tr": [
                    r"\bgiriş çok kirliydi\b",
                    r"\bbinanın girişi bakımsızdı\b",
                    r"\bmerdiven boşluğu/podyez çok pis kokuyordu\b",
                    r"\bkapının önü pek temiz değildi\b",
                ],
                "ar": [
                    r"\bالمدخل وسخ\b",
                    r"\bباب العمارة شكله مهمل\b",
                    r"\bريحة مش حلوة عند المدخل\b",
                    r"\bالسلم/المدخل مش نضيف\b",
                ],
                "zh": [
                    r"入口很脏",
                    r"楼道/门口有股味道",
                    r"楼栋入口看起来很破旧脏乱",
                    r"门口区域不干净",
                ],
            },
        ),
        
        
        "people_loitering": AspectRule(
            aspect_code="people_loitering",
            polarity_hint="negative",
            display="Подозрительные компании у входа",
            display_short="подозрительные люди у входа",
            long_hint="Гости сообщают о группах людей, постоянно находящихся у входа/в подъезде (курят, выпивают, шумят), что вызывает дискомфорт, ощущение небезопасности и снижает имидж района.",
            patterns_by_lang={{
                "ru": [
                    r"\bподозрительные люди у входа\b",
                    r"\bпостоянно тусуются у двери\b",
                    r"\bалкаши/компании у подъезда\b",
                    r"\bнеприятный контингент перед входом\b",
                    r"\bу входа стоят сомнительные типы\b",
                ],
                "en": [
                    r"\bpeople hanging around outside\b",
                    r"\bsketchy people by the entrance\b",
                    r"\bpeople loitering outside the building\b",
                    r"\bthere were drunks/drug users outside\b",
                    r"\bfelt uncomfortable because of people at the door\b",
                ],
                "tr": [
                    r"\bkapının önünde tipler takılıyordu\b",
                    r"\bgirişte sürekli insanlar oyalanıyordu\b",
                    r"\bşüpheli tipler bina önündeydi\b",
                    r"\bgece girişte rahatsız edici kişiler vardı\b",
                ],
                "ar": [
                    r"\bفي ناس واقفين دايمًا تحت العمارة\b",
                    r"\bفي ناس مش مريحة عند الباب\b",
                    r"\bفي ناس بتقعد قدام المدخل وبيدخنوا/يشربوا\b",
                    r"\bالمدخل عليه ناس شكلهم مش مريح\b",
                ],
                "zh": [
                    r"楼门口经常有人徘徊",
                    r"门口有一些让人不太安心的人聚着",
                    r"楼下老有人喝酒抽烟不走",
                    r"门口的人让人感觉不安全",
                ],
            },
        ),
        
        
        "easy_to_find": AspectRule(
            aspect_code="easy_to_find",
            polarity_hint="positive",
            display="Объект легко найти",
            display_short="легко найти адрес",
            long_hint="Гости отмечают, что место легко найти с первого раза: понятный адрес, чёткая навигация, заметная вывеска/вход.",
            patterns_by_lang={{
                "ru": [
                    r"\bлегко найти\b",
                    r"\bадрес нашли без проблем\b",
                    r"\bсразу нашли здание\b",
                    r"\bничего не пришлось долго искать\b",
                    r"\bвывеска заметная\b",
                ],
                "en": [
                    r"\beasy to find\b",
                    r"\bthe property was easy to find\b",
                    r"\bwe found it straight away\b",
                    r"\bclear to locate the building\b",
                    r"\bthe sign was easy to spot\b",
                ],
                "tr": [
                    r"\bbulması çok kolaydı\b",
                    r"\badresi hemen bulduk\b",
                    r"\bbinayı ilk seferde bulduk\b",
                    r"\btabela kolay fark ediliyordu\b",
                ],
                "ar": [
                    r"\bسهل جداً تلاقي المكان\b",
                    r"\bلقينا العنوان بسهولة\b",
                    r"\bالمكان باين وواضح\b",
                    r"\bمفيش لفة طويلة عشان نلاقيه\b",
                ],
                "zh": [
                    r"很容易找到这个地方",
                    r"地址很好找一下就到了",
                    r"楼/门牌很好认",
                    r"基本不用找很久就能找到",
                ],
            },
        ),
        
        
        "clear_instructions": AspectRule(
            aspect_code="clear_instructions",
            polarity_hint="positive",
            display="Понятные инструкции по заезду и доступу",
            display_short="чёткие инструкции по заезду",
            long_hint="Гости отмечают, что им заранее дали ясные пошаговые инструкции: как попасть в здание, где забрать ключ/код, куда именно идти.",
            patterns_by_lang={{
                "ru": [
                    r"\bочень понятные инструкции\b",
                    r"\bподробные инструкции по заселению\b",
                    r"\bнам заранее всё расписали\b",
                    r"\bвсё объяснили куда идти и что нажать\b",
                    r"\bинструкции по доступу были чёткие\b",
                ],
                "en": [
                    r"\bclear check-in instructions\b",
                    r"\bvery clear instructions on how to get in\b",
                    r"\bthey sent detailed self check-in info\b",
                    r"\bwe knew exactly what to do to enter\b",
                    r"\bclear directions on where to go\b",
                ],
                "tr": [
                    r"\bcheck\-in talimatları çok açıktı\b",
                    r"\bnasıl gireceğimizi çok iyi anlattılar\b",
                    r"\bdetaylı yönergeler gönderdiler\b",
                    r"\bnereden anahtar/kodu alacağımız nett i\b",
                ],
                "ar": [
                    r"\bإدونا تعليمات واضحة جداً للدخول\b",
                    r"\bالشيك إن كان متشرح خطوة بخطوة\b",
                    r"\bقالولنا بالظبط نمشي فين ونفتح إزاي\b",
                    r"\bالخطوات كانت واضحة ومش معقدة\b",
                ],
                "zh": [
                    r"入住指引很清楚",
                    r"提前给了详细的自助入住步骤",
                    r"告诉我们具体怎么进楼拿钥匙/输密码",
                    r"基本一步步写清楚不用问人",
                ],
            },
        ),
        
        
        "luggage_access_ok": AspectRule(
            aspect_code="luggage_access_ok",
            polarity_hint="positive",
            display="Удобство доступа с багажом",
            display_short="удобно с багажом",
            long_hint="Гости указывают, что дойти до номера с чемоданами было несложно: удобный вход, лифт на нужный этаж, без длинных лестниц/ступеней.",
            patterns_by_lang={{
                "ru": [
                    r"\bс чемоданами было удобно\b",
                    r"\bлегко добраться с багажом\b",
                    r"\bне пришлось таскать чемоданы по лестнице\b",
                    r"\bс багажом проблем не было\b",
                    r"\bудобный доступ с сумками\b",
                ],
                "en": [
                    r"\beasy access with luggage\b",
                    r"\bno problem with our suitcases\b",
                    r"\bwe could get the luggage in easily\b",
                    r"\bno stairs with heavy bags\b",
                    r"\bconvenient with luggage\b",
                ],
                "tr": [
                    r"\bvalizlerle girmek çok kolaydı\b",
                    r"\bçantalarla çıkmak sorun olmadı\b",
                    r"\bmerdiven taşımak zorunda kalmadık\b",
                    r"\bvalizle ulaşmak rahattı\b",
                ],
                "ar": [
                    r"\bسهل تدخل بالشنط\b",
                    r"\bما اضطريناش نطلع شنط على سلالم كتير\b",
                    r"\bالموضوع كان سهل مع الشنط التقيلة\b",
                    r"\bقدرنا ناخد الشنط للغرفة من غير معاناة\b",
                ],
                "zh": [
                    r"带行李进去很方便",
                    r"拿着箱子进出很轻松",
                    r"不用扛箱子爬楼梯",
                    r"拖着行李到房间没什么困难",
                ],
            },
        ),

        "hard_to_find_entrance": AspectRule(
            aspect_code="hard_to_find_entrance",
            polarity_hint="negative",
            display="Сложно найти вход",
            display_short="трудно найти вход",
            long_hint="Гости сообщают, что сам вход в объект найти сложно: неприметная дверь, несколько подъездов, непонятно куда именно заходить.",
            patterns_by_lang={{
                "ru": [
                    r"\bсложно найти вход\b",
                    r"\bдолго искали подъезд\b",
                    r"\bнепонятно где вход\b",
                    r"\bтрудно понять куда заходить\b",
                    r"\bдверь не обозначена\b",
                ],
                "en": [
                    r"\bhard to find the entrance\b",
                    r"\bwe couldn't find the door\b",
                    r"\bit was not obvious where to go in\b",
                    r"\bthe entrance was difficult to locate\b",
                    r"\bnot clear which door to use\b",
                ],
                "tr": [
                    r"\bgirişi bulmak zordu\b",
                    r"\bhangi kapıdan girileceği belli değildi\b",
                    r"\bkapıyı bulmakta zorlandık\b",
                    r"\bgiriş net değildi\b",
                ],
                "ar": [
                    r"\bمدخل المكان مش واضح\b",
                    r"\bقعدنا ندور على الباب\b",
                    r"\bمش باين منين تدخل\b",
                    r"\bمفيش مدخل باين بوضوح\b",
                ],
                "zh": [
                    r"很难找到入口",
                    r"找半天才找到门",
                    r"不太清楚从哪扇门进去",
                    r"入口不明显",
                ],
            },
        ),
        
        
        "confusing_access": AspectRule(
            aspect_code="confusing_access",
            polarity_hint="negative",
            display="Запутанный доступ в здание/номер",
            display_short="неочевидный доступ",
            long_hint="Гости фиксируют, что схема доступа сложная: коды, несколько дверей подряд, во двор через арку и т.д. Нужны лишние шаги, которые неочевидны без детальной подсказки.",
            patterns_by_lang={{
                "ru": [
                    r"\bнепонятно как попасть внутрь\b",
                    r"\bзапутанный доступ\b",
                    r"\bслишком сложная система кодов\b",
                    r"\bчтобы зайти надо пройти через двор/арку и ещё одну дверь\b",
                    r"\bпришлось долго разбираться как войти\b",
                ],
                "en": [
                    r"\bconfusing access\b",
                    r"\bgetting into the building was confusing\b",
                    r"\btoo many doors and codes\b",
                    r"\bnot straightforward to get in\b",
                    r"\btook us a while to figure out how to enter\b",
                ],
                "tr": [
                    r"\bbinaya giriş çok karışıktı\b",
                    r"\bşifreler/kapılar çok karmaşıktı\b",
                    r"\bgirmek kolay değildi\b",
                    r"\bhangi kapıdan girileceği anlaşılmıyordu\b",
                ],
                "ar": [
                    r"\bالدخول معقد شوية\b",
                    r"\bمحتاج أكتر من كود وبوابة عشان تدخل\b",
                    r"\bالدخول للمبنى مش واضح\b",
                    r"\bاخد مننا شوية وقت نفهم بندخل منين\b",
                ],
                "zh": [
                    r"进楼过程很复杂",
                    r"要输好几个门码才进得去",
                    r"进门步骤太绕",
                    r"不是很好理解怎么进去",
                ],
            },
        ),
        
        
        "no_signage": AspectRule(
            aspect_code="no_signage",
            polarity_hint="negative",
            display="Отсутствие навигации и указателей",
            display_short="нет указателей",
            long_hint="Гости указывают, что нет нормальной навигации: отсутствует вывеска объекта, номер апартаментов не подписан, невозможно понять куда идти внутри здания или двора.",
            patterns_by_lang={{
                "ru": [
                    r"\bнет вывески\b",
                    r"\bникаких указателей\b",
                    r"\bничего не подписано\b",
                    r"\bневозможно понять где именно номер\b",
                    r"\bадрес есть но табличек нет\b",
                ],
                "en": [
                    r"\bno signage\b",
                    r"\bno signs anywhere\b",
                    r"\bthe place wasn't marked\b",
                    r"\bnot signposted at all\b",
                    r"\bnothing was labeled inside the building\b",
                ],
                "tr": [
                    r"\bhiç tabela yoktu\b",
                    r"\byer işaretlenmemişti\b",
                    r"\bhangi daire olduğu yazmıyordu\b",
                    r"\bhiçbir yönlendirme yoktu\b",
                ],
                "ar": [
                    r"\bمافيش أي يفطة/لافتة\b",
                    r"\bمش مكتوب اسم المكان على الباب\b",
                    r"\bمفيش علامات توجهك جوا العمارة\b",
                    r"\bمش عارفين الشقة/الأوضة فين بالظبط\b",
                ],
                "zh": [
                    r"门口没有任何标识",
                    r"楼里没有指示牌",
                    r"没写是哪间房",
                    r"完全没有指引标识",
                ],
            },
        ),
        
        
        "luggage_access_hard": AspectRule(
            aspect_code="luggage_access_hard",
            polarity_hint="negative",
            display="Сложный доступ с багажом",
            display_short="тяжело с багажом",
            long_hint="Гости сообщают, что с чемоданами было тяжело: крутые лестницы, отсутствие лифта, узкие проходы, неудобный подъем до этажа.",
            patterns_by_lang={{
                "ru": [
                    r"\bс багажом было тяжело\b",
                    r"\bпришлось тащить чемоданы по лестнице\b",
                    r"\bнет лифта и чемоданы на руках\b",
                    r"\bузкие лестницы с чемоданом\b",
                    r"\bочень неудобно с тяжёлыми сумками\b",
                ],
                "en": [
                    r"\bhard with luggage\b",
                    r"\bnot convenient with suitcases\b",
                    r"\bwe had to carry our bags up the stairs\b",
                    r"\bno lift so we carried heavy luggage\b",
                    r"\bstairs were a problem with luggage\b",
                ],
                "tr": [
                    r"\bvalizlerle çok zor oldu\b",
                    r"\bmerdivenleri çantayla çıkmak çok yorucuydu\b",
                    r"\basansör yoktu ve valiz taşımak zorundaydık\b",
                    r"\bvalizle ulaşım rahat değildi\b",
                ],
                "ar": [
                    r"\bالموضوع كان مرهق مع الشنط\b",
                    r"\bطلعنا الشنط على السلم بنفسنا\b",
                    r"\bمفيش أسانسير وإحنا شايلين شنط تقيلة\b",
                    r"\bالسلالم مع الشنط كانت متعبة جداً\b",
                ],
                "zh": [
                    r"拿行李很吃力",
                    r"没有电梯只能提着箱子上楼",
                    r"楼梯很不方便拖行李",
                    r"带着大箱子上去很麻烦",
                ],
            },
        ),
        
        
        "cozy_atmosphere": AspectRule(
            aspect_code="cozy_atmosphere",
            polarity_hint="positive",
            display="Уютная атмосфера объекта",
            display_short="уютная атмосфера",
            long_hint="Гости описывают общую атмосферу как тёплую и домашнюю: приятно находиться, не 'безликий отель', есть ощущение уюта и комфорта в интерьерах и общих зонах.",
            patterns_by_lang={{
                "ru": [
                    r"\bуютная атмосфера\b",
                    r"\bочень уютно\b",
                    r"\bдомашняя обстановка\b",
                    r"\bчувствуется уют\b",
                    r"\bне как бездушный отель\b",
                ],
                "en": [
                    r"\bcozy atmosphere\b",
                    r"\bvery cozy\b",
                    r"\bhomey feeling\b",
                    r"\bfelt very homely\b",
                    r"\bnot a cold/impersonal hotel vibe\b",
                ],
                "tr": [
                    r"\bsamimi/ev gibi bir ortam vardı\b",
                    r"\bortam çok sıcaktı ve rahattı\b",
                    r"\bçok sıcak ve huzurlu bir atmosfer\b",
                    r"\bsoğuk bir otel havası yoktu\b",
                ],
                "ar": [
                    r"\bجو دافي ومريح\b",
                    r"\bالإحساس كأنه بيت مش أوتيل رسمي\b",
                    r"\bالمكان مريح ومش بارد في الإحساس\b",
                    r"\bفيه جو مريح وبيتوتي\b",
                ],
                "zh": [
                    r"整体氛围很温馨",
                    r"感觉很有家的感觉",
                    r"氛围很舒服不冰冷",
                    r"不是那种很冷的标准酒店感觉",
                ],
            },
        ),

        "nice_design": AspectRule(
            aspect_code="nice_design",
            polarity_hint="positive",
            display="Приятный дизайн и интерьер",
            display_short="красивый интерьер",
            long_hint="Гости отмечают, что интерьер выглядит современно, эстетично и продуманно: привлекательный декор, аккуратный ремонт, визуально приятные общие зоны и номер.",
            patterns_by_lang={{
                "ru": [
                    r"\bкрасивый интерьер\b",
                    r"\bочень стильный дизайн\b",
                    r"\bсовременный интерьер\b",
                    r"\bуютный и красивый декор\b",
                    r"\bприятно оформлено\b",
                ],
                "en": [
                    r"\bnice design\b",
                    r"\bstylish interior\b",
                    r"\bmodern design\b",
                    r"\bthe room was nicely decorated\b",
                    r"\bthe place looks well designed\b",
                ],
                "tr": [
                    r"\bdekorasyon çok güzeldi\b",
                    r"\boda çok şık tasarlanmıştı\b",
                    r"\bmodern tasarım\b",
                    r"\bortamın dekoru çok hoştu\b",
                ],
                "ar": [
                    r"\bالديكور شكله حلو\b",
                    r"\bالمكان متصمم بشكل شيك\b",
                    r"\bشكل الأوضة حلو ومرتب\b",
                    r"\bالديزاين حديث ومرتب\b",
                ],
                "zh": [
                    r"房间的设计很好看",
                    r"装修很有设计感",
                    r"整体风格很漂亮很现代",
                    r"摆设和装潢都很讲究",
                ],
            },
        ),
        
        
        "good_vibe": AspectRule(
            aspect_code="good_vibe",
            polarity_hint="positive",
            display="Приятная атмосфера в объекте",
            display_short="хорошая атмосфера",
            long_hint="Гости описывают общее ощущение как комфортное и приятное: спокойная энергетика места, приятная общая ‘аура’, ощущение, что находиться внутри приятно и расслабленно.",
            patterns_by_lang={{
                "ru": [
                    r"\bочень приятная атмосфера\b",
                    r"\bхорошая атмосфера\b",
                    r"\bочень уютная энергетика\b",
                    r"\bкомфортная обстановка\b",
                    r"\bприятное ощущение в пространстве\b",
                ],
                "en": [
                    r"\bgood vibe\b",
                    r"\bnice vibe\b",
                    r"\bgreat atmosphere\b",
                    r"\bthe place has a really nice vibe\b",
                    r"\bvery pleasant atmosphere\b",
                ],
                "tr": [
                    r"\bortamın enerjisi çok iyiydi\b",
                    r"\bçok güzel bir atmosfer vardı\b",
                    r"\brahat ve pozitif bir ortam\b",
                    r"\bmekanın havası çok hoştu\b",
                ],
                "ar": [
                    r"\bالمكان جوّه حلو\b",
                    r"\bفيه فيب لطيف\b",
                    r"\bجو مريح ومبسوطين فيه\b",
                    r"\bالإحساس العام في المكان مريح\b",
                ],
                "zh": [
                    r"整体氛围很好",
                    r"感觉气氛很舒服",
                    r"地方的感觉很好很放松",
                    r"整体 vibe 很好",
                ],
            },
        ),
        
        
        "not_cozy": AspectRule(
            aspect_code="not_cozy",
            polarity_hint="negative",
            display="Отсутствие уюта",
            display_short="неуютно",
            long_hint="Гости сообщают, что пространство воспринимается холодным и неуютным: 'не по-домашнему', без ощущения комфорта, слишком формально или безлико.",
            patterns_by_lang={{
                "ru": [
                    r"\bнеуютно\b",
                    r"\bнет уюта\b",
                    r"\bатмосфера холодная\b",
                    r"\bкак в офисе, а не как дома\b",
                    r"\bслишком безлично\b",
                ],
                "en": [
                    r"\bnot cozy\b",
                    r"\bdidn't feel cozy\b",
                    r"\bthe place felt cold\b",
                    r"\bnot a very welcoming atmosphere\b",
                    r"\bfelt a bit impersonal\b",
                ],
                "tr": [
                    r"\bçok sıcak/rahat bir ortam değildi\b",
                    r"\bortam pek samimi gelmedi\b",
                    r"\bodada pek bir sıcaklık yoktu\b",
                    r"\bçok soğuk ve ruhsuz hissettirdi\b",
                ],
                "ar": [
                    r"\bالمكان مش دافي الإحساس\b",
                    r"\bمش مريح نفسياً\b",
                    r"\bالجو بارد ومش بيحسسك براحة\b",
                    r"\bحاسينه شوية رسمي زيادة\b",
                ],
                "zh": [
                    r"感觉不太温馨",
                    r"整体氛围有点冷淡",
                    r"房间没有家的感觉",
                    r"环境有点生硬不太放松",
                ],
            },
        ),
        
        
        "gloomy_feel": AspectRule(
            aspect_code="gloomy_feel",
            polarity_hint="negative",
            display="Унылая/мрачная визуальная атмосфера",
            display_short="мрачная атмосфера",
            long_hint="Гости описывают пространство как мрачное или угнетающее: тусклый свет, тёмная отделка, давящее впечатление.",
            patterns_by_lang={{
                "ru": [
                    r"\bмрачная атмосфера\b",
                    r"\bкакой-то унылый вид\b",
                    r"\bтёмно и давяще\b",
                    r"\bмрачновато в номере\b",
                    r"\bсерая угнетающая обстановка\b",
                ],
                "en": [
                    r"\bgloomy atmosphere\b",
                    r"\bthe room felt gloomy\b",
                    r"\bvery dark and depressing\b",
                    r"\bthe place felt a bit depressing\b",
                    r"\bthe lighting was very dark and dull\b",
                ],
                "tr": [
                    r"\boda biraz kasvetliydi\b",
                    r"\bhava karanlık ve iç karartıcıydı\b",
                    r"\bortam moral bozucuydu\b",
                    r"\bışık çok yetersiz ve kasvetliydi\b",
                ],
                "ar": [
                    r"\bالمكان كئيب شوية\b",
                    r"\bالإضاءة ضعيفة ومكبوسة الجو\b",
                    r"\bالشكل العام محبط شوية\b",
                    r"\bالإحساس في الأوضة غامق وكئيب\b",
                ],
                "zh": [
                    r"房间感觉有点阴沉压抑",
                    r"整体光线很暗很压抑",
                    r"氛围有点让人郁闷",
                    r"环境显得有点沉闷灰暗",
                ],
            },
        ),
        
        
        "dated_look": AspectRule(
            aspect_code="dated_look",
            polarity_hint="negative",
            display="Старомодный/устаревший вид интерьера",
            display_short="устаревший вид",
            long_hint="Гости отмечают, что интерьер выглядит морально устаревшим: старые материалы отделки, мебель без обновления, визуально не соответствует современным ожиданиям продукта данной категории.",
            patterns_by_lang={{
                "ru": [
                    r"\bустаревший интерьер\b",
                    r"\bвсё старомодно выглядит\b",
                    r"\bремонт старый\b",
                    r"\bсоветский стиль\b",
                    r"\bвыглядит как из прошлого\b",
                ],
                "en": [
                    r"\bdated look\b",
                    r"\bthe room felt dated\b",
                    r"\bvery old fashioned decor\b",
                    r"\blooks old and outdated\b",
                    r"\bthe place needs updating\b",
                ],
                "tr": [
                    r"\beski bir görünüme sahipti\b",
                    r"\bdekorasyon çok demodeydi\b",
                    r"\bodanın tarzı çok eskiydi\b",
                    r"\byenilenmeye ihtiyacı var\b",
                ],
                "ar": [
                    r"\bالديكور قديم قوي\b",
                    r"\bالستايـل قديم ومش معمول أبديت\b",
                    r"\bالمكان حاسينه قديم ومتعبان شوية بالشكل\b",
                    r"\bحاسينه محتاج تجديد شكلًا\b",
                ],
                "zh": [
                    r"装修很老旧",
                    r"房间风格显得很过时",
                    r"整体感觉很老派需要翻新",
                    r"看起来像很久没更新过",
                ],
            },
        ),

        "soulless_feel": AspectRule(
            aspect_code="soulless_feel",
            polarity_hint="negative",
            display="Безликая / 'неживая' атмосфера",
            display_short="без атмосферы",
            long_hint="Гости описывают пространство как безликое, 'без души': ощущение, что интерьер функциональный, но холодный, нереспектабельный, не вызывает эмоционального комфорта и не создаёт ощущения места с характером.",
            patterns_by_lang={{
                "ru": [
                    r"\bбез души\b",
                    r"\bсовсем нет атмосферы\b",
                    r"\bочень безлико\b",
                    r"\bощущение как офис, а не отель\b",
                    r"\bникакого характера у места\b",
                ],
                "en": [
                    r"\bsoulless\b",
                    r"\bvery soulless vibe\b",
                    r"\bno character at all\b",
                    r"\bfelt very impersonal\b",
                    r"\bsterile and cold atmosphere\b",
                ],
                "tr": [
                    r"\bruhtsuz bir ortam vardı\b",
                    r"\bhiç bir karakteri yoktu mekanın\b",
                    r"\bçok soğuk ve kişiliksizdi\b",
                    r"\bpek bir atmosfer yoktu\b",
                ],
                "ar": [
                    r"\bالمكان مافيش روح\b",
                    r"\bالإحساس بارد ومفيش شخصية للمكان\b",
                    r"\bمفيش جو خاص للمكان\b",
                    r"\bحاسينه رسمي وبس من غير روح\b",
                ],
                "zh": [
                    r"感觉很没有灵魂",
                    r"地方很冷淡没个性",
                    r"氛围很生硬很公式化",
                    r"完全没有什么独特感觉",
                ],
            },
        ),
        
        
        "fresh_smell_common": AspectRule(
            aspect_code="fresh_smell_common",
            polarity_hint="positive",
            display="Свежий запах в общих зонах",
            display_short="свежий запах в коридорах",
            long_hint="Гости отмечают, что в коридорах, лобби и у входа приятно пахнет: нет затхлости, нет запаха мусора или канализации, ощущение свежести и ухоженности общих зон.",
            patterns_by_lang={{
                "ru": [
                    r"\bсвежий запах в коридоре\b",
                    r"\bприятно пахнет в общих зонах\b",
                    r"\bв лобби приятно пахнет\b",
                    r"\bникакой вони на входе\b",
                    r"\bчистый свежий запах\b",
                ],
                "en": [
                    r"\bfresh smell in the hallway\b",
                    r"\bthe corridors smelled fresh\b",
                    r"\bthe lobby smelled nice\b",
                    r"\bno bad odors in common areas\b",
                    r"\bsmelled clean everywhere\b",
                ],
                "tr": [
                    r"\bkoridorlar ferah kokuyordu\b",
                    r"\blobide güzel bir koku vardı\b",
                    r"\bortak alanlarda kötü koku yoktu\b",
                    r"\btemiz/fresh bir koku vardı\b",
                ],
                "ar": [
                    r"\bريحة المكان في الطرقات حلوة\b",
                    r"\bاللوبي ريحته حلوة ونضيفة\b",
                    r"\bمفيش ريحة وحشة في الأُسنسير أو الممر\b",
                    r"\bالريحة عامةً نظيفة ومش مخنوقة\b",
                ],
                "zh": [
                    r"走廊有清新的味道",
                    r"大堂闻起来很干净很舒服",
                    r"公共区域没有异味反而很清新",
                    r"整体有种干净的香味",
                ],
            },
        ),
        
        
        "no_bad_smell": AspectRule(
            aspect_code="no_bad_smell",
            polarity_hint="positive",
            display="Отсутствие неприятных запахов",
            display_short="нет неприятных запахов",
            long_hint="Гости подчёркивают, что по всему объекту нет посторонних запахов (сигареты, сырость, канализация).",
            patterns_by_lang={{
                "ru": [
                    r"\bнет неприятного запаха\b",
                    r"\bне пахнет сыростью\b",
                    r"\bне пахнет канализацией\b",
                    r"\bзапаха сигарет не было\b",
                    r"\bв номере не воняло\b",
                ],
                "en": [
                    r"\bno bad smell\b",
                    r"\bno strange smell\b",
                    r"\bno cigarette smell\b",
                    r"\bno damp smell\b",
                    r"\bthe room didn't smell at all\b",
                ],
                "tr": [
                    r"\bkötü bir koku yoktu\b",
                    r"\bsigara kokusu yoktu\b",
                    r"\bnem/ küf kokusu yoktu\b",
                    r"\bkanalizasyon kokusu gelmiyordu\b",
                ],
                "ar": [
                    r"\bمفيش ريحة وحشة\b",
                    r"\bمش ريحة سجاير ولا ريحة كمكمة\b",
                    r"\bمفيش ريحة مجاري\b",
                    r"\bالأوضة ماكانتش ريحتها وحشة\b",
                ],
                "zh": [
                    r"没有异味",
                    r"没有烟味或者霉味",
                    r"房间里没有怪味",
                    r"没有下水道味道",
                ],
            },
        ),
        
        
        "bad_smell_common": AspectRule(
            aspect_code="bad_smell_common",
            polarity_hint="negative",
            display="Неприятный запах в общих зонах",
            display_short="запах в коридорах",
            long_hint="Гости сообщают о неприятных запахах в коридорах, лифтовом холле, на лестнице или у входа: запах сигарет, канализации, мусора, сырости.",
            patterns_by_lang={{
                "ru": [
                    r"\bнеприятный запах в коридоре\b",
                    r"\bвоняло в подъезде\b",
                    r"\bв холле пахло сыростью\b",
                    r"\bзапах канализации в общем коридоре\b",
                    r"\bпахло мусором у лифта\b",
                ],
                "en": [
                    r"\bbad smell in the hallway\b",
                    r"\bthe corridor smelled bad\b",
                    r"\bthere was a sewage smell near the entrance\b",
                    r"\bit smelled like garbage in the lobby\b",
                    r"\bstrong damp smell in the corridor\b",
                ],
                "tr": [
                    r"\bkoridorda kötü bir koku vardı\b",
                    r"\bmerdiven boşluğu kötü kokuyordu\b",
                    r"\bgirişte lağım kokusu vardı\b",
                    r"\blobide çöp gibi kokuyordu\b",
                    r"\bnem/küf kokusu vardı\b",
                ],
                "ar": [
                    r"\bفي ريحة وحشة في الممر\b",
                    r"\bسلم/مدخل ريحته مش حلوة\b",
                    r"\bريحة مجاري عند المدخل\b",
                    r"\bريحة زبالة عند الأسانسير\b",
                    r"\bريحة كمكمة قوية في الكوريدور\b",
                ],
                "zh": [
                    r"走廊有股异味",
                    r"电梯口有垃圾味/下水道味",
                    r"楼道里有霉味很重",
                    r"入口处有一股臭味",
                ],
            },
        ),
        
        
        "cigarette_smell": AspectRule(
            aspect_code="cigarette_smell",
            polarity_hint="negative",
            display="Запах табака / сигарет в общих зонах",
            display_short="запах сигарет",
            long_hint="Гости фиксируют запах сигаретного дыма (или в номере, или в коридоре / подъезде), часто несмотря на заявленный запрет на курение.",
            patterns_by_lang={{
                "ru": [
                    r"\bзапах сигарет\b",
                    r"\bвоняло табаком\b",
                    r"\bв коридоре пахло дымом\b",
                    r"\bв номере пахло как после курения\b",
                    r"\bсильный запах курева\b",
                ],
                "en": [
                    r"\bsmelled of cigarettes\b",
                    r"\bstrong cigarette smell\b",
                    r"\bthe room smelled like smoke\b",
                    r"\bsmelled like someone had been smoking\b",
                    r"\bcigarette smoke in the hallway\b",
                ],
                "tr": [
                    r"\bsigara kokusu vardı\b",
                    r"\bodada sigara içilmiş gibi kokuyordu\b",
                    r"\bkoridorda sigara dumanı kokusu vardı\b",
                    r"\bçok yoğun sigara kokuyordu\b",
                ],
                "ar": [
                    r"\bريحة سجاير قوية\b",
                    r"\bالأوضة كانت ريحتها دخان\b",
                    r"\bالممر ريحته سجاير\b",
                    r"\bكأن في حد كان بيشرب سجاير جوه\b",
                ],
                "zh": [
                    r"有很重的烟味",
                    r"房间里有烟味像被抽过烟",
                    r"走廊里都是烟味",
                    r"闻起来像有人刚抽完烟",
                ],
            },
        ),
    
        "sewage_smell": AspectRule(
            aspect_code="sewage_smell",
            polarity_hint="negative",
            display="Запах канализации в общих зонах",
            display_short="запах канализации",
            long_hint="Гости фиксируют выраженный запах канализации или сточных вод в коридорах, у входа, в лифтовом холле или санузле.",
            patterns_by_lang={{
                "ru": [
                    r"\bзапах канализац(ии|ией)\b",
                    r"\bвоняло канализацией\b",
                    r"\bзапах сточных вод\b",
                    r"\bвонь из труб\b",
                    r"\bзапах из туалета в коридоре\b",
                ],
                "en": [
                    r"\bsewage smell\b",
                    r"\bsmelled like sewage\b",
                    r"\bstrong sewer smell\b",
                    r"\bdrain smell in the hallway\b",
                    r"\bsmelled like the drains\b",
                ],
                "tr": [
                    r"\blağım kokusu vardı\b",
                    r"\bkanalizasyon kokusu\b",
                    r"\bkoridorda tuvalet gibi kokuyordu\b",
                    r"\bçok ağır bir lağım kokusu vardı\b",
                ],
                "ar": [
                    r"\bريحة مجاري\b",
                    r"\bريحة صرف صحي\b",
                    r"\bريحة حمام في الممر\b",
                    r"\bريحة مجاري طالعة عند المدخل\b",
                ],
                "zh": [
                    r"有下水道的味道",
                    r"一股臭水沟味",
                    r"楼道里有股下水道臭味",
                    r"厕所的味道飘到走廊",
                ],
            },
        ),
        
        
        "musty_smell": AspectRule(
            aspect_code="musty_smell",
            polarity_hint="negative",
            display="Запах сырости / плесени",
            display_short="запах сырости",
            long_hint="Гости отмечают затхлый запах, запах сырости или плесени в номере или в общих зонах.",
            patterns_by_lang={{
                "ru": [
                    r"\bзапах сырости\b",
                    r"\bзатхлый запах\b",
                    r"\bпахнет плесенью\b",
                    r"\bсыростью пахнет\b",
                    r"\bзапах старой влажной комнаты\b",
                ],
                "en": [
                    r"\bmusty smell\b",
                    r"\bthe room smelled musty\b",
                    r"\bdamp smell\b",
                    r"\bmoldy smell\b",
                    r"\bsmelled like mildew\b",
                ],
                "tr": [
                    r"\bküf kokusu vardı\b",
                    r"\bnem kokuyordu\b",
                    r"\brutubet kokusu vardı\b",
                    r"\boda küf gibi kokuyordu\b",
                ],
                "ar": [
                    r"\bريحة كمكمة\b",
                    r"\bريحة عفن/رطوبة\b",
                    r"\bريحة رطوبة في الأوضة\b",
                    r"\bالريحة عاملة زي رطوبة قديمة\b",
                ],
                "zh": [
                    r"有一股霉味",
                    r"房间有潮味",
                    r"闻起来很潮很闷",
                    r"有发霉的味道在房间里",
                ],
            },
        )    
}


ASPECT_TO_SUBTOPICS: Dict[str, List[Tuple[str, str]]] = {
            
            "spir_friendly": [("staff_spir", "staff_attitude")],
            "spir_welcoming": [("staff_spir", "staff_attitude")],
            "spir_helpful": [("staff_spir", "staff_attitude")],
            "spir_unfriendly": [("staff_spir", "staff_attitude")],
            "spir_rude": [("staff_spir", "staff_attitude")],
            "spir_unprofessional": [("staff_spir", "staff_attitude")],
            "spir_professional": [("staff_spir", "staff_attitude")],

            "spir_quick_response": [("staff_spir", "staff_responsiveness")],
            "spir_slow_response": [("staff_spir", "staff_responsiveness")],
            "spir_unresponsive": [("staff_spir", "staff_responsiveness")],
            "spir_ignored_requests": [("staff_spir", "staff_responsiveness")],

            "spir_easy_contact": [("staff_spir", "staff_availability")],
            "spir_hard_to_contact": [("staff_spir", "staff_availability")],
            "spir_available": [("staff_spir", "staff_availability")],
            "spir_not_available": [("staff_spir", "staff_availability")],

            "spir_problem_fixed_fast": [("staff_spir", "issue_resolution")],
            "spir_problem_not_fixed": [("staff_spir", "issue_resolution")],

            "spir_language_support_good": [("staff_spir", "language_support")],
            "spir_language_support_bad": [("staff_spir", "language_support")],

            "bed_uncomfortable": [("comfort", "sleep_quality")],
            "spir_rude": [("staff_spir", "staff_attitude")],

            "checkin_fast": [("checkin_stay", "checkin_speed")],
            "no_wait_checkin": [("checkin_stay", "checkin_speed")],
            "checkin_wait_long": [("checkin_stay", "checkin_speed")],
            "room_not_ready_delay": [("checkin_stay", "checkin_speed")],

            "room_ready_on_arrival": [("checkin_stay", "room_ready")],
            "clean_on_arrival": [("checkin_stay", "room_ready")],
            "room_not_ready": [("checkin_stay", "room_ready")],
            "dirty_on_arrival": [("checkin_stay", "room_ready")],
            "leftover_trash_previous_guest": [("checkin_stay", "room_ready")],

            "access_smooth": [("checkin_stay", "access")],
            "door_code_worked": [("checkin_stay", "access")],
            "tech_access_issue": [("checkin_stay", "access")],
            "entrance_hard_to_find": [("checkin_stay", "access")],
            "no_elevator_baggage_issue": [("checkin_stay", "access")],

            "payment_clear": [("checkin_stay", "docs_payment")],
            "deposit_clear": [("checkin_stay", "docs_payment")],
            "docs_provided": [("checkin_stay", "docs_payment")],
            "no_hidden_fees": [("checkin_stay", "docs_payment")],
            "payment_confusing": [("checkin_stay", "docs_payment")],
            "unexpected_charge": [("checkin_stay", "docs_payment")],
            "hidden_fees": [("checkin_stay", "docs_payment")],
            "deposit_problematic": [("checkin_stay", "docs_payment")],
            "billing_mistake": [("checkin_stay", "docs_payment")],
            "overcharge": [("checkin_stay", "docs_payment")],

            "instructions_clear": [("checkin_stay", "instructions")],
            "self_checkin_easy": [("checkin_stay", "instructions")],
            "wifi_info_given": [("checkin_stay", "instructions")],
            "instructions_confusing": [("checkin_stay", "instructions")],
            "late_access_code": [("checkin_stay", "instructions")],
            "wifi_info_missing": [("checkin_stay", "instructions")],
            "had_to_figure_out": [("checkin_stay", "instructions")],

            "support_during_stay_good": [("checkin_stay", "stay_support")],
            "issue_fixed_immediately": [("checkin_stay", "stay_support")],
            "support_during_stay_slow": [("checkin_stay", "stay_support")],
            "support_ignored": [("checkin_stay", "stay_support")],
            "promised_not_done": [("checkin_stay", "stay_support")],

            "checkout_easy": [("checkin_stay", "checkout")],
            "checkout_fast": [("checkin_stay", "checkout")],
            "checkout_slow": [("checkin_stay", "checkout")],
            "deposit_return_issue": [("checkin_stay", "checkout")],
            "checkout_no_staff": [("checkin_stay", "checkout")],

            "clean_on_arrival": [("cleanliness", "arrival_clean")],
            "fresh_bedding": [("cleanliness", "arrival_clean")],
            "no_dust_surfaces": [("cleanliness", "arrival_clean")],
            "floor_clean": [("cleanliness", "arrival_clean")],
            "dirty_on_arrival": [("cleanliness", "arrival_clean")],
            "dusty_surfaces": [("cleanliness", "arrival_clean")],
            "sticky_surfaces": [("cleanliness", "arrival_clean")],
            "stained_bedding": [("cleanliness", "arrival_clean")],
            "hair_on_bed": [("cleanliness", "arrival_clean")],
            "leftover_trash_previous_guest": [("cleanliness", "arrival_clean")],
            "used_towels_left": [("cleanliness", "arrival_clean")],
            "crumbs_left": [("cleanliness", "arrival_clean")],

            "bathroom_clean_on_arrival": [("cleanliness", "bathroom_state")],
            "no_mold_visible": [("cleanliness", "bathroom_state")],
            "sink_clean": [("cleanliness", "bathroom_state")],
            "shower_clean": [("cleanliness", "bathroom_state")],
            "bathroom_dirty_on_arrival": [("cleanliness", "bathroom_state")],
            "hair_in_shower": [("cleanliness", "bathroom_state")],
            "hair_in_sink": [("cleanliness", "bathroom_state")],
            "mold_in_shower": [("cleanliness", "bathroom_state")],
            "limescale_stains": [("cleanliness", "bathroom_state")],
            "sewage_smell_bathroom": [("cleanliness", "bathroom_state")],

            "housekeeping_regular": [("cleanliness", "stay_cleaning")],
            "trash_taken_out": [("cleanliness", "stay_cleaning")],
            "bed_made": [("cleanliness", "stay_cleaning")],
            "housekeeping_missed": [("cleanliness", "stay_cleaning")],
            "trash_not_taken": [("cleanliness", "stay_cleaning")],
            "bed_not_made": [("cleanliness", "stay_cleaning")],
            "had_to_request_cleaning": [("cleanliness", "stay_cleaning")],
            "dirt_accumulated": [("cleanliness", "stay_cleaning")],

            "towels_changed": [("cleanliness", "linen_towels")],
            "fresh_towels_fast": [("cleanliness", "linen_towels")],
            "linen_changed": [("cleanliness", "linen_towels")],
            "amenities_restocked": [("cleanliness", "linen_towels")],
            "towels_dirty": [("cleanliness", "linen_towels")],
            "towels_stained": [("cleanliness", "linen_towels")],
            "towels_smell": [("cleanliness", "linen_towels")],
            "towels_not_changed": [("cleanliness", "linen_towels")],
            "linen_not_changed": [("cleanliness", "linen_towels")],
            "no_restock": [("cleanliness", "linen_towels")],

            "smell_of_smoke": [("cleanliness", "smell")],
            "chemical_smell_strong": [("cleanliness", "smell")],
            "fresh_smell": [("cleanliness", "smell")],

            "hallway_clean": [("cleanliness", "public_areas")],
            "common_areas_clean": [("cleanliness", "public_areas")],
            "hallway_dirty": [("cleanliness", "public_areas")],
            "elevator_dirty": [("cleanliness", "public_areas")],
            "hallway_bad_smell": [("cleanliness", "public_areas")],
            "entrance_feels_unsafe": [("cleanliness", "public_areas")],

            "room_well_equipped": [("comfort", "room_equipment")],
            "kettle_available": [("comfort", "room_equipment")],
            "fridge_available": [("comfort", "room_equipment")],
            "hairdryer_available": [("comfort", "room_equipment")],
            "sockets_enough": [("comfort", "room_equipment")],
            "workspace_available": [("comfort", "room_equipment")],
            "luggage_space_ok": [("comfort", "room_equipment")],
            "kettle_missing": [("comfort", "room_equipment")],
            "fridge_missing": [("comfort", "room_equipment")],
            "hairdryer_missing": [("comfort", "room_equipment")],
            "sockets_not_enough": [("comfort", "room_equipment")],
            "no_workspace": [("comfort", "room_equipment")],
            "no_luggage_space": [("comfort", "room_equipment")],

            "bed_comfy": [("comfort", "sleep_quality")],
            "mattress_comfy": [("comfort", "sleep_quality")],
            "pillow_comfy": [("comfort", "sleep_quality")],
            "slept_well": [("comfort", "sleep_quality")],
            "bed_uncomfortable": [("comfort", "sleep_quality")],
            "mattress_too_soft": [("comfort", "sleep_quality")],
            "mattress_too_hard": [("comfort", "sleep_quality")],
            "mattress_sagging": [("comfort", "sleep_quality")],
            "bed_creaks": [("comfort", "sleep_quality")],
            "pillow_uncomfortable": [("comfort", "sleep_quality")],
            "pillow_too_hard": [("comfort", "sleep_quality")],
            "pillow_too_high": [("comfort", "sleep_quality")],

            "quiet_room": [("comfort", "noise")],
            "good_soundproofing": [("comfort", "noise")],
            "no_street_noise": [("comfort", "noise")],
            "noisy_room": [("comfort", "noise")],
            "street_noise": [("comfort", "noise")],
            "thin_walls": [("comfort", "noise")],
            "hallway_noise": [("comfort", "noise")],
            "night_noise_trouble_sleep": [("comfort", "noise")],

            # climate
            "temp_comfortable": [("comfort", "climate")],
            "ventilation_ok": [("comfort", "climate")],
            "ac_working": [("comfort", "climate")],
            "heating_working": [("comfort", "climate")],
            "too_hot_sleep_issue": [("comfort", "climate")],
            "too_cold": [("comfort", "climate")],
            "stuffy_no_air": [("comfort", "climate")],
            "no_ventilation": [("comfort", "climate")],
            "ac_not_working": [("comfort", "climate")],
            "no_ac": [("comfort", "climate")],
            "heating_not_working": [("comfort", "climate")],
            "draft_window": [("comfort", "climate")],

            "room_spacious": [("comfort", "space_light")],
            "good_layout": [("comfort", "space_light")],
            "cozy_feel": [("comfort", "space_light")],
            "bright_room": [("comfort", "space_light")],
            "big_windows": [("comfort", "space_light")],
            "room_small": [("comfort", "space_light")],
            "no_space_for_luggage": [("comfort", "space_light")],
            "dark_room": [("comfort", "space_light")],
            "no_natural_light": [("comfort", "space_light")],

            "hot_water_ok": [("tech_state", "plumbing_water")],
            "water_pressure_ok": [("tech_state", "plumbing_water")],
            "shower_ok": [("tech_state", "plumbing_water")],
            "no_leak": [("tech_state", "plumbing_water")],
            "no_hot_water": [("tech_state", "plumbing_water")],
            "weak_pressure": [("tech_state", "plumbing_water")],
            "shower_broken": [("tech_state", "plumbing_water")],
            "leak_water": [("tech_state", "plumbing_water")],
            "bathroom_flooding": [("tech_state", "plumbing_water")],
            "drain_clogged": [("tech_state", "plumbing_water")],
            "drain_smell": [("tech_state", "plumbing_water")],

            "ac_working_device": [("tech_state", "appliances_equipment")],
            "heating_working_device": [("tech_state", "appliances_equipment")],
            "appliances_ok": [("tech_state", "appliances_equipment")],
            "tv_working": [("tech_state", "appliances_equipment")],
            "fridge_working": [("tech_state", "appliances_equipment")],
            "kettle_working": [("tech_state", "appliances_equipment")],
            "door_secure": [("tech_state", "appliances_equipment")],
            "ac_broken": [("tech_state", "appliances_equipment")],
            "heating_broken": [("tech_state", "appliances_equipment")],
            "tv_broken": [("tech_state", "appliances_equipment")],
            "fridge_broken": [("tech_state", "appliances_equipment")],
            "kettle_broken": [("tech_state", "appliances_equipment")],
            "socket_danger": [("tech_state", "appliances_equipment")],
            "door_not_closing": [("tech_state", "appliances_equipment")],
            "lock_broken": [("tech_state", "appliances_equipment")],
            "furniture_broken": [("tech_state", "appliances_equipment")],
            "room_worn_out": [("tech_state", "appliances_equipment")],

            "wifi_fast": [("tech_state", "wifi_internet")],
            "internet_stable": [("tech_state", "wifi_internet")],
            "good_for_work": [("tech_state", "wifi_internet")],
            "wifi_down": [("tech_state", "wifi_internet")],
            "wifi_slow": [("tech_state", "wifi_internet")],
            "wifi_unstable": [("tech_state", "wifi_internet")],
            "wifi_hard_to_connect": [("tech_state", "wifi_internet")],
            "internet_not_suitable_for_work": [("tech_state", "wifi_internet")],

            "ac_noisy": [("tech_state", "tech_noise")],
            "fridge_noisy": [("tech_state", "tech_noise")],
            "pipes_noise": [("tech_state", "tech_noise")],
            "ventilation_noisy": [("tech_state", "tech_noise")],
            "night_mechanical_hum": [("tech_state", "tech_noise")],
            "tech_noise_sleep_issue": [("tech_state", "tech_noise")],
            "ac_quiet": [("tech_state", "tech_noise")],
            "fridge_quiet": [("tech_state", "tech_noise")],
            "no_tech_noise_night": [("tech_state", "tech_noise")],

            "elevator_working": [("tech_state", "elevator_infrastructure")],
            "luggage_easy": [("tech_state", "elevator_infrastructure")],
            "elevator_broken": [("tech_state", "elevator_infrastructure")],
            "elevator_stuck": [("tech_state", "elevator_infrastructure")],
            "no_elevator_heavy_bags": [("tech_state", "elevator_infrastructure")],

            "door_secure": [("tech_state", "lock_security")],
            "felt_safe": [("tech_state", "lock_security")],
            "door_not_closing": [("tech_state", "lock_security")],
            "lock_broken": [("tech_state", "lock_security")],
            "felt_unsafe": [("tech_state", "lock_security")],

            "breakfast_tasty": [("breakfast", "food_quality")],
            "food_fresh": [("breakfast", "food_quality")],
            "food_hot_served_hot": [("breakfast", "food_quality")],
            "coffee_good": [("breakfast", "food_quality")],
            "breakfast_bad_taste": [("breakfast", "food_quality")],
            "food_not_fresh": [("breakfast", "food_quality")],
            "food_cold": [("breakfast", "food_quality")],
            "coffee_bad": [("breakfast", "food_quality")],

            "breakfast_variety_good": [("breakfast", "variety_offering")],
            "buffet_rich": [("breakfast", "variety_offering")],
            "fresh_fruit_available": [("breakfast", "variety_offering")],
            "pastries_available": [("breakfast", "variety_offering")],
            "breakfast_variety_poor": [("breakfast", "variety_offering")],
            "breakfast_repetitive": [("breakfast", "variety_offering")],
            "hard_to_find_food": [("breakfast", "variety_offering")],

            "breakfast_staff_friendly": [("breakfast", "service_dining_staff")],
            "breakfast_staff_attentive": [("breakfast", "service_dining_staff")],
            "buffet_refilled_quickly": [("breakfast", "service_dining_staff")],
            "tables_cleared_fast": [("breakfast", "service_dining_staff")],
            "breakfast_staff_rude": [("breakfast", "service_dining_staff")],
            "no_refill_food": [("breakfast", "service_dining_staff")],
            "tables_left_dirty": [("breakfast", "service_dining_staff")],
            "ignored_requests": [("breakfast", "service_dining_staff")],

            "food_enough_for_all": [("breakfast", "availability_flow")],
            "kept_restocking": [("breakfast", "availability_flow")],
            "tables_available": [("breakfast", "availability_flow")],
            "no_queue": [("breakfast", "availability_flow")],
            "breakfast_flow_ok": [("breakfast", "availability_flow")],
            "food_ran_out": [("breakfast", "availability_flow")],
            "not_restocked": [("breakfast", "availability_flow")],
            "had_to_wait_food": [("breakfast", "availability_flow")],
            "no_tables_available": [("breakfast", "availability_flow")],
            "long_queue": [("breakfast", "availability_flow")],

            "breakfast_area_clean": [("breakfast", "cleanliness_breakfast")],
            "tables_cleaned_quickly": [("breakfast", "cleanliness_breakfast")],
            "dirty_tables": [("breakfast", "cleanliness_breakfast")],
            "dirty_dishes_left": [("breakfast", "cleanliness_breakfast")],
            "buffet_area_messy": [("breakfast", "cleanliness_breakfast")],

            "good_value": [("value", "value_for_money")],
            "worth_the_price": [("value", "value_for_money")],
            "affordable_for_level": [("value", "value_for_money")],
            "overpriced": [("value", "value_for_money")],
            "not_worth_price": [("value", "value_for_money")],
            "expected_better_for_price": [("value", "value_for_money")],

            "photos_misleading": [("value", "expectations_vs_price")],
            "quality_below_expectation": [("value", "expectations_vs_price")],

            "great_location": [("location", "proximity_area")],
            "central_convenient": [("location", "proximity_area")],
            "near_transport": [("location", "proximity_area")],
            "area_has_food_shops": [("location", "proximity_area")],
            "location_inconvenient": [("location", "proximity_area")],
            "far_from_center": [("location", "proximity_area")],
            "nothing_around": [("location", "proximity_area")],

            "area_safe": [("location", "safety_environment")],
            "area_quiet_at_night": [("location", "safety_environment")],
            "entrance_clean": [("cleanliness", "public_areas"), ("location", "safety_environment")],
            "area_unsafe": [("location", "safety_environment")],
            "uncomfortable_at_night": [("location", "safety_environment")],
            "entrance_dirty": [("cleanliness", "public_areas"), ("location", "safety_environment")],
            "people_loitering": [("location", "safety_environment")],

            "easy_to_find": [("location", "access_navigation")],
            "clear_instructions": [("location", "access_navigation")],
            "luggage_access_ok": [("location", "access_navigation")],
            "hard_to_find_entrance": [("location", "access_navigation")],
            "confusing_access": [("location", "access_navigation")],
            "no_signage": [("location", "access_navigation")],
            "luggage_access_hard": [("location", "access_navigation")],

            "cozy_atmosphere": [("atmosphere", "style_feel")],
            "nice_design": [("atmosphere", "style_feel")],
            "good_vibe": [("atmosphere", "style_feel")],
            "not_cozy": [("atmosphere", "style_feel")],
            "gloomy_feel": [("comfort", "space_light"), ("atmosphere", "style_feel")],
            "dated_look": [("atmosphere", "style_feel")],
            "soulless_feel": [("atmosphere", "style_feel")],

            "fresh_smell_common": [("atmosphere", "smell_common_areas")],
            "no_bad_smell": [("cleanliness", "smell"), ("atmosphere", "smell_common_areas")],
            "bad_smell_common": [("atmosphere", "smell_common_areas")],
            "cigarette_smell": [("atmosphere", "smell_common_areas")],
            "sewage_smell": [("cleanliness", "smell"), ("atmosphere", "smell_common_areas")],
            "musty_smell": [("cleanliness", "smell"), ("atmosphere", "smell_common_areas")],

        }

###############################################################################
# 5. Вспомогательные функции
###############################################################################

def _compile_regex_list(patterns: Iterable[str]) -> List[re.Pattern]:
    """
    Скомпилировать список регексов с флагами UNICODE / IGNORECASE / MULTILINE.
    Пустой вход -> пустой выход.
    """
    compiled: List[re.Pattern] = []
    for pat in patterns:
        try:
            compiled.append(
                re.compile(pat, re.IGNORECASE | re.UNICODE | re.MULTILINE)
            )
        except re.error:
            logging.exception("Regex compilation failed for pattern: %r", pat)
    return compiled


def _candidate_langs(lang: str) -> List[str]:
    """
    Возвращает список кандидатов языков для матчинга.
    Например: "en-US" -> ["en-us", "en", "en"] (повторы не страшны).
    Логика:
    - сначала как есть,
    - потом укороченный префикс до '-' (если есть),
    - потом "en" как дефолтный fallback.
    """
    lang = (lang or "").strip().lower()
    cands = []
    if lang:
        cands.append(lang)
        if "-" in lang:
            short = lang.split("-")[0]
            if short not in cands:
                cands.append(short)
    if "en" not in cands:
        cands.append("en")
    return cands

###############################################################################
# 6. Основной класс Lexicon
###############################################################################


class Lexicon:
    """
    Централизованный доступ к всем лексическим правилам.

    Основные публичные методы, которые будут использовать другие модули:
    - sentiment_for_sentence(text, lang)
    - get_sentiment_group_for_text(text, lang)
    - match_aspects_in_sentence(text, lang)
    - iter_aspect_rules(lang)
    - aspect_subtopics(aspect_code)
    - get_aspect_rule(aspect_code)
    - get_aspect_polarity_hint(aspect_code)
    - get_topic_schema()

    Внутри:
    - мы компилируем все регексы один раз при инициализации.
    """

    def __init__(
        self,
        sentiment_lexicon: Dict[str, Dict[str, List[str]]] = SENTIMENT_LEXICON,
        sentiment_key_to_group: Dict[str, str] = SENTIMENT_KEY_TO_GROUP,
        aspect_rules: Dict[str, AspectRule] = ASPECT_RULES,
        aspect_to_subtopics: Dict[str, List[Tuple[str, str]]] = ASPECT_TO_SUBTOPICS,
        topic_schema: Dict[str, Dict[str, Any]] = TOPIC_SCHEMA,
    ) -> None:
        # -------- тональность --------
        self._sentiment_lexicon_raw = sentiment_lexicon
        self._sentiment_key_to_group = sentiment_key_to_group
        self._compiled_sentiment_lexicon: Dict[str, Dict[str, List[re.Pattern]]] = (
            self._compile_sentiments(sentiment_lexicon)
        )

        # -------- аспекты --------
        self.aspect_rules: Dict[str, AspectRule] = aspect_rules
        self.aspect_to_subtopics: Dict[str, List[Tuple[str, str]]] = (
            aspect_to_subtopics
        )
        self._compiled_aspect_rules: Dict[str, Dict[str, List[re.Pattern]]] = (
            self._compile_aspects(aspect_rules)
        )

        # -------- топики / подтемы --------
         self._topic_schema = topic_schema
         self._compiled_topics = self._compile_topics(topic_schema)


    # ------------------------------------------------------------------
    # Компиляция тональностей
    # ------------------------------------------------------------------
    def _compile_sentiments(
        self, sentiment_lexicon: Dict[str, Dict[str, List[str]]]
    ) -> Dict[str, Dict[str, List[re.Pattern]]]:
        """
        Превращает:
            {
              'positive_soft': {'ru': [r'хорошо',...], 'en':[...], ...},
              'negative_strong': {...},
              ...
            }
        в:
            {
              'positive_soft': {'ru':[compiled,...], 'en':[compiled,...], ...},
              ...
            }
        """
        compiled: Dict[str, Dict[str, List[re.Pattern]]] = {}
        for sent_key, lang_map in sentiment_lexicon.items():
            compiled[sent_key] = {}
            for lang_code, patterns in lang_map.items():
                compiled[sent_key][lang_code] = _compile_regex_list(patterns)
        return compiled

    # ------------------------------------------------------------------
    # Компиляция аспектов
    # ------------------------------------------------------------------
    def _compile_aspects(
        self, aspect_rules: Dict[str, AspectRule]
    ) -> Dict[str, Dict[str, List[re.Pattern]]]:
        """
        Для каждого аспекта компилируем регексы по языкам:
            {
              'spir_friendly': {
                  'ru': [compiled_regex,...],
                  'en': [...],
              },
              ...
            }
        """
        compiled: Dict[str, Dict[str, List[re.Pattern]]] = {}
        for aspect_code, rule in aspect_rules.items():
            compiled[aspect_code] = {}
            for lang_code, patterns in rule.patterns_by_lang.items():
                compiled[aspect_code][lang_code] = _compile_regex_list(patterns)
        return compiled

    # ------------------------------------------------------------------
    # ТОНАЛЬНОСТЬ
    # ------------------------------------------------------------------
    def sentiment_for_sentence(
        self, text: str, lang: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Определить тональность КОНКРЕТНОГО ПРЕДЛОЖЕНИЯ.

        Возвращает:
            (sentiment_key, sentiment_group)
        где:
            sentiment_key   ~ 'positive_strong', 'negative_soft', ...
            sentiment_group ~ 'positive' / 'negative' / 'neutral'
        Если матчей нет -> (None, None)

        Heuristic priority задаётся SENTIMENT_EVAL_ORDER.
        Берём первое совпадение по приоритету.
        """
        if not text:
            return (None, None)

        candidates = _candidate_langs(lang)

        for sent_key in SENTIMENT_EVAL_ORDER:
            compiled_by_lang = self._compiled_sentiment_lexicon.get(sent_key, {})
            for cand_lang in candidates:
                patterns = compiled_by_lang.get(cand_lang, [])
                for rgx in patterns:
                    if rgx.search(text):
                        group = self._sentiment_key_to_group.get(sent_key)
                        return sent_key, group

        return (None, None)

    def get_sentiment_group_for_text(
        self, text: str, lang: str
    ) -> Optional[str]:
        """
        Упрощённая версия, если нам важна только полярность:
        'positive' / 'negative' / 'neutral' или None (если ничего не нашли).
        """
        _, group = self.sentiment_for_sentence(text, lang)
        return group

    # ------------------------------------------------------------------
    # АСПЕКТЫ
    # ------------------------------------------------------------------
    def match_aspects_in_sentence(
        self, text: str, lang: str
    ) -> List[str]:
        """
        Найти все аспекты, которые упоминаются в данном предложении.
        Возвращает список aspect_code (уникальных).
        """
        if not text:
            return []

        found: List[str] = []
        seen: set[str] = set()
        candidates = _candidate_langs(lang)

        for aspect_code, compiled_lang_map in self._compiled_aspect_rules.items():
            for cand_lang in candidates:
                patterns = compiled_lang_map.get(cand_lang, [])
                for rgx in patterns:
                    if rgx.search(text):
                        if aspect_code not in seen:
                            seen.add(aspect_code)
                            found.append(aspect_code)
                        # не break, т.к. хотим прогнать все языки/паттерны
        return found

    def iter_aspect_rules(
        self, lang: str
    ) -> Iterable[Tuple[str, AspectRule, List[re.Pattern]]]:
        """
        Удобный генератор: пройтись по всем аспектам и получить паттерны
        для конкретного языка (с fallback на en).
        Возвращает кортежи:
            (aspect_code, AspectRule, [compiled_regex,...])
        """
        candidates = _candidate_langs(lang)

        for aspect_code, rule in self.aspect_rules.items():
            # собираем паттерны для языка-кандидата
            collected: List[re.Pattern] = []
            compiled_lang_map = self._compiled_aspect_rules.get(aspect_code, {})
            for cand_lang in candidates:
                cand_patterns = compiled_lang_map.get(cand_lang, [])
                if cand_patterns:
                    collected.extend(cand_patterns)
            yield (aspect_code, rule, collected)

    # ------------------------------------------------------------------
    # МЕТА И СХЕМА
    # ------------------------------------------------------------------
    def aspect_subtopics(
        self, aspect_code: str
    ) -> List[Tuple[str, str]]:
        """
        Вернёт список (category_key, subtopic_key) для данного аспекта.
        Это нужно, чтобы сгруппировать упоминания по блокам отчёта
        (Персонал → Отношение и вежливость и т.д.).

        Пример:
            "spir_friendly" -> [("staff_spir", "staff_attitude")]
        """
        return self.aspect_to_subtopics.get(aspect_code, [])

    def get_aspect_rule(self, aspect_code: str) -> Optional[AspectRule]:
        """
        Вернуть сам объект AspectRule по его коду.
        """
        return self.aspect_rules.get(aspect_code)

    def get_aspect_polarity_hint(self, aspect_code: str) -> Optional[str]:
        """
        Быстрый доступ к polarity_hint аспекта ('positive'/'negative'/'neutral').
        """
        rule = self.get_aspect_rule(aspect_code)
        if rule is None:
            return None
        return rule.polarity_hint

    def get_topic_schema(self) -> Dict[str, Dict[str, Any]]:
        """
        Вернуть TOPIC_SCHEMA (например, чтобы узнать display-названия
        категорий и подтем при построении отчёта).
        """
        return self._topic_schema

    def get_aspect_display_short(self, aspect_code: str) -> Optional[str]:
        """
        Удобно для репорта/дашборда:
        возвращает короткое человекочитаемое имя аспекта.
        Если нет display_short -> пробуем display.
        Если нет display -> возвращаем сам код аспекта.
        """
        rule = self.get_aspect_rule(aspect_code)
        if rule is None:
            return None
        if rule.display_short:
            return rule.display_short
        if rule.display:
            return rule.display
        return aspect_code

    def get_aspect_long_hint(self, aspect_code: str) -> Optional[str]:
        """
        Возвращает human-readable пояснение аспекта.
        Используем в аналитических блоках отчёта (инсайты).
        """
        rule = self.get_aspect_rule(aspect_code)
        if rule is None:
            return None
        return rule.long_hint
      
      # --- Публичные свойства для совместимости с LexiconProtocol ---
      
      @property
      def compiled_sentiment(self) -> Dict[str, Dict[str, List[re.Pattern]]]:
          """
          Псевдоним к внутреннему словарю скомпилированных паттернов тональностей.
          Ожидаемая структура:
          {
            "positive_strong": {"ru": [re.Pattern,...], "en": [...] , ...},
            "positive_soft":   {...},
            "negative_soft":   {...},
            "negative_strong": {...},
            "neutral":         {...},
          }
          """
          return self._compiled_sentiment_lexicon
      
      @property
      def compiled_aspects(self) -> Dict[str, Dict[str, List[re.Pattern]]]:
          """
          Псевдоним к внутренним скомпилированным правилам аспектов.
          Ожидаемая структура:
          { "aspect_code": {"ru":[...], "en":[...], ...}, ... }
          """
          return self._compiled_aspect_rules
      
      @property
      def topic_schema(self) -> Dict[str, Dict[str, Any]]:
          """
          Исходная схема тем (как загружена/передана при инициализации).
          Ожидается формат с категориями -> подтемами.
          """
          return self._topic_schema
      
      @property
      def compiled_topics(self) -> Dict[str, Dict[str, Dict[str, List[re.Pattern]]]]:
          """
          Скомпилированные паттерны тем/подтем по языкам.
          Структура:
          {
            "<topic_key>": {
              "<subtopic_key>": {
                "ru": [re.Pattern, ...],
                "en": [...],
                ...
              },
              ...
            },
            ...
          }
          """
          return self._compiled_topics
      
      # --- Компиляция тем/подтем ---
      
      def _compile_topics(
          self,
          topic_schema: Dict[str, Dict[str, Any]],
      ) -> Dict[str, Dict[str, Dict[str, List[re.Pattern]]]]:
          """
          Поддерживает оба варианта схемы:
          - "patterns": {"ru":[...], "en":[...], ...}
          - "patterns_by_lang": {"ru":[...], ...}
          """
          compiled: Dict[str, Dict[str, Dict[str, List[re.Pattern]]]] = {}
          for topic_key, topic_def in topic_schema.items():
              subtopics: Dict[str, Any] = topic_def.get("subtopics", {})
              for sub_key, sub_def in subtopics.items():
                  raw = sub_def.get("patterns")
                  if raw is None:
                      raw = sub_def.get("patterns_by_lang")  # fallback на альтернативное имя
                  if not raw:
                      continue
                  for lang_code, patterns in raw.items():
                      compiled \
                          .setdefault(topic_key, {}) \
                          .setdefault(sub_key, {}) \
                          .setdefault(lang_code, []) \
                          .extend(_compile_regex_list(patterns))
          return compiled
      
      # --- Матчинг тем/подтем в тексте (по предложениям можно вызывать наружу при желании) ---
      
      def match_topics(self, text: str, lang: str) -> List[Tuple[str, str]]:
          """
          Простой матчинг пар (topic, subtopic) в целом тексте.
          Возвращает уникальные пары.
          """
          if not text:
              return []
          found: List[Tuple[str, str]] = []
          seen: set[Tuple[str, str]] = set()
          candidates = _candidate_langs(lang)
          for topic_key, sub_map in self._compiled_topics.items():
              for sub_key, lang_map in sub_map.items():
                  pats: List[re.Pattern] = []
                  for c in candidates:
                      pats.extend(lang_map.get(c, []))
                  if any(p.search(text) for p in pats):
                      pair = (topic_key, sub_key)
                      if pair not in seen:
                          seen.add(pair)
                          found.append(pair)
          return found
      
      # --- Детект языка (минималистичный хак без внешних зависимостей) ---
      
      def detect_lang(self, text: str) -> str:
          """
          Грубая эвристика:
          - кириллица -> 'ru'
          - арабская письменность -> 'ar'
          - турецкие специфичные буквы -> 'tr'
          - CJK диапазоны -> 'zh'
          - иначе -> 'en'
          """
          if not text:
              return "en"
          s = text
          # Упростим определение по диапазонам/символам
          if re.search(r"[А-Яа-яЁё]", s):
              return "ru"
          if re.search(r"[\u0600-\u06FF]", s):  # Arabic
              return "ar"
          if re.search(r"[ıİğĞşŞçÇöÖüÜ]", s):
              return "tr"
          if re.search(r"[\u4E00-\u9FFF]", s):  # CJK Unified Ideographs
              return "zh"
          return "en"

    # ------------------------------------------------------------------
    # Утилиты для дебага / отладки
    # ------------------------------------------------------------------
    def debug_all_languages(self) -> Dict[str, List[str]]:
        """
        Вернёт словарь:
           lang -> [aspect_codes, ...]
        чтобы посмотреть покрытие языков.
        Чисто для отладки/аналитики.
        """
        coverage: Dict[str, set] = {}
        for aspect_code, rule in self.aspect_rules.items():
            for lang_code in rule.patterns_by_lang.keys():
                coverage.setdefault(lang_code, set()).add(aspect_code)

        return {lang: sorted(list(aspects)) for lang, aspects in coverage.items()}

      LexiconModule = Lexicon  # совместимость с импортами вида lexicon_module.LexiconModule
