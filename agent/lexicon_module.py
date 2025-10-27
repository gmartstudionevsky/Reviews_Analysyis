from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Iterable


###############################################################################
# 1. Типы данных
###############################################################################

@dataclass(frozen=True)
class SentimentPatterns:
    """
    Регекс-паттерны для одной тональности.
    Ключ - язык (ru/en/tr/ar/zh), значение - список паттернов (строки regex).
    """
    patterns_by_lang: Dict[str, List[str]]


@dataclass(frozen=True)
class Subtopic:
    """
    Подтема внутри большой категории.
    - display: как показываем в отчёте
    - patterns_by_lang: { lang: [regex, ...] } — триггеры упоминания этой подтемы
    - aspects: список кодов аспектов, которые могут быть упомянуты внутри этой подтемы
    """
    display: str
    patterns_by_lang: Dict[str, List[str]]
    aspects: List[str]


@dataclass(frozen=True)
class TopicCategory:
    """
    Категория (пример: 'Персонал', 'Чистота', 'Комфорт проживания', и т.д.)
    - display: название категории для отчёта
    - subtopics: словарь {subtopic_key -> Subtopic}
    """
    display: str
    subtopics: Dict[str, Subtopic]


@dataclass(frozen=True)
class AspectMeta:
    """
    Метаданные аспекта.
    aspect_code: машинный код (напр. 'wifi_unstable')
    display_short: короткое человекочитаемое имя для буллетов и таблиц.
                   Пример: 'нестабильный интернет'
    long_hint (опц.): чуть более развёрнутое описание/контекст для генерации текста отчёта
                      (если нужно автоматически формировать фразы).
    """
    aspect_code: str
    display_short: str
    long_hint: Optional[str] = None

@dataclass(frozen=True)
class AspectRule:
    aspect_code: str
    patterns_by_lang: Dict[str, List[str]]
    polarity_hint: str  # "positive" / "negative" / "neutral"


###############################################################################
# 2. Класс Lexicon
###############################################################################

class Lexicon:
    """
    Хранилище:
    - словари тональностей
    - схема тематик/подтем/аспектов
    - человекочитаемые описания аспектов

    Это ЕДИНЫЙ источник правды, который импортируют остальные модули.
    Вся логика поиска по тексту (sentiment_tagging, topic_tagging и т.д.)
    должна читать паттерны только отсюда.
    """

    def __init__(self):
        # Версия словаря (меняем вручную при апдейтах)
        self.version = "2025-10-27_v1"

        #######################################################################
        # 2.1. Тональности
        #######################################################################
        self.sentiment_lexicon: Dict[str, SentimentPatterns] = {
            # Сильный позитив
            "positive_strong": SentimentPatterns(
                patterns_by_lang={
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
            ),

            # Мягкий позитив
            "positive_soft": SentimentPatterns(
                patterns_by_lang={
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
            ),

            # Мягкий негатив
            "negative_soft": SentimentPatterns(
                patterns_by_lang={
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
            ),

            # Сильный негатив
            "negative_strong": SentimentPatterns(
                patterns_by_lang={
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
            ),

            # Нейтрально / приемлемо
            "neutral": SentimentPatterns(
                patterns_by_lang={
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
            ),
        }

        #######################################################################
        # 2.2. Тематическая схема (категория -> подтемы -> аспекты)
        #######################################################################

        # Здесь мы напрямую используем предоставленный TOPIC_SCHEMA,
        # но храним его уже как объекты TopicCategory/Subtopic.
        # ВНИМАНИЕ: это будет длинно, но это единственный источник правды.

        self.topic_schema: Dict[str, TopicCategory] = {
            "staff_spir": TopicCategory(
                display="Персонал",
                subtopics={
                    "staff_attitude": Subtopic(
                        display="Отношение и вежливость",
                        patterns_by_lang={
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
                        aspects=[
                            "spir_friendly",
                            "spir_polite",
                            "spir_rude",
                            "spir_unrespectful",
                        ],
                    ),

                    "staff_helpfulness": Subtopic(
                        display="Помощь и решение вопросов",
                        patterns_by_lang={
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
                        aspects=[
                            "spir_helpful_fast",
                            "spir_problem_solved",
                            "spir_info_clear",
                            "spir_unhelpful",
                            "spir_problem_ignored",
                            "spir_info_confusing",
                        ],
                    ),

                    "staff_speed": Subtopic(
                        display="Оперативность и скорость реакции",
                        patterns_by_lang={
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
                        aspects=[
                            "spir_helpful_fast",
                            "spir_fast_response",
                            "spir_slow_response",
                            "spir_absent",
                            "spir_no_answer",
                        ],
                    ),

                    "staff_professionalism": Subtopic(
                        display="Профессионализм и компетентность",
                        patterns_by_lang={
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
                        aspects=[
                            "spir_professional",
                            "spir_info_clear",
                            "spir_payment_clear",
                            "spir_unprofessional",
                            "spir_info_confusing",
                            "spir_payment_issue",
                            "spir_booking_mistake",
                        ],
                    ),

                    "staff_availability": Subtopic(
                        display="Доступность персонала",
                        patterns_by_lang={
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
                        aspects=[
                            "spir_available",
                            "spir_24h_support",
                            "spir_absent",
                            "spir_no_answer",
                            "spir_no_night_support",
                        ],
                    ),

                    "staff_communication": Subtopic(
                        display="Коммуникация и понятность объяснений",
                        patterns_by_lang={
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
                        aspects=[
                            "spir_info_clear",
                            "spir_language_ok",
                            "spir_info_confusing",
                            "spir_language_barrier",
                        ],
                    ),
                },
            ),
            "checkin_stay": TopicCategory(
                display="Заселение и проживание",
                subtopics={
            
                    "checkin_speed": Subtopic(
                        display="Скорость заселения",
                        patterns_by_lang={
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
                        aspects=[
                            "checkin_fast", "no_wait_checkin",
                            "checkin_wait_long", "room_not_ready_delay",
                        ],
                    ),
            
                    "room_ready": Subtopic(
                        display="Готовность номера к заселению",
                        patterns_by_lang={
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
                        aspects=[
                            "room_ready_on_arrival", "clean_on_arrival",
                            "room_not_ready", "dirty_on_arrival", "leftover_trash_previous_guest",
                        ],
                    ),
            
                    "access": Subtopic(
                        display="Доступ и вход в отель / номер",
                        patterns_by_lang={
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
                        aspects=[
                            "access_smooth", "door_code_worked",
                            "tech_access_issue", "entrance_hard_to_find", "no_elevator_baggage_issue",
                        ],
                    ),
            
                    "docs_payment": Subtopic(
                        display="Оплата, депозиты и документы",
                        patterns_by_lang={
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
                        aspects=[
                            "payment_clear", "deposit_clear", "docs_provided", "no_hidden_fees",
                            "payment_confusing", "unexpected_charge", "hidden_fees",
                            "deposit_problematic", "billing_mistake", "overcharge",
                        ],
                    ),
            
                    "instructions": Subtopic(
                        display="Инструкции по заселению и проживанию",
                        patterns_by_lang={
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
                        aspects=[
                            "instructions_clear", "self_checkin_easy", "wifi_info_given",
                            "instructions_confusing", "late_access_code", "wifi_info_missing", "had_to_figure_out",
                        ],
                    ),
            
                    "stay_support": Subtopic(
                        display="Поддержка во время проживания",
                        patterns_by_lang={
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
                        aspects=[
                            "support_during_stay_good", "issue_fixed_immediately",
                            "support_during_stay_slow", "support_ignored", "promised_not_done",
                        ],
                    ),
            
                    "checkout": Subtopic(
                        display="Выезд",
                        patterns_by_lang={
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
                        aspects=[
                            "checkout_easy", "checkout_fast",
                            "checkout_slow", "deposit_return_issue", "checkout_no_staff",
                        ],
                    ),
            
                },
            ),
            "cleanliness": TopicCategory(
                display="Чистота",
                subtopics={
            
                    "arrival_clean": Subtopic(
                        display="Чистота номера при заезде",
                        patterns_by_lang={
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
                        aspects=[
                            "clean_on_arrival", "fresh_bedding", "no_dust_surfaces",
                            "floor_clean",
                            "dirty_on_arrival", "dusty_surfaces", "sticky_surfaces",
                            "stained_bedding", "hair_on_bed",
                            "leftover_trash_previous_guest", "used_towels_left", "crumbs_left",
                        ],
                    ),
            
                    "bathroom_state": Subtopic(
                        display="Санузел при заезде",
                        patterns_by_lang={
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
                        aspects=[
                            "bathroom_clean_on_arrival", "no_mold_visible", "sink_clean", "shower_clean",
                            "bathroom_dirty_on_arrival", "hair_in_shower", "hair_in_sink",
                            "mold_in_shower", "limescale_stains", "sewage_smell_bathroom",
                        ],
                    ),
            
                    "stay_cleaning": Subtopic(
                        display="Уборка во время проживания",
                        patterns_by_lang={
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
                        aspects=[
                            "housekeeping_regular", "trash_taken_out", "bed_made",
                            "housekeeping_missed", "trash_not_taken", "bed_not_made",
                            "had_to_request_cleaning", "dirt_accumulated",
                        ],
                    ),
            
                    "linen_towels": Subtopic(
                        display="Полотенца, бельё и принадлежности",
                        patterns_by_lang={
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
                        aspects=[
                            "towels_changed", "fresh_towels_fast", "linen_changed",
                            "amenities_restocked",
                            "towels_dirty", "towels_stained", "towels_smell",
                            "towels_not_changed", "linen_not_changed", "no_restock",
                        ],
                    ),
            
                    "smell": Subtopic(
                        display="Запахи",
                        patterns_by_lang={
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
                        aspects=[
                            "smell_of_smoke", "sewage_smell", "musty_smell",
                            "chemical_smell_strong",
                            "no_bad_smell", "fresh_smell",
                        ],
                    ),
            
                    "public_areas": Subtopic(
                        display="Общие зоны и входные группы",
                        patterns_by_lang={
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
                        aspects=[
                            "entrance_clean", "hallway_clean", "common_areas_clean",
                            "entrance_dirty", "hallway_dirty", "elevator_dirty",
                            "hallway_bad_smell", "entrance_feels_unsafe",
                        ],
                    ),
                },
            ),
            "comfort": TopicCategory(
                display="Комфорт проживания",
                subtopics={
            
                    "room_equipment": Subtopic(
                        display="Оснащение и удобство номера",
                        patterns_by_lang={
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
                        aspects=[
                            "room_well_equipped", "kettle_available", "fridge_available",
                            "hairdryer_available", "sockets_enough", "workspace_available",
                            "luggage_space_ok",
                            "kettle_missing", "fridge_missing", "hairdryer_missing",
                            "sockets_not_enough", "no_workspace", "no_luggage_space",
                        ],
                    ),
            
                    "sleep_quality": Subtopic(
                        display="Сон и кровать",
                        patterns_by_lang={
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
                        aspects=[
                            "bed_comfy", "mattress_comfy", "pillow_comfy", "slept_well",
                            "bed_uncomfortable", "mattress_too_soft", "mattress_too_hard",
                            "mattress_sagging", "bed_creaks",
                            "pillow_uncomfortable", "pillow_too_hard", "pillow_too_high",
                        ],
                    ),
            
                    "noise": Subtopic(
                        display="Шум и звукоизоляция",
                        patterns_by_lang={
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
                        aspects=[
                            "quiet_room", "good_soundproofing", "no_street_noise",
                            "noisy_room", "street_noise", "thin_walls",
                            "hallway_noise", "night_noise_trouble_sleep",
                        ],
                    ),
            
                    "climate": Subtopic(
                        display="Температура и воздух",
                        patterns_by_lang={
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
                        aspects=[
                            "temp_comfortable", "ventilation_ok", "ac_working", "heating_working",
                            "too_hot_sleep_issue", "too_cold", "stuffy_no_air",
                            "no_ventilation", "ac_not_working", "no_ac",
                            "heating_not_working", "draft_window",
                        ],
                    ),
            
                    "space_light": Subtopic(
                        display="Пространство и освещённость",
                        patterns_by_lang={
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
                        aspects=[
                            "room_spacious", "good_layout", "cozy_feel",
                            "bright_room", "big_windows",
                            "room_small", "no_space_for_luggage",
                            "dark_room", "no_natural_light", "gloomy_feel",
                        ],
                    ),
                },
            ),
            "tech_state": TopicCategory(
                display="Техническое состояние и инфраструктура",
                subtopics={
            
                    "plumbing_water": Subtopic(
                        display="Вода и сантехника",
                        patterns_by_lang={
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
                        aspects=[
                            "hot_water_ok", "water_pressure_ok", "shower_ok", "no_leak",
                            "no_hot_water", "weak_pressure", "shower_broken",
                            "leak_water", "bathroom_flooding",
                            "drain_clogged", "drain_smell",
                        ],
                    ),
            
                    "appliances_equipment": Subtopic(
                        display="Оборудование и состояние номера",
                        patterns_by_lang={
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
                        aspects=[
                            "ac_working_device", "heating_working_device",
                            "appliances_ok", "tv_working", "fridge_working", "kettle_working",
                            "door_secure",
                            "ac_broken", "heating_broken", "tv_broken", "fridge_broken",
                            "kettle_broken", "socket_danger",
                            "door_not_closing", "lock_broken", "furniture_broken",
                            "room_worn_out",
                        ],
                    ),
            
                    "wifi_internet": Subtopic(
                        display="Wi-Fi и интернет",
                        patterns_by_lang={
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
                        aspects=[
                            "wifi_fast", "internet_stable", "good_for_work",
                            "wifi_down", "wifi_slow", "wifi_unstable",
                            "wifi_hard_to_connect", "internet_not_suitable_for_work",
                        ],
                    ),
            
                    "tech_noise": Subtopic(
                        display="Шум оборудования и инженерки",
                        patterns_by_lang={
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
                        aspects=[
                            "ac_noisy", "fridge_noisy", "pipes_noise",
                            "ventilation_noisy", "night_mechanical_hum",
                            "tech_noise_sleep_issue",
                            "ac_quiet", "fridge_quiet", "no_tech_noise_night",
                        ],
                    ),
            
                    "elevator_infrastructure": Subtopic(
                        display="Лифт и доступ с багажом",
                        patterns_by_lang={
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
                        aspects=[
                            "elevator_working", "luggage_easy",
                            "elevator_broken", "elevator_stuck",
                            "no_elevator_heavy_bags",
                        ],
                    ),
            
                    "lock_security": Subtopic(
                        display="Двери и безопасность",
                        patterns_by_lang={
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
                        aspects=[
                            "door_secure", "felt_safe",
                            "door_not_closing", "lock_broken", "felt_unsafe",
                        ],
                    ),
                },
            ),
            "breakfast": TopicCategory(
                display="Завтрак и питание",
                subtopics={
            
                    "food_quality": Subtopic(
                        display="Качество и вкус блюд",
                        patterns_by_lang={
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
                        aspects=[
                            "breakfast_tasty", "food_fresh", "food_hot_served_hot", "coffee_good",
                            "breakfast_bad_taste", "food_not_fresh", "food_cold", "coffee_bad",
                        ],
                    ),
            
                    "variety_offering": Subtopic(
                        display="Разнообразие и выбор",
                        patterns_by_lang={
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
                        aspects=[
                            "breakfast_variety_good", "buffet_rich", "fresh_fruit_available", "pastries_available",
                            "breakfast_variety_poor", "breakfast_repetitive", "hard_to_find_food",
                        ],
                    ),
            
                    "service_dining_staff": Subtopic(
                        display="Сервис завтрака (персонал)",
                        patterns_by_lang={
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
                        aspects=[
                            "breakfast_staff_friendly", "breakfast_staff_attentive",
                            "buffet_refilled_quickly", "tables_cleared_fast",
                            "breakfast_staff_rude", "no_refill_food",
                            "tables_left_dirty", "ignored_requests",
                        ],
                    ),
            
                    "availability_flow": Subtopic(
                        display="Наличие еды и организация завтрака",
                        patterns_by_lang={
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
                        aspects=[
                            "food_enough_for_all", "kept_restocking",
                            "tables_available", "no_queue", "breakfast_flow_ok",
                            "food_ran_out", "not_restocked",
                            "had_to_wait_food", "no_tables_available", "long_queue",
                        ],
                    ),
            
                    "cleanliness_breakfast": Subtopic(
                        display="Чистота на завтраке",
                        patterns_by_lang={
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
                        aspects=[
                            "breakfast_area_clean", "tables_cleaned_quickly",
                            "dirty_tables", "dirty_dishes_left",
                            "buffet_area_messy",
                        ],
                    ),
                },
            ),
            "value": TopicCategory(
                display="Цена и ценность",
                subtopics={
            
                    "value_for_money": Subtopic(
                        display="Соотношение цена/качество",
                        patterns_by_lang={
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
                        aspects=[
                            "good_value", "worth_the_price", "affordable_for_level",
                            "overpriced", "not_worth_price", "expected_better_for_price",
                        ],
                    ),
            
                    "expectations_vs_price": Subtopic(
                        display="Ожидания vs цена",
                        patterns_by_lang={
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
                        aspects=[
                            "photos_misleading", "quality_below_expectation",
                        ],
                    ),
                },
            ),
            "location": TopicCategory(
                display="Локация и окружение",
                subtopics={
            
                    "proximity_area": Subtopic(
                        display="Расположение и окружение",
                        patterns_by_lang={
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
                        aspects=[
                            "great_location", "central_convenient", "near_transport", "area_has_food_shops",
                            "location_inconvenient", "far_from_center", "nothing_around",
                        ],
                    ),
            
                    "safety_environment": Subtopic(
                        display="Ощущение района и безопасность",
                        patterns_by_lang={
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
                        aspects=[
                            "area_safe", "area_quiet_at_night", "entrance_clean",
                            "area_unsafe", "uncomfortable_at_night", "entrance_dirty", "people_loitering",
                        ],
                    ),
            
                    "access_navigation": Subtopic(
                        display="Доступ и навигация",
                        patterns_by_lang={
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
                        aspects=[
                            "easy_to_find", "clear_instructions", "luggage_access_ok",
                            "hard_to_find_entrance", "confusing_access", "no_signage", "luggage_access_hard",
                        ],
                    ),
                },
            ),
            "atmosphere": TopicCategory(
                display="Атмосфера и общее впечатление",
                subtopics={
            
                    "style_feel": Subtopic(
                        display="Атмосфера и уют",
                        patterns_by_lang={
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
                        aspects=[
                            "cozy_atmosphere", "nice_design", "good_vibe",
                            "not_cozy", "gloomy_feel", "dated_look", "soulless_feel",
                        ],
                    ),
            
                    "smell_common_areas": Subtopic(
                        display="Запах и ощущение общих зон",
                        patterns_by_lang={
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
                        aspects=[
                            "fresh_smell_common", "no_bad_smell",
                            "bad_smell_common", "cigarette_smell", "sewage_smell", "musty_smell",
                        ],
                    ),
                },
            ),
        }

        #######################################################################
        # 2.3. Метаданные по аспектам
        #
        # Это словарь "aspect_code -> AspectMeta".
        # Эти тексты будут использоваться в отчёте (буллеты, подписи графиков).
        # Здесь мы даём короткие человекочитаемые ярлыки.
        #
        # Важно: если аспект встречается в нескольких категориях
        # (напр. door_secure всплывает и в "tech_state.lock_security", и в "location.safety_environment"
        #   как 'entrance_clean/dirty' и ощущение безопасности),
        # мы всё равно заводим одну запись, чтобы репорт говорил одинаково.
        #######################################################################

        self.aspects_meta: Dict[str, AspectMeta] = {
            # =========================
            # spir_staff
            # =========================
            "spir_friendly": AspectMeta(
                aspect_code="spir_friendly",
                display_short="дружелюбный персонал",
                long_hint="Гости подчеркивают приветливость и доброжелательность сотрудников."
            ),
            "spir_polite": AspectMeta(
                aspect_code="spir_polite",
                display_short="вежливый персонал",
                long_hint="Отмечают корректное, уважительное общение: персонал приветлив, учтив, относится с уважением."
            ),
            "spir_rude": AspectMeta(
                aspect_code="spir_rude",
                display_short="грубость персонала",
                long_hint="Жёстко негативный фидбек об общении: грубость, хамство, резкий тон."
            ),
            "spir_unrespectful": AspectMeta(
                aspect_code="spir_unrespectful",
                display_short="неуважительное отношение",
                long_hint="Гости пишут, что с ними разговаривали свысока, без уважения, позволяли себе хамство."
            ),

            "spir_helpful_fast": AspectMeta(
                aspect_code="spir_helpful_fast",
                display_short="быстро помогли",
                long_hint="Сотрудники сразу занялись вопросом гостя: пришли/принесли/починили без промедления."
            ),
            "spir_problem_solved": AspectMeta(
                aspect_code="spir_problem_solved",
                display_short="проблему решили",
                long_hint="Гости отмечают, что персонал реально решил их запрос: исправили проблему, нашли решение."
            ),
            "spir_info_clear": AspectMeta(
                aspect_code="spir_info_clear",
                display_short="всё ясно объяснили",
                long_hint="Подробно и понятно объяснили правила, доступ, оплату, куда идти и что делать."
            ),
            "spir_unhelpful": AspectMeta(
                aspect_code="spir_unhelpful",
                display_short="персонал не помогает",
                long_hint="Жалобы, что сотрудники не стали помогать, отмахнулись или 'это не наша проблема'."
            ),
            "spir_problem_ignored": AspectMeta(
                aspect_code="spir_problem_ignored",
                display_short="игнор проблемы",
                long_hint="Гости пишут, что обращение проигнорировали: обещали и не сделали / никто не занялся."
            ),
            "spir_info_confusing": AspectMeta(
                aspect_code="spir_info_confusing",
                display_short="непонятные объяснения",
                long_hint="Инструкции были путаные или противоречивые, пришлось разбираться самостоятельно."
            ),

            "spir_fast_response": AspectMeta(
                aspect_code="spir_fast_response",
                display_short="быстрая реакция персонала",
                long_hint="Персонал откликался сразу: быстро пришли, быстро оформили, моментально ответили."
            ),
            "spir_slow_response": AspectMeta(
                aspect_code="spir_slow_response",
                display_short="медленная реакция персонала",
                long_hint="Долго ждали оформления/ответа/помощи, отмечают затяжные задержки."
            ),
            "spir_absent": AspectMeta(
                aspect_code="spir_absent",
                display_short="персонала нет на месте",
                long_hint="На стойке/ресепшене никого нет, никто не пришёл по запросу, не дождались помощи."
            ),
            "spir_no_answer": AspectMeta(
                aspect_code="spir_no_answer",
                display_short="не отвечают",
                long_hint="Не берут трубку / не отвечают на сообщения / невозможно дозвониться."
            ),

            "spir_professional": AspectMeta(
                aspect_code="spir_professional",
                display_short="профессиональный подход",
                long_hint="Гости называют персонал компетентным и организованным: всё оформили правильно, без сумбура."
            ),
            "spir_payment_clear": AspectMeta(
                aspect_code="spir_payment_clear",
                display_short="прозрачная оплата",
                long_hint="Чётко объяснили оплату и депозит, оформили документы, выдали чек/счёт без вопросов."
            ),
            "spir_unprofessional": AspectMeta(
                aspect_code="spir_unprofessional",
                display_short="непрофессионально",
                long_hint="Некомпетентность, бардак с документами/бронью, не могут ничего толком объяснить."
            ),
            "spir_payment_issue": AspectMeta(
                aspect_code="spir_payment_issue",
                display_short="путаница с оплатой",
                long_hint="Жалобы на непрозрачные списания, ошибки в счёте, странные депозиты."
            ),
            "spir_booking_mistake": AspectMeta(
                aspect_code="spir_booking_mistake",
                display_short="ошибка с бронированием",
                long_hint="Перепутали или потеряли бронь, неправильно оформили заезд, путают даты / тип номера."
            ),

            "spir_available": AspectMeta(
                aspect_code="spir_available",
                display_short="персонал доступен",
                long_hint="Гости пишут, что сотрудников легко найти, они всегда на связи и помогают."
            ),
            "spir_24h_support": AspectMeta(
                aspect_code="spir_24h_support",
                display_short="помощь 24/7",
                long_hint="Персонал доступен круглосуточно и помогает даже ночью."
            ),
            "spir_no_night_support": AspectMeta(
                aspect_code="spir_no_night_support",
                display_short="нет ночной поддержки",
                long_hint="Ночью никого нет: ресепшен закрыт, никто не отвечает и не выходит."
            ),

            "spir_language_ok": AspectMeta(
                aspect_code="spir_language_ok",
                display_short="без языкового барьера",
                long_hint="Персонал нормально коммуницирует, говорит на понятном гостю языке (часто отмечают хороший английский)."
            ),
            "spir_language_barrier": AspectMeta(
                aspect_code="spir_language_barrier",
                display_short="языковой барьер",
                long_hint="Гостям сложно объясниться: персонал не говорит на нужном языке, тяжело понять друг друга."
            ),
            # =========================
            # checkin_stay
            # =========================

            "checkin_fast": AspectMeta(
                aspect_code="checkin_fast",
                display_short="быстрое заселение",
                long_hint="Гости отмечают, что заселили сразу или оформили за пару минут, без проволочек и очередей."
            ),
            "no_wait_checkin": AspectMeta(
                aspect_code="no_wait_checkin",
                display_short="без ожидания при заезде",
                long_hint="Не пришлось ждать — номер выдали сразу, без задержек на ресепшене и без очереди."
            ),
            "checkin_wait_long": AspectMeta(
                aspect_code="checkin_wait_long",
                display_short="долго ждали заселения",
                long_hint="Жалобы, что чек-ин занял слишком много времени: большая очередь, долгое оформление, пришлось стоять и ждать."
            ),
            "room_not_ready_delay": AspectMeta(
                aspect_code="room_not_ready_delay",
                display_short="номер не был готов вовремя",
                long_hint="Номер не был подготовлен к указанному времени заезда, поэтому гостям пришлось ждать, пока уберут или подготовят."
            ),

            "room_ready_on_arrival": AspectMeta(
                aspect_code="room_ready_on_arrival",
                display_short="номер готов при заезде",
                long_hint="Номер был полностью подготовлен к моменту прибытия: убрано, всё на месте, можно сразу заезжать."
            ),
            "clean_on_arrival": AspectMeta(
                aspect_code="clean_on_arrival",
                display_short="чисто при заезде",
                long_hint="Гости пишут, что при заселении было чисто: свежее бельё, порядок, никаких следов предыдущих гостей."
            ),
            "room_not_ready": AspectMeta(
                aspect_code="room_not_ready",
                display_short="номер не подготовили к заезду",
                long_hint="Гости жалуются, что номер ещё не был готов: не убрано, постель не сменили, остались следы предыдущих гостей."
            ),
            "dirty_on_arrival": AspectMeta(
                aspect_code="dirty_on_arrival",
                display_short="грязно при заселении",
                long_hint="Жалобы на грязь сразу при заезде: пыль, мусор, неубранные поверхности, пятна, волосы."
            ),
            "leftover_trash_previous_guest": AspectMeta(
                aspect_code="leftover_trash_previous_guest",
                display_short="следы прошлых гостей",
                long_hint="Гости обнаружили мусор, использованные полотенца, бутылки или другие следы предыдущих постояльцев."
            ),

            "access_smooth": AspectMeta(
                aspect_code="access_smooth",
                display_short="удобный доступ",
                long_hint="Легко попасть внутрь здания и в номер: понятный вход, код/карта работают без проблем."
            ),
            "door_code_worked": AspectMeta(
                aspect_code="door_code_worked",
                display_short="код/карта работают",
                long_hint="Электронный доступ (код от двери, ключ-карта) сработал сразу, без сбоев."
            ),
            "tech_access_issue": AspectMeta(
                aspect_code="tech_access_issue",
                display_short="проблема с доступом",
                long_hint="Гости не могли попасть внутрь из-за проблем с кодом/картой/замком или дверь просто не открывалась."
            ),
            "entrance_hard_to_find": AspectMeta(
                aspect_code="entrance_hard_to_find",
                display_short="сложно найти вход",
                long_hint="Описывают, что трудно понять, куда заходить, непонятная навигация, не сразу нашли нужную дверь/подъезд."
            ),
            "no_elevator_baggage_issue": AspectMeta(
                aspect_code="no_elevator_baggage_issue",
                display_short="нет лифта, тяжело с багажом",
                long_hint="Жалобы, что лифт не работал или его нет, и чемоданы пришлось тащить по лестнице."
            ),

            "payment_clear": AspectMeta(
                aspect_code="payment_clear",
                display_short="понятная оплата",
                long_hint="Гости пишут, что оплату, налоги и депозиты объяснили прозрачно, всё по чеку, без сюрпризов."
            ),
            "deposit_clear": AspectMeta(
                aspect_code="deposit_clear",
                display_short="депозит объяснили",
                long_hint="Размер и условия депозита были заранее озвучены и понятны гостю."
            ),
            "docs_provided": AspectMeta(
                aspect_code="docs_provided",
                display_short="дали документы/чеки",
                long_hint="Гостям предоставили все нужные документы: чек, счёт, отчётные бумаги."
            ),
            "no_hidden_fees": AspectMeta(
                aspect_code="no_hidden_fees",
                display_short="без скрытых платежей",
                long_hint="Подчёркивают, что не было неожиданных доплат, ничего лишнего не списали."
            ),
            "payment_confusing": AspectMeta(
                aspect_code="payment_confusing",
                display_short="путаница с оплатой",
                long_hint="Жалобы на то, что не объяснили налоги, непонятные суммы, запутанный расчёт."
            ),
            "unexpected_charge": AspectMeta(
                aspect_code="unexpected_charge",
                display_short="неожиданный платеж",
                long_hint="Гости столкнулись с незапланированными списаниями, дополнительными блокировками средств или внезапным депозитом."
            ),
            "hidden_fees": AspectMeta(
                aspect_code="hidden_fees",
                display_short="скрытые платежи",
                long_hint="Гости считают, что им попытались выставить неоговорённые заранее суммы."
            ),
            "deposit_problematic": AspectMeta(
                aspect_code="deposit_problematic",
                display_short="проблемы с депозитом",
                long_hint="Депозит взяли непредсказуемо, не объяснили условия или заблокировали деньги без предупреждения."
            ),
            "billing_mistake": AspectMeta(
                aspect_code="billing_mistake",
                display_short="ошибка в счёте",
                long_hint="Гости сообщают о неправильном счёте, неверных суммах или некорректных начислениях."
            ),
            "overcharge": AspectMeta(
                aspect_code="overcharge",
                display_short="перевыставили / переплата",
                long_hint="Гости уверены, что с них попытались взять больше, чем положено, или списали завышенную сумму."
            ),

            "instructions_clear": AspectMeta(
                aspect_code="instructions_clear",
                display_short="понятные инструкции",
                long_hint="Гости получили подробные и понятные инструкции по заселению, входу, использованию помещения."
            ),
            "self_checkin_easy": AspectMeta(
                aspect_code="self_checkin_easy",
                display_short="самостоятельное заселение удобное",
                long_hint="Отмечают, что self check-in был простым и бесконтактным, без лишних шагов."
            ),
            "wifi_info_given": AspectMeta(
                aspect_code="wifi_info_given",
                display_short="сразу дали Wi-Fi",
                long_hint="Пароль от Wi-Fi и доступ к сети дали сразу, не пришлось спрашивать отдельно."
            ),
            "instructions_confusing": AspectMeta(
                aspect_code="instructions_confusing",
                display_short="неясные инструкции",
                long_hint="Инструкции по заселению/доступу/правилам были неполные, путаные или противоречивые."
            ),
            "late_access_code": AspectMeta(
                aspect_code="late_access_code",
                display_short="код доступа прислали поздно",
                long_hint="Код от двери или инструкции прислали слишком поздно, пришлось ждать перед входом."
            ),
            "wifi_info_missing": AspectMeta(
                aspect_code="wifi_info_missing",
                display_short="не дали Wi-Fi",
                long_hint="Гости жалуются, что им не сообщили пароль от Wi-Fi или вообще не дали доступ к сети."
            ),
            "had_to_figure_out": AspectMeta(
                aspect_code="had_to_figure_out",
                display_short="пришлось разбираться самим",
                long_hint="Гости описывают, что никто ничего толком не объяснил, пришлось методом тыка разбираться с доступом и проживанием."
            ),

            "support_during_stay_good": AspectMeta(
                aspect_code="support_during_stay_good",
                display_short="помогали во время проживания",
                long_hint="Персонал был на связи и реально помогал в процессе проживания (донесли, починили, заменили)."
            ),
            "issue_fixed_immediately": AspectMeta(
                aspect_code="issue_fixed_immediately",
                display_short="проблему устранили сразу",
                long_hint="Просьбы гостей выполняли моментально: что-то сломалось — сразу починили или заменили."
            ),
            "support_during_stay_slow": AspectMeta(
                aspect_code="support_during_stay_slow",
                display_short="медленная реакция во время проживания",
                long_hint="Гости жалуются, что помощь приходилось ждать долго, никто не приходил сразу."
            ),
            "support_ignored": AspectMeta(
                aspect_code="support_ignored",
                display_short="запросы игнорировали",
                long_hint="Персонал не реагировал на просьбы гостей, приходилось напоминать несколько раз."
            ),
            "promised_not_done": AspectMeta(
                aspect_code="promised_not_done",
                display_short="обещали и не сделали",
                long_hint="Гости пишут, что им пообещали решить вопрос, но так и не сделали ничего."
            ),

            "checkout_easy": AspectMeta(
                aspect_code="checkout_easy",
                display_short="удобный выезд",
                long_hint="Оформление выезда прошло спокойно и без сложностей, сдали ключи и уехали без задержек."
            ),
            "checkout_fast": AspectMeta(
                aspect_code="checkout_fast",
                display_short="быстрый выезд",
                long_hint="Чек-аут занял буквально минуту-две, оформили моментально."
            ),
            "checkout_slow": AspectMeta(
                aspect_code="checkout_slow",
                display_short="медленный выезд",
                long_hint="Гости жалуются, что выписывание заняло слишком много времени, пришлось ждать."
            ),
            "deposit_return_issue": AspectMeta(
                aspect_code="deposit_return_issue",
                display_short="не вернули депозит сразу",
                long_hint="Пишут, что при выселении деньги не вернули сразу или с возвратом залога возникли сложности."
            ),
            "checkout_no_staff": AspectMeta(
                aspect_code="checkout_no_staff",
                display_short="некому оформить выезд",
                long_hint="При отъезде никого не было на ресепшене, некуда сдать ключ, пришлось выкручиваться самим."
            ),
            # =========================
            # cleanliness
            # =========================

            "fresh_bedding": AspectMeta(
                aspect_code="fresh_bedding",
                display_short="свежее бельё",
                long_hint="При заезде постель была свежая и чистая: чистые простыни, наволочки без запахов и следов использования."
            ),
            "no_dust_surfaces": AspectMeta(
                aspect_code="no_dust_surfaces",
                display_short="без пыли",
                long_hint="Гости отмечают, что на поверхностях не было пыли: полки, тумбы, столы чистые."
            ),
            "floor_clean": AspectMeta(
                aspect_code="floor_clean",
                display_short="чистый пол",
                long_hint="Пол чистый, без крошек, пятен или липких участков к моменту заселения."
            ),
            "dusty_surfaces": AspectMeta(
                aspect_code="dusty_surfaces",
                display_short="пыли много",
                long_hint="Жалобы на пыль и грязный налёт на поверхностях, подоконниках, полках, столах."
            ),
            "sticky_surfaces": AspectMeta(
                aspect_code="sticky_surfaces",
                display_short="липкие поверхности",
                long_hint="Гости описывают липкие полы, липкие столы — ощущение, что не протёрли после предыдущих гостей."
            ),
            "stained_bedding": AspectMeta(
                aspect_code="stained_bedding",
                display_short="пятна на постели",
                long_hint="На простынях/пододеяльнике были пятна, следы использования, неприятный вид."
            ),
            "hair_on_bed": AspectMeta(
                aspect_code="hair_on_bed",
                display_short="волосы на постели",
                long_hint="Гости находят волосы на кровати или подушке сразу при заселении."
            ),
            "used_towels_left": AspectMeta(
                aspect_code="used_towels_left",
                display_short="старые полотенца остались",
                long_hint="В номере остались использованные полотенца от предыдущих гостей, их не убрали."
            ),
            "crumbs_left": AspectMeta(
                aspect_code="crumbs_left",
                display_short="крошки и мусор остались",
                long_hint="На столах/полу остались крошки, упаковки, другой мелкий мусор от прошлых гостей."
            ),

            "bathroom_clean_on_arrival": AspectMeta(
                aspect_code="bathroom_clean_on_arrival",
                display_short="чистый санузел при заезде",
                long_hint="Санузел/душ/раковина были вымыты, без следов использования и неприятного осадка."
            ),
            "no_mold_visible": AspectMeta(
                aspect_code="no_mold_visible",
                display_short="без плесени",
                long_hint="Гости отмечают отсутствие плесени и грибка в душе, на плитке и в швах."
            ),
            "sink_clean": AspectMeta(
                aspect_code="sink_clean",
                display_short="чистая раковина",
                long_hint="Раковина без волос, известкового налёта и следов грязи на момент заезда."
            ),
            "shower_clean": AspectMeta(
                aspect_code="shower_clean",
                display_short="чистый душ",
                long_hint="Душевая зона чистая: нет волос, налёта, ржавчины."
            ),
            "bathroom_dirty_on_arrival": AspectMeta(
                aspect_code="bathroom_dirty_on_arrival",
                display_short="грязный санузел при заезде",
                long_hint="Жалобы на грязный туалет/раковину/душ сразу при заселении: следы, волосы, несмытый унитаз."
            ),
            "hair_in_shower": AspectMeta(
                aspect_code="hair_in_shower",
                display_short="волосы в душе",
                long_hint="Гости обнаруживают волосы в душе или в сливе душевой, что воспринимается как неубрано."
            ),
            "hair_in_sink": AspectMeta(
                aspect_code="hair_in_sink",
                display_short="волосы в раковине",
                long_hint="Гости жалуются на волосы/грязь, оставшиеся в раковине после предыдущих гостей."
            ),
            "mold_in_shower": AspectMeta(
                aspect_code="mold_in_shower",
                display_short="плесень в душе",
                long_hint="Пишут, что в душе, на швах плитки или у слива есть плесень/чёрные пятна."
            ),
            "limescale_stains": AspectMeta(
                aspect_code="limescale_stains",
                display_short="известковый налёт / ржавчина",
                long_hint="Гости замечают следы налёта, ржавчины или минеральные отложения на сантехнике."
            ),
            "sewage_smell_bathroom": AspectMeta(
                aspect_code="sewage_smell_bathroom",
                display_short="запах канализации в ванной",
                long_hint="Жалобы на запах канализации/туалета из слива санузла."
            ),

            "housekeeping_regular": AspectMeta(
                aspect_code="housekeeping_regular",
                display_short="убирали регулярно",
                long_hint="Отмечают, что уборка проводилась во время проживания: заходили убирать, поддерживали чистоту."
            ),
            "trash_taken_out": AspectMeta(
                aspect_code="trash_taken_out",
                display_short="выносили мусор",
                long_hint="Персонал забирал мусор из номера, не приходилось самим выносить пакеты."
            ),
            "bed_made": AspectMeta(
                aspect_code="bed_made",
                display_short="застилали кровать",
                long_hint="Кровать регулярно заправляли, визуально поддерживали порядок в комнате."
            ),
            "housekeeping_missed": AspectMeta(
                aspect_code="housekeeping_missed",
                display_short="уборки не было",
                long_hint="Гости жалуются, что за время проживания никто не пришёл убирать номер."
            ),
            "trash_not_taken": AspectMeta(
                aspect_code="trash_not_taken",
                display_short="мусор не забирали",
                long_hint="Мусор копился: корзины не опустошали, пакеты не забирали."
            ),
            "bed_not_made": AspectMeta(
                aspect_code="bed_not_made",
                display_short="не заправляли кровать",
                long_hint="Гости отмечают, что кровать так и оставалась неубранной между днями проживания."
            ),
            "had_to_request_cleaning": AspectMeta(
                aspect_code="had_to_request_cleaning",
                display_short="уборку пришлось просить",
                long_hint="Чтобы убрать номер / вынести мусор / поменять полотенца, приходилось отдельно просить или напоминать."
            ),
            "dirt_accumulated": AspectMeta(
                aspect_code="dirt_accumulated",
                display_short="грязь накапливалась",
                long_hint="Во время проживания становилось всё грязнее, и это не убирали."
            ),

            "towels_changed": AspectMeta(
                aspect_code="towels_changed",
                display_short="полотенца меняли",
                long_hint="Полотенца регулярно заменяли на чистые по запросу или сами по себе."
            ),
            "fresh_towels_fast": AspectMeta(
                aspect_code="fresh_towels_fast",
                display_short="чистые полотенца сразу",
                long_hint="Гости пишут, что по просьбе быстро принесли свежие полотенца."
            ),
            "linen_changed": AspectMeta(
                aspect_code="linen_changed",
                display_short="меняли постельное бельё",
                long_hint="Постель сменили на чистую во время проживания."
            ),
            "amenities_restocked": AspectMeta(
                aspect_code="amenities_restocked",
                display_short="пополняли принадлежности",
                long_hint="Регулярно пополняли расходники: воду, бумагу, мыло, шампунь."
            ),
            "towels_dirty": AspectMeta(
                aspect_code="towels_dirty",
                display_short="грязные полотенца",
                long_hint="Жалобы, что выдали грязные полотенца или оставили использованные чужие."
            ),
            "towels_stained": AspectMeta(
                aspect_code="towels_stained",
                display_short="пятна на полотенцах",
                long_hint="Гости упоминают пятна, следы косметики/грязи на полотенцах."
            ),
            "towels_smell": AspectMeta(
                aspect_code="towels_smell",
                display_short="полотенца неприятно пахнут",
                long_hint="Полотенца имеют затхлый или неприятный запах, ощущаются б/у."
            ),
            "towels_not_changed": AspectMeta(
                aspect_code="towels_not_changed",
                display_short="полотенца не меняли",
                long_hint="Полотенца не заменяли даже после просьбы, приходилось пользоваться старыми."
            ),
            "linen_not_changed": AspectMeta(
                aspect_code="linen_not_changed",
                display_short="бельё не меняли",
                long_hint="Гости жалуются, что постельное бельё так и не сменили за всё время проживания."
            ),
            "no_restock": AspectMeta(
                aspect_code="no_restock",
                display_short="не пополняли расходники",
                long_hint="Не пополняли туалетную бумагу, мыло, воду, шампунь и т.д."
            ),

            "smell_of_smoke": AspectMeta(
                aspect_code="smell_of_smoke",
                display_short="запах сигарет",
                long_hint="В номере/помещении чувствуется запах табака/сигаретного дыма."
            ),
            "sewage_smell": AspectMeta(
                aspect_code="sewage_smell",
                display_short="запах канализации",
                long_hint="Гости чувствуют запах канализации/сточных вод (обычно из санузла или слива)."
            ),
            "musty_smell": AspectMeta(
                aspect_code="musty_smell",
                display_short="сырой / затхлый запах",
                long_hint="Запах сырости, плесени, влажности; иногда описывают как 'запах подвала'."
            ),
            "chemical_smell_strong": AspectMeta(
                aspect_code="chemical_smell_strong",
                display_short="запах химии",
                long_hint="Слишком резкий запах хлорки/средств для уборки, мешающий находиться внутри."
            ),
            "no_bad_smell": AspectMeta(
                aspect_code="no_bad_smell",
                display_short="без неприятных запахов",
                long_hint="Гости подчёркивают отсутствие любых посторонних запахов."
            ),
            "fresh_smell": AspectMeta(
                aspect_code="fresh_smell",
                display_short="свежий запах",
                long_hint="Отмечают, что в помещении пахнет свежо и приятно (чистый, 'свежий' воздух)."
            ),

            "hallway_clean": AspectMeta(
                aspect_code="hallway_clean",
                display_short="чистый коридор",
                long_hint="Общие зоны, коридоры, лестницы выглядят аккуратно и убрано."
            ),
            "common_areas_clean": AspectMeta(
                aspect_code="common_areas_clean",
                display_short="чистые общие зоны",
                long_hint="Гости упоминают чистый холл, входную группу, лифт, общие пространства."
            ),
            "hallway_dirty": AspectMeta(
                aspect_code="hallway_dirty",
                display_short="грязный коридор/подъезд",
                long_hint="Коридоры, подъезд или лестница выглядят неухоженно: грязно, пыльно, мусор на полу."
            ),
            "elevator_dirty": AspectMeta(
                aspect_code="elevator_dirty",
                display_short="грязный лифт",
                long_hint="Гости жалуются, что лифт грязный, липкий, с неприятным запахом."
            ),
            "hallway_bad_smell": AspectMeta(
                aspect_code="hallway_bad_smell",
                display_short="запах в коридоре",
                long_hint="В коридоре/подъезде неприятный запах (сигареты, канализация, затхлость)."
            ),
            "entrance_feels_unsafe": AspectMeta(
                aspect_code="entrance_feels_unsafe",
                display_short="вход выглядит небезопасно",
                long_hint="Гости пишут, что подъезд/вход выглядит страшно, грязно, 'стрёмно', им некомфортно туда заходить."
            ),
            # =========================
            # comfort
            # =========================

            "room_well_equipped": AspectMeta(
                aspect_code="room_well_equipped",
                display_short="номер хорошо оснащён",
                long_hint="Гости пишут, что в номере есть всё необходимое: чайник, посуда, холодильник, фен, розетки, рабочее место и т.д."
            ),
            "kettle_available": AspectMeta(
                aspect_code="kettle_available",
                display_short="есть чайник",
                long_hint="Упоминают наличие чайника (иногда с чашками и чаем/кофе)."
            ),
            "fridge_available": AspectMeta(
                aspect_code="fridge_available",
                display_short="есть холодильник",
                long_hint="Гости отмечают наличие рабочего холодильника или минибара."
            ),
            "hairdryer_available": AspectMeta(
                aspect_code="hairdryer_available",
                display_short="есть фен",
                long_hint="В отзыве подчёркивают, что в номере есть фен, не пришлось просить отдельно."
            ),
            "sockets_enough": AspectMeta(
                aspect_code="sockets_enough",
                display_short="достаточно розеток",
                long_hint="Гости довольны количеством и расположением розеток, особенно у кровати / рабочего места."
            ),
            "workspace_available": AspectMeta(
                aspect_code="workspace_available",
                display_short="есть где работать",
                long_hint="Есть нормальный стол/поверхность и стул, удобно работать с ноутбуком."
            ),
            "luggage_space_ok": AspectMeta(
                aspect_code="luggage_space_ok",
                display_short="есть место под багаж",
                long_hint="Есть куда разложить вещи и развернуть чемоданы, удобные поверхности/полки."
            ),
            "kettle_missing": AspectMeta(
                aspect_code="kettle_missing",
                display_short="нет чайника",
                long_hint="Гости жалуются, что чайника нет, хотя он ожидался или был бы полезен."
            ),
            "fridge_missing": AspectMeta(
                aspect_code="fridge_missing",
                display_short="нет холодильника",
                long_hint="Отмечают отсутствие холодильника, что создало дискомфорт (негде хранить еду/детское питание и т.п.)."
            ),
            "hairdryer_missing": AspectMeta(
                aspect_code="hairdryer_missing",
                display_short="нет фена",
                long_hint="Жалуются, что фена не было в номере и пришлось обходиться без него или просить отдельно."
            ),
            "sockets_not_enough": AspectMeta(
                aspect_code="sockets_not_enough",
                display_short="не хватает розеток",
                long_hint="Мало розеток или они далеко от кровати/стола; неудобно заряжать устройства."
            ),
            "no_workspace": AspectMeta(
                aspect_code="no_workspace",
                display_short="нет рабочего места",
                long_hint="Гости говорят, что негде сесть и поработать: нет стола, стул неудобный, поверхность не подходит."
            ),
            "no_luggage_space": AspectMeta(
                aspect_code="no_luggage_space",
                display_short="некуда разложить чемодан",
                long_hint="Жалуются, что поставить или развернуть чемодан негде — слишком мало поверхности / нет подставки."
            ),

            "bed_comfy": AspectMeta(
                aspect_code="bed_comfy",
                display_short="удобная кровать",
                long_hint="Гости подчёркивают, что кровать удобная, на ней приятно спать."
            ),
            "mattress_comfy": AspectMeta(
                aspect_code="mattress_comfy",
                display_short="удобный матрас",
                long_hint="Отмечают качественный/комфортный матрас, правильной жёсткости."
            ),
            "pillow_comfy": AspectMeta(
                aspect_code="pillow_comfy",
                display_short="удобные подушки",
                long_hint="Подушки понравились по высоте/жёсткости, способствовали хорошему сну."
            ),
            "slept_well": AspectMeta(
                aspect_code="slept_well",
                display_short="хорошо спалось",
                long_hint="Гости пишут, что отлично выспались, сон был комфортным."
            ),
            "bed_uncomfortable": AspectMeta(
                aspect_code="bed_uncomfortable",
                display_short="неудобная кровать",
                long_hint="Жалобы, что кровать жёсткая/мягкая/узкая/скрипит и на ней неудобно спать."
            ),
            "mattress_too_soft": AspectMeta(
                aspect_code="mattress_too_soft",
                display_short="матрас слишком мягкий",
                long_hint="Матрас проваливается, нет поддержки спины."
            ),
            "mattress_too_hard": AspectMeta(
                aspect_code="mattress_too_hard",
                display_short="матрас слишком жёсткий",
                long_hint="Матрас жёсткий до дискомфорта, тяжело спать."
            ),
            "mattress_sagging": AspectMeta(
                aspect_code="mattress_sagging",
                display_short="матрас продавлен",
                long_hint="Жалуются, что матрас 'убитый', с ямами или просевший."
            ),
            "bed_creaks": AspectMeta(
                aspect_code="bed_creaks",
                display_short="скрипучая кровать",
                long_hint="Кровать шумит/скрипит при движении, мешает спать."
            ),
            "pillow_uncomfortable": AspectMeta(
                aspect_code="pillow_uncomfortable",
                display_short="неудобные подушки",
                long_hint="Подушки описываются как неудобные, портящие качество сна."
            ),
            "pillow_too_hard": AspectMeta(
                aspect_code="pillow_too_hard",
                display_short="подушки слишком жёсткие",
                long_hint="Гости жалуются, что подушки слишком твёрдые."
            ),
            "pillow_too_high": AspectMeta(
                aspect_code="pillow_too_high",
                display_short="подушки слишком высокие",
                long_hint="Подушки слишком толстые/высокие, неудобно для шеи."
            ),

            "quiet_room": AspectMeta(
                aspect_code="quiet_room",
                display_short="тихий номер",
                long_hint="Отмечают тишину днём и ночью, можно отдохнуть без лишнего шума."
            ),
            "good_soundproofing": AspectMeta(
                aspect_code="good_soundproofing",
                display_short="хорошая звукоизоляция",
                long_hint="Гости не слышали соседей/коридор/улицу, стены глушат звук."
            ),
            "no_street_noise": AspectMeta(
                aspect_code="no_street_noise",
                display_short="не слышно улицу",
                long_hint="Шума машин/дороги/баров снаружи не слышно, даже если окна на улицу."
            ),
            "noisy_room": AspectMeta(
                aspect_code="noisy_room",
                display_short="шумный номер",
                long_hint="Гости жалуются, что в номере шумно: сложно расслабиться и поспать."
            ),
            "street_noise": AspectMeta(
                aspect_code="street_noise",
                display_short="шум с улицы",
                long_hint="Слышен уличный трафик, люди, музыка снаружи."
            ),
            "thin_walls": AspectMeta(
                aspect_code="thin_walls",
                display_short="тонкие стены",
                long_hint="Гости слышат разговоры/телевизор/шум соседей через стены."
            ),
            "hallway_noise": AspectMeta(
                aspect_code="hallway_noise",
                display_short="шум из коридора",
                long_hint="Слышно лифт, ресепшен, разговоры в коридоре или хлопающие двери."
            ),
            "night_noise_trouble_sleep": AspectMeta(
                aspect_code="night_noise_trouble_sleep",
                display_short="шум мешал спать",
                long_hint="Жалуются на ночной шум (громкая музыка, крики, тусовки), из-за которого было тяжело уснуть."
            ),

            "temp_comfortable": AspectMeta(
                aspect_code="temp_comfortable",
                display_short="комфортная температура",
                long_hint="В комнате не жарко и не холодно, приятно находиться и спать."
            ),
            "ventilation_ok": AspectMeta(
                aspect_code="ventilation_ok",
                display_short="нормально проветривается",
                long_hint="Хорошо проветривается / есть свежий воздух / можно открыть окна."
            ),
            "ac_working": AspectMeta(
                aspect_code="ac_working",
                display_short="кондиционер работает",
                long_hint="Кондиционер охлаждает/греет нормально, держит комфортную температуру."
            ),
            "heating_working": AspectMeta(
                aspect_code="heating_working",
                display_short="отопление работает",
                long_hint="В номере тепло за счёт отопления или обогревателя; не мёрзли."
            ),
            "too_hot_sleep_issue": AspectMeta(
                aspect_code="too_hot_sleep_issue",
                display_short="жарко, сложно спать",
                long_hint="Гости жалуются, что в комнате душно/жарко, тяжело уснуть."
            ),
            "too_cold": AspectMeta(
                aspect_code="too_cold",
                display_short="холодно в номере",
                long_hint="Гости пишут, что в помещении холодно, особенно ночью."
            ),
            "stuffy_no_air": AspectMeta(
                aspect_code="stuffy_no_air",
                display_short="душно, нет воздуха",
                long_hint="Ощущение духоты: нечем дышать, воздух тяжёлый."
            ),
            "no_ventilation": AspectMeta(
                aspect_code="no_ventilation",
                display_short="нет вентиляции",
                long_hint="Гости отмечают, что комната не проветривается, окна не открыть или притока свежего воздуха нет."
            ),
            "ac_not_working": AspectMeta(
                aspect_code="ac_not_working",
                display_short="кондиционер не работает",
                long_hint="Кондиционер не включался / не охлаждал / не охлаждал достаточно."
            ),
            "no_ac": AspectMeta(
                aspect_code="no_ac",
                display_short="нет кондиционера",
                long_hint="Гости жалуются на отсутствие кондиционера в жару."
            ),
            "heating_not_working": AspectMeta(
                aspect_code="heating_not_working",
                display_short="нет отопления",
                long_hint="Отопление не работало или батареи были холодные, приходилось мёрзнуть."
            ),
            "draft_window": AspectMeta(
                aspect_code="draft_window",
                display_short="сквозняк из окна",
                long_hint="Гости пишут про сильный холодный поток воздуха из окна/рам, который мешал комфорту."
            ),

            "room_spacious": AspectMeta(
                aspect_code="room_spacious",
                display_short="просторный номер",
                long_hint="Гости говорят, что номер большой, хватает места свободно двигаться и разложить вещи."
            ),
            "good_layout": AspectMeta(
                aspect_code="good_layout",
                display_short="удобная планировка",
                long_hint="Расстановка мебели удобная, всё логично организовано, ничего не мешает."
            ),
            "cozy_feel": AspectMeta(
                aspect_code="cozy_feel",
                display_short="уютный номер",
                long_hint="Номер воспринимается как уютный, тёплый, 'как дома'."
            ),
            "bright_room": AspectMeta(
                aspect_code="bright_room",
                display_short="светлый номер",
                long_hint="Много света, приятное освещение, много дневного света."
            ),
            "big_windows": AspectMeta(
                aspect_code="big_windows",
                display_short="большие окна",
                long_hint="Гости отмечают большие окна и хороший естественный свет."
            ),
            "room_small": AspectMeta(
                aspect_code="room_small",
                display_short="тесный номер",
                long_hint="Жалобы, что номер маленький, тесный, не развернуться."
            ),
            "no_space_for_luggage": AspectMeta(
                aspect_code="no_space_for_luggage",
                display_short="некуда поставить чемодан",
                long_hint="Гости пишут, что для чемодана нет места — его негде открыть/оставить."
            ),
            "dark_room": AspectMeta(
                aspect_code="dark_room",
                display_short="тёмный номер",
                long_hint="В номере мрачно, не хватает света, слабое освещение."
            ),
            "no_natural_light": AspectMeta(
                aspect_code="no_natural_light",
                display_short="нет дневного света",
                long_hint="Гости жалуются на отсутствие/почти отсутствие естественного освещения, маленькое окно или нет окна вообще."
            ),
            "gloomy_feel": AspectMeta(
                aspect_code="gloomy_feel",
                display_short="мрачная атмосфера в номере",
                long_hint="Номер давит, кажется мрачным, неуютным из-за темноты/тесноты/серой отделки."
            ),
            # =========================
            # tech_state
            # =========================

            "hot_water_ok": AspectMeta(
                aspect_code="hot_water_ok",
                display_short="горячая вода есть",
                long_hint="Гости пишут, что горячая вода была сразу и без перебоев, не приходилось ждать."
            ),
            "water_pressure_ok": AspectMeta(
                aspect_code="water_pressure_ok",
                display_short="нормальное давление воды",
                long_hint="Сильная/стабильная струя, комфортно пользоваться душем и раковиной."
            ),
            "shower_ok": AspectMeta(
                aspect_code="shower_ok",
                display_short="душ работает нормально",
                long_hint="Душ исправен, лейка держится, вода льётся равномерно."
            ),
            "no_leak": AspectMeta(
                aspect_code="no_leak",
                display_short="ничего не течёт",
                long_hint="Гости отмечают отсутствие протечек: кран не капает, нигде не подтекает."
            ),
            "no_hot_water": AspectMeta(
                aspect_code="no_hot_water",
                display_short="нет горячей воды",
                long_hint="Жалобы, что не было горячей воды (особенно утром) или она быстро заканчивалась."
            ),
            "weak_pressure": AspectMeta(
                aspect_code="weak_pressure",
                display_short="слабый напор",
                long_hint="Очень слабое давление воды: 'еле течёт', неудобно мыться/смывать."
            ),
            "shower_broken": AspectMeta(
                aspect_code="shower_broken",
                display_short="душ сломан",
                long_hint="Лейка/крепление душа сломаны или душ толком не работает."
            ),
            "leak_water": AspectMeta(
                aspect_code="leak_water",
                display_short="протечки воды",
                long_hint="Краны текут, что-то капает, вода сочится где не должна."
            ),
            "bathroom_flooding": AspectMeta(
                aspect_code="bathroom_flooding",
                display_short="вода на полу в ванной",
                long_hint="После душа вся ванная в воде / пол заливается."
            ),
            "drain_clogged": AspectMeta(
                aspect_code="drain_clogged",
                display_short="засор слива",
                long_hint="Слив в душе или раковине забит, вода уходит плохо или не уходит."
            ),
            "drain_smell": AspectMeta(
                aspect_code="drain_smell",
                display_short="запах из слива",
                long_hint="Гости жалуются на запах канализации/стоков из раковины или душевого слива."
            ),

            "ac_working_device": AspectMeta(
                aspect_code="ac_working_device",
                display_short="кондиционер исправен",
                long_hint="Кондиционер технически работает как устройство: включается, охлаждает/греет нормально."
            ),
            "heating_working_device": AspectMeta(
                aspect_code="heating_working_device",
                display_short="отопление исправно",
                long_hint="Отопление/обогреватель физически работает, в номере тепло."
            ),
            "appliances_ok": AspectMeta(
                aspect_code="appliances_ok",
                display_short="всё оборудование работает",
                long_hint="Гости отмечают, что техника и оснащение номера исправны: ничего не ломалось."
            ),
            "tv_working": AspectMeta(
                aspect_code="tv_working",
                display_short="телевизор работает",
                long_hint="Телевизор включается, есть каналы/контент, всё ок со звуком и картинкой."
            ),
            "fridge_working": AspectMeta(
                aspect_code="fridge_working",
                display_short="холодильник работает",
                long_hint="Холодильник/минибар охлаждает как надо, нет замечаний."
            ),
            "kettle_working": AspectMeta(
                aspect_code="kettle_working",
                display_short="чайник работает",
                long_hint="Чайник/кипятильник исправен, можно вскипятить воду без проблем."
            ),
            "door_secure": AspectMeta(
                aspect_code="door_secure",
                display_short="дверь нормально закрывается",
                long_hint="Дверь плотно закрывается, замок работает, гости чувствуют безопасность вещей в номере."
            ),
            "ac_broken": AspectMeta(
                aspect_code="ac_broken",
                display_short="кондиционер не работает",
                long_hint="Жалобы на сломанный кондиционер: не включается, не охлаждает или очень слабый."
            ),
            "heating_broken": AspectMeta(
                aspect_code="heating_broken",
                display_short="отопление не работает",
                long_hint="Отопление не срабатывает / батареи холодные / в номере мёрзли из-за этого."
            ),
            "tv_broken": AspectMeta(
                aspect_code="tv_broken",
                display_short="не работает ТВ",
                long_hint="Телевизор не включается, нет каналов, экран/пульт неисправен."
            ),
            "fridge_broken": AspectMeta(
                aspect_code="fridge_broken",
                display_short="холодильник не работает",
                long_hint="Холодильник не охлаждает или полностью нерабочий."
            ),
            "kettle_broken": AspectMeta(
                aspect_code="kettle_broken",
                display_short="чайник сломан",
                long_hint="Чайник не греет воду, течёт или искрит — использовать нельзя."
            ),
            "socket_danger": AspectMeta(
                aspect_code="socket_danger",
                display_short="опасная розетка",
                long_hint="Гости отмечают, что розетки болтаются, искрят или выглядят небезопасно."
            ),
            "door_not_closing": AspectMeta(
                aspect_code="door_not_closing",
                display_short="дверь плохо закрывается",
                long_hint="Входная дверь неплотно закрывается или не прижимается нормально, можно не до конца закрыть."
            ),
            "lock_broken": AspectMeta(
                aspect_code="lock_broken",
                display_short="проблема с замком",
                long_hint="Замок клинит, не закрывается или вообще не работает; сложно запереться."
            ),
            "furniture_broken": AspectMeta(
                aspect_code="furniture_broken",
                display_short="сломанная мебель",
                long_hint="Гости жалуются на поломанный шкаф, шатающийся стол, отваливающиеся дверцы и т.п."
            ),
            "room_worn_out": AspectMeta(
                aspect_code="room_worn_out",
                display_short="номер уставший",
                long_hint="Общее состояние номера уставшее: облезлые стены, старая мебель, чувствуется, что 'требует ремонта'."
            ),

            "wifi_fast": AspectMeta(
                aspect_code="wifi_fast",
                display_short="быстрый Wi-Fi",
                long_hint="Гости отмечают высокую скорость Wi-Fi, комфортно серфить/смотреть видео."
            ),
            "internet_stable": AspectMeta(
                aspect_code="internet_stable",
                display_short="стабильный интернет",
                long_hint="Интернет не обрывается, подключение держится без лагов."
            ),
            "good_for_work": AspectMeta(
                aspect_code="good_for_work",
                display_short="интернет подходит для работы",
                long_hint="Можно полноценно работать удалённо — достаточно скорости и стабильности."
            ),
            "wifi_down": AspectMeta(
                aspect_code="wifi_down",
                display_short="Wi-Fi не работал",
                long_hint="Гости пишут, что Wi-Fi отсутствовал или вообще не удавалось подключиться."
            ),
            "wifi_slow": AspectMeta(
                aspect_code="wifi_slow",
                display_short="медленный интернет",
                long_hint="Очень низкая скорость Wi-Fi, страницы грузятся с трудом, невозможно нормально пользоваться."
            ),
            "wifi_unstable": AspectMeta(
                aspect_code="wifi_unstable",
                display_short="Wi-Fi отваливается",
                long_hint="Соединение постоянно рвётся, теряется сигнал, приходится переподключаться."
            ),
            "wifi_hard_to_connect": AspectMeta(
                aspect_code="wifi_hard_to_connect",
                display_short="сложно подключиться к Wi-Fi",
                long_hint="Гости жалуются, что пароль не подходит, сеть не принимает, процесс подключения мучительный."
            ),
            "internet_not_suitable_for_work": AspectMeta(
                aspect_code="internet_not_suitable_for_work",
                display_short="интернет не для удалёнки",
                long_hint="Из-за скорости/нестабильности невозможно было работать удалённо (звонки, митинги, VPN)."
            ),

            "ac_noisy": AspectMeta(
                aspect_code="ac_noisy",
                display_short="шумный кондиционер",
                long_hint="Кондиционер громко гудит/жужжит, мешает отдыху или сну."
            ),
            "fridge_noisy": AspectMeta(
                aspect_code="fridge_noisy",
                display_short="шумный холодильник",
                long_hint="Холодильник громко гудит, трещит или вибрирует, особенно ночью."
            ),
            "pipes_noise": AspectMeta(
                aspect_code="pipes_noise",
                display_short="шум труб",
                long_hint="Гости слышат шум/гул/стук в трубах или стояке."
            ),
            "ventilation_noisy": AspectMeta(
                aspect_code="ventilation_noisy",
                display_short="шумная вентиляция",
                long_hint="Вентилятор/вентиляция гудит, свистит, шумит заметно."
            ),
            "night_mechanical_hum": AspectMeta(
                aspect_code="night_mechanical_hum",
                display_short="гул техники ночью",
                long_hint="Системы (кондиционер, холодильник, вентиляторы и т.п.) издают постоянный гул ночью."
            ),
            "tech_noise_sleep_issue": AspectMeta(
                aspect_code="tech_noise_sleep_issue",
                display_short="шум техники мешал спать",
                long_hint="Шум оборудования мешал заснуть или просыпались из-за звуков устройств."
            ),
            "ac_quiet": AspectMeta(
                aspect_code="ac_quiet",
                display_short="тихий кондиционер",
                long_hint="Кондиционер работает почти бесшумно, не мешает сну."
            ),
            "fridge_quiet": AspectMeta(
                aspect_code="fridge_quiet",
                display_short="тихий холодильник",
                long_hint="Холодильник не шумит, не вибрирует, не мешает отдыхать."
            ),
            "no_tech_noise_night": AspectMeta(
                aspect_code="no_tech_noise_night",
                display_short="тихо от техники ночью",
                long_hint="Гости подчёркивают, что ночью не было жужжания приборов, шума труб или вентиляции."
            ),

            "elevator_working": AspectMeta(
                aspect_code="elevator_working",
                display_short="лифт работает",
                long_hint="Лифт в рабочем состоянии, можно комфортно пользоваться."
            ),
            "luggage_easy": AspectMeta(
                aspect_code="luggage_easy",
                display_short="удобно с багажом",
                long_hint="Было легко подняться с чемоданами: лифт работает или доступ хорошо организован."
            ),
            "elevator_broken": AspectMeta(
                aspect_code="elevator_broken",
                display_short="лифт не работает",
                long_hint="Лифт был сломан/отключён, приходилось ходить пешком."
            ),
            "elevator_stuck": AspectMeta(
                aspect_code="elevator_stuck",
                display_short="застряли в лифте",
                long_hint="Гости пишут, что лифт завис/заело внутри, был неприятный опыт."
            ),
            "no_elevator_heavy_bags": AspectMeta(
                aspect_code="no_elevator_heavy_bags",
                display_short="без лифта тяжело с чемоданами",
                long_hint="Не было лифта или он не работал, чемоданы пришлось тащить по лестнице, это было тяжело."
            ),

            "felt_safe": AspectMeta(
                aspect_code="felt_safe",
                display_short="чувствовали себя в безопасности",
                long_hint="Гости отмечают, что дверь хорошо закрывается и они спокойно оставляли вещи в номере."
            ),
            "felt_unsafe": AspectMeta(
                aspect_code="felt_unsafe",
                display_short="не чувствовали безопасность",
                long_hint="Гости переживали за вещи или за личную безопасность из-за двери/замка."
            ),
            # =========================
            # breakfast
            # =========================

            "breakfast_tasty": AspectMeta(
                aspect_code="breakfast_tasty",
                display_short="вкусный завтрак",
                long_hint="Гости пишут, что завтрак вкусный, еда нравится, блюда приготовлены хорошо."
            ),
            "food_fresh": AspectMeta(
                aspect_code="food_fresh",
                display_short="свежие продукты",
                long_hint="Отмечают свежесть блюд и ингредиентов, нет ощущения 'вчерашнего'."
            ),
            "food_hot_served_hot": AspectMeta(
                aspect_code="food_hot_served_hot",
                display_short="горячее — горячее",
                long_hint="Горячие блюда реально подаются горячими, не остывшие."
            ),
            "coffee_good": AspectMeta(
                aspect_code="coffee_good",
                display_short="хороший кофе",
                long_hint="Гости выделяют кофе как вкусный/качественный, не 'порошковый'."
            ),
            "breakfast_bad_taste": AspectMeta(
                aspect_code="breakfast_bad_taste",
                display_short="невкусный завтрак",
                long_hint="Жалобы, что еда невкусная, пересоленная, пережаренная или недожаренная."
            ),
            "food_not_fresh": AspectMeta(
                aspect_code="food_not_fresh",
                display_short="несвежая еда",
                long_hint="Гости описывают блюда как несвежие, 'вчерашние', с неприятным вкусом."
            ),
            "food_cold": AspectMeta(
                aspect_code="food_cold",
                display_short="холодная еда",
                long_hint="Горячие блюда поданы остывшими: холодные яйца, холодные горячие блюда."
            ),
            "coffee_bad": AspectMeta(
                aspect_code="coffee_bad",
                display_short="плохой кофе",
                long_hint="Жалуются, что кофе невкусный, совсем плохого качества или только растворимый."
            ),

            "breakfast_variety_good": AspectMeta(
                aspect_code="breakfast_variety_good",
                display_short="большой выбор на завтраке",
                long_hint="Гости отмечают разнообразие блюд, много позиций, есть из чего выбрать."
            ),
            "buffet_rich": AspectMeta(
                aspect_code="buffet_rich",
                display_short="богатый шведский стол",
                long_hint="Отмечают, что шведский стол 'насыщенный': всего много, постоянно подают."
            ),
            "fresh_fruit_available": AspectMeta(
                aspect_code="fresh_fruit_available",
                display_short="свежие фрукты",
                long_hint="Гости упоминают наличие свежих фруктов/овощей/сырых нарезок и т.д."
            ),
            "pastries_available": AspectMeta(
                aspect_code="pastries_available",
                display_short="выпечка / сладкое есть",
                long_hint="В отзывах хвалят круассаны, выпечку, десерты, сладкие варианты завтрака."
            ),
            "breakfast_variety_poor": AspectMeta(
                aspect_code="breakfast_variety_poor",
                display_short="маленький выбор на завтраке",
                long_hint="Жалуются, что выбор очень скудный, мало позиций."
            ),
            "breakfast_repetitive": AspectMeta(
                aspect_code="breakfast_repetitive",
                display_short="каждый день одно и то же",
                long_hint="Гости отмечают однотипный завтрак без изменений по дням."
            ),
            "hard_to_find_food": AspectMeta(
                aspect_code="hard_to_find_food",
                display_short="нечего поесть",
                long_hint="Гости пишут, что по факту не нашли ничего подходящего, тяжело выбрать еду."
            ),

            "breakfast_staff_friendly": AspectMeta(
                aspect_code="breakfast_staff_friendly",
                display_short="приветливый персонал на завтраке",
                long_hint="Отмечают дружелюбие и приветливость персонала в зоне завтрака."
            ),
            "breakfast_staff_attentive": AspectMeta(
                aspect_code="breakfast_staff_attentive",
                display_short="внимательный персонал на завтраке",
                long_hint="Сотрудники вежливые, отзывчивые, помогают гостям, реагируют быстро."
            ),
            "buffet_refilled_quickly": AspectMeta(
                aspect_code="buffet_refilled_quickly",
                display_short="быстро пополняют еду",
                long_hint="Пустые позиции на шведском столе оперативно пополняли, ничего не простаивало пустым."
            ),
            "tables_cleared_fast": AspectMeta(
                aspect_code="tables_cleared_fast",
                display_short="быстро убирают столы",
                long_hint="Столы очищают и протирают сразу после гостей, нет залежей грязной посуды."
            ),
            "breakfast_staff_rude": AspectMeta(
                aspect_code="breakfast_staff_rude",
                display_short="грубый персонал на завтраке",
                long_hint="Жалобы на невежливость/грубость сотрудников в зоне завтрака."
            ),
            "no_refill_food": AspectMeta(
                aspect_code="no_refill_food",
                display_short="не пополняли блюда",
                long_hint="Гости отмечают, что еду не доливали: лотки стоят пустыми, никто не подносит."
            ),
            "tables_left_dirty": AspectMeta(
                aspect_code="tables_left_dirty",
                display_short="грязные столы",
                long_hint="Гости жалуются, что столы оставались грязными, посуду не убирали."
            ),
            "ignored_requests": AspectMeta(
                aspect_code="ignored_requests",
                display_short="игнорировали просьбы на завтраке",
                long_hint="Чтобы попросить что-то (чашки, приборы, еду), приходилось повторять несколько раз, персонал игнорировал."
            ),

            "food_enough_for_all": AspectMeta(
                aspect_code="food_enough_for_all",
                display_short="еды хватает всем",
                long_hint="Отмечают, что еду постоянно подносили и хватало даже при большом потоке гостей."
            ),
            "kept_restocking": AspectMeta(
                aspect_code="kept_restocking",
                display_short="регулярно подносили еду",
                long_hint="Гости пишут, что позиции на буфете регулярно обновляли, ничего не заканчивалось надолго."
            ),
            "tables_available": AspectMeta(
                aspect_code="tables_available",
                display_short="было где сесть",
                long_hint="Гости без проблем находили свободный стол, не приходилось ждать место."
            ),
            "no_queue": AspectMeta(
                aspect_code="no_queue",
                display_short="без очередей",
                long_hint="Не было очередей ни за едой, ни за посадкой; спокойный поток гостей."
            ),
            "breakfast_flow_ok": AspectMeta(
                aspect_code="breakfast_flow_ok",
                display_short="хорошо организован завтрак",
                long_hint="Гости отмечают удобную организацию зоны завтрака — логично расставлено, не толкаются."
            ),
            "food_ran_out": AspectMeta(
                aspect_code="food_ran_out",
                display_short="еда быстро закончилась",
                long_hint="К моменту, когда гость пришёл (часто называют конкретное время), почти ничего не осталось."
            ),
            "not_restocked": AspectMeta(
                aspect_code="not_restocked",
                display_short="не пополняли буфет",
                long_hint="Пустые лотки долго стояли пустыми, еду не возвращали."
            ),
            "had_to_wait_food": AspectMeta(
                aspect_code="had_to_wait_food",
                display_short="пришлось ждать еду",
                long_hint="Гости ждали, пока вынесут новые блюда / доложат то, что закончилось."
            ),
            "no_tables_available": AspectMeta(
                aspect_code="no_tables_available",
                display_short="не было свободных столов",
                long_hint="Не найти место, где сесть и поесть; приходилось стоять или ждать, пока кто-то уйдёт."
            ),
            "long_queue": AspectMeta(
                aspect_code="long_queue",
                display_short="очередь на завтрак",
                long_hint="Гости отмечают большую очередь за едой или очередь, чтобы вообще попасть на завтрак."
            ),

            "breakfast_area_clean": AspectMeta(
                aspect_code="breakfast_area_clean",
                display_short="чистая зона завтрака",
                long_hint="Столовая/зона завтрака была аккуратной и чистой, без грязных поверхностей."
            ),
            "tables_cleaned_quickly": AspectMeta(
                aspect_code="tables_cleaned_quickly",
                display_short="быстро чистят столы",
                long_hint="Столы быстро протирали после гостей, не оставляли крошки и грязную посуду."
            ),
            "dirty_tables": AspectMeta(
                aspect_code="dirty_tables",
                display_short="грязные столы на завтраке",
                long_hint="Жалобы на то, что столы долго остаются липкими/в крошках, никто не протирает."
            ),
            "dirty_dishes_left": AspectMeta(
                aspect_code="dirty_dishes_left",
                display_short="грязная посуда на столах",
                long_hint="Гости пишут, что использованная посуда стоит на столах и её долго не убирают."
            ),
            "buffet_area_messy": AspectMeta(
                aspect_code="buffet_area_messy",
                display_short="грязно у раздачи",
                long_hint="Гости жалуются на беспорядок у линии буфета: крошки, пролитое, неопрятно разложено."
            ),

            # =========================
            # value
            # =========================

            "good_value": AspectMeta(
                aspect_code="good_value",
                display_short="хорошее соотношение цена/качество",
                long_hint="Гости считают, что за эту цену качество отличное; говорят 'очень выгодно', 'отличный value for money'."
            ),
            "worth_the_price": AspectMeta(
                aspect_code="worth_the_price",
                display_short="оправдывает цену",
                long_hint="Прямо пишут, что проживание стоит своих денег, цена честная."
            ),
            "affordable_for_level": AspectMeta(
                aspect_code="affordable_for_level",
                display_short="дёшево для такого уровня",
                long_hint="Гости удивлены, что за такой комфорт/локацию цена невысокая."
            ),
            "overpriced": AspectMeta(
                aspect_code="overpriced",
                display_short="слишком дорого",
                long_hint="Жалобы, что цена завышена относительно условий и качества."
            ),
            "not_worth_price": AspectMeta(
                aspect_code="not_worth_price",
                display_short="не стоит этих денег",
                long_hint="Гости считают, что качество не соответствует цене, money/value плохой."
            ),
            "expected_better_for_price": AspectMeta(
                aspect_code="expected_better_for_price",
                display_short="за такие деньги ожидали лучше",
                long_hint="Говорят, что за такую стоимость ожидали более высокий уровень сервиса/номера."
            ),

            "photos_misleading": AspectMeta(
                aspect_code="photos_misleading",
                display_short="в реальности хуже, чем на фото",
                long_hint="Гости пишут, что номер/объект выглядит хуже, чем на фотографиях в объявлении."
            ),
            "quality_below_expectation": AspectMeta(
                aspect_code="quality_below_expectation",
                display_short="качество ниже ожиданий",
                long_hint="Ожидали более высокий уровень по описанию/рейтингу, но получили менее качественный опыт."
            ),

            # =========================
            # location
            # =========================

            "great_location": AspectMeta(
                aspect_code="great_location",
                display_short="отличное расположение",
                long_hint="Гости хвалят локацию: удобно, всё рядом, хорошая точка для поездок."
            ),
            "central_convenient": AspectMeta(
                aspect_code="central_convenient",
                display_short="близко к центру",
                long_hint="Пишут, что локация фактически центральная или очень близко ко всем основным зонам/достопримечательностям."
            ),
            "near_transport": AspectMeta(
                aspect_code="near_transport",
                display_short="рядом транспорт",
                long_hint="Метро, остановки, транспортная доступность — в пешей доступности, легко добираться."
            ),
            "area_has_food_shops": AspectMeta(
                aspect_code="area_has_food_shops",
                display_short="рядом кафе и магазины",
                long_hint="Гости отмечают наличие вокруг супермаркетов, кафе, ресторанов, баров."
            ),
            "location_inconvenient": AspectMeta(
                aspect_code="location_inconvenient",
                display_short="неудобное расположение",
                long_hint="Локация неудобная, сложно добираться, нет ничего полезного рядом."
            ),
            "far_from_center": AspectMeta(
                aspect_code="far_from_center",
                display_short="далеко от центра",
                long_hint="Гости жалуются, что место находится далеко от ключевых точек города."
            ),
            "nothing_around": AspectMeta(
                aspect_code="nothing_around",
                display_short="ничего нет вокруг",
                long_hint="В округе нет кафе, магазинов, инфраструктуры — 'нечего делать рядом'."
            ),

            "area_safe": AspectMeta(
                aspect_code="area_safe",
                display_short="безопасный район",
                long_hint="Гости говорят, что район спокойный и безопасный, не страшно находиться снаружи."
            ),
            "area_quiet_at_night": AspectMeta(
                aspect_code="area_quiet_at_night",
                display_short="тихо ночью снаружи",
                long_hint="Отмечают, что район остаётся тихим ночью, нет уличного шума, можно спать с открытым окном."
            ),
            "entrance_clean": AspectMeta(
                aspect_code="entrance_clean",
                display_short="чистый вход/подъезд",
                long_hint="Пишут, что вход, подъезд или лестничная клетка выглядят чистыми и ухоженными."
            ),
            "area_unsafe": AspectMeta(
                aspect_code="area_unsafe",
                display_short="район небезопасный",
                long_hint="Гости говорят, что район 'стрёмный', неприятный, есть подозрительные люди."
            ),
            "uncomfortable_at_night": AspectMeta(
                aspect_code="uncomfortable_at_night",
                display_short="неуютно выходить вечером",
                long_hint="Гости не чувствуют себя комфортно на улице ночью, не хочется выходить."
            ),
            "entrance_dirty": AspectMeta(
                aspect_code="entrance_dirty",
                display_short="грязный подъезд",
                long_hint="Жалуются на грязный вход/подъезд, неприятный вид при заходе в здание."
            ),
            "people_loitering": AspectMeta(
                aspect_code="people_loitering",
                display_short="подозрительные люди у входа",
                long_hint="Гости отмечают пьяных/шумных/подозрительных людей у двери, 'тусовку у входа'."
            ),

            "easy_to_find": AspectMeta(
                aspect_code="easy_to_find",
                display_short="лёгко найти",
                long_hint="Гости пишут, что адрес/вход легко найти, проблем с навигацией не было."
            ),
            "clear_instructions": AspectMeta(
                aspect_code="clear_instructions",
                display_short="понятные инструкции по доступу",
                long_hint="Инструкции о том, как попасть внутрь, были простыми и понятными."
            ),
            "luggage_access_ok": AspectMeta(
                aspect_code="luggage_access_ok",
                display_short="удобно с багажом",
                long_hint="Гости отмечают, что с чемоданами было несложно зайти / подняться / добраться до номера."
            ),
            "hard_to_find_entrance": AspectMeta(
                aspect_code="hard_to_find_entrance",
                display_short="сложно найти вход",
                long_hint="Гости жалуются, что вход/дверь/подъезд плохо обозначен, тяжело обнаружить."
            ),
            "confusing_access": AspectMeta(
                aspect_code="confusing_access",
                display_short="запутанный вход",
                long_hint="Попасть внутрь оказалось сложно: непонятный домофон, сложная система доступа."
            ),
            "no_signage": AspectMeta(
                aspect_code="no_signage",
                display_short="нет вывески",
                long_hint="Гости отмечают, что нет нормальной таблички/указателя, непонятно, что это то самое место."
            ),
            "luggage_access_hard": AspectMeta(
                aspect_code="luggage_access_hard",
                display_short="тяжело с чемоданами",
                long_hint="Пишут, что занести багаж было сложно: много ступенек, узкие пролёты, нет лифта и т.д."
            ),

            # =========================
            # atmosphere
            # =========================

            "cozy_atmosphere": AspectMeta(
                aspect_code="cozy_atmosphere",
                display_short="уютная атмосфера",
                long_hint="Гости описывают атмосферу как тёплую, домашнюю, приятную, 'как дома'."
            ),
            "nice_design": AspectMeta(
                aspect_code="nice_design",
                display_short="красивый дизайн",
                long_hint="Хвалят интерьер, стиль, декор, визуально приятную обстановку."
            ),
            "good_vibe": AspectMeta(
                aspect_code="good_vibe",
                display_short="классная атмосфера",
                long_hint="Гости говорят про приятный вайб, общую приятную энергетику места, 'нам очень понравилось быть там'."
            ),
            "not_cozy": AspectMeta(
                aspect_code="not_cozy",
                display_short="неуютно",
                long_hint="Пишут, что атмосфера холодная, неуютная, 'не чувствуешь себя как дома'."
            ),
            # gloomy_feel уже задан выше в comfort (gloomy_feel), не переопределяем
            "dated_look": AspectMeta(
                aspect_code="dated_look",
                display_short="устаревший вид",
                long_hint="Интерьер выглядит старым, 'советский ремонт', всё визуально уставшее."
            ),
            "soulless_feel": AspectMeta(
                aspect_code="soulless_feel",
                display_short="без души",
                long_hint="Гости описывают место как безликое, холодное, 'неуютно и не по-домашнему'."
            ),

            "fresh_smell_common": AspectMeta(
                aspect_code="fresh_smell_common",
                display_short="приятно пахнет в общих зонах",
                long_hint="Гости отмечают приятный или нейтрально-свежий запах в коридоре/холле."
            ),
            # no_bad_smell уже задан ранее (no_bad_smell), не переопределяем
            "bad_smell_common": AspectMeta(
                aspect_code="bad_smell_common",
                display_short="запах в коридоре",
                long_hint="Жалобы на неприятный запах в коридоре/подъезде (канализация, табак, затхлость)."
            ),
            "cigarette_smell": AspectMeta(
                aspect_code="cigarette_smell",
                display_short="запах сигарет в общих зонах",
                long_hint="Гости пишут, что в коридорах пахнет сигаретами/дымом."
            ),
            # sewage_smell уже задан ранее (sewage_smell), не переопределяем
            # musty_smell уже задан ранее (musty_smell), не переопределяем

        }

        #######################################################################
        # 2.3.1 Словарь аспектов
        #
        # правила матчинга аспектов напрямую по регуляркам
        #######################################################################

        self.aspect_rules: Dict[str, AspectRule] = {
            "spir_friendly": AspectRule(
                aspect_code="spir_friendly",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bдружелюб", r"\bприветлив", r"\bрадушн", r"\bс улыбкой\b", r"\bтепло встретил"
                    ],
                    "en": [
                        r"\bfriendly staff\b", r"\bvery friendly\b", r"\bwelcoming\b", r"\bwelcomed us\b"
                    ],
                    "tr": [
                        r"\bgüler yüzlü\b", r"\bsıcak karşıladılar\b"
                    ],
                    "ar": [
                        r"\bموظفين لطيفين\b", r"\bاستقبال دافئ\b"
                    ],
                    "zh": [
                        r"服务很友好", r"前台很热情"
                    ],
                },
            ),
            
            "spir_polite": AspectRule(
                aspect_code="spir_polite",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bвежлив", r"\bдоброжелательн", r"\bочень вежлив", r"\bочень вежливы\b"
                    ],
                    "en": [
                        r"\bpolite\b", r"\bvery polite\b", r"\bkind\b"
                    ],
                    "tr": [
                        r"\bnazik\b", r"\bkibar\b"
                    ],
                    "ar": [
                        r"\bتعامل محترم\b", r"\bمحترم\b", r"\bبأدب\b"
                    ],
                    "zh": [
                        r"很有礼貌"
                    ],
                },
            ),
            
            "spir_rude": AspectRule(
                aspect_code="spir_rude",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bхамил", r"\bхамство\b", r"\bнагруб", r"\bгруб(о|ые|ый|ая)\b", r"\bнеприветлив"
                    ],
                    "en": [
                        r"\brude staff\b", r"\bunfriendly\b", r"\bimpolite\b"
                    ],
                    "tr": [
                        r"\bçok kaba\b", r"\bsaygısız\b", r"\bters davrandılar\b"
                    ],
                    "ar": [
                        r"\bموظفين وقحين\b", r"\bسيء التعامل\b"
                    ],
                    "zh": [
                        r"态度很差", r"服务很差", r"很不耐烦", r"不礼貌", r"很凶"
                    ],
                },
            ),
            
            "spir_unrespectful": AspectRule(
                aspect_code="spir_unrespectful",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bнедружелюб", r"\bразговаривал[аи]? свысока\b", r"\bнеуважительн"
                    ],
                    "en": [
                        r"\bdisrespectful\b", r"\btreated us badly\b"
                    ],
                    "tr": [
                        r"\bsaygısız\b"
                    ],
                    "ar": [
                        r"\bغير محترمين\b"
                    ],
                    "zh": [
                        r"态度很差", r"很不尊重"
                    ],
                },
            ),
            
            "spir_helpful_fast": AspectRule(
                aspect_code="spir_helpful_fast",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bпомог(ли|ли нам)\b",
                        r"\bрешили проблему\b",
                        r"\bбыстро отреагировал[аи]?\b",
                        r"\bпришли сразу\b",
                        r"\bоперативно\b"
                    ],
                    "en": [
                        r"\bvery helpful\b",
                        r"\bhelpful\b",
                        r"\bfixed it quickly\b",
                        r"\bthey came right away\b",
                        r"\bresponded immediately\b"
                    ],
                    "tr": [
                        r"\byardımcı oldular\b",
                        r"\bhemen çözdüler\b",
                        r"\bhemen geldiler\b",
                        r"\banında yardımcı oldular\b",
                        r"\bçok hızlı ilgilendiler\b"
                    ],
                    "ar": [
                        r"\bساعدونا\b",
                        r"\bحلّوا المشكلة\b",
                        r"\bاستجابوا بسرعة\b",
                        r"\bجاءوا مباشرة\b",
                        r"\bاستجابوا فورًا\b"
                    ],
                    "zh": [
                        r"很帮忙",
                        r"马上处理",
                        r"马上解决",
                        r"很快就来了",
                        r"反应很快",
                        r"马上帮忙"
                    ],
                },
            ),
            "spir_ignored_requests": AspectRule(
                aspect_code="spir_ignored_requests",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bпришлось просить несколько раз\b",
                        r"\bнас игнорировал[аи]\b",
                        r"\bпроигнорировал(и)? нашу просьбу\b",
                        r"\bникто не (приш[ëе]л|подош[ёе]л)\b",
                        r"\bникакой реакции от персонала\b",
                    ],
                    "en": [
                        r"\bignored our request\b",
                        r"\bwe had to ask (several|many) times\b",
                        r"\bnobody came\b",
                        r"\bno one came\b",
                        r"\bno response from staff\b",
                    ],
                    "tr": [
                        r"\bdefalarca istemek zorunda kaldık\b",
                        r"\bkimse gelmedi\b",
                        r"\bpersonel ilgilenmedi\b",
                        r"\byeşlik etmediler\b",
                    ],
                    "ar": [
                        r"\bتجاهلونا\b",
                        r"\bاضطرينا نطلب أكتر من مرة\b",
                        r"\bما حدا إجا\b",
                        r"\bما حدا رد\b",
                    ],
                    "zh": [
                        r"没人理我们",
                        r"我们说了好几次",
                        r"服务员一直不来",
                        r"没有人过来帮忙",
                    ],
                },
            ),
            
            "spir_slow_response": AspectRule(
                aspect_code="spir_slow_response",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bочень долго реагирова[лл][аи]?\b",
                        r"\bждали (очень )?долг[оa]\b",
                        r"\bпришл[и] только через\b",
                        r"\bмедленно обслуживали\b",
                        r"\bперсонал долго тянул\b",
                    ],
                    "en": [
                        r"\bvery slow to respond\b",
                        r"\bit took forever\b",
                        r"\b(long|very long) wait for staff\b",
                        r"\bstaff was slow\b",
                        r"\bthey were so slow to help\b",
                    ],
                    "tr": [
                        r"\bçok yavaş ilgilendiler\b",
                        r"\bçok bekledik\b",
                        r"\bbizi uzun süre beklettiler\b",
                        r"\bgeç geldiler\b",
                    ],
                    "ar": [
                        r"\bاستنينا كتير\b",
                        r"\bاتأخروا كتير\b",
                        r"\bالتعامل بطيء\b",
                        r"\bجاءوا بعد فترة طويلة\b",
                    ],
                    "zh": [
                        r"等了很久才有人来",
                        r"服务员特别慢",
                        r"反应很慢",
                        r"拖了很久才处理",
                    ],
                },
            ),
            
            "spir_not_available": AspectRule(
                aspect_code="spir_not_available",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bникого не было на ресепшен[е]?\b",
                        r"\bресепшен был пустой\b",
                        r"\bневозможно дозвониться\b",
                        r"\bникого не найти\b",
                        r"\bперсонала не было\b",
                    ],
                    "en": [
                        r"\bno one at reception\b",
                        r"\breception was empty\b",
                        r"\bcouldn't reach (the )?staff\b",
                        r"\bno staff available\b",
                        r"\bnobody answered\b",
                    ],
                    "tr": [
                        r"\bresepsiyonda kimse yoktu\b",
                        r"\bkimse yoktu\b",
                        r"\bulaşamadık\b",
                        r"\bkimse cevap vermedi\b",
                    ],
                    "ar": [
                        r"\bما في حدا بالاستقبال\b",
                        r"\bما لقينا حدا\b",
                        r"\bما قدرنا نوصل لحدا\b",
                        r"\bحدا يرد ما في\b",
                    ],
                    "zh": [
                        r"前台没人",
                        r"找不到工作人员",
                        r"没有人接电话",
                        r"联系不到工作人员",
                    ],
                },
            ),
            
            "spir_went_extra_mile": AspectRule(
                aspect_code="spir_went_extra_mile",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bсделали больше, чем ожидали\b",
                        r"\bпошли навстречу\b",
                        r"\bреально постарались\b",
                        r"\bсверх ожиданий\b",
                        r"\bсделали исключение для нас\b",
                    ],
                    "en": [
                        r"\bwent above and beyond\b",
                        r"\babove and beyond\b",
                        r"\bthey really went the extra mile\b",
                        r"\bthey made an exception for us\b",
                        r"\bthey did more than expected\b",
                    ],
                    "tr": [
                        r"\bbeklediğimizden fazlasını yaptılar\b",
                        r"\bözellikle yardımcı oldular\b",
                        r"\bözverili davrandılar\b",
                        r"\bçok uğraştılar bizim için\b",
                    ],
                    "ar": [
                        r"\bساعدونا أكثر من اللازم\b",
                        r"\bعملوا المستحيل\b",
                        r"\bعنجد تعبوا معنا\b",
                        r"\bعملوا شي مو لازم يعملوه بس كرمالنا\b",
                    ],
                    "zh": [
                        r"服务超出了预期",
                        r"真的为我们多做了一步",
                        r"帮了我们很多额外的忙",
                        r"特别为我们破例",
                    ],
                },
            ),
            "spir_professional": AspectRule(
                aspect_code="spir_professional",
                patterns_by_lang={
                    "ru": [
                        r"\bпрофессионал",
                        r"\bочень профессионал",
                        r"\bкомпетентн",
                        r"\bзнают свою работу\b",
                        r"\bвсё ч(е|ё)тко объяснил",
                        r"\bвсё грамотно объяснил",
                        r"\bвсё прозрачно\b",
                        r"\bоформили документы\b",
                        r"\bдали все чеки\b",
                    ],
                    "en": [
                        r"\bprofessional\b",
                        r"\bvery professional\b",
                        r"\bso professional\b",
                        r"\bhighly professional\b",
                        r"\bknowledgeable\b",
                        r"\bthey knew what they were doing\b",
                        r"\bclear explanation\b",
                        r"\bexplained everything clearly\b",
                        r"\btransparent\b",
                        r"\bsorted all paperwork\b",
                        r"\bgave (us )?an? invoice\b",
                    ],
                    "tr": [
                        r"\bprofesyonel\b",
                        r"\bçok profesyonel\b",
                        r"\bprofesyoneldi\b",
                        r"\bişini biliyor(du)?\b",
                        r"\baçıkça anlattı(lar)?\b",
                        r"\bfaturayı düzgün verdiler\b",
                    ],
                    "ar": [
                        r"\bمحترفين\b",
                        r"\bجداً محترفين\b",
                        r"\bيعرفون شغلهم\b",
                        r"\bشرح واضح\b",
                        r"\bكل شيء كان واضح\b",
                        r"\bأعطونا كل الفواتير\b",
                    ],
                    "zh": [
                        r"很专业",
                        r"非常专业",
                        r"服务很专业",
                        r"解释很清楚",
                        r"流程很清楚",
                        r"收费很透明",
                        r"单据都给了",
                    ],
                },
                polarity_hint="positive",
            ),
        
            "spir_unprofessional": AspectRule(
                aspect_code="spir_unprofessional",
                patterns_by_lang={
                    "ru": [
                        r"\bнекомпетентн",
                        r"\bсовсем не компетентн",
                        r"\bне знают\b",
                        r"\bне знали что делать\b",
                        r"\bбардак с документами\b",
                        r"\bполный бардак\b",
                    ],
                    "en": [
                        r"\bunprofessional\b",
                        r"\bvery unprofessional\b",
                        r"\bnot professional\b",
                        r"\bthey didn't know\b",
                        r"\bclueless staff\b",
                        r"\bdidn't know what they were doing\b",
                    ],
                    "tr": [
                        r"\bprofesyonel değildi\b",
                        r"\bhiç profesyonel değillerdi\b",
                        r"\bprofesyonel olmayan\b",
                        r"\bbilmiyorlardı\b",
                        r"\bne yapacaklarını bilmiyorlardı\b",
                    ],
                    "ar": [
                        r"\bغير محترفين\b",
                        r"\bمش محترفين\b",
                        r"\bمش فاهمين الإجراءات\b",
                        r"\bما يعرفوا شو يعملوا\b",
                    ],
                    "zh": [
                        r"不专业",
                        r"非常不专业",
                        r"完全不专业",
                        r"搞不清楚",
                        r"他们搞不清楚在做什么",
                    ],
                },
                polarity_hint="negative",
            ),
        
            "spir_payment_clear": AspectRule(
                aspect_code="spir_payment_clear",
                patterns_by_lang={
                    "ru": [
                        r"\bвсё прозрачно\b",
                        r"\bвсё понятно с оплатой\b",
                        r"\bобъяснили оплату\b",
                        r"\bобъяснили как списали деньги\b",
                        r"\bдали все чеки\b",
                        r"\bчек(и)? выдали\b",
                    ],
                    "en": [
                        r"\btransparent billing\b",
                        r"\btransparent with payment\b",
                        r"\bclear about payment\b",
                        r"\bexplained the charges\b",
                        r"\bexplained everything about payment\b",
                        r"\bgave (us )?an? invoice\b",
                        r"\bgave receipts?\b",
                    ],
                    "tr": [
                        r"\bödeme konusu çok netti\b",
                        r"\bödemeyi güzel açıkladılar\b",
                        r"\bher şeyi faturada gösterdiler\b",
                        r"\bfatura verdiler\b",
                    ],
                    "ar": [
                        r"\bكل شيء كان واضح بالدفع\b",
                        r"\bشرحوا الدفع\b",
                        r"\bأعطونا فاتورة واضحة\b",
                        r"\bأعطونا كل الفواتير\b",
                    ],
                    "zh": [
                        r"收费很透明",
                        r"账单很清楚",
                        r"把费用解释清楚了",
                        r"把收费解释得很明白",
                        r"给了发票",
                        r"单据都给了",
                    ],
                },
                polarity_hint="positive",
            ),
        
            "spir_payment_issue": AspectRule(
                aspect_code="spir_payment_issue",
                patterns_by_lang={
                    "ru": [
                        r"\bне смогли объяснить оплату\b",
                        r"\bне смогли объяснить за что сняли деньги\b",
                        r"\bпутаница с оплатой\b",
                        r"\bнепрозрачно\b",
                        r"\bошибка в сч(е|ё)те\b",
                        r"\bнеправильный сч(е|ё)т\b",
                        r"\bденьги списали неправильно\b",
                    ],
                    "en": [
                        r"\bconfused about payment\b",
                        r"\bconfusing payment\b",
                        r"\bnot transparent about payment\b",
                        r"\bnot clear about charges\b",
                        r"\bwrong charge\b",
                        r"\bwe were overcharged\b",
                        r"\bbilling mistake\b",
                        r"\bcharge error\b",
                    ],
                    "tr": [
                        r"\bödeme konusunda karışıklık\b",
                        r"\bödemeyi açıklayamadılar\b",
                        r"\bşeffaf değildi\b",
                        r"\byanlış ücret\b",
                        r"\bfaturada hata\b",
                    ],
                    "ar": [
                        r"\bمش واضح بالدفع\b",
                        r"\bما قدروا يشرحوا الدفع\b",
                        r"\bخطأ في الفاتورة\b",
                        r"\bحسبونا غلط\b",
                        r"\bدفعنا أكثر\b",
                    ],
                    "zh": [
                        r"收费不明",
                        r"解释不清楚收费",
                        r"账单有问题",
                        r"多收了钱",
                        r"收费错误",
                        r"价钱算错了",
                    ],
                },
                polarity_hint="negative",
            ),
        
            "spir_booking_mistake": AspectRule(
                aspect_code="spir_booking_mistake",
                patterns_by_lang={
                    "ru": [
                        r"\bошиблись в брон[иь]\b",
                        r"\bперепутали нашу бронь\b",
                        r"\bбардак с бронированием\b",
                        r"\bпроблема с бронированием\b",
                        r"\bне нашли нашу бронь\b",
                    ],
                    "en": [
                        r"\bmessed up (the )?reservation\b",
                        r"\breservation was messed up\b",
                        r"\bthey lost our reservation\b",
                        r"\bthey couldn't find our booking\b",
                        r"\bbooking problem\b",
                        r"\bbooking issue\b",
                    ],
                    "tr": [
                        r"\brezervasyonu karıştırdılar\b",
                        r"\brezervasyonumuzu bulamadılar\b",
                        r"\brezervasyonla sorun oldu\b",
                        r"\brezervasyon problemi\b",
                    ],
                    "ar": [
                        r"\bخطأ في الحجز\b",
                        r"\bضاع الحجز\b",
                        r"\bقالوا ما في حجز\b",
                        r"\bمشكلة بالحجز\b",
                    ],
                    "zh": [
                        r"搞错预订",
                        r"把我们的预订搞丢了",
                        r"预订有问题",
                        r"说找不到预订",
                    ],
                },
                polarity_hint="negative",
            ),
            "spir_24h_support": AspectRule(
                aspect_code="spir_24h_support",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bкруглосуточн(о|ая поддержка)\b",
                        r"\b24\s*час[аов] на связи\b",
                        r"\bна ресепшене кто-то есть всегда\b",
                        r"\bможно написать в любое время\b",
                        r"\bотвечали даже ночью\b",
                    ],
                    "en": [
                        r"\b24/?7 support\b",
                        r"\b24/?7 reception\b",
                        r"\breception is 24/?7\b",
                        r"\bsomeone (was|is) always available\b",
                        r"\bthey answered even at night\b",
                        r"\bthey replied no matter the time\b",
                    ],
                    "tr": [
                        r"\b24 saat resepsiyon\b",
                        r"\b24 saat ulaşılabilir\b",
                        r"\bgece bile cevap verdiler\b",
                        r"\bher zaman birileri vardı\b",
                        r"\bistediğimiz saatte ulaşabildik\b",
                    ],
                    "ar": [
                        r"\bاستقبال 24 ساعة\b",
                        r"\bحدا موجود طول الوقت\b",
                        r"\bدعم 24/7\b",
                        r"\bحتى بالليل بيردوا\b",
                        r"\bبيردوا بأي وقت\b",
                    ],
                    "zh": [
                        r"前台24小时有人",
                        r"24小时服务",
                        r"随时都有人可以联系",
                        r"半夜也有人回复",
                        r"任何时间都能联系到人",
                    ],
                },
            ),
        
            "spir_no_night_support": AspectRule(
                aspect_code="spir_no_night_support",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bночью никого нет\b",
                        r"\bночью никого не было на ресепшен[е]?\b",
                        r"\bпосле (10|11|12) никого\b",
                        r"\bночью не дозвониться\b",
                        r"\bночью нам никто не помог\b",
                    ],
                    "en": [
                        r"\bno reception at night\b",
                        r"\bat night there was no one\b",
                        r"\bcouldn't reach anyone at night\b",
                        r"\bno support during the night\b",
                        r"\bnobody picked up at night\b",
                    ],
                    "tr": [
                        r"\bgece resepsiyon yoktu\b",
                        r"\bgece kimse yoktu\b",
                        r"\bgece ulaşamadık\b",
                        r"\bgece yardım eden olmadı\b",
                    ],
                    "ar": [
                        r"\bبالليل ما في حدا\b",
                        r"\bبالليل الاستقبال مسكر\b",
                        r"\bما قدرنا نلاقي حدا بالليل\b",
                        r"\bما حدا رد علينا بالليل\b",
                    ],
                    "zh": [
                        r"晚上前台没人",
                        r"半夜联系不到任何人",
                        r"晚上没有工作人员",
                        r"夜里没法找到人帮忙",
                    ],
                },
            ),
        
            "spir_fast_response": AspectRule(
                aspect_code="spir_fast_response",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bответили сразу\b",
                        r"\bочень быстро ответил[аи]\b",
                        r"\bмгновенно ответил[аи]\b",
                        r"\bреакция буквально за минуту\b",
                        r"\bпринесли (сразу|буквально сразу)\b",
                    ],
                    "en": [
                        r"\bresponded right away\b",
                        r"\binstant response\b",
                        r"\bthey answered immediately\b",
                        r"\bthey came immediately\b",
                        r"\bbrought it right away\b",
                    ],
                    "tr": [
                        r"\banında cevap verdiler\b",
                        r"\bhemen döndüler\b",
                        r"\bçok hızlı cevap\b",
                        r"\bhemen geldiler\b",
                        r"\bistediğimiz şeyi hemen getirdiler\b",
                    ],
                    "ar": [
                        r"\bردوا فورًا\b",
                        r"\bرد سريع جدًا\b",
                        r"\bإجا الرد دغري\b",
                        r"\bإجوا فورًا\b",
                        r"\bجابوا الشي فورًا\b",
                    ],
                    "zh": [
                        r"马上回复",
                        r"立刻回复我们",
                        r"很快就有人来",
                        r"马上就送来了",
                        r"回应特别快",
                    ],
                },
            ),
        
            "spir_language_ok": AspectRule(
                aspect_code="spir_language_ok",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bговорили по-(русски|русскому)\b",
                        r"\bговорили на русском\b",
                        r"\bхорошо говорили по-английски\b",
                        r"\bговорят на английском\b",
                        r"\bможно нормально объясниться\b",
                    ],
                    "en": [
                        r"\bstaff spoke good English\b",
                        r"\bgood English\b",
                        r"\bthey spoke English very well\b",
                        r"\bcommunication was easy\b",
                        r"\bno language barrier\b",
                    ],
                    "tr": [
                        r"\bingilizceleri iyiydi\b",
                        r"\bingilizce konuştuk sorunsuz\b",
                        r"\biletişim çok rahattı\b",
                        r"\banlaşmak kolaydı\b",
                    ],
                    "ar": [
                        r"\bبيحكوا إنجليزي منيح\b",
                        r"\bالتواصل كان سهل\b",
                        r"\bما في مشكلة لغة\b",
                        r"\bفاهمين علينا مباشرة\b",
                    ],
                    "zh": [
                        r"英文很好",
                        r"沟通很顺畅",
                        r"交流没有问题",
                        r"可以直接用英语沟通",
                    ],
                },
            ),
        
            "spir_language_barrier": AspectRule(
                aspect_code="spir_language_barrier",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bтрудно объясниться\b",
                        r"\bне говорят по(-| )английски\b",
                        r"\bанглийского почти нет\b",
                        r"\bникто не говорит по-английски\b",
                        r"\bязык — это проблема\b",
                    ],
                    "en": [
                        r"\blanguage barrier\b",
                        r"\bno one spoke English\b",
                        r"\bnobody speaks English\b",
                        r"\bdifficult to communicate\b",
                        r"\bhard to communicate\b",
                    ],
                    "tr": [
                        r"\bingilizce bilmiyorlardı\b",
                        r"\biletişim zordu\b",
                        r"\banlaşmak zordu\b",
                        r"\bdil sorunu vardı\b",
                    ],
                    "ar": [
                        r"\bما حدا بيحكي إنجليزي\b",
                        r"\bصعب نتفاهم\b",
                        r"\bصفحة اللغة كانت مشكلة\b",
                        r"\bفي مشكلة لغة\b",
                    ],
                    "zh": [
                        r"沟通很困难",
                        r"没人会说英语",
                        r"基本不会英文",
                        r"语言不通",
                        r"交流很难",
                    ],
                },
            ),
            
            "checkin_fast": AspectRule(
                aspect_code="checkin_fast",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bбыстро заселили\b",
                        r"\bзаселение прошло быстро\b",
                        r"\bчек-?ин занял (пару минут|минуту)\b",
                        r"\bоформили быстро\b",
                    ],
                    "en": [
                        r"\bquick check[- ]?in\b",
                        r"\bfast check[- ]?in\b",
                        r"\bcheck ?in was fast\b",
                        r"\bcheck[- ]?in took (only|just) a few minutes\b",
                    ],
                    "tr": [
                        r"\bhızlı check[- ]?in\b",
                        r"\bhemen yerleştirdiler\b",
                        r"\bbeklemeden oda verdiler\b",
                        r"\bcheck[- ]?in çok hızlıydı\b",
                    ],
                    "ar": [
                        r"\bتسجيل دخول سريع\b",
                        r"\bدخلنا فورًا\b",
                        r"\bخلصنا تسجيل الدخول بسرعة\b",
                        r"\bالإجراءات كانت سريعة\b",
                    ],
                    "zh": [
                        r"办理入住很快",
                        r"很快就办好入住",
                        r"入住办理得很快",
                        r"几分钟就办好",
                    ],
                },
            ),
        
            "no_wait_checkin": AspectRule(
                aspect_code="no_wait_checkin",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bбез ожидания заселили\b",
                        r"\bзаселили без задержек\b",
                        r"\bникакой очереди на заселение\b",
                        r"\bзаселили сразу\b",
                    ],
                    "en": [
                        r"\bno wait\b",
                        r"\bno waiting to check[- ]?in\b",
                        r"\bgot our room immediately\b",
                        r"\bwe could check in right away\b",
                    ],
                    "tr": [
                        r"\bbeklemek zorunda kalmadık\b",
                        r"\bbeklemeden check[- ]?in yaptık\b",
                        r"\bhemen odaya geçtik\b",
                    ],
                    "ar": [
                        r"\bما انتظرنا\b",
                        r"\bدخلنا فورًا بدون انتظار\b",
                        r"\bبدون ما نوقف بالطابور\b",
                    ],
                    "zh": [
                        r"几乎不用等就办好入住",
                        r"不用等就给我们房间",
                        r"直接就让我们入住",
                    ],
                },
            ),
        
            "checkin_wait_long": AspectRule(
                aspect_code="checkin_wait_long",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bждали долго\b",
                        r"\bпришлось долго ждать заселения\b",
                        r"\bдолго оформляли\b",
                        r"\bочередь на заселение\b",
                        r"\bчек-?ин занял слишком долго\b",
                    ],
                    "en": [
                        r"\bhad to wait a long time\b",
                        r"\blong wait to check[- ]?in\b",
                        r"\bcheck[- ]?in took too long\b",
                        r"\bwe waited forever\b",
                        r"\bslow check[- ]?in\b",
                    ],
                    "tr": [
                        r"\bçok bekledik\b",
                        r"\bcheck[- ]?in çok yavaştı\b",
                        r"\bcheck[- ]?in çok uzun sürdü\b",
                        r"\buzun sıra vardı\b",
                    ],
                    "ar": [
                        r"\bانتظرنا كثيرًا\b",
                        r"\bتسجيل الدخول كان بطيء\b",
                        r"\bتشيك إن طول كثير\b",
                        r"\bاضطرينا نوقف بالطابور وقت طويل\b",
                    ],
                    "zh": [
                        r"办理入住等了很久",
                        r"入住花了很久才办好",
                        r"入住手续特别慢",
                        r"排队很久才入住",
                    ],
                },
            ),
        
            "room_not_ready_delay": AspectRule(
                aspect_code="room_not_ready_delay",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bномер не был готов\b.*\bпришлось ждать\b",
                        r"\bждали пока подготовят номер\b",
                        r"\bпопросили подождать пока уберут\b",
                        r"\bне могли заселиться сразу\b.*\bномер не подготовлен\b",
                    ],
                    "en": [
                        r"\broom wasn't ready so we had to wait\b",
                        r"\bhad to wait for the room to be ready\b",
                        r"\bhad to wait for cleaning\b",
                        r"\bthey told us to wait until the room was ready\b",
                    ],
                    "tr": [
                        r"\boda hazır değildi, beklemek zorunda kaldık\b",
                        r"\boda hazır değildi o yüzden bekledik\b",
                        r"\btemizlenmesi için bekledik\b",
                    ],
                    "ar": [
                        r"\bالغرفة ما كانت جاهزة واضطرينا ننتظر\b",
                        r"\bخلونا ننطر ليجهزوا الغرفة\b",
                        r"\bاستنينا لحتى ينظفوا الغرفة\b",
                    ],
                    "zh": [
                        r"房间还没准备好我们只能等",
                        r"房间没打扫好所以我们得等",
                        r"他们让我们等房间弄好",
                    ],
                },
            ),
        
            "room_ready_on_arrival": AspectRule(
                aspect_code="room_ready_on_arrival",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bномер был готов\b",
                        r"\bвсё готово к нашему приезду\b",
                        r"\bготовый номер жд(а|о)л\b",
                    ],
                    "en": [
                        r"\broom was ready\b",
                        r"\broom ready on arrival\b",
                        r"\bour room was already ready\b",
                    ],
                    "tr": [
                        r"\boda hazırdı\b",
                        r"\bgeldiğimizde oda hazırdı\b",
                        r"\bhemen hazır odayı verdiler\b",
                    ],
                    "ar": [
                        r"\bالغرفة جاهزة وقت وصلنا\b",
                        r"\bكانت الغرفة جاهزة مباشرة\b",
                        r"\bفورًا عطونا غرفة جاهزة\b",
                    ],
                    "zh": [
                        r"房间一开始就准备好了",
                        r"到的时候房间已经准备好了",
                        r"我们一到房间就好了",
                    ],
                },
            ),
            
            "clean_on_arrival": AspectRule(
                aspect_code="clean_on_arrival",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bчисто при заселении\b",
                        r"\bномер был чистый\b",
                        r"\bвсё убрано\b",
                        r"\bидеально чисто при заезде\b",
                        r"\bочень чисто\b",
                        r"\bсвежая постель\b",
                    ],
                    "en": [
                        r"\bclean on arrival\b",
                        r"\broom was clean on arrival\b",
                        r"\bspotless when we arrived\b",
                        r"\bvery clean\b",
                        r"\bfresh bedding\b",
                    ],
                    "tr": [
                        r"\bgeldiğimizde tertemizdi\b",
                        r"\boda temizdi\b",
                        r"\bher yer temizdi\b",
                        r"\byeni çarşaf\b",
                        r"\btemiz çarşaf\b",
                    ],
                    "ar": [
                        r"\bالحمام نظيف\b",  # общая чистота иногда упоминается через ванную
                        r"\bالغرفة نظيفة عند الوصول\b",
                        r"\bكل شي كان نظيف\b",
                        r"\bشرشف نظيف\b",
                        r"\bالمكان نظيف وقت وصلنا\b",
                    ],
                    "zh": [
                        r"一进来就很干净",
                        r"房间很干净",
                        r"刚入住的时候就很干净",
                        r"床单很干净",
                        r"新的床单",
                    ],
                },
            ),
        
            "room_not_ready": AspectRule(
                aspect_code="room_not_ready",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bномер не был готов\b",
                        r"\bк заселению номер не подготовили\b",
                        r"\bномер не подготовлен\b",
                        r"\bв номере не убрано\b",
                    ],
                    "en": [
                        r"\broom was not ready\b",
                        r"\broom not prepared\b",
                        r"\broom wasn't ready\b",
                        r"\bnot cleaned before we arrived\b",
                    ],
                    "tr": [
                        r"\boda hazır değildi\b",
                        r"\boda temizlenmemişti\b",
                        r"\boda hazırlanmadı\b",
                    ],
                    "ar": [
                        r"\bالغرفة ما كانت جاهزة\b",
                        r"\bلسا ما نظفوا\b",
                        r"\bما كانت محضرة\b",
                    ],
                    "zh": [
                        r"房间还没准备好",
                        r"房间没打扫",
                        r"房间没准备好就让我们来",
                    ],
                },
            ),
        
            "dirty_on_arrival": AspectRule(
                aspect_code="dirty_on_arrival",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bгрязно при заселении\b",
                        r"\bгрязный номер\b",
                        r"\bв номере не убрано\b",
                        r"\bпыль на поверхностях\b",
                        r"\bгрязный пол\b",
                        r"\bлипкий пол\b",
                        r"\bлипкий стол\b",
                        r"\bгрязная постель\b",
                    ],
                    "en": [
                        r"\bdirty when we arrived\b",
                        r"\broom was dirty\b",
                        r"\bdusty surfaces\b",
                        r"\bsticky floor\b",
                        r"\bsticky table\b",
                        r"\bstained sheets\b",
                    ],
                    "tr": [
                        r"\boda kirliydi\b",
                        r"\bgeldiğimizde kirliydi\b",
                        r"\btoz vardı\b",
                        r"\byapış yapış zemin\b",
                        r"\byapış yapış masa\b",
                        r"\bçarşafta leke vardı\b",
                    ],
                    "ar": [
                        r"\bالغرفة كانت وسخة عند الوصول\b",
                        r"\bمكان ما كان منظف\b",
                        r"\bغبار على السطوح\b",
                        r"\bالأرض وسخة\b",
                        r"\bالطاولة لزجة\b",
                        r"\bالشرشف وسخ\b",
                    ],
                    "zh": [
                        r"刚入住的时候就很脏",
                        r"房间没打扫",
                        r"到处是灰",
                        r"地板黏",
                        r"桌子黏黏的",
                        r"床单有污渍",
                    ],
                },
            ),
        
            "leftover_trash_previous_guest": AspectRule(
                aspect_code="leftover_trash_previous_guest",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bмусор от прошлых гостей\b",
                        r"\bостался мусор\b",
                        r"\bбутылки остались\b",
                        r"\bгрязные полотенца от предыдущих\b",
                        r"\bследы предыдущих гостей\b",
                    ],
                    "en": [
                        r"\btrash from previous guest\b",
                        r"\bprevious guest's garbage\b",
                        r"\bold towels left\b",
                        r"\bleftover trash\b",
                        r"\btheir stuff was still there\b",
                    ],
                    "tr": [
                        r"\bönceki misafirin çöpleri\b",
                        r"\beski havlular duruyordu\b",
                        r"\bönceki misafirden kalan şeyler vardı\b",
                    ],
                    "ar": [
                        r"\bزبالة من الضيف السابق\b",
                        r"\bمناشف وسخة من قبل\b",
                        r"\bأغراض الضيف السابق بعدها موجودة\b",
                    ],
                    "zh": [
                        r"上个客人的垃圾还在",
                        r"旧的毛巾还在",
                        r"还有上个客人的东西没收",
                    ],
                },
            ),
        
            "access_smooth": AspectRule(
                aspect_code="access_smooth",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bлегко нашли вход\b",
                        r"\bдоступ в номер без проблем\b",
                        r"\bкод от двери сработал\b",
                        r"\bвойти было просто\b",
                        r"\bвсё с доступом понятно\b",
                    ],
                    "en": [
                        r"\beasy to get in\b",
                        r"\baccess was simple\b",
                        r"\bentrance was clear\b",
                        r"\bdoor code worked\b",
                        r"\bthe code worked first time\b",
                    ],
                    "tr": [
                        r"\bgirişi bulmak kolaydı\b",
                        r"\bkapı kodu çalıştı\b",
                        r"\bodaya girmek sorunsuzdu\b",
                        r"\biçeri girmek kolaydı\b",
                    ],
                    "ar": [
                        r"\bالدخول سهل\b",
                        r"\bالكود اشتغل من أول مرة\b",
                        r"\bدخلنا بدون مشكلة\b",
                    ],
                    "zh": [
                        r"很容易进来",
                        r"门码直接能用",
                        r"进房间没问题",
                        r"入口很清楚",
                    ],
                },
            ),
            
            "door_code_worked": AspectRule(
                aspect_code="door_code_worked",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bкод от двери сработал\b",
                        r"\bкод сразу сработал\b",
                        r"\bкод подош[ëе]л\b",
                        r"\bкарта работала\b",
                        r"\bкарта от двери работала\b",
                        r"\bзамок открылся без проблем\b",
                    ],
                    "en": [
                        r"\bdoor code worked\b",
                        r"\bthe code worked\b",
                        r"\bkey card worked\b",
                        r"\bkeycard worked\b",
                        r"\bthe lock opened with no problem\b",
                        r"\bno problem with the lock\b",
                    ],
                    "tr": [
                        r"\bkapı kodu çalıştı\b",
                        r"\bkart çalıştı\b",
                        r"\bkapı hemen açıldı\b",
                        r"\bkilit sorunsuz açıldı\b",
                    ],
                    "ar": [
                        r"\bالكود اشتغل من أول مرة\b",
                        r"\bالكرت اشتغل\b",
                        r"\bالباب فتح بسهولة\b",
                        r"\bالقفل فتح بدون مشكلة\b",
                    ],
                    "zh": [
                        r"门码直接能用",
                        r"门禁码能用",
                        r"房卡能用",
                        r"门很容易就打开了",
                    ],
                },
            ),
        
            "tech_access_issue": AspectRule(
                aspect_code="tech_access_issue",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bкод не сработал\b",
                        r"\bкод не подходил\b",
                        r"\bкарта не работал[ае]\b",
                        r"\bключ-карта не работал[ае]\b",
                        r"\bзамок не открывался\b",
                        r"\bдверь не открывалась\b",
                        r"\bне могли попасть внутрь\b",
                    ],
                    "en": [
                        r"\bdoor code didn't work\b",
                        r"\bthe code didn't work\b",
                        r"\bkey card didn't work\b",
                        r"\bkeycard didn't work\b",
                        r"\block didn't open\b",
                        r"\bcouldn't get in\b",
                        r"\bcouldn't get inside\b",
                    ],
                    "tr": [
                        r"\bkapı kodu çalışmadı\b",
                        r"\bkart çalışmadı\b",
                        r"\bkapı açılmadı\b",
                        r"\biçeri giremedik\b",
                        r"\bkilit açılmadı\b",
                    ],
                    "ar": [
                        r"\bالكود ما اشتغل\b",
                        r"\bالكرت ما يشتغل\b",
                        r"\bالباب ما فتح\b",
                        r"\bما قدرنا ندخل\b",
                        r"\bالقفل ما كان يفتح\b",
                    ],
                    "zh": [
                        r"门码不好用",
                        r"门码不能用",
                        r"房卡不好用",
                        r"门打不开",
                        r"进不去房间",
                    ],
                },
            ),
        
            "entrance_hard_to_find": AspectRule(
                aspect_code="entrance_hard_to_find",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bсложно найти вход\b",
                        r"\bнепонятно куда заходить\b",
                        r"\bне могли найти вход\b",
                        r"\bтрудно найти подъезд\b",
                        r"\bвход неочевидный\b",
                        r"\bнепонятно где вход\b",
                    ],
                    "en": [
                        r"\bhard to find the entrance\b",
                        r"\bdifficult to find the entrance\b",
                        r"\bentrance was hard to find\b",
                        r"\bconfusing entrance\b",
                        r"\bnot clear where to enter\b",
                    ],
                    "tr": [
                        r"\bgirişi bulmak zordu\b",
                        r"\bhangi kapıdan gireceğimizi anlamadık\b",
                        r"\bgiriş karışıktı\b",
                        r"\bbinayı bulmak zordu\b",
                    ],
                    "ar": [
                        r"\bصعب نلاقي المدخل\b",
                        r"\bما عرفنا من وين نفوت\b",
                        r"\bالمدخل مو واضح\b",
                        r"\bالدخول معقّد\b",
                    ],
                    "zh": [
                        r"入口很难找",
                        r"很难找到门",
                        r"不知道从哪里进去",
                        r"入口不明显",
                        r"进门很麻烦",
                    ],
                },
            ),
        
            "no_elevator_baggage_issue": AspectRule(
                aspect_code="no_elevator_baggage_issue",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bлифт не работал\b",
                        r"\bлифт сломан\b",
                        r"\bбез лифта очень тяжело\b",
                        r"\bтащить чемоданы по лестнице\b",
                        r"\bтаскать чемоданы наверх\b",
                        r"\bтащить багаж наверх\b",
                        r"\bбез лифта с чемоданами тяжело\b",
                    ],
                    "en": [
                        r"\bno elevator\b",
                        r"\bno working elevator\b",
                        r"\belevator was not working\b",
                        r"\belevator was broken\b",
                        r"\bhad to carry luggage up the stairs\b",
                        r"\bcarrying suitcases upstairs was hard\b",
                        r"\bnot easy with luggage\b",
                    ],
                    "tr": [
                        r"\basansör yoktu\b",
                        r"\basansör çalışmıyordu\b",
                        r"\basansör bozuktu\b",
                        r"\bmerdivenle bavul taşımak çok zordu\b",
                        r"\bbavullarla çıkmak çok zordu\b",
                    ],
                    "ar": [
                        r"\bما في مصعد\b",
                        r"\bالمصعد معطل\b",
                        r"\bالمصعد خربان\b",
                        r"\bاضطرينا نطلع الدرج مع الشنط\b",
                        r"\bصعب كتير مع الشنط\b",
                    ],
                    "zh": [
                        r"没有电梯",
                        r"电梯坏了",
                        r"电梯不能用",
                        r"只能扛行李上楼",
                        r"拿行李走楼梯很辛苦",
                    ],
                },
            ),
            
            "payment_clear": AspectRule(
                aspect_code="payment_clear",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bвсё прозрачно по оплате\b",
                        r"\bвсё объяснили по оплате\b",
                        r"\bобъяснили как будет списание\b",
                        r"\bс оплатой всё понятно\b",
                        r"\bвсё по оплате честно\b",
                    ],
                    "en": [
                        r"\btransparent payment\b",
                        r"\bclear about charges\b",
                        r"\bthey explained the charges\b",
                        r"\bthey explained the payment\b",
                        r"\bclear payment process\b",
                    ],
                    "tr": [
                        r"\bödeme şeffaftı\b",
                        r"\bher şeyi anlattılar\b",
                        r"\bödemeyi açıkça anlattılar\b",
                        r"\bödeme konusunda çok nettiler\b",
                    ],
                    "ar": [
                        r"\bالدفع كان واضح\b",
                        r"\bشرحوا الرسوم\b",
                        r"\bشرحوا كل شي بخصوص الدفع\b",
                        r"\bما كان في شي موضح بالدفع\b",
                    ],
                    "zh": [
                        r"收费很透明",
                        r"把费用解释清楚了",
                        r"付款流程说得很清楚",
                        r"收费讲得很明白",
                    ],
                },
            ),
        
            "deposit_clear": AspectRule(
                aspect_code="deposit_clear",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bдепозит объяснили\b",
                        r"\bобъяснили про депозит\b",
                        r"\bс депозитом всё понятно\b",
                        r"\bсразу сказали про залог\b",
                    ],
                    "en": [
                        r"\bexplained the deposit\b",
                        r"\bdeposit was clearly explained\b",
                        r"\bclear deposit policy\b",
                        r"\bthey told us about the deposit upfront\b",
                    ],
                    "tr": [
                        r"\bdepozitoyu açıkladılar\b",
                        r"\bdepozito konusunda nett(iler|i)\b",
                        r"\bdepozito önceden açıklandı\b",
                    ],
                    "ar": [
                        r"\bشرحوا العربون\b",
                        r"\bالعربون كان واضح\b",
                        r"\bشرحوا قديش رح يحجزوا\b",
                    ],
                    "zh": [
                        r"押金说明得很清楚",
                        r"提前说清楚押金",
                        r"押金政策讲清楚了",
                    ],
                },
            ),
        
            "docs_provided": AspectRule(
                aspect_code="docs_provided",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bдали чеки\b",
                        r"\bдали отчетные документы\b",
                        r"\bвсе документы на месте\b",
                        r"\bвсе бумаги дали\b",
                    ],
                    "en": [
                        r"\bgave invoice\b",
                        r"\bgave us an invoice\b",
                        r"\bgave us receipts?\b",
                        r"\bprovided all documents\b",
                        r"\bwe received all paperwork\b",
                    ],
                    "tr": [
                        r"\bfatura verdiler\b",
                        r"\bbelgeleri verdiler\b",
                        r"\btüm evrakları verdiler\b",
                        r"\bfişimizi verdiler\b",
                    ],
                    "ar": [
                        r"\bأعطونا الفاتورة\b",
                        r"\bأعطونا الإيصال\b",
                        r"\bأعطونا كل الأوراق\b",
                        r"\bفي فاتورة رسمية\b",
                    ],
                    "zh": [
                        r"给了发票",
                        r"给了收据",
                        r"所有单据都给了",
                        r"文件都提供了",
                    ],
                },
            ),
        
            "no_hidden_fees": AspectRule(
                aspect_code="no_hidden_fees",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bникаких скрытых платежей\b",
                        r"\bбез скрытых платежей\b",
                        r"\bничего лишнего не списали\b",
                        r"\bникаких дополнительных платежей\b",
                    ],
                    "en": [
                        r"\bno hidden fees\b",
                        r"\bno extra fees\b",
                        r"\bno unexpected charges\b",
                        r"\bnothing extra was charged\b",
                    ],
                    "tr": [
                        r"\bekstra ücret yoktu\b",
                        r"\bgizli ücret yoktu\b",
                        r"\bsonradan ekstra bir şey çıkmadı\b",
                    ],
                    "ar": [
                        r"\bمافي رسوم مخفية\b",
                        r"\bما في شي زيادة فجأة\b",
                        r"\bما أخدوا شي زيادة\b",
                    ],
                    "zh": [
                        r"没有乱收费",
                        r"没有隐藏收费",
                        r"没有多收钱",
                    ],
                },
            ),
        
            "payment_confusing": AspectRule(
                aspect_code="payment_confusing",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bпутаница с оплатой\b",
                        r"\bне объяснили налоги\b",
                        r"\bнепонятные доплаты\b",
                        r"\bне смогли объяснить оплату\b",
                        r"\bнеясно за что взяли\b",
                    ],
                    "en": [
                        r"\bconfusing payment\b",
                        r"\bnot clear about payment\b",
                        r"\bthey didn't explain the charges\b",
                        r"\bthey couldn't explain why we were charged\b",
                        r"\bextra charge we didn't understand\b",
                    ],
                    "tr": [
                        r"\bödeme karışıktı\b",
                        r"\bödemeyi düzgün açıklamadılar\b",
                        r"\bneye para aldıklarını anlatamadılar\b",
                        r"\banlaşılmaz ek ücret\b",
                    ],
                    "ar": [
                        r"\bالدفع كان مو واضح\b",
                        r"\bما شرحوا الرسوم\b",
                        r"\bدفعنا شي مش فاهمينه\b",
                        r"\bفي مبلغ زيادة ومش فاهمين ليش\b",
                    ],
                    "zh": [
                        r"付款不清楚",
                        r"收费解释不清楚",
                        r"有额外收费但没说明白",
                        r"不知道为什么要多收钱",
                    ],
                },
            ),
            
            "unexpected_charge": AspectRule(
                aspect_code="unexpected_charge",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bпопросили неожиданный депозит\b",
                        r"\bзаблокировали деньги без объяснения\b",
                        r"\bвзяли деньги без предупреждения\b",
                        r"\bдополнительный плат(е|ё)ж без предупреждения\b",
                        r"\bсняли лишние деньги\b",
                    ],
                    "en": [
                        r"\bunexpected deposit\b",
                        r"\bthey blocked money without explaining\b",
                        r"\bextra charge we didn't know about\b",
                        r"\bcharged us extra without telling\b",
                        r"\bthey took extra money without warning\b",
                    ],
                    "tr": [
                        r"\bbeklenmedik depozito istediler\b",
                        r"\bhaber vermeden para bloke ettiler\b",
                        r"\bekstra ücret talep ettiler\b",
                        r"\bhaber vermeden ekstra ücret kestiler\b",
                    ],
                    "ar": [
                        r"\bطلبوا عربون بدون توضيح\b",
                        r"\bسحبوا مبلغ إضافي بدون ما يشرحوا\b",
                        r"\bدفعونا زيادة بدون ما نعرف\b",
                        r"\bخصموا مصاري زيادة فجأة\b",
                    ],
                    "zh": [
                        r"要了额外押金没说明",
                        r"没提前说就多收钱",
                        r"多扣了钱也没解释",
                        r"突然多收一笔费用",
                    ],
                },
            ),
        
            "hidden_fees": AspectRule(
                aspect_code="hidden_fees",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bскрыт(ы|ые) платежи\b",
                        r"\bскрыт(ая|ые) комисси[яи]\b",
                        r"\bнавязанные доплаты\b",
                        r"\bнеоговоренные доплаты\b",
                        r"\bс нас попытались взять больше\b",
                    ],
                    "en": [
                        r"\bhidden fees\b",
                        r"\bextra fees we didn't know about\b",
                        r"\bunexpected fees\b",
                        r"\bovercharge with hidden fees\b",
                        r"\bthey tried to charge us more\b",
                    ],
                    "tr": [
                        r"\bgizli ücret\b",
                        r"\bekstra ücret çıkardılar\b",
                        r"\bhaber vermeden ek ücret yazdılar\b",
                        r"\bfazla ücret almaya çalıştılar\b",
                    ],
                    "ar": [
                        r"\bرسوم مخفية\b",
                        r"\bأخدوا رسوم زيادة بدون ما يقولوا\b",
                        r"\bدفعونا رسوم إضافية فجأة\b",
                        r"\bحاولوا ياخدوا أكتر\b",
                    ],
                    "zh": [
                        r"有隐藏收费",
                        r"乱收费",
                        r"偷偷多收费用",
                        r"想多收我们钱",
                    ],
                },
            ),
        
            "deposit_problematic": AspectRule(
                aspect_code="deposit_problematic",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bдепозит не вернули сразу\b",
                        r"\bзалог не вернули\b",
                        r"\bзаморозили слишком большую сумму\b",
                        r"\bудержали депозит\b",
                        r"\bпроблемы с возвратом депозита\b",
                    ],
                    "en": [
                        r"\bthey didn't return the deposit\b",
                        r"\bdeposit was not returned\b",
                        r"\bthey held our deposit\b",
                        r"\bproblem with the deposit refund\b",
                        r"\bdeposit took too long to get back\b",
                    ],
                    "tr": [
                        r"\bdepozitoyu geri vermediler\b",
                        r"\bdepozito hemen iade edilmedi\b",
                        r"\bdepozitoyu tuttular\b",
                        r"\bdepozito geri almak sorun oldu\b",
                    ],
                    "ar": [
                        r"\bما رجعوا العربون\b",
                        r"\bالعربون ما انرد\b",
                        r"\bتأخروا يرجعوا العربون\b",
                        r"\bمشكلة باسترجاع العربون\b",
                    ],
                    "zh": [
                        r"押金没退",
                        r"押金拖很久才退",
                        r"押金一直不退给我们",
                        r"押金有问题",
                    ],
                },
            ),
        
            "billing_mistake": AspectRule(
                aspect_code="billing_mistake",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bошибка в сч(е|ё)те\b",
                        r"\bсч(е|ё)т был неправильный\b",
                        r"\bневерный сч(е|ё)т\b",
                        r"\bнас посчитали не так\b",
                        r"\bне та сумма в сч(е|ё)те\b",
                    ],
                    "en": [
                        r"\bwrong bill\b",
                        r"\bincorrect bill\b",
                        r"\bbilling mistake\b",
                        r"\bcharge error\b",
                        r"\bthe bill was wrong\b",
                    ],
                    "tr": [
                        r"\bfaturada hata vardı\b",
                        r"\byanlış ücret yazdılar\b",
                        r"\bfatura yanlış kesildi\b",
                        r"\bhesap yanlış\b",
                    ],
                    "ar": [
                        r"\bفي خطأ بالفاتورة\b",
                        r"\bحسبونا غلط\b",
                        r"\bالفاتورة مو صحيحة\b",
                        r"\bالمبلغ عالفاتورة مو صحيح\b",
                    ],
                    "zh": [
                        r"账单有问题",
                        r"账单算错了",
                        r"账单金额不对",
                        r"收费用算错了",
                    ],
                },
            ),
        
            "overcharge": AspectRule(
                aspect_code="overcharge",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bнас попытались взять больше\b",
                        r"\bсняли больше чем должны\b",
                        r"\bвзяли лишние деньги\b",
                        r"\bзавышенная сумма\b",
                        r"\bпереплата\b",
                    ],
                    "en": [
                        r"\bovercharged\b",
                        r"\bthey overcharged us\b",
                        r"\bcharged us more than (agreed|they should)\b",
                        r"\bthey took more money\b",
                        r"\bextra charge we shouldn't have paid\b",
                    ],
                    "tr": [
                        r"\bfazla ücret aldılar\b",
                        r"\bnormalden fazla para kestiler\b",
                        r"\bfazla para çektiler\b",
                        r"\bgereğinden fazla ücret yazıldı\b",
                    ],
                    "ar": [
                        r"\bحسبونا زيادة\b",
                        r"\bأخدوا مصاري زيادة\b",
                        r"\bدفعنا أكتر من اللازم\b",
                        r"\bدفعونا زيادة عن الاتفاق\b",
                    ],
                    "zh": [
                        r"多收钱了",
                        r"被多收费用",
                        r"收的钱比说好的多",
                        r"收了不该收的费用",
                    ],
                },
            ),
            
            "instructions_clear": AspectRule(
                aspect_code="instructions_clear",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bвсё подробно объяснили\b",
                        r"\bполучили понятные инструкции\b",
                        r"\bвсе инструкции заранее\b",
                        r"\bнам всё разжевали\b",
                        r"\bвсё было понятно как заселиться\b",
                    ],
                    "en": [
                        r"\bclear instructions\b",
                        r"\bthey sent all instructions\b",
                        r"\bthey explained how to enter\b",
                        r"\beverything was explained\b",
                        r"\bwe got clear check[- ]?in instructions\b",
                    ],
                    "tr": [
                        r"\btalimatlar çok netti\b",
                        r"\btalimatlar çok açıktı\b",
                        r"\bnereden gireceğimizi anlattılar\b",
                        r"\bgirişle ilgili her şeyi anlattılar\b",
                    ],
                    "ar": [
                        r"\bالتعليمات كانت واضحة\b",
                        r"\bرسلوا كل التعليمات\b",
                        r"\bشرحوا لنا كل شي\b",
                        r"\bشرحوا كيف نفوت\b",
                    ],
                    "zh": [
                        r"指示很清楚",
                        r"入住说明很清楚",
                        r"一开始就跟我们说清楚怎么进门",
                        r"告诉我们怎么进门",
                    ],
                },
            ),
        
            "self_checkin_easy": AspectRule(
                aspect_code="self_checkin_easy",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bсамостоятельное заселение было (простым|удобным)\b",
                        r"\bсамостоятельно заселиться было легко\b",
                        r"\bселф ?чек-?ин был простой\b",
                        r"\bчек-?ин без персонала прош(ë|е)л легко\b",
                    ],
                    "en": [
                        r"\bself check[- ]?in was easy\b",
                        r"\beasy self check[- ]?in\b",
                        r"\bsuper easy self check[- ]?in\b",
                        r"\bself check[- ]?in was super straightforward\b",
                    ],
                    "tr": [
                        r"\bself check[- ]?in kolaydı\b",
                        r"\bkendi kendine giriş çok kolaydı\b",
                        r"\bpersonelsiz check[- ]?in çok rahattı\b",
                    ],
                    "ar": [
                        r"\bتسجيل الدخول الذاتي كان سهل\b",
                        r"\bدخول ذاتي كان سهل\b",
                        r"\bالتشيك إن لحالنا كان سهل\b",
                    ],
                    "zh": [
                        r"自助入住很简单",
                        r"自助入住很方便",
                        r"自助check in很容易",
                    ],
                },
            ),
        
            "wifi_info_given": AspectRule(
                aspect_code="wifi_info_given",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bпароль от ?wi[- ]?fi сразу дали\b",
                        r"\bсразу дали пароль от вай-?фая\b",
                        r"\bдали пароль от wi[- ]?fi сразу\b",
                        r"\bдали доступ к вайфаю сразу\b",
                    ],
                    "en": [
                        r"\bwifi password (was|got) (given|provided) (right away|immediately)\b",
                        r"\bthey gave us the wifi password immediately\b",
                        r"\bthey sent the wifi info in advance\b",
                        r"\bwe got the wifi details right away\b",
                    ],
                    "tr": [
                        r"\bwifi şifresini hemen verdiler\b",
                        r"\bwifi şifresini direkt söylediler\b",
                        r"\binternete hemen bağlandık\b",
                    ],
                    "ar": [
                        r"\bعطونا كلمة سر الواي فاي مباشرة\b",
                        r"\bعطونا الواي فاي فورًا\b",
                        r"\bعطونا الباسورد دغري\b",
                    ],
                    "zh": [
                        r"一开始就给了wifi密码",
                        r"马上告诉我们wifi密码",
                        r"很快就把WiFi信息给我们",
                    ],
                },
            ),
        
            "instructions_confusing": AspectRule(
                aspect_code="instructions_confusing",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bникаких инструкций\b",
                        r"\bинструкций не было\b",
                        r"\bинструкции непонятные\b",
                        r"\bнепонятно куда идти\b",
                        r"\bнепонятно куда заходить\b",
                        r"\bразбираться пришлось самим\b",
                    ],
                    "en": [
                        r"\bno instructions\b",
                        r"\binstructions unclear\b",
                        r"\b(confusing|complicated) self check[- ]?in\b",
                        r"\bwe had to figure it out ourselves\b",
                        r"\bthe instructions were confusing\b",
                    ],
                    "tr": [
                        r"\btalimat yoktu\b",
                        r"\btalimatlar net değildi\b",
                        r"\btalimatlar karışıktı\b",
                        r"\bcheck[- ]?in talimatları anlaşılır değildi\b",
                    ],
                    "ar": [
                        r"\bما في تعليمات واضحة\b",
                        r"\bالتعليمات مربكة\b",
                        r"\bما عطونا تعليمات\b",
                        r"\bاضطرينا نكتشف لوحدنا\b",
                    ],
                    "zh": [
                        r"没有说明",
                        r"指示不清楚",
                        r"自助入住很复杂",
                        r"只能自己摸索",
                        r"我们得自己摸索怎么进门",
                    ],
                },
            ),
        
            "wifi_info_missing": AspectRule(
                aspect_code="wifi_info_missing",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bне сказали пароль от ?wi[- ]?fi\b",
                        r"\bне сказали пароль от вай-?фая\b",
                        r"\bне дали пароль от вай-?фая\b",
                        r"\bпароль от wi[- ]?fi так и не дали\b",
                        r"\bпароль от интернета не сообщили\b",
                    ],
                    "en": [
                        r"\bdidn't tell us the wifi password\b",
                        r"\bno wifi password\b",
                        r"\bthey never gave us the wifi password\b",
                        r"\bwe never got the wifi code\b",
                    ],
                    "tr": [
                        r"\bwifi şifresini söylemediler\b",
                        r"\bwifi şifresi yoktu\b",
                        r"\bşifreyi vermediler\b",
                    ],
                    "ar": [
                        r"\bما عطونا كلمة سر الواي فاي\b",
                        r"\bما قالولنا الباسورد\b",
                        r"\bما عطونا الباسورد أبداً\b",
                    ],
                    "zh": [
                        r"wifi密码没人说",
                        r"没有给我们wifi密码",
                        r"到最后都没告诉我们WiFi密码",
                    ],
                },
            ),
            
            "late_access_code": AspectRule(
                aspect_code="late_access_code",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bкод прислали поздно\b",
                        r"\bкод от двери прислали слишком поздно\b",
                        r"\bдолго ждали код\b",
                        r"\bкод от входа отправили в последний момент\b",
                    ],
                    "en": [
                        r"\bcode (arrived|was sent) late\b",
                        r"\bdoor code was sent too late\b",
                        r"\bwe had to wait for the code\b",
                        r"\bwe didn't get the code in time\b",
                    ],
                    "tr": [
                        r"\bkodu geç gönderdiler\b",
                        r"\bkapı kodu çok geç geldi\b",
                        r"\bkodu beklemek zorunda kaldık\b",
                    ],
                    "ar": [
                        r"\bالكود تأخر\b",
                        r"\bبعتولنا الكود متأخر\b",
                        r"\bضطرينا ننطر الكود\b",
                    ],
                    "zh": [
                        r"门码很晚才发",
                        r"进门的密码很晚才告诉我们",
                        r"我们还得等开门码",
                    ],
                },
            ),
        
            "had_to_figure_out": AspectRule(
                aspect_code="had_to_figure_out",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bразбираться пришлось самим\b",
                        r"\bпришлось самим разбираться как зайти\b",
                        r"\bвсё пришлось выяснять самим\b",
                        r"\bникто не объяснил, пришлось самим понять\b",
                    ],
                    "en": [
                        r"\bwe had to figure it out ourselves\b",
                        r"\bwe had to figure everything out on our own\b",
                        r"\bno help, we had to work it out ourselves\b",
                        r"\bwe had to work it out by ourselves\b",
                    ],
                    "tr": [
                        r"\bkendimiz anlamak zorunda kaldık\b",
                        r"\bher şeyi kendi başımıza çözmek zorunda kaldık\b",
                        r"\bkimse anlatmadı, biz çözdük\b",
                    ],
                    "ar": [
                        r"\bاضطرينا نكتشف لوحدنا\b",
                        r"\bما حدا شرح، فهمنا لحالنا\b",
                        r"\bكل شي لحالنا اكتشفناه\b",
                    ],
                    "zh": [
                        r"只能自己摸索",
                        r"全都要我们自己搞清楚",
                        r"没人解释 只能自己想办法",
                    ],
                },
            ),
        
            "support_during_stay_good": AspectRule(
                aspect_code="support_during_stay_good",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bподдержка во время проживания отличная\b",
                        r"\bочень отзывчивы во время проживания\b",
                        r"\bотреагировали за пару минут\b",
                        r"\bочень быстро помогали\b",
                        r"\bвсё, что просили, приносили\b",
                    ],
                    "en": [
                        r"\bvery responsive during (our|the) stay\b",
                        r"\bthey helped us right away\b",
                        r"\bthey were very helpful during our stay\b",
                        r"\banything we asked for was brought\b",
                        r"\bthey reacted in minutes\b",
                    ],
                    "tr": [
                        r"\bkonaklama sırasında çok yardımcı oldular\b",
                        r"\bhemen ilgilendiler\b",
                        r"\bisteklerimize hemen cevap verdiler\b",
                        r"\bçok hızlı yardımcı oldular\b",
                    ],
                    "ar": [
                        r"\bكتير متعاونين طول الإقامة\b",
                        r"\bساعدونا فورًا كل مرة\b",
                        r"\bكل شي طلبناه جابوه\b",
                        r"\bاستجابوا بسرعة خلال الإقامة\b",
                    ],
                    "zh": [
                        r"入住期间服务反应很快",
                        r"我们要什么他们马上送来",
                        r"住的时候他们特别配合",
                        r"随叫随到",
                    ],
                },
            ),
        
            "issue_fixed_immediately": AspectRule(
                aspect_code="issue_fixed_immediately",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bрешили сразу\b",
                        r"\bмгновенно помогли\b",
                        r"\bсразу поменяли\b",
                        r"\bпочинили сразу\b",
                        r"\bтут же исправили\b",
                    ],
                    "en": [
                        r"\bfixed it immediately\b",
                        r"\bfixed it right away\b",
                        r"\bthey replaced (it|them) immediately\b",
                        r"\bcame and fixed it in minutes\b",
                        r"\bbrought it right away\b",
                    ],
                    "tr": [
                        r"\banında hallettiler\b",
                        r"\bhemen değiştirdiler\b",
                        r"\bhemen çözdüler\b",
                        r"\bçok hızlı çözüldü\b",
                    ],
                    "ar": [
                        r"\bصلّحوه فورًا\b",
                        r"\bبدلوا مباشرة\b",
                        r"\bجابوه فورًا\b",
                        r"\bحلّوا المشكلة فورًا\b",
                    ],
                    "zh": [
                        r"马上修好了",
                        r"立刻换了新的",
                        r"马上就给我们换了",
                        r"当场就解决了",
                    ],
                },
            ),
        
            "support_during_stay_slow": AspectRule(
                aspect_code="support_during_stay_slow",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bникто не приш[её]л\b",
                        r"\bпришлось просить несколько раз\b",
                        r"\bждали очень долго пока кто-то прид(е|ё)т\b",
                        r"\bникакой реакции\b",
                        r"\bперсонал долго не реагировал\b",
                    ],
                    "en": [
                        r"\bnobody came\b",
                        r"\bwe had to ask multiple times\b",
                        r"\bno response during stay\b",
                        r"\bthey were slow to respond\b",
                        r"\bwe had to wait a long time for help\b",
                    ],
                    "tr": [
                        r"\bkimse gelmedi\b",
                        r"\bdefalarca istemek zorunda kaldık\b",
                        r"\btepki yoktu\b",
                        r"\bçok geç ilgilendiler\b",
                    ],
                    "ar": [
                        r"\bما حدا إجا\b",
                        r"\bطلبنا أكتر من مرة\b",
                        r"\bما في استجابة\b",
                        r"\bانتظرنا كثير لحدا يساعدنا\b",
                    ],
                    "zh": [
                        r"没有人来",
                        r"说了好几次才有人理",
                        r"没人回应",
                        r"等了很久才有人来帮忙",
                    ],
                },
            ),
            
            "support_ignored": AspectRule(
                aspect_code="support_ignored",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bигнорировал[аи]\s+просьбы\b",
                        r"\bигнорировали нашу просьбу\b",
                        r"\bникакой реакции\b",
                        r"\bперсонал вообще не реагировал\b",
                        r"\bнас просто проигнорировали\b",
                    ],
                    "en": [
                        r"\bignored our request\b",
                        r"\bthey ignored us\b",
                        r"\bno response at all\b",
                        r"\bthey just ignored\b",
                        r"\bthey didn't respond at all\b",
                    ],
                    "tr": [
                        r"\btaleplerimizi görmezden geldiler\b",
                        r"\bhiç cevap vermediler\b",
                        r"\byanıt bile vermediler\b",
                        r"\byokmuşuz gibi davrandılar\b",
                    ],
                    "ar": [
                        r"\bتجاهلونا\b",
                        r"\bتجاهلوا طلبنا\b",
                        r"\bما ردوا علينا أبداً\b",
                        r"\bما حدا عطانا جواب\b",
                    ],
                    "zh": [
                        r"他们完全不理我们",
                        r"直接无视我们的请求",
                        r"根本没人回应我们的请求",
                        r"就当我们不存在一样",
                    ],
                },
            ),
        
            "promised_not_done": AspectRule(
                aspect_code="promised_not_done",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bобещали и не сделали\b",
                        r"\bсказали что сделают но так и не сделали\b",
                        r"\bпообещали принести но не принесли\b",
                        r"\bобещали починить но не починили\b",
                    ],
                    "en": [
                        r"\bpromised but never did\b",
                        r"\bthey said they'd fix it but didn't\b",
                        r"\bthey said they'd bring it but never did\b",
                        r"\bthey promised to come but nobody came\b",
                    ],
                    "tr": [
                        r"\bsöz verdiler ama yapmadılar\b",
                        r"\bgetireceğiz dediler ama getirmediler\b",
                        r"\bhalledeceğiz dediler ama halletmediler\b",
                    ],
                    "ar": [
                        r"\bقالوا بيعملوا وما عملوا\b",
                        r"\bوعدوا بس ما صار شي\b",
                        r"\bقالوا رح يجيبوها وما جابوها\b",
                    ],
                    "zh": [
                        r"说会处理但没处理",
                        r"说会拿过来结果没拿",
                        r"答应了但没做到",
                        r"说明天修结果没有人来",
                    ],
                },
            ),
        
            "checkout_easy": AspectRule(
                aspect_code="checkout_easy",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bвыезд удобный\b",
                        r"\bс выездом не было проблем\b",
                        r"\bбез проблем с выездом\b",
                        r"\bчек-?аут был очень удобный\b",
                        r"\bлегко выехать\b",
                    ],
                    "en": [
                        r"\beasy checkout\b",
                        r"\bsmooth checkout\b",
                        r"\bleft with no problem\b",
                        r"\bcheckout was no hassle\b",
                        r"\bcheckout was simple\b",
                    ],
                    "tr": [
                        r"\bçıkış kolaydı\b",
                        r"\bsorunsuz ayrıldık\b",
                        r"\bcheck[- ]?out çok rahattı\b",
                        r"\bçıkış yapmak çok kolaydı\b",
                    ],
                    "ar": [
                        r"\bالخروج كان سهل\b",
                        r"\bطلعنا بدون مشاكل\b",
                        r"\bتشيك آوت كان سلس\b",
                        r"\bالإجراءات سهلة وقت الخروج\b",
                    ],
                    "zh": [
                        r"退房很方便",
                        r"退房很顺利",
                        r"走得很顺利",
                        r"退房基本没什么麻烦",
                    ],
                },
            ),
        
            "checkout_fast": AspectRule(
                aspect_code="checkout_fast",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bвыписали быстро\b",
                        r"\bчек-?аут занял минуту\b",
                        r"\bбыстро оформили выезд\b",
                        r"\bбыстро рассчитали\b",
                    ],
                    "en": [
                        r"\bcheckout was fast\b",
                        r"\bvery fast checkout\b",
                        r"\bwe checked out in a minute\b",
                        r"\bcheckout took a minute\b",
                    ],
                    "tr": [
                        r"\bcheck[- ]?out çok hızlıydı\b",
                        r"\bhemen çıkış yaptık\b",
                        r"\b1 dakikada çıkış yaptık\b",
                    ],
                    "ar": [
                        r"\bطلّعونا بسرعة\b",
                        r"\bالخروج كان سريع\b",
                        r"\bخلصنا تسجيل الخروج بسرعة\b",
                    ],
                    "zh": [
                        r"退房很快",
                        r"一分钟就退好了房",
                        r"很快就办好退房",
                    ],
                },
            ),
        
            "checkout_slow": AspectRule(
                aspect_code="checkout_slow",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bпроблемы с выездом\b",
                        r"\bвыписывали очень долго\b",
                        r"\bчек-?аут был долгий\b",
                        r"\bочень долго оформляли выезд\b",
                    ],
                    "en": [
                        r"\bcheckout was slow\b",
                        r"\bproblem with checkout\b",
                        r"\bcheckout took too long\b",
                        r"\bwe had to wait to check out\b",
                    ],
                    "tr": [
                        r"\bcheck[- ]?out yavaştı\b",
                        r"\bçıkış çok sürdü\b",
                        r"\bçıkışta sorun yaşadık\b",
                        r"\bçıkış yapmak çok uzun sürdü\b",
                    ],
                    "ar": [
                        r"\bتأخير بالخروج\b",
                        r"\bتشيك آوت معقّد\b",
                        r"\bأخد وقت طويل لنطلع\b",
                    ],
                    "zh": [
                        r"退房很慢",
                        r"退房花了很久",
                        r"退房过程很麻烦",
                        r"退房的时候拖了很久",
                    ],
                },
            ),
        
            "deposit_return_issue": AspectRule(
                aspect_code="deposit_return_issue",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bнам не вернули депозит сразу\b",
                        r"\bдепозит не вернули\b",
                        r"\bзалог долго не возвращали\b",
                        r"\bпри выезде депозит сразу не вернули\b",
                    ],
                    "en": [
                        r"\bthey didn't return the deposit\b",
                        r"\bdeposit was not returned at checkout\b",
                        r"\bthey held our deposit\b",
                        r"\bissue with deposit refund at checkout\b",
                    ],
                    "tr": [
                        r"\bdepozitoyu geri vermediler hemen\b",
                        r"\bdepozito iadesi hemen yapılmadı\b",
                        r"\bçıkışta depozitoyu alamadık\b",
                    ],
                    "ar": [
                        r"\bما رجعوا العربون بسرعة\b",
                        r"\bالعربون ما انرد وقت الخروج\b",
                        r"\bمسكوا العربون\b",
                    ],
                    "zh": [
                        r"退押金拖很久",
                        r"退房的时候押金没退",
                        r"押金不给马上退",
                        r"押金卡着不退",
                    ],
                },
            ),
        
            "checkout_no_staff": AspectRule(
                aspect_code="checkout_no_staff",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bникого не было на ресепшен[е] когда выезжали\b",
                        r"\bуехали а ресепшена нет\b",
                        r"\bне могли сдать ключ\b",
                        r"\bнекому было сдать ключ\b",
                    ],
                    "en": [
                        r"\bnobody at reception when we left\b",
                        r"\breception was empty at checkout\b",
                        r"\bwe couldn't give the key back\b",
                        r"\bno one to return the keys to\b",
                    ],
                    "tr": [
                        r"\bayrılırken resepsiyonda kimse yoktu\b",
                        r"\banahtar teslim edecek kimse yoktu\b",
                        r"\bçıkarken kimse yoktu resepsiyonda\b",
                    ],
                    "ar": [
                        r"\bما لقينا حدا بالاستقبال وقت رحنا\b",
                        r"\bما كان في حدا نستلم منه المفتاح\b",
                        r"\bما قدرنا نسلم المفتاح لأنه ما كان في حدا\b",
                    ],
                    "zh": [
                        r"退房的时候前台没人",
                        r"走的时候前台是空的",
                        r"我们都找不到人交钥匙",
                        r"没人收钥匙",
                    ],
                },
            ),
            
            "fresh_bedding": AspectRule(
                aspect_code="fresh_bedding",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bсвежее бель[её]\b",
                        r"\bсвежая постель\b",
                        r"\bпостель чистая\b",
                        r"\bчистое постельное бель[её]\b",
                        r"\bсвежее чистое бель[её]\b",
                    ],
                    "en": [
                        r"\bfresh bedding\b",
                        r"\bclean sheets\b",
                        r"\bfresh sheets\b",
                        r"\bthe bed linen was fresh\b",
                    ],
                    "tr": [
                        r"\byeni çarşaf\b",
                        r"\btemiz çarşaf\b",
                        r"\bçarşaflar tertemizdi\b",
                        r"\bçarşaflar yeniydi\b",
                    ],
                    "ar": [
                        r"\bمفارش نظيفة\b",
                        r"\bشرشف نظيف\b",
                        r"\bالشراشف كانت نضيفة وطازة\b",
                        r"\bملايات \S+ نضاف\b",
                    ],
                    "zh": [
                        r"床单很干净",
                        r"新的床单",
                        r"床上用品是干净的",
                        r"床单是新的/换过的",
                    ],
                },
            ),
        
            "no_dust_surfaces": AspectRule(
                aspect_code="no_dust_surfaces",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bникакой пыли\b",
                        r"\bпыль отсутствовала\b",
                        r"\bповерхности без пыли\b",
                        r"\bнигде не было пыли\b",
                    ],
                    "en": [
                        r"\bno dust\b",
                        r"\bno dust on surfaces\b",
                        r"\bsurfaces were dust[- ]?free\b",
                    ],
                    "tr": [
                        r"\btoz yoktu\b",
                        r"\bhiç toz yoktu\b",
                        r"\byüzeylerde toz yoktu\b",
                    ],
                    "ar": [
                        r"\bمافي غبار\b",
                        r"\bالسطح كان نظيف بدون غبرة\b",
                        r"\bما كان في ولا غبرة\b",
                    ],
                    "zh": [
                        r"没有灰尘",
                        r"到处都很干净没有灰",
                        r"表面都没有灰尘",
                    ],
                },
            ),
        
            "floor_clean": AspectRule(
                aspect_code="floor_clean",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bполы чистые\b",
                        r"\bчистый пол\b",
                        r"\bпол был чистый\b",
                    ],
                    "en": [
                        r"\bfloors were clean\b",
                        r"\bclean floor\b",
                        r"\bthe floor was clean\b",
                    ],
                    "tr": [
                        r"\bzemin temizdi\b",
                        r"\byerler tertemizdi\b",
                        r"\byer çok temizdi\b",
                    ],
                    "ar": [
                        r"\bالأرض نظيفة\b",
                        r"\bالأرضية نظيفة\b",
                        r"\bالأرض كانت نضيفة\b",
                    ],
                    "zh": [
                        r"地板很干净",
                        r"地面很干净",
                        r"地板打扫得很干净",
                    ],
                },
            ),
        
            "dusty_surfaces": AspectRule(
                aspect_code="dusty_surfaces",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bпыль на поверхностях\b",
                        r"\bвсё в пыли\b",
                        r"\bпыльно\b",
                        r"\bпыль на полках\b",
                    ],
                    "en": [
                        r"\bdust on surfaces\b",
                        r"\bdusty surfaces\b",
                        r"\bvery dusty\b",
                        r"\bdust everywhere\b",
                    ],
                    "tr": [
                        r"\btoz vardı\b",
                        r"\byüzeyler tozluydu\b",
                        r"\bher yer tozluydu\b",
                        r"\btoz içindeydi\b",
                    ],
                    "ar": [
                        r"\bغبار على السطوح\b",
                        r"\bكلو مغبر\b",
                        r"\bالسطح مليان غبرة\b",
                        r"\bفي غبرة بكل مكان\b",
                    ],
                    "zh": [
                        r"到处是灰",
                        r"表面上都是灰尘",
                        r"很脏有灰尘",
                        r"家具上全是灰",
                    ],
                },
            ),
        
            "sticky_surfaces": AspectRule(
                aspect_code="sticky_surfaces",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bлипкий пол\b",
                        r"\bлипкий стол\b",
                        r"\bпол липкий\b",
                        r"\bстол липкий\b",
                    ],
                    "en": [
                        r"\bsticky floor\b",
                        r"\bsticky table\b",
                        r"\bfloor was sticky\b",
                        r"\btable was sticky\b",
                    ],
                    "tr": [
                        r"\byapış yapış zemin\b",
                        r"\byapış yapış masa\b",
                        r"\bzemin yapış yapıştı\b",
                        r"\bmasa yapış yapıştı\b",
                    ],
                    "ar": [
                        r"\bالأرض لزجة\b",
                        r"\bالطاولة لزقة\b",
                        r"\bالأرضية كانت عم تلزق\b",
                        r"\bالطاولة كانت بتلزق\b",
                    ],
                    "zh": [
                        r"地板黏黏的",
                        r"桌子黏黏的",
                        r"地板很黏",
                        r"桌面很黏",
                    ],
                },
            ),
            
            "stained_bedding": AspectRule(
                aspect_code="stained_bedding",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bпятна на постел[еи]\b",
                        r"\bгрязная постель\b",
                        r"\bгрязное постельное бель[её]\b",
                        r"\bпостель в пятнах\b",
                        r"\bбель[её] в пятнах\b",
                    ],
                    "en": [
                        r"\bstains on the bed\b",
                        r"\bstained sheets\b",
                        r"\bdirty sheets\b",
                        r"\bbedding had stains\b",
                    ],
                    "tr": [
                        r"\bçarşafta leke vardı\b",
                        r"\bçarşaf lekeli\b",
                        r"\bçarşaf kirliydi\b",
                        r"\byatak örtüsü lekeli\b",
                    ],
                    "ar": [
                        r"\bبقع على الشرشف\b",
                        r"\bالشرشف وسخ\b",
                        r"\bالفرشة عليها بقع\b",
                        r"\bالملاية مبقعة\b",
                    ],
                    "zh": [
                        r"床单有污渍",
                        r"被单有污渍",
                        r"床上有脏污",
                        r"床上用品有污渍",
                    ],
                },
            ),
        
            "hair_on_bed": AspectRule(
                aspect_code="hair_on_bed",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bволосы на кроват[еию]\b",
                        r"\bволосы на подушк[е]\b",
                        r"\bволосы на постели\b",
                        r"\bнашли волосы в постели\b",
                    ],
                    "en": [
                        r"\bhair on the bed\b",
                        r"\bhair on the pillow\b",
                        r"\bhair in the sheets\b",
                        r"\bfound hair in the bed\b",
                    ],
                    "tr": [
                        r"\byatakta saç vardı\b",
                        r"\byastıkta saç vardı\b",
                        r"\bçarşafın üstünde saç vardı\b",
                    ],
                    "ar": [
                        r"\bشعر عالتخت\b",
                        r"\bشعر عالشرشف\b",
                        r"\bشعر عالمخدة\b",
                        r"\bلقينا شعر بالسرير\b",
                    ],
                    "zh": [
                        r"床上有头发",
                        r"枕头上有头发",
                        r"床单上有头发",
                        r"被子上有头发",
                    ],
                },
            ),
        
            "used_towels_left": AspectRule(
                aspect_code="used_towels_left",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bгрязные полотенца от предыдущих\b",
                        r"\bгрязные полотенца остались\b",
                        r"\bгрязные полотенца лежали\b",
                        r"\bполотенца не поменяли после прошлых гостей\b",
                    ],
                    "en": [
                        r"\bold towels left\b",
                        r"\bdirty towels left\b",
                        r"\bused towels from previous guest\b",
                        r"\bprevious guest's towels were still there\b",
                    ],
                    "tr": [
                        r"\beski havlular duruyordu\b",
                        r"\bkullanılmış havlular bırakılmıştı\b",
                        r"\bönceki misafirin havluları hâlâ oradaydı\b",
                    ],
                    "ar": [
                        r"\bمناشف وسخة من قبل\b",
                        r"\bالمناشف المستعملة ظلّت بالمكان\b",
                        r"\bمناشف الضيف السابق بعدها موجودة\b",
                    ],
                    "zh": [
                        r"旧的毛巾还在",
                        r"还有上个客人的毛巾",
                        r"脏毛巾还没拿走",
                        r"之前的用过的毛巾还在房间",
                    ],
                },
            ),
        
            "crumbs_left": AspectRule(
                aspect_code="crumbs_left",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bкрошки на столе\b",
                        r"\bкрошки везде\b",
                        r"\bвсё в крошках\b",
                        r"\bоставили крошки после прошлых гостей\b",
                    ],
                    "en": [
                        r"\bcrumbs everywhere\b",
                        r"\bcrumbs on the table\b",
                        r"\btable had crumbs\b",
                        r"\bfull of crumbs\b",
                    ],
                    "tr": [
                        r"\bher yerde kırıntı vardı\b",
                        r"\bmasada kırıntılar vardı\b",
                        r"\bmasa kırıntı içindeydi\b",
                    ],
                    "ar": [
                        r"\bفتات على الطاولة\b",
                        r"\bفتافيت بكل مكان\b",
                        r"\bالطاولة كلها فتافيت\b",
                    ],
                    "zh": [
                        r"桌上都是碎屑",
                        r"到处都是碎屑",
                        r"桌子上全是渣",
                        r"房间里全是食物碎渣",
                    ],
                },
            ),
        
            "bathroom_clean_on_arrival": AspectRule(
                aspect_code="bathroom_clean_on_arrival",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bванная чистая\b",
                        r"\bсанузел чистый\b",
                        r"\bдуш чистый\b",
                        r"\bчистая раковина\b",
                        r"\bвсё блестит\b",
                    ],
                    "en": [
                        r"\bbathroom was clean\b",
                        r"\bbathroom spotless\b",
                        r"\bshower was very clean\b",
                        r"\bsink was clean\b",
                        r"\bthe bathroom was sparkling\b",
                    ],
                    "tr": [
                        r"\bbanyo temizdi\b",
                        r"\bduş çok temizdi\b",
                        r"\blavabo tertemizdi\b",
                        r"\bher yer pırıl pırıldı\b",
                    ],
                    "ar": [
                        r"\bالحمام نظيف\b",
                        r"\bالدوش نظيف\b",
                        r"\bمغسلة نظيفة\b",
                        r"\bكلو نظيف بالحمام\b",
                    ],
                    "zh": [
                        r"卫生间很干净",
                        r"浴室很干净",
                        r"淋浴很干净",
                        r"洗手池很干净",
                        r"浴室特别干净",
                    ],
                },
            ),
            
            "no_mold_visible": AspectRule(
                aspect_code="no_mold_visible",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bникакой плесени\b",
                        r"\bбез плесени\b",
                        r"\bне было плесени\b",
                        r"\bчистые швы без плесени\b",
                    ],
                    "en": [
                        r"\bno mold\b",
                        r"\bno mildew\b",
                        r"\bno sign of mold\b",
                        r"\btiles were clean with no mold\b",
                    ],
                    "tr": [
                        r"\bküf yoktu\b",
                        r"\bhiç küf yoktu\b",
                        r"\bfayanslarda küf yoktu\b",
                    ],
                    "ar": [
                        r"\bمافي عفن\b",
                        r"\bما في عفن أبداً\b",
                        r"\bما كان في ولا عفن بالحمام\b",
                    ],
                    "zh": [
                        r"没有霉",
                        r"没有霉斑",
                        r"瓷砖缝很干净没有霉",
                    ],
                },
            ),
        
            "sink_clean": AspectRule(
                aspect_code="sink_clean",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bчистая раковина\b",
                        r"\bраковина чистая\b",
                        r"\bраковина была чистой\b",
                    ],
                    "en": [
                        r"\bsink was clean\b",
                        r"\bvery clean sink\b",
                        r"\bthe sink was spotless\b",
                    ],
                    "tr": [
                        r"\blavabo tertemizdi\b",
                        r"\blavabo temizdi\b",
                        r"\btemiz bir lavabo vardı\b",
                    ],
                    "ar": [
                        r"\bمغسلة نظيفة\b",
                        r"\bالمغسلة كانت نظيفة\b",
                        r"\bالمغسلة نضيفة كتير\b",
                    ],
                    "zh": [
                        r"洗手池很干净",
                        r"洗手池特别干净",
                        r"洗手池没有污垢",
                    ],
                },
            ),
        
            "shower_clean": AspectRule(
                aspect_code="shower_clean",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bдуш чистый\b",
                        r"\bкабина была чистой\b",
                        r"\bстекло душа чистое\b",
                        r"\bникакого налёта в душе\b",
                    ],
                    "en": [
                        r"\bshower was very clean\b",
                        r"\bshower was clean\b",
                        r"\bthe shower was spotless\b",
                        r"\bno grime in the shower\b",
                    ],
                    "tr": [
                        r"\bduş çok temizdi\b",
                        r"\bduş temizdi\b",
                        r"\bduş kabini tertemizdi\b",
                    ],
                    "ar": [
                        r"\bالدوش نظيف\b",
                        r"\bالدوش كان نضيف كتير\b",
                        r"\bالشاور نظيف وما عليه أوساخ\b",
                    ],
                    "zh": [
                        r"淋浴很干净",
                        r"淋浴间很干净",
                        r"淋浴玻璃很干净",
                    ],
                },
            ),
        
            "bathroom_dirty_on_arrival": AspectRule(
                aspect_code="bathroom_dirty_on_arrival",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bгрязный санузел\b",
                        r"\bгрязный унитаз\b",
                        r"\bгрязный туалет\b",
                        r"\bгрязная раковина\b",
                        r"\bгрязный слив\b",
                    ],
                    "en": [
                        r"\bdirty bathroom\b",
                        r"\bdirty toilet\b",
                        r"\bdirty sink\b",
                        r"\bthe bathroom was dirty on arrival\b",
                    ],
                    "tr": [
                        r"\bbanyo kirliydi\b",
                        r"\btuvalet kirliydi\b",
                        r"\blavabo kirliydi\b",
                    ],
                    "ar": [
                        r"\bحمام وسخ\b",
                        r"\bتواليت وسخ\b",
                        r"\bمغسلة وسخة\b",
                    ],
                    "zh": [
                        r"卫生间很脏",
                        r"厕所很脏",
                        r"洗手池很脏",
                        r"刚入住时卫生间很脏",
                    ],
                },
            ),
        
            "hair_in_shower": AspectRule(
                aspect_code="hair_in_shower",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bволосы в душе\b",
                        r"\bволосы в сливе душа\b",
                        r"\bволосы в ванной\b",
                    ],
                    "en": [
                        r"\bhair in the shower\b",
                        r"\bhair in the drain\b",
                        r"\bhair in the bathroom floor\b",
                    ],
                    "tr": [
                        r"\bduşta saç vardı\b",
                        r"\bduş giderinde saç vardı\b",
                        r"\bduşta saç telleri vardı\b",
                    ],
                    "ar": [
                        r"\bشعر في الدوش\b",
                        r"\bشعر بالمصرف\b",
                        r"\bشعر بالحمام\b",
                    ],
                    "zh": [
                        r"淋浴间有头发",
                        r"下水口有头发",
                        r"浴室地上有头发",
                    ],
                },
            ),
            
            "hair_in_sink": AspectRule(
                aspect_code="hair_in_sink",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bволосы в раковине\b",
                        r"\bволосы в умывальнике\b",
                        r"\bраковина с волосами\b",
                    ],
                    "en": [
                        r"\bhair in the sink\b",
                        r"\bhair in the basin\b",
                        r"\bthe sink had hair\b",
                    ],
                    "tr": [
                        r"\blavaboda saç vardı\b",
                        r"\blavaboda saç telleri vardı\b",
                        r"\blavabo saç doluydu\b",
                    ],
                    "ar": [
                        r"\bشعر في المغسلة\b",
                        r"\bالمغسلة فيها شعر\b",
                        r"\bشعر عالحوض\b",
                    ],
                    "zh": [
                        r"洗手池里有头发",
                        r"洗手池里面全是头发",
                        r"洗手池有很多头发没清理",
                    ],
                },
            ),
        
            "mold_in_shower": AspectRule(
                aspect_code="mold_in_shower",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bплесень в душе\b",
                        r"\bч(е|ё)рная плесень\b",
                        r"\bплесень в швах\b",
                        r"\bплесень на плитке в душе\b",
                    ],
                    "en": [
                        r"\bmold in the shower\b",
                        r"\bblack mold\b",
                        r"\bmildew in the shower\b",
                        r"\bmold in the tiles\b",
                    ],
                    "tr": [
                        r"\bduşta küf vardı\b",
                        r"\bfayanslarda küf\b",
                        r"\bduş kabininde küf vardı\b",
                    ],
                    "ar": [
                        r"\bعفن في الدوش\b",
                        r"\bعفن أسود\b",
                        r"\bكأنه في عفن حوالين البلاط\b",
                    ],
                    "zh": [
                        r"淋浴间有霉",
                        r"黑色霉斑",
                        r"瓷砖发霉",
                        r"浴室有霉斑",
                    ],
                },
            ),
        
            "limescale_stains": AspectRule(
                aspect_code="limescale_stains",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bнал[её]т\b",
                        r"\bизвестковый нал[её]т\b",
                        r"\bизвестковый осадок\b",
                        r"\bржавчина\b",
                        r"\bржавые потёки\b",
                    ],
                    "en": [
                        r"\blimescale\b",
                        r"\blimescale stains\b",
                        r"\brust stains\b",
                        r"\brust marks\b",
                        r"\bcalcium buildup\b",
                    ],
                    "tr": [
                        r"\bkireç lekeleri\b",
                        r"\bkireç kalıntısı vardı\b",
                        r"\bpas lekeleri\b",
                        r"\bpas izi vardı\b",
                    ],
                    "ar": [
                        r"\bرواسب كلس\b",
                        r"\bترسّبات كلس\b",
                        r"\bبقع صدأ\b",
                        r"\bآثار صدأ\b",
                    ],
                    "zh": [
                        r"有水垢",
                        r"有水垢痕迹",
                        r"有锈迹",
                        r"有锈斑",
                    ],
                },
            ),
        
            "sewage_smell_bathroom": AspectRule(
                aspect_code="sewage_smell_bathroom",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bвоняет из туалета\b",
                        r"\bзапах канализации в ванной\b",
                        r"\bвонь из слива\b",
                        r"\bзапах из труб\b",
                    ],
                    "en": [
                        r"\bsewage smell in the bathroom\b",
                        r"\bbad smell from the toilet\b",
                        r"\bsewer smell\b",
                        r"\bsmelled like sewage in the bathroom\b",
                    ],
                    "tr": [
                        r"\btuvaletten koku geliyordu\b",
                        r"\blağım kokusu vardı\b",
                        r"\blağım gibi kokuyordu\b",
                        r"\bkanalizasyon kokusu\b",
                    ],
                    "ar": [
                        r"\bريحة مجاري من الحمام\b",
                        r"\bريحة مجاري\b",
                        r"\bريحة تواليت\b",
                        r"\bريحة صرف\b",
                    ],
                    "zh": [
                        r"卫生间有下水道味",
                        r"厕所味很重",
                        r"有下水道的味道",
                        r"浴室有一股下水道味道",
                    ],
                },
            ),
        
            "housekeeping_regular": AspectRule(
                aspect_code="housekeeping_regular",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bубирали каждый день\b",
                        r"\bуборка ежедневно\b",
                        r"\bприходили убирать\b",
                        r"\bрегулярная уборка\b",
                        r"\bкаждый день наводили порядок\b",
                    ],
                    "en": [
                        r"\bthey cleaned every day\b",
                        r"\bdaily cleaning\b",
                        r"\bthey came to clean\b",
                        r"\bthe room was cleaned daily\b",
                    ],
                    "tr": [
                        r"\bher gün temizlediler\b",
                        r"\bgünlük temizlik yapıldı\b",
                        r"\bgeldiler temizlediler\b",
                        r"\bhazır odayı her gün toparladılar\b",
                    ],
                    "ar": [
                        r"\bنظفوا كل يوم\b",
                        r"\bيجوا ينظفوا كل يوم\b",
                        r"\bيجوا ينضفوا بشكل يومي\b",
                        r"\bنضافة يومية\b",
                    ],
                    "zh": [
                        r"每天都会打扫",
                        r"每天有人来打扫",
                        r"每天都来打扫房间",
                        r"每天整理房间",
                    ],
                },
            ),
            
            "trash_taken_out": AspectRule(
                aspect_code="trash_taken_out",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bвыносили мусор\b",
                        r"\bмусор выносили регулярно\b",
                        r"\bмусор забирали каждый день\b",
                        r"\bмусор сразу вынесли\b",
                    ],
                    "en": [
                        r"\bthey took out the trash\b",
                        r"\btrash was taken\b",
                        r"\bthey removed the trash every day\b",
                        r"\bthey emptied the bin every day\b",
                    ],
                    "tr": [
                        r"\bçöpü aldılar\b",
                        r"\bçöpü topladılar\b",
                        r"\bher gün çöp alındı\b",
                    ],
                    "ar": [
                        r"\bشالوا الزبالة\b",
                        r"\bشالوا الزبالة كل يوم\b",
                        r"\bفضّوا سلة الزبالة كل يوم\b",
                    ],
                    "zh": [
                        r"垃圾每天都拿走",
                        r"垃圾有拿走",
                        r"每天都有人倒垃圾",
                        r"垃圾桶每天都会清",
                    ],
                },
            ),
        
            "bed_made": AspectRule(
                aspect_code="bed_made",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bзастилали кровать\b",
                        r"\bкровать заправляли\b",
                        r"\bкаждый день заправляли кровать\b",
                        r"\bпостель аккуратно застелена\b",
                    ],
                    "en": [
                        r"\bthey made the bed\b",
                        r"\bbed was made every day\b",
                        r"\bthey fixed the bed nicely\b",
                        r"\bthe bed was neatly made\b",
                    ],
                    "tr": [
                        r"\byatağı düzelttiler\b",
                        r"\byatağı her gün topladılar\b",
                        r"\byatağı düzenlediler\b",
                    ],
                    "ar": [
                        r"\bسووا السرير\b",
                        r"\bكل يوم كانوا يرتبوا السرير\b",
                        r"\bالسرير كان مترتب\b",
                    ],
                    "zh": [
                        r"床每天都有整理",
                        r"床整理好了",
                        r"每天都会把床铺好",
                    ],
                },
            ),
        
            "housekeeping_missed": AspectRule(
                aspect_code="housekeeping_missed",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bне убирали\b",
                        r"\bуборки не было\b",
                        r"\bникто не убирался\b",
                        r"\bвообще не приходили убирать\b",
                    ],
                    "en": [
                        r"\bnobody cleaned\b",
                        r"\bno cleaning during (our|the) stay\b",
                        r"\bthey never came to clean\b",
                        r"\bno housekeeping\b",
                    ],
                    "tr": [
                        r"\btemizlik yapılmadı\b",
                        r"\bhiç temizlenmedi\b",
                        r"\bkimse temizlemeye gelmedi\b",
                    ],
                    "ar": [
                        r"\bما حد نظف\b",
                        r"\bما في تنظيف طول الإقامة\b",
                        r"\bما إجو ينظفوا أبداً\b",
                    ],
                    "zh": [
                        r"没人打扫",
                        r"住着期间没有打扫",
                        r"从来没人来打扫",
                    ],
                },
            ),
        
            "trash_not_taken": AspectRule(
                aspect_code="trash_not_taken",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bмусор не выносили\b",
                        r"\bмусор так и остался\b",
                        r"\bурна переполнена\b",
                        r"\bмусор накопился\b",
                    ],
                    "en": [
                        r"\btrash was not taken\b",
                        r"\btrash piled up\b",
                        r"\bbin was never emptied\b",
                        r"\boverflowing trash\b",
                    ],
                    "tr": [
                        r"\bçöpü almadılar\b",
                        r"\bçöp birikti\b",
                        r"\bçöp kovası dolup taştı\b",
                    ],
                    "ar": [
                        r"\bالزبالة ظلت\b",
                        r"\bالزبالة تراكمت\b",
                        r"\bسلة الزبالة مليانة وما حد شالها\b",
                    ],
                    "zh": [
                        r"垃圾没人倒",
                        r"垃圾越积越多",
                        r"垃圾桶一直没人清",
                    ],
                },
            ),
        
            "bed_not_made": AspectRule(
                aspect_code="bed_not_made",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bкровать не заправили\b",
                        r"\bкровать так и не заправили\b",
                        r"\bникто не застелил кровать\b",
                    ],
                    "en": [
                        r"\bbed was never made\b",
                        r"\bthey didn't make the bed\b",
                        r"\bnobody made the bed\b",
                    ],
                    "tr": [
                        r"\byatağı düzeltmediler\b",
                        r"\byatağı hiç toplamadılar\b",
                        r"\byatağı hiç yapmadılar\b",
                    ],
                    "ar": [
                        r"\bما سووا السرير\b",
                        r"\bالسرير ما رتبوه\b",
                        r"\bما حدا رتب السرير\b",
                    ],
                    "zh": [
                        r"床也不整理",
                        r"床一直没整理",
                        r"没有人整理床铺",
                    ],
                },
            ),
            
            "had_to_request_cleaning": AspectRule(
                aspect_code="had_to_request_cleaning",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bпришлось просить уборку\b",
                        r"\bуборку пришлось просить\b",
                        r"\bприходилось напоминать про уборку\b",
                        r"\bуборку сделали только после просьбы\b",
                    ],
                    "en": [
                        r"\bwe had to ask for cleaning\b",
                        r"\bwe had to request cleaning\b",
                        r"\bthey only cleaned after we asked\b",
                        r"\bwe had to remind them to clean\b",
                    ],
                    "tr": [
                        r"\btemizlik için istemek zorunda kaldık\b",
                        r"\bhatırlatmak zorunda kaldık\b",
                        r"\bancak söyleyince temizlediler\b",
                    ],
                    "ar": [
                        r"\bاضطرينا نطلب تنظيف\b",
                        r"\bاضطرينا نذكرهم ينضفوا\b",
                        r"\bما نضفوا إلا بعد ما طلبنا\b",
                    ],
                    "zh": [
                        r"我们得自己要求打扫",
                        r"我们得提醒他们才来打扫",
                        r"他们只有我们说了才来打扫",
                    ],
                },
            ),
        
            "dirt_accumulated": AspectRule(
                aspect_code="dirt_accumulated",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bгрязь копилась\b",
                        r"\bгрязно оставалось\b",
                        r"\bстановилось всё грязнее\b",
                        r"\bгрязь просто накапливалась\b",
                    ],
                    "en": [
                        r"\bit stayed dirty\b",
                        r"\bit got dirty and stayed that way\b",
                        r"\bthe dirt just built up\b",
                        r"\bkept getting dirtier\b",
                    ],
                    "tr": [
                        r"\bkir kaldı\b",
                        r"\bkir birikti\b",
                        r"\bodada kir birikmeye başladı\b",
                        r"\btemizlenmeyince daha da kirlendi\b",
                    ],
                    "ar": [
                        r"\bظل وسخ\b",
                        r"\bصار وسخ وما نظفوه\b",
                        r"\bالوسخ عم يتراكم\b",
                    ],
                    "zh": [
                        r"一直很脏",
                        r"越住越脏",
                        r"房间越来越脏也没人打扫",
                        r"脏东西越积越多",
                    ],
                },
            ),
        
            "towels_changed": AspectRule(
                aspect_code="towels_changed",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bменяли полотенца\b",
                        r"\bполотенца меняли регулярно\b",
                        r"\bпринесли чистые полотенца\b",
                        r"\bпринесли новые полотенца\b",
                    ],
                    "en": [
                        r"\bthey changed the towels\b",
                        r"\bfresh towels\b",
                        r"\bclean towels were provided\b",
                        r"\bthey brought clean towels\b",
                    ],
                    "tr": [
                        r"\bhavluları değiştirdiler\b",
                        r"\btemiz havlu getirdiler\b",
                        r"\bher gün havlu değişti\b",
                    ],
                    "ar": [
                        r"\bمناشف نظيفة\b",
                        r"\bجابوا مناشف جديدة\b",
                        r"\bيغيروا المناشف\b",
                        r"\bبدلوا المناشف\b",
                    ],
                    "zh": [
                        r"给了新的毛巾",
                        r"毛巾很干净",
                        r"每天都有换毛巾",
                        r"有送干净的毛巾",
                    ],
                },
            ),
        
            "fresh_towels_fast": AspectRule(
                aspect_code="fresh_towels_fast",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bпринесли чистые полотенца сразу\b",
                        r"\bмоментально принесли свежие полотенца\b",
                        r"\bполотенца принесли очень быстро\b",
                    ],
                    "en": [
                        r"\bbrought new towels right away\b",
                        r"\bfresh towels were brought immediately\b",
                        r"\bthey brought clean towels in minutes\b",
                    ],
                    "tr": [
                        r"\bhemen yeni havlu getirdiler\b",
                        r"\bçok hızlı temiz havlu getirdiler\b",
                        r"\bisteyince anında havlu geldi\b",
                    ],
                    "ar": [
                        r"\bجابوا مناشف جديدة فورًا\b",
                        r"\bبدّلوا المناشف فورًا\b",
                        r"\bعطونا مناشف نضيفة بسرعة\b",
                    ],
                    "zh": [
                        r"马上送了干净的毛巾",
                        r"立刻给我们新的毛巾",
                        r"说了以后很快就拿干净毛巾来了",
                    ],
                },
            ),
        
            "linen_changed": AspectRule(
                aspect_code="linen_changed",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bсменили постельное бель[её]\b",
                        r"\bпоменяли бель[её]\b",
                        r"\bновое бель[её]\b",
                        r"\bпостель поменяли\b",
                    ],
                    "en": [
                        r"\bchanged the sheets\b",
                        r"\bfresh bedding\b",
                        r"\bnew bed linen\b",
                        r"\bthey replaced the bedding\b",
                    ],
                    "tr": [
                        r"\bçarşafları değiştirdiler\b",
                        r"\btemiz çarşaf getirdiler\b",
                        r"\byeni çarşaf serildi\b",
                    ],
                    "ar": [
                        r"\bغيروا الشراشف\b",
                        r"\bشرشف نظيف\b",
                        r"\bجابوا ملايات نظيفة\b",
                        r"\bبدّلوا الفرش\b",
                    ],
                    "zh": [
                        r"换了床单",
                        r"换了新的床单",
                        r"床单是干净的/刚换的",
                        r"把床上用品都换新的了",
                    ],
                },
            ),
        
            "amenities_restocked": AspectRule(
                aspect_code="amenities_restocked",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bпополняли воду\b",
                        r"\bпринесли воду\b",
                        r"\bпополняли туалетную бумагу\b",
                        r"\bпринесли туалетную бумагу\b",
                        r"\bпринесли мыло\b",
                        r"\bпополняли шампунь\b",
                    ],
                    "en": [
                        r"\brestocked toiletries\b",
                        r"\bgave us toilet paper\b",
                        r"\bbrought water\b",
                        r"\breplaced the soap\b",
                        r"\brestocked shampoo\b",
                    ],
                    "tr": [
                        r"\btuvalet kağıdı getirdiler\b",
                        r"\bşampuan yenilediler\b",
                        r"\bsu bıraktılar\b",
                        r"\beksikleri tamamladılar\b",
                    ],
                    "ar": [
                        r"\bجابوا ورق تواليت\b",
                        r"\bجابوا صابون\b",
                        r"\bجابوا مي\b",
                        r"\bرجعوا عبّوا الشامبو\b",
                    ],
                    "zh": [
                        r"补了卫生纸",
                        r"补了洗浴用品",
                        r"补了水",
                        r"又给我们新的香皂洗发水",
                    ],
                },
            ),
            
            "towels_dirty": AspectRule(
                aspect_code="towels_dirty",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bполотенца грязные\b",
                        r"\bгрязные полотенца\b",
                        r"\bполотенца были грязные\b",
                        r"\bполотенца (какие-то )?грязные на вид\b",
                    ],
                    "en": [
                        r"\bdirty towels\b",
                        r"\btowels were dirty\b",
                        r"\bthe towels looked dirty\b",
                        r"\bthe towels weren't clean\b",
                    ],
                    "tr": [
                        r"\bhavlular kirliydi\b",
                        r"\bhavlular temiz değildi\b",
                        r"\bkirli havlu verdiler\b",
                    ],
                    "ar": [
                        r"\bمناشف وسخة\b",
                        r"\bالمناشف كانت وسخة\b",
                        r"\bعطونا مناشف مو نضيفة\b",
                    ],
                    "zh": [
                        r"毛巾很脏",
                        r"毛巾不干净",
                        r"给我们的毛巾是脏的",
                    ],
                },
            ),
        
            "towels_stained": AspectRule(
                aspect_code="towels_stained",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bпятна на полотенцах\b",
                        r"\bполотенца в пятнах\b",
                        r"\bполотенца были в пятнах\b",
                    ],
                    "en": [
                        r"\bstains on the towels\b",
                        r"\bstained towels\b",
                        r"\btowels had stains\b",
                    ],
                    "tr": [
                        r"\bhavluda leke vardı\b",
                        r"\bhavlular lekeli\b",
                        r"\blekeli havlu verdiler\b",
                    ],
                    "ar": [
                        r"\bبقع على المناشف\b",
                        r"\bالمناشف مبقعة\b",
                        r"\bالمناشف عليها بقع\b",
                    ],
                    "zh": [
                        r"毛巾有污渍",
                        r"毛巾上有斑渍",
                        r"毛巾上都是印子",
                    ],
                },
            ),
        
            "towels_smell": AspectRule(
                aspect_code="towels_smell",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bполотенца пахли\b",
                        r"\bнеприятный запах от полотенец\b",
                        r"\bполотенца воняли\b",
                        r"\bзатхлый запах от полотенец\b",
                    ],
                    "en": [
                        r"\btowels smelled bad\b",
                        r"\btowels smelled\b",
                        r"\bbad smell from the towels\b",
                        r"\bthe towels had a bad smell\b",
                    ],
                    "tr": [
                        r"\bhavlu kötü kokuyordu\b",
                        r"\bhavlularda kötü koku vardı\b",
                        r"\bhavlular kokuyordu\b",
                    ],
                    "ar": [
                        r"\bريحة المناشف سيئة\b",
                        r"\bالمناشف ريحتها مو طيبة\b",
                        r"\bالمناشف كانت عم تشم\b",
                    ],
                    "zh": [
                        r"毛巾有味道",
                        r"毛巾有股怪味",
                        r"毛巾闻起来很臭",
                    ],
                },
            ),
        
            "towels_not_changed": AspectRule(
                aspect_code="towels_not_changed",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bне меняли полотенца\b",
                        r"\bполотенца не меняли\b",
                        r"\bполотенца не сменили\b",
                        r"\bнам не поменяли полотенца\b",
                    ],
                    "en": [
                        r"\bthey never changed the towels\b",
                        r"\bno fresh towels\b",
                        r"\btowels were never replaced\b",
                        r"\bthey didn't replace the towels\b",
                    ],
                    "tr": [
                        r"\bhavlu değiştirmediler\b",
                        r"\bhiç yeni havlu getirmediler\b",
                        r"\bhavluları yenilemediler\b",
                    ],
                    "ar": [
                        r"\bما غيروا المناشف\b",
                        r"\bما جابوا مناشف جديدة\b",
                        r"\bما بدّلوا المناشف\b",
                    ],
                    "zh": [
                        r"毛巾一直没换",
                        r"没有给我们新的毛巾",
                        r"毛巾都不换",
                    ],
                },
            ),
        
            "linen_not_changed": AspectRule(
                aspect_code="linen_not_changed",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bбель[её] не поменяли\b",
                        r"\bне поменяли бель[её]\b",
                        r"\bпостель не меняли\b",
                        r"\bпростыни так и не сменили\b",
                    ],
                    "en": [
                        r"\bsheets not changed\b",
                        r"\bthey didn't change the sheets\b",
                        r"\bthe bedding was never replaced\b",
                        r"\bno fresh sheets\b",
                    ],
                    "tr": [
                        r"\bçarşaf değiştirmediler\b",
                        r"\bçarşaflar hiç değişmedi\b",
                        r"\byatak çarşafını yenilemediler\b",
                    ],
                    "ar": [
                        r"\bما غيروا الشراشف\b",
                        r"\bما غيروا الفرش\b",
                        r"\bنفس الشرشف طول الإقامة\b",
                    ],
                    "zh": [
                        r"床单没换",
                        r"床单一直没换",
                        r"一直用同一套床单",
                    ],
                },
            ),
            
            "no_restock": AspectRule(
                aspect_code="no_restock",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bне пополняли воду\b",
                        r"\bводу не пополняли\b",
                        r"\bне принесли туалетную бумагу\b",
                        r"\bне пополняли мыло\b",
                        r"\bне пополняли шампунь\b",
                        r"\bтуалетной бумаги не было\b",
                        r"\bне принесли воду\b",
                    ],
                    "en": [
                        r"\bno toilet paper\b",
                        r"\bthey didn't restock soap\b",
                        r"\bthey didn't restock shampoo\b",
                        r"\bno water refill\b",
                        r"\bthey didn't bring more water\b",
                        r"\bthey didn't replenish toiletries\b",
                    ],
                    "tr": [
                        r"\btuvalet kağıdı yoktu\b",
                        r"\byenilemediler\b",
                        r"\bşampuan yenilenmedi\b",
                        r"\bsu getirmediler\b",
                        r"\beksikler tamamlanmadı\b",
                    ],
                    "ar": [
                        r"\bما رجعوا ورق تواليت\b",
                        r"\bما زودونا بالصابون\b",
                        r"\bما جابوا مي\b",
                        r"\bما عبّوا الشامبو\b",
                        r"\bما زودونا بالأغراض الأساسية\b",
                    ],
                    "zh": [
                        r"没有卫生纸",
                        r"不补洗漱用品",
                        r"没有补水",
                        r"没有再给我们水",
                        r"不补香皂洗发水",
                    ],
                },
            ),
        
            "smell_of_smoke": AspectRule(
                aspect_code="smell_of_smoke",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bзапах сигарет\b",
                        r"\bпахло табаком\b",
                        r"\bпахло дымом\b",
                        r"\bзапах курева\b",
                        r"\bвоняло сигаретами\b",
                    ],
                    "en": [
                        r"\bsmelled like smoke\b",
                        r"\bcigarette smell\b",
                        r"\bsmelled of tobacco\b",
                        r"\bsmelled like cigarettes\b",
                        r"\bsmelled like someone had been smoking\b",
                    ],
                    "tr": [
                        r"\bsigara kokusu\b",
                        r"\btütün kokuyordu\b",
                        r"\bsigara içilmiş gibi kokuyordu\b",
                        r"\boda sigara kokuyordu\b",
                    ],
                    "ar": [
                        r"\bريحة دخان\b",
                        r"\bريحة سجاير\b",
                        r"\bريحة تبغ\b",
                        r"\bريحة كأنو حدا عم يدخن\b",
                    ],
                    "zh": [
                        r"有烟味",
                        r"有香烟味",
                        r"房间有烟味",
                        r"一股烟味像有人抽过烟",
                    ],
                },
            ),
        
            "sewage_smell": AspectRule(
                aspect_code="sewage_smell",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bзапах канализации\b",
                        r"\bвоняет из канализации\b",
                        r"\bпахло (из|как из) канализации\b",
                        r"\bвонь из слива\b",
                    ],
                    "en": [
                        r"\bsewage smell\b",
                        r"\bsewer smell\b",
                        r"\bsmelled like sewage\b",
                        r"\bbad smell from the drain\b",
                    ],
                    "tr": [
                        r"\bkanalizasyon kokusu\b",
                        r"\blağım kokusu\b",
                        r"\blağım gibi kokuyordu\b",
                    ],
                    "ar": [
                        r"\bريحة مجاري\b",
                        r"\bريحة صرف\b",
                        r"\bريحة مجاري قوية\b",
                        r"\bريحة بالوعة\b",
                    ],
                    "zh": [
                        r"下水道味",
                        r"下水道的味道",
                        r"有下水道的臭味",
                        r"排水口有臭味",
                    ],
                },
            ),
        
            "musty_smell": AspectRule(
                aspect_code="musty_smell",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bзапах плесени\b",
                        r"\bзапах сырости\b",
                        r"\bзатхлый запах\b",
                        r"\bсырой запах\b",
                        r"\bпахло сыростью\b",
                    ],
                    "en": [
                        r"\bmoldy smell\b",
                        r"\bmusty smell\b",
                        r"\bdamp smell\b",
                        r"\bsmelled damp\b",
                    ],
                    "tr": [
                        r"\bnem kokusu\b",
                        r"\bküf kokusu\b",
                        r"\brutubet kokusu\b",
                        r"\boda nem kokuyordu\b",
                    ],
                    "ar": [
                        r"\bريحة رطوبة\b",
                        r"\bريحة عفن\b",
                        r"\bريحة عفن رطوبة\b",
                        r"\bريحة رطوبة قوية\b",
                    ],
                    "zh": [
                        r"霉味",
                        r"潮味",
                        r"发霉的味道",
                        r"很潮很闷的味道",
                    ],
                },
            ),
        
            "chemical_smell_strong": AspectRule(
                aspect_code="chemical_smell_strong",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bвоняло хлоркой\b",
                        r"\bсильный запах химии\b",
                        r"\bзапах бытовой химии слишком сильный\b",
                        r"\bзапах чистящих средств очень резкий\b",
                    ],
                    "en": [
                        r"\bstrong bleach smell\b",
                        r"\bsmelled like chemicals\b",
                        r"\bstrong cleaning product smell\b",
                        r"\bchemical smell was too strong\b",
                    ],
                    "tr": [
                        r"\başırı çamaşır suyu kokusu\b",
                        r"\bkimyasal kokuyordu\b",
                        r"\btemizlik malzemesi kokusu çok güçlüydü\b",
                    ],
                    "ar": [
                        r"\bريحة كلور قوية\b",
                        r"\bريحة مواد تنظيف قوية\b",
                        r"\bريحة معقّم قوية\b",
                    ],
                    "zh": [
                        r"一股消毒水味",
                        r"消毒水味太重",
                        r"化学品的味道很重",
                        r"清洁剂味道很刺鼻",
                    ],
                },
            ),
            
            "no_bad_smell": AspectRule(
                aspect_code="no_bad_smell",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bникакого неприятного запаха\b",
                        r"\bничем не пахло\b",
                        r"\bне пахло плохо\b",
                        r"\bзапаха грязи не было\b",
                    ],
                    "en": [
                        r"\bno bad smell\b",
                        r"\bno smell at all\b",
                        r"\bit didn't smell bad\b",
                        r"\bno unpleasant smell\b",
                    ],
                    "tr": [
                        r"\bkötü koku yoktu\b",
                        r"\bhiç koku yoktu\b",
                        r"\bkötü bir koku yoktu\b",
                        r"\bhoş olmayan koku yoktu\b",
                    ],
                    "ar": [
                        r"\bما في ريحة مزعجة\b",
                        r"\bما في ريحة\b",
                        r"\bما كان في ريحة بشعة\b",
                    ],
                    "zh": [
                        r"没有异味",
                        r"没有什么味道",
                        r"没有臭味",
                    ],
                },
            ),
        
            "fresh_smell": AspectRule(
                aspect_code="fresh_smell",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bсвежий запах\b",
                        r"\bприятно пахнет\b",
                        r"\bсвежо пахнет\b",
                        r"\bсвежий воздух в номере\b",
                    ],
                    "en": [
                        r"\bfresh smell\b",
                        r"\bsmelled clean\b",
                        r"\bfresh and clean smell\b",
                        r"\bthe room smelled fresh\b",
                    ],
                    "tr": [
                        r"\btemiz kokuyordu\b",
                        r"\bferah kokuyordu\b",
                        r"\boda ferah kokuyordu\b",
                        r"\bmis gibi kokuyordu\b",
                    ],
                    "ar": [
                        r"\bريحة نظيفة\b",
                        r"\bريحة حلوة\b",
                        r"\bريحة منعشة\b",
                        r"\bريحة المكان كانت حلوة\b",
                    ],
                    "zh": [
                        r"闻起来很干净",
                        r"味道很清新",
                        r"房间有清新的味道",
                    ],
                },
            ),
        
            "entrance_clean": AspectRule(
                aspect_code="entrance_clean",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bчистый подъезд\b",
                        r"\bчистая входная зона\b",
                        r"\bвход чистый\b",
                        r"\bподъезд в хорошем состоянии\b",
                    ],
                    "en": [
                        r"\bentrance was clean\b",
                        r"\bclean entrance\b",
                        r"\bstairwell was clean\b",
                        r"\bthe entryway was clean\b",
                    ],
                    "tr": [
                        r"\bgiriş temizdi\b",
                        r"\bmerdivenler temizdi\b",
                        r"\blokasyonun girişi çok temizdi\b",
                        r"\bapartman girişi düzgündü\b",
                    ],
                    "ar": [
                        r"\bالمدخل نظيف\b",
                        r"\bالدرج نظيف\b",
                        r"\bالمدخل مرتب\b",
                    ],
                    "zh": [
                        r"入口很干净",
                        r"楼道很干净",
                        r"门口很干净整洁",
                    ],
                },
            ),
        
            "hallway_clean": AspectRule(
                aspect_code="hallway_clean",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bчисто в коридоре\b",
                        r"\bаккуратный коридор\b",
                        r"\bчистый коридор\b",
                        r"\bкоридор ухоженный\b",
                    ],
                    "en": [
                        r"\bclean hallway\b",
                        r"\bclean corridor\b",
                        r"\bhallway was clean\b",
                        r"\bthe corridor was tidy\b",
                    ],
                    "tr": [
                        r"\bkoridor temizdi\b",
                        r"\bkoridor çok temizdi\b",
                        r"\bkoridor düzenliydi\b",
                    ],
                    "ar": [
                        r"\bالممر نظيف\b",
                        r"\bالكوريدور نظيف\b",
                        r"\bالممر كان مرتب ونضيف\b",
                    ],
                    "zh": [
                        r"走廊很干净",
                        r"走廊很整洁",
                        r"公共走廊保持得很干净",
                    ],
                },
            ),
        
            "common_areas_clean": AspectRule(
                aspect_code="common_areas_clean",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bчистая общая зона\b",
                        r"\bчисто в холле\b",
                        r"\bобщие зоны чистые\b",
                        r"\bлоби чистый\b",
                    ],
                    "en": [
                        r"\bcommon areas were clean\b",
                        r"\blobby was clean\b",
                        r"\bthe shared areas were very clean\b",
                        r"\bthe public areas were tidy\b",
                    ],
                    "tr": [
                        r"\bortak alanlar temizdi\b",
                        r"\blobi temizdi\b",
                        r"\bpaylaşılan alanlar çok temizdi\b",
                    ],
                    "ar": [
                        r"\bالمناطق المشتركة نظيفة\b",
                        r"\bاللوبي نظيف\b",
                        r"\bالمساحات المشتركة كانت نظيفة\b",
                    ],
                    "zh": [
                        r"公共区域很干净",
                        r"大厅很干净",
                        r"公共区域保持得很整洁",
                    ],
                },
            ),
            
            "entrance_dirty": AspectRule(
                aspect_code="entrance_dirty",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bгрязный подъезд\b",
                        r"\bстарый грязный подъезд\b",
                        r"\bподъезд в ужасном состоянии\b",
                        r"\bгрязный вход\b",
                        r"\bвход грязный\b",
                    ],
                    "en": [
                        r"\bdirty entrance\b",
                        r"\bthe entrance looked terrible\b",
                        r"\bthe stairwell was dirty\b",
                        r"\bfilthy entryway\b",
                    ],
                    "tr": [
                        r"\bkirli giriş\b",
                        r"\bmerdivenler kirliydi\b",
                        r"\bgiriş bölgesi çok pisti\b",
                        r"\bapartman girişi kirliydi\b",
                    ],
                    "ar": [
                        r"\bمدخل وسخ\b",
                        r"\bالمدخل وسخ\b",
                        r"\bالدرج وسخ\b",
                        r"\bالمدخل شكله مو نظيف\b",
                    ],
                    "zh": [
                        r"入口很脏",
                        r"楼道很脏",
                        r"门口环境很差很脏",
                        r"楼道环境很糟",
                    ],
                },
            ),
        
            "hallway_dirty": AspectRule(
                aspect_code="hallway_dirty",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bгрязные коридоры\b",
                        r"\bгрязно в коридоре\b",
                        r"\bпыльно в коридоре\b",
                        r"\bкоридор грязный\b",
                    ],
                    "en": [
                        r"\bdirty hallway\b",
                        r"\bdirty corridor\b",
                        r"\bdusty corridor\b",
                        r"\bhallway looked dirty\b",
                    ],
                    "tr": [
                        r"\bkoridor kirliydi\b",
                        r"\bkoridor tozluydu\b",
                        r"\bkoridor çok pisti\b",
                    ],
                    "ar": [
                        r"\bالممر وسخ\b",
                        r"\bالممر وسخان\b",
                        r"\bالممر ريحته بشعة\b",
                        r"\bالممر شكله مش نضيف\b",
                    ],
                    "zh": [
                        r"走廊很脏",
                        r"走廊很灰",
                        r"走廊地上很脏",
                        r"走廊看起来又脏又旧",
                    ],
                },
            ),
        
            "elevator_dirty": AspectRule(
                aspect_code="elevator_dirty",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bгрязный лифт\b",
                        r"\bлифт грязный\b",
                        r"\bлифт внутри грязный\b",
                    ],
                    "en": [
                        r"\bdirty elevator\b",
                        r"\belevator was dirty\b",
                        r"\belevator was disgusting\b",
                    ],
                    "tr": [
                        r"\basansör kirliydi\b",
                        r"\basansör çok pisti\b",
                        r"\basansörün içi kirliydi\b",
                    ],
                    "ar": [
                        r"\bالمصعد وسخ\b",
                        r"\bالمصعد من جوّا وسخ\b",
                        r"\bالأسانسير كان وصخ\b",
                    ],
                    "zh": [
                        r"电梯很脏",
                        r"电梯里面很脏",
                        r"电梯看起来很脏很旧",
                    ],
                },
            ),
        
            "hallway_bad_smell": AspectRule(
                aspect_code="hallway_bad_smell",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bнеприятный запах в подъезде\b",
                        r"\bвоняет в коридоре\b",
                        r"\bвонь в коридоре\b",
                        r"\bпахло плохо в коридоре\b",
                    ],
                    "en": [
                        r"\bsmelled bad in the hallway\b",
                        r"\bbad smell in the hallway\b",
                        r"\bhallway smelled terrible\b",
                    ],
                    "tr": [
                        r"\bkoridorda kötü koku vardı\b",
                        r"\bkoridor kokuyordu\b",
                        r"\bkoridorda berbat bir koku vardı\b",
                    ],
                    "ar": [
                        r"\bالممر ريحته سيئة\b",
                        r"\bريحة بشعة بالممر\b",
                        r"\bريحة مزعجة بالممر\b",
                    ],
                    "zh": [
                        r"走廊有异味",
                        r"走廊有股臭味",
                        r"走廊一股很难闻的味道",
                    ],
                },
            ),
        
            "entrance_feels_unsafe": AspectRule(
                aspect_code="entrance_feels_unsafe",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bмрачно в подъезде\b",
                        r"\bвыглядит небезопасно\b",
                        r"\bпугающий вход\b",
                        r"\bвход стр[её]мный\b",
                        r"\bнеуютный подъезд\b",
                    ],
                    "en": [
                        r"\bthe entrance felt sketchy\b",
                        r"\bnot safe looking entrance\b",
                        r"\bthe area by the entrance felt unsafe\b",
                        r"\bentrance looked dodgy\b",
                    ],
                    "tr": [
                        r"\bgiriş güvensiz görünüyordu\b",
                        r"\bgiriş korkutucuydu\b",
                        r"\bpek güvenli hissettirmeyen giriş\b",
                        r"\bgiriş çok tekin değildi\b",
                    ],
                    "ar": [
                        r"\bالمدخل شكله يخوف\b",
                        r"\bالمدخل مش مريح\b",
                        r"\bمبين مو آمن\b",
                        r"\bما حسّينا بأمان عند المدخل\b",
                    ],
                    "zh": [
                        r"入口让人不舒服",
                        r"入口看起来不安全",
                        r"感觉门口有点吓人",
                        r"楼道感觉不太安全",
                    ],
                },
            ),
            
            "room_well_equipped": AspectRule(
                aspect_code="room_well_equipped",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bвсё продумано\b",
                        r"\bочень удобно\b",
                        r"\bвсё необходимое есть\b",
                        r"\bесть всё необходимое\b",
                        r"\bномер хорошо оснащ[её]н\b",
                        r"\bв номере было всё что нужно\b",
                    ],
                    "en": [
                        r"\bwell equipped room\b",
                        r"\bthe room had everything we needed\b",
                        r"\bthe room had everything\b",
                        r"\bfully equipped room\b",
                    ],
                    "tr": [
                        r"\bodada her şey vardı\b",
                        r"\bihtiyacımız olan her şey vardı\b",
                        r"\bokul her şey düşünülmüştü\b",
                        r"\bodada ihtiyaç duyulan her şey vardı\b",
                    ],
                    "ar": [
                        r"\bكل شي موجود بالغرفة\b",
                        r"\bكل الأشياء المهمة موجودة\b",
                        r"\bكل اللي منحتاجه موجود\b",
                        r"\bالغرفة مجهزة بكل شي\b",
                    ],
                    "zh": [
                        r"房间设备齐全",
                        r"需要的东西都有",
                        r"房间配得很全",
                        r"房间里什么都有",
                    ],
                },
            ),
        
            "kettle_available": AspectRule(
                aspect_code="kettle_available",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bв номере есть чайник\b",
                        r"\bесть чайник и посуда\b",
                        r"\bбыл чайник\b",
                        r"\bчайник в номере\b",
                    ],
                    "en": [
                        r"\bthere was a kettle\b",
                        r"\bkettle in the room\b",
                        r"\bthe room had a kettle\b",
                        r"\bwe had a kettle\b",
                    ],
                    "tr": [
                        r"\bsu ısıtıcısı vardı\b",
                        r"\bodada su ısıtıcısı vardı\b",
                        r"\bketıl vardı\b",
                    ],
                    "ar": [
                        r"\bفي غلاية ماء\b",
                        r"\bفي كاتل بالغرفة\b",
                        r"\bفي غلاية بالغرفة\b",
                    ],
                    "zh": [
                        r"有水壶",
                        r"有烧水壶",
                        r"房间里有热水壶",
                    ],
                },
            ),
        
            "fridge_available": AspectRule(
                aspect_code="fridge_available",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bесть холодильник\b",
                        r"\bхолодильник в номере\b",
                        r"\bбыл холодильник\b",
                    ],
                    "en": [
                        r"\bthere was a fridge\b",
                        r"\bthe room had a fridge\b",
                        r"\bfridge in the room\b",
                    ],
                    "tr": [
                        r"\bbuzdolabı vardı\b",
                        r"\bodada buzdolabı vardı\b",
                        r"\bmini buzdolabı vardı\b",
                    ],
                    "ar": [
                        r"\bفي براد\b",
                        r"\bفي برّاد بالغرفة\b",
                        r"\bفي ثلاجة صغيرة بالغرفة\b",
                    ],
                    "zh": [
                        r"有冰箱",
                        r"房间里有冰箱",
                        r"有小冰箱",
                    ],
                },
            ),
        
            "hairdryer_available": AspectRule(
                aspect_code="hairdryer_available",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bесть фен\b",
                        r"\bв номере был фен\b",
                        r"\bфен предоставили\b",
                    ],
                    "en": [
                        r"\bthere was a hairdryer\b",
                        r"\bthe room had a hairdryer\b",
                        r"\bhairdryer provided\b",
                    ],
                    "tr": [
                        r"\bsaç kurutma makinesi vardı\b",
                        r"\bodada saç kurutma vardı\b",
                        r"\bsaç kurutma makinesi mevcuttu\b",
                    ],
                    "ar": [
                        r"\bفي سيشوار\b",
                        r"\bمعنّا سشوار بالغرفة\b",
                        r"\bأمنولنا سيشوار\b",
                    ],
                    "zh": [
                        r"有吹风机",
                        r"房间里有吹风机",
                        r"提供吹风机",
                    ],
                },
            ),
        
            "sockets_enough": AspectRule(
                aspect_code="sockets_enough",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bмного розеток\b",
                        r"\bрозетки рядом с кроватью\b",
                        r"\bдостаточно розеток\b",
                        r"\bрозеток хватало\b",
                    ],
                    "en": [
                        r"\benough outlets\b",
                        r"\bsockets next to the bed\b",
                        r"\bplenty of plugs\b",
                        r"\blots of outlets\b",
                    ],
                    "tr": [
                        r"\byeterince priz vardı\b",
                        r"\byatağın yanında priz vardı\b",
                        r"\bprizler yeterliydi\b",
                    ],
                    "ar": [
                        r"\bفي فيش جنب التخت\b",
                        r"\bفي فيش كفاية\b",
                        r"\bفي مَخارج كهربا كفاية\b",
                    ],
                    "zh": [
                        r"插座很多",
                        r"床边有插座",
                        r"插座很够用",
                        r"充电很方便",
                    ],
                },
            ),
            
            "workspace_available": AspectRule(
                aspect_code="workspace_available",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bудобный рабочий стол\b",
                        r"\bесть рабочий стол\b",
                        r"\bможно работать за столом\b",
                        r"\bудобно работать\b",
                        r"\bбыло где поработать\b",
                    ],
                    "en": [
                        r"\bgood desk to work\b",
                        r"\bdesk for work\b",
                        r"\bspace to work\b",
                        r"\bplace to work in the room\b",
                        r"\bcomfortable workspace\b",
                    ],
                    "tr": [
                        r"\bçalışmak için masa vardı\b",
                        r"\bçalışacak masa vardı\b",
                        r"\bçalışmak rahattı odada\b",
                        r"\bmasa uygundu çalışmak için\b",
                    ],
                    "ar": [
                        r"\bفي مكتب للشغل\b",
                        r"\bفي طاولة فينا نشتغل عليها\b",
                        r"\bفي مكان نشتغل فيه بالغرفة\b",
                        r"\bمريح للشغل\b",
                    ],
                    "zh": [
                        r"有书桌可以办公",
                        r"有桌子可以工作",
                        r"房间里可以办公",
                        r"有办公区域",
                    ],
                },
            ),
        
            "luggage_space_ok": AspectRule(
                aspect_code="luggage_space_ok",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bесть где разложить чемоданы\b",
                        r"\bесть куда разложить вещи\b",
                        r"\bбыло место для чемодана\b",
                        r"\bудобно с багажом\b",
                    ],
                    "en": [
                        r"\bspace for luggage\b",
                        r"\bplace to unpack\b",
                        r"\broom for suitcases\b",
                        r"\bthere was space for our bags\b",
                    ],
                    "tr": [
                        r"\bbavulu açacak yer vardı\b",
                        r"\beşyaları koyacak yer vardı\b",
                        r"\bvaliz için yer vardı\b",
                        r"\bvaliz koymak rahattı\b",
                    ],
                    "ar": [
                        r"\bفي مكان للشنط\b",
                        r"\bفي مساحة نرتب أغراضنا\b",
                        r"\bفيك تحط الشنطة وتفتحها براحة\b",
                    ],
                    "zh": [
                        r"有地方放行李",
                        r"行李可以打开",
                        r"放箱子的空间够用",
                        r"有空间整理行李",
                    ],
                },
            ),
        
            "kettle_missing": AspectRule(
                aspect_code="kettle_missing",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bнет чайника\b",
                        r"\bне хватает чайника\b",
                        r"\bчайника не было\b",
                        r"\bне было чайника в номере\b",
                    ],
                    "en": [
                        r"\bno kettle\b",
                        r"\bthere was no kettle\b",
                        r"\bwe didn't have a kettle\b",
                        r"\bmissing kettle\b",
                    ],
                    "tr": [
                        r"\bsu ısıtıcısı yoktu\b",
                        r"\bketıl yoktu\b",
                        r"\bodada su ısıtıcısı yoktu\b",
                    ],
                    "ar": [
                        r"\bما في غلاية\b",
                        r"\bما في كاتل\b",
                        r"\bما كان في غلاية بالغرفة\b",
                    ],
                    "zh": [
                        r"没有水壶",
                        r"没有热水壶",
                        r"房间里没有烧水壶",
                    ],
                },
            ),
        
            "fridge_missing": AspectRule(
                aspect_code="fridge_missing",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bнет холодильника\b",
                        r"\bхолодильника не было\b",
                        r"\bне хватает холодильника\b",
                    ],
                    "en": [
                        r"\bno fridge\b",
                        r"\bthere was no fridge\b",
                        r"\bmissing fridge\b",
                        r"\bno refrigerator in the room\b",
                    ],
                    "tr": [
                        r"\bbuzdolabı yoktu\b",
                        r"\bodada buzdolabı yoktu\b",
                        r"\bmini buzdolabı yoktu\b",
                    ],
                    "ar": [
                        r"\bما في براد\b",
                        r"\bما كان في برّاد بالغرفة\b",
                        r"\bما في ثلاجة صغيرة\b",
                    ],
                    "zh": [
                        r"没有冰箱",
                        r"房间里没有冰箱",
                        r"没有小冰箱",
                    ],
                },
            ),
        
            "hairdryer_missing": AspectRule(
                aspect_code="hairdryer_missing",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bнет фена\b",
                        r"\bфена не было\b",
                        r"\bне хватает фена\b",
                        r"\bбез фена\b",
                    ],
                    "en": [
                        r"\bno hairdryer\b",
                        r"\bno hair dryer\b",
                        r"\bthere was no hairdryer\b",
                        r"\bno hair dryer in the room\b",
                    ],
                    "tr": [
                        r"\bsaç kurutma yoktu\b",
                        r"\bsaç kurutma makinesi yoktu\b",
                        r"\bodada saç kurutma makinesi yoktu\b",
                    ],
                    "ar": [
                        r"\bما في سيشوار\b",
                        r"\bما في سشوار بالغرفة\b",
                        r"\bما عطونا سشوار\b",
                    ],
                    "zh": [
                        r"没有吹风机",
                        r"房间里没有吹风机",
                        r"没有提供吹风机",
                    ],
                },
            ),
            
            "sockets_not_enough": AspectRule(
                aspect_code="sockets_not_enough",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bмало розеток\b",
                        r"\bрозеток не хватает\b",
                        r"\bпочти нет розеток\b",
                        r"\bрядом с кроватью нет розеток\b",
                    ],
                    "en": [
                        r"\bnot enough outlets\b",
                        r"\bnot enough sockets\b",
                        r"\bno sockets near the bed\b",
                        r"\bwe needed more plugs\b",
                    ],
                    "tr": [
                        r"\bpriz azdı\b",
                        r"\byeterince priz yoktu\b",
                        r"\byatağın yanında priz yoktu\b",
                        r"\bpriz sayısı yetersizdi\b",
                    ],
                    "ar": [
                        r"\bما في فيش كفاية\b",
                        r"\bما في فيش حد التخت\b",
                        r"\bما في مقابس كهربا كفاية\b",
                        r"\bكنا محتاجين مقابس أكتر\b",
                    ],
                    "zh": [
                        r"插座不够",
                        r"插座太少",
                        r"床边没有插座",
                        r"充电很不方便",
                    ],
                },
            ),
        
            "no_workspace": AspectRule(
                aspect_code="no_workspace",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bнет нормального стола\b",
                        r"\bнеудобно работать\b",
                        r"\bнегде поработать\b",
                        r"\bнет рабочего места\b",
                    ],
                    "en": [
                        r"\bno proper desk\b",
                        r"\bnowhere to work\b",
                        r"\bno desk to work\b",
                        r"\bnot comfortable to work in the room\b",
                    ],
                    "tr": [
                        r"\bçalışacak masa yoktu\b",
                        r"\bçalışmak için uygun masa yoktu\b",
                        r"\bodada çalışmak zordu\b",
                        r"\bçalışacak alan yoktu\b",
                    ],
                    "ar": [
                        r"\bما في طاولة نشتغل عليها\b",
                        r"\bما في مكان نشتغل فيه بالغرفة\b",
                        r"\bالشغل صعب من الغرفة\b",
                        r"\bما في مكتب للشغل\b",
                    ],
                    "zh": [
                        r"没有桌子可以办公",
                        r"没有书桌",
                        r"房间里没办法工作",
                        r"没有合适的办公区域",
                    ],
                },
            ),
        
            "no_luggage_space": AspectRule(
                aspect_code="no_luggage_space",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bнеудобно разложить вещи\b",
                        r"\bнекуда разложить вещи\b",
                        r"\bне было места для чемодана\b",
                        r"\bс чемоданом неудобно\b",
                    ],
                    "en": [
                        r"\bno place for luggage\b",
                        r"\bno space to unpack\b",
                        r"\bno room for suitcases\b",
                        r"\bhard to manage luggage in the room\b",
                    ],
                    "tr": [
                        r"\bbavulu açacak yer yoktu\b",
                        r"\beşyaları koyacak yer yoktu\b",
                        r"\bvaliz için yer yoktu\b",
                        r"\bbavulla çok zordu\b",
                    ],
                    "ar": [
                        r"\bما في مكان للشنط\b",
                        r"\bما في مساحة نفتح الشنطة\b",
                        r"\bما عنا محل نرتب أغراضنا\b",
                        r"\bمع الشنط الوضع مو مريح\b",
                    ],
                    "zh": [
                        r"没地方放行李",
                        r"行李没法打开",
                        r"房间里没空间放箱子",
                        r"带箱子很不方便",
                    ],
                },
            ),
        
            "bed_comfy": AspectRule(
                aspect_code="bed_comfy",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bкровать удобная\b",
                        r"\bочень удобная кровать\b",
                        r"\bспать было комфортно\b",
                        r"\bкровать прям очень удобная\b",
                    ],
                    "en": [
                        r"\bthe bed was very comfortable\b",
                        r"\bcomfortable bed\b",
                        r"\bthe bed was comfy\b",
                        r"\bbed was super comfortable\b",
                    ],
                    "tr": [
                        r"\byatak rahattı\b",
                        r"\bçok rahat yatak\b",
                        r"\byatak çok konforluydu\b",
                    ],
                    "ar": [
                        r"\bالسرير مريح\b",
                        r"\bالسرير كتير مريح\b",
                        r"\bالسرير راحة\b",
                    ],
                    "zh": [
                        r"床很舒服",
                        r"床非常舒服",
                        r"床睡起来很舒服",
                    ],
                },
            ),
        
            "mattress_comfy": AspectRule(
                aspect_code="mattress_comfy",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bудобный матрас\b",
                        r"\bматрас удобный\b",
                        r"\bидеальный матрас\b",
                        r"\bматрас очень комфортный\b",
                    ],
                    "en": [
                        r"\bcomfortable mattress\b",
                        r"\bgood mattress\b",
                        r"\bthe mattress was really comfortable\b",
                        r"\bthe mattress felt great\b",
                    ],
                    "tr": [
                        r"\bşilte rahattı\b",
                        r"\bşilte çok rahattı\b",
                        r"\brahat bir yatak vardı\b",
                    ],
                    "ar": [
                        r"\bالماترس مريح\b",
                        r"\bالماترس كتير مريح\b",
                        r"\bالفرشة مريحة\b",
                    ],
                    "zh": [
                        r"床垫很舒服",
                        r"床垫很软很舒服",
                        r"床垫很舒服很好睡",
                    ],
                },
            ),
            
            "pillow_comfy": AspectRule(
                aspect_code="pillow_comfy",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bудобные подушки\b",
                        r"\bподушки удобные\b",
                        r"\bподушки очень удобные\b",
                        r"\bподушки комфортные\b",
                    ],
                    "en": [
                        r"\bpillows were comfortable\b",
                        r"\bvery comfortable pillows\b",
                        r"\bthe pillows were comfy\b",
                        r"\bgood pillows\b",
                    ],
                    "tr": [
                        r"\byastıklar rahattı\b",
                        r"\byastıklar çok rahattı\b",
                        r"\brahat yastıklar vardı\b",
                    ],
                    "ar": [
                        r"\bالمخدات مريحة\b",
                        r"\bالمخدات كتير مريحة\b",
                        r"\bالمخدات كانت مريحة بالنوم\b",
                    ],
                    "zh": [
                        r"枕头很舒服",
                        r"枕头很软很舒服",
                        r"枕头睡起来很舒服",
                    ],
                },
            ),
        
            "slept_well": AspectRule(
                aspect_code="slept_well",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bспать было комфортно\b",
                        r"\bспалось отлично\b",
                        r"\bвыспались отлично\b",
                        r"\bхорошо выспались\b",
                    ],
                    "en": [
                        r"\bslept really well\b",
                        r"\bslept great\b",
                        r"\bhad a good night's sleep\b",
                        r"\bwe slept very well\b",
                    ],
                    "tr": [
                        r"\bçok iyi uyuduk\b",
                        r"\biyı dinlendik\b",
                        r"\bgece çok iyi uyudum\b",
                        r"\bçok rahat uyuduk\b",
                    ],
                    "ar": [
                        r"\bنمنا منيح\b",
                        r"\bنمنا كتير منيح\b",
                        r"\bنمنا بارتياح\b",
                        r"\bنمنا مرتاحين\b",
                    ],
                    "zh": [
                        r"睡得很好",
                        r"睡得很香",
                        r"晚上睡得很棒",
                        r"睡得很舒服",
                    ],
                },
            ),
        
            "bed_uncomfortable": AspectRule(
                aspect_code="bed_uncomfortable",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bкровать неудобная\b",
                        r"\bнеудобная кровать\b",
                        r"\bкровать вообще не удобная\b",
                        r"\bспать было неудобно\b",
                    ],
                    "en": [
                        r"\buncomfortable bed\b",
                        r"\bthe bed was uncomfortable\b",
                        r"\bthe bed wasn't comfortable\b",
                        r"\bbed not comfortable at all\b",
                    ],
                    "tr": [
                        r"\byatak rahatsızdı\b",
                        r"\brahat değildi\b",
                        r"\byatak konforsuzdu\b",
                        r"\byatak hiç rahat değildi\b",
                    ],
                    "ar": [
                        r"\bالسرير مو مريح\b",
                        r"\bالسرير مش مريح\b",
                        r"\bالنوم عالسرير مو مريح\b",
                    ],
                    "zh": [
                        r"床不舒服",
                        r"床一点都不舒服",
                        r"睡起来不舒服",
                        r"床很难睡",
                    ],
                },
            ),
        
            "mattress_too_soft": AspectRule(
                aspect_code="mattress_too_soft",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bматрас слишком мягк(ий|ий)\b",
                        r"\bслишком мягкий матрас\b",
                        r"\bматрас очень мягкий\b",
                        r"\bматрас проваливался как в яму\b",
                    ],
                    "en": [
                        r"\bmattress too soft\b",
                        r"\bthe mattress was too soft\b",
                        r"\bway too soft mattress\b",
                        r"\bmattress felt too soft\b",
                    ],
                    "tr": [
                        r"\bşilte çok yumuşaktı\b",
                        r"\byatak çok yumuşaktı\b",
                        r"\bşilte fazla yumuşaktı\b",
                    ],
                    "ar": [
                        r"\bالماترس لين كتير\b",
                        r"\bالفرشة كتير طرية\b",
                        r"\bالمرتبة كتير طرية\b",
                    ],
                    "zh": [
                        r"床垫太软",
                        r"床垫特别软",
                        r"床垫软得不行",
                    ],
                },
            ),
        
            "mattress_too_hard": AspectRule(
                aspect_code="mattress_too_hard",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bматрас слишком ж(ё|е)сткий\b",
                        r"\bслишком жесткий матрас\b",
                        r"\bочень жёсткий матрас\b",
                        r"\bматрас как доска\b",
                    ],
                    "en": [
                        r"\bmattress too hard\b",
                        r"\bthe mattress was too hard\b",
                        r"\bvery hard mattress\b",
                        r"\bmattress felt like a rock\b",
                    ],
                    "tr": [
                        r"\bşilte çok sertti\b",
                        r"\byatak çok sertti\b",
                        r"\bşilte fazla sertti\b",
                    ],
                    "ar": [
                        r"\bالماترس قاسي كتير\b",
                        r"\bالمرتبة قاسية\b",
                        r"\bالفرشة كتير قاسية\b",
                    ],
                    "zh": [
                        r"床垫太硬",
                        r"床垫特别硬",
                        r"床垫硬得像板子",
                    ],
                },
            ),
            
            "mattress_sagging": AspectRule(
                aspect_code="mattress_sagging",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bпродавленный матрас\b",
                        r"\bматрас проваливался\b",
                        r"\bматрас провисает\b",
                        r"\bматрас с ямой\b",
                    ],
                    "en": [
                        r"\bmattress was sagging\b",
                        r"\bsaggy mattress\b",
                        r"\bthe mattress was caving in\b",
                        r"\bthe mattress had a dip\b",
                    ],
                    "tr": [
                        r"\bşilte çökmüştü\b",
                        r"\bşilte göçmüştü\b",
                        r"\byatak çökmüştü\b",
                        r"\byatağın yayları hissediliyordu\b",
                    ],
                    "ar": [
                        r"\bالماترس غاطس\b",
                        r"\bالماترس نازل بالنص\b",
                        r"\bالفرشة هابطة\b",
                        r"\bالمرتبة خربانة ونازلة\b",
                    ],
                    "zh": [
                        r"床垫塌了",
                        r"床垫塌陷",
                        r"床垫中间塌下去了",
                        r"床垫凹下去了一块",
                    ],
                },
            ),
        
            "bed_creaks": AspectRule(
                aspect_code="bed_creaks",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bкровать скрипела\b",
                        r"\bскрипучая кровать\b",
                        r"\bкровать всё время скрипела\b",
                        r"\bкровать издавала звуки\b",
                    ],
                    "en": [
                        r"\bcreaky bed\b",
                        r"\bthe bed was creaking\b",
                        r"\bthe bed made noise\b",
                        r"\bthe bed squeaked all night\b",
                    ],
                    "tr": [
                        r"\byatak gıcırdıyordu\b",
                        r"\byatak çok gıcırdıyordu\b",
                        r"\byatak ses yapıyordu\b",
                    ],
                    "ar": [
                        r"\bالسرير بيوزن\b",
                        r"\bالسرير بيصرّف\b",
                        r"\bالسرير بيصرخ\b",
                        r"\bالسرير عم يطلع صوت طول الوقت\b",
                    ],
                    "zh": [
                        r"床会吱吱响",
                        r"床一直响",
                        r"床吱呀吱呀的",
                        r"床动一下就响",
                    ],
                },
            ),
        
            "pillow_uncomfortable": AspectRule(
                aspect_code="pillow_uncomfortable",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bподушки неудобные\b",
                        r"\bнеудобная подушка\b",
                        r"\bподушки не понравились\b",
                        r"\bподушки так себе\b",
                    ],
                    "en": [
                        r"\bpillows were uncomfortable\b",
                        r"\bthe pillow was uncomfortable\b",
                        r"\bdidn't like the pillows\b",
                        r"\bthe pillows were bad\b",
                    ],
                    "tr": [
                        r"\byastıklar rahatsızdı\b",
                        r"\brahatsız yastık\b",
                        r"\byastık pek rahat değildi\b",
                    ],
                    "ar": [
                        r"\bالمخدات مو مريحة\b",
                        r"\bالمخدة مو مريحة\b",
                        r"\bالمخدة ما كانت مريحة بالنوم\b",
                    ],
                    "zh": [
                        r"枕头不舒服",
                        r"枕头不太舒服",
                        r"枕头很难睡",
                    ],
                },
            ),
        
            "pillow_too_hard": AspectRule(
                aspect_code="pillow_too_hard",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bжесткие подушки\b",
                        r"\bподушка слишком ж(ё|е)сткая\b",
                        r"\bподушка очень твёрдая\b",
                    ],
                    "en": [
                        r"\bpillows too hard\b",
                        r"\bthe pillow was too hard\b",
                        r"\bvery hard pillow\b",
                    ],
                    "tr": [
                        r"\byastık çok sertti\b",
                        r"\bçok sert yastık\b",
                        r"\byastık fazla sertti\b",
                    ],
                    "ar": [
                        r"\bالمخدة قاسية كتير\b",
                        r"\bالمخدات قاسية\b",
                        r"\bالمخدة كتير قاسية\b",
                    ],
                    "zh": [
                        r"枕头太硬",
                        r"枕头很硬",
                        r"枕头硬得不舒服",
                    ],
                },
            ),
        
            "pillow_too_high": AspectRule(
                aspect_code="pillow_too_high",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bслишком высокие подушки\b",
                        r"\bподушка слишком высокая\b",
                        r"\bподушки очень высокие\b",
                    ],
                    "en": [
                        r"\bpillows too high\b",
                        r"\bthe pillow was too high\b",
                        r"\bthe pillows were too thick\b",
                        r"\bpillow was too big and high\b",
                    ],
                    "tr": [
                        r"\byastık çok yüksekti\b",
                        r"\byastık fazla yüksekti\b",
                        r"\bçok kalın yastık\b",
                    ],
                    "ar": [
                        r"\bالمخدة عالية كتير\b",
                        r"\bالمخدة كتير عالية\b",
                        r"\bالمخدات عالية\b",
                    ],
                    "zh": [
                        r"枕头太高",
                        r"枕头太厚太高",
                        r"枕头太鼓太高",
                    ],
                },
            ),
            
            "quiet_room": AspectRule(
                aspect_code="quiet_room",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bтихо\b",
                        r"\bочень тихо\b",
                        r"\bспокойно ночью\b",
                        r"\bтихо в номере\b",
                        r"\bбыло тихо спать\b",
                    ],
                    "en": [
                        r"\bvery quiet\b",
                        r"\bnice and quiet\b",
                        r"\bquiet at night\b",
                        r"\bthe room was quiet\b",
                        r"\bquiet to sleep\b",
                    ],
                    "tr": [
                        r"\bçok sessizdi\b",
                        r"\bgece çok sakindi\b",
                        r"\bodada çok sessizdi\b",
                        r"\bgeceleri sessizdi\b",
                    ],
                    "ar": [
                        r"\bهادئ\b",
                        r"\bكتير هادي\b",
                        r"\bبالليل هادي\b",
                        r"\bالغرفة هادية\b",
                    ],
                    "zh": [
                        r"很安静",
                        r"晚上很安静",
                        r"房间很安静",
                        r"睡觉很安静",
                    ],
                },
            ),
        
            "good_soundproofing": AspectRule(
                aspect_code="good_soundproofing",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bхорошая звукоизоляция\b",
                        r"\bничего не слышно\b",
                        r"\bсоседей не слышно\b",
                        r"\bулицу не слышно\b",
                    ],
                    "en": [
                        r"\bgood soundproofing\b",
                        r"\bwell soundproofed\b",
                        r"\bwe couldn't hear the neighbors\b",
                        r"\bwe couldn't hear anything from outside\b",
                    ],
                    "tr": [
                        r"\biyı ses yalıtımı vardı\b",
                        r"\bkomşuları duymuyorduk\b",
                        r"\bsokağın sesi yoktu\b",
                        r"\bhiç ses gelmiyordu dışarıdan\b",
                    ],
                    "ar": [
                        r"\bالعزل منيح\b",
                        r"\bما بنسمع حدا\b",
                        r"\bما عم نسمع الجيران\b",
                        r"\bما عم نسمع شي من برا\b",
                    ],
                    "zh": [
                        r"隔音很好",
                        r"听不到邻居",
                        r"听不到外面的声音",
                        r"基本听不见外面的声音",
                    ],
                },
            ),
        
            "no_street_noise": AspectRule(
                aspect_code="no_street_noise",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bулицу не слышно\b",
                        r"\bшум с улицы не слышно\b",
                        r"\bне слышно улицу ночью\b",
                    ],
                    "en": [
                        r"\bno street noise\b",
                        r"\bno traffic noise\b",
                        r"\bwe didn't hear the street\b",
                        r"\bquiet despite being near the street\b",
                    ],
                    "tr": [
                        r"\bsokağın sesi yoktu\b",
                        r"\btrafik sesi gelmiyordu\b",
                        r"\bdışarıdan sokak sesi gelmedi\b",
                    ],
                    "ar": [
                        r"\bما في صوت شارع\b",
                        r"\bما عم نسمع صوت الشارع\b",
                        r"\bما في ضجة من الشارع\b",
                    ],
                    "zh": [
                        r"没有街上的噪音",
                        r"听不到马路的声音",
                        r"基本听不到外面马路声",
                    ],
                },
            ),
        
            "noisy_room": AspectRule(
                aspect_code="noisy_room",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bшумно\b",
                        r"\bочень шумно\b",
                        r"\bв номере шумно\b",
                        r"\bне могли уснуть из-за шума\b",
                        r"\bшумно даже ночью\b",
                    ],
                    "en": [
                        r"\bnoisy\b",
                        r"\bvery noisy\b",
                        r"\bthe room was noisy\b",
                        r"\bhard to sleep because of noise\b",
                        r"\btoo noisy at night\b",
                    ],
                    "tr": [
                        r"\bçok gürültülüydü\b",
                        r"\bgece gürültülüydü\b",
                        r"\bodada gürültü vardı\b",
                        r"\buyuyamadık gürültüden\b",
                    ],
                    "ar": [
                        r"\bفي ازعاج\b",
                        r"\bصوت عالي\b",
                        r"\bالغرفة مزعجة\b",
                        r"\bما قدرنا ننام من الصوت\b",
                    ],
                    "zh": [
                        r"很吵",
                        r"晚上很吵",
                        r"房间很吵",
                        r"吵得睡不着",
                    ],
                },
            ),
        
            "street_noise": AspectRule(
                aspect_code="street_noise",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bшум с улицы\b",
                        r"\bгромко с улицы\b",
                        r"\bслышно улицу\b",
                        r"\bслышно трафик\b",
                    ],
                    "en": [
                        r"\bstreet noise\b",
                        r"\btraffic noise\b",
                        r"\bnoise from the street\b",
                        r"\bcould hear cars all night\b",
                    ],
                    "tr": [
                        r"\bsokak çok gürültülüydü\b",
                        r"\btrafik sesi vardı\b",
                        r"\bsokaktan ses geliyordu\b",
                        r"\bgece boyunca araba sesleri vardı\b",
                    ],
                    "ar": [
                        r"\bصوت الشارع عالي\b",
                        r"\bصوت سيارات طول الليل\b",
                        r"\bفي ضجة من الشارع\b",
                        r"\bفي ضجة من برا\b",
                    ],
                    "zh": [
                        r"街上很吵",
                        r"马路太吵",
                        r"一直听到车声",
                        r"能听到街上的噪音",
                    ],
                },
            ),
            
            "thin_walls": AspectRule(
                aspect_code="thin_walls",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bтонкие стены\b",
                        r"\bслышно соседей\b",
                        r"\bслышно всё из коридора\b",
                        r"\bслышно всё что говорят\b",
                        r"\bслышно разговоры соседей\b",
                    ],
                    "en": [
                        r"\bthin walls\b",
                        r"\byou can hear everything\b",
                        r"\bwe could hear the neighbors\b",
                        r"\bwe could hear people in the next room\b",
                        r"\bheard everything from the hallway\b",
                    ],
                    "tr": [
                        r"\bduvarlar inceydi\b",
                        r"\bher şeyi duyabiliyorduk\b",
                        r"\byan odanın sesini duyuyorduk\b",
                        r"\bkoridordan gelen sesleri duyduk\b",
                    ],
                    "ar": [
                        r"\bالجدران رفيعة\b",
                        r"\bعم نسمع الجيران\b",
                        r"\bعم نسمع كل شي من برا\b",
                        r"\bكله مسموع من الغرف التانية\b",
                    ],
                    "zh": [
                        r"隔音很差",
                        r"墙很薄",
                        r"能听到隔壁",
                        r"隔壁说话都听得很清楚",
                        r"能听到走廊里的声音",
                    ],
                },
            ),
        
            "hallway_noise": AspectRule(
                aspect_code="hallway_noise",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bслышно всё из коридора\b",
                        r"\bшум из коридора\b",
                        r"\bслышно коридор\b",
                        r"\bслышно ресепшен\b",
                        r"\bслышно лифт\b",
                    ],
                    "en": [
                        r"\bnoise from the hallway\b",
                        r"\bnoise from reception\b",
                        r"\bnoise from the elevator\b",
                        r"\bwe could hear people in the hallway\b",
                    ],
                    "tr": [
                        r"\bresepsiyondan ses geliyordu\b",
                        r"\bkoridordan ses geliyordu\b",
                        r"\basansör sesi geliyordu\b",
                        r"\bkoridor çok gürültülüydü\b",
                    ],
                    "ar": [
                        r"\bصوت الريسيبشن\b",
                        r"\bصوت الأسانسير\b",
                        r"\bعم نسمع الأصوات من الممر\b",
                        r"\bالناس بالممر عم يزعجوا\b",
                    ],
                    "zh": [
                        r"能听到走廊的声音",
                        r"能听到前台的声音",
                        r"能听到电梯的声音",
                        r"走廊很吵",
                    ],
                },
            ),
        
            "night_noise_trouble_sleep": AspectRule(
                aspect_code="night_noise_trouble_sleep",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bмузыка ночью\b",
                        r"\bкрики ночью\b",
                        r"\bне могли уснуть из-за шума\b",
                        r"\bшумно ночью, спать невозможно\b",
                    ],
                    "en": [
                        r"\bloud music at night\b",
                        r"\bpeople shouting at night\b",
                        r"\bhard to sleep because of noise\b",
                        r"\bcouldn't sleep because of the noise\b",
                    ],
                    "tr": [
                        r"\bgece müzik vardı\b",
                        r"\bgece bağırış vardı\b",
                        r"\buyuyamadık gürültüden\b",
                        r"\bgece çok ses vardı uyumak zordu\b",
                    ],
                    "ar": [
                        r"\bموسيقى بالليل\b",
                        r"\bصراخ بالليل\b",
                        r"\bما قدرنا ننام من الصوت\b",
                        r"\bفي دوشة طول الليل\b",
                    ],
                    "zh": [
                        r"晚上有人大声讲话",
                        r"晚上有音乐很吵",
                        r"吵得睡不着觉",
                        r"晚上太吵没法睡觉",
                    ],
                },
            ),
        
            "temp_comfortable": AspectRule(
                aspect_code="temp_comfortable",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bтемпература комфортная\b",
                        r"\bтемпература идеальная\b",
                        r"\bне жарко\b",
                        r"\bне холодно\b",
                        r"\bв номере тепло\b",
                        r"\bбыло достаточно тепло\b",
                    ],
                    "en": [
                        r"\btemperature was comfortable\b",
                        r"\bperfect temperature\b",
                        r"\bnot too hot\b",
                        r"\bnot too cold\b",
                        r"\bthe room was warm enough\b",
                    ],
                    "tr": [
                        r"\boda sıcaklığı rahattı\b",
                        r"\bne çok sıcak ne çok soğuk\b",
                        r"\bodada yeterince sıcaktı\b",
                        r"\boda ısısı iyiydi\b",
                    ],
                    "ar": [
                        r"\bالحرارة مريحة\b",
                        r"\bالجو بالغرفة كان تمام\b",
                        r"\bمش حار ومش برد\b",
                        r"\bالغرفة دافية كفاية\b",
                    ],
                    "zh": [
                        r"房间温度很舒服",
                        r"温度正好",
                        r"不热不冷",
                        r"房间很暖和",
                    ],
                },
            ),
        
            "ventilation_ok": AspectRule(
                aspect_code="ventilation_ok",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bможно проветрить\b",
                        r"\bхорошо проветривается\b",
                        r"\bможно открыть окно и проветрить\b",
                        r"\bсвежий воздух в комнате\b",
                    ],
                    "en": [
                        r"\beasy to air the room\b",
                        r"\bgood ventilation\b",
                        r"\bcould air out the room\b",
                        r"\bfresh air in the room\b",
                    ],
                    "tr": [
                        r"\boda havalanabiliyordu\b",
                        r"\bhava sirkülasyonu iyiydi\b",
                        r"\bpencere açınca hava hemen değişiyordu\b",
                    ],
                    "ar": [
                        r"\bفي تهوية كويسة\b",
                        r"\bقدرنا نهوّي\b",
                        r"\bفيك تفتح الشباك وتهوي الغرفة\b",
                    ],
                    "zh": [
                        r"可以通风",
                        r"通风很好",
                        r"房间可以开窗通风",
                        r"空气流通还可以",
                    ],
                },
            ),
            
            "ac_working": AspectRule(
                aspect_code="ac_working",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bкондиционер работает\b",
                        r"\bкондиционер отлично работал\b",
                        r"\bкондиционер справлялся\b",
                        r"\bхорошо работал кондиционер\b",
                    ],
                    "en": [
                        r"\bAC worked well\b",
                        r"\bair conditioning worked well\b",
                        r"\bAC worked fine\b",
                        r"\bair con worked perfectly\b",
                    ],
                    "tr": [
                        r"\bklima çalışıyordu\b",
                        r"\bklima sorunsuzdu\b",
                        r"\bklima iyi çalışıyordu\b",
                    ],
                    "ar": [
                        r"\bالمكيف شغال منيح\b",
                        r"\bالمكيف شغال تمام\b",
                        r"\bالتكييف شغال كويس\b",
                    ],
                    "zh": [
                        r"空调很好用",
                        r"空调正常工作",
                        r"空调很给力",
                    ],
                },
            ),
        
            "heating_working": AspectRule(
                aspect_code="heating_working",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bотопление хорошее\b",
                        r"\bотопление работало\b",
                        r"\bв номере тепло\b",
                        r"\bбатареи тёплые\b",
                    ],
                    "en": [
                        r"\bheating worked\b",
                        r"\bthe heater worked\b",
                        r"\bthe room was warm enough\b",
                        r"\bthe heating was good\b",
                    ],
                    "tr": [
                        r"\bısıtma çalışıyordu\b",
                        r"\bodada yeterince sıcaktı\b",
                        r"\boda ısısı cihaz olarak iyiydi\b",
                    ],
                    "ar": [
                        r"\bالتدفئة شغالة\b",
                        r"\bالتدفئة كانت كويسة\b",
                        r"\bالغرفة دافية\b",
                    ],
                    "zh": [
                        r"暖气很好",
                        r"暖气正常",
                        r"房间很暖和",
                    ],
                },
            ),
        
            "too_hot_sleep_issue": AspectRule(
                aspect_code="too_hot_sleep_issue",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bжарко\b",
                        r"\bочень жарко\b",
                        r"\bспать жарко\b",
                        r"\bневозможно спать от жары\b",
                        r"\bночью душно и жарко\b",
                    ],
                    "en": [
                        r"\btoo hot\b",
                        r"\bvery hot in the room\b",
                        r"\bhard to sleep because it was hot\b",
                        r"\bwe couldn't sleep because it was so hot\b",
                    ],
                    "tr": [
                        r"\boda çok sıcaktı\b",
                        r"\buyuyamayacak kadar sıcaktı\b",
                        r"\bgece çok sıcaktı\b",
                        r"\bçok sıcak uyumak zor\b",
                    ],
                    "ar": [
                        r"\bحر كتير\b",
                        r"\bحر ما منقدر ننام\b",
                        r"\bالغرفة حامية\b",
                        r"\bما قدرنا ننام من الحر\b",
                    ],
                    "zh": [
                        r"房间太热",
                        r"太热了睡不着",
                        r"晚上很闷很热",
                        r"太热影响睡觉",
                    ],
                },
            ),
        
            "too_cold": AspectRule(
                aspect_code="too_cold",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bслишком холодно\b",
                        r"\bв номере холодно\b",
                        r"\bочень холодно в комнате\b",
                        r"\bхолодно спать\b",
                    ],
                    "en": [
                        r"\btoo cold\b",
                        r"\bcold in the room\b",
                        r"\bthe room was freezing\b",
                        r"\bit was very cold inside\b",
                    ],
                    "tr": [
                        r"\boda soğuktu\b",
                        r"\bçok soğuktu içerisi\b",
                        r"\bgece çok soğuktu\b",
                    ],
                    "ar": [
                        r"\bبرد بالغرفة\b",
                        r"\bالغرفة باردة\b",
                        r"\bكان برد كتير جوّا\b",
                    ],
                    "zh": [
                        r"房间很冷",
                        r"很冷",
                        r"屋里特别冷",
                        r"晚上冷得不行",
                    ],
                },
            ),
        
            "stuffy_no_air": AspectRule(
                aspect_code="stuffy_no_air",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bдушно\b",
                        r"\bнечем дышать\b",
                        r"\bне проветривается\b",
                        r"\bв комнате спертый воздух\b",
                        r"\bвоздух тяжёлый\b",
                    ],
                    "en": [
                        r"\bstuffy\b",
                        r"\bno air\b",
                        r"\bno ventilation\b",
                        r"\bair felt stuffy\b",
                        r"\bthe room felt stuffy\b",
                    ],
                    "tr": [
                        r"\bhava boğucuydu\b",
                        r"\bhava alamadık\b",
                        r"\bhavalandırma yoktu\b",
                        r"\bodada çok havasızdı\b",
                    ],
                    "ar": [
                        r"\bمخنوقين\b",
                        r"\bما في هوا\b",
                        r"\bما في تهوية\b",
                        r"\bالجو خانق جوا\b",
                    ],
                    "zh": [
                        r"很闷",
                        r"空气很闷",
                        r"没有空气流通",
                        r"没有通风",
                        r"房间空气很闷",
                    ],
                },
            ),
            
            "no_ventilation": AspectRule(
                aspect_code="no_ventilation",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bне проветривается\b",
                        r"\bневозможно проветрить\b",
                        r"\bнет проветривания\b",
                        r"\bнет воздуха\b",
                    ],
                    "en": [
                        r"\bno ventilation\b",
                        r"\bno way to air the room\b",
                        r"\bcouldn't air the room\b",
                        r"\bno fresh air\b",
                    ],
                    "tr": [
                        r"\bhavalandırma yoktu\b",
                        r"\bhava alamadık\b",
                        r"\bodada hava dolaşımı yoktu\b",
                        r"\bodanın içi havasızdı\b",
                    ],
                    "ar": [
                        r"\bما في تهوية\b",
                        r"\bما في هوا نظيف\b",
                        r"\bما في ولا هوا بالغرفة\b",
                        r"\bالغرفة بدون تهوية\b",
                    ],
                    "zh": [
                        r"没有通风",
                        r"房间完全不通风",
                        r"没有新鲜空气",
                        r"空气不流通",
                    ],
                },
            ),
        
            "ac_not_working": AspectRule(
                aspect_code="ac_not_working",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bкондиционер не работал\b",
                        r"\bкондиционер сломан\b",
                        r"\bкондиционер не спасал\b",
                        r"\bкондиционер почти не дул\b",
                    ],
                    "en": [
                        r"\bAC didn't work\b",
                        r"\bAC was not working\b",
                        r"\bair conditioner not working\b",
                        r"\bthe air con was broken\b",
                    ],
                    "tr": [
                        r"\bklima çalışmıyordu\b",
                        r"\bklima bozuktu\b",
                        r"\bklima işe yaramıyordu\b",
                    ],
                    "ar": [
                        r"\bالمكيف ما بيشتغل\b",
                        r"\bالمكيف خربان\b",
                        r"\bالتكييف ما اشتغل\b",
                    ],
                    "zh": [
                        r"空调不好用",
                        r"空调不工作",
                        r"空调坏了",
                        r"空调几乎没用",
                    ],
                },
            ),
        
            "no_ac": AspectRule(
                aspect_code="no_ac",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bкондиционера нет\b",
                        r"\bнет кондиционера\b",
                        r"\bбез кондиционера\b",
                    ],
                    "en": [
                        r"\bno AC\b",
                        r"\bno air conditioning\b",
                        r"\bthere was no AC\b",
                    ],
                    "tr": [
                        r"\bklima yoktu\b",
                        r"\bodada klima yoktu\b",
                        r"\bklima hiç yok\b",
                    ],
                    "ar": [
                        r"\bما في مكيف\b",
                        r"\bما كان في تكييف\b",
                        r"\bالغرفة بدون مكيف\b",
                    ],
                    "zh": [
                        r"没有空调",
                        r"房间里没有空调",
                        r"没配空调",
                    ],
                },
            ),
        
            "heating_not_working": AspectRule(
                aspect_code="heating_not_working",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bотопление не работало\b",
                        r"\bобогрев не работал\b",
                        r"\bхолодные батареи\b",
                        r"\bбатареи не греют\b",
                    ],
                    "en": [
                        r"\bheating didn't work\b",
                        r"\bthe heater was not working\b",
                        r"\bno heating\b",
                        r"\bheating was off and it was cold\b",
                    ],
                    "tr": [
                        r"\bısıtma çalışmıyordu\b",
                        r"\bısıtma yoktu\b",
                        r"\bodada ısıtma yoktu\b",
                    ],
                    "ar": [
                        r"\bالتدفئة ما اشتغلت\b",
                        r"\bما في تدفئة\b",
                        r"\bالدفاية ما عم تشتغل\b",
                    ],
                    "zh": [
                        r"暖气不工作",
                        r"没有暖气",
                        r"暖气是冷的",
                        r"房间没有供暖",
                    ],
                },
            ),
        
            "draft_window": AspectRule(
                aspect_code="draft_window",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bсквозняк\b",
                        r"\bдует из окна\b",
                        r"\bхолодно от окна\b",
                        r"\bпродувает через окно\b",
                    ],
                    "en": [
                        r"\bdraft from the window\b",
                        r"\bcold draft\b",
                        r"\bcold air coming from the window\b",
                        r"\bwind coming through the window\b",
                    ],
                    "tr": [
                        r"\bcamdan rüzgar geliyordu\b",
                        r"\bcereyan vardı\b",
                        r"\bpencere çekiyordu\b",
                        r"\bpencereden soğuk hava geliyordu\b",
                    ],
                    "ar": [
                        r"\bفي هوا بارد من الشباك\b",
                        r"\bفي هوا داخل من الشباك\b",
                        r"\bهوا عم يفوت من الشبابيك\b",
                    ],
                    "zh": [
                        r"窗户漏风",
                        r"有冷风进来",
                        r"窗户关不严有风",
                        r"窗边一直有冷风",
                    ],
                },
            ),
            
            "room_spacious": AspectRule(
                aspect_code="room_spacious",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bпросторный номер\b",
                        r"\bмного места\b",
                        r"\bномер большой\b",
                        r"\bномер просторный\b",
                        r"\bмного пространства\b",
                    ],
                    "en": [
                        r"\bspacious room\b",
                        r"\bthe room was spacious\b",
                        r"\ba lot of space\b",
                        r"\broom was big\b",
                        r"\bvery spacious\b",
                    ],
                    "tr": [
                        r"\boda genişti\b",
                        r"\bferah oda\b",
                        r"\bçok boş alan vardı\b",
                        r"\boda çok ferahtı\b",
                    ],
                    "ar": [
                        r"\bالغرفة واسعة\b",
                        r"\bمساحة واسعة\b",
                        r"\bفيها مجال\b",
                        r"\bالغرفة كتير فسيحة\b",
                    ],
                    "zh": [
                        r"房间很大",
                        r"很宽敞",
                        r"空间很大",
                        r"房间很宽敞不压抑",
                    ],
                },
            ),
        
            "good_layout": AspectRule(
                aspect_code="good_layout",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bудобная планировка\b",
                        r"\bвсё удобно расположено\b",
                        r"\bграмотно организовано пространство\b",
                        r"\bхорошо продумано как всё стоит\b",
                    ],
                    "en": [
                        r"\bgood layout\b",
                        r"\bwell organized\b",
                        r"\bsmart use of space\b",
                        r"\bthe room was well laid out\b",
                        r"\bthe layout made sense\b",
                    ],
                    "tr": [
                        r"\bdüzeni iyiydi\b",
                        r"\bdüzen çok kullanışlıydı\b",
                        r"\bodanın yerleşimi mantıklıydı\b",
                        r"\boda düzeni rahattı\b",
                    ],
                    "ar": [
                        r"\bمرتبة بشكل مريح\b",
                        r"\bكل شي بمكانه\b",
                        r"\bالترتيب عملي\b",
                        r"\bتوزيع الغرفة مريح\b",
                    ],
                    "zh": [
                        r"布局很合理",
                        r"摆放得很合理",
                        r"空间利用得很好",
                        r"房间动线很好",
                    ],
                },
            ),
        
            "cozy_feel": AspectRule(
                aspect_code="cozy_feel",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bуютный номер\b",
                        r"\bочень уютно\b",
                        r"\bуютно\b",
                        r"\bочень домашне\b",
                        r"\bприятная атмосфера в номере\b",
                    ],
                    "en": [
                        r"\bcozy\b",
                        r"\bvery cozy\b",
                        r"\bfelt cozy\b",
                        r"\bcozy room\b",
                        r"\bthe room felt homelike\b",
                    ],
                    "tr": [
                        r"\boda çok rahat bir his veriyor\b",
                        r"\bçok samimi bir atmosfer vardı\b",
                        r"\bev gibi hissettirdi\b",
                        r"\boda çok sıcak bir his veriyor\b",
                    ],
                    "ar": [
                        r"\bالغرفة مريحة ودافئة\b",
                        r"\bجو كتير دافئ\b",
                        r"\bبتحس كأنك ببيتك\b",
                        r"\bبتحس المكان مريح\b",
                    ],
                    "zh": [
                        r"很温馨",
                        r"很舒适",
                        r"很有家的感觉",
                        r"房间感觉很温馨",
                    ],
                },
            ),
        
            "bright_room": AspectRule(
                aspect_code="bright_room",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bсветлый номер\b",
                        r"\bмного света\b",
                        r"\bмного дневного света\b",
                        r"\bочень светло в номере\b",
                    ],
                    "en": [
                        r"\bbright room\b",
                        r"\blots of natural light\b",
                        r"\ba lot of daylight\b",
                        r"\bthe room was very bright\b",
                    ],
                    "tr": [
                        r"\baydınlık odaydı\b",
                        r"\boda çok aydınlıktı\b",
                        r"\bdoğal ışık çoktu\b",
                        r"\bgüneş alan bir odaydı\b",
                    ],
                    "ar": [
                        r"\bالغرفة منوّرة\b",
                        r"\bفيها ضو طبيعي\b",
                        r"\bالغرفة فيها كتير ضو\b",
                        r"\bالغرفة فاتحة وبتفتح النفس\b",
                    ],
                    "zh": [
                        r"房间很亮",
                        r"光线很好",
                        r"有自然光",
                        r"阳光很好",
                    ],
                },
            ),
        
            "big_windows": AspectRule(
                aspect_code="big_windows",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bбольшие окна\b",
                        r"\bогромные окна\b",
                        r"\bпанорамные окна\b",
                    ],
                    "en": [
                        r"\bbig windows\b",
                        r"\blarge windows\b",
                        r"\bfloor to ceiling windows\b",
                        r"\bhuge windows\b",
                    ],
                    "tr": [
                        r"\bbüyük pencere vardı\b",
                        r"\bpencereler büyüktü\b",
                        r"\bferah geniş pencereler vardı\b",
                    ],
                    "ar": [
                        r"\bفي شبابيك كبيرة\b",
                        r"\bالشباك كبير\b",
                        r"\bفي شبابيك واسعة بتفوت ضو\b",
                    ],
                    "zh": [
                        r"窗户很大",
                        r"大窗户",
                        r"落地窗",
                        r"窗户特别大采光很好",
                    ],
                },
            ),
            
            "room_small": AspectRule(
                aspect_code="room_small",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bтесный номер\b",
                        r"\bномер очень маленький\b",
                        r"\bочень тесно\b",
                        r"\bмало места\b",
                        r"\bне развернуться\b",
                    ],
                    "en": [
                        r"\bsmall room\b",
                        r"\bvery small room\b",
                        r"\bvery small\b",
                        r"\bcramped\b",
                        r"\bhard to move around\b",
                    ],
                    "tr": [
                        r"\boda küçüktü\b",
                        r"\bçok küçüktü\b",
                        r"\bsıkışıktı\b",
                        r"\bhareket etmek zor\b",
                    ],
                    "ar": [
                        r"\bالغرفة صغيرة\b",
                        r"\bكتير صغيرة\b",
                        r"\bمخانقة\b",
                        r"\bما في مجال تتحرك\b",
                    ],
                    "zh": [
                        r"房间很小",
                        r"很挤",
                        r"空间很小",
                        r"走不开",
                    ],
                },
            ),
        
            "no_space_for_luggage": AspectRule(
                aspect_code="no_space_for_luggage",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bнекуда поставить чемодан\b",
                        r"\bнекуда положить чемодан\b",
                        r"\bс чемоданом неудобно\b",
                        r"\bне было места для чемодана\b",
                    ],
                    "en": [
                        r"\bno space for luggage\b",
                        r"\bno room for suitcases\b",
                        r"\bno place to put the suitcase\b",
                        r"\bnowhere to put our bags\b",
                    ],
                    "tr": [
                        r"\bbavulu koyacak yer yoktu\b",
                        r"\bvaliz için yer yoktu\b",
                        r"\bbavulla çok zordu\b",
                        r"\bvalizi açacak yer yoktu\b",
                    ],
                    "ar": [
                        r"\bما في محل للشنط\b",
                        r"\bما في مكان نفتح الشنطة\b",
                        r"\bمع الشنط الوضع مو مريح\b",
                        r"\bما في وين نحط الشنط\b",
                    ],
                    "zh": [
                        r"没地方放行李",
                        r"行李没法打开",
                        r"房间里没空间放箱子",
                        r"带箱子很不方便",
                    ],
                },
            ),
        
            "dark_room": AspectRule(
                aspect_code="dark_room",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bтемно в номере\b",
                        r"\bмало света\b",
                        r"\bмало освещения\b",
                        r"\bочень тёмный номер\b",
                        r"\bмрачно в номере\b",
                    ],
                    "en": [
                        r"\bdark room\b",
                        r"\bthe room was dark\b",
                        r"\bnot enough light\b",
                        r"\bpoor lighting\b",
                        r"\bvery little light in the room\b",
                    ],
                    "tr": [
                        r"\boda karanlıktı\b",
                        r"\byeterince ışık yoktu\b",
                        r"\bçok loştu\b",
                        r"\bışık azdı odada\b",
                    ],
                    "ar": [
                        r"\bالغرفة معتمة\b",
                        r"\bما فيها ضو\b",
                        r"\bالإضاءة ضعيفة\b",
                        r"\bالغرفة كتير غامقة\b",
                    ],
                    "zh": [
                        r"房间很暗",
                        r"灯光不够",
                        r"光线很差",
                        r"房间几乎没光",
                    ],
                },
            ),
        
            "no_natural_light": AspectRule(
                aspect_code="no_natural_light",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bпочти нет окна\b",
                        r"\bокно маленькое\b",
                        r"\bнет дневного света\b",
                        r"\bпочти нет естественного света\b",
                    ],
                    "en": [
                        r"\bno natural light\b",
                        r"\bno daylight\b",
                        r"\bno window\b",
                        r"\btiny window\b",
                        r"\balmost no window\b",
                    ],
                    "tr": [
                        r"\bdoğal ışık yoktu\b",
                        r"\bpencere çok küçüktü\b",
                        r"\bpencere yok gibi\b",
                        r"\bgün ışığı neredeyse yoktu\b",
                    ],
                    "ar": [
                        r"\bما فيها ضو طبيعي\b",
                        r"\bما في شباك تقريبا\b",
                        r"\bشباك صغير كتير\b",
                        r"\bما بتفوت شمس\b",
                    ],
                    "zh": [
                        r"没有自然光",
                        r"几乎没阳光",
                        r"没有窗户",
                        r"窗户很小几乎没光进来",
                    ],
                },
            ),
        
            "gloomy_feel": AspectRule(
                aspect_code="gloomy_feel",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bмрачно\b",
                        r"\bдавит\b",
                        r"\bугнетающе\b",
                        r"\bтяжелая атмосфера\b",
                        r"\bнеуютно\b",
                        r"\bнеуютная атмосфера\b",
                    ],
                    "en": [
                        r"\bfelt gloomy\b",
                        r"\bfelt depressing\b",
                        r"\bgloomy atmosphere\b",
                        r"\bnot cozy\b",
                        r"\bdidn't feel welcoming\b",
                        r"\bcold atmosphere\b",
                    ],
                    "tr": [
                        r"\bortam biraz kasvetliydi\b",
                        r"\brahat hissettirmedi\b",
                        r"\batmosfer pek sıcak değildi\b",
                        r"\bsoğuk bir his vardı\b",
                    ],
                    "ar": [
                        r"\bالمكان كئيب\b",
                        r"\bشوي كئيب\b",
                        r"\bالجو بارد\b",
                        r"\bمش مريح\b",
                        r"\bما في راحة بالمكان\b",
                    ],
                    "zh": [
                        r"有点压抑",
                        r"感觉有点压抑",
                        r"氛围有点冷淡",
                        r"没有家的感觉",
                        r"不太温馨",
                    ],
                },
            ),
            
            "hot_water_ok": AspectRule(
                aspect_code="hot_water_ok",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bгорячая вода сразу\b",
                        r"\bгорячая вода без перебоев\b",
                        r"\bгорячая вода всегда была\b",
                        r"\bс горячей водой всё ок\b",
                    ],
                    "en": [
                        r"\bhot water right away\b",
                        r"\bgood hot water\b",
                        r"\bhot water all the time\b",
                        r"\bplenty of hot water\b",
                    ],
                    "tr": [
                        r"\bsıcak su hemen vardı\b",
                        r"\bsıcak su sürekli vardı\b",
                        r"\bsıcak su sorunsuzdu\b",
                    ],
                    "ar": [
                        r"\bفي مي سخنة ع طول\b",
                        r"\bفي مي سخنة بدون انقطاع\b",
                        r"\bالمي السخنة شغالة طول الوقت\b",
                    ],
                    "zh": [
                        r"热水马上就有",
                        r"热水很稳定",
                        r"一直都有热水",
                        r"热水供应很好",
                    ],
                },
            ),
        
            "water_pressure_ok": AspectRule(
                aspect_code="water_pressure_ok",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bнормальное давление воды\b",
                        r"\bхорошее давление\b",
                        r"\bсильный напор\b",
                        r"\bнапор отличный\b",
                    ],
                    "en": [
                        r"\bgood water pressure\b",
                        r"\bstrong water pressure\b",
                        r"\bwater pressure was great\b",
                    ],
                    "tr": [
                        r"\bsu basıncı iyiydi\b",
                        r"\bbasınç güçlüydü\b",
                        r"\bsu basıncı çok iyiydi\b",
                    ],
                    "ar": [
                        r"\bضغط المي منيح\b",
                        r"\bالضغط قوي\b",
                        r"\bالضغط كان ممتاز\b",
                    ],
                    "zh": [
                        r"水压很好",
                        r"水压很大",
                        r"水压不错",
                    ],
                },
            ),
        
            "shower_ok": AspectRule(
                aspect_code="shower_ok",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bдуш работал отлично\b",
                        r"\bдуш работал хорошо\b",
                        r"\bв душе всё работало\b",
                        r"\bдуш нормальный\b",
                    ],
                    "en": [
                        r"\bshower worked fine\b",
                        r"\bshower worked perfectly\b",
                        r"\bshower was good\b",
                    ],
                    "tr": [
                        r"\bduş sorunsuz çalışıyordu\b",
                        r"\bduş iyi çalışıyordu\b",
                        r"\bduşta sıkıntı yoktu\b",
                    ],
                    "ar": [
                        r"\bالدوش شغال تمام\b",
                        r"\bالدوش كان منيح\b",
                        r"\bالشاور شغال بدون مشاكل\b",
                    ],
                    "zh": [
                        r"淋浴正常",
                        r"淋浴很好用",
                        r"洗澡水很正常",
                    ],
                },
            ),
        
            "no_leak": AspectRule(
                aspect_code="no_leak",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bничего не текло\b",
                        r"\bничего не капало\b",
                        r"\bникаких протечек\b",
                        r"\bвсе краны без протечек\b",
                    ],
                    "en": [
                        r"\bno leaks\b",
                        r"\bno leaking\b",
                        r"\bno dripping\b",
                        r"\bno water leakage\b",
                    ],
                    "tr": [
                        r"\bhiç sızıntı yoktu\b",
                        r"\bsu kaçırma yoktu\b",
                        r"\bmusluk akıtmıyordu\b",
                    ],
                    "ar": [
                        r"\bما في تسريب\b",
                        r"\bما في مي عم تنقط\b",
                        r"\bما كان يهرب مي\b",
                    ],
                    "zh": [
                        r"没有漏水",
                        r"水龙头不漏",
                        r"没有渗水",
                    ],
                },
            ),
        
            "no_hot_water": AspectRule(
                aspect_code="no_hot_water",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bне было горячей воды\b",
                        r"\bбез горячей воды\b",
                        r"\bнет горячей воды утром\b",
                        r"\bгорячая вода пропадала\b",
                        r"\bгорячая вода только на пару минут\b",
                    ],
                    "en": [
                        r"\bno hot water\b",
                        r"\bno hot water in the morning\b",
                        r"\bhot water cuts off\b",
                        r"\bhot water stops after a minute\b",
                    ],
                    "tr": [
                        r"\bsıcak su yoktu\b",
                        r"\bsabah sıcak su yoktu\b",
                        r"\bsıcak su gidip geliyordu\b",
                        r"\bsıcak su birden kesiliyordu\b",
                    ],
                    "ar": [
                        r"\bما في مي سخنة\b",
                        r"\bما كان في مي سخنة الصبح\b",
                        r"\bالمي السخنة بتقطع\b",
                        r"\bالمي السخنة وقفت\b",
                    ],
                    "zh": [
                        r"没有热水",
                        r"早上没有热水",
                        r"热水用一下就没了",
                        r"热水一会儿就没了",
                    ],
                },
            ),
            
            "weak_pressure": AspectRule(
                aspect_code="weak_pressure",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bслабый напор\b",
                        r"\bслабое давление\b",
                        r"\bеле теч(е|ё)т\b",
                        r"\bвода еле теч(е|ё)т\b",
                        r"\bдавление воды очень слабое\b",
                    ],
                    "en": [
                        r"\blow water pressure\b",
                        r"\bweak water pressure\b",
                        r"\bthe water pressure was weak\b",
                        r"\bbarely any pressure\b",
                    ],
                    "tr": [
                        r"\bsu basıncı çok düşüktü\b",
                        r"\bbasınç zayıftı\b",
                        r"\bsu çok az akıyordu\b",
                        r"\bneredeyse su akmıyordu\b",
                    ],
                    "ar": [
                        r"\bالضغط ضعيف\b",
                        r"\bضغط المي كتير ضعيف\b",
                        r"\bالمي طالعة خفيفة\b",
                        r"\bما في ضغط مي منيح\b",
                    ],
                    "zh": [
                        r"水压很低",
                        r"水压很小",
                        r"水流得很小",
                        r"水几乎不出来",
                    ],
                },
            ),
        
            "shower_broken": AspectRule(
                aspect_code="shower_broken",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bдуш сломан\b",
                        r"\bсломанный душ\b",
                        r"\bдуш не держится\b",
                        r"\bлейка не держится\b",
                        r"\bдержатель душа сломан\b",
                    ],
                    "en": [
                        r"\bshower was broken\b",
                        r"\bshower head broken\b",
                        r"\bshower holder broken\b",
                        r"\bthe shower wouldn't stay up\b",
                        r"\bthe shower didn't work\b",
                    ],
                    "tr": [
                        r"\bduş bozuktu\b",
                        r"\bduş kafası kırıktı\b",
                        r"\bduş askısı kırıktı\b",
                        r"\bduş sabit durmuyordu\b",
                    ],
                    "ar": [
                        r"\bالدوش خربان\b",
                        r"\bرأس الدش مكسور\b",
                        r"\bما بيثبت الدوش\b",
                        r"\bالشور ما عم يشتغل منيح\b",
                    ],
                    "zh": [
                        r"淋浴坏了",
                        r"花洒坏了",
                        r"花洒架坏了",
                        r"淋浴头挂不住",
                        r"淋浴不好用",
                    ],
                },
            ),
        
            "leak_water": AspectRule(
                aspect_code="leak_water",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bкран теч(е|ё)т\b",
                        r"\bвода капает\b",
                        r"\bпротечка\b",
                        r"\bподтекает\b",
                        r"\bвсё текло\b",
                        r"\bвода сочилась\b",
                    ],
                    "en": [
                        r"\bleaking tap\b",
                        r"\bthe tap was leaking\b",
                        r"\bwater was leaking\b",
                        r"\bwater was leaking everywhere\b",
                        r"\bleaking everywhere\b",
                    ],
                    "tr": [
                        r"\bmusluk akıtıyordu\b",
                        r"\blavabo akıtıyordu\b",
                        r"\bsu sızdırıyordu\b",
                        r"\bsu kaçırıyordu\b",
                    ],
                    "ar": [
                        r"\bالحنفية عم تسرّب\b",
                        r"\bفي تسريب مي\b",
                        r"\bالمي عم تنقط\b",
                        r"\bالمي عم تطلع لبرا\b",
                    ],
                    "zh": [
                        r"水龙头漏水",
                        r"一直滴水",
                        r"有漏水",
                        r"到处都在漏水",
                    ],
                },
            ),
        
            "bathroom_flooding": AspectRule(
                aspect_code="bathroom_flooding",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bвода на полу после душа\b",
                        r"\bвся ванная в воде\b",
                        r"\bзалило пол\b",
                        r"\bпол весь мокрый после душа\b",
                    ],
                    "en": [
                        r"\bflooded bathroom\b",
                        r"\bwater all over the floor\b",
                        r"\bthe floor was covered in water after showering\b",
                        r"\bbathroom floor was soaked\b",
                    ],
                    "tr": [
                        r"\bduştan sonra her yer su oldu\b",
                        r"\byer hep ıslaktı\b",
                        r"\bbanyo su bastı\b",
                        r"\bbanyo zemin su içindeydi\b",
                    ],
                    "ar": [
                        r"\bمي بكل الحمام\b",
                        r"\bالأرض كلها مي بعد الشور\b",
                        r"\bالمية غرقت الحمام\b",
                        r"\bالأرض غرقت مي\b",
                    ],
                    "zh": [
                        r"洗完澡地上都是水",
                        r"卫生间全是水",
                        r"整个卫生间都是水",
                        r"地上全是水",
                    ],
                },
            ),
        
            "drain_clogged": AspectRule(
                aspect_code="drain_clogged",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bзасор в раковине\b",
                        r"\bзасор в душе\b",
                        r"\bслив не работает\b",
                        r"\bплохо уходит вода\b",
                        r"\bвода не уходит\b",
                    ],
                    "en": [
                        r"\bclogged drain\b",
                        r"\bdrain was clogged\b",
                        r"\bsink clogged\b",
                        r"\bshower drain clogged\b",
                        r"\bwater didn't drain\b",
                    ],
                    "tr": [
                        r"\blavabo tıkalıydı\b",
                        r"\bduş gideri tıkalıydı\b",
                        r"\bgider çalışmıyordu\b",
                        r"\bsu gitmiyordu\b",
                    ],
                    "ar": [
                        r"\bالمصرف مسدود\b",
                        r"\bالمجلى مسدود\b",
                        r"\bالبالوعة مسكّرة\b",
                        r"\bالمي ما عم تنزل\b",
                    ],
                    "zh": [
                        r"下水道堵了",
                        r"排水不通",
                        r"洗手池堵了",
                        r"淋浴下水道堵了",
                        r"水下不去",
                    ],
                },
            ),
        
            "drain_smell": AspectRule(
                aspect_code="drain_smell",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bвонял[ao]? из слива\b",
                        r"\bзапах из труб\b",
                        r"\bзапах канализации из душа\b",
                        r"\bвонь из раковины\b",
                    ],
                    "en": [
                        r"\bbad smell from the drain\b",
                        r"\bsewage smell from the shower\b",
                        r"\bsmell coming from the pipes\b",
                        r"\bbad smell from the sink\b",
                    ],
                    "tr": [
                        r"\bgidermekten kötü koku geliyordu\b",
                        r"\blağım kokusu\b",
                        r"\bduş giderinden kötü koku geliyordu\b",
                        r"\blavabodan koku geliyordu\b",
                    ],
                    "ar": [
                        r"\bريحة مجاري من البالوعة\b",
                        r"\bريحة مجاري من الحمام\b",
                        r"\bريحة صرف\b",
                        r"\bريحة طالعة من المجلى\b",
                    ],
                    "zh": [
                        r"下水道有臭味",
                        r"排水口有臭味",
                        r"卫生间有下水道味",
                        r"有下水道的味道",
                    ],
                },
            ),
            
            "ac_working_device": AspectRule(
                aspect_code="ac_working_device",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bкондиционер работал\b",
                        r"\bкондиционер отлично работал\b",
                        r"\bкондиционер работал нормально\b",
                        r"\bкондиционер исправен\b",
                    ],
                    "en": [
                        r"\bAC worked fine\b",
                        r"\bAC worked well\b",
                        r"\bair conditioning worked well\b",
                        r"\bthe air conditioner was working\b",
                    ],
                    "tr": [
                        r"\bklima çalışıyordu\b",
                        r"\bklima sorunsuzdu\b",
                        r"\bklima düzgün çalışıyordu\b",
                    ],
                    "ar": [
                        r"\bالمكيف شغال\b",
                        r"\bالمكيف شغال تمام\b",
                        r"\bالتكييف شغال منيح\b",
                    ],
                    "zh": [
                        r"空调正常工作",
                        r"空调很好用",
                        r"空调没问题",
                    ],
                },
            ),
        
            "heating_working_device": AspectRule(
                aspect_code="heating_working_device",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bотопление работало\b",
                        r"\bотопление нормальное\b",
                        r"\bобогрев работал\b",
                        r"\bбатареи тёплые\b",
                    ],
                    "en": [
                        r"\bheating worked\b",
                        r"\bthe heater worked\b",
                        r"\bheating was working\b",
                        r"\bthe heating was fine\b",
                    ],
                    "tr": [
                        r"\bısıtma çalışıyordu\b",
                        r"\bısıtma sorunsuzdu\b",
                        r"\boda ısındı\b",
                    ],
                    "ar": [
                        r"\bالتدفئة شغالة\b",
                        r"\bالدفاية شغالة\b",
                        r"\bالتدفئة كانت تمام\b",
                    ],
                    "zh": [
                        r"暖气正常",
                        r"暖气在工作",
                        r"房间有暖气而且很暖",
                    ],
                },
            ),
        
            "appliances_ok": AspectRule(
                aspect_code="appliances_ok",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bвсё оборудование исправно\b",
                        r"\bвсё работало\b",
                        r"\bвсё в номере работало\b",
                        r"\bничего не сломано\b",
                    ],
                    "en": [
                        r"\beverything worked\b",
                        r"\beverything in the room was working\b",
                        r"\ball appliances worked\b",
                        r"\bnothing was broken\b",
                    ],
                    "tr": [
                        r"\bher şey çalışıyordu\b",
                        r"\bodadaki tüm ekipman sorunsuzdu\b",
                        r"\bhiçbir şey bozuk değildi\b",
                    ],
                    "ar": [
                        r"\bكل الأجهزة شغالة\b",
                        r"\bكل شي بالغرفة كان شغال\b",
                        r"\bما في شي خربان\b",
                    ],
                    "zh": [
                        r"所有东西都正常",
                        r"房间里的设备都能用",
                        r"没有坏的东西",
                    ],
                },
            ),
        
            "tv_working": AspectRule(
                aspect_code="tv_working",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bтелевизор работает\b",
                        r"\bтелевизор показывал нормально\b",
                        r"\bтв работал\b",
                    ],
                    "en": [
                        r"\bTV worked\b",
                        r"\bTV was working\b",
                        r"\bTV channels were fine\b",
                        r"\bthe television worked\b",
                    ],
                    "tr": [
                        r"\bTV çalışıyordu\b",
                        r"\btelevizyon sorunsuzdu\b",
                        r"\bkanallar vardı\b",
                    ],
                    "ar": [
                        r"\bالتلفزيون شغال\b",
                        r"\bالتلفزيون كان شغال تمام\b",
                        r"\bالقنوات شغالة\b",
                    ],
                    "zh": [
                        r"电视能看",
                        r"电视正常工作",
                        r"电视有频道可以看",
                    ],
                },
            ),
        
            "fridge_working": AspectRule(
                aspect_code="fridge_working",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bхолодильник работает\b",
                        r"\bхолодильник нормально холодил\b",
                        r"\bхолодильник исправен\b",
                    ],
                    "en": [
                        r"\bfridge worked\b",
                        r"\bfridge was working\b",
                        r"\bfridge was cold\b",
                        r"\bthe refrigerator worked fine\b",
                    ],
                    "tr": [
                        r"\bbuzdolabı çalışıyordu\b",
                        r"\bbuzdolabı sorunsuzdu\b",
                        r"\bbuzdolabı iyi soğutuyordu\b",
                    ],
                    "ar": [
                        r"\bالبراد شغال\b",
                        r"\bالبراد كان يبرد منيح\b",
                        r"\bالبراد شغال تمام\b",
                    ],
                    "zh": [
                        r"冰箱正常",
                        r"冰箱在制冷",
                        r"冰箱工作正常",
                    ],
                },
            ),

            "kettle_working": AspectRule(
                aspect_code="kettle_working",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bчайник работает\b",
                        r"\bчайник был рабочий\b",
                        r"\bчайник исправен\b",
                        r"\bчайник нормально грел\b",
                    ],
                    "en": [
                        r"\bkettle worked\b",
                        r"\bthe kettle was working\b",
                        r"\bthe kettle worked fine\b",
                        r"\bwater boiler worked\b",
                    ],
                    "tr": [
                        r"\bsu ısıtıcısı çalışıyordu\b",
                        r"\bsu ısıtıcısı sorunsuzdu\b",
                        r"\bketıl çalışıyordu\b",
                    ],
                    "ar": [
                        r"\bالكاتل شغال\b",
                        r"\bالسخان شغال\b",
                        r"\bالغلاية شغالة تمام\b",
                    ],
                    "zh": [
                        r"烧水壶能用",
                        r"水壶正常工作",
                        r"热水壶可以正常烧水",
                    ],
                },
            ),
        
            "door_secure": AspectRule(
                aspect_code="door_secure",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bдверь закрывается плотно\b",
                        r"\bдверь нормально закрывалась\b",
                        r"\bнормальный замок\b",
                        r"\bзамок нормальный\b",
                        r"\bчувствовали себя в безопасности\b",
                        r"\bбезопасно хранить вещи\b",
                    ],
                    "en": [
                        r"\bdoor closed properly\b",
                        r"\bthe lock felt secure\b",
                        r"\bthe door locked well\b",
                        r"\bwe felt safe\b",
                        r"\bfelt safe leaving our stuff\b",
                    ],
                    "tr": [
                        r"\bkapı düzgün kapanıyordu\b",
                        r"\bkilit sağlamdı\b",
                        r"\bkendimizi güvende hissettik\b",
                    ],
                    "ar": [
                        r"\bالباب بيسكّر منيح\b",
                        r"\bالقفل منيح\b",
                        r"\bحاسين بأمان\b",
                        r"\bمأمنين على أغراضنا\b",
                    ],
                    "zh": [
                        r"门关得很严",
                        r"门锁很安全",
                        r"锁很靠谱",
                        r"感觉很安全",
                        r"放心把东西放房间",
                    ],
                },
            ),
        
            "ac_broken": AspectRule(
                aspect_code="ac_broken",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bкондиционер не работал\b",
                        r"\bкондиционер сломан\b",
                        r"\bкондиционер не включался\b",
                        r"\bкондиционер еле дул\b",
                        r"\bкондиционер не охлаждал\b",
                    ],
                    "en": [
                        r"\bAC didn't work\b",
                        r"\bAC was broken\b",
                        r"\bair conditioner not working\b",
                        r"\bthe air con didn't cool\b",
                    ],
                    "tr": [
                        r"\bklima çalışmıyordu\b",
                        r"\bklima bozuktu\b",
                        r"\bklima soğutmuyordu\b",
                    ],
                    "ar": [
                        r"\bالمكيف ما بيشتغل\b",
                        r"\bالمكيف خربان\b",
                        r"\bالمكيف ما عم يبرد\b",
                    ],
                    "zh": [
                        r"空调不好用",
                        r"空调不工作",
                        r"空调坏了",
                        r"空调不制冷",
                    ],
                },
            ),
        
            "heating_broken": AspectRule(
                aspect_code="heating_broken",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bотопление не работало\b",
                        r"\bобогрев не работал\b",
                        r"\bбатареи холодные\b",
                        r"\bв номере холодно из-за отопления\b",
                    ],
                    "en": [
                        r"\bheating didn't work\b",
                        r"\bthe heater was not working\b",
                        r"\bno heating\b",
                        r"\bthe radiators were cold\b",
                    ],
                    "tr": [
                        r"\bısıtma çalışmıyordu\b",
                        r"\bodada ısıtma yoktu\b",
                        r"\bradyatör çalışmıyordu\b",
                    ],
                    "ar": [
                        r"\bالتدفئة ما اشتغلت\b",
                        r"\bما في تدفئة\b",
                        r"\bالدفاية ما عم تشتغل\b",
                        r"\bالدفاية باردة\b",
                    ],
                    "zh": [
                        r"暖气不工作",
                        r"没有暖气",
                        r"暖气是冷的",
                        r"房间很冷因为没有暖气",
                    ],
                },
            ),
        
            "tv_broken": AspectRule(
                aspect_code="tv_broken",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bтелевизор не работал\b",
                        r"\bтелевизор не показывал\b",
                        r"\bтв не работал\b",
                        r"\bне показывали каналы\b",
                        r"\bтв без каналов\b",
                    ],
                    "en": [
                        r"\bTV didn't work\b",
                        r"\bTV was not working\b",
                        r"\bTV had no channels\b",
                        r"\bno channels on the TV\b",
                        r"\bthe television didn't work\b",
                    ],
                    "tr": [
                        r"\bTV çalışmıyordu\b",
                        r"\btelevizyon çalışmıyordu\b",
                        r"\bkanal yoktu\b",
                        r"\bTV açılmıyordu\b",
                    ],
                    "ar": [
                        r"\bالتلفزيون ما اشتغل\b",
                        r"\bالتلفزيون خربان\b",
                        r"\bما في قنوات\b",
                        r"\bالشاشة ما عم تشتغل\b",
                    ],
                    "zh": [
                        r"电视看不了",
                        r"电视不工作",
                        r"电视没有频道",
                        r"电视坏了",
                    ],
                },
            ),
            
            "fridge_broken": AspectRule(
                aspect_code="fridge_broken",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bхолодильник не работал\b",
                        r"\bхолодильник еле холодил\b",
                        r"\bхолодильник плохо холодил\b",
                        r"\bхолодильник сломан\b",
                    ],
                    "en": [
                        r"\bfridge didn't work\b",
                        r"\bfridge was not working\b",
                        r"\bfridge barely cooled\b",
                        r"\bthe refrigerator didn't get cold\b",
                    ],
                    "tr": [
                        r"\bbuzdolabı çalışmıyordu\b",
                        r"\bbuzdolabı soğutmuyordu\b",
                        r"\bbuzdolabı bozuktu\b",
                    ],
                    "ar": [
                        r"\bالبراد ما بيبرد\b",
                        r"\bالبراد ما بيشتغل\b",
                        r"\bالبراد خربان\b",
                    ],
                    "zh": [
                        r"冰箱不制冷",
                        r"冰箱不工作",
                        r"冰箱坏了",
                        r"冰箱基本不凉",
                    ],
                },
            ),
        
            "kettle_broken": AspectRule(
                aspect_code="kettle_broken",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bчайник не работал\b",
                        r"\bчайник сломан\b",
                        r"\bчайник не грел\b",
                        r"\bсломанный чайник\b",
                    ],
                    "en": [
                        r"\bkettle was broken\b",
                        r"\bthe kettle didn't work\b",
                        r"\bwater boiler didn't work\b",
                        r"\bthe kettle was not heating\b",
                    ],
                    "tr": [
                        r"\bsu ısıtıcısı bozuktu\b",
                        r"\bketıl çalışmıyordu\b",
                        r"\bsu ısıtıcısı ısıtmıyordu\b",
                    ],
                    "ar": [
                        r"\bالكاتل خربان\b",
                        r"\bالسخان ما اشتغل\b",
                        r"\bالغلاية ما عم تسخن\b",
                    ],
                    "zh": [
                        r"水壶坏了",
                        r"烧水壶不能用",
                        r"烧水壶不加热",
                    ],
                },
            ),
        
            "socket_danger": AspectRule(
                aspect_code="socket_danger",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bрозетка искрит\b",
                        r"\bрозетка болтается\b",
                        r"\bрозетка вываливается\b",
                        r"\bнебезопасная розетка\b",
                        r"\bопасная розетка\b",
                    ],
                    "en": [
                        r"\bsocket was loose\b",
                        r"\bpower socket was loose\b",
                        r"\bsocket was sparking\b",
                        r"\belectrical outlet was sparking\b",
                        r"\bdangerous socket\b",
                        r"\bunsafe outlet\b",
                    ],
                    "tr": [
                        r"\bpriz gevşekti\b",
                        r"\bpriz kıvılcım atıyordu\b",
                        r"\bpriz yerinden oynuyordu\b",
                        r"\bpriz güvenli değildi\b",
                    ],
                    "ar": [
                        r"\bفيش الكهرباء عم يشرّر\b",
                        r"\bفيش مرتخي\b",
                        r"\bفيش مو ثابت\b",
                        r"\bالفيش مو آمن\b",
                    ],
                    "zh": [
                        r"插座松",
                        r"插座打火",
                        r"插座不安全",
                        r"插座松动有火花",
                    ],
                },
            ),
        
            "door_not_closing": AspectRule(
                aspect_code="door_not_closing",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bдверь плохо закрывается\b",
                        r"\bдверь не закрывалась до конца\b",
                        r"\bдверь не закрывалась нормально\b",
                        r"\bдверь не фиксируется\b",
                    ],
                    "en": [
                        r"\bdoor didn't close properly\b",
                        r"\bthe door wouldn't close\b",
                        r"\bthe door wouldn't lock properly\b",
                        r"\bthe door didn't shut all the way\b",
                    ],
                    "tr": [
                        r"\bkapı tam kapanmıyordu\b",
                        r"\bkapı kilitlenmiyordu\b",
                        r"\bkapı düzgün kapanmıyordu\b",
                    ],
                    "ar": [
                        r"\bالباب ما بيسكّر منيح\b",
                        r"\bالباب ما يسكّر صح\b",
                        r"\bالباب ما كان ينقفل مزبوط\b",
                    ],
                    "zh": [
                        r"门关不严",
                        r"门关不上",
                        r"门关不好",
                        r"门扣不上",
                    ],
                },
            ),
        
            "lock_broken": AspectRule(
                aspect_code="lock_broken",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bзамок заедал\b",
                        r"\bзамок не закрывался\b",
                        r"\bзамок не работал\b",
                        r"\bзамок сломан\b",
                        r"\bне чувствовали себя в безопасности\b",
                    ],
                    "en": [
                        r"\block was broken\b",
                        r"\block was sticking\b",
                        r"\bdoor wouldn't lock\b",
                        r"\bdidn't feel safe\b",
                        r"\bdidn't feel safe leaving our belongings\b",
                    ],
                    "tr": [
                        r"\bkilit bozuktu\b",
                        r"\bkilit takılıyordu\b",
                        r"\bkapı kilitlenmiyordu\b",
                        r"\bkendimizi güvende hissetmedik\b",
                    ],
                    "ar": [
                        r"\bالقفل خربان\b",
                        r"\bالقفل بيعلق\b",
                        r"\bالقفل ما عم يسكّر\b",
                        r"\bما حسّينا بأمان\b",
                    ],
                    "zh": [
                        r"锁坏了",
                        r"锁老卡",
                        r"门锁不上",
                        r"感觉不安全",
                        r"不敢把行李放里面",
                    ],
                },
            ),
        
            "furniture_broken": AspectRule(
                aspect_code="furniture_broken",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bсломанный шкаф\b",
                        r"\bдверца шкафа отваливается\b",
                        r"\bшатается стол\b",
                        r"\bмебель сломана\b",
                        r"\bмебель убитая\b",
                    ],
                    "en": [
                        r"\bwardrobe was broken\b",
                        r"\bcloset door falling off\b",
                        r"\bfurniture damaged\b",
                        r"\btable was wobbly\b",
                        r"\bbroken furniture\b",
                    ],
                    "tr": [
                        r"\bdolap kırılmıştı\b",
                        r"\bmasa sallanıyordu\b",
                        r"\bmobilya kırılmıştı\b",
                        r"\bmobilyalar yıpranmıştı\b",
                    ],
                    "ar": [
                        r"\bالخزانة مكسورة\b",
                        r"\bباب الخزانة عم يطيح\b",
                        r"\bالطاولة مكسورة\b",
                        r"\bالأثاث مكسّر\b",
                    ],
                    "zh": [
                        r"衣柜坏了",
                        r"衣柜门要掉了",
                        r"桌子摇摇晃晃",
                        r"家具很破",
                        r"家具坏了",
                    ],
                },
            ),
        
            "room_worn_out": AspectRule(
                aspect_code="room_worn_out",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bпошарпанные стены\b",
                        r"\bоблезлые стены\b",
                        r"\bтребует ремонта\b",
                        r"\bвсё уставшее\b",
                        r"\bномер уставший\b",
                        r"\bстарый ремонт\b",
                    ],
                    "en": [
                        r"\bwalls were damaged\b",
                        r"\broom looks worn out\b",
                        r"\broom was very tired looking\b",
                        r"\bdated decor\b",
                        r"\bthe place felt old\b",
                        r"\bfelt cheap and worn\b",
                    ],
                    "tr": [
                        r"\bduvarlar çok yıpranmıştı\b",
                        r"\boda yorgun görünüyordu\b",
                        r"\bdekorasyon eskiydi\b",
                        r"\boda bakımsızdı\b",
                    ],
                    "ar": [
                        r"\bالشكل قديم\b",
                        r"\bالمكان شكله تعبان\b",
                        r"\bمبين مستهلَك\b",
                        r"\bالديكور قديم\b",
                    ],
                    "zh": [
                        r"墙很旧",
                        r"房间看起来很旧",
                        r"房间很破旧",
                        r"装修很旧",
                        r"感觉很老旧",
                    ],
                },
            ),
            
            "wifi_fast": AspectRule(
                aspect_code="wifi_fast",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bбыстрый wi[- ]?fi\b",
                        r"\bотличный wi[- ]?fi\b",
                        r"\bwi[- ]?fi работал хорошо\b",
                        r"\bинтернет быстрый\b",
                        r"\bхороший интернет\b",
                    ],
                    "en": [
                        r"\bwifi was fast\b",
                        r"\bwifi was very fast\b",
                        r"\bgood wifi\b",
                        r"\bfast internet\b",
                        r"\binternet was fast\b",
                    ],
                    "tr": [
                        r"\bwifi hızlıydı\b",
                        r"\binternet çok iyiydi\b",
                        r"\bhızlı internet\b",
                        r"\bbağlantı hızlıydı\b",
                    ],
                    "ar": [
                        r"\bالواي فاي سريع\b",
                        r"\bالانترنت سريع\b",
                        r"\bالانترنت ممتاز\b",
                        r"\bالواي فاي كتير منيح\b",
                    ],
                    "zh": [
                        r"wifi很快",
                        r"网速很快",
                        r"网络很快",
                        r"上网速度很快",
                    ],
                },
            ),
        
            "internet_stable": AspectRule(
                aspect_code="internet_stable",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bинтернет стабильный\b",
                        r"\bсоединение стабильное\b",
                        r"\bwi[- ]?fi не отваливался\b",
                        r"\bподключение без сбоев\b",
                    ],
                    "en": [
                        r"\breliable wifi\b",
                        r"\bstable connection\b",
                        r"\bwifi was stable\b",
                        r"\bwifi was reliable\b",
                        r"\binternet was reliable\b",
                    ],
                    "tr": [
                        r"\bbağlantı stabildi\b",
                        r"\bwifi stabildi\b",
                        r"\binternet bağlantısı kararlıydı\b",
                    ],
                    "ar": [
                        r"\bالاتصال ثابت\b",
                        r"\bالواي فاي ثابت\b",
                        r"\bالنت ما يقطع\b",
                        r"\bما عم يقطع النت\b",
                    ],
                    "zh": [
                        r"网络很稳定",
                        r"上网很稳定",
                        r"连接很稳定",
                        r"wifi很稳定不掉线",
                    ],
                },
            ),
        
            "good_for_work": AspectRule(
                aspect_code="good_for_work",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bможно работать удал[её]нно\b",
                        r"\bподходит для удалённой работы\b",
                        r"\bхватало для работы\b",
                        r"\bинтернет позволял работать\b",
                    ],
                    "en": [
                        r"\binternet was great for work\b",
                        r"\bgood enough to work remotely\b",
                        r"\bcould work remotely\b",
                        r"\bgood wifi for work\b",
                    ],
                    "tr": [
                        r"\bçalışmak için yeterince iyiydi\b",
                        r"\buzaktan çalışmaya uygundu\b",
                        r"\buzaktan çalışmak mümkündü\b",
                    ],
                    "ar": [
                        r"\bفيك تشتغل أونلاين عادي\b",
                        r"\bالنت بكفي للشغل\b",
                        r"\bمناسب للشغل أونلاين\b",
                    ],
                    "zh": [
                        r"可以正常远程工作",
                        r"网速可以办公",
                        r"网络适合远程办公",
                    ],
                },
            ),
        
            "wifi_down": AspectRule(
                aspect_code="wifi_down",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bwi[- ]?fi не работал\b",
                        r"\bwi[- ]?fi не ловил\b",
                        r"\bwi[- ]?fi вообще не было\b",
                        r"\bинтернет не работал\b",
                        r"\bсовсем без интернета\b",
                    ],
                    "en": [
                        r"\bwifi didn't work\b",
                        r"\bwifi was not working\b",
                        r"\bno wifi\b",
                        r"\binternet was down\b",
                        r"\bno internet at all\b",
                    ],
                    "tr": [
                        r"\bwifi çalışmıyordu\b",
                        r"\bwifi yoktu\b",
                        r"\binternet hiç yoktu\b",
                        r"\binternet çalışmıyordu\b",
                    ],
                    "ar": [
                        r"\bالواي فاي ما اشتغل\b",
                        r"\bما في واي فاي\b",
                        r"\bما كان في نت\b",
                        r"\bالنت مو شغال\b",
                    ],
                    "zh": [
                        r"wifi不好用",
                        r"wifi不能用",
                        r"没网",
                        r"完全没有网络",
                    ],
                },
            ),
        
            "wifi_slow": AspectRule(
                aspect_code="wifi_slow",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bочень медленный интернет\b",
                        r"\bинтернет ужасно медленный\b",
                        r"\bмедленный wi[- ]?fi\b",
                        r"\bскорость интернета никакая\b",
                    ],
                    "en": [
                        r"\bvery slow wifi\b",
                        r"\bwifi was slow\b",
                        r"\bslow internet\b",
                        r"\binternet was super slow\b",
                        r"\bunusable wifi\b",
                    ],
                    "tr": [
                        r"\binternet çok yavaştı\b",
                        r"\bwifi yavaştı\b",
                        r"\bbağlantı aşırı yavaştı\b",
                        r"\bkullanılamayacak kadar yavaştı\b",
                    ],
                    "ar": [
                        r"\bالانترنت بطيء كتير\b",
                        r"\bالواي فاي بطيء\b",
                        r"\bالنت بطيء لدرجة ما بينفتح شي\b",
                        r"\bمستحيل تستعمله\b",
                    ],
                    "zh": [
                        r"网速很慢",
                        r"wifi很慢",
                        r"基本没网速",
                        r"网慢得根本用不了",
                    ],
                },
            ),
            
            "wifi_unstable": AspectRule(
                aspect_code="wifi_unstable",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bwi[- ]?fi постоянно отваливался\b",
                        r"\bинтернет обрывался\b",
                        r"\bсоединение постоянно пропадало\b",
                        r"\bинтернет всё время падал\b",
                    ],
                    "en": [
                        r"\bkept disconnecting\b",
                        r"\bkept dropping\b",
                        r"\bwifi kept cutting out\b",
                        r"\bunstable wifi\b",
                        r"\bconnection kept dropping\b",
                    ],
                    "tr": [
                        r"\bbağlantı sürekli koptu\b",
                        r"\bsürekli düşüyordu\b",
                        r"\bwifi kopup duruyordu\b",
                        r"\binternet gidip geliyordu\b",
                    ],
                    "ar": [
                        r"\bالواي فاي عم يقطع\b",
                        r"\bالنت كل شوي بينقطع\b",
                        r"\bالاتصال مو ثابت وعم يطيح\b",
                        r"\bالواي فاي مو ثابت\b",
                    ],
                    "zh": [
                        r"老是掉线",
                        r"一直断线",
                        r"wifi老是断",
                        r"网络很不稳定",
                    ],
                },
            ),
        
            "wifi_hard_to_connect": AspectRule(
                aspect_code="wifi_hard_to_connect",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bсложно подключиться к wi[- ]?fi\b",
                        r"\bне могли подключиться к wi[- ]?fi\b",
                        r"\bпароль от wi[- ]?fi не работал\b",
                        r"\bне подключается wi[- ]?fi\b",
                    ],
                    "en": [
                        r"\bhard to connect to wifi\b",
                        r"\bcouldn't connect to the wifi\b",
                        r"\bwifi password didn't work\b",
                        r"\bthe wifi login didn't work\b",
                    ],
                    "tr": [
                        r"\bbağlanmak çok zordu\b",
                        r"\bwifiye bağlanamadık\b",
                        r"\bşifre çalışmadı\b",
                        r"\bwifi şifresi kabul etmedi\b",
                    ],
                    "ar": [
                        r"\bما عرفنا ندخل عالواي فاي\b",
                        r"\bالباسورد ما اشتغل\b",
                        r"\bصعب تتصل بالواي فاي\b",
                        r"\bمنقدرش نوصل على الواي فاي\b",
                    ],
                    "zh": [
                        r"连不上wifi",
                        r"wifi连不上去",
                        r"密码连不上",
                        r"wifi登陆不上",
                    ],
                },
            ),
        
            "internet_not_suitable_for_work": AspectRule(
                aspect_code="internet_not_suitable_for_work",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bневозможно было работать\b",
                        r"\bне могли работать удал[её]нно\b",
                        r"\bинтернет не подходит для работы\b",
                        r"\bинтернет слишком плохой для работы\b",
                    ],
                    "en": [
                        r"\bcouldn't work remotely because of the internet\b",
                        r"\binternet was not good enough to work\b",
                        r"\bnot possible to work remotely\b",
                        r"\bwifi not usable for work\b",
                    ],
                    "tr": [
                        r"\buzaktan çalışmak imkansızdı\b",
                        r"\bçalışmak mümkün değildi internet yüzünden\b",
                        r"\binternet işe uygun değildi\b",
                    ],
                    "ar": [
                        r"\bما قدرنا نشتغل عن بُعد\b",
                        r"\bالنت مو مناسب للشغل\b",
                        r"\bمستحيل تشتغل أونلاين بهالنت\b",
                    ],
                    "zh": [
                        r"没法远程办公",
                        r"网太差没法工作",
                        r"网络不适合工作",
                        r"因为网络没法上班远程",
                    ],
                },
            ),
        
            "ac_noisy": AspectRule(
                aspect_code="ac_noisy",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bкондиционер очень шумный\b",
                        r"\bгромко гудел кондиционер\b",
                        r"\bкондиционер шумел\b",
                        r"\bшум от кондиционера\b",
                    ],
                    "en": [
                        r"\bAC was very loud\b",
                        r"\bair conditioner was noisy\b",
                        r"\bthe AC was noisy at night\b",
                        r"\bnoise from the air conditioner\b",
                    ],
                    "tr": [
                        r"\bklima çok gürültülüydü\b",
                        r"\bklima ses yapıyordu\b",
                        r"\bklima gece çok ses yaptı\b",
                    ],
                    "ar": [
                        r"\bالمكيف صوته عالي\b",
                        r"\bالمكيف مزعج\b",
                        r"\bصوت المكيف كان عالي بالليل\b",
                    ],
                    "zh": [
                        r"空调声音很大",
                        r"空调很吵",
                        r"空调晚上很吵",
                        r"空调噪音很大",
                    ],
                },
            ),
        
            "fridge_noisy": AspectRule(
                aspect_code="fridge_noisy",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bхолодильник шумел\b",
                        r"\bгромко жужжал холодильник\b",
                        r"\bхолодильник трещит\b",
                        r"\bшум от холодильника мешал спать\b",
                    ],
                    "en": [
                        r"\bfridge was noisy\b",
                        r"\bfridge was humming loudly\b",
                        r"\bfridge was rattling\b",
                        r"\bnoise from the fridge kept us up\b",
                    ],
                    "tr": [
                        r"\bbuzdolabı çok ses yapıyordu\b",
                        r"\bbuzdolabı sürekli uğulduyordu\b",
                        r"\bbuzdolabının sesi rahatsız ediyordu\b",
                    ],
                    "ar": [
                        r"\bالبراد عم يطن بصوت عالي\b",
                        r"\bصوت البراد عالي\b",
                        r"\bالبراد مضايقنا بالصوت\b",
                    ],
                    "zh": [
                        r"冰箱很吵",
                        r"冰箱一直嗡嗡响",
                        r"冰箱一直在响",
                        r"冰箱的声音影响睡觉",
                    ],
                },
            ),
            
            "pipes_noise": AspectRule(
                aspect_code="pipes_noise",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bгул труб\b",
                        r"\bшум в трубах\b",
                        r"\bгул стояка\b",
                        r"\bшум от стояка\b",
                        r"\bв трубах шумело\b",
                    ],
                    "en": [
                        r"\bpipes were making noise\b",
                        r"\bloud pipes\b",
                        r"\bnoise from the pipes\b",
                        r"\brattling pipes\b",
                        r"\bplumbing noise\b",
                    ],
                    "tr": [
                        r"\bborulardan ses geliyordu\b",
                        r"\btesisattan ses geliyordu\b",
                        r"\bborular çok ses yapıyordu\b",
                        r"\bborulardan uğultu geliyordu\b",
                    ],
                    "ar": [
                        r"\bصوت مواسير\b",
                        r"\bصوت المواسير عالي\b",
                        r"\bصوت المواسير كان مزعج\b",
                        r"\bفي صوت طالع من المواسير\b",
                    ],
                    "zh": [
                        r"水管有声音",
                        r"水管一直响",
                        r"管道一直响",
                        r"水管噪音很大",
                    ],
                },
            ),
        
            "ventilation_noisy": AspectRule(
                aspect_code="ventilation_noisy",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bшумит вентиляция\b",
                        r"\bгудит вентилятор\b",
                        r"\bвентиляция очень шумная\b",
                        r"\bвентилятор слишком громкий\b",
                    ],
                    "en": [
                        r"\bventilation was loud\b",
                        r"\bthe fan was loud\b",
                        r"\bnoisy ventilation\b",
                        r"\bextractor fan was noisy\b",
                    ],
                    "tr": [
                        r"\bhavalandırma çok sesliydi\b",
                        r"\bfan çok ses çıkartıyordu\b",
                        r"\bhavalandırma uğulduyordu\b",
                    ],
                    "ar": [
                        r"\bالشفاط صوته عالي\b",
                        r"\bالتهوية صوتها عالي\b",
                        r"\bالمروحة صوتها مزعج\b",
                    ],
                    "zh": [
                        r"排风很吵",
                        r"风扇很吵",
                        r"通风设备噪音很大",
                        r"排风扇声音很大",
                    ],
                },
            ),
        
            "night_mechanical_hum": AspectRule(
                aspect_code="night_mechanical_hum",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bночью что-то гудело\b",
                        r"\bкакой-то агрегат жужжал всю ночь\b",
                        r"\bжужжание всю ночь\b",
                        r"\bпостоянный гул ночью\b",
                    ],
                    "en": [
                        r"\bsomething was buzzing all night\b",
                        r"\bconstant humming at night\b",
                        r"\bmechanical noise all night\b",
                        r"\bthere was a constant hum during the night\b",
                    ],
                    "tr": [
                        r"\bgece boyunca bir şey uğulduyordu\b",
                        r"\bsürekli bir uğultu vardı\b",
                        r"\bgece sürekli bir vınlama sesi vardı\b",
                    ],
                    "ar": [
                        r"\bصوت أزيز طول الليل\b",
                        r"\bفي أزاز طول الليل\b",
                        r"\bكان في طنين طول الليل\b",
                        r"\bصوت مزعج طول الليل\b",
                    ],
                    "zh": [
                        r"半夜一直有嗡嗡声",
                        r"一晚上都在响",
                        r"整晚都有机器的嗡嗡声",
                    ],
                },
            ),
        
            "tech_noise_sleep_issue": AspectRule(
                aspect_code="tech_noise_sleep_issue",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bне могли уснуть из-за шума техники\b",
                        r"\bшум техники мешал спать\b",
                        r"\bшум от оборудования мешал спать\b",
                        r"\bиз-за шума не спали\b",
                    ],
                    "en": [
                        r"\bhard to sleep because of the noise from the unit\b",
                        r"\bmechanical noise kept us awake\b",
                        r"\bnoise from the AC kept us up\b",
                        r"\bcouldn't sleep because of the humming\b",
                    ],
                    "tr": [
                        r"\bbu sesten uyumak zordu\b",
                        r"\bcihaz sesi yüzünden uyuyamadık\b",
                        r"\bklima sesi uykumuzu böldü\b",
                    ],
                    "ar": [
                        r"\bما قدرنا ننام من صوت الأجهزة\b",
                        r"\bالصوت المضغوط هاد ما خلانا ننام\b",
                        r"\bالصوت ما خلاني نام\b",
                    ],
                    "zh": [
                        r"吵得睡不着",
                        r"机器的声音影响睡觉",
                        r"噪音让我们睡不好",
                    ],
                },
            ),
        
            "ac_quiet": AspectRule(
                aspect_code="ac_quiet",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bкондиционер тихий\b",
                        r"\bтихий кондиционер\b",
                        r"\bкондиционер вообще не шумел\b",
                    ],
                    "en": [
                        r"\bAC was quiet\b",
                        r"\bvery quiet AC\b",
                        r"\bthe air conditioner was quiet at night\b",
                    ],
                    "tr": [
                        r"\bklima sessizdi\b",
                        r"\bklima gece sessizdi\b",
                        r"\bklima rahatsız etmiyordu\b",
                    ],
                    "ar": [
                        r"\bالمكيف هادي\b",
                        r"\bالمكيف ما إلو صوت تقريبًا\b",
                        r"\bالمكيف مو مزعج\b",
                    ],
                    "zh": [
                        r"空调很安静",
                        r"空调几乎没声音",
                        r"空调晚上不吵",
                    ],
                },
            ),
            
            "fridge_quiet": AspectRule(
                aspect_code="fridge_quiet",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bтихий холодильник\b",
                        r"\bхолодильник не шумел\b",
                        r"\bхолодильник вообще не слышно\b",
                    ],
                    "en": [
                        r"\bfridge was quiet\b",
                        r"\bthe fridge was quiet\b",
                        r"\bno noise from the fridge\b",
                    ],
                    "tr": [
                        r"\bbuzdolabı sessizdi\b",
                        r"\bbuzdolabı hiç ses yapmıyordu\b",
                        r"\bbuzdolabından ses gelmiyordu\b",
                    ],
                    "ar": [
                        r"\bالبراد هادي\b",
                        r"\bالبراد ما له صوت تقريبًا\b",
                        r"\bما في صوت من البراد\b",
                    ],
                    "zh": [
                        r"冰箱很安静",
                        r"冰箱基本没声音",
                        r"冰箱一点都不吵",
                    ],
                },
            ),
        
            "no_tech_noise_night": AspectRule(
                aspect_code="no_tech_noise_night",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bничего не шумело ночью\b",
                        r"\bтихо ночью\b",
                        r"\bоборудование не шумело ночью\b",
                        r"\bникаких звуков техники ночью\b",
                    ],
                    "en": [
                        r"\bno mechanical noise at night\b",
                        r"\bquiet at night from the appliances\b",
                        r"\bno noise from AC or fridge at night\b",
                        r"\bno humming at night\b",
                    ],
                    "tr": [
                        r"\bgece hiçbir cihaz ses çıkarmıyordu\b",
                        r"\bgece çok sessizdi\b",
                        r"\bklima ve buzdolabı gece sessizdi\b",
                    ],
                    "ar": [
                        r"\bما كان في أي صوت بالليل\b",
                        r"\bبالليل كان هادي من ناحية الأجهزة\b",
                        r"\bما في أزيز بالليل\b",
                    ],
                    "zh": [
                        r"晚上没有机器的噪音",
                        r"晚上很安静没有嗡嗡声",
                        r"晚上空调和冰箱都不吵",
                    ],
                },
            ),
        
            "elevator_working": AspectRule(
                aspect_code="elevator_working",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bлифт работал\b",
                        r"\bлифт исправен\b",
                        r"\bлифт всегда работал\b",
                        r"\bлифт нормальный\b",
                    ],
                    "en": [
                        r"\bthe elevator was working\b",
                        r"\belevator worked fine\b",
                        r"\bworking elevator\b",
                        r"\belevator in service\b",
                    ],
                    "tr": [
                        r"\basansör çalışıyordu\b",
                        r"\basansör sorunsuzdu\b",
                        r"\basansör hizmetteydi\b",
                    ],
                    "ar": [
                        r"\bالمصعد شغال\b",
                        r"\bالمصعد تمام\b",
                        r"\bالمصعد كان يشتغل عادي\b",
                    ],
                    "zh": [
                        r"电梯正常",
                        r"电梯可以用",
                        r"电梯工作正常",
                    ],
                },
            ),
        
            "luggage_easy": AspectRule(
                aspect_code="luggage_easy",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bудобно с чемоданами\b",
                        r"\bлегко подняться с багажом\b",
                        r"\bс багажом без проблем\b",
                        r"\bс чемоданами было удобно\b",
                    ],
                    "en": [
                        r"\beasy with luggage\b",
                        r"\beasy to bring luggage up\b",
                        r"\bno problem with suitcases\b",
                        r"\bgood access with bags\b",
                    ],
                    "tr": [
                        r"\bbavullarla çıkmak kolaydı\b",
                        r"\bvalizle çıkmak rahattı\b",
                        r"\bvalizle sorun olmadı\b",
                    ],
                    "ar": [
                        r"\bسهل تطلع مع الشنط\b",
                        r"\bما في مشكلة مع الشنط\b",
                        r"\bالوصول مع الشنط كان سهل\b",
                    ],
                    "zh": [
                        r"带行李上去很方便",
                        r"拿行李很轻松",
                        r"带行李没什么问题",
                    ],
                },
            ),
        
            "elevator_broken": AspectRule(
                aspect_code="elevator_broken",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bлифт не работал\b",
                        r"\bлифт сломан\b",
                        r"\bлифт отключали\b",
                        r"\bлифт выключен\b",
                        r"\bне было лифта\b",
                    ],
                    "en": [
                        r"\bthe elevator was not working\b",
                        r"\belevator was broken\b",
                        r"\belevator was out of service\b",
                        r"\bno working elevator\b",
                        r"\bno elevator\b",
                    ],
                    "tr": [
                        r"\basansör çalışmıyordu\b",
                        r"\basansör bozuktu\b",
                        r"\basansör kapalıydı\b",
                        r"\basansör yoktu\b",
                    ],
                    "ar": [
                        r"\bالمصعد معطل\b",
                        r"\bالمصعد خربان\b",
                        r"\bما في مصعد شغال\b",
                        r"\bما في أسانسير شغال\b",
                    ],
                    "zh": [
                        r"电梯坏了",
                        r"电梯不能用",
                        r"电梯停用",
                        r"没有电梯",
                    ],
                },
            ),
            
            "elevator_stuck": AspectRule(
                aspect_code="elevator_stuck",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bзастряли в лифте\b",
                        r"\bзависли в лифте\b",
                        r"\bнас заклинило в лифте\b",
                        r"\bзастревал лифт\b",
                    ],
                    "en": [
                        r"\bwe got stuck in the elevator\b",
                        r"\bwe were stuck in the elevator\b",
                        r"\bthe elevator got stuck\b",
                        r"\belevator kept getting stuck\b",
                    ],
                    "tr": [
                        r"\basansörde kaldık\b",
                        r"\basansörde sıkıştık\b",
                        r"\basansör takıldı\b",
                        r"\basansör takılı kaldı\b",
                    ],
                    "ar": [
                        r"\bعلقنا بالمصعد\b",
                        r"\bحبسنا بالمصعد\b",
                        r"\bالأسانسير وقف فينا\b",
                    ],
                    "zh": [
                        r"我们被困在电梯里",
                        r"电梯卡住了人",
                        r"电梯把我们困住了",
                        r"电梯老是卡住",
                    ],
                },
            ),
        
            "no_elevator_heavy_bags": AspectRule(
                aspect_code="no_elevator_heavy_bags",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bбез лифта очень тяжело с багажом\b",
                        r"\bтащить чемоданы по лестнице\b",
                        r"\bпришлось тащить чемоданы наверх\b",
                        r"\bс багажом тяжело без лифта\b",
                    ],
                    "en": [
                        r"\bhad to carry luggage up the stairs\b",
                        r"\bcarrying suitcases upstairs was hard\b",
                        r"\bno elevator and we had luggage\b",
                        r"\bnot easy with heavy bags\b",
                    ],
                    "tr": [
                        r"\bmerdivenle bavul taşımak çok zordu\b",
                        r"\bvalizle çıkmak çok zordu\b",
                        r"\basansör yoktu bavullarla çok zor oldu\b",
                    ],
                    "ar": [
                        r"\bاضطرينا نطلع الدرج مع الشنط\b",
                        r"\bصعب كتير مع الشنط\b",
                        r"\bما في أسانسير ومعنا شنط تقيلة\b",
                    ],
                    "zh": [
                        r"只能扛行李上楼",
                        r"拿行李走楼梯很辛苦",
                        r"没有电梯拿行李特别累",
                    ],
                },
            ),
        
            "felt_safe": AspectRule(
                aspect_code="felt_safe",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bчувствовал[аи]?сь? в безопасности\b",
                        r"\bчувствовали себя в безопасности\b",
                        r"\bбезопасно хранить вещи\b",
                        r"\bнам было спокойно\b",
                    ],
                    "en": [
                        r"\bwe felt safe\b",
                        r"\bfelt safe\b",
                        r"\bfelt safe leaving our stuff\b",
                        r"\bfelt secure\b",
                        r"\bwe felt secure\b",
                    ],
                    "tr": [
                        r"\bkendimizi güvende hissettik\b",
                        r"\bgüvende hissettik\b",
                        r"\besyalarımız güvendeydi\b",
                    ],
                    "ar": [
                        r"\bحاسين بأمان\b",
                        r"\bحسّينا بأمان\b",
                        r"\bمأمنين على أغراضنا\b",
                        r"\bحسّينا إنه المكان آمن\b",
                    ],
                    "zh": [
                        r"感觉很安全",
                        r"我们觉得很安全",
                        r"我们放心把东西放房间",
                        r"觉得很安心很安全",
                    ],
                },
            ),
        
            "felt_unsafe": AspectRule(
                aspect_code="felt_unsafe",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bне чувствовали себя в безопасности\b",
                        r"\bне чувствовали себя безопасно\b",
                        r"\bбыло страшно\b",
                        r"\bлюбому можно войти\b",
                        r"\bне чувствовали себя спокойно\b",
                    ],
                    "en": [
                        r"\bdidn't feel safe\b",
                        r"\bwe didn't feel safe\b",
                        r"\bfelt unsafe leaving our belongings\b",
                        r"\bdidn't feel secure\b",
                        r"\bfelt unsafe\b",
                    ],
                    "tr": [
                        r"\bkendimizi güvende hissetmedik\b",
                        r"\bpek güvenli hissettirmedi\b",
                        r"\besyalarımızı bırakmak güvenli gelmedi\b",
                    ],
                    "ar": [
                        r"\bما حسّينا بأمان\b",
                        r"\bما كان آمن\b",
                        r"\bحاسين إنه أي حدا بفوت\b",
                        r"\bما حسّينا إنه أغراضنا بأمان\b",
                    ],
                    "zh": [
                        r"感觉不安全",
                        r"我们不觉得安全",
                        r"不敢把行李放里面",
                        r"感觉谁都能进来",
                    ],
                },
            ),
            
            "breakfast_tasty": AspectRule(
                aspect_code="breakfast_tasty",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bвкусный завтрак\b",
                        r"\bочень вкусный завтрак\b",
                        r"\bзавтрак был вкусн\w*\b",
                        r"\bзавтрак прям вкусный\b",
                        r"\bеда была вкусная\b",
                        r"\bочень вкусно\b",
                    ],
                    "en": [
                        r"\bbreakfast was delicious\b",
                        r"\bvery tasty breakfast\b",
                        r"\bthe breakfast was tasty\b",
                        r"\bfood was tasty\b",
                        r"\bgreat breakfast\b",
                    ],
                    "tr": [
                        r"\bkahvaltı çok lezzetliydi\b",
                        r"\bkahvaltı lezzetliydi\b",
                        r"\byemekler lezzetliydi\b",
                        r"\bkahvaltı harikaydı\b",
                    ],
                    "ar": [
                        r"\bالفطور طيب\b",
                        r"\bالفطور كتير طيب\b",
                        r"\bالأكل طعمو طيب\b",
                        r"\bالفطور كان رائع\b",
                    ],
                    "zh": [
                        r"早餐很好吃",
                        r"早餐很棒",
                        r"东西很好吃",
                        r"早餐味道很好",
                        r"早餐非常好吃",
                    ],
                },
            ),
        
            "food_fresh": AspectRule(
                aspect_code="food_fresh",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bвсё было свежим\b",
                        r"\bсвежие продукты\b",
                        r"\bвсё свежее\b",
                        r"\bеда свежая\b",
                    ],
                    "en": [
                        r"\beverything was fresh\b",
                        r"\bfresh ingredients\b",
                        r"\bfood was fresh\b",
                        r"\bvery fresh food\b",
                    ],
                    "tr": [
                        r"\bher şey tazeydi\b",
                        r"\btaze ürünler vardı\b",
                        r"\byemekler tazeydi\b",
                    ],
                    "ar": [
                        r"\bكلو طازة\b",
                        r"\bأكل طازة\b",
                        r"\bالأكل كان فريش\b",
                    ],
                    "zh": [
                        r"食材很新鲜",
                        r"都很新鲜",
                        r"早餐很新鲜",
                        r"东西很新鲜很好",
                    ],
                },
            ),
        
            "food_hot_served_hot": AspectRule(
                aspect_code="food_hot_served_hot",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bгорячие блюда\b.*\bгоряч\w*\b",
                        r"\bподавали горячее горячим\b",
                        r"\bгорячее было горячим\b",
                        r"\bвсё горячее и свежее\b",
                    ],
                    "en": [
                        r"\bhot dishes were actually hot\b",
                        r"\bthe hot food was hot\b",
                        r"\bserved hot\b",
                        r"\bfood was served hot\b",
                    ],
                    "tr": [
                        r"\bsıcak yemekler sıcaktı\b",
                        r"\byemekler sıcak servis edildi\b",
                        r"\bsıcaklar gerçekten sıcaktı\b",
                    ],
                    "ar": [
                        r"\bالأكل السخن كان سخن\b",
                        r"\bالأكلات السخنة كانت عنجد سخنة\b",
                        r"\bقدموه وهو سخن\b",
                    ],
                    "zh": [
                        r"热的菜是热的",
                        r"热菜真的很热",
                        r"端上来还是热的",
                        r"食物是热的",
                    ],
                },
            ),
        
            "coffee_good": AspectRule(
                aspect_code="coffee_good",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bвкусный кофе\b",
                        r"\bкофе хороший\b",
                        r"\bхороший кофе\b",
                        r"\bкофе очень вкусный\b",
                    ],
                    "en": [
                        r"\bgood coffee\b",
                        r"\bcoffee was good\b",
                        r"\bcoffee was nice\b",
                        r"\breally good coffee\b",
                    ],
                    "tr": [
                        r"\bkahve iyiydi\b",
                        r"\bkahve güzeldi\b",
                        r"\bkahve lezzetliydi\b",
                    ],
                    "ar": [
                        r"\bالقهوة طيبة\b",
                        r"\bالقهوة منيحة\b",
                        r"\bالقهوة كانت كتير منيحة\b",
                    ],
                    "zh": [
                        r"咖啡很好喝",
                        r"咖啡不错",
                        r"咖啡味道很好",
                    ],
                },
            ),
        
            "breakfast_bad_taste": AspectRule(
                aspect_code="breakfast_bad_taste",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bневкусный завтрак\b",
                        r"\bзавтрак был не очень\b",
                        r"\bзавтрак отвратительный\b",
                        r"\bневкусная еда\b",
                        r"\bеда была невкусная\b",
                        r"\bпересолено\b",
                        r"\bпережарен\w*\b",
                        r"\bсырая яичниц\w*\b",
                    ],
                    "en": [
                        r"\bbreakfast was not good\b",
                        r"\bbreakfast was bad\b",
                        r"\bterrible breakfast\b",
                        r"\bfood tasted bad\b",
                        r"\bovercooked\b",
                        r"\btoo salty\b",
                        r"\bundercooked eggs\b",
                        r"\bthe eggs were raw\b",
                    ],
                    "tr": [
                        r"\bkahvaltı iyi değildi\b",
                        r"\bkahvaltı berbattı\b",
                        r"\byemeklerin tadı kötüydü\b",
                        r"\başırı tuzluydu\b",
                        r"\bçok pişmişti\b",
                        r"\baz pişmişti\b",
                    ],
                    "ar": [
                        r"\bالفطور مو طيب\b",
                        r"\bالفطور سيء\b",
                        r"\bفطور بشع\b",
                        r"\bالأكل مش طيب\b",
                        r"\bمبين الأكل مو طازة\b",
                        r"\bالأكل مالح كتير\b",
                    ],
                    "zh": [
                        r"早餐不好吃",
                        r"早餐很差",
                        r"早餐太糟了",
                        r"东西不好吃",
                        r"太咸",
                        r"没熟",
                        r"煎蛋没熟",
                    ],
                },
            ),
            
            "food_not_fresh": AspectRule(
                aspect_code="food_not_fresh",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bнесвежие продукты\b",
                        r"\bне свежие продукты\b",
                        r"\bиспорченный вкус\b",
                        r"\bпросроченное ощущение\b",
                        r"\bеда не свежая\b",
                        r"\bчувствовалось что не свежее\b",
                    ],
                    "en": [
                        r"\bnot fresh\b",
                        r"\bfood was not fresh\b",
                        r"\bfelt old\b",
                        r"\bstale\b",
                        r"\bthe food tasted old\b",
                    ],
                    "tr": [
                        r"\btaze değildi\b",
                        r"\byemekler taze değildi\b",
                        r"\bbayat gibiydi\b",
                        r"\byemek bayattı\b",
                    ],
                    "ar": [
                        r"\bمش طازة\b",
                        r"\bالأكل بايت\b",
                        r"\bالأكل قديم\b",
                        r"\bمبين الأكل مو طازة\b",
                    ],
                    "zh": [
                        r"不新鲜",
                        r"有点不新鲜",
                        r"像放很久了",
                        r"食物不新鲜",
                    ],
                },
            ),
        
            "food_cold": AspectRule(
                aspect_code="food_cold",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bвсё холодное\b",
                        r"\bхолодные блюда\b",
                        r"\bвсё подали холодным\b",
                        r"\bхолодная яичниц\w*\b",
                        r"\bяичница холодная\b",
                    ],
                    "en": [
                        r"\beverything was cold\b",
                        r"\bthe food was cold\b",
                        r"\bcold eggs\b",
                        r"\bthe eggs were cold\b",
                        r"\bserved cold\b",
                    ],
                    "tr": [
                        r"\bher şey soğuktu\b",
                        r"\byemekler soğuktu\b",
                        r"\byumurta soğuktu\b",
                        r"\bsıcak yemekler bile soğuktu\b",
                    ],
                    "ar": [
                        r"\bالأكل كان بارد\b",
                        r"\bكلشي بارد\b",
                        r"\bالبيض بارد\b",
                        r"\bقدموا الأكل بارد\b",
                    ],
                    "zh": [
                        r"都是凉的",
                        r"全是冷的",
                        r"鸡蛋是凉的",
                        r"早餐都是冷的",
                        r"食物是冷的端上来",
                    ],
                },
            ),
        
            "coffee_bad": AspectRule(
                aspect_code="coffee_bad",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bкофе ужасный\b",
                        r"\bмерзкий кофе\b",
                        r"\bкофе невкусный\b",
                        r"\bкофе растворимый и невкусный\b",
                        r"\bтолько растворимый кофе\b",
                    ],
                    "en": [
                        r"\bcoffee was terrible\b",
                        r"\bbad coffee\b",
                        r"\bthe coffee was bad\b",
                        r"\binstant coffee only\b",
                        r"\bonly instant coffee\b",
                    ],
                    "tr": [
                        r"\bkahve çok kötüydü\b",
                        r"\bkahve berbattı\b",
                        r"\bsadece hazır kahve vardı\b",
                        r"\binstant kahve vardı sadece\b",
                    ],
                    "ar": [
                        r"\bالقهوة سيئة\b",
                        r"\bالقهوة زبالة\b",
                        r"\bالقهوة مش طيبة\b",
                        r"\bبس قهوة فورية\b",
                    ],
                    "zh": [
                        r"咖啡很难喝",
                        r"咖啡不好喝",
                        r"只有速溶咖啡",
                        r"只有即溶咖啡",
                    ],
                },
            ),
        
            "breakfast_variety_good": AspectRule(
                aspect_code="breakfast_variety_good",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bбольшой выбор\b",
                        r"\bогромный выбор\b",
                        r"\bмного всего\b",
                        r"\bразнообразный завтрак\b",
                        r"\bразнообразие блюд\b",
                        r"\bхороший выбор на завтрак\b",
                    ],
                    "en": [
                        r"\ba lot of choice\b",
                        r"\bgreat selection\b",
                        r"\bbig variety\b",
                        r"\bgood selection at breakfast\b",
                        r"\bwide choice at breakfast\b",
                    ],
                    "tr": [
                        r"\bçeşit çok fazlaydı\b",
                        r"\bseçenek çoktu\b",
                        r"\bkahvaltı çok çeşitliydi\b",
                        r"\bkahvaltıda bol seçenek vardı\b",
                    ],
                    "ar": [
                        r"\bفي كتير خيارات\b",
                        r"\bخيارات متنوعة\b",
                        r"\bفطور منوّع\b",
                        r"\bالخيارات عالفطور حلوة\b",
                    ],
                    "zh": [
                        r"选择很多",
                        r"种类很多",
                        r"早餐很丰富",
                        r"早餐选择很多",
                    ],
                },
            ),
        
            "buffet_rich": AspectRule(
                aspect_code="buffet_rich",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bшведский стол отличный\b",
                        r"\bочень хороший шведский стол\b",
                        r"\bклассный буфет\b",
                        r"\bбуфет отличный\b",
                    ],
                    "en": [
                        r"\bbuffet was great\b",
                        r"\bgood buffet\b",
                        r"\bthe breakfast buffet was great\b",
                        r"\breally nice buffet\b",
                    ],
                    "tr": [
                        r"\baçık büfe çok iyiydi\b",
                        r"\bbüfe zengindi\b",
                        r"\bkahvaltı büfesi çok iyiydi\b",
                    ],
                    "ar": [
                        r"\bالبوفيه ممتاز\b",
                        r"\bبوفيه غني\b",
                        r"\bالبوفيه تبع الفطور عنجد منيح\b",
                    ],
                    "zh": [
                        r"自助很不错",
                        r"自助很丰盛",
                        r"早餐自助很棒",
                        r"自助早餐很丰富",
                    ],
                },
            ),
            
            "fresh_fruit_available": AspectRule(
                aspect_code="fresh_fruit_available",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bфрукты\b",
                        r"\bсвежие фрукты\b",
                        r"\bбыли фрукты\b",
                        r"\bмного фруктов\b",
                    ],
                    "en": [
                        r"\bfresh fruit\b",
                        r"\bthere was fresh fruit\b",
                        r"\bthey had fruit\b",
                        r"\bfruit available\b",
                    ],
                    "tr": [
                        r"\bmeyve vardı\b",
                        r"\btaze meyve vardı\b",
                        r"\bkahvaltıda meyve vardı\b",
                    ],
                    "ar": [
                        r"\bفي فواكه\b",
                        r"\bفواكه طازة\b",
                        r"\bكان في فواكه عالفطور\b",
                    ],
                    "zh": [
                        r"有水果",
                        r"有新鲜水果",
                        r"早餐有水果",
                    ],
                },
            ),
        
            "pastries_available": AspectRule(
                aspect_code="pastries_available",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bвыпечка\b",
                        r"\bсвежая выпечка\b",
                        r"\bбулочки\b",
                        r"\bкруассаны\b",
                        r"\bкруассаны на завтрак\b",
                    ],
                    "en": [
                        r"\bpastries available\b",
                        r"\bthey had pastries\b",
                        r"\bcroissants\b",
                        r"\bfresh pastries\b",
                        r"\bpastries for breakfast\b",
                    ],
                    "tr": [
                        r"\bhamur işi vardı\b",
                        r"\bçörekler vardı\b",
                        r"\bkruvasan vardı\b",
                        r"\bfırından çıkmış gibiydi\b",
                    ],
                    "ar": [
                        r"\bفي معجنات\b",
                        r"\bفي كرواسون\b",
                        r"\bمعجنات طازة\b",
                        r"\bكان في مخبوزات على الفطور\b",
                    ],
                    "zh": [
                        r"有面包",
                        r"有糕点",
                        r"有牛角包",
                        r"早餐有烘焙点心",
                    ],
                },
            ),
        
            "breakfast_variety_poor": AspectRule(
                aspect_code="breakfast_variety_poor",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bвыбор маленький\b",
                        r"\bразнообразия нет\b",
                        r"\bочень скудный завтрак\b",
                        r"\bскудный выбор\b",
                        r"\bвыбор очень маленький\b",
                    ],
                    "en": [
                        r"\bvery limited choice\b",
                        r"\bpoor selection\b",
                        r"\bnot much choice\b",
                        r"\bnot much to choose from\b",
                        r"\bbreakfast selection was poor\b",
                    ],
                    "tr": [
                        r"\bseçenek azdı\b",
                        r"\bçeşit azdı\b",
                        r"\bkahvaltı çok zayıftı\b",
                        r"\bkahvaltı fakirdi\b",
                    ],
                    "ar": [
                        r"\bما في خيارات\b",
                        r"\bخيارات قليلة\b",
                        r"\bفطور فقير\b",
                        r"\bالاختيارات كانت ضعيفة\b",
                    ],
                    "zh": [
                        r"选择很少",
                        r"种类不多",
                        r"早餐选择很少",
                        r"没什么可以选的",
                    ],
                },
            ),
        
            "breakfast_repetitive": AspectRule(
                aspect_code="breakfast_repetitive",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bкаждый день одно и то же\b",
                        r"\bвсё одно и то же\b",
                        r"\bкаждый день одинаковый завтрак\b",
                    ],
                    "en": [
                        r"\bsame food every day\b",
                        r"\brepetitive breakfast\b",
                        r"\bthe breakfast was the same every day\b",
                        r"\bidentical breakfast every morning\b",
                    ],
                    "tr": [
                        r"\bher gün aynı şeyler\b",
                        r"\bkahvaltı çok tekdüze\b",
                        r"\bhep aynı kahvaltı vardı\b",
                    ],
                    "ar": [
                        r"\bكل يوم نفس الأكل\b",
                        r"\bكل يوم نفس الفطور\b",
                        r"\bمكرر كل يوم\b",
                    ],
                    "zh": [
                        r"每天都一样",
                        r"每天都是同样的东西",
                        r"早餐每天都一样",
                    ],
                },
            ),
        
            "hard_to_find_food": AspectRule(
                aspect_code="hard_to_find_food",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bничего нормального поесть\b",
                        r"\bесть особо нечего\b",
                        r"\bнечего есть на завтрак\b",
                        r"\bтяжело было что-то выбрать\b",
                    ],
                    "en": [
                        r"\bhard to find anything to eat\b",
                        r"\bnothing to eat\b",
                        r"\bdidn't really find anything to eat\b",
                        r"\bnothing suitable to eat\b",
                    ],
                    "tr": [
                        r"\byiyebileceğimiz bir şey bulmak zordu\b",
                        r"\bdoğru dürüst yiyecek bir şey yoktu\b",
                        r"\bpek yenilecek bir şey yoktu\b",
                    ],
                    "ar": [
                        r"\bما في شي ناكله\b",
                        r"\bصعب تلاقي شي تاكله الصبح\b",
                        r"\bما لقينا شي مناسب ناكله\b",
                    ],
                    "zh": [
                        r"没什么可以吃的",
                        r"基本没什么能吃的",
                        r"很难找到能吃的东西",
                    ],
                },
            ),
            
            "breakfast_staff_friendly": AspectRule(
                aspect_code="breakfast_staff_friendly",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bприветлив(ый|ые) персонал на завтраке\b",
                        r"\bперсонал завтрака очень дружелюбн\w*\b",
                        r"\bперсонал вежливый\b",
                        r"\bперсонал заботливый\b",
                        r"\bочень приятные сотрудники на завтраке\b",
                    ],
                    "en": [
                        r"\bstaff at breakfast were very friendly\b",
                        r"\bbreakfast staff were nice\b",
                        r"\bpolite staff\b",
                        r"\bvery kind breakfast staff\b",
                        r"\bthe staff were super friendly at breakfast\b",
                    ],
                    "tr": [
                        r"\bkahvaltı personeli çok nazikti\b",
                        r"\bpersonel çok güler yüzlüydü\b",
                        r"\bçok yardımcıydılar\b",
                        r"\bkahvaltıda çalışanlar çok kibardı\b",
                    ],
                    "ar": [
                        r"\bالموظفين تبع الفطور كتير لطيفين\b",
                        r"\bالموظفين محترمين\b",
                        r"\bالموظفين معاملة حلوة\b",
                        r"\bتعاملهم كتير لطيف\b",
                    ],
                    "zh": [
                        r"早餐的服务员很友好",
                        r"服务员很热情",
                        r"早餐工作人员很客气",
                        r"服务态度很好（早餐）",
                    ],
                },
            ),
        
            "breakfast_staff_attentive": AspectRule(
                aspect_code="breakfast_staff_attentive",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bперсонал заботливый\b",
                        r"\bвнимательный персонал\b",
                        r"\bследили за всем\b",
                        r"\bбыстро приносили\b",
                        r"\bбыстро пополняли блюда\b",
                    ],
                    "en": [
                        r"\battentive staff\b",
                        r"\bthe staff were very attentive\b",
                        r"\bthey refilled everything quickly\b",
                        r"\bthey were paying attention\b",
                        r"\bthey were on top of everything\b",
                    ],
                    "tr": [
                        r"\bçok yardımcıydılar\b",
                        r"\bhemen tazeliyorlardı\b",
                        r"\bpersonel çok ilgiliydi\b",
                        r"\bher şeyi takip ediyorlardı\b",
                    ],
                    "ar": [
                        r"\bبينتبهوا بسرعة\b",
                        r"\bكتير منتبهين\b",
                        r"\bبيعبيوا الأكل بسرعة\b",
                        r"\bالخدمة سريعة بالفطور\b",
                    ],
                    "zh": [
                        r"服务很周到",
                        r"很照顾我们",
                        r"很快就补菜",
                        r"一直在留意补东西",
                    ],
                },
            ),
        
            "buffet_refilled_quickly": AspectRule(
                aspect_code="buffet_refilled_quickly",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bбыстро пополняли блюда\b",
                        r"\bсразу добавляли\b",
                        r"\bподносили сразу\b",
                        r"\bвсё постоянно подносили\b",
                    ],
                    "en": [
                        r"\bthey refilled everything quickly\b",
                        r"\bbuffet was refilled quickly\b",
                        r"\bthey kept bringing more food\b",
                        r"\bthey topped up the food fast\b",
                    ],
                    "tr": [
                        r"\bsürekli tazeliyorlardı\b",
                        r"\byeniden getiriyorlardı\b",
                        r"\bboşalan ürünleri hemen doldurdular\b",
                    ],
                    "ar": [
                        r"\bكل شوي عم يجيبوا أكل جديد\b",
                        r"\bكانوا يعَبّوا الأكل بسرعة\b",
                        r"\bكل ما يفضى الشي يرجعوا يعبّوه\b",
                    ],
                    "zh": [
                        r"一直有补菜",
                        r"很快就补菜",
                        r"菜一空就补上",
                        r"补得很及时",
                    ],
                },
            ),
        
            "tables_cleared_fast": AspectRule(
                aspect_code="tables_cleared_fast",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bубирали со стола сразу\b",
                        r"\bстолы быстро протирали\b",
                        r"\bчистили стол сразу\b",
                        r"\bсразу убирали посуду\b",
                    ],
                    "en": [
                        r"\btable was cleaned immediately\b",
                        r"\bthey cleaned the tables quickly\b",
                        r"\bthey cleared tables fast\b",
                        r"\bused dishes were taken right away\b",
                    ],
                    "tr": [
                        r"\bmasaları hemen temizlediler\b",
                        r"\bmasalar hemen toplandı\b",
                        r"\bkullanılan tabakları hemen aldılar\b",
                    ],
                    "ar": [
                        r"\bبينضفوا الطاولة دغري\b",
                        r"\bيشيلوا الصحون بسرعة\b",
                        r"\bالطاولات انمسحت بسرعة\b",
                    ],
                    "zh": [
                        r"很快就把桌子收拾好了",
                        r"桌子很快就擦干净",
                        r"盘子马上就收走了",
                    ],
                },
            ),
        
            "breakfast_staff_rude": AspectRule(
                aspect_code="breakfast_staff_rude",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bперсонал неприветливый\b",
                        r"\bгрубо общал\w*\b",
                        r"\bхамское отношение на завтраке\b",
                        r"\bобслуживание грубое\b",
                    ],
                    "en": [
                        r"\bunfriendly staff\b",
                        r"\brude staff\b",
                        r"\bbreakfast staff were rude\b",
                        r"\bservice was rude at breakfast\b",
                        r"\bstaff attitude was bad at breakfast\b",
                    ],
                    "tr": [
                        r"\bpersonel kaba davrandı\b",
                        r"\bpersonel ilgisizdi\b",
                        r"\bkahvaltıda çalışanlar kaba\b",
                        r"\bçok kaba bir tavır vardı\b",
                    ],
                    "ar": [
                        r"\bالموظفين مو لطيفين\b",
                        r"\bتعامل سيء\b",
                        r"\bالموظفين معاملة مو حلوة\b",
                        r"\bتعاملهم كان بجّح\b",
                    ],
                    "zh": [
                        r"服务员态度不好",
                        r"服务员很不客气",
                        r"早餐服务员很凶",
                        r"服务态度很差（早餐）",
                    ],
                },
            ),
            
            "no_refill_food": AspectRule(
                aspect_code="no_refill_food",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bникто не пополнял\b",
                        r"\bничего не добавляли\b",
                        r"\bпустые лотки стояли\b",
                        r"\bлотки пустые и не пополняли\b",
                        r"\bне пополняли еду\b",
                    ],
                    "en": [
                        r"\bthey didn't refill anything\b",
                        r"\bempty trays not refilled\b",
                        r"\bbuffet not restocked\b",
                        r"\bno one refilled the food\b",
                        r"\bfood was gone and not replaced\b",
                    ],
                    "tr": [
                        r"\bboşalan ürünleri doldurmadılar\b",
                        r"\btepsiler boş kaldı\b",
                        r"\byenilemediler\b",
                        r"\byemek yenilenmedi\b",
                    ],
                    "ar": [
                        r"\bما عبّوا الأكل\b",
                        r"\bكل شي فاضي وتركوه\b",
                        r"\bما رجعوا حطوا أكل\b",
                        r"\bالبوفيه بقي فاضي\b",
                    ],
                    "zh": [
                        r"没人补菜",
                        r"盘子都是空的也没人管",
                        r"自助台没人再补",
                        r"菜没再上",
                    ],
                },
            ),
        
            "tables_left_dirty": AspectRule(
                aspect_code="tables_left_dirty",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bникто не убирал со стола\b",
                        r"\bгрязные столы\b",
                        r"\bстолы не убирают\b",
                        r"\bгрязная посуда стоит\b",
                        r"\bпосуду не уносят\b",
                    ],
                    "en": [
                        r"\bnobody cleaned the tables\b",
                        r"\btables were left dirty\b",
                        r"\bdirty tables\b",
                        r"\bused dishes left everywhere\b",
                        r"\btables not cleaned\b",
                    ],
                    "tr": [
                        r"\bmasalar temizlenmedi\b",
                        r"\bmasalar kirliydi\b",
                        r"\bkirli tabaklar kaldı masada\b",
                        r"\btabakları almadılar\b",
                    ],
                    "ar": [
                        r"\bالطاولات وسخة\b",
                        r"\bالطاولة ما نضفوها\b",
                        r"\bصحون وسخة ضلت عالطاولة\b",
                        r"\bما حدا عم ينضف الطاولات\b",
                    ],
                    "zh": [
                        r"桌子很脏",
                        r"桌子没人擦",
                        r"桌上都是脏盘子",
                        r"盘子都没收",
                    ],
                },
            ),
        
            "ignored_requests": AspectRule(
                aspect_code="ignored_requests",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bпришлось просить несколько раз\b",
                        r"\bигнорировал\w* просьбы\b",
                        r"\bпросили и не принесли\b",
                        r"\bобращались но не реагировали\b",
                    ],
                    "en": [
                        r"\bhad to ask several times\b",
                        r"\bignored us\b",
                        r"\bwe asked but no one came\b",
                        r"\bwe asked and nothing happened\b",
                    ],
                    "tr": [
                        r"\bdefalarca istemek zorunda kaldık\b",
                        r"\bisteyince bile getirmediler\b",
                        r"\bricaya rağmen ilgilenmediler\b",
                        r"\bpersonel ilgilenmedi\b",
                    ],
                    "ar": [
                        r"\bتجاهلونا\b",
                        r"\bبدنا نطلب أكتر من مرة\b",
                        r"\bطلبنا وما جابوا شي\b",
                        r"\bولا حدا رد علينا\b",
                    ],
                    "zh": [
                        r"我们说了好几次才有人理",
                        r"他们基本不理我们",
                        r"我们要了也没人拿来",
                        r"服务员不管我们说什么都不理",
                    ],
                },
            ),
        
            "food_enough_for_all": AspectRule(
                aspect_code="food_enough_for_all",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bеды хватало всем\b",
                        r"\bвсем хватило\b",
                        r"\bвсего было достаточно\b",
                        r"\bеды достаточно\b",
                    ],
                    "en": [
                        r"\bthere was enough food for everyone\b",
                        r"\bplenty of food for everyone\b",
                        r"\bfood for everyone\b",
                        r"\bmore than enough food\b",
                    ],
                    "tr": [
                        r"\byemek herkese yetiyordu\b",
                        r"\bherkese yeterince vardı\b",
                        r"\byeterince yiyecek vardı\b",
                    ],
                    "ar": [
                        r"\bالأكل كان مكفي الكل\b",
                        r"\bالأكل مكفي للجميع\b",
                        r"\bالأكل كتير ومكفي\b",
                    ],
                    "zh": [
                        r"食物够大家吃",
                        r"东西很多够大家吃",
                        r"吃的很充足",
                    ],
                },
            ),
        
            "kept_restocking": AspectRule(
                aspect_code="kept_restocking",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bвсё постоянно подносили\b",
                        r"\bпостоянно пополняли\b",
                        r"\bбыстро подносили еду\b",
                        r"\bничего не заканчивалось\b",
                    ],
                    "en": [
                        r"\bthey kept bringing more food\b",
                        r"\bconstantly restocking\b",
                        r"\bthey refilled everything quickly\b",
                        r"\bthe buffet was constantly topped up\b",
                    ],
                    "tr": [
                        r"\bsürekli tazeliyorlardı\b",
                        r"\byeniden getiriyorlardı\b",
                        r"\bdevamlı yenilendi\b",
                        r"\bbüfe sürekli dolduruldu\b",
                    ],
                    "ar": [
                        r"\bكل شوي عم يجيبوا أكل جديد\b",
                        r"\bضلوا يعبّوا الأكل\b",
                        r"\bما كان يفضى البوفيه\b",
                    ],
                    "zh": [
                        r"一直有补菜",
                        r"不停地在补菜",
                        r"菜一空就补上",
                        r"自助台一直有人补",
                    ],
                },
            ),
            
            "tables_available": AspectRule(
                aspect_code="tables_available",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bместо всегда было\b",
                        r"\bнашли стол без проблем\b",
                        r"\bлегко найти стол\b",
                        r"\bвсегда были свободные столы\b",
                    ],
                    "en": [
                        r"\beasy to find a table\b",
                        r"\balways found a table\b",
                        r"\bplenty of tables\b",
                        r"\bno problem finding a table\b",
                    ],
                    "tr": [
                        r"\bhemen masa bulduk\b",
                        r"\bmasa bulmak kolaydı\b",
                        r"\bboş masa vardı\b",
                        r"\bher zaman masa vardı\b",
                    ],
                    "ar": [
                        r"\bلقينا طاولة بسرعة\b",
                        r"\bعلى طول لقينا طاولة\b",
                        r"\bدايمًا في مكان تقعد\b",
                    ],
                    "zh": [
                        r"很容易找到位子",
                        r"很容易有桌子",
                        r"一直有空位",
                        r"找座位完全没问题",
                    ],
                },
            ),
        
            "no_queue": AspectRule(
                aspect_code="no_queue",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bбез очередей\b",
                        r"\bбез толпы\b",
                        r"\bне пришлось ждать в очереди\b",
                        r"\bне было очереди\b",
                    ],
                    "en": [
                        r"\bno line\b",
                        r"\bno long line\b",
                        r"\bno queue for breakfast\b",
                        r"\bdidn't have to wait in line\b",
                    ],
                    "tr": [
                        r"\bsıra yoktu\b",
                        r"\bkuyruk yoktu\b",
                        r"\bkahvaltıda sıra beklemedik\b",
                    ],
                    "ar": [
                        r"\bما كان في طوابير\b",
                        r"\bما اضطرّينا نوقف بالدور\b",
                        r"\bما في دور عالفطور\b",
                    ],
                    "zh": [
                        r"不用排队",
                        r"几乎不用排队",
                        r"早餐不用排队拿",
                        r"没有排长队",
                    ],
                },
            ),
        
            "breakfast_flow_ok": AspectRule(
                aspect_code="breakfast_flow_ok",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bорганизовано удобно\b",
                        r"\bзавтрак хорошо организован\b",
                        r"\bвсё очень удобно организовано\b",
                        r"\bудобная организация завтрака\b",
                    ],
                    "en": [
                        r"\bbreakfast was well organized\b",
                        r"\bvery well organized breakfast\b",
                        r"\bgood breakfast organization\b",
                        r"\bthe breakfast setup was efficient\b",
                    ],
                    "tr": [
                        r"\borganizasyon iyiydi\b",
                        r"\bkahvaltı iyi organize edilmişti\b",
                        r"\bkahvaltı düzeni rahattı\b",
                    ],
                    "ar": [
                        r"\bالتنظيم منيح\b",
                        r"\bالفطور منظم\b",
                        r"\bكل شي منظم بفترة الفطور\b",
                    ],
                    "zh": [
                        r"早餐安排得很好",
                        r"早餐动线安排得很好",
                        r"早餐组织得很顺",
                        r"早餐流程很顺畅",
                    ],
                },
            ),
        
            "food_ran_out": AspectRule(
                aspect_code="food_ran_out",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bничего не осталось\b",
                        r"\bк \d+.* уже ничего не было\b",
                        r"\bк \d+ утра уже ничего не было\b",
                        r"\bвсё съели и не обновляли\b",
                        r"\bпочти ничего не осталось\b",
                    ],
                    "en": [
                        r"\bnothing left by\b",
                        r"\balmost nothing left\b",
                        r"\bfood was gone\b",
                        r"\bby the time we came there was no food\b",
                    ],
                    "tr": [
                        r"\b9'da neredeyse hiçbir şey kalmamıştı\b",
                        r"\byemek kalmamıştı\b",
                        r"\bkahvaltıya indiğimizde neredeyse hiçbir şey yoktu\b",
                    ],
                    "ar": [
                        r"\bما بقي شي عالبوفيه\b",
                        r"\bكلو مخلص\b",
                        r"\bنزلنا ما لقينا كتير أكل\b",
                    ],
                    "zh": [
                        r"九点多几乎没东西了",
                        r"到九点什么都没了",
                        r"去的时候基本都被拿光了",
                        r"几乎什么都不剩",
                    ],
                },
            ),
        
            "had_to_wait_food": AspectRule(
                aspect_code="had_to_wait_food",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bпришлось ждать еду\b",
                        r"\bждали пока что-то вынесут\b",
                        r"\bдолго ждали, когда что-то принесут\b",
                    ],
                    "en": [
                        r"\bwe had to wait for food\b",
                        r"\bhad to wait for them to bring more\b",
                        r"\bwe waited for them to restock\b",
                        r"\bwaited a long time for food\b",
                    ],
                    "tr": [
                        r"\byemek beklemek zorunda kaldık\b",
                        r"\byeni yemek gelmesini bekledik\b",
                        r"\buzun süre bekledik\b",
                    ],
                    "ar": [
                        r"\bاستنينا ليجيبوا أكل\b",
                        r"\bضلّينا ناطرين يزيدوا الأكل\b",
                        r"\bتأخروا ليعَبّوا الأكل\b",
                    ],
                    "zh": [
                        r"我们还得等他们再拿出来",
                        r"等很久才补菜",
                        r"等了很久才有吃的",
                    ],
                },
            ),
            
            "no_tables_available": AspectRule(
                aspect_code="no_tables_available",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bнегде сесть\b",
                        r"\bне было свободных столов\b",
                        r"\bне было где сесть\b",
                        r"\bне нашли стол\b",
                        r"\bмест не было\b",
                    ],
                    "en": [
                        r"\bno free tables\b",
                        r"\bhard to find a table\b",
                        r"\bwe couldn't find a table\b",
                        r"\bno tables available\b",
                        r"\bthere were no seats\b",
                    ],
                    "tr": [
                        r"\bboş masa yoktu\b",
                        r"\byer bulmak çok zordu\b",
                        r"\bmasa bulamadık\b",
                        r"\boturacak yer yoktu\b",
                    ],
                    "ar": [
                        r"\bما في طاولة فاضية\b",
                        r"\bما لقينا محل نقعد\b",
                        r"\bما قدرنا نلاقي طاولة\b",
                        r"\bما في محل تقعد\b",
                    ],
                    "zh": [
                        r"没有空位",
                        r"很难找座位",
                        r"找不到桌子",
                        r"没有空桌",
                    ],
                },
            ),
        
            "long_queue": AspectRule(
                aspect_code="long_queue",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bбольшая очередь\b",
                        r"\bпришлось стоять в очереди\b",
                        r"\bдлинная очередь на завтрак\b",
                        r"\bочередь на завтрак\b",
                    ],
                    "en": [
                        r"\blong line\b",
                        r"\bhad to queue for breakfast\b",
                        r"\bwe had to wait in a long line\b",
                        r"\bline for breakfast was long\b",
                    ],
                    "tr": [
                        r"\buzun kuyruk vardı\b",
                        r"\bkahvaltı için sıra bekledik\b",
                        r"\bsırada beklemek zorunda kaldık\b",
                        r"\bkahvaltıda uzun sıra vardı\b",
                    ],
                    "ar": [
                        r"\bانتظرنا بالدور لحتى نفطر\b",
                        r"\bكان في دور طويل\b",
                        r"\bطابور طويل على الفطور\b",
                        r"\bوقفنا بالطابور لوقت الفطور\b",
                    ],
                    "zh": [
                        r"要排很长的队",
                        r"排了很久的队才吃早餐",
                        r"早餐排队很长",
                        r"为了早餐排队很久",
                    ],
                },
            ),
        
            "breakfast_area_clean": AspectRule(
                aspect_code="breakfast_area_clean",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bчистый зал\b",
                        r"\bв столовой чисто\b",
                        r"\bвсё аккуратно\b",
                        r"\bчисто на завтраке\b",
                        r"\bаккуратно сервировано\b",
                    ],
                    "en": [
                        r"\bdining area was clean\b",
                        r"\beverything was clean and tidy\b",
                        r"\bthe breakfast area was clean\b",
                        r"\bvery clean breakfast area\b",
                    ],
                    "tr": [
                        r"\bkahvaltı alanı temizdi\b",
                        r"\bher yer çok düzenliydi\b",
                        r"\bortam çok temizdi\b",
                        r"\bkahvaltı salonu tertemizdi\b",
                    ],
                    "ar": [
                        r"\bالمكان نظيف\b",
                        r"\bمحل الفطور نظيف\b",
                        r"\bكلشي نضيف\b",
                        r"\bمنطقة الفطور كتير نظيفة\b",
                    ],
                    "zh": [
                        r"用餐区很干净",
                        r"早餐区域很干净",
                        r"环境很干净整洁",
                        r"早餐地方很整洁",
                    ],
                },
            ),
        
            "tables_cleaned_quickly": AspectRule(
                aspect_code="tables_cleaned_quickly",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bстолы быстро протирали\b",
                        r"\bсразу убирали посуду\b",
                        r"\bубирали со стола сразу\b",
                        r"\bстол быстро убрали\b",
                    ],
                    "en": [
                        r"\bthey cleaned the tables quickly\b",
                        r"\bthey cleared tables fast\b",
                        r"\btable was cleaned immediately\b",
                        r"\bused dishes were taken right away\b",
                    ],
                    "tr": [
                        r"\bmasaları hemen temizlediler\b",
                        r"\bmasalar hemen toplandı\b",
                        r"\bkirli tabakları hemen aldılar\b",
                        r"\bçok hızlı masayı sildiler\b",
                    ],
                    "ar": [
                        r"\bبينضفوا الطاولات بسرعة\b",
                        r"\bبيشيلوا الصحون بسرعة\b",
                        r"\bالطاولة انمسحت بسرعة\b",
                        r"\bنضفوا السفرة فورًا\b",
                    ],
                    "zh": [
                        r"很快就收桌子",
                        r"很快就把桌子擦干净",
                        r"盘子马上就收走了",
                        r"桌子很快就清理了",
                    ],
                },
            ),
        
            "dirty_tables": AspectRule(
                aspect_code="dirty_tables",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bгрязные столы\b",
                        r"\bстолы не убирают\b",
                        r"\bгрязная посуда стоит\b",
                        r"\bстол липкий\b",
                        r"\bлипкий стол\b",
                        r"\bвсё в крошках\b",
                    ],
                    "en": [
                        r"\bdirty tables\b",
                        r"\btables not cleaned\b",
                        r"\bused dishes left everywhere\b",
                        r"\bsticky tables\b",
                        r"\bcrumbs everywhere\b",
                        r"\btables were left dirty\b",
                    ],
                    "tr": [
                        r"\bmasalar kirliydi\b",
                        r"\bmasalar temizlenmiyordu\b",
                        r"\bkirli tabaklar kaldı masada\b",
                        r"\byapış yapış masa\b",
                        r"\bher yerde kırıntı\b",
                    ],
                    "ar": [
                        r"\bالطاولات وسخة\b",
                        r"\bالطاولة ما نضفوها\b",
                        r"\bصحون وسخة ضلت عالطاولة\b",
                        r"\bالطاولة لزقة\b",
                        r"\bفتافيت بكل مكان\b",
                    ],
                    "zh": [
                        r"桌子很脏",
                        r"桌子没人擦",
                        r"桌上都是脏盘子",
                        r"桌子黏黏的",
                        r"到处都是碎屑",
                    ],
                },
            ),
            
            "dirty_dishes_left": AspectRule(
                aspect_code="dirty_dishes_left",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bгрязная посуда стоит\b",
                        r"\bгрязная посуда осталась\b",
                        r"\bпосуду не уносят\b",
                        r"\bгрязная посуда на столах\b",
                    ],
                    "en": [
                        r"\bused dishes left everywhere\b",
                        r"\bdirty dishes left on the tables\b",
                        r"\bthey didn't clear the dishes\b",
                        r"\bplates left on the tables\b",
                    ],
                    "tr": [
                        r"\bkirli tabaklar kaldı masada\b",
                        r"\btabakları almadılar\b",
                        r"\bkirli tabakları toplamıyorlardı\b",
                    ],
                    "ar": [
                        r"\bصحون وسخة ضلت عالطاولة\b",
                        r"\bما شالوا الصحون\b",
                        r"\bالصحون الوسخة بعدها عالطاولة\b",
                    ],
                    "zh": [
                        r"桌上都是脏盘子",
                        r"盘子都没收",
                        r"脏的餐具还留在桌上",
                    ],
                },
            ),
        
            "buffet_area_messy": AspectRule(
                aspect_code="buffet_area_messy",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bгрязно возле еды\b",
                        r"\bгрязно у раздачи\b",
                        r"\bгрязно у буфета\b",
                        r"\bбардак у шведского стола\b",
                    ],
                    "en": [
                        r"\barea around the buffet was messy\b",
                        r"\bmessy around the food\b",
                        r"\bthe buffet area was dirty\b",
                        r"\bfood area was not clean\b",
                    ],
                    "tr": [
                        r"\bbüfe etrafı dağınıktı\b",
                        r"\byemek kısmı dağınıktı\b",
                        r"\bservis kısmı kirliydi\b",
                        r"\byemek alanı pek temiz değildi\b",
                    ],
                    "ar": [
                        r"\bالمنطقة تبع الفطور وسخة\b",
                        r"\bحوالي الأكل كان وسخ\b",
                        r"\bمكان الأكل كان مكركب\b",
                    ],
                    "zh": [
                        r"自助台那边很乱",
                        r"自助台那边很脏",
                        r"拿吃的地方很脏",
                        r"早餐台一片乱",
                    ],
                },
            ),
        
            "good_value": AspectRule(
                aspect_code="good_value",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bотличное соотношение цена и качеств\w*\b",
                        r"\bочень хорошее качество за эти деньги\b",
                        r"\bза такие деньги просто супер\b",
                        r"\bцена оправдана\b",
                        r"\bцена полностью оправдана\b",
                    ],
                    "en": [
                        r"\bgreat value for money\b",
                        r"\bexcellent value\b",
                        r"\bgood value\b",
                        r"\bgood quality for the price\b",
                        r"\baffordable for this level\b",
                    ],
                    "tr": [
                        r"\bfiyat performansı çok iyiydi\b",
                        r"\bfiyatına göre harika\b",
                        r"\bbu fiyata gayet iyi\b",
                        r"\bparasına değer\b",
                    ],
                    "ar": [
                        r"\bالسعر مناسب\b",
                        r"\bالقيمة مقابل السعر ممتازة\b",
                        r"\bبهاد السعر كتير منيح\b",
                        r"\bعنجد بيسوى هالمصاري\b",
                    ],
                    "zh": [
                        r"性价比很高",
                        r"很值这个价",
                        r"这个价位很不错",
                        r"物有所值",
                    ],
                },
            ),
        
            "worth_the_price": AspectRule(
                aspect_code="worth_the_price",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bцена оправдана\b",
                        r"\bцена полностью оправдана\b",
                        r"\bполностью стоит своих денег\b",
                        r"\bстоит своих денег\b",
                    ],
                    "en": [
                        r"\bworth the price\b",
                        r"\bworth the money\b",
                        r"\btotally worth it for the price\b",
                        r"\bdefinitely worth the money\b",
                    ],
                    "tr": [
                        r"\bparasına değer\b",
                        r"\bbu paraya değer\b",
                        r"\bgayet değiyor\b",
                    ],
                    "ar": [
                        r"\bعنجد بيسوى هالمصاري\b",
                        r"\bبيستاهل السعر\b",
                        r"\bيسوى كل قرش\b",
                    ],
                    "zh": [
                        r"很值这个价",
                        r"很划算这个价格",
                        r"真的值这个钱",
                        r"物超所值",
                    ],
                },
            ),
        
            "affordable_for_level": AspectRule(
                aspect_code="affordable_for_level",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bнедорого для такого уровня\b",
                        r"\bдля такого уровня очень недорого\b",
                        r"\bза такие условия недорого\b",
                    ],
                    "en": [
                        r"\baffordable for this level\b",
                        r"\bgood price for this level\b",
                        r"\bcheap for what you get\b",
                        r"\bgood deal for this quality\b",
                    ],
                    "tr": [
                        r"\bbu seviyeye göre uygun fiyatlı\b",
                        r"\bbu kalite için uygun fiyat\b",
                        r"\bseviyesine göre ucuzdu\b",
                    ],
                    "ar": [
                        r"\bرخيص مقارنة بالمستوى\b",
                        r"\bالسعر كتير منيح لهالمستوى\b",
                        r"\bبهالخدمة السعر منيح\b",
                    ],
                    "zh": [
                        r"这个水平来说不算贵",
                        r"这个质量来说很便宜",
                        r"以这个水平来说价格很可以",
                    ],
                },
            ),
            
            "overpriced": AspectRule(
                aspect_code="overpriced",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bслишком дорого\b",
                        r"\bдорого для такого уровня\b",
                        r"\bзавышенная цена\b",
                        r"\bцена завышена\b",
                        r"\bдорого за такое\b",
                    ],
                    "en": [
                        r"\btoo expensive\b",
                        r"\boverpriced\b",
                        r"\bprice was too high\b",
                        r"\bway too expensive for what it is\b",
                    ],
                    "tr": [
                        r"\bçok pahalıydı\b",
                        r"\bfiyat fazla yüksekti\b",
                        r"\bfazla pahalı\b",
                        r"\bbuna göre pahalı\b",
                    ],
                    "ar": [
                        r"\bغالي\b",
                        r"\bغالي عالفاضي\b",
                        r"\bالسعر غالي ع شي متلو\b",
                        r"\bالسعر عالي زيادة\b",
                    ],
                    "zh": [
                        r"太贵了",
                        r"价格太高",
                        r"这个价格太夸张",
                        r"性价比太低",
                    ],
                },
            ),
        
            "not_worth_price": AspectRule(
                aspect_code="not_worth_price",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bне стоит этих денег\b",
                        r"\bне оправдывает цену\b",
                        r"\bне стоит своих денег\b",
                        r"\bза эти деньги это не то\b",
                    ],
                    "en": [
                        r"\bnot worth the price\b",
                        r"\bnot worth the money\b",
                        r"\bpoor value\b",
                        r"\bbad value for money\b",
                    ],
                    "tr": [
                        r"\bparasına değmez\b",
                        r"\bbu paraya değmez\b",
                        r"\bverilen paraya değmedi\b",
                        r"\bfiyatına değmezdi\b",
                    ],
                    "ar": [
                        r"\bما بيسوى هالمصاري\b",
                        r"\bما بيستاهل السعر\b",
                        r"\bدافع أكتر من اللازم ع شي ما بيستاهل\b",
                    ],
                    "zh": [
                        r"不值这个价",
                        r"不值这个钱",
                        r"不值这么贵",
                        r"性价比很低",
                    ],
                },
            ),
        
            "expected_better_for_price": AspectRule(
                aspect_code="expected_better_for_price",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bза такие деньги ожидаешь лучше\b",
                        r"\bза такие деньги должно быть лучше\b",
                        r"\bмы ожидали больше за эту цену\b",
                        r"\bожидали лучше за такие деньги\b",
                    ],
                    "en": [
                        r"\bfor that price we expected more\b",
                        r"\bfor the price we expected better\b",
                        r"\bnot the level we expected for this price\b",
                        r"\bshould be better for the price\b",
                    ],
                    "tr": [
                        r"\bbu fiyata daha iyisini beklersin\b",
                        r"\bbu fiyata böyle olmasını beklemezdik\b",
                        r"\bbu fiyata daha iyi olmalıydı\b",
                    ],
                    "ar": [
                        r"\bبهالسعر كنا متوقعين أحسن\b",
                        r"\bعلى هالسعر توقعت شي أحسن\b",
                        r"\bبهالمصاري لازم يكون أحسن من هيك\b",
                    ],
                    "zh": [
                        r"这个价格应该更好",
                        r"这个价位我们以为会更好",
                        r"按这个价位来说应该更好一点",
                    ],
                },
            ),
        
            "photos_misleading": AspectRule(
                aspect_code="photos_misleading",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bна фото выглядело лучше\b",
                        r"\bна фото номер лучше\b",
                        r"\bна фото всё лучше\b",
                        r"\bв реальности хуже\b",
                        r"\bвживую хуже чем на фото\b",
                    ],
                    "en": [
                        r"\blooked better in the pictures\b",
                        r"\broom looked better in the photos\b",
                        r"\blooked better in the photos than in real life\b",
                        r"\bnot like the photos\b",
                    ],
                    "tr": [
                        r"\bfotoğraflarda daha iyi görünüyordu\b",
                        r"\bgerçekte fotoğraflardaki gibi değildi\b",
                        r"\bilanla aynı değildi\b",
                        r"\breklamdaki gibi değil\b",
                    ],
                    "ar": [
                        r"\bبالصور أرتب\b",
                        r"\bبالصور شكله أحسن\b",
                        r"\bمش متل الصور\b",
                        r"\bالصور بتبينه أحسن من الحقيقة\b",
                    ],
                    "zh": [
                        r"照片上看起来更好",
                        r"实物没有照片好",
                        r"和照片不一样",
                        r"跟介绍/图片差很多",
                    ],
                },
            ),
        
            "quality_below_expectation": AspectRule(
                aspect_code="quality_below_expectation",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bожидали выше уровень\b",
                        r"\bожидали уровень повыше\b",
                        r"\bпо описанию думали будет лучше\b",
                        r"\bэто не тянет на такой ценник\b",
                        r"\bкачество ниже ожиданий\b",
                    ],
                    "en": [
                        r"\bwe expected more from the description\b",
                        r"\bnot the level we expected\b",
                        r"\bnot the level we expected for this price\b",
                        r"\bquality below expectations\b",
                        r"\bbelow expectations\b",
                    ],
                    "tr": [
                        r"\bilanına göre daha düşük seviyede\b",
                        r"\bbeklenti altındaydı\b",
                        r"\bbu seviyede olmasını beklerdik ama daha düşüktü\b",
                        r"\bbeklentimizin altındaydı\b",
                    ],
                    "ar": [
                        r"\bحسب الوصف توقعنا مستوى أعلى\b",
                        r"\bتحت التوقعات\b",
                        r"\bمو ع قد التوقع\b",
                        r"\bالمستوى أقل من اللي كنا متخيلينه\b",
                    ],
                    "zh": [
                        r"跟介绍的不一样",
                        r"达不到我们的预期",
                        r"没有达到预期的水平",
                        r"比想象的要差",
                    ],
                },
            ),
            
            "great_location": AspectRule(
                aspect_code="great_location",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bотличное расположение\b",
                        r"\bрасположение супер\b",
                        r"\bлокация супер\b",
                        r"\bместо супер\b",
                        r"\bлокация отличная\b",
                    ],
                    "en": [
                        r"\bgreat location\b",
                        r"\bexcellent location\b",
                        r"\bperfect location\b",
                        r"\bthe location was amazing\b",
                    ],
                    "tr": [
                        r"\bkonum harikaydı\b",
                        r"\blokasyon mükemmeldi\b",
                        r"\bkonumu çok iyiydi\b",
                        r"\bkonumu şahane\b",
                    ],
                    "ar": [
                        r"\bالموقع ممتاز\b",
                        r"\bالمكان كتير منيح\b",
                        r"\bالموقع رهيب\b",
                        r"\bالموقع ممتاز فعلياً\b",
                    ],
                    "zh": [
                        r"位置很好",
                        r"地段很好",
                        r"位置非常方便",
                        r"位置太好了",
                    ],
                },
            ),
        
            "central_convenient": AspectRule(
                aspect_code="central_convenient",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bцентр рядом\b",
                        r"\bвсё рядом\b",
                        r"\bвсё под рукой\b",
                        r"\bудобно гулять\b",
                        r"\bудобно добираться до центра\b",
                    ],
                    "en": [
                        r"\bvery central\b",
                        r"\bclose to everything\b",
                        r"\bclose to all the sights\b",
                        r"\bwalking distance to everything\b",
                        r"\beasy to get to the center\b",
                    ],
                    "tr": [
                        r"\bher yere yakın\b",
                        r"\bmerkeze çok yakın\b",
                        r"\bmerkeze yakın konum\b",
                        r"\bmerkeze erişim kolay\b",
                    ],
                    "ar": [
                        r"\bقريب من كل شي\b",
                        r"\bقريب من السنتر\b",
                        r"\bكلشي حدّك\b",
                        r"\bبتوصل عالسنتر بسهولة\b",
                    ],
                    "zh": [
                        r"离市中心很近",
                        r"去哪儿都方便",
                        r"走路去哪都很方便",
                        r"很方便去市中心",
                    ],
                },
            ),
        
            "near_transport": AspectRule(
                aspect_code="near_transport",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bблизко к метро\b",
                        r"\bрядом метро\b",
                        r"\bрядом станция метро\b",
                        r"\bудобно с транспортом\b",
                        r"\bрядом транспорт\b",
                    ],
                    "en": [
                        r"\bclose to the metro\b",
                        r"\bnear public transport\b",
                        r"\bclose to public transport\b",
                        r"\bsubway station nearby\b",
                        r"\bmetro station nearby\b",
                    ],
                    "tr": [
                        r"\bmetroya yakın\b",
                        r"\btoplu taşımaya yakın\b",
                        r"\bulaşım çok rahattı\b",
                        r"\bmetro hemen yakındı\b",
                    ],
                    "ar": [
                        r"\bقريب عالـ مترو\b",
                        r"\bسهل توصل بالمواصلات\b",
                        r"\bالمواصلات قريبة كتير\b",
                        r"\bفي محطة قريبة\b",
                    ],
                    "zh": [
                        r"离地铁很近",
                        r"交通方便",
                        r"地铁站就在附近",
                        r"附近有公交/地铁",
                    ],
                },
            ),
        
            "area_has_food_shops": AspectRule(
                aspect_code="area_has_food_shops",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bрядом магазины\b",
                        r"\bрядом кафе\b",
                        r"\bмного ресторанов рядом\b",
                        r"\bрядом супермаркет\b",
                        r"\bрядом есть где поесть\b",
                    ],
                    "en": [
                        r"\blots of restaurants nearby\b",
                        r"\bshops nearby\b",
                        r"\bplenty of places to eat nearby\b",
                        r"\bconvenience stores nearby\b",
                        r"\bthere are cafes nearby\b",
                    ],
                    "tr": [
                        r"\byakında market vardı\b",
                        r"\byakında restoranlar vardı\b",
                        r"\byakında kafeler vardı\b",
                        r"\byemek için yer çoktu yakında\b",
                    ],
                    "ar": [
                        r"\bفي مطاعم وسوبرماركت حدّك\b",
                        r"\bمحلات حوالينا\b",
                        r"\bكل شي موجود حدّك (أكل/سوبرماركت)\b",
                        r"\bمحلات أكل قريبة\b",
                    ],
                    "zh": [
                        r"附近有超市",
                        r"附近有餐厅",
                        r"附近吃的很多",
                        r"楼下就有便利店/超市",
                    ],
                },
            ),
        
            "location_inconvenient": AspectRule(
                aspect_code="location_inconvenient",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bне очень удобное расположение\b",
                        r"\bрасположение неудобное\b",
                        r"\bнеудобно добираться\b",
                        r"\bместоположение неудачное\b",
                    ],
                    "en": [
                        r"\bnot a good location\b",
                        r"\blocation was not convenient\b",
                        r"\bnot a convenient location\b",
                        r"\bthe location is inconvenient\b",
                    ],
                    "tr": [
                        r"\bkonum pek iyi değildi\b",
                        r"\bkonum uygun değildi\b",
                        r"\bkonum rahatsızdı\b",
                        r"\bkonum çok elverişli değildi\b",
                    ],
                    "ar": [
                        r"\bالموقع مش منيح\b",
                        r"\bالموقع مو مريح\b",
                        r"\bالمكان مو عملي\b",
                        r"\bالموقع ما ساعدنا أبداً\b",
                    ],
                    "zh": [
                        r"位置不太好",
                        r"位置不方便",
                        r"地点不太方便",
                        r"位置不太理想",
                    ],
                },
            ),
            
            "far_from_center": AspectRule(
                aspect_code="far_from_center",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bдалеко от центра\b",
                        r"\bдалековато от центра\b",
                        r"\bдо центра далеко\b",
                        r"\bне в центре\b",
                    ],
                    "en": [
                        r"\bfar from the center\b",
                        r"\bfar from the city center\b",
                        r"\bquite far from downtown\b",
                        r"\bnot central at all\b",
                    ],
                    "tr": [
                        r"\bmerkeze uzak\b",
                        r"\bmerkezden uzakta\b",
                        r"\bşehir merkezine uzak\b",
                        r"\bpek merkezi değil\b",
                    ],
                    "ar": [
                        r"\bبعيد عن المركز\b",
                        r"\bمو قريب من السنتر\b",
                        r"\bبعيد شوي عن السنتر\b",
                        r"\bمش قريب من الداون تاون\b",
                    ],
                    "zh": [
                        r"离市中心很远",
                        r"离中心有点远",
                        r"不在市中心",
                        r"位置比较偏",
                    ],
                },
            ),
        
            "nothing_around": AspectRule(
                aspect_code="nothing_around",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bничего нет рядом\b",
                        r"\bнет магазинов рядом\b",
                        r"\bвокруг ничего нет\b",
                        r"\bпоесть негде рядом\b",
                        r"\bоколо вообще ничего\b",
                    ],
                    "en": [
                        r"\bnothing around\b",
                        r"\bnothing nearby\b",
                        r"\bno shops nearby\b",
                        r"\bno restaurants nearby\b",
                        r"\bnowhere to eat nearby\b",
                    ],
                    "tr": [
                        r"\byakında hiçbir şey yoktu\b",
                        r"\byakında market yoktu\b",
                        r"\byakında restoran yoktu\b",
                        r"\bçevrede pek bir şey yok\b",
                    ],
                    "ar": [
                        r"\bما في شي حوالي\b",
                        r"\bما في شي قريب\b",
                        r"\bما في محلات قريبة\b",
                        r"\bما في مطاعم قريبة\b",
                    ],
                    "zh": [
                        r"周围什么都没有",
                        r"附近没什么",
                        r"附近没有店",
                        r"附近没地方吃饭",
                    ],
                },
            ),
        
            "area_safe": AspectRule(
                aspect_code="area_safe",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bчувствовал[аи] себя в безопасности\b",
                        r"\bчувствовали себя в безопасности\b",
                        r"\bбезопасный район\b",
                        r"\bрайон безопасный\b",
                    ],
                    "en": [
                        r"\bfelt safe in the area\b",
                        r"\bthe area felt safe\b",
                        r"\bsafe neighborhood\b",
                        r"\bwe felt safe around the building\b",
                    ],
                    "tr": [
                        r"\bbölge güvenliydi\b",
                        r"\bkendimizi güvende hissettik\b",
                        r"\bçevre güvenliydi\b",
                        r"\bmahalle güvenliydi\b",
                    ],
                    "ar": [
                        r"\bالمنطقة أمان\b",
                        r"\bحسينا بأمان\b",
                        r"\bحوالين المكان آمن\b",
                        r"\bالمنطقة آمنة\b",
                    ],
                    "zh": [
                        r"附近很安全",
                        r"感觉很安全",
                        r"区域很安全",
                        r"这附近挺安全的",
                    ],
                },
            ),
        
            "area_quiet_at_night": AspectRule(
                aspect_code="area_quiet_at_night",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bспокойный район\b",
                        r"\bтихий район\b",
                        r"\bтихо ночью\b",
                        r"\bночью спокойно\b",
                    ],
                    "en": [
                        r"\bquiet area at night\b",
                        r"\bvery quiet at night\b",
                        r"\bpeaceful at night\b",
                        r"\bcalm neighborhood at night\b",
                    ],
                    "tr": [
                        r"\bgece de sakin\b",
                        r"\bgeceleri sessizdi\b",
                        r"\bgece çok sessizdi çevre\b",
                        r"\bmahalle gece sakin\b",
                    ],
                    "ar": [
                        r"\bالمنطقة هادية بالليل\b",
                        r"\bبالليل هادي\b",
                        r"\bالحي هادي بالليل\b",
                    ],
                    "zh": [
                        r"晚上也很安静",
                        r"夜里很安静",
                        r"周围晚上很安静",
                        r"夜里这边很安静",
                    ],
                },
            ),
        
            "entrance_clean": AspectRule(
                aspect_code="entrance_clean",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bнормальный подъезд\b",
                        r"\bчистый подъезд\b",
                        r"\bвход чистый\b",
                        r"\bподъезд чистый\b",
                        r"\bаккуратный подъезд\b",
                    ],
                    "en": [
                        r"\bentrance was clean\b",
                        r"\bclean entrance\b",
                        r"\bthe entrance was tidy\b",
                        r"\bthe hallway/entrance was clean\b",
                    ],
                    "tr": [
                        r"\bgiriş temizdi\b",
                        r"\bmerdivenler temizdi\b",
                        r"\bapartman girişi temizdi\b",
                    ],
                    "ar": [
                        r"\bالمدخل نظيف\b",
                        r"\bالمدخل كان مرتب\b",
                        r"\bالمدخل شكله منيح\b",
                    ],
                    "zh": [
                        r"入口很干净",
                        r"楼道很干净",
                        r"入口很整洁",
                        r"大门口很干净",
                    ],
                },
            ),
            
            "area_unsafe": AspectRule(
                aspect_code="area_unsafe",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bрайон стр(е|ё)мный\b",
                        r"\bрайон стремный\b",
                        r"\bнебезопасно\b",
                        r"\bнебезопасный район\b",
                        r"\bчувствуешь себя небезопасно\b",
                        r"\bрайон не самый безопасный\b",
                    ],
                    "en": [
                        r"\barea felt unsafe\b",
                        r"\bwe didn't feel safe outside\b",
                        r"\bnot a safe area\b",
                        r"\bsketchy area\b",
                        r"\bdodgy area\b",
                    ],
                    "tr": [
                        r"\bbölge pek güvenli değildi\b",
                        r"\bpek güvenli hissettirmedi\b",
                        r"\bmahalle güvensizdi\b",
                        r"\bçok güvenli bir bölge değildi\b",
                    ],
                    "ar": [
                        r"\bالمنطقة مو آمنة\b",
                        r"\bما حسّينا بأمان برا\b",
                        r"\bالمنطقة بتخوف\b",
                        r"\bالمنطقة شوي بتخوف\b",
                        r"\bالحي مو مريح أبدًا\b",
                    ],
                    "zh": [
                        r"附近不太安全",
                        r"感觉不安全",
                        r"周围有点乱",
                        r"区域不安全",
                        r"这边治安不好",
                    ],
                },
            ),
        
            "uncomfortable_at_night": AspectRule(
                aspect_code="uncomfortable_at_night",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bнеуютно выходить вечером\b",
                        r"\bночью страшно выходить\b",
                        r"\bвечером выходить некомфортно\b",
                        r"\bночью лучше не выходить\b",
                    ],
                    "en": [
                        r"\bwe didn't feel safe outside at night\b",
                        r"\bdidn't feel comfortable going out at night\b",
                        r"\bscary to go out at night\b",
                        r"\bwouldn't go out at night\b",
                    ],
                    "tr": [
                        r"\bgece dışarı çıkmak pek rahat değildi\b",
                        r"\bgece dışarı çıkmak istemedik\b",
                        r"\bgece çok güvenli hissettirmedi\b",
                    ],
                    "ar": [
                        r"\bبالليل مش مريح تطلع\b",
                        r"\bبالليل مو آمن تطلع\b",
                        r"\bبالليل ما حسّينا بأمان برا\b",
                        r"\bالليل بيخوّف شوي\b",
                    ],
                    "zh": [
                        r"晚上不太敢出门",
                        r"晚上出去有点怕",
                        r"晚上附近不太安全",
                        r"夜里不太放心出门",
                    ],
                },
            ),
        
            "entrance_dirty": AspectRule(
                aspect_code="entrance_dirty",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bподъезд грязный\b",
                        r"\bподъезд ужасный\b",
                        r"\bгрязный вход\b",
                        r"\bвход грязный\b",
                        r"\bмерзкий подъезд\b",
                    ],
                    "en": [
                        r"\bentrance was dirty\b",
                        r"\bdirty entrance\b",
                        r"\bthe entrance looked terrible\b",
                        r"\bthe hallway was dirty\b",
                    ],
                    "tr": [
                        r"\bgiriş kirliydi\b",
                        r"\bmerdivenler kirliydi\b",
                        r"\bapartman girişi kirliydi\b",
                        r"\bgiriş pek temiz değildi\b",
                    ],
                    "ar": [
                        r"\bالمدخل وسخ\b",
                        r"\bالمدخل شكله مو مريح\b",
                        r"\bالمدخل مش نظيف\b",
                    ],
                    "zh": [
                        r"入口很脏",
                        r"楼道很脏",
                        r"门口很脏",
                        r"入口看起来脏兮兮的",
                    ],
                },
            ),
        
            "people_loitering": AspectRule(
                aspect_code="people_loitering",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bподозрительные люди\b",
                        r"\bмного пьяных\b",
                        r"\bу входа какие-то личности\b",
                        r"\bтолпились у подъезда\b",
                    ],
                    "en": [
                        r"\bdrunk people outside\b",
                        r"\bpeople hanging around the entrance\b",
                        r"\bshady people outside\b",
                        r"\bsketchy people at the door\b",
                    ],
                    "tr": [
                        r"\betrafta tuhaf tipler vardı\b",
                        r"\bkapıda garip tipler vardı\b",
                        r"\bdışarıda sarhoşlar vardı\b",
                        r"\bgirişte hoş olmayan tipler vardı\b",
                    ],
                    "ar": [
                        r"\bفي ناس مزعجين عالباب\b",
                        r"\bفي عالم واقفين قدام المدخل شكلهم مو مريح\b",
                        r"\bفي ناس سكرانين برا\b",
                    ],
                    "zh": [
                        r"门口有人喝酒闹事",
                        r"门口有乱七八糟的人",
                        r"楼下有一些奇怪的人聚着",
                        r"楼下有人在门口闲晃",
                    ],
                },
            ),
        
            "easy_to_find": AspectRule(
                aspect_code="easy_to_find",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bлегко найти\b",
                        r"\bадрес найти легко\b",
                        r"\bнашли вход без проблем\b",
                        r"\bпонятно как зайти в здание\b",
                    ],
                    "en": [
                        r"\beasy to find\b",
                        r"\beasy to find the entrance\b",
                        r"\bthe building was easy to find\b",
                        r"\bclear where to go\b",
                    ],
                    "tr": [
                        r"\bbulması kolaydı\b",
                        r"\bgirişi bulmak kolaydı\b",
                        r"\bbinayı bulmak kolaydı\b",
                        r"\bnereye gireceğimizi hemen anladık\b",
                    ],
                    "ar": [
                        r"\bسهل نلاقي المدخل\b",
                        r"\bالمحل سهل تلاقيه\b",
                        r"\bالدخول سهل\b",
                        r"\bعرفنا دغري من وين نفوت\b",
                    ],
                    "zh": [
                        r"很容易找到入口",
                        r"很容易找到地址",
                        r"很好找这个楼",
                        r"一眼就能看到入口",
                    ],
                },
            ),
            
            "clear_instructions": AspectRule(
                aspect_code="clear_instructions",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bинструкции по заселению понятные\b",
                        r"\bпонятные инструкции\b",
                        r"\bинструкции заранее и всё ясно\b",
                        r"\bпонятно объяснили как зайти\b",
                        r"\bвсё объяснили как зайти\b",
                    ],
                    "en": [
                        r"\bcheck-?in instructions were clear\b",
                        r"\bclear instructions\b",
                        r"\bthey sent clear instructions\b",
                        r"\bit was clear how to get in\b",
                        r"\bthey explained how to enter\b",
                    ],
                    "tr": [
                        r"\btalimatlar çok açıktı\b",
                        r"\btalimatlar netti\b",
                        r"\bcheck[- ]?in talimatları çok netti\b",
                        r"\bnereden gireceğimizi anlattılar\b",
                    ],
                    "ar": [
                        r"\bالتعليمات واضحة\b",
                        r"\bبعتولنا تعليمات واضحة\b",
                        r"\bشرحوا كل شي بوضوح\b",
                        r"\bكان واضح كيف نفوت\b",
                    ],
                    "zh": [
                        r"指引很清楚",
                        r"入住说明很清楚",
                        r"进楼的指引很清楚",
                        r"告诉我们怎么进门，很清楚",
                    ],
                },
            ),
        
            "luggage_access_ok": AspectRule(
                aspect_code="luggage_access_ok",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bудобно добраться с чемоданом\b",
                        r"\bс чемоданами удобно\b",
                        r"\bлегко с багажом\b",
                        r"\bудобно с багажом\b",
                        r"\bне тяжело с чемоданами\b",
                    ],
                    "en": [
                        r"\beasy access with luggage\b",
                        r"\beasy to bring luggage\b",
                        r"\beasy with suitcases\b",
                        r"\bnot hard with luggage\b",
                    ],
                    "tr": [
                        r"\bvalizle girmek rahattı\b",
                        r"\bvalizle girmek kolaydı\b",
                        r"\bçanta/valizle sorun olmadı\b",
                    ],
                    "ar": [
                        r"\bسهل مع الشنط\b",
                        r"\bالدخول مع الشنط كان سهل\b",
                        r"\bما كان صعب مع الشنط\b",
                    ],
                    "zh": [
                        r"带行李进去也还可以",
                        r"拿行李进去很方便",
                        r"拖行李进去不难",
                        r"带箱子进来很方便",
                    ],
                },
            ),
        
            "hard_to_find_entrance": AspectRule(
                aspect_code="hard_to_find_entrance",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bсложно найти вход\b",
                        r"\bтрудно найти вход\b",
                        r"\bне могли найти подъезд\b",
                        r"\bне могли найти домофон\b",
                        r"\bтрудно найти здание\b",
                    ],
                    "en": [
                        r"\bhard to find the entrance\b",
                        r"\bdifficult to find the building\b",
                        r"\bhard to find the building\b",
                        r"\bnot easy to find the entrance\b",
                        r"\bwe couldn't find the entrance\b",
                    ],
                    "tr": [
                        r"\bgirişi bulmak zor\b",
                        r"\bbinayı bulmak zordu\b",
                        r"\bgirişi bulmak kolay değildi\b",
                        r"\bkapıyı bulmak zor oldu\b",
                    ],
                    "ar": [
                        r"\bصعب تلاقي المدخل\b",
                        r"\bما عرفنا من وين نفوت\b",
                        r"\bلقينا صعوبة نلاقي الباب\b",
                        r"\bما قدرنا نلاقي المدخل بسهولة\b",
                    ],
                    "zh": [
                        r"入口很难找",
                        r"很难找到门",
                        r"我们一开始找不到入口",
                        r"这个楼不好找",
                    ],
                },
            ),
        
            "confusing_access": AspectRule(
                aspect_code="confusing_access",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bнеочевидный вход\b",
                        r"\bзапутанный вход\b",
                        r"\bочень сложно попасть внутрь\b",
                        r"\bнепонятно как попасть внутрь\b",
                        r"\bнепонятно как зайти\b",
                    ],
                    "en": [
                        r"\bconfusing entrance\b",
                        r"\bconfusing access\b",
                        r"\bwe couldn't figure out how to get in\b",
                        r"\bhard to figure out how to enter\b",
                        r"\baccess was confusing\b",
                    ],
                    "tr": [
                        r"\bbina girişi karışıktı\b",
                        r"\biçeri girmek zor oldu\b",
                        r"\bkapıyı anlamak zordu\b",
                        r"\bgiriş sistemi çok karışıktı\b",
                    ],
                    "ar": [
                        r"\bالدخول معقّد\b",
                        r"\bالمدخل معقّد\b",
                        r"\bما عرفنا كيف نفوت بالبداية\b",
                        r"\bصعب تفهم كيف تفوت\b",
                    ],
                    "zh": [
                        r"进门很麻烦",
                        r"不知道怎么进楼",
                        r"进楼的流程很复杂",
                        r"入口设计很迷惑",
                    ],
                },
            ),
        
            "no_signage": AspectRule(
                aspect_code="no_signage",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bнет нормальной вывески\b",
                        r"\bникакой вывески\b",
                        r"\bнет таблички\b",
                        r"\bникакой таблички\b",
                        r"\bникаких указателей\b",
                    ],
                    "en": [
                        r"\bno sign\b",
                        r"\bno signage\b",
                        r"\bno signs outside\b",
                        r"\bnothing on the door\b",
                        r"\bthe building wasn't marked\b",
                    ],
                    "tr": [
                        r"\btabela yoktu\b",
                        r"\bhiç tabela yok\b",
                        r"\bkapıda isim yoktu\b",
                        r"\bbina üzerinde isim yoktu\b",
                    ],
                    "ar": [
                        r"\bما في أي اشارة\b",
                        r"\bما في علامة\b",
                        r"\bما في سِاين عالباب\b",
                        r"\bما في اسم عالباب\b",
                    ],
                    "zh": [
                        r"没有指示牌",
                        r"没有标识",
                        r"门口没有任何牌子",
                        r"楼下没有写名字",
                    ],
                },
            ),
        
            "luggage_access_hard": AspectRule(
                aspect_code="luggage_access_hard",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bс чемоданами тяжело\b",
                        r"\bнеудобно с багажом\b",
                        r"\bс багажом тяжело\b",
                        r"\bс чемоданами ад\b",
                        r"\bтяжело тащить чемоданы\b",
                    ],
                    "en": [
                        r"\bnot easy with luggage\b",
                        r"\bhard with suitcases\b",
                        r"\bdifficult with luggage\b",
                        r"\bhad to carry our bags up\b",
                    ],
                    "tr": [
                        r"\bvalizle girmek çok zordu\b",
                        r"\bvaliz taşımak zordu\b",
                        r"\bvalizlerle çıkmak zor oldu\b",
                        r"\bçantalarla çok zorlandık\b",
                    ],
                    "ar": [
                        r"\bصعب مع الشنط\b",
                        r"\bتعبنا مع الشنط\b",
                        r"\bحمل الشنط كان صعب\b",
                        r"\bكان مرهق مع الشنط\b",
                    ],
                    "zh": [
                        r"拿着行李很不方便",
                        r"拖着行李很麻烦",
                        r"提着行李上去很累",
                        r"拿行李进去很辛苦",
                    ],
                },
            ),
            
            "cozy_atmosphere": AspectRule(
                aspect_code="cozy_atmosphere",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bочень уютно\b",
                        r"\bуютная атмосфера\b",
                        r"\bприятная атмосфера\b",
                        r"\bдомашняя атмосфера\b",
                        r"\bкак дома\b",
                        r"\bочень приятно находиться\b",
                    ],
                    "en": [
                        r"\bvery cozy\b",
                        r"\bsuper cozy\b",
                        r"\bnice atmosphere\b",
                        r"\bpleasant atmosphere\b",
                        r"\bfelt like home\b",
                        r"\bfelt homelike\b",
                        r"\bhomey atmosphere\b",
                    ],
                    "tr": [
                        r"\bçok samimi bir atmosfer vardı\b",
                        r"\bçok sıcak bir ortam vardı\b",
                        r"\bçok rahat bir his veriyor\b",
                        r"\bev gibi hissettirdi\b",
                        r"\bçok huzurluydu ortam\b",
                    ],
                    "ar": [
                        r"\bجو كتير دافئ\b",
                        r"\bجو كتير مريح\b",
                        r"\bبتحس كأنك ببيتك\b",
                        r"\bجو مريح\b",
                        r"\bالإحساس دافي\b",
                    ],
                    "zh": [
                        r"很温馨",
                        r"很舒适",
                        r"很有家的感觉",
                        r"气氛很好",
                        r"氛围很好很温暖",
                    ],
                },
            ),
        
            "nice_design": AspectRule(
                aspect_code="nice_design",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bкрасивый интерьер\b",
                        r"\bстильно\b",
                        r"\bдизайн очень красивый\b",
                        r"\bкрасивый дизайн\b",
                        r"\bочень красивый интерьер\b",
                    ],
                    "en": [
                        r"\bstylish interior\b",
                        r"\bbeautiful design\b",
                        r"\bnice design\b",
                        r"\bdecor was beautiful\b",
                        r"\bwell designed\b",
                    ],
                    "tr": [
                        r"\btasarım çok şıktı\b",
                        r"\bdekorasyon çok güzeldi\b",
                        r"\bçok şık dekore edilmişti\b",
                        r"\bortam çok zevkli döşenmişti\b",
                    ],
                    "ar": [
                        r"\bالديكور حلو\b",
                        r"\bالمكان شكله حلو\b",
                        r"\bستايل حلو\b",
                        r"\bالتصميم كتير حلو\b",
                    ],
                    "zh": [
                        r"装修很好看",
                        r"装修很有设计感",
                        r"很有风格",
                        r"布置很好看",
                        r"装潢很漂亮",
                    ],
                },
            ),
        
            "good_vibe": AspectRule(
                aspect_code="good_vibe",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bприятное место\b",
                        r"\bхотелось остаться дольше\b",
                        r"\bочень приятная атмосфера\b",
                        r"\bочень приятное место\b",
                        r"\bклассная атмосфера\b",
                    ],
                    "en": [
                        r"\bgreat vibe\b",
                        r"\bwe loved the vibe\b",
                        r"\bnice vibe\b",
                        r"\bdidn't want to leave\b",
                        r"\breally nice atmosphere\b",
                    ],
                    "tr": [
                        r"\bortamın havasını çok sevdik\b",
                        r"\bçok güzel bir atmosfer vardı\b",
                        r"\baurası çok iyiydi\b",
                        r"\borgu çok güzeldi\b",
                    ],
                    "ar": [
                        r"\bحبّينا الجو\b",
                        r"\bعنجد الجو حلو\b",
                        r"\bالمكان حلو جوًا\b",
                        r"\bما بدك تطلع من المكان\b",
                    ],
                    "zh": [
                        r"很喜欢这里的感觉",
                        r"氛围很好",
                        r"整体感觉很棒",
                        r"都不想走了",
                    ],
                },
            ),
        
            "not_cozy": AspectRule(
                aspect_code="not_cozy",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bнеуютно\b",
                        r"\bнеуютная атмосфера\b",
                        r"\bнет ощущения уюта\b",
                        r"\bне чувствуется уют\b",
                        r"\bатмосфера .*холодная\b",
                        r"\bхолодная атмосфера\b",
                    ],
                    "en": [
                        r"\bnot cozy\b",
                        r"\bnot very cozy\b",
                        r"\bcold atmosphere\b",
                        r"\bdidn't feel welcoming\b",
                        r"\bdidn't feel cozy\b",
                        r"\bnot homely at all\b",
                    ],
                    "tr": [
                        r"\batmosfer pek sıcak değildi\b",
                        r"\bsoğuk bir his vardı\b",
                        r"\brahat hissettirmedi\b",
                        r"\bev gibi hissettirmedi\b",
                        r"\bçok da sıcak ortam değildi\b",
                    ],
                    "ar": [
                        r"\bالجو بارد\b",
                        r"\bما في راحة بالمكان\b",
                        r"\bمش مريح\b",
                        r"\bالمكان ما حسّيته دافئ\b",
                        r"\bما حسّيته بيتي\b",
                    ],
                    "zh": [
                        r"不太温馨",
                        r"没有家的感觉",
                        r"氛围有点冷淡",
                        r"感觉不太舒服",
                        r"不是很有家的感觉",
                    ],
                },
            ),
        
            "gloomy_feel": AspectRule(
                aspect_code="gloomy_feel",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bмрачно\b",
                        r"\bугнетающе\b",
                        r"\bдавит\b",
                        r"\bатмосфера мрачная\b",
                        r"\bнеуютно и мрачно\b",
                    ],
                    "en": [
                        r"\bfelt depressing\b",
                        r"\bfelt gloomy\b",
                        r"\bgloomy atmosphere\b",
                        r"\bdepressing vibe\b",
                        r"\bplace felt depressing\b",
                    ],
                    "tr": [
                        r"\bortam biraz kasvetliydi\b",
                        r"\bkasvetli bir hava vardı\b",
                        r"\bpek iç açıcı değildi\b",
                        r"\bçok karanlık ve kasvetli bir his vardı\b",
                    ],
                    "ar": [
                        r"\bالمكان كئيب\b",
                        r"\bشوي كئيب\b",
                        r"\bالجو شوي كئيب\b",
                        r"\bالإحساس مو مريح، كئيب\b",
                    ],
                    "zh": [
                        r"有点压抑",
                        r"感觉有点压抑",
                        r"氛围有点压抑",
                        r"感觉有点沉闷",
                        r"有点阴沉的感觉",
                    ],
                },
            ),
        
            "dated_look": AspectRule(
                aspect_code="dated_look",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bсоветский ремонт\b",
                        r"\bстарый ремонт\b",
                        r"\bвсё уставшее\b",
                        r"\bвсё старенькое\b",
                        r"\bсмотрится старо\b",
                    ],
                    "en": [
                        r"\btired looking\b",
                        r"\bdated decor\b",
                        r"\bdated look\b",
                        r"\blooked old\b",
                        r"\bworn out look\b",
                        r"\bfelt old and tired\b",
                    ],
                    "tr": [
                        r"\bdekorasyon eskiydi\b",
                        r"\beski görünüyordu\b",
                        r"\boda yorgun görünüyordu\b",
                        r"\bçok eski duruyordu\b",
                        r"\byıpranmış görünüyordu\b",
                    ],
                    "ar": [
                        r"\bالشكل قديم\b",
                        r"\bالديكور قديم\b",
                        r"\bمبين قديم\b",
                        r"\bالمكان شكله تعبان\b",
                        r"\bمبين مستهلَك\b",
                    ],
                    "zh": [
                        r"装修很旧",
                        r"显得很旧",
                        r"看起来很老旧",
                        r"房间有点旧旧的感觉",
                        r"感觉比较老式",
                    ],
                },
            ),
        
            "soulless_feel": AspectRule(
                aspect_code="soulless_feel",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bнет ощущения уюта\b",
                        r"\bне чувствуется уют\b",
                        r"\bбез души\b",
                        r"\bатмосфера безликая\b",
                        r"\bказённая атмосфера\b",
                    ],
                    "en": [
                        r"\bfelt cheap\b",
                        r"\bfelt soulless\b",
                        r"\bvery impersonal vibe\b",
                        r"\bdidn't feel welcoming\b",
                        r"\bno character\b",
                    ],
                    "tr": [
                        r"\brahat hissettirmedi\b",
                        r"\bev gibi hissettirmedi\b",
                        r"\bir ruhu yok gibiydi\b",
                        r"\batmosfer çok ruhsuzdu\b",
                        r"\bpek sıcak bir hava yoktu\b",
                    ],
                    "ar": [
                        r"\bالمكان بارد (مش إحساس دافي)\b",
                        r"\bما في روح بالمكان\b",
                        r"\bما بتحس فيه روح\b",
                        r"\bما حسّينا بترحيب\b",
                    ],
                    "zh": [
                        r"很没有感觉",
                        r"感觉很冷清没有人情味",
                        r"没有什么氛围",
                        r"感觉很商业化没什么温度",
                    ],
                },
            ),
        
            "fresh_smell_common": AspectRule(
                aspect_code="fresh_smell_common",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bв коридоре приятно пахнет\b",
                        r"\bприятный запах\b",
                        r"\bсвежо в коридоре\b",
                        r"\bсвежий запах\b",
                        r"\bникаких запахов\b",
                    ],
                    "en": [
                        r"\bhallway smelled fresh\b",
                        r"\bnice smell in the hallway\b",
                        r"\bsmelled nice\b",
                        r"\bpleasant smell\b",
                    ],
                    "tr": [
                        r"\bkoridor temiz kokuyordu\b",
                        r"\bgüzel kokuyordu\b",
                        r"\bhoş bir koku vardı\b",
                        r"\bkokusu güzeldi\b",
                    ],
                    "ar": [
                        r"\bريحة حلوة بالممر\b",
                        r"\bالريحة حلوة\b",
                        r"\bما في ريحة مزعجة\b",
                        r"\bما كان في أي ريحة بشعة\b",
                    ],
                    "zh": [
                        r"走廊味道很好",
                        r"走廊很清新",
                        r"没有异味",
                        r"闻起来很干净",
                    ],
                },
            ),
        
            "no_bad_smell": AspectRule(
                aspect_code="no_bad_smell",
                polarity_hint="positive",
                patterns_by_lang={
                    "ru": [
                        r"\bникаких запахов\b",
                        r"\bнет неприятного запаха\b",
                        r"\bне пахло плохо\b",
                    ],
                    "en": [
                        r"\bno smell\b",
                        r"\bno bad smell\b",
                        r"\bno unpleasant smell\b",
                        r"\bthere was no bad odor\b",
                    ],
                    "tr": [
                        r"\bkötü koku yoktu\b",
                        r"\bhiç koku yoktu\b",
                        r"\brahatsız eden bir koku yoktu\b",
                    ],
                    "ar": [
                        r"\bما في ريحة مزعجة\b",
                        r"\bما كان في ريحة بشعة\b",
                        r"\bما في ريحة مو حلوة\b",
                    ],
                    "zh": [
                        r"没有异味",
                        r"没有什么味道",
                        r"没有难闻的味道",
                    ],
                },
            ),
        
            "bad_smell_common": AspectRule(
                aspect_code="bad_smell_common",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bв коридоре воняет\b",
                        r"\bвонь в коридоре\b",
                        r"\bнеприятный запах\b",
                        r"\bпахло неприятно\b",
                    ],
                    "en": [
                        r"\bhallway smelled bad\b",
                        r"\bbad smell in the hallway\b",
                        r"\bsmelled bad in the hallway\b",
                        r"\bunpleasant smell in the corridor\b",
                    ],
                    "tr": [
                        r"\bkoridorda kötü koku vardı\b",
                        r"\bkoridorda kötü kokuyordu\b",
                        r"\brahatsız edici bir koku vardı\b",
                    ],
                    "ar": [
                        r"\bريحة مش منيحة بالممر\b",
                        r"\bريحة بشعة\b",
                        r"\bريحة مو حلوة بالممر\b",
                    ],
                    "zh": [
                        r"走廊有味道",
                        r"走廊有臭味",
                        r"有股很难闻的味道在走廊",
                    ],
                },
            ),
        
            "cigarette_smell": AspectRule(
                aspect_code="cigarette_smell",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bзапах сигарет\b",
                        r"\bпахло сигаретами\b",
                        r"\bпахло табаком\b",
                        r"\bзапах табака\b",
                    ],
                    "en": [
                        r"\bsmelled like cigarettes\b",
                        r"\bcigarette smell\b",
                        r"\bsmelled of tobacco\b",
                        r"\bsmelled like smoke\b",
                    ],
                    "tr": [
                        r"\bsigara kokusu\b",
                        r"\btütün kokuyordu\b",
                        r"\bsigara kokuyordu koridorda\b",
                    ],
                    "ar": [
                        r"\bريحة دخان\b",
                        r"\bريحة سجاير\b",
                        r"\bريحة سيجارة بالممر\b",
                    ],
                    "zh": [
                        r"有烟味",
                        r"都是烟味",
                        r"走廊有烟味",
                        r"有很重的香烟味",
                    ],
                },
            ),
        
            "sewage_smell": AspectRule(
                aspect_code="sewage_smell",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bзапах канализации\b",
                        r"\bпахло канализацией\b",
                        r"\bвонь из канализации\b",
                    ],
                    "en": [
                        r"\bsewage smell\b",
                        r"\bsmelled like sewage\b",
                        r"\bbad sewer smell\b",
                        r"\bdrain smell\b",
                    ],
                    "tr": [
                        r"\blağım gibi kokuyordu\b",
                        r"\bkanalizasyon kokusu\b",
                        r"\bçok kötü gider kokusu\b",
                    ],
                    "ar": [
                        r"\bريحة مجاري\b",
                        r"\bريحة صرف\b",
                        r"\bريحة مجاري قوية\b",
                    ],
                    "zh": [
                        r"下水道味",
                        r"有下水道的味道",
                        r"一股下水道的臭味",
                    ],
                },
            ),
        
            "musty_smell": AspectRule(
                aspect_code="musty_smell",
                polarity_hint="negative",
                patterns_by_lang={
                    "ru": [
                        r"\bзапах плесени\b",
                        r"\bзапах сырости\b",
                        r"\bзатхлый запах\b",
                        r"\bпахло сырым\b",
                    ],
                    "en": [
                        r"\bmusty smell\b",
                        r"\bdamp smell\b",
                        r"\bsmelled damp\b",
                        r"\bmoldy smell\b",
                    ],
                    "tr": [
                        r"\bnem kokusu\b",
                        r"\bküf kokusu\b",
                        r"\brutubet kokusu\b",
                        r"\bkokusu rutubet gibiydi\b",
                    ],
                    "ar": [
                        r"\bريحة رطوبة\b",
                        r"\bريحة عفن\b",
                        r"\bريحة عفن رطوبة\b",
                        r"\bريحة رطوبة قوية\b",
                    ],
                    "zh": [
                        r"霉味",
                        r"潮味",
                        r"发霉的味道",
                        r"很潮很闷的味道",
                    ],
                },
            ),
        }

            # обратный индекс: куда относить найденный аспект в отчёте 
            # ключ = aspect_code (тот же самый ключ, что в aspects_meta / aspect_rules) 
            # значение = список (topic_code, subtopic_code)

            self.aspect_to_subtopics: Dict[str, List[Tuple[str, str]]] = {
            # --- staff_spir / отношение и работа персонала ---

            # тон / настроение / вежливость
            "spir_friendly": [("staff_spir", "staff_attitude")],
            "spir_welcoming": [("staff_spir", "staff_attitude")],
            "spir_helpful": [("staff_spir", "staff_attitude")],
            "spir_unfriendly": [("staff_spir", "staff_attitude")],
            "spir_rude": [("staff_spir", "staff_attitude")],
            "spir_unprofessional": [("staff_spir", "staff_attitude")],
            "spir_professional": [("staff_spir", "staff_attitude")],

            # скорость и реакция
            "spir_quick_response": [("staff_spir", "staff_responsiveness")],
            "spir_slow_response": [("staff_spir", "staff_responsiveness")],
            "spir_unresponsive": [("staff_spir", "staff_responsiveness")],
            "spir_ignored_requests": [("staff_spir", "staff_responsiveness")],

            # контактность / доступность
            "spir_easy_contact": [("staff_spir", "staff_availability")],
            "spir_hard_to_contact": [("staff_spir", "staff_availability")],
            "spir_available": [("staff_spir", "staff_availability")],
            "spir_not_available": [("staff_spir", "staff_availability")],

            # решение проблем
            "spir_problem_fixed_fast": [("staff_spir", "issue_resolution")],
            "spir_problem_not_fixed": [("staff_spir", "issue_resolution")],

            # язык общения
            "spir_language_support_good": [("staff_spir", "language_support")],
            "spir_language_support_bad": [("staff_spir", "language_support")],

            # примеры из заготовки
            "bed_uncomfortable": [("comfort", "sleep_quality")],
            "spir_rude": [("staff_spir", "staff_attitude")],

            # --- checkin_stay / Заселение и проживание ---

            # checkin_speed
            "checkin_fast": [("checkin_stay", "checkin_speed")],
            "no_wait_checkin": [("checkin_stay", "checkin_speed")],
            "checkin_wait_long": [("checkin_stay", "checkin_speed")],
            "room_not_ready_delay": [("checkin_stay", "checkin_speed")],

            # room_ready
            "room_ready_on_arrival": [("checkin_stay", "room_ready")],
            "clean_on_arrival": [("checkin_stay", "room_ready")],
            "room_not_ready": [("checkin_stay", "room_ready")],
            "dirty_on_arrival": [("checkin_stay", "room_ready")],
            "leftover_trash_previous_guest": [("checkin_stay", "room_ready")],

            # access
            "access_smooth": [("checkin_stay", "access")],
            "door_code_worked": [("checkin_stay", "access")],
            "tech_access_issue": [("checkin_stay", "access")],
            "entrance_hard_to_find": [("checkin_stay", "access")],
            "no_elevator_baggage_issue": [("checkin_stay", "access")],

            # docs_payment
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

            # instructions
            "instructions_clear": [("checkin_stay", "instructions")],
            "self_checkin_easy": [("checkin_stay", "instructions")],
            "wifi_info_given": [("checkin_stay", "instructions")],
            "instructions_confusing": [("checkin_stay", "instructions")],
            "late_access_code": [("checkin_stay", "instructions")],
            "wifi_info_missing": [("checkin_stay", "instructions")],
            "had_to_figure_out": [("checkin_stay", "instructions")],

            # stay_support
            "support_during_stay_good": [("checkin_stay", "stay_support")],
            "issue_fixed_immediately": [("checkin_stay", "stay_support")],
            "support_during_stay_slow": [("checkin_stay", "stay_support")],
            "support_ignored": [("checkin_stay", "stay_support")],
            "promised_not_done": [("checkin_stay", "stay_support")],

            # checkout
            "checkout_easy": [("checkin_stay", "checkout")],
            "checkout_fast": [("checkin_stay", "checkout")],
            "checkout_slow": [("checkin_stay", "checkout")],
            "deposit_return_issue": [("checkin_stay", "checkout")],
            "checkout_no_staff": [("checkin_stay", "checkout")],

            # --- cleanliness / Чистота ---

            # arrival_clean
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

            # bathroom_state
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

            # stay_cleaning
            "housekeeping_regular": [("cleanliness", "stay_cleaning")],
            "trash_taken_out": [("cleanliness", "stay_cleaning")],
            "bed_made": [("cleanliness", "stay_cleaning")],
            "housekeeping_missed": [("cleanliness", "stay_cleaning")],
            "trash_not_taken": [("cleanliness", "stay_cleaning")],
            "bed_not_made": [("cleanliness", "stay_cleaning")],
            "had_to_request_cleaning": [("cleanliness", "stay_cleaning")],
            "dirt_accumulated": [("cleanliness", "stay_cleaning")],

            # linen_towels
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

            # smell
            "smell_of_smoke": [("cleanliness", "smell")],
            "chemical_smell_strong": [("cleanliness", "smell")],
            "fresh_smell": [("cleanliness", "smell")],

            # public_areas
            "hallway_clean": [("cleanliness", "public_areas")],
            "common_areas_clean": [("cleanliness", "public_areas")],
            "hallway_dirty": [("cleanliness", "public_areas")],
            "elevator_dirty": [("cleanliness", "public_areas")],
            "hallway_bad_smell": [("cleanliness", "public_areas")],
            "entrance_feels_unsafe": [("cleanliness", "public_areas")],

            # --- comfort / Комфорт проживания ---

            # room_equipment
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

            # sleep_quality
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

            # noise
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

            # space_light
            "room_spacious": [("comfort", "space_light")],
            "good_layout": [("comfort", "space_light")],
            "cozy_feel": [("comfort", "space_light")],
            "bright_room": [("comfort", "space_light")],
            "big_windows": [("comfort", "space_light")],
            "room_small": [("comfort", "space_light")],
            "no_space_for_luggage": [("comfort", "space_light")],
            "dark_room": [("comfort", "space_light")],
            "no_natural_light": [("comfort", "space_light")],

            # --- tech_state / Техническое состояние и инфраструктура ---

            # plumbing_water
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

            # appliances_equipment
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

            # wifi_internet
            "wifi_fast": [("tech_state", "wifi_internet")],
            "internet_stable": [("tech_state", "wifi_internet")],
            "good_for_work": [("tech_state", "wifi_internet")],
            "wifi_down": [("tech_state", "wifi_internet")],
            "wifi_slow": [("tech_state", "wifi_internet")],
            "wifi_unstable": [("tech_state", "wifi_internet")],
            "wifi_hard_to_connect": [("tech_state", "wifi_internet")],
            "internet_not_suitable_for_work": [("tech_state", "wifi_internet")],

            # tech_noise
            "ac_noisy": [("tech_state", "tech_noise")],
            "fridge_noisy": [("tech_state", "tech_noise")],
            "pipes_noise": [("tech_state", "tech_noise")],
            "ventilation_noisy": [("tech_state", "tech_noise")],
            "night_mechanical_hum": [("tech_state", "tech_noise")],
            "tech_noise_sleep_issue": [("tech_state", "tech_noise")],
            "ac_quiet": [("tech_state", "tech_noise")],
            "fridge_quiet": [("tech_state", "tech_noise")],
            "no_tech_noise_night": [("tech_state", "tech_noise")],

            # elevator_infrastructure
            "elevator_working": [("tech_state", "elevator_infrastructure")],
            "luggage_easy": [("tech_state", "elevator_infrastructure")],
            "elevator_broken": [("tech_state", "elevator_infrastructure")],
            "elevator_stuck": [("tech_state", "elevator_infrastructure")],
            "no_elevator_heavy_bags": [("tech_state", "elevator_infrastructure")],

            # lock_security
            "door_secure": [("tech_state", "lock_security")],
            "felt_safe": [("tech_state", "lock_security")],
            "door_not_closing": [("tech_state", "lock_security")],
            "lock_broken": [("tech_state", "lock_security")],
            "felt_unsafe": [("tech_state", "lock_security")],

            # --- breakfast / Завтрак и питание ---

            # food_quality
            "breakfast_tasty": [("breakfast", "food_quality")],
            "food_fresh": [("breakfast", "food_quality")],
            "food_hot_served_hot": [("breakfast", "food_quality")],
            "coffee_good": [("breakfast", "food_quality")],
            "breakfast_bad_taste": [("breakfast", "food_quality")],
            "food_not_fresh": [("breakfast", "food_quality")],
            "food_cold": [("breakfast", "food_quality")],
            "coffee_bad": [("breakfast", "food_quality")],

            # variety_offering
            "breakfast_variety_good": [("breakfast", "variety_offering")],
            "buffet_rich": [("breakfast", "variety_offering")],
            "fresh_fruit_available": [("breakfast", "variety_offering")],
            "pastries_available": [("breakfast", "variety_offering")],
            "breakfast_variety_poor": [("breakfast", "variety_offering")],
            "breakfast_repetitive": [("breakfast", "variety_offering")],
            "hard_to_find_food": [("breakfast", "variety_offering")],

            # service_dining_staff
            "breakfast_staff_friendly": [("breakfast", "service_dining_staff")],
            "breakfast_staff_attentive": [("breakfast", "service_dining_staff")],
            "buffet_refilled_quickly": [("breakfast", "service_dining_staff")],
            "tables_cleared_fast": [("breakfast", "service_dining_staff")],
            "breakfast_staff_rude": [("breakfast", "service_dining_staff")],
            "no_refill_food": [("breakfast", "service_dining_staff")],
            "tables_left_dirty": [("breakfast", "service_dining_staff")],
            "ignored_requests": [("breakfast", "service_dining_staff")],

            # availability_flow
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

            # cleanliness_breakfast
            "breakfast_area_clean": [("breakfast", "cleanliness_breakfast")],
            "tables_cleaned_quickly": [("breakfast", "cleanliness_breakfast")],
            "dirty_tables": [("breakfast", "cleanliness_breakfast")],
            "dirty_dishes_left": [("breakfast", "cleanliness_breakfast")],
            "buffet_area_messy": [("breakfast", "cleanliness_breakfast")],


            # --- value / Цена и ценность ---

            # value_for_money
            "good_value": [("value", "value_for_money")],
            "worth_the_price": [("value", "value_for_money")],
            "affordable_for_level": [("value", "value_for_money")],
            "overpriced": [("value", "value_for_money")],
            "not_worth_price": [("value", "value_for_money")],
            "expected_better_for_price": [("value", "value_for_money")],

            # expectations_vs_price
            "photos_misleading": [("value", "expectations_vs_price")],
            "quality_below_expectation": [("value", "expectations_vs_price")],


            # --- location / Локация и окружение ---

            # proximity_area
            "great_location": [("location", "proximity_area")],
            "central_convenient": [("location", "proximity_area")],
            "near_transport": [("location", "proximity_area")],
            "area_has_food_shops": [("location", "proximity_area")],
            "location_inconvenient": [("location", "proximity_area")],
            "far_from_center": [("location", "proximity_area")],
            "nothing_around": [("location", "proximity_area")],

            # safety_environment
            "area_safe": [("location", "safety_environment")],
            "area_quiet_at_night": [("location", "safety_environment")],
            "entrance_clean": [("cleanliness", "public_areas"), ("location", "safety_environment")],
            "area_unsafe": [("location", "safety_environment")],
            "uncomfortable_at_night": [("location", "safety_environment")],
            "entrance_dirty": [("cleanliness", "public_areas"), ("location", "safety_environment")],
            "people_loitering": [("location", "safety_environment")],

            # access_navigation
            "easy_to_find": [("location", "access_navigation")],
            "clear_instructions": [("location", "access_navigation")],
            "luggage_access_ok": [("location", "access_navigation")],
            "hard_to_find_entrance": [("location", "access_navigation")],
            "confusing_access": [("location", "access_navigation")],
            "no_signage": [("location", "access_navigation")],
            "luggage_access_hard": [("location", "access_navigation")],


            # --- atmosphere / Атмосфера и общее впечатление ---

            # style_feel
            "cozy_atmosphere": [("atmosphere", "style_feel")],
            "nice_design": [("atmosphere", "style_feel")],
            "good_vibe": [("atmosphere", "style_feel")],
            "not_cozy": [("atmosphere", "style_feel")],
            "gloomy_feel": [("comfort", "space_light"), ("atmosphere", "style_feel")],
            "dated_look": [("atmosphere", "style_feel")],
            "soulless_feel": [("atmosphere", "style_feel")],

            # smell_common_areas
            "fresh_smell_common": [("atmosphere", "smell_common_areas")],
            "no_bad_smell": [("cleanliness", "smell"), ("atmosphere", "smell_common_areas")],
            "bad_smell_common": [("atmosphere", "smell_common_areas")],
            "cigarette_smell": [("atmosphere", "smell_common_areas")],
            "sewage_smell": [("cleanliness", "smell"), ("atmosphere", "smell_common_areas")],
            "musty_smell": [("cleanliness", "smell"), ("atmosphere", "smell_common_areas")],

        }

    ###########################################################################
    # 3. Публичные методы доступа
    ###########################################################################

    def get_version(self) -> str:
        """Версия словаря."""
        return self.version

    def get_sentiment_categories(self) -> List[str]:
        """Список ключей тональностей, например ['positive_strong', 'positive_soft', ...]."""
        return list(self.sentiment_lexicon.keys())

    def get_sentiment_patterns(self, sentiment_key: str, lang: str) -> List[str]:
        """
        Вернуть список regex-паттернов для заданной тональности и языка.
        sentiment_key: 'positive_strong' | 'negative_soft' | ...
        lang: 'ru' | 'en' | ...
        """
        pat = self.sentiment_lexicon.get(sentiment_key)
        if not pat:
            return []
        return pat.patterns_by_lang.get(lang, [])

    def iter_all_sentiment_patterns_for_lang(self, lang: str) -> Dict[str, List[str]]:
        """
        Удобно для таггера: получить {sentiment_key -> [regex...]} только для одного языка.
        """
        out: Dict[str, List[str]] = {}
        for skey, sdata in self.sentiment_lexicon.items():
            out[skey] = sdata.patterns_by_lang.get(lang, [])
        return out

    def get_categories(self) -> Iterable[str]:
        """Вернуть список ключей категорий (например 'staff_spir', 'cleanliness', ...)."""
        return self.topic_schema.keys()

    def get_category_display(self, category_key: str) -> Optional[str]:
        cat = self.topic_schema.get(category_key)
        return cat.display if cat else None

    def get_subtopics(self, category_key: str) -> Dict[str, Subtopic]:
        cat = self.topic_schema.get(category_key)
        return cat.subtopics if cat else {}

    def get_subtopic(self, category_key: str, subtopic_key: str) -> Optional[Subtopic]:
        cat = self.topic_schema.get(category_key)
        if not cat:
            return None
        return cat.subtopics.get(subtopic_key)

    def get_subtopic_patterns(self, category_key: str, subtopic_key: str, lang: str) -> List[str]:
        """
        Получить regex-паттерны для конкретной подтемы и языка.
        Это будет использоваться модулем topic_tagging.
        """
        st = self.get_subtopic(category_key, subtopic_key)
        if not st:
            return []
        return st.patterns_by_lang.get(lang, [])

    def get_subtopic_aspects(self, category_key: str, subtopic_key: str) -> List[str]:
        """
        Какие аспекты (aspect_code) может поднять данная подтема.
        """
        st = self.get_subtopic(category_key, subtopic_key)
        return st.aspects if st else []

    def get_aspect_meta(self, aspect_code: str) -> Optional[AspectMeta]:
        """Получить метаданные конкретного аспекта."""
        return self.aspects_meta.get(aspect_code)

    def aspect_display(self, aspect_code: str) -> str:
        """
        Короткое имя аспекта для буллетов и сводок.
        Если не знаем аспект — вернём сам код, чтобы не падать.
        """
        meta = self.get_aspect_meta(aspect_code)
        return meta.display_short if meta else aspect_code

    def aspect_hint(self, aspect_code: str) -> Optional[str]:
        """
        Более длинная подсказка/контекст аспекта.
        Полезно для генератора текста отчёта ('report_text_builder').
        """
        meta = self.get_aspect_meta(aspect_code)
        return meta.long_hint if meta else None
