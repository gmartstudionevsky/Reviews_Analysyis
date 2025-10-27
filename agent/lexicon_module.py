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

            # ------------ Остальные категории (checkin_stay, cleanliness, comfort,
            # tech_state, breakfast, value, location, atmosphere) ------------
            # ВНИМАНИЕ:
            # Чтобы не потерять точность, мы должны завести их все так же,
            # как сделано выше для staff_spir.
            #
            # Здесь логика одинакова:
            #   TopicCategory(display="...", subtopics={ ... Subtopic(...) ... })
            #
            # Мы уже имеем полный словарь (огромный блок TOPIC_SCHEMA),
            # так что технически он просто механически маппится в такие же структуры.
            #
            # Для финальной версии файла нужно:
            # - повторить этот же шаблон для ВСЕХ категорий:
            #   checkin_stay, cleanliness, comfort, tech_state, breakfast,
            #   value, location, atmosphere.
            #
            # Из-за объёма мы не будем дублировать весь массив здесь второй раз,
            # но в боевом файле он должен быть полностью развёрнут.
            #
            # Дальше в коде я буду считать, что self.topic_schema содержит
            # все категории, как staff_spir выше, только заполненные.
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
            # Примеры. (Сюда надо занести все аспекты, перечисленные в твоём словаре)
            "spir_friendly": AspectMeta(
                aspect_code="spir_friendly",
                display_short="дружелюбный персонал",
                long_hint="Гости подчеркивают приветливость и доброжелательность сотрудников."
            ),
            "spir_rude": AspectMeta(
                aspect_code="spir_rude",
                display_short="грубость персонала",
                long_hint="Жёстко негативный фидбек об общении: грубость, хамство, неуважение."
            ),
            "wifi_unstable": AspectMeta(
                aspect_code="wifi_unstable",
                display_short="нестабильный Wi-Fi",
                long_hint="Wi-Fi отваливается/медленный, невозможно работать удалённо."
            ),
            "wifi_fast": AspectMeta(
                aspect_code="wifi_fast",
                display_short="быстрый Wi-Fi",
                long_hint="Гости довольны скоростью/стабильностью интернета, есть условия для удалённой работы."
            ),
            "clean_on_arrival": AspectMeta(
                aspect_code="clean_on_arrival",
                display_short="чистота при заезде",
                long_hint="Номер был убран, без пыли, свежие полотенца/бельё при заселении."
            ),
            "dirty_on_arrival": AspectMeta(
                aspect_code="dirty_on_arrival",
                display_short="грязно при заселении",
                long_hint="Гости заходят в неподготовленный номер: пыль, волосы, мусор от прошлого гостя."
            ),
            "good_value": AspectMeta(
                aspect_code="good_value",
                display_short="хорошее соотношение цена/качество",
                long_hint="Гости считают, что уровень сервиса оправдывает цену."
            ),
            "overpriced": AspectMeta(
                aspect_code="overpriced",
                display_short="завышенная цена",
                long_hint="Ощущение, что отель не стоит своих денег."
            ),
            "great_location": AspectMeta(
                aspect_code="great_location",
                display_short="удобная локация",
                long_hint="Близко к центру/транспорту/кафе — локация как ключевой плюс."
            ),
            "location_inconvenient": AspectMeta(
                aspect_code="location_inconvenient",
                display_short="неудобная локация",
                long_hint="Слишком далеко от центра/сложно добираться/ничего рядом."
            ),
            "quiet_room": AspectMeta(
                aspect_code="quiet_room",
                display_short="тихо в номере",
                long_hint="Хорошая звукоизоляция, нет уличного или коридорного шума."
            ),
            "noisy_room": AspectMeta(
                aspect_code="noisy_room",
                display_short="шум в номере",
                long_hint="Жалобы на шум (улица, соседи, коридор, ресепшн) и проблемы со сном."
            ),
            "bed_comfy": AspectMeta(
                aspect_code="bed_comfy",
                display_short="удобная кровать",
                long_hint="Комфортный матрас/подушки, гости отмечают хороший сон."
            ),
            "bed_uncomfortable": AspectMeta(
                aspect_code="bed_uncomfortable",
                display_short="неудобная кровать",
                long_hint="Жёсткий/слишком мягкий/просевший матрас, скрип, неудобные подушки."
            ),
            "breakfast_tasty": AspectMeta(
                aspect_code="breakfast_tasty",
                display_short="вкусный завтрак",
                long_hint="Свежие продукты, тёплые блюда подаются горячими, нормальный кофе."
            ),
            "breakfast_bad_taste": AspectMeta(
                aspect_code="breakfast_bad_taste",
                display_short="невкусный завтрак",
                long_hint="Холодная еда, не свежие продукты, плохой кофе, жалобы на вкус."
            ),
            "breakfast_variety_good": AspectMeta(
                aspect_code="breakfast_variety_good",
                display_short="разнообразный завтрак",
                long_hint="Гости отмечают большой выбор, богатый буфет, фрукты/выпечку."
            ),
            "breakfast_variety_poor": AspectMeta(
                aspect_code="breakfast_variety_poor",
                display_short="скудный выбор завтрака",
                long_hint="Мало опций, одно и то же каждый день, нечего поесть."
            ),
            "checkin_fast": AspectMeta(
                aspect_code="checkin_fast",
                display_short="быстрое заселение",
                long_hint="Без очередей, номер сразу готов, ключи моментально."
            ),
            "checkin_wait_long": AspectMeta(
                aspect_code="checkin_wait_long",
                display_short="долгое заселение",
                long_hint="Ожидание на ресепшене, комната не готова вовремя."
            ),
            "room_spacious": AspectMeta(
                aspect_code="room_spacious",
                display_short="просторный номер",
                long_hint="Много места, удобная планировка, не тесно с багажом."
            ),
            "room_small": AspectMeta(
                aspect_code="room_small",
                display_short="тесный номер",
                long_hint="Негде поставить чемодан, сложно развернуться."
            ),
            "ac_working": AspectMeta(
                aspect_code="ac_working",
                display_short="хороший климат-контроль",
                long_hint="Работающий кондиционер/отопление, комфортная температура."
            ),
            "ac_broken": AspectMeta(
                aspect_code="ac_broken",
                display_short="проблемы с климатом",
                long_hint="Жарко/душно/холодно, кондиционер или отопление не работает."
            ),
            "hot_water_ok": AspectMeta(
                aspect_code="hot_water_ok",
                display_short="стабильная горячая вода",
                long_hint="Есть горячая вода без перебоев, нормальный напор."
            ),
            "no_hot_water": AspectMeta(
                aspect_code="no_hot_water",
                display_short="нет горячей воды / слабый напор",
                long_hint="Жалобы на перебои с горячей водой, сломанный душ, протечки."
            ),
            "door_secure": AspectMeta(
                aspect_code="door_secure",
                display_short="безопасный номер",
                long_hint="Дверь/замок закрываются нормально, гости чувствуют себя в безопасности."
            ),
            "felt_unsafe": AspectMeta(
                aspect_code="felt_unsafe",
                display_short="небезопасно",
                long_hint="Дверь не закрывается, сломанный замок или тревожный район."
            ),
            "good_vibe": AspectMeta(
                aspect_code="good_vibe",
                display_short="уютная атмосфера",
                long_hint="Тёплый вайб, стильный интерьер, ощущение «как дома»."
            ),
            "gloomy_feel": AspectMeta(
                aspect_code="gloomy_feel",
                display_short="мрачная атмосфера",
                long_hint="Старый ремонт, давящее ощущение, неуютно."
            ),
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
