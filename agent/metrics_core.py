# agent/metrics_core.py
# Библиотека: периоды, агрегаты (взвешенные), источники, подписи интервалов
import math
import datetime as dt
from datetime import date
import pandas as pd

# --- Константы листов Google Sheets (наши «хранилища»)
HISTORY_TAB       = "history"          # агрегаты по неделям/месяцам/кварталам/годам (сейчас используем недельные)
TOPICS_TAB        = "topics_history"   # темы по неделям
SOURCES_TAB       = "sources_history"  # НОВОЕ: по каждому источнику и неделе
SURVEYS_TAB       = "surveys_history"  # НОВОЕ: TL: Marketing (анкеты) — заполним позже

# --- Хелперы дат/периодов
RU_MONTHS_FULL  = ["январь","февраль","март","апрель","май","июнь","июль","август","сентябрь","октябрь","ноябрь","декабрь"]
RU_MONTHS_SHORT = ["янв","фев","мар","апр","май","июн","июл","авг","сен","окт","ноя","дек"]
RU_ROMAN_Q      = {1:"I",2:"II",3:"III",4:"IV"}

def iso_week_monday(week_key: str) -> date:
    """'YYYY-Wxx' → дата понедельника этой недели (ISO)."""
    y, w = week_key.split("-W")
    return date(int(y), 1, 4).fromisocalendar(int(y), int(w), 1)

def week_range_for_monday(monday: date) -> tuple[date, date]:
    """Понедельник → (пн, вс) этой недели."""
    return monday, monday + dt.timedelta(days=6)

def month_range_for_date(d: date) -> tuple[date, date]:
    start = d.replace(day=1)
    if d.month == 12:
        end = date(d.year, 12, 31)
    else:
        end = date(d.year, d.month + 1, 1) - dt.timedelta(days=1)
    return start, end

def quarter_of(d: date) -> int:
    return (d.month - 1)//3 + 1

def quarter_range_for_date(d: date) -> tuple[date, date]:
    q = quarter_of(d)
    start_month = (q - 1)*3 + 1
    start = date(d.year, start_month, 1)
    if q == 4:
        end = date(d.year, 12, 31)
    else:
        end = date(d.year, start_month + 3, 1) - dt.timedelta(days=1)
    return start, end

def ytd_range_for_date(d: date) -> tuple[date, date]:
    return date(d.year, 1, 1), date(d.year, 12, 31)

# --- Подписи интервалов (для письма/отчёта)
def week_label(d1: date, d2: date) -> str:
    """Например: 13–19 окт 2025 (если месяцы разные: 30 сен – 6 окт 2025)."""
    if d1.month == d2.month and d1.year == d2.year:
        return f"{d1.day}–{d2.day} {RU_MONTHS_SHORT[d1.month-1]} {d1.year}"
    else:
        return f"{d1.day} {RU_MONTHS_SHORT[d1.month-1]} – {d2.day} {RU_MONTHS_SHORT[d2.month-1]} {d2.year}"

def month_label(d: date) -> str:
    return f"{RU_MONTHS_FULL[d.month-1]} {d.year}"

def quarter_label(d: date) -> str:
    return f"{RU_ROMAN_Q[quarter_of(d)]} кв. {d.year}"

def year_label(d: date) -> str:
    return f"{d.year}"

# --- Безопасные касты и операции
def _num(x):
    """Надёжно приводит к float, понимает '9,05', пустые, тире и т.п."""
    try:
        if x is None:
            return float("nan")
        if isinstance(x, str):
            s = x.strip()
            if s in ("", "—", "-", "–"):
                return float("nan")
            s = s.replace(",", ".")
            return float(s)
        return float(x)
    except:
        return float("nan")

def _safe_pct(a: float, b: float) -> float | None:
    """Процент a/b*100, если b>0, иначе None."""
    if b and b > 0:
        return round(100.0 * a / b, 1)
    return None

def _weighted_avg(values: pd.Series, weights: pd.Series) -> float | None:
    """Взвешенная средняя, игнорируя NaN; если суммарный вес 0 → None."""
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    m = (~v.isna()) & (~w.isna()) & (w > 0)
    if not m.any():
        return None
    s = (v[m] * w[m]).sum()
    W = w[m].sum()
    if W <= 0:
        return None
    return round(float(s / W), 2)

# --- Базовые агрегаты неделя/периоды из листа history (используем недельные строки)
def aggregate_weeks_from_history(history_df: pd.DataFrame, start_d: date, end_d: date) -> dict:
    """
    Берёт лист history, фильтрует строки типа 'week' по диапазону (по понедельникам) и
    возвращает сводку: reviews, avg10 (взвеш.), pos, neu, neg, pos_share, neg_share.
    """
    empty = {"reviews": 0, "avg10": None, "pos": 0, "neu": 0, "neg": 0, "pos_share": None, "neg_share": None}
    if history_df is None or history_df.empty:
        return empty

    wk = history_df[history_df["period_type"] == "week"].copy()
    if wk.empty:
        return empty

    # числовые: приводим к float, понимаем запятую
    for c in ["reviews", "avg10", "pos", "neu", "neg"]:
        wk[c] = pd.to_numeric(wk[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")

    # фильтр по дате понедельника (из period_key 'YYYY-Wxx')
    def in_range(k):
        try:
            mon = iso_week_monday(str(k))
            return (mon >= start_d) and (mon <= end_d)
        except:
            return False

    wk = wk[wk["period_key"].apply(in_range)]
    if wk.empty:
        return empty

    n = int(wk["reviews"].sum())
    avg10 = _weighted_avg(wk["avg10"], wk["reviews"])
    pos = int(wk["pos"].sum())
    neu = int(wk["neu"].sum())
    neg = int(wk["neg"].sum())
    return {
        "reviews": n,
        "avg10": avg10,
        "pos": pos,
        "neu": neu,
        "neg": neg,
        "pos_share": _safe_pct(pos, n),
        "neg_share": _safe_pct(neg, n),
    }
def role_of_week_in_period(week_agg: dict, period_agg: dict) -> float | None:
    """Доля отзывов недели от агрегата периода, %."""
    if not week_agg or not period_agg: 
        return None
    return _safe_pct(week_agg.get("reviews",0), period_agg.get("reviews",0))

# --- Агрегаты по источникам из sources_history
def aggregate_sources_from_history(sources_df: pd.DataFrame, start_d: date, end_d: date) -> pd.DataFrame:
    """
    На вход df листа sources_history: week_key | source | reviews | avg10 | pos | neu | neg
    Возвращает df с метриками по каждому источнику за указанный диапазон дат.
    """
    cols = {"week_key", "source", "reviews", "avg10", "pos", "neu", "neg"}
    if sources_df is None or sources_df.empty or not cols.issubset(set(sources_df.columns)):
        return pd.DataFrame(columns=["source", "reviews", "avg10", "pos_share", "neg_share", "pos", "neu", "neg"])

    df = sources_df.copy()

    # типы и фильтр по датам
    df["mon"] = df["week_key"].map(lambda k: iso_week_monday(str(k)))
    df = df[(df["mon"] >= start_d) & (df["mon"] <= end_d)].copy()
    if df.empty:
        return pd.DataFrame(columns=["source", "reviews", "avg10", "pos_share", "neg_share", "pos", "neu", "neg"])

    for c in ["reviews", "avg10", "pos", "neu", "neg"]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")

    def agg_fn(g: pd.DataFrame):
        n = int(g["reviews"].sum())
        pos = int(g["pos"].sum())
        neu = int(g["neu"].sum())
        neg = int(g["neg"].sum())
        return pd.Series({
            "reviews": n,
            "avg10": _weighted_avg(g["avg10"], g["reviews"]),
            "pos": pos,
            "neu": neu,
            "neg": neg,
            "pos_share": _safe_pct(pos, n),
            "neg_share": _safe_pct(neg, n),
        })

    # group_keys=False — чтобы не было лишнего уровня индекса; include_groups=False — подавляет будущий DeprecationWarning (если поддерживается)
    try:
        out = df.groupby("source", group_keys=False).apply(agg_fn, include_groups=False).reset_index()
    except TypeError:
        # на старой pandas параметр include_groups отсутствует
        out = df.groupby("source", group_keys=False).apply(agg_fn).reset_index()

    return out.sort_values(["reviews", "avg10"], ascending=[False, False])

# --- Диапазоны «MTD/QTD/YTD» от недели + подписи
def period_ranges_for_week(week_start: date) -> dict:
    """Возвращает словарь с ranges и понятными подписями для Неделя/MTD/QTD/YTD."""
    w_start, w_end = week_start, week_start + dt.timedelta(days=6)

    m_start, m_end = month_range_for_date(week_start)
    q_start, q_end = quarter_range_for_date(week_start)
    y_start, y_end = ytd_range_for_date(week_start)

    return {
        "week":   {"start": w_start, "end": w_end, "label": week_label(w_start, w_end)},
        "mtd":    {"start": m_start, "end": w_end, "label": month_label(week_start)},   # MTD до конца недели
        "qtd":    {"start": q_start, "end": w_end, "label": quarter_label(week_start)}, # QTD до конца недели
        "ytd":    {"start": y_start, "end": w_end, "label": year_label(week_start)},    # YTD до конца недели
        # Полные периоды (календарные), пригодятся для YoY/Prev:
        "month":  {"start": m_start, "end": m_end, "label": month_label(week_start)},
        "quarter":{"start": q_start, "end": q_end, "label": quarter_label(week_start)},
        "year":   {"start": y_start, "end": y_end, "label": year_label(week_start)},
    }

def prev_period_ranges(week_start: date) -> dict:
    """Диапазоны «предыдущих» календарных периодов к текущему месяцу/кварталу/году (не MTD/QTD/YTD)."""
    # месяц назад
    first_this_month = week_start.replace(day=1)
    prev_month_end = first_this_month - dt.timedelta(days=1)
    pm_start, pm_end = month_range_for_date(prev_month_end)

    # предыдущий квартал
    q = quarter_of(week_start)
    if q == 1:
        pq_end = date(week_start.year - 1, 12, 31)
    else:
        pq_end = date(week_start.year, (q-1)*3, 1) - dt.timedelta(days=1)
    pq_start, pq_end = quarter_range_for_date(pq_end)

    # предыдущий год
    py_start, py_end = date(week_start.year - 1, 1, 1), date(week_start.year - 1, 12, 31)

    return {
        "prev_month":   {"start": pm_start, "end": pm_end, "label": month_label(pm_end)},
        "prev_quarter": {"start": pq_start, "end": pq_end, "label": quarter_label(pq_end)},
        "prev_year":    {"start": py_start, "end": py_end, "label": str(py_end.year)},
    }

# --- Дельты и вклад недели
def deltas_week_vs_period(week_agg: dict, period_agg: dict) -> dict:
    """Разница «неделя – период» по средней /10 и по долям позитив/негатив (п.п.)."""
    def dd(a, b):
        if a is None or b is None: return None
        return round(float(a) - float(b), 2)
    week_pos = None if not week_agg.get("reviews") else 100.0 * week_agg["pos"]/week_agg["reviews"]
    week_neg = None if not week_agg.get("reviews") else 100.0 * week_agg["neg"]/week_agg["reviews"]
    return {
        "avg10_delta": dd(week_agg.get("avg10"), period_agg.get("avg10")),
        "pos_delta_pp": dd(week_pos, period_agg.get("pos_share")),
        "neg_delta_pp": dd(week_neg, period_agg.get("neg_share")),
        "week_share_pct": role_of_week_in_period(week_agg, period_agg),  # вклад недели в период, %
    }

# --- Таблица по источникам (набор данных, без HTML)
def sources_summary_for_periods(sources_df: pd.DataFrame, week_start: date) -> dict:
    """
    Возвращает словарь с 4 сводками по источникам:
    - week / mtd / qtd / ytd → DataFrame: source | reviews | avg10 | pos_share | neg_share | pos | neu | neg
    и аналогичные «prev_» (prev_month, prev_quarter, prev_year), если захотим сравнивать.
    """
    ranges = period_ranges_for_week(week_start)
    prev    = prev_period_ranges(week_start)

    def run(r):
        return aggregate_sources_from_history(sources_df, r["start"], r["end"])

    out = {
        "week": run(ranges["week"]),
        "mtd":  run(ranges["mtd"]),
        "qtd":  run(ranges["qtd"]),
        "ytd":  run(ranges["ytd"]),
        "prev_month":   run(prev["prev_month"]),
        "prev_quarter": run(prev["prev_quarter"]),
        "prev_year":    run(prev["prev_year"]),
        "labels": {
            "week":  ranges["week"]["label"],
            "mtd":   ranges["mtd"]["label"],
            "qtd":   ranges["qtd"]["label"],
            "ytd":   ranges["ytd"]["label"],
            "prev_month":   prev["prev_month"]["label"],
            "prev_quarter": prev["prev_quarter"]["label"],
            "prev_year":    prev["prev_year"]["label"],
        }
    }
    return out
