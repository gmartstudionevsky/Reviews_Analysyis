from __future__ import annotations

"""
metrics_core.py

Общие утилиты для расчёта периодов и метрик, используемые как модулем
анкет (surveys_*), так и будущими отчётами по отзывам (reviews_*).

⚠️ Важно: этот модуль спроектирован БЕЗ изменения действующего API,
от которого уже зависят surveys_* файлы. Все новые возможности
добавлены аддитивно.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Tuple, Optional, Iterable, Any

import pandas as pd
import numpy as np

# -----------------------------
# Русские подписи для месяцев
# -----------------------------
RU_MONTH_SHORT = {
    1: "янв",
    2: "фев",
    3: "мар",
    4: "апр",
    5: "май",
    6: "июн",
    7: "июл",
    8: "авг",
    9: "сен",
    10: "окт",
    11: "ноя",
    12: "дек",
}
RU_MONTH_FULL = {
    1: "январь",
    2: "февраль",
    3: "март",
    4: "апрель",
    5: "май",
    6: "июнь",
    7: "июль",
    8: "август",
    9: "сентябрь",
    10: "октябрь",
    11: "ноябрь",
    12: "декабрь",
}

_ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV"}


# -----------------------------
# Вспомогательные функции дат
# -----------------------------
def _last_day_of_month(d: date) -> date:
    nm = d.month + 1
    ny = d.year + (1 if nm == 13 else 0)
    nm = 1 if nm == 13 else nm
    first_next = date(ny, nm, 1)
    return first_next - timedelta(days=1)


def _quarter_start(d: date) -> date:
    q = (d.month - 1) // 3 + 1
    start_month = 3 * (q - 1) + 1
    return date(d.year, start_month, 1)


def _quarter_end(d: date) -> date:
    qs = _quarter_start(d)
    # конец квартала = 3 месяца от начала минус 1 день
    nm = qs.month + 3
    ny = qs.year + (1 if nm > 12 else 0)
    nm = nm - 12 if nm > 12 else nm
    first_after = date(ny, nm, 1)
    return first_after - timedelta(days=1)


def iso_week_monday(week_key: str) -> date:
    """
    Преобразует строку ISO недели вида 'YYYY-W##' в дату понедельника.
    Примеры допустимых форматов: '2025-W1', '2025-W01', '2025-W52'.

    :param week_key: ISO-неделя как строка.
    :return: дата (понедельник) начала недели.
    """
    week_key = week_key.strip()
    # Ожидаем 'YYYY-W##'
    try:
        yr_part, w_part = week_key.split("-W")
        y = int(yr_part)
        w = int(w_part)
        return date.fromisocalendar(y, w, 1)  # Monday
    except Exception as e:
        raise ValueError(f"iso_week_monday: неверный формат week_key='{week_key}'. Ожидалось 'YYYY-W##'.") from e


def week_label(start: date, end: date, ref_year: Optional[int] = None) -> str:
    """
    Возвращает короткую подпись недели: '13–19 окт 2025' или '27 янв – 2 фев 2025',
    если неделя пересекает месяцы. Год добавляется, если он отличается от ref_year
    (или ref_year не указан) либо неделя пересекает границу года.

    :param start: дата понедельника
    :param end: дата воскресенья
    :param ref_year: Год-эталон (обычно текущий год отчёта). Если совпадает со start.year и end.year,
                     подпись может не включать год.
    """
    same_month = start.year == end.year and start.month == end.month
    show_year = (ref_year is None) or (start.year != ref_year) or (start.year != end.year)

    if same_month:
        core = f"{start.day}–{end.day} {RU_MONTH_SHORT[start.month]}"
        if show_year:
            core += f" {end.year}"
        return core

    # пересечение месяцев/лет
    core = f"{start.day} {RU_MONTH_SHORT[start.month]} – {end.day} {RU_MONTH_SHORT[end.month]}"
    if show_year:
        # если разные годы, лучше явно указать год конца
        core += f" {end.year}"
    return core


def month_label(d: date) -> str:
    """'октябрь 2025' (в нижнем регистре для единообразия; обёртки могут капитализировать)."""
    return f"{RU_MONTH_FULL[d.month]} {d.year}"


def quarter_label(d: date) -> str:
    """'IV кв. 2025'."""
    q = (d.month - 1) // 3 + 1
    return f"{_ROMAN[q]} кв. {d.year}"


def year_label(d: date) -> str:
    """'2025'."""
    return str(d.year)


def period_ranges_for_week(week_monday: date) -> Dict[str, Dict[str, date]]:
    """
    Возвращает границы периодов для недельного отчёта:
    - 'week': [понедельник; воскресенье]
    - 'mtd' : [1 число месяца; min(конец месяца, воскресенье недели)]
    - 'qtd' : [начало квартала; min(конец квартала, воскресенье недели)]
    - 'ytd' : [01.01; min(31.12, воскресенье недели)]

    Эти границы ожидаются surveys_weekly_report_agent.
    """
    w_start = week_monday
    w_end = week_monday + timedelta(days=6)

    # MTD
    m_start = date(w_start.year, w_start.month, 1)
    m_end = min(_last_day_of_month(w_start), w_end)

    # QTD
    q_start = _quarter_start(w_start)
    q_end = min(_quarter_end(w_start), w_end)

    # YTD
    y_start = date(w_start.year, 1, 1)
    y_end = min(date(w_start.year, 12, 31), w_end)

    return {
        "week": {"start": w_start, "end": w_end},
        "mtd": {"start": m_start, "end": m_end},
        "qtd": {"start": q_start, "end": q_end},
        "ytd": {"start": y_start, "end": y_end},
    }


# =============================================================================
# Ниже — функции агрегации «истории» отзывов (не используются анкетами напрямую),
# оставлены ради совместимости и будущего расширения.
# =============================================================================

def _require_columns(df: pd.DataFrame, cols: Iterable[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{where}: отсутствуют обязательные колонки: {missing}")


def build_history(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Превращает сырые отзывы в «историю» по неделям.
    Ожидаемые колонки raw_df: ['review_id','source','created_at','week_key','rating10','sentiment_overall'].
    Необязательные: ['language','topic','aspect'].

    Возвращает DataFrame с колонками:
      ['week_key','week_start','week_end','reviews','avg10','pos','neu','neg']
    """
    if raw_df is None or len(raw_df) == 0:
        return pd.DataFrame(columns=[
            "week_key","week_start","week_end","reviews","avg10","pos","neu","neg"
        ])

    df = raw_df.copy()

    # week_key — обязателен для агрегирования по неделям
    _require_columns(df, ["week_key", "rating10", "sentiment_overall"], "build_history")

    # Нормализуем week_start/week_end для удобства последующего мерджа
    df["week_start"] = df["week_key"].astype(str).map(iso_week_monday)
    df["week_end"] = df["week_start"] + pd.to_timedelta(6, unit="D")

    # Сентимент -> one-hot
    def _one_hot_sentiment(s: Any) -> Tuple[int, int, int]:
        s = str(s).lower()
        if s.startswith("pos"):
            return 1, 0, 0
        if s.startswith("neg"):
            return 0, 0, 1
        return 0, 1, 0

    oh = df["sentiment_overall"].map(_one_hot_sentiment)
    df[["pos","neu","neg"]] = pd.DataFrame(list(oh), index=df.index)

    grp = (
        df.groupby(["week_key","week_start","week_end"], as_index=False)
          .agg(
              reviews=("review_id", "nunique") if "review_id" in df.columns else ("week_key","size"),
              avg10=("rating10", "mean"),
              pos=("pos", "sum"),
              neu=("neu", "sum"),
              neg=("neg", "sum"),
          )
    )
    # Безопасно округлим «красиво», оставляя float
    grp["avg10"] = grp["avg10"].astype(float).round(2)
    return grp.sort_values("week_start").reset_index(drop=True)


def build_sources_history(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """
    История по источникам (week_key × source).
    Ожидаемые колонки: как в build_history + 'source'.

    Возвращает DataFrame с колонками:
      ['week_key','week_start','week_end','source','reviews','avg10','pos','neu','neg']
    """
    if reviews_df is None or len(reviews_df) == 0:
        return pd.DataFrame(columns=[
            "week_key","week_start","week_end","source","reviews","avg10","pos","neu","neg"
        ])

    df = reviews_df.copy()
    _require_columns(df, ["week_key", "rating10", "sentiment_overall", "source"], "build_sources_history")

    df["week_start"] = df["week_key"].astype(str).map(iso_week_monday)
    df["week_end"] = df["week_start"] + pd.to_timedelta(6, unit="D")

    def _one_hot_sentiment(s: Any) -> Tuple[int, int, int]:
        s = str(s).lower()
        if s.startswith("pos"):
            return 1, 0, 0
        if s.startswith("neg"):
            return 0, 0, 1
        return 0, 1, 0

    oh = df["sentiment_overall"].map(_one_hot_sentiment)
    df[["pos","neu","neg"]] = pd.DataFrame(list(oh), index=df.index)

    grp = (
        df.groupby(["week_key","week_start","week_end","source"], as_index=False)
          .agg(
              reviews=("review_id", "nunique") if "review_id" in df.columns else ("week_key","size"),
              avg10=("rating10", "mean"),
              pos=("pos", "sum"),
              neu=("neu", "sum"),
              neg=("neg", "sum"),
          )
    )
    grp["avg10"] = grp["avg10"].astype(float).round(2)
    return grp.sort_values(["week_start","source"]).reset_index(drop=True)


# Экспортируем публичный API, на который опираются другие модули.
__all__ = [
    # даты/подписи (используются анкетами)
    "iso_week_monday",
    "week_label",
    "month_label",
    "quarter_label",
    "year_label",
    "period_ranges_for_week",
    # будущие и совместимые функции для отзывов
    "build_history",
    "build_sources_history",
]
