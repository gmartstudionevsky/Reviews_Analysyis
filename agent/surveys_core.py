# agent/surveys_core.py
# Аналитика анкет TL: Marketing
# - Жёсткая мапа колонок -> тех.поля
# - Шкала только 1..5
# - Разделение "сколько анкет всего" vs "сколько ответили на вопрос"
# - NPS: 1-2 Detractors / 3-4 Neutral / 5 Promoters

from __future__ import annotations

import re
import math
import datetime as dt
from datetime import date
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# =====================================================
# Каноническая схема колонок исходной выгрузки анкеты
# =====================================================

SURVEYS_TAB = "surveys_history"

# Мапим ЧЁТКИЕ заголовки из листа "Оценки гостей" -> наши техназвания.
COLUMN_MAP_SCORE: Dict[str, str] = {
    "Средняя оценка гостя": "overall",

    "№ 1.1 Оцените работу службы приёма и размещения при заезде": (
        "spir_checkin"
    ),
    "№ 1.2 Оцените чистоту номера при заезде": "clean_checkin",
    "№ 1.3 Оцените комфорт и оснащение номера": "comfort",

    "№ 2.1 Оцените работу службы приёма и размещения во время проживания": (
        "spir_stay"
    ),
    "№ 2.2 Оцените работу технической службы": "tech_service",
    "№ 2.3 Оцените уборку номера во время проживания": "housekeeping",
    "№ 2.4 Оцените завтраки": "breakfast",

    "№ 3.1 Оцените атмосферу в отеле": "atmosphere",
    "№ 3.2 Оцените расположение отеля": "location",
    "№ 3.3 Оцените соотношение цены и качества": "value",
    "№ 3.4 Хотели бы вы вернуться в ARTSTUDIO Nevsky?": "return_intent",

    # Шкала 1..5 для оценки готовности рекомендовать (наш NPS-вопрос)
    "№ 3.5 Оцените вероятность того, что вы порекомендуете нас друзьям и близким (по шкале от 1 до 5)": "nps",
}

# Служебные/мета-колонки (не идут в агрегацию оценок как метрики):
COLUMN_MAP_META: Dict[str, str] = {
    "Дата анкетирования": "date",
    "Комментарий гостя": "comment",
    "ФИО": "fio",
    "Номер брони": "booking",
    "Телефон": "phone",
    "Email": "email",
}

# Порядок метрик, как хотим видеть их в отчётах / письме.
PARAM_ORDER: List[str] = [
    "overall",
    "spir_checkin", "clean_checkin", "comfort",
    "spir_stay", "tech_service", "housekeeping", "breakfast",
    "atmosphere", "location", "value", "return_intent",
    "nps",
]


# =====================================================
# Хелперы
# =====================================================

def _norm_header(h: str) -> str:
    """
    Приводим заголовок столбца к унифицированному виду, чтобы поймать
    случайные двойные пробелы, неразрывные пробелы и т.п.
    """
    s = str(h).replace("\u00a0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _parse_date_cell(x) -> Optional[date]:
    """
    '13.10.2025', '2025-10-13', datetime -> date
    """
    try:
        d = pd.to_datetime(x, errors="coerce", dayfirst=True)
        if pd.isna(d):
            return None
        return d.date()
    except Exception:
        return None

def _parse_score_1to5(x) -> float:
    """
    Строго шкала 1..5.
    Понимаем варианты типа '4', '4,0', '5 из 5'.
    Всё, что не 1..5 -> np.nan.
    """
    if x is None:
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan

    # заменяем запятую на точку и вытаскиваем первое число
    s = s.replace(",", ".")
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if not m:
        return np.nan

    try:
        v = float(m.group(1))
    except Exception:
        return np.nan

    # допускаем дроби (4.5 и т.п.), затем клип до диапазона
    if v < 1 or v > 5:
        return np.nan
    return float(v)

def _iso_week_key(d: date) -> str:
    """
    d -> 'YYYY-W##' по ISO-неделе (ISO Monday-based).
    """
    iso = d.isocalendar()
    # iso.week уже ISO-неделя, iso.year - ISO-год
    return f"{iso.year}-W{iso.week:02d}"


# =====================================================
# NPS
# =====================================================

def compute_nps_from_1to5(series: pd.Series) -> dict:
    """
    На вход: pd.Series чисел 1..5 (уже почищенных).
    NPS-правило (твоя бизнес-логика):
      1-2  -> детракторы
      3-4  -> нейтралы (не считаются ни туда ни сюда)
      5    -> промоутеры
      пусто/0 -> не учитываем
    Возвращает dict:
      {
        "promoters": int,
        "detractors": int,
        "nps_answers": int,
        "nps_value": float | None,  # в п.п.
      }
    """
    # оставляем только валидные 1..5
    v = pd.to_numeric(series, errors="coerce")
    v = v.where(v.between(1, 5))
    v = v.dropna()

    total = int(len(v))
    if total == 0:
        return {
            "promoters": 0,
            "detractors": 0,
            "nps_answers": 0,
            "nps_value": None,
        }

    detractors = int(((v == 1) | (v == 2)).sum())
    promoters  = int((v == 5).sum())

    nps_val = ((promoters / total) - (detractors / total)) * 100.0

    return {
        "promoters": promoters,
        "detractors": detractors,
        "nps_answers": total,
        "nps_value": round(float(nps_val), 2),
    }


# =====================================================
# 1. Нормализация сырых данных анкеты
# =====================================================

def normalize_surveys_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Принимает DataFrame из листа "Оценки гостей" (или исторического листа "Reviews")
    и возвращает нормализованный DataFrame, где:
      - каждая строка = одна анкета
      - колонки:
          date (date)
          surveys_total (=1 для каждой строки)
          fio, booking, phone, email, comment
          overall, spir_checkin, ..., nps  (float 1..5 или NaN)
    Никаких пересчётов шкал, только 1..5.
    """

    # 1. Построим карту {техполе -> реальное имя столбца в df_raw}
    colmap: Dict[str, str] = {}  # canonical_name -> df_raw_column
    inv_lookup = { _norm_header(k): v for k,v in COLUMN_MAP_SCORE.items() }
    inv_meta   = { _norm_header(k): v for k,v in COLUMN_MAP_META.items() }

    # сделаем нормализованный -> исходный
    raw_cols_norm = { _norm_header(c): c for c in df_raw.columns }

    # мета-поля
    for norm_header, canon_name in inv_meta.items():
        if norm_header in raw_cols_norm:
            colmap[canon_name] = raw_cols_norm[norm_header]

    # оценочные поля 1..5
    for norm_header, canon_name in inv_lookup.items():
        if norm_header in raw_cols_norm:
            colmap[canon_name] = raw_cols_norm[norm_header]

    # 2. Собираем выходной df
    out = pd.DataFrame()

    # дата анкетирования
    if "date" not in colmap:
        raise RuntimeError("Не найдена колонка 'Дата анкетирования' в анкете.")
    out["date"] = df_raw[colmap["date"]].map(_parse_date_cell)

    # базовые метаданные
    for meta_field in ("fio","booking","phone","email","comment"):
        if meta_field in colmap:
            out[meta_field] = df_raw[colmap[meta_field]].astype(str)
        else:
            out[meta_field] = ""

    # оценки 1..5
    for param in PARAM_ORDER:
        if param == "nps":
            # NPS мы тоже парсим как число 1..5 — так же, как остальные
            src_col = colmap.get("nps")
            out["nps"] = df_raw[src_col].map(_parse_score_1to5) if src_col else np.nan
        else:
            src_col = colmap.get(param)
            out[param] = df_raw[src_col].map(_parse_score_1to5) if src_col else np.nan

    # каждая строка = одна анкета
    out["surveys_total"] = 1

    # убираем строки без даты вообще
    out = out[pd.notna(out["date"])].reset_index(drop=True)

    return out


# =====================================================
# 2. Недельная агрегация
# =====================================================

def weekly_aggregate(df_norm: pd.DataFrame) -> pd.DataFrame:
    """
    Из нормализованного df строим недельные агрегаты.
    Возвращает DataFrame с колонками:
        week_key        'YYYY-W##'
        param           см. PARAM_ORDER
        surveys_total   сколько анкет в этой неделе (все строки)
        answered        сколько ответило по этому параметру (не NaN)
        avg5            средняя по шкале /5 среди ответивших
        avg10           avg5 * 2
        promoters       только для param == 'nps'
        detractors      только для param == 'nps'
        nps_answers     только для param == 'nps'
        nps_value       только для param == 'nps'
    """

    if df_norm.empty:
        return pd.DataFrame(columns=[
            "week_key","param","surveys_total","answered",
            "avg5","avg10",
            "promoters","detractors","nps_answers","nps_value",
        ])

    df = df_norm.copy()
    df["week_key"] = df["date"].map(_iso_week_key)

    rows = []

    for wk, wdf in df.groupby("week_key"):
        week_total = int(len(wdf))

        for param in PARAM_ORDER:
            vals = pd.to_numeric(wdf[param], errors="coerce")
            vals = vals.where(vals.between(1,5))
            answered = int(vals.notna().sum())

            if answered > 0:
                avg5_val = round(float(vals.mean()), 2)
                avg10_val = round(float(avg5_val * 2.0), 2)
            else:
                avg5_val  = None
                avg10_val = None

            # базовые поля
            row = {
                "week_key": wk,
                "param": param,
                "surveys_total": week_total,
                "answered": answered,
                "avg5": avg5_val,
                "avg10": avg10_val,
                "promoters": None,
                "detractors": None,
                "nps_answers": None,
                "nps_value": None,
            }

            # если это NPS — считаем деталку
            if param == "nps":
                nps_stats = compute_nps_from_1to5(vals)
                row["promoters"]   = nps_stats["promoters"]
                row["detractors"]  = nps_stats["detractors"]
                row["nps_answers"] = nps_stats["nps_answers"]
                row["nps_value"]   = nps_stats["nps_value"]

            rows.append(row)

    out = pd.DataFrame(rows, columns=[
        "week_key","param","surveys_total","answered",
        "avg5","avg10",
        "promoters","detractors","nps_answers","nps_value",
    ])

    # фиксируем порядок строк по неделе и по логическому списку параметров
    out["param"] = pd.Categorical(out["param"], categories=PARAM_ORDER, ordered=True)
    out = out.sort_values(["week_key","param"]).reset_index(drop=True)
    return out


# =====================================================
# 3. Фасад для агентов
# =====================================================

def parse_and_aggregate_weekly(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Удобный фасад: нормализуем + считаем недельные метрики одной функцией.
    Возвращаем:
        df_norm (поанкетно, одна строка = одна анкета)
        agg_week (недельные строки для записи в surveys_history)
    """
    norm = normalize_surveys_df(df_raw)
    agg  = weekly_aggregate(norm)
    return norm, agg
