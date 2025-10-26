# agent/surveys_core.py
#
# Ядро обработки анкет TL: Marketing.
# Что делает:
#   1. Приводит выгрузку ("Оценки гостей" или исторический "Reviews") к нормальному табличному виду
#      - одна строка = одна анкета гостя
#      - оценки строго по шкале 1..5
#   2. Сводит по неделям:
#      - считает средние по /5
#      - считает NPS (1–2 детракторы, 3–4 нейтралы, 5 промоутеры)
#      - НЕ считает больше avg10
#
# Результат недельной агрегации — то, что мы пишем в лист surveys_history
# (одна строка = один параметр за неделю)

from __future__ import annotations

import re
import math
import datetime as dt
from datetime import date
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# =====================================================
# Куда пишем историю анкет
# =====================================================
SURVEYS_TAB = "surveys_history"


# =====================================================
# Каноническая схема колонок исходной анкеты
# =====================================================

# Жёстко мапим заголовки из выгрузки -> наши тех.поля.
# Это важно, чтобы не было магии по регуляркам.
COLUMN_MAP_SCORE: Dict[str, str] = {
    "Средняя оценка гостя": "overall",

    "№ 1.1 Оцените работу службы приёма и размещения при заезде": "spir_checkin",
    "№ 1.2 Оцените чистоту номера при заезде": "clean_checkin",
    "№ 1.3 Оцените комфорт и оснащение номера": "comfort",

    "№ 2.1 Оцените работу службы приёма и размещения во время проживания": "spir_stay",
    "№ 2.2 Оцените работу технической службы": "tech_service",
    "№ 2.3 Оцените уборку номера во время проживания": "housekeeping",
    "№ 2.4 Оцените завтраки": "breakfast",

    "№ 3.1 Оцените атмосферу в отеле": "atmosphere",
    "№ 3.2 Оцените расположение отеля": "location",
    "№ 3.3 Оцените соотношение цены и качества": "value",
    "№ 3.4 Хотели бы вы вернуться в ARTSTUDIO Nevsky?": "return_intent",

    # NPS-вопрос (шкала 1..5)
    "№ 3.5 Оцените вероятность того, что вы порекомендуете нас друзьям и близким (по шкале от 1 до 5)": "nps",
}

# Служебные/метаданные анкеты
COLUMN_MAP_META: Dict[str, str] = {
    "Дата анкетирования": "date",
    "Комментарий гостя": "comment",
    "ФИО": "fio",
    "Номер брони": "booking",
    "Телефон": "phone",
    "Email": "email",
}

# Порядок параметров в отчётах / графиках
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
    Нормализуем заголовок: убираем неразрывные пробелы, лишние пробелы.
    Это помогает, если выгрузка слегка «дрожит» по формату.
    """
    s = str(h).replace("\u00a0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _parse_date_cell(x) -> Optional[date]:
    """
    Пробуем распарсить дату анкетирования.
    Поддерживает форматы типа '13.10.2025', '2025-10-13', datetime.
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
    Берём значение по шкале 1..5.
    Понимаем '4', '4,0', '5 из 5', и т.д.
    Всё, что не попадает в диапазон [1;5], превращаем в NaN.
    """
    if x is None:
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan

    s = s.replace(",", ".")
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if not m:
        return np.nan

    try:
        v = float(m.group(1))
    except Exception:
        return np.nan

    if v < 1 or v > 5:
        return np.nan
    return float(v)

def _iso_week_key(d: date) -> str:
    """
    Преобразуем дату в ключ недели формата 'YYYY-W##' по ISO-неделе,
    где неделя начинается с понедельника.
    """
    iso = d.isocalendar()  # (iso_year, iso_week, iso_weekday)
    return f"{iso.year}-W{iso.week:02d}"


# =====================================================
# NPS
# =====================================================

def compute_nps_from_1to5(series: pd.Series) -> dict:
    """
    Считаем NPS по правилу:
      1–2  -> детракторы
      3–4  -> нейтралы
      5    -> промоутеры
      пусто/0 -> исключаем из расчёта
    Возвращаем словарь с детализацией.
    """
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
# 1. Нормализация сырых анкет
# =====================================================

def normalize_surveys_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Приводим "Оценки гостей" (или исторический лист "Reviews") к унифицированному df:
      - каждая строка = одна анкета гостя
      - date, fio, booking, phone, email, comment
      - для каждого параметра (overall, spir_checkin, ... nps) колонка float 1..5 или NaN
      - surveys_total = 1 на строку (чтобы потом считать количество анкет)
    """

    # строим отображение "наш тех.ключ" → "имя колонки в df_raw"
    colmap: Dict[str, str] = {}

    inv_lookup = { _norm_header(k): v for k,v in COLUMN_MAP_SCORE.items() }
    inv_meta   = { _norm_header(k): v for k,v in COLUMN_MAP_META.items() }

    raw_cols_norm = { _norm_header(c): c for c in df_raw.columns }

    # метаданные
    for norm_header, canon_name in inv_meta.items():
        if norm_header in raw_cols_norm:
            colmap[canon_name] = raw_cols_norm[norm_header]

    # оценки (1..5)
    for norm_header, canon_name in inv_lookup.items():
        if norm_header in raw_cols_norm:
            colmap[canon_name] = raw_cols_norm[norm_header]

    # собираем выходной df
    out = pd.DataFrame()

    # дата анкетирования — ОБЯЗАТЕЛЬНА
    if "date" not in colmap:
        raise RuntimeError("Не найдена колонка 'Дата анкетирования' в анкете.")
    out["date"] = df_raw[colmap["date"]].map(_parse_date_cell)

    # стандартные поля гостя
    for meta_field in ("fio","booking","phone","email","comment"):
        if meta_field in colmap:
            out[meta_field] = df_raw[colmap[meta_field]].astype(str)
        else:
            out[meta_field] = ""

    # оценки по параметрам
    for param in PARAM_ORDER:
        src_col = colmap.get(param)
        if src_col:
            out[param] = df_raw[src_col].map(_parse_score_1to5)
        else:
            out[param] = np.nan

    # каждая строка = одна анкета
    out["surveys_total"] = 1

    # выкидываем строки без даты вообще (мусор из выгрузки)
    out = out[pd.notna(out["date"])].reset_index(drop=True)

    return out


# =====================================================
# 2. Недельная агрегация
# =====================================================

def weekly_aggregate(df_norm: pd.DataFrame) -> pd.DataFrame:
    """
    На вход: нормализованный df (один гость = одна строка).
    На выход: покомпонентная метрика по каждой неделе и каждому параметру.

    Возвращаем DataFrame с колонками:
        week_key        'YYYY-W##'
        param           'overall', 'spir_checkin', ..., 'nps'
        surveys_total   сколько анкет в этой неделе (все строки)
        answered        сколько человек ответили на этот конкретный вопрос (не NaN)
        avg5            средняя оценка по шкале /5 (только среди ответивших)
        promoters       только для param == 'nps'
        detractors      только для param == 'nps'
        nps_answers     только для param == 'nps'
        nps_value       только для param == 'nps'

    Никакой avg10 больше НЕ считаем.
    """

    if df_norm.empty:
        return pd.DataFrame(columns=[
            "week_key",
            "param",
            "surveys_total",
            "answered",
            "avg5",
            "promoters",
            "detractors",
            "nps_answers",
            "nps_value",
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
            else:
                avg5_val = None

            row = {
                "week_key":      wk,
                "param":         param,
                "surveys_total": week_total,
                "answered":      answered,
                "avg5":          avg5_val,
                "promoters":     None,
                "detractors":    None,
                "nps_answers":   None,
                "nps_value":     None,
            }

            if param == "nps":
                nps_stats = compute_nps_from_1to5(vals)
                row["promoters"]   = nps_stats["promoters"]
                row["detractors"]  = nps_stats["detractors"]
                row["nps_answers"] = nps_stats["nps_answers"]
                row["nps_value"]   = nps_stats["nps_value"]

            rows.append(row)

    out = pd.DataFrame(rows, columns=[
        "week_key",
        "param",
        "surveys_total",
        "answered",
        "avg5",
        "promoters",
        "detractors",
        "nps_answers",
        "nps_value",
    ])

    # Упорядочим красиво: в пределах недели — по PARAM_ORDER, а недели по возрастанию
    out["param"] = pd.Categorical(out["param"], categories=PARAM_ORDER, ordered=True)
    out = out.sort_values(["week_key","param"]).reset_index(drop=True)

    return out


# =====================================================
# 3. Фасад для агентов
# =====================================================

def parse_and_aggregate_weekly(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Удобный фасад:
      - нормализуем сырые данные анкеты → norm
      - считаем недельные метрики → agg
    Возвращаем (norm, agg).
    norm  пригодится для отладки/расширений в будущем,
    agg   пишем в Google Sheets (surveys_history).
    """
    norm = normalize_surveys_df(df_raw)
    agg  = weekly_aggregate(norm)
    return norm, agg
