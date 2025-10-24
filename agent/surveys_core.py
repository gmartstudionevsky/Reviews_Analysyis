# agent/surveys_core.py
# TL: Marketing surveys — parsing & weekly aggregation (+NPS)
import re
import math
import datetime as dt
from datetime import date
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ====== Константы Google Sheet (вкладка с историей анкет) ======
SURVEYS_TAB = "surveys_history"   # будет: week_key | param | responses | avg5 | avg10 | promoters | detractors | nps

# ====== Словарь параметров (ключ -> список вариантов названия в выгрузке) ======
# Ты дал «длинные» русские заголовки. Мы сопоставляем их с короткими ключами.
PARAM_ALIASES: Dict[str, List[str]] = {
    "overall": [
        "Средняя оценка гостя", "Итоговая оценка", "Общая оценка",
    ],
    "fo_checkin": [
        "№ 1.1 Оцените работу службы приёма и размещения при заезде",
        "1.1 прием и размещение при заезде",
    ],
    "clean_checkin": [
        "№ 1.2 Оцените чистоту номера при заезде",
        "1.2 чистота при заезде",
    ],
    "room_comfort": [
        "№ 1.3 Оцените комфорт и оснащение номера",
        "1.3 комфорт и оснащение",
    ],
    "fo_stay": [
        "№ 2.1 Оцените работу службы приёма и размещения во время проживания",
        "2.1 прием и размещение во время проживания",
    ],
    "its_service": [
        "№ 2.2 Оцените работу технической службы",
        "2.2 техническая служба",
    ],
    "hsk_stay": [
        "№ 2.3 Оцените уборку номера во время проживания",
        "2.3 уборка во время проживания",
    ],
    "breakfast": [
        "№ 2.4 Оцените завтраки",
        "2.4 завтраки",
    ],
    "atmosphere": [
        "№ 3.1 Оцените атмосферу в отеле",
        "3.1 атмосфера",
    ],
    "location": [
        "№ 3.2 Оцените расположение отеля",
        "3.2 расположение",
    ],
    "value": [
        "№ 3.3 Оцените соотношение цены и качества",
        "3.3 цена/качество",
    ],
    "would_return": [
        "№ 3.4 Хотели бы вы вернуться в ARTSTUDIO Nevsky?",
        "3.4 вернулись бы",
    ],
    "nps_1_5": [
        "№ 3.5 Оцените вероятность того, что вы порекомендуете нас друзьям и близким (по шкале от 1 до 5)",
        "3.5 nps 1-5",
        "nps (1-5)",
    ],
    # служебные/контактные
    "survey_date": ["Дата анкетирования", "Дата опроса", "Дата"],
    "comment": ["Комментарий гостя", "Комментарий", "Отзыв"],
    "fio": ["ФИО", "Имя", "Имя гостя"],
    "booking": ["Номер брони", "Бронь", "Бронирование"],
    "phone": ["Телефон", "Тел."],
    "email": ["Email", "E-mail", "Почта"],
}

# Порядок параметров для отчётов
PARAM_ORDER: List[str] = [
    "overall",
    "fo_checkin", "clean_checkin", "room_comfort",
    "fo_stay", "its_service", "hsk_stay", "breakfast",
    "atmosphere", "location", "value", "would_return",
    "nps_1_5",
]

# ====== Утилиты ======
def _colkey(s: str) -> str:
    """нормализуем имя колонки: нижний регистр, убираем лишние пробелы/nbsp"""
    return re.sub(r"\s+", " ", str(s).replace("\u00a0", " ").strip().lower())

def _find_col(df: pd.DataFrame, aliases: List[str]) -> str | None:
    low = {_colkey(c): c for c in df.columns}
    for a in aliases:
        k = _colkey(a)
        if k in low:
            return low[k]
    # поможем себе «содержанием»
    for a in aliases:
        pat = re.escape(_colkey(a))
        for lk, orig in low.items():
            if re.search(pat, lk):
                return orig
    return None

def _num5(x):
    """приводим к шкале /5 (float), понимаем '4,5' и '—'."""
    try:
        if x is None: return np.nan
        s = str(x).strip().replace(",", ".")
        if s in ("", "—", "-", "–"): return np.nan
        v = float(s)
        # если внезапно 1..10 — приводим к /5
        if 0 <= v <= 10 and v > 5:
            v = v / 2.0
        return v
    except:
        return np.nan

def _to10(x):
    return x * 2.0 if pd.notna(x) else np.nan

def iso_week_key(d: date) -> str:
    iso = d.isocalendar()
    return f"{iso[0]}-W{iso[1]}"

def _parse_date_any(s) -> date | None:
    try:
        return pd.to_datetime(s, errors="coerce", dayfirst=True).date()
    except Exception:
        return None

def compute_nps_from_1to5(series: pd.Series) -> Tuple[int, int, float]:
    """
    1–2 = детракторы, 3 = нейтралы, 4–5 = промоутеры.
    Возвращает (promoters, detractors, NPS%), где NPS = (P% − D%).
    """
    s = series.dropna().astype(float)
    if s.empty:
        return 0, 0, np.nan
    promoters   = int(((s >= 4.0).astype(int)).sum())
    detractors  = int(((s <= 2.0).astype(int)).sum())
    total       = len(s)
    nps = ((promoters / total) - (detractors / total)) * 100.0
    return promoters, detractors, round(nps, 1)

# ====== Основные функции ======
def normalize_surveys_df(df0: pd.DataFrame) -> pd.DataFrame:
    """
    На вход «как есть» датафрейм из Report_*.xlsx.
    На выход — нормализованный DF с колонками:
      date, comment, fio, booking, phone, email,
      overall5, overall10, <param5/10 по ключам>, nps5
    """
    df = df0.copy()
    # подберём колонки
    cols = {}
    for key, aliases in PARAM_ALIASES.items():
        hit = _find_col(df, aliases)
        if hit: cols[key] = hit

    # обязательные поля: дата + хотя бы один параметр
    if "survey_date" not in cols:
        raise RuntimeError("В файле не найдена колонка с датой анкетирования.")
    # формируем результирующий df
    out = pd.DataFrame()
    out["date"] = df[cols["survey_date"]].map(_parse_date_any)

    for k in ("comment", "fio", "booking", "phone", "email"):
        if k in cols:
            out[k] = df[cols[k]].astype(str)
        else:
            out[k] = ""

    # оценки по параметрам
    for p in PARAM_ORDER:
        if p == "nps_1_5":
            col = cols.get(p)
            out["nps5"] = df[col].map(_num5) if col else np.nan
            continue
        col = cols.get(p)
        if not col:
            out[f"{p}5"]  = np.nan
            out[f"{p}10"] = np.nan
        else:
            v5 = df[col].map(_num5)
            out[f"{p}5"]  = v5
            out[f"{p}10"] = v5 * 2.0

    # если нет явной "overall", попробуем усреднить по тематическим полям
    if "overall5" not in out.columns or out["overall5"].isna().all():
        value_cols5 = [c for c in out.columns if c.endswith("5") and c not in ("nps5",)]
        if value_cols5:
            out["overall5"]  = pd.to_numeric(out[value_cols5], errors="coerce").mean(axis=1)
            out["overall10"] = out["overall5"] * 2.0

    # нормализуем дату, выбрасываем пустые
    out = out[pd.notna(out["date"])].reset_index(drop=True)
    return out

def weekly_aggregate(df_norm: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует нормализованные ответы по неделям и параметрам.
    Возвращает DF с колонками:
      week_key | param | responses | avg5 | avg10 | promoters | detractors | nps
    """
    if df_norm.empty:
        return pd.DataFrame(columns=["week_key","param","responses","avg5","avg10","promoters","detractors","nps"])

    df = df_norm.copy()
    df["week_key"] = df["date"].map(iso_week_key)

    rows = []

    # все параметрические поля, кроме NPS
    params5 = [c for c in df.columns if c.endswith("5") and c not in ("nps5",)]
    for col in params5:
        param = col[:-1]  # отрежем '5' → 'overall' / 'fo_checkin' ...
        sub = df[[ "week_key", col, f"{param}10" ]].dropna(subset=[col])
        if sub.empty: 
            continue
        g = sub.groupby("week_key")
        for wk, grp in g:
            responses = int(len(grp))
            avg5 = float(np.nanmean(grp[col].values)) if responses else np.nan
            avg10 = float(np.nanmean(grp[f"{param}10"].values)) if responses else np.nan
            rows.append([wk, param, responses, round(avg5,2) if not np.isnan(avg5) else np.nan,
                         round(avg10,2) if not np.isnan(avg10) else np.nan, None, None, None])

    # NPS по неделям
    if "nps5" in df.columns:
        sub = df[["week_key","nps5"]].dropna(subset=["nps5"])
        g = sub.groupby("week_key")
        for wk, grp in g:
            promoters, detractors, nps = compute_nps_from_1to5(grp["nps5"])
            rows.append([wk, "nps", int(len(grp)), np.nan, np.nan, promoters, detractors, nps])

    out = pd.DataFrame(rows, columns=["week_key","param","responses","avg5","avg10","promoters","detractors","nps"])
    # порядок параметров в выводе
    cat_order = [p for p in PARAM_ORDER if p != "nps_1_5"] + ["nps"]
    out["param"] = pd.Categorical(out["param"], categories=cat_order, ordered=True)
    out = out.sort_values(["week_key","param"]).reset_index(drop=True)
    return out

# Удобная фасадная функция: из «сырых» данных → нормализованный DF + недельные агрегаты
def parse_and_aggregate_weekly(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    norm = normalize_surveys_df(df_raw)
    agg  = weekly_aggregate(norm)
    return norm, agg
