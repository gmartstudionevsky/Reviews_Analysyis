# agent/surveys_core.py
# TL: Marketing surveys — нормализация + недельная агрегация (включая NPS)
from __future__ import annotations
import re
import math
from datetime import date
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# куда пишем историю
SURVEYS_TAB = "surveys_history"   # week_key | param | responses | avg5 | avg10 | promoters | detractors | nps

# Алиасы (точные названия из выгрузок)
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
        "№  2.2 Оцените работу технической службы",
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
        "3.5 nps 1-5", "nps (1-5)", "nps 1-5",
    ],
    # сервисные
    "survey_date": [
        "Дата анкетирования", "Дата прохождения опроса", "Дата заполнения",
        "Дата и время", "Дата опроса", "Дата анкеты", "Дата",
    ],
    "comment": ["Комментарий гостя", "Комментарий", "Отзыв"],
    "fio": ["ФИО", "Имя", "Имя гостя"],
    "booking": ["Номер брони", "Бронь", "Бронирование"],
    "phone": ["Телефон", "Тел."],
    "email": ["Email", "E-mail", "Почта"],
}

# Порядок в отчетах
PARAM_ORDER: List[str] = [
    "overall",
    "fo_checkin", "clean_checkin", "room_comfort",
    "fo_stay", "its_service", "hsk_stay", "breakfast",
    "atmosphere", "location", "value", "would_return",
    "nps_1_5",
]

# --------- нормализация заголовков и «умный» поиск колонок ----------
def _colkey(s: str) -> str:
    t = str(s).replace("\u00a0", " ").strip().lower()
    t = re.sub(r"\s+", " ", t)
    # оставляем буквы/цифры/пробел/точки и № — чтобы ловить «№ 1.1»
    t = re.sub(r"[^0-9a-zа-яё №\.\-\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

SMART_REGEX: Dict[str, List[re.Pattern]] = {
    "overall":       [re.compile(r"(средн\w*\s+оценк\w*|итог\w*|общ\w*\s+оценк\w*)")],
    "fo_checkin":    [re.compile(r"(№\s*1\.?1\b|1\.?1\b).*?(при[её]м|размещен|заезд|ресеп)"),
                      re.compile(r"(заезд|чек[\-\s]?ин).*?(спир|при[её]м)")],
    "clean_checkin": [re.compile(r"(№\s*1\.?2\b|1\.?2\b).*?(чист)"),
                      re.compile(r"\bчистот[аы]\b.*(заезд)")],
    "room_comfort":  [re.compile(r"(№\s*1\.?3\b|1\.?3\b).*?(комфорт|оснащ)"),
                      re.compile(r"(комфорт|оснащ).*номер")],
    "fo_stay":       [re.compile(r"(№\s*2\.?1\b|2\.?1\b).*?(при[её]м|размещен)"),
                      re.compile(r"(ресепш|ресепшен|администрат).*?(проживан)")],
    "its_service":   [re.compile(r"(№\s*2\.?2\b|2\.?2\b).*?(тех|служб)"),
                      re.compile(r"(техслужб|инженер|ремонт|почин)")],
    "hsk_stay":      [re.compile(r"(№\s*2\.?3\b|2\.?3\b).*?(уборк)"),
                      re.compile(r"\bуборк[аи]\b.*(проживан)")],
    "breakfast":     [re.compile(r"(№\s*2\.?4\b|2\.?4\b).*?(завтрак)"),
                      re.compile(r"\bзавтрак")],
    "atmosphere":    [re.compile(r"(№\s*3\.?1\b|3\.?1\b).*?(атмосфер)")],
    "location":      [re.compile(r"(№\s*3\.?2\b|3\.?2\b).*?(располож)"),
                      re.compile(r"\bрасполож")],
    "value":         [re.compile(r"(№\s*3\.?3\b|3\.?3\b).*?(цен|качеств)"),
                      re.compile(r"(цена|стоимост).*(качест)")],
    "would_return":  [re.compile(r"(№\s*3\.?4\b|3\.?4\b).*?(верн)"),
                      re.compile(r"(вернул.*бы|снова при[её]хал)"),
                     ],
    "nps_1_5":       [re.compile(r"(№\s*3\.?5\b|3\.?5\b).*?(рекоменд|nps)"),
                      re.compile(r"\bnps\b")],
}

def _find_col(df: pd.DataFrame, aliases: List[str]) -> str | None:
    low = {_colkey(c): c for c in df.columns}
    # точные попадания
    for a in aliases:
        k = _colkey(a)
        if k in low: return low[k]
    # подстроки
    for a in aliases:
        k = _colkey(a)
        for lk, orig in low.items():
            if k and k in lk: return orig
    # все слова из алиаса
    for a in aliases:
        words = [w for w in _colkey(a).split() if len(w) > 1]
        for lk, orig in low.items():
            if all(w in lk for w in words): return orig
    return None

def _find_col_smart(df: pd.DataFrame, key: str) -> str | None:
    pats = SMART_REGEX.get(key, [])
    if not pats: return None
    low = {_colkey(c): c for c in df.columns}
    best = None
    for lk, orig in low.items():
        for p in pats:
            if p.search(lk):
                best = orig
                break
        if best: break
    return best

# --------- числовая нормализация и NPS ----------
def to_5_scale(x) -> float:
    """
    Приводим значение к шкале /5.
    Понимаем '4,5', '5 из 5', '9.0' (→ 4.5), '80' (→ 4.0).
    Вне диапазона 1..5 → NaN.
    """
    if x is None: return np.nan
    s = str(x).strip().replace(",", ".")
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if not m: return np.nan
    v = float(m.group(1))
    if 0 <= v <= 5:
        v5 = v
    elif 0 <= v <= 10:
        v5 = v / 2.0
    elif 0 <= v <= 100:
        v5 = v / 20.0
    else:
        return np.nan
    # фильтруем мусор вроде 0.0 или 5.5
    return v5 if 1.0 <= v5 <= 5.0 else np.nan

def compute_nps_from_1to5(series: pd.Series) -> Tuple[int, int, float | np.nan]:
    """
    Новое правило: 1–2 = детракторы, 3–4 = нейтралы, 5 = промоутеры.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    s = s[(s >= 1) & (s <= 5)]
    if s.empty: return 0, 0, np.nan
    promoters  = int((s >= 5.0).sum())
    detractors = int((s <= 2.0).sum())
    total      = int(len(s))
    nps = ((promoters / total) - (detractors / total)) * 100.0
    return promoters, detractors, round(float(nps), 1)

def iso_week_key(d: date) -> str:
    iso = d.isocalendar()
    return f"{iso.year}-W{iso.week}"

# =======================
# Нормализация анкет
# =======================
def _parse_date_any(x) -> date | None:
    try:
        return pd.to_datetime(x, errors="coerce", dayfirst=True).date()
    except Exception:
        return None

def normalize_surveys_df(df0: pd.DataFrame) -> pd.DataFrame:
    """
    На вход «сырая» таблица из Report_*.xlsx (лист с ответами).
    На выход — DF с колонками:
      date, comment, fio, booking, phone, email,
      overall5, overall10, <param5/10 по ключам>, nps5
    Все оценки приводятся к шкале /5 (и /10 для совместимости).
    """
    df = df0.copy()

    # 1) находим колонки по алиасам (+эвристика для даты)
    cols: Dict[str, str] = {}
    for key, aliases in PARAM_ALIASES.items():
        hit = _find_col(df, aliases)
        if hit:
            cols[key] = hit

    if "survey_date" not in cols:
        # эвристика: колонка, в которой >=50% значений парсятся как дата
        best, best_hits, n = None, 0, len(df)
        for c in df.columns:
            try:
                parsed = pd.to_datetime(cast(pd.Series, df[c]), errors="coerce", dayfirst=True)
                hits = int(parsed.notna().sum())
                if hits > best_hits and hits >= max(5, int(0.5 * n)):
                    best, best_hits = c, hits
            except Exception:
                continue
        if best:
            cols["survey_date"] = best
        else:
            raise RuntimeError("В файле не найдена колонка с датой анкетирования.")

    # 2) базовый фрейм
    out = pd.DataFrame()
    out["date"] = df[cols["survey_date"]].map(_parse_date_any)

    for k in ("comment", "fio", "booking", "phone", "email"):
        out[k] = df[cols[k]].astype(str) if k in cols else ""

    # 3) параметрические поля → /5 и /10, плюс NPS (в /5 шкале для вычисления)
    for p in PARAM_ORDER:
        if p == "nps_1_5":
            col = cols.get(p)
            out["nps5"] = df[col].map(to_5_scale) if col else np.nan
            continue
        col = cols.get(p)
        v5 = df[col].map(to_5_scale) if col else np.nan
        out[f"{p}5"] = v5
        out[f"{p}10"] = v5 * 2.0

    # 4) ФОЛБЭК ДЛЯ overall5: заполняем ПОСТРОЧНО средним по доступным параметрам,
    #    если явное значение пустое (или колонки overall5 не было вовсе).
    if "overall5" not in out.columns:
        out["overall5"] = np.nan
        out["overall10"] = np.nan
    value_cols5 = [c for c in out.columns if c.endswith("5") and c not in ("nps5", "overall5")]
    if value_cols5:
        row_mean5 = pd.to_numeric(out[value_cols5], errors="coerce").mean(axis=1)
        out["overall5"] = pd.to_numeric(out["overall5"], errors="coerce")
        out["overall5"] = out["overall5"].where(out["overall5"].notna(), row_mean5)

    out["overall10"] = out["overall5"] * 2.0

    # 5) выбрасываем строки без даты
    out = out[pd.notna(out["date"])].reset_index(drop=True)
    return out


# =======================
# Недельная агрегация
# =======================
def weekly_aggregate(df_norm: pd.DataFrame) -> pd.DataFrame:
    """
    week_key | param | responses | avg5 | avg10 | promoters | detractors | nps
    responses:
      - для 'overall' = число анкет (строк) в неделе
      - для прочих параметров = число валидных ответов по параметру
    """
    if df_norm.empty:
        return pd.DataFrame(columns=["week_key","param","responses","avg5","avg10","promoters","detractors","nps"])

    df = df_norm.copy()
    df["week_key"] = df["date"].map(iso_week_key)

    rows = []
    params = [p for p in PARAM_ORDER if p != "nps_1_5"]

    for wk, wdf in df.groupby("week_key"):
        total_surveys = int(len(wdf))  # это и есть анкеты недели

        for p in params:
            s = pd.to_numeric(wdf[f"{p}5"], errors="coerce")
            s = s.where(s.between(1, 5))
            cnt = int(s.notna().sum())
            avg5 = float(s.mean()) if cnt > 0 else np.nan
            avg10 = (avg5 * 2.0) if cnt > 0 else np.nan
            responses = total_surveys if p == "overall" else cnt
            rows.append([
                wk, p, responses,
                (None if isinstance(avg5,float) and math.isnan(avg5) else round(avg5, 2)),
                (None if isinstance(avg10,float) and math.isnan(avg10) else round(avg10, 2)),
                None, None, None
            ])

        # NPS (1–5: 1–2 D, 3–4 N, 5 P)
        if "nps5" in wdf.columns:
            v = pd.to_numeric(wdf["nps5"], errors="coerce").where(lambda x: x.between(1, 5))
            promoters, detractors, nps = compute_nps_from_1to5(v)
            rows.append([wk, "nps", int(v.notna().sum()), None, None, promoters, detractors, nps])

    out = pd.DataFrame(rows, columns=["week_key","param","responses","avg5","avg10","promoters","detractors","nps"])
    order = [p for p in PARAM_ORDER if p != "nps_1_5"] + ["nps"]
    out["param"] = pd.Categorical(out["param"], categories=order, ordered=True)
    out = out.sort_values(["week_key","param"]).reset_index(drop=True)
    return out

# Фасад
def parse_and_aggregate_weekly(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    norm = normalize_surveys_df(df_raw)
    agg  = weekly_aggregate(norm)
    return norm, agg
