# agent/surveys_core.py
# TL: Marketing surveys — нормализация + недельная агрегация (включая NPS)

from __future__ import annotations
import re
import math
from datetime import date
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ====== Константа: вкладка в Google Sheet, куда пишем историю ======
SURVEYS_TAB = "surveys_history"   # week_key | param | responses | avg5 | avg10 | promoters | detractors | nps

# ====== Алиасы колонок (то, как поля могут называться в выгрузках) ======
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

    # сервисные/контактные
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

# Порядок параметров для вывода/сводов
PARAM_ORDER: List[str] = [
    "overall",
    "fo_checkin", "clean_checkin", "room_comfort",
    "fo_stay", "its_service", "hsk_stay", "breakfast",
    "atmosphere", "location", "value", "would_return",
    "nps_1_5",
]

# =======================
# Вспомогательные утилиты
# =======================
def _colkey(s: str) -> str:
    """Нормализация заголовка: нижний регистр, убираем nbsp/повторы пробелов и пунктуацию."""
    t = str(s).replace("\u00a0", " ").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\sа-яё]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def _find_col(df: pd.DataFrame, aliases: List[str]) -> str | None:
    low = {_colkey(c): c for c in df.columns}
    # точное совпадение
    for a in aliases:
        k = _colkey(a)
        if k in low:
            return low[k]
    # по подстроке
    for a in aliases:
        k = _colkey(a)
        for lk, orig in low.items():
            if k and k in lk:
                return orig
    # все слова из алиаса встречаются
    for a in aliases:
        words = [w for w in _colkey(a).split() if len(w) > 1]
        for lk, orig in low.items():
            if all(w in lk for w in words):
                return orig
    return None

def _parse_date_any(x) -> date | None:
    try:
        return pd.to_datetime(x, errors="coerce", dayfirst=True).date()
    except Exception:
        return None

def to_5_scale(x) -> float:
    """
    Приводим значение к шкале /5.
    Понимаем '4,5', '9.0' (→ 4.5), '80' (→ 4.0), пустые/прочерки → NaN.
    """
    if x is None:
        return np.nan
    s = str(x).strip().replace(",", ".")
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if not m:
        return np.nan
    v = float(m.group(1))
    if 0 <= v <= 5:
        return v
    if 0 <= v <= 10:
        return v / 2.0
    if 0 <= v <= 100:
        return v / 20.0
    return np.nan

def compute_nps_from_1to5(series: pd.Series) -> Tuple[int, int, float | np.nan]:
    """1–2 = детракторы, 3 = нейтралы, 4–5 = промоутеры. Возвращает (P, D, NPS%)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0, 0, np.nan
    promoters  = int((s >= 4.0).sum())
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
def normalize_surveys_df(df0: pd.DataFrame) -> pd.DataFrame:
    """
    На вход «сырая» таблица из Report_*.xlsx (лист «Оценки гостей»).
    На выход — DF с колонками:
      date, comment, fio, booking, phone, email,
      overall5, overall10, <param5/10 по ключам>, nps5
    Все оценки приведены к шкале /5 (и /10 — для совместимости).
    """
    df = df0.copy()

    # находим колонки по алиасам (+эвристика для даты)
    cols: Dict[str, str] = {}
    for key, aliases in PARAM_ALIASES.items():
        hit = _find_col(df, aliases)
        if hit:
            cols[key] = hit

    if "survey_date" not in cols:
        # эвристика: колонка, в которой распарсилось >=50% значений как дата
        best, best_hits, n = None, 0, len(df)
        for c in df.columns:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
                hits = int(parsed.notna().sum())
                if hits > best_hits and hits >= max(5, int(0.5 * n)):
                    best, best_hits = c, hits
            except Exception:
                continue
        if best:
            cols["survey_date"] = best
        else:
            raise RuntimeError(
                "В файле не найдена колонка с датой анкетирования. "
                "Проверьте заголовок столбца с датой (например: «Дата анкетирования»)."
            )

    out = pd.DataFrame()
    out["date"] = df[cols["survey_date"]].map(_parse_date_any)

    # текстовые поля (не критичны)
    for k in ("comment", "fio", "booking", "phone", "email"):
        out[k] = df[cols[k]].astype(str) if k in cols else ""

    # все параметрические колонки → /5
    for p in PARAM_ORDER:
        if p == "nps_1_5":
            col = cols.get(p)
            out["nps5"] = df[col].map(to_5_scale) if col else np.nan
            continue
        short = p  # ключ
        col = cols.get(p)
        v5 = df[col].map(to_5_scale) if col else np.nan
        out[f"{short}5"] = v5
        out[f"{short}10"] = v5 * 2.0

    # если нет явной overall5 — усредним по тематическим шкалам /5
    if out["overall5"].isna().all():
        value_cols5 = [c for c in out.columns if c.endswith("5") and c not in ("nps5",)]
        if value_cols5:
            out["overall5"] = pd.to_numeric(out[value_cols5], errors="coerce").mean(axis=1)
            out["overall10"] = out["overall5"] * 2.0

    # выбрасываем строки без даты (не анкеты)
    out = out[pd.notna(out["date"])].reset_index(drop=True)
    return out

# =======================
# Недельная агрегация
# =======================
def weekly_aggregate(df_norm: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует нормализованные ответы по неделям и параметрам.
    ВАЖНО: число «responses» для param=='overall' = число анкет (строк) в неделе,
           для прочих параметров — число валидных ответов по этому параметру.
    Возвращает DF: week_key | param | responses | avg5 | avg10 | promoters | detractors | nps
    """
    if df_norm.empty:
        return pd.DataFrame(columns=["week_key","param","responses","avg5","avg10","promoters","detractors","nps"])

    df = df_norm.copy()
    df["week_key"] = df["date"].map(iso_week_key)

    rows = []
    params = [p for p in PARAM_ORDER if p != "nps_1_5"]

    for wk, wdf in df.groupby("week_key"):
        total_surveys = int(len(wdf))  # это и есть кол-во анкет в неделе

        # параметрические (включая overall)
        for p in params:
            s = pd.to_numeric(wdf[f"{p}5"], errors="coerce")
            s = s.where(s.between(0, 5))
            cnt = int(s.notna().sum())
            avg5 = float(s.mean()) if cnt > 0 else np.nan
            avg10 = (avg5 * 2.0) if cnt > 0 else np.nan
            responses = total_surveys if p == "overall" else cnt
            rows.append([
                wk, p, responses,
                (None if math.isnan(avg5) else round(avg5, 2)),
                (None if math.isnan(avg10) else round(avg10, 2)),
                None, None, None
            ])

        # NPS (шкала 1–5)
        if "nps5" in wdf.columns:
            v = pd.to_numeric(wdf["nps5"], errors="coerce").where(lambda x: x.between(1, 5))
            promoters, detractors, nps = compute_nps_from_1to5(v)
            rows.append([wk, "nps", int(v.notna().sum()), None, None, promoters, detractors, nps])

    out = pd.DataFrame(rows, columns=["week_key","param","responses","avg5","avg10","promoters","detractors","nps"])
    # порядок параметров в выводе
    order = [p for p in PARAM_ORDER if p != "nps_1_5"] + ["nps"]
    out["param"] = pd.Categorical(out["param"], categories=order, ordered=True)
    out = out.sort_values(["week_key","param"]).reset_index(drop=True)
    return out

# Фасад: из «сырых» данных → (нормализованный DF, недельные агрегаты)
def parse_and_aggregate_weekly(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    norm = normalize_surveys_df(df_raw)
    agg = weekly_aggregate(norm)
    return norm, agg
