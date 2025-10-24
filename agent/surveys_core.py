# TL: Marketing surveys — нормализация + недельная агрегация (+NPS) + дедуп истории
from __future__ import annotations
import re
import math
from datetime import date
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SURVEYS_TAB = "surveys_history"   # week_key | param | responses | avg5 | avg10 | promoters | detractors | nps

# Алиасы колонок (быстрее поймаем «человеческие» заголовки)
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

PARAM_ORDER: List[str] = [
    "overall",
    "fo_checkin", "clean_checkin", "room_comfort",
    "fo_stay", "its_service", "hsk_stay", "breakfast",
    "atmosphere", "location", "value", "would_return",
    "nps_1_5",
]

def _colkey(s: str) -> str:
    t = str(s).replace("\u00a0", " ").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\sа-яё]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def _find_col(df: pd.DataFrame, aliases: List[str]) -> str | None:
    low = {_colkey(c): c for c in df.columns}
    for a in aliases:
        k = _colkey(a)
        if k in low: return low[k]
    for a in aliases:
        k = _colkey(a)
        for lk, orig in low.items():
            if k and k in lk: return orig
    for a in aliases:
        words = [w for w in _colkey(a).split() if len(w) > 1]
        for lk, orig in low.items():
            if all(w in lk for w in words): return orig
    return None

def _parse_date_any(x) -> date | None:
    try:
        return pd.to_datetime(x, errors="coerce", dayfirst=True).date()
    except Exception:
        return None

def to_5_scale(x) -> float:
    """Любыe цифры → шкала /5. Понимаем запятую/точку/проценты/пустые/прочерки."""
    if x is None: return np.nan
    s = str(x).strip().replace(",", ".")
    if s in ("", "—", "-", "–"): return np.nan
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if not m: return np.nan
    v = float(m.group(1))
    if 0 <= v <= 5: return v
    if 0 <= v <= 10: return v / 2.0
    if 0 <= v <= 100: return v / 20.0
    return np.nan

def compute_nps_from_1to5(series: pd.Series) -> Tuple[int, int, float | np.nan]:
    """1–2 D, 3–4 N, 5 P → (P,D,NPS%)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return 0, 0, np.nan
    promoters  = int((s >= 5.0).sum())
    detractors = int((s <= 2.0).sum())
    total      = int(len(s))
    nps = ((promoters / total) - (detractors / total)) * 100.0
    return promoters, detractors, round(float(nps), 1)

def iso_week_key(d: date) -> str:
    iso = d.isocalendar()
    return f"{iso.year}-W{iso.week}"

# ----------------- Нормализация анкеты (1 строка = 1 анкета) -----------------
def normalize_surveys_df(df0: pd.DataFrame) -> pd.DataFrame:
    """
    На вход — «сырая» таблица листа «Оценки гостей».
    На выход — DF колонок:
      date, comment, fio, booking, phone, email,
      overall5, overall10, <param5/param10 по ключам>, nps5
    """
    df = df0.copy()

    # подобрать колонки по алиасам (+эвристика для даты)
    cols: Dict[str, str] = {}
    for key, aliases in PARAM_ALIASES.items():
        hit = _find_col(df, aliases)
        if hit: cols[key] = hit

    if "survey_date" not in cols:
        best, best_hits, n = None, 0, len(df)
        for c in df.columns:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
                hits = int(parsed.notna().sum())
                if hits > best_hits and hits >= max(3, int(0.5 * n)):
                    best, best_hits = c, hits
            except Exception:
                continue
        if best: cols["survey_date"] = best
        else:
            raise RuntimeError("В файле не найдена колонка с датой анкетирования.")

    out = pd.DataFrame()
    out["date"] = df[cols["survey_date"]].map(_parse_date_any)

    for k in ("comment", "fio", "booking", "phone", "email"):
        out[k] = df[cols[k]].astype(str) if k in cols else ""

    # все параметрические поля → /5 и /10
    for p in PARAM_ORDER:
        if p == "nps_1_5":
            col = cols.get(p)
            out["nps5"] = df[col].map(to_5_scale) if col else np.nan
            continue
        col = cols.get(p)
        v5 = df[col].map(to_5_scale) if col else np.nan
        out[f"{p}5"]  = v5
        out[f"{p}10"] = v5 * 2.0

    # фоллбэк Итоговой: если пусто — среднее по тематическим колонкам /5
    if "overall5" not in out.columns or out["overall5"].isna().all():
        value_cols5 = [c for c in out.columns if c.endswith("5") and c not in ("nps5",)]
        if value_cols5:
            sub = out[value_cols5].apply(pd.to_numeric, errors="coerce")
            out["overall5"]  = sub.mean(axis=1, skipna=True)
            out["overall10"] = out["overall5"] * 2.0

    out = out[pd.notna(out["date"])].reset_index(drop=True)
    return out

# ----------------- Недельная агрегация (строго 1 анкета = 1 ответчик) -----------------
def weekly_aggregate(df_norm: pd.DataFrame) -> pd.DataFrame:
    """
    Возвращает DF: week_key | param | responses | avg5 | avg10 | promoters | detractors | nps
    ВАЖНО:
      - responses для 'overall' = число анкет недели (строк);
      - для остальных — число НЕ-пустых ответов по параметру.
    """
    if df_norm.empty:
        return pd.DataFrame(columns=["week_key","param","responses","avg5","avg10","promoters","detractors","nps"])

    df = df_norm.copy()
    df["week_key"] = df["date"].map(iso_week_key)

    rows = []
    params = [p for p in PARAM_ORDER if p != "nps_1_5"]

    for wk, wdf in df.groupby("week_key"):
        total_surveys = int(len(wdf))  # это и есть «анкет за неделю»

        for p in params:
            s = pd.to_numeric(wdf[f"{p}5"], errors="coerce").where(lambda x: x.between(0,5))
            cnt = int(s.notna().sum())
            avg5  = float(s.mean()) if cnt > 0 else np.nan
            avg10 = (avg5 * 2.0) if cnt > 0 else np.nan
            responses = total_surveys if p == "overall" else cnt
            rows.append([wk, p, responses,
                         (None if math.isnan(avg5) else round(avg5, 2)),
                         (None if math.isnan(avg10) else round(avg10, 2)),
                         None, None, None])

        if "nps5" in wdf.columns:
            v = pd.to_numeric(wdf["nps5"], errors="coerce").where(lambda x: x.between(1,5))
            promoters, detractors, nps = compute_nps_from_1to5(v)
            rows.append([wk, "nps", int(v.notna().sum()), None, None, promoters, detractors, nps])

    out = pd.DataFrame(rows, columns=["week_key","param","responses","avg5","avg10","promoters","detractors","nps"])
    order = [p for p in PARAM_ORDER if p != "nps_1_5"] + ["nps"]
    out["param"] = pd.Categorical(out["param"], categories=order, ordered=True)
    return out.sort_values(["week_key","param"]).reset_index(drop=True)

# ----------------- Дедуп истории (на случай дублей недель) -----------------
def dedupe_surveys_history(hist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Дедуп по (week_key,param).
    param!='nps': средняя взвешенная по responses, responses = MAX по дублям
    param=='nps' : promoters/detractors суммируются, responses = MAX, NPS пересчитывается
    """
    if hist_df is None or hist_df.empty:
        return hist_df

    df = hist_df.copy()
    for c in ["responses", "avg5", "avg10", "promoters", "detractors", "nps"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    def combine(grp: pd.DataFrame) -> pd.Series:
        param = str(grp["param"].iloc[0])
        if param == "nps":
            prom = grp["promoters"].fillna(0).sum()
            detr = grp["detractors"].fillna(0).sum()
            resp = grp["responses"].fillna(0).max()
            nps_val = ((prom/(resp or 1)) - (detr/(resp or 1))) * 100.0 if resp and resp > 0 else np.nan
            return pd.Series({
                "week_key": grp["week_key"].iloc[0], "param": param,
                "responses": resp if not np.isnan(resp) else None,
                "avg5": None, "avg10": None,
                "promoters": int(prom), "detractors": int(detr),
                "nps": (None if np.isnan(nps_val) else float(round(nps_val, 1))),
            })
        w = grp["responses"].fillna(0)
        wsum = float(w.sum())
        avg5  = float((grp["avg5"]  * w).sum()/wsum) if wsum>0 else np.nan
        avg10 = float((grp["avg10"] * w).sum()/wsum) if wsum>0 else np.nan
        resp = float(w.max())
        return pd.Series({
            "week_key": grp["week_key"].iloc[0], "param": param,
            "responses": (None if np.isnan(resp) else resp),
            "avg5": (None if np.isnan(avg5) else round(avg5, 2)),
            "avg10": (None if np.isnan(avg10) else round(avg10, 2)),
            "promoters": None, "detractors": None, "nps": None,
        })

    return (df.groupby(["week_key","param"], as_index=False)
              .apply(combine).reset_index(drop=True))

# ----------------- Аггрегирование периода (неделя/месяц/квартал/год) -----------------
def surveys_aggregate_period(history_df: pd.DataFrame, start: date, end: date, iso_week_monday_fn) -> dict:
    """
    Взвешенная агрегация по параметрам в [start; end].
    Возвращает {'by_param': DF, 'totals': {...}}
    """
    if history_df is None or history_df.empty:
        return {"by_param": pd.DataFrame(columns=["param","responses","avg5","avg10","promoters","detractors","nps"]),
                "totals": {"responses": 0, "overall5": None, "nps": None}}

    df = history_df.copy()
    for c in ["responses","avg5","avg10","promoters","detractors","nps"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["monday"] = df["week_key"].map(iso_week_monday_fn)
    df = df[(df["monday"]>=start) & (df["monday"]<=end)].copy()
    if df.empty:
        return {"by_param": pd.DataFrame(columns=["param","responses","avg5","avg10","promoters","detractors","nps"]),
                "totals": {"responses": 0, "overall5": None, "nps": None}}

    rows=[]
    for param, grp in df.groupby("param"):
        resp = int(grp["responses"].fillna(0).sum())
        if param == "nps":
            prom = int(grp["promoters"].fillna(0).sum())
            detr = int(grp["detractors"].fillna(0).sum())
            nps  = ((prom/(resp or 1)) - (detr/(resp or 1))) * 100.0 if resp>0 else np.nan
            rows.append([param, resp, np.nan, np.nan, prom, detr, (None if np.isnan(nps) else round(float(nps),1))])
            continue
        sub5  = grp[pd.notna(grp["avg5"])  & pd.notna(grp["responses"])]
        sub10 = grp[pd.notna(grp["avg10"]) & pd.notna(grp["responses"])]
        avg5  = (sub5["avg5"]  * sub5["responses"]).sum()/sub5["responses"].sum() if not sub5.empty else np.nan
        avg10 = (sub10["avg10"]* sub10["responses"]).sum()/sub10["responses"].sum() if not sub10.empty else np.nan
        rows.append([param, resp,
                     (None if (isinstance(avg5,float) and np.isnan(avg5)) else round(float(avg5),2)),
                     (None if (isinstance(avg10,float)and np.isnan(avg10)) else round(float(avg10),2)),
                     None,None,None])

    by_param = pd.DataFrame(rows, columns=["param","responses","avg5","avg10","promoters","detractors","nps"]).sort_values("param")
    ov = by_param[by_param["param"]=="overall"]
    totals = {
        "responses": int(ov["responses"].sum()) if not ov.empty else 0,
        "overall5": (None if ov.empty or pd.isna(ov.iloc[0]["avg5"]) else float(ov.iloc[0]["avg5"])),
        "nps":      (None if by_param[by_param["param"]=="nps"].empty
                     or pd.isna(by_param[by_param["param"]=="nps"].iloc[0]["nps"])
                     else float(by_param[by_param["param"]=="nps"].iloc[0]["nps"])),
    }
    return {"by_param": by_param, "totals": totals}


    # --- Фасад: из сырых данных -> (нормализованный DF, недельные агрегаты) ---
def parse_and_aggregate_weekly(df_raw: pd.DataFrame):
    """
    Совместимость со старыми агентами:
    возвращает (norm_df, weekly_agg_df)
    """
    norm = normalize_surveys_df(df_raw)
    agg  = weekly_aggregate(norm)
    return norm, agg

