# agent/surveys_core.py
import re
import math
import datetime as dt
from datetime import date, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SURVEYS_TAB = "surveys_history"  # week_key | param | responses | avg5 | promoters | detractors | nps

PARAM_ALIASES: Dict[str, List[str]] = {
    "overall": ["Средняя оценка гостя", "Итоговая оценка", "Общая оценка"],
    "fo_checkin": ["№ 1.1 Оцените работу службы приёма и размещения при заезде", "1.1 прием и размещение при заезде"],
    "clean_checkin": ["№ 1.2 Оцените чистоту номера при заезде", "1.2 чистота при заезде"],
    "room_comfort": ["№ 1.3 Оцените комфорт и оснащение номера", "1.3 комфорт и оснащение"],
    "fo_stay": ["№ 2.1 Оцените работу службы приёма и размещения во время проживания", "2.1 прием и размещение во время проживания"],
    "its_service": ["№ 2.2 Оцените работу технической службы", "2.2 техническая служба"],
    "hsk_stay": ["№ 2.3 Оцените уборку номера во время проживания", "2.3 уборка во время проживания"],
    "breakfast": ["№ 2.4 Оцените завтраки", "2.4 завтраки"],
    "atmosphere": ["№ 3.1 Оцените атмосферу в отеле", "3.1 атмосфера"],
    "location": ["№ 3.2 Оцените расположение отеля", "3.2 расположение"],
    "value": ["№ 3.3 Оцените соотношение цены и качества", "3.3 цена/качество"],
    "would_return": ["№ 3.4 Хотели бы вы вернуться в ARTSTUDIO Nevsky?", "3.4 вернулись бы"],
    "nps_1_5": ["№ 3.5 Оцените вероятность того, что вы порекомендуете нас друзьям и близким (по шкале от 1 до 5)", "3.5 nps 1-5"],
    "survey_date": ["Дата анкетирования", "Дата", "Дата прохождения опроса"],
    "comment": ["Комментарий гостя", "Комментарий", "Отзыв"],
    "fio": ["ФИО", "Имя"],
    "booking": ["Номер брони"],
    "phone": ["Телефон"],
    "email": ["Email"],
}

PARAM_ORDER: List[str] = [
    "overall", "fo_checkin", "clean_checkin", "room_comfort",
    "fo_stay", "its_service", "hsk_stay", "breakfast",
    "atmosphere", "location", "value", "would_return", "nps_1_5"
]

PARAM_NAMES: Dict[str, str] = {
    "overall": "Итоговая оценка",
    "fo_checkin": "СПиР при заезде",
    "clean_checkin": "Чистота при заезде",
    "room_comfort": "Комфорт и оснащение",
    "fo_stay": "СПиР в проживании",
    "its_service": "ИТС (техслужба)",
    "hsk_stay": "Уборка в проживании",
    "breakfast": "Завтраки",
    "atmosphere": "Атмосфера",
    "location": "Расположение",
    "value": "Цена/качество",
    "would_return": "Готовность вернуться",
    "nps_1_5": "NPS (1-5)",
}

def _colkey(s: str) -> str:
    t = str(s).replace("\u00a0", " ").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^0-9a-zа-яё №\.\-\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def find_columns(df: pd.DataFrame) -> Dict[str, str]:
    cols = {}
    headers = {_colkey(h): h for h in df.columns}
    for p, aliases in PARAM_ALIASES.items():
        for a in aliases:
            k = _colkey(a)
            if k in headers:
                cols[p] = headers[k]
                break
    if "survey_date" not in cols:
        for c in df.columns:
            try:
                parsed = pd.to_datetime(df[c], errors='coerce')
                if parsed.notna().sum() > len(df) * 0.5:
                    cols["survey_date"] = c
                    break
            except:
                pass
    return cols

def to_5_scale(x):
    if pd.isna(x) or x == "Нет оценки":
        return np.nan
    if isinstance(x, str):
        x = x.replace(',', '.').strip()
    try:
        v = float(x)
        return v if 1 <= v <= 5 else np.nan
    except:
        return np.nan

def normalize_surveys_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    cols = find_columns(df)
    if "survey_date" not in cols:
        raise ValueError("No date column found")
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[cols["survey_date"]], errors='coerce').dt.date
    for k in ("comment", "fio", "booking", "phone", "email"):
        out[k] = df.get(cols.get(k, ""), "")
    for p in PARAM_ORDER:
        col = cols.get(p)
        if col:
            out[p] = df[col].map(to_5_scale)
    out = out.dropna(subset=["date"])
    if out["overall"].isna().all():
        value_cols = [p for p in PARAM_ORDER if p not in ("overall", "nps_1_5") and p in out.columns]
        if value_cols:
            out["overall"] = out[value_cols].mean(axis=1)
    return out

def compute_nps(s: pd.Series) -> Tuple[int, int, float]:
    valid = s.notna().sum()
    if valid == 0:
        return None, None, None
    promoters = (s == 5).sum()
    detractors = (s <= 2).sum()
    nps = round(100 * (promoters - detractors) / valid, 2)
    return promoters, detractors, nps

def iso_week_key(d: date) -> str:
    year, week, _ = d.isocalendar()
    return f"{year}-W{week:02d}"

def weekly_aggregate(df_norm: pd.DataFrame) -> pd.DataFrame:
    if df_norm.empty:
        return pd.DataFrame()
    df = df_norm.copy()
    df["week_key"] = df["date"].apply(iso_week_key)
    rows = []
    params = [p for p in PARAM_ORDER if p != "nps_1_5"]
    for wk, wdf in df.groupby("week_key"):
        total = len(wdf)
        for p in params:
            s = wdf[p].dropna()
            cnt = len(s)
            if cnt == 0 and p != "overall":
                continue
            avg = s.mean() if cnt > 0 else np.nan
            responses = total if p == "overall" else cnt
            rows.append([wk, p, responses, round(avg, 2) if not np.isnan(avg) else None, None, None, None])
        nps_s = wdf.get("nps_1_5").dropna()
        nps_cnt = len(nps_s)
        if nps_cnt > 0:
            prom, det, nps = compute_nps(nps_s)
            rows.append([wk, "nps", nps_cnt, None, prom, det, nps])
    out = pd.DataFrame(rows, columns=["week_key", "param", "responses", "avg5", "promoters", "detractors", "nps"])
    return out.sort_values(["week_key", "param"])

def parse_and_aggregate_weekly(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    norm = normalize_surveys_df(df_raw)
    agg = weekly_aggregate(norm)
    return norm, agg

# Date utilities
def iso_week_monday(week_key: str) -> date:
    y, w = map(int, week_key.split("-W"))
    jan4 = date(y, 1, 4)
    return jan4 - timedelta(days=jan4.isoweekday() - 1) + timedelta(weeks=w-1)

def period_ranges_for_week(week_start: date) -> Dict:
    week_end = week_start + timedelta(days=6)
    month_start = week_start.replace(day=1)
    month_end = (month_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    quarter = (week_start.month - 1) // 3 + 1
    q_start_month = (quarter - 1) * 3 + 1
    q_start = date(week_start.year, q_start_month, 1)
    q_end = (date(week_start.year, q_start_month + 3, 1) - timedelta(days=1)) if quarter < 4 else date(week_start.year, 12, 31)
    year_start = date(week_start.year, 1, 1)
    year_end = date(week_start.year, 12, 31)
    return {
        "week": {"start": week_start, "end": week_end},
        "mtd": {"start": month_start, "end": month_end},
        "qtd": {"start": q_start, "end": q_end},
        "ytd": {"start": year_start, "end": year_end},
    }

def aggregate_period(hist: pd.DataFrame, start: date, end: date) -> Dict:
    wk_start = iso_week_key(start)
    wk_end = iso_week_key(end)
    period_df = hist[(hist["week_key"] >= wk_start) & (hist["week_key"] <= wk_end)]
    by_param = {}
    params = [p for p in PARAM_ORDER if p != "nps_1_5"]
    for p in params:
        pdf = period_df[period_df["param"] == p]
        if pdf.empty:
            by_param[p] = {"responses": 0, "avg5": None}
            continue
        weights = pdf["responses"]
        values = pdf["avg5"]
        avg = np.average(values, weights=weights) if weights.sum() > 0 else None
        by_param[p] = {"responses": weights.sum(), "avg5": round(avg, 2) if avg is not None else None}
    nps_pdf = period_df[period_df["param"] == "nps"]
    if not nps_pdf.empty:
        prom = nps_pdf["promoters"].sum()
        det = nps_pdf["detractors"].sum()
        tot = nps_pdf["responses"].sum()
        nps = round(100 * (prom - det) / tot, 2) if tot > 0 else None
    else:
        nps, tot = None, 0
    by_param["nps"] = {"responses": tot, "avg5": None, "nps": nps}
    total_surveys = by_param["overall"]["responses"]
    overall_avg = by_param["overall"]["avg5"]
    return {"total_surveys": total_surveys, "overall_avg5": overall_avg, "nps": by_param["nps"]["nps"], "by_param": by_param}
