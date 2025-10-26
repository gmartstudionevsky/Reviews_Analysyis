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
    "fo_stay":       [re.compile(r"(№\s*2\.?1\b|2\.?1\b).*?(при[её]м|размещен|прожив)"),
                      re.compile(r"(прожив|стей).*?(спир|при[её]м)")],
    "its_service":   [re.compile(r"(№\s*2\.?2\b|2\.?2\b).*?(техн|служб|инженер|ремонт)"),
                      re.compile(r"(техн|инженер|ремонт|служб)")],
    "hsk_stay":      [re.compile(r"(№\s*2\.?3\b|2\.?3\b).*?(уборк|чистот|прожив)"),
                      re.compile(r"\b(уборк|чистот[аы])\b.*(прожив)")],
    "breakfast":     [re.compile(r"(№\s*2\.?4\b|2\.?4\b).*?(завтрак|питан)"),
                      re.compile(r"(завтрак|питан|еда|ресторан)")],
    "atmosphere":    [re.compile(r"(№\s*3\.?1\b|3\.?1\b).*?(атмосфер|уют|дизайн)"),
                      re.compile(r"(атмосфер|уют|дизайн)")],
    "location":      [re.compile(r"(№\s*3\.?2\b|3\.?2\b).*?(располож|локац|местополож)"),
                      re.compile(r"(располож|локац|местополож|удобств.*мест)")],
    "value":         [re.compile(r"(№\s*3\.?3\b|3\.?3\b).*?(цен|качеств|value)"),
                      re.compile(r"(цен|качеств|value)")],
    "would_return":  [re.compile(r"(№\s*3\.?4\b|3\.?4\b).*?(верн|рекоменд|повтор)"),
                      re.compile(r"(верн|повтор|снова|хотел.*верн)")],
    "nps_1_5":       [re.compile(r"(№\s*3\.?5\b|3\.?5\b).*?(рекоменд|nps|друзь|вероятн)"),
                      re.compile(r"(nps|рекоменд|друзь|вероятн)")],
}

def find_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Находит колонки параметров по алиасам или умным регекспам."""
    cols = {}
    headers = {_colkey(h): h for h in df.columns}
    for p, aliases in PARAM_ALIASES.items():
        for a in aliases:
            k = _colkey(a)
            if k in headers:
                cols[p] = headers[k]
                break
    # для не найденных — умный поиск
    for p, regexes in SMART_REGEX.items():
        if p in cols: continue
        best_score, best_col = 0, None
        for c in df.columns:
            ck = _colkey(c)
            score = sum(bool(r.search(ck)) for r in regexes)
            if score > best_score:
                best_score, best_col = score, c
        if best_score > 0:
            cols[p] = best_col
    # дата — обязательна
    if "survey_date" not in cols:
        candidates = ["Дата", "Дата анкетирования", "Комментарий", "Средняя оценка", "№ 1", "№ 2", "№ 3"]
        best, best_hits = None, 0
        n = len(df)
        for c in df.columns:
            try:
                parsed = df[c].map(_parse_date_any)
                hits = int(parsed.notna().sum())
                if hits > best_hits and hits >= max(5, int(0.5 * n)):
                    best, best_hits = c, hits
            except Exception:
                continue
        if best:
            cols["survey_date"] = best
        else:
            raise RuntimeError("В файле не найдена колонка с датой анкетирования.")

    return cols

def _parse_date_any(x) -> date | None:
    if pd.isna(x): return None
    try:
        if isinstance(x, (int, float)):
            return (dt.datetime(1900, 1, 1) + dt.timedelta(days=int(x) - 2)).date()
        elif isinstance(x, str):
            if match := re.match(r"(\d{4})-(\d{2})-(\d{2})", x):
                return date(*map(int, match.groups()))
            elif match := re.match(r"(\d{2})\.(\d{2})\.(\d{4})", x):
                dd, mm, yyyy = map(int, match.groups())
                return date(yyyy, mm, dd)
        return pd.to_datetime(x).date()
    except Exception:
        return None

def to_5_scale(x):
    if isinstance(x, str):
        x = x.replace(',', '.')
    try:
        v = float(x)
        return round(v, 1) if 1 <= v <= 5 else np.nan
    except:
        return np.nan

def normalize_surveys_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    cols = find_columns(df)

    # 1) дата
    out = pd.DataFrame()
    out["date"] = df[cols["survey_date"]].map(_parse_date_any)

    # сервисные
    for k in ("comment", "fio", "booking", "phone", "email"):
        out[k] = df[cols[k]].astype(str) if k in cols else ""

    # 2) значения /5 по параметрам
    for p in PARAM_ORDER:
        if p == "nps_1_5":
            col = cols.get(p)
            out["nps5"] = df[col].map(to_5_scale) if col else np.nan
            continue
        col = cols.get(p)
        v5 = df[col].map(to_5_scale) if col else np.nan
        out[f"{p}5"]  = v5
        out[f"{p}10"] = v5 * 2.0

    # 3) если overall не дан — считаем среднее по доступным шкалам /5 (кроме nps5)
    if out["overall5"].isna().all():
        value_cols5 = [c for c in out.columns if c.endswith("5") and c not in ("nps5",)]
        if value_cols5:
            out["overall5"]  = pd.to_numeric(out[value_cols5], errors="coerce").mean(axis=1)
            out["overall10"] = out["overall5"] * 2.0

    # 4) удаляем строки без даты
    out = out[pd.notna(out["date"])].reset_index(drop=True)
    return out

# =======================
# Недельная агрегация
# =======================
def compute_nps_from_1to5(s: pd.Series) -> Tuple[int,int,float]:
    valid = s.notna().sum()
    if valid == 0: return None, None, None
    prom = int((s ==5).sum())
    detr = int((s <=2).sum())
    nps = round(100 * (prom - detr) / valid, 2)
    return prom, detr, nps

def iso_week_key(d: date) -> str:
    year, week, _ = d.isocalendar()
    return f"{year}-W{week:02d}"

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
            if cnt == 0 and p != "overall":
                continue
            avg5 = float(s.mean()) if cnt > 0 else np.nan
            avg10 = (avg5 * 2.0) if cnt > 0 else np.nan
            responses = total_surveys if p == "overall" else cnt
            rows.append([
                wk, p, responses,
                (None if math.isnan(avg5) else round(avg5, 2)),
                (None if math.isnan(avg10) else round(avg10, 2)),
                None, None, None
            ])

        # NPS (1–5: 1–2 D, 3–4 N, 5 P)
        if "nps5" in wdf.columns:
            v = pd.to_numeric(wdf["nps5"], errors="coerce").where(lambda x: x.between(1, 5))
            valid_count = int(v.notna().sum())
            if valid_count > 0:
                promoters, detractors, nps = compute_nps_from_1to5(v)
                rows.append([wk, "nps", valid_count, None, None, promoters, detractors, nps])

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
