# agent/surveys_core.py
# Нормализация "сырых" анкет TL: Marketing в единую структуру.
# Каждая строка = одна анкета гостя.

from __future__ import annotations
import re, hashlib
from datetime import date
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


# ==== Маппинг названий колонок из выгрузки -> наши внутренние ключи ====
# Мы приводим их к коротким машиночитаемым именам.
# Формат: наш_ключ : [возможные названия колонок в выгрузке]
COLUMN_ALIASES: Dict[str, List[str]] = {
    # служебные
    "fio": [
        "ФИО", "Имя", "Имя гостя", "Ф.И.О", "Фамилия Имя",
    ],
    "booking": [
        "Номер брони", "Бронирование", "Бронь", "Номер заказа", "Номер резервации",
    ],
    "phone": ["Телефон", "Тел.", "Номер телефона", "Контактный номер"],
    "email": ["Email", "E-mail", "Почта", "Адрес электронной почты"],
    "comment": ["Комментарий гостя", "Комментарий", "Отзыв", "Комментарий клиента", "Ваши комментарии"],
    "survey_date": [
        "Дата анкетирования", "Дата прохождения опроса", "Дата заполнения",
        "Дата и время", "Дата опроса", "Дата анкеты", "Дата", "Дата обращения",
    ],

    # общее впечатление
    "overall": [
        "Средняя оценка гостя", "Итоговая оценка", "Общая оценка", "Общая удовлетворенность",
    ],

    # блок 1: заезд / заселение
    "fo_checkin": [
        "№ 1.1 Оцените работу службы приёма и размещения при заезде",
        "1.1 прием и размещение при заезде",
        "Оцените работу службы приёма и размещения при заезде",
        "Служба приёма при заселении",
    ],
    "clean_checkin": [
        "№ 1.2 Оцените чистоту номера при заезде",
        "1.2 чистота при заезде",
        "Чистота номера при заезде",
    ],
    "room_comfort": [
        "№ 1.3 Оцените комфорт и оснащение номера",
        "1.3 комфорт и оснащение номера",
        "Комфорт и оснащение номера",
        "Комфорт номера",
    ],

    # блок 2: проживание
    "fo_stay": [
        "№ 2.1 Оцените работу службы приёма и размещения во время проживания",
        "2.1 прием и размещение во время проживания",
        "Служба приёма во время проживания",
        "Служба размещения (проживание)",
    ],
    "its_service": [
        "№ 2.2 Оцените работу технической службы",
        "2.2 техническая служба",
        "Работа технической службы",
        "Техническая служба",
    ],
    "hsk_stay": [
        "№ 2.3 Оцените уборку номера во время проживания",
        "2.3 уборка во время проживания",
        "Уборка номера во время проживания",
        "Housekeeping во время проживания",
    ],
    "breakfast": [
        "№ 2.4 Оцените завтраки",
        "2.4 завтраки",
        "Завтраки",
        "Качество завтрака",
    ],

    # блок 3: отель в целом
    "atmosphere": [
        "№ 3.1 Оцените атмосферу в отеле",
        "3.1 атмосфера",
        "Атмосфера в отеле",
    ],
    "location": [
        "№ 3.2 Оцените расположение отеля",
        "3.2 расположение",
        "Расположение отеля",
        "Локация отеля",
    ],
    "value": [
        "№ 3.3 Оцените соотношение цены и качества",
        "3.3 цена/качество",
        "Соотношение цены и качества",
        "Цена / качество",
    ],
    "would_return": [
        "№ 3.4 Хотели бы вы вернуться в ARTSTUDIO Nevsky?",
        "3.4 вернулись бы",
        "Готовность вернуться",
        "Вернулись бы к нам снова",
    ],

    # NPS-вопрос
    "nps5": [
        "№ 3.5 Оцените вероятность того, что вы порекомендуете нас друзьям и близким (по шкале от 1 до 5)",
        "3.5 nps 1-5",
        "nps (1-5)",
        "nps 1-5",
        "Готовность рекомендовать (1-5)",
    ],
}


# ===== Нормализация имён столбцов =====
def _norm_name_for_match(s: str) -> str:
    """Опускаем регистр, убираем nbsp, лишнюю пунктуацию и дубли пробелов."""
    t = str(s).replace("\u00a0", " ").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\sа-яё0-9]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def _find_col(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    """
    Пытаемся найти реальное имя колонки в df, соответствующее одному из вариантов aliases.
    Делаем несколько проходов: точное совпадение после нормализации, подстрока и т.д.
    """
    lowmap = {_norm_name_for_match(c): c for c in df.columns}

    # точное совпадение (после нормализации)
    for cand in aliases:
        key = _norm_name_for_match(cand)
        if key in lowmap:
            return lowmap[key]

    # подстрока
    for cand in aliases:
        key = _norm_name_for_match(cand)
        for lk, orig in lowmap.items():
            if key and key in lk:
                return orig

    # все слова встречаются
    for cand in aliases:
        words = [w for w in _norm_name_for_match(cand).split() if len(w) > 1]
        for lk, orig in lowmap.items():
            if all(w in lk for w in words):
                return orig

    return None


# ===== Парсинг отдельной оценки в шкалу /5 =====
def _to_5_scale(val) -> float:
    """
    Универсально приводим оценку гостя к шкале /5.
    Поддерживаются форматы:
    - "5", "4,5", "4.0"
    - "9.0" (считаем что это /10 -> делим на 2)
    - "80" (считаем как % -> делим на 20)
    - пусто / мусор -> NaN
    """
    if val is None:
        return np.nan
    s = str(val).strip().replace(",", ".")
    if s in ("", "-", "–", "—"):
        return np.nan
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if not m:
        return np.nan
    v = float(m.group(1))

    # если выглядит как классическая 1..5
    if 0 <= v <= 5:
        return v
    # часто встречается 0..10 → делим на 2
    if 0 <= v <= 10:
        return v / 2.0
    # иногда % или 0..100 → делим на 20 (100 => 5.0)
    if 0 <= v <= 100:
        return v / 20.0

    return np.nan


# ===== Парсинг даты анкеты =====
def _parse_date_any(x) -> Optional[date]:
    try:
        d = pd.to_datetime(x, errors="coerce", dayfirst=True)
        if pd.isna(d):
            return None
        return d.date()
    except Exception:
        return None


# ===== Преобразование "сырого" Excel-листа с анкетами в нормализованный датафрейм =====
def normalize_surveys_file(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Вход: df_raw = DataFrame из XLSX (лист с анкетами).
    Выход: DataFrame с колонками:
      survey_id (строка хэша)
      date (YYYY-MM-DD)
      week_key (YYYY-W##, ISO неделя)
      overall5, fo_checkin5, clean_checkin5, room_comfort5,
      fo_stay5, its_service5, hsk_stay5, breakfast5,
      atmosphere5, location5, value5, would_return5,
      nps5,
      comment
    Все оценки в шкале /5 (float). nps5 — шкала 1–5 (как дал гость).
    """

    df = df_raw.copy()

    # 1) сопоставляем колонки
    colmap: Dict[str, Optional[str]] = {}
    for key, alias_list in COLUMN_ALIASES.items():
        hit = _find_col(df, alias_list)
        colmap[key] = hit  # может быть None, если в выгрузке такого столбца нет

    # 2) достаем дату анкетирования (обязательное поле)
    date_col = colmap.get("survey_date")
    if not date_col:
        # fallback эвристика: ищем колонку, где как минимум половина значений парсится как дата
        best_name = None
        best_hits = 0
        n = len(df)
        for c in df.columns:
            parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
            hits = int(parsed.notna().sum())
            if hits > best_hits and hits >= max(3, int(0.5 * n)):
                best_name = c
                best_hits = hits
        if best_name:
            date_col = best_name
        else:
            raise RuntimeError("Не нашли колонку с датой анкетирования 😬")

    out = pd.DataFrame()
    out["date"] = df[date_col].map(_parse_date_any)

    # 3) комментарий (оставим как есть)
    if colmap.get("comment"):
        out["comment"] = df[colmap["comment"]].astype(str)
    else:
        out["comment"] = ""

    # 4) технические поля гостя (нужны только для уникальности)
    tmp_fio     = df[colmap["fio"]].astype(str)      if colmap.get("fio")      else ""
    tmp_booking = df[colmap["booking"]].astype(str)  if colmap.get("booking")  else ""
    tmp_phone   = df[colmap["phone"]].astype(str)    if colmap.get("phone")    else ""
    tmp_email   = df[colmap["email"]].astype(str)    if colmap.get("email")    else ""

    # 5) оценки → шкала /5
    def grab_to5(key: str):
        if not colmap.get(key):
            return np.nan
        return df[colmap[key]].map(_to_5_scale)

    out["overall5"]       = grab_to5("overall")
    out["fo_checkin5"]    = grab_to5("fo_checkin")
    out["clean_checkin5"] = grab_to5("clean_checkin")
    out["room_comfort5"]  = grab_to5("room_comfort")
    out["fo_stay5"]       = grab_to5("fo_stay")
    out["its_service5"]   = grab_to5("its_service")
    out["hsk_stay5"]      = grab_to5("hsk_stay")
    out["breakfast5"]     = grab_to5("breakfast")
    out["atmosphere5"]    = grab_to5("atmosphere")
    out["location5"]      = grab_to5("location")
    out["value5"]         = grab_to5("value")
    out["would_return5"]  = grab_to5("would_return")
    out["nps5"]           = grab_to5("nps5")  # эта колонка уже 1..5 от гостя

    # 6) неделя (ISO-неделя) для быстрой фильтрации потом
    def _week_key(d: Optional[date]) -> str:
        if d is None or pd.isna(d):
            return ""
        iso = d.isocalendar()  # (year, week, weekday)
        return f"{iso.year}-W{iso.week:02d}"

    out["week_key"] = out["date"].map(_week_key)

    # 7) делаем стабильный идентификатор строки анкеты (для дедуплика в Google Sheet)
    # используем дату + бронь + телефон + email + общий комментарий (это почти наверняка уникально)
    def _mk_id(row):
        raw_id = "|".join([
            str(row.get("date", "")),
            str(row.get("comment", "")).strip(),
            # персональные поля тоже участвуют в хэше, но мы их не сохраняем в таблицу в открытом виде
            # чтобы одинаковые анкеты не грузились дважды:
            str(tmp_fio[row.name]),
            str(tmp_booking[row.name]),
            str(tmp_phone[row.name]),
            str(tmp_email[row.name]),
        ])
        return hashlib.sha1(raw_id.encode("utf-8")).hexdigest()

    out["survey_id"] = out.apply(_mk_id, axis=1)

    # 8) выбрасываем строки без даты (это невалидные строки выгрузки)
    out = out[ out["date"].notna() ].reset_index(drop=True)

    # финальный порядок колонок
    final_cols = [
        "survey_id", "date", "week_key",
        "overall5",
        "fo_checkin5", "clean_checkin5", "room_comfort5",
        "fo_stay5", "its_service5", "hsk_stay5", "breakfast5",
        "atmosphere5", "location5", "value5", "would_return5",
        "nps5",
        "comment",
    ]
    return out[final_cols]
