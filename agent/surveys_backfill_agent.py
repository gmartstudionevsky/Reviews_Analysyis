# agent/surveys_backfill_agent.py
#
# Полная пересборка истории анкет TL: Marketing.
#
# Что делает:
#   1. Идёт в папку DRIVE_FOLDER_ID в Google Drive.
#   2. Собирает:
#        - Report_history.xlsx (агрегированная история)
#        - все Report_DD-MM-YYYY.xlsx (недельные выгрузки)
#   3. Для каждого файла:
#        - читает лист "Оценки гостей" (или "Reviews" в старых файлах)
#        - прогоняет через parse_and_aggregate_weekly() из surveys_core
#          -> получаем df_week (week_key, param, surveys_total, answered, avg5,
#             promoters, detractors, nps_answers, nps_value)
#   4. Сливает все результаты:
#        - ключ = (week_key, param)
#        - если одна и та же неделя+показатель встречается в нескольких файлах,
#          берём последний по дате файла (т.е. более новый перезаписывает старый)
#   5. Сортирует недели по возрастанию, параметры — по нашему порядку PARAM_ORDER.
#   6. Полностью ПЕРЕЗАПИСЫВАЕТ лист surveys_history в Google Sheets:
#        - сначала шапка
#        - затем все строки.
#
# ЭТОТ СКРИПТ нужно запускать вручную (через отдельный workflow_dispatch),
# а не по расписанию. В еженедельном отчёте backfill не гоняем.

import os, io, re, sys, json, datetime as dt
from datetime import date
import pandas as pd
import numpy as np

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# импортируем ядро обработки анкет
try:
    from agent.surveys_core import (
        parse_and_aggregate_weekly,
        SURVEYS_TAB,
        PARAM_ORDER,
    )
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(__file__))
    from surveys_core import (
        parse_and_aggregate_weekly,
        SURVEYS_TAB,
        PARAM_ORDER,
    )

# =========================
# ENV & Google API clients
# =========================
SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/spreadsheets",
]

SA_PATH    = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
SA_CONTENT = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON_CONTENT")

if SA_CONTENT and SA_CONTENT.strip().startswith("{"):
    CREDS = Credentials.from_service_account_info(json.loads(SA_CONTENT), scopes=SCOPES)
else:
    if not SA_PATH:
        raise RuntimeError("No GOOGLE_SERVICE_ACCOUNT_JSON or GOOGLE_SERVICE_ACCOUNT_JSON_CONTENT provided.")
    CREDS = Credentials.from_service_account_file(SA_PATH, scopes=SCOPES)

DRIVE  = build("drive",  "v3", credentials=CREDS)
SHEETS = build("sheets", "v4", credentials=CREDS).spreadsheets()

DRIVE_FOLDER_ID   = os.environ["DRIVE_FOLDER_ID"]
HISTORY_SHEET_ID  = os.environ["SHEETS_HISTORY_ID"]

# =========================
# Настройки форматов
# =========================

# файлы вида Report_19-10-2025.xlsx
WEEKLY_RE   = re.compile(r"^Report_(\d{2})-(\d{2})-(\d{4})\.xlsx$", re.IGNORECASE)
# исторический слепок
HISTORY_RE  = re.compile(r"^Report_history\.xlsx$", re.IGNORECASE)

# Целевая шапка листа surveys_history (без avg10)
SURVEYS_HEADER = [
    "week_key",
    "param",
    "surveys_total",
    "answered",
    "avg5",
    "promoters",
    "detractors",
    "nps_answers",
    "nps_value",
]

def _week_sort_key(week_key: str) -> int:
    """
    Сортируем недели как YYYY*100 + WW.
    Например '2025-W02' -> 202502,
             '2025-W42' -> 202542.
    """
    try:
        y, w = str(week_key).split("-W")
        return int(y) * 100 + int(w)
    except Exception:
        return 0

# =========================
# Google Drive helpers
# =========================

def drive_list_all_reports():
    """
    Возвращает список файлов (file_id, filename, sort_date) из папки DRIVE_FOLDER_ID,
    которые выглядят как:
      - Report_history.xlsx  (sort_date очень ранняя, чтобы он шёл первым)
      - Report_DD-MM-YYYY.xlsx (sort_date = эта дата)
    Мы сортируем по sort_date по возрастанию, так что более новые файлы
    будут перезаписывать ранние данные той же недели.
    """
    res = DRIVE.files().list(
        q=f"'{DRIVE_FOLDER_ID}' in parents and mimeType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' and trashed=false",
        fields="files(id,name,modifiedTime)",
        pageSize=200,
    ).execute()

    out = []
    for f in res.get("files", []):
        nm = f["name"]

        # 1) Исторический файл без даты
        if HISTORY_RE.match(nm):
            sort_date = dt.date(2000,1,1)
            out.append((f["id"], nm, sort_date))
            continue

        # 2) Регулярные файлы с датой
        m = WEEKLY_RE.match(nm)
        if m:
            dd, mm, yyyy = m.groups()
            try:
                sort_date = dt.date(int(yyyy), int(mm), int(dd))
            except Exception:
                # если не удалось адекватно распарсить дату — даём очень раннюю, чтобы не убить историю
                sort_date = dt.date(2000,1,2)
            out.append((f["id"], nm, sort_date))

    # Сначала самые старые, потом всё свежее.
    # Важно: чем ПОЗЖЕ файл, тем ПОЗЖЕ он пройдёт цикл и "переедет"
    # недели, найденные ранее.
    out.sort(key=lambda x: x[2])
    return out

def drive_download(file_id: str) -> bytes:
    req = DRIVE.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
    return buf.getvalue()

# =========================
# Google Sheets helpers
# =========================

def ensure_tab(spreadsheet_id: str, tab_name: str, header: list[str]):
    """
    Убедиться, что лист есть и в A1:I1 лежит корректная шапка.
    Мы будем перезаписывать лист полностью, включая шапку.
    """
    meta = SHEETS.get(spreadsheetId=spreadsheet_id).execute()
    tabs = [s["properties"]["title"] for s in meta.get("sheets", [])]

    if tab_name not in tabs:
        # создаём лист, если его не было
        SHEETS.batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"requests":[{"addSheet":{"properties":{"title":tab_name}}}]}
        ).execute()

    # пишем шапку в A1:I1 (в любом случае обновляем)
    SHEETS.values().update(
        spreadsheetId=spreadsheet_id,
        range=f"{tab_name}!A1:I1",
        valueInputOption="RAW",
        body={"values":[header]},
    ).execute()

def write_full_history(df_all: pd.DataFrame):
    """
    Полностью очищает лист (кроме шапки, которую мы только что перезаписали)
    и пишет туда все строки по порядку.
    """
    # удаляем всё, что было ниже заголовка
    SHEETS.values().clear(
        spreadsheetId=HISTORY_SHEET_ID,
        range=f"{SURVEYS_TAB}!A2:I",
    ).execute()

    if df_all.empty:
        print("[INFO] История пустая, нечего писать.")
        return

    # раскладываем df_all в list[list], в порядке SURVEYS_HEADER
    rows = []
    for _, r in df_all.iterrows():
        rows.append([
            r["week_key"],
            r["param"],
            int(r["surveys_total"]) if not pd.isna(r["surveys_total"]) else 0,
            int(r["answered"])      if not pd.isna(r["answered"])      else 0,
            (None if pd.isna(r["avg5"])          else float(r["avg5"])),
            (None if pd.isna(r["promoters"])     else int(r["promoters"])),
            (None if pd.isna(r["detractors"])    else int(r["detractors"])),
            (None if pd.isna(r["nps_answers"])   else int(r["nps_answers"])),
            (None if pd.isna(r["nps_value"])     else float(r["nps_value"])),
        ])

    SHEETS.values().append(
        spreadsheetId=HISTORY_SHEET_ID,
        range=f"{SURVEYS_TAB}!A2",
        valueInputOption="RAW",
        body={"values": rows},
    ).execute()

    print(f"[INFO] Загружено {len(rows)} строк в {SURVEYS_TAB}.")


# =========================
# Основная логика backfill
# =========================

def main():
    # Шаг 1. Собрать список всех источников
    reports = drive_list_all_reports()
    if not reports:
        raise RuntimeError("Не нашли ни одного файла Report_*.xlsx в папке DRIVE_FOLDER_ID")

    # Копим сюда все week_key+param, с приоритетом более "свежих" файлов
    # (то есть в случае конфликта мы просто перезапишем ключ последним файлом)
    weeks_map = {}

    # Шаг 2. Пройти по каждому файлу в порядке sort_date (от старых к новым)
    for file_id, fname, sort_date in reports:
        print(f"[INFO] читаем {fname} ({sort_date})")
        blob = drive_download(file_id)
        xls = pd.ExcelFile(io.BytesIO(blob))

        # Нам нужно то, что содержит анкеты.
        if "Оценки гостей" in xls.sheet_names:
            raw = pd.read_excel(xls, sheet_name="Оценки гостей")
        else:
            # fallback для старых исторических файлов
            raw = pd.read_excel(xls, sheet_name="Reviews")

        # Нормализация+агрегация недели с учётом НОВОЙ логики (без avg10)
        _, agg_week = parse_and_aggregate_weekly(raw)
        # agg_week:
        # week_key, param, surveys_total, answered, avg5,
        # promoters, detractors, nps_answers, nps_value

        for _, row in agg_week.iterrows():
            key = (str(row["week_key"]), str(row["param"]))
            weeks_map[key] = row.to_dict()

    # Шаг 3. Превратить weeks_map обратно в DataFrame
    if not weeks_map:
        raise RuntimeError("После агрегации не осталось строк (скорее всего выгрузки пустые).")

    df_all = pd.DataFrame(list(weeks_map.values()))

    # Добавим сортировку:
    #  - сначала недели по возрастанию,
    #  - внутри недели — по нашему порядку PARAM_ORDER
    def param_sort_key(p):
        try:
            return PARAM_ORDER.index(p)
        except ValueError:
            return 999

    df_all["__week_sort"]  = df_all["week_key"].map(_week_sort_key)
    df_all["__param_sort"] = df_all["param"].map(param_sort_key)

    df_all = (
        df_all.sort_values(["__week_sort", "__param_sort"])
              .drop(columns=["__week_sort", "__param_sort"])
              .reset_index(drop=True)
    )

    # Убедимся, что порядок колонок соответствует нашей шапке
    df_all = df_all[[
        "week_key",
        "param",
        "surveys_total",
        "answered",
        "avg5",
        "promoters",
        "detractors",
        "nps_answers",
        "nps_value",
    ]]

    # Шаг 4. Перезаписываем лист в Google Sheets
    ensure_tab(HISTORY_SHEET_ID, SURVEYS_TAB, SURVEYS_HEADER)
    write_full_history(df_all)

    print("[INFO] Backfill завершён успешно.")

if __name__ == "__main__":
    main()
