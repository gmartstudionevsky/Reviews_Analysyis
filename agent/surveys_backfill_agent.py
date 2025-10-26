# agent/surveys_backfill_agent.py
# Бэкфилл всей истории анкет в единый лист surveys_raw (1 строка = 1 анкета).
#
# Запуск: python -m agent.surveys_backfill_agent
#
# Что делает:
# 1. Лезет в папку на Google Drive (DRIVE_FOLDER_ID).
# 2. Находит все файлы с анкетами:
#    - Report_dd-mm-yyyy.xlsx
#    - любые XLSX/XLS, где в имени есть "history" или "истор"
# 3. Парсит каждый, нормализует (surveys_core.normalize_surveys_file).
# 4. В таблице SHEETS_HISTORY_ID создаёт лист "surveys_raw", если его нет.
# 5. Достраивает новые строки без дублей (по survey_id).

import os, io, re, json, sys
import pandas as pd
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# импорт ядра нормализации
try:
    from agent.surveys_core import normalize_surveys_file
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(__file__))
    from surveys_core import normalize_surveys_file


# -------------------
# ENV & авторизация
# -------------------
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
        raise RuntimeError("Нет GOOGLE_SERVICE_ACCOUNT_JSON / GOOGLE_SERVICE_ACCOUNT_JSON_CONTENT")
    CREDS = Credentials.from_service_account_file(SA_PATH, scopes=SCOPES)

DRIVE  = build("drive",  "v3", credentials=CREDS)
SHEETS = build("sheets", "v4", credentials=CREDS).spreadsheets()

DRIVE_FOLDER_ID   = os.environ["DRIVE_FOLDER_ID"]
HISTORY_SHEET_ID  = os.environ["SHEETS_HISTORY_ID"]

SURVEYS_SHEET_TAB = "surveys_raw"

SURVEYS_HEADER = [
    "survey_id", "date", "week_key",
    "overall5",
    "fo_checkin5", "clean_checkin5", "room_comfort5",
    "fo_stay5", "its_service5", "hsk_stay5", "breakfast5",
    "atmosphere5", "location5", "value5", "would_return5",
    "nps5",
    "comment",
]


# -------------------
# Работа с Google Sheets
# -------------------
def ensure_tab(spreadsheet_id: str, tab_name: str, header: list[str]):
    """Если листа нет – создаём и записываем заголовок."""
    meta = SHEETS.get(spreadsheetId=spreadsheet_id).execute()
    have_titles = [s["properties"]["title"] for s in meta.get("sheets", [])]
    if tab_name not in have_titles:
        # создаём новый лист
        SHEETS.batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"requests":[{"addSheet":{"properties":{"title":tab_name}}}]}
        ).execute()
        # пишем заголовок
        SHEETS.values().update(
            spreadsheetId=spreadsheet_id,
            range=f"{tab_name}!A1:{chr(64+len(header))}1",
            valueInputOption="RAW",
            body={"values":[header]},
        ).execute()

def gs_read_all_raw() -> pd.DataFrame:
    """Читаем уже загруженные анкеты из surveys_raw (если он вообще есть)."""
    try:
        res = SHEETS.values().get(
            spreadsheetId=HISTORY_SHEET_ID,
            range=f"{SURVEYS_SHEET_TAB}!A1:Z999999"  # с запасом
        ).execute()
        vals = res.get("values", [])
        if not vals or len(vals) < 2:
            return pd.DataFrame(columns=SURVEYS_HEADER)
        header = vals[0]
        data   = vals[1:]
        df = pd.DataFrame(data, columns=header)
        return df
    except Exception:
        # если вкладки ещё нет - вернём пустой DF
        return pd.DataFrame(columns=SURVEYS_HEADER)

def gs_append_rows(rows: list[list[str | float | None]]):
    """Дописать строки в конец surveys_raw."""
    if not rows:
        return
    SHEETS.values().append(
        spreadsheetId=HISTORY_SHEET_ID,
        range=f"{SURVEYS_SHEET_TAB}!A1",
        valueInputOption="RAW",
        body={"values": rows},
    ).execute()


# -------------------
# Работа с Google Drive
# -------------------
WEEKLY_RE   = re.compile(r"^Report_(\d{2})-(\d{2})-(\d{4})\.xlsx?$", re.IGNORECASE)
HISTORY_RE  = re.compile(r"(history|истор)", re.IGNORECASE)

def list_candidate_files(folder_id: str):
    """
    Забрать все XLS/XLSX из папки:
    - которые выглядят как Report_dd-mm-yyyy.xlsx
    - или содержат 'history'/'истор' в названии
    """
    items = []
    page_token = None
    while True:
        resp = DRIVE.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id,name,mimeType,modifiedTime)",
            pageSize=1000,
            pageToken=page_token
        ).execute()
        for f in resp.get("files", []):
            name = f["name"]
            lname = name.lower()
            if lname.endswith(".xlsx") or lname.endswith(".xls"):
                if WEEKLY_RE.match(name) or HISTORY_RE.search(name):
                    items.append((f["id"], f["name"], f.get("modifiedTime","")))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    # упорядочим: сначала history файлы (они, как правило, общие выгрузки), потом недельные по дате
    hist_files   = [it for it in items if HISTORY_RE.search(it[1])]
    weekly_files = [it for it in items if WEEKLY_RE.match(it[1])]
    # weekly отсортируем по имени (дата в имени файла)
    weekly_files.sort(key=lambda x: x[1])
    return hist_files + weekly_files

def drive_download_bytes(file_id: str) -> bytes:
    """Скачать бинарник XLS/XLSX из Google Drive."""
    req = DRIVE.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
    return buf.getvalue()


def pick_sheet_with_surveys(xls: pd.ExcelFile) -> pd.DataFrame:
    """
    Эвристика выбора нужного листа внутри Excel-файла.
    Ищем лист по названию и/или по колонкам.
    """
    preferred_names = ["Оценки гостей", "Оценки", "Ответы", "Анкеты", "Responses"]
    norm_pref = {p.lower() for p in preferred_names}

    # 1) прямое совпадение по имени
    for sh in xls.sheet_names:
        if sh.strip().lower() in norm_pref:
            return pd.read_excel(xls, sh)

    # 2) эвристика по колонкам
    probe_keywords = [
        "Дата", "Дата анкетирования", "Комментарий", "Средняя оценка", "№ 1", "№ 2", "№ 3",
        "Оцените работу службы приёма", "готовность вернуться", "расположение отеля",
    ]

    best_sheet   = None
    best_score   = -1
    for sh in xls.sheet_names:
        try:
            small = pd.read_excel(xls, sh, nrows=1)
        except Exception:
            continue
        cols = [str(c) for c in small.columns]
        score = sum(any(pk.lower() in c.lower() for c in cols) for pk in probe_keywords)
        if score > best_score:
            best_sheet = sh
            best_score = score

    if best_sheet is None:
        # fallback — просто первый лист
        best_sheet = xls.sheet_names[0]

    return pd.read_excel(xls, best_sheet)


# -------------------
# Основная логика бэкфилла
# -------------------
def backfill_all():
    # Убедимся, что лист существует
    ensure_tab(HISTORY_SHEET_ID, SURVEYS_SHEET_TAB, SURVEYS_HEADER)

    # Читаем то, что уже лежит в surveys_raw
    already_df = gs_read_all_raw()
    existing_ids = set(already_df["survey_id"].tolist()) if "survey_id" in already_df.columns else set()

    candidates = list_candidate_files(DRIVE_FOLDER_ID)

    new_rows_total = 0

    for file_id, name, mtime in candidates:
        print(f"[INFO] Обрабатываем файл {name} ...")
        try:
            blob = drive_download_bytes(file_id)
            xls  = pd.ExcelFile(io.BytesIO(blob))
            raw  = pick_sheet_with_surveys(xls)

            norm = normalize_surveys_file(raw)
            if norm.empty:
                print(f"[WARN] {name}: после нормализации пусто")
                continue

            # фильтруем только новые анкеты (по survey_id)
            norm_new = norm[~norm["survey_id"].isin(existing_ids)].copy()
            if norm_new.empty:
                print(f"[INFO] {name}: нет новых анкет, всё уже есть")
                continue

            # добавляем их в existing_ids, чтобы не задублировать далее в этом же прогонах
            for sid in norm_new["survey_id"]:
                existing_ids.add(sid)

            # готовим к апенду в Sheets
            # (каждую строку превращаем в массив значений в порядке SURVEYS_HEADER)
            to_append = []
            for _, r in norm_new.iterrows():
                row_list = [
                    str(r.get("survey_id","")),
                    str(r.get("date","")),
                    str(r.get("week_key","")),
                    _safe_num(r.get("overall5")),
                    _safe_num(r.get("fo_checkin5")),
                    _safe_num(r.get("clean_checkin5")),
                    _safe_num(r.get("room_comfort5")),
                    _safe_num(r.get("fo_stay5")),
                    _safe_num(r.get("its_service5")),
                    _safe_num(r.get("hsk_stay5")),
                    _safe_num(r.get("breakfast5")),
                    _safe_num(r.get("atmosphere5")),
                    _safe_num(r.get("location5")),
                    _safe_num(r.get("value5")),
                    _safe_num(r.get("would_return5")),
                    _safe_num(r.get("nps5")),
                    str(r.get("comment","")),
                ]
                to_append.append(row_list)

            gs_append_rows(to_append)
            new_rows_total += len(to_append)
            print(f"[OK] {name}: добавлено {len(to_append)} новых анкет")

        except Exception as e:
            # не падаем на одном битом файле — просто логируем
            print(f"[ERROR] {name}: {e}")

    print(f"[DONE] Бэкфилл завершён. Новых анкет добавлено: {new_rows_total}")


def _safe_num(x):
    """Приводим число к float с точкой или '' если NaN, чтобы красиво легло в Google Sheets."""
    if x is None:
        return ""
    if isinstance(x, (float, int, np.floating, np.integer)):
        if pd.isna(x):
            return ""
        return float(x)
    try:
        f = float(x)
        if pd.isna(f):
            return ""
        return f
    except Exception:
        return ""


def main():
    backfill_all()


if __name__ == "__main__":
    main()
