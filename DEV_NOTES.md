# Reviews_Analysyis — DEV NOTES / ARCHITECTURE

Этот файл фиксирует ключевую архитектуру проекта и инварианты, которые **нельзя ломать** при дальнейшей оптимизации.

## 1. Общая структура проекта

Проект состоит из двух «линий» обработки данных:

- **Surveys (анкеты TL: Marketing)**:
  - `agent/surveys_core.py` — парсинг и агрегация анкет.
  - `agent/surveys_weekly_report_agent.py` — еженедельный отчёт + почта.
  - `agent/surveys_backfill_agent.py` — бэкфилл истории анкет.
- **Reviews (текстовые отзывы)**:
  - `agent/reviews_io.py` — парсинг Excel с отзывами.
  - `agent/reviews_core.py` — лингвистический анализ и метрики по отзывам.
  - `agent/reviews_weekly_report_agent.py` — еженедельный отчёт + почта.
  - `agent/reviews_backfill_agent.py` — бэкфилл истории отзывов.

Общие модули:

- `agent/metrics_core.py` — работа с датами, неделями и периодами (week / MTD / QTD / YTD / All).
- `agent/lexicon_module.py` — лексикон и правила для анализа текстов отзывов.
- `agent/connectors.py` — единая точка создания Google Credentials и клиентов Drive/Sheets.

Запуск из GitHub Actions:

- Workflows в `.github/workflows/*.yml`:
  - `surveys_weekly_report.yml`, `surveys_backfill.yml`
  - `reviews_weekly_report.yml`, `reviews_backfill.yml`

## 2. Формат недель и ключей

### 2.1. week_key

Во всех модулях используется формат недели:

- `week_key` строго в виде `YYYY-W##`, например: `2025-W06`.

Это критичный инвариант — его нельзя менять:

- `metrics_core`, `reviews_core`, `surveys_core` и оба weekly-агента ожидают именно такой формат.
- В истории (Google Sheets) неделя тоже хранится в таком виде.

### 2.2. review_id / review_key

Для отзывов используются стабильные идентификаторы:

- `review_id` — хэш по `(source_code, author, date, normalized_text)`:
  - формируется в `reviews_io.make_review_id(...)`.
  - зависит только от содержимого отзыва и автора.
- `review_key` — то же самое, что `review_id`, только в таблице истории (`reviews_history`).

Инвариант: **формула генерации `review_id` / `review_key` не должна меняться**, иначе:

- backfill начнёт плодить дубликаты в `reviews_history`;
- weekly-агент перестанет корректно определять, какие отзывы уже были записаны.

## 3. Исторические таблицы в Google Sheets

Есть один общий Google Sheet `SHEETS_HISTORY_ID` с несколькими вкладками.

### 3.1. surveys_history

- Вкладка: `surveys_history`.
- Заполняется:
  - полностью — `surveys_backfill_agent.py` (бэкфилл);
  - по одной неделе — `surveys_weekly_report_agent.py` (upsert).

Структура (колонки) — инвариант. Базовые поля:

- `week_key` — неделя в формате `YYYY-W##`.
- `param` — код параметра (`overall`, `spir_checkin`, `comfort`, `engineering`, …, `nps`).
- `surveys_total` — всего анкет за неделю.
- `answered` — количество ответивших по параметру.
- `avg5` — средняя оценка по 5-балльной шкале.
- Для NPS:
  - `promoters`, `detractors`, `nps_answers`, `nps_value` (в п.п.).

Любые изменения колонок надо делать осознанно и одновременно в:

- `surveys_core.py` (агрегация),
- `surveys_backfill_agent.py`,
- `surveys_weekly_report_agent.py`.

### 3.2. reviews_history

- Вкладка: `reviews_history`.
- Заполняется:
  - через `reviews_backfill_agent.py` (bulk-бэкфилл),
  - через `reviews_weekly_report_agent.py` (upsert только текущей недели).

Ключевые поля (минимальный набор):

- `date` — дата отзыва.
- `iso_week` / `week_key` — неделя (формат `YYYY-W##`).
- `source` — код источника (`booking`, `yandex`, `google_maps`, …).
- `lang` — язык (`ru`, `en`, `tr`, …).
- `rating10` — оценка по 10-балльной шкале.
- `sentiment_score`, `sentiment_overall`.
- `aspects` — сериализованный список аспектов.
- `topics` — сериализованный список тем/подтем.
- `has_response` — есть ли ответ на отзыв.
- `review_key` — стабильный идентификатор (см. выше).
- `text_trimmed` — обрезанный текст (summary для таблицы).
- `ingested_at` — дата/время загрузки в историю.

Инварианты:

- `review_key` хранится в одной и той же колонке (используется backfill-агентом для идемпотентности).
- Структура колонок должна соответствовать ожиданиям `_parse_history_df` в `reviews_weekly_report_agent.py` и функциям записи в `reviews_backfill_agent.py`.

## 4. Связи между модулями

### 4.1. Surveys-линия

Поток данных:

1. Google Drive:
   - файлы `Report_DD-MM-YYYY.xlsx` + исторический `Report_history.xlsx`.
2. `surveys_backfill_agent.py`:
   - читает все файлы,
   - через `surveys_core.parse_and_aggregate_weekly` собирает агрегаты по неделям,
   - полностью перезаписывает `surveys_history`.
3. `surveys_weekly_report_agent.py`:
   - читает **последний** `Report_*.xlsx` с Диска,
   - через `parse_and_aggregate_weekly` считает текущую неделю,
   - обновляет только строки текущей недели в `surveys_history`,
   - строит агрегаты (week / MTD / QTD / YTD / All),
   - формирует HTML-письмо и отправляет его по SMTP.

### 4.2. Reviews-линия

Поток данных:

1. Google Drive:
   - файлы `Reviews_DD-MM-YYYY.xls` и/или `reviews_YYYY-MM-DD.xls`,
   - возможен агрегированный файл `reviews_YYYY-YY.*` для бэкфилла.
2. `reviews_backfill_agent.py`:
   - выбирает либо конкретный файл (`BACKFILL_FILE`), либо агрегированный `reviews_YYYY-YY.*`,
   - читает через `reviews_io.read_reviews_xls`,
   - строит `ReviewRecordInput` через `reviews_io.df_to_inputs`,
   - анализирует тексты через `reviews_core` + `lexicon_module`,
   - пишет новые строки в `reviews_history`, не создавая дублей по `review_key`.
3. `reviews_weekly_report_agent.py`:
   - выбирает лучший файл под якорную неделю (аргумент `WEEK_KEY` или последняя завершившаяся неделя),
   - читает и анализирует отзывы (как в backfill),
   - объединяет историю из `reviews_history` с текущей неделей (без дублей по `review_id`),
   - считает метрики по периодам (week / MTD / QTD / YTD / All),
   - считает влияния аспектов,
   - формирует HTML-письмо + вложения (CSV + графики) и отправляет по SMTP.

## 5. Google API и креденшелсы

Все агенты теперь используют единый модуль:

- `agent/connectors.py`

Важные функции:

- `build_credentials_from_b64(env_var="GOOGLE_SERVICE_ACCOUNT_JSON_B64")`:
  - используется в reviews-агентах;
  - ожидание: в `env` есть base64-encoded JSON сервис-аккаунта.
- `build_credentials_from_env(path_var="GOOGLE_SERVICE_ACCOUNT_JSON", content_var="GOOGLE_SERVICE_ACCOUNT_JSON_CONTENT")`:
  - используется в surveys-агентах;
  - либо путь к JSON, либо сырой JSON в переменных окружения.
- `get_drive_client(creds)` → клиент Google Drive API v3.
- `get_sheets_client(creds)` → клиент Google Sheets API v4.

Инварианты:

- Scopes: `["https://www.googleapis.com/auth/drive.readonly", "https://www.googleapis.com/auth/spreadsheets"]`.
- Поведение работы с креденшелсами при оптимизации менять нельзя без одновременного обновления всех агентов и CI.

## 6. Важные ENV-переменные

Общий набор:

- `DRIVE_FOLDER_ID` — ID папки на Google Drive с отчётами.
- `SHEETS_HISTORY_ID` — ID Google Sheet с историей.

Для reviews-агентов:

- `GOOGLE_SERVICE_ACCOUNT_JSON_B64` — base64 JSON сервис-аккаунта.
- `RECIPIENTS` — список email-адресов (через запятую).
- `SMTP_USER`, `SMTP_PASS`, `SMTP_FROM`, `SMTP_HOST`, `SMTP_PORT`.
- `WEEK_KEY` (опционально) — якорная неделя в формате `YYYY-W##`.
- `DRY_RUN` — `"true"` / `"false"`: не отправлять письмо и/или не писать в историю.

Для surveys-агентов:

- `GOOGLE_SERVICE_ACCOUNT_JSON` или `GOOGLE_SERVICE_ACCOUNT_JSON_CONTENT` — креденшелсы сервис-аккаунта.
- `RECIPIENTS`, `SMTP_*` — для почты.
- `DRY_RUN` — `"true"` / `"false"` (для weekly-отчёта и backfill).

## 7. Что важно не ломать при дальнейших изменениях

1. Формат `week_key` (`YYYY-W##`).
2. Формулу и смысл `review_id` / `review_key`.
3. Структуру вкладок `surveys_history` и `reviews_history` в Google Sheets:
   - имена вкладок;
   - базовый набор колонок, особенно `review_key` и `week_key`.
4. Публичные функции:
   - из `metrics_core`: `iso_week_monday`, `period_ranges_for_week`, `build_history`, `build_sources_history`;
   - из `surveys_core`: вся внешняя API для weekly/backfill-агентов;
   - из `reviews_core`: `analyze_reviews_bulk`, `build_reviews_dataframe`, `build_aspects_dataframe`, `compute_aspect_impacts`, `slice_periods`, `build_source_pivot`.
5. Интерфейс `LexiconProtocol` и API `lexicon_module` (используется `reviews_core`).
6. Ожидаемые ENV-переменные — особенно те, которые проверяются через `_require_env` или явные `RuntimeError`.

Если что-то из этого меняется — нужно одновременно обновлять:
- код агентов;
- схемы Google Sheets;
- (при необходимости) документацию и настройки GitHub Actions.
