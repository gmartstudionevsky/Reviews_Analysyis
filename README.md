Добавьте сервис-аккаунт в доступ к папке (Viewer) и Google Sheet (Editor).

Secrets (Repo → Settings → Secrets → Actions):

GOOGLE_SERVICE_ACCOUNT_JSON_B64 — base64 всего JSON ключа (пример macOS: base64 -i sa.json | pbcopy; Linux: base64 -w0 sa.json | xclip).

DRIVE_FOLDER_ID = 1zUpTnNNiyveqnfVdxzZ91_Ph4Y-xMQmb

SHEETS_HISTORY_ID = 1btPmgxOMMNZVimPZ_6cICkC1P8dZtFMKRqAoCFgep5k

RECIPIENTS = gm@artstudionevsky.ru

SMTP_USER = gm.artstudionevsky@gmail.com

SMTP_PASS = hmkvdcccepolwtes ← без пробелов

SMTP_FROM = gm.artstudionevsky@gmail.com

Запуск по расписанию: пн 12:00 МСК; ручной запуск — вкладка Actions → Run workflow.
