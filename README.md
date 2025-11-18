Добавьте сервис-аккаунт в доступ к папке (Viewer) и Google Sheet (Editor).

В репозитории должны быть настроены Secrets (Repo → Settings → Secrets → Actions):

- `GOOGLE_SERVICE_ACCOUNT_JSON_B64` — base64 всего JSON ключа сервис-аккаунта.
- `DRIVE_FOLDER_ID` — ID папки на Google Drive с файлами отчётов.
- `SHEETS_HISTORY_ID` — ID Google Sheet с историей.
- `RECIPIENTS` — список email-адресов получателей отчёта, через запятую.
- `SMTP_USER` — логин SMTP (например, Gmail).
- `SMTP_PASS` — пароль/ключ SMTP.
- `SMTP_FROM` — адрес отправителя (обычно совпадает с SMTP_USER).

Запуск по расписанию:
- еженедельные отчёты — по понедельникам (см. cron в `.github/workflows/*_weekly_report.yml`);
- бэкфиллы — через ручной запуск (Actions → Run workflow).
