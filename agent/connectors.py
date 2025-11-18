# agent/connectors.py
from __future__ import annotations

import os
import base64

from google.oauth2 import service_account
from googleapiclient.discovery import build

# Scopes такие же, как в reviews-агентах
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
SHEETS_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


def b64_to_sa_json_path(env_var: str = "GOOGLE_SERVICE_ACCOUNT_JSON_B64") -> str:
    """
    Декодирует base64-encoded JSON ключ сервис-аккаунта из переменной окружения env_var
    в файл /tmp/sa.json и возвращает путь к этому файлу.
    """
    content_b64 = os.environ.get(env_var) or ""
    if not content_b64:
        raise RuntimeError(f"{env_var} не задан.")
    raw = base64.b64decode(content_b64)
    out_path = "/tmp/sa.json"
    with open(out_path, "wb") as f:
        f.write(raw)
    return out_path


def build_credentials_from_b64(env_var: str = "GOOGLE_SERVICE_ACCOUNT_JSON_B64") -> "service_account.Credentials":
    """
    Строит объект Credentials из base64-encoded JSON ключа в переменной окружения env_var.
    Поведение полностью эквивалентно тому, что было в reviews-агентах.
    """
    sa_path = b64_to_sa_json_path(env_var)
    return service_account.Credentials.from_service_account_file(
        sa_path, scopes=DRIVE_SCOPES + SHEETS_SCOPES
    )


def get_drive_client(creds: "service_account.Credentials"):
    """
    Возвращает клиент Google Drive API v3 с отключённым discovery cache
    (так же, как было в reviews-агентах).
    """
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def get_sheets_client(creds: "service_account.Credentials"):
    """
    Возвращает корневой клиент Google Sheets API v4.
    Используем его так же, как раньше:
      service.spreadsheets().values().get(...)
    """
    return build("sheets", "v4", credentials=creds, cache_discovery=False)
