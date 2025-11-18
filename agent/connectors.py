# agent/connectors.py
from __future__ import annotations

import os
import base64
import json

from google.oauth2 import service_account
from googleapiclient.discovery import build

# Те же права, что и раньше в агентах:
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
SHEETS_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
ALL_SCOPES = DRIVE_SCOPES + SHEETS_SCOPES


def b64_to_sa_json_path(env_var: str = "GOOGLE_SERVICE_ACCOUNT_JSON_B64") -> str:
    """
    Декодирует base64-encoded JSON ключ сервис-аккаунта из переменной окружения env_var
    в файл /tmp/sa.json и возвращает путь к этому файлу.
    Используется в reviews-агентах через build_credentials_from_b64().
    """
    content_b64 = os.environ.get(env_var) or ""
    if not content_b64:
        raise RuntimeError(f"{env_var} не задан.")
    raw = base64.b64decode(content_b64)
    out_path = "/tmp/sa.json"
    with open(out_path, "wb") as f:
        f.write(raw)
    return out_path


def build_credentials_from_b64(
    env_var: str = "GOOGLE_SERVICE_ACCOUNT_JSON_B64",
) -> "service_account.Credentials":
    """
    Строит Credentials из base64-ключа (как это уже делает reviews_weekly/backfill).
    Поведение остаётся прежним, только вынесено в общий модуль.
    """
    sa_path = b64_to_sa_json_path(env_var)
    return service_account.Credentials.from_service_account_file(
        sa_path, scopes=ALL_SCOPES
    )


def build_credentials_from_env(
    path_var: str = "GOOGLE_SERVICE_ACCOUNT_JSON",
    content_var: str = "GOOGLE_SERVICE_ACCOUNT_JSON_CONTENT",
) -> "service_account.Credentials":
    """
    Строит Credentials из:
      - либо GOOGLE_SERVICE_ACCOUNT_JSON_CONTENT (сырой JSON),
      - либо GOOGLE_SERVICE_ACCOUNT_JSON (путь к файлу).
    Это точная копия логики, которая была в surveys-агентах.
    """
    sa_path = os.environ.get(path_var)
    sa_content = os.environ.get(content_var)

    if sa_content and sa_content.strip().startswith("{"):
        info = json.loads(sa_content)
        return service_account.Credentials.from_service_account_info(
            info, scopes=ALL_SCOPES
        )

    if not sa_path:
        raise RuntimeError(f"No {path_var} or {content_var} provided.")

    return service_account.Credentials.from_service_account_file(
        sa_path, scopes=ALL_SCOPES
    )


def get_drive_client(creds: "service_account.Credentials"):
    """
    Возвращает клиент Google Drive API v3 с отключённым discovery cache.
    """
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def get_sheets_client(creds: "service_account.Credentials"):
    """
    Возвращает корневой клиент Google Sheets API v4.
    В коде дальше можно вызывать sheets.spreadsheets() или sheets.spreadsheets().values().
    """
    return build("sheets", "v4", credentials=creds, cache_discovery=False)
