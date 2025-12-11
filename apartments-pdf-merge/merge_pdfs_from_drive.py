import os
import io
from typing import List, Dict

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from PyPDF2 import PdfMerger

SCOPES = ["https://www.googleapis.com/auth/drive"]
ROOT_FOLDER_ID = os.environ.get("ROOT_DRIVE_FOLDER_ID")
SERVICE_ACCOUNT_FILE = "service_account.json"


def get_drive_service():
    """Создаёт клиент Google Drive по файлу сервисного аккаунта."""
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        raise RuntimeError(
            f"Service account file '{SERVICE_ACCOUNT_FILE}' not found. "
            f"Make sure GitHub Actions step created it."
        )

    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)


def list_subfolders(service, parent_id: str) -> List[Dict]:
    """Возвращает все подпапки внутри parent_id, отсортированные по имени."""
    folders: List[Dict] = []
    page_token = None

    while True:
        response = (
            service.files()
            .list(
                q=(
                    f"'{parent_id}' in parents and "
                    "mimeType='application/vnd.google-apps.folder' and "
                    "trashed=false"
                ),
                spaces="drive",
                fields="nextPageToken, files(id, name)",
                pageSize=1000,
                orderBy="name",
                pageToken=page_token,
            )
            .execute()
        )
        folders.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return folders


def list_pdfs(service, folder_id: str) -> List[Dict]:
    """Список всех PDF в папке, отсортированных по имени."""
    pdfs: List[Dict] = []
    page_token = None

    while True:
        response = (
            service.files()
            .list(
                q=(
                    f"'{folder_id}' in parents and "
                    "mimeType='application/pdf' and "
                    "trashed=false"
                ),
                spaces="drive",
                fields="nextPageToken, files(id, name)",
                pageSize=1000,
                orderBy="name",
                pageToken=page_token,
            )
            .execute()
        )
        pdfs.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return pdfs


def download_pdf(service, file_id: str) -> io.BytesIO:
    """Скачивает PDF в память (BytesIO)."""
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False

    while not done:
        status, done = downloader.next_chunk()

    fh.seek(0)
    return fh


def delete_existing_output(service, folder_id: str, output_name: str) -> None:
    """Удаляет уже существующий итоговый PDF, если он есть."""
    response = (
        service.files()
        .list(
            q=(
                f"'{folder_id}' in parents and "
                f"name='{output_name}' and "
                "mimeType='application/pdf' and "
                "trashed=false"
            ),
            spaces="drive",
            fields="files(id)",
            pageSize=1000,
        )
        .execute()
    )

    for file in response.get("files", []):
        service.files().delete(fileId=file["id"]).execute()
        print(f"    удалён старый файл: {output_name} (id={file['id']})")


def upload_pdf(
    service, folder_id: str, name: str, fh: io.BytesIO
) -> str:
    """Загружает итоговый PDF в указанную папку."""
    media = MediaIoBaseUpload(
        fh, mimetype="application/pdf", resumable=True
    )
    file_metadata = {
        "name": name,
        "parents": [folder_id],
        "mimeType": "application/pdf",
    }

    created = (
        service.files()
        .create(body=file_metadata, media_body=media, fields="id")
        .execute()
    )
    return created.get("id")


def merge_pdfs_for_folder(service, folder: Dict) -> None:
    """Сшивает все PDF-файлы в папке folder в один итоговый."""
    folder_id = folder["id"]
    folder_name = folder["name"]

    pdfs = list_pdfs(service, folder_id)
    if not pdfs:
        print(f"[SKIP] Папка '{folder_name}' — нет PDF-файлов")
        return

    print(
        f"[MERGE] Папка '{folder_name}' (id={folder_id}), файлов: {len(pdfs)}"
    )

    merger = PdfMerger(strict=False)
    for f in pdfs:
        print(f"    добавляем: {f['name']}")
        fh = download_pdf(service, f["id"])
        merger.append(fh)

    # записываем результат в память
    out_stream = io.BytesIO()
    merger.write(out_stream)
    merger.close()
    out_stream.seek(0)

    output_name = f"{folder_name}.pdf"  # например: "204 ап.pdf"
    delete_existing_output(service, folder_id, output_name)
    new_id = upload_pdf(service, folder_id, output_name, out_stream)
    print(f"    → создан файл '{output_name}' (id={new_id})")


def main():
    if not ROOT_FOLDER_ID:
        raise SystemExit(
            "Не задана переменная окружения ROOT_DRIVE_FOLDER_ID"
        )

    service = get_drive_service()
    print(f"Корневая папка: {ROOT_FOLDER_ID}")

    folders = list_subfolders(service, ROOT_FOLDER_ID)
    print(f"Найдено папок-апартаментов: {len(folders)}")

    for folder in folders:
        merge_pdfs_for_folder(service, folder)


if __name__ == "__main__":
    main()
