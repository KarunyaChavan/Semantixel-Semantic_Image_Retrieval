import io
import os
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from semantixel.core.config import config
from semantixel.core.logging import logger
from semantixel.media import MediaDescriptor, build_media_id


class GoogleDriveSource:
    SOURCE_NAME = "gdrive"
    DRIVE_API_BASE = "https://www.googleapis.com/drive/v3"
    DRIVE_UPLOAD_API_BASE = "https://www.googleapis.com/drive/v3/files"
    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

    def __init__(self):
        self._state: Optional[str] = None
        self._code_verifier: Optional[str] = None

    @property
    def settings(self):
        return config.google_drive

    def is_enabled(self) -> bool:
        return self.settings.enabled

    def is_configured(self) -> bool:
        return self.is_enabled() and bool(self.settings.client_secret_file)

    def dependencies_available(self) -> bool:
        try:
            import google_auth_oauthlib.flow  # noqa: F401
        except ImportError:
            return False
        return True

    def _client_secret_path(self) -> str:
        if not self.settings.client_secret_file:
            return ""
        return os.path.abspath(self.settings.client_secret_file)

    def _token_path(self) -> str:
        return os.path.abspath(self.settings.token_file)

    def _load_credentials(self):
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        token_path = self._token_path()
        creds = None
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, self.SCOPES)

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            self._save_credentials(creds)

        return creds

    def _save_credentials(self, credentials) -> None:
        token_path = self._token_path()
        directory = os.path.dirname(token_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(token_path, "w", encoding="utf-8") as handle:
            handle.write(credentials.to_json())

    def _build_flow(self, state: Optional[str] = None):
        from google_auth_oauthlib.flow import Flow

        flow = Flow.from_client_secrets_file(
            self._client_secret_path(),
            scopes=self.SCOPES,
            state=state,
        )
        if self.settings.redirect_uri:
            flow.redirect_uri = self.settings.redirect_uri
        return flow

    def get_status(self) -> Dict[str, Any]:
        token_exists = os.path.exists(self._token_path()) if self.settings.token_file else False
        return {
            "enabled": self.is_enabled(),
            "configured": self.is_configured(),
            "dependencies_available": self.dependencies_available(),
            "authorized": bool(self.get_credentials()),
            "client_secret_file": self.settings.client_secret_file,
            "redirect_uri": self.settings.redirect_uri,
            "folder_ids": self.settings.folder_ids,
            "token_file": self.settings.token_file,
            "token_present": token_exists,
        }

    def get_credentials(self):
        if not self.is_configured() or not self.dependencies_available():
            return None

        try:
            creds = self._load_credentials()
        except Exception as exc:
            logger.warning(f"Failed to load Google Drive credentials: {exc}")
            return None

        if creds and creds.valid:
            return creds
        return None

    def get_authorization_url(self) -> Dict[str, str]:
        if not self.dependencies_available():
            raise RuntimeError("google-auth-oauthlib is not installed.")
        if not self.is_configured():
            raise RuntimeError("Google Drive is not configured.")

        import secrets

        self._state = secrets.token_urlsafe(24)
        flow = self._build_flow(state=self._state)
        authorization_url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
        )
        self._code_verifier = getattr(flow, "code_verifier", None)
        return {"authorization_url": authorization_url, "state": self._state}

    def exchange_code(self, code: str, state: Optional[str]) -> None:
        if not self.dependencies_available():
            raise RuntimeError("google-auth-oauthlib is not installed.")
        if self._state and state and state != self._state:
            raise RuntimeError("Invalid OAuth state.")

        flow = self._build_flow(state=state)
        if hasattr(self, "_code_verifier") and self._code_verifier:
            flow.fetch_token(code=code, code_verifier=self._code_verifier)
        else:
            flow.fetch_token(code=code)
        self._save_credentials(flow.credentials)

    def _authorized_session(self):
        from google.auth.transport.requests import AuthorizedSession

        creds = self.get_credentials()
        if creds is None:
            raise RuntimeError("Google Drive is not authorized.")
        return AuthorizedSession(creds)

    def _mime_query(self) -> str:
        mime_types = self.settings.image_mime_types
        if not mime_types:
            return ""
        return "(" + " or ".join([f"mimeType='{mime}'" for mime in mime_types]) + ")"

    def _folder_query(self) -> str:
        if not self.settings.folder_ids:
            return ""
        return "(" + " or ".join([f"'{folder_id}' in parents" for folder_id in self.settings.folder_ids]) + ")"

    def _build_query(self) -> str:
        clauses = ["trashed=false", self._mime_query()]
        folder_query = self._folder_query()
        if folder_query:
            clauses.append(folder_query)
        return " and ".join([clause for clause in clauses if clause])

    def _request_json(self, session, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        response = session.get(url, params=params, timeout=60)
        response.raise_for_status()
        return response.json()

    def list_media(self) -> List[MediaDescriptor]:
        if not self.is_enabled():
            return []

        session = self._authorized_session()
        params = {
            "q": self._build_query(),
            "pageSize": self.settings.page_size,
            "fields": "nextPageToken, files(id, name, mimeType, modifiedTime, webViewLink, thumbnailLink)",
            "supportsAllDrives": str(self.settings.include_shared_drives).lower(),
            "includeItemsFromAllDrives": str(self.settings.include_shared_drives).lower(),
        }

        items: List[MediaDescriptor] = []
        next_page_token = None
        while True:
            if next_page_token:
                params["pageToken"] = next_page_token
            payload = self._request_json(session, f"{self.DRIVE_API_BASE}/files", params=params)
            for file_data in payload.get("files", []):
                display_path = f"Google Drive/{file_data['name']}"
                media_id = build_media_id(self.SOURCE_NAME, file_data["id"])
                items.append(
                    MediaDescriptor(
                        source=self.SOURCE_NAME,
                        locator=file_data["id"],
                        media_type="image",
                        media_id=media_id,
                        display_path=display_path,
                    )
                )
            next_page_token = payload.get("nextPageToken")
            if not next_page_token:
                break

        logger.info(f"Loaded {len(items)} Google Drive media items")
        return items

    def fetch_image(self, locator: str) -> Image.Image:
        content, _, _ = self.fetch_bytes(locator)
        return Image.open(io.BytesIO(content)).convert("RGB")

    def fetch_bytes(self, locator: str) -> Tuple[bytes, str, str]:
        session = self._authorized_session()
        metadata = self._request_json(
            session,
            f"{self.DRIVE_API_BASE}/files/{locator}",
            params={"fields": "id,name,mimeType"},
        )
        response = session.get(
            f"{self.DRIVE_UPLOAD_API_BASE}/{locator}",
            params={"alt": "media"},
            timeout=120,
        )
        response.raise_for_status()
        mime_type = metadata.get("mimeType", "application/octet-stream")
        file_name = metadata.get("name", locator)
        return response.content, mime_type, file_name
