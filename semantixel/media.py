import base64
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


LOCAL_SOURCE = "local"
GOOGLE_DRIVE_SOURCE = "gdrive"
FRAME_SEPARATOR = ":::"


def _b64_encode(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")


def _b64_decode(value: str) -> str:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii")).decode("utf-8")


def normalize_local_path(path: str) -> str:
    return os.path.abspath(path.strip('"').strip("'"))


def build_media_id(source: str, locator: str, timestamp: Optional[float] = None) -> str:
    encoded_locator = _b64_encode(locator)
    if timestamp is None:
        return f"{source}|{encoded_locator}"
    return f"{source}|{encoded_locator}|{timestamp:.6f}"


@dataclass(frozen=True)
class MediaDescriptor:
    source: str
    locator: str
    media_type: str
    media_id: str
    display_path: str
    timestamp: Optional[float] = None

    @property
    def is_video_frame(self) -> bool:
        return self.timestamp is not None

    @property
    def composite_id(self) -> str:
        if self.timestamp is None:
            return self.media_id
        return f"{self.media_id}{FRAME_SEPARATOR}{self.timestamp:.6f}"

    def to_result(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "media_id": self.media_id,
            "source": self.source,
            "path": self.display_path,
            "display_path": self.display_path,
            "type": "video" if self.is_video_frame else self.media_type,
            "locator": self.locator,
        }
        if self.timestamp is not None:
            payload["timestamp"] = self.timestamp
            payload["composite_id"] = self.composite_id
        else:
            payload["composite_id"] = self.media_id
        return payload


def describe_local_media(path: str, timestamp: Optional[float] = None) -> MediaDescriptor:
    normalized_path = normalize_local_path(path)
    media_type = "video_frame" if timestamp is not None else "image"
    return MediaDescriptor(
        source=LOCAL_SOURCE,
        locator=normalized_path,
        media_type=media_type,
        media_id=build_media_id(LOCAL_SOURCE, normalized_path),
        display_path=normalized_path,
        timestamp=timestamp,
    )


def parse_media_id(raw_id: str) -> MediaDescriptor:
    if "|" not in raw_id:
        if FRAME_SEPARATOR in raw_id:
            locator, timestamp_fragment = raw_id.rsplit(FRAME_SEPARATOR, 1)
            return describe_local_media(locator, timestamp=float(timestamp_fragment))
        return describe_local_media(raw_id)

    base_id, _, timestamp_fragment = raw_id.partition(FRAME_SEPARATOR)
    parts = base_id.split("|")
    if len(parts) != 2:
        raise ValueError(f"Unsupported media identifier: {raw_id}")

    source, encoded_locator = parts
    locator = _b64_decode(encoded_locator)
    timestamp = float(timestamp_fragment) if timestamp_fragment else None

    if source == LOCAL_SOURCE:
        return describe_local_media(locator, timestamp=timestamp)
    if source == GOOGLE_DRIVE_SOURCE:
        return MediaDescriptor(
            source=GOOGLE_DRIVE_SOURCE,
            locator=locator,
            media_type="image",
            media_id=build_media_id(GOOGLE_DRIVE_SOURCE, locator),
            display_path=f"Google Drive/{locator}",
            timestamp=timestamp,
        )

    raise ValueError(f"Unsupported media source: {source}")


def is_media_id(value: str) -> bool:
    parts = value.split("|")
    return len(parts) >= 2 and parts[0] in {LOCAL_SOURCE, GOOGLE_DRIVE_SOURCE}
