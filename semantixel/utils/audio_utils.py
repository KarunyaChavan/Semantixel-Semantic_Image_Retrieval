import subprocess
from semantixel.core.logging import logger

def has_audio_stream(file_path: str) -> bool:
    """Check whether a media file contains an audio track.

    Uses ``ffprobe`` to inspect the file's stream list. Returns ``True``
    when an audio stream is detected. If ``ffprobe`` is unavailable or
    the probe fails, it assumes an audio stream exists (fail-safe) so
    downstream processing can proceed and surface the real error.

    Args:
        file_path: Absolute path to the media file.

    Returns:
        ``True`` if an audio stream is present (or probe failed).
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", "csv=p=0", file_path],
            capture_output=True, text=True, timeout=30
        )
        return "audio" in result.stdout
    except Exception as exc:
        logger.debug(f"ffprobe check failed for {file_path}: {exc}")
        return True
