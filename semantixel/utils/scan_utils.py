"""Directory scanning utilities (delegates to :mod:`semantixel.services.media_scanner`).

Kept for backward compatibility — new code should import directly from
``semantixel.services.media_scanner``.
"""

from semantixel.services.media_scanner import scan_directory, fast_scan_for_media

__all__ = ["scan_directory", "fast_scan_for_media"]
