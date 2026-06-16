"""Video frame extraction with histogram-based deduplication."""

import cv2
import os
from typing import Generator, Dict, Optional
from PIL import Image
from semantixel.core.logging import logger


def _get_histogram(frame):
    """Compute a 2D HSV histogram for a video frame.

    Args:
        frame: BGR image array from OpenCV.

    Returns:
        Normalised histogram, or ``None`` if *frame* is ``None``.
    """
    if frame is None:
        return None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def _calculate_histogram_difference(hist1, hist2) -> float:
    """Bhattacharyya distance between two histograms (0 = identical)."""
    if hist1 is None or hist2 is None:
        return 1.0
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)


def extract_frames_in_memory(
    video_path: str, fps: float = 0.5, similarity_threshold: float = 0.6
) -> Generator[Dict, None, None]:
    """Extract keyframes from a video at a fixed sampling rate.

    Skips frames that are too similar (Bhattacharyya distance below
    *similarity_threshold*) to the previous extracted frame.

    Args:
        video_path: Absolute path to the video file.
        fps: Target frames per second to sample.
        similarity_threshold: Minimum histogram distance between
            consecutive frames.  Higher values = fewer frames.

    Yields:
        Dicts with keys ``"image"`` (PIL Image) and ``"timestamp"`` (float seconds).
    """
    if not os.path.exists(video_path):
        logger.error("Video file not found at %s", video_path)
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error("Could not open video file %s", video_path)
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps == 0:
        cap.release()
        return

    video_duration_sec = total_frames / video_fps

    current_sec = 0.0
    prev_hist = None
    frames_yielded = 0
    frames_skipped = 0

    try:
        while current_sec < video_duration_sec:
            cap.set(cv2.CAP_PROP_POS_MSEC, current_sec * 1000.0)
            ret, frame = cap.read()

            if not ret:
                break

            current_hist = _get_histogram(frame)

            if prev_hist is not None:
                dist = _calculate_histogram_difference(prev_hist, current_hist)
                if dist < similarity_threshold:
                    frames_skipped += 1
                    current_sec += 1.0 / fps
                    continue

            prev_hist = current_hist

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            frames_yielded += 1

            yield {"image": pil_image, "timestamp": round(current_sec, 3)}

            current_sec += 1.0 / fps
    finally:
        total_checked = frames_yielded + frames_skipped
        if total_checked > 0:
            logger.debug(
                "[Video Indexed] %s | Kept: %d frames | Skipped: %d redundant frames",
                os.path.basename(video_path),
                frames_yielded,
                frames_skipped,
            )
        cap.release()
