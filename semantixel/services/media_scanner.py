"""Media file discovery and scanning service."""

import os
from typing import List, Optional, Tuple
from semantixel.core.logging import logger
from semantixel.media_types import is_media_file


def scan_directory(directory: str, exclude_directories: List[str]) -> List[str]:
    """Recursively scan a directory for media files.

    Media extensions include images, videos, and audio formats.

    Args:
        directory: Root directory to scan.
        exclude_directories: Absolute paths of directories to skip.

    Returns:
        List of media file paths found.
    """
    files = []
    try:
        if not os.path.isdir(directory):
            return []
        with os.scandir(directory) as entries:
            for entry in entries:
                if (
                    entry.is_file()
                    and not entry.name.startswith("._")
                    and is_media_file(entry.name)
                ):
                    files.append(entry.path)
                elif entry.is_dir():
                    abs_path = os.path.abspath(entry.path)
                    if not any(
                        os.path.commonpath([abs_path, os.path.abspath(excl)])
                        == os.path.abspath(excl)
                        for excl in exclude_directories
                    ):
                        files.extend(scan_directory(entry.path, exclude_directories))
    except PermissionError:
        logger.debug("Permission denied: %s", directory)
    except Exception as exc:
        logger.error("Error scanning %s: %s", directory, exc)
    return files


def fast_scan_for_media(
    directories: List[str], exclude_directories: Optional[List[str]] = None
) -> Tuple[List[str], float]:
    """Scan multiple directories in parallel using a thread pool.

    Args:
        directories: List of root directories to scan.
        exclude_directories: Directories to skip (merged with config values).

    Returns:
        A tuple of ``(file_paths, elapsed_seconds)``.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    if exclude_directories is None:
        exclude_directories = []

    start_time = time.time()
    all_files: List[str] = []

    cpu_count = os.cpu_count() or 1

    with tqdm(total=len(directories), desc="Scanning directories") as pbar:
        with ThreadPoolExecutor(max_workers=cpu_count) as executor:
            future_to_dir = {
                executor.submit(scan_directory, d, exclude_directories): d
                for d in directories
            }
            for future in as_completed(future_to_dir):
                d = future_to_dir[future]
                try:
                    all_files.extend(future.result())
                except Exception as exc:
                    logger.error("Error processing %s: %s", d, exc)
                pbar.update(1)

    elapsed = time.time() - start_time
    return all_files, elapsed
