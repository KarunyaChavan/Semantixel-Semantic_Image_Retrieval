"""Logging configuration for Semantixel.

Provides a pre-configured :data:`logger` instance and a convenience
:func:`log_exception` helper for capturing tracebacks.
"""

import logging
import os
import sys
import traceback


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Create and return the root ``semantixel`` logger.

    Writes structured log lines to ``stdout`` with format::

        YYYY-MM-DD HH:MM:SS - semantixel - LEVEL - [filename.py:line] - message

    If the ``SEMANTIXEL_LOG_FILE`` environment variable is set, logs
    are also appended to that file.

    Args:
        level: Logging level (default ``logging.INFO``).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger("semantixel")
    logger.setLevel(level)

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    log_file = os.getenv("SEMANTIXEL_LOG_FILE")
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_exception(logger_instance: logging.Logger, message: str, *args, level: int = logging.ERROR) -> None:
    """Log an exception with its full traceback.

    Args:
        logger_instance: The logger to use.
        message: Format string (can include ``%s`` placeholders).
        *args: Format arguments.
        level: Logging level (default ``logging.ERROR``).
    """
    tb = traceback.format_exc()
    if args:
        message = message % args
    logger_instance.log(level, "%s\n%s", message, tb)


logger: logging.Logger = setup_logging()
"""Pre-configured module-level logger.  Import and use directly."""
