"""Central logging configuration for Semantixel.

Provides a pre-configured :data:`logger` instance that writes application
history to a durable log file, plus a convenience 
:func:`log_exception` helper for capturing tracebacks.
"""

import logging
import os
import traceback
from logging.handlers import RotatingFileHandler

DEFAULT_LOG_FILE = os.path.join("logs", "semantixel.log")
DEFAULT_MAX_BYTES = 5 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 3


def _env_flag(name: str, default: bool = False) -> bool:
    """Return a boolean value from an environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Create and return the root ``semantixel`` logger.

    Writes structured log lines to a central log file with format::

        YYYY-MM-DD HH:MM:SS - semantixel - LEVEL - [filename.py:line] - message

    By default logs are written to ``logs/semantixel.log``. Override the path
    with ``SEMANTIXEL_LOG_FILE``. Console logging is disabled by default and
    can be enabled with ``SEMANTIXEL_LOG_CONSOLE=true``.

    Args:
        level: Logging level (default ``logging.INFO``).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger("semantixel")
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )

    if not any(
        getattr(handler, "_semantixel_file_handler", False)
        for handler in logger.handlers
    ):
        log_file = os.getenv("SEMANTIXEL_LOG_FILE", DEFAULT_LOG_FILE)
        log_dir = os.path.dirname(os.path.abspath(log_file))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=DEFAULT_MAX_BYTES,
            backupCount=DEFAULT_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler._semantixel_file_handler = True
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if _env_flag("SEMANTIXEL_LOG_CONSOLE"):
        import sys

        if not any(
            getattr(handler, "_semantixel_console_handler", False)
            for handler in logger.handlers
        ):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler._semantixel_console_handler = True
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

    return logger


def log_exception(
    logger_instance: logging.Logger, message: str, *args, level: int = logging.ERROR
) -> None:
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
