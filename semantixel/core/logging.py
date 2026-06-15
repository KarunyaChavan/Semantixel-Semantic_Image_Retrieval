import logging
import sys
import os
import traceback

def setup_logging(level=logging.INFO):
    """
    Sets up a centralized logging configuration for the entire application.
    """
    logger = logging.getLogger("semantixel")
    logger.setLevel(level)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(console_handler)
    
    # Optional: File Handler for production
    log_file = os.getenv("SEMANTIXEL_LOG_FILE")
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def log_exception(logger_instance, message: str, level=logging.ERROR):
    """Log an exception with full traceback at the given log level.

    Captures the current exception context via traceback.format_exc
    and appends it to the log message. Use inside an except block.

    Args:
        logger_instance: A :class:`logging.Logger` instance.
        message: Human-readable description of the error context.
        level: Logging level (default ``logging.ERROR``).
    """
    tb = traceback.format_exc()
    logger_instance.log(level, f"{message}\n{tb}")

# Primary logger for the application
logger = setup_logging()
