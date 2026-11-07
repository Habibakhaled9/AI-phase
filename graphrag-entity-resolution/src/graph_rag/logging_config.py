"""Application logging configuration."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from graph_rag.config import settings


def configure_logging() -> None:
    """Write application logs to the terminal and a rotating data/app.log file."""
    root_logger = logging.getLogger()
    if getattr(root_logger, "_graph_rag_configured", False):
        return

    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S%z",
    )
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    file_handler = RotatingFileHandler(
        Path(settings.APP_LOG_FILE),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    root_logger._graph_rag_configured = True
