"""
logging_config.py

Centralized logging configuration for the GraphRAG project.

Usage
-----
Import and call setup_logging() once at application startup (app.py does this).
Every other module simply does:

    import logging
    logger = logging.getLogger(__name__)

Log levels are controlled via the LOG_LEVEL environment variable (default: INFO).
Logs are written to both the console and logs/graphrag.log with rotation.

Format
------
2026-10-19 14:30:10 | INFO     | graph_builder | Graph created successfully: 87 nodes, 97 edges

Security
--------
Never log passwords, API keys, or sensitive document contents.
"""

import logging
import logging.handlers
import os
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent / "logs"
LOG_FILE = LOG_DIR / "graphrag.log"

# Max size per log file before rotation (10 MB)
LOG_MAX_BYTES = 10 * 1024 * 1024
# Number of backup files to keep
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Respect LOG_LEVEL env var; default INFO
_DEFAULT_LEVEL = "INFO"
_ENV_LEVEL = os.getenv("LOG_LEVEL", _DEFAULT_LEVEL).upper()

# Map string to logging level, fall back to INFO on invalid values
_LEVEL_MAP = {
    "DEBUG":    logging.DEBUG,
    "INFO":     logging.INFO,
    "WARNING":  logging.WARNING,
    "ERROR":    logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
LOG_LEVEL = _LEVEL_MAP.get(_ENV_LEVEL, logging.INFO)

# Guard so setup_logging() is idempotent - calling it multiple times (e.g.
# from different Streamlit reruns) must not add duplicate handlers.
_logging_configured = False


def setup_logging() -> None:
    """
    Configure the root logger with:
    - A rotating file handler writing to logs/graphrag.log
    - A stream handler writing to stderr/stdout (console)

    Safe to call multiple times; subsequent calls are no-ops.
    Noisy third-party libraries (neo4j, httpx, chromadb, etc.) are
    quieted to WARNING so they do not flood the log.
    """
    global _logging_configured
    if _logging_configured:
        return

    # Ensure the logs/ directory exists
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        # If we cannot create the log directory, fall back to console-only.
        # Never let a logging failure break the application.
        logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        logging.getLogger(__name__).warning(
            "Could not create log directory %s: %s - logging to console only.", LOG_DIR, exc
        )
        _logging_configured = True
        return

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # ── Rotating file handler ─────────────────────────────────────────────────
    file_handler = None
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(LOG_FILE),
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(LOG_LEVEL)
    except Exception as exc:
        file_handler = None
        logging.warning("Could not create log file handler: %s", exc)

    # ── Console (stream) handler ──────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(LOG_LEVEL)

    # ── Root logger ───────────────────────────────────────────────────────────
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)

    # Remove any pre-existing handlers (e.g. from basicConfig in tests) so we
    # do not duplicate output on re-import in long-running Streamlit sessions.
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    if file_handler:
        root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # ── Quiet noisy third-party loggers ──────────────────────────────────────
    _noisy_loggers = [
        "neo4j",
        "neo4j.bolt",
        "httpx",
        "httpcore",
        "chromadb",
        "chromadb.segment",
        "chromadb.telemetry",
        "urllib3",
        "langchain",
        "langchain_core",
        "langchain_neo4j",
        "langchain_ollama",
        "openai",
        "PIL",
        "matplotlib",
        "streamlit",
    ]
    for name in _noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)

    _logging_configured = True

    # First log entry after setup
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging initialised | level=%s | file=%s | backup_count=%d",
        _ENV_LEVEL, LOG_FILE, LOG_BACKUP_COUNT,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Convenience helper: ensures logging is configured then returns a named logger.

    Equivalent to:
        setup_logging()
        logger = logging.getLogger(name)
    """
    setup_logging()
    return logging.getLogger(name)
