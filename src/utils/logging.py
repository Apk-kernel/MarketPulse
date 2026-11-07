"""Structured logging configuration for MarketPulse."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "MarketPulse",
    log_level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Configures and returns a structured logger.

    Args:
        name: Name of the logger.
        log_level: Logging level (e.g. logging.INFO, logging.DEBUG).
        log_file: Optional path to a file where logs should be appended.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "MarketPulse") -> logging.Logger:
    """Retrieves an existing logger or initializes a default one.

    Args:
        name: Module or logger name.

    Returns:
        logging.Logger instance.
    """
    return logging.getLogger(name)

