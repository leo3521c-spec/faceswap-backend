"""Structured logging setup."""
import logging
import sys

from config import get_settings


def setup_logger(name: str = "faceswap") -> logging.Logger:
    settings = get_settings()
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger