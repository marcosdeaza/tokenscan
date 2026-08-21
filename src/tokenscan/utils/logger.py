import logging
import sys
from datetime import datetime, timezone

FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logger(name: str = "tokenscan", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(FORMAT))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
