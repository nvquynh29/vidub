import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore", message=".*cpp extensions.*torch.*", category=UserWarning)
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")

_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


_level_map = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARN,
    "error": logging.ERROR,
}


def configure_logging(level_name: str = "info") -> None:
    level = _level_map.get(level_name.lower(), logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
