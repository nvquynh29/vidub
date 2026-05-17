import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore", message=".*cpp extensions.*torch.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*", category=UserWarning)
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")

_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


_level_map = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARN,
    "error": logging.ERROR,
}


def configure_logging(level_name: str = "info", log_file: str = "progress.log") -> None:
    level = _level_map.get(level_name.lower(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    fmt = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(fmt)
    root.addHandler(handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
