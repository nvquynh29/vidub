import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, TypeVar

from vidub.audio import MEDIA_EXTENSIONS, AUDIO_EXTENSIONS

R = TypeVar("R")

ALL_MEDIA_EXTENSIONS = MEDIA_EXTENSIONS | AUDIO_EXTENSIONS
SRT_EXTENSION = ".srt"


def scan_folder(folder: str, extensions: set[str] | None = None) -> list[str]:
    if extensions is None:
        extensions = ALL_MEDIA_EXTENSIONS
    files = []
    for root, _, filenames in os.walk(folder):
        for fn in sorted(filenames):
            if Path(fn).suffix.lower() in extensions:
                files.append(os.path.join(root, fn))
    return files


def batch_process(
    files: list[str],
    worker_fn: Callable[[str], R],
    gpu: bool = False,
) -> list[R]:
    max_workers = os.cpu_count() or 1
    if gpu:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(worker_fn, files))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(worker_fn, files))
