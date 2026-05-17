from dataclasses import dataclass
from pathlib import Path


@dataclass
class Segment:
    text: str
    start: float
    end: float


@dataclass
class MediaFile:
    path: str
    ext: str


@dataclass
class FileJob:
    input_path: str
    rel_path: Path
    stem: str
    ext: str
    is_video: bool
    subtitle_path: str | None = None
    audio_path: str | None = None
    segments: list[Segment] | None = None
    translated: list[Segment] | None = None
    audio_files: list[str] | None = None
    composed_path: str | None = None
