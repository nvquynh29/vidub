from abc import ABC, abstractmethod

from vidub.config import TTSConfig
from vidub.models import Segment


class TTSEngine(ABC):
    def __init__(self, config: TTSConfig):
        self.config = config

    @abstractmethod
    def synthesize_segments(self, segments: list[Segment], output_dir: str, stem: str = "") -> list[str]:
        ...
