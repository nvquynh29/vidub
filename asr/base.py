from abc import ABC, abstractmethod

from vidub.config import ASRConfig
from vidub.models import Segment


class ASREngine(ABC):
    def __init__(self, config: ASRConfig):
        self.config = config

    @abstractmethod
    def transcribe(self, audio_path: str, language: str | None = None, duration: float = 0.0) -> list[Segment]:
        ...
