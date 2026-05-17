from abc import ABC, abstractmethod

from vidub.config import TranslateConfig
from vidub.models import Segment


class TranslationEngine(ABC):
    def __init__(self, config: TranslateConfig):
        self.config = config

    @abstractmethod
    def translate(self, segments: list[Segment], target_lang: str, source_lang: str | None = None) -> list[Segment]:
        ...

    @staticmethod
    def _word_count(text: str) -> int:
        return len(text.split())

    def _split_segments_by_words(self, segments: list[Segment]) -> list[list[Segment]]:
        max_words = self.config.max_words
        batches: list[list[Segment]] = []
        current_batch: list[Segment] = []
        current_words = 0

        for seg in segments:
            wc = self._word_count(seg.text)
            if current_words + wc > max_words and current_batch:
                batches.append(current_batch)
                current_batch = [seg]
                current_words = wc
            else:
                current_batch.append(seg)
                current_words += wc

        if current_batch:
            batches.append(current_batch)

        return batches
