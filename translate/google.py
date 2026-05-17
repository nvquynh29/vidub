import re

from vidub.translate.base import TranslationEngine
from vidub.config import TranslateConfig
from vidub.log_utils import get_logger
from vidub.models import Segment
from vidub.registry import register_translate


log = get_logger("vidub.translate.google")

_SEPARATOR = " @|@ "
_MAX_CHARS = 4800


@register_translate("google-translate")
class GoogleTranslateEngine(TranslationEngine):
    def __init__(self, config: TranslateConfig):
        super().__init__(config)
        self._translator = None

    def _load_translator(self, target_lang: str, source_lang: str | None):
        source = source_lang or "auto"
        try:
            from deep_translator import GoogleTranslator
        except ImportError:
            raise ImportError(
                "deep-translator is required for Google Translate engine. "
                "Install it with: pip install deep-translator"
            )
        log.debug("Loading GoogleTranslator: %s -> %s", source, target_lang)
        self._translator = GoogleTranslator(source=source, target=target_lang)

    def translate(self, segments: list[Segment], target_lang: str, source_lang: str | None = None) -> list[Segment]:
        if not segments:
            return []
        self._load_translator(target_lang, source_lang)

        word_batches = self._split_segments_by_words(segments)
        if len(word_batches) > 1:
            log.info("Split %d segments into %d word-count batches (max %d words each)", len(segments), len(word_batches), self.config.max_words)

        all_translated: list[Segment] = []
        for batch in word_batches:
            texts = [seg.text.strip() for seg in batch]
            translated_texts = self._translate_with_context(texts)
            all_translated.extend(
                Segment(text=t, start=seg.start, end=seg.end)
                for seg, t in zip(batch, translated_texts)
            )

        log.info("Translation complete: %d segments", len(all_translated))
        return all_translated

    def _translate_with_context(self, texts: list[str]) -> list[str]:
        """Translate texts with context by joining nearby segments within char limit."""
        result: list[str | None] = [None] * len(texts)
        i = 0
        while i < len(texts):
            batch_indices = [i]
            batch_texts = [texts[i]]
            char_count = len(texts[i])
            j = i + 1
            while j < len(texts):
                added = len(_SEPARATOR) + len(texts[j])
                if char_count + added > _MAX_CHARS:
                    break
                batch_indices.append(j)
                batch_texts.append(texts[j])
                char_count += added
                j += 1

            if len(batch_texts) == 1:
                result[i] = self._translator.translate(batch_texts[0])
                i += 1
            else:
                combined = _SEPARATOR.join(batch_texts)
                translated = self._translator.translate(combined)
                split = re.split(r"\s*@\|@\s*", translated.strip())
                if len(split) == len(batch_texts):
                    for idx, t in zip(batch_indices, split):
                        result[idx] = t
                else:
                    for idx in batch_indices:
                        result[idx] = self._translator.translate(texts[idx])
                i = j

        return [r for r in result if r is not None]
