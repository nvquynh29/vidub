import os

from vidub.asr.base import ASREngine
from vidub.config import ASRConfig
from vidub.log_utils import get_logger
from vidub.models import Segment
from vidub.registry import register_asr


log = get_logger("vidub.asr.whisper_cpp")


@register_asr("whisper-cpp")
class WhisperCppEngine(ASREngine):
    def __init__(self, config: ASRConfig):
        super().__init__(config)
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from pywhispercpp.model import Model
        except ImportError:
            raise ImportError(
                "pywhispercpp is required for whisper.cpp ASR engine. "
                "Install it with: pip install pywhispercpp"
            )
        n_threads = os.cpu_count() or 4
        log.info("Loading whisper.cpp model: %s (threads=%d)", self.config.model, n_threads)
        self._model = Model(
            self.config.model,
            n_threads=n_threads,
        )

    def transcribe(self, audio_path: str, language: str | None = None, duration: float = 0.0) -> list[Segment]:
        self._load_model()

        log.debug("Transcribing %s (language=%s, duration=%.3fs)", audio_path, language, duration)
        segs = self._model.transcribe(
            audio_path,
            language=language or "",
            print_progress=False,
            print_realtime=False,
        )

        result = [
            Segment(text=s.text.strip(), start=s.t0 / 1000.0, end=s.t1 / 1000.0)
            for s in segs
            if s.text.strip()
        ]
        log.info("Transcription returned %d segments", len(result))
        return result
