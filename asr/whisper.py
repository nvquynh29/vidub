from vidub.asr.base import ASREngine
from vidub.config import ASRConfig
from vidub.log_utils import get_logger
from vidub.models import Segment
from vidub.registry import register_asr


log = get_logger("vidub.asr.whisper")

_DEFAULT_MODEL = "distil-large-v3"
_QWEN_MODEL = "Qwen/Qwen3-ASR-1.7B"


def _resolve_hf_model(model: str) -> str:
    if "/" not in model:
        return model
    prefix = "openai/whisper-"
    if model.startswith(prefix):
        suffix = model[len(prefix):]
        if suffix == "large":
            suffix = "large-v2"
        return f"Systran/faster-whisper-{suffix}"
    return model


def _detect_compute_type(device: str) -> str:
    if device == "cpu":
        return "int8"
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).lower()
            if "t4" in name:
                return "int8_float16"
        return "float16"
    except ImportError:
        return "float16"


@register_asr("faster-whisper")
@register_asr("whisper")
class FasterWhisperEngine(ASREngine):
    def __init__(self, config: ASRConfig):
        super().__init__(config)
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper is required for this ASR engine. "
                "Install it with: pip install faster-whisper"
            )
        model = self.config.model
        if model == _QWEN_MODEL:
            model = _DEFAULT_MODEL
        model = _resolve_hf_model(model)
        gpu = self.config.device == "cuda"
        device = "cuda" if gpu else "cpu"
        compute_type = _detect_compute_type(device)
        log.info("Loading faster-whisper model: %s on %s (compute_type=%s)", model, device, compute_type)
        self._model = WhisperModel(
            model,
            device=device,
            compute_type=compute_type,
        )

    def transcribe(self, audio_path: str, language: str | None = None, duration: float = 0.0) -> list[Segment]:
        self._load_model()

        log.debug("Transcribing %s (language=%s, duration=%.3fs)", audio_path, language, duration)
        segments, info = self._model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            condition_on_previous_text=False,
            word_timestamps=False,
        )

        result = [
            Segment(text=seg.text.strip(), start=seg.start, end=seg.end)
            for seg in segments
            if seg.text.strip()
        ]
        log.info("Transcription returned %d segments (language: %s)", len(result), info.language)
        return result
