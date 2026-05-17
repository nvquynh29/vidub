from vidub.asr.base import ASREngine
from vidub.config import ASRConfig
from vidub.lang_utils import to_qwen_language
from vidub.log_utils import get_logger
from vidub.models import Segment
from vidub.registry import register_asr


log = get_logger("vidub.asr.qwen")


@register_asr("qwen")
class QwenASREngine(ASREngine):
    def __init__(self, config: ASRConfig):
        super().__init__(config)
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError:
            raise ImportError(
                "qwen-asr is required for Qwen ASR engine. "
                "Install it with: pip install qwen-asr"
            )

        dtype = torch.bfloat16 if "cuda" in self.config.device else torch.float32
        device = self.config.device if self.config.device != "cuda" else "cuda:0"
        log.info("Loading Qwen model: %s (backend=%s, device=%s)", self.config.model, self.config.backend, device)

        if self.config.backend == "vllm":
            self._model = Qwen3ASRModel.LLM(
                model=self.config.model,
                gpu_memory_utilization=0.7,
                max_inference_batch_size=self.config.batch_size,
                max_new_tokens=256,
                forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
                forced_aligner_kwargs=dict(
                    dtype=dtype,
                    device_map=device,
                ),
            )
        else:
            self._model = Qwen3ASRModel.from_pretrained(
                self.config.model,
                dtype=dtype,
                device_map=device,
                max_inference_batch_size=self.config.batch_size,
                max_new_tokens=256,
                forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
                forced_aligner_kwargs=dict(
                    dtype=dtype,
                    device_map=device,
                ),
            )

    @staticmethod
    def _is_cjk(text: str) -> bool:
        return bool(text and ('\u4e00' <= text[0] <= '\u9fff' or '\u3040' <= text[0] <= '\u30ff'))

    @staticmethod
    def _is_sentence_end(word: str) -> bool:
        return word[-1] in ".!?。！？"

    def transcribe(self, audio_path: str, language: str | None = None, duration: float = 0.0) -> list[Segment]:
        self._load_model()
        language = to_qwen_language(language)
        log.debug("Transcribing %s (language=%s, duration=%.3fs)", audio_path, language, duration)

        results = self._model.transcribe(
            audio=audio_path,
            language=language,
            return_time_stamps=True,
        )

        segments = []
        for r in results:
            if r.time_stamps is not None:
                sentence_words: list[str] = []
                sentence_start: float | None = None
                is_cjk = None
                for ts in r.time_stamps:
                    word = ts.text.strip()
                    if not word:
                        continue
                    if is_cjk is None:
                        is_cjk = self._is_cjk(word)
                    if sentence_start is None:
                        sentence_start = ts.start_time
                    sentence_words.append(word)
                    if self._is_sentence_end(word):
                        sep = "" if is_cjk else " "
                        segments.append(Segment(
                            text=sep.join(sentence_words),
                            start=sentence_start,
                            end=ts.end_time,
                        ))
                        sentence_words = []
                        sentence_start = None
                        is_cjk = None
                if sentence_words:
                    sep = "" if is_cjk else " "
                    segments.append(Segment(
                        text=sep.join(sentence_words),
                        start=sentence_start or 0.0,
                        end=r.time_stamps[-1].end_time,
                    ))
            else:
                text = r.text.strip()
                if text:
                    segments.append(Segment(text=text, start=0.0, end=0.0))

        log.info("Transcription returned %d segments", len(segments))
        return segments
