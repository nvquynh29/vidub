import logging
import os
from typing import Any

from vidub.tts.base import TTSEngine
from vidub.config import TTSConfig
from vidub.models import Segment
from vidub.registry import register_tts

log = logging.getLogger(__name__)


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _cuda_compute_capability() -> tuple[int, int] | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_capability(0)
        return None
    except (ImportError, RuntimeError):
        return None


def _lmdeploy_supported() -> bool:
    """LMDeploy turbomind requires native bf16 (compute capability >= 8.0)."""
    cap = _cuda_compute_capability()
    if cap is None:
        return False
    return cap[0] >= 8


@register_tts("vieneu")
class VieNeuTTSEngine(TTSEngine):
    def __init__(self, config: TTSConfig):
        super().__init__(config)
        from vieneu import Vieneu

        self._voice_embedding: Any = None
        self._voice_ref_path = config.voice_ref
        self._preset_voice: dict | None = None
        self._model: Any = None
        self._mode: str = ""

        mode = config.mode
        use_cuda = _cuda_available() and config.device == "cuda"

        if mode == "fast":
            if use_cuda and _lmdeploy_supported():
                try:
                    self._model = Vieneu(mode="fast")
                    self._mode = "fast"
                    log.info("Fast mode (LMDeploy GPU)")
                except Exception as exc:
                    log.warning("Fast mode failed: %s", exc)
            else:
                if not use_cuda:
                    log.info("Fast mode requires CUDA, falling back to standard mode")
                else:
                    log.info("GPU lacks native bf16 (need sm_80+), falling back to standard PyTorch CUDA mode")

        # PyTorch CUDA standard mode (v2 highest quality)
        if self._model is None and mode in ("standard", "pytorch", "fast"):
            if use_cuda:
                try:
                    os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")
                    self._model = Vieneu(
                        gguf_filename=None,
                        backbone_device="cuda",
                        codec_repo="neuphonic/neucodec",
                        codec_device="cuda",
                    )
                    self._mode = "standard"
                    log.info("Standard mode (PyTorch CUDA, v2)")
                except Exception as exc:
                    log.warning("Standard PyTorch CUDA mode failed: %s", exc)

        # GGUF standard mode (fallback when PyTorch not available)
        if self._model is None and mode in ("standard", "fast"):
            if use_cuda:
                try:
                    self._model = Vieneu(backbone_device="cuda")
                    self._mode = "standard"
                    log.info("Standard mode (GGUF GPU, CUDA)")
                except Exception as exc:
                    log.warning("Standard GGUF GPU mode failed: %s", exc)
            if self._model is None:
                try:
                    self._model = Vieneu()
                    self._mode = "standard"
                    log.info("Standard mode (GGUF CPU)")
                except Exception as exc:
                    log.warning("Standard GGUF CPU mode failed: %s", exc)

        if self._model is None:
            try:
                kwargs = {"mode": "turbo"}
                if use_cuda:
                    kwargs["device"] = "cuda"
                self._model = Vieneu(**kwargs)
                self._mode = "turbo"
                log.info("Turbo mode (GGUF, device=%s)", "cuda" if use_cuda else "cpu")
            except Exception as exc:
                log.warning("Turbo mode failed: %s", exc)
                raise RuntimeError("No viable TTS backend could be initialized") from exc

        if config.voice_ref:
            if self._mode == "turbo":
                log.info("Encoding voice reference: %s", config.voice_ref)
                self._voice_embedding = self._model.encode_reference(config.voice_ref)
            else:
                log.info("Voice reference will be passed as ref_audio: %s", config.voice_ref)
        else:
            try:
                self._preset_voice = self._model.get_preset_voice("Binh")
                log.info("Using default voice: Binh (Thanh Bình, nam miền Bắc)")
            except Exception:
                log.info("No voice reference or preset voice configured, using model default")

    def synthesize_segments(self, segments: list[Segment], output_dir: str, stem: str = "") -> list[str]:
        valid = [(i, s.text.strip()) for i, s in enumerate(segments) if s.text.strip()]
        if not valid:
            return [""] * len(segments)

        indices, texts = zip(*valid)
        prefix = f"{stem}_" if stem else ""

        infer_kwargs: dict[str, Any] = {}

        if self._mode == "turbo":
            infer_kwargs["show_progress"] = False
            if self._voice_embedding is not None:
                infer_kwargs["voice"] = self._voice_embedding
        else:
            if self._voice_ref_path is not None:
                infer_kwargs["ref_audio"] = self._voice_ref_path
            elif self._preset_voice is not None:
                infer_kwargs["voice"] = self._preset_voice

        if hasattr(self._model, "infer_batch"):
            audio_arrays = self._model.infer_batch(list(texts), **infer_kwargs)
        else:
            audio_arrays = [self._model.infer(text=t, **infer_kwargs) for t in texts]

        file_map: dict[int, str] = {}
        for idx, audio in zip(indices, audio_arrays):
            path = os.path.join(output_dir, f"{prefix}seg_{idx:04d}.wav")
            self._model.save(audio, path)
            file_map[idx] = path

        return [file_map.get(i, "") for i in range(len(segments))]

    def close(self) -> None:
        if self._model is not None:
            self._model.close()
            self._model = None
