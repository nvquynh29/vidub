from dataclasses import dataclass


def _default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@dataclass
class ASRConfig:
    engine: str = "faster-whisper"
    model: str = "distil-large-v3"
    backend: str = "transformers"
    device: str = "cuda"
    batch_size: int = 32
    language: str | None = None

    @classmethod
    def from_args(cls, args):
        return cls(
            engine=args.asr_engine,
            model=args.asr_model,
            backend=args.asr_backend,
            device=args.device,
            batch_size=args.batch_size,
            language=args.source_lang,
        )


@dataclass
class TranslateConfig:
    engine: str = "google-translate"
    target_lang: str = "vi"
    source_lang: str | None = None
    api_key: str | None = None
    api_base: str | None = None
    model: str = "unsloth/gemma-4-E2B-it-GGUF"
    model_quant_type: str = "UD-Q4_K_XL"
    max_words: int = 80000

    @classmethod
    def from_args(cls, args):
        return cls(
            engine=args.translate_engine,
            target_lang=args.target_lang,
            source_lang=args.source_lang,
            api_key=args.api_key,
            api_base=args.api_base,
            model=args.translate_model,
            model_quant_type=args.model_quant_type,
            max_words=args.max_words,
        )


@dataclass
class TTSConfig:
    engine: str = "vieneu"
    mode: str = "fast"
    voice_ref: str | None = None
    emotion: str = "natural"
    device: str = "cuda"
    silence_p: float = 0.05
    crossfade_p: float = 0.0

    @classmethod
    def from_args(cls, args):
        return cls(
            engine=args.tts_engine,
            mode=args.tts_mode,
            voice_ref=args.voice_ref,
            emotion=args.tts_emotion,
            device=getattr(args, "tts_device", _default_device()),
            silence_p=getattr(args, "tts_silence_p", 0.05),
            crossfade_p=getattr(args, "tts_crossfade_p", 0.0),
        )
