# vidub — Agentic Workflow Guide

## Project Overview

vidub is a video dubbing tool: ASR → Translate → TTS pipeline.

## Commands

- **`python -m vidub dub`**: Full pipeline (ASR → Translate → TTS → compose video)
- **`python -m vidub sub`**: Subtitles only (ASR → Translate)

## Code Conventions

### Architecture
- **Plugin registry**: Engines are registered via decorators (`@register_asr`, `@register_translate`, `@register_tts`). Add new engines in their respective dirs (`asr/`, `translate/`, `tts/`) and the decorator auto-registers them.
- **Config dataclasses**: `ASRConfig`, `TranslateConfig`, `TTSConfig` in `config.py`. Each has a `from_args(cls, args)` factory.
- **Base classes**: Each engine type inherits from a base in `asr/base.py`, `translate/base.py`, `tts/base.py`.
- **Pipeline**: `DubbingPipeline` (single file) and `BatchPipeline` (folder) in `pipeline.py`.

### Style
- No docstrings or comments unless the logic is non-obvious.
- Type hints everywhere.
- Logger via `vidub.log_utils.get_logger(__name__)`.
- f-strings for logging in info/warn/error, %-style for debug.
- Use `pathlib.Path` over `os.path` where ergonomic.
- Prefer `@dataclass` for simple data containers.

### Testing
- No formal test framework set up. Run `python test.py` for manual integration testing.
- If adding tests, use `pytest` and place in `tests/`.

## Key Files

| File | Purpose |
|---|---|
| `__main__.py` | CLI entry point, argparse |
| `pipeline.py` | `DubbingPipeline`, `BatchPipeline` |
| `config.py` | `ASRConfig`, `TranslateConfig`, `TTSConfig` dataclasses |
| `registry.py` | `register_asr`/`register_translate`/`register_tts` decorators, engine dicts |
| `models.py` | `Segment`, `FileJob` dataclasses |
| `audio.py` | FFmpeg audio extraction, composition, replacement |
| `downloader.py` | URL download helper |
| `srt_utils.py` | SRT read/write |
| `parallel.py` | `scan_folder` for batch input discovery |
| `lang_utils.py` | Language code utilities |
| `log_utils.py` | Logging configuration |
| `asr/` | ASR engine implementations (`whisper.py`, `whisper_cpp.py`, `qwen.py`) |
| `translate/` | Translation engines (`google.py`) |
| `tts/` | TTS engines (`vieneu.py`) |
