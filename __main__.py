#!/usr/bin/env python3
import argparse
import os
import sys

from .config import ASRConfig, TranslateConfig, TTSConfig, _default_device
from .downloader import is_url
from .log_utils import configure_logging, get_logger
from .pipeline import DubbingPipeline, BatchPipeline
from .registry import ASR_ENGINES, TRANSLATE_ENGINES


log = get_logger("vidub")


def _add_log_level(p):
    p.add_argument("--log-level", default="info", choices=["debug", "info", "warn", "error"],
                   help="Logging level")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m vidub", description="Video Dubbing Tool")
    sub = parser.add_subparsers(dest="command", required=True)

    dub = sub.add_parser("dub", help="Full pipeline: ASR -> Translate -> TTS")
    _add_log_level(dub)
    _add_common_args(dub)
    _add_asr_args(dub)
    _add_translate_args(dub)
    _add_tts_args(dub)
    dub.add_argument("-i", "--input", required=True, help="Video/audio file, URL, or folder path")
    dub.add_argument("-o", "--output", default="./output", help="Output directory")

    sub_cmd = sub.add_parser("sub", help="ASR -> Translate (generate translated subtitles from video/audio)")
    _add_log_level(sub_cmd)
    _add_common_args(sub_cmd)
    _add_asr_args(sub_cmd)
    _add_translate_args(sub_cmd)
    sub_cmd.add_argument("-i", "--input", required=True, help="Video/audio file, URL, or folder path")
    sub_cmd.add_argument("-o", "--output", default="./output", help="Output directory")

    return parser


def _add_common_args(p):
    default_dev = _default_device()
    p.add_argument("--device", default=default_dev, choices=["cuda", "cpu"], help="Device to use (auto: %s)" % default_dev)
    p.add_argument("--source-lang", "-sl", default=None, help="Source language code (auto-detect if omitted)")


def _add_asr_args(p):
    p.add_argument("--asr-engine", default="faster-whisper", choices=list(ASR_ENGINES.keys()) or ["faster-whisper"], help="ASR engine")
    p.add_argument("--asr-model", default="distil-large-v3", help="ASR model name/path")
    p.add_argument("--asr-backend", default="transformers", choices=["transformers", "vllm"], help="ASR backend")
    p.add_argument("--batch-size", type=int, default=32, help="ASR batch size")


def _add_translate_args(p):
    p.add_argument("--translate-engine", default="google-translate", choices=list(TRANSLATE_ENGINES.keys()) or ["google-translate", "openai"], help="Translation engine")
    p.add_argument("--target-lang", "-tl", default="vi", help="Target language code")
    p.add_argument("--api-key", default=None, help="API key (for OpenAI/LLM engine)")
    p.add_argument("--api-base", default="http://localhost:8888/v1", help="API base URL (for OpenAI engine)")
    p.add_argument("--translate-model", default="unsloth/gemma-4-E2B-it-GGUF", help="Model HF repo ID (for OpenAI/LLM engine)")
    p.add_argument("--model-quant-type", default="UD-Q4_K_XL", help="Model quantization type (for local LLM)")
    p.add_argument("--max-words", type=int, default=80000, help="Maximum words per translation batch (split SRT into batches if exceeded)")


def _add_tts_args(p):
    p.add_argument("--tts-engine", default="vieneu", help="TTS engine")
    p.add_argument("--tts-mode", default="fast", choices=["standard", "turbo", "fast"], help="TTS mode")
    p.add_argument("--voice-ref", default=None, help="Voice reference audio path")
    p.add_argument("--tts-emotion", default="natural", choices=["natural", "storytelling"], help="TTS emotion")
    default_dev = _default_device()
    p.add_argument("--tts-device", default=default_dev, choices=["cuda", "cpu"], help="Device to use for TTS inference (auto: %s)" % default_dev)


def cmd_dub(args):
    asr_cfg = ASRConfig.from_args(args)
    translate_cfg = TranslateConfig.from_args(args)
    tts_cfg = TTSConfig.from_args(args)
    pipeline = DubbingPipeline(asr_cfg, translate_cfg, tts_cfg)

    if is_url(args.input) or os.path.isfile(args.input):
        result = pipeline.run(args.input, args.output)
        results = [result]
    elif os.path.isdir(args.input):
        output_dir = os.path.join(args.output, os.path.basename(os.path.normpath(args.input)))
        batch = BatchPipeline(pipeline)
        results = batch.run(args.input, output_dir)
    else:
        log.error("input is not a file, folder, or valid URL: %s", args.input)
        sys.exit(1)

    for r in results:
        parts = []
        if "srt_path" in r:
            parts.append(f"SRT: {r['srt_path']}")
        if "translated_srt_path" in r:
            parts.append(f"Translated: {r['translated_srt_path']}")
        if "output_video" in r:
            parts.append(f"Video: {r['output_video']}")
        print(" | ".join(parts))


def cmd_sub(args):
    asr_cfg = ASRConfig.from_args(args)
    translate_cfg = TranslateConfig.from_args(args)
    pipeline = DubbingPipeline(asr_cfg, translate_cfg)

    if is_url(args.input) or os.path.isfile(args.input):
        result = pipeline.run(args.input, args.output)
        results = [result]
    elif os.path.isdir(args.input):
        output_dir = os.path.join(args.output, os.path.basename(os.path.normpath(args.input)))
        batch = BatchPipeline(pipeline)
        results = batch.run(args.input, output_dir)
    else:
        log.error("input is not a file, folder, or valid URL: %s", args.input)
        sys.exit(1)

    for r in results:
        if "translated_srt_path" in r:
            print(f"Translated: {r['translated_srt_path']}")
        elif "srt_path" in r:
            print(f"SRT: {r['srt_path']}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(getattr(args, "log_level", "info"))
    log.debug("Starting vidub with args: %s", vars(args))
    if args.command == "dub":
        cmd_dub(args)
    elif args.command == "sub":
        cmd_sub(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
