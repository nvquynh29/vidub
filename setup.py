#!/usr/bin/env python3
"""Install dependencies for vidub using uv pip."""

import shutil
import subprocess
import sys


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _run(cmd: list[str]) -> None:
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    if not shutil.which("uv"):
        print("uv not found. Installing uv...")
        _run([sys.executable, "-m", "pip", "install", "uv"])

    # Core dependencies
    core = [
        "pydub",
        "tqdm",
        "charset-normalizer",
        "deep-translator",
        "openai",
        "torch",
    ]

    # Optional ASR engines
    asr = [
        "faster-whisper",
        "pywhispercpp",
        "qwen-asr",
    ]

    _run([sys.executable, "-m", "uv", "pip", "install", *core])
    _run([sys.executable, "-m", "uv", "pip", "install", *asr])

    if _has_gpu():
        print("GPU detected — installing vieneu[gpu] (includes LMDeploy)")
        _run([sys.executable, "-m", "uv", "pip", "install", "vieneu[gpu]"])
    else:
        print("No GPU — installing vieneu (CPU only)")
        _run([sys.executable, "-m", "uv", "pip", "install", "vieneu"])


if __name__ == "__main__":
    main()
