#!/usr/bin/env python3
"""Install dependencies for vidub using uv."""

import shutil
import subprocess
from typing import Sequence


def _run(cmd: Sequence[str]) -> None:
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _run_capture(cmd: Sequence[str]) -> str:
    print(f"Running: {' '.join(cmd)}")
    p = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return p.stdout.strip()


def _verify_runtime() -> None:
    print("\n[verify] Checking torch/torchaudio/onnxruntime GPU runtime...")
    py = (
        "import torch, torchaudio, onnxruntime as ort; "
        "print('torch', torch.__version__); "
        "print('torch_cuda', torch.version.cuda); "
        "print('cuda_ok', torch.cuda.is_available()); "
        "print('torchaudio', torchaudio.__version__); "
        "print('providers', ','.join(ort.get_available_providers()))"
    )
    try:
        out = _run_capture(["uv", "run", "python", "-c", py])
    except subprocess.CalledProcessError as e:
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        raise RuntimeError("Verification command failed; see stdout/stderr above") from e
    print(out)

    lines = out.splitlines()
    cuda_ok = any("cuda_ok True" in ln for ln in lines)
    providers_line = next((ln for ln in lines if ln.startswith("providers ")), "")
    has_cuda_provider = "CUDAExecutionProvider" in providers_line

    if not cuda_ok:
        raise RuntimeError("Verification failed: torch.cuda.is_available() is False")
    if not has_cuda_provider:
        raise RuntimeError("Verification failed: onnxruntime missing CUDAExecutionProvider")
    print("[verify] ✅ GPU runtime verification passed")


def main() -> None:
    if not shutil.which("uv"):
        raise RuntimeError("uv not found. Please install uv first: https://docs.astral.sh/uv/")

    print("Updating apt package index and installing ffmpeg...")
    _run(["sudo", "apt", "update"])
    _run(["sudo", "apt", "install", "-y", "ffmpeg"])

    # Keep these aligned to avoid ABI mismatch (torch/torchaudio/vision)
    torch_index = "https://download.pytorch.org/whl/cu128"

    print("Uninstalling potentially conflicting runtime packages...")
    _run([
        "uv", "pip", "uninstall",
        "torch", "torchaudio", "torchvision", "onnxruntime", "onnxruntime-gpu",
    ])

    print("Installing CUDA-enabled PyTorch stack...")
    _run([
        "uv", "pip", "install",
        "--index-url", torch_index,
        "torch", "torchaudio", "torchvision",
    ])

    print("Installing ONNX Runtime GPU...")
    _run(["uv", "pip", "install", "onnxruntime-gpu"])

    # Core dependencies
    core = [
        "pydub",
        "tqdm",
        "charset-normalizer",
        "deep-translator",
        "openai",
    ]

    # Optional ASR engines
    asr = [
        "faster-whisper",
        # "pywhispercpp",
        # "qwen-asr",
    ]

    _run(["uv", "pip", "install", *core])
    _run(["uv", "pip", "install", *asr])
    _run(["uv", "pip", "install", "vieneu[gpu]"])

    print("Re-applying CUDA-aligned runtime packages after vieneu install...")
    _run([
        "uv", "pip", "install",
        "--index-url", torch_index,
        "torch", "torchaudio", "torchvision",
    ])
    _run(["uv", "pip", "install", "onnxruntime-gpu"])

    _verify_runtime()


if __name__ == "__main__":
    main()
