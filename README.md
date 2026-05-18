# vidub

Video dubbing tool with ASR, translation, and TTS pipeline.

## Requirements

### FFmpeg

```bash
sudo apt update
sudo apt install -y ffmpeg
```

### CUDA 12.8 (for GPU inference with faster-whisper)

```bash
sudo apt-get install -y libcublas-12-8
```

### Recommended GPU stack (RTX 6000 PRO)

For best throughput and stability on RTX 6000 PRO, use:

- CUDA `12.8`
- PyTorch `2.11`
- GPU codec for TTS (`neuphonic/neucodec` with `--tts-codec-device cuda`)

You can install the validated runtime stack with:

```bash
python setup.py
```

## Installation

```bash
pip install -e .
```

### Unsloth (for local LLM translation)

```bash
curl -fsSL https://unsloth.ai/install.sh | sh
```

Optional engine dependencies:

```bash
pip install -e ".[qwen]"           # Qwen3 ASR
pip install -e ".[faster-whisper]" # faster-whisper ASR (recommended)
pip install -e ".[whisper-cpp]"    # whisper.cpp ASR
pip install -e ".[translate]"      # Google Translate and OpenAI/LLM translation
pip install -e ".[tts]"            # VieNeu TTS (GPU with CUDA, PyTorch + GGUF)

# For fast mode (LMDeploy GPU — recommended for ADA GPU and later):
uv pip install lmdeploy
```

## Usage

### Full pipeline: ASR → Translate → TTS

```bash
python -m vidub dub -i input.mp4 -o ./output -sl en -tl vi
```

### ASR → Translate (subtitles only)

```bash
python -m vidub sub -i input.mp4 -o ./output -sl en -tl vi
```

### Local LLM translation (Unsloth)

First, start the Unsloth server in a separate terminal:

```bash
unsloth run --model unsloth/gemma-4-E2B-it-GGUF --disable-tools
```

Then run vidub with the API key from the server output:

```bash
python -m vidub sub -i input.mp4 -o ./output -sl en -tl vi \
  --translate-engine llm \
  --api-key <key_from_unsloth>
```

If no API key is provided, the engine defaults to `http://localhost:8888/v1` with no auth for local servers that don't require one.

### Full pipeline with TTS options

```bash
python -m vidub dub -i input.mp4 -o ./output -sl en -tl vi \
  --tts-mode fast --tts-device cuda --voice-ref ref.wav
```

### TTS with voice cloning

```bash
python -m vidub dub -i input.mp4 -o ./output -sl en -tl vi \
  --voice-ref /path/to/3-5s_reference_audio.wav
```

### Batch processing (folder of videos)

```bash
python -m vidub sub -i ./videos/ -o ./output -sl en -tl vi \
  --asr-engine faster-whisper
```

### Recommended command (RTX 6000 PRO)

```bash
python -m vidub dub -i input_folder -o ./output -sl en -tl vi --device cuda --tts-device cuda --tts-mode standard --tts-profile max-gpu --tts-backbone-repo pnnbao-ump/VieNeu-TTS --tts-codec-repo neuphonic/neucodec --tts-codec-device cuda --tts-workers 0 --log-level info
```

`--tts-workers 0` enables auto-tuning (the runtime picks a safe worker count automatically).

## TTS

The pipeline uses [VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS) for Vietnamese text-to-speech synthesis.

| Argument | Default | Description |
|---|---|---|
| `--tts-engine` | `vieneu` | TTS engine |
| `--tts-mode` | `standard` | `turbo` (GGUF), `standard` (PyTorch GPU/GGUF CPU), `fast` (LMDeploy GPU) |
| `--tts-profile` | `max-gpu` | Runtime profile: `max-gpu` or `balanced` |
| `--tts-device` | `cuda` | Target device: `cuda` (GPU) or `cpu` |
| `--tts-workers` | `0` | Number of concurrent TTS workers in batch mode (`0` = auto-tune) |
| `--tts-emotion` | `natural` | `natural` or `storytelling` (standard mode only) |
| `--voice-ref` | — | Path to 3–5s reference audio for zero-shot voice cloning |

**Mode selection** follows the [official VieNeu-TTS SDK docs](https://docs.vieneu.io/docs/sdk/overview):

| `--tts-mode` | CUDA available | Backend | Init |
|---|---|---|---|
| `standard` | Yes | `Vieneu(backbone_device="cuda")` — PyTorch full precision on GPU | `[gpu]` extras |
| `standard` | No | `Vieneu()` — GGUF Q4 on CPU | base install |
| `fast` | Yes | `Vieneu(mode="fast")` — LMDeploy GPU | `[gpu]` extras |
| `fast` | No | falls back to standard (PyTorch/GGUF) | — |
| `turbo` | Either | `Vieneu(mode="turbo")` — lightweight GGUF | base install |

When `--tts-mode` fails to initialize, the engine automatically falls back through the chain: preferred mode → standard (PyTorch GPU if CUDA) → standard (GGUF CPU) → turbo (GGUF CPU). This ensures the pipeline always completes even without GPU.

### Ambient sound preservation

The dubbed video preserves the original audio's ambient/background sounds in gaps between speech segments. Instead of inserting pure silence, the pipeline uses the original audio as a canvas, mutes only the speech portions, and overlays the synthesized TTS audio. This eliminates the "weird sounds"/void effect in silent gaps typical of naive dubbing pipelines.
