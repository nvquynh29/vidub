# State & Resumable Pipeline

vidub supports **automatic resumption** of interrupted batch and single-file pipelines. Instead of restarting from scratch, it detects previously completed work by checking **on-disk artifacts** and a **state file**.

---

## How It Works

### State File

After building the job list, the pipeline writes a state file at:

```
<output_root>/.vidub_state.json
```

This JSON file records the set of **completed stages** for every file in the batch:

```json
{
  "config_hash": "a1b2c3d4e5f6",
  "jobs": {
    "subfolder/video1.mp4": {
      "input_path": "/abs/path/videos/subfolder/video1.mp4",
      "stem": "video1",
      "completed_stages": ["audio_extracted", "asr_done", "srt_written", "translated", "composed"]
    },
    "subfolder/video2.mp4": {
      "input_path": "/abs/path/videos/subfolder/video2.mp4",
      "stem": "video2",
      "completed_stages": ["audio_extracted", "asr_done", "srt_written", "translated"]
    }
  }
}
```

### Config Hash

The `config_hash` is a SHA-256 digest of the current pipeline configuration:

- ASR: engine, model, backend, device
- Translate: engine, target_lang, source_lang, model
- TTS: engine, mode

If the configuration changes between runs, the hash won't match and the pipeline **starts from scratch** with a warning.

---

## Stages

Each file progresses through these stages (tracked individually):

| Stage | Description | Artifact Check |
|---|---|---|
| `audio_extracted` | Audio extracted to WAV | `tmp/audios/<input_name>/original/<rel_path>/<stem>.wav` |
| `asr_done` | Speech-to-text completed | (in-memory, tracked by state only) |
| `srt_written` | Original SRT written to disk | `<output_root>/<rel_path>/<stem>.original.srt` |
| `translated` | Translation completed + SRT written | `<output_root>/<rel_path>/<stem>.srt` |
| `composed` | TTS generated + audio composed + video output | `<output_root>/<rel_path>/<stem>.<ext>` (video) |

---

## Resume Behavior

### Batch Mode (folder input)

When `BatchPipeline.run()` is called:

1. Scans the input folder and builds the job list.
2. Calls `load_state()` to check for `.vidub_state.json` in the output directory.
3. If a valid state is found (matching `config_hash`), the completed stages for each file are loaded into memory.
4. Each stage method checks the state **per file** before processing:

```python
for j in tqdm(jobs, desc="Translating"):
    if state and "translated" in state.get(str(j.rel_path), []):
        # Skip: load translated SRT from disk
        j.translated = read_srt(srt_path)
        continue
    # Process normally
    j.translated = translate_engine.translate(...)
    write_srt(srt_path, j.translated)
    mark_stage_completed(output_root, str(j.rel_path), "translated")
```

5. After each file completes a stage, `mark_stage_completed()` updates `.vidub_state.json` immediately.

**Example:** If interrupted at file 20/42 during translation:
- Files 1-19 have `"translated"` in their completed stages → skipped on re-run
- Files 20-42 have no `"translated"` stage → processed from file 20 onward
- Prior stages (audio_extracted, asr_done, srt_written) are also checked per-file

### Single-File Mode

`DubbingPipeline.run()` does not use a state file. Instead, it checks for existing output artifacts directly:

| Artifact | If Exists |
|---|---|
| `tmp/audios/original/<stem>.wav` | Skip audio extraction |
| `<output>/<stem>.original.srt` | Skip ASR, load segments from SRT |
| `<output>/<stem>.srt` | Skip translation, load from SRT |
| `<output>/<stem>.<ext>` | Skip TTS + compose entirely |

This makes single-file re-runs instant — if the output video already exists, the function returns immediately.

---

## Implementation Details

### `vidub/state.py`

| Function | Purpose |
|---|---|
| `compute_config_hash()` | Generates a deterministic hash of pipeline config |
| `load_state(output_root, config_hash)` | Loads `.vidub_state.json`, validates hash, returns `{rel_path: [stages...]}` or `None` |
| `_init_state(output_root, jobs, config_hash)` | Creates initial state file with empty completed_stages for all jobs |
| `mark_stage_completed(output_root, rel_path, stage)` | Appends a stage to a job's completed list and writes to disk |

### `vidub/pipeline.py` — `BatchPipeline`

| Method | Changes for Resume |
|---|---|
| `run()` | Computes `config_hash`, calls `load_state()`, calls `_init_state()`, passes `state` to all stage methods |
| `_stage1_audio_extraction()` | Checks `audio_extracted` + WAV file existence |
| `_stage2_asr()` | Checks `asr_done` in state |
| `_write_original_srts()` | Checks `srt_written` + SRT file existence; loads segments from existing SRT |
| `_stage3_translation()` | Checks `translated` + translated SRT existence |
| `_stage4_tts_and_compose()` | Checks `composed` + output video existence |

### `vidub/pipeline.py` — `DubbingPipeline`

| Artifact | Check Location |
|---|---|
| `<stem>.wav` | Before `extract_audio()` |
| `<stem>.original.srt` | Before ASR engine call |
| `<stem>.srt` | Before translation engine call |
| `<stem>.<ext>` (output video) | Before TTS + compose |

---

## Limitations

1. **TTS partial resume**: Per-segment WAV generation happens inside `tts_engine.synthesize_segments()`, which is a single opaque call. If TTS is interrupted mid-file, the entire TTS stage for that file is re-run on resume. This is an acceptable tradeoff since TTS is fast per-file.

2. **Config changes**: Changing engine, model, language, or TTS mode invalidates the state. The pipeline logs a warning and starts from scratch.

3. **Deleted input files**: If an input file is removed from the folder between runs, its state entry is silently ignored (the file won't appear in the new job list).

4. **State file corruption**: If `.vidub_state.json` is corrupted or unreadable, the pipeline logs a warning and starts from scratch.

5. **Concurrent runs**: Running two instances of vidub on the same output directory simultaneously is not supported and may corrupt the state file.
