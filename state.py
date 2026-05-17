import hashlib
import json
import os
from pathlib import Path

from vidub.log_utils import get_logger
from vidub.models import FileJob


log = get_logger("vidub.state")

_STATE_FILE = ".vidub_state.json"

STAGES = [
    "audio_extracted",
    "asr_done",
    "srt_written",
    "translated",
    "composed",
]


def compute_config_hash(asr_config, translate_config, tts_config) -> str:
    h = hashlib.sha256()
    parts = [
        asr_config.engine or "",
        asr_config.model or "",
        asr_config.backend or "",
        str(asr_config.device or ""),
        translate_config.engine if translate_config else "none",
        translate_config.target_lang if translate_config else "none",
        translate_config.source_lang or "none" if translate_config else "none",
        translate_config.model if translate_config else "none",
        tts_config.engine if tts_config else "none",
        tts_config.mode if tts_config else "none",
    ]
    h.update("|".join(parts).encode())
    return h.hexdigest()[:12]


def _state_path(output_root: Path) -> Path:
    return output_root / _STATE_FILE


def load_state(output_root: Path, config_hash: str) -> dict[str, list[str]] | None:
    """Load saved state if config matches. Returns {rel_path: [stages...]} or None."""
    path = _state_path(output_root)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        saved_hash = data.get("config_hash", "")
        if saved_hash != config_hash:
            log.warning(
                "Config hash mismatch (saved=%s, current=%s). Starting from scratch.",
                saved_hash, config_hash,
            )
            return None
        jobs = data.get("jobs", {})
        done = sum(1 for j in jobs.values() if "composed" in j.get("completed_stages", []))
        if done:
            log.info("Resuming from saved state: %d jobs fully completed", done)
        return {str(k): v.get("completed_stages", []) for k, v in jobs.items()}
    except Exception as e:
        log.warning("Failed to load state file: %s. Starting from scratch.", e)
        return None


def _init_state(output_root: Path, jobs: list[FileJob], config_hash: str) -> None:
    """Create initial state file with all jobs and empty completed_stages."""
    state_jobs = {}
    for j in jobs:
        state_jobs[str(j.rel_path)] = {
            "input_path": j.input_path,
            "stem": j.stem,
            "completed_stages": [],
        }
    data = {"config_hash": config_hash, "jobs": state_jobs}
    path = _state_path(output_root)
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def mark_stage_completed(output_root: Path, rel_path: str, stage: str) -> None:
    """Add a stage to a job's completed list and save."""
    path = _state_path(output_root)
    if not path.exists():
        return
    try:
        with open(path) as f:
            data = json.load(f)
        key = str(rel_path)
        if key not in data["jobs"]:
            data["jobs"][key] = {"completed_stages": []}
        if stage not in data["jobs"][key]["completed_stages"]:
            data["jobs"][key]["completed_stages"].append(stage)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        log.warning("Failed to update state for %s: %s", rel_path, e)
