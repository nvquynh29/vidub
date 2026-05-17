import os
import subprocess
import tempfile
from pathlib import Path

from vidub.log_utils import get_logger
from vidub.models import Segment


log = get_logger("vidub.audio")

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a"}
MEDIA_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}


def extract_audio(input_path: str, sample_rate: int = 16000, output_dir: str | None = None) -> str:
    ext = Path(input_path).suffix.lower()
    if ext in AUDIO_EXTENSIONS:
        log.debug("Audio file detected, no extraction needed: %s", input_path)
        return input_path

    if ext not in MEDIA_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {ext}")

    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError("pydub is required for audio extraction. Install it with: pip install pydub")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{Path(input_path).stem}.wav")
    else:
        out_dir = tempfile.mkdtemp(prefix="vdub_audio_")
        out_path = os.path.join(out_dir, f"{Path(input_path).stem}.wav")

    log.debug("Extracting audio from %s", input_path)
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(sample_rate).set_channels(1)
    audio.export(out_path, format="wav")

    log.info("Audio extracted: %s (%d Hz, mono)", out_path, sample_rate)
    return out_path


def get_audio_duration(audio_path: str) -> float:
    try:
        from pydub import AudioSegment
        return AudioSegment.from_file(audio_path).duration_seconds
    except Exception:
        return 0.0


def _speed_audio(input_path: str, speed: float, output_path: str) -> None:
    if abs(speed - 1.0) < 0.01:
        subprocess.run(["cp", input_path, output_path], check=True)
        return
    speed = max(0.5, min(speed, 2.0))
    filters = [f"atempo={speed}"]
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-filter:a", ",".join(filters),
        "-ac", "1",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def compose_audio(
    segments: list[Segment],
    audio_files: list[str],
    output_dir: str,
    original_audio_path: str | None = None,
    crossfade_ms: int = 20,
    stem: str = "",
) -> str:
    from pydub import AudioSegment

    if not segments or not audio_files:
        raise ValueError("No segments or audio files to compose")

    total_end = max(s.end for s in segments)
    total_ms = int(total_end * 1000) + 1000

    full = AudioSegment.silent(duration=total_ms)

    tmp_dir = tempfile.mkdtemp(prefix="vdub_speed_")
    prev_end_ms = 0
    for seg, path in zip(segments, audio_files):
        if not path or not os.path.exists(path):
            continue
        seg_audio = AudioSegment.from_file(path)
        pos = int(seg.start * 1000)
        max_dur = int((seg.end - seg.start) * 1000)

        if len(seg_audio) > max_dur:
            speed = len(seg_audio) / max(1, max_dur)
            speed = max(0.5, min(speed, 2.0))
            speed_path = os.path.join(tmp_dir, f"speed_{Path(path).stem}.wav")
            _speed_audio(path, speed, speed_path)
            seg_audio = AudioSegment.from_file(speed_path)
            if len(seg_audio) > max_dur:
                seg_audio = seg_audio[:max_dur]

        seg_audio = seg_audio.fade_in(5).fade_out(5)

        overlap = prev_end_ms - pos
        if overlap > 0:
            crossfade = min(crossfade_ms, overlap // 2, len(seg_audio))
            if crossfade > 0:
                full = full[:pos] + full[pos:].overlay(seg_audio, position=0, gain_during_overlay=-6)
            else:
                full = full.overlay(seg_audio, position=pos)
        else:
            full = full.overlay(seg_audio, position=pos)

        seg_dur = min(len(seg_audio), max_dur)
        prev_end_ms = pos + seg_dur

    try:
        for f in os.listdir(tmp_dir):
            os.remove(os.path.join(tmp_dir, f))
        os.rmdir(tmp_dir)
    except OSError:
        pass

    prefix = f"{stem}_" if stem else ""
    out_path = os.path.join(output_dir, f"{prefix}tts_full.wav")
    full.export(out_path, format="wav")
    log.info("Composed full TTS audio: %s (%.2f sec)", out_path, full.duration_seconds)
    return out_path


def replace_video_audio(video_path: str, audio_path: str, output_path: str) -> str:
    log.info("Replacing audio in %s with %s -> %s", video_path, audio_path, output_path)
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    log.info("Output video: %s", output_path)
    return output_path


def cleanup_audio(audio_path: str) -> None:
    if audio_path:
        os.remove(audio_path)
        parent = os.path.dirname(audio_path)
        try:
            os.rmdir(parent)
        except OSError:
            pass
        log.debug("Cleaned up temp audio: %s", audio_path)
