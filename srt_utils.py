from charset_normalizer import from_bytes

from vidub.log_utils import get_logger
from vidub.models import Segment


log = get_logger("vidub.srt_utils")


def _format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _parse_timestamp(ts: str) -> float:
    parts = ts.replace(",", ".").split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    secs = float(parts[2])
    return hours * 3600 + minutes * 60 + secs


def segments_to_srt(segments: list[Segment]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        start = _format_timestamp(seg.start)
        end = _format_timestamp(seg.end)
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)


def srt_to_segments(srt_text: str) -> list[Segment]:
    segments = []
    blocks = srt_text.strip().split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        time_line = lines[1]
        parts = time_line.split("-->")
        if len(parts) != 2:
            continue
        start = _parse_timestamp(parts[0].strip())
        end = _parse_timestamp(parts[1].strip())
        text = "\n".join(lines[2:]).strip()
        segments.append(Segment(text=text, start=start, end=end))
    return segments


def read_srt(path: str) -> list[Segment]:
    log.debug("Reading SRT: %s", path)
    with open(path, "rb") as f:
        raw = f.read()
    result = from_bytes(raw).best()
    text = str(result) if result else raw.decode("utf-8", errors="replace")
    segments = srt_to_segments(text)
    log.debug("Read %d segments from %s", len(segments), path)
    return segments


def write_srt(path: str, segments: list[Segment]) -> None:
    log.debug("Writing SRT with %d segments: %s", len(segments), path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(segments_to_srt(segments))
