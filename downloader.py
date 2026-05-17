import os
import re
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

from vidub.log_utils import get_logger


log = get_logger("vidub.downloader")

_URL_REGEX = re.compile(r"^(https?://|www\.)", re.IGNORECASE)


def is_url(path: str) -> bool:
    return bool(_URL_REGEX.match(path))


def download(url: str, output_dir: str | None = None) -> str:
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="vdub_dl_")
    else:
        os.makedirs(output_dir, exist_ok=True)

    log.info("Downloading from %s", url)
    with urllib.request.urlopen(url, timeout=120) as resp:
        content = resp.read()
    disp = resp.headers.get("Content-Disposition", "")
    fname = None
    if "filename=" in disp:
        fname = disp.split("filename=")[-1].strip('";\'')
    if not fname:
        fname = Path(urllib.parse.urlparse(url).path).name or "download"
    out = os.path.join(output_dir, fname)
    with open(out, "wb") as f:
        f.write(content)
    log.info("Downloaded to %s", out)
    return out
