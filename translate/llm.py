import atexit
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

from vidub.translate.base import TranslationEngine
from vidub.config import TranslateConfig
from vidub.lang_utils import get_language_name
from vidub.log_utils import get_logger
from vidub.models import Segment
from vidub.registry import register_translate


log = get_logger("vidub.translate.openai")

_SEPARATOR = "\n---\n"
_MAX_CHARS = 8000
_LOCAL_BASE = "http://127.0.0.1:8888"
_LLAMA_SERVER = os.path.expanduser("~/.unsloth/llama.cpp/llama-server")
_ACTIVE_PROCESS: subprocess.Popen | None = None
_NUM_TRANSLATE_WORKERS = 2


def stop_server():
    global _ACTIVE_PROCESS
    if _ACTIVE_PROCESS:
        log.info("Shutting down llama-server")
        _ACTIVE_PROCESS.terminate()
        try:
            _ACTIVE_PROCESS.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _ACTIVE_PROCESS.kill()
        _ACTIVE_PROCESS = None


@register_translate("openai")
class OpenAITranslateEngine(TranslationEngine):
    def __init__(self, config: TranslateConfig):
        super().__init__(config)
        self._process: subprocess.Popen | None = None
        self._client = None
        atexit.register(self._cleanup)

    def _cleanup(self):
        stop_server()
        if hasattr(self, "_stderr_file"):
            self._stderr_file.close()

    def _get_client(self):
        if self._client:
            return self._client

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAI translation engine. "
                "Install it with: pip install openai"
            )

        if self.config.api_key:
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base,
            )
        else:
            if not os.path.isfile(_LLAMA_SERVER):
                log.error("llama-server not found at %s. Install Unsloth first:\n  curl -fsSL https://unsloth.ai/install.sh | sh", _LLAMA_SERVER)
                sys.exit(1)

            hf_model = f"{self.config.model}:{self.config.model_quant_type}"
            cmd = [
                _LLAMA_SERVER,
                "-hf", hf_model,
                "-np", str(_NUM_TRANSLATE_WORKERS),
                "--temp", "1.0",
                "--alias", self.config.model,
                "--port", "8888",
                "--reasoning", "off",
            ]
            log.info("Starting llama-server: %s", " ".join(cmd))
            self._stderr_file = open("llama-server.log", "a")
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr_file,
            )
            global _ACTIVE_PROCESS
            _ACTIVE_PROCESS = self._process

            self._wait_for_server(f"{_LOCAL_BASE}/health")

            self._client = OpenAI(
                base_url=f"{_LOCAL_BASE}/v1",
                api_key="sk-no-key-required",
            )

        return self._client

    def _wait_for_server(self, health_url: str, timeout: int = 120):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._process and self._process.poll() is not None:
                log.error("llama-server exited prematurely with code %d", self._process.returncode)
                if hasattr(self, "_stderr_file"):
                    self._stderr_file.flush()
                sys.exit(1)
            try:
                urllib.request.urlopen(health_url, timeout=2)
                log.info("llama-server is ready")
                return
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(1)
        log.error("llama-server did not start within %d seconds", timeout)
        sys.exit(1)

    def translate(
        self, segments: list[Segment], target_lang: str, source_lang: str | None = None
    ) -> list[Segment]:
        if not segments:
            return []

        client = self._get_client()
        source_name = get_language_name(source_lang)
        target_name = get_language_name(target_lang)

        word_batches = self._split_segments_by_words(segments)
        if len(word_batches) > 1:
            log.info("Split %d segments into %d word-count batches (max %d words each)", len(segments), len(word_batches), self.config.max_words)

        all_translated: list[Segment] = []
        for batch in word_batches:
            texts = [seg.text.strip() for seg in batch]
            translated_texts = self._translate_texts(client, texts, source_name, target_name)
            all_translated.extend(
                Segment(text=t, start=seg.start, end=seg.end)
                for seg, t in zip(batch, translated_texts)
            )

        return all_translated

    def _translate_texts(
        self, client, texts: list[str], source_name: str, target_name: str
    ) -> list[str]:
        result: list[str] = []
        i = 0
        while i < len(texts):
            batch_texts = [texts[i]]
            char_count = len(texts[i])
            j = i + 1
            while j < len(texts):
                added = len(_SEPARATOR) + len(texts[j])
                if char_count + added > _MAX_CHARS:
                    break
                batch_texts.append(texts[j])
                char_count += added
                j += 1

            combined = _SEPARATOR.join(batch_texts)
            prompt = (
                f"You are a professional video subtitle translator. "
                f"Translate the following {source_name} text into {target_name}. "
                f"CRITICAL: The input blocks are separated by '---'. You MUST return EXACTLY the same number of blocks, each separated by '---'. "
                f"Never merge, split, or drop any block. "
                f"Do not add any commentary, explanations, or extra lines. "
                f"If a word or phrase is an object name (person, place, brand, proper noun), keep it as-is. "
                f"Ensure the translation matches the tone of the original video.\n\n"
                f"{combined}"
            )

            response = client.responses.create(
                model=self.config.model,
                input=prompt,
                temperature=1.0,
                stream=False,
            )

            translated = response.output_text.strip()
            split = [s.strip() for s in translated.split("---")]

            if len(split) == len(batch_texts):
                result.extend(split)
            else:
                log.warning(
                    "Expected %d segments, got %d. Using original text for missing segments.",
                    len(batch_texts), len(split),
                )
                for idx in range(len(batch_texts)):
                    if idx < len(split):
                        result.append(split[idx])
                    else:
                        result.append(batch_texts[idx])

            i = j

        return result
