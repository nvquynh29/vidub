import concurrent.futures
import gc
import os
import tempfile
import time
from pathlib import Path

from tqdm import tqdm

from vidub.audio import (
    AUDIO_EXTENSIONS,
    MEDIA_EXTENSIONS,
    cleanup_audio,
    compose_audio,
    extract_audio,
    get_audio_duration,
    replace_video_audio,
)
from vidub.config import ASRConfig, TranslateConfig, TTSConfig
from vidub.downloader import is_url, download as download_url
from vidub.log_utils import get_logger
from vidub.models import FileJob, Segment
from vidub.parallel import scan_folder
from vidub.registry import ASR_ENGINES, TRANSLATE_ENGINES, TTS_ENGINES
from vidub.srt_utils import read_srt, write_srt
from vidub.state import (
    STAGES,
    compute_config_hash,
    load_state,
    _init_state,
    mark_stage_completed,
)


log = get_logger("vidub.pipeline")

_GPU_TAG = "GPU"
_CPU_TAG = "CPU"

_BASE_DIR = Path("tmp/audios")
_ORIGINAL_AUDIO_DIR = _BASE_DIR / "original"
_TTS_AUDIO_DIR = _BASE_DIR / "tts"

_SUBTITLE_EXTS = {".srt", ".vtt"}


def _find_existing_subtitle(video_path: str) -> str | None:
    p = Path(video_path)
    parent = p.parent
    stem = p.stem
    for ext in _SUBTITLE_EXTS:
        exact = parent / f"{stem}{ext}"
        if exact.exists():
            return str(exact)
        for f in sorted(parent.iterdir()):
            if f.suffix.lower() == ext and f.stem.startswith(stem + "_"):
                return str(f)
    return None


def _tts_segments(j: FileJob) -> list[Segment]:
    return j.translated if j.translated is not None else j.segments


def _flush_gpu() -> None:
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception as e:
        log.debug("GPU flush error: %s", e)
    gc.collect()
    try:
        from vidub.translate.llm import stop_server
        stop_server()
    except Exception as e:
        log.debug("Failed to stop llama-server: %s", e)


class DubbingPipeline:
    def __init__(
        self,
        asr_config: ASRConfig,
        translate_config: TranslateConfig | None = None,
        tts_config: TTSConfig | None = None,
    ):
        self.asr_config = asr_config
        self.translate_config = translate_config
        self.tts_config = tts_config
        self._downloaded: list[str] = []

    def run(self, input_path: str, output_dir: str) -> dict:
        t_start = time.time()
        os.makedirs(output_dir, exist_ok=True)

        if is_url(input_path):
            dl_dir = tempfile.mkdtemp(prefix="vdub_dl_")
            input_path = download_url(input_path, dl_dir)
            self._downloaded.append(input_path)

        stem = Path(input_path).stem
        ext = Path(input_path).suffix.lower()
        is_video = ext in MEDIA_EXTENSIONS

        t_audio = time.time()
        original_audio_dir = str(_ORIGINAL_AUDIO_DIR)
        os.makedirs(original_audio_dir, exist_ok=True)
        audio_path = os.path.join(original_audio_dir, f"{stem}.wav")
        if os.path.exists(audio_path):
            log.info("Resuming: using existing audio: %s", audio_path)
        else:
            audio_path = extract_audio(input_path, output_dir=original_audio_dir)
        audio_elapsed = time.time() - t_audio
        duration = get_audio_duration(audio_path)
        log.info("[SUMMARY] Audio extracted in %.2f seconds", audio_elapsed)

        srt_path = os.path.join(output_dir, f"{stem}.original.srt")
        if os.path.exists(srt_path):
            segments = read_srt(srt_path)
            log.info("Resuming: loaded original SRT (%d segments)", len(segments))
        else:
            t_asr = time.time()
            asr_cls = ASR_ENGINES[self.asr_config.engine]
            asr_engine = asr_cls(self.asr_config)
            segments = asr_engine.transcribe(audio_path, language=self.asr_config.language, duration=duration)
            asr_elapsed = time.time() - t_asr
            log.info("[SUMMARY] Transcribed %s took %.2f seconds (%d segments)", input_path, asr_elapsed, len(segments))
            write_srt(srt_path, segments)
            log.debug("Wrote original SRT: %s", srt_path)

        result: dict[str, str | list[str]] = {"srt_path": srt_path}
        tr_elapsed = 0.0

        translated: list[Segment] | None = None
        if self.translate_config:
            translated_srt = os.path.join(output_dir, f"{stem}.srt")
            if os.path.exists(translated_srt):
                translated = read_srt(translated_srt)
                log.info("Resuming: loaded translated SRT (%d segments)", len(translated))
            else:
                t_tr = time.time()
                translate_cls = TRANSLATE_ENGINES[self.translate_config.engine]
                translate_engine = translate_cls(self.translate_config)
                translated = translate_engine.translate(
                    segments, self.translate_config.target_lang, self.translate_config.source_lang
                )
                tr_elapsed = time.time() - t_tr
                write_srt(translated_srt, translated)
                log.info("[SUMMARY] Translated %s took %.2f seconds", input_path, tr_elapsed)
                log.debug("Wrote translated SRT: %s", translated_srt)
            result["translated_srt_path"] = translated_srt

        tts_segments = translated if translated is not None else segments

        if self.tts_config:
            output_video = os.path.join(output_dir, f"{stem}{ext}")
            if is_video and os.path.exists(output_video):
                log.info("Resuming: output video already exists: %s", output_video)
                result["output_video"] = output_video
            else:
                t_tts = time.time()
                tts_audio_dir = str(_TTS_AUDIO_DIR)
                os.makedirs(tts_audio_dir, exist_ok=True)
                tts_cls = TTS_ENGINES[self.tts_config.engine]
                tts_engine = tts_cls(self.tts_config)
                num_segments = len(tts_segments)
                log.info("[%s] - Generating audio file for video %s (%d segments)", _GPU_TAG, stem, num_segments)
                audio_files = tts_engine.synthesize_segments(tts_segments, tts_audio_dir, stem=stem)
                tts_elapsed = time.time() - t_tts
                log.info("[%s] - Generated %d audio segments for video %s took %.2f seconds",
                         _GPU_TAG, len([f for f in audio_files if f]), stem, tts_elapsed)
                result["audio_files"] = audio_files

                log.info("[%s] - Composing a full audio file by ffmpeg", _CPU_TAG)
                composed = compose_audio(tts_segments, audio_files, tts_audio_dir, original_audio_path=audio_path, stem=stem)
                result["composed_audio"] = composed

                if is_video:
                    log.info("[%s] - %s", _CPU_TAG, f"Replacing audio in {input_path} -> {output_video}")
                    replace_video_audio(input_path, composed, output_video)
                    result["output_video"] = output_video
                else:
                    result["output_audio"] = composed

        if str(audio_path).startswith(tempfile.gettempdir()):
            cleanup_audio(audio_path)
        self._cleanup_downloads()

        elapsed = time.time() - t_start
        parts = [f"Audio: {audio_elapsed:.2f}s", f"ASR: 0.00s"]
        if tr_elapsed:
            parts.append(f"Translate: {tr_elapsed:.2f}s")
        parts.append(f"Total: {elapsed:.2f}s")
        log.info("[SUMMARY] %s | %s", input_path, " | ".join(parts))
        return result

    def _cleanup_downloads(self) -> None:
        for path in self._downloaded:
            try:
                os.remove(path)
                parent = os.path.dirname(path)
                os.rmdir(parent)
                log.debug("Cleaned up temp download: %s", path)
            except OSError:
                pass
        self._downloaded.clear()


class BatchPipeline:
    def __init__(self, pipeline: DubbingPipeline):
        self.pipeline = pipeline
        self._config_hash = compute_config_hash(
            pipeline.asr_config, pipeline.translate_config, pipeline.tts_config
        )

    def _build_jobs(self, file_list: list[str], input_root: Path) -> list[FileJob]:
        jobs: list[FileJob] = []
        for f in file_list:
            p = Path(f).resolve()
            rel = p.relative_to(input_root)
            ext = p.suffix.lower()
            jobs.append(FileJob(
                input_path=f,
                rel_path=rel,
                stem=p.stem,
                ext=ext,
                is_video=ext in MEDIA_EXTENSIONS,
                subtitle_path=_find_existing_subtitle(f),
            ))
        return jobs

    def _is_stage_done(self, state: dict[str, list[str]] | None, rel_path: str, stage: str) -> bool:
        return state is not None and stage in state.get(str(rel_path), [])

    def _stage1_audio_extraction(
        self, jobs: list[FileJob], original_audio_root: Path, state: dict[str, list[str]] | None
    ) -> None:
        t1 = time.time()
        for j in tqdm(jobs, desc="Extracting audio", unit="video", position=0, leave=False):
            if j.subtitle_path:
                j.audio_path = j.input_path
                log.debug("  Skipping audio extraction, using video directly: %s", j.input_path)
                continue
            audio_path = original_audio_root / j.rel_path.parent / f"{j.stem}.wav"
            if self._is_stage_done(state, j.rel_path, "audio_extracted") and audio_path.exists():
                j.audio_path = str(audio_path)
                log.debug("Resuming: using existing audio: %s", j.audio_path)
                continue
            audio_dir = original_audio_root / j.rel_path.parent
            os.makedirs(audio_dir, exist_ok=True)
            j.audio_path = extract_audio(j.input_path, output_dir=str(audio_dir))
            mark_stage_completed(self._output_root, str(j.rel_path), "audio_extracted")
        log.info("[STAGE 1] Audio extraction: %.2fs (%d extracted, %d skipped)",
                 time.time() - t1,
                 sum(1 for j in jobs if not j.subtitle_path),
                 sum(1 for j in jobs if j.subtitle_path))

    def _stage2_asr(
        self, jobs: list[FileJob], state: dict[str, list[str]] | None
    ) -> None:
        asr_jobs = [j for j in jobs if j.subtitle_path is None]
        t2 = time.time()
        if asr_jobs:
            asr_cls = ASR_ENGINES[self.pipeline.asr_config.engine]
            asr_engine = asr_cls(self.pipeline.asr_config)
            for j in tqdm(asr_jobs, desc="Transcribing", unit="video", position=0, leave=False):
                if self._is_stage_done(state, j.rel_path, "asr_done"):
                    log.debug("Resuming: ASR already done for %s", j.rel_path)
                    j.segments = []
                    continue
                duration = get_audio_duration(j.audio_path)
                j.segments = asr_engine.transcribe(
                    j.audio_path,
                    language=self.pipeline.asr_config.language,
                    duration=duration,
                )
                mark_stage_completed(self._output_root, str(j.rel_path), "asr_done")
        for j in tqdm(jobs, desc="Loading SRTs", unit="file", position=0, leave=False):
            if j.subtitle_path:
                j.segments = read_srt(j.subtitle_path)
                log.info("  Using existing subtitle: %s", j.subtitle_path)
        log.info("[STAGE 2] ASR: %.2fs (%d files transcribed, %d from existing SRT)",
                 time.time() - t2, len(asr_jobs), len(jobs) - len(asr_jobs))

    def _write_original_srts(
        self, jobs: list[FileJob], output_root: Path, state: dict[str, list[str]] | None
    ) -> None:
        for j in jobs:
            srt_path = output_root / j.rel_path.parent / f"{j.stem}.original.srt"
            if self._is_stage_done(state, j.rel_path, "srt_written") and srt_path.exists():
                log.debug("Resuming: original SRT already written: %s", srt_path)
                j.segments = read_srt(str(srt_path))
                continue
            if j.segments is None:
                j.segments = []
            out_subdir = output_root / j.rel_path.parent
            os.makedirs(out_subdir, exist_ok=True)
            write_srt(str(srt_path), j.segments)
            mark_stage_completed(self._output_root, str(j.rel_path), "srt_written")

    def _stage3_translation(
        self, jobs: list[FileJob], output_root: Path, state: dict[str, list[str]] | None
    ) -> None:
        t3 = time.time()
        if not self.pipeline.translate_config:
            return

        from vidub.translate.llm import _NUM_TRANSLATE_WORKERS

        translate_cls = TRANSLATE_ENGINES[self.pipeline.translate_config.engine]
        translate_engine = translate_cls(self.pipeline.translate_config)

        def _do_translate(j: FileJob) -> None:
            srt_path = output_root / j.rel_path.parent / f"{j.stem}.srt"
            if self._is_stage_done(state, j.rel_path, "translated") and srt_path.exists():
                log.info("Resuming: skipping translation for %s", j.rel_path)
                j.translated = read_srt(str(srt_path))
                return
            log.info("Translating: %s (%d segments)", j.rel_path, len(j.segments))
            j.translated = translate_engine.translate(
                j.segments,
                self.pipeline.translate_config.target_lang,
                self.pipeline.translate_config.source_lang,
            )
            out_subdir = output_root / j.rel_path.parent
            os.makedirs(out_subdir, exist_ok=True)
            write_srt(str(srt_path), j.translated)
            mark_stage_completed(self._output_root, str(j.rel_path), "translated")

        pending = [j for j in jobs if not (
            self._is_stage_done(state, j.rel_path, "translated")
            and (output_root / j.rel_path.parent / f"{j.stem}.srt").exists()
        )]
        done = [j for j in jobs if j not in pending]

        for j in tqdm(done, desc="Loading SRTs", unit="file", position=0, leave=False):
            _do_translate(j)

        if pending:
            with concurrent.futures.ThreadPoolExecutor(max_workers=_NUM_TRANSLATE_WORKERS) as executor:
                list(tqdm(
                    executor.map(_do_translate, pending),
                    desc="Translating",
                    unit="video",
                    total=len(pending),
                    position=0,
                    leave=False,
                ))

        log.info("[STAGE 3] Translation: %.2fs for %d files", time.time() - t3, len(jobs))

    def _stage4_tts_and_compose(
        self,
        jobs: list[FileJob],
        tts_audio_root: Path,
        output_root: Path,
        state: dict[str, list[str]] | None,
    ) -> None:
        t4 = time.time()
        tts_jobs = [j for j in jobs if self.pipeline.tts_config]
        if not tts_jobs:
            return

        tts_cls = TTS_ENGINES[self.pipeline.tts_config.engine]
        tts_engine = tts_cls(self.pipeline.tts_config)

        max_workers = os.cpu_count() or 1

        def _compose_worker(j: FileJob) -> None:
            try:
                tts_dir = str(tts_audio_root / j.rel_path.parent)
                log.info("[%s] - Composing a full audio file by ffmpeg", _CPU_TAG)
                j.composed_path = compose_audio(
                    _tts_segments(j),
                    j.audio_files,
                    tts_dir,
                    original_audio_path=j.audio_path,
                    stem=j.stem,
                )
                if j.is_video:
                    out_subdir = output_root / j.rel_path.parent
                    os.makedirs(out_subdir, exist_ok=True)
                    out_video = str(out_subdir / f"{j.stem}{j.ext}")
                    log.info("[%s] - %s", _CPU_TAG, f"Replacing audio in {j.input_path} -> {out_video}")
                    replace_video_audio(j.input_path, j.composed_path, out_video)
                mark_stage_completed(self._output_root, str(j.rel_path), "composed")
            except Exception as exc:
                log.error("[%s] - Compose failed for %s: %s", _CPU_TAG, j.input_path, exc)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for j in tqdm(tts_jobs, desc="Generating TTS", unit="video", position=0, leave=False):
                out_video = output_root / j.rel_path.parent / f"{j.stem}{j.ext}"
                if self._is_stage_done(state, j.rel_path, "composed") and (not j.is_video or out_video.exists()):
                    log.info("Resuming: TTS+compose already done for %s", j.rel_path)
                    j.audio_files = []
                    j.composed_path = None
                    continue
                tts_dir = str(tts_audio_root / j.rel_path.parent)
                os.makedirs(tts_dir, exist_ok=True)
                num_segments = len(_tts_segments(j))
                log.info("[%s] - Generating audio file for video %s (%d segments)",
                         _GPU_TAG, j.stem, num_segments)
                tts_start = time.time()
                j.audio_files = tts_engine.synthesize_segments(_tts_segments(j), tts_dir, stem=j.stem)
                tts_elapsed = time.time() - tts_start
                log.info("[%s] - Generated %d audio segments for video %s took %.2f seconds",
                         _GPU_TAG, len([f for f in j.audio_files if f]), j.stem, tts_elapsed)
                futures.append(executor.submit(_compose_worker, j))

            for f in tqdm(
                concurrent.futures.as_completed(futures),
                desc="Composing",
                unit="video",
                total=len(futures),
                position=0,
                leave=False,
            ):
                f.result()

        log.info("[STAGE 4] TTS: %.2fs for %d files", time.time() - t4, len(tts_jobs))

    def _build_results(self, jobs: list[FileJob], output_root: Path) -> list[dict[str, str | list[str]]]:
        results: list[dict[str, str | list[str]]] = []
        for j in jobs:
            r: dict[str, str | list[str]] = {"input_path": j.input_path}
            if j.subtitle_path:
                r["srt_path"] = j.subtitle_path
            out_subdir = output_root / j.rel_path.parent
            orig_srt = out_subdir / f"{j.stem}.original.srt"
            if orig_srt.exists():
                r["srt_path"] = str(orig_srt)
            trans_srt = out_subdir / f"{j.stem}.srt"
            if trans_srt.exists():
                r["translated_srt_path"] = str(trans_srt)
            if j.composed_path:
                r["composed_audio"] = j.composed_path
            if j.is_video:
                out_video = out_subdir / f"{j.stem}{j.ext}"
                if out_video.exists():
                    r["output_video"] = str(out_video)
            results.append(r)
        return results

    def run(self, input_path: str, output_dir: str) -> list[dict[str, str | list[str]]]:
        t_start = time.time()
        _all = scan_folder(input_path)
        input_root = Path(input_path).resolve()
        output_root = Path(output_dir).resolve()
        input_name = input_root.name

        self._output_root = output_root

        audio_root = _BASE_DIR / input_name
        original_audio_root = audio_root / "original"
        tts_audio_root = audio_root / "tts"

        jobs = self._build_jobs(_all, input_root)
        log.info("Found %d media files in %s", len(jobs), input_path)

        state = load_state(output_root, self._config_hash)
        _init_state(output_root, jobs, self._config_hash)

        self._stage1_audio_extraction(jobs, original_audio_root, state)
        self._stage2_asr(jobs, state)
        self._write_original_srts(jobs, output_root, state)
        self._stage3_translation(jobs, output_root, state)
        _flush_gpu()
        self._stage4_tts_and_compose(jobs, tts_audio_root, output_root, state)
        results = self._build_results(jobs, output_root)

        log.info("[SUMMARY] Total batch: %.2fs for %d files", time.time() - t_start, len(jobs))
        return results
