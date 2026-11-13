"""Bounded, resident Whisper transcription service for generated CKE bundles.

Run this as a separate process from text serving and Kokoro. The encoder and
decoder workers each retain one native model across independent recordings.
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import Future
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import io
import json
from pathlib import Path
from collections import deque
import re
import tempfile
import threading
import time
import uuid
import wave
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from server.speech_usage import SpeechUsage, SpeechUsageMiddleware


class TranscriptionCancelled(Exception):
    pass


@dataclass
class _Job:
    request_id: str
    wav: bytes
    queued_at: float
    result: Future[dict[str, Any]] = field(default_factory=Future)
    cancelled: threading.Event = field(default_factory=threading.Event)
    vocabulary_hint: str = ""


def _wav_frames(data: bytes, *, max_bytes: int, max_seconds: float) -> int:
    if len(data) > max_bytes:
        raise ValueError("audio upload exceeds byte limit")
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            if (wav.getcomptype() != "NONE" or wav.getnchannels() != 1
                    or wav.getsampwidth() != 2 or wav.getframerate() != 16000):
                raise ValueError("audio must be mono 16 kHz PCM16 WAV")
            frames = wav.getnframes()
            if frames < 1 or frames > int(16000 * max_seconds):
                raise ValueError("recording duration exceeds limit")
            if len(wav.readframes(frames)) != frames * 2:
                raise ValueError("truncated WAV payload")
            return frames
    except (EOFError, wave.Error) as error:
        raise ValueError("malformed WAV payload") from error


class _WhisperBackend:
    """One process-owned encoder/decoder pair; only the queue thread calls run."""

    def __init__(self, encoder_dir: Path, decoder_dir: Path,
                 timeout_seconds: float, temp_root: Path | None) -> None:
        from version.v8.scripts import run_whisper_v8 as whisper

        self._whisper = whisper
        self.encoder_dir = encoder_dir.resolve()
        self.decoder_dir = decoder_dir.resolve()
        self.temp_root = temp_root
        config = json.loads((self.encoder_dir / "config.json").read_text())
        self.max_frames = int(config["audio_sample_extent"])
        if int(config["audio_sample_rate"]) != 16000 or self.max_frames <= 0:
            raise ValueError("resident Whisper requires a 16 kHz encoder bundle")
        environment = whisper._worker_environment(config)
        self.encoder = whisper._PersistentWorker(
            "encoder", self.encoder_dir, environment, timeout_seconds)
        try:
            self.decoder = whisper._PersistentWorker(
                "decoder", self.decoder_dir, environment, timeout_seconds)
            self.identity = self._loaded_identity()
        except BaseException:
            for worker in (getattr(self, "decoder", None), self.encoder):
                if worker is not None:
                    try:
                        worker.close()
                    except Exception:
                        pass
            raise

    def _loaded_identity(self) -> dict[str, Any]:
        identity: dict[str, Any] = {"schema": "cke.whisper.loaded.v1"}
        for role, worker, directory in (
            ("encoder", self.encoder, self.encoder_dir),
            ("decoder", self.decoder, self.decoder_dir),
        ):
            maps = Path(f"/proc/{worker._process.pid}/maps").read_text()
            mapped = {Path(line.split()[-1]).resolve() for line in maps.splitlines()
                      if line.split()[-1].startswith("/")}
            files = ("libmodel.so", "libckernel_engine.so", "weights_manifest.map")
            for name in files[:2]:
                if (directory / name).resolve() not in mapped:
                    raise RuntimeError(f"{role} loaded library backing differs: {name}")
            identity[role] = {
                "bundle": str(directory),
                "libraries": {name: self._whisper._sha256(directory / name)
                              for name in files[:2]},
                "manifest_sha256": self._whisper._sha256(directory / files[2]),
                "weights_sha256": self._whisper._sha256(directory / "weights.bump"),
                "model_init_seconds": worker.model_init_seconds,
                "worker_pid": worker._process.pid,
            }
        return identity

    def run(self, wav: bytes, *, vocabulary_hint: str = "") -> dict[str, Any]:
        if _wav_frames(wav, max_bytes=len(wav),
                       max_seconds=self.max_frames / 16000) > self.max_frames:
            raise ValueError("recording exceeds encoder capacity")
        whisper = self._whisper
        with tempfile.TemporaryDirectory(prefix="cke-whisper-http-",
                                         dir=self.temp_root) as name:
            temp = Path(name)
            path = temp / "recording.wav"
            path.write_bytes(wav)
            args = argparse.Namespace(language="en", task="transcribe",
                                      max_tokens=128, timestamps=False,
                                      attention_trace_dir=None, vocabulary_hint=vocabulary_hint)
            encoder, decoder = whisper._run_segment(
                args, common=[], encoder_dir=self.encoder_dir,
                decoder_dir=self.decoder_dir, wav_path=path, temp=temp,
                index=0, window_start_frame=0, worker_env={},
                encoder_worker=self.encoder, decoder_worker=self.decoder)
            return {
                "text": str(decoder["transcript_text"]),
                "stop": str(decoder["stop"]),
                "vocabulary_hint_applied": decoder.get("vocabulary_hint_applied", False),
                "vocabulary_hint_token_count": decoder.get("vocabulary_hint_token_count", 0),
                "inference_seconds": (float(encoder["worker_wall_seconds"])
                                      + float(decoder["worker_wall_seconds"])),
            }

    def abort(self) -> None:
        for worker in (self.encoder, self.decoder):
            if worker._process.is_alive():
                worker._process.kill()

    def close(self) -> None:
        errors = []
        for worker in (self.decoder, self.encoder):
            try:
                worker.close()
            except (OSError, RuntimeError, TimeoutError) as error:
                errors.append(error)
        if errors:
            raise RuntimeError("resident worker cleanup failed") from errors[0]


class _Owner:
    def __init__(self, factory: Callable[[], Any], *, max_waiters: int,
                 preload: bool, shutdown_timeout: float = 10.0) -> None:
        if max_waiters < 1:
            raise ValueError("max_waiters must be positive")
        self._factory = factory
        self._queue: deque[_Job] = deque()
        self._max_waiters = max_waiters
        self._shutdown_timeout = shutdown_timeout
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._jobs: dict[str, _Job] = {}
        self._active: _Job | None = None
        self._running_native = False
        self._backend: Any | None = None
        self._generation = 0
        self._preload = preload
        self._startup_error: str | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="cke-whisper-owner",
                                        daemon=True)

    def start(self) -> None:
        self._thread.start()

    @property
    def status(self) -> dict[str, Any]:
        with self._lock:
            return {"ready": self._backend is not None and not self._closed,
                    "active": self._active.request_id if self._active else None,
                    "queued": len(self._queue), "worker_generation": self._generation,
                    "running_native": self._running_native,
                    "startup_error": self._startup_error}

    def submit(self, job: _Job) -> bool:
        with self._lock:
            if self._closed or job.request_id in self._jobs:
                return False
            if len(self._queue) >= self._max_waiters:
                return False
            self._queue.append(job)
            self._jobs[job.request_id] = job
            self._condition.notify()
            return True

    def cancel(self, request_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(request_id)
            if job is None:
                return False
            if job.result.done():
                return False
            job.cancelled.set()
            if self._active is not job:
                self._queue.remove(job)
                self._jobs.pop(request_id)
                job.result.set_exception(TranscriptionCancelled())
                return True
            active = self._active is job and self._running_native
            backend = self._backend if active else None
        if backend is not None:
            try:
                backend.abort()
            except (OSError, RuntimeError):
                pass
        return True

    def _discard_backend(self) -> None:
        backend, self._backend = self._backend, None
        if backend is not None:
            try:
                backend.close()
            except (OSError, RuntimeError, TimeoutError):
                pass

    def _run(self) -> None:
        if self._preload:
            try:
                self._backend = self._factory()
                self._generation += 1
            except Exception as error:
                self._startup_error = str(error)
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._queue)
                if self._closed:
                    break
                job = self._queue.popleft()
                self._active = job
            entered_native = False
            try:
                if job.cancelled.is_set():
                    raise TranscriptionCancelled()
                if self._backend is None:
                    self._backend = self._factory()
                    self._generation += 1
                    self._startup_error = None
                with self._lock:
                    if job.cancelled.is_set():
                        raise TranscriptionCancelled()
                    self._running_native = True
                    entered_native = True
                started = time.monotonic()
                response = (self._backend.run(job.wav, vocabulary_hint=job.vocabulary_hint)
                            if job.vocabulary_hint else self._backend.run(job.wav))
                with self._lock:
                    self._running_native = False
                    if job.cancelled.is_set():
                        raise TranscriptionCancelled()
                    job.result.set_result({
                        "id": job.request_id, **response,
                        "loaded_artifact": self._backend.identity,
                        "queue_seconds": started - job.queued_at,
                        "worker_generation": self._generation,
                    })
            except BaseException as error:
                with self._lock:
                    self._running_native = False
                    if not job.result.done():
                        job.result.set_exception(
                            TranscriptionCancelled() if job.cancelled.is_set() else error)
                if entered_native or not isinstance(error, TranscriptionCancelled):
                    self._discard_backend()
            finally:
                with self._lock:
                    self._jobs.pop(job.request_id, None)
                    self._active = None
        self._discard_backend()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            ids = list(self._jobs)
            self._condition.notify_all()
        for request_id in ids:
            self.cancel(request_id)
        self._thread.join(timeout=self._shutdown_timeout)
        if self._thread.is_alive():
            raise RuntimeError("Whisper shutdown deadline exceeded; initialization or cleanup is still pending")


def create_app(factory: Callable[[], Any], *, max_upload_bytes: int = 1024 * 1024,
               max_seconds: float = 30.0, max_waiters: int = 2,
               preload: bool = False, max_uploads: int = 2,
               vocabulary_validator: Callable[[str], Any] | None = None,
               worker_timeout_seconds: float | None = None) -> FastAPI:
    if max_upload_bytes < 44 or max_seconds <= 0 or max_uploads < 1:
        raise ValueError("invalid audio resource limits")
    owner = _Owner(factory, max_waiters=max_waiters, preload=preload)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        owner.start()
        try:
            yield
        finally:
            await asyncio.to_thread(owner.close)

    app = FastAPI(lifespan=lifespan)
    usage = SpeechUsage()
    service_instance_id = usage.boot_id
    # This describes this HTTP interface, not every capability in the CLI or
    # generated model. Unsupported features must never be inferred from a name.
    from version.v8.scripts.run_whisper_v8 import (
        MAX_VOCABULARY_HINT_BYTES, MAX_VOCABULARY_HINT_TOKENS,
    )
    capabilities = {
        "schema": "cke.whisper.capabilities.v1",
        "audio": {"format": "wav_pcm16", "sample_rate_hz": 16000, "channels": 1,
                  "max_upload_bytes": max_upload_bytes, "max_seconds": max_seconds},
        "decoding": {"language": "en", "task": "transcribe", "strategy": "greedy",
                     "max_output_tokens": 128},
        "vocabulary_hint": {"supported": vocabulary_validator is not None,
                            "max_utf8_bytes": MAX_VOCABULARY_HINT_BYTES,
                            "max_tokens": MAX_VOCABULARY_HINT_TOKENS,
                            "capacity_check": "hint plus generation must fit decoder context"},
        "segment_timestamps": False, "attention_traces": False,
        "word_alignment": False, "editing_review": False,
        "long_audio": False, "artifact_retrieval": False,
        "max_waiters": max_waiters, "max_concurrent_uploads": max_uploads,
        "worker_timeout_seconds": worker_timeout_seconds,
        "request_deadlines": False,
        "retry_semantics": "active IDs rejected; completed requests are not deduplicated",
        "cancellation": "queued requests release admission; active requests signal worker abort and recover before reuse",
    }
    app.add_middleware(SpeechUsageMiddleware, usage=usage)

    @app.get("/telemetry")
    async def telemetry():
        return usage.snapshot()

    uploads = threading.BoundedSemaphore(max_uploads)

    @app.get("/health")
    async def health():
        return {"status": "ok", "service_instance_id": service_instance_id,
                "capabilities": capabilities, **owner.status}

    @app.post("/audio/transcriptions")
    async def transcribe(request: Request):
        if any(key != "vocabulary_hint" for key in request.query_params):
            raise HTTPException(400, {
                "code": "unsupported_option",
                "message": "This service accepts only vocabulary_hint; consult /health capabilities. Timestamps, alignment and long audio require a separately supported interface.",
            })
        hint = request.query_params.get("vocabulary_hint", "")
        if len(request.query_params.getlist("vocabulary_hint")) > 1:
            raise HTTPException(400, "duplicate vocabulary hint")
        if hint:
            if vocabulary_validator is None:
                raise HTTPException(501, "vocabulary prompting is unavailable")
            try:
                vocabulary_validator(hint)
            except ValueError as error:
                raise HTTPException(400, "invalid or over-capacity vocabulary hint") from error
        if request.headers.get("content-type", "").split(";", 1)[0] != "audio/wav":
            raise HTTPException(415, "expected audio/wav request body")
        if not uploads.acquire(blocking=False):
            raise HTTPException(429, "audio upload concurrency limit reached")
        try:
            chunks = []
            total = 0
            async for chunk in request.stream():
                total += len(chunk)
                if total > max_upload_bytes:
                    raise HTTPException(413, "audio upload exceeds byte limit")
                chunks.append(chunk)
            wav = b"".join(chunks)
        finally:
            uploads.release()
        try:
            audio_frames = _wav_frames(wav, max_bytes=max_upload_bytes, max_seconds=max_seconds)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        request_id = request.headers.get("x-cke-request-id") or uuid.uuid4().hex
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", request_id):
            raise HTTPException(400, "invalid request ID")
        job = _Job(request_id, wav, time.monotonic(), vocabulary_hint=hint)
        if not owner.submit(job):
            raise HTTPException(429, "transcription queue is full or request ID is active")
        usage.admit(audio_frames, bool(hint.strip()))
        job.result.add_done_callback(lambda result: usage.settled(result, TranscriptionCancelled, audio_frames))
        waiting = asyncio.wrap_future(job.result)

        def consume_after_disconnect(future: asyncio.Future) -> None:
            if not future.cancelled():
                future.exception()

        try:
            while True:
                done, _ = await asyncio.wait({waiting}, timeout=0.1)
                if done:
                    result = await waiting
                    return JSONResponse({**result, "service_instance_id": service_instance_id},
                                        headers={"X-CKE-Request-ID": request_id})
                if await request.is_disconnected():
                    waiting.add_done_callback(consume_after_disconnect)
                    owner.cancel(request_id)
                    raise HTTPException(499, "client disconnected")
        except asyncio.CancelledError:
            waiting.add_done_callback(consume_after_disconnect)
            owner.cancel(request_id)
            raise
        except TranscriptionCancelled as error:
            raise HTTPException(409, "transcription cancelled",
                                headers={"X-CKE-Request-ID": request_id}) from error
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(503, "transcription worker failed",
                                headers={"X-CKE-Request-ID": request_id}) from error

    @app.post("/audio/transcriptions/{request_id}/cancel")
    async def cancel(request_id: str):
        if not owner.cancel(request_id):
            raise HTTPException(404, "request is not active or queued")
        return {"id": request_id, "status": "cancelling"}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-run-dir", type=Path, required=True)
    parser.add_argument("--decoder-run-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8110)
    parser.add_argument("--max-upload-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--max-waiters", type=int, default=2)
    parser.add_argument("--worker-timeout-seconds", type=float, default=600.0)
    args = parser.parse_args()
    if args.worker_timeout_seconds <= 0:
        parser.error("worker timeout must be positive")
    config = json.loads((args.encoder_run_dir / "config.json").read_text())
    if (int(config["audio_sample_rate"]) != 16000
            or args.max_seconds > int(config["audio_sample_extent"]) / 16000):
        parser.error("recording limit exceeds the 16 kHz encoder capacity")
    from tokenizers import Tokenizer
    from version.v8.scripts.run_whisper_v8 import vocabulary_decoder_prefix
    tokenizer = Tokenizer.from_file(str(args.decoder_run_dir / "tokenizer.json"))
    generation = json.loads((args.decoder_run_dir / "generation_config.json").read_text())
    decoder_config = json.loads((args.decoder_run_dir / "config.json").read_text())
    def vocabulary_validator(hint):
        return vocabulary_decoder_prefix(generation, decoder_config, tokenizer,
                                         "en", "transcribe", hint, 128)
    factory = lambda: _WhisperBackend(args.encoder_run_dir, args.decoder_run_dir,
                                      args.worker_timeout_seconds, None)
    import uvicorn
    uvicorn.run(create_app(factory, max_upload_bytes=args.max_upload_bytes,
                           max_seconds=args.max_seconds,
                           max_waiters=args.max_waiters, preload=True,
                           vocabulary_validator=vocabulary_validator,
                           worker_timeout_seconds=args.worker_timeout_seconds),
                # Vocabulary travels in a query parameter for compatibility.
                # Uvicorn access logs would otherwise retain that private text.
                host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
