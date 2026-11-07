"""Opt-in resident Kokoro speech service; run separately from ASR and text serving.

POST /audio/speech accepts JSON and streams ordered NDJSON records. Each chunk
contains a base64-encoded mono 24 kHz PCM16 WAV. The client owns playback and
must discard old chunks when replacing or cancelling a request.
"""

from __future__ import annotations

from collections import deque
import argparse
import asyncio
import base64
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import select
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from version.v8.tts.prepare_kokoro_accessibility_text import (
    MAX_TEXT_BYTES, TextPlanError, prepare,
)


class SpeechCancelled(Exception):
    pass


class _KokoroBackend:
    def __init__(self, config: Path, bundle: Path,
                 worker_timeout_seconds: float = 600.0) -> None:
        from version.v8.tts.ck_kokoro_play import _checked_wav, _exchange

        self.config, self.bundle = config.resolve(), bundle.resolve()
        if worker_timeout_seconds <= 0:
            raise ValueError("worker timeout must be positive")
        self.worker_timeout_seconds = worker_timeout_seconds
        self._checked_wav, self._exchange = _checked_wav, _exchange
        dispatcher = self.bundle / "native-kokoro-dispatch"
        if not dispatcher.is_file():
            raise RuntimeError("native Kokoro dispatcher unavailable")
        self.worker = subprocess.Popen(
            [str(dispatcher), str(self.bundle), "--session"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1, start_new_session=True)
        try:
            ready, _, _ = select.select([self.worker.stdout], [], [],
                                        min(30.0, worker_timeout_seconds))
            if not ready or self.worker.stdout.readline().strip() != "READY\tloads=1":
                raise RuntimeError("native Kokoro session did not become ready")
            maps = Path(f"/proc/{self.worker.pid}/maps").read_text()
            library = (self.bundle / "64" / "generated.so").resolve()
            mapped = {Path(line.split()[-1]).resolve() for line in maps.splitlines()
                      if line.split()[-1].startswith("/")}
            if library not in mapped:
                raise RuntimeError("Kokoro generated library backing differs")
            self.identity = {
                "bundle": str(self.bundle), "worker_pid": self.worker.pid,
                "dispatcher_sha256": _sha256(dispatcher),
                "manifest_sha256": _sha256(self.bundle / "bundle_manifest.json"),
                "generated_library_sha256": _sha256(library),
                "request_package_sha256": _sha256(self.bundle / "64" / "request.cktts"),
                "voice_sha256": _sha256(self.bundle / "af_heart.ckvoice"),
            }
        except BaseException:
            self.abort()
            self.close()
            raise

    def run(self, text: str, seed: int, technical_mode: str,
            emit: Callable[[dict[str, Any]], None], cancelled: threading.Event) -> None:
        plan = prepare(text, self.config, self.bundle, technical_mode=technical_mode)
        source = plan["source_text"]
        source_hash = hashlib.sha256(source.encode()).hexdigest()
        with tempfile.TemporaryDirectory(prefix="cke-kokoro-http-") as folder:
            total = 0
            for chunk in plan["chunks"]:
                if cancelled.is_set():
                    raise SpeechCancelled()
                index = chunk["index"]
                output = Path(folder) / f"chunk-{index}.wav"
                ids = ",".join(map(str, chunk["phoneme_ids"]))
                started = time.monotonic()
                watchdog = threading.Timer(self.worker_timeout_seconds, self.abort)
                watchdog.start()
                try:
                    fields = self._exchange(
                        self.worker, f"RUN\t{output}\t{seed + index}\t{ids}",
                        "WAV_READY")
                finally:
                    watchdog.cancel()
                if len(fields) != 3 or fields[2] != str(output):
                    raise RuntimeError("native chunk response has wrong output path")
                samples = int(fields[1])
                if output.stat().st_size > 8 * 1024 * 1024:
                    raise RuntimeError("native audio chunk exceeds delivery limit")
                self._checked_wav(output, samples)
                audio = output.read_bytes()
                total += samples
                emit({"type": "chunk", "index": index,
                      "source_start": chunk["source_start"],
                      "source_end": chunk["source_end"],
                      "source_text": chunk["source_text"],
                      "source_sha256": source_hash,
                      "sample_rate_hz": 24000, "samples": samples,
                      "audio_start_sample": total - samples,
                      "audio_end_sample": total,
                      "wav_sha256": hashlib.sha256(audio).hexdigest(),
                      "wav_base64": base64.b64encode(audio).decode("ascii"),
                      "synthesis_seconds": time.monotonic() - started})
            emit({"type": "complete", "source_sha256": source_hash,
                  "total_chunks": len(plan["chunks"]), "samples": total,
                  "bundle_identity_sha256": plan["bundle_identity_sha256"]})

    def abort(self) -> None:
        if self.worker.poll() is None:
            try:
                os.killpg(self.worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def close(self) -> None:
        if self.worker.poll() is None:
            try:
                assert self.worker.stdin is not None
                self.worker.stdin.write("QUIT\n")
                self.worker.stdin.flush()
                self.worker.wait(timeout=5)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self.abort()
                self.worker.wait()
        for stream in (self.worker.stdin, self.worker.stdout):
            if stream is not None:
                stream.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class _Job:
    request_id: str
    text: str
    seed: int
    mode: str
    queued_at: float
    events: queue.Queue[dict[str, Any]] = field(default_factory=lambda: queue.Queue(maxsize=2))
    cancelled: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)
    completed: threading.Event = field(default_factory=threading.Event)


class _Owner:
    def __init__(self, factory: Callable[[], Any], max_waiters: int) -> None:
        self.factory = factory
        self.waiting: deque[_Job] = deque()
        self.max_waiters = max_waiters
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.jobs: dict[str, _Job] = {}
        self.active: _Job | None = None
        self.backend: Any | None = None
        self.running_native = False
        self.generation = 0
        self.closed = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def submit(self, job: _Job) -> bool:
        with self.lock:
            if self.closed or job.request_id in self.jobs:
                return False
            if len(self.waiting) >= self.max_waiters:
                return False
            self.waiting.append(job)
            self.jobs[job.request_id] = job
            self.condition.notify()
            return True

    def cancel(self, request_id: str) -> bool:
        with self.lock:
            job = self.jobs.get(request_id)
            if job is None or job.completed.is_set():
                return False
            job.cancelled.set()
            if self.active is not job:
                self.waiting.remove(job)
                self.jobs.pop(request_id)
                self._terminal(job, {"type": "cancelled"})
                job.finished.set()
                return True
            backend = self.backend if self.active is job and self.running_native else None
        if backend is not None:
            backend.abort()
        return True

    def _emit(self, job: _Job, event: dict[str, Any]) -> None:
        event["id"] = job.request_id
        deadline = time.monotonic() + 10.0
        while not job.cancelled.is_set():
            try:
                job.events.put(event, timeout=0.1)
                if event["type"] == "complete":
                    job.completed.set()
                return
            except queue.Full:
                if time.monotonic() >= deadline:
                    job.cancelled.set()
                    break
        raise SpeechCancelled()

    def _terminal(self, job: _Job, event: dict[str, Any]) -> None:
        event["id"] = job.request_id
        if job.cancelled.is_set():
            while True:
                try:
                    job.events.get_nowait()
                except queue.Empty:
                    break
        try:
            job.events.put_nowait(event)
        except queue.Full:
            # Preserve the terminal verdict even if delivery is backed up.
            try:
                job.events.get_nowait()
            except queue.Empty:
                pass
            job.events.put_nowait(event)

    def _run(self) -> None:
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.waiting)
                if self.closed:
                    break
                job = self.waiting.popleft()
                self.active = job
            entered_native = False
            try:
                if job.cancelled.is_set():
                    raise SpeechCancelled()
                if self.backend is None:
                    self.backend = self.factory()
                    self.generation += 1
                if job.cancelled.is_set():
                    raise SpeechCancelled()
                self._emit(job, {"type": "started",
                                 "queue_seconds": time.monotonic() - job.queued_at,
                                 "worker_generation": self.generation,
                                 "loaded_artifact": self.backend.identity})
                with self.lock:
                    if job.cancelled.is_set():
                        raise SpeechCancelled()
                    self.running_native = True
                    entered_native = True
                self.backend.run(job.text, job.seed, job.mode,
                                 lambda event: self._emit(job, event), job.cancelled)
            except BaseException as error:
                if self.backend is not None and (entered_native or
                                                 not isinstance(error, SpeechCancelled)):
                    try:
                        self.backend.close()
                    except Exception:
                        pass
                    self.backend = None
                self._terminal(job, {"type": "cancelled" if job.cancelled.is_set()
                                     else "error", "detail": type(error).__name__})
            finally:
                with self.lock:
                    self.running_native = False
                    self.jobs.pop(job.request_id, None)
                    self.active = None
                job.finished.set()
        if self.backend is not None:
            self.backend.close()

    def close(self) -> None:
        with self.lock:
            self.closed = True
            ids = list(self.jobs)
            self.condition.notify_all()
        for request_id in ids:
            self.cancel(request_id)
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError("Kokoro owner did not stop")


def create_app(factory: Callable[[], Any], *, max_waiters: int = 2) -> FastAPI:
    if max_waiters < 1:
        raise ValueError("max_waiters must be positive")
    owner = _Owner(factory, max_waiters)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        owner.start()
        try:
            yield
        finally:
            await asyncio.to_thread(owner.close)

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        with owner.lock:
            return {"ready": owner.backend is not None,
                    "active": owner.active.request_id if owner.active else None,
                    "queued": len(owner.waiting),
                    "worker_generation": owner.generation}

    @app.post("/audio/speech")
    async def speech(request: Request):
        try:
            length = int(request.headers.get("content-length", "0"))
        except ValueError as error:
            raise HTTPException(400, "invalid content length") from error
        if length > MAX_TEXT_BYTES + 512:
            raise HTTPException(413, "speech request exceeds limit")
        chunks, total = [], 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > MAX_TEXT_BYTES + 512:
                raise HTTPException(413, "speech request exceeds limit")
            chunks.append(chunk)
        raw = b"".join(chunks)
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError) as error:
            raise HTTPException(400, "invalid JSON request") from error
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            raise HTTPException(400, "text must be a string")
        text = payload["text"]
        if not text.strip() or "\x00" in text or len(text.encode()) > MAX_TEXT_BYTES:
            raise HTTPException(400, "invalid or overlong speech text")
        seed, mode = payload.get("seed", 7), payload.get("technical_mode", "natural")
        if type(seed) is not int or seed < 0 or seed > 2**64 - 513:
            raise HTTPException(400, "invalid seed")
        if mode not in ("natural", "exact"):
            raise HTTPException(400, "invalid technical mode")
        request_id = request.headers.get("x-cke-request-id") or uuid.uuid4().hex
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", request_id):
            raise HTTPException(400, "invalid request ID")
        job = _Job(request_id, text, seed, mode, time.monotonic())
        if not owner.submit(job):
            raise HTTPException(429, "speech queue is full or request ID is active")

        async def records():
            terminal = False
            try:
                while True:
                    if await request.is_disconnected():
                        owner.cancel(request_id)
                        return
                    try:
                        event = await asyncio.to_thread(job.events.get, True, 0.1)
                    except queue.Empty:
                        if job.finished.is_set():
                            return
                        continue
                    if event["type"] == "end":
                        return
                    if event["type"] in ("complete", "cancelled", "error"):
                        terminal = True
                    yield (json.dumps(event, separators=(",", ":")) + "\n").encode()
                    if terminal:
                        return
            finally:
                if not terminal and not job.finished.is_set():
                    owner.cancel(request_id)

        return StreamingResponse(records(), media_type="application/x-ndjson",
                                 headers={"X-CKE-Request-ID": request_id})

    @app.post("/audio/speech/{request_id}/cancel")
    async def cancel(request_id: str):
        if not owner.cancel(request_id):
            raise HTTPException(404, "request is not active or queued")
        return {"id": request_id, "status": "cancelling"}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8111)
    parser.add_argument("--max-waiters", type=int, default=2)
    parser.add_argument("--worker-timeout-seconds", type=float, default=600.0)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(lambda: _KokoroBackend(
        args.config, args.bundle_dir, args.worker_timeout_seconds),
                           max_waiters=args.max_waiters),
                host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
