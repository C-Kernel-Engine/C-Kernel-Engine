"""Asset-free resident speech request, ordering, and recovery checks."""

from concurrent.futures import ThreadPoolExecutor
import base64
import hashlib
import io
import json
import struct
import threading
import time
import wave

from fastapi.testclient import TestClient

from server.kokoro_http import create_app


def _wav(value: int) -> bytes:
    result = io.BytesIO()
    with wave.open(result, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(struct.pack("<h", value) * 16)
    return result.getvalue()


class Backend:
    def __init__(self, generation: int, block: bool = False):
        self.identity = {"generation": generation}
        self.calls = []
        self.closed = False
        self.block = block
        self.started = threading.Event()
        self.released = threading.Event()

    def run(self, text, seed, mode, emit, cancelled):
        self.calls.append(text)
        self.started.set()
        if self.block:
            self.released.wait(5)
        if self.closed or cancelled.is_set():
            raise RuntimeError("injected abort")
        if text == "fail":
            raise RuntimeError("injected native failure")
        for index, span in enumerate(text.split(" ")):
            wav = _wav(index + 1)
            emit({"type": "chunk", "index": index,
                  "source_start": sum(len(part) + 1 for part in text.split(" ")[:index]),
                  "source_end": sum(len(part) + 1 for part in text.split(" ")[:index]) + len(span),
                  "source_text": span, "samples": 16,
                  "wav_sha256": hashlib.sha256(wav).hexdigest(),
                  "wav_base64": base64.b64encode(wav).decode()})
        emit({"type": "complete", "total_chunks": len(text.split(" "))})

    def abort(self):
        self.closed = True
        self.released.set()

    def close(self):
        self.closed = True
        self.released.set()


def _events(response):
    return [json.loads(line) for line in response.text.splitlines()]


def test_a_b_a_reuses_worker_and_preserves_ordered_audio() -> None:
    backends = []

    def factory():
        backend = Backend(len(backends) + 1)
        backends.append(backend)
        return backend

    with TestClient(create_app(factory)) as client:
        for request_id, text in (("a1", "a b"), ("b", "c"), ("a2", "a b")):
            response = client.post("/audio/speech", json={"text": text},
                                   headers={"x-cke-request-id": request_id})
            assert response.status_code == 200
            events = _events(response)
            assert events[0]["type"] == "started"
            assert events[0]["worker_generation"] == 1
            assert events[-1]["type"] == "complete"
            assert [event["index"] for event in events if event["type"] == "chunk"] == list(range(len(text.split(" "))))
            assert all(event["id"] == request_id for event in events)
            for event in events:
                if event["type"] == "chunk":
                    assert hashlib.sha256(base64.b64decode(event["wav_base64"])).hexdigest() == event["wav_sha256"]
        assert len(backends) == 1
        assert backends[0].calls == ["a b", "c", "a b"]


def test_invalid_input_rejects_before_native_work_and_failure_recovers() -> None:
    backends = []

    def factory():
        backend = Backend(len(backends) + 1)
        backends.append(backend)
        return backend

    with TestClient(create_app(factory)) as client:
        for payload in ({"text": ""}, {"text": "x\x00"},
                        {"text": "x", "seed": -1},
                        {"text": "x", "technical_mode": "wrong"}):
            assert client.post("/audio/speech", json=payload).status_code == 400
        assert client.post("/audio/speech", content=b"x" * 9000).status_code == 413
        assert not backends
        assert _events(client.post("/audio/speech", json={"text": "fail"}))[-1]["type"] == "error"
        recovered = _events(client.post("/audio/speech", json={"text": "good"}))
        assert recovered[0]["worker_generation"] == 2
        assert recovered[-1]["type"] == "complete"


def test_cancel_active_then_recover_without_reusing_model() -> None:
    backends = []

    def factory():
        backend = Backend(len(backends) + 1, block=len(backends) == 0)
        backends.append(backend)
        return backend

    with TestClient(create_app(factory)) as client:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(client.post, "/audio/speech", json={"text": "a b"},
                                  headers={"x-cke-request-id": "cancel-me"})
            for _ in range(100):
                if backends and backends[0].started.is_set():
                    break
                time.sleep(0.01)
            assert backends[0].started.is_set()
            assert client.post("/audio/speech/cancel-me/cancel").status_code == 200
            events = _events(pending.result(timeout=3))
            assert events[-1]["type"] == "cancelled"
        assert backends[0].closed
        recovered = _events(client.post("/audio/speech", json={"text": "next"}))
        assert recovered[0]["worker_generation"] == 2
        assert recovered[-1]["type"] == "complete"


def test_queued_cancel_and_overflow_preserve_active_worker() -> None:
    backends = []

    def factory():
        backend = Backend(len(backends) + 1, block=len(backends) == 0)
        backends.append(backend)
        return backend

    with TestClient(create_app(factory, max_waiters=1)) as client:
        def post(request_id, text):
            return client.post("/audio/speech", json={"text": text},
                               headers={"x-cke-request-id": request_id})

        with ThreadPoolExecutor(max_workers=2) as pool:
            active = pool.submit(post, "active", "first")
            for _ in range(100):
                if backends and backends[0].started.is_set():
                    break
                time.sleep(0.01)
            assert backends[0].started.is_set()
            queued = pool.submit(post, "queued", "second")
            for _ in range(100):
                if client.get("/health").json()["queued"]:
                    break
                time.sleep(0.01)
            assert client.get("/health").json()["queued"] == 1
            assert post("overflow", "third").status_code == 429
            assert client.post("/audio/speech/queued/cancel").status_code == 200
            backends[0].released.set()
            assert _events(active.result(timeout=3))[-1]["type"] == "complete"
            assert _events(queued.result(timeout=3))[-1]["type"] == "cancelled"
        assert _events(post("next", "fourth"))[0]["worker_generation"] == 1
        assert len(backends) == 1 and not backends[0].closed
