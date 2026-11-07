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


def test_completion_and_cancel_are_one_atomic_transition() -> None:
    """Pause after enqueue but before put returns: cancel must observe completion."""
    import queue
    from server.kokoro_http import _Owner, _Job

    entered, release = threading.Event(), threading.Event()

    class PausedQueue(queue.Queue):
        def put(self, item, block=True, timeout=None):
            super().put(item, block, timeout)
            entered.set()
            assert release.wait(3)

    owner = _Owner(lambda: None, 1)
    job = _Job('race', 'hello', 7, 'natural', time.monotonic())
    job.events = PausedQueue(2)
    backend = Backend(1)
    owner.backend, owner.active, owner.running_native = backend, job, True
    owner.jobs[job.request_id] = job
    with ThreadPoolExecutor(max_workers=2) as pool:
        publishing = pool.submit(owner._emit, job, {'type': 'complete'})
        try:
            assert entered.wait(2)
            assert owner.lock.locked(), "completion must hold admission lock through publication"
            cancelling = pool.submit(owner.cancel, job.request_id)
        finally:
            release.set()
        publishing.result(timeout=3)
        assert cancelling.result(timeout=3) is False
    assert job.completed.is_set()
    assert not job.cancelled.is_set()
    assert not backend.closed
    assert job.events.get_nowait()['type'] == 'complete'
    assert job.events.empty()


def test_backend_must_finish_with_exactly_one_terminal_event() -> None:
    class MissingComplete(Backend):
        def run(self, text, seed, mode, emit, cancelled):
            return

    class AfterComplete(Backend):
        def run(self, text, seed, mode, emit, cancelled):
            emit({'type': 'complete', 'total_chunks': 0})
            raise RuntimeError('failure after terminal')

    from server.kokoro_http import _Owner, _Job
    for bad, expected in ((MissingComplete, 'error'), (AfterComplete, 'complete')):
        backends = []
        def factory():
            b = bad(1) if not backends else Backend(2)
            backends.append(b)
            return b
        owner = _Owner(factory, 1)
        owner.start()
        try:
            job = _Job('first', 'hello', 7, 'natural', time.monotonic())
            assert owner.submit(job)
            assert job.finished.wait(3)
            events = []
            while not job.events.empty():
                events.append(job.events.get_nowait())
            assert [e['type'] for e in events if e['type'] in ('complete', 'error', 'cancelled')] == [expected]
            followup = _Job('next', 'hello', 7, 'natural', time.monotonic())
            assert owner.submit(followup)
            received = []
            while True:
                event = followup.events.get(timeout=3)
                received.append(event)
                if event['type'] in ('complete', 'error', 'cancelled'):
                    break
            assert received[-1]['type'] == 'complete'
            assert followup.finished.wait(3)
            assert len(backends) == 2
        finally:
            owner.close()


def test_cancel_pins_active_backend_until_abort_signal_is_sent() -> None:
    from server.kokoro_http import _Owner, _Job
    owner = _Owner(lambda: None, 1)
    job = _Job('active', 'hello', 7, 'natural', time.monotonic())

    class SignalBackend(Backend):
        def abort(self):
            assert owner.lock.locked(), 'next request must not activate before this signal'
            assert owner.active is job and job.cancelled.is_set()
            super().abort()

    backend = SignalBackend(1)
    owner.backend, owner.active, owner.running_native = backend, job, True
    owner.jobs[job.request_id] = job
    assert owner.cancel(job.request_id)
    assert backend.closed


def test_repeated_stalled_readers_release_owner_and_followup_recovers() -> None:
    from server.kokoro_http import _Owner, _Job
    backends = []
    def factory():
        b = Backend(len(backends) + 1)
        backends.append(b)
        return b
    owner = _Owner(factory, 1, delivery_timeout_seconds=.05)
    owner.start()
    try:
        for i in range(3):
            job = _Job(f'slow-{i}', 'one two three', 7, 'natural', time.monotonic())
            assert owner.submit(job)
            # Do not read: started and first chunk fill the bounded queue.
            assert job.finished.wait(2)
            assert job.events.qsize() == 1
            assert job.events.get_nowait()['type'] == 'cancelled'
            assert backends[-1].closed
            assert not owner.jobs and owner.active is None
        followup = _Job('next', 'good', 7, 'natural', time.monotonic())
        assert owner.submit(followup)
        events = []
        while True:
            event = followup.events.get(timeout=2)
            events.append(event)
            if event['type'] in ('complete', 'error', 'cancelled'):
                break
        assert events[-1]['type'] == 'complete'
        assert followup.finished.wait(2)
        assert len(backends) == 4
    finally:
        owner.close()
