"""Asset-free admission, routing, and recovery tests for resident Whisper."""

from concurrent.futures import ThreadPoolExecutor
import io
import json
import socket
import struct
import threading
import time
import wave
import urllib.request

from fastapi.testclient import TestClient
import uvicorn

from server.whisper_http import _WhisperBackend, _wav_frames, create_app


def _wav(value: int, *, frames: int = 160) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(struct.pack("<h", value) * frames)
    return output.getvalue()


class _FakeBackend:
    def __init__(self, generation: int, *, block: bool = False) -> None:
        self.identity = {"schema": "cke.whisper.loaded.v1", "generation": generation}
        self.calls = 0
        self.closed = False
        self.started = threading.Event()
        self.released = threading.Event()
        self.block = block

    def run(self, data: bytes) -> dict:
        self.calls += 1
        self.started.set()
        if self.block:
            self.released.wait(5)
        value = struct.unpack_from("<h", data, 44)[0]
        if value == 9:
            raise RuntimeError("injected native failure")
        if self.closed:
            raise RuntimeError("worker aborted")
        return {"text": "" if value == 0 else f"heard-{value}",
                "stop": "eos", "inference_seconds": 0.01}

    def abort(self) -> None:
        self.closed = True
        self.released.set()

    def close(self) -> None:
        self.closed = True
        self.released.set()


def _wait_queued(client: TestClient) -> None:
    for _ in range(100):
        if client.get("/health").json()["queued"]:
            return
        time.sleep(0.01)
    raise AssertionError("request never entered the queue")


def test_wav_validation_rejects_bad_format_and_truncation() -> None:
    assert _wav_frames(_wav(0), max_bytes=1000, max_seconds=1) == 160
    for data in (_wav(1)[:30], _wav(1)[:-2], b"not a wave"):
        try:
            _wav_frames(data, max_bytes=1000, max_seconds=1)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed audio was accepted")


def test_resident_a_b_a_and_failure_recovery() -> None:
    backends = []

    def factory():
        backend = _FakeBackend(len(backends) + 1)
        backends.append(backend)
        return backend

    with TestClient(create_app(factory, max_upload_bytes=1000, max_seconds=1)) as client:
        for request_id, value in (("a1", 1), ("b", 2), ("a2", 1)):
            response = client.post("/audio/transcriptions", content=_wav(value),
                                   headers={"content-type": "audio/wav",
                                            "x-cke-request-id": request_id})
            assert response.status_code == 200, response.text
            assert response.json()["text"] == f"heard-{value}"
            assert response.json()["worker_generation"] == 1
            assert response.json()["id"] == request_id
        assert len(backends) == 1 and backends[0].calls == 3
        silence = client.post("/audio/transcriptions", content=_wav(0),
                              headers={"content-type": "audio/wav"})
        assert silence.status_code == 200 and silence.json()["text"] == ""
        failed = client.post("/audio/transcriptions", content=_wav(9),
                             headers={"content-type": "audio/wav"})
        assert failed.status_code == 503
        recovered = client.post("/audio/transcriptions", content=_wav(1),
                                headers={"content-type": "audio/wav"})
        assert recovered.status_code == 200
        assert recovered.json()["worker_generation"] == 2
        assert len(backends) == 2 and backends[0].closed


def test_runner_reuses_the_same_encoder_and_decoder_workers(monkeypatch, tmp_path) -> None:
    import numpy as np
    from version.v8.scripts import run_whisper_v8 as whisper

    encoder_dir, decoder_dir = tmp_path / "encoder", tmp_path / "decoder"
    encoder_dir.mkdir()
    decoder_dir.mkdir()
    (encoder_dir / "config.json").write_text(
        json.dumps({"audio_sample_rate": 16000, "audio_sample_extent": 480000}))
    workers = []

    class Worker:
        def __init__(self, role, _directory, _environment, _timeout):
            self.role = role
            self.calls = 0
            self.closed = False
            workers.append(self)

        def run(self, arguments):
            self.calls += 1
            if self.role == "encoder":
                np.save(arguments["encoder_output"], np.zeros((2, 2), dtype=np.float32))
                result = {"audio": {"source_sample_rate": 16000}}
            else:
                result = {"transcript_text": f"run-{self.calls}", "stop": "eos"}
            arguments["worker_report"].write_text(json.dumps(result))

        def close(self):
            self.closed = True

    monkeypatch.setattr(whisper, "_PersistentWorker", Worker)
    monkeypatch.setattr(_WhisperBackend, "_loaded_identity",
                        lambda _self: {"schema": "cke.whisper.loaded.v1"})
    backend = _WhisperBackend(encoder_dir, decoder_dir, 10, tmp_path)
    try:
        assert backend.run(_wav(1))["text"] == "run-1"
        assert backend.run(_wav(2))["text"] == "run-2"
        try:
            backend.run(_wav(3, frames=480001))
        except ValueError:
            pass
        else:
            raise AssertionError("recording beyond native extent was accepted")
        assert len(workers) == 2
        assert [worker.calls for worker in workers] == [2, 2]
    finally:
        backend.close()
    assert all(worker.closed for worker in workers)


def test_identity_failure_closes_both_workers_even_if_one_close_fails(
        monkeypatch, tmp_path) -> None:
    from version.v8.scripts import run_whisper_v8 as whisper

    encoder_dir, decoder_dir = tmp_path / "encoder", tmp_path / "decoder"
    encoder_dir.mkdir()
    decoder_dir.mkdir()
    (encoder_dir / "config.json").write_text(
        json.dumps({"audio_sample_rate": 16000, "audio_sample_extent": 16000}))
    workers = []

    class Worker:
        def __init__(self, role, _directory, _environment, _timeout):
            self.role = role
            self.closed = False
            workers.append(self)

        def close(self):
            self.closed = True
            if self.role == "decoder":
                raise RuntimeError("injected cleanup failure")

    def reject_identity(_self):
        raise ValueError("injected identity failure")

    monkeypatch.setattr(whisper, "_PersistentWorker", Worker)
    monkeypatch.setattr(_WhisperBackend, "_loaded_identity", reject_identity)
    try:
        _WhisperBackend(encoder_dir, decoder_dir, 10, tmp_path)
    except ValueError as error:
        assert "identity failure" in str(error)
    else:
        raise AssertionError("invalid loaded identity was accepted")
    assert len(workers) == 2
    assert all(worker.closed for worker in workers)


def test_preload_reports_ready_without_a_request() -> None:
    backends = []

    def factory():
        backends.append(_FakeBackend(1))
        return backends[-1]

    with TestClient(create_app(factory, preload=True)) as client:
        for _ in range(100):
            if client.get("/health").json()["ready"]:
                break
            time.sleep(0.01)
        assert client.get("/health").json()["ready"]
        assert len(backends) == 1


def test_limits_reject_before_worker_execution() -> None:
    backends = []

    def factory():
        backends.append(_FakeBackend(1))
        return backends[-1]

    with TestClient(create_app(factory, max_upload_bytes=400, max_seconds=0.005)) as client:
        for data, content_type, status in (
            (_wav(1), "text/plain", 415),
            (_wav(1) * 2, "audio/wav", 413),
            (_wav(1), "audio/wav", 400),
            (b"bad", "audio/wav", 400),
        ):
            response = client.post("/audio/transcriptions", content=data,
                                   headers={"content-type": content_type})
            assert response.status_code == status, response.text
        assert not backends


def test_queued_cancel_overflow_and_active_abort() -> None:
    backends = []

    def factory():
        backend = _FakeBackend(len(backends) + 1, block=len(backends) == 0)
        backends.append(backend)
        return backend

    with TestClient(create_app(factory, max_upload_bytes=1000,
                               max_seconds=1, max_waiters=1)) as client:
        def post(request_id: str, value: int):
            return client.post("/audio/transcriptions", content=_wav(value),
                               headers={"content-type": "audio/wav",
                                        "x-cke-request-id": request_id})

        with ThreadPoolExecutor(max_workers=3) as pool:
            a = pool.submit(post, "a", 1)
            for _ in range(100):
                if backends:
                    break
                time.sleep(0.01)
            assert backends
            assert backends[0].started.wait(2)
            b = pool.submit(post, "b", 2)
            _wait_queued(client)
            assert post("overflow", 3).status_code == 429
            assert client.post("/audio/transcriptions/b/cancel").status_code == 200
            assert client.post("/audio/transcriptions/a/cancel").status_code == 200
            assert a.result(timeout=3).status_code == 409
            assert b.result(timeout=3).status_code == 409
            assert post("after", 4).json()["text"] == "heard-4"
            assert len(backends) == 2


def test_queued_cancellation_keeps_the_active_resident_model() -> None:
    backends = []

    def factory():
        backend = _FakeBackend(len(backends) + 1, block=len(backends) == 0)
        backends.append(backend)
        return backend

    with TestClient(create_app(factory, max_upload_bytes=1000,
                               max_seconds=1, max_waiters=1)) as client:
        def post(request_id: str):
            return client.post("/audio/transcriptions", content=_wav(1),
                               headers={"content-type": "audio/wav",
                                        "x-cke-request-id": request_id})

        with ThreadPoolExecutor(max_workers=2) as pool:
            active = pool.submit(post, "active")
            for _ in range(100):
                if backends and backends[0].started.is_set():
                    break
                time.sleep(0.01)
            assert backends[0].started.is_set()
            queued = pool.submit(post, "queued")
            _wait_queued(client)
            assert client.post("/audio/transcriptions/queued/cancel").status_code == 200
            backends[0].released.set()
            assert active.result(timeout=3).status_code == 200
            assert queued.result(timeout=3).status_code == 409
            assert post("next").json()["worker_generation"] == 1
            assert len(backends) == 1 and not backends[0].closed


def test_socket_disconnect_aborts_and_follow_up_succeeds() -> None:
    backends = []

    def factory():
        backend = _FakeBackend(len(backends) + 1, block=len(backends) == 0)
        backends.append(backend)
        return backend

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(
        create_app(factory), host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.01)
        assert server.started
        data = _wav(1)
        with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
            client.sendall((
                "POST /audio/transcriptions HTTP/1.1\r\n"
                "Host: 127.0.0.1\r\nContent-Type: audio/wav\r\n"
                f"Content-Length: {len(data)}\r\n"
                "X-CKE-Request-ID: disconnected\r\n\r\n"
            ).encode() + data)
            for _ in range(100):
                if backends and backends[0].started.is_set():
                    break
                time.sleep(0.01)
            assert backends[0].started.is_set()
        for _ in range(100):
            if backends[0].closed:
                break
            time.sleep(0.01)
        assert backends[0].closed
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/audio/transcriptions", data=_wav(2),
            headers={"content-type": "audio/wav"}, method="POST")
        with urllib.request.urlopen(request, timeout=5) as response:
            result = json.load(response)
        assert result["worker_generation"] == 2
        assert result["text"] == "heard-2"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        assert not thread.is_alive()
