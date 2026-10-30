"""Harness-compat error envelope, Retry-After, queue, and rejection logging.

Covers the failure mode where an agent harness hammers the single-flight
server (429 storm) and then dies on an unparseable 400 when context runs out.
"""

from __future__ import annotations

import concurrent.futures
import asyncio
import json
import re
import socket
import struct
import threading
import time
import urllib.request

from fastapi.testclient import TestClient
import pytest
import uvicorn
from starlette.requests import ClientDisconnect

from server.live import create_app as _create_app
from server import live


def create_app(*args, **kwargs):
    # Error-envelope fixtures use a deliberately untemplated fake runtime.
    return _create_app(*args, allow_untemplated=True, **kwargs)


def test_stream_disconnected_before_worker_start_is_stored_as_cancelled(monkeypatch):
    class UnusedSession:
        def generate(self, *args, **kwargs):
            pytest.fail("disconnected stream started native generation")

        def cancel(self):
            pytest.fail("no worker exists to cancel")

    def disconnect_before_start(lease, cancelled):
        lease.disconnect()
        return False

    monkeypatch.setattr(live._FlightLease, "start_worker", disconnect_before_start)
    app = _create_app(UnusedSession(), model="m", chat_template="{{ messages[0].content }}")
    response = TestClient(app).post(
        "/v1/responses", json={"model": "m", "input": "hello", "stream": True}
    )
    assert response.status_code == 200
    match = re.search(r"resp_[0-9a-f]{24}", response.text)
    assert match is not None
    stored = TestClient(app).get(f"/v1/responses/{match.group()}")
    assert stored.status_code == 200
    assert stored.json()["status"] == "cancelled"
    assert not app.state.flight_lock.locked()


def test_stream_send_failure_runs_disconnect_settlement_before_worker_start():
    class Lease:
        def __init__(self):
            self.disconnected = False

        def disconnect(self):
            self.disconnected = True

    lease = Lease()
    settled = []

    async def chunks():
        yield b"data: response.created\n\n"
        pytest.fail("stream advanced after the client disconnected")

    async def send(message):
        if message["type"] == "http.response.body":
            raise OSError("client disconnected")

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    response = live._OwnedStreamingResponse(
        chunks(), lease=lease, on_disconnect=lambda: settled.append(lease.disconnected)
    )
    scope = {"type": "http", "method": "GET", "path": "/", "headers": [],
             "asgi": {"version": "3.0", "spec_version": "2.4"}}
    with pytest.raises(ClientDisconnect):
        asyncio.run(response(scope, receive, send))
    assert settled == [True]


def test_nonstream_disconnect_cancels_native_generation(monkeypatch):
    class WaitingSession:
        def __init__(self):
            self.started = threading.Event()
            self.cancelled = threading.Event()

        def generate(self, system, user, *, max_tokens, temperature, top_p,
                     on_token, flags=0, stop_on_text=(), stop_at_eos=False):
            self.started.set()
            if not self.cancelled.wait(timeout=2):
                raise AssertionError("native generation was not cancelled")
            on_token(0, "")
            return {"prompt_tokens": 1, "generated_tokens": 0, "stop_reason": 3}

        def cancel(self):
            self.cancelled.set()

        def close(self):
            pass

    session = WaitingSession()

    async def disconnected(_request):
        # Admission must still succeed; the disconnect occurs during native work.
        return session.started.is_set()

    monkeypatch.setattr("server.live.Request.is_disconnected", disconnected)
    app = create_app(session, model="m")
    response = TestClient(app).post("/v1/responses", json={"model": "m", "input": "hello"})
    assert response.status_code == 200
    assert session.cancelled.is_set()
    assert not app.state.flight_lock.locked()


def test_nonstream_socket_disconnect_releases_session():
    class WaitingSession:
        def __init__(self):
            self.started = threading.Event()
            self.cancelled = threading.Event()

        def generate(self, system, user, *, max_tokens, temperature, top_p,
                     on_token, flags=0, stop_on_text=(), stop_at_eos=False):
            self.started.set()
            if not self.cancelled.wait(timeout=5):
                raise AssertionError("disconnected request kept generating")
            return {"prompt_tokens": 1, "generated_tokens": 0, "stop_reason": 3}

        def cancel(self):
            self.cancelled.set()

        def close(self):
            pass

    session = WaitingSession()
    app = _create_app(session, model="m", chat_template="{{ messages[0].content }}")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.02)
        assert server.started
        payload = json.dumps({"model": "m", "input": "hello"}).encode()
        connection = socket.create_connection(("127.0.0.1", port), timeout=3)
        try:
            connection.sendall(
                b"POST /v1/responses HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                b"Content-Type: application/json\r\nContent-Length: "
                + str(len(payload)).encode() + b"\r\nConnection: close\r\n\r\n" + payload
            )
            assert session.started.wait(timeout=3)
        finally:
            connection.close()
        assert session.cancelled.wait(timeout=3)
        for _ in range(100):
            if not app.state.flight_lock.locked():
                break
            time.sleep(0.02)
        assert not app.state.flight_lock.locked()
    finally:
        server.should_exit = True
        thread.join(timeout=3)
        listener.close()


def test_unread_socket_stream_does_not_hold_completed_native_session():
    class ControlledSession:
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()
            self.calls = 0
            self.cancel_calls = 0

        def generate(self, system, user, *, on_token, **kwargs):
            self.calls += 1
            if self.calls == 1:
                self.started.set()
                assert self.release.wait(5)
            on_token(0, "done")
            return {"prompt_tokens": 1, "generated_tokens": 1, "stop_reason": 1}

        def cancel(self):
            self.cancel_calls += 1

    session = ControlledSession()
    app = _create_app(session, model="m", chat_template="{{ messages[0].content }}")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    connection = None
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(.02)
        assert server.started
        payload = json.dumps({"model": "m", "input": "first", "stream": True}).encode()
        connection = socket.create_connection(("127.0.0.1", port), timeout=3)
        connection.sendall(
            b"POST /v1/responses HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\nContent-Length: "
            + str(len(payload)).encode() + b"\r\n\r\n" + payload
        )
        assert session.started.wait(3)
        headers = b""
        while b"\r\n\r\n" not in headers:
            chunk = connection.recv(4096)
            assert chunk, "stream socket closed before HTTP headers"
            headers += chunk
        assert b"200 OK" in headers.split(b"\r\n", 1)[0]
        assert app.state.flight_lock.locked()
        # Leave the first socket open without consuming its terminal SSE data.
        session.release.set()
        for _ in range(100):
            if not app.state.flight_lock.locked():
                break
            time.sleep(.01)
        assert not app.state.flight_lock.locked()
        second = json.dumps({"model": "m", "input": "second"}).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/responses", data=second,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            assert json.load(response)["output_text"] == "done"
        assert session.calls == 2
        assert session.cancel_calls == 0
    finally:
        session.release.set()
        if connection is not None:
            connection.close()
        server.should_exit = True
        thread.join(timeout=3)
        listener.close()


@pytest.mark.parametrize("native_error", [False, True])
def test_stream_socket_disconnect_during_silent_prefill_cancels_and_recovers(native_error):
    class SilentPrefillSession:
        def __init__(self):
            self.started = threading.Event()
            self.cancelled = threading.Event()
            self.calls = 0
            self.cancel_calls = 0

        def generate(self, system, user, *, on_token, **kwargs):
            self.calls += 1
            if self.calls == 1:
                self.started.set()
                if not self.cancelled.wait(5):
                    raise AssertionError("disconnected stream kept prefill active")
                if native_error:
                    raise RuntimeError("native prefill failed after disconnect")
                return {"prompt_tokens": 1, "generated_tokens": 0, "stop_reason": 3}
            on_token(0, "done")
            return {"prompt_tokens": 1, "generated_tokens": 1, "stop_reason": 1}

        def cancel(self):
            self.cancel_calls += 1
            self.cancelled.set()

    session = SilentPrefillSession()
    app = _create_app(session, model="m", chat_template="{{ messages[0].content }}")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    connection = None
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.02)
        assert server.started
        payload = json.dumps({"model": "m", "input": "silent prefill", "stream": True}).encode()
        connection = socket.create_connection(("127.0.0.1", port), timeout=3)
        connection.sendall(
            b"POST /v1/responses HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\nContent-Length: "
            + str(len(payload)).encode() + b"\r\n\r\n" + payload
        )
        assert session.started.wait(3)
        received = b""
        while b"response.in_progress" not in received:
            received += connection.recv(4096)
            assert received, "stream closed before in-progress event"
        match = re.search(rb"resp_[0-9a-f]{24}", received)
        assert match is not None
        response_id = match.group().decode()
        # Abort the socket while generated prefill is still silent. The next
        # keep-alive send must surface the disconnect and cancel native work.
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        connection.close()
        connection = None
        assert session.cancelled.wait(4)
        for _ in range(100):
            if not app.state.flight_lock.locked():
                break
            time.sleep(0.02)
        assert not app.state.flight_lock.locked()
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/v1/responses/{response_id}", timeout=3
        ) as stored:
            result = json.load(stored)
            assert result["status"] == ("failed" if native_error else "cancelled")
            if native_error:
                assert result["error"]["code"] == "server_error"
        second = json.dumps({"model": "m", "input": "second"}).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/responses", data=second,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            assert json.load(response)["output_text"] == "done"
        assert session.calls == 2
        assert session.cancel_calls == 1
    finally:
        session.cancelled.set()
        if connection is not None:
            connection.close()
        server.should_exit = True
        thread.join(timeout=3)
        listener.close()


class FakeSession:
    def __init__(self, chunks=("hello",), *, delay=0.0):
        self.chunks = list(chunks)
        self.delay = delay

    def generate(
        self,
        system,
        user,
        *,
        max_tokens,
        temperature,
        top_p,
        on_token,
        flags=0,
        stop_on_text=(),
        stop_at_eos=False,
    ):
        for i, text in enumerate(self.chunks):
            if self.delay:
                time.sleep(self.delay)
            on_token(i, text)
        return {"prompt_tokens": 1, "generated_tokens": len(self.chunks), "stop_reason": 1}

    def cancel(self):
        pass

    def close(self):
        pass


class CountingSession(FakeSession):
    def __init__(self, *args, prompt_tokens=0, **kwargs):
        super().__init__(*args, **kwargs)
        self._prompt_tokens = prompt_tokens

    def count_tokens(self, text):
        return self._prompt_tokens


def test_capacity_400_has_openai_envelope():
    client = TestClient(create_app(FakeSession(), model="m", context_length=128))
    resp = client.post("/v1/responses", json={"model": "m", "input": "hi", "max_output_tokens": 128})
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "context_length_exceeded"
    assert body["error"]["type"] == "invalid_request_error"
    assert "leaves no room" in body["error"]["message"]
    assert "leaves no room" in body["detail"]


def test_prompt_over_budget_400_has_envelope():
    session = CountingSession(prompt_tokens=100)
    client = TestClient(create_app(session, model="m", context_length=110))
    resp = client.post("/v1/responses", json={"model": "m", "input": "hi", "max_output_tokens": 32})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "context_length_exceeded"


@pytest.mark.parametrize("route", ["/v1/responses", "/v1/chat/completions"])
def test_native_token_count_failure_is_structured_and_recoverable(route):
    import sys
    from pathlib import Path
    from server.session_v8 import SessionError

    scripts = str(Path(__file__).resolve().parents[2] / "version" / "v8" / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from ck_serve_v8 import create_app as create_full_app

    class FailingCountSession(FakeSession):
        def __init__(self):
            super().__init__()
            self.fail_count = True
            self.generate_calls = 0

        def count_tokens(self, text):
            if self.fail_count:
                raise SessionError(-7, "native tokenizer failed")
            return 2

        def generate(self, *args, **kwargs):
            self.generate_calls += 1
            return super().generate(*args, **kwargs)

    session = FailingCountSession()
    client = TestClient(create_full_app(session, model="m", context_length=128,
                                        allow_untemplated=True))
    payload = ({"model": "m", "input": "long", "max_output_tokens": 8}
               if route.endswith("responses") else
               {"model": "m", "messages": [{"role": "user", "content": "long"}],
                "max_tokens": 8})
    rejected = client.post(route, json=payload)
    assert rejected.status_code == 500, rejected.text
    assert rejected.json()["error"]["code"] == "native_tokenization_failed"
    assert session.generate_calls == 0
    session.fail_count = False
    recovered = client.post(route, json=payload)
    assert recovered.status_code == 200, recovered.text
    assert session.generate_calls == 1


@pytest.mark.parametrize("route", ["/v1/responses", "/v1/chat/completions"])
def test_busy_token_count_keeps_busy_status(route):
    from server.session_v8 import SessionBusyError

    class BusySession(FakeSession):
        def count_tokens(self, text):
            raise SessionBusyError(-6, "session busy")

    from version.v8.scripts.ck_serve_v8 import create_app as create_full_app
    client = TestClient(create_full_app(BusySession(), model="m", context_length=128,
                                        allow_untemplated=True))
    payload = ({"model": "m", "input": "hello", "max_output_tokens": 8}
               if route.endswith("responses") else
               {"model": "m", "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 8})
    response = client.post(route, json=payload)
    assert response.status_code == 429, response.text
    assert response.json()["error"]["code"] == "rate_limit_exceeded"
    assert int(response.headers["Retry-After"]) >= 1


@pytest.mark.parametrize("route", ["/v1/responses", "/v1/chat/completions"])
def test_explicit_rendered_byte_cap_rejects_before_native_work(route):
    class CountingSession(FakeSession):
        def __init__(self):
            super().__init__()
            self.count_calls = 0

        def count_tokens(self, text):
            self.count_calls += 1
            return 2

    from version.v8.scripts.ck_serve_v8 import create_app as create_full_app
    session = CountingSession()
    client = TestClient(create_full_app(session, model="m", context_length=128,
                                        request_prompt_byte_cap=8, allow_untemplated=True))
    payload = ({"model": "m", "input": "a" * 9, "max_output_tokens": 8}
               if route.endswith("responses") else
               {"model": "m", "messages": [{"role": "user", "content": "a" * 9}],
                "max_tokens": 8})
    response = client.post(route, json=payload)
    assert response.status_code == 413, response.text
    assert response.json()["error"]["code"] == "prompt_bytes_limit_exceeded"
    assert session.count_calls == 0


def test_unknown_call_id_400_has_envelope_and_detail():
    client = TestClient(create_app(FakeSession(), model="m"))
    resp = client.post(
        "/v1/responses",
        json={
            "model": "m",
            "input": [{"type": "function_call_output", "call_id": "call_nope", "output": "x"}],
        },
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "unknown_call_id"
    assert "unknown call_id" in body["detail"]


def test_model_404_keeps_detail_with_envelope():
    client = TestClient(create_app(FakeSession(), model="m"))
    resp = client.post("/v1/responses", json={"model": "other", "input": "hi"})
    assert resp.status_code == 404
    body = resp.json()
    assert "not loaded" in body["detail"]
    assert body["error"]["message"] == body["detail"]


def test_queued_second_request_succeeds():
    session = FakeSession(("done",), delay=0.4)
    app = create_app(session, model="m")
    client = TestClient(app)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(client.post, "/v1/responses", json={"model": "m", "input": "one"})
        time.sleep(0.1)  # let the first request take the flight lock
        second = client.post("/v1/responses", json={"model": "m", "input": "two"})
        assert second.status_code == 200
        assert second.json()["status"] == "completed"
        assert first.result(timeout=10).status_code == 200


def test_429_carries_retry_after_and_envelope(monkeypatch):
    import server.live as live

    monkeypatch.setattr(live, "_FLIGHT_WAIT_SECONDS", 0.3)
    app = create_app(FakeSession(), model="m")
    client = TestClient(app)
    assert app.state.flight_lock.acquire(blocking=False)
    try:
        resp = client.post("/v1/responses", json={"model": "m", "input": "hi"})
    finally:
        app.state.flight_lock.release()
    assert resp.status_code == 429
    assert resp.headers["retry-after"] == "1"
    body = resp.json()
    assert body["error"]["code"] == "rate_limit_exceeded"
    assert body["error"]["type"] == "rate_limit_error"
    assert "Session busy" in body["detail"]


def test_rejection_is_logged(capsys):
    client = TestClient(create_app(FakeSession(), model="m", context_length=64))
    resp = client.post("/v1/responses", json={"model": "m", "input": "hi", "max_output_tokens": 64})
    assert resp.status_code == 400
    out = capsys.readouterr().out
    assert "rejected 400" in out
    assert "context_length_exceeded" in out
