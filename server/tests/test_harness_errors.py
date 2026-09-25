"""Harness-compat error envelope, Retry-After, queue, and rejection logging.

Covers the failure mode where an agent harness hammers the single-flight
server (429 storm) and then dies on an unparseable 400 when context runs out.
"""

from __future__ import annotations

import concurrent.futures
import time

from fastapi.testclient import TestClient

from server.live import create_app


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
    assert resp.headers["retry-after"] == "0"
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
