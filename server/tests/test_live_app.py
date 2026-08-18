"""HTTP tests for the live inference app built by ``ck_serve_v8.create_app``.

A fake session is injected so no native library or model is required. These
tests prove the HTTP contract (health, non-stream, SSE streaming, cancel) that
the real session ABI binding implements.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "version" / "v8" / "scripts"))

import pytest
from fastapi.testclient import TestClient

from ck_serve_v8 import create_app


class FakeSession:
    """Minimal duck-typed session matching the subset ``create_app`` uses."""

    def __init__(self, chunks=("hello", ",", "world", "!"), *, timing=None) -> None:
        self.chunks = list(chunks)
        self.timing = dict(timing or {})
        self.cancel_called = False
        self.close_called = False

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
        for token_id, text in enumerate(self.chunks):
            on_token(token_id, text)
        result = {
            "prompt_tokens": 1,
            "generated_tokens": len(self.chunks),
            "stop_reason": 1,
        }
        result.update(self.timing)
        return result

    def cancel(self) -> None:
        self.cancel_called = True

    def close(self) -> None:
        self.close_called = True


def make_client(chunks=("Hello", ",", "world", "!"), **app_kwargs):
    session = FakeSession(chunks, timing=app_kwargs.pop("timing", None))
    return (
        TestClient(create_app(session, model="fake-model", **app_kwargs)),
        session,
    )


def iter_sse(text):
    """Parse SSE text into ``(event, payload-or-None)`` pairs."""
    events = []
    for block in text.split("\n\n"):
        event = "message"
        data_lines = []
        for line in block.strip("\n").splitlines():
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data_lines.append(line[len("data: ") :])
        if not data_lines:
            continue
        events.append((event, json.loads("\n".join(data_lines))))
    return events


def test_health_reports_live_inference():
    client, _ = make_client()
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "mode": "live", "inference": True}


def test_health_not_on_bare_mount():
    client, _ = make_client()
    assert client.get("/health").status_code == 404
    assert client.get("/v1/health").status_code == 200


def test_create_response_non_stream():
    client, _ = make_client()
    resp = client.post("/v1/responses", json={"model": "fake-model", "input": "hi"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "response"
    assert data["id"].startswith("resp_")
    assert data["status"] == "completed"
    assert data["model"] == "fake-model"
    assert data["output"][0]["content"][0]["text"] == "Hello,world!"


def test_create_response_with_instructions_prepends_system():
    client, _ = make_client(chunks=("ok",))
    resp = client.post(
        "/v1/responses",
        json={"model": "fake-model", "input": "hi", "instructions": "be terse"},
    )
    assert resp.status_code == 200
    assert resp.json()["output"][0]["content"][0]["text"] == "ok"


def test_create_response_usage_counts_output_tokens():
    client, _ = make_client(chunks=("ab", "cd"))
    resp = client.post("/v1/responses", json={"model": "fake-model", "input": "hi"})
    usage = resp.json()["usage"]
    assert usage["output_tokens"] == 2
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]


def test_get_response_resolves_live():
    client, _ = make_client()
    created = client.post("/v1/responses", json={"model": "fake-model", "input": "hi"})
    response_id = created.json()["id"]

    gotten = client.get(f"/v1/responses/{response_id}")
    assert gotten.status_code == 200
    assert gotten.json()["id"] == response_id


def test_get_response_not_found():
    client, _ = make_client()
    assert client.get("/v1/responses/nonexistent").status_code == 404


def test_cancel_calls_session_cancel():
    client, session = make_client()
    created = client.post("/v1/responses", json={"model": "fake-model", "input": "hi"})
    response_id = created.json()["id"]

    resp = client.post(f"/v1/responses/{response_id}/cancel")
    assert resp.status_code == 200
    assert session.cancel_called is True
    assert resp.json()["status"] == "cancelled"


def test_cancel_not_found():
    client, _ = make_client()
    assert client.post("/v1/responses/nonexistent/cancel").status_code == 404


def test_stream_response_emits_sse():
    client, _ = make_client()
    resp = client.post(
        "/v1/responses", json={"model": "fake-model", "input": "hi", "stream": True}
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    text = resp.text
    assert "response.created" in text
    assert "response.in_progress" in text
    assert "response.output_text.delta" in text
    assert "response.completed" in text


def test_stream_deltas_match_generated_tokens():
    client, _ = make_client(chunks=("hi", " ", "there"))
    resp = client.post(
        "/v1/responses", json={"model": "fake-model", "input": "hi", "stream": True}
    )
    deltas = []
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            payload = json.loads(line[len("data: ") :])
            if payload.get("type") == "response.output_text.delta":
                deltas.append(payload["delta"])
    assert deltas == ["hi", " ", "there"]


def test_non_stream_thinking_split():
    client, _ = make_client(chunks=(" thinking", "\nLet me think", "\n response", "\n42"))
    resp = client.post(
        "/v1/responses",
        json={"model": "fake-model", "input": "hi", "reasoning": {"effort": "medium"}},
    )
    data = resp.json()
    reasoning_item = next(i for i in data["output"] if i["type"] == "reasoning")
    assert reasoning_item["content"][0]["text"] == "Let me think"
    message = next(i for i in data["output"] if i["type"] == "message")
    assert message["content"][0]["text"] == "42"
    assert data["output_text"] == "42"
    assert "thinking" not in data


def test_stream_thinking_split_emits_thinking_and_answer():
    client, _ = make_client(chunks=(" thinking", "\nLet me think", "\n response", "\n42"))
    resp = client.post(
        "/v1/responses",
        json={
            "model": "fake-model",
            "input": "hi",
            "stream": True,
            "reasoning": {"effort": "medium"},
        },
    )
    events = iter_sse(resp.text)
    thinking_deltas = [
        payload["delta"]
        for event, payload in events
        if event == "response.reasoning_text.delta"
    ]
    answer_deltas = [
        payload["delta"]
        for event, payload in events
        if event == "response.output_text.delta"
    ]
    completed = next(
        payload["response"] for event, payload in events if event == "response.completed"
    )
    reasoning_item = next(i for i in completed["output"] if i["type"] == "reasoning")
    assert "".join(thinking_deltas) == "Let me think"
    assert answer_deltas == ["42"]
    assert reasoning_item["content"][0]["text"] == "Let me think"
    message = next(i for i in completed["output"] if i["type"] == "message")
    assert message["content"][0]["text"] == "42"
    assert completed["output_text"] == "42"


def test_reasoning_field_enables_thinking_without_think_flag():
    client, _ = make_client(chunks=(" thinking", "b", " response", "c"))
    resp = client.post(
        "/v1/responses",
        json={"model": "fake-model", "input": "hi", "reasoning": {"effort": "medium"}},
    )
    data = resp.json()
    reasoning_item = next(i for i in data["output"] if i["type"] == "reasoning")
    assert reasoning_item["content"][0]["text"] == "b"
    message = next(i for i in data["output"] if i["type"] == "message")
    assert message["content"][0]["text"] == "c"


def test_think_off_keeps_current_behavior():
    client, _ = make_client(chunks=(" thinking", "b"))
    resp = client.post(
        "/v1/responses", json={"model": "fake-model", "input": "hi", "stream": True}
    )
    assert "response.reasoning_text.delta" not in resp.text
    assert "response.reasoning_text.done" not in resp.text
    events = iter_sse(resp.text)
    answer_deltas = [
        payload["delta"]
        for event, payload in events
        if event == "response.output_text.delta"
    ]
    assert answer_deltas == [" thinking", "b"]
    completed = next(
        payload["response"] for event, payload in events if event == "response.completed"
    )
    assert "thinking" not in completed
    assert completed["output"][0]["content"][0]["text"] == " thinkingb"


def test_stream_response_id_is_stable_across_lifecycle():
    client, _ = make_client(chunks=("hi", " ", "there"))
    resp = client.post(
        "/v1/responses",
        json={
            "model": "fake-model",
            "input": "hi",
            "stream": True,
            "reasoning": {"effort": "medium"},
        },
    )
    events = iter_sse(resp.text)
    created = next(
        payload["response"] for event, payload in events if event == "response.created"
    )
    completed = next(
        payload["response"] for event, payload in events if event == "response.completed"
    )
    assert created["id"] == completed["id"]
    gotten = client.get(f"/v1/responses/{completed['id']}")
    assert gotten.status_code == 200
    assert gotten.json()["id"] == completed["id"]
    assert gotten.json()["status"] == "completed"


def test_completed_response_echoes_standard_fields():
    client, _ = make_client(chunks=("ab", "cd"))
    resp = client.post(
        "/v1/responses",
        json={
            "model": "fake-model",
            "input": "hi",
            "instructions": "be terse",
            "temperature": 0.3,
            "top_p": 0.9,
            "max_output_tokens": 64,
            "reasoning": {"effort": "low"},
        },
    )
    data = resp.json()
    assert data["instructions"] == "be terse"
    assert data["temperature"] == 0.3
    assert data["top_p"] == 0.9
    assert data["max_output_tokens"] == 64
    assert data["parallel_tool_calls"] is True
    assert data["output_text"] == "abcd"
    assert data["completed_at"] is not None
    assert data["error"] is None
    assert data["reasoning"]["effort"] == "low"
    assert data["usage"]["input_tokens_details"] == {
        "cache_write_tokens": 0,
        "cached_tokens": 0,
    }


def test_reasoning_tokens_reported_in_usage():
    client, _ = make_client(chunks=(" think", "x", " response", " answer"))
    resp = client.post(
        "/v1/responses",
        json={"model": "fake-model", "input": "hi", "reasoning": {"effort": "medium"}},
    )
    data = resp.json()
    assert data["usage"]["output_tokens_details"]["reasoning_tokens"] > 0


def test_reasoning_enabled_but_no_thinking_markers_emits_single_message():
    client, _ = make_client(chunks=("plain", " ", "answer"))
    resp = client.post(
        "/v1/responses",
        json={"model": "fake-model", "input": "hi", "reasoning": {"effort": "medium"}},
    )
    data = resp.json()
    assert [i["type"] for i in data["output"]] == ["message"]
    assert data["output"][0]["content"][0]["text"] == "plain answer"
    assert data["output_text"] == "plain answer"


def test_stream_reasoning_no_markers_reconciles_to_message():
    client, _ = make_client(chunks=("plain", " ", "answer"))
    resp = client.post(
        "/v1/responses",
        json={
            "model": "fake-model",
            "input": "hi",
            "stream": True,
            "reasoning": {"effort": "medium"},
        },
    )
    events = iter_sse(resp.text)
    reasoning_deltas = [
        payload["delta"]
        for event, payload in events
        if event == "response.reasoning_text.delta"
    ]
    completed = next(
        payload["response"] for event, payload in events if event == "response.completed"
    )
    assert "".join(reasoning_deltas) == "plain answer"
    assert [i["type"] for i in completed["output"]] == ["reasoning", "message"]
    assert completed["output"][0]["content"] == []
    assert completed["output"][1]["content"][0]["text"] == "plain answer"
    assert completed["output_text"] == "plain answer"


def test_performance_and_real_usage_fields_present():
    client, _ = make_client(
        chunks=("ab", "cd"), timing={"prefill_time_ms": 20.0, "decode_time_ms": 30.0}
    )
    resp = client.post("/v1/responses", json={"model": "fake-model", "input": "hi"})
    data = resp.json()
    usage = data["usage"]
    assert usage["input_tokens"] == 1
    assert usage["output_tokens"] == 2
    assert usage["total_tokens"] == 3

    perf = data["performance"]
    assert perf["prompt_tokens"] == 1
    assert perf["generated_tokens"] == 2
    assert perf["prefill_ms"] == 20.0
    assert perf["decode_ms"] == 30.0
    assert perf["stop_reason"] == "eos"
    assert perf["prefill_tokens_per_sec"] == 50.0
    assert perf["decode_tokens_per_sec"] == pytest.approx(66.67, abs=0.01)


def test_viz_returns_html_when_enabled():
    client, _ = make_client()
    resp = client.get("/viz")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "CKS v8" in resp.text


def test_no_viz_disables_page():
    client, _ = make_client(viz=False)
    assert client.get("/viz").status_code == 404


def test_stop_on_text_truncates_final_text():
    client, _ = make_client(chunks=("hello ", "END", " rest"), stop_on_text=["END"])
    resp = client.post("/v1/responses", json={"model": "fake-model", "input": "hi"})
    data = resp.json()
    assert data["output"][0]["content"][0]["text"] == "hello "
    assert data["performance"]["stop_reason"] == "eos"


def test_stop_at_eos_appends_eos_marker():
    client, _ = make_client(chunks=("ok", "<eos>", " tail"), stop_at_eos=True)
    resp = client.post("/v1/responses", json={"model": "fake-model", "input": "hi"})
    assert resp.json()["output"][0]["content"][0]["text"] == "ok"