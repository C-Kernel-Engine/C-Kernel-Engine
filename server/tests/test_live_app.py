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

from ck_serve_v8 import (
    create_app,
    SessionBusyError,
    _load_builtin_chat_contract,
    _load_runtime_chat_contract,
    _format_prompt_with_chat_contract,
    _resolve_contract_thinking_overrides,
    CK_SESSION_REQUEST_RAW_PROMPT,
)


class FakeSession:
    """Minimal duck-typed session matching the subset ``create_app`` uses."""

    def __init__(self, chunks=("hello", ",", "world", "!"), *, timing=None) -> None:
        self.chunks = list(chunks)
        self.timing = dict(timing or {})
        self.cancel_called = False
        self.close_called = False
        self.last_flags: int = 0
        self.last_user: str | None = None

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
        self.last_flags = flags
        self.last_user = user
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


class FakeBusySession:
    """Session that always reports busy (ERROR_BUSY = -6)."""

    def generate(self, *args, **kwargs):
        raise SessionBusyError(-6, "session already has an active request")

    def cancel(self) -> None:
        pass

    def close(self) -> None:
        pass


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


def test_cancel_completed_response_is_rejected():
    client, session = make_client()
    created = client.post("/v1/responses", json={"model": "fake-model", "input": "hi"})
    response_id = created.json()["id"]

    resp = client.post(f"/v1/responses/{response_id}/cancel")
    assert resp.status_code == 409
    assert session.cancel_called is False


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
    client, _ = make_client(chunks=("<think>", "\nLet me think", "\n</think>", "\n42"))
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
    client, _ = make_client(chunks=("<think>", "\nLet me think", "\n</think>", "\n42"))
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
    answer_deltas = [
        payload["delta"]
        for event, payload in events
        if event == "response.output_text.delta"
    ]
    assert reasoning_deltas == []
    assert "".join(answer_deltas) == "plain answer"
    assert [i["type"] for i in completed["output"]] == ["message"]
    assert completed["output"][0]["content"][0]["text"] == "plain answer"
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


# --- Busy / single-flight tests -----------------------------------------------


def _make_busy_client():
    session = FakeBusySession()
    return TestClient(create_app(session, model="busy-model"))


def test_busy_non_stream_returns_429():
    client = _make_busy_client()
    resp = client.post("/v1/responses", json={"model": "busy-model", "input": "hi"})
    assert resp.status_code == 429
    assert resp.json()["detail"] == "Session busy: another request is in progress. Retry later."


def test_busy_stream_returns_429():
    """Concurrent streaming request gets 429 when the flight lock is held."""
    import threading as _threading

    lock_held = _threading.Event()
    lock_release = _threading.Event()

    class BlockingSession:
        def generate(self, *a, **kw):
            lock_held.set()
            lock_release.wait(timeout=5)
            return {"prompt_tokens": 0, "generated_tokens": 0, "stop_reason": 0}

        def cancel(self):
            pass

        def close(self):
            pass

    session = BlockingSession()
    app = create_app(session, model="busy-model")
    client = TestClient(app, raise_server_exceptions=False)

    # Acquire the flight lock to simulate an active request.
    # We reach into the closure via the route handler's __wrapped__ or
    # by creating a second app that shares the same lock.
    # Instead, start a streaming request in a thread and wait for it to
    # hold the lock, then fire a second request.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        # First request: streaming, will block in generate()
        first = pool.submit(
            client.post,
            "/v1/responses",
            json={"model": "busy-model", "input": "hi", "stream": True},
        )
        # Wait for the worker thread to enter generate() and hold the lock
        lock_held.wait(timeout=5)

        # Second request: should get 429 because lock is held
        resp = client.post(
            "/v1/responses", json={"model": "busy-model", "input": "hi", "stream": True}
        )
        assert resp.status_code == 429
        assert resp.json()["detail"] == "Session busy: another request is in progress. Retry later."

        # Unblock the first request
        lock_release.set()
        first.result(timeout=5)


# --- Chat contract / reasoning control tests ---------------------------------

QWEN3_CONTRACT = _load_builtin_chat_contract("qwen3")
assert QWEN3_CONTRACT is not None, "qwen3 circuit must exist for chat contract tests"


def test_suppressed_thinking_skips_reasoning():
    session = FakeSession(chunks=("answer",))
    client = TestClient(
        create_app(
            session,
            model="fake-model",
            chat_contract=QWEN3_CONTRACT,
            thinking_mode="auto",
        )
    )
    resp = client.post("/v1/responses", json={"model": "fake-model", "input": "hi"})
    assert resp.status_code == 200
    assert session.last_flags & CK_SESSION_REQUEST_RAW_PROMPT
    assert "/no_think" in (session.last_user or "")
    data = resp.json()
    assert data["output"][0]["content"][0]["text"] == "answer"


def test_visible_thinking_preserves_markers():
    session = FakeSession(chunks=("<think>\nthink\n</think>\nanswer",))
    client = TestClient(
        create_app(
            session,
            model="fake-model",
            chat_contract=QWEN3_CONTRACT,
            thinking_mode="auto",
        )
    )
    resp = client.post(
        "/v1/responses",
        json={"model": "fake-model", "input": "hi", "reasoning": {"effort": "medium"}},
    )
    assert resp.status_code == 200
    assert session.last_flags & CK_SESSION_REQUEST_RAW_PROMPT
    user = session.last_user or ""
    assert "/no_think" not in user
    assert "<|im_start|>assistant\n" in user


def test_no_chat_contract_falls_back_to_c_path():
    session = FakeSession(chunks=("answer",))
    client = TestClient(
        create_app(session, model="fake-model", chat_contract=None)
    )
    resp = client.post("/v1/responses", json={"model": "fake-model", "input": "hi"})
    assert resp.status_code == 200
    assert not (session.last_flags & CK_SESSION_REQUEST_RAW_PROMPT)


def test_thinking_mode_overrides_body():
    session = FakeSession(chunks=("answer",))
    client = TestClient(
        create_app(
            session,
            model="fake-model",
            chat_contract=QWEN3_CONTRACT,
            thinking_mode="suppressed",
        )
    )
    resp = client.post(
        "/v1/responses",
        json={"model": "fake-model", "input": "hi", "reasoning": {"effort": "medium"}},
    )
    assert resp.status_code == 200
    assert session.last_flags & CK_SESSION_REQUEST_RAW_PROMPT
    assert "/no_think" in (session.last_user or "")


def test_suppressed_thinking_stream():
    session = FakeSession(chunks=("answer",))
    client = TestClient(
        create_app(
            session,
            model="fake-model",
            chat_contract=QWEN3_CONTRACT,
            thinking_mode="auto",
        )
    )
    resp = client.post(
        "/v1/responses",
        json={"model": "fake-model", "input": "hi", "stream": True},
    )
    assert resp.status_code == 200
    events = iter_sse(resp.text)
    thinking_deltas = [
        payload["delta"]
        for event, payload in events
        if event == "response.reasoning_text.delta"
    ]
    assert thinking_deltas == []


def test_load_builtin_chat_contract_returns_dict():
    contract = _load_builtin_chat_contract("qwen3")
    assert isinstance(contract, dict)
    assert "assistant_generation_prefix" in contract
    assert "thinking_mode_default" in contract


def test_composite_circuit_inherits_decoder_chat_contract():
    assert _load_builtin_chat_contract("qwen36vl") == _load_builtin_chat_contract("qwen35")


def test_load_builtin_chat_contract_unknown_name():
    assert _load_builtin_chat_contract("nonexistent") is None


@pytest.mark.parametrize(
    "circuit_name",
    ["qwen35", "qwen38", "nemotron_h", "cohere2", "laguna"],
)
def test_runtime_manifest_chat_contract_is_authoritative(tmp_path, circuit_name):
    expected = _load_builtin_chat_contract(circuit_name)
    assert expected is not None
    (tmp_path / "weights_manifest.json").write_text(
        json.dumps({"chat_contract": expected}),
        encoding="utf-8",
    )

    assert _load_runtime_chat_contract(tmp_path) == expected


def test_runtime_manifest_contract_precedes_legacy_nested_copies(tmp_path):
    expected = {"name": "root", "assistant_generation_prefix": "<root>"}
    (tmp_path / "weights_manifest.json").write_text(
        json.dumps(
            {
                "chat_contract": expected,
                "config": {"chat_contract": {"name": "config"}},
                "template": {
                    "contract": {"chat_contract": {"name": "template"}}
                },
            }
        ),
        encoding="utf-8",
    )

    assert _load_runtime_chat_contract(tmp_path) == expected


def test_runtime_manifest_chat_contract_missing(tmp_path):
    assert _load_runtime_chat_contract(tmp_path) is None


def test_resolve_contract_thinking_overrides_visible():
    prefix, last_user = _resolve_contract_thinking_overrides(QWEN3_CONTRACT, "visible")
    assert prefix == "<|im_start|>assistant\n"
    assert last_user == ""


def test_resolve_contract_thinking_overrides_suppressed():
    prefix, last_user = _resolve_contract_thinking_overrides(QWEN3_CONTRACT, "suppressed")
    assert "<think>" in prefix
    assert "/no_think" in last_user


def test_format_prompt_with_chat_contract_suppressed():
    prompt = _format_prompt_with_chat_contract("hi", QWEN3_CONTRACT, thinking_mode="suppressed")
    assert "/no_think" in prompt
    assert "<think>" in prompt


def test_format_prompt_with_chat_contract_visible():
    prompt = _format_prompt_with_chat_contract("hi", QWEN3_CONTRACT, thinking_mode="visible")
    assert "/no_think" not in prompt
    assert "<|im_start|>assistant\n" in prompt


def test_format_prompt_with_no_contract():
    prompt = _format_prompt_with_chat_contract("hi", None, thinking_mode="visible")
    assert prompt == "hi"


def test_instructions_use_contract_system_turn():
    session = FakeSession(chunks=("ok",))
    client = TestClient(
        create_app(session, model="fake-model", chat_contract=QWEN3_CONTRACT)
    )
    response = client.post(
        "/v1/responses",
        json={"model": "fake-model", "input": "hi", "instructions": "be terse"},
    )
    assert response.status_code == 200
    assert "<|im_start|>system\nbe terse<|im_end|>" in session.last_user


def test_request_rejects_model_that_is_not_loaded():
    client, _ = make_client()
    response = client.post(
        "/v1/responses", json={"model": "different-model", "input": "hi"}
    )
    assert response.status_code == 404


def test_request_rejects_unimplemented_tools():
    client, _ = make_client()
    response = client.post(
        "/v1/responses",
        json={
            "model": "fake-model",
            "input": "hi",
            "tools": [{"type": "function", "name": "lookup", "parameters": {}}],
        },
    )
    assert response.status_code == 501
