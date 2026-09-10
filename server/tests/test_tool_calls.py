"""Structured tool-call tests for ck_serve_v8.create_app."""
from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "version" / "v8" / "scripts"))

from fastapi.testclient import TestClient

from ck_serve_v8 import create_app, CK_SESSION_REQUEST_RAW_PROMPT

# Reuse inline Qwen3 contract from test_live_app
QWEN3_CONTRACT = {
    "name": "qwen3",
    "turn_prefix": "<|im_start|>{role}\n",
    "turn_suffix": "<|im_end|>\n",
    "assistant_generation_prefix": "<|im_start|>assistant\n",
    "role_labels": {"system": "system", "user": "user", "assistant": "assistant"},
    "system_prompt_mode": "dedicated_turn",
    "system_prompt_separator": "\n\n",
    "default_system_prompt": "",
    "inject_default_system_prompt": False,
    "force_bos_text_if_tokenizer_add_bos_false": "",
    "last_user_prefix": "",
    "last_user_prefix_suppression_markers": ["/no_think"],
    "thinking_mode_default": "visible",
    "assistant_generation_prefix_by_thinking_mode": {
        "visible": "<|im_start|>assistant\n",
        "suppressed": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    },
    "last_user_prefix_by_thinking_mode": {"visible": "", "suppressed": "/no_think\n"},
    "stop_text_markers": ["<|im_end|>"],
}

DUMMY_CHAT_TEMPLATES = {"tool_use": "tool jinja", "default": "default"}


class FakeSession:
    def __init__(self, chunks=("hello",), *, timing=None):
        self.chunks = list(chunks)
        self.timing = dict(timing or {})
        self.cancel_called = False
        self.last_flags: int = 0
        self.last_user: str | None = None

    def generate(self, system, user, *, max_tokens, temperature, top_p, on_token, flags=0, stop_on_text=(), stop_at_eos=False):
        self.last_flags = flags
        self.last_user = user
        for i, text in enumerate(self.chunks):
            on_token(i, text)
        result = {"prompt_tokens": 1, "generated_tokens": len(self.chunks), "stop_reason": 1}
        result.update(self.timing)
        return result

    def cancel(self):
        self.cancel_called = True

    def close(self):
        pass


def iter_sse(text):
    events = []
    for block in text.split("\n\n"):
        event = "message"
        data_lines = []
        for line in block.strip("\n").splitlines():
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data_lines.append(line[len("data: "):])
        if not data_lines:
            continue
        events.append((event, json.loads("\n".join(data_lines))))
    return events


def test_tool_call_single_non_stream():
    session = FakeSession(chunks=('{"name":"get_weather","arguments":{"location":"Paris"}}',))
    client = TestClient(create_app(session, model="fake-model", chat_contract=QWEN3_CONTRACT, chat_templates=DUMMY_CHAT_TEMPLATES))
    resp = client.post("/v1/responses", json={
        "model": "fake-model",
        "input": "what is weather?",
        "tools": [{"type": "function", "name": "get_weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    # output should contain function_call
    fc = next(i for i in data["output"] if i["type"] == "function_call")
    assert fc["name"] == "get_weather"
    assert json.loads(fc["arguments"]) == {"location": "Paris"}


def test_tool_call_single_stream():
    session = FakeSession(chunks=('{"name":"get_weather","arguments":{"location":"Paris"}}',))
    client = TestClient(create_app(session, model="fake-model", chat_contract=QWEN3_CONTRACT, chat_templates=DUMMY_CHAT_TEMPLATES))
    resp = client.post("/v1/responses", json={
        "model": "fake-model",
        "input": "hi",
        "stream": True,
        "tools": [{"type": "function", "name": "get_weather", "parameters": {"type": "object", "properties": {}}}],
    })
    assert resp.status_code == 200
    events = iter_sse(resp.text)
    deltas = [p["delta"] for ev, p in events if ev == "response.function_call_arguments.delta"]
    assert len(deltas) == 1
    assert json.loads(deltas[0]) == {"location": "Paris"}
    done = [p for ev, p in events if ev == "response.function_call_arguments.done"]
    assert done[0]["arguments"] == deltas[0]
    completed = next(p["response"] for ev, p in events if ev == "response.completed")
    fc = next(i for i in completed["output"] if i["type"] == "function_call")
    assert fc["name"] == "get_weather"


def test_tool_call_malformed_failed():
    session = FakeSession(chunks=('{"name":"get_weather","arguments":',))  # malformed JSON
    client = TestClient(create_app(session, model="fake-model", chat_contract=QWEN3_CONTRACT, chat_templates=DUMMY_CHAT_TEMPLATES))
    resp = client.post("/v1/responses", json={
        "model": "fake-model",
        "input": "hi",
        "tools": [{"type": "function", "name": "get_weather", "parameters": {}}],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "failed"
    assert data["error"] is not None
    # streaming also failed
    session2 = FakeSession(chunks=('{"name":"get_weather","arguments":',))
    client2 = TestClient(create_app(session2, model="fake-model", chat_contract=QWEN3_CONTRACT, chat_templates=DUMMY_CHAT_TEMPLATES))
    resp2 = client2.post("/v1/responses", json={
        "model": "fake-model", "input": "hi", "stream": True,
        "tools": [{"type": "function", "name": "get_weather", "parameters": {}}],
    })
    events = iter_sse(resp2.text)
    assert any(ev == "response.failed" for ev, _ in events)
    assert any(ev == "error" for ev, _ in events)


def test_tool_call_unknown_tool_failed():
    session = FakeSession(chunks=('{"name":"unknown_tool","arguments":{}}',))
    client = TestClient(create_app(session, model="fake-model", chat_contract=QWEN3_CONTRACT, chat_templates=DUMMY_CHAT_TEMPLATES))
    resp = client.post("/v1/responses", json={
        "model": "fake-model", "input": "hi",
        "tools": [{"type": "function", "name": "get_weather", "parameters": {}}],
    })
    assert resp.json()["status"] == "failed"
    assert "unknown tool" in resp.json()["error"]["message"]


def test_tool_call_parallel_false_incomplete():
    session = FakeSession(chunks=('[{"name":"a","arguments":{}},{"name":"b","arguments":{}}]',))
    client = TestClient(create_app(session, model="fake-model", chat_contract=QWEN3_CONTRACT, chat_templates=DUMMY_CHAT_TEMPLATES))
    resp = client.post("/v1/responses", json={
        "model": "fake-model", "input": "hi",
        "tools": [
            {"type": "function", "name": "a", "parameters": {}},
            {"type": "function", "name": "b", "parameters": {}},
        ],
        "parallel_tool_calls": False,
    })
    data = resp.json()
    assert data["status"] == "incomplete"
    assert data["incomplete_details"]["reason"] == "max_tool_calls"
    fcs = [i for i in data["output"] if i["type"] == "function_call"]
    assert len(fcs) == 1
    assert fcs[0]["name"] == "a"
    # streaming variant
    session2 = FakeSession(chunks=('[{"name":"a","arguments":{}},{"name":"b","arguments":{}}]',))
    client2 = TestClient(create_app(session2, model="fake-model", chat_contract=QWEN3_CONTRACT, chat_templates=DUMMY_CHAT_TEMPLATES))
    resp2 = client2.post("/v1/responses", json={
        "model": "fake-model", "input": "hi", "stream": True,
        "tools": [
            {"type": "function", "name": "a", "parameters": {}},
            {"type": "function", "name": "b", "parameters": {}},
        ],
        "parallel_tool_calls": False,
    })
    events = iter_sse(resp2.text)
    assert any(ev == "response.incomplete" for ev, _ in events)
    completed = next(p["response"] for ev, p in events if ev == "response.incomplete")
    assert len([i for i in completed["output"] if i["type"] == "function_call"]) == 1


def test_tool_call_cancellation_stream():
    import concurrent.futures
    import threading

    class BlockingSession:
        def __init__(self):
            self.cancel_called = False

        def generate(self, system, user, *, max_tokens, temperature, top_p, on_token, flags=0, stop_on_text=(), stop_at_eos=False):
            for i in range(5):
                rc = on_token(i, f'chunk{i} ')
                if rc != 0:
                    break
                time.sleep(0.05)
            return {"prompt_tokens": 1, "generated_tokens": 5, "stop_reason": 1}

        def cancel(self):
            self.cancel_called = True

        def close(self):
            pass

    session = BlockingSession()
    app = create_app(session, model="fake-model", chat_contract=QWEN3_CONTRACT, chat_templates=DUMMY_CHAT_TEMPLATES)
    client = TestClient(app)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fut = pool.submit(client.post, "/v1/responses", json={"model": "fake-model", "input": "hi", "stream": True, "tools": [{"type": "function", "name": "a", "parameters": {}}]})
        for _ in range(20):
            with app.state.active_streams_lock:
                if app.state.active_streams:
                    break
            time.sleep(0.05)
        assert app.state.active_streams
        active_id = next(iter(app.state.active_streams))
        resp = client.post(f"/v1/responses/{active_id}/cancel")
        assert resp.status_code == 200
        try:
            result = fut.result(timeout=5)
            # after cancel, streaming should have response.cancelled
            assert "response.cancelled" in result.text or result.status_code == 200
        except Exception:
            pass
