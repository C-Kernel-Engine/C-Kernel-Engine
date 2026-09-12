"""Chat Completions compatibility tests for local coding-agent clients."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "version" / "v8" / "scripts")
)

from ck_serve_v8 import create_app


TOOL_TEMPLATE = """
{% for message in messages %}
{{ message.role }}:{{ message.content }}
{% for call in message.get('tool_calls', []) %}
CALL:{{ call.id }}:{{ call.function.name }}:{{ call.function.arguments | tojson }}
{% endfor %}
{% if message.get('tool_call_id') %}RESULT:{{ message.tool_call_id }}{% endif %}
{% endfor %}
{% for tool in tools or [] %}TOOL:{{ tool.name }}{% endfor %}
assistant:
"""


class FakeSession:
    def __init__(self, chunks=("hello",), *, stop_reason=1, error=None):
        self.chunks = list(chunks)
        self.stop_reason = stop_reason
        self.error = error
        self.last_prompt = None

    def generate(self, _system, prompt, *, on_token, **_kwargs):
        self.last_prompt = prompt
        for token_id, text in enumerate(self.chunks):
            on_token(token_id, text)
        if self.error:
            raise self.error
        return {
            "prompt_tokens": 7,
            "generated_tokens": len(self.chunks),
            "stop_reason": self.stop_reason,
            "prefill_time_ms": 2.0,
            "decode_time_ms": 3.0,
        }

    def cancel(self):
        pass

    def close(self):
        pass


def make_client(chunks=("hello",), **session_kwargs):
    session = FakeSession(chunks, **session_kwargs)
    app = create_app(
        session,
        model="qwen38-local",
        chat_template=TOOL_TEMPLATE,
        chat_contract={"name": "test"},
        viz=False,
    )
    return TestClient(app), session


def parse_chat_sse(text):
    rows = []
    for block in text.split("\n\n"):
        data = "\n".join(
            line[6:] for line in block.splitlines() if line.startswith("data: ")
        )
        if data:
            rows.append(data if data == "[DONE]" else json.loads(data))
    return rows


def tool_request(*, stream=False):
    return {
        "model": "qwen38-local",
        "messages": [{"role": "user", "content": "read the file"}],
        "stream": stream,
        "stream_options": {"include_usage": True} if stream else None,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_non_stream_text_completion_uses_responses_runtime():
    client, session = make_client(("hello", " world"))
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen38-local",
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hello"},
            ],
            "max_tokens": 32,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"] == {
        "role": "assistant",
        "content": "hello world",
    }
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 2,
        "total_tokens": 9,
    }
    assert "system:be terse" in session.last_prompt
    assert "user:hello" in session.last_prompt


def test_non_stream_tool_call_and_result_history_preserve_call_id():
    client, session = make_client(
        ('{"name":"read_file","arguments":{"path":"README.md"}}',)
    )
    first = client.post("/v1/chat/completions", json=tool_request())
    assert first.status_code == 200
    choice = first.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "read_file"
    assert json.loads(call["function"]["arguments"]) == {"path": "README.md"}

    session.chunks = ["The file is a project README."]
    second_body = tool_request()
    second_body["messages"] = [
        {"role": "user", "content": "read the file"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [call],
        },
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "content": "CKE README contents",
        },
    ]
    second = client.post("/v1/chat/completions", json=second_body)
    assert second.status_code == 200
    assert second.json()["choices"][0]["message"]["content"].startswith("The file")
    assert f"CALL:{call['id']}:read_file" in session.last_prompt
    assert f"RESULT:{call['id']}" in session.last_prompt
    assert "tool:CKE README contents" in session.last_prompt


def test_streaming_tool_call_has_one_argument_stream_and_usage():
    client, _ = make_client(
        ('{"name":"read_file","arguments":{"path":"README.md"}}',)
    )
    response = client.post("/v1/chat/completions", json=tool_request(stream=True))
    assert response.status_code == 200
    rows = parse_chat_sse(response.text)
    assert rows[-1] == "[DONE]"
    chunks = [row for row in rows[:-1] if "error" not in row]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    tool_deltas = [
        choice["delta"]["tool_calls"][0]
        for row in chunks
        for choice in row["choices"]
        if choice["delta"].get("tool_calls")
    ]
    assert tool_deltas[0]["function"]["name"] == "read_file"
    arguments = "".join(
        delta["function"].get("arguments", "") for delta in tool_deltas
    )
    assert json.loads(arguments) == {"path": "README.md"}
    finishes = [
        choice["finish_reason"]
        for row in chunks
        for choice in row["choices"]
        if choice["finish_reason"] is not None
    ]
    assert finishes == ["tool_calls"]
    usage_rows = [row for row in chunks if not row["choices"]]
    assert usage_rows[0]["usage"]["total_tokens"] == 8
    assert not any(
        choice["delta"].get("content")
        for row in chunks
        for choice in row["choices"]
    )


def test_chat_completions_rejects_lossy_or_unsupported_inputs():
    client, _ = make_client()
    base = {
        "model": "qwen38-local",
        "messages": [{"role": "user", "content": "hi"}],
    }
    for update, expected in (
        ({"n": 2}, "n=1"),
        ({"stop": ["END"]}, "stop"),
        ({"seed": 3}, "seed"),
        ({"chat_template_kwargs": {"unknown": True}}, "unknown"),
    ):
        response = client.post("/v1/chat/completions", json={**base, **update})
        assert response.status_code == 422
        assert expected in response.json()["detail"]

    unknown = client.post(
        "/v1/chat/completions",
        json={
            **base,
            "messages": [{"role": "tool", "tool_call_id": "missing", "content": "x"}],
        },
    )
    assert unknown.status_code == 400
    assert "unknown tool_call_id" in unknown.json()["detail"]


def test_non_stream_runtime_failure_is_not_a_successful_completion():
    client, _ = make_client(error=RuntimeError("native failure"))
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen38-local",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 500
    assert "native failure" in response.json()["detail"]
