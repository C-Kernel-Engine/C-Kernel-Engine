"""Incremental streaming when tools are attached (no reasoning requested).

Regression cover for harnesses that saw zero SSE until ``done``: with
function tools present, raw chunks used to buffer fully. Now the first-chunk
classifier streams tool-shaped output as ``function_call_arguments.delta``
and prose as ``output_text.delta`` live; terminals are still decided by the
full-text done-parse, so outcomes never change — only visibility does.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "version" / "v8" / "scripts"))

from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from ck_serve_v8 import _classify_stream_mode, create_app
from server.schemas.streaming import ResponseStreamEvent

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
NATIVE_TOOL_TEMPLATES = {
    "tool_use": "{% for m in messages %}{{ m.content }}{% endfor %}<tool_call></tool_call>"
}
STREAM_EVENT_ADAPTER = TypeAdapter(ResponseStreamEvent)
TOOLS = [{"type": "function", "name": "get_weather", "parameters": {"type": "object"}}]


class FakeSession:
    def __init__(self, chunks=("hello",)):
        self.chunks = list(chunks)

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
            on_token(i, text)
        return {"prompt_tokens": 1, "generated_tokens": len(self.chunks), "stop_reason": 1}

    def cancel(self):
        pass

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


def assert_valid_stream(events):
    assert [payload["sequence_number"] for _, payload in events] == list(range(len(events)))
    for event, payload in events:
        assert payload["type"] == event
        STREAM_EVENT_ADAPTER.validate_python(payload)


def test_classify_stream_mode():
    assert _classify_stream_mode("") is None
    assert _classify_stream_mode("   \n  ") is None
    assert _classify_stream_mode('{"name":"a"}') == "tool"
    assert _classify_stream_mode('  [{"name":"a"}]') == "tool"
    assert _classify_stream_mode("<tool_call><function=a>") == "tool"
    assert _classify_stream_mode("<TOOL_CALL>") == "tool"
    assert _classify_stream_mode("Hello there") == "text"
    assert _classify_stream_mode("<think>hmm</think>done") == "text"


def test_tool_mode_streams_each_chunk():
    chunks = ('{"name":"get', '_weather","arguments":', '{"location":"Paris"}}')
    session = FakeSession(chunks=chunks)
    client = TestClient(create_app(session, model="m", chat_contract=QWEN3_CONTRACT, chat_templates=DUMMY_CHAT_TEMPLATES))
    resp = client.post("/v1/responses", json={"model": "m", "input": "hi", "stream": True, "tools": TOOLS})
    assert resp.status_code == 200
    events = iter_sse(resp.text)
    assert_valid_stream(events)
    # No text deltas in tool mode; one args delta per chunk, before completed.
    assert not any(ev == "response.output_text.delta" for ev, _ in events)
    deltas = [p for ev, p in events if ev == "response.function_call_arguments.delta"]
    assert [p["delta"] for p in deltas] == list(chunks)
    first_delta_idx = next(i for i, (ev, _) in enumerate(events) if ev == "response.function_call_arguments.delta")
    completed_idx = next(i for i, (ev, _) in enumerate(events) if ev == "response.completed")
    assert first_delta_idx < completed_idx
    # Deltas concatenate to the raw generation; done carries parsed arguments.
    assert "".join(p["delta"] for p in deltas) == "".join(chunks)
    done = [p for ev, p in events if ev == "response.function_call_arguments.done"]
    assert len(done) == 1
    assert json.loads(done[0]["arguments"]) == {"location": "Paris"}
    # added -> delta(s) -> done share one item id, reused in the final payload.
    added = [p for ev, p in events if ev == "response.output_item.added"]
    assert len(added) == 1
    assert {p["item_id"] for p in deltas} == {done[0]["item_id"]} == {added[0]["item"]["id"]}
    completed = next(p["response"] for ev, p in events if ev == "response.completed")
    fc = next(i for i in completed["output"] if i["type"] == "function_call")
    assert fc["id"] == added[0]["item"]["id"]
    assert fc["call_id"] == added[0]["item"]["call_id"]
    assert done[0]["item_id"] == fc["id"]
    assert json.loads(fc["arguments"]) == {"location": "Paris"}


def test_text_mode_streams_prose_with_tools_attached():
    session = FakeSession(chunks=("Hello, ", "world!"))
    client = TestClient(create_app(session, model="m", chat_contract=QWEN3_CONTRACT, chat_templates=DUMMY_CHAT_TEMPLATES))
    resp = client.post("/v1/responses", json={"model": "m", "input": "hi", "stream": True, "tools": TOOLS})
    assert resp.status_code == 200
    events = iter_sse(resp.text)
    assert_valid_stream(events)
    deltas = [p["delta"] for ev, p in events if ev == "response.output_text.delta"]
    assert deltas == ["Hello, ", "world!"]
    assert not any(ev == "response.function_call_arguments.delta" for ev, _ in events)
    completed = next(p["response"] for ev, p in events if ev == "response.completed")
    assert completed["status"] == "completed"
    assert completed["output_text"] == "Hello, world!"


def test_tagged_tool_call_streams_live():
    session = FakeSession(
        chunks=(
            "<tool_call>",
            '{"name":"get_weather",',
            '"arguments":{"location":"Paris"}}',
            "</tool_call>",
        )
    )
    client = TestClient(create_app(session, model="m", chat_contract=QWEN3_CONTRACT, chat_templates=NATIVE_TOOL_TEMPLATES))
    resp = client.post("/v1/responses", json={"model": "m", "input": "hi", "stream": True, "tools": TOOLS})
    events = iter_sse(resp.text)
    assert_valid_stream(events)
    deltas = [p for ev, p in events if ev == "response.function_call_arguments.delta"]
    assert len(deltas) == 4
    completed = next(p["response"] for ev, p in events if ev == "response.completed")
    fc = next(i for i in completed["output"] if i["type"] == "function_call")
    assert json.loads(fc["arguments"]) == {"location": "Paris"}


def test_tool_mode_malformed_still_fails_at_done():
    session = FakeSession(chunks=('{"name":"get_weather",', '"arguments":'))
    client = TestClient(create_app(session, model="m", chat_contract=QWEN3_CONTRACT, chat_templates=DUMMY_CHAT_TEMPLATES))
    resp = client.post("/v1/responses", json={"model": "m", "input": "hi", "stream": True, "tools": TOOLS})
    events = iter_sse(resp.text)
    assert_valid_stream(events)
    # Raw chunks streamed (visibility), but the terminal still reports failure.
    assert any(ev == "response.function_call_arguments.delta" for ev, _ in events)
    assert any(ev == "response.failed" for ev, _ in events)
    assert not any(ev == "response.completed" for ev, _ in events)


def test_non_json_prose_starting_with_brace_keeps_terminal():
    # False-positive shaped prose: streams as tool deltas, terminal follows
    # the same done-parse as the old buffered flow (failed, not completed).
    session = FakeSession(chunks=("{not a tool}",))
    client = TestClient(create_app(session, model="m", chat_contract=QWEN3_CONTRACT, chat_templates=DUMMY_CHAT_TEMPLATES))
    resp = client.post("/v1/responses", json={"model": "m", "input": "hi", "stream": True, "tools": TOOLS})
    events = iter_sse(resp.text)
    assert_valid_stream(events)
    assert any(ev == "response.failed" for ev, _ in events)
