"""Model-free regression tests for the localhost server E2E harness."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "test_serve_localhost_e2e.py"
SPEC = importlib.util.spec_from_file_location("test_serve_localhost_e2e_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _events(text: str = "Hello") -> list[tuple[str, dict]]:
    response_id = "resp_test"
    return [
        (
            "response.created",
            {
                "type": "response.created",
                "response": {"id": response_id},
                "sequence_number": 0,
            },
        ),
        (
            "response.in_progress",
            {
                "type": "response.in_progress",
                "response": {"id": response_id},
                "sequence_number": 1,
            },
        ),
        (
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": "msg_test",
                "output_index": 0,
                "content_index": 0,
                "delta": text,
                "logprobs": None,
                "sequence_number": 2,
            },
        ),
        (
            "response.completed",
            {
                "type": "response.completed",
                "response": {"id": response_id, "output_text": text},
                "sequence_number": 3,
            },
        ),
    ]


def test_stream_contract_accepts_consistent_events() -> None:
    response_id, emitted, terminal, response = MODULE._validate_stream(_events())
    assert (response_id, emitted, terminal) == (
        "resp_test",
        "Hello",
        "response.completed",
    )
    assert response["output_text"] == emitted


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda events: events[2][1].update(type="response.created"), "type mismatch"),
        (lambda events: events[2][1].update(delta=7), "schema validation"),
        (lambda events: events[2][1].update(sequence_number=10), "non-contiguous"),
        (
            lambda events: events[-1][1]["response"].update(id="resp_other"),
            "inconsistent response ids",
        ),
        (
            lambda events: events[-1][1]["response"].update(output_text="different"),
            "do not match",
        ),
    ],
)
def test_stream_contract_rejects_malformed_evidence(mutate, message: str) -> None:
    events = copy.deepcopy(_events())
    mutate(events)
    with pytest.raises(RuntimeError, match=message):
        MODULE._validate_stream(events)


def test_stalled_stream_preserves_partial_transcript() -> None:
    class Clock:
        value = 0.0

        def now(self) -> float:
            return self.value

    class SlowResponse:
        def __init__(self, clock: Clock) -> None:
            self.clock = clock
            self.lines = [b"event: response.created\n", b"data: {}\n"]

        def readline(self) -> bytes:
            self.clock.value += 0.6
            return self.lines.pop(0) if self.lines else b""

    clock = Clock()
    partial: list[str] = []
    with pytest.raises(TimeoutError, match="overall deadline"):
        MODULE._read_response_incrementally(
            SlowResponse(clock), 1.0, now=clock.now, on_chunk=partial.append
        )
    assert partial == ["event: response.created\n", "data: {}\n"]


def test_cleanup_failure_cannot_preserve_success() -> None:
    failure = MODULE._combine_failure(None, RuntimeError("port remains open"))
    assert "cleanup failed" in str(failure)
    assert "port remains open" in str(failure)


def test_stored_response_must_match_stream_and_terminal() -> None:
    doc = {
        "id": "resp_test",
        "created_at": 1,
        "status": "completed",
        "model": "ck-v8",
        "output": [],
        "output_text": "different",
    }
    with pytest.raises(RuntimeError, match="stored response output_text"):
        MODULE._validate_final_response(
            doc,
            "resp_test",
            "Hello",
            {"id": "resp_test", "output_text": "Hello"},
        )
