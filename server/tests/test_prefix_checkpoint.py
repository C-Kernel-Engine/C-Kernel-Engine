"""Exact-token and lifecycle controls for the opt-in prefix cache."""

from server.prefix_checkpoint import ExactPrefixCheckpointCache
from server.session_v8 import CK_SESSION_REQUEST_RAW_PROMPT
from server.live import create_app, _OwnedStreamingResponse
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect
import asyncio
import pytest


class FakeSession:
    context_length = 512

    def __init__(self):
        self.tokens = {}
        self.state = ()
        self.prefills = []
        self.restores = []
        self.snapshots = {}

    def prefill_contract(self):
        return 64, 64, 1

    def checkpoint_requirements(self):
        return 128, 64

    def encode_ids(self, text):
        return self.tokens.get(text, [ord(c) for c in text])

    def decode_ids(self, ids):
        return bytes(ids)

    def prefill_ids(self, ids, *, reserved_tokens):
        self.prefills.append((tuple(ids), reserved_tokens))
        self.state = tuple(ids)

    def checkpoint_save(self, snapshot):
        self.snapshots[snapshot.address] = self.state

    def checkpoint_restore(self, snapshot):
        state = self.snapshots[snapshot.address]
        self.restores.append(state)
        self.state = state

    def generate(self, _system, prompt, *, max_tokens, temperature, top_p,
                 on_token, flags=0, prompt_ids=None, **_kwargs):
        on_token(1, "ok")
        return {"prompt_tokens": len(prompt_ids) if prompt_ids is not None
                else len(self.encode_ids(prompt)), "generated_tokens": 1,
                "prefill_time_ms": 1.0, "decode_time_ms": 1.0,
                "stop_reason": 2}

    def cancel(self):
        pass


def test_exact_prefix_hit_and_changed_prefix_fallback():
    session = FakeSession()
    cache = ExactPrefixCheckpointCache(session, max_bytes=192, identity="loaded-instance")
    first = cache.prepare("a" * 80, output_tokens=4, flags=CK_SESSION_REQUEST_RAW_PROMPT)
    assert first.reused_tokens == 0
    assert first.suffix_tokens == 80
    assert first.prompt_ids == (97,) * 16
    assert session.prefills == [((97,) * 64, 20)]
    assert cache.bytes_used == 192  # in-flight reservation counts against the budget
    cache.settle(first, success=True)
    assert cache.bytes_used == 192
    hit = cache.prepare("a" * 80 + "b" * 8, output_tokens=4,
                        flags=CK_SESSION_REQUEST_RAW_PROMPT)
    assert hit.reused_tokens == 64
    assert hit.suffix_tokens == 24
    assert session.restores == [(97,) * 64]
    cache.settle(hit, success=True)
    changed = cache.prepare("c" + "a" * 79, output_tokens=4,
                            flags=CK_SESSION_REQUEST_RAW_PROMPT)
    assert changed.reused_tokens == 0
    assert len(session.restores) == 1


def test_exact_hit_never_samples_missing_logits_and_cancel_discards_candidate():
    session = FakeSession()
    cache = ExactPrefixCheckpointCache(session, max_bytes=192, identity="loaded-instance")
    first = cache.prepare("x" * 66, output_tokens=1, flags=CK_SESSION_REQUEST_RAW_PROMPT)
    cache.settle(first, success=True)
    # A prompt exactly equal to the saved 64-ID prefix has no suffix logits.
    exact = cache.prepare("x" * 64, output_tokens=1, flags=CK_SESSION_REQUEST_RAW_PROMPT)
    assert exact.reused_tokens == 0
    assert not session.restores
    cache.settle(exact, success=False)
    assert cache.bytes_used == 0


def test_budget_and_special_token_suffix_uses_exact_ids():
    session = FakeSession()
    try:
        ExactPrefixCheckpointCache(session, max_bytes=127, identity="instance")
    except ValueError as exc:
        assert "budget" in str(exc)
    else:
        assert False, "insufficient cache budget accepted"
    cache = ExactPrefixCheckpointCache(session, max_bytes=192, identity="instance")
    session.tokens["changed"] = [255] * 80
    miss = cache.prepare("changed", output_tokens=2, flags=CK_SESSION_REQUEST_RAW_PROMPT)
    assert miss.prompt == ""
    assert miss.reused_tokens == 0
    assert miss.prompt_ids == (255,) * 16
    assert session.prefills
    cache.settle(miss, success=True)
    session.tokens["changed2"] = [255] * 80 + [1] * 8
    hit = cache.prepare("changed2", output_tokens=2, flags=CK_SESSION_REQUEST_RAW_PROMPT)
    assert hit.reused_tokens == 64
    assert hit.prompt_ids == (255,) * 16 + (1,) * 8


def test_returned_but_unprocessed_sample_is_not_in_checkpoint_state():
    session = FakeSession()
    session.tokens["first"] = [10] * 80
    session.tokens["next"] = [10] * 80 + [777]
    cache = ExactPrefixCheckpointCache(session, max_bytes=192, identity="instance")
    first = cache.prepare("first", output_tokens=2, flags=CK_SESSION_REQUEST_RAW_PROMPT)
    assert first.candidate is not None
    assert first.candidate[0] == (10,) * 64
    # The model may return token 777 before it is processed into persistent state.
    cache.settle(first, success=True)
    next_turn = cache.prepare("next", output_tokens=2, flags=CK_SESSION_REQUEST_RAW_PROMPT)
    assert next_turn.reused_tokens == 64
    assert next_turn.prompt_ids == (10,) * 16 + (777,)
    assert session.restores == [(10,) * 64]


def test_bounded_checkpoint_tail_survives_revised_harness_prompt_item():
    session = FakeSession()
    session.context_length = 4096
    session.tokens["initial"] = [10] * 2000
    session.tokens["continued"] = [10] * 1600 + [20] * 420
    cache = ExactPrefixCheckpointCache(session, max_bytes=192, identity="instance")
    first = cache.prepare("initial", output_tokens=4, flags=CK_SESSION_REQUEST_RAW_PROMPT)
    assert first.candidate is not None
    assert len(first.candidate[0]) == 1472
    cache.settle(first, success=True)
    next_turn = cache.prepare("continued", output_tokens=4,
                              flags=CK_SESSION_REQUEST_RAW_PROMPT)
    assert next_turn.reused_tokens == 1472
    assert next_turn.prompt_ids == tuple([10] * 128 + [20] * 420)


def test_a_b_a_replacement_never_restores_wrong_snapshot():
    session = FakeSession()
    cache = ExactPrefixCheckpointCache(session, max_bytes=192, identity="instance")
    a = cache.prepare("a" * 80, output_tokens=1, flags=CK_SESSION_REQUEST_RAW_PROMPT)
    cache.settle(a, success=True)
    b = cache.prepare("b" * 80, output_tokens=1, flags=CK_SESSION_REQUEST_RAW_PROMPT)
    assert b.reused_tokens == 0
    cache.settle(b, success=True)
    again = cache.prepare("a" * 88, output_tokens=1, flags=CK_SESSION_REQUEST_RAW_PROMPT)
    assert again.reused_tokens == 0
    assert not session.restores


def test_two_owners_share_a_global_budget_and_evict_oldest():
    session = FakeSession()
    cache = ExactPrefixCheckpointCache(session, max_bytes=384, identity="instance")
    for owner, char in (("A", "a"), ("B", "b")):
        prepared = cache.prepare(char * 80, output_tokens=1,
                                 flags=CK_SESSION_REQUEST_RAW_PROMPT,
                                 owner_key=owner)
        cache.settle(prepared, success=True)
    assert cache.bytes_used == 384
    hit_a = cache.prepare("a" * 88, output_tokens=1,
                          flags=CK_SESSION_REQUEST_RAW_PROMPT, owner_key="A")
    assert hit_a.reused_tokens == 64
    cache.settle(hit_a, success=True)
    newest = cache.prepare("c" * 80, output_tokens=1,
                           flags=CK_SESSION_REQUEST_RAW_PROMPT, owner_key="C")
    cache.settle(newest, success=True)
    assert cache.bytes_used == 384
    miss_b = cache.prepare("b" * 88, output_tokens=1,
                           flags=CK_SESSION_REQUEST_RAW_PROMPT, owner_key="B")
    assert miss_b.reused_tokens == 0


def test_replacement_reserves_before_allocating_at_one_snapshot_budget(monkeypatch):
    import server.prefix_checkpoint as module
    original = module.SequenceCheckpointV8
    live = 0
    peak = 0

    class TrackedSnapshot(original):
        def __init__(self, *args):
            nonlocal live, peak
            super().__init__(*args)
            live += 1
            peak = max(peak, live)

        def __del__(self):
            nonlocal live
            live -= 1

    monkeypatch.setattr(module, "SequenceCheckpointV8", TrackedSnapshot)
    session = FakeSession()
    cache = ExactPrefixCheckpointCache(session, max_bytes=192, identity="instance")
    for owner, text in (("A", "a" * 80), ("A", "b" * 80),
                        ("B", "c" * 80)):
        prepared = cache.prepare(text, output_tokens=1,
                                 flags=CK_SESSION_REQUEST_RAW_PROMPT,
                                 owner_key=owner)
        assert cache.bytes_used <= cache.max_bytes
        cache.settle(prepared, success=True)
        del prepared
        assert cache.bytes_used == 192
    assert peak == 1


def test_cancelled_preparation_releases_reservation():
    session = FakeSession()
    cache = ExactPrefixCheckpointCache(session, max_bytes=192, identity="instance")
    def cancel_after_prefill(ids, *, reserved_tokens):
        session.state = tuple(ids)
        cancelled[0] = True
    cancelled = [False]
    session.prefill_ids = cancel_after_prefill
    try:
        cache.prepare("a" * 80, output_tokens=1,
                      flags=CK_SESSION_REQUEST_RAW_PROMPT,
                      cancelled=lambda: cancelled[0])
    except Exception as exc:
        assert "cancelled" in str(exc)
    else:
        assert False, "cancelled prefix preparation succeeded"
    assert cache.bytes_used == 0


def test_late_disconnect_cannot_evict_newer_owner_checkpoint():
    session = FakeSession()
    cache = ExactPrefixCheckpointCache(session, max_bytes=192, identity="instance")
    old = cache.prepare("a" * 80, output_tokens=1,
                        flags=CK_SESSION_REQUEST_RAW_PROMPT,
                        owner_key="A", request_id="old")
    cache.settle(old, success=True)
    new = cache.prepare("b" * 80, output_tokens=1,
                        flags=CK_SESSION_REQUEST_RAW_PROMPT,
                        owner_key="A", request_id="new")
    cache.settle(new, success=True)
    cache.discard("A", request_id="old")
    assert cache.bytes_used == 192
    cache.discard("A", request_id="new")
    assert cache.bytes_used == 0


def test_http_reuses_exact_prefix_and_reports_actual_input_usage():
    session = FakeSession()
    app = create_app(session, model="test", context_length=512,
                     chat_template="", allow_untemplated=True,
                     flags=CK_SESSION_REQUEST_RAW_PROMPT,
                     prefix_cache_bytes=192)
    with TestClient(app) as client:
        first = client.post("/v1/responses", json={"model": "test", "input": "a" * 80,
                                                    "max_output_tokens": 1})
        second = client.post("/v1/responses", json={"model": "test", "input": "a" * 80 + "b" * 8,
                                                     "max_output_tokens": 1})
    assert first.status_code == second.status_code == 200
    first_data, second_data = first.json(), second.json()
    assert first_data["performance"]["prefix_cache"]["reused_tokens"] == 0
    assert first_data["performance"]["computed_prefill_tokens"] == 80
    assert second_data["performance"]["prefix_cache"]["reused_tokens"] == 64
    assert second_data["performance"]["computed_prefill_tokens"] == 24
    assert second_data["usage"]["input_tokens"] == 88
    assert second_data["usage"]["input_tokens_details"]["cached_tokens"] == 64
    assert second_data["output_text"] == "ok"


def test_http_cancelled_native_stop_does_not_publish_checkpoint():
    class CancelledSession(FakeSession):
        def generate(self, *args, **kwargs):
            result = super().generate(*args, **kwargs)
            result["stop_reason"] = 3
            return result

    session = CancelledSession()
    app = create_app(session, model="test", context_length=512,
                     chat_template="", allow_untemplated=True,
                     flags=CK_SESSION_REQUEST_RAW_PROMPT,
                     prefix_cache_bytes=192)
    with TestClient(app) as client:
        response = client.post("/v1/responses", json={
            "model": "test", "input": "a" * 80, "max_output_tokens": 1})
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert app.state.prefix_cache.bytes_used == 0


def test_incomplete_stream_delivery_discards_after_disconnect():
    order = []
    class Lease:
        def disconnect(self):
            order.append("disconnect")
    async def chunks():
        yield b"data: response.created\n\n"
    async def send(message):
        if message["type"] == "http.response.body":
            raise OSError("client disconnected")
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}
    response = _OwnedStreamingResponse(
        chunks(), lease=Lease(),
        on_incomplete_delivery=lambda: order.append("discard"))
    scope = {"type": "http", "method": "GET", "path": "/", "headers": [],
             "asgi": {"version": "3.0", "spec_version": "2.4"}}
    with pytest.raises(ClientDisconnect):
        asyncio.run(response(scope, receive, send))
    assert order == ["disconnect", "discard"]
