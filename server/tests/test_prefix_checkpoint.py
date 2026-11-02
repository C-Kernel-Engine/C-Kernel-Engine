"""Exact-token and lifecycle controls for the opt-in prefix cache."""

from server.prefix_checkpoint import ExactPrefixCheckpointCache
from server.session_v8 import CK_SESSION_REQUEST_RAW_PROMPT
from server.live import create_app
from fastapi.testclient import TestClient


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
    assert first.suffix_tokens == 16
    assert first.prompt_ids == (97,) * 16
    assert session.prefills == [((97,) * 64, 20)]
    assert cache.bytes_used == 0
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
    assert second_data["performance"]["prefix_cache"]["reused_tokens"] == 64
    assert second_data["usage"]["input_tokens"] == 88
    assert second_data["usage"]["input_tokens_details"]["cached_tokens"] == 64
    assert second_data["output_text"] == "ok"
