"""Keep request handling separate from asset-backed Drama arithmetic tests."""

from threading import Lock
from types import SimpleNamespace

import numpy as np
import pytest

from version.v8.tts.drama3_native_harness import Cancelled, Session, WIDTH, speech_token


def test_semantic_retry_and_stop():
    logits = np.array([3.5, 20, 8, 4, 3], dtype=np.float32)
    assert speech_token(logits, 2, 4, 0, []) == 2
    assert speech_token(logits, 2, 4, 0, [2]) == 3
    assert speech_token(logits, 2, 4, 0, [2, 3]) == 0
    logits[3] = np.nan
    with pytest.raises(RuntimeError, match="nonfinite"):
        speech_token(logits, 2, 4, 0, [])


def test_request_rejection_cancel_and_recovery():
    session = Session.__new__(Session)
    session.lock = Lock()
    session.closed = False
    session.tower_capacity = 64
    session.rotary = np.empty((64, 32, 2))
    session.begin, session.end, session.eos = 1, 4, 0
    session.tokenizer = SimpleNamespace(encode=lambda text: SimpleNamespace(ids=[1] * len(text.split())))
    calls = []
    logits = np.zeros(5, dtype=np.float32)

    def prefill(ids):
        calls.append(("prefill", len(ids)))
        logits[:] = [0, 4, 3, 2, 1]
        return np.zeros((len(ids), WIDTH), dtype=np.float32)

    def frame(hidden, unseen, token, position):
        calls.append(("frame", position))
        return np.full(10, token, dtype=np.int32)

    def decode(embedding):
        calls.append(("feedback",))
        logits[0] = 5
        return np.zeros((1, WIDTH), dtype=np.float32)

    def waveform(frames, check_cancel):
        calls.append(("codec", len(frames)))
        check_cancel()
        return np.zeros(len(frames) * 2048, dtype=np.float32), 0.1

    session.backbone = SimpleNamespace(prefill=prefill, logits=logits, decode=decode)
    session.prefill_tower = lambda rows: calls.append(("tower", len(rows)))
    session.frame = frame
    session.prepared_row = lambda *_: (None, None)
    session.waveform = waveform

    with pytest.raises(ValueError, match="prompt-control"):
        session.synthesize("<|voice|>", 2)
    with pytest.raises(ValueError, match="capacity"):
        session.synthesize("Hello", 64)
    assert not calls

    samples, a = session.synthesize("Hello", 2)
    assert a["finish_reason"] == "eos" and len(samples) == 2048
    _, b = session.synthesize("A longer sentence", 1)
    assert b["finish_reason"] == "length" and a["text_tokens"] < b["text_tokens"]

    calls.clear()
    with pytest.raises(Cancelled):
        session.synthesize("Stop this", 2, stop_requested=lambda: ("feedback",) in calls)
    assert not any(call[0] == "codec" for call in calls)
    repeated, again = session.synthesize("Hello", 2)
    np.testing.assert_array_equal(samples, repeated)
    assert a["code_frames"] == again["code_frames"]


def test_pinned_requests_reuse_generated_backbone_and_tower(tmp_path):
    import json
    import os
    from pathlib import Path

    configured = os.environ.get("CK_DRAMA3_NATIVE_ACCEPTANCE_JSON")
    if not configured:
        pytest.skip("configure pinned Drama checkpoint, generated backbone and BUMP directories")
    assets = json.loads(Path(configured).read_text())
    checkpoint, bundle, weights = (Path(assets[key]) for key in ("checkpoint", "bundle", "weights"))
    with Session(checkpoint, bundle, weights, tmp_path / "build") as session:
        first, a = session.synthesize("Hello, CKE.", 2)
        other, b = session.synthesize("Test complete.", 2)
        assert a["text_tokens"] != b["text_tokens"]
        assert a["finish_reason"] == b["finish_reason"] == "length"
        assert len(first) == len(other) == 4096
        assert a["code_frames"] == [
            [2048, 69, 149, 225, 808, 115, 427, 402, 469, 838],
            [933, 740, 216, 413, 571, 785, 749, 579, 2, 213],
        ]
        assert b["code_frames"] != a["code_frames"]
        with pytest.raises(ValueError, match="capacity"):
            session.synthesize("Hello, CKE.", session.tower_capacity + 1)
        polls = iter((False, False, False, True))
        with pytest.raises(Cancelled):
            session.synthesize("Stop this.", 3, stop_requested=lambda: next(polls, True))
        repeated, again = session.synthesize("Hello, CKE.", 2)
        np.testing.assert_array_equal(first, repeated)
        assert a["code_frames"] == again["code_frames"]
