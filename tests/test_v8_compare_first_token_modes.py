#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "compare_first_token_logits_v8.py"
sys.path.insert(0, str(SCRIPT.parent))


def _load_module():
    spec = importlib.util.spec_from_file_location("compare_first_token_modes", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


compare = _load_module()


def test_logit_comparison_rejects_truncated_vocabularies() -> None:
    with pytest.raises(ValueError, match="vocabulary extent mismatch: ck=2 llama=3"):
        compare.compare_logits(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([1.0, 2.0, 100.0], dtype=np.float32),
            2,
        )
    with pytest.raises(ValueError, match="vocabulary extent mismatch: ck=0 llama=0"):
        compare.compare_logits(np.array([]), np.array([]), 1)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_logit_comparison_rejects_nonfinite_inputs(bad: float) -> None:
    good = np.array([1.0, 2.0], dtype=np.float32)
    invalid = np.array([1.0, bad], dtype=np.float32)
    with pytest.raises(ValueError, match="finite values"):
        compare.compare_logits(invalid, good, 2)
    with pytest.raises(ValueError, match="finite values"):
        compare.compare_logits(good, invalid, 2)


def test_logit_comparison_rejects_wrong_axes() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        compare.compare_logits(np.array([[1.0, 2.0]]), np.array([1.0, 2.0]), 2)


def test_logit_comparison_retains_complete_finite_metrics() -> None:
    result = compare.compare_logits(
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([1.0, 2.5, 3.0], dtype=np.float32),
        2,
    )
    assert result["n_compared"] == 3
    assert result["max_abs_diff"] == 0.5
    assert result["top1_match"] is True


def test_auto_llama_mode_mirrors_sequential_contract() -> None:
    assert (
        compare.resolve_llama_decode_mode("auto", "auto", "sequential_decode")
        == "sequential"
    )


def test_auto_llama_mode_mirrors_explicit_ck_mode() -> None:
    assert compare.resolve_llama_decode_mode("auto", "batched", "sequential_decode") == "batched"
    assert compare.resolve_llama_decode_mode("auto", "sequential", "batched") == "sequential"
    assert compare.resolve_llama_decode_mode("auto", "hybrid", "sequential_decode") == "batched"


def test_explicit_llama_mode_is_never_rewritten() -> None:
    assert compare.resolve_llama_decode_mode("batched", "sequential", "sequential_decode") == "batched"
    assert compare.resolve_llama_decode_mode("sequential", "batched", "batched") == "sequential"


def test_logit_hash_is_stable_for_canonical_float32_storage() -> None:
    values = np.array([1.0, -2.5, 3.25], dtype=np.float64)
    expected = compare.logits_sha256(values.astype("<f4"))
    assert compare.logits_sha256(values) == expected


def test_repeatability_reports_first_bitwise_difference() -> None:
    first = np.array([1.0, 2.0], dtype=np.float32)
    same = first.copy()
    changed = np.array([1.0, 2.0001], dtype=np.float32)

    exact = compare.summarize_repeatability([first, same])
    drift = compare.summarize_repeatability([first, same, changed])

    assert exact["exact"] is True
    assert exact["first_different_run"] is None
    assert drift["exact"] is False
    assert drift["first_different_run"] == 2


def test_repeatability_distinguishes_signed_zero_bits() -> None:
    positive_zero = np.array([0.0], dtype=np.float32)
    negative_zero = np.array([-0.0], dtype=np.float32)

    result = compare.summarize_repeatability([positive_zero, negative_zero])

    assert result["exact"] is False
    assert result["first_different_run"] == 1


def test_batched_mode_requires_compiled_prefill_capability() -> None:
    compare.validate_compiled_prefill_capability(
        "batched", compare.CK_MODEL_CAP_MIXED_EMBEDDING_PREFILL
    )

    for capabilities in (None, 0):
        try:
            compare.validate_compiled_prefill_capability("batched", capabilities)
        except RuntimeError as exc:
            assert "batched prefill requested" in str(exc)
        else:
            raise AssertionError("missing compiled prefill capability was accepted")


def test_sequential_mode_does_not_require_compiled_prefill_capability() -> None:
    compare.validate_compiled_prefill_capability("sequential_decode", None)


def test_batched_replay_forces_runtime_schedule_and_restores_environment() -> None:
    name = "CK_V8_FORCE_BATCHED_PREFILL"
    previous = os.environ.pop(name, None)
    try:
        with compare.runtime_prefill_environment("hybrid"):
            assert os.environ[name] == "1"
        assert name not in os.environ

        os.environ[name] = "caller-value"
        with compare.runtime_prefill_environment("batched"):
            assert os.environ[name] == "1"
        assert os.environ[name] == "caller-value"
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def test_sequential_replay_does_not_force_batched_runtime() -> None:
    name = "CK_V8_FORCE_BATCHED_PREFILL"
    previous = os.environ.pop(name, None)
    try:
        with compare.runtime_prefill_environment("sequential_decode"):
            assert name not in os.environ
    finally:
        if previous is not None:
            os.environ[name] = previous


def test_long_oracle_replay_passes_exact_ids_through_a_file() -> None:
    for token_id, length, expect_file in ((12345, 21_000, True),
                                          (1, 42_132, False),
                                          (12345, 3000, False)):
        tokens = [token_id] * length

        def fake_run(command, **_kwargs):
            assert ("--tokens-file" in command) == expect_file
            if expect_file:
                token_file = Path(command[command.index("--tokens-file") + 1])
                actual = compare.parse_tokens_csv(token_file.read_text(encoding="ascii"))
            else:
                actual = compare.parse_tokens_csv(command[command.index("--tokens") + 1])
            assert actual == tokens
            assert command[command.index("--batch-tokens") + 1] == "512"
            np.array([1.0, 2.0], dtype=np.float32).tofile(
                command[command.index("--logits-out") + 1]
            )
            return mock.Mock(returncode=0, stdout=json.dumps({"ok": True, "n_vocab": 2}), stderr="")

        with mock.patch.object(compare, "ensure_llama_helper", return_value=Path("/tmp/helper")), \
             mock.patch.object(compare, "_run", side_effect=fake_run):
            result = compare.run_llama_logits(Path("model.gguf"), tokens, 32768, 2, 4,
                                              batch_tokens=512)
        assert result["logits"].tolist() == [1.0, 2.0]


def test_short_oracle_replay_retains_inline_ids() -> None:
    def fake_run(command, **_kwargs):
        assert command[command.index("--tokens") + 1] == "1,2,3"
        assert "--tokens-file" not in command
        np.array([3.0], dtype=np.float32).tofile(
            command[command.index("--logits-out") + 1]
        )
        return mock.Mock(returncode=0, stdout=json.dumps({"ok": True, "n_vocab": 1}), stderr="")

    with mock.patch.object(compare, "ensure_llama_helper", return_value=Path("/tmp/helper")), \
         mock.patch.object(compare, "_run", side_effect=fake_run):
        result = compare.run_llama_logits(Path("model.gguf"), [1, 2, 3], 256, 1, 1)
    assert result["logits"].tolist() == [3.0]
