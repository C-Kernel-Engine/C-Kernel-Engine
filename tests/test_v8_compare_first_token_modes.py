#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np


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
