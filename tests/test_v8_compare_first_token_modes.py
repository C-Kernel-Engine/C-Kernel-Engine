#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
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
