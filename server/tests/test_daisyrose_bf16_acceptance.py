"""Tests for the full-vocabulary scorer used by DaisyRose acceptance."""

import numpy as np
import pytest

from server.daisyrose_bf16_acceptance import compare_logits


def test_comparison_checks_entire_vocabulary_at_each_step():
    oracle = np.array([[2, -2, 0], [0, 3, 1]], dtype=np.float32)
    native = oracle.copy()
    native[0, 2] += 0.125
    native[1, 2] += 0.375
    rows = compare_logits(native, oracle, max_abs_limit=0.25, rmse_limit=0.2)
    assert [(row["phase"], row["numeric_pass"]) for row in rows] == [
        ("prefill", True), ("teacher_forced_decode", False),
    ]
    assert [row["native_argmax"] for row in rows] == [0, 1]
    assert [row["oracle_argmax"] for row in rows] == [0, 1]


def test_comparison_rejects_invalid_arrays():
    with pytest.raises(ValueError, match="geometry"):
        compare_logits(np.ones((2, 2)), np.ones((2, 3)), max_abs_limit=1, rmse_limit=1)
    with pytest.raises(ValueError, match="nonfinite"):
        compare_logits(np.array([[np.inf]]), np.zeros((1, 1)), max_abs_limit=1, rmse_limit=1)
