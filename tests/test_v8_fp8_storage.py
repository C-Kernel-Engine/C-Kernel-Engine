"""Block-scaled FP8 checkpoint weights with explicit row and column blocks."""
import numpy as np
import pytest

from version.v8.scripts.fp8_storage_v8 import expand_block_scaled_e4m3fn


def test_expands_distinct_blocks_and_partial_tail():
    weights = np.full((3, 5), 0x38, dtype=np.uint8)
    weights[2, 4] = 0xB8
    scales = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    actual = expand_block_scaled_e4m3fn(weights, scales, row_block=2, column_block=2)
    np.testing.assert_array_equal(actual, [[1, 1, 2, 2, 3], [1, 1, 2, 2, 3], [4, 4, 5, 5, -6]])
    with pytest.raises(ValueError, match="multiplier grid"):
        expand_block_scaled_e4m3fn(weights, scales.T, row_block=2, column_block=2)
    scales[-1, -1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        expand_block_scaled_e4m3fn(weights, scales, row_block=2, column_block=2)
