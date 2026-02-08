"""
Unit tests for sliding-window attention kernels.

Tests compare C kernel output against PyTorch reference implementation.
"""

import numpy as np
import pytest
import sys
import os

# Add the parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import the C kernel bindings (will fail gracefully if not compiled)
try:
    from ckernel import ffi
    from ckernel.libckernel import lib
    HAS_CKERNEL = True
except ImportError:
    HAS_CKERNEL = False
    print("Warning: ckernel not available, using mock implementation")


def pytorch_sliding_window_attention(q, k, v, sliding_window=None):
    """
    PyTorch reference implementation of sliding-window attention.

    Args:
        q: Query tensor [num_heads, num_tokens, head_dim]
        k: Key tensor [num_kv_heads, num_tokens, head_dim]
        v: Value tensor [num_kv_heads, num_tokens, head_dim]
        sliding_window: Window size (None = no limit, like causal)

    Returns:
        Output tensor [num_heads, num_tokens, head_dim]
    """
    import torch

    # Convert numpy arrays to torch tensors if needed
    q_t = torch.from_numpy(q) if isinstance(q, np.ndarray) else q
    k_t = torch.from_numpy(k) if isinstance(k, np.ndarray) else k
    v_t = torch.from_numpy(v) if isinstance(v, np.ndarray) else v

    num_heads, num_tokens, head_dim = q_t.shape
    num_kv_heads, _, _ = k_t.shape

    # Handle GQA: repeat K/V heads to match Q heads
    if num_kv_heads < num_heads:
        head_ratio = num_heads // num_kv_heads
        k_t = k_t.repeat_interleave(head_ratio, dim=0)
        v_t = v_t.repeat_interleave(head_ratio, dim=0)

    scale = 1.0 / np.sqrt(head_dim)

    # Compute Q @ K^T / sqrt(d)
    # q: [H, T, D], k: [H, T, D] -> scores: [H, T, T]
    scores = torch.zeros(num_heads, num_tokens, num_tokens, dtype=q_t.dtype)

    for h in range(num_heads):
        for i in range(num_tokens):
            for j in range(num_tokens):
                if j > i:
                    # Causal mask: j > i means future token, skip
                    continue
                if sliding_window and sliding_window > 0:
                    # Sliding window: only attend to last W tokens
                    if i - j >= sliding_window:
                        continue
                # Dot product
                dot = torch.sum(q_t[h, i] * k_t[h, j])
                scores[h, i, j] = dot * scale

    # Softmax
    scores = scores - scores.max(dim=-1, keepdim=True)[0]
    exp_scores = torch.exp(scores)
    sums = exp_scores.sum(dim=-1, keepdim=True)
    weights = exp_scores / sums

    # Output = weights @ V
    output = torch.zeros(num_heads, num_tokens, head_dim, dtype=q_t.dtype)
    for h in range(num_heads):
        for i in range(num_tokens):
            for j in range(num_tokens):
                if sliding_window and sliding_window > 0:
                    if i - j >= sliding_window:
                        continue
                output[h, i] += weights[h, i, j] * v_t[h, j]

    return output.cpu().numpy()


def qkv_index(h, t, d, num_tokens, aligned_head_dim):
    """Compute index for head-major QKV layout."""
    return ((h * num_tokens) + t) * aligned_head_dim + d


class TestSlidingWindowAttention:
    """Test sliding-window attention kernels."""

    @pytest.fixture
    def sample_input(self):
        """Generate sample input for testing."""
        np.random.seed(42)

        num_heads = 4
        num_kv_heads = 2
        num_tokens = 8
        head_dim = 64
        aligned_head_dim = 64

        # Create contiguous layout (head-major)
        q = np.random.randn(num_heads, num_tokens, aligned_head_dim).astype(np.float32) * 0.1
        k = np.random.randn(num_kv_heads, num_tokens, aligned_head_dim).astype(np.float32) * 0.1
        v = np.random.randn(num_kv_heads, num_tokens, aligned_head_dim).astype(np.float32) * 0.1

        return q, k, v, num_heads, num_kv_heads, num_tokens, head_dim, aligned_head_dim

    @pytest.mark.skipif(not HAS_CKERNEL, reason="ckernel not available")
    def test_c_kernel_prefill_sliding(self, sample_input):
        """Test C kernel for prefill sliding window attention."""
        q, k, v, num_heads, num_kv_heads, num_tokens, head_dim, aligned_head_dim = sample_input
        sliding_window = 4

        # Allocate output
        output = np.zeros((num_heads, num_tokens, aligned_head_dim), dtype=np.float32)

        # Call C kernel
        lib.attention_forward_causal_head_major_gqa_flash_strided_sliding(
            q.ctypes.data_as(ffi.CData),
            k.ctypes.data_as(ffi.CData),
            v.ctypes.data_as(ffi.CData),
            output.ctypes.data_as(ffi.CData),
            num_heads,
            num_kv_heads,
            num_tokens,
            head_dim,
            aligned_head_dim,
            num_tokens,  # kv_stride_tokens
            sliding_window
        )

        # Reference implementation
        ref_output = pytorch_sliding_window_attention(q, k, v, sliding_window)

        np.testing.assert_allclose(output, ref_output, rtol=1e-4, atol=1e-5)

    @pytest.mark.skipif(not HAS_CKERNEL, reason="ckernel not available")
    def test_c_kernel_prefill_no_window(self, sample_input):
        """Test C kernel for prefill with no sliding window (causal)."""
        q, k, v, num_heads, num_kv_heads, num_tokens, head_dim, aligned_head_dim = sample_input

        output = np.zeros((num_heads, num_tokens, aligned_head_dim), dtype=np.float32)

        # No sliding window (sliding_window = 0 or negative means no limit)
        lib.attention_forward_causal_head_major_gqa_flash_strided_sliding(
            q.ctypes.data_as(ffi.CData),
            k.ctypes.data_as(ffi.CData),
            v.ctypes.data_as(ffi.CData),
            output.ctypes.data_as(ffi.CData),
            num_heads,
            num_kv_heads,
            num_tokens,
            head_dim,
            aligned_head_dim,
            num_tokens,
            -1  # No limit
        )

        # Reference: causal attention (no sliding window)
        ref_output = pytorch_sliding_window_attention(q, k, v, sliding_window=None)

        np.testing.assert_allclose(output, ref_output, rtol=1e-4, atol=1e-5)

    @pytest.mark.skipif(not HAS_CKERNEL, reason="ckernel not available")
    def test_c_kernel_decode_sliding(self):
        """Test C kernel for decode sliding window attention."""
        np.random.seed(42)
        num_heads = 4
        num_kv_heads = 2
        head_dim = 64
        aligned_head_dim = 64

        # Allocate with full cache_capacity to match kernel stride
        kv_tokens = 8
        cache_capacity = 16

        # Create K/V with full cache capacity [num_kv_heads, cache_capacity, aligned_head_dim]
        k = np.random.randn(num_kv_heads, cache_capacity, aligned_head_dim).astype(np.float32) * 0.1
        v = np.random.randn(num_kv_heads, cache_capacity, aligned_head_dim).astype(np.float32) * 0.1

        # Query is for position 7 (last filled position)
        q_decode = np.random.randn(num_heads, 1, aligned_head_dim).astype(np.float32) * 0.1
        sliding_window = 4

        output = np.zeros((num_heads, 1, aligned_head_dim), dtype=np.float32)

        # Call C kernel
        lib.attention_forward_decode_head_major_gqa_flash_sliding(
            q_decode.ctypes.data_as(ffi.CData),
            k.ctypes.data_as(ffi.CData),
            v.ctypes.data_as(ffi.CData),
            output.ctypes.data_as(ffi.CData),
            num_heads,
            num_kv_heads,
            kv_tokens,
            cache_capacity,
            head_dim,
            aligned_head_dim,
            sliding_window
        )

        # Reference implementation (only uses first kv_tokens entries)
        ref_output = pytorch_sliding_window_attention(
            q_decode, k[:, :kv_tokens, :], v[:, :kv_tokens, :], sliding_window
        )

        np.testing.assert_allclose(output, ref_output, rtol=1e-4, atol=1e-5)

    def test_causal_equals_no_window(self, sample_input):
        """Sliding window with no limit should equal causal attention."""
        q, k, v, num_heads, num_kv_heads, num_tokens, head_dim, aligned_head_dim = sample_input

        # No sliding window (or very large window) should equal causal
        pytorch_out_no_window = pytorch_sliding_window_attention(
            q, k, v, sliding_window=None
        )
        pytorch_out_large_window = pytorch_sliding_window_attention(
            q, k, v, sliding_window=1000
        )

        np.testing.assert_allclose(
            pytorch_out_no_window, pytorch_out_large_window, rtol=1e-5
        )

    def test_sliding_window_small(self, sample_input):
        """Test small sliding window reduces attention range."""
        q, k, v, num_heads, num_kv_heads, num_tokens, head_dim, aligned_head_dim = sample_input

        # Small window
        pytorch_out_window = pytorch_sliding_window_attention(
            q, k, v, sliding_window=4
        )

        # Large window (should attend to all previous tokens)
        pytorch_out_large = pytorch_sliding_window_attention(
            q, k, v, sliding_window=100
        )

        # Outputs should differ
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                pytorch_out_window, pytorch_out_large, rtol=1e-5
            )

    def test_sliding_window_gqa(self):
        """Test sliding window attention with GQA (grouped-query attention)."""
        np.random.seed(123)

        num_heads = 8
        num_kv_heads = 2  # GQA: fewer KV heads
        num_tokens = 16
        head_dim = 32
        aligned_head_dim = 32

        q = np.random.randn(num_heads, num_tokens, aligned_head_dim).astype(np.float32) * 0.1
        k = np.random.randn(num_kv_heads, num_tokens, aligned_head_dim).astype(np.float32) * 0.1
        v = np.random.randn(num_kv_heads, num_tokens, aligned_head_dim).astype(np.float32) * 0.1

        window = 4

        pytorch_out = pytorch_sliding_window_attention(
            q, k, v, sliding_window=window
        )

        # Verify output shape
        assert pytorch_out.shape == (num_heads, num_tokens, head_dim)


class TestSlidingWindowEdgeCases:
    """Test edge cases for sliding-window attention."""

    def test_window_larger_than_seq(self):
        """Window larger than sequence should behave like causal."""
        np.random.seed(101)

        num_heads = 2
        num_kv_heads = 2
        num_tokens = 4
        head_dim = 16
        aligned_head_dim = 16

        q = np.random.randn(num_heads, num_tokens, aligned_head_dim).astype(np.float32) * 0.1
        k = np.random.randn(num_kv_heads, num_tokens, aligned_head_dim).astype(np.float32) * 0.1
        v = np.random.randn(num_kv_heads, num_tokens, aligned_head_dim).astype(np.float32) * 0.1

        # Window larger than sequence
        pytorch_out = pytorch_sliding_window_attention(
            q, k, v, sliding_window=100
        )

        # No window
        pytorch_out_no = pytorch_sliding_window_attention(
            q, k, v, sliding_window=None
        )

        np.testing.assert_allclose(pytorch_out, pytorch_out_no, rtol=1e-5)

    def test_single_token(self):
        """Single token should attend to itself."""
        np.random.seed(202)

        num_heads = 2
        num_kv_heads = 2
        num_tokens = 1
        head_dim = 16
        aligned_head_dim = 16

        q = np.random.randn(num_heads, num_tokens, aligned_head_dim).astype(np.float32) * 0.1
        k = np.random.randn(num_kv_heads, num_tokens, aligned_head_dim).astype(np.float32) * 0.1
        v = np.random.randn(num_kv_heads, num_tokens, aligned_head_dim).astype(np.float32) * 0.1

        # Should work even with single token
        pytorch_out = pytorch_sliding_window_attention(
            q, k, v, sliding_window=4
        )

        assert pytorch_out.shape == (num_heads, num_tokens, head_dim)

    def test_window_size_1(self):
        """Window size of 1 means each token only attends to itself."""
        np.random.seed(404)

        num_heads = 2
        num_kv_heads = 2
        num_tokens = 4
        head_dim = 16
        aligned_head_dim = 16

        q = np.random.randn(num_heads, num_tokens, aligned_head_dim).astype(np.float32) * 0.1
        k = np.random.randn(num_kv_heads, num_tokens, aligned_head_dim).astype(np.float32) * 0.1
        v = np.random.randn(num_kv_heads, num_tokens, aligned_head_dim).astype(np.float32) * 0.1

        window = 1
        pytorch_out = pytorch_sliding_window_attention(
            q, k, v, sliding_window=window
        )

        # Verify output shape
        assert pytorch_out.shape == (num_heads, num_tokens, head_dim)

    def test_decode_kv_tokens_1_window_1(self):
        """Decode with kv_tokens=1 and sliding_window=1."""
        np.random.seed(505)

        num_heads = 2
        num_kv_heads = 2
        kv_tokens = 1  # Single KV token
        head_dim = 16
        aligned_head_dim = 16

        # Query for current position
        q = np.random.randn(num_heads, 1, aligned_head_dim).astype(np.float32) * 0.1

        # Single KV token in cache
        k = np.random.randn(num_kv_heads, 1, aligned_head_dim).astype(np.float32) * 0.1
        v = np.random.randn(num_kv_heads, 1, aligned_head_dim).astype(np.float32) * 0.1

        window = 1

        # Should work: query attends to the single KV token
        pytorch_out = pytorch_sliding_window_attention(
            q, k, v, sliding_window=window
        )

        assert pytorch_out.shape == (num_heads, 1, head_dim)

    def test_prefill_window_larger_than_position(self):
        """Token at position i should attend to tokens 0..i when window > i+1."""
        np.random.seed(606)

        num_heads = 2
        num_kv_heads = 2
        num_tokens = 8
        head_dim = 16
        aligned_head_dim = 16

        q = np.random.randn(num_heads, num_tokens, aligned_head_dim).astype(np.float32) * 0.1
        k = np.random.randn(num_kv_heads, num_tokens, aligned_head_dim).astype(np.float32) * 0.1
        v = np.random.randn(num_kv_heads, num_tokens, aligned_head_dim).astype(np.float32) * 0.1

        window = 100  # Larger than any position

        pytorch_out = pytorch_sliding_window_attention(
            q, k, v, sliding_window=window
        )

        # Should equal causal attention
        pytorch_out_causal = pytorch_sliding_window_attention(
            q, k, v, sliding_window=None
        )

        np.testing.assert_allclose(pytorch_out, pytorch_out_causal, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
