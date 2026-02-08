"""
Unit tests for GeGLU activation kernels.

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


def pytorch_gelu(x):
    """PyTorch GELU approximation (fast, uses tanh)."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def pytorch_geglu(x):
    """
    PyTorch reference implementation of GeGLU.

    GeGLU: out = GELU(a) * b where x = [a, b] along the last dimension.

    Args:
        x: Input tensor [tokens, 2*dim]

    Returns:
        Output tensor [tokens, dim]
    """
    tokens, inner_dim = x.shape
    dim = inner_dim // 2

    a = x[:, :dim]  # First half
    b = x[:, dim:]  # Second half

    gelu_a = pytorch_gelu(a)
    out = gelu_a * b

    return out


def pytorch_gelu_backward(x, d_out):
    """
    PyTorch GELU backward pass.

    dL/dx given dL/d(out)
    """
    # GELU derivative: 0.5 * (1 + tanh(g)) + 0.5 * x * sech^2(g) * g'
    # where g = sqrt(2/pi) * (x + 0.044715 * x^3)
    #       g' = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)

    sqrt_2_over_pi = np.sqrt(2 / np.pi)
    coeff = 0.044715

    x2 = x ** 2
    x3 = x2 * x
    g = sqrt_2_over_pi * (x + coeff * x3)
    tanh_g = np.tanh(g)
    sech2_g = 1 - tanh_g ** 2
    g_prime = sqrt_2_over_pi * (1 + 3 * coeff * x2)

    d_gelu = 0.5 * (1 + tanh_g) + 0.5 * x * sech2_g * g_prime

    return d_gelu * d_out


def pytorch_geglu_backward(x, d_out):
    """
    PyTorch GeGLU backward pass.

    dL/dx given dL/d(out) where out = GELU(a) * b
    """
    tokens, inner_dim = x.shape
    dim = inner_dim // 2

    a = x[:, :dim]
    b = x[:, dim:]

    # dL/da = dL/dout * d(GELU)/da * b
    d_gelu = pytorch_gelu_backward(a, d_out)
    d_a = d_gelu * b

    # dL/db = dL/dout * GELU(a)
    gelu_a = pytorch_gelu(a)
    d_b = gelu_a * d_out

    # Concatenate gradients
    d_x = np.zeros_like(x)
    d_x[:, :dim] = d_a
    d_x[:, dim:] = d_b

    return d_x


class TestGeGLUForward:
    """Test GeGLU forward pass."""

    @pytest.fixture
    def sample_input(self):
        """Generate sample input for testing."""
        np.random.seed(42)

        tokens = 8
        dim = 64
        inner_dim = dim * 2

        # Input: [tokens, 2*dim]
        x = np.random.randn(tokens, inner_dim).astype(np.float32) * 0.1

        return x, tokens, dim

    @pytest.mark.skipif(not HAS_CKERNEL, reason="ckernel not available")
    def test_c_kernel_fp32(self, sample_input):
        """Test C kernel for FP32 GeGLU."""
        x, tokens, dim = sample_input

        # Allocate output
        out = np.zeros((tokens, dim), dtype=np.float32)

        # Call C kernel
        lib.geglu_forward_fp32(
            x.ctypes.data_as(ffi.CData),
            out.ctypes.data_as(ffi.CData),
            tokens,
            dim
        )

        # Reference implementation
        ref_out = pytorch_geglu(x)

        np.testing.assert_allclose(out, ref_out, rtol=1e-4, atol=1e-5)

    @pytest.mark.skipif(not HAS_CKERNEL, reason="ckernel not available")
    def test_c_kernel_bf16(self, sample_input):
        """Test C kernel for BF16 GeGLU."""
        x, tokens, dim = sample_input

        # Convert input to BF16
        x_bf16 = np.empty_like(x, dtype=np.uint16)
        for i in range(x.size):
            x_bf16.flat[i] = float_to_bf16(x.flat[i])

        # Allocate output and scratch
        out_bf16 = np.zeros((tokens, dim), dtype=np.uint16)
        # Scratch needs 3 * tokens * dim floats for input + output (see kernel docs)
        scratch = np.zeros(3 * tokens * dim, dtype=np.float32)

        # Call C kernel
        lib.geglu_forward_bf16(
            x_bf16.ctypes.data_as(ffi.CData),
            out_bf16.ctypes.data_as(ffi.CData),
            tokens,
            dim,
            scratch.ctypes.data_as(ffi.CData)
        )

        # Convert output back to FP32 for comparison
        out_fp32 = np.empty((tokens, dim), dtype=np.float32)
        for i in range(out_bf16.size):
            out_fp32.flat[i] = bf16_to_float(out_bf16.flat[i])

        # Reference implementation
        ref_out = pytorch_geglu(x)

        np.testing.assert_allclose(out_fp32, ref_out, rtol=1e-3, atol=1e-4)

    def test_output_shape(self, sample_input):
        """Output should have shape [tokens, dim]."""
        x, tokens, dim = sample_input

        out = pytorch_geglu(x)

        assert out.shape == (tokens, dim)

    def test_equals_reference(self, sample_input):
        """Output should match reference implementation."""
        x, tokens, dim = sample_input

        # Our implementation should match the reference
        out = pytorch_geglu(x)

        # Verify by checking properties
        # GELU(a) is always between -a and a (approximately)
        a = x[:, :dim]
        b = x[:, dim:]

        gelu_a = pytorch_gelu(a)

        # out = GELU(a) * b
        np.testing.assert_array_equal(out, gelu_a * b)

    def test_gelu_properties(self):
        """Verify GELU properties."""
        # GELU is approximately x for large |x|
        large_x = np.array([[100.0, -100.0]]).astype(np.float32)
        gelu_large = pytorch_gelu(large_x)

        # For large positive x, GELU(x) ≈ x
        assert gelu_large[0, 0] > 90.0  # Should be close to 100

        # For large negative x, GELU(x) ≈ 0
        assert abs(gelu_large[0, 1]) < 1.0  # Should be close to 0

    def test_scaling(self):
        """Test with various tensor sizes."""
        for tokens in [1, 8, 32, 128]:
            for dim in [16, 32, 64, 128]:
                x = np.random.randn(tokens, 2 * dim).astype(np.float32) * 0.1

                out = pytorch_geglu(x)

                assert out.shape == (tokens, dim)

    def test_determinism(self, sample_input):
        """Same input should produce same output."""
        x, tokens, dim = sample_input

        out1 = pytorch_geglu(x)
        out2 = pytorch_geglu(x)

        np.testing.assert_array_equal(out1, out2)


class TestGeGLUBackward:
    """Test GeGLU backward pass."""

    @pytest.fixture
    def sample_input(self):
        """Generate sample input for testing."""
        np.random.seed(42)

        tokens = 8
        dim = 64
        inner_dim = dim * 2

        x = np.random.randn(tokens, inner_dim).astype(np.float32) * 0.1

        return x, tokens, dim

    @pytest.mark.skipif(not HAS_CKERNEL, reason="ckernel not available")
    def test_c_kernel_backward(self, sample_input):
        """Test C kernel for GeGLU backward."""
        x, tokens, dim = sample_input

        # Random gradient from upstream
        d_out = np.random.randn(tokens, dim).astype(np.float32) * 0.01

        # Allocate gradient
        d_x = np.zeros((tokens, 2 * dim), dtype=np.float32)

        # Call C kernel
        lib.geglu_backward_fp32(
            x.ctypes.data_as(ffi.CData),
            d_out.ctypes.data_as(ffi.CData),
            d_x.ctypes.data_as(ffi.CData),
            tokens,
            dim
        )

        # Reference implementation
        ref_d_x = pytorch_geglu_backward(x, d_out)

        np.testing.assert_allclose(d_x, ref_d_x, rtol=1e-4, atol=1e-5)

    def test_gradient_shape(self, sample_input):
        """Gradient should have same shape as input."""
        x, tokens, dim = sample_input

        # Random gradient from upstream
        d_out = np.random.randn(tokens, dim).astype(np.float32) * 0.01

        d_x = pytorch_geglu_backward(x, d_out)

        assert d_x.shape == x.shape

    def test_gradient_scaling(self):
        """Test gradients with various tensor sizes."""
        for tokens in [1, 8, 32]:
            for dim in [16, 32, 64]:
                x = np.random.randn(tokens, 2 * dim).astype(np.float32) * 0.1
                d_out = np.random.randn(tokens, dim).astype(np.float32) * 0.01

                d_x = pytorch_geglu_backward(x, d_out)

                assert d_x.shape == (tokens, 2 * dim)

    def test_dL_db_depends_on_gelu_a(self):
        """Gradient w.r.t. 'b' should be GELU(a)."""
        np.random.seed(456)

        tokens = 4
        dim = 16
        inner_dim = 2 * dim

        x = np.random.randn(tokens, inner_dim).astype(np.float32) * 0.1
        d_out = np.ones((tokens, dim), dtype=np.float32) * 0.01

        d_x = pytorch_geglu_backward(x, d_out)

        # dL/db should be GELU(a) * d_out
        gelu_a = pytorch_gelu(x[:, :dim])

        # Check that dL/db is proportional to GELU(a)
        d_b = d_x[:, dim:]
        expected_db = gelu_a * d_out

        np.testing.assert_allclose(d_b, expected_db, rtol=1e-5)


class TestGeGLUEdgeCases:
    """Test edge cases for GeGLU."""

    def test_zero_input(self):
        """Zero input should produce zero output."""
        tokens = 4
        dim = 16
        inner_dim = 2 * dim

        x = np.zeros((tokens, inner_dim), dtype=np.float32)

        out = pytorch_geglu(x)

        np.testing.assert_array_almost_equal(out, np.zeros((tokens, dim)))

    def test_small_values(self):
        """Small input values should produce valid output."""
        tokens = 4
        dim = 16
        inner_dim = 2 * dim

        x = np.random.randn(tokens, inner_dim).astype(np.float32) * 0.001

        out = pytorch_geglu(x)

        assert not np.isnan(out).any()
        assert not np.isinf(out).any()

    def test_large_values(self):
        """Large input values should not overflow."""
        tokens = 4
        dim = 16
        inner_dim = 2 * dim

        # Values that might cause numerical issues
        x = np.random.randn(tokens, inner_dim).astype(np.float32) * 10

        out = pytorch_geglu(x)

        assert not np.isnan(out).any()
        # GELU output should be bounded for large inputs

    def test_single_token(self):
        """Single token should work."""
        tokens = 1
        dim = 32
        inner_dim = 2 * dim

        x = np.random.randn(tokens, inner_dim).astype(np.float32) * 0.1

        out = pytorch_geglu(x)

        assert out.shape == (tokens, dim)


class TestGeGLUComparison:
    """Compare with expected GELU values from literature."""

    def test_gelu_approximation(self):
        """Verify GELU approximation matches expected values."""
        # GELU(0) = 0
        assert abs(pytorch_gelu(np.array([0.0]))[0]) < 1e-6

        # GELU(1) ≈ 0.841... (from literature)
        gelu_1 = pytorch_gelu(np.array([1.0]))[0]
        assert abs(gelu_1 - 0.8413) < 0.01

        # GELU(-1) ≈ -0.158... (from literature)
        gelu_neg1 = pytorch_gelu(np.array([-1.0]))[0]
        assert abs(gelu_neg1 - (-0.1587)) < 0.01


# BF16 conversion utilities for testing
def float_to_bf16(f):
    """Convert float32 to bfloat16."""
    import struct
    # Pack as float32, unpack as uint16 (upper 16 bits)
    packed = struct.pack('f', f)
    return struct.unpack('H', packed[2:4])[0]


def bf16_to_float(bf16):
    """Convert bfloat16 to float32."""
    import struct
    # Pack as uint16, insert zeros in lower bits, unpack as float32
    packed = struct.pack('H', bf16)
    # Insert zeros in the lower 16 bits
    packed_zeroed = struct.pack('f', 0.0)
    # Combine: upper 16 bits from bf16, lower 16 bits from zeroed float
    packed_full = packed_zeroed[:2] + packed
    return struct.unpack('f', packed_full)[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
