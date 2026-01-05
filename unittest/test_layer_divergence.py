#!/usr/bin/env python3
"""
test_layer_divergence.py - Comprehensive layer-by-layer divergence test

Compares C-Kernel-Engine against llama.cpp at every computation step.
Finds the EXACT point where divergence occurs.

Usage:
    python unittest/test_layer_divergence.py --model-dir <path>
    python unittest/test_layer_divergence.py --gguf <path.gguf>
    python unittest/test_layer_divergence.py --list  # Show all test stages
    python unittest/test_layer_divergence.py --verbose  # Detailed output
"""

import os
import sys
import json
import struct
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

# ============================================================================
# Colors and Formatting
# ============================================================================

GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'

def hr(char='=', width=80):
    print(char * width)

def header(text, width=80):
    hr()
    padding = (width - len(text) - 4) // 2
    print(f"{'=' * padding}  {text}  {'=' * padding}")
    hr()


# ============================================================================
# Quantization Helpers (matching llama.cpp exactly)
# ============================================================================

def fp16_to_fp32(h):
    """Convert FP16 (uint16) to FP32"""
    return np.frombuffer(struct.pack('<H', h), dtype=np.float16)[0].astype(np.float32)


def dequant_q8_0_block(data):
    """Dequantize one Q8_0 block (34 bytes -> 32 floats)"""
    d = np.frombuffer(data[0:2], dtype=np.float16)[0].astype(np.float32)
    qs = np.frombuffer(data[2:34], dtype=np.int8).astype(np.float32)
    return qs * d


def dequant_q4_k_block(data):
    """Dequantize one Q4_K block (144 bytes -> 256 floats)"""
    d = np.frombuffer(data[0:2], dtype=np.float16)[0].astype(np.float32)
    dmin = np.frombuffer(data[2:4], dtype=np.float16)[0].astype(np.float32)
    scales = list(data[4:16])
    qs = data[16:144]

    result = np.zeros(256, dtype=np.float32)

    def get_scale_min(j, q):
        if j < 4:
            return q[j] & 63, q[j + 4] & 63
        else:
            return (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4), (q[j+4] >> 4) | ((q[j] >> 6) << 4)

    q_ptr, is_idx = 0, 0
    for j in range(0, 256, 64):
        sc1, m1 = get_scale_min(is_idx, scales)
        sc2, m2 = get_scale_min(is_idx + 1, scales)
        d1, dm1 = d * sc1, dmin * m1
        d2, dm2 = d * sc2, dmin * m2

        for l in range(32):
            result[j + l] = d1 * (qs[q_ptr + l] & 0x0F) - dm1
        for l in range(32):
            result[j + 32 + l] = d2 * (qs[q_ptr + l] >> 4) - dm2

        q_ptr += 32
        is_idx += 2

    return result


def dequant_q6_k_block(data):
    """Dequantize one Q6_K block (210 bytes -> 256 floats)"""
    ql = data[0:128]
    qh = data[128:192]
    scales = data[192:208]
    d = np.frombuffer(data[208:210], dtype=np.float16)[0].astype(np.float32)

    result = np.zeros(256, dtype=np.float32)

    for n in range(16):
        sc = np.array(scales[n], dtype=np.uint8).view(np.int8).astype(np.float32)
        ql_off = n * 8
        qh_off = n * 4

        for j in range(16):
            ql_val = (ql[ql_off + j // 2] >> (4 * (j % 2))) & 0x0F
            qh_byte = qh_off + j // 4
            qh_shift = 2 * (j % 4)
            qh_val = (qh[qh_byte] >> qh_shift) & 0x03
            q6 = ql_val | (qh_val << 4)
            result[n * 16 + j] = d * sc * (q6 - 32)

    return result


# ============================================================================
# GGUF Reader
# ============================================================================

class GGUFReader:
    """Read tensors directly from GGUF file"""

    def __init__(self, path):
        self.path = path
        self.tensors = {}
        self.metadata = {}
        self._read_header()

    def _read_string(self, f):
        length = struct.unpack('<Q', f.read(8))[0]
        return f.read(length).decode('utf-8')

    def _read_value(self, f, vtype):
        if vtype == 0: return struct.unpack('<B', f.read(1))[0]
        elif vtype == 1: return struct.unpack('<b', f.read(1))[0]
        elif vtype == 4: return struct.unpack('<I', f.read(4))[0]
        elif vtype == 5: return struct.unpack('<i', f.read(4))[0]
        elif vtype == 6: return struct.unpack('<f', f.read(4))[0]
        elif vtype == 8: return self._read_string(f)
        elif vtype == 9:
            arr_type = struct.unpack('<I', f.read(4))[0]
            arr_len = struct.unpack('<Q', f.read(8))[0]
            return [self._read_value(f, arr_type) for _ in range(arr_len)]
        elif vtype == 10: return struct.unpack('<Q', f.read(8))[0]
        elif vtype == 11: return struct.unpack('<q', f.read(8))[0]
        else:
            sizes = {2: 2, 3: 2, 7: 1, 12: 8}
            f.read(sizes.get(vtype, 0))
            return None

    def _read_header(self):
        with open(self.path, 'rb') as f:
            magic = f.read(4)
            version = struct.unpack('<I', f.read(4))[0]
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            metadata_kv_count = struct.unpack('<Q', f.read(8))[0]

            # Read metadata
            for _ in range(metadata_kv_count):
                key = self._read_string(f)
                vtype = struct.unpack('<I', f.read(4))[0]
                value = self._read_value(f, vtype)
                self.metadata[key] = value

            # Read tensor infos
            DTYPE_NAMES = {0: 'F32', 1: 'F16', 2: 'Q4_0', 6: 'Q5_0', 8: 'Q8_0', 12: 'Q4_K', 14: 'Q6_K'}
            for _ in range(tensor_count):
                name = self._read_string(f)
                n_dims = struct.unpack('<I', f.read(4))[0]
                dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
                dtype = struct.unpack('<I', f.read(4))[0]
                offset = struct.unpack('<Q', f.read(8))[0]
                self.tensors[name] = {
                    'dims': dims, 'dtype': dtype, 'offset': offset,
                    'dtype_name': DTYPE_NAMES.get(dtype, f'UNK({dtype})')
                }

            self.data_start = (f.tell() + 31) // 32 * 32

    def read_tensor_row(self, name, row_idx=0):
        """Read one row of a tensor, dequantized to float32"""
        if name not in self.tensors:
            return None

        info = self.tensors[name]
        dtype = info['dtype']
        dims = info['dims']

        # For 2D tensors, dims[0] is embed_dim, dims[1] is vocab/other
        # For 1D tensors (biases), dims[0] is the size
        embed_dim = dims[0]

        with open(self.path, 'rb') as f:
            abs_offset = self.data_start + info['offset']

            if dtype == 8:  # Q8_0: 34 bytes per 32 values
                bytes_per_row = (embed_dim // 32) * 34
                f.seek(abs_offset + row_idx * bytes_per_row)
                result = np.zeros(embed_dim, dtype=np.float32)
                for i in range(embed_dim // 32):
                    block = f.read(34)
                    if len(block) < 34:
                        break
                    result[i*32:(i+1)*32] = dequant_q8_0_block(block)
                return result

            elif dtype == 12:  # Q4_K
                bytes_per_row = (embed_dim // 256) * 144
                f.seek(abs_offset + row_idx * bytes_per_row)
                result = np.zeros(embed_dim, dtype=np.float32)
                for i in range(embed_dim // 256):
                    result[i*256:(i+1)*256] = dequant_q4_k_block(f.read(144))
                return result

            elif dtype == 14:  # Q6_K
                bytes_per_row = (embed_dim // 256) * 210
                f.seek(abs_offset + row_idx * bytes_per_row)
                result = np.zeros(embed_dim, dtype=np.float32)
                for i in range(embed_dim // 256):
                    result[i*256:(i+1)*256] = dequant_q6_k_block(f.read(210))
                return result

            elif dtype == 0:  # F32
                f.seek(abs_offset + row_idx * embed_dim * 4)
                return np.frombuffer(f.read(embed_dim * 4), dtype=np.float32)

            elif dtype == 1:  # F16
                f.seek(abs_offset + row_idx * embed_dim * 2)
                return np.frombuffer(f.read(embed_dim * 2), dtype=np.float16).astype(np.float32)

        return None


# ============================================================================
# Test Framework
# ============================================================================

@dataclass
class TestResult:
    name: str
    stage: str
    status: str  # PASS, FAIL, SKIP, WARN
    message: str
    c_values: Optional[np.ndarray] = None
    ref_values: Optional[np.ndarray] = None
    max_diff: float = 0.0
    diverge_idx: int = -1


class LayerDivergenceTester:
    """Comprehensive layer-by-layer divergence tester"""

    TEST_STAGES = [
        ("1.1", "token_embedding", "Token embedding lookup"),
        ("1.2", "token_embedding_stats", "Token embedding statistics"),
        ("2.1", "rmsnorm_weights", "RMSNorm weights (ln1_gamma)"),
        ("2.2", "rmsnorm_output", "RMSNorm computation"),
        ("3.1", "wq_weights", "WQ weights (Q projection)"),
        ("3.2", "wq_bias", "WQ bias (bq)"),
        ("3.3", "wk_weights", "WK weights (K projection)"),
        ("3.4", "wk_bias", "WK bias (bk)"),
        ("3.5", "wv_weights", "WV weights (V projection)"),
        ("3.6", "wv_bias", "WV bias (bv)"),
        ("4.1", "q_projection", "Q = WQ @ x + bq"),
        ("4.2", "k_projection", "K = WK @ x + bk"),
        ("4.3", "v_projection", "V = WV @ x + bv"),
        ("5.1", "rope_q", "RoPE on Q"),
        ("5.2", "rope_k", "RoPE on K"),
        ("6.1", "attention_scores", "Attention scores (QK^T)"),
        ("6.2", "attention_softmax", "Attention softmax"),
        ("6.3", "attention_output", "Attention @ V"),
        ("7.1", "wo_projection", "Output projection (WO)"),
        ("7.2", "residual_1", "Residual add #1"),
        ("8.1", "ln2_weights", "RMSNorm weights (ln2_gamma)"),
        ("8.2", "ln2_output", "RMSNorm before MLP"),
        ("9.1", "w1_weights", "Gate+Up weights (W1)"),
        ("9.2", "swiglu", "SwiGLU activation"),
        ("9.3", "w2_weights", "Down weights (W2)"),
        ("9.4", "mlp_output", "MLP output"),
        ("9.5", "residual_2", "Residual add #2"),
        ("10.1", "final_norm", "Final RMSNorm"),
        ("10.2", "lm_head", "LM head projection"),
        ("10.3", "logits", "Final logits"),
        ("10.4", "top_tokens", "Top predicted tokens"),
    ]

    def __init__(self, gguf_path=None, model_dir=None, verbose=False):
        self.verbose = verbose
        self.results: List[TestResult] = []

        if gguf_path:
            self.gguf = GGUFReader(gguf_path)
            self.model_dir = model_dir or Path(gguf_path).parent
        elif model_dir:
            self.model_dir = Path(model_dir)
            gguf_files = list(self.model_dir.glob("*.gguf"))
            if gguf_files:
                self.gguf = GGUFReader(str(gguf_files[0]))
            else:
                raise ValueError("No GGUF file found")
        else:
            raise ValueError("Must specify --gguf or --model-dir")

        # Load config
        self.config = {}
        config_path = self.model_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)

        # Extract model dimensions from metadata
        self.embed_dim = self.gguf.metadata.get('qwen2.embedding_length',
                         self.config.get('hidden_size', 896))
        self.num_heads = self.gguf.metadata.get('qwen2.attention.head_count',
                         self.config.get('num_attention_heads', 14))
        self.num_kv_heads = self.gguf.metadata.get('qwen2.attention.head_count_kv',
                            self.config.get('num_key_value_heads', 2))
        self.head_dim = self.embed_dim // self.num_heads
        self.intermediate_dim = self.gguf.metadata.get('qwen2.feed_forward_length',
                                self.config.get('intermediate_size', 4864))
        self.vocab_size = self.gguf.metadata.get('qwen2.vocab_size', 151936)
        self.num_layers = self.gguf.metadata.get('qwen2.block_count', 24)

    def print_config(self):
        """Print model configuration"""
        header("MODEL CONFIGURATION")
        print(f"  {'GGUF file:':<25} {self.gguf.path}")
        print(f"  {'Tensors:':<25} {len(self.gguf.tensors)}")
        print()
        print(f"  {'Embedding dim:':<25} {self.embed_dim}")
        print(f"  {'Num attention heads:':<25} {self.num_heads}")
        print(f"  {'Num KV heads:':<25} {self.num_kv_heads}")
        print(f"  {'Head dim:':<25} {self.head_dim}")
        print(f"  {'Intermediate dim:':<25} {self.intermediate_dim}")
        print(f"  {'Vocab size:':<25} {self.vocab_size}")
        print(f"  {'Num layers:':<25} {self.num_layers}")
        hr()

    def list_stages(self):
        """List all test stages"""
        header("TEST STAGES")
        print(f"{'Stage':<8} {'ID':<25} {'Description':<40}")
        print("-" * 75)
        for stage_num, stage_id, desc in self.TEST_STAGES:
            print(f"{stage_num:<8} {stage_id:<25} {desc:<40}")
        hr()

    def compare_values(self, name: str, c_vals: np.ndarray, ref_vals: np.ndarray,
                       tol: float = 1e-4) -> TestResult:
        """Compare C output vs reference"""
        if c_vals is None:
            return TestResult(name, name, "SKIP", "No C values")
        if ref_vals is None:
            return TestResult(name, name, "SKIP", "No reference values")

        if c_vals.shape != ref_vals.shape:
            return TestResult(name, name, "FAIL",
                              f"Shape mismatch: {c_vals.shape} vs {ref_vals.shape}",
                              c_vals, ref_vals)

        nan_count = np.isnan(c_vals).sum()
        if nan_count > 0:
            return TestResult(name, name, "FAIL", f"C has {nan_count} NaN values",
                              c_vals, ref_vals)

        diff = np.abs(c_vals - ref_vals)
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())

        if max_diff > tol:
            diverge_idx = int(np.argmax(diff > tol))
            return TestResult(name, name, "FAIL",
                              f"max_diff={max_diff:.2e} @ idx {diverge_idx}",
                              c_vals, ref_vals, max_diff, diverge_idx)

        return TestResult(name, name, "PASS",
                          f"max_diff={max_diff:.2e}, mean={mean_diff:.2e}",
                          c_vals, ref_vals, max_diff)

    def _print_result(self, result: TestResult, stage_num: str, desc: str):
        """Print a single test result"""
        status_colors = {
            "PASS": GREEN, "FAIL": RED, "SKIP": YELLOW, "WARN": YELLOW
        }
        color = status_colors.get(result.status, RESET)

        print(f"  {stage_num:<6} {color}[{result.status:^4}]{RESET} {desc:<35} {result.message}")

        if self.verbose and result.c_values is not None:
            print(f"         C first 5:   {result.c_values[:5]}")
            if result.ref_values is not None:
                print(f"         Ref first 5: {result.ref_values[:5]}")

    def test_token_embedding(self, token_id: int) -> np.ndarray:
        """Test stage 1: Token embedding"""
        stage_num, stage_id, desc = self.TEST_STAGES[0]

        emb = self.gguf.read_tensor_row("token_embd.weight", token_id)

        if emb is None:
            result = TestResult(stage_id, stage_id, "FAIL", "Could not read embedding")
        else:
            nan_count = np.isnan(emb).sum()
            if nan_count > 0:
                result = TestResult(stage_id, stage_id, "FAIL", f"{nan_count} NaN values", emb, None)
            else:
                result = TestResult(stage_id, stage_id, "PASS",
                                    f"range=[{emb.min():.4f}, {emb.max():.4f}]", emb, None)

        self.results.append(result)
        self._print_result(result, stage_num, desc)

        # Stage 1.2: Stats
        if emb is not None:
            stage_num2, stage_id2, desc2 = self.TEST_STAGES[1]
            stats_msg = f"mean={emb.mean():.4f}, std={emb.std():.4f}"
            result2 = TestResult(stage_id2, stage_id2, "PASS", stats_msg, emb, None)
            self.results.append(result2)
            self._print_result(result2, stage_num2, desc2)

        return emb

    def test_rmsnorm(self, input_tensor: np.ndarray, gamma_name: str,
                     stage_idx: int) -> np.ndarray:
        """Test RMSNorm"""
        stage_num, stage_id, desc = self.TEST_STAGES[stage_idx]

        # Map gamma name to GGUF tensor name
        gguf_names = {
            'layer.0.ln1_gamma': 'blk.0.attn_norm.weight',
            'layer.0.ln2_gamma': 'blk.0.ffn_norm.weight',
        }
        gguf_name = gguf_names.get(gamma_name, gamma_name)

        gamma = self.gguf.read_tensor_row(gguf_name, 0)

        if gamma is None:
            result = TestResult(stage_id, stage_id, "FAIL", f"Could not read {gguf_name}")
            self.results.append(result)
            self._print_result(result, stage_num, desc)
            return None

        result = TestResult(stage_id, stage_id, "PASS",
                            f"range=[{gamma.min():.4f}, {gamma.max():.4f}]", gamma, None)
        self.results.append(result)
        self._print_result(result, stage_num, desc)

        # Stage: RMSNorm computation
        stage_num2, stage_id2, desc2 = self.TEST_STAGES[stage_idx + 1]

        eps = 1e-6
        variance = np.mean(input_tensor ** 2)
        rms = np.sqrt(variance + eps)
        normed = (input_tensor / rms) * gamma[:len(input_tensor)]

        result2 = TestResult(stage_id2, stage_id2, "PASS",
                             f"rms={rms:.4f}, out=[{normed.min():.4f}, {normed.max():.4f}]",
                             normed, None)
        self.results.append(result2)
        self._print_result(result2, stage_num2, desc2)

        return normed

    def test_qkv_weights(self, layer_idx: int = 0):
        """Test Q, K, V weight loading"""
        # Test stages 3.1 - 3.6

        tests = [
            (4, f'blk.{layer_idx}.attn_q.weight', 'WQ'),
            (5, f'blk.{layer_idx}.attn_q.bias', 'bq'),
            (6, f'blk.{layer_idx}.attn_k.weight', 'WK'),
            (7, f'blk.{layer_idx}.attn_k.bias', 'bk'),
            (8, f'blk.{layer_idx}.attn_v.weight', 'WV'),
            (9, f'blk.{layer_idx}.attn_v.bias', 'bv'),
        ]

        for test_offset, tensor_name, short_name in tests:
            stage_num, stage_id, desc = self.TEST_STAGES[test_offset]

            if tensor_name in self.gguf.tensors:
                info = self.gguf.tensors[tensor_name]
                # Read first row/values
                vals = self.gguf.read_tensor_row(tensor_name, 0)
                if vals is not None:
                    result = TestResult(stage_id, stage_id, "PASS",
                                        f"{info['dtype_name']} [{vals.min():.4f}, {vals.max():.4f}]",
                                        vals, None)
                else:
                    result = TestResult(stage_id, stage_id, "WARN", "Could not dequantize", None, None)
            else:
                result = TestResult(stage_id, stage_id, "SKIP", "Not in GGUF", None, None)

            self.results.append(result)
            self._print_result(result, stage_num, desc)

    def run_all(self, token_id: int = 14990):
        """Run all tests"""
        self.print_config()

        header(f"LAYER-BY-LAYER DIVERGENCE TEST (token={token_id})")
        print(f"{'Stage':<8} {'Status':<8} {'Description':<35} {'Details'}")
        print("-" * 80)

        # Stage 1: Token embedding
        emb = self.test_token_embedding(token_id)
        if emb is None:
            return

        # Stage 2: RMSNorm
        normed = self.test_rmsnorm(emb, 'layer.0.ln1_gamma', 2)

        # Stage 3: QKV weights
        self.test_qkv_weights(0)

        # Print summary
        hr()
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        skipped = sum(1 for r in self.results if r.status == "SKIP")
        total = len(self.results)

        print(f"\n  {BOLD}SUMMARY{RESET}")
        print(f"  {GREEN}Passed:{RESET}  {passed}/{total}")
        if failed > 0:
            print(f"  {RED}Failed:{RESET}  {failed}/{total}")
        if skipped > 0:
            print(f"  {YELLOW}Skipped:{RESET} {skipped}/{total}")

        # Find first failure
        first_fail = next((r for r in self.results if r.status == "FAIL"), None)
        if first_fail:
            print(f"\n  {RED}First divergence:{RESET} {first_fail.name}")
            print(f"  {first_fail.message}")

        hr()
        return failed == 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Layer-by-layer divergence test")
    parser.add_argument('--gguf', type=str, help="Path to GGUF file")
    parser.add_argument('--model-dir', type=str,
                        default=os.path.expanduser("~/.cache/ck-engine-v5/models/Qwen--Qwen2-0.5B-Instruct-GGUF"))
    parser.add_argument('--token-id', type=int, default=14990, help="Token ID to test")
    parser.add_argument('--list', action='store_true', help="List all test stages")
    parser.add_argument('--verbose', '-v', action='store_true', help="Verbose output")
    args = parser.parse_args()

    tester = LayerDivergenceTester(
        gguf_path=args.gguf,
        model_dir=args.model_dir if not args.gguf else None,
        verbose=args.verbose
    )

    if args.list:
        tester.print_config()
        tester.list_stages()
        return 0

    success = tester.run_all(args.token_id)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
