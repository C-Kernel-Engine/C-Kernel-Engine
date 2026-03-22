#!/usr/bin/env python3
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).with_name("check_decode_attention_contract.py")
SPEC = importlib.util.spec_from_file_location("check_decode_attention_contract", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"failed to load module spec for {SCRIPT_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CheckDecodeAttentionContractTest(unittest.TestCase):
    def test_detect_rope_layout_prefers_pairwise_kernel(self) -> None:
        manifest = {
            "template": {
                "name": "llama",
                "family": "llama",
                "kernels": {"rope_qk": "rope_forward_qk_pairwise"},
            }
        }
        self.assertEqual(MODULE._detect_rope_layout(manifest), "pairwise")

    def test_detect_rope_layout_falls_back_to_llama_architecture(self) -> None:
        manifest = {"metadata": {"general.architecture": "llama"}}
        self.assertEqual(MODULE._detect_rope_layout(manifest), "pairwise")

    def test_detect_rope_layout_defaults_to_split_half(self) -> None:
        manifest = {"template": {"name": "qwen2", "family": "qwen2"}}
        self.assertEqual(MODULE._detect_rope_layout(manifest), "split_half")

    def test_detect_rope_layout_uses_pairwise_for_gemma_template_kernel(self) -> None:
        manifest = {
            "template": {
                "name": "gemma3",
                "family": "llama",
                "kernels": {"rope_qk": "rope_forward_qk_pairwise"},
            }
        }
        self.assertEqual(MODULE._detect_rope_layout(manifest), "pairwise")

    def test_pairwise_rotation_matches_manual_pairs(self) -> None:
        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        out = MODULE._apply_rope_single(
            x,
            pos=1,
            theta=10000.0,
            rotary_dim=4,
            layout="pairwise",
        )

        half = 2
        inv_freq = 1.0 / (10000.0 ** (np.arange(half, dtype=np.float32) / float(half)))
        freqs = inv_freq
        cos = np.cos(freqs).astype(np.float32)
        sin = np.sin(freqs).astype(np.float32)
        expected = np.array(
            [[
                (1.0 * cos[0]) - (2.0 * sin[0]),
                (1.0 * sin[0]) + (2.0 * cos[0]),
                (3.0 * cos[1]) - (4.0 * sin[1]),
                (3.0 * sin[1]) + (4.0 * cos[1]),
            ]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(out, expected, atol=1e-6, rtol=0.0)

    def test_pairwise_and_split_half_layouts_differ(self) -> None:
        x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
        pairwise = MODULE._apply_rope_single(
            x,
            pos=3,
            theta=10000.0,
            rotary_dim=8,
            layout="pairwise",
        )
        split_half = MODULE._apply_rope_single(
            x,
            pos=3,
            theta=10000.0,
            rotary_dim=8,
            layout="split_half",
        )
        self.assertFalse(np.allclose(pairwise, split_half, atol=1e-7, rtol=0.0))


if __name__ == "__main__":
    unittest.main()
