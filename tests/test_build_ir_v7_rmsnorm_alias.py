#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version" / "v7" / "scripts"))

import build_ir_v7  # type: ignore


class BuildIrV7RmsnormAliasTests(unittest.TestCase):
    def test_generate_ir_lower_3_accepts_rstd_output_alias_for_rmsnorm(self) -> None:
        lowered_ir = {
            "config": {
                "seq_len": 4,
                "embed_dim": 8,
                "rms_eps": 1e-6,
            },
            "operations": [
                {
                    "idx": 0,
                    "function": "rmsnorm_forward",
                    "op": "attn_norm",
                    "layer": 0,
                    "section": "body",
                    "activations": {
                        "input": {
                            "buffer": "embedded_input",
                            "activation_offset": 128,
                            "dtype": "fp32",
                        }
                    },
                    "outputs": {
                        "output": {
                            "buffer": "embedded_input",
                            "activation_offset": 128,
                            "dtype": "fp32",
                        },
                        "rstd": {
                            "buffer": "rstd_tmp",
                            "activation_offset": 256,
                            "dtype": "fp32",
                        },
                    },
                    "weights": {
                        "ln1_gamma": {
                            "name": "layer.0.ln1_gamma",
                            "bump_offset": 64,
                            "dtype": "fp32",
                        }
                    },
                    "params": {
                        "seq_len": 4,
                        "embed_dim": 8,
                        "rms_eps": 1e-6,
                    },
                    "scratch": [],
                }
            ],
        }

        lowered_call = build_ir_v7.generate_ir_lower_3(lowered_ir, "prefill")

        self.assertEqual(lowered_call.get("errors"), [])
        ops = lowered_call.get("operations") or []
        self.assertEqual(len(ops), 1)
        op = ops[0]
        self.assertEqual(op.get("errors"), [])
        rstd_arg = next(arg for arg in (op.get("args") or []) if arg.get("name") == "rstd_cache")
        self.assertEqual(rstd_arg.get("expr"), "NULL")
        self.assertEqual(rstd_arg.get("source"), "null")


if __name__ == "__main__":
    unittest.main()
