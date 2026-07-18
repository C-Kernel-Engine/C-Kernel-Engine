#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "stitched_parity_v8.py"
SPEC = importlib.util.spec_from_file_location("stitched_parity_v8_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
stitched = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stitched)


def args_for(mode: str) -> argparse.Namespace:
    return argparse.Namespace(
        decoder_gguf=Path("decoder.gguf"),
        mmproj_gguf=Path("mmproj.gguf"),
        prompt="prompt",
        chat_template="qwen3vl",
        image_path=Path("image.ppm"),
        ctx_len=4096,
        top_k=16,
        image_min_tokens=None,
        image_max_tokens=1024,
        threads=20,
        compiler="gcc",
        execution_mode=mode,
        granular_dump_names="kqv_out",
        granular_ck_stop=True,
    )


class StitchedParityModeTests(unittest.TestCase):
    def test_parity_runtime_uses_one_explicit_compiler_family(self) -> None:
        args = args_for("production")
        env = stitched._runtime_environment(args)
        self.assertEqual(env["CC"], "gcc")
        args.compiler = "icx"
        self.assertEqual(stitched._runtime_environment(args)["CC"], "icx")

    def test_strict_mode_is_consistent_across_phases(self) -> None:
        args = args_for("strict")
        bridge = stitched._bridge_command(args, Path("bridge"), Path("prefix.f32"))
        numeric = stitched._encoder_numeric_command(args, Path("numeric"), Path("numeric.json"))
        granular = stitched._granular_command(args, 0, Path("granular"), Path("granular.json"))
        self.assertIn("--strict-parity", bridge)
        self.assertIn("--strict-parity", numeric)
        self.assertIn("--strict-parity", granular)
        self.assertEqual(numeric[numeric.index("--llama-flash-attn") + 1], "disabled")
        self.assertEqual(granular[granular.index("--llama-flash-attn") + 1], "disabled")

    def test_production_mode_never_injects_strict_and_uses_llama_flash(self) -> None:
        args = args_for("production")
        bridge = stitched._bridge_command(args, Path("bridge"), Path("prefix.f32"))
        numeric = stitched._encoder_numeric_command(args, Path("numeric"), Path("numeric.json"))
        granular = stitched._granular_command(args, 0, Path("granular"), Path("granular.json"))
        self.assertNotIn("--strict-parity", bridge)
        self.assertNotIn("--strict-parity", numeric)
        self.assertNotIn("--strict-parity", granular)
        self.assertEqual(numeric[numeric.index("--llama-flash-attn") + 1], "enabled")
        self.assertEqual(granular[granular.index("--llama-flash-attn") + 1], "enabled")

    def test_encoder_numeric_observe_policy_is_exposed(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('choices=("require", "observe", "skip")', source)
        self.assertIn('"non_causal_for_observed_token_window"', source)

    def test_failed_runtime_is_retained_by_default(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('"--prune-failed-weights"', source)
        self.assertIn('"failed_weight_bumps_retained"', source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
