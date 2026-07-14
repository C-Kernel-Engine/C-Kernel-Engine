#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version/v8/scripts/compare_multimodal_multitoken_logits_v8.py"


def load_module():
    spec = importlib.util.spec_from_file_location("multitoken_eos_contract", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MultitokenEOSContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load_module()

    def test_bridge_stop_token_list_is_authoritative(self) -> None:
        self.assertEqual(
            self.runner._resolve_stop_token_ids(
                {"stop_token_ids": [151645, 151643], "eos_token_id": 7}
            ),
            {151645, 151643},
        )

    def test_eos_token_is_used_when_bridge_has_no_stop_list(self) -> None:
        self.assertEqual(
            self.runner._resolve_stop_token_ids({"eos_token_id": 151645}),
            {151645},
        )

    def test_only_a_matched_declared_token_stops_parity(self) -> None:
        stops = {151645}
        self.assertTrue(self.runner._is_matched_stop_token(151645, 151645, stops))
        self.assertFalse(self.runner._is_matched_stop_token(151645, 4, stops))
        self.assertFalse(self.runner._is_matched_stop_token(4, 4, stops))

    def test_exact_runtime_reuse_loads_only_declared_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            decoder_dir = Path(tmp)
            (decoder_dir / "layout_decode.json").write_text(
                json.dumps(
                    {
                        "config": {
                            "embed_dim": 4096,
                            "num_deepstack_layers": 3,
                            "context_length": 4096,
                            "vocab_size": 151936,
                        }
                    }
                ),
                encoding="utf-8",
            )
            for name in ("weights.bump", "weights_manifest.map", "libdecoder_v8.so"):
                (decoder_dir / name).touch()

            runtime = self.runner._load_exact_decoder_runtime(
                Path("decoder.gguf"),
                decoder_dir,
                so_override=None,
                manifest_map_override=None,
            )

            self.assertEqual(runtime["embed_dim"], 4096)
            self.assertEqual(runtime["input_embed_dim"], 16384)
            self.assertEqual(runtime["context_length"], 4096)
            self.assertEqual(runtime["so_path"], decoder_dir / "libdecoder_v8.so")
            evidence = self.runner._runtime_evidence(runtime, exact_reuse=True)
            self.assertTrue(evidence["exact_reuse"])
            self.assertEqual(
                evidence["shared_library"]["sha256"],
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            )

    def test_exact_runtime_reuse_fails_instead_of_regenerating(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "refusing to regenerate or guess"):
                self.runner._load_exact_decoder_runtime(
                    Path("decoder.gguf"),
                    Path(tmp),
                    so_override=None,
                    manifest_map_override=None,
                )

    def test_runner_records_eos_step_without_decoding_past_it(self) -> None:
        class Tokenizer:
            def decode(self, tokens, skip_special=False):
                return ",".join(str(token) for token in tokens)

        class Library:
            decode_calls = 0

            def ck_model_decode(self, token, logits):
                self.decode_calls += 1
                return 0

            def ck_model_free(self):
                return None

        library = Library()
        inputs = {
            "tokenizer": Tokenizer(),
            "bridge_report": {"stop_token_ids": [151645]},
            "gguf_path": Path("decoder.gguf"),
            "workdir": Path("work"),
            "ctx_len": 4096,
            "requested_ctx_len": 4096,
            "tokens_before": [1],
            "tokens_after": [2],
            "prefix_source": "fixture",
            "prefix_tokens": 1008,
            "prefix_row_dim": 16384,
            "prefix_path": Path("prefix.f32"),
            "prefix_grid": (36, 28),
            "prefix_text_pos": 41,
            "runtime": {},
        }
        comparison = {
            "top1_ck": 151645,
            "top1_llama": 151645,
            "cosine": 1.0,
            "rmse": 0.0,
            "mean_abs_diff": 0.0,
            "max_abs_diff": 0.0,
            "ck_top1_margin": 1.0,
            "llama_top1_margin": 1.0,
            "topk_overlap_count": 1,
            "topk_overlap_ratio": 1.0,
            "ck_topk_ids": [151645],
            "llama_topk_ids": [151645],
            "topk_logits": [],
        }
        args = Namespace(
            ck_strict_parity=False,
            max_new_tokens=64,
            top_k=1,
            llama_no_repack=False,
            append_on_divergence="stop",
            bridge_report=Path("bridge_report.json"),
            threads=20,
            llama_decode_mode="persistent",
        )
        llama_sequence = {
            "meta": {"greedy_generated": [151645, 7]},
            "logits": np.zeros((2, 3), dtype=np.float32),
        }
        with mock.patch.object(self.runner, "_prepare_inputs", return_value=inputs), \
             mock.patch.object(self.runner, "_run_llama_greedy_sequence", return_value=llama_sequence), \
             mock.patch.object(self.runner, "_init_ck_state", return_value=(library, object(), 3)), \
             mock.patch.object(self.runner, "_ck_logits_from_buffer", return_value=np.zeros(3, dtype=np.float32)), \
             mock.patch.object(self.runner, "_runtime_evidence", return_value={}), \
             mock.patch.object(self.runner.first_token.compare_first_token_logits_v7, "compare_logits", return_value=comparison), \
             mock.patch.object(self.runner, "_decode_topk", return_value=[]):
            report = self.runner.run_multimodal_multitoken_parity(args)

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["stop_reason"], "matched_stop_token")
        self.assertEqual(len(report["steps"]), 1)
        self.assertEqual(report["steps"][0]["ck_next"], 151645)
        self.assertEqual(library.decode_calls, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
