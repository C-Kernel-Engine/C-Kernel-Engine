#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from array import array
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V8_DECODER_PARITY_PATH = ROOT / "version" / "v8" / "scripts" / "decoder_first_token_parity_v8.py"


def _load_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


decoder_parity_v8 = _load_module("decoder_first_token_parity_v8_tests", V8_DECODER_PARITY_PATH)


class V8DecoderFirstTokenParityTests(unittest.TestCase):
    def test_load_llama_dump_dir_parses_jsonl_index(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_decoder_llama_dump_") as tmpdir:
            tmp = Path(tmpdir)
            raw = np.array([1.0, -2.0, 3.5, 4.0], dtype=np.float32)
            bin_name = "Qcur-0-token-000001-occ-000"
            (tmp / f"{bin_name}.bin").write_bytes(raw.tobytes())
            (tmp / "index.json").write_text(
                json.dumps(
                    {
                        "name": bin_name,
                        "base_name": "Qcur-0",
                        "token_id": 1,
                        "occurrence": 0,
                        "dtype": 0,
                        "rank": 2,
                        "shape": [2, 2, 1, 1],
                        "elem_count": 4,
                        "nbytes": 16,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            dumps = decoder_parity_v8._load_llama_dump_dir(tmp)

            self.assertEqual(len(dumps), 1)
            self.assertEqual(dumps[0].layer_id, 0)
            self.assertEqual(dumps[0].op_name, "q_proj")
            self.assertEqual(dumps[0].token_id, 1)
            np.testing.assert_allclose(dumps[0].data, raw.reshape(2, 2))

    def test_compare_dump_sets_reports_failures(self) -> None:
        ck_dumps = [
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "Qcur",
                np.array([1.0, 2.0], dtype=np.float32),
                1,
                "fp32",
            )
        ]
        llama_dumps = [
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "Qcur",
                np.array([1.0, 2.5], dtype=np.float32),
                1,
                "fp32",
            )
        ]

        report = decoder_parity_v8._compare_dump_sets(
            ck_dumps,
            llama_dumps,
            atol=1.0e-4,
            rtol=1.0e-3,
            pass_filter="decode",
        )

        self.assertEqual(report["summary"]["fail"], 1)
        self.assertEqual(report["first_issue"]["op"], "Qcur")
        self.assertEqual(report["first_issue"]["status"], "FAIL")

    def test_report_passes_when_top1_and_overlap_match(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_decoder_parity_pass_") as tmpdir:
            tmp = Path(tmpdir)
            report_path = tmp / "report.json"
            fake_gguf = tmp / "decoder.gguf"

            class FakeTokenizer:
                def encode(self, text: str) -> list[int]:
                    self.last_text = text
                    return [11, 22]

                def decode(self, ids: list[int], skip_special: bool = False) -> str:
                    return ",".join(str(x) for x in ids)

            fake_runtime = {
                "embed_dim": 16,
                "vocab_size": 4,
                "so_path": tmp / "libdecoder_v8.so",
                "c_path": tmp / "decoder_v8.c",
            }

            with mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_prepare_decoder_runtime", return_value=fake_runtime), \
                 mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_run_decoder", return_value={"vocab_size": 4, "logits": array("f", [0.1, 0.9, 0.2, -0.4])}), \
                 mock.patch.object(decoder_parity_v8.GGUFTokenizer, "from_gguf", return_value=FakeTokenizer()), \
                 mock.patch.object(
                     decoder_parity_v8.compare_first_token_logits_v7,
                     "run_llama_logits",
                     return_value={
                         "meta": {
                             "n_vocab": 4,
                             "token_count": 2,
                             "topk": [{"token_id": 1, "logit": 0.95}, {"token_id": 2, "logit": 0.18}],
                         },
                         "logits": np.array([0.0, 1.0, 0.1, -0.5], dtype=np.float32),
                     },
                 ):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = decoder_parity_v8.main(
                        [
                            "--gguf",
                            str(fake_gguf),
                            "--workdir",
                            str(tmp / "work"),
                            "--prompt",
                            "Hello",
                            "--top-k",
                            "2",
                            "--json-out",
                            str(report_path),
                        ]
                    )

            self.assertEqual(rc, 0)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "pass")
            self.assertTrue(report["pass"])
            self.assertEqual(report["tokens"], [11, 22])
            self.assertEqual(report["prefix"]["source"], "none")
            self.assertEqual(report["compare"]["top1_ck"], 1)
            self.assertEqual(report["compare"]["top1_llama"], 1)
            self.assertTrue(report["compare"]["top1_match"])
            self.assertGreaterEqual(report["compare"]["topk_overlap_ratio"], 0.5)

    def test_report_fails_when_top1_mismatches(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_decoder_parity_fail_") as tmpdir:
            tmp = Path(tmpdir)
            report_path = tmp / "report.json"
            fake_gguf = tmp / "decoder.gguf"

            class FakeTokenizer:
                def encode(self, text: str) -> list[int]:
                    return [7, 8]

                def decode(self, ids: list[int], skip_special: bool = False) -> str:
                    return "|".join(str(x) for x in ids)

            fake_runtime = {
                "embed_dim": 16,
                "vocab_size": 4,
                "so_path": tmp / "libdecoder_v8.so",
                "c_path": tmp / "decoder_v8.c",
            }

            with mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_prepare_decoder_runtime", return_value=fake_runtime), \
                 mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_run_decoder", return_value={"vocab_size": 4, "logits": array("f", [0.7, 0.1, 0.2, -0.4])}), \
                 mock.patch.object(decoder_parity_v8.GGUFTokenizer, "from_gguf", return_value=FakeTokenizer()), \
                 mock.patch.object(
                     decoder_parity_v8.compare_first_token_logits_v7,
                     "run_llama_logits",
                     return_value={
                         "meta": {
                             "n_vocab": 4,
                             "token_count": 2,
                             "topk": [{"token_id": 2, "logit": 0.8}, {"token_id": 0, "logit": 0.2}],
                         },
                         "logits": np.array([0.1, 0.2, 0.8, -0.1], dtype=np.float32),
                     },
                 ):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = decoder_parity_v8.main(
                        [
                            "--gguf",
                            str(fake_gguf),
                            "--workdir",
                            str(tmp / "work"),
                            "--prompt",
                            "Hello",
                            "--top-k",
                            "2",
                            "--json-out",
                            str(report_path),
                        ]
                    )

            self.assertEqual(rc, 3)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "fail")
            self.assertFalse(report["pass"])
            self.assertFalse(report["compare"]["top1_match"])
            self.assertEqual(report["compare"]["top1_ck"], 0)
            self.assertEqual(report["compare"]["top1_llama"], 2)


if __name__ == "__main__":
    unittest.main()
