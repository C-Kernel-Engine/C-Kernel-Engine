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
    def test_llama_helper_fingerprint_tracks_root_source_and_library_content(self) -> None:
        helper_module = decoder_parity_v8.compare_first_token_logits_v7
        with tempfile.TemporaryDirectory(prefix="v8_llama_helper_identity_") as tmpdir:
            root = Path(tmpdir)
            source = root / "llama_token_replay_v8.cpp"
            source.write_text("int main() { return 0; }\n", encoding="utf-8")
            lib_dir = root / "build" / "bin"
            lib_dir.mkdir(parents=True)
            for name in ("libllama.so", "libggml.so", "libggml-cpu.so", "libggml-base.so"):
                (lib_dir / name).write_bytes(name.encode("ascii"))

            with mock.patch.object(helper_module, "LLAMA_CPP", root), mock.patch.object(
                helper_module, "HELPER_SRC", source
            ):
                original = helper_module._llama_helper_fingerprint()
                (lib_dir / "libggml-cpu.so").write_bytes(b"different provider")
                changed_library = helper_module._llama_helper_fingerprint()
                source.write_text("int main() { return 1; }\n", encoding="utf-8")
                changed_source = helper_module._llama_helper_fingerprint()

            self.assertNotEqual(original, changed_library)
            self.assertNotEqual(changed_library, changed_source)

    def test_ck_dump_filter_names_expands_llama_aliases(self) -> None:
        self.assertEqual(
            decoder_parity_v8._ck_dump_filter_names("Qcur-0,Kcur_normed-2,ffn_inp-0,l_out-3"),
            "Qcur-0,q_proj-0,Qcur_normed-0,qcur_normed-0,Qcur_rope-0,qcur_rope-0,"
            "Kcur_normed-2,kcur_normed-2,ffn_inp-0,l_out-3,layer_out-3",
        )
        self.assertEqual(
            decoder_parity_v8._ck_dump_filter_names("Kcur-1"),
            "Kcur-1,k_proj-1,Kcur_normed-1,kcur_normed-1,Kcur_rope-1,kcur_rope-1",
        )

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

    def test_load_llama_dump_dir_labels_post_rope_occurrence(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_decoder_llama_rope_dump_") as tmpdir:
            tmp = Path(tmpdir)
            rows = []
            for occurrence in (0, 1, 2):
                name = f"Qcur-0-token-000001-occ-{occurrence:03d}"
                raw = np.full(8, float(occurrence), dtype=np.float32)
                (tmp / f"{name}.bin").write_bytes(raw.tobytes())
                rows.append(
                    {
                        "name": name,
                        "base_name": "Qcur-0",
                        "token_id": 1,
                        "occurrence": occurrence,
                        "dtype": 0,
                        "rank": 2,
                        "shape": [4, 2, 1, 1],
                        "elem_count": 8,
                        "nbytes": 32,
                    }
                )
            (tmp / "index.json").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            dumps = decoder_parity_v8._load_llama_dump_dir(tmp)

            self.assertEqual(
                [dump.op_name for dump in dumps],
                ["q_proj", "qcur_normed", "qcur_rope"],
            )


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

    def test_compare_dump_sets_preserves_graph_order_for_first_issue(self) -> None:
        def dump(op_name: str, value: float):
            return decoder_parity_v8.parity_test_v7.ParityDump(
                0, op_name, np.array([value], dtype=np.float32), 0, "fp32"
            )

        report = decoder_parity_v8._compare_dump_sets(
            [dump("ffn_inp", 2.0), dump("down_proj", 4.0)],
            [dump("ffn_inp", 1.0), dump("down_proj", 3.0)],
            atol=1.0e-4,
            rtol=1.0e-3,
            pass_filter="decode",
        )

        self.assertEqual([row["op"] for row in report["results"]], ["ffn_inp", "down_proj"])
        self.assertEqual(report["first_issue"]["op"], "ffn_inp")

    def test_compare_dump_sets_rejects_distinct_ambiguous_occurrences(self) -> None:
        dump = decoder_parity_v8.parity_test_v7.ParityDump
        report = decoder_parity_v8._compare_dump_sets(
            [dump(0, "v_proj", np.array([1.0], dtype=np.float32), 7, "fp32")],
            [
                dump(0, "v_proj", np.array([1.0], dtype=np.float32), 7, "fp32"),
                dump(0, "v_proj", np.array([2.0], dtype=np.float32), 7, "fp32"),
            ],
            atol=0.0,
            rtol=0.0,
            pass_filter="decode",
        )

        self.assertEqual(report["summary"]["error"], 1)
        self.assertEqual(report["first_issue"]["reason"], "ambiguous_alignment")

    def test_compare_dump_sets_accepts_identical_reshape_occurrences(self) -> None:
        dump = decoder_parity_v8.parity_test_v7.ParityDump
        values = np.array([1.0, 2.0], dtype=np.float32)
        report = decoder_parity_v8._compare_dump_sets(
            [dump(0, "v_proj", values.copy(), 7, "fp32")],
            [
                dump(0, "v_proj", values.copy(), 7, "fp32"),
                dump(0, "v_proj", values.reshape(2, 1).copy(), 7, "fp32"),
            ],
            atol=0.0,
            rtol=0.0,
            pass_filter="decode",
        )

        self.assertEqual(report["summary"]["pass"], 1)
        self.assertFalse(report["results"][0]["alignment_ambiguous"])

    def test_expand_ck_prefill_decode_dumps_splits_prompt_rows(self) -> None:
        ck_dumps = [
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "q_proj",
                np.arange(24, dtype=np.float32),
                0,
                "fp32",
            ),
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "kqv_out",
                np.arange(100, 124, dtype=np.float32),
                0,
                "fp32",
            ),
        ]
        llama_dumps = [
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "q_proj",
                np.zeros(6, dtype=np.float32),
                0,
                "fp32",
            ),
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "kqv_out",
                np.zeros(6, dtype=np.float32),
                0,
                "fp32",
            ),
        ]

        expanded = decoder_parity_v8._expand_ck_prefill_decode_dumps(
            ck_dumps,
            llama_dumps,
            prompt_start_token=0,
            prompt_token_count=2,
        )

        q_rows = [d for d in expanded if d.layer_id == 0 and d.op_name == "q_proj"]
        self.assertEqual([d.token_id for d in q_rows], [0, 1])
        np.testing.assert_allclose(q_rows[0].data, np.arange(12, 18, dtype=np.float32))
        np.testing.assert_allclose(q_rows[1].data, np.arange(18, 24, dtype=np.float32))

        attn_rows = [d for d in expanded if d.layer_id == 0 and d.op_name == "kqv_out"]
        self.assertEqual([d.token_id for d in attn_rows], [0, 1])
        np.testing.assert_allclose(attn_rows[0].data, np.arange(112, 118, dtype=np.float32))
        np.testing.assert_allclose(attn_rows[1].data, np.arange(118, 124, dtype=np.float32))

    def test_expand_ck_prefill_decode_dumps_selects_exact_trailing_segment(self) -> None:
        row_elems = 4

        def ck_segment(rows: int, base: float):
            data = np.arange(rows * row_elems, dtype=np.float32) + base
            return decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "q_proj",
                data,
                1083,
                "fp32",
            )

        # Reproduces Qwen3-VL segmented mixed prefill: text-before, visual,
        # then the requested text-after/generated-token replay window.
        ck_dumps = [
            ck_segment(5, 1000.0),
            ck_segment(1008, 2000.0),
            ck_segment(71, 3000.0),
        ]
        llama_dumps = [
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "q_proj",
                np.zeros(row_elems, dtype=np.float32),
                token_id,
                "fp32",
            )
            for token_id in range(71)
        ]

        expanded = decoder_parity_v8._expand_ck_prefill_decode_dumps(
            ck_dumps,
            llama_dumps,
            prompt_start_token=1013,
            prompt_token_count=71,
        )

        rows = [d for d in expanded if d.layer_id == 0 and d.op_name == "q_proj"]
        self.assertEqual([d.token_id for d in rows], list(range(71)))
        self.assertTrue(all(d.data.size == row_elems for d in rows))
        np.testing.assert_allclose(rows[0].data, np.arange(4, dtype=np.float32) + 3000.0)
        np.testing.assert_allclose(rows[-1].data, np.arange(280, 284, dtype=np.float32) + 3000.0)

    def test_expand_ck_prefill_decode_dumps_extracts_head_major_norm_rows(self) -> None:
        # CK stores qcur_normed head-major as [heads, tokens, dim].
        # The parity comparator flattens both sides, so expansion must preserve
        # CK's native flat order rather than transposing into dim-major form.
        ck_tensor = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        ck_dumps = [
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "qcur_normed",
                ck_tensor.reshape(-1),
                0,
                "fp32",
            )
        ]
        llama_dumps = [
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "qcur_normed",
                np.zeros((4, 2), dtype=np.float32),
                0,
                "fp32",
            )
        ]

        expanded = decoder_parity_v8._expand_ck_prefill_decode_dumps(
            ck_dumps,
            llama_dumps,
            prompt_start_token=0,
            prompt_token_count=2,
        )

        rows = [d for d in expanded if d.layer_id == 0 and d.op_name == "qcur_normed"]
        self.assertEqual([d.token_id for d in rows], [0, 1])
        np.testing.assert_allclose(rows[0].data, ck_tensor[:, 1, :])
        np.testing.assert_allclose(rows[1].data, ck_tensor[:, 2, :])

    def test_expand_ck_prefill_decode_dumps_extracts_token_major_rope_rows(self) -> None:
        # The dedicated CK post-RoPE dumper writes [tokens, heads, dim], unlike
        # the raw qcur_normed scratch tensor above.
        ck_tensor = np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4)
        ck_dumps = [
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "qcur_rope",
                ck_tensor.reshape(-1),
                0,
                "fp32",
            )
        ]
        llama_dumps = [
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "qcur_rope",
                np.zeros((4, 2), dtype=np.float32),
                0,
                "fp32",
            )
        ]

        expanded = decoder_parity_v8._expand_ck_prefill_decode_dumps(
            ck_dumps,
            llama_dumps,
            prompt_start_token=0,
            prompt_token_count=2,
        )

        rows = [d for d in expanded if d.layer_id == 0 and d.op_name == "qcur_rope"]
        self.assertEqual([d.token_id for d in rows], [0, 1])
        np.testing.assert_allclose(rows[0].data, ck_tensor[1])
        np.testing.assert_allclose(rows[1].data, ck_tensor[2])

    def test_build_llama_row_specs_prefers_ranked_norm_shape_on_tie(self) -> None:
        specs = decoder_parity_v8._build_llama_row_specs(
            [
                decoder_parity_v8.parity_test_v7.ParityDump(
                    0,
                    "qcur_normed",
                    np.zeros(8, dtype=np.float32),
                    0,
                    "fp32",
                ),
                decoder_parity_v8.parity_test_v7.ParityDump(
                    0,
                    "qcur_normed",
                    np.zeros((4, 2), dtype=np.float32),
                    0,
                    "fp32",
                ),
            ]
        )

        self.assertEqual(specs[(0, "qcur_normed")], (8, (4, 2)))

    def test_trim_llama_prefill_decode_dumps_preserves_duplicate_occurrences(self) -> None:
        llama_dumps = [
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "q_proj",
                np.array([1.0], dtype=np.float32),
                1,
                "fp32",
            ),
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "q_proj",
                np.array([2.0], dtype=np.float32),
                1,
                "fp32",
            ),
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "q_proj",
                np.array([3.0], dtype=np.float32),
                0,
                "fp32",
            ),
        ]

        trimmed = decoder_parity_v8._trim_llama_prefill_decode_dumps(
            llama_dumps,
            prompt_start_token=1,
            prompt_token_count=1,
        )

        q_rows = [d for d in trimmed if d.layer_id == 0 and d.op_name == "q_proj"]
        self.assertEqual(len(q_rows), 2)
        self.assertEqual([d.token_id for d in q_rows], [0, 0])
        np.testing.assert_allclose(q_rows[0].data, np.array([1.0], dtype=np.float32))
        np.testing.assert_allclose(q_rows[1].data, np.array([2.0], dtype=np.float32))

    def test_expand_llama_prefill_decode_dumps_selects_execution_tail_not_largest_position(self) -> None:
        dump = decoder_parity_v8.parity_test_v7.ParityDump
        row_elems = 4
        llama_dumps = [
            dump(0, "q_proj", np.arange(5 * row_elems, dtype=np.float32), 4, "fp32"),
            dump(0, "q_proj", np.arange(1008 * row_elems, dtype=np.float32), 1012, "fp32"),
            dump(
                0,
                "q_proj",
                np.arange(59 * row_elems, dtype=np.float32) + 10000.0,
                99,
                "fp32",
            ),
        ]

        expanded = decoder_parity_v8._expand_llama_prefill_decode_dumps(
            llama_dumps,
            prompt_token_count=59,
        )

        self.assertEqual([item.token_id for item in expanded], list(range(59)))
        self.assertTrue(all(item.data.size == row_elems for item in expanded))
        np.testing.assert_allclose(
            expanded[-1].data,
            np.arange(232, 236, dtype=np.float32) + 10000.0,
        )

    def test_resolve_decode_prompt_start_tokens_uses_stage_and_rope_windows(self) -> None:
        ck_start, llama_start = decoder_parity_v8._resolve_decode_prompt_start_tokens(
            tokens_before_count=4,
            prefix_tokens=9,
            prefix_text_pos=7,
            llama_meta={"prefix_text_pos": 7},
        )

        self.assertEqual(ck_start, 13)
        self.assertEqual(llama_start, 7)

    def test_build_multimodal_position_contract_matches_qwen3vl_grid_contract(self) -> None:
        contract = decoder_parity_v8._build_multimodal_position_contract(
            tokens_before_count=4,
            prefix_tokens=9,
            prefix_grid=(3, 3),
            prefix_text_pos=7,
            llama_meta={"prefix_start_pos": 4, "prefix_text_pos": 7},
        )

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertTrue(contract["rows_match"])
        self.assertTrue(contract["text_pos_match"])
        self.assertEqual(contract["ck"]["rows"][0], [4, 4, 4, 0])
        self.assertEqual(contract["ck"]["rows"][1], [4, 4, 5, 0])
        self.assertEqual(contract["ck"]["rows"][3], [4, 5, 4, 0])
        self.assertEqual(contract["llama"]["rows"], contract["ck"]["rows"])

    def test_compare_dump_sets_keeps_native_kqv_out_distinct_from_attn_output(self) -> None:
        ck_dumps = [
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "kqv_out",
                np.array([1.0, 2.0], dtype=np.float32),
                0,
                "fp32",
            ),
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "attn_output",
                np.array([9.0, 9.0], dtype=np.float32),
                0,
                "fp32",
            ),
        ]
        llama_dumps = [
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "kqv_out",
                np.array([1.0, 2.0], dtype=np.float32),
                0,
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

        self.assertEqual(report["summary"]["pass"], 1)
        self.assertEqual(report["summary"]["warn"], 1)
        self.assertIsNone(report["first_issue"])
        by_op = {row["op"]: row for row in report["results"]}
        self.assertEqual(by_op["kqv_out"]["status"], "PASS")
        self.assertEqual(by_op["attn_output"]["status"], "WARN")

    def test_compare_dump_sets_legacy_attn_output_falls_back_to_kqv_out(self) -> None:
        ck_dumps = [
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "attn_output",
                np.array([1.0, 2.0], dtype=np.float32),
                0,
                "fp32",
            )
        ]
        llama_dumps = [
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "kqv_out",
                np.array([1.0, 2.0], dtype=np.float32),
                0,
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

        self.assertEqual(report["summary"]["pass"], 1)
        self.assertEqual(report["summary"]["warn"], 1)
        self.assertIsNone(report["first_issue"])
        by_op = {row["op"]: row for row in report["results"]}
        self.assertEqual(by_op["kqv_out"]["status"], "PASS")
        self.assertEqual(by_op["attn_output"]["status"], "WARN")

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
                "input_embed_dim": 64,
                "vocab_size": 4,
                "so_path": tmp / "libdecoder_v8.so",
                "c_path": tmp / "decoder_v8.c",
            }

            with mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_prepare_decoder_runtime", return_value=fake_runtime), \
                 mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_run_decoder", return_value={"vocab_size": 4, "logits": array("f", [0.1, 0.9, 0.2, -0.4])}), \
                 mock.patch.object(decoder_parity_v8.GGUFTokenizer, "from_gguf", return_value=FakeTokenizer()), \
                 mock.patch.object(
                     decoder_parity_v8,
                     "_run_llama_capture",
                     return_value={
                         "meta": {
                             "ok": True,
                             "n_vocab": 4,
                             "token_count": 2,
                             "prefix_token_count": 0,
                             "topk": [{"id": 1, "logit": 0.95}, {"id": 2, "logit": 0.18}],
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
                "input_embed_dim": 64,
                "vocab_size": 4,
                "so_path": tmp / "libdecoder_v8.so",
                "c_path": tmp / "decoder_v8.c",
            }

            with mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_prepare_decoder_runtime", return_value=fake_runtime), \
                 mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_run_decoder", return_value={"vocab_size": 4, "logits": array("f", [0.7, 0.1, 0.2, -0.4])}), \
                 mock.patch.object(decoder_parity_v8.GGUFTokenizer, "from_gguf", return_value=FakeTokenizer()), \
                 mock.patch.object(
                     decoder_parity_v8,
                     "_run_llama_capture",
                     return_value={
                         "meta": {
                             "ok": True,
                             "n_vocab": 4,
                             "token_count": 2,
                             "prefix_token_count": 0,
                             "topk": [{"id": 2, "logit": 0.8}, {"id": 0, "logit": 0.2}],
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

    def test_report_replays_prefix_file_on_llama_side(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_decoder_parity_prefix_") as tmpdir:
            tmp = Path(tmpdir)
            report_path = tmp / "report.json"
            fake_gguf = tmp / "decoder.gguf"
            prefix_path = tmp / "prefix.f32"
            prefix = array("f", [0.0] * (3 * 16))
            prefix_path.write_bytes(prefix.tobytes())

            class FakeTokenizer:
                def encode(self, text: str) -> list[int]:
                    return [101, 202]

                def decode(self, ids: list[int], skip_special: bool = False) -> str:
                    return ",".join(str(x) for x in ids)

            fake_runtime = {
                "embed_dim": 16,
                "input_embed_dim": 64,
                "vocab_size": 4,
                "so_path": tmp / "libdecoder_v8.so",
                "c_path": tmp / "decoder_v8.c",
            }

            with mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_prepare_decoder_runtime", return_value=fake_runtime), \
                 mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_run_decoder", return_value={"vocab_size": 4, "logits": array("f", [0.1, 0.9, 0.2, -0.4])}), \
                 mock.patch.object(decoder_parity_v8.GGUFTokenizer, "from_gguf", return_value=FakeTokenizer()), \
                 mock.patch.object(
                     decoder_parity_v8,
                     "_run_llama_capture",
                     return_value={
                         "meta": {
                             "ok": True,
                             "n_vocab": 4,
                             "token_count": 2,
                             "prefix_token_count": 3,
                             "topk": [{"id": 1, "logit": 0.95}, {"id": 2, "logit": 0.18}],
                         },
                         "logits": np.array([0.0, 1.0, 0.1, -0.5], dtype=np.float32),
                     },
                 ) as llama_capture:
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = decoder_parity_v8.main(
                        [
                            "--gguf",
                            str(fake_gguf),
                            "--workdir",
                            str(tmp / "work"),
                            "--prompt",
                            "Hello",
                            "--prefix-f32",
                            str(prefix_path),
                            "--top-k",
                            "2",
                            "--json-out",
                            str(report_path),
                        ]
                    )

            self.assertEqual(rc, 0)
            _, kwargs = llama_capture.call_args
            self.assertEqual(kwargs["prefix_path"], prefix_path.resolve())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["prefix"]["source"], "file")
            self.assertEqual(report["prefix"]["tokens"], 3)

    def test_report_replays_synthetic_prefix_on_llama_side(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_decoder_parity_synth_prefix_") as tmpdir:
            tmp = Path(tmpdir)
            report_path = tmp / "report.json"
            fake_gguf = tmp / "decoder.gguf"

            class FakeTokenizer:
                def encode(self, text: str) -> list[int]:
                    return [101, 202]

                def decode(self, ids: list[int], skip_special: bool = False) -> str:
                    return ",".join(str(x) for x in ids)

            fake_runtime = {
                "embed_dim": 16,
                "input_embed_dim": 64,
                "vocab_size": 4,
                "so_path": tmp / "libdecoder_v8.so",
                "c_path": tmp / "decoder_v8.c",
            }

            with mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_prepare_decoder_runtime", return_value=fake_runtime), \
                 mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_run_decoder", return_value={"vocab_size": 4, "logits": array("f", [0.1, 0.9, 0.2, -0.4])}), \
                 mock.patch.object(decoder_parity_v8.GGUFTokenizer, "from_gguf", return_value=FakeTokenizer()), \
                 mock.patch.object(
                     decoder_parity_v8,
                     "_run_llama_capture",
                     return_value={
                         "meta": {
                             "ok": True,
                             "n_vocab": 4,
                             "token_count": 2,
                             "prefix_token_count": 3,
                             "topk": [{"id": 1, "logit": 0.95}, {"id": 2, "logit": 0.18}],
                         },
                         "logits": np.array([0.0, 1.0, 0.1, -0.5], dtype=np.float32),
                     },
                 ) as llama_capture:
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = decoder_parity_v8.main(
                        [
                            "--gguf",
                            str(fake_gguf),
                            "--workdir",
                            str(tmp / "work"),
                            "--prompt",
                            "Hello",
                            "--synthetic-prefix-tokens",
                            "3",
                            "--top-k",
                            "2",
                            "--json-out",
                            str(report_path),
                        ]
                    )

            self.assertEqual(rc, 0)
            _, kwargs = llama_capture.call_args
            prefix_path = Path(kwargs["prefix_path"])
            self.assertTrue(prefix_path.exists())
            self.assertEqual(kwargs["prefix_row_dim"], 64)
            self.assertEqual(prefix_path.stat().st_size, 3 * 64 * 4)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["prefix"]["source"], "synthetic_zero")
            self.assertEqual(report["prefix"]["tokens"], 3)
            self.assertEqual(report["prefix"]["row_dim"], 64)

    def test_main_passes_explicit_prefix_grid_through_replay(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_decoder_parity_explicit_grid_") as tmpdir:
            tmp = Path(tmpdir)
            report_path = tmp / "report.json"
            prefix_path = tmp / "prefix.f32"
            prefix_path.write_bytes(array("f", [0.0] * (6 * 64)).tobytes())
            fake_gguf = tmp / "decoder.gguf"

            class FakeTokenizer:
                def encode(self, text: str) -> list[int]:
                    return [101, 202]

                def decode(self, ids: list[int], skip_special: bool = False) -> str:
                    return ",".join(str(x) for x in ids)

            fake_runtime = {
                "embed_dim": 16,
                "input_embed_dim": 64,
                "vocab_size": 4,
                "so_path": tmp / "libdecoder_v8.so",
                "c_path": tmp / "decoder_v8.c",
            }

            with mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_prepare_decoder_runtime", return_value=fake_runtime), \
                 mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_run_decoder", return_value={"vocab_size": 4, "logits": array("f", [0.1, 0.9, 0.2, -0.4])}) as run_decoder, \
                 mock.patch.object(decoder_parity_v8.GGUFTokenizer, "from_gguf", return_value=FakeTokenizer()), \
                 mock.patch.object(
                     decoder_parity_v8,
                     "_run_llama_capture",
                     return_value={
                         "meta": {
                             "ok": True,
                             "n_vocab": 4,
                             "token_count": 2,
                             "prefix_token_count": 6,
                             "prefix_position_count": 3,
                             "topk": [{"id": 1, "logit": 0.95}, {"id": 2, "logit": 0.18}],
                         },
                         "logits": np.array([0.0, 1.0, 0.1, -0.5], dtype=np.float32),
                     },
                 ) as llama_capture:
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = decoder_parity_v8.main(
                        [
                            "--gguf",
                            str(fake_gguf),
                            "--workdir",
                            str(tmp / "work"),
                            "--prompt",
                            "Hello",
                            "--prefix-f32",
                            str(prefix_path),
                            "--prefix-row-dim",
                            "64",
                            "--prefix-grid-x",
                            "2",
                            "--prefix-grid-y",
                            "3",
                            "--prefix-text-pos",
                            "7",
                            "--top-k",
                            "2",
                            "--json-out",
                            str(report_path),
                        ]
                    )

            self.assertEqual(rc, 0)
            _, llama_kwargs = llama_capture.call_args
            self.assertEqual(llama_kwargs["prefix_grid"], (2, 3))
            _, decoder_kwargs = run_decoder.call_args
            self.assertEqual(decoder_kwargs["prefix_grid"], (2, 3))
            self.assertEqual(decoder_kwargs["prefix_text_pos"], 7)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["prefix"]["grid"], [2, 3])
            self.assertEqual(report["prefix"]["text_pos"], 7)

    def test_main_passes_ctx_len_into_decoder_runtime_prep(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_decoder_ctx_len_") as tmpdir:
            tmp = Path(tmpdir)
            report_path = tmp / "report.json"
            fake_gguf = tmp / "decoder.gguf"

            class FakeTokenizer:
                def encode(self, text: str) -> list[int]:
                    return [1, 2]

                def decode(self, ids: list[int], skip_special: bool = False) -> str:
                    return ",".join(str(x) for x in ids)

            fake_runtime = {
                "embed_dim": 16,
                "vocab_size": 4,
                "so_path": tmp / "libdecoder_v8.so",
                "c_path": tmp / "decoder_v8.c",
            }

            with mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_prepare_decoder_runtime", return_value=fake_runtime) as prepare_runtime, \
                 mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_run_decoder", return_value={"vocab_size": 4, "logits": array("f", [0.1, 0.9, 0.2, -0.4])}), \
                 mock.patch.object(decoder_parity_v8.GGUFTokenizer, "from_gguf", return_value=FakeTokenizer()), \
                 mock.patch.object(
                     decoder_parity_v8,
                     "_run_llama_capture",
                     return_value={
                         "meta": {
                             "ok": True,
                             "n_vocab": 4,
                             "token_count": 2,
                             "prefix_token_count": 0,
                             "topk": [{"id": 1, "logit": 0.95}, {"id": 2, "logit": 0.18}],
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
                            "--ctx-len",
                            "123",
                            "--json-out",
                            str(report_path),
                        ]
                    )

            self.assertEqual(rc, 0)
            _, kwargs = prepare_runtime.call_args
            self.assertEqual(kwargs["context_override"], 123)


    def test_main_auto_bumps_ctx_len_for_prefix_budget(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_decoder_ctx_autobump_") as tmpdir:
            tmp = Path(tmpdir)
            report_path = tmp / "report.json"
            fake_gguf = tmp / "decoder.gguf"

            class FakeTokenizer:
                def encode(self, text: str) -> list[int]:
                    return [1, 2]

                def decode(self, ids: list[int], skip_special: bool = False) -> str:
                    return ",".join(str(x) for x in ids)

            fake_runtime = {
                "embed_dim": 16,
                "input_embed_dim": 16,
                "vocab_size": 4,
                "so_path": tmp / "libdecoder_v8.so",
                "c_path": tmp / "decoder_v8.c",
            }

            with mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_prepare_decoder_runtime", return_value=fake_runtime) as prepare_runtime,                  mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_run_decoder", return_value={"vocab_size": 4, "logits": array("f", [0.1, 0.9, 0.2, -0.4])}),                  mock.patch.object(decoder_parity_v8.GGUFTokenizer, "from_gguf", return_value=FakeTokenizer()),                  mock.patch.object(decoder_parity_v8, "_load_prefix_embeddings", return_value=(array("f", [0.0] * 3 * 16), 3, 16, "synthetic_zero")),                  mock.patch.object(
                     decoder_parity_v8,
                     "_run_llama_capture",
                     return_value={
                         "meta": {
                             "ok": True,
                             "n_vocab": 4,
                             "token_count": 2,
                             "prefix_token_count": 3,
                             "topk": [{"id": 1, "logit": 0.95}, {"id": 2, "logit": 0.18}],
                         },
                         "logits": np.array([0.0, 1.0, 0.1, -0.5], dtype=np.float32),
                     },
                 ) as llama_capture:
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = decoder_parity_v8.main(
                        [
                            "--gguf",
                            str(fake_gguf),
                            "--workdir",
                            str(tmp / "work"),
                            "--prompt",
                            "Hello",
                            "--ctx-len",
                            "1",
                            "--json-out",
                            str(report_path),
                        ]
                    )

            self.assertEqual(rc, 0)
            self.assertEqual(prepare_runtime.call_count, 2)
            self.assertEqual(prepare_runtime.call_args_list[0].kwargs["context_override"], 1)
            self.assertEqual(prepare_runtime.call_args_list[1].kwargs["context_override"], 5)
            self.assertEqual(llama_capture.call_args.args[2], 5)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["requested_ctx_len"], 1)
            self.assertEqual(report["ctx_len"], 5)

    def test_capture_dump_compare_replays_prefix_for_decode_pass(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_decoder_dump_prefix_") as tmpdir:
            tmp = Path(tmpdir)
            prefix = array("f", [1.0, 2.0, 3.0, 4.0])
            ck_dump = decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "q_proj",
                np.array([1.0, 2.0], dtype=np.float32),
                0,
                "fp32",
            )
            llama_dump = decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "q_proj",
                np.array([1.0, 2.0], dtype=np.float32),
                0,
                "fp32",
            )
            with mock.patch.object(
                decoder_parity_v8,
                "_run_llama_capture",
                return_value={"meta": {"decode_mode": "sequential", "dumped": 1}, "logits": np.array([0.0], dtype=np.float32)},
            ) as llama_capture, \
                 mock.patch.object(
                     decoder_parity_v8,
                     "_capture_ck_dump",
                     return_value={"vocab_size": 1, "logits": array("f", [0.0])},
                 ), \
                 mock.patch.object(decoder_parity_v8.parity_test_v7, "read_dump_file", return_value=[ck_dump]), \
                 mock.patch.object(decoder_parity_v8, "_load_llama_dump_dir", return_value=[llama_dump]):
                ck, report = decoder_parity_v8._capture_dump_compare(
                    Path("/tmp/model.gguf"),
                    {"embed_dim": 2},
                    prefix,
                    2,
                    [11, 22],
                    prefix_row_dim=2,
                    ctx_len=6,
                    top_k=2,
                    threads=1,
                    dump_root=tmp,
                    dump_names="Qcur-0",
                    dump_pass="decode",
                    dump_atol=1.0e-4,
                    dump_rtol=1.0e-3,
                )

            self.assertEqual(ck["vocab_size"], 1)
            _, kwargs = llama_capture.call_args
            self.assertEqual(kwargs["decode_mode"], "sequential")
            self.assertTrue(Path(kwargs["prefix_path"]).exists())
            self.assertEqual(report["status"], "ok")
            self.assertTrue(str(report["prefix_path"]).endswith("prefix.f32"))

    def test_capture_dump_compare_expands_prefill_batch_rows_for_decode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_decoder_dump_expand_") as tmpdir:
            tmp = Path(tmpdir)
            prefix = array("f", [1.0, 2.0, 3.0, 4.0])
            ck_dump = decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "q_proj",
                np.arange(8, dtype=np.float32),
                0,
                "fp32",
            )
            llama_dump = decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "q_proj",
                np.zeros(4, dtype=np.float32),
                0,
                "fp32",
            )

            captured: dict[str, object] = {}

            def _fake_compare(ck_dumps, llama_dumps, *, atol, rtol, pass_filter):
                captured["ck_dumps"] = ck_dumps
                return {"summary": {"total": 1, "pass": 1, "fail": 0, "error": 0, "warn": 0, "missing": 0}, "first_issue": None, "results": []}

            with mock.patch.object(
                decoder_parity_v8,
                "_run_llama_capture",
                return_value={"meta": {"decode_mode": "sequential", "dumped": 1}, "logits": np.array([0.0], dtype=np.float32)},
            ), \
                 mock.patch.object(
                     decoder_parity_v8,
                     "_capture_ck_dump",
                     return_value={"vocab_size": 1, "logits": array("f", [0.0])},
                 ), \
                 mock.patch.object(decoder_parity_v8.parity_test_v7, "read_dump_file", return_value=[ck_dump]), \
                 mock.patch.object(decoder_parity_v8, "_load_llama_dump_dir", return_value=[llama_dump]), \
                 mock.patch.object(decoder_parity_v8, "_compare_dump_sets", side_effect=_fake_compare):
                _, report = decoder_parity_v8._capture_dump_compare(
                    Path("/tmp/model.gguf"),
                    {"embed_dim": 2},
                    prefix,
                    2,
                    [11, 22],
                    prefix_row_dim=2,
                    ctx_len=6,
                    top_k=2,
                    threads=1,
                    dump_root=tmp,
                    dump_names="Qcur-0",
                    dump_pass="decode",
                    dump_atol=1.0e-4,
                    dump_rtol=1.0e-3,
                )

            self.assertEqual(report["status"], "ok")
            ck_rows = captured["ck_dumps"]
            self.assertIsInstance(ck_rows, list)
            self.assertEqual([d.token_id for d in ck_rows], [0, 1])
            np.testing.assert_allclose(ck_rows[0].data, np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32))
            np.testing.assert_allclose(ck_rows[1].data, np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32))

    def test_capture_dump_compare_rebases_segmented_prompt_window(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_decoder_dump_segmented_") as tmpdir:
            tmp = Path(tmpdir)
            prefix = array("f", [1.0, 2.0, 3.0, 4.0])
            ck_dumps = [
                decoder_parity_v8.parity_test_v7.ParityDump(
                    0,
                    "q_proj",
                    np.array([1.0, 2.0], dtype=np.float32),
                    11,
                    "fp32",
                ),
                decoder_parity_v8.parity_test_v7.ParityDump(
                    0,
                    "q_proj",
                    np.array([3.0, 4.0], dtype=np.float32),
                    12,
                    "fp32",
                ),
            ]
            llama_dumps = [
                decoder_parity_v8.parity_test_v7.ParityDump(
                    0,
                    "q_proj",
                    np.array([1.0, 2.0], dtype=np.float32),
                    5,
                    "fp32",
                ),
                decoder_parity_v8.parity_test_v7.ParityDump(
                    0,
                    "q_proj",
                    np.array([3.0, 4.0], dtype=np.float32),
                    6,
                    "fp32",
                ),
            ]

            captured: dict[str, object] = {}

            def _fake_compare(ck_rows, llama_rows, *, atol, rtol, pass_filter):
                captured["ck_rows"] = ck_rows
                captured["llama_rows"] = llama_rows
                return {"summary": {"total": 1, "pass": 1, "fail": 0, "error": 0, "warn": 0, "missing": 0}, "first_issue": None, "results": []}

            with mock.patch.object(
                decoder_parity_v8,
                "_run_llama_capture",
                return_value={
                    "meta": {"decode_mode": "sequential", "dumped": 2, "prefix_text_pos": 5},
                    "logits": np.array([0.0], dtype=np.float32),
                },
            ), \
                 mock.patch.object(
                     decoder_parity_v8,
                     "_capture_ck_dump",
                     return_value={"vocab_size": 1, "logits": array("f", [0.0])},
                 ), \
                 mock.patch.object(decoder_parity_v8.parity_test_v7, "read_dump_file", return_value=ck_dumps), \
                 mock.patch.object(decoder_parity_v8, "_load_llama_dump_dir", return_value=llama_dumps), \
                 mock.patch.object(decoder_parity_v8, "_compare_dump_sets", side_effect=_fake_compare):
                _, report = decoder_parity_v8._capture_dump_compare(
                    Path("/tmp/model.gguf"),
                    {"embed_dim": 2},
                    prefix,
                    9,
                    [33, 44],
                    tokens_before=[11, 22],
                    prefix_row_dim=2,
                    ctx_len=16,
                    top_k=2,
                    threads=1,
                    dump_root=tmp,
                    dump_names="Qcur-0",
                    dump_pass="decode",
                    dump_atol=1.0e-4,
                    dump_rtol=1.0e-3,
                    prefix_grid=(3, 3),
                    prefix_text_pos=5,
                )

            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["ck_prompt_start_token"], 11)
            self.assertEqual(report["llama_prompt_start_token"], 5)
            ck_rows = captured["ck_rows"]
            llama_rows = captured["llama_rows"]
            self.assertIsInstance(ck_rows, list)
            self.assertIsInstance(llama_rows, list)
            self.assertEqual([d.token_id for d in ck_rows], [0, 1])
            self.assertEqual([d.token_id for d in llama_rows], [0, 1])

    def test_capture_dump_compare_skips_prefix_for_prefill_pass(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_decoder_dump_prefix_skip_") as tmpdir:
            tmp = Path(tmpdir)
            prefix = array("f", [0.0] * 8)
            with mock.patch.object(
                decoder_parity_v8.bridge_runner_v8,
                "_run_decoder",
                return_value={"vocab_size": 4, "logits": array("f", [0.0, 0.0, 0.0, 0.0])},
            ) as run_decoder, \
                 mock.patch.object(decoder_parity_v8, "_run_llama_capture") as llama_capture:
                ck, report = decoder_parity_v8._capture_dump_compare(
                    Path("/tmp/model.gguf"),
                    {"embed_dim": 4},
                    prefix,
                    2,
                    [1, 2],
                    prefix_row_dim=4,
                    ctx_len=8,
                    top_k=2,
                    threads=1,
                    dump_root=tmp,
                    dump_names="Qcur-0",
                    dump_pass="prefill",
                    dump_atol=1.0e-4,
                    dump_rtol=1.0e-3,
                )

            self.assertEqual(ck["vocab_size"], 4)
            self.assertEqual(report["status"], "skipped")
            self.assertIn("dump-pass decode", report["reason"])
            run_decoder.assert_called_once()
            llama_capture.assert_not_called()

    def test_main_replays_segmented_prompt_from_bridge_report(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_decoder_bridge_report_") as tmpdir:
            tmp = Path(tmpdir)
            report_path = tmp / "report.json"
            bridge_report_path = tmp / "bridge_report.json"
            prefix_path = tmp / "prefix.f32"
            fake_gguf = tmp / "decoder.gguf"
            prefix_path.write_bytes(array("f", [0.0] * (9 * 64)).tobytes())
            bridge_report_path.write_text(
                json.dumps(
                    {
                        "decoder_runtime": {"gguf": str(fake_gguf)},
                        "decoder_context_len": 57,
                        "prompt": "Explain this image.",
                        "formatted_prompt": "<|im_start|>user\n<|vision_start|><image_embeds><|vision_end|>Explain this image.<|im_end|>\n<|im_start|>assistant\n",
                        "prompt_tokens_before_image": [11, 22],
                        "prompt_tokens_after_image": [33, 44, 55],
                        "multimodal_prompt_segmented": True,
                        "prefix_dump_path": str(prefix_path),
                        "prefix_grid_x": 3,
                        "prefix_grid_y": 3,
                        "prefix_text_pos": 5,
                    }
                ),
                encoding="utf-8",
            )

            class FakeTokenizer:
                def decode(self, ids: list[int], skip_special: bool = False) -> str:
                    return ",".join(str(x) for x in ids)

            fake_runtime = {
                "embed_dim": 16,
                "input_embed_dim": 64,
                "vocab_size": 4,
                "so_path": tmp / "libdecoder_v8.so",
                "c_path": tmp / "decoder_v8.c",
            }

            with mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_prepare_decoder_runtime", return_value=fake_runtime), \
                 mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_run_decoder", return_value={"vocab_size": 4, "logits": array("f", [0.1, 0.9, 0.2, -0.4])}) as run_decoder, \
                 mock.patch.object(decoder_parity_v8.GGUFTokenizer, "from_gguf", return_value=FakeTokenizer()), \
                 mock.patch.object(
                     decoder_parity_v8,
                     "_run_llama_capture",
                     return_value={
                         "meta": {
                             "ok": True,
                             "n_vocab": 4,
                             "token_count": 5,
                             "token_count_before": 2,
                             "token_count_after": 3,
                             "prefix_token_count": 9,
                             "prefix_position_count": 3,
                             "prefix_start_pos": 2,
                             "prefix_text_pos": 5,
                             "topk": [{"id": 1, "logit": 0.95}, {"id": 2, "logit": 0.18}],
                         },
                         "logits": np.array([0.0, 1.0, 0.1, -0.5], dtype=np.float32),
                     },
                 ) as llama_capture:
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = decoder_parity_v8.main(
                        [
                            "--bridge-report",
                            str(bridge_report_path),
                            "--workdir",
                            str(tmp / "work"),
                            "--json-out",
                            str(report_path),
                        ]
                    )

            self.assertEqual(rc, 0)
            self.assertEqual(llama_capture.call_args.args[0], fake_gguf.resolve())
            self.assertEqual(llama_capture.call_args.args[1], [33, 44, 55])
            self.assertEqual(llama_capture.call_args.args[2], 57)
            self.assertEqual(llama_capture.call_args.kwargs["tokens_before"], [11, 22])
            self.assertEqual(llama_capture.call_args.kwargs["prefix_grid"], (3, 3))
            self.assertEqual(llama_capture.call_args.kwargs["prefix_text_pos"], 5)
            self.assertEqual(Path(llama_capture.call_args.kwargs["prefix_path"]), prefix_path.resolve())
            _, decoder_kwargs = run_decoder.call_args
            self.assertEqual(decoder_kwargs["tokens_before"], [11, 22])
            self.assertEqual(decoder_kwargs["prefix_grid"], (3, 3))
            self.assertEqual(decoder_kwargs["prefix_text_pos"], 5)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(report["multimodal_prompt_segmented"])
            self.assertEqual(report["formatted_prompt"], "<|im_start|>user\n<|vision_start|><image_embeds><|vision_end|>Explain this image.<|im_end|>\n<|im_start|>assistant\n")
            self.assertEqual(report["prompt_tokens_before_image"], [11, 22])
            self.assertEqual(report["prompt_tokens_after_image"], [33, 44, 55])
            self.assertEqual(report["prefix"]["path"], str(prefix_path.resolve()))
            self.assertEqual(report["position_contract"]["ck"]["rows"][0], [2, 2, 2, 0])
            self.assertEqual(report["position_contract"]["ck"]["rows"][4], [2, 3, 3, 0])
            self.assertTrue(report["position_contract"]["rows_match"])
            self.assertTrue(report["position_contract"]["text_pos_match"])


if __name__ == "__main__":
    unittest.main()
