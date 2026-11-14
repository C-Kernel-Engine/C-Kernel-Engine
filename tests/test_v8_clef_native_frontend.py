"""Text/schema and native-feature boundary contracts; no synthetic E2E claim."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import subprocess
import shutil
import struct
from unittest.mock import patch
import numpy as np
from version.v8.decision import clef_text_frontend as frontend
from version.v8.scripts.native_decoder_features_v8 import final_feature_contract
from version.v8.scripts.native_decoder_features_v8 import DecoderFeatures


class CharacterTokenizer:
    def __call__(self, text, *, add_special_tokens=False):
        return SimpleNamespace(input_ids=list(map(ord, text)))


def request():
    return dict(
        id="example",
        state="A happy customer.",
        questions=dict(
            sentiment=dict(
                type="choice",
                instructions="Choose sentiment",
                criteria={"positive": "Good", "negative": "Bad"},
            )
        ),
    )


class FrontendTests(unittest.TestCase):
    def test_labels_are_bound_to_question_and_candidate_identities(self):
        from version.v8.decision.train_native_clef import label_indices

        results = [
            dict(question_id="a", candidate_ids=["no", "yes"]),
            dict(question_id="b", candidate_ids=["x", "y", "z"]),
        ]
        np.testing.assert_array_equal(
            label_indices(results, dict(a="yes", b="z")), [1, 2]
        )
        for labels in ({}, dict(a="yes"), dict(a="yes", b="bad"), dict(a=1, b="z")):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                label_indices(results, labels)

    def test_training_budget_rejected_before_runtime_construction(self):
        from version.v8.decision.train_clef_head import Runtime

        for limit in (0, -1, True, 1.0, 4 * 1024**3 + 1):
            with self.subTest(limit=limit), self.assertRaisesRegex(
                ValueError, "workspace limit"
            ):
                Runtime(None, {}, None, None, workspace_limit_bytes=limit)

    def test_spans_choices_and_full_text(self):
        record = request()
        encoded = frontend.prepare(CharacterTokenizer(), record, max_tokens=2048)
        text = "".join(map(chr, encoded.input_ids))
        q = encoded.questions[0]
        self.assertIn(record["state"], text)
        self.assertEqual(q.option_ids, ("negative", "positive"))
        self.assertEqual(text[slice(*q.question_span)], "Choose sentiment")
        for span, ident in zip(q.option_spans, q.option_ids):
            self.assertEqual(json.loads(text[slice(*span)])["option_id"], ident)
        self.assertTrue(text.endswith("JOINT SCHEMA DECISIONS:"))
        with self.assertRaises(ValueError):
            frontend.prepare(
                CharacterTokenizer(), record, max_tokens=len(encoded.input_ids) - 1
            )

    def test_no_media_or_empty_schema(self):
        for change in (dict(images=["x"]), dict(questions={})):
            with self.assertRaises(ValueError):
                frontend.prepare(
                    CharacterTokenizer(), dict(request(), **change), max_tokens=2048
                )

    def test_noul_and_score_options(self):
        self.assertEqual(
            [x[0] for x in frontend.question_options(dict(type="noul"))],
            ["true", "false"],
        )
        self.assertEqual(
            frontend.question_options(dict(type="score", criteria=["low", "high"])),
            [("0", "low"), ("1", "high")],
        )


class FeatureContracts(unittest.TestCase):
    def test_qwen_backbone_gate_selects_storage_contract_without_changing_fp32(self):
        from version.v8.decision.build_native_clef_backbone import ROOT
        import sys

        sys.path.insert(0, str(ROOT / "version/v8/scripts"))
        from build_ir_v8 import _resolve_manifest_execution_contracts

        circuit = json.loads((ROOT / "version/v8/circuits/qwen35.json").read_text())
        circuit["required_numerical_contracts"] = {
            k: v
            for k, v in circuit["required_numerical_contracts"].items()
            if k.startswith("decoder.attention_gate")
        }
        for dtype, expected in [
            ("bf16", "attn_gate_sigmoid_mul_pytorch_bf16_storage"),
            ("q4_k", "attn_gate_sigmoid_mul_forward"),
            ("fp32", "attn_gate_sigmoid_mul_forward"),
        ]:
            for phase in ("prefill", "decode"):
                with self.subTest(dtype=dtype, phase=phase):
                    plans = _resolve_manifest_execution_contracts(
                        dict(
                            template=circuit,
                            config=dict(recurrent_qkv_weight_dtype=dtype),
                        ),
                        phase,
                    )
                    self.assertEqual(len(plans), 1)
                    self.assertEqual(plans[0]["kernel"]["id"], expected)

    def test_qwen_backbone_rope_selects_storage_contract_without_changing_fp32(self):
        from version.v8.decision.build_native_clef_backbone import ROOT
        import sys

        sys.path.insert(0, str(ROOT / "version/v8/scripts"))
        from build_ir_v8 import _resolve_manifest_execution_contracts

        circuit = json.loads((ROOT / "version/v8/circuits/qwen35.json").read_text())
        circuit["required_numerical_contracts"] = {
            k: v
            for k, v in circuit["required_numerical_contracts"].items()
            if k.startswith("decoder.mrope")
        }
        for dtype, expected in [
            ("bf16", "mrope_qk_text_imrope_bf16_pytorch_storage"),
            ("q4_k", "mrope_qk_text"),
            ("fp32", "mrope_qk_text"),
        ]:
            for phase in ("prefill", "decode"):
                with self.subTest(dtype=dtype, phase=phase):
                    plans = _resolve_manifest_execution_contracts(
                        dict(
                            template=circuit,
                            config=dict(recurrent_qkv_weight_dtype=dtype),
                        ),
                        phase,
                    )
                    self.assertEqual(len(plans), 1)
                    self.assertEqual(plans[0]["kernel"]["id"], expected)

    def test_qwen_backbone_dt_gate_selects_storage_contract_without_changing_fp32(self):
        from version.v8.decision.build_native_clef_backbone import ROOT
        import sys

        sys.path.insert(0, str(ROOT / "version/v8/scripts"))
        from build_ir_v8 import _resolve_manifest_execution_contracts

        circuit = json.loads((ROOT / "version/v8/circuits/qwen35.json").read_text())
        circuit["required_numerical_contracts"] = {
            k: v
            for k, v in circuit["required_numerical_contracts"].items()
            if k.startswith("decoder.recurrent_dt_gate")
        }
        for dtype, expected in [
            ("bf16", "recurrent_dt_gate_forward_pytorch_fp32"),
            ("q4_k", None),
            ("fp32", None),
        ]:
            for phase in ("prefill", "decode"):
                with self.subTest(dtype=dtype, phase=phase):
                    plans = _resolve_manifest_execution_contracts(
                        dict(
                            template=circuit,
                            config=dict(recurrent_qkv_weight_dtype=dtype),
                        ),
                        phase,
                    )
                    self.assertEqual(len(plans), 1 if expected else 0)
                    if expected:
                        self.assertEqual(plans[0]["kernel"]["id"], expected)

    def test_native_dependencies_fail_before_execution(self):
        from version.v8.decision.run_native_clef import native_dependencies

        with patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(
            ValueError, "CK_SLEEF_LIBRARY"
        ):
            native_dependencies()

    def test_embedding_rows_follow_token_identity_and_reject_truncation(self):
        from version.v8.decision.run_native_clef import (
            embedding_rows,
            load_float_weights,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.safetensors.index.json").write_text(
                json.dumps(dict(weight_map={"lm_head.weight": "shard.safetensors"}))
            )
            # BF16 values 1,2,3,4; request rows in reversed token order.
            header = json.dumps(
                {
                    "lm_head.weight": dict(
                        dtype="BF16", shape=[2, 2], data_offsets=[0, 8]
                    )
                }
            ).encode()
            payload = (
                struct.pack("<Q", len(header))
                + header
                + struct.pack("<4H", 0x3F80, 0x4000, 0x4040, 0x4080)
            )
            shard = root / "shard.safetensors"
            shard.write_bytes(payload)
            np.testing.assert_array_equal(
                embedding_rows(root, [1, 0]), [[3, 4], [1, 2]]
            )
            np.testing.assert_array_equal(
                load_float_weights(shard)["lm_head.weight"], [[1, 2], [3, 4]]
            )
            with self.assertRaises(ValueError):
                embedding_rows(root, [2])
            shard.write_bytes(payload[:-1])
            with self.assertRaisesRegex(ValueError, "truncated"):
                embedding_rows(root, [1])

    def test_profile_changes_only_declared_projection_contracts(self):
        from version.v8.decision.build_native_clef_backbone import (
            author_profile,
            OLD,
            NEW,
            ROOT,
        )

        original = json.loads((ROOT / "version/v8/circuits/qwen35.json").read_text())
        actual = author_profile(original)
        (
            self.assertEqual(actual["body"], original["body"])
            if "body" in original
            else None
        )
        differences = []
        for name, entry in actual["required_numerical_contracts"].items():
            before = original["required_numerical_contracts"][name]
            if before != entry:
                differences.append(name)
                for phase, c in entry["phases"].items():
                    self.assertEqual(c["contract_id"], NEW)
                    self.assertEqual(before["phases"][phase]["contract_id"], OLD)
                entry["phases"] = before["phases"]
        self.assertEqual(len(differences), 2)
        self.assertEqual(actual, original)
        with self.assertRaises(ValueError):
            author_profile(dict(original, name="unknown"))

    def fixture(self):
        return dict(
            mode="decode",
            errors=[],
            memory=dict(
                activations=dict(buffers=[dict(name="hidden", dtype="fp32", size=16)])
            ),
            operations=[
                dict(
                    op="final_rmsnorm",
                    section="footer",
                    call_abi=dict(kernel_id="norm"),
                    args=[
                        dict(
                            name="output", source="output:output", buffer_ref="hidden"
                        ),
                        dict(name="d_model", expr="4"),
                        dict(name="aligned_embed_dim", expr="4"),
                    ],
                )
            ],
        )

    def test_declared_output_and_rejections(self):
        x = self.fixture()
        self.assertEqual(final_feature_contract(x)["channels"], 4)
        for change in ("dtype", "stride", "ambiguous", "errors"):
            x = self.fixture()
            if change == "dtype":
                x["memory"]["activations"]["buffers"][0]["dtype"] = "bf16"
            if change == "stride":
                x["operations"][0]["args"][-1]["expr"] = "8"
            if change == "ambiguous":
                x["operations"] *= 2
            if change == "errors":
                x["errors"] = ["unresolved"]
            with self.subTest(change=change), self.assertRaises(ValueError):
                final_feature_contract(x)

    def test_bundle_identity_before_execution(self):
        from version.v8.decision.run_native_clef import (
            BUNDLE_FILES,
            bind_bundle,
            verify_bundle,
            file_sha,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in BUNDLE_FILES:
                (root / name).write_bytes(b"fixture")
            lock = "a" * 64
            descriptor = root / "descriptor.json"
            descriptor.write_text(json.dumps(bind_bundle(root, lock)))
            digest = file_sha(descriptor)
            verify_bundle(root, descriptor, digest, lock)
            with self.assertRaises(ValueError):
                verify_bundle(root, descriptor, digest, "b" * 64)
            (root / "weights.bump").write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                verify_bundle(root, descriptor, digest, lock)


class OfflineHeadTests(unittest.TestCase):
    def test_complete_generated_head_and_failure_recovery(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable: same-input head oracle NOT_TESTED")
        from tests.test_v8_clef_flash_complete_head import synthetic_weights, reference
        from version.v8.decision.package_clef_head import build_package
        from version.v8.decision.run_native_clef import execute_offline_head
        from server.decision.native import NativeHead

        raw = synthetic_weights()
        inputs = dict(
            hidden=np.random.default_rng(811).normal(0, 0.2, (6, 8)).astype("f4"),
            input_ids=np.arange(1, 7, dtype="i4"),
            question_spans=np.array([[0, 2], [3, 6]], dtype="i4"),
            option_spans=np.array([[1, 3], [2, 5], [5, 6]], dtype="i4"),
            global_span=np.array([[5, 6]], dtype="i4"),
            option_counts=np.array([2, 1], dtype="i4"),
            question_type_ids=np.array([0, 2], dtype="i4"),
        )
        expected = reference(
            inputs["hidden"],
            inputs["input_ids"],
            inputs["question_spans"],
            inputs["option_spans"],
            inputs["option_counts"],
            inputs["question_type_ids"],
            raw,
        )["option_logits"].ravel()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "head"
            digest = build_package(
                folder,
                raw,
                [
                    dict(
                        id="test",
                        inputs=inputs,
                        question_index=0,
                        candidate_ids=["a", "b"],
                    )
                ],
                dict(id="test-head", revision="0" * 40),
                heads=2,
                provenance=dict(features="synthetic"),
            )
            head = NativeHead(folder / "package.json", digest)
            try:
                baseline = execute_offline_head(head, "test")
                np.testing.assert_allclose(baseline, expected, atol=2e-6, rtol=0)
                with patch.object(head, "execute", return_value=-1), self.assertRaises(
                    RuntimeError
                ):
                    execute_offline_head(head, "test")
                np.testing.assert_array_equal(
                    execute_offline_head(head, "test"), baseline
                )
            finally:
                head.close()
            with self.assertRaises(RuntimeError):
                execute_offline_head(head, "test")


class NativeFeatureLifecycle(unittest.TestCase):
    """A tiny native ABI fixture tests transport/state, not model arithmetic."""

    fixture = FeatureContracts.fixture

    def test_native_state_rejection_failure_and_recovery(self):
        if not shutil.which("cc"):
            self.skipTest("C compiler unavailable")
        source = r"""
        #include <stdint.h>
        #include "ck_decoder_features.h"
        size_t test_context_bytes(void) {return sizeof(CKDecoderFeatures);}
        #include <string.h>
        static float features[4];
        static int position, state, fail_init, frees;
        void test_fail_init(int enabled) {fail_init=enabled;}
        int test_frees(void) {return frees;}
        int ck_model_init_with_manifest(const char *a,const char *b) {return fail_init;}
        void ck_model_free(void) {frees++;}
        void ck_model_kv_cache_reset(void) {position=state=0;}
        int ck_model_get_context_window(void) {return 5;}
        int ck_model_get_vocab_size(void) {return 16;}
        int ck_model_get_active_tokens(void) {return position;}
        uintptr_t ck_model_get_named_activation_ptr(const char *name) {
            return !strcmp(name,"hidden") ? (uintptr_t)features : 0;
        }
        intptr_t ck_model_get_named_activation_nbytes(const char *name) {
            return !strcmp(name,"hidden") ? sizeof(features) : -1;
        }
        int ck_model_decode(int32_t token,float *logits) {
            state += token; position++;
            if(token==15) return -1;
            for(int i=0;i<4;i++) features[i]=state+i;
            return 0;
        }
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.c").write_text(source)
            subprocess.run(
                [
                    "cc",
                    "-shared",
                    "-fPIC",
                    "-o",
                    str(root / "libmodel.so"),
                    str(root / "model.c"),
                    "-I" + str(Path(__file__).resolve().parents[1] / "include"),
                    str(
                        Path(__file__).resolve().parents[1]
                        / "src/runtime/decoder_features_sequence.c"
                    ),
                ],
                check=True,
            )
            (root / "lowered_decode_call.json").write_text(json.dumps(self.fixture()))
            model = DecoderFeatures(root)
            try:
                with self.assertRaisesRegex(RuntimeError, "already has an owner"):
                    DecoderFeatures(root)
                baseline = model.encode([1, 2])
                np.testing.assert_array_equal(baseline, [[1, 2, 3, 4], [3, 4, 5, 6]])
                # Borrowed native pointers execute the complete token loop in C.
                import ctypes as C

                function = model.lib.decoder_features_sequence_checked_f32
                function.argtypes = [
                    C.c_void_p,
                    C.c_size_t,
                    C.POINTER(C.c_int32),
                    C.c_size_t,
                    C.c_size_t,
                    C.c_size_t,
                    C.POINTER(C.c_float),
                    C.c_size_t,
                    C.POINTER(C.c_float),
                    C.c_size_t,
                ]
                function.restype = C.c_int
                model.lib.test_context_bytes.restype = C.c_size_t
                ids = np.array([1, 2], dtype="i4")
                output = np.full((2, 4), -1, dtype="f4")
                scratch = np.empty_like(output)
                with model.borrow_native_descriptor() as descriptor:
                    self.assertEqual(
                        C.sizeof(descriptor), model.lib.test_context_bytes()
                    )
                    with self.assertRaises(RuntimeError):
                        model.close()
                    with self.assertRaises(RuntimeError):
                        model.encode([1])
                    with self.assertRaises(RuntimeError):
                        with model.borrow_native_descriptor():
                            pass
                    with patch.object(
                        model,
                        "_encode",
                        side_effect=AssertionError("Python token loop"),
                    ):
                        status = function(
                            C.byref(descriptor),
                            C.sizeof(descriptor),
                            ids.ctypes.data_as(C.POINTER(C.c_int32)),
                            ids.size,
                            2,
                            4,
                            output.ctypes.data_as(C.POINTER(C.c_float)),
                            output.size,
                            scratch.ctypes.data_as(C.POINTER(C.c_float)),
                            scratch.size,
                        )
                    self.assertEqual(status, 0)
                np.testing.assert_array_equal(output, baseline)
                self.assertEqual(model.lib.ck_model_get_active_tokens(), 0)
                model.encode([3, 4, 5, 6, 7])
                np.testing.assert_array_equal(model.encode([1, 2]), baseline)
                for bad in ([], [True], [-1], [16], [1] * 6):
                    with self.subTest(bad=bad), self.assertRaises(ValueError):
                        model.encode(bad)
                with self.assertRaises(RuntimeError):
                    model.encode([1, 15])
                self.assertEqual(model.lib.ck_model_get_active_tokens(), 0)
                np.testing.assert_array_equal(model.encode([1, 2]), baseline)
            finally:
                model.close()
            with self.assertRaises(RuntimeError):
                model.encode([1])
            with self.assertRaises(RuntimeError):
                with model.borrow_native_descriptor():
                    pass
            frees = model.lib.test_frees()
            model.lib.test_fail_init(1)
            with self.assertRaisesRegex(RuntimeError, "initialization failed"):
                DecoderFeatures(root)
            self.assertEqual(model.lib.test_frees(), frees + 1)
            model.lib.test_fail_init(0)
            recovered = DecoderFeatures(root)
            recovered.close()
            # A later writer of the same exported buffer must fail at loading.
            call = self.fixture()
            call["operations"].append(
                dict(args=[dict(source="output:other", buffer_ref="hidden")])
            )
            (root / "lowered_decode_call.json").write_text(json.dumps(call))
            with self.assertRaisesRegex(ValueError, "lifetime"):
                DecoderFeatures(root)


class NativeReleasedAcceptance(unittest.TestCase):
    def test_asset_backed_native_request_training_and_reload(self):
        import os
        from version.v8.decision.run_native_clef import run, file_sha
        from version.v8.decision.train_native_clef import train

        config = os.environ.get("CKE_CLEF_NATIVE_ACCEPTANCE")
        if not config:
            self.skipTest(
                "released 9B assets and explicit native acceptance configuration unavailable"
            )
        doc = json.loads(Path(config).read_text())
        required = {
            "release",
            "lock",
            "lock_sha",
            "bundle",
            "bundle_descriptor",
            "bundle_sha",
            "request",
            "labels",
        }
        self.assertEqual(set(doc), required)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            result = run(
                **{k: v for k, v in doc.items() if k != "labels"},
                output=folder / "native"
            )
            self.assertEqual(result["execution"], "PASS")
            self.assertEqual(result["numerical_parity"], "NOT_TESTED")
            trained = train(
                doc["release"],
                folder / "native",
                file_sha(folder / "native/result.json"),
                doc["labels"],
                folder / "training",
                workspace_mib=4096,
            )
            self.assertEqual(trained["execution"], "PASS")
            self.assertEqual(trained["backbone_training"], "FROZEN")
            self.assertTrue(trained["updated_parameter_tensors"])
            self.assertLessEqual(trained["reload_max_abs"], 2e-6)
            # A generated execution pass does not certify connected arithmetic.
            self.assertEqual(trained["numerical_gradient_parity"], "NOT_TESTED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
