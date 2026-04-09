#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version" / "v7" / "scripts"))

import build_ir_train_v7  # type: ignore
import codegen_train_runtime_v7  # type: ignore
import generate_train_layout_v7  # type: ignore
import lower_ir2_backward_v7  # type: ignore


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _entry(name: str) -> dict:
    return {
        "name": name,
        "dtype": "fp32",
        "shape": [1, 1],
        "offset": 0,
        "file_offset": 0,
        "size": 4,
    }


def _test_manifest(template_name: str) -> dict:
    template = _load_json(ROOT / "version" / "v7" / "templates" / f"{template_name}.json")
    return {
        "config": {
            "model": template_name,
            "num_layers": 1,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_heads": 8,
            "num_kv_heads": 4,
            "head_dim": 16,
            "aligned_head_dim": 16,
            "vocab_size": 512,
            "train_tokens": 2,
        },
        "template": template,
        "entries": [
            _entry("token_emb"),
            _entry("output.weight"),
            _entry("norm.weight"),
            _entry("layer.0.ln1_gamma"),
            _entry("layer.0.ln2_gamma"),
            _entry("layer.0.wq"),
            _entry("layer.0.wk"),
            _entry("layer.0.wv"),
            _entry("layer.0.q_norm"),
            _entry("layer.0.k_norm"),
            _entry("layer.0.wo"),
            _entry("layer.0.w1"),
            _entry("layer.0.w2"),
            _entry("layer.0.w3"),
        ],
    }


class TrainBridgeLoweringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = _load_json(ROOT / "version" / "v7" / "kernel_maps" / "KERNEL_REGISTRY.json")
        cls.bindings = _load_json(ROOT / "version" / "v7" / "kernel_maps" / "kernel_bindings.json")
        cls.grad_rules = _load_json(ROOT / "version" / "v7" / "scripts" / "grad_rules_v7.json")
        cls.manifest = _test_manifest("qwen3")

    def test_explicit_bridge_lowering_emits_bridge_ops_and_plans(self) -> None:
        ir1 = build_ir_train_v7.build_ir1_train(
            manifest=self.manifest,
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            max_layers=1,
            strict=False,
            bridge_lowering="explicit",
        )
        self.assertEqual(ir1.get("bridge_lowering"), "explicit")
        self.assertGreater(int((ir1.get("stats") or {}).get("bridge_ops", 0) or 0), 0)

        bridge_ops = [op for op in ir1.get("ops", []) if op.get("phase") == "bridge"]
        self.assertTrue(any(op.get("op") == "bridge_token_to_head_major" for op in bridge_ops))
        self.assertTrue(any(op.get("op") == "bridge_head_to_token_major" for op in bridge_ops))

        semantic = {
            str(op.get("op")): op
            for op in ir1.get("ops", [])
            if str(op.get("op")) in {"qk_norm", "rope_qk", "attn"}
        }
        self.assertIn("qk_norm", semantic)
        self.assertIn("rope_qk", semantic)
        self.assertIn("attn", semantic)
        for op_name in ("qk_norm", "rope_qk", "attn"):
            plan = semantic[op_name].get("bridge_plan")
            self.assertIsInstance(plan, dict)
            self.assertEqual(plan.get("mode"), "explicit")
            self.assertTrue(plan.get("pre"))
            self.assertTrue(plan.get("post"))
        attn_contract = semantic["attn"].get("runtime_contract")
        self.assertIsInstance(attn_contract, dict)
        self.assertEqual(attn_contract.get("kernel_id"), "attention_forward_causal_head_major_gqa_exact")
        self.assertTrue(bool(attn_contract.get("materialize_saved_attn_weights")))
        attn_plan_contract = (semantic["attn"].get("bridge_plan") or {}).get("runtime_contract")
        self.assertIsInstance(attn_plan_contract, dict)
        self.assertEqual(attn_plan_contract.get("kernel_id"), "attention_forward_causal_head_major_gqa_exact")
        for op_name in ("qk_norm", "rope_qk", "attn"):
            op_id = int(semantic[op_name].get("op_id", -1))
            plan = semantic[op_name].get("bridge_plan") or {}
            self.assertTrue(all(int(row.get("op_id", -1)) < op_id for row in plan.get("pre", [])))
            self.assertTrue(all(int(row.get("op_id", -1)) > op_id for row in plan.get("post", [])))

    def test_attention_runtime_contract_can_be_overridden_from_template_data(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        attention_contract = (((manifest.get("template") or {}).get("contract") or {}).get("attention_contract") or {})
        runtime_contract = attention_contract.setdefault("train_runtime_contract", {})
        runtime_contract["saved_tensor_kernel_overrides"] = {
            "attn_weights": "attention_forward_causal_head_major_gqa_flash_strided"
        }
        ir1 = build_ir_train_v7.build_ir1_train(
            manifest=manifest,
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            max_layers=1,
            strict=False,
            bridge_lowering="legacy",
        )
        attn_ops = [op for op in ir1.get("ops", []) if str(op.get("op")) == "attn"]
        self.assertEqual(len(attn_ops), 1)
        contract = attn_ops[0].get("runtime_contract") or {}
        self.assertEqual(contract.get("kernel_id"), "attention_forward_causal_head_major_gqa_flash_strided")
        self.assertEqual(contract.get("saved_tensor_kernel_key"), "attn_weights")
        self.assertEqual(contract.get("contract_source"), "template.contract.attention_contract.train_runtime_contract")

    def test_embedded_manifest_template_missing_train_runtime_contract_uses_compatibility_warning(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        attention_contract = (((manifest.get("template") or {}).get("contract") or {}).get("attention_contract") or {})
        attention_contract.pop("train_runtime_contract", None)
        ir1 = build_ir_train_v7.build_ir1_train(
            manifest=manifest,
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            max_layers=1,
            strict=False,
            bridge_lowering="legacy",
        )
        attn_ops = [op for op in ir1.get("ops", []) if str(op.get("op")) == "attn"]
        self.assertEqual(len(attn_ops), 1)
        contract = attn_ops[0].get("runtime_contract") or {}
        self.assertEqual(contract.get("kernel_id"), "attention_forward_causal_head_major_gqa_exact")
        warnings = ir1.get("warnings", [])
        self.assertTrue(
            any("Embedded manifest template" in str(item) and "train_runtime_contract" in str(item) for item in warnings)
        )

    def test_repo_template_requires_explicit_train_runtime_contract(self) -> None:
        template = copy.deepcopy(self.manifest.get("template") or {})
        attention_contract = (((template.get("contract") or {}).get("attention_contract") or {}))
        attention_contract.pop("train_runtime_contract", None)
        base_manifest = {
            "config": copy.deepcopy(self.manifest.get("config") or {}),
            "entries": copy.deepcopy(self.manifest.get("entries") or []),
        }
        original_load_json = build_ir_train_v7._load_json

        def fake_load_json(path: Path) -> dict:
            if Path(path).name == "qwen3.json":
                return copy.deepcopy(template)
            return original_load_json(path)

        with mock.patch.object(build_ir_train_v7, "_load_json", side_effect=fake_load_json):
            with self.assertRaisesRegex(
                RuntimeError,
                "Repo template `qwen3` must define contract.attention_contract.train_runtime_contract",
            ):
                build_ir_train_v7.build_ir1_train(
                    manifest=base_manifest,
                    registry=self.registry,
                    bindings_doc=self.bindings,
                    grad_rules=self.grad_rules,
                    max_layers=1,
                    strict=False,
                    bridge_lowering="legacy",
                )

    def test_gemma_sliding_attention_contract_uses_template_runtime_policy(self) -> None:
        manifest = _test_manifest("gemma3")
        ir1 = build_ir_train_v7.build_ir1_train(
            manifest=manifest,
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            max_layers=1,
            strict=False,
            bridge_lowering="explicit",
        )
        attn_ops = [op for op in ir1.get("ops", []) if str(op.get("op")) == "attn_sliding"]
        self.assertEqual(len(attn_ops), 1)
        contract = attn_ops[0].get("runtime_contract") or {}
        self.assertEqual(contract.get("kernel_id"), "attention_forward_causal_head_major_gqa_exact")
        self.assertTrue(bool(contract.get("requires_zero_sliding_window")))

    def test_ir2_propagates_bridge_lowering(self) -> None:
        ir1 = build_ir_train_v7.build_ir1_train(
            manifest=self.manifest,
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            max_layers=1,
            strict=False,
            bridge_lowering="explicit",
        )
        ir2 = lower_ir2_backward_v7.synthesize_ir2_backward(
            ir1=ir1,
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            strict=False,
            allow_partial=True,
            checkpoint_policy="none",
        )
        self.assertEqual(ir2.get("bridge_lowering"), "explicit")
        bridge_ops = {
            str(op.get("op")): op
            for op in ir2.get("backward", [])
            if str(op.get("op")) in {
                "attn_backward_core",
                "rope_qk_backward_core",
                "qk_norm_backward_core",
            }
        }
        self.assertIn("attn_backward_core", bridge_ops)
        self.assertIn("rope_qk_backward_core", bridge_ops)
        self.assertIn("qk_norm_backward_core", bridge_ops)
        for op_name in ("attn_backward_core", "rope_qk_backward_core", "qk_norm_backward_core"):
            plan = bridge_ops[op_name].get("bridge_plan")
            self.assertIsInstance(plan, dict)
            self.assertEqual(plan.get("mode"), "explicit")
            self.assertTrue(plan.get("pre"))
            self.assertTrue(plan.get("post"))

    def test_codegen_uses_explicit_forward_bridge_plans(self) -> None:
        ir1 = build_ir_train_v7.build_ir1_train(
            manifest=self.manifest,
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            max_layers=1,
            strict=False,
            bridge_lowering="explicit",
        )
        ir2 = lower_ir2_backward_v7.synthesize_ir2_backward(
            ir1=ir1,
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            strict=False,
            allow_partial=True,
            checkpoint_policy="none",
        )
        layout = generate_train_layout_v7.build_layout(ir2, self.manifest, 64, strict=False)
        c_src, summary = codegen_train_runtime_v7.generate_c(
            ir2,
            self.registry,
            manifest=self.manifest,
            layout=layout,
            exec_plan=None,
        )
        self.assertEqual(summary.get("bridge_lowering"), "explicit")
        self.assertGreaterEqual(int(summary.get("explicit_forward_bridge_plans", 0) or 0), 3)
        self.assertGreaterEqual(int(summary.get("explicit_backward_bridge_plans", 0) or 0), 3)
        self.assertIn("explicit bridge plan via shared attention_forward bridge", c_src)
        self.assertIn("explicit backward bridge plan via shared", c_src)
        self.assertIn("runtime_kernel_id=attention_forward_causal_head_major_gqa_exact", c_src)
        self.assertIn("attention_forward_causal_head_major_gqa_exact(", c_src)

    def test_legacy_codegen_honors_attention_runtime_contract(self) -> None:
        ir1 = build_ir_train_v7.build_ir1_train(
            manifest=self.manifest,
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            max_layers=1,
            strict=False,
            bridge_lowering="legacy",
        )
        attn_ops = [op for op in ir1.get("ops", []) if str(op.get("op")) == "attn"]
        self.assertEqual(len(attn_ops), 1)
        self.assertEqual((attn_ops[0].get("runtime_contract") or {}).get("kernel_id"), "attention_forward_causal_head_major_gqa_exact")
        ir2 = lower_ir2_backward_v7.synthesize_ir2_backward(
            ir1=ir1,
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            strict=False,
            allow_partial=True,
            checkpoint_policy="none",
        )
        layout = generate_train_layout_v7.build_layout(ir2, self.manifest, 64, strict=False)
        c_src, _summary = codegen_train_runtime_v7.generate_c(
            ir2,
            self.registry,
            manifest=self.manifest,
            layout=layout,
            exec_plan=None,
        )
        self.assertIn("runtime_kernel_id=attention_forward_causal_head_major_gqa_exact", c_src)
        self.assertIn("attention_forward_causal_head_major_gqa_exact(", c_src)

    def test_legacy_codegen_binds_rmsnorm_rstd_cache_for_training(self) -> None:
        ir1 = build_ir_train_v7.build_ir1_train(
            manifest=self.manifest,
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            max_layers=1,
            strict=False,
            bridge_lowering="legacy",
        )
        ir2 = lower_ir2_backward_v7.synthesize_ir2_backward(
            ir1=ir1,
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            strict=False,
            allow_partial=True,
            checkpoint_policy="none",
        )
        layout = generate_train_layout_v7.build_layout(ir2, self.manifest, 64, strict=False)
        c_src, _summary = codegen_train_runtime_v7.generate_c(
            ir2,
            self.registry,
            manifest=self.manifest,
            layout=layout,
            exec_plan=None,
        )
        self.assertIn(
            "rmsnorm_forward(act_Sheader_dense_embedding_lookup_0_out, weight_layer_0_ln1_gamma, act_L0_rmsnorm_0_output, act_L0_rmsnorm_0_rstd_cache",
            c_src,
        )

    def test_legacy_codegen_uses_shared_qk_norm_bridge_emitter(self) -> None:
        ir1 = build_ir_train_v7.build_ir1_train(
            manifest=self.manifest,
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            max_layers=1,
            strict=False,
            bridge_lowering="legacy",
        )
        ir2 = lower_ir2_backward_v7.synthesize_ir2_backward(
            ir1=ir1,
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            strict=False,
            allow_partial=True,
            checkpoint_policy="none",
        )
        layout = generate_train_layout_v7.build_layout(ir2, self.manifest, 64, strict=False)
        c_src, _summary = codegen_train_runtime_v7.generate_c(
            ir2,
            self.registry,
            manifest=self.manifest,
            layout=layout,
            exec_plan=None,
        )
        self.assertIn("token-major IR <-> head-major kernel bridge via shared qk_norm bridge", c_src)
        self.assertIn("qk_norm_forward(", c_src)

    def test_legacy_codegen_uses_shared_rope_forward_bridge_emitter(self) -> None:
        ir1 = build_ir_train_v7.build_ir1_train(
            manifest=self.manifest,
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            max_layers=1,
            strict=False,
            bridge_lowering="legacy",
        )
        ir2 = lower_ir2_backward_v7.synthesize_ir2_backward(
            ir1=ir1,
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            strict=False,
            allow_partial=True,
            checkpoint_policy="none",
        )
        layout = generate_train_layout_v7.build_layout(ir2, self.manifest, 64, strict=False)
        c_src, _summary = codegen_train_runtime_v7.generate_c(
            ir2,
            self.registry,
            manifest=self.manifest,
            layout=layout,
            exec_plan=None,
        )
        self.assertIn("IR token-major <-> kernel head-major bridge via shared rope_forward_qk bridge", c_src)
        self.assertIn("rope_forward_qk_with_rotary_dim(", c_src)

    def test_legacy_codegen_uses_shared_rope_backward_bridge_emitter(self) -> None:
        ir1 = build_ir_train_v7.build_ir1_train(
            manifest=self.manifest,
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            max_layers=1,
            strict=False,
            bridge_lowering="legacy",
        )
        ir2 = lower_ir2_backward_v7.synthesize_ir2_backward(
            ir1=ir1,
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            strict=False,
            allow_partial=True,
            checkpoint_policy="none",
        )
        layout = generate_train_layout_v7.build_layout(ir2, self.manifest, 64, strict=False)
        c_src, _summary = codegen_train_runtime_v7.generate_c(
            ir2,
            self.registry,
            manifest=self.manifest,
            layout=layout,
            exec_plan=None,
        )
        self.assertIn("token-major IR <-> head-major kernel bridge via shared rope_backward_qk bridge", c_src)
        self.assertIn("rope_backward_qk(", c_src)

    def test_legacy_codegen_uses_shared_qk_norm_backward_bridge_emitter(self) -> None:
        ir1 = build_ir_train_v7.build_ir1_train(
            manifest=self.manifest,
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            max_layers=1,
            strict=False,
            bridge_lowering="legacy",
        )
        ir2 = lower_ir2_backward_v7.synthesize_ir2_backward(
            ir1=ir1,
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            strict=False,
            allow_partial=True,
            checkpoint_policy="none",
        )
        layout = generate_train_layout_v7.build_layout(ir2, self.manifest, 64, strict=False)
        c_src, _summary = codegen_train_runtime_v7.generate_c(
            ir2,
            self.registry,
            manifest=self.manifest,
            layout=layout,
            exec_plan=None,
        )
        self.assertIn("token-major IR <-> head-major kernel bridge via shared qk_norm_backward bridge", c_src)
        self.assertIn("qk_norm_backward(", c_src)

    def test_legacy_codegen_uses_shared_attention_backward_bridge_emitter(self) -> None:
        ir1 = build_ir_train_v7.build_ir1_train(
            manifest=self.manifest,
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            max_layers=1,
            strict=False,
            bridge_lowering="legacy",
        )
        ir2 = lower_ir2_backward_v7.synthesize_ir2_backward(
            ir1=ir1,
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            strict=False,
            allow_partial=True,
            checkpoint_policy="none",
        )
        layout = generate_train_layout_v7.build_layout(ir2, self.manifest, 64, strict=False)
        c_src, _summary = codegen_train_runtime_v7.generate_c(
            ir2,
            self.registry,
            manifest=self.manifest,
            layout=layout,
            exec_plan=None,
        )
        self.assertIn("token-major IR <-> head-major kernel bridge via shared attention_backward bridge", c_src)
        self.assertIn("attention_backward_causal_head_major_gqa(", c_src)

    def test_checkpoint_policy_recompute_attn_inserts_rematerialization(self) -> None:
        ir1 = build_ir_train_v7.build_ir1_train(
            manifest=self.manifest,
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            max_layers=1,
            strict=False,
            bridge_lowering="explicit",
        )
        ir2 = lower_ir2_backward_v7.synthesize_ir2_backward(
            ir1=ir1,
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            strict=False,
            allow_partial=True,
            checkpoint_policy="recompute_attn",
        )
        self.assertEqual(ir2.get("checkpoint_policy"), "recompute_attn")
        self.assertEqual((ir2.get("checkpoint_summary") or {}).get("checkpoint_rematerialize_ops"), 1)
        fwd_attn = [op for op in ir2.get("forward", []) if str(op.get("op")) == "attn"]
        self.assertEqual(len(fwd_attn), 1)
        fwd_contract = fwd_attn[0].get("runtime_contract") or {}
        self.assertEqual(fwd_contract.get("kernel_id"), "attention_forward_causal_head_major_gqa_flash_strided")
        self.assertFalse(bool(fwd_contract.get("materialize_saved_attn_weights")))
        remat_ops = [op for op in ir2.get("backward", []) if str(op.get("op")) == "checkpoint_rematerialize_saved_tensor"]
        self.assertEqual(len(remat_ops), 1)
        self.assertEqual((remat_ops[0].get("runtime_contract") or {}).get("kernel_id"), "attention_forward_causal_head_major_gqa_exact")
        self.assertEqual((remat_ops[0].get("rematerialize_contract") or {}).get("saved_key"), "attn_weights")
        self.assertEqual((remat_ops[0].get("bridge_plan") or {}).get("mode"), "explicit")
        ir1_attn = [op for op in ir1.get("ops", []) if str(op.get("op")) == "attn"]
        self.assertEqual(len(ir1_attn), 1)
        saved_tid = str(((ir1_attn[0].get("save_for_backward") or {}).get("attn_weights") or {}).get("tensor", ""))
        self.assertTrue(saved_tid)
        saved_meta = (ir2.get("tensors") or {}).get(saved_tid) or {}
        self.assertEqual(saved_meta.get("kind"), "aux")
        self.assertFalse(bool(saved_meta.get("persistent")))
        attn_core = [op for op in ir2.get("backward", []) if str(op.get("op")) == "attn_backward_core"]
        self.assertEqual(len(attn_core), 1)
        d_scores_tid = (((attn_core[0].get("dataflow") or {}).get("outputs") or {}).get("d_scores") or {}).get("tensor")
        self.assertTrue(isinstance(d_scores_tid, str) and d_scores_tid)
        d_scores_meta = (ir2.get("tensors") or {}).get(d_scores_tid) or {}
        self.assertEqual(d_scores_meta.get("shape"), saved_meta.get("shape"))
        self.assertEqual(d_scores_meta.get("numel"), saved_meta.get("numel"))

    def test_layout_and_codegen_reflect_recompute_attn_policy(self) -> None:
        ir1 = build_ir_train_v7.build_ir1_train(
            manifest=self.manifest,
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            max_layers=1,
            strict=False,
            bridge_lowering="explicit",
        )
        ir2_none = lower_ir2_backward_v7.synthesize_ir2_backward(
            ir1=ir1,
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            strict=False,
            allow_partial=True,
            checkpoint_policy="none",
        )
        ir2_recompute = lower_ir2_backward_v7.synthesize_ir2_backward(
            ir1=ir1,
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            strict=False,
            allow_partial=True,
            checkpoint_policy="recompute_attn",
        )
        layout_none = generate_train_layout_v7.build_layout(ir2_none, self.manifest, 64, strict=False)
        layout_recompute = generate_train_layout_v7.build_layout(ir2_recompute, self.manifest, 64, strict=False)
        self.assertEqual(layout_recompute.get("checkpoint_policy"), "recompute_attn")
        saved_bytes_none = int(((layout_none.get("summary") or {}).get("region_bytes") or {}).get("saved", 0) or 0)
        saved_bytes_recompute = int(((layout_recompute.get("summary") or {}).get("region_bytes") or {}).get("saved", 0) or 0)
        self.assertGreater(saved_bytes_none, saved_bytes_recompute)
        c_src, summary = codegen_train_runtime_v7.generate_c(
            ir2_recompute,
            self.registry,
            manifest=self.manifest,
            layout=layout_recompute,
            exec_plan=None,
        )
        self.assertGreaterEqual(int(summary.get("checkpoint_rematerialize_ops", 0) or 0), 1)
        self.assertIn("checkpoint_rematerialize_saved_tensor", c_src)
        self.assertIn("explicit checkpoint rematerialization via shared attention_forward bridge", c_src)

    def test_codegen_uses_shared_global_grad_norm_helper(self) -> None:
        ir1 = build_ir_train_v7.build_ir1_train(
            manifest=self.manifest,
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            max_layers=1,
            strict=False,
            bridge_lowering="explicit",
        )
        ir2 = lower_ir2_backward_v7.synthesize_ir2_backward(
            ir1=ir1,
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            strict=False,
            allow_partial=True,
            checkpoint_policy="none",
        )
        layout = generate_train_layout_v7.build_layout(ir2, self.manifest, 64, strict=False)
        c_src, _summary = codegen_train_runtime_v7.generate_c(
            ir2,
            self.registry,
            manifest=self.manifest,
            layout=layout,
            exec_plan=None,
        )
        self.assertIn("gradient_global_norm_multi_f32(grads, numels,", c_src)
        self.assertNotIn("double gv = (double)", c_src)


if __name__ == "__main__":
    unittest.main()
