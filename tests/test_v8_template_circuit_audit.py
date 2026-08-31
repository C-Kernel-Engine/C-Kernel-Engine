from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "version/v8/scripts/audit_template_circuit_v8.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_template_circuit_v8", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _op(op_id: int, layer: int, op: str, inputs: dict, outputs: dict) -> dict:
    return {
        "op_id": op_id,
        "layer": layer,
        "op": op,
        "kernel": op,
        "dataflow": {"inputs": inputs, "outputs": outputs},
    }


class TemplateCircuitAuditTests(unittest.TestCase):
    def test_mamba_in_proj_must_consume_block_rmsnorm_output(self) -> None:
        audit = _load_module()
        ir = {
            "ops": [
                _op(0, -1, "dense_embedding_lookup", {}, {"out": {"slot": "main_stream", "dtype": "fp32"}}),
                _op(1, 0, "residual_save", {"src": {"from_op": 0, "slot": "main_stream"}}, {"dst": {"slot": "residual", "dtype": "fp32"}}),
                _op(2, 0, "block_rmsnorm", {"input": {"from_op": 0, "slot": "main_stream"}}, {"output": {"slot": "layer_input", "dtype": "fp32"}}),
                _op(3, 0, "mamba_in_proj", {"x": {"from_op": 0, "slot": "main_stream"}}, {"y": {"slot": "recurrent_packed", "dtype": "fp32"}}),
            ]
        }
        errors = audit.audit_ir1(ir)
        self.assertTrue(any("mamba_in_proj.x" in e and "expected block_rmsnorm" in e for e in errors), errors)

    def test_fixed_mamba_circuit_and_lowered_buffers_pass(self) -> None:
        audit = _load_module()
        ir = {
            "ops": [
                _op(0, -1, "dense_embedding_lookup", {}, {"out": {"slot": "main_stream", "dtype": "fp32"}}),
                _op(1, 0, "residual_save", {"src": {"from_op": 0, "slot": "main_stream"}}, {"dst": {"slot": "residual", "dtype": "fp32"}}),
                _op(2, 0, "block_rmsnorm", {"input": {"from_op": 0, "slot": "main_stream"}}, {"output": {"slot": "layer_input", "dtype": "fp32"}}),
                _op(3, 0, "mamba_in_proj", {"x": {"from_op": 2, "slot": "layer_input"}}, {"y": {"slot": "recurrent_packed", "dtype": "fp32"}}),
                _op(4, 0, "mamba_in_proj_split", {"projected": {"from_op": 3, "slot": "recurrent_packed"}}, {"gate": {"slot": "recurrent_z"}, "hidden_bc": {"slot": "recurrent_conv_qkv"}, "dt": {"slot": "recurrent_g"}}),
            ]
        }
        lowered = {
            "operations": [
                {"idx": 2, "layer": 0, "op": "block_rmsnorm", "activations": {"input": {"buffer": "embedded_input"}}, "outputs": {"output": {"buffer": "layer_input"}}},
                {"idx": 3, "layer": 0, "op": "mamba_in_proj", "activations": {"x": {"buffer": "layer_input"}}, "outputs": {"y": {"buffer": "recurrent_packed"}}},
                {"idx": 4, "layer": 0, "op": "mamba_in_proj_split", "activations": {"projected": {"buffer": "recurrent_packed"}}, "outputs": {}},
            ]
        }
        self.assertEqual(audit.audit_ir1(ir), [])
        self.assertEqual(audit.audit_lowered(lowered), [])

    def test_template_explicit_edge_audit_reports_missing_and_present_edges(self) -> None:
        audit = _load_module()
        template = {
            "name": "synthetic",
            "block_types": {
                "decoder": {
                    "body": {
                        "ops": [
                            "rmsnorm",
                            {"op": "qkv_proj", "graph_slots": {"inputs": {"x": "main_stream"}}},
                            "mlp_gate_up",
                        ]
                    }
                }
            },
        }
        report = audit.audit_template_explicit_edges(template)
        self.assertEqual(report["explicit"], ["decoder.body[1].qkv_proj.x=main_stream"])
        self.assertEqual(report["missing"], ["decoder.body[2].mlp_gate_up.x"])

    def test_template_edge_audit_reports_canonical_linear_a_port(self) -> None:
        audit = _load_module()
        template = {
            "name": "synthetic",
            "block_types": {
                "decoder": {
                    "body": {
                        "ops": [
                            {
                                "op": "mlp_down",
                                "graph_slots": {"inputs": {"A": "mlp_scratch"}},
                            }
                        ]
                    }
                }
            },
        }
        report = audit.audit_template_explicit_edges(template)
        self.assertEqual(report["explicit"], ["decoder.body[0].mlp_down.A=mlp_scratch"])
        self.assertEqual(report["missing"], [])

    def test_glm4_template_declares_critical_projection_edges(self) -> None:
        audit = _load_module()
        template = json.loads((REPO / "version/v8/circuits/glm4.json").read_text(encoding="utf-8"))
        report = audit.audit_template_explicit_edges(template)
        self.assertIn("decoder.body[1].qkv_proj.x=main_stream", report["explicit"])
        self.assertIn("decoder.body[4].out_proj.x=attn_scratch", report["explicit"])
        self.assertIn("decoder.body[8].mlp_gate_up.x=main_stream", report["explicit"])
        self.assertIn("decoder.body[10].mlp_down.A=mlp_scratch", report["explicit"])
        self.assertIn("decoder.footer[2].logits.x=main_stream", report["explicit"])
        self.assertEqual(report["missing"], [])


    def test_kimi_vl_template_declares_mla_and_moe_edges(self) -> None:
        audit = _load_module()
        template = json.loads((REPO / "version/v8/circuits/kimi_vl.json").read_text(encoding="utf-8"))
        report = audit.audit_template_explicit_edges(template)
        explicit = set(report["explicit"])
        self.assertIn("decoder.body:mla_dense_mlp[2].q_proj.x=layer_input", explicit)
        self.assertIn("decoder.body:mla_dense_mlp[3].kv_a_proj.x=layer_input", explicit)
        self.assertIn("decoder.body:mla_dense_mlp[5].kv_lora_decompress.compressed_kv=compressed_kv_normed", explicit)
        self.assertIn("decoder.body:mla_dense_mlp[6].partial_rope_concat.q_packed=q_scratch", explicit)
        self.assertIn("decoder.body:mla_dense_mlp[7].mla_attention.q=q_scratch", explicit)
        self.assertIn("decoder.body:mla_dense_mlp[7].mla_attention.k=k_scratch", explicit)
        self.assertIn("decoder.body:mla_dense_mlp[7].mla_attention.v=v_scratch", explicit)
        self.assertIn("decoder.body:mla_moe[14].moe_swiglu_expert_mlp.routing_weights=k_scratch", explicit)
        self.assertIn("decoder.body:mla_moe[15].shared_swiglu_expert_mlp.routed=mlp_scratch", explicit)
        self.assertIn("decoder.footer[2].logits.x=main_stream", explicit)
        self.assertEqual(report["missing"], [])

    def test_mla_dataflow_must_consume_normed_compressed_kv(self) -> None:
        audit = _load_module()
        ir = {
            "ops": [
                _op(0, -1, "dense_embedding_lookup", {}, {"out": {"slot": "main_stream", "dtype": "fp32"}}),
                _op(1, 0, "residual_save", {"src": {"from_op": 0, "slot": "main_stream"}}, {"dst": {"slot": "residual", "dtype": "fp32"}}),
                _op(2, 0, "block_rmsnorm", {"input": {"from_op": 0, "slot": "main_stream"}}, {"output": {"slot": "layer_input", "dtype": "fp32"}}),
                _op(3, 0, "q_proj", {"x": {"from_op": 2, "slot": "layer_input"}}, {"y": {"slot": "q_proj", "dtype": "fp32"}}),
                _op(4, 0, "kv_a_proj", {"x": {"from_op": 2, "slot": "layer_input"}}, {"y": {"slot": "compressed_kv", "dtype": "fp32"}}),
                _op(5, 0, "kv_a_layernorm", {"input": {"from_op": 4, "slot": "compressed_kv"}}, {"y": {"slot": "compressed_kv_normed", "dtype": "fp32"}}),
                _op(6, 0, "kv_lora_decompress", {"compressed_kv": {"from_op": 4, "slot": "compressed_kv"}}, {"k_nope": {"slot": "k_nope"}, "value": {"slot": "mla_value"}}),
            ]
        }
        errors = audit.audit_ir1(ir)
        self.assertTrue(any("kv_lora_decompress.compressed_kv" in e and "expected kv_a_layernorm" in e for e in errors), errors)

    def test_mla_dataflow_passes_for_normed_kv_and_partial_rope(self) -> None:
        audit = _load_module()
        ir = {
            "ops": [
                _op(0, -1, "dense_embedding_lookup", {}, {"out": {"slot": "main_stream", "dtype": "fp32"}}),
                _op(1, 0, "residual_save", {"src": {"from_op": 0, "slot": "main_stream"}}, {"dst": {"slot": "residual", "dtype": "fp32"}}),
                _op(2, 0, "block_rmsnorm", {"input": {"from_op": 0, "slot": "main_stream"}}, {"output": {"slot": "layer_input", "dtype": "fp32"}}),
                _op(3, 0, "q_proj", {"x": {"from_op": 2, "slot": "layer_input"}}, {"y": {"slot": "q_proj", "dtype": "fp32"}}),
                _op(4, 0, "kv_a_proj", {"x": {"from_op": 2, "slot": "layer_input"}}, {"y": {"slot": "compressed_kv", "dtype": "fp32"}}),
                _op(5, 0, "kv_a_layernorm", {"input": {"from_op": 4, "slot": "compressed_kv"}}, {"y": {"slot": "compressed_kv_normed", "dtype": "fp32"}}),
                _op(6, 0, "kv_lora_decompress", {"compressed_kv": {"from_op": 5, "slot": "compressed_kv_normed"}}, {"k_nope": {"slot": "k_nope"}, "value": {"slot": "mla_value"}}),
                _op(7, 0, "partial_rope_concat", {"q_packed": {"from_op": 3, "slot": "q_scratch"}, "k_nope": {"from_op": 6, "slot": "k_nope"}, "k_pe": {"from_op": 4, "slot": "compressed_kv"}}, {"query": {"slot": "q_scratch"}, "key": {"slot": "k_scratch"}}),
                _op(8, 0, "mla_attention", {"q": {"from_op": 7, "slot": "q_scratch"}, "k": {"from_op": 7, "slot": "k_scratch"}, "v": {"from_op": 6, "slot": "v_scratch"}}, {"out": {"slot": "attn_scratch"}}),
            ]
        }
        self.assertEqual(audit.audit_ir1(ir), [])

    def test_lowered_rejects_quantized_activation_bound_to_fp32_stream(self) -> None:
        audit = _load_module()
        lowered = {
            "operations": [
                {
                    "idx": 4,
                    "layer": 0,
                    "op": "q_proj",
                    "activations": {"x": {"dtype": "q8_k", "buffer": "embedded_input"}},
                    "outputs": {"y": {"buffer": "q_scratch"}},
                }
            ]
        }
        errors = audit.audit_lowered(lowered)
        self.assertTrue(any("quantized activation" in e and "embedded_input" in e for e in errors), errors)

    def test_lowered_allows_quantized_activation_bound_to_physical_q8_view(self) -> None:
        audit = _load_module()
        lowered = {
            "operations": [
                {
                    "idx": 4,
                    "layer": 0,
                    "op": "q_proj",
                    "activations": {"x": {"dtype": "q8_k", "buffer": "layer_input"}},
                    "outputs": {"y": {"buffer": "q_scratch"}},
                }
            ]
        }
        self.assertEqual(audit.audit_lowered(lowered), [])

    def test_cli_reports_generated_c_mamba_input_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            c_path = Path(td) / "model_v8.c"
            c_path.write_text(
                """
    /* Op 3: gemv_bf16 (mamba_in_proj) layer=0 section=body */
    gemv_bf16(
        (float*)(model->bump + A_RECURRENT_PACKED),
        (const void*)(model->bump + W_LAYER_0_MAMBA_IN_PROJ),
        (const float*)(model->bump + A_EMBEDDED_INPUT),
        22656,
        4480
    );
""",
                encoding="utf-8",
            )
            audit = _load_module()
            errors = audit.audit_c_source(c_path)
            self.assertEqual(len(errors), 1)
            self.assertIn("expected=A_LAYER_INPUT", errors[0])


if __name__ == "__main__":
    unittest.main()
