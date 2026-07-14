from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str((ROOT / "version" / "v8" / "scripts").resolve()))


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load("audit_dsl_policy_v8", ROOT / "version" / "v8" / "scripts" / "audit_dsl_policy_v8.py")
build_ir = _load("build_ir_v8_dsl_policy", ROOT / "version" / "v8" / "scripts" / "build_ir_v8.py")


class V8DSLPolicyTests(unittest.TestCase):
    def test_cleaned_compiler_functions_have_no_model_literals(self) -> None:
        report = audit.audit()
        self.assertEqual(report["status"], "pass", report["findings"])
        self.assertGreaterEqual(report["checked_functions"], 9)
        policy = json.loads(audit.DEFAULT_POLICY.read_text(encoding="utf-8"))
        allowed_sites = sum(policy["model_literal_site_limits"].values())
        self.assertLessEqual(report["model_literal_sites"], allowed_sites)

    def test_model_literal_inventory_accepts_debt_reduction(self) -> None:
        policy = json.loads(audit.DEFAULT_POLICY.read_text(encoding="utf-8"))
        relative_path = "version/v8/scripts/codegen_prefill_v8.py"
        current = audit.audit()["model_literal_inventory"][relative_path]["sites"]
        policy["model_literal_site_limits"][relative_path] = current + 1
        with tempfile.TemporaryDirectory() as temp:
            path = pathlib.Path(temp) / "policy.json"
            path.write_text(json.dumps(policy), encoding="utf-8")
            report = audit.audit(path)
        self.assertEqual(report["status"], "pass", report["findings"])

    def test_model_literal_inventory_ignores_docs_and_counts_code(self) -> None:
        source = '''
"""Qwen in a module docstring is not executable specialization."""
GEMMA_PROVIDER = "gemma_runtime_provider"

def lower():
    """Qwen in a function docstring is not executable specialization."""
    # Gemma in a comment is also not executable specialization.
    return "qwen_runtime_branch"
'''
        row = audit.count_model_literal_sites(source, ["qwen", "gemma"], path="synthetic.py")
        self.assertEqual(row["sites"], 2)
        self.assertEqual(row["functions"], {"<module>": 1, "lower": 1})

    def test_model_literal_site_limit_is_fail_closed(self) -> None:
        policy = json.loads(audit.DEFAULT_POLICY.read_text(encoding="utf-8"))
        relative_path = "version/v8/scripts/codegen_prefill_v8.py"
        current = audit.audit()["model_literal_inventory"][relative_path]["sites"]
        self.assertGreater(current, 0)
        policy["model_literal_site_limits"][relative_path] = current - 1
        with tempfile.TemporaryDirectory() as temp:
            path = pathlib.Path(temp) / "policy.json"
            path.write_text(json.dumps(policy), encoding="utf-8")
            report = audit.audit(path)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(
            any(finding.get("kind") == "model_literal_site_limit" for finding in report["findings"])
        )

    def test_ast_policy_rejects_model_specific_branch(self) -> None:
        source = """
def lower(config):
    if config.get('model') == 'qwen_new':
        return 'special_kernel'
    return 'generic_kernel'
"""
        findings = audit.scan_source(source, ["lower"], ["qwen", "gemma"], path="synthetic.py")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["function"], "lower")

    def test_ast_policy_rejects_aliased_model_dispatch(self) -> None:
        source = """
def lower(config):
    family = config.get('model_type', '')
    selected = family
    if selected == 'future_model':
        return 'special_kernel'
    return 'generic_kernel'
"""
        findings = audit.scan_model_dispatch_source(
            source,
            ["model", "model_type", "arch", "family"],
            path="synthetic.py",
        )
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["kind"], "model_dispatch")

    def test_ast_policy_allows_exact_operation_dispatch(self) -> None:
        source = """
def lower(op):
    if op == 'registered_family_named_kernel':
        return op
    return 'other_registered_kernel'
"""
        self.assertEqual(
            audit.scan_model_dispatch_source(
                source,
                ["model", "model_type", "arch", "family"],
                path="synthetic.py",
            ),
            [],
        )

    def test_runtime_defaults_are_key_order_deterministic(self) -> None:
        first = {
            "contract": {
                "runtime_defaults": {
                    "prefer_q8_0_contract": True,
                    "activation_preference_by_op": {"mlp_down": "fp32", "out_proj": "q8_0"},
                }
            }
        }
        second = json.loads(json.dumps(first, sort_keys=True))
        self.assertEqual(
            build_ir._apply_circuit_runtime_defaults({}, first, source="first"),
            build_ir._apply_circuit_runtime_defaults({}, second, source="second"),
        )

    def test_runtime_default_override_is_explicit_and_stable(self) -> None:
        circuit = {
            "contract": {
                "runtime_defaults": {
                    "prefer_q8_0_contract": True,
                    "activation_preference_by_op": {"mlp_down": "fp32", "out_proj": "q8_0"},
                }
            }
        }
        actual = build_ir._apply_circuit_runtime_defaults(
            {"prefer_q8_0_contract": False, "activation_preference_by_op": {"mlp_down": "q8_0"}},
            circuit,
            source="override",
        )
        self.assertFalse(actual["prefer_q8_0_contract"])
        self.assertEqual(actual["activation_preference_by_op"]["mlp_down"], "q8_0")
        self.assertEqual(actual["activation_preference_by_op"]["out_proj"], "q8_0")

    def test_special_kernel_resolution_has_no_silent_default(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "rope_qk requires an exact circuit kernel mapping"):
            build_ir._resolve_rope_qk_kernel({"rope_layout": "split"}, {})
        with self.assertRaisesRegex(RuntimeError, "position_embeddings requires an exact circuit kernel mapping"):
            build_ir._resolve_position_embeddings_kernel({}, {})
        self.assertEqual(
            build_ir._resolve_rope_qk_kernel(
                {"rope_layout": "multi_section_1d"},
                {"rope_qk": "mrope_qk_text"},
            ),
            "mrope_qk_text",
        )
        position_kernels = {
            "position_embeddings": {
                "default": "position_embeddings_add_tiled_2d",
                "align_corners_bilinear": "position_embeddings_add_tiled_2d_align_corners",
            }
        }
        self.assertEqual(
            build_ir._resolve_position_embeddings_kernel({}, position_kernels),
            "position_embeddings_add_tiled_2d",
        )
        self.assertEqual(
            build_ir._resolve_position_embeddings_kernel(
                {"position_interpolation_policy": "align_corners_bilinear"},
                position_kernels,
            ),
            "position_embeddings_add_tiled_2d_align_corners",
        )
        with self.assertRaisesRegex(RuntimeError, "no exact circuit kernel mapping"):
            build_ir._resolve_position_embeddings_kernel(
                {"position_interpolation_policy": "misspelled_policy"},
                position_kernels,
            )

    def test_kernel_availability_does_not_substitute_broader_operation(self) -> None:
        registry = {"kernels": [{"op": "attention"}]}
        self.assertEqual(
            build_ir.validate_kernel_availability(
                registry,
                ["attention", "attention_sliding", "attn_sliding"],
            ),
            {
                "attention": True,
                "attention_sliding": False,
                "attn_sliding": False,
            },
        )

    def test_hydration_propagates_multimodal_capability_without_family_dispatch(self) -> None:
        manifest = {
            "config": {"model": "qwen3vl"},
            "template": {"name": "qwen3vl"},
        }
        hydrated = build_ir._hydrate_manifest_template(manifest)
        bridge = hydrated["config"].get("multimodal_bridge_contract")
        self.assertIsInstance(bridge, dict)
        self.assertEqual(bridge.get("position_policy"), "mrope_2d")
        self.assertFalse(hydrated["config"]["prefill_gateup_swiglu_fusion_default"])

    def test_attention_dimensions_are_config_driven_across_layouts(self) -> None:
        uniform = {
            "embed_dim": 1024,
            "num_heads": 8,
            "num_kv_heads": 2,
            "head_dim": 128,
            "attn_out_dim": 1024,
            "rotary_dim": 64,
            "rope_theta": 500000.0,
            "use_rope_freq_factors": 1,
        }
        params: dict[str, object] = {}
        build_ir.apply_layer_attention_dims("rope_qk", params, 0, uniform)
        self.assertEqual(params["q_dim"], 1024)
        self.assertEqual(params["k_dim"], 256)
        self.assertEqual(params["v_dim"], 256)
        self.assertEqual(params["n_dims"], 64)
        self.assertEqual(params["use_rope_freq_factors"], 1)

        per_layer = dict(uniform)
        per_layer.update(
            {
                "layer_q_head_dim": [128, 256],
                "layer_k_head_dim": [128, 256],
                "layer_v_head_dim": [256, 256],
                "layer_q_dim": [1024, 2048],
                "layer_rotary_dim": [64, 128],
                "layer_sliding_window": [4096, 0],
                "layer_rope_kind": ["swa", "full"],
                "rope_theta_swa": 10000.0,
            }
        )
        params = {}
        build_ir.apply_layer_attention_dims("attn_shared_kv", params, 1, per_layer)
        self.assertEqual(params["q_head_dim"], 256)
        self.assertEqual(params["q_dim"], 2048)
        self.assertEqual(params["k_dim"], 2048)
        self.assertEqual(params["v_dim"], 2048)
        self.assertEqual(params["num_kv_heads"], 8)

    def test_incomplete_per_layer_dimensions_hard_fail(self) -> None:
        config = {
            "embed_dim": 1024,
            "num_heads": 8,
            "num_kv_heads": 2,
            "head_dim": 128,
            "layer_q_head_dim": [128],
        }
        with self.assertRaisesRegex(ValueError, "no value for layer 1"):
            build_ir.apply_layer_attention_dims("q_proj", {}, 1, config)

    def test_final_quant_dimensions_use_declared_projector_width(self) -> None:
        config = {"embed_dim": 896, "projector_in_dim": 4096, "model": "unknown_family"}
        self.assertEqual(
            build_ir.compute_matmul_dims("quantize_final_output", config),
            (4096, 4096),
        )

    def test_lowering_preserves_renamed_ir1_decode_attention_kernel(self) -> None:
        op = {
            "op": "attn_shared_kv",
            "kernel": "renamed_exact_decode_provider_id",
            "function": "renamed_exact_decode_provider_function",
            "resolved_contract": {
                "kernel_id": "renamed_exact_decode_provider_id",
                "function": "renamed_exact_decode_provider_function",
            },
        }
        self.assertEqual(
            build_ir._require_resolved_decode_attention_kernel(op),
            "renamed_exact_decode_provider_id",
        )

    def test_lowering_rejects_incomplete_or_ambiguous_attention_selection(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "without an exact IR1 kernel"):
            build_ir._require_resolved_decode_attention_kernel({"op": "attn"})
        with self.assertRaisesRegex(RuntimeError, "does not match resolved contract"):
            build_ir._require_resolved_decode_attention_kernel(
                {
                    "op": "attn_sliding",
                    "kernel": "provider_a",
                    "function": "provider_function",
                    "resolved_contract": {
                        "kernel_id": "provider_b",
                        "function": "provider_function",
                    },
                }
            )
        with self.assertRaisesRegex(RuntimeError, "does not match resolved contract function"):
            build_ir._require_resolved_decode_attention_kernel(
                {
                    "op": "attn",
                    "kernel": "provider_a",
                    "function": "provider_function_a",
                    "resolved_contract": {
                        "kernel_id": "provider_a",
                        "function": "provider_function_b",
                    },
                }
            )

    def test_weight_policy_is_circuit_owned_and_conditional(self) -> None:
        circuit = {
            "contract": {
                "weight_policy": {
                    "ignore": [
                        {
                            "pattern": "layer.*.bridge_scale",
                            "reason": "external bridge owns this scale",
                            "when": {"config_key": "bridge.mode", "equals": "external"},
                        }
                    ]
                }
            }
        }
        weights = {"layer.0.bridge_scale", "layer.0.attn.weight"}
        self.assertEqual(
            build_ir._ignored_manifest_weights(circuit, {"bridge": {"mode": "external"}}, weights),
            {"layer.0.bridge_scale": "external bridge owns this scale"},
        )
        self.assertEqual(
            build_ir._ignored_manifest_weights(circuit, {"bridge": {"mode": "internal"}}, weights),
            {},
        )

    def test_invalid_weight_policy_hard_fails(self) -> None:
        circuit = {
            "contract": {
                "weight_policy": {
                    "ignore": [
                        {"pattern": "layer.*", "reason": "test", "model_family": "forbidden"}
                    ]
                }
            }
        }
        with self.assertRaisesRegex(RuntimeError, "HARD CIRCUIT WEIGHT POLICY FAULT"):
            build_ir._ignored_manifest_weights(circuit, {}, {"layer.0.weight"})

    def test_overlapping_weight_policy_rules_hard_fail(self) -> None:
        circuit = {
            "contract": {
                "weight_policy": {
                    "ignore": [
                        {"pattern": "layer.*", "reason": "broad"},
                        {"pattern": "*.weight", "reason": "suffix"},
                    ]
                }
            }
        }
        with self.assertRaisesRegex(RuntimeError, "matches multiple ignore rules"):
            build_ir._ignored_manifest_weights(circuit, {}, {"layer.0.weight"})

    def test_op_weight_binding_is_circuit_owned(self) -> None:
        circuit = {
            "contract": {
                "weight_policy": {
                    "op_bindings": [
                        {"section": "body", "op": "rope_qk", "weights": ["rope_freqs"]}
                    ]
                }
            }
        }
        self.assertEqual(
            build_ir._circuit_op_weight_keys(circuit, "body", "rope_qk"),
            ["rope_freqs"],
        )
        self.assertIsNone(build_ir._circuit_op_weight_keys(circuit, "body", "q_proj"))

    def test_policy_rejects_missing_function_instead_of_weakening_scope(self) -> None:
        policy = {
            "schema": "cke.v8_dsl_policy",
            "schema_version": 1,
            "forbidden_model_literals": ["qwen"],
            "compiler_functions": {"version/v8/scripts/build_ir_v8.py": ["function_does_not_exist"]},
        }
        with tempfile.TemporaryDirectory() as temp:
            path = pathlib.Path(temp) / "policy.json"
            path.write_text(json.dumps(policy), encoding="utf-8")
            with self.assertRaisesRegex(audit.DSLPolicyError, "policy function not found"):
                audit.audit(path)


if __name__ == "__main__":
    unittest.main()
