import importlib.util
import ctypes
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from array import array
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "resolve_layout_chain_v8.py"
SPEC = importlib.util.spec_from_file_location("resolve_layout_chain_v8", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
resolver = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = resolver
SPEC.loader.exec_module(resolver)

BUILD_IR_SCRIPT = ROOT / "version" / "v8" / "scripts" / "build_ir_v8.py"
sys.path.insert(0, str(BUILD_IR_SCRIPT.parent))
BUILD_IR_SPEC = importlib.util.spec_from_file_location("build_ir_v8_layout_test", BUILD_IR_SCRIPT)
assert BUILD_IR_SPEC is not None and BUILD_IR_SPEC.loader is not None
build_ir = importlib.util.module_from_spec(BUILD_IR_SPEC)
BUILD_IR_SPEC.loader.exec_module(build_ir)

CODEGEN_SCRIPT = ROOT / "version" / "v8" / "scripts" / "codegen_prefill_v8.py"
CODEGEN_SPEC = importlib.util.spec_from_file_location("codegen_prefill_v8_layout_test", CODEGEN_SCRIPT)
assert CODEGEN_SPEC is not None and CODEGEN_SPEC.loader is not None
codegen = importlib.util.module_from_spec(CODEGEN_SPEC)
CODEGEN_SPEC.loader.exec_module(codegen)

CODEGEN_CORE_SCRIPT = ROOT / "version" / "v8" / "scripts" / "codegen_core_v8.py"
CODEGEN_CORE_SPEC = importlib.util.spec_from_file_location(
    "codegen_core_v8_layout_test", CODEGEN_CORE_SCRIPT
)
assert CODEGEN_CORE_SPEC is not None and CODEGEN_CORE_SPEC.loader is not None
codegen_core = importlib.util.module_from_spec(CODEGEN_CORE_SPEC)
CODEGEN_CORE_SPEC.loader.exec_module(codegen_core)


def provider(provider_id, role, layout, priority, placement="local"):
    field = "outputs" if role == "producer" else "inputs"
    port = "y" if role == "producer" else "x"
    return {
        "id": provider_id,
        "selection": {"priority": priority},
        field: [{"name": port, "layout": layout, "placement": placement}],
    }


class LayoutProviderSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="cke-layout-provider-")
        cls._lib_path = Path(cls._tmp.name) / "liblayout.so"
        subprocess.run(
            [
                "gcc", "-std=c11", "-O2", "-shared", "-fPIC",
                str(ROOT / "src" / "kernels" / "layout_kernels.c"),
                "-o", str(cls._lib_path),
            ],
            check=True,
        )
        cls._lib = ctypes.CDLL(str(cls._lib_path))
        signature = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        cls._lib.ck_layout_token_to_head_f32.argtypes = signature
        cls._lib.ck_layout_head_to_token_f32.argtypes = signature

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_layout_providers_are_bit_exact_in_both_directions(self):
        tokens, heads, dim = 7, 3, 5
        source = array("f", (float(index) / 7.0 for index in range(tokens * heads * dim)))
        head = array("f", [0.0]) * len(source)
        restored = array("f", [0.0]) * len(source)
        src_ptr = (ctypes.c_float * len(source)).from_buffer(source)
        head_ptr = (ctypes.c_float * len(head)).from_buffer(head)
        restored_ptr = (ctypes.c_float * len(restored)).from_buffer(restored)
        self._lib.ck_layout_token_to_head_f32(
            src_ptr, head_ptr, tokens, heads, dim
        )
        self._lib.ck_layout_head_to_token_f32(
            head_ptr, restored_ptr, heads, tokens, dim
        )
        self.assertEqual(source.tobytes(), restored.tobytes())

    def test_checked_in_layout_converters_match_physical_schema(self):
        schema = json.loads(
            (ROOT / "version" / "v8" / "schemas" / "kernel_physical_layout.schema.json").read_text()
        )
        validator = Draft202012Validator(schema)
        maps = ROOT / "version" / "v8" / "kernel_maps"
        checked = 0
        for path in maps.glob("*.json"):
            document = json.loads(path.read_text())
            if not isinstance(document.get("layout_conversion"), dict):
                continue
            checked += 1
            conversion = document["layout_conversion"]
            for key in ("from_layout", "to_layout"):
                errors = list(validator.iter_errors({"layout": conversion[key]}))
                self.assertEqual(errors, [], path.name)
            self.assertGreater(conversion.get("cost_rank", 0), 0)
        self.assertEqual(checked, 2)

    def test_call_ir_preserves_selected_physical_provider(self):
        lowered = {
            "config": {},
            "operations": [{
                "idx": 7,
                "kernel": "layout_convert_token_to_head_f32",
                "function": "transpose_inplace",
                "op": "transpose_qkv_to_head_major",
                "layer": 0,
                "section": "body",
            }],
        }
        call_ir = build_ir.generate_ir_lower_3(lowered, "prefill")
        operation = call_ir["operations"][0]
        self.assertEqual(operation["kernel"], "layout_convert_token_to_head_f32")
        self.assertEqual(
            operation["resolved_physical_execution"]["layout_conversion"]["to_layout"],
            "head_major_contiguous",
        )

    def test_attention_output_selects_direct_token_major_provider(self):
        registry = json.loads(
            (ROOT / "version" / "v8" / "kernel_maps" / "KERNEL_REGISTRY.json").read_text()
        )
        managed, converter, execution = build_ir._resolve_layout_edge(
            registry,
            producer_kernel="attention_forward_causal_head_major_gqa_flash_strided",
            producer_port="out",
            consumer_kernel="gemm_nt_q8_0_q8_0",
            consumer_port="A",
        )
        self.assertTrue(managed)
        self.assertIsNone(converter)
        self.assertEqual(
            execution["provider_id"],
            "attention_forward_causal_head_major_gqa_flash_strided_token_output",
        )
        self.assertEqual(execution["output_layout"], "token_major_contiguous")
        self.assertEqual(
            execution["mixed_visual_chunk_function"],
            "attention_forward_mixed_visual_chunk_head_major_gqa_flash_strided_gemma4_token_output",
        )

    def test_bf16_vision_attention_selects_direct_token_major_provider(self):
        registry = json.loads(
            (ROOT / "version" / "v8" / "kernel_maps" / "KERNEL_REGISTRY.json").read_text()
        )
        managed, converter, execution = build_ir._resolve_layout_edge(
            registry,
            producer_kernel="attention_forward_full_head_major_gqa_pytorch_cpu_flash_bf16_storage",
            producer_port="out",
            consumer_kernel="gemm_nt_bf16_pytorch_onednn_brgemm_bf16_storage",
            consumer_port="A",
        )
        self.assertTrue(managed)
        self.assertIsNone(converter)
        self.assertEqual(
            execution["provider_id"],
            "attention_forward_full_head_major_gqa_pytorch_cpu_flash_bf16_storage_token_output",
        )
        self.assertEqual(execution["output_layout"], "token_major_contiguous")

    def test_gemma4_attention_selects_direct_token_major_providers(self):
        registry = json.loads(
            (ROOT / "version" / "v8" / "kernel_maps" / "KERNEL_REGISTRY.json").read_text()
        )
        cases = (
            (
                "attention_forward_causal_head_major_gqa_flash_strided_gemma4",
                "attention_forward_causal_head_major_gqa_flash_strided_gemma4_token_output",
            ),
            (
                "attention_forward_causal_head_major_gqa_flash_strided_sliding_gemma4",
                "attention_forward_causal_head_major_gqa_flash_strided_sliding_gemma4_token_output",
            ),
        )
        for numerical, physical in cases:
            with self.subTest(numerical=numerical):
                managed, converter, execution = build_ir._resolve_layout_edge(
                    registry,
                    producer_kernel=numerical,
                    producer_port="out",
                    consumer_kernel="gemm_nt_q8_0_q8_0",
                    consumer_port="A",
                )
                self.assertTrue(managed)
                self.assertIsNone(converter)
                self.assertEqual(execution["provider_id"], physical)
                self.assertEqual(execution["output_layout"], "token_major_contiguous")

    def test_direct_layout_provider_explicitly_aliases_numerical_owner(self):
        provider_map = json.loads(
            (ROOT / "version" / "v8" / "kernel_maps" /
             "attention_forward_causal_head_major_gqa_flash_strided_token_output.json").read_text()
        )
        self.assertEqual(
            provider_map["physical_alias_of"],
            "attention_forward_causal_head_major_gqa_flash_strided",
        )
        self.assertNotIn("numerical_capabilities", provider_map)

    def test_physical_provider_with_different_abi_fails_closed(self):
        registry = json.loads(
            (ROOT / "version" / "v8" / "kernel_maps" / "KERNEL_REGISTRY.json").read_text()
        )
        mutated = copy.deepcopy(registry)
        provider = next(
            item for item in mutated["kernels"]
            if item.get("id") == "attention_forward_causal_head_major_gqa_flash_strided_token_output"
        )
        provider["call_abi"]["params"][-1]["source"] = "dim:wrong_stride"
        with self.assertRaisesRegex(RuntimeError, "does not preserve the call ABI"):
            build_ir._resolve_layout_edge(
                mutated,
                producer_kernel="attention_forward_causal_head_major_gqa_flash_strided",
                producer_port="out",
                consumer_kernel="gemm_nt_q8_0_q8_0",
                consumer_port="A",
            )

    def test_physical_provider_requires_explicit_numerical_alias(self):
        registry = json.loads(
            (ROOT / "version" / "v8" / "kernel_maps" / "KERNEL_REGISTRY.json").read_text()
        )
        mutated = copy.deepcopy(registry)
        provider = next(
            item for item in mutated["kernels"]
            if item.get("id") == "attention_forward_causal_head_major_gqa_flash_strided_token_output"
        )
        provider.pop("physical_alias_of")
        managed, converter, execution = build_ir._resolve_layout_edge(
            mutated,
            producer_kernel="attention_forward_causal_head_major_gqa_flash_strided",
            producer_port="out",
            consumer_kernel="gemm_nt_q8_0_q8_0",
            consumer_port="A",
        )
        self.assertFalse(managed)
        self.assertIsNone(converter)
        self.assertIsNone(execution)

    def test_sole_output_name_is_read_from_provider_metadata(self):
        registry = json.loads(
            (ROOT / "version" / "v8" / "kernel_maps" / "KERNEL_REGISTRY.json").read_text()
        )
        kernel_id = "attention_forward_causal_head_major_gqa_prefill_append_f16cache_flash_auto_qtile64"
        self.assertEqual(
            build_ir._single_kernel_port_name(registry, kernel_id, "outputs"),
            "output",
        )

    def test_unknown_physical_variant_fails_closed(self):
        registry = json.loads(
            (ROOT / "version" / "v8" / "kernel_maps" / "KERNEL_REGISTRY.json").read_text()
        )
        mutated = copy.deepcopy(registry)
        provider = next(
            item for item in mutated["kernels"]
            if item.get("id") == "attention_forward_causal_head_major_gqa_flash_strided_token_output"
        )
        provider["physical_variants"]["unvalidated_function"] = "unsafe_symbol"
        with self.assertRaisesRegex(RuntimeError, "unknown physical variants"):
            build_ir._resolve_layout_edge(
                mutated,
                producer_kernel="attention_forward_causal_head_major_gqa_flash_strided",
                producer_port="out",
                consumer_kernel="gemm_nt_q8_0_q8_0",
                consumer_port="A",
            )

    def test_codegen_calls_resolved_physical_function(self):
        args = [
            {"name": name, "expr": expr}
            for name, expr in (
                ("q", "q"), ("k", "k"), ("v", "v"), ("output", "out"),
                ("num_heads", "4"), ("num_kv_heads", "2"),
                ("num_tokens", "num_tokens"), ("head_dim", "64"),
                ("aligned_head_dim", "64"), ("kv_stride_tokens", "max_seq_len"),
            )
        ]
        code = codegen.emit_prefill_op({
            "function": "attention_forward_causal_head_major_gqa_flash_strided",
            "op": "attn",
            "layer": 0,
            "args": args,
            "resolved_physical_execution": {
                "function": "attention_forward_causal_head_major_gqa_flash_strided_token_output",
                "mixed_visual_chunk_function": "attention_forward_mixed_visual_chunk_head_major_gqa_flash_strided_gemma4_token_output",
            },
        }, 1, {})
        self.assertIn("attention_forward_causal_head_major_gqa_flash_strided_token_output(", code)
        self.assertIn(
            "attention_forward_mixed_visual_chunk_head_major_gqa_flash_strided_gemma4_token_output(",
            code,
        )
        self.assertNotIn("\n    attention_forward_causal_head_major_gqa_flash_strided(", code)

    def test_decode_codegen_calls_resolved_physical_function(self):
        code = codegen_core.emit_op({
            "idx": 8,
            "function": "attention_reference",
            "op": "attn",
            "layer": 0,
            "section": "body",
            "args": [{"name": "input", "expr": "input"}],
            "resolved_physical_execution": {"function": "attention_direct_layout"},
        })
        self.assertIn("attention_direct_layout(", code)
        self.assertNotIn("\n    attention_reference(", code)

    def test_direct_compatible_chain_beats_higher_priority_converted_chain(self):
        routes = resolver.rank_layout_routes(
            [
                provider("token_fast", "producer", "token_major_contiguous", 500),
                provider("head_direct", "producer", "head_major_contiguous", 100),
            ],
            producer_port="y",
            consumers=[provider("attention", "consumer", "head_major_contiguous", 100)],
            consumer_port="x",
            converters=[{
                "id": "token_to_head",
                "from_layout": "token_major_contiguous",
                "to_layout": "head_major_contiguous",
                "cost_rank": 10,
            }],
        )
        self.assertEqual(routes[0].producer.provider_id, "head_direct")
        self.assertIsNone(routes[0].converter_id)

    def test_priority_ranks_equally_compatible_direct_providers(self):
        routes = resolver.rank_layout_routes(
            [
                provider("baseline", "producer", "head_major_contiguous", 100),
                provider("measured", "producer", "head_major_contiguous", 200),
            ],
            producer_port="y",
            consumers=[provider("attention", "consumer", "head_major_contiguous", 100)],
            consumer_port="x",
        )
        self.assertEqual(routes[0].producer.provider_id, "measured")

    def test_distributed_placement_requires_explicit_transport_converter(self):
        with self.assertRaisesRegex(RuntimeError, "no compatible physical provider chain"):
            resolver.rank_layout_routes(
                [provider("local_q", "producer", "head_major_contiguous", 100, "local")],
                producer_port="y",
                consumers=[provider("remote_attn", "consumer", "head_major_contiguous", 100, "sharded")],
                consumer_port="x",
            )

        routes = resolver.rank_layout_routes(
            [provider("local_q", "producer", "head_major_contiguous", 100, "local")],
            producer_port="y",
            consumers=[provider("remote_attn", "consumer", "head_major_contiguous", 100, "sharded")],
            consumer_port="x",
            converters=[{
                "id": "head_all_to_all",
                "from_layout": "head_major_contiguous",
                "to_layout": "head_major_contiguous",
                "from_placement": "local",
                "to_placement": "sharded",
                "cost_rank": 50,
            }],
        )
        self.assertEqual(routes[0].converter_id, "head_all_to_all")

    def test_missing_layout_fails_closed(self):
        missing = provider("bad", "producer", "head_major_contiguous", 100)
        del missing["outputs"][0]["layout"]
        with self.assertRaisesRegex(RuntimeError, "has no physical layout"):
            resolver.rank_layout_routes(
                [missing],
                producer_port="y",
                consumers=[provider("attention", "consumer", "head_major_contiguous", 100)],
                consumer_port="x",
            )

    def test_equal_rank_chains_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, "ambiguous equal-rank"):
            resolver.rank_layout_routes(
                [
                    provider("a", "producer", "head_major_contiguous", 100),
                    provider("b", "producer", "head_major_contiguous", 100),
                ],
                producer_port="y",
                consumers=[provider("attention", "consumer", "head_major_contiguous", 100)],
                consumer_port="x",
            )


if __name__ == "__main__":
    unittest.main()
