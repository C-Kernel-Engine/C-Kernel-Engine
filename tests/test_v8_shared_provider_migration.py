import importlib.util
import json
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
MAPS = ROOT / "version" / "v8" / "kernel_maps"
SELECTION_SCHEMA = ROOT / "version" / "v8" / "schemas" / "kernel_provider_selection.schema.json"
TRACE_FIXTURE = (
    ROOT / "version" / "v8" / "tests" / "fixtures" / "xray" / "provider_selection_trace.json"
)

SCRIPTS = ROOT / "version" / "v8" / "scripts"
sys.path.insert(0, str(SCRIPTS))
BUILD_IR_SPEC = importlib.util.spec_from_file_location(
    "build_ir_v8_shared_provider_migration", SCRIPTS / "build_ir_v8.py"
)
assert BUILD_IR_SPEC is not None and BUILD_IR_SPEC.loader is not None
build_ir = importlib.util.module_from_spec(BUILD_IR_SPEC)
BUILD_IR_SPEC.loader.exec_module(build_ir)

AUDIT_SPEC = importlib.util.spec_from_file_location(
    "audit_kernel_map_interfaces_v8", SCRIPTS / "audit_kernel_map_interfaces_v8.py"
)
assert AUDIT_SPEC is not None and AUDIT_SPEC.loader is not None
audit = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(audit)


MIGRATED_PROVIDERS = [
    "ck_residual_add_token_major",
    "ck_residual_add_token_major_bf16_storage",
    "kv_cache_store",
    "kv_cache_store_batch_bf16",
    "kv_cache_store_batch_f16",
    "kv_cache_store_batch_f32",
    "kv_cache_store_bf16",
    "kv_cache_store_f16",
    "logits_copy_to_position",
    "memcpy",
    "mrope_qk_imrope_positions",
    "mrope_qk_text",
    "mrope_qk_text_imrope",
    "mrope_qk_text_imrope_bf16_pytorch_storage",
    "mrope_qk_text_imrope_positions_bf16_pytorch_storage",
    "residual_add_backward_f32",
    "rope_backward_qk_f32",
    "rope_backward_qk_pairwise_f32",
    "rope_forward_qk",
    "rope_forward_qk_pairwise",
    "rope_forward_qk_pairwise_llama_cpu",
    "rope_forward_qk_split",
    "rope_forward_qk_split_direct",
    "rope_precompute_cache",
    "rope_precompute_cache_llama_cpu",
    "rope_precompute_cache_split",
    "yarn_rope_cache_explicit_positions_bf16",
    "yarn_rope_cache_explicit_positions_f32",
]

ROPE_FORWARD_PROVIDERS = [
    "rope_forward_qk",
    "rope_forward_qk_pairwise",
    "rope_forward_qk_pairwise_llama_cpu",
    "rope_forward_qk_split",
    "rope_forward_qk_split_direct",
]
KV_SINGLE_PROVIDERS = ["kv_cache_store", "kv_cache_store_bf16", "kv_cache_store_f16"]
KV_BATCH_PROVIDERS = ["kv_cache_store_batch_f32", "kv_cache_store_batch_bf16", "kv_cache_store_batch_f16"]
KV_STATE_PROVIDERS = KV_SINGLE_PROVIDERS + KV_BATCH_PROVIDERS
ROPE_INIT_PROVIDERS = [
    "rope_precompute_cache",
    "rope_precompute_cache_llama_cpu",
    "rope_precompute_cache_split",
    "yarn_rope_cache_explicit_positions_bf16",
    "yarn_rope_cache_explicit_positions_f32",
]
BACKWARD_PROVIDERS = [
    "residual_add_backward_f32",
    "rope_backward_qk_f32",
    "rope_backward_qk_pairwise_f32",
]
RESIDUAL_PROVIDERS = [
    "ck_residual_add_token_major",
    "ck_residual_add_token_major_bf16_storage",
]


def load_map(provider_id):
    return json.loads((MAPS / f"{provider_id}.json").read_text(encoding="utf-8"))


def synthetic_provider(
    kernel_id,
    *,
    weight="fp32",
    status="production",
    priority=100,
    group="residual_add.fp32.v1",
    phases=("prefill", "decode"),
):
    return {
        "id": kernel_id,
        "op": "residual_add",
        "variant": "forward",
        "quant": {"weight": weight, "activation": "fp32"},
        "modes": {"inference": True, "backward": False},
        "selection": {
            "status": status,
            "priority": priority,
            "equivalence_group": group,
            "phases": list(phases),
        },
    }


def interface_doc(*, inputs, outputs, params):
    return {
        "id": "synthetic",
        "op": "residual_add",
        "operation_interface": "synthetic.fp32.v1",
        "inputs": inputs,
        "outputs": outputs,
        "call_abi": {"version": 1, "params": params},
    }


def port(name, access, **extra):
    value = {"name": name, "access": access}
    value.update(extra)
    return value


class MigratedProviderMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validator = Draft202012Validator(
            json.loads(SELECTION_SCHEMA.read_text(encoding="utf-8"))
        )
        cls.docs = {provider_id: load_map(provider_id) for provider_id in MIGRATED_PROVIDERS}

    def test_migrated_provider_list_is_exact(self):
        self.assertEqual(MIGRATED_PROVIDERS, sorted(MIGRATED_PROVIDERS))
        self.assertEqual(len(MIGRATED_PROVIDERS), 28)
        self.assertEqual(len(set(MIGRATED_PROVIDERS)), 28)

    def test_selection_blocks_match_schema(self):
        for provider_id, doc in self.docs.items():
            with self.subTest(provider=provider_id):
                errors = sorted(
                    self.validator.iter_errors(doc["selection"]), key=str
                )
                self.assertEqual(errors, [])

    def test_operation_interface_names_equivalence_group(self):
        for provider_id, doc in self.docs.items():
            with self.subTest(provider=provider_id):
                self.assertEqual(
                    doc["operation_interface"],
                    doc["selection"]["equivalence_group"],
                )

    def test_call_abi_ports_cover_declared_ports(self):
        for provider_id, doc in self.docs.items():
            with self.subTest(provider=provider_id):
                self.assertTrue(audit._has_complete_interface_abi(doc))


class ContractSeparationTests(unittest.TestCase):
    def test_rope_forward_groups_are_pairwise_distinct(self):
        groups = [
            load_map(provider_id)["selection"]["equivalence_group"]
            for provider_id in ROPE_FORWARD_PROVIDERS
        ]
        self.assertEqual(len(groups), len(set(groups)))

    def test_kv_store_dtype_groups_are_pairwise_distinct(self):
        groups = [
            load_map(provider_id)["selection"]["equivalence_group"]
            for provider_id in KV_SINGLE_PROVIDERS
        ]
        self.assertEqual(len(groups), len(set(groups)))

    def test_memcpy_group_is_distinct_from_residual_add_groups(self):
        memcpy_group = load_map("memcpy")["selection"]["equivalence_group"]
        residual_groups = {
            load_map(provider_id)["selection"]["equivalence_group"]
            for provider_id in RESIDUAL_PROVIDERS
        }
        self.assertNotIn(memcpy_group, residual_groups)

    def test_memcpy_is_a_byte_copy_not_residual_arithmetic(self):
        doc = load_map("memcpy")
        self.assertEqual(doc["op"], "residual_save")
        prose = " ".join(
            [str(doc.get("notes", "")), json.dumps(doc.get("constraints", {}))]
        ).lower()
        self.assertIn("copy", prose)


class ProviderPhaseTests(unittest.TestCase):
    def phases(self, provider_id):
        return load_map(provider_id)["selection"]["phases"]

    def test_rope_init_and_yarn_providers_are_init_only(self):
        for provider_id in ROPE_INIT_PROVIDERS:
            with self.subTest(provider=provider_id):
                self.assertEqual(self.phases(provider_id), ["init"])

    def test_backward_providers_are_training_only(self):
        for provider_id in BACKWARD_PROVIDERS:
            with self.subTest(provider=provider_id):
                phases = self.phases(provider_id)
                self.assertTrue(set(phases) <= {"training", "backward"})
                self.assertTrue(set(phases).isdisjoint({"prefill", "decode"}))

    def test_kv_batch_providers_are_prefill_only(self):
        for provider_id in KV_BATCH_PROVIDERS:
            with self.subTest(provider=provider_id):
                self.assertEqual(self.phases(provider_id), ["prefill"])

    def test_single_kv_store_providers_are_decode_only(self):
        for provider_id in KV_SINGLE_PROVIDERS:
            with self.subTest(provider=provider_id):
                self.assertEqual(self.phases(provider_id), ["decode"])

    def test_selection_does_not_claim_uncertified_numerical_phases(self):
        for provider_id in MIGRATED_PROVIDERS:
            doc = load_map(provider_id)
            capabilities = doc.get("numerical_capabilities") or []
            if not capabilities:
                continue
            certified = set().union(
                *(set(capability.get("phases", [])) for capability in capabilities)
            )
            with self.subTest(provider=provider_id):
                self.assertTrue(set(doc["selection"]["phases"]) <= certified)


class LegacyBindingAbsenceTests(unittest.TestCase):
    @staticmethod
    def binding_keys(path):
        document = json.loads(path.read_text(encoding="utf-8"))
        keys = set()

        def walk(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    keys.add(key)
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(document)
        return keys

    def test_migrated_providers_have_no_legacy_binding(self):
        keys = self.binding_keys(MAPS / "kernel_bindings.json") | self.binding_keys(
            MAPS / "kernel_bindings.overlay.json"
        )
        for provider_id in MIGRATED_PROVIDERS:
            doc = load_map(provider_id)
            function = doc.get("impl", {}).get("function")
            with self.subTest(provider=provider_id):
                self.assertNotIn(provider_id, keys)
                self.assertNotIn(function, keys)


class KVCacheStateModelTests(unittest.TestCase):
    def test_kv_cache_outputs_are_read_write_state(self):
        for provider_id in KV_STATE_PROVIDERS:
            doc = load_map(provider_id)
            outputs = {port["name"]: port for port in doc["outputs"]}
            for name in ("kv_cache_k", "kv_cache_v"):
                with self.subTest(provider=provider_id, port=name):
                    self.assertIn(name, outputs)
                    self.assertEqual(outputs[name]["storage_class"], "state")
                    self.assertEqual(outputs[name]["access"], "read_write")
                    self.assertEqual(len(outputs[name]["shape"]), 3)


class FailClosedAuditTests(unittest.TestCase):
    def test_missing_required_port_fails_abi_crossvalidation(self):
        doc = interface_doc(
            inputs=[port("src", "read")],
            outputs=[port("dst", "write")],
            params=[{"name": "src", "source": "activation:src", "ports": ["input:src"]}],
        )
        self.assertFalse(audit._has_complete_interface_abi(doc))

    def test_unknown_abi_port_fails_abi_crossvalidation(self):
        doc = interface_doc(
            inputs=[port("src", "read")],
            outputs=[port("dst", "write")],
            params=[
                {"name": "src", "source": "activation:src", "ports": ["input:src"]},
                {"name": "dst", "source": "output:dst", "ports": ["output:dst"]},
                {"name": "ghost", "source": "output:dst", "ports": ["output:ghost"]},
            ],
        )
        self.assertFalse(audit._has_complete_interface_abi(doc))

    def test_duplicate_port_ownership_fails_abi_crossvalidation(self):
        doc = interface_doc(
            inputs=[port("src", "read")],
            outputs=[port("dst", "write")],
            params=[
                {"name": "src", "source": "activation:src", "ports": ["input:src"]},
                {"name": "dst", "source": "output:dst", "ports": ["output:dst"]},
                {"name": "dst_again", "source": "output:dst", "ports": ["output:dst"]},
            ],
        )
        self.assertFalse(audit._has_complete_interface_abi(doc))

    def test_invalid_alias_is_a_hard_failure(self):
        doc = interface_doc(
            inputs=[port("src", "read")],
            outputs=[port("dst", "write", alias_of="input:ghost")],
            params=[],
        )
        with self.assertRaisesRegex(RuntimeError, "invalid port alias"):
            audit._validate_port_aliases(doc)
        doc["outputs"][0]["alias_of"] = "output:dst"
        with self.assertRaisesRegex(RuntimeError, "invalid port alias"):
            audit._validate_port_aliases(doc)

    def test_unsafe_writable_overlap_is_a_hard_failure(self):
        doc = interface_doc(
            inputs=[port("q", "read")],
            outputs=[port("q", "write")],
            params=[],
        )
        with self.assertRaisesRegex(RuntimeError, "unsafe writable overlap"):
            audit._validate_port_aliases(doc)
        doc["outputs"].append(port("q", "write"))
        doc["outputs"][0]["alias_of"] = "input:q"
        with self.assertRaisesRegex(RuntimeError, "unsafe writable overlap"):
            audit._validate_port_aliases(doc)

    def test_unmigrated_maps_are_not_faulted(self):
        audit._validate_port_aliases(
            {"id": "legacy", "outputs": [{"name": "q", "access": "write"}]}
        )


class FailClosedResolverTraceTests(unittest.TestCase):
    def resolve(self, providers, trace):
        return build_ir.find_kernel(
            {"kernels": providers},
            op="residual_add",
            quant={"weight": "fp32"},
            mode="prefill",
            prefer_q8_activation=False,
            selection_trace=trace,
        )

    def test_wrong_phase_is_not_selected_and_traced(self):
        provider = synthetic_provider("decode_only", phases=("decode",))
        trace = []
        self.assertIsNone(self.resolve([provider], trace))
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["decision"], "rejected")
        self.assertEqual(trace[0]["stage"], "provider_selection")
        self.assertEqual(trace[0]["reason"], "phase_mismatch")

    def test_candidate_only_provider_does_not_resolve_and_is_traced(self):
        provider = synthetic_provider(
            "experiment", status="candidate", priority=1000
        )
        trace = []
        self.assertIsNone(self.resolve([provider], trace))
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["decision"], "rejected")
        self.assertEqual(trace[0]["reason"], "status_not_production:candidate")

    def test_priority_cannot_compare_different_equivalence_groups(self):
        with self.assertRaisesRegex(RuntimeError, "different equivalence groups"):
            self.resolve(
                [
                    synthetic_provider("one", group="residual_add.a.v1"),
                    synthetic_provider("two", group="residual_add.b.v1"),
                ],
                [],
            )

    def test_parallel_policy_precedes_priority_and_trace_matches_return(self):
        serial = synthetic_provider("serial_high", priority=200)
        serial["op"] = "gemv"
        serial["selection"]["phases"] = ["decode"]
        parallel = synthetic_provider("parallel_low", priority=100)
        parallel["op"] = "gemv"
        parallel["parallel"] = True
        parallel["selection"]["phases"] = ["decode"]
        trace = []
        selected = build_ir.find_kernel(
            {"kernels": [serial, parallel]},
            op="gemv",
            quant={"weight": "fp32"},
            mode="decode",
            prefer_q8_activation=False,
            prefer_parallel=True,
            selection_trace=trace,
        )
        self.assertEqual(selected, "parallel_low")
        self.assertEqual(
            [entry["provider"] for entry in trace if entry["decision"] == "selected"],
            [selected],
        )


class ProviderSelectionTraceFixtureTests(unittest.TestCase):
    """Compatibility filtering must run before priority ranking."""

    def build_registry(self):
        return {
            "kernels": [
                synthetic_provider(
                    "candidate_900", status="candidate", priority=900
                ),
                synthetic_provider(
                    "wrong_phase_800", priority=800, phases=("decode",)
                ),
                synthetic_provider(
                    "wrong_dtype_700", priority=700, weight="q4_k"
                ),
                synthetic_provider("compatible_100", priority=100),
            ]
        }

    def test_filtering_precedes_priority_ranking(self):
        trace = []
        resolved = build_ir.find_kernel(
            self.build_registry(),
            op="residual_add",
            quant={"weight": "fp32"},
            mode="prefill",
            prefer_q8_activation=False,
            selection_trace=trace,
        )
        self.assertEqual(resolved, "compatible_100")

        rejected_ranks = [
            index
            for index, entry in enumerate(trace)
            if entry["decision"] == "rejected"
        ]
        ranked_ranks = [
            index
            for index, entry in enumerate(trace)
            if entry["stage"] == "priority_ranking"
        ]
        self.assertTrue(rejected_ranks)
        self.assertTrue(ranked_ranks)
        self.assertLess(max(rejected_ranks), min(ranked_ranks))

        reasons = [entry["reason"] for entry in trace if entry["decision"] == "rejected"]
        self.assertEqual(
            reasons,
            [
                "status_not_production:candidate",
                "phase_mismatch",
                "weight_dtype_mismatch",
            ],
        )

        fixture = json.loads(TRACE_FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(trace, fixture)


class GraphIRExecutionMetadataTests(unittest.TestCase):
    def capability(self, **extra):
        value = {
            "id": "synthetic",
            "op": "residual_add",
            "implementation": {
                "isa_dispatch": "scalar",
                "threading": {"runtime": "serial"},
            },
        }
        value.update(extra)
        return value

    def test_provider_selection_is_disclosed(self):
        metadata = build_ir._graph_ir_execution_metadata(
            self.capability(
                selection={
                    "status": "production",
                    "priority": 100,
                    "equivalence_group": "residual_add.fp32.v1",
                    "phases": ["prefill", "decode"],
                }
            )
        )
        self.assertEqual(
            metadata["provider_selection"],
            {
                "status": "production",
                "priority": 100,
                "equivalence_group": "residual_add.fp32.v1",
                "phases": ["prefill", "decode"],
            },
        )

    def test_legacy_provider_discloses_null_selection(self):
        metadata = build_ir._graph_ir_execution_metadata(self.capability())
        self.assertIsNone(metadata["provider_selection"])


if __name__ == "__main__":
    unittest.main()
