"""Model-independent runtime extent contract and layout tests."""

import importlib.util
from pathlib import Path
import unittest
import sys


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "version" / "v8" / "scripts" / "runtime_extent_contract_v8.py"
spec = importlib.util.spec_from_file_location("runtime_extent_contract_v8", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    return ({
        "runtime_lengths": {
            "expanded_frames": {"producer": "checked_sum", "result": "frames",
                                "capacity": 12, "allow_zero": True}
        },
        "runtime_views": {
            "expanded": {"buffer": "expanded_buffer", "length": "expanded_frames", "channels": 3,
                         "physical_stride": 16, "element_bytes": 4,
                         "buffer_bytes": 3 * 16 * 4}
        },
        "runtime_constants": {"phoneme_count": 3, "input_elements": 9},
    }, [
        {"op_id": "checked_sum", "returns_status": True,
         "produces_runtime_lengths": {"expanded_frames": "frames"}},
        {"op_id": "expand", "returns_status": True,
         "consumes_runtime_lengths": ["expanded_frames"]},
        {"op_id": "consume", "returns_status": True,
         "consumes_runtime_lengths": ["expanded_frames"]},
    ])


class RuntimeExtentContractTest(unittest.TestCase):
    def test_capacity_valid_extent_and_stride_stay_distinct(self):
        circuit, ops = fixture()
        result = module.normalize_runtime_extents(circuit, ops)
        self.assertEqual(result["runtime_lengths"]["expanded_frames"]["capacity"], 12)
        self.assertTrue(result["runtime_lengths"]["expanded_frames"]["allow_zero"])
        view = result["runtime_views"]["expanded"]
        self.assertEqual(view["physical_stride"], 16)
        self.assertEqual(view["required_bytes"], ((3 - 1) * 16 + 12) * 4)
        self.assertEqual(result["runtime_constants"]["phoneme_count"], 3)

    def test_rejects_invalid_metadata_before_lowering(self):
        for name, mutate in [
            ("missing_producer", lambda c, o: c["runtime_lengths"]["expanded_frames"].update(producer="missing")),
            ("missing_result", lambda c, o: o[0].update(produces_runtime_lengths={})),
            ("unchecked_producer", lambda c, o: o[0].update(returns_status=False)),
            ("short_stride", lambda c, o: c["runtime_views"]["expanded"].update(physical_stride=11)),
            ("short_buffer", lambda c, o: c["runtime_views"]["expanded"].update(buffer_bytes=100)),
            ("future_producer", lambda c, o: o.reverse()),
            ("bad_capacity", lambda c, o: c["runtime_lengths"]["expanded_frames"].update(capacity=0)),
            ("i32_capacity", lambda c, o: c["runtime_lengths"]["expanded_frames"].update(capacity=1 << 31)),
            ("unknown_length", lambda c, o: o[1].update(consumes_runtime_lengths=["unknown"])),
            ("overflow", lambda c, o: c["runtime_views"]["expanded"].update(physical_stride=module.SIZE_MAX)),
            ("shadowed_constant", lambda c, o: c["runtime_constants"].update(expanded_frames=3)),
        ]:
            with self.subTest(name=name):
                circuit, ops = fixture()
                mutate(circuit, ops)
                with self.assertRaises(module.RuntimeExtentContractError):
                    module.normalize_runtime_extents(circuit, ops)

    def test_lower3_binds_named_result_to_runtime_extent_storage(self):
        scripts = ROOT / "version" / "v8" / "scripts"
        sys.path.insert(0, str(scripts))
        import build_ir_v8

        circuit, ops = fixture()
        circuit["runtime_constants"].update(value_elements=3, max_duration=8,
                                             expanded_capacity=12)
        contract = module.normalize_runtime_extents(circuit, ops)
        lowered = {"config": {}, "operations": [{
            "idx": 0, "op": "runtime_extent_sum", "section": "header", "layer": -1,
            "kernel": "runtime_extent_sum_i32", "function": "ck_runtime_sum_i32_checked",
            "template_op_id": "checked_sum",
            "activations": {"values": {"activation_offset": 0, "buffer": "durations"}},
            "outputs": {"valid_extent": {"activation_offset": 64, "buffer": "valid_scalar"}},
            "weights": {}, "scratch": [], "params": {},
            "runtime_extent_contract": contract,
            "produces_runtime_lengths": {"expanded_frames": "valid_extent"},
            "returns_status": True,
        }]}
        call = build_ir_v8.generate_ir_lower_3(lowered, "prefill")["operations"][0]
        self.assertEqual(call["errors"], [])
        args = {arg["name"]: arg["expr"] for arg in call["args"]}
        self.assertEqual(args["valid_extent"], "&runtime_extents.expanded_frames")
        self.assertEqual(args["capacity"], "12")
        self.assertEqual(call["produces_runtime_lengths"], {"expanded_frames": "valid_extent"})


if __name__ == "__main__":
    unittest.main()
