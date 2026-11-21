"""Asset-free tests for Inkling loop state ownership and mixed storage."""
import copy
import io
import json
import math
import struct
import unittest
from unittest.mock import patch

from version.v8.scripts.audit_inkling2_mini_v8 import audit, fetch_metadata, text_schedule


class InklingAuditTests(unittest.TestCase):
    def setUp(self):
        self.text = {
            "num_hidden_layers": 4, "hidden_size": 32, "intermediate_size": 32,
            "n_routed_experts": 2, "vocab_size": 8, "local_layer_ids": [0, 1, 3],
            "n_layers_per_attn_res_block": 1,
            "loop_config": dict(core_start=1, core_end=3, num_passes=3, gate_rank=4, is_varloop=False),
        }

    def test_fixed_passes_alias_weights_but_keep_state_disjoint(self):
        plan = text_schedule(self.text)
        self.assertEqual([x["weight_layer"] for x in plan], [0, 1, 2, 1, 2, 1, 2, 3])
        self.assertEqual([x["state_slot"] for x in plan], [0, 1, 2, 4, 5, 6, 7, 3])
        self.assertEqual([x["takes_prior_pass_message"] for x in plan],
                         [False, False, False, True, True, True, True, False])
        self.assertEqual([x["state_slot"] for x in plan if x["attention"] == "full"], [2, 5, 7])
        self.text["loop_config"]["is_varloop"] = True
        with self.assertRaisesRegex(ValueError, "fixed loop"):
            text_schedule(self.text)

    def test_metadata_fetch_reads_only_valid_ranges(self):
        index = {"metadata": {"total_size": 1},
                 "weight_map": {"x": "model-00001-of-00001.safetensors"}}

        class Response(io.BytesIO):
            def __init__(self, data, status=200, content_range=None):
                super().__init__(data)
                self.status = status
                self.headers = {"Content-Range": content_range}

        def metadata():
            return [Response(b"{}"), Response(json.dumps(index).encode()), Response(b"{}")]
        replies = metadata() + [
            Response(struct.pack("<Q", 2), 206, "bytes 0-7/11"),
            Response(b"{}", 206, "bytes 8-9/11"),
        ]
        with patch("version.v8.scripts.audit_inkling2_mini_v8.urlopen", side_effect=replies) as http:
            self.assertEqual(fetch_metadata("a" * 40)["safetensors-header.json"], b"{}")
            ranges = [c.args[0].get_header("Range") for c in http.call_args_list[3:]]
            self.assertEqual(ranges, ["bytes=0-7", "bytes=8-9"])
        with patch("version.v8.scripts.audit_inkling2_mini_v8.urlopen",
                   side_effect=metadata() + [Response(b"")]):
            with self.assertRaisesRegex(ValueError, "honor.*byte range"):
                fetch_metadata("a" * 40)

    def test_header_extent_and_both_quantization_contracts(self):
        header, offset = {}, 0

        def add(name, dtype, shape):
            nonlocal offset
            size = math.prod(shape) * {"BF16": 2, "F32": 4, "U8": 1, "F8_E4M3": 1}[dtype]
            header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
            offset += size

        for part in ("embed", "unembed"):
            add(f"model.llm.{part}.weight", "BF16", [8, 32])
        for part in ("middle_cycle_cross_pass_queries", "middle_cycle_cross_pass_kv_queries"):
            add("model.llm." + part, "F32", [4, 32])
        formats = {"nvfp4": [], "mxfp8": []}
        for layer in range(4):
            for name, out in (("w13", 64), ("w2", 32)):
                stem = f"model.llm.layers.{layer}.mlp.experts.{name}_weight"
                is_fp4 = layer < 3
                formats["nvfp4" if is_fp4 else "mxfp8"].append(stem)
                add(stem, "U8" if is_fp4 else "F8_E4M3", [2, out, 16 if is_fp4 else 32])
                add(stem + "_scale", "F8_E4M3" if is_fp4 else "U8", [2, out, 2 if is_fp4 else 1])
                if is_fp4:
                    add(stem + "_amax", "F32", [2])
                    add(stem + "_outer_scale_denominator", "F32", [1])
            if layer in (1, 2):
                for port in ("attn", "k", "v", "mlp"):
                    stem = f"model.llm.layers.{layer}.controllers.{port}."
                    for name, shape in (("down", [4, 64]), ("gain", [32, 4]), ("forget", [32, 4])):
                        add(stem + name + "_weight", "BF16", shape)

        params = [
            dict(compute_dtype="nvfp4", weight_transform="nvfp4_nearest", weight_block_shape=[1, 16],
                 tensors=formats["nvfp4"], required_activation_quantization={
                     "amax_type": "per_token", "block_size": 16, "format": "nvfp4_e2m1",
                     "rounding": "nearest", "scale_dtype": "float8_e4m3fn",
                     "scale_selection_rule": "fouroversix_mae"}),
            dict(compute_dtype="mxfp8", weight_transform="mxfp8_nearest", weight_block_shape=[32, 32],
                 tensors=formats["mxfp8"], required_activation_quantization={
                     "block_size": 32, "format": "mxfp8_e4m3", "rounding": "nearest", "scale_dtype": "ue8m0"}),
        ]
        config = {"model_type": "inkling2_loop", "text_config": self.text}
        quant = {"quant_method": "inkling2", "model": {"parameter_groups": params}}
        index = {"metadata": {"total_size": offset}, "weight_map": dict.fromkeys(header, "shard")}
        result = audit(config, index, header, quant)
        self.assertEqual(result["execution"], "NOT_TESTED")
        self.assertEqual(result["executed_expert_layers_by_format"], {"nvfp4": 7, "mxfp8": 1})

        malformed = copy.deepcopy(header)
        malformed[formats["mxfp8"][0] + "_scale"]["dtype"] = "F8_E4M3"
        with self.assertRaisesRegex(ValueError, "dtype or shape"):
            audit(config, index, malformed, quant)
        malformed[formats["mxfp8"][0]]["data_offsets"][1] += 1
        with self.assertRaisesRegex(ValueError, "storage extent"):
            audit(config, index, malformed, quant)
        params[0]["required_activation_quantization"]["scale_selection_rule"] = "other"
        with self.assertRaisesRegex(ValueError, "activation|quantization contract"):
            audit(config, index, header, quant)


if __name__ == "__main__":
    unittest.main()
