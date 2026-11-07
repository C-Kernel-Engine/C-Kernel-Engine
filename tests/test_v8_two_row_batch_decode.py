"""Guard the mapped M=2 arithmetic and fail-closed generated batch selection."""

from __future__ import annotations

import copy
import ctypes
import json
import math
import platform
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version/v8/scripts"))
from batch_decode_contract_v8 import (  # noqa: E402
    _crossing_at, resolve_mixed_prefill_local_stages, resolve_two_row_batch_contract,
)
from batch_decode_codegen_v8 import emit_two_row_batch_api, emit_full_layer_batch_api  # noqa: E402
from certify_batch_decode_v8 import _measure_native_call, _require_projection_groups  # noqa: E402
from codegen_core_v8 import emit_decode_function  # noqa: E402
from codegen_v8 import _emit_runtime_capability_api  # noqa: E402
from server.serving_bundle import verified_loaded_symbol_backing  # noqa: E402


def _fixture() -> tuple[list[dict], dict, dict]:
    buffers = [
        {"name": name, "size": size, "abs_offset": offset, "define": macro,
         "lifetime": lifetime, "mutable": True}
        for name, size, offset, macro, lifetime in (
            ("kv_cache", 64, 64, "A_KV_CACHE", "sequence"),
            ("embedded_input", 128, 128, "A_EMBEDDED_INPUT", "call"),
            ("residual", 128, 256, "A_RESIDUAL", "call"),
            ("q_scratch", 64, 384, "A_Q_SCRATCH", "call"),
            ("k_scratch", 32, 448, "A_K_SCRATCH", "call"),
            ("logits", 256, 512, "A_LOGITS", "call"),
        )
    ]
    layout = {"memory": {"arena": {"total_size": 768}, "activations": {"buffers": buffers}}}
    def arg(name: str, expr: str, ref: str | None = None, source: str = "dim:test") -> dict:
        row = {"name": name, "expr": expr, "source": source}
        if ref:
            row["buffer_ref"] = ref
        return row
    ops = [
        {"op": "dense_embedding_lookup", "args": [
            arg("token_ids", "(int32_t*)(model->bump + A_TOKEN_IDS)", "token_ids", "activation:tokens"),
            arg("token_count", "1"),
            arg("output", "(float*)(model->bump + A_EMBEDDED_INPUT)", "embedded_input", "output:output")]},
        {"op": "residual_save", "args": [
            arg("dst", "(void*)(model->bump + A_RESIDUAL)", "residual", "output:dst"),
            arg("src", "(const void*)(model->bump + A_EMBEDDED_INPUT)", "embedded_input", "activation:src"),
            arg("size", "128")]},
        {"op": "attn_norm", "args": [
            arg("input", "(const float*)(model->bump + A_EMBEDDED_INPUT)", "embedded_input", "activation:input"),
            arg("output", "(float*)(model->bump + A_EMBEDDED_INPUT)", "embedded_input", "output:output")]},
        {"op": "q_proj", "layer": 0, "function": "gemv_q5_1_q8_1", "call_abi": {"kernel_id": "gemv_q5_1"}, "args": [
            arg("x", "(const float*)(model->bump + A_EMBEDDED_INPUT)", "embedded_input", "activation:x"),
            arg("y", "(float*)(model->bump + A_Q_SCRATCH)", "q_scratch", "output:y"),
            arg("W", "(const void*)(model->bump + W_LAYER_0_WQ)"),
            arg("M", "16"), arg("K", "32")]},
        {"op": "k_proj", "layer": 0, "function": "gemv_q5_1_q8_1", "call_abi": {"kernel_id": "gemv_q5_1"}, "args": [
            arg("x", "(const float*)(model->bump + A_EMBEDDED_INPUT)", "embedded_input", "activation:x"),
            arg("y", "(float*)(model->bump + A_K_SCRATCH)", "k_scratch", "output:y"),
            arg("W", "(const void*)(model->bump + W_LAYER_0_WK)"),
            arg("M", "8"), arg("K", "32")]},
        {"op": "attention", "args": [
            arg("input", "(const float*)(model->bump + A_EMBEDDED_INPUT)", "embedded_input", "activation:input"),
            arg("residual", "(const float*)(model->bump + A_RESIDUAL)", "residual", "activation:residual"),
            arg("q", "(const float*)(model->bump + A_Q_SCRATCH)", "q_scratch", "scratch:q"),
            arg("k", "(const float*)(model->bump + A_K_SCRATCH)", "k_scratch", "scratch:k")]},
    ]
    return ops, layout, {"vocab_size": 64}


def _recurrent_fixture() -> tuple[list[dict], dict, dict]:
    def arg(name: str, expr: str, ref: str | None = None,
            source: str = "dim:test") -> dict:
        item = {"name": name, "expr": expr, "source": source}
        if ref is not None:
            item["buffer_ref"] = ref
        return item

    specs = (
        ("kv_cache", 64, 64, "sequence"),
        ("embedded_input", 1024, 128, "call"),
        ("residual", 1024, 1152, "call"),
        ("main_stream_q8", 292, 2176, "call"),
        ("recurrent_packed", 64, 2496, "call"),
        ("recurrent_conv_state", 64, 2560, "sequence"),
        ("recurrent_ssm_state", 64, 2624, "sequence"),
        ("logits", 256, 2688, "call"),
    )
    buffers = [
        {"name": name, "size": size, "abs_offset": offset,
         "define": "A_" + name.upper(), "lifetime": lifetime,
         "mutable": True, **({"shape": "[1, 16]", "dtype": "fp32"}
                            if name.startswith("recurrent_") and lifetime == "sequence" else {})}
        for name, size, offset, lifetime in specs
    ]
    layout = {"memory": {"arena": {"total_size": 3008},
                         "activations": {"buffers": buffers}}}
    ops = [
        {"op": "dense_embedding_lookup", "args": [
            arg("token_ids", "(int32_t*)(model->bump + A_TOKEN_IDS)", "token_ids", "activation:tokens"),
            arg("token_count", "1"),
            arg("output", "(float*)(model->bump + A_EMBEDDED_INPUT)", "embedded_input", "output:output")]},
        {"op": "residual_save", "layer": 0, "args": [
            arg("src", "(const void*)(model->bump + A_EMBEDDED_INPUT)", "embedded_input", "activation:src"),
            arg("dst", "(void*)(model->bump + A_RESIDUAL)", "residual", "output:dst"),
            arg("size", "1024")]},
        {"op": "attn_norm", "layer": 0, "args": [
            arg("input", "(const float*)(model->bump + A_EMBEDDED_INPUT)", "embedded_input", "activation:input"),
            arg("output", "(float*)(model->bump + A_EMBEDDED_INPUT)", "embedded_input", "output:output")]},
        {"op": "quantize_input_0", "layer": 0, "args": [
            arg("x", "(const float*)(model->bump + A_EMBEDDED_INPUT)", "embedded_input", "activation:input"),
            arg("y", "(void*)(model->bump + A_MAIN_STREAM_Q8)", "main_stream_q8", "output:output"),
            arg("k", "256"), arg("rows", "1")]},
        {"op": "recurrent_qkv_proj", "layer": 0,
         "function": "gemv_q6_k_q8_k", "call_abi": {"kernel_id": "gemv_q6_k_q8_k"},
         "args": [
             arg("x_q8", "(void*)(model->bump + A_MAIN_STREAM_Q8)", "main_stream_q8", "activation:x"),
             arg("y", "(float*)(model->bump + A_RECURRENT_PACKED)", "recurrent_packed", "output:y"),
             arg("W", "(const void*)(model->bump + W_LAYER_0_ATTN_QKV)"),
             arg("M", "16"), arg("K", "256")]},
        {"op": "recurrent_core", "layer": 0, "args": [
            arg("packed", "(const float*)(model->bump + A_RECURRENT_PACKED)",
                "recurrent_packed", "activation:packed"),
            arg("residual", "(const float*)(model->bump + A_RESIDUAL)", "residual", "activation:residual"),
            arg("conv", "(float*)(model->bump + A_RECURRENT_CONV_STATE)",
                "recurrent_conv_state", "output:conv"),
            arg("ssm", "(float*)(model->bump + A_RECURRENT_SSM_STATE)",
                "recurrent_ssm_state", "output:ssm"),
            arg("output", "(float*)(model->bump + A_EMBEDDED_INPUT)",
                "embedded_input", "output:output")]},
        {"op": "logits", "layer": 1, "args": [
            arg("x", "(const float*)(model->bump + A_EMBEDDED_INPUT)",
                "embedded_input", "activation:x"),
            arg("q8", "(const void*)(model->bump + A_MAIN_STREAM_Q8)",
                "main_stream_q8", "activation:q8")]},
    ]
    return ops, layout, {"num_layers": 1, "vocab_size": 64,
                          "batch_decode_recurrent_first_projection": True,
                          "layer_recurrent_policy": ["recurrent"]}


def _hybrid_recurrent_layout(row_layout: dict) -> dict:
    """Make prefill-sized call arenas with the same checked decode graph."""
    combined = copy.deepcopy(row_layout)
    offset = 4096
    for row in combined["memory"]["activations"]["buffers"]:
        if row["lifetime"] != "call":
            continue
        if row["name"] != "logits":
            row["size"] *= 8192
        row["abs_offset"] = offset
        offset += row["size"] + 64
    combined["memory"]["arena"]["total_size"] = offset
    return combined


def _recurrent_two_group_fixture() -> tuple[list[dict], dict, dict]:
    ops, layout, config = _recurrent_fixture()
    config["batch_decode_recurrent_output_projection"] = True
    layout["memory"]["activations"]["buffers"].append({
        "name": "recurrent_normed", "size": 128, "abs_offset": 3008,
        "define": "A_RECURRENT_NORMED", "lifetime": "call", "mutable": True,
    })
    layout["memory"]["arena"]["total_size"] = 3200
    output = next(arg for arg in ops[5]["args"] if arg["name"] == "output")
    output["buffer_ref"] = "recurrent_normed"
    output["expr"] = "(float*)(model->bump + A_RECURRENT_NORMED)"
    ops[6:6] = [
        {"op": "recurrent_norm_gate", "layer": 0, "args": [
            {"name": "x", "expr": "(const float*)(model->bump + A_RECURRENT_NORMED)",
             "source": "activation:x", "buffer_ref": "recurrent_normed"},
            {"name": "out", "expr": "(float*)(model->bump + A_RECURRENT_NORMED)",
             "source": "output:out", "buffer_ref": "recurrent_normed"}]},
        {"op": "recurrent_out_proj", "layer": 0,
         "function": "gemv_q8_0_q8_0_contract",
         "call_abi": {"kernel_id": "gemv_q8_0_q8_0_contract"}, "args": [
            {"name": "x", "expr": "(const float*)(model->bump + A_RECURRENT_NORMED)",
             "source": "activation:x", "buffer_ref": "recurrent_normed"},
            {"name": "y", "expr": "(float*)(model->bump + A_EMBEDDED_INPUT)",
             "source": "output:y", "buffer_ref": "embedded_input"},
            {"name": "W", "expr": "(const void*)(model->bump + W_LAYER_0_SSM_OUT)",
             "source": "weight:W"},
            {"name": "M", "expr": "16", "source": "dim:M"},
            {"name": "K", "expr": "32", "source": "dim:K"}]},
    ]
    ops[8]["args"].append({
        "name": "residual", "expr": "(const float*)(model->bump + A_RESIDUAL)",
        "source": "activation:residual", "buffer_ref": "residual",
    })
    return ops, layout, config


def _recurrent_three_group_fixture() -> tuple[list[dict], dict, dict]:
    ops, layout, config = _recurrent_two_group_fixture()
    config["batch_decode_recurrent_gate_projection"] = True
    layout["memory"]["activations"]["buffers"].append({
        "name": "recurrent_z", "size": 64, "abs_offset": 3200,
        "define": "A_RECURRENT_Z", "lifetime": "call", "mutable": True,
    })
    layout["memory"]["arena"]["total_size"] = 3328
    ops.insert(5, {"op": "recurrent_gate_proj", "layer": 0,
                   "function": "gemv_q4_k_q8_k_repacked_parallel_dispatch",
                   "call_abi": {"kernel_id": "gemv_q4_k_q8_k"}, "args": [
        {"name": "y", "expr": "(float*)(model->bump + A_RECURRENT_Z)",
         "source": "output:y", "buffer_ref": "recurrent_z"},
        {"name": "W", "expr": "(const void*)(model->bump + W_LAYER_0_ATTN_GATE)",
         "source": "weight:_first_weight"},
        {"name": "x_q8", "expr": "(void*)(model->bump + A_MAIN_STREAM_Q8)",
         "source": "activation:x", "buffer_ref": "main_stream_q8"},
        {"name": "M", "expr": "16", "source": "dim:_output_dim"},
        {"name": "K", "expr": "256", "source": "dim:_input_dim"},
    ]})
    ops[6]["args"].append({
        "name": "gate", "expr": "(const float*)(model->bump + A_RECURRENT_Z)",
        "source": "activation:gate", "buffer_ref": "recurrent_z",
    })
    return ops, layout, config


def _recurrent_six_group_fixture() -> tuple[list[dict], dict, dict]:
    ops, layout, config = _recurrent_three_group_fixture()
    for row in layout["memory"]["activations"]["buffers"]:
        if row["name"] in ("recurrent_conv_state", "recurrent_ssm_state"):
            row["shape"] = "[2, 8]"
    second = copy.deepcopy(ops[1:9])
    for op in second:
        op["layer"] = 1
        for arg in op.get("args", []):
            arg["expr"] = arg["expr"].replace("W_LAYER_0_", "W_LAYER_1_")
            for name in ("RECURRENT_CONV_STATE", "RECURRENT_SSM_STATE"):
                arg["expr"] = arg["expr"].replace(
                    f"model->bump + A_{name})",
                    f"model->bump + (A_{name} + 32))")
    ops[9]["layer"] = 2
    ops[9:9] = second
    config["num_layers"] = 2
    config["layer_recurrent_policy"] = ["recurrent", "recurrent"]
    config["batch_decode_recurrent_shared_layers"] = 2
    return ops, layout, config


def _recurrent_nine_group_fixture() -> tuple[list[dict], dict, dict]:
    ops, layout, config = _recurrent_six_group_fixture()
    third = copy.deepcopy([op for op in ops if op.get("layer") == 1])
    for op in third:
        op["layer"] = 2
        for arg in op.get("args", []):
            arg["expr"] = arg["expr"].replace("W_LAYER_1_", "W_LAYER_2_")
            for name in ("RECURRENT_CONV_STATE", "RECURRENT_SSM_STATE"):
                arg["expr"] = arg["expr"].replace(
                    f"model->bump + (A_{name} + 32))",
                    f"model->bump + (A_{name} + 64))")
    last = next(i for i, op in enumerate(ops) if op.get("op") == "logits")
    ops[last]["layer"] = 3
    ops[last:last] = third
    buffers = {row["name"]: row for row in layout["memory"]["activations"]["buffers"]}
    for name in ("recurrent_conv_state", "recurrent_ssm_state"):
        buffers[name].update(size=96, shape="[3, 8]")
    buffers["recurrent_ssm_state"]["abs_offset"] = 2688
    buffers["logits"]["abs_offset"] = 2816
    buffers["recurrent_normed"]["abs_offset"] = 3136
    buffers["recurrent_z"]["abs_offset"] = 3328
    layout["memory"]["arena"]["total_size"] = 3456
    config.update(num_layers=3, layer_recurrent_policy=["recurrent"] * 3,
                  batch_decode_recurrent_shared_layers=3)
    return ops, layout, config


def _layer_fixture() -> tuple[list[dict], dict, dict]:
    ops, layout, config = _fixture()
    def arg(name: str, expr: str, ref: str | None = None, source: str = "dim:test") -> dict:
        item = {"name": name, "expr": expr, "source": source}
        if ref is not None:
            item["buffer_ref"] = ref
        return item
    layout["memory"]["arena"]["total_size"] = 1280
    layout["memory"]["activations"]["buffers"].append(
        {"name": "mlp_scratch", "size": 256, "abs_offset": 800,
         "define": "A_MLP_SCRATCH", "lifetime": "call", "mutable": True})
    ops[5]["layer"] = 0
    ops[5]["args"].append(arg("output", "(float*)(model->bump + A_EMBEDDED_INPUT)",
                              "embedded_input", "output:output"))
    ops.extend([
        {"op": "mlp_gate_up", "layer": 0, "function": "gemv_q5_1_q8_1",
         "call_abi": {"kernel_id": "gemv_q5_1"}, "args": [
             arg("y", "(float*)(model->bump + A_MLP_SCRATCH)", "mlp_scratch", "output:y"),
             arg("W", "(const void*)(model->bump + W_LAYER_0_W1)"),
             arg("x", "(const float*)(model->bump + A_EMBEDDED_INPUT)", "embedded_input", "activation:x"),
             arg("M", "64"), arg("K", "32")]},
        {"op": "geglu", "layer": 0, "args": [
            arg("x", "(const float*)(model->bump + A_MLP_SCRATCH)", "mlp_scratch", "activation:input"),
            arg("out", "(float*)(model->bump + A_MLP_SCRATCH)", "mlp_scratch", "output:output"),
            arg("tokens", "1"), arg("dim", "32")]},
        {"op": "finish", "layer": 0, "args": [
            arg("residual", "(const float*)(model->bump + A_RESIDUAL)", "residual", "activation:residual"),
            arg("input", "(const float*)(model->bump + A_MLP_SCRATCH)", "mlp_scratch", "activation:input"),
            arg("output", "(float*)(model->bump + A_EMBEDDED_INPUT)", "embedded_input", "output:output")]},
    ])
    return ops, layout, config


def _two_layer_fixture() -> tuple[list[dict], dict, dict]:
    fixture = json.loads((ROOT / "tests/fixtures/batch_decode_gemma3_first_layer.json").read_text())
    ops, layout, config = fixture["ops"], fixture["layout"], fixture["config"]
    second = copy.deepcopy(ops[1:21])
    for op in second:
        op["layer"] = 1
        for arg in op.get("args", []):
            arg["expr"] = str(arg["expr"]).replace("W_LAYER_0_", "W_LAYER_1_")
    # The second layer's publisher-selected down projection consumes FP32
    # GeGLU output directly. Its compact 8 KiB row is held in the same
    # 16 KiB buffer that holds the gate/up result.
    second = [op for op in second if op["op"] != "quantize_mlp_down_input"]
    down = next(op for op in second if op["op"] == "mlp_down")
    down["function"] = "gemv_q5_k"
    down["call_abi"]["kernel_id"] = "gemv_q5_k"
    down_input = next(arg for arg in down["args"] if arg["name"] == "x_q8")
    down_input.update({"name": "x", "buffer_ref": "mlp_scratch",
                       "expr": "(const float*)(model->bump + A_MLP_SCRATCH)"})
    final = copy.deepcopy(ops[21])
    final["layer"] = 2
    return ops[:21] + second + [final], layout, config


def _avx2_available() -> bool:
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        return False
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
    except OSError:
        return False
    return any(line.startswith(("flags", "Features")) and
               "avx2" in line.partition(":")[2].split()
               for line in cpuinfo.splitlines())


class BatchContractTests(unittest.TestCase):
    def test_recurrent_asset_gate_never_inherits_a_missing_bundle_pass(self) -> None:
        script = ROOT / "version/v8/scripts/certify_recurrent_batch_decode_v8.py"
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "latest.json"
            report.write_text('{"status":"PASS"}')
            run = subprocess.run(
                [sys.executable, str(script), "--bundle", str(Path(directory) / "missing"),
                 "--report", str(report), "--expected-projection-groups", "9",
                 "--expected-shared-layers", "3"], capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            self.assertEqual(json.loads(report.read_text())["status"], "NOT_TESTED")

    def test_recurrent_first_projection_is_checked_and_state_stays_local(self) -> None:
        ops, layout, config = _recurrent_fixture()
        resolved = resolve_two_row_batch_contract(ops, layout, config)
        self.assertIsNotNone(resolved)
        self.assertIsNone(resolve_two_row_batch_contract(
            ops, layout, {**config, "batch_decode_recurrent_first_projection": False}))
        capability = _emit_runtime_capability_api(
            {"config": config, "operations": ops}, layout, None)
        self.assertIn("CK_MODEL_CAP_BATCH_DECODE_TWO_ROWS", capability)
        ordinary = _emit_runtime_capability_api(
            {"config": {**config, "batch_decode_recurrent_first_projection": False},
             "operations": ops}, layout, None)
        self.assertNotIn("CK_MODEL_CAP_BATCH_DECODE_TWO_ROWS", ordinary)
        plan = resolved["full_layer_plan"]
        self.assertEqual([(s["kind"], s["start"], s["stop"])
                          for s in plan["stages"]],
                         [("local", 0, 4), ("shared", 4, 5), ("local", 5, 7)])
        self.assertEqual(plan["projection_groups"], 1)
        self.assertEqual(plan["stages"][1]["projections"][0]["function"],
                         "gemm_nt_q6_k_q8_k_m2")
        self.assertEqual(plan["extents"]["main_stream_q8"], 292)
        self.assertEqual(plan["extents"]["recurrent_packed"], 64)
        emitted = emit_full_layer_batch_api(plan)
        self.assertIn("live->conv", emitted)
        self.assertIn("live->ssm", emitted)
        self.assertIn("gemm_nt_q6_k_q8_k_m2", emitted)
        self.assertIn("ck_batch_stage_2(g_model, rows[i].token)", emitted)

        # Selected-provider, pointer, and recurrent ownership failures must
        # reject before a generated batch capability can be advertised.
        for label, mutation in (
            ("unmapped provider", lambda o, l: o[4]["call_abi"].update(
                kernel_id="gemv_q4_k_q8_k")),
            ("offset input", lambda o, l: o[4]["args"][0].update(
                expr="(void*)(model->bump + (A_MAIN_STREAM_Q8 + 16))")),
            ("untyped state", lambda o, l: o[5]["args"][2].pop("buffer_ref")),
            ("overlapping state", lambda o, l: l["memory"]["activations"]["buffers"][6].update(
                abs_offset=2560)),
        ):
            with self.subTest(label=label):
                bad_ops, bad_layout = copy.deepcopy(ops), copy.deepcopy(layout)
                mutation(bad_ops, bad_layout)
                if label == "overlapping state":
                    with self.assertRaisesRegex(ValueError, "overlaps"):
                        resolve_two_row_batch_contract(bad_ops, bad_layout, config)
                else:
                    self.assertIsNone(resolve_two_row_batch_contract(
                        bad_ops, bad_layout, config))

    def test_hybrid_recurrent_snapshots_use_checked_decode_row_extents(self) -> None:
        ops, row_layout, config = _recurrent_fixture()
        combined = _hybrid_recurrent_layout(row_layout)
        resolved = resolve_two_row_batch_contract(
            ops, combined, config, row_layout=row_layout)
        self.assertIsNotNone(resolved)
        plan = resolved["full_layer_plan"]
        self.assertEqual(plan["extents"], {
            "embedded_input": 1024, "residual": 1024,
            "main_stream_q8": 292, "recurrent_packed": 64,
        })
        emitted = emit_full_layer_batch_api(plan)
        self.assertIn("return 9488u;", emitted)  # two snapshots at each cut
        self.assertNotIn("8388608u", emitted)
        capability = _emit_runtime_capability_api(
            {"config": config, "operations": ops}, combined, None,
            row_layout_obj=row_layout)
        self.assertIn("CK_MODEL_CAP_BATCH_DECODE_TWO_ROWS", capability)

        # A mismatched row plan must never advertise the optimized entry.
        for label, change in (
            ("missing row", lambda rows: rows.pop("residual")),
            ("wrong define", lambda rows: rows["residual"].update(define="A_OTHER")),
            ("undersized input", lambda rows: rows["main_stream_q8"].update(size=291)),
            ("oversized row", lambda rows: rows["recurrent_packed"].update(size=999999999)),
        ):
            with self.subTest(label=label):
                bad_layout = copy.deepcopy(row_layout)
                rows = {row["name"]: row for row in bad_layout["memory"]["activations"]["buffers"]}
                change(rows)
                if label == "missing row":
                    bad_layout["memory"]["activations"]["buffers"] = list(rows.values())
                self.assertIsNone(resolve_two_row_batch_contract(
                    ops, combined, config, row_layout=bad_layout))

    def test_second_recurrent_projection_preserves_all_live_values(self) -> None:
        ops, row_layout, config = _recurrent_two_group_fixture()
        combined = _hybrid_recurrent_layout(row_layout)
        plan = resolve_two_row_batch_contract(
            ops, combined, config, row_layout=row_layout)["full_layer_plan"]
        self.assertEqual(plan["projection_groups"], 2)
        self.assertEqual([(s["kind"], s["start"], s["stop"])
                          for s in plan["stages"]],
                         [("local", 0, 4), ("shared", 4, 5),
                          ("local", 5, 7), ("shared", 7, 8),
                          ("local", 8, 9)])
        self.assertIn("recurrent_normed", plan["stages"][2]["live_after"])
        self.assertIn("residual", plan["stages"][3]["live_after"])
        self.assertIn("gemm_nt_q8_0_q8_0_contract_m2",
                      emit_full_layer_batch_api(plan))
        self.assertEqual(resolve_two_row_batch_contract(
            ops, combined, {**config, "batch_decode_recurrent_output_projection": False},
            row_layout=row_layout)["full_layer_plan"]["projection_groups"], 1)
        self.assertIsNone(resolve_two_row_batch_contract(
            ops, combined, {**config, "batch_decode_recurrent_output_projection": "true"},
            row_layout=row_layout))

        for label, mutation in (
            ("unmapped output", lambda o, l: o[7]["call_abi"].update(
                kernel_id="gemv_q4_k_q8_k")),
            ("wrong layer", lambda o, l: o[7].update(layer=1)),
            ("missing core", lambda o, l: o[5].update(op="other")),
            ("hidden live value", lambda o, l: (
                o[6]["args"].append({
                    "name": "extra", "expr": "(float*)(model->bump + A_MISSING)",
                    "source": "output:extra", "buffer_ref": "missing"}),
                o[8]["args"].append({
                    "name": "extra", "expr": "(const float*)(model->bump + A_MISSING)",
                    "source": "activation:extra", "buffer_ref": "missing"}))),
            ("offset input", lambda o, l: o[7]["args"][0].update(
                expr="(const float*)(model->bump + (A_RECURRENT_NORMED + 4))")),
        ):
            with self.subTest(label=label):
                bad_ops, bad_layout = copy.deepcopy(ops), copy.deepcopy(combined)
                mutation(bad_ops, bad_layout)
                self.assertIsNone(resolve_two_row_batch_contract(
                    bad_ops, bad_layout, config, row_layout=row_layout))

    def test_recurrent_gate_uses_map_selected_decode_order_provider(self) -> None:
        ops, row_layout, config = _recurrent_three_group_fixture()
        combined = _hybrid_recurrent_layout(row_layout)
        admitted = resolve_two_row_batch_contract(
            ops, combined, config, row_layout=row_layout)
        self.assertIsNotNone(admitted)
        plan = admitted["full_layer_plan"]
        self.assertEqual(plan["projection_groups"], 3)
        self.assertEqual([(stage["kind"], stage["start"], stage["stop"])
                          for stage in plan["stages"]],
                         [("local", 0, 4), ("shared", 4, 6),
                          ("local", 6, 8), ("shared", 8, 9),
                          ("local", 9, 10)])
        self.assertEqual([p["function"] for p in plan["stages"][1]["projections"]],
                         ["gemm_nt_q6_k_q8_k_m2", "gemm_nt_q4_k_q8_k_decode_m2"])
        self.assertIn("recurrent_z", plan["stages"][1]["live_after"])
        self.assertIn("gemm_nt_q4_k_q8_k_decode_m2", emit_full_layer_batch_api(plan))
        without_gate = resolve_two_row_batch_contract(
            ops, combined, {**config, "batch_decode_recurrent_gate_projection": False},
            row_layout=row_layout)
        self.assertEqual(without_gate["full_layer_plan"]["projection_groups"], 2)
        self.assertIsNone(resolve_two_row_batch_contract(
            ops, combined, {**config, "batch_decode_recurrent_gate_projection": "true"},
            row_layout=row_layout))
        for label, change in (
            ("unmapped gate", lambda o, l: o[5]["call_abi"].update(kernel_id="gemv_q5_k")),
            ("wrong gate layer", lambda o, l: o[5].update(layer=1)),
            ("offset input", lambda o, l: o[5]["args"][2].update(
                expr="(void*)(model->bump + (A_MAIN_STREAM_Q8 + 1))")),
            ("overlapping gate output", lambda o, l: next(
                b for b in l["memory"]["activations"]["buffers"]
                if b["name"] == "recurrent_z").update(abs_offset=next(
                    b["abs_offset"] for b in l["memory"]["activations"]["buffers"]
                    if b["name"] == "recurrent_packed"))),
        ):
            with self.subTest(label=label):
                bad_ops, bad_layout = copy.deepcopy(ops), copy.deepcopy(combined)
                change(bad_ops, bad_layout)
                self.assertIsNone(resolve_two_row_batch_contract(
                    bad_ops, bad_layout, config, row_layout=row_layout))

    def test_two_recurrent_layers_share_only_mapped_projection_cuts(self) -> None:
        ops, row_layout, config = _recurrent_six_group_fixture()
        combined = _hybrid_recurrent_layout(row_layout)
        admitted = resolve_two_row_batch_contract(
            ops, combined, config, row_layout=row_layout)
        self.assertIsNotNone(admitted)
        plan = admitted["full_layer_plan"]
        self.assertEqual(plan["shared_layers"], [0, 1])
        self.assertEqual(plan["projection_groups"], 6)
        self.assertEqual([stage["kind"] for stage in plan["stages"]],
                         ["local", "shared", "local", "shared", "local",
                          "shared", "local", "shared", "local"])
        self.assertEqual([len(stage["projections"]) for stage in plan["stages"]
                          if stage["kind"] == "shared"], [2, 1, 2, 1])
        self.assertIn("recurrent_z", plan["stages"][5]["live_after"])
        self.assertIn("W_LAYER_1_ATTN_GATE", emit_full_layer_batch_api(plan))
        self.assertIsNone(resolve_two_row_batch_contract(
            ops, combined, {**config, "batch_decode_recurrent_shared_layers": 3},
            row_layout=row_layout))
        for label, mutation in (
            ("missing second QKV", lambda o: o[12].update(op="unsupported_qkv")),
            ("wrong second gate binding", lambda o: o[13]["call_abi"].update(
                kernel_id="gemv_q5_k")),
            ("noncontiguous layer", lambda o: o[14].update(layer=0)),
            ("layer order resumes earlier layer", lambda o: o[17].update(layer=0)),
            ("wrong layer state slice", lambda o: next(
                arg for arg in o[14]["args"] if arg["name"] == "conv").update(
                    expr="(float*)(model->bump + A_RECURRENT_CONV_STATE)")),
        ):
            with self.subTest(label=label):
                bad = copy.deepcopy(ops)
                mutation(bad)
                self.assertIsNone(resolve_two_row_batch_contract(
                    bad, combined, config, row_layout=row_layout))
        # A value produced after the first gate and used in layer 1 must be
        # preserved at the intervening cuts. An unproven extent fails closed.
        bad = copy.deepcopy(ops)
        bad[7]["args"].append({"name": "extra", "source": "output:extra",
            "expr": "(float*)(model->bump + A_UNPROVEN)", "buffer_ref": "unproven"})
        bad[14]["args"].append({"name": "extra", "source": "activation:extra",
            "expr": "(const float*)(model->bump + A_UNPROVEN)", "buffer_ref": "unproven"})
        self.assertIsNone(resolve_two_row_batch_contract(
            bad, combined, config, row_layout=row_layout))

    def test_two_recurrent_layers_execute_generated_entry(self) -> None:
        ops, row_layout, config = _recurrent_six_group_fixture()
        plan = resolve_two_row_batch_contract(
            ops, _hybrid_recurrent_layout(row_layout), config,
            row_layout=row_layout)["full_layer_plan"]
        fixture = (ROOT / "tests/fixtures/recurrent_multi_layer_entry.c").read_text()
        emitted = emit_full_layer_batch_api(plan)
        self.assertIn("return 6;", emitted)
        self.assertIn("return 2;", emitted)
        source = fixture.replace("/* BATCH_API */", emitted)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "entry.c"
            binary = Path(directory) / "entry"

            def run(text: str) -> subprocess.CompletedProcess[str]:
                path.write_text(text)
                built = subprocess.run(
                    ["cc", "-std=c11", "-O0", "-Wall", "-Werror",
                     "-Wno-unused-function", "-Wno-unused-variable",
                     str(path), "-o", str(binary)], capture_output=True, text=True)
                self.assertEqual(built.returncode, 0, built.stderr)
                return subprocess.run([str(binary)], capture_output=True, text=True)

            result = run(source)
            self.assertEqual(result.returncode, 0, result.stderr)
            restores = list(re.finditer(
                r"(?m)^        ck_batch_memcpy\(g_model->bump \+ A_RECURRENT_Z, base \+ [^\n]+\n",
                source))
            self.assertGreaterEqual(len(restores), 2)
            missing_layer_one_gate = (source[:restores[-1].start()]
                                      + source[restores[-1].end():])
            self.assertNotEqual(run(missing_layer_one_gate).returncode, 0)
            wrong_owner = source.replace(
                "static void ck_batch_stage_6(CKModel *model, int token) {\n"
                "    (void)token;\n"
                "    CKSequenceStateV8 *s = g_active_sequence;",
                "static void ck_batch_stage_6(CKModel *model, int token) {\n"
                "    (void)token;\n"
                "    CKSequenceStateV8 *s = &g_sequence_states[0];")
            self.assertNotEqual(wrong_owner, source)
            self.assertNotEqual(run(wrong_owner).returncode, 0)

    def test_three_recurrent_layers_share_checked_projection_cuts(self) -> None:
        ops, row_layout, config = _recurrent_nine_group_fixture()
        combined = _hybrid_recurrent_layout(row_layout)
        plan = resolve_two_row_batch_contract(
            ops, combined, config, row_layout=row_layout)["full_layer_plan"]
        self.assertEqual(plan["shared_layers"], [0, 1, 2])
        self.assertEqual(plan["projection_groups"], 9)
        self.assertEqual([len(stage["projections"]) for stage in plan["stages"]
                          if stage["kind"] == "shared"], [2, 1, 2, 1, 2, 1])
        emitted = emit_full_layer_batch_api(plan)
        self.assertIn("W_LAYER_2_ATTN_QKV", emitted)
        self.assertIn("W_LAYER_2_SSM_OUT", emitted)
        for label, mutate in (
            ("missing third gate", lambda o: next(op for op in o
                 if op.get("layer") == 2 and op.get("op") == "recurrent_gate_proj")
             .update(op="unsupported_gate")),
            ("wrong third state slice", lambda o: next(arg for op in o
                 if op.get("layer") == 2 and op.get("op") == "recurrent_core"
                 for arg in op["args"] if arg.get("name") == "conv")
             .update(expr="(float*)(model->bump + (A_RECURRENT_CONV_STATE + 32))")),
            ("backtracking", lambda o: next(op for op in o
                 if op.get("layer") == 2 and op.get("op") == "recurrent_core")
             .update(layer=1)),
        ):
            with self.subTest(label=label):
                bad = copy.deepcopy(ops)
                mutate(bad)
                self.assertIsNone(resolve_two_row_batch_contract(
                    bad, combined, config, row_layout=row_layout))
        # A later layer may consume a value created before the prior cut.
        # Unknown extents cannot be reconstructed between the two rows.
        bad = copy.deepcopy(ops)
        first_layer_one = next(op for op in bad if op.get("layer") == 1
                               and op.get("op") == "recurrent_core")
        first_layer_two = next(op for op in bad if op.get("layer") == 2
                               and op.get("op") == "recurrent_core")
        first_layer_one["args"].append({
            "name": "extra", "source": "output:extra",
            "expr": "(float*)(model->bump + A_UNPROVEN)", "buffer_ref": "unproven"})
        first_layer_two["args"].append({
            "name": "extra", "source": "activation:extra",
            "expr": "(const float*)(model->bump + A_UNPROVEN)", "buffer_ref": "unproven"})
        self.assertIsNone(resolve_two_row_batch_contract(
            bad, combined, config, row_layout=row_layout))
        self.assertIsNone(resolve_two_row_batch_contract(
            ops, combined, {**config, "batch_decode_recurrent_shared_layers": 4},
            row_layout=row_layout))

    def test_three_recurrent_layers_execute_generated_entry(self) -> None:
        ops, row_layout, config = _recurrent_nine_group_fixture()
        plan = resolve_two_row_batch_contract(
            ops, _hybrid_recurrent_layout(row_layout), config,
            row_layout=row_layout)["full_layer_plan"]
        fixture = (ROOT / "tests/fixtures/recurrent_multi_layer_entry.c").read_text()
        source = "#define RECUR_THREE_LAYER 1\n" + fixture.replace(
            "/* BATCH_API */", emit_full_layer_batch_api(plan))
        self.assertIn("return 9;", source)
        self.assertIn("return 3;", source)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "entry.c"
            binary = Path(directory) / "entry"

            def run(text: str) -> subprocess.CompletedProcess[str]:
                path.write_text(text)
                built = subprocess.run(
                    ["cc", "-std=c11", "-O0", "-Wall", "-Werror",
                     "-Wno-unused-function", "-Wno-unused-variable",
                     str(path), "-o", str(binary)], capture_output=True, text=True)
                self.assertEqual(built.returncode, 0, built.stderr)
                return subprocess.run([str(binary)], capture_output=True, text=True)

            self.assertEqual(run(source).returncode, 0)
            restores = list(re.finditer(
                r"(?m)^        ck_batch_memcpy\(g_model->bump \+ A_RECURRENT_Z, base \+ [^\n]+\n",
                source))
            self.assertGreaterEqual(len(restores), 3)
            missing_third_gate = (source[:restores[-1].start()]
                                  + source[restores[-1].end():])
            self.assertNotEqual(run(missing_third_gate).returncode, 0)
            wrong_owner = source.replace(
                "static void ck_batch_stage_10(CKModel *model, int token) {\n"
                "    (void)token;\n"
                "    CKSequenceStateV8 *s = g_active_sequence;",
                "static void ck_batch_stage_10(CKModel *model, int token) {\n"
                "    (void)token;\n"
                "    CKSequenceStateV8 *s = &g_sequence_states[0];")
            self.assertNotEqual(wrong_owner, source)
            self.assertNotEqual(run(wrong_owner).returncode, 0)

    def test_recurrent_generated_entry_matches_isolated_state(self) -> None:
        ops, row_layout, config = _recurrent_fixture()
        combined = _hybrid_recurrent_layout(row_layout)
        plan = resolve_two_row_batch_contract(
            ops, combined, config, row_layout=row_layout)["full_layer_plan"]
        emitted = emit_full_layer_batch_api(plan)
        source_text = r'''
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#define CK_EXPORT
#define VOCAB_SIZE 16
#define MAX_SEQ_LEN 16
#define KV_CACHE_SIZE 64
#define A_EMBEDDED_INPUT 128
#define A_RESIDUAL 1152
#define A_MAIN_STREAM_Q8 2176
#define A_RECURRENT_PACKED 2496
#define W_LAYER_0_ATTN_QKV 3008
typedef struct { uint64_t sequence_handle; int token, position; size_t row_offset, token_count; float *logits; } CKModelBatchDecodeRowV8;
typedef struct { uint8_t *bump; size_t bump_size; int pos; float logits[VOCAB_SIZE]; } CKModel;
typedef struct { int occupied, pos; uint64_t handle; uint8_t *kv, *conv, *ssm; } CKSequenceStateV8;
static CKModel object, *g_model=&object;
static CKSequenceStateV8 g_sequence_states[3], *g_active_sequence;
static int g_ck_skip_decode_logits;
static int g_owner_fault;
static CKSequenceStateV8 *ck_sequence_find(uint64_t handle) {
    for (int i=0;i<3;++i) if (g_sequence_states[i].occupied && g_sequence_states[i].handle==handle) return &g_sequence_states[i];
    return NULL;
}
static int ck_model_sequence_state_activate(uint64_t handle) {
    CKSequenceStateV8 *next=ck_sequence_find(handle);
    if (!next) return -1;
    if (g_active_sequence) g_active_sequence->pos=g_model->pos;
    g_model->pos=next->pos; g_active_sequence=next;
    return 0;
}
static int ck_model_cancel_requested(void) { return 0; }
static void ck_batch_stage_0(CKModel *model, int token) {
    ((float *)(model->bump+A_EMBEDDED_INPUT))[0]=(float)(token*2);
    ((float *)(model->bump+A_RESIDUAL))[0]=(float)(token+20);
    model->bump[A_MAIN_STREAM_Q8]=(uint8_t)(token+1);
}
static void gemm_nt_q6_k_q8_k_m2(const void *input, const void *weight,
                                  const float *bias, float *output,
                                  int rows, int channels, int width) {
    (void)bias; (void)width;
    for (int row=0;row<rows;++row) for (int col=0;col<channels;++col)
        output[row*channels+col]=(float)(((const uint8_t *)input)[row*292]+col+
                                            ((const uint8_t *)weight)[0]);
}
static void ck_batch_stage_2(CKModel *model, int token) {
    float *packed=(float *)(model->bump+A_RECURRENT_PACKED);
    float *residual=(float *)(model->bump+A_RESIDUAL);
    g_active_sequence->conv[0]+=(uint8_t)packed[0];
    g_active_sequence->ssm[0]+=(uint8_t)residual[0];
    (g_owner_fault ? &g_sequence_states[0] : g_active_sequence)->kv[model->pos]=(uint8_t)token;
    for (int j=0;j<VOCAB_SIZE;++j)
        model->logits[j]=packed[j]+(float)g_active_sequence->conv[0]+
                         (float)g_active_sequence->ssm[0]+residual[0];
    model->pos++;
}
''' + emitted + r'''
static void isolated(int token) {
    ck_batch_stage_0(g_model,token);
    gemm_nt_q6_k_q8_k_m2(g_model->bump+A_MAIN_STREAM_Q8,
                          g_model->bump+W_LAYER_0_ATTN_QKV,NULL,
                          (float *)(g_model->bump+A_RECURRENT_PACKED),1,16,256);
    ck_batch_stage_2(g_model,token);
}
int main(void) {
    g_model->bump=calloc(1,4096); g_model->bump_size=4096;
    if (!g_model->bump) return 1;
    g_model->bump[W_LAYER_0_ATTN_QKV]=3;
    static uint8_t kv[3][64]={{0}},conv[3][64]={{0}},ssm[3][64]={{0}};
    for (int i=0;i<3;++i) {
        g_sequence_states[i]=(CKSequenceStateV8){.occupied=1,.pos=i,.handle=(uint64_t)(i+1),
            .kv=kv[i],.conv=conv[i],.ssm=ssm[i]};
        conv[i][0]=(uint8_t)(i+1); ssm[i][0]=(uint8_t)(i+4);
    }
    float expected[2][VOCAB_SIZE],actual[2][VOCAB_SIZE]={{0}};
    uint8_t expected_kv[3][64],expected_conv[3][64],expected_ssm[3][64];
    int expected_pos[3];
    for (int i=0;i<2;++i) {
        if (ck_model_sequence_state_activate((uint64_t)(i+1))) return 2;
        isolated(i ? 7 : 3);
        memcpy(expected[i],g_model->logits,sizeof(expected[i]));
    }
    if (kv[0][0]!=3 || kv[1][1]!=7 || kv[0][1] || kv[1][0] ||
        conv[0][0]!=8 || conv[1][0]!=13 ||
        ssm[0][0]!=27 || ssm[1][0]!=32) return 14;
    if (g_active_sequence) g_active_sequence->pos=g_model->pos;
    memcpy(expected_kv,kv,sizeof(kv)); memcpy(expected_conv,conv,sizeof(conv));
    memcpy(expected_ssm,ssm,sizeof(ssm));
    for (int i=0;i<3;++i) expected_pos[i]=g_sequence_states[i].pos;
    memset(kv,0,sizeof(kv)); memset(conv,0,sizeof(conv)); memset(ssm,0,sizeof(ssm));
    for (int i=0;i<3;++i) {
        g_sequence_states[i].pos=i; conv[i][0]=(uint8_t)(i+1); ssm[i][0]=(uint8_t)(i+4);
    }
    g_active_sequence=NULL; g_model->pos=0;
    CKModelBatchDecodeRowV8 rows[2]={
        {.sequence_handle=1,.token=3,.position=0,.row_offset=0,.token_count=1,.logits=actual[0]},
        {.sequence_handle=2,.token=7,.position=1,.row_offset=1,.token_count=1,.logits=actual[1]}
    };
    size_t bytes=0,alignment=0;
    if (ck_model_batch_decode_projection_groups()!=1 ||
        ck_model_batch_decode_workspace(&bytes,&alignment) || alignment!=64) return 3;
    void *workspace=aligned_alloc(alignment,(bytes+alignment-1)/alignment*alignment);
    if (!workspace) return 4;
    if (ck_model_decode_batch2(rows,2,workspace,bytes)) return 5;
    if (g_active_sequence) g_active_sequence->pos=g_model->pos;
    for (int i=0;i<3;++i) {
        if (g_sequence_states[i].pos!=expected_pos[i] ||
            memcmp(kv[i],expected_kv[i],64) ||
            memcmp(conv[i],expected_conv[i],64) ||
            memcmp(ssm[i],expected_ssm[i],64)) return 6+i;
        if (i<2 && memcmp(actual[i],expected[i],sizeof(expected[i]))) return 10+i;
    }
    /* A third live sequence's convolution arena cannot serve as workspace. */
    uint8_t *aliased=aligned_alloc(64,(bytes+63)/64*64);
    if (!aliased) return 12;
    memset(aliased,0x5a,bytes); g_sequence_states[2].conv=aliased;
    if (ck_model_decode_batch2(rows,2,aliased,bytes)!=-2 || aliased[0]!=0x5a)
        return 13;
    free(aliased); free(workspace); free(g_model->bump);
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "recurrent_entry.c"
            binary = Path(directory) / "recurrent_entry"
            def run(text: str) -> subprocess.CompletedProcess[str]:
                source.write_text(text)
                built = subprocess.run(
                    ["cc", "-std=c11", "-O0", "-Wall", "-Werror",
                     "-Wno-unused-function", "-Wno-unused-variable",
                     str(source), "-o", str(binary)], capture_output=True, text=True)
                self.assertEqual(built.returncode, 0, built.stderr)
                return subprocess.run([str(binary)], capture_output=True, text=True)

            self.assertEqual(run(source_text).returncode, 0)
            missing_restore = re.sub(
                r"(?m)^        ck_batch_memcpy\(g_model->bump \+ A_RECURRENT_PACKED, base \+ [^\n]+\n",
                "", source_text, count=1)
            self.assertNotEqual(missing_restore, source_text)
            self.assertNotEqual(run(missing_restore).returncode, 0)
            wrong_owner = source_text.replace(
                "if (ck_model_decode_batch2(rows,2,workspace,bytes)) return 5;",
                "g_owner_fault=1; if (ck_model_decode_batch2(rows,2,workspace,bytes)) return 5;")
            self.assertNotEqual(wrong_owner, source_text)
            self.assertNotEqual(run(wrong_owner).returncode, 0)

    def test_recurrent_two_group_generated_entry_matches_isolated_state(self) -> None:
        ops, row_layout, config = _recurrent_two_group_fixture()
        plan = resolve_two_row_batch_contract(
            ops, _hybrid_recurrent_layout(row_layout), config,
            row_layout=row_layout)["full_layer_plan"]
        emitted = emit_full_layer_batch_api(plan)
        source_text = r'''
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#define CK_EXPORT
#define VOCAB_SIZE 16
#define MAX_SEQ_LEN 16
#define KV_CACHE_SIZE 64
#define A_EMBEDDED_INPUT 128
#define A_RESIDUAL 1152
#define A_MAIN_STREAM_Q8 2176
#define A_RECURRENT_PACKED 2496
#define A_RECURRENT_NORMED 3008
#define W_LAYER_0_ATTN_QKV 3400
#define W_LAYER_0_SSM_OUT 3408
typedef struct { uint64_t sequence_handle; int token, position; size_t row_offset, token_count; float *logits; } CKModelBatchDecodeRowV8;
typedef struct { uint8_t *bump; size_t bump_size; int pos; float logits[VOCAB_SIZE]; } CKModel;
typedef struct { int occupied, pos; uint64_t handle; uint8_t *kv, *conv, *ssm; } CKSequenceStateV8;
static CKModel object, *g_model=&object;
static CKSequenceStateV8 g_sequence_states[3], *g_active_sequence;
static int g_ck_skip_decode_logits;
static int g_owner_fault;
static CKSequenceStateV8 *ck_sequence_find(uint64_t handle) {
    for (int i=0;i<3;++i) if (g_sequence_states[i].occupied && g_sequence_states[i].handle==handle) return &g_sequence_states[i];
    return NULL;
}
static int ck_model_sequence_state_activate(uint64_t handle) {
    CKSequenceStateV8 *next=ck_sequence_find(handle);
    if (!next) return -1;
    if (g_active_sequence) g_active_sequence->pos=g_model->pos;
    g_model->pos=next->pos; g_active_sequence=next;
    return 0;
}
static int ck_model_cancel_requested(void) { return 0; }
static void ck_batch_stage_0(CKModel *model, int token) {
    ((float *)(model->bump+A_EMBEDDED_INPUT))[0]=(float)(token*2);
    ((float *)(model->bump+A_RESIDUAL))[0]=(float)(token+20);
    model->bump[A_MAIN_STREAM_Q8]=(uint8_t)(token+1);
}
static void gemm_nt_q6_k_q8_k_m2(const void *input, const void *weight,
                                  const float *bias, float *output,
                                  int rows, int channels, int width) {
    (void)bias; (void)width;
    for (int row=0;row<rows;++row) for (int col=0;col<channels;++col)
        output[row*channels+col]=(float)(((const uint8_t *)input)[row*292]+col+
                                            ((const uint8_t *)weight)[0]);
}
static void ck_batch_stage_2(CKModel *model, int token) {
    (void)token;
    float *packed=(float *)(model->bump+A_RECURRENT_PACKED);
    float *residual=(float *)(model->bump+A_RESIDUAL);
    float *normed=(float *)(model->bump+A_RECURRENT_NORMED);
    g_active_sequence->conv[0]+=(uint8_t)packed[0];
    g_active_sequence->ssm[0]+=(uint8_t)residual[0];
    for (int j=0;j<32;++j)
        normed[j]=packed[j%16]+(float)g_active_sequence->conv[0]+(float)g_active_sequence->ssm[0];
}
static void gemm_nt_q8_0_q8_0_contract_m2(const float *input, const void *weight,
                                           const float *bias, float *output,
                                           int rows, int channels, int width) {
    (void)bias;
    for (int row=0;row<rows;++row) for (int col=0;col<channels;++col)
        output[row*channels+col]=input[row*width+col]+((const uint8_t *)weight)[0];
}
static void ck_batch_stage_4(CKModel *model, int token) {
    float *output=(float *)(model->bump+A_EMBEDDED_INPUT);
    float *residual=(float *)(model->bump+A_RESIDUAL);
    (g_owner_fault ? &g_sequence_states[0] : g_active_sequence)->kv[model->pos]=(uint8_t)token;
    for (int j=0;j<VOCAB_SIZE;++j)
        model->logits[j]=output[j]+residual[0]+(float)g_active_sequence->conv[0];
    model->pos++;
}
''' + emitted + r'''
static void isolated(int token) {
    ck_batch_stage_0(g_model,token);
    gemm_nt_q6_k_q8_k_m2(g_model->bump+A_MAIN_STREAM_Q8,
                          g_model->bump+W_LAYER_0_ATTN_QKV,NULL,
                          (float *)(g_model->bump+A_RECURRENT_PACKED),1,16,256);
    ck_batch_stage_2(g_model,token);
    gemm_nt_q8_0_q8_0_contract_m2((float *)(g_model->bump+A_RECURRENT_NORMED),
                                   g_model->bump+W_LAYER_0_SSM_OUT,NULL,
                                   (float *)(g_model->bump+A_EMBEDDED_INPUT),1,16,32);
    ck_batch_stage_4(g_model,token);
}
int main(void) {
    g_model->bump=calloc(1,4096); g_model->bump_size=4096;
    if (!g_model->bump) return 1;
    g_model->bump[W_LAYER_0_ATTN_QKV]=3;
    g_model->bump[W_LAYER_0_SSM_OUT]=5;
    static uint8_t kv[3][64]={{0}},conv[3][64]={{0}},ssm[3][64]={{0}};
    for (int i=0;i<3;++i) {
        g_sequence_states[i]=(CKSequenceStateV8){.occupied=1,.pos=i,.handle=(uint64_t)(i+1),
            .kv=kv[i],.conv=conv[i],.ssm=ssm[i]};
        conv[i][0]=(uint8_t)(i+1); ssm[i][0]=(uint8_t)(i+4);
    }
    float expected[2][VOCAB_SIZE],actual[2][VOCAB_SIZE]={{0}};
    uint8_t expected_kv[3][64],expected_conv[3][64],expected_ssm[3][64];
    int expected_pos[3];
    for (int i=0;i<2;++i) {
        if (ck_model_sequence_state_activate((uint64_t)(i+1))) return 2;
        isolated(i ? 7 : 3);
        memcpy(expected[i],g_model->logits,sizeof(expected[i]));
    }
    if (g_active_sequence) g_active_sequence->pos=g_model->pos;
    memcpy(expected_kv,kv,sizeof(kv)); memcpy(expected_conv,conv,sizeof(conv));
    memcpy(expected_ssm,ssm,sizeof(ssm));
    for (int i=0;i<3;++i) expected_pos[i]=g_sequence_states[i].pos;
    memset(kv,0,sizeof(kv)); memset(conv,0,sizeof(conv)); memset(ssm,0,sizeof(ssm));
    for (int i=0;i<3;++i) {
        g_sequence_states[i].pos=i; conv[i][0]=(uint8_t)(i+1); ssm[i][0]=(uint8_t)(i+4);
    }
    g_active_sequence=NULL; g_model->pos=0;
    CKModelBatchDecodeRowV8 rows[2]={
        {.sequence_handle=1,.token=3,.position=0,.row_offset=0,.token_count=1,.logits=actual[0]},
        {.sequence_handle=2,.token=7,.position=1,.row_offset=1,.token_count=1,.logits=actual[1]}
    };
    size_t bytes=0,alignment=0;
    if (ck_model_batch_decode_projection_groups()!=2 ||
        ck_model_batch_decode_workspace(&bytes,&alignment) || alignment!=64) return 3;
    void *workspace=aligned_alloc(alignment,(bytes+alignment-1)/alignment*alignment);
    if (!workspace) return 4;
    if (ck_model_decode_batch2(rows,2,workspace,bytes)) return 5;
    if (g_active_sequence) g_active_sequence->pos=g_model->pos;
    for (int i=0;i<3;++i) {
        if (g_sequence_states[i].pos!=expected_pos[i] ||
            memcmp(kv[i],expected_kv[i],64) ||
            memcmp(conv[i],expected_conv[i],64) ||
            memcmp(ssm[i],expected_ssm[i],64)) return 6+i;
        if (i<2 && memcmp(actual[i],expected[i],sizeof(expected[i]))) return 10+i;
    }
    free(workspace); free(g_model->bump);
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "two_groups.c"
            binary = Path(directory) / "two_groups"
            def run(text: str) -> subprocess.CompletedProcess[str]:
                source.write_text(text)
                built = subprocess.run(
                    ["cc", "-std=c11", "-O0", "-Wall", "-Werror",
                     "-Wno-unused-function", "-Wno-unused-variable",
                     str(source), "-o", str(binary)], capture_output=True, text=True)
                self.assertEqual(built.returncode, 0, built.stderr)
                return subprocess.run([str(binary)], capture_output=True, text=True)

            self.assertEqual(run(source_text).returncode, 0)
            restore = list(re.finditer(
                r"(?m)^        ck_batch_memcpy\(g_model->bump \+ A_EMBEDDED_INPUT, base \+ [^\n]+\n",
                source_text))
            self.assertGreaterEqual(len(restore), 2)
            bad = source_text[:restore[-1].start()] + source_text[restore[-1].end():]
            self.assertNotEqual(run(bad).returncode, 0)
            missing_intermediate = re.sub(
                r"(?m)^        ck_batch_memcpy\(base \+ [^\n]+g_model->bump \+ A_RECURRENT_NORMED, [^\n]+\n",
                "", source_text, count=1)
            self.assertNotEqual(missing_intermediate, source_text)
            self.assertNotEqual(run(missing_intermediate).returncode, 0)
            wrong_owner = source_text.replace(
                "if (ck_model_decode_batch2(rows,2,workspace,bytes)) return 5;",
                "g_owner_fault=1; if (ck_model_decode_batch2(rows,2,workspace,bytes)) return 5;")
            self.assertNotEqual(run(wrong_owner).returncode, 0)

            # The same compiled fixture now executes the real emitted
            # three-projection split: QKV and gate share one packed input,
            # then the stateful core remains sequence-local.
            three_ops, three_layout, three_config = _recurrent_three_group_fixture()
            three_plan = resolve_two_row_batch_contract(
                three_ops, _hybrid_recurrent_layout(three_layout), three_config,
                row_layout=three_layout)["full_layer_plan"]
            three_emitted = emit_full_layer_batch_api(three_plan)
            three_source = source_text.replace(emitted, three_emitted)
            three_source = three_source.replace(
                "#define A_RECURRENT_NORMED 3008",
                "#define A_RECURRENT_NORMED 3008\n#define A_RECURRENT_Z 3200\n"
                "#define W_LAYER_0_ATTN_GATE 3416")
            three_source = three_source.replace(
                "static void ck_batch_stage_2(CKModel *model, int token) {",
                """static void gemm_nt_q4_k_q8_k_decode_m2(
    const void *input, const void *weight, const float *bias, float *output,
    int rows, int channels, int width) {
    (void)bias; (void)width;
    for (int row=0;row<rows;++row) for (int col=0;col<channels;++col)
        output[row*channels+col]=(float)(((const uint8_t *)input)[row*292]+col+
                                            ((const uint8_t *)weight)[0]);
}
static void ck_batch_stage_2(CKModel *model, int token) {""")
            three_source = three_source.replace(
                "float *packed=(float *)(model->bump+A_RECURRENT_PACKED);",
                "float *packed=(float *)(model->bump+A_RECURRENT_PACKED);\n"
                "    float *gate=(float *)(model->bump+A_RECURRENT_Z);")
            three_source = three_source.replace(
                "normed[j]=packed[j%16]+(float)g_active_sequence->conv[0]+(float)g_active_sequence->ssm[0];",
                "normed[j]=packed[j%16]+gate[j%16]+(float)g_active_sequence->conv[0]+(float)g_active_sequence->ssm[0];")
            three_source = three_source.replace(
                "    ck_batch_stage_2(g_model,token);",
                "    gemm_nt_q4_k_q8_k_decode_m2(g_model->bump+A_MAIN_STREAM_Q8,\n"
                "        g_model->bump+W_LAYER_0_ATTN_GATE,NULL,\n"
                "        (float *)(g_model->bump+A_RECURRENT_Z),1,16,256);\n"
                "    ck_batch_stage_2(g_model,token);")
            three_source = three_source.replace(
                "g_model->bump[W_LAYER_0_SSM_OUT]=5;",
                "g_model->bump[W_LAYER_0_SSM_OUT]=5;\n"
                "    g_model->bump[W_LAYER_0_ATTN_GATE]=9;")
            three_source = three_source.replace(
                "ck_model_batch_decode_projection_groups()!=2",
                "ck_model_batch_decode_projection_groups()!=3")
            self.assertNotEqual(three_source, source_text)
            self.assertEqual(run(three_source).returncode, 0)
            missing_gate = re.sub(
                r"(?m)^        ck_batch_memcpy\(g_model->bump \+ A_RECURRENT_Z, base \+ [^\n]+\n",
                "", three_source, count=1)
            self.assertNotEqual(missing_gate, three_source)
            self.assertNotEqual(run(missing_gate).returncode, 0)

    def test_recurrent_q6_two_row_provider_matches_selected_decode(self) -> None:
        import random

        lanes = [()]
        if _avx2_available():
            lanes.append(("-mavx2", "-mfma"))
        for flags in lanes:
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                stub = root / "stub.c"
                stub.write_text("int ck_strict_parity_enabled(void) { return 0; }\n")
                library = root / "libq6.so"
                built = subprocess.run(
                    ["cc", "-shared", "-fPIC", "-O2", *flags,
                     "-I", str(ROOT / "include"),
                     str(ROOT / "src/kernels/gemm_kernels_q6k_q8k.c"),
                     str(stub), "-lm", "-o", str(library)],
                    capture_output=True, text=True)
                self.assertEqual(built.returncode, 0, built.stderr)
                native = ctypes.CDLL(str(library))
                single = native.gemv_q6_k_q8_k
                single.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
                                   ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
                batch = native.gemm_nt_q6_k_q8_k_m2
                batch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                                  ctypes.c_int, ctypes.c_int]
                for width in (256, 512, 1024, 5120):
                    channels = 7
                    rng = random.Random(911 + width)
                    weights = bytearray()
                    for block in range(channels * width // 256):
                        weights += bytes(rng.randrange(256) for _ in range(192))
                        weights += struct.pack(
                            "<16b", *(rng.randrange(-13, 14) for _ in range(16)))
                        weights += struct.pack("<e", .01 * (block % 7 + 1))
                    inputs = bytearray()
                    for row in range(2):
                        for block in range(width // 256):
                            values = [rng.randrange(-50, 51) for _ in range(256)]
                            inputs += struct.pack(
                                "<f256b16h", .008 * (row + block + 1), *values,
                                *(sum(values[i:i + 16]) for i in range(0, 256, 16)))
                    weight_buffer = ctypes.create_string_buffer(bytes(weights))
                    input_buffer = ctypes.create_string_buffer(bytes(inputs))
                    output = (ctypes.c_float * (2 * channels))()
                    batch(input_buffer, weight_buffer, None, output,
                          2, channels, width)
                    for row in range(2):
                        expected = (ctypes.c_float * channels)()
                        input_row = ctypes.c_void_p(
                            ctypes.addressof(input_buffer) + row * (width // 256) * 292)
                        single(expected, weight_buffer, input_row, channels, width)
                        self.assertEqual(
                            list(output[row * channels:(row + 1) * channels]),
                            list(expected))

    def test_mixed_prefill_phase_requires_map_declared_local_binding(self) -> None:
        ops, layout, config = _two_layer_fixture()
        plan = resolve_two_row_batch_contract(ops, layout, config)["full_layer_plan"]
        attention = next(i for i, op in enumerate(ops) if op["op"] == "attn_sliding")
        projection = next(i for i, op in enumerate(ops) if op["op"] == "q_proj")
        declared = copy.deepcopy(ops)
        declared[attention]["prefill_continuation"] = {"function": "prefill_attention"}
        alternate = copy.deepcopy(declared)
        alternate[attention]["function"] = "prefill_attention"
        self.assertTrue(resolve_mixed_prefill_local_stages(declared, alternate, plan))
        self.assertIn("ck_model_decode_batch2_mixed_prefill",
                      emit_full_layer_batch_api(plan, prefill_phase=True))
        self.assertIn("static int ck_model_decode_batch2_impl",
                      emit_full_layer_batch_api(plan, prefill_phase=True))
        self.assertFalse(resolve_mixed_prefill_local_stages(ops, alternate, plan))
        changed_argument = copy.deepcopy(alternate)
        changed_argument[attention]["args"][0]["expr"] = "wrong pointer"
        self.assertFalse(resolve_mixed_prefill_local_stages(declared, changed_argument, plan))
        changed_projection = copy.deepcopy(alternate)
        changed_projection[projection]["function"] = "different projection"
        self.assertFalse(resolve_mixed_prefill_local_stages(declared, changed_projection, plan))

    def test_q8_two_row_provider_preserves_compact_row_stride(self) -> None:
        for flags in ([], ["-mavx2", "-mfma"] if _avx2_available() else []):
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                stub = root / "stubs.c"
                stub.write_text(
                    "int ck_strict_parity_enabled(void) { return 0; }\n"
                    "const void *ck_strict_consume_next_gemm_a(void) { return 0; }\n"
                    "void quantize_row_q8_k(void) {}\n"
                )
                library = root / "libq8.so"
                subprocess.run(
                    ["cc", "-shared", "-fPIC", "-O2", *flags, "-I", str(ROOT / "include"),
                     str(ROOT / "src/kernels/gemm_kernels_q8_0_q8_0_contract.c"),
                     str(ROOT / "src/kernels/gemm_kernels_q8_0.c"), str(stub),
                     "-lm", "-o", str(library)],
                    check=True, capture_output=True, text=True,
                )
                native = ctypes.CDLL(str(library))
                single = native.gemv_q8_0_q8_0_contract
                single.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
                                   ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
                batch = native.gemm_nt_q8_0_q8_0_contract_m2
                batch.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
                                  ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                  ctypes.c_int, ctypes.c_int, ctypes.c_int]
                for width in (32, 64, 640):
                    channels = 8
                    packed = bytearray()
                    for channel in range(channels):
                        for block in range(width // 32):
                            values = [((channel * 13 + block * 7 + j * 11) % 251) - 125
                                      for j in range(32)]
                            packed += struct.pack("<e", 0.02 * (channel + block + 1))
                            packed += struct.pack("<32b", *values)
                    weights = ctypes.create_string_buffer(bytes(packed))
                    values = [((i * 17) % 53 - 26) / 13 for i in range(2 * width)]
                    inputs = (ctypes.c_float * len(values))(*values)
                    bias = (ctypes.c_float * channels)(*((i - 4) / 9 for i in range(channels)))
                    actual = (ctypes.c_float * (2 * channels))()
                    batch(inputs, weights, bias, actual, 2, channels, width)
                    for row in range(2):
                        isolated = (ctypes.c_float * channels)()
                        input_row = ctypes.cast(ctypes.byref(inputs, row * width * 4),
                                                ctypes.POINTER(ctypes.c_float))
                        single(isolated, weights, input_row, channels, width)
                        for channel in range(channels):
                            self.assertEqual(actual[row * channels + channel],
                                             ctypes.c_float(isolated[channel] + bias[channel]).value)

    def test_complete_layer_plan_follows_ir_and_rejects_unsafe_changes(self) -> None:
        fixture = json.loads((ROOT / "tests/fixtures/batch_decode_gemma3_first_layer.json").read_text())
        ops, layout, config = fixture["ops"], fixture["layout"], fixture["config"]
        contract = resolve_two_row_batch_contract(ops, layout, config)
        self.assertIsNotNone(contract)
        plan = contract["full_layer_plan"]
        self.assertEqual(
            [(stage["kind"], stage["start"], stage["stop"]) for stage in plan["stages"]],
            [("local", 0, 3), ("shared", 3, 6), ("local", 6, 10),
             ("shared", 10, 11), ("local", 11, 15), ("shared", 15, 16),
             ("local", 16, 18), ("shared", 18, 19), ("local", 19, 22)],
        )
        self.assertEqual([len(stage["projections"]) for stage in plan["stages"]
                          if stage["kind"] == "shared"], [3, 1, 1, 1])
        self.assertEqual(plan["extents"]["main_stream_q8"], 2336)
        emitted = emit_full_layer_batch_api(plan)
        self.assertIn("clock_gettime(CLOCK_MONOTONIC, &now)", emitted)
        self.assertIn("gemm_nt_q8_0_q8_0_contract_m2", emitted)
        self.assertIn("gemm_nt_q5_k_q8_k_m2", emitted)
        self.assertIn("gemm_nt_q6_k_q8_k_m2", emitted)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "full_layer_entry.c"
            defines = "\n".join(f"#define {name} 0" for name in
                                sorted(set(plan["buffers"].values()) |
                                       {p["weight"] for s in plan["stages"]
                                        for p in s.get("projections", [])}))
            stages = "\n".join(
                f"static void ck_batch_stage_{i}(CKModel *model, int token) "
                "{ (void)model; (void)token; }"
                for i, stage in enumerate(plan["stages"]) if stage["kind"] == "local"
            )
            kernels = "\n".join(
                f"static void {name}(const void *a, const void *b, const float *bias, "
                "float *c, int m, int n, int k) "
                "{ (void)a; (void)b; (void)bias; (void)c; (void)m; (void)n; (void)k; }"
                for name in sorted({p["function"] for stage in plan["stages"]
                                    for p in stage.get("projections", [])})
            )
            source.write_text(
                "#define _POSIX_C_SOURCE 200809L\n"
                "#include <stdint.h>\n#include <stddef.h>\n#include <stdlib.h>\n#include <string.h>\n"
                "#define CK_EXPORT\n#define VOCAB_SIZE 32\n#define MAX_SEQ_LEN 16\n"
                "#define KV_CACHE_SIZE 64\n" + defines + "\n"
                "typedef struct { uint64_t sequence_handle; int token, position; "
                "size_t row_offset, token_count; float *logits; } CKModelBatchDecodeRowV8;\n"
                "typedef struct { int occupied, pos; uint64_t handle; uint8_t *kv; } CKSequenceStateV8;\n"
                "typedef struct { uint8_t *bump; size_t bump_size; int pos; float logits[32]; } CKModel;\n"
                "static CKModel *g_model; static CKSequenceStateV8 g_sequence_states[3];\n"
                "static CKSequenceStateV8 *g_active_sequence; static int g_ck_skip_decode_logits;\n"
                "static int ck_model_cancel_requested(void) { return 0; }\n"
                "static CKSequenceStateV8 *ck_sequence_find(uint64_t h) "
                "{ (void)h; return &g_sequence_states[0]; }\n"
                "static int ck_model_sequence_state_activate(uint64_t h) "
                "{ (void)h; return 0; }\n" + stages + "\n" + kernels + "\n" + emitted
            )
            subprocess.run(["cc", "-std=c11", "-fsyntax-only", "-Werror",
                            "-Wno-unused-function", "-Wno-unused-variable", str(source)],
                           check=True, capture_output=True, text=True)
        for change in ("v_provider", "out_offset", "down_offset", "extra_live",
                       "cross_layer_live",
                       "persistent_state", "partial_projection", "fp32_override",
                       "down_quant_extent", "projection_alias"):
            bad = copy.deepcopy(fixture)
            bad_ops, bad_layout = bad["ops"], bad["layout"]
            if change == "v_provider":
                bad_ops[5]["call_abi"]["kernel_id"] = "gemv_q5_0"
            elif change == "out_offset":
                next(arg for arg in bad_ops[10]["args"] if arg["name"] == "x")["expr"] += " + 4"
            elif change == "down_offset":
                next(arg for arg in bad_ops[18]["args"] if arg["name"] == "x_q8")["expr"] += " + 4"
            elif change == "persistent_state":
                bad_layout["memory"]["activations"]["buffers"].append(
                    {"name": "new_state", "size": 64, "abs_offset": 1 << 32,
                     "define": "A_NEW_STATE", "lifetime": "sequence", "mutable": True})
            elif change == "partial_projection":
                next(arg for arg in bad_ops[15]["args"] if arg["name"] == "y")["expr"] += " + 4"
            elif change == "fp32_override":
                bad["config"]["out_proj_input_policy"] = "force_fp32"
            elif change == "down_quant_extent":
                next(arg for arg in bad_ops[17]["args"] if arg["name"] == "k")["expr"] = "1024"
            elif change == "projection_alias":
                output = next(arg for arg in bad_ops[5]["args"] if arg["name"] == "y")
                output["buffer_ref"] = "embedded_input"
                output["expr"] = "(float*)(model->bump + A_EMBEDDED_INPUT)"
            elif change == "cross_layer_live":
                bad_layout["memory"]["activations"]["buffers"].append(
                    {"name": "cross_layer_value", "size": 64, "abs_offset": 1 << 32,
                     "define": "A_CROSS_LAYER_VALUE", "lifetime": "call", "mutable": True})
                bad_ops[16]["args"].append(
                    {"name": "cross_layer_out", "source": "output:cross_layer_out",
                     "buffer_ref": "cross_layer_value",
                     "expr": "(float*)(model->bump + A_CROSS_LAYER_VALUE)"})
                bad_ops[21]["args"].append(
                    {"name": "cross_layer_in", "source": "activation:cross_layer_in",
                     "buffer_ref": "cross_layer_value",
                     "expr": "(const float*)(model->bump + A_CROSS_LAYER_VALUE)"})
            else:
                bad_layout["memory"]["activations"]["buffers"].append(
                    {"name": "extra", "size": 64, "abs_offset": 1 << 32,
                     "define": "A_EXTRA", "lifetime": "call", "mutable": True})
                bad_ops[9]["args"].append(
                    {"name": "extra_out", "source": "output:extra", "buffer_ref": "extra",
                     "expr": "(float*)(model->bump + A_EXTRA)"})
                bad_ops[11]["args"].append(
                    {"name": "extra_in", "source": "activation:extra", "buffer_ref": "extra",
                     "expr": "(const float*)(model->bump + A_EXTRA)"})
            with self.subTest(change=change):
                fallback = resolve_two_row_batch_contract(bad_ops, bad_layout, bad["config"])
                if change == "persistent_state":
                    self.assertIsNone(fallback)
                else:
                    self.assertIsNotNone(fallback)
                    self.assertNotIn("full_layer_plan", fallback)

    def test_full_layer_entry_executes_four_groups_with_isolated_rows(self) -> None:
        fixture = json.loads((ROOT / "tests/fixtures/batch_decode_gemma3_first_layer.json").read_text())
        plan = resolve_two_row_batch_contract(
            fixture["ops"], fixture["layout"], fixture["config"])["full_layer_plan"]
        names = sorted(set(plan["buffers"].values()) |
                       {projection["weight"] for stage in plan["stages"]
                        for projection in stage.get("projections", [])})
        defines = "\n".join(f"#define {name} {32768 * (index + 1)}" for index, name in enumerate(names))
        weights = sorted({projection["weight"] for stage in plan["stages"]
                          for projection in stage.get("projections", [])})
        initialize_weights = "\n".join(
            f"    g_model->bump[{name}] = {index + 1};" for index, name in enumerate(weights))
        kernels = "\n".join(
            f"static void {name}(const {'void' if dtype == 'q8_k' else 'float'} *input, "
            "const void *weight, const float *bias, float *output, int rows, int channels, int width) "
            "{ (void)bias; "
            "for (int row=0; row<rows; ++row) for (int channel=0; channel<channels; ++channel) "
            "output[row*channels+channel]="
            + ("((const float *)((const uint8_t *)input+row*(width/256*292)))[0]"
               if dtype == "q8_k" else "((const float *)input)[row*width]")
            + "+((const uint8_t *)weight)[0]; }"
            for name, dtype in sorted({(projection["function"], projection["dtype"])
                                       for stage in plan["stages"]
                                       for projection in stage.get("projections", [])}))
        prelude = r'''
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#define CK_EXPORT
#define VOCAB_SIZE 16
#define MAX_SEQ_LEN 16
#define KV_CACHE_SIZE 64
'''
        prelude += defines + r'''
typedef struct { uint64_t sequence_handle; int token, position; size_t row_offset, token_count; float *logits; } CKModelBatchDecodeRowV8;
typedef struct { int occupied, pos; uint64_t handle; uint8_t *kv; } CKSequenceStateV8;
typedef struct { uint8_t *bump; size_t bump_size; int pos; float logits[VOCAB_SIZE]; } CKModel;
static CKModel object;
static CKModel *g_model=&object;
static CKSequenceStateV8 g_sequence_states[3];
static CKSequenceStateV8 *g_active_sequence;
static int g_ck_skip_decode_logits;
static CKSequenceStateV8 *ck_sequence_find(uint64_t handle) {
    for (int i=0;i<3;++i) if (g_sequence_states[i].occupied && g_sequence_states[i].handle==handle) return &g_sequence_states[i];
    return NULL;
}
static int ck_model_sequence_state_activate(uint64_t handle) {
    CKSequenceStateV8 *next=ck_sequence_find(handle);
    if (!next) return -1;
    if (g_active_sequence) g_active_sequence->pos=g_model->pos;
    g_model->pos=next->pos;
    g_active_sequence=next;
    return 0;
}
static int ck_model_cancel_requested(void) { return 0; }
static float *value(CKModel *model, size_t offset) { return (float *)(model->bump+offset); }
static void ck_batch_stage_0(CKModel *model, int token) {
    value(model,A_EMBEDDED_INPUT)[0]=(float)token;
    value(model,A_RESIDUAL)[0]=(float)(10*token);
}
static void ck_batch_stage_2(CKModel *model, int token) {
    float q=value(model,A_Q_SCRATCH)[0], k=value(model,A_K_SCRATCH)[0];
    float v=value(model,A_V_SCRATCH)[0];
    value(model,A_ATTN_SCRATCH)[0]=q+k+v;
    value(model,A_RESIDUAL)[0]+=(float)token;
    g_active_sequence->kv[g_model->pos]=(uint8_t)q;
}
static void ck_batch_stage_4(CKModel *model, int token) {
    (void)token;
    value(model,A_EMBEDDED_INPUT)[0]+=value(model,A_RESIDUAL)[0];
}
static void ck_batch_stage_6(CKModel *model, int token) {
    (void)token;
    value(model,A_MAIN_STREAM_Q8)[0]=value(model,A_MLP_SCRATCH)[0]+value(model,A_RESIDUAL)[0];
}
static void ck_batch_stage_8(CKModel *model, int token) {
    (void)token;
    model->logits[0]=value(model,A_EMBEDDED_INPUT)[0]+value(model,A_RESIDUAL)[0];
    model->pos++;
}
static void project_one(CKModel *model, size_t input, size_t weight, size_t output) {
    value(model,output)[0]=value(model,input)[0]+model->bump[weight];
}
'''
        reference = r'''
static void isolated_decode(CKModel *model, int token) {
    ck_batch_stage_0(model,token);
    project_one(model,A_EMBEDDED_INPUT,W_LAYER_0_WQ,A_Q_SCRATCH);
    project_one(model,A_EMBEDDED_INPUT,W_LAYER_0_WK,A_K_SCRATCH);
    project_one(model,A_EMBEDDED_INPUT,W_LAYER_0_WV,A_V_SCRATCH);
    ck_batch_stage_2(model,token);
    project_one(model,A_ATTN_SCRATCH,W_LAYER_0_WO,A_EMBEDDED_INPUT);
    ck_batch_stage_4(model,token);
    project_one(model,A_EMBEDDED_INPUT,W_LAYER_0_W1,A_MLP_SCRATCH);
    ck_batch_stage_6(model,token);
    project_one(model,A_MAIN_STREAM_Q8,W_LAYER_0_W2,A_EMBEDDED_INPUT);
    ck_batch_stage_8(model,token);
}
int main(void) {
    g_model->bump_size=16*32768;
    g_model->bump=calloc(1,g_model->bump_size);
    if (!g_model->bump) return 1;
    uint8_t kv[2][64]={{0}};
    for (int i=0;i<2;++i) g_sequence_states[i]=(CKSequenceStateV8){.occupied=1,.handle=(uint64_t)(i+1),.kv=kv[i]};
'''
        reference += initialize_weights + r'''
    float expected[2]={0}, actual[2][VOCAB_SIZE]={{0}};
    int expected_pos[2]={0}; uint8_t expected_kv[2]={0};
    for (int i=0;i<2;++i) {
        if (ck_model_sequence_state_activate((uint64_t)(i+1))) return 2;
        isolated_decode(g_model,i ? 7 : 3);
        expected[i]=g_model->logits[0];
        expected_pos[i]=g_model->pos;
        expected_kv[i]=kv[i][0];
    }
    memset(g_model->bump,0,g_model->bump_size);
'''
        reference += initialize_weights + r'''
    memset(kv,0,sizeof(kv));
    for (int i=0;i<2;++i) g_sequence_states[i].pos=0;
    g_active_sequence=NULL; g_model->pos=0;
    CKModelBatchDecodeRowV8 rows[2]={
        {.sequence_handle=1,.token=3,.position=0,.row_offset=0,.token_count=1,.logits=actual[0]},
        {.sequence_handle=2,.token=7,.position=0,.row_offset=1,.token_count=1,.logits=actual[1]}
    };
    size_t bytes=0,alignment=0;
    if (ck_model_batch_decode_projection_groups()!=4 ||
        ck_model_batch_decode_workspace(&bytes,&alignment) || alignment!=64) return 3;
    void *workspace=aligned_alloc(alignment,(bytes+alignment-1)/alignment*alignment);
    if (!workspace || ck_model_decode_batch2(rows,2,workspace,bytes)) return 4;
    for (int i=0;i<2;++i) {
        if (ck_model_sequence_state_activate((uint64_t)(i+1))) return 5;
        if (actual[i][0]!=expected[i] || g_model->pos!=expected_pos[i] ||
            kv[i][0]!=expected_kv[i]) return 6+i;
    }
    free(workspace); free(g_model->bump);
    return 0;
}
'''
        emitted = emit_full_layer_batch_api(plan)
        source_text = prelude + kernels + emitted + reference
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "four_group.c"
            binary = Path(directory) / "four_group"

            def run(text: str) -> subprocess.CompletedProcess[str]:
                source.write_text(text)
                compiled = subprocess.run(["cc", "-std=c11", "-O0", "-Wall", "-Werror",
                                           "-Wno-unused-function", "-Wno-unused-variable",
                                           str(source), "-o", str(binary)],
                                          capture_output=True, text=True)
                self.assertEqual(compiled.returncode, 0, compiled.stderr)
                return subprocess.run([str(binary)], capture_output=True, text=True)

            self.assertEqual(run(source_text).returncode, 0)
            # This control removes one per-row restoration after shared Q/K/V.
            # The same two distinct inputs must now diverge from isolation.
            lost_q = re.sub(
                r"(?m)^        ck_batch_memcpy\(g_model->bump \+ A_Q_SCRATCH, base \+ [^\n]+\n",
                "", source_text, count=1,
            )
            self.assertNotEqual(lost_q, source_text)
            self.assertNotEqual(run(lost_q).returncode, 0)

    def test_two_layer_plan_executes_compact_fp32_rows(self) -> None:
        ops, layout, config = _two_layer_fixture()
        contract = resolve_two_row_batch_contract(ops, layout, config)
        self.assertIsNotNone(contract)
        plan = contract["full_layer_plan"]
        self.assertEqual(plan["shared_layers"], [0, 1])
        self.assertEqual(plan["projection_groups"], 8)
        self.assertEqual(plan["stages"][15]["projections"][0]["input_bytes"], 8192)
        self.assertEqual(plan["extents"]["mlp_scratch"], 16384)
        limited = resolve_two_row_batch_contract(ops, layout,
                                                 {**config, "batch_decode_max_shared_layers": 1})
        self.assertEqual(limited["full_layer_plan"]["shared_layers"], [0])
        bad = copy.deepcopy(ops)
        next(op for op in bad if op.get("layer") == 1 and op["op"] == "v_proj")["call_abi"]["kernel_id"] = "gemv_q5_0"
        fallback = resolve_two_row_batch_contract(bad, layout, config)
        self.assertEqual(fallback["full_layer_plan"]["shared_layers"], [0])
        # A value produced in layer 1 and consumed after that layer must not
        # disappear merely because the selected projection groups are valid.
        bad = copy.deepcopy(ops)
        bad_layout = copy.deepcopy(layout)
        bad_layout["memory"]["activations"]["buffers"].append(
            {"name": "later_live", "size": 64, "abs_offset": 1 << 32,
             "define": "A_LATER_LIVE", "lifetime": "call", "mutable": True})
        next(op for op in bad if op.get("layer") == 1 and op["op"] == "geglu")["args"].append(
            {"name": "later_out", "source": "output:later_out", "buffer_ref": "later_live",
             "expr": "(float*)(model->bump + A_LATER_LIVE)"})
        bad[-1]["args"].append(
            {"name": "later_in", "source": "activation:later_in", "buffer_ref": "later_live",
             "expr": "(const float*)(model->bump + A_LATER_LIVE)"})
        fallback = resolve_two_row_batch_contract(bad, bad_layout, config)
        self.assertNotIn("full_layer_plan", fallback)

        symbols = sorted(set(plan["buffers"].values()) |
                         {projection["weight"] for stage in plan["stages"]
                          for projection in stage.get("projections", [])})
        defines = "\n".join(f"#define {name} {32768 * (index + 1)}"
                            for index, name in enumerate(symbols))
        kernels = "\n".join(
            f"static void {name}(const {'void' if dtype == 'q8_k' else 'float'} *input, "
            "const void *weight, const float *bias, float *output, int rows, int channels, int width) "
            "{ (void)bias; for (int row=0; row<rows; ++row) for (int channel=0; channel<channels; ++channel) "
            "output[row*channels+channel]="
            + ("((const float *)((const uint8_t *)input+row*(width/256*292)))[0]"
               if dtype == "q8_k" else "((const float *)input)[row*width]")
            + "+((const uint8_t *)weight)[0]; }"
            for name, dtype in sorted({(projection["function"], projection["dtype"])
                                       for stage in plan["stages"]
                                       for projection in stage.get("projections", [])})
        )
        bodies = {
            0: "value(model,A_EMBEDDED_INPUT)[0]=(float)token; value(model,A_RESIDUAL)[0]=(float)token;",
            2: "g_active_sequence->kv[model->pos]=(uint8_t)token;",
            8: "value(model,A_RESIDUAL)[0]+=(float)token;",
            10: "g_active_sequence->kv[16+model->pos]=(uint8_t)token;",
            14: "value(model,A_MLP_SCRATCH)[0]=(float)(10*token)+value(model,A_RESIDUAL)[0];",
            16: "model->logits[0]=value(model,A_EMBEDDED_INPUT)[0]; model->pos++;",
        }
        stages = "\n".join(
            f"static void ck_batch_stage_{index}(CKModel *model, int token) "
            "{ " + bodies.get(index, "(void)model; (void)token;") + " }"
            for index, stage in enumerate(plan["stages"]) if stage["kind"] == "local"
        )
        prelude = r'''
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#define CK_EXPORT
#define VOCAB_SIZE 16
#define MAX_SEQ_LEN 16
#define KV_CACHE_SIZE 64
'''
        prelude += defines + f"\n#define BUMP_BYTES {(len(symbols) + 1) * 32768}\n" + r'''
typedef struct { uint64_t sequence_handle; int token, position; size_t row_offset, token_count; float *logits; } CKModelBatchDecodeRowV8;
typedef struct { int occupied, pos; uint64_t handle; uint8_t *kv; } CKSequenceStateV8;
typedef struct { uint8_t *bump; size_t bump_size; int pos; float logits[VOCAB_SIZE]; } CKModel;
static CKModel object;
static CKModel *g_model=&object;
static CKSequenceStateV8 g_sequence_states[3];
static CKSequenceStateV8 *g_active_sequence;
static int g_ck_skip_decode_logits;
static CKSequenceStateV8 *ck_sequence_find(uint64_t handle) {
    for (int i=0;i<3;++i) if (g_sequence_states[i].occupied && g_sequence_states[i].handle==handle) return &g_sequence_states[i];
    return NULL;
}
static int ck_model_sequence_state_activate(uint64_t handle) {
    CKSequenceStateV8 *next=ck_sequence_find(handle);
    if (!next) return -1;
    if (g_active_sequence) g_active_sequence->pos=g_model->pos;
    g_model->pos=next->pos; g_active_sequence=next;
    return 0;
}
static int ck_model_cancel_requested(void) { return 0; }
static float *value(CKModel *model, size_t offset) { return (float *)(model->bump+offset); }
'''
        run = r'''
int main(void) {
    g_model->bump_size=BUMP_BYTES;
    g_model->bump=calloc(1,BUMP_BYTES);
    if (!g_model->bump) return 1;
    uint8_t kv[2][64]={{0}};
    for (int i=0;i<2;++i) g_sequence_states[i]=(CKSequenceStateV8){.occupied=1,.handle=(uint64_t)(i+1),.kv=kv[i]};
    g_model->bump[W_LAYER_1_W2]=5;
    float logits[2][VOCAB_SIZE]={{0}};
    CKModelBatchDecodeRowV8 rows[2]={
        {.sequence_handle=1,.token=3,.position=0,.row_offset=0,.token_count=1,.logits=logits[0]},
        {.sequence_handle=2,.token=7,.position=0,.row_offset=1,.token_count=1,.logits=logits[1]}
    };
    size_t bytes=0,alignment=0;
    if (ck_model_batch_decode_projection_groups()!=8 || ck_model_batch_decode_shared_layers()!=2 ||
        ck_model_batch_decode_packing_copy_bytes()<16384 ||
        ck_model_batch_decode_workspace(&bytes,&alignment) || alignment!=64) return 2;
    void *workspace=aligned_alloc(alignment,(bytes+alignment-1)/alignment*alignment);
    if (!workspace || ck_model_decode_batch2(rows,2,workspace,bytes)) return 3;
    if (logits[0][0]!=41.0f || logits[1][0]!=89.0f) return 4;
    for (int i=0;i<2;++i) {
        if (ck_model_sequence_state_activate((uint64_t)(i+1)) || g_model->pos!=1 ||
            kv[i][0]!=(uint8_t)(i?7:3) || kv[i][16]!=(uint8_t)(i?7:3)) return 5;
    }
    free(workspace); free(g_model->bump);
    return 0;
}
'''
        source_text = prelude + stages + kernels + emit_full_layer_batch_api(plan) + run
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "two_layer.c"
            binary = Path(directory) / "two_layer"

            def execute(text: str) -> subprocess.CompletedProcess[str]:
                source.write_text(text)
                compiled = subprocess.run(["cc", "-std=c11", "-O0", "-Wall", "-Werror",
                                           "-Wno-unused-function", "-Wno-unused-variable",
                                           str(source), "-o", str(binary)],
                                          capture_output=True, text=True)
                self.assertEqual(compiled.returncode, 0, compiled.stderr)
                return subprocess.run([str(binary)], capture_output=True, text=True)

            self.assertEqual(execute(source_text).returncode, 0)
            # Reintroducing the 16 KiB arena stride as the kernel's compact
            # 8 KiB row stride corrupts only the second sequence's input.
            broken = source_text.replace("i * 16384u, 8192u);", "i * 8192u, 8192u);", 1)
            self.assertNotEqual(broken, source_text)
            self.assertNotEqual(execute(broken).returncode, 0)
            lost_cross_layer = re.sub(
                r"(?m)^        ck_batch_memcpy\(g_model->bump \+ A_RESIDUAL, base \+ [^\n]+\n(?=        ck_batch_stage_8)",
                "", source_text, count=1,
            )
            self.assertNotEqual(lost_cross_layer, source_text)
            self.assertNotEqual(execute(lost_cross_layer).returncode, 0)
            wrong_kv_owner = source_text.replace(
                "g_active_sequence->kv[16+model->pos]=(uint8_t)token;",
                "g_sequence_states[0].kv[16+model->pos]=(uint8_t)token;", 1,
            )
            self.assertNotEqual(wrong_kv_owner, source_text)
            self.assertNotEqual(execute(wrong_kv_owner).returncode, 0)

            # Compile and execute the new wrapper with distinct local-stage
            # arithmetic. The generated row selector must change only the
            # chosen staged row; the other row keeps ordinary decode.
            prefill_stages = "\n".join(
                f"static void ck_batch_prefill_stage_{index}(CKModel *model, int token) "
                "{ " + ("g_active_sequence->kv[model->pos]=(uint8_t)(token+1);"
                        if index == 2 else bodies.get(index, "(void)model; (void)token;"))
                + " }"
                for index, stage in enumerate(plan["stages"])
                if stage["kind"] == "local"
            )
            phase_run = run.replace(
                "ck_model_decode_batch2(rows,2,workspace,bytes)",
                "ck_model_decode_batch2_mixed_prefill(rows,2,workspace,bytes,0)",
            ).replace("kv[i][0]!=(uint8_t)(i?7:3)",
                      "kv[i][0]!=(uint8_t)(i?7:4)")
            self.assertNotEqual(phase_run, run)
            phase_source = (prelude + stages + prefill_stages + kernels +
                            emit_full_layer_batch_api(plan, prefill_phase=True) +
                            phase_run)
            self.assertEqual(execute(phase_source).returncode, 0)
            wrong_row = phase_source.replace(
                "ck_model_decode_batch2_mixed_prefill(rows,2,workspace,bytes,0)",
                "ck_model_decode_batch2_mixed_prefill(rows,2,workspace,bytes,1)")
            self.assertNotEqual(execute(wrong_row).returncode, 0)

    def test_in_place_liveness_ignores_argument_order(self) -> None:
        produced = {"args": [{"source": "output:x", "buffer_ref": "extra"}]}
        read = {"source": "activation:x", "buffer_ref": "extra"}
        write = {"source": "output:x", "buffer_ref": "extra"}
        for args in ([read, write], [write, read]):
            with self.subTest(args=args):
                self.assertEqual(_crossing_at([produced, {"args": args}], 1), {"extra"})
                self.assertEqual(
                    _crossing_at([produced, {"args": args}], 1,
                                 frozenset({(1, "extra")})), {"extra"})

    def test_milestone_requires_two_projection_groups(self) -> None:
        _require_projection_groups(1, None)
        _require_projection_groups(2, 2)
        _require_projection_groups(4, 4)
        _require_projection_groups(8, 8)
        _require_projection_groups(72, 72)
        with self.assertRaisesRegex(AssertionError, "expected 2 shared projection groups"):
            _require_projection_groups(1, 2)
        with self.assertRaisesRegex(AssertionError, "expected 4 shared projection groups"):
            _require_projection_groups(2, 4)
        with self.assertRaisesRegex(AssertionError, "invalid generated batch projection-group count"):
            _require_projection_groups(7, None)

    def test_native_timing_excludes_slow_validation_on_both_paths(self) -> None:
        ticks = [0]

        def native_call() -> int:
            ticks[0] += 5_000_000
            return 0

        def slow_validation(status: int) -> int:
            ticks[0] += 900_000_000
            return status

        with mock.patch("certify_batch_decode_v8.time.perf_counter_ns",
                        side_effect=lambda: ticks[0]):
            for path in ("isolated", "batched"):
                with self.subTest(path=path):
                    result, native_ms = _measure_native_call(native_call, slow_validation)
                    self.assertEqual(result, 0)
                    self.assertEqual(native_ms, 5.0)

    def test_hybrid_arena_capacity_does_not_change_decode_row_liveness(self) -> None:
        ops, layout, config = _layer_fixture()
        offset = 128
        for row in layout["memory"]["activations"]["buffers"]:
            if row["lifetime"] != "call":
                continue
            if row["name"] != "logits":
                row["size"] *= 4
            row["abs_offset"] = offset
            offset += row["size"] + 64
        layout["memory"]["arena"]["total_size"] = offset + 64
        contract = resolve_two_row_batch_contract(ops, layout, config)
        self.assertIsNotNone(contract)
        self.assertIn("layer_extension", contract)

    def test_layer_extension_is_liveness_and_provider_bound(self) -> None:
        ops, layout, config = _layer_fixture()
        contract = resolve_two_row_batch_contract(ops, layout, config)
        self.assertEqual(contract["layer_extension"]["cut"], 6)
        self.assertEqual(contract["layer_extension"]["output_dim"], 64)
        for change in ("offset_input", "offset_output", "wrong_provider", "bad_geglu",
                       "extra_mid_live", "partial_write_live", "aliased_output", "other_layer"):
            bad_ops, bad_layout = copy.deepcopy(ops), copy.deepcopy(layout)
            if change == "offset_input":
                bad_ops[6]["args"][2]["expr"] += " + 4"
            elif change == "offset_output":
                bad_ops[6]["args"][0]["expr"] += " + 4"
            elif change == "wrong_provider":
                bad_ops[6]["call_abi"]["kernel_id"] = "gemv_q5_0"
            elif change == "bad_geglu":
                bad_ops[7]["args"][-1]["expr"] = "16"
            elif change == "other_layer":
                bad_ops[5]["layer"] = 1
            elif change == "aliased_output":
                bad_layout["memory"]["activations"]["buffers"].append(
                    {"name": "alias", "size": 16, "abs_offset": 812,
                     "define": "A_ALIAS", "lifetime": "call", "mutable": True})
            else:
                bad_layout["memory"]["activations"]["buffers"].append(
                    {"name": "extra", "size": 16, "abs_offset": 1100,
                     "define": "A_EXTRA", "lifetime": "call", "mutable": True})
                bad_ops[5]["args"].append(
                    {"name": "extra", "expr": "(float*)(model->bump + A_EXTRA)",
                     "source": "output:extra", "buffer_ref": "extra"})
                bad_ops[8]["args"].append(
                    {"name": "extra", "expr": "(const float*)(model->bump + A_EXTRA)",
                     "source": "activation:extra", "buffer_ref": "extra"})
                if change == "partial_write_live":
                    bad_ops.insert(6, {"op": "partial_write", "layer": 0, "args": [
                        {"name": "out", "expr": "(float*)(model->bump + A_EXTRA)",
                         "source": "output:out", "buffer_ref": "extra"}]})
            with self.subTest(change=change):
                fallback = resolve_two_row_batch_contract(bad_ops, bad_layout, config)
                if change == "offset_input":
                    self.assertIsNone(fallback)
                else:
                    self.assertIsNotNone(fallback)
                    self.assertNotIn("layer_extension", fallback)

    def test_real_emitter_split_matches_isolated_execution(self) -> None:
        prelude = r'''
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#define CK_EXPORT
#define VOCAB_SIZE 64
#define MAX_SEQ_LEN 16
#define KV_CACHE_SIZE 64
#define EMBED_DIM 32
#define A_TOKEN_IDS 0
#define A_EMBEDDED_INPUT 128
#define A_RESIDUAL 256
#define A_Q_SCRATCH 384
#define A_K_SCRATCH 448
#define A_MLP_SCRATCH 800
#define W_LAYER_0_WQ 600
#define W_LAYER_0_WK 608
#define W_LAYER_0_W1 616
typedef struct { uint64_t sequence_handle; int token, position; size_t row_offset, token_count; float *logits; } CKModelBatchDecodeRowV8;
typedef struct { int occupied, pos, rope_pos; uint64_t handle; unsigned char *kv; } CKSequenceStateV8;
typedef struct { unsigned char bump[1280]; unsigned char *activations; size_t bump_size; int pos, rope_pos, bridge_has_explicit_positions; float logits[64]; } CKModel;
static CKModel object = {.bump_size = 1280};
static CKModel *g_model = &object;
static CKSequenceStateV8 g_sequence_states[3];
static CKSequenceStateV8 *g_active_sequence;
static int g_ck_skip_decode_logits;
static CKSequenceStateV8 *ck_sequence_find(uint64_t handle) {
    for (int i=0;i<3;i++) if (g_sequence_states[i].occupied && g_sequence_states[i].handle == handle) return &g_sequence_states[i];
    return NULL;
}
static int ck_model_sequence_state_activate(uint64_t handle) {
    CKSequenceStateV8 *next = ck_sequence_find(handle);
    if (!next) return -1;
    if (g_active_sequence && g_active_sequence != next) {
        g_active_sequence->pos = g_model->pos;
        g_active_sequence->rope_pos = g_model->rope_pos;
    }
    if (g_active_sequence != next) {
        g_model->pos = next->pos;
        g_model->rope_pos = next->rope_pos;
        g_active_sequence = next;
    }
    return 0;
}
static int ck_model_cancel_requested(void) { return 0; }
static void ck_debug_export_hidden(CKModel *m, int l, const char *n, const float *v, size_t c) { (void)m;(void)l;(void)n;(void)v;(void)c; }
static void ck_debug_import_checkpoint(CKModel *m, int l, const char *n, float *v, size_t c) { (void)m;(void)l;(void)n;(void)v;(void)c; }
static void fixture_dense_embedding_lookup(const int32_t *tokens, int count, float *out) {
    (void)count; for (int i=0;i<32;i++) out[i]=(float)(tokens[0]+i);
}
static void fixture_residual_save(void *dst, const void *src, size_t bytes) { memcpy(dst,src,bytes); }
static void fixture_attn_norm(const float *src, float *dst) { (void)src;(void)dst; }
static float weight_tag(const void *weight) {
    return weight == (void *)(g_model->bump + W_LAYER_0_WQ) ? 1.0f :
           weight == (void *)(g_model->bump + W_LAYER_0_WK) ? 2.0f : 3.0f;
}
static void gemv_q5_1_q8_1(const float *input, float *output, const void *weight, int channels, int width) {
    (void)width; for (int c=0;c<channels;c++) output[c]=input[0]+weight_tag(weight);
}
static void fixture_mlp_gate_up(float *output, const void *weight, const float *input, int channels, int width) {
    gemv_q5_1_q8_1(input,output,weight,channels,width);
}
static void gemm_nt_q5_1_q8_1_m2(const float *input, const void *weight, const float *bias,
                                  float *output, int rows, int channels, int width) {
    (void)bias; for (int row=0;row<rows;row++) for (int c=0;c<channels;c++)
        output[row*channels+c]=input[row*width]+weight_tag(weight);
}
static void fixture_attention_terminal(const float *input, const float *residual, const float *q, const float *k) {
    (void)input;(void)residual;
    g_model->logits[0]=q[0];g_model->logits[1]=k[0];g_model->logits[2]=q[0]+k[0];
    g_active_sequence->kv[g_model->pos]=(unsigned char)q[0];
}
static void fixture_attention(const float *input, const float *residual, const float *q,
                              const float *k, float *output) {
    output[0]=input[0]+q[0]+k[0];
    ((float *)residual)[0]+=q[0];
    g_active_sequence->kv[g_model->pos]=(unsigned char)q[0];
}
static void fixture_geglu(const float *input, float *output, int tokens, int dim) {
    (void)tokens;(void)dim;output[0]=input[0]+1.0f;
}
static void fixture_finish(const float *residual, const float *input, float *output) {
    (void)output;g_model->logits[0]=input[0];g_model->logits[1]=residual[0];
    g_model->logits[2]=input[0]+residual[0];
}
'''
        postlude = r'''
int main(void) {
    unsigned char kv[2][64]={{0}};
    float isolated[2][64]={{0}}, batched[2][64]={{0}};
    int positions[2], ropes[2];
    unsigned char expected_kv[2][64];
    for (int i=0;i<2;i++) g_sequence_states[i]=(CKSequenceStateV8){.occupied=1,.handle=(uint64_t)(i+1),.kv=kv[i]};
    ck_model_sequence_state_activate(1);
    for (int i=0;i<2;i++) {
        ck_model_sequence_state_activate(i+1);
        ck_decode(g_model,i?7:3);
        memcpy(isolated[i],g_model->logits,sizeof(isolated[i]));
        positions[i]=g_model->pos;ropes[i]=g_model->rope_pos;
        memcpy(expected_kv[i],kv[i],64);
    }
    memset(g_model->bump,0,sizeof(g_model->bump));
    memset(g_model->logits,0,sizeof(g_model->logits));
    memset(kv,0,sizeof(kv));
    for (int i=0;i<2;i++) {g_sequence_states[i].pos=0;g_sequence_states[i].rope_pos=0;}
    g_active_sequence=NULL;g_model->pos=0;g_model->rope_pos=0;
    ck_model_sequence_state_activate(1);
    CKModelBatchDecodeRowV8 rows[2]={
        {.sequence_handle=1,.token=3,.position=0,.row_offset=0,.token_count=1,.logits=batched[0]},
        {.sequence_handle=2,.token=7,.position=0,.row_offset=1,.token_count=1,.logits=batched[1]}
    };
    size_t bytes=0, alignment=0;
    if (ck_model_batch_decode_workspace(&bytes,&alignment) || alignment!=64) return 1;
    void *workspace=aligned_alloc(alignment,(bytes+alignment-1)/alignment*alignment);
    if (!workspace || ck_model_decode_batch2(rows,2,workspace,bytes)) return 2;
    for (int i=0;i<2;i++) {
        ck_model_sequence_state_activate(i+1);
        if (memcmp(isolated[i],batched[i],sizeof(isolated[i])) ||
            memcmp(expected_kv[i],kv[i],64) ||
            g_model->pos!=positions[i] || g_model->rope_pos!=ropes[i]) return 3+i;
    }
    free(workspace);
    return 0;
}
'''
        for extended, fixture in ((False, _fixture()), (True, _layer_fixture())):
            with self.subTest(extended=extended), tempfile.TemporaryDirectory() as directory:
                ops, layout, config = fixture
                contract = resolve_two_row_batch_contract(ops, layout, config)
                self.assertIsNotNone(contract)
                self.assertEqual("layer_extension" in contract, extended)
                ops = copy.deepcopy(ops)
                for op in ops:
                    op["function"] = "fixture_" + op["op"]
                for op in ops[3:5]:
                    op["function"] = "gemv_q5_1_q8_1"
                if not extended:
                    ops[5]["function"] = "fixture_attention_terminal"
                emission = [emit_decode_function(ops, 0, "model->bump", config=config)]
                emission.append(emit_decode_function(
                    ops[:contract["prefix_len"]], 0, "model->bump", config=config,
                    function_name="ck_batch_decode_prefix", advance_position=False))
                extension = contract.get("layer_extension")
                if extension:
                    emission.append(emit_decode_function(
                        ops[contract["suffix_start"]:extension["cut"]], 0,
                        "model->bump", config=config,
                        function_name="ck_batch_decode_before_gateup",
                        store_token=False, advance_position=False))
                    emission.append(emit_decode_function(
                        ops[extension["post_cut"]:], 0, "model->bump", config=config,
                        function_name="ck_batch_decode_after_gateup", store_token=False))
                else:
                    emission.append(emit_decode_function(
                        ops[contract["suffix_start"]:], 0, "model->bump", config=config,
                        function_name="ck_batch_decode_suffix", store_token=False))
                source = Path(directory) / "emitted.c"
                binary = Path(directory) / "emitted"
                source.write_text(prelude + "\n".join(emission) +
                                  emit_two_row_batch_api(contract) + postlude)
                subprocess.run(["cc", "-std=c11", "-O0", "-Wall", "-Werror",
                                "-Wno-unused-function", "-Wno-unused-variable",
                                str(source), "-o", str(binary)],
                               check=True, capture_output=True, text=True)
                subprocess.run([str(binary)], check=True, capture_output=True, text=True)

    def test_compiled_generated_entry_and_nonparticipating_arena(self) -> None:
        prelude = r'''
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#define CK_EXPORT
#define VOCAB_SIZE 64
#define MAX_SEQ_LEN 16
#define KV_CACHE_SIZE 64
#define A_EMBEDDED_INPUT 128
#define A_RESIDUAL 256
#define A_Q_SCRATCH 384
#define A_K_SCRATCH 448
#define A_MLP_SCRATCH 800
#define W_LAYER_0_WQ 600
#define W_LAYER_0_WK 608
#define W_LAYER_0_W1 616
typedef struct { uint64_t sequence_handle; int token, position; size_t row_offset, token_count; float *logits; } CKModelBatchDecodeRowV8;
typedef struct { int occupied, pos; uint64_t handle; unsigned char *kv; } CKSequenceStateV8;
typedef struct { unsigned char bump[1280]; size_t bump_size; int pos; float logits[64]; } Model;
static Model object = {.bump_size = 1280};
static Model *g_model = &object;
static CKSequenceStateV8 g_sequence_states[3];
static CKSequenceStateV8 *g_active_sequence;
static int g_ck_skip_decode_logits;
static CKSequenceStateV8 *ck_sequence_find(uint64_t handle) {
    for (int i=0;i<3;i++) if (g_sequence_states[i].occupied && g_sequence_states[i].handle == handle) return &g_sequence_states[i];
    return NULL;
}
static int ck_model_sequence_state_activate(uint64_t handle) {
    g_active_sequence = ck_sequence_find(handle);
    if (!g_active_sequence) return -1;
    g_model->pos = g_active_sequence->pos;
    return 0;
}
static int ck_model_cancel_requested(void) { return 0; }
static void ck_batch_decode_prefix(Model *model, int token) {
    float *input = (float *)(model->bump + A_EMBEDDED_INPUT);
    for (int i=0;i<32;i++) input[i] = (float)(token+i);
    memcpy(model->bump + A_RESIDUAL, input, 128);
}
static void gemm_nt_q5_1_q8_1_m2(const float *input, const void *weight, const float *bias,
                                  float *output, int rows, int channels, int width) {
    (void)bias;
    float tag = weight == (void *)(g_model->bump + W_LAYER_0_WQ) ? 1.0f :
                weight == (void *)(g_model->bump + W_LAYER_0_WK) ? 2.0f : 3.0f;
    for (int row=0;row<rows;row++) for (int c=0;c<channels;c++) output[row*channels+c] = input[row*width] + tag;
}
static void ck_batch_decode_suffix(Model *model, int token) {
    float *q = (float *)(model->bump + A_Q_SCRATCH);
    float *k = (float *)(model->bump + A_K_SCRATCH);
    model->logits[0] = q[0]; model->logits[1] = k[0]; model->logits[2] = (float)token;
    g_active_sequence->kv[0] = (unsigned char)token;
    g_active_sequence->pos++; model->pos++;
}
static void ck_batch_decode_before_gateup(Model *model, int token) {
    float *input = (float *)(model->bump + A_EMBEDDED_INPUT);
    float *residual = (float *)(model->bump + A_RESIDUAL);
    float *q = (float *)(model->bump + A_Q_SCRATCH);
    float *k = (float *)(model->bump + A_K_SCRATCH);
    input[0] += q[0] + k[0];
    residual[0] += (float)token;
    g_active_sequence->kv[0] = (unsigned char)token;
}
static void ck_batch_decode_after_gateup(Model *model, int token) {
    float *residual = (float *)(model->bump + A_RESIDUAL);
    float *gateup = (float *)(model->bump + A_MLP_SCRATCH);
    model->logits[0] = gateup[0]; model->logits[1] = residual[0];
    model->logits[2] = (float)token;
    g_active_sequence->pos++; model->pos++;
}
'''
        postlude = r'''
int main(void) {
    unsigned char kv[3][64] = {{0}};
    for (int i=0;i<3;i++) g_sequence_states[i] = (CKSequenceStateV8){.occupied=1,.handle=(uint64_t)(i+1),.kv=kv[i]};
    ck_model_sequence_state_activate(1);
    float logits[2][64] = {{0}};
    CKModelBatchDecodeRowV8 rows[2] = {
        {.sequence_handle=1,.token=3,.row_offset=0,.token_count=1,.logits=logits[0]},
        {.sequence_handle=2,.token=7,.row_offset=1,.token_count=1,.logits=logits[1]}
    };
    size_t bytes=0, alignment=0;
    if (ck_model_batch_decode_workspace(&bytes,&alignment) || alignment != 64) return 1;
    void *workspace = aligned_alloc(alignment, (bytes+alignment-1)/alignment*alignment);
    void *third_arena = aligned_alloc(alignment, (bytes+alignment-1)/alignment*alignment);
    if (!workspace || !third_arena) return 2;
    memset(third_arena, 0x5a, bytes);
    g_sequence_states[2].kv = third_arena;
    if (ck_model_decode_batch2(rows,2,third_arena,bytes) != -2) return 3;
    if (((unsigned char *)third_arena)[0] != 0x5a) return 4;
    if (ck_model_decode_batch2(rows,2,workspace,bytes)) return 5;
    if (EXTENDED) {
        if (logits[0][0] != 15 || logits[0][1] != 6 ||
            logits[1][0] != 27 || logits[1][1] != 14) return 6;
    } else {
        if (logits[0][0] != 4 || logits[0][1] != 5 ||
            logits[1][0] != 8 || logits[1][1] != 9) return 6;
    }
    if (kv[0][0] != 3 || kv[1][0] != 7 || ((unsigned char *)third_arena)[0] != 0x5a) return 7;
    free(third_arena);
    free(workspace);
    return 0;
}
'''
        for extended, fixture in ((False, _fixture()), (True, _layer_fixture())):
            with self.subTest(extended=extended), tempfile.TemporaryDirectory() as directory:
                contract = resolve_two_row_batch_contract(*fixture)
                self.assertIsNotNone(contract)
                self.assertEqual("layer_extension" in contract, extended)
                source = Path(directory) / "entry.c"
                binary = Path(directory) / "entry"
                source.write_text(f"#define EXTENDED {int(extended)}\n" + prelude +
                                  emit_two_row_batch_api(contract) + postlude)
                subprocess.run(["cc", "-std=c11", "-O0", "-Wall", "-Werror",
                                "-Wno-unused-function", str(source), "-o", str(binary)],
                               check=True, capture_output=True, text=True)
                subprocess.run([str(binary)], check=True, capture_output=True, text=True)

    def test_loaded_symbol_rejects_replaced_library(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "library.so"
            other = Path(directory) / "replacement.so"
            for target, value in ((path, 11), (other, 22)):
                source = target.with_suffix(".c")
                source.write_text(f"int cke_test_symbol(void) {{ return {value}; }}\n")
                subprocess.run(["cc", "-shared", "-fPIC", str(source), "-o", str(target)],
                               check=True, capture_output=True, text=True)
            loaded = ctypes.CDLL(str(path))
            self.assertEqual(verified_loaded_symbol_backing(loaded, "cke_test_symbol", path)["sha256"],
                             __import__("hashlib").sha256(path.read_bytes()).hexdigest())
            other.replace(path)
            with self.assertRaisesRegex(ValueError, "changed|replaced"):
                verified_loaded_symbol_backing(loaded, "cke_test_symbol", path)

    def test_admits_only_declared_kv_and_map_pair(self) -> None:
        ops, layout, config = _fixture()
        resolved = resolve_two_row_batch_contract(ops, layout, config)
        self.assertEqual(resolved["gemm_function"], "gemm_nt_q5_1_q8_1_m2")
        self.assertEqual(resolved["input_dim"], 32)
        for change in ("extra_state", "changed_weight", "changed_input", "reordered", "oversized_k",
                       "offset_input", "offset_output", "offset_suffix", "wrong_provider", "new_live_value"):
            bad_ops, bad_layout = copy.deepcopy(ops), copy.deepcopy(layout)
            if change == "extra_state":
                bad_layout["memory"]["activations"]["buffers"].append(
                    {"name": "unknown_state", "size": 16, "abs_offset": 704,
                     "lifetime": "sequence", "mutable": True})
            elif change == "changed_weight":
                bad_ops[3]["args"][2]["expr"] = "(const void*)0"
            elif change == "changed_input":
                bad_ops[4]["args"][0]["buffer_ref"] = "residual"
            elif change == "oversized_k":
                bad_ops[3]["args"][-1]["expr"] = "8224"
                bad_ops[4]["args"][-1]["expr"] = "8224"
                bad_layout["memory"]["activations"]["buffers"][1]["size"] = 8224 * 4
            elif change == "offset_input":
                bad_ops[3]["args"][0]["expr"] = "(const float*)(model->bump + A_EMBEDDED_INPUT + 4)"
            elif change == "offset_output":
                bad_ops[4]["args"][1]["expr"] = "(float*)(model->bump + A_K_SCRATCH + 4)"
            elif change == "offset_suffix":
                bad_ops[5]["args"][0]["expr"] = "(const float*)(model->bump + A_EMBEDDED_INPUT + 4)"
            elif change == "wrong_provider":
                bad_ops[4]["call_abi"]["kernel_id"] = "gemv_q5_0"
            elif change == "new_live_value":
                bad_layout["memory"]["activations"]["buffers"].append(
                    {"name": "extra", "size": 16, "abs_offset": 704, "define": "A_EXTRA",
                     "lifetime": "call", "mutable": True})
                bad_ops[2]["args"].append({"name": "extra", "expr": "(float*)(model->bump + A_EXTRA)",
                                              "source": "output:extra", "buffer_ref": "extra"})
                bad_ops[5]["args"].append({"name": "extra", "expr": "(const float*)(model->bump + A_EXTRA)",
                                              "source": "activation:extra", "buffer_ref": "extra"})
            else:
                bad_ops[3], bad_ops[4] = bad_ops[4], bad_ops[3]
            with self.subTest(change=change):
                self.assertIsNone(resolve_two_row_batch_contract(bad_ops, bad_layout, config))

    def test_m2_kernel_portable(self) -> None:
        self._check_m2_kernel([[]])

    def test_m2_kernel_avx2(self) -> None:
        if not _avx2_available():
            self.skipTest("AVX2 is unavailable on this runner")
        self._check_m2_kernel([["-mavx2"]])

    def _check_m2_kernel(self, modes: list[list[str]]) -> None:
        source = ROOT / "src/kernels/gemm_kernels_q5_1_q8_1.c"
        for flags in modes:
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as directory:
                library = Path(directory) / "libq51.so"
                subprocess.run(["cc", "-shared", "-fPIC", "-O2", *flags,
                                "-I", str(ROOT / "include"), str(source),
                                "-lm", "-o", str(library)], check=True,
                               capture_output=True, text=True)
                native = ctypes.CDLL(str(library))
                gemv = native.gemv_q5_1_q8_1
                gemv.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
                                 ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
                batch = native.gemm_nt_q5_1_q8_1_m2
                batch.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
                                  ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                  ctypes.c_int, ctypes.c_int, ctypes.c_int]
                for width, channels, with_bias in ((32, 3, False), (64, 7, True), (96, 5, True)):
                    blocks = width // 32
                    packed = bytearray()
                    specs = []
                    for channel in range(channels):
                        row_specs = []
                        for block_index in range(blocks):
                            scale = (0.125, 0.3125, 0.1875)[(channel + block_index) % 3]
                            minimum = (-0.75, 0.375, -0.25)[(channel * 2 + block_index) % 3]
                            quants = [((channel * 11 + block_index * 7 + i * 13) % 32) for i in range(32)]
                            high = sum(((q >> 4) & 1) << i for i, q in enumerate(quants))
                            nibbles = bytes((quants[i] & 15) | ((quants[i + 16] & 15) << 4)
                                            for i in range(16))
                            packed.extend(struct.pack('<eeI', scale, minimum, high) + nibbles)
                            row_specs.append((scale, minimum, quants))
                        specs.append(row_specs)
                    weights = ctypes.create_string_buffer(bytes(packed))
                    values = [((i * 17) % 47 - 23) / 9 for i in range(2 * width)]
                    inputs = (ctypes.c_float * len(values))(*values)
                    bias_values = [(channel - 2) / 7 for channel in range(channels)]
                    bias = (ctypes.c_float * channels)(*bias_values) if with_bias else None
                    actual = (ctypes.c_float * (2 * channels))()
                    batch(inputs, weights, bias, actual, 2, channels, width)
                    for row in range(2):
                        isolated = (ctypes.c_float * channels)()
                        input_row = ctypes.cast(ctypes.byref(inputs, row * width * 4),
                                                ctypes.POINTER(ctypes.c_float))
                        gemv(isolated, weights, input_row, channels, width)
                        for channel in range(channels):
                            observed = actual[row * channels + channel]
                            if bias is None:
                                self.assertEqual(observed, isolated[channel])
                            else:
                                self.assertAlmostEqual(observed, isolated[channel] + bias[channel], delta=2e-5)
                            oracle = bias_values[channel] if bias is not None else 0.0
                            for block_index, (scale, minimum, quants) in enumerate(specs[channel]):
                                fragment = list(inputs)[row * width + block_index * 32:
                                                        row * width + (block_index + 1) * 32]
                                d = max(abs(v) for v in fragment) / 127.0
                                q8 = [int(math.copysign(math.floor(abs(v / d) + 0.5), v)) if d else 0
                                      for v in fragment]
                                d8 = struct.unpack('<e', struct.pack('<e', d))[0]
                                s8 = struct.unpack('<e', struct.pack('<e', sum(q8) * d))[0]
                                # Independently dequantize the Q5 contribution;
                                # Q8_1's rounded sum is its separate offset term.
                                oracle += (scale * d8) * sum(a * b for a, b in zip(quants, q8)) + minimum * s8
                            self.assertAlmostEqual(observed, oracle, delta=3e-4)


if __name__ == "__main__":
    unittest.main()
