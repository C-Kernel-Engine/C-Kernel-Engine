"""Checked generated CLIP block and projector used by Instella-VL bring-up."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from version.v8.scripts.checked_circuit_authoring_v8 import linear
from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph


class Graph:
    def __init__(self, name, rows, width, heads, inner):
        self.rows = rows
        self.weights = {}
        self.buffers = {"input": {"shape": [rows, width]}}
        self.ops = []
        self.circuit = {
            "version": 3, "name": name, "family": "bounded_graph", "checked_native_entry": True,
            "contract": {"runtime_invariants": {
                "inference_only": True, "tokens": rows, "width": width, "heads": heads, "inner": inner,
                "mult": 1,
            }},
            "activation_buffers": self.buffers,
            "sequence": ["block"],
            "block_types": {"block": {
                "sequence": ["header", "body", "footer"], "header": [], "footer": [],
                "body": {"type": "dense", "ops": self.ops},
            }},
            "native_entry": {"function": "ck_" + name, "params": [
                {"c_type": "uint8_t *", "name": "arena"}, {"c_type": "size_t", "name": "arena_bytes"}],
                "arena": {"pointer": "arena", "bytes": "arena_bytes"}},
        }

    def op(self, name, operation, kernel, sources, width, scalars, *, refs=None, **params):
        self.buffers[name] = {"shape": [self.rows, width]}
        node = {"id": name, "op": operation, "kernel": kernel, "returns_status": True,
                "graph_slots": {"inputs": sources, "outputs": {"output": name}},
                "params": {**params, "call_constants": scalars}}
        if refs:
            node["weight_refs"] = refs
        self.ops.append(node)

    def weight(self, name, *shape):
        self.weights[name] = (list(shape), "F32")
        return name

    def project(self, name, source, k, n, key):
        self.weight(key + ".weight", n, k)
        self.weight(key + ".bias", n)
        self.buffers[name] = {"shape": [self.rows, n]}
        self.ops.append(linear(name, source, name, k, n, self.rows, key))

    def norm(self, name, source, width, key):
        count, scratch = self.rows * width, self.rows * (width + 2)
        self.op(name, "layernorm_rows_param_checked", "layernorm_rows_checked_param_epsilon_f32",
                {"input": source}, width, {
                    "norm_input_elements": count, "norm_input_stride": width,
                    "norm_gamma_elements": width, "norm_beta_elements": width,
                    "norm_output_elements": count, "norm_output_stride": width,
                    "norm_scratch_elements": scratch, "norm_rows": self.rows, "norm_channels": width,
                }, refs={"gamma": self.weight(key + ".weight", width),
                         "beta": self.weight(key + ".bias", width)},
                M=self.rows, C=width, Q=scratch, epsilon=1e-5)

    def add(self, name, left, right, width):
        count = self.rows * width
        self.op(name, "add_rows_checked", "add_rows_checked_f32", {"left": left, "right": right},
                width, {"left_elements": count, "right_elements": count, "output_elements": count,
                        "elements": count}, N=count)

    def finish(self):
        self.circuit["activation_bindings"] = {key: key for key in self.buffers}
        return self


def clip_block(rows=577, width=1024, heads=16, inner=4096):
    if any(type(n) is not int or n <= 0 for n in (rows, width, heads, inner)) or width % heads:
        raise ValueError("invalid CLIP block geometry")
    g = Graph("instella_clip_block", rows, width, heads, inner)
    g.norm("norm1", "external:input", width, "layer_norm1")
    for port in ("q", "k", "v"):
        g.project(port, "norm1", width, width, f"self_attn.{port}_proj")
    count, scratch = rows * width, rows * (width + 1)
    g.op("context", "attention_cross_token_major_checked", "attention_cross_token_major_checked_f32",
         {"query": "q", "key": "k", "value": "v"}, width,
         {"query_elements": count, "key_elements": count, "value_elements": count,
          "output_elements": count, "scratch_elements": scratch, "query_tokens": rows,
          "memory_tokens": rows, "heads": heads, "head_dim": width // heads},
         Q=rows, S=rows, H=heads, D=width // heads, X=scratch)
    g.project("attention_out", "context", width, width, "self_attn.out_proj")
    g.add("attention_residual", "external:input", "attention_out", width)
    g.norm("norm2", "attention_residual", width, "layer_norm2")
    g.project("mlp_hidden", "norm2", width, inner, "mlp.fc1")
    count = rows * inner
    g.op("quick_gelu_scaled", "channel_scale_rows_checked", "channel_scale_rows_checked_f32",
         {"input": "mlp_hidden"}, inner, {
             "input_elements": count, "input_stride": inner, "scale_elements": inner,
             "output_elements": count, "output_stride": inner, "rows": rows, "channels": inner,
         }, refs={"scale": g.weight("quick_gelu_scale", inner)}, R=rows, C=inner)
    g.op("quick_gelu", "sigmoid_gate_rows_checked", "sigmoid_gate_rows_checked_f32",
         {"input": "mlp_hidden", "gate": "quick_gelu_scaled"}, inner,
         {"input_elements": count, "gate_elements": count, "output_elements": count, "elements": count},
         N=count)
    g.project("mlp_output", "quick_gelu", inner, width, "mlp.fc2")
    g.add("result", "attention_residual", "mlp_output", width)
    return g.finish()


def projector(rows=576, width=1024, output_width=2048):
    g = Graph("instella_clip_projector", rows, width, 1, output_width)
    g.project("hidden", "external:input", width, output_width, "0")
    count = rows * output_width
    g.op("activated", "gelu", "gelu_rows_erf_checked_f32", {"input": "hidden"},
         output_width, {"gelu_input_elements": count, "gelu_input_stride": output_width,
                        "gelu_output_elements": count, "gelu_output_stride": output_width,
                        "gelu_scratch_elements": count, "gelu_rows": rows,
                        "gelu_channels": output_width}, M=rows, C=output_width, Q=count)
    g.project("result", "activated", output_width, output_width, "2")
    return g.finish()


def patch_projection(rows=576, patch_width=588, width=1024):
    g = Graph("instella_clip_patch", rows, patch_width, 1, width)
    op = linear("result", "external:input", "result", patch_width, width, rows, "patch")
    op.update(op="linear_rows_checked_no_bias", kernel="linear_rows_checked_no_bias_f32",
              weight_refs={"weight": g.weight("patch.weight", width, patch_width)})
    del op["params"]["call_constants"]["linear_bias_elements"]
    g.buffers["result"] = {"shape": [rows, width]}
    g.ops.append(op)
    return g.finish()


def input_norm(rows=577, width=1024):
    g = Graph("instella_clip_input_norm", rows, width, 1, width)
    g.buffers["position"] = {"shape": [rows, width]}
    g.add("positioned", "external:input", "external:position", width)
    g.norm("result", "positioned", width, "pre_layrnorm")
    return g.finish()


class CompiledClipStage:
    def __init__(self, directory, graph, initial_weights, *, openmp=False):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.graph = graph
        self.run, self.arena, self.view, self.library = compile_checked_weight_graph(
            directory, graph.circuit, initial_weights, graph.weights, openmp=openmp)
        layout = json.loads((directory / "layout.json").read_text())
        self.weights = {slot["name"]: slot["abs_offset"]
                        for slot in layout["memory"]["weights"]["entries"]}

    def load(self, weights):
        if weights.keys() != self.graph.weights.keys():
            raise ValueError("CLIP stage weight keys differ from the compiled circuit")
        for name, (shape, _) in self.graph.weights.items():
            value = weights[name]
            if value.dtype != np.float32 or tuple(shape) != value.shape:
                raise ValueError(f"CLIP stage weight properties changed: {name}")
        for name, (shape, _) in self.graph.weights.items():
            value = weights[name]
            np.ndarray(shape, np.float32, buffer=self.arena, offset=self.weights[name])[:] = value

    def __call__(self, data, *, position=None):
        inputs = {"input": data}
        if "position" in self.graph.buffers:
            inputs["position"] = position
        elif position is not None:
            raise ValueError("CLIP stage does not accept a position input")
        for name, value in inputs.items():
            target = self.view(name)
            if value is None or target.shape != value.shape or value.dtype != np.float32:
                raise ValueError(f"CLIP {name} does not match generated arena")
            target[:] = value
        status = self.run(self.arena, len(self.arena))
        if status != 0:
            raise RuntimeError(f"generated CLIP stage failed with status {status}")
        return self.view("result").copy()
