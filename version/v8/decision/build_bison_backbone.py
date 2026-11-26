"""Compose a bounded dense, packed Gemma decision backbone and slot readout."""
from dataclasses import dataclass

from version.v8.decision.build_bison_backbone_ingress import build_circuit as ingress
from version.v8.decision.build_packed_option_readout import option_readout_graph


@dataclass(frozen=True)
class Layer:
    kv_heads: int
    head_dim: int
    visibility: str
    tied_kv: bool = False


class Backbone:
    """Author operations, buffer dimensions and checkpoint bindings for native lowering."""

    def __init__(self, rows, table_rows, width, heads, inner, epsilon, *, blas=False):
        if any(type(n) is not int or n <= 0 for n in (rows, table_rows, width, heads, inner)):
            raise ValueError("positive integer backbone dimensions required")
        self.rows, self.width, self.heads, self.inner, self.epsilon = rows, width, heads, inner, epsilon
        self.blas = blas
        self.graph, self.config = ingress(
            rows, table_rows=table_rows, width=width, q_width=heads, kv_width=heads, epsilon=epsilon,
        )
        self.graph["name"] = "packed_gemma_decision_backbone"
        self.graph["native_entry"]["function"] = "ck_packed_decision_backbone"
        block = self.graph["block_types"]["block"]["body"]
        self.ops = block["ops"] = block["ops"][:1]
        buffers = self.graph["activation_buffers"]
        for name in ("normalized", "q", "k", "v"):
            buffers.pop(name)
            self.graph["activation_bindings"].pop(name)
            self.config["activation_buffer_dtypes"].pop(name)
        self.weights = {}
        self.buffer("positions", [rows], "i32")
        self.buffer("sliding_ranges", [rows, 4], "i32")
        self.buffer("full_ranges", [rows, 4], "i32")

    def buffer(self, name, shape, dtype="bf16"):
        self.graph["activation_buffers"][name] = {"shape": shape}
        self.graph["activation_bindings"][name] = name
        self.config["activation_buffer_dtypes"][name] = dtype

    def emit(self, name, op, inputs, params, shape, *, weights=None, provider=None, output="output"):
        self.buffer(name, shape)
        node = {
            "id": name, "op": op, "kernel": provider or op, "returns_status": True,
            "graph_slots": {"inputs": inputs, "outputs": {output: name}},
            "params": params,
        }
        if weights:
            node["weight_refs"] = {port: spec[0] for port, spec in weights.items()}
            for weight, dimensions, dtype in weights.values():
                declaration = (dimensions, dtype)
                if self.weights.setdefault(weight, declaration) != declaration:
                    raise ValueError(f"incompatible reused weight {weight}")
        self.ops.append(node)
        return name

    def norm(self, name, source, weight, tiles, width, shape):
        n = tiles * width
        return self.emit(name, "rmsnorm_rows_bf16_checked", {"input": source}, {
            "M": tiles, "K": width, "epsilon": self.epsilon,
            "call_constants": {"input_elements": n, "gamma_elements": width,
                               "output_elements": n, "work_elements": n + width,
                               "rows": tiles, "width": width},
        }, shape, weights={"gamma": (weight, [width], "BF16")},
            provider="rmsnorm_rows_weighted_single_round_bf16_checked")

    def project(self, name, source, weight, width, channels):
        rows = self.rows
        constants = {
            "rows": rows, "input_width": width, "output_width": channels,
            "input_elements": rows * width, "weight_elements": channels * width,
            "output_elements": rows * channels, "staged_elements": rows * channels,
        }
        if self.blas:
            constants["work_elements"] = rows * width + channels * width + rows * channels
        return self.emit(name, "linear_rows_bf16_checked", {"input": source}, {
            "M": rows, "K": width, "N": channels, "call_constants": constants,
        }, [rows, channels], weights={"weight": (weight, [channels, width], "BF16")},
            provider="linear_rows_bf16_blas_checked" if self.blas else None)

    def rotate(self, name, source, heads, dim, visibility):
        return self.emit(name, "rope_split_positions_bf16_checked",
                         {"input": source, "positions": "external:positions"},
                         {"T": self.rows, "H": heads, "D": dim, "call_constants": {
                             "tokens": self.rows, "heads": heads, "dim": dim,
                             "elements": self.rows * heads * dim, "frequencies": dim // 2,
                         }}, [self.rows, heads * dim],
                         weights={"inv_freq": (f"rope.{visibility}.{dim}", [dim // 2], "F32")},
                         output="out")

    def pointwise(self, name, op, inputs, width, weights=None):
        count = self.rows * width
        constants = {"elements": count}
        if weights:
            constants["gain_elements"] = 1
        return self.emit(name, op, inputs, {"N": count, "call_constants": constants},
                         [self.rows, width], weights=weights, output="out")

    def layer(self, index, layer, source):
        if (layer.visibility not in {"sliding", "full"} or type(layer.kv_heads) is not int or
            layer.kv_heads < 1 or self.heads % layer.kv_heads or type(layer.head_dim) is not int or
            layer.head_dim < 2 or layer.head_dim % 2 or type(layer.tied_kv) is not bool):
            raise ValueError("invalid packed attention geometry")
        rows, width, heads, inner = self.rows, self.width, self.heads, self.inner
        kv, dim = layer.kv_heads, layer.head_dim
        qwidth, kvwidth = heads * dim, kv * dim
        prefix = f"l{index}_"
        weight = f"model.language_model.layers.{index}."

        def hidden_norm(name, buffer, suffix):
            return self.norm(prefix + name, buffer, weight + suffix, rows, width, [rows, width])

        norm = hidden_norm("norm", source, "input_layernorm.weight")
        q = self.project(prefix + "q", norm, weight + "self_attn.q_proj.weight", width, qwidth)
        k = self.project(prefix + "k", norm, weight + "self_attn.k_proj.weight", width, kvwidth)
        v = k if layer.tied_kv else self.project(
            prefix + "v", norm, weight + "self_attn.v_proj.weight", width, kvwidth)
        q = self.norm(prefix + "qnorm", q, weight + "self_attn.q_norm.weight", rows * heads, dim, [rows, qwidth])
        k = self.norm(prefix + "knorm", k, weight + "self_attn.k_norm.weight", rows * kv, dim, [rows, kvwidth])
        v = self.norm(prefix + "vnorm", v, f"unit.{dim}", rows * kv, dim, [rows, kvwidth])
        q = self.rotate(prefix + "qrope", q, heads, dim, layer.visibility)
        k = self.rotate(prefix + "krope", k, kv, dim, layer.visibility)
        attention = self.emit(prefix + "context", "attention_ranges_checked", {
            "query": q, "key": k, "value": v, "ranges": f"external:{layer.visibility}_ranges",
        }, {"Q": rows, "S": rows, "H": heads, "G": kv, "D": dim,
            "X": rows * qwidth + rows, "scale": 1.0, "call_constants": {
                "query_elements": rows * qwidth, "key_elements": rows * kvwidth,
                "value_elements": rows * kvwidth, "output_elements": rows * qwidth,
                "scratch_elements": rows * qwidth + rows, "range_elements": rows * 4,
                "query_tokens": rows, "memory_tokens": rows, "heads": heads,
                "kv_heads": kv, "head_dim": dim,
            }}, [rows, qwidth], provider="attention_ranges_checked_bf16")
        attention = self.project(prefix + "attn_out", attention, weight + "self_attn.o_proj.weight", qwidth, width)
        attention = hidden_norm("attn_norm", attention, "post_attention_layernorm.weight")
        residual = self.pointwise(prefix + "residual", "add_bf16_checked",
                                  {"left": source, "right": attention}, width)
        norm = hidden_norm("mlp_norm", residual, "pre_feedforward_layernorm.weight")
        gate = self.project(prefix + "gate", norm, weight + "mlp.gate_proj.weight", width, inner)
        up = self.project(prefix + "up", norm, weight + "mlp.up_proj.weight", width, inner)
        product = self.pointwise(prefix + "geglu", "geglu_tanh_bf16_checked", {"gate": gate, "up": up}, inner)
        mlp = self.project(prefix + "down", product, weight + "mlp.down_proj.weight", inner, width)
        mlp = hidden_norm("down_norm", mlp, "post_feedforward_layernorm.weight")
        result = self.pointwise(prefix + "sum", "add_bf16_checked",
                                {"left": residual, "right": mlp}, width)
        return self.pointwise(prefix + "output", "scale_bf16_checked", {"input": result}, width,
                              weights={"weight": (weight + "layer_scalar", [1], "BF16")})

    def readout(self, source, questions, labels):
        norm = self.norm("final_hidden", source, "model.language_model.norm.weight",
                         self.rows, self.width, [self.rows, self.width])
        readout = option_readout_graph(self.rows, questions, self.width, labels)
        for name, dtype in (("answer_rows", "i32"), ("selected", "bf16"), ("scores", "bf16")):
            self.buffer(name, readout["activation_buffers"][name]["shape"], dtype)
        ops = readout["block_types"]["block"]["body"]["ops"]
        ops[0]["graph_slots"]["inputs"]["input"] = norm
        self.ops.extend(ops)
        self.weights["option.weight"] = ([labels, self.width], "BF16")
