"""Author the dense-plus-routed Gemma decision layer's expert branch."""
from dataclasses import dataclass

from version.v8.decision.build_bison_backbone import Backbone


@dataclass(frozen=True)
class Experts:
    count: int
    top_k: int
    inner: int
    partition_topk: bool = False

    def __post_init__(self):
        if (any(type(n) is not int or n < 1 for n in (self.count, self.top_k, self.inner)) or
            self.top_k > self.count):
            raise ValueError("invalid routed expert geometry")


def add_moe(backbone, index, geometry, residual):
    """Declare router and expert edges. The sole expert loop runs inside its provider."""
    rows, width = backbone.rows, backbone.width
    experts, k, inner = geometry.count, geometry.top_k, geometry.inner
    backbone.config.update(num_experts=experts, experts_per_tok=k)
    p = f"l{index}_"
    w = f"model.language_model.layers.{index}."
    router = backbone.norm(p + "router_norm", residual, f"unit.{width}", rows, width, [rows, width])
    router = backbone.emit(p + "router_weighted", "scale_rows_bf16_checked", {"input": router}, {
        "N": rows * width, "D": width,
        "call_constants": {"elements": rows * width, "gain_elements": width},
    }, [rows, width], weights={"weight": (w + "router.scale", [width], "BF16")}, output="out")
    router = backbone.emit(p + "router_scaled", "scale_bf16_checked", {"input": router}, {
        "N": rows * width, "scale": width ** -0.5, "call_constants": {"elements": rows * width},
    }, [rows, width], provider="scale_scalar_f32_bf16_checked", output="out")
    logits = backbone.project(p + "router_logits", router, w + "router.proj.weight", width, experts)
    widened = backbone.emit(p + "router_f32", "widen_bf16_checked", {"input": logits}, {
        "N": rows * experts, "call_constants": {"elements": rows * experts},
    }, [rows, experts], output="out")
    backbone.config["activation_buffer_dtypes"][widened] = "fp32"
    ids, probs = p + "expert_ids", p + "expert_probabilities"
    backbone.buffer(ids, [rows, k], "i32")
    backbone.buffer(probs, [rows, k], "fp32")
    backbone.ops.append({
        "id": p + "route", "op": "full_softmax_topk_router",
        "kernel": ("moe_softmax_topk_router_partition_f32" if geometry.partition_topk
                   else "moe_softmax_topk_router_llama_f32"),
        "returns_status": True,
        "graph_slots": {"inputs": {"logits": widened}, "outputs": {"indices": ids, "weights": probs}},
        "params": {"R": rows, "E": experts, "K": k, "routed_scaling_factor": 1.0},
    })
    scaled = backbone.emit(p + "expert_weights", "indexed_scale_checked", {"values": probs, "ids": ids}, {
        "N": rows * k, "V": experts, "call_constants": {"count": rows * k, "scale_count": experts},
    }, [rows, k], weights={"scales": (w + "router.per_expert_scale", [experts], "F32")},
        provider="indexed_scale_checked_f32")
    backbone.config["activation_buffer_dtypes"][scaled] = "fp32"

    normalized = backbone.norm(p + "expert_norm", residual, w + "pre_feedforward_layernorm_2.weight",
                               rows, width, [rows, width])
    projection = {} if backbone.cblas else {"use_blas": int(backbone.blas)}
    routed = backbone.emit(p + "expert_sum", "routed_geglu_bf16_checked", {
        "input": normalized, "ids": ids, "scores": scaled,
    }, {
        "R": rows, "H": width, "I": inner, "E": experts, "K": k, **projection,
        "call_constants": {
            "rows": rows, "width": width, "inner": inner, "experts": experts, "top_k": k,
            "hidden_elements": rows * width, "selected_elements": rows * k,
            "route_elements": rows * k * width, "gate_elements": experts * 2 * inner * width,
            "down_elements": experts * width * inner,
        },
    }, [rows, width], weights={
        "gate_up": (w + "experts.gate_up_proj", [experts, 2 * inner, width], "BF16"),
        "down": (w + "experts.down_proj", [experts, width, inner], "BF16"),
    }, output="out", provider="routed_geglu_bf16_cblas_checked" if backbone.cblas else None)
    backbone.buffer(p + "expert_routes", [rows, k, width])
    backbone.ops[-1]["graph_slots"]["outputs"]["routes"] = p + "expert_routes"
    return routed


def build_moe_replay(rows, width, geometry, epsilon, *, blas=False, cblas=False):
    """Take residual rows at the exact ingress of a decoder-layer expert branch."""
    backbone = Backbone(rows, 1, width, 1, width, epsilon, blas=blas, cblas=cblas)
    backbone.graph["name"] = "packed_gemma_moe_replay"
    backbone.graph["native_entry"]["function"] = "ck_packed_gemma_moe_replay"
    backbone.ops.clear()
    for name in ("token_rows", "embedding_table", "positions", "sliding_ranges", "full_ranges"):
        backbone.graph["activation_buffers"].pop(name)
        backbone.graph["activation_bindings"].pop(name)
        backbone.config["activation_buffer_dtypes"].pop(name)
    add_moe(backbone, 0, geometry, "external:input")
    return backbone
