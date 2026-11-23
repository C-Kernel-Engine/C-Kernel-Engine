"""Extend Drama 3's codebook graph through its windowed codec transformer."""
from version.v8.scripts.checked_circuit_authoring_v8 import linear
from version.v8.tts.build_drama3_codec_rvq_circuit import build_circuit as rvq_circuit


def build_circuit(frames: int, *, layers: int = 8, heads: int = 16, head_dim: int = 64,
                  intermediate: int = 3072, window: int = 128, **rvq) -> tuple[dict, dict]:
    """Declare a fixed-frame graph from code IDs through the final transformer norm."""
    if any(type(n) is not int or n < 1 for n in (layers, heads, head_dim, intermediate, window)):
        raise ValueError("transformer dimensions must be positive integers")
    if head_dim % 2 or type(frames) is not int or not 1 <= frames <= 2048:
        raise ValueError("expected an even rotary head width and at most 2048 frames")
    width = heads * head_dim
    graph, config = rvq_circuit(frames, channels=width, **rvq)
    graph.update(name="drama3_codec_transformer")
    graph["native_entry"]["function"] = "ck_drama3_codec_transformer"
    ops = graph["block_types"]["rvq"]["body"]["ops"]

    def buffer(name, channels):
        graph["activation_buffers"][name] = {"shape": [frames, channels]}
        graph["activation_bindings"][name] = name

    buffer("rotary_cos", head_dim // 2)
    buffer("rotary_sin", head_dim // 2)

    def append(name, channels, op, kernel, inputs, params, *, weights=None):
        buffer(name, channels)
        entry = {"id": name, "op": op, "kernel": kernel, "returns_status": True,
                 "graph_slots": {"inputs": inputs, "outputs": {"output": name}},
                 "params": params}
        if weights is not None:
            entry["weight_refs"] = weights
        ops.append(entry)

    def project(name, source, input_width, channels, weight):
        buffer(name, channels)
        entry = linear(name, source, name, input_width, channels, frames, weight)
        entry.update(op="linear_rows_checked_no_bias", kernel="linear_rows_checked_no_bias_f32",
                     weight_refs={"weight": f"{weight}.weight"})
        del entry["params"]["call_constants"]["linear_bias_elements"]
        ops.append(entry)

    def norm(name, source, weight):
        append(name, width, "rmsnorm_rows_checked", "rmsnorm_rows_checked_f32", {"input": source},
               {"R": frames, "C": width, "eps": 1e-5, "call_constants": {
                   "input_elements": frames * width, "input_stride": width, "gamma_elements": width,
                   "output_elements": frames * width, "output_stride": width, "rows": frames,
                   "width": width}}, weights={"gamma": weight})

    def rotate(name, source):
        append(name, width, "rope_pairwise_rows_checked", "rope_pairwise_rows_f32_checked",
               {"input": source, "cosine": "external:rotary_cos", "sine": "external:rotary_sin"},
               {"T": frames, "H": heads, "D": head_dim, "call_constants": {
                   "rotation_input_elements": frames * width, "rotation_output_elements": frames * width,
                   "rotation_table_elements": frames * head_dim // 2, "rotation_tokens": frames,
                   "rotation_heads": heads, "rotation_head_dim": head_dim}})

    def change_layout(name, source, *, to_head):
        kernel = "layout_token_to_head_checked_f32" if to_head else "layout_head_to_token_checked_f32"
        count = frames * width
        append(name, width, "token_head_layout_checked", kernel, {"input": source},
               {"T": frames, "H": heads, "D": head_dim, "call_constants": {
                   "input_elements": count, "output_elements": count, "tokens": frames,
                   "heads": heads, "head_dim": head_dim}})
        if to_head:
            graph["activation_buffers"][name]["shape"] = [heads, frames, head_dim]

    def residual(name, source, delta, scale):
        scaled = f"{name}_scaled"
        count = frames * width
        append(scaled, width, "channel_scale_rows_checked", "channel_scale_rows_checked_f32",
               {"input": delta}, {"R": frames, "C": width, "call_constants": {
                   "input_elements": count, "input_stride": width, "scale_elements": width,
                   "output_elements": count, "output_stride": width, "rows": frames,
                   "channels": width}}, weights={"scale": scale})
        append(name, width, "add_rows_checked", "add_rows_checked_f32",
               {"left": source, "right": scaled}, {"N": count, "call_constants": {
                   "left_elements": count, "right_elements": count, "output_elements": count,
                   "elements": count}})

    hidden = "quantized"
    for i in range(layers):
        prefix = f"codec_layer{i}"
        norm(f"{prefix}_attn_norm", hidden, f"{prefix}.attention_norm.weight")
        for projection in ("q", "k", "v"):
            project(f"{prefix}_{projection}", f"{prefix}_attn_norm", width, width,
                    f"{prefix}.attention.{projection}")
        rotate(f"{prefix}_q_rotary", f"{prefix}_q")
        rotate(f"{prefix}_k_rotary", f"{prefix}_k")
        count = frames * width
        for port, suffix in (("q", "q_rotary"), ("k", "k_rotary"), ("v", "v")):
            change_layout(f"{prefix}_{port}_head", f"{prefix}_{suffix}", to_head=True)
        work, output = f"{prefix}_attention_work", f"{prefix}_attention_head"
        for name in (work, output):
            buffer(name, width)
            graph["activation_buffers"][name]["shape"] = [heads, frames, head_dim]
        ops.append({
            "id": output, "op": "attention_sliding_bounded",
            "kernel": "attention_sliding_head_major_bounded_f32", "returns_status": True,
            "graph_slots": {"inputs": {port: f"{prefix}_{port}_head" for port in ("q", "k", "v")},
                            "outputs": {"work": work, "out": output}},
            "params": {"T": frames, "H": heads, "KV": heads, "D": head_dim, "W": window,
                       "call_constants": {"q_elements": count, "k_elements": count, "v_elements": count,
                           "work_elements": count, "output_elements": count, "tokens": frames,
                           "heads": heads, "kv_heads": heads, "head_dim": head_dim,
                           "sliding_window": window}},
        })
        change_layout(f"{prefix}_attention", output, to_head=False)
        project(f"{prefix}_attn_proj", f"{prefix}_attention", width, width, f"{prefix}.attention.wo")
        residual(f"{prefix}_attn_residual", hidden, f"{prefix}_attn_proj",
                 f"{prefix}.attention_layer_scale.gamma")
        norm(f"{prefix}_ffn_norm", f"{prefix}_attn_residual", f"{prefix}.ffn_norm.weight")
        for output, weight in (("gate", "w1"), ("up", "w3")):
            project(f"{prefix}_{output}", f"{prefix}_ffn_norm", width, intermediate,
                    f"{prefix}.feed_forward.{weight}")
        inner_count = frames * intermediate
        append(f"{prefix}_gated", intermediate, "silu_mul_rows_checked", "silu_mul_rows_checked_f32",
               {"gate": f"{prefix}_gate", "up": f"{prefix}_up"}, {"N": inner_count, "call_constants": {
                   "gate_elements": inner_count, "up_elements": inner_count, "elements": inner_count,
                   "output_elements": inner_count}})
        project(f"{prefix}_ffn", f"{prefix}_gated", intermediate, width, f"{prefix}.feed_forward.w2")
        hidden = f"{prefix}_out"
        residual(hidden, f"{prefix}_attn_residual", f"{prefix}_ffn", f"{prefix}.ffn_layer_scale.gamma")
    norm("codec_transformer_out", hidden, "codec_transformer_norm.weight")
    graph["contract"]["runtime_invariants"]["codec_attention_window"] = window
    return graph, config
