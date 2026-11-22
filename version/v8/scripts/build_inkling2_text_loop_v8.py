"""Compose Inkling's fixed text loop as one checked generated graph."""

from copy import deepcopy
import math

from version.v8.scripts.audit_inkling2_mini_v8 import text_schedule
from version.v8.scripts.build_inkling2_decoder_block_v8 import _bind_q8_projection, decoder_block
from version.v8.scripts.build_inkling2_mini_projection_circuit import _graph, _project
from version.v8.scripts.checked_circuit_authoring_v8 import (
    bind_runtime_scalars, execution_port, runtime_extent,
)


def hidden_text_loop(config, prefill_tokens=4, *, cache_capacity=None, circular_sliding=False):
    """Execute all occurrences for supplied hidden rows, with isolated slot state."""
    schedule = text_schedule(config)
    width = config["hidden_size"]
    block_size = config["n_layers_per_attn_res_block"]
    loop = config["loop_config"]
    start, end = loop["core_start"], loop["core_end"]
    if end - start != 2 * block_size or len(schedule) > 48 or config["num_hidden_layers"] % block_size:
        raise ValueError("expected complete residual blocks, two per core pass, and at most 48 state slots")
    experts = config["n_routed_experts"]
    if circular_sliding and cache_capacity is None:
        raise ValueError("circular sliding KV requires an explicit context capacity")
    programs, circular_slots = {}, {}
    for occurrence in schedule:
        layer, slot = occurrence["weight_layer"], occurrence["state_slot"]
        full = occurrence["attention"] == "full"
        sliding_heads = config["swa_num_attention_heads"]
        sliding_kv = config["swa_num_key_value_heads"]
        boundary = layer % block_size == 0
        window = 0 if full else config["sliding_window_size"]
        physical = cache_capacity
        if circular_sliding and window:
            bounded = max(prefill_tokens + 1, window + prefill_tokens - 1)
            if cache_capacity > bounded:
                physical = bounded
                circular_slots[str(slot)] = physical
        programs[slot] = decoder_block(
            prefill_tokens, width, config["num_key_value_heads"] if full else sliding_kv,
            config["num_attention_heads"] if full else sliding_heads,
            config["head_dim"] if full else config["swa_head_dim"],
            config["d_rel"],
            config["rel_extent"] if full else config["sliding_window_size"],
            window, config["intermediate_size"],
            experts, config["num_experts_per_tok"], experts, config["sconv_kernel_size"],
            config["rms_norm_eps"], True,
            layer=layer, state_slot=slot, depth_states=(layer + block_size - 1) // block_size,
            previous_pass_states=int(occurrence["takes_prior_pass_message"]),
            block_boundary=boundary, control_rank=loop["gate_rank"], full_attention=full,
            log_floor=config["log_scaling_n_floor"], log_alpha=config["log_scaling_alpha"],
            cache_capacity=physical,
        )
        if str(slot) in circular_slots:
            for node in programs[slot]["block_types"]["block"]["body"]["ops"]:
                if node["kernel"] == "kv_cache_l2_rms_append_checked_f32":
                    node["kernel"] = "kv_cache_l2_rms_append_ring_checked_f32"
                if node["kernel"] == "gqa_relative_sink_checked_f32":
                    node["kernel"] = "gqa_relative_sink_ring_checked_f32"

    graph = _graph("text_loop", prefill_tokens, width, {}, [], layer=0,
                   scope="bounded hidden rows through the entire fixed loop and final normalization")
    graph["name"] = "inkling2_mini_text_loop"
    graph["native_entry"]["function"] = "ck_inkling2_mini_text_loop"
    graph["native_entry"]["segments"] = []
    graph["contract"]["runtime_invariants"].update(
        layer_schedule=schedule, circular_sliding_slots=circular_slots)
    buffers = graph["activation_buffers"]
    bindings = graph["activation_bindings"]
    ops = graph["block_types"]["block"]["body"]["ops"]

    def buffer(name, shape):
        buffers[name] = {"shape": shape}
        bindings[name] = name
        return name

    for phase, rows in (("prefill", prefill_tokens), ("append", 1)):
        current = "external:" + buffer(phase + "_hidden", [rows, width])
        depth_bank = None
        core_hidden = core_bank = previous_bank = previous_output = next_message_bank = None
        for item in schedule:
            slot, layer = item["state_slot"], item["weight_layer"]
            if layer == start:
                if not item["loop_pass"]:
                    core_hidden, core_bank = current, depth_bank
                else:
                    previous_bank, previous_output = next_message_bank, current
                    current, depth_bank = core_hidden, core_bank
            if layer == start + block_size:
                next_message_bank = current

            block = programs[slot]
            entry = f"inkling2_{phase}_slot{slot}"
            lookup = {
                entry + "_hidden": current,
                entry + "_depth_bank": depth_bank,
                entry + "_previous_bank": previous_bank,
                entry + "_previous_output": previous_output,
                "router_bias": f"external:router_bias_{layer}",
                "expert_slots": "external:expert_slots",
                "history_attn_post_conv": f"external:history_attn_post_conv_{slot}",
                "history_ffn_post_conv": f"external:history_ffn_post_conv_{slot}",
            }
            body = block["block_types"]["block"]["body"]["ops"]
            current_output = entry + "_layer_output"
            for original in body:
                if not original["id"].startswith(entry + "_"):
                    continue
                node = deepcopy(original)
                for ports in node["graph_slots"].values():
                    for port, source in ports.items():
                        key = source.removeprefix("external:")
                        if key not in block["activation_buffers"]:
                            raise ValueError(f"missing block tensor {key}")
                        bound = lookup.get(key, source)
                        if not bound:
                            raise ValueError(f"missing incoming state for {key}")
                        ports[port] = bound
                        if key in lookup and not bound.startswith("external:"):
                            continue
                        target = bound.removeprefix("external:")
                        if target not in buffers:
                            buffers[target] = deepcopy(block["activation_buffers"][key])
                            bindings[target] = target
                ops.append(node)
            if layer % block_size == 0:
                depth_bank = entry + "_next_depth_bank" if depth_bank is not None else current
            current = current_output
            producer = next(op for op in body if op["id"] == current)
            graph["semantic_checkpoints"]["exports"][current] = {
                "section": "body", "template_op_id": current, "op": producer["op"],
                "checkpoints": [{
                    "id": f"inkling2.loop.{phase}.slot{slot}.layer_output",
                    "producer": current, "tensor": current,
                    "logical_layout": "token_major_contiguous",
                    "axis_names": ["token", "channel"], "storage_dtype": "fp32",
                }],
            }

        count, states = rows * width, config["num_hidden_layers"] // block_size
        mixed = buffer(phase + "_final_depth_mix", [rows, width])
        output = buffer(phase + "_hidden_output", [rows, width])
        eps = config["rms_norm_eps"]
        ops.extend((
            {"id": mixed, "op": "depth_residual_mix_checked", "kernel": "depth_residual_mix_checked_f32",
             "returns_status": True, "weight_refs": {"query": "output_res_query.weight"},
             "graph_slots": {"inputs": {"bank": depth_bank, "current": current}, "outputs": {"output": mixed}},
             "params": {"R": rows, "S": states, "C": width, "epsilon": eps, "call_constants": {
                 "rows": rows, "states": states, "width": width, "bank_elements": count * states,
                 "current_elements": count, "query_elements": width, "output_elements": count}}},
            {"id": output, "op": "rmsnorm_rows_checked", "kernel": "rmsnorm_rows_checked_f32",
             "returns_status": True, "weight_refs": {"gamma": "output_norm.weight"},
             "graph_slots": {"inputs": {"input": mixed}, "outputs": {"output": output}},
             "params": {"R": rows, "C": width, "eps": eps, "call_constants": {
                 "rows": rows, "width": width, "gamma_elements": width,
                 "input_stride": width, "output_stride": width,
                 "input_elements": count, "output_elements": count}}},
        ))
        graph["semantic_checkpoints"]["exports"][output] = {
            "section": "body", "template_op_id": output, "op": "rmsnorm_rows_checked",
            "checkpoints": [{"id": f"inkling2.loop.{phase}.last_hidden", "producer": output, "tensor": output,
                             "logical_layout": "token_major_contiguous",
                             "axis_names": ["token", "channel"], "storage_dtype": "fp32"}],
        }
        graph["native_entry"]["segments"].append({"function": f"ck_inkling2_loop_{phase}", "last_op": output})

    return graph


def text_token_loop(config, prefill_tokens=4, *, cache_capacity=None, f64_norm_sum=False, last_prefill_logit=False,
                    reference_moe=False, fp32_static_conv=False, circular_sliding=False):
    """Bind packed token lookup and capped full-vocabulary logits to the fixed loop."""
    graph = hidden_text_loop(config, prefill_tokens, cache_capacity=cache_capacity,
                             circular_sliding=circular_sliding)
    width, vocab = config["hidden_size"], config["vocab_size"]
    logits_width = config.get("unpadded_vocab_size", vocab)
    cap, eps = config["final_logit_softcapping"], config["rms_norm_eps"]
    if (type(vocab) is not int or type(logits_width) is not int or not 1 <= logits_width <= vocab
            or not math.isfinite(cap) or cap <= 0):
        raise ValueError("invalid text vocabulary or logit cap")
    graph["contract"]["runtime_invariants"].update(
        vocab=vocab, logits_width=logits_width, scope="bounded packed token IDs through the fixed loop to capped logits"
    )
    buffers, bindings = graph["activation_buffers"], graph["activation_bindings"]
    body = graph["block_types"]["block"]["body"]
    segments = graph["native_entry"]["segments"]

    def buffer(name, shape):
        buffers[name] = {"shape": shape}
        bindings[name] = name
        return name

    old_ops = iter(body["ops"])
    body["ops"] = ops = []
    for index, (phase, rows) in enumerate((("prefill", prefill_tokens), ("append", 1))):
        ids = buffer(phase + "_token_ids", [rows])
        embedding = buffer(phase + "_embeddings", [rows, width])
        hidden = phase + "_hidden"
        count = rows * width
        ops.extend((
            {"id": embedding, "op": "embedding_lookup_q8_0_checked",
             "kernel": "embedding_lookup_q8_0_checked_f32", "returns_status": True,
             "weight_refs": {"table": "token_embd.weight"},
             "graph_slots": {"inputs": {"ids": "external:" + ids}, "outputs": {"output": embedding}},
             "params": {"T": rows, "V": vocab, "C": width, "call_constants": {
                 "tokens": rows, "vocabulary": vocab, "width": width, "ids_elements": rows,
                 "table_bytes": vocab * (width // 32) * 34, "output_elements": count}}},
            {"id": hidden, "op": "rmsnorm_rows_checked", "kernel": "rmsnorm_rows_checked_f32",
             "returns_status": True, "weight_refs": {"gamma": "token_embd_norm.weight"},
             "graph_slots": {"inputs": {"input": embedding}, "outputs": {"output": hidden}},
             "params": {"R": rows, "C": width, "eps": eps, "call_constants": {
                 "rows": rows, "width": width, "gamma_elements": width, "input_stride": width,
                 "output_stride": width, "input_elements": count, "output_elements": count}}},
        ))
        output = segments[index]["last_op"]
        for node in old_ops:
            inputs = node["graph_slots"]["inputs"]
            for port, source in inputs.items():
                if source == "external:" + hidden:
                    inputs[port] = hidden
            ops.append(node)
            if node["id"] == output:
                break
        logit_rows = rows
        if phase == "prefill" and last_prefill_logit and rows > 1:
            last = buffer("prefill_last_hidden", [1, width])
            ops.append({
                "id": last, "op": "copy_last_row_checked", "kernel": "copy_last_row_checked_f32",
                "returns_status": True,
                "graph_slots": {"inputs": {"input": output}, "outputs": {"output": last}},
                "params": {"R": rows, "C": width, "call_constants": {
                    "rows": rows, "width": width, "input_elements": count, "output_elements": width,
                }},
            })
            output, logit_rows = last, 1
        raw = buffer(phase + "_raw_logits", [logit_rows, logits_width])
        project = _project(raw, output, width, logits_width, logit_rows, "output.weight", weight_prefix="")
        _bind_q8_projection(project, True)
        ops.append(project)
        logits = buffer(phase + "_logits", [logit_rows, logits_width])
        ops.append({
            "id": logits, "op": "softcap_logits_checked", "kernel": "softcap_logits_checked_f32",
            "returns_status": True,
            "graph_slots": {"inputs": {"input": raw}, "outputs": {"output": logits}},
            "params": {"N": logit_rows * logits_width, "cap": cap, "call_constants": {
                "elements": logit_rows * logits_width, "input_elements": logit_rows * logits_width,
                "output_elements": logit_rows * logits_width}},
        })
        segments[index]["last_op"] = logits
    if f64_norm_sum:
        norm = "rmsnorm_f32_square_f64_sum_rows_checked"
        for node in ops:
            if node["op"] == "rmsnorm_rows_checked":
                node.update(op=norm, kernel=norm)
        for export in graph["semantic_checkpoints"]["exports"].values():
            if export["op"] == "rmsnorm_rows_checked":
                export["op"] = norm
    if reference_moe:
        for node in ops:
            if node["kernel"] == "routed_softcapped_experts_checked_q8_0_q8_activation":
                node["kernel"] += "_scaled_situ"
    if fp32_static_conv:
        for node in ops:
            if node["kernel"] == "causal_depthwise_residual_checked_f32":
                node["kernel"] = "causal_depthwise_residual_checked_ordered_f32"
    public_ports = [
        "prefill_token_ids", "append_token_ids", "prefill_logits", "append_logits", "expert_slots",
        *(f"router_bias_{i}" for i in range(config["num_hidden_layers"])),
    ]
    graph["execution_exports"] = {
        "schema": "cke.checked.execution_exports.v1",
        "entries": {phase: f"ck_inkling2_loop_{phase}" for phase in ("prefill", "append")},
        "ports": {name: execution_port(graph, name) for name in public_ports},
        "properties": {"context_capacity": cache_capacity or (prefill_tokens + 1),
                       "chunk_tokens": prefill_tokens, "supports_explicit_position": False,
                       "prefill_logits": "last_row" if last_prefill_logit else "all_rows",
                       "expert_accumulation": "router_slot" if reference_moe else "expert_id",
                       "static_conv_accumulator": "fp32" if fp32_static_conv else "fp64",
                       "circular_sliding_slots": graph["contract"]["runtime_invariants"]["circular_sliding_slots"]},
    }
    return graph


def resident_text_loop(config, chunk_tokens, context_capacity, *, f64_norm_sum=False, last_prefill_logit=False,
                       reference_moe=False, fp32_static_conv=False, circular_sliding=False):
    """Generate repeated fixed-size prefill chunks and one-row decode over shared state."""
    graph = text_token_loop(config, chunk_tokens, cache_capacity=context_capacity, f64_norm_sum=f64_norm_sum,
                            last_prefill_logit=last_prefill_logit, reference_moe=reference_moe,
                            fp32_static_conv=fp32_static_conv, circular_sliding=circular_sliding)
    graph["name"] = "inkling2_mini_resident_text_loop"
    graph["native_entry"]["function"] = "ck_inkling2_mini_resident_text_loop"
    graph["contract"]["runtime_invariants"].update(context_capacity=context_capacity, chunk_tokens=chunk_tokens)
    exports = graph["execution_exports"]
    exports["properties"]["supports_explicit_position"] = True
    graph["runtime_constants"] = {
        "value_elements": 1, "phoneme_count": 1,
        "max_duration": context_capacity - 1, "expanded_capacity": context_capacity - 1,
    }
    graph["runtime_lengths"] = lengths = {}
    body = graph["block_types"]["block"]["body"]
    old_ops = iter(body["ops"])
    body["ops"] = ops = []
    positioned = {"kv_cache_l2_rms_append_checked", "gqa_relative_sink_checked", "log_position_scale_rows_checked"}
    for segment, (phase, rows) in zip(graph["native_entry"]["segments"], (("prefill", chunk_tokens), ("append", 1))):
        request_position = phase + "_request_position"
        position = phase + "_position"
        for name in (request_position, position):
            graph["activation_buffers"][name] = {"shape": [1]}
            graph["activation_bindings"][name] = name
        exports["ports"][request_position] = execution_port(graph, request_position)
        producer = "check_" + position
        limit = context_capacity - rows
        lengths[position] = {"producer": producer, "result": "valid_extent", "capacity": limit, "allow_zero": True}
        ops.append(runtime_extent(producer, "runtime_extent_sum", request_position, position, position, capacity=limit))
        for node in old_ops:
            if node["op"] in positioned:
                bind_runtime_scalars(node, position=position)
            ops.append(node)
            if node["id"] == segment["last_op"]:
                break
    return graph
