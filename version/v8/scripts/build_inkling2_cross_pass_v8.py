"""Author cross-pass message attention and its checked controller graph."""
from version.v8.scripts.build_inkling2_mini_projection_circuit import _graph, _project

CONTROL_PORTS = ("q", "k", "v", "ffn")


def append_cross_pass_controls(ops, shapes, *, tokens, width, rank, states, layer,
                               bank, previous, attention_input, ffn_input,
                               ports=CONTROL_PORTS, prefix="", epsilon=1e-6):
    """Append the selected controls to a generated block and return their outputs."""
    area = f"blk.{layer}."
    count = tokens * width

    def add(suffix, op, output_width, inputs, params, weights=None, kernel=None):
        node_id = prefix + suffix
        shapes[node_id] = [tokens, output_width]
        node = {
            "id": node_id, "op": op, "kernel": kernel or op + "_f32", "returns_status": True,
            "graph_slots": {"inputs": inputs, "outputs": {"output": node_id}},
            "params": params,
        }
        if weights:
            node["weight_refs"] = weights
        ops.append(node)
        return node_id

    result = {}
    for port in ports:
        # Queries are one four-row GGUF tensor. The bounded loader holds four named row slices.
        message = add(f"{port}_message", "depth_residual_mix_checked", width,
                      {"bank": bank, "current": previous},
                      {"R": tokens, "S": states, "C": width, "epsilon": epsilon, "call_constants": {
                          "rows": tokens, "states": states, "width": width,
                          "bank_elements": count * states, "current_elements": count,
                          "query_elements": width, "output_elements": count,
                      }}, weights={"query": area + f"loop_msg_query.{port}"})
        source = ffn_input if port == "ffn" else attention_input
        joined = add(f"{port}_join", "rms_pair_concat_rows_checked", 2 * width,
                     {"left": source, "right": message},
                     {"R": tokens, "C": width, "J": width * 2, "epsilon": epsilon,
                      "call_constants": {
                          "rows": tokens, "width": width, "joined_width": width * 2,
                          "left_elements": count, "right_elements": count, "output_elements": count * 2,
                      }})
        down = prefix + f"{port}_down"
        shapes[down] = [tokens, rank]
        ops.append(_project(down, joined, width * 2, rank, tokens,
                            f"loop_{port}_ctrl_down.weight", weight_prefix=area))
        gates = {}
        for gate in ("forget", "gain"):
            logits = prefix + f"{port}_{gate}_logits"
            shapes[logits] = [tokens, width]
            ops.append(_project(logits, down, rank, width, tokens,
                                f"loop_{port}_ctrl_{gate}.weight", weight_prefix=area))
            gates[gate] = add(f"{port}_{gate}", "tanh_strided_checked", width, {"input": logits},
                              {"R": tokens, "C": width, "call_constants": {
                                  "tanh_input_elements": count, "tanh_input_stride": width,
                                  "tanh_output_elements": count, "tanh_output_stride": width,
                                  "tanh_rows": tokens, "tanh_columns": width,
                              }}, kernel="tanh_strided_f32_checked")
        result[port] = add(f"{port}_controlled", "message_control_rows_checked", width,
                           {"input": source, "message": message, **gates},
                           {"N": count, "call_constants": {
                               "elements": count, "input_elements": count, "message_elements": count,
                               "forget_elements": count, "gain_elements": count, "output_elements": count,
                           }})
    return result


def cross_pass_controls(tokens=4, width=2304, rank=64, states=1, layer=6):
    """Run four controllers using one physical layer's shared GGUF weights."""
    if type(layer) is not int or not 6 <= layer < 18 or type(states) is not int or not 1 <= states <= 64:
        raise ValueError("expected a core layer and a nonempty prior-pass depth bank")
    shapes = {"prior_bank": [tokens, states, width], "prior_output": [tokens, width],
              "attention_input": [tokens, width], "ffn_input": [tokens, width]}
    ops = []
    append_cross_pass_controls(ops, shapes, tokens=tokens, width=width, rank=rank, states=states,
                               layer=layer, bank="external:prior_bank", previous="external:prior_output",
                               attention_input="external:attention_input", ffn_input="external:ffn_input")
    return _graph("cross_pass_control", tokens, width, shapes, ops, layer=layer,
                  scope="one physical core layer; generated messages and controller outputs from supplied previous pass")


def control_weight_shapes(graph):
    weights = {}
    for op in graph["block_types"]["block"]["body"]["ops"]:
        for port, name in op.get("weight_refs", {}).items():
            params = op["params"]
            weights[name] = [params["C"]] if port == "query" else [params["N"], params["K"]]
    return weights
