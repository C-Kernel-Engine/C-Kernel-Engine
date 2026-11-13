"""Opt-in fixed-capacity projection precision authoring for the shared trainer.

FP32 masters and activation gradients; RNE BF16 saved compute operands; FP32
output and ordered accumulation. This is not a global BF16/autocast switch.
"""

from copy import deepcopy


def use_bf16_projection(ir, op_id):
    """Return IR with one checked projection using the declared mixed VJP."""
    result = deepcopy(ir)
    matches = [o for o in result["ops"] if o["op_id"] == op_id]
    if len(matches) != 1:
        raise ValueError("expected one projection operation")
    op = matches[0]
    if op.get("kernel_id") not in (
        "gemm_blocked_serial",
        "linear_rows_checked_f32",
        "linear_rows_checked_no_bias_f32",
    ):
        raise ValueError("only explicit FP32 GEMM/checked projections are supported")
    weight_key = "W" if "W" in op["weights"] else "weight"
    if set(op["weights"]) not in ({weight_key}, {weight_key, "bias"}):
        raise ValueError("unexpected projection weight ports")
    bias = op["weights"].get("bias")
    x = op["dataflow"]["inputs"]["input"]
    w = op["weights"][weight_key]
    outputs = op["dataflow"]["outputs"]
    output_key = "y" if "y" in outputs else "output"
    if set(outputs) != {output_key}:
        raise ValueError("expected one projection output")
    y = outputs[output_key]
    if any(ref.get("dtype") != "fp32" for ref in (x, w, y)):
        raise ValueError("projection masters and external activations must be FP32")
    if any(len(ref.get("shape", [])) != 2 for ref in (x, w, y)):
        raise ValueError("expected bounded row-major matrices")
    rows, inputs = x["shape"]
    out, wi = w["shape"]
    if (
        any(type(n) is not int or n <= 0 for n in (rows, inputs, out))
        or wi != inputs
        or y["shape"] != [rows, out]
    ):
        raise ValueError("inconsistent projection geometry")
    if bias is not None and (
        bias.get("dtype") != "fp32" or bias.get("shape") not in ([out], [1, out])
    ):
        raise ValueError("bias must be an explicit FP32 output-channel vector")

    def aux(name, shape, dtype):
        tid = "aux.projection_%s.%s" % (op_id, name)
        if tid in result["tensors"]:
            raise ValueError("projection auxiliary identity collision")
        n = 1
        for d in shape:
            n *= d
        meta = dict(
            shape=shape,
            numel=n,
            dtype=dtype,
            kind="aux",
            requires_grad=False,
            persistent=False,
        )
        result["tensors"][tid] = meta
        return dict(meta, tensor=tid)

    sx = aux("saved_input", [rows, inputs], "bf16")
    sw = aux("saved_weight", [out, inputs], "bf16")
    scratch = aux("scratch", [rows * out], "fp32")
    op.update(
        kernel_id="projection_bf16_train_forward",
        map_call=True,
        grad_rule="projection_bf16_train_backward",
        weights={"weight": w},
        dataflow=dict(
            inputs={"input": x},
            outputs=dict(output=y, saved_input=sx, saved_weight=sw, scratch=scratch),
        ),
        save_for_backward=dict(saved_input=sx, saved_weight=sw),
        params=dict(
            call_constants=dict(
                rows=rows,
                inputs=inputs,
                outputs=out,
                input_elements=rows * inputs,
                weight_elements=out * inputs,
                output_elements=rows * out,
                saved_input_elements=rows * inputs,
                saved_weight_elements=out * inputs,
                d_output_elements=rows * out,
                d_input_elements=rows * inputs,
                d_weight_elements=out * inputs,
                scratch_elements=rows * out,
                backward_scratch_elements=(rows + out) * inputs,
            )
        ),
    )
    if bias is not None:
        op["kernel_id"] = "projection_bf16_train_bias_forward"
        op["grad_rule"] = "projection_bf16_train_bias_backward"
        op["weights"]["bias"] = bias
        constants = op["params"]["call_constants"]
        constants.update(bias_elements=out, d_bias_elements=out)
        constants["backward_scratch_elements"] += out
    result["config"]["train_batch_policy"] = "fixed"
    return result
