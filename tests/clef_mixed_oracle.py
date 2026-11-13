"""Test-only PyTorch policy adapter; pinned reference still owns head topology."""


def install(test, torch, head, *, fp32_weights=()):
    from unittest.mock import patch

    original_linear = torch.nn.functional.linear
    captures = {}
    owners = {
        p.untyped_storage().data_ptr(): (name, p) for name, p in head.named_parameters()
    }

    class Linear(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w, bias):
            xr = x.to(torch.bfloat16).float()
            wr = w.to(torch.bfloat16).float()
            ctx.save_for_backward(xr, wr)
            ctx.has_bias = bias is not None
            y = xr @ wr.T
            return y if bias is None else y + bias

        @staticmethod
        def backward(ctx, dy):
            x, w = ctx.saved_tensors
            d = dy.reshape(-1, w.shape[0])
            xx = x.reshape(-1, w.shape[1])
            return dy @ w, d.T @ xx, d.sum(0) if ctx.has_bias else None

    def linear(x, w, bias=None):
        name, param = owners[w.untyped_storage().data_ptr()]
        if name in fp32_weights:
            return original_linear(x, w, bias)
        y = Linear.apply(x, w, bias)
        parts = []
        if name.endswith("in_proj_weight"):
            width = w.shape[1]
            start = (w.storage_offset() - param.storage_offset()) // (width * width)
            parts = [
                (
                    name.removesuffix("in_proj_weight") + axis + "_proj.weight",
                    i * width,
                    (i + 1) * width,
                )
                for i, axis in enumerate(
                    ("q", "k", "v")[start : start + w.shape[0] // width]
                )
            ]
        else:
            parts = [(name, 0, w.shape[0])]
        for key, start, end in parts:
            entry = dict(
                input=x.detach().reshape(-1, w.shape[1]).numpy().copy(),
                weight=w[start:end].detach().numpy().copy(),
                output=y[..., start:end]
                .detach()
                .reshape(-1, end - start)
                .numpy()
                .copy(),
                bias=None if bias is None else bias[start:end].detach().numpy().copy(),
            )
            captures.setdefault(key, []).append(entry)

            def save_grad(g, e=entry, a=start, b=end):
                e["d_output"] = g[..., a:b].detach().reshape(-1, b - a).numpy().copy()

            y.register_hook(save_grad)
        return y

    patcher = patch.object(torch.nn.functional, "linear", linear)
    patcher.start()
    test.addCleanup(patcher.stop)
    return captures


def diagnose(
    ir,
    captures,
    activations,
    weights,
    parameter_gradients,
    layout,
    accumulation,
    library,
):
    """Replay projections with actual CKE operands; never alter production inputs."""
    import numpy as np
    import torch

    rounded = lambda a: torch.tensor(a).bfloat16().float()
    regions = {r["name"]: r for r in layout["regions"]}
    tensors = {t["id"]: t for t in layout["tensors"]}

    def gradient(tid):
        t = tensors["grad.act." + tid]
        offset = (
            regions["grads"]["bytes"]
            + t["offset"]
            - regions["grad_activations"]["offset"]
        ) // 4
        return accumulation[offset : offset + t["numel"]].reshape(t["shape"])

    comparisons = []
    packet = None
    for op in ir["ops"]:
        if not op["kernel_id"].startswith("projection_bf16_train"):
            continue
        name = op["weights"]["weight"]["name"]
        refs = captures[name]
        reference = np.concatenate([r["output"] for r in refs])
        output_id = op["dataflow"]["outputs"]["output"]["tensor"]
        actual = activations[output_id].reshape(reference.shape)
        error = float(np.max(np.abs(actual - reference)))
        if error <= 3e-5:
            continue
        rx = np.concatenate([r["input"] for r in refs])
        x = activations[op["dataflow"]["inputs"]["input"]["tensor"]].reshape(rx.shape)
        w, rw = weights[name], refs[0]["weight"]
        replay = (rounded(x) @ rounded(w).T).numpy()
        bias = (
            weights[op["weights"]["bias"]["name"]] if "bias" in op["weights"] else None
        )
        if bias is not None:
            replay += bias.reshape(-1)
        item = dict(
            op=op["op"],
            weight=name,
            output_error=error,
            input_error=float(np.max(np.abs(x - rx))),
            weight_error=float(np.max(np.abs(w - rw))),
            rounded_input_difference=int(
                torch.count_nonzero(rounded(x) != rounded(rx))
            ),
            rounded_weight_difference=int(
                torch.count_nonzero(rounded(w) != rounded(rw))
            ),
            same_input_output_error=float(np.max(np.abs(replay - actual))),
        )
        if packet is None:
            dy = gradient(output_id).reshape(actual.shape)
            tx, tw = rounded(x).requires_grad_(), rounded(w).requires_grad_()
            ty = tx @ tw.T
            tb = None
            if bias is not None:
                tb = torch.tensor(bias.reshape(-1), requires_grad=True)
                ty = ty + tb
            ty.backward(torch.tensor(dy))
            # The generated input gradient may sum multiple branches. Replay the
            # selected provider into separate buffers instead of comparing that sum.
            import ctypes as C

            sx = torch.tensor(x).bfloat16().view(torch.int16).numpy().copy()
            sw = torch.tensor(w).bfloat16().view(torch.int16).numpy().copy()
            actual_dx = np.empty_like(x)
            local_dw = np.empty_like(w)
            arrays = [dy, sx, sw, actual_dx, local_dw]
            if bias is not None:
                local_db = np.empty(w.shape[0], dtype="f4")
                arrays.append(local_db)
            scratch = np.empty(
                x.size + w.size + (w.shape[0] if bias is not None else 0), dtype="f4"
            )
            arrays.append(scratch)
            fn = getattr(
                library,
                "ck_projection_bf16_train_"
                + ("bias_" if bias is not None else "")
                + "backward",
            )
            fn.argtypes = [t for _ in arrays for t in (C.c_void_p, C.c_size_t)] + [
                C.c_size_t
            ] * 3
            fn.restype = C.c_int
            args = [v for array in arrays for v in (array.ctypes.data, array.size)]
            if fn(*args, x.shape[0], x.shape[1], w.shape[0]) != 0:
                raise RuntimeError(
                    "local projection backward replay rejected captured operands"
                )
            item["same_input_d_input_error"] = float(
                np.max(np.abs(actual_dx - tx.grad.numpy()))
            )
            item["d_input_evidence"] = (
                "isolated provider replay, not accumulated generated gradient"
            )
            actual_dw = parameter_gradients[name].reshape(w.shape)
            item["same_input_d_weight_error"] = float(
                np.max(np.abs(actual_dw - tw.grad.numpy()))
            )
            if tb is not None:
                actual_db = parameter_gradients[op["weights"]["bias"]["name"]].reshape(
                    -1
                )
                item["same_input_d_bias_error"] = float(
                    np.max(np.abs(actual_db - tb.grad.numpy()))
                )
            input_changes = np.argwhere(rounded(x).numpy() != rounded(rx).numpy())
            if len(input_changes):
                ix = tuple(input_changes[0])
                item["first_rounded_input_difference"] = dict(
                    index=list(map(int, ix)),
                    actual=float(x[ix]),
                    reference=float(rx[ix]),
                    rounded_actual=float(rounded(x).numpy()[ix]),
                    rounded_reference=float(rounded(rx).numpy()[ix]),
                )
            changed = np.argwhere(rounded(w).numpy() != rounded(rw).numpy())
            if len(changed):
                index = tuple(changed[0])
                item["first_rounded_weight_difference"] = dict(
                    index=list(map(int, index)),
                    actual=float(w[index]),
                    reference=float(rw[index]),
                    rounded_actual=float(rounded(w).numpy()[index]),
                    rounded_reference=float(rounded(rw).numpy()[index]),
                )
            packet = dict(
                input=x,
                reference_input=rx,
                weight=w,
                reference_weight=rw,
                output=actual,
                reference_output=reference,
                d_output=dy,
                d_weight=actual_dw,
                oracle_d_weight=tw.grad.numpy(),
                d_input=actual_dx,
                oracle_d_input=tx.grad.numpy(),
            )
            if bias is not None:
                packet.update(
                    bias=bias, d_bias=actual_db, oracle_d_bias=tb.grad.numpy()
                )
        if packet is not None and "upstream_layernorm" not in item:
            input_id = op["dataflow"]["inputs"]["input"]["tensor"]
            producers = [
                p
                for p in ir["ops"]
                if p["kernel_id"] == "layernorm_train_forward_f32"
                and any(
                    r.get("tensor") == input_id
                    for r in p["dataflow"]["outputs"].values()
                )
            ]
            if producers:
                producer = producers[0]
                source = producer["dataflow"]["inputs"]["input"]["tensor"]
                operand = (
                    weights[source.removeprefix("weight.")]
                    if source.startswith("weight.")
                    else activations[source]
                )
                operand = operand.reshape(
                    producer["dataflow"]["inputs"]["input"]["shape"]
                )
                gamma = weights[producer["weights"]["gamma"]["name"]]
                beta = weights[producer["weights"]["beta"]["name"]]
                eps = producer["params"]["epsilon"]

                def norm(dtype):
                    return (
                        torch.nn.functional.layer_norm(
                            torch.tensor(operand, dtype=dtype),
                            [operand.shape[-1]],
                            torch.tensor(gamma, dtype=dtype),
                            torch.tensor(beta, dtype=dtype),
                            eps,
                        )
                        .float()
                        .numpy()
                    )

                fp32, fp64 = norm(torch.float32), norm(torch.float64)
                item["upstream_layernorm"] = dict(
                    op=producer["op"],
                    same_input_torch_to_captured_reference=float(
                        np.max(np.abs(fp32 - rx))
                    ),
                    native_to_fp64=float(np.max(np.abs(x - fp64))),
                    torch_to_fp64=float(np.max(np.abs(fp32 - fp64))),
                )
                if not comparisons:
                    packet.update(
                        layernorm_input=operand,
                        layernorm_gamma=gamma,
                        layernorm_beta=beta,
                        layernorm_epsilon=np.array(eps),
                    )
        comparisons.append(item)
    return comparisons, packet


def projection_policy(ir, *, fp32_memory_projection=False):
    """Bounded experiment authoring, not a compiler or public training policy."""
    from version.v8.training.bf16_projection import use_bf16_projection

    memory = [op for op in ir["ops"] if op["op"] == "memory_projection"]
    if fp32_memory_projection and (
        len(memory) != 1
        or memory[0]["kernel_id"] != "linear_rows_checked_no_bias_f32"
        or memory[0]["weights"]["weight"]["name"] != "memory_projection.weight"
    ):
        raise ValueError("expected one declared FP32 memory projection")
    for op in list(ir["ops"]):
        if op["kernel_id"] in (
            "linear_rows_checked_f32",
            "linear_rows_checked_no_bias_f32",
        ):
            if fp32_memory_projection and op["op"] == "memory_projection":
                continue
            ir = use_bf16_projection(ir, op["op_id"])
    return ir
