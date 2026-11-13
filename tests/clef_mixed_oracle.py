"""Test-only PyTorch policy adapter; pinned reference still owns head topology."""


def install(test, torch, head):
    from unittest.mock import patch

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
        y = Linear.apply(x, w, bias)
        name, param = owners[w.untyped_storage().data_ptr()]
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
    ir, captures, activations, weights, parameter_gradients, layout, accumulation
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
            actual_dx = gradient(op["dataflow"]["inputs"]["input"]["tensor"]).reshape(
                x.shape
            )
            item["same_input_d_input_error"] = float(
                np.max(np.abs(actual_dx - tx.grad.numpy()))
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
        comparisons.append(item)
    return comparisons, packet
