"""Diagnostic-only native-to-PyTorch synchronization; never changes CKE state."""

import hashlib


def synchronize(torch, head, optimizer, weights, moments, parameter_order, step):
    """Restore packed QKV owners and AdamW state, proving byte-exact round trips."""
    import numpy as np

    def unpack(name, values):
        for suffix, leaf in (("in_proj_weight", "weight"), ("in_proj_bias", "bias")):
            if name.endswith(suffix):
                stem = name[: -len(suffix)]
                return np.concatenate(
                    [values[stem + axis + "_proj." + leaf] for axis in ("q", "k", "v")]
                )
        return values[name]

    groups = []
    offset = 0
    for _ in range(2):
        group = {}
        for name in parameter_order:
            shape = weights[name].shape
            count = weights[name].size
            group[name] = moments[offset : offset + count].reshape(shape)
            offset += count
        groups.append(group)
    if offset != moments.size:
        raise ValueError("optimizer snapshot extent mismatch")
    restored = []
    with torch.no_grad():
        for name, parameter in head.named_parameters():
            expected = unpack(name, weights).reshape(tuple(parameter.shape))
            parameter.copy_(torch.from_numpy(expected.copy()))
            np.testing.assert_array_equal(
                parameter.numpy().view("u4"), expected.view("u4")
            )
            state = optimizer.state[parameter]
            state["step"] = torch.tensor(float(step))
            for kind, group in zip(("exp_avg", "exp_avg_sq"), groups):
                expected_moment = unpack(name, group).reshape(tuple(parameter.shape))
                state[kind] = torch.from_numpy(expected_moment.copy())
                np.testing.assert_array_equal(
                    state[kind].numpy().view("u4"), expected_moment.view("u4")
                )
            restored.append(name)
    digest = hashlib.sha256()
    for name in sorted(weights):
        digest.update(name.encode())
        digest.update(weights[name].tobytes())
    digest.update(moments.tobytes())
    digest.update(str(step).encode())
    return dict(
        status="PASS",
        direction="native_to_oracle_only",
        optimizer_step=step,
        parameter_owners=len(restored),
        state_sha256=digest.hexdigest(),
    )
