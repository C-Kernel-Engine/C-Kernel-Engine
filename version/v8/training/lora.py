"""Bounded FP32 LoRA authoring for the existing v8/shared training IR.

The graph owns the adapter; the existing providers own all forward/backward math.
This is not automatic model-wide adapter admission or quantized-base training.
"""

from copy import deepcopy
import math
import re


def attach_lora(ir, *, target, rank, alpha, name):
    """Return (IR, specification) for one unique, contiguous biasless projection.

    Weights use [output,input]. A=[rank,input], B=[output,rank]. Only the selected
    base weight is frozen; other trainability declarations are preserved.
    """
    if ir.get("format") != "ir1-train-v7":
        raise ValueError("expected authored training IR before backward generation")
    if type(name) is not str or not re.fullmatch("[A-Za-z][A-Za-z0-9_]{0,47}", name):
        raise ValueError("invalid adapter name")
    if type(target) is not str or not target:
        raise ValueError("explicit base parameter required")
    if (
        type(rank) is not int
        or rank <= 0
        or type(alpha) not in (int, float)
        or not math.isfinite(alpha)
        or alpha <= 0
    ):
        raise ValueError("rank and alpha must be positive finite values")
    if any(spec.get("target") == target for spec in ir.get("adapter_specs", [])):
        raise ValueError("target already has an adapter")
    result = deepcopy(ir)
    ops = result["ops"]
    tid = "weight." + target
    ids = [o.get("op_id") for o in ops]
    if any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("unique nonnegative operation identities required")
    matches = [
        o
        for o in ops
        if any(w.get("tensor") == tid for w in o.get("weights", {}).values())
    ]
    if len(matches) != 1:
        raise ValueError(
            "target must have one unambiguous projection use; tied weights are not admitted"
        )
    base = matches[0]
    if (
        base.get("kernel_id")
        not in ("gemm_blocked_serial", "linear_rows_checked_no_bias_f32")
        or len(base["weights"]) != 1
    ):
        raise ValueError("only biasless FP32 GEMM/checked projections are admitted")
    if (
        set(base["dataflow"]["inputs"]) != {"input"}
        or len(base["dataflow"]["outputs"]) != 1
    ):
        raise ValueError("expected one projection input and output")
    x = base["dataflow"]["inputs"]["input"]
    w = next(iter(base["weights"].values()))
    y = next(iter(base["dataflow"]["outputs"].values()))
    for ref in (x, w, y):
        meta = result["tensors"].get(ref["tensor"], {})
        if (
            ref.get("dtype") != "fp32"
            or meta.get("dtype") != "fp32"
            or len(ref.get("shape", [])) != 2
            or meta.get("shape") != ref["shape"]
        ):
            raise ValueError("explicit contiguous FP32 matrix references required")
        if (
            any(type(d) is not int or d <= 0 for d in ref["shape"])
            or ref.get("numel") != math.prod(ref["shape"])
            or meta.get("numel") != ref["numel"]
        ):
            raise ValueError("invalid bounded matrix extent")
    rows, inputs = x["shape"]
    outputs, wi = w["shape"]
    if wi != inputs or y["shape"] != [rows, outputs] or rank > min(inputs, outputs):
        raise ValueError("inconsistent geometry or rank outside projection bounds")
    if base.get("requires_grad") is not True or y.get("requires_grad") is not True:
        raise ValueError(
            "target must belong to an explicitly differentiable training graph"
        )
    # No implicit padded-stride reinterpretation of an existing checked provider.
    constants = base.get("params", {}).get("call_constants", {})
    for key, expected in (
        ("linear_input_stride", inputs),
        ("linear_weight_stride", inputs),
        ("linear_output_stride", outputs),
        ("linear_rows", rows),
        ("linear_input_channels", inputs),
        ("linear_output_channels", outputs),
        ("linear_input_elements", rows * inputs),
        ("linear_weight_elements", outputs * inputs),
        ("linear_output_elements", rows * outputs),
        ("T", rows),
        ("aligned_in", inputs),
        ("aligned_out", outputs),
    ):
        if key in constants and constants[key] != expected:
            raise ValueError(
                "projection constants must match contiguous bounded geometry"
            )
    for other in ops:
        if other is not base and any(
            r.get("tensor") == tid
            for r in other.get("dataflow", {}).get("inputs", {}).values()
        ):
            raise ValueError("shared base parameter inputs are not admitted")
    prefix = "lora." + name
    if any(
        k.startswith(
            ("weight." + prefix + ".", "act." + prefix + ".", "aux." + prefix + ".")
        )
        for k in result["tensors"]
    ):
        raise ValueError("adapter identity collision")

    def ref(tid, shape, kind="activation", trainable=True):
        meta = dict(
            shape=shape,
            numel=math.prod(shape),
            dtype="fp32",
            kind=kind,
            requires_grad=trainable,
            persistent=kind == "weight",
        )
        result["tensors"][tid] = dict(meta)
        return dict(
            meta,
            tensor=tid,
            **({"name": tid.removeprefix("weight.")} if kind == "weight" else {})
        )

    a = ref("weight." + prefix + ".a", [rank, inputs], "weight")
    b = ref("weight." + prefix + ".b", [outputs, rank], "weight")
    gain = ref("weight." + prefix + ".scale", [1], "weight", False)
    low = ref("act." + prefix + ".low", [rows, rank])
    # Flat contiguous projection output is legal under the checked map. This
    # exposes scale's existing vector contract without an implicit reshape.
    delta = ref("act." + prefix + ".delta", [rows * outputs])
    scaled = ref("act." + prefix + ".scaled", [rows * outputs])
    original = ref("act." + prefix + ".base", [rows, outputs])
    scratch = ref(
        "aux." + prefix + ".scale_scratch", [rows * outputs + 1], "aux", False
    )

    def freeze(value):
        if isinstance(value, dict):
            if value.get("tensor") == tid:
                value["requires_grad"] = False
            for v in value.values():
                freeze(v)
        elif isinstance(value, list):
            for v in value:
                freeze(v)

    freeze(result)
    result["tensors"][tid]["requires_grad"] = False
    old_kind = base["op"]
    base["op"] = "projection"
    base["requires_grad"] = True
    base["dataflow"]["outputs"] = {next(iter(base["dataflow"]["outputs"])): original}

    def operation(kind, kernel, ins, out, rule, c, weights=None, saved=None):
        return dict(
            op_id=-1,
            op=kind,
            kernel_id=kernel,
            phase="forward",
            section=base.get("section", "body"),
            layer=base.get("layer", 0),
            requires_grad=True,
            map_call=True,
            weights=weights or {},
            params=dict(call_constants=c),
            dataflow=dict(inputs=ins, outputs=dict(output=out)),
            grad_rule=rule,
            save_for_backward=saved or {},
        )

    def projection(label, source, weight, out, k, n):
        return operation(
            label,
            "linear_rows_checked_no_bias_f32",
            dict(input=source),
            out,
            "linear_rows_train_backward_nobias",
            dict(
                linear_input_elements=rows * k,
                linear_input_stride=k,
                linear_weight_elements=n * k,
                linear_weight_stride=k,
                linear_output_elements=rows * n,
                linear_output_stride=n,
                linear_rows=rows,
                linear_input_channels=k,
                linear_output_channels=n,
                T=rows,
                aligned_in=k,
                aligned_out=n,
                num_threads=1,
            ),
            dict(weight=weight),
            dict(input=source, weight=weight),
        )

    down = projection(prefix + ".a", x, a, low, inputs, rank)
    up = projection(prefix + ".b", low, b, delta, rank, outputs)
    count = rows * outputs
    scale = operation(
        prefix + ".scale",
        "scale_vector_checked_f32",
        dict(input=delta, gain=gain),
        scaled,
        "scale_vector_checked_backward",
        dict(
            input_elements=count,
            gain_elements=1,
            output_elements=count,
            scratch_elements=count + 1,
            rows=count,
            d_output_elements=count,
            d_input_elements=count,
            d_gain_elements=1,
            backward_scratch_elements=count + 1,
        ),
        saved=dict(input=delta, gain=gain),
    )
    scale["dataflow"]["outputs"]["scratch"] = scratch
    join = operation(
        old_kind,
        "add_rows_checked_f32",
        dict(left=original, right=scaled),
        y,
        "add_rows_checked_backward",
        dict(
            left_elements=count,
            right_elements=count,
            output_elements=count,
            elements=count,
            d_output_elements=count,
            d_left_elements=count,
            d_right_elements=count,
        ),
    )
    index = ops.index(base)
    ops[index + 1 : index + 1] = [down, up, scale, join]
    old_id = base["op_id"]
    mapping = {o["op_id"]: i for i, o in enumerate(ops) if o["op_id"] != -1}
    mapping[old_id] = index + 4
    for i, o in enumerate(ops):
        o["op_id"] = i
    for meta in result["tensors"].values():
        producer = meta.get("producer")
        if isinstance(producer, dict) and producer.get("op_id") in mapping:
            producer["op_id"] = mapping[producer["op_id"]]
    result["config"]["train_batch_policy"] = "fixed"
    spec = dict(
        schema="cke.lora.fp32.v1",
        name=name,
        target=target,
        rank=rank,
        alpha=float(alpha),
        scale=float(alpha) / rank,
        rows=rows,
        inputs=inputs,
        outputs=outputs,
        parameter_count=rank * (inputs + outputs),
        parameters=dict(a=a["name"], b=b["name"], scale=gain["name"]),
        base_frozen=True,
        dtype="fp32",
        initialization="PCG64 normal A std=1/sqrt(inputs); zero B; FP32 scale",
    )
    if (
        not math.isfinite(spec["scale"])
        or spec["scale"] > 3.402823466e38
        or spec["scale"] < 1.175494351e-38
    ):
        raise ValueError("scale outside finite FP32 range")
    result.setdefault("adapter_specs", []).append(deepcopy(spec))
    return result, spec


def _validate_spec(spec):
    """Validate metadata without generating or reading parameter values."""
    if spec.get("schema") != "cke.lora.fp32.v1":
        raise ValueError("unsupported adapter specification")
    for field in ("rows", "inputs", "outputs", "rank"):
        if type(spec.get(field)) is not int or spec[field] <= 0:
            raise ValueError("invalid adapter geometry")
    if spec["rank"] > min(spec["inputs"], spec["outputs"]) or spec.get(
        "parameter_count"
    ) != spec["rank"] * (spec["inputs"] + spec["outputs"]):
        raise ValueError("inconsistent adapter geometry")
    if type(spec.get("target")) is not str or not spec["target"]:
        raise ValueError("explicit base parameter required")
    name = spec.get("name")
    if type(name) is not str or not re.fullmatch("[A-Za-z][A-Za-z0-9_]{0,47}", name):
        raise ValueError("invalid adapter identity")
    expected = {
        key: "lora." + name + "." + suffix
        for key, suffix in (("a", "a"), ("b", "b"), ("scale", "scale"))
    }
    if (
        spec.get("parameters") != expected
        or spec.get("dtype") != "fp32"
        or spec.get("base_frozen") is not True
    ):
        raise ValueError("invalid adapter parameter contract")
    alpha = spec.get("alpha")
    scale = spec.get("scale")
    if (
        type(alpha) not in (int, float)
        or not math.isfinite(alpha)
        or type(scale) not in (int, float)
        or not math.isfinite(scale)
        or not 1.175494351e-38 <= scale <= 3.402823466e38
        or alpha / spec["rank"] != scale
    ):
        raise ValueError("invalid adapter scale")
    if (
        spec.get("initialization")
        != "PCG64 normal A std=1/sqrt(inputs); zero B; FP32 scale"
    ):
        raise ValueError("unsupported initialization policy")
    return spec["parameters"]


def initialize_lora(spec, *, seed):
    """Deterministic author-time initialization, never a training weight update."""
    import numpy as np

    names = _validate_spec(spec)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be a uint32 integer")
    rng = np.random.Generator(np.random.PCG64(seed))
    return {
        names["a"]: rng.normal(
            0, 1 / math.sqrt(spec["inputs"]), (spec["rank"], spec["inputs"])
        ).astype("float32"),
        names["b"]: np.zeros((spec["outputs"], spec["rank"]), dtype="float32"),
        names["scale"]: np.array([spec["scale"]], dtype="float32"),
    }


def attach_lora_adapters(ir, *, adapters, max_parameters=None):
    """Declare multiple bounded projections before backward synthesis.

    Each declaration has exactly target/rank/alpha/name. No target discovery,
    rank guessing, parameter initialization or compilation occurs here.
    """
    if not isinstance(adapters, (list, tuple)) or not adapters:
        raise ValueError("nonempty explicit adapter declarations required")
    if max_parameters is not None and (
        type(max_parameters) is not int or max_parameters <= 0
    ):
        raise ValueError("parameter budget must be a positive integer")
    targets = set()
    names = set()
    for declaration in adapters:
        if not isinstance(declaration, dict) or set(declaration) != {
            "target",
            "rank",
            "alpha",
            "name",
        }:
            raise ValueError("adapter requires target, rank, alpha and name")
        target, name = declaration["target"], declaration["name"]
        if (
            type(target) is not str
            or type(name) is not str
            or target in targets
            or name in names
        ):
            raise ValueError("adapter targets and names must be unique")
        targets.add(target)
        names.add(name)
    result = ir
    specs = []
    for declaration in adapters:
        result, spec = attach_lora(result, **declaration)
        specs.append(spec)
    total = sum(spec["parameter_count"] for spec in result["adapter_specs"])
    if max_parameters is not None and total > max_parameters:
        raise ValueError("resolved adapter parameters exceed budget")
    return result, specs
