"""Checked FP32 adapter weight preparation; no model arithmetic or optimizer.

The returned arrays are inputs to the existing generated training loader.
Checkpoint restoration of optimizer moments/counters remains its own contract.
"""

import hashlib
import json
import math

from .lora import _validate_spec, initialize_lora


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def resolve_lora_parameters(ir):
    """Resolve bounded parameter metadata independently of numerical values."""
    if ir.get("format") != "ir1-train-v7" or not ir.get("adapter_specs"):
        raise ValueError("authored training IR with explicit adapters required")
    schema = {}
    for tid, meta in ir["tensors"].items():
        if meta.get("kind") != "weight":
            continue
        shape = meta.get("shape")
        if (
            not tid.startswith("weight.")
            or not tid[7:]
            or meta.get("dtype") != "fp32"
            or not isinstance(shape, list)
            or not shape
            or any(type(n) is not int or n <= 0 for n in shape)
            or meta.get("numel") != math.prod(shape)
            or type(meta.get("requires_grad")) is not bool
            or meta.get("persistent") is not True
        ):
            raise ValueError("explicit contiguous FP32 parameter schema required")
        schema[tid[7:]] = dict(
            shape=list(shape),
            dtype="fp32",
            layout="row_major",
            trainable=meta["requires_grad"],
        )
    seen = set()
    for spec in ir["adapter_specs"]:
        names = _validate_spec(spec)
        if spec["name"] in seen:
            raise ValueError("duplicate adapter identity")
        seen.add(spec["name"])
        required = {
            spec["target"]: ([spec["outputs"], spec["inputs"]], False),
            names["a"]: ([spec["rank"], spec["inputs"]], True),
            names["b"]: ([spec["outputs"], spec["rank"]], True),
            names["scale"]: ([1], False),
        }
        for name, (shape, trainable) in required.items():
            if schema.get(name) != dict(
                shape=shape, dtype="fp32", layout="row_major", trainable=trainable
            ):
                raise ValueError("adapter and parameter schema disagree: " + name)
    return schema


def prepare_lora_weights(ir, weights, *, seeds=None, mode="initialize_missing"):
    """Validate supplied weights, initialize wholly absent adapters, report hashes.

    Existing complete adapters are preserved byte-for-byte. Partial adapters are
    rejected in either mode. Resume requires every parameter and accepts no seeds.
    No in-place changes to input arrays or the authored graph occur.
    """
    import numpy as np

    if mode not in ("initialize_missing", "resume"):
        raise ValueError("unsupported weight preparation mode")
    schema = resolve_lora_parameters(ir)
    seeds = {} if seeds is None else dict(seeds)
    if mode == "resume" and seeds:
        raise ValueError("resume must not request initialization")
    if set(weights) - set(schema):
        raise ValueError("weights outside resolved parameter schema")
    values = dict(weights)
    initialized = {}
    for spec in ir["adapter_specs"]:
        names = set(spec["parameters"].values())
        present = names & values.keys()
        if present and present != names:
            raise ValueError("partial adapter weights: " + spec["name"])
        if not present:
            if mode == "resume" or spec["name"] not in seeds:
                raise ValueError(
                    "missing adapter weights or explicit initialization seed"
                )
            seed = seeds.pop(spec["name"])
            generated = initialize_lora(spec, seed=seed)
            values.update(generated)
            for name in generated:
                initialized[name] = dict(
                    policy=spec["initialization"],
                    seed=seed,
                    implementation="cke.lora.initialize.v1",
                    numpy_version=np.__version__,
                )
    if seeds:
        raise ValueError(
            "unused initialization seed; existing adapters cannot be overwritten"
        )
    if set(values) != set(schema):
        raise ValueError("missing base parameters")
    records = {}
    for name, meta in schema.items():
        array = values[name]
        if (
            not isinstance(array, np.ndarray)
            or array.dtype != np.dtype("float32")
            or list(array.shape) != meta["shape"]
            or not array.flags.c_contiguous
            or not np.isfinite(array).all()
        ):
            raise ValueError("invalid parameter values/layout: " + name)
        records[name] = dict(
            **meta,
            sha256=hashlib.sha256(
                array.astype("<f4", copy=False).tobytes(order="C")
            ).hexdigest(),
            origin="initialized" if name in initialized else "supplied",
            initialization=initialized.get(name)
        )
    for spec in ir["adapter_specs"]:
        if values[spec["parameters"]["scale"]][0] != np.float32(spec["scale"]):
            raise ValueError("adapter scale disagrees with declaration")
    report = dict(
        schema="cke.lora.weight-preparation.v1",
        mode=mode,
        parameter_schema_sha256=_digest(schema),
        adapter_specs_sha256=_digest(ir["adapter_specs"]),
        parameters=records,
    )
    report["weights_sha256"] = _digest({n: r["sha256"] for n, r in records.items()})
    return values, report
