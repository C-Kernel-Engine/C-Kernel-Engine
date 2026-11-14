"""Bounded FP32 head-only training authoring from the canonical Clef circuit.

This is an experimental fixed-feature training graph, not a public job recipe.
No forward/backward arithmetic or layer schedule is executed in Python.
"""

import numpy as np
from .package_clef_head import build_circuit, effective_weights
from version.v8.scripts.lower_circuit_training_v8 import (
    lower_circuit_training,
    training_registry,
)


def build_training_ir(
    raw, hidden, *, questions, options, heads, feedforward, frozen=()
):
    if (
        not isinstance(hidden, np.ndarray)
        or hidden.ndim != 2
        or hidden.dtype != np.float32
    ):
        raise ValueError("frozen hidden features must be a rank-two FP32 array")
    if not np.isfinite(hidden).all():
        raise ValueError("frozen hidden features must be finite")
    for name, value in (
        ("questions", questions),
        ("options", options),
        ("heads", heads),
        ("feedforward", feedforward),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(name + " must be a positive integer")
    for name, value in raw.items():
        if not isinstance(value, np.ndarray) or value.dtype != np.float32:
            raise ValueError(
                "FP32 head training requires explicit FP32 weight arrays: " + name
            )
    tokens, channels = hidden.shape
    width = raw["question_projection.weight"].shape[0]
    graph = build_circuit(
        tokens=tokens,
        questions=questions,
        options=options,
        channels=channels,
        width=width,
        vocabulary=len(raw["output_embedding.weight"]),
        heads=heads,
        feedforward=feedforward,
    )
    weights = effective_weights(graph, raw)
    frozen = set(frozen) | {"output_embedding.weight"}
    if frozen - set(weights):
        raise ValueError("unknown frozen parameter")
    loss = dict(
        kernel_id="grouped_cross_entropy_checked_f32",
        inputs=dict(
            logits="act.option_logits",
            counts="input.option_counts",
            targets="input.labels",
        ),
        params=dict(
            call_constants=dict(
                logits_elements=options,
                count_elements=questions,
                target_elements=questions,
                gradient_elements=options,
                loss_elements=1,
                scratch_elements=options + 1,
                groups=questions,
                options=options,
            )
        ),
        scratch=dict(scratch=options + 1),
    )
    integers = {
        "input_ids",
        "question_spans",
        "option_spans",
        "global_span",
        "option_counts",
        "question_type_ids",
    }
    inputs = {
        n: dict(shape=graph["activation_buffers"][n]["shape"], dtype="i32")
        for n in integers
    }
    inputs.update(
        labels=dict(shape=[questions], dtype="i32"),
    )
    inputs["hidden"] = dict(shape=list(hidden.shape), dtype="fp32")
    bindings = {n: "input." + n for n in inputs}
    ir = lower_circuit_training(
        graph,
        weights,
        inputs=inputs,
        bindings=bindings,
        frozen=frozen,
        config=dict(
            hidden_size=width,
            intermediate_size=feedforward,
            num_heads=heads,
            num_kv_heads=heads,
            head_dim=width // heads,
            vocab_size=len(raw["output_embedding.weight"]),
            train_tokens=tokens,
            train_batch_policy="fixed",
        ),
        loss=loss,
        activation_dtypes={
            n: "i32" for n in ("option_row_indices", "option_question_indices")
        },
    )
    return ir, weights, graph
