"""Shared circuit adapter, non-Clef graph, native gradients and AdamW."""

from copy import deepcopy
import json
from pathlib import Path
import unittest
import numpy as np
from version.v8.scripts.lower_circuit_training_v8 import (
    lower_circuit_training,
    training_registry,
)
from version.v8.decision.build_clef_training_ir import build_training_ir
from tests import test_v8_selective_training as backend
from tests.v8_training_test_helpers import certify_training

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    rng = np.random.default_rng(96)
    weights = {
        n: rng.normal(0, 0.3, (4, 4)).astype("f4")
        for n in ("features", "up", "bridge", "score")
    }
    weights["score"] = weights["score"][:1].copy()
    buffers = {n: dict(shape=[4, 4]) for n in ("embedded", "up", "bridge", "score")}
    buffers["score"]["shape"] = [4, 1]
    ops = [
        dict(
            id="lookup",
            kernel="audio_text_embedding_token_major_f32",
            params=dict(call_constants=dict(tokens=4, vocabulary=4, channels=4)),
            graph_slots=dict(inputs=dict(ids="ids"), outputs=dict(output="embedded")),
            weight_refs=dict(table="features"),
        )
    ]
    prev = "embedded"
    for name in ("up", "bridge", "score"):
        oc = 1 if name == "score" else 4
        ops.append(
            dict(
                id=name,
                kernel="linear_rows_checked_no_bias_f32",
                params=dict(
                    call_constants=dict(
                        linear_rows=4,
                        linear_input_channels=4,
                        linear_output_channels=oc,
                        linear_input_elements=16,
                        linear_input_stride=4,
                        linear_weight_elements=4 * oc,
                        linear_weight_stride=4,
                        linear_output_elements=4 * oc,
                        linear_output_stride=oc,
                    )
                ),
                graph_slots=dict(inputs=dict(input=prev), outputs=dict(output=name)),
                weight_refs=dict(weight=name),
            )
        )
        prev = name
    graph = dict(
        sequence=["component"],
        block_types=dict(component=dict(body=dict(ops=ops))),
        activation_buffers=buffers,
    )
    inputs = dict(
        ids=dict(shape=[4], dtype="i32"),
        counts=dict(shape=[2], dtype="i32"),
        labels=dict(shape=[2], dtype="i32"),
    )
    loss = dict(
        kernel_id="grouped_cross_entropy_checked_f32",
        inputs=dict(logits="act.score", counts="input.counts", targets="input.labels"),
        params=dict(
            call_constants=dict(
                logits_elements=4,
                count_elements=2,
                target_elements=2,
                gradient_elements=4,
                loss_elements=1,
                scratch_elements=5,
                groups=2,
                options=4,
            )
        ),
        scratch=dict(scratch=5),
    )
    args = dict(
        inputs=inputs,
        bindings={n: "input." + n for n in inputs},
        frozen={"features", "bridge"},
        config=dict(
            hidden_size=4,
            intermediate_size=4,
            num_heads=1,
            num_kv_heads=1,
            head_dim=4,
            vocab_size=4,
            train_tokens=4,
            train_batch_policy="fixed",
        ),
        loss=loss,
    )
    return graph, weights, args


def broadcast_fixture():
    graph, weights, args = fixture()
    weights["offsets"] = np.array([[0.2, -0.1, 0.3, -0.2]], dtype="f4")
    args["bindings"]["row_shift"] = "weight.offsets"
    graph["activation_buffers"].update(
        shifted=dict(shape=[4, 4]), nonlinear=dict(shape=[4, 4])
    )
    operations = graph["block_types"]["component"]["body"]["ops"]
    operations[2]["graph_slots"]["inputs"]["input"] = "nonlinear"
    operations[2:2] = [
        dict(
            id="unrelated_row_adjustment",
            kernel="broadcast_add_rows_checked_f32",
            graph_slots=dict(
                inputs=dict(left="up", right="row_shift"),
                outputs=dict(output="shifted"),
            ),
            params=dict(call_constants={}),
        ),
        dict(
            id="nonlinear_shift",
            kernel="gelu_rows_erf_checked_f32",
            graph_slots=dict(
                inputs=dict(input="shifted"), outputs=dict(output="nonlinear")
            ),
            params=dict(
                call_constants=dict(
                    gelu_input_elements=16,
                    gelu_input_stride=4,
                    gelu_output_elements=16,
                    gelu_output_stride=4,
                    gelu_scratch_elements=16,
                    gelu_rows=4,
                    gelu_channels=4,
                )
            ),
        ),
    ]
    return graph, weights, args


class CircuitTrainingTests(unittest.TestCase):
    def test_provider_prefix_does_not_establish_training_support(self):
        graph, weights, args = fixture()
        for block in graph["block_types"].values():
            for op in block.get("body", {}).get("ops", []):
                if op["kernel"] == "linear_rows_checked_no_bias_f32":
                    op["kernel"] = "linear_rows_checked_unregistered_variant"
        with self.assertRaisesRegex(ValueError, "provider"):
            lower_circuit_training(graph, weights, **args)

    def test_direct_binding_rejects_undeclared_output_port(self):
        graph, weights, args = fixture()
        for block in graph["block_types"].values():
            for op in block.get("body", {}).get("ops", []):
                if op["kernel"] == "linear_rows_checked_no_bias_f32":
                    outputs = op["graph_slots"]["outputs"]
                    outputs["unexpected"] = outputs.pop("output")
        with self.assertRaisesRegex(ValueError, "output ports"):
            lower_circuit_training(graph, weights, **args)

    def test_native_non_model_specific_composition(self):
        try:
            import torch
        except ImportError as exc:
            self.skipTest("independent PyTorch comparison unavailable: " + str(exc))
        graph, weights, args = fixture()
        ir = lower_circuit_training(graph, weights, **args)
        registry = training_registry()
        rules = json.loads((ROOT / "version/v8/scripts/grad_rules_v8.json").read_text())
        lowered = backend.backward.synthesize_ir2_backward(
            ir,
            registry,
            backend.BINDINGS,
            rules,
            strict=True,
            allow_partial=False,
            checkpoint_policy="none",
        )
        from version.v8.scripts.generated_composition_contract_v8 import (
            audit_generated_composition,
        )

        layout = backend.planner.build_layout(
            lowered, {"config": lowered["config"]}, 64, strict=True
        )
        source, summary = backend.codegen.generate_c(
            lowered, registry, manifest={"config": lowered["config"]}, layout=layout
        )
        evidence = audit_generated_composition(
            ir,
            [dict(tensor="act.bridge", producer="bridge", consumer="score")],
            "act.score",
            summary,
        )
        self.assertEqual(evidence["structural"], "PASS")
        self.assertIn("ck_train_step", source)
        self.assertIn("ck_train_forward_step", source)
        forward = lambda p: p["features"] @ p["up"].T @ p["bridge"].T @ p["score"].T
        # Runtime packs integer inputs in sorted name order: counts, ids, labels.
        ints = np.concatenate(
            (
                np.full(2, 2, dtype="i4"),
                np.arange(4, dtype="i4"),
                np.arange(2, dtype="i4"),
            )
        )
        certify_training(
            self,
            lowered,
            registry,
            weights,
            {"up", "score"},
            forward,
            integer_inputs=ints,
            loss_function=lambda z: torch.nn.functional.cross_entropy(
                z.reshape(2, 2), torch.arange(2)
            ),
            extra_sources=("src/kernels/linear_rows_checked.c",),
            check_evaluation=True,
        )

    def test_broadcast_uses_compiler_constants_with_independent_gradient_oracle(self):
        try:
            import torch
        except ImportError as exc:
            self.skipTest("independent PyTorch unavailable: " + str(exc))
        graph, weights, args = broadcast_fixture()
        ir = lower_circuit_training(graph, weights, **args)
        constants = ir["config"]["training_integer_constants"]
        self.assertEqual(list(constants.values()), [[0, 0, 0, 0]])
        self.assertNotIn("input.broadcast_zeros", ir["tensors"])
        registry = training_registry()
        rules = json.loads((ROOT / "version/v8/scripts/grad_rules_v8.json").read_text())
        lowered = backend.backward.synthesize_ir2_backward(
            ir,
            registry,
            backend.BINDINGS,
            rules,
            strict=True,
            allow_partial=False,
            checkpoint_policy="none",
        )
        forward = (
            lambda p: torch.nn.functional.gelu(p["features"] @ p["up"].T + p["offsets"])
            @ p["bridge"].T
            @ p["score"].T
        )
        ints = np.concatenate(
            (
                np.full(2, 2, dtype="i4"),
                np.arange(4, dtype="i4"),
                np.arange(2, dtype="i4"),
            )
        )
        certify_training(
            self,
            lowered,
            registry,
            weights,
            {"up", "score", "offsets"},
            forward,
            integer_inputs=ints,
            loss_function=lambda z: torch.nn.functional.cross_entropy(
                z.reshape(2, 2), torch.arange(2)
            ),
            extra_sources=("src/kernels/linear_rows_checked.c",),
            check_evaluation=True,
        )
        layout = backend.planner.build_layout(
            lowered, {"config": lowered["config"]}, 64, strict=True
        )
        for defect in ("length", "value", "trainable", "write"):
            changed = deepcopy(lowered)
            name = next(iter(constants))
            if defect == "length":
                changed["config"]["training_integer_constants"][name] = [0]
            if defect == "value":
                changed["config"]["training_integer_constants"][name][0] = 2**32
            if defect == "trainable":
                changed["tensors"][name]["requires_grad"] = True
            if defect == "write":
                next(iter(changed["forward"][0]["dataflow"]["outputs"].values()))[
                    "tensor"
                ] = name
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                backend.codegen.generate_c(
                    changed,
                    registry,
                    manifest={"config": changed["config"]},
                    layout=layout,
                )

    def test_unsupported_operation_fails_before_codegen(self):
        graph, weights, args = fixture()
        graph["block_types"]["component"]["body"]["ops"][1]["kernel"] = "unknown"
        with self.assertRaisesRegex(
            ValueError, "unsupported circuit training provider"
        ):
            lower_circuit_training(graph, weights, **args)

    def test_vjp_semantics_do_not_depend_on_op_name(self):
        from tests.test_v8_clef_flash_complete_head import synthetic_weights

        ir, weights, graph = build_training_ir(
            synthetic_weights(),
            np.ones((6, 8), dtype="f4"),
            questions=2,
            options=3,
            heads=2,
            feedforward=16,
        )
        prior = next(
            op
            for b in graph["block_types"].values()
            for op in b.get("body", {}).get("ops", [])
            if op["id"] == "prior_cosine"
        )
        prior["id"] = "arbitrary_name"
        inputs = {
            n.removeprefix("input."): dict(shape=t["shape"], dtype=t["dtype"])
            for n, t in ir["tensors"].items()
            if t["kind"] == "input"
        }
        bindings = {n: "input." + n for n in inputs}
        result = lower_circuit_training(
            graph,
            weights,
            inputs=inputs,
            bindings=bindings,
            frozen={"output_embedding.weight"},
            config=ir["config"],
            activation_dtypes={
                n: "i32" for n in ("option_row_indices", "option_question_indices")
            },
            loss=ir["loss"],
        )
        self.assertEqual(
            next(op for op in result["ops"] if op["op"] == "arbitrary_name")[
                "grad_rule"
            ],
            "indexed_normalized_dot_checked_backward",
        )
        prior.pop("training")
        with self.assertRaisesRegex(ValueError, "explicit training VJP"):
            lower_circuit_training(
                graph,
                weights,
                inputs=inputs,
                bindings=bindings,
                frozen={"output_embedding.weight"},
                config=ir["config"],
                loss=ir["loss"],
            )
