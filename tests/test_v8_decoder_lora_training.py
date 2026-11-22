"""Generated decoder LoRA with a development-only PyTorch oracle.

CKE_RUN_SCALED_LORA=1 enables the 160-update depth tests.
CKE_RUN_LORA_WIDE_STRESS=1 additionally probes width 512.
"""

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version/v7/scripts"))
import build_ir_train_v7 as author
import init_tiny_train_model_v7 as initializer
import lower_ir2_backward_v7 as backward
from version.v8.decision.build_clef_training_ir import training_registry
from version.v8.training.lora import attach_lora_adapters
from version.v8.training.lora_weights import prepare_lora_weights, resolve_lora_parameters


def load(path):
    return json.loads((ROOT / path).read_text())


def build(folder, tokens, *, layers=2, width=32, vocab=32, kv_heads=4, epsilon=1e-5, rank=2):
    import numpy as np

    initializer.build_tiny_model(
        folder, seed=41, init="xavier_uniform", template_name="qwen3",
        template_doc=load("version/v7/templates/qwen3.json"), n_layers=layers,
        vocab_size=vocab, embed_dim=width, hidden_dim=2 * width, num_heads=4,
        num_kv_heads=kv_heads, context_len=tokens, rope_theta=10000.0,
        kernel_policy="fp32", adamw_beta1=.9, adamw_beta2=.999,
        adamw_eps=1e-8, adamw_weight_decay=.01, omit_linear_biases=True,
    )
    manifest = json.loads((folder / "weights_manifest.json").read_text())
    manifest["config"]["train_tokens"] = tokens
    manifest["config"]["rms_eps"] = epsilon
    registry = training_registry()
    ir = author.build_ir1_train(
        manifest, registry, load("version/v7/kernel_maps/kernel_bindings.json"),
        load("version/v8/scripts/grad_rules_v8.json"), max_layers=layers, strict=True,
        bridge_lowering="explicit",
    )
    # These template annotations have no tensor ports or neural computation.
    # Never silently discard an unsupported arithmetic operation.
    markers = [op for op in ir["ops"] if op.get("kernel_id") is None and op.get("phase") != "bridge"]
    for op in markers:
        if (op["op"] not in {"bpe_tokenizer", "lm_head"}
                or op["weights"] or op["dataflow"] != {"inputs": {}, "outputs": {}}
                or op["requires_grad"]):
            raise ValueError("unsupported executable decoder operation")
    ir["ops"] = [op for op in ir["ops"] if op not in markers]
    original = copy.deepcopy(ir)
    base_ir = ir
    adapters = [
        dict(target="layer.0.wq", name="query0", rank=rank, alpha=2 * rank),
        dict(target=f"layer.{layers - 1}.wv", name="value1", rank=rank + 1, alpha=2 * (rank + 1)),
    ]
    ir, specs = attach_lora_adapters(base_ir, adapters=adapters, freeze_base=True)
    if original != base_ir:
        raise AssertionError("adapter declaration mutated the base IR")
    schema = resolve_lora_parameters(ir)
    blob = (folder / "weights.bump").read_bytes()
    base = {
        row["name"]: np.frombuffer(blob, dtype="<f4", count=row["size"] // 4,
                                  offset=row["offset"]).reshape(row["shape"]).copy()
        for row in manifest["entries"] if row["name"] in schema
    }
    values, evidence = prepare_lora_weights(
        ir, base, seeds={"query0": 42, "value1": 43})
    lowered = backward.synthesize_ir2_backward(
        ir, registry, load("version/v7/kernel_maps/kernel_bindings.json"),
        load("version/v8/scripts/grad_rules_v8.json"), strict=True,
        allow_partial=False, checkpoint_policy="none",
    )
    return ir, lowered, registry, values, specs, manifest, evidence


class DecoderLoRATests(unittest.TestCase):
    def test_generated_decoder_adapter_training(self):
        try:
            import numpy as np
            import torch
        except ImportError as exc:
            self.skipTest("independent PyTorch/NumPy oracle unavailable: " + str(exc))
        import oracle_snapshot_torch_v7 as oracle
        from tests.v8_training_test_helpers import certify_training

        cases = [(4, {}), (7, {}), (5, dict(layers=3, kv_heads=2, epsilon=1e-6))]
        if os.environ.get("CKE_RUN_SCALED_LORA") == "1":
            cases += [
                (9, dict(layers=12, width=128, vocab=257, rank=4)),
                (11, dict(layers=28, width=256, vocab=509, kv_heads=2, epsilon=1e-6, rank=8)),
            ]
        if os.environ.get("CKE_RUN_LORA_WIDE_STRESS") == "1":
            cases.append((23, dict(layers=28, width=512, vocab=1021,
                                   kv_heads=2, epsilon=1e-6, rank=8)))
        deep_updates = int(os.environ.get("CKE_SCALED_LORA_UPDATES", "160"))
        for tokens, geometry in cases:
            scaled = geometry.get("layers", 2) > 3
            with self.subTest(tokens=tokens, **geometry), tempfile.TemporaryDirectory() as tmp:
                ir, lowered, registry, values, specs, manifest, evidence = build(Path(tmp), tokens, **geometry)
                trainable = {s["parameters"][k] for s in specs for k in ("a", "b")}
                self.assertEqual(set(ir["adapter_training_policy"]["trainable_parameters"]), trainable)
                self.assertEqual({tid[7:] for tid, meta in ir["tensors"].items()
                                  if meta.get("kind") == "weight" and meta["requires_grad"]}, trainable)
                self.assertEqual({tid[len("grad.weight."):] for tid in lowered["tensors"]
                                  if tid.startswith("grad.weight.")}, trainable)
                ids = np.arange(tokens, dtype="i4")
                epsilon = geometry.get("epsilon")
                if epsilon is not None:
                    from tests.test_v8_selective_training import codegen, planner
                    layout = planner.build_layout(lowered, {"config": lowered["config"]}, 64, strict=True)
                    src, _ = codegen.generate_c(lowered, registry, layout=layout)
                    for provider in ("qk_norm_forward", "qk_norm_backward"):
                        calls = [line for line in src.splitlines() if line.strip().startswith(provider + "(")]
                        self.assertTrue(calls)
                        expected = "%.9ef" % float(np.float32(epsilon))
                        self.assertTrue(all(line.endswith(expected + ");") for line in calls))
                vocab = int(ir["config"]["vocab_size"])
                labels = (ids + 1) % vocab
                cfg = dict(ir["config"], rope_layout="split",
                           mlp_activation="swiglu", attn_variant="dense")
                affine = oracle._apply_affine

                def forward(params):
                    def with_adapter(x, w, bias):
                        y = affine(x, w, bias)
                        for spec in specs:
                            if w is params[spec["target"]]:
                                names = spec["parameters"]
                                y = y + ((x @ params[names["a"]].T) @ params[names["b"]].T) * params[names["scale"]]
                        return y
                    with patch.object(oracle, "_apply_affine", with_adapter):
                        model = oracle.SnapshotQwenLikeOracle(params, cfg)
                        logits, forward.slots = model.forward_with_slots(
                            torch.from_numpy(ids.astype("int64")).view(1, -1))
                        return logits.reshape(tokens, vocab)

                def check_gradients(step, params):
                    for spec in specs:
                        a, b = (params[spec["parameters"][k]] for k in ("a", "b"))
                        self.assertTrue(torch.isfinite(a.grad).all())
                        self.assertTrue(torch.isfinite(b.grad).all())
                        if step == 0:
                            self.assertEqual(float(a.grad.abs().max()), 0)
                            self.assertGreater(float(b.grad.abs().max()), 0)
                        if step == 1:
                            self.assertGreater(float(a.grad.abs().max()), 0)

                certify_training(
                    self, lowered, registry, values, trainable, forward,
                    token_ids=ids, target_ids=labels, output_tensor="act.Sfooter.logits.0.y",
                    probe_parameter="lora.query0.b", gradient_check=check_gradients,
                    finite_difference=True, fresh_process=True, check_evaluation=True,
                    checked_runtime=True,
                    loss_ratio=.95, gradient_floor=0,
                    learning_rate=(float(os.environ.get("CKE_SCALED_LORA_LR", "0.0003"))
                                   if scaled else .03),
                    updates=deep_updates if scaled else 40,
                    precision_label=f"FP32 {ir['num_layers']} layers width={ir['config']['embed_dim']} tokens={tokens}",
                    initialization_evidence=dict(base_seed=41, adapters=evidence),
                    extra_sources=("src/kernels/linear_rows_checked.c",
                                   "src/kernels/transformer_rows_checked.c"),
                )


if __name__ == "__main__":
    unittest.main()
