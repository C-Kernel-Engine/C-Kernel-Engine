"""Complete small decoder LoRA; oracle code is development-only PyTorch."""

import copy
import json
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


ADAPTERS = [
    dict(target="layer.0.wq", name="query0", rank=2, alpha=4),
    dict(target="layer.1.wv", name="value1", rank=3, alpha=6),
]


def build(folder, tokens):
    import numpy as np

    initializer.build_tiny_model(
        folder, seed=41, init="xavier_uniform", template_name="qwen3",
        template_doc=load("version/v7/templates/qwen3.json"), n_layers=2,
        vocab_size=32, embed_dim=32, hidden_dim=64, num_heads=4,
        num_kv_heads=4, context_len=tokens, rope_theta=10000.0,
        kernel_policy="fp32", adamw_beta1=.9, adamw_beta2=.999,
        adamw_eps=1e-8, adamw_weight_decay=.01, omit_linear_biases=True,
    )
    manifest = json.loads((folder / "weights_manifest.json").read_text())
    manifest["config"]["train_tokens"] = tokens
    manifest["config"]["rms_eps"] = 1e-5
    registry = training_registry()
    ir = author.build_ir1_train(
        manifest, registry, load("version/v7/kernel_maps/kernel_bindings.json"),
        load("version/v8/scripts/grad_rules_v8.json"), max_layers=2, strict=True,
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
    ir, specs = attach_lora_adapters(base_ir, adapters=ADAPTERS, freeze_base=True)
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

        for tokens in (4, 7):
            with self.subTest(tokens=tokens), tempfile.TemporaryDirectory() as tmp:
                ir, lowered, registry, values, specs, manifest, evidence = build(Path(tmp), tokens)
                trainable = {s["parameters"][k] for s in specs for k in ("a", "b")}
                self.assertEqual(set(ir["adapter_training_policy"]["trainable_parameters"]), trainable)
                self.assertEqual({tid[7:] for tid, meta in ir["tensors"].items()
                                  if meta.get("kind") == "weight" and meta["requires_grad"]}, trainable)
                self.assertEqual({tid[len("grad.weight."):] for tid in lowered["tensors"]
                                  if tid.startswith("grad.weight.")}, trainable)
                ids = np.arange(tokens, dtype="i4")
                labels = (ids + 1) % 32
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
                        return logits.reshape(tokens, 32)

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
                    initialization_evidence=dict(base_seed=41, adapters=evidence),
                    extra_sources=("src/kernels/linear_rows_checked.c",
                                   "src/kernels/transformer_rows_checked.c"),
                )


if __name__ == "__main__":
    unittest.main()
