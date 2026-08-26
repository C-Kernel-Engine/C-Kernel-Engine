import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version/v8/scripts/xray_decoder_pytorch_v8.py"
SPEC = importlib.util.spec_from_file_location("xray_decoder_pytorch_v8", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class DecoderPyTorchXRayTests(unittest.TestCase):
    def test_structured_module_output_uses_primary_tensor(self) -> None:
        primary = np.asarray([1.0, 2.0], dtype=np.float32)
        auxiliary = np.asarray([3.0, 4.0], dtype=np.float32)

        self.assertIs(MODULE._tensor_from_output((primary, auxiliary)), primary)

    def test_last_feature_row_handles_latent_kv_projection_shape(self) -> None:
        values = np.arange(1 * 1 * 16 * 8, dtype=np.float32).reshape(1, 1, 16, 8)

        np.testing.assert_array_equal(
            MODULE._last_feature_row(values),
            values[0, 0, -1],
        )

    def test_split_mla_kv_projection_matches_flat_ck_exports(self) -> None:
        import torch

        values = torch.arange(1 * 1 * 3 * 16, dtype=torch.float32).reshape(
            1, 1, 3, 16
        )
        k_nope, value = MODULE._split_mla_kv_projection(
            values, heads=2, k_width=5, v_width=3
        )
        expanded = values[0, 0, -1].reshape(2, 8)

        np.testing.assert_array_equal(k_nope.numpy(), expanded[:, :5].reshape(-1))
        np.testing.assert_array_equal(value.numpy(), expanded[:, 5:].reshape(-1))

    def test_split_full_mla_kv_projection_preserves_token_order(self) -> None:
        import torch

        values = torch.arange(1 * 1 * 3 * 16, dtype=torch.float32).reshape(
            1, 1, 3, 16
        )
        k_nope, value = MODULE._split_mla_kv_projection_full(
            values, heads=2, k_width=5, v_width=3
        )
        expanded = values.reshape(3, 2, 8)

        np.testing.assert_array_equal(k_nope.numpy(), expanded[:, :, :5].numpy())
        np.testing.assert_array_equal(value.numpy(), expanded[:, :, 5:].numpy())

    def test_explicit_last_row_capture_is_not_reshaped_as_a_full_trajectory(self) -> None:
        values = np.arange(12, dtype=np.float32)

        boundary, logical = MODULE._logical_ck_capture(
            values, "mlp_gate_last", token_count=4
        )

        self.assertEqual(boundary, "mlp_gate")
        np.testing.assert_array_equal(logical, values)

    def test_instrumented_attention_resolver_captures_semantic_mla_edges(self) -> None:
        import torch

        captured = {3: {}}

        def resolver(_config):
            def interface(_module, query, _key, _value, *_args, **_kwargs):
                return query.transpose(1, 2), None

            return interface

        interface = MODULE._instrument_attention_resolver(
            resolver, captured, frozenset({3})
        )(object())
        query = torch.arange(1 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 3, 4)
        result = interface(SimpleNamespace(layer_idx=3), query, query + 1, query + 2)

        expected_query = query[0, :, -1, :].reshape(-1)
        np.testing.assert_array_equal(captured[3]["mla_query"].numpy(), expected_query)
        np.testing.assert_array_equal(
            captured[3]["mla_key"].numpy(), expected_query + 1
        )
        np.testing.assert_array_equal(
            captured[3]["mla_context"].numpy(),
            result[0][0, -1].reshape(-1),
        )
        np.testing.assert_array_equal(
            captured[3]["mla_query_full"].numpy(),
            query[0].transpose(0, 1).numpy(),
        )
        np.testing.assert_array_equal(
            captured[3]["mla_key_full"].numpy(),
            (query[0] + 1).transpose(0, 1).numpy(),
        )
        np.testing.assert_array_equal(
            captured[3]["mla_value_full"].numpy(),
            (query[0] + 2).transpose(0, 1).numpy(),
        )
        np.testing.assert_array_equal(
            captured[3]["mla_context_full"].numpy(),
            result[0][0].numpy(),
        )

    def test_mla_operation_requires_one_numeric_explicit_contract(self) -> None:
        operation = {
            "op": "mla_attention",
            "layer": 2,
            "args": [{"name": "scale", "expr": "0.125"}],
        }
        call_ir = {"operations": [operation]}

        self.assertIs(MODULE._mla_operation(call_ir, 2), operation)
        self.assertEqual(MODULE._literal_call_arg(operation, "scale"), 0.125)
        with self.assertRaisesRegex(RuntimeError, "missing"):
            MODULE._literal_call_arg(operation, "num_heads")

    def test_parse_token_ids_accepts_explicit_csv(self) -> None:
        self.assertEqual(MODULE.parse_token_ids("1, 2,3"), [1, 2, 3])

    def test_teacher_forced_split_requires_prompt_and_suffix(self) -> None:
        self.assertEqual(
            MODULE.split_teacher_forced_tokens([10, 11, 12, 13], 2),
            ([10, 11], [12, 13]),
        )
        with self.assertRaisesRegex(ValueError, "leave at least one"):
            MODULE.split_teacher_forced_tokens([10, 11], 2)

    def test_persistent_capture_rejects_a_trajectory_that_left_the_prefix(self) -> None:
        trajectory = {
            "generated_tokens": [41, 99],
            "logits": [object(), object()],
        }
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                MODULE,
                "load_ck_greedy_trajectory",
                return_value=trajectory,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "left the teacher-forced prefix"
                ):
                    MODULE.capture_ck_persistent(
                        Path(directory),
                        [1, 2],
                        [41, 42],
                        Path(directory) / "capture",
                    )

    def test_run_reports_persistent_vs_replay_divergence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            call_ir = tmp_path / "call.json"
            call_ir.write_text('{"operations": []}\n', encoding="utf-8")

            def tensors(prefix: str, layer_one: list[float]):
                return {
                    0: {
                        "layer_out": MODULE._write_tensor(
                            tmp_path / f"{prefix}_layer0.f32",
                            np.asarray([1.0, 2.0], dtype=np.float32),
                        )
                    },
                    1: {
                        "layer_out": MODULE._write_tensor(
                            tmp_path / f"{prefix}_layer1.f32",
                            np.asarray(layer_one, dtype=np.float32),
                        )
                    },
                }

            replay = tensors("replay", [3.0, 4.0])
            oracle = tensors("oracle", [3.0, 4.0])
            persistent = tensors("persistent", [6.0, 8.0])
            replay_logits = np.asarray([0.0, 2.0, 1.0], dtype=np.float32)
            persistent_logits = np.asarray([3.0, 2.0, 1.0], dtype=np.float32)
            args = SimpleNamespace(
                token_ids="10,11,12",
                prompt_token_count=2,
                output_dir=tmp_path / "report",
                call_ir=call_ir,
                checkpoint=tmp_path,
                runtime=tmp_path,
                threads=1,
                model_name="fixture",
                top_k=2,
            )
            with (
                mock.patch.object(
                    MODULE,
                    "capture_pytorch",
                    return_value=(oracle, replay_logits),
                ),
                mock.patch.object(
                    MODULE,
                    "capture_ck",
                    return_value=(replay, replay_logits),
                ),
                mock.patch.object(
                    MODULE,
                    "capture_ck_persistent",
                    return_value=(persistent, persistent_logits),
                ),
            ):
                result = MODULE.run(args)

            state = result["persistent_vs_replay"]
            self.assertEqual(result["attribution_scope"], "persistent_vs_replay")
            self.assertEqual(result["persistent_state_status"], "diverged")
            self.assertFalse(state["ranking"]["top1_match"])
            self.assertEqual(
                state["first_material"]["checkpoint_id"],
                "decoder.layer.1.layer_out",
            )

    def test_decoder_layers_discovers_nested_language_model_without_model_name(self) -> None:
        layers = [object(), object(), object()]
        model = SimpleNamespace(
            language_model=SimpleNamespace(model=SimpleNamespace(layers=layers))
        )
        self.assertEqual(MODULE.decoder_layers(model), layers)

    def test_invalid_meta_initialized_rotary_caches_are_rebuilt_and_shared(self) -> None:
        import torch

        class Rotary(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dim = 4
                self.base = 10000.0
                self.max_position_embeddings = 8
                self.max_seq_len_cached = None
                self.register_buffer(
                    "inv_freq", torch.full((2,), float("nan")), persistent=False
                )
                self.register_buffer(
                    "cos_cached", torch.zeros((8, 4)), persistent=False
                )
                self.register_buffer(
                    "sin_cached", torch.zeros((8, 4)), persistent=False
                )

            def _set_cos_sin_cache(self, seq_len, device, dtype) -> None:
                positions = torch.arange(seq_len, device=device, dtype=torch.float32)
                frequencies = torch.outer(positions, self.inv_freq)
                embedding = torch.cat((frequencies, frequencies), dim=-1)
                self.cos_cached = embedding.cos().to(dtype)
                self.sin_cached = embedding.sin().to(dtype)
                self.max_seq_len_cached = seq_len

        class Attention(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rotary_emb = Rotary()

        model = torch.nn.ModuleList([Attention(), Attention()])

        repaired = MODULE._repair_invalid_rotary_buffers(model)

        self.assertEqual(repaired, 2)
        self.assertTrue(torch.isfinite(model[0].rotary_emb.inv_freq).all())
        self.assertTrue(torch.equal(model[0].rotary_emb.cos_cached[0], torch.ones(4)))
        self.assertIs(
            model[0].rotary_emb.cos_cached,
            model[1].rotary_emb.cos_cached,
        )

    def test_sparse_manifest_keeps_resolved_provider_identity(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            tensor = tmp_path / "layer.f32"
            tensor.write_bytes(b"\0" * 16)
            record = {
                "path": str(tensor),
                "shape": [4],
                "sha256": MODULE._sha256(tensor),
            }
            call_ir = {
                "operations": [
                    {
                        "op": "residual_add",
                        "layer": 0,
                        "function": "residual_add_f32",
                        "kernel_id": "residual_add_f32",
                        "resolved_contract_id": "residual.fp32",
                    }
                ]
            }
            result = MODULE.manifest(
                "ck",
                "fixture",
                tmp_path,
                {0: {"layer_out": record}},
                call_ir,
                [(0, "layer_out")],
            )
            checkpoint = result["checkpoints"][0]
            self.assertEqual(checkpoint["checkpoint_id"], "decoder.layer.0.layer_out")
            self.assertEqual(checkpoint["kernel_id"], "residual_add_f32")
            self.assertEqual(checkpoint["function"], "residual_add_f32")
            self.assertEqual(checkpoint["resolved_contract_id"], "residual.fp32")

    def test_mla_boundary_metadata_tracks_the_actual_producer(self) -> None:
        call_ir = {
            "operations": [
                {
                    "op": "attention_gate_projection",
                    "layer": 0,
                    "kernel_id": "gemm_nt_bf16",
                    "resolved_contract_id": "bf16.gate",
                }
            ]
        }

        result = MODULE.operation_metadata(call_ir, 0, "attn_gate")

        self.assertEqual(result["producer"], "attention_gate_projection")
        self.assertEqual(result["kernel_id"], "gemm_nt_bf16")


if __name__ == "__main__":
    unittest.main()
