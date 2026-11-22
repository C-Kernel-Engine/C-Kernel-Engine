"""Asset-free checks for the Inkling GGUF importer contract."""
from dataclasses import replace
import unittest

from version.v8.scripts import convert_gguf_to_bump_v8 as gguf


ARCH = "inkling2_loop"


def fixture():
    properties = {
        "block_count": 4, "context_length": 128, "embedding_length": 32, "feed_forward_length": 32,
        "expert_feed_forward_length": 32, "attention_head_count": [2] * 4,
        "attention_head_count_kv": [1] * 4, "attention_key_length": 32, "attention_value_length": 32,
        "attention_key_length_swa": 32, "attention_value_length_swa": 32,
        "attention_layer_norm_rms_epsilon": 1e-6, "attention_sliding_window": 8,
        "attention_sliding_window_pattern": [True, False, True, False],
        "attention_relative_length": 4, "attention_relative_extent": 8,
        "attention_qk_clip": 8., "attention_relative_clip": 4.,
        "attention_log_scale_floor": 128, "attention_log_scale_alpha": .1,
        "expert_count": 4, "expert_used_count": 2, "expert_weights_scale": 2.,
        "gate_softcap": 4., "up_softcap": 25., "vocab_size": 64, "vocab_size_unpadded": 60,
        "conv_taps": 4, "block_residual_size": 1, "logit_softcap": 30., "rope_dimension_count": 0,
        "loop_start": 1, "loop_end": 3, "loop_count": 3, "loop_gate_rank": 8, "loop_varying": False,
    }
    metadata = {"general.architecture": ARCH}
    metadata.update({gguf.gguf_ck_metadata_key(ARCH, key): value for key, value in properties.items()})
    tensors = {}
    offset = 0

    def put(name, dims, kind=gguf.GGML_TYPE_F32):
        nonlocal offset
        info = gguf.TensorInfo(name, dims, kind, offset)
        tensors[name] = info
        offset = gguf.align_up(offset + gguf.ggml_tensor_bytes(info), 32)

    put("token_embd.weight", (32, 64), gguf.GGML_TYPE_BF16)
    put("output.weight", (32, 64), gguf.GGML_TYPE_Q8_0)
    for name in ("token_embd_norm.weight", "output_norm.weight", "output_res_query.weight"):
        put(name, (32,))
    for layer in range(4):
        common = {
            "attn_k_conv.weight": (4, 32), "attn_v_conv.weight": (4, 32),
            "attn_q_clip.weight": (32, 2), "attn_rel_clip.weight": (4, 2),
            "attn_rel_bias.weight": (4, 8), "attn_post_conv.weight": (4, 32),
            "ffn_post_conv.weight": (4, 32), "ffn_norm.weight": (32,),
            "ffn_res_query.weight": (32,), "exp_probs_b.bias": (4,), "ffn_gate_inp.weight": (32, 4),
        }
        if layer in (0, 2):
            common["attn_norm.weight"] = (32,)
        else:
            common.update({f"attn_{port}_inp_norm.weight": (32,) for port in ("q", "k", "v")})
            common.update({f"attn_{port}_conv_proj.weight": (32, 4) for port in ("k", "v")})
        if layer:
            common["attn_res_query.weight"] = (32, 3) if layer in (1, 3) else (32,)
        if layer in (1, 2):
            common["loop_msg_query.weight"] = (32, 4)
            for port in ("q", "k", "v", "ffn"):
                common[f"loop_{port}_ctrl_down.weight"] = (64, 8)
                for branch in ("gain", "forget"):
                    common[f"loop_{port}_ctrl_{branch}.weight"] = (8, 32)
        prefix = f"blk.{layer}."
        for name, shape in common.items():
            put(prefix + name, shape)
        projections = {
            "attn_q.weight": (32, 64), "attn_k.weight": (32, 32), "attn_v.weight": (32, 32),
            "attn_gate.weight": (32, 64), "attn_output.weight": (64, 32),
            "attn_rel.weight": (32, 8), "attn_sink_proj.weight": (32, 2),
        }
        for name, shape in projections.items():
            put(prefix + name, shape, gguf.GGML_TYPE_Q8_0)
        kind = (gguf.GGML_TYPE_BF16, gguf.GGML_TYPE_IQ4_NL, gguf.GGML_TYPE_Q8_0)[layer % 3]
        put(prefix + "ffn_gate_up_exps.weight", (32, 64, 4), kind)
        put(prefix + "ffn_down_exps.weight", (32, 32, 4), kind)
    return tensors, metadata


class InklingGGUFContractTests(unittest.TestCase):
    def test_reports_storage_and_disjoint_state_slots(self):
        tensors, meta = fixture()
        audit = gguf.audit_inkling2_loop_gguf_contract(tensors, meta, data_start=1024, file_size=1024)
        self.assertEqual(audit["payload_extent"], "header_or_truncated")
        self.assertEqual(audit["execution"], "NOT_TESTED")
        self.assertEqual(audit["tensor_count"], len(tensors))
        self.assertEqual([row["weight_layer"] for row in audit["text_schedule"]], [0, 1, 2, 1, 2, 1, 2, 3])
        self.assertEqual([row["state_slot"] for row in audit["text_schedule"]], [0, 1, 2, 4, 5, 6, 7, 3])
        self.assertEqual(audit["expert_storage_by_layer"][0]["gate_up"], "BF16")
        self.assertEqual(audit["expert_storage_by_layer"][1]["down"], "IQ4_NL")
        self.assertEqual(audit["expert_storage_by_layer"][2]["down"], "Q8_0")
        self.assertEqual(gguf.ggml_row_bytes(gguf.GGML_TYPE_IQ4_NL, 32), 18)
        self.assertIn("inspection_only", gguf.gguf_ck_conversion_blocker(ARCH))

    def test_rejects_wrong_width_shape_and_payload_overlap(self):
        tensors, meta = fixture()
        for name, change, pattern in (
            ("blk.1.ffn_down_exps.weight", {"dims": (32, 4, 32)}, "unexpected shape"),
            ("blk.1.ffn_down_exps.weight", {"offset": 0}, "overlapping"),
        ):
            with self.subTest(change=change), self.assertRaisesRegex(gguf.GGUFError, pattern):
                gguf.audit_inkling2_loop_gguf_contract({**tensors, name: replace(tensors[name], **change)}, meta)
        key = gguf.gguf_ck_metadata_key(ARCH, "expert_feed_forward_length")
        with self.assertRaisesRegex(gguf.GGUFError, "feed-forward extents disagree"):
            gguf.audit_inkling2_loop_gguf_contract(tensors, {**meta, key: 64})

    def test_rejects_missing_loop_tensors_and_invalid_mask(self):
        tensors, meta = fixture()
        del tensors["blk.1.loop_msg_query.weight"]
        with self.assertRaisesRegex(gguf.GGUFError, "missing .*loop_msg_query"):
            gguf.audit_inkling2_loop_gguf_contract(tensors, meta)
        key = gguf.gguf_ck_metadata_key(ARCH, "attention_sliding_window_pattern")
        with self.assertRaisesRegex(gguf.GGUFError, "invalid layer mask"):
            gguf.audit_inkling2_loop_gguf_contract(tensors, {**meta, key: [True, False]})


if __name__ == "__main__":
    unittest.main()
