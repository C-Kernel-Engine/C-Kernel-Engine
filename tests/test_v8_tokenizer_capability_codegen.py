#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "version" / "v8" / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "version" / "v8" / "scripts"))

import build_ir_v8  # type: ignore


class TestV8TokenizerCapabilityCodegen(unittest.TestCase):
    def test_native_cli_prefers_generated_chat_and_stop_contracts(self) -> None:
        source = (ROOT / "version/v8/src/ck_cli_v8.c").read_text(encoding="utf-8")
        self.assertIn('resolve_symbol(api->handle, "ck_model_format_chat"', source)
        self.assertIn('resolve_symbol(api->handle, "ck_model_is_stop_token"', source)
        self.assertIn("const bool generated_chat = !opt->no_chat_template && api->format_chat", source)
        self.assertIn(
            "if (api && api->is_stop_token) {\n"
            "        return api->is_stop_token((int32_t)token) != 0;\n"
            "    }",
            source,
        )
        self.assertIn('resolve_symbol(api->handle, "ck_model_get_runtime_descriptor"', source)
        self.assertIn("generated runtime capability descriptor is inconsistent", source)
        self.assertIn(
            "if (generated_stop_contract || opt->ignore_eos)", source
        )
        self.assertIn("output_token(out_buf, &out_len, word);", source)

    def test_runtime_descriptor_is_derived_from_declared_ir_semantics(self) -> None:
        import codegen_v8  # type: ignore

        audio_ops = [{"op": name} for name in codegen_v8._AUDIO_FRONTEND_OPS]
        audio = codegen_v8._emit_runtime_capability_api(
            {
                "config": {
                    "artifact_scope": "encoder_only",
                    "context_length": 1500,
                    "embed_dim": 512,
                },
                "operations": audio_ops,
            },
            {
                "config": {"embed_dim": 512},
                "memory": {
                    "activations": {
                        "buffers": [
                            {
                                "name": "embedded_input",
                                "size_bytes": 1500 * 512 * 4,
                            }
                        ]
                    }
                },
            },
            None,
        )
        self.assertIn("CK_MODEL_CAP_AUDIO_WAV_ENCODER", audio)
        self.assertIn("CK_MODEL_CAP_ENCODER_OUTPUT", audio)
        self.assertNotIn("CK_MODEL_CAP_AUTOREGRESSIVE_DECODE,", audio)
        self.assertNotIn("CK_MODEL_CAP_TEXT_ENCODE", audio)

        vision = codegen_v8._emit_runtime_capability_api(
            {
                "config": {
                    "artifact_scope": "encoder_only",
                    "context_length": 256,
                    "image_size": 16,
                },
                "operations": [],
            },
            {
                "memory": {
                    "activations": {
                        "buffers": [
                            {"name": "image_input", "size_bytes": 3 * 16 * 16 * 4}
                        ]
                    }
                }
            },
            None,
        )
        self.assertIn("CK_MODEL_CAP_IMAGE_TENSOR_ENCODER", vision)
        self.assertIn("CK_EXPORT int ck_model_get_image_tensor_shape(", vision)
        self.assertIn("CK_EXPORT int ck_model_run_image_tensor_f32(", vision)
        self.assertNotIn("CK_MODEL_CAP_RAW_IMAGE_ENCODER,", vision)

        legacy_vision = codegen_v8._emit_runtime_capability_api(
            {"config": {"image_size": 16}, "operations": []},
            {
                "config": {"image_size": 16},
                "memory": {
                    "activations": {
                        "buffers": [
                            {"name": "image_input", "size_bytes": 3 * 16 * 16 * 4},
                            {"name": "embedded_input", "size_bytes": 8 * 32 * 4},
                        ]
                    }
                },
            },
            None,
        )
        self.assertIn("CK_MODEL_ROLE_ENCODER", legacy_vision)
        self.assertNotIn("CK_MODEL_CAP_AUTOREGRESSIVE_DECODE,", legacy_vision)

    def test_generation_policy_is_generated_from_metadata(self) -> None:
        import codegen_v8  # type: ignore

        source = codegen_v8._emit_generation_policy_api(
            {
                "decoder_start_token_id": 100,
                "eos_token_id": 101,
                "no_timestamps_token_id": 200,
                "lang_to_id": {"<|en|>": 102},
                "task_to_id": {"transcribe": 103},
                "suppress_tokens": [4, 5],
                "begin_suppress_tokens": [6],
            }
        )
        self.assertIn('strcmp(language, "en")', source)
        self.assertIn('strcmp(task, "transcribe")', source)
        self.assertIn("tokens[0] = 100", source)
        self.assertIn("tokens[3] = 200", source)
        self.assertIn("g_ck_suppress_tokens[] = { 4, 5 }", source)
        self.assertIn("g_ck_begin_suppress_tokens[] = { 6 }", source)
        self.assertIn("flags & CK_GENERATION_FLAG_TIMESTAMPS", source)

    def test_cli_audio_path_is_capability_driven(self) -> None:
        source = (ROOT / "version/v8/src/ck_cli_v8.c").read_text(encoding="utf-8")
        self.assertIn("CK_MODEL_CAP_AUDIO_WAV_ENCODER", source)
        self.assertIn("CK_MODEL_CAP_ENCODER_OUTPUT", source)
        self.assertIn("CK_MODEL_CAP_ENCODER_MEMORY", source)
        self.assertIn("encoder/decoder contract mismatch", source)
        self.assertIn("decoder->build_generation_prefix(", source)
        self.assertIn("CK_MODEL_CAP_IMAGE_TENSOR_ENCODER", source)
        self.assertIn("encoder.run_image_tensor_f32(", source)
        self.assertIn("run_bridge_report_with_prefix(", source)
        self.assertNotIn('strstr(opt->model_name, "whisper")', source)

    def test_bpe_distinguishes_decode_tables_from_text_encode(self) -> None:
        generated = build_ir_v8._generate_tokenizer_c_code(
            "bpe",
            vocab_size=256,
            num_merges=0,
            special_tokens={"bos_token_id": 1, "eos_token_id": 2},
            model_type="nemotron_h",
        )
        self.assertIsNotNone(generated)
        c_code = str(generated["api_functions"])
        self.assertIn("CK_EXPORT int ck_model_has_tokenizer(void)", c_code)
        self.assertIn("#ifdef W_VOCAB_OFFSETS", c_code)
        self.assertIn("CK_DISABLE_FULL_BPE_TOKENIZER", c_code)
        self.assertNotIn("CK_ENABLE_FULL_BPE_TOKENIZER", c_code)
        self.assertIn("CK_EXPORT int ck_model_can_encode_text(void)", c_code)
        self.assertIn("return (g_model && g_model->tokenizer) ? 1 : 0;", c_code)
        self.assertIn('printf("[Tokenizer] Registered special: %s -> %d\\n",', str(generated["init"]))

    def test_bpe_codegen_applies_declared_pretokenizer_profile(self) -> None:
        generated = build_ir_v8._generate_tokenizer_c_code(
            "bpe",
            vocab_size=256,
            num_merges=0,
            special_tokens={"add_bos_token": False, "add_eos_token": False},
            tokenizer_contract={
                "tokenizer_type": "bpe",
                "pretokenizer": "unicode_split_isolated",
            },
        )
        self.assertIsNotNone(generated)
        init = str(generated["init"])
        self.assertIn(
            "cfg.pretokenizer = CK_BPE_PRETOKENIZER_UNICODE_SPLIT_ISOLATED;",
            init,
        )

    def test_sentencepiece_exports_text_encode_capability(self) -> None:
        generated = build_ir_v8._generate_tokenizer_c_code(
            "sentencepiece",
            vocab_size=256,
            num_merges=0,
            special_tokens={"bos_token_id": 1, "eos_token_id": 2},
            model_type="gemma3",
        )
        self.assertIsNotNone(generated)
        c_code = str(generated["api_functions"])
        self.assertIn("CK_EXPORT int ck_model_has_tokenizer(void)", c_code)
        self.assertIn("CK_EXPORT int ck_model_can_encode_text(void)", c_code)
        self.assertIn("return (g_model && g_model->tokenizer) ? 1 : 0;", c_code)
        self.assertIn('printf("[Tokenizer] Registered special: %s -> %d\\n",', str(generated["init"]))

    def test_chat_contract_exports_generated_native_formatter(self) -> None:
        generated = build_ir_v8._generate_tokenizer_c_code(
            "bpe",
            vocab_size=256,
            num_merges=0,
            special_tokens={"bos_token_id": 1, "eos_token_id": 2},
            chat_contract={
                "turn_prefix": "<|{role}|>\n",
                "turn_suffix": "\n",
                "assistant_generation_prefix": "<|assistant|>",
                "role_labels": {"system": "system", "user": "user"},
                "system_prompt_mode": "dedicated_turn",
                "default_system_prompt": "Be concise.",
                "inject_default_system_prompt": True,
                "force_bos_text_if_tokenizer_add_bos_false": "[gMASK]<sop>\n",
            },
        )
        self.assertIsNotNone(generated)
        c_code = str(generated["api_functions"])
        self.assertIn("CK_EXPORT int ck_model_has_chat_template(void)", c_code)
        self.assertIn("CK_EXPORT int ck_model_format_chat(", c_code)
        self.assertIn('"<|system|>\\n"', c_code)
        self.assertIn('"<|user|>\\n"', c_code)
        self.assertIn('"[gMASK]<sop>\\n"', c_code)

    def test_generated_formatter_omits_text_bos_when_tokenizer_adds_bos(self) -> None:
        generated = build_ir_v8._generate_tokenizer_c_code(
            "bpe",
            vocab_size=256,
            num_merges=0,
            special_tokens={
                "bos_token_id": 2,
                "eos_token_id": 1,
                "add_bos_token": True,
            },
            chat_contract={
                "turn_prefix": "<start_of_turn>{role}\\n",
                "turn_suffix": "<end_of_turn>\\n",
                "assistant_generation_prefix": "<start_of_turn>model\\n",
                "role_labels": {"user": "user", "assistant": "model"},
                "force_bos_text_if_tokenizer_add_bos_false": "<bos>",
            },
        )
        self.assertIsNotNone(generated)
        c_code = str(generated["api_functions"])
        self.assertIn("CK_EXPORT int ck_model_format_chat(", c_code)
        self.assertNotIn(
            "ck_model_chat_append(output, output_capacity, position, \"<bos>\")",
            c_code,
        )

    def test_generated_formatter_preserves_declarative_role_affixes(self) -> None:
        generated = build_ir_v8._generate_tokenizer_c_code(
            "bpe",
            vocab_size=256,
            num_merges=0,
            special_tokens={"bos_token_id": 1, "eos_token_id": 12, "add_bos_token": True},
            chat_contract={
                "conversation_prefix": "<SPECIAL_10>System\n",
                "turn_prefix": "<SPECIAL_11>{role}\n",
                "turn_prefix_by_role": {"system": ""},
                "turn_suffix": "\n",
                "assistant_generation_prefix": "<SPECIAL_11>Assistant\n<think>\n",
                "role_labels": {"system": "System", "user": "User"},
                "system_prompt_mode": "dedicated_turn",
                "render_empty_system_turn": True,
            },
        )
        self.assertIsNotNone(generated)
        c_code = str(generated["api_functions"])
        self.assertIn('position, "<SPECIAL_10>System\\n")', c_code)
        self.assertIn('position, "<SPECIAL_11>User\\n")', c_code)
        self.assertIn("system_value[0] || 1", c_code)

    def test_generated_formatter_keeps_text_bos_when_tokenizer_does_not_add_bos(self) -> None:
        generated = build_ir_v8._generate_tokenizer_c_code(
            "bpe",
            vocab_size=256,
            num_merges=0,
            special_tokens={
                "bos_token_id": 2,
                "eos_token_id": 1,
                "add_bos_token": False,
            },
            chat_contract={
                "turn_prefix": "<start_of_turn>{role}\\n",
                "turn_suffix": "<end_of_turn>\\n",
                "assistant_generation_prefix": "<start_of_turn>model\\n",
                "role_labels": {"user": "user", "assistant": "model"},
                "force_bos_text_if_tokenizer_add_bos_false": "<bos>",
            },
        )
        self.assertIsNotNone(generated)
        self.assertIn(
            "ck_model_chat_append(output, output_capacity, position, \"<bos>\")",
            str(generated["api_functions"]),
        )

    def test_incomplete_multimodal_marker_contract_does_not_export_formatter(self) -> None:
        generated = build_ir_v8._generate_tokenizer_c_code(
            "sentencepiece",
            vocab_size=256,
            num_merges=0,
            special_tokens={"bos_token_id": 1, "eos_token_id": 2},
            chat_contract={"image_begin_marker": "<image>", "image_end_marker": "</image>"},
        )
        self.assertIsNotNone(generated)
        self.assertNotIn("ck_model_format_chat", str(generated["api_functions"]))


if __name__ == "__main__":
    unittest.main()
