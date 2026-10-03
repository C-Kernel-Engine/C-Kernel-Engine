"""Guard the mapped M=2 arithmetic and fail-closed generated batch selection."""

from __future__ import annotations

import copy
import ctypes
import json
import math
import platform
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version/v8/scripts"))
from batch_decode_contract_v8 import resolve_two_row_batch_contract  # noqa: E402
from batch_decode_codegen_v8 import emit_two_row_batch_api  # noqa: E402
from server.serving_bundle import verified_loaded_symbol_backing  # noqa: E402


def _fixture() -> tuple[list[dict], dict, dict]:
    buffers = [
        {"name": name, "size": size, "abs_offset": offset, "define": macro,
         "lifetime": lifetime, "mutable": True}
        for name, size, offset, macro, lifetime in (
            ("kv_cache", 64, 64, "A_KV_CACHE", "sequence"),
            ("embedded_input", 128, 128, "A_EMBEDDED_INPUT", "call"),
            ("residual", 128, 256, "A_RESIDUAL", "call"),
            ("q_scratch", 64, 384, "A_Q_SCRATCH", "call"),
            ("k_scratch", 32, 448, "A_K_SCRATCH", "call"),
            ("logits", 256, 512, "A_LOGITS", "call"),
        )
    ]
    layout = {"memory": {"arena": {"total_size": 768}, "activations": {"buffers": buffers}}}
    def arg(name: str, expr: str, ref: str | None = None, source: str = "dim:test") -> dict:
        row = {"name": name, "expr": expr, "source": source}
        if ref:
            row["buffer_ref"] = ref
        return row
    ops = [
        {"op": "dense_embedding_lookup", "args": [
            arg("token_ids", "(int32_t*)(model->bump + A_TOKEN_IDS)", "token_ids", "activation:tokens"),
            arg("token_count", "1"),
            arg("output", "(float*)(model->bump + A_EMBEDDED_INPUT)", "embedded_input", "output:output")]},
        {"op": "residual_save", "args": [
            arg("dst", "(void*)(model->bump + A_RESIDUAL)", "residual", "output:dst"),
            arg("src", "(const void*)(model->bump + A_EMBEDDED_INPUT)", "embedded_input", "activation:src"),
            arg("size", "128")]},
        {"op": "attn_norm", "args": [
            arg("input", "(const float*)(model->bump + A_EMBEDDED_INPUT)", "embedded_input", "activation:input"),
            arg("output", "(float*)(model->bump + A_EMBEDDED_INPUT)", "embedded_input", "output:output")]},
        {"op": "q_proj", "layer": 0, "function": "gemv_q5_1_q8_1", "call_abi": {"kernel_id": "gemv_q5_1"}, "args": [
            arg("x", "(const float*)(model->bump + A_EMBEDDED_INPUT)", "embedded_input", "activation:x"),
            arg("y", "(float*)(model->bump + A_Q_SCRATCH)", "q_scratch", "output:y"),
            arg("W", "(const void*)(model->bump + W_LAYER_0_WQ)"),
            arg("M", "16"), arg("K", "32")]},
        {"op": "k_proj", "layer": 0, "function": "gemv_q5_1_q8_1", "call_abi": {"kernel_id": "gemv_q5_1"}, "args": [
            arg("x", "(const float*)(model->bump + A_EMBEDDED_INPUT)", "embedded_input", "activation:x"),
            arg("y", "(float*)(model->bump + A_K_SCRATCH)", "k_scratch", "output:y"),
            arg("W", "(const void*)(model->bump + W_LAYER_0_WK)"),
            arg("M", "8"), arg("K", "32")]},
        {"op": "attention", "args": [
            arg("input", "(const float*)(model->bump + A_EMBEDDED_INPUT)", "embedded_input", "activation:input"),
            arg("residual", "(const float*)(model->bump + A_RESIDUAL)", "residual", "activation:residual"),
            arg("q", "(const float*)(model->bump + A_Q_SCRATCH)", "q_scratch", "scratch:q"),
            arg("k", "(const float*)(model->bump + A_K_SCRATCH)", "k_scratch", "scratch:k")]},
    ]
    return ops, layout, {"vocab_size": 64}


def _layer_fixture() -> tuple[list[dict], dict, dict]:
    ops, layout, config = _fixture()
    def arg(name: str, expr: str, ref: str | None = None, source: str = "dim:test") -> dict:
        item = {"name": name, "expr": expr, "source": source}
        if ref is not None:
            item["buffer_ref"] = ref
        return item
    layout["memory"]["arena"]["total_size"] = 1280
    layout["memory"]["activations"]["buffers"].append(
        {"name": "mlp_scratch", "size": 256, "abs_offset": 800,
         "define": "A_MLP_SCRATCH", "lifetime": "call", "mutable": True})
    ops[5]["layer"] = 0
    ops[5]["args"].append(arg("output", "(float*)(model->bump + A_EMBEDDED_INPUT)",
                              "embedded_input", "output:output"))
    ops.extend([
        {"op": "mlp_gate_up", "layer": 0, "function": "gemv_q5_1_q8_1",
         "call_abi": {"kernel_id": "gemv_q5_1"}, "args": [
             arg("y", "(float*)(model->bump + A_MLP_SCRATCH)", "mlp_scratch", "output:y"),
             arg("W", "(const void*)(model->bump + W_LAYER_0_W1)"),
             arg("x", "(const float*)(model->bump + A_EMBEDDED_INPUT)", "embedded_input", "activation:x"),
             arg("M", "64"), arg("K", "32")]},
        {"op": "geglu", "layer": 0, "args": [
            arg("x", "(const float*)(model->bump + A_MLP_SCRATCH)", "mlp_scratch", "activation:input"),
            arg("out", "(float*)(model->bump + A_MLP_SCRATCH)", "mlp_scratch", "output:output"),
            arg("tokens", "1"), arg("dim", "32")]},
        {"op": "finish", "layer": 0, "args": [
            arg("residual", "(const float*)(model->bump + A_RESIDUAL)", "residual", "activation:residual"),
            arg("input", "(const float*)(model->bump + A_MLP_SCRATCH)", "mlp_scratch", "activation:input"),
            arg("output", "(float*)(model->bump + A_EMBEDDED_INPUT)", "embedded_input", "output:output")]},
    ])
    return ops, layout, config


def _avx2_available() -> bool:
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        return False
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
    except OSError:
        return False
    return any(line.startswith(("flags", "Features")) and
               "avx2" in line.partition(":")[2].split()
               for line in cpuinfo.splitlines())


class BatchContractTests(unittest.TestCase):
    def test_layer_extension_is_liveness_and_provider_bound(self) -> None:
        ops, layout, config = _layer_fixture()
        contract = resolve_two_row_batch_contract(ops, layout, config)
        self.assertEqual(contract["layer_extension"]["cut"], 6)
        self.assertEqual(contract["layer_extension"]["output_dim"], 64)
        for change in ("offset_input", "offset_output", "wrong_provider", "bad_geglu",
                       "extra_mid_live", "other_layer"):
            bad_ops, bad_layout = copy.deepcopy(ops), copy.deepcopy(layout)
            if change == "offset_input":
                bad_ops[6]["args"][2]["expr"] += " + 4"
            elif change == "offset_output":
                bad_ops[6]["args"][0]["expr"] += " + 4"
            elif change == "wrong_provider":
                bad_ops[6]["call_abi"]["kernel_id"] = "gemv_q5_0"
            elif change == "bad_geglu":
                bad_ops[7]["args"][-1]["expr"] = "16"
            elif change == "other_layer":
                bad_ops[5]["layer"] = 1
            else:
                bad_layout["memory"]["activations"]["buffers"].append(
                    {"name": "extra", "size": 16, "abs_offset": 1100,
                     "define": "A_EXTRA", "lifetime": "call", "mutable": True})
                bad_ops[5]["args"].append(
                    {"name": "extra", "expr": "(float*)(model->bump + A_EXTRA)",
                     "source": "output:extra", "buffer_ref": "extra"})
                bad_ops[8]["args"].append(
                    {"name": "extra", "expr": "(const float*)(model->bump + A_EXTRA)",
                     "source": "activation:extra", "buffer_ref": "extra"})
            with self.subTest(change=change):
                fallback = resolve_two_row_batch_contract(bad_ops, bad_layout, config)
                if change == "offset_input":
                    self.assertIsNone(fallback)
                else:
                    self.assertIsNotNone(fallback)
                    self.assertNotIn("layer_extension", fallback)

    def test_compiled_generated_entry_and_nonparticipating_arena(self) -> None:
        prelude = r'''
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#define CK_EXPORT
#define VOCAB_SIZE 64
#define MAX_SEQ_LEN 16
#define KV_CACHE_SIZE 64
#define A_EMBEDDED_INPUT 128
#define A_RESIDUAL 256
#define A_Q_SCRATCH 384
#define A_K_SCRATCH 448
#define A_MLP_SCRATCH 800
#define W_LAYER_0_WQ 600
#define W_LAYER_0_WK 608
#define W_LAYER_0_W1 616
typedef struct { uint64_t sequence_handle; int token, position; size_t row_offset, token_count; float *logits; } CKModelBatchDecodeRowV8;
typedef struct { int occupied, pos; uint64_t handle; unsigned char *kv; } CKSequenceStateV8;
typedef struct { unsigned char bump[1280]; size_t bump_size; int pos; float logits[64]; } Model;
static Model object = {.bump_size = 1280};
static Model *g_model = &object;
static CKSequenceStateV8 g_sequence_states[3];
static CKSequenceStateV8 *g_active_sequence;
static int g_ck_skip_decode_logits;
static CKSequenceStateV8 *ck_sequence_find(uint64_t handle) {
    for (int i=0;i<3;i++) if (g_sequence_states[i].occupied && g_sequence_states[i].handle == handle) return &g_sequence_states[i];
    return NULL;
}
static int ck_model_sequence_state_activate(uint64_t handle) {
    g_active_sequence = ck_sequence_find(handle);
    if (!g_active_sequence) return -1;
    g_model->pos = g_active_sequence->pos;
    return 0;
}
static int ck_model_cancel_requested(void) { return 0; }
static void ck_batch_decode_prefix(Model *model, int token) {
    float *input = (float *)(model->bump + A_EMBEDDED_INPUT);
    for (int i=0;i<32;i++) input[i] = (float)(token+i);
    memcpy(model->bump + A_RESIDUAL, input, 128);
}
static void gemm_nt_q5_1_q8_1_m2(const float *input, const void *weight, const float *bias,
                                  float *output, int rows, int channels, int width) {
    (void)bias;
    float tag = weight == (void *)(g_model->bump + W_LAYER_0_WQ) ? 1.0f :
                weight == (void *)(g_model->bump + W_LAYER_0_WK) ? 2.0f : 3.0f;
    for (int row=0;row<rows;row++) for (int c=0;c<channels;c++) output[row*channels+c] = input[row*width] + tag;
}
static void ck_batch_decode_suffix(Model *model, int token) {
    float *q = (float *)(model->bump + A_Q_SCRATCH);
    float *k = (float *)(model->bump + A_K_SCRATCH);
    model->logits[0] = q[0]; model->logits[1] = k[0]; model->logits[2] = (float)token;
    g_active_sequence->kv[0] = (unsigned char)token;
    g_active_sequence->pos++; model->pos++;
}
static void ck_batch_decode_before_gateup(Model *model, int token) {
    float *input = (float *)(model->bump + A_EMBEDDED_INPUT);
    float *residual = (float *)(model->bump + A_RESIDUAL);
    float *q = (float *)(model->bump + A_Q_SCRATCH);
    float *k = (float *)(model->bump + A_K_SCRATCH);
    input[0] += q[0] + k[0];
    residual[0] += (float)token;
    g_active_sequence->kv[0] = (unsigned char)token;
}
static void ck_batch_decode_after_gateup(Model *model, int token) {
    float *residual = (float *)(model->bump + A_RESIDUAL);
    float *gateup = (float *)(model->bump + A_MLP_SCRATCH);
    model->logits[0] = gateup[0]; model->logits[1] = residual[0];
    model->logits[2] = (float)token;
    g_active_sequence->pos++; model->pos++;
}
'''
        postlude = r'''
int main(void) {
    unsigned char kv[3][64] = {{0}};
    for (int i=0;i<3;i++) g_sequence_states[i] = (CKSequenceStateV8){.occupied=1,.handle=(uint64_t)(i+1),.kv=kv[i]};
    ck_model_sequence_state_activate(1);
    float logits[2][64] = {{0}};
    CKModelBatchDecodeRowV8 rows[2] = {
        {.sequence_handle=1,.token=3,.row_offset=0,.token_count=1,.logits=logits[0]},
        {.sequence_handle=2,.token=7,.row_offset=1,.token_count=1,.logits=logits[1]}
    };
    size_t bytes=0, alignment=0;
    if (ck_model_batch_decode_workspace(&bytes,&alignment) || alignment != 64) return 1;
    void *workspace = aligned_alloc(alignment, (bytes+alignment-1)/alignment*alignment);
    void *third_arena = aligned_alloc(alignment, (bytes+alignment-1)/alignment*alignment);
    if (!workspace || !third_arena) return 2;
    memset(third_arena, 0x5a, bytes);
    g_sequence_states[2].kv = third_arena;
    if (ck_model_decode_batch2(rows,2,third_arena,bytes) != -2) return 3;
    if (((unsigned char *)third_arena)[0] != 0x5a) return 4;
    if (ck_model_decode_batch2(rows,2,workspace,bytes)) return 5;
    if (EXTENDED) {
        if (logits[0][0] != 15 || logits[0][1] != 6 ||
            logits[1][0] != 27 || logits[1][1] != 14) return 6;
    } else {
        if (logits[0][0] != 4 || logits[0][1] != 5 ||
            logits[1][0] != 8 || logits[1][1] != 9) return 6;
    }
    if (kv[0][0] != 3 || kv[1][0] != 7 || ((unsigned char *)third_arena)[0] != 0x5a) return 7;
    free(third_arena);
    free(workspace);
    return 0;
}
'''
        for extended, fixture in ((False, _fixture()), (True, _layer_fixture())):
            with self.subTest(extended=extended), tempfile.TemporaryDirectory() as directory:
                contract = resolve_two_row_batch_contract(*fixture)
                self.assertIsNotNone(contract)
                self.assertEqual("layer_extension" in contract, extended)
                source = Path(directory) / "entry.c"
                binary = Path(directory) / "entry"
                source.write_text(f"#define EXTENDED {int(extended)}\n" + prelude +
                                  emit_two_row_batch_api(contract) + postlude)
                subprocess.run(["cc", "-std=c11", "-O0", "-Wall", "-Werror",
                                "-Wno-unused-function", str(source), "-o", str(binary)],
                               check=True, capture_output=True, text=True)
                subprocess.run([str(binary)], check=True, capture_output=True, text=True)

    def test_loaded_symbol_rejects_replaced_library(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "library.so"
            other = Path(directory) / "replacement.so"
            for target, value in ((path, 11), (other, 22)):
                source = target.with_suffix(".c")
                source.write_text(f"int cke_test_symbol(void) {{ return {value}; }}\n")
                subprocess.run(["cc", "-shared", "-fPIC", str(source), "-o", str(target)],
                               check=True, capture_output=True, text=True)
            loaded = ctypes.CDLL(str(path))
            self.assertEqual(verified_loaded_symbol_backing(loaded, "cke_test_symbol", path)["sha256"],
                             __import__("hashlib").sha256(path.read_bytes()).hexdigest())
            other.replace(path)
            with self.assertRaisesRegex(ValueError, "changed|replaced"):
                verified_loaded_symbol_backing(loaded, "cke_test_symbol", path)

    def test_admits_only_declared_kv_and_map_pair(self) -> None:
        ops, layout, config = _fixture()
        resolved = resolve_two_row_batch_contract(ops, layout, config)
        self.assertEqual(resolved["gemm_function"], "gemm_nt_q5_1_q8_1_m2")
        self.assertEqual(resolved["input_dim"], 32)
        for change in ("extra_state", "changed_weight", "changed_input", "reordered", "oversized_k",
                       "offset_input", "offset_output", "offset_suffix", "wrong_provider", "new_live_value"):
            bad_ops, bad_layout = copy.deepcopy(ops), copy.deepcopy(layout)
            if change == "extra_state":
                bad_layout["memory"]["activations"]["buffers"].append(
                    {"name": "unknown_state", "size": 16, "abs_offset": 704,
                     "lifetime": "sequence", "mutable": True})
            elif change == "changed_weight":
                bad_ops[3]["args"][2]["expr"] = "(const void*)0"
            elif change == "changed_input":
                bad_ops[4]["args"][0]["buffer_ref"] = "residual"
            elif change == "oversized_k":
                bad_ops[3]["args"][-1]["expr"] = "8224"
                bad_ops[4]["args"][-1]["expr"] = "8224"
                bad_layout["memory"]["activations"]["buffers"][1]["size"] = 8224 * 4
            elif change == "offset_input":
                bad_ops[3]["args"][0]["expr"] = "(const float*)(model->bump + A_EMBEDDED_INPUT + 4)"
            elif change == "offset_output":
                bad_ops[4]["args"][1]["expr"] = "(float*)(model->bump + A_K_SCRATCH + 4)"
            elif change == "offset_suffix":
                bad_ops[5]["args"][0]["expr"] = "(const float*)(model->bump + A_EMBEDDED_INPUT + 4)"
            elif change == "wrong_provider":
                bad_ops[4]["call_abi"]["kernel_id"] = "gemv_q5_0"
            elif change == "new_live_value":
                bad_layout["memory"]["activations"]["buffers"].append(
                    {"name": "extra", "size": 16, "abs_offset": 704, "define": "A_EXTRA",
                     "lifetime": "call", "mutable": True})
                bad_ops[2]["args"].append({"name": "extra", "expr": "(float*)(model->bump + A_EXTRA)",
                                              "source": "output:extra", "buffer_ref": "extra"})
                bad_ops[5]["args"].append({"name": "extra", "expr": "(const float*)(model->bump + A_EXTRA)",
                                              "source": "activation:extra", "buffer_ref": "extra"})
            else:
                bad_ops[3], bad_ops[4] = bad_ops[4], bad_ops[3]
            with self.subTest(change=change):
                self.assertIsNone(resolve_two_row_batch_contract(bad_ops, bad_layout, config))

    def test_m2_kernel_portable(self) -> None:
        self._check_m2_kernel([[]])

    def test_m2_kernel_avx2(self) -> None:
        if not _avx2_available():
            self.skipTest("AVX2 is unavailable on this runner")
        self._check_m2_kernel([["-mavx2"]])

    def _check_m2_kernel(self, modes: list[list[str]]) -> None:
        source = ROOT / "src/kernels/gemm_kernels_q5_1_q8_1.c"
        for flags in modes:
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as directory:
                library = Path(directory) / "libq51.so"
                subprocess.run(["cc", "-shared", "-fPIC", "-O2", *flags,
                                "-I", str(ROOT / "include"), str(source),
                                "-lm", "-o", str(library)], check=True,
                               capture_output=True, text=True)
                native = ctypes.CDLL(str(library))
                gemv = native.gemv_q5_1_q8_1
                gemv.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
                                 ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
                batch = native.gemm_nt_q5_1_q8_1_m2
                batch.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
                                  ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                  ctypes.c_int, ctypes.c_int, ctypes.c_int]
                for width, channels, with_bias in ((32, 3, False), (64, 7, True), (96, 5, True)):
                    blocks = width // 32
                    packed = bytearray()
                    specs = []
                    for channel in range(channels):
                        row_specs = []
                        for block_index in range(blocks):
                            scale = (0.125, 0.3125, 0.1875)[(channel + block_index) % 3]
                            minimum = (-0.75, 0.375, -0.25)[(channel * 2 + block_index) % 3]
                            quants = [((channel * 11 + block_index * 7 + i * 13) % 32) for i in range(32)]
                            high = sum(((q >> 4) & 1) << i for i, q in enumerate(quants))
                            nibbles = bytes((quants[i] & 15) | ((quants[i + 16] & 15) << 4)
                                            for i in range(16))
                            packed.extend(struct.pack('<eeI', scale, minimum, high) + nibbles)
                            row_specs.append((scale, minimum, quants))
                        specs.append(row_specs)
                    weights = ctypes.create_string_buffer(bytes(packed))
                    values = [((i * 17) % 47 - 23) / 9 for i in range(2 * width)]
                    inputs = (ctypes.c_float * len(values))(*values)
                    bias_values = [(channel - 2) / 7 for channel in range(channels)]
                    bias = (ctypes.c_float * channels)(*bias_values) if with_bias else None
                    actual = (ctypes.c_float * (2 * channels))()
                    batch(inputs, weights, bias, actual, 2, channels, width)
                    for row in range(2):
                        isolated = (ctypes.c_float * channels)()
                        input_row = ctypes.cast(ctypes.byref(inputs, row * width * 4),
                                                ctypes.POINTER(ctypes.c_float))
                        gemv(isolated, weights, input_row, channels, width)
                        for channel in range(channels):
                            observed = actual[row * channels + channel]
                            if bias is None:
                                self.assertEqual(observed, isolated[channel])
                            else:
                                self.assertAlmostEqual(observed, isolated[channel] + bias[channel], delta=2e-5)
                            oracle = bias_values[channel] if bias is not None else 0.0
                            for block_index, (scale, minimum, quants) in enumerate(specs[channel]):
                                fragment = list(inputs)[row * width + block_index * 32:
                                                        row * width + (block_index + 1) * 32]
                                d = max(abs(v) for v in fragment) / 127.0
                                q8 = [int(math.copysign(math.floor(abs(v / d) + 0.5), v)) if d else 0
                                      for v in fragment]
                                d8 = struct.unpack('<e', struct.pack('<e', d))[0]
                                s8 = struct.unpack('<e', struct.pack('<e', sum(q8) * d))[0]
                                # Independently dequantize the Q5 contribution;
                                # Q8_1's rounded sum is its separate offset term.
                                oracle += (scale * d8) * sum(a * b for a, b in zip(quants, q8)) + minimum * s8
                            self.assertAlmostEqual(observed, oracle, delta=3e-4)


if __name__ == "__main__":
    unittest.main()
