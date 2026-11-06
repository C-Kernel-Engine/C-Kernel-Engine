#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODEGEN_PATH = ROOT / "version" / "v8" / "scripts" / "codegen_core_v8.py"
PREFILL_CODEGEN_PATH = ROOT / "version" / "v8" / "scripts" / "codegen_prefill_v8.py"
sys.path.insert(0, str(CODEGEN_PATH.parent))


def _load_codegen():
    spec = importlib.util.spec_from_file_location("codegen_core_hidden_export_tests", CODEGEN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {CODEGEN_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


codegen = _load_codegen()


def _load_prefill_codegen():
    spec = importlib.util.spec_from_file_location(
        "codegen_prefill_hidden_export_tests", PREFILL_CODEGEN_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {PREFILL_CODEGEN_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prefill_codegen = _load_prefill_codegen()


def _arg(name: str, expr: str) -> dict[str, str]:
    return {"name": name, "expr": expr}


class HiddenExportExtentTests(unittest.TestCase):
    def test_dynamic_prefill_ffn_exports_full_valid_grid(self) -> None:
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("C compiler unavailable")
        ops = [
            ("ffn_norm", "fake_norm", [
                _arg("x", "input"), _arg("out", "normed"),
                _arg("tokens", "num_tokens"), _arg("dim", "4"),
            ]),
            ("mlp_gate_up", "fake_gate_up", [
                _arg("x", "normed"), _arg("out", "gate_up"),
                _arg("M", "num_tokens"), _arg("N", "8"), _arg("K", "4"),
            ]),
            ("geglu", "fake_geglu", [
                _arg("x", "gate_up"), _arg("out", "compact"),
                _arg("tokens", "num_tokens"), _arg("dim", "4"),
            ]),
            ("post_ffn_norm", "fake_post_norm", [
                _arg("x", "compact"), _arg("out", "post"),
                _arg("tokens", "num_tokens"), _arg("dim", "4"),
            ]),
        ]
        emitted = "\n".join(
            prefill_codegen.emit_prefill_op(
                {"op": op, "function": function, "layer": 0, "args": args},
                index, {},
            )
            for index, (op, function, args) in enumerate(ops)
        )
        source = r'''
#include <stddef.h>
#include <string.h>
#define EMBED_DIM 4
static float captured[4][24];
static int seen[4], capture_enabled;
static void fake_norm(const float *x, float *y, int rows, int dim) {
    for (int i = 0; i < rows * dim; ++i) y[i] = 2.0f * x[i] + 1.0f;
}
static void fake_gate_up(const float *x, float *y, int rows, int width, int input_dim) {
    for (int row = 0; row < rows; ++row)
        for (int col = 0; col < width; ++col)
            y[row * width + col] = x[row * input_dim + col % input_dim] + (float)(col * 10);
}
static void fake_geglu(const float *x, float *y, int rows, int dim) {
    for (int row = 0; row < rows; ++row)
        for (int col = 0; col < dim; ++col)
            y[row * dim + col] = x[row * 2 * dim + col] + x[row * 2 * dim + dim + col];
}
static void fake_post_norm(const float *x, float *y, int rows, int dim) {
    for (int i = 0; i < rows * dim; ++i) y[i] = x[i] * 0.5f;
}
static void ck_debug_export_hidden(void *model, int layer, const char *name,
        const float *src, size_t count) {
    (void)model;
    if (!capture_enabled) return;
    const char *names[] = {"ffn_norm", "mlp_gate_up", "mlp_geglu", "post_ffn_norm"};
    const size_t counts[] = {12, 24, 12, 12};
    for (int i = 0; i < 4; ++i) {
        if (strcmp(name, names[i]) == 0) {
            if (layer != 0 || count != counts[i]) { seen[i] = -100; return; }
            memcpy(captured[i], src, count * sizeof(float));
            seen[i]++;
            return;
        }
    }
}
int main(void) {
    const int num_tokens = 3;
    void *model = NULL;
    float input[20], normed[20], gate_up[40], compact[20], post[20], baseline[12];
    for (int i = 0; i < 20; ++i) input[i] = (float)(i + 1);
    for (int pass = 0; pass < 2; ++pass) {
        capture_enabled = pass;
        for (int i = 0; i < 20; ++i) normed[i] = compact[i] = post[i] = -777.0f;
        for (int i = 0; i < 40; ++i) gate_up[i] = -777.0f;
        memset(seen, 0, sizeof(seen));
EMITTED
        if (pass == 0) memcpy(baseline, post, sizeof(baseline));
        else {
            const float *stages[] = {normed, gate_up, compact, post};
            const int counts[] = {12, 24, 12, 12};
            for (int stage = 0; stage < 4; ++stage) {
                if (seen[stage] != 1) return 1;
                for (int i = 0; i < counts[stage]; ++i)
                    if (captured[stage][i] != stages[stage][i]) return 2;
            }
            if (memcmp(baseline, post, sizeof(baseline)) != 0) return 3;
        }
        for (int i = 12; i < 20; ++i)
            if (normed[i] != -777.0f || compact[i] != -777.0f || post[i] != -777.0f) return 4;
        for (int i = 24; i < 40; ++i)
            if (gate_up[i] != -777.0f) return 5;
    }
    return 0;
}
'''.replace("EMITTED", emitted)
        with tempfile.TemporaryDirectory(prefix="prefill_ffn_capture_") as tmpdir:
            path = Path(tmpdir)
            (path / "fixture.c").write_text(source, encoding="utf-8")
            compiled = subprocess.run(
                [compiler, "-std=c11", "-Wall", "-Werror", str(path / "fixture.c"),
                 "-o", str(path / "fixture")],
                capture_output=True, text=True,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            subprocess.run([str(path / "fixture")], check=True, capture_output=True)

    def test_geglu_capture_executes_full_grid_and_last_row(self) -> None:
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("C compiler unavailable")
        source_head = r'''
#include <stddef.h>
#include <string.h>
static float captured[12], captured_last[4];
static int full_seen, last_seen, invalid_capture;
static void geglu_forward_quick_fp16_table(const float *x, float *out, int tokens, int dim) {
    for (int row = 0; row < tokens; ++row)
        for (int col = 0; col < dim; ++col)
            out[row * dim + col] = x[row * 2 * dim + col] + x[row * 2 * dim + dim + col];
}
static void ck_debug_export_hidden(void *model, int layer, const char *name,
        const float *src, size_t count) {
    (void)model;
    if (layer != 0) { invalid_capture = 1; return; }
    if (strcmp(name, "mlp_geglu") == 0 && count == 12) {
        memcpy(captured, src, count * sizeof(float));
        full_seen++;
    } else if (strcmp(name, "mlp_geglu_last") == 0 && count == 4) {
        memcpy(captured_last, src, count * sizeof(float));
        last_seen++;
    } else invalid_capture = 1;
}
int main(void) {
    float input[26], separate[14];
    void *model = NULL;
    for (int alias = 0; alias < 2; ++alias) {
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 8; ++col)
                input[row * 8 + col] = (float)(row * 100 + col);
        input[24] = 12345.0f;
        input[25] = -12345.0f;
        separate[12] = 23456.0f;
        separate[13] = -23456.0f;
        full_seen = last_seen = invalid_capture = 0;
        float *output = alias ? input : separate;
'''
        source_tail = r'''
        if (invalid_capture || full_seen != 1 || last_seen != 1) return 1;
        if (input[24] != 12345.0f || input[25] != -12345.0f ||
                separate[12] != 23456.0f || separate[13] != -23456.0f) return 2;
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 4; ++col) {
                float expected = (float)(row * 200 + 4 + 2 * col);
                int index = row * 4 + col;
                if (output[index] != expected || captured[index] != expected) return 3;
                if (row == 2 && captured_last[col] != expected) return 4;
            }
    }
    return 0;
}
'''
        emitted = codegen.emit_op({
            "op": "geglu",
            "function": "geglu_forward_quick_fp16_table",
            "layer": 0,
            "args": [
                _arg("x", "input"),
                _arg("out", "output"),
                _arg("tokens", "3"),
                _arg("dim", "4"),
            ],
        })
        with tempfile.TemporaryDirectory(prefix="geglu_grid_capture_") as tmpdir:
            path = Path(tmpdir)
            (path / "fixture.c").write_text(source_head + emitted + source_tail, encoding="utf-8")
            subprocess.run(
                [compiler, "-std=c11", str(path / "fixture.c"), "-o", str(path / "fixture")],
                check=True, capture_output=True,
            )
            subprocess.run([str(path / "fixture")], check=True, capture_output=True)

    def test_spatial_merge_exports_only_written_elements(self) -> None:
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("C compiler unavailable")
        source_head = r'''
#include <stddef.h>
#include <string.h>
static int expected_count;
static int capture_ok;
static float captured[32];
static void write_output(float *output) {
    for (int i = 0; i < expected_count; ++i) output[i] = (float)(i + 1);
}
static void spatial_average_pool_contiguous(const float *input, float *output,
        int grid_h, int grid_w, int embed_dim, int merge_size) {
    (void)input; (void)grid_h; (void)grid_w; (void)embed_dim; (void)merge_size;
    write_output(output);
}
static void spatial_merge_contiguous_tiled(const float *input, float *output,
        int grid_h, int grid_w, int embed_dim, int merge_size) {
    (void)input; (void)grid_h; (void)grid_w; (void)embed_dim; (void)merge_size;
    write_output(output);
}
static void spatial_merge_2x2(const float *input, float *output,
        int grid_h, int grid_w, int embed_dim) {
    (void)input; (void)grid_h; (void)grid_w; (void)embed_dim;
    write_output(output);
}
static void ck_debug_export_hidden(void *model, int layer, const char *name,
        const float *src, size_t count) {
    (void)model;
    if (count == 0) return;
    if (layer != -1 || strcmp(name, "vision_spatial_merge") != 0 ||
            count != (size_t)expected_count) return;
    memcpy(captured, src, count * sizeof(float));
    capture_ok = 1;
}
int main(void) {
    float input[30] = {0};
    float output[32] = {0};
    void *model = NULL;
    expected_count = EXPECTED_COUNT;
    output[expected_count] = 12345.0f;
    output[expected_count + 1] = -12345.0f;
'''
        source_tail = r'''
    if (capture_ok != (expected_count > 0) || output[expected_count] != 12345.0f ||
            output[expected_count + 1] != -12345.0f) return 1;
    for (int i = 0; i < expected_count; ++i)
        if (captured[i] != (float)(i + 1)) return 2;
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="spatial_merge_capture_") as tmpdir:
            path = Path(tmpdir)
            for function, expected in (
                ("spatial_average_pool_contiguous", 4),
                ("spatial_merge_2x2", 16),
                ("spatial_merge_contiguous_tiled", 24),
            ):
                for invalid in (False, True):
                    with self.subTest(function=function, invalid=invalid):
                        args = [
                            _arg("input", "input"),
                            _arg("output", "output"),
                            _arg("grid_h", "0" if invalid and function == "spatial_merge_2x2" else "3"),
                            _arg("grid_w", "5"),
                            _arg("embed_dim", "2"),
                        ]
                        if function != "spatial_merge_2x2":
                            args.append(_arg("merge_size", "0" if invalid else "2"))
                        emitted = codegen.emit_op({
                            "op": "spatial_merge",
                            "function": function,
                            "layer": -1,
                            "args": args,
                        })
                        (path / "fixture.c").write_text(
                            source_head.replace("EXPECTED_COUNT", str(0 if invalid else expected)) + emitted + source_tail,
                            encoding="utf-8",
                        )
                        subprocess.run(
                            [compiler, "-std=c11", str(path / "fixture.c"), "-o", str(path / "fixture")],
                            check=True, capture_output=True,
                        )
                        subprocess.run([str(path / "fixture")], check=True, capture_output=True)

    def test_ffn_norm_captures_execute_full_grid(self) -> None:
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("C compiler unavailable")
        source_head = r'''
#include <stddef.h>
#include <string.h>
#define EMBED_DIM 4
static float captured[12];
static int capture_ok;
static void rmsnorm_forward_parallel_dispatch(const float *input, const float *gamma,
        float *output, float *rstd_cache, int tokens, int dim, int stride, float eps) {
    (void)gamma; (void)rstd_cache; (void)eps;
    for (int row = 0; row < tokens; ++row)
        for (int col = 0; col < dim; ++col)
            output[row * stride + col] = input[row * stride + col] + 1.0f;
}
static void ck_debug_export_hidden(void *model, int layer, const char *name,
        const float *src, size_t count) {
    (void)model;
    if (layer == 10 && count == 12 &&
            (strcmp(name, "ffn_norm") == 0 || strcmp(name, "post_ffn_norm") == 0)) {
        memcpy(captured, src, count * sizeof(float));
        capture_ok = 1;
    }
}
int main(void) {
    float input[12], output[14] = {0};
    void *model = NULL;
    for (int i = 0; i < 12; ++i) input[i] = (float)(10 * i);
    output[12] = 12345.0f;
    output[13] = -12345.0f;
'''
        source_tail = r'''
    if (!capture_ok || output[12] != 12345.0f || output[13] != -12345.0f) return 1;
    for (int i = 0; i < 12; ++i)
        if (captured[i] != input[i] + 1.0f) return 2;
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="ffn_norm_grid_capture_") as tmpdir:
            path = Path(tmpdir)
            for name in ("ffn_norm", "post_ffn_norm"):
                with self.subTest(name=name):
                    op = {
                        "op": name,
                        "function": "rmsnorm_forward_parallel_dispatch",
                        "layer": 10,
                        "args": [
                            _arg("input", "input"),
                            _arg("gamma", "NULL"),
                            _arg("output", "output"),
                            _arg("rstd_cache", "NULL"),
                            _arg("tokens", "3"),
                            _arg("d_model", "4"),
                            _arg("aligned_embed_dim", "4"),
                            _arg("eps", "0.000001f"),
                        ],
                    }
                    emitted = codegen.emit_op(op)
                    (path / "fixture.c").write_text(source_head + emitted + source_tail, encoding="utf-8")
                    subprocess.run(
                        [compiler, "-std=c11", str(path / "fixture.c"), "-o", str(path / "fixture")],
                        check=True, capture_output=True,
                    )
                    subprocess.run([str(path / "fixture")], check=True, capture_output=True)

    def test_rmsnorm_capture_executes_full_resolved_grid(self) -> None:
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("C compiler unavailable")
        op = {
            "op": "rmsnorm",
            "function": "rmsnorm_forward_parallel_dispatch",
            "layer": 2,
            "args": [
                _arg("input", "input"),
                _arg("gamma", "NULL"),
                _arg("output", "output"),
                _arg("rstd_cache", "NULL"),
                _arg("tokens", "3"),
                _arg("d_model", "4"),
                _arg("aligned_embed_dim", "4"),
                _arg("eps", "0.000001f"),
            ],
        }
        source_head = r'''
#include <stddef.h>
#include <string.h>
#define EMBED_DIM 4
static float captured[12];
static int capture_ok;
static void rmsnorm_forward_parallel_dispatch(const float *input, const float *gamma,
        float *output, float *rstd_cache, int tokens, int dim, int stride, float eps) {
    (void)gamma; (void)rstd_cache; (void)eps;
    for (int row = 0; row < tokens; ++row)
        for (int col = 0; col < dim; ++col)
            output[row * stride + col] = input[row * stride + col] + 1.0f;
}
static void ck_debug_export_hidden(void *model, int layer, const char *name,
        const float *src, size_t count) {
    (void)model;
    if (layer == 2 && count == 12 &&
            (strcmp(name, "block_rmsnorm") == 0 || strcmp(name, "ffn_norm") == 0)) {
        memcpy(captured, src, count * sizeof(float));
        capture_ok = 1;
    }
}
int main(void) {
    float input[12], output[14] = {0};
    void *model = NULL;
    for (int i = 0; i < 12; ++i) input[i] = (float)i;
    output[12] = 12345.0f;
    output[13] = -12345.0f;
'''
        source_tail = r'''
    if (!capture_ok || output[12] != 12345.0f || output[13] != -12345.0f) return 1;
    for (int i = 0; i < 12; ++i)
        if (captured[i] != input[i] + 1.0f) return 2;
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="rmsnorm_grid_capture_") as tmpdir:
            path = Path(tmpdir)
            for index in (0, 1):
                emitted = codegen.emit_op(op, op_instance_idx=index)
                (path / "fixture.c").write_text(source_head + emitted + source_tail, encoding="utf-8")
                subprocess.run(
                    [compiler, "-std=c11", str(path / "fixture.c"), "-o", str(path / "fixture")],
                    check=True, capture_output=True,
                )
                subprocess.run([str(path / "fixture")], check=True, capture_output=True)

    def test_v_norm_parity_capture_uses_resolved_head_major_geometry(self) -> None:
        op = {
            "op": "v_norm",
            "function": "gemma4_v_norm_forward_parallel_dispatch",
            "layer": 1,
            "args": [
                _arg("input", "V_INPUT"),
                _arg("output", "V_OUTPUT"),
                _arg("tokens", "3"),
                _arg("num_kv_heads", "2"),
                _arg("head_dim", "4"),
                _arg("eps", "0.000001f"),
            ],
        }
        for emitted, rows in (
            (codegen.emit_op(op, dump=True), "3"),
            (prefill_codegen.emit_prefill_op(op, 0, {}, dump=True), "num_tokens"),
        ):
            self.assertIn(
                'ck_dump_tensor_head_major_token_major((float*)V_OUTPUT, 1, '
                f'"Vcur_normed", 2, {rows}, 4);',
                emitted,
            )
            self.assertIn('"v_norm", (const float*)V_OUTPUT,', emitted)

    def test_v_norm_prefill_capture_executes_all_rows_without_overrun(self) -> None:
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("C compiler unavailable")
        op = {
            "op": "v_norm",
            "function": "gemma4_v_norm_forward_parallel_dispatch",
            "layer": 1,
            "args": [
                _arg("input", "input"),
                _arg("output", "output"),
                _arg("rstd_cache", "NULL"),
                _arg("tokens", "3"),
                _arg("num_kv_heads", "2"),
                _arg("head_dim", "4"),
                _arg("eps", "0.000001f"),
            ],
        }
        emitted = prefill_codegen.emit_prefill_op(op, 0, {}, dump=True)
        source = r'''
#include <stddef.h>
#include <string.h>
static float captured[24];
static float hidden[24];
static int capture_ok;
static int hidden_ok;
static void ck_debug_export_hidden(void *model, int layer, const char *name,
        const float *src, size_t count) {
    (void)model;
    hidden_ok = layer == 1 && strcmp(name, "v_norm") == 0 && count == 24;
    if (hidden_ok) memcpy(hidden, src, count * sizeof(float));
}
static void gemma4_v_norm_forward_parallel_dispatch(const float *input, float *output,
        float *rstd_cache, int tokens, int heads, int dim, float eps) {
    (void)rstd_cache; (void)eps;
    for (int i = 0; i < tokens * heads * dim; ++i) output[i] = input[i] + 1.0f;
}
static void ck_dump_tensor_head_major_token_major(const float *src, int layer,
        const char *name, int heads, int tokens, int dim) {
    capture_ok = layer == 1 && strcmp(name, "Vcur_normed") == 0
            && heads == 2 && tokens == 3 && dim == 4;
    for (int t = 0; t < tokens; ++t)
        for (int h = 0; h < heads; ++h)
            for (int d = 0; d < dim; ++d)
                captured[(t * heads + h) * dim + d] = src[(h * tokens + t) * dim + d];
}
int main(void) {
    float input[24], output[26] = {0};
    int num_tokens = 3;
    void *model = NULL;
    output[24] = 12345.0f;
    output[25] = -12345.0f;
    for (int h = 0; h < 2; ++h)
        for (int t = 0; t < 3; ++t)
            for (int d = 0; d < 4; ++d)
                input[(h * 3 + t) * 4 + d] = (float)(100 * h + 10 * t + d);
''' + emitted + r'''
    if (!capture_ok || !hidden_ok || output[24] != 12345.0f || output[25] != -12345.0f) return 1;
    for (int t = 0; t < 3; ++t)
        for (int h = 0; h < 2; ++h)
            for (int d = 0; d < 4; ++d)
                if (captured[(t * 2 + h) * 4 + d] != (float)(100 * h + 10 * t + d + 1)) return 2;
    for (int i = 0; i < 24; ++i)
        if (hidden[i] != output[i]) return 3;
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="v_norm_capture_") as tmpdir:
            path = Path(tmpdir)
            (path / "fixture.c").write_text(source, encoding="utf-8")
            subprocess.run(
                [compiler, "-std=c11", "-DCK_PARITY_DUMP", str(path / "fixture.c"),
                 "-o", str(path / "fixture")], check=True, capture_output=True,
            )
            subprocess.run([str(path / "fixture")], check=True, capture_output=True)

    def test_compact_kv_export_executes_with_mixed_head_widths(self) -> None:
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("C compiler unavailable")
        source = codegen.emit_model_and_api(
            layout={"memory": {"activations": {"buffers": []}}},
            config={
                "num_layers": 2,
                "num_kv_heads": 2,
                "head_dim": 4,
                "decode_kv_cache_dtype": "fp16",
                "layer_k_cache_offset": [0, 8],
                "layer_v_cache_offset": [4, 16],
                "layer_k_head_dim": [2, 4],
                "layer_v_head_dim": [2, 4],
            },
        )
        declarations = source[
            source.index("static const int64_t g_layer_k_cache_offset") :
            source.index("};", source.index("static const int32_t g_layer_v_head_dim")) + 2
        ]
        exporter = source[
            source.index("CK_EXPORT int ck_model_debug_export_kv(") :
            source.index("/* Compatibility ABI for existing FP16-only X-ray consumers. */")
        ]
        harness = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define CK_EXPORT
#define NUM_LAYERS 2
#define NUM_KV_HEADS 2
#define MAX_SEQ_LEN 4
typedef struct { uint16_t *kv_cache_f16; int pos; } TestModel;
static TestModel *g_model;
""" + declarations + "\n" + exporter + r"""
int main(int argc, char **argv) {
    if (argc != 3) return 10;
    uint16_t cache[24 * MAX_SEQ_LEN];
    for (size_t i = 0; i < sizeof(cache) / sizeof(cache[0]); ++i) cache[i] = 0xeeee;
    const int widths[2] = {2, 4};
    const int offsets[2][2] = {{0, 4}, {8, 16}};
    for (int layer = 0; layer < 2; ++layer)
        for (int kind = 0; kind < 2; ++kind)
            for (int head = 0; head < 2; ++head)
                for (int token = 0; token < 3; ++token)
                    for (int channel = 0; channel < widths[layer]; ++channel)
                        cache[offsets[layer][kind] * MAX_SEQ_LEN
                              + head * MAX_SEQ_LEN * widths[layer]
                              + token * widths[layer] + channel] =
                            1000 * layer + 400 * kind + 100 * head + 10 * token + channel;
    TestModel model = {cache, 3};
    g_model = &model;
    if (ck_model_debug_export_kv(argv[1], 0)) return 11;
    if (ck_model_debug_export_kv(argv[2], 1)) return 12;
    return 0;
}
"""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            c_path = base / "export.c"
            executable = base / "export"
            c_path.write_text(harness, encoding="ascii")
            subprocess.run(
                [compiler, "-std=c11", "-Wall", "-Wextra", "-Werror", str(c_path), "-o", str(executable)],
                check=True, capture_output=True, text=True,
            )
            paths = [base / "layer0.bin", base / "layer1.bin"]
            subprocess.run([str(executable), *(str(path) for path in paths)], check=True)
            for layer, (path, width) in enumerate(zip(paths, (2, 4))):
                payload = path.read_bytes()
                self.assertEqual(len(payload), 32 + 2 * 2 * 3 * width * 2)
                self.assertEqual(
                    struct.unpack_from("<8I", payload),
                    (0x564B5843, 2, layer, 3, 2, 4, width, 1),
                )
                values = struct.unpack_from(f"<{2 * 2 * 3 * width}H", payload, 32)
                expected = tuple(
                    1000 * layer + 400 * kind + 100 * head + 10 * token + channel
                    for kind in range(2)
                    for head in range(2)
                    for token in range(3)
                    for channel in range(width)
                )
                self.assertEqual(values, expected)

    def test_compact_kv_export_uses_each_layer_head_width(self) -> None:
        source = codegen.emit_model_and_api(
            layout={"memory": {"activations": {"buffers": []}}},
            config={
                "num_layers": 2,
                "num_kv_heads": 2,
                "head_dim": 512,
                "decode_kv_cache_dtype": "fp16",
                "layer_k_cache_offset": [0, 1024],
                "layer_v_cache_offset": [512, 2048],
                "layer_k_head_dim": [256, 512],
                "layer_v_head_dim": [256, 512],
            },
        )

        self.assertIn("g_layer_k_head_dim[NUM_LAYERS] = {256, 512}", source)
        self.assertIn("g_layer_v_head_dim[NUM_LAYERS] = {256, 512}", source)
        self.assertIn("const size_t cache_head_dim = (size_t)(g_layer_k_head_dim[layer]);", source)
        self.assertIn("(uint32_t)cache_head_dim, UINT32_C(1)", source)
        self.assertIn("(size_t)MAX_SEQ_LEN * cache_head_dim", source)
        self.assertIn("(size_t)g_model->pos * cache_head_dim * sizeof(uint16_t)", source)
        self.assertIn("g_layer_k_head_dim[layer] != g_layer_v_head_dim[layer]", source)

    def test_compact_kv_export_rejects_incomplete_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "per-layer K/V dimensions"):
            codegen.emit_model_and_api(
                layout={"memory": {"activations": {"buffers": []}}},
                config={
                    "num_layers": 2,
                    "num_kv_heads": 2,
                    "head_dim": 512,
                    "layer_k_cache_offset": [0, 1024],
                    "layer_v_cache_offset": [512, 2048],
                    "layer_k_head_dim": [256],
                    "layer_v_head_dim": [256, 512],
                },
            )

    def test_compact_kv_export_rejects_incomplete_offsets(self) -> None:
        base = {
            "num_layers": 2,
            "num_kv_heads": 2,
            "head_dim": 4,
            "layer_k_head_dim": [2, 4],
            "layer_v_head_dim": [2, 4],
        }
        for metadata in (
            {"layer_k_cache_offset": [0], "layer_v_cache_offset": [4, 16]},
            {"layer_k_cache_offset": [0, 8]},
            {"layer_k_cache_offset": "invalid", "layer_v_cache_offset": [4, 16]},
            {"kv_cache_layer_stride_variable": True},
        ):
            with self.subTest(metadata=metadata):
                with self.assertRaisesRegex(ValueError, "complete per-layer K/V offsets"):
                    codegen.emit_model_and_api(
                        layout={"memory": {"activations": {"buffers": []}}},
                        config={**base, **metadata},
                    )

    def test_compact_kv_export_rejects_missing_dimensions(self) -> None:
        base = {
            "num_layers": 2,
            "num_kv_heads": 2,
            "head_dim": 4,
            "layer_k_cache_offset": [0, 8],
            "layer_v_cache_offset": [4, 16],
        }
        for metadata in (
            {},
            {"layer_k_head_dim": [2, 4]},
            {"layer_k_head_dim": "invalid", "layer_v_head_dim": [2, 4]},
        ):
            with self.subTest(metadata=metadata):
                with self.assertRaisesRegex(ValueError, "per-layer K/V dimensions"):
                    codegen.emit_model_and_api(
                        layout={"memory": {"activations": {"buffers": []}}},
                        config={**base, **metadata},
                    )

    def test_fixed_width_kv_export_remains_supported(self) -> None:
        source = codegen.emit_model_and_api(
            layout={"memory": {"activations": {"buffers": []}}},
            config={"num_layers": 2, "num_kv_heads": 2, "head_dim": 4},
        )
        self.assertIn("const size_t cache_head_dim = (size_t)(HEAD_DIM);", source)
        self.assertNotIn("g_layer_k_cache_offset[NUM_LAYERS]", source)

    def test_q_only_geometry_controls_prefill_layout_bridges(self) -> None:
        ops = [
            {
                "op": "transpose_qkv_to_head_major",
                "function": "transpose_inplace",
                "layer": 24,
                "args": [],
            },
            {
                "op": "q_norm",
                "function": "q_norm_forward",
                "layer": 24,
                "args": [
                    _arg("q", "Q"),
                    _arg("num_heads", "8"),
                    _arg("num_tokens", "1024"),
                    _arg("head_dim", "256"),
                ],
            },
            {
                "op": "transpose_attn_out_to_token_major",
                "function": "transpose_inplace",
                "layer": 24,
                "args": [],
            },
        ]

        prefill_codegen._annotate_kv_transpose_roles(ops)
        q_bridge = prefill_codegen.emit_prefill_op(
            ops[0], 1, {"num_heads": 8, "head_dim": 512}
        )
        out_bridge = prefill_codegen.emit_prefill_op(
            ops[2], 2, {"num_heads": 8, "head_dim": 512}
        )

        self.assertIn("const int H = 8;", q_bridge)
        self.assertIn("const int D = 256;", q_bridge)
        self.assertIn("const int H = 8;", out_bridge)
        self.assertIn("const int D = 256;", out_bridge)

    def test_conflicting_prefill_geometry_is_rejected(self) -> None:
        ops = [
            {
                "op": "q_norm",
                "layer": 24,
                "args": [_arg("num_heads", "8"), _arg("head_dim", "256")],
            },
            {
                "op": "rope_q",
                "layer": 24,
                "args": [_arg("num_heads", "8"), _arg("head_dim", "512")],
            },
        ]

        with self.assertRaisesRegex(
            ValueError, "conflicting resolved head_dim for layer 24"
        ):
            prefill_codegen._annotate_kv_transpose_roles(ops)

    def test_prefill_attention_export_covers_every_token_row(self) -> None:
        emitted = prefill_codegen.emit_prefill_op(
            {
                "function": "quantize_row_q8_k",
                "op": "quantize_out_proj_input",
                "layer": 2,
                "args": [
                    _arg("x", "ATTN"),
                    _arg("y", "ATTN_Q8"),
                    _arg("k", "4096"),
                    _arg("rows", "4096"),
                ],
                "resolved_codegen_capability": {
                    "operator_family": "activation_quantization",
                    "function": "quantize_row_q8_k",
                    "output_storage": {
                        "format": "q8_k",
                        "block_elements": 256,
                        "block_bytes": 292,
                        "block_elements_symbol": "QK_K",
                        "c_block_type": "block_q8_K",
                    },
                    "prefill_batch": {
                        "function": "quantize_batch_q8_k_4row_nearest_even_parallel_dispatch",
                        "row_group": 4,
                        "tail_function": "quantize_row_q8_k",
                        "rounding_contract": "llama_repack_q8_k_4row_nearest_even",
                    },
                },
            },
            16,
            {"embed_dim": 4096},
        )

        self.assertIn(
            '"attn_out", (const float*)(ATTN), '
            "(int)((size_t)num_tokens * (size_t)(4096))",
            emitted,
        )
        self.assertIn('"attn_out_last"', emitted)

    def test_fused_q4_gateup_swiglu_exports_materialized_output(self) -> None:
        gate = {
            "op": "mlp_gate_up",
            "function": "gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch",
            "layer": 0,
            "resolved_execution": {
                "numerical_contract": "q4_k_x_q8_k_repacked_matmul_fp32",
                "implementation": {
                    "weight_storage": {
                        "format": "q4_k",
                        "block_elements": 256,
                        "block_bytes": 144,
                    },
                    "activation_storage": {"format": "q8_k", "block_elements": 256},
                    "diagnostic_providers": {
                        "fp32_activation": "gemm_nt_q4_k",
                        "row_quantized": "gemv_q4_k_q8_k",
                    },
                },
                "reference": {
                    "function": "gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch"
                },
                "production": {
                    "function": "gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch"
                },
            },
            "args": [
                _arg("A", "A"),
                _arg("B", "B"),
                _arg("bias", "NULL"),
                _arg("C", "OUT"),
                _arg("M", "num_tokens"),
                _arg("N", "34816"),
                _arg("K", "5120"),
            ],
        }
        swiglu = {
            "op": "silu_mul",
            "function": "swiglu_forward_ggml",
            "layer": 0,
            "args": [_arg("output", "OUT")],
        }

        emitted = prefill_codegen._emit_prefill_q4_gateup_swiglu_x16(
            gate,
            swiglu,
            23,
            "fused_23",
            {},
            debug_flag_name="debug_gate",
            debug_input_name="debug_input",
        )

        self.assertIn(
            'if (fused_23) ck_debug_export_hidden(model, 0, "mlp_swiglu", '
            "(const float*)OUT, (num_tokens) * (((34816) / 2)))",
            emitted,
        )

    def test_parity_dump_captures_only_distinct_layer_input_edge(self) -> None:
        op = {
            "op": "residual_save",
            "function": "memcpy",
            "layer": 0,
            "args": [
                _arg("dst", "RESIDUAL"),
                _arg("src", "LAYER_INPUT"),
                _arg("size", "20480"),
            ],
        }
        layer_entry = codegen.emit_op(op, dump=True, op_instance_idx=0)
        after_attention = codegen.emit_op(op, dump=True, op_instance_idx=1)

        self.assertIn(
            'ck_dump_tensor((const float*)(LAYER_INPUT), 0, "layer_input", '
            "((size_t)(20480)) / sizeof(float))",
            layer_entry,
        )
        self.assertNotIn("ck_dump_tensor", after_attention)

    def test_mlp_projection_exports_cover_all_rows_and_channels(self) -> None:
        up = codegen.emit_op(
            {
                "op": "mlp_up",
                "function": "gemm_nt_bf16",
                "layer": 1,
                "args": [
                    _arg("a", "A"),
                    _arg("b", "B"),
                    _arg("bias", "BIAS"),
                    _arg("c", "UP"),
                    _arg("m", "4032"),
                    _arg("n", "4304"),
                    _arg("k", "1152"),
                ],
            }
        )
        down = codegen.emit_op(
            {
                "op": "mlp_down",
                "function": "gemm_nt_bf16",
                "layer": 1,
                "args": [
                    _arg("a", "UP"),
                    _arg("b", "B"),
                    _arg("bias", "BIAS"),
                    _arg("c", "DOWN"),
                    _arg("m", "4032"),
                    _arg("n", "1152"),
                    _arg("k", "4304"),
                ],
            }
        )

        self.assertIn('"mlp_up", (const float*)UP, (4032) * (4304)', up)
        self.assertIn('"mlp_up_last"', up)
        self.assertIn('(size_t)(4304)', up)
        self.assertIn('"mlp_down", (const float*)DOWN, (4032) * (1152)', down)
        self.assertIn('"mlp_down_last"', down)
        self.assertIn('(size_t)(1152)', down)

    def test_gelu_exports_the_full_post_activation_tensor(self) -> None:
        emitted = codegen.emit_op(
            {
                "op": "gelu",
                "function": "gelu_ggml_inplace",
                "layer": 1,
                "args": [_arg("data", "UP"), _arg("n", "17353728")],
            }
        )

        self.assertIn('"ffn_gelu", (const float*)UP, 17353728', emitted)

    def test_attention_norm_exports_the_projection_input_boundary(self) -> None:
        emitted = codegen.emit_op(
            {
                "op": "attn_norm",
                "function": "rmsnorm_forward",
                "layer": 0,
                "args": [
                    _arg("input", "X"),
                    _arg("weight", "W"),
                    _arg("output", "Y"),
                    _arg("bias", "NULL"),
                    _arg("num_tokens", "1"),
                    _arg("dim", "1024"),
                    _arg("aligned_dim", "1024"),
                    _arg("eps", "1e-6f"),
                ],
            }
        )

        self.assertIn('"attn_norm", (const float*)Y, (1) * (1024)', emitted)

    def test_recurrent_scalar_projections_export_every_output_channel(self) -> None:
        for op_name, label in (
            ("recurrent_alpha_proj", "alpha"),
            ("recurrent_beta_proj", "beta"),
        ):
            emitted = codegen.emit_op(
                {
                    "op": op_name,
                    "function": "gemm_nt_f32_llama_production",
                    "layer": 1,
                    "args": [
                        _arg("a", "A"),
                        _arg("b", "B"),
                        _arg("bias", "NULL"),
                        _arg("c", "OUT"),
                        _arg("m", "1"),
                        _arg("n", "48"),
                        _arg("k", "5120"),
                    ],
                }
            )

            self.assertIn(f'"{label}", (const float*)OUT, (1) * (48)', emitted)

    def test_recurrent_output_projection_exports_rows_times_output_dim(self) -> None:
        emitted = codegen.emit_op(
            {
                "op": "recurrent_out_proj",
                "function": "gemm_nt_bf16_pytorch_onednn_3_12_brgemm_bf16_storage",
                "layer": 1,
                "args": [
                    _arg("A", "A"),
                    _arg("B", "B"),
                    _arg("bias", "NULL"),
                    _arg("C", "OUT"),
                    _arg("M", "num_tokens"),
                    _arg("N", "256"),
                    _arg("K", "6144"),
                ],
            }
        )

        self.assertIn(
            '"linear_attn_out", (const float*)OUT, (num_tokens) * (256)',
            emitted,
        )

    def test_qwen35_attention_exports_gate_and_pregate_boundaries(self) -> None:
        split = codegen.emit_op(
            {
                "op": "split_q_gate",
                "function": "split_q_gate_forward",
                "layer": 3,
                "args": [
                    _arg("packed_qg", "PACKED"),
                    _arg("q", "Q"),
                    _arg("gate", "GATE"),
                    _arg("rows", "1"),
                    _arg("q_dim", "2048"),
                    _arg("gate_dim", "2048"),
                ],
            }
        )
        attention = codegen.emit_op(
            {
                "op": "attn",
                "function": "attention_forward_decode_head_major_gqa_ggml_regular",
                "layer": 3,
                "args": [
                    _arg("q_token", "Q"),
                    _arg("k_cache", "K"),
                    _arg("v_cache", "V"),
                    _arg("out_token", "ATTN"),
                    _arg("num_heads", "8"),
                    _arg("num_tokens", "1"),
                    _arg("aligned_head_dim", "256"),
                ],
            }
        )
        sliding_attention = codegen.emit_op(
            {
                "op": "attn_sliding",
                "function": "attention_forward_causal_head_major_gqa_flash_strided_sliding_gemma4_token_output",
                "layer": 3,
                "args": [
                    _arg("output", "SLIDING_ATTN"),
                    _arg("num_heads", "8"),
                    _arg("q_tokens", "61"),
                    _arg("aligned_head_dim", "256"),
                ],
            }
        )
        qsa_attention = codegen.emit_op(
            {
                "op": "qsa_attention",
                "function": "attention_forward_sparse_token_major_gqa_bf16cache_pytorch_cpu_flash_contract",
                "layer": 3,
                "args": [
                    _arg("output", "QSA_ATTN"),
                    _arg("query_heads", "8"),
                    _arg("rows", "1"),
                    _arg("head_dim", "256"),
                ],
            }
        )
        shared_attention = codegen.emit_op(
            {
                "op": "attn_shared_kv",
                "function": "attention_forward_causal_head_major_gqa_flash_strided_gemma4_token_output",
                "layer": 24,
                "args": [
                    _arg("output", "SHARED_ATTN"),
                    _arg("num_heads", "8"),
                    _arg("q_tokens", "61"),
                    _arg("aligned_head_dim", "512"),
                ],
            }
        )
        sliding_shared_attention = codegen.emit_op(
            {
                "op": "attn_sliding_shared_kv",
                "function": "attention_forward_causal_head_major_gqa_flash_strided_sliding_gemma4_token_output",
                "layer": 24,
                "args": [
                    _arg("output", "SLIDING_SHARED_ATTN"),
                    _arg("num_heads", "8"),
                    _arg("num_tokens", "61"),
                    _arg("aligned_head_dim", "256"),
                ],
            }
        )
        gated = codegen.emit_op(
            {
                "op": "attn_gate_sigmoid_mul",
                "function": "attn_gate_sigmoid_mul_forward",
                "layer": 3,
                "args": [
                    _arg("x", "ATTN"),
                    _arg("gate", "GATE"),
                    _arg("out", "ATTN"),
                    _arg("rows", "18"),
                    _arg("num_heads", "8"),
                    _arg("state_dim", "256"),
                ],
            }
        )

        self.assertIn('"attn_gate", (const float*)GATE, (1) * (2048)', split)
        self.assertIn('"attn_pregate", (const float*)ATTN, (8) * (1) * (256)', attention)
        self.assertIn(
            '"attn_pregate", (const float*)SLIDING_ATTN, (8) * (61) * (256)',
            sliding_attention,
        )
        self.assertIn(
            '"attn_pregate", (const float*)QSA_ATTN, (8) * (1) * (256)',
            qsa_attention,
        )
        self.assertIn(
            '"attn_pregate", (const float*)SHARED_ATTN, (8) * (61) * (512)',
            shared_attention,
        )
        self.assertIn(
            '"attn_pregate", (const float*)SLIDING_SHARED_ATTN, (8) * (61) * (256)',
            sliding_shared_attention,
        )
        self.assertIn('"attn_out", (const float*)ATTN, (8) * (18) * (256)', gated)

    def test_mla_exports_each_semantic_attention_boundary(self) -> None:
        gate = codegen.emit_op(
            {
                "op": "attention_gate_projection",
                "function": "gemm_nt_bf16",
                "layer": 0,
                "args": [
                    _arg("output", "GATE"),
                    _arg("rows", "30"),
                    _arg("N", "2048"),
                ],
            }
        )
        kv_a = codegen.emit_op(
            {
                "op": "kv_a_proj",
                "function": "gemm_nt_bf16",
                "layer": 0,
                "args": [
                    _arg("output", "KV_A"),
                    _arg("M", "30"),
                    _arg("N", "576"),
                ],
            }
        )
        kv_norm = codegen.emit_op(
            {
                "op": "kv_a_layernorm",
                "function": "rmsnorm_forward_kv_lora",
                "layer": 0,
                "args": [
                    _arg("output", "KV_NORM"),
                    _arg("tokens", "30"),
                    _arg("d_model", "512"),
                ],
            }
        )
        decompress = codegen.emit_op(
            {
                "op": "kv_lora_decompress",
                "function": "deepseek_mla_kv_decompress_f32",
                "layer": 0,
                "args": [
                    _arg("k_nope", "K_NOPE"),
                    _arg("value", "VALUE"),
                    _arg("tokens", "30"),
                    _arg("heads", "16"),
                    _arg("qk_nope_dim", "128"),
                    _arg("v_dim", "128"),
                ],
            }
        )
        rope = codegen.emit_op(
            {
                "op": "partial_rope_concat",
                "function": "deepseek_mla_partial_rope_concat_packed_f32",
                "layer": 0,
                "args": [
                    _arg("query", "QUERY"),
                    _arg("key", "KEY"),
                    _arg("tokens", "30"),
                    _arg("heads", "16"),
                    _arg("qk_nope_dim", "128"),
                    _arg("qk_rope_dim", "64"),
                ],
            }
        )
        attention = codegen.emit_op(
            {
                "op": "mla_attention",
                "function": "deepseek_mla_attention_f32",
                "layer": 0,
                "args": [
                    _arg("output", "CONTEXT"),
                    _arg("num_tokens", "30"),
                    _arg("num_heads", "16"),
                    _arg("v_head_dim", "128"),
                ],
            }
        )

        self.assertIn('"attn_gate", (const float*)GATE, (30) * (2048)', gate)
        self.assertIn('"mla_kv_a", (const float*)KV_A, (30) * (576)', kv_a)
        self.assertIn('"mla_kv_norm", (const float*)KV_NORM, (30) * (512)', kv_norm)
        self.assertIn('"mla_k_nope", (const float*)K_NOPE, (30) * (16) * (128)', decompress)
        self.assertIn('"mla_value", (const float*)VALUE, (30) * (16) * (128)', decompress)
        self.assertIn('"mla_query", (const float*)QUERY, (30) * (16) * ((128 + 64))', rope)
        self.assertIn('"mla_key", (const float*)KEY, (30) * (16) * ((128 + 64))', rope)
        self.assertIn('"mla_context", (const float*)CONTEXT, (30) * (16) * (128)', attention)

    def test_kimi_block_norm_exports_distinguish_attention_and_ffn_inputs(self) -> None:
        first = codegen.emit_op(
            {
                "op": "block_rmsnorm",
                "function": "rmsnorm_forward",
                "layer": 0,
                "args": [
                    _arg("input", "ATTN_INPUT"),
                    _arg("output", "ATTN_NORM"),
                    _arg("rows", "30"),
                ],
            },
            op_instance_idx=0,
        )
        second = codegen.emit_op(
            {
                "op": "block_rmsnorm",
                "function": "rmsnorm_forward",
                "layer": 0,
                "args": [
                    _arg("input", "FFN_INPUT"),
                    _arg("output", "FFN_NORM"),
                    _arg("rows", "30"),
                ],
            },
            op_instance_idx=1,
        )

        self.assertIn('"block_rmsnorm", (const float*)ATTN_NORM', first)
        self.assertNotIn('"ffn_input"', first)
        self.assertNotIn('"ffn_norm"', first)
        self.assertIn('"ffn_input", (const float*)FFN_INPUT', second)
        self.assertIn('"ffn_norm", (const float*)FFN_NORM', second)
        self.assertNotIn('"block_rmsnorm"', second)

    def test_moe_exports_router_and_expert_boundaries(self) -> None:
        router = codegen.emit_op(
            {
                "op": "moe_router",
                "function": "gemm_blocked_serial",
                "layer": 1,
                "args": [
                    _arg("y", "ROUTER"),
                    _arg("M", "30"),
                    _arg("N", "64"),
                ],
            }
        )
        selection = codegen.emit_op(
            {
                "op": "group_limited_topk_router",
                "function": "group_limited_topk_router_sigmoid_f32",
                "layer": 1,
                "args": [
                    _arg("indices", "SELECTED"),
                    _arg("weights", "ROUTING"),
                    _arg("rows", "30"),
                    _arg("top_k", "6"),
                ],
            }
        )
        routed = codegen.emit_op(
            {
                "op": "moe_swiglu_packed_expert_mlp",
                "function": "moe_swiglu_packed_expert_forward_bf16",
                "layer": 1,
                "args": [
                    _arg("output", "ROUTED"),
                    _arg("rows", "30"),
                    _arg("hidden_dim", "2048"),
                ],
            }
        )
        combined = codegen.emit_op(
            {
                "op": "shared_swiglu_expert_mlp",
                "function": "moe_swiglu_shared_forward_bf16",
                "layer": 1,
                "args": [
                    _arg("output", "COMBINED"),
                    _arg("rows", "30"),
                    _arg("hidden_dim", "2048"),
                ],
            }
        )
        self.assertIn('"moe_router_logits", (const float*)ROUTER, (30) * (64)', router)
        self.assertIn(
            '"moe_selected_experts", (const int32_t*)SELECTED, (30) * (6)',
            selection,
        )
        self.assertIn('"moe_routing_weights", (const float*)ROUTING, (30) * (6)', selection)
        self.assertIn('"moe_routed_output", (const float*)ROUTED, (30) * (2048)', routed)
        self.assertIn('"moe_combined_output", (const float*)COMBINED, (30) * (2048)', combined)

    def test_qwen4_composite_stream_boundaries_are_observable(self) -> None:
        mix = codegen.emit_op(
            {
                "op": "hyper_mix_attn",
                "function": "hyper_connection_mix_bf16",
                "layer": 1,
                "args": [
                    _arg("mixed_output", "MIXED"),
                    _arg("injection_output", "INJECT"),
                    _arg("normalized_scratch", "NORMALIZED"),
                    _arg("dynamic_scratch", "DYNAMIC"),
                    _arg("mix_scratch", "GATE"),
                    _arg("rows", "1"),
                    _arg("streams", "4"),
                    _arg("hidden_dim", "256"),
                    _arg("dynamic_dim", "64"),
                ],
            }
        )
        inject = codegen.emit_op(
            {
                "op": "hyper_inject_mlp",
                "function": "hyper_stream_inject_bf16",
                "layer": 1,
                "args": [
                    _arg("output", "HYPER"),
                    _arg("rows", "1"),
                    _arg("streams", "4"),
                    _arg("hidden_dim", "256"),
                ],
            }
        )
        ple = codegen.emit_op(
            {
                "op": "ple_gate_conv_inject",
                "function": "qwen4_ple_gate_conv_inject_bf16",
                "layer": 1,
                "args": [
                    _arg("hyper_output", "PLE_OUT"),
                    _arg("key_norm_scratch", "KEY_NORM"),
                    _arg("query_norm_scratch", "QUERY_NORM"),
                    _arg("gated_scratch", "GATED"),
                    _arg("conv_norm_scratch", "CONV_NORM"),
                    _arg("rows", "1"),
                    _arg("streams", "4"),
                    _arg("hidden_dim", "256"),
                ],
            }
        )

        self.assertIn('"attn_hyper_norm", (const float*)NORMALIZED, (1) * (4) * (256)', mix)
        self.assertIn('"attn_hyper_dynamic", (const float*)DYNAMIC, (1) * (64)', mix)
        self.assertIn('"attn_hyper_gate", (const float*)GATE, (1) * (4) * (256)', mix)
        self.assertIn('"attn_mixed_input", (const float*)MIXED, (1) * (256)', mix)
        self.assertIn('"attn_injection_weights", (const float*)INJECT, (1) * (4)', mix)
        self.assertIn('"layer_out", (const float*)HYPER, (1) * (4) * (256)', inject)
        self.assertIn('"ple_key_normed", (const float*)KEY_NORM, (1) * (4) * (256)', ple)
        self.assertIn('"ple_query_normed", (const float*)QUERY_NORM, (1) * (4) * (256)', ple)
        self.assertIn('"ple_gated_value", (const float*)GATED, (1) * (4) * (256)', ple)
        self.assertIn('"ple_conv_normed", (const float*)CONV_NORM, (1) * (4) * (256)', ple)
        self.assertIn('"ple_layer_out", (const float*)PLE_OUT, (1) * (4) * (256)', ple)

    def test_bf16_logits_export_covers_the_vocabulary(self) -> None:
        gemv = codegen.emit_op(
            {
                "op": "logits",
                "function": "gemv_bf16_parallel_dispatch",
                "layer": -1,
                "args": [
                    _arg("output", "LOGITS"),
                    _arg("weights", "WEIGHTS"),
                    _arg("input", "HIDDEN"),
                    _arg("M", "248320"),
                    _arg("K", "256"),
                ],
            }
        )
        self.assertIn(
            'ck_debug_export_hidden(model, -1, "logits", '
            "(const float*)LOGITS, 248320)",
            gemv,
        )

        gemm = codegen.emit_op(
            {
                "op": "logits",
                "function": "gemm_nt_bf16_pytorch_onednn_3_12_brgemm_bf16_storage",
                "layer": -1,
                "args": [
                    _arg("A", "HIDDEN"),
                    _arg("B", "WEIGHTS"),
                    _arg("bias", "NULL"),
                    _arg("C", "LOGITS"),
                    _arg("M", "1"),
                    _arg("N", "248320"),
                    _arg("K", "256"),
                ],
            }
        )
        self.assertIn(
            'ck_debug_export_hidden(model, -1, "logits", '
            "(const float*)LOGITS, 248320)",
            gemm,
        )

    def test_bf16_projection_exports_use_provider_shape_convention(self) -> None:
        gemv = codegen.emit_op(
            {
                "op": "v_proj",
                "function": "gemv_bf16_parallel_dispatch",
                "layer": 3,
                "args": [
                    _arg("output", "VALUE"),
                    _arg("weights", "WEIGHTS"),
                    _arg("input", "HIDDEN"),
                    _arg("M", "512"),
                    _arg("K", "256"),
                ],
            }
        )
        self.assertIn(
            'ck_debug_export_hidden(model, 3, "v_proj", '
            "(const float*)VALUE, 512)",
            gemv,
        )

        gemm = codegen.emit_op(
            {
                "op": "v_proj",
                "function": "gemm_nt_bf16_pytorch_onednn_3_12_brgemm_bf16_storage",
                "layer": 3,
                "args": [
                    _arg("A", "HIDDEN"),
                    _arg("B", "WEIGHTS"),
                    _arg("bias", "NULL"),
                    _arg("C", "VALUE"),
                    _arg("M", "4"),
                    _arg("N", "512"),
                    _arg("K", "256"),
                ],
            }
        )
        self.assertIn(
            'ck_debug_export_hidden(model, 3, "v_proj", '
            "(const float*)VALUE, (4) * (512))",
            gemm,
        )

    def test_qwen4_ple_embedding_export_uses_history_ngram_count(self) -> None:
        emitted = codegen.emit_op(
            {
                "op": "ple_ngram_embed",
                "function": "qwen4_ple_ngram_embed_bf16",
                "layer": 1,
                "args": [
                    _arg("output", "PLE_EMBED"),
                    _arg("rows", "2"),
                    _arg("ngram_size", "3"),
                    _arg("heads_per_ngram", "2"),
                    _arg("head_dim", "64"),
                ],
            }
        )

        self.assertIn(
            '"ple_embedding", (const float*)PLE_EMBED, (2) * (((3) - 1)) * (2) * (64)',
            emitted,
        )

    def test_farskip_combine_exports_both_persistent_streams(self) -> None:
        emitted = codegen.emit_op(
            {
                "op": "farskip_routed_shared_combine",
                "function": "farskip_swiglu_shared_combine_bf16",
                "layer": 1,
                "args": [
                    _arg("main_output", "MAIN"),
                    _arg("routed_free_output", "ROUTED_FREE"),
                    _arg("rows", "30"),
                    _arg("hidden_dim", "2048"),
                ],
            }
        )

        self.assertIn('"layer_out", (const float*)MAIN, (30) * (2048)', emitted)
        self.assertIn(
            '"routed_free_out", (const float*)ROUTED_FREE, (30) * (2048)',
            emitted,
        )

    def test_attention_checkpoint_name_comes_from_call_ir_contract(self) -> None:
        emitted = codegen.emit_op(
            {
                "op": "attn",
                "function": "attention_forward",
                "layer": 2,
                "args": [
                    _arg("out_token", "ATTN"),
                    _arg("num_heads", "16"),
                    _arg("num_tokens", "1008"),
                    _arg("aligned_head_dim", "72"),
                ],
                "semantic_checkpoints": [
                    {
                        "id": "vision.layer.2.attention.output",
                        "tensor": "attn_out_head_major",
                    }
                ],
            }
        )

        self.assertIn(
            '"attn_out_head_major", (const float*)ATTN, (16) * (1008) * (72)',
            emitted,
        )
        self.assertNotIn('"attn_pregate"', emitted)

    def test_qwen35_prefill_exports_full_norm_and_swiglu_extents(self) -> None:
        norm = codegen.emit_op(
            {
                "op": "post_attention_norm",
                "function": "rmsnorm_forward_llama_production",
                "layer": 3,
                "args": [
                    _arg("output", "NORM"),
                    _arg("tokens", "18"),
                    _arg("d_model", "1024"),
                ],
            }
        )
        swiglu = codegen.emit_op(
            {
                "op": "silu_mul",
                "function": "swiglu_forward_ggml",
                "layer": 3,
                "args": [
                    _arg("output", "MLP"),
                    _arg("tokens", "18"),
                    _arg("dim", "3584"),
                ],
            }
        )
        self.assertIn('"post_attn_norm", (const float*)NORM, (18) * (1024)', norm)
        self.assertIn('"mlp_swiglu", (const float*)MLP, (18) * (3584)', swiglu)

    def test_recurrent_core_exports_state_before_in_place_update(self) -> None:
        op = {
            "op": "recurrent_core",
            "function": "gated_deltanet_forward",
            "layer": 4,
            "args": [
                _arg("state_in", "STATE"),
                _arg("state_out", "STATE"),
                _arg("num_heads", "48"),
                _arg("state_dim", "128"),
                _arg("tokens", "1"),
                _arg("output", "OUTPUT"),
            ],
        }
        decode = codegen.emit_op(op)
        prefill = prefill_codegen.emit_prefill_op(op, 1, {"embed_dim": 5120})

        expected = '"state_predelta", (const float*)STATE, (48) * (128) * (128)'
        self.assertIn(expected, decode)
        self.assertIn(expected, prefill)
        self.assertLess(decode.index('"state_predelta"'), decode.index("gated_deltanet_forward("))
        self.assertLess(prefill.index('"state_predelta"'), prefill.index("gated_deltanet_forward("))

    def test_sliding_attention_prefill_exports_complete_attention_output(self) -> None:
        emitted = prefill_codegen.emit_prefill_op(
            {
                "op": "attn_sliding",
                "function": "attention_forward_causal_head_major_gqa_flash_strided_sliding_gemma4_token_output",
                "layer": 0,
                "args": [
                    _arg("output", "SLIDING_ATTN"),
                    _arg("num_heads", "8"),
                    _arg("num_tokens", "num_tokens"),
                    _arg("aligned_head_dim", "256"),
                ],
            },
            61,
            {"embed_dim": 2560},
        )

        self.assertIn(
            '"attn_pregate", (const float*)SLIDING_ATTN, '
            "(num_tokens) * (8) * (256)",
            emitted,
        )

    def test_split_rope_exports_complete_runtime_extent(self) -> None:
        cases = (
            ("rope_q", "Q", "num_heads", "8", "rope_q"),
            ("rope_k", "K", "num_kv_heads", "2", "rope_k"),
        )
        for op_name, tensor, heads_name, heads, label in cases:
            with self.subTest(op=op_name):
                op = {
                    "op": op_name,
                    "function": f"{op_name}_forward",
                    "layer": 24,
                    "args": [
                        _arg("q" if op_name == "rope_q" else "k", tensor),
                        _arg(heads_name, heads),
                        _arg("num_tokens", "1024"),
                        _arg("aligned_head_dim", "256"),
                    ],
                }

                decode = codegen.emit_op(op)
                prefill = prefill_codegen.emit_prefill_op(
                    op, 61, {"embed_dim": 2560}
                )

                self.assertIn(
                    f'"{label}", (const float*){tensor}, '
                    f"({heads}) * (1024) * (256)",
                    decode,
                )
                self.assertIn(
                    f'"{label}", (const float*){tensor}, '
                    f"({heads}) * (num_tokens) * (256)",
                    prefill,
                )
                self.assertIn(f'"{label}_last"', prefill)

    def test_joint_qk_norm_exports_all_heads_and_tokens(self) -> None:
        op = {
            "op": "qk_norm",
            "function": "qk_norm_forward_parallel_dispatch",
            "layer": 0,
            "args": [
                _arg("q", "Q"),
                _arg("k", "K"),
                _arg("num_heads", "3"),
                _arg("num_kv_heads", "2"),
                _arg("num_tokens", "7"),
                _arg("head_dim", "4"),
            ],
        }
        emitted = codegen.emit_op(op)
        self.assertIn('"qk_norm_q", (const float*)Q, (3) * (7) * (4)', emitted)
        self.assertIn('"qk_norm_k", (const float*)K, (2) * (7) * (4)', emitted)
        without_tokens = dict(op, args=[arg for arg in op["args"] if arg["name"] != "num_tokens"])
        with self.assertRaisesRegex(ValueError, "token extent"):
            codegen.emit_op(without_tokens)

    def test_split_qk_norm_exports_complete_runtime_extent(self) -> None:
        cases = (
            ("q_norm", "Q", "num_heads", "8", "qk_norm_q"),
            ("k_norm", "K", "num_kv_heads", "2", "qk_norm_k"),
        )
        for op_name, tensor, heads_name, heads, label in cases:
            with self.subTest(op=op_name):
                op = {
                    "op": op_name,
                    "function": f"{op_name}_forward",
                    "layer": 24,
                    "args": [
                        _arg("q" if op_name == "q_norm" else "k", tensor),
                        _arg(heads_name, heads),
                        _arg("num_tokens", "1024"),
                        _arg("aligned_head_dim", "256"),
                    ],
                }

                decode = codegen.emit_op(op)
                prefill = prefill_codegen.emit_prefill_op(
                    op, 61, {"embed_dim": 2560}
                )

                self.assertIn(
                    f'"{label}", (const float*){tensor}, '
                    f"({heads}) * (1024) * (256)",
                    decode,
                )
                self.assertIn(
                    f'"{label}", (const float*){tensor}, '
                    f"({heads}) * (num_tokens) * (256)",
                    prefill,
                )
                self.assertIn(f'"{label}_last"', prefill)

    def test_shared_attention_prefill_uses_runtime_segment_extent(self) -> None:
        for op_name in ("attn_shared_kv", "attn_sliding_shared_kv"):
            with self.subTest(op=op_name):
                emitted = prefill_codegen.emit_prefill_op(
                    {
                        "op": op_name,
                        "function": "attention_forward_causal_head_major_gqa_prefill_append_f16cache_gemma4_workspace",
                        "layer": 24,
                        "args": [
                            _arg("output", "SHARED_ATTN"),
                            _arg("num_heads", "8"),
                            _arg("q_tokens", "1024"),
                            _arg("aligned_head_dim", "256"),
                        ],
                    },
                    61,
                    {"embed_dim": 2560},
                )

                self.assertIn(
                    '"attn_pregate", (const float*)SHARED_ATTN, '
                    "(num_tokens) * (8) * (256)",
                    emitted,
                )

    def test_gemma4_prepared_input_exports_complete_runtime_extent(self) -> None:
        op = {
            "op": "gemma4_per_layer_prepare",
            "function": "gemma4_per_layer_prepare_forward",
            "layer": -1,
            "args": [
                _arg("output", "PREPARED"),
                _arg("per_layer_dim", "256"),
            ],
        }
        emitted = prefill_codegen.emit_prefill_op(op, 1, {"embed_dim": 2560})
        self.assertIn(
            '"gemma4_per_layer_prepare", (const float*)PREPARED, '
            "(num_tokens) * (NUM_LAYERS * (256))",
            emitted,
        )
        self.assertIn('"gemma4_per_layer_prepare_last"', emitted)

    def test_gemma4_prepared_input_export_executes_full_stream(self) -> None:
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("C compiler unavailable")
        op = {
            "op": "gemma4_per_layer_prepare",
            "function": "gemma4_per_layer_prepare_forward",
            "layer": -1,
            "args": [_arg("output", "PREPARED"), _arg("per_layer_dim", "256")],
        }
        emitted = prefill_codegen.emit_prefill_op(op, 1, {"embed_dim": 2560})
        source = r"""
#include <stddef.h>
#include <string.h>
#define NUM_LAYERS 2
#define PREPARED (&prepared[1])
typedef struct { int calls; } Model;
static float prepared[3 * NUM_LAYERS * 256 + 2];
static void gemma4_per_layer_prepare_forward(float *out, int width) {
    if (width != 256) __builtin_trap();
    for (int token = 0; token < 3; ++token)
        for (int layer = 0; layer < NUM_LAYERS; ++layer)
            for (int channel = 0; channel < 256; ++channel)
                out[(token * NUM_LAYERS + layer) * 256 + channel] =
                    (float)(token * 10000 + layer * 1000 + channel);
}
static void ck_debug_export_hidden(Model *model, int layer, const char *label,
                                    const float *data, int count) {
    int last = model->calls++;
    if (layer != -1 || count != (last ? NUM_LAYERS * 256 : 3 * NUM_LAYERS * 256))
        __builtin_trap();
    if (strcmp(label, last ? "gemma4_per_layer_prepare_last" :
                            "gemma4_per_layer_prepare") != 0) __builtin_trap();
    for (int i = 0; i < count; ++i) {
        int index = last ? i + 2 * NUM_LAYERS * 256 : i;
        int token = index / (NUM_LAYERS * 256);
        int layer_index = (index / 256) % NUM_LAYERS;
        int channel = index % 256;
        if (data[i] != (float)(token * 10000 + layer_index * 1000 + channel))
            __builtin_trap();
    }
}
int main(void) {
    Model storage = {0};
    Model *model = &storage;
    int num_tokens = 3;
    prepared[0] = -1234.0f;
    prepared[3 * NUM_LAYERS * 256 + 1] = -5678.0f;
""" + emitted + r"""
    if (storage.calls != 2 || prepared[0] != -1234.0f ||
        prepared[3 * NUM_LAYERS * 256 + 1] != -5678.0f) return 1;
    return 0;
}
"""
        with tempfile.TemporaryDirectory() as directory:
            c_path = Path(directory) / "prepared.c"
            executable = Path(directory) / "prepared"
            c_path.write_text(source, encoding="ascii")
            subprocess.run(
                [compiler, "-std=c11", "-Wall", "-Wextra", "-Werror",
                 str(c_path), "-o", str(executable)],
                check=True, capture_output=True, text=True,
            )
            subprocess.run([str(executable)], check=True)

    def test_gemma4_layer_output_exports_complete_runtime_extent(self) -> None:
        op = {
            "op": "gemma4_per_layer_embed",
            "function": "gemma4_per_layer_embed_forward",
            "layer": 23,
            "args": [
                _arg("hidden", "HIDDEN"),
                _arg("tokens", "1024"),
            ],
        }

        decode = codegen.emit_op(op)
        decode_dump = codegen.emit_op(op, dump=True)
        prefill = prefill_codegen.emit_prefill_op(op, 61, {"embed_dim": 2560})
        prefill_dump = prefill_codegen.emit_prefill_op(
            op, 61, {"embed_dim": 2560}, dump=True
        )

        self.assertIn(
            '"gemma4_per_layer_embed", (const float*)HIDDEN, (1024) * EMBED_DIM',
            decode,
        )
        self.assertIn(
            '"gemma4_per_layer_embed", (const float*)HIDDEN, '
            "(num_tokens) * (EMBED_DIM)",
            prefill,
        )
        self.assertIn(
            'ck_dump_tensor((float*)HIDDEN, 23, "gemma4_per_layer_embed", '
            "(1024) * (EMBED_DIM));",
            decode_dump,
        )
        self.assertIn(
            'ck_dump_tensor((float*)HIDDEN, 23, "gemma4_per_layer_embed", '
            "(num_tokens) * (2560));",
            prefill_dump,
        )

    def test_quantized_projection_exports_full_prefill_extents(self) -> None:
        resolved = {
            "numerical_contract": "q4_k_x_q8_k_repacked_matmul_fp32",
            "implementation": {
                "weight_storage": {"format": "q4_k", "block_elements": 256, "block_bytes": 144},
                "activation_storage": {"format": "q8_k", "block_elements": 256},
                "diagnostic_providers": {"fp32_activation": "gemm_nt_q4_k"},
            }
        }
        emitted = codegen.emit_op(
            {
                "op": "out_proj",
                "function": "gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch",
                "layer": 3,
                "resolved_execution": resolved,
                "args": [
                    _arg("A", "Q8"),
                    _arg("B", "WEIGHT"),
                    _arg("C", "OUT"),
                    _arg("M", "18"),
                    _arg("N", "1024"),
                    _arg("K", "2048"),
                ],
            }
        )
        self.assertIn('"out_proj", (const float*)OUT, (18) * (1024)', emitted)

    def test_fused_gate_up_prefill_exports_combined_row_major_matrix(self) -> None:
        emitted = codegen.emit_op(
            {
                "op": "mlp_gate_up",
                "function": "gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch",
                "layer": 3,
                "args": [
                    _arg("A", "Q8"),
                    _arg("B", "WEIGHT"),
                    _arg("C", "GATE_UP"),
                    _arg("M", "18"),
                    _arg("N", "7168"),
                    _arg("K", "1024"),
                ],
            }
        )
        self.assertIn('"mlp_gate_up", (const float*)GATE_UP, (18) * (7168)', emitted)
        self.assertIn('"mlp_gate_up_last"', emitted)
        self.assertIn('(size_t)(7168)', emitted)


if __name__ == "__main__":
    unittest.main()
