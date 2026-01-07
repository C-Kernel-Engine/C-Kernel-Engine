#include "ckernel_ir.h"
#include "ckernel_ir_v2.h"
#include "ckernel_ir_v2_lower.h"
#include "ckernel_codegen_v2.h"

#include <stdio.h>
#include <string.h>

static void print_usage(const char *argv0)
{
    fprintf(stderr,
            "Usage:\n"
            "  %s /path/to/config.json [--meta weights_meta.json] [--emit out.c] [--emit-lib]\n"
            "  %s --ir /path/to/ir_v2.json [--emit out.c] [--emit-lib]\n"
            "\n"
            "Options:\n"
            "  --emit out.c    Write generated C code to file\n"
            "  --emit-lib      Generate shared library API (no main)\n"
            "  --meta path     Apply weights meta JSON (ties/quant info)\n"
            "  --lower mode    Emit lowered IR JSON (prefill/decode/backward)\n"
            "  --lower-out path  Write lowered IR JSON to path (default build/ir_v2_<mode>.json)\n",
            argv0, argv0);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    CKIRV2Graph graph = {0};

    const char *emit_path = NULL;
    const char *meta_path = NULL;
    const char *lower_mode_str = NULL;
    const char *lower_out = NULL;
    char lower_out_buf[128] = {0};
    CKEmitMode emit_mode = CK_EMIT_STANDALONE;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--emit") == 0 && i + 1 < argc) {
            emit_path = argv[++i];
        } else if (strcmp(argv[i], "--emit-lib") == 0) {
            emit_mode = CK_EMIT_LIBRARY;
        } else if (strcmp(argv[i], "--meta") == 0 && i + 1 < argc) {
            meta_path = argv[++i];
        } else if (strcmp(argv[i], "--lower") == 0 && i + 1 < argc) {
            lower_mode_str = argv[++i];
        } else if (strcmp(argv[i], "--lower-out") == 0 && i + 1 < argc) {
            lower_out = argv[++i];
        }
    }

    if (strcmp(argv[1], "--ir") == 0) {
        if (argc < 3) {
            fprintf(stderr, "Missing IR v2 JSON path after --ir\n");
            return 1;
        }
        const char *ir_path = argv[2];
        if (ck_ir_v2_parse_json(ir_path, &graph) != 0) {
            fprintf(stderr, "Failed to parse IR v2 JSON: %s\n", ir_path);
            return 1;
        }
    } else {
        const char *config_path = argv[1];
        CKModelConfig cfg;
        if (ck_model_config_from_hf_json(config_path, &cfg) != 0) {
            fprintf(stderr, "Failed to parse config.json: %s\n", config_path);
            return 1;
        }
        if (ck_ir_v2_build_decoder(&cfg, &graph) != 0) {
            fprintf(stderr, "Failed to build IR v2 decoder\n");
            return 1;
        }
        if (meta_path) {
            if (ck_ir_v2_apply_meta(meta_path, &graph) != 0) {
                fprintf(stderr, "Failed to apply weights meta: %s\n", meta_path);
                ck_ir_v2_free(&graph);
                return 1;
            }
        }
    }

    printf("[ck_ir_v2_demo] buffers=%d nodes=%d layers=%d\n",
           graph.num_buffers, graph.num_nodes, graph.config.num_layers);

    if (emit_path) {
        const char *mode_str = (emit_mode == CK_EMIT_LIBRARY) ? "library" : "standalone";
        if (ck_codegen_v2_emit_runtime(&graph, emit_path, emit_mode) == 0) {
            fprintf(stderr, "[ck_ir_v2_demo] %s runtime written to %s\n", mode_str, emit_path);
        } else {
            fprintf(stderr, "[ck_ir_v2_demo] failed to write %s runtime to %s\n", mode_str, emit_path);
        }
    }

    if (lower_mode_str) {
        CKIRV2LowerMode mode;
        if (ck_ir_v2_lower_mode_from_string(lower_mode_str, &mode) != 0) {
            fprintf(stderr, "[ck_ir_v2_demo] invalid lower mode: %s\n", lower_mode_str);
            ck_ir_v2_free(&graph);
            return 1;
        }
        if (!lower_out) {
            snprintf(lower_out_buf, sizeof(lower_out_buf), "build/ir_v2_%s.json",
                     ck_ir_v2_lower_mode_name(mode));
            lower_out = lower_out_buf;
        }
        if (mode == CK_IR_V2_LOWER_BACKWARD) {
            CKIRV2Graph backward = {0};
            if (ck_ir_v2_build_decoder_backward(&graph, &backward) != 0) {
                fprintf(stderr, "[ck_ir_v2_demo] failed to build backward graph\n");
                ck_ir_v2_free(&graph);
                return 1;
            }
            if (ck_ir_v2_lower_emit_json(&backward, mode, lower_out) == 0) {
                fprintf(stderr, "[ck_ir_v2_demo] lowered IR v2 JSON written to %s\n", lower_out);
            } else {
                fprintf(stderr, "[ck_ir_v2_demo] failed to write lowered IR v2 JSON to %s\n", lower_out);
            }
            ck_ir_v2_free(&backward);
        } else {
            if (ck_ir_v2_lower_emit_json(&graph, mode, lower_out) == 0) {
                fprintf(stderr, "[ck_ir_v2_demo] lowered IR v2 JSON written to %s\n", lower_out);
            } else {
                fprintf(stderr, "[ck_ir_v2_demo] failed to write lowered IR v2 JSON to %s\n", lower_out);
            }
        }
    }

    if (strcmp(argv[1], "--ir") != 0) {
        if (ck_ir_v2_serialize_json(&graph, "build/ir_v2.json") == 0) {
            fprintf(stderr, "[ck_ir_v2_demo] IR v2 JSON written to build/ir_v2.json\n");
        } else {
            fprintf(stderr, "[ck_ir_v2_demo] Failed to write IR v2 JSON\n");
        }
    }

    ck_ir_v2_free(&graph);
    return 0;
}
