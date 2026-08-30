/*
 * ck_v4.c - C-Kernel-Engine v4 CLI (builder/orchestrator)
 *
 * Usage:
 *   ck-v4 build <model|config.json> [--modes=prefill,decode]
 *   ck-v4 gen <model|config.json>   # generate only (no compile)
 *
 * This CLI uses the v4 pipeline:
 *   - scripts/ck_download.py (download config/tokenizer)
 *   - scripts/v4/convert_hf_to_bump_v4.py (weights -> BUMPWGT4)
 *   - scripts/v4/build_ir_v4.py (IR v4 + codegen)
 */

#include <ctype.h>
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CK_V4_VERSION "0.1.0"
#define CK_CACHE_DIR ".cache/ck-engine/models"
#define MAX_PATH 4096
#define MAX_CMD  8192

typedef enum {
    INPUT_LOCAL_PATH,
    INPUT_HF_MODEL_ID,
} InputType;

typedef struct {
    char input[MAX_PATH];
    char config_path[MAX_PATH];
    char cache_dir[MAX_PATH];
    char weights_path[MAX_PATH];
    char weights_manifest[MAX_PATH];
    char output_dir[MAX_PATH];
    char model_id[512];
    char model_name[256];
    char modes[128];
    char checkpoint_dir[MAX_PATH];
    char prefix_dir[MAX_PATH];
    bool force_download;
    bool force_convert;
    bool no_compile;
    bool verbose;
} CKV4Config;

static void print_usage(void) {
    printf("C-Kernel-Engine v4 CLI (ck-v4) v%s\n", CK_V4_VERSION);
    printf("\nUsage:\n");
    printf("  ck-v4 build <model|config.json> [options]\n");
    printf("  ck-v4 gen   <model|config.json> [options]\n");
    printf("  ck-v4 help\n");
    printf("\nOptions:\n");
    printf("  --modes=LIST        Comma list (prefill,decode) [default: prefill,decode]\n");
    printf("  --prefix=DIR        Output directory for generated v4 files\n");
    printf("  --checkpoint=DIR    HF checkpoint directory (for local config)\n");
    printf("  --force-download    Re-download model files\n");
    printf("  --force-convert     Re-convert weights to BUMPWGT4\n");
    printf("  --no-compile        Skip compilation step\n");
    printf("  --verbose           Verbose output\n");
    printf("\nExamples:\n");
    printf("  ck-v4 build Qwen/Qwen2-0.5B\n");
    printf("  ck-v4 build ./config.json --checkpoint ./model_dir\n");
    printf("  ck-v4 gen Qwen/Qwen2-0.5B --modes=prefill\n");
}

static bool file_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static bool dir_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static const char *get_home_dir(void) {
    const char *home = getenv("HOME");
    return home ? home : ".";
}

static int run_cmd(const char *cmd, bool verbose) {
    if (verbose) {
        printf("[cmd] %s\n", cmd);
    }
    int ret = system(cmd);
    if (ret != 0) {
        fprintf(stderr, "Command failed: %s\n", cmd);
    }
    return ret;
}

static const char *detect_simd_flags(void) {
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) return "-march=native";

    char line[4096];
    bool has_avx512f = false, has_avx2 = false, has_fma = false;
    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, "flags")) {
            if (strstr(line, "avx512f")) has_avx512f = true;
            if (strstr(line, "avx2")) has_avx2 = true;
            if (strstr(line, "fma")) has_fma = true;
        }
    }
    fclose(f);

    if (has_avx512f) return "-mavx512f -mfma";
    if (has_avx2) return "-mavx2 -mfma";
    if (has_fma) return "-mfma";
    return "-march=native";
}

static bool json_get_string(const char *json, const char *key, char *out, size_t out_size) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char *pos = strstr(json, pattern);
    if (!pos) return false;
    pos += strlen(pattern);
    while (*pos == ' ' || *pos == '\t' || *pos == '\n') pos++;
    if (*pos != '"') return false;
    pos++;
    const char *end = strchr(pos, '"');
    if (!end) return false;
    size_t len = (size_t)(end - pos);
    if (len >= out_size) len = out_size - 1;
    memcpy(out, pos, len);
    out[len] = '\0';
    return true;
}

static void model_id_to_name(const char *model_id, char *out, size_t out_size) {
    size_t j = 0;
    bool prev_us = false;
    for (size_t i = 0; model_id[i] && j + 1 < out_size; ++i) {
        char c = model_id[i];
        if (c == '/') {
            continue;
        }
        if (isalnum((unsigned char)c)) {
            out[j++] = (char)tolower((unsigned char)c);
            prev_us = false;
        } else {
            if (!prev_us) {
                out[j++] = '_';
                prev_us = true;
            }
        }
    }
    while (j > 0 && out[j - 1] == '_') j--;
    out[j] = '\0';
}

static void path_dirname(const char *path, char *out, size_t out_size) {
    strncpy(out, path, out_size - 1);
    out[out_size - 1] = '\0';
    char *slash = strrchr(out, '/');
    if (slash) {
        *slash = '\0';
    } else {
        strncpy(out, ".", out_size - 1);
        out[out_size - 1] = '\0';
    }
}

static InputType detect_input_type(const char *input) {
    if (file_exists(input)) return INPUT_LOCAL_PATH;
    return INPUT_HF_MODEL_ID;
}

static int download_model(CKV4Config *cfg) {
    const char *script_paths[] = {
        "./scripts/ck_download.py",
        "../scripts/ck_download.py",
        "scripts/ck_download.py",
        NULL
    };
    const char *script = NULL;
    for (int i = 0; script_paths[i]; i++) {
        if (file_exists(script_paths[i])) {
            script = script_paths[i];
            break;
        }
    }
    if (!script) {
        fprintf(stderr, "Download script not found (scripts/ck_download.py)\n");
        return -1;
    }

    char cmd[MAX_CMD];
    snprintf(cmd, sizeof(cmd),
             "python3 %s --model '%s' --cache-dir '%s/%s' --no-convert --quiet %s 2>&1",
             script, cfg->input, get_home_dir(), CK_CACHE_DIR,
             cfg->force_download ? "--force" : "");

    if (cfg->verbose) printf("[cmd] %s\n", cmd);

    FILE *fp = popen(cmd, "r");
    if (!fp) {
        fprintf(stderr, "Failed to run download script\n");
        return -1;
    }

    char json_output[16384] = {0};
    size_t total = 0;
    char buf[1024];
    while (fgets(buf, sizeof(buf), fp) && total < sizeof(json_output) - 1) {
        size_t len = strlen(buf);
        if (total + len < sizeof(json_output)) {
            strcat(json_output, buf);
            total += len;
        }
    }
    int ret = pclose(fp);
    if (ret != 0) {
        fprintf(stderr, "Download script failed\n");
        return -1;
    }

    if (!json_get_string(json_output, "config_path", cfg->config_path, sizeof(cfg->config_path)) ||
        !json_get_string(json_output, "cache_dir", cfg->cache_dir, sizeof(cfg->cache_dir)) ||
        !json_get_string(json_output, "model_id", cfg->model_id, sizeof(cfg->model_id))) {
        fprintf(stderr, "Failed to parse download output\n");
        return -1;
    }

    return 0;
}

static int convert_weights_v4(CKV4Config *cfg) {
    const char *script_paths[] = {
        "./scripts/v4/convert_hf_to_bump_v4.py",
        "../scripts/v4/convert_hf_to_bump_v4.py",
        "scripts/v4/convert_hf_to_bump_v4.py",
        "./scripts/convert_hf_to_bump_v4.py",
        "../scripts/convert_hf_to_bump_v4.py",
        "scripts/convert_hf_to_bump_v4.py",
        NULL
    };
    const char *script = NULL;
    for (int i = 0; script_paths[i]; i++) {
        if (file_exists(script_paths[i])) {
            script = script_paths[i];
            break;
        }
    }
    if (!script) {
        fprintf(stderr, "convert_hf_to_bump_v4.py not found\n");
        return -1;
    }

    bool weights_ok = file_exists(cfg->weights_path);
    bool manifest_ok = file_exists(cfg->weights_manifest);
    if (!cfg->force_convert && weights_ok && manifest_ok) {
        if (cfg->verbose) {
            printf("[info] weights already converted: %s\n", cfg->weights_path);
        }
        return 0;
    }
    if (!cfg->force_convert && weights_ok && !manifest_ok && cfg->verbose) {
        printf("[info] weights manifest missing, re-running conversion: %s\n", cfg->weights_manifest);
    }

    char cmd[MAX_CMD];
    snprintf(cmd, sizeof(cmd),
             "python3 %s --checkpoint '%s' --output '%s' --dtype float32 --manifest-out '%s'",
             script, cfg->checkpoint_dir, cfg->weights_path, cfg->weights_manifest);
    return run_cmd(cmd, cfg->verbose);
}

static int build_ir_v4(CKV4Config *cfg) {
    const char *script_paths[] = {
        "./scripts/v4/build_ir_v4.py",
        "../scripts/v4/build_ir_v4.py",
        "scripts/v4/build_ir_v4.py",
        "./scripts/build_ir_v4.py",
        "../scripts/build_ir_v4.py",
        "scripts/build_ir_v4.py",
        NULL
    };
    const char *script = NULL;
    for (int i = 0; script_paths[i]; i++) {
        if (file_exists(script_paths[i])) {
            script = script_paths[i];
            break;
        }
    }
    if (!script) {
        fprintf(stderr, "build_ir_v4.py not found\n");
        return -1;
    }

    char cmd[MAX_CMD];
    snprintf(cmd, sizeof(cmd),
             "python3 %s --config='%s' --name='%s' --prefix='%s' "
             "--dtype=fp32 --fusion=off --emit=lib --modes=%s --weights-manifest='%s'",
             script, cfg->config_path, cfg->model_name, cfg->output_dir, cfg->modes,
             cfg->weights_manifest);
    return run_cmd(cmd, cfg->verbose);
}

static int compile_mode(const char *project_root,
                        const char *output_dir,
                        const char *model_name,
                        const char *mode,
                        bool verbose) {
    char gen_c[MAX_PATH];
    char out_so[MAX_PATH];

    snprintf(gen_c, sizeof(gen_c), "%s/generated_%s_%s.c", output_dir, model_name, mode);
    snprintf(out_so, sizeof(out_so), "%s/libmodel_%s.so", output_dir, mode);

    if (!file_exists(gen_c)) {
        fprintf(stderr, "Generated C not found: %s\n", gen_c);
        return -1;
    }

    char cmd[MAX_CMD];
    const char *simd = detect_simd_flags();
    snprintf(cmd, sizeof(cmd),
             "cc -O3 -fPIC -shared -fopenmp %s -I%s/include -I%s "
             "-o %s %s -L%s/build -lckernel_engine -Wl,-rpath,%s/build -lm",
             simd, project_root, output_dir,
             out_so, gen_c, project_root, project_root);
    return run_cmd(cmd, verbose);
}

static bool parse_args(int argc, char **argv, CKV4Config *cfg) {
    if (argc < 2) return false;
    if (strcmp(argv[1], "help") == 0) return false;
    if (argc < 3) return false;

    memset(cfg, 0, sizeof(*cfg));
    strncpy(cfg->input, argv[2], sizeof(cfg->input) - 1);
    strncpy(cfg->modes, "prefill,decode", sizeof(cfg->modes) - 1);

    if (strcmp(argv[1], "gen") == 0) cfg->no_compile = true;

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--force-download") == 0) cfg->force_download = true;
        else if (strcmp(argv[i], "--force-convert") == 0) cfg->force_convert = true;
        else if (strcmp(argv[i], "--no-compile") == 0) cfg->no_compile = true;
        else if (strcmp(argv[i], "--verbose") == 0) cfg->verbose = true;
        else if (strncmp(argv[i], "--modes=", 8) == 0) {
            strncpy(cfg->modes, argv[i] + 8, sizeof(cfg->modes) - 1);
        } else if (strncmp(argv[i], "--prefix=", 9) == 0) {
            strncpy(cfg->prefix_dir, argv[i] + 9, sizeof(cfg->prefix_dir) - 1);
        } else if (strncmp(argv[i], "--checkpoint=", 13) == 0) {
            strncpy(cfg->checkpoint_dir, argv[i] + 13, sizeof(cfg->checkpoint_dir) - 1);
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv) {
    CKV4Config cfg;
    if (!parse_args(argc, argv, &cfg)) {
        print_usage();
        return 1;
    }

    InputType input_type = detect_input_type(cfg.input);
    if (input_type == INPUT_LOCAL_PATH) {
        strncpy(cfg.config_path, cfg.input, sizeof(cfg.config_path) - 1);
        path_dirname(cfg.config_path, cfg.cache_dir, sizeof(cfg.cache_dir));
        if (cfg.checkpoint_dir[0] == '\0') {
            strncpy(cfg.checkpoint_dir, cfg.cache_dir, sizeof(cfg.checkpoint_dir) - 1);
        }
        strncpy(cfg.model_id, cfg.config_path, sizeof(cfg.model_id) - 1);
    } else {
        if (download_model(&cfg) != 0) return 1;
        if (cfg.checkpoint_dir[0] == '\0') {
            strncpy(cfg.checkpoint_dir, cfg.cache_dir, sizeof(cfg.checkpoint_dir) - 1);
        }
    }

    if (cfg.model_id[0] == '\0') {
        strncpy(cfg.model_id, cfg.input, sizeof(cfg.model_id) - 1);
    }
    model_id_to_name(cfg.model_id, cfg.model_name, sizeof(cfg.model_name));

    if (cfg.prefix_dir[0]) {
        strncpy(cfg.output_dir, cfg.prefix_dir, sizeof(cfg.output_dir) - 1);
    } else {
        snprintf(cfg.output_dir, sizeof(cfg.output_dir), "%s/v4", cfg.cache_dir);
    }
    snprintf(cfg.weights_path, sizeof(cfg.weights_path), "%s/weights_v4.bump", cfg.cache_dir);
    snprintf(cfg.weights_manifest, sizeof(cfg.weights_manifest), "%s/weights_v4.manifest.json", cfg.cache_dir);

    if (!dir_exists(cfg.output_dir)) {
        if (mkdir(cfg.output_dir, 0755) != 0 && errno != EEXIST) {
            fprintf(stderr, "Failed to create output dir: %s\n", cfg.output_dir);
            return 1;
        }
    }

    if (convert_weights_v4(&cfg) != 0) return 1;
    if (build_ir_v4(&cfg) != 0) return 1;

    if (cfg.no_compile) {
        printf("[ok] Generated v4 files in %s\n", cfg.output_dir);
        return 0;
    }

    /* Find project root */
    const char *roots[] = {".", "..", "../..", NULL};
    const char *project_root = NULL;
    char probe[MAX_PATH];
    for (int i = 0; roots[i]; i++) {
        snprintf(probe, sizeof(probe), "%s/include/ckernel_engine.h", roots[i]);
        if (file_exists(probe)) {
            project_root = roots[i];
            break;
        }
    }
    if (!project_root) {
        fprintf(stderr, "Project root not found (include/ckernel_engine.h missing)\n");
        return 1;
    }

    /* Build core library if missing */
    snprintf(probe, sizeof(probe), "%s/build/libckernel_engine.so", project_root);
    if (!file_exists(probe)) {
        char cmd[MAX_CMD];
        snprintf(cmd, sizeof(cmd), "make -C %s build/libckernel_engine.so", project_root);
        if (run_cmd(cmd, cfg.verbose) != 0) return 1;
    }

    /* Compile requested modes */
    char modes_copy[128];
    strncpy(modes_copy, cfg.modes, sizeof(modes_copy) - 1);
    modes_copy[sizeof(modes_copy) - 1] = '\0';

    char *tok = strtok(modes_copy, ",");
    while (tok) {
        if (strcmp(tok, "prefill") == 0 || strcmp(tok, "decode") == 0) {
            if (compile_mode(project_root, cfg.output_dir, cfg.model_name, tok, cfg.verbose) != 0) {
                return 1;
            }
        } else {
            fprintf(stderr, "Skipping unsupported mode: %s\n", tok);
        }
        tok = strtok(NULL, ",");
    }

    printf("[ok] Built v4 libs in %s\n", cfg.output_dir);
    return 0;
}
