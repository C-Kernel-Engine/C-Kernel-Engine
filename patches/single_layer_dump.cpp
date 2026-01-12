/**
 * single_layer_dump.cpp - Single Layer Tensor Dump Tool for llama.cpp
 *
 * This tool runs llama.cpp with tensor dumping enabled for a specific layer,
 * outputting intermediate activations for parity comparison with C-Kernel-Engine.
 *
 * Build:
 *   cd llama.cpp
 *   cmake -B build -DGGML_CPU=ON
 *   g++ -I. -I./include -I./ggml/include \
 *       ../patches/single_layer_dump.cpp \
 *       -L./build/src -L./build/ggml/src \
 *       -lllama -lggml -lm -lpthread \
 *       -o build/bin/single-layer-dump
 *
 * Usage:
 *   ./build/bin/single-layer-dump \
 *       --model model.gguf \
 *       --layer 0 \
 *       --output-dir ./dumps \
 *       --prompt "Hello"
 *
 * Output files (binary FP32):
 *   dumps/inp_embd.bin         - Token embeddings
 *   dumps/attn_norm-0.bin      - Attention RMSNorm output
 *   dumps/Qcur-0.bin           - Q projection
 *   dumps/Kcur-0.bin           - K projection
 *   dumps/Vcur-0.bin           - V projection
 *   dumps/kq-0.bin             - Attention scores
 *   dumps/kq_softmax-0.bin     - Softmax output
 *   dumps/attn_out-0.bin       - Attention output
 *   dumps/ffn_norm-0.bin       - FFN RMSNorm output
 *   dumps/ffn_gate-0.bin       - FFN gate
 *   dumps/ffn_up-0.bin         - FFN up
 *   dumps/ffn_out-0.bin        - FFN output
 *   dumps/res-0.bin            - Layer output (residual)
 */

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

// Configuration
struct dump_config {
    std::string model_path;
    std::string output_dir = "./layer_dumps";
    std::string prompt = "Hello";
    int target_layer = 0;
    int n_tokens = 1;
    bool verbose = false;
};

// Global state for callback
static dump_config g_config;
static std::map<std::string, bool> g_dumped;

/**
 * Dump a tensor to binary file
 */
static void dump_tensor(const char* name, const ggml_tensor* t) {
    if (!t) return;

    // Create output directory if needed
    fs::create_directories(g_config.output_dir);

    // Build filename
    std::string filename = g_config.output_dir + "/" + name + ".bin";

    // Skip if already dumped
    if (g_dumped.count(filename)) return;
    g_dumped[filename] = true;

    // Get tensor data
    size_t n_bytes = ggml_nbytes(t);
    std::vector<uint8_t> data(n_bytes);

    // Backend-aware read
    ggml_backend_tensor_get(t, data.data(), 0, n_bytes);

    // Convert to FP32 if needed
    std::vector<float> fp32_data;
    float* output_data = nullptr;
    size_t output_bytes = 0;

    if (t->type == GGML_TYPE_F32) {
        output_data = (float*)data.data();
        output_bytes = n_bytes;
    } else if (t->type == GGML_TYPE_F16) {
        size_t n_elems = n_bytes / sizeof(ggml_fp16_t);
        fp32_data.resize(n_elems);
        ggml_fp16_t* fp16_ptr = (ggml_fp16_t*)data.data();
        for (size_t i = 0; i < n_elems; i++) {
            fp32_data[i] = ggml_fp16_to_fp32(fp16_ptr[i]);
        }
        output_data = fp32_data.data();
        output_bytes = n_elems * sizeof(float);
    } else if (t->type == GGML_TYPE_BF16) {
        size_t n_elems = n_bytes / sizeof(uint16_t);
        fp32_data.resize(n_elems);
        uint16_t* bf16_ptr = (uint16_t*)data.data();
        for (size_t i = 0; i < n_elems; i++) {
            // BF16 to FP32 conversion
            uint32_t tmp = ((uint32_t)bf16_ptr[i]) << 16;
            float val;
            memcpy(&val, &tmp, sizeof(float));
            fp32_data[i] = val;
        }
        output_data = fp32_data.data();
        output_bytes = n_elems * sizeof(float);
    } else {
        fprintf(stderr, "Warning: tensor %s has unsupported type %d, skipping\n",
                name, (int)t->type);
        return;
    }

    // Write to file
    std::ofstream out(filename, std::ios::binary);
    if (out.is_open()) {
        out.write((const char*)output_data, output_bytes);
        out.close();
        if (g_config.verbose) {
            printf("Dumped: %s (%zu bytes, shape=[%ld,%ld,%ld,%ld])\n",
                   filename.c_str(), output_bytes,
                   (long)t->ne[0], (long)t->ne[1], (long)t->ne[2], (long)t->ne[3]);
        }
    } else {
        fprintf(stderr, "Error: could not write to %s\n", filename.c_str());
    }
}

/**
 * Eval callback - called for each tensor during computation
 */
static bool eval_callback(struct ggml_tensor* t, bool ask, void* user_data) {
    (void)user_data;

    if (!t || !t->name) return true;

    const char* name = t->name;

    // If asking, tell llama.cpp which tensors we want to capture
    if (ask) {
        // Check if this tensor name contains our target layer
        char layer_suffix[16];
        snprintf(layer_suffix, sizeof(layer_suffix), "-%d", g_config.target_layer);

        // Tensors we care about
        static const char* tensor_patterns[] = {
            "inp_embd",
            "attn_norm",
            "Qcur",
            "Kcur",
            "Vcur",
            "Qcur_rope",
            "Kcur_rope",
            "kq",
            "kq_softmax",
            "kq_softmax_cast",
            "kqv",
            "kqv_merged",
            "attn_out",
            "ffn_norm",
            "ffn_gate",
            "ffn_up",
            "ffn_gate_par",
            "ffn_down",
            "ffn_out",
            "res",
            "inpL",
            "inpSA",
            "inpFF",
            NULL
        };

        // Check if this tensor should be captured
        for (int i = 0; tensor_patterns[i] != NULL; i++) {
            if (strstr(name, tensor_patterns[i]) != NULL) {
                // For layer-specific tensors, check the layer number
                if (strstr(name, "-") != NULL) {
                    if (strstr(name, layer_suffix) != NULL) {
                        return true;  // Capture this tensor
                    }
                } else {
                    // Global tensors (like inp_embd)
                    return true;
                }
            }
        }
        return false;  // Don't capture
    }

    // Not asking - this is the actual tensor data, dump it
    dump_tensor(name, t);
    return true;
}

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("\nOptions:\n");
    printf("  --model, -m PATH      Path to GGUF model file (required)\n");
    printf("  --layer, -l N         Layer index to dump (default: 0)\n");
    printf("  --output-dir, -o DIR  Output directory for dumps (default: ./layer_dumps)\n");
    printf("  --prompt, -p TEXT     Prompt text (default: 'Hello')\n");
    printf("  --tokens, -n N        Number of tokens to generate (default: 1)\n");
    printf("  --verbose, -v         Verbose output\n");
    printf("  --help, -h            Show this help\n");
    printf("\nExample:\n");
    printf("  %s -m model.gguf -l 0 -o ./dumps -p 'Hello world'\n", prog);
}

int main(int argc, char** argv) {
    dump_config config;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--model" || arg == "-m") {
            if (++i >= argc) { fprintf(stderr, "Error: --model requires a path\n"); return 1; }
            config.model_path = argv[i];
        } else if (arg == "--layer" || arg == "-l") {
            if (++i >= argc) { fprintf(stderr, "Error: --layer requires a number\n"); return 1; }
            config.target_layer = atoi(argv[i]);
        } else if (arg == "--output-dir" || arg == "-o") {
            if (++i >= argc) { fprintf(stderr, "Error: --output-dir requires a path\n"); return 1; }
            config.output_dir = argv[i];
        } else if (arg == "--prompt" || arg == "-p") {
            if (++i >= argc) { fprintf(stderr, "Error: --prompt requires text\n"); return 1; }
            config.prompt = argv[i];
        } else if (arg == "--tokens" || arg == "-n") {
            if (++i >= argc) { fprintf(stderr, "Error: --tokens requires a number\n"); return 1; }
            config.n_tokens = atoi(argv[i]);
        } else if (arg == "--verbose" || arg == "-v") {
            config.verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    if (config.model_path.empty()) {
        fprintf(stderr, "Error: --model is required\n");
        print_usage(argv[0]);
        return 1;
    }

    // Set global config for callback
    g_config = config;

    printf("=== Single Layer Tensor Dump ===\n");
    printf("Model: %s\n", config.model_path.c_str());
    printf("Layer: %d\n", config.target_layer);
    printf("Output: %s\n", config.output_dir.c_str());
    printf("Prompt: '%s'\n", config.prompt.c_str());
    printf("\n");

    // Initialize llama
    llama_backend_init();

    // Load model
    llama_model_params model_params = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(config.model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "Error: failed to load model\n");
        return 1;
    }

    // Get vocab for tokenization
    const llama_vocab* vocab = llama_model_get_vocab(model);

    // Create context with eval callback
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 512;
    ctx_params.cb_eval = eval_callback;
    ctx_params.cb_eval_user_data = nullptr;

    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Error: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    // Tokenize prompt
    std::vector<llama_token> tokens(config.prompt.length() + 16);
    int n_tokens = llama_tokenize(vocab, config.prompt.c_str(), config.prompt.length(),
                                   tokens.data(), tokens.size(), true, true);
    if (n_tokens < 0) {
        fprintf(stderr, "Error: tokenization failed\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    tokens.resize(n_tokens);

    printf("Tokens: %d\n", n_tokens);

    // Create batch
    llama_batch batch = llama_batch_init(512, 0, 1);

    // Add tokens to batch
    for (int i = 0; i < n_tokens; i++) {
        llama_batch_add(batch, tokens[i], i, {0}, i == n_tokens - 1);
    }

    printf("Running forward pass for layer %d...\n", config.target_layer);

    // Run forward pass (this triggers the callback)
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Error: decode failed\n");
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    printf("\nDumped %zu tensors to %s\n", g_dumped.size(), config.output_dir.c_str());

    // Cleanup
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
