/**
 * llama_dump_logits.c - Dump llama.cpp logits for a single token
 *
 * Compile:
 *   gcc -o llama_dump_logits scripts/llama_dump_logits.c \
 *       -I llama.cpp/include \
 *       -L llama.cpp/build/bin -lllama -lggml -lggml-base \
 *       -Wl,-rpath,llama.cpp/build/bin
 *
 * Usage:
 *   ./llama_dump_logits model.gguf 9707 output.bin
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "llama.h"

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <model.gguf> <token_id> <output.bin>\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    int token_id = atoi(argv[2]);
    const char *output_path = argv[3];

    printf("Model: %s\n", model_path);
    printf("Token: %d\n", token_id);

    // Initialize backend
    llama_backend_init();

    // Load model
    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;  // CPU only

    struct llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // Get model info
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);
    int n_embd = llama_model_n_embd(model);

    printf("Vocab size: %d\n", n_vocab);
    printf("Embed dim: %d\n", n_embd);

    // Create context
    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    cparams.n_batch = 512;
    cparams.n_threads = 4;
    // cparams.flash_attn_type = 0; // disabled

    struct llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    // Create batch with single token
    struct llama_batch batch = llama_batch_init(1, 0, 1);
    batch.n_tokens = 1;
    batch.token[0] = token_id;
    batch.pos[0] = 0;
    batch.n_seq_id[0] = 1;
    llama_seq_id seq_id = 0;
    batch.seq_id[0] = &seq_id;
    batch.logits[0] = 1;  // Get logits for this token

    printf("Running forward pass...\n");

    // Decode
    int ret = llama_decode(ctx, batch);
    if (ret != 0) {
        fprintf(stderr, "Decode failed: %d\n", ret);
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    // Get logits
    float *logits = llama_get_logits_ith(ctx, 0);
    if (!logits) {
        fprintf(stderr, "Failed to get logits\n");
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    // Print some stats
    float max_val = logits[0];
    int max_idx = 0;
    float sum = 0;
    for (int i = 0; i < n_vocab; i++) {
        sum += logits[i];
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    printf("Max logit: %.4f at token %d\n", max_val, max_idx);
    printf("Mean logit: %.4f\n", sum / n_vocab);

    // Save to file
    FILE *f = fopen(output_path, "wb");
    if (f) {
        fwrite(&n_vocab, sizeof(int), 1, f);
        fwrite(logits, sizeof(float), n_vocab, f);
        fclose(f);
        printf("\nSaved %d logits to %s\n", n_vocab, output_path);
    } else {
        fprintf(stderr, "Failed to write output file\n");
    }

    // Cleanup
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
