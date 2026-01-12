/**
 * @file v6_inference.c
 * @brief C-Kernel-Engine v6 Inference
 *
 * Loads weights from BUMP file and runs real inference using
 * generated model code with proper BPE tokenization.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include "ckernel_engine.h"
#include "ck_tokenizer.h"

/* Version */
#define CK_VERSION "6.0.0"

/* ANSI colors */
#define ANSI_RESET   "\033[0m"
#define ANSI_BOLD    "\033[1m"
#define ANSI_DIM     "\033[2m"
#define ANSI_GREEN   "\033[0;32m"
#define ANSI_YELLOW  "\033[0;33m"
#define ANSI_CYAN    "\033[0;36m"

/* Generated model includes */
#include "ck-kernel-inference.h"
#include "ck-kernel-prefill.h"

/* Manifest entry */
typedef struct {
    char name[128];
    char dtype[32];
    size_t file_offset;
    size_t size;
    size_t runtime_offset;
} ManifestEntry;

/* Top-k sampling */
static int sample_topk(float *probs, int vocab_size, int topk) {
    int start = vocab_size > topk ? vocab_size - topk : 0;
    int best_idx = start;
    float best_val = probs[start];

    for (int i = start + 1; i < vocab_size; i++) {
        if (probs[i] > best_val) {
            best_val = probs[i];
            best_idx = i;
        }
    }
    return best_idx;
}

/* Load manifest from JSON */
static int load_manifest(const char *path, ManifestEntry **entries, int *num_entries) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Failed to open manifest: %s\n", path);
        return -1;
    }

    /* Read entire file */
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *content = malloc(size + 1);
    fread(content, 1, size, f);
    content[size] = '\0';
    fclose(f);

    /* Count entries */
    int count = 0;
    char *p = content;
    while ((p = strstr(p, "\"name\":")) != NULL) {
        count++;
        p += 7;
    }

    *entries = malloc(count * sizeof(ManifestEntry));
    *num_entries = count;

    /* Parse entries (simplified) */
    p = content;
    int idx = 0;
    while ((p = strstr(p, "\"name\":")) != NULL && idx < count) {
        char *start = strchr(p, '"') + 1;
        char *end = strchr(start, '"');
        size_t len = end - start;
        if (len >= sizeof((*entries)[idx].name)) len = sizeof((*entries)[idx].name) - 1;
        strncpy((*entries)[idx].name, start, len);
        (*entries)[idx].name[len] = '\0';

        /* Find dtype */
        char *dtype_p = strstr(p, "\"dtype\":");
        if (dtype_p) {
            char *d_start = strchr(dtype_p, '"') + 1;
            char *d_end = strchr(d_start, '"');
            size_t d_len = d_end - d_start;
            if (d_len >= sizeof((*entries)[idx].dtype)) d_len = sizeof((*entries)[idx].dtype) - 1;
            strncpy((*entries)[idx].dtype, d_start, d_len);
            (*entries)[idx].dtype[d_len] = '\0';
        }

        /* Find file_offset */
        char *fo_p = strstr(p, "\"file_offset\":");
        if (fo_p) {
            sscanf(fo_p + 14, "%zu", &(*entries)[idx].file_offset);
        }

        /* Find size */
        char *size_p = strstr(p, "\"size\":");
        if (size_p) {
            sscanf(size_p + 7, "%zu", &(*entries)[idx].size);
        }

        /* Find runtime_offset */
        char *ro_p = strstr(p, "\"runtime_offset\":");
        if (ro_p) {
            sscanf(ro_p + 17, "%zu", &(*entries)[idx].runtime_offset);
        }

        idx++;
        p++;
    }

    free(content);
    return 0;
}

/* Load weights from BUMP file */
static int load_weights(QWEN2_DECODEModel *model, const char *bump_path,
                        const char *manifest_path) {
    printf("[INFO] Loading weights from: %s\n", bump_path);

    /* Open BUMP file */
    int fd = open(bump_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Failed to open BUMP file: %s\n", bump_path);
        return -1;
    }

    /* Get file size */
    off_t file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    printf("[INFO] BUMP file size: %ld bytes\n", (long)file_size);

    /* Load manifest */
    ManifestEntry *entries = NULL;
    int num_entries = 0;
    if (load_manifest(manifest_path, &entries, &num_entries) != 0) {
        close(fd);
        return -1;
    }
    printf("[INFO] Manifest entries: %d\n", num_entries);

    /* Load weights from BUMP using mmap for efficiency */
    void *bump_base = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (bump_base == MAP_FAILED) {
        fprintf(stderr, "Failed to mmap BUMP file\n");
        close(fd);
        free(entries);
        return -1;
    }
    close(fd);

    /* Copy each weight entry */
    for (int i = 0; i < num_entries; i++) {
        ManifestEntry *e = &entries[i];

        if (e->runtime_offset + e->size > model->total_bytes) {
            fprintf(stderr, "Warning: Entry %s exceeds model memory\n", e->name);
            continue;
        }

        memcpy((char *)model->base + e->runtime_offset,
               (char *)bump_base + e->file_offset,
               e->size);
    }

    munmap(bump_base, file_size);
    free(entries);

    printf("[INFO] Weights loaded successfully\n");
    return 0;
}

/* Run inference with tokenizer */
static int run_inference(const char *bump_path, const char *manifest_path,
                        const char *tokenizer_path, const char *prompt,
                        int max_tokens, float temperature, int topk) {
    printf(ANSI_CYAN);
    printf("\n  C-Kernel-Engine v6 Inference\n");
    printf(ANSI_RESET);
    printf("\n");

    /* Load tokenizer */
    printf("[INFO] Loading tokenizer: %s\n", tokenizer_path);
    CKTokenizer tokenizer;
    if (ck_tokenizer_init(&tokenizer) != 0) {
        fprintf(stderr, "Failed to init tokenizer\n");
        return -1;
    }

    if (ck_tokenizer_load(&tokenizer, tokenizer_path) != 0) {
        fprintf(stderr, "Failed to load tokenizer: %s\n", tokenizer_path);
        ck_tokenizer_free(&tokenizer);
        return -1;
    }
    printf("[INFO] Tokenizer loaded: %d tokens\n", ck_tokenizer_vocab_size(&tokenizer));

    /* Allocate model */
    QWEN2_DECODEModel model;
    printf("[INFO] Allocating model (%zu bytes)...\n", (size_t)QWEN2_DECODE_TOTAL_BYTES);

    if (qwen2_decode_model_allocate(&model) != 0) {
        fprintf(stderr, "Failed to allocate model\n");
        ck_tokenizer_free(&tokenizer);
        return -1;
    }
    printf("[INFO] Model allocated at %p\n", model.base);

    /* Load weights */
    if (load_weights(&model, bump_path, manifest_path) != 0) {
        qwen2_decode_model_free(&model);
        ck_tokenizer_free(&tokenizer);
        return -1;
    }

    /* Tokenize prompt */
    int32_t tokens[512];
    int num_tokens = ck_tokenizer_encode(&tokenizer, prompt, -1, tokens, 512);
    printf("[INFO] Tokenized prompt: %d tokens\n", num_tokens);

    if (num_tokens <= 0) {
        fprintf(stderr, "Failed to tokenize prompt\n");
        qwen2_decode_model_free(&model);
        ck_tokenizer_free(&tokenizer);
        return -1;
    }

    /* Print tokenized input */
    printf("[INFO] Input tokens: ");
    for (int i = 0; i < num_tokens; i++) {
        const char *tok_str = ck_tokenizer_id_to_token(&tokenizer, tokens[i]);
        if (tok_str) {
            printf("%d(%s) ", tokens[i], tok_str);
        } else {
            printf("%d(?) ", tokens[i]);
        }
    }
    printf("\n");

    /* Run prefill */
    printf("[INFO] Running prefill for %d tokens...\n", num_tokens);
    qwen2_decode_forward(&model, tokens, num_tokens);

    /* Get logits for first token */
    float *logits = (float *)((char *)model.base + QWEN2_DECODE_FOOTER.logits);

    /* Generate tokens */
    printf("\n[INFO] Generating %d tokens...\n", max_tokens);
    printf(ANSI_YELLOW);

    int eos_token = 151645;  /* Qwen EOS */
    int32_t token = tokens[num_tokens - 1];

    /* Buffer for decoded output */
    char output_buffer[4096];
    int output_pos = 0;
    output_buffer[0] = '\0';

    for (int i = 0; i < max_tokens; i++) {
        /* Run decode for this token */
        qwen2_decode_decode(&model, &token, 0);

        /* Get logits */
        logits = (float *)((char *)model.base + QWEN2_DECODE_FOOTER.logits);

        /* Sample next token */
        int next_token = sample_topk(logits, QWEN2_DECODE_VOCAB_SIZE, topk);

        /* Get token string and append to output */
        const char *tok_str = ck_tokenizer_id_to_token(&tokenizer, next_token);
        if (tok_str) {
            printf("%s", tok_str);
            fflush(stdout);

            /* Add to output buffer */
            size_t len = strlen(tok_str);
            if (output_pos + len < sizeof(output_buffer) - 1) {
                strcpy(output_buffer + output_pos, tok_str);
                output_pos += len;
            }
        } else if (next_token == eos_token) {
            printf(ANSI_RESET);
            printf("\n[INFO] EOS token received\n");
            break;
        }

        token = next_token;
    }

    printf(ANSI_RESET);
    printf("\n\n[INFO] Inference complete\n");

    /* Cleanup */
    qwen2_decode_model_free(&model);
    ck_tokenizer_free(&tokenizer);

    return 0;
}

/* Print banner */
static void print_banner(void) {
    printf(ANSI_CYAN);
    printf("\n");
    printf("  ____  _            _ _       _   _           _   _  _____ _          _ \n");
    printf(" |  _ \\| |          | (_)     | | (_)         | | | |/  ___| |        | |\n");
    printf(" | |_) | |_   _  ___| |_  __ _| |_ ___   _____| | | |\\ `--.| |__   __| |\n");
    printf(" |  _ <| | | | |/ __| | |/ _` | __| \\ \\ / / _ \\ | | | |`--. \\ '_ \\ / _` |\n");
    printf(" | |_) | | |_| | (__| | | (_| | |_| |\\ V /  __/ |_| |/\\__/ / | | | (_| |\n");
    printf(" |____/|_|\\__,_|\\___|_|_|\\__,_|\\__|_| \\_/ \\___|\\___|\\___/\\____/|_| |_|\\__,_|\n");
    printf(ANSI_RESET);
    printf(ANSI_DIM);
    printf("  v%s - Native C Inference\n", CK_VERSION);
    printf("\n");
}

int main(int argc, char **argv) {
    const char *bump_path = NULL;
    const char *manifest_path = NULL;
    const char *tokenizer_path = NULL;
    const char *prompt = "Hello";
    int max_tokens = 100;
    float temperature = 0.7f;
    int topk = 40;

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_banner();
            printf("Usage: %s <weights.bump> [options]\n", argv[0]);
            printf("\nOptions:\n");
            printf("  -m, --model <file>      Weights BUMP file\n");
            printf("  -t, --tokenizer <file>  Tokenizer JSON file\n");
            printf("  -p, --prompt <text>     Prompt (default: Hello)\n");
            printf("  -n, --tokens <n>        Max tokens (default: 100)\n");
            printf("  --temp <float>          Temperature (default: 0.7)\n");
            printf("  --top-k <n>             Top-k (default: 40)\n");
            printf("  -h, --help              Show help\n");
            return 0;
        }
        else if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            bump_path = argv[++i];
        }
        else if ((strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--tokenizer") == 0) && i + 1 < argc) {
            tokenizer_path = argv[++i];
        }
        else if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) && i + 1 < argc) {
            prompt = argv[++i];
        }
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            temperature = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            topk = atoi(argv[++i]);
        }
    }

    if (!bump_path) {
        print_banner();
        fprintf(stderr, "Error: No weights file specified\n");
        fprintf(stderr, "Usage: %s <weights.bump> [options]\n", argv[0]);
        return 1;
    }

    /* Default paths */
    if (!manifest_path) {
        manifest_path = "generated/weights_manifest.json";
    }
    if (!tokenizer_path) {
        tokenizer_path = "generated/tokenizer.json";
    }

    if (access(bump_path, F_OK) != 0) {
        fprintf(stderr, "Error: Weights file not found: %s\n", bump_path);
        return 1;
    }

    if (access(tokenizer_path, F_OK) != 0) {
        fprintf(stderr, "Error: Tokenizer not found: %s\n", tokenizer_path);
        return 1;
    }

    return run_inference(bump_path, manifest_path, tokenizer_path,
                        prompt, max_tokens, temperature, topk);
}
