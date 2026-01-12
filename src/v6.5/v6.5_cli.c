/**
 * @file v6_cli.c
 * @brief C-Kernel-Engine v6 CLI
 *
 * Usage:
 *   ./ck-engine-v6 -m <model.gguf> [options]
 *   ./ck-engine-v6 -m <model.gguf> -p "Hello world"
 *   ./ck-engine-v6 -m <model.gguf> -p @prompt.txt --temp 0.7
 *
 * Options:
 *   -m, --model <file>       Model file (GGUF or BUMP)
 *   -p, --prompt <text>      Prompt (use @file.txt to read from file)
 *   -t, --tokens <n>         Max tokens to generate (default: 100)
 *   -t, --temp <float>       Temperature (default: 0.7)
 *   -p, --top-p <float>      Top-p sampling (default: 0.9)
 *   -k, --top-k <n>          Top-k sampling (default: 40)
 *   --seed <n>              Random seed (default: random)
 *   --threads <n>           Number of threads (default: auto)
 *   --no-kv-cache           Disable KV cache
 *   --ignore-eos            Ignore EOS token
 *   --verbose               Verbose output
 *   -v, --version           Show version
 *   -h, --help              Show this help
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <limits.h>
#include <unistd.h>

#include "ckernel_engine.h"

/* Version */
#define CK_VERSION "6.0.0"
#define CK_BUILD_DATE __DATE__

/* ANSI colors */
#define ANSI_RESET   "\033[0m"
#define ANSI_BOLD    "\033[1m"
#define ANSI_DIM     "\033[2m"
#define ANSI_GREEN   "\033[0;32m"
#define ANSI_YELLOW  "\033[0;33m"
#define ANSI_BLUE    "\033[0;34m"
#define ANSI_CYAN    "\033[0;36m"

/* Model configuration (defaults) */
typedef struct {
    const char *model_path;
    const char *prompt;
    int max_tokens;
    float temperature;
    float top_p;
    int top_k;
    int seed;
    int threads;
    int kv_cache;
    int ignore_eos;
    int verbose;
} CLIArgs;

/* Print banner */
static void print_banner(void) {
    printf(ANSI_CYAN);
    printf("\n");
    printf("  C-Kernel-Engine\n");
    printf("  ------------------------------\n");
    printf(ANSI_RESET);
    printf(ANSI_DIM);
    printf("  v%s - %s - Native C Implementation\n", CK_VERSION, CK_BUILD_DATE);
    printf("  " ANSI_GREEN "https://github.com/antshiv/C-Kernel-Engine" ANSI_RESET "\n");
    printf("\n");
}

/* Print help */
static void print_help(const char *prog) {
    printf(ANSI_BOLD "usage:" ANSI_RESET " %s [options]\n\n", prog);
    printf(ANSI_BOLD "options:" ANSI_RESET "\n");
    printf("  -m, --model <file>       Model file (GGUF or BUMP)\n");
    printf("  -p, --prompt <text>      Prompt (use @file.txt to read from file)\n");
    printf("  -t, --tokens <n>         Max tokens to generate (default: 100)\n");
    printf("  -t, --temp <float>       Temperature (default: 0.7)\n");
    printf("      --top-p <float>      Top-p sampling (default: 0.9)\n");
    printf("      --top-k <n>          Top-k sampling (default: 40)\n");
    printf("      --seed <n>           Random seed (default: random)\n");
    printf("      --threads <n>        Number of threads (default: auto)\n");
    printf("      --no-kv-cache        Disable KV cache\n");
    printf("      --ignore-eos         Ignore EOS token\n");
    printf("      --verbose            Verbose output\n");
    printf("  -v, --version            Show version\n");
    printf("  -h, --help               Show this help\n");
    printf("\n");
    printf(ANSI_BOLD "examples:" ANSI_RESET "\n");
    printf("  %s -m model.gguf -p \"Hello\"\n", prog);
    printf("  %s -m model.gguf -p @prompt.txt -t 50 --temp 0.8\n", prog);
    printf("  %s -m model.gguf --seed 42 --verbose\n", prog);
    printf("\n");
}

/* Print version */
static void print_version(void) {
    printf(ANSI_BOLD "C-Kernel-Engine" ANSI_RESET " v" CK_VERSION "\n");
    printf("Build: " CK_BUILD_DATE "\n");
    printf("C标准: " ANSI_YELLOW "C11" ANSI_RESET "\n");
    printf("OMP支持: " ANSI_YELLOW "Yes" ANSI_RESET "\n");
}

/* Parse arguments */
static CLIArgs parse_args(int argc, char **argv) {
    CLIArgs args = {
        .model_path = NULL,
        .prompt = "Hello",
        .max_tokens = 100,
        .temperature = 0.7f,
        .top_p = 0.9f,
        .top_k = 40,
        .seed = -1,
        .threads = 0,
        .kv_cache = 1,
        .ignore_eos = 0,
        .verbose = 0,
    };

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_banner();
            print_help(argv[0]);
            exit(0);
        }
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--version") == 0) {
            print_version();
            exit(0);
        }
        if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            args.model_path = argv[++i];
        }
        else if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) && i + 1 < argc) {
            args.prompt = argv[++i];
        }
        else if ((strcmp(argv[i], "-t") == 0) && i + 1 < argc) {
            args.max_tokens = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            args.temperature = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            args.top_p = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            args.top_k = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            args.seed = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            args.threads = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--no-kv-cache") == 0) {
            args.kv_cache = 0;
        }
        else if (strcmp(argv[i], "--ignore-eos") == 0) {
            args.ignore_eos = 1;
        }
        else if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            args.verbose = 1;
        }
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            fprintf(stderr, "Use --help for usage\n");
            exit(1);
        }
    }

    return args;
}

/* Read prompt from file */
static char *read_prompt_file(const char *path) {
    if (path[0] != '@') return (char *)path;

    FILE *f = fopen(path + 1, "r");
    if (!f) {
        fprintf(stderr, "Cannot open prompt file: %s\n", path + 1);
        return (char *)path;
    }

    char *content = malloc(4096);
    size_t len = fread(content, 1, 4095, f);
    content[len] = '\0';
    fclose(f);

    // Remove trailing newlines
    while (len > 0 && (content[len-1] == '\n' || content[len-1] == '\r')) {
        content[--len] = '\0';
    }

    return content;
}

/* Simple tokenization (placeholder) */
static int32_t *tokenize(const char *text, int *num_tokens) {
    static int32_t tokens[1024];
    *num_tokens = strlen(text);
    for (int i = 0; i < *num_tokens && i < 1024; i++) {
        tokens[i] = (int32_t)text[i];
    }
    return tokens;
}

/* Sample from logits (simplified) */
static int sample_token(float *logits, int vocab_size, float temp, int top_k) {
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] = expf((logits[i] - max_val) / temp);
        sum += logits[i];
    }

    // Top-k filter
    int start = vocab_size > top_k ? vocab_size - top_k : 0;

    // Simple argmax
    float best_val = logits[start];
    int best_idx = start;
    for (int i = start + 1; i < vocab_size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_idx = i;
        }
    }

    return best_idx;
}

/* Print progress */
static void print_progress(int token_id, float token_per_sec) {
    static time_t last_time = 0;
    time_t now = time(NULL);

    if (now - last_time >= 1) {
        printf(ANSI_DIM "[ %.1f tok/s ]" ANSI_RESET "\r", token_per_sec);
        fflush(stdout);
        last_time = now;
    }
}

int main(int argc, char **argv) {
    CLIArgs args = parse_args(argc, argv);

    /* Print banner */
    if (!args.verbose) {
        print_banner();
    }

    /* Validate arguments */
    if (!args.model_path) {
        fprintf(stderr, ANSI_YELLOW "Error:" ANSI_RESET " No model specified\n");
        fprintf(stderr, "Use " ANSI_BOLD "-m <model>" ANSI_RESET " or " ANSI_BOLD "--help" ANSI_RESET "\n");
        return 1;
    }

    /* Check if model file exists */
    if (access(args.model_path, F_OK) != 0) {
        fprintf(stderr, ANSI_YELLOW "Error:" ANSI_RESET " Model file not found: %s\n", args.model_path);
        return 1;
    }

    /* Read prompt */
    char *prompt = read_prompt_file(args.prompt);

    /* Set random seed */
    if (args.seed < 0) {
        args.seed = (int)time(NULL);
    }
    srand(args.seed);

    /* Print inference parameters */
    printf(ANSI_BOLD "Parameters:" ANSI_RESET "\n");
    printf("  Model:     " ANSI_CYAN "%s" ANSI_RESET "\n", args.model_path);
    printf("  Prompt:    \"" ANSI_YELLOW "%s" ANSI_RESET "\"\n", prompt);
    printf("  Tokens:    %d\n", args.max_tokens);
    printf("  Temp:      %.2f\n", args.temperature);
    printf("  Top-p:     %.2f\n", args.top_p);
    printf("  Top-k:     %d\n", args.top_k);
    printf("  Seed:      %d\n", args.seed);
    printf("  Threads:   %d\n", args.threads > 0 ? args.threads : 4);
    printf("\n");

    /* Start generation */
    printf(ANSI_BOLD "Output:" ANSI_RESET "\n");
    printf(ANSI_YELLOW);
    fflush(stdout);

    /* Placeholder inference - in real impl would load model and run */
    printf("<Model loading would happen here>\n");
    printf("<Inference would run here>\n");
    printf("\n");

    /* Demo output */
    printf(ANSI_RESET);
    printf(ANSI_BOLD "Note:" ANSI_RESET " This is v6 placeholder CLI.\n");
    printf("Full model loading and inference requires:\n");
    printf("  1. Generated model code from IR\n");
    printf("  2. All kernel implementations compiled\n");
    printf("  3. Weight loading from GGUF/BUMP\n");
    printf("\n");

    /* Cleanup */
    if (prompt != args.prompt) {
        free(prompt);
    }

    return 0;
}
