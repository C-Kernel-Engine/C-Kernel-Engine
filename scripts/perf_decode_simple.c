// Simple wrapper to profile decode
// Compile: gcc -O3 -o perf_decode_simple perf_decode_simple.c -L~/.cache/ck-engine-v5/models/Qwen--Qwen2-0.5B-Instruct-GGUF -lmodel -lm -ldl -lpthread

#include <stdio.h>
#include <time.h>
#include <dlfcn.h>

int main() {
    void *handle = dlopen("/home/antshiv/.cache/ck-engine-v5/models/Qwen--Qwen2-0.5B-Instruct-GGUF/libmodel.so", RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "Error loading library: %s\n", dlerror());
        return 1;
    }

    // Load function pointers
    int (*ck_model_init)(const char*) = dlsym(handle, "ck_model_init");
    int (*ck_model_decode)(int, float*) = dlsym(handle, "ck_model_decode");
    int (*ck_model_get_vocab_size)() = dlsym(handle, "ck_model_get_vocab_size");

    if (!ck_model_init || !ck_model_decode || !ck_model_get_vocab_size) {
        fprintf(stderr, "Error loading functions\n");
        dlclose(handle);
        return 1;
    }

    // Initialize
    printf("Loading model...\n");
    ck_model_init("/home/antshiv/.cache/ck-engine-v5/models/Qwen--Qwen2-0.5B-Instruct-GGUF/weights.bump");

    // Decode loop
    printf("Running 50 decode steps...\n");
    for (int i = 0; i < 50; i++) {
        float logits[151936];
        ck_model_decode(1000 + i, logits);
    }

    printf("Done\n");
    dlclose(handle);
    return 0;
}
