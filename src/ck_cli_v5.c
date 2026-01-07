#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>

#include "ck_tokenizer.h"

// Types
typedef int (*init_t)(const char *weights_path);
typedef int (*embed_t)(const int32_t *tokens, int num_tokens);
typedef int (*forward_t)(float *logits_out);
typedef int (*kv_enable_t)(int capacity);
typedef int (*decode_t)(int32_t token, float *logits_out);
typedef int (*sample_argmax_t)(void);
typedef int (*get_int_t)(void);
typedef void* (*get_ptr_t)(void);
typedef void (*void_func_t)(void);

// Shared state
typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t cond_task;
    pthread_cond_t cond_done;
    
    // Task parameters
    int32_t prompt_tokens[1024];
    int n_prompt;
    int max_gen;
    bool quit;
    bool task_ready;
    
    // Result feedback
    int32_t last_token;
    bool token_ready;
    
    // Functions
    init_t init;
    embed_t embed;
    forward_t forward;
    kv_enable_t kv_enable;
    decode_t decode;
    sample_argmax_t sample;
    get_int_t get_context;
    
    const char *weights_path;
} SharedState;

// Engine Thread: Handles the heavy lifting
void *engine_thread_func(void *arg) {
    SharedState *s = (SharedState *)arg;
    
    printf("[Engine] Thread started. Initializing model...\n");
    if (s->init(s->weights_path) != 0) {
        fprintf(stderr, "[Engine] Failed to init model\n");
        return NULL;
    }
    s->kv_enable(s->get_context());

    while (1) {
        pthread_mutex_lock(&s->mutex);
        while (!s->task_ready && !s->quit) {
            pthread_cond_wait(&s->cond_task, &s->mutex);
        }
        if (s->quit) {
            pthread_mutex_unlock(&s->mutex);
            break;
        }
        
        // Start Task
        int n_prompt = s->n_prompt;
        int32_t prompt[1024];
        memcpy(prompt, s->prompt_tokens, n_prompt * sizeof(int32_t));
        int max_gen = s->max_gen;
        s->task_ready = false;
        pthread_mutex_unlock(&s->mutex);

        // 1. Prefill
        s->embed(prompt, n_prompt);
        s->forward(NULL);
        int32_t next_token = s->sample();

        // 2. Feedback first token
        pthread_mutex_lock(&s->mutex);
        s->last_token = next_token;
        s->token_ready = true;
        pthread_cond_signal(&s->cond_done);
        pthread_mutex_unlock(&s->mutex);

        // 3. Decode Loop
        for (int i = 0; i < max_gen; i++) {
            if (s->decode(next_token, NULL) != 0) break;
            next_token = s->sample();

            pthread_mutex_lock(&s->mutex);
            s->last_token = next_token;
            s->token_ready = true;
            pthread_cond_signal(&s->cond_done);
            pthread_mutex_unlock(&s->mutex);

            if (next_token == 151643 || next_token == 151645) break; 
        }
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s <libmodel.so> <weights.bump>\n", argv[0]);
        return 1;
    }

    SharedState state = {0};
    pthread_mutex_init(&state.mutex, NULL);
    pthread_cond_init(&state.cond_task, NULL);
    pthread_cond_init(&state.cond_done, NULL);
    state.weights_path = argv[2];

    void *handle = dlopen(argv[1], RTLD_NOW);
    if (!handle) { fprintf(stderr, "%s\n", dlerror()); return 1; }

    state.init = dlsym(handle, "ck_model_init");
    state.embed = dlsym(handle, "ck_model_embed_tokens");
    state.forward = dlsym(handle, "ck_model_forward");
    state.kv_enable = dlsym(handle, "ck_model_kv_cache_enable");
    state.decode = dlsym(handle, "ck_model_decode");
    state.sample = dlsym(handle, "ck_model_sample_argmax");
    state.get_context = dlsym(handle, "ck_model_get_context_window");
    get_ptr_t get_offsets = dlsym(handle, "ck_model_get_vocab_offsets");
    get_ptr_t get_strings = dlsym(handle, "ck_model_get_vocab_strings");
    get_int_t get_vocab_size = dlsym(handle, "ck_model_get_vocab_size");
    get_int_t get_num_merges = dlsym(handle, "ck_model_get_num_merges");

    // Start Engine
    pthread_t engine_thread;
    pthread_create(&engine_thread, NULL, engine_thread_func, &state);

    // Wait for engine to init (simple wait for this demo)
    sleep(1);

    // Tokenizer setup
    CKTokenizer tokenizer;
    ck_tokenizer_init(&tokenizer);
    ck_tokenizer_load_binary(&tokenizer, get_vocab_size(), get_offsets(), get_strings(), get_num_merges(), NULL);

    char input[1024];
    while (1) {
        printf("\nYou: ");
        if (!fgets(input, sizeof(input), stdin)) break;
        if (strncmp(input, "/exit", 5) == 0) break;

        // Tokenize
        int32_t ids[1024];
        int n = ck_tokenizer_encode(&tokenizer, input, -1, ids, 1024);

        // Submit Task
        pthread_mutex_lock(&state.mutex);
        memcpy(state.prompt_tokens, ids, n * sizeof(int32_t));
        state.n_prompt = n;
        state.max_gen = 100;
        state.task_ready = true;
        state.token_ready = false;
        pthread_cond_signal(&state.cond_task);
        pthread_mutex_unlock(&state.mutex);

        printf("Assistant: ");
        fflush(stdout);

        // UI Detokenizer Loop
        while (1) {
            pthread_mutex_lock(&state.mutex);
            while (!state.token_ready && !state.quit) {
                pthread_cond_wait(&state.cond_done, &state.mutex);
            }
            int32_t tok = state.last_token;
            state.token_ready = false;
            pthread_mutex_unlock(&state.mutex);

            if (tok == 151643 || tok == 151645) break;

            const char *word = ck_tokenizer_id_to_token(&tokenizer, tok);
            if (word) {
                if ((unsigned char)word[0] == 0xC4 && (unsigned char)word[1] == 0xA0) {
                    printf(" %s", word + 2);
                } else {
                    printf("%s", word);
                }
                fflush(stdout);
            }
        }
        printf("\n");
    }

    pthread_mutex_lock(&state.mutex);
    state.quit = true;
    pthread_cond_signal(&state.cond_task);
    pthread_mutex_unlock(&state.mutex);
    pthread_join(engine_thread, NULL);

    return 0;
}
