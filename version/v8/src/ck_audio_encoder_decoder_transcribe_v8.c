#define _POSIX_C_SOURCE 200809L

#include <dlfcn.h>
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int (*ck_init_fn)(const char *, const char *);
typedef void (*ck_free_fn)(void);
typedef int (*ck_int_query_fn)(void);
typedef int (*ck_frontend_fn)(const uint8_t *, size_t, float *, int, int *, void *);
typedef size_t (*ck_encoder_workspace_fn)(int, int);
typedef int (*ck_encoder_fn)(const float *, int, int, float *, int, int *, void *, size_t);
typedef int (*ck_set_encoder_fn)(const float *, int, int);
typedef void (*ck_reset_fn)(void);
typedef int32_t (*ck_prompt_token_fn)(int);
typedef int (*ck_embed_fn)(const int32_t *, int);
typedef int (*ck_forward_fn)(float *);
typedef int (*ck_decode_step_fn)(int32_t, float *);
typedef int (*ck_decode_tokens_fn)(const int32_t *, int, char *, int);
typedef int (*ck_stop_fn)(int32_t);
typedef void (*ck_profile_reset_fn)(void);
typedef size_t (*ck_profile_count_fn)(void);
typedef uint64_t (*ck_profile_overflow_fn)(void);
typedef int (*ck_profile_get_fn)(
    size_t, int *, int *, int *, int *, int *, uint64_t *, uint64_t *);
typedef int (*ck_get_threads_fn)(void);
typedef void *(*ck_threadpool_global_fn)(void);
typedef void (*ck_threadpool_profile_reset_fn)(void *);
typedef struct {
    uint64_t dispatch_count;
    uint64_t dispatch_total_ns;
    uint64_t main_work_ns;
    uint64_t completion_wait_ns;
} CKThreadpoolProfile;
typedef void (*ck_threadpool_profile_snapshot_fn)(const void *, CKThreadpoolProfile *);

typedef struct {
    void *handle;
    ck_init_fn init;
    ck_free_fn free_model;
} CKComponent;

typedef struct {
    uint8_t *bytes;
    size_t size;
    const uint8_t *pcm;
    uint32_t pcm_bytes;
    uint32_t frames;
    uint32_t sample_rate;
    uint16_t channels;
    uint16_t bits_per_sample;
} CKWav;

typedef struct {
    uint32_t start_frame;
    uint32_t end_frame;
} CKAudioSegment;

static uint16_t ck_u16le(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t ck_u32le(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static int ck_read_file(const char *path, uint8_t **bytes, size_t *size) {
    FILE *file = fopen(path, "rb");
    long length;
    uint8_t *data;
    if (!file) return -1;
    if (fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) <= 0 ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return -2;
    }
    data = (uint8_t *)malloc((size_t)length);
    if (!data || fread(data, 1, (size_t)length, file) != (size_t)length) {
        free(data);
        fclose(file);
        return -3;
    }
    fclose(file);
    *bytes = data;
    *size = (size_t)length;
    return 0;
}

static void ck_put_u16le(uint8_t *p, uint16_t value) {
    p[0] = (uint8_t)value;
    p[1] = (uint8_t)(value >> 8);
}

static void ck_put_u32le(uint8_t *p, uint32_t value) {
    p[0] = (uint8_t)value;
    p[1] = (uint8_t)(value >> 8);
    p[2] = (uint8_t)(value >> 16);
    p[3] = (uint8_t)(value >> 24);
}

static int ck_parse_wav(const char *path, CKWav *wav) {
    size_t offset = 12;
    int have_fmt = 0;
    if (ck_read_file(path, &wav->bytes, &wav->size) != 0) return -1;
    if (wav->size < 44 || memcmp(wav->bytes, "RIFF", 4) != 0 ||
        memcmp(wav->bytes + 8, "WAVE", 4) != 0) return -2;
    while (offset + 8 <= wav->size) {
        const uint8_t *chunk = wav->bytes + offset;
        uint32_t chunk_size = ck_u32le(chunk + 4);
        size_t data = offset + 8;
        if (data > wav->size || chunk_size > wav->size - data) return -3;
        if (memcmp(chunk, "fmt ", 4) == 0) {
            if (chunk_size < 16 || ck_u16le(wav->bytes + data) != 1) return -4;
            wav->channels = ck_u16le(wav->bytes + data + 2);
            wav->sample_rate = ck_u32le(wav->bytes + data + 4);
            wav->bits_per_sample = ck_u16le(wav->bytes + data + 14);
            have_fmt = 1;
        } else if (memcmp(chunk, "data", 4) == 0) {
            wav->pcm = wav->bytes + data;
            wav->pcm_bytes = chunk_size;
        }
        offset = data + chunk_size + (chunk_size & 1u);
    }
    if (!have_fmt || !wav->pcm || !wav->pcm_bytes || wav->channels != 1 ||
        wav->bits_per_sample != 16 || (wav->pcm_bytes & 1u) != 0) return -5;
    wav->frames = wav->pcm_bytes / 2u;
    return wav->frames ? 0 : -6;
}

static uint8_t *ck_make_wav_window(const CKWav *wav, uint32_t start_frame,
                                   uint32_t end_frame, size_t *size) {
    uint32_t payload;
    uint8_t *output;
    if (!wav || !size || start_frame >= end_frame || end_frame > wav->frames ||
        end_frame - start_frame > UINT32_MAX / 2u) return NULL;
    payload = (end_frame - start_frame) * 2u;
    output = (uint8_t *)malloc((size_t)payload + 44u);
    if (!output) return NULL;
    memcpy(output, "RIFF", 4);
    ck_put_u32le(output + 4, payload + 36u);
    memcpy(output + 8, "WAVEfmt ", 8);
    ck_put_u32le(output + 16, 16u);
    ck_put_u16le(output + 20, 1u);
    ck_put_u16le(output + 22, 1u);
    ck_put_u32le(output + 24, wav->sample_rate);
    ck_put_u32le(output + 28, wav->sample_rate * 2u);
    ck_put_u16le(output + 32, 2u);
    ck_put_u16le(output + 34, 16u);
    memcpy(output + 36, "data", 4);
    ck_put_u32le(output + 40, payload);
    memcpy(output + 44, wav->pcm + (size_t)start_frame * 2u, payload);
    *size = (size_t)payload + 44u;
    return output;
}

static int ck_load_segment_plan(const char *path, uint32_t sample_rate,
                                uint32_t source_frames, CKAudioSegment **segments,
                                size_t *segment_count) {
    FILE *file = fopen(path, "r");
    char schema[64], trailing[2];
    unsigned plan_rate, count;
    CKAudioSegment *items;
    if (!file) return -1;
    if (fscanf(file, "%63s %u %u", schema, &plan_rate, &count) != 3 ||
        strcmp(schema, "cke_audio_segments_v1") != 0 ||
        plan_rate != sample_rate || count == 0) {
        fclose(file);
        return -2;
    }
    items = (CKAudioSegment *)calloc((size_t)count, sizeof(*items));
    if (!items) {
        fclose(file);
        return -3;
    }
    for (unsigned index = 0; index < count; ++index) {
        unsigned start, end;
        if (fscanf(file, "%u %u", &start, &end) != 2 || start >= end ||
            end > source_frames || (index && start < items[index - 1].end_frame)) {
            free(items);
            fclose(file);
            return -4;
        }
        items[index].start_frame = start;
        items[index].end_frame = end;
    }
    if (fscanf(file, "%1s", trailing) == 1) {
        free(items);
        fclose(file);
        return -5;
    }
    fclose(file);
    *segments = items;
    *segment_count = count;
    return 0;
}

static int ck_append_text(char **output, size_t *length, size_t *capacity,
                          const char *piece) {
    size_t amount = strlen(piece);
    int separator = *length && amount && (*output)[*length - 1] != ' ' && piece[0] != ' ';
    size_t required;
    if (*length > SIZE_MAX - amount - (size_t)separator - 1u) return -1;
    required = *length + amount + (size_t)separator + 1u;
    if (required > *capacity) {
        size_t next = *capacity ? *capacity : 4096u;
        while (next < required) {
            if (next > SIZE_MAX / 2u) return -1;
            next *= 2u;
        }
        char *grown = (char *)realloc(*output, next);
        if (!grown) return -1;
        *output = grown;
        *capacity = next;
    }
    if (separator) (*output)[(*length)++] = ' ';
    memcpy(*output + *length, piece, amount + 1u);
    *length += amount;
    return 0;
}

static void *ck_symbol(void *library, const char *name) {
    void *symbol = dlsym(library, name);
    if (!symbol) fprintf(stderr, "missing generated symbol %s: %s\n", name, dlerror());
    return symbol;
}

static int ck_open_component(CKComponent *component, const char *library_path,
                             const char *weights_path, const char *manifest_path) {
    component->handle = dlopen(library_path, RTLD_NOW | RTLD_LOCAL);
    if (!component->handle) {
        fprintf(stderr, "cannot load generated component %s: %s\n", library_path, dlerror());
        return -1;
    }
    *(void **)(&component->init) = ck_symbol(component->handle, "ck_model_init_with_manifest");
    *(void **)(&component->free_model) = ck_symbol(component->handle, "ck_model_free");
    if (!component->init || !component->free_model) return -2;
    if (component->init(weights_path, manifest_path) != 0) {
        fprintf(stderr, "generated component initialization failed: %s\n", library_path);
        return -3;
    }
    return 0;
}

static void ck_close_component(CKComponent *component) {
    if (component->free_model) component->free_model();
    if (component->handle) dlclose(component->handle);
    memset(component, 0, sizeof(*component));
}

static double ck_now(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec / 1.0e9;
}

static int ck_argmax(const float *values, int count) {
    int best = 0;
    for (int index = 1; index < count; ++index) {
        if (values[index] > values[best]) best = index;
    }
    return best;
}

static int ck_profile_requested(void) {
    const char *value = getenv("CK_AUDIO_PERF_PROFILE");
    return value && value[0] && strcmp(value, "0") != 0;
}

static void ck_print_gemm_profile(
    size_t window_index, ck_profile_count_fn count_fn,
    ck_profile_overflow_fn overflow_fn, ck_profile_get_fn get_fn) {
    const size_t count = count_fn();
    for (size_t index = 0; index < count; ++index) {
        int M, N, K, active_threads, parallel;
        uint64_t calls, elapsed_ns;
        if (get_fn(index, &M, &N, &K, &active_threads, &parallel,
                   &calls, &elapsed_ns) != 0) continue;
        fprintf(stderr,
                "gemm_profile window=%zu M=%d N=%d K=%d active_threads=%d "
                "mode=%s calls=%" PRIu64 " elapsed_ns=%" PRIu64 "\n",
                window_index, M, N, K, active_threads,
                parallel ? "parallel" : "serial", calls, elapsed_ns);
    }
    fprintf(stderr, "gemm_profile_summary window=%zu shapes=%zu overflow_calls=%" PRIu64 "\n",
            window_index, count, overflow_fn());
}

static void ck_print_threadpool_profile(
    size_t window_index, ck_threadpool_profile_snapshot_fn snapshot_fn,
    void *pool) {
    CKThreadpoolProfile profile = {0};
    snapshot_fn(pool, &profile);
    fprintf(stderr,
            "threadpool_profile window=%zu dispatches=%" PRIu64
            " total_ns=%" PRIu64 " main_work_ns=%" PRIu64
            " completion_wait_ns=%" PRIu64 "\n",
            window_index, profile.dispatch_count, profile.dispatch_total_ns,
            profile.main_work_ns, profile.completion_wait_ns);
}

int main(int argc, char **argv) {
    CKComponent encoder = {0}, decoder = {0};
    CKWav wav = {0};
    float *features = NULL, *encoder_output = NULL, *logits = NULL;
    void *workspace = NULL;
    int32_t *prompt = NULL, *tokens = NULL;
    char *text = NULL, *transcript = NULL;
    size_t transcript_length = 0, transcript_capacity = 0;
    CKAudioSegment single_segment, *segments = NULL;
    size_t segment_count = 0;
    uint64_t consumed_frames = 0;
    int last_token_count = 0;
    int exit_code = 1, encoder_initialized = 0, decoder_initialized = 0;
    ck_profile_reset_fn profile_reset = NULL;
    ck_profile_count_fn profile_count = NULL;
    ck_profile_overflow_fn profile_overflow = NULL;
    ck_profile_get_fn profile_get = NULL;
    void *profile_pool = NULL;
    ck_threadpool_profile_reset_fn threadpool_profile_reset = NULL;
    ck_threadpool_profile_snapshot_fn threadpool_profile_snapshot = NULL;

    if (argc != 8 && argc != 9) {
        fprintf(stderr, "usage: %s ENCODER_SO ENCODER_WEIGHTS ENCODER_MAP "
                        "DECODER_SO DECODER_WEIGHTS DECODER_MAP INPUT_WAV "
                        "[SEGMENT_PLAN]\n", argv[0]);
        return 2;
    }
    if (ck_parse_wav(argv[7], &wav) != 0) {
        fprintf(stderr, "input must be uncompressed mono PCM16 WAV\n");
        goto done;
    }
    if (argc == 9) {
        if (ck_load_segment_plan(argv[8], wav.sample_rate, wav.frames,
                                 &segments, &segment_count) != 0) {
            fprintf(stderr, "invalid native audio segment plan\n");
            goto done;
        }
    } else {
        single_segment.start_frame = 0;
        single_segment.end_frame = wav.frames;
        segments = &single_segment;
        segment_count = 1;
    }
    if (ck_open_component(&encoder, argv[1], argv[2], argv[3]) != 0) goto done;
    encoder_initialized = 1;
    if (ck_open_component(&decoder, argv[4], argv[5], argv[6]) != 0) goto done;
    decoder_initialized = 1;

    if (ck_profile_requested()) {
        *(void **)(&profile_reset) = dlsym(encoder.handle, "ck_f32_gemm_profile_reset");
        *(void **)(&profile_count) = dlsym(encoder.handle, "ck_f32_gemm_profile_count");
        *(void **)(&profile_overflow) = dlsym(
            encoder.handle, "ck_f32_gemm_profile_overflow_calls");
        *(void **)(&profile_get) = dlsym(encoder.handle, "ck_f32_gemm_profile_get");
        if (!profile_reset || !profile_count || !profile_overflow || !profile_get) {
            fprintf(stderr, "generated encoder lacks FP32 GEMM profiling API\n");
            goto done;
        }
        ck_get_threads_fn get_threads = NULL;
        ck_threadpool_global_fn threadpool_global = NULL;
        *(void **)(&get_threads) = dlsym(encoder.handle, "ck_get_num_threads");
        *(void **)(&threadpool_global) = dlsym(encoder.handle, "ck_threadpool_global");
        *(void **)(&threadpool_profile_reset) = dlsym(
            encoder.handle, "ck_threadpool_profile_reset");
        *(void **)(&threadpool_profile_snapshot) = dlsym(
            encoder.handle, "ck_threadpool_profile_snapshot");
        if (!threadpool_global || !threadpool_profile_reset ||
            !threadpool_profile_snapshot) {
            fprintf(stderr, "generated encoder lacks thread-pool profiling API\n");
            goto done;
        }
        profile_pool = threadpool_global();
        fprintf(stderr, "audio_runtime_profile requested_threads=%s actual_threads=%d\n",
                getenv("CK_NUM_THREADS") ? getenv("CK_NUM_THREADS") : "auto",
                get_threads ? get_threads() : -1);
    }

#define LOAD_ENCODER(type, name) type name; *(void **)(&name) = ck_symbol(encoder.handle, #name)
#define LOAD_DECODER(type, name) type name; *(void **)(&name) = ck_symbol(decoder.handle, #name)
    LOAD_ENCODER(ck_int_query_fn, ck_model_audio_sample_rate);
    LOAD_ENCODER(ck_int_query_fn, ck_model_audio_max_source_frames);
    LOAD_ENCODER(ck_int_query_fn, ck_model_audio_hop_length);
    LOAD_ENCODER(ck_int_query_fn, ck_model_audio_feature_channels);
    LOAD_ENCODER(ck_int_query_fn, ck_model_audio_subsampling_factor);
    LOAD_ENCODER(ck_int_query_fn, ck_model_audio_encoder_output_dim);
    LOAD_ENCODER(ck_int_query_fn, ck_model_audio_encoder_frame_capacity);
    LOAD_ENCODER(ck_frontend_fn, ck_model_prepare_audio_wav_features);
    LOAD_ENCODER(ck_encoder_workspace_fn, ck_model_audio_encoder_workspace_bytes);
    LOAD_ENCODER(ck_encoder_fn, ck_model_run_audio_encoder);
    LOAD_DECODER(ck_set_encoder_fn, ck_model_set_encoder_memory);
    LOAD_DECODER(ck_reset_fn, ck_model_kv_cache_reset);
    LOAD_DECODER(ck_int_query_fn, ck_model_get_encoder_memory_capacity);
    LOAD_DECODER(ck_int_query_fn, ck_model_get_encoder_memory_dim);
    LOAD_DECODER(ck_int_query_fn, ck_model_get_vocab_size);
    LOAD_DECODER(ck_int_query_fn, ck_model_audio_prompt_token_count);
    LOAD_DECODER(ck_prompt_token_fn, ck_model_audio_prompt_token_id);
    LOAD_DECODER(ck_embed_fn, ck_model_embed_tokens);
    LOAD_DECODER(ck_forward_fn, ck_model_forward);
    LOAD_DECODER(ck_decode_step_fn, ck_model_decode);
    LOAD_DECODER(ck_decode_tokens_fn, ck_model_decode_tokens);
    LOAD_DECODER(ck_stop_fn, ck_model_is_stop_token);
#undef LOAD_ENCODER
#undef LOAD_DECODER
    if (!ck_model_audio_sample_rate || !ck_model_audio_max_source_frames ||
        !ck_model_audio_hop_length || !ck_model_audio_feature_channels ||
        !ck_model_audio_subsampling_factor || !ck_model_audio_encoder_output_dim ||
        !ck_model_audio_encoder_frame_capacity || !ck_model_prepare_audio_wav_features ||
        !ck_model_audio_encoder_workspace_bytes || !ck_model_run_audio_encoder ||
        !ck_model_set_encoder_memory || !ck_model_kv_cache_reset ||
        !ck_model_get_encoder_memory_capacity ||
        !ck_model_get_encoder_memory_dim || !ck_model_get_vocab_size ||
        !ck_model_audio_prompt_token_count || !ck_model_audio_prompt_token_id ||
        !ck_model_embed_tokens || !ck_model_forward ||
        !ck_model_decode || !ck_model_decode_tokens || !ck_model_is_stop_token) goto done;

    const int sample_rate = ck_model_audio_sample_rate();
    const int max_source_frames = ck_model_audio_max_source_frames();
    const int hop_length = ck_model_audio_hop_length();
    const int feature_channels = ck_model_audio_feature_channels();
    const int subsampling_factor = ck_model_audio_subsampling_factor();
    const int encoder_dim = ck_model_audio_encoder_output_dim();
    const int encoder_capacity = ck_model_audio_encoder_frame_capacity();
    if (sample_rate <= 0 || max_source_frames <= 0 || hop_length <= 0 ||
        feature_channels <= 0 || subsampling_factor <= 0 || encoder_dim <= 0 ||
        encoder_capacity <= 0 || wav.sample_rate != (uint32_t)sample_rate ||
        encoder_dim != ck_model_get_encoder_memory_dim() ||
        encoder_capacity > ck_model_get_encoder_memory_capacity()) {
        fprintf(stderr, "generated component geometry is incompatible with the input or decoder\n");
        goto done;
    }
    for (size_t index = 0; index < segment_count; ++index) {
        if (segments[index].end_frame - segments[index].start_frame >
            (uint32_t)max_source_frames) {
            fprintf(stderr, "segment %zu exceeds generated frontend capacity\n", index);
            goto done;
        }
    }
    const int feature_capacity = max_source_frames / hop_length + 1;
    const int output_capacity =
        (feature_capacity + subsampling_factor - 1) / subsampling_factor;
    if (output_capacity <= 0 || output_capacity > encoder_capacity) goto done;
    if ((size_t)feature_capacity > SIZE_MAX / (size_t)feature_channels / sizeof(float) ||
        (size_t)output_capacity > SIZE_MAX / (size_t)encoder_dim / sizeof(float)) goto done;
    features = (float *)calloc(
        (size_t)feature_capacity * (size_t)feature_channels, sizeof(float));
    encoder_output = (float *)calloc(
        (size_t)output_capacity * (size_t)encoder_dim, sizeof(float));
    const size_t workspace_bytes = ck_model_audio_encoder_workspace_bytes(
        feature_capacity, output_capacity);
    workspace = workspace_bytes ? malloc(workspace_bytes) : NULL;
    if (!features || !encoder_output || !workspace) goto done;

    const int prompt_count = ck_model_audio_prompt_token_count();
    prompt = prompt_count > 0
        ? (int32_t *)calloc((size_t)prompt_count, sizeof(int32_t)) : NULL;
    if (!prompt) goto done;
    for (int index = 0; index < prompt_count; ++index) {
        prompt[index] = ck_model_audio_prompt_token_id(index);
        if (prompt[index] < 0) {
            fprintf(stderr, "generated tokenizer lacks required prompt token at index %d\n", index);
            goto done;
        }
    }
    const int vocab_size = ck_model_get_vocab_size();
    const int max_tokens = 512;
    logits = vocab_size > 0 ? (float *)malloc((size_t)vocab_size * sizeof(float)) : NULL;
    tokens = (int32_t *)calloc((size_t)max_tokens, sizeof(int32_t));
    if (!logits || !tokens) {
        goto done;
    }
    for (size_t window_index = 0; window_index < segment_count; ++window_index) {
        CKAudioSegment segment = segments[window_index];
        size_t window_size = 0;
        uint8_t *window = ck_make_wav_window(
            &wav, segment.start_frame, segment.end_frame, &window_size);
        double frontend_started = ck_now();
        int feature_frames = 0, encoder_frames = 0, token_count = 0;
        double encoder_started, decoder_started, completed;
        memset(tokens, 0, (size_t)max_tokens * sizeof(*tokens));
        if (!window || ck_model_prepare_audio_wav_features(
                window, window_size, features, feature_capacity,
                &feature_frames, NULL) != 0) {
            free(window);
            fprintf(stderr, "generated frontend failed for window %zu\n", window_index);
            goto done;
        }
        encoder_started = ck_now();
        if (profile_reset) {
            profile_reset();
            threadpool_profile_reset(profile_pool);
        }
        if (ck_model_run_audio_encoder(
                features, feature_frames,
                (int)((segment.end_frame - segment.start_frame) / (uint32_t)hop_length),
                encoder_output, output_capacity, &encoder_frames,
                workspace, workspace_bytes) != 0) {
            free(window);
            fprintf(stderr, "generated encoder failed for window %zu\n", window_index);
            goto done;
        }
        if (profile_reset) {
            ck_print_gemm_profile(
                window_index, profile_count, profile_overflow, profile_get);
            ck_print_threadpool_profile(
                window_index, threadpool_profile_snapshot, profile_pool);
        }
        decoder_started = ck_now();
        ck_model_kv_cache_reset();
        if (ck_model_set_encoder_memory(encoder_output, encoder_frames, encoder_dim) != 0 ||
            ck_model_embed_tokens(prompt, prompt_count) != 0 ||
            ck_model_forward(logits) != 0) {
            free(window);
            fprintf(stderr, "generated decoder initialization failed for window %zu\n",
                    window_index);
            goto done;
        }
        while (token_count < max_tokens) {
            int token = ck_argmax(logits, vocab_size);
            if (ck_model_is_stop_token(token)) break;
            tokens[token_count++] = token;
            if (ck_model_decode((int32_t)token, logits) != 0) {
                free(window);
                fprintf(stderr, "generated decoder failed in window %zu at token %d\n",
                        window_index, token_count);
                goto done;
            }
        }
        if (token_count == max_tokens) {
            free(window);
            fprintf(stderr, "generated decoder reached token limit in window %zu\n",
                    window_index);
            goto done;
        }
        free(text);
        text = (char *)calloc((size_t)token_count * 32u + 1u, 1);
        if (!text || ck_model_decode_tokens(
                tokens, token_count, text, token_count * 32 + 1) <= 0 ||
            ck_append_text(&transcript, &transcript_length, &transcript_capacity,
                           text) != 0) {
            free(window);
            fprintf(stderr, "generated tokenizer failed for window %zu\n", window_index);
            goto done;
        }
        completed = ck_now();
        fprintf(stderr,
                "window=%zu source_frames=%u:%u frontend=%.6fs encoder=%.6fs "
                "decoder=%.6fs encoder_frames=%d tokens=%d\n",
                window_index, segment.start_frame, segment.end_frame,
                encoder_started - frontend_started, decoder_started - encoder_started,
                completed - decoder_started, encoder_frames, token_count);
        fprintf(stderr, "window_token_ids[%zu]=", window_index);
        for (int index = 0; index < token_count; ++index) {
            fprintf(stderr, "%s%d", index ? "," : "", (int)tokens[index]);
        }
        fputc('\n', stderr);
        last_token_count = token_count;
        consumed_frames += segment.end_frame - segment.start_frame;
        free(window);
    }
    if (segment_count == 1 && argc == 8) {
        printf("%s\n", transcript ? transcript : "");
        fputs("token_ids=", stderr);
        for (int index = 0; index < last_token_count; ++index) {
            fprintf(stderr, "%s%d", index ? "," : "", (int)tokens[index]);
        }
        fputc('\n', stderr);
    } else {
        printf("%s\n", transcript ? transcript : "");
        fprintf(stderr, "completed_windows=%zu source_frames=%u consumed_frames=%" PRIu64 "\n",
                segment_count, wav.frames, consumed_frames);
    }
    exit_code = 0;

done:
    free(prompt);
    free(text);
    free(transcript);
    free(tokens);
    free(logits);
    free(workspace);
    free(encoder_output);
    free(features);
    if (decoder_initialized) ck_close_component(&decoder);
    else if (decoder.handle) dlclose(decoder.handle);
    if (encoder_initialized) ck_close_component(&encoder);
    else if (encoder.handle) dlclose(encoder.handle);
    free(wav.bytes);
    if (segments != &single_segment) free(segments);
    return exit_code;
}
