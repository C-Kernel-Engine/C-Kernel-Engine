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
typedef int32_t (*ck_prompt_token_fn)(int);
typedef int (*ck_embed_fn)(const int32_t *, int);
typedef int (*ck_forward_fn)(float *);
typedef int (*ck_decode_step_fn)(int32_t, float *);
typedef int (*ck_decode_tokens_fn)(const int32_t *, int, char *, int);
typedef int (*ck_stop_fn)(int32_t);

typedef struct {
    void *handle;
    ck_init_fn init;
    ck_free_fn free_model;
} CKComponent;

typedef struct {
    uint8_t *bytes;
    size_t size;
    uint32_t frames;
    uint32_t sample_rate;
    uint16_t channels;
    uint16_t bits_per_sample;
} CKWav;

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

static int ck_parse_wav(const char *path, CKWav *wav) {
    size_t offset = 12;
    uint32_t pcm_bytes = 0;
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
            pcm_bytes = chunk_size;
        }
        offset = data + chunk_size + (chunk_size & 1u);
    }
    if (!have_fmt || !pcm_bytes || wav->channels != 1 ||
        wav->bits_per_sample != 16 || (pcm_bytes & 1u) != 0) return -5;
    wav->frames = pcm_bytes / 2u;
    return wav->frames ? 0 : -6;
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

int main(int argc, char **argv) {
    CKComponent encoder = {0}, decoder = {0};
    CKWav wav = {0};
    float *features = NULL, *encoder_output = NULL, *logits = NULL;
    void *workspace = NULL;
    int32_t *tokens = NULL;
    char *text = NULL;
    int exit_code = 1, encoder_initialized = 0, decoder_initialized = 0;

    if (argc != 8) {
        fprintf(stderr, "usage: %s ENCODER_SO ENCODER_WEIGHTS ENCODER_MAP "
                        "DECODER_SO DECODER_WEIGHTS DECODER_MAP INPUT_WAV\n", argv[0]);
        return 2;
    }
    if (ck_parse_wav(argv[7], &wav) != 0) {
        fprintf(stderr, "input must be uncompressed mono PCM16 WAV\n");
        goto done;
    }
    if (ck_open_component(&encoder, argv[1], argv[2], argv[3]) != 0) goto done;
    encoder_initialized = 1;
    if (ck_open_component(&decoder, argv[4], argv[5], argv[6]) != 0) goto done;
    decoder_initialized = 1;

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
        !ck_model_set_encoder_memory || !ck_model_get_encoder_memory_capacity ||
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
        wav.frames > (uint32_t)max_source_frames ||
        encoder_dim != ck_model_get_encoder_memory_dim() ||
        encoder_capacity > ck_model_get_encoder_memory_capacity()) {
        fprintf(stderr, "generated component geometry is incompatible with the input or decoder\n");
        goto done;
    }
    const int feature_capacity = (int)(wav.frames / (uint32_t)hop_length) + 1;
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

    double frontend_started = ck_now();
    int feature_frames = 0;
    if (ck_model_prepare_audio_wav_features(
            wav.bytes, wav.size, features, feature_capacity, &feature_frames, NULL) != 0) {
        fprintf(stderr, "generated frontend failed\n");
        goto done;
    }
    double encoder_started = ck_now();
    int encoder_frames = 0;
    if (ck_model_run_audio_encoder(
            features, feature_frames, (int)(wav.frames / (uint32_t)hop_length),
            encoder_output, output_capacity, &encoder_frames,
            workspace, workspace_bytes) != 0) {
        fprintf(stderr, "generated encoder failed\n");
        goto done;
    }
    double decoder_started = ck_now();
    if (ck_model_set_encoder_memory(encoder_output, encoder_frames, encoder_dim) != 0) {
        fprintf(stderr, "generated decoder rejected encoder memory\n");
        goto done;
    }

    const int prompt_count = ck_model_audio_prompt_token_count();
    int32_t *prompt = prompt_count > 0
        ? (int32_t *)calloc((size_t)prompt_count, sizeof(int32_t)) : NULL;
    if (!prompt) goto done;
    for (int index = 0; index < prompt_count; ++index) {
        prompt[index] = ck_model_audio_prompt_token_id(index);
        if (prompt[index] < 0) {
            fprintf(stderr, "generated tokenizer lacks required prompt token at index %d\n", index);
            free(prompt);
            goto done;
        }
    }
    const int vocab_size = ck_model_get_vocab_size();
    const int max_tokens = 512;
    logits = vocab_size > 0 ? (float *)malloc((size_t)vocab_size * sizeof(float)) : NULL;
    tokens = (int32_t *)calloc((size_t)max_tokens, sizeof(int32_t));
    if (!logits || !tokens ||
        ck_model_embed_tokens(prompt, prompt_count) != 0 ||
        ck_model_forward(logits) != 0) {
        free(prompt);
        goto done;
    }
    free(prompt);
    int token_count = 0;
    while (token_count < max_tokens) {
        int token = ck_argmax(logits, vocab_size);
        if (ck_model_is_stop_token(token)) break;
        tokens[token_count++] = token;
        if (ck_model_decode((int32_t)token, logits) != 0) {
            fprintf(stderr, "generated decoder failed at token %d\n", token_count);
            goto done;
        }
    }
    if (token_count == max_tokens) {
        fprintf(stderr, "generated decoder reached token limit without a stop token\n");
        goto done;
    }
    text = (char *)calloc((size_t)token_count * 32u + 1u, 1);
    if (!text || ck_model_decode_tokens(
            tokens, token_count, text, token_count * 32 + 1) <= 0) {
        fprintf(stderr, "generated tokenizer could not decode transcript\n");
        goto done;
    }
    double completed = ck_now();
    printf("%s\n", text);
    fprintf(stderr,
            "frontend=%.6fs encoder=%.6fs decoder=%.6fs frames=%d tokens=%d\n",
            encoder_started - frontend_started, decoder_started - encoder_started,
            completed - decoder_started, encoder_frames, token_count);
    fputs("token_ids=", stderr);
    for (int index = 0; index < token_count; ++index) {
        fprintf(stderr, "%s%d", index ? "," : "", (int)tokens[index]);
    }
    fputc('\n', stderr);
    exit_code = 0;

done:
    free(text);
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
    return exit_code;
}
