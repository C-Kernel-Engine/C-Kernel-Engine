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
typedef size_t (*ck_workspace_fn)(const uint8_t *, size_t, int);
typedef int (*ck_transcribe_fn)(const uint8_t *, size_t, int32_t *, int32_t *,
                               int, int *, void *, size_t);
typedef int (*ck_decode_fn)(const int32_t *, int, char *, int);
typedef int (*ck_int_query_fn)(void);
typedef int (*ck_output_capacity_fn)(const uint8_t *, size_t);

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
    char *text;
    size_t length;
    size_t capacity;
    uint64_t start_frame_x2;
    uint64_t end_frame_x2;
} CKWord;

static uint16_t ck_u16le(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t ck_u32le(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
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

static int ck_read_file(const char *path, uint8_t **bytes, size_t *size) {
    FILE *file = fopen(path, "rb");
    long length;
    uint8_t *data;
    if (!file) return -1;
    if (fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0 ||
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
    if (!have_fmt || !wav->pcm || wav->channels != 1 ||
        wav->sample_rate != 16000 || wav->bits_per_sample != 16 ||
        wav->pcm_bytes == 0 || (wav->pcm_bytes & 1u) != 0) return -5;
    wav->frames = wav->pcm_bytes / 2u;
    return 0;
}

static uint8_t *ck_make_window(const CKWav *wav, uint32_t start, uint32_t end,
                               size_t *size) {
    uint32_t payload = (end - start) * 2u;
    uint8_t *out = (uint8_t *)malloc((size_t)payload + 44u);
    if (!out) return NULL;
    memcpy(out, "RIFF", 4);
    ck_put_u32le(out + 4, payload + 36u);
    memcpy(out + 8, "WAVEfmt ", 8);
    ck_put_u32le(out + 16, 16);
    ck_put_u16le(out + 20, 1);
    ck_put_u16le(out + 22, 1);
    ck_put_u32le(out + 24, 16000);
    ck_put_u32le(out + 28, 32000);
    ck_put_u16le(out + 32, 2);
    ck_put_u16le(out + 34, 16);
    memcpy(out + 36, "data", 4);
    ck_put_u32le(out + 40, payload);
    memcpy(out + 44, wav->pcm + (size_t)start * 2u, payload);
    *size = (size_t)payload + 44u;
    return out;
}

static double ck_now(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec / 1.0e9;
}

static int ck_append(char **text, size_t *length, size_t *capacity,
                     const char *piece) {
    size_t amount = strlen(piece);
    if (*length > SIZE_MAX - amount - 1) return -1;
    if (*length + amount + 1 > *capacity) {
        size_t next = *capacity ? *capacity : 4096;
        while (next < *length + amount + 1) {
            if (next > SIZE_MAX / 2) return -1;
            next *= 2;
        }
        char *grown = (char *)realloc(*text, next);
        if (!grown) return -1;
        *text = grown;
        *capacity = next;
    }
    memcpy(*text + *length, piece, amount);
    *length += amount;
    (*text)[*length] = '\0';
    return 0;
}

static void ck_free_words(CKWord *words, size_t count) {
    for (size_t i = 0; i < count; ++i) free(words[i].text);
    free(words);
}

static void ck_json_string(FILE *file, const char *text) {
    fputc('"', file);
    for (const unsigned char *p = (const unsigned char *)text; *p; ++p) {
        switch (*p) {
            case '"': fputs("\\\"", file); break;
            case '\\': fputs("\\\\", file); break;
            case '\n': fputs("\\n", file); break;
            case '\r': fputs("\\r", file); break;
            case '\t': fputs("\\t", file); break;
            default:
                if (*p < 0x20) fprintf(file, "\\u%04x", (unsigned)*p);
                else fputc(*p, file);
        }
    }
    fputc('"', file);
}

static void *ck_symbol(void *library, const char *name) {
    void *symbol = dlsym(library, name);
    if (!symbol) fprintf(stderr, "missing generated symbol %s: %s\n", name, dlerror());
    return symbol;
}

int main(int argc, char **argv) {
    const uint32_t default_window = 300u * 16000u;
    const uint32_t default_overlap = 30u * 16000u;
    CKWav wav = {0};
    void *library = NULL;
    ck_init_fn init;
    ck_free_fn model_free;
    ck_workspace_fn workspace_bytes;
    ck_transcribe_fn transcribe;
    ck_decode_fn decode;
    ck_int_query_fn blank_token_id;
    ck_int_query_fn pad_token_id;
    ck_int_query_fn frame_stride_samples;
    ck_output_capacity_fn output_capacity_for_wav;
    char *transcript = NULL;
    size_t transcript_length = 0, transcript_capacity = 0;
    uint32_t window_frames = default_window, overlap_frames = default_overlap;
    uint32_t start = 0, window_index = 0;
    double started = ck_now();
    int exit_code = 1;

    if (argc < 5 || argc > 7) {
        fprintf(stderr, "usage: %s MODEL_SO WEIGHTS MANIFEST_MAP INPUT_WAV [WINDOW_SECONDS [OVERLAP_SECONDS]]\n", argv[0]);
        return 2;
    }
    if (argc >= 6) window_frames = (uint32_t)(strtod(argv[5], NULL) * 16000.0 + 0.5);
    if (argc >= 7) overlap_frames = (uint32_t)(strtod(argv[6], NULL) * 16000.0 + 0.5);
    if (!window_frames || overlap_frames >= window_frames) {
        fprintf(stderr, "invalid window policy\n");
        return 2;
    }
    if (ck_parse_wav(argv[4], &wav) != 0) {
        fprintf(stderr, "input must be uncompressed mono PCM16 16 kHz WAV\n");
        goto done;
    }
    library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) {
        fprintf(stderr, "cannot load generated model: %s\n", dlerror());
        goto done;
    }
    *(void **)(&init) = ck_symbol(library, "ck_model_init_with_manifest");
    *(void **)(&model_free) = ck_symbol(library, "ck_model_free");
    *(void **)(&workspace_bytes) = ck_symbol(library, "ck_model_audio_transcription_workspace_bytes");
    *(void **)(&transcribe) = ck_symbol(library, "ck_model_transcribe_audio_wav");
    *(void **)(&decode) = ck_symbol(library, "ck_model_decode_tokens");
    *(void **)(&blank_token_id) = ck_symbol(library, "ck_model_audio_blank_token_id");
    *(void **)(&pad_token_id) = ck_symbol(library, "ck_model_audio_pad_token_id");
    *(void **)(&frame_stride_samples) = ck_symbol(
        library, "ck_model_audio_frame_stride_samples");
    *(void **)(&output_capacity_for_wav) = ck_symbol(
        library, "ck_model_audio_transcription_output_capacity");
    if (!init || !model_free || !workspace_bytes || !transcribe || !decode ||
        !blank_token_id || !pad_token_id || !frame_stride_samples ||
        !output_capacity_for_wav) goto done;
    if (init(argv[2], argv[3]) != 0) {
        fprintf(stderr, "generated model initialization failed\n");
        goto done;
    }

    while (start < wav.frames) {
        uint32_t end = start + window_frames < wav.frames ? start + window_frames : wav.frames;
        uint32_t owner_start = window_index == 0 ? 0 : start + overlap_frames / 2u;
        uint32_t owner_end = end == wav.frames ? wav.frames : end - overlap_frames / 2u;
        size_t window_size = 0;
        uint8_t *window = ck_make_window(&wav, start, end, &window_size);
        int output_capacity = window
            ? output_capacity_for_wav(window, window_size) : 0;
        int32_t *tokens = (int32_t *)calloc((size_t)output_capacity, sizeof(int32_t));
        int32_t *durations = (int32_t *)calloc((size_t)output_capacity, sizeof(int32_t));
        size_t required = window ? workspace_bytes(window, window_size, output_capacity) : 0;
        void *workspace = required ? malloc(required) : NULL;
        int count = 0;
        double window_started = ck_now();
        int status = window && tokens && durations && workspace
            ? transcribe(window, window_size, tokens, durations, output_capacity,
                         &count, workspace, required)
            : -100;
        if (status != 0 || count <= 0 || count > output_capacity) {
            fprintf(stderr, "window %u failed: status=%d count=%d workspace=%zu\n",
                    window_index, status, count, required);
            free(workspace); free(durations); free(tokens); free(window);
            model_free();
            goto done;
        }
        CKWord *words = NULL;
        size_t word_count = 0, word_capacity = 0;
        int frame = 0;
        for (int i = 0; i < count; ++i) {
            int duration = durations[i];
            int token_start = frame;
            frame += duration;
            if (tokens[i] == blank_token_id() || tokens[i] == pad_token_id()) continue;
            char piece[512] = {0};
            if (decode(&tokens[i], 1, piece, (int)sizeof(piece)) <= 0) {
                fprintf(stderr, "token decoding failed in window %u\n", window_index);
                ck_free_words(words, word_count);
                free(workspace); free(durations); free(tokens); free(window);
                model_free();
                goto done;
            }
            int attach = word_count > 0 &&
                (piece[0] != ' ' && piece[0] != '\t' && piece[0] != '\n' &&
                 piece[0] != '\r');
            if (!attach) {
                if (word_count == word_capacity) {
                    size_t next = word_capacity ? word_capacity * 2u : 256u;
                    CKWord *grown = (CKWord *)realloc(words, next * sizeof(*words));
                    if (!grown) {
                        ck_free_words(words, word_count);
                        free(workspace); free(durations); free(tokens); free(window);
                        model_free();
                        goto done;
                    }
                    words = grown;
                    word_capacity = next;
                }
                memset(&words[word_count], 0, sizeof(words[word_count]));
                words[word_count].start_frame_x2 = (uint64_t)start * 2u +
                    (uint64_t)(2 * token_start) * (uint64_t)frame_stride_samples();
                words[word_count].end_frame_x2 = (uint64_t)start * 2u +
                    (uint64_t)(2 * (token_start + duration)) *
                        (uint64_t)frame_stride_samples();
                word_count++;
            } else {
                words[word_count - 1].end_frame_x2 = (uint64_t)start * 2u +
                    (uint64_t)(2 * (token_start + duration)) *
                        (uint64_t)frame_stride_samples();
            }
            CKWord *word = &words[word_count - 1];
            if (ck_append(&word->text, &word->length, &word->capacity, piece) != 0) {
                ck_free_words(words, word_count);
                free(workspace); free(durations); free(tokens); free(window);
                model_free();
                goto done;
            }
        }
        for (size_t i = 0; i < word_count; ++i) {
            uint64_t midpoint = (words[i].start_frame_x2 + words[i].end_frame_x2) / 2u;
            if (midpoint < (uint64_t)owner_start * 2u ||
                midpoint > (uint64_t)owner_end * 2u ||
                (midpoint == (uint64_t)owner_end * 2u && end != wav.frames)) continue;
            if (ck_append(&transcript, &transcript_length, &transcript_capacity,
                          words[i].text) != 0) {
                ck_free_words(words, word_count);
                free(workspace); free(durations); free(tokens); free(window);
                model_free();
                goto done;
            }
        }
        ck_free_words(words, word_count);
        fprintf(stderr, "window %u frames=%u:%u tokens=%d seconds=%.3f\n",
                window_index, start, end, count, ck_now() - window_started);
        free(workspace); free(durations); free(tokens); free(window);
        window_index++;
        if (end == wav.frames) break;
        start = end - overlap_frames;
    }
    model_free();
    printf("{\"schema\":\"cke.generated_audio_transcription.v1\",\"status\":\"pass\","
           "\"input_frames\":%u,\"sample_rate\":%u,\"windows\":%u,"
           "\"elapsed_seconds\":%.6f,\"transcript\":",
           wav.frames, wav.sample_rate, window_index, ck_now() - started);
    ck_json_string(stdout, transcript ? transcript : "");
    puts("}");
    exit_code = 0;

done:
    if (library) dlclose(library);
    free(transcript);
    free(wav.bytes);
    return exit_code;
}
