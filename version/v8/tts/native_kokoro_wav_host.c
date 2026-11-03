#define _POSIX_C_SOURCE 200809L
#ifdef CK_TTS_EXPECTED_LIBRARY_SHA256
#define _GNU_SOURCE
#endif
/* Native Kokoro diagnostic host. The generated model owns every neural
 * operation; this host loads caller-owned storage, optionally fills
 * request-local excitation, and writes only the returned valid WAV samples.
 * CK_TTS_RUNTIME_WAVEFORM selects the bounded runtime-length entry ABI. */
#include "ckernel_tts.h"
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef CK_TTS_PARALLEL_CONV
#include "ck_threadpool.h"
#if !defined(CK_TTS_CONV_WORKERS) || CK_TTS_CONV_WORKERS < 1 || CK_TTS_CONV_WORKERS > 16
#error "parallel Conv1D requires a declared 1..16 worker count"
#endif
#endif
#ifdef CK_TTS_RUNTIME_WAVEFORM
#define CK_TTS_GENERATED_ENTRY ck_kokoro_runtime_waveform_bounded
#define CK_TTS_GENERATED_SYMBOL "ck_kokoro_runtime_waveform_bounded"
#else
#define CK_TTS_GENERATED_ENTRY ck_kokoro_waveform_bounded
#define CK_TTS_GENERATED_SYMBOL "ck_kokoro_waveform_bounded"
#endif
#ifdef CK_TTS_EXPECTED_LIBRARY_SHA256
#include <dlfcn.h>
#include <openssl/evp.h>

/* The resolved generated-entry symbol identifies the mapped library. The
 * digest identifies its on-disk artifact at capture time; this is not an
 * embedded build identity or proof against concurrent file replacement. */
static int verify_loaded_library(void) {
    void *symbol = dlsym(RTLD_DEFAULT, CK_TTS_GENERATED_SYMBOL);
    Dl_info info;
    if (!symbol || !dladdr(symbol, &info) || !info.dli_fname) return 0;
    FILE *stream = fopen(info.dli_fname, "rb");
    if (!stream) return 0;
    EVP_MD_CTX *context = EVP_MD_CTX_new();
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int count = 0;
    unsigned char buffer[65536];
    int okay = context && EVP_DigestInit_ex(context, EVP_sha256(), NULL) == 1;
    size_t length;
    while (okay && (length = fread(buffer, 1, sizeof(buffer), stream)) != 0)
        okay = EVP_DigestUpdate(context, buffer, length) == 1;
    if (ferror(stream)) okay = 0;
    if (okay) okay = EVP_DigestFinal_ex(context, digest, &count) == 1;
    if (fclose(stream) != 0) okay = 0;
    EVP_MD_CTX_free(context);
    const char *expected = CK_TTS_EXPECTED_LIBRARY_SHA256;
    if (!okay || count != 32 || strlen(expected) != 64) return 0;
    static const char hex[] = "0123456789abcdef";
    for (unsigned int i = 0; i < count; ++i) {
        if (expected[2 * i] != hex[digest[i] >> 4] ||
            expected[2 * i + 1] != hex[digest[i] & 15]) return 0;
    }
    fprintf(stderr, "resolved generated symbol in %s; artifact SHA-256 verified\n",
            info.dli_fname);
    return 1;
}
#endif

/* The host build declares the graph's token capacity. CKTTS003 binds exactly
 * that many IDs; CKTTS004 binds a checked valid length within capacity. */
#ifndef CK_TTS_COMPILED_TOKEN_COUNT
#define CK_TTS_COMPILED_TOKEN_COUNT 0
#endif

#ifdef CK_TTS_RUNTIME_WAVEFORM
extern int CK_TTS_GENERATED_ENTRY(uint8_t *, size_t, int32_t *, int32_t *,
    int32_t *, int32_t *, int32_t *, int32_t *, int32_t *, int32_t *, int32_t *
#ifdef CK_TTS_PARALLEL_CONV
    , void *
#endif
    );
#else
extern int CK_TTS_GENERATED_ENTRY(uint8_t *, size_t, int32_t *, int32_t *,
    int32_t *, int32_t *, int32_t *, int32_t *, int32_t *
#ifdef CK_TTS_PARALLEL_CONV
    , void *
#endif
    );
#endif

static int read_u32(FILE *stream, uint32_t *value) {
    unsigned char bytes[4];
    if (fread(bytes, 1, 4, stream) != 4) return 0;
    *value = (uint32_t)bytes[0] | (uint32_t)bytes[1] << 8 |
        (uint32_t)bytes[2] << 16 | (uint32_t)bytes[3] << 24;
    return 1;
}

static int read_u64(FILE *stream, uint64_t *value) {
    uint32_t low, high;
    if (!read_u32(stream, &low) || !read_u32(stream, &high)) return 0;
    *value = (uint64_t)low | (uint64_t)high << 32;
    return 1;
}

static int bounded_region(uint64_t offset, uint64_t elements,
    uint64_t element_bytes, uint64_t arena_bytes) {
    return element_bytes && offset <= arena_bytes &&
        !(offset % element_bytes) &&
        elements <= (arena_bytes - offset) / element_bytes;
}

static int overlaps(uint64_t a, uint64_t a_bytes,
    uint64_t b, uint64_t b_bytes) {
    return a < b + b_bytes && b < a + a_bytes;
}

static int parse_phoneme_ids(const char *text, int32_t *ids, size_t count) {
    if (count < 3 || count > 40 || !text) return 0;
    const char *cursor = text;
    for (size_t i = 0; i < count; ++i) {
        if (*cursor < '0' || *cursor > '9') return 0;
        char *end = NULL;
        errno = 0;
        const long parsed = strtol(cursor, &end, 10);
        if (errno || end == cursor || parsed < 0 || parsed >= 178) return 0;
        ids[i] = (int32_t)parsed;
        if (i + 1 == count) {
            if (*end) return 0;
        } else {
            if (*end != ',') return 0;
            cursor = end + 1;
        }
    }
    return ids[0] == 0 && ids[count - 1] == 0;
}

static int parse_runtime_phoneme_ids(const char *text, int32_t ids[64],
                                     size_t *valid_count) {
    if (!text || !*text) return 0;
    size_t count = 1;
    for (const char *cursor = text; *cursor; ++cursor)
        if (*cursor == ',' && ++count > 40) return 0;
    if (count < 3 || count > 40 || !parse_phoneme_ids(text, ids, count))
        return 0;
    for (size_t index = count; index < 64; ++index) ids[index] = 178;
    *valid_count = count;
    return 1;
}

/* CKVOICE1: magic, LE row count, LE 256 columns, then row-major FP32.
 * The importer verifies the source artifact; the host validates the native
 * package and reads only the row selected by this request. */
static int read_voice_row(const char *path, size_t row, float values[256]) {
    const uint16_t endian = 1;
    if (sizeof(float) != 4 || FLT_RADIX != 2 ||
        FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128 ||
        *(const unsigned char *)&endian != 1) return 0;
    FILE *stream = fopen(path, "rb");
    if (!stream) return 0;
    char magic[8];
    uint32_t rows, columns;
    int okay = fread(magic, 1, 8, stream) == 8 &&
        memcmp(magic, "CKVOICE1", 8) == 0 &&
        read_u32(stream, &rows) && read_u32(stream, &columns) &&
        rows <= 4096 && columns == 256 && row < rows;
    if (okay) {
        const long expected = 16L + (long)rows * 256L * 4L;
        okay = fseek(stream, 0, SEEK_END) == 0 &&
            ftell(stream) == expected &&
            fseek(stream, 16L + (long)row * 256L * 4L, SEEK_SET) == 0 &&
            fread(values, sizeof(float), 256, stream) == 256;
        for (size_t i = 0; okay && i < 256; ++i)
            if (!isfinite(values[i])) okay = 0;
    }
    if (fclose(stream) != 0) okay = 0;
    return okay;
}

static int write_u16(FILE *stream, uint16_t value) {
    return fputc(value & 255, stream) != EOF &&
        fputc((value >> 8) & 255, stream) != EOF;
}

static int write_u32(FILE *stream, uint32_t value) {
    return write_u16(stream, (uint16_t)value) &&
        write_u16(stream, (uint16_t)(value >> 16));
}

static int write_wav_header(FILE *stream, size_t count) {
    if (count > (UINT32_MAX - 36u) / 2u) return 0;
    const uint32_t payload = (uint32_t)(count * 2u);
    return fwrite("RIFF", 1, 4, stream) == 4 &&
        write_u32(stream, payload + 36u) &&
        fwrite("WAVEfmt ", 1, 8, stream) == 8 &&
        write_u32(stream, 16) && write_u16(stream, 1) &&
        write_u16(stream, 1) && write_u32(stream, 24000) &&
        write_u32(stream, 48000) && write_u16(stream, 2) &&
        write_u16(stream, 16) &&
        fwrite("data", 1, 4, stream) == 4 && write_u32(stream, payload);
}

static int append_pcm(FILE *stream, const float *samples, size_t count) {
    int okay = 1;
    for (size_t i = 0; okay && i < count; ++i) {
        if (!isfinite(samples[i])) { okay = 0; break; }
        float value = samples[i];
        if (value > 1.0f) value = 1.0f;
        if (value < -1.0f) value = -1.0f;
        const int32_t scaled = (int32_t)lrintf(value * 32767.0f);
        okay = write_u16(stream, (uint16_t)(int16_t)scaled);
    }
    return okay;
}

static int write_wav(const char *path, const float *samples, size_t count) {
    if (count > (UINT32_MAX - 36u) / 2u) return 0;
    /* O_EXCL also rejects a symlink or hard link to an existing bundle asset.
     * A preflight lstat in main gives an early error; this open closes the race. */
    int descriptor = open(path, O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (descriptor < 0) return 0;
    FILE *stream = fdopen(descriptor, "wb");
    if (!stream) {
        close(descriptor);
        remove(path);
        return 0;
    }
    int okay = write_wav_header(stream, count) &&
               append_pcm(stream, samples, count);
    if (fclose(stream) != 0) okay = 0;
    if (!okay) remove(path);
    return okay;
}

typedef struct {
    uint8_t *arena;
    size_t arena_bytes;
    uint64_t waveform_offset, waveform_capacity;
    uint64_t gaussian_offset, gaussian_capacity;
    uint64_t ids_offset, ids_count;
    uint64_t predictor_offset, decoder_offset, request_length_offset;
    int runtime_request, native_noise, request;
    void *worker_pool;
} loaded_package;

static int synthesize(loaded_package *package, const char *output,
                      const char *seed_text, const char *ids_text,
                      const char *voice_path, FILE *aggregate,
                      size_t *written_samples) {
    struct stat existing;
    if (!aggregate && (lstat(output, &existing) == 0 || errno != ENOENT)) return 10;
    uint64_t seed = 0;
    if (seed_text) {
        char *end = NULL;
        errno = 0;
        const unsigned long long parsed = strtoull(seed_text, &end, 0);
        if (errno || !end || end == seed_text || *end || seed_text[0] == '-')
            return 2;
        seed = (uint64_t)parsed;
    }
    if (package->request) {
        int32_t ids[64];
        float voice[256];
        size_t valid_count = (size_t)package->ids_count;
        if (!ids_text || !voice_path || !(package->runtime_request ?
                parse_runtime_phoneme_ids(ids_text, ids, &valid_count) :
                parse_phoneme_ids(ids_text, ids, valid_count)) ||
                !read_voice_row(voice_path, valid_count - 3u, voice)) return 4;
        memcpy(package->arena + (size_t)package->ids_offset, ids,
               (size_t)package->ids_count * 4u);
        if (package->runtime_request) {
            const int32_t length = (int32_t)valid_count;
            memcpy(package->arena + (size_t)package->request_length_offset,
                   &length, 4);
        }
        memcpy(package->arena + (size_t)package->decoder_offset, voice, 128u * 4u);
        memcpy(package->arena + (size_t)package->predictor_offset,
               voice + 128, 128u * 4u);
    }
    if (package->native_noise && audio_noise_counter_gaussian_f32(
            (float *)(package->arena + (size_t)package->gaussian_offset),
            (size_t)package->gaussian_capacity,
            (size_t)package->gaussian_capacity, seed, 0) != CK_AUDIO_EXTENT_OK)
        return 6;
    int32_t lengths[9] = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
    const int status = CK_TTS_GENERATED_ENTRY(package->arena, package->arena_bytes,
        &lengths[0], &lengths[1], &lengths[2], &lengths[3],
        &lengths[4], &lengths[5], &lengths[6]
#ifdef CK_TTS_RUNTIME_WAVEFORM
        , &lengths[7], &lengths[8]
#endif
#ifdef CK_TTS_PARALLEL_CONV
        , package->worker_pool
#endif
        );
#ifdef CK_TTS_RUNTIME_WAVEFORM
    const int sample_index = 5, frame_index = 6;
#else
    const int sample_index = 3, frame_index = 4;
#endif
    if (status || lengths[sample_index] <= 0 ||
        (uint64_t)lengths[sample_index] > package->waveform_capacity ||
        lengths[frame_index] <= 1 ||
        (int64_t)(lengths[frame_index] - 1) * 5 != lengths[sample_index])
        return 7;
    const float *waveform = (const float *)(package->arena +
                                            (size_t)package->waveform_offset);
    if (aggregate ? !append_pcm(aggregate, waveform,
                               (size_t)lengths[sample_index]) :
                    !write_wav(output, waveform,
                               (size_t)lengths[sample_index])) return 8;
    if (written_samples) *written_samples = (size_t)lengths[sample_index];
    return 0;
}

typedef struct {
    FILE *stream;
    char *temporary;
    char *output;
    size_t samples, chunks;
} wav_sequence;

static void abort_sequence(wav_sequence *sequence) {
    if (sequence->stream) fclose(sequence->stream);
    if (sequence->temporary) unlink(sequence->temporary);
    free(sequence->temporary);
    free(sequence->output);
    memset(sequence, 0, sizeof(*sequence));
}

static int begin_sequence(wav_sequence *sequence, const char *output) {
    struct stat existing;
    if (sequence->stream || !*output || strchr(output, '\t') ||
        lstat(output, &existing) == 0 || errno != ENOENT) return 0;
    const size_t length = strlen(output);
    if (length > SIZE_MAX - 16u) return 0;
    sequence->output = strdup(output);
    sequence->temporary = malloc(length + 16u);
    if (!sequence->output || !sequence->temporary) goto failed;
    snprintf(sequence->temporary, length + 16u, "%s.partial.XXXXXX", output);
    const int descriptor = mkstemp(sequence->temporary);
    if (descriptor < 0) goto failed;
    sequence->stream = fdopen(descriptor, "wb+");
    if (!sequence->stream) { close(descriptor); goto failed; }
    if (!write_wav_header(sequence->stream, 0)) goto failed;
    return 1;
failed:
    abort_sequence(sequence);
    return 0;
}

static int finish_sequence(wav_sequence *sequence) {
    if (!sequence->stream || !sequence->chunks ||
        sequence->samples > (UINT32_MAX - 36u) / 2u ||
        fseek(sequence->stream, 0, SEEK_SET) != 0 ||
        !write_wav_header(sequence->stream, sequence->samples) ||
        fflush(sequence->stream) != 0) {
        abort_sequence(sequence);
        return 0;
    }
    const int closed = fclose(sequence->stream);
    sequence->stream = NULL;
    if (closed != 0) { abort_sequence(sequence); return 0; }
    const int okay = link(sequence->temporary, sequence->output) == 0;
    if (okay) printf("WAV_READY\t%zu\t%s\n", sequence->samples,
                     sequence->output);
    abort_sequence(sequence);
    return okay;
}

/* The supervisor can keep this process alive across ordinary chunks. A STOP
 * kills the worker process; a replacement starts a fresh one. Each command
 * has its own seed, IDs, voice row, lengths and output publication. */
static int serve(loaded_package *package, const char *voice_path) {
    if (!package->runtime_request) return 2;
    puts("READY\tloads=1");
    fflush(stdout);
    char line[1024];
    wav_sequence sequence = {0};
    while (fgets(line, sizeof(line), stdin)) {
        const size_t size = strlen(line);
        if (!size || line[size - 1] != '\n') {
            int character;
            while ((character = fgetc(stdin)) != '\n' && character != EOF) {}
            abort_sequence(&sequence);
            puts("ERROR\toversized command");
            fflush(stdout);
            continue;
        }
        line[size - 1] = 0;
        if (!strcmp(line, "QUIT")) { abort_sequence(&sequence); return 0; }
        if (!strncmp(line, "BEGIN\t", 6)) {
            if (!begin_sequence(&sequence, line + 6)) puts("ERROR\tbegin");
            else puts("BEGUN");
            fflush(stdout);
            continue;
        }
        if (!strcmp(line, "END")) {
            if (!finish_sequence(&sequence)) puts("ERROR\tend");
            fflush(stdout);
            continue;
        }
        const int append = !strncmp(line, "APPEND\t", 7);
        if (!append && strncmp(line, "RUN\t", 4)) {
            puts("ERROR\tunknown command");
            fflush(stdout);
            continue;
        }
        if (!append && sequence.stream) {
            puts("ERROR\tsequence active");
            fflush(stdout);
            continue;
        }
        char *output = append ? NULL : line + 4;
        char *seed = append ? line + 7 : strchr(output, '\t');
        if (!append && seed) *seed++ = 0;
        char *ids = seed ? strchr(seed, '\t') : NULL;
        if (ids) *ids++ = 0;
        if (!seed || !ids || (!append && !*output) || !*seed || !*ids ||
            (append && (!sequence.stream || sequence.chunks >= 64))) {
            if (append) abort_sequence(&sequence);
            puts("ERROR\tinvalid request");
            fflush(stdout);
            continue;
        }
        size_t samples = 0;
        const int result = synthesize(package, output, seed, ids, voice_path,
                                      append ? sequence.stream : NULL, &samples);
        if (result) {
            if (append) abort_sequence(&sequence);
            printf("ERROR\t%d\n", result);
        } else if (append) {
            if (samples > (UINT32_MAX - 36u) / 2u - sequence.samples) {
                abort_sequence(&sequence);
                puts("ERROR\taggregate capacity");
            } else {
                sequence.samples += samples;
                ++sequence.chunks;
                printf("CHUNK_READY\t%zu\t%zu\n", samples, sequence.samples);
            }
        } else printf("WAV_READY\t%zu\t%s\n", samples, output);
        fflush(stdout);
    }
    abort_sequence(&sequence);
    return ferror(stdin) ? 4 : 0;
}

int main(int argc, char **argv) {
#ifdef CK_TTS_EXPECTED_LIBRARY_SHA256
    if (!verify_loaded_library()) {
        fprintf(stderr, "generated library identity mismatch\n");
        return 9;
    }
#endif
    const int session_mode = argc == 4 && !strcmp(argv[2], "--session");
    if (!session_mode && argc != 3 && argc != 4 && argc != 6) {
        fprintf(stderr, "usage: %s prepared-arena.cktts output.wav "
            "[seed [phoneme-ids-csv voice-bank.ckvoice]]\n"
            "       %s prepared-arena.cktts --session voice-bank.ckvoice\n",
            argv[0], argv[0]);
        return 2;
    }
    FILE *input = fopen(argv[1], "rb");
    if (!input) return 3;
    char magic[8];
    uint64_t bytes64, waveform_offset, waveform_capacity;
    uint64_t gaussian_offset = 0, gaussian_capacity = 0;
    uint64_t ids_offset = 0, ids_count = 0;
    uint64_t predictor_offset = 0, decoder_offset = 0;
    uint64_t request_length_offset = 0;
    uint32_t segments;
    if (fread(magic, 1, 8, input) != 8) {
        fclose(input); return 4;
    }
    const int runtime_request = memcmp(magic, "CKTTS004", 8) == 0;
#ifndef CK_TTS_RUNTIME_WAVEFORM
    if (runtime_request) { fclose(input); return 4; }
#endif
    const int request = runtime_request || memcmp(magic, "CKTTS003", 8) == 0;
    const int native_noise = request || memcmp(magic, "CKTTS002", 8) == 0;
    if ((!native_noise && memcmp(magic, "CKTTS001", 8) != 0) ||
        (session_mode ? !runtime_request :
         request ? argc != 6 : native_noise ? argc != 4 : argc != 3) ||
        !read_u64(input, &bytes64) || !read_u64(input, &waveform_offset) ||
        !read_u64(input, &waveform_capacity) ||
        (native_noise && (!read_u64(input, &gaussian_offset) ||
                          !read_u64(input, &gaussian_capacity))) ||
        (request && (!read_u64(input, &ids_offset) ||
                     !read_u64(input, &ids_count) ||
                     !read_u64(input, &predictor_offset) ||
                     !read_u64(input, &decoder_offset))) ||
        (runtime_request && !read_u64(input, &request_length_offset)) ||
        !read_u32(input, &segments) ||
        bytes64 > SIZE_MAX - 63u || bytes64 == 0 ||
        bytes64 > UINT64_C(4) * 1024u * 1024u * 1024u ||
        !waveform_capacity ||
        !bounded_region(waveform_offset, waveform_capacity, 4, bytes64) ||
        (native_noise && (
            !bounded_region(gaussian_offset, gaussian_capacity, 4, bytes64) ||
            gaussian_capacity != 76800u * 9u ||
            overlaps(gaussian_offset, gaussian_capacity * 4,
                waveform_offset, waveform_capacity * 4))) ||
        (request && (CK_TTS_COMPILED_TOKEN_COUNT < 3 ||
            ids_count != CK_TTS_COMPILED_TOKEN_COUNT ||
            (runtime_request ? ids_count != 64 : ids_count > 40) ||
            !bounded_region(ids_offset, ids_count, 4, bytes64) ||
            !bounded_region(predictor_offset, 128, 4, bytes64) ||
            !bounded_region(decoder_offset, 128, 4, bytes64) ||
            overlaps(ids_offset, ids_count * 4, gaussian_offset,
                gaussian_capacity * 4) ||
            overlaps(ids_offset, ids_count * 4, waveform_offset,
                waveform_capacity * 4) ||
            overlaps(predictor_offset, 512, decoder_offset, 512) ||
            overlaps(predictor_offset, 512, gaussian_offset,
                gaussian_capacity * 4) ||
            overlaps(decoder_offset, 512, gaussian_offset,
                gaussian_capacity * 4) ||
            overlaps(predictor_offset, 512, waveform_offset,
                waveform_capacity * 4) ||
            overlaps(decoder_offset, 512, waveform_offset,
                waveform_capacity * 4) ||
            overlaps(ids_offset, ids_count * 4, predictor_offset, 512) ||
            overlaps(ids_offset, ids_count * 4, decoder_offset, 512)))) {
        fclose(input); return 4;
    }
    if (runtime_request && (!bounded_region(request_length_offset, 1, 4, bytes64) ||
            overlaps(request_length_offset, 4, ids_offset, ids_count * 4) ||
            overlaps(request_length_offset, 4, predictor_offset, 512) ||
            overlaps(request_length_offset, 4, decoder_offset, 512) ||
            overlaps(request_length_offset, 4, gaussian_offset,
                     gaussian_capacity * 4) ||
            overlaps(request_length_offset, 4, waveform_offset,
                     waveform_capacity * 4))) {
        fclose(input); return 4;
    }
    const size_t arena_bytes = (size_t)bytes64;
    const size_t padded = (arena_bytes + 63u) & ~(size_t)63u;
    uint8_t *arena = aligned_alloc(64, padded);
    if (!arena) { fclose(input); return 5; }
    memset(arena, 0, padded);
    int okay = segments <= 1000000u;
    for (uint32_t i = 0; okay && i < segments; ++i) {
        uint64_t offset, length;
        okay = read_u64(input, &offset) && read_u64(input, &length) &&
            offset <= bytes64 && length <= bytes64 - offset &&
            fread(arena + (size_t)offset, 1, (size_t)length, input) == length;
    }
    if (fgetc(input) != EOF) okay = 0;
    fclose(input);
    if (!okay) { free(arena); return 6; }
    loaded_package package = {
        .arena = arena, .arena_bytes = arena_bytes,
        .waveform_offset = waveform_offset,
        .waveform_capacity = waveform_capacity,
        .gaussian_offset = gaussian_offset,
        .gaussian_capacity = gaussian_capacity,
        .ids_offset = ids_offset, .ids_count = ids_count,
        .predictor_offset = predictor_offset,
        .decoder_offset = decoder_offset,
        .request_length_offset = request_length_offset,
        .runtime_request = runtime_request,
        .native_noise = native_noise, .request = request,
    };
#ifdef CK_TTS_PARALLEL_CONV
    ck_threadpool_t *pool = ck_threadpool_create(CK_TTS_CONV_WORKERS);
    if (!pool) { free(arena); return 5; }
    package.worker_pool = pool;
#endif
    if (session_mode) {
        const int result = serve(&package, argv[3]);
#ifdef CK_TTS_PARALLEL_CONV
        ck_threadpool_destroy(pool);
#endif
        free(arena);
        return result;
    }
    size_t samples = 0;
    const int result = synthesize(&package, argv[2],
                                  argc >= 4 ? argv[3] : NULL,
                                  argc == 6 ? argv[4] : NULL,
                                  argc == 6 ? argv[5] : NULL, NULL, &samples);
#ifdef CK_TTS_PARALLEL_CONV
    ck_threadpool_destroy(pool);
#endif
    free(arena);
    if (!result) printf("wrote %zu samples at 24000 Hz\n", samples);
    return result;
}
