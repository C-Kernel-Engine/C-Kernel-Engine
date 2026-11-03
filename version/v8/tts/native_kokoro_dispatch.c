/* Kokoro request selector. The bundle builder emits pinned artifact identities;
 * each selected native host performs the generated model call. */
#define _POSIX_C_SOURCE 200809L
#include "bundle_identity.h"
#include <errno.h>
#include <openssl/evp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#define CK_TTS_MAX_REQUEST_IDS 48u

static int digest_matches(const char *path, const char *expected) {
    if (!expected || strlen(expected) != 64) return 0;
    FILE *stream = fopen(path, "rb");
    if (!stream) return 0;
    EVP_MD_CTX *context = EVP_MD_CTX_new();
    unsigned char digest[EVP_MAX_MD_SIZE], bytes[65536];
    unsigned int digest_len = 0;
    int okay = context && EVP_DigestInit_ex(context, EVP_sha256(), NULL) == 1;
    size_t length;
    while (okay && (length = fread(bytes, 1, sizeof(bytes), stream)) != 0)
        okay = EVP_DigestUpdate(context, bytes, length) == 1;
    if (ferror(stream)) okay = 0;
    if (okay) okay = EVP_DigestFinal_ex(context, digest, &digest_len) == 1;
    if (fclose(stream) != 0) okay = 0;
    EVP_MD_CTX_free(context);
    if (!okay || digest_len != 32) return 0;
    static const char hex[] = "0123456789abcdef";
    for (unsigned i = 0; i < digest_len; ++i)
        if (expected[2 * i] != hex[digest[i] >> 4] ||
            expected[2 * i + 1] != hex[digest[i] & 15]) return 0;
    return 1;
}

static int joined(char *buffer, size_t capacity, const char *directory,
                  unsigned count, const char *name) {
    int result = snprintf(buffer, capacity, "%s/%u/%s", directory, count, name);
    return result > 0 && (size_t)result < capacity;
}

static int count_ids(const char *csv, unsigned *count) {
    if (!csv || !*csv || strlen(csv) > CK_TTS_MAX_REQUEST_IDS * 4u - 1u)
        return 0;
    unsigned found = 0;
    const char *cursor = csv;
    for (;;) {
        if (found == CK_TTS_MAX_REQUEST_IDS ||
            *cursor < '0' || *cursor > '9') return 0;
        char *end = NULL;
        errno = 0;
        long id = strtol(cursor, &end, 10);
        if (errno || end == cursor || id < 0 || id >= 178) return 0;
        ++found;
        if (!*end) break;
        if (*end != ',') return 0;
        cursor = end + 1;
    }
    *count = found;
    return 1;
}

int main(int argc, char **argv) {
    const int session = argc == 3 && strcmp(argv[2], "--session") == 0;
    if (argc != 5 && !session) {
        fprintf(stderr, "usage: %s bundle-dir output.wav seed phoneme-ids-csv\n"
                        "       %s bundle-dir --session (runtime bundle only)\n",
                argv[0], argv[0]);
        return 2;
    }
#ifndef CK_TTS_RUNTIME_BUNDLE
    if (session) {
        fprintf(stderr, "persistent session requires a runtime-length bundle\n");
        return 2;
    }
#endif
    struct stat existing_output;
    if (!session && (lstat(argv[2], &existing_output) == 0 || errno != ENOENT)) {
        fprintf(stderr, "output path already exists or cannot be inspected\n");
        return 6;
    }
    unsigned count = 0;
    if (!session && !count_ids(argv[4], &count)) {
        fprintf(stderr, "invalid phoneme IDs\n");
        return 2;
    }
    const ck_tts_bucket *selected = NULL;
    const size_t bucket_count = sizeof(ck_tts_buckets) / sizeof(ck_tts_buckets[0]);
    for (size_t i = 0; i < bucket_count; ++i)
        if (session || ck_tts_buckets[i].count == count)
            selected = &ck_tts_buckets[i];
    if (!selected) {
        fprintf(stderr, "unsupported phoneme count %u; supported:", count);
        for (size_t i = 0; i < sizeof(ck_tts_buckets) / sizeof(ck_tts_buckets[0]); ++i)
            fprintf(stderr, " %u", ck_tts_buckets[i].count);
        fputc('\n', stderr);
        return 2;
    }
    if (session) {
        /* Runtime bundles must bind every supported length to one host,
         * library and package before a shared worker can be trusted. */
        for (size_t i = 0; i < bucket_count; ++i)
            if (strcmp(selected->host_sha256, ck_tts_buckets[i].host_sha256) ||
                strcmp(selected->library_sha256, ck_tts_buckets[i].library_sha256) ||
                strcmp(selected->package_sha256, ck_tts_buckets[i].package_sha256)) {
                fprintf(stderr, "runtime bundle has inconsistent artifacts\n");
                return 4;
            }
    }
    char host[4096], library[4096], package[4096], voice[4096];
    int voice_length = snprintf(voice, sizeof(voice), "%s/af_heart.ckvoice", argv[1]);
    const unsigned directory_count =
#ifdef CK_TTS_RUNTIME_BUNDLE
        64;
#else
        count;
#endif
    if (!joined(host, sizeof(host), argv[1], directory_count, "native-kokoro-wav") ||
        !joined(library, sizeof(library), argv[1], directory_count, "generated.so") ||
        !joined(package, sizeof(package), argv[1], directory_count, "request.cktts") ||
        voice_length <= 0 || (size_t)voice_length >= sizeof(voice)) return 3;
    if (!digest_matches(host, selected->host_sha256) ||
        !digest_matches(library, selected->library_sha256) ||
        !digest_matches(package, selected->package_sha256) ||
        !digest_matches(voice, ck_tts_voice_sha256)) {
        fprintf(stderr, "Kokoro bundle artifact identity mismatch\n");
        return 4;
    }
    if (session) {
        char *const child[] = {host, package, "--session", voice, NULL};
        execv(host, child);
    } else {
        char *const child[] = {host, package, argv[2], argv[3], argv[4], voice, NULL};
        execv(host, child);
    }
    perror("native Kokoro host execution failed");
    return 5;
}
