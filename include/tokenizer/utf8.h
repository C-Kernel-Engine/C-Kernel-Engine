/*
 * UTF-8 Utilities
 *
 * Provides UTF-8 character handling for tokenization.
 * Handles multi-byte sequences, validation, and normalization.
 *
 * By Anthony Shivakumar
 */

#ifndef CK_TOKENIZER_UTF8_H
#define CK_TOKENIZER_UTF8_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Get the length of a UTF-8 character from its first byte.
 *
 * @param c        First byte of UTF-8 character
 * @return         Number of bytes in character (1-4), or 0 for invalid
 */
int ck_utf8_char_length(unsigned char c);

/**
 * Validate a UTF-8 string.
 *
 * @param str      String to validate
 * @param len      Length in bytes, or -1 for null-terminated
 * @return         0 if valid, position of first invalid byte otherwise
 */
size_t ck_utf8_validate(const char *str, size_t len);

/**
 * Check if a byte sequence is valid UTF-8.
 *
 * @param str      String to check
 * @param len      Length in bytes
 * @return         true if all bytes form valid UTF-8
 */
bool ck_utf8_is_valid(const char *str, size_t len);

/**
 * Count UTF-8 characters in a string.
 *
 * @param str      String to count
 * @param len      Length in bytes, or -1 for null-terminated
 * @return         Number of Unicode code points
 */
size_t ck_utf8_count_chars(const char *str, size_t len);

/**
 * Get next UTF-8 character, return its code point.
 *
 * @param str      String pointer (will be updated)
 * @param out_len  Output: length of character in bytes
 * @return         Unicode code point, or -1 on error
 */
int32_t ck_utf8_next_char(const char **str, int *out_len);

/**
 * Write a Unicode code point as UTF-8.
 *
 * @param cp       Unicode code point
 * @param out      Output buffer (must have 4+ bytes)
 * @return         Number of bytes written
 */
int ck_utf8_from_cp(uint32_t cp, char *out);

/**
 * Get the byte offset of the N-th character.
 *
 * @param str      String
 * @param len      String length in bytes
 * @param n        Character index (0-based)
 * @return         Byte offset, or len if n >= char count
 */
size_t ck_utf8_offset_to_byte(const char *str, size_t len, size_t n);

/**
 * Get the character index from byte offset.
 *
 * @param str      String
 * @param len      String length in bytes
 * @param byte_offset  Byte offset
 * @return         Character index, or total chars if offset beyond end
 */
size_t ck_utf8_byte_to_offset(const char *str, size_t len, size_t byte_offset);

/**
 * Check if character is whitespace (Unicode White_Space property).
 *
 * @param cp       Unicode code point
 * @return         true if whitespace
 */
bool ck_utf8_is_whitespace(uint32_t cp);

/**
 * Normalize UTF-8 string (Unicode normalization form NFC).
 *
 * @param src      Source string
 * @param src_len  Source length
 * @param dst      Destination buffer
 * @param dst_size Destination size
 * @return         Length written, or required size if dst=NULL
 */
size_t ck_utf8_normalize_nfc(const char *src, size_t src_len,
                             char *dst, size_t dst_size);

/**
 * Get the first byte of a UTF-8 character.
 */
static inline unsigned char ck_utf8_first_byte(const char *s) {
    return (unsigned char)s[0];
}

/**
 * Get the continuation byte mask and value.
 */
static inline int ck_utf8_is_continuation(unsigned char c) {
    return (c & 0xC0) == 0x80;
}

/**
 * Get 2-byte UTF-8 sequence value.
 */
static inline uint32_t ck_utf8_decode_2(const char *s) {
    return ((s[0] & 0x1F) << 6) | (s[1] & 0x3F);
}

/**
 * Get 3-byte UTF-8 sequence value.
 */
static inline uint32_t ck_utf8_decode_3(const char *s) {
    return ((s[0] & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F);
}

/**
 * Get 4-byte UTF-8 sequence value.
 */
static inline uint32_t ck_utf8_decode_4(const char *s) {
    return ((s[0] & 0x07) << 18) | ((s[1] & 0x3F) << 12) |
           ((s[2] & 0x3F) << 6) | (s[3] & 0x3F);
}

/* Common Unicode code points */
#define CK_UTF8_SPACE           0x0020
#define CK_UTF8_TAB             0x0009
#define CK_UTF8_NEWLINE         0x000A
#define CK_UTF8_CARRIAGE_RETURN 0x000D
#define CK_UTF8_NBSP            0x00A0
#define CK_UTF8_WORD_JOINER     0x2060

/* GPT-2 space marker: Ġ (U+0120 in UTF-8: 0xC4 0xA0) */
#define CK_UTF8_GPT2_SPACE_HIGH 0xC4
#define CK_UTF8_GPT2_SPACE_LOW  0xA0

/* SentencePiece space marker: ▁ (U+2581) */
#define CK_UTF8_SPM_SPACE       0xE2 0x96 0x81

#ifdef __cplusplus
}
#endif

#endif /* CK_TOKENIZER_UTF8_H */
