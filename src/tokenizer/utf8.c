/*
 * UTF-8 Utilities Implementation
 *
 * Provides UTF-8 character handling for tokenization.
 */

#include "tokenizer/utf8.h"
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

int ck_utf8_char_length(unsigned char c) {
    if ((c & 0x80) == 0) {
        return 1;  /* ASCII */
    } else if ((c & 0xE0) == 0xC0) {
        return 2;  /* 2-byte sequence */
    } else if ((c & 0xF0) == 0xE0) {
        return 3;  /* 3-byte sequence */
    } else if ((c & 0xF8) == 0xF0) {
        return 4;  /* 4-byte sequence */
    }
    return 0;  /* Invalid */
}

size_t ck_utf8_validate(const char *str, size_t len) {
    if (len == (size_t)-1) {
        len = strlen(str);
    }

    size_t i = 0;
    while (i < len) {
        unsigned char c = (unsigned char)str[i];
        int char_len = ck_utf8_char_length(c);

        if (char_len == 0) {
            return i;  /* Invalid byte */
        }

        /* Check continuation bytes */
        for (int j = 1; j < char_len; j++) {
            if (i + j >= len) {
                return i + j;  /* Truncated sequence */
            }
            unsigned char cont = (unsigned char)str[i + j];
            if ((cont & 0xC0) != 0x80) {
                return i + j;  /* Invalid continuation */
            }
        }

        /* Validate code point for overlong encodings */
        if (char_len == 2 && c < 0xC2) {
            return i;  /* Overlong 2-byte */
        } else if (char_len == 3) {
            if (c == 0xE0 && (str[i + 1] & 0xE0) == 0x80) {
                return i;  /* Overlong 3-byte */
            }
            if (c == 0xED && (str[i + 1] & 0xE0) == 0xA0) {
                return i;  /* Surrogate */
            }
        } else if (char_len == 4) {
            if (c == 0xF0 && (str[i + 1] & 0xF0) == 0x80) {
                return i;  /* Overlong 4-byte */
            }
            if (c > 0xF4) {
                return i;  /* Invalid */
            }
            if (c == 0xF4 && (str[i + 1] & 0xF0) == 0x80) {
                return i;  /* Out of range */
            }
        }

        i += char_len;
    }

    return 0;  /* Valid */
}

bool ck_utf8_is_valid(const char *str, size_t len) {
    return ck_utf8_validate(str, len) == 0;
}

size_t ck_utf8_count_chars(const char *str, size_t len) {
    if (len == (size_t)-1) {
        len = strlen(str);
    }

    size_t count = 0;
    size_t i = 0;

    while (i < len) {
        int char_len = ck_utf8_char_length((unsigned char)str[i]);
        if (char_len == 0) {
            /* Invalid UTF-8, count as 1 byte */
            i++;
        } else {
            i += char_len;
            count++;
        }
    }

    return count;
}

int32_t ck_utf8_next_char(const char **str, int *out_len) {
    if (!str || !*str || !out_len) {
        return -1;
    }

    unsigned char c = (unsigned char)(**str);
    int char_len = ck_utf8_char_length(c);

    if (char_len == 0) {
        *out_len = 1;
        (*str)++;
        return -1;
    }

    uint32_t cp;

    switch (char_len) {
        case 1:
            cp = c;
            break;
        case 2:
            cp = ck_utf8_decode_2(*str);
            break;
        case 3:
            cp = ck_utf8_decode_3(*str);
            break;
        case 4:
            cp = ck_utf8_decode_4(*str);
            break;
        default:
            cp = -1;
            break;
    }

    *out_len = char_len;
    *str += char_len;

    return (int32_t)cp;
}

int ck_utf8_from_cp(uint32_t cp, char *out) {
    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    } else if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    } else if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    } else {
        out[0] = (char)(0xF0 | (cp >> 18));
        out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[3] = (char)(0x80 | (cp & 0x3F));
        return 4;
    }
}

size_t ck_utf8_offset_to_byte(const char *str, size_t len, size_t n) {
    if (len == (size_t)-1) {
        len = strlen(str);
    }

    size_t byte_offset = 0;
    size_t char_idx = 0;

    while (byte_offset < len && char_idx < n) {
        int char_len = ck_utf8_char_length((unsigned char)str[byte_offset]);
        if (char_len == 0) {
            byte_offset++;
        } else {
            byte_offset += char_len;
        }
        char_idx++;
    }

    return byte_offset;
}

size_t ck_utf8_byte_to_offset(const char *str, size_t len, size_t byte_offset) {
    if (len == (size_t)-1) {
        len = strlen(str);
    }

    if (byte_offset >= len) {
        return ck_utf8_count_chars(str, len);
    }

    size_t char_idx = 0;
    size_t i = 0;

    while (i < byte_offset) {
        int char_len = ck_utf8_char_length((unsigned char)str[i]);
        if (char_len == 0) {
            i++;
        } else {
            i += char_len;
        }
        char_idx++;
    }

    return char_idx;
}

/* Unicode White_Space characters (common ones) */
static const struct {
    uint32_t start;
    uint32_t end;
} CK_UTF8_WHITESPACE_RANGES[] = {
    {0x0009, 0x000D},  /* Tab, LF, VT, FF, CR */
    {0x0020, 0x0020},  /* Space */
    {0x0085, 0x0085},  /* NEL */
    {0x00A0, 0x00A0},  /* NBSP */
    {0x1680, 0x1680},  /* Ogham space mark */
    {0x2000, 0x200A},  /* Various widths of spaces */
    {0x2028, 0x2029},  /* Line/Paragraph separator */
    {0x202F, 0x202F},  /* Narrow NBSP */
    {0x205F, 0x205F},  /* Medium mathematical space */
    {0x3000, 0x3000},  /* Ideographic space */
};

bool ck_utf8_is_whitespace(uint32_t cp) {
    for (size_t i = 0; i < sizeof(CK_UTF8_WHITESPACE_RANGES) / sizeof(CK_UTF8_WHITESPACE_RANGES[0]); i++) {
        if (cp >= CK_UTF8_WHITESPACE_RANGES[i].start &&
            cp <= CK_UTF8_WHITESPACE_RANGES[i].end) {
            return true;
        }
    }
    return false;
}

/* Simple NFC normalization (handles common cases) */
size_t ck_tokenizer_utf8_normalize_nfc(const char *src, size_t src_len,
                                       char *dst, size_t dst_size) {
    if (src_len == (size_t)-1) {
        src_len = strlen(src);
    }

    /* For now, just copy (full NFC is complex) */
    if (dst == NULL) {
        return src_len;
    }

    size_t to_copy = src_len < dst_size - 1 ? src_len : dst_size - 1;
    memcpy(dst, src, to_copy);
    dst[to_copy] = '\0';

    return to_copy;
}
