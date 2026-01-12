# UTF-8 Decoder Implementation Plan for ck-cli-v6

## Problem Statement

The native C CLI (`ck_cli_v6.c`) outputs malformed UTF-8 characters like `Ċ` instead of proper newlines and special characters, while the Python version (`ck_chat.py`) prints correctly.

### Symptoms
```
C CLI output:  and gender is femaleĊĊ```ĊSELECT * FROM...
Expected:      and gender is female

```sql
SELECT * FROM...
```

### Root Cause

The `output_token()` function in `ck_cli_v6.c` only handles two special byte sequences:
- `Ġ` (0xC4 0xA0) → space
- `▁` (0xE2 0x96 0x81) → space

But Qwen/BPE tokenizers use many more UTF-8 sequences that are not handled.

---

## Why Python Works

Python's `ck_chat.py` uses:
```python
from tokenizers import Tokenizer  # or gguf_tokenizer
def decode(self, token_ids: list) -> str:
    return self.tokenizer.decode(token_ids)  # Returns proper Python str (UTF-8)
```

The HuggingFace `tokenizers` library handles all UTF-8 decoding internally. In C, we must replicate this.

---

## Implementation Plan

### Phase 1: Diagnostic (Debug Output)

Add debug logging to understand what bytes are being returned.

**File:** `src/v6/ck_cli_v6.c`

```c
// Add at top of file
#define DEBUG_UTF8 0  // Set to 1 to enable debug output

#if DEBUG_UTF8
static void debug_token_bytes(const char *token, const char *context) {
    fprintf(stderr, "[UTF8-DEBUG] %s: ", context);
    for (const char *p = token; *p; p++) {
        fprintf(stderr, "%02X ", (unsigned char)*p);
    }
    fprintf(stderr, "\n");
}
#else
#define debug_token_bytes(...)
#endif
```

**Modifications:**
1. Add `--debug-tokens` CLI flag to enable byte debug
2. Call `debug_token_bytes()` in `run_prompt()` before `output_token()`

**Output format:**
```
[UTF8-DEBUG] token_id=1234: C2 8A 0A 0D ...
```

---

### Phase 2: UTF-8 Validation Helpers

Create a standalone UTF-8 decoder module for validation and decoding.

**File:** `src/tokenizer/utf8_decoder.h`

```c
#ifndef UTF8_DECODER_H
#define UTF8_DECODER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/**
 * Validate a UTF-8 string.
 *
 * @param str      String to validate
 * @param len      Length of string (-1 for null-terminated)
 * @return         true if valid UTF-8
 */
bool utf8_validate(const char *str, size_t len);

/**
 * Check if a byte is a valid UTF-8 start byte.
 */
bool utf8_is_start_byte(unsigned char c);

/**
 * Get the number of continuation bytes expected for a start byte.
 */
int utf8_continuation_bytes(unsigned char c);

/**
 * Decode a UTF-8 sequence to Unicode codepoint.
 *
 * @param str      UTF-8 string pointer (will be advanced)
 * @param codepoint Output Unicode codepoint
 * @return         true if successful, false if invalid
 */
bool utf8_decode(const char **str, uint32_t *codepoint);

/**
 * Encode a Unicode codepoint to UTF-8.
 *
 * @param codepoint Unicode codepoint
 * @param out       Output buffer (min 4 bytes)
 * @return          Number of bytes written
 */
int utf8_encode(uint32_t codepoint, char *out);

#endif // UTF8_DECODER_H
```

**File:** `src/tokenizer/utf8_decoder.c`

```c
#include "utf8_decoder.h"
#include <string.h>

bool utf8_is_start_byte(unsigned char c) {
    return (c & 0x80) == 0x00 ||  // ASCII
           (c & 0xE0) == 0xC0 ||  // 2-byte
           (c & 0xF0) == 0xE0 ||  // 3-byte
           (c & 0xF8) == 0xF0;    // 4-byte
}

int utf8_continuation_bytes(unsigned char c) {
    if ((c & 0x80) == 0x00) return 0;   // ASCII
    if ((c & 0xE0) == 0xC0) return 1;   // 2-byte: 1 continuation
    if ((c & 0xF0) == 0xE0) return 2;   // 3-byte: 2 continuations
    if ((c & 0xF8) == 0xF0) return 3;   // 4-byte: 3 continuations
    return -1;  // Invalid
}

bool utf8_validate(const char *str, size_t len) {
    if (len == (size_t)-1) len = strlen(str);

    while (len > 0) {
        unsigned char c = (unsigned char)*str;
        int cont = utf8_continuation_bytes(c);

        if (cont < 0) return false;  // Invalid start byte

        if (cont == 0) {
            str++;
            len--;
            continue;
        }

        // Check continuation bytes
        if ((size_t)(cont + 1) > len) return false;

        for (int i = 1; i <= cont; i++) {
            unsigned char cont_byte = (unsigned char)str[i];
            if ((cont_byte & 0xC0) != 0x80) return false;
        }

        str += cont + 1;
        len -= cont + 1;
    }
    return true;
}

bool utf8_decode(const char **str, uint32_t *codepoint) {
    unsigned char c = (unsigned char)**str;

    if (c < 0x80) {
        *codepoint = c;
        (*str)++;
        return true;
    }

    int cont = utf8_continuation_bytes(c);
    if (cont < 0) return false;

    // Decode based on byte count
    if (cont == 1) {
        // 2-byte: 110x xxxx 10xx xxxx
        *codepoint = (c & 0x1F) << 6;
        c = (unsigned char)(*str)[1];
        *codepoint |= (c & 0x3F);
        *str += 2;
        return (c & 0xC0) == 0x80;
    }

    if (cont == 2) {
        // 3-byte: 1110 xxxx 10xx xxxx 10xx xxxx
        *codepoint = (c & 0x0F) << 12;
        c = (unsigned char)(*str)[1];
        *codepoint |= (c & 0x3F) << 6;
        c = (unsigned char)(*str)[2];
        *codepoint |= (c & 0x3F);
        *str += 3;
        return true;
    }

    if (cont == 3) {
        // 4-byte: 1111 0xxx 10xx xxxx 10xx xxxx 10xx xxxx
        *codepoint = (c & 0x07) << 18;
        c = (unsigned char)(*str)[1];
        *codepoint |= (c & 0x3F) << 12;
        c = (unsigned char)(*str)[2];
        *codepoint |= (c & 0x3F) << 6;
        c = (unsigned char)(*str)[3];
        *codepoint |= (c & 0x3F);
        *str += 4;
        return true;
    }

    return false;
}

int utf8_encode(uint32_t codepoint, char *out) {
    if (codepoint < 0x80) {
        out[0] = (char)codepoint;
        return 1;
    }
    if (codepoint < 0x800) {
        out[0] = (char)(0xC0 | (codepoint >> 6));
        out[1] = (char)(0x80 | (codepoint & 0x3F));
        return 2;
    }
    if (codepoint < 0x10000) {
        out[0] = (char)(0xE0 | (codepoint >> 12));
        out[1] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
        out[2] = (char)(0x80 | (codepoint & 0x3F));
        return 3;
    }
    // Max valid Unicode is 0x10FFFF
    out[0] = (char)(0xF0 | (codepoint >> 18));
    out[1] = (char)(0x80 | ((codepoint >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
    out[3] = (char)(0x80 | (codepoint & 0x3F));
    return 4;
}
```

---

### Phase 3: Special Character Mapping

Define common special characters that need transformation.

**File:** `src/v6/ck_utf8_special.c`

```c
#include <stdint.h>
#include <stdbool.h>

/**
 * Check if codepoint should be transformed and return replacement.
 *
 * @param codepoint  Input Unicode codepoint
 * @param out        Output buffer for replacement
 * @param out_len    Number of bytes written
 * @return           true if replacement was made
 */
bool utf8_special_transform(uint32_t codepoint, char *out, int *out_len) {
    switch (codepoint) {
        // Newline variants
        case 0x000A:  // LF (\n)
        case 0x0085:  // NEL (next line)
        case 0x2028:  // Line separator
        case 0x2029:  // Paragraph separator
            out[0] = '\n';
            *out_len = 1;
            return true;

        // Carriage return variants
        case 0x000D:  // CR (\r)
            out[0] = '\r';
            *out_len = 1;
            return true;

        // Tab
        case 0x0009:  // Tab
            out[0] = '\t';
            *out_len = 1;
            return true;

        // BPE space tokens (should become regular space)
        case 0x00A0:  // Non-breaking space
        case 0x1680:  // Ogham space mark
        case 0x2000:  // En quad
        case 0x2001:  // Em quad
        case 0x2002:  // En space
        case 0x2003:  // Em space
        case 0x2004:  // Three-per-em space
        case 0x2005:  // Four-per-em space
        case 0x2006:  // Six-per-em space
        case 0x2007:  // Figure space
        case 0x2008:  // Punctuation space
        case 0x2009:  // Thin space
        case 0x200A:  // Hair space
        case 0x202F:  // Narrow no-break space
        case 0x205F:  // Medium mathematical space
        case 0x3000:  // Ideographic space
            out[0] = ' ';
            *out_len = 1;
            return true;

        // BPE special: Ġ (U+0120 + U+00A0 merged in GPT-2 BPE)
        // Handled separately in tokenizer, but just in case
        case 0x0120:  // Latin capital letter G with breve
            out[0] = 'G';
            *out_len = 1;
            return true;

        default:
            return false;
    }
}
```

---

### Phase 4: Main Output Function

Replace `output_token()` with proper UTF-8 handling.

**File:** `src/v6/ck_utf8_output.c`

```c
#include <string.h>
#include <stdio.h>
#include "utf8_decoder.h"
#include "ck_utf8_special.h"

#define OUTPUT_BUF_SIZE 4096

/**
 * Output a UTF-8 token properly decoded.
 *
 * Handles:
 * - UTF-8 multi-byte sequences
 * - Special control characters (newlines, tabs)
 * - BPE space variants (converts to regular space)
 * - Invalid UTF-8 (replaces with �)
 *
 * @param buf       Output buffer
 * @param len       Current buffer length (updated on output)
 * @param token     Raw token string from tokenizer
 */
static void output_utf8_token(char *buf, size_t *len, const char *token) {
    if (!token || !*token) return;

    const char *p = token;

    while (*p) {
        uint32_t codepoint;

        // Decode UTF-8 sequence
        if (!utf8_decode(&p, &codepoint)) {
            // Invalid UTF-8: skip byte and output replacement char
            p++;
            const char *replacement = "\xEF\xBF\xBD";  // �
            for (int i = 0; replacement[i]; i++) {
                if (*len >= OUTPUT_BUF_SIZE - 1) {
                    fwrite(buf, 1, *len, stdout);
                    *len = 0;
                }
                buf[(*len)++] = replacement[i];
            }
            continue;
        }

        // Check for special character transformations
        char special[8];
        int special_len;
        if (utf8_special_transform(codepoint, special, &special_len)) {
            for (int i = 0; i < special_len; i++) {
                if (*len >= OUTPUT_BUF_SIZE - 1) {
                    fwrite(buf, 1, *len, stdout);
                    *len = 0;
                }
                buf[(*len)++] = special[i];
            }
            continue;
        }

        // Valid codepoint, output as-is (re-encode to UTF-8)
        char encoded[4];
        int encoded_len = utf8_encode(codepoint, encoded);

        for (int i = 0; i < encoded_len; i++) {
            if (*len >= OUTPUT_BUF_SIZE - 1) {
                fwrite(buf, 1, *len, stdout);
                *len = 0;
            }
            buf[(*len)++] = encoded[i];
        }
    }
}

/**
 * Output token with BPE-specific space handling.
 *
 * BPE tokenizers use special prefixes for spaces:
 * - Ġ (0xC4 0xA0 in UTF-8) → space at start of word
 * - ▁ (U+2581) → space before token
 *
 * This function handles these by outputting a leading space
 * if the token starts with these patterns.
 *
 * @param buf       Output buffer
 * @param len       Current buffer length
 * @param token     Raw token string
 */
static void output_bpe_token(char *buf, size_t *len, const char *token) {
    if (!token || !*token) return;

    // Check for Ġ prefix (GPT-2 BPE style)
    // Ġ is U+0120, but BPE stores it as 0xC4 0xA0
    const unsigned char *b = (const unsigned char *)token;
    if (b[0] == 0xC4 && b[1] == 0xA0) {
        if (*len >= OUTPUT_BUF_SIZE - 1) {
            fwrite(buf, 1, *len, stdout);
            *len = 0;
        }
        buf[(*len)++] = ' ';
        token += 2;
        if (!*token) return;
    }

    // Check for ▁ prefix (SentencePiece style U+2581)
    if (b[0] == 0xE2 && b[1] == 0x96 && b[2] == 0x81) {
        if (*len >= OUTPUT_BUF_SIZE - 1) {
            fwrite(buf, 1, *len, stdout);
            *len = 0;
        }
        buf[(*len)++] = ' ';
        token += 3;
        if (!*token) return;
    }

    // Handle U+00A0 (NBSP) as space
    if (b[0] == 0xC2 && b[1] == 0xA0) {
        if (*len >= OUTPUT_BUF_SIZE - 1) {
            fwrite(buf, 1, *len, stdout);
            *len = 0;
        }
        buf[(*len)++] = ' ';
        token += 2;
        if (!*token) return;
    }

    // Use main UTF-8 decoder for rest
    output_utf8_token(buf, len, token);
}
```

---

### Phase 5: Integrate with ck_cli_v6.c

Replace the existing `output_token()` with the new UTF-8 aware version.

**In `src/v6/ck_cli_v6.c`:**

```c
// Remove old output_token() function

// Add includes
#include "utf8_decoder.h"
#include "ck_utf8_special.h"
#include "ck_utf8_output.c"  // Or link separately

// Replace calls to output_token() with output_bpe_token()
// Line 294: output_token(out_buf, &out_len, word);
// Becomes:   output_bpe_token(out_buf, &out_len, word);
```

---

### Phase 6: Build System Updates

**Update `src/v6/Makefile` or `build.sh`:**

```makefile
# Add new source files
UTF8_SRCS = \
    ../tokenizer/utf8_decoder.c \
    ck_utf8_special.c

# Compile into ck_cli_v6
ck_cli_v6: $(UTF8_SRCS) ck_cli_v6.c
    gcc -O3 -o $@ $^ -lm
```

---

### Phase 7: Testing

Create comprehensive tests for UTF-8 handling.

**File:** `unittest/test_utf8_decoder.c`

```c
#include <stdio.h>
#include <string.h>
#include "utf8_decoder.h"
#include "ck_utf8_special.h"

int test_utf8_validate() {
    // Valid UTF-8
    assert(utf8_validate("hello", -1) == true);
    assert(utf8_validate("café", -1) == true);
    assert(utf8_validate("日本語", -1) == true);
    assert(utf8_validate("🎉", -1) == true);

    // Invalid UTF-8
    assert(utf8_validate("\x80", -1) == false);  // Continuation byte alone
    assert(utf8_validate("\xC0\x80", -1) == false);  // Overlong encoding

    printf("UTF-8 validation tests: PASS\n");
    return 0;
}

int test_utf8_encode_decode() {
    uint32_t codepoints[] = {0x0041, 0x00A9, 0x03B1, 0x4E2D, 0x1F389};
    char encoded[16];

    for (int i = 0; i < 5; i++) {
        int len = utf8_encode(codepoints[i], encoded);
        const char *p = encoded;
        uint32_t decoded;
        utf8_decode(&p, &decoded);
        assert(decoded == codepoints[i]);
    }

    printf("UTF-8 encode/decode tests: PASS\n");
    return 0;
}

int test_special_transform() {
    char out[8];
    int out_len;

    // Newline variants
    assert(utf8_special_transform(0x000A, out, &out_len) && out_len == 1 && out[0] == '\n');
    assert(utf8_special_transform(0x0085, out, &out_len) && out_len == 1 && out[0] == '\n');

    // Space variants
    assert(utf8_special_transform(0x00A0, out, &out_len) && out_len == 1 && out[0] == ' ');
    assert(utf8_special_transform(0x202F, out, &out_len) && out_len == 1 && out[0] == ' ');

    // Not special
    assert(utf8_special_transform('A', out, &out_len) == false);

    printf("Special transform tests: PASS\n");
    return 0;
}

int main() {
    test_utf8_validate();
    test_utf8_encode_decode();
    test_special_transform();

    printf("\nAll UTF-8 tests PASSED!\n");
    return 0;
}
```

**Run tests:**
```bash
gcc -o test_utf8_decoder unittest/test_utf8_decoder.c \
    src/tokenizer/utf8_decoder.c src/v6/ck_utf8_special.c
./test_utf8_decoder
```

---

### Phase 8: Parity Testing

Compare C CLI output with Python output.

**Script:** `scripts/test_utf8_parity.sh`

```bash
#!/bin/bash
set -e

MODEL_DIR="$HOME/.cache/ck-engine-v6/models/qwen2-0_5b-instruct-q4_k_m"
PROMPT="Give me an sql statement to extract all people from the profile table whose age is greater than 20"

# Test Python
echo "=== Python Output ==="
python scripts/ck_chat.py --model-dir "$MODEL_DIR" \
    --prompt "$PROMPT" --max-tokens 50 --no-stats 2>/dev/null

# Test C CLI
echo ""
echo "=== C CLI Output ==="
LD_LIBRARY_PATH=build:$LD_LIBRARY_PATH ./build/ck-cli-v6 \
    "$MODEL_DIR/ck-kernel-inference.so" \
    "$MODEL_DIR/weights.bump" \
    --prompt "$PROMPT" --max-tokens 50 -s 2>/dev/null

# Compare byte-by-byte (strip timing info)
echo ""
echo "=== Comparison ==="
PY_OUT=$(python scripts/ck_chat.py --model-dir "$MODEL_DIR" \
    --prompt "$PROMPT" --max-tokens 50 --no-stats 2>/dev/null)

C_OUT=$(LD_LIBRARY_PATH=build:$LD_LIBRARY_PATH ./build/ck-cli-v6 \
    "$MODEL_DIR/ck-kernel-inference.so" \
    "$MODEL_DIR/weights.bump" \
    --prompt "$PROMPT" --max-tokens 50 2>/dev/null)

if [ "$PY_OUT" = "$C_OUT" ]; then
    echo "UTF-8 output: EXACT MATCH"
else
    echo "UTF-8 output: MISMATCH"
    echo "Python bytes: $(echo "$PY_OUT" | wc -c)"
    echo "C CLI bytes:  $(echo "$C_OUT" | wc -c)"
fi
```

---

## Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `src/tokenizer/utf8_decoder.h` | UTF-8 decoder public API |
| `src/tokenizer/utf8_decoder.c` | UTF-8 validation and decoding |
| `src/v6/ck_utf8_special.c` | Special character transformations |
| `src/v6/ck_utf8_output.c` | Main output functions |
| `unittest/test_utf8_decoder.c` | Unit tests |
| `scripts/test_utf8_parity.sh` | Parity comparison script |
| `src/v6/UTF8_PLAN.md` | This document |

### Modified Files

| File | Change |
|------|--------|
| `src/v6/ck_cli_v6.c` | Replace `output_token()` with `output_bpe_token()` |
| `src/v6/Makefile` or `build.sh` | Add new source files |
| `include/Makefile` | Install new headers |

---

## Implementation Order

1. **Phase 1:** Add diagnostic debug output to understand the problem
2. **Phase 2:** Implement `utf8_decoder.c` and `utf8_decoder.h`
3. **Phase 3:** Implement `ck_utf8_special.c` for character mappings
4. **Phase 4:** Implement `ck_utf8_output.c` with main logic
5. **Phase 5:** Integrate into `ck_cli_v6.c`
6. **Phase 6:** Update build system
7. **Phase 7:** Write and run unit tests
8. **Phase 8:** Run parity tests against Python

---

## Quick Reference: Common UTF-8 Sequences in Tokenizers

| Sequence | Unicode | Meaning | Output |
|----------|---------|---------|--------|
| `0xC4 0xA0` | U+0120 + U+00A0 (BPE merge) | Word space | ` ` (space) |
| `0xE2 0x96 0x81` | U+2581 | SentencePiece space | ` ` (space) |
| `0xC2 0x8A` | U+008A | NEL (newline) | `\n` (newline) |
| `0xC2 0xA0` | U+00A0 | NBSP | ` ` (space) |
| `0x0A` | U+000A | LF | `\n` (newline) |
| `0x0D` | U+000D | CR | `\r` (carriage return) |
| `0x09` | U+0009 | Tab | `\t` (tab) |
| Invalid bytes | - | Invalid UTF-8 | `�` (replacement char) |

---

## Verification Checklist

- [ ] Python and C CLI produce identical output for ASCII text
- [ ] Python and C CLI produce identical output for UTF-8 text (accented chars, emoji)
- [ ] Newlines are properly rendered
- [ ] Tabs are properly rendered
- [ ] BPE space prefixes are handled
- [ ] Invalid UTF-8 doesn't crash
- [ ] All unit tests pass
- [ ] Parity tests pass with Python version

---

## References

- [Unicode UTF-8 Encoding](https://en.wikipedia.org/wiki/UTF-8)
- [UTF-8 Decoder Algorithm](https://www.cl.cam.ac.uk/~mgk25/unicode.html#utf-8)
- [HuggingFace tokenizers](https://huggingface.co/docs/tokenizers)
- [Original Python ck_chat.py](scripts/ck_chat.py)
