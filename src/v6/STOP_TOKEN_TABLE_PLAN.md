# Stop Token Lookup Table Implementation

## Overview

Create a boolean lookup table at runtime from the bump manifest for O(1) stop token detection.

---

## Implementation Plan

### 1. Modify `chat_template_utils.py`

Add function to find all stop token IDs:

```python
def find_stop_token_ids(tokenizer_json: Path) -> list[int]:
    """Find all stop token IDs from tokenizer.json."""
    stop_names = [
        "<|im_end|>",
        "<|eot_id|>",
        "</s>",
        "<|endoftext|>",
        "<|end|>",
        "<eos>",
    ]

    try:
        with open(tokenizer_json, 'r') as f:
            data = json.load(f)

        vocab = data.get('model', {}).get('vocab', {})
        stop_ids = []

        for name in stop_names:
            if name in vocab:
                stop_ids.append(vocab[name])

        return stop_ids
    except Exception:
        return []
```

---

### 2. Modify `convert_gguf_to_bump_v6.py`

In the template detection section, also extract stop token IDs:

```python
# In main(), after detecting template:
stop_token_ids = []
if os.path.exists(tokenizer_json_path):
    from .chat_template_utils import find_stop_token_ids
    stop_token_ids = find_stop_token_ids(Path(tokenizer_json_path))

# Add to manifest
if chat_template_info and chat_template_info.get("type") != "none":
    manifest = add_template_to_manifest(manifest, chat_template_info, eos_token_id)

# Add stop token IDs
manifest["special_tokens"] = {
    "eos_token_id": eos_token_id,
    "stop_token_ids": stop_token_ids,  # e.g., [151643, 151645]
}
```

---

### 3. Modify `ck_cli_v6.c`

**Add global lookup table:**

```c
/* Stop token lookup table - O(1) check */
static bool *g_stop_token_table = NULL;
static int g_vocab_size = 0;

/**
 * Initialize stop token lookup table from manifest metadata.
 *
 * @param stop_ids Array of stop token IDs
 * @param count    Number of stop token IDs
 * @param vocab_size Vocab size for table allocation
 * @return true on success
 */
static bool init_stop_token_table(const int32_t *stop_ids, int count, int vocab_size) {
    /* Allocate table (zero-initialized) */
    g_stop_token_table = (bool *)calloc((size_t)vocab_size, sizeof(bool));
    if (!g_stop_token_table) {
        fprintf(stderr, "[Error] Failed to allocate stop token table\n");
        return false;
    }

    g_vocab_size = vocab_size;

    /* Mark stop tokens */
    for (int i = 0; i < count; i++) {
        int id = stop_ids[i];
        if (id >= 0 && id < vocab_size) {
            g_stop_token_table[id] = true;
        }
    }

    fprintf(stderr, "[StopTable] Initialized with %d stop tokens\n", count);
    return true;
}

/**
 * Check if token is a stop token (O(1)).
 */
static inline bool is_stop_token_id(int32_t token_id) {
    if (!g_stop_token_table || token_id < 0 || token_id >= g_vocab_size) {
        return false;
    }
    return g_stop_token_table[token_id];
}
```

---

**Modify `load_chat_template()` to also load stop token IDs:**

```c
static bool load_chat_template(const char *weights_path) {
    // ... existing code to load JSON ...

    /* Extract stop_token_ids array */
    int32_t stop_ids[8];
    int stop_count = extract_json_int_array(json_content, "stop_token_ids", stop_ids, 8);

    /* Extract vocab_size */
    int vocab_size = extract_json_int(json_content, "vocab_size");
    if (vocab_size <= 0) {
        vocab_size = 200000;  /* Default fallback */
    }

    /* Initialize lookup table */
    if (stop_count > 0) {
        init_stop_token_table(stop_ids, stop_count, vocab_size);
    }

    free(json_copy);
    g_has_template = (g_chat_template[0] != '\0');
    return g_has_template;
}
```

---

**Modify generation loop to use lookup table:**

```c
    for (int generated = 0; generated < max_decode_tokens && !g_exit_requested; generated++) {
        if (next_token < 0) break;

        /* CHECK STOP TOKEN FIRST (before any processing) */
        if (is_stop_token_id(next_token)) {
            fprintf(stderr, "\n[Stop] Stop token %d detected\n", next_token);
            break;
        }

        /* Only decode and print if not a stop token */
        const char *word = ck_true_bpe_id_to_token(tokenizer, next_token);
        output_token(out_buf, &out_len, word);

        /* ... rest of loop ... */
    }
```

---

### 4. Add JSON Array Extraction Helper

```c
/**
 * Extract an array of integers from JSON.
 *
 * @param json      JSON string
 * @param key       Key to find
 * @param out       Output array
 * @param max_count Maximum elements to extract
 * @return Number of elements extracted
 */
static int extract_json_int_array(const char *json, const char *key, int32_t *out, int max_count) {
    if (!json || !key || !out) return 0;

    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return 0;

    /* Find the opening bracket */
    p = strchr(p, '[');
    if (!p) return 0;
    p++; /* Skip '[' */

    int count = 0;
    while (count < max_count && *p) {
        /* Skip whitespace */
        while (*p == ' ' || *p == '\t' || *p == '\n') p++;

        if (*p == ']') break;

        /* Parse integer */
        long val = strtol(p, NULL, 10);
        out[count++] = (int32_t)val;

        /* Skip to next number */
        while (*p && *p != ',' && *p != ']') p++;
        if (*p == ',') p++;
    }

    return count;
}
```

---

## Expected Output

**Startup:**
```
[Template] Loaded: type detected
[StopTable] Initialized with 2 stop tokens (151643, 151645)
```

**Generation:**
```
Assistant: Hello! How can I help you?

[Stop] Stop token 151643 detected
Timing:
  ...
```

---

## Memory Usage

| Vocab Size | Table Size | Notes |
|------------|------------|-------|
| 32K | 32 KB | Small models |
| 64K | 64 KB | LLaMA 7B |
| 151K | 151 KB | Qwen2-0.5B |
| 200K | 200 KB | LLaMA 8B |

Negligible memory cost for a huge speed improvement!

---

## Files Modified

| File | Change |
|------|--------|
| `scripts/v6/chat_template_utils.py` | Add `find_stop_token_ids()` |
| `scripts/v6/convert_gguf_to_bump_v6.py` | Save `stop_token_ids` to manifest |
| `src/v6/ck_cli_v6.c` | Add lookup table and `is_stop_token_id()` |

---

## Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    Runtime Flow                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Load manifest.json                                          │
│     └─→ Extract stop_token_ids: [151643, 151645]                │
│                                                                  │
│  2. Initialize lookup table                                     │
│     └─→ stop_table[151643] = true                               │
│         stop_table[151645] = true                               │
│                                                                  │
│  3. Generation loop                                             │
│     next_token = api->sample()                                   │
│                                                                  │
│     if (stop_table[next_token]) {  // O(1) lookup               │
│         break;  // Stop immediately!                             │
│     }                                                           │
│                                                                  │
│     // Only decode/print if not stop                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

This is the cleanest, fastest approach. No string matching, no UTF-8 handling - just a simple array lookup.
