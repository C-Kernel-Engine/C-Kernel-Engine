# Model Metadata and Special Token Handling Plan

## Problem 1: Template Tokens Printed Literally

### Symptom
```
Assistant: ...I'm here to help! im_end!n!im_start!user!!im_end!n!im_start!assistant\nSure, ...
```

The `<|im_end|>` token is being printed as raw bytes instead of being recognized as EOS.

### Root Cause
- The `<|im_end|>` token ID (e.g., 151643) is being decoded to bytes
- Those bytes (`<|im_end|>` in UTF-8) aren't being recognized as a stop token
- The UTF-8 output function doesn't handle these special tokens

### Solution: Special Token Detection

In the output loop, check if the token string matches known stop tokens:

```c
static bool is_stop_token(const char *token_str) {
    if (!token_str) return false;

    /* Check for common stop sequences */
    static const char *stop_tokens[] = {
        "<|im_end|>",
        "<|endoftext|>",
        "</s>",
        "<|eot_id|>",
        "<|end|>",
        NULL
    };

    for (int i = 0; stop_tokens[i]; i++) {
        if (strcmp(token_str, stop_tokens[i]) == 0) {
            return true;
        }
    }
    return false;
}
```

Then in the generation loop:
```c
const char *word = ck_true_bpe_id_to_token(tokenizer, next_token);
if (is_stop_token(word)) {
    break;  /* Don't output stop tokens */
}
output_token(out_buf, &out_len, word);
```

---

## Problem 2: Bump File Should Include Model Capabilities

### Solution: Add Model Metadata Section to Bump

**New entry in `weights_manifest.json`:**
```json
{
  "model_metadata": {
    "model_name": "Qwen2-0.5B-Instruct",
    "architecture": "qwen2",
    "context_window": 32768,
    "special_tokens": {
      "eos_token": "<|im_end|>",
      "eos_token_id": 151643,
      "bos_token": "<|im_start|>",
      "bos_token_id": 151644,
      "stop_tokens": ["<|im_end|>", "<|eot_id|>"],
      "stop_token_ids": [151643, 151645]
    },
    "capabilities": {
      "supports_thinking": false,
      "thinking_token": null,
      "supports_tools": false,
      "supports_json_mode": false
    },
    "chat_template": {
      "type": "qwen",
      "template": "...",
      "stop_sequence": "<|im_end|>"
    }
  }
}
```

### Modify Conversion Scripts

**In `scripts/v6/convert_gguf_to_bump_v6.py`:**

```python
def get_model_metadata(config_json: Path, tokenizer_json: Path) -> Dict:
    """Extract model metadata including special tokens and capabilities."""
    metadata = {}

    # Load config
    with open(config_json, 'r') as f:
        config = json.load(f)

    metadata["model_name"] = config.get("model_type", "unknown")
    metadata["architecture"] = config.get("model_type", "unknown")
    metadata["context_window"] = config.get("max_position_embeddings", 4096)

    # Extract special tokens from config or tokenizer
    special_tokens = {}

    # Try to find EOS token in config
    eos_token = config.get("eos_token", "<|im_end|>")
    if isinstance(eos_token, dict):
        eos_token = eos_token.get("content", "<|im_end|>")
    special_tokens["eos_token"] = eos_token

    # Try to find EOS token ID in tokenizer
    if tokenizer_json.exists():
        with open(tokenizer_json, 'r') as f:
            tok_data = json.load(f)
        vocab = tok_data.get("model", {}).get("vocab", {})
        if eos_token in vocab:
            special_tokens["eos_token_id"] = vocab[eos_token]

    metadata["special_tokens"] = special_tokens

    # Detect capabilities based on model type
    arch = metadata["architecture"].lower()
    capabilities = {
        "supports_thinking": "qwen2.5" in arch or "qwen3" in arch,
        "supports_tools": "qwen2.5" in arch or "qwen3" in arch,
        "supports_json_mode": "qwen" in arch,
    }
    metadata["capabilities"] = capabilities

    return metadata
```

### Modify CLI to Read Metadata

**In `src/v6/ck_cli_v6.c`:**

```c
typedef struct {
    int32_t eos_id;
    int32_t bos_id;
    int32_t stop_ids[8];
    int stop_id_count;
    bool supports_thinking;
    const char *thinking_token;
    const char *stop_sequence;
} ModelMetadata;

static ModelMetadata g_metadata = {0};

/**
 * Load model metadata from weights_manifest.json or chat_template.json
 */
static bool load_model_metadata(const char *weights_path) {
    char path[4096];
    strncpy(path, weights_path, sizeof(path) - 25);
    char *slash = strrchr(path, '/');
    if (slash) {
        strcpy(slash + 1, "weights_manifest.json");
    } else {
        strcpy(path, "weights_manifest.json");
    }

    FILE *f = fopen(path, "r");
    if (!f) return false;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *json = (char *)malloc((size_t)size + 1);
    if (!json) {
        fclose(f);
        return false;
    }

    fread(json, 1, (size_t)size, f);
    json[size] = '\0';
    fclose(f);

    /* Parse metadata section */
    const char *meta_str = strstr(json, "\"model_metadata\"");
    if (meta_str) {
        /* Extract and parse metadata JSON */
        g_metadata.eos_id = extract_json_int(json, "eos_token_id");
        /* ... extract other fields ... */
    }

    free(json);
    return g_metadata.eos_id > 0;
}
```

---

## Problem 3: UTF-8 Token Output

The `<|im_end|>` token is being output as raw bytes. Fix the output function:

**In `src/v6/ck_cli_v6.c`:**

```c
static void output_token(char *buf, size_t *len, const char *token) {
    if (!token || !*token) return;

    /* Check for special stop tokens and don't output them */
    if (is_stop_token(token)) {
        return;  /* Skip stop tokens in output */
    }

    /* Handle UTF-8 special cases */
    const unsigned char *b = (const unsigned char *)token;

    /* Ġ (GPT-2 space) -> regular space */
    if (b[0] == 0xC4 && b[1] == 0xA0) {
        if (*len >= CK_CLI_OUTPUT_BUF_SIZE - 1) {
            fwrite(buf, 1, *len, stdout);
            *len = 0;
        }
        buf[(*len)++] = ' ';
        token += 2;
        if (!*token) return;
    }

    /* ▁ (SentencePiece space) -> regular space */
    if (b[0] == 0xE2 && b[1] == 0x96 && b[2] == 0x81) {
        if (*len >= CK_CLI_OUTPUT_BUF_SIZE - 1) {
            fwrite(buf, 1, *len, stdout);
            *len = 0;
        }
        buf[(*len)++] = ' ';
        token += 3;
        if (!*token) return;
    }

    /* Copy rest of token */
    while (*token && *len < CK_CLI_OUTPUT_BUF_SIZE - 1) {
        buf[(*len)++] = *token++;
    }
}
```

---

## Summary of Changes

### File: `src/v6/ck_cli_v6.c`

| Change | Purpose |
|--------|---------|
| Add `is_stop_token()` function | Detect stop tokens like `<\|im_end\|>` |
| Modify `output_token()` | Skip output of stop tokens |
| Add `load_model_metadata()` | Read model capabilities from manifest |
| Modify `run_prompt()` | Use metadata for EOS detection |

### File: `scripts/v6/convert_gguf_to_bump_v6.py`

| Change | Purpose |
|--------|---------|
| Add `get_model_metadata()` | Extract special tokens and capabilities |
| Modify manifest output | Include `model_metadata` section |

### File: `scripts/v6/chat_template_utils.py`

| Change | Purpose |
|--------|---------|
| Add special token detection | Find EOS/BOS/stop tokens in tokenizer |
| Add capability detection | Detect thinking/tools support |

---

## Expected Behavior After Fix

**Before:**
```
Assistant: I'm here to help! im_end!n!im_start!user!!im_end!...
```

**After:**
```
Assistant: I'm here to help!
[Stop token detected: <|im_end|>]
```

**With metadata loaded:**
```
[Template] Loaded: type detected
[Metadata] EOS=151643, stop_tokens=["<|im_end|>", "<|eot_id|>"]
Assistant: I'm here to help!
```
