# Embed Chat Template in Bump File Plan

## Overview

Embed the model's chat template directly in the `weights.bump` file during conversion from HF/GGUF. The C CLI then reads the template from the weights file - no codegen changes needed!

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Conversion Pipeline                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  model.gguf / HF model                                                   │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────┐                                                     │
│  │ convert_gguf_   │     Read config.json → Detect model family          │
│  │ to_bump_v6.py   │     Lookup appropriate chat template                │
│  └────────┬────────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────┐                                                     │
│  │ weights.bump    │     Write template string after weight data         │
│  │ + chat_template │     Add template metadata to weights_manifest.json  │
│  └────────┬────────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────┐                                                     │
│  │ ck_cli_v6.c     │     Read template from weights                      │
│  │                 │     Apply to prompts at runtime                     │
│  └─────────────────┘                                                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Define Template Storage Format

### In `weights_manifest.json`

Add template info to the manifest:

```json
{
  "format": "ck-bumpwgt4-v1",
  "model": "Qwen2-0.5B-Instruct",
  "chat_template": {
    "type": "qwen",
    "template": "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
    "default_system": "You are a helpful AI assistant.",
    "eos_token": "<|im_end|>",
    "eos_token_id": 151643
  },
  "entries": [...]
}
```

### In `weights.bump` file

Add template as a metadata section at the end:

```
┌─────────────────────────┐
│  BUMPWGT4 magic number  │  8 bytes
├─────────────────────────┤
│  Weight data            │  Variable size
├─────────────────────────┤
│  Template string        │  Prefixed with length (4 bytes)
│  (UTF-8 encoded)        │  Padded to 8-byte alignment
├─────────────────────────┤
│  Metadata checksum      │  32 bytes (optional)
└─────────────────────────┘
```

---

## Step 2: Create Template Detection Library

**File:** `scripts/v6/chat_template_utils.py`

```python
"""
Chat template detection and management utilities.
Used by both conversion scripts and CLI.
"""
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any


# ═══════════════════════════════════════════════════════════════════════
# Template Definitions
# ═══════════════════════════════════════════════════════════════════════

CHAT_TEMPLATES = {
    "qwen": {
        "template": (
            "<|im_start|>system\n"
            "{system}"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "{prompt}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "default_system": "You are a helpful AI assistant.",
    },
    "llama3": {
        "template": (
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "{system}"
            "<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "{prompt}"
            "<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        "default_system": "You are a helpful, honest assistant.",
    },
    "llama2": {
        "template": (
            "[INST] <<SYS>>\n"
            "{system}\n"
            "<</SYS>>\n\n"
            "{prompt}"
            " [/INST]"
        ),
        "default_system": "You are a helpful and honest assistant.",
    },
    "mistral": {
        "template": (
            "[INST] <<SYS>>\n"
            "{system}\n"
            "<</SYS>>\n\n"
            "{prompt}"
            " [/INST]"
        ),
        "default_system": "You are a helpful AI assistant.",
    },
    "gemma": {
        "template": (
            "<start_of_turn>system\n"
            "{system}"
            "<end_of_turn>\n"
            "<start_of_turn>user\n"
            "{prompt}"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        ),
        "default_system": "You are a helpful AI assistant.",
    },
    "phi3": {
        "template": (
            "<|system|>\n"
            "{system}"
            "<|end|>\n"
            "<|user|>\n"
            "{prompt}"
            "<|end|>\n"
            "<|assistant|>\n"
        ),
        "default_system": "You are a helpful AI assistant.",
    },
    "smollm": {
        "template": (
            "<|im_start|>system\n"
            "{system}"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "{prompt}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "default_system": "You are a helpful AI assistant.",
    },
    "none": {
        "template": "{prompt}",
        "default_system": "",
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Template Detection
# ═══════════════════════════════════════════════════════════════════════

def detect_template_from_model_name(model_name: str) -> Dict[str, str]:
    """Detect template from model name string."""
    model_lower = model_name.lower()

    # Direct model family detection
    if "qwen" in model_lower:
        return CHAT_TEMPLATES["qwen"]
    elif "llama-3" in model_lower or "llama3" in model_lower:
        return CHAT_TEMPLATES["llama3"]
    elif "llama-2" in model_lower or "llama2" in model_lower:
        return CHAT_TEMPLATES["llama2"]
    elif "mistral" in model_lower or "mixtral" in model_lower:
        return CHAT_TEMPLATES["mistral"]
    elif "gemma" in model_lower:
        return CHAT_TEMPLATES["gemma"]
    elif "phi-3" in model_lower or "phi3" in model_lower:
        return CHAT_TEMPLATES["phi3"]
    elif "smollm" in model_lower:
        return CHAT_TEMPLATES["smollm"]

    return CHAT_TEMPLATES["none"]


def detect_template_from_tokenizer(tokenizer_json: Path) -> Dict[str, str]:
    """Detect template from tokenizer.json content."""
    try:
        with open(tokenizer_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        vocab = data.get('model', {}).get('vocab', {})

        # Check for ChatML tokens (Qwen)
        if '<|im_start|>' in vocab or '<|im_end|>' in vocab:
            return CHAT_TEMPLATES["qwen"]

        # Check for Llama 3 tokens
        if '<|start_header_id|>' in vocab:
            return CHAT_TEMPLATES["llama3"]

        # Check for Gemma tokens
        if '<start_of_turn>' in vocab:
            return CHAT_TEMPLATES["gemma"]

    except Exception:
        pass

    return CHAT_TEMPLATES["none"]


def detect_template(config_json: Path, tokenizer_json: Optional[Path] = None) -> Dict[str, str]:
    """
    Detect chat template from model configuration.

    Args:
        config_json: Path to config.json
        tokenizer_json: Optional path to tokenizer.json

    Returns:
        Dict with 'template', 'default_system', 'type'
    """
    # Try config.json first
    try:
        with open(config_json, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Some models store template in config
        if 'chat_template' in config:
            template_str = config['chat_template']
            if isinstance(template_str, str):
                return {
                    "template": template_str,
                    "default_system": config.get('default_system_prompt', ""),
                    "type": "custom"
                }

        # Try to get model name
        model_name = config.get('model_type') or config.get('name') or ""
        if model_name:
            result = detect_template_from_model_name(model_name)
            if result["template"] != CHAT_TEMPLATES["none"]["template"]:
                return {**result, "type": "detected"}
    except Exception:
        pass

    # Try tokenizer.json
    if tokenizer_json and tokenizer_json.exists():
        result = detect_template_from_tokenizer(tokenizer_json)
        if result["template"] != CHAT_TEMPLATES["none"]["template"]:
            return {**result, "type": "detected"}

    # Default: no template
    return {**CHAT_TEMPLATES["none"], "type": "none"}


# ═══════════════════════════════════════════════════════════════════════
# EOS Token Detection
# ═══════════════════════════════════════════════════════════════════════

def find_eos_token_id(tokenizer_json: Path, eos_token: str = None) -> int:
    """Find the token ID for an EOS token in tokenizer.json."""
    try:
        with open(tokenizer_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        vocab = data.get('model', {}).get('vocab', {})

        # Try specified token first
        if eos_token:
            if eos_token in vocab:
                return vocab[eos_token]

        # Try common EOS token names
        eos_names = [
            "<|im_end|>",
            "<|endoftext|>",
            "</s>",
            "<|eot_id|>",
            "<eos>",
            "<|end|>",
        ]

        for name in eos_names:
            if name in vocab:
                return vocab[name]

    except Exception:
        pass

    return -1  # Not found


# ═══════════════════════════════════════════════════════════════════════
# Template Application
# ═══════════════════════════════════════════════════════════════════════

def apply_template(template: str, prompt: str, system: str = None) -> str:
    """
    Apply chat template to user prompt.

    Args:
        template: Chat template string with {system} and {prompt} placeholders
        prompt: User message
        system: System prompt (or use default)

    Returns:
        Formatted prompt ready for tokenization
    """
    if system is None:
        # Use empty system for templates that don't need it
        system = ""

    result = template.replace("{system}", system)
    result = result.replace("{prompt}", prompt)
    return result


# ═══════════════════════════════════════════════════════════════════════
# Serialization for Bump File
# ═══════════════════════════════════════════════════════════════════════

def serialize_template_for_bump(template_info: Dict[str, str]) -> bytes:
    """
    Serialize template info to bytes for writing to bump file.

    Format:
        [4 bytes: template_length][template_bytes][padding to 8-byte align]
        [4 bytes: system_length][system_bytes][padding to 8-byte align]
        [4 bytes: type_length][type_bytes][padding to 8-byte align]
    """
    import struct

    template_bytes = template_info["template"].encode('utf-8')
    system_bytes = template_info.get("default_system", "").encode('utf-8')
    type_bytes = template_info.get("type", "none").encode('utf-8')

    def pad_to_8(data: bytes) -> bytes:
        pad_len = (8 - (len(data) % 8)) % 8
        return data + b'\x00' * pad_len

    result = struct.pack('<I', len(template_bytes))
    result += pad_to_8(template_bytes)

    result += struct.pack('<I', len(system_bytes))
    result += pad_to_8(system_bytes)

    result += struct.pack('<I', len(type_bytes))
    result += pad_to_8(type_bytes)

    return result


def deserialize_template_from_bump(data: bytes, offset: int = 0) -> Dict[str, str]:
    """
    Deserialize template info from bump file bytes.

    Returns:
        Dict with 'template', 'default_system', 'type', and 'offset' (next read position)
    """
    import struct

    def read_chunk(data: bytes, offset: int) -> tuple[bytes, int]:
        length = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        chunk = data[offset:offset+length]
        offset += length
        # Remove padding
        pad_len = (8 - (length % 8)) % 8
        offset += pad_len
        return chunk, offset

    template, offset = read_chunk(data, offset)
    system, offset = read_chunk(data, offset)
    type_str, offset = read_chunk(data, offset)

    return {
        "template": template.decode('utf-8'),
        "default_system": system.decode('utf-8'),
        "type": type_str.decode('utf-8'),
        "offset": offset,
    }


def get_template_checksum(template_bytes: bytes) -> str:
    """Get SHA256 checksum of template for validation."""
    return hashlib.sha256(template_bytes).hexdigest()[:16]
```

---

## Step 3: Modify GGUF Conversion Script

**File:** `scripts/v6/convert_gguf_to_bump_v6.py`

Add template embedding to bump file.

```python
# Add at top
from .chat_template_utils import (
    detect_template,
    find_eos_token_id,
    serialize_template_for_bump,
    get_template_checksum,
)


def write_weights_with_template(
    weights_path: Path,
    entries: list,
    gguf_path: Path,
    template_info: Dict[str, str],
    eos_token_id: int,
) -> None:
    """
    Write weights.bump with embedded chat template.

    This is a replacement for the normal weight writing that adds
    template metadata after the weight data.
    """
    import struct

    HEADER_SIZE = 128  # Standard bump header
    DATA_ALIGN = 64    # 64-byte alignment for memory mapping

    # First pass: calculate sizes
    total_weight_size = 0
    for entry in entries:
        offset = max(entry.get("file_offset", 0), HEADER_SIZE)
        entry["bump_offset"] = offset
        size = entry.get("size", 0)
        # Align size to 64 bytes
        aligned_size = (size + DATA_ALIGN - 1) // DATA_ALIGN * DATA_ALIGN
        total_weight_size = max(total_weight_size, offset + aligned_size)
        entry["bump_size"] = aligned_size

    # Serialize template
    template_bytes = serialize_template_for_bump(template_info)
    template_checksum = get_template_checksum(template_info["template"].encode())

    # Calculate total file size
    # (Header + weight data + template + checksum + alignment)
    total_size = (
        ((total_weight_size + DATA_ALIGN - 1) // DATA_ALIGN) * DATA_ALIGN +
        len(template_bytes) +
        32  # Checksum
    )

    # Open output file
    with open(weights_path, 'wb') as f:
        # Write bump header (placeholder, filled later)
        header_pos = f.tell()
        f.write(b'\x00' * HEADER_SIZE)

        # Copy weights from GGUF
        with open(gguf_path, 'rb') as gguf:
            for entry in entries:
                # Seek to source position in GGUF
                gguf_offset = entry.get("gguf_offset", 0)
                size = entry.get("size", 0)

                if gguf_offset and size:
                    gguf.seek(gguf_offset)
                    data = gguf.read(size)

                    # Write to bump file at aligned position
                    f.seek(entry["bump_offset"])
                    f.write(data)

                    # Pad to alignment
                    pad_len = (DATA_ALIGN - (size % DATA_ALIGN)) % DATA_ALIGN
                    if pad_len:
                        f.write(b'\x00' * pad_len)

        # Write template at end
        template_offset = ((total_weight_size + DATA_ALIGN - 1) // DATA_ALIGN) * DATA_ALIGN
        f.seek(template_offset)
        f.write(template_bytes)

        # Write checksum
        checksum_pos = f.tell()
        f.write(template_checksum.encode())

        # Write header
        f.seek(0)
        f.write(b'BUMPWGT4')  # Magic
        f.write(struct.pack('<I', 1))  # Version
        f.write(struct.pack('<I', len(entries)))  # Num entries
        f.write(struct.pack('<Q', template_offset))  # Template offset
        f.write(struct.pack('<I', len(template_bytes)))  # Template size
        f.write(struct.pack('<I', eos_token_id))  # EOS token ID
        f.write(struct.pack('<Q', checksum_pos))  # Checksum position

        # Extend file to total size
        f.truncate(total_size)

    print(f"  Wrote weights.bump with chat template ({len(template_bytes)} bytes)")
    print(f"  Template type: {template_info.get('type', 'none')}")


# Modify main conversion function to add template
def main():
    # ... existing code ...

    # Detect template
    config_path = output_dir / "config.json"
    tokenizer_json = output_dir / "tokenizer.json"

    if config_path.exists():
        template_info = detect_template(config_path, tokenizer_json if tokenizer_json.exists() else None)
        eos_token_id = find_eos_token_id(tokenizer_json) if tokenizer_json.exists() else -1

        print(f"  Detected template: {template_info.get('type', 'none')}")

        # Write weights with template
        write_weights_with_template(
            weights_path,
            entries,
            gguf_path,
            template_info,
            eos_token_id,
        )
    else:
        # Fallback to normal weight writing
        write_weights_binary(weights_path, entries, gguf_path)

    # ... rest of code ...
```

---

## Step 4: Modify HF Conversion Script

**File:** `scripts/v6/convert_hf_to_bump_v6.py`

Similar modifications for HF conversion.

```python
from .chat_template_utils import (
    detect_template,
    find_eos_token_id,
    serialize_template_for_bump,
)

def main():
    # ... existing code ...

    # Detect and embed template
    config_path = Path(checkpoint) / "config.json"
    tokenizer_json = Path(checkpoint) / "tokenizer.json"

    if config_path.exists():
        template_info = detect_template(config_path, tokenizer_json if tokenizer_json.exists() else None)
        eos_token_id = find_eos_token_id(tokenizer_json) if tokenizer_json.exists() else -1

        # Write manifest with template info
        manifest["chat_template"] = {
            "type": template_info.get("type", "none"),
            "template_checksum": hashlib.sha256(
                template_info["template"].encode()
            ).hexdigest()[:16],
        }

        # Save template to separate file (easier for C to read)
        template_file = output_dir / "chat_template.json"
        with open(template_file, 'w', encoding='utf-8') as f:
            json.dump({
                "template": template_info["template"],
                "default_system": template_info.get("default_system", ""),
                "type": template_info.get("type", "none"),
                "eos_token_id": eos_token_id,
            }, f, indent=2, ensure_ascii=False)

        print(f"  Saved chat template to {template_file}")

    # ... continue with weight conversion ...
```

---

## Step 5: C CLI Reads Template from Weights

**File:** `src/v6/ck_cli_v6.c`

Add template reading and application.

```c
/* Add to ModelAPI struct */
typedef struct {
    // ... existing fields ...
    const char *chat_template;     /* Read from weights */
    const char *default_system;    /* Read from weights */
    int32_t eos_token_id;          /* Read from weights */
} ModelAPI;

/* Add template loading */
static bool load_chat_template(ModelAPI *api, const char *weights_path) {
    /* Try to load from chat_template.json first */
    char template_path[4096];
    strncpy(template_path, weights_path, sizeof(template_path) - 20);
    char *slash = strrchr(template_path, '/');
    if (slash) {
        strcpy(slash + 1, "chat_template.json");
    } else {
        strcpy(template_path, "chat_template.json");
    }

    FILE *f = fopen(template_path, "r");
    if (!f) {
        fprintf(stderr, "[Template] not found, using bare prompting\n");
        return false;
    }

    /* Parse JSON (simple parsing for speed) */
    char buf[16384];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    buf[n] = '\0';
    fclose(f);

    /* Extract template field */
    const char *tpl = strstr(buf, "\"template\"");
    if (!tpl) return false;

    /* Simple JSON string extraction - for production use a real parser */
    /* This is a placeholder for the logic */
    api->chat_template = strdup(tpl);  /* Would parse properly */

    return true;
}

/* Apply template to prompt */
static char *apply_chat_template(const char *template, const char *prompt, const char *system) {
    if (!template || !prompt) return NULL;

    size_t tpl_len = strlen(template);
    size_t prompt_len = strlen(prompt);
    size_t system_len = system ? strlen(system) : 0;

    /* Count placeholder replacements */
    int system_count = 0;
    int prompt_count = 0;
    for (const char *p = template; *p; p++) {
        if (strncmp(p, "{system}", 8) == 0) { system_count++; p += 7; }
        if (strncmp(p, "{prompt}", 8) == 0) { prompt_count++; p += 7; }
    }

    size_t total = tpl_len
                 - (system_count * 8)   /* Remove {system} */
                 - (prompt_count * 8)   /* Remove {prompt} */
                 + system_len
                 + prompt_len
                 + 1;  /* NULL */

    char *result = (char *)malloc(total);
    if (!result) return NULL;

    /* Build string with replacements */
    char *out = result;
    const char *p = template;

    while (*p) {
        if (strncmp(p, "{system}", 8) == 0) {
            if (system) { memcpy(out, system, system_len); out += system_len; }
            p += 8;
        } else if (strncmp(p, "{prompt}", 8) == 0) {
            memcpy(out, prompt, prompt_len);
            out += prompt_len;
            p += 8;
        } else {
            *out++ = *p++;
        }
    }
    *out = '\0';

    return result;
}
```

---

## Step 6: C CLI Uses Template at Runtime

**File:** `src/v6/ck_cli_v6.c` (run_prompt function)

```c
static int run_prompt(ModelAPI *api, CKTrueBPE *tokenizer, const CLIOptions *opt, const char *input) {
    // ...

    /* Apply chat template if available */
    char *formatted_input = NULL;
    const char *input_to_tokenize = input;

    if (api->chat_template && api->chat_template[0]) {
        const char *system = NULL;  /* Could add --system flag */
        if (!system) system = api->default_system ? api->default_system : "";

        formatted_input = apply_chat_template(api->chat_template, input, system);
        if (formatted_input) {
            input_to_tokenize = formatted_input;
        }
    }

    /* Tokenize the (possibly formatted) input */
    int32_t *ids = (int32_t *)malloc((size_t)ctx * sizeof(int32_t));
    int n = ck_true_bpe_encode(tokenizer, input_to_tokenize, -1, ids, ctx);

    /* Free formatted input if created */
    free(formatted_input);

    // ... rest of run_prompt ...
}
```

---

## Step 7: Update weights_manifest.json Format

**File:** `scripts/v6/convert_gguf_to_bump_v6.py` (in main)

```python
# Add to manifest
manifest = {
    "format": "ck-bumpwgt4-v1",
    "model": model_name,
    "chat_template": {
        "type": template_info.get("type", "none"),
        "template": template_info["template"],  # Only in JSON, not bump!
        "default_system": template_info.get("default_system", ""),
        "eos_token_id": eos_token_id,
    },
    "entries": entries,
}

# Write manifest
manifest_path = output_dir / "weights_manifest.json"
with open(manifest_path, 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
```

---

## Step 8: Add Template API to Model Loading

**File:** `src/v4/ckernel_model_load_v4.c`

Add accessor functions to read template from weights.

```c
/* In ckernel_model_load_v4.h */
int ck_load_chat_template(
    void *model_base,
    const char *template_out,   /* Output buffer */
    size_t template_out_size,   /* Buffer size */
    const char *system_out,     /* Output buffer for default system */
    size_t system_out_size,     /* Buffer size */
    int32_t *eos_token_id       /* Output: EOS token ID */
);

/* In ckernel_model_load_v4.c */
int ck_load_chat_template(
    void *model_base,
    char *template_out,
    size_t template_out_size,
    char *system_out,
    size_t system_out_size,
    int32_t *eos_token_id
) {
    CKModel *model = (CKModel *)model_base;

    /* Read from end of bump file */
    /* Implementation depends on bump format */

    if (template_out && template_out_size > 0) {
        strncpy(template_out, model->chat_template, template_out_size);
    }

    return 0;
}
```

---

## Step 9: Wrapper Generation Reads Template

**File:** `scripts/v6/ck_run_v6.py` (step_codegen)

```python
wrapper = f"""
// AUTO-GENERATED v6 wrapper: {model_name}
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// ...

// Chat template loaded from weights at init time
static char g_chat_template[16384] = {{0}};
static char g_default_system[1024] = {{0}};
static int32_t g_eos_token_id = -1;

/**
 * Load chat template from weights file.
 * Called automatically during ck_model_init.
 */
static int load_chat_template_from_weights(const char *weights_path) {{
    char manifest_path[4096];
    strncpy(manifest_path, weights_path, sizeof(manifest_path) - 20);
    char *slash = strrchr(manifest_path, '/');
    if (slash) {{
        strcpy(slash + 1, "weights_manifest.json");
    }} else {{
        strcpy(manifest_path, "weights_manifest.json");
    }}

    FILE *f = fopen(manifest_path, "r");
    if (!f) return -1;

    /* Parse JSON - read template from manifest */
    /* This would be generated code that parses the JSON */

    fclose(f);
    return 0;
}}
"""

# In ck_model_init:
wrapper += """
int ck_model_init(const char *weights_path) {
    if (g_initialized) return 0;

    // ... existing init ...

    // Load chat template
    load_chat_template_from_weights(weights_path);

    return 0;
}
"""
```

---

## Files Summary

### New Files

| File | Purpose |
|------|---------|
| `scripts/v6/chat_template_utils.py` | Template detection, serialization, application |

### Modified Files

| File | Change |
|------|--------|
| `scripts/v6/convert_gguf_to_bump_v6.py` | Embed template in bump file |
| `scripts/v6/convert_hf_to_bump_v6.py` | Save template to JSON, add to manifest |
| `src/v6/ck_cli_v6.c` | Read and apply template at runtime |
| `src/v4/ckernel_model_load_v4.c` | Add template loading API |
| `src/v4/ckernel_model_load_v4.h` | Add template accessor declarations |

### Output Files

| File | Contents |
|------|----------|
| `weights.bump` | Weights + embedded template (GGUF path) |
| `weights_manifest.json` | Manifest with template metadata |
| `chat_template.json` | Standalone template file (HF path) |

---

## Implementation Order

1. **Create `chat_template_utils.py`** - Template definitions and detection
2. **Modify `convert_gguf_to_bump_v6.py`** - Embed template in bump file
3. **Modify `convert_hf_to_bump_v6.py`** - Save template to JSON + manifest
4. **Modify `ck_cli_v6.c`** - Load and apply template at runtime
5. **Test conversion** - Verify template is saved correctly
6. **Test CLI** - Verify template is applied correctly
7. **Parity test** - Compare C output with Python ck_chat.py

---

## Verification Commands

```bash
# Test GGUF conversion with template
python scripts/v6/convert_gguf_to_bump_v6.py \
    --gguf ~/.cache/ck-engine-v6/models/Qwen2-0.5B-Instruct/qwen2-0_5b-instruct-q4_k_m.gguf \
    --output /tmp/test/weights.bump \
    --config-out /tmp/test/config.json \
    --manifest-out /tmp/test/weights_manifest.json

# Check manifest
cat /tmp/test/weights_manifest.json | jq '.chat_template'

# Test CLI
./build/ck-cli-v6 /tmp/test/ck-kernel-inference.so /tmp/test/weights.bump \
    --prompt "Hello" --max-tokens 20

# Compare with Python
python scripts/ck_chat.py --model-dir /tmp/test --prompt "Hello" --max-tokens 20
```

---

## Benefits of This Approach

1. **Weights are self-contained** - Template travels with weights
2. **No codegen changes** - Template detected at conversion time
3. **CLI stays simple** - Just reads template from weights
4. **Single source of truth** - Template in one place (weights/manifest)
5. **Easy updates** - Just re-convert to change template
6. **Portable** - Works for any model type without code changes
