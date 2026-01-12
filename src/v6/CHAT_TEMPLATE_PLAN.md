# Compile-Time Chat Template Embedding Plan

## Overview

Instead of handling chat templates in the CLI at runtime (Python `ck_chat.py`), embed the correct template directly into the generated C code during codegen. This makes the CLI simpler and ensures model-specific templates are always applied correctly.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Current Approach (Python CLI)                    │
├─────────────────────────────────────────────────────────────────────────┤
│  User: "Hello"                                                           │
│  CLI adds template: "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n" │
│  Model sees: Template + prompt                                           │
│  Output: Clean chat response                                             │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    Proposed Approach (Compile-Time)                      │
├─────────────────────────────────────────────────────────────────────────┤
│  codegen_v6.py reads model config → detects "qwen" template              │
│  Embeds template in generated C code as string constant                  │
│  User: "Hello"                                                           │
│  CLI passes "Hello" → Generated code adds template → Model sees both     │
│  Output: Clean chat response                                             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Why This Approach is Better

| Aspect | Runtime (Python) | Compile-Time (C) |
|--------|------------------|------------------|
| CLI complexity | Must parse `--chat-template` flag | Simple pass-through |
| Model-specific | CLI must guess/accept user input | Template baked in per-model |
| C CLI parity | Must reimplement all templates | Template is C string constant |
| Updates | CLI code changes | Just regenerate code |
| Size | Templates in binary | Templates in generated .c |

---

## Plan Summary

1. **Detect** model type from config (Qwen, Llama, Mistral, etc.)
2. **Lookup** the correct chat template for that model family
3. **Embed** template as C string constant in generated wrapper code
4. **Apply** template in `ck_model_embed_tokens()` or before tokenization
5. **Test** parity with Python version

---

## Step 1: Define Chat Templates

**File:** `scripts/v6/chat_templates.py`

```python
"""
Chat templates for different model families.

Templates use these placeholders:
  {system}  - System prompt
  {prompt}  - User message

All templates end with the assistant prefix (model starts generating here).
"""

CHAT_TEMPLATES = {
    # Qwen/Qwen2 chat format (ChatML)
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
        "eos_token": "<|im_end|>",
    },

    # LLaMA 3 / Llama 3.1 / 3.2
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
        "default_system": "You are a helpful assistant.",
        "eos_token": "<|eot_id|>",
    },

    # LLaMA 2 / earlier Llama
    "llama2": {
        "template": (
            "[INST] <<SYS>>\n"
            "{system}\n"
            "<</SYS>>\n\n"
            "{prompt}"
            " [/INST]"
        ),
        "default_system": "You are a helpful and honest assistant.",
        "eos_token": "</s>",
    },

    # Mistral / Mixtral
    "mistral": {
        "template": (
            "[INST] <<SYS>>\n"
            "{system}\n"
            "<</SYS>>\n\n"
            "{prompt}"
            " [/INST]"
        ),
        "default_system": "You are a helpful AI assistant.",
        "eos_token": "</s>",
    },

    # Gemma
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
        "eos_token": "<end_of_turn>",
    },

    # Phi-3
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
        "eos_token": "<|end|>",
    },

    # SmolLM
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
        "eos_token": "<|im_end|>",
    },

    # Generic / No template (bare prompting)
    "none": {
        "template": "{prompt}",
        "default_system": "",
        "eos_token": None,
    },
}


def get_template_for_model(model_name: str, tokenizer_json: str = None) -> dict:
    """
    Determine the appropriate chat template for a model.

    Args:
        model_name: Model name (e.g., "Qwen2-0.5B-Instruct")
        tokenizer_json: Path to tokenizer.json (can be inspected for template)

    Returns:
        Dict with 'template', 'default_system', 'eos_token'
    """
    model_lower = model_name.lower()

    # Check model name patterns
    if "qwen" in model_lower:
        return CHAT_TEMPLATES["qwen"]
    elif "llama-3" in model_lower or "llama3" in model_lower:
        return CHAT_TEMPLATES["llama3"]
    elif "llama2" in model_lower or "llama-2" in model_lower:
        return CHAT_TEMPLATES["llama2"]
    elif "mistral" in model_lower or "mixtral" in model_lower:
        return CHAT_TEMPLATES["mistral"]
    elif "gemma" in model_lower:
        return CHAT_TEMPLATES["gemma"]
    elif "phi-3" in model_lower or "phi3" in model_lower:
        return CHAT_TEMPLATES["phi3"]
    elif "smollm" in model_lower:
        return CHAT_TEMPLATES["smollm"]

    # Try to detect from tokenizer.json
    if tokenizer_json:
        try:
            import json
            with open(tokenizer_json, 'r') as f:
                tok_data = json.load(f)

            # Check for ChatML tokens
            vocab = tok_data.get('model', {}).get('vocab', {})
            if '<|im_start|>' in vocab:
                return CHAT_TEMPLATES["qwen"]

            # Check for Llama 3 tokens
            if '<|start_header_id|>' in vocab:
                return CHAT_TEMPLATES["llama3"]
        except Exception:
            pass

    # Default: no template
    return CHAT_TEMPLATES["none"]
```

---

## Step 2: Modify Codegen to Embed Template

**File:** `scripts/v6/codegen_v6.py`

Add template embedding to the wrapper code generation.

```python
# At top of file, add:
from .chat_templates import get_template_for_model, CHAT_TEMPLATES


def escape_c_string(s: str) -> str:
    """Escape special characters for C string literal."""
    s = s.replace('\\', '\\\\')
    s = s.replace('"', '\\"')
    s = s.replace('\n', '\\n')
    s = s.replace('\t', '\\t')
    s = s.replace('\r', '\\r')
    return s


def generate_chat_template_section(layout_json: dict, output_dir: Path) -> dict:
    """
    Generate C code for chat template embedding.

    Returns dict with template info for use in wrapper.
    """
    # Get model name
    model_name = layout_json.get("model", "unknown")
    if isinstance(model_name, dict):
        model_name = model_name.get("name", "unknown")

    # Find tokenizer.json
    tokenizer_json = None
    for path in [output_dir / "tokenizer.json",
                 output_dir.parent / "tokenizer.json"]:
        if path.exists():
            tokenizer_json = str(path)
            break

    # Get template for this model
    template_info = get_template_for_model(model_name, tokenizer_json)
    template_str = template_info["template"]
    default_system = template_info["default_system"]
    eos_token = template_info["eos_token"]

    # Generate C code for template
    template_c_name = f"{safe_name.upper()}_CHAT_TEMPLATE"
    system_c_name = f"{safe_name.upper()}_DEFAULT_SYSTEM"

    template_c = f'''
/* ============================================================================
 * CHAT TEMPLATE (Auto-embedded during codegen)
 * ============================================================================
 * Model: {model_name}
 * Template type: {template_info.get("type", "unknown")}
 *
 * Template string:
 * {template_str[:100]}{"..." if len(template_str) > 100 else ""}
 * ============================================================================
 */

static const char *{template_c_name} = "{escape_c_string(template_str)}";
static const char *{system_c_name} = "{escape_c_string(default_system)}";

/**
 * Apply chat template to user prompt.
 *
 * @param user_prompt  Raw user input
 * @param system_prompt System prompt (or NULL to use default)
 * @return Allocated string with template applied (caller must free)
 */
static char *{safe_name_lower}_apply_chat_template(
    const char *user_prompt,
    const char *system_prompt
) {{
    if (!user_prompt) return NULL;

    const char *system = system_prompt ? system_prompt : {system_c_name};

    // Calculate required size
    size_t system_len = system ? strlen(system) : 0;
    size_t prompt_len = strlen(user_prompt);
    size_t template_len = strlen({template_c_name});

    // Count {system} and {prompt} placeholders
    int system_count = 0;
    int prompt_count = 0;
    for (const char *p = {template_c_name}; *p; p++) {{
        if (p[0] == '{{' && p[1] == 's') system_count++;
        if (p[0] == '{{' && p[1] == 'p') prompt_count++;
    }}

    size_t total_len = template_len
                     - (system_count * 8)   // Remove "{{system}}"
                     - (prompt_count * 9)   // Remove "{{prompt}}"
                     + system_len
                     + prompt_len
                     + 1;  // NULL terminator

    char *result = (char *)malloc(total_len);
    if (!result) return NULL;

    // Build the formatted string
    char *out = result;
    const char *p = {template_c_name};

    while (*p) {{
        if (p[0] == '{' && p[1] == '{' && strncmp(p, "{{system}}", 9) == 0) {{
            if (system) {{
                memcpy(out, system, system_len);
                out += system_len;
            }}
            p += 9;
        }} else if (p[0] == '{' && p[1] == '{' && strncmp(p, "{{prompt}}", 9) == 0) {{
            memcpy(out, user_prompt, prompt_len);
            out += prompt_len;
            p += 9;
        }} else {{
            *out++ = *p++;
        }}
    }}
    *out = '\\0';

    return result;
}}
'''

    # Write template C code to file
    template_file = output_dir / "ck-chat-template.c"
    template_file.write_text(template_c)

    return {
        "template_c_name": template_c_name,
        "system_c_name": system_c_name,
        "template_file": template_file,
        "eos_token": eos_token,
    }
```

---

## Step 3: Modify step_codegen() in ck_run_v6.py

**File:** `scripts/v6/ck_run_v6.py`

Update the wrapper generation to use the embedded template.

```python
def step_codegen(layout_path: Path, output_dir: Path, force: bool = False) -> Path:
    """Generate v6 wrapper C code with embedded chat template."""

    # ... existing code ...

    # NEW: Generate chat template section
    from .chat_templates import CHAT_TEMPLATES
    from .codegen_v6 import generate_chat_template_section

    try:
        with layout_path.open("r", encoding="utf-8") as f:
            layout_json = json.load(f)
        template_info = generate_chat_template_section(layout_json, output_dir)

        # Track template info for later steps
        template_c_name = template_info["template_c_name"]
        template_apply_fn = f"{safe_name_lower}_apply_chat_template"

        # Add to generated file list
        extra_sources.append(str(template_info["template_file"]))

    except Exception as e:
        log(f"  Warning: Could not generate chat template: {e}", C_DIM)
        template_c_name = None
        template_apply_fn = None

    # ... rest of existing code ...
```

---

## Step 4: Modify ck_model_embed_tokens() in Wrapper

**File:** `scripts/v6/ck_run_v6.py` (in step_codegen's wrapper template)

Modify the generated wrapper to apply template before tokenization.

```python
wrapper = f"""\
// AUTO-GENERATED v6 wrapper: {model_name}
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "{kernel_header}"
#include "ckernel_model_load_v4.h"
#include "{kernel_source}"
"""

# Add chat template include if generated
if template_apply_fn:
    wrapper += f'#include "ck-chat-template.c"\n'

wrapper += f"""
static {safe_name}Model g_model;
static int g_initialized = 0;
static int g_active_tokens = 0;
static int g_kv_cache_enabled = 0;
static int g_kv_cache_capacity = {safe_name}_MAX_SEQ_LEN;
static int g_kv_cache_tokens = 0;

static int32_t *g_tokens = NULL;
static int g_tokens_cap = 0;

static float *g_logits = NULL;
static size_t g_logits_cap = 0;

// NEW: Store formatted prompt for chat template
static char *g_formatted_prompt = NULL;

static int ensure_tokens_capacity(int n) {{
    if (n <= g_tokens_cap) return 0;
    int32_t *buf = (int32_t *)realloc(g_tokens, (size_t)n * sizeof(int32_t));
    if (!buf) return -1;
    g_tokens = buf;
    g_tokens_cap = n;
    return 0;
}}

static int ensure_logits_capacity(int n) {{
    size_t needed = (size_t)n * (size_t){safe_name}_VOCAB_SIZE;
    if (needed <= g_logits_cap) return 0;
    float *buf = (float *)realloc(g_logits, needed * sizeof(float));
    if (!buf) return -1;
    g_logits = buf;
    g_logits_cap = needed;
    return 0;
}}

// NEW: API to set system prompt (optional override)
static const char *g_system_prompt = NULL;

int ck_model_set_system_prompt(const char *prompt) {{
    if (g_formatted_prompt) {{
        free(g_formatted_prompt);
        g_formatted_prompt = NULL;
    }}
    g_system_prompt = prompt;
    return 0;
}}

"""

# Modify ck_model_embed_tokens to apply template
if template_apply_fn:
    wrapper += f"""
int ck_model_embed_tokens(const int32_t *tokens, int num_tokens) {{
    if (!g_initialized || !tokens) return -1;

    // Apply chat template if this is the first call (prompt mode)
    if (!g_kv_cache_enabled && g_formatted_prompt) {{
        free(g_formatted_prompt);
        g_formatted_prompt = NULL;
    }}

    int cap = {safe_name}_MAX_SEQ_LEN;
    if (g_kv_cache_enabled && g_kv_cache_capacity > 0 && g_kv_cache_capacity < cap) {{
        cap = g_kv_cache_capacity;
    }}
    if (num_tokens > cap) num_tokens = cap;
    if (num_tokens < 1) num_tokens = 1;
    if (ensure_tokens_capacity(num_tokens) != 0) return -2;
    memcpy(g_tokens, tokens, (size_t)num_tokens * sizeof(int32_t));
    g_active_tokens = num_tokens;
    if (g_kv_cache_enabled) {{
        g_kv_cache_tokens = 0;
    }}
    return 0;
}}

/**
 * NEW: Embed tokens with automatic chat template application.
 *
 * This is the main entry point for chat mode.
 * It applies the model's chat template before tokenizing.
 */
int ck_model_embed_tokens_chat(const char *user_prompt) {{
    if (!g_initialized || !user_prompt) return -1;

    // Free any previous formatted prompt
    if (g_formatted_prompt) {{
        free(g_formatted_prompt);
        g_formatted_prompt = NULL;
    }}

    // Apply chat template
    g_formatted_prompt = {template_apply_fn}(user_prompt, g_system_prompt);
    if (!g_formatted_prompt) return -2;

    // Tokenize the formatted prompt
    // Note: This requires access to the tokenizer from the model
    // We'll need to add this function or call existing tokenization

    return 0;
}}

"""
else:
    wrapper += """
int ck_model_embed_tokens(const int32_t *tokens, int num_tokens) {
    // ... existing implementation ...
}
"""
```

---

## Step 5: Update CLI to Use New API

**File:** `src/v6/ck_cli_v6.c`

Simplify the CLI - no template handling needed, just pass prompt.

```c
typedef int (*embed_t)(const int32_t *tokens, int num_tokens);
// NEW:
typedef int (*embed_chat_t)(const char *user_prompt);
typedef void (*set_system_t)(const char *system_prompt);

typedef struct {
    // ... existing fields ...
    embed_chat_t embed_chat;  // NEW: Chat-mode embed
    set_system_t set_system;  // NEW: Set system prompt
} ModelAPI;

// In load_model_api():
resolve_symbol(api->handle, "ck_model_embed_tokens_chat", (void **)&api->embed_chat, false);
resolve_symbol(api->handle, "ck_model_set_system_prompt", (void **)&api->set_system, false);

// In run_prompt():
// OLD: Tokenize then embed
int32_t *ids = (int32_t *)malloc((size_t)ctx * sizeof(int32_t));
ck_true_bpe_encode(tokenizer, input, -1, ids, ctx);
api->embed(ids, n);

// NEW: Use chat template API (if available)
if (api->embed_chat && api->set_system) {
    // Chat template embedded in model - just pass raw prompt
    if (api->set_system) api->set_system(NULL);  // Use default system prompt
    // But we still need to tokenize...
    // Actually, for true parity, we should let model handle tokenization
}
```

---

## Step 6: The Key Insight - Tokenization Location

There's a complication: the CLI uses `ck_true_bpe_encode()` to tokenize. If we embed the template in the generated C code, we need the C code to also handle tokenization.

**Option A: CLI tokenizes, model applies template**
- CLI: `"Hello"` → tokenize → `[15496, 1234, ...]`
- Model: Detects it's a list of tokens, doesn't apply template
- Problem: Template needs to be applied BEFORE tokenization

**Option B: Model handles tokenization**
- CLI passes raw string to model
- Model has tokenizer access, applies template, then tokenizes
- This requires adding tokenizer API to model

**Option C: Hybrid (Recommended)**
- CLI always passes raw prompt string
- Generated code has `ck_model_embed_prompt()` that:
  1. Applies embedded chat template
  2. Calls `ck_true_bpe_encode()` on result
  3. Embeds tokens

---

## Step 6 Revised: Full Tokenization in Model

This requires extending the model API to include tokenization:

**Generated C code includes:**
```c
// In ck-chat-template.c (expanded)
#include "../tokenizer/true_bpe.h"  // Include tokenizer

static CKTrueBPE *g_tokenizer = NULL;

int ck_model_embed_tokens(const char *user_prompt) {
    // Apply template
    char *formatted = apply_chat_template(user_prompt, g_system_prompt);

    // Tokenize
    if (!g_tokenizer) {
        g_tokenizer = ck_true_bpe_create();
        // Initialize from model's vocab data
    }

    int32_t *ids = (int32_t *)malloc(32768 * sizeof(int32_t));
    int n = ck_true_bpe_encode(g_tokenizer, formatted, -1, ids, 32768);

    free(formatted);

    // ... rest of embed logic ...
    return n;  // Return token count
}
```

But this requires the generated code to have access to the BPE tokenizer code and vocab data. This is complex but provides the cleanest API.

---

## Simpler Alternative: CLI Still Tokenizes

Keep tokenization in CLI, but have model provide template info:

```c
// In ck-chat-template.c (simpler)
#include <stdio.h>
#include <string.h>

// Template is applied in C as a simple string replacement
static const char *CHAT_TEMPLATE = "...";

static void format_chat_prompt(char *out, size_t out_size,
                               const char *user_prompt,
                               const char *system_prompt) {
    // Simple find-replace for {{system}} and {{prompt}}
    snprintf(out, out_size, CHAT_TEMPLATE, system_prompt, user_prompt);
}

// CLI uses this:
char formatted[16384];
format_chat_prompt(formatted, sizeof(formatted), input, system_prompt);

// Then tokenize formatted string
int32_t *ids = (int32_t *)malloc((size_t)ctx * sizeof(int32_t));
ck_true_bpe_encode(tokenizer, formatted, -1, ids, ctx);
```

This is much simpler! The template is embedded as a C string, CLI formats the prompt, then tokenizes.

---

## Step 7: Generate EOS Token ID

The template includes EOS tokens. We need to detect these:

**In `chat_templates.py`:**
```python
def find_eos_token_id(template: str, eos_token: str, tokenizer_json: str) -> int:
    """Find the token ID for the EOS token in the template."""
    if not eos_token or not tokenizer_json:
        return -1

    try:
        import json
        with open(tokenizer_json, 'r') as f:
            data = json.load(f)

        # Get vocab from tokenizer.json
        if 'model' in data and 'vocab' in data:
            vocab = data['model']['vocab']
            if eos_token in vocab:
                return vocab[eos_token]
    except Exception:
        pass

    return -1  # Not found
```

**Generate in C:**
```c
// Add to generated wrapper
static const int32_t EOS_TOKEN_ID = 151643;  // Example for Qwen
```

---

## Step 8: Files Summary

| File | Type | Purpose |
|------|------|---------|
| `scripts/v6/chat_templates.py` | New | Template definitions and model detection |
| `scripts/v6/codegen_v6.py` | Modify | Add `generate_chat_template_section()` |
| `scripts/v6/ck_run_v6.py` | Modify | Call template generation, embed in wrapper |
| `src/v6/ck_cli_v6.c` | Modify | Use new API (optional system prompt) |
| `src/v6/ck-chat-template.c` | Auto-generated | Template string and formatting function |

---

## Implementation Order

1. **Create `chat_templates.py`** - Define all templates and detection logic
2. **Test template detection** - Verify it correctly identifies Qwen, Llama, etc.
3. **Modify `codegen_v6.py`** - Add template embedding function
4. **Modify `ck_run_v6.py`** - Integrate template generation into pipeline
5. **Update generated wrapper** - Add template string and formatting code
6. **Update CLI** - Optionally use new API (mostly unchanged)
7. **Test parity** - Verify C output matches Python with same templates

---

## Quick Test: Template Detection

```python
# Test chat_templates.py
from chat_templates import get_template_for_model

templates = [
    ("Qwen2-0.5B-Instruct", "qwen"),
    ("meta-llama/Llama-3.1-8B-Instruct", "llama3"),
    ("microsoft/Phi-3-mini-4k-instruct", "phi3"),
    ("HuggingFaceTB/SmolLM-135M-Instruct", "smollm"),
]

for model, expected in templates:
    result = get_template_for_model(model)
    template_type = [k for k, v in CHAT_TEMPLATES.items() if v == result][0]
    status = "✓" if template_type == expected else "✗"
    print(f"{status} {model}: {template_type}")
```

---

## Comparison: Before vs After

### Before (Python ck_chat.py)
```python
# CLI does everything
prompt = model.format_chat_prompt(user_input)  # Applies template
token_ids = model.encode(prompt)               # Tokenizes
logits = model.forward(token_ids)              # Runs model
```

### After (Compile-time template)
```c
// Generated C code has:
static const char *QWEN2_CHAT_TEMPLATE =
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n";

// CLI just tokenizes and runs
char formatted[16384];
snprintf(formatted, sizeof(formatted), TEMPLATE, system, prompt);
int32_t *ids = ck_true_bpe_encode(tokenizer, formatted, ...);
api->embed(ids, n);
```

---

## References

- [Qwen Chat Templates](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct#chat-template)
- [Llama 3 Chat Templates](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct#chat-templates)
- [Chat Templates in HF](https://huggingface.co/docs/transformers/main/en/chat_templating)
