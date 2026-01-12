# Context Window Limit Plan

## Problem

The current CLI doesn't properly respect the model's context window limit when generating tokens.

### Current Behavior
```c
int max_decode_tokens = (opt->max_tokens > 0) ? opt->max_tokens : 8192;
for (int generated = 0; generated < max_decode_tokens && !g_exit_requested; generated++) {
    // Generate until max_tokens or EOS
}
```

**Issue:** This ignores the prefill tokens. If:
- Context window = 32768
- Prefill = 52 tokens
- Max decode = 8192 tokens

Total tokens = 52 + 8192 = 8244, which is fine. But if max_decode was 32768, total would be 32820 > 32768!

### What Happens When Exceeding Context Window

1. **Prefill phase:** Model processes all N input tokens and stores K/V vectors in cache
2. **Decode phase:** For each new token, model attends to all previous K/V vectors
3. **Overflow:** If total tokens > context_window, either:
   - Oldest K/V vectors get evicted (loss of context)
   - Model generates gibberish
   - Buffer overflow/crash

---

## Solution Plan

### Step 1: Calculate Available Decode Slots

```c
/* Available decode slots = context_window - prefill_tokens - 1 (for current token) */
int available_decode_slots = ctx - n - 1;
if (available_decode_slots <= 0) {
    fprintf(stderr, "[Error] Prompt too long for context window\n");
    return -1;
}
```

### Step 2: Use Available Slots as Limit

```c
/* max_decode_tokens cannot exceed available slots */
int effective_max = (opt->max_tokens > 0)
    ? min(opt->max_tokens, available_decode_slots)
    : available_decode_slots;  /* Stop at EOS or context limit */

fprintf(stderr, "[Context] window=%d, prompt=%d, max_decode=%d\n",
        ctx, n, effective_max);
```

### Step 3: Add Safety Check After Each Decode

```c
/* After each decode, check if we're at context limit */
if (n + generated + 1 >= ctx) {
    fprintf(stderr, "[Warning] Reached context window limit\n");
    break;
}
```

### Step 4: Update Loop Condition

```c
for (int generated = 0; generated < effective_max && !g_exit_requested; generated++) {
    // ... generate token ...

    /* Check for context limit */
    if (n + generated + 1 >= ctx) break;
}
```

---

## Detailed Code Changes

### File: `src/v6/ck_cli_v6.c`

**Function: `run_prompt()`**

```c
static int run_prompt(ModelAPI *api, CKTrueBPE *tokenizer, const CLIOptions *opt, const char *input) {
    // ... existing setup code ...

    int ctx = opt->context_override;
    if (ctx <= 0 && api->get_context) {
        ctx = api->get_context();
    }
    if (ctx <= 0) ctx = 4096;

    // ... template application ...

    /* Calculate available decode slots based on context window */
    int available_decode_slots = ctx - n - 1;  /* -1 for current token */
    if (available_decode_slots <= 0) {
        fprintf(stderr, "[Error] Prompt too long for context window (%d > %d)\n", n, ctx);
        free(ids);
        return -1;
    }

    /* Determine effective max tokens */
    int effective_max;
    if (opt->max_tokens > 0) {
        effective_max = (opt->max_tokens < available_decode_slots)
            ? opt->max_tokens
            : available_decode_slots;
    } else {
        /* No max specified - use available slots, stop at EOS */
        effective_max = available_decode_slots;
    }

    if (!opt->timing) {
        fprintf(stderr, "[Context] window=%d, prompt=%d, max_output=%d\n",
                ctx, n, effective_max);
    }

    // ... existing code ...

    /* Main generation loop */
    for (int generated = 0; generated < effective_max && !g_exit_requested; generated++) {
        if (next_token < 0) break;
        if (is_eos_token(opt, next_token)) break;

        const char *word = ck_true_bpe_token(tokenizer, next_token);
        output_token(out_buf, &out_len, word);

        // ... streaming output ...

        /* Safety check: don't exceed context window */
        if (n + generated + 1 >= ctx) {
            if (!opt->timing) {
                fprintf(stderr, "\n[Warning] Reached context window limit\n");
            }
            break;
        }

        if (generated + 1 >= effective_max) break;

        // ... decode and sample ...
    }

    // ... timing output ...
}
```

---

## Additional Improvements

### 1. Add `--max-output` Flag (Distinct from `--max-tokens`)

```
--max-tokens N     = Total output tokens (capped by context window)
--max-output N     = Same as --max-tokens (alias for clarity)
```

### 2. Show Context Usage in Timing

```
Timing:
  Prefill:   52 tokens in 22159.63 ms (1.49 tok/s)
  Decode:    29 tokens in 14415.15 ms (2.01 tok/s) avg 497.07 ms/tok
  Total:     81 tokens (52 prompt + 29 output) / 32768 context window
```

### 3. Add `--interactive` Mode for Long Conversations

```
--interactive, -i  = Reset KV cache after each response
                     (allows unlimited conversation length)
```

---

## Verification Test

```bash
# With small context override to test limit
./ck-cli-v6 model.so weights.bump \
    --prompt "Hello" \
    --context 100 \
    --max-tokens 200

# Expected output:
# [Context] window=100, prompt=5, max_output=94
# ... generation ...
# [Warning] Reached context window limit
```

---

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| Token limit | Fixed 8192 or user-specified | `min(user_specified, ctx - prompt - 1)` |
| Context awareness | Ignores prefill | Uses `ctx - n - 1` |
| Safety check | None | Checks `n + generated + 1 >= ctx` |
| User feedback | Silent | Shows `[Context] window=X, prompt=Y, max_output=Z` |

This ensures the model never tries to generate more tokens than its context window can hold, preventing garbage output or crashes.
