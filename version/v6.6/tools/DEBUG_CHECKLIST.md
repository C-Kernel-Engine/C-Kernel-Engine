# V6.6 Debugging Checklist

## Problem
Garbled output despite correct quantization and memory layout.

## Tools Available
- `audit_quantization_chain.py` - Verify dtype/kernel matching [PASSED]
- `validate_generated_c.py` - Verify C code against JSON [PASSED]
- `ir_visualizer.html` - Visual memory/kernel inspection
- `trace_layer_divergence.py` - Layer-by-layer tensor comparison

## Debugging Steps

### Step 1: Verify Embedding Layer
```bash
# Add to model_v6_6.c after embedding:
printf("Embedding first 10: ");
for(int i=0; i<10; i++) printf("%.6f ", ((float*)(model->bump + A_EMBEDDED_INPUT))[i]);
printf("\n");
```
**Expected**: Values should be in reasonable range (-10 to 10), not NaN/Inf

### Step 2: Verify First RMSNorm
```bash
# Add after first rmsnorm_forward call:
printf("RMSNorm1 L0 first 10: ");
for(int i=0; i<10; i++) printf("%.6f ", ((float*)(model->bump + A_LAYER_INPUT))[i]);
printf("\n");
```
**Expected**: Normalized values, typically -3 to 3

### Step 3: Compare Q/K/V Projections
Key question: Is Q projection outputting sensible values?

```bash
# After q_proj in layer 0:
printf("Q proj L0 first 10: ");
for(int i=0; i<10; i++) printf("%.6f ", ((float*)(model->bump + A_Q_SCRATCH))[i]);
printf("\n");
```

### Step 4: Check RoPE Output
```bash
# After rope_forward_qk:
printf("Q after RoPE first 10: ");
for(int i=0; i<10; i++) printf("%.6f ", ((float*)(model->bump + A_Q_SCRATCH))[i]);
printf("\n");
```

### Step 5: Attention Output
```bash
# After attention_forward:
printf("Attn out first 10: ");
for(int i=0; i<10; i++) printf("%.6f ", ((float*)(model->bump + A_ATTN_SCRATCH))[i]);
printf("\n");
```

### Step 6: Check Residual Addition
```bash
# After ck_residual_add:
printf("Residual1 first 10: ");
for(int i=0; i<10; i++) printf("%.6f ", ((float*)(model->bump + A_RESIDUAL))[i]);
printf("\n");
```

### Step 7: Final Logits
Check if top-k tokens make sense:
```bash
# At end of forward:
int max_idx = 0;
float max_val = -1e9;
for(int i=0; i<VOCAB_SIZE; i++) {
    if(model->logits[i] > max_val) { max_val = model->logits[i]; max_idx = i; }
}
printf("Top token: %d with logit %.4f\n", max_idx, max_val);
```

## Quick Test Script

Add this to the generated C to stop at first layer and dump values:

```c
static void debug_layer0(CKModel *model) {
    float *emb = (float*)(model->bump + A_EMBEDDED_INPUT);
    float *layer_in = (float*)(model->bump + A_LAYER_INPUT);
    float *q = (float*)(model->bump + A_Q_SCRATCH);
    float *attn_out = (float*)(model->bump + A_ATTN_SCRATCH);

    printf("=== Layer 0 Debug ===\n");
    printf("Embedding: [%.4f, %.4f, %.4f, ...]\n", emb[0], emb[1], emb[2]);
    printf("After LN1: [%.4f, %.4f, %.4f, ...]\n", layer_in[0], layer_in[1], layer_in[2]);
    printf("Q proj:    [%.4f, %.4f, %.4f, ...]\n", q[0], q[1], q[2]);
    printf("Attn out:  [%.4f, %.4f, %.4f, ...]\n", attn_out[0], attn_out[1], attn_out[2]);

    // Check for NaN/Inf
    int nan_count = 0, inf_count = 0;
    for(int i=0; i<896; i++) {
        if(isnan(emb[i])) nan_count++;
        if(isinf(emb[i])) inf_count++;
    }
    if(nan_count || inf_count) printf("WARNING: %d NaN, %d Inf in embedding!\n", nan_count, inf_count);
}
```

## Common Causes of Garbled Output

| Symptom | Likely Cause |
|---------|--------------|
| All zeros | Wrong offset / not reading weights |
| NaN values | Division by zero in RMSNorm or attention |
| Very large values | Missing scaling in attention |
| Repeating patterns | Wrong stride / data layout |
| Random garbage | Memory aliasing / buffer overlap |

## Comparing with V6.5

If v6.5 works but v6.6 doesn't:

1. **Compare kernel implementations** - Are the same kernels being called?
2. **Compare argument order** - Did argument positions change?
3. **Compare buffer shapes** - Same data layout?

```bash
# Run existing comparison test
python version/v6.6/test/compare_v65_v66.py
```

## Files to Check

- `src/kernels/kv_cache_kernels.c:kv_cache_store()` - KV cache storage
- `src/kernels/attention_kernels.c` - Attention implementations
- `src/kernels/gemv_kernels.c` - GEMV implementations
- `version/v6.6/src/generated/ck-kernel-inference.c` - Template kernels
