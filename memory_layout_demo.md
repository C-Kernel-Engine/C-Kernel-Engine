# Memory Layout Demo: Vision-Language Model

## Model: LLaVA-style (Vision Encoder + Text Decoder)

This demonstrates the header/body/footer pattern with real byte offsets.

---

## Section 0: Vision Encoder (ViT-L/14)

```
Config:
  embed_dim:        1024
  num_heads:        16
  num_kv_heads:     16
  head_dim:         64
  intermediate_dim: 4096
  num_layers:       24
  max_seq_len:      577  (1 CLS + 24×24 patches)
  vocab_size:       0    (no token embedding, uses patch projection)
  dtype:            bf16 (2 bytes)
```

### HEADER (Vision)

```
Offset          Size        Buffer                      Shape
─────────────────────────────────────────────────────────────────────
0x00000000      3,145,728   patch_embed_weight          [768, 1024] (3×16×16 → 1024)
0x00300000      1,181,696   patch_embed_output          [577, 1024]
0x00420000      1,181,696   pos_embed_weight            [577, 1024]
0x00540000      1,181,696   pos_embed_output            [577, 1024]
0x00660000      1,181,696   header_output               [577, 1024] (embed + pos)
─────────────────────────────────────────────────────────────────────
HEADER TOTAL:   7,872,512 bytes (7.5 MB)
```

### BODY Layer 0 (Vision) - Execution Order

```
Offset          Size        Buffer                      Shape
─────────────────────────────────────────────────────────────────────
0x00780000      2,048       ln1_gamma                   [1024]
0x00780800      2,048       ln1_beta                    [1024]
0x00781000      1,181,696   ln1_output                  [577, 1024]

0x008A1800      2,097,152   wq                          [1024, 1024]
0x00AA1800      2,048       bq                          [1024]
0x00AA2000      1,181,696   q                           [577, 16, 64]

0x00BC2800      2,097,152   wk                          [1024, 1024]
0x00DC2800      2,048       bk                          [1024]
0x00DC3000      1,181,696   k                           [577, 16, 64]

0x00EE3800      2,097,152   wv                          [1024, 1024]
0x010E3800      2,048       bv                          [1024]
0x010E4000      1,181,696   v                           [577, 16, 64]

0x01204800      21,296,912  attn_scores                 [16, 577, 577]
0x02654800      21,296,912  attn_probs                  [16, 577, 577]
0x03AA4800      1,181,696   attn_out                    [577, 16, 64]

0x03BC5000      2,097,152   wo                          [1024, 1024]
0x03DC5000      2,048       bo                          [1024]
0x03DC5800      1,181,696   proj_out                    [577, 1024]

0x03EE6000      1,181,696   residual1                   [577, 1024]

0x04006800      2,048       ln2_gamma                   [1024]
0x04007000      2,048       ln2_beta                    [1024]
0x04007800      1,181,696   ln2_output                  [577, 1024]

0x04128000      8,388,608   mlp_up_w                    [1024, 4096]
0x04928000      8,192       mlp_up_b                    [4096]
0x0492A000      4,726,784   mlp_up_out                  [577, 4096]

0x04DB2000      4,726,784   mlp_act_out                 [577, 4096] (after GELU)

0x0523A000      8,388,608   mlp_down_w                  [4096, 1024]
0x05A3A000      2,048       mlp_down_b                  [1024]
0x05A3A800      1,181,696   mlp_down_out                [577, 1024]

0x05B5B000      1,181,696   residual2                   [577, 1024]
─────────────────────────────────────────────────────────────────────
LAYER 0 TOTAL:  85,286,912 bytes (81.3 MB)
```

### BODY Summary (Vision) - 24 Layers

```
Layer       Start           End             Size
─────────────────────────────────────────────────────────
Layer 0     0x00780000      0x05C7B800      85.3 MB
Layer 1     0x05C7B800      0x0B577000      85.3 MB
Layer 2     0x0B577000      0x11472800      85.3 MB
...
Layer 23    0x4E0F0000      0x53A0B800      85.3 MB
─────────────────────────────────────────────────────────
BODY TOTAL: 24 × 85.3 MB = 2,047 MB (~2 GB)
```

### FOOTER (Vision)

```
Offset          Size        Buffer                      Shape
─────────────────────────────────────────────────────────────────────
0x53A0B800      2,048       final_ln_gamma              [1024]
0x53A0C000      2,048       final_ln_beta               [1024]
0x53A0C800      1,181,696   final_ln_output             [577, 1024]

0x53B2D000      8,388,608   bridge_w                    [1024, 4096] (to text decoder)
0x5432D000      8,192       bridge_b                    [4096]
0x5432F000      4,726,784   bridge_output               [577, 4096]
─────────────────────────────────────────────────────────────────────
FOOTER TOTAL:   14,309,376 bytes (13.6 MB)
```

### Section 0 (Vision) Complete

```
Component       Start           End             Size
─────────────────────────────────────────────────────────
HEADER          0x00000000      0x00780000      7.5 MB
BODY            0x00780000      0x53A0B800      2,047 MB
FOOTER          0x53A0B800      0x547B7800      13.6 MB
─────────────────────────────────────────────────────────
SECTION 0:      0x00000000      0x547B7800      2,068 MB (~2 GB)
```

---

## Section 1: Text Decoder (LLaMA-7B style)

```
Config:
  embed_dim:        4096
  num_heads:        32
  num_kv_heads:     32
  head_dim:         128
  intermediate_dim: 11008
  num_layers:       32
  max_seq_len:      2048
  vocab_size:       32000
  dtype:            bf16 (2 bytes)
```

### HEADER (Text)

```
Offset          Size        Buffer                      Shape
─────────────────────────────────────────────────────────────────────
0x547B7800      262,144,000 token_embed_weight          [32000, 4096]
0x644B7800      16,777,216  embed_output                [2048, 4096]
0x654B7800      16,777,216  header_output               [2048, 4096]
─────────────────────────────────────────────────────────────────────
HEADER TOTAL:   295,698,432 bytes (282 MB)
```

### BODY Layer 0 (Text) - Execution Order

```
Offset          Size        Buffer                      Shape
─────────────────────────────────────────────────────────────────────
0x664B7800      8,192       ln1_gamma                   [4096] (RMSNorm, no beta)
0x664B9800      16,777,216  ln1_output                  [2048, 4096]

0x674B9800      33,554,432  wq                          [4096, 4096]
0x694B9800      16,777,216  q                           [2048, 32, 128]

0x6A4B9800      33,554,432  wk                          [4096, 4096]
0x6C4B9800      16,777,216  k                           [2048, 32, 128]

0x6D4B9800      33,554,432  wv                          [4096, 4096]
0x6F4B9800      16,777,216  v                           [2048, 32, 128]

0x704B9800      16,777,216  q_rope                      [2048, 32, 128]
0x714B9800      16,777,216  k_rope                      [2048, 32, 128]

0x724B9800      536,870,912 attn_scores                 [32, 2048, 2048]
0x924B9800      536,870,912 attn_probs                  [32, 2048, 2048]
0xB24B9800      16,777,216  attn_out                    [2048, 32, 128]

0xB34B9800      33,554,432  wo                          [4096, 4096]
0xB54B9800      16,777,216  proj_out                    [2048, 4096]

0xB64B9800      16,777,216  residual1                   [2048, 4096]

0xB74B9800      8,192       ln2_gamma                   [4096]
0xB74BB800      16,777,216  ln2_output                  [2048, 4096]

0xB84BB800      90,177,536  mlp_gate_w                  [4096, 11008]
0xC0CBB800      45,088,768  mlp_gate_out                [2048, 11008]

0xC38BB800      90,177,536  mlp_up_w                    [4096, 11008]
0xCC4BB800      45,088,768  mlp_up_out                  [2048, 11008]

0xCF0BB800      45,088,768  mlp_act_out                 [2048, 11008] (SiLU gate × up)

0xD1CBB800      90,177,536  mlp_down_w                  [11008, 4096]
0xDA8BB800      16,777,216  mlp_down_out                [2048, 4096]

0xDB8BB800      16,777,216  residual2                   [2048, 4096]
─────────────────────────────────────────────────────────────────────
LAYER 0 TOTAL:  1,779,040,256 bytes (1.66 GB per layer!)
```

### BODY Summary (Text) - 32 Layers

```
Layer       Start           End             Size
─────────────────────────────────────────────────────────
Layer 0     0x664B7800      0xDC8BB800      1.66 GB
Layer 1     0xDC8BB800      0x152CBF800     1.66 GB
...
Layer 31    0x37E4BF800     0x3F64C3800     1.66 GB
─────────────────────────────────────────────────────────
BODY TOTAL: 32 × 1.66 GB = 53.1 GB
```

### FOOTER (Text)

```
Offset          Size        Buffer                      Shape
─────────────────────────────────────────────────────────────────────
0x3F64C3800     8,192       final_ln_gamma              [4096]
0x3F64C5800     16,777,216  final_ln_output             [2048, 4096]
0x3F74C5800     262,144,000 lm_head_w                   [4096, 32000]
0x3FFFC5800     131,072,000 logits                      [2048, 32000]
─────────────────────────────────────────────────────────────────────
FOOTER TOTAL:   410,001,408 bytes (391 MB)
```

---

## Complete Model Memory Map

```
Section         Component   Start           End             Size
───────────────────────────────────────────────────────────────────────
SECTION 0       HEADER      0x000000000     0x000780000     7.5 MB
(Vision)        BODY        0x000780000     0x053A0B800     2,047 MB
                FOOTER      0x053A0B800     0x0547B7800     13.6 MB
───────────────────────────────────────────────────────────────────────
SECTION 1       HEADER      0x0547B7800     0x0664B7800     282 MB
(Text)          BODY        0x0664B7800     0x3F64C3800     53.1 GB
                FOOTER      0x3F64C3800     0x407EC9800     391 MB
───────────────────────────────────────────────────────────────────────
TOTAL FORWARD:  ~55.8 GB
───────────────────────────────────────────────────────────────────────

If training (add gradients + optimizer):
  GRADIENTS:    0x407EC9800     0x80FD93000     ~55.8 GB
  ADAM M:       0x80FD93000     0x8A7E5C800     ~10 GB (weights only)
  ADAM V:       0x8A7E5C800     0x937D26000     ~10 GB (weights only)
───────────────────────────────────────────────────────────────────────
TOTAL TRAINING: ~131.6 GB
```

---

## Generated C Code Structure

```c
// ============================================================
// AUTO-GENERATED BY C-KERNEL-ENGINE
// Model: LLaVA-7B (Vision + Text)
// Total Memory: 55.8 GB (inference) / 131.6 GB (training)
// ============================================================

#include <stddef.h>

// Single base pointer - THE allocation
static void *BASE;

// ============================================================
// SECTION 0: VISION ENCODER
// ============================================================

static inline void vision_header(void) {
    patch_embed(BASE + 0x00000000,   // weight
                BASE + 0x00300000);  // output

    add(BASE + 0x00300000,           // embed output
        BASE + 0x00420000,           // pos embed
        BASE + 0x00660000);          // combined output
}

static inline void vision_layer_0(void) {
    layernorm(BASE + 0x00660000,     // input (from header)
              BASE + 0x00780000,     // gamma
              BASE + 0x00780800,     // beta
              BASE + 0x00781000);    // output

    linear(BASE + 0x00781000,        // input
           BASE + 0x008A1800,        // wq
           BASE + 0x00AA1800,        // bq
           BASE + 0x00AA2000);       // q output

    linear(BASE + 0x00781000,        // input (same)
           BASE + 0x00BC2800,        // wk
           BASE + 0x00DC2800,        // bk
           BASE + 0x00DC3000);       // k output

    linear(BASE + 0x00781000,        // input (same)
           BASE + 0x00EE3800,        // wv
           BASE + 0x010E3800,        // bv
           BASE + 0x010E4000);       // v output

    attention(BASE + 0x00AA2000,     // q
              BASE + 0x00DC3000,     // k
              BASE + 0x010E4000,     // v
              BASE + 0x01204800,     // scores
              BASE + 0x02654800,     // probs
              BASE + 0x03AA4800);    // output

    linear(BASE + 0x03AA4800,        // attn output
           BASE + 0x03BC5000,        // wo
           BASE + 0x03DC5000,        // bo
           BASE + 0x03DC5800);       // proj output

    residual_add(BASE + 0x00660000,  // original input
                 BASE + 0x03DC5800,  // proj output
                 BASE + 0x03EE6000); // residual1

    layernorm(BASE + 0x03EE6000,     // residual1
              BASE + 0x04006800,     // gamma
              BASE + 0x04007000,     // beta
              BASE + 0x04007800);    // ln2 output

    linear(BASE + 0x04007800,        // ln2 output
           BASE + 0x04128000,        // mlp_up_w
           BASE + 0x04928000,        // mlp_up_b
           BASE + 0x0492A000);       // mlp_up_out

    gelu(BASE + 0x0492A000,          // mlp_up_out
         BASE + 0x04DB2000);         // mlp_act_out

    linear(BASE + 0x04DB2000,        // mlp_act_out
           BASE + 0x0523A000,        // mlp_down_w
           BASE + 0x05A3A000,        // mlp_down_b
           BASE + 0x05A3A800);       // mlp_down_out

    residual_add(BASE + 0x03EE6000,  // residual1
                 BASE + 0x05A3A800,  // mlp_down_out
                 BASE + 0x05B5B000); // residual2 = layer output
}

// ... vision_layer_1 through vision_layer_23 (same pattern, different offsets)

static inline void vision_footer(void) {
    layernorm(BASE + 0x53A0B800,     // last layer output
              BASE + 0x53A0C000,     // gamma
              BASE + 0x53A0C800,     // beta
              BASE + 0x53B2D000);    // final ln output

    linear(BASE + 0x53B2D000,        // final ln output
           BASE + 0x5432D000,        // bridge_w
           BASE + 0x5432F000,        // bridge_b
           BASE + 0x547B7800);       // bridge output → text decoder input
}

// ============================================================
// SECTION 1: TEXT DECODER
// ============================================================

static inline void text_header(void) {
    // Token embedding (text tokens only, vision already embedded)
    embedding_lookup(tokens,
                     BASE + 0x547B7800,   // embed weight
                     BASE + 0x644B7800);  // embed output
}

static inline void text_layer_0(void) {
    rmsnorm(BASE + 0x654B7800,       // input
            BASE + 0x664B7800,       // gamma (no beta for RMSNorm)
            BASE + 0x664B9800);      // output

    // ... same pattern as vision but with different dims and offsets
    // RoPE instead of absolute position
    // SwiGLU instead of GELU
}

// ... text_layer_1 through text_layer_31

static inline void text_footer(void) {
    rmsnorm(BASE + 0x3F64C3800,      // last layer output
            BASE + 0x3F64C5800,      // gamma
            BASE + 0x3F74C5800);     // final output

    linear(BASE + 0x3F74C5800,       // final output
           BASE + 0x3FFFC5800,       // lm_head (may alias embed)
           NULL,                      // no bias
           BASE + 0x407EC9800);      // logits
}

// ============================================================
// MAIN FORWARD PASS - PURE STRAIGHT LINE
// ============================================================

void forward(void *base, int *tokens, int num_tokens) {
    BASE = base;

    // Section 0: Vision
    vision_header();
    vision_layer_0();
    vision_layer_1();
    vision_layer_2();
    vision_layer_3();
    vision_layer_4();
    vision_layer_5();
    vision_layer_6();
    vision_layer_7();
    vision_layer_8();
    vision_layer_9();
    vision_layer_10();
    vision_layer_11();
    vision_layer_12();
    vision_layer_13();
    vision_layer_14();
    vision_layer_15();
    vision_layer_16();
    vision_layer_17();
    vision_layer_18();
    vision_layer_19();
    vision_layer_20();
    vision_layer_21();
    vision_layer_22();
    vision_layer_23();
    vision_footer();

    // Section 1: Text
    text_header();
    text_layer_0();
    text_layer_1();
    text_layer_2();
    text_layer_3();
    text_layer_4();
    text_layer_5();
    text_layer_6();
    text_layer_7();
    text_layer_8();
    text_layer_9();
    text_layer_10();
    text_layer_11();
    text_layer_12();
    text_layer_13();
    text_layer_14();
    text_layer_15();
    text_layer_16();
    text_layer_17();
    text_layer_18();
    text_layer_19();
    text_layer_20();
    text_layer_21();
    text_layer_22();
    text_layer_23();
    text_layer_24();
    text_layer_25();
    text_layer_26();
    text_layer_27();
    text_layer_28();
    text_layer_29();
    text_layer_30();
    text_layer_31();
    text_footer();

    // Done. Logits at BASE + 0x407EC9800
}
```

---

## The Antsand Connection

```
ANTSAND HMVC                    C-KERNEL-ENGINE
────────────────────────────────────────────────────────────
header.php                      section.header (embeddings)
  └── site config                 └── model config
  └── auth state                  └── pos embeddings
  └── navigation                  └── patch/token embed

body.php                        section.body (transformer layers)
  └── main content                └── layer[0..N]
  └── business logic                  └── norm → attn → mlp
  └── data processing                 └── weights + activations

footer.php                      section.footer (output)
  └── closing scripts              └── final norm
  └── analytics                    └── lm_head / classifier
  └── links to next page           └── bridge to next section

DETERMINISTIC                   DETERMINISTIC
  └── same URL = same page         └── same input = same output
  └── no runtime DB queries        └── no runtime allocations
  └── cached/precomputed           └── all offsets hardcoded
```

**Same philosophy, different domain.**

---

## Summary

```
ONE ALLOCATION:     131.6 GB (training mode)
ONE BASE POINTER:   void *base
ALL OFFSETS:        hardcoded at codegen time
ZERO BRANCHES:      straight-line execution
ZERO MALLOC:        everything pre-planned
DETERMINISTIC:      bit-exact reproducibility
```
