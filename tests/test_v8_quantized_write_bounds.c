#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ckernel_quant.h"

void quantize_row_q8_k_sse(const float *x, void *vy, int k);

enum { GUARD_BYTES = 64, BLOCKS = 2 };

typedef struct {
    uint8_t before[GUARD_BYTES];
    block_q8_K output[BLOCKS];
    uint8_t after[GUARD_BYTES];
} guarded_output;

static int guard_is_intact(const uint8_t *guard)
{
    for (int i = 0; i < GUARD_BYTES; ++i) {
        if (guard[i] != UINT8_C(0xA5)) return 0;
    }
    return 1;
}

int main(void)
{
    float input[BLOCKS * QK_K];
    guarded_output guarded;

    for (int i = 0; i < BLOCKS * QK_K; ++i) {
        input[i] = (float)((i % 31) - 15) / 7.0f;
    }
    memset(&guarded, 0xA5, sizeof(guarded));

    quantize_row_q8_k_sse(input, guarded.output, BLOCKS * QK_K);

    if (!guard_is_intact(guarded.before) || !guard_is_intact(guarded.after)) {
        fputs("Q8_K quantizer wrote outside its declared output extent\n", stderr);
        return 1;
    }
    return 0;
}
