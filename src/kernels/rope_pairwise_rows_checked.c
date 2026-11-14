#include "checked_float_buffer.h"
#include "bf16_utils.h"

static float rounded(float value, int bf16) {
    return bf16 ? bf16_to_float(float_to_bf16(value)) : value;
}

/* Explicit per-position trigonometric tables separate frequency generation
 * from rotation. Both Q and K can use this same checked operation. */
static int rotate_rows(const float *input, size_t input_elements,
    const float *cosine, const float *sine, size_t table_elements,
    float *output, size_t output_elements, size_t tokens, size_t heads,
    size_t head_dim, int bf16)
{
    size_t ni, nt;
    if (!head_dim || (head_dim & 1) || !heads || heads > SIZE_MAX / head_dim ||
        !ck_checked_span(tokens, heads * head_dim, heads * head_dim, &ni) ||
        !ck_checked_span(tokens, head_dim / 2, head_dim / 2, &nt) ||
        input_elements < ni || output_elements < ni || table_elements < nt) return -2;
    if (!ck_checked_region(input, ni) || !ck_checked_region(output, ni) ||
        !ck_checked_region(cosine, nt) || !ck_checked_region(sine, nt) ||
        ck_checked_overlap(input, ni, output, ni) ||
        ck_checked_overlap(cosine, nt, output, ni) ||
        ck_checked_overlap(sine, nt, output, ni)) return -1;
    if (!ck_checked_finite(input, ni) || !ck_checked_finite(cosine, nt) ||
        !ck_checked_finite(sine, nt)) return -3;
    for (int pass = 0; pass < 2; ++pass)
        for (size_t t = 0; t < tokens; ++t)
            for (size_t h = 0; h < heads; ++h)
                for (size_t d = 0; d < head_dim; d += 2) {
                    size_t at = (t * heads + h) * head_dim + d;
                    size_t phase = t * (head_dim / 2) + d / 2;
                    float a = input[at], b = input[at + 1];
                    float c = cosine[phase], s = sine[phase];
                    if (bf16 && (rounded(a, 1) != a || rounded(b, 1) != b ||
                        rounded(c, 1) != c || rounded(s, 1) != s)) return -3;
                    float left = rounded(rounded(a * c, bf16) - rounded(b * s, bf16), bf16);
                    float right = rounded(rounded(b * c, bf16) + rounded(a * s, bf16), bf16);
                    if (!isfinite(left) || !isfinite(right)) return -3;
                    if (pass) { output[at] = left; output[at + 1] = right; }
                }
    return 0;
}

int rope_pairwise_rows_f32_checked(const float *input, size_t input_elements,
    const float *cosine, const float *sine, size_t table_elements,
    float *output, size_t output_elements, size_t tokens, size_t heads, size_t head_dim)
{
    return rotate_rows(input, input_elements, cosine, sine, table_elements,
        output, output_elements, tokens, heads, head_dim, 0);
}

int rope_pairwise_rows_bf16_storage_checked(const float *input, size_t input_elements,
    const float *cosine, const float *sine, size_t table_elements,
    float *output, size_t output_elements, size_t tokens, size_t heads, size_t head_dim)
{
    return rotate_rows(input, input_elements, cosine, sine, table_elements,
        output, output_elements, tokens, heads, head_dim, 1);
}
