/* Bounded mixed-storage projection. FP32 masters/gradients; BF16 RNE saved
 * operands; ordered FP32 products and sums (no contraction). No allocation.
 * The declared VJP uses an identity derivative for operand rounding. */
#include "bf16_utils.h"
#include <math.h>
#include <string.h>

static int product(size_t a, size_t b, size_t *n) {
    if (!a || !b || a > SIZE_MAX/b) return 0;
    *n=a*b; return 1;
}
static int regions(const void **p, const size_t *n, const size_t *width, size_t count, size_t reads) {
    for (size_t i=0;i<count;i++) {
        if (!p[i] || !n[i] || (uintptr_t)p[i]%width[i] || n[i]>SIZE_MAX/width[i] ||
            (uintptr_t)p[i]>UINTPTR_MAX-n[i]*width[i]) return 0;
        if (i>=reads) for (size_t j=0;j<i;j++)
            if ((uintptr_t)p[i]<(uintptr_t)p[j]+n[j]*width[j] &&
                (uintptr_t)p[j]<(uintptr_t)p[i]+n[i]*width[i]) return 0;
    }
    return 1;
}
static float rounded(float x) { return bf16_to_float(float_to_bf16(x)); }
static float add_product(float sum, float a, float b) {
    volatile float p=a*b; /* prohibit FMA, including under normal -O3 */
    volatile float next=sum+p;
    return next;
}
int ck_projection_bf16_train_forward(
    const float *input, size_t input_elements, const float *weight, size_t weight_elements,
    float *output, size_t output_elements, uint16_t *saved_input, size_t saved_input_elements,
    uint16_t *saved_weight, size_t saved_weight_elements, float *scratch, size_t scratch_elements,
    size_t rows, size_t inputs, size_t outputs) {
    size_t x,w,y;
    if (!product(rows,inputs,&x) || !product(outputs,inputs,&w) || !product(rows,outputs,&y) ||
        input_elements<x || weight_elements<w || output_elements<y ||
        saved_input_elements<x || saved_weight_elements<w || scratch_elements<y) return -2;
    const void *p[]={input,weight,output,saved_input,saved_weight,scratch};
    const size_t n[]={input_elements,weight_elements,output_elements,saved_input_elements,saved_weight_elements,scratch_elements};
    const size_t widths[]={4,4,4,2,2,4};
    if (!regions(p,n,widths,6,2)) return -1;
    for(size_t i=0;i<x;i++) if(!isfinite(input[i]) || !isfinite(rounded(input[i]))) return -1;
    for(size_t i=0;i<w;i++) if(!isfinite(weight[i]) || !isfinite(rounded(weight[i]))) return -1;
    for(size_t r=0;r<rows;r++) for(size_t o=0;o<outputs;o++) {
        float sum=0;
        for(size_t k=0;k<inputs;k++) sum=add_product(sum,rounded(input[r*inputs+k]),rounded(weight[o*inputs+k]));
        if(!isfinite(sum)) return -1;
        scratch[r*outputs+o]=sum;
    }
    for(size_t i=0;i<x;i++) saved_input[i]=float_to_bf16(input[i]);
    for(size_t i=0;i<w;i++) saved_weight[i]=float_to_bf16(weight[i]);
    memcpy(output,scratch,y*sizeof(float)); return 0;
}
int ck_projection_bf16_train_backward(
    const float *d_output, size_t d_output_elements,
    const uint16_t *saved_input, size_t saved_input_elements,
    const uint16_t *saved_weight, size_t saved_weight_elements,
    float *d_input, size_t d_input_elements, float *d_weight, size_t d_weight_elements,
    float *scratch, size_t scratch_elements, size_t rows, size_t inputs, size_t outputs) {
    size_t x,w,y;
    if(!product(rows,inputs,&x) || !product(outputs,inputs,&w) || !product(rows,outputs,&y) ||
        x>SIZE_MAX-w || d_output_elements<y || saved_input_elements<x || saved_weight_elements<w ||
        d_input_elements<x || d_weight_elements<w || scratch_elements<x+w) return -2;
    const void *p[]={d_output,saved_input,saved_weight,d_input,d_weight,scratch};
    const size_t n[]={d_output_elements,saved_input_elements,saved_weight_elements,d_input_elements,d_weight_elements,scratch_elements};
    const size_t widths[]={4,2,2,4,4,4};
    if(!regions(p,n,widths,6,3)) return -1;
    for(size_t i=0;i<y;i++) if(!isfinite(d_output[i])) return -1;
    for(size_t i=0;i<x;i++) if(!isfinite(bf16_to_float(saved_input[i]))) return -1;
    for(size_t i=0;i<w;i++) if(!isfinite(bf16_to_float(saved_weight[i]))) return -1;
    for(size_t r=0;r<rows;r++) for(size_t k=0;k<inputs;k++) {
        float sum=0;
        for(size_t o=0;o<outputs;o++) sum=add_product(sum,d_output[r*outputs+o],bf16_to_float(saved_weight[o*inputs+k]));
        if(!isfinite(sum)) return -1;
        scratch[r*inputs+k]=sum;
    }
    for(size_t o=0;o<outputs;o++) for(size_t k=0;k<inputs;k++) {
        float sum=0;
        for(size_t r=0;r<rows;r++) sum=add_product(sum,d_output[r*outputs+o],bf16_to_float(saved_input[r*inputs+k]));
        if(!isfinite(sum)) return -1;
        scratch[x+o*inputs+k]=sum;
    }
    memcpy(d_input,scratch,x*sizeof(float)); memcpy(d_weight,scratch+x,w*sizeof(float)); return 0;
}
