/* BF16-storage adapters for the checked FP32 VJPs. No model scheduling or
 * allocation: conversion, staging and underlying VJP workspace are caller-owned. */
#include "ckernel_head_backward.h"
#include "bf16_utils.h"
#include <math.h>

typedef struct { const void *ptr; size_t elements, unit; } Region;
static int product(size_t a,size_t b,size_t *v) {
    if (!a || !b || a>SIZE_MAX/b) return 0;
    *v=a*b; return 1;
}
static int sum_size(size_t a,size_t b,size_t *v) {
    if (a>SIZE_MAX-b) return 0;
    *v=a+b; return 1;
}
static int extent(size_t rows,size_t cols,size_t stride,size_t size) {
    return rows && cols && stride>=cols && rows-1<=(SIZE_MAX-cols)/stride &&
        (rows-1)*stride+cols<=size;
}
static int regions(const Region *r,size_t count,size_t reads) {
    for (size_t i=0;i<count;++i) {
        if (!r[i].ptr || !r[i].elements || (uintptr_t)r[i].ptr%r[i].unit ||
            r[i].elements>SIZE_MAX/r[i].unit ||
            (uintptr_t)r[i].ptr>UINTPTR_MAX-r[i].elements*r[i].unit) return 0;
    }
    for (size_t i=reads;i<count;++i) for (size_t j=0;j<i;++j)
        if ((uintptr_t)r[i].ptr<(uintptr_t)r[j].ptr+r[j].elements*r[j].unit &&
            (uintptr_t)r[j].ptr<(uintptr_t)r[i].ptr+r[i].elements*r[i].unit) return 0;
    return 1;
}
static int expand(const uint16_t *src,float *dst,size_t rows,size_t cols,size_t stride) {
    for (size_t r=0;r<rows;++r) for (size_t c=0;c<cols;++c) {
        float x=bf16_to_float(src[r*stride+c]);
        if (!isfinite(x)) return 0;
        dst[r*cols+c]=x;
    }
    return 1;
}
static int roundable(const float *src,size_t n) {
    for (size_t i=0;i<n;++i)
        if (!isfinite(src[i]) || (float_to_bf16(src[i])&0x7f80u)==0x7f80u) return 0;
    return 1;
}
static void publish(uint16_t *dst,const float *src,size_t rows,size_t cols,size_t stride) {
    for (size_t r=0;r<rows;++r) for (size_t c=0;c<cols;++c)
        dst[r*stride+c]=float_to_bf16(src[r*cols+c]);
}
int ck_token_span_mean_backward_bf16(
    const uint16_t *d_output,size_t d_output_elements,size_t d_output_stride,
    const int32_t *spans,size_t span_elements,size_t span_count,
    uint16_t *d_input,size_t d_input_elements,size_t d_input_stride,
    float *scratch,size_t scratch_elements,
    size_t token_capacity,size_t valid_tokens,size_t channels) {
    size_t sc,tc,vc,needed;
    if (!valid_tokens || valid_tokens>token_capacity || span_count>SIZE_MAX/2 ||
        !product(span_count,channels,&sc) || !product(token_capacity,channels,&tc) ||
        !product(valid_tokens,channels,&vc) || !sum_size(sc,tc,&needed) ||
        !sum_size(needed,vc,&needed) || scratch_elements<needed || span_elements<2*span_count ||
        !extent(span_count,channels,d_output_stride,d_output_elements) ||
        !extent(token_capacity,channels,d_input_stride,d_input_elements)) return -2;
    Region r[]={{d_output,d_output_elements,2},{spans,span_elements,4},
                {d_input,d_input_elements,2},{scratch,scratch_elements,4}};
    if (!regions(r,4,2)) return -1;
    if (!expand(d_output,scratch,span_count,channels,d_output_stride)) return -1;
    float *result=scratch+sc,*work=result+tc;
    int rc=ck_token_span_mean_backward_f32(scratch,sc,channels,spans,span_elements,span_count,
        result,tc,channels,work,vc,token_capacity,valid_tokens,channels);
    if (rc) return rc;
    if (!roundable(result,vc)) return -1;
    publish(d_input,result,valid_tokens,channels,d_input_stride); return 0;
}
int ck_gelu_erf_backward_bf16(
    const uint16_t *input,size_t input_elements,size_t input_stride,
    const uint16_t *d_output,size_t d_output_elements,size_t d_output_stride,
    uint16_t *d_input,size_t d_input_elements,size_t d_input_stride,
    float *scratch,size_t scratch_elements,size_t rows,size_t channels) {
    size_t n,needed;
    if (!product(rows,channels,&n) || !product(n,4,&needed) || scratch_elements<needed ||
        !extent(rows,channels,input_stride,input_elements) ||
        !extent(rows,channels,d_output_stride,d_output_elements) ||
        !extent(rows,channels,d_input_stride,d_input_elements)) return -2;
    Region r[]={{input,input_elements,2},{d_output,d_output_elements,2},
                {d_input,d_input_elements,2},{scratch,scratch_elements,4}};
    if (!regions(r,4,2)) return -1;
    if (!expand(input,scratch,rows,channels,input_stride) ||
        !expand(d_output,scratch+n,rows,channels,d_output_stride)) return -1;
    int rc=ck_gelu_erf_backward_f32(scratch,n,channels,scratch+n,n,channels,
        scratch+2*n,n,channels,scratch+3*n,n,rows,channels);
    if (rc) return rc;
    if (!roundable(scratch+2*n,n)) return -1;
    publish(d_input,scratch+2*n,rows,channels,d_input_stride); return 0;
}
int ck_grouped_weighted_rows_backward_bf16(
    const uint16_t *options,size_t option_elements,size_t option_stride,
    const uint16_t *fields,size_t field_elements,size_t field_stride,
    const int32_t *group_counts,size_t count_elements,
    const uint16_t *d_output,size_t d_output_elements,size_t d_output_stride,
    uint16_t *d_options,size_t d_options_elements,size_t d_options_stride,
    uint16_t *d_fields,size_t d_fields_elements,size_t d_fields_stride,
    float *scratch,size_t scratch_elements,
    size_t groups,size_t option_rows,size_t channels) {
    size_t oc,gc,twice_o,three_gc,inner,needed,three_oc,six_gc;
    if (!product(option_rows,channels,&oc) || !product(groups,channels,&gc) ||
        !product(option_rows,2,&twice_o) || !product(gc,3,&three_gc) ||
        !sum_size(twice_o,three_gc,&inner) || !sum_size(inner,oc,&inner) ||
        !product(oc,3,&three_oc) || !product(gc,6,&six_gc) ||
        !sum_size(three_oc,six_gc,&needed) || !sum_size(needed,twice_o,&needed) ||
        scratch_elements<needed || count_elements<groups ||
        !extent(option_rows,channels,option_stride,option_elements) ||
        !extent(groups,channels,field_stride,field_elements) ||
        !extent(groups,channels,d_output_stride,d_output_elements) ||
        !extent(option_rows,channels,d_options_stride,d_options_elements) ||
        !extent(groups,channels,d_fields_stride,d_fields_elements)) return -2;
    Region r[]={{options,option_elements,2},{fields,field_elements,2},
                {group_counts,count_elements,4},{d_output,d_output_elements,2},
                {d_options,d_options_elements,2},{d_fields,d_fields_elements,2},
                {scratch,scratch_elements,4}};
    if (!regions(r,7,4)) return -1;
    float *x=scratch,*f=x+oc,*dy=f+gc,*dx=dy+gc,*df=dx+oc,*work=df+gc;
    if (!expand(options,x,option_rows,channels,option_stride) ||
        !expand(fields,f,groups,channels,field_stride) ||
        !expand(d_output,dy,groups,channels,d_output_stride)) return -1;
    int rc=ck_grouped_weighted_rows_backward_f32(x,oc,channels,f,gc,channels,
        group_counts,count_elements,dy,gc,channels,dx,oc,channels,df,gc,channels,
        work,inner,groups,option_rows,channels);
    if (rc) return rc;
    if (!roundable(dx,oc) || !roundable(df,gc)) return -1;
    publish(d_options,dx,option_rows,channels,d_options_stride);
    publish(d_fields,df,groups,channels,d_fields_stride); return 0;
}
