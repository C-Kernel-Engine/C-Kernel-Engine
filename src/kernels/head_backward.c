#include "ckernel_head_backward.h"
#include "ckernel_grouped_weighted_rows.h"
#include <float.h>
#include <math.h>
#include <string.h>

static int mul(size_t a, size_t b, size_t *out) {
    if (!a || !b || a > SIZE_MAX / b) return 0;
    *out = a * b; return 1;
}
static int span(size_t rows, size_t cols, size_t stride, size_t elements) {
    return rows && cols && stride >= cols &&
        rows - 1 <= (SIZE_MAX - cols) / stride &&
        (rows - 1) * stride + cols <= elements;
}
static int region(const void *p, size_t n) {
    return p && n && (uintptr_t)p % 4 == 0 && n <= SIZE_MAX / 4 &&
        (uintptr_t)p <= UINTPTR_MAX - 4 * n;
}
static int overlaps(const void *a, size_t n, const void *b, size_t m) {
    return (uintptr_t)a < (uintptr_t)b + 4 * m && (uintptr_t)b < (uintptr_t)a + 4 * n;
}
static int check_regions(const void **ptrs, const size_t *sizes, size_t count, size_t read_count) {
    for (size_t i=0; i<count; ++i) if (!region(ptrs[i], sizes[i])) return 0;
    for (size_t i=read_count; i<count; ++i)
        for (size_t j=0; j<i; ++j)
            if (overlaps(ptrs[i],sizes[i],ptrs[j],sizes[j])) return 0;
    return 1;
}
static int finite_rows(const float *p, size_t rows, size_t cols, size_t stride) {
    for (size_t r=0; r<rows; ++r) for (size_t c=0; c<cols; ++c)
        if (!isfinite(p[r*stride+c])) return 0;
    return 1;
}
static int store(float *p, double x) {
    if (!isfinite(x) || fabs(x)>FLT_MAX) return 0;
    *p=(float)x; return 1;
}
static void publish(float *dest, size_t stride, const float *src, size_t rows, size_t cols) {
    for (size_t r=0; r<rows; ++r) memcpy(dest+r*stride,src+r*cols,cols*sizeof(float));
}
int ck_token_span_mean_backward_f32(
    const float *d_output, size_t d_output_elements, size_t d_output_stride,
    const int32_t *spans, size_t span_elements, size_t span_count,
    float *d_input, size_t d_input_elements, size_t d_input_stride,
    float *scratch, size_t scratch_elements,
    size_t token_capacity, size_t valid_tokens, size_t channels) {
    size_t staged;
    if (!valid_tokens || valid_tokens>token_capacity || !span_count || span_count>SIZE_MAX/2 ||
        !mul(valid_tokens,channels,&staged) || scratch_elements<staged || span_elements<2*span_count ||
        !span(span_count,channels,d_output_stride,d_output_elements) ||
        !span(token_capacity,channels,d_input_stride,d_input_elements)) return -2;
    const void *ptrs[]={d_output,spans,d_input,scratch};
    size_t sizes[]={d_output_elements,span_elements,d_input_elements,scratch_elements};
    if (!check_regions(ptrs,sizes,4,2) || !finite_rows(d_output,span_count,channels,d_output_stride)) return -1;
    for (size_t s=0; s<span_count; ++s)
        if (spans[2*s]<0 || spans[2*s+1]<=spans[2*s] || (size_t)spans[2*s+1]>valid_tokens) return -2;
    /* Ascending span order; one rounded sum per token/channel. Unreferenced
       valid tokens receive zero. Inactive capacity and row padding stay intact. */
    for (size_t t=0; t<valid_tokens; ++t) for (size_t c=0; c<channels; ++c) {
        double sum=0;
        for (size_t s=0; s<span_count; ++s)
            if ((size_t)spans[2*s]<=t && t<(size_t)spans[2*s+1])
                sum+=(double)d_output[s*d_output_stride+c]/(double)(spans[2*s+1]-spans[2*s]);
        if (!store(scratch+t*channels+c,sum)) return -1;
    }
    publish(d_input,d_input_stride,scratch,valid_tokens,channels); return 0;
}
int ck_gelu_erf_backward_f32(
    const float *input, size_t input_elements, size_t input_stride,
    const float *d_output, size_t d_output_elements, size_t d_output_stride,
    float *d_input, size_t d_input_elements, size_t d_input_stride,
    float *scratch, size_t scratch_elements, size_t rows, size_t channels) {
    size_t staged;
    if (!mul(rows,channels,&staged) || scratch_elements<staged ||
        !span(rows,channels,input_stride,input_elements) ||
        !span(rows,channels,d_output_stride,d_output_elements) ||
        !span(rows,channels,d_input_stride,d_input_elements)) return -2;
    const void *ptrs[]={input,d_output,d_input,scratch};
    size_t sizes[]={input_elements,d_output_elements,d_input_elements,scratch_elements};
    if (!check_regions(ptrs,sizes,4,2) || !finite_rows(input,rows,channels,input_stride) ||
        !finite_rows(d_output,rows,channels,d_output_stride)) return -1;
    const double inv_sqrt_two=0.707106781186547524400844362104849;
    const double inv_sqrt_two_pi=0.398942280401432677939946059934382;
    for (size_t r=0; r<rows; ++r) for (size_t c=0; c<channels; ++c) {
        double x=input[r*input_stride+c];
        double derivative=0.5*(1.0+erf(x*inv_sqrt_two)) + x*exp(-0.5*x*x)*inv_sqrt_two_pi;
        if (!store(scratch+r*channels+c,(double)d_output[r*d_output_stride+c]*derivative)) return -1;
    }
    publish(d_input,d_input_stride,scratch,rows,channels); return 0;
}
int ck_grouped_weighted_rows_backward_f32(
    const float *options, size_t option_elements, size_t option_stride,
    const float *fields, size_t field_elements, size_t field_stride,
    const int32_t *group_counts, size_t count_elements,
    const float *d_output, size_t d_output_elements, size_t d_output_stride,
    float *d_options, size_t d_options_elements, size_t d_options_stride,
    float *d_fields, size_t d_fields_elements, size_t d_fields_stride,
    float *scratch, size_t scratch_elements,
    size_t groups, size_t option_rows, size_t channels) {
    size_t gc,oc,required;
    if (!mul(groups,channels,&gc) || !mul(option_rows,channels,&oc) ||
        gc>SIZE_MAX/3 || option_rows>SIZE_MAX/2 ||
        oc>SIZE_MAX-3*gc || 2*option_rows>SIZE_MAX-(oc+3*gc)) return -2;
    required=2*option_rows+3*gc+oc;
    if (scratch_elements<required || count_elements<groups ||
        !span(option_rows,channels,option_stride,option_elements) ||
        !span(groups,channels,field_stride,field_elements) ||
        !span(groups,channels,d_output_stride,d_output_elements) ||
        !span(option_rows,channels,d_options_stride,d_options_elements) ||
        !span(groups,channels,d_fields_stride,d_fields_elements)) return -2;
    const void *ptrs[]={options,fields,group_counts,d_output,d_options,d_fields,scratch};
    size_t sizes[]={option_elements,field_elements,count_elements,d_output_elements,
                    d_options_elements,d_fields_elements,scratch_elements};
    if (!check_regions(ptrs,sizes,7,4) || !finite_rows(d_output,groups,channels,d_output_stride)) return -1;
    /* Recompute probabilities through the same forward provider, preserving
       its FP32 storage stages. Scratch regions do not overlap one another. */
    float *summary=scratch+option_rows+gc;
    float *dx=summary+gc, *df=dx+oc, *dz=df+gc;
    int status=ck_grouped_softmax_weighted_rows_f32(options,option_elements,option_stride,
        fields,field_elements,field_stride,group_counts,count_elements,summary,gc,channels,
        scratch,option_rows+gc,groups,option_rows,channels);
    if (status) return status;
    size_t first=0;
    const double scale=1.0/sqrt((double)channels);
    for (size_t q=0; q<groups; ++q) {
        size_t count=(size_t)group_counts[q];
        double weighted=0;
        for (size_t i=0; i<count; ++i) {
            size_t r=first+i; double sum=0;
            for (size_t c=0; c<channels; ++c) {
                volatile double term=(double)d_output[q*d_output_stride+c]*options[r*option_stride+c];
                sum+=term;
            }
            if (!store(dz+r,sum)) return -1;
            volatile double term=(double)scratch[r]*dz[r]; weighted+=term;
        }
        if (!isfinite(weighted)) return -1;
        for (size_t i=0; i<count; ++i) {
            size_t r=first+i;
            if (!store(dz+r,(double)scratch[r]*((double)dz[r]-weighted)*scale)) return -1;
            for (size_t c=0; c<channels; ++c) {
                volatile float direct=scratch[r]*d_output[q*d_output_stride+c];
                volatile float via_score=dz[r]*fields[q*field_stride+c];
                if (!store(dx+r*channels+c,(double)direct+(double)via_score)) return -1;
            }
        }
        for (size_t c=0; c<channels; ++c) {
            double sum=0;
            for (size_t i=0; i<count; ++i) {
                volatile double term=(double)dz[first+i]*options[(first+i)*option_stride+c];
                sum+=term;
            }
            if (!store(df+q*channels+c,sum)) return -1;
        }
        first+=count;
    }
    publish(d_options,d_options_stride,dx,option_rows,channels);
    publish(d_fields,d_fields_stride,df,groups,channels); return 0;
}
