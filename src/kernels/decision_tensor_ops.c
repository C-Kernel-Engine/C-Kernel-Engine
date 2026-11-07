/* Checked contiguous tensor operations. No allocation or model scheduling. */
#include "ckernel_decision_tensor_ops.h"
#include <math.h>
#include <float.h>
#include <string.h>

static int region(const void *p, size_t n) {
    return p && n && (uintptr_t)p % 4 == 0 && n <= SIZE_MAX / 4 &&
           (uintptr_t)p <= UINTPTR_MAX - n * 4;
}
static int overlap(const void *a, size_t n, const void *b, size_t m) {
    return (uintptr_t)a < (uintptr_t)b + m * 4 &&
           (uintptr_t)b < (uintptr_t)a + n * 4;
}
static int product(size_t a, size_t b, size_t *n) {
    if (!a || !b || a > SIZE_MAX / b) return 0;
    *n = a * b; return 1;
}
/* Validate all regions before address arithmetic or any scratch mutation. */
static int buffers(const void **inputs, const size_t *sizes, size_t count,
                   float *out, size_t on, float *scratch, size_t sn, size_t needed) {
    if (on < needed || sn < needed || !region(out, on) || !region(scratch, sn)) return 0;
    if (overlap(out, on, scratch, sn)) return 0;
    for (size_t i=0; i<count; ++i) {
        if (!region(inputs[i], sizes[i]) || overlap(out,on,inputs[i],sizes[i]) ||
            overlap(scratch,sn,inputs[i],sizes[i])) return 0;
    }
    return 1;
}
static int all_finite(const float *x, size_t n) {
    for (size_t i=0; i<n; ++i) if (!isfinite(x[i])) return 0;
    return 1;
}
static int publish(float *out, float *scratch, size_t n) {
    if (!all_finite(scratch,n)) return -1;
    memcpy(out,scratch,n*sizeof(float)); return 0;
}
int ck_broadcast_add_rows_f32(const float *left, size_t left_elements,
    const float *right, size_t right_elements, float *output, size_t output_elements,
    float *scratch, size_t scratch_elements, size_t rows, size_t channels) {
    size_t n;
    if (!product(rows,channels,&n) || left_elements<n || right_elements<channels) return -2;
    const void *in[]={left,right}; size_t sizes[]={left_elements,right_elements};
    if (!buffers(in,sizes,2,output,output_elements,scratch,scratch_elements,n) ||
        !all_finite(left,n) || !all_finite(right,channels)) return -1;
    for (size_t i=0; i<n; ++i) scratch[i]=left[i]+right[i%channels];
    return publish(output,scratch,n);
}
static int pair_check(const float *left,size_t ln,const float *right,size_t rn,
    const int32_t *indices,size_t in,float *out,size_t on,float *scratch,size_t sn,
    size_t groups,size_t rows,size_t channels,size_t need) {
    size_t l,r;
    if (!product(groups,channels,&l) || !product(rows,channels,&r) ||
        ln<l || rn<r || in<rows) return 0;
    const void *inputs[]={left,right,indices}; size_t sizes[]={ln,rn,in};
    if (!buffers(inputs,sizes,3,out,on,scratch,sn,need) || !all_finite(left,l) || !all_finite(right,r)) return 0;
    for (size_t i=0; i<rows; ++i) if (indices[i]<0 || (size_t)indices[i]>=groups) return 0;
    return 1;
}
int ck_indexed_pair_features_f32(const float *left,size_t left_elements,
    const float *right,size_t right_elements,const int32_t *indices,size_t indices_elements,
    float *output,size_t output_elements,float *scratch,size_t scratch_elements,
    size_t groups,size_t rows,size_t channels) {
    size_t n;
    if (!product(rows,channels,&n) || n>SIZE_MAX/4) return -2;
    if (!pair_check(left,left_elements,right,right_elements,indices,indices_elements,
        output,output_elements,scratch,scratch_elements,groups,rows,channels,n*4)) return -1;
    for (size_t r=0;r<rows;++r) for (size_t c=0;c<channels;++c) {
        float a=left[(size_t)indices[r]*channels+c], b=right[r*channels+c];
        size_t k=r*channels*4+c;
        scratch[k]=a; scratch[k+channels]=b;
        scratch[k+2*channels]=a*b; scratch[k+3*channels]=fabsf(a-b);
    }
    return publish(output,scratch,n*4);
}
int ck_indexed_cosine_rows_f32(const float *left,size_t left_elements,
    const float *right,size_t right_elements,const int32_t *indices,size_t indices_elements,
    float *output,size_t output_elements,float *scratch,size_t scratch_elements,
    size_t groups,size_t rows,size_t channels,float epsilon) {
    if (!isfinite(epsilon) || epsilon<=0 || !pair_check(left,left_elements,right,right_elements,
        indices,indices_elements,output,output_elements,scratch,scratch_elements,
        groups,rows,channels,rows)) return -1;
    for (size_t r=0;r<rows;++r) {
        const float *a=left+(size_t)indices[r]*channels, *b=right+r*channels;
        double aa=0,bb=0;
        for(size_t c=0;c<channels;++c) {
            volatile double x=(double)a[c]*a[c], y=(double)b[c]*b[c];
            aa+=x; bb+=y;
        }
        double na=sqrt(aa),nb=sqrt(bb);
        if (!isfinite(na)||!isfinite(nb)||na>FLT_MAX||nb>FLT_MAX) return -1;
        float fa=fmaxf((float)na,epsilon), fb=fmaxf((float)nb,epsilon);
        double sum=0;
        for(size_t c=0;c<channels;++c) {
            float x=a[c]/fa,y=b[c]/fb;
            volatile float term=x*y;
            sum+=(double)term;
        }
        scratch[r]=(float)sum;
    }
    return publish(output,scratch,rows);
}
static int scalar_check(const float *input,size_t in,float *output,size_t on) {
    return region(input,in)&&region(output,on)&&!overlap(input,in,output,on)&&isfinite(input[0]);
}
int ck_scalar_exp_clamped_f32(const float *input,size_t input_elements,
    float *output,size_t output_elements,float maximum) {
    if (!scalar_check(input,input_elements,output,output_elements)||!isfinite(maximum)) return -1;
    float value=expf(fminf(input[0],maximum));
    if (!isfinite(value)) return -1;
    output[0]=value; return 0;
}
int ck_scalar_sigmoid_f32(const float *input,size_t input_elements,
    float *output,size_t output_elements) {
    if (!scalar_check(input,input_elements,output,output_elements)) return -1;
    float x=input[0],e=expf(-fabsf(x));
    float value=x>=0 ? 1.0f/(1.0f+e) : e/(1.0f+e);
    output[0]=value; return 0;
}
int ck_scale_vector_f32(const float *input,size_t input_elements,
    const float *gain,size_t gain_elements,float *output,size_t output_elements,
    float *scratch,size_t scratch_elements,size_t rows) {
    if(!rows || input_elements<rows || gain_elements<1) return -2;
    const void *inputs[]={input,gain}; size_t sizes[]={input_elements,gain_elements};
    if(!buffers(inputs,sizes,2,output,output_elements,scratch,scratch_elements,rows)||
       !all_finite(input,rows)||!all_finite(gain,1)) return -1;
    for(size_t r=0;r<rows;++r) scratch[r]=input[r]*gain[0];
    return publish(output,scratch,rows);
}
