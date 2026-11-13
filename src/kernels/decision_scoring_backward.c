/* Checked indexed scoring VJPs. No allocation, scheduler or model identity. */
#include "ckernel_head_backward.h"
#include "checked_float_buffer.h"
#include <float.h>
#include <math.h>
#include <string.h>

static int product(size_t a, size_t b, size_t *out) {
    if (!a || !b || a > SIZE_MAX/b) return 0;
    *out=a*b; return 1;
}
static int store(float *p, double value) {
    if (!isfinite(value) || fabs(value)>FLT_MAX) return 0;
    *p=(float)value; return 1;
}
static int validate(const float *left,size_t ln,const float *right,size_t rn,
    const int32_t *indices,size_t in,const float *dy,size_t yn,
    float *dl,size_t dln,float *dr,size_t drn,float *scratch,size_t sn,
    size_t groups,size_t rows,size_t channels,int pair,size_t *nl,size_t *nr) {
    size_t ny;
    if (!product(groups,channels,nl) || !product(rows,channels,nr) ||
        *nr>(SIZE_MAX-*nl)/2 || (pair && *nr>SIZE_MAX/4)) return -2;
    ny=pair ? 4*(*nr) : rows;
    if (ln<*nl || rn<*nr || in<rows || yn<ny || dln<*nl || drn<*nr || sn<*nl+2*(*nr)) return -2;
    const void *ptrs[]={left,right,indices,dy,dl,dr,scratch};
    size_t sizes[]={ln,rn,in,yn,dln,drn,sn};
    for (size_t i=0;i<7;++i) if (!ck_checked_region(ptrs[i],sizes[i])) return -1;
    for (size_t i=4;i<7;++i) for(size_t j=0;j<i;++j)
        if (ck_checked_overlap(ptrs[i],sizes[i],ptrs[j],sizes[j])) return -1;
    for(size_t r=0;r<rows;++r) if(indices[r]<0 || (size_t)indices[r]>=groups) return -2;
    if (!ck_checked_finite(left,*nl) || !ck_checked_finite(right,*nr) || !ck_checked_finite(dy,ny)) return -3;
    return 0;
}
/* One rounded row contribution, then ascending row FP64 scatter reduction.
 * Never overwrite a repeated owner's gradient with its last option's value. */
static int publish(const int32_t *indices,float *dl,float *dr,float *scratch,
    size_t groups,size_t rows,size_t channels,size_t nl,size_t nr) {
    float *row_left=scratch+nl, *row_right=row_left+nr;
    for(size_t g=0;g<groups;++g) for(size_t c=0;c<channels;++c) {
        double sum=0;
        for(size_t r=0;r<rows;++r) if((size_t)indices[r]==g) sum+=(double)row_left[r*channels+c];
        if(!store(scratch+g*channels+c,sum)) return -4;
    }
    memcpy(dl,scratch,nl*sizeof(float)); memcpy(dr,row_right,nr*sizeof(float)); return 0;
}
int ck_indexed_pair_features_backward_f32(
    const float *left,size_t left_elements,const float *right,size_t right_elements,
    const int32_t *indices,size_t indices_elements,const float *d_output,size_t d_output_elements,
    float *d_left,size_t d_left_elements,float *d_right,size_t d_right_elements,
    float *scratch,size_t scratch_elements,size_t groups,size_t rows,size_t channels) {
    size_t nl,nr;
    int rc=validate(left,left_elements,right,right_elements,indices,indices_elements,d_output,d_output_elements,
        d_left,d_left_elements,d_right,d_right_elements,scratch,scratch_elements,groups,rows,channels,1,&nl,&nr);
    if(rc) return rc;
    for(size_t r=0;r<rows;++r) for(size_t c=0;c<channels;++c) {
        double a=left[(size_t)indices[r]*channels+c],b=right[r*channels+c];
        const float *dy=d_output+4*r*channels+c;
        double sign=(a>b)-(a<b); /* abs'(0)=0, including signed zeros. */
        volatile double pa=(double)dy[2*channels]*b, pb=(double)dy[2*channels]*a;
        volatile double diff=(double)dy[3*channels]*sign;
        if(!store(scratch+nl+r*channels+c,((double)dy[0]+pa)+diff) ||
           !store(scratch+nl+nr+r*channels+c,((double)dy[channels]+pb)-diff)) return -4;
    }
    return publish(indices,d_left,d_right,scratch,groups,rows,channels,nl,nr);
}
static int cosine_backward(
    const float *left,size_t left_elements,const float *right,size_t right_elements,
    const int32_t *indices,size_t indices_elements,const float *d_output,size_t d_output_elements,
    float *d_left,size_t d_left_elements,float *d_right,size_t d_right_elements,
    float *scratch,size_t scratch_elements,size_t groups,size_t rows,size_t channels,
    float epsilon,int straight_through_clamp) {
    size_t nl,nr;
    if(!isfinite(epsilon)||epsilon<=0) return -2;
    int rc=validate(left,left_elements,right,right_elements,indices,indices_elements,d_output,d_output_elements,
        d_left,d_left_elements,d_right,d_right_elements,scratch,scratch_elements,groups,rows,channels,0,&nl,&nr);
    if(rc) return rc;
    for(size_t r=0;r<rows;++r) {
        const float *a=left+(size_t)indices[r]*channels,*b=right+r*channels;
        double aa=0,bb=0;
        for(size_t c=0;c<channels;++c) {
            volatile double ap=(double)a[c]*a[c],bp=(double)b[c]*b[c];aa+=ap;bb+=bp;
        }
        double na=sqrt(aa),nb=sqrt(bb);
        if(!isfinite(na)||!isfinite(nb)||na>FLT_MAX||nb>FLT_MAX) return -4;
        float raw_a=(float)na,raw_b=(float)nb,fa=fmaxf(raw_a,epsilon),fb=fmaxf(raw_b,epsilon);
        double xy=0;
        for(size_t c=0;c<channels;++c) {
            float x=a[c]/fa,y=b[c]/fb;volatile float term=x*y;xy+=(double)term;
        }
        /* normalize(...).clamp_min has zero derivative below epsilon.
         * PyTorch 2.12.1 cosine_similarity clamps a cloned norm under NoGrad,
         * retaining the norm derivative there. Zero norm still has zero VJP. */
        int da=raw_a>0 && (straight_through_clamp || raw_a>=epsilon);
        int db=raw_b>0 && (straight_through_clamp || raw_b>=epsilon);
        for(size_t c=0;c<channels;++c) {
            float x=a[c]/fa,y=b[c]/fb;
            double ga=(double)y/fa,gb=(double)x/fb;
            if(da) ga-=xy*((double)a[c]/raw_a)/fa;
            if(db) gb-=xy*((double)b[c]/raw_b)/fb;
            if(!store(scratch+nl+r*channels+c,(double)d_output[r]*ga) ||
               !store(scratch+nl+nr+r*channels+c,(double)d_output[r]*gb)) return -4;
        }
    }
    return publish(indices,d_left,d_right,scratch,groups,rows,channels,nl,nr);
}
#define ARGS const float *left,size_t left_elements,const float *right,size_t right_elements, \
    const int32_t *indices,size_t indices_elements,const float *d_output,size_t d_output_elements, \
    float *d_left,size_t d_left_elements,float *d_right,size_t d_right_elements, \
    float *scratch,size_t scratch_elements,size_t groups,size_t rows,size_t channels,float epsilon
#define PASS left,left_elements,right,right_elements,indices,indices_elements,d_output,d_output_elements, \
    d_left,d_left_elements,d_right,d_right_elements,scratch,scratch_elements,groups,rows,channels,epsilon
int ck_indexed_normalized_dot_backward_f32(ARGS) {return cosine_backward(PASS,0);}
int ck_indexed_cosine_rows_backward_f32(ARGS) {return cosine_backward(PASS,1);}
#undef ARGS
#undef PASS

static int scalar_vjp_validate(const float *input, size_t input_elements,
    const float *dy, size_t dy_elements, float *dx, size_t dx_elements) {
    if (!input_elements || !dy_elements || !dx_elements) return -2;
    if (!ck_checked_region(input,input_elements) || !ck_checked_region(dy,dy_elements) ||
        !ck_checked_region(dx,dx_elements) ||
        ck_checked_overlap(dx,dx_elements,input,input_elements) ||
        ck_checked_overlap(dx,dx_elements,dy,dy_elements)) return -1;
    return isfinite(input[0]) && isfinite(dy[0]) ? 0 : -3;
}
int ck_scalar_exp_clamped_backward_f32(const float *input, size_t input_elements,
    const float *d_output, size_t d_output_elements,
    float *d_input, size_t d_input_elements, float maximum) {
    int rc=scalar_vjp_validate(input,input_elements,d_output,d_output_elements,d_input,d_input_elements);
    if (rc) return rc;
    if (!isfinite(maximum)) return -2;
    float y=expf(fminf(input[0],maximum)), result;
    if (!isfinite(y)) return -4;
    /* clamp(max=...) has derivative one AT the boundary, zero above it. */
    if (!store(&result,input[0]<=maximum ? (double)d_output[0]*y : 0.0)) return -4;
    d_input[0]=result; return 0;
}
int ck_scalar_sigmoid_backward_f32(const float *input, size_t input_elements,
    const float *d_output, size_t d_output_elements,
    float *d_input, size_t d_input_elements) {
    int rc=scalar_vjp_validate(input,input_elements,d_output,d_output_elements,d_input,d_input_elements);
    if (rc) return rc;
    float x=input[0], e=expf(-fabsf(x));
    float y=x>=0 ? 1.0f/(1.0f+e) : e/(1.0f+e), result;
    volatile double slope=(double)y*(1.0-(double)y);
    if (!store(&result,(double)d_output[0]*slope)) return -4;
    d_input[0]=result; return 0;
}
int ck_scale_vector_backward_f32(const float *input, size_t input_elements,
    const float *gain, size_t gain_elements, const float *d_output, size_t d_output_elements,
    float *d_input, size_t d_input_elements, float *d_gain, size_t d_gain_elements,
    float *scratch, size_t scratch_elements, size_t rows) {
    if (!rows || rows==SIZE_MAX || input_elements<rows || gain_elements<1 ||
        d_output_elements<rows || d_input_elements<rows || d_gain_elements<1 ||
        scratch_elements<rows+1) return -2;
    const void *ptrs[]={input,gain,d_output,d_input,d_gain,scratch};
    size_t sizes[]={input_elements,gain_elements,d_output_elements,d_input_elements,d_gain_elements,scratch_elements};
    for (size_t i=0;i<6;++i) if (!ck_checked_region(ptrs[i],sizes[i])) return -1;
    for (size_t i=3;i<6;++i) for (size_t j=0;j<i;++j)
        if (ck_checked_overlap(ptrs[i],sizes[i],ptrs[j],sizes[j])) return -1;
    if (!ck_checked_finite(input,rows) || !ck_checked_finite(gain,1) || !ck_checked_finite(d_output,rows)) return -3;
    double sum=0;
    for (size_t r=0;r<rows;++r) {
        if (!store(scratch+r,(double)d_output[r]*gain[0])) return -4;
        volatile double product=(double)d_output[r]*input[r]; sum+=product;
    }
    if (!store(scratch+rows,sum)) return -4;
    memcpy(d_input,scratch,rows*sizeof(float)); d_gain[0]=scratch[rows]; return 0;
}
