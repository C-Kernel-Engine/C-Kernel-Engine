/* Generic checked contiguous gather/scatter for fixed-capacity training. */
#include "checked_float_buffer.h"
#include <float.h>
#include <string.h>
static int extents(size_t rows,size_t capacity,size_t channels,size_t *n,size_t *table) {
    if (!rows || !capacity || !channels || rows>SIZE_MAX/channels || capacity>SIZE_MAX/channels) return 0;
    *n=rows*channels;*table=capacity*channels;return 1;
}
int ck_indexed_rows_f32(const float *table,size_t table_elements,const int32_t *ids,size_t id_elements,
    float *output,size_t output_elements,size_t rows,size_t capacity,size_t channels) {
    size_t n,t;if (!extents(rows,capacity,channels,&n,&t) || table_elements<t || id_elements<rows || output_elements<n) return -2;
    if (!ck_checked_region(table,t)||!ck_checked_region((const float *)ids,rows)||!ck_checked_region(output,n)||
        ck_checked_overlap(table,t,output,n)||ck_checked_overlap((const float *)ids,rows,output,n)) return -1;
    for(size_t r=0;r<rows;++r) if(ids[r]<0 || (size_t)ids[r]>=capacity || !ck_checked_finite(table+(size_t)ids[r]*channels,channels)) return -3;
    for(size_t r=0;r<rows;++r) memcpy(output+r*channels,table+(size_t)ids[r]*channels,channels*sizeof(float));
    return 0;
}
int ck_indexed_rows_backward_f32(const float *d_output,size_t d_output_elements,const int32_t *ids,size_t id_elements,
    float *d_table,size_t d_table_elements,float *scratch,size_t scratch_elements,size_t rows,size_t capacity,size_t channels) {
    size_t n,t;if(!extents(rows,capacity,channels,&n,&t)||d_output_elements<n||id_elements<rows||d_table_elements<t||scratch_elements<t)return -2;
    const float *p[]={d_output,(const float *)ids,d_table,scratch};size_t sizes[]={n,rows,t,t};
    for(size_t i=0;i<4;++i)if(!ck_checked_region(p[i],sizes[i]))return -1;
    for(size_t i=2;i<4;++i)for(size_t j=0;j<i;++j)if(ck_checked_overlap(p[i],sizes[i],p[j],sizes[j]))return -1;
    for(size_t r=0;r<rows;++r)if(ids[r]<0||(size_t)ids[r]>=capacity)return -3;
    if(!ck_checked_finite(d_output,n))return -3;
    for(size_t k=0;k<capacity;++k)for(size_t c=0;c<channels;++c){
        double sum=0;for(size_t r=0;r<rows;++r)if((size_t)ids[r]==k)sum+=(double)d_output[r*channels+c];
        if(!isfinite(sum)||fabs(sum)>FLT_MAX)return -4;
        scratch[k*channels+c]=(float)sum;
    }
    memcpy(d_table,scratch,t*sizeof(float));return 0;
}
