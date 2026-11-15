#include "ckernel_checked_latent_loop.h"
#include <math.h>
#include <string.h>
static int region(const void *p,size_t n,size_t align) {
    return p && n && (uintptr_t)p%align==0 && (uintptr_t)p<=UINTPTR_MAX-n;
}
static int overlap(const void *a,size_t na,const void *b,size_t nb) {
    return (uintptr_t)a<(uintptr_t)b+nb && (uintptr_t)b<(uintptr_t)a+na;
}
static int slices(size_t bytes,const size_t *off,const size_t *len,size_t count) {
    for(size_t i=0;i<count;++i) {
        if(off[i]%4 || off[i]>bytes || len[i]>(bytes-off[i])/4) return 0;
        for(size_t j=0;j<i;++j) if(off[i]<off[j]+len[j]*4 && off[j]<off[i]+len[i]*4) return 0;
    }
    return 1;
}
static int finite_values(const float *p,size_t n) {
    for(size_t i=0;i<n;++i) if(!isfinite(p[i])) return 0;
    return 1;
}
int ck_run_checked_latent_loop(const ck_checked_latent_loop *r) {
    if(!r || r->policy_version!=1 || !r->embed || !r->prefix || !r->aggregate || !r->flow ||
       ck_validate_guided_flow_request(r->flow) || !r->width || r->width>8192 ||
       !r->latent_width || r->latent_width>4096 || !r->capacity || r->capacity>512 ||
       !r->prompt_rows || r->prompt_rows>r->capacity || !r->max_frames || r->max_frames>1024 ||
       !r->min_frames || r->min_frames>r->max_frames ||
       r->flow->latent_elements!=r->latent_width || r->flow->history_elements!=r->latent_width ||
       r->flow->condition_elements!=r->width || r->flow->stream>UINT64_MAX-r->max_frames ||
       r->staging_elements<r->max_frames*r->latent_width || r->staging_elements>SIZE_MAX/4 ||
       r->output_elements<r->max_frames*r->latent_width || r->output_elements>SIZE_MAX/4) return -1;
    const ck_guided_flow_request *f=r->flow;
    const void *ptr[]={r->backbone_arena,r->aggregator_arena,f->model.arena,f->update.arena,
        f->condition,f->history,f->times,f->workspace,f->output,r->staging,r->output,r->valid_frames,r,f};
    size_t sizes[]={r->backbone_bytes,r->aggregator_bytes,f->model.arena_bytes,f->update.arena_bytes,
        r->width*4,r->latent_width*4,(f->steps+1)*4,f->workspace_elements*4,f->output_elements*4,
        r->staging_elements*4,r->output_elements*4,sizeof(size_t),sizeof(*r),sizeof(*f)};
    for(size_t i=0;i<14;++i) {
        if(!region(ptr[i],sizes[i],i==11?_Alignof(size_t):i==12?_Alignof(ck_checked_latent_loop):i==13?_Alignof(ck_guided_flow_request):4)) return -1;
        for(size_t j=0;j<i;++j) if(overlap(ptr[i],sizes[i],ptr[j],sizes[j])) return -1;
    }
    size_t bo[]={r->request_rows,r->embeddings,r->hidden,r->stop_logits,
        r->decode_input,r->decode_hidden,r->decode_stop_logits,r->decode_position};
    size_t bn[]={1,r->capacity*r->width,r->capacity*r->width,r->capacity*2,r->width,r->width,2,1};
    size_t ao[]={r->aggregate_latent,r->aggregate_class_id,r->aggregate_output};
    size_t an[]={r->latent_width,1,r->width};
    if(!slices(r->backbone_bytes,bo,bn,r->decode?8:4) || !slices(r->aggregator_bytes,ao,an,3)) return -1;
    *r->valid_frames=0;
    ck_guided_flow_request flow=*f;
    memset((float *)flow.history,0,r->latent_width*4);
    int32_t *rows=(int32_t *)(r->backbone_arena+r->request_rows);
    float *emb=(float *)(r->backbone_arena+r->embeddings);
    float *hidden=(float *)(r->backbone_arena+r->hidden);
    float *stop=(float *)(r->backbone_arena+r->stop_logits);
    *rows=(int32_t)r->prompt_rows;
    if(flow.should_cancel && flow.should_cancel(flow.cancel_context)) return -7;
    if(r->embed(r->backbone_arena,r->backbone_bytes)) return -2;
    for(size_t step=0;step<r->max_frames;++step) {
        if(flow.should_cancel && flow.should_cancel(flow.cancel_context)) return -7;
        if(*rows<1 || (size_t)*rows>r->capacity) return -1;
        size_t row=(size_t)*rows-1;
        if(step && r->decode) {
            memcpy(r->backbone_arena+r->decode_input,emb+row*r->width,r->width*4);
            *(int32_t *)(r->backbone_arena+r->decode_position)=(int32_t)row;
            if(r->decode(r->backbone_arena,r->backbone_bytes)) return -2;
            hidden=(float *)(r->backbone_arena+r->decode_hidden);
            stop=(float *)(r->backbone_arena+r->decode_stop_logits);row=0;
        } else {
            if(r->prefix(r->backbone_arena,r->backbone_bytes)) return -2;
            if(*rows<1 || (size_t)*rows>r->capacity) return -1;
            row=(size_t)*rows-1;
        }
        if(!finite_values(hidden+row*r->width,r->width) || !finite_values(stop+row*2,2)) return -3;
        memcpy((float *)flow.condition,hidden+row*r->width,r->width*4);
        flow.stream=f->stream+step;
        int status=ck_run_checked_guided_flow(&flow);
        if(status) return status;
        memcpy(r->staging+step*r->latent_width,flow.output,r->latent_width*4);
        if(step+1>=r->min_frames && stop[row*2+1]>stop[row*2]) {
            if(flow.should_cancel && flow.should_cancel(flow.cancel_context)) return -7;
            memcpy(r->output,r->staging,(step+1)*r->latent_width*4);
            *r->valid_frames=step+1;return 0;
        }
        if(step+1==r->max_frames || (size_t)*rows>=r->capacity) return 2;
        memcpy(r->aggregator_arena+r->aggregate_latent,flow.output,r->latent_width*4);
        *(int32_t *)(r->aggregator_arena+r->aggregate_class_id)=0;
        if(flow.should_cancel && flow.should_cancel(flow.cancel_context)) return -7;
        if(r->aggregate(r->aggregator_arena,r->aggregator_bytes)) return -2;
        float *next=(float *)(r->aggregator_arena+r->aggregate_output);
        if(!finite_values(next,r->width)) return -3;
        memcpy(emb+(size_t)*rows*r->width,next,r->width*4);
        memcpy((float *)flow.history,flow.output,r->latent_width*4);
        ++*rows;
    }
    return 2;
}
