#include "ckernel_checked_guided_flow.h"
#include <math.h>
#include <string.h>

static int region(const void *p,size_t n,size_t alignment) {
    return p && n && (uintptr_t)p % alignment == 0 && (uintptr_t)p <= UINTPTR_MAX-n;
}
static int overlap(const void *a,size_t na,const void *b,size_t nb) {
    return (uintptr_t)a < (uintptr_t)b+nb && (uintptr_t)b < (uintptr_t)a+na;
}
static int finite_values(const float *p,size_t n) {
    for(size_t i=0;i<n;++i) if(!isfinite(p[i])) return 0;
    return 1;
}
static int slices(size_t bytes,const size_t *offsets,const size_t *counts,size_t n) {
    for(size_t i=0;i<n;++i) {
        if(offsets[i]%_Alignof(float) || offsets[i]>bytes || counts[i]>(bytes-offsets[i])/sizeof(float)) return 0;
        for(size_t j=0;j<i;++j)
            if(offsets[i]<offsets[j]+counts[j]*sizeof(float) && offsets[j]<offsets[i]+counts[i]*sizeof(float)) return 0;
    }
    return 1;
}
#define MF(name) ((float *)(m->arena+m->name))
#define UF(name) ((float *)(u->arena+u->name))
static int cancelled(const ck_guided_flow_request *r) {
    return r->should_cancel && r->should_cancel(r->cancel_context);
}
int ck_validate_guided_flow_request(const ck_guided_flow_request *r) {
    if(!r || !r->model.entry || !r->update.entry || !r->noise ||
       !r->latent_elements || r->latent_elements>4096 ||
       !r->condition_elements || r->condition_elements>8192 ||
       !r->history_elements || r->history_elements>65536 || !r->steps || r->steps>64 ||
       !isfinite(r->cfg) || fabsf(r->cfg)>100 || !isfinite(r->sigma) || r->sigma<0 || r->sigma>100 ||
       !isfinite(r->temperature) || r->temperature<0 || r->temperature>100) return -1;
    const ck_flow_model *m=&r->model;const ck_flow_update *u=&r->update;
    size_t n=r->latent_elements,c=r->condition_elements,h=r->history_elements;
    size_t noise_count=(r->steps+1)*n;
    if(r->workspace_elements<noise_count || r->workspace_elements>SIZE_MAX/sizeof(float) ||
       r->output_elements<n || r->output_elements>SIZE_MAX/sizeof(float)) return -1;
    const void *p[]={m->arena,u->arena,r->condition,r->history,r->times,r->workspace,r->output,r};
    size_t sizes[]={m->arena_bytes,u->arena_bytes,c*4,h*4,(r->steps+1)*4,r->workspace_elements*4,r->output_elements*4,sizeof(*r)};
    for(size_t i=0;i<8;++i) {
        if(!region(p[i],sizes[i],i==7?_Alignof(ck_guided_flow_request):_Alignof(float))) return -1;
        for(size_t j=0;j<i;++j) if(overlap(p[i],sizes[i],p[j],sizes[j])) return -1;
    }
    size_t mo[]={m->latent,m->time,m->condition,m->history,m->prediction};size_t mn[]={n,1,c,h,n};
    size_t uo[]={u->latent,u->prediction,u->null_prediction,u->noise,u->neg_one,u->cfg,u->dt,u->amplitude,u->output};
    size_t un[]={n,n,n,n,1,1,1,1,n};
    if(!slices(m->arena_bytes,mo,mn,5) || !slices(u->arena_bytes,uo,un,9) ||
       !finite_values(r->condition,c) || !finite_values(r->history,h) || !finite_values(r->times,r->steps+1) ||
       r->times[0]!=0 || r->times[r->steps]!=1) return -1;
    for(size_t i=1;i<=r->steps;++i) if(!(r->times[i]>r->times[i-1])) return -1;
    return 0;
}
int ck_run_checked_guided_flow(const ck_guided_flow_request *r) {
    if(ck_validate_guided_flow_request(r)) return -1;
    const ck_flow_model *m=&r->model;const ck_flow_update *u=&r->update;
    size_t n=r->latent_elements,c=r->condition_elements,h=r->history_elements;
    size_t noise_count=(r->steps+1)*n;
    if(cancelled(r)) return -7;
    if(r->noise(r->workspace,r->workspace_elements,noise_count,r->seed,r->stream)) return -2;
    if(!finite_values(r->workspace,noise_count)) return -3;
    memcpy(UF(latent),r->workspace,n*4);
    *UF(neg_one)=-1;*UF(cfg)=r->cfg;
    for(size_t step=0;step<r->steps;++step) {
        if(cancelled(r)) return -7;
        memcpy(MF(latent),UF(latent),n*4);memcpy(MF(history),r->history,h*4);
        memcpy(MF(condition),r->condition,c*4);*MF(time)=r->times[step];
        if(m->entry(m->arena,m->arena_bytes)) return -2;
        if(!finite_values(MF(prediction),n)) return -3;
        memcpy(UF(prediction),MF(prediction),n*4);
        if(cancelled(r)) return -7;
        /* Refill all inputs: an entry's internal scratch may overlap its inputs. */
        memcpy(MF(latent),UF(latent),n*4);memcpy(MF(history),r->history,h*4);
        memset(MF(condition),0,c*4);*MF(time)=r->times[step];
        if(m->entry(m->arena,m->arena_bytes)) return -2;
        if(!finite_values(MF(prediction),n)) return -3;
        memcpy(UF(null_prediction),MF(prediction),n*4);
        if(cancelled(r)) return -7;
        float dt=r->times[step+1]-r->times[step];
        *UF(dt)=dt;*UF(amplitude)=(r->sigma*sqrtf(r->temperature))*sqrtf(dt);
        memcpy(UF(noise),r->workspace+(step+1)*n,n*4);
        if(u->entry(u->arena,u->arena_bytes)) return -2;
        if(!finite_values(UF(output),n)) return -3;
        memcpy(UF(latent),UF(output),n*4);
    }
    if(cancelled(r)) return -7;
    memcpy(r->output,UF(latent),n*4);
    return 0;
}
