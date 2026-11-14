/* Synthetic stateful model for native session ownership tests, not an oracle. */
#include "ck_model_abi_v8.h"
#include <stdint.h>
#include <stdatomic.h>
#include <string.h>
#include <sched.h>
#include <math.h>
static int pos, fail_once, bad_logits_once, calls, resets;
static float state, logits[8];
static atomic_int hold, entered;
static const atomic_int *cancel;
static int32_t ids[1]={1};
static const uint64_t caps = CK_MODEL_CAP_INIT | CK_MODEL_CAP_AUTOREGRESSIVE_DECODE |
    CK_MODEL_CAP_TEXT_ENCODE | CK_MODEL_CAP_TOKEN_DECODE | CK_MODEL_CAP_MIXED_EMBEDDING_PREFILL;
uint32_t ck_model_get_abi_version(void) { return CK_MODEL_ABI_V8_VERSION; }
uint64_t ck_model_get_capabilities(void) { return caps; }
int ck_model_get_runtime_descriptor(CKModelRuntimeDescriptorV8 *out, size_t size) {
    if(size<sizeof(*out)) return -1;
    *out=(CKModelRuntimeDescriptorV8){.struct_size=sizeof(*out),.abi_version=CK_MODEL_ABI_V8_VERSION,
        .capabilities=caps,.artifact_role=CK_MODEL_ROLE_DECODER,.context_length=32,.vocab_size=8};return 0;
}
int ck_model_init(const char *path) {(void)path;pos=0;state=0;return 0;}
void ck_model_free(void) {}
void ck_model_kv_cache_reset(void) {pos=0;state=0;resets++;}
void ck_model_set_cancel_flag(const atomic_int *flag) {cancel=flag;}
int ck_model_get_context_window(void) {return 32;}
int ck_model_get_vocab_size(void) {return 8;}
int ck_model_get_active_tokens(void) {return pos;}
int ck_model_get_logits_stride(void) {return 0;}
float *ck_model_get_logits(void) {return logits;}
#ifndef OMIT_WIDTH
int ck_model_get_visual_prefix_width(void) {return 4;}
#endif
static void update(void) {for(int i=0;i<8;i++) logits[i]=-fabsf(state-(float)i);}
int ck_model_forward_segments_grid_ex(const int32_t *before,int nb,const float *prefix,
        int np,int width,int gx,int gy,int tp,const int32_t *after,int na,float *out) {
    calls++;atomic_store(&entered,1);
    state=0;pos=0;
    for(int i=0;i<nb;i++) {state=state*0.5f+before[i];pos++;}
    for(int i=0;i<np*width;i++) state=state*0.5f+prefix[i];
    pos+=np;state+=(float)(gx+gy+tp)*0.03125f;
    for(int i=0;i<na;i++) {state=state*0.5f+after[i];pos++;}
    while(atomic_load(&hold) && !(cancel && atomic_load(cancel))) sched_yield();
    if(fail_once) {fail_once=0;return -1;}
    update();if(bad_logits_once) {bad_logits_once=0;logits[0]=NAN;}
    if(out) memcpy(out,logits,sizeof(logits));return 0;
}
int ck_model_forward_mixed(const float *p,int n,const int32_t *ids,int count,float *out) {
    return ck_model_forward_segments_grid_ex(0,0,p,n,4,n,1,n,ids,count,out);
}
int ck_model_embed_tokens(const int32_t *t,int n) {for(int i=0;i<n;i++)state+=t[i];pos+=n;return 0;}
int ck_model_forward(float *out) {update();if(out)memcpy(out,logits,sizeof(logits));return 0;}
int ck_model_decode(int32_t token,float *out) {state=state*0.5f+token;pos++;return ck_model_forward(out);}
int ck_model_encode_text(const char *s,int n) {(void)s;(void)n;return 1;}
int ck_model_can_encode_text(void) {return 1;}
const int32_t *ck_model_get_token_buffer(void) {return ids;}
int ck_model_decode_tokens(const int32_t *ids,int n,char *text,int capacity) {
    if(n!=1 || capacity<2)return -1;text[0]='A'+ids[0];text[1]=0;return 1;
}
float test_state(void) {return state;}
int test_calls(void) {return calls;}
int test_resets(void) {return resets;}
void test_fail(void) {fail_once=1;}
void test_hold(int value) {atomic_store(&entered,0);atomic_store(&hold,value);}
int test_entered(void) {return atomic_load(&entered);}

void test_bad_logits(void) {bad_logits_once=1;}
