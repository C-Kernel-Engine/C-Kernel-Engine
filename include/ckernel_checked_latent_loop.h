#ifndef CKERNEL_CHECKED_LATENT_LOOP_H
#define CKERNEL_CHECKED_LATENT_LOOP_H
#include "ckernel_checked_guided_flow.h"
typedef struct {
    ck_flow_entry embed,prefix,aggregate;
    uint8_t *backbone_arena,*aggregator_arena;
    size_t backbone_bytes,aggregator_bytes;
    size_t request_rows,embeddings,hidden,stop_logits;
    size_t aggregate_latent,aggregate_class_id,aggregate_output;
    size_t width,latent_width,capacity,prompt_rows,max_frames,min_frames;
    ck_guided_flow_request *flow;
    float *staging,*output;
    size_t staging_elements,output_elements;
    size_t *valid_frames;
    ck_flow_entry decode; /* optional generated cached continuation */
    size_t decode_input,decode_hidden,decode_stop_logits,decode_position;
    uint32_t policy_version; /* checked_latent_loop.v1 */
} ck_checked_latent_loop;
/* Executes the generated embedding/backbone, native guided flow and generated
 * latent aggregator. Recomputes the prefix, does not claim KV caching. On EOS
 * publishes all latents; budget (2), cancellation (-7) or failure publishes none.
 * This driver owns and writes flow condition/history staging, despite the inner
 * guided-flow API treating those buffers as read-only. They must be writable,
 * request-local arrays. Model arenas are exclusive for the duration of the call.
 * Caller fills prompt IDs in the generated backbone arena before this call. */
int ck_run_checked_latent_loop(const ck_checked_latent_loop *request);
#endif
