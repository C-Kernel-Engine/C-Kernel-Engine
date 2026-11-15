#ifndef CKERNEL_CHECKED_GUIDED_FLOW_H
#define CKERNEL_CHECKED_GUIDED_FLOW_H
#include <stddef.h>
#include <stdint.h>
typedef int (*ck_flow_entry)(uint8_t *,size_t);
typedef int (*ck_flow_noise)(float *,size_t,size_t,uint64_t,uint64_t);
typedef int (*ck_flow_cancel)(void *);
/* Artifact-bound offsets, in bytes, are supplied from generated layout metadata.
 * Model output points to the selected patch, not the complete sequence output. */
typedef struct {
    ck_flow_entry entry;
    uint8_t *arena;
    size_t arena_bytes;
    size_t latent,time,condition,history,prediction;
} ck_flow_model;
typedef struct {
    ck_flow_entry entry;
    uint8_t *arena;
    size_t arena_bytes;
    size_t latent,prediction,null_prediction,noise,neg_one,cfg,dt,amplitude,output;
} ck_flow_update;
typedef struct {
    ck_flow_model model;
    ck_flow_update update;
    ck_flow_noise noise;
    const float *condition,*history,*times;
    size_t latent_elements,condition_elements,history_elements,steps;
    float cfg,sigma,temperature;
    uint64_t seed,stream;
    float *workspace;
    size_t workspace_elements;
    float *output;
    size_t output_elements;
    ck_flow_cancel should_cancel;
    void *cancel_context;
} ck_guided_flow_request;
/* 0=complete; -1=invalid contract; -2=native stage failed; -3=invalid
 * numeric output; -7=cancelled. Output is unchanged unless the run completes.
 * Timesteps and native counter RNG are explicit, separate from PyTorch RNG.
 * The caller holds both libraries/arenas alive and serializes access. */
int ck_validate_guided_flow_request(const ck_guided_flow_request *request);
int ck_run_checked_guided_flow(const ck_guided_flow_request *request);
#endif
