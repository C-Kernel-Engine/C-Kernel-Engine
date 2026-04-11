/*
 * ckernel_model_load_v8.h - Load BUMPWGT5 weights using manifest
 *
 * Provides model loading functions and weight buffer management.
 */

#ifndef CKERNEL_MODEL_LOAD_V8_H
#define CKERNEL_MODEL_LOAD_V8_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckernel_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * MANIFEST MAP
 * ============================================================================ */

/* Maximum line length for manifest parsing */
#define CK_MANIFEST_LINE_MAX 4096

/* Weight info from manifest */
typedef struct {
    const char *name;
    int dtype;
    uint64_t file_offset;
    uint64_t size;
    uint64_t runtime_offset;
} ck_weight_info_t;

/* Manifest map handle */
typedef struct {
    ck_weight_info_t *entries;
    int count;
    FILE *file;
    uint8_t *mapped_base;
    int weights_materialized;
} ck_manifest_map_t;

/* ============================================================================
 * LOAD FUNCTIONS
 * ============================================================================ */
/* Note: Model struct (e.g., QWEN2_DECODEModel) is generated per-model by codegen */

/*
 * ck_open_weights_manifest_v8() - Open and parse a weights manifest
 *
 * When materialize_weights is non-zero, tensor data is copied into the runtime
 * bump arena. When it is zero, the manifest is parsed but tensor bytes are
 * assumed to already be addressable from the arena backing.
 *
 * Returns: Manifest map handle or NULL on error
 */
ck_manifest_map_t *ck_open_weights_manifest_v8(void *base,
                                               const char *weights_path,
                                               const char *manifest_path,
                                               int materialize_weights);

/*
 * ck_load_weights_manifest_v7() - Backward-compatible eager materialization wrapper
 *
 * Returns: Manifest map handle or NULL on error
 */
ck_manifest_map_t *ck_load_weights_manifest_v7(void *base,
                                               const char *weights_path,
                                               const char *manifest_path);

/*
 * ck_unload_manifest_map() - Unload manifest map and close files
 */
void ck_unload_manifest_map(ck_manifest_map_t *manifest);

/*
 * ck_get_weight_info() - Get weight info by name
 *
 * Returns: Pointer to weight info or NULL if not found
 */
ck_weight_info_t *ck_get_weight_info(ck_manifest_map_t *manifest, const char *name);

/*
 * ck_load_weights() - Load all weights from BUMP file using manifest
 *
 * Returns: 0 on success, -1 on error
 */
int ck_load_weights(
    ck_manifest_map_t *manifest,
    const char *bump_path,
    void *arena,
    size_t arena_size
);

/*
 * ck_load_weight_by_name() - Load a single weight by name
 *
 * Returns: Bytes loaded or -1 on error
 */
int64_t ck_load_weight_by_name(
    ck_manifest_map_t *manifest,
    const char *bump_path,
    const char *weight_name,
    void *dest
);

/* ============================================================================
 * MODEL INITIALIZATION (model-specific, generated)
 * ============================================================================ */

/*
 * ck_init_model_from_bump() - Initialize model struct from BUMP file
 *
 * This function is generated per-model and populates the model's weight
 * pointers based on the layout information.
 *
 * Returns: 0 on success, -1 on error
 */
int ck_init_model_from_bump(
    void *model_struct,      /* Model struct to populate */
    const char *bump_path,   /* Path to .bump file */
    void *arena,             /* Pre-allocated weight buffer */
    size_t arena_size,
    const char *layout_json_path  /* Layout JSON for offsets */
);

/*
 * ck_get_model_offsets() - Get model weight offsets from layout
 *
 * Returns: Pointer to layout JSON dict or NULL on error
 */
void *ck_get_model_offsets(const char *layout_json_path);

#ifdef __cplusplus
}
#endif

#endif /* CKERNEL_MODEL_LOAD_V8_H */
