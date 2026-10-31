#ifndef CK_SESSION_V8_H
#define CK_SESSION_V8_H

#include <stddef.h>
#include <stdint.h>

#include "ck_model_abi_v8.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CK_SESSION_ABI_V8_VERSION UINT32_C(1)

typedef struct CKSessionV8 CKSessionV8;

/* v1 ownership and concurrency contract:
 * - The caller owns all request strings, output buffers, and callback user_data.
 *   Keep them alive until the synchronous call returns; the session retains none.
 * - A token callback runs synchronously inside generate. Its utf8_text pointer is
 *   borrowed only for that callback invocation. Copy it before returning if needed.
 * - The host must serialize encode, decode, format_chat, generate, and reset with
 *   one another. Current generated-model state is not certified for concurrent
 *   generation by separate sessions in one process.
 * - cancel may be called from another thread while generate runs. It requests a
 *   cooperative stop; only generate returning acknowledges that native work has
 *   finished. A slow kernel can delay that return. Do not release request buffers,
 *   callback user_data, or session storage on the strength of cancel returning.
 * - Quiesce all other callers before close; close is not a concurrent free API.
 *   The pointer returned by last_error is session-owned and becomes invalid after
 *   another session call or close.
 */

enum CKSessionStatusV8 {
    CK_SESSION_V8_OK = 0,
    CK_SESSION_V8_ERROR_INVALID_ARGUMENT = -1,
    CK_SESSION_V8_ERROR_ABI = -2,
    CK_SESSION_V8_ERROR_LOAD = -3,
    CK_SESSION_V8_ERROR_INIT = -4,
    CK_SESSION_V8_ERROR_CAPABILITY = -5,
    CK_SESSION_V8_ERROR_BUSY = -6,
    CK_SESSION_V8_ERROR_RUNTIME = -7,
    CK_SESSION_V8_ERROR_BUFFER_TOO_SMALL = -8,
    CK_SESSION_V8_ERROR_CANCELLED = -9
};

enum CKSessionStopReasonV8 {
    CK_SESSION_STOP_NONE = 0,
    CK_SESSION_STOP_EOS = 1,
    CK_SESSION_STOP_TOKEN_LIMIT = 2,
    CK_SESSION_STOP_CANCELLED = 3,
    CK_SESSION_STOP_CALLBACK = 4,
    CK_SESSION_STOP_RUNTIME_ERROR = 5
};

enum CKSessionRequestFlagsV8 {
    CK_SESSION_REQUEST_RAW_PROMPT = UINT32_C(1) << 0,
    CK_SESSION_REQUEST_IGNORE_EOS = UINT32_C(1) << 1,
    CK_SESSION_REQUEST_TIMESTAMPS = UINT32_C(1) << 2,
    /* Requests reset generated recurrent/KV state by default. Set this only
     * when the caller deliberately continues the preceding request. */
    CK_SESSION_REQUEST_CONTINUE_STATE = UINT32_C(1) << 3,
    /* Fail the request instead of silently truncating a prompt that does not
     * fit the remaining context budget. Serving layers set this so their
     * admission policy can never be bypassed by the native truncation
     * fallback. */
    CK_SESSION_REQUEST_REJECT_PROMPT_TRUNCATION = UINT32_C(1) << 4
};

typedef struct CKSessionConfigV8 {
    uint32_t struct_size;
    uint32_t abi_version;
    const char *model_library_path;
    const char *weights_path;
    const char *manifest_path;
    int32_t context_length;
    int32_t num_threads;
    uint64_t required_capabilities;
    uint64_t reserved[8];
} CKSessionConfigV8;

typedef struct CKSessionGenerateRequestV8 {
    uint32_t struct_size;
    uint32_t abi_version;
    const char *system_text;
    const char *user_text;
    int32_t max_tokens;
    /* Sampling is a host policy applied after the generated model's mandatory
     * generation policy (for example Whisper timestamp masking). */
    float temperature;
    float top_p;
    uint32_t flags;
    uint32_t reserved0;
    uint64_t reserved[8];
} CKSessionGenerateRequestV8;

typedef struct CKSessionGenerateResultV8 {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t prompt_tokens;
    int32_t generated_tokens;
    int32_t stop_reason;
    int32_t reserved0;
    double prefill_time_ms;
    double decode_time_ms;
    uint64_t reserved[8];
} CKSessionGenerateResultV8;

/* Read-only progress for the current or most recently completed generate.
 * Safe to query from another thread while generate runs, like cancel. A
 * completed layer means all its generated prefill operations returned for
 * the current chunk; it is not a count of completed prompt tokens.
 * chunk_start_position identifies that chunk's absolute starting position.
 * Older model libraries may not publish layer progress, in which case
 * total_layers remains zero. */
enum CKSessionProgressPhaseV8 {
    CK_SESSION_PROGRESS_IDLE = 0,
    CK_SESSION_PROGRESS_PREFILL = 1,
    CK_SESSION_PROGRESS_DECODE = 2,
    CK_SESSION_PROGRESS_COMPLETE = 3
};

typedef struct CKSessionProgressV8 {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t request_serial;
    int32_t phase;
    int32_t completed_layers;
    int32_t total_layers;
    int32_t chunk_tokens;
    int32_t chunk_start_position;
} CKSessionProgressV8;

/* Return zero to continue generation. A non-zero result requests a clean stop. */
typedef int (*ck_session_token_callback_v8)(
    void *user_data,
    int32_t token_id,
    const char *utf8_text,
    size_t utf8_length,
    int32_t sequence_index);

uint32_t ck_session_v8_get_abi_version(void);

int ck_session_v8_open(
    const CKSessionConfigV8 *config,
    CKSessionV8 **session_out);

void ck_session_v8_close(CKSessionV8 *session);

int ck_session_v8_get_model_descriptor(
    const CKSessionV8 *session,
    CKModelRuntimeDescriptorV8 *descriptor,
    size_t descriptor_size);

/* Size-query convention: pass output=NULL/capacity=0 to receive the required
 * element or byte count. The returned count excludes a text NUL terminator. */
int ck_session_v8_encode(
    CKSessionV8 *session,
    const char *text,
    int32_t *output,
    int32_t capacity);

int ck_session_v8_decode(
    CKSessionV8 *session,
    const int32_t *tokens,
    int32_t token_count,
    char *output,
    int32_t capacity);

/* Stop IDs from the actually loaded generated model. NULL output queries the
 * count; a caller buffer smaller than that count is rejected. */
int ck_session_v8_get_stop_token_ids(
    CKSessionV8 *session, int32_t *output, int32_t capacity);

int ck_session_v8_format_chat(
    CKSessionV8 *session,
    const char *system_text,
    const char *user_text,
    char *output,
    int32_t capacity);

int ck_session_v8_generate(
    CKSessionV8 *session,
    const CKSessionGenerateRequestV8 *request,
    ck_session_token_callback_v8 callback,
    void *user_data,
    CKSessionGenerateResultV8 *result);

void ck_session_v8_cancel(CKSessionV8 *session);
int ck_session_v8_get_progress(CKSessionV8 *session, CKSessionProgressV8 *progress);
int ck_session_v8_reset(CKSessionV8 *session);
const char *ck_session_v8_last_error(const CKSessionV8 *session);

/* Optional bounded two-slot decode mode. This mode owns the generated model
 * until close and is mutually exclusive with generate/reset. Both prompts are
 * prefilled independently using token IDs supplied by the caller. A step
 * advances each ready, uncancelled slot at most once; when both are eligible
 * it calls the generated two-row entry, otherwise ordinary single-row decode.
 * Cancellation is sampled at the next step boundary: a request made during a
 * native kernel can still allow that in-flight step to finish. On success,
 * advanced_mask identifies exactly which slots advanced. On a native error,
 * advancement is unknown and both slots are poisoned until individually reset.
 * All functions except request_cancel must
 * be serialized by the host; cancellation is thread-safe. This is a native
 * scheduling primitive, not an HTTP admission queue.
 * The legacy whole-prompt prefill is synchronous and has no ticket until it
 * returns. The staged decode-replay prefill below has a ticket before work.
 */
/* max_extra_bytes is a mandatory admission budget for two state arenas plus
 * the batch workspace. needed_bytes reports that requirement even when the
 * budget is too small; alignment padding is handled by the allocator. */
int ck_session_v8_batch2_enable(CKSessionV8 *session, size_t max_extra_bytes,
                                size_t *needed_bytes);
int ck_session_v8_batch2_prefill(CKSessionV8 *session, uint32_t slot,
                                const int32_t *tokens, int32_t count,
                                float *logits, uint64_t *ticket_out);
/* Staged prefill reserves a slot and publishes its ticket before model work.
 * Append replays at most count tokens through the generated single-token
 * decoder. The caller chooses a bounded count and may schedule another ready
 * slot between calls. Logits are scratch until the full prompt is consumed;
 * the slot becomes decode-eligible only after exactly total_tokens tokens.
 * Cancellation is observed between token calls, not inside a kernel. A
 * cancelled or failed partial prefill must be reset before slot reuse.
 * scratch_logits must point to vocab_size writable floats, remain valid for
 * the duration of the call, and not alias native model or sequence memory.
 * consumed_out is updated after each completed token; it is unchanged when
 * validation or a previously requested cancellation rejects the call.
 * Decode replay can differ numerically from the generated batched prefill
 * path; callers must certify their selected artifact before using this mode.
 * This ABI does not perform mixed prefill/decode projection batching. */
int ck_session_v8_batch2_prefill_begin(CKSessionV8 *session, uint32_t slot,
                                      int32_t total_tokens,
                                      uint64_t *ticket_out);
int ck_session_v8_batch2_prefill_append(CKSessionV8 *session, uint32_t slot,
                                       uint64_t ticket, const int32_t *tokens,
                                       int32_t count, float *scratch_logits,
                                       int32_t *consumed_out);
/* Opt-in map-resolved prefill arithmetic for a standalone staged row.
 * Requires an exported phase entry and ungrouped append contract; otherwise
 * returns CAPABILITY before changing either sequence. */
int ck_session_v8_batch2_prefill_append_phase(
    CKSessionV8 *session, uint32_t slot, uint64_t ticket,
    const int32_t *tokens, int32_t count, float *scratch_logits,
    int32_t *consumed_out);
/* Read-only loaded-model capability bits: 1 = standalone phase append,
 * 2 = phase-aware mixed row. Negative values are session errors. */
int ck_session_v8_batch2_phase_capabilities(CKSessionV8 *session);
int ck_session_v8_batch2_step(CKSessionV8 *session, const int32_t tokens[2],
                             float *const logits[2], uint32_t *advanced_mask);
/* Opt-in one-row mixed step: one ready decode slot and one staged prefill slot
 * enter the same generated batch2 call. prefill_slot identifies which token
 * comes from the staged prompt; tickets[slot] must match each live generation.
 * Both rows use their own sequence handle and current position. On success,
 * advanced_mask is 3 and consumed_out is the staged prompt's cumulative token
 * count. A pre-execution cancellation returns CANCELLED with no advancement.
 * A runtime error may leave advancement unknown and poisons both slots; reset
 * both before reuse. Cancellation during generated execution is acknowledged
 * only after that call returns, so successful work is reported before reset.
 * This is one prefill token plus one decode token, not a multi-row chunk. */
int ck_session_v8_batch2_mixed_step(CKSessionV8 *session,
                                    const int32_t tokens[2],
                                    const uint64_t tickets[2],
                                    uint32_t prefill_slot,
                                    float *const logits[2],
                                    uint32_t *advanced_mask,
                                    int32_t *consumed_out);
/* Separate opt-in semantics: the staged row uses the generated prefill-phase
 * local stages. An older model without the matching entry returns CAPABILITY
 * before either sequence advances. The decode-replay ABI above is unchanged. */
int ck_session_v8_batch2_mixed_step_prefill_phase(
    CKSessionV8 *session, const int32_t tokens[2],
    const uint64_t tickets[2], uint32_t prefill_slot,
    float *const logits[2], uint32_t *advanced_mask,
    int32_t *consumed_out);
int ck_session_v8_batch2_reset_slot(CKSessionV8 *session, uint32_t slot);
int ck_session_v8_batch2_position(CKSessionV8 *session, uint32_t slot,
                                 int32_t *position_out);
void ck_session_v8_batch2_request_cancel(CKSessionV8 *session, uint64_t ticket);

#ifdef __cplusplus
}
#endif

#endif
