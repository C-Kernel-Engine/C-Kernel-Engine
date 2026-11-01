"""Emit the serialized, planner-bounded sequence-switching ABI."""


def emit_sequence_state_api(contract: dict[str, int] | None = None) -> str:
    recurrent = bool(contract and "recurrent_conv_state_bytes" in contract)
    header = ""
    if recurrent:
        header = "\n".join(
            f"#define CK_SEQ_{name.upper()} {contract[key]}ULL"
            for name, key in (
                ("ARENA_BYTES", "arena_bytes"),
                ("CONV_OFFSET", "recurrent_conv_state_arena_offset"),
                ("CONV_BYTES", "recurrent_conv_state_bytes"),
                ("SSM_OFFSET", "recurrent_ssm_state_arena_offset"),
                ("SSM_BYTES", "recurrent_ssm_state_bytes"),
            )
        ) + "\n"
    return header + r'''
#ifndef CK_SEQ_ARENA_BYTES
#define CK_SEQ_ARENA_BYTES KV_CACHE_SIZE
#endif
#ifndef MAX_SEQ_LEN
#define MAX_SEQ_LEN INT32_MAX
#endif
/* Execution is serialized. Handles are generation-tagged IDs, never pointers. */
typedef struct CKSequenceStateV8 {
    uint8_t *kv;
#ifdef CK_SEQ_CONV_BYTES
    uint8_t *conv;
    uint8_t *ssm;
#endif
    int pos;
    int rope_pos;
    uint64_t generation;
    int occupied;
} CKSequenceStateV8;

/* One default state and two caller-backed extra states. The caller owns and
 * budgets each 64-byte-aligned state arena until destroy/model_free completes. */
static CKSequenceStateV8 g_sequence_states[3];
static CKSequenceStateV8 *g_active_sequence = NULL;
static uint64_t g_sequence_generation = 0;
static uint64_t g_sequence_model_epoch = 1;

/* Checkpoints are process-local and caller-owned. The host must additionally
 * bind them to the exact model artifact and token prefix before reuse. */
typedef struct CKSequenceCheckpointV8 {
    uint64_t magic;
    uint64_t model_epoch;
    uint64_t model_cookie;
    uint64_t state_bytes;
    uint64_t checksum;
    int32_t pos;
    int32_t rope_pos;
} CKSequenceCheckpointV8;

#define CK_SEQUENCE_CHECKPOINT_MAGIC UINT64_C(0x314b504351534b43)

static uint64_t ck_sequence_checkpoint_hash(const CKSequenceCheckpointV8 *head,
                                            const uint8_t *payload) {
    uint64_t hash = UINT64_C(14695981039346656037);
    const uint64_t fields[] = {
        head->magic, head->model_epoch, head->model_cookie,
        head->state_bytes, (uint64_t)(uint32_t)head->pos,
        (uint64_t)(uint32_t)head->rope_pos,
    };
    for (size_t i = 0; i < sizeof(fields) / sizeof(fields[0]); ++i) {
        uint64_t value = fields[i];
        for (unsigned byte = 0; byte < 8; ++byte) {
            hash = (hash ^ (uint8_t)value) * UINT64_C(1099511628211);
            value >>= 8;
        }
    }
    for (size_t i = 0; i < (size_t)head->state_bytes; ++i)
        hash = (hash ^ payload[i]) * UINT64_C(1099511628211);
    return hash;
}

static int ck_sequence_checkpoint_disjoint(const void *memory, size_t bytes) {
    uintptr_t start = (uintptr_t)memory;
    if (!memory || start > UINTPTR_MAX - bytes) return 0;
    uintptr_t end = start + bytes;
    uintptr_t descriptors = (uintptr_t)&g_sequence_states[0];
    if (start < descriptors + sizeof(g_sequence_states) &&
        descriptors < end) return 0;
    uintptr_t model_object = (uintptr_t)g_model;
    if (start < model_object + sizeof(*g_model) && model_object < end) return 0;
    uintptr_t model = (uintptr_t)g_model->bump;
    if (g_model->bump_size && model <= UINTPTR_MAX - g_model->bump_size &&
        start < model + g_model->bump_size && model < end) return 0;
    for (unsigned i = 1; i < 3; ++i) {
        const CKSequenceStateV8 *state = &g_sequence_states[i];
        if (!state->occupied) continue;
        uintptr_t arena = (uintptr_t)state->kv;
        if (arena > UINTPTR_MAX - (size_t)CK_SEQ_ARENA_BYTES ||
            (start < arena + (size_t)CK_SEQ_ARENA_BYTES && arena < end)) return 0;
    }
    return 1;
}

static uint64_t ck_sequence_new_id(unsigned slot) {
    if (g_sequence_generation >= (UINT64_MAX >> 2)) return 0;
    return (++g_sequence_generation << 2) | (uint64_t)(slot + 1);
}

static CKSequenceStateV8 *ck_sequence_find(uint64_t handle) {
    unsigned encoded = (unsigned)(handle & 3u);
    if (encoded < 1 || encoded > 3) return NULL;
    CKSequenceStateV8 *state = &g_sequence_states[encoded - 1];
    return state->occupied && state->generation == handle ? state : NULL;
}

static void ck_sequence_ensure_default(void) {
    if (g_sequence_states[0].occupied || !g_model) return;
    CKSequenceStateV8 *state = &g_sequence_states[0];
    state->kv = g_model->bump + A_KV_CACHE;
#ifdef CK_SEQ_CONV_BYTES
    state->conv = g_model->bump + A_RECURRENT_CONV_STATE;
    state->ssm = g_model->bump + A_RECURRENT_SSM_STATE;
#endif
    state->pos = g_model->pos;
    state->rope_pos = g_model->rope_pos;
    state->generation = ck_sequence_new_id(0);
    state->occupied = state->generation != 0;
    g_active_sequence = state->occupied ? state : NULL;
}

static void ck_sequence_release_all(void) {
    memset(g_sequence_states, 0, sizeof(g_sequence_states));
    g_active_sequence = NULL;
    if (g_sequence_model_epoch != UINT64_MAX) ++g_sequence_model_epoch;
}

CK_EXPORT int ck_model_sequence_state_requirements(size_t *bytes, size_t *alignment) {
    if (!bytes || !alignment || KV_CACHE_SIZE <= 0) return -1;
    *bytes = (size_t)CK_SEQ_ARENA_BYTES;
    *alignment = 64;
    return 0;
}

CK_EXPORT uint64_t ck_model_sequence_state_default(void) {
    if (!g_model) return 0;
    ck_sequence_ensure_default();
    return g_sequence_states[0].generation;
}

CK_EXPORT int ck_model_sequence_state_create(void *arena, size_t arena_bytes,
                                              uint64_t *out) {
    if (!g_model || !out) return -1;
    *out = 0;
    ck_sequence_ensure_default();
    if (!g_active_sequence) return -1;
    if (!arena || ((uintptr_t)arena & 63u) || arena_bytes < (size_t)CK_SEQ_ARENA_BYTES)
        return -2;
    uintptr_t start = (uintptr_t)arena;
    if (start > UINTPTR_MAX - (size_t)CK_SEQ_ARENA_BYTES) return -2;
    uintptr_t end = start + (size_t)CK_SEQ_ARENA_BYTES;
    uintptr_t model_start = (uintptr_t)g_model->bump;
    if (g_model->bump_size && model_start <= UINTPTR_MAX - g_model->bump_size &&
        start < model_start + g_model->bump_size && model_start < end) return -2;
    for (unsigned i = 0; i < 3; ++i) {
        CKSequenceStateV8 *existing = &g_sequence_states[i];
        if (!existing->occupied) continue;
        uintptr_t other = (uintptr_t)existing->kv;
        size_t other_bytes = i == 0 ? (size_t)KV_CACHE_SIZE : (size_t)CK_SEQ_ARENA_BYTES;
        if (other <= UINTPTR_MAX - other_bytes &&
            start < other + other_bytes && other < end) return -2;
#ifdef CK_SEQ_CONV_BYTES
        if (i == 0) {
            uintptr_t conv = (uintptr_t)existing->conv;
            uintptr_t ssm = (uintptr_t)existing->ssm;
            if ((conv <= UINTPTR_MAX - (size_t)CK_SEQ_CONV_BYTES &&
                 start < conv + (size_t)CK_SEQ_CONV_BYTES && conv < end) ||
                (ssm <= UINTPTR_MAX - (size_t)CK_SEQ_SSM_BYTES &&
                 start < ssm + (size_t)CK_SEQ_SSM_BYTES && ssm < end)) return -2;
        }
#endif
    }
    for (unsigned i = 1; i < 3; ++i) {
        CKSequenceStateV8 *state = &g_sequence_states[i];
        if (state->occupied) continue;
        uint64_t id = ck_sequence_new_id(i);
        if (!id) return -3;
        memset(arena, 0, (size_t)CK_SEQ_ARENA_BYTES);
        state->kv = (uint8_t *)arena;
#ifdef CK_SEQ_CONV_BYTES
        state->conv = (uint8_t *)arena + CK_SEQ_CONV_OFFSET;
        state->ssm = (uint8_t *)arena + CK_SEQ_SSM_OFFSET;
#endif
        state->pos = 0;
        state->rope_pos = 0;
        state->generation = id;
        state->occupied = 1;
        *out = id;
        return 0;
    }
    return -3;
}

CK_EXPORT int ck_model_sequence_state_activate(uint64_t handle) {
    if (!g_model) return -1;
    ck_sequence_ensure_default();
    CKSequenceStateV8 *state = ck_sequence_find(handle);
    if (!state) return -1;
    if (state == g_active_sequence) return 0;
    if (g_active_sequence) {
        g_active_sequence->pos = g_model->pos;
        g_active_sequence->rope_pos = g_model->rope_pos;
    }
    g_model->kv_cache = (float *)state->kv;
    g_model->kv_cache_f16 = (uint16_t *)state->kv;
#ifdef CK_SEQ_CONV_BYTES
    g_model->recurrent_conv_state = (float *)state->conv;
    g_model->recurrent_ssm_state = (float *)state->ssm;
#endif
    g_model->pos = state->pos;
    g_model->rope_pos = state->rope_pos;
    g_active_sequence = state;
    return 0;
}

CK_EXPORT int ck_model_sequence_state_destroy(uint64_t handle) {
    if (!g_model) return -1;
    ck_sequence_ensure_default();
    CKSequenceStateV8 *state = ck_sequence_find(handle);
    if (!state || state == &g_sequence_states[0]) return -1;
    if (state == g_active_sequence &&
        ck_model_sequence_state_activate(g_sequence_states[0].generation) != 0) return -1;
    memset(state, 0, sizeof(*state));
    return 0;
}

CK_EXPORT int ck_model_sequence_checkpoint_requirements(size_t *bytes,
                                                        size_t *alignment) {
    if (!bytes || !alignment || !g_model ||
        CK_SEQ_ARENA_BYTES > SIZE_MAX - sizeof(CKSequenceCheckpointV8))
        return -1;
    *bytes = sizeof(CKSequenceCheckpointV8) + (size_t)CK_SEQ_ARENA_BYTES;
    *alignment = 64;
    return 0;
}

CK_EXPORT int ck_model_sequence_checkpoint_save(uint64_t handle, void *out,
                                               size_t out_bytes) {
    size_t needed = 0, alignment = 0;
    if (ck_model_sequence_checkpoint_requirements(&needed, &alignment) != 0 ||
        !out || ((uintptr_t)out & (alignment - 1)) || out_bytes < needed ||
        !ck_sequence_checkpoint_disjoint(out, needed)) return -2;
    CKSequenceStateV8 *state = ck_sequence_find(handle);
    if (!state) return -1;
    const int pos = state == g_active_sequence ? g_model->pos : state->pos;
    const int rope_pos = state == g_active_sequence ? g_model->rope_pos : state->rope_pos;
    if (pos < 0 || pos > MAX_SEQ_LEN || rope_pos < 0) return -2;
    CKSequenceCheckpointV8 head = {
        .magic = CK_SEQUENCE_CHECKPOINT_MAGIC,
        .model_epoch = g_sequence_model_epoch,
        .model_cookie = (uint64_t)((uintptr_t)g_model ^ (uintptr_t)&g_sequence_states),
        .state_bytes = (uint64_t)CK_SEQ_ARENA_BYTES,
        .pos = pos, .rope_pos = rope_pos,
    };
    uint8_t *payload = (uint8_t *)out + sizeof(head);
    memset(payload, 0, (size_t)CK_SEQ_ARENA_BYTES);
    memcpy(payload, state->kv, (size_t)KV_CACHE_SIZE);
#ifdef CK_SEQ_CONV_BYTES
    memcpy(payload + CK_SEQ_CONV_OFFSET, state->conv, (size_t)CK_SEQ_CONV_BYTES);
    memcpy(payload + CK_SEQ_SSM_OFFSET, state->ssm, (size_t)CK_SEQ_SSM_BYTES);
#endif
    head.checksum = ck_sequence_checkpoint_hash(&head, payload);
    memcpy(out, &head, sizeof(head));
    return 0;
}

CK_EXPORT int ck_model_sequence_checkpoint_restore(uint64_t handle,
                                                  const void *snapshot,
                                                  size_t snapshot_bytes) {
    size_t needed = 0, alignment = 0;
    if (ck_model_sequence_checkpoint_requirements(&needed, &alignment) != 0 ||
        !snapshot || ((uintptr_t)snapshot & (alignment - 1)) ||
        snapshot_bytes != needed ||
        !ck_sequence_checkpoint_disjoint(snapshot, needed)) return -2;
    CKSequenceStateV8 *state = ck_sequence_find(handle);
    if (!state) return -1;
    CKSequenceCheckpointV8 head;
    memcpy(&head, snapshot, sizeof(head));
    const uint8_t *payload = (const uint8_t *)snapshot + sizeof(head);
    if (head.magic != CK_SEQUENCE_CHECKPOINT_MAGIC ||
        head.model_epoch != g_sequence_model_epoch ||
        head.model_cookie !=
            (uint64_t)((uintptr_t)g_model ^ (uintptr_t)&g_sequence_states) ||
        head.state_bytes != (uint64_t)CK_SEQ_ARENA_BYTES ||
        head.pos < 0 || head.pos > MAX_SEQ_LEN || head.rope_pos < 0 ||
        head.checksum != ck_sequence_checkpoint_hash(&head, payload)) return -2;
    memcpy(state->kv, payload, (size_t)KV_CACHE_SIZE);
#ifdef CK_SEQ_CONV_BYTES
    memcpy(state->conv, payload + CK_SEQ_CONV_OFFSET, (size_t)CK_SEQ_CONV_BYTES);
    memcpy(state->ssm, payload + CK_SEQ_SSM_OFFSET, (size_t)CK_SEQ_SSM_BYTES);
#endif
    state->pos = head.pos;
    state->rope_pos = head.rope_pos;
    if (state == g_active_sequence) {
        g_model->pos = head.pos;
        g_model->rope_pos = head.rope_pos;
    }
    return 0;
}
'''
