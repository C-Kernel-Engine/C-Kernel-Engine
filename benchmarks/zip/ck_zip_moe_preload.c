#define _GNU_SOURCE

#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

/*
 * Research-only coordinator shim for two-rank ZIP MoE prefill.
 *
 * Numerical kernels remain pure. The coordinator sends the worker complete
 * token rows, computes its own rows locally, and receives the worker's final
 * routed-plus-shared SwiGLU rows at the first necessary layer boundary.
 */

#define CK_ZIP_MAGIC UINT32_C(0x434b5a50)
#define CK_ZIP_VERSION UINT32_C(1)
#define CK_ZIP_REQUEST UINT32_C(1)
#define CK_ZIP_RESPONSE UINT32_C(2)
#define CK_ZIP_DEFAULT_PORT 29535
#define CK_ZIP_DEFAULT_LOCAL_PERCENT 69

typedef int (*ck_routed_fn)(
    const float *, const int *, const float *, const void *, const void *,
    const void *, float *, int, int, int, int, int, void *, size_t);

typedef int (*ck_shared_fn)(
    const float *, const float *, const void *, const void *, const void *,
    const float *, float *, int, int, int, void *, size_t);

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t sequence;
    uint32_t total_rows;
    uint32_t hidden_dim;
    uint32_t intermediate_dim;
    uint32_t n_experts;
    uint32_t top_k;
    uint32_t remote_begin;
    uint32_t remote_rows;
    uint32_t kind;
    uint32_t reserved;
    uint64_t hidden_bytes;
    uint64_t index_bytes;
    uint64_t routing_bytes;
    uint64_t output_bytes;
} ck_zip_header_t;

_Static_assert(sizeof(ck_zip_header_t) == 80,
               "ZIP worker protocol header must remain 80 bytes");
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "The research ZIP protocol currently requires little-endian ranks"
#endif

typedef struct {
    int enabled;
    int port;
    int local_percent;
    int socket_fd;
    int request_pending;
    uint32_t sequence;
    uint64_t routed_calls;
    uint64_t shared_calls;
    uint64_t bytes_sent;
    uint64_t bytes_received;
    uint64_t routed_ns;
    uint64_t shared_ns;
    uint64_t request_send_ns;
    uint64_t response_wait_ns;
    char host[64];
    char report_path[512];
} ck_zip_state_t;

static ck_zip_state_t ck_zip_state = {
    .port = CK_ZIP_DEFAULT_PORT,
    .local_percent = CK_ZIP_DEFAULT_LOCAL_PERCENT,
    .socket_fd = -1,
};
static pthread_once_t ck_zip_once = PTHREAD_ONCE_INIT;
static ck_routed_fn ck_real_routed_parallel = NULL;
static ck_routed_fn ck_real_routed_prepared = NULL;
static ck_shared_fn ck_real_shared = NULL;

static uint64_t ck_zip_now_ns(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

static int ck_zip_parse_int(const char *value, int fallback)
{
    if (!value || !*value) return fallback;
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    if (!end || *end != '\0' || parsed < 0 || parsed > 1000000) return fallback;
    return (int)parsed;
}

static void ck_zip_initialize(void)
{
    const char *role = getenv("CK_ZIP_RESEARCH_ROLE");
    if (!role || strcmp(role, "coordinator") != 0) return;

    ck_zip_state.port = ck_zip_parse_int(
        getenv("CK_ZIP_RESEARCH_PORT"), CK_ZIP_DEFAULT_PORT);
    ck_zip_state.local_percent = ck_zip_parse_int(
        getenv("CK_ZIP_RESEARCH_LOCAL_PERCENT"),
        CK_ZIP_DEFAULT_LOCAL_PERCENT);
    if (ck_zip_state.port <= 0 || ck_zip_state.port > 65535 ||
        ck_zip_state.local_percent <= 0 ||
        ck_zip_state.local_percent >= 100) {
        fprintf(stderr, "ck_zip: invalid port or local row percentage\n");
        return;
    }

    const char *host = getenv("CK_ZIP_RESEARCH_HOST");
    if (!host || !*host) host = "127.0.0.1";
    if (snprintf(ck_zip_state.host, sizeof(ck_zip_state.host), "%s", host) >=
        (int)sizeof(ck_zip_state.host)) {
        fprintf(stderr, "ck_zip: worker address is too long\n");
        return;
    }
    const char *report = getenv("CK_ZIP_RESEARCH_REPORT");
    if (report && *report &&
        snprintf(ck_zip_state.report_path, sizeof(ck_zip_state.report_path),
                 "%s", report) >= (int)sizeof(ck_zip_state.report_path)) {
        fprintf(stderr, "ck_zip: report path is too long\n");
        return;
    }

    ck_real_routed_parallel = (ck_routed_fn)dlsym(
        RTLD_NEXT, "moe_swiglu_expert_forward_q4k_q5k_parallel_workspace");
    ck_real_routed_prepared = (ck_routed_fn)dlsym(
        RTLD_NEXT,
        "moe_swiglu_expert_forward_q4k_q5k_auto_prepared_workspace");
    ck_real_shared = (ck_shared_fn)dlsym(
        RTLD_NEXT, "moe_swiglu_shared_forward_q8_0_gated_workspace");
    if ((!ck_real_routed_parallel && !ck_real_routed_prepared) ||
        !ck_real_shared) {
        fprintf(stderr, "ck_zip: failed to resolve production MoE providers: %s\n",
                dlerror());
        return;
    }
    ck_zip_state.enabled = 1;
}

static void ck_zip_fail(const char *operation)
{
    fprintf(stderr, "ck_zip: %s failed: %s\n", operation, strerror(errno));
    fflush(stderr);
    _exit(90);
}

static void ck_zip_open_connection(void)
{
    if (ck_zip_state.socket_fd >= 0) return;
    struct sockaddr_in address = {
        .sin_family = AF_INET,
        .sin_port = htons((uint16_t)ck_zip_state.port),
    };
    if (inet_pton(AF_INET, ck_zip_state.host, &address.sin_addr) != 1) {
        errno = EINVAL;
        ck_zip_fail("inet_pton");
    }
    for (int attempt = 0; attempt < 600; ++attempt) {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) ck_zip_fail("socket");
        if (connect(fd, (struct sockaddr *)&address, sizeof(address)) == 0) {
            ck_zip_state.socket_fd = fd;
            break;
        }
        close(fd);
        struct timespec delay = {.tv_sec = 0, .tv_nsec = 100000000};
        nanosleep(&delay, NULL);
    }
    if (ck_zip_state.socket_fd < 0) {
        errno = ETIMEDOUT;
        ck_zip_fail("connect");
    }
    int one = 1;
    (void)setsockopt(ck_zip_state.socket_fd, IPPROTO_TCP, TCP_NODELAY,
                     &one, sizeof(one));
    (void)setsockopt(ck_zip_state.socket_fd, SOL_SOCKET, SO_KEEPALIVE,
                     &one, sizeof(one));
}

static void ck_zip_send_all(const void *data, size_t bytes)
{
    const uint8_t *cursor = (const uint8_t *)data;
    while (bytes > 0) {
        ssize_t sent = send(ck_zip_state.socket_fd, cursor, bytes,
                            MSG_NOSIGNAL);
        if (sent < 0) {
            if (errno == EINTR) continue;
            ck_zip_fail("send");
        }
        if (sent == 0) {
            errno = EPIPE;
            ck_zip_fail("send-zero");
        }
        cursor += (size_t)sent;
        bytes -= (size_t)sent;
        ck_zip_state.bytes_sent += (uint64_t)sent;
    }
}

static void ck_zip_receive_all(void *data, size_t bytes)
{
    uint8_t *cursor = (uint8_t *)data;
    while (bytes > 0) {
        ssize_t received = recv(ck_zip_state.socket_fd, cursor, bytes, 0);
        if (received < 0) {
            if (errno == EINTR) continue;
            ck_zip_fail("recv");
        }
        if (received == 0) {
            errno = ECONNRESET;
            ck_zip_fail("recv-eof");
        }
        cursor += (size_t)received;
        bytes -= (size_t)received;
        ck_zip_state.bytes_received += (uint64_t)received;
    }
}

static int ck_zip_local_rows(int rows)
{
    int local = (rows * ck_zip_state.local_percent + 50) / 100;
    if (local < 1) local = 1;
    if (local >= rows) local = rows - 1;
    return local;
}

static void ck_zip_send_request(const float *hidden, const int *indices,
                                const float *routing_weights, int rows,
                                int hidden_dim, int intermediate_dim,
                                int n_experts, int top_k)
{
    ck_zip_open_connection();
    const int remote_begin = ck_zip_local_rows(rows);
    const int remote_rows = rows - remote_begin;
    const size_t hidden_bytes =
        (size_t)remote_rows * (size_t)hidden_dim * sizeof(float);
    const size_t index_bytes =
        (size_t)remote_rows * (size_t)top_k * sizeof(int);
    const size_t routing_bytes =
        (size_t)remote_rows * (size_t)top_k * sizeof(float);
    ck_zip_header_t header = {
        .magic = CK_ZIP_MAGIC,
        .version = CK_ZIP_VERSION,
        .sequence = ck_zip_state.sequence,
        .total_rows = (uint32_t)rows,
        .hidden_dim = (uint32_t)hidden_dim,
        .intermediate_dim = (uint32_t)intermediate_dim,
        .n_experts = (uint32_t)n_experts,
        .top_k = (uint32_t)top_k,
        .remote_begin = (uint32_t)remote_begin,
        .remote_rows = (uint32_t)remote_rows,
        .kind = CK_ZIP_REQUEST,
        .hidden_bytes = (uint64_t)hidden_bytes,
        .index_bytes = (uint64_t)index_bytes,
        .routing_bytes = (uint64_t)routing_bytes,
        .output_bytes = (uint64_t)hidden_bytes,
    };
    ck_zip_send_all(&header, sizeof(header));
    ck_zip_send_all(hidden + (size_t)remote_begin * (size_t)hidden_dim,
                    hidden_bytes);
    ck_zip_send_all(indices + (size_t)remote_begin * (size_t)top_k,
                    index_bytes);
    ck_zip_send_all(
        routing_weights + (size_t)remote_begin * (size_t)top_k,
        routing_bytes);
    ck_zip_state.request_pending = 1;
}

static void ck_zip_receive_response(float *output, int rows, int hidden_dim)
{
    if (!ck_zip_state.request_pending) {
        fprintf(stderr, "ck_zip: shared provider reached without routed request\n");
        fflush(stderr);
        _exit(91);
    }
    const int remote_begin = ck_zip_local_rows(rows);
    const int remote_rows = rows - remote_begin;
    const size_t output_bytes =
        (size_t)remote_rows * (size_t)hidden_dim * sizeof(float);
    ck_zip_header_t header;
    ck_zip_receive_all(&header, sizeof(header));
    if (header.magic != CK_ZIP_MAGIC || header.version != CK_ZIP_VERSION ||
        header.sequence != ck_zip_state.sequence ||
        header.total_rows != (uint32_t)rows ||
        header.hidden_dim != (uint32_t)hidden_dim ||
        header.remote_begin != (uint32_t)remote_begin ||
        header.remote_rows != (uint32_t)remote_rows ||
        header.kind != CK_ZIP_RESPONSE ||
        header.output_bytes != (uint64_t)output_bytes) {
        fprintf(stderr, "ck_zip: response contract mismatch at layer %u\n",
                ck_zip_state.sequence);
        fflush(stderr);
        _exit(92);
    }
    ck_zip_receive_all(output + (size_t)remote_begin * (size_t)hidden_dim,
                       output_bytes);
    ck_zip_state.sequence += 1;
    ck_zip_state.request_pending = 0;
}

static int ck_zip_routed_workspace(
    ck_routed_fn *real_routed, const char *symbol,
    const float *hidden, const int *indices, const float *routing_weights,
    const void *expert_gate, const void *expert_up, const void *expert_down,
    float *output, int rows, int hidden_dim, int intermediate_dim,
    int n_experts, int top_k, void *workspace, size_t workspace_bytes)
{
    pthread_once(&ck_zip_once, ck_zip_initialize);
    if (!*real_routed) {
        *real_routed = (ck_routed_fn)dlsym(RTLD_NEXT, symbol);
    }
    if (!*real_routed) {
        fprintf(stderr, "ck_zip: failed to resolve routed provider %s: %s\n",
                symbol, dlerror());
        return -1;
    }
    if (!ck_zip_state.enabled || rows <= 1) {
        return (*real_routed)(
            hidden, indices, routing_weights, expert_gate, expert_up,
            expert_down, output, rows, hidden_dim, intermediate_dim, n_experts,
            top_k, workspace, workspace_bytes);
    }

    memset(output, 0, (size_t)rows * (size_t)hidden_dim * sizeof(float));
    const uint64_t send_started = ck_zip_now_ns();
    ck_zip_send_request(hidden, indices, routing_weights, rows, hidden_dim,
                        intermediate_dim, n_experts, top_k);
    ck_zip_state.request_send_ns += ck_zip_now_ns() - send_started;

    const int local_rows = ck_zip_local_rows(rows);
    const uint64_t compute_started = ck_zip_now_ns();
    const int status = (*real_routed)(
        hidden, indices, routing_weights, expert_gate, expert_up, expert_down,
        output, local_rows, hidden_dim, intermediate_dim, n_experts, top_k,
        workspace, workspace_bytes);
    ck_zip_state.routed_ns += ck_zip_now_ns() - compute_started;
    ck_zip_state.routed_calls += 1;
    return status;
}

int moe_swiglu_expert_forward_q4k_q5k_parallel_workspace(
    const float *hidden, const int *indices, const float *routing_weights,
    const void *expert_gate, const void *expert_up, const void *expert_down,
    float *output, int rows, int hidden_dim, int intermediate_dim,
    int n_experts, int top_k, void *workspace, size_t workspace_bytes)
{
    return ck_zip_routed_workspace(
        &ck_real_routed_parallel,
        "moe_swiglu_expert_forward_q4k_q5k_parallel_workspace", hidden,
        indices, routing_weights, expert_gate, expert_up, expert_down, output,
        rows, hidden_dim, intermediate_dim, n_experts, top_k, workspace,
        workspace_bytes);
}

int moe_swiglu_expert_forward_q4k_q5k_auto_prepared_workspace(
    const float *hidden, const int *indices, const float *routing_weights,
    const void *expert_gate, const void *expert_up, const void *expert_down,
    float *output, int rows, int hidden_dim, int intermediate_dim,
    int n_experts, int top_k, void *workspace, size_t workspace_bytes)
{
    return ck_zip_routed_workspace(
        &ck_real_routed_prepared,
        "moe_swiglu_expert_forward_q4k_q5k_auto_prepared_workspace", hidden,
        indices, routing_weights, expert_gate, expert_up, expert_down, output,
        rows, hidden_dim, intermediate_dim, n_experts, top_k, workspace,
        workspace_bytes);
}

int moe_swiglu_shared_forward_q8_0_gated_workspace(
    const float *hidden, const float *routed, const void *shared_gate,
    const void *shared_up, const void *shared_down,
    const float *shared_gate_input, float *output, int rows, int hidden_dim,
    int intermediate_dim, void *workspace, size_t workspace_bytes)
{
    pthread_once(&ck_zip_once, ck_zip_initialize);
    if (!ck_zip_state.enabled || rows <= 1) {
        if (!ck_real_shared) {
            ck_real_shared = (ck_shared_fn)dlsym(
                RTLD_NEXT, "moe_swiglu_shared_forward_q8_0_gated_workspace");
        }
        return ck_real_shared(
            hidden, routed, shared_gate, shared_up, shared_down,
            shared_gate_input, output, rows, hidden_dim, intermediate_dim,
            workspace, workspace_bytes);
    }

    const int local_rows = ck_zip_local_rows(rows);
    const uint64_t compute_started = ck_zip_now_ns();
    const int status = ck_real_shared(
        hidden, routed, shared_gate, shared_up, shared_down,
        shared_gate_input, output, local_rows, hidden_dim, intermediate_dim,
        workspace, workspace_bytes);
    ck_zip_state.shared_ns += ck_zip_now_ns() - compute_started;
    ck_zip_state.shared_calls += 1;
    if (status != 0) return status;

    const uint64_t wait_started = ck_zip_now_ns();
    ck_zip_receive_response(output, rows, hidden_dim);
    ck_zip_state.response_wait_ns += ck_zip_now_ns() - wait_started;
    return 0;
}

static void ck_zip_write_report(void) __attribute__((destructor));

static void ck_zip_write_report(void)
{
    if (!ck_zip_state.enabled) return;
    if (ck_zip_state.socket_fd >= 0) close(ck_zip_state.socket_fd);
    if (!ck_zip_state.report_path[0]) return;
    FILE *report = fopen(ck_zip_state.report_path, "w");
    if (!report) return;
    fprintf(report,
            "{\n"
            "  \"schema_version\": 1,\n"
            "  \"role\": \"coordinator\",\n"
            "  \"local_percent\": %d,\n"
            "  \"routed_calls\": %llu,\n"
            "  \"shared_calls\": %llu,\n"
            "  \"bytes_sent\": %llu,\n"
            "  \"bytes_received\": %llu,\n"
            "  \"routed_ms\": %.6f,\n"
            "  \"shared_ms\": %.6f,\n"
            "  \"request_send_ms\": %.6f,\n"
            "  \"response_wait_ms\": %.6f\n"
            "}\n",
            ck_zip_state.local_percent,
            (unsigned long long)ck_zip_state.routed_calls,
            (unsigned long long)ck_zip_state.shared_calls,
            (unsigned long long)ck_zip_state.bytes_sent,
            (unsigned long long)ck_zip_state.bytes_received,
            (double)ck_zip_state.routed_ns / 1.0e6,
            (double)ck_zip_state.shared_ns / 1.0e6,
            (double)ck_zip_state.request_send_ns / 1.0e6,
            (double)ck_zip_state.response_wait_ns / 1.0e6);
    fclose(report);
}
