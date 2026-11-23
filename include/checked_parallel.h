#ifndef CK_CHECKED_PARALLEL_H
#define CK_CHECKED_PARALLEL_H

#include "ck_threadpool.h"
#include "ckernel_engine.h"

#include <limits.h>

typedef void (*ck_checked_range_fn)(size_t begin, size_t end, void *args);

struct ck_checked_range {
    ck_checked_range_fn fn;
    void *args;
    size_t offset;
};

static inline void ck_checked_pool_range(int begin, int end, void *data)
{
    struct ck_checked_range *range = data;
    range->fn(range->offset + (size_t)begin, range->offset + (size_t)end, range->args);
}

/* Each range owns disjoint output and scratch. No reduction crosses ranges. */
static inline void ck_checked_parallel_for(size_t count, int grain, ck_checked_range_fn fn, void *args)
{
    if (count <= (size_t)grain) {
        fn(0, count, args);
        return;
    }
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_in_dispatch(pool)) {
        fn(0, count, args);
        return;
    }
    int threads = ck_threadpool_n_threads(pool);
    const int requested = ck_get_num_threads();
    if (threads > requested) threads = requested;
    const size_t ranges = count / (size_t)grain + (count % (size_t)grain != 0);
    if (ranges < (size_t)threads) threads = (int)ranges;
    struct ck_checked_range range = {fn, args, 0};
    const int limit = INT_MAX - grain * (CK_THREADPOOL_MAX_THREADS + 1);
    while (range.offset < count) {
        const size_t remaining = count - range.offset;
        const int extent = remaining > (size_t)limit ? limit : (int)remaining;
        ck_threadpool_parallel_for_n(pool, threads, 0, extent, grain, ck_checked_pool_range, &range);
        range.offset += (size_t)extent;
    }
}

#endif
