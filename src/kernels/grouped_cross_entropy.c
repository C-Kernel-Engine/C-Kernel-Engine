/* Mean per-group cross entropy and its logits VJP. No allocations or model state. */
#include "checked_float_buffer.h"
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

int ck_grouped_cross_entropy_f32(
    const float *logits, size_t logits_elements,
    const int32_t *counts, size_t count_elements,
    const int32_t *targets, size_t target_elements,
    float *d_logits, size_t gradient_elements,
    float *loss, size_t loss_elements,
    float *scratch, size_t scratch_elements,
    size_t groups, size_t options) {
    if (!groups || !options || groups > options || options == SIZE_MAX ||
        options > SIZE_MAX / sizeof(float) - 1 || logits_elements < options ||
        gradient_elements < options || count_elements < groups || target_elements < groups ||
        !loss_elements || scratch_elements < options + 1) return -2;
    const void *ptrs[] = {logits, counts, targets, d_logits, loss, scratch};
    size_t sizes[] = {options, groups, groups, options, 1, options+1};
    for (size_t i=0; i<6; ++i) if (!ck_checked_region(ptrs[i], sizes[i])) return -1;
    for (size_t i=3; i<6; ++i) for (size_t j=0; j<i; ++j)
        if (ck_checked_overlap(ptrs[i], sizes[i], ptrs[j], sizes[j])) return -1;
    size_t total=0;
    for (size_t g=0; g<groups; ++g) {
        if (counts[g]<=0 || (size_t)counts[g]>options-total || targets[g]<0 || targets[g]>=counts[g]) return -2;
        total+=(size_t)counts[g];
    }
    if (total!=options || !ck_checked_finite(logits,options)) return -3;
    size_t start=0; double accumulated_loss=0;
    for (size_t g=0; g<groups; ++g) {
        size_t n=(size_t)counts[g]; double maximum=logits[start];
        for (size_t i=1; i<n; ++i) if ((double)logits[start+i]>maximum) maximum=logits[start+i];
        double sum=0;
        for (size_t i=0; i<n; ++i) sum+=exp((double)logits[start+i]-maximum);
        /* Subtract before adding log(sum), avoiding cancellation for large offsets. */
        double group_loss=(maximum-(double)logits[start+(size_t)targets[g]])+log(sum);
        accumulated_loss+=group_loss/(double)groups;
        if (!isfinite(accumulated_loss) || accumulated_loss>FLT_MAX) return -4;
        for (size_t i=0; i<n; ++i) {
            double p=exp((double)logits[start+i]-maximum)/sum;
            double gradient=(p-(i==(size_t)targets[g] ? 1.0 : 0.0))/(double)groups;
            if (!isfinite(gradient) || fabs(gradient)>FLT_MAX) return -4;
            scratch[start+i]=(float)gradient;
        }
        start+=n;
    }
    scratch[options]=(float)accumulated_loss;
    memcpy(d_logits,scratch,options*sizeof(float));
    loss[0]=scratch[options];
    return 0;
}
