#include "ds4.h"
#include "ds4_gpu.h"

#include <stdio.h>

int main(void) {
    if (!ds4_gpu_init()) {
        fprintf(stderr, "qwen4-exp-mtp-frontier: Metal initialization failed\n");
        return 1;
    }

    const int rc = ds4_test_qwen4_exp_frontier_roundtrip();
    ds4_gpu_cleanup();
    if (rc != 0) {
        fprintf(stderr,
                "qwen4-exp-mtp-frontier: recurrent snapshot/restore failed\n");
        return 1;
    }

    printf("qwen4-exp-mtp-frontier: 36 GDN states + PLE + QSA counters "
           "round-trip byte-exact; unsafe prefix/missing-shadow paths refused\n");
    return 0;
}
