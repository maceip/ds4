/* Qwen4Exp 512-expert / top-10 router primitives.
 *
 * The model routes on raw FP32 logits, then applies a precise softmax only to
 * the selected ten entries.  This is intentionally separate from DS4's
 * DeepSeek router, whose sqrt(softplus(logit)) transform is not equivalent.
 * One 512-thread threadgroup owns each token row, so selection has no global
 * scratch and batched prefill uses the same ordering as single-token decode.
 */

constant uint QWEN4_EXP_MOE_EXPERTS = 512;
constant uint QWEN4_EXP_MOE_TOP_K = 10;

struct qwen4_exp_moe_router_args {
    uint n_tokens;
    uint pad0;
    uint pad1;
    uint pad2;
};

static bool qwen4_exp_moe_less(float a_score, int a_id,
                               float b_score, int b_id) {
    return a_score < b_score || (a_score == b_score && a_id < b_id);
}

kernel void kernel_qwen4_exp_moe_router_top10(
        constant qwen4_exp_moe_router_args &args [[buffer(0)]],
        device const float                 *logits [[buffer(1)]],
        device int                         *selected [[buffer(2)]],
        device float                       *weights [[buffer(3)]],
        threadgroup float                  *scores [[threadgroup(0)]],
        threadgroup int                    *indices [[threadgroup(1)]],
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    if (row >= args.n_tokens) return;

    float score = logits[ulong(row) * QWEN4_EXP_MOE_EXPERTS + uint(tid)];
    /* A model-produced router row must be finite. Keep malformed NaNs out of
     * the chosen set deterministically instead of poisoning the sort. */
    if (isnan(score)) score = -INFINITY;
    scores[tid] = score;
    indices[tid] = int(tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Stable ascending bitonic sort by (logit, expert id). Reading the final
     * ten entries in reverse produces MLX argpartition(-logits)'s observed
     * descending Qwen route order. */
    for (uint width = 2u; width <= QWEN4_EXP_MOE_EXPERTS; width <<= 1u) {
        for (uint stride = width >> 1u; stride > 0u; stride >>= 1u) {
            const uint peer = uint(tid) ^ stride;
            if (peer > uint(tid)) {
                const float a_score = scores[tid];
                const int a_id = indices[tid];
                const float b_score = scores[peer];
                const int b_id = indices[peer];
                const bool ascending = (uint(tid) & width) == 0u;
                const bool a_less_b = qwen4_exp_moe_less(
                    a_score, a_id, b_score, b_id);
                const bool swap = ascending ? !a_less_b : a_less_b;
                if (swap && (a_score != b_score || a_id != b_id)) {
                    scores[tid] = b_score;
                    indices[tid] = b_id;
                    scores[peer] = a_score;
                    indices[peer] = a_id;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (tid < QWEN4_EXP_MOE_TOP_K) {
        const uint source = QWEN4_EXP_MOE_EXPERTS - 1u - uint(tid);
        const float shifted = scores[source] - scores[QWEN4_EXP_MOE_EXPERTS - 1u];
        indices[tid] = indices[source];
        scores[tid] = precise::exp(shifted);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float denominator = 0.0f;
        for (uint slot = 0; slot < QWEN4_EXP_MOE_TOP_K; slot++) {
            denominator += scores[slot];
        }
        scores[QWEN4_EXP_MOE_TOP_K] = denominator;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < QWEN4_EXP_MOE_TOP_K) {
        const ulong output = ulong(row) * QWEN4_EXP_MOE_TOP_K + uint(tid);
        selected[output] = indices[tid];
        weights[output] = scores[tid] / scores[QWEN4_EXP_MOE_TOP_K];
    }
}

/* The generic resident routed-MoE path emits one F32 row per selected expert.
 * Qwen needs ten rows instead of the existing DeepSeek six / GLM eight. Keep
 * this final reduction in one dispatch and preserve left-to-right slot order. */
kernel void kernel_qwen4_exp_moe_sum10_f32(
        constant ds4_metal_dsv4_moe_sum6_args &args [[buffer(0)]],
        device const char *src [[buffer(1)]],
        device       char *dst [[buffer(2)]],
        uint token [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if (token >= args.tokens) return;

    device const float *s =
        (device const float *)(src + ulong(token) * args.src_token_stride);
    device float *d =
        (device float *)(dst + ulong(token) * args.dst_token_stride);

    for (uint col = tid; col < args.width; col += ntg) {
        float value = s[col];
        for (uint slot = 1u; slot < QWEN4_EXP_MOE_TOP_K; slot++) {
            value += s[slot * args.width + col];
        }
        d[col] = value;
    }
}
