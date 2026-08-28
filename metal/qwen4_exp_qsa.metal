/* Qwen4Exp query-selected attention (QSA).
 *
 * Completed four-token indexer blocks are pooled independently from the
 * incomplete tail.  A decode query attends exactly the selected complete
 * blocks plus positions [4*floor((pos+1)/4), pos].  Keeping that tail derived
 * from the committed decode position is the causal boundary required by MTP
 * rejection/rollback.
 */

constant uint QWEN4_EXP_QSA_INDEX_HEADS = 4;
constant uint QWEN4_EXP_QSA_INDEX_DIM = 128;
constant uint QWEN4_EXP_QSA_BLOCK = 4;
constant uint QWEN4_EXP_QSA_HEADS = 24;
constant uint QWEN4_EXP_QSA_KV_HEADS = 2;
constant uint QWEN4_EXP_QSA_HEAD_DIM = 256;

struct qwen4_exp_qsa_rope_args {
    uint n_ctx_orig;
    uint n_rot;
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
};

struct qwen4_exp_qsa_pool_args {
    uint block_index;
    uint cache_f16;
    float eps;
    uint pad0;
    qwen4_exp_qsa_rope_args rope;
};

struct qwen4_exp_qsa_query_args {
    uint position;
    float eps;
    uint pad0;
    uint pad1;
    qwen4_exp_qsa_rope_args rope;
};

struct qwen4_exp_qsa_score_args {
    uint n_blocks;
    uint cache_f16;
    uint pad0;
    uint pad1;
};

struct qwen4_exp_qsa_decode_args {
    uint n_selected_blocks;
    uint position;
    uint cache_capacity;
    uint cache_f16;
};

static float2 qwen4_exp_qsa_rope_pair(
        float x0,
        float x1,
        uint pair,
        uint position,
        constant qwen4_exp_qsa_rope_args &args) {
    if (args.n_rot == 0) return float2(x0, x1);
    const float exponent = 2.0f * float(pair) / float(args.n_rot);
    const float freq_extra = pow(args.freq_base, exponent);
    float theta = float(position) * args.freq_scale / freq_extra;
    float mscale = args.attn_factor;
    if (args.ext_factor != 0.0f) {
        float corr_dims[2] = {0.0f, 0.0f};
        rope_yarn_corr_dims((int)args.n_rot, (int)args.n_ctx_orig,
                            args.freq_base, args.beta_fast, args.beta_slow,
                            corr_dims);
        const float y = (float(pair) - corr_dims[0]) /
            max(0.001f, corr_dims[1] - corr_dims[0]);
        const float ramp = min(1.0f, max(0.0f, y));
        const float freq_mask = 1.0f - ramp * args.ext_factor;
        const float factor = 1.0f / args.freq_scale;
        const float freq_inter = factor * freq_extra;
        const float period = (freq_inter * freq_extra) /
            (freq_inter * freq_mask + freq_extra * (1.0f - freq_mask));
        theta = float(position) / period;
        mscale *= 1.0f + 0.1f * log(factor);
    }
    const float c = cos(theta) * mscale;
    const float s = sin(theta) * mscale;
    return float2(x0 * c - x1 * s, x0 * s + x1 * c);
}

template <typename CacheT>
METAL_FUNC float qwen4_exp_qsa_cache_load(
        device const CacheT *cache,
        uint index) {
    return float(cache[index]);
}

template <typename CacheT>
METAL_FUNC void qwen4_exp_qsa_pool_block_impl(
        constant qwen4_exp_qsa_pool_args &args,
        device const CacheT              *raw_keys,
        device const float               *norm_weight,
        device CacheT                    *pooled_keys,
        ushort lane) {
    float4 mean_value;
    const uint d0 = 4u * uint(lane);
    for (uint j = 0; j < 4u; j++) {
        float sum = 0.0f;
        for (uint r = 0; r < QWEN4_EXP_QSA_BLOCK; r++) {
            const uint token = args.block_index * QWEN4_EXP_QSA_BLOCK + r;
            sum += qwen4_exp_qsa_cache_load(
                raw_keys, token * QWEN4_EXP_QSA_INDEX_DIM + d0 + j);
        }
        float value = 0.25f * sum;
        if (args.cache_f16 != 0u) value = float(half(value));
        mean_value[j] = value;
    }
    const float sum_sq = simd_sum(dot(mean_value, mean_value));
    const float inv_rms = rsqrt(sum_sq / 128.0f + args.eps);
    const uint out_base = args.block_index * QWEN4_EXP_QSA_INDEX_DIM;
    const uint position = args.block_index * QWEN4_EXP_QSA_BLOCK;

    for (uint j = 0; j < 4u; j++) {
        const uint d = d0 + j;
        float value = mean_value[j] * inv_rms * norm_weight[d];
        if (d < args.rope.n_rot) {
            const uint half_rot = args.rope.n_rot / 2u;
            const uint pair = d < half_rot ? d : d - half_rot;
            const uint partner = d < half_rot ? d + half_rot : d - half_rot;
            float partner_mean = 0.0f;
            for (uint r = 0; r < QWEN4_EXP_QSA_BLOCK; r++) {
                const uint token = args.block_index * QWEN4_EXP_QSA_BLOCK + r;
                partner_mean += qwen4_exp_qsa_cache_load(
                    raw_keys, token * QWEN4_EXP_QSA_INDEX_DIM + partner);
            }
            partner_mean *= 0.25f;
            if (args.cache_f16 != 0u) partner_mean = float(half(partner_mean));
            const float partner_value =
                partner_mean * inv_rms * norm_weight[partner];
            const float2 rotated = d < half_rot
                ? qwen4_exp_qsa_rope_pair(value, partner_value, pair,
                                          position, args.rope)
                : qwen4_exp_qsa_rope_pair(partner_value, value, pair,
                                          position, args.rope);
            value = d < half_rot ? rotated.x : rotated.y;
        }
        pooled_keys[out_base + d] = CacheT(value);
    }
}

kernel void kernel_qwen4_exp_qsa_pool_block_f16(
        constant qwen4_exp_qsa_pool_args &args [[buffer(0)]],
        device const half                *raw_keys [[buffer(1)]],
        device const float               *norm_weight [[buffer(2)]],
        device half                      *pooled_keys [[buffer(3)]],
        ushort lane [[thread_index_in_simdgroup]]) {
    qwen4_exp_qsa_pool_block_impl(
        args, raw_keys, norm_weight, pooled_keys, lane);
}

kernel void kernel_qwen4_exp_qsa_pool_block_f32(
        constant qwen4_exp_qsa_pool_args &args [[buffer(0)]],
        device const float               *raw_keys [[buffer(1)]],
        device const float               *norm_weight [[buffer(2)]],
        device float                     *pooled_keys [[buffer(3)]],
        ushort lane [[thread_index_in_simdgroup]]) {
    qwen4_exp_qsa_pool_block_impl(
        args, raw_keys, norm_weight, pooled_keys, lane);
}

kernel void kernel_qwen4_exp_qsa_prepare_query(
        constant qwen4_exp_qsa_query_args &args [[buffer(0)]],
        device const float                *raw_query [[buffer(1)]],
        device const float                *norm_weight [[buffer(2)]],
        device float                      *prepared_query [[buffer(3)]],
        uint head [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    if (head >= QWEN4_EXP_QSA_INDEX_HEADS) return;
    const uint d0 = 4u * uint(lane);
    device const float4 *raw4 = reinterpret_cast<device const float4 *>(
        raw_query + head * QWEN4_EXP_QSA_INDEX_DIM);
    const float4 x = raw4[lane];
    const float sum_sq = simd_sum(dot(x, x));
    const float inv_rms = rsqrt(sum_sq / 128.0f + args.eps);
    const uint row = head * QWEN4_EXP_QSA_INDEX_DIM;

    for (uint j = 0; j < 4u; j++) {
        const uint d = d0 + j;
        float value = x[j] * inv_rms * norm_weight[d];
        if (d < args.rope.n_rot) {
            const uint half_rot = args.rope.n_rot / 2u;
            const uint pair = d < half_rot ? d : d - half_rot;
            const uint partner = d < half_rot ? d + half_rot : d - half_rot;
            const float partner_value =
                raw_query[row + partner] * inv_rms * norm_weight[partner];
            const float2 rotated = d < half_rot
                ? qwen4_exp_qsa_rope_pair(value, partner_value, pair,
                                          args.position, args.rope)
                : qwen4_exp_qsa_rope_pair(partner_value, value, pair,
                                          args.position, args.rope);
            value = d < half_rot ? rotated.x : rotated.y;
        }
        prepared_query[row + d] = value;
    }
}

template <typename CacheT>
METAL_FUNC void qwen4_exp_qsa_score_blocks_impl(
        constant qwen4_exp_qsa_score_args &args,
        device const float                *prepared_query,
        device const CacheT               *pooled_keys,
        device float                      *scores,
        threadgroup float                 *head_scores,
        uint block,
        ushort lane,
        ushort subgroup,
        ushort tid) {
    if (block >= args.n_blocks || subgroup >= QWEN4_EXP_QSA_INDEX_HEADS) return;
    device const float4 *q = reinterpret_cast<device const float4 *>(
        prepared_query + uint(subgroup) * QWEN4_EXP_QSA_INDEX_DIM);
    const uint base = block * QWEN4_EXP_QSA_INDEX_DIM + 4u * uint(lane);
    float4 key;
    for (uint j = 0; j < 4u; j++) key[j] = float(pooled_keys[base + j]);
    const float dot_value = simd_sum(dot(q[lane], key));
    if (lane == 0) head_scores[subgroup] = max(dot_value, 0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        scores[block] = (head_scores[0] + head_scores[1] +
                         head_scores[2] + head_scores[3]) *
                        0.08838834764831845f;
    }
}

kernel void kernel_qwen4_exp_qsa_score_blocks_f16(
        constant qwen4_exp_qsa_score_args &args [[buffer(0)]],
        device const float                *prepared_query [[buffer(1)]],
        device const half                 *pooled_keys [[buffer(2)]],
        device float                      *scores [[buffer(3)]],
        threadgroup float                 *head_scores [[threadgroup(0)]],
        uint block [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort subgroup [[simdgroup_index_in_threadgroup]],
        ushort tid [[thread_index_in_threadgroup]]) {
    qwen4_exp_qsa_score_blocks_impl(
        args, prepared_query, pooled_keys, scores, head_scores,
        block, lane, subgroup, tid);
}

kernel void kernel_qwen4_exp_qsa_score_blocks_f32(
        constant qwen4_exp_qsa_score_args &args [[buffer(0)]],
        device const float                *prepared_query [[buffer(1)]],
        device const float                *pooled_keys [[buffer(2)]],
        device float                      *scores [[buffer(3)]],
        threadgroup float                 *head_scores [[threadgroup(0)]],
        uint block [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort subgroup [[simdgroup_index_in_threadgroup]],
        ushort tid [[thread_index_in_threadgroup]]) {
    qwen4_exp_qsa_score_blocks_impl(
        args, prepared_query, pooled_keys, scores, head_scores,
        block, lane, subgroup, tid);
}

static uint qwen4_exp_qsa_selected_token(
        uint slot,
        constant qwen4_exp_qsa_decode_args &args,
        device const int *selected_blocks) {
    const uint selected_tokens = args.n_selected_blocks * QWEN4_EXP_QSA_BLOCK;
    if (slot < selected_tokens) {
        const int block = selected_blocks[slot / QWEN4_EXP_QSA_BLOCK];
        return block < 0 ? UINT_MAX :
            uint(block) * QWEN4_EXP_QSA_BLOCK + slot % QWEN4_EXP_QSA_BLOCK;
    }
    const uint complete_tokens =
        ((args.position + 1u) / QWEN4_EXP_QSA_BLOCK) * QWEN4_EXP_QSA_BLOCK;
    return complete_tokens + slot - selected_tokens;
}

template <typename CacheT>
METAL_FUNC void qwen4_exp_qsa_decode_impl(
        constant qwen4_exp_qsa_decode_args &args,
        device const float                 *query,
        device const float                 *gate,
        device const CacheT                *key_cache,
        device const CacheT                *value_cache,
        device const int                   *selected_blocks,
        device float                       *out,
        threadgroup float                  *scratch,
        uint head,
        ushort tid) {
    if (head >= QWEN4_EXP_QSA_HEADS) return;
    const uint tail = (args.position + 1u) % QWEN4_EXP_QSA_BLOCK;
    const uint n_keys = args.n_selected_blocks * QWEN4_EXP_QSA_BLOCK + tail;
    const uint kv_head = head / (QWEN4_EXP_QSA_HEADS / QWEN4_EXP_QSA_KV_HEADS);
    device const float *q = query + head * QWEN4_EXP_QSA_HEAD_DIM;
    threadgroup float *reduce = scratch;
    threadgroup float *attention_scores = scratch + 256u;

    float local_max = -INFINITY;
    for (uint slot = tid; slot < n_keys; slot += 256u) {
        const uint token = qwen4_exp_qsa_selected_token(
            slot, args, selected_blocks);
        float score = -INFINITY;
        if (token <= args.position && token < args.cache_capacity) {
            const uint64_t base =
                ((uint64_t)token * QWEN4_EXP_QSA_KV_HEADS + kv_head) *
                QWEN4_EXP_QSA_HEAD_DIM;
            float dot_value = 0.0f;
            for (uint d = 0; d < QWEN4_EXP_QSA_HEAD_DIM; d++) {
                dot_value += q[d] * float(key_cache[base + d]);
            }
            score = dot_value * 0.0625f;
        }
        attention_scores[slot] = score;
        local_max = max(local_max, score);
    }
    reduce[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 128u; step > 0u; step >>= 1u) {
        if (tid < step) reduce[tid] = max(reduce[tid], reduce[tid + step]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_score = reduce[0];

    float local_sum = 0.0f;
    for (uint slot = tid; slot < n_keys; slot += 256u) {
        const float probability = exp(attention_scores[slot] - max_score);
        attention_scores[slot] = probability;
        local_sum += probability;
    }
    reduce[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 128u; step > 0u; step >>= 1u) {
        if (tid < step) reduce[tid] += reduce[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float denom = max(reduce[0], 1.0e-20f);

    const uint d = uint(tid);
    float value_sum = 0.0f;
    for (uint slot = 0; slot < n_keys; slot++) {
        const uint token = qwen4_exp_qsa_selected_token(
            slot, args, selected_blocks);
        if (token <= args.position && token < args.cache_capacity) {
            const uint64_t base =
                ((uint64_t)token * QWEN4_EXP_QSA_KV_HEADS + kv_head) *
                QWEN4_EXP_QSA_HEAD_DIM;
            value_sum += attention_scores[slot] * float(value_cache[base + d]);
        }
    }
    const float gate_value = gate[head * QWEN4_EXP_QSA_HEAD_DIM + d];
    const float gate_scale = gate_value >= 0.0f
        ? 1.0f / (1.0f + exp(-gate_value))
        : exp(gate_value) / (1.0f + exp(gate_value));
    out[head * QWEN4_EXP_QSA_HEAD_DIM + d] =
        value_sum / denom * gate_scale;
}

kernel void kernel_qwen4_exp_qsa_decode_f16(
        constant qwen4_exp_qsa_decode_args &args [[buffer(0)]],
        device const float                 *query [[buffer(1)]],
        device const float                 *gate [[buffer(2)]],
        device const half                  *key_cache [[buffer(3)]],
        device const half                  *value_cache [[buffer(4)]],
        device const int                   *selected_blocks [[buffer(5)]],
        device float                       *out [[buffer(6)]],
        threadgroup float                  *scratch [[threadgroup(0)]],
        uint head [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    qwen4_exp_qsa_decode_impl(
        args, query, gate, key_cache, value_cache, selected_blocks,
        out, scratch, head, tid);
}

kernel void kernel_qwen4_exp_qsa_decode_f32(
        constant qwen4_exp_qsa_decode_args &args [[buffer(0)]],
        device const float                 *query [[buffer(1)]],
        device const float                 *gate [[buffer(2)]],
        device const float                 *key_cache [[buffer(3)]],
        device const float                 *value_cache [[buffer(4)]],
        device const int                   *selected_blocks [[buffer(5)]],
        device float                       *out [[buffer(6)]],
        threadgroup float                  *scratch [[threadgroup(0)]],
        uint head [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    qwen4_exp_qsa_decode_impl(
        args, query, gate, key_cache, value_cache, selected_blocks,
        out, scratch, head, tid);
}
