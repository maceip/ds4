/* Qwen4Exp per-layer embedding (PLE) hash and gather.
 *
 * One threadgroup scans a caller-sized token chunk in order.  That ordering is
 * part of the state contract: EOS resets the two-token n-gram history, and the
 * final four-word state can be copied verbatim for MTP snapshot/rollback.
 * Table rows are gathered directly from the model mmap in F16, Q8_0, or
 * MXFP4 form; no temporary allocation or generic arena is used.
 */

constant uint QWEN4_EXP_PLE_HEADS = 16;
constant uint QWEN4_EXP_PLE_HEAD_DIM = 160;
constant uint QWEN4_EXP_PLE_EMBED_DIM = 2560;
constant uint QWEN4_EXP_PLE_TYPE_F16 = 1;
constant uint QWEN4_EXP_PLE_TYPE_Q8_0 = 8;
constant uint QWEN4_EXP_PLE_TYPE_MXFP4 = 39;

struct qwen4_exp_ple_args {
    uint n_tokens;
    uint eos_token;
    uint table_type;
    uint pad0;
    ulong table_rows;
    ulong row_bytes;
    ulong multipliers[3];
    ulong head_vocab_sizes[QWEN4_EXP_PLE_HEADS];
    ulong head_offsets[QWEN4_EXP_PLE_HEADS];
};

static float qwen4_exp_ple_table_value(
        device const char *table,
        ulong              row_bytes,
        ulong              row,
        uint               dim,
        uint               table_type) {
    device const char *row_ptr = table + row * row_bytes;
    if (table_type == QWEN4_EXP_PLE_TYPE_F16) {
        return float(((device const half *)row_ptr)[dim]);
    }
    const uint block = dim / 32u;
    const uint inner = dim % 32u;
    if (table_type == QWEN4_EXP_PLE_TYPE_Q8_0) {
        const device block_q8_0 *qb =
            ((device const block_q8_0 *)row_ptr) + block;
        return float(qb->d) * float(qb->qs[inner]);
    }
    const device block_mxfp4 *qb =
        ((device const block_mxfp4 *)row_ptr) + block;
    const uint packed = uint(qb->qs[inner & 15u]);
    const uint code = (packed >> (inner < 16u ? 0u : 4u)) & 15u;
    return ds4_metal_e8m0_to_f32(qb->e) * ds4_metal_mxfp4_values[code];
}

kernel void kernel_qwen4_exp_ple_hash_gather(
        constant qwen4_exp_ple_args &args [[buffer(0)]],
        device const char           *table [[buffer(1)]],
        device const int            *token_ids [[buffer(2)]],
        device uint                 *token_state [[buffer(3)]],
        device float                *out [[buffer(4)]],
        threadgroup ulong           *gids [[threadgroup(0)]],
        threadgroup uint            *history [[threadgroup(1)]],
        ushort tid [[thread_index_in_threadgroup]]) {
    if (tid == 0) {
        history[0] = token_state[0];
        history[1] = token_state[1];
        history[2] = min(token_state[2], 2u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint token_index = 0; token_index < args.n_tokens; token_index++) {
        const uint current = uint(token_ids[token_index]);
        const uint previous = history[2] >= 1u ? history[1] : args.eos_token;
        const uint previous2 = history[2] >= 2u ? history[0] : args.eos_token;
        if (tid < QWEN4_EXP_PLE_HEADS) {
            const ulong pair_hash =
                (ulong(current) * args.multipliers[0]) ^
                (ulong(previous) * args.multipliers[1]);
            const ulong hash = tid < 8u ? pair_hash :
                pair_hash ^ (ulong(previous2) * args.multipliers[2]);
            gids[tid] = hash % args.head_vocab_sizes[tid] +
                        args.head_offsets[tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint index = uint(tid); index < QWEN4_EXP_PLE_EMBED_DIM;
             index += 256u) {
            const uint head = index / QWEN4_EXP_PLE_HEAD_DIM;
            const uint dim = index % QWEN4_EXP_PLE_HEAD_DIM;
            const ulong gid = gids[head];
            out[ulong(token_index) * QWEN4_EXP_PLE_EMBED_DIM + index] =
                gid < args.table_rows
                    ? qwen4_exp_ple_table_value(
                          table, args.row_bytes, gid, dim, args.table_type)
                    : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            if (current == args.eos_token) {
                history[0] = args.eos_token;
                history[1] = args.eos_token;
                history[2] = 0u;
            } else if (history[2] == 0u) {
                history[1] = current;
                history[2] = 1u;
            } else {
                history[0] = history[1];
                history[1] = current;
                history[2] = 2u;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        token_state[0] = history[0];
        token_state[1] = history[1];
        token_state[2] = history[2];
        token_state[3] = 0u;
    }
}
