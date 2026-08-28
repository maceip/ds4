/*
 * Qwen4Exp GatedDeltaNet kernels.
 *
 * The recurrent scan is deliberately sequential in token order.  One
 * simdgroup owns one 128-wide state row, keeps it register-resident across a
 * prefill chunk, and writes it back once.  The same arithmetic is used for a
 * one-token decode dispatch, which makes chunked prefill and repeated decode
 * agree without a second scan implementation.
 */

constant uint QWEN4_EXP_GDN_KEY_HEADS = 16;
constant uint QWEN4_EXP_GDN_VALUE_HEADS = 48;
constant uint QWEN4_EXP_GDN_HEAD_DIM = 128;
constant uint QWEN4_EXP_GDN_KEY_DIM = 2048;
constant uint QWEN4_EXP_GDN_VALUE_DIM = 6144;
constant uint QWEN4_EXP_GDN_CONV_DIM = 10240;
constant uint QWEN4_EXP_GDN_CONV_STATE = 3;

struct qwen4_exp_gdn_args {
    uint n_tokens;
    uint pad0;
    uint pad1;
    uint pad2;
};

template <typename W>
METAL_FUNC void qwen4_exp_gdn_conv_scan_impl(
        device float       *out,
        device float       *conv_state,
        device const float *input,
        device const W     *weight,
        constant qwen4_exp_gdn_args &args,
        uint channel) {
    if (channel >= QWEN4_EXP_GDN_CONV_DIM) return;

    float s0 = conv_state[channel];
    float s1 = conv_state[QWEN4_EXP_GDN_CONV_DIM + channel];
    float s2 = conv_state[2 * QWEN4_EXP_GDN_CONV_DIM + channel];
    device const W *w = weight + 4 * channel;
    const float w0 = float(w[0]);
    const float w1 = float(w[1]);
    const float w2 = float(w[2]);
    const float w3 = float(w[3]);

    for (uint t = 0; t < args.n_tokens; t++) {
        const uint off = t * QWEN4_EXP_GDN_CONV_DIM + channel;
        const float x = input[off];
        const float y = w0 * s0 + w1 * s1 + w2 * s2 + w3 * x;
        out[off] = y / (1.0f + exp(-y));
        s0 = s1;
        s1 = s2;
        s2 = x;
    }

    conv_state[channel] = s0;
    conv_state[QWEN4_EXP_GDN_CONV_DIM + channel] = s1;
    conv_state[2 * QWEN4_EXP_GDN_CONV_DIM + channel] = s2;
}

kernel void kernel_qwen4_exp_gdn_conv_scan_f16(
        constant qwen4_exp_gdn_args &args [[buffer(0)]],
        device const half           *weight [[buffer(1)]],
        device const float          *input [[buffer(2)]],
        device float                *conv_state [[buffer(3)]],
        device float                *out [[buffer(4)]],
        uint channel [[thread_position_in_grid]]) {
    qwen4_exp_gdn_conv_scan_impl(out, conv_state, input, weight, args, channel);
}

kernel void kernel_qwen4_exp_gdn_conv_scan_f32(
        constant qwen4_exp_gdn_args &args [[buffer(0)]],
        device const float          *weight [[buffer(1)]],
        device const float          *input [[buffer(2)]],
        device float                *conv_state [[buffer(3)]],
        device float                *out [[buffer(4)]],
        uint channel [[thread_position_in_grid]]) {
    qwen4_exp_gdn_conv_scan_impl(out, conv_state, input, weight, args, channel);
}

/* Normalize q/k in place and materialize scalar decay/update gates.  Value
 * heads are a 3:1 expansion of key heads, so only the first value head in
 * each triplet normalizes its shared q/k row. */
kernel void kernel_qwen4_exp_gdn_prepare(
        constant qwen4_exp_gdn_args &args [[buffer(0)]],
        device float                *mixed_qkv [[buffer(1)]],
        device const float          *alpha [[buffer(2)]],
        device const float          *beta [[buffer(3)]],
        device const float          *ssm_a [[buffer(4)]],
        device const float          *dt_bias [[buffer(5)]],
        device float                *gates [[buffer(6)]],
        uint group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint t = group / QWEN4_EXP_GDN_VALUE_HEADS;
    const uint hv = group - t * QWEN4_EXP_GDN_VALUE_HEADS;
    if (t >= args.n_tokens) return;

    if (lane == 0) {
        const float x = alpha[t * QWEN4_EXP_GDN_VALUE_HEADS + hv] + dt_bias[hv];
        const float softplus = x > 20.0f ? x : log(1.0f + exp(x));
        const float g = exp(ssm_a[hv] * softplus);
        const float b = beta[t * QWEN4_EXP_GDN_VALUE_HEADS + hv];
        const float sigmoid = b >= 0.0f
            ? 1.0f / (1.0f + exp(-b))
            : exp(b) / (1.0f + exp(b));
        const uint gate_off = 2 * (t * QWEN4_EXP_GDN_VALUE_HEADS + hv);
        gates[gate_off] = g;
        gates[gate_off + 1] = sigmoid;
    }

    if (hv % 3 != 0) return;
    const uint hk = hv / 3;
    const uint row = t * QWEN4_EXP_GDN_CONV_DIM + hk * QWEN4_EXP_GDN_HEAD_DIM;
    device float4 *q = reinterpret_cast<device float4 *>(mixed_qkv + row);
    device float4 *k = reinterpret_cast<device float4 *>(
        mixed_qkv + row + QWEN4_EXP_GDN_KEY_DIM);
    float4 qv = q[lane];
    float4 kv = k[lane];
    const float qsum = simd_sum(dot(qv, qv));
    const float ksum = simd_sum(dot(kv, kv));
    q[lane] = qv * rsqrt(qsum + 1.0e-6f) * 0.08838834764831845f;
    k[lane] = kv * rsqrt(ksum + 1.0e-6f);
}

kernel void kernel_qwen4_exp_gdn_recurrent_scan(
        constant qwen4_exp_gdn_args &args [[buffer(0)]],
        device const float          *mixed_qkv [[buffer(1)]],
        device const float          *gates [[buffer(2)]],
        device float                *state [[buffer(3)]],
        device float                *out [[buffer(4)]],
        uint3 gid [[thread_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint dv = gid.y;
    const uint hv = gid.z;
    if (dv >= QWEN4_EXP_GDN_HEAD_DIM || hv >= QWEN4_EXP_GDN_VALUE_HEADS) return;

    const uint hk = hv / 3;
    const uint state_vec_off = (hv * QWEN4_EXP_GDN_HEAD_DIM + dv) * 32 + lane;
    device float4 *state4 = reinterpret_cast<device float4 *>(state);
    float4 s = state4[state_vec_off];

    for (uint t = 0; t < args.n_tokens; t++) {
        const uint token_off = t * QWEN4_EXP_GDN_CONV_DIM;
        device const float4 *q = reinterpret_cast<device const float4 *>(
            mixed_qkv + token_off + hk * QWEN4_EXP_GDN_HEAD_DIM);
        device const float4 *k = reinterpret_cast<device const float4 *>(
            mixed_qkv + token_off + QWEN4_EXP_GDN_KEY_DIM +
            hk * QWEN4_EXP_GDN_HEAD_DIM);
        const uint gate_off = 2 * (t * QWEN4_EXP_GDN_VALUE_HEADS + hv);
        const float g = gates[gate_off];
        const float beta = gates[gate_off + 1];

        const float4 kval = k[lane];
        s *= g;
        const float remembered = simd_sum(dot(s, kval));
        const float value = mixed_qkv[token_off + 2 * QWEN4_EXP_GDN_KEY_DIM +
                                      hv * QWEN4_EXP_GDN_HEAD_DIM + dv];
        const float delta = (value - remembered) * beta;
        s += kval * delta;
        const float result = simd_sum(dot(s, q[lane]));
        if (lane == 0) {
            out[t * QWEN4_EXP_GDN_VALUE_DIM +
                hv * QWEN4_EXP_GDN_HEAD_DIM + dv] = result;
        }
    }

    state4[state_vec_off] = s;
}
