#define _DARWIN_C_SOURCE

#include "ds4_gpu.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define GDN_KEY_HEADS 16u
#define GDN_VALUE_HEADS 48u
#define GDN_HEAD_DIM 128u
#define GDN_KEY_DIM (GDN_KEY_HEADS * GDN_HEAD_DIM)
#define GDN_VALUE_DIM (GDN_VALUE_HEADS * GDN_HEAD_DIM)
#define GDN_CONV_DIM (2u * GDN_KEY_DIM + GDN_VALUE_DIM)
#define GDN_STATE_COUNT ((uint64_t)GDN_VALUE_HEADS * GDN_HEAD_DIM * GDN_HEAD_DIM)
#define GDN_CONV_TOKENS 5u
#define GDN_SCAN_TOKENS 3u
#define GGML_TYPE_F32 0u
#define GGML_TYPE_F16 1u

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

static float fixture_value(uint64_t index, uint32_t salt, float scale) {
    const int32_t centered =
        (int32_t)((index * (17u + salt) + salt * 29u) % 257u) - 128;
    return (float)centered * scale;
}

static float softplus_ref(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

static float sigmoid_ref(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float e = expf(x);
    return e / (1.0f + e);
}

static int compare_values(const char *name, const float *actual,
                          const float *expected, uint64_t count,
                          float tolerance) {
    float max_abs = 0.0f;
    double sum_sq = 0.0;
    uint64_t max_index = 0;
    for (uint64_t i = 0; i < count; i++) {
        if (!isfinite(actual[i]) || !isfinite(expected[i])) {
            fprintf(stderr, "Qwen4Exp GDN %s non-finite at %llu\n",
                    name, (unsigned long long)i);
            return 0;
        }
        const float error = fabsf(actual[i] - expected[i]);
        if (error > max_abs) {
            max_abs = error;
            max_index = i;
        }
        sum_sq += (double)error * error;
    }
    fprintf(stderr,
            "Qwen4Exp GDN %-18s max_abs=%g rms=%g at=%llu tolerance=%g\n",
            name, max_abs, sqrt(sum_sq / (double)count),
            (unsigned long long)max_index, tolerance);
    return max_abs <= tolerance;
}

static void conv_reference(float *out, float *state, const float *input,
                           const void *weight, uint32_t weight_type,
                           uint32_t n_tokens) {
    for (uint32_t channel = 0; channel < GDN_CONV_DIM; channel++) {
        float s0 = state[channel];
        float s1 = state[GDN_CONV_DIM + channel];
        float s2 = state[2u * GDN_CONV_DIM + channel];
        const _Float16 *weight_f16 = weight;
        const float *weight_f32 = weight;
        const uint32_t woff = 4u * channel;
        const float w0 = weight_type == GGML_TYPE_F16
            ? (float)weight_f16[woff] : weight_f32[woff];
        const float w1 = weight_type == GGML_TYPE_F16
            ? (float)weight_f16[woff + 1u] : weight_f32[woff + 1u];
        const float w2 = weight_type == GGML_TYPE_F16
            ? (float)weight_f16[woff + 2u] : weight_f32[woff + 2u];
        const float w3 = weight_type == GGML_TYPE_F16
            ? (float)weight_f16[woff + 3u] : weight_f32[woff + 3u];
        for (uint32_t t = 0; t < n_tokens; t++) {
            const uint64_t off = (uint64_t)t * GDN_CONV_DIM + channel;
            const float x = input[off];
            const float y = w0 * s0 + w1 * s1 + w2 * s2 + w3 * x;
            out[off] = y / (1.0f + expf(-y));
            s0 = s1;
            s1 = s2;
            s2 = x;
        }
        state[channel] = s0;
        state[GDN_CONV_DIM + channel] = s1;
        state[2u * GDN_CONV_DIM + channel] = s2;
    }
}

static void recurrent_reference(float *out, float *state, const float *mixed,
                                const float *alpha, const float *beta,
                                const float *ssm_a, const float *dt_bias,
                                uint32_t n_tokens) {
    float *q_norm = malloc((uint64_t)n_tokens * GDN_KEY_DIM * sizeof(float));
    float *k_norm = malloc((uint64_t)n_tokens * GDN_KEY_DIM * sizeof(float));
    if (!q_norm || !k_norm) {
        fprintf(stderr, "Qwen4Exp GDN CPU reference allocation failed\n");
        exit(1);
    }

    for (uint32_t t = 0; t < n_tokens; t++) {
        const float *row = mixed + (uint64_t)t * GDN_CONV_DIM;
        for (uint32_t h = 0; h < GDN_KEY_HEADS; h++) {
            const float *q = row + (uint64_t)h * GDN_HEAD_DIM;
            const float *k = row + GDN_KEY_DIM + (uint64_t)h * GDN_HEAD_DIM;
            float qsum = 0.0f;
            float ksum = 0.0f;
            for (uint32_t d = 0; d < GDN_HEAD_DIM; d++) {
                qsum += q[d] * q[d];
                ksum += k[d] * k[d];
            }
            const float qscale =
                0.08838834764831845f / sqrtf(qsum + 1.0e-6f);
            const float kscale = 1.0f / sqrtf(ksum + 1.0e-6f);
            float *qn = q_norm + ((uint64_t)t * GDN_KEY_HEADS + h) * GDN_HEAD_DIM;
            float *kn = k_norm + ((uint64_t)t * GDN_KEY_HEADS + h) * GDN_HEAD_DIM;
            for (uint32_t d = 0; d < GDN_HEAD_DIM; d++) {
                qn[d] = q[d] * qscale;
                kn[d] = k[d] * kscale;
            }
        }
    }

    for (uint32_t t = 0; t < n_tokens; t++) {
        const float *values = mixed + (uint64_t)t * GDN_CONV_DIM + 2u * GDN_KEY_DIM;
        for (uint32_t hv = 0; hv < GDN_VALUE_HEADS; hv++) {
            const uint32_t hk = hv / 3u;
            const float *q = q_norm +
                ((uint64_t)t * GDN_KEY_HEADS + hk) * GDN_HEAD_DIM;
            const float *k = k_norm +
                ((uint64_t)t * GDN_KEY_HEADS + hk) * GDN_HEAD_DIM;
            const float g = expf(ssm_a[hv] * softplus_ref(
                alpha[(uint64_t)t * GDN_VALUE_HEADS + hv] + dt_bias[hv]));
            const float update = sigmoid_ref(
                beta[(uint64_t)t * GDN_VALUE_HEADS + hv]);
            for (uint32_t dv = 0; dv < GDN_HEAD_DIM; dv++) {
                float *state_row = state +
                    ((uint64_t)hv * GDN_HEAD_DIM + dv) * GDN_HEAD_DIM;
                float remembered = 0.0f;
                for (uint32_t dk = 0; dk < GDN_HEAD_DIM; dk++) {
                    state_row[dk] *= g;
                    remembered += state_row[dk] * k[dk];
                }
                const float delta =
                    (values[(uint64_t)hv * GDN_HEAD_DIM + dv] - remembered) * update;
                float result = 0.0f;
                for (uint32_t dk = 0; dk < GDN_HEAD_DIM; dk++) {
                    state_row[dk] += k[dk] * delta;
                    result += state_row[dk] * q[dk];
                }
                out[(uint64_t)t * GDN_VALUE_DIM +
                    (uint64_t)hv * GDN_HEAD_DIM + dv] = result;
            }
        }
    }

    free(k_norm);
    free(q_norm);
}

static int run_conv_test(const void *model, uint64_t model_size,
                         uint64_t weight_offset, uint32_t weight_type,
                         const void *weights, const char *label) {
    const uint64_t input_count = (uint64_t)GDN_CONV_TOKENS * GDN_CONV_DIM;
    const uint64_t state_count = 3ull * GDN_CONV_DIM;
    float *input = malloc(input_count * sizeof(float));
    float *initial_state = malloc(state_count * sizeof(float));
    float *expected = malloc(input_count * sizeof(float));
    float *expected_state = malloc(state_count * sizeof(float));
    float *scan_out = malloc(input_count * sizeof(float));
    float *step_out = malloc(input_count * sizeof(float));
    float *scan_state = malloc(state_count * sizeof(float));
    float *step_state = malloc(state_count * sizeof(float));
    if (!input || !initial_state || !expected || !expected_state ||
        !scan_out || !step_out || !scan_state || !step_state) {
        fprintf(stderr, "Qwen4Exp GDN convolution host allocation failed\n");
        return 0;
    }
    for (uint64_t i = 0; i < input_count; i++) {
        input[i] = fixture_value(i, 3u, 1.0f / 192.0f);
    }
    for (uint64_t i = 0; i < state_count; i++) {
        initial_state[i] = fixture_value(i, 5u, 1.0f / 256.0f);
    }
    memcpy(expected_state, initial_state, state_count * sizeof(float));
    conv_reference(expected, expected_state, input, weights, weight_type,
                   GDN_CONV_TOKENS);

    ds4_gpu_tensor *input_tensor = ds4_gpu_tensor_alloc(input_count * sizeof(float));
    ds4_gpu_tensor *scan_out_tensor = ds4_gpu_tensor_alloc(input_count * sizeof(float));
    ds4_gpu_tensor *step_out_tensor = ds4_gpu_tensor_alloc(input_count * sizeof(float));
    ds4_gpu_tensor *scan_state_tensor = ds4_gpu_tensor_alloc(state_count * sizeof(float));
    ds4_gpu_tensor *step_state_tensor = ds4_gpu_tensor_alloc(state_count * sizeof(float));
    int ok = input_tensor && scan_out_tensor && step_out_tensor &&
             scan_state_tensor && step_state_tensor;
    ok = ok && ds4_gpu_tensor_write(
        input_tensor, 0, input, input_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(
        scan_state_tensor, 0, initial_state, state_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(
        step_state_tensor, 0, initial_state, state_count * sizeof(float));
    ok = ok && ds4_gpu_qwen4_exp_gdn_conv_scan_tensor(
        scan_out_tensor, scan_state_tensor, input_tensor,
        model, model_size, weight_offset, weight_type, GDN_CONV_TOKENS);

    const uint64_t row_bytes = (uint64_t)GDN_CONV_DIM * sizeof(float);
    for (uint32_t t = 0; t < GDN_CONV_TOKENS && ok; t++) {
        ds4_gpu_tensor *input_row = ds4_gpu_tensor_view(
            input_tensor, (uint64_t)t * row_bytes, row_bytes);
        ds4_gpu_tensor *output_row = ds4_gpu_tensor_view(
            step_out_tensor, (uint64_t)t * row_bytes, row_bytes);
        ok = input_row && output_row &&
             ds4_gpu_qwen4_exp_gdn_conv_scan_tensor(
                 output_row, step_state_tensor, input_row,
                 model, model_size, weight_offset, weight_type, 1);
        ds4_gpu_tensor_free(output_row);
        ds4_gpu_tensor_free(input_row);
    }
    ok = ok && ds4_gpu_tensor_read(
        scan_out_tensor, 0, scan_out, input_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_read(
        step_out_tensor, 0, step_out, input_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_read(
        scan_state_tensor, 0, scan_state, state_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_read(
        step_state_tensor, 0, step_state, state_count * sizeof(float));
    if (ok && (memcmp(scan_out, step_out, input_count * sizeof(float)) != 0 ||
               memcmp(scan_state, step_state, state_count * sizeof(float)) != 0)) {
        fprintf(stderr, "Qwen4Exp GDN convolution %s chunk-vs-step mismatch\n",
                label);
        ok = 0;
    } else if (ok) {
        fprintf(stderr,
                "Qwen4Exp GDN convolution %s chunk-vs-step bit exact\n",
                label);
    }
    ok = ok && compare_values(
        "conv output", scan_out, expected, input_count, 3.0e-5f);
    ok = ok && compare_values(
        "conv state", scan_state, expected_state, state_count, 1.0e-7f);

    ds4_gpu_tensor_free(step_state_tensor);
    ds4_gpu_tensor_free(scan_state_tensor);
    ds4_gpu_tensor_free(step_out_tensor);
    ds4_gpu_tensor_free(scan_out_tensor);
    ds4_gpu_tensor_free(input_tensor);
    free(step_state);
    free(scan_state);
    free(step_out);
    free(scan_out);
    free(expected_state);
    free(expected);
    free(initial_state);
    free(input);
    return ok;
}

static int run_recurrent_test(void) {
    const uint64_t mixed_count = (uint64_t)GDN_SCAN_TOKENS * GDN_CONV_DIM;
    const uint64_t scalar_count = (uint64_t)GDN_SCAN_TOKENS * GDN_VALUE_HEADS;
    const uint64_t output_count = (uint64_t)GDN_SCAN_TOKENS * GDN_VALUE_DIM;
    float *mixed = malloc(mixed_count * sizeof(float));
    float *alpha = malloc(scalar_count * sizeof(float));
    float *beta = malloc(scalar_count * sizeof(float));
    float *ssm_a = malloc(GDN_VALUE_HEADS * sizeof(float));
    float *dt_bias = malloc(GDN_VALUE_HEADS * sizeof(float));
    float *initial_state = malloc(GDN_STATE_COUNT * sizeof(float));
    float *expected_state = malloc(GDN_STATE_COUNT * sizeof(float));
    float *expected_out = malloc(output_count * sizeof(float));
    float *scan_state = malloc(GDN_STATE_COUNT * sizeof(float));
    float *step_state = malloc(GDN_STATE_COUNT * sizeof(float));
    float *scan_out = malloc(output_count * sizeof(float));
    float *step_out = malloc(output_count * sizeof(float));
    if (!mixed || !alpha || !beta || !ssm_a || !dt_bias ||
        !initial_state || !expected_state || !expected_out ||
        !scan_state || !step_state || !scan_out || !step_out) {
        fprintf(stderr, "Qwen4Exp GDN recurrent host allocation failed\n");
        return 0;
    }
    for (uint64_t i = 0; i < mixed_count; i++) {
        mixed[i] = fixture_value(i, 7u, 1.0f / 384.0f);
    }
    for (uint64_t i = 0; i < scalar_count; i++) {
        alpha[i] = fixture_value(i, 11u, 1.0f / 128.0f);
        beta[i] = fixture_value(i, 13u, 1.0f / 96.0f);
    }
    for (uint32_t h = 0; h < GDN_VALUE_HEADS; h++) {
        ssm_a[h] = -0.08f - 0.005f * (float)(h % 11u);
        dt_bias[h] = -0.35f + 0.025f * (float)(h % 17u);
    }
    for (uint64_t i = 0; i < GDN_STATE_COUNT; i++) {
        initial_state[i] = fixture_value(i, 17u, 1.0f / 8192.0f);
    }
    memcpy(expected_state, initial_state, GDN_STATE_COUNT * sizeof(float));
    recurrent_reference(expected_out, expected_state, mixed, alpha, beta,
                        ssm_a, dt_bias, GDN_SCAN_TOKENS);

    ds4_gpu_tensor *mixed_scan_tensor = ds4_gpu_tensor_alloc(mixed_count * sizeof(float));
    ds4_gpu_tensor *mixed_step_tensor = ds4_gpu_tensor_alloc(mixed_count * sizeof(float));
    ds4_gpu_tensor *alpha_tensor = ds4_gpu_tensor_alloc(scalar_count * sizeof(float));
    ds4_gpu_tensor *beta_tensor = ds4_gpu_tensor_alloc(scalar_count * sizeof(float));
    ds4_gpu_tensor *ssm_a_tensor = ds4_gpu_tensor_alloc(GDN_VALUE_HEADS * sizeof(float));
    ds4_gpu_tensor *dt_bias_tensor = ds4_gpu_tensor_alloc(GDN_VALUE_HEADS * sizeof(float));
    ds4_gpu_tensor *scan_state_tensor = ds4_gpu_tensor_alloc(GDN_STATE_COUNT * sizeof(float));
    ds4_gpu_tensor *step_state_tensor = ds4_gpu_tensor_alloc(GDN_STATE_COUNT * sizeof(float));
    ds4_gpu_tensor *scan_out_tensor = ds4_gpu_tensor_alloc(output_count * sizeof(float));
    ds4_gpu_tensor *step_out_tensor = ds4_gpu_tensor_alloc(output_count * sizeof(float));
    ds4_gpu_tensor *scan_gate_tensor = ds4_gpu_tensor_alloc(2u * scalar_count * sizeof(float));
    ds4_gpu_tensor *step_gate_tensor = ds4_gpu_tensor_alloc(2u * GDN_VALUE_HEADS * sizeof(float));
    int ok = mixed_scan_tensor && mixed_step_tensor && alpha_tensor && beta_tensor &&
             ssm_a_tensor && dt_bias_tensor && scan_state_tensor && step_state_tensor &&
             scan_out_tensor && step_out_tensor && scan_gate_tensor && step_gate_tensor;
    ok = ok && ds4_gpu_tensor_write(
        mixed_scan_tensor, 0, mixed, mixed_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(
        mixed_step_tensor, 0, mixed, mixed_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(
        alpha_tensor, 0, alpha, scalar_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(
        beta_tensor, 0, beta, scalar_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(
        ssm_a_tensor, 0, ssm_a, GDN_VALUE_HEADS * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(
        dt_bias_tensor, 0, dt_bias, GDN_VALUE_HEADS * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(
        scan_state_tensor, 0, initial_state, GDN_STATE_COUNT * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(
        step_state_tensor, 0, initial_state, GDN_STATE_COUNT * sizeof(float));
    ok = ok && ds4_gpu_qwen4_exp_gdn_recurrent_scan_tensor(
        scan_out_tensor, scan_state_tensor, mixed_scan_tensor,
        alpha_tensor, beta_tensor, ssm_a_tensor, dt_bias_tensor,
        scan_gate_tensor, GDN_SCAN_TOKENS);

    const uint64_t mixed_row_bytes = (uint64_t)GDN_CONV_DIM * sizeof(float);
    const uint64_t scalar_row_bytes = (uint64_t)GDN_VALUE_HEADS * sizeof(float);
    const uint64_t out_row_bytes = (uint64_t)GDN_VALUE_DIM * sizeof(float);
    for (uint32_t t = 0; t < GDN_SCAN_TOKENS && ok; t++) {
        ds4_gpu_tensor *mixed_row = ds4_gpu_tensor_view(
            mixed_step_tensor, (uint64_t)t * mixed_row_bytes, mixed_row_bytes);
        ds4_gpu_tensor *alpha_row = ds4_gpu_tensor_view(
            alpha_tensor, (uint64_t)t * scalar_row_bytes, scalar_row_bytes);
        ds4_gpu_tensor *beta_row = ds4_gpu_tensor_view(
            beta_tensor, (uint64_t)t * scalar_row_bytes, scalar_row_bytes);
        ds4_gpu_tensor *out_row = ds4_gpu_tensor_view(
            step_out_tensor, (uint64_t)t * out_row_bytes, out_row_bytes);
        ok = mixed_row && alpha_row && beta_row && out_row &&
             ds4_gpu_qwen4_exp_gdn_recurrent_scan_tensor(
                 out_row, step_state_tensor, mixed_row, alpha_row, beta_row,
                 ssm_a_tensor, dt_bias_tensor, step_gate_tensor, 1);
        ds4_gpu_tensor_free(out_row);
        ds4_gpu_tensor_free(beta_row);
        ds4_gpu_tensor_free(alpha_row);
        ds4_gpu_tensor_free(mixed_row);
    }
    ok = ok && ds4_gpu_tensor_read(
        scan_out_tensor, 0, scan_out, output_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_read(
        step_out_tensor, 0, step_out, output_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_read(
        scan_state_tensor, 0, scan_state, GDN_STATE_COUNT * sizeof(float));
    ok = ok && ds4_gpu_tensor_read(
        step_state_tensor, 0, step_state, GDN_STATE_COUNT * sizeof(float));
    if (ok && (memcmp(scan_out, step_out, output_count * sizeof(float)) != 0 ||
               memcmp(scan_state, step_state, GDN_STATE_COUNT * sizeof(float)) != 0)) {
        fprintf(stderr, "Qwen4Exp GDN recurrent chunk-vs-step mismatch\n");
        ok = 0;
    } else if (ok) {
        fprintf(stderr, "Qwen4Exp GDN recurrent chunk-vs-step bit exact\n");
    }
    ok = ok && compare_values(
        "recurrent output", scan_out, expected_out, output_count, 1.0e-4f);
    ok = ok && compare_values(
        "recurrent state", scan_state, expected_state, GDN_STATE_COUNT, 1.0e-4f);

    ds4_gpu_tensor_free(step_gate_tensor);
    ds4_gpu_tensor_free(scan_gate_tensor);
    ds4_gpu_tensor_free(step_out_tensor);
    ds4_gpu_tensor_free(scan_out_tensor);
    ds4_gpu_tensor_free(step_state_tensor);
    ds4_gpu_tensor_free(scan_state_tensor);
    ds4_gpu_tensor_free(dt_bias_tensor);
    ds4_gpu_tensor_free(ssm_a_tensor);
    ds4_gpu_tensor_free(beta_tensor);
    ds4_gpu_tensor_free(alpha_tensor);
    ds4_gpu_tensor_free(mixed_step_tensor);
    ds4_gpu_tensor_free(mixed_scan_tensor);
    free(step_out);
    free(scan_out);
    free(step_state);
    free(scan_state);
    free(expected_out);
    free(expected_state);
    free(initial_state);
    free(dt_bias);
    free(ssm_a);
    free(beta);
    free(alpha);
    free(mixed);
    return ok;
}

int main(void) {
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t weight_count = 4ull * GDN_CONV_DIM;
    const uint64_t f16_bytes = weight_count * sizeof(_Float16);
    const uint64_t f32_offset = align_up(f16_bytes, page);
    const uint64_t f32_bytes = weight_count * sizeof(float);
    const uint64_t model_size = align_up(f32_offset + f32_bytes, page);
    void *model = NULL;
    if (posix_memalign(&model, (size_t)page, (size_t)model_size) != 0) {
        fprintf(stderr, "Qwen4Exp GDN test model allocation failed\n");
        return 1;
    }
    memset(model, 0, (size_t)model_size);
    _Float16 *weights_f16 = model;
    float *weights_f32 = (float *)((uint8_t *)model + f32_offset);
    for (uint64_t i = 0; i < weight_count; i++) {
        const float value = fixture_value(i, 19u, 1.0f / 512.0f);
        weights_f16[i] = (_Float16)value;
        weights_f32[i] = value;
    }

    int ok = ds4_gpu_init() && ds4_gpu_set_model_map(model, model_size);
    ok = ok && run_conv_test(model, model_size, 0, GGML_TYPE_F16,
                             weights_f16, "F16");
    ok = ok && run_conv_test(model, model_size, f32_offset, GGML_TYPE_F32,
                             weights_f32, "F32");
    ok = ok && run_recurrent_test();
    ds4_gpu_cleanup();
    free(model);

    if (!ok) {
        fprintf(stderr, "Qwen4Exp GatedDeltaNet Metal tests FAILED\n");
        return 1;
    }
    fprintf(stderr, "Qwen4Exp GatedDeltaNet Metal tests PASS\n");
    return 0;
}
