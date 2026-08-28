#define _DARWIN_C_SOURCE

#include "ds4_gpu.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MXFP4_TYPE 39u
#define QK_MXFP4 32u
#define N_TOTAL_EXPERT 512u
#define N_EXPERT 10u
#define DIM 256u
#define ROUTER_TOKENS 2u

typedef struct {
    uint8_t e;
    uint8_t qs[QK_MXFP4 / 2u];
} block_mxfp4;

typedef struct {
    float score;
    int32_t expert;
} route_entry;

static const float mxfp4_values[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
   -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f,
};

/* Generated from MTPLX qwen4_exp.py on 2026-08-27 with:
 *   idx = mx.argpartition(-logits, 9, axis=-1)[..., :10]
 *   w = mx.softmax(mx.take_along_axis(logits, idx, axis=-1),
 *                  axis=-1, precise=True)
 * This is the authoritative route ordering and precise-softmax fixture. */
static const int32_t mlx_selected[ROUTER_TOKENS][N_EXPERT] = {
    {255, 31, 383, 64, 127, 7, 500, 447, 300, 511},
    {384, 222, 80, 505, 448, 510, 3, 301, 490, 129},
};

static const float mlx_weights[ROUTER_TOKENS][N_EXPERT] = {
    {0.63214928f, 0.23255473f, 0.085552111f, 0.031472862f,
     0.011578219f, 0.0042593884f, 0.0015669420f, 0.00057644548f,
     0.00021206250f, 0.000078013451f},
    {0.63214928f, 0.23255473f, 0.085552111f, 0.031472862f,
     0.011578219f, 0.0042593884f, 0.0015669420f, 0.00057644548f,
     0.00021206250f, 0.000078013451f},
};

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

static float e8m0_to_f32(uint8_t e) {
    uint32_t bits = e == 0 ? 0x00400000u : (uint32_t)e << 23u;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float dot_mxfp4(const block_mxfp4 *row, const float *x) {
    float sum = 0.0f;
    for (uint32_t block = 0; block < DIM / QK_MXFP4; block++) {
        const block_mxfp4 *b = row + block;
        const float scale = e8m0_to_f32(b->e);
        for (uint32_t i = 0; i < QK_MXFP4 / 2u; i++) {
            const uint8_t q = b->qs[i];
            sum += scale * mxfp4_values[q & 15u] *
                   x[block * QK_MXFP4 + i];
            sum += scale * mxfp4_values[q >> 4u] *
                   x[block * QK_MXFP4 + i + QK_MXFP4 / 2u];
        }
    }
    return sum;
}

static void fill_matrix(block_mxfp4 *matrix, uint32_t salt) {
    const uint32_t blocks_per_row = DIM / QK_MXFP4;
    for (uint32_t expert = 0; expert < N_TOTAL_EXPERT; expert++) {
        for (uint32_t row = 0; row < DIM; row++) {
            block_mxfp4 *blocks = matrix +
                ((uint64_t)expert * DIM + row) * blocks_per_row;
            for (uint32_t block = 0; block < blocks_per_row; block++) {
                block_mxfp4 *b = blocks + block;
                b->e = (uint8_t)(116u +
                    ((salt + expert * 3u + row + block * 5u) % 4u));
                for (uint32_t i = 0; i < QK_MXFP4 / 2u; i++) {
                    const uint8_t lo = (uint8_t)(
                        (salt + expert * 7u + row * 3u + block + i) & 15u);
                    const uint8_t hi = (uint8_t)(
                        (salt * 3u + expert + row * 5u + block * 7u +
                         i * 3u) & 15u);
                    b->qs[i] = (uint8_t)(lo | (hi << 4u));
                }
            }
        }
    }
}

static void fill_router_logits(float logits[ROUTER_TOKENS][N_TOTAL_EXPERT]) {
    static const int32_t ids[ROUTER_TOKENS][N_EXPERT] = {
        {7, 31, 64, 127, 255, 300, 383, 447, 500, 511},
        {3, 80, 129, 222, 301, 384, 448, 490, 505, 510},
    };
    static const float scores[ROUTER_TOKENS][N_EXPERT] = {
        {5.0f, 9.0f, 7.0f, 6.0f, 10.0f, 2.0f, 8.0f, 3.0f, 4.0f, 1.0f},
        {4.5f, 8.5f, 1.5f, 9.5f, 3.5f, 10.5f, 6.5f, 2.5f, 7.5f, 5.5f},
    };
    for (uint32_t token = 0; token < ROUTER_TOKENS; token++) {
        for (uint32_t expert = 0; expert < N_TOTAL_EXPERT; expert++) {
            logits[token][expert] =
                -20.0f + (float)((expert * 37u + token * 11u) % 101u) /
                             1000.0f;
        }
        for (uint32_t slot = 0; slot < N_EXPERT; slot++) {
            logits[token][(uint32_t)ids[token][slot]] = scores[token][slot];
        }
    }
}

static int compare_values(const char *name, const float *actual,
                          const float *expected, uint64_t count,
                          float tolerance) {
    float max_abs = 0.0f;
    uint64_t max_index = 0;
    for (uint64_t i = 0; i < count; i++) {
        if (!isfinite(actual[i]) || !isfinite(expected[i])) {
            fprintf(stderr, "Qwen4Exp MoE %s non-finite at %llu\n",
                    name, (unsigned long long)i);
            return 0;
        }
        const float error = fabsf(actual[i] - expected[i]);
        if (error > max_abs) {
            max_abs = error;
            max_index = i;
        }
    }
    fprintf(stderr, "Qwen4Exp MoE %-14s max_abs=%g at=%llu\n",
            name, max_abs, (unsigned long long)max_index);
    return max_abs <= tolerance;
}

int main(void) {
    float logits[ROUTER_TOKENS][N_TOTAL_EXPERT];
    int32_t selected[ROUTER_TOKENS][N_EXPERT];
    float weights[ROUTER_TOKENS][N_EXPERT];
    int32_t step_selected[ROUTER_TOKENS][N_EXPERT];
    float step_weights[ROUTER_TOKENS][N_EXPERT];
    fill_router_logits(logits);

    int ok = ds4_gpu_init();
    ds4_gpu_tensor *logits_tensor = ds4_gpu_tensor_alloc(sizeof(logits));
    ds4_gpu_tensor *selected_tensor = ds4_gpu_tensor_alloc(sizeof(selected));
    ds4_gpu_tensor *weights_tensor = ds4_gpu_tensor_alloc(sizeof(weights));
    ds4_gpu_tensor *step_logits_tensor =
        ds4_gpu_tensor_alloc(N_TOTAL_EXPERT * sizeof(float));
    ds4_gpu_tensor *step_selected_tensor =
        ds4_gpu_tensor_alloc(N_EXPERT * sizeof(int32_t));
    ds4_gpu_tensor *step_weights_tensor =
        ds4_gpu_tensor_alloc(N_EXPERT * sizeof(float));
    ok = ok && logits_tensor && selected_tensor && weights_tensor &&
         step_logits_tensor && step_selected_tensor && step_weights_tensor;
    ok = ok && ds4_gpu_tensor_write(logits_tensor, 0, logits, sizeof(logits));
    ok = ok && ds4_gpu_qwen4_exp_moe_router_tensor(
        selected_tensor, weights_tensor, logits_tensor, ROUTER_TOKENS);
    ok = ok && ds4_gpu_tensor_read(
        selected_tensor, 0, selected, sizeof(selected));
    ok = ok && ds4_gpu_tensor_read(weights_tensor, 0, weights, sizeof(weights));

    for (uint32_t token = 0; token < ROUTER_TOKENS && ok; token++) {
        ok = ds4_gpu_tensor_write(
                 step_logits_tensor, 0, logits[token], sizeof(logits[token])) &&
             ds4_gpu_qwen4_exp_moe_router_tensor(
                 step_selected_tensor, step_weights_tensor,
                 step_logits_tensor, 1u) &&
             ds4_gpu_tensor_read(
                 step_selected_tensor, 0, step_selected[token],
                 sizeof(step_selected[token])) &&
             ds4_gpu_tensor_read(
                 step_weights_tensor, 0, step_weights[token],
                 sizeof(step_weights[token]));
    }
    if (ok && memcmp(selected, mlx_selected, sizeof(selected)) != 0) {
        fprintf(stderr, "Qwen4Exp MoE router selected IDs differ from MLX fixture\n");
        ok = 0;
    }
    if (ok && memcmp(selected, step_selected, sizeof(selected)) != 0) {
        fprintf(stderr, "Qwen4Exp MoE router batch/step selected IDs differ\n");
        ok = 0;
    }
    if (ok && memcmp(weights, step_weights, sizeof(weights)) != 0) {
        fprintf(stderr, "Qwen4Exp MoE router batch/step weights differ\n");
        ok = 0;
    }
    ok = ok && compare_values(
        "MLX weights", &weights[0][0], &mlx_weights[0][0],
        ROUTER_TOKENS * N_EXPERT, 2.5e-7f);
    if (ok) {
        fprintf(stderr,
                "Qwen4Exp MoE router 512/top-10 MLX golden and batch/step exact PASS\n");
    }

    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t row_bytes =
        (DIM / QK_MXFP4) * sizeof(block_mxfp4);
    const uint64_t expert_bytes = DIM * row_bytes;
    const uint64_t tensor_bytes = N_TOTAL_EXPERT * expert_bytes;
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = align_up(tensor_bytes, page);
    const uint64_t down_offset = align_up(up_offset + tensor_bytes, page);
    const uint64_t model_size = align_up(down_offset + tensor_bytes, page);
    void *model = NULL;
    if (ok && posix_memalign(&model, (size_t)page, (size_t)model_size) != 0) {
        fprintf(stderr, "Qwen4Exp MoE model allocation failed\n");
        ok = 0;
    }
    if (ok) {
        memset(model, 0, (size_t)model_size);
        fill_matrix((block_mxfp4 *)((uint8_t *)model + gate_offset), 1u);
        fill_matrix((block_mxfp4 *)((uint8_t *)model + up_offset), 5u);
        fill_matrix((block_mxfp4 *)((uint8_t *)model + down_offset), 9u);
        ok = ds4_gpu_set_model_map(model, model_size);
    }

    float x[DIM];
    for (uint32_t i = 0; i < DIM; i++) {
        x[i] = (float)((int32_t)((i * 13u) % 31u) - 15) / 128.0f;
    }
    const uint64_t pair_count = (uint64_t)N_EXPERT * DIM;
    float *gate_ref = calloc((size_t)pair_count, sizeof(float));
    float *up_ref = calloc((size_t)pair_count, sizeof(float));
    float *mid_ref = calloc((size_t)pair_count, sizeof(float));
    float *out_ref = calloc(DIM, sizeof(float));
    float *gate_gpu = calloc((size_t)pair_count, sizeof(float));
    float *up_gpu = calloc((size_t)pair_count, sizeof(float));
    float *mid_gpu = calloc((size_t)pair_count, sizeof(float));
    float *out_gpu = calloc(DIM, sizeof(float));
    ok = ok && gate_ref && up_ref && mid_ref && out_ref &&
         gate_gpu && up_gpu && mid_gpu && out_gpu;

    if (ok) {
        const block_mxfp4 *gate_matrix =
            (const block_mxfp4 *)((const uint8_t *)model + gate_offset);
        const block_mxfp4 *up_matrix =
            (const block_mxfp4 *)((const uint8_t *)model + up_offset);
        const block_mxfp4 *down_matrix =
            (const block_mxfp4 *)((const uint8_t *)model + down_offset);
        const uint64_t blocks_per_expert =
            expert_bytes / sizeof(block_mxfp4);
        const uint64_t blocks_per_row = row_bytes / sizeof(block_mxfp4);
        for (uint32_t slot = 0; slot < N_EXPERT; slot++) {
            const uint32_t expert = (uint32_t)selected[0][slot];
            for (uint32_t row = 0; row < DIM; row++) {
                const uint64_t pair = (uint64_t)slot * DIM + row;
                gate_ref[pair] = dot_mxfp4(
                    gate_matrix + (uint64_t)expert * blocks_per_expert +
                        (uint64_t)row * blocks_per_row,
                    x);
                up_ref[pair] = dot_mxfp4(
                    up_matrix + (uint64_t)expert * blocks_per_expert +
                        (uint64_t)row * blocks_per_row,
                    x);
                const float gate = gate_ref[pair];
                mid_ref[pair] =
                    (gate / (1.0f + expf(-gate))) * up_ref[pair] *
                    weights[0][slot];
            }
        }
        for (uint32_t row = 0; row < DIM; row++) {
            for (uint32_t slot = 0; slot < N_EXPERT; slot++) {
                const uint32_t expert = (uint32_t)selected[0][slot];
                out_ref[row] += dot_mxfp4(
                    down_matrix + (uint64_t)expert * blocks_per_expert +
                        (uint64_t)row * blocks_per_row,
                    mid_ref + (uint64_t)slot * DIM);
            }
        }
    }

    ds4_gpu_tensor *x_tensor = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *selected_row = ds4_gpu_tensor_view(
        selected_tensor, 0, N_EXPERT * sizeof(int32_t));
    ds4_gpu_tensor *weights_row = ds4_gpu_tensor_view(
        weights_tensor, 0, N_EXPERT * sizeof(float));
    ds4_gpu_tensor *gate_tensor =
        ds4_gpu_tensor_alloc(pair_count * sizeof(float));
    ds4_gpu_tensor *up_tensor =
        ds4_gpu_tensor_alloc(pair_count * sizeof(float));
    ds4_gpu_tensor *mid_tensor =
        ds4_gpu_tensor_alloc(pair_count * sizeof(float));
    ds4_gpu_tensor *experts_tensor =
        ds4_gpu_tensor_alloc(pair_count * sizeof(float));
    ds4_gpu_tensor *out_tensor = ds4_gpu_tensor_alloc(DIM * sizeof(float));
    ok = ok && x_tensor && selected_row && weights_row && gate_tensor &&
         up_tensor && mid_tensor && experts_tensor && out_tensor;
    ok = ok && ds4_gpu_tensor_write(x_tensor, 0, x, sizeof(x));
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);
    ok = ok && ds4_gpu_routed_moe_one_tensor(
        out_tensor, gate_tensor, up_tensor, mid_tensor, experts_tensor,
        model, model_size, gate_offset, up_offset, down_offset,
        MXFP4_TYPE, MXFP4_TYPE, expert_bytes, row_bytes,
        expert_bytes, row_bytes, DIM, DIM, DIM,
        selected_row, weights_row, N_TOTAL_EXPERT, N_EXPERT,
        0.0f, x_tensor, NULL, 0u, true);
    ok = ok && ds4_gpu_tensor_read(
        gate_tensor, 0, gate_gpu, pair_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_read(
        up_tensor, 0, up_gpu, pair_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_read(
        mid_tensor, 0, mid_gpu, pair_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_read(out_tensor, 0, out_gpu, DIM * sizeof(float));
    ok = ok && compare_values("gate", gate_gpu, gate_ref, pair_count, 2.0e-5f);
    ok = ok && compare_values("up", up_gpu, up_ref, pair_count, 2.0e-5f);
    ok = ok && compare_values("mid no clamp", mid_gpu, mid_ref,
                              pair_count, 2.0e-5f);
    ok = ok && compare_values("sum10 output", out_gpu, out_ref, DIM, 2.0e-4f);
    if (ok) {
        fprintf(stderr,
                "Qwen4Exp MoE 512-expert MXFP4 top-10 routed dispatch PASS\n");
    }

    ds4_gpu_tensor_free(out_tensor);
    ds4_gpu_tensor_free(experts_tensor);
    ds4_gpu_tensor_free(mid_tensor);
    ds4_gpu_tensor_free(up_tensor);
    ds4_gpu_tensor_free(gate_tensor);
    ds4_gpu_tensor_free(weights_row);
    ds4_gpu_tensor_free(selected_row);
    ds4_gpu_tensor_free(x_tensor);
    ds4_gpu_tensor_free(step_weights_tensor);
    ds4_gpu_tensor_free(step_selected_tensor);
    ds4_gpu_tensor_free(step_logits_tensor);
    ds4_gpu_tensor_free(weights_tensor);
    ds4_gpu_tensor_free(selected_tensor);
    ds4_gpu_tensor_free(logits_tensor);
    ds4_gpu_cleanup();
    free(out_gpu);
    free(mid_gpu);
    free(up_gpu);
    free(gate_gpu);
    free(out_ref);
    free(mid_ref);
    free(up_ref);
    free(gate_ref);
    free(model);
    return ok ? 0 : 1;
}
