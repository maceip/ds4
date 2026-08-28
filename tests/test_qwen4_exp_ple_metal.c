#define _DARWIN_C_SOURCE

#include "ds4_gpu.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PLE_HEADS 16u
#define PLE_HEAD_DIM 160u
#define PLE_EMBED_DIM (PLE_HEADS * PLE_HEAD_DIM)
#define PLE_ROWS_PER_HEAD 17u
#define PLE_TABLE_ROWS (PLE_HEADS * PLE_ROWS_PER_HEAD)
#define PLE_TOKENS 7u
#define PLE_EOS 248044u
#define GGML_TYPE_F16 1u
#define GGML_TYPE_Q8_0 8u
#define GGML_TYPE_MXFP4 39u

typedef struct {
    _Float16 d;
    int8_t qs[32];
} test_block_q8_0;

typedef struct {
    uint8_t e;
    uint8_t qs[16];
} test_block_mxfp4;

_Static_assert(sizeof(test_block_q8_0) == 34, "Q8_0 test block ABI changed");
_Static_assert(sizeof(test_block_mxfp4) == 17, "MXFP4 test block ABI changed");

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

static const uint64_t multipliers[3] = {
    UINT64_C(4788054244585),
    UINT64_C(5075510189727),
    UINT64_C(24189832309785),
};

static const uint32_t tokens[PLE_TOKENS] = {
    11, 12, 13, PLE_EOS, 21, 22, 23,
};

/* Generated with the authoritative NGramEmbedding signed-int64 expression.
 * The tiny prime-17 heads keep the synthetic table small without changing the
 * hash/XOR/modulo contract. */
static const uint32_t golden_gids[PLE_TOKENS][PLE_HEADS] = {
    {7, 24, 41, 58, 75, 92, 109, 126, 143, 160, 177, 194, 211, 228, 245, 262},
    {6, 23, 40, 57, 74, 91, 108, 125, 145, 162, 179, 196, 213, 230, 247, 264},
    {2, 19, 36, 53, 70, 87, 104, 121, 141, 158, 175, 192, 209, 226, 243, 260},
    {9, 26, 43, 60, 77, 94, 111, 128, 139, 156, 173, 190, 207, 224, 241, 258},
    {5, 22, 39, 56, 73, 90, 107, 124, 136, 153, 170, 187, 204, 221, 238, 255},
    {1, 18, 35, 52, 69, 86, 103, 120, 136, 153, 170, 187, 204, 221, 238, 255},
    {12, 29, 46, 63, 80, 97, 114, 131, 139, 156, 173, 190, 207, 224, 241, 258},
};

static uint64_t table_row_bytes(uint32_t type) {
    switch (type) {
    case GGML_TYPE_F16: return PLE_HEAD_DIM * sizeof(_Float16);
    case GGML_TYPE_Q8_0: return (PLE_HEAD_DIM / 32u) * sizeof(test_block_q8_0);
    case GGML_TYPE_MXFP4: return (PLE_HEAD_DIM / 32u) * sizeof(test_block_mxfp4);
    default: return 0;
    }
}

static float mxfp4_value(uint8_t code) {
    static const float values[16] = {
         0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    return values[code & 15u];
}

static float e8m0_scale(uint8_t e) {
    const uint32_t bits = e == 0 ? UINT32_C(0x00400000)
                                 : (uint32_t)e << 23;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float table_value(const void *table, uint32_t type,
                         uint64_t row, uint32_t dim) {
    const uint64_t row_bytes = table_row_bytes(type);
    const uint8_t *row_ptr = (const uint8_t *)table + row * row_bytes;
    if (type == GGML_TYPE_F16) {
        return (float)((const _Float16 *)row_ptr)[dim];
    }
    const uint32_t block = dim / 32u;
    const uint32_t inner = dim % 32u;
    if (type == GGML_TYPE_Q8_0) {
        const test_block_q8_0 *qb =
            ((const test_block_q8_0 *)row_ptr) + block;
        return (float)qb->d * (float)qb->qs[inner];
    }
    const test_block_mxfp4 *qb =
        ((const test_block_mxfp4 *)row_ptr) + block;
    const uint8_t packed = qb->qs[inner & 15u];
    const uint8_t code = (packed >> (inner < 16u ? 0u : 4u)) & 15u;
    return e8m0_scale(qb->e) * mxfp4_value(code);
}

static void fill_table(void *table, uint32_t type) {
    const uint64_t row_bytes = table_row_bytes(type);
    memset(table, 0, (size_t)(PLE_TABLE_ROWS * row_bytes));
    for (uint32_t row = 0; row < PLE_TABLE_ROWS; row++) {
        uint8_t *row_ptr = (uint8_t *)table + (uint64_t)row * row_bytes;
        if (type == GGML_TYPE_F16) {
            _Float16 *values = (_Float16 *)row_ptr;
            for (uint32_t d = 0; d < PLE_HEAD_DIM; d++) {
                const int32_t centered =
                    (int32_t)(((uint64_t)row * 23u + d * 11u) % 251u) - 125;
                values[d] = (_Float16)((float)centered / 64.0f);
            }
        } else if (type == GGML_TYPE_Q8_0) {
            test_block_q8_0 *blocks = (test_block_q8_0 *)row_ptr;
            for (uint32_t b = 0; b < PLE_HEAD_DIM / 32u; b++) {
                blocks[b].d = (_Float16)(0.03125f * (float)(1u + row % 4u));
                for (uint32_t d = 0; d < 32u; d++) {
                    blocks[b].qs[d] = (int8_t)
                        ((int32_t)((row * 7u + b * 5u + d) % 31u) - 15);
                }
            }
        } else {
            test_block_mxfp4 *blocks = (test_block_mxfp4 *)row_ptr;
            for (uint32_t b = 0; b < PLE_HEAD_DIM / 32u; b++) {
                blocks[b].e = (uint8_t)(126u + (row + b) % 3u);
                for (uint32_t d = 0; d < 16u; d++) {
                    const uint8_t lo = (uint8_t)((row + b + d) & 15u);
                    const uint8_t hi = (uint8_t)((row * 3u + b + d + 5u) & 15u);
                    blocks[b].qs[d] = (uint8_t)(lo | (uint8_t)(hi << 4));
                }
            }
        }
    }
}

static uint64_t hash_gid(uint32_t token, uint32_t previous,
                         uint32_t previous2, uint32_t head) {
    const uint64_t pair =
        ((uint64_t)token * multipliers[0]) ^
        ((uint64_t)previous * multipliers[1]);
    const uint64_t hash = head < 8u ? pair :
        pair ^ ((uint64_t)previous2 * multipliers[2]);
    return hash % PLE_ROWS_PER_HEAD + (uint64_t)head * PLE_ROWS_PER_HEAD;
}

static int build_reference(float *out, const void *table, uint32_t type) {
    uint32_t history[2] = {PLE_EOS, PLE_EOS};
    uint32_t valid = 0;
    for (uint32_t t = 0; t < PLE_TOKENS; t++) {
        const uint32_t current = tokens[t];
        const uint32_t previous = valid >= 1u ? history[1] : PLE_EOS;
        const uint32_t previous2 = valid >= 2u ? history[0] : PLE_EOS;
        for (uint32_t head = 0; head < PLE_HEADS; head++) {
            const uint64_t gid = hash_gid(current, previous, previous2, head);
            if (gid != golden_gids[t][head]) {
                fprintf(stderr,
                        "Qwen4Exp PLE hash golden mismatch token=%u head=%u got=%llu expected=%u\n",
                        t, head, (unsigned long long)gid, golden_gids[t][head]);
                return 0;
            }
            for (uint32_t d = 0; d < PLE_HEAD_DIM; d++) {
                out[(uint64_t)t * PLE_EMBED_DIM +
                    (uint64_t)head * PLE_HEAD_DIM + d] =
                    table_value(table, type, gid, d);
            }
        }
        if (current == PLE_EOS) {
            valid = 0;
            history[0] = history[1] = PLE_EOS;
        } else if (valid == 0) {
            history[1] = current;
            valid = 1;
        } else {
            history[0] = history[1];
            history[1] = current;
            valid = 2;
        }
    }
    return 1;
}

static int compare_exact(const char *name, const float *actual,
                         const float *expected, uint64_t count) {
    for (uint64_t i = 0; i < count; i++) {
        if (memcmp(&actual[i], &expected[i], sizeof(float)) != 0) {
            fprintf(stderr,
                    "Qwen4Exp PLE %s mismatch at %llu: got=%g expected=%g\n",
                    name, (unsigned long long)i, actual[i], expected[i]);
            return 0;
        }
    }
    fprintf(stderr, "Qwen4Exp PLE %s bit exact (%llu floats)\n",
            name, (unsigned long long)count);
    return 1;
}

static int run_table_test(const void *model, uint64_t model_size,
                          uint64_t table_offset, uint32_t type,
                          const char *label) {
    const uint64_t value_count = (uint64_t)PLE_TOKENS * PLE_EMBED_DIM;
    const void *table = (const uint8_t *)model + table_offset;
    uint64_t sizes[PLE_HEADS];
    uint64_t offsets[PLE_HEADS];
    for (uint32_t h = 0; h < PLE_HEADS; h++) {
        sizes[h] = PLE_ROWS_PER_HEAD;
        offsets[h] = (uint64_t)h * PLE_ROWS_PER_HEAD;
    }
    float *expected = malloc((size_t)value_count * sizeof(float));
    float *chunk = malloc((size_t)value_count * sizeof(float));
    float *step = malloc((size_t)value_count * sizeof(float));
    float *rollback_a = malloc(2u * PLE_EMBED_DIM * sizeof(float));
    float *rollback_b = malloc(2u * PLE_EMBED_DIM * sizeof(float));
    if (!expected || !chunk || !step || !rollback_a || !rollback_b) {
        fprintf(stderr, "Qwen4Exp PLE host allocation failed\n");
        return 0;
    }
    int ok = build_reference(expected, table, type);

    ds4_gpu_tensor *ids = ds4_gpu_tensor_alloc(PLE_TOKENS * sizeof(uint32_t));
    ds4_gpu_tensor *chunk_out = ds4_gpu_tensor_alloc(value_count * sizeof(float));
    ds4_gpu_tensor *step_out = ds4_gpu_tensor_alloc(value_count * sizeof(float));
    ds4_gpu_tensor *chunk_state = ds4_gpu_tensor_alloc(4u * sizeof(uint32_t));
    ds4_gpu_tensor *step_state = ds4_gpu_tensor_alloc(4u * sizeof(uint32_t));
    const uint32_t zero_state[4] = {PLE_EOS, PLE_EOS, 0, 0};
    ok = ok && ids && chunk_out && step_out && chunk_state && step_state;
    ok = ok && ds4_gpu_tensor_write(ids, 0, tokens, sizeof(tokens));
    ok = ok && ds4_gpu_tensor_write(chunk_state, 0, zero_state,
                                    sizeof(zero_state));
    ok = ok && ds4_gpu_tensor_write(step_state, 0, zero_state,
                                    sizeof(zero_state));
    ok = ok && ds4_gpu_qwen4_exp_ple_hash_gather_tensor(
        chunk_out, chunk_state, ids, PLE_TOKENS, model, model_size,
        table_offset, PLE_TABLE_ROWS, type, multipliers, sizes, offsets, PLE_EOS);

    const uint64_t row_bytes = (uint64_t)PLE_EMBED_DIM * sizeof(float);
    for (uint32_t t = 0; t < PLE_TOKENS && ok; t++) {
        ds4_gpu_tensor *id_view = ds4_gpu_tensor_view(
            ids, (uint64_t)t * sizeof(uint32_t), sizeof(uint32_t));
        ds4_gpu_tensor *out_view = ds4_gpu_tensor_view(
            step_out, (uint64_t)t * row_bytes, row_bytes);
        ok = id_view && out_view &&
             ds4_gpu_qwen4_exp_ple_hash_gather_tensor(
                 out_view, step_state, id_view, 1, model, model_size,
                 table_offset, PLE_TABLE_ROWS, type, multipliers,
                 sizes, offsets, PLE_EOS);
        ds4_gpu_tensor_free(out_view);
        ds4_gpu_tensor_free(id_view);
    }
    ok = ok && ds4_gpu_tensor_read(chunk_out, 0, chunk,
                                   value_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_read(step_out, 0, step,
                                   value_count * sizeof(float));
    ok = ok && compare_exact(label, chunk, expected, value_count);
    if (ok && memcmp(chunk, step, value_count * sizeof(float)) != 0) {
        fprintf(stderr, "Qwen4Exp PLE %s chunk-vs-step mismatch\n", label);
        ok = 0;
    } else if (ok) {
        fprintf(stderr, "Qwen4Exp PLE %s chunk-vs-step bit exact\n", label);
    }
    uint32_t chunk_state_host[4] = {0};
    uint32_t step_state_host[4] = {0};
    ok = ok && ds4_gpu_tensor_read(chunk_state, 0, chunk_state_host,
                                   sizeof(chunk_state_host));
    ok = ok && ds4_gpu_tensor_read(step_state, 0, step_state_host,
                                   sizeof(step_state_host));
    const uint32_t final_state[4] = {22, 23, 2, 0};
    if (ok && (memcmp(chunk_state_host, final_state, sizeof(final_state)) != 0 ||
               memcmp(step_state_host, final_state, sizeof(final_state)) != 0)) {
        fprintf(stderr, "Qwen4Exp PLE %s final token state mismatch\n", label);
        ok = 0;
    }

    /* Replay a suffix from the serialized four-word state.  The production
     * speculative frontier performs the same byte copy GPU-to-GPU while its
     * command batch is active; this standalone test uses read/write because
     * the public tensor-copy helper intentionally requires such a batch. */
    ok = ok && ds4_gpu_tensor_write(step_state, 0, zero_state,
                                    sizeof(zero_state));
    ds4_gpu_tensor *prefix_ids = ds4_gpu_tensor_view(
        ids, 0, 5u * sizeof(uint32_t));
    ds4_gpu_tensor *prefix_out = ds4_gpu_tensor_view(
        step_out, 0, 5u * row_bytes);
    ds4_gpu_tensor *suffix_ids = ds4_gpu_tensor_view(
        ids, 5u * sizeof(uint32_t), 2u * sizeof(uint32_t));
    ds4_gpu_tensor *suffix_out = ds4_gpu_tensor_view(
        step_out, 5u * row_bytes, 2u * row_bytes);
    ok = ok && prefix_ids && prefix_out && suffix_ids && suffix_out;
    ok = ok && ds4_gpu_qwen4_exp_ple_hash_gather_tensor(
        prefix_out, step_state, prefix_ids, 5, model, model_size,
        table_offset, PLE_TABLE_ROWS, type, multipliers, sizes, offsets, PLE_EOS);
    uint32_t snapshot_host[4] = {0};
    ok = ok && ds4_gpu_tensor_read(step_state, 0, snapshot_host,
                                   sizeof(snapshot_host));
    ok = ok && ds4_gpu_qwen4_exp_ple_hash_gather_tensor(
        suffix_out, step_state, suffix_ids, 2, model, model_size,
        table_offset, PLE_TABLE_ROWS, type, multipliers, sizes, offsets, PLE_EOS);
    ok = ok && ds4_gpu_tensor_read(suffix_out, 0, rollback_a,
                                   2u * row_bytes);
    ok = ok && ds4_gpu_tensor_write(step_state, 0, snapshot_host,
                                    sizeof(snapshot_host));
    ok = ok && ds4_gpu_qwen4_exp_ple_hash_gather_tensor(
        suffix_out, step_state, suffix_ids, 2, model, model_size,
        table_offset, PLE_TABLE_ROWS, type, multipliers, sizes, offsets, PLE_EOS);
    ok = ok && ds4_gpu_tensor_read(suffix_out, 0, rollback_b,
                                   2u * row_bytes);
    ok = ok && compare_exact("snapshot replay", rollback_a, rollback_b,
                             2u * PLE_EMBED_DIM);

    ds4_gpu_tensor_free(suffix_out);
    ds4_gpu_tensor_free(suffix_ids);
    ds4_gpu_tensor_free(prefix_out);
    ds4_gpu_tensor_free(prefix_ids);
    ds4_gpu_tensor_free(step_state);
    ds4_gpu_tensor_free(chunk_state);
    ds4_gpu_tensor_free(step_out);
    ds4_gpu_tensor_free(chunk_out);
    ds4_gpu_tensor_free(ids);
    free(rollback_b);
    free(rollback_a);
    free(step);
    free(chunk);
    free(expected);
    return ok;
}

int main(void) {
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t f16_offset = 0;
    const uint64_t f16_bytes =
        PLE_TABLE_ROWS * table_row_bytes(GGML_TYPE_F16);
    const uint64_t q8_offset = align_up(f16_bytes, page);
    const uint64_t q8_bytes =
        PLE_TABLE_ROWS * table_row_bytes(GGML_TYPE_Q8_0);
    const uint64_t mxfp4_offset = align_up(q8_offset + q8_bytes, page);
    const uint64_t mxfp4_bytes =
        PLE_TABLE_ROWS * table_row_bytes(GGML_TYPE_MXFP4);
    const uint64_t model_size = align_up(mxfp4_offset + mxfp4_bytes, page);
    void *model = NULL;
    if (posix_memalign(&model, (size_t)page, (size_t)model_size) != 0) {
        fprintf(stderr, "Qwen4Exp PLE test model allocation failed\n");
        return 1;
    }
    memset(model, 0, (size_t)model_size);
    fill_table((uint8_t *)model + f16_offset, GGML_TYPE_F16);
    fill_table((uint8_t *)model + q8_offset, GGML_TYPE_Q8_0);
    fill_table((uint8_t *)model + mxfp4_offset, GGML_TYPE_MXFP4);

    int ok = ds4_gpu_init() && ds4_gpu_set_model_map(model, model_size);
    ok = ok && run_table_test(model, model_size, f16_offset,
                              GGML_TYPE_F16, "F16 gather");
    ok = ok && run_table_test(model, model_size, q8_offset,
                              GGML_TYPE_Q8_0, "Q8_0 gather");
    ok = ok && run_table_test(model, model_size, mxfp4_offset,
                              GGML_TYPE_MXFP4, "MXFP4 gather");
    ds4_gpu_cleanup();
    free(model);

    if (!ok) {
        fprintf(stderr, "Qwen4Exp PLE Metal tests FAILED\n");
        return 1;
    }
    fprintf(stderr, "Qwen4Exp PLE Metal tests PASS\n");
    return 0;
}
