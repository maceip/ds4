/* Focused Qwen4Exp GGUF metadata and fixed-weight binding contract tests. */

#define _POSIX_C_SOURCE 200809L

#include "ds4.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum {
    GGUF_VALUE_UINT32 = 4,
    GGUF_VALUE_FLOAT32 = 6,
    GGUF_VALUE_STRING = 8,
    GGUF_VALUE_ARRAY = 9,
    GGUF_VALUE_UINT64 = 10,
};

static void fail(const char *what) {
    fprintf(stderr, "test_qwen4_exp_loader: %s\n", what);
    exit(1);
}

static void write_bytes(FILE *fp, const void *ptr, size_t len) {
    if (len && fwrite(ptr, 1, len, fp) != len) fail("fixture write failed");
}

static void write_u32(FILE *fp, uint32_t value) {
    write_bytes(fp, &value, sizeof(value));
}

static void write_u64(FILE *fp, uint64_t value) {
    write_bytes(fp, &value, sizeof(value));
}

static void write_f32(FILE *fp, float value) {
    write_bytes(fp, &value, sizeof(value));
}

static void write_string(FILE *fp, const char *value) {
    const uint64_t len = (uint64_t)strlen(value);
    write_u64(fp, len);
    write_bytes(fp, value, (size_t)len);
}

static void write_key(FILE *fp, const char *key, uint32_t type) {
    write_string(fp, key);
    write_u32(fp, type);
}

static void write_kv_u32(FILE *fp, const char *key, uint32_t value) {
    write_key(fp, key, GGUF_VALUE_UINT32);
    write_u32(fp, value);
}

static void write_kv_u64(FILE *fp, const char *key, uint64_t value) {
    write_key(fp, key, GGUF_VALUE_UINT64);
    write_u64(fp, value);
}

static void write_kv_f32(FILE *fp, const char *key, float value) {
    write_key(fp, key, GGUF_VALUE_FLOAT32);
    write_f32(fp, value);
}

static void write_kv_string(FILE *fp, const char *key, const char *value) {
    write_key(fp, key, GGUF_VALUE_STRING);
    write_string(fp, value);
}

static void write_kv_u32_array(
        FILE           *fp,
        const char     *key,
        const uint32_t *values,
        uint64_t        count) {
    write_key(fp, key, GGUF_VALUE_ARRAY);
    write_u32(fp, GGUF_VALUE_UINT32);
    write_u64(fp, count);
    write_bytes(fp, values, (size_t)count * sizeof(values[0]));
}

static void write_kv_u64_array(
        FILE           *fp,
        const char     *key,
        const uint64_t *values,
        uint64_t        count) {
    write_key(fp, key, GGUF_VALUE_ARRAY);
    write_u32(fp, GGUF_VALUE_UINT64);
    write_u64(fp, count);
    write_bytes(fp, values, (size_t)count * sizeof(values[0]));
}

static void write_qwen4_exp_metadata_fixture(FILE *fp) {
    static const uint64_t ple_multipliers[3] = {
        UINT64_C(23703573157769),
        UINT64_C(20109073645365),
        UINT64_C(8052911324071),
    };
    static const uint64_t ple_offsets[16] = {
        0, 20000003, 40000026, 60000059,
        80000106, 100000165, 120000228, 140000297,
        160000374, 180000455, 200000548, 220000655,
        240000802, 260000955, 280001114, 300001275,
    };
    static const uint64_t ple_sizes[16] = {
        20000003, 20000023, 20000033, 20000047,
        20000059, 20000063, 20000069, 20000077,
        20000081, 20000093, 20000107, 20000147,
        20000153, 20000159, 20000161, 20000171,
    };
    uint32_t compress_ratios[48] = {0};
    for (uint32_t il = 3; il < 48; il += 4) compress_ratios[il] = 4;
    const uint32_t ple_layers[1] = {1};

    /* GGUF v3 header: one tiny tensor keeps the parser's zero-count path out
     * of this test; the loader gate intentionally never maps real weights. */
    write_u32(fp, UINT32_C(0x46554747));
    write_u32(fp, 3);
    write_u64(fp, 1);
    write_u64(fp, 37);

    write_kv_string(fp, "general.architecture", "qwen4exp");
    write_kv_u32(fp, "qwen4exp.block_count", 48);
    write_kv_u64(fp, "qwen4exp.context_length", UINT64_C(262144));
    write_kv_u32(fp, "qwen4exp.embedding_length", 2560);
    write_kv_u32(fp, "qwen4exp.vocab_size", 248320);
    write_kv_u32(fp, "qwen4exp.attention.head_count", 24);
    write_kv_u32(fp, "qwen4exp.attention.head_count_kv", 2);
    write_kv_u32(fp, "qwen4exp.attention.key_length", 256);
    write_kv_u32(fp, "qwen4exp.attention.value_length", 256);
    write_kv_u32(fp, "qwen4exp.rope.dimension_count", 64);
    write_kv_f32(fp, "qwen4exp.rope.freq_base", 10000000.0f);
    write_kv_u32(fp, "qwen4exp.full_attention_interval", 4);
    write_kv_u32(fp, "qwen4exp.hyper_connection.count", 4);
    write_kv_u32(fp, "qwen4exp.hyper_connection.low_rank", 320);
    write_kv_u32(fp, "qwen4exp.ssm.conv_kernel", 4);
    write_kv_u32(fp, "qwen4exp.ssm.inner_size", 6144);
    write_kv_u32(fp, "qwen4exp.ssm.state_size", 128);
    write_kv_u32(fp, "qwen4exp.ssm.time_step_rank", 48);
    write_kv_u32(fp, "qwen4exp.ssm.group_count", 16);
    write_kv_u32(fp, "qwen4exp.attention.indexer.head_count", 4);
    write_kv_u32(fp, "qwen4exp.attention.indexer.key_length", 128);
    write_kv_u32(fp, "qwen4exp.attention.indexer.top_k", 2048);
    write_kv_u32_array(fp, "qwen4exp.attention.compress_ratios",
                       compress_ratios, 48);
    write_kv_u32(fp, "qwen4exp.expert_count", 512);
    write_kv_u32(fp, "qwen4exp.expert_used_count", 10);
    write_kv_u32(fp, "qwen4exp.expert_feed_forward_length", 640);
    write_kv_u32(fp, "qwen4exp.expert_shared_feed_forward_length", 640);
    write_kv_f32(fp, "qwen4exp.attention.layer_norm_rms_epsilon", 1.0e-6f);
    write_kv_u32(fp, "qwen4exp.embedding_length_per_layer_input", 160);
    write_kv_u32_array(fp, "qwen4exp.ple.layers", ple_layers, 1);
    write_kv_u32(fp, "qwen4exp.ple.ngram_size", 3);
    write_kv_u32(fp, "qwen4exp.ple.heads_per_ngram", 8);
    write_kv_u32(fp, "qwen4exp.ple.conv_kernel", 4);
    write_kv_u32(fp, "qwen4exp.ple.eos_token_id", 248044);
    write_kv_u64_array(fp, "qwen4exp.ple.layer_multipliers",
                       ple_multipliers, 3);
    write_kv_u64_array(fp, "qwen4exp.ple.head_offsets", ple_offsets, 16);
    write_kv_u64_array(fp, "qwen4exp.ple.head_vocab_sizes", ple_sizes, 16);

    write_string(fp, "dummy");
    write_u32(fp, 1);
    write_u64(fp, 1);
    write_u32(fp, 0); /* F32 */
    write_u64(fp, 0);

    const long pos = ftell(fp);
    if (pos < 0) fail("fixture position query failed");
    const unsigned pad = (unsigned)((32 - ((uint64_t)pos % 32)) % 32);
    for (unsigned i = 0; i < pad; i++) {
        if (fputc(0, fp) == EOF) fail("fixture padding write failed");
    }
    write_f32(fp, 0.0f);
}

int main(void) {
    char path[] = "/tmp/ds4-qwen4-exp-loader.XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) fail(strerror(errno));
    FILE *fp = fdopen(fd, "wb");
    if (!fp) {
        close(fd);
        unlink(path);
        fail("fdopen failed");
    }

    write_qwen4_exp_metadata_fixture(fp);
    if (fclose(fp) != 0) {
        unlink(path);
        fail("fixture close failed");
    }

    if (!ds4_test_qwen4_exp_validate_metadata_file(path)) {
        unlink(path);
        fail("GGUF metadata validation hook rejected the Qwen4Exp fixture");
    }
    if (unlink(path) != 0) fail("fixture cleanup failed");

    if (!ds4_test_qwen4_exp_weight_binding()) {
        fail("fixed Qwen4Exp weight binding contract failed");
    }

    puts("test_qwen4_exp_loader: PASS");
    return 0;
}
