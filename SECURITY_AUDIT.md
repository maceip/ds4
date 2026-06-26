# DS4 Inference Engine Security Audit

**Scope:** `ds4.c` and `ds4.h` — the core ~26k-line inference engine  
**Date:** 2026-06-26  
**Severity threshold:** MEDIUM and above  

---

## Executive Summary

The DS4 inference engine demonstrates generally good defensive coding for a
C codebase of this size.  GGUF parsing uses a bounded cursor abstraction that
prevents most reads beyond the mmap, tensor offsets are validated against the
file size, and integer overflow is checked in several critical size calculations.

However, the audit identified **8 findings at MEDIUM severity or higher**.  The
most impactful are related to GGUF file parsing (the primary untrusted-input
surface), the instance lock file, and potential memory-exhaustion denial-of-service
through unchecked allocation sizes derived from attacker-controlled GGUF header
fields.

---

## Findings

### FINDING 1 — GGUF metadata/tensor count can trigger memory-exhaustion DoS

**File:** `ds4.c`  
**Lines:** 1838, 1868, 1960–1961  
**Severity:** MEDIUM  
**Category:** Denial of Service / Resource Exhaustion  

**Description:**  
`n_kv` and `n_tensors` are read as raw `uint64_t` values from the GGUF header
and used directly as allocation counts:

```c
// line 1838
m->kv = calloc((size_t)m->n_kv, sizeof(m->kv[0]));

// line 1868
m->tensors = calloc((size_t)m->n_tensors, sizeof(m->tensors[0]));
```

A crafted GGUF file can set `n_kv` or `n_tensors` to a very large value (e.g.
`2^60`).  The `calloc` will attempt to allocate an enormous amount of memory.
While `calloc` may return `NULL` (handled by `ds4_die`), on Linux with
overcommit enabled, the allocation may succeed but later cause OOM-killer
invocations or swap thrashing, effectively denying service to the entire
machine.

There is no upper-bound check that `n_kv` or `n_tensors` is reasonable
relative to the file size.  For example, each metadata KV entry requires at
minimum ~13 bytes (key-length prefix + type + smallest value), so `n_kv`
cannot exceed `file_size / 13`.

**Exploitability:** An attacker supplies a malicious GGUF model file.  This
is the primary untrusted-input surface.  Any user or operator who downloads a
GGUF from an untrusted source (model-sharing sites, p2p, etc.) is affected.

**Recommendation:**  
Add sanity checks after parsing the GGUF header:

```c
if (m->n_kv > m->size / 13 || m->n_tensors > m->size / 16)
    ds4_die("GGUF header claims more metadata/tensors than the file could hold");
```

---

### FINDING 2 — `next_pow2` infinite loop on large hash table sizes from crafted GGUF

**File:** `ds4.c`  
**Lines:** 20420–20424, 20427  
**Severity:** MEDIUM  
**Category:** Denial of Service (CPU hang)  

**Description:**  
The `next_pow2` function uses a left-shift loop:

```c
static uint64_t next_pow2(uint64_t n) {
    uint64_t p = 1;
    while (p < n) p <<= 1;
    return p;
}
```

`table_init` calls it with `expected * 2 + 16`:

```c
static void table_init(str_i32_table *t, uint64_t expected) {
    t->cap = next_pow2(expected * 2 + 16);
```

If a malicious GGUF declares a tokenizer with `tokens.len` close to
`UINT64_MAX / 2` (the `len` field is a uint64 read from GGUF metadata), then
`expected * 2 + 16` wraps around to a small value or `expected * 2` itself
overflows.  Additionally, if `expected * 2 + 16 > 2^63`, the `next_pow2`
loop will shift `p` through `2^63` (which is `INT64_MIN` as signed, but as
`uint64_t` is fine), then to `0` via overflow, and then loop forever
(`0 < n` is always true for any `n > 0`).

**Attack path:** The `tokens.len` and `merges.len` fields are uint64 values
read from the GGUF tokenizer arrays.  While `tokens.len` is bounded by
`INT32_MAX` (line 20952), `merges.len` has no upper bound check (line 20955–20958).

**Exploitability:** A crafted GGUF with a large `merges.len` causes a CPU
hang during model load.  The process becomes unresponsive.

**Recommendation:**  
1. Add a bounds check on `merges.len` similar to `tokens.len`.
2. Use bit manipulation instead of a loop: `p = 1ULL << (64 - __builtin_clzll(n - 1))` with appropriate edge-case guards.

---

### FINDING 3 — `/tmp/ds4.lock` symlink attack (local privilege escalation / DoS)

**File:** `ds4.c`  
**Lines:** 21879–21921  
**Severity:** MEDIUM  
**Category:** TOCTOU / Symlink attack  

**Description:**  
The instance lock file defaults to `/tmp/ds4.lock`, a world-writable directory:

```c
const char *path = getenv("DS4_LOCK_FILE");
if (!path || !path[0]) path = "/tmp/ds4.lock";

const int fd = open(path, O_RDWR | O_CREAT, 0600);
```

The `open()` call follows symlinks.  A local attacker can create a symlink at
`/tmp/ds4.lock` pointing to an arbitrary file (e.g. `/etc/crontab`, a user's
`.bashrc`, or a critical system file).  When DS4 runs (potentially as a
different user, or as root in some deployment configurations), it will:

1. Open the target file via the symlink with `O_RDWR | O_CREAT`.
2. Call `ftruncate(fd, 0)` (line 21913), **destroying the target file's contents**.
3. Write `dprintf(fd, "%ld\n", (long)getpid())` into the file.

This is a classic `/tmp` symlink race.  Even if the attacker cannot escalate
privileges, they can cause denial of service by pointing the symlink at DS4's
own model file or other critical data.

**Exploitability:** Requires local access to the `/tmp` directory on the same
machine.  Exploitable in any multi-user or containerized environment where DS4
runs.  The `DS4_LOCK_FILE` environment variable override does not help since
the default is always `/tmp`.

**Recommendation:**  
Use `O_NOFOLLOW` to prevent following symlinks, and create the file in a
user-owned directory (e.g. `$XDG_RUNTIME_DIR` or `$HOME/.ds4/`):

```c
const int fd = open(path, O_RDWR | O_CREAT | O_NOFOLLOW, 0600);
```

Or use a directory under `/run/user/$(id -u)/` which is per-user and not
world-writable.

---

### FINDING 4 — `/tmp/ds4-session-payload.XXXXXX` temp file in world-writable directory

**File:** `ds4.c`  
**Lines:** 22895–22900  
**Severity:** MEDIUM  
**Category:** Information Disclosure / Symlink attack  

**Description:**  
Session payloads are staged through a temporary file in `/tmp`:

```c
char tmpl[] = "/tmp/ds4-session-payload.XXXXXX";
int fd = mkstemp(tmpl);
```

While `mkstemp` itself is safe against symlink races (it creates the file
with `O_EXCL`), the predictable prefix `ds4-session-payload` in a
world-readable directory creates two risks:

1. **Information disclosure:** The session payload contains the full KV cache,
   logits, and token history.  On some systems, the file permissions from
   `mkstemp` may inherit a permissive umask (the mask is applied to `0600`,
   but some environments override this).  A local attacker can monitor `/tmp`
   for new files matching the pattern.

2. **Temp file exhaustion DoS:** An attacker can pre-create many files matching
   the `XXXXXX` pattern space to cause `mkstemp` to fail.

**Attack path:** Local attacker on the same machine monitors `/tmp` for session
payload files.  The payload contains the user's conversation tokens, which may
include sensitive prompts.

**Recommendation:**  
Use a private directory (e.g. under `$TMPDIR`, `$XDG_RUNTIME_DIR`, or a
subdirectory created by the server with restricted permissions).

---

### FINDING 5 — Integer overflow in `bpe_rank` length calculation

**File:** `ds4.c`  
**Lines:** 20673–20676  
**Severity:** LOW-MEDIUM  
**Category:** Integer Overflow / Heap Buffer Overflow  

**Description:**  
The `bpe_rank` function computes a concatenated key length:

```c
static int bpe_rank(const ds4_vocab *vocab, const owned_str *a, const owned_str *b) {
    uint64_t len = a->len + 1 + b->len;
    char stack[512];
    char *buf = len <= sizeof(stack) ? stack : xmalloc((size_t)len);

    memcpy(buf, a->ptr, (size_t)a->len);
    buf[a->len] = ' ';
    memcpy(buf + a->len + 1, b->ptr, (size_t)b->len);
```

If `a->len + 1 + b->len` overflows `uint64_t` (theoretically requires
>= 2^63 sized strings), or more practically if the `(size_t)` cast truncates
on a 32-bit platform, the allocated buffer could be smaller than expected,
causing a heap buffer overflow.

In practice, `a` and `b` are `owned_str` values derived from `byte_encode`
which allocates `in.len * 4 + 1` bytes (line 20640).  The `in.len` comes
from the GGUF-declared token strings.  On a 64-bit platform, overflow requires
impractically large token strings, but the code lacks an explicit guard.

**Exploitability:** Very difficult on 64-bit.  Could matter on 32-bit
platforms if the engine is ever ported.

**Recommendation:**  
Add an overflow check: `if (a->len > SIZE_MAX - b->len - 1) ds4_die(...)`.

---

### FINDING 6 — GPU allocation failures are silently aggregated, not individually reported

**File:** `ds4.c`  
**Lines:** 10849–10997, 11054–11097  
**Severity:** MEDIUM  
**Category:** Insufficient Error Handling / Undefined Behavior  

**Description:**  
The `metal_graph_alloc_raw_cap` function performs approximately 80+ GPU tensor
allocations sequentially.  Each `ds4_gpu_tensor_alloc()` call may return `NULL`
on failure, but the results are not checked individually.  Instead, all
allocations proceed, and a single aggregate NULL check happens at the end
(lines 11054–11097):

```c
const bool ok = state_init_ok && layer_cache_ok &&
                g->cur_hc && g->flat_hc && g->hc_mix && ...
```

**Problems with this approach:**

1. **NULL pointer dereference between allocations:** Between lines 10849 and
   11054, several operations use earlier allocations that may have returned
   NULL.  For example, `ds4_gpu_tensor_view` (lines 10853–10859) is called on
   `g->hc_split` which may be NULL.  Similarly, `metal_tensor_fill_f32`
   (line 10891) is called on tensors that may be NULL — though it does check
   `if (g->layer_attn_state_kv[il])`.

2. **Partial GPU state on failure:** If the aggregate check fails at line 11097,
   `metal_graph_free` is called, which must correctly handle a mix of NULL and
   non-NULL tensors.

3. **Silent failure mode:** The caller (`ds4_session_create`) gets a boolean
   `false` return with no diagnostic about which allocation failed or how much
   GPU memory was available.

**Attack path:** An attacker cannot directly control GPU memory pressure in most
deployment models, but in shared-GPU environments (cloud inference, multi-tenant
setups), a co-tenant could exhaust GPU memory to trigger this path.

**Recommendation:**  
Check each allocation immediately and return with a diagnostic error on the
first failure.  This also avoids wasting GPU memory on subsequent allocations
that will be freed anyway.

---

### FINDING 7 — GGUF `general.alignment` can be set to zero (division in `align_up`)

**File:** `ds4.c`  
**Lines:** 1498–1501, 1854–1858  
**Severity:** LOW-MEDIUM  
**Category:** Division by Zero / Undefined Behavior  

**Description:**  
The `align_up` function computes a modulo:

```c
static uint64_t align_up(uint64_t value, uint64_t alignment) {
    uint64_t rem = value % alignment;
    return rem == 0 ? value : value + alignment - rem;
}
```

The `general.alignment` metadata value is used as the alignment for tensor
data:

```c
if (cursor_u32(&tmp, &alignment) && alignment != 0) {
    m->alignment = alignment;
}
```

While the zero check (line 1856) guards the initial assignment, the default
`m->alignment = 32` (line 1841) means a zero value in the GGUF just keeps
the default.  However, `align_up` itself has no guard against a zero
`alignment` argument.  If any other code path calls `align_up` with a zero
divisor, undefined behavior (division by zero) occurs.

**Exploitability:** Currently low because the alignment assignment is guarded.
This is a latent risk if `align_up` is reused elsewhere.

**Recommendation:**  
Add `if (alignment == 0) return value;` at the top of `align_up`.

---

### FINDING 8 — Token ID bounds not checked in `ds4_session_eval` path

**File:** `ds4.c`  
**Lines:** 25569–25602, 25660  
**Severity:** MEDIUM  
**Category:** Out-of-Bounds Memory Access  

**Description:**  
The `ds4_session_eval` function accepts an arbitrary `int token` from the
caller and passes it directly through to the forward pass.  While the
embedding function `embed_token_f16` (line 4295–4299) does check bounds:

```c
static void embed_token_f16(const ds4_model *m, const ds4_weights *w, int token, float *out) {
    ds4_tensor *te = w->token_embd;
    if (token < 0 || (uint64_t)token >= te->dim[1]) {
        ds4_die("token id is outside the embedding table");
    }
```

This check calls `ds4_die` which terminates the entire process.  For a server
deployment (ds4-server), a single malicious API request with an out-of-range
token ID kills the server process, denying service to all other clients.

The `ds4_session_eval` public API (line 25660) does not validate the token
range itself, relying on the fatal `ds4_die` inside the forward pass.

Similarly, `ds4_session_sync` passes caller-provided token arrays into the
prefill path.  Any out-of-range token in the prompt triggers a fatal abort
rather than a recoverable error.

**Attack path:** An attacker sends an API request (e.g. via the HTTP server)
with a crafted token ID outside `[0, n_vocab)`.  The server process crashes.

**Recommendation:**  
1. Check token bounds in `ds4_session_eval` and `ds4_session_sync` before
   entering the forward pass, returning an error code instead of aborting.
2. Change `embed_token_f16` to return an error rather than calling `ds4_die`.

---

## Summary Table

| # | Severity | Category | Location | Description |
|---|----------|----------|----------|-------------|
| 1 | MEDIUM | DoS / Resource Exhaustion | L1838,1868 | Unbounded n_kv/n_tensors calloc from GGUF header |
| 2 | MEDIUM | DoS / CPU Hang | L20420 | `next_pow2` infinite loop via GGUF merges.len overflow |
| 3 | MEDIUM | Symlink Attack | L21879 | `/tmp/ds4.lock` follows symlinks; arbitrary file truncation |
| 4 | MEDIUM | Info Disclosure | L22895 | Session payload in world-writable `/tmp` |
| 5 | LOW-MEDIUM | Integer Overflow | L20673 | `bpe_rank` length calc can overflow on 32-bit |
| 6 | MEDIUM | Error Handling | L10849 | GPU alloc failures not checked individually |
| 7 | LOW-MEDIUM | Div-by-Zero | L1498 | `align_up` has no zero-alignment guard |
| 8 | MEDIUM | Process Crash DoS | L25660 | Out-of-range token ID kills server via `ds4_die` |

---

## Areas Reviewed Without Significant Findings

### GGUF Tensor Offset Validation (GOOD)
Lines 1904–1912 properly check for both addition overflow and bounds:
```c
if (t->rel_offset > UINT64_MAX - m->tensor_data_pos)
    ds4_die("tensor offset overflow");
t->abs_offset = m->tensor_data_pos + t->rel_offset;
if (t->bytes != 0 &&
    (t->abs_offset > m->size || t->bytes > m->size - t->abs_offset))
    ds4_die("tensor points outside GGUF file");
```

### Tensor Element Count Overflow (GOOD)
Line 1883 checks for multiplication overflow before computing element count:
```c
if (t->dim[d] != 0 && t->elements > UINT64_MAX / t->dim[d])
    ds4_die("tensor element count overflow");
```

### Tensor Byte Count Overflow (GOOD)
Line 1687 in `tensor_nbytes` checks for overflow:
```c
if (blocks > UINT64_MAX / info->block_bytes) return false;
```

### GGUF Metadata Array Nesting (GOOD)
Line 1633 limits recursion depth to prevent stack overflow:
```c
if (depth > 8) { cursor_error(c, "metadata array nesting is too deep"); return false; }
```

### Cursor Bounds Checking (GOOD)
The `cursor_has` function (line 1459) correctly prevents integer overflow in
the bounds check by restructuring the comparison:
```c
if (n > c->size || c->pos > c->size - n) // avoids pos+n overflow
```

### Session Payload Validation (GOOD)
`ds4_session_load_payload` validates magic, version, layout parameters, and
remaining byte count.  Compressed/raw row counts are bounds-checked against
capacity before being used to size reads.

### Tokenizer BPE (GOOD)
The BPE implementation uses dynamic arrays with `xrealloc` (which aborts on
OOM) and operates on heap-allocated `owned_str` values, avoiding stack buffer
overflows.

### Mmap Region Size (GOOD)
The mmap uses `fstat` to determine file size and maps exactly that many bytes.
The cursor abstraction prevents reads beyond the mapped region.

---

## Threat Model Notes

The primary untrusted-input surface is **GGUF model files**.  Users commonly
download GGUF files from model-sharing platforms (Hugging Face, etc.), and a
malicious model file is the most realistic attack vector.  Findings 1, 2, 5,
and 7 are in this attack surface.

The secondary surface is **API requests** to the ds4 HTTP server, where
crafted token IDs or prompts can trigger crashes (Finding 8).

The tertiary surface is **local system access** for the lock file and temp
file issues (Findings 3 and 4).
