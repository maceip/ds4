# Security Audit: `ds4_server.c`

**Date:** 2026-06-26
**Scope:** HTTP parsing, JSON parsing, memory safety, DoS, race conditions, information disclosure, request smuggling
**Exclusions:** Missing auth/CORS wildcard/DNS rebinding/TLS (known/accepted)

---

## Findings

### 1. `buf_reserve` — Integer Overflow in Capacity Doubling Loop

**Severity: MEDIUM**
**Location:** Lines 118–119

```c
size_t cap = b->cap ? b->cap * 2 : 256;
while (cap < need) cap *= 2;
```

**Description:**
The capacity-doubling loop can overflow `size_t` and wrap to zero (or a small value) on platforms where `SIZE_MAX + 1 == 0`. For example, if `b->cap` is `2^63` (on 64-bit), the first doubling wraps to 0, causing `cap < need` to remain true indefinitely — creating an **infinite loop** (DoS). Even if the loop terminates at a wrapped-around small value, `xrealloc(b->ptr, cap)` allocates a tiny buffer while `b->cap` records the wrapped value, leading to a subsequent **heap buffer overflow** in `buf_append`.

**Attack path:**
The initial overflow guard at line 115 (`add > SIZE_MAX - b->len - 1`) prevents the *sum* from overflowing, but does not prevent the doubling loop from wrapping. A sufficiently large `need` value (e.g., `need ≈ SIZE_MAX/2 + 2`) passes the guard but overflows during doubling. In practice, this requires allocating >4 GiB of request data on 32-bit platforms, or >8 EiB on 64-bit. On 64-bit systems, `xrealloc` will fail before reaching such sizes, so exploitability is limited to 32-bit builds or unusual memory configurations.

**Exploitability: LOW** on 64-bit (memory exhaustion first), **MEDIUM** on 32-bit.

**Recommendation:** Add an overflow check in the doubling loop:
```c
while (cap < need) {
    if (cap > SIZE_MAX / 2) die("buffer overflow");
    cap *= 2;
}
```

---

### 2. `content_length` — First-Match Semantics Enable Request Smuggling via Duplicate Headers

**Severity: MEDIUM**
**Location:** Lines 10988–11003

```c
static long content_length(const char *h, size_t n) {
    // ...
    if (len >= 15 && strncasecmp(line, "Content-Length:", 15) == 0) {
        const char *v = line + 15;
        while (v < line + len && isspace((unsigned char)*v)) v++;
        return strtol(v, NULL, 10);   // returns on FIRST match
    }
    // ...
    return 0;
}
```

**Description:**
The function returns the value of the **first** `Content-Length` header it encounters. RFC 9110 §8.6 requires that if multiple `Content-Length` headers are present with different values, the message must be rejected. This server silently uses the first value.

If a reverse proxy sits in front of this server and uses the **last** `Content-Length` header (as some proxies do), an attacker can send:
```
Content-Length: 0\r\n
Content-Length: 42\r\n
```
The proxy forwards 42 bytes of body, but ds4 reads 0 bytes of body. The remaining 42 bytes become the start of the **next** HTTP request on the same connection — a classic request smuggling attack.

**Attack path:**
Attacker → reverse proxy → ds4. Duplicate Content-Length headers with different values. Proxy uses last value, ds4 uses first. Leads to request smuggling (cache poisoning, auth bypass, request routing manipulation).

**Exploitability: MEDIUM** (requires a front-end proxy with last-match CL semantics; the `Connection: close` response header mitigates pipelining on the response side, but the attack targets the *request* parsing side).

**Recommendation:** Reject requests with multiple `Content-Length` headers bearing different values, or reject any request with duplicate `Content-Length` headers.

---

### 3. No `Transfer-Encoding` Handling — Request Smuggling via Chunked Encoding

**Severity: MEDIUM**
**Location:** `read_http_request`, lines 11005–11051

**Description:**
The server does not parse or reject `Transfer-Encoding: chunked` requests. If a client sends a chunked-encoded request, the server ignores `Transfer-Encoding` entirely and falls back to `Content-Length` (or 0 if absent). The chunked body is then either:
- Silently discarded (if `Content-Length: 0` or absent), or
- Partially read as raw chunked framing, corrupting the JSON body parse.

When a reverse proxy decodes chunked encoding before forwarding, this is safe. But when a proxy passes `Transfer-Encoding: chunked` through without decoding, the server and proxy disagree on message boundaries — the classic CL/TE desync.

**Attack path:**
Attacker sends `Transfer-Encoding: chunked` with a crafted chunked body. A forwarding proxy interprets the body via chunked framing; ds4 interprets it via Content-Length. This desync can smuggle a second request.

**Exploitability: MEDIUM** (depends on proxy configuration).

**Recommendation:** If `Transfer-Encoding` is present, either:
- Decode chunked encoding, or
- Reject the request with 501 Not Implemented.

---

### 4. `json_number` Accepts Non-JSON Number Formats via `strtod`

**Severity: MEDIUM**
**Location:** Lines 258–266

```c
static bool json_number(const char **p, double *out) {
    json_ws(p);
    char *end = NULL;
    double v = strtod(*p, &end);
    if (end == *p) return false;
    *p = end;
    *out = v;
    return true;
}
```

**Description:**
`strtod` accepts formats that are not valid JSON numbers:
- Hexadecimal floats: `0x1.0p10` → 1024.0
- Infinity: `inf`, `infinity` (locale-dependent)
- NaN: `nan`, `nan(...)` (locale-dependent)

When `strtod` returns `NaN` or `Inf`, these propagate into `r->temperature`, `r->top_p`, `r->min_p`, or `r->seed`. Passing `NaN`/`Inf` to downstream sampling functions may cause undefined behavior, assertion failures, or infinite loops in the inference engine.

**Attack path:**
```json
{"messages":[...], "temperature": nan, "top_p": inf}
```
These parse successfully and propagate to `ds4_session` sampling parameters. Impact depends on how the engine handles non-finite floats — at minimum it produces undefined sampling behavior; at worst it causes a crash or hang in the model decode loop.

**Exploitability: MEDIUM** (easy to trigger, impact depends on engine internals).

**Recommendation:** After `strtod`, reject non-finite values:
```c
if (!isfinite(v)) return false;
```

---

### 5. Unbounded Client Thread Creation — Resource Exhaustion DoS

**Severity: MEDIUM**
**Location:** Lines 11657–11688

```c
while (!g_stop_requested) {
    int fd = accept(lfd, NULL, NULL);
    // ...
    pthread_mutex_lock(&s.mu);
    s.clients++;
    pthread_mutex_unlock(&s.mu);
    pthread_t th;
    if (pthread_create(&th, NULL, client_main, ca) != 0) {
        // cleanup on failure
    }
    pthread_detach(th);
}
```

**Description:**
There is no limit on the number of concurrent client threads. Each accepted connection spawns a new `pthread` without checking `s.clients` against any maximum. An attacker can open thousands of connections simultaneously. Each thread consumes ~8 MiB of stack (default `pthread` stack size on Linux), plus per-request heap allocations. With 1000 connections, that's ~8 GiB of stack memory alone.

Even though the single worker thread serializes inference, the HTTP parsing and body reading in `client_main` runs in the per-client thread. An attacker can hold connections open (slowloris-style) while consuming thread/memory resources.

**Attack path:**
Attacker opens thousands of TCP connections. Each spawns a thread. The server exhausts virtual memory or hits the OS thread limit, preventing legitimate clients from connecting.

**Exploitability: HIGH** (trivial to execute, no authentication required).

**Recommendation:** Add a `max_clients` limit. Before creating a thread, check `s.clients < max_clients` and reject with HTTP 503 if exceeded.

---

### 6. Unbounded JSON Arrays/Objects in Request Body — Memory Exhaustion DoS

**Severity: MEDIUM**
**Location:** `parse_messages` (line 1598), `parse_stop` (line 927), `parse_tools_value` (line 1553), `parse_tool_calls_value` (line 1105)

**Description:**
While the HTTP body is capped at 64 MiB (`max_body`), the JSON parsing functions impose no limits on the number of array elements. A 64 MiB request body can contain millions of messages, tool definitions, or stop sequences. Each element triggers heap allocations (strings, structs), and the resulting data structures can amplify memory usage significantly beyond the 64 MiB input:
- Each `chat_msg` has multiple heap-allocated string fields
- Each `tool_call` has 3 string fields plus struct overhead
- `stop_list` entries are individually heap-allocated

A carefully crafted request with millions of tiny messages or tool definitions could consume hundreds of MiB or GiB of heap memory during parsing.

**Attack path:**
```json
{"messages": [{"role":"user","content":"a"},{"role":"user","content":"a"}, ... (millions)]}
```
Each message allocates role string, content string, and the `chat_msg` struct. With 1M messages, heap usage is ~100+ MiB for a ~20 MiB JSON input.

**Exploitability: MEDIUM** (bounded by 64 MiB input, but amplification factor is significant).

**Recommendation:** Impose reasonable limits on array sizes (e.g., max 10,000 messages, max 1,000 tools, max 100 stop sequences).

---

### 7. Lone UTF-16 Surrogates Encoded as Invalid UTF-8

**Severity: MEDIUM**
**Location:** Lines 235–243 (`json_string`) and lines 183–199 (`utf8_put`)

```c
case 'u': {
    *p -= 2;
    uint32_t cp = 0, lo = 0;
    if (!json_u16(p, &cp)) goto fail;
    if (cp >= 0xd800 && cp <= 0xdbff && json_u16(p, &lo) && lo >= 0xdc00 && lo <= 0xdfff) {
        cp = 0x10000u + ((cp - 0xd800u) << 10) + (lo - 0xdc00u);
    }
    utf8_put(&b, cp);
    break;
}
```

**Description:**
When a high surrogate (0xD800–0xDBFF) is not followed by a low surrogate, or when a low surrogate (0xDC00–0xDFFF) appears alone, the raw surrogate codepoint is passed to `utf8_put`, which encodes it as a 3-byte CESU-8 / WTF-8 sequence (ED A0 80 – ED BF BF). This is explicitly forbidden by the Unicode standard (surrogates are not valid scalar values) and produces bytes that are invalid UTF-8 per RFC 3629.

These invalid sequences propagate into the model prompt text and may:
- Cause the tokenizer to produce unexpected token sequences
- Corrupt KV cache prefix matching (a later client that properly escapes the same text will produce different bytes, breaking cache alignment)
- Trigger undefined behavior in downstream string processing that assumes valid UTF-8

**Attack path:**
```json
{"messages":[{"role":"user","content":"hello \uD800 world"}]}
```
The lone `\uD800` is encoded as ED A0 80, producing an invalid UTF-8 string used as the model prompt. If the tokenizer has assertions on valid UTF-8, this may crash. If it silently accepts it, the result is at least a cache-alignment problem.

**Exploitability: MEDIUM** (easy to trigger via any API endpoint that accepts string content).

**Recommendation:** Reject lone surrogates in `json_string`, or replace them with U+FFFD:
```c
if (cp >= 0xd800 && cp <= 0xdfff) goto fail; // reject
```

---

### 8. `content_length` — `strtol` on Non-NUL-Bounded Slice

**Severity: LOW–MEDIUM**
**Location:** Line 10998

```c
return strtol(v, NULL, 10);
```

**Description:**
`strtol` reads from `v` until it encounters a non-digit character. The pointer `v` points into the `buf b` which *is* NUL-terminated by `buf_append`, so in practice `strtol` will stop at `\r`, `\n`, or `\0`. However, the function conceptually operates on `line + 15` to `line + len`, and `strtol` may read past `line + len` if the line content is all digits with no trailing CR/LF. This is safe only because the buffer is always NUL-terminated by `buf_append` — a fragile invariant.

Additionally, `strtol` does not validate that *only* digits follow the whitespace. A header like `Content-Length: 10abc` would return 10 without error. While not exploitable on its own, it violates RFC strictness and could interact with proxies that parse the same header differently.

**Exploitability: LOW** (mitigated by NUL-termination invariant of `buf`).

**Recommendation:** Use a bounded parsing approach, e.g., `strtol` with `end` pointer validation, or manually parse the digits within the `[v, line+len)` range.

---

## Summary Table

| # | Issue | Severity | Exploitability | Category |
|---|-------|----------|----------------|----------|
| 1 | `buf_reserve` cap-doubling integer overflow | MEDIUM | Low (64-bit) / Medium (32-bit) | Memory Safety |
| 2 | Duplicate `Content-Length` first-match semantics | MEDIUM | Medium (needs proxy) | Request Smuggling |
| 3 | Missing `Transfer-Encoding` handling | MEDIUM | Medium (needs proxy) | Request Smuggling |
| 4 | `strtod` accepts NaN/Inf/hex floats | MEDIUM | Medium | Type Confusion / DoS |
| 5 | Unbounded client thread creation | MEDIUM | High | DoS |
| 6 | Unbounded JSON array sizes | MEDIUM | Medium | DoS |
| 7 | Lone UTF-16 surrogates → invalid UTF-8 | MEDIUM | Medium | Memory Safety / Data Integrity |
| 8 | `strtol` on non-bounded header value | LOW–MEDIUM | Low | Parsing Robustness |

## Positive Observations

Several security-relevant areas are handled well:

- **Header size limits:** `max_header = 64 * 1024` prevents unbounded header reading.
- **Body size limits:** `max_body = 64 * 1024 * 1024` caps request bodies.
- **`sscanf` field widths:** `%7s` and `%255s` match the fixed buffer sizes (8 and 256 bytes).
- **JSON nesting depth:** `JSON_MAX_NESTING = 256` prevents stack exhaustion from deeply nested JSON.
- **Error message sanitization:** All error messages pass through `json_escape` before being included in HTTP responses, preventing JSON injection.
- **Tool memory bounds:** `DS4_TOOL_MEMORY_MAX_BYTES` and `max_entries` limit memory growth.
- **Mutex discipline:** The `tool_mu` mutex is consistently used for all `tool_memory` and `live_tool_state` accesses from client threads. The worker thread is single-threaded, avoiding races on `session`/`engine` state.
- **Send timeout:** `DS4_SERVER_SEND_STALL_TIMEOUT_MS` prevents slow clients from blocking the worker indefinitely.
- **I/O timeouts:** `SO_RCVTIMEO` / `SO_SNDTIMEO` are set on client sockets.
- **Connection: close:** Every response includes `Connection: close`, mitigating persistent-connection-based attacks (though not request-side smuggling).
