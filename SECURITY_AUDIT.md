# Security Audit Report: ds4_web.c and ds4_distributed.c

**Date:** 2026-06-26
**Scope:** WebSocket/CDP implementation in `ds4_web.c`, distributed protocol in `ds4_distributed.c`
**Severity threshold:** MEDIUM and above
**Excludes:** Previously known issues (CDP `--remote-allow-origins=*`, SSRF in `visit_page`, unauthenticated distributed protocol, no TLS on distributed protocol)

---

## Finding 1: WebSocket Continuation Frames Bypass Per-Frame Size Limit (Memory Exhaustion)

**File:** `ds4_web.c`
**Lines:** 556–609 (function `web_ws_read_message`)
**Severity:** MEDIUM
**Category:** Denial of Service / Memory Exhaustion

### Description

The WebSocket message reader enforces a per-frame payload size limit of `DS4_WEB_MAX_RESULT_BYTES * 4` (4 MB) at line 581, but does **not** enforce any limit on the total reassembled message size across continuation frames. An attacker who can inject or influence WebSocket frames can send hundreds of continuation frames (opcode 0x0), each under 4 MB, that are accumulated in the `msg` buffer without bound.

### Vulnerable Code

```c
// Line 581: per-frame check only
if (len > DS4_WEB_MAX_RESULT_BYTES * 4ULL) {
    web_set_err(err, err_len, "websocket message too large");
    ...
}
// Line 604: no total-size check before appending
} else if (opcode == 0x1 || opcode == 0x0) {
    web_buf_append(&msg, (const char *)payload, (size_t)len);
    ...
}
```

### Exploitability

Requires the ability to send WebSocket messages to the ds4 process via Chrome's DevTools WebSocket endpoint. Since CDP is exposed on a network-accessible port (known issue), a network attacker can open a separate WebSocket connection and send crafted frames. The vulnerability itself (unbounded accumulation) is distinct from the exposure surface.

### Attack Path

1. Attacker connects to the Chrome DevTools WebSocket endpoint on port 9333
2. Attacker sends an initial text frame (opcode 0x1) with `FIN=0` (start of fragmented message), payload < 4 MB
3. Attacker sends hundreds of continuation frames (opcode 0x0, `FIN=0`), each < 4 MB
4. Each frame passes the per-frame size check and is appended to the `msg` buffer
5. Memory grows unbounded until OOM kills the process

### Recommendation

Add a total message size check inside the continuation-frame accumulation loop:

```c
web_buf_append(&msg, (const char *)payload, (size_t)len);
if (msg.len > DS4_WEB_MAX_RESULT_BYTES * 4ULL) {
    free(payload);
    web_set_err(err, err_len, "websocket message too large");
    free(msg.ptr);
    return NULL;
}
```

---

## Finding 2: Unbounded HTTP Response Accumulation from CDP

**File:** `ds4_web.c`
**Lines:** 264–276 (function `web_http_request`)
**Severity:** MEDIUM
**Category:** Denial of Service / Memory Exhaustion

### Description

The `web_http_request` function reads HTTP responses from the local CDP HTTP endpoints in a loop with no upper bound on the accumulated response size. The buffer grows via `web_buf_append` until the connection closes or an error occurs, with no size cap.

### Vulnerable Code

```c
// Lines 264-276
web_buf resp = {0};
char tmp[4096];
for (;;) {
    ssize_t n = web_read_some(fd, tmp, sizeof(tmp), DS4_WEB_CONNECT_TIMEOUT_MS);
    if (n < 0) { ... }
    if (n == 0) break;
    web_buf_append(&resp, tmp, (size_t)n);  // No size limit
}
```

### Exploitability

Requires a malicious or compromised service on the CDP port (127.0.0.1:9333). If an attacker squats the port before Chrome starts (see Finding 3), they can serve arbitrarily large HTTP responses to exhaust memory.

### Attack Path

1. Attacker pre-occupies port 9333 on the same host
2. ds4 calls `web_cdp_alive()` or `web_browser_ws_url()`, which issues HTTP requests
3. Attacker responds with a huge HTTP response body (streaming, never closing until OOM)
4. The `web_buf_append` loop accumulates without bound

### Recommendation

Add a maximum response size check (e.g., 16 MB) in the HTTP response read loop.

---

## Finding 3: CDP Port Pre-Occupation Allows Request Interception

**File:** `ds4_web.c`
**Lines:** 29, 1106, 294–300
**Severity:** MEDIUM
**Category:** Information Disclosure / Local Privilege Escalation

### Description

The Chrome DevTools port defaults to 9333 (`DS4_WEB_DEFAULT_PORT`). The `web_ensure_browser` function (line 1106) first checks if CDP is already alive by connecting to this port and looking for `"webSocketDebuggerUrl"` in the JSON response. If the check succeeds, it assumes Chrome is running and proceeds to use that endpoint.

A local attacker can bind to port 9333 before ds4 starts, serve a fake CDP response that passes the health check, and then receive all subsequent navigation URLs, page content extractions, and any credentials in URLs that ds4 sends via CDP.

### Vulnerable Code

```c
// Line 298: trivially spoofable health check
bool ok = strstr(body, "webSocketDebuggerUrl") != NULL;
```

### Exploitability

Requires local access to the same host. The attacker must bind the port before ds4 starts Chrome. The health check is trivially spoofable (just return any JSON containing the string "webSocketDebuggerUrl").

### Attack Path

1. Local attacker starts a TCP server on port 9333 serving fake CDP responses
2. ds4 calls `web_ensure_browser` → `web_cdp_alive` returns true
3. ds4 connects to attacker's WebSocket, sends `Page.navigate` with target URLs
4. ds4 sends `Runtime.evaluate` with JavaScript extraction code
5. Attacker captures all URLs, page contents, and any credential data

### Recommendation

- Verify Chrome process identity (e.g., check the PID returned by Chrome or use a randomly chosen port)
- Use a nonce in the profile directory to authenticate the Chrome instance
- Bind to port 0 and let the OS assign an ephemeral port, passing it to Chrome

---

## Finding 4: Plaintext Password Storage via --password-store=basic

**File:** `ds4_web.c`
**Lines:** 1057, 1069, 1074
**Severity:** MEDIUM
**Category:** Credential Exposure

### Description

Chrome is launched with `--password-store=basic` on all platforms. This flag instructs Chrome to store saved passwords using its built-in plaintext storage rather than the OS keychain (macOS Keychain, Linux Secret Service, etc.). Any passwords that a user saves while browsing via ds4 are stored unencrypted in the profile directory at `~/.ds4/browser/`.

### Vulnerable Code

```c
// Line 1069 (Linux as root):
execlp(exe, exe, port_arg, "--remote-allow-origins=*",
       profile_arg, ..., "--password-store=basic", ...);

// Line 1074 (Linux non-root):
execlp(exe, exe, port_arg, "--remote-allow-origins=*",
       profile_arg, ..., "--password-store=basic", ...);
```

### Exploitability

Any process or user with read access to `~/.ds4/browser/` can extract saved passwords. While the directory is created with mode 0700 (line 163), this provides no protection against same-user processes, backup software that preserves file contents, or root access.

### Attack Path

1. User browses a website via ds4 and Chrome offers to save a password
2. Password is stored in `~/.ds4/browser/Default/Login Data` (SQLite, unencrypted)
3. Any malicious software running as the same user reads the SQLite database
4. Passwords are extracted in cleartext

### Recommendation

Remove `--password-store=basic` and let Chrome use the system keychain, or add `--password-store=detect` to prefer the system keychain when available.

---

## Finding 5: Memory Amplification via Attacker-Controlled Frame Payload Size

**File:** `ds4_distributed.c`
**Lines:** 7592–7608 (function `dist_worker_handle_work`), 7777–7794 (prefetch loop)
**Severity:** HIGH
**Category:** Denial of Service / Memory Amplification

### Description

When a WORK frame is received, the worker immediately allocates `bytes` bytes of memory based on the frame header's declared payload size, **before** reading the payload data. The `bytes` field is a `uint32_t` from the wire, allowing values up to ~4 GB. The `dist_read_full` function has **no receive timeout by default** (documented at line 1046–1048), so the allocation persists indefinitely while the attacker trickles data slowly or simply holds the connection open.

This creates a severe memory amplification: 12 bytes of attacker input (frame header) can cause up to 4 GB of memory allocation that persists indefinitely.

### Vulnerable Code

```c
// Lines 7596-7601
static int dist_worker_handle_work(..., uint32_t bytes) {
    void *payload = malloc(bytes);  // bytes from wire, up to 4 GB
    if (!payload) { ... }
    int rc = dist_read_full(upstream->fd, payload, bytes);  // blocks indefinitely
    ...
}

// Lines 7785-7794 (prefetch path, same pattern)
job->payload = malloc(bytes);
...
rc = dist_read_full(fd, job->payload, bytes);
```

### Exploitability

Directly exploitable by any host that can connect to the worker's data port. No authentication is required. The attacker sends a single 12-byte frame header. Multiple concurrent connections can exhaust all available memory.

### Attack Path

1. Attacker connects to the worker data port
2. Attacker sends a valid frame header: magic=DS4D, type=WORK, bytes=0xFFFFFFFF
3. Worker calls `malloc(0xFFFFFFFF)` — allocates ~4 GB
4. Worker calls `dist_read_full()` which blocks waiting for ~4 GB of data
5. Attacker keeps the TCP connection alive but sends no data (slowloris-style)
6. Repeat with additional connections — each ties up 4 GB + 1 thread

### Recommendation

- Enforce a maximum frame payload size (e.g., 256 MB) at the frame-header parsing level in `dist_read_frame_header`
- Enable a receive timeout by default (not just via optional env var)
- Consider reading the payload in chunks instead of pre-allocating the entire buffer

---

## Finding 6: Unlimited Thread/Connection Creation (Thread Exhaustion DoS)

**File:** `ds4_distributed.c`
**Lines:** 4280–4321 (coordinator accept loop), 7869–7907 (worker data listener)
**Severity:** MEDIUM
**Category:** Denial of Service / Resource Exhaustion

### Description

Both the coordinator's accept loop and the worker's data listener accept unlimited incoming connections and spawn a detached thread for each. There is no limit on concurrent connections, no connection rate limiting, and no backpressure mechanism. An attacker can open thousands of connections to exhaust threads, file descriptors, and memory.

This is distinct from the "unauthenticated protocol" known issue — even with authentication, there is no connection limit to prevent resource exhaustion.

### Vulnerable Code

```c
// Lines 4280-4321 (coordinator)
for (;;) {
    int fd = accept(listen_fd, ...);
    ds4_dist_client_ctx *ctx = calloc(1, sizeof(*ctx));
    pthread_create(&tid, NULL, dist_coordinator_client_main, ctx);
    pthread_detach(tid);   // No tracking, no limit
}

// Lines 7869-7907 (worker data listener) — identical pattern
for (;;) {
    int fd = accept(listen_fd, ...);
    ds4_dist_data_client_ctx *ctx = calloc(1, sizeof(*ctx));
    pthread_create(&tid, NULL, dist_worker_data_client_main, ctx);
    pthread_detach(tid);   // No tracking, no limit
}
```

### Exploitability

Directly exploitable by any host that can reach the coordinator or worker port. The default `pthread_create` stack size is typically 2–8 MB, so 1000 connections consume 2–8 GB of stack space alone, plus the `ctx` allocations and any per-connection work.

### Attack Path

1. Attacker opens thousands of TCP connections to the coordinator or worker port
2. Each connection spawns a detached thread consuming stack memory
3. Threads block waiting for HELLO frames (coordinator) or WORK frames (worker) that never arrive
4. Without a receive timeout (default), threads block indefinitely
5. System runs out of threads, file descriptors, or memory

### Recommendation

- Maintain a count of active connections and reject new ones above a configurable limit
- Set a mandatory receive timeout (e.g., 300s) for idle connections
- Consider using a thread pool instead of unbounded thread creation

---

## Finding 7: Unbounded Disk Write via Crafted Snapshot payload_bytes

**File:** `ds4_distributed.c`
**Lines:** 6991, 7050–7100 (function `dist_worker_handle_snapshot_load`)
**Severity:** MEDIUM
**Category:** Denial of Service / Disk Exhaustion

### Description

During snapshot loading, the `payload_bytes` field is a 64-bit value read from the wire (lines 6991). The worker reads chunks into a temp file until `received >= payload_bytes` (line 7051). Individual chunks are bounded to 8 MB (`DS4_DIST_SNAPSHOT_CHUNK_BYTES`), but there is no upper bound on the total `payload_bytes`. An attacker can set `payload_bytes` to an extremely large value and continuously send valid 8 MB chunks to fill disk.

### Vulnerable Code

```c
// Line 6991: 64-bit value from wire, no upper bound check
const uint64_t payload_bytes = dist_u64_from_halves(begin.payload_hi, begin.payload_lo);

// Line 7051: loop until attacker-controlled target reached
while (!err[0] && received < payload_bytes) {
    // ... reads chunks up to 8 MB each, writes to temp file ...
    received += chunk_bytes;
}
```

### Exploitability

Requires the ability to send snapshot-load frames to a worker. The coordinator sends these during normal operation, but an attacker on the network (unauthenticated access is a known issue) can craft snapshot-load sequences.

### Attack Path

1. Attacker connects to worker, sends a HELLO frame to register
2. Attacker sends a SNAPSHOT_LOAD_BEGIN with `payload_bytes = 0xFFFFFFFFFFFFFFFF`
3. Attacker sends continuous SNAPSHOT_CHUNK frames (8 MB each)
4. Worker writes all chunks to a temp file on disk
5. Disk fills up, affecting the entire system

### Recommendation

- Validate `payload_bytes` against a reasonable maximum (e.g., based on model size and context length)
- Monitor disk usage during writes and abort if a threshold is exceeded
- Use `tmpfile()` with filesystem quotas where possible

---

## Finding 8: Predictable Session ID Generation

**File:** `ds4_distributed.c`
**Lines:** 3895, 4325–4328
**Severity:** MEDIUM
**Category:** Session Management

### Description

Session IDs are generated using easily predictable or observable values: `time(NULL)`, `getpid()`, `clock()`, and a pointer address. An attacker who can observe process start time and PID (commonly available via `/proc`) can predict or brute-force session IDs. This is distinct from the "unauthenticated protocol" issue — if authentication were added in the future, the weak session IDs would still enable session hijacking.

### Vulnerable Code

```c
// Line 3895
const uint64_t session_id = ((uint64_t)(uint32_t)time(NULL) << 32) ^ (uint64_t)getpid();

// Lines 4325-4328
static uint64_t dist_make_session_id(const void *ptr) {
    uint64_t id = ((uint64_t)(uint32_t)time(NULL) << 32) ^ (uint64_t)getpid();
    id ^= ((uint64_t)(uintptr_t)ptr << 17) ^ (uint64_t)(uintptr_t)ptr;
    id ^= (uint64_t)clock();
    return id ? id : 1u;
}
```

### Exploitability

An attacker on the same network who can send WORK frames can use a predicted session ID to:
- Target a specific user's KV cache state
- Inject crafted token sequences into an existing session
- Cause session state corruption by sending conflicting work with the same session ID

### Attack Path

1. Attacker observes the coordinator's PID (e.g., via `/proc` or network timing)
2. Attacker knows approximate start time (coarse timestamp)
3. Attacker brute-forces the remaining entropy (pointer XOR, clock value)
4. Attacker sends WORK frames with the predicted session ID
5. Worker's KV cache for that session is corrupted

### Recommendation

Use a cryptographically secure random number generator (e.g., read from `/dev/urandom`) for session ID generation.

---

## Summary Table

| # | File | Lines | Severity | Category | Description |
|---|------|-------|----------|----------|-------------|
| 1 | ds4_web.c | 556–609 | MEDIUM | DoS | WebSocket continuation frames bypass per-frame size limit |
| 2 | ds4_web.c | 264–276 | MEDIUM | DoS | Unbounded HTTP response accumulation |
| 3 | ds4_web.c | 29, 1106, 294–300 | MEDIUM | Info Disclosure | CDP port pre-occupation intercepts requests |
| 4 | ds4_web.c | 1057, 1069, 1074 | MEDIUM | Credential Exposure | Plaintext password storage via --password-store=basic |
| 5 | ds4_distributed.c | 7596, 7785 | HIGH | DoS | Memory amplification via frame header size field (12 bytes → 4 GB) |
| 6 | ds4_distributed.c | 4280–4321, 7869–7907 | MEDIUM | DoS | Unlimited thread creation per connection |
| 7 | ds4_distributed.c | 6991, 7050–7100 | MEDIUM | DoS | Unbounded disk write via crafted snapshot payload_bytes |
| 8 | ds4_distributed.c | 3895, 4325–4328 | MEDIUM | Session Mgmt | Predictable session ID generation |
