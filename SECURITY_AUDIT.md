# Security Audit: ds4_kvstore.c / ds4_kvstore.h

**Auditor:** Automated Security Review  
**Date:** 2026-06-26  
**Scope:** `ds4_kvstore.c`, `ds4_kvstore.h`, with cross-references to `ds4.c` for payload staging  
**Branch:** `cursor/at-rest-security-review-753a`

---

## Finding 1 — Plaintext Sensitive Data at Rest (HIGH)

**Files:** `ds4_kvstore.c:1058`, `ds4_kvstore.c:1085-1091`  
**Category:** Data at rest / Confidentiality

### Description

KV cache files written to disk contain the full rendered prompt text in plaintext
(line 1087: `fwrite(text, 1, text_len, fp)`), followed by a raw binary payload of
the session's internal KV state. The text portion includes the complete
conversation history — system prompts, user messages, assistant responses, and
any tool-call results or tokens embedded in the prompt.

There is no encryption, MAC, or authenticated-encryption envelope. The cache
directory is created with `0700` permissions (`kv_mkdir_p`, line 361/367), but
individual files are created via `fopen(tmp, "wb")` (line 1058), which applies
the process umask. If the umask is `0022` (common default), files are created
`0644` (world-readable).

### Exploitability

**HIGH.** Any local user with read access to the cache directory can:
1. Read all cached prompt text verbatim from the `.kv` files (the text starts at
   a fixed offset: `DS4_KVSTORE_FIXED_HEADER + 4` = byte 52).
2. Extract API tokens, secrets, or PII that appeared anywhere in the prompt.
3. The file format is trivial to parse — magic bytes `KVC\x01`, then a 48-byte
   header, a 4-byte little-endian text length, then that many bytes of plaintext.

### Recommendation

- Set the umask or use `open(path, O_WRONLY|O_CREAT|O_EXCL, 0600)` + `fdopen()`
  to ensure cache files are created with `0600` permissions.
- Consider encrypting the text and payload portions with an authenticated cipher
  (e.g., XChaCha20-Poly1305) keyed to the running user/service.

---

## Finding 2 — Cache Files Created Without Restrictive Permissions (HIGH)

**File:** `ds4_kvstore.c:1058`  
**Category:** File permissions

### Description

```c
FILE *fp = fopen(tmp, "wb");   // line 1058
```

`fopen` with mode `"wb"` does not allow specifying the file creation mode.
The resulting permissions are `0666 & ~umask`. With the common default umask of
`0022`, the temp file is created as `0644` — readable by every user on the
system. After `rename(tmp, path)` (line 1107), the final `.kv` file retains
those permissions.

Similarly, `ds4_kvstore_touch_file` (line 486) and `kv_cache_rewrite_trailer`
(line 893) open existing files with `fopen(path, "r+b")` — this does not modify
permissions, but if the file was originally created world-readable it remains so.

The cache directory itself is created with `0700` (lines 361, 367), which blocks
traversal for other users. However, if the directory already exists with looser
permissions (e.g., manually created or set by a deployment script), the code does
not tighten them.

### Exploitability

**HIGH** in shared-machine or container-escape scenarios. The directory mode
provides a partial mitigation, but relying solely on directory permissions is
fragile — a single misconfiguration exposes all cached data.

### Recommendation

- Use `open(tmp, O_WRONLY|O_CREAT|O_EXCL, 0600)` followed by `fdopen(fd, "wb")`
  for all file-creation paths.
- After `kv_mkdir_p`, verify the directory mode with `stat()` and `chmod()` to
  `0700` if it differs, or fail with a clear error.

---

## Finding 3 — No Symlink Attack Protection (MEDIUM)

**Files:** `ds4_kvstore.c:1058`, `ds4_kvstore.c:1107`, `ds4_kvstore.c:448`,
`ds4_kvstore.c:486`, `ds4_kvstore.c:817`, `ds4_kvstore.c:893`, `ds4_kvstore.c:1237`  
**Category:** Symlink / link-following attacks

### Description

Every `fopen()` call in the kvstore follows symlinks. An attacker who can create
a symlink at the expected cache path (or the `.tmp.<pid>` intermediate) can:

1. **Read arbitrary files:** Place a symlink at
   `<cache_dir>/<sha>.kv` pointing to `/etc/shadow` or another sensitive file.
   When `ds4_kvstore_try_load_text` opens it and reads `text_bytes` of data
   (line 1256), the contents of the target file are loaded into memory and
   processed.

2. **Write/overwrite arbitrary files:** The temp file path is
   `<cache_dir>/<sha>.kv.tmp.<pid>` (line 1055). If an attacker can predict the
   PID and SHA (the SHA is deterministic from prompt text), they can pre-create a
   symlink at that path. The `fopen(tmp, "wb")` (line 1058) will follow the
   symlink and overwrite the target. The `rename(tmp, path)` (line 1107) then
   replaces the symlink with a regular file, but the damage (truncation of the
   target) is already done.

3. **Eviction unlink as targeted deletion:** `unlink(e.path)` at line 589
   follows the directory entry; if `e.path` was scanned from a directory listing
   containing a symlink, `unlink` removes the symlink itself (not its target),
   but `ds4_kvstore_read_entry_file` at line 448 uses `stat()` (which follows
   symlinks) to validate the entry, so a carefully crafted symlink to a valid KV
   file elsewhere could cause the wrong entry's metadata to be trusted.

### Exploitability

**MEDIUM.** Requires the attacker to be able to write into the cache directory.
The `0700` directory creation mitigates this for new directories, but does not
protect against:
- A pre-existing directory with looser permissions.
- A shared `/tmp`-based cache path.
- A compromised co-process running as the same user.

### Recommendation

- Use `O_NOFOLLOW` on all `open()` calls for cache files.
- Use `openat()` relative to a directory fd obtained with `O_NOFOLLOW|O_DIRECTORY`.
- Use `lstat()` instead of `stat()` in `ds4_kvstore_read_entry_file` (line 445).
- Use `renameat()` with `RENAME_NOREPLACE` where available.

---

## Finding 4 — TOCTOU Race in File Creation (MEDIUM)

**File:** `ds4_kvstore.c:997-1058`  
**Category:** Race conditions

### Description

The store path has a classic check-then-act race:

1. `kv_cache_existing_compatible()` (line 997) calls `access(path, F_OK)` (via
   line 847), then reads the file to check compatibility.
2. If the file does not exist or is incompatible, the code proceeds to create a
   new temp file and `rename()` it into place.
3. Between step 1 and step 2, another process (or another thread in a
   multi-session server) could create the same file.

The `rename()` at line 1107 is atomic on POSIX, so the final state is
consistent, but:
- The losing writer wastes CPU/IO serializing the payload.
- If the winning writer wrote a *different* text (SHA collision — see Finding 7),
  the loser's `rename()` silently replaces it.

Additionally, `ds4_kvstore_touch_file` (line 486) does a read-modify-write of
the header without any file locking, so concurrent callers can lose hit-count
updates.

### Exploitability

**MEDIUM** in multi-process server deployments sharing a cache directory. Mostly
a reliability/data-integrity issue rather than a confidentiality breach, but
combined with Finding 7 it could be used to inject a crafted cache file.

### Recommendation

- Use advisory locking (`flock` or `fcntl`) around the write-and-rename
  sequence.
- Use `O_EXCL` when creating the temp file to detect races.
- Use `flock` in `ds4_kvstore_touch_file` for the read-modify-write cycle.

---

## Finding 5 — SHA-1 Collision Allows Cache Poisoning (MEDIUM)

**Files:** `ds4_kvstore.c:215-328` (SHA-1 implementation),
`ds4_kvstore.c:992-993`, `ds4_kvstore.c:1208-1210`  
**Category:** Cryptographic weakness

### Description

Cache file naming and lookup rely entirely on SHA-1 of the prompt text prefix:

```c
ds4_kvstore_sha1_bytes_hex(text, text_len, sha);        // line 993
char *path = ds4_kvstore_path_for_sha(kc, sha);         // line 994
```

SHA-1 is cryptographically broken — practical chosen-prefix collision attacks
exist (SHAttered, 2017; Shambles, 2020). An attacker who controls part of the
prompt text can construct two distinct prompts that hash to the same SHA-1,
causing:

1. **Cache poisoning:** Attacker stores a cache entry under a colliding hash.
   When a victim's prompt resolves to the same hash, `ds4_kvstore_find_text_prefix`
   (line 1208-1210) matches by SHA only, and `ds4_kvstore_try_load_text` loads
   the attacker's payload. The byte-prefix verification at line 1266-1268
   provides a second check, but a chosen-prefix collision produces matching
   prefixes for the colliding portion.

2. **Cache replacement:** Two legitimately different prompts that collide cause
   silent overwrites, since the `rename()` path replaces any existing file with
   the same SHA-derived name.

### Exploitability

**MEDIUM.** A chosen-prefix SHA-1 collision costs roughly $45k-$75k in GPU time
(2020 estimates, likely cheaper now). This is out of reach for casual attackers
but feasible for well-resourced adversaries targeting high-value inference
pipelines. The practical barrier is also lowered because the attacker only needs
a prefix collision — the suffix can differ.

The code does perform a byte-prefix match after the SHA lookup (line 1266), which
acts as a partial second barrier. However, in a chosen-prefix collision scenario,
the attacker controls the matching prefix bytes.

### Recommendation

- Migrate to SHA-256 or BLAKE3 for cache keying. The hash is only used for file
  naming and lookup — performance is not a concern.
- As a defense-in-depth measure, store the full text length in the filename or
  a secondary index.

---

## Finding 6 — Insufficient Deserialization Validation on Load (MEDIUM)

**Files:** `ds4_kvstore.c:1245-1281`, `ds4.c:23146-23196`  
**Category:** Deserialization / Untrusted input

### Description

When loading a cache file, `ds4_kvstore_try_load_text` performs header
validation (magic, version, ABI, model_id, text prefix match) and then calls
`ds4_session_load_payload(session, fp, hdr.payload_bytes, ...)` (line 1277).

The payload loader in `ds4.c` (line 23146+) does check the payload magic,
version, architectural constants (`DS4_N_LAYER`, `DS4_N_HEAD_DIM`, etc.), and
context-size bounds. However:

1. **`payload_bytes` is trusted from the file header** (line 434:
   `e->payload_bytes = kv_le_get64(h + 40)`). A corrupted or malicious file can
   set `payload_bytes` to any 64-bit value. The payload loader reads exactly
   `payload_bytes` worth of data. If `payload_bytes` is larger than the actual
   file, `fread` will return short and the loader should detect this — but if
   `payload_bytes` is crafted to be *smaller* than the real payload, the loader
   stops early, leaving the file pointer at an attacker-controlled position for
   subsequent trailer parsing.

2. **Trailer hooks parse data after the payload** (line 1291-1293). The trailer
   `load` callback receives an `fp` at whatever position the payload loader left
   it. If `payload_bytes` was manipulated, the trailer parser reads
   attacker-controlled bytes as structured data. The severity depends on the
   trailer hook implementation (external to this file), but the kvstore layer
   does not validate that the file pointer is at the expected position after
   payload load.

3. **`text_bytes` (uint32_t from file)** controls a heap allocation at line 1255:
   `cached_text = kv_xmalloc((size_t)text_bytes + 1)`. A malicious file with
   `text_bytes = 0xFFFFFFFF` causes a 4 GiB allocation. While this is a
   denial-of-service rather than code execution, it can OOM-kill the process.
   The only guard is the earlier `st.st_size` check in
   `ds4_kvstore_read_entry_file`, which is not performed in the load path
   (`ds4_kvstore_try_load_text` re-reads the header from the already-opened
   file without rechecking file size).

### Exploitability

**MEDIUM.** Requires the attacker to place a crafted `.kv` file in the cache
directory (possible via Finding 3 or 4, or if the cache dir has loose
permissions). The OOM vector is straightforward; the trailer-confusion vector
depends on hook implementations.

### Recommendation

- After `ds4_session_load_payload` returns, verify `ftell(fp)` matches the
  expected position (`header_size + 4 + text_bytes + payload_bytes`).
- Validate `text_bytes` against the file size before allocating:
  `if (text_bytes > st.st_size - DS4_KVSTORE_FIXED_HEADER - 4) return false;`
- Consider adding a whole-file HMAC or checksum that is verified before any
  parsing.

---

## Finding 7 — Temp File Path Predictable (MEDIUM)

**File:** `ds4_kvstore.c:1054-1056`  
**Category:** Predictable temp file

### Description

```c
kv_buf_printf(&tmpb, "%s.tmp.%ld", path, (long)getpid());
```

The temporary file path is `<cache_dir>/<sha>.kv.tmp.<pid>`. Both the SHA (a
deterministic hash of prompt text) and the PID are predictable to a local
attacker:
- The PID can be enumerated from `/proc`.
- The SHA can be computed if the attacker knows or can guess the prompt text.

This enables a pre-creation attack: the attacker creates a symlink (or a file)
at the predicted temp path before the victim process opens it. Since `fopen(tmp,
"wb")` does not use `O_EXCL`, it will silently open the existing file (or follow
a symlink).

### Exploitability

**MEDIUM.** Requires local access and ability to write to the cache directory
(same constraints as Finding 3). In containerized deployments with shared
volumes, this is realistic.

### Recommendation

- Use `mkstemp()` or `open(tmp, O_WRONLY|O_CREAT|O_EXCL, 0600)` to create temp
  files with guaranteed uniqueness and no symlink following.
- Alternatively, include a random suffix (e.g., from `getrandom()`) instead of
  the PID.

---

## Finding 8 — Staged Payload Uses World-Readable /tmp (MEDIUM)

**File:** `ds4.c:22895-22896`  
**Category:** Data at rest / temp file exposure

### Description

```c
char tmpl[] = "/tmp/ds4-session-payload.XXXXXX";
int fd = mkstemp(tmpl);
```

The session payload (full KV cache state including embedded prompt data) is
staged to a file in `/tmp`. While `mkstemp()` creates the file with `0600`
permissions, the file exists in a world-traversable directory. On systems with
`/tmp` on a shared filesystem, or in container environments where `/tmp` is
bind-mounted, other processes may be able to observe the file's existence and
race to open it between creation and first write.

More critically, `mkstemp` creates the file with `0600` but the `fdopen` call
on line 22901 does not re-verify permissions, and the file persists on disk
(potentially containing gigabytes of session state) until
`ds4_session_payload_file_free` calls `unlink` — which may be significantly
later.

### Exploitability

**MEDIUM.** The `mkstemp` + `0600` permissions make direct reads difficult, but
the predictable `/tmp` prefix aids targeted attacks on shared systems. The
payload file can be very large and long-lived.

### Recommendation

- Use a private temp directory under the cache dir (already `0700`) instead of
  `/tmp`.
- Consider using `O_TMPFILE` on Linux to create an unnamed temp file that never
  appears in the directory listing.

---

## Summary Table

| # | Finding | Severity | Category |
|---|---------|----------|----------|
| 1 | Plaintext sensitive data in cache files | HIGH | Confidentiality |
| 2 | Files created without 0600 permissions | HIGH | File permissions |
| 3 | No symlink protection on file operations | MEDIUM | Symlink attacks |
| 4 | TOCTOU race in file creation/compatibility check | MEDIUM | Race conditions |
| 5 | SHA-1 collision enables cache poisoning | MEDIUM | Cryptographic weakness |
| 6 | Insufficient deserialization validation on load | MEDIUM | Deserialization |
| 7 | Predictable temp file path | MEDIUM | Predictable temp |
| 8 | Staged payload in world-traversable /tmp | MEDIUM | Temp file exposure |
