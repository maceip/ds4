# Security Findings Report

Generated: 2026-07-17

This report covers NEW vulnerabilities found in the ds4 codebase, excluding
issues already identified in prior scans (agent shell execution, path sandbox,
DSML close-tag injection, search ReDoS, prompt injection persistence, trace
permissions, session forgery, terminal escape injection, Chrome CDP flags,
linenoise UTF-8 OOB, CLI/eval terminal escape injection).

---

## Finding 1

```json
{
  "severity": "high",
  "location": "rax.c",
  "title": "raxAddChildNoAlloc: use-after-realloc on OOM after node resize",
  "description": "In raxAddChildNoAlloc(), after a successful rax_realloc() of the parent node at line 320 (which may move the allocation to a new address and free the old one), a subsequent OOM failure from raxNewValueNode() at line 363 causes the function to return NULL. The caller (raxAddChild at line 470, or raxGenericInsert at line 1039) still holds the original pre-realloc pointer, which now points to freed memory. The oom handler in raxGenericInsert (line 1075) then dereferences this stale pointer (h->size), causing a use-after-free.",
  "impact": "Memory corruption leading to potential code execution. In a server context (ds4_server.c uses rax for tool_memory), an attacker who can trigger memory pressure during tree insertion (e.g., by filling tool memory with many entries) could exploit this for arbitrary write via heap corruption.",
  "attack_path": "1. Attacker sends many tool memory entries to ds4_server, growing a rax tree to have 13+ children on a node. 2. A new insertion triggers raxAddChildNoAlloc which needs to materialize child 12 (inline->real). 3. Memory pressure causes raxNewValueNode to fail. 4. rax_realloc already moved the node, so the caller holds a dangling pointer. 5. The oom handler at line 1075 writes to freed memory via h->isnull=1, h->iskey=1. 6. Heap metadata corruption enables further exploitation.",
  "evidence": ["rax.c:320", "rax.c:321", "rax.c:362", "rax.c:363", "rax.c:1039", "rax.c:1075-1079"],
  "remediation": "raxAddChildNoAlloc must allocate materialized_child BEFORE performing rax_realloc, or track the new node pointer via an output parameter even on failure. Alternatively, pass back the reallocated node through parentlink or an extra out-parameter so the caller can update its pointer regardless of success."
}
```

## Finding 2

```json
{
  "severity": "high",
  "location": "rax.c",
  "title": "raxFindParentLink / raxRemoveChild: unbounded out-of-bounds read on child mismatch",
  "description": "raxFindParentLink (line 1124) and raxRemoveChild (line 1211) both scan the parent node's child pointer array in an unbounded while(1) loop with no size check. The code assumes the child pointer will always be found. If tree state becomes inconsistent (e.g., after a partial OOM modification from Finding 1, or due to concurrent access without locking in ds4_server tool_memory), the scan reads past the node allocation without limit.",
  "impact": "Out-of-bounds heap read that continues until a coincidental pointer match or segfault. In the OOB read path, sensitive heap data (keys, values, function pointers) may be compared against the searched child pointer, leaking information through timing or side channels. A segfault causes denial of service in the server process.",
  "attack_path": "1. Trigger Finding 1 to corrupt tree state. 2. A subsequent raxRemove operation calls raxRemoveCleanup (line 1245), which calls raxRemoveChild with a child pointer freed at line 1235. 3. If the allocator has reused the freed child's address, raxRemoveChild scans past the node boundary. 4. The OOB read causes crash or heap information disclosure. 5. If the scan matches a wrong pointer, raxRemoveChildAtPtr (line 1141) performs memmove operations on corrupt data, enabling heap corruption.",
  "evidence": ["rax.c:1119-1133", "rax.c:1211-1220", "rax.c:1235", "rax.c:1245"],
  "remediation": "Add bounds checking to both raxFindParentLink and raxRemoveChild: iterate only up to n->iscompr ? 1 : n->size child pointers, and return NULL (or assert) if the child is not found. All callers must handle the failure."
}
```

## Finding 3

```json
{
  "severity": "high",
  "location": "rax.c",
  "title": "raxRemoveCleanup: use of freed pointer value as search key",
  "description": "In raxRemoveCleanup (line 1226), the function frees child nodes in a loop (line 1235: rax_free(child)), then uses the freed pointer's value to locate and remove the child from the parent via raxRemoveChild(h, child) at line 1245. While the parent's child pointer array still holds the old address, using a freed pointer's value is undefined behavior in C. More critically, if malloc returns the same address for a new allocation between the free at line 1235 and the search at line 1245 (possible in multi-threaded server use), the search could match the wrong child slot, causing structural corruption of the radix tree.",
  "impact": "Radix tree structural corruption. In ds4_server's tool_memory (which uses rax for key-value lookup), this can cause tools to return data from the wrong tool memory entry, or crash the server. The corruption could propagate to affect request routing.",
  "attack_path": "1. Create tool memory entries to build a multi-level rax tree. 2. Delete entries to trigger raxRemoveCleanup with multi-level cleanup. 3. In the server's multi-threaded context, another thread's malloc reuses the freed child address. 4. raxRemoveChild matches the wrong slot, corrupting the parent node's child array. 5. Subsequent lookups return wrong data or crash.",
  "evidence": ["rax.c:1231-1245", "rax.c:1211-1219"],
  "remediation": "Save the child pointer from the parent BEFORE freeing the child node, or restructure the cleanup to call raxRemoveChild before rax_free."
}
```

## Finding 4

```json
{
  "severity": "high",
  "location": "ds4_web.c",
  "title": "WebSocket frame reassembly: no total message size limit enables OOM denial of service",
  "description": "web_ws_read_message (line 556) reassembles multi-frame WebSocket messages in a loop. Each individual frame's payload is checked against DS4_WEB_MAX_RESULT_BYTES * 4 (4 MB) at line 581. However, there is no limit on the total accumulated message size across continuation frames (opcode 0x0). An attacker who controls or MITMs the WebSocket connection to Chrome's CDP port can send arbitrarily many continuation frames of ~4 MB each, growing the web_buf without bound. web_buf_append calls realloc which will eventually fail, and web_xmalloc (line 66) calls exit(1) on allocation failure, crashing the entire agent process.",
  "impact": "Denial of service via process crash. The agent process exits immediately on OOM because web_xmalloc calls exit(1). Since the agent is a single process managing model inference, a crash kills the entire session including unsaved state.",
  "attack_path": "1. Agent's visit_page tool connects to a malicious or compromised website. 2. If the page redirects to a rogue WebSocket or the attacker intercepts CDP traffic on localhost (CDP binds to 127.0.0.1 with --remote-allow-origins=*). 3. The rogue server sends thousands of continuation frames (opcode 0x0, fin=0) each near the 4MB single-frame limit. 4. web_buf_append keeps growing via realloc. 5. Eventually realloc returns NULL, web_xmalloc calls exit(1). 6. The agent process dies.",
  "evidence": ["ds4_web.c:556-610", "ds4_web.c:581", "ds4_web.c:604", "ds4_web.c:66-71"],
  "remediation": "Add a total accumulated message size check in the web_ws_read_message loop. Reject messages exceeding DS4_WEB_MAX_RESULT_BYTES total. Also replace exit(1) in web_xmalloc with a graceful error return path."
}
```

## Finding 5

```json
{
  "severity": "high",
  "location": "ds4_web.c",
  "title": "CDP WebSocket ID matching via naive string search allows response spoofing",
  "description": "web_json_id_matches (line 617) uses strstr(json, '\"id\"') to locate the id field, then atoi on the value after the colon. This naive JSON parsing matches ANY occurrence of '\"id\"' in the response, including inside string values. A crafted CDP response (from a compromised Chrome extension or malicious page injecting CDP events) could include '\"id\":TARGET_ID' inside a string field of an event message, causing the agent to accept an attacker-controlled response as the legitimate CDP call result. Since CDP call results drive page navigation, JavaScript evaluation, and screenshot capture, this enables the attacker to inject arbitrary content into the agent's tool results.",
  "impact": "An attacker can inject fake CDP responses that the agent treats as legitimate. This allows returning attacker-controlled text as visit_page content, or manipulating the agent's view of page state. Since tool results are fed directly into the LLM context, this enables prompt injection through web content.",
  "attack_path": "1. Agent browses a page that triggers Chrome to emit a CDP event containing a string like '\"id\":42' (where 42 is the pending request ID). 2. web_cdp_call receives this event message first. 3. web_json_id_matches finds the '\"id\":42' pattern inside the event string and returns true. 4. The agent treats the event as the CDP response, getting attacker-controlled content. 5. The real response is then discarded or matches a future request ID.",
  "evidence": ["ds4_web.c:617-625", "ds4_web.c:649-655"],
  "remediation": "Implement proper JSON parsing for response ID matching. At minimum, verify that the '\"id\"' key is at the top level of the JSON object (preceded by '{' or ',' after accounting for whitespace) rather than nested inside a string value."
}
```

## Finding 6

```json
{
  "severity": "medium",
  "location": "ds4_agent.c",
  "title": "Bash tool output temp file race: predictable path and insufficient cleanup",
  "description": "agent_bash_start (line 6718) creates temporary output files using mkstemp with the template '/tmp/ds4_agent_output_XXXXXX'. While mkstemp itself is safe, the generated file path is stored in job->path and the file persists until the job is freed. The file is created with default umask permissions (typically 0600, but depends on process umask). More critically, there is no cleanup of temp files if the agent process crashes (e.g., from Finding 4's exit(1)). These files contain the full stdout/stderr of shell commands, which may include secrets, API keys, database credentials, or other sensitive data from the user's tool execution.",
  "impact": "Sensitive command output persists in world-readable temp files after agent crash. An attacker with local access can read /tmp/ds4_agent_output_* files to extract secrets from prior agent sessions.",
  "attack_path": "1. Agent executes bash tool commands that output credentials (e.g., env vars, config files). 2. Output is captured to /tmp/ds4_agent_output_XXXXXX. 3. Agent crashes due to any bug (OOM, signal, etc.). 4. Temp files remain on disk. 5. Local attacker reads the files to extract credentials.",
  "evidence": ["ds4_agent.c:6718", "ds4_agent.c:6764", "ds4_agent.c:6604-6613"],
  "remediation": "Set restrictive permissions (0600) explicitly after mkstemp. Implement an atexit handler or signal handler to clean up temp files on abnormal exit. Consider using tmpfile() or O_TMPFILE to create files that are automatically cleaned up."
}
```

## Finding 7

```json
{
  "severity": "medium",
  "location": "ds4_agent.c",
  "title": "Race condition between worker thread tool execution and main thread session switch",
  "description": "The agent worker thread executes tool calls (agent_execute_tool_calls at line 7791) without holding the mutex, while the main thread can process /switch commands (line 5367-5412) that completely replace the session state. The /switch command checks w->status.state != AGENT_WORKER_IDLE to gate on 'model is busy', but this check and the subsequent session state replacement are not atomic with respect to the worker thread's state transitions. Specifically, the worker can be between tool_result generation (line 7791) and session transcript append (line 7834), at which point status may transiently appear idle during a tool that yields (e.g., bash with wait), allowing the main thread to switch sessions. The worker then appends tool_result to the wrong session's transcript.",
  "impact": "Transcript corruption where tool results from one session are appended to a different session. This can cause the LLM to hallucinate based on wrong context, or leak sensitive tool output (file contents, command output) from one session into another.",
  "attack_path": "1. User starts a long-running bash tool (e.g., sleep 5 && cat /etc/shadow). 2. While the bash job is waiting, the agent status shows AGENT_WORKER_GENERATING but the tool is between poll cycles. 3. User rapidly sends /switch to change to a different session. 4. The race window allows the switch to complete. 5. When the bash job finishes, its output is appended to the new session's transcript. 6. The new session now contains the cat /etc/shadow output.",
  "evidence": ["ds4_agent.c:7791", "ds4_agent.c:7834", "ds4_agent.c:5367", "ds4_agent.c:5401-5411"],
  "remediation": "Hold the mutex during the critical section that spans tool execution through transcript append. Alternatively, use a generation counter to detect stale tool results after session switch."
}
```

## Finding 8

```json
{
  "severity": "medium",
  "location": "ds4_web.c",
  "title": "web_random_bytes: weak PRNG fallback for WebSocket masking key",
  "description": "web_random_bytes (line 320) falls back to a trivially predictable LCG seeded with time(NULL) ^ (getpid() << 32) when /dev/urandom cannot be opened. This generates the WebSocket Sec-WebSocket-Key for the CDP handshake (used indirectly through web_base64 at line 340). While the WebSocket key is primarily used for upgrade validation, if the fallback is used, the masking keys for client-to-server frames (generated via the same or similar mechanism at line 528) become predictable, allowing a network observer to unmask WebSocket traffic containing CDP commands with sensitive page content.",
  "impact": "If /dev/urandom is unavailable (container/chroot environments), CDP WebSocket traffic is protected only by a trivially predictable PRNG. A network observer on localhost can unmask frames to read CDP commands containing page text, JavaScript evaluation results, and potentially captured credentials.",
  "attack_path": "1. Agent runs in a restricted container where /dev/urandom is not mounted. 2. web_random_bytes falls back to LCG seeded with time^pid. 3. Attacker on same host observes the WebSocket connection to localhost:9333. 4. Attacker predicts the PRNG state from known time and pid. 5. Attacker unmasks all client-to-server frames. 6. CDP commands including Runtime.evaluate results (which may contain page credentials or sensitive content) are exposed.",
  "evidence": ["ds4_web.c:320-337", "ds4_web.c:510-541"],
  "remediation": "Try additional entropy sources (getrandom() syscall, /dev/random) before falling back to the LCG. If no good entropy source is available, abort the WebSocket connection rather than proceeding with predictable keys."
}
```

## Finding 9

```json
{
  "severity": "medium",
  "location": "rax.c",
  "title": "raxAddChildNoAlloc: materialized_child memory leak on subsequent memmove failure",
  "description": "In raxAddChildNoAlloc, if raxNewValueNode succeeds at line 362 (allocating materialized_child), but the function encounters an error in any subsequent step before reaching line 447 where materialized_child is stored, the allocated node is leaked. While the current code has no explicit error checks after line 363 before 447, the memmove operations at lines 376-428 operate on the reallocated buffer which is guaranteed valid. However, if a caller wraps this in a context where failure between these points is possible (or if the code is modified), the leak becomes active. More concretely, the leak occurs because raxAddChild (line 465) which wraps raxAddChildNoAlloc frees the newly allocated child on failure but does NOT know about materialized_child to free it.",
  "impact": "Memory leak of raxNode allocations. In long-running server processes, repeated OOM-triggered insertions and retries can accumulate leaked nodes, gradually consuming available memory.",
  "evidence": ["rax.c:357-363", "rax.c:447-451", "rax.c:465-477"],
  "remediation": "Track materialized_child as part of the function's cleanup path. If the function returns NULL for any reason after allocating materialized_child, free it."
}
```

## Finding 10

```json
{
  "severity": "medium",
  "location": "ds4_agent.c",
  "title": "agent_bash_start: command injection via unvalidated shell metacharacters in temp file path",
  "description": "agent_bash_start stores the mkstemp-generated temp path in job->path (line 6764). While mkstemp generates safe characters, the path prefix '/tmp/ds4_agent_output_' is hardcoded and safe. However, agent_bash_read_head (line 6787) opens this path with fopen, and if the file is replaced via a symlink between mkstemp and later reads (the file descriptor tmpfd is not used for reading, fopen reopens by path), an attacker could redirect reads to an arbitrary file. The agent then feeds this file's content into the model's context as 'bash output', allowing data exfiltration from arbitrary files the agent process can read.",
  "impact": "Local privilege escalation via symlink attack. An attacker with write access to /tmp can replace the temp file between creation and subsequent read, injecting arbitrary file content into the agent's tool output. This can leak sensitive system files.",
  "attack_path": "1. Attacker monitors /tmp for ds4_agent_output_* files. 2. Agent starts a bash command, creating /tmp/ds4_agent_output_XXXXXX. 3. Attacker replaces the file with a symlink to /etc/shadow (or SSH keys, etc.). 4. Agent reads the temp file to get bash output (via fopen at line 6793). 5. fopen follows the symlink, reading /etc/shadow. 6. Contents are displayed as bash output and fed into the LLM context.",
  "evidence": ["ds4_agent.c:6718-6719", "ds4_agent.c:6764", "ds4_agent.c:6787-6813"],
  "remediation": "Use the already-open file descriptor (job->tmp_fd) for reading instead of reopening by path. Use fdopen or lseek+read on the fd. Alternatively, open with O_NOFOLLOW to prevent symlink following."
}
```
