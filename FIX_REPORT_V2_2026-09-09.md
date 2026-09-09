# MNEMOS Bridge OpenAI Fix Report V2

Date: 2026-09-09

Repository: `/tmp/mnemos-audit-clones/mnemos-bridge-openai`

Branch: `fix/audit-findings-v2`

Base: `master` / `origin/master` at `f64ec5830bd518f174580a068d9ff8a25b4eca1d`

## Outcome

Both requested high-severity findings are fixed on the current default-branch code.

- H1 is fail-closed before discovery, during discovery, across overlapping discovery calls, and after a failed refresh.
- H2 always selects a credential-bearing legacy connection shape, invokes the selected factory once, propagates internal `TypeError`, and rejects credentialless signatures without calling them.
- The full test suite passes: **24 passed, 1 skipped**.
- No push was performed.

The pre-existing untracked `SOL_REVIEW.md` was treated as user-owned and was not edited or staged.

## H1: Tool allowlist failed open before discovery

### Root cause

The pending audit branch's allowlist design treated an unknown policy state as permissive. Before `openai_tools()` populated the discovered tool names, `handle_tool_call()` could dispatch any model-supplied name, including `delete_memory`. Its test `test_no_allowlist_configured_stays_permissive` explicitly locked in that bypass.

### Fix

`MnemosOpenAIAdapter` now maintains two valid sources of dispatch authority:

1. An explicit `allowed_tools` iterable supplied when constructing the adapter.
2. The normalized tool names produced by a successful `openai_tools()` discovery.

If neither source exists, dispatch raises `ValueError` before parsing arguments or reaching the dispatcher/client. An empty explicit or discovered set denies every tool.

Discovery refreshes invalidate the prior discovered set immediately. Generation-aware in-flight tracking prevents an earlier overlapping discovery from publishing an allowlist while a newer discovery is still running. The adapter publishes only the newest successful discovery snapshot, and only after every overlapping discovery call has exited. If the newest discovery fails, dispatch stays closed.

README API documentation now states that callers must complete `openai_tools()` before dispatch or provide `allowed_tools=[...]` explicitly.

### Regression coverage

- No configured/discovered allowlist rejects a forged `delete_memory` call without invoking the client.
- A single in-progress discovery keeps dispatch closed.
- Two overlapping discoveries keep dispatch closed after the first completes and open only after the second completes successfully.
- Successful discovery permits an offered tool and rejects an unoffered destructive tool.
- A failed discovery refresh invalidates the prior allowlist and leaves dispatch closed.
- Existing dispatch tests now declare their expected explicit allowlists instead of relying on permissive pre-discovery behavior.

## H2: Legacy connection fallback dropped credentials and retried internal TypeError

### Root cause

`_call_from_url()` speculatively invoked `McpClient.from_url()` with several keyword shapes, caught every `TypeError`, and retried. Its candidate list ended with `{"timeout": timeout}` and `{}`, so an older factory with no credential parameter could connect successfully without receiving the bearer token. A factory that accepted a candidate but raised an internal `TypeError` could be invoked repeatedly under different credential shapes.

### Fix

The fallback now builds credential-bearing candidates only:

- `token=mcp_token`
- `bearer_token=mcp_token`
- `mcp_token=mcp_token`
- `headers={"Authorization": "Bearer ..."}`
- token-derived `auth=...`
- token-derived `auth_resolver=...`

Timeout and no-timeout variants preserve older signatures, but there is no credential-free candidate. Explicitly declared credential parameters take precedence over a generic `**kwargs` sink so a factory cannot silently ignore a different accepted keyword shape.

`inspect.signature(...).bind(...)` selects one compatible shape without invoking the factory. The factory is then called exactly once. A `TypeError` from inside that call propagates immediately. If no credential-bearing shape binds, the adapter raises before opening any connection. For non-introspectable callables, the canonical `token` shape is attempted once and any error propagates.

### Regression coverage

- Direct `token`, `bearer_token`, `mcp_token`, bearer `headers`, `auth`, and `auth_resolver` signatures all receive the credential.
- A declared `auth` parameter is preferred over a generic `**kwargs` sink.
- Timeout and no-timeout legacy variants are covered.
- A credentialless `(url, *, timeout)` factory is rejected without invocation.
- A factory accepting `**kwargs` that raises an internal `TypeError` is invoked once; the original error propagates.

## Files changed

- `src/mnemos_bridge_openai/adapter.py`
  - Added fail-closed explicit/discovered tool allowlisting.
  - Added overlap-safe discovery generation tracking.
  - Replaced speculative connection fallback calls with signature binding and one credential-bearing invocation.
- `tests/test_handle_tool_call_offline.py`
  - Updated existing direct-dispatch tests to declare the tools they intentionally allow.
- `tests/test_security_regressions.py`
  - Added focused H1 and H2 regression tests.
- `README.md`
  - Documented fail-closed discovery and the explicit `allowed_tools` option.
- `FIX_REPORT_V2_2026-09-09.md`
  - This report.

## Verification

### Full test suite with runtime dependencies available

Command:

```text
PYTHONPATH=/tmp/mnemos-audit-clones/mnemos-bridge-core/src /Users/jasonperlow/Projects/ic-engine/.venv/bin/python -m pytest -q -ra
```

Result:

```text
24 passed, 1 skipped in 0.85s
```

The one skipped test was the live integration test, which requires `OPENAI_API_KEY`, `MNEMOS_TEST_BASE`, and `MNEMOS_MCP_TOKEN`.

### Changed-file lint and formatting

Commands:

```text
/tmp/mnemos-audit-clones/mnemos-oauth-fix/venv/bin/ruff check src/mnemos_bridge_openai/adapter.py tests/test_handle_tool_call_offline.py tests/test_security_regressions.py
/tmp/mnemos-audit-clones/mnemos-oauth-fix/venv/bin/ruff format --check src/mnemos_bridge_openai/adapter.py tests/test_handle_tool_call_offline.py tests/test_security_regressions.py
git diff --check
```

Results:

```text
All checks passed!
3 files already formatted
git diff --check: clean
```

Repository-wide Ruff still reports the pre-existing `E402` at `tests/integration/test_openai_tool_loop.py:23`; this change introduces no new lint errors.

### Adversarial review gate

The required zoder review was attempted three times, including explicit model pins. All attempts failed before review because the configured NVIDIA reviewer endpoint was unreachable from the sandbox; zoder correctly reported `0/1 reviewers completed` and did not produce a false approval.

The repository's secondary-review escalation path was then used. The first independent Codex review found the overlapping-discovery race described above. After the generation/in-flight fix and its regression test were added, the reviewer reproduced the former race against the updated code and returned **APPROVE**, with no remaining blocking H1/H2 findings.

## Git handoff

- Commit identity: `Jason Perlow <jperlow@gmail.com>`
- Commit message: `fix(adapter): close audit security gaps`
- Push status: not pushed, as requested.
