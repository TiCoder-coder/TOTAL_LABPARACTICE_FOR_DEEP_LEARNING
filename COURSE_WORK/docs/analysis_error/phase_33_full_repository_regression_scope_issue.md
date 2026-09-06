# Phase 33 Full Repository Regression Scope Issue

## 1. Issue identity

```text
Issue ID: CW-PHASE-33-FULL-REPOSITORY-REGRESSION-SCOPE-001
Detected after: Phase 33 canonical completion
Phase 33 acceptance impact: NONE
Current status: ISOLATED
```

## 2. Verified Phase 33 state

```text
Phase state: VALID_REUSABLE
Resolved action: RENDER_ONLY
Canonical sign-off: PASS
Relevant tests: 86 PASS
Registry audit: 13 PASS, 0 FAIL
Test metric count: 0
Notebook checksum: preserved
Phase 32 canonical checksum set: preserved
```

## 3. Full-suite findings

The repository-wide test run produced 256 passes, 7 failures and 6 setup errors.

The findings are outside the Phase 33 scientific path:

```text
Six splitting setup errors and one EDA failure are caused by sandboxed MPS visibility during Phase 1 materialization.
Two integration assertions still require a one-run Phase 13 registry although the approved sweep chain now contains nineteen completed runs.
One integration assertion treats mutable registry-derived files as immutable Phase 13 or Phase 14 outputs.
One presentation assertion detects stale source checksums in Phase 15-22 processing logs.
One acquisition assertion detects two Phase 2 files changed after their signed checksum was created.
One notebook assertion rejects the preserved mixed execution-count state of forty-eight executed cells and one unexecuted cell.
```

## 4. Boundary decision

No Phase 33 source, artifact, winner or processing log may be changed to conceal these findings.

The notebook must not be cleared, re-executed or normalized under the Phase 33 plan because its exact checksum was explicitly protected.

Historical Phase 2 and Phase 13-22 artifacts must not be silently re-signed. Their correction requires a separate revision-aware recovery decision.

## 5. Conclusion

Phase 33 is complete and scientifically reusable. The full-suite findings form separate repository-maintenance debt and are governed by the corresponding corrective plan.
