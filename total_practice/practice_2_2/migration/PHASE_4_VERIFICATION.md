# Phase 4 — Runtime Path and Registry Verification

## Implementation

- `paths.py` discovers repository, Practice 2 and Practice 2.2 roots by traversing from `__file__`, with optional validated `PRACTICE_2_2_REPO_ROOT` override.
- `canonical_registry.json` contains only Practice 2.2-root-relative paths.
- `resources.py` resolves existing resources, verifies hashes/fingerprints and fails closed when resources are missing.
- The immutable adapter reads JSON without mutation and falls back to a hash-verified registry resource only when a persisted path is missing.
- The old shared implementation remains the source of truth; canonical entry points were not switched.

## Working-directory matrix

| Working directory | Root/resource resolution |
|---|---|
| Repository root | PASS |
| `total_practice/` | PASS |
| `practice_2/` | PASS |
| `practice_2_2/` | PASS |
| `/private/tmp` | PASS |

Every location resolved the same dataset root, split manifest, E2 checkpoint, Final Test directory, report notebook and HTML report.

## Canonical verification

| Contract | Result |
|---|---|
| Live dataset fingerprint | `26dc4625f96c7cb86bc6a4df0fbed34bc8b22fb6472ef0d008df341f99f71fd6` — PASS |
| Split fingerprint | `52aaf97499ede5dd4689c2bb36dc0679fe04b8ec8139aae7e8fdcde88042043d` — PASS |
| Split manifest SHA-256 | `aa7d6d4bb77f8ca6e7e5599ac2d49e8d55ce8192e8353f5cb90bbd0f0a4f49b4` — PASS |
| E2 checkpoint SHA-256 | `4f65bec200d023c51f95493833d057029b83a92729c3b88dd9159158dd949ae3` — PASS |
| Final Test guard SHA-256 | `532172911f09936c33d5fb999effe0e516ffd8e6f6e2bc36df9cd5f92c53e345` — PASS |
| Train / Validation / Test manifest counts | 2016 / 438 / 440 — PASS |
| Generated excluded / quarantine | 306 / 2 — PASS |

## Checkpoint reload Validation

| Metric | Result |
|---|---:|
| Validation samples | 438 |
| Accuracy | 78.31050228310502% |
| Accuracy delta | 0.0 |
| Loss | 1.021770380947688 |
| Loss delta | 0.0 |
| Macro F1 | 0.7820960879325867 |
| Test DataLoader constructed | false |
| Final Test evaluated | false |

## Automated tests

- New package/tests: **43 passed in 3.89s**.
- Full old Practice 2 regression suite: **87 passed, 3 deprecation warnings in 12.45s**.
- Old and new imports: PASS.
- Registry, path discovery, immutable adapter, submission safety and Test guard: PASS.

No canonical resource was copied, moved, renamed, rewritten or regenerated.
