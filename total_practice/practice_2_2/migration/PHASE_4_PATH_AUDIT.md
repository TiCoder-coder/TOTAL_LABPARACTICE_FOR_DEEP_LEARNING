# Phase 4 — Runtime Path Dependency Audit

This inventory was completed before Phase 4 path modules were added.

| File/scope | Path usage | Current behavior | Risk | Recommended handling |
|---|---|---|---|---|
| `src/practice_2_2/__init__.py` | `Path(__file__).parents[3] / "practice_2"` | Finds the old shared source by package location | MEDIUM: fixed tree-depth and sibling-name assumption | Delegate root discovery to `paths.py`; retain sibling layout contract |
| Copied canonical training/E3/E4 modules | Dataset, manifest, runs and outputs are mandatory function/CLI parameters | Not cwd-dependent when callers pass absolute paths | LOW | Do not change algorithms; resolver supplies explicit absolute paths to callers |
| Copied data/audit modules | Accept explicit dataset/manifest/output paths | `Path(...).resolve()` interprets relative caller input using cwd | MEDIUM for ad-hoc callers | Canonical runtime should obtain paths from registry; preserve generic APIs |
| `final_test_practice_2_2.py` | Explicit dataset, manifest, checkpoint and final directory arguments | Guarded but high-risk entry point | CRITICAL if executed | Do not modify or execute; resolver exposes read-only Final Test resources only |
| `analyze_practice_2_2.py` | Explicit staging root; `/private/tmp` matplotlib cache default | No canonical discovery; temporary cache is machine-independent POSIX path | LOW | Leave algorithm unchanged; registry not required for staging analysis |
| Shared wrappers | Import `processing_own_phase.*` | Depends on package initializer adding sibling Practice 2 root | MEDIUM | Use cwd-independent `get_practice_2_root()` in package initializer |
| `tests/test_submission_practice_2_2.py` | `Path(__file__).parents[...]` | Works from installed source tree but duplicates discovery logic | LOW | New registry tests become canonical path contract; existing test remains unchanged |
| Other copied tests | `tmp_path` fixtures and explicit paths | Isolated and cwd-independent | SAFE | No change |
| Immutable canonical JSON | Persisted `/Users/vientu/...` checkpoint/dataset paths | Works only while old absolute location exists | HIGH for relocated checkout | Read-only adapter falls back to registry and verifies SHA-256; never rewrite JSON |
| Canonical notebook | Searches parents of `Path.cwd()` for repository | Presentation copy still depends on launch location | OUT OF PHASE | Canonical entry point is not switched in Phase 4; notebook is not modified |
| Canonical report/HTML paths | Fixed by current project layout | No registry abstraction | LOW | Register relative paths from Practice 2.2 root |

## Findings

- No user-home absolute path literal exists in the new Python package or new tests.
- The only absolute literal is a `/private/tmp` matplotlib cache fallback.
- Canonical functions already accept explicit paths; Phase 4 does not alter their algorithms or defaults.
- The package compatibility initializer had no cwd dependency, but it encoded tree depth directly. Phase 4 centralizes this logic.
- Immutable artifacts contain historical absolute paths and require a read-only fallback adapter rather than edits.
