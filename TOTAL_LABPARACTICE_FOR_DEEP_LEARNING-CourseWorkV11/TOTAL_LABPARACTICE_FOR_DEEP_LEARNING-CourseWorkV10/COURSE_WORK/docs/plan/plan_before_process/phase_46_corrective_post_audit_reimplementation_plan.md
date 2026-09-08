# Phase 46 — Corrective Post-Audit Reimplementation Plan

**Document version:** `phase_46_corrective_post_audit_reimplementation_plan.md`
**Created:** 2026-09-06
**Author:** Read-only audit + corrective governance (Part 2D / Part 2E)
**Classification:** Corrective pre-process plan — Phase 46
**Authority:** Supersedes the stale assumptions in `phase_46_corrective_reimplementation_plan.md` (2026-09-01). Specifically supersedes its `expected 50 epochs` assumption and its TR_C0-era expectations. Valid archive / checksum / Test-firewall procedures from that older plan are retained.
**Status:** PLAN_READY_FOR_HUMAN_REVIEW
**Execution authorized:** NO

---

## 0. TL;DR

The active Phase 46 artifacts were produced against an earlier configuration and do not match the current Phase 45 lock. A corrective Phase 46 re-materialization is required.

The corrective run is **not** a new scientific selection step. The model is already locked by Phase 45:

```
candidate_id        = TR_C2_ALT_LOOKBACK
lookback_steps      = 72
FINAL_REFIT_EPOCHS  = 30
seeds               = [42, 123, 2026]
config_fingerprint  = 585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24
final_lock_sha256   = 81fb87c44b6af31b6f65eff13956d9dc94a38dc75731a2c0af088228dd1bd4ec
```

These values are **strict** and come from the active Phase 45 lock (see `phase_46_47_lineage_drift_audit_2026_09_06.md` §2). The corrective run must source them verbatim. **No deviation is permitted without an amendment and re-approval of the Phase 45 lock itself.**

---

## 1. What this plan supersedes

The older in-progress plan (`docs/plan/plan_before_process/phase_46_corrective_reimplementation_plan.md`, 2026-09-01) contains stale assumptions:

- `FINAL_REFIT_EPOCHS expected = 50` — replaced by **30** (current Phase 45 lock).
- TR_C0_PRIMARY-era model references — replaced by **TR_C2_ALT_LOOKBACK**.
- Lookback 36 references — replaced by **72**.
- `config_fingerprint = a711a9b8...` references — replaced by **585c5e79...**.

Valid procedures from the older plan are retained:

- The 14 preflight-gate structure (extended below to 17 gates).
- The historical-checkpoint archive procedure (`artifacts/three_seed_final_runs/historical_checkpoints/<run_id>/`, `historical_checkpoint_archive_manifest.json`, `historical_checkpoint_archive_audit.csv`).
- The `phase47_test_release.json` gate semantics (`released=true` only after all gates pass).
- The Test firewall rules and the new-run-id-excludes-historical-run-ids rule.
- The `_history/PHASE{NN}_CORRECTIVE_<UTC>/` archive convention.

---

## 2. Hard corrective Phase 46 contract

### 2.1 Locked inputs (verbatim from current Phase 45)

```
candidate_id        = TR_C2_ALT_LOOKBACK
lookback_steps      = 72
final_refit_epochs  = 30
seeds               = [42, 123, 2026]
config_fingerprint  = 585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24
final_lock_sha256   = 81fb87c44b6af31b6f65eff13956d9dc94a38dc75731a2c0af088228dd1bd4ec
recipe_sha256       = 857dbaf7903792cdba3e126a50d8919992f02e0fff838cba86180026c9e0220c
population_sha256   = 552e6895dbc6f7a93b0ea9a2ab4de77abd58d8780ad4bc85b697a485ccbc31a9
lineage_sha256      = 9ba532539d548156e1d5d933abe8c1463617cdc790741e48d5249b452995d9fe
feature_sha256      = fc9c428964285d1ad74e97f3ae2c18e7efeb3e61d61f7acd0b54039bc2ca6dee
feature_variant     = FS2_TF1
target_scaling      = YS1
boundary_protocol   = WB0_CONTEXT_CARRY_OVER
```

`config_fingerprint` and `final_lock_sha256` are **two distinct identifiers**. They must be stored and compared separately at every artifact. Do **not** treat one as a substitute for the other.

### 2.2 Hard forbidden behaviors

The corrective Phase 46 run MUST NOT perform any of:

- Hyperparameter search.
- Best-seed selection.
- Early stopping.
- Validation-based stopping.
- Architecture change.
- Target-scaling change.
- Feature-variant change.
- Lookback change.
- Seed-set change.
- Ensemble creation.
- Post-Test tuning.
- Scaler refit on Test.
- Any Test-driven decision (including reading any Test y value).
- Warm start from any previous checkpoint.
- Optimizer-state reuse across seeds.
- Reusing the historical Phase 46 run IDs `RUN_TR_FSD_0153`, `RUN_TR_FSD_0154`, `RUN_TR_FSD_0155` (these are `INVALIDATED_BY_CONTRACT_DEVIATION` per the older corrective plan §12.2).

### 2.3 Hard required behaviors

For each of `seeds = [42, 123, 2026]`:

- Fresh model construction from initialization.
- Fresh optimizer construction from initialization.
- Same scientific config, same data, same scalers, same epoch count (30), same boundary protocol across all 3 seeds.
- No validation loader passed to the training engine.
- `evaluate_validation = False` set explicitly.
- `early_stopping_enabled = False` set explicitly.
- Save the FINAL_REFIT checkpoint exactly at `official_epoch = max_epochs = 30` (not BEST).
- Persist the actual `.pt` checkpoint binary at:
  - `artifacts/three_seed_final_runs/official_checkpoints/seed_42/seed_42_FINAL_REFIT.pt`
  - `artifacts/three_seed_final_runs/official_checkpoints/seed_123/seed_123_FINAL_REFIT.pt`
  - `artifacts/three_seed_final_runs/official_checkpoints/seed_2026/seed_2026_FINAL_REFIT.pt`
- Persist a sidecar metadata JSON next to each `.pt` carrying: `seed`, `run_id`, `phase`, `checkpoint_type`, `config_fingerprint`, `final_lock_sha256`, `population_sha256`, `recipe_sha256`, `feature_sha256`, `lineage_sha256`, `scaler_x_sha256`, `scaler_y_sha256`, `final_refit_epochs`, `lookback_steps`, `created_at`, `status`.
- Generate a new `run_id` from a `RUN_TR_FSD_0256+` prefix — never reuse `0153`/`0154`/`0155`.

### 2.4 Hard identity requirements

- All 3 seeds' `config_fingerprint` MUST be identical and equal to `585c5e79...` (Phase 45 lock).
- All 3 seeds' `final_lock_sha256` MUST be identical and equal to `81fb87c4...` (Phase 45 lock).
- All 3 seeds' `recipe_sha256` / `population_sha256` / `lineage_sha256` / `feature_sha256` MUST be identical and equal to the corresponding Phase 45 values.
- All 3 seeds' `model_state_sha256` MUST be **distinct** (each seed is an independent materialization).
- The 3 corrected `x_scaler_sha256` and `y_scaler_sha256` MUST be identical and equal to the FINAL_SCALING-v1 values produced from fitting on `FINAL_DEV_REGION-v1` (TRAIN+VALIDATION, Test excluded).

---

## 3. Required implementation fixes (DOCUMENTED ONLY — NOT EDITED IN THIS PLAN)

These fixes are required before the corrective Phase 46 run can execute. They are described here for documentation purposes only. **No code has been edited in this Part 2E task.**

| # | File | Fix |
|---|---|---|
| 1 | `src/course_work/scripts/p46_three_seed_runs.py:1530` | Remove the `or "TR_C0_PRIMARY"` fallback. The runner must STOP if `candidate_id` cannot be resolved from `phase46_three_seed_handoff.json`. No silent acceptance of a stale candidate. |
| 2 | `src/course_work/scripts/p46_three_seed_runs.py:1297, 1433` | Replace `int(phase46_handoff.get("FINAL_REFIT_EPOCHS", 30))` with an explicit read from `phase_45_signoff.json.final_refit_epochs`, plus a hard assertion that `phase46_handoff.FINAL_REFIT_EPOCHS == phase_45_signoff.final_refit_epochs`. STOP on mismatch. |
| 3 | `src/course_work/scripts/p46_three_seed_runs.py:424` | Replace `int(training.get("max_epochs", 50))` (default 50) with a read from the locked Phase 45 contract. STOP if `training.max_epochs != phase_45_signoff.final_refit_epochs`. |
| 4 | `src/course_work/final_test_evaluation/__init__.py:52` | Split the single `LOCKED_CONFIG_FP` constant into two: `LOCKED_CONFIG_FINGERPRINT` (the config_fingerprint) and `LOCKED_FINAL_LOCK_SHA256` (the final_lock_sha256). Update all references in `p47_final_test_evaluation.py` to use the correct one per field. |
| 5 | `src/course_work/scripts/p47_final_test_evaluation.py:191` | Write `LOCKED_FINAL_LOCK_SHA256` (not config fingerprint) into `final_test_evaluation_contract.json.final_lock_sha256`. |
| 6 | `src/course_work/scripts/p47_final_test_evaluation.py:257-258` | Split the two equality checks: `lock_hash_match` compares against `LOCKED_FINAL_LOCK_SHA256`; `config_hash_match` compares against `LOCKED_CONFIG_FINGERPRINT`. |
| 7 | (Phase 46 runner) | Add explicit persistence of the actual `.pt` checkpoint binaries, not just metadata. Verify SHA256 of each `.pt` matches metadata `model_state_sha256` before signing off. |
| 8 | (Phase 46 runner) | Enforce `identical config_fingerprint across all 3 seeds` as an explicit gate (not just an internal assertion). |
| 9 | (Phase 46 runner) | Refuse to emit `phase47_test_release.json` with `released=true` until all 17 pre-Test gates pass. During preflight, only emit `released=false` placeholder. |
| 10 | (Phase 46 runner) | Emit a `phase46_archive_manifest.json` documenting the preservation of the old Phase 46 evidence under `_history/PHASE46_PRE_CORRECTIVE_<UTC>/` before any corrected run starts. |
| 11 | NEW: `src/course_work/scripts/p46_corrective_pretrain_gate.py` | Consolidate the 17 pre-Test gates listed in §4 below. Replace the 14-gate structure currently in `p46_pretrain_gate.py` with this extended set. |
| 12 | NEW: `src/course_work/scripts/p47_archive_current_evidence.py` | Mirror the existing `p46_archive_historical_checkpoints.py` for Phase 47 evidence. Archive under `artifacts/final_test/_history/PHASE47_PRE_CORRECTIVE_<UTC>/`. |
| 13 | (Phase 47 runner) | Persist the actual prediction bundle CSVs (4 files) and write their SHA256 into `prediction_checksums.json`. The CSV files are absent from the current repo and must be materialized. |

These fixes will be **reviewed and approved by the human user** in a separate step before any implementation begins.

---

## 4. 17-gate pre-Test gate (HARD STOP if any fail)

The corrective Phase 46 must pass **all 17** of the following gates before any Phase 47 Test access is permitted. **Any single failure halts the entire pipeline.**

| Gate | Check | On FAIL |
|---|---|---|
| **G1** | `phase_45_signoff.json.status == "PASS"` AND `ready_for_phase46 == true` | STOP — Phase 45 invalid. |
| **G2** | `candidate_id == "TR_C2_ALT_LOOKBACK"` (from Phase 45, not from a `TR_C0_PRIMARY` fallback) | STOP — candidate drift. |
| **G3** | `lookback_steps == 72` | STOP — lookback drift. |
| **G4** | `final_refit_epochs == 30` | STOP — epoch drift. |
| **G5** | `seeds == [42, 123, 2026]` exactly, no additions, no substitutions | STOP — seed drift. |
| **G6** | All 3 seeds' `config_fingerprint` identical and equal to `585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24` | STOP — config drift. |
| **G7** | All 3 seeds' `final_lock_sha256` equal to `81fb87c44b6af31b6f65eff13956d9dc94a38dc75731a2c0af088228dd1bd4ec` | STOP — lock_sha drift. |
| **G8** | All 3 seeds' `recipe_sha256` equal to `857dbaf7903792cdba3e126a50d8919992f02e0fff838cba86180026c9e0220c` | STOP — recipe drift. |
| **G9** | All 3 seeds' `population_sha256` equal to `552e6895dbc6f7a93b0ea9a2ab4de77abd58d8780ad4bc85b697a485ccbc31a9` | STOP — population drift. |
| **G10** | All 3 seeds' `feature_sha256` equal to `fc9c428964285d1ad74e97f3ae2c18e7efeb3e61d61f7acd0b54039bc2ca6dee` | STOP — feature drift. |
| **G11** | Scalers fit on `FINAL_DEV_REGION-v1` (TRAIN+VALIDATION), `Test_count == 0`. `x_scaler_sha256` and `y_scaler_sha256` match across all 3 seeds. | STOP — scaler drift / Test leakage. |
| **G12** | All 3 `.pt` checkpoint binaries exist at canonical paths. SHA256 of each `.pt` equals metadata `model_state_sha256`. | STOP — checkpoint missing. |
| **G13** | No Test access occurred during Phase 46 (Test access log cleared for Phase 46). | STOP — Test firewall breach. |
| **G14** | No best-seed selection, no ensemble, no post-lock tuning, no warm start, no validation stopping, no optimizer-state reuse. | STOP — protocol drift. |
| **G15** | The 3 corrected `run_id`s are NOT equal to `RUN_TR_FSD_0153`, `RUN_TR_FSD_0154`, `RUN_TR_FSD_0155`. The corrected `phase47_test_release.json` includes a `historical_run_ids_excluded` field listing these three IDs. | STOP — historical id collision. |
| **G16** | `phase47_test_release.json` exists with `released == true`, `seed_count == 3`, all 3 corrected `run_id`s present, all 3 historical `run_id`s absent from `run_records`. | STOP — release gate not satisfied. |
| **G17** | `phase_46_signoff.json.overall_status == "PASS"` AND `discrepancies == []`. | STOP — signoff not PASS. |

### On any gate failure

```
STOP.
Do NOT access Test.
Do NOT execute Phase 47.
Report the failing gate to the human user.
```

---

## 5. Current-evidence preservation (BEFORE any corrected run starts)

Before the corrective Phase 46 run begins, the current active Phase 46 evidence must be archived. The archive procedure follows the project's existing convention (see `phase_46_corrective_reimplementation_plan.md` §12 and the existing `src/course_work/scripts/p46_archive_historical_checkpoints.py`).

### 5.1 Phase 46 archive destinations

```
COURSE_WORK/artifacts/three_seed_final_runs/_history/PHASE46_PRE_CORRECTIVE_<UTC>/
COURSE_WORK/artifacts/three_seed_final_runs/historical_checkpoints/
├── RUN_TR_FSD_0153_B15A19DC/   (seed 42, INVALIDATED)
├── RUN_TR_FSD_0154_DD82D743/   (seed 123, INVALIDATED)
└── RUN_TR_FSD_0155_59A50ADD/   (seed 2026, INVALIDATED)
+ historical_checkpoint_archive_manifest.json
+ historical_checkpoint_archive_audit.csv
+ _archive_manifest.json
```

### 5.2 Files to preserve (Phase 46)

| Source path | Status |
|---|---|
| `phase_46_signoff.json` | STALE (`TR_C0_PRIMARY` / 50 epochs / `average_rmse_wh: 0.0`); preserved, not deleted. |
| `three_seed_manifest.json` | STALE; preserved. |
| `three_seed_contract.json` | Preserved. |
| `three_seed_final_runs_summary.json` | Preserved (real per-seed rmse 41.71 / 39.62 / 39.89). |
| `final_lock_verification.json` | STALE (verifies `a711a9b8...` universe); preserved. |
| `phase46_reuse_cache.json` | Preserved (placeholder rmse). |
| `phase47_final_test_evaluation_handoff.json` | Preserved. |
| `final_dev_population_manifest.json` | Preserved. |
| `official_checkpoints/seed_*/seed_*_FINAL_REFIT.pt` | If present: copied to archive; if absent: noted as missing. |
| `official_checkpoints/seed_*/seed_*_FINAL_REFIT_metadata.json` | Preserved. |
| `README_THREE_SEED_FINAL_RUNS.md` | Preserved. |
| `figures/*` | Preserved. |

### 5.3 Archive procedure (safe by construction)

For each file in §5.2:

1. Compute SHA256 of source.
2. Copy (not move) into the archive destination under `PHASE46_PRE_CORRECTIVE_<UTC>/`.
3. Compute SHA256 of the archived copy.
4. Require `source_sha == archived_sha`.
5. Append to `_archive_manifest.json` with `reason = "PHASE46_PRE_CORRECTIVE"`.
6. Only AFTER all files are checksum-verified, write a placeholder metadata at the canonical path declaring `status = AWAITING_CORRECTED_PHASE46_RUN`. Do **not** write fake corrected data.

**Rollback on any SHA mismatch.** No source file is removed unless its archived copy verifies the same SHA.

### 5.4 Additive (non-destructive) invalidation manifests

Write additive `historical_phase46_invalidation_manifest.json` into each `artifacts/runs/RUN_TR_FSD_015{3,4,5}_*/` directory, recording:

- The historical `run_id`.
- The reason for invalidation: `PHASE46_HISTORICAL_INVALIDATED_BY_PROTOCOL_DEVIATION`.
- Reference to this corrective plan and the audit document.

**Never delete or modify the historical run directories.** They remain valid as historical evidence.

---

## 6. Expected corrective Phase 46 outputs

### 6.1 Per seed

```
artifacts/three_seed_final_runs/official_checkpoints/seed_<seed>/seed_<seed>_FINAL_REFIT.pt            (binary)
artifacts/three_seed_final_runs/official_checkpoints/seed_<seed>/seed_<seed>_FINAL_REFIT_metadata.json  (sidecar)
run_id ∈ RUN_TR_FSD_0256+                                                                            (NEVER 0153/0154/0155)
```

### 6.2 Collective

```
artifacts/three_seed_final_runs/three_seed_manifest.json                    (candidate_id=TR_C2_ALT_LOOKBACK, final_refit_epochs=30, config_sha256=585c5e79..., final_lock_sha256=81fb87c4...)
artifacts/three_seed_final_runs/three_seed_final_runs_summary.json          (per-seed FINAL_DEV RMSE diagnostic only)
artifacts/three_seed_final_runs/final_lock_verification.json                (all 4 hashes match Phase 45)
artifacts/three_seed_final_runs/final_dev_population_manifest.json          (Test excluded)
artifacts/three_seed_final_runs/phase47_final_test_evaluation_handoff.json  (handoff to Phase 47)
artifacts/three_seed_final_runs/phase_46_signoff.json                       (overall_status=PASS, discrepancies=[])
artifacts/three_seed_final_runs/phase47_test_release.json                   (released=true ONLY after 17 gates pass)
artifacts/three_seed_final_runs/phase46_archive_manifest.json               (NEW — documents §5 preservation)
artifacts/three_seed_final_runs/figures/FINAL_46_01_training_variability.png
```

---

## 7. Expected run budget (indicative only)

- 3 FINAL_REFIT training runs × 30 epochs × TR_C2_ALT_LOOKBACK architecture.
- Wall-clock estimate depends on hardware. On a single MPS device for ~16,630 training rows × 33 features × lookback 72, ~6-10 minutes per seed for 30 epochs (Transformer 2-layer, d_model=64, num_heads=4).
- Indicative total wall-clock: ~25-35 minutes on MPS / single GPU / CPU. **Label as estimate only.**

This is the same model and the same epoch count already approved by Phase 45. No new scientific compute is being requested.

---

## 8. Stop conditions

STOP and DO NOT proceed if any of the following occur:

1. Phase 45 lock artifacts not found or invalid.
2. `phase46_three_seed_handoff.json` does not exist or has wrong values.
3. `phase47_test_evaluation_guard.json` indicates Test access has occurred.
4. `candidate_id` cannot be resolved from `phase46_three_seed_handoff.json` (after removal of `TR_C0_PRIMARY` fallback per §3 #1).
5. Any of the 17 pre-Test gates in §4 fails.
6. Any unit / integration test fails.
7. The corrected preflight script reports a non-PASS gate.
8. Any `.pt` checkpoint binary cannot be persisted or its SHA256 cannot be matched against metadata.
9. Historical checkpoint bytes are modified or deleted (after archive verification passes).
10. Any Test access is detected during Phase 46 execution.

---

## 9. Human approval gates

Per `rule_code.md` Rule #10-#14 and Rule #76, no execution proceeds without explicit human approval:

1. **Human approval #1:** Review of this plan + the root-cause audit + the Phase 47 corrective plan. Authorizes code implementation per §3.
2. **Human approval #2:** Review of the implementation diff. Authorizes the corrective Phase 46 training run.
3. **Human approval #3:** Review of the corrective Phase 46 PASS signoff (with all 17 gates passing). Authorizes corrective Phase 47 Test re-evaluation.

Each approval is explicit and recorded. No auto-execution under any circumstance.

---

## 10. Status

```
STATUS: PLAN_READY_FOR_HUMAN_REVIEW
EXECUTION_AUTHORIZED: NO

MODEL_SELECTION_FROZEN_AT_PHASE_45: YES
CONFIG_FINGERPRINT: 585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24
FINAL_LOCK_SHA256:  81fb87c44b6af31b6f65eff13956d9dc94a38dc75731a2c0af088228dd1bd4ec
CANDIDATE_ID:       TR_C2_ALT_LOOKBACK
LOOKBACK_STEPS:     72
FINAL_REFIT_EPOCHS: 30
SEEDS:              [42, 123, 2026]

PRIOR_PHASE_46_STATUS: STALE / INVALIDATED_BY_CONTRACT_DEVIATION
ARCHIVE_PLAN_DEFINED: YES
ARCHIVE_EXECUTED:     NO
TRAINING_EXECUTED:    NO
```

---

**This plan is documentation only. It does not authorize any execution.**
