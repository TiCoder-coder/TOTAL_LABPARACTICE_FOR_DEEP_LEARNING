# Phase 43 Corrective Implementation Plan

**Date:** 2026-09-02  
**Status:** APPROVED-BY-USER-PROMPT (recovery plan + 14-step task already specified)  
**Scope:** Read-only inspection, Phase 43 implementation modification, tests, dry-run utilities. NO scientific training, NO new LSTM_TUNING scientific runs, NO Test access, NO modification to Phase 45/46.

---

## 1. Confirmed Root Causes (from physical artifacts)

| ID | Deviation | Evidence |
|---|---|---|
| A | Scientific override `max_epochs=2, patience=2` | `scripts/phase43_lstm_tuning.py:169-170`; all 13 LT/LST runs in registry have `max_epochs: 2, patience: 2` |
| B | Winner `RUN_LS_LST_0049_7B728C6B` produced under truncated training | `best_epoch: 2` in winner config and registry |
| C | Signoff reports PASS despite budget violation | `artifacts/lstm_tuning/phase_43_signoff.json` `status: PASS, ready_for_phase44: true` |
| D1 | Lookback mismatch: signoff/handoff say 36, winner config + 35 LSTM_TUNING runs say 144 | cross-artifact comparison |
| D2 | Stage winner HP mismatch: signoff reports `H64/N2/LR3e-4/WD0.001`; registry + winner config + lstm_stage_lineage.csv show `H128/N1/LR1e-3/WD1e-4`; dropout skipped because N1 won but reported as `lt3_dropout: null` | cross-artifact comparison |
| D3 | Runtime DataLoader batch_size=1024 vs config batch_size=64 | `scripts/phase43_lstm_tuning.py:131` |
| D4 | All dummy audit CSVs say PASS without real checks | 10 audit CSVs in `artifacts/lstm_tuning/` |
| D5 | `validation_mae_wh: 0.0, validation_r2: 0.0` in winner | `lstm_tuned_winner.json` |
| D6 | signoff `reference_run_id = RUN_TR_S14_0023_A711A9B8` (Transformer Phase 42 primary) but `reference_source = REUSED_PHASE20` — false | cross-reference |
| D7 | Reference RMSE `79.0749631124407` from Transformer primary run, not from a real LSTM_B0 | cross-reference |
| D8 | LT3 formally `SKIPPED` (because N1 won) but no fallback to dropout=0 (winner said `effective_dropout=0`); LT5 winner run `RUN_LS_LST_0049_7B728C6B` = same as LT4 winner (LWD1 = same config as LWD3+dr=0), so LWD2 `RUN_LS_LST_0052_E28F7D72` exists but not selected | structural check |
| D9 | Population fingerprint hardcoded literal `a40ded88...` in script instead of resolved from upstream context | `scripts/phase43_lstm_tuning.py:166` |
| D10 | All 22 RO1-A/B/RO2-A/B/RO3-A/B LSTM_TUNING runs in registry are Phase 44 (not Phase 43) but share Phase 43 contamination | registry |

---

## 2. Implementation Plan

### 2.1 Preserve historical Phase 43 evidence (Step 2)
- Write `COURSE_WORK/artifacts/lstm_tuning/_history/PHASE43_RECOVERY_20260902/invalidation_manifest.json` listing all 35 LSTM_TUNING runs.
- Each entry: `run_id, classification: HISTORICAL_INVALIDATED_BY_PROTOCOL_DEVIATION, reason: PHASE43_FAST_MODE_E2_P2_VIOLATION, fingerprint, max_epochs, patience, lookback`.
- Do NOT mutate registry, run bytes, or status.

### 2.2 Shared data contract (Step 3)
- New module `src/course_work/lstm_tuning/__init__.py` + `shared_data_contract.py`.
- Reads `artifacts/candidate_synthesis/phase43_lstm_tuning_handoff.json` and resolves canonical:
  - `lookback = 36` (handoff)
  - `target_scaling = YS1`
  - `feature_variant = FS2_TF1`
  - `boundary_protocol = WB0`
  - `forecast_horizon = H1`
  - `window_population = WINDOWPOP-v1`
  - `B*` from dataloader registry fingerprint.
  - X/Y scaler checksums from upstream context (NOT hardcoded literal).
  - Train target IDs and Validation target IDs fingerprints from upstream context.
- Output `lstm_shared_data_contract.json` (replace dummy lookback=144 with canonical 36).

### 2.3 Preflight audit (Step 3 sub)
- Producer for `phase43_preflight_audit.csv` and `lstm_common_data_audit.csv` that does real checks (status: PASS/FAIL) using shared_data_contract helpers.

### 2.4 Remove fast-mode (Step 4)
- `scripts/phase43_lstm_tuning.py` REWRITTEN to NOT override `max_epochs/patience`.
- Uses `build_reference_run_config(root, "LSTM")` directly which already returns `max_epochs=50, patience=10`.
- Adds explicit assertion: runtime config `max_epochs == 50 and patience == 10`.

### 2.5 Fix DataLoader (Step 5)
- Use `train_loader`/`val_loader` from `build_train_validation_loaders(...)` tuple, NOT rewrap with `batch_size=1024`.
- Runtime batch size = `baseline_batch_size` from dataloader upstream context.
- Add assertion: `train_loader.batch_size == run_config["training"]["batch_size"]`.

### 2.6 LT1–LT5 (Step 6)
- New module `src/course_work/lstm_tuning/tuning_space.py` exporting:
  - LT1 candidates, LT2 candidates, LT3 applicability, LT4 candidates, LT5 candidates.
  - `build_stage_candidates(current_winner_config, stage) -> dict[str, dict]`.
- New module `src/course_work/lstm_tuning/stages.py` exporting:
  - `run_stage(stage, candidates, registry, engine, ...)` that registers each candidate, ensures one-factor diff, returns stage result with winner.
- Reference inheritance via config equality check; NO warm-start; fresh seed42.

### 2.7 Winner-selection (Step 7)
- Stage winners selected by full-precision Validation RMSE Wh (no rounding).
- Exact tie parsimony per plan (smaller hidden / fewer layers / lower dropout / lower LR / lower WD).
- Failed candidate retained (status FAILED in registry; row present in metrics CSV with empty cells but NOT omitted).
- Final `LSTM_TUNED` = LT5 winner run_id (a real registered, completed run).

### 2.8 Stage lineage consistency (Step 8)
- `lstm_stage_lineage.csv` produced by code, not hard-coded.
- `lstm_tuned_winner.json` produced from final LT5 winner config fingerprint.
- Cross-artifact consistency check (assertion + JSON producer): signoff HP must equal winner config HP.

### 2.9 O43.1–O43.37 producers (Step 9)
- Each producer real:
  - `lstm_tuning_manifest.json` (status: PREPARED)
  - `lstm_tuning_contract.json` (max_epochs=50, patience=10, MSE, AdamW, GC1)
  - `phase43_preflight_audit.csv`
  - `lstm_shared_data_contract.json`
  - `lstm_reference_resolution.json`
  - `lstm_reference_audit.csv`
  - `lstm_tuning_space.json`
  - `lstm_run_matrix.csv`
  - `lstm_stage_lineage.csv`
  - `lt1_hidden_size_metrics.csv` + `lt1_hidden_size_winner.json`
  - `lt2_layers_metrics.csv` + `lt2_layers_winner.json`
  - `lt3_dropout_applicability.json` + `lt3_dropout_metrics.csv` + `lt3_dropout_winner.json` (or SKIPPED_NOT_APPLICABLE)
  - `lt4_learning_rate_metrics.csv` + `lt4_learning_rate_winner.json`
  - `lt5_weight_decay_metrics.csv` + `lt5_weight_decay_winner.json`
  - `lstm_architecture_audit.csv`
  - `lstm_training_config_delta_audit.csv`
  - `lstm_common_data_audit.csv`
  - `lstm_initialization_audit.csv`
  - `lstm_sample_order_audit.csv`
  - `lstm_optimizer_group_audit.csv`
  - `lstm_optimizer_budget_audit.csv`
  - `lstm_gradient_diagnostics.csv`
  - `lstm_convergence_diagnostics.csv`
  - `lstm_runtime_diagnostics.csv`
  - `lstm_run_provenance.csv`
  - `lstm_tuning_effect.csv`
  - `lstm_contextual_baseline_comparison.csv`
  - `lstm_tuned_winner.json`
  - `phase44_rolling_origin_lstm_handoff.json` (status: NOT_READY — depends on corrected Phase 43)
  - `lstm_tuning_findings.csv`
  - `lstm_tuning_tests.csv`
  - `lstm_tuning_discrepancies.json` (lists preserved historical invalidation + new fix-time discrepancies)
  - `lstm_tuning_summary.json`
  - `lstm_tuning_report.md`
  - `figures/*.png` (real plots or omitted per plan)
  - `README_LSTM_TUNING.md`
  - `phase_43_signoff.json`

### 2.10 Signoff gate (Step 10)
- `phase_43_signoff.json.overall_status ∈ {NOT_RUN, PREPARED}` before human scientific training.
- After human training, real status determined from actual run evidence.

### 2.11 Tests (Step 11)
New file `tests/unit/test_phase43_corrective_implementation.py` with ≥24 focused tests covering:
1. official Phase43 max_epochs == 50
2. official patience == 10
3. no scientific max_epochs=2 path
4. no scientific patience=2 path
5. smoke mode cannot write official artifacts
6. exact Phase42 shared context resolution
7. lookback exact parity (36)
8. Train target IDs parity
9. Validation target IDs parity
10. X scaler checksum parity
11. Y scaler checksum parity
12. runtime DataLoader batch size == run config batch size
13. LT1 candidate set exact = [32,64,128]
14. LT2 candidate set exact = [1,2]
15. LT3 applicability logic exact
16. LT4 candidate set exact = [1e-4, 3e-4, 1e-3]
17. LT5 candidate set exact = [0, 1e-4, 1e-3]
18. one-factor-only delta per stage
19. current reference included per stage
20. winner becomes next reference
21. no warm-start (different configs -> different seed)
22. no optimizer-state reuse (verified by fresh seed per candidate)
23. Test firewall (no test sample_count, test_loss, test_loader access)
24. final signoff/winner/stage-lineage consistency
25. max fresh scientific runs <= 10

### 2.12 Non-scientific dry-run (Step 12)
New script `scripts/phase43_dry_run.py`:
- Loads handoff, validates shared data contract.
- Resolves reference config.
- Materializes LT1-LT5 candidates.
- Computes planned run budget.
- Validates DataLoader batch parity.
- Validates artifact schemas.
- Validates Test firewall.
- STOPS before `registry.register_run()` and `engine.train()` and any optimizer step.
- Output: structured print + `artifacts/lstm_tuning/_history/PHASE43_RECOVERY_20260902/dry_run_report.json`.

### 2.13 Regression safety (Step 13)
- Do NOT modify `src/course_work/data/`, `models/`, `training/engine.py`, `evaluation/`, `experiments/registry.py`, `attention/`, `baselines/`, `sanity/`, `sweeps/`, `diagnostics/`, `reporting/`, `utils/`.
- Do NOT modify Phase 1-42 artifacts.
- New code lives under `src/course_work/lstm_tuning/` + revised `scripts/phase43_lstm_tuning.py` + new `scripts/phase43_dry_run.py` + new `tests/unit/test_phase43_corrective_implementation.py` + additive `_history/PHASE43_RECOVERY_20260902/` files.

### 2.14 Stop (Step 14)
After all gates pass:
- STOP. Do NOT execute official training.
- Provide exact human terminal command + caffeinate macOS command.

---

## 3. Files to Modify / Create

### Modified
- `COURSE_WORK/scripts/phase43_lstm_tuning.py` — remove fast-mode, fix DataLoader, real LT1-LT5 producer, real O43 producers, signoff gate (status PREPARED).
- (No upstream module changes)

### Created
- `COURSE_WORK/src/course_work/lstm_tuning/__init__.py`
- `COURSE_WORK/src/course_work/lstm_tuning/shared_data_contract.py`
- `COURSE_WORK/src/course_work/lstm_tuning/reference_resolution.py`
- `COURSE_WORK/src/course_work/lstm_tuning/tuning_space.py`
- `COURSE_WORK/src/course_work/lstm_tuning/stages.py`
- `COURSE_WORK/src/course_work/lstm_tuning/winners.py`
- `COURSE_WORK/src/course_work/lstm_tuning/artifacts.py`
- `COURSE_WORK/src/course_work/lstm_tuning/preflight.py`
- `COURSE_WORK/scripts/phase43_dry_run.py`
- `COURSE_WORK/tests/unit/test_phase43_corrective_implementation.py`
- `COURSE_WORK/artifacts/lstm_tuning/_history/PHASE43_RECOVERY_20260902/invalidation_manifest.json`
- `COURSE_WORK/artifacts/lstm_tuning/_history/PHASE43_RECOVERY_20260902/dry_run_report.json`

### NOT touched
- Phase 1-42 artifacts.
- `src/course_work/data/`, `models/`, `training/`, `evaluation/`, `experiments/`, `attention/`, `baselines/`, `sanity/`, `sweeps/`, `diagnostics/`, `reporting/`, `utils/`.
- All historical `artifacts/runs/RUN_LS_LST_*` files and registry entries.
- Phase 45/46 outputs.
- Test-related files.

---

## 4. Stop Conditions

- Rule conflict with `working_rule.md`, `architecture_rule.md`, `rule_code.md` → STOP.
- Phase 42 signoff altered → STOP.
- Test target accessed → STOP.
- Scientific training executed by agent → STOP and revert.
- Modification of upstream Phase 1-42 artifacts → STOP and revert.
- Any scope expansion outside the 14-step task → STOP and report.

---

## 5. Definition of Done

- All O43.1–O43.37 producers implemented and tested.
- ≥24 focused tests PASS.
- Dry-run PASS (no registry.register_run, no engine.train).
- Phase 43 signoff status = PREPARED (NOT_PASS).
- 0 official scientific run IDs created by agent.
- 0 Test accesses by agent.
- READY_FOR_PHASE43_HUMAN_RERUN = YES.
- Exact human command + caffeinate macOS command returned.
