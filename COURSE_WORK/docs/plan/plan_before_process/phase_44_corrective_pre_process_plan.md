# Phase 44 — Corrective Pre-Process Plan
## Rolling-Origin Robustness

**Document version:** `phase_44_corrective_pre_process_plan.md`
**Created:** 2026-09-02
**Author:** Senior Time-Series ML Engineer / Scientific Protocol Auditor
**Classification:** Corrective Pre-Process Plan — Phase 44
**Authority:** Supersedes the historical Phase 44 implementation
**Status:** PLAN ONLY — implementation requires human approval

---

## 0. TL;DR

The current `scripts/phase44_rolling_origin.py` and its resulting `phase_44_signoff.json`
are HISTORICAL and were produced **before** the corrected Phase 43 (`status: PASS`,
winner `RUN_LS_LST_0175_BCE5A2CD`, fingerprint `bce5a2cd…`).

The historical Phase 44 implementation has **at least 16 CRITICAL protocol deviations**
relative to the Phase 44 plan (`Phase_44_Rolling-origin_robustness.md`, sections 1-198).
It cannot be salvaged by patch edits. A full corrective re-implementation is required,
backed by new reusable modules under `src/course_work/rolling_origin/`, a fold-local
scaling subsystem (`RO_SCALING-v1`), extension of `TrainingEngine` with
`evaluate_validation=False` + exact-epoch refit, and a target-ID subset loader helper.

The corrected execution will:

1. Consume the corrected Phase 43 winner verbatim.
2. Rebuild the Transformer fold manifest with 3 expanding origins over
   `WINDOWPOP-v1 \ Test`.
3. For each (learned candidate × fold), run Stage A (epoch selection on inner block)
   and Stage B (fresh full-history refit at the selected epoch) on **fold-local
   scalers fit only on allowed training history**.
4. Use proper `dataset.Subset` over the candidate's full `SequenceWindowDataset` and
   override scaler bundles per fold/stage via a new loader builder.
5. Implement **true pooled metrics** from concatenated outer predictions
   (NOT from `mean(fold_RMSE)`).
6. Implement persistence with an explicit prior-history lookup keyed by
   `target_timestamp - 10min`, not by shifting `y_true`.
7. Archive the historical Phase 44 artifacts (no deletion), archive invalid
   `ROLLING_ORIGIN` runs in the registry (do not reuse), and register a new
   family of scientific runs (`ROLLING_ORIGIN_INNER_SELECTION`,
   `ROLLING_ORIGIN_REFIT`, `ROLLING_ORIGIN_PERSISTENCE`).
8. Provide a `--dry-run` preflight that performs ZERO optimizer steps and a
   disposable rehearsal harness that exercises the entire code path against a
   synthetic dataset.

The expected official scientific budget is **(3 + 1) × 3 × 2 = 24** learned
training jobs (Transformer × 3, LSTM × 1, K=3 folds, Stage A + Stage B per fold)
plus 3 persistence evaluations.

Test firewall is preserved at every level.

---

## 1. Problem statement

`artifacts/rolling_origin/phase_44_signoff.json` carries:

- `phase43_lstm_fingerprint = "3715a8a7d26d6d7d6f10684d6f2edbe1f4c6792c2cb8d5264edb3d9d341a3b31"`
- `recommended_transformer_fingerprint = "a711a9b8e2f23c43e3409a5f01bf1946c0a7f292d1028f92df33bee06fbd1b14"`
- `recommended_pooled_rmse_wh = 61.50746536254883`

The corrected Phase 43 winner fingerprint is
`bce5a2cd6ba86435b7c02a1f1a9d25a6e214493dbd8d6da22886e328287f8593`. The historical
Phase 44 therefore consumed a stale Phase 43 lineage. In addition, the Phase 44
driver script contains multiple protocol violations documented below.

The output of the historical run is therefore labelled `HISTORICAL /
PENDING CORRECTIVE RERUN` and is **not** scientific evidence for the corrected
pipeline.

---

## 2. Root causes (audit summary)

The detailed audit is in §4. The condensed root-cause matrix is:

| # | Root cause | Severity |
|---|---|---|
| 1 | Phase 44 script not updated after Phase 43 corrective reimplementation (consumes stale Phase 43 winner via fingerprint mismatch) | CRITICAL |
| 2 | `stage_a_cfg["training"]["max_epochs"] = 2`, `patience = 2` — fast-mode override on every Stage A run | CRITICAL |
| 3 | Stage A `inner_train_loader` and `inner_val_loader` use the full Train / full Validation datasets instead of fold-specific subsets | CRITICAL |
| 4 | Stage B `outer_history_loader` uses the full Train dataset for RO1 and Train+V_n for RO_n but is built without true fold-local scalers | CRITICAL |
| 5 | Stage B `inner_val_loader` is passed as the validation loader (incorrect usage even with `evaluate_validation=False`) | CRITICAL |
| 6 | Stage B "no validation selection" is asserted by `early_stopping_enabled=False` only; the `TrainingEngine` signature passes `evaluate_validation` which is left at default | CRITICAL |
| 7 | Global `load_validated_target_scaler` (Phase 9 Train-only YS1) is reused for every fold and every stage | CRITICAL |
| 8 | Stage A population fingerprint is hard-coded to `"a40ded8802…"` — fold-local pop fingerprint is never computed | MAJOR |
| 9 | Persistence prediction is implemented as `y_pred = [y_true[0]] + y_true[:-1]`, i.e. shifting the outer y_true vector — this misrepresents the semantics at the first target of each block | CRITICAL |
| 10 | Pooled RMSE is computed as `sqrt(mean(fold_RMSE**2))` — incorrect unless fold sizes are equal and the per-fold residuals are pooled first | CRITICAL |
| 11 | Per-fold outer prediction bundles are never saved (O44.21/O44.22 missing) | MAJOR |
| 12 | Stage A/B scaler-fit audit (O44.10), temporal leakage tests (O44.11), common-target audit (O44.12), refit-epoch audit (O44.16), gradient diagnostics (O44.19), runtime diagnostics (O44.20), pairwise effects (O44.26), fold ranks (O44.27), model-family comparison (O44.29) — all missing | MAJOR |
| 13 | Stage A/B registry `rerun_reason="REPRODUCIBILITY_CHECK"` is invalid for non-duplicate registrations (per registry policy) | MAJOR |
| 14 | The same `ExecutionType.TRAINING` is used for both Stage A and Stage B; plan requires `ROBUSTNESS` execution type | MINOR |
| 15 | The script registers Stage A and Stage B under `experiment_family` `"ROLLING_ORIGIN"` for Transformers but `"LSTM_TUNING"` for LSTM — inconsistent family routing | MAJOR |
| 16 | No parent_run_id linkage between Stage A → Stage B; no candidate_id or fold_id propagated to registry metadata | MAJOR |
| 17 | No auditable ensemble of O44.4 (frozen fold manifest), O44.5 (fold table), O44.7 (candidate matrix), O44.8 (compatibility audit) — fold construction is in-memory only | MAJOR |
| 18 | The historical `rolling_origin_manifest.json` is written as CSV (`[["Field","Value"]]`) instead of the JSON manifest schema required by the plan (O44.1) | MINOR |
| 19 | Signoff writes `phase_44_signoff.json` unconditionally with `status: PASS` regardless of any verification logic — no consistency checks | CRITICAL |
| 20 | Phase 45 handoff writes the entire Transformer config including a serialized `Runtime` block (MPS device, torch version 2.13.0) which leaks host environment fingerprint into the lock contract | MAJOR |

The cumulative effect is that the existing Phase 44 evidence is **NOT scientifically
valid** even for the historical pipeline. It must be archived and re-executed.

---

## 3. Scientific contract (extracted from `Phase_44_Rolling-origin_robustness.md`)

### 3.1 Contract matrix

| # | Requirement | Plan section | Expected behavior | Current implementation status |
|---|---|---|---|---|
| C1 | Candidate universe is `Phase42 shortlist ∪ {LSTM_TUNED} ∪ {PERSISTENCE_LAST_VALUE}` | §5 | 3 Transformer + 1 LSTM + 1 Persistence | PARTIAL — shortlist loaded; LSTM_TUNED consumed (stale winner); Persistence inlined, not a registered evaluation |
| C2 | K=3 fold protocol, `RO3_EXPANDING_PRETEST-v1` | §12-§18 | 3 folds over original Validation | PARTIAL — `np.array_split(range(2960), 3)` |
| C3 | ROBASE-v1 = Train ∪ Validation (TRAIN + VALIDATION only), no Test | §14-§15 | `RVAL_IDS = range(13670, 16630)`, then `V1/V2/V3 = array_split(RVAL_IDS, 3)` | INCORRECT — uses `range(2960)` only; outer history is original Train, not Train ∪ prior Vs |
| C4 | `V1/V2/V3` non-empty, disjoint, union = RVAL_IDS, ordered, no Test | §106-§107 | balanced 3-way split | PARTIAL — disjoint, balanced; but outer history construction misses prior V blocks |
| C5 | Inner validation is the immediately preceding block of the same size | §26-§28 | RO1 inner_val = last |V1| Train IDs; RO2 inner_val = V1; RO3 inner_val = V2 | INCORRECT — RO1 uses `Subset(train_dataset, range(13670-V1,V1))` but reuses full train_dataset later in Stage A loader (line 257) |
| C6 | Inner train < inner val < outer origin (temporal) | §27 / §111 | strict ordering | PARTIAL — asserted at indices level but Stage A loader uses full train |
| C7 | Stage A — inner training with candidate max_epochs/patience/min_delta/monitor=inner_validation_rmse_wh | §29-§31 | use candidate's frozen training config | INCORRECT — `stage_a_cfg["training"]["max_epochs"] = 2; patience = 2` |
| C8 | Stage A — fold-inner scalers fit only on inner training history | §37 | no inner-val rows in scaler fit | INCORRECT — global Phase 9 scaler reused; no per-fold scaler |
| C9 | Stage B — fold-outer-history scalers fit only on outer history | §36 | no outer-eval rows in scaler fit | INCORRECT — global Phase 9 scaler reused; no per-fold scaler |
| C10 | Stage B — fresh seed42, fresh model, fresh optimizer, exact best_epoch_inner epochs | §33-§34 | no validation, no warm-start | INCORRECT — traces of warm-start behaviour (parent_run_id absent) and validation loader still passed |
| C11 | Stage B — REFIT_FINAL at best_epoch_inner | §35 | no BEST-on-outer | INCORRECT — engine writes `best_*` even when `evaluate_validation=False` is left at default; `result_a.best_epoch` cannot be set to "exact refit epoch" |
| C12 | No early stopping in Stage B | §33 | early_stopping_enabled=False | PARTIAL — flag set in cfg; engine still runs validation loop unless `evaluate_validation=False` is explicitly passed |
| C13 | Common outer target IDs across all candidate/fold | §46-§48 / §63 | identical `y_true` per fold | NOT ENFORCED — no explicit assertion |
| C14 | Per-fold outer prediction bundle saved | §63 | parquet/csv with target_id, target_timestamp, y_true, y_pred, residual | MISSING — only summary metrics saved |
| C15 | Pooled outer predictions concatenate RO1+RO2+RO3 | §64 / §108 | one bundle per model | MISSING |
| C16 | True pooled RMSE/MAE/R² from pooled residuals | §67 / §108 | compute on concatenated residuals | INCORRECT — computed via `sqrt(mean(fold_RMSE**2))` |
| C17 | Macro fold metrics | §68 | mean ± std across folds | PARTIAL — saved in summary but not as dedicated artifact (O44.25) |
| C18 | Persistence baseline exact semantics | §44 / §103 | `y_pred[t+1] = y_observed[t]` | INCORRECT — implementation shifts `y_true` within the outer block |
| C19 | Persistence prediction bundle with target_id / target_timestamp / y_true / y_pred | §64 | required | MISSING — only per-fold metric row |
| C20 | Transformer ranking by pooled_outer_rmse_wh ascending, exact tie by worst_fold_rmse then fold_rmse_sd then shortlist_position | §76-§78 | primary ranking | PARTIAL — selects by pooled RMSE only |
| C21 | LSTM/Persistence do not enter Transformer tie-break | §78 | ranking isolation | OK |
| C22 | All models use same outer target IDs | §46-§48 | required | NOT ENFORCED |
| C23 | Population fingerprint per fold | §154 / §162 | `rolling_origin_<fold>_population_fingerprint` derived from ROBASE | MISSING — hard-coded |
| C24 | Stage A/B registry IDs `RO_<fold>_<candidate>_INNER/REFIT`, family `ROLLING_ORIGIN_INNER_SELECTION/REFIT` | §113-§115 | required | INCORRECT — uses ROLLING_ORIGIN family and TRAINING execution type |
| C25 | Test firewall — no Test row in any prediction bundle / scaler / history | §65 / §13 | required | PARTIAL — test_locked=True in contract; not asserted per fold |
| C26 | O44.1-O44.39 artifact completeness | §153 | all 39 artifacts | MISSING — only ~9 artifacts produced |
| C27 | Phase 45 handoff emits `phase45_final_model_lock_handoff.json` with locked_model_id, locked_rmse_wh, config, config_fingerprint, recipe_sha256, seed_policy, scaler checksums, etc. | §32 / §165 | required | PARTIAL — emits config but leaks host runtime fingerprint |
| C28 | Signoff gated on actual consistency checks | §168 / §195 | required | INCORRECT — always PASS |

### 3.2 Formalised contract (excerpt)

```
ROBASE-v1 := {target_id ∈ ROBASE : target_split_id ∈ {TRAIN, VALIDATION}}
RVAL_IDS  := sorted ROBASE ∩ VALIDATION
RTRN_IDS  := sorted ROBASE ∩ TRAIN
assert RVAL_IDS ⊆ ROBASE, RTRN_IDS ⊆ ROBASE, RVAL_IDS ∩ RTRN_IDS == ∅
assert |ROBASE ∩ TEST| == 0

V1, V2, V3 := np.array_split(RVAL_IDS, 3)
assert len(V1) == len(V2) == len(V3)
assert V1 ∩ V2 == V2 ∩ V3 == V3 ∩ V1 == ∅
assert V1 ∪ V2 ∪ V3 == RVAL_IDS

fold_k outer_eval := V_k
fold_k inner_val  := V_{k-1}    for k≥2
fold_k inner_val  := last |V1| of RTRN_IDS   for k=1
fold_k outer_train_history := RTRN_IDS ∪ ⋃_{i<k} V_i
fold_k inner_train_history := RTRN_IDS ∪ ⋃_{i<k-1} V_i    (RO2/3)
fold_k inner_train_history := RTRN_IDS \ inner_val       (RO1)

assert all targets strictly increasing within each set
assert max(inner_train) < min(inner_val) < min(outer_eval)
```

---

## 4. Full audit of `scripts/phase44_rolling_origin.py`

Audit method: line-by-line review against `Phase_44_Rolling-origin_robustness.md`
sections 1-198 and the 39 artifact schemas (O44.1-O44.39).

### 4.1 CRITICAL deviations

| ID | Location | Deviation | Plan reference |
|---|---|---|---|
| C-01 | line 257-259 | `inner_train_loader = DataLoader(train_dataset, ...)` — uses full Train dataset for Stage A instead of `Subset(train_dataset, inner_train_train_idx)` | §26, §111 |
| C-02 | line 258 | `inner_val_loader = DataLoader(val_dataset, ...)` — uses full Validation dataset for Stage A inner validation, ignoring fold-specific `inner_val_val_idx` | §26, §111 |
| C-03 | line 260 | `outer_history_loader = DataLoader(train_dataset, ...)` — uses full Train for Stage B of RO1 (correct) but never applies Train ∪ V1 / Train ∪ V1 ∪ V2 for RO2/3; the `ConcatDataset` lines 245-249 are constructed but never fed into Stage B | §36, §138-145 |
| C-04 | line 257-259 | `shuffle=True` for Stage A inner_train is correct; but combined with full Train loader it effectively trains on Train → potentially uses future rows past `inner_val_train_idx` | §27, §111 |
| C-05 | line 263-264 | `stage_a_cfg["training"]["max_epochs"] = 2; patience = 2` — fast-mode override; plan §31 requires candidate-specific max_epochs / patience | §31, §112 |
| C-06 | line 277 | `result_a = engine.train(run_id_a, inner_train_loader, inner_val_loader, model, device, target_scaler, ...)` — passes fold-incorrect loaders and a global target_scaler | §37-§38 |
| C-07 | line 281 | `best_epoch_inner = result_a.best_epoch` — but with max_epochs=2 this is meaningless (best is always epoch 1 or 2) | §31 |
| C-08 | line 306 | `stage_b_cfg["training"]["max_epochs"] = best_epoch_inner` — drives Stage B at the selected epoch (correct intent) | §33 |
| C-09 | line 309 | `stage_b_cfg["training"]["early_stopping_enabled"] = False` — set in config but TrainingEngine signature `evaluate_validation` is left at default True | §33 |
| C-10 | line 321 | `result_b = engine.train(run_id_b, outer_history_loader, inner_val_loader, ...)` — inner_val_loader is still passed as the "validation_loader" argument; engine runs validation each epoch unless `evaluate_validation=False` is explicitly passed | §33-§34 |
| C-11 | line 322 | Stage B scaler is the same global `target_scaler` from Phase 9 | §36 |
| C-12 | line 348-365 | Stage C inference iterates `outer_eval_loader` using the global target_scaler inverse — fold-local scaler never fitted or used | §36, §40 |
| C-13 | line 388-414 | Persistence prediction is implemented by `pers_y_pred = [pers_y_true[0]] + pers_y_true[:-1]` — i.e. shifting the outer y_true vector. The first target uses `y_true[0]` as its own prediction (zero residual). | §44, §103 |
| C-14 | line 432 | `pooled_rmse = np.sqrt(np.mean(np.square(res["rmses"])))` — derived from per-fold RMSE, not from concatenated residuals | §67, §108 |
| C-15 | line 433-434 | `pooled_mae = np.mean(res["maes"])`, `pooled_r2 = np.mean(res["r2s"])` — also mean-of-fold, not pooled | §67 |
| C-16 | line 517-525 | Signoff unconditionally writes `status: PASS` without checking any consistency invariant (no per-fold assertion that all candidates completed, no Test firewall audit, no scaler-fit audit) | §195 |
| C-17 | line 480-486 | `rolling_origin_summary.json` reports `best_transformer_rmse = min(pooled_rmse among TR_*)` but ranking uses fold-level mean-of-RMSE rather than true pooled residual RMSE | §76 |

### 4.2 MAJOR deviations

| ID | Location | Deviation | Plan reference |
|---|---|---|---|
| M-01 | line 113 | `models_evaluated` lists `TR_C1_ALT_WEIGHT_DECAY` but the actual `transformer_candidate_shortlist.json` uses `TR_C1_ALT_WEIGHT_DECAY` AND `TR_C2_ALT_LOOKBACK`; RO3 mentions a 4th model `PERSISTENCE_LAST_VALUE` not in the contract | §5.1, §155 |
| M-02 | line 165 | `cached_loaders` caches candidate-specific loaders by `(variant_id, lookback, target_option, bp_code)` — this is correct but does not include scaler fingerprint, so the same cache key could mix scaler revisions across reruns | §8 (data contract) |
| M-03 | line 167-170 | `bp_code = "WB0"` → `"WB0_CONTEXT_CARRY_OVER"` — Phase 42 candidate config already carries `"WB0_CONTEXT_CARRY_OVER"` so the branch is mostly dead; but if any future candidate carries `"WB0"` string the conversion is hard-coded here | §23, §49 |
| M-04 | line 260, 302 | Population fingerprint hard-coded to `"a40ded8802…"` — fold-local pop fingerprint never computed | §154, §162 |
| M-05 | line 269 | `experiment_family = "ROLLING_ORIGIN" if family == "TRANSFORMER_ENCODER" else "LSTM_TUNING"` — inconsistent family routing; plan requires `ROLLING_ORIGIN_INNER_SELECTION` / `ROLLING_ORIGIN_REFIT` | §113 |
| M-06 | line 269, 309 | `ExecutionType.TRAINING` (not `ExecutionType.ROBUSTNESS`) | §113 |
| M-07 | line 269, 310 | `rerun_reason="REPRODUCIBILITY_CHECK"` — only valid for duplicate fingerprints; here Stage A/B configs may not be duplicates of any prior registration | §55, registry `register_run` semantics |
| M-08 | line 269-271 | No `parent_run_id` linking Stage A → Stage B; no `candidate_id`, no `fold_id`, no `sweep_stage` beyond `f"RO{fold_idx}_A"` | §114-§115 |
| M-09 | line 326-340 | `engine.persist_run_artifacts` is called for both Stage A and Stage B using `result_b.best_*` — for Stage B with `evaluate_validation=True` this would persist a "BEST checkpoint" that is semantically wrong (Stage B has no validation) | §35, §120 |
| M-10 | line 348-365 | No `outer prediction bundle` saved per (model, fold) — O44.21 missing | §63, §21 |
| M-11 | line 388-414 | Persistence bundle not saved as artifact | §64 |
| M-12 | line 463-471 | Findings CSV produced but findings are computed from in-memory fold RMSE — no actual primary ranking audit per tie-break rules | §76 |
| M-13 | line 488-501 | `phase45_final_model_lock_handoff.json` emits the full `best_tr_cfg` (including a `runtime` block with `mps` device, `python_version`, `torch_version`, `platform`) which leaks host environment fingerprint into the lock contract | §32, §165 |
| M-14 |  | O44.4 (frozen fold manifest), O44.5 (fold table), O44.6 (population audit), O44.7 (candidate matrix), O44.8 (compatibility audit), O44.9 (fold-local scaling contract), O44.10 (scaler-fit audit), O44.11 (temporal leakage tests), O44.12 (common-target audit), O44.13 (inner-selection run registry), O44.14 (inner best-epoch table), O44.15 (refit run registry), O44.16 (refit-epoch audit), O44.17-§O44.20 (audit CSVs), O44.21-§O44.22 (prediction bundles), O44.26 (pairwise effects), O44.27 (fold ranks), O44.29 (model-family comparison), O44.32 (Phase45 handoff beyond the current shape), O44.34 (tests CSV), O44.35 (discrepancies JSON) — all missing | §153 |

### 4.3 MINOR deviations

| ID | Location | Deviation | Plan reference |
|---|---|---|---|
| MI-01 | line 96-101 | Preflight only checks `phase42_signoff.status` and `phase43_signoff.status` strings; no fingerprint check, no candidate universe check, no fold manifest validation | §3-§4, §154 |
| MI-02 | line 504 | `rolling_origin_manifest.json` written as CSV (`[["Field","Value"], ...]`) instead of JSON manifest | §154 |
| MI-03 | line 524 | `signoff.completed_at = now_iso()` and `created_at = now_iso()` both at write-time — no `trained_at` / `evaluated_at` separation | §167 |
| MI-04 | line 75 | `bp_code = "WB0"` → `"WB0_CONTEXT_CARRY_OVER"` translation not idempotent against `"WB0_CONTEXT_CARRY_OVER"` (already correct value passes through unchanged) — but if any candidate config contains `"WB0_STRICT"` the script silently falls through to wrong enum | §23 |

### 4.4 NOT_AN_ISSUE

- line 178: `build_train_validation_loaders(...)` is correct for building the
  candidate-specific dataset.
- line 209-211: candidate iteration loop structure is correct.
- line 469: `findings` are advisory only (audit instrument), not required to be
  tied to ranking.
- line 530: figure generation is correct.

---

## 5. Phase 43 → 44 handoff audit

| Field | Plan / corrected Phase 43 | Phase 44 script uses |
|---|---|---|
| Winner run_id | `RUN_LS_LST_0175_BCE5A2CD` | LSTM_WINNER.config is read but no check vs Phase 43 winner fingerprint |
| Config fingerprint | `bce5a2cd6ba86435b7c02a1f1a9d25a6e214493dbd8d6da22886e328287f8593` | Not asserted; `lstm_winner_data["config"]` is consumed directly |
| `hidden_size` | 64 | implicit (model module built from cfg) |
| `num_layers` | 2 | implicit |
| `dropout` | 0.1 | implicit |
| `learning_rate` | 0.0003 | implicit |
| `weight_decay` | 0.0 | implicit |
| `max_epochs` | 50 (Phase 43 candidate contract) | OVERRIDDEN to 2 in Stage A, then to `best_epoch_inner` in Stage B — but candidate-specific |
| `patience` | 10 | OVERRIDDEN to 2 in Stage A |
| `loss` | MSE | implicit |
| `gradient_clip` | 1.0 | implicit |
| `lookback_steps` | 36 (from lookback_id L36_H01_WB0) | implicit (passed into loader) |
| `feature_variant_id` | FS2_TF1 | implicit |
| `target_scaling_option` | YS1 | implicit; global Phase 9 scaler reused |
| `boundary_protocol` | WB0 (Ph 43 winner config) | re-translated to `WB0_CONTEXT_CARRY_OVER` for the loader |
| `batch_size` | 64 | used as `bs = cfg["training"]["batch_size"]` |

**Mismatch summary:**

- **CRITICAL**: The script does **not** assert that
  `lstm_winner_data["config"]` matches the corrected Phase 43 signoff winner
  fingerprint. The historical run therefore silently consumed the pre-corrective
  config; the corrected run would either need explicit verification or
  reconstruction from `phase44_rolling_origin_lstm_handoff.json`.
- **CRITICAL**: `max_epochs` and `patience` for the LSTM candidate are overridden
  in Stage A (`2`/`2`), bypassing the corrected candidate contract (`50`/`10`).
- **MAJOR**: `boundary_protocol` re-translation is fragile (see MI-04).

The corrected Phase 44 driver **must** read
`artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json` and verify that
the LSTM config fingerprint equals the corrected Phase 43 winner fingerprint
`bce5a2cd…` before any registry write.

---

## 6. Phase 42 Transformer shortlist audit

Shortlist: `artifacts/candidate_synthesis/transformer_candidate_shortlist.json`

Frozen? **YES** — `"frozen": true` (line 371) and
`"approved_for_phase44_after_phase43": true` (line 1). Per-candidate
`ready_for_phase44 = true` (lines 31, 124, 240).

| Field | TR_C0_PRIMARY | TR_C1_ALT_WEIGHT_DECAY | TR_C2_ALT_LOOKBACK |
|---|---|---|---|
| candidate_id | `TR_C0_PRIMARY` | `TR_C1_ALT_WEIGHT_DECAY` | `TR_C2_ALT_LOOKBACK` |
| config_fingerprint | `a711a9b8…` | `3b08be3d…` | `585c5e79…` |
| feature_variant | FS2_TF1 | FS2_TF1 | FS2_TF1 |
| target_scaling | YS1 | YS1 | YS1 |
| lookback_steps | 36 | 36 | **72** (alt) |
| batch_size | 32 | 32 | 32 |
| learning_rate | 0.0003 | 0.0003 | 0.0003 |
| weight_decay | 0.001 | **0.0001** (alt) | 0.001 |
| dropout | 0.1 | 0.1 | 0.1 |
| loss | MSE | MSE | MSE |
| max_epochs | 50 | 50 | 50 |
| patience | 10 | 10 | 10 |
| gradient_clip | 1.0 | 1.0 | 1.0 |
| RevIN | false | false | false |
| boundary_protocol | WB0_CONTEXT_CARRY_OVER | WB0_CONTEXT_CARRY_OVER | WB0_CONTEXT_CARRY_OVER |

Notes:
- TR_C2 has lookback 72 (different from TR_C0/C1 = 36). The Phase 44 loader
  builder must therefore key cache by lookback (it already does at line 165).
- TR_C1 has different weight_decay. The fold-local Y scaler is independent of
  weight_decay, but optimizer hyperparams are candidate-specific.
- All three candidates share feature variant FS2_TF1 (33 features), target
  scaling YS1, and boundary protocol WB0_CONTEXT_CARRY_OVER.
- All three have `evidence_class != "PRIMARY_EXISTING"` for TR_C1 and TR_C2
  (synthesized from S9/S4 local alternatives). They require **fresh** Stage A
  and Stage B training in Phase 44 (no historical exact-config run to reuse).

The corrected Phase 44 preflight MUST verify:

1. `transformer_candidate_shortlist.frozen == true`
2. All three candidates have `ready_for_phase44 == true`
3. Each candidate's `config_fingerprint` is reproducible by
   `compute_config_fingerprint(candidate["config"])`
4. None of the candidate configs references `target_split_id == TEST`

---

## 7. Corrected ROBASE-v1 design

### 7.1 Population definition

```
ROBASE := WINDOWPOP-v1
RTRN   := sort(ROBASE ∩ {target_split_id == "TRAIN"})
RVAL   := sort(ROBASE ∩ {target_split_id == "VALIDATION"})
assert RTRN ∪ RVAL == ROBASE (within pre-Test)
assert RTRN ∩ RVAL == ∅
assert |ROBASE ∩ TEST| == 0
```

### 7.2 Fold construction

```
V := np.array_split(RVAL, 3)
V1, V2, V3 := [list(v) for v in V]
assert len(V1) == len(V2) == len(V3)
assert V1 ∩ V2 == V2 ∩ V3 == V3 ∩ V1 == ∅
assert V1 ∪ V2 ∪ V3 == sorted(RVAL)
```

### 7.3 Outer / inner splits per fold

```
for k ∈ {1, 2, 3}:
    fold_k.outer_eval_ids  := V_k
    fold_k.outer_train_ids := RTRN ∪ V_1 ∪ … ∪ V_{k-1}    # expanding
    fold_k.inner_val_ids   := V_{k-1}                       # k≥2
    fold_k.inner_val_ids   := last |V1| of RTRN             # k=1
    fold_k.inner_train_ids := RTRN \ fold_k.inner_val_ids   # k=1
    fold_k.inner_train_ids := RTRN ∪ V_1 ∪ … ∪ V_{k-2}      # k≥2

    # Temporal ordering
    assert max(fold_k.inner_train_ids) < min(fold_k.inner_val_ids)
    assert max(fold_k.inner_val_ids) < min(fold_k.outer_eval_ids)
```

### 7.4 Population fingerprints

```
fold_k.population_fingerprint := sha256(
    join(
        fold_k.inner_train_ids + fold_k.inner_val_ids + fold_k.outer_eval_ids,
        ","
    )
)
```

A single ROBASE fingerprint (constant across folds) is also computed for the
manifest.

---

## 8. Corrected fold definitions (algorithm)

```python
def build_folds(robase_train_ids, robase_val_ids, k=3):
    splits = np.array_split(robase_val_ids, k)
    V = [list(s) for s in splits]
    nV1 = len(V[0])
    train_full = list(robase_train_ids)
    folds = []
    for k_idx in range(k):
        fold = {
            "fold_id": f"RO{k_idx+1}",
            "outer_eval_ids":   V[k_idx],
            "outer_train_ids":  list(train_full) + sum((V[:k_idx]), []),
            "inner_val_ids":    V[k_idx-1] if k_idx >= 1 else train_full[-nV1:],
            "inner_train_ids":  (
                train_full[:-nV1] if k_idx == 0
                else list(train_full) + sum((V[:k_idx-1]), [])
            ),
        }
        # Temporal assertions
        assert max(fold["inner_train_ids"]) < min(fold["inner_val_ids"])
        assert max(fold["inner_val_ids"]) <= min(fold["outer_eval_ids"]) - 1
        # Disjoint
        assert set(fold["inner_train_ids"]).isdisjoint(fold["inner_val_ids"])
        assert set(fold["outer_train_ids"]).isdisjoint(fold["outer_eval_ids"])
        folds.append(fold)
    return folds
```

---

## 9. Fold-specific dataset / loader path

### 9.1 Gap in current architecture

`build_train_validation_loaders` (in `src/course_work/data/datasets.py:777`)
returns TRAIN + VALIDATION loaders for the **full** chronological split. There is
no public helper that builds a `SequenceWindowDataset` restricted to a
user-supplied list of target IDs.

### 9.2 Proposed helper

A new module `src/course_work/rolling_origin/populations.py` provides:

```python
def build_fold_subset_loader(
    *,
    base_dataset: SequenceWindowDataset,
    target_ids: list[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    drop_last: bool = False,
    seed: int = 42,
) -> tuple[DataLoader, list[int]]:
    """
    Build a DataLoader over `base_dataset` restricted to the subset of
    `SequenceWindowDataset` whose `target_id` ∈ `target_ids`.

    Returns (loader, kept_indices).
    """
    base_df = base_dataset.window_records
    id_to_pos = {tid: pos for pos, tid in enumerate(base_df["target_id"].tolist())}
    indices = [id_to_pos[t] for t in target_ids if t in id_to_pos]
    subset = Subset(base_dataset, indices)
    g = torch.Generator(); g.manual_seed(seed)
    return (
        DataLoader(subset, batch_size=batch_size, shuffle=shuffle,
                   num_workers=num_workers, drop_last=drop_last,
                   worker_init_fn=... , generator=g),
        indices,
    )
```

This helper is needed for **all four** roles per fold:

| Role | Shuffle | drop_last | Source dataset |
|---|---|---|---|
| Stage A inner_train | True | False | `TRAIN` base dataset |
| Stage A inner_val   | False | False | `TRAIN` (k=1) or `VALIDATION` (k≥2) base dataset |
| Stage B outer_train | True | False | combined `TRAIN` + `(V_1..V_{k-1})` base datasets |
| Stage C outer_eval  | False | False | `VALIDATION` base dataset, restricted to V_k |

### 9.3 Note on ROBASE population

ROBASE = pre-Test population only. The base TRAIN / VALIDATION datasets already
exclude TEST by Phase 5 split policy. The fold subset helper therefore inherits
the Test firewall.

---

## 10. RO_SCALING-v1 (fold-local scaling)

### 10.1 Stage A scaler

For each candidate × fold:
- Re-load the candidate's **fold-inner** feature matrix X (rows restricted to
  `inner_train_ids`).
- For Y: use the candidate's `target_scaling_option`.
  - `YS0` → identity (no scaling).
  - `YS1` → fit a fresh `StandardScaler` on the inner-train target values.
- Cyclical (sin/cos) and binary time features pass through unchanged.
- RevIN (RN1) statistics are computed per input window from observed historical
  rows only.

### 10.2 Stage B scaler

For each candidate × fold:
- Fit fold-outer-history scaler on
  `inner_train_ids ∪ outer_train_ids` (i.e. everything strictly before
  `min(outer_eval_ids)`).
- Otherwise same rules as Stage A.
- The Stage A scaler is **not** reused for Stage B (plan §36).

### 10.3 Stage C

No scaler fitting in Stage C. Inference uses the Stage B scaler bundle, which
is serialized as `run_dir/scalers/` next to the REFIT_FINAL checkpoint.

### 10.4 What is **forbidden**

- Using Phase 9 global Train-only scaler for any fold or any stage.
- Reusing the Stage A scaler for Stage B.
- Fitting any scaler on outer-eval rows.

### 10.5 Where the code lives

`src/course_work/rolling_origin/scaling.py`:

```python
@dataclass(frozen=True)
class FoldLocalScalerBundle:
    bundle_id: str
    target_scaling_option: str
    x_scaler: object | None       # sklearn StandardScaler, or None for identity
    y_scaler: object | None       # sklearn StandardScaler, or None for YS0
    fit_population_fingerprint: str
    x_feature_count: int
    lookback_steps: int
    feature_variant_id: str
    boundary_protocol: str
    revin: dict | None

def fit_fold_a_scaler(candidate_cfg, fold) -> FoldLocalScalerBundle: ...
def fit_fold_b_scaler(candidate_cfg, fold) -> FoldLocalScalerBundle: ...
def serialize_bundle(bundle, run_dir) -> Path: ...
def load_bundle(path) -> FoldLocalScalerBundle: ...
```

The `Stage C` loader / inference receives a `FoldLocalScalerBundle` and uses it
to invert predictions back to Wh.

---

## 11. Stage A — inner training

For each `(candidate m, fold k)`:

```
run_id := f"RO{k}_{m}_INNER"   (informational; canonical id is registry-allocated)
parent_run_id := None
candidate_id := m
fold_id := k
stage := "A"
seed := 42
config := deep_copy(candidate_m.config)
# Candidate's frozen training config is used at face value:
#   max_epochs, patience, min_delta, lr, wd, dropout, loss,
#   batch, grad_clip, revin — from candidate_m.config["training"]
assert config["training"]["max_epochs"] == 50   # Phase 42 candidate contract
assert config["training"]["patience"] == 10
scaler_bundle := fit_fold_a_scaler(candidate_m.config, fold_k)
inner_train_loader := build_fold_subset_loader(TRAIN_ds, fold_k.inner_train_ids,
                                              batch=cfg.batch_size, shuffle=True)
inner_val_loader   := build_fold_subset_loader(TRAIN_ds (k=1) or VAL_ds (k≥2),
                                              fold_k.inner_val_ids, shuffle=False)

model := build_model_from_run_config(config)
record := registry.register_run(
    config,
    experiment_family="ROLLING_ORIGIN_INNER_SELECTION",
    execution_type=ExecutionType.ROBUSTNESS.value,
    candidate_id=m, sweep_stage=f"RO{k}_A",
    rerun_reason=None, notes="Stage A inner selection"
)
engine.train(
    record["run_id"],
    inner_train_loader, inner_val_loader,
    model, device, scaler_bundle,
    population_fingerprint=fold_k.population_fingerprint,
    evaluate_validation=True,
    final_refit_mode=False,
)
best_epoch_inner := engine.best_epoch_for(run_id)
best_inner_rmse_wh := engine.best_validation_rmse_wh_for(run_id)

assert best_epoch_inner is not None
assert 1 <= best_epoch_inner <= config["training"]["max_epochs"]
```

Outputs per `(m, k)`:
- `artifacts/runs/<run_id>/config.json`
- `artifacts/runs/<run_id>/training_history.csv`
- `artifacts/runs/<run_id>/checkpoints/best_checkpoint.pt`
- `artifacts/runs/<run_id>/metrics/best_validation_metrics.json`
- `artifacts/runs/<run_id>/scalers/stage_a_scaler_bundle.json` (or
  `.joblib`)
- `artifacts/runs/<run_id>/audit/scaler_fit_audit.json`
- Inner best-epoch is logged to `rolling_origin_inner_best_epochs.csv`
  (O44.14).

---

## 12. Stage B — full-history refit

For each `(candidate m, fold k)`:

```
stage_a_run_id := record of Stage A above
parent_run_id := stage_a_run_id
candidate_id := m
fold_id := k
stage := "B"
config_b := deep_copy(candidate_m.config)
config_b["training"]["max_epochs"] := best_epoch_inner      # exact epoch
config_b["training"]["early_stopping_enabled"] := False
config_b["training"]["early_stopping_patience"] := max(best_epoch_inner, 1)
config_b["lineage"]["population_fingerprint"] := fold_k.population_fingerprint
config_b["lineage"]["refit_seed"] := 42
config_b["lineage"]["refit_stage_a_run_id"] := stage_a_run_id
config_b["lineage"]["refit_epoch_count"] := best_epoch_inner

scaler_bundle_b := fit_fold_b_scaler(candidate_m.config, fold_k)
outer_train_loader := build_fold_subset_loader(
    TRAIN_ds ∪ Concat([VAL_ds]), fold_k.outer_train_ids,
    batch=cfg.batch_size, shuffle=True
)
outer_eval_loader := build_fold_subset_loader(
    VAL_ds, fold_k.outer_eval_ids, shuffle=False
)

model_b := build_model_from_run_config(config_b)
# Fresh seed42, fresh optimizer inside engine.train.
record_b := registry.register_run(
    config_b,
    experiment_family="ROLLING_ORIGIN_REFIT",
    execution_type=ExecutionType.ROBUSTNESS.value,
    parent_run_id=stage_a_run_id,
    candidate_id=m, sweep_stage=f"RO{k}_B",
    rerun_reason=None, notes=f"Stage B refit at best_epoch={best_epoch_inner}"
)
engine.train(
    record_b["run_id"],
    outer_train_loader, outer_eval_loader,    # validation_loader unused
    model_b, device, scaler_bundle_b,
    population_fingerprint=fold_k.population_fingerprint,
    evaluate_validation=False,                # exact-epoch, no validation loop
    final_refit_mode=True,
)
# REFIT_FINAL checkpoint is the model after the (best_epoch_inner)th epoch
```

### 12.1 TrainingEngine extension

The current `engine.train` signature already has `evaluate_validation` and
`final_refit_mode` flags. The corrective implementation must:

1. **Always pass `evaluate_validation=False`** in Stage B calls (currently the
   script leaves it at default True).
2. Verify that when `final_refit_mode=True`, the engine:
   - Disables early stopping unconditionally.
   - Trains exactly `max_epochs` full epochs.
   - Does not run validation after any epoch.
   - Saves the final-epoch weights as the official checkpoint (not "best").
   - Records `best_epoch=max_epochs` for compatibility, with a separate
     `official_epoch=max_epochs` field in `status.json`.

If any of these pre-conditions are not satisfied by the current engine, the
plan proposes a minimal extension module
`src/course_work/rolling_origin/refit_engine.py` that wraps `TrainingEngine` and
enforces the contract explicitly. This module belongs under `src/course_work/`
because it is reusable Phase 44+ logic, not driver glue.

---

## 13. Stage C — outer evaluation

```
for each (model_b := record_b.load_model()):
    model_b.eval()
    scaler_bundle := load_bundle(run_dir/scalers/stage_b_scaler_bundle.json)
    y_pred := []
    y_true := []
    target_ids := []
    target_timestamps := []
    with torch.inference_mode():
        for batch in outer_eval_loader:
            x := batch["x"].to(device)
            y_raw_wh := batch["y_raw_wh"]    # already in Wh (raw)
            out := model_b(x)                # scaled space
            pred_wh := scaler_bundle.inverse_y(out)
            y_pred.append(pred_wh)
            y_true.append(y_raw_wh)
            target_ids.append(batch["target_id"])
            target_timestamps.append(batch["target_timestamp"])

    bundle := concat_predictions(fold_id=k, candidate_id=m,
                                  target_id=target_ids,
                                  target_timestamp=target_timestamps,
                                  y_true_wh=y_true, y_pred_wh=y_pred,
                                  model_run_id=record_b["run_id"],
                                  refit_epoch=best_epoch_inner)
    save bundle as artifacts/rolling_origin/predictions/<m>_<k>.parquet
```

`assert all(target_ids == fold_k.outer_eval_ids)` (same order, no dups).

---

## 14. Persistence — fixed semantics

The plan (§44 / §103) requires:

```
y_pred[t+1] = y_observed[t]   for t in ROBASE ∪ outer-block
```

The current implementation shifts `y_true` within the outer block, which
incorrectly assigns zero residual to the first target.

### 14.1 Correct implementation

```python
def persistence_predictions(fold_k, source_y_lookup):
    """
    source_y_lookup : dict[timestamp -> y_observed_Wh]
    Returns (target_ids, target_timestamps, y_true, y_pred) for fold_k.
    """
    rows = []
    for tid, ts in fold_k.outer_eval_pairs:
        y_true = source_y_lookup[ts]
        prior_ts = ts - sampling_interval
        y_pred = source_y_lookup[prior_ts]      # explicit prior lookup
        rows.append((tid, ts, y_true, y_pred))
    return rows
```

`source_y_lookup` is built from the concatenated ROBASE observed Appliances
column (`dataset["Appliances"]`) restricted to strictly pre-`min(outer_eval)`.

For the **first** target of each outer block, `prior_ts` is the last observed
Appliances timestamp strictly before `min(outer_eval)`. This is the correct
WB0 one-step rolling context.

### 14.2 Persistence bundle

Saved as `artifacts/rolling_origin/predictions/PERSISTENCE_<k>.parquet` with the
same schema as Stage C.

---

## 15. True pooled metrics

```
for each model m:
    bundles := []
    for k in {1, 2, 3}:
        df_k := load bundle for (m, k)
        bundles.append(df_k)
    pooled := pd.concat(bundles, ignore_index=True)

    assert pooled["target_id"].is_unique
    assert set(pooled["target_id"]) == union of fold_k.outer_eval_ids
    assert len(pooled) == sum(len(fold_k.outer_eval_ids))

    pooled_mae  = (pooled.y_true - pooled.y_pred).abs().mean()
    pooled_rmse = sqrt(((pooled.y_true - pooled.y_pred) ** 2).mean())
    pooled_r2   = 1 - ((pooled.y_true - pooled.y_pred)**2).sum() / \
                       ((pooled.y_true - pooled.y_true.mean())**2).sum()

    macro_rmse = sqrt(mean(fold_k.rmse_wh ** 2 for k in [1,2,3]))
    macro_mae  = mean(fold_k.mae_wh)
    worst_fold_rmse = max(fold_k.rmse_wh)
    best_fold_rmse  = min(fold_k.rmse_wh)
    fold_rmse_sd    = std(fold_k.rmse_wh)
```

The canonical pooled values are `pooled_mae`, `pooled_rmse`, `pooled_r2`.
The macro values are reported alongside but do **not** participate in ranking
or signoff.

---

## 16. Transformer ranking

```
candidates := [m for m in pool.keys() if m.startswith("TR_")]
sorted := sorted(candidates,
                 key=lambda m: (
                     pooled_results[m].pooled_rmse_wh,
                     pooled_results[m].worst_fold_rmse_wh,    # tie-break 1
                     pooled_results[m].fold_rmse_sd,          # tie-break 2
                     shortlist_position[m],                  # tie-break 3
                 ))
recommended_transformer_id := sorted[0]
```

LSTM / Persistence pooled metrics are reported but excluded from the
Transformer ranking (plan §78).

---

## 17. Artifact gap audit (O44.1 — O44.39)

| O44 | Name | Required? | Exists today? | Scientifically valid? | Reconstructable from existing runs? | Must rerun? | Producer planned |
|---|---|---|---|---|---|---|---|
| O44.1 | `rolling_origin_manifest.json` | yes | yes (wrong format) | no | no (fingerprint stale) | yes | corrected driver |
| O44.2 | `rolling_origin_contract.json` | yes | yes | partial | no | yes | corrected driver |
| O44.3 | `phase44_preflight_audit.csv` | yes | yes | partial | no | yes | corrected driver (preflight helper) |
| O44.4 | `rolling_origin_fold_manifest.json` | yes | no | — | no | yes | corrected driver |
| O44.5 | `rolling_origin_fold_table.csv` | yes | no | — | no | yes | corrected driver |
| O44.6 | `rolling_origin_population_audit.csv` | yes | no | — | no | yes | corrected driver |
| O44.7 | `rolling_origin_candidate_matrix.csv` | yes | no | — | no | yes | corrected driver |
| O44.8 | `rolling_origin_candidate_compatibility_audit.csv` | yes | no | — | no | yes | corrected driver |
| O44.9 | `rolling_origin_fold_local_scaling_contract.json` | yes | no | — | no | yes | corrected driver |
| O44.10 | `rolling_origin_scaler_fit_audit.csv` | yes | no | — | no | yes | corrected driver |
| O44.11 | `rolling_origin_temporal_leakage_tests.csv` | yes | no | — | no | yes | corrected driver |
| O44.12 | `rolling_origin_common_target_audit.csv` | yes | no | — | no | yes | corrected driver |
| O44.13 | `rolling_origin_inner_selection_run_registry.csv` | yes | partial (via ExperimentRegistry) | no | no | yes | corrected driver |
| O44.14 | `rolling_origin_inner_best_epochs.csv` | yes | no | — | no | yes | corrected driver |
| O44.15 | `rolling_origin_refit_run_registry.csv` | yes | partial | no | no | yes | corrected driver |
| O44.16 | `rolling_origin_refit_epoch_audit.csv` | yes | no | — | no | yes | corrected driver |
| O44.17 | `rolling_initialization_audit.csv` | yes | no | — | no | yes | corrected driver |
| O44.18 | `rolling_origin_sample_order_audit.csv` | yes | no | — | no | yes | corrected driver |
| O44.19 | `rolling_origin_gradient_diagnostics.csv` | yes | no | — | no | yes | TrainingEngine emits per-run → aggregator |
| O44.20 | `rolling_origin_runtime_diagnostics.csv` | yes | no | — | no | yes | TrainingEngine emits per-run → aggregator |
| O44.21 | `predictions/<model>_<fold>.parquet` | yes | no | — | no | yes | corrected driver (Stage C writer) |
| O44.22 | `predictions/<model>_pooled.parquet` | yes | no | — | no | yes | corrected driver (Stage C writer) |
| O44.23 | `rolling_origin_fold_metrics.csv` | yes | partial (rolling_origin_results.csv) | partial | no | yes | corrected driver |
| O44.24 | `rolling_origin_pooled_metrics.csv` | yes | no | — | no | yes | corrected driver |
| O44.25 | `rolling_origin_macro_robustness_metrics.csv` | yes | no | — | no | yes | corrected driver |
| O44.26 | `rolling_origin_pairwise_effects.csv` | yes | no | — | no | yes | corrected driver |
| O44.27 | `rolling_origin_fold_ranks.csv` | yes | no | — | no | yes | corrected driver |
| O44.28 | `rolling_origin_transformer_robustness_ranking.csv` | yes | partial | no | no | yes | corrected driver |
| O44.29 | `rolling_origin_model_family_robustness_comparison.csv` | yes | no | — | no | yes | corrected driver |
| O44.30 | `rolling_origin_findings.csv` | yes | yes | no | no | yes | corrected driver |
| O44.31 | `rolling_origin_recommended_transformer.json` | yes | no | — | no | yes | corrected driver |
| O44.32 | `phase45_final_model_lock_handoff.json` | yes | yes | no (host runtime leaked) | no | yes | corrected driver |
| O44.33 | Figures (`RO_44_*.png`) | yes | one figure only | partial | no | yes | corrected driver |
| O44.34 | `rolling_origin_tests.csv` | yes | no | — | no | yes | test plan + driver |
| O44.35 | `rolling_origin_discrepancies.json` | yes | no | — | no | yes | consistency.py |
| O44.36 | `rolling_origin_summary.json` | yes | yes (wrong pooled metric) | no | no | yes | corrected driver |
| O44.37 | `rolling_origin_report.md` | yes | no | — | no | yes | corrected driver |
| O44.38 | `README_ROLLING_ORIGIN_ROBUSTNESS.md` | yes | README_ROLLING_ORIGIN.md (older) | partial | no | yes | corrected driver |
| O44.39 | `phase_44_signoff.json` | yes | yes | no (stale fingerprint, unconditional PASS) | no | yes | corrected driver (gated by consistency checks) |

**Existing artifacts reusable scientifically:** **NONE.**

Every Phase 44 aggregate artifact must be regenerated. The Phase 44 registry
records themselves are archived as `INVALIDATED` (not deleted) and must not be
reused as scientific evidence.

---

## 18. Historical artifact policy

### 18.1 Archive

```
ARTIFACTS_TO_ARCHIVE := [
    "phase_44_signoff.json",
    "rolling_origin_contract.json",
    "rolling_origin_findings.csv",
    "rolling_origin_manifest.json",
    "rolling_origin_results.csv",
    "rolling_origin_summary.json",
    "phase44_preflight_audit.csv",
    "phase45_final_model_lock_handoff.json",
    "README_ROLLING_ORIGIN.md",
    "figures/RO_44_01_model_comparison.png",
    # The processing log is rebuilt by the corrective driver.
    # Historical logs (if any) are archived under PHASE44_PRE_CORRECTIVE_<ts>/
]
ARCHIVE_DIR := artifacts/rolling_origin/_history/PHASE44_PRE_CORRECTIVE_<UTC>/
for each path in ARTIFACTS_TO_ARCHIVE:
    if path.exists():
        sha256 := sha256(path)
        archived := ARCHIVE_DIR / path.name
        archived.write_bytes(path.read_bytes())
        manifest.append({
            "source": str(path.relative_to(PROJECT_ROOT)),
            "archive_path": str(archived.relative_to(PROJECT_ROOT)),
            "sha256": sha256,
            "size": path.stat().st_size,
            "reason": "PHASE44_HISTORICAL_INVALIDATED_BY_PROTOCOL_DEVIATION",
            "archived_at": <now>,
            "status": "HISTORICAL_INVALIDATED_BY_PROTOCOL_DEVIATION",
        })
path.unlink()    # remove from active location only after archive is verified
write_json(ARCHIVE_DIR / "_archive_manifest.json", manifest)
```

### 18.2 Registry policy

The historical `RUN_TR_FS_*` runs registered under `experiment_family=
ROLLING_ORIGIN` (if any) are not deleted. The registry policy is:

- A new `register_run` for a corrected Phase 44 run **must not** reuse any
  pre-corrective `config_fingerprint`. Because the corrected execution carries
  new lineage fields (`population_fingerprint`, `refit_stage_a_run_id`,
  `fold_id`, `candidate_id`, etc.), the corrected config_fingerprint will
  differ from any historical Phase 44 run.
- If a corrected registration would collide (same fingerprint), the script
  raises an explicit `ValueError` with the offending fingerprint and never
  silently overwrites.

Historical `RUN_LS_LST_*` Phase 43 runs are not touched by Phase 44 archival.
They remain valid scientific evidence (per the corrected Phase 43 signoff).

---

## 19. Run budget

```
T := number of Transformer candidates = 3
+1 LSTM_TUNED
K := 3 folds
Stage A jobs : (T + 1) × K = 4 × 3 = 12
Stage B jobs : (T + 1) × K = 4 × 3 = 12
Persistence evaluations : K = 3 (no training)
Total official learned training jobs : 24
Total official registry records added : 24 + 3 = 27
```

Time estimate (no actual training): each Stage A run with max_epochs=50 on a
single MPS device for ~13.7k training rows × 33 features × lookback 36 was
historically ~3-4 minutes per fold for the LSTM; the corrected Stage A at the
real contract (`max_epochs=50`, `patience=10`) is expected to be ~10-15 minutes
per fold for the LSTM and similar for the 3 Transformer candidates (slightly
longer for TR_C2 with lookback 72). Stage B at the selected epoch (typ. 4-9
epochs) is ~30-60 seconds per fold.

Indicative total wall-clock: **~2.5-3.5 hours** on MPS / single GPU / CPU,
labeled as an estimate only.

---

## 20. Registry design (canonical identities)

```
experiment_family := {
    "ROLLING_ORIGIN_INNER_SELECTION",   # Stage A
    "ROLLING_ORIGIN_REFIT",              # Stage B
}
execution_type := ExecutionType.ROBUSTNESS

# Stage A registration
register_run(
    config=stage_a_cfg,
    experiment_family="ROLLING_ORIGIN_INNER_SELECTION",
    execution_type=ExecutionType.ROBUSTNESS.value,
    sweep_stage=f"RO{k}_A",
    parent_run_id=None,
    candidate_id=m,
    rerun_reason=None,
    notes=f"Stage A inner selection / fold RO{k} / {m}",
)

# Stage B registration
register_run(
    config=stage_b_cfg,
    experiment_family="ROLLING_ORIGIN_REFIT",
    execution_type=ExecutionType.ROBUSTNESS.value,
    sweep_stage=f"RO{k}_B",
    parent_run_id=stage_a_run_id,
    candidate_id=m,
    rerun_reason=None,
    notes=f"Stage B refit at best_epoch={best_epoch_inner}",
)
```

No arbitrary `rerun_reason`. Reruns are only allowed for `REPRODUCIBILITY_CHECK`,
`CHECKPOINT_RECOVERY`, `MANUAL_RERUN`, `PHASE46_CORRECTIVE_RERUN` (which is
out-of-scope for Phase 44). For a deterministic re-execution triggered by the
corrective driver (different `population_fingerprint` lineage field), the new
config_fingerprint naturally differs and no rerun reason is needed.

`candidate_id` and `sweep_stage` are persisted in the registry record for
audit. The Phase 44 driver also writes a parallel CSV (O44.13 / O44.15) for
human readability.

---

## 21. Failure / resume design

### 21.1 Reuse criteria

A Stage A run may be reused only if:

- registry status = `COMPLETED`
- `config_fingerprint` matches the planned `stage_a_cfg`
- inner-validation predictions file exists with
  - sample_count == len(fold_k.inner_val_ids)
  - all sample-order-fingerprints identical
- best_epoch within [1, max_epochs]

A Stage B run may be reused only if:

- registry status = `COMPLETED`
- `config_fingerprint` matches the planned `stage_b_cfg`
- `parent_run_id` points to a valid Stage A run
- `refit_epoch_count == best_epoch_inner` of the parent
- REFIT_FINAL checkpoint exists with `sha256` recorded in
  `run_dir/checkpoints/refit_final.pt.sha256`
- Stage B scaler bundle `sha256` matches the planned fold-B scaler fingerprint
- no outer selection leakage audit hit

### 21.2 Resume logic

The driver maintains a `phase44_resume.json` keyed by
`(candidate_id, fold_id, stage)`. On restart, each job is checked against the
reuse criteria; reusable jobs are skipped, non-reusable are re-executed.

If `phase44_resume.json` is missing, the driver rebuilds it from a registry
scan (`SELECT * FROM registry WHERE experiment_family IN (...) AND
sweep_stage LIKE 'RO%'`). Any registry record that fails reuse criteria is
flagged `INVALIDATED` (not deleted) and a new corrected run is registered.

### 21.3 Stage C

Stage C is pure inference from a frozen Stage B checkpoint. It may be
recomputed at any time as long as `torch.use_deterministic_algorithms(True)` +
seed42 are set. The driver re-runs Stage C if the prediction bundle's
population fingerprint does not match the planned fold's.

### 21.4 Never-reuse blacklist

The following registry records are explicitly **never** reused as scientific
evidence:

- Any pre-corrective `experiment_family="ROLLING_ORIGIN"` record
  (carries `max_epochs=2` lineage fingerprint).
- Any record with `rerun_reason="SCORE_BASED_RERUN"` (forbidden by plan §56;
  registry should reject, but a defensive blacklist is applied).
- Any record missing the new lineage fields
  (`refit_stage_a_run_id`, `population_fingerprint`).

---

## 22. Dry-run (preflight) gates

`scripts/phase44_rolling_origin.py --mode dry-run` performs ZERO optimizer
steps and asserts:

1. Phase 42 signoff exists and `status` ∈ {PASS, PASS_WITH_WARNING}
2. Phase 43 signoff exists and `status == PASS`
3. `phase_43_signoff.winner.final_winner.run_id` == `RUN_LS_LST_0175_BCE5A2CD`
4. `compute_config_fingerprint(phase44_rolling_origin_lstm_handoff.config)`
   == `bce5a2cd6ba86435b7c02a1f1a9d25a6e214493dbd8d6da22886e328287f8593`
5. `transformer_candidate_shortlist.frozen == true`
6. All 3 Transformer candidates have `ready_for_phase44 == true`
7. ROBASE = Train ∪ Validation (no Test rows)
8. V1/V2/V3 non-empty, disjoint, union = RVAL, ordered
9. Fold inner/outer sets satisfy temporal ordering (no future leakage)
10. Fold-local scaler construction succeeds (synthetic 1-row fit)
11. Stage A inner loader populations match expected sizes
12. Stage B outer loader populations match expected sizes
13. Stage C outer populations match V_k sizes
14. Persistence prior lookup exists (timestamp continuity)
15. Registry `experiment_family` enum values exist for
    `ROLLING_ORIGIN_INNER_SELECTION` and `ROLLING_ORIGIN_REFIT`
    (otherwise preflight reports missing enum and stops)
16. `Test firewall`: no Test target_id in any fold population
17. (Optional) Drive the full population fingerprint computation and write to
    `dry_run_population_fingerprint.json` for human audit

The dry-run aborts BEFORE:

- any `register_run` call (Stage A or B)
- any model construction
- any optimizer construction
- any `.train()` call
- any checkpoint write

---

## 23. Rehearsal design (disposable code-path harness)

`scripts/phase44_rolling_origin_rehearsal.py`:

- Uses a temp directory for all artifacts
- Uses a temp registry (or the existing one with `RUN_REHEARSAL_ONLY`
  flag) — never writes to `artifacts/runs/<id>/` for an official id.
- Uses REHEARSAL_EPOCHS = 2 (not 50) and REHEARSAL_FOLDS = 3 (same K)
- Constructs a synthetic fold population of 3×30 windows to exercise:
  - fold construction
  - ROBASE extraction
  - fold-local scaler fitting
  - Stage A short training (2 epochs)
  - Stage B exact-epoch refit (1 epoch)
  - Stage C inference
  - persistence prior lookup
  - pooled metric computation
  - ranking
  - manifest / contract / signoff skeleton writers
- Asserts all O44 schemas (manifest, fold table, candidate matrix, scaler
  contract, scaler fit audit, temporal leakage tests, common-target audit,
  inner best-epochs, refit epoch audit, initialization/sample-order audits,
  gradient/runtime diagnostics, fold metrics, pooled metrics, macro metrics,
  pairwise effects, fold ranks, transformer ranking, model-family comparison,
  findings, recommended transformer, Phase 45 handoff, tests, discrepancies,
  summary, README, signoff skeleton).
- NEVER writes any official scientific artifact under
  `COURSE_WORK/artifacts/runs/` or `COURSE_WORK/artifacts/rolling_origin/`.

Rehearsal exit code: 0 = code-path OK; non-zero = code-path broken.

---

## 24. Tests (focused)

A new `tests/integration/test_phase44_corrective.py` covers:

1. `test_fold_partition` — V1, V2, V3 disjoint, union=RVAL, ordered
2. `test_ro1_inner_val_size` — `len(RO1.inner_val) == len(V1)`
3. `test_ro1_inner_temporal_ordering` — `max(inner_train) < min(inner_val)`
4. `test_ro2_outer_train_includes_v1`
5. `test_ro2_inner_val_is_v1`
6. `test_ro3_outer_train_includes_v1_v2`
7. `test_outer_fold_disjoint` — `outer_train ∩ outer_eval == ∅`
8. `test_outer_fold_union_v123` — `union(V1,V2,V3) == RVAL`
9. `test_training_history_expansion_ro2` — `len(outer_train_RO2) ==
   len(TRAIN) + len(V1)`
10. `test_same_target_ids_across_candidates_in_fold`
11. `test_same_y_true_across_candidates_in_fold`
12. `test_stage_a_scaler_no_inner_val_leakage`
13. `test_stage_b_scaler_no_outer_eval_leakage`
14. `test_stage_a_outer_firewall` — no outer target value used during Stage A
    (audit: scaler x/y fit history rows restricted to inner_train_ids)
15. `test_stage_b_exact_epoch` — REFIT_FINAL saved at exactly best_epoch_inner
16. `test_stage_b_fresh_model` — model hash differs from Stage A's hash
17. `test_stage_b_fresh_optimizer` — first-step gradient direction differs
    from Stage A (defensive)
18. `test_no_warm_start` — `parent_run_id` is set but model state is NOT
    loaded from the parent checkpoint
19. `test_wb0_actual_history_semantics` — predictions for fold_outer_eval
    use actual historical `Appliances[t-1]`, not predicted values
20. `test_persistence_first_target_prior_lookup` — first target of RO1 outer
    uses the last observed Appliances BEFORE `min(RVAL)`
21. `test_true_pooled_rmse` — pooled_rmse computed on concatenated
    predictions, not from mean-of-fold-RMSE
22. `test_macro_fold_rmse_distinct` — macro metric reported alongside but
    not used in primary ranking
23. `test_ranking_tie_rules` — equal pooled_rmse → tie-break by
    worst_fold_rmse then fold_rmse_sd then shortlist_position
24. `test_candidate_completeness_3_of_3` — all 3 Transformer candidates
    produce a complete set of stage A/B artifacts for all 3 folds
25. `test_lstm_winner_fingerprint_match` — Phase 44 LSTM config fingerprint
    equals `bce5a2cd…`
26. `test_phase42_shortlist_fingerprint_match` — Phase 44 Transformer
    configs match `transformer_candidate_shortlist` fingerprints
27. `test_test_firewall` — no Test target_id in any fold population
28. `test_artifact_schema_completeness` — every O44 file exists and parses
29. `test_no_outer_selection_leakage` — Phase 45 handoff points to the
    pool-correct winner
30. `test_signoff_gated` — Phase 44 signoff writes PASS only if all of the
    above are green
31. `test_historical_invalid_runs_blacklisted` — historical
    `ROLLING_ORIGIN` records are never reused
32. `test_dry_run_zero_optimizer_steps` — `--mode dry-run` completes without
    touching `engine.train`

---

## 25. File-by-file implementation plan

### 25.1 CREATE (under `COURSE_WORK/src/course_work/rolling_origin/`)

- `__init__.py`
  - Re-exports `build_folds`, `ROBASE`, `fit_fold_a_scaler`, `fit_fold_b_scaler`,
    `train_stage_a`, `train_stage_b`, `evaluate_outer`, `persistence_predictions`,
    `pooled_metrics`, `rank_transformers`, `Phase44Artifacts`,
    `Phase44Consistency`, `run_preflight`, `run_rehearsal`.
- `populations.py` — `ROBASE`, fold construction, population fingerprints
- `folds.py` — Fold dataclass, fold manifest serialization, temporal assertions
- `scaling.py` — `FoldLocalScalerBundle`, Stage A/B scaler fit + serialize/load
- `stages.py` — `train_stage_a`, `train_stage_b`, `evaluate_outer`
  wrappers around `TrainingEngine` that enforce fold-local scalers and
  `evaluate_validation=False` in Stage B
- `persistence.py` — `persistence_predictions(robase_y_lookup, fold_k)`
- `pooling.py` — `pooled_metrics(predictions_per_fold)`, macro metrics
- `ranking.py` — `rank_transformers(pooled_results, shortlist_position)`
- `refit_engine.py` — Thin wrapper around `TrainingEngine` that hardens the
  final_refit_mode contract (optional fallback if TrainingEngine behaviour is
  not yet fully aligned)
- `artifacts.py` — All O44 writers (`O44.1`..`O44.39`)
- `consistency.py` — `Phase44Consistency.run(artifacts)` that gates signoff
- `preflight.py` — `run_preflight()` for `--mode dry-run`
- `rehearsal.py` — `run_rehearsal()` for disposable code-path exercise

### 25.2 CREATE (tests)

- `tests/unit/rolling_origin/test_populations.py`
- `tests/unit/rolling_origin/test_folds.py`
- `tests/unit/rolling_origin/test_scaling.py`
- `tests/unit/rolling_origin/test_pooling.py`
- `tests/unit/rolling_origin/test_ranking.py`
- `tests/unit/rolling_origin/test_persistence.py`
- `tests/integration/test_phase44_corrective.py` (the 32 tests in §24)

### 25.3 MODIFY

- `scripts/phase44_rolling_origin.py` — Total rewrite. Replace the existing
  549-line driver with a thin orchestrator that imports
  `course_work.rolling_origin.*`. Add `--mode {preflight, scientific, dry-run,
  rehearsal, finalize-from-canonical-runs}` argparse flags. Default mode is
  `preflight`. The corrected driver MUST:
  - verify corrected Phase 43 fingerprint
  - construct ROBASE from `WINDOWPOP-v1`
  - archive historical artifacts under
    `_history/PHASE44_PRE_CORRECTIVE_<UTC>/`
  - call `course_work.rolling_origin.preflight.run_preflight()`
  - if `--mode scientific`: invoke the full pipeline
  - if `--mode dry-run`: invoke preflight only and exit 0
  - if `--mode rehearsal`: invoke rehearsal harness
  - never touch Test data
- `docs/plan-doc/plan_before_process/phase_44_corrective_pre_process_plan.md`
  (this document)
- `notebook_course_work/CourseWork.ipynb` cells 118-119 (Phase 44 markdown
  + code) — only after the corrected artifacts exist, flip the cells from
  HISTORICAL to PASS once the rerun completes.

### 25.4 KEEP UNCHANGED

- `src/course_work/data/datasets.py` — uses `SequenceWindowDataset` as is;
  no schema change required for fold subsets (the helper in `populations.py`
  reuses the existing `Subset` mechanism).
- `src/course_work/experiments/registry.py` — no API change. New family
  values must be added to the experiment-family enum and the model-family
  mapping table. (See §25.5.)
- `src/course_work/data/scaling.py` — `load_validated_target_scaler` is the
  Phase 9 helper; Phase 44 uses fold-local scalers and does NOT call it.
- `src/course_work/evaluation/metrics.py` — unchanged.
- All Phase 1-43, 45-46 modules — unchanged.

### 25.5 Registry enum extension

`src/course_work/experiments/registry.py` requires two minimal additions:

1. Add `"ROLLING_ORIGIN_INNER_SELECTION"` and `"ROLLING_ORIGIN_REFIT"` to the
   experiment-family allowlist. (Currently `_family` looks up families via a
   static mapping; both new families map to `model_family = LSTM |
   TRANSFORMER_ENCODER` and a new `family_code = "RO_INNER" /
   "RO_REFIT"`.)
2. Optionally extend `family_codes_for_runtime_validation()` if Phase 44 wants
   runtime-validation hooks similar to sweeps.

---

## 26. Implementation rollout sequence

Once human approves:

1. Implement §25.1 modules and §25.2 tests (dry).
2. Run `pytest tests/unit/rolling_origin tests/integration/test_phase44_corrective.py`
   in red phase, then green phase.
3. Run `python scripts/phase44_rolling_origin.py --mode dry-run`
   (exit code 0; archive happens after preflight passes).
4. Archive historical Phase 44 artifacts per §18.
5. Run `python scripts/phase44_rolling_origin.py --mode rehearsal`
   (exit code 0; confirms code-path readiness on synthetic data).
6. Run `python scripts/phase44_rolling_origin.py --mode scientific`
   (24 learned training jobs + 3 persistence evaluations).
7. Inspect signoff gating: every consistency check must pass.
8. Update notebook cells 118-119 from HISTORICAL → PASS.
9. Handoff to Phase 45 (`phase45_final_model_lock_handoff.json` regenerated).

Steps 6-8 are the only steps that mutate `artifacts/runs/` or
`artifacts/rolling_origin/` permanently. Steps 1-5 touch only
`artifacts/rolling_origin/_history/` and `artifacts/rolling_origin/predictions/`.

---

## 27. Rollback policy

If the corrective execution fails for any scientific reason (numerical
instability on TR_C2 with lookback 72, etc.):

1. The new run records remain in the registry (status `FAILED`) and are NOT
   silently overwritten.
2. The new aggregate artifacts under `artifacts/rolling_origin/` (except
   `_history/PHASE44_PRE_CORRECTIVE_<ts>/`) are archived under
   `_history/PHASE44_CORRECTIVE_ABORTED_<ts>/`.
3. The historical Phase 44 artifacts (which are archived under
   `_history/PHASE44_PRE_CORRECTIVE_<ts>/`) can be **restored verbatim** to
   `artifacts/rolling_origin/` by running
   `python scripts/phase44_rolling_origin.py --mode restore-historical`.
4. The notebook cells 118-119 are reverted to HISTORICAL / BLOCKED until the
   corrective execution is re-attempted.

The `--mode restore-historical` mode is a small addition to the driver. It is
safe: it copies the archived files back, computes their SHA256, and verifies
the post-restore contents match the archive manifest.

---

## 28. Do-NOW checklist for human reviewer

- [ ] Confirm the corrected Phase 43 winner fingerprint
      (`bce5a2cd6ba86435b7c02a1f1a9d25a6e214493dbd8d6da22886e328287f8593`) is
      still the canonical Phase 43 winner after `phase_43_signoff.json`
      finalization on 2026-09-02T11:54:29Z.
- [ ] Confirm `phase44_rolling_origin_lstm_handoff.json` exists and the LSTM
      `config_fingerprint` inside equals the corrected Phase 43 winner.
- [ ] Approve this plan (§0-§27).
- [ ] Confirm rollout sequence (§26) may begin.
- [ ] Decide whether the historical Phase 44 artifacts under
      `_history/PHASE44_PRE_CORRECTIVE_<ts>/` are sufficient for audit, or
      whether the full historical registry records (TR/LSTM runs from
      2026-08-30) also need archiving.

---

## 29. End of plan

Implementation may begin ONLY after human approval.

Scientific training executed by this planning task: **NO**.
Official run IDs created: **0**.
Scientific artifacts modified: **NONE**.
Test access: **NO**.
