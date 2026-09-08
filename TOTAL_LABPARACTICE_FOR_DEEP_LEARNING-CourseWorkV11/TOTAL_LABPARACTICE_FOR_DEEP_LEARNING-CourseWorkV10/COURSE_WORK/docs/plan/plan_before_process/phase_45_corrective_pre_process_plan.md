# PHASE 45 — CORRECTIVE PRE-PROCESS PLAN

**Status:** Phase 44 = PASS (post-finalize, post-audit)
**Goal:** Re-execute Phase 45 (Final Model Lock) against the NEW canonical Phase 44 evidence with the corrected, lineage-traced recommended Transformer.

---

## 1. Current Phase 44 Canonical Evidence (post-finalize)

Source: `artifacts/rolling_origin/phase_44_signoff.json` (written 2026-09-03 by `--mode finalize`)

| Field | Value |
| --- | --- |
| `overall_status` | PASS |
| `approved_for_phase45` | true |
| `recommended_transformer_candidate_id` | `TR_C2_ALT_LOOKBACK` |
| `recommended_transformer_fingerprint` | `585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24` |
| `test_status` | NOT_ACCESSED |
| Stage A reused | 12/12 |
| Stage B reused | 12/12 |
| Outer prediction bundles | 12/12 (987×2 + 986×1 per fold) |
| Persistence bundles | 3/3 |
| Pooled RMSE | TR_C2 = 59.86, LSTM = 59.24, PERS = 66.43 |

Inner best epochs (from `rolling_origin_inner_best_epochs.csv`):
- RO1 inner best epoch = ?
- RO2 inner best epoch = ?
- RO3 inner best epoch = ?

(MUST be re-read from canonical CSV in implementation.)

---

## 2. Historical Phase 45 Audit

The existing `artifacts/final_model_lock/phase_45_signoff.json` is **STALE**:

- `recommended_candidate_id = None` (uninitialized placeholder)
- `final_refit_epochs = 50` (default fallback, not derived from Phase 44)
- `created_at = 2026-08-30T01:34:54Z` (predates the corrected Phase 44 finalize)

This means the existing Phase 45 artifacts are NOT traceable to the current Phase 44 evidence and must be quarantined and re-emitted.

---

## 3. Exact Phase 45 Objective

From `docs/plan-doc/plan_detail_for_each_phase/Phase_45_Final_model_lock.md`:

1. Receive the recommended Transformer from Phase 44 (`TR_C2_ALT_LOOKBACK`).
2. Verify the full S1–S19 + Phase 42–44 lineage for this candidate.
3. Freeze the exact final Transformer scientific configuration.
4. Freeze the final data/preprocessing contract.
5. Freeze the final training recipe (FINAL_REFIT_EPOCHS, seeds, scaler protocol).
6. Compute `FINAL_REFIT_EPOCHS = median(RO1_inner_best, RO2_inner_best, RO3_inner_best)` for `TR_C2_ALT_LOOKBACK`.
7. Freeze three final seeds `[42, 123, 2026]`.
8. Freeze checkpoint semantics (`FINAL_REFIT`, not `BEST`).
9. Freeze no-Test, no-retuning rules.
10. Emit all `O45.*` artifacts and `phase_45_signoff.json`.
11. Produce `phase46_three_seed_handoff.json` and `phase47_test_evaluation_guard.json`.

Hard constraints (from plan §2):

```text
new_training_runs = 0
new_validation_evaluations = 0
test_access = 0
```

Phase 45 reads, verifies, resolves, freezes, fingerprints, and signs off — nothing more.

---

## 4. Candidate Lock Semantics

Phase 45 MUST lock exactly the Phase 44 recommended candidate:

- Source-of-truth: `phase_44_signoff.json.recommended_transformer_candidate_id`
- No manual override.
- No candidate reranking (plan §8).
- The candidate ID, fingerprint, scientific config, and RO inner epochs must all reference the SAME candidate.

If the Phase 44 signoff is invalid or `approved_for_phase45 = false`, Phase 45 MUST STOP and emit `PHASE44_NOT_APPROVED` discrepancy.

---

## 5. Epoch Selection Semantics

Plan §30–§34: **median of three rolling-origin inner best epochs**.

```text
e1 = TR_C2_ALT_LOOKBACK inner_best_epoch at RO1
e2 = TR_C2_ALT_LOOKBACK inner_best_epoch at RO2
e3 = TR_C2_ALT_LOOKBACK inner_best_epoch at RO3

FINAL_REFIT_EPOCHS = median([e1, e2, e3])
```

Required assertions (plan §34):

```text
1 <= e1, e2, e3 <= candidate.max_epochs_development
1 <= FINAL_REFIT_EPOCHS <= candidate.max_epochs_development
```

No averaging (plan §194.5).
No seed-specific epochs (plan §38).
No Test/Validation dependency (plan §35, §38).

---

## 6. Required Config Lineage

Phase 45 must reconstruct and verify the full lineage:

| Source | Decision | Artifact |
| --- | --- | --- |
| S1 | feature set | `S01_*` artifact |
| S2 | time features | `S02_*` artifact |
| S3 | target scaling (YS0/YS1) | `S03_*` artifact |
| S4 | lookback (L36/L72/L144) | `S04_*` artifact |
| S5 | pooling | `S05_*` artifact |
| S6 | activation | `S06_*` artifact |
| S7 | batch size | `S07_*` artifact |
| S8 | learning rate | `S08_*` artifact |
| S9 | weight decay | `S09_*` artifact |
| S10 | dropout | `S10_*` artifact |
| S11 | d_model | `S11_*` artifact |
| S12 | heads | `S12_*` artifact |
| S13 | layers | `S13_*` artifact |
| S14 | FFN dim | `S14_*` artifact |
| S15 | loss | `S15_*` artifact |
| S16 | epoch cap | `S16_*` artifact |
| S17 | gradient clipping | `S17_*` artifact |
| S18 | RevIN | `S18_*` artifact |
| S19 | boundary sensitivity (WB0 primary) | `S19_*` artifact |
| Phase 42 | transformer shortlist | `phase_42_signoff.json` |
| Phase 43 | LSTM context | `phase_43_signoff.json` |
| Phase 44 | rolling-origin recommendation | `phase_44_signoff.json` |

All lineage items must resolve to the same `TR_C2_ALT_LOOKBACK` candidate. Any unresolved drift is a hard discrepancy (plan §193).

---

## 7. Required Scaler / Data Contract

Phase 45 freezes:

| Field | Value |
| --- | --- |
| `final_data_region` | `FINAL_DEV_REGION-v1` = TRAIN + VALIDATION |
| `excluded_splits` | `[TEST]` |
| `target_population` | `WINDOWPOP-v1`-compatible pre-Test universe, filtered to final candidate lookback / H1 / WB0 / continuity |
| `x_scaler_fit_protocol` | fit on allowed pre-Test rows per SCALING-v1; once; reused for all 3 seeds |
| `y_scaler_fit_protocol` | fit on allowed pre-Test training target history per locked YS*; once; reused for all 3 seeds |
| `test_rows_used_in_scaler_fit` | 0 (hard) |

---

## 8. Registry Policy

Phase 45 emits NO new training runs in the registry.

The only registry interaction is to read the canonical Phase 44 recommended candidate's scientific config (and its 3 inner-best-epoch runs) for lineage verification.

If Phase 45 implementation code attempts to register a new run, it must FAIL with `NEW_TRAINING_ATTEMPT` discrepancy.

---

## 9. Artifact Lifecycle Policy

| Action | Target |
| --- | --- |
| Quarantine existing Phase 45 artifacts | `artifacts/final_model_lock/_history/stale_pre_corrective_2026-08-30/` |
| Re-emit all O45.* artifacts under | `artifacts/final_model_lock/` |
| Re-emit signoff | `artifacts/final_model_lock/phase_45_signoff.json` |
| Re-emit handoff | `artifacts/final_model_lock/phase46_three_seed_handoff.json` |
| Re-emit test guard | `artifacts/final_model_lock/phase47_test_evaluation_guard.json` |
| NO checkpoint created | n/a |
| NO Test label read | n/a |

All Phase 45 output artifacts must be reproducible by SHA256 over canonical JSON (no wall-clock timestamps in fingerprint input; only in human-readable audit metadata).

---

## 10. Test Firewall

Hard:

```text
test_target_values_accessed = false
test_target_timestamps_read_for_decision = false
test_population_in_final_fit = false
test_rows_in_final_scaler_fit = false
```

Phase 47 is the FIRST phase allowed to access Test labels.

---

## 11. Failure / Resume Policy

| Failure | Action |
| --- | --- |
| Phase 44 signoff not PASS/PASS_WITH_WARNING | STOP. Emit `PHASE44_NOT_APPROVED`. Do not emit Phase 45 signoff. |
| Recommended candidate ID differs from Phase 42 source | STOP. Emit `RECOMMENDED_CANDIDATE_MISSING` or `SHORTLIST_FINGERPRINT_MISMATCH`. |
| Lineage item unresolved | STOP. Emit `LINEAGE_BREAK`. |
| Missing one of RO1/RO2/RO3 inner best epoch | STOP. Emit `RO_INNER_EPOCH_MISSING`. |
| Inner best epoch out of range | STOP. Emit `RO_INNER_EPOCH_OUT_OF_RANGE`. |
| Test target observed | STOP. Emit `TEST_LABEL_ACCESSED`. |
| Phase 45 attempts to register a new run | STOP. Emit `NEW_TRAINING_ATTEMPT`. |

Idempotent re-execution is REQUIRED: a second `--mode lock` invocation with the same inputs must produce the same artifacts (modulo timestamps).

---

## 12. Required Phase 45 Outputs

All `O45.*` artifacts from plan §99. Required at minimum:

- `final_model_lock_manifest.json`
- `final_model_lock_contract.json`
- `final_candidate_source_audit.csv`
- `final_lineage_audit.csv`
- `final_model_scientific_config.json`
- `final_feature_contract.json`
- `final_preprocessing_contract.json`
- `final_boundary_contract.json`
- `final_revin_contract.json`
- `final_optimizer_contract.json`
- `final_loss_contract.json`
- `final_epoch_policy.json`
- `final_epoch_source_audit.csv`
- `final_data_region_contract.json`
- `final_scaling_contract.json`
- `final_seed_contract.json`
- `final_training_recipe.json`
- `final_checkpoint_contract.json`
- `final_attention_compatibility_audit.csv`
- `final_environment_contract.json`
- `final_three_seed_run_matrix.csv` (exactly 3 rows)
- `final_model_config_fingerprint.json`
- `final_training_recipe_fingerprint.json`
- `final_lineage_fingerprint.json`
- `final_model_lock_fingerprint.json`
- `rolling_origin_selection_evidence.json`
- `boundary_sensitivity_evidence.json`
- `baseline_context_evidence.json`
- `phase46_three_seed_handoff.json`
- `phase47_test_evaluation_guard.json`
- `final_model_lock_findings.csv`
- `final_model_lock_tests.csv`
- `final_model_lock_discrepancies.json`
- `final_model_lock_summary.json`
- `final_model_lock_report.md`
- `README_FINAL_MODEL_LOCK.md`
- `phase_45_signoff.json`

---

## 13. Consistency Gates (Pre-Signoff Hard Checks)

Before `phase_45_signoff.json.overall_status = PASS`:

1. Phase 44 signoff PASS/PASS_WITH_WARNING + `approved_for_phase45=true`.
2. Recommended candidate resolves to one of `TR_C0_PRIMARY / TR_C1_ALT_WEIGHT_DECAY / TR_C2_ALT_LOOKBACK`.
3. Candidate fingerprint matches Phase 42 shortlist entry.
4. Shortlist fingerprint matches Phase 42 signoff.
5. S1–S19 lineage reconciles to the recommended candidate.
6. RO1/RO2/RO3 inner best epochs all present for the recommended candidate.
7. `FINAL_REFIT_EPOCHS = median(e1,e2,e3)` ∈ [1, candidate.max_epochs_development].
8. `final_data_region.excluded_splits = [TEST]`.
9. Scaler fit rows = 0 test rows.
10. `seed_contract.seeds = [42,123,2026]` (exact).
11. `planned_final_seed_runs = 3` (exact).
12. `checkpoint_type = FINAL_REFIT`, `BEST_semantics = not_applicable`.
13. All four fingerprint artifacts generated and SHA256 reproducible.
14. `phase46_three_seed_handoff.ready_for_phase46 = true`.
15. `phase47_test_evaluation_guard.test_access_first_allowed_phase = 47`.

---

## 14. Phase 46 Downstream Impact

After Phase 45 PASS:

- Phase 46 may execute exactly 3 final refit runs (one per seed).
- Phase 46 must verify `FINAL_MODEL_LOCK_SHA256` before each run (plan §180).
- Phase 46 may NOT change any locked field. Any change = Protocol Amendment = re-lock.
- Phase 46 must use `FINAL_REFIT_MODE-v1` (fixed epochs, no validation, no early stopping, fresh optimizer per seed).

---

## 15. Implementation File List

| File | Role |
| --- | --- |
| `scripts/phase45_final_model_lock.py` | New `--mode lock` orchestrator (reuse existing scaffolding; do not rewrite). |
| `src/course_work/phase45/manifest.py` | Manifest writer O45.1. |
| `src/course_work/phase45/contract.py` | Lock contract writer O45.2. |
| `src/course_work/phase45/candidate_source.py` | Candidate-source audit O45.4. |
| `src/course_work/phase45/lineage.py` | Lineage audit O45.5. |
| `src/course_work/phase45/scientific_config.py` | Final scientific config O45.6. |
| `src/course_work/phase45/preprocessing.py` | Preprocessing/feature/boundary/RevIN/optimizer/loss contracts O45.7–O45.12. |
| `src/course_work/phase45/epoch_policy.py` | Median epoch policy O45.13 + epoch-source audit O45.14. |
| `src/course_work/phase45/data_region.py` | Final data region contract O45.15. |
| `src/course_work/phase45/scaling.py` | Final scaling contract O45.16. |
| `src/course_work/phase45/seeds.py` | Final seed contract O45.17. |
| `src/course_work/phase45/recipe.py` | Training recipe O45.18 + fingerprint. |
| `src/course_work/phase45/checkpoint.py` | Checkpoint contract O45.19. |
| `src/course_work/phase45/attention.py` | Attention compatibility audit O45.20. |
| `src/course_work/phase45/environment.py` | Environment contract O45.21. |
| `src/course_work/phase45/run_matrix.py` | Three-seed run matrix O45.22. |
| `src/course_work/phase45/fingerprints.py` | Config/recipe/lineage/lock fingerprints O45.23–O45.26. |
| `src/course_work/phase45/handoff.py` | Phase 46/47 handoffs O45.30–O45.31. |
| `src/course_work/phase45/signoff.py` | Signoff writer O45.38. |

`src/course_work/phase45/__init__.py` re-exports for clean import surface.

---

## 16. Focused Tests

Under `tests/unit/`:

- `test_phase45_candidate_lock.py` — recommended candidate matches Phase 44 + Phase 42.
- `test_phase45_epoch_policy.py` — median deterministic, integer, in range.
- `test_phase45_scaler_contract.py` — fit on pre-Test only, deterministic, reused.
- `test_phase45_seed_contract.py` — exactly `[42,123,2026]`, run matrix has 3 rows.
- `test_phase45_test_firewall.py` — zero Test rows in scaler fit, zero Test targets in data region.
- `test_phase45_fingerprint_reproducibility.py` — re-running yields identical SHA256.
- `test_phase45_no_training.py` — Phase 45 never instantiates an optimizer or saves a checkpoint.
- `test_phase45_quarantine_stale.py` — historical Phase 45 artifacts moved to `_history/stale_pre_corrective_*`.
- `test_phase45_protocol_amendment.py` — changing any locked field after signoff raises `LOCK_IMMUTABLE`.

Under `scripts/`:

- `phase45_lock_audit.py` — independent recompute of all O45.* artifacts against canonical Phase 44 evidence.
- `_phase45_stale_quarantine.py` — moves pre-corrective artifacts to `_history/`.

---

## 17. Human-Run Command Design

```bash
cd /Users/vientu/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK && \
caffeinate -dim \
env PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
./.venv/bin/python \
scripts/phase45_final_model_lock.py --mode lock --seed 42
```

Flags:
- `--mode lock` (required; idempotent; default = lock)
- `--seed N` (default 42; only used for any non-deterministic ordering checks; does NOT affect fingerprints)
- `--dry-run` (optional; emit findings/discrepancies without writing artifacts)

Acceptance: exit code 0 + `phase_45_signoff.json.overall_status = PASS` + `ready_for_phase46 = true`.

---

## 18. Rollback Policy

If Phase 45 (corrective) fails after partial emission:

1. Move all newly-written Phase 45 artifacts to `artifacts/final_model_lock/_history/rollback_<timestamp>/`.
2. Restore the previous canonical `phase_45_signoff.json` from `_history/stale_pre_corrective_2026-08-30/` (or from `_history/`).
3. Emit `phase45_corrective_implementation_plan.md` failure summary into `_history/`.

If Phase 45 succeeds but a downstream Phase 46 run discovers a defect in Phase 45:

1. STOP Phase 46 immediately.
2. Invalidate affected Phase 45 artifacts (move to `_history/invalidated_<timestamp>/`).
3. Open Protocol Amendment per plan §64.
4. Re-lock Phase 45 with explicit amendment version.
5. Re-run Phase 46 only after new Phase 45 signoff PASS.

---

## 19. Non-Negotiable Rules

```text
NO TRAINING RUNS.
NO OPTIMIZER.STEP().
NO TEST LABEL ACCESS.
NO TEST PREDICTION GENERATION.
NO TEST METRIC COMPUTATION.
NO NEW OFFICIAL RUN IDS.
NO MUTATION OF CANONICAL PHASE 44 ARTIFACTS.
NO MUTATION OF PHASE 43 SCIENTIFIC EVIDENCE.
NO WEAKENING OF GLOBAL ARTIFACT IMMUTABILITY.
NO SILENT REPAIR OF EXISTING STALE PHASE 45 ARTIFACTS WITHOUT QUARANTINE.
```

---

## 20. Definition of Done

Phase 45 corrective rerun is DONE only when:

- All O45.* artifacts are emitted under `artifacts/final_model_lock/`.
- `phase_45_signoff.json.overall_status = PASS`.
- `recommended_candidate_id = TR_C2_ALT_LOOKBACK` (traceable to Phase 44).
- `FINAL_REFIT_EPOCHS = median(e1, e2, e3)` for `TR_C2_ALT_LOOKBACK`.
- `seed_contract.seeds = [42, 123, 2026]`.
- `phase46_three_seed_handoff.ready_for_phase46 = true`.
- `phase47_test_evaluation_guard.test_access_first_allowed_phase = 47`.
- All four fingerprints reproducible.
- Stale pre-corrective artifacts quarantined.
- Test untouched.
- Focused tests pass.
- Human-run command exits 0.

Then: STOP and WAIT for human approval to proceed to **PHASE 46 — Three-Seed Final Runs**.
