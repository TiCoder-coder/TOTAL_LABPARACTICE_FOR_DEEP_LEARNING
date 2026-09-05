# Phase 49 — Residual Analysis (Pre-Process Plan)

**Project:** UCI Appliances Energy Prediction
**Task:** Multivariate Time-Series Regression
**Forecasting contract:** Sequence-to-One, One-Step-Ahead
**Residual convention:** `residual = y_true - y_pred`
**Phase:** 49 — RESIDUAL ANALYSIS
**Phase version (target):** `RESIDUAL_ANALYSIS-v1`
**Phase detail:** `docs/plan-doc/plan_detail_for_each_phase/Phase_49_Residual_analysis.md`
**Architecture file:** `docs/RULE_BASE/architecture_rule.md`
**Rule code file:** `docs/RULE_BASE/rule_code.md`
**This plan version:** v1.0
**Status:** DRAFT — AWAITING human approval (per rule_code.md §10–§17)
**Created at:** 2026-09-04

---

## 1. Objective

Phase 49 performs a strictly **derived** (not generative) residual analysis of the three
frozen Phase 47 Transformer prediction bundles (seeds 42 / 123 / 2026) on the held-out
Test (N_TEST = 2961). It answers whether residual structure shows signed bias,
distribution skewness/heavy tails, gap-safe temporal autocorrelation, sign persistence,
rolling variance change, prediction-magnitude association, cross-seed agreement,
and baseline comparison — all without modifying any prediction, retraining, or
introducing a Test-derived target regime threshold.

The **goal is diagnostic classification**, not model retuning and not ensemble promotion.

## 2. Upstream dependencies

| Upstream | Required artifact | Required status | Verified at audit time |
|---|---|---|---|
| Phase 47 Final Test Evaluation | `artifacts/final_test/predictions/final_test_predictions_seed{42,123,2026,persistence}.csv` | byte-identical to `prediction_checksums.json` | **PASS** — all 4 sha match |
| Phase 47 Final Test Evaluation | `artifacts/final_test/phase_47_signoff.json` | `overall_status == PASS` and `ready_for_phase48 == true` | PASS (Phase 47 signoff has overall_status PASS; already unlocked by previous rounds) |
| Phase 47 Final Test Evaluation | `artifacts/final_test/final_test_population_manifest.json` | `target_ids_sha256 == d7dbc0b3…8da87` | PASS — verified |
| Phase 48 Prediction Analysis | `artifacts/prediction_analysis/phase_48_signoff.json` | `overall_status ∈ {PASS, PASS_WITH_WARNING}`, `ready_for_phase49 == true` | **PASS** — `overall_status == PASS`, `ready_for_phase49 == true` |
| Phase 48 Prediction Analysis | `artifacts/prediction_analysis/phase49_residual_analysis_handoff.json` | `residual_convention == "y_true - y_pred"`, `phase48_executed_residual_analysis == false` | **PASS** — both verified |
| Architecture | `docs/RULE_BASE/architecture_rule.md` v1.10 | explicit Phase 49 authorization | **NOT YET** — v1.10 explicitly defers; new amendment v1.11 REQUIRED |

## 3. Downstream dependencies

| Downstream | Required handoff from Phase 49 |
|---|---|
| Phase 50 Error-by-Regime | `phase50_target_regime_handoff.json` (Test-derived thresholds forbidden; Train-only mandated) |
| Phase 51 Worst-Error Analysis | `phase51_worst_error_handoff.json` (Phase 49 must NOT pre-rank worst targets) |
| Phase 52+ Attention Analysis | `phase52_attention_handoff.json` (Phase 49 must NOT extract attention) |

## 4. Files to read (sequential)

1. `docs/RULE_BASE/working_rule.md` (mandatory first read per rule_code §42)
2. `docs/RULE_BASE/architecture_rule.md` (already read; freeze as authorized scope for Phase 49)
3. `docs/RULE_BASE/rule_code.md` (already read)
4. `docs/plan-doc/plan_detail_for_each_phase/Phase_49_Residual_analysis.md`
5. `docs/plan-doc/plan_before_process/phase_49_residual_analysis_plan.md` (this file)
6. `artifacts/final_test/phase_47_signoff.json`
7. `artifacts/final_test/prediction_checksums.json`
8. `artifacts/final_test/final_test_population_manifest.json`
9. `artifacts/final_test/predictions/final_test_predictions_seed{42,123,2026,persistence}.csv`
10. `artifacts/prediction_analysis/phase_48_signoff.json`
11. `artifacts/prediction_analysis/phase49_residual_analysis_handoff.json`
12. `artifacts/prediction_analysis/prediction_wide_table.csv` (optional sanity reference)
13. `artifacts/phase48_dashboard.py`-equivalent modules for Phase 49 module placement
14. `tests/unit/test_phase48_e_finalization.py` (test style template for Phase 49 tests)

## 5. Files expected to be created or modified (after human approval + amendment v1.11)

### 5.1 New canonical paths (authorization-contingent)

```
src/course_work/phase49/
    __init__.py
    contract.py            # frozen contract re-export / mirror of phase49_a_contract.json
    sources.py             # Phase47-bundle loading + checksum verify + recompute residual
    distributions.py       # distribution summary tables, MAD, skew, kurtosis, MBE, sign balance
    tails.py               # ECDF, common-bin histogram, Q-Q against Gaussian reference
    autocorrelation.py     # gap-safe residual ACF + sign-run + rolling 144
    rolling.py             # exact 144-sample rolling mean/std with window_valid_count == 144 gate
    magnitude.py           # residual-vs-prediction and |residual|-vs-prediction associations
    decile.py              # prediction-decile residual diagnostics (y_pred-based only)
    cross_seed.py          # pairwise seed agreement, sign consensus, per-seed-only statistics summary
    baseline.py            # Persistence baseline residual context; LSTM_TUNED_DEV NOT_ELIGIBLE handling
    writers.py             # atomic CSV/JSON writers + contract-aware CSV column order
    findings.py            # Phase49 findings codes (safe wording; descriptive only)
    handoffs.py            # Phase50/51/52+ handoff JSON writers
    signoff.py             # phase_49_signoff.json writer
    materialize_a.py       # phase49-A (this audit) — already done above
    materialize_b.py       # distribute / bias / tails / sign balance
    materialize_c.py       # gap-safe ACF / rolling / sign runs
    materialize_d.py       # magnitude / decile / cross-seed / baseline
    materialize_e.py       # findings / handoffs / summary / signoff / tests
src/course_work/reporting/
    phase_49_dashboard.py  # notebook presentation layer (sibling of phase_48_dashboard.py)
artifacts/residual_analysis/
    prediction_bundle_checksums.json                # raw sha of source bundles (read-only reference)
    residual_long_table.csv                         # 8883 rows (3 seeds × 2961)
    residual_wide_table.csv                         # 2961 rows × 11 cols
    residual_distribution_summary.csv               # 3 rows (one per seed)
    residual_bias_summary.csv                       # 3 rows
    residual_tail_summary.csv                       # 3 rows
    residual_sign_balance.csv                       # 3 rows
    residual_acf_diagnostics.csv                    # registered-lag ACF per seed (3 series × 6 lags = 18+ rows)
    residual_ljung_box_diagnostics.csv              # per-seed or per-segment, lags [6, 36, 144]
    residual_sign_run_summary.csv                   # 3 rows per seed or per-segment
    residual_sign_transition_summary.csv            # 3 rows
    residual_rolling_diagnostics.csv                # 144-sample rolling per-seed (3 × (2961-143)= 3 × 2818 rows)
    residual_magnitude_association.csv              # per seed scatter / band stats
    residual_prediction_decile_diagnostics.csv      # exactly 10 bins per seed (3 × 10 = 30 rows, with caveat)
    residual_seed_pairwise_agreement.csv            # 3 rows
    residual_seed_sign_consensus.csv                # 2961 rows × 3 (one row per t, one col per seed), plus aggregate
    residual_cross_seed_stat_summary.csv            # 3 rows
    residual_baseline_context.csv                   # 2 rows (PERSISTENCE + LSTM_TUNED_DEV)
    figures/                                         # deterministic PNG figures
        RESID_49_01_residual_distribution_ecdf.png
        RESID_49_02_residual_common_histogram.png
        RESID_49_03_residual_qq_seed42.png
        RESID_49_04_residual_qq_seed123.png
        RESID_49_05_residual_qq_seed2026.png
        RESID_49_06_residual_over_time_seed42.png
        RESID_49_07_residual_over_time_seed123.png
        RESID_49_08_residual_over_time_seed2026.png
        RESID_49_09_residual_acf_per_seed.png
        RESID_49_10_residual_vs_prediction_seed42.png
        RESID_49_11_residual_vs_prediction_seed123.png
        RESID_49_12_residual_vs_prediction_seed2026.png
        RESID_49_13_abs_residual_vs_prediction.png
        RESID_49_14_prediction_decile_residual_std.png
        RESID_49_15_residual_sign_run_lengths.png
        RESID_49_16_seed_pairwise_residual_agreement.png
        RESID_49_17_seed_sign_consensus_over_time.png
        RESID_49_18_residual_heatmap_seed42.png
        RESID_49_19_residual_heatmap_seed123.png
        RESID_49_20_residual_heatmap_seed2026.png
        RESID_49_21_rolling_24h_residual_mean.png
        RESID_49_22_rolling_24h_residual_std.png
    phase49_findings.csv                            # categorical findings codes
    phase50_target_regime_handoff.json
    phase51_worst_error_handoff.json
    phase52_attention_handoff.json
    phase49_summary.json
    phase49_report.md
    README_RESIDUAL_ANALYSIS.md
    phase_49_signoff.json
docs/save_log_in_processing/
    phase_49_residual_analysis_log.json             # Phase49-A entry added in this round (audit only)
    phase_49_architecture_amendment_log.json        # PROPOSAL ONLY (not active yet)
tests/unit/
    test_phase49_a_contract.py
    test_phase49_b_distributions.py
    test_phase49_c_autocorrelation.py
    test_phase49_d_agreement.py
    test_phase49_e_finalization.py
```

### 5.2 Files expected to remain unchanged

- All files under `data/`
- All files under `artifacts/final_test/` (frozen Phase 47 prediction bundles and signoff)
- All files under `artifacts/prediction_analysis/` (Phase 48 canonical artifacts)
- All source code under `src/course_work/data/`, `src/course_work/models/`, `src/course_work/training/`
- All `tests/unit/test_phase*.py` from earlier phases
- The notebook cell sequence Phase 0–48

## 6. Architecture constraints (architecture_rule.md v1.10 §7.37.1 style)

Phase 49 must:

1. Be added to architecture_rule.md §7.37 as a NEW sub-entry (§7.37.2) with the same
   forbidden-actions discipline as §7.37.1 (Phase 48):
   - no new Test inference
   - no training
   - no checkpoint reload
   - no scaler fitting
   - no best-seed selection
   - no ensemble metric
   - no prediction shift / clipping / recalibration / correction
   - no Test-derived threshold / regime definition
   - no residual correction model
   - no worst-error ranking
   - no attention analysis
2. Be added to amendment_log §28 as a NEW amendment (v1.11) with explicit human approval.
3. Use the same architecture-tree pattern as Phase 48:
   - `src/course_work/phase49/` (sibling of `src/course_work/phase48/`)
   - `src/course_work/reporting/phase_49_dashboard.py` (sibling of `phase_48_dashboard.py`)
   - `artifacts/residual_analysis/` (sibling of `artifacts/prediction_analysis/`)
   - `docs/save_log_in_processing/phase_49_residual_analysis_log.json`

## 7. Working rule constraints (working_rule.md + rule_code.md)

Hard:

1. Pre-process plan must precede any implementation (rule_code §11)
2. User must explicitly approve this plan before implementation (§15–§17)
3. No code comments in source (`rule_code.md §34`)
4. No icons/emoji/decorative symbols in source (`rule_code.md §35`)
5. Strictly sequential execution — no parallel tasks (§20–§21)
6. Stop if upstream regression detected (§24–§26)
7. No silent repair / fallback / data fix without plan amendment (§103–§108)
8. Architecture conformance must be checked again at GATE 14 (`rule_code.md §96`)
9. Phase requirement compliance must be checked at GATE 15 (`rule_code.md §96`)
10. Each step must produce explicit validation evidence before the next step starts
    (`rule_code.md §29–§33`)

## 8. Phase requirements mapping

| Plan section | Requirement | Implementation evidence (planned, NOT executed yet) | Test / Check |
|---|---|---|---|
| §2, §14, §15 | Residual convention y - ŷ, sign semantic, zero policy | `sources.py` recomputes residual and verifies exact-column match to `residual_wh` in source | unit test on the first 2961 rows per seed |
| §5, §6 | Source lineage from Phase 47 frozen bundles | `sources.py` checksum-verify all 3 transformer bundles against `prediction_checksums.json` | unit test on sha + fixture |
| §13 | Long and wide residual tables with canonical column names | `writers.py` emits `residual_long_table.csv` and `residual_wide_table.csv` matching plan §13 column list | unit test on schema |
| §15 | Residual integrity checks | `sources.py` recomputes and asserts equality with stored columns | unit test |
| §16 | MAE/RMSE reconstruction matches Phase 47 metrics at full tolerance | `distributions.py` reconstructs and asserts agreement | unit test (metric reconstruction tolerance = 0 absolute) |
| §17, §18, §19, §20 | Distribution summary stats with locked MAD, skew, kurtosis definitions | `distributions.py` emits `residual_distribution_summary.csv` | unit test on definitions (sample skewness bias=False; Fisher excess, bias=False, normal reference 0) |
| §21, §22 | Residual normality is NOT a Phase49 PASS/FAIL requirement | contract test asserts signoff does not check any normality p-value | unit test |
| §23, §24, §25 | 50 common bins, common residual range, bins are visualization only | `tails.py` emits `RESID_49_02_residual_common_histogram.png`; contract test asserts bins are NOT reused as Phase50 regimes | unit test (info-level) |
| §27, §29, §30 | Signed bias metrics (MBE, NMBE, MedBias, underprediction fraction, overprediction fraction) | `distributions.py` emits `residual_bias_summary.csv` and `residual_sign_balance.csv` | unit test on direction semantics |
| §35–§40 | Gap-safe residual ACF at registered lags | `autocorrelation.py` emits `residual_acf_diagnostics.csv` with gap-safe computation | unit test (NO partial windows, NO padding) |
| §43, §44, §137 | Optional contiguous-only Ljung-Box at predeclared lags [6, 36, 144]; secondary diagnostic | `autocorrelation.py` emits `residual_ljung_box_diagnostics.csv`; signoff explicitly marks it as SECONDARY_DIAGNOSTIC | unit test on applicable/not-applicable policy; signoff does not check p-value |
| §45–§51 | Gap-safe sign runs and sign transitions | `autocorrelation.py` emits `residual_sign_run_summary.csv` and `residual_sign_transition_summary.csv` | unit test on break conditions |
| §52–§57 | Exact 144-sample rolling mean/std with window_valid_count == 144 gate | `rolling.py` emits `residual_rolling_diagnostics.csv` | unit test (skip partial windows) |
| §59, §60 | Prediction decile diagnostics from y_pred only | `decile.py` emits `residual_prediction_decile_diagnostics.csv`; contract test asserts decile boundaries are NOT used as Phase50 regimes | unit test (info-level) |
| §62–§68 | Cross-seed residual agreement and sign consensus | `cross_seed.py` emits `residual_seed_pairwise_agreement.csv` and `residual_seed_sign_consensus.csv` | unit test |
| §69–§73 | Persistence baseline residual context; LSTM_TUNED_DEV NOT_ELIGIBLE | `baseline.py` emits `residual_baseline_context.csv`; LSTM row must remain `NOT_ELIGIBLE_CONFIG_MISMATCH` | unit test |
| §74–§78 | Phase47 frozen predictions immutable; no correction | contract test asserts no predicted_corrected / residual_corrected / bias_corrected columns | unit test |
| §163, §164 | Handoffs to Phase 50/51/52+ | `handoffs.py` emits 3 handoff JSONs with role-for-next-phase explicit | unit test |
| §74, §75 | Phase 50 must use Train-derived regimes | Phase50 handoff JSON must include `phase50_threshold_policy == TRAIN_DERIVED_ONLY` | unit test |

## 9. Current project state (verified this audit)

- Phase 47 signoff: `overall_status == PASS`, all 4 source bundles byte-identical
  to `prediction_checksums.json`.
- Phase 48 signoff: `overall_status == PASS`, `ready_for_phase49 == true`,
  `source_predictions_modified == false`, `phase47_artifacts_modified == false`.
- Phase 48 handoff to Phase 49 (`phase49_residual_analysis_handoff.json`):
  `status == "PASS"`, `residual_convention == "y_true - y_pred"`,
  `phase48_executed_residual_analysis == false`,
  `phase48_executed_attention_analysis == false`,
  `phase48_executed_worst_error_ranking == false`,
  `ready_for_phase49 == true`.
- Test population fingerprint: `d7dbc0b3cde772dc3f62c181f0a09a4a7a5b0e7c78e567361dbfde089578da87`.
- N_TEST = 2961. Cadence is exactly 10 minutes for the entire Test period (0 gaps).
- All 3 Transformer seed bundles contain `residual_wh`, `absolute_error_wh`, `squared_error_wh`
  (squared-error column is named `_wh` not `_wh2`; see D-COLNAME-1).

## 10. Preconditions before Phase49-B/C/D/E can begin

1. ✅ Phase 47 signoff PASS
2. ✅ Phase 48 signoff PASS with `ready_for_phase49 == true`
3. ✅ Phase 48 handoff JSON present and passes the verification checks above
4. ⏸ Pre-process plan created (this file)
5. ⏸ Architecture amendment v1.11 drafted and approved by human (NOT YET)
6. ⏸ This pre-process plan approved by human (NOT YET)

If conditions 5 or 6 are not satisfied, Phase49-B/C/D/E implementation MUST NOT start.

## 11. Known risks

| Risk | Mitigation |
|---|---|
| ACF / rolling / sign-run calculators accidentally cross temporal gaps | Use the same gap-safe primitive from Phase 48 and assert gap-free in `sources.py`. N_TEST cadence is currently perfectly continuous at 10 min so the gap policy is essentially defensive, but still locked. |
| Ljung-Box p-value misused as PASS/FAIL | Hard rule in contract; signoff explicitly skips p-value check. |
| Phase 50 defines regimes from Phase 49 decile boundaries | Decile emitter writes both the boundary column and a `purpose = "DIAGNOSTIC_ONLY_NOT_REGIME"` annotation. Handoff to Phase 50 carries the same annotation. |
| Notebook re-executes Phase 0–48 cells when Phase 49 cell is added | Phase 49 cell uses a single import-and-display call only; no inline computation. |
| Implementation begun before human approval | rule_code §122 violation; agent must STOP. |

## 12. Implementation steps (sequential, after human approval + amendment v1.11)

1. **Step 1 — Architecture amendment.** Draft and ask human to approve amendment v1.11
   that adds `src/course_work/phase49/`, `src/course_work/reporting/phase_49_dashboard.py`,
   `artifacts/residual_analysis/`, and `docs/save_log_in_processing/phase_49_residual_analysis_log.json`
   to the authorized paths (mirroring v1.10 for Phase 48), and explicitly forbids new Test
   inference, training, scaler fitting, checkpoint reload, model selection, ensemble metric,
   prediction shift/clip/correction, Test-derived threshold tuning, residual correction model,
   worst-error ranking, and attention analysis.

2. **Step 2 — `src/course_work/phase49/__init__.py` + `contract.py`.** Re-export the
   frozen constants from `phase49_a_contract.json`. Loading is fail-loud if the contract
   JSON is missing.

3. **Step 3 — `sources.py`.** Read-only Phase 47 bundle loader with checksum verification,
   recompute `residual = y_true_wh - y_pred_wh`, `abs_error`, `squared_error`, sign
   classification, source sha carryover. Emit no writer until this passes its unit test.

4. **Step 4 — `distributions.py` + `tails.py`.** Distribution summary, MAD, skew,
   kurtosis, MBE, NMBE, MedBias, under/overprediction fractions, residual vs residual
   ECDF, common-bin histogram, Q-Q per seed. All deterministic. All CSV/PNG outputs
   atomic.

5. **Step 5 — `autocorrelation.py` + `rolling.py`.** Gap-safe residual ACF at registered
   lags, sign-run and sign-transition diagnostics, rolling 144-sample mean/std. Optional
   contiguous Ljung-Box at lags [6, 36, 144] with `not_used_for_pass_fail = true`. All
   gap-aware — assert `window_valid_count == 144` before emitting any rolling row.

6. **Step 6 — `magnitude.py` + `decile.py` + `cross_seed.py` + `baseline.py`.**
   Residual-vs-prediction, |residual|-vs-prediction, exactly-10 intended prediction
   deciles (y_pred-based), cross-seed pairwise agreement and sign consensus, Persistence
   baseline residual context (LSTM_TUNED_DEV preserved as NOT_ELIGIBLE_CONFIG_MISMATCH).

7. **Step 7 — `findings.py` + `handoffs.py` + `summary.py` + `signoff.py`.**
   Categorical findings codes (descriptive only, no causal claims), 3 handoff JSONs
   (Phase 50/51/52+), summary JSON, sign-off with mandatory fields including
   `source_predictions_modified: false`, `phase47_artifacts_modified: false`,
   `phase48_artifacts_modified: false`, `prediction_correction_applied: false`,
   `bias_correction_applied: false`, `residual_correction_model_applied: false`,
   `worst_error_ranking_executed: false`, `attention_analysis_executed: false`,
   `ljung_box_used_for_pass_fail: false`, `deciles_used_as_phase50_regimes: false`,
   `seed_pooling_as_3N_iid: false`, `best_seed_selected: false`.

8. **Step 8 — Tests.** Add unit tests under `tests/unit/`:
   - `test_phase49_a_contract.py`: contract fields, residual convention, sign semantics
   - `test_phase49_b_distributions.py`: stats correctness, definitions, sign balance
   - `test_phase49_c_autocorrelation.py`: gap-safe ACF, sign runs, rolling 144,
     Ljung-Box policy (APPLICABLE / NOT_APPLICABLE never PASS/FAIL)
   - `test_phase49_d_agreement.py`: pairwise seed agreement, sign consensus,
     deciles NOT used as Phase50 regimes
   - `test_phase49_e_finalization.py`: signoff structure, handoff structure,
     JSON serialization safety, deterministic output

9. **Step 9 — Phase49 notebook cell.** Append exactly one markdown + one code cell
   after Phase 48 cell: `display(render_phase_49_dashboard(PROJECT_ROOT))`. Execute
   **only** that cell. Persist HTML output.

10. **Step 10 — Architecture + working-rule conformance audit.** Re-read
    architecture_rule.md + rule_code.md + working_rule.md; assert no violation.

11. **Step 11 — Phase49 sign-off.** Write `phase_49_signoff.json`. PASS only if all
    required gates pass.

12. **Step 12 — Phase49 report + README.** Human-readable `phase49_report.md` and
    `README_RESIDUAL_ANALYSIS.md`.

13. **Step 13 — Processing log update.** Append Phase49-A entry (this audit) and
    every subsequent letter (B/C/D/E) to `phase_49_residual_analysis_log.json`.

## 13. Validation strategy

For each step, the agent must run its focused unit test and the architecture /
working-rule audit. Phase49 sign-off requires:

- All artifacts in §5.1 emitted with valid schema and valid JSON/CSV
- All source bundle sha256 byte-identical to `prediction_checksums.json`
- All Materialization deterministic (re-running with same inputs yields identical
  outputs at full precision)
- `phase_47_signoff.json.overall_status == "PASS"` (unmodified)
- `phase_48_signoff.json.overall_status == "PASS"` (unmodified)
- No artifact under `artifacts/final_test/` modified
- No artifact under `artifacts/prediction_analysis/` modified
- Notebook Phase 49 cell exists; Phase 0–48 cells untouched

## 14. Regression strategy

After Phase49 implementation:

- Re-run all Phase48 tests (must remain 368/368 passing)
- Re-run all Phase47 tests (must remain passing)
- Re-run all Phase 0–46 tests for the artifacts Phase49 reads
- Re-read `rule_code.md §129–§131 Definition of Done / Rule Violation` checklist
  before sign-off

## 15. Expected outputs

- All files listed in §5.1
- 1 processing log JSON (this file's audit entry already appended)
- 1 architecture amendment proposal (separate JSON, only for human review)
- 22+ deterministic figures
- 1 sign-off JSON (PASS only when all gates pass)

## 16. Acceptance criteria

Phase49 is PASS only when:

- All required artifacts in §5.1 are present and valid
- All unit tests pass
- No forbidden action (listed in contract `executed_in_phase49_forbidden`) occurred
- `source_predictions_modified` == false in the Phase49 sign-off
- `phase47_artifacts_modified` == false
- `phase48_artifacts_modified` == false
- `prediction_correction_applied` == false
- `bias_correction_applied` == false
- `worst_error_ranking_executed` == false
- `attention_analysis_executed` == false
- `ljung_box_used_for_pass_fail` == false
- `best_seed_selected` == false
- `ensemble_promoted` == false
- All 4 Phase 47 prediction bundles byte-identical to `prediction_checksums.json`
- All 32 Phase 48 derived artifacts byte-identical to their pre-Phase49 sha (snapshot
  taken before Phase49-B starts)

## 17. Rollback / stop conditions

Stop and roll back Phase49 if any of:

- Source bundle sha mismatch detected
- Target ID / timestamp / y_true mismatch across the 3 seed bundles
- Ljung-Box used as PASS/FAIL in any emitted artifact
- Best-seed selection logic emitted anywhere
- Test-derived target regimes emitted in any Phase49 artifact
- Prediction correction or bias correction applied anywhere
- Notebook Phase 0–48 cells modified
- Any comment / icon introduced into source
- Architecture-rule / working-rule violation detected

## 18. User approval gate

This pre-process plan must be approved by the human owner BEFORE any code is written.
Approval must be explicit (per rule_code.md §15–§17). Without human approval, the
agent MUST NOT create `src/course_work/phase49/`, edit
`docs/RULE_BASE/architecture_rule.md`, or modify any Phase 47 / Phase 48 artifact.
