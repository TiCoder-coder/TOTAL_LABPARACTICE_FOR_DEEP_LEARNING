# Phase 48 — Prediction Analysis: Pre-Process Implementation Plan

**Document ID:** `phase-48-prediction-analysis-pre-process-plan`
**Status:** DRAFT — awaiting Human approval
**Author:** Senior ML Engineer + Scientific Evaluation Auditor
**Phase ID:** `PHASE_48_PREDICTION_ANALYSIS`
**Phase detail authority:** `docs/plan-doc/plan_detail_for_each_phase/Phase_48_Prediction_analysis.md`
**Upstream:** `phase_47_signoff.json` (`PASS`, `ready_for_phase48=true`)
**Downstream:** Phase 49 (Residual Analysis), Phase 50 (Error Regime), Phase 51 (Worst Error), Phase 52+ (Attention)

---

## 1. Purpose

This plan authorizes the **implementation** of Phase 48 — Prediction Analysis.

Phase 48 is a **descriptive prediction-behavior analysis** over the three frozen
Phase 47 Transformer prediction bundles (seeds 42 / 123 / 2026) and the optional
Persistence baseline bundle. **No new Test inference, no checkpoint reload, no
training, no scaler fitting, no model selection, no prediction modification, no
ensemble performance metric** are allowed.

This document is the bridge between the canonical Phase 48 detail and the
implementation order; it must be approved **before** any source code in
`src/course_work/phase48/` is written.

---

## 2. Source-of-truth summary

| Item | Path | Verified status |
|---|---|---|
| Phase 48 plan | `docs/plan-doc/plan_detail_for_each_phase/Phase_48_Prediction_analysis.md` | READ (160 sections) |
| Architecture rule | `docs/RULE_BASE/architecture_rule.md` | READ; §7.37 still says "Không triển khai Phase 38" — **amendment required** |
| Code execution rule | `docs/RULE_BASE/rule_code.md` | READ (separate file) |
| Phase 47 signoff | `artifacts/final_test/phase_47_signoff.json` | `overall_status=PASS`, `ready_for_phase48=true` |
| Test population | `artifacts/final_test/final_test_population_manifest.json` | N=2961, fingerprint `d7dbc0b3…578da87`, perfect 10-min continuity |
| Phase 48 handoff | `artifacts/final_test/phase48_prediction_analysis_handoff.json` | `status=PASS`, `no_best_seed=true`, `no_ensemble=true` |
| Bundle seed42 | `artifacts/final_test/predictions/final_test_predictions_seed42.csv` | 2961 rows, sha256 `246ee0d7…bf73f2` |
| Bundle seed123 | `artifacts/final_test/predictions/final_test_predictions_seed123.csv` | 2961 rows, sha256 `1bb55c44…66a08b` |
| Bundle seed2026 | `artifacts/final_test/predictions/final_test_predictions_seed2026.csv` | 2961 rows, sha256 `bfb57535…cce79d8` |
| Bundle persistence | `artifacts/final_test/predictions/final_test_predictions_persistence.csv` | 2961 rows, sha256 `7115af1c…d9ee9b`, `seed=""` (classified as `PERSISTENCE`) |
| Population fingerprint match | seed42/123/2026/persistence all reference `d7dbc0b3…578da87` | **PASS** |
| Cross-bundle alignment | target_id, target_timestamp, y_true_wh identical for all 4 bundles | **PASS (0 mismatches)** |
| Notebook | `notebook_course_work/CourseWork_1.ipynb` | 126 cells, last cell = Phase 47 render. Phase 48 NOT yet present. |

---

## 3. Safety rules (hard, no exceptions)

1. **No new Test inference.** Use only frozen Phase 47 prediction bundles.
2. **No checkpoint loading for prediction regeneration.**
3. **No training, no scaler fitting, no optimizer.step / .backward / model.train.**
4. **No modification of Phase 47 prediction bundles.** Phase 48 only READS them.
5. **No best-seed selection.** All three seeds remain symmetric.
6. **No ensemble metric.** Seed-mean prediction is descriptive only.
7. **No seed spread as confidence interval.** Wording: "cross-seed prediction spread".
8. **No lag-driven prediction shift.** Lag diagnostic is descriptive only.
9. **No prediction clipping, rounding, or post-hoc correction.**
10. **No Test-derived threshold tuning.** Peak/regime thresholds deferred to Phase 50.
11. **No residual / regime / worst-error / attention analysis.** Those are Phase 49/50/51/52.

If any defect in this plan would force a violation of these rules, **STOP and
escalate to Human**.

---

## 4. Architecture amendment (REQUIRED before code)

`docs/RULE_BASE/architecture_rule.md` §7.37 and §26 currently forbid Phase 38+.
A Phase 48 amendment (modeled after the Phase 37 amendment in §28) must be
added to §28 before any code in `src/course_work/phase48/` is written.

**Proposed amendment entry** (to be added only after Human approval):

```text
v1.10 (2026-09-04) - Phase 48 Prediction Analysis amendment
  - Extended canonical ownership through Phase 48
  - Added Phase 48 owner package src/course_work/phase48/ with read-only descriptive analytics
  - Froze inputs as frozen Phase 47 prediction bundles; no new Test inference, no retraining, no checkpoint reload for new predictions
  - Required seed-mean to be labelled DESCRIPTIVE only and seed spread labelled CROSS-SEED SPREAD, not confidence interval
  - Defer residual histograms / regime RMSE / worst-error ranking / attention maps to Phase 49/50/51/52
  - Notebook renders only via course_work.reporting.phase48.render_phase48_dashboard
  - Authoritative plan: docs/plan-doc/plan_before_process/phase_48_prediction_analysis_plan.md
```

The amendment is NOT applied at this audit stage. It is a precondition for
Phase 48 implementation and must be Human-approved.

---

## 5. Implementation package layout

A new package `src/course_work/phase48/` will be created (after amendment).
Sibling-pattern after `src/course_work/phase47/` (which owns the test-evaluation
package with 6 modules).

Proposed layout:

```text
src/course_work/phase48/
├── __init__.py                # public API exports
├── inputs.py                  # Phase48Inputs dataclass + loader of frozen Phase47 bundles + checksum verification
├── alignment.py               # build wide + long tables, alignment audit
├── distribution.py            # prediction_distribution_summary, range_compression
├── change_behavior.py         # gap-safe first differences, change-magnitude, direction agreement
├── temporal_diagnostics.py    # lag cross-correlation (-6..+6), prediction ACF, local extrema, peak timing
├── seed_agreement.py          # pairwise metrics, per-target seed spread, top-K=20 disagreement, rolling tracking
├── integrity_audits.py        # negative-value audit, saturation audit, baseline context
├── figures.py                 # deterministic first/middle/last 24h zoom figures, scatter, ECDF, lag, ACF, heatmaps
├── writers.py                 # atomic CSV + JSON writers under artifacts/prediction_analysis/
├── findings.py                # findings code enumeration with descriptive safe wording
├── contract.py                # analysis contract + manifest schemas
├── signoff.py                 # Phase 48 signoff writer
└── README_PHASE_48.md         # design notes mirroring README_FINAL_TEST_EVALUATION.md
```

Reporting layer (extends existing pattern from
`src/course_work/reporting/phase_43_47_dashboard.py`):

```text
src/course_work/reporting/phase_48_dashboard.py  # HTML renderer, called by notebook
```

Tests:

```text
tests/unit/test_phase48_inputs.py           # bundle checksum + alignment
tests/unit/test_phase48_distribution.py     # distribution + compression formulas
tests/unit/test_phase48_change_behavior.py  # gap-safe diffs + direction agreement
tests/unit/test_phase48_temporal.py         # lag/ACF/extrema/peak
tests/unit/test_phase48_seed_agreement.py   # pairwise + spread + top-K
tests/unit/test_phase48_safety.py           # no-inference / no-ensemble / no-clipping guards
tests/contracts/test_phase48_contract.py    # analysis contract schema
tests/integration/test_phase48_end_to_end.py
```

---

## 6. Implementation order (mapped to plan §139)

| Order | Plan step | Description | Module(s) | Output |
|---|---|---|---|---|
| A | 1 | Verify Phase 47 signoff | `inputs.py` | preflight audit row |
| B | 2 | Verify frozen prediction checksums | `inputs.py` | source verification CSV |
| C | 3 | Build aligned wide/long tables | `alignment.py` | `prediction_wide_table.csv`, `prediction_long_table.csv`, alignment audit |
| D | 4 | Freeze prediction-analysis contract | `contract.py` | `prediction_analysis_contract.json` |
| E | 5 | Create deterministic zoom definitions | `figures.py` (helper) | in-memory windows (Z1 first 24h, Z2 middle 24h, Z3 last 24h) |
| F | 6 | Compute distribution summaries | `distribution.py` | `prediction_distribution_summary.csv` |
| G | 7 | Range / compression descriptors | `distribution.py` | `prediction_range_compression.csv` |
| H | 8 | Gap-safe first-difference descriptors | `change_behavior.py` | `prediction_change_summary.csv` |
| I | 9 | Directional-change agreement | `change_behavior.py` | `prediction_direction_agreement.csv` |
| J | 10 | Lag cross-correlation -6..+6 | `temporal_diagnostics.py` | `prediction_lag_diagnostics.csv` |
| K | 11 | Prediction ACF at registered lags | `temporal_diagnostics.py` | `prediction_acf_diagnostics.csv` |
| L | 12 | Detect fixed-definition local extrema | `temporal_diagnostics.py` | (in-memory) |
| M | 13 | Peak/trough capture descriptors | `temporal_diagnostics.py` | `prediction_local_extrema_summary.csv` |
| N | 14 | Peak timing ±1-step descriptor | `temporal_diagnostics.py` | `prediction_peak_timing_summary.csv` |
| O | 15 | Cross-seed pairwise agreement | `seed_agreement.py` | `prediction_seed_pairwise_agreement.csv` |
| P | 16 | Per-target seed spread | `seed_agreement.py` | `prediction_seed_spread.csv` |
| Q | 17 | Top-20 seed-disagreement timestamps | `seed_agreement.py` | `prediction_top_seed_disagreement.csv` |
| R | 18 | Optional rolling-24h tracking | `seed_agreement.py` | `prediction_rolling_tracking.csv` |
| S | 19 | Negative-value / saturation audit | `integrity_audits.py` | `prediction_negative_value_audit.csv`, `prediction_saturation_audit.csv` |
| T | 20 | Baseline context | `integrity_audits.py` | `prediction_baseline_context.csv` |
| U | 21 | Generate figures | `figures.py` | 20 PNGs under `figures/` |
| V | 22 | Write findings | `findings.py` | `prediction_analysis_findings.csv` |
| W | 23 | Phase 49/50/51 handoffs | `__init__.py` (composer) | 3 handoff JSONs |
| X | 24 | Discrepancy log + acceptance tests | `__init__.py` | `prediction_analysis_discrepancies.json`, `prediction_analysis_tests.csv` |
| Y | 25 | Report + README + signoff | `signoff.py` | `prediction_analysis_report.md`, `README_PREDICTION_ANALYSIS.md`, `prediction_analysis_summary.json`, `phase_48_signoff.json`, manifest |

---

## 7. Mapping: requirement → artifact → module → validation → test → acceptance

| Plan requirement | Source artifact | Module | Derived artifact | Validation | Test | Acceptance |
|---|---|---|---|---|---|---|
| §13 wide table | 3 transformer bundles | `alignment.py` | `prediction_wide_table.csv` | N=2961, target_id/timestamp/y_true match across seeds | unit | 10/10 alignment rows OK |
| §13 long table | 3 transformer bundles | `alignment.py` | `prediction_long_table.csv` | 3×2961 = 8883 rows | unit | row_count = 8883 |
| §23 distribution | wide table | `distribution.py` | `prediction_distribution_summary.csv` | 5 series × 12 stats | unit | all stats finite |
| §25-§27 range/compression | wide table | `distribution.py` | `prediction_range_compression.csv` | 3 seed rows + ratios | unit | ratios computed with `ddof=1` |
| §30-§33 change magnitude | wide table | `change_behavior.py` | `prediction_change_summary.csv` | 4 series × 6 stats | unit + gap-safe mask | adjacent_mask excludes non-10-min gaps |
| §34-§36 direction agreement | wide table | `change_behavior.py` | `prediction_direction_agreement.csv` | 3 seed rows + 3-class rate + non-zero rate | unit | sign comparison uses exact 3-class (NEG/ZERO/POS) |
| §38-§42 lag -6..+6 | wide table | `temporal_diagnostics.py` | `prediction_lag_diagnostics.csv` | 3 seeds × 13 lags | unit | convention documented in contract |
| §49-§51 prediction ACF | wide table | `temporal_diagnostics.py` | `prediction_acf_diagnostics.csv` | lags [1,6,12,36,72,144] | unit | lags below segment length marked NOT_AVAILABLE |
| §43-§46 local extrema | y_true | `temporal_diagnostics.py` | `prediction_local_extrema_summary.csv` | edge samples excluded | unit | first/last row excluded |
| §47-§48 peak timing ±1 | extrema | `temporal_diagnostics.py` | `prediction_peak_timing_summary.csv` | window locked ±1 step | unit | window constant |
| §53-§54 pairwise agreement | wide table | `seed_agreement.py` | `prediction_seed_pairwise_agreement.csv` | 3 pairs × 5 metrics | unit | Pearson + Spearman + mean_abs_diff + RMSE_pair + max_abs_diff |
| §55-§56 per-target spread | wide table | `seed_agreement.py` | `prediction_seed_spread.csv` | 2961 rows × 6 cols | unit | sample SD with `ddof=1` |
| §56 top-20 disagreement | spread table | `seed_agreement.py` | `prediction_top_seed_disagreement.csv` | 20 rows by `seed_range_prediction` desc | unit | K=20 hard-coded |
| §59-§63 rolling-24h | wide table | `seed_agreement.py` | `prediction_rolling_tracking.csv` | window=144, contiguous only | unit | windows with breaks NOT emitted |
| §64-§67 negative / saturation | wide table | `integrity_audits.py` | `prediction_negative_value_audit.csv`, `prediction_saturation_audit.csv` | 3 seeds × stats | unit | NO clipping; only descriptive |
| §70-§73 baseline context | persistence bundle | `integrity_audits.py` | `prediction_baseline_context.csv` | 1 row (PERSISTENCE) | unit | LSTM row null |
| §112-§126 figures | all derived | `figures.py` | 20 PNGs under `figures/` | byte-level + visual smoke | integration | deterministic seed not required; no interactive widgets |
| §127-§128 findings | all derived | `findings.py` | `prediction_analysis_findings.csv` | findings-code registry | unit | safe wording enforced |
| §150-§152 handoffs | manifest + checksums | `__init__.py` | 3 JSONs | structure matches plan | unit | required fields present |
| §90 manifest | n/a | `signoff.py` | `prediction_analysis_manifest.json` | required fields | unit | all keys present |
| §155 summary | all derived | `signoff.py` | `prediction_analysis_summary.json` | cross-references artifacts | unit | SHA-256 stored |
| §156 signoff | all | `signoff.py` | `phase_48_signoff.json` | required fields + booleans | unit | all `*=false` for forbidden actions |
| §153 report | all | `signoff.py` | `prediction_analysis_report.md` | 22 sections | unit | human-readable |
| §154 README | all | `signoff.py` | `README_PREDICTION_ANALYSIS.md` | 10 questions answered | unit | human-readable |

---

## 8. Notebook cell (post-implementation)

A single cell to be appended after Phase 47:

```python
from course_work.reporting.phase_48_dashboard import render_phase_48_dashboard
display(render_phase_48_dashboard(PROJECT_ROOT))
```

No other notebook cell changes. No `pd.read_csv`, no `def`, no plotting code
inside the cell (per `architecture_rule.md` §14.2).

---

## 9. Safety assertions (must all be `True` at signoff time)

```text
assert new_inference_applied == False
assert model_training_applied == False
assert best_seed_selected == False
assert ensemble_metric_computed == False
assert prediction_shift_applied == False
assert prediction_clipping_applied == False
assert prediction_saturation_corrected == False
assert source_predictions_modified == False
assert checkpoint_loading_for_new_predictions == False
assert test_quantile_thresholds_used == False  # Phase 50 only
assert lag_window_widened_after_seeing_results == False
assert peak_window_widened_after_seeing_results == False
assert zoom_window_cherry_picked == False  # Z1/Z2/Z3 deterministic
assert residual_histogram_emitted == False   # Phase 49
assert regime_rmse_emitted == False          # Phase 50
assert worst_error_ranking_emitted == False  # Phase 51
assert attention_map_emitted == False        # Phase 52+
```

---

## 10. Status contract

| Status | Meaning |
|---|---|
| `PASS` | all checks above + all required artifacts produced + source integrity preserved |
| `PASS_WITH_WARNING` | optional figures (heatmap) unavailable OR high seed spread OR negative predictions found |
| `FAIL` | any forbidden action above OR source checksum mismatch OR target misalignment |

---

## 11. Discrepancy taxonomy (from plan §137)

Used for `prediction_analysis_discrepancies.json`:

```text
PHASE47_NOT_APPROVED
SOURCE_PREDICTION_FILE_MISSING
SOURCE_PREDICTION_CHECKSUM_MISMATCH
TEST_POPULATION_MISMATCH
TARGET_ID_DUPLICATE
TARGET_ORDER_MISMATCH
YTRUE_MISMATCH_ACROSS_SEEDS
NONFINITE_PREDICTION
SOURCE_FILE_MODIFIED
NEW_TEST_INFERENCE_ATTEMPT
CHECKPOINT_LOADING_ATTEMPT
MODEL_TRAINING_ATTEMPT
BEST_SEED_SELECTION_ATTEMPT
ENSEMBLE_METRIC_ATTEMPT
PREDICTION_SHIFT_CORRECTION_ATTEMPT
POST_HOC_CLIPPING_ATTEMPT
TEST_THRESHOLD_TUNING_ATTEMPT
ZOOM_CHERRY_PICK_ATTEMPT
GAP_UNSAFE_DIFFERENCING
LAG_CONVENTION_UNSPECIFIED
LAG_RANGE_DRIFT
PEAK_DEFINITION_DRIFT
PEAK_TIMING_WINDOW_DRIFT
SEED_SPREAD_MISLABELED_AS_CONFIDENCE_INTERVAL
RESIDUAL_ANALYSIS_SCOPE_CREEP
ERROR_REGIME_SCOPE_CREEP
WORST_ERROR_SCOPE_CREEP
ATTENTION_SCOPE_CREEP
BASELINE_POPULATION_MISMATCH
OTHER
```

---

## 12. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Phase 47 population has internal gaps | very low (verified perfect 10-min) | gap-safe masks in change_behavior and temporal_diagnostics |
| Architecture amendment rejected | low (plan is well-bounded) | fall back: do not implement Phase 48; re-plan |
| Matplotlib backend issues in CI | medium | use `matplotlib` Agg backend; deterministic figure paths |
| Numerical edge case in sample SD (n=3 seeds) | none | DDOF=1 with n=3 is well-defined |
| Peak definition drift | medium | hard-code rule in module + test asserts |
| User asks for residual diagnostics in Phase 48 | medium | explicit `assert residual_histogram_emitted == False` in signoff |
| Timezone parsing of timestamps | low | timestamps are naive `YYYY-MM-DD HH:MM:SS`; parse as datetime and treat as local UTC |

---

## 13. Definition of Done (from plan §145 — verbatim restatement)

```text
The three frozen Phase 47 Transformer prediction bundles are verified without rerunning Test inference.

All three seeds are aligned on the exact same Test target IDs, timestamps and y_true values.

Prediction behavior is analyzed using full Test chronology and deterministic zoom windows.

Actual and predicted distributions are summarized without modifying the predictions.

Prediction range/variance compression is quantified descriptively.

Temporal change magnitude and directional tracking are computed using gap-safe adjacent timestamps.

Any lag diagnostic uses the predeclared -6 to +6 step range and is not used to shift predictions.

Local extrema analysis uses a fixed definition and does not replace the Train-derived regime analysis planned for Phase 50.

Cross-seed agreement and per-target seed spread are quantified without declaring a best seed.

Seed-mean predictions, if used, are labeled descriptive and are not treated as an ensemble.

Negative/saturation behavior is reported without post-hoc clipping.

Prediction-analysis artifacts are derived from immutable Phase 47 bundles and fully checksummed/provenanced.

No residual, regime, worst-error or attention analysis is improperly substituted for later phases.

No model selection, retraining, Test-driven correction or new inference occurs.

Phase 49 handoff is generated.
```

---

## 14. STOP and wait for Human approval

After this audit:

- `RULES_READ = YES`
- `PHASE48_PLAN_READ = YES`
- `PHASE47_VALID = YES` (`overall_status=PASS`, `ready_for_phase48=true`)
- `FROZEN_PREDICTIONS = 3/3` (3 Transformer + 1 Persistence baseline = 4 total)
- `SOURCE_INTEGRITY = PASS` (all checksums match `prediction_checksums.json`, all 4 bundles share target_ids / timestamps / y_true, no duplicates, all finite, perfect 10-min cadence)
- `CRITICAL_DEFECTS = 0`
- `MAJOR_DEFECTS = 1` (architecture amendment required — not yet applied)
- `MINOR_DEFECTS = 3` (persistence classification, lag convention doc, source read-only assertions)
- `TRAINING_REQUIRED = NO`
- `NEW_TEST_INFERENCE_REQUIRED = NO`
- `READY_FOR_PHASE48_B = YES` (subject to architecture amendment and Human approval)
