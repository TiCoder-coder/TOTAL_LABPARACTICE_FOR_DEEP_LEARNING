# PHASE 51 WORST-ERROR ANALYSIS — PRE-PROCESS PLAN

**Phase ID:** `PHASE_51_WORST_ERROR_ANALYSIS`
**Version:** `WORST_ERROR_ANALYSIS-v1`
**Sub-phase:** 51-A (Governance + Preflight + Source Audit + Architecture Approval)
**Plan Status:** **APPROVED 2026-09-04** (Human approval recorded in this task)
**Architecture Authorization:** v1.13 amendment PROPOSED → HUMAN_APPROVED → APPLIED 2026-09-04

---

## 1. OBJECTIVE

Phase 51 is a **case-level failure analysis** over frozen Test prediction errors from three Final Transformer seeds (42 / 123 / 2026), annotated with frozen Phase 50 regime labels.

Phase 51-A is a read-only governance and source-audit phase. It does NOT rank errors, inspect individual worst cases, or create casebook rows. It verifies frozen upstream artifacts and registers the selection contract for Phase 51-B execution.

---

## 2. SCIENTIFIC SCOPE (Phase 51 overall)

Phase 51 is POST-HOC TEST ERROR ANALYSIS (analysis-only).

**Allowed uses of Test errors:**
- Diagnosis: rank, annotate, document
- Casebook construction (deterministic, contract-frozen)
- Error concentration analysis
- Cross-seed overlap / Jaccard analysis
- Regime enrichment (using frozen Phase 50 labels)
- Baseline (Persistence/LSTM) context at selected timestamps
- Temporal context ±6 steps
- Input-window context inspection (from raw data)
- Phase 52 attention handoff preparation

**Forbidden uses:**
- Retraining / fine-tuning
- Model selection / hyperparameter tuning
- Threshold tuning (Test-derived)
- Seed selection / ensemble construction
- Prediction correction / recalibration / clipping
- Changing Phase 47 metrics
- Changing Phase 50 thresholds/regimes
- Causal interpretation

---

## 3. UPSTREAM DEPENDENCIES

### Phase 47 (FINAL_TEST_EVAL-v1)
| Artifact | Path | SHA256 (first 16) | Status |
|---|---|---|---|
| Phase 47 signoff | `artifacts/final_test/phase_47_signoff.json` | `80614523b6091e21` | PASS |
| Transformer seed 42 predictions | `artifacts/final_test/predictions/final_test_predictions_seed42.csv` | `246ee0d725af972b` | VERIFIED |
| Transformer seed 123 predictions | `artifacts/final_test/predictions/final_test_predictions_seed123.csv` | `1bb55c445ffe132d` | VERIFIED |
| Transformer seed 2026 predictions | `artifacts/final_test/predictions/final_test_predictions_seed2026.csv` | `bfb575357dd6a8e3` | VERIFIED |
| Persistence predictions | `artifacts/final_test/predictions/final_test_predictions_persistence.csv` | `7115af1c479b8957` | VERIFIED |

### Phase 48 (PREDICTION_ANALYSIS-v1)
| Artifact | Path | SHA256 (first 16) |
|---|---|---|
| Phase 48 signoff | `artifacts/prediction_analysis/phase_48_signoff.json` | `e8c102d582a35dd2` |
| Phase 51 context handoff | `artifacts/prediction_analysis/phase51_worst_error_context_handoff.json` | `dd767dd87fcd250b` |
| Seed spread table | `artifacts/prediction_analysis/prediction_seed_spread.csv` | — |

### Phase 49 (RESIDUAL_ANALYSIS-v1)
| Artifact | Path | SHA256 (first 16) |
|---|---|---|
| Phase 49 signoff | `artifacts/residual_analysis/phase_49_signoff.json` | `9d5fc659717dbc15` |
| Residual long table | `artifacts/residual_analysis/residual_long_table.csv` | `8418a99110bfda70` |
| Residual wide table | `artifacts/residual_analysis/residual_wide_table.csv` | `931ff9109aef236d` |
| Cross-seed sign consensus | `artifacts/residual_analysis/residual_cross_seed_sign_consensus.csv` | (TBC) |

### Phase 50 (ERROR_BY_REGIME-v1)
| Artifact | Path | SHA256 (first 16) |
|---|---|---|
| Phase 50 signoff | `artifacts/error_by_regime/phase_50_signoff.json` | `31d9575c441b6f42` |
| Test regime assignment | `artifacts/error_by_regime/test_regime_assignment.csv` | `e90553cfc747a3f1` |
| Test regime assignment fingerprint | `artifacts/error_by_regime/test_regime_assignment_fingerprint.json` | `cc92ce15454d7267` |
| Train-derived thresholds | `artifacts/error_by_regime/regime_thresholds_train_only.json` | `2fe9ad4f873e3b3e` |
| Train reference manifest | `artifacts/error_by_regime/regime_reference_train_manifest.json` | `ed5532c51e02013b` |
| Phase 51 handoff (error_by_regime) | `artifacts/error_by_regime/phase51_handoff.json` | `e5ee74215a40ad32` |
| Phase 50 → Phase 51 handoff (final_test) | `artifacts/final_test/phase51_worst_error_handoff.json` | `4e97e304a52a3a72` |
| Regime metrics long | `artifacts/error_by_regime/regime_metrics_long.csv` | `8b05a3b458684095` |
| Regime metrics persistence | `artifacts/error_by_regime/regime_metrics_persistence.csv` | `b8eebabc9a82ecc6` |
| Regime rank stability | `artifacts/error_by_regime/regime_rank_stability.csv` | `99e32a885c3f45e1` |
| Regime seed spread | `artifacts/error_by_regime/regime_seed_spread.csv` | `2d6e2ddbd5d74984` |
| Regime sign consensus | `artifacts/error_by_regime/regime_sign_consensus.csv` | `72bc59055820b652` |

---

## 4. FROZEN SCIENTIFIC CONTEXT

| Parameter | Value | Verified |
|---|---|---|
| Seeds | 42 / 123 / 2026 | YES |
| N_TEST | 2,961 | YES |
| TEST_POPULATION_FINGERPRINT | `d7dbc0b3cde772dc3f62c181f0a09a4a7a5b0e7c78e567361dbfde089578da87` | YES |
| RESIDUAL_CONVENTION | `residual = y_true - y_pred` | YES |
| POSITIVE_RESIDUAL | UNDERPREDICTION | YES |
| NEGATIVE_RESIDUAL | OVERPREDICTION | YES |
| PHASE50_TEST_ASSIGNMENT_SHA256 | `e90553cfc747a3f15e0e9ec9e6868ae497e7ade797dc14a81999e416b74219ac` | YES |
| PHASE50_TRAIN_THRESHOLD_SHA256 | `2fe9ad4f873e3b3e42013fe3b2d630e377e4e568120769bd2234a76d6974b109` | YES |

---

## 5. PHASE 50 REGIME CONTRACT

Phase 51 MUST reuse Phase 50 regime labels EXACTLY. No recomputation.

| Family Code | Family Name | Labels |
|---|---|---|
| R1_TARGET_LEVEL | Target Level | TL_LOW, TL_MID, TL_HIGH |
| R2_EXTREME_HIGH | Extreme High | EXTREME_HIGH, NON_EXTREME |
| R3_CHANGE_MAGNITUDE | Change Magnitude | CHANGE_NORMAL, CHANGE_RAPID, CHANGE_UNCLASSIFIED |
| R4_CHANGE_DIRECTION | Change Direction | DIR_DOWN, DIR_FLAT, DIR_UP, DIR_UNCLASSIFIED |
| R5_TIME_OF_DAY | Time of Day | TOD_NIGHT, TOD_MORNING, TOD_AFTERNOON, TOD_EVENING |
| R6_DAY_TYPE | Day Type | DAY_WEEKDAY, DAY_WEEKEND |

Thresholds (TRAIN-derived only, frozen):
- Q25_y = 50 Wh
- Q75_y = 100 Wh
- Q90_y = 210 Wh
- Q90_abs_delta = 80 Wh

---

## 6. SELECTION CONTRACT FREEZE (registered, NOT executed in 51-A)

### W1 — PER_SEED_WORST (K_ABS = 20)
- Rank: `absolute_error_wh` DESCENDING
- Scope: separate ranking for seed=42, seed=123, seed=2026
- Output: top 20 per seed
- Tie-break: `target_id ASC`

### W2 — SHARED_WORST (K_SHARED = 20)
- Formula: `SharedHardness_t = (|e_42| + |e_123| + |e_2026|) / 3`
- Rank: SharedHardness DESCENDING
- Output: top 20 across all Test targets
- Tie-break: `target_id ASC`

### W3 — UNDERPREDICTION_WORST (K_UNDER = 10)
- Filter: `residual > 0` (UNDERPREDICTION)
- Rank: `absolute_error_wh` DESCENDING
- Scope: separate per seed
- Output: top 10 per seed

### W4 — OVERPREDICTION_WORST (K_OVER = 10)
- Filter: `residual < 0` (OVERPREDICTION)
- Rank: `absolute_error_wh` DESCENDING
- Scope: separate per seed
- Output: top 10 per seed

### Shared signed ranking (K_SHARED_SIGNED = 10)
- W3_sh: shared all-under top 10 (residual > 0 for ALL 3 seeds)
- W4_sh: shared all-over top 10 (residual < 0 for ALL 3 seeds)

### Primary ranking metric: `absolute_error_wh DESC`
### Tie-break: `target_id ASC` (deterministic, no random)
### NO manual cherry-picking before ranking rules are frozen.

---

## 7. SUB-PHASE SEQUENCE

| Sub-phase | Description | Status |
|---|---|---|
| **51-A** | Governance + preflight + source audit | **COMPLETE** |
| **51-B** | Selection-contract freeze | PENDING Human approval |
| **51-C** | Worst-case ranking: W1/W2/W3/W4 + shared signed | PENDING |
| **51-D** | Cross-seed overlap + error concentration + hardness/seed-spread | PENDING |
| **51-E** | Phase 50 regime context + Persistence/LSTM context | PENDING |
| **51-F** | Temporal ±6 context + input-window context + casebook | PENDING |
| **51-G** | Figures + findings + discrepancies + report + README + Phase52 handoff + signoff | PENDING |
| **51-H** | Notebook presentation-only dashboard synchronization | PENDING |

---

## 8. PHASE 51 OUTPUT INVENTORY (O51.1–O51.38)

| ID | Logical Name | Expected Path | Source Dependency | Sub-phase | Scientific Purpose |
|---|---|---|---|---|---|
| O51.1 | Analysis manifest | `worst_error_analysis_manifest.json` | — | 51-B | Phase metadata + K values + safety flags |
| O51.2 | Analysis contract | `worst_error_analysis_contract.json` | — | 51-B | Formal W1–W4 + signed ranking specification |
| O51.3 | Preflight audit | `phase51_preflight_audit.json` | — | 51-A | Architecture + source audit results |
| O51.4 | Source verification | `phase51_source_verification.json` | — | 51-A | SHA256 verification of all upstream artifacts |
| O51.5 | Frozen selection contract | `worst_error_selection_contract.json` | — | 51-B | K values + tie-break rules frozen before ranking |
| O51.6 | Selection contract fingerprint | `selection_contract_fingerprint.json` | O51.5 | 51-B | SHA256 of O51.5 for provenance |
| O51.7 | Alignment audit | `phase51_alignment_audit.csv` | O51.4 | 51-B | Verify target_id alignment across Phase 47/49/50 |
| O51.8 | Per-seed top20 absolute-error table | `worst_per_seed_top20.csv` | Phase 47+49 | 51-C | W1: Top 20 per seed |
| O51.9 | All-Test shared hardness table | `shared_hardness_all_test.csv` | Phase 47+49 | 51-C | W2: SharedHardness for all 2,961 Test targets |
| O51.10 | Shared top20 table | `worst_shared_top20.csv` | Phase 47+49 | 51-C | W2: Top 20 by SharedHardness |
| O51.11 | Worst underprediction table | `worst_underprediction_top10.csv` | Phase 47+49 | 51-C | W3: Top 10 per seed, residual > 0 |
| O51.12 | Worst overprediction table | `worst_overprediction_top10.csv` | Phase 47+49 | 51-C | W4: Top 10 per seed, residual < 0 |
| O51.13 | Shared all-under top10 | `shared_all_under_top10.csv` | Phase 47+49 | 51-C | W3_sh: residual > 0 for ALL 3 seeds |
| O51.14 | Shared all-over top10 | `shared_all_over_top10.csv` | Phase 47+49 | 51-C | W4_sh: residual < 0 for ALL 3 seeds |
| O51.15 | Seed-overlap table | `seed_overlap_table.csv` | O51.8 | 51-D | Venn overlap of W1 across seeds |
| O51.16 | Membership matrix | `worst_case_membership_matrix.csv` | O51.8+O51.10 | 51-D | Binary: is target in each W list |
| O51.17 | Error-concentration table | `error_concentration_table.csv` | O51.8+O51.10 | 51-D | Top-K share of total SAE/SSE |
| O51.18 | Regime composition | `regime_composition.csv` | O51.8+O51.10+O51.11+O51.12 | 51-E | Phase 50 regime labels for selected cases |
| O51.19 | Regime overrepresentation | `regime_overrepresentation.csv` | O51.18 | 51-E | Regime share in worst cases vs Test population |
| O51.20 | Hardness vs seed-disagreement table | `hardness_vs_seed_disagreement.csv` | O51.9+O51.15 | 51-D | Cross-seed disagreement vs mean absolute error |
| O51.21 | Baseline context | `baseline_context.csv` | O51.8+O51.10+Phase 47 | 51-E | Persistence MAE/RMSE at worst-case timestamps |
| O51.22 | Casebook index | `casebook_index.csv` | O51.8+O51.10+O51.18 | 51-F | Master index of all selected cases |
| O51.23 | Casebook Markdown | `casebook.md` | O51.22+O51.24 | 51-F | Human-readable worst-case narratives |
| O51.24 | Local temporal context | `local_temporal_context.csv` | O51.22+raw data | 51-F | ±6 step context around each selected case |
| O51.25 | Input-window manifest | `input_window_manifest.csv` | O51.22+Phase 11 | 51-F | Manifest of input windows for selected cases |
| O51.26 | Model-visible feature summary | `model_visible_feature_summary.csv` | O51.25+Phase 7+8 | 51-F | Feature summary at worst-case input windows |
| O51.27 | Diagnostic target-history summary | `target_history_summary.csv` | O51.24+raw data | 51-F | Historical target values for context |
| O51.28 | Context-integrity audit | `context_integrity_audit.json` | O51.24 | 51-F | Verify temporal context completeness |
| O51.29 | Figures | `figures/` | O51.8–O51.20 | 51-G | 6–8 diagnostic figures |
| O51.30 | Attention handoff case table | `phase52_attention_case_table.csv` | O51.22+O51.24 | 51-G | Case IDs + context for Phase 52 |
| O51.31 | Findings | `phase51_findings.json` | O51.8–O51.30 | 51-G | Structured findings per ranking family |
| O51.32 | Phase52 handoff | `phase52_attention_extraction_handoff.json` | O51.30 | 51-G | Phase 52 canonical handoff |
| O51.33 | Tests | `tests/` | — | 51-G | Unit tests for ranking determinism |
| O51.34 | Discrepancies | `phase51_discrepancies.json` | — | 51-G | Any deviations from contract |
| O51.35 | Summary JSON | `phase51_summary.json` | O51.1+O51.31 | 51-G | Machine-readable phase summary |
| O51.36 | Human-readable report | `phase51_report.md` | O51.1+O51.31 | 51-G | Full phase narrative |
| O51.37 | README | `README.md` | — | 51-G | Phase overview + provenance |
| O51.38 | Sign-off | `phase51_signoff.json` | O51.1–O51.37 | 51-G | Final phase signoff |

---

## 9. ARCHITECTURE CONSTRAINTS

- Phase 51 is analysis-only: NO training, NO new inference, NO checkpoint loading
- Selection contract must be FROZEN before any ranking execution
- Phase 50 regime labels must be reused EXACTLY (no recomputation)
- Phase 47/48/49/50 canonical artifacts must NOT be modified
- No best-seed selection, no ensemble, no prediction correction
- Worst-case inspection must NOT influence any upstream Phase
- Attention analysis is deferred to Phase 52
- Notebook is presentation-only (no compute)

---

## 10. TEST / SAFETY AUDIT

Phase 51-A verified:
- All Phase 47 prediction bundles: exist + SHA256 match handoff
- Phase 49 residual long table: 8,883 rows (3 seeds × 2,961 Test targets)
- Phase 50 test regime assignment: 2,961 rows (1:1 with Test population)
- Phase 50 signoff: ready_for_phase51 = True
- Phase 50 did NOT modify Phase 47/48/49 artifacts
- Architecture v1.12 explicitly defers worst-error ranking to Phase 51

---

## 11. ROLLBACK STRATEGY

If Phase 51-B+ encounters contract violations:
1. Halt execution immediately
2. Report discrepancy in `phase51_discrepancies.json`
3. Do NOT modify any upstream canonical artifacts
4. Do NOT modify any Phase 50 regime labels
5. Await Human approval before proceeding

---

## 12. HUMAN APPROVAL GATES

| Gate | Trigger | Action |
|---|---|---|
| G1 | Phase 51-A complete | Human reviews checkpoint → approves/rejects Phase 51-B |
| G2 | Phase 51-B selection contract frozen | Human confirms K values + tie-break rules |
| G3 | Phase 51-C ranking complete | Human reviews W1/W2/W3/W4 results before casebook |
| G4 | Phase 51-G before signoff | Human reviews findings + Phase52 handoff |
| G5 | Phase 51-G signoff | Human approves final Phase 51 completion |

---

## 13. IMPLEMENTATION AUTHORIZATION

**Architecture amendment v1.13 PROPOSED and HUMAN_APPROVED 2026-09-04.**

Amendment proposal is in:
`docs/save_log_in_processing/phase_51_architecture_amendment_log.json`

v1.13 amendment authorizes:
- `artifacts/worst_error_analysis/` — Phase 51 output directory (Phase 51-B+ implementation)
- `src/course_work/phase51/` — Phase 51 implementation modules (Phase 51-B+)
- `src/course_work/reporting/phase_51_dashboard.py` — presentation-only renderer (Phase 51-H)
- `tests/unit/test_phase51_*.py` — unit tests (Phase 51-B+)

v1.13 KEPT UNAUTHORIZED:
- Phase 52+ implementation
- All Phase 47/48/49/50 canonical artifact modifications
- Training / inference / checkpoint loading / scaler fitting / optimizer steps
