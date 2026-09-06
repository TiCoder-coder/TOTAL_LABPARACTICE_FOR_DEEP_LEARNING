# Phase 35 — Plan Compliance Gap Analysis

## Issue identity

- Issue ID: `CW-PHASE35-COMPLIANCE-001`
- Affected Phase: 35 — S13 Layer Sweep
- Status: `RESOLVED_WITH_DOCUMENTED_WARNING`
- Severity: `BLOCKING_PHASE36`
- Scientific result status: `FROZEN_VALID`

## User evidence

The final read-only compliance audit was requested against:

`docs/plan-doc/plan_detail_for_each_phase/Phase_35_S13_Layer_sweep.md`

The audit found the finalized scientific comparison intact but the Phase 35 evidence package and canonical schemas incomplete.

## Observed behavior

- N1 run: `RUN_TR_S13_0021_9CA63891`
- N1 Validation RMSE: `59.78262924121282` Wh
- N2 reference: `RUN_TR_S09_0016_AE0FB819`
- N2 Validation RMSE: `58.08190056355405` Wh
- Winner: `N2`
- Selected `num_layers`: `2`
- Phase status: `PASS_WITH_WARNING`
- Test: `FORBIDDEN`
- Current compliance result: `NOT_COMPLIANT`

The canonical finalizer generates several abbreviated summary artifacts instead of the exact schemas required by Sections 148–185. It also emits the wrong sweep identity and incomplete sign-off/handoff fields.

## Expected behavior

Phase 35 must preserve the frozen scientific result while satisfying all mandatory output schemas, BEST verification, provenance, sign-off, warning propagation, Test firewall, and Phase 36 handoff requirements in the Phase Detail.

## Confirmed causes

1. `_phase_35_extended_payloads()` in `src/course_work/sweeps/sweep_results.py` generates abbreviated rows for several O35 outputs.
2. `finalize_verified_sweep()` derives Phase 35 artifact identity from `spec.sweep_code`, producing `SWEEP_S13-v1` instead of `SWEEP_S13_LAYERS-v1`.
3. `_validate_phase_35_n1_evidence()` validates stored artifacts and saved predictions but does not perform the prescribed fresh-model strict-load full ordered Validation evaluation.
4. The Phase 35 sign-off branch omits required Section 218 fields and uses aliases such as `phase_id`, `selected_d_model`, and `warnings` without also emitting the exact required keys.
5. The Phase 35 reference-update branch omits exact required keys `dropout_probability` and `head_dim` and does not encode the F64/F128/F256 execution policy.
6. `inspect_layer_geometry()` returns summary booleans instead of all per-layer, per-parameter, optimizer-reference, attention, and initialization evidence needed by the required CSV schemas.
7. Focused tests verify only a reduced artifact subset and do not enforce all Section 164 cases or the Section 218 schema.

## Contributing factors

- Phase 35 was originally implemented as preparation logic and later finalized through a shared generic finalizer.
- Existing artifact existence and checksum validation did not validate the detailed Phase-specific CSV/JSON schema.
- N2 is a historical reference with incomplete retained artifacts, so unavailable evidence must remain explicitly unavailable rather than being inferred or fabricated.

## Affected files

- `src/course_work/sweeps/layers.py`
- `src/course_work/sweeps/sweep_results.py`
- `tests/unit/test_layers.py`
- `tests/unit/test_sweep_finalization.py`
- Phase 35 canonical artifacts under `artifacts/sweeps/S13_layers/`
- `docs/save_log_in_processing/phase_35_s13_layer_log.json`
- Phase 35 presentation in `notebook_course_work/CourseWork.ipynb` only if its canonical presentation contract requires refresh

## Architecture and working-rule ownership

- Phase 35 geometry/preflight owner: `src/course_work/sweeps/layers.py`
- Canonical sweep result/finalization owner: `src/course_work/sweeps/sweep_results.py`
- Presentation owner: `src/course_work/reporting/phase_summary.py`
- Notebook responsibility: orchestration/presentation only
- Scientific artifacts: `artifacts/sweeps/S13_layers/`
- Processing log: `docs/save_log_in_processing/phase_35_s13_layer_log.json`
- Test remains inaccessible and Phase 36 remains outside scope

## Reproduction path

1. Read Phase Detail Sections 148–185 and Section 218.
2. Compare the required schemas with current Phase 35 CSV/JSON headers and fields.
3. Inspect `phase_35_signoff.json`, `s13_reference_update.json`, and `s13_layer_sweep_manifest.json`.
4. Inspect `_validate_phase_35_n1_evidence()` and confirm it does not run strict-load full Validation verification.
5. Compare Section 164 with current `s13_layer_unit_tests.csv` and focused tests.

## Upstream impact

No upstream scientific result is invalidated. Phase 34 handoff and the frozen N2 historical-reference warning remain unchanged.

## Downstream impact

Phase 36 must remain blocked until Phase 35 produces a compliant reference update and sign-off. No Phase 36 code or execution is authorized by this correction.

## What must not change

- N1 or N2 model weights
- N1 or N2 training runs
- Winner `N2`
- Selected `num_layers=2`
- Full-precision RMSE values
- Phase 34 artifacts
- Experiment Registry scientific records
- Historical N2 missing-artifact declarations
- Test firewall
- Phase 36 execution state

## Recommended correction direction

Implement deterministic Phase-35-owned audit builders and a read-only N1 BEST verifier, then regenerate Phase 35 outputs only through the canonical finalizer with exact schemas and checksum verification.

## Open questions

None affecting the corrective design. Human approval of the corrective pre-process plan remains required before implementation.

## Evidence references

- `docs/plan-doc/plan_detail_for_each_phase/Phase_35_S13_Layer_sweep.md`
- `artifacts/sweeps/S13_layers/phase_35_signoff.json`
- `artifacts/sweeps/S13_layers/s13_reference_update.json`
- `artifacts/sweeps/S13_layers/s13_layer_discrepancies.json`
- `artifacts/runs/RUN_TR_S13_0021_9CA63891/`
- `artifacts/runs/RUN_TR_S09_0016_AE0FB819/`

## Decision

The Human approved `CW-PHASE35-COMPLIANCE-FIX-001`. The Phase-35-owned corrective implementation completed without retraining, Test access, winner changes, or Phase 36 execution.

## Resolution evidence

- Exact sweep identity is now `SWEEP_S13_LAYERS-v1`.
- N1 BEST strict-load and full ordered Validation verification passed.
- O35 mandatory artifacts are complete; optional omissions are explicit.
- Section 218 sign-off fields are complete and checksum-valid.
- Phase 36 handoff explicitly records `F64=TRAIN_NEW`, `F128=REUSE_REFERENCE`, and `F256=TRAIN_NEW`.
- Focused tests passed: `72 passed`.
- Audit-only and dry-run both report Phase 35 as `VALID_REUSABLE`.
- Idempotent finalization produced no checksum changes.
- The inherited N2 incomplete-retention warning remains, so the scientific status stays `PASS_WITH_WARNING`.
