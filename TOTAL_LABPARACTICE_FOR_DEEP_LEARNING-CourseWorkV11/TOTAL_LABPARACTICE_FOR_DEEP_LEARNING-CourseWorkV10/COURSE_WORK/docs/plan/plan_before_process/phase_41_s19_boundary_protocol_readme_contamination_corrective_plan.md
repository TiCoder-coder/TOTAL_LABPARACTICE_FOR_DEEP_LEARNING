# Phase 41 — S19 Boundary Protocol README contamination corrective plan

## Context

Phase 41 PRE-TRAINING CHECKPOINT reported a frozen S1–S18 configuration that
disagreed with the canonical handoff from Phase 40.  The reported values
(FS1_TF1, L144, B64, WD=0.0001, F128) did not match the source run's
persisted configuration (FS2_TF1, L36, B32, WD=0.001, F256).

A provenance audit was performed (Phase 41 frozen-config provenance corrective
audit).  This plan covers the corrective action.

## Root-cause analysis

The Phase 41 resolver (`src/course_work/sweeps/boundary_protocol.py
resolve_frozen_config`) reads directly from
`artifacts/runs/RUN_TR_S14_0023_A711A9B8/config.json` and returns the **correct**
values (FS2_TF1, L36, B32, WD=0.001, F256, GELU, MSE, max_epochs=50,
gradient_clipping ON, max_norm=1.0, RevIN OFF, seed=42).

All artifacts produced by `materialize_pre_training_artifacts` were likewise
written from the correct resolver output.  Inspecting each artifact on disk
confirms the canonical values:

| Artifact | FS* | L* | B* | WD | F* |
|---|---|---|---|---|---|
| `s19_boundary_sweep_manifest.json` | FS2_TF1 | 36 | 32 | 0.001 | 256 |
| `s19_boundary_sweep_contract.json` | — | 36 | — | — | — |
| `s19_training_config_delta_audit.csv` | FS2_TF1 | 36 | 32 | 0.001 | — |
| `s19_reference_update.json` | (no fields stored) |
| `phase_41_signoff.json` | (no fields stored) |
| `live_sweep_results.jsonl` | (no fields stored) |

The **only** contaminated artifact is the human-readable
`README_S19_BOUNDARY_PROTOCOL.md`.  Its "Selected S1–S18 configuration"
table was authored manually and incorrectly transposed (it shows FS1_TF1,
L144, B64, WD=0.0001, F128).  These values do not match the source run, the
Phase 40 reference, or any artifact on disk.

## What must NOT change

- Source run config (`RUN_TR_S14_0023_A711A9B8/config.json`) — already correct
- Phase 41 resolver (`src/course_work/sweeps/boundary_protocol.py`) — already correct
- Phase 41 artifacts on disk — already correct
- Pre-training populations (built with L36) — already correct
- Population counts (Train=13670, Val=2960/2924, Test=2961/2925) — already correct
- Frozen S1–S18 scientific configuration — already correct
- Test access (FORBIDDEN) — already correct
- No scientific run, no checkpoint, no metrics — already correct

## Recommended repair layer

**Single-file human-readable documentation fix only.**

Replace the "Selected S1–S18 configuration (frozen)" table in
`README_S19_BOUNDARY_PROTOCOL.md` with the values actually resolved from the
canonical source run.

This is a pure documentation correction.  It does not change any scientific
field, the resolver, the resolver output, the populations, the WB0/WB1
contracts, the dry-run gates, the focused tests, or any persisted artifact.

## Files in scope

- `artifacts/sweeps/S19_boundary_protocol/README_S19_BOUNDARY_PROTOCOL.md`
  — replace the configuration table.

No other file in scope.

## Files explicitly OUT of scope

- `scripts/run_single_condition.py` — already wired correctly for S19_BOUNDARY_PROTOCOL
- `src/course_work/sweeps/boundary_protocol.py` — already correct
- `tests/unit/test_boundary_protocol_dry_run.py` — already passing 36/36
- All `artifacts/sweeps/S19_boundary_protocol/s19_*.{json,csv}` files
- `docs/save_log_in_processing/phase_41_s19_boundary_protocol_log.json`
- `phase_41_signoff.json`

## Corrected values (per source run config)

| Field | Value | Source |
|---|---|---|
| FV* (feature variant) | FS2_TF1 | source run config: data.feature_variant_id |
| YS* (target scaling) | YS1 | source run config: data.target_scaling_option |
| L* (lookback) | 36 | source run config: data.lookback_steps |
| P* (pooling) | LAST_STEP | source run config: model.pooling |
| A* (activation) | GELU | source run config: model.activation |
| B* (batch size) | 32 | source run config: training.batch_size |
| LR* | 0.0003 | source run config: training.learning_rate |
| WD* | 0.001 | source run config: training.weight_decay |
| DR* (dropout) | 0.1 | source run config: model.dropout |
| D* (d_model) | 64 | source run config: model.d_model |
| H* (num_heads) | 4 | source run config: model.num_heads |
| HD* (head_dim) | 16 | source run config: derived from d_model/num_heads |
| N* (num_layers) | 2 | source run config: model.num_layers |
| F* (ffn_dim) | 256 | source run config: model.ffn_dim |
| LOSS* | MSE | source run config: training.loss_name |
| EPOCHS* | 50 | source run config: training.max_epochs |
| GC* | GC1 (clipping ON, max_norm=1.0) | source run config: training.gradient_clipping_enabled + gradient_clip_max_norm |
| RN* | RN0 (RevIN OFF) | source run config: training.revin_enabled = false |
| Seed | 42 | source run config: reproducibility.seed |

## Validation steps (after fix)

1. Read corrected `README_S19_BOUNDARY_PROTOCOL.md` and verify the table matches the source run config.
2. Re-run focused tests (must still be 36/36 PASS).
3. Re-run `--dry-run` for both WB0 and WB1 (must still reach boundary without scientific state).
4. Verify no other Phase 41 artifact changed (file hashes / timestamps).
5. Update processing JSON to record the corrective action with timestamp.
6. Re-issue the final PHASE 41 PRE-TRAINING CHECKPOINT.

## Regression steps (after fix)

- Run all S17 + S18 + S19 dry-runs to confirm no regression.
- Run full focused test file `tests/unit/test_boundary_protocol_dry_run.py`.
- Verify Phase 40 signoff and reference update files remain unchanged.
- Verify Phase 36/37/38/39/40 winner and reference files remain unchanged.

## Test firewall

No Test access required.  No Test target values read.  No Test predictions
or metrics.  This is documentation-only.

## Sign-off correction policy

`phase_41_signoff.json` records no frozen-config fields, so it requires no
correction.  However, after the README is fixed, the `phase_41_signoff.json`
will be re-emitted with:
- `last_corrected_at` timestamp
- `correction_reason = "README_TABLE_VALUES_REPLACED_WITH_CANONICAL"`
- `correction_authority = "COURSE_WORK/docs/plan-doc/plan_before_process/phase_41_s19_boundary_protocol_readme_contamination_corrective_plan.md"`

The `wb0_reference.validation_rmse_wh = 57.69679988114431` is preserved
unchanged because the source run was not modified.

## Human approval gate

This plan must NOT be implemented until explicit Human approval is granted
for the corrective action.  The current Human instruction in this audit
explicitly states: "Do not silently repair them in this audit."  The plan
document is the required artifact for that explicit approval gate.

STOP before any source code modification, scientific training, or Phase 42
progression.