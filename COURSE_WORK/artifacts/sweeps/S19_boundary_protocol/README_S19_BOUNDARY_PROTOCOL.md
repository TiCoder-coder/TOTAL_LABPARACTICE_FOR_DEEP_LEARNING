# S19 BOUNDARY PROTOCOL — README

## Phase 41 — Boundary Protocol Sensitivity Check

### What this sweep tests
This sweep is a **sensitivity check**, not a hyperparameter tuning exercise.  It compares two ways of assigning samples to Train / Validation / Test near split boundaries:

- **WB0 — Context Carry-Over (primary protocol):**  target split determines sample split; input rows may originate from the previous split if their timestamps are strictly less than the target timestamp and continuity is preserved.  Reflects rolling one-step deployment.
- **WB1 — Strict Split Isolation (sensitivity protocol):**  every input row AND the target row must belong to the same split.  No padding, no short windows, no synthetic history, no boundary interpolation.

Both protocols run on the **same frozen S1–S18 configuration**, selected from the canonical Phase 40 sign-off.  The only factor that changes between them is `window_boundary_protocol`.

### Selected S1–S18 configuration (frozen)
Resolved at runtime from the canonical Phase 40 winner run config (`artifacts/runs/RUN_TR_S14_0023_A711A9B8/config.json`).  Verified against Phase 36→40 handoff chain.  Only `window_boundary_protocol` may differ between WB0 and WB1.

| Field | Value | Source |
|------|-------|--------|
| FV* (feature variant) | FS2_TF1 | source run config: `data.feature_variant_id` |
| YS* (target scaling) | YS1 | source run config: `data.target_scaling_option` |
| L* (lookback) | 36 | source run config: `data.lookback_steps` |
| P* (pooling) | LAST_STEP | source run config: `model.pooling` |
| A* (activation) | GELU | source run config: `model.activation` |
| B* (batch size) | 32 | source run config: `training.batch_size` |
| LR* | 0.0003 | source run config: `training.learning_rate` |
| WD* | 0.001 | source run config: `training.weight_decay` |
| DR* (dropout) | 0.1 | source run config: `model.dropout` |
| D* (d_model) | 64 | source run config: `model.d_model` |
| H* (num_heads) | 4 | source run config: `model.num_heads` |
| HD* (head_dim) | 16 | derived = d_model / num_heads |
| N* (num_layers) | 2 | source run config: `model.num_layers` |
| F* (ffn_dim) | 256 | source run config: `model.ffn_dim` |
| LOSS* | MSE | source run config: `training.loss_name` |
| EPOCHS* | 50 | source run config: `training.max_epochs` |
| GC* | GC1 (clipping ON, max_norm=1.0) | source run config: `training.gradient_clipping_enabled` + `gradient_clip_max_norm` |
| RN* | RN0 (RevIN OFF) | source run config: `training.revin_enabled = false` |
| Seed | 42 | source run config: `reproducibility.seed` |

### Population Summary
| Split | WB0 native | WB1 native | COMMON | WB0 ONLY | WB1 ONLY | WB1 ⊆ WB0 |
|-------|-----------|-----------|--------|----------|----------|-----------|
| TRAIN | 13670 | 13670 | 13670 | 0 | 0 | EQUAL |
| VALIDATION | 2960 | 2924 | 2924 | 36 | 0 | YES |
| TEST | 2961 | 2925 | 2925 | 36 | 0 | YES |

WB1 strips exactly the **first L rows after the boundary** in each of Validation and Test.

### Selected run / Carry-over
- Primary protocol (WB0): `RUN_TR_S14_0023_A711A9B8` — REUSE_REFERENCE (no training, returns Phase 40 winning evidence).
- Sensitivity protocol (WB1): `PENDING_WB1_TRAINING` — TRAIN_NEW contract prepared; not yet executed.

### Status
Phase 41 PRE_TRAINING_COMPLETE.  WB1 scientific training is gated on explicit Human authorization only.

### Corrective action history
- 2026-08-24T22:45:00Z — initial pre-training artifacts materialized.
- 2026-08-24T22:55:00Z — frozen-config provenance corrective audit detected a human-authored documentation error in this README's configuration table.
- 2026-08-24T23:08:00Z — Human approved corrective plan at `COURSE_WORK/docs/plan-doc/plan_before_process/phase_41_s19_boundary_protocol_readme_contamination_corrective_plan.md`; this README's frozen-config table replaced with canonical values from `artifacts/runs/RUN_TR_S14_0023_A711A9B8/config.json`.
- No scientific result, resolver, population, or persisted artifact was modified by the corrective action.

### Files
- `phase_41_signoff.json` — pre-training sign-off (re-emitted with correction metadata)
- `s19_boundary_sweep_manifest.json` — sweep manifest
- `s19_boundary_sweep_contract.json` — sweep contract (WB0/WB1 semantics)
- `s19_reference_update.json` — Phase 42 handoff
- `live_sweep_results.jsonl` — sweep results (WB0 reused, WB1 pending)
- `s19_*.csv` — pre-training population and invariance audits