# Phase 35 — S13 Layer Sweep Preparation Plan

## Scope and approval

This plan prepares Phase 35 only. Human approval was granted on 2026-08-24 for implementation, focused tests, audit-only, dry-run and preflight. Scientific N1 training and all Test access remain outside this checkpoint.

## Scientific contract

- Objective: compare Transformer Encoder depth while every selected Phase 34 factor remains frozen.
- Swept field: `model.num_layers` only.
- N1: `num_layers=1`, `TRAIN_NEW`, fresh seed 42 model/loaders/optimizer.
- N2: `num_layers=2`, `REUSE_REFERENCE`, exact Phase 34 H4 winner `RUN_TR_S09_0016_AE0FB819`.
- Frozen architecture: `d_model=64`, `num_heads=4`, `head_dim=16`, `ffn_dim=128`, GELU, LAST_STEP, dropout 0.1, sinusoidal PE, POST_NORM, no masks.
- Frozen training: B32, AdamW, learning rate 0.0003, weight decay 0.001, MSE, E50, patience10, min_delta0, clip1, no scheduler/warmup, accumulation1, seed42.
- Primary metric: full-precision Validation RMSE Wh, lower is better.
- Exact tie: N1.
- Test: `FORBIDDEN`.

## Phase 34 handoff

Canonical inputs:

- `artifacts/sweeps/S12_heads/s12_head_winner.json`
- `artifacts/sweeps/S12_heads/s12_reference_update.json`
- `artifacts/sweeps/S12_heads/phase_34_signoff.json`

The inherited H4 warning remains active: `H4_SOURCE_ARTIFACT_RETENTION_INCOMPLETE`. N2 is accepted only as `HISTORICAL_REFERENCE_WITH_INCOMPLETE_ARTIFACT_RETENTION`; its missing checkpoint/history/prediction/log must not be fabricated or silently retrained.

## Preparation implementation

1. Register Phase 35 and N1/N2 in selective phase execution.
2. Validate Phase 34 signed handoff and exact N2 reference evidence.
3. Build disposable N1/N2 models for geometry-only preflight.
4. Verify output/attention shapes, strict state-dict subset, independent N2 layers, shared tensor shapes and positive parameter delta.
5. Permit the manual runner to change only `model.num_layers` from 2 to 1 for N1.
6. Keep N2 reuse-only and Test forbidden.
7. Expose a presentation-only Phase 35 notebook section backed by the JSON processing log.
8. Keep Phase 35 canonical scientific outputs absent until N1 training and finalization occur.

## Safe checks

Use the canonical interpreter:

```bash
cd COURSE_WORK
MPLCONFIGDIR=/private/tmp/course-work-mpl-cache \
PYTHONPATH=src \
/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10 \
  -m pytest tests/unit/test_layers.py tests/unit/test_phase_execution.py \
  tests/unit/test_condition_runner_config.py tests/unit/test_selective_phase_reporting.py -q
```

Audit-only and dry-run:

```bash
cd COURSE_WORK
PYTHONPATH=src /Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10 \
  scripts/run_all_pending.py --phase-id 35 --audit-only

PYTHONPATH=src /Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10 \
  scripts/run_all_pending.py --phase-id 35 --dry-run
```

These commands must report N1 missing/TRAIN_NEW, N2 reusable, upstream/environment ready and no scientific artifact writes.

## Manual scientific execution gate

Do not run during preparation. After explicit Human authorization, run exactly one N1 condition from `COURSE_WORK/` and route raw output to `artifacts/sweeps/logs/`:

```bash
caffeinate -i env MPLCONFIGDIR=/private/tmp/course-work-mpl-cache PYTHONPATH=src \
/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10 -u \
scripts/run_single_condition.py S13_LAYERS N1 \
2>&1 | tee artifacts/sweeps/logs/phase_35_n1_terminal.log
```

When the runner prints the allocated run ID, the terminal log must be renamed to the canonical descriptive form:

```text
artifacts/sweeps/logs/phase_35_n1_<RUN_ID>_terminal.log
```

No N2 retraining and no Test access are authorized.

## Finalization record — 2026-08-24

- N1 completed as `RUN_TR_S13_0021_9CA63891` with 25 epochs, best epoch 15 and full-precision Validation RMSE `59.78262924121282` Wh.
- N2 remained the historical Phase 34 reference `RUN_TR_S09_0016_AE0FB819` with full-precision Validation RMSE `58.08190056355405` Wh.
- N2 won by the predeclared Validation-only rule; exact-tie handling was not invoked.
- Phase 35 canonical status is `PASS_WITH_WARNING` because `H4_SOURCE_ARTIFACT_RETENTION_INCOMPLETE` remains inherited.
- Phase 35 canonical artifacts and Phase 36 handoff were materialized under `artifacts/sweeps/S13_layers/`.
- Test remained forbidden and Phase 36 was not executed.
