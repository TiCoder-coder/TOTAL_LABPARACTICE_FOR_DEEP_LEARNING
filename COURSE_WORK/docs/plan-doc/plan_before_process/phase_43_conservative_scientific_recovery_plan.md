# Phase 43 Conservative Scientific Recovery Plan

## Objective

Restore Phase 43 to the authoritative `LSTM_TUNING-v1` contract without changing upstream Phase 1–42 scientific results and without accessing Test.

Official scientific training is human-only.

## Entry gate

Before implementation work:

1. Phase 42 signoff must be PASS/PASS_WITH_WARNING and ready for Phase 43.
2. Canonical Phase 42 → 43 handoff must be physically present and audited.
3. Selected feature variant, target scaling, lookback, boundary protocol, population fingerprints, scaler checksums, and batch policy must be resolved from the handoff rather than hard-coded.
4. Test firewall must be active.

## Recovery steps

### 1. Preserve historical Phase 43 evidence

Inventory all historical `LSTM_TUNING` runs and current Phase 43 artifacts.

Mark current Phase 43 outputs as historical-invalidated-by-protocol-deviation through additive metadata only. Do not delete checkpoints, logs, metrics, histories, or registry records.

### 2. Remove unauthorized fast-mode behavior

The official scientific code path must not set:

- `max_epochs = 2`;
- `patience = 2`;
- any reduced candidate budget not declared by the plan.

The official configuration must use:

- `max_epochs = 50`;
- `patience = 10`;
- `min_delta = 0`;
- AdamW;
- MSE;
- gradient clipping max-norm 1.0;
- scheduler none;
- warmup none;
- seed 42.

Any fast/smoke mode must be explicitly non-scientific and must never write official Phase 43 signoff/winner artifacts.

### 3. Reconcile shared-data contract

Before training, verify exact equality against the canonical Phase 42 handoff for:

- feature variant;
- feature count/order/fingerprint;
- target scaling;
- lookback;
- WB0 boundary protocol;
- Train target IDs;
- Validation target IDs;
- population fingerprint;
- X/Y scaler checksums;
- split and metric versions;
- batch size policy.

Resolve the current mismatch where the uploaded Phase 43 signoff reports lookback 36 but the uploaded tuned winner config reports lookback 144.

### 4. Correct DataLoader semantics

Construct fresh Train/Validation DataLoaders for each fresh candidate using the fixed batch size `B*` from the Phase 42 shared context.

Do not create batch-size-1024 runtime loaders while recording batch size 64 in run config.

For each fresh candidate:

- fresh loader;
- same sample IDs;
- same loader seed policy;
- same batch size;
- fresh model;
- fresh optimizer;
- fresh early-stop state.

### 5. Restore sequential one-factor tuning

Execute only:

- LT1: hidden size 32/64/128;
- LT2: layers 1/2;
- LT3: dropout 0/0.1/0.2 only if selected layers >=2;
- LT4: LR 1e-4/3e-4/1e-3;
- LT5: WD 0/1e-4/1e-3.

Every stage must include the current reference and the exact stage winner becomes the next reference.

No Cartesian grid, warm-start, backtracking combination, or extra candidate.

### 6. Restore winner selection semantics

For each stage:

- evaluate full ordered Validation every epoch;
- choose BEST by full-precision Validation RMSE Wh;
- use MAE/R² only as secondary metrics;
- apply exact-tie parsimony rules;
- retain failed candidates rather than silently omitting them.

The final `LSTM_TUNED` must be an actually completed registered LT5 run.

### 7. Rebuild required Phase 43 artifacts

Generate all O43.1–O43.37 outputs from real run evidence.

No dummy PASS audit rows and no synthetic/placeholder metric values.

At minimum verify:

- run matrix;
- stage lineage;
- per-stage metrics and winners;
- common-data audit;
- initialization/sample-order audits where verifiable;
- optimizer budget/group audits;
- gradient/convergence/runtime diagnostics;
- tuned winner;
- Phase 44 handoff;
- discrepancies;
- summary/report/README;
- signoff.

### 8. Signoff consistency gate

Before Phase 43 signoff:

- winner hyperparameters in signoff must exactly match `lstm_tuned_winner.json`;
- winner run ID must match;
- lookback/shared-data fields must match;
- reference/tuned metrics must match canonical run artifacts;
- fresh/reused/failed run counts must match registry evidence;
- Test status must be `NOT_ACCESSED`;
- discrepancies must be empty or correctly reflected in status.

### 9. Tests and dry-run

Before official training, run non-scientific tests proving:

- E50/P10 on scientific path;
- no fast-mode contamination;
- exact Phase 42 handoff resolution;
- correct batch-size semantics;
- stage candidate spaces;
- one-factor deltas;
- stage reference inclusion;
- Test firewall;
- no official run registration during dry-run.

The dry-run must stop before `registry.register_run()` / optimizer steps.

## Human-only scientific execution

After all gates pass, stop and provide the human with the exact Phase 43 terminal command.

The AI agent must not execute official Phase 43 scientific training.

## Post-run acceptance

Phase 43 may be declared PASS only when:

- all applicable stages completed under E50/P10;
- maximum fresh run budget <=10;
- final winner checkpoint is verified;
- final winner is not worse than the reference in exact stage lineage;
- shared Train/Validation populations and scaler provenance are identical across candidates;
- all required artifacts are real and internally consistent;
- Phase 44 handoff is ready;
- Test was never accessed.

## Downstream gate

Phase 44 remains BLOCKED until corrected Phase 43 receives a verified PASS/PASS_WITH_WARNING signoff.
Phase 45, Phase 46, and Phase 47 remain BLOCKED.
