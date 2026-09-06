# Phase 35 — Plan Compliance Corrective Plan

## Objective

Bring Phase 35 into compliance with Phase_35_S13_Layer_sweep.md without retraining, Test access, or changing the scientific winner.

## Frozen result

- N1: RUN_TR_S13_0021_9CA63891
- N1 RMSE: 59.78262924121282 Wh
- N2: RUN_TR_S09_0016_AE0FB819
- N2 RMSE: 58.08190056355405 Wh
- Winner: N2
- selected_num_layers: 2

## Required corrective work

1. Perform the prescribed read-only N1 BEST verification:
   fresh num_layers=1 model -> strict-load BEST checkpoint -> full ordered Validation -> METRICS-v1 -> compare to stored metrics.

2. Materialize required checkpoint provenance:
   model config fingerprint and architecture-role fingerprint.

3. Upgrade all required Phase 35 audit outputs to the exact schemas in the Phase Detail. Do not use summary PASS rows where per-layer/per-parameter evidence is required.

4. Expand layer unit-test evidence to the minimum cases defined by the Phase Detail.

5. Complete common-data audit fields:
   Train/Validation counts, sample-ID equality, ordered-ID equality where verifiable, feature/scaler/target/population equality, frozen configuration equality, Test lock.

6. Regenerate phase_35_signoff.json through the canonical finalization path with all minimum required fields.

7. Regenerate s13_reference_update.json with:
   dropout_probability, head_dim, selected_num_layers=2, current_ffn_dim=128, approved_for_phase36=true.

8. Explicitly encode Phase 36 condition policy:
   F128=REUSE_REFERENCE, F64=TRAIN_NEW, F256=TRAIN_NEW.

9. Correct canonical sweep identity to SWEEP_S13_LAYERS-v1.

10. Run focused tests, audit-only, dry-run, and idempotent finalization verification.

## Forbidden

- no N1 retraining
- no N2 retraining
- no winner change
- no Test access
- no fabricated historical evidence
- no checksum/sign-off bypass
- no Phase 36 execution

Phase 36 may begin only after a final compliance audit reports compliant status and Phase36 may start = YES.
