# Phase 36 S14 FFN Sweep

## 1. Objective

Compare absolute FFN widths 64, 128 and 256 under one frozen Validation-only protocol.

## 2. Current reference from S13

F128 reuses RUN_TR_S09_0016_AE0FB819 without retraining and carries the inherited retention warning.

## 3. F64/F128/F256 definitions

F64=64, F128=128, F256=256.

## 4. Selected D/H/N context

D64, H4, head_dim16 and N2 are fixed.

## 5. FFN geometry

Every active layer uses D→M→D with one global candidate M.

## 6. Expansion-ratio context

Ratios 1.0, 2.0 and 4.0 are derived from absolute width and D64.

## 7. Frozen-variable contract

Data, optimizer, MHA, PE, normalization, pooling, head, activation and dropout are fixed.

## 8. Architecture shape-delta whitelist

Only linear1/linear2 FFN hidden-dependent shapes differ.

## 9. MHA/non-FFN invariance

Both audits PASS.

## 10. Parameter-count and delta audit

52673 < 69185 < 102209; all pairwise deltas match the registered FFN formula.

## 11. Initialization/sample-order fairness

Fresh F64/F256 use seed 42 and the same loader policy. Historical F128 order traces are not retained.

## 12. F128 reference provenance

The exact Phase 35 winner is reused with no hidden rerun.

## 13. F64/F256 run provenance

RUN_TR_S14_0022_AA048302 and RUN_TR_S14_0023_A711A9B8 are complete fresh runs.

## 14. Attention API sanity

Shapes, finiteness, non-negativity, row sums and standard/inspection prediction agreement PASS.

## 15. Validation metrics

- F64: RMSE 58.39803077736952 Wh; MAE 27.061105884921158; R² 0.5991643867137383
- F128: RMSE 58.08190056355405 Wh; MAE 27.595002038670813; R² 0.6034923842647237
- F256: RMSE 57.69679988114431 Wh; MAE 27.029547974302105; R² 0.6087328858180434

## 16. Pairwise FFN effects

All registered pairs are recorded in s14_ffn_pairwise_effects.csv.

## 17. Width-trend diagnostics

The registered response is monotonic improvement with width.

## 18. Learning-curve/convergence analysis

F64/F256 histories are retained; F128 history remains unavailable.

## 19. Gradient/clipping analysis

The frozen clip policy is verified; detailed batch gradient traces were not retained.

## 20. Runtime/capacity context

Runtime, checkpoint and parameter evidence are secondary and never override RMSE.

## 21. Optional generalization analysis

Not generated because the required historical evidence is incomplete; no Test proxy was used.

## 22. S14 winner

F256 (`ffn_dim=256`) wins by full-precision Validation RMSE 57.69679988114431 Wh.

## 23. Interpretation cautions

F256 is an upper-boundary, single-seed Validation winner; no F512 condition was added.

## 24. Interaction limitations

This sweep does not test FFN interactions with D, H, N, activation, dropout, optimizer or training budget.

## 25. Phase 37 handoff

Reuse the selected MSE winner; train only Huber. Phase 37 was not executed. Test remained forbidden.
