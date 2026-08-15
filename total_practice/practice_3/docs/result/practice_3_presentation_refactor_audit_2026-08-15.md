# Practice 3 presentation refactor — final audit

Date: 2026-08-15

## Scope decision

The existing `checkpoint-1068` and one-time Test result remain authoritative. No post-Test learning-rate search was run. This preserves methodology but means controlled experiments are not claimed.

## Phase audit

| Phase | Status | What changed | Visualization added | Validation |
|---|---|---|---|---|
| 0 | PASS | Concise Problem Definition retained | Existing compact workflow | Rendered |
| 1–4 | PASS | No methodology/source change | Existing outputs retained | Executed |
| 5 | PASS | Added percentile and truncation presentation | Split sizes, token histogram with 80, token boxplot by class | P95=47, P99=56, max=78, over 80=0 |
| 6 | PASS | Added readable preprocessing examples | Compact flow presentation | Existing preprocessing and dynamic-padding PASS |
| 7 | PASS | Reduced architecture explanation to lecturer-facing flow | DistilBERT architecture | Forward sanity PASS |
| 8 | PASS | Baseline configuration retained | No artificial experiment chart | Synthetic metrics/config PASS |
| 9 | PASS | No retraining | Existing epoch/checkpoint tables | Guard loaded verified artifacts |
| 10 | PASS | Existing artifact-only analysis retained | Loss and Validation metric curves | Epoch 2 remains selected |
| 11 | PASS | Added compact gap analysis | Validation vs Final Test | Test count remains 1 |
| 12 | PASS | Added error-direction and confidence presentation | Confusion matrix, FP/FN, correct-vs-wrong confidence | Frozen predictions only |
| 13 | PASS | Existing custom examples retained; probability presentation added | Class-probability bars | No Test access |
| 14 | PASS | Existing equivalence table retained | Table-based equivalence | Package loaded, not resaved |
| 15 | PASS | Added final lecturer-facing metric view | Final Test metrics | Artifact-only summary PASS |

## Validation evidence

- Notebook Run All: 16/16 code cells executed; zero error outputs, 19 rendered PNG outputs, and zero stale Jupyter widget outputs.
- Phase 9: verified artifacts loaded without retraining.
- Phase 11: verified evaluation artifacts loaded without reevaluating Test.
- Phase 14: verified package loaded without resaving.
- `test_evaluation_count == 1`.
- Checkpoint and saved-package weight SHA-256 both equal `22fcd24d3db7bfebc8ea1dd4f6c0665be5176e34de005e168a747ee83acfa660`.
- Presentation artifact reports `training_performed=false`, `test_evaluated=false`, and `package_resaved=false`.

## Final status

- Practice 3 presentation quality: PASS
- Practice 3 methodology: PASS
- Visualization coverage: PASS
- Controlled experiments: FAIL / NOT RUN BY APPROVED OPTION 1
- Notebook Run All: PASS
- Final Test isolation: PASS

The controlled-experiment line is intentionally not presented as PASS because there is no comparable LR sweep artifact. This is the honest consequence of preserving the completed one-time-Test protocol.
