## FT10 - Evidence and Limitation Summary

_Upstream-supported claim trace with caveats._

| question | primary_evidence | evidence_class | what_can_be_concluded | what_cannot_be_concluded | source_phase |
|---|---|---|---|---|---|
| Does Transformer generalize on held-out Test? | FT02 (Phase 47 final Test metrics) | HELD_OUT_TEST_EVIDENCE | Three-seed final Transformer metrics on FINAL_TEST_POP-v1 (N=2961). | No multi-house generalization. Single-house dataset only. | Phase 47 |
| Does Transformer compare with LSTM? | FT02 + FA01 | HELD_OUT_TEST_EVIDENCE | Side-by-side metric comparison on shared FINAL_TEST_POP-v1. | No significance test; no bootstrap CI; no Diebold-Mariano. | Phase 47 |
| Is performance temporally robust pre-Test? | FT03 (Phase 44 pooled rolling-origin RMSE) | DEVELOPMENT_EVIDENCE | Pooled outer-fold RMSE only — development evidence, not final generalization. | No substitution for Held-Out Test evidence. | Phase 44 |
| Where are largest errors? | FT05 + FA05 | POST_TEST_DIAGNOSTIC_EVIDENCE | Phase 50/51 worst-case concentration. | Worst cases are not deployment regimes. | Phase 50/51 |
| Which regimes are difficult? | FT05 Panel A | POST_TEST_DIAGNOSTIC_EVIDENCE | Frozen Phase 50 regime labels; per-regime MAE/RMSE/R². | No regime threshold modification. | Phase 50 |
| What temporal lags receive attention? | FT06 + FA06 | POST_TEST_DIAGNOSTIC_EVIDENCE | Phase 54 quantitative last-query summaries — temporal allocation only. | Attention is NOT raw-feature importance; NOT causal. | Phase 54 |
| Are heads diverse? | FT07 + FA07 | POST_TEST_DIAGNOSTIC_EVIDENCE | Phase 55 layer head diversity (no composite score). | Similarity does NOT prove functional redundancy. No pruning/ablation. | Phase 55 |
| Does attention co-vary with error? | FT08 + FA08 | POST_TEST_DIAGNOSTIC_EVIDENCE | Phase 56 error-conditioned association (Spearman / Cliff's delta). | No causal interpretation; HIGH/LOW are diagnostic only. | Phase 56 |
| Is attention stable across seeds? | FT09 + FA09/FA10 | POST_TEST_DIAGNOSTIC_EVIDENCE | Phase 57 stability (S57-A layer head-mean + S57-B canonical matching). | Same-index heads NOT assumed semantically aligned; no STABLE/UNSTABLE threshold. | Phase 57 |
| Can attention be interpreted causally? | Mandatory caveat in all FT06-FT09 | EVIDENCE_AND_LIMITATION | Attention is descriptive temporal allocation only. | Attention does NOT establish causality or feature importance. | Phase 54-57 |

> Single-house dataset. No multi-house generalization possible.
