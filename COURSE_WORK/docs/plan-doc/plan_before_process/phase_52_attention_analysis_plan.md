# Phase 52 — Attention Extraction Pre-Process Plan

## Status

- **Phase ID**: PHASE_52_ATTENTION_EXTRACTION
- **Plan version**: ATTENTION_EXTRACTION-v1
- **Plan status**: APPROVED (Human approval granted in this run prompt)
- **Created**: 2026-09-04

## 1. Objective

Phase 52 is the **extraction + integrity** phase for attention. It does NOT
interpret attention.

It produces:

- Per-seed dense attention `[K, L, L, L, L]` for the frozen Phase51 case set
- Per-seed all-Test last-query attention `[N_test, L, H, L]`
- Per-target/per-layer/per-head summaries (entropy, lag, top-k, recent-mass)
- Audits and reproducibility evidence
- Phase53–57 handoff files (no Phase53–57 implementation)

## 2. Authorized sub-phases

- 52-A: governance + preflight + source verification + architecture amendment
- 52-A2: architecture amendment v1.14 application (governance log entry)
- 52-B: extraction infrastructure + extraction contract freeze
- 52-C: official raw attention extraction (3 seeds, dense + last-query)
- 52-D: integrity / probability / ordering / reproducibility / mutation audits
- 52-E: derived summaries
- 52-F: Phase53–57 handoffs + findings + discrepancies
- 52-G: tests + summary + report + README + signoff
- 52-H: notebook visualization (NOT IN THIS RUN; gated separately)

## 3. Scientific boundary

Phase52 MUST NOT:

- train, fine-tune, optimize, backward, fit scaler
- alter Phase47 prediction values
- re-rank Phase51 cases
- select a best seed
- create an ensemble
- interpret attention causally
- claim attention = feature importance
- decide best head
- start heatmap interpretation

Phase52 MAY:

- strict-load the 3 final checkpoints (`FINAL_TR_SEED42/123/2026`)
- reconstruct the exact final Transformer
- execute eval/inference-mode forward passes
- obtain raw attention via `forward_with_attention`
- verify predictions against Phase47 frozen values
- persist raw attention (float32, no averaging)
- compute extraction diagnostics
- write downstream handoffs (Phase53–57)

## 4. Inputs (frozen)

- Phase47 prediction bundles (test population, target_ids, y_true_wh, y_pred_wh)
- Phase48 prediction analysis (seed spread)
- Phase49 residual artifacts
- Phase50 regime assignment
- Phase51 worst-error case set + handoff
- Phase45 final model lock (config + lock + recipe)
- Phase46 three-seed final checkpoint manifest
- Final scaler registry (FINAL_SCALING-v1)
- Feature contract (FS2_TF1, 33 features)
- Boundary contract (WB0_CONTEXT_CARRY_OVER)
- ATTENTION_VERIFY-v1 attention schema

## 5. Outputs (O52.1–O52.37)

| ID    | File                                                 |
|-------|------------------------------------------------------|
| O52.1 | attention_extraction_manifest.json                   |
| O52.2 | attention_extraction_contract.json                   |
| O52.3 | phase52_preflight_audit.csv                          |
| O52.4 | attention_source_verification.csv                    |
| O52.5 | attention_checkpoint_verification.csv                |
| O52.6 | attention_environment_audit.csv                      |
| O52.7 | attention_extraction_batch_audit.csv                 |
| O52.8 | attention_prediction_equivalence_audit.csv           |
| O52.9 | attention_tensor_integrity_audit.csv                 |
| O52.10| attention_probability_audit.csv                      |
| O52.11| attention_model_mutation_audit.csv                   |
| O52.12| attention_reproducibility_audit.csv                  |
| O52.13| attention_test_target_order.csv                      |
| O52.14| attention_dense_case_order.csv                       |
| O52.15| attention_relative_position_map.csv                  |
| O52.16| attention_case_position_map.csv                      |
| O52.17| attention_case_metadata.csv                          |
| O52.18| raw/dense_case_attention_seed{42,123,2026}.npz       |
| O52.19| raw/last_query_attention_seed{42,123,2026}.npz       |
| O52.20| raw_attention_checksums.json                         |
| O52.21| attention_dense_last_query_consistency.csv           |
| O52.22| attention_last_query_summary.csv                     |
| O52.23| attention_full_matrix_summary.csv                    |
| O52.24| attention_recent_mass_summary.csv                    |
| O52.25| attention_top_source_summary.csv                     |
| O52.26| attention_extraction_findings.csv                    |
| O52.27| phase53_attention_heatmaps_handoff.json              |
| O52.28| phase54_last_query_attention_handoff.json            |
| O52.29| phase55_head_comparison_handoff.json                 |
| O52.30| phase56_error_conditioned_attention_handoff.json     |
| O52.31| phase57_seed_stability_attention_handoff.json        |
| O52.32| attention_extraction_tests.csv                       |
| O52.33| attention_extraction_discrepancies.json              |
| O52.34| attention_extraction_summary.json                    |
| O52.35| attention_extraction_report.md                       |
| O52.36| README_ATTENTION_EXTRACTION.md                       |
| O52.37| phase_52_signoff.json                                |

Total: 37 artifacts.

## 6. Hard contracts

- Lookback L = 72, Feature count F = 33, FS2_TF1
- `forward_with_attention(x)` returns `(pred, [B,H,L,L] per layer)`
- `need_weights=True`, `average_attn_weights=False`, no masks
- Last-query definition: `A[:, :, L-1, :]`
- Storage dtype: float32, no averaging, no thresholding, no smoothing
- Two-tier storage: dense cases + all-Test last-query

## 7. Human authorization gates

- Architecture amendment v1.14 (Phase52): **APPROVED in this run prompt**
- Pre-process plan: **APPROVED in this run prompt**
- Notebook visualization (Phase52-H): **NOT APPROVED** (deferred)
- Phase53–57 implementation: **NOT APPROVED** (handoffs only)
