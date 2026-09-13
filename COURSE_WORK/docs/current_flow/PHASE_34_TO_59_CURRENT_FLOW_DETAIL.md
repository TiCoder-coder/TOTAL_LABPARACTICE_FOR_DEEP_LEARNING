# PHASE 34 TO 59 — Current Flow Detail

> Tài liệu chi tiết dòng chảy của từng phase từ 34 đến 59, dựa trên kiến trúc source code đã được refactor clean architecture.

## Trạng thái recovery hiện tại của Phase 34–41

| Phase | State | Effective action | Ghi chú lineage/reporting |
|---:|---|---|---|
| 34 | `VALID_REUSABLE` | `RENDER_ONLY` | Historical signoff/output được kiểm tra checksum; processing log hiện hành hợp lệ. |
| 35 | `VALID_REUSABLE` | `RENDER_ONLY` | Historical signoff/output được kiểm tra checksum; processing log hiện hành hợp lệ. |
| 36 | `VALID_REUSABLE` | `RENDER_ONLY` | Historical signoff/output được kiểm tra checksum; processing log được rebuild chỉ cho reporting. |
| 37 | `VALID_REUSABLE` | `RENDER_ONLY` | Exact legacy signoff và registry-bound scientific core được xác minh; các CSV reporting không có trong historical tree vẫn là debt. |
| 38 | `VALID_REUSABLE` | `RENDER_ONLY` | Exact legacy signoff và registry-bound scientific core được xác minh; các CSV reporting không có checksum lịch sử vẫn là debt. |
| 39 | `VALID_REUSABLE` | `RENDER_ONLY` | Exact legacy contract và canonical run core được xác minh; reporting-only omissions không được fabricate. |
| 40 | `VALID_REUSABLE` | `RENDER_ONLY` | Exact legacy contract và canonical RN1 scientific core được xác minh. |
| 41 | `VALID_REUSABLE` | `RENDER_ONLY` | Exact legacy contract, boundary comparison và canonical WB1 scientific core được xác minh. |

Các compatibility rule tương ứng là fail-closed: signoff/hash, phase identity,
canonical run và registry-bound core artifact phải khớp. Việc thiếu artifact chỉ
được ghi là reporting debt khi exact historical contract cho phép; không file
khoa học nào được tổng hợp lại để làm validator PASS.

---

## Phase 34 — S12: Heads Sweep

**Module:** `src/course_work/sweeps/heads.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | Phase 33 winner (d_model) |
| **Output** | `artifacts/sweeps/S12_heads/{sweep_manifest.json, results.csv, s12_head_winner.json, s12_reference_update.json}`, `phase_34_signoff.json` |
| **Output (figures)** | `artifacts/sweeps/S12_heads/figures/` |

**Nhiệm vụ:**
- Sweep `num_heads ∈ {2, 4, 8, 16}`
- Validation RMSE, MAE, learning curves
- Reference update

---

## Phase 35 — S13: Layers Sweep

**Module:** `src/course_work/sweeps/layers.py`

| Thành phần | Chi tiết |
|---|---|
| **Output** | `artifacts/sweeps/S13_layers/`, `phase_35_signoff.json` |

**Nhiệm vụ:**
- Sweep `num_layers ∈ {1, 2, 3, 4, 6}`

---

## Phase 36 — S14: FFN Sweep

**Module:** `src/course_work/sweeps/ffn.py`

| Thành phần | Chi tiết |
|---|---|
| **Output** | `artifacts/sweeps/S14_ffn/`, `phase_36_signoff.json` |

**Nhiệm vụ:**
- Sweep FFN width multipliers

---

## Phase 37 — S15: Loss Sweep

**Module:** `src/course_work/sweeps/loss.py`

| Thành phần | Chi tiết |
|---|---|
| **Output** | `artifacts/sweeps/S15_loss/{s15_loss_winner.json, s15_reference_update.json}`, `phase_37_signoff.json` |

**Nhiệm vụ:**
- Sweep loss functions (MSE, Huber, Smooth L1)

---

## Phase 38 — S16: Epoch Cap Sweep

**Module:** `src/course_work/sweeps/epoch_cap.py`

| Thành phần | Chi tiết |
|---|---|
| **Output** | `artifacts/sweeps/S16_epoch_cap/`, `phase_38_signoff.json` |

**Nhiệm vụ:**
- Sweep max epochs cap

---

## Phase 39 — S17: Gradient Clipping Sweep

**Module:** `src/course_work/sweeps/gradient_clip.py`

| Thành phần | Chi tiết |
|---|---|
| **Output** | `artifacts/sweeps/S17_gradient_clipping/`, `phase_39_signoff.json` |
| **Scripts** | `src/course_work/scripts/p39_*.py` (4 scripts) |

**Nhiệm vụ:**
- Sweep gradient clipping thresholds
- Strict best (gc0) verification
- Compliance corrective

---

## Phase 40 — S18: RevIN Sweep

**Module:** `src/course_work/sweeps/revin.py` + `src/course_work/models/revin.py`

| Thành phần | Chi tiết |
|---|---|
| **Output** | `artifacts/sweeps/S18_revin/`, `phase_40_signoff.json` |
| **Scripts** | `src/course_work/scripts/p40_*.py` (3 scripts) |

**Nhiệm vụ:**
- Sweep RevIN applicability (on/off)
- Strict best (rn1) verification
- Scaler-bridge audit

---

## Phase 41 — S19: Boundary Protocol Sweep

**Module:** `src/course_work/sweeps/boundary_protocol.py`

| Thành phần | Chi tiết |
|---|---|
| **Output** | `artifacts/sweeps/S19_boundary_protocol/`, `phase_41_signoff.json` |

**Nhiệm vụ:**
- Test boundary handling for windowing
- Validate no leakage at boundaries

---

## Phase 42 — Candidate Synthesis

**Script:** `src/course_work/scripts/p42_candidate_synthesis.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | Phase 23-41 sweep winners |
| **Output** | `artifacts/candidate_synthesis/{candidate_synthesis_manifest.json, candidate_synthesis_report.md, transformer_candidate_shortlist.json, selected_lineage.json, baseline_anchor_context.json, boundary_sensitivity_context.json}`, `phase_42_signoff.json` |

**Nhiệm vụ:**
- Tổng hợp các sweep winners
- Tạo candidate shortlist cho Transformer
- Phase handoff to LSTM tuning

---

## Phase 43 — LSTM Tuning

**Module:** `src/course_work/lstm_tuning/{stages, tuning_space, winners, ...}.py`

| Thành phần | Chi tiết |
|---|---|
| **Output** | `artifacts/lstm_tuning/{lstm_tuning_manifest.json, lstm_tuning_report.md, lstm_tuned_winner.json, lt{1,2,3,4,5}_*_winner.json}`, `phase_43_signoff.json` |
| **Output (figures)** | `artifacts/lstm_tuning/figures/` |
| **Scripts** | `src/course_work/scripts/p43_*.py` (6 scripts) |

**Nhiệm vụ:**
- Sweep LSTM hyperparameters (hidden_size, layers, dropout, lr, weight_decay)
- Sequential stages (lt1-lt5)
- Output LSTM tuned winner

---

## Phase 44 — Rolling Origin Robustness

**Module:** `src/course_work/rolling_origin/{folds, persistence, refit_engine, ...}.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | Phase 42 candidates + Phase 43 LSTM tuned |
| **Output** | `artifacts/rolling_origin/{rolling_origin_manifest.json, rolling_origin_summary.json, rolling_origin_report.md, rolling_origin_fold_manifest.json, rolling_origin_fold_local_scaling_contract.json, rolling_origin_recommended_transformer.json}`, `phase_44_signoff.json` |
| **Output (figures)** | `artifacts/rolling_origin/figures/` |
| **Scripts** | `src/course_work/scripts/p44_*.py` (3 scripts) |

**Nhiệm vụ:**
- Rolling-origin cross-validation
- Test generalization across time slices
- Robust Lane (full refit on each fold) vs Fast Lane (warm-start)
- Pick recommended Transformer candidate

---

## Phase 45 — Final Model Lock (NO-TRAIN)

**Module:** `src/course_work/final_model_lock/{candidate_lock, recipe, lineage, fingerprints, ...}.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | Phase 44 rolling origin results |
| **Output** | `artifacts/final_model_lock/{final_model_lock_manifest.json, final_model_lock_contract.json, final_model_lock_fingerprint.json, final_model_lock_summary.json, final_*_contract.json, final_*_fingerprint.json, final_*_evidence.json}`, `phase_45_signoff.json` |
| **Scripts** | `src/course_work/scripts/p45_*.py` (2 scripts) |

**Nhiệm vụ:**
- **NO-TRAIN phase** — chỉ lock config, KHÔNG train
- Freeze final model config: architecture, optimizer, loss, scaler, RevIN, seed, feature set, data region, environment
- Generate fingerprints for every contract
- Phase handoff to three-seed runs

---

## Phase 46 — Three-Seed Final Runs

**Script:** `src/course_work/scripts/p46_three_seed_runs.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | Phase 45 locked config |
| **Output** | `artifacts/three_seed_final_runs/{three_seed_manifest.json, three_seed_contract.json, three_seed_final_runs_summary.json, final_lock_verification.json, final_dev_population_manifest.json, phase47_final_test_evaluation_handoff.json}`, `phase_46_signoff.json` |
| **Output (checkpoints)** | `artifacts/three_seed_final_runs/official_checkpoints/seed_{42,123,2026}/*_FINAL_REFIT_metadata.json` |
| **Output (figures)** | `artifacts/three_seed_final_runs/figures/` |
| **Scripts** | `src/course_work/scripts/p46_*.py` (7 scripts: archive, preflight, gate, smoke, schema_sim, registration, three_seed) |

**Nhiệm vụ:**
- Train final model với 3 seeds: 42, 123, 2026
- Each seed produces a FINAL_REFIT checkpoint + metadata
- Verify against Phase 45 lock
- Phase handoff to final test evaluation

---

## Phase 47 — Final Test Evaluation (FINAL GATE, NO-TRAIN)

**Module:** `src/course_work/final_test_evaluation/{evaluation, checkpoint_loader, scaler_loader, ...}.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | Phase 46 official checkpoints |
| **Output** | `artifacts/final_test/{final_test_evaluation_contract.json, final_test_evaluation_manifest.json, final_test_release_verification.json, final_test_population_manifest.json, final_test_lstm_eligibility.json, final_test_access_event.json, final_test_access_log.jsonl, final_test_discrepancies.json, final_test_summary.json, final_test_report.md, prediction_checksums.json}`, `phase_47_signoff.json` |
| **Output (figures)** | `artifacts/final_test/figures/` |
| **Handoffs** | `phase48_prediction_analysis_handoff.json`, `phase49_residual_analysis_handoff.json`, `phase50_error_regime_handoff.json`, `phase51_worst_error_handoff.json`, `phase52_attention_extraction_handoff.json` |
| **Scripts** | `src/course_work/scripts/p47_*.py` (2 scripts) |

**Nhiệm vụ:**
- **FINAL GATE** — Đánh giá cuối cùng trên Test set
- Sử dụng checkpoints đã trained từ Phase 46
- Test access events audit (ghi lại mọi lần truy cập Test)
- Generate final test report với metrics cho cả 3 seeds
- Phase handoffs cho analysis phases 48-52

---

## Phase 48 — Prediction Analysis

**Module:** `src/course_work/analysis/prediction_analysis/{alignment, distribution, change_behavior, ...}.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | Phase 47 final test predictions |
| **Output** | `artifacts/prediction_analysis/{prediction_analysis_manifest.json, prediction_analysis_summary.json, prediction_analysis_report.md}`, `phase_48_signoff.json` |
| **Output (figures)** | `artifacts/prediction_analysis/figures/` |

**Nhiệm vụ:**
- Actual vs Predicted (full test)
- Zoom views (first/middle/last 24h)
- Per-seed scatter, ECDF
- Change magnitude distribution
- Cross-seed spread over time
- Lag cross-correlation

---

## Phase 49 — Residual Analysis

**Module:** `src/course_work/analysis/residual_analysis/{distributions, bias, autocorrelation, ...}.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | Phase 47 + 48 |
| **Output** | `artifacts/residual_analysis/{phase49_*.json, phase49_report.json}`, `phase_49_signoff.json` |
| **Output (figures)** | `artifacts/residual_analysis/figures/` |

**Nhiệm vụ:**
- Residual time series, distribution, ECDF
- Signed bias, sign balance
- Tail diagnostics
- ACF, sign runs, sign transitions
- Rolling statistics
- Magnitude associations (residual vs y_true, y_pred)
- Cross-seed agreement
- Persistence context

---

## Phase 50 — Error by Regime Analysis

**Module:** `src/course_work/analysis/error_regime_analysis/{regime_assignment, thresholds, cross_seed, ...}.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | Phase 47 + 49 |
| **Output** | `artifacts/error_by_regime/{phase50_findings.json, phase50_report.md, regime_*_fingerprint.json, regime_*_mapping.json, regime_thresholds_train_only.json, test_regime_assignment_fingerprint.json}`, `phase_50_signoff.json` |
| **Output (figures)** | `artifacts/error_by_regime/figures/` |

**Nhiệm vụ:**
- Define regimes: target level, time of day, day type, extreme high, change direction, change magnitude
- Compute regime thresholds (Train only)
- Assign regimes to Test samples
- Per-regime metrics (RMSE, MAE)
- Cross-seed regime stability

---

## Phase 51 — Worst Error Analysis

**Module:** `src/course_work/analysis/worst_error_analysis/{ranking, hardness, casebook, regime_context, ...}.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | Phase 47, 50 |
| **Output** | `artifacts/worst_error_analysis/{worst_error_analysis_manifest.json, worst_error_ranking_manifest.json, worst_case_casebook.md, phase51_findings.json, phase51_report.md, phase51_summary.json, exact_input_reconstruction_manifest.json, lstm_eligibility_context.json, selection_contract_fingerprint.json}`, `phase51_signoff.json` |
| **Output (figures)** | `artifacts/worst_error_analysis/figures/` |

**Nhiệm vụ:**
- Rank top-K worst errors per seed
- Identify shared worst cases across seeds
- Build casebook (markdown)
- Regime context for worst cases
- Exact input reconstruction (provenance)
- LSTM eligibility context (chỉ dùng LSTM nếu được phép)

---

## Phase 52 — Attention Extraction

**Module:** `src/course_work/analysis/attention_extraction/{extract, finalize, inputs, materialization, ...}.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | Phase 46 checkpoints + Phase 51 case IDs |
| **Output** | `artifacts/attention_extraction/{attention_extraction_manifest.json, attention_extraction_summary.json, attention_extraction_report.md, raw_attention_checksums.json, o52_inventory.json}`, `phase_52_signoff.json` |
| **Handoffs** | `phase53_attention_heatmaps_handoff.json`, `phase54_last_query_attention_handoff.json`, ... (5 handoffs) |

**Nhiệm vụ:**
- Extract attention weights for shared worst cases
- Per-seed × per-layer × per-head
- Checksums cho raw attention tensors
- Phase handoffs cho attention analyses 53-57

---

## Phase 53 — Attention Heatmaps

**Module:** `src/course_work/analysis/attention_heatmaps/{rendering, orientation, scales, catalog, ...}.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | Phase 52 raw attention |
| **Output** | `artifacts/attention_heatmaps/{attention_heatmaps_manifest.json, attention_heatmap_summary.json, attention_heatmap_report.md, attention_heatmap_catalog.md, attention_heatmap_render_config.json, attention_heatmap_image_checksums.json}`, `phase_53_signoff.json` |
| **Output (figures)** | `artifacts/attention_heatmaps/{case_grids, cross_seed, fixed_probability, individual_maps}/` |

**Nhiệm vụ:**
- Render attention heatmaps (V1: case grids, V2: cross-seed, V3: individual maps)
- Fixed probability mode + case-shared scale mode
- Cross-orientation audit
- Render config fingerprint

---

## Phase 54 — Last Query Attention

**Module:** `src/course_work/analysis/last_query_attention/{aggregations, profiles, top1_freq, coverage, ...}.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | Phase 52 |
| **Output** | `artifacts/last_query_attention/{last_query_attention_manifest.json, last_query_attention_summary.json, last_query_attention_report.md, o54_inventory.json}`, `phase_54_signoff.json` |
| **Output (figures)** | `artifacts/last_query_attention/figures/` |

**Nhiệm vụ:**
- Focus on attention từ last query token
- Top1 frequency, normalized entropy, expected lag
- Lag-bin coverage, recency mass
- Cumulative attention profiles

---

## Phase 55 — Head Comparison

**Module:** `src/course_work/analysis/head_comparison/{matrices, behavior_cards, profile_metrics, ...}.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | Phase 52 + 54 |
| **Output** | `artifacts/head_comparison/{head_comparison_manifest.json, head_comparison_summary.json, head_comparison_report.md}`, `phase_55_signoff.json` |
| **Output (figures)** | `artifacts/head_comparison/figures/` |

**Nhiệm vụ:**
- Paired head metrics (JSD, cosine, Wasserstein)
- Behavior cards cho mỗi head
- Layer-head diversity summary
- Mean temporal profiles by head

---

## Phase 56 — Error-Conditioned Attention

**Module:** `src/course_work/analysis/error_conditioned_attention/{analyses, cohort, core_metrics, ...}.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | Phase 51 case IDs + Phase 52 attention |
| **Output** | `artifacts/error_conditioned_attention/{error_conditioned_attention_manifest.json, error_conditioned_attention_summary.json, error_conditioned_attention_report.md, error_conditioning_assignment_fingerprint.json, shared_error_conditioning_audit.json}`, `phase_56_signoff.json` |
| **Output (figures)** | `artifacts/error_conditioned_attention/figures/` |

**Nhiệm vụ:**
- So sánh attention cho high-error vs low-error samples
- Cliffs Delta, signed error
- High vs low profile, decile trends
- Shared cohort analysis

---

## Phase 57 — Seed Stability Attention

**Module:** `src/course_work/analysis/seed_stability_attention/{analyses, core_metrics, ...}.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | Phase 52 |
| **Output** | `artifacts/seed_stability_attention/{seed_stability_attention_manifest.json, seed_stability_attention_summary.json, seed_stability_attention_report.md, head_matching_fingerprint.json}`, `phase_57_signoff.json` |
| **Output (figures)** | `artifacts/seed_stability_attention/figures/` |

**Nhiệm vụ:**
- Cross-seed attention stability
- Canonical head mapping
- Layer/head stability metrics
- Prediction spread vs attention disagreement

---

## Phase 58 — Final Tables

**Module:** `src/course_work/analysis/final_tables/{builders, writers, orchestrator, ...}.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | All analysis phases 48-57 |
| **Output** | `artifacts/final_tables/{final_tables_manifest.json, final_tables_summary.json, final_tables_report.md, final_table_inventory.json, final_table_checksums.json, final_table_render_config.json, final_figure_inventory.json}`, `phase_58_signoff.json` |
| **Output (tables)** | `artifacts/final_tables/tables/{latex,markdown,metadata}/` |

**Nhiệm vụ:**
- FT01-FT10: Final numerical tables
- FA01-FA12: Final appendix tables
- Render in LaTeX + Markdown formats
- Metadata + checksums

---

## Phase 59 — Final Conclusions

**Module:** `src/course_work/analysis/final_conclusions/{builders, findings, writers, ...}.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | Phase 58 + all upstream |
| **Output** | `artifacts/final_conclusions/{final_conclusions_manifest.json, FINAL_PROJECT_SUMMARY.md}, final_submission_conclusion_package/{final_*.md}`, `phase_59_signoff.json` |

**Nhiệm vụ:**
- Final abstract results summary
- Final conclusion section
- Final research question answers
- Final key takeaways
- Final limitations
- Final future work
- Final viva defense notes
- Final short conclusion

---

## Tổng Kết Phases 34-59

| Nhóm | Số phases | Đặc điểm |
|---|---|---|
| **Sweeps 2** (34-41) | 8 | S12-S19 hyperparameter sweeps |
| **Candidate & Lock** (42-47) | 6 | Candidate synthesis → LSTM tuning → rolling origin → model lock → 3-seed runs → test eval |
| **Analysis** (48-59) | 12 | Prediction → residual → error regime → worst error → attention (ext, heatmaps, last query, head comparison, error-conditioned, seed stability) → tables → conclusions |

---

## Cấu Trúc Phân Tầng Sau Refactor

Tất cả các analysis phases đều nằm trong `course_work.analysis.*`:
```
analysis/
├── prediction_analysis/         # 48
├── residual_analysis/           # 49
├── error_regime_analysis/       # 50
├── worst_error_analysis/        # 51
├── attention_extraction/        # 52
├── attention_heatmaps/          # 53
├── last_query_attention/        # 54
├── head_comparison/             # 55
├── error_conditioned_attention/ # 56
├── seed_stability_attention/    # 57
├── final_tables/                # 58
└── final_conclusions/           # 59
```

Final execution phases:
- `course_work.final_model_lock` (45)
- `course_work.final_test_evaluation` (47)

Các phase runner scripts trong `course_work/scripts/`:
```
src/course_work/scripts/
├── p39_*.py    (4 scripts)
├── p40_*.py    (3 scripts)
├── p42_p43_p44_p45_p46_p47_*.py
└── p47_*.py    (2 scripts)
```

---

**Phiên bản:** 06/09/2026 — sau lần refactor clean architecture lớn.
