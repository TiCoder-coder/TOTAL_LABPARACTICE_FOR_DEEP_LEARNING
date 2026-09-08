# Phase 22: Learning-Curve Diagnostics

## Mục đích

Phase 22 phân tích learning dynamics của LSTM B0 và Transformer B0 từ các training histories đã được lưu.

## Nguồn dữ liệu

- `RUN_LS_LS_0013_63C7E5ED` - LSTM Baseline Run (Phase 20)
- `RUN_TR_B0_0014_00EF3A31` - Transformer B0 Run (Phase 21)

## Outputs

Xem `artifacts/learning_diagnostics/` cho danh sách đầy đủ.

## Diagnostic Codes

- D0: HEALTHY_LEARNING
- D1: UNDERFITTING_LIKE
- D2: OVERFITTING_LIKE
- D3: PLATEAU
- D4: OPTIMIZATION_INSTABILITY
- D5: GRADIENT_STRESS
- D6: EARLY_BEST_PATHOLOGY
- D7: LATE_CONVERGENCE
- D8: METRIC_DIVERGENCE
- D9: RUNTIME_ANOMALY
- D10: PIPELINE_INTEGRITY_ANOMALY
- D11: INCONCLUSIVE

## Hypotheses

Xem `learning_curve_hypothesis_registry.csv` cho danh sách hypotheses được tạo từ diagnostic findings.

## Công việc chính

1. Load và verify training histories
2. Validate epoch integrity
3. Recompute best epochs
4. Build epoch summary tables
5. Compute diagnostic metrics
6. Generate visualizations
7. Classify findings
8. Create hypothesis registry
9. Generate diagnostic report
