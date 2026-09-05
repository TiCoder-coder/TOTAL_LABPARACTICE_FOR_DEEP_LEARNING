# LUỒNG HIỆN TẠI TỪ PHASE 1 ĐẾN PHASE 33

## 1. Phạm vi và nguồn sự thật

Tài liệu này mô tả luồng đã triển khai từ Phase 1 đến Phase 33 của coursework dự báo điện năng bằng Transformer Encoder. Nội dung được đối chiếu theo thứ tự ưu tiên sau:

1. `docs/RULE_BASE/architecture_rule.md` xác định owner, dependency và ranh giới notebook.
2. Phase detail tương ứng xác định mục tiêu khoa học.
3. Canonical sign-off, manifest, run config, metrics, winner và reference update xác định kết quả thực tế.
4. Processing log trong `docs/save_log_in_processing` phục vụ presentation, không thay thế bằng chứng canonical.
5. Notebook chỉ là lớp gọi public API và hiển thị kết quả.

Phase 0 không xuất hiện như một Phase riêng trong notebook nhưng vẫn khóa bài toán, target, horizon, option ID, metric, split protocol, research questions và Test firewall. Tài liệu này bắt đầu từ Phase 1 theo yêu cầu.

## 2. Bài toán và nguyên tắc xuyên suốt

```text
Dataset: UCI Appliances Energy Prediction
Target: Appliances
Đơn vị: Wh
Sampling interval: 10 phút
Horizon: 1 bước, tương đương 10 phút
Dạng mẫu: X[t-L+1:t] -> Appliances[t+1]
Mô hình chính: Transformer Encoder
Baselines: Persistence và LSTM
Metrics: MAE Wh, RMSE Wh và R²
Selection metric: Validation RMSE Wh nhỏ nhất
Split: chronological 70/15/15
Test: khóa đến Phase 47
```

Các nguyên tắc không thay đổi:

- Không shuffle trước chronological split.
- Không dùng Validation hoặc Test để fit scaler.
- Không dùng Test để chọn feature, lookback, kiến trúc hoặc hyperparameter.
- Metric chọn winner trong Phase 23–33 được tính trên toàn bộ Validation population và ở Wh gốc.
- Mỗi sweep chỉ thay đổi một yếu tố; winner Phase trước trở thành reference Phase sau.
- Exact-match run hợp lệ được tái sử dụng, không huấn luyện lại.
- Processing log chỉ phục vụ hiển thị; canonical artifact và registry mới quyết định trạng thái khoa học.
- Scientific execution thuộc source và terminal; notebook chỉ gọi reporting API.

## 3. Dependency flow hiện tại

```text
Phase 1 Environment
-> Phase 2 Data Acquisition
-> Phase 3 Schema Audit
-> Phase 4 Temporal Integrity Audit
-> Phase 5 Chronological Split
-> Phase 6 Train-only EDA
-> Phase 7 Deterministic Feature Engineering
-> Phase 8 Feature-Set Registry
-> Phase 9 Train-only Scaling
-> Phase 10 Window Population
-> Phase 11 Dataset và DataLoader
-> Phase 12 Shared Metrics
-> Phase 13 Experiment Registry
-> Phase 14 Persistence Baseline
-> Phase 15 LSTM Implementation
-> Phase 16 Transformer Implementation
-> Phase 17 Attention Verification
-> Phase 18 Forward Sanity
-> Phase 19 Training Engine
-> Phase 20 LSTM Baseline Run
-> Phase 21 Transformer B0 Run
-> Phase 22 Learning Diagnostics
-> Phase 23–33 Controlled Transformer Sweeps S1–S11
```

## 4. Phase 1 - Environment

### Làm gì

Xác minh môi trường có khả năng tái lập và chạy pipeline PyTorch: interpreter, kernel, dependency, dtype, device, seed và training smoke test.

### Làm như thế nào

Owner `src/course_work/utils/environment.py` thu thập inventory, kiểm tra kernel khớp interpreter, phát hiện CUDA/MPS/CPU, đóng băng dependency và chạy tensor allocation, forward, loss, autograd, finite-gradient và optimizer-step smoke tests.

### Kết quả

```text
Status: PASS
Artifact: ENV-v1
Python: 3.10.11
Kernel: python3, khớp interpreter
Selected device: mps
Default dtype: torch.float32
Deterministic mode: D0
Smoke test: PASS
```

Package lõi: PyTorch 2.13.0, NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, matplotlib 3.10.9, Jupyter 1.1.1 và ipykernel 7.3.0.

### Output

`artifacts/environment` chứa environment report, smoke report, dependency freeze, revision history và `phase_1_signoff.json`.

## 5. Phase 2 - Data Acquisition

### Làm gì

Xác minh dataset đúng nguồn UCI, raw CSV còn nguyên byte, archive hợp lệ, metadata đầy đủ và extraction path an toàn.

### Làm như thế nào

Owner `src/course_work/data/acquisition.py` kiểm tra DOI, provider, license, archive members, SHA-256, CSV header và row count. Notebook không tải hoặc sửa raw data; Phase chỉ xác minh acquisition offline.

### Kết quả

```text
Processing-log status: PASS
Artifact: DATA-v1
Dataset: Appliances Energy Prediction
Provider: UCI Machine Learning Repository
DOI: 10.24432/C5VC8G
License: CC BY 4.0
Rows: 19735
Reported features: 28
Sampling: 10 phút
Raw CSV SHA-256: 2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d
```

### Output và lưu ý hiện tại

Raw data/provenance nằm trong `data/raw_data`; audit/sign-off nằm trong `artifacts/acquisition`. Raw CSV vẫn đúng checksum. Full regression gần nhất phát hiện `README_SOURCE.md` và `dataset_manifest.json` không còn khớp checksum đã ký; đây là drift tài liệu/manifest cần recovery theo revision, không phải raw dataset bị đổi.

## 6. Phase 3 - Schema Audit

### Làm gì

Kiểm tra shape, thứ tự cột, dtype, semantic role, missing value, constant column, finite value và khả năng parse timestamp.

### Làm như thế nào

Owner `src/course_work/data/schema.py` đối chiếu 29 cột thực tế với schema, phân biệt timestamp, target, feature thường và hai random controls `rv1`, `rv2`; numeric coercion chỉ để audit, không mutate raw DataFrame.

### Kết quả

```text
Status: PASS_WITH_WARNING
Artifact: SCHEMA-v1
Rows: 19735
Columns: 29
Target: Appliances
Timestamp: date
Missing/unexpected/duplicate columns: 0
Constant/all-null/non-finite columns: 0
Timestamp parse failures: 0
```

Warning `SD-001` giải thích UCI báo 28 feature trong khi raw CSV có 29 cột tổng cộng gồm timestamp, target, 25 feature thường và hai random controls. Warning đã resolved, không có cột nào bị xóa.

### Output

`artifacts/schema` chứa summary, variable dictionary, comparison, fingerprint, discrepancy record, manifest và sign-off.

## 7. Phase 4 - Temporal Integrity Audit

### Làm gì

Xác minh timestamp đủ an toàn để split và window: parse strict, monotonic, không trùng, đúng grid 10 phút và không mất mẫu.

### Làm như thế nào

Owner `src/course_work/data/temporal.py` tính timestamp delta, phân loại gap/duplicate/grid mismatch và tạo `continuity_segment_id`. Một window chỉ hợp lệ nếu input và target nằm trong cùng continuity segment.

### Kết quả

```text
Status: PASS
Artifact: TEMPORAL-v1
Rows: 19735
Start: 2016-01-11 17:00:00
End: 2016-05-27 18:00:00
Cadence: 10 phút
Parse success: 100%
Coverage: 100%
Duplicates/missing/gaps: 0
Continuity segments: 1
```

### Output

`artifacts/temporal` chứa integrity audits, continuity metadata, manifest và `phase_4_signoff.json`.

## 8. Phase 5 - Chronological Split

### Làm gì

Chia dữ liệu theo thời gian trước EDA và mọi quyết định data-dependent.

### Làm như thế nào

Owner `src/course_work/data/splitting.py` dùng tỷ lệ 70/15/15, floor boundary, không shuffle. Mỗi row nhận `split_id`, `split_position` và fingerprint. WB0 context carry-over là protocol chính; WB1 strict isolation là phương án thay thế. Test chỉ được audit cấu trúc.

### Kết quả

| Split | Rows | Tỷ lệ | Bắt đầu | Kết thúc |
|---|---:|---:|---|---|
| Train | 13,814 | 70% | 2016-01-11 17:00 | 2016-04-16 15:10 |
| Validation | 2,960 | 15% | 2016-04-16 15:20 | 2016-05-07 04:30 |
| Test | 2,961 | 15% | 2016-05-07 04:40 | 2016-05-27 18:00 |

```text
Status: PASS
Artifact: SPLIT-v1
Primary boundary: WB0_CONTEXT_CARRY_OVER
Alternative boundary: WB1_STRICT_ISOLATION
Test distribution: LOCKED_UNTIL_PHASE_47
```

### Output

`artifacts/splits` chứa membership CSV, summaries, boundary neighborhoods, fingerprints, audits, timeline, manifest và sign-off.

## 9. Phase 6 - Exploratory Data Analysis

### Làm gì

Mô tả Train data để hiểu target, features, seasonality, correlation, redundancy, zero inflation và giá trị bất thường trước sweep.

### Làm như thế nào

Owner khoa học là `src/course_work/data/eda.py`; `src/course_work/reporting/eda.py` tạo bảng/hình. Chỉ 13,814 Train rows được phân tích. Mọi phát hiện chỉ là hypothesis; Phase không xóa outlier, impute, interpolate, chọn feature cuối hoặc tune model. Notebook có ngoại lệ đã duyệt để hiển thị descriptive EDA nhưng không sở hữu artifact.

### Kết quả

```text
Status: PASS
Artifact: EDA-v1
Train rows: 13814
Tables: 16
Figures: 16
Hypotheses: 10
Anomalies: 3
Rows removed: 0
Imputation/interpolation/feature selection/model tuning: false
```

Ghi nhận chính: 726 Train observations từ percentile 95 của `Appliances` trở lên; 22 feature pairs đạt ngưỡng correlation 0.8; 10,123 observations có `lights=0`. Không ghi nhận nào tự động thay đổi data.

### Output

`artifacts/eda` chứa tables, figures, hypotheses/anomalies, scope audit, manifest và `phase_6_signoff.json`.

## 10. Phase 7 - Feature Engineering

### Làm gì

Tạo năm deterministic time features từ timestamp.

### Làm như thế nào

Owner `src/course_work/data/features.py` giữ raw values, target, timestamp, lineage, split và continuity; chỉ thêm:

```text
hour_sin
hour_cos
dow_sin
dow_cos
weekend
```

Không có parameter thống kê nào được fit. Vì công thức chỉ phụ thuộc timestamp, feature view có thể materialize đủ 19,735 rows mà không leakage; mọi quyết định data-dependent trước đó vẫn chỉ dùng Train.

### Kết quả

```text
Status: PASS
Artifact: FEATURES-v1
Rows: 19735
Columns: 37
Raw model features: 27
Engineered features: 5
Metadata columns: 4
Row/target/timestamp/raw features preserved: true
New missing values: 0
Leakage audit: PASS
```

### Output

Feature view nằm trong `data/interim/uci_appliances_energy_prediction`; registry, lineage, audits, manifest và sign-off nằm trong `artifacts/features`.

## 11. Phase 8 - Feature-Set Variants

### Làm gì

Khóa ordered feature variants để training không tự tạo danh sách cột tùy ý.

### Làm như thế nào

Owner `src/course_work/data/feature_sets.py` kết hợp ba base groups FS0/FS1/FS2 với TF0/TF1. Mỗi variant có ordered columns, count, semantic audit và fingerprint; sáu variants cùng tham chiếu FEATURES-v1, không sao chép data.

### Kết quả

| Variant | Số feature | Vai trò |
|---|---:|---|
| FS0_TF0 | 25 | FS0 không time features |
| FS0_TF1 | 30 | FS0 cộng time features |
| FS1_TF0 | 26 | Baseline sensors không time features |
| FS1_TF1 | 31 | Baseline ban đầu |
| FS2_TF0 | 28 | Extended set không time features |
| FS2_TF1 | 33 | Extended set cộng time features |

```text
Status: PASS
Artifact: FEATURESETS-v1
Variants: 6
Baseline ban đầu: FS1_TF1
Order/semantic/leakage audits: PASS
```

### Output

`artifacts/feature_sets` chứa component registry, variants, fingerprints, lineage, specification, audits, manifest và sign-off.

## 12. Phase 9 - Train-Only Scaling

### Làm gì

Tạo scaler cho từng feature variant và hai lựa chọn target scaling, chỉ fit bằng Train.

### Làm như thế nào

Owner `src/course_work/data/scaling.py` fit sáu X-scaler bundles trên 13,814 Train rows. Numeric sensors dùng StandardScaler; time features được pass-through. Target có YS0 identity và YS1 StandardScaler. YS1 bắt buộc inverse-transform trước metric Wh.

### Kết quả

```text
Status: PASS
Artifact: SCALING-v1
X bundles: 6
X/Y fit split: TRAIN
Train fit rows: 13814
Target options: YS0, YS1
Leakage audit: PASS
Test inspection: STRUCTURAL_ONLY
```

### Output

`artifacts/scalers` chứa serialized bundles; `artifacts/scaling` chứa registry, statistics, audits, manifest và sign-off.

## 13. Phase 10 - Window Builder

### Làm gì

Định nghĩa sequence-to-one windows cho lookback 36, 72, 144 và horizon một bước.

### Làm như thế nào

Owner `src/course_work/data/windows.py` tạo lazy window index với input range, target index/timestamp/split và boundary validity. WB0 cho Validation/Test dùng lịch sử trước boundary nhưng target phải đúng split. Common population neo theo L144 để mọi comparison dùng cùng target rows.

### Kết quả

```text
Status: PASS
Artifact: WINDOWS-v1
Lookbacks: L36, L72, L144
Horizon: 1 step, 10 phút
Boundary: WB0_CONTEXT_CARRY_OVER
Train targets: 13670
Validation targets: 2960
Test targets: 2961
Total common targets: 19591
Test targets: LOCKED_UNTIL_PHASE_47
```

Train mất 144 dòng đầu để tạo common population; Validation/Test vẫn đủ target nhờ context carry-over.

### Output

`artifacts/windows` chứa index, population, rejected candidates, audits, fingerprints, manifest và sign-off.

## 14. Phase 11 - DataLoaders

### Làm gì

Chuyển window metadata thành PyTorch Dataset/DataLoader dùng chung cho LSTM và Transformer.

### Làm như thế nào

Owner `src/course_work/data/datasets.py` triển khai lazy `SequenceWindowDataset`. Train shuffle bằng split-specific seeded generator; Validation/Test giữ chronology; mọi loader `drop_last=False`. Test dataset không cung cấp target khi chưa có authorization Phase 47.

### Kết quả

```text
Status: PASS
Artifact: DATALOADERS-v1
Train/Validation/Test samples: 13670/2960/2961
Supported batch sizes: 32, 64
Baseline batch: 64
Workers: 0
Train shuffle: true
Validation/Test shuffle: false
Test targets: locked
```

### Output

`artifacts/dataloaders` chứa registries, coverage/order/shuffle/worker/device/Test-firewall audits, manifest và sign-off.

## 15. Phase 12 - Shared Metrics

### Làm gì

Khóa một cách tính metric duy nhất cho mọi baseline và sweep.

### Làm như thế nào

Owner `src/course_work/evaluation/metrics.py` căn sample ID theo full split population, inverse-transform YS1 về Wh rồi tính global MAE, RMSE, R². Mean batch RMSE/R² bị cấm. Residual được định nghĩa là actual trừ prediction.

### Kết quả

| Metric | Field | Đơn vị | Hướng |
|---|---|---|---|
| MAE | `mae_wh` | Wh | Thấp hơn tốt hơn |
| RMSE | `rmse_wh` | Wh | Metric chọn winner |
| R² | `r2` | Không đơn vị | Cao hơn tốt hơn |

```text
Status: PASS
Artifact: METRICS-v1
Selection split: VALIDATION
Validation population: 2960
Test metrics/targets: không tính, không materialize
```

### Output

`artifacts/metrics` chứa registry, schemas, reference/implementation/comparison/Test-firewall audits, manifest và sign-off.

## 16. Phase 13 - Experiment Registry

### Làm gì

Quản lý vòng đời experiment và liên kết config, run, checkpoint, predictions, metrics và sweeps.

### Làm như thế nào

Owner `src/course_work/experiments/registry.py` tạo unique run ID, canonical config fingerprint, lifecycle status và artifact checksum. Completed config không được sửa. JSONL là nguồn chính; CSV là audit views. Test metric bị chặn trước final gate.

### Kết quả

```text
Status: PASS
Artifact: EXPERIMENTS-v1
Production runs: 19
Completed runs: 19
Failed runs: 0
Experiment families: 26
Registry validation: 13/13 PASS
```

19 runs gồm persistence, hai learned baselines và các condition mới của S1–S11; reused references không bị nhân bản.

### Output

`artifacts/experiments` chứa JSONL registry, CSV views, audits, index, manifest và sign-off; run vật lý nằm trong `artifacts/runs/<RUN_ID>`.

## 17. Phase 14 - Persistence Baseline

### Làm gì

Tạo baseline không học để mọi model có mốc so sánh time-series hợp lý.

### Làm như thế nào

Owner `src/course_work/baselines/persistence.py` dùng `y_hat(t+1)=y(t)`, căn prediction với 2,960 Validation targets rồi đánh giá bằng METRICS-v1. Không optimizer, không training và không trainable parameters.

### Kết quả

```text
Status: PASS
Run: RUN_PS_PS_0001_AFFD3E3F
Validation samples: 2960
MAE: 26.162162 Wh
RMSE: 66.429703 Wh
R²: 0.481326
Test: LOCKED_UNTIL_PHASE_47
```

### Output

`artifacts/baselines/persistence` chứa predictions, metrics, summary, audit, manifest, sign-off; run config/status nằm trong `artifacts/runs`.

## 18. Phase 15 - LSTM Implementation

### Làm gì

Triển khai LSTM baseline ở cấp model contract, chưa thực hiện canonical baseline training.

### Làm như thế nào

Owner `src/course_work/models/lstm_regressor.py` nhận `[B,L,F]`, chạy LSTM batch-first, lấy representation ở bước cuối và dùng linear head trả `[B,1]`. Model unidirectional, stateless giữa các forward, zero-init hidden state và chỉ dùng inter-layer dropout khi số layer lớn hơn một.

### Kết quả

```text
Status: PASS
Implementation: LSTM_IMPL-v1
Reference input size: 31
Hidden size: 64
Layers: 2
Dropout: 0.1
Pooling: LAST_STEP
Trainable parameters: 58177
Supported lookbacks: 36, 72, 144
Unit tests: 13 PASS
Serialization: PASS
```

### Output

`artifacts/models/lstm` chứa contract, shape/parameter audits, manifest và sign-off.

## 19. Phase 16 - Transformer Implementation

### Làm gì

Triển khai Transformer Encoder cho sequence-to-one regression và giữ khả năng trích xuất attention theo layer/head.

### Làm như thế nào

Owner chính `src/course_work/models/transformer_regressor.py`; positional encoding và encoder layer có module riêng. Input `[B,L,F]` được project sang `d_model`, cộng sinusoidal positional encoding, đi qua encoder layers, pooling và linear head. `forward()` dùng cho training; `forward_with_attention()` dùng cho inspection. Không dùng causal mask vì toàn bộ input window đều đứng trước target.

### Kết quả

```text
Status: PASS
Implementation: TRANSFORMER_IMPL-v1
Reference input size: 31
d_model: 64
Heads: 4
Layers: 2
FFN width: 128
Dropout: 0.1
Pooling: LAST_STEP
Positional encoding: SINUSOIDAL
Norm order: POST_NORM
Attention layout: [B,H,L,L]
Reference trainable parameters: 69057
Unit tests: PASS
```

Activation là thành phần cấu hình; các canonical training runs từ B0 trở đi dùng GELU.

### Output

`artifacts/models/transformer` chứa model contract, tensor/parameter/positional audits, manifest và sign-off.

## 20. Phase 17 - Attention-Aware Encoder Verification

### Làm gì

Xác minh attention weights có đúng shape/xác suất và instrumentation không làm thay đổi prediction.

### Làm như thế nào

Owner `src/course_work/attention/verification.py` kiểm tra từng layer/head, batch independence, finite values, non-negative weights và tổng theo key dimension bằng một. Prediction từ `forward()` được so với `forward_with_attention()`.

### Kết quả

```text
Status: PASS
Artifact: ATTENTION_VERIFY-v1
Reference lookback: 144
Layout: B_H_L_L
Probability: softmax over keys
Per-head weights: preserved
Causal mask: none
Verification tests: 7 PASS
```

### Output

`artifacts/attention_verification` chứa attention contract, mapping/audit tables, manifest và sign-off.

## 21. Phase 18 - Forward-Pass Sanity Tests

### Làm gì

Kiểm tra end-to-end tensor path từ DataLoader qua LSTM/Transformer tới prediction trước khi cho Training Engine chạy.

### Làm như thế nào

Owner `src/course_work/sanity/forward_sanity.py` lấy batch theo contract, chạy LSTM train-mode forward, Transformer Validation forward và attention smoke check. Phase chỉ forward, không optimizer step; nó kiểm tra batch schema, output `[B,1]`, finite values và attention layer count.

### Kết quả

```text
Status: PASS
Artifact: FORWARD_SANITY-v1
Feature count: 31
Sanity tests: 3 PASS
Finite outputs: PASS
No optimizer step: PASS
Approved for Phase 19: true
```

Canonical recovery manifest ghi `device_type=cpu`; các sweep khoa học gần nhất chạy bằng MPS. Device identity được giữ theo lineage, không âm thầm thay thế.

### Output

`artifacts/forward_sanity` chứa sanity matrix, audits, training-engine handoff, manifest và sign-off.

## 22. Phase 19 - Baseline Training Engine

### Làm gì

Tạo một engine chung để LSTM và Transformer được train/evaluate/checkpoint theo cùng protocol.

### Làm như thế nào

Owners `src/course_work/training/engine.py` và `engine_materialize.py` quản lý train epoch, full Validation evaluation, MSE, AdamW, gradient clipping, early stopping, best checkpoint, history, predictions, registry lifecycle và heartbeat. Best model luôn theo Validation RMSE Wh nhỏ nhất.

### Kết quả

```text
Status: PASS
Artifact: TRAINING_ENGINE-v1
Optimizer: AdamW
Loss: MSE
Selection split: VALIDATION
Selection metric: rmse_wh
Early-stopping mode: MIN
Checkpoint policy: BEST_VALIDATION_RMSE_WH
Unit tests: 4 PASS
```

### Output

`artifacts/training_engine` chứa engine contract, smoke tests, manifest và sign-off. Learned outputs nằm trong `artifacts/runs/<RUN_ID>`.

## 23. Phase 20 - LSTM Baseline Run

### Làm gì

Huấn luyện canonical LSTM baseline để so sánh công bằng với Transformer.

### Làm như thế nào

Owner `src/course_work/baselines/lstm_baseline.py` dùng FS1_TF1, YS1, L144, batch 64, seed 42, learning rate `3e-4`, weight decay `1e-4`, MSE, max 50 epochs, patience 10 và gradient clipping 1.0. Model có hidden size 64, hai layers, dropout 0.1, LAST_STEP. Prediction được inverse-transform về Wh.

### Kết quả canonical hiện tại

```text
Status: PASS
Run: RUN_LS_LS_0002_22A25637
Best epoch: 6
Stopped reason: EARLY_STOPPING
Validation MAE: 27.528816 Wh
Validation RMSE: 60.446456 Wh
Validation R²: 0.570551
Persistence RMSE: 66.429703 Wh
```

LSTM giảm RMSE 5.983246 Wh so với Persistence. Không có Test metric.

### Output

`artifacts/lstm_baseline` chứa summary/history/metrics/comparison/sign-off; checkpoint, config, status và predictions nằm trong `artifacts/runs/RUN_LS_LS_0002_22A25637`.

## 24. Phase 21 - Transformer B0 Run

### Làm gì

Huấn luyện Transformer B0 để tạo reference ban đầu cho chuỗi sweep.

### Làm như thế nào

Owner `src/course_work/baselines/transformer_b0.py` dùng FS1_TF1, YS1, L144, LAST_STEP, GELU, d_model 64, 4 heads, 2 layers, FFN 128, dropout 0.1, batch 64, LR `3e-4`, WD `1e-4`, seed 42, MSE, early stopping và gradient clipping 1.0.

### Kết quả canonical hiện tại

```text
Status: PASS
Run: RUN_TR_B0_0003_44457570
Best epoch: 12
Stopped reason: EARLY_STOPPING
Validation MAE: 29.262829 Wh
Validation RMSE: 61.030760 Wh
Validation R²: 0.562208
```

Transformer B0 tốt hơn Persistence nhưng kém LSTM B0 khoảng 0.584303 Wh RMSE, do đó Phase 23–33 tiếp tục tối ưu Transformer trên Validation.

### Output

`artifacts/transformer_b0` chứa comparisons, summary, history linkage và sign-off; run nằm trong `artifacts/runs/RUN_TR_B0_0003_44457570`.

## 25. Phase 22 - Learning-Curve Diagnostics

### Làm gì

Đọc histories của LSTM B0 và Transformer B0 để nhận diện overfitting, underfitting hoặc instability trước sweep.

### Làm như thế nào

Owners `src/course_work/diagnostics/learning_diagnostics.py` và `learning_curves.py` dùng canonical history/best metrics, không train lại. Findings được tạo theo quy tắc xác định và map tới phase tương lai.

### Kết quả canonical hiện tại

```text
Status: PASS
Artifact: LEARN-DIAG-v1
Models: lstm_b0, transformer_b0
Findings: 2
LSTM best epoch/RMSE: 6 / 60.446456 Wh
Transformer best epoch/RMSE: 12 / 61.030760 Wh
```

LSTM có D2 mức MODERATE: Validation RMSE xấu đi sau best epoch trong khi train loss tiếp tục giảm, phù hợp overfitting-like pattern. Transformer có D0 mức INFO: learning pattern khỏe.

### Output

`artifacts/learning_diagnostics` chứa summary CSV, manifest, figures và sign-off.

### Lưu ý presentation

Các con số canonical Phase 20–22 ở trên là kết quả sau recovery. Một số processing log/cached notebook output Phase 15–22 vẫn chứa checksum hoặc run ID trước recovery và cần được refresh riêng; chúng không được dùng thay canonical artifacts.

## 26. Protocol chung của Phase 23–33

Mỗi Phase là một controlled one-factor-at-a-time sweep:

1. Validate sign-off, winner và reference update của Phase liền trước.
2. Đọc experiment registry và exact run config.
3. Reuse reference khi config, checkpoint, history, metric và checksum đều khớp.
4. Chỉ train condition còn thiếu trên terminal.
5. Cấm Test metric và Test target access.
6. Chọn winner bằng full-precision Validation RMSE Wh nhỏ nhất.
7. Chỉ dùng tie rule khi RMSE bằng nhau tuyệt đối.
8. Ghi results CSV, manifest, winner, reference update và sign-off.
9. Dùng winner làm reference cho Phase tiếp theo.

`VALID_REUSABLE` là trạng thái thành công: canonical evidence đầy đủ và lần chạy sau chỉ `RENDER_ONLY`.

## 27. Phase 23 - S1 Feature-Set Sweep

### Làm gì và làm như thế nào

So sánh FS0_TF1, FS1_TF1 và FS2_TF1. Transformer B0 FS1_TF1 được reuse; hai condition còn lại train mới. Các thành phần model/training khác giữ theo B0. Tie rule ưu tiên ít input features hơn khi RMSE bằng tuyệt đối.

### Kết quả

| Condition | Run | Validation MAE Wh | Validation RMSE Wh | Validation R² | Quyết định |
|---|---|---:|---:|---:|---|
| FS0_TF1 | RUN_TR_S01_0004_C8A249B8 | 46.756144 | 86.189688 | 0.126867 | Không chọn |
| FS1_TF1 | RUN_TR_B0_0003_44457570 | 29.262829 | 61.030760 | 0.562208 | Reference |
| FS2_TF1 | RUN_TR_S01_0005_57E974CF | 27.525244 | 59.432242 | 0.584841 | Winner |

Winner FS2_TF1 có 33 features. Sign-off PASS, Test FORBIDDEN, approved for Phase 24.

### Output

`artifacts/sweeps/s1_feature_set` chứa results, manifest, winner, reference update và sign-off.

## 28. Phase 24 - S2 Time-Feature Sweep

### Làm gì và làm như thế nào

Kiểm tra năm time features có giúp FS2 không. TF0 dùng FS2_TF0; TF1 dùng FS2_TF1 và reuse Phase 23 winner. Tie rule ưu tiên TF0 nếu exact tie.

### Kết quả

| Condition | Feature variant | Validation RMSE Wh | Quyết định |
|---|---|---:|---|
| TF0 | FS2_TF0 | 60.239117 | Không chọn |
| TF1 | FS2_TF1 | 59.432242 | Winner |

Time features được giữ; winner/reference vẫn là `RUN_TR_S01_0005_57E974CF`. Test FORBIDDEN, approved for Phase 25.

### Output

`artifacts/sweeps/s2_time_feature` chứa canonical sweep artifacts.

## 29. Phase 25 - S3 Target-Scaling Sweep

### Làm gì và làm như thế nào

So sánh raw target YS0 với Train-only StandardScaler YS1. YS1 reference được reuse. Mọi YS1 prediction phải inverse-transform về Wh trước metric. Tie rule ưu tiên YS0.

### Kết quả

| Condition | Validation MAE Wh | Validation RMSE Wh | Validation R² | Quyết định |
|---|---:|---:|---:|---|
| YS0 | 26.038453 | 61.880141 | 0.549938 | Không chọn |
| YS1 | 27.525244 | 59.432242 | 0.584841 | Winner |

YS1 thắng vì selection metric là RMSE, không phải MAE. Test FORBIDDEN, approved for Phase 26.

### Output

`artifacts/sweeps/s3_target_scaling` chứa canonical sweep artifacts.

## 30. Phase 26 - S4 Lookback Sweep

### Làm gì và làm như thế nào

So sánh L36, L72, L144 trên cùng common target population. L144 reuse Phase 25 reference; L36/L72 train mới. Tie rule ưu tiên lookback ngắn hơn.

### Kết quả

| Condition | Thời lượng | Validation RMSE Wh | Quyết định |
|---|---:|---:|---|
| L36 | 6 giờ | 59.032873 | Winner |
| L72 | 12 giờ | 59.183522 | Không chọn |
| L144 | 24 giờ | 59.432242 | Reference cũ |

L36 cho kết quả tốt nhất; reference mới là `RUN_TR_S04_0008_133EF4E3`. Test FORBIDDEN, approved for Phase 27.

### Output

`artifacts/sweeps/s4_lookback` chứa canonical sweep artifacts.

## 31. Phase 27 - S5 Pooling Sweep

### Làm gì và làm như thế nào

So sánh LAST_STEP với MEAN pooling. LAST_STEP reuse Phase 26 winner; MEAN train mới. Tie rule ưu tiên LAST_STEP.

### Kết quả

| Condition | Validation MAE Wh | Validation RMSE Wh | Quyết định |
|---|---:|---:|---|
| LAST_STEP | 27.418594 | 59.032873 | Winner |
| MEAN | 27.412043 | 59.675502 | Không chọn |

MEAN có MAE thấp hơn rất nhẹ nhưng LAST_STEP thắng theo RMSE. Test FORBIDDEN, approved for Phase 28.

### Output

`artifacts/sweeps/s5_pooling` chứa canonical sweep artifacts.

## 32. Phase 28 - S6 Activation Sweep

### Làm gì và làm như thế nào

So sánh ReLU và GELU trong Transformer FFN. GELU reuse reference; ReLU train mới. Tie rule ưu tiên ReLU.

### Kết quả

| Condition | Validation RMSE Wh | Quyết định |
|---|---:|---|
| RELU | 59.185933 | Không chọn |
| GELU | 59.032873 | Winner |

GELU được giữ; winner run vẫn `RUN_TR_S04_0008_133EF4E3`. Test FORBIDDEN, approved for Phase 29.

### Output

`artifacts/sweeps/s6_activation` chứa canonical sweep artifacts.

## 33. Phase 29 - S7 Batch-Size Sweep

### Làm gì và làm như thế nào

So sánh B32 và B64. B64 reuse Phase 28 reference; B32 train mới. Population, seed, optimizer, model và early-stopping protocol giữ nguyên. Tie rule ưu tiên B64.

### Kết quả

| Condition | Validation MAE Wh | Validation RMSE Wh | Validation R² | Quyết định |
|---|---:|---:|---:|---|
| B32 | 27.594696 | 58.083927 | 0.603465 | Winner |
| B64 | 27.418594 | 59.032873 | 0.590402 | Reference cũ |

B32 cải thiện khoảng 0.948946 Wh; reference mới là `RUN_TR_S07_0012_D99B2B4F`. Test FORBIDDEN, approved for Phase 30.

### Output

`artifacts/sweeps/s7_batch_size` chứa canonical sweep artifacts.

## 34. Phase 30 - S8 Learning-Rate Sweep

### Làm gì và làm như thế nào

So sánh AdamW LR1=`1e-4`, LR2=`3e-4`, LR3=`1e-3`. LR2 reuse Phase 29 winner; LR1/LR3 train mới. Tie rule ưu tiên learning rate thấp hơn.

### Kết quả

| Condition | Learning rate | Validation MAE Wh | Validation RMSE Wh | Quyết định |
|---|---:|---:|---:|---|
| LR1 | 0.0001 | 27.354409 | 58.517648 | Không chọn |
| LR2 | 0.0003 | 27.594696 | 58.083927 | Winner |
| LR3 | 0.0010 | 25.729457 | 58.409680 | Không chọn |

LR3 có MAE thấp nhất nhưng LR2 thắng theo RMSE. Reference vẫn `RUN_TR_S07_0012_D99B2B4F`. Test FORBIDDEN, approved for Phase 31.

### Output

`artifacts/sweeps/s8_learning_rate` chứa canonical sweep artifacts.

## 35. Phase 31 - S9 Weight-Decay Sweep

### Làm gì và làm như thế nào

So sánh AdamW WD0=0, WD1=`1e-4`, WD2=`1e-3`. WD1 reuse Phase 30 reference; WD0/WD2 train mới. Không thêm explicit L2 loss và không đổi optimizer groups. Tie rule ưu tiên weight decay thấp hơn.

### Kết quả

| Condition | Weight decay | Validation MAE Wh | Validation RMSE Wh | Quyết định |
|---|---:|---:|---:|---|
| WD0 | 0 | 27.594705 | 58.084723 | Không chọn |
| WD1 | 0.0001 | 27.594696 | 58.083927 | Reference cũ |
| WD2 | 0.001 | 27.595002 | 58.081901 | Winner |

Chênh lệch rất nhỏ nhưng được chọn bằng full precision. Winner `RUN_TR_S09_0016_AE0FB819`. Canonical sign-off PASS; selective state `VALID_REUSABLE`; Test FORBIDDEN; approved for Phase 32.

### Output

Canonical outputs trong `artifacts/sweeps/S9_weight_decay`; processing log `phase_31_s9_weight_decay_log.json`.

## 36. Phase 32 - S10 Dropout Sweep

### Làm gì và làm như thế nào

So sánh DR01=0.1, DR02=0.2, DR03=0.3 tại toàn bộ controlled dropout sites trong hai encoder layers. DR01 reuse Phase 31 winner; DR02/DR03 train mới. Train behavior phải stochastic, eval deterministic, MC Dropout bị cấm. Tie rule ưu tiên dropout thấp hơn.

### Kết quả

| Condition | Dropout | Validation MAE Wh | Validation RMSE Wh | Quyết định |
|---|---:|---:|---:|---|
| DR01 | 0.1 | 27.595002 | 58.081901 | Winner |
| DR02 | 0.2 | 27.554862 | 58.776962 | Không chọn |
| DR03 | 0.3 | 26.183263 | 58.360214 | Không chọn |

DR03 có MAE thấp hơn nhưng DR01 thắng theo RMSE. Winner/reference `RUN_TR_S09_0016_AE0FB819`. Canonical sign-off PASS; selective state `VALID_REUSABLE`; Test FORBIDDEN; approved for Phase 33.

### Output

Canonical outputs trong `artifacts/sweeps/S10_dropout`; processing log `phase_32_s10_dropout_log.json`.

## 37. Phase 33 - S11 d_model Sweep

### Làm gì và làm như thế nào

So sánh D32 và D64, giữ cố định 4 heads, 2 layers, FFN 128. D64 reuse exact Phase 32 reference; chỉ D32 train mới trên terminal. D32 có head dimension 8 và FFN ratio 4; D64 có head dimension 16 và FFN ratio 2. Không đổi heads/layers/FFN để bù width. Tie rule ưu tiên D32.

### Kết quả

| Condition | d_model | Trainable parameters | Validation MAE Wh | Validation RMSE Wh | Quyết định |
|---|---:|---:|---:|---:|---|
| D32 | 32 | 26,529 | 25.700495 | 58.457258 | Không chọn |
| D64 | 64 | 69,185 | 27.595002 | 58.081901 | Winner |

D32 run `RUN_TR_S11_0019_193AC913` chạy MPS, best epoch 18 và early stop sau epoch 28. D64 thắng và giữ reference `RUN_TR_S09_0016_AE0FB819`. D64 thêm 42,656 parameters, tăng 160.79% so D32; capacity chỉ là context, không thay RMSE selection.

```text
Canonical sign-off: PASS
Selective state: VALID_REUSABLE
Next action: RENDER_ONLY
Test metric count: 0
Approved for Phase 34: true
```

### Output

`artifacts/sweeps/S11_d_model` chứa metrics, manifest, winner, reference update, sign-off và live evidence; processing log là `phase_33_s11_d_model_log.json`.

## 38. Cấu hình Transformer sau Phase 33

| Thành phần | Giá trị hiện tại | Phase quyết định |
|---|---|---:|
| Feature set | FS2_TF1, 33 features | 23–24 |
| Target scaling | YS1 Train-only StandardScaler | 25 |
| Lookback | L36, 6 giờ | 26 |
| Pooling | LAST_STEP | 27 |
| Activation | GELU | 28 |
| Batch size | 32 | 29 |
| Learning rate | 0.0003 | 30 |
| AdamW weight decay | 0.001 | 31 |
| Dropout | 0.1 | 32 |
| d_model | 64 | 33 |
| Heads | 4 | Frozen, chưa sweep |
| Layers | 2 | Frozen, chưa sweep |
| FFN width | 128 | Frozen, chưa sweep |
| Loss | MSE | Frozen, chưa sweep |
| Max epochs | 50 | Frozen, chưa sweep |
| Early-stopping patience | 10 | Training contract |
| Gradient clipping | 1.0 | Frozen, chưa sweep |
| RevIN | Disabled | Chưa sweep |
| Boundary | WB0 context carry-over | Chưa kiểm tra S19 |

Current reference là `RUN_TR_S09_0016_AE0FB819`, Validation RMSE `58.08190056355405 Wh`.

## 39. So sánh hiệu năng hiện tại

| Mốc | Validation RMSE Wh | Current reference cải thiện |
|---|---:|---:|
| Persistence | 66.429703 | 8.347802 Wh, 12.57% |
| LSTM B0 | 60.446456 | 2.364556 Wh, 3.91% |
| Transformer B0 | 61.030760 | 2.948859 Wh, 4.83% |
| Transformer sau S11 | 58.081901 | Reference hiện tại |

Đây là Validation performance, không phải final Test performance.

## 40. Notebook, processing log và canonical state

Phase 31–33 đều `VALID_REUSABLE`, nghĩa là thành công và lần sau chỉ `RENDER_ONLY`. Processing logs có thể rebuild để hiển thị mà không train lại.

Notebook hiện có section đến Phase 32. Phase 33 hoàn thành trên terminal và có canonical artifacts/log nhưng chưa được chèn section riêng vì notebook được khóa bảo toàn checksum trong đợt Phase 33.

Cached BLOCKED output cũ của Phase 31–32 chỉ là output trước recovery; chạy lại riêng `render_phase_resume(31)` và `render_phase_resume(32)` cập nhật presentation mà không training.

Bảng tổng hiện chỉ đếm literal `PASS`, nên đang phân loại nhầm `VALID_REUSABLE` thành `Needs attention`. Về khoa học, Phase 1–33 đã thành công; riêng Phase 3 là `PASS_WITH_WARNING`. Đây là lỗi presentation, không phải lỗi canonical.

## 41. Điều kiện trước Phase 34

1. Phase 33 đã có `approved_for_phase34=true`.
2. Không chạy lại D64 hoặc exact-match condition hợp lệ.
3. Phase 34 phải kế thừa FS2_TF1, YS1, L36, LAST_STEP, GELU, B32, LR `3e-4`, WD `1e-3`, dropout 0.1 và d_model 64.
4. Phase 34 chỉ thay đổi head count theo Phase detail; không đồng thời đổi d_model, layers hoặc FFN.
5. Test vẫn khóa; chỉ Validation RMSE chọn winner.
6. Terminal sở hữu execution; notebook chỉ đọc log và render static HTML.
7. Trước phase mới phải inspect canonical evidence và chỉ chạy missing conditions.

## 42. Kết luận

Pipeline đã đi từ environment/raw-data verification, schema/temporal/split locking, Train-only EDA, feature/scaler/window/DataLoader/metric contracts, registry và ba baseline, tới 11 controlled Transformer sweeps.

Kết quả tốt nhất hiện tại là Transformer reference `RUN_TR_S09_0016_AE0FB819` với Validation RMSE `58.081901 Wh`. Nó tốt hơn Persistence, LSTM B0 và Transformer B0 trên cùng Validation protocol. Phase 33 hoàn thành, Test chưa được sử dụng và Phase 34 đã có canonical authorization nhưng chỉ được bắt đầu theo plan riêng được duyệt.
