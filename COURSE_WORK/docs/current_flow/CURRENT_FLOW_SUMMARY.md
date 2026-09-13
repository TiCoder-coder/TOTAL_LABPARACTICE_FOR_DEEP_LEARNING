# CURRENT FLOW SUMMARY — Deep Learning Coursework

> Tài liệu tổng hợp dòng chảy hiện hành của project coursework, dựa trên toàn bộ 59 phases và kiến trúc source code sau lần refactor clean architecture lớn (06/09/2026).

---

## 1. Tổng Quan Project

### 1.1. Mục tiêu Coursework

Project Deep Learning cho **Multivariate Time-Series Regression**, sử dụng dataset **UCI Appliances Energy Prediction**.

| Thành phần | Mô tả |
|---|---|
| **Bài toán** | Dự đoán mức tiêu thụ điện năng (`Appliances`, đo bằng `Wh`) của các thiết bị gia dụng |
| **Input** | Cửa sổ lookback (`lookback ∈ {36, 72, 144}` step tương đương 6h, 12h, 24h) chứa nhiều biến cảm biến |
| **Target** | `Appliances` (Wh), horizon = 1 step (10 phút) |
| **Sample definition** | $X[t-L+1:t] \rightarrow Appliances[t+1]$ |
| **Task type** | Sequence-to-one regression |
| **Models so sánh** | Persistence baseline, LSTM, Transformer Encoder |
| **Evaluation** | MAE, RMSE, R² ở đơn vị Wh gốc |
| **Split** | Chronological Train / Validation / Test (70 / 15 / 15) |
| **Interpretability** | Phân tích Transformer attention maps |

### 1.2. Tech Stack

| Component | Version / Setting |
|---|---|
| Python | 3.10.11 |
| torch | 2.13.0 |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| matplotlib | 3.10.9 |
| jupyter | 1.1.1 |
| Selected device | cpu |
| Deterministic mode | D0 |
| Default dtype | torch.float32 |

---

## 2. Kiến Trúc Source Code (Sau Refactor)

### 2.1. Clean Architecture — Phân theo Tầng

```
src/course_work/
│
├── # ============ CORE DOMAIN ============
├── models/              # Neural architectures (LSTM, Transformer, RevIN)
├── training/            # Training engine + losses
├── attention/           # Core attention utilities
├── baselines/           # Baseline models (persistence, LSTM_B0, Transformer_B0)
├── evaluation/          # Evaluation metrics
│
├── # ============ DATA PIPELINE ============
├── data/                # Data acquisition, schema, features, scaling, windows
│
├── # ============ CONFIGURATION & GOVERNANCE ============
├── contracts/           # Coursework contract (single source of truth)
├── sweeps/              # Hyperparameter sweeps (S1-S19)
├── sanity/              # Forward pass sanity tests
├── diagnostics/         # Learning curve diagnostics
├── lstm_tuning/         # LSTM tuning helpers
├── rolling_origin/      # Rolling origin robustness evaluation
├── metric_addendum/     # MAPE metric addendum
├── experiments/         # Experiment tracking & registry
│
├── # ============ INFRASTRUCTURE ============
├── utils/               # Artifacts, environment, reproducibility
├── verification/        # Phase 46 verification (NO-TRAIN)
├── scaling/             # Public re-export of data.scaling
│
├── # ============ EXECUTION (NO-TRAIN) ============
├── final_model_lock/    # Phase 45 - Final model lock
├── final_test_evaluation/  # Phase 47 - Final test evaluation
│
├── # ============ ANALYSIS PHASES ============
├── analysis/
│   ├── prediction_analysis/         # Phase 48
│   ├── residual_analysis/           # Phase 49
│   ├── error_regime_analysis/       # Phase 50
│   ├── worst_error_analysis/        # Phase 51
│   ├── attention_extraction/        # Phase 52
│   ├── attention_heatmaps/          # Phase 53
│   ├── last_query_attention/        # Phase 54
│   ├── head_comparison/             # Phase 55
│   ├── error_conditioned_attention/ # Phase 56
│   ├── seed_stability_attention/    # Phase 57
│   ├── final_tables/                # Phase 58
│   └── final_conclusions/           # Phase 59
│
├── # ============ REPORTING & SCRIPTS ============
├── reporting/           # Dashboards and summaries
└── scripts/             # Pipeline runner scripts (p39-p47)
```

### 2.2. Nguyên Tắc Phân Tầng

| Tầng | Trách nhiệm | KHÔNG ĐƯỢC chứa |
|---|---|---|
| **models/** | Kiến trúc neural network | Training loop, data loading |
| **training/** | Engine huấn luyện | Model architecture, business logic |
| **data/** | ETL, feature engineering, scaling | Model code, metrics |
| **contracts/** | Single source of truth cho cấu hình | Logic xử lý |
| **analysis/** | Phân tích post-training, đọc-only artifacts | Training, mutation |
| **execution/** | Final pipeline (NO-TRAIN) | Sweep code, exploratory |
| **scripts/** | Pipeline orchestrators | Business logic |
| **utils/** | I/O atomic, environment | Domain logic |

---

## 3. Tổng Quan 59 Phases

### 3.1. Phases Theo Nhóm

| Nhóm | Phases | Mục đích |
|---|---|---|
| **0. Foundation** | 0–10 | Contract, environment, data acquisition, schema, temporal, EDA, feature engineering, splits, scaling, windows |
| **1. Modeling** | 11–22 | DataLoaders, metrics, experiments registry, persistence baseline, LSTM, Transformer, attention verification, forward sanity, training engine, baseline runs, learning diagnostics |
| **2. Sweeps** | 23–41 | S1–S19 hyperparameter sweeps: feature set, time feature, target scaling, lookback, pooling, activation, batch, learning rate, weight decay, dropout, d_model, heads, layers, ffn, loss, epoch cap, gradient clip, RevIN, boundary protocol |
| **3. Candidate & Lock** | 42–47 | Candidate synthesis, LSTM tuning, rolling origin robustness, final model lock, three-seed final runs, final test evaluation |
| **4. Analysis** | 48–59 | Prediction, residual, error by regime, worst error, attention extraction, attention heatmaps, last query attention, head comparison, error conditioned attention, seed stability, final tables, final conclusions |

### 3.2. Phase Mapping → Module

| Phase | Module Path | Trạng thái |
|---|---|---|
| 0 | `course_work.contracts` | ✅ |
| 1 | `course_work.utils.environment` | ✅ |
| 2 | `course_work.data.acquisition` | ✅ |
| 3 | `course_work.data.schema` | ✅ |
| 4 | `course_work.data.temporal` | ✅ |
| 5 | `course_work.data.splitting` | ✅ |
| 6 | `course_work.data.eda` | ✅ |
| 7 | `course_work.data.features` | ✅ |
| 8 | `course_work.data.feature_sets` | ✅ |
| 9 | `course_work.data.scaling` | ✅ |
| 10 | `course_work.data.windows` | ✅ |
| 11 | `course_work.data.datasets` | ✅ |
| 12 | `course_work.evaluation.metrics` | ✅ |
| 13 | `course_work.experiments.registry` | ✅ |
| 14 | `course_work.baselines.persistence` | ✅ |
| 15 | `course_work.models.lstm_regressor` | ✅ |
| 16 | `course_work.models.transformer_regressor` | ✅ |
| 17 | `course_work.attention.verification` | ✅ |
| 18 | `course_work.sanity.forward_sanity` | ✅ |
| 19 | `course_work.training.engine` | ✅ |
| 20 | `course_work.baselines.lstm_baseline` | ✅ |
| 21 | `course_work.baselines.transformer_b0` | ✅ |
| 22 | `course_work.diagnostics.learning_diagnostics` | ✅ |
| 23–30 | `course_work.sweeps.{feature_set,time_feature,...}` | ✅ |
| 31–40 | `course_work.sweeps.{weight_decay,dropout,...,revin}` | ✅ |
| 41 | `course_work.sweeps.boundary_protocol` | ✅ |
| 42 | `course_work.experiments.phase_execution` | ✅ |
| 43 | `course_work.lstm_tuning.stages` | ✅ |
| 44 | `course_work.rolling_origin.folds` | ✅ |
| 45 | `course_work.final_model_lock.candidate_lock` | ✅ |
| 46 | `course_work.scripts.p46_three_seed_runs` | ✅ |
| 47 | `course_work.final_test_evaluation.evaluation` | ✅ |
| 48 | `course_work.analysis.prediction_analysis` | ✅ |
| 49 | `course_work.analysis.residual_analysis` | ✅ |
| 50 | `course_work.analysis.error_regime_analysis` | ✅ |
| 51 | `course_work.analysis.worst_error_analysis` | ✅ |
| 52 | `course_work.analysis.attention_extraction` | ✅ |
| 53 | `course_work.analysis.attention_heatmaps` | ✅ |
| 54 | `course_work.analysis.last_query_attention` | ✅ |
| 55 | `course_work.analysis.head_comparison` | ✅ |
| 56 | `course_work.analysis.error_conditioned_attention` | ✅ |
| 57 | `course_work.analysis.seed_stability_attention` | ✅ |
| 58 | `course_work.analysis.final_tables` | ✅ |
| 59 | `course_work.analysis.final_conclusions` | ✅ |

---

## 4. Artifacts Layout

```
artifacts/
├── contracts/                   # Phase 0
├── environment/                 # Phase 1
├── acquisition/                 # Phase 2
├── schema/                      # Phase 3
├── temporal/                    # Phase 4
├── eda/                         # Phase 5
├── features/                    # Phase 6
├── feature_sets/                # Phase 7
├── splits/                      # Phase 8
├── scaling/                     # Phase 9
├── scalers/                     # Phase 9 (joblib)
├── windows/                     # Phase 10
├── dataloaders/                 # Phase 11
├── metrics/                     # Phase 12
├── experiments/                 # Phase 13
├── baselines/persistence/       # Phase 14
├── runs/                        # All training runs (LSTM, Transformer, sweeps)
├── sweeps/S1..S19/              # Sweep results
├── lstm_tuning/                 # Phase 43
├── rolling_origin/              # Phase 44
├── final_model_lock/            # Phase 45 (NO-TRAIN)
├── three_seed_final_runs/       # Phase 46
├── final_test/                  # Phase 47 (NO-TRAIN)
├── prediction_analysis/         # Phase 48
├── residual_analysis/           # Phase 49
├── error_by_regime/             # Phase 50
├── worst_error_analysis/        # Phase 51
├── attention_extraction/        # Phase 52
├── attention_heatmaps/          # Phase 53
├── last_query_attention/        # Phase 54
├── head_comparison/             # Phase 55
├── error_conditioned_attention/ # Phase 56
├── seed_stability_attention/    # Phase 57
├── final_tables/                # Phase 58
└── final_conclusions/           # Phase 59
```

Mỗi phase có:
- `phase_N_signoff.json` — sign-off artifact (gate)
- `_manifest.json` — inputs/outputs fingerprint
- `_contract.json` — config lock
- `_discrepancies.json` — issues tracker
- `_report.md` — human-readable summary
- `figures/` — plots
- `tables/` — CSV/LaTeX/MD tables

---

## 5. Data Flow Tổng Quan

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  RAW DATA (Phase 2)                                                          │
│  data/raw_data/energydata_complete.csv                                       │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  ACQUISITION + SCHEMA (Phases 2-3)                                           │
│  src/course_work/data/acquisition.py, schema.py                              │
│  → artifacts/acquisition/, schema/                                           │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  TEMPORAL + EDA (Phases 4-5)                                                 │
│  → artifacts/temporal/, eda/                                                 │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  FEATURE ENGINEERING + FEATURE SETS (Phases 6-7)                             │
│  data/interim/uci_appliances_energy_prediction/                              │
│      energydata_feature_engineered_v1.csv                                    │
│  → artifacts/features/, feature_sets/                                        │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  SPLIT (Phase 8)                                                             │
│  → artifacts/splits/                                                         │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  SCALING + WINDOWS (Phases 9-10)                                             │
│  → artifacts/scaling/, scalers/, windows/                                    │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  DATALOADERS + METRICS (Phases 11-12)                                        │
│  → artifacts/dataloaders/, metrics/                                          │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  MODELS (Phases 13-22)                                                       │
│  - Persistence baseline (Phase 14)                                           │
│  - LSTM implementation (Phase 15)                                           │
│  - Transformer implementation (Phase 16)                                      │
│  - Attention verification (Phase 17)                                         │
│  - Forward sanity (Phase 18)                                                 │
│  - Training engine (Phase 19)                                                │
│  - Baseline runs (Phases 20-21)                                              │
│  - Learning diagnostics (Phase 22)                                          │
│  → artifacts/baselines/, runs/                                               │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  HYPERPARAMETER SWEEPS (Phases 23-41)                                        │
│  S1-S19 (feature set, lookback, d_model, layers, ffn, dropout, RevIN, ...)  │
│  → artifacts/sweeps/S1..S19/                                                 │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  CANDIDATE SYNTHESIS (Phase 42)                                              │
│  → artifacts/candidate_synthesis/                                             │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  LSTM TUNING (Phase 43)                                                      │
│  → artifacts/lstm_tuning/                                                    │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  ROLLING ORIGIN (Phase 44)                                                    │
│  → artifacts/rolling_origin/                                                 │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  FINAL MODEL LOCK (Phase 45 - NO-TRAIN)                                      │
│  src/course_work/final_model_lock/                                           │
│  → artifacts/final_model_lock/                                               │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  THREE-SEED FINAL RUNS (Phase 46)                                            │
│  src/course_work/scripts/p46_three_seed_runs.py                              │
│  → artifacts/three_seed_final_runs/                                          │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  FINAL TEST EVALUATION (Phase 47 - NO-TRAIN, FINAL GATE)                     │
│  src/course_work/final_test_evaluation/                                      │
│  → artifacts/final_test/                                                     │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  ANALYSIS (Phases 48-59)                                                     │
│  - Prediction analysis (48)                                                  │
│  - Residual analysis (49)                                                    │
│  - Error by regime (50)                                                      │
│  - Worst error (51)                                                          │
│  - Attention extraction → heatmaps → last query → head comparison →          │
│    error-conditioned → seed stability (52-57)                                 │
│  - Final tables (58)                                                         │
│  - Final conclusions (59)                                                    │
│  → artifacts/prediction_analysis/, residual_analysis/, ...                   │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Sign-off Lattice (Phase Gates)

Mỗi phase tạo ra một `phase_N_signoff.json` làm gate cho phase sau. Phase N+1 sẽ kiểm tra:
- Phase N's `phase_N_signoff.json` tồn tại
- Manifest checksums khớp
- Required artifacts có mặt
- Không có discrepancies mở

Đây là cơ chế **frozen contract** đảm bảo upstream integrity trước khi chạy phase mới.

---

## 7. Scripts Layer

```
src/course_work/scripts/
├── p39_cleanup_unauthorized_runs.py
├── p39_compliance_corrective.py
├── p39_finalize.py
├── p39_strict_best_gc0.py
├── p40_finalize.py
├── p40_prepare_revin.py
├── p40_strict_best_rn1.py
├── p42_candidate_synthesis.py
├── p43_disposable_harness.py
├── p43_dry_run.py
├── p43_lstm_tuning.py
├── p43_one_epoch_harness.py
├── p43_stage_simulation.py
├── p44_pretrain_gate.py
├── p44_quarantine_invalid_official.py
├── p44_rolling_origin.py
├── p45_final_model_lock.py
├── p45_pretrain_gate.py
├── p46_archive_historical_checkpoints.py
├── p46_final_dev_metric_smoke_test.py
├── p46_pre_train_schema_simulation.py
├── p46_pretrain_gate.py
├── p46_registration_preflight.py
├── p46_three_seed_runs.py
├── p47_final_test_evaluation.py
└── p47_pretest_gate.py
```

Mỗi script `pXX_*.py` là **pipeline runner** chính thức cho phase XX, có thể gọi trực tiếp từ CLI hoặc notebook.

---

## 8. Testing Layer

```
tests/
├── contracts/         # Coursework contract tests
├── integration/       # Cross-phase integration tests
└── unit/              # Per-module unit tests (~80+ files)
```

---

## 9. Tài Liệu Liên Quan

| File | Mô tả |
|---|---|
| `CURRENT_FLOW_SUMMARY.md` | File này — tổng quan kiến trúc |
| `PHASE_1_TO_33_CURRENT_FLOW_DETAIL.md` | Chi tiết phases 1–33 (foundation + modeling + sweeps) |
| `PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md` | Chi tiết phases 34–59 (final + analysis) |
| `README.md` | Project root README |
| `docs/plan-doc/` | Plan documents cho từng phase |
| `docs/code_base_audit.md` | Code audit report |

---

**Phiên bản:** 06/09/2026 — sau lần refactor clean architecture lớn.
