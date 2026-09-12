# CODE BASE AUDIT — Deep Learning Coursework

> **Audit Timestamp:** 06/09/2026 (UTC+7)
> **Repository:** `/Users/vientu/Deep Learning/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK`
> **Audit Scope:** Toàn bộ luồng, code, follow, structure, conventions, design patterns, anomalies của project COURSE_WORK
> **Mục đích:** Tài liệu duy nhất để hiểu project, onboarding developer mới, đánh giá độ chuẩn chỉnh của codebase

---

## 📑 Mục Lục

1. [Tổng Quan Project](#1-tổng-quan-project)
2. [Tech Stack & Môi Trường](#2-tech-stack--môi-trường)
3. [Cấu Trúc Thư Mục](#3-cấu-trúc-thư-mục)
4. [Source Code Organization](#4-source-code-organization)
5. [Configuration & Contracts](#5-configuration--contracts)
6. [Tests Layer](#6-tests-layer)
7. [Artifacts Layer](#7-artifacts-layer)
8. [Data Layer](#8-data-layer)
9. [Notebook Layer](#9-notebook-layer)
10. [Documentation Layer](#10-documentation-layer)
11. [Phase Mapping Chi Tiết (0-59)](#11-phase-mapping-chi-tiết-0-59)
12. [Data Flow Tổng Quan](#12-data-flow-tổng-quan)
13. [Design Patterns & SOLID](#13-design-patterns--solid)
14. [Reproducibility Story](#14-reproducibility-story)
15. [Key Python Modules - Detailed](#15-key-python-modules---detailed)
16. [Anomalies & Observations](#16-anomalies--observations)
17. [Quy Ước & Rule Base](#17-quy-ước--rule-base)
18. [Phụ Lục](#18-phụ-lục)

---

## 1. Tổng Quan Project

### 1.1. Mục tiêu
Project Deep Learning cho **Multivariate Time-Series Regression** trên dataset **UCI Appliances Energy Prediction**:
- **Input:** Cửa sổ lookback (36/72/144 step = 6h/12h/24h) chứa 29 biến cảm biến (nhiệt độ, độ ẩm, ánh sáng, ...).
- **Target:** `Appliances` (Wh) tại step `t+1` (10 phút sau).
- **Sample definition:** $X[t-L+1:t] \rightarrow Appliances[t+1]$.
- **Models:** Persistence, LSTM, Transformer Encoder.
- **Evaluation:** MAE, RMSE, R² ở đơn vị Wh gốc (không scaled space).
- **Split:** Chronological 70/15/15 (TRAIN / VALIDATION / TEST).
- **Final seeds:** 42, 123, 2026.

### 1.2. Quy mô project
| Thành phần | Quy mô |
|---|---|
| **Phases** | 60 (0-59, đánh số 0-based cho environment, 1-based cho data) |
| **Source code** | ~388 file Python, ~50,000 LOC |
| **Tests** | ~108 file Python, ~30,500 LOC |
| **Notebook** | 152 cells (~10 MB) |
| **Artifacts** | 46 directories |
| **Docs** | 3 current_flow + 1 notebook walkthrough + 1 audit (file này) + 79 analysis_error + 112 save logs |
| **Scripts** | 76 files ở root + 26 in-package |

### 1.3. Đặc điểm nổi bật
- **Single Source of Truth:** Một file `configs/base/coursework_contract.json` quy định toàn bộ.
- **Test Firewall:** Dữ liệu Test bị LOCKED cho đến Phase 47 (final gate duy nhất).
- **Phase Gates:** Mỗi phase tạo `phase_N_signoff.json` với SHA-256, phase sau kiểm tra trước khi chạy.
- **NO-TRAIN Phases:** Phases 45, 47 chỉ lock/eval, không train.
- **FINAL_REFIT mode:** Phase 46 train over TRAIN+VAL (không hold out VAL).
- **Fold-local scalers:** Phase 44 fit scaler per-fold (không leak từ fold khác).
- **Strict reproducibility:** D0 mode, seed 42 mặc định, hash mọi artifact.

---

## 2. Tech Stack & Môi Trường

### 2.1. Python Target
- **Python:** `>=3.10,<3.11` (locked trong `pyproject.toml`)
- **Lý do:** Tương thích tốt nhất với torch 2.13 và tất cả dependencies.

### 2.2. Pinned Dependencies (`requirements.txt`)
| Package | Version | Vai trò |
|---|---|---|
| `ipykernel` | 7.3.0 | Jupyter kernel |
| `joblib` | 1.5.3 | Scaler serialization |
| `jupyter` | 1.1.1 | Notebook interface |
| `matplotlib` | 3.10.9 | Visualization |
| `nbclient` | 0.11.0 | Notebook execution |
| `nbformat` | 5.11.0 | Notebook I/O |
| `numpy` | 2.2.6 | Numerical ops |
| `pandas` | 2.3.3 | DataFrames |
| `pytest` | 9.1.1 | Testing framework |
| `scikit-learn` | 1.7.2 | Scalers, metrics, ML utils |
| `torch` | 2.13.0 | Deep learning core |
| `seaborn` | 0.13.2 | Statistical viz |

### 2.3. Device Selection
Code dùng `utils.environment.select_device()` với thứ tự ưu tiên:
1. **CUDA** (nếu có GPU NVIDIA)
2. **MPS** (Apple Silicon)
3. **CPU** (fallback)

Current default: **CPU** (do chạy trên macOS dev environment).

### 2.4. Determinism
- **Default mode:** D0 (fully deterministic)
- **DEVELOPMENT_SEED:** 42
- **FINAL_SEEDS:** (42, 123, 2026)
- **Configuration:** `utils/reproducibility.configure_reproducibility(mode="D0"|"D1")`

---

## 3. Cấu Trúc Thư Mục

### 3.1. Top-Level Layout
```
COURSE_WORK/
├── artifacts/            # 46 directories - output của mỗi phase
├── configs/              # Single source of truth contract
│   └── base/
│       └── coursework_contract.json
├── docs/                 # Documentation layer
│   ├── analysis_error/   # 79 issue analyses
│   ├── code_base_audit.md   # File này
│   ├── current_flow/     # 3 files - current architecture
│   ├── link&discussion_to_result/  # Notebook walkthrough
│   ├── plan/             # Historical & current plans
│   ├── rule_base/        # Coding conventions
│   └── save_log_in_processing/  # 112 phase logs
├── link/                 # Symlink mirror của data/
│   ├── interim/
│   └── raw_data/
├── notebook_course_work/
│   └── CourseWork.ipynb  # Single notebook, 152 cells
├── scripts/              # 76 root-level scripts (~50 disposable probes)
├── src/                  # Python package
│   └── course_work/      # 22 subpackages
├── tests/                # 108 test files
├── pyproject.toml        # Package metadata
└── requirements.txt      # Pinned deps
```

### 3.2. `src/course_work/` — Source Package
22 subpackages, tổ chức theo **Clean Architecture**:

| Tầng | Subpackages | Trách nhiệm |
|---|---|---|
| **CORE DOMAIN** | `models/`, `training/`, `attention/`, `evaluation/` | Neural architectures, training loop, metrics |
| **DATA PIPELINE** | `data/` | ETL: acquisition → features → splits → scaling → windows → datasets |
| **CONFIG & GOVERNANCE** | `contracts/`, `sweeps/`, `sanity/`, `diagnostics/`, `lstm_tuning/`, `rolling_origin/`, `metric_addendum/`, `experiments/`, `verification/` | Phase orchestration & validation |
| **INFRASTRUCTURE** | `utils/`, `scaling/` (re-export) | I/O, environment, reproducibility |
| **EXECUTION (NO-TRAIN)** | `final_model_lock/`, `final_test_evaluation/` | Final governance |
| **ANALYSIS** | `analysis/` | 12 subpackages cho phases 48-59 |
| **REPORTING** | `reporting/` | HTML dashboards |
| **PIPELINE** | `scripts/` | Phase orchestrators (in-package) |
| **BASELINES** | `baselines/` | Persistence, LSTM_B0, Transformer_B0 |

### 3.3. `tests/` Structure
```
tests/
├── conftest.py           # Setup: adds src/ to sys.path, sets COURSE_WORK_ROOT
├── contracts/            # 2 files
│   ├── test_coursework_contract.py
│   └── test_selective_execution_policy.py
├── integration/          # 9 files - cross-phase end-to-end
│   ├── test_phase_0_to_5_chain.py
│   ├── test_phase_0_to_8_presentation.py
│   ├── test_notebook_boundary.py
│   ├── test_final_dev_dataset.py
│   ├── test_phase46_*.py  (3 files)
│   ├── test_phase47_infrastructure.py
│   ├── test_mape_addendum.py
│   └── dry_run_final_dev.py
└── unit/                 # ~97 files - per-phase/per-module
```

### 3.4. `artifacts/` Structure (46 directories)
Mỗi phase có directory riêng với pattern:
- `phase_N_signoff.json` — phase gate (status: OK/FAIL)
- `phase_N_manifest.json` — input/output checksums
- `phase_N_contract.json` — config lock
- `phase_N_report.md` — human-readable summary
- `phase_N_discrepancies.json` — issues tracker
- `figures/`, `tables/` — visualization

---

## 4. Source Code Organization

### 4.1. Module Dependency Graph
```
                    ┌─────────────────┐
                    │   contracts     │  (coursework.py)
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
   ┌─────────┐         ┌──────────┐         ┌──────────┐
   │  data/  │         │  utils/  │         │evaluation│
   └────┬────┘         └─────┬────┘         └────┬─────┘
        │                    │                    │
        └─────────┬──────────┴──────────┬─────────┘
                  │                     │
                  ▼                     ▼
            ┌──────────┐         ┌────────────┐
            │  models/ │         │experiments/│
            └─────┬────┘         └──────┬─────┘
                  │                     │
                  ▼                     │
            ┌──────────┐                │
            │ training/│                │
            └─────┬────┘                │
                  │                     │
                  ▼                     │
            ┌──────────┐                │
            │ baselines│────────────────┤
            └──────────┘                │
                                        │
        ┌───────────────────────────────┼─────────────────────┐
        │                               │                     │
        ▼                               ▼                     ▼
   ┌─────────┐                   ┌──────────┐          ┌───────────┐
   │ sweeps/ │                   │lstm_tuning│         │rolling_origin│
   └─────────┘                   └──────────┘          └───────┬───┘
                                                                │
        ┌───────────────────────────────────────────────────────┤
        │                                                       │
        ▼                                                       ▼
   ┌──────────────┐                                   ┌──────────────────┐
   │final_model_  │                                   │final_test_       │
   │lock/         │                                   │evaluation/       │
   └──────────────┘                                   └──────────────────┘
        │                                                       │
        └──────────────────┬────────────────────────────────────┘
                           ▼
                  ┌─────────────────┐
                  │    analysis/    │  (12 subpackages, 48-59)
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   reporting/    │  (HTML dashboards)
                  └─────────────────┘
```

### 4.2. Quy Tắc Phân Tầng (Strict)
- `models/` KHÔNG import `data/`, `training/`, `experiments/`.
- `data/` KHÔNG import `models/`, `training/`.
- `evaluation/` chỉ phụ thuộc `numpy`, `pandas`, `utils/` — pure functions.
- `analysis/` chỉ đọc artifacts, KHÔNG mutate.
- `scripts/` ở ngoài cùng, được phép import bất kỳ tầng nào.
- Mọi I/O đi qua `utils.artifacts` (single entry point).

### 4.3. Module Inventory

| Module | LOC | Files | Purpose |
|---|---|---|---|
| `analysis/` | n/a | 12 packages + `__init__.py` | Read-only analysis phases 48-59 |
| `attention/` | 352 | 6 | Attention contracts + Phase 17 verification |
| `baselines/` | 2,047 | 3 | Persistence (P14), LSTM_B0 (P20), Transformer_B0 (P21) |
| `contracts/` | 183 | 1 | `coursework.py` - Phase 0 contract loader |
| `data/` | 6,365 | 11 | Phases 2-11 pipeline |
| `diagnostics/` | 1,030 | 2 | Phase 22 learning diagnostics |
| `evaluation/` | 1,182 | 1 | `metrics.py` - Phase 12 metrics |
| `experiments/` | 2,862 | 3 | `registry.py` - Phase 13 + sweep execution |
| `final_model_lock/` | 2,393 | 10 | Phase 45 NO-TRAIN governance |
| `final_test_evaluation/` | 3,852 | 6 | Phase 47 final test |
| `lstm_tuning/` | 1,785 | 9 | Phase 43 multi-stage tuning |
| `metric_addendum/` | 460 | 5 | MAPE addendum |
| `models/` | 1,440 | 6 | LSTM, Transformer, RevIN, positional encoding |
| `reporting/` | n/a | 16 | Per-phase HTML dashboards |
| `rolling_origin/` | ~9,700 | 22 (+tests) | Phase 44 rolling-origin robustness |
| `sanity/` | 287 | 1 | Phase 18 forward sanity |
| `scaling/` | 907 | 1 | Public re-export of `data.scaling` |
| `scripts/` | ~15,500 | 28 | Phase 39-47 orchestrators |
| `sweeps/` | ~11,000 | 12 | Hyperparameter sweeps S1-S19 |
| `training/` | 937 | 3 | `engine.py` - Phase 19 |
| `utils/` | 714 | 3 | `artifacts.py`, `environment.py`, `reproducibility.py` |
| `verification/` | 1,208 | 2 | Phase 46 verification |

---

## 5. Configuration & Contracts

### 5.1. Single Source of Truth
**File:** `configs/base/coursework_contract.json`
**Version:** `COURSEWORK-CONTRACT-v1`
**Fingerprint:** SHA-256 của canonical JSON

### 5.2. Contract Schema
```json
{
  "contract_version": "COURSEWORK-CONTRACT-v1",
  "problem": {
    "task": "multivariate_time_series_regression",
    "dataset": "UCI Appliances Energy Prediction",
    "target": "Appliances",
    "target_unit": "Wh",
    "sampling_minutes": 10,
    "forecast_horizon_steps": 1,
    "sample_definition": "X[t-L+1:t] -> Appliances[t+1]"
  },
  "lookbacks": {"options": [36, 72, 144], "primary": 144},
  "split": {"type": "chronological", "train": 0.7, "val": 0.15, "test": 0.15},
  "models": ["persistence", "lstm", "transformer_encoder"],
  "metrics": {"selection": "validation_rmse", "final": ["mae", "rmse", "r2"]},
  "final_seeds": [42, 123, 2026],
  "option_registry": { /* 18 groups, see below */ },
  "baseline_transformer": { /* reference config */ },
  "research_questions": [/* RQ1-RQ7 */],
  "protocol_violations": [/* 10 named violations */]
}
```

### 5.3. Option Registry (18 Groups)
Mỗi group có explicit ID để sweep sử dụng:

| Group | IDs | Values |
|---|---|---|
| `feature_sets` | FS0, FS1, FS2 | exogenous_only / +autoregressive / +random_controls |
| `time_features` | TF0, TF1 | off / sin/cos + weekend |
| `target_scaling` | YS0, YS1 | off / train_only_standardization |
| `lookbacks` | L36, L72, L144 | 36 / 72 / 144 steps |
| `pooling` | P0, P1 | last_step / mean |
| `activation` | A0, A1 | relu / gelu |
| `batch_size` | B32, B64 | 32 / 64 |
| `learning_rate` | LR1, LR2, LR3 | 1e-4 / 3e-4 / 1e-3 |
| `weight_decay` | WD0, WD1, WD2, WD3 | 0 / 1e-4 / 1e-3 / 1e-2 |
| `dropout` | DR01, DR02, DR03 | 0.1 / 0.2 / 0.3 |
| `d_model` | D32, D64 | 32 / 64 |
| `heads` | H2, H4 | 2 / 4 |
| `layers` | N1, N2 | 1 / 2 |
| `ffn` | F64, F128, F256 | 64 / 128 / 256 |
| `loss` | L0, L1 | mse / huber_delta_1 |
| `epoch_cap` | E50, E100 | 50 / 100 |
| `gradient_clip` | GC0, GC1 | off / 1.0 |
| `revin` | RN0, RN1 | off / on |
| `boundary` | WB0, WB1 | context_carry_over / drop_boundary |

### 5.4. Contract API (`src/course_work/contracts/coursework.py`)
- `EXPECTED_OPTION_IDS` — set of valid IDs per group
- `load_coursework_contract(path=None)` → dict
- `validate_coursework_contract(contract)` → `tuple[str, ...]` errors
- `coursework_contract_fingerprint(contract)` → SHA-256 hex
- `materialize_phase_0(project_root=None)` — Phase 0 entry point

### 5.5. Baseline Transformer Reference Config
```json
{
  "feature_set": "FS1",
  "time_feature": "TF1",
  "target_scaling": "YS1",
  "revin": "RN0",
  "lookback": 144,
  "d_model": 64,
  "heads": 4,
  "layers": 2,
  "ffn": 128,
  "dropout": 0.1,
  "activation": "gelu",
  "pooling": "last_step",
  "batch_size": 64,
  "learning_rate": 3e-4,
  "weight_decay": 1e-4,
  "loss": "mse",
  "epochs": 50,
  "patience": 10,
  "gradient_clip": 1.0,
  "seed": 42
}
```

---

## 6. Tests Layer

### 6.1. Test Count & Distribution
| Category | Count | Mục đích |
|---|---|---|
| **Unit** | ~97 | Per-module, per-phase |
| **Integration** | 9 | Multi-phase chains |
| **Contracts** | 2 | Option registry, execution policy |
| **Tổng** | **108 files** | **~30,500 LOC** |

### 6.2. Unit Tests Pattern
- Mỗi sweep có 1 unit test (vd: `test_d_model.py`, `test_dropout.py`)
- Mỗi phase từ 43-59 có 1+ test files
- `test_phase47_infrastructure.py` (1,080 LOC) — strictest invariants
- `test_phase43_corrective_implementation.py` (1,763 LOC) — lớn nhất
- 5 RevIN-specific tests

### 6.3. Integration Tests Pattern
- `test_phase_0_to_5_chain.py` (318 LOC) — phase 0→5 end-to-end
- `test_phase_0_to_8_presentation.py` — phase 0→8
- `test_notebook_boundary.py` (515 LOC) — notebook kernel/sandbox
- `test_final_dev_dataset.py` (564 LOC) — FINAL_DEV region
- `test_phase46_*.py` (3 files) — pretrain gates
- `test_phase47_infrastructure.py` — test firewall enforcement
- `test_mape_addendum.py` — MAPE addendum
- `dry_run_final_dev.py` — no-train run

### 6.4. Contract Tests
- `test_coursework_contract.py` — validate option registry
- `test_selective_execution_policy.py` (283 LOC) — phase resume rules

### 6.5. `conftest.py` Setup
```python
# tests/conftest.py adds src/ to sys.path,
# sets os.environ["COURSE_WORK_ROOT"],
# calls os.chdir() so get_project_root() works.
```

---

## 7. Artifacts Layer

### 7.1. Phase Artifacts (46 directories)
| # | Directory | Phase | Key Files |
|---|---|---|---|
| 1 | `contracts/` | 0 | `coursework_contract.json`, `phase_0_signoff.json` |
| 2 | `environment/` | 1 | `environment_report.json`, `requirements_freeze.txt` |
| 3 | `acquisition/` | 2 | `acquisition_log.json`, `phase_2_signoff.json` |
| 4 | `schema/` | 3 | `schema_manifest.json` |
| 5 | `temporal/` | 4 | `temporal_manifest.json` |
| 6 | `eda/` | 5 | `eda_manifest.json`, `figures/`, `tables/` |
| 7 | `features/` | 6 | `feature_engineering_manifest.json` |
| 8 | `feature_sets/` | 7 | `feature_set_manifest.json` |
| 9 | `splits/` | 8 | `split_manifest.json`, `split_membership.csv` |
| 10 | `scaling/` | 9 | `scaling_manifest.json`, `scaler_registry.json` |
| 11 | `scalers/` | 9 | (joblib files - empty, see anomaly) |
| 12 | `windows/` | 10 | `window_manifest.json`, `window_index.csv` |
| 13 | `dataloaders/` | 11 | `dataloader_manifest.json` |
| 14 | `metrics/` | 12 | `metric_manifest.json` |
| 15 | `experiments/` | 13 | `experiment_registry.jsonl`, `registry_manifest.json` |
| 16 | `baselines/persistence/` | 14 | `persistence_manifest.json` |
| 17 | `models/lstm/` | 15 | `lstm_model_manifest.json` |
| 18 | `models/transformer/` | 16 | `transformer_model_manifest.json` |
| 19 | `attention_verification/` | 17 | `attention_verification_manifest.json` |
| 20 | `forward_sanity/` | 18 | `forward_sanity_manifest.json` |
| 21 | `training_engine/` | 19 | `training_engine_manifest.json` |
| 22 | `lstm_baseline/` | 20 | `lstm_baseline_run_summary.csv` |
| 23 | `transformer_b0/` | 21 | `transformer_b0_run_summary.csv` |
| 24 | `learning_diagnostics/` | 22 | `learning_diagnostics_manifest.json` |
| 25 | `sweeps/s1..s8/` | 23-30 | per-sweep manifests, winners |
| 26 | `sweeps/S9..S19/` | 31-41 | per-sweep manifests, winners |
| 27 | `candidate_synthesis/` | 42 | `candidate_synthesis_manifest.json` |
| 28 | `lstm_tuning/` | 43 | `lstm_tuning_manifest.json` |
| 29 | `rolling_origin/` | 44 | 39 O44.* artifacts |
| 30 | `final_model_lock/` | 45 | 38 O45.* artifacts |
| 31 | `final_dev_region/` | 46 | FINAL_DEV region manifest |
| 32 | `three_seed_final_runs/` | 46 | Official checkpoints |
| 33 | `final_test/` | 47 | `final_test_evaluation_contract.json` |
| 34 | `prediction_analysis/` | 48 | per-phase dashboard |
| 35 | `residual_analysis/` | 49 | per-phase dashboard |
| 36 | `error_by_regime/` | 50 | per-phase dashboard |
| 37 | `worst_error_analysis/` | 51 | per-phase dashboard |
| 38 | `attention_extraction/` | 52 | per-phase dashboard |
| 39 | `attention_heatmaps/` | 53 | per-phase dashboard |
| 40 | `last_query_attention/` | 54 | per-phase dashboard |
| 41 | `head_comparison/` | 55 | per-phase dashboard |
| 42 | `error_conditioned_attention/` | 56 | per-phase dashboard |
| 43 | `seed_stability_attention/` | 57 | per-phase dashboard |
| 44 | `final_tables/` | 58 | per-phase dashboard |
| 45 | `final_conclusions/` | 59 | per-phase dashboard |
| 46 | `runs/` | various | ~25 RUN_* directories |
| - | `_history/` | recovery | timestamped snapshots |

### 7.2. Universal Artifact Schema
Mỗi phase directory có các file pattern:
- `phase_N_signoff.json` — gate với status PASS/FAIL
- `phase_N_manifest.json` — input/output checksums
- `phase_N_contract.json` — config lock
- `phase_N_report.md` — human-readable summary
- `phase_N_discrepancies.json` — issues tracker
- `figures/`, `tables/` — visualization

### 7.3. `_history/` Mechanism
- `utils.environment.recover_environment_revision()` move existing artifacts
- Snapshot pattern: `recover_X()` → move to `_history/<UTC_stamp>/`
- Cho phép rollback mà không mất dữ liệu

---

## 8. Data Layer

### 8.1. Canonical Data Location
**Hiện tại:** `link/raw_data/`, `link/interim/`

| Path | Nội dung |
|---|---|
| `link/raw_data/energydata_complete.csv` | UCI CSV, ~19,735 rows × 29 cols |
| `link/raw_data/dataset_manifest.json` | SHA-256 + metadata |
| `link/raw_data/source_metadata.json` | Source provenance |
| `link/raw_data/variable_metadata.csv` | 29 variable dictionary |
| `link/raw_data/checksums.sha256` | Integrity checksums |
| `link/raw_data/README_SOURCE.md` | Source documentation |
| `link/raw_data/source/appliances_energy_prediction.zip` | Protected archive |
| `link/interim/uci_appliances_energy_prediction/energydata_feature_engineered_v1.csv` | Feature-engineered output |

### 8.2. ⚠️ Data Path Anomaly
- Code references: `data/raw_data/...`
- Physical location: `link/raw_data/...`
- Có thể `data/` là symlink → `link/` hoặc runtime remap. Cần verify.

### 8.3. Missing Directories
- `data_after_processing/` — KHÔNG tồn tại
- `data_after_split/` — KHÔNG tồn tại
- (Có thể do cleanup trong lần refactor gần đây)

---

## 9. Notebook Layer

### 9.1. Structure
- **File:** `notebook_course_work/CourseWork.ipynb`
- **Cells:** 152 (75 markdown + 75 code + 2 headers)
- **Size:** ~10 MB JSON

### 9.2. Cell-to-Phase Mapping
| Cells | Phase | Purpose |
|---|---|---|
| 0-3 | (intro) | Title, problem, imports, display |
| 4-5 | 1 | Environment |
| 6-7 | 2 | Data Acquisition |
| 8-9 | 3 | Schema Audit |
| 10-11 | 4 | Temporal Integrity |
| 12-13 | 5 | Chronological Split |
| 14-43 | 6 | EDA (14 sub-sections) |
| 44-45 | 7 | Feature Engineering |
| 46-47 | 8 | Feature-Set Variants |
| 48-49 | 9 | Train-Only Scaling |
| 50-51 | 10 | Window Builder |
| 52-53 | 11 | DataLoaders |
| 54-55 | 12 | Shared Metrics |
| 56-57 | 13 | Experiment Registry |
| 58-59 | 14 | Persistence Baseline |
| 60-61 | 15 | LSTM Implementation |
| 62-63 | 16 | Transformer Implementation |
| 64-65 | 17 | Attention Verification |
| 66-67 | 18 | Forward Sanity |
| 68-69 | 19 | Training Engine |
| 70-71 | 20 | LSTM Baseline Run |
| 72-73 | 21 | Transformer B0 Run |
| 74-75 | 22 | Learning Diagnostics |
| 76-95 | 23-32 | Sweeps S1-S10 |
| 96-97 | 33 | Transformer Config snapshot |
| 98-113 | 34-41 | Sweeps S12-S19 |
| 114-115 | 42 | Candidate Synthesis |
| 116-125 | 43-47 | Final pipeline (dashboards) |
| 126-145 | 48-57 | Analysis dashboards |
| 146-147 | 58 | Final Tables |
| 148 | 59 | Final Conclusions (markdown) |
| 149-151 | — | MAPE Addendum |

### 9.3. ⚠️ Notebook Phase Numbering
Notebook dùng **+1 offset** so với artifacts cho phases 5-10:
- Notebook: "Phase 5 = Chronological Split", "Phase 6 = EDA"
- Artifacts: `phase_5_signoff.json` = splits, `phase_6_signoff.json` = features

**Quy ước:** Artifacts là source of truth. Notebook dùng naming khác (legacy?).

---

## 10. Documentation Layer

### 10.1. `docs/` Structure
```
docs/
├── analysis_error/                 # 79 issue analysis files
├── code_base_audit.md              # File này
├── current_flow/
│   ├── CURRENT_FLOW_SUMMARY.md            # 410 lines
│   ├── PHASE_1_TO_33_CURRENT_FLOW_DETAIL.md  # 523 lines
│   └── PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md # 505 lines
├── link&discussion_to_result/
│   └── NOTEBOOK_CELLS_WALKTHROUGH.md      # 1,094 lines
├── plan/
│   ├── plan_before_process/         # Pre-process plans (~30 files)
│   ├── plan_detail_for_each_phase/
│   ├── plan_overview/
│   └── plan_to_refactor&fix/        # Recent refactor plans
├── rule_base/
│   ├── architecture_rule.md
│   └── rule_code.md
└── save_log_in_processing/          # 112 JSON processing logs
```

### 10.2. Doc Usage Guide
| Doc | Khi nào đọc |
|---|---|
| `code_base_audit.md` | Hiểu tổng quan toàn project |
| `CURRENT_FLOW_SUMMARY.md` | Quick reference về kiến trúc |
| `PHASE_1_TO_33_CURRENT_FLOW_DETAIL.md` | Chi tiết foundation + modeling |
| `PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md` | Chi tiết sweeps + final + analysis |
| `NOTEBOOK_CELLS_WALKTHROUGH.md` | Đọc notebook output |
| `analysis_error/<file>.md` | Khi gặp bug, tìm issue đã biết |
| `rule_base/architecture_rule.md` | Conventions |
| `rule_base/rule_code.md` | Coding style |

---

## 11. Phase Mapping Chi Tiết (0-59)

### 11.1. Foundation (0-10)
| Phase | Primary Module | Script | Purpose |
|---|---|---|---|
| 0 | `contracts.coursework` | — | Contract materialization |
| 1 | `utils.environment` | — | Env report, freeze, smoke test |
| 2 | `data.acquisition` | — | UCI ZIP + SHA-256 |
| 3 | `data.schema` | — | Schema audit |
| 4 | `data.temporal` | — | Temporal integrity |
| 5 | `data.splitting` (artifact: splits) | — | Chronological split |
| 6 | `data.features` (artifact: features) | — | Feature engineering |
| 7 | `data.feature_sets` | — | Feature-set variants |
| 8 | `data.scaling` (artifact: scaling) | — | Train-only scaling |
| 9 | `data.windows` (artifact: windows) | — | Window builder |
| 10 | `data.datasets` (artifact: dataloaders) | — | DataLoaders |
| 11 | `data.datasets` (artifact: dataloaders) | — | DataLoaders |

### 11.2. Modeling (12-22)
| Phase | Primary Module | Purpose |
|---|---|---|
| 12 | `evaluation.metrics` | Shared metrics (MAE/RMSE/R²/MAPE) |
| 13 | `experiments.registry` | Experiment registry |
| 14 | `baselines.persistence` | Persistence baseline |
| 15 | `models.lstm_regressor` | LSTM architecture |
| 16 | `models.transformer_regressor` | Transformer architecture |
| 17 | `attention.verification` | Attention contract audit |
| 18 | `sanity.forward_sanity` | Forward pass tests |
| 19 | `training.engine` | Generic training engine |
| 20 | `baselines.lstm_baseline` | LSTM_B0 training run |
| 21 | `baselines.transformer_b0` | Transformer_B0 training run |
| 22 | `diagnostics.learning_diagnostics` | Learning curve analysis |

### 11.3. Sweeps (23-41)
| Phase | Primary Module | Factor | Conditions |
|---|---|---|---|
| 23 | `sweeps.sweep_results` (feature_set) | feature_set | FS0, FS1, FS2 |
| 24 | `sweeps.sweep_results` (time_feature) | time_feature | TF0, TF1 |
| 25 | `sweeps.sweep_results` (target_scaling) | target_scaling | YS0, YS1 |
| 26 | `sweeps.sweep_results` (lookback) | lookback | L36, L72, L144 |
| 27 | `sweeps.sweep_results` (pooling) | pooling | P0, P1 |
| 28 | `sweeps.sweep_results` (activation) | activation | A0, A1 |
| 29 | `sweeps.sweep_results` (batch_size) | batch_size | B32, B64 |
| 30 | `sweeps.sweep_results` (learning_rate) | learning_rate | LR1, LR2, LR3 |
| 31 | `sweeps.weight_decay` | weight_decay | WD0, WD1, WD2, WD3 |
| 32 | `sweeps.dropout` | dropout | DR01, DR02, DR03 |
| 33 | `sweeps.d_model` | d_model | D32, D64 |
| 34 | `sweeps.heads` | num_heads | H2, H4 |
| 35 | `sweeps.layers` | num_layers | N1, N2 |
| 36 | `sweeps.ffn` | ffn_dim | F64, F128, F256 |
| 37 | `sweeps.loss` | loss_name | L0 (MSE), L1 (Huber) |
| 38 | `sweeps.epoch_cap` | max_epochs | E50, E100 |
| 39 | `sweeps.gradient_clip` | gradient_clipping | GC0 (off), GC1 (1.0) |
| 40 | `sweeps.revin` + `models.revin` | revin_enabled | RN0, RN1 |
| 41 | `sweeps.boundary_protocol` | boundary | WB0, WB1 |

### 11.4. Candidate & Lock (42-47)
| Phase | Primary Module | Script | Purpose |
|---|---|---|---|
| 42 | `experiments.phase_execution` | `scripts/p42_candidate_synthesis.py` | Candidate synthesis |
| 43 | `lstm_tuning.stages` | `scripts/p43_lstm_tuning.py` + 5 harnesses | LSTM tuning |
| 44 | `rolling_origin.pipeline` | `scripts/p44_rolling_origin.py` | Rolling-origin robustness |
| 45 | `final_model_lock.candidate_lock` | `scripts/p45_final_model_lock.py` | **NO-TRAIN** lock |
| 46 | `scaling.final_scaling` + `data.final_dev` | `scripts/p46_three_seed_runs.py` | Three-seed FINAL_REFIT |
| 47 | `final_test_evaluation.evaluation` | `scripts/p47_final_test_evaluation.py` | **FINAL GATE** |

### 11.5. Analysis (48-59)
| Phase | Primary Module | Purpose |
|---|---|---|
| 48 | `analysis.prediction_analysis` | Read-only prediction analysis |
| 49 | `analysis.residual_analysis` | Residual analysis (ACF, sign runs, tails) |
| 50 | `analysis.error_regime_analysis` | Per-regime error metrics |
| 51 | `analysis.worst_error_analysis` | Worst-K cases + casebook |
| 52 | `analysis.attention_extraction` | Extract attention tensors |
| 53 | `analysis.attention_heatmaps` | Render heatmaps (V1, V2, V3) |
| 54 | `analysis.last_query_attention` | Last-query attention metrics |
| 55 | `analysis.head_comparison` | Inter-head comparison (JSD, cosine) |
| 56 | `analysis.error_conditioned_attention` | High vs low error attention |
| 57 | `analysis.seed_stability_attention` | Cross-seed attention stability |
| 58 | `analysis.final_tables` | FT01-FT10 final tables |
| 59 | `analysis.final_conclusions` | Final abstract + conclusions |

---

## 12. Data Flow Tổng Quan

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 0: CONTRACT                                                           │
│  configs/base/coursework_contract.json → contracts/coursework.py             │
│  → artifacts/contracts/coursework_contract.{json,sha256}                      │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: ENVIRONMENT                                                        │
│  utils.environment.py → requirements_freeze.txt, environment_report.json    │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASES 2-4: RAW DATA                                                        │
│  data.acquisition → data.schema → data.temporal                              │
│  Input: link/raw_data/energydata_complete.csv                                │
│  Output: artifacts/{acquisition, schema, temporal}/                           │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASES 5-7: PROCESSING                                                      │
│  data.splitting → data.features → data.feature_sets                          │
│  → link/interim/.../energydata_feature_engineered_v1.csv                     │
│  → artifacts/{splits, features, feature_sets}/                                │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASES 8-11: SCALING + WINDOWS + DATALOADERS                                 │
│  data.scaling → data.windows → data.datasets                                 │
│  Output: artifacts/{scaling, scalers, windows, dataloaders}/                  │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASES 12-22: METRICS + MODELS + BASELINES                                   │
│  evaluation.metrics → experiments.registry → baselines.persistence           │
│  → models.{lstm_regressor, transformer_regressor}                            │
│  → attention.verification → sanity.forward_sanity → training.engine           │
│  → baselines.{lstm_baseline, transformer_b0} → diagnostics                    │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASES 23-41: HYPERPARAMETER SWEEPS (S1-S19)                                 │
│  sweeps.* → artifacts/sweeps/S*/                                              │
│  Each sweep: run variants → pick winner by val RMSE → update reference       │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 42: CANDIDATE SYNTHESIS                                                │
│  experiments.phase_execution.py → scripts/p42_candidate_synthesis.py        │
│  Output: artifacts/candidate_synthesis/                                       │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 43: LSTM TUNING                                                        │
│  lstm_tuning.* (5 stages: lt1-lt5)                                            │
│  → scripts/p43_lstm_tuning.py + 5 harness/sim scripts                        │
│  → artifacts/lstm_tuning/                                                     │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 44: ROLLING-ORIGIN ROBUSTNESS                                          │
│  rolling_origin.{folds, refit_engine, pipeline, scaling, ...}                  │
│  → scripts/p44_rolling_origin.py                                              │
│  → 5 folds × candidates → recommended Transformer                            │
│  → artifacts/rolling_origin/ (39 O44.* artifacts)                              │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 45: FINAL MODEL LOCK (NO-TRAIN)                                        │
│  final_model_lock.{candidate_lock, recipe, lineage, fingerprints}              │
│  → scripts/p45_final_model_lock.py                                            │
│  → artifacts/final_model_lock/ (38 O45.* artifacts)                            │
│  → FREEZE config: architecture, optimizer, loss, scaler, RevIN, seeds, ...   │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 46: THREE-SEED FINAL RUNS (TRAIN + VAL, FINAL_REFIT)                   │
│  data.final_dev + scaling.final_scaling                                        │
│  → scripts/p46_three_seed_runs.py + 5 corrective scripts                      │
│  → artifacts/three_seed_final_runs/official_checkpoints/seed_{42,123,2026}/*.pt │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 47: FINAL TEST EVALUATION (FINAL GATE)                                 │
│  final_test_evaluation.{evaluation, checkpoint_loader, scaler_loader, ...}     │
│  → scripts/p47_final_test_evaluation.py                                       │
│  → Load Phase 46 checkpoints → Eval on Test (FINAL_TEST_POP-v1, L72)         │
│  → artifacts/final_test/ (RMSE, MAE, R² for 3 seeds + mean ± std)            │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASES 48-59: ANALYSIS (READ-ONLY)                                           │
│  analysis.* (12 subpackages)                                                  │
│  → Each phase: read Phase 47 predictions → produce figures/tables/findings   │
│  → reporting.* renders HTML dashboards                                       │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 13. Design Patterns & SOLID

### 13.1. Design Patterns Identified

| Pattern | Where Applied | How |
|---|---|---|
| **Single Source of Truth** | `configs/base/coursework_contract.json` | One JSON drives all phases; SHA-256 fingerprint |
| **Immutable Dataclass** | Everywhere | `@dataclass(frozen=True)` for configs (TransformerModelConfig, DatasetConfig, PredictionBundle, MetricResult, PhaseState, SweepPhaseSpec, FoldDefinition, ...) |
| **Factory** | Model/Dataset construction | `build_reference_transformer_config()`, `build_model_from_run_config()`, `build_dataset_suite()`, `build_dataloader()`, `build_test_evaluation_loader()`, `build_final_dev_dataset()` |
| **Strategy** | Loss/activation/pooling/scaling selection | Enum-driven: `criterion_config()`, `_resolve_activation()`, target_scaling YS0/YS1 |
| **Registry** | Experiment tracking | `ExperimentRegistry` (JSONL + CSV), `experiment_families.csv`, register_run/start_run/complete_run/fail_run |
| **State Machine** | Run lifecycle | `RunStatus` (PLANNED→REGISTERED→RUNNING→{COMPLETED,FAILED,CANCELLED}→{INVALIDATED,ARCHIVED}), `STATUS_TRANSITIONS` |
| **Template Method** | Every phase | `materialize_phase_N()` family — verify upstream → build artifacts → write sign-off |
| **Manifest + Sign-off** | Universal artifact shape | `{phase_N_signoff.json, _manifest.json, _contract.json, _discrepancies.json, _report.md}` |
| **Single-write-once-or-verify** | All artifact writes | `utils.artifacts.write_bytes_once_or_verify()`, `write_json_once_or_verify()` (strict), `*_permissive` variants for metadata-only updates |
| **Atomic Write + fsync** | Crash-safe I/O | `utils.artifacts.atomic_write_bytes()` — tempfile + fsync + os.replace |
| **Test Firewall** | Data leakage prevention | `validate_evaluation_access()`, `TargetAccessMode.TEST_LOCKED`, `PHASE_47_AUTHORIZATION` |
| **FINAL_REFIT Mode** | Phase 46 train-on-TRAIN+VAL | `TrainingEngine.train(final_refit_mode=True)`, `EvaluationMode.FINAL_DEV_DIAGNOSTIC` |
| **Fold-local Scalers** | Phase 44 robustness | `rolling_origin.scaling.{fit_fold_a_x_scaler, fit_fold_b_x_scaler, build_bundle}` |
| **NO-TRAIN Orchestration** | Phases 45, 47 | Governance phases with `optimizer_steps=0, new_test_runs=0` |
| **Population Fingerprinting** | Cross-phase consistency | `derive_population_fingerprint()`, `MetricPopulationContext` |
| **Heartbeat** | Training monitoring | `engine.py:SWEEP_HEARTBEAT_PATH`, external watchdog |
| **Wall-clock Timeout** | Sweep safety | `install_training_timeout()` / `disable_training_timeout()`, POSIX SIGALRM |
| **Determinism** | Reproducibility | `set_seed()`, `configure_reproducibility("D0"|"D1")`, seed=42 default |

### 13.2. SOLID Principles

| Principle | Adherence |
|---|---|
| **S (Single Responsibility)** | ✓ Strong: mỗi module 1 concern (metrics.py = metrics only, datasets.py = datasets only) |
| **O (Open/Closed)** | ✓ Extends via enums (BoundaryProtocol, TargetAccessMode, EvaluationMode, FailureType) without modifying consumers |
| **L (Liskov Substitution)** | ✓ LSTMRegressor và TransformerRegressor cùng `forward(x)` signature; symmetric `materialize_phase_15/16` |
| **I (Interface Segregation)** | ✓ SequenceWindowDataset returns minimal dicts (TEST: `{x, sample_idx}`, TRAIN/VAL: `{x, y_model, y_raw_wh}`) |
| **D (Dependency Inversion)** | ✓ High-level (`final_test_evaluation`, `rolling_origin`) depend on abstractions (ExperimentRegistry, MetricResult) not concretes |

### 13.3. Configuration Injection
- Mọi hyperparameter đọc từ `configs/base/coursework_contract.json` hoặc per-phase sign-off.
- KHÔNG hard-code constants trong code.
- Runtime configs (`run_config.json` ở `artifacts/runs/<RUN_ID>/`) validated against `EXPERIMENT_FAMILIES`.

---

## 14. Reproducibility Story

### 14.1. Determinism Layers
1. **Seed Layer:** `utils/reproducibility.set_seed(seed=42)` cho Python, NumPy, PyTorch (CPU + CUDA), DataLoader workers.
2. **Mode Layer:** `configure_reproducibility(mode="D0")` — fully deterministic (cuDNN benchmark off, deterministic algorithms).
3. **Hardware Layer:** `select_device()` returns CPU/MPS/CUDA deterministically.
4. **Contract Layer:** Single SHA-256 fingerprint per phase sign-off.
5. **Artifact Layer:** Every artifact SHA-256 verified on write (single-write-once-or-verify).

### 14.2. Seed Strategy
- **DEVELOPMENT_SEED:** 42 (cho tất cả dev work, sweeps, baseline runs).
- **FINAL_SEEDS:** (42, 123, 2026) — chỉ dùng cho Phase 46 three-seed FINAL_REFIT.

### 14.3. Reproducibility Tests
- `tests/integration/test_phase_46_cache_exclusion.py` (198 LOC) — verify cache exclusion.
- `tests/unit/test_phase46_verification.py` (290 LOC) — verify Phase 46 reproducibility.
- `tests/unit/test_reproducibility.py` — direct seed determinism tests.

### 14.4. Failure Recovery
- `utils.environment.recover_environment_revision()` — move existing artifacts to `_history/<UTC_stamp>/`.
- Per-phase `recover_X()` functions cho selective rollback.
- `RERUN_REASONS` registry trong `experiments/registry.py` cho audit trail.

---

## 15. Key Python Modules - Detailed

### 15.1. `src/course_work/utils/artifacts.py` (215 LOC)
**Purpose:** Single I/O entry point. Tất cả writes phải qua đây.

**Key APIs:**
- `get_project_root()` — `parents[3]` từ file location
- `canonical_json_bytes()` — sorted JSON bytes
- `sha256_bytes()`, `sha256_file()`
- `atomic_write_bytes()` — tempfile + fsync + os.replace
- `write_bytes_once_or_verify()` — strict: re-write fail unless identical
- `write_bytes_once_or_verify_permissive()` — allow metadata updates with `compare_keys`
- `_strip_provenance_metadata()` — strip `created_at` for stable fingerprints
- `write_json_once_or_verify()`, `write_text_once_or_verify()` (CSV-aware)

### 15.2. `src/course_work/utils/environment.py` (423 LOC)
**Purpose:** Environment capture, device selection, dependency freeze.

**Key APIs:**
- `CORE_DISTRIBUTIONS`, `STABLE_ENVIRONMENT_FIELDS`
- `select_device()` — CUDA > MPS > CPU
- `resolve_kernel_contract()`
- `environment_inventory()`, `environment_identity()`, `environment_identity_differences()`
- `device_smoke_test()`
- `dependency_freeze()` — writes `requirements_freeze.txt`
- `materialize_phase_1()` — Phase 1 entry
- `recover_environment_revision()` — `_history/` snapshot

### 15.3. `src/course_work/utils/reproducibility.py` (76 LOC)
**Purpose:** Seed & determinism.

**Constants:**
- `DEVELOPMENT_SEED = 42`
- `FINAL_SEEDS = (42, 123, 2026)`

**Key APIs:**
- `set_seed(seed)` — Python + NumPy + PyTorch (CPU + CUDA) + DataLoader workers
- `configure_reproducibility(mode="D0"|"D1")`
- `seed_worker(worker_id)` — DataLoader worker init
- `build_torch_generator(seed)`
- `randomness_smoke_test()`

### 15.4. `src/course_work/contracts/coursework.py` (183 LOC)
**Purpose:** Phase 0 contract loader/validator.

**Key APIs:**
- `EXPECTED_OPTION_IDS` — set per group
- `load_coursework_contract(path=None)`
- `validate_coursework_contract(contract)` → `tuple[str, ...]`
- `coursework_contract_fingerprint(contract)` — SHA-256
- `materialize_phase_0(project_root=None)`

### 15.5. `src/course_work/data/datasets.py` (1,273 LOC)
**Purpose:** Phase 11 DataLoader construction + Test firewall.

**Constants:**
- `DATALOADER_VERSION = "DATALOADERS-v1"`
- `TargetAccessMode` enum: TRAIN, VALIDATION, TEST_LOCKED, TEST_EVALUATION, FINAL_DEV

**Key Classes:**
- `DatasetConfig` (frozen)
- `LoaderConfig` (frozen)
- `SequenceWindowDataset` (map-style PyTorch Dataset)

**Key Builders:**
- `build_dataset_suite()` — TRAIN/VAL/TEST_LOCKED
- `build_test_evaluation_dataset()` — for Phase 47
- `build_final_dev_dataset()` — for Phase 46
- `build_dataloader()`, `build_train_validation_loaders()`, `build_test_locked_loader()`, `build_test_evaluation_loader()`

**Test Firewall:** Mỗi builder enforce access mode. TEST_LOCKED chỉ accessible khi `PHASE_47_AUTHORIZATION` granted.

### 15.6. `src/course_work/models/transformer_regressor.py` (507 LOC)
**Purpose:** Phase 16 Transformer architecture.

**Constants:**
- `TRANSFORMER_IMPL_VERSION = "TRANSFORMER_IMPL-v1"`
- `PHASE_VERSION = "PHASE-16-v1"`

**Key Classes:**
- `TransformerModelConfig` (frozen): d_model=64, num_heads=4, num_layers=2, ffn_dim=128, dropout=0.1, GELU, LAST_STEP, SINUSOIDAL, attention_aware=True, post-norm
- `validate_transformer_config()` — strict shape/dtype validation
- `TransformerRegressor(nn.Module)` — `forward()` and `forward_with_attention()` (returns per-head [B,H,L,L])
- `build_reference_transformer_config()`, `materialize_phase_16()`

### 15.7. `src/course_work/models/lstm_regressor.py` (466 LOC)
**Purpose:** Phase 15 LSTM architecture.

**Constants:**
- `LSTM_IMPL_VERSION = "LSTM_IMPL-v1"`

**Config:**
- Unidirectional only (`bidirectional=False`)
- `batch_first=True`
- `LAST_STEP` pooling
- Stateless zero-init hidden state

### 15.8. `src/course_work/models/revin.py` (273 LOC)
**Purpose:** RevIN (Reversible Instance Normalization) for Phase 40.

**Constants:**
- `REVIN_EPS = 1e-5`
- Affine, target-selective (RN1 contract)

**Key Classes:**
- `TargetSelectiveRevIN(nn.Module)`
- `RevINScope` dataclass

### 15.9. `src/course_work/training/engine.py` (615 LOC)
**Purpose:** Phase 19 generic training loop.

**Key Classes:**
- `TrainingResult` dataclass
- `EarlyStopping` (MIN mode only, patience-based)
- `TrainingEngine` — main loop with heartbeat, non-finite-gradient guard, gradient clip, optional FINAL_REFIT mode

**Key Functions:**
- `build_model_from_run_config()` — factory
- `persist_run_artifacts()` — save checkpoint (best+last), history, predictions, metrics
- `install_training_timeout()` / `disable_training_timeout()` — POSIX SIGALRM
- `metric_unit_for()` — helper

### 15.10. `src/course_work/evaluation/metrics.py` (1,182 LOC)
**Purpose:** Phase 12 shared metrics.

**Enums:**
- `EvaluationMode`: TRAIN_DIAGNOSTIC, VALIDATION, FINAL_TEST, FINAL_DEV_DIAGNOSTIC

**Key Classes:**
- `EvaluationContext`
- `MetricPopulationContext` (Phase 44 fold-aware)
- `PredictionBundle`
- `MetricResult`
- `SupplementaryMapeResult`

**Key Functions:**
- `validate_evaluation_access()` — strict split/mode firewall
- `derive_population_fingerprint()` — canonical
- `convert_predictions_to_wh()` — inverse scale
- `compute_mae_wh()`, `compute_rmse_wh()`, `compute_r2()`
- `compute_regression_metrics()` — central function
- `build_mape_metric_contract()`, `compute_mape_pct()`
- `compare_to_baseline()`, `materialize_phase_12()`

### 15.11. `src/course_work/baselines/persistence.py` (1,161 LOC)
**Purpose:** Phase 14 persistence baseline.

**Constants:**
- `PERSISTENCE_VERSION = "PERSISTENCE-v1"`
- `TEST_ACCESS_POLICY = "LOCKED_UNTIL_PHASE_47"`

**Key Classes:**
- `PersistenceConfig`
- `PersistencePreparedData`
- `PersistenceEvaluationResult`

**Key Functions:**
- `predict_persistence()` — pure: $\hat{y}[t+1] = y[t]$ với 4 test cases
- `prepare_validation_persistence_data()`
- `run_persistence_contract_tests()` — 8 unit tests + lookback invariance (L36/L72/L144)
- `materialize_phase_14()`

### 15.12. `src/course_work/experiments/registry.py` (2,191 LOC)
**Purpose:** Phase 13 experiment tracking.

**Constants:**
- `EXPERIMENT_VERSION = "EXPERIMENTS-v1"`
- `RECORD_SCHEMA_VERSION = 1`
- `FAILURE_TAXONOMY_VERSION = "FAILURES-v1"`
- `FEATURE_VARIANTS = {"FS0_TF0", "FS0_TF1", "FS1_TF0", "FS1_TF1", "FS2_TF0", "FS2_TF1"}`
- `MODEL_FAMILIES = {"PERSISTENCE", "LSTM", "TRANSFORMER_ENCODER"}`
- `EXPERIMENT_FAMILIES` — 26 entries
- `RERUN_REASONS` — incl. `PHASE44_CORRECTIVE_RERUN`, `PHASE46_CORRECTIVE_RERUN`, `PHASE46_HARD_STOP_PROBE`

**Enums:**
- `RunStatus` (PLANNED→...→COMPLETED/FAILED/CANCELLED→INVALIDATED/ARCHIVED)
- `ExecutionType`
- `FailureType` (16 codes)
- `ArtifactType` (13 codes)
- `STATUS_TRANSITIONS`

**Key Class:** `ExperimentRegistry`
- Manages: `experiment_registry.jsonl`, `experiment_registry.csv`, `experiment_families.csv`, `run_artifact_registry.csv`
- APIs: `register_run()`, `start_run()`, `complete_run()`, `fail_run()`, `cancel_run()`, `validate_registry()`

### 15.13. `src/course_work/experiments/phase_execution.py` (671 LOC)
**Purpose:** Sweep phase execution orchestration.

**Enums:**
- `PhaseState` (9 states: VALID_REUSABLE, LOG_MISSING, LOG_STALE, DERIVED_ARTIFACT_MISSING, CONDITION_INCOMPLETE, SIGNOFF_INVALID, UPSTREAM_INVALID, ENVIRONMENT_INVALID, RUNNING, FAILED)
- `PhaseAction` (6 actions: PROCEED, SKIP, RE_RUN, RECOVER, ABORT, FORCE)

**Key Class:**
- `SweepPhaseSpec` — declares conditions, prerequisite paths, winner/reference files, log file

**Dict:** `SWEEP_PHASE_SPECS` — specs cho phases 23-41.

### 15.14. `src/course_work/final_test_evaluation/__init__.py` (hard-locked constants)
**Purpose:** Phase 47 final test gate constants.

**Hard-locked Constants:**
- `LOCKED_CANDIDATE = "TR_C2_ALT_LOOKBACK"`
- `LOCKED_LOOKBACK = 72`
- `LOCKED_FEATURES = 33`
- `LOCKED_BOUNDARY_PROTOCOL = "WB0_CONTEXT_CARRY_OVER"`
- `LOCKED_SEEDS = [42, 123, 2026]`
- `LOCKED_CONFIG_FP = "585c5e79e6a1c8c4..."` (SHA-256)
- `OFFICIAL_RUNS` — dict với checkpoint_sha256 per seed
- `FINAL_SCALING` — dict với x/y_scaler_sha256
- `FINAL_DEV_POP_FP`, `FINAL_DEV_WINDOW_COUNT = 16630`
- `LSTM_TUNED_DEV` — metadata (known L36 vs L72 fairness caveat)
- `PHASE_47_SPLIT = "TEST"`
- `LSTM_ELIGIBILITY_STATUSES`

---

## 16. Anomalies & Observations

### 16.1. Anomalies

| ID | Anomaly | Impact | Resolution |
|---|---|---|---|
| **A1** | Missing `COURSE_WORK/README.md` | Developer onboarding khó | Tạo file (TODO) |
| **A2** | `data/` directory missing, only `link/` exists | Code may fail if `data/` not symlinked | Verify symlink or remap paths |
| **A3** | Empty `artifacts/scalers/` | Phase 9 scalers missing on disk | Regenerate from `scaling/scaler_registry.json` |
| **A4** | Notebook phase numbering offset (+1) vs artifacts | Confusing docs | Standardize on artifact numbering (current_flow docs already follow) |
| **A5** | Empty stubs: `attention/{extraction,head_comparison,heatmaps,last_query}.py` | 0 bytes; real code in `analysis/` | Delete stubs or move logic |
| **A6** | Two parallel `scripts/` directories | Confusion (in-src vs root-level) | Document distinction |
| **A7** | ~50 disposable probe scripts (`_phase4{4,5,6,7}_*.py`) | Pollute root `scripts/` | Move to `_disposable/` or delete |
| **A8** | Multiple `_history/` directories | Spread across artifacts/ | OK — intentional snapshot pattern |
| **A10** | LSTM known fairness caveat | LSTM not eval-able on same pop as Transformer | Documented in `final_test_evaluation/__init__.py:LSTM_TUNED_DEV` |
| **A11** | Manual rerun authorization registry | Same fingerprints registered multiple times | Intentional recovery pattern |
| **A12** | FINAL_DEV vs FINAL_TEST_POP-v1 naming | Different populations | Documented, intentional |
| **A13** | `__pycache__` operation-not-permitted | macOS sandbox artifact | Ignore |

### 16.2. Large Modules (god-modules)
- `experiments/registry.py` — 2,191 LOC
- `baselines/persistence.py` — 1,161 LOC
- `data/datasets.py` — 1,273 LOC
- `evaluation/metrics.py` — 1,182 LOC
- `scripts/p43_lstm_tuning.py` — 2,882 LOC
- `scripts/p40_prepare_revin.py` — 1,509 LOC
- `scripts/p40_finalize.py` — 1,439 LOC
- `scripts/p46_three_seed_runs.py` — 2,242 LOC
- `scripts/p47_final_test_evaluation.py` — 891 LOC
- `rolling_origin/pipeline.py` — 1,539 LOC
- `rolling_origin/real_run.py` — 1,673 LOC
- `final_test_evaluation/writers.py` — 1,815 LOC

**Recommendation:** Tương lai nên split thành các submodules nhỏ hơn. Hiện tại chấp nhận được do mỗi file có 1 concern rõ ràng.

### 16.3. Empty/Dead Code
- 4 empty stubs in `attention/`
- ~50 disposable probe scripts in root `scripts/`
- Empty `artifacts/scalers/` directory

---

## 17. Quy Ước & Rule Base

### 17.1. Code Style (`docs/rule_base/rule_code.md`)
- Type hints required cho public APIs.
- Frozen dataclasses cho configs.
- Docstrings cho mọi module/class.
- Snake_case cho functions/variables.
- PascalCase cho classes.
- ALL_CAPS cho constants.
- Imports grouped: stdlib, third-party, local.

### 17.2. Architecture Rules (`docs/rule_base/architecture_rule.md`)
- Clean Architecture tầng (Models → Data → Evaluation → Experiments → Phase-specific).
- `models/` KHÔNG import `data/`, `training/`, `experiments/`.
- `data/` KHÔNG import `models/`, `training/`.
- Mọi I/O qua `utils.artifacts`.
- KHÔNG hard-code hyperparameters.
- Test firewall: Test data chỉ accessible từ Phase 47.

### 17.3. Phase Sign-off Pattern
```json
{
  "phase": 47,
  "status": "PASS",  // or "FAIL"
  "input_checksums": {"...": "sha256"},
  "output_checksums": {"...": "sha256"},
  "config_fingerprint": "sha256",
  "tests": [...],
  "warnings": [...],
  "discrepancies": [...],
  "created_at": "ISO timestamp"
}
```

### 17.4. Artifact Naming
- `phase_N_signoff.json` — gate
- `phase_N_manifest.json` — checksums
- `phase_N_contract.json` — config lock
- `phase_N_report.md` — human-readable
- `phase_N_discrepancies.json` — issues
- `phase_N_summary.json` — quick metrics

---

## 18. Phụ Lục

### 18.1. Glossary

| Term | Definition |
|---|---|
| **Phase** | Một bước trong pipeline (0-59) |
| **Sign-off** | JSON gate file xác nhận phase hoàn thành |
| **Manifest** | JSON liệt kê inputs/outputs với checksums |
| **FINAL_REFIT** | Train mode train-on-TRAIN+VAL (no early stop) |
| **Test Firewall** | Cơ chế chặn truy cập Test data trước Phase 47 |
| **Population Fingerprint** | SHA-256 định danh một population (fold/split) |
| **NO-TRAIN Phase** | Phase chỉ lock config/governance, không train |
| **Sweep** | Hyperparameter sweep qua nhiều conditions |
| **Winner** | Condition tốt nhất trong sweep (theo validation RMSE) |
| **Reference Update** | Update config cho sweep kế tiếp dựa trên winner |

### 18.2. Quick Links

| Document | Path |
|---|---|
| Current Flow Summary | `docs/current_flow/CURRENT_FLOW_SUMMARY.md` |
| Phase 1-33 Detail | `docs/current_flow/PHASE_1_TO_33_CURRENT_FLOW_DETAIL.md` |
| Phase 34-59 Detail | `docs/current_flow/PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md` |
| Notebook Walkthrough | `docs/link&discussion_to_result/NOTEBOOK_CELLS_WALKTHROUGH.md` |
| Code Style Rules | `docs/rule_base/rule_code.md` |
| Architecture Rules | `docs/rule_base/architecture_rule.md` |
| Main Contract | `configs/base/coursework_contract.json` |
| Notebook | `notebook_course_work/CourseWork.ipynb` |

### 18.3. Recommended Reading Order

1. **Bắt đầu:** Đọc file này (`code_base_audit.md`) → hiểu tổng quan.
2. **Architecture:** Đọc `CURRENT_FLOW_SUMMARY.md` → hiểu clean architecture.
3. **Phase Details:** Đọc `PHASE_1_TO_33_CURRENT_FLOW_DETAIL.md` → hiểu foundation + modeling.
4. **Phase Details (cont.):** Đọc `PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md` → hiểu sweeps + final + analysis.
5. **Code Style:** Đọc `docs/rule_base/rule_code.md` và `architecture_rule.md`.
6. **Run:** Open notebook `CourseWork.ipynb` với `NOTEBOOK_CELLS_WALKTHROUGH.md` bên cạnh.
7. **Debug:** Search `docs/analysis_error/` cho issue tương tự.

---

**Phiên bản:** 06/09/2026 — sau lần refactor clean architecture lớn.
**Maintainer:** AI-assisted documentation
