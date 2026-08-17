# CURRENT FLOW SUMMARY — Deep Learning Coursework

> Tài liệu này là báo cáo tổng hợp tổng thể dòng chảy của project coursework hiện hành, dựa trên toàn bộ 59 cells của notebook `CourseWork.ipynb` và 31 file Python trong `src/course_work/`. Mọi thông tin dưới đây được tổng hợp từ việc đọc chi tiết source và notebook, không chạy bất kỳ cell nào.

---

## 1. Tổng quan Project

### 1.1. Mục tiêu coursework

Đây là một project Deep Learning cho **Multivariate Time-Series Regression**, sử dụng dataset **UCI Appliances Energy Prediction**.

- **Bài toán:** Dự đoán mức tiêu thụ điện năng (`Appliances` đo bằng `Wh`) của các thiết bị gia dụng từ dữ liệu cảm biến đa biến trong quá khứ.
- **Input:** Một cửa sổ lookback (`lookback ∈ {36, 72, 144}` step, tương đương 6h, 12h, 24h) chứa nhiệt độ trong nhà, độ ẩm, thời tiết ngoài trời, ánh sáng và các biến khác.
- **Target:** `Appliances` (Wh), horizon = 1 step (10 phút).
- **Sample definition:** $X[t-L+1:t] \rightarrow Appliances[t+1]$.
- **Task type:** Sequence-to-one regression.
- **Models so sánh:** Persistence baseline (Phase 14 — đã hoàn thành), LSTM (Phase 15 — chưa triển khai), Transformer Encoder (Phase 16 — chưa triển khai).
- **Evaluation:** MAE, RMSE, R² ở đơn vị Wh gốc, với chronological Train/Validation/Test split (70/15/15).
- **Interpretability:** Phân tích Transformer attention maps để xem timestep nào nhận attention (dành cho Phase 21–23 — chưa triển khai).
- **Ràng buộc toàn vẹn:** Giữ thứ tự thời gian, fit transformation trên Train only, ngăn leakage, dành Test cho final gate.

### 1.2. Tech stack chính

Dựa trên Phase 1 environment fingerprint đã capture từ notebook output:

| Component | Version / Setting |
|---|---|
| Python | 3.10.11 |
| Kernel | python3 |
| ipykernel | 7.3.0 |
| jupyter | 1.1.1 |
| matplotlib | 3.10.9 |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| scikit_learn | 1.7.2 |
| torch | 2.13.0 |
| Selected device | cpu |
| Deterministic mode | D0 |
| Default dtype | torch.float32 |
| ENV artifact version | ENV-v1 |

### 1.3. Cấu trúc thư mục source (canonical)

```
COURSE_WORK/
├── README.md
├── requirements.txt
├── pyproject.toml
├── working_rule.md
├── configs/
│   └── base/
│       └── coursework_contract.json
├── data/
│   ├── raw_data/
│   │   ├── source/
│   │   ├── energydata_complete.csv
│   │   ├── checksums.sha256
│   │   ├── dataset_manifest.json
│   │   ├── source_metadata.json
│   │   ├── variable_metadata.csv
│   │   └── README_SOURCE.md
│   ├── interim/
│   │   └── uci_appliances_energy_prediction/
│   │       └── energydata_feature_engineered_v1.csv
│   ├── data_after_processing/
│   └── data_after_split/
├── artifacts/
│   ├── contracts/
│   ├── environment/
│   ├── acquisition/
│   ├── schema/
│   ├── temporal/
│   ├── eda/
│   │   ├── tables/
│   │   └── figures/
│   ├── features/
│   ├── feature_sets/
│   ├── splits/
│   ├── scaling/
│   ├── scalers/
│   ├── windows/
│   ├── dataloaders/
│   ├── metrics/
│   ├── experiments/
│   ├── baselines/
│   │   └── persistence/
│   └── runs/
│       └── RUN_PS_PS_0001_AFFD3E3F/
├── src/
│   └── course_work/
│       ├── __init__.py
│       ├── contracts/
│       │   ├── __init__.py
│       │   └── coursework.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── acquisition.py
│       │   ├── schema.py
│       │   ├── temporal.py
│       │   ├── eda.py
│       │   ├── features.py
│       │   ├── feature_sets.py
│       │   ├── splitting.py
│       │   ├── scaling.py
│       │   ├── windows.py
│       │   └── datasets.py
│       ├── reporting/
│       │   ├── __init__.py
│       │   ├── eda.py
│       │   └── phase_summary.py
│       ├── evaluation/
│       │   └── metrics.py
│       ├── experiments/
│       │   └── registry.py
│       ├── baselines/
│       │   └── persistence.py
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── artifacts.py
│       │   ├── environment.py
│       │   └── reproducibility.py
│       └── attention/
│           ├── __init__.py
│           ├── extraction.py    (rỗng — chưa triển khai)
│           ├── heatmaps.py      (rỗng — chưa triển khai)
│           ├── last_query.py    (rỗng — chưa triển khai)
│           └── head_comparison.py (rỗng — chưa triển khai)
├── tests/
│   ├── contracts/
│   ├── unit/
│   └── integration/
├── notebook_course_work/
│   └── CourseWork.ipynb
└── docs/
    ├── RULE_BASE/
    ├── plan-doc/
    └── save_log_in_processing/
```

Source ownership đã được khóa trong `architecture_rule.md` (COURSE-WORK-ARCHITECTURE-v1, status `ACTIVE_HUMAN_APPROVED`).

### 1.4. Nguyên tắc kiến trúc cốt lõi

- **Source-owned processing:** Mọi logic nghiệp vụ phải có owner duy nhất dưới `src/course_work/`.
- **Human-approved direct Phase 5 EDA exception:** Notebook được phép thực hiện direct descriptive EDA và plotting cho Phase 5, nhưng **không** fetch dataset, ghi artifact, hoặc thay đổi raw data.
- **Sign-off lineage:** Mỗi Phase tạo `phase_N_signoff.json` chỉ khi upstream đã sign-off và input checksum khớp.
- **Test firewall:** Phase 12 trở đi không được materialize Test metrics cho đến khi Phase 47 mở `FINAL_TEST` gate.

---

## 2. Phase 0 — Coursework Contract

### 2.1. Mục đích

Định nghĩa **coursework contract** ở dạng machine-readable để khoá tất cả các option ID mà mọi Phase sau phải tuân theo. Đây là canonical prerequisite nội bộ nhưng **không xuất hiện trong notebook** (Phase 0 vẫn là canonical prerequisite nội bộ nhưng không có heading, orchestration cell, import hoặc output trong notebook).

### 2.2. Notebook representation

Phase 0 **không có** heading, code cell, import, hoặc output nào trong `CourseWork.ipynb`. Quy trình Phase 0 được gọi nội bộ bởi các Phase sau thông qua import trong cell 3.

### 2.3. Code Python liên quan: `src/course_work/contracts/coursework.py` (185 dòng)

#### Public API

| Function | Mục đích |
|---|---|
| `EXPECTED_OPTION_IDS` | Dict định nghĩa 19 option groups (`feature_sets`, `time_features`, `target_scaling`, `lookbacks`, `pooling`, `activation`, `batch_size`, `learning_rate`, `weight_decay`, `dropout`, `d_model`, `heads`, `layers`, `ffn`, `loss`, `epoch_cap`, `gradient_clip`, `revin`, `boundary`) với tập ID hợp lệ. |
| `load_coursework_contract(path=None)` | Đọc `configs/base/coursework_contract.json` thông qua `read_json`. Trả về dict. |
| `validate_coursework_contract(contract)` | Validate toàn bộ schema: contract_version, problem, lookbacks, split fractions (0.7/0.15/0.15), metrics (`validation_rmse` selection, final=`['mae', 'rmse', 'r2']`, scale=`original_wh`), models (`['persistence', 'lstm', 'transformer_encoder']`), final_seeds (`[42, 123, 2026]`), option_registry completeness, research_questions RQ1–RQ7, 8 policy fields. Trả về tuple các error strings. |
| `coursework_contract_fingerprint(contract)` | SHA-256 hash từ canonical JSON bytes. |
| `materialize_phase_0(project_root=None)` | Entry point Phase 0: load + validate + compute fingerprint + atomic write artifact + atomic write checksum + verify reload + tạo `phase_0_signoff.json`. |

#### Schema yêu cầu (key fields)

```python
EXPECTED_PROBLEM = {
    "task": "multivariate_time_series_regression",
    "dataset": "UCI Appliances Energy Prediction",
    "target": "Appliances",
    "target_unit": "Wh",
    "sampling_minutes": 10,
    "forecast_horizon_steps": 1,
    "forecast_horizon_minutes": 10,
}

# Lookbacks: options=[36, 72, 144], primary=144
# Split: type=chronological, fractions=[0.7, 0.15, 0.15], membership_basis=target_timestamp
# Metrics: selection=validation_rmse, final=['mae', 'rmse', 'r2'], report_scale=original_wh
# Models: ['persistence', 'lstm', 'transformer_encoder']
# final_seeds: [42, 123, 2026]
# research_questions: RQ1..RQ7
```

#### Sign-off output (cấu trúc `phase_0_signoff.json`)

```python
{
    "artifact_version": "COURSEWORK-CONTRACT-v1",
    "phase_id": 0,
    "phase_version": "PHASE-0-v1",
    "created_at": "<ISO UTC timestamp>",
    "environment_id": None,
    "dataset_revision": None,
    "input_paths": ["configs/base/coursework_contract.json"],
    "input_checksums": {...},
    "output_paths": [
        "artifacts/contracts/coursework_contract.json",
        "artifacts/contracts/coursework_contract.sha256"
    ],
    "output_checksums": {...},
    "config_fingerprint": "<sha256>",
    "status": "PASS",
    "tests": ["contract_schema", "option_registry_completeness",
              "chronological_split_sum", "forecast_target_unambiguous",
              "data_independence"],
    "warnings": [],
    "discrepancies": []
}
```

### 2.4. Kết quả & artifacts

| Artifact path | Mô tả |
|---|---|
| `COURSE_WORK/configs/base/coursework_contract.json` | Contract gốc được duyệt |
| `COURSE_WORK/artifacts/contracts/coursework_contract.json` | Bản sao canonical (atomic write) |
| `COURSE_WORK/artifacts/contracts/coursework_contract.sha256` | SHA-256 checksum |
| `COURSE_WORK/artifacts/contracts/phase_0_signoff.json` | Sign-off với status `PASS` |

---

## 3. Phase 1 — Environment

### 3.1. Mục đích

Verify interpreter, notebook kernel, dependency versions, compute device, deterministic configuration và training smoke test. Tạo **environment identity** phục vụ fingerprint cho mọi artifact Phase sau.

### 3.2. Cells trong notebook

#### Cell 4 (markdown): `## Phase 1 - Environment`

> "Verify the interpreter, notebook kernel, dependency versions, compute device, deterministic configuration and training smoke test."

#### Cell 5 (code)

```python
phase_1_signoff = materialize_phase_1(PROJECT_ROOT)
display(render_phase_summary(1, PROJECT_ROOT))
```

Cell này gọi `materialize_phase_1` (từ `utils/environment.py`) và hiển thị HTML summary qua `render_phase_summary` (từ `reporting/phase_summary.py`).

### 3.3. Outputs đã ghi nhận

**Environment overview (từ HTML summary):**

| Field | Value |
|---|---|
| Python | 3.10.11 |
| Kernel | python3 |
| Kernel matches interpreter | PASS |
| Selected device | cpu |
| Deterministic mode | D0 |
| Default dtype | torch.float32 |

**Core package versions:**

| Package | Version |
|---|---|
| ipykernel | 7.3.0 |
| jupyter | 1.1.1 |
| matplotlib | 3.10.9 |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| scikit_learn | 1.7.2 |
| torch | 2.13.0 |

**Status badge:** `PASS` (ENV-v1)

### 3.4. Code Python liên quan: `src/course_work/utils/environment.py` (317 dòng)

#### Public API

| Function | Mục đích |
|---|---|
| `select_device()` | Phát hiện CUDA → MPS → CPU cho PyTorch, trả về string. |
| `resolve_kernel_contract()` | Xác minh kernel khớp với interpreter. |
| `package_versions()` | Dict các phiên bản package core. |
| `environment_inventory()` | Tổng hợp interpreter, OS, platform, working dir, GPU info. |
| `environment_identity()` | Tạo canonical env identity (hash-stable). |
| `environment_identity_differences()` | So sánh hai identity. |
| `device_smoke_test()` | Smoke test tensor alloc + autograd + module + optimizer. |
| `dependency_freeze()` | Snapshot dependency versions cho reproducibility. |
| `materialize_phase_1(project_root)` | Entry point Phase 1: env audit + smoke test + dependency freeze + sign-off `phase_1_signoff.json`. |

#### Artifact ownership

Theo `architecture_rule.md`:

```
utils/environment.py sở hữu:
- Interpreter inventory
- Package-version inventory
- Platform inventory
- Working-directory validation
- CUDA, MPS và CPU detection
- Device selection
- Tensor, autograd, module và optimizer smoke tests
- Environment report
```

### 3.5. Kết quả & artifacts

| Artifact path | Mô tả |
|---|---|
| `COURSE_WORK/artifacts/environment/phase_1_signoff.json` | Sign-off ENV-v1 |
| `COURSE_WORK/artifacts/environment/*.json` | Env inventory, identity, freeze |

---

## 4. Phase 2 — Data Acquisition

### 4.1. Mục đích

Verify canonical UCI provenance, immutable raw bytes, archive membership, dataset metadata và acquisition sign-off **mà không fetch data từ notebook**. Phase này chỉ verify artifact đã tồn tại từ setup offline trước.

### 4.2. Cells trong notebook

#### Cell 6 (markdown): `## Phase 2 - Data Acquisition`

> "Verify canonical UCI provenance, immutable raw bytes, archive membership, dataset metadata and acquisition sign-off without fetching data from the notebook."

#### Cell 7 (code)

```python
phase_2_signoff = materialize_phase_2(PROJECT_ROOT)
display(render_phase_summary(2, PROJECT_ROOT))
```

### 4.3. Outputs đã ghi nhận

**Dataset overview (từ HTML summary):**

| Field | Value |
|---|---|
| Dataset | Appliances Energy Prediction |
| Provider | UCI Machine Learning Repository |
| License | CC BY 4.0 |
| Instances | 19735 |
| Reported features | 28 |
| Sampling interval | 10 minutes |

**Status badge:** `PASS` (DATA-v1)

### 4.4. Code Python liên quan: `src/course_work/data/acquisition.py` (309 dòng)

#### Vai trò

Theo architecture rule, Phase 2 sở hữu:

- Official source identity
- Acquisition method
- Archive integrity
- Safe extraction
- Archive và CSV hashing
- Minimal CSV smoke test
- Dataset manifest
- Source metadata
- Acquisition log
- DATA-v1 sign-off eligibility

#### Public API (key)

| Function | Mục đích |
|---|---|
| `VARIABLE_METADATA` | Danh sách các biến với units, descriptions (29 cột). |
| `validate_archive(zip_path, expected_members)` | Verify ZIP integrity và danh sách file thành viên. |
| `extract_expected_member(zip_path, member_name, output_dir)` | Safe extraction chỉ các file mong đợi. |
| `smoke_test_csv(csv_path)` | Kiểm tra tối thiểu CSV (header parse, row count). |
| `materialize_phase_2(project_root)` | Entry point Phase 2: verify checksum + manifest + sign-off `phase_2_signoff.json`. |

#### Ràng buộc

```
Không preprocessing, split, scale, drop column hoặc overwrite raw CSV.
```

### 4.5. Kết quả & artifacts

| Artifact path | Mô tả |
|---|---|
| `COURSE_WORK/data/raw_data/energydata_complete.csv` | Raw CSV (immutable) |
| `COURSE_WORK/data/raw_data/checksums.sha256` | SHA-256 |
| `COURSE_WORK/data/raw_data/dataset_manifest.json` | Manifest dataset |
| `COURSE_WORK/data/raw_data/source_metadata.json` | Metadata nguồn |
| `COURSE_WORK/data/raw_data/variable_metadata.csv` | Metadata biến (29 dòng) |
| `COURSE_WORK/data/raw_data/README_SOURCE.md` | Provenance docs |
| `COURSE_WORK/artifacts/acquisition/*` | Acquisition logs |
| `COURSE_WORK/artifacts/acquisition/phase_2_signoff.json` | Sign-off DATA-v1 |
| Protected raw SHA-256 (Phase 0-5 baseline) | `2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d` |

---

## 5. Phase 3 — Schema Audit

### 5.1. Mục đích

Audit shape, column order, types, missingness, finite values, timestamp parseability và documented differences giữa UCI metadata và canonical raw file.

### 5.2. Cells trong notebook

#### Cell 8 (markdown): `## Phase 3 - Schema Audit`

> "Audit shape, column order, types, missingness, finite values, timestamp parseability and documented differences between UCI metadata and the canonical raw file."

#### Cell 9 (code)

```python
phase_3_signoff = materialize_phase_3(PROJECT_ROOT)
display(render_phase_summary(3, PROJECT_ROOT))
```

### 5.3. Outputs đã ghi nhận

**Schema overview (từ HTML summary):**

| Field | Value |
|---|---|
| Rows | 19735 |
| Columns | 29 |
| Target | Appliances |
| Timestamp | date |
| First timestamp | 2016-01-11 17:00:00 |
| Last timestamp | 2016-05-27 18:00:00 |

**Critical schema checks:**

| Check | Result |
|---|---|
| Missing expected columns | None |
| Unexpected columns | None |
| Duplicate column names | None |
| Non-finite columns | {} |

**Status badge:** `PASS_WITH_WARNING` (SCHEMA-v1) — warning có thể do các khác biệt metadata.

### 5.4. Code Python liên quan: `src/course_work/data/schema.py` (404 dòng)

#### Vai trò

Sở hữu:

- Expected raw schema (`EXPECTED_COLUMNS` = 29 cột từ `VARIABLE_METADATA`)
- Expected-versus-actual comparison
- Raw dtype audit
- Semantic roles: target, time, feature, random_control
- Feature groups: indoor_temperature (T1-T9 trừ T6), indoor_humidity (RH_1-RH_9 trừ RH_6), local_outdoor (T6, RH_6), weather_station (T_out, Press_mm_hg, RH_out, Windspeed, Visibility, Tdewpoint), lighting (lights), random_controls (rv1, rv2)
- Units và descriptions
- Null, all-null và constant audit
- Numeric coercion
- Non-finite audit
- Timestamp parse probe
- Schema fingerprint
- SCHEMA-v1 sign-off eligibility

#### Public API (key)

| Function | Mục đích |
|---|---|
| `EXPECTED_COLUMNS` | List 29 cột mong đợi. |
| `RANDOM_CONTROLS` | `['rv1', 'rv2']` |
| `INDOOR_TEMPERATURE` | `['T1','T2','T3','T4','T5','T7','T8','T9']` |
| `INDOOR_HUMIDITY` | `['RH_1','RH_2','RH_3','RH_4','RH_5','RH_7','RH_8','RH_9']` |
| `LOCAL_OUTDOOR` | `['T6','RH_6']` |
| `WEATHER_STATION` | `['T_out','Press_mm_hg','RH_out','Windspeed','Visibility','Tdewpoint']` |
| `expected_schema()` | Trả về dict role/group/unit/description cho 29 cột. |
| `load_raw_csv(csv_path)` | Đọc CSV qua pandas. |
| `serialized_header(csv_path)` | Lấy header từ CSV stream. |
| `dataframe_fingerprint(dataframe)` | SHA-256 hash toàn bộ DataFrame rows. |
| `audit_raw_dataframe(dataframe)` | Audit chi tiết: missing, unexpected, duplicate, null, constant, non-finite. |
| `materialize_phase_3(project_root)` | Entry point Phase 3. |

#### Ràng buộc

```
Không convert timestamp chính thức và không mutate raw DataFrame.
```

### 5.5. Kết quả & artifacts

| Artifact path | Mô tả |
|---|---|
| `COURSE_WORK/artifacts/schema/schema_manifest.json` | Manifest schema |
| `COURSE_WORK/artifacts/schema/phase_3_signoff.json` | Sign-off SCHEMA-v1 |

---

## 6. Phase 4 — Temporal Integrity Audit

### 6.1. Mục đích

Verify strict timestamp parsing, ordering, duplicates, ten-minute cadence, missing timestamps, continuity segments và window-safety rules.

### 6.2. Cells trong notebook

#### Cell 10 (markdown): `## Phase 4 - Temporal Integrity Audit`

> "Verify strict timestamp parsing, ordering, duplicates, ten-minute cadence, missing timestamps, continuity segments and window-safety rules."

#### Cell 11 (code)

```python
phase_4_signoff = materialize_phase_4(PROJECT_ROOT)
display(render_phase_summary(4, PROJECT_ROOT))
```

### 6.3. Outputs đã ghi nhận

**Temporal coverage (từ HTML summary):**

| Field | Value |
|---|---|
| Rows | 19735 |
| Start | 2016-01-11 17:00:00 |
| End | 2016-05-27 18:00:00 |
| Expected cadence | 10 minutes |
| Coverage completeness | 100.00% |

**Critical temporal checks:**

| Check | Result |
|---|---|
| Duplicate timestamps | 0 |
| Missing timestamps | 0 |
| Gaps | 0 |
| Largest gap in minutes | Not applicable |

**Status badge:** `PASS` (TEMPORAL-v1)

### 6.4. Code Python liên quan: `src/course_work/data/temporal.py` (464 dòng)

#### Vai trò

Sở hữu:

- Strict timestamp parsing (`parse_timestamps_strict`)
- Original-order audit
- Delta classification
- Duplicate timestamp classification
- Grid alignment (10-minute cadence)
- Missing timestamps
- Gap events
- Coverage metrics
- Continuity segmentation (`build_temporal_view` với `continuity_segment_id`)
- Window-safety contract (`is_temporally_valid_window`)
- TEMPORAL-v1 sign-off eligibility

#### Public API (key)

| Function | Mục đích |
|---|---|
| `parse_timestamps_strict(series)` | Parse không chấp nhận ambiguity. |
| `is_temporally_valid_window(...)` | Check window nằm trong một continuity segment. |
| `build_temporal_view(dataframe)` | Tạo temporal view có `continuity_segment_id`. |
| `load_validated_temporal_view(project_root)` | Load temporal view đã sign-off. |
| `materialize_phase_4(project_root)` | Entry point Phase 4. |

#### Ràng buộc

```
Không interpolation, resampling, split hoặc window construction.
```

### 6.5. Kết quả & artifacts

| Artifact path | Mô tả |
|---|---|
| `COURSE_WORK/artifacts/temporal/temporal_manifest.json` | Temporal manifest |
| `COURSE_WORK/artifacts/temporal/phase_4_signoff.json` | Sign-off TEMPORAL-v1 |

---

## 7. Phase 5 — Exploratory Data Analysis (EDA)

### 7.1. Mục đích

EDA được phép trong notebook dưới Human-approved direct exception `CW-PHASE-5-EDA-DIRECT-001`:

- Mô tả target và features
- Tạo EDA-only temporal columns trong `df_eda` (deep copy)
- Visualize temporal pattern
- Tạo descriptive correlation
- Tạo hypothesis cho Phase sau

Phase 5 **KHÔNG ĐƯỢC**:

- Chọn final feature set, final lookback, final loss
- Bật RevIN mặc định
- Xóa spike, thay outlier bằng NaN
- Interpolate sensor hoặc target
- Tune bằng Test
- Kết luận quan hệ nhân quả từ correlation

### 7.2. Setup

#### Cell 12 (markdown): `## Phase 5 - Exploratory Data Analysis`

#### Cell 13 (code) — Setup

```python
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def display_figure(figure):
    buffer = BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(figure)
    display(Image(data=buffer.getvalue()))

sns.set_theme(style="whitegrid")
phase_5_signoff = materialize_phase_5(PROJECT_ROOT)
phase_5_schema = read_json(PROJECT_ROOT / "artifacts/schema/schema_manifest.json")
eda_temporal_view = load_validated_temporal_view(PROJECT_ROOT)
df = eda_temporal_view[phase_5_schema["ordered_columns"]].copy(deep=True)
```

Cell này gọi `materialize_phase_5` (từ `reporting/eda.py`), nạp schema manifest và validated temporal view, sau đó tạo deep copy `df` cho EDA-only operations.

### 7.3. Sub-cells Phase 5

#### 5.1 Data Overview

**Cell 14 (markdown):** `### 5.1 Data Overview`

**Cell 15 (code):**

```python
data_overview_sample = df.head(8)
data_overview_statistics = df.describe(include="all").transpose()
data_overview_statistics.insert(0, "dtype", df.dtypes.reindex(data_overview_statistics.index).astype(str))
data_overview_memory_mib = df.memory_usage(deep=True).sum() / (1024 ** 2)
display(render_dataframe_table(...))
```

**Outputs:** Hiển thị bảng `Representative Data Samples` (8 dòng đầu) và `Descriptive Statistics by Variable` cho 29 cột.

Một số key statistics:

| Variable | dtype | count | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|---|---|
| Appliances | int64 | 19,735 | 97.6950 | 102.5249 | 10 | 50 | 60 | 100 | 1,080 |
| lights | int64 | 19,735 | 3.8019 | 7.9360 | 0 | 0 | 0 | 0 | 70 |
| T1 (°C) | float64 | 19,735 | 21.6866 | 1.6061 | 16.79 | 20.76 | 21.60 | 22.60 | 26.26 |
| RH_out (%) | float64 | 19,735 | 79.7504 | 14.9011 | 24 | 70.33 | 83.67 | 91.67 | 100 |
| rv1 | float64 | 19,735 | 24.9880 | 14.4966 | 0.0053 | 12.4979 | 24.8977 | 37.5838 | 49.9965 |

#### 5.2 Missing-Value Analysis

**Cell 16 (markdown):** `### 5.2 Missing-Value Analysis`

**Cell 17 (code):**

```python
missing_counts = df.isna().sum()
missing_percentage = missing_counts.div(len(df)).mul(100)
missing_df = pd.DataFrame({"Total Missing": missing_counts, "Percentage (%)": missing_percentage})
missing_only = missing_df.loc[missing_df["Total Missing"].gt(0)]
display(missing_df)
display(missing_only if not missing_only.empty else pd.DataFrame({"Result": ["No missing values detected"]}))
figure, axis = plt.subplots(figsize=(15, 6))
sns.heatmap(df.isna().transpose(), cbar=False, cmap="viridis", yticklabels=True, xticklabels=False, ax=axis)
axis.set_title("Missingness Matrix")
```

**Outputs:**
- Bảng missing counts cho 29 cột
- Bảng "No missing values detected" (vì dataset sạch)
- Heatmap "Missingness Matrix" (hình ảnh)

#### 5.3 Calendar Derivations and Sampling Coverage

**Cell 18 (markdown):** `### 5.3 Calendar Derivations and Sampling Coverage`

**Cell 19 (code):**

```python
df["hour"] = df["date"].dt.hour
df["day_of_week"] = df["date"].dt.dayofweek
df["day_name"] = df["date"].dt.day_name()
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
hour_counts = df["hour"].value_counts().sort_index()
day_counts = df["day_name"].value_counts().reindex(day_order)
weekend_counts = df["is_weekend"].value_counts().sort_index().rename(index={0: "Weekday", 1: "Weekend"})
display(df[["date", "hour", "day_of_week", "day_name", "is_weekend"]].head())
display(hour_counts.rename("Samples").to_frame())
display(day_counts.rename("Samples").to_frame())
display(weekend_counts.to_frame())
# 4 figures: hour-of-day, day-of-week, is_weekend, missing-vs-imputed
```

**Outputs:** Bảng `hour`, `day_of_week`, `day_name`, `is_weekend` head; 3 bảng tần suất; 4 hình phân phối.

#### 5.4 Target Distribution

**Cell 20 (markdown):** `### 5.4 Target Distribution`

**Cell 21 (code):**

```python
target_summary = pd.read_csv(eda_table_root / "eda_numeric_summary.csv").query("variable == 'Appliances'")
target_quantiles = pd.read_csv(eda_table_root / "target_quantiles.csv")
display(target_summary)
display(target_quantiles)
display(pd.DataFrame({"Target": ["Appliances"], "Skewness": [df["Appliances"].skew()]}))
display(Image(filename=eda_figure_root / "EDA_01_target_histogram.png"))
display(Image(filename=eda_figure_root / "EDA_02_target_ecdf.png"))
display(Image(filename=eda_figure_root / "EDA_03_target_boxplot.png"))
display(Image(filename=eda_figure_root / "EDA_05_energy_spikes.png"))
```

**Outputs:** Bảng numeric summary cho Appliances, bảng quantiles, bảng skewness, 4 hình:
- `EDA_01_target_histogram.png`
- `EDA_02_target_ecdf.png`
- `EDA_03_target_boxplot.png`
- `EDA_05_energy_spikes.png`

#### 5.5 Target Timeline and Representative Week

**Cell 22 (markdown):** `### 5.5 Target Timeline and Representative Week`

**Cell 23 (code):**

```python
df_time = df.set_index("date")
daily_average = df_time["Appliances"].resample("D").mean()
first_week = df_time.iloc[:1008]
figure, axes = plt.subplots(2, 1, figsize=(15, 10))
axes[0].plot(daily_average.index, daily_average.values, color="teal", linewidth=1.5)
axes[0].set_title("Daily Average Appliances Energy")
axes[1].plot(first_week.index, first_week["Appliances"], color="darkorange", linewidth=1.2)
axes[1].set_title("Appliances Energy During the First Seven Days")
display_figure(figure)
# Display representative week
display(render_dataframe_table(...))
```

**Outputs:** Hình 2-panel (daily average timeline + first 1008 timesteps ≈ 7 days); bảng representative week.

#### 5.6 Calendar Energy Patterns

**Cell 24 (markdown):** `### 5.6 Calendar Energy Patterns`

**Cell 25 (code):**

```python
display(pd.read_csv(eda_table_root / "hourly_energy_profile.csv"))
display(pd.read_csv(eda_table_root / "weekday_energy_profile.csv"))
display(pd.read_csv(eda_table_root / "weekend_energy_profile.csv"))
display(Image(filename=eda_figure_root / "EDA_06_hourly_profile.png"))
display(Image(filename=eda_figure_root / "EDA_07_weekday_profile.png"))
display(Image(filename=eda_figure_root / "EDA_08_weekday_weekend.png"))
display(Image(filename=eda_figure_root / "EDA_09_hour_weekday_heatmap.png"))
```

**Outputs:** 3 bảng (hourly, weekday, weekend energy profile); 4 hình:
- `EDA_06_hourly_profile.png`
- `EDA_07_weekday_profile.png`
- `EDA_08_weekday_weekend.png`
- `EDA_09_hour_weekday_heatmap.png`

#### 5.7 Feature Distributions

**Cell 26 (markdown):** `### 5.7 Feature Distributions`

**Cell 27 (code):**

```python
display(Image(filename=eda_figure_root / "EDA_10_temperature_distributions.png"))
display(Image(filename=eda_figure_root / "EDA_11_humidity_distributions.png"))
display(Image(filename=eda_figure_root / "EDA_12_lights_distribution.png"))
```

**Outputs:** 3 hình:
- `EDA_10_temperature_distributions.png`
- `EDA_11_humidity_distributions.png`
- `EDA_12_lights_distribution.png`

#### 5.8 Numerical Relationships

**Cell 28 (markdown):** `### 5.8 Numerical Relationships`

**Cell 29 (code):**

```python
sensor_columns = [column for column in phase_5_schema["ordered_columns"] if column != "date"]
correlation_matrix = df[sensor_columns].corr(method="pearson")
top_correlations = correlation_matrix["Appliances"].drop("Appliances").abs().sort_values(ascending=False).head(5)
display(top_correlations.rename("Absolute Pearson Correlation").to_frame())
display(pd.read_csv(eda_table_root / "high_correlation_pairs.csv").head(20))
display(Image(filename=eda_figure_root / "EDA_13_correlation_heatmap.png"))
# Inline scatter & boxplot using top correlated feature
```

**Outputs:** Bảng top 5 correlations, bảng high_correlation_pairs (20 rows), hình `EDA_13_correlation_heatmap.png`, inline scatter+boxplot.

#### 5.9 Cross-Correlation at Short Lags

**Cell 30 (markdown):** `### 5.9 Cross-Correlation at Short Lags`

**Cell 31 (code):**

```python
def calculate_segment_cross_correlation(dataframe, feature, target, maximum_lag):
    """Segment-aware cross-correlation to respect continuity_segment_id boundaries."""
    rows = []
    for lag in range(-maximum_lag, maximum_lag + 1):
        feature_parts = []
        target_parts = []
        for _, segment in dataframe.groupby("continuity_segment_id", sort=False):
            paired = pd.concat([segment[feature], segment[target].shift(-lag)], axis=1).dropna()
            feature_parts.append(paired.iloc[:, 0])
            target_parts.append(paired.iloc[:, 1])
        feature_values = pd.concat(feature_parts, ignore_index=True)
        target_values = pd.concat(target_parts, ignore_index=True)
        ...
```

**Outputs:** Hình cross-correlation plot cho top correlated feature.

#### 5.10 Categorical and Numerical Relationships

**Cell 32 (markdown):** `### 5.10 Categorical and Numerical Relationships`

**Cell 33 (code):**

```python
display(df.groupby("is_weekend")["Appliances"].describe().rename(index={0: "Weekday", 1: "Weekend"}))
weekend_groups = [df.loc[df["is_weekend"].eq(value), "Appliances"].to_numpy() for value in [0, 1]]
hour_groups = [df.loc[df["hour"].eq(hour), "Appliances"].to_numpy() for hour in range(24)]
figure, axes = plt.subplots(1, 2, figsize=(17, 6))
axes[0].boxplot(weekend_groups, tick_labels=["Weekday", "Weekend"], ...)
axes[1].boxplot(hour_groups, tick_labels=list(range(24)), ...)
```

**Outputs:** Bảng describe theo is_weekend; hình 2-panel boxplot (weekday/weekend + 24-hour).

#### 5.11 Descriptive IQR Outlier Diagnostics

**Cell 34 (markdown):** `### 5.11 Descriptive IQR Outlier Diagnostics`

**Cell 35 (code):**

```python
def summarize_iqr_outliers(dataframe, columns):
    rows = []
    for column in columns:
        first_quartile = dataframe[column].quantile(0.25)
        third_quartile = dataframe[column].quantile(0.75)
        interquartile_range = third_quartile - first_quartile
        lower_bound = first_quartile - 1.5 * interquartile_range
        upper_bound = third_quartile + 1.5 * interquartile_range
        outlier_mask = dataframe[column].lt(lower_bound) | dataframe[column].gt(upper_bound)
        rows.append({"Feature": column, "Q1": first_quartile, "Q3": third_quartile,
                     "IQR": interquartile_range, "Lower Bound": lower_bound,
                     "Upper Bound": upper_bound, "Outliers": int(outlier_mask.sum()),
                     "Outlier %": outlier_mask.mean() * 100.0})
    return pd.DataFrame(rows)
```

**Outputs:** Bảng IQR outlier summary cho sensor columns; hình outlier counts.

#### 5.12 Isolated Outlier-Smoothing Demonstration

**Cell 36 (markdown):** `### 5.12 Isolated Outlier-Smoothing Demonstration`

**Cell 37 (code):**

```python
df_before_smoothing = df.copy(deep=True)
df_eda_smoothed_demo = df.copy(deep=True)
for column in iqr_sensor_columns:
    bounds = outlier_summary.set_index("Feature").loc[column]
    outlier_mask = df_eda_smoothed_demo[column].lt(bounds["Lower Bound"]) | df_eda_smoothed_demo[column].gt(bounds["Upper Bound"])
    df_eda_smoothed_demo.loc[outlier_mask, column] = np.nan
    df_eda_smoothed_demo[column] = df_eda_smoothed_demo[column].interpolate(method="linear", limit_direction="both")
top_outlier_feature = outlier_summary.iloc[0]["Feature"]
display(pd.DataFrame({"Canonical DataFrame Unchanged": [
    df is df_before_smoothing  # assertion
]}))
# before/after plot for top outlier feature
```

**Outputs:** Bảng assertion (canonical df unchanged); hình before/after smoothing demonstration.

#### 5.13 Time-Series Diagnostics

**Cell 38 (markdown):** `### 5.13 Time-Series Diagnostics`

**Cell 39 (code):**

```python
display(pd.read_csv(eda_table_root / "selected_lag_correlations.csv"))
display(pd.read_csv(eda_table_root / "selected_exogenous_lag_correlations.csv"))
display(Image(filename=eda_figure_root / "EDA_14_target_autocorrelation.png"))
display(Image(filename=eda_figure_root / "EDA_15_rolling_mean.png"))
display(Image(filename=eda_figure_root / "EDA_16_rolling_standard_deviation.png"))
```

**Outputs:** 2 bảng (selected_lag_correlations, selected_exogenous_lag_correlations); 3 hình:
- `EDA_14_target_autocorrelation.png`
- `EDA_15_rolling_mean.png`
- `EDA_16_rolling_standard_deviation.png`

#### 5.14 Extreme Samples, Hypotheses and Sign-Off

**Cell 40 (markdown):** `### 5.14 Extreme Samples, Hypotheses and Sign-Off`

**Cell 41 (code):**

```python
display(pd.read_csv(eda_table_root / "extreme_target_samples.csv"))
display(pd.read_csv(eda_table_root / "eda_hypotheses.csv"))
```

**Outputs:** Bảng extreme target samples, bảng eda_hypotheses (Phase 6 input).

### 7.4. Figures & tables sinh ra

**Tables (CSV/HTML):**

| Tên file | Phase 5.§ | Mô tả |
|---|---|---|
| `eda_numeric_summary.csv` | 5.1, 5.4 | Numeric summary (mean, std, quartiles) cho 29 biến |
| `target_quantiles.csv` | 5.4 | Quantiles của Appliances |
| `hourly_energy_profile.csv` | 5.6 | Mean energy theo hour |
| `weekday_energy_profile.csv` | 5.6 | Mean energy theo weekday |
| `weekend_energy_profile.csv` | 5.6 | Mean energy theo weekday-weekend |
| `high_correlation_pairs.csv` | 5.8 | Top correlated pairs |
| `selected_lag_correlations.csv` | 5.13 | Lag correlations cho Appliances |
| `selected_exogenous_lag_correlations.csv` | 5.13 | Lag correlations cho exogenous features |
| `extreme_target_samples.csv` | 5.14 | Top/bottom extreme samples |
| `eda_hypotheses.csv` | 5.14 | Hypothesis registry cho Phase 6 |
| `eda_anomalies.json` | (cấu hình) | Anomaly registry |
| `eda_discrepancies.json` | (cấu hình) | Discrepancy log |

**Figures (PNG):**

| Tên file | Phase 5.§ | Mô tả |
|---|---|---|
| `EDA_01_target_histogram.png` | 5.4 | Histogram Appliances |
| `EDA_02_target_ecdf.png` | 5.4 | Empirical CDF Appliances |
| `EDA_03_target_boxplot.png` | 5.4 | Boxplot Appliances |
| `EDA_05_energy_spikes.png` | 5.4 | Energy spikes |
| `EDA_06_hourly_profile.png` | 5.6 | Hourly mean profile |
| `EDA_07_weekday_profile.png` | 5.6 | Weekday mean profile |
| `EDA_08_weekday_weekend.png` | 5.6 | Weekday vs Weekend |
| `EDA_09_hour_weekday_heatmap.png` | 5.6 | Hour x Weekday heatmap |
| `EDA_10_temperature_distributions.png` | 5.7 | Indoor temperature dists |
| `EDA_11_humidity_distributions.png` | 5.7 | Indoor humidity dists |
| `EDA_12_lights_distribution.png` | 5.7 | Lights distribution |
| `EDA_13_correlation_heatmap.png` | 5.8 | Correlation heatmap |
| `EDA_14_target_autocorrelation.png` | 5.13 | ACF plot |
| `EDA_15_rolling_mean.png` | 5.13 | Rolling mean |
| `EDA_16_rolling_standard_deviation.png` | 5.13 | Rolling std |

### 7.5. Code Python liên quan

#### `src/course_work/data/eda.py` (443 dòng)

Sở hữu các calculation của Phase 5:

- Derived EDA view (`df_eda`)
- Numeric summary
- Target quantiles
- Hourly và weekday profiles
- Feature summaries
- Correlation tables
- High-correlation pairs
- Selected-lag autocorrelation
- Segment-aware cross-correlation
- Segment-aware rolling statistics
- Extreme-target samples
- Hypothesis registry
- Anomaly registry
- EDA-v1 sign-off eligibility

Key functions:

| Function | Mục đích |
|---|---|
| `build_enhanced_eda_view(...)` | Xây dựng view EDA-enhanced từ validated temporal view. |
| `calculate_numeric_summaries(...)` | Numeric summary table. |
| `compute_target_quantiles(...)` | Quantiles Appliances. |
| `compute_calendar_profiles(...)` | Hourly/weekday profiles. |
| `compute_correlation_tables(...)` | Correlation matrix + top pairs. |
| `compute_segment_cross_correlation(...)` | Segment-aware cross-correlation. |
| `compute_segment_rolling_statistics(...)` | Segment-aware rolling stats. |
| `identify_extreme_samples(...)` | Extreme target samples. |
| `generate_hypotheses(...)` | Hypothesis registry. |
| `materialize_phase_5(...)` | Tên đặt trong reporting.eda thực tế (xem dưới). |

#### `src/course_work/reporting/eda.py` (347 dòng)

Sở hữu **figure rendering** của Phase 5:

- Canonical EDA figure names (danh sách 16 file trên)
- Plot style (`sns.set_theme(style="whitegrid")`, màu sắc, figsize)
- Axis labels
- Figure export (PNG qua `BytesIO`)
- Figure checksum handoff

Chỉ nhận validated tables hoặc validated derived views từ `data/eda.py`. Không tự tính lại scientific tables theo logic khác.

Key API:

| Function | Mục đích |
|---|---|
| `EDA_FIGURE_FILENAMES` | Tuple các tên file figure canonical. |
| `configure_plot_style()` | Set matplotlib defaults. |
| `save_figure(figure, filename, root)` | Save PNG và checksum. |
| `plot_target_histogram/cdf/boxplot/spikes(...)` | Các plotter cho target. |
| `plot_calendar_profiles(...)` | Hourly/weekday plotter. |
| `plot_feature_distributions(...)` | Temperature/humidity/lights. |
| `plot_correlation_heatmap(...)` | Heatmap. |
| `plot_target_autocorrelation(...)` | ACF. |
| `plot_rolling_statistics(...)` | Rolling mean/std. |
| `materialize_phase_5(project_root)` | Entry point Phase 5 (gọi từ cell 13). |

### 7.6. Kết quả & artifacts

| Artifact path | Mô tả |
|---|---|
| `COURSE_WORK/artifacts/eda/tables/*.csv` | Tất cả bảng EDA |
| `COURSE_WORK/artifacts/eda/figures/*.png` | 16 figure PNG |
| `COURSE_WORK/artifacts/eda/eda_manifest.json` | Manifest EDA-v1 |
| `COURSE_WORK/artifacts/eda/phase_5_signoff.json` | Sign-off EDA-v1 |

---

## 8. Phase 6 — Feature Engineering

### 8.1. Mục đích

Xây dựng deterministic **FEATURES-v1 view** bằng cách thêm 5 calendar features vào validated Phase 4 temporal view, **không thêm manual lag/rolling model features** và **không interpolation**. Phase 6 chỉ verify signed DATA-v1, SCHEMA-v1, TEMPORAL-v1 và EDA-v1 inputs.

### 8.2. Cells trong notebook

#### Cell 42 (markdown): `## Phase 6 - Feature Engineering`

#### Cell 43 (code)

```python
phase_6_signoff = materialize_phase_6(PROJECT_ROOT)
display(render_phase_summary(6, PROJECT_ROOT))
```

### 8.3. Outputs đã ghi nhận

**Feature overview (từ HTML summary):**

| Field | Value |
|---|---|
| Rows | 19,735 |
| Columns | 37 |
| Raw features | 27 |
| Engineered features | 5 |
| Metadata columns | 4 |

**Engineered feature registry:**

| Feature | Formula version |
|---|---|
| hour_sin | TIME-FEATURES-v1 |
| hour_cos | TIME-FEATURES-v1 |
| dow_sin | TIME-FEATURES-v1 |
| dow_cos | TIME-FEATURES-v1 |
| weekend | TIME-FEATURES-v1 |

**Status badge:** `PASS` (FEATURES-v1)

### 8.4. Code Python liên quan: `src/course_work/data/features.py` (438 dòng)

#### Vai trò

Sở hữu toàn bộ canonical Phase 6 feature engineering:

- Verify signed DATA-v1, SCHEMA-v1, TEMPORAL-v1 và EDA-v1 inputs
- Build deterministic FEATURES-v1 view
- Create `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` và `weekend` (cyclical + binary)
- Preserve raw values, target, timestamps, row lineage và continuity segments
- Create feature registry, lineage, availability và leakage audits
- Write the derived CSV, manifest, checksum, discrepancy log và Phase 6 sign-off

#### Key API

| Function | Mục đích |
|---|---|
| `add_time_features(dataframe)` | Thêm 5 calendar features cyclical. |
| `build_feature_view(...)` | Xây dựng FEATURES-v1 view từ temporal view. |
| `validate_feature_invariants(...)` | Đảm bảo không leakage, không mất raw lineage. |
| `materialize_phase_6(project_root)` | Entry point Phase 6. |

#### Ràng buộc

```
Không sở hữu:
- Final feature-set selection
- Chronological split
- Scaling hoặc imputation
- Manual lag hoặc rolling model features
- Window construction
- Training
```

### 8.5. Kết quả & artifacts

| Artifact path | Mô tả |
|---|---|
| `COURSE_WORK/data/interim/uci_appliances_energy_prediction/energydata_feature_engineered_v1.csv` | Derived CSV master table |
| `COURSE_WORK/artifacts/features/feature_engineering_manifest.json` | Manifest |
| `COURSE_WORK/artifacts/features/feature_engineered_v1.sha256` | Checksum |
| `COURSE_WORK/artifacts/features/feature_engineering_discrepancies.json` | Discrepancies log |
| `COURSE_WORK/artifacts/features/phase_6_signoff.json` | Sign-off FEATURES-v1 |

---

## 9. Phase 7 — Feature Set Variants

### 9.1. Mục đích

Tạo 6 ordered variants: **FS0/FS1/FS2 × TF0/TF1** dựa trên immutable feature components. **Không transform feature values**, chỉ tham chiếu FEATURES-v1 master table và ghi ordered registries dưới `artifacts/feature_sets`.

### 9.2. Cells trong notebook

#### Cell 44 (markdown): `## Phase 7 - Feature-Set Variants`

#### Cell 45 (code)

```python
phase_7_signoff = materialize_phase_7(PROJECT_ROOT)
display(render_phase_summary(7, PROJECT_ROOT))
```

### 9.3. Outputs đã ghi nhận

**Feature-set registry (từ HTML summary):**

| Variant | Feature count | Baseline |
|---|---|---|
| FS0_TF0 | 25 | FAIL |
| FS0_TF1 | 30 | FAIL |
| FS1_TF0 | 26 | FAIL |
| FS1_TF1 | 31 | PASS |
| FS2_TF0 | 28 | FAIL |
| FS2_TF1 | 33 | FAIL |

`Baseline` cột ở đây là baseline-eligibility cho persistence baseline (FS1_TF1 = PASS, các variant khác = FAIL vì không chứa historical target feature).

**Status badge:** `PASS` (FEATURESETS-v1)

### 9.4. Code Python liên quan: `src/course_work/data/feature_sets.py` (574 dòng)

#### Vai trò

Sở hữu:

- Verify signed FEATURES-v1 inputs và checksums
- Define immutable ordered feature components: exogenous, historical_target, random_controls, time_features, metadata
- Build FS0/FS1/FS2 kết hợp TF0/TF1 (6 variants)
- Validate metadata, historical-target, random-control và time-feature isolation
- Validate numeric compatibility, missingness và finite values
- Compute deterministic ordered-feature fingerprints
- Write FEATURESETS-v1 registry, lineage, audits, manifest và sign-off
- Expose defensive-copy feature-list lookup

#### Key API

| Function | Mục đích |
|---|---|
| `FEATURE_COMPONENTS` | Dict các component groups (exogenous, historical_target, random_controls, time_features, metadata). |
| `build_feature_set(variant, time_feature)` | Build ordered feature list cho variant × time_feature. |
| `validate_feature_set(variant, time_feature)` | Validate contract (numeric, missingness, finite). |
| `compute_feature_set_fingerprint(...)` | SHA-256 của ordered features. |
| `materialize_phase_7(project_root)` | Entry point Phase 7. |

#### Ràng buộc

```
Không sở hữu:
- Feature value transformation
- Feature-set winner selection
- Chronological split
- Scaling hoặc imputation
- Window construction
- Training
```

### 9.5. Kết quả & artifacts

| Artifact path | Mô tả |
|---|---|
| `COURSE_WORK/artifacts/feature_sets/feature_components.json` | Component groups |
| `COURSE_WORK/artifacts/feature_sets/feature_set_registry.json` | Registry 6 variants |
| `COURSE_WORK/artifacts/feature_sets/feature_set_manifest.json` | Manifest |
| `COURSE_WORK/artifacts/feature_sets/feature_set_discrepancies.json` | Discrepancies |
| `COURSE_WORK/artifacts/feature_sets/phase_7_signoff.json` | Sign-off FEATURESETS-v1 |

---

## 10. Phase 8 — Chronological Split

### 10.1. Mục đích

Compute deterministic floor-based **70/15/15** chronological split theo `target_timestamp`. Preserve one full master timeline cho WB0 context carry-over. Prepare WB1 strict-isolation metadata cho downstream comparison.

### 10.2. Cells trong notebook

#### Cell 46 (markdown): `## Phase 8 - Chronological Split`

#### Cell 47 (code)

```python
phase_8_signoff = materialize_phase_8(PROJECT_ROOT)
display(render_phase_summary(8, PROJECT_ROOT))
```

### 10.3. Outputs đã ghi nhận

**Chronological membership (từ HTML summary):**

| Split | Rows | Ratio | Start | End | Duration (min) |
|---|---|---|---|---|---|
| Train | 13,814 | 70.00% | 2016-01-11 17:00:00 | 2016-04-16 15:10:00 | 138,130 |
| Validation | 2,960 | 15.00% | 2016-04-16 15:20:00 | 2016-05-07 04:30:00 | 29,590 |
| Test | 2,961 | 15.00% | 2016-05-07 04:40:00 | 2016-05-27 18:00:00 | 29,600 |

**Status badge:** `PASS` (SPLIT-v1)

### 10.4. Code Python liên quan: `src/course_work/data/splitting.py` (503 dòng)

#### Vai trò

Sở hữu:

- Verify signed FEATURES-v1, FEATURESETS-v1 và TEMPORAL-v1 inputs
- Compute deterministic floor-based 70/15/15 boundaries (`compute_split_boundaries`)
- Assign TRAIN, VALIDATION và TEST row membership (`build_chronological_membership`)
- Preserve one full master timeline for WB0 context carry-over
- Prepare WB1 strict-isolation metadata for downstream comparison
- Audit chronology, coverage, disjointness và Test firewall
- Compute per-split và global membership fingerprints
- Create Train/Validation-only descriptive diagnostics
- Create timestamp-only split timeline (`figures/SPLIT_01_timeline.png`)
- Write SPLIT-v1 manifest, artifacts và sign-off

#### Key API

| Function | Mục đích |
|---|---|
| `SPLIT_RATIOS` | `{'train': 0.7, 'validation': 0.15, 'test': 0.15}` |
| `compute_split_boundaries(row_count)` | Floor-based boundaries. |
| `build_chronological_membership(row_count)` | Member labels. |
| `build_split_fingerprints(...)` | Hash per-split. |
| `audit_splits(...)` | Test firewall. |
| `materialize_phase_8(project_root)` | Entry point Phase 8. |

#### Ràng buộc

```
Không sở hữu:
- Scaling hoặc imputation
- Feature-set winner selection
- Window construction
- DataLoader construction
- Training hoặc evaluation metrics
- Detailed Test distribution analysis trước Phase 47
```

### 10.5. Kết quả & artifacts

| Artifact path | Mô tả |
|---|---|
| `COURSE_WORK/artifacts/splits/split_manifest.json` | Manifest |
| `COURSE_WORK/artifacts/splits/split_discrepancies.json` | Discrepancies |
| `COURSE_WORK/artifacts/splits/figures/SPLIT_01_timeline.png` | Timestamp-only timeline |
| `COURSE_WORK/artifacts/splits/phase_8_signoff.json` | Sign-off SPLIT-v1 |

---

## 11. Phase 9 — Train-Only Scaling

### 11.1. Mục đích

Fit **6 variant-specific X StandardScaler bundles** chỉ trên TRAIN rows. Scale continuous channels, giữ cyclical/binary channels pass-through. Fit **1 YS1 target scaler** chỉ trên TRAIN target period (YS0 = identity).

### 11.2. Cells trong notebook

#### Cell 48 (markdown): `## Phase 9 - Train-Only Scaling`

#### Cell 49 (code)

```python
phase_9_signoff = materialize_phase_9(PROJECT_ROOT)
display(render_phase_summary(9, PROJECT_ROOT))
```

### 11.3. Outputs đã ghi nhận

**X scaler bundles (từ HTML summary):**

| Variant | Features | Scaled | Pass-through | Status |
|---|---|---|---|---|
| FS0_TF0 | 25 | 25 | 0 | PASS |
| FS0_TF1 | 30 | 25 | 5 | PASS |
| FS1_TF0 | 26 | 26 | 0 | PASS |
| FS1_TF1 | 31 | 26 | 5 | PASS |
| FS2_TF0 | 28 | 28 | 0 | PASS |
| FS2_TF1 | 33 | 28 | 5 | PASS |

**Target scaling options:**

| Option | Method | Fit split | Inverse transform |
|---|---|---|---|
| YS0 | Identity | TRAIN | Identity |
| YS1 | StandardScaler | TRAIN | Required before Wh metrics |

**Status badge:** `PASS` (SCALING-v1)

### 11.4. Code Python liên quan: `src/course_work/data/scaling.py` (875 dòng)

#### Vai trò

Sở hữu:

- Verify signed FEATURES-v1, FEATURESETS-v1 và SPLIT-v1 inputs
- Fit sáu variant-specific X StandardScaler bundles chỉ trên TRAIN rows
- Scale continuous channels và giữ cyclical/binary channels pass-through
- Fit một YS1 target scaler chỉ trên TRAIN target period
- Expose YS0 identity cùng target inverse-transform utility
- Bind scaler với feature order, feature fingerprint và split fingerprint
- Audit Train standardization, Validation transform và structural-only Test transform
- Serialize trusted local scaler artifacts (`*.joblib`), statistics, checksums, manifest và sign-off

#### Key API

| Function | Mục đích |
|---|---|
| `ScalingPolicy` | Define per-variant scaling policy. |
| `fit_x_scaler(variant, time_feature, train_rows)` | StandardScaler fit trên TRAIN. |
| `fit_y_scaler(target_train)` | YS1 scaler. |
| `transform_x(scaler_bundle, features)` | Apply scaler. |
| `inverse_transform_y(scaler_bundle, y_scaled)` | Inverse YS1. |
| `materialize_phase_9(project_root)` | Entry point Phase 9. |

#### Ràng buộc

```
Không sở hữu:
- Window construction
- DataLoader construction
- Feature-set hoặc target-scaling winner selection
- Model training hoặc metrics
- Detailed Test distribution inspection
```

### 11.5. Kết quả & artifacts

| Artifact path | Mô tả |
|---|---|
| `COURSE_WORK/artifacts/scalers/x/XSCALER__FS0_TF0__SCALING-v1.joblib` | X scaler FS0_TF0 |
| `COURSE_WORK/artifacts/scalers/x/XSCALER__FS0_TF1__SCALING-v1.joblib` | X scaler FS0_TF1 |
| `COURSE_WORK/artifacts/scalers/x/XSCALER__FS1_TF0__SCALING-v1.joblib` | X scaler FS1_TF0 |
| `COURSE_WORK/artifacts/scalers/x/XSCALER__FS1_TF1__SCALING-v1.joblib` | X scaler FS1_TF1 |
| `COURSE_WORK/artifacts/scalers/x/XSCALER__FS2_TF0__SCALING-v1.joblib` | X scaler FS2_TF0 |
| `COURSE_WORK/artifacts/scalers/x/XSCALER__FS2_TF1__SCALING-v1.joblib` | X scaler FS2_TF1 |
| `COURSE_WORK/artifacts/scalers/y/YSCALER__YS1__SCALING-v1.joblib` | Y target scaler |
| `COURSE_WORK/artifacts/scaling/scaler_registry.json` | Registry |
| `COURSE_WORK/artifacts/scaling/scaler_checksums.sha256` | Checksums |
| `COURSE_WORK/artifacts/scaling/scaling_manifest.json` | Manifest |
| `COURSE_WORK/artifacts/scaling/scaling_discrepancies.json` | Discrepancies |
| `COURSE_WORK/artifacts/scaling/phase_9_signoff.json` | Sign-off SCALING-v1 |

---

## 12. Phase 10 — Window Builder

### 12.1. Mục đích

Build deterministic **L36, L72, L144** native window indices cho **H1** (horizon=1). Validate timestamp cadence, continuity segments, target exclusion và sequence direction. Assign sample split bằng target timestamp. Register WB0 context carry-over và WB1 strict-isolation eligibility. Lock **WINDOWPOP-v1 common target population** cho controlled comparisons.

### 12.2. Cells trong notebook

#### Cell 50 (markdown): `## Phase 10 - Window Builder`

#### Cell 51 (code)

```python
phase_10_signoff = materialize_phase_10(PROJECT_ROOT)
display(render_phase_summary(10, PROJECT_ROOT))
```

### 12.3. Outputs đã ghi nhận

**Window contract (từ HTML summary):**

| Field | Value |
|---|---|
| Lookbacks | 36, 72, 144 |
| Horizon | 1 step / 10 minutes |
| Boundary protocol | WB0_CONTEXT_CARRY_OVER |
| Sequence direction | oldest_to_newest |
| Test target access | LOCKED_UNTIL_PHASE_47 |

**Common target population:**

| Split | Targets |
|---|---|
| Train | 13,670 |
| Validation | 2,960 |
| Test | 2,961 |
| Total | 19,591 |

**Status badge:** `PASS` (WINDOWS-v1 + WINDOWPOP-v1)

### 12.4. Code Python liên quan: `src/course_work/data/windows.py` (913 dòng)

#### Vai trò

Sở hữu:

- Verify signed TEMPORAL-v1, FEATURES-v1, FEATURESETS-v1, SPLIT-v1 và SCALING-v1 inputs
- Build deterministic L36, L72 và L144 native window indices cho H1
- Validate timestamp cadence, continuity segments, target exclusion và sequence direction
- Assign sample split bằng target timestamp
- Register WB0 context carry-over và WB1 strict-isolation eligibility
- Lock WINDOWPOP-v1 common target population cho controlled comparisons
- Transform frozen SCALING-v1 feature timelines và materialize deterministic probes lazily
- Preserve Test target firewall và không export target values
- Write window index, population, audits, fingerprints, manifest, README và Phase 10 sign-off

#### Key API

| Function | Mục đích |
|---|---|
| `LOOKBACKS` | `[36, 72, 144]` |
| `compute_window_bounds(lookback, total_rows)` | Sample id bounds. |
| `generate_window_sample_ids(...)` | Sample ids. |
| `validate_temporal_windows(...)` | Check continuity. |
| `build_native_window_indices(lookback)` | Native indices. |
| `build_common_window_indices(lookback)` | Common (intersection of L36/L72/L144). |
| `materialize_phase_10(project_root)` | Entry point Phase 10. |

#### Ràng buộc

```
Không sở hữu:
- PyTorch Dataset hoặc DataLoader
- Model training hoặc evaluation metrics
- Lookback winner selection
- Full 3D tensor persistence
- Test target outcome analysis trước Phase 47
```

### 12.5. Kết quả & artifacts

| Artifact path | Mô tả |
|---|---|
| `COURSE_WORK/artifacts/windows/window_manifest.json` | Manifest |
| `COURSE_WORK/artifacts/windows/window_fingerprints.json` | Fingerprints |
| `COURSE_WORK/artifacts/windows/window_discrepancies.json` | Discrepancies |
| `COURSE_WORK/artifacts/windows/phase_10_signoff.json` | Sign-off WINDOWS-v1 + WINDOWPOP-v1 |

---

## 13. Phase 11 — DataLoaders

### 13.1. Mục đích

Build **map-style `SequenceWindowDataset`** bằng lazy slicing từ read-only float32 feature timeline. Expose TRAIN, VALIDATION, TEST_LOCKED và explicit Phase 47 TEST_EVALUATION target access modes. Build split-specific DataLoaders cho B32 và B64. Shuffle TRAIN reproducibly, giữ VALIDATION/TEST chronological.

### 13.2. Cells trong notebook

#### Cell 52 (markdown): `## Phase 11 - DataLoaders`

#### Cell 53 (code)

```python
phase_11_signoff = materialize_phase_11(PROJECT_ROOT)
display(render_phase_summary(11, PROJECT_ROOT))
```

### 13.3. Outputs đã ghi nhận

**Dataset population (từ HTML summary):**

| Split | Samples | Target access |
|---|---|---|
| Train | 13,670 | Available |
| Validation | 2,960 | Available |
| Test | 2,961 | Locked until Phase 47 |

**Loader policy:**

| Split | Batch size | Shuffle | Drop last | Workers |
|---|---|---|---|---|
| Train | 64 | Yes | No | 0 |
| Validation | 64 | No | No | 0 |
| Test | 64 | No | No | 0 |

**Status badge:** `PASS` (DATALOADERS-v1)

### 13.4. Code Python liên quan: `src/course_work/data/datasets.py` (1014 dòng)

#### Vai trò

Sở hữu:

- Verify signed FEATURESETS-v1, SPLIT-v1, SCALING-v1, WINDOWS-v1 và WINDOWPOP-v1 inputs
- Build map-style `SequenceWindowDataset` bằng lazy slicing từ read-only float32 feature timeline
- Bind Dataset với variant, lookback, target option, window fingerprint và population fingerprint
- Expose TRAIN, VALIDATION, TEST_LOCKED và explicit Phase 47 TEST_EVALUATION target access modes
- Return batch-first CPU tensors với stable int64 `sample_idx`
- Build split-specific DataLoaders cho B32 và B64
- Shuffle TRAIN reproducibly và giữ VALIDATION/TEST chronological
- Use `drop_last=False`, separate split generators và canonical worker seed utility
- Apply CUDA-only pin-memory policy và keep device transfer outside Dataset
- Audit batch shapes, dtypes, coverage, ordering, reproducibility và Test firewall
- Write registries, audits, device policy, manifest, README và Phase 11 sign-off

#### Key API

| Class / Function | Mục đích |
|---|---|
| `SequenceWindowDataset` | PyTorch map-style dataset lazy-load. |
| `DatasetConfig` | Bind variant, lookback, target option, window fingerprint, population fingerprint. |
| `build_dataloader(dataset, split, batch_size)` | PyTorch DataLoader với split-specific policy. |
| `materialize_phase_11(project_root)` | Entry point Phase 11. |

#### Ràng buộc

```
Không sở hữu:
- Window construction hoặc scaler fitting
- Sample population selection
- Model training hoặc metric calculation
- Batch-size winner selection
- Serialized Dataset, DataLoader hoặc full 3D tensors
- Test target evaluation trước explicit Phase 47 gate
```

### 13.5. Kết quả & artifacts

| Artifact path | Mô tả |
|---|---|
| `COURSE_WORK/artifacts/dataloaders/dataloader_manifest.json` | Manifest |
| `COURSE_WORK/artifacts/dataloaders/device_transfer_policy.json` | Device policy |
| `COURSE_WORK/artifacts/dataloaders/dataloader_discrepancies.json` | Discrepancies |
| `COURSE_WORK/artifacts/dataloaders/phase_11_signoff.json` | Sign-off DATALOADERS-v1 |

---

## 14. Phase 12 — Shared Metrics

### 14.1. Mục đích

Define metric contract & implementation: **MAE Wh**, **RMSE Wh**, **R²**. Compute trên full aligned split population một lần ở đơn vị Wh gốc (YS0/YS1 inverse). Enforce residual convention = actual − prediction. Enforce sample index uniqueness, population coverage và chronological canonical order. Aggregate epoch loss theo sample count. **Test metrics not computed cho đến khi Phase 47 mở FINAL_TEST gate.**

### 14.2. Cells trong notebook

#### Cell 54 (markdown): `## Phase 12 - Shared Metrics`

#### Cell 55 (code)

```python
phase_12_signoff = materialize_phase_12(PROJECT_ROOT)
display(render_phase_summary(12, PROJECT_ROOT))
```

### 14.3. Outputs đã ghi nhận

**Metric registry (từ HTML summary):**

| Metric | Field | Direction | Unit | Role |
|---|---|---|---|---|
| MAE | mae_wh | Lower | Wh | Required |
| RMSE | rmse_wh | Lower | Wh | Primary selection |
| R² | r2 | Higher | Dimensionless | Required |

**Evaluation policy:**

| Policy | Value |
|---|---|
| Selection split | Validation |
| Prediction scale | Original Wh |
| Aggregation | Full aligned split once |
| Residual | Actual minus prediction |
| Test metrics in Phase 12 | Not computed |
| Test targets in Phase 12 | Not materialized |

**Status badge:** `PASS` (METRICS-v1)

### 14.4. Code Python liên quan: `src/course_work/evaluation/metrics.py` (922 dòng)

#### Vai trò

Sở hữu:

- Verify signed SCALING-v1, WINDOWS-v1, WINDOWPOP-v1 và DATALOADERS-v1 inputs
- Normalize single-output arrays có shape N hoặc N x 1 về NumPy float64
- Convert YS0 identity và inverse-transform YS1 bằng frozen Train-only target scaler
- Compute MAE Wh, RMSE Wh và R² trên full aligned split population một lần
- Preserve negative R² và explicit undefined states cho constant target hoặc N nhỏ hơn 2
- Enforce residual bằng actual trừ prediction
- Enforce sample index uniqueness, population coverage và chronological canonical order
- Aggregate epoch loss theo sample count
- Compare baseline và model chỉ khi contract, split, population, unit, horizon và sample count khớp
- Enforce FINAL_TEST mode cùng model lock id trước mọi Test metric
- Write metric contract, registry, schemas, audits, reference examples, manifest, README và Phase 12 sign-off

#### Key API

| Function | Mục đích |
|---|---|
| `METRIC_REGISTRY` | Dict 3 metrics (mae_wh, rmse_wh, r2). |
| `compute_mae(y_true, y_pred)` | MAE. |
| `compute_rmse(y_true, y_pred)` | RMSE. |
| `compute_r2(y_true, y_pred)` | R² (preserve negative). |
| `compute_metrics(y_true, y_pred, target_scaler=None)` | Full aligned split. |
| `align_predictions(prediction_bundle)` | Schema-aligned bundle. |
| `aggregate_epoch_loss(losses, counts)` | Sample-weighted. |
| `materialize_phase_12(project_root)` | Entry point Phase 12. |

#### Ràng buộc

```
Không sở hữu:
- Model training hoặc model selection run
- Prediction generation
- Persistence baseline implementation
- Test prediction hoặc Test target materialization trong Phase 12
- Batch-mean RMSE hoặc batch-mean R² aggregation
- Prediction clipping hoặc rounding trước metric
```

### 14.5. Kết quả & artifacts

| Artifact path | Mô tả |
|---|---|
| `COURSE_WORK/artifacts/metrics/metric_contract.json` | Metric contract |
| `COURSE_WORK/artifacts/metrics/metric_comparison_schema.json` | Comparison schema |
| `COURSE_WORK/artifacts/metrics/prediction_bundle_schema.json` | Prediction schema |
| `COURSE_WORK/artifacts/metrics/metric_manifest.json` | Manifest |
| `COURSE_WORK/artifacts/metrics/metric_discrepancies.json` | Discrepancies |
| `COURSE_WORK/artifacts/metrics/phase_12_signoff.json` | Sign-off METRICS-v1 |

---

## 15. Phase 13 — Experiment Registry

### 15.1. Mục đích

Canonical **EXPERIMENTS-v1** registry contract với unique run ID, canonical config fingerprint (SHA-256), complete lineage (parent-child), lifecycle states (REGISTERED, RUNNING, COMPLETED, FAILED, CANCELLED, INVALIDATED, ARCHIVED), artifact/metric links, sweep consistency, completed-config immutability và **final-Test firewall**. **Production registry khởi tạo với zero fabricated runs.**

### 15.2. Cells trong notebook

#### Cell 56 (markdown): `## Phase 13 - Experiment Registry`

#### Cell 57 (code)

```python
phase_13_signoff = materialize_phase_13(PROJECT_ROOT)
display(render_phase_summary(13, PROJECT_ROOT))
```

### 15.3. Outputs đã ghi nhận

**Registry state (từ HTML summary):**

| Item | Value |
|---|---|
| Production runs | 1 |
| Experiment families | 26 |
| Registered sweeps | 0 |
| Completed runs | 1 |
| Failed runs | 0 |

**Core safeguards:**

| Safeguard | State |
|---|---|
| Unique run identity | Enabled |
| Canonical config fingerprint | Enabled |
| Atomic registry writes | Enabled |
| Completed config immutability | Enabled |
| Sweep consistency guard | Enabled |
| Final Test firewall | Enabled |

**Status badge:** `PASS` (EXPERIMENTS-v1)

### 15.4. Code Python liên quan: `src/course_work/experiments/registry.py` (1735 dòng)

#### Vai trò

Sở hữu:

- Verify signed contracts từ ENV-v1 đến METRICS-v1
- Canonicalize nested run config và tạo deterministic SHA-256 config fingerprint (`compute_config_fingerprint`)
- Validate data, model, training, reproducibility, runtime và upstream lineage fields (`build_registry_contract`)
- Load upstream context (`load_upstream_context`)
- Build reference run config (`build_reference_run_config`)
- Allocate unique run ID theo model family, experiment family, sequence và config hash (`register_sweep`, `register_run`)
- Start run (`start_run`)
- Register artifacts và metrics (`register_artifact`, `register_metric`)
- Complete / fail / cancel / invalidate run (`complete_run`, `fail_run`, `cancel_run`, `invalidate_run`)
- Compare runs (`compare_runs`)
- Validate registry (`validate_registry`)
- Require canonical rerun reason cho duplicate config
- Enforce parent-child lineage và final-model-lock references
- Reject development Test targets và Test metrics
- Allow final Test chỉ với FINAL_TEST family, execution type, lock id và authorization
- Validate one-factor sweep consistency
- Write canonical JSONL, derived CSV views, families, audits, manifest, index, README và Phase 13 sign-off
- Initialize Phase 13 production registry without fabricated runs or results
- Allow Phase 14+ owners to register real runs through public lifecycle APIs

#### Key API

| Enum / Class | Mục đích |
|---|---|
| `RunStatus` | Enum: REGISTERED, RUNNING, COMPLETED, FAILED, CANCELLED, INVALIDATED, ARCHIVED. |
| `ExecutionType` | TRAINING, EVALUATION, BASELINE_EVAL. |
| `FailureType` | Enum failure causes. |
| `ArtifactType` | Enum: CONFIG, METRICS, PREDICTION, AUDIT, etc. |
| `compute_config_fingerprint(config)` | SHA-256 nested config. |
| `build_registry_contract(...)` | Validate config schema. |
| `load_upstream_context(project_root)` | Load ENV..METRICS sign-offs. |
| `build_reference_run_config(...)` | Build reference config. |
| `register_sweep(...)`, `register_run(...)` | Register new sweep/run. |
| `start_run(run_id)` | Transition REGISTERED→RUNNING. |
| `register_artifact(run_id, artifact_type, path, checksum, size)` | Register artifact. |
| `register_metric(run_id, metric_id, value, unit, sample_count)` | Register metric row. |
| `complete_run(run_id)` | Transition RUNNING→COMPLETED. |
| `fail_run(run_id, failure_type, evidence)` | Transition→FAILED. |
| `cancel_run(run_id)`, `invalidate_run(run_id)` | Other transitions. |
| `compare_runs(run_id_a, run_id_b)` | Compatibility check. |
| `validate_registry()` | Audit entire registry. |
| `materialize_phase_13(project_root)` | Entry point Phase 13. |

#### Ràng buộc

```
Không sở hữu:
- Model training hoặc evaluation execution
- Persistence, LSTM hoặc Transformer implementation
- Winner selection
- Checkpoint, prediction hoặc actual metric generation
- Test authorization trước Phase 47
- Manual registry edits hoặc Excel source of truth
```

### 15.5. Kết quả & artifacts

| Artifact path | Mô tả |
|---|---|
| `COURSE_WORK/artifacts/experiments/experiment_registry.jsonl` | Canonical JSONL registry |
| `COURSE_WORK/artifacts/experiments/registry_manifest.json` | Manifest |
| `COURSE_WORK/artifacts/experiments/registry_discrepancies.json` | Discrepancies |
| `COURSE_WORK/artifacts/experiments/EXPERIMENT_INDEX.md` | Index |
| `COURSE_WORK/artifacts/experiments/phase_13_signoff.json` | Sign-off EXPERIMENTS-v1 |
| `COURSE_WORK/artifacts/runs/RUN_PS_PS_0001_AFFD3E3F/config.json` | Run config (PERSISTENCE_BASELINE) |
| `COURSE_WORK/artifacts/runs/RUN_PS_PS_0001_AFFD3E3F/status.json` | Run status |

---

## 16. Phase 14 — Persistence Baseline

### 16.1. Mục đích

Implement **Persistence Baseline** với formula `y_hat(t+1) = y(t)`. Sử dụng full L144-anchored common Validation population dưới WB0. Read raw Appliances values **chỉ thông qua final required Validation row**. Keep Test target values và Test metrics inaccessible. Compute MAE Wh, RMSE Wh, R² qua METRICS-v1. Register và start một PERSISTENCE_BASELINE evaluation run trước khi compute metric.

### 16.2. Cells trong notebook

#### Cell 58 (markdown): `## Phase 14 - Persistence Baseline`

#### Cell 59 (code)

```python
phase_14_signoff = materialize_phase_14(PROJECT_ROOT)
display(render_phase_summary(14, PROJECT_ROOT))
```

### 16.3. Outputs đã ghi nhận

**Validation performance (từ HTML summary):**

| Metric | Value | Unit |
|---|---|---|
| MAE | 26.1622 | Wh |
| RMSE | 66.4297 | Wh |
| R² | 0.481326 | Dimensionless |
| Validation samples | 2,960 | Rows |

**Baseline contract:**

| Item | Value |
|---|---|
| Formula | y_hat(t+1) = y(t) |
| Forecast horizon | 1 step / 10 minutes |
| Baseline scope | TASK_LEVEL |
| Population | WINDOWPOP-v1 |
| Source lag | 1 step / 10 minutes |
| Test status | LOCKED_UNTIL_PHASE_47 |

**Status badge:** `PASS` (PERSISTENCE-v1)

### 16.4. Code Python liên quan: `src/course_work/baselines/persistence.py` (1161 dòng)

#### Vai trò

Sở hữu toàn bộ canonical Phase 14 Persistence baseline:

- Verify signed WINDOWS-v1, WINDOWPOP-v1, DATALOADERS-v1, METRICS-v1 và EXPERIMENTS-v1 inputs
- Lock PERSISTENCE-v1 formula `y_hat(t+1) = y(t)`
- Use the full L144-anchored common Validation population under WB0
- Read raw Appliances values only through the final required Validation row
- Keep Test target values and Test metrics inaccessible
- Validate source-target H1 alignment, ten-minute cadence, continuity and population order
- Generate raw-Wh predictions directly from input-end target history
- Compute MAE Wh, RMSE Wh, R² and residuals through METRICS-v1
- Register and start one PERSISTENCE_BASELINE evaluation run before metric computation
- Register prediction, metric, audit and supporting artifacts before completion
- Complete the run without training, scaler, DataLoader, seed, optimizer or checkpoint
- Write manifest, summary, predictions, metrics, audits, discrepancies, README và Phase 14 sign-off
- Expose `materialize_phase_14` as the only notebook orchestration API

#### Key API

| Function | Mục đích |
|---|---|
| `PERSISTENCE_FORMULA` | `'y_hat(t+1) = y(t)'` |
| `build_baseline_contract(...)` | Build PERSISTENCE-v1 contract. |
| `prepare_validation_data(...)` | Build raw Appliances prefix cho L144. |
| `predict_persistence(input_features)` | Compute raw-Wh predictions. |
| `validate_predictions(...)` | Validate source-target alignment. |
| `compute_metrics_via_phase12(...)` | Qua METRICS-v1. |
| `register_persistence_run(...)` | Đăng ký run với EXPERIMENTS-v1. |
| `materialize_phase_14(project_root)` | Entry point Phase 14. |

#### Ràng buộc

```
Không sở hữu:
- Learned model implementation
- Feature-set hoặc target-scaling selection
- Training, optimizer, checkpoint hoặc hyperparameter sweep
- Test evaluation trước Phase 47
- Moving-average hoặc seasonal baseline extension
```

### 16.5. Kết quả & artifacts

| Artifact path | Mô tả |
|---|---|
| `COURSE_WORK/artifacts/baselines/persistence/persistence_manifest.json` | Manifest |
| `COURSE_WORK/artifacts/baselines/persistence/persistence_baseline_summary.json` | Baseline summary |
| `COURSE_WORK/artifacts/baselines/persistence/persistence_validation_metrics.json` | Validation metrics |
| `COURSE_WORK/artifacts/baselines/persistence/persistence_discrepancies.json` | Discrepancies |
| `COURSE_WORK/artifacts/baselines/persistence/phase_14_signoff.json` | Sign-off PERSISTENCE-v1 |
| `COURSE_WORK/artifacts/baselines/persistence/README_PERSISTENCE.md` | README |

Production run:

- `COURSE_WORK/artifacts/runs/RUN_PS_PS_0001_AFFD3E3F/config.json` (canonical run config)
- `COURSE_WORK/artifacts/runs/RUN_PS_PS_0001_AFFD3E3F/status.json` (status)

EXPERIMENTS-v1 hiện đang quản lý **một completed canonical PERSISTENCE_BASELINE run** với null feature/scaler/seed fields và zero trainable parameters.

### 16.6. Phân tích kết quả baseline

| Metric | Value | Diễn giải |
|---|---|---|
| MAE = 26.16 Wh | Trung bình sai số tuyệt đối trên Validation set (2,960 mẫu). Với Appliances mean = 97.7 Wh, sai số ~26%. |
| RMSE = 66.43 Wh | Căn bậc hai trung bình bình phương sai số. Cho thấy có outlier/spike lớn ảnh hưởng đến RMSE. |
| R² = 0.481 | Persistence giải thích ~48% phương sai. Đây là một baseline có ích nhưng chưa cao — các model Phase 15+ (LSTM, Transformer) cần beat R² này trên Validation. |

---

## 17. Tổng hợp luồng dữ liệu (Data Flow Pipeline)

### 17.1. Sơ đồ tổng quan (từ raw data → persistence baseline)

```text
┌─────────────────────────────────────────────────────────────────────┐
│              RAW DATA (immutable, SHA-256 protected)                │
│ COURSE_WORK/data/raw_data/energydata_complete.csv                  │
│ SHA-256: 2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d│
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 0: Coursework Contract                                        │
│ Owner: contracts/coursework.py                                      │
│ Output: configs/base/coursework_contract.json (fingerprint)         │
│ Sign-off: phase_0_signoff.json (PASS, COURSEWORK-CONTRACT-v1)       │
│ [INTERNAL — không xuất hiện trong notebook]                         │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 1: Environment                                                │
│ Owner: utils/environment.py                                         │
│ Output: env inventory, dependency freeze, sign-off                  │
│ Sign-off: phase_1_signoff.json (PASS, ENV-v1)                       │
│ Python 3.10.11, torch 2.13.0, numpy 2.2.6, pandas 2.3.3, ...       │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 2: Data Acquisition                                           │
│ Owner: data/acquisition.py                                          │
│ Output: dataset_manifest.json, source_metadata.json,                │
│         variable_metadata.csv, README_SOURCE.md                     │
│ Sign-off: phase_2_signoff.json (PASS, DATA-v1)                      │
│ 19,735 rows × 29 columns × 10-minute cadence                        │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 3: Schema Audit                                               │
│ Owner: data/schema.py                                               │
│ Output: schema_manifest.json (29 columns, expected schema)          │
│ Sign-off: phase_3_signoff.json (PASS_WITH_WARNING, SCHEMA-v1)        │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 4: Temporal Integrity Audit                                   │
│ Owner: data/temporal.py                                             │
│ Output: temporal_manifest.json (continuity segments, gaps=0)        │
│ Sign-off: phase_4_signoff.json (PASS, TEMPORAL-v1)                  │
│ Coverage 100%, 19,735 rows chronological                            │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 5: EDA                                                        │
│ Owners: data/eda.py (calculation) + reporting/eda.py (figures)      │
│ Notebook: direct descriptive supplement (Human-approved exception)  │
│ Output: eda/tables/*.csv, eda/figures/EDA_*.png                     │
│ Sign-off: phase_5_signoff.json (PASS, EDA-v1)                       │
│ 16 figures + 10+ tables + hypotheses + anomalies                    │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 6: Feature Engineering                                        │
│ Owner: data/features.py                                             │
│ Output: data/interim/.../energydata_feature_engineered_v1.csv        │
│         (19,735 rows × 37 columns: 27 raw + 5 calendar + 4 metadata)│
│ Sign-off: phase_6_signoff.json (PASS, FEATURES-v1)                   │
│ New: hour_sin, hour_cos, dow_sin, dow_cos, weekend                   │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 7: Feature Set Variants                                       │
│ Owner: data/feature_sets.py                                         │
│ Output: feature_components.json, feature_set_registry.json          │
│ 6 variants: FS0/FS1/FS2 × TF0/TF1                                   │
│ Sign-off: phase_7_signoff.json (PASS, FEATURESETS-v1)                │
│ FS1_TF1 = baseline-eligible (PASS), others FAIL                     │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 8: Chronological Split                                        │
│ Owner: data/splitting.py                                            │
│ Output: split_manifest.json, figures/SPLIT_01_timeline.png          │
│ 70/15/15 by target_timestamp:                                       │
│   Train: 13,814 (2016-01-11 → 2016-04-16)                           │
│   Validation: 2,960 (2016-04-16 → 2016-05-07)                        │
│   Test: 2,961 (2016-05-07 → 2016-05-27)                              │
│ Sign-off: phase_8_signoff.json (PASS, SPLIT-v1)                     │
│ WB0 = context carry-over, WB1 = strict isolation                    │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 9: Train-Only Scaling                                         │
│ Owner: data/scaling.py                                              │
│ Output: scalers/x/XSCALER__FS{0,1,2}_TF{0,1}__SCALING-v1.joblib (6) │
│         scalers/y/YSCALER__YS1__SCALING-v1.joblib (1)                │
│ Sign-off: phase_9_signoff.json (PASS, SCALING-v1)                   │
│ YS0 = identity, YS1 = StandardScaler on TRAIN target                │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 10: Window Builder                                            │
│ Owner: data/windows.py                                              │
│ Output: window_index, population (WINDOWPOP-v1):                    │
│   Common targets Train: 13,670, Validation: 2,960, Test: 2,961      │
│   Total: 19,591                                                     │
│ Lookbacks: L36, L72, L144; H1, oldest_to_newest, WB0 primary        │
│ Sign-off: phase_10_signoff.json (PASS, WINDOWS-v1 + WINDOWPOP-v1)   │
│ Test target LOCKED_UNTIL_PHASE_47                                   │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 11: DataLoaders                                               │
│ Owner: data/datasets.py                                             │
│ Output: SequenceWindowDataset (lazy map-style)                       │
│         Split-specific DataLoaders (B32, B64)                       │
│ Target access modes: TRAIN, VALIDATION, TEST_LOCKED,                │
│                      TEST_EVALUATION (Phase 47 only)                │
│ Sign-off: phase_11_signoff.json (PASS, DATALOADERS-v1)              │
│ Batch=64, Shuffle=Train only, Workers=0, Pin-memory=CUDA-only       │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 12: Shared Metrics                                            │
│ Owner: evaluation/metrics.py                                        │
│ Output: metric_contract, METRIC_REGISTRY:                           │
│   MAE (mae_wh, lower Wh, required)                                  │
│   RMSE (rmse_wh, lower Wh, primary selection)                       │
│   R² (r2, higher, required)                                         │
│ Test metrics LOCKED in Phase 12                                     │
│ Sign-off: phase_12_signoff.json (PASS, METRICS-v1)                  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 13: Experiment Registry                                       │
│ Owner: experiments/registry.py                                      │
│ Output: experiment_registry.jsonl (1 production run)                │
│         26 experiment families defined                              │
│ Lifecycle: REGISTERED → RUNNING → COMPLETED/FAILED/...               │
│ Run: RUN_PS_PS_0001_AFFD3E3F (PERSISTENCE_BASELINE)                  │
│ Sign-off: phase_13_signoff.json (PASS, EXPERIMENTS-v1)              │
│ Zero fabricated runs                                                │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 14: Persistence Baseline                                      │
│ Owner: baselines/persistence.py                                     │
│ Formula: y_hat(t+1) = y(t)                                          │
│ Population: WINDOWPOP-v1 (L144, WB0, Validation only)               │
│ Validation metrics (raw Wh):                                        │
│   MAE = 26.1622 Wh, RMSE = 66.4297 Wh, R² = 0.481326               │
│ Sign-off: phase_14_signoff.json (PASS, PERSISTENCE-v1)              │
│ Test FIREWALLED until Phase 47                                      │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
            ┌────────────────────────────────────────┐
            │  NEXT: Phase 15 (LSTM) — TODO         │
            │  NEXT: Phase 16 (Transformer) — TODO   │
            │  Phase 47 (FINAL_TEST gate) — TODO     │
            └────────────────────────────────────────┘
```

### 17.2. Liệt kê artifacts theo phase

#### Phase 0

| Path | Purpose |
|---|---|
| `configs/base/coursework_contract.json` | Original contract |
| `artifacts/contracts/coursework_contract.json` | Canonical artifact |
| `artifacts/contracts/coursework_contract.sha256` | Checksum |
| `artifacts/contracts/phase_0_signoff.json` | Sign-off (COURSEWORK-CONTRACT-v1) |

#### Phase 1

| Path | Purpose |
|---|---|
| `artifacts/environment/phase_1_signoff.json` | Sign-off (ENV-v1) |
| `artifacts/environment/*` | Inventory, freeze, smoke tests |

#### Phase 2

| Path | Purpose |
|---|---|
| `data/raw_data/energydata_complete.csv` | Raw (immutable) |
| `data/raw_data/checksums.sha256` | SHA-256 |
| `data/raw_data/dataset_manifest.json` | Manifest |
| `data/raw_data/source_metadata.json` | Metadata |
| `data/raw_data/variable_metadata.csv` | Variable metadata |
| `data/raw_data/README_SOURCE.md` | Provenance |
| `artifacts/acquisition/phase_2_signoff.json` | Sign-off (DATA-v1) |

#### Phase 3

| Path | Purpose |
|---|---|
| `artifacts/schema/schema_manifest.json` | Schema manifest |
| `artifacts/schema/phase_3_signoff.json` | Sign-off (SCHEMA-v1) |

#### Phase 4

| Path | Purpose |
|---|---|
| `artifacts/temporal/temporal_manifest.json` | Temporal manifest |
| `artifacts/temporal/phase_4_signoff.json` | Sign-off (TEMPORAL-v1) |

#### Phase 5

| Path | Purpose |
|---|---|
| `artifacts/eda/tables/*.csv` | 10+ bảng EDA |
| `artifacts/eda/figures/EDA_*.png` | 16 figure PNG |
| `artifacts/eda/eda_manifest.json` | Manifest |
| `artifacts/eda/phase_5_signoff.json` | Sign-off (EDA-v1) |

#### Phase 6

| Path | Purpose |
|---|---|
| `data/interim/uci_appliances_energy_prediction/energydata_feature_engineered_v1.csv` | Derived CSV (37 cols) |
| `artifacts/features/feature_engineering_manifest.json` | Manifest |
| `artifacts/features/feature_engineered_v1.sha256` | Checksum |
| `artifacts/features/feature_engineering_discrepancies.json` | Discrepancies |
| `artifacts/features/phase_6_signoff.json` | Sign-off (FEATURES-v1) |

#### Phase 7

| Path | Purpose |
|---|---|
| `artifacts/feature_sets/feature_components.json` | Components |
| `artifacts/feature_sets/feature_set_registry.json` | 6 variants |
| `artifacts/feature_sets/feature_set_manifest.json` | Manifest |
| `artifacts/feature_sets/feature_set_discrepancies.json` | Discrepancies |
| `artifacts/feature_sets/phase_7_signoff.json` | Sign-off (FEATURESETS-v1) |

#### Phase 8

| Path | Purpose |
|---|---|
| `artifacts/splits/split_manifest.json` | Manifest |
| `artifacts/splits/split_discrepancies.json` | Discrepancies |
| `artifacts/splits/figures/SPLIT_01_timeline.png` | Timeline figure |
| `artifacts/splits/phase_8_signoff.json` | Sign-off (SPLIT-v1) |

#### Phase 9

| Path | Purpose |
|---|---|
| `artifacts/scalers/x/XSCALER__FS0_TF0__SCALING-v1.joblib` | X scaler 1 |
| `artifacts/scalers/x/XSCALER__FS0_TF1__SCALING-v1.joblib` | X scaler 2 |
| `artifacts/scalers/x/XSCALER__FS1_TF0__SCALING-v1.joblib` | X scaler 3 |
| `artifacts/scalers/x/XSCALER__FS1_TF1__SCALING-v1.joblib` | X scaler 4 |
| `artifacts/scalers/x/XSCALER__FS2_TF0__SCALING-v1.joblib` | X scaler 5 |
| `artifacts/scalers/x/XSCALER__FS2_TF1__SCALING-v1.joblib` | X scaler 6 |
| `artifacts/scalers/y/YSCALER__YS1__SCALING-v1.joblib` | Y target scaler |
| `artifacts/scaling/scaler_registry.json` | Registry |
| `artifacts/scaling/scaler_checksums.sha256` | Checksums |
| `artifacts/scaling/scaling_manifest.json` | Manifest |
| `artifacts/scaling/scaling_discrepancies.json` | Discrepancies |
| `artifacts/scaling/phase_9_signoff.json` | Sign-off (SCALING-v1) |

#### Phase 10

| Path | Purpose |
|---|---|
| `artifacts/windows/window_manifest.json` | Manifest |
| `artifacts/windows/window_fingerprints.json` | Fingerprints |
| `artifacts/windows/window_discrepancies.json` | Discrepancies |
| `artifacts/windows/phase_10_signoff.json` | Sign-off (WINDOWS-v1 + WINDOWPOP-v1) |

#### Phase 11

| Path | Purpose |
|---|---|
| `artifacts/dataloaders/dataloader_manifest.json` | Manifest |
| `artifacts/dataloaders/device_transfer_policy.json` | Device policy |
| `artifacts/dataloaders/dataloader_discrepancies.json` | Discrepancies |
| `artifacts/dataloaders/phase_11_signoff.json` | Sign-off (DATALOADERS-v1) |

#### Phase 12

| Path | Purpose |
|---|---|
| `artifacts/metrics/metric_contract.json` | Metric contract |
| `artifacts/metrics/metric_comparison_schema.json` | Comparison schema |
| `artifacts/metrics/prediction_bundle_schema.json` | Prediction schema |
| `artifacts/metrics/metric_manifest.json` | Manifest |
| `artifacts/metrics/metric_discrepancies.json` | Discrepancies |
| `artifacts/metrics/phase_12_signoff.json` | Sign-off (METRICS-v1) |

#### Phase 13

| Path | Purpose |
|---|---|
| `artifacts/experiments/experiment_registry.jsonl` | Canonical JSONL |
| `artifacts/experiments/registry_manifest.json` | Manifest |
| `artifacts/experiments/registry_discrepancies.json` | Discrepancies |
| `artifacts/experiments/EXPERIMENT_INDEX.md` | Index |
| `artifacts/experiments/phase_13_signoff.json` | Sign-off (EXPERIMENTS-v1) |
| `artifacts/runs/RUN_PS_PS_0001_AFFD3E3F/config.json` | Run config |
| `artifacts/runs/RUN_PS_PS_0001_AFFD3E3F/status.json` | Run status |

#### Phase 14

| Path | Purpose |
|---|---|
| `artifacts/baselines/persistence/persistence_manifest.json` | Manifest |
| `artifacts/baselines/persistence/persistence_baseline_summary.json` | Summary |
| `artifacts/baselines/persistence/persistence_validation_metrics.json` | Validation metrics |
| `artifacts/baselines/persistence/persistence_discrepancies.json` | Discrepancies |
| `artifacts/baselines/persistence/phase_14_signoff.json` | Sign-off (PERSISTENCE-v1) |
| `artifacts/baselines/persistence/README_PERSISTENCE.md` | README |

### 17.3. Quy ước atomic write

Mọi machine-readable artifact đều được ghi **atomic** (qua `write_bytes_once_or_verify` / `write_json_once_or_verify` / `write_text_once_or_verify` trong `utils/artifacts.py`). Mỗi sign-off ghi rồi **reload verify** trước khi pass. Không overwrite artifact đã sign-off.

---

## 18. Tổng hợp các utility modules

### 18.1. `utils/artifacts.py` (74 dòng)

#### Vai trò

Sở hữu:

- Canonical path resolution (`get_project_root`)
- Directory creation trong approved scope
- Atomic write (`atomic_write_bytes`, `write_bytes_once_or_verify`, `write_json_once_or_verify`, `write_text_once_or_verify`)
- JSON và CSV serialization (`canonical_json_bytes`, `csv_text`)
- SHA-256 helper (`sha256_bytes`, `sha256_file`)
- Artifact fingerprint
- Artifact reload verification

#### Public API

| Function | Mục đích |
|---|---|
| `get_project_root()` | Resolve canonical `COURSE_WORK/` root. |
| `canonical_json_bytes(obj)` | Serialize JSON với sorted keys, indent=2. |
| `sha256_bytes(data)` | SHA-256 hex của bytes. |
| `sha256_file(path)` | SHA-256 hex của file. |
| `atomic_write_bytes(path, data)` | Ghi atomic (temp + rename). |
| `write_bytes_once_or_verify(path, data)` | Write-once hoặc verify-on-read. |
| `write_json_once_or_verify(path, obj)` | Same cho JSON. |
| `write_text_once_or_verify(path, text)` | Same cho text. |
| `csv_text(rows)` | Serialize rows → CSV string. |
| `read_json(path)` | Load JSON. |

#### Ràng buộc

```
Không sở hữu schema khoa học, EDA calculation hoặc Phase decision.
```

### 18.2. `utils/environment.py` (317 dòng)

Xem chi tiết tại **§3.4**. Tóm tắt:

| Function | Mục đích |
|---|---|
| `select_device()` | CUDA → MPS → CPU. |
| `resolve_kernel_contract()` | Verify kernel ↔ interpreter. |
| `package_versions()` | Core package versions. |
| `environment_inventory()` | Interpreter, OS, platform, working dir, GPU. |
| `environment_identity()` | Hash-stable env identity. |
| `environment_identity_differences()` | Compare identities. |
| `device_smoke_test()` | Tensor alloc + autograd + module + optimizer. |
| `dependency_freeze()` | Snapshot for reproducibility. |
| `materialize_phase_1(project_root)` | Phase 1 entry point. |

### 18.3. `utils/reproducibility.py` (76 dòng)

#### Vai trò

Sở hữu:

- Python seed
- NumPy seed
- PyTorch seed
- CUDA seed khi áp dụng
- Deterministic policy
- Randomness smoke tests

#### Public API

| Symbol | Mục đích |
|---|---|
| `DEVELOPMENT_SEED` | Seed mặc định cho development (vd. 42). |
| `FINAL_SEEDS` | Tuple final seeds `[42, 123, 2026]` (từ contract). |
| `REPRODUCIBILITY_MODES` | Enum: D0 (deterministic), D1 (strict). |
| `set_seed(seed)` | Set Python + NumPy + PyTorch seeds. |
| `configure_reproducibility(mode)` | Cấu hình deterministic algorithms. |
| `seed_worker(worker_id)` | Worker init fn cho DataLoader. |
| `build_torch_generator(seed)` | Build torch.Generator. |
| `randomness_smoke_test(...)` | Smoke test reproducibility. |

#### Ràng buộc

```
Không tuyên bố reproducibility tuyệt đối giữa mọi platform và package version.
```

---

## 19. Contracts & Reporting modules

### 19.1. `contracts/coursework.py` (185 dòng)

Xem chi tiết tại **§2.3**. Tóm tắt:

| Function | Mục đích |
|---|---|
| `EXPECTED_OPTION_IDS` | 19 option groups với IDs hợp lệ. |
| `load_coursework_contract(path=None)` | Đọc `configs/base/coursework_contract.json`. |
| `validate_coursework_contract(contract)` | Validate schema đầy đủ (contract_version, problem, lookbacks, split, metrics, models, seeds, registry, research_questions, policy fields). |
| `coursework_contract_fingerprint(contract)` | SHA-256 fingerprint. |
| `materialize_phase_0(project_root=None)` | Phase 0 entry point: load + validate + write artifact + checksum + sign-off. |

### 19.2. `reporting/phase_summary.py` (992 dòng)

#### Vai trò

Sở hữu presentation layer dùng chung cho Phase 0–14:

- Đọc canonical machine-readable artifacts đã được materialize
- Tạo processing log đầy đủ độc lập với notebook presentation
- Áp dụng presentation allowlist riêng cho từng Phase (xem §19.3)
- Chỉ render bảng hoặc visualization trực tiếp phục vụ quyết định của Phase
- Ghi một processing log JSON atomically cho mỗi Phase dưới `docs/save_log_in_processing`
- Render HTML/CSS cục bộ, không dùng JavaScript hoặc external resource
- Escape mọi artifact value trước khi đưa vào HTML
- Không render warnings, discrepancies, checksum, fingerprint, source artifacts hoặc technical lineage

#### Public API

| Function | Mục đích |
|---|---|
| `SOURCE_SPECS` | Dict định nghĩa presentation allowlist cho mỗi Phase. |
| `deduplicate(items)` | Dedup helper. |
| `render_dataframe_table(dataframe, title, subtitle, metrics)` | Render DataFrame → HTML table card. |
| `render_split_allocation_bar(splits)` | Render visual split bar. |
| `render_phase_summary(phase_id, project_root)` | Main entry point — render phase HTML. |

#### Presentation allowlist (architecture rule §7.9.1)

| Phase | Allowlist hiển thị |
|---|---|
| 1 | Environment overview, Core package versions |
| 2 | Dataset overview |
| 3 | Schema overview, Critical schema checks |
| 4 | Temporal coverage, Critical temporal checks |
| 5 | Header + các output EDA 5.1-5.14 đã duyệt |
| 6 | Feature overview, Engineered feature registry |
| 7 | Feature-set registry (KHÔNG fingerprint) |
| 8 | Chronological membership, Split allocation |
| 9 | X scaler bundles, Target scaling options |
| 10 | Window contract, Common target population |
| 11 | Dataset population, Loader policy |
| 12 | Metric registry, Evaluation policy |
| 13 | Registry state, Core safeguards |
| 14 | Validation performance, Baseline contract |

#### Ràng buộc

```
Không tính lại scientific result, không thay đổi canonical artifact và không thay thế sign-off.
```

---

## 20. Trạng thái hiện tại & gaps cần làm tiếp

### 20.1. Phases đã hoàn thành (sign-off PASS)

| Phase | Status | Artifact version | Owner module |
|---|---|---|---|
| 0 | PASS (internal) | COURSEWORK-CONTRACT-v1 | `contracts/coursework.py` |
| 1 | PASS | ENV-v1 | `utils/environment.py` |
| 2 | PASS | DATA-v1 | `data/acquisition.py` |
| 3 | PASS_WITH_WARNING | SCHEMA-v1 | `data/schema.py` |
| 4 | PASS | TEMPORAL-v1 | `data/temporal.py` |
| 5 | PASS | EDA-v1 | `data/eda.py` + `reporting/eda.py` |
| 6 | PASS | FEATURES-v1 | `data/features.py` |
| 7 | PASS | FEATURESETS-v1 | `data/feature_sets.py` |
| 8 | PASS | SPLIT-v1 | `data/splitting.py` |
| 9 | PASS | SCALING-v1 | `data/scaling.py` |
| 10 | PASS | WINDOWS-v1 + WINDOWPOP-v1 | `data/windows.py` |
| 11 | PASS | DATALOADERS-v1 | `data/datasets.py` |
| 12 | PASS | METRICS-v1 | `evaluation/metrics.py` |
| 13 | PASS | EXPERIMENTS-v1 | `experiments/registry.py` |
| 14 | PASS | PERSISTENCE-v1 | `baselines/persistence.py` |

### 20.2. Persistence baseline numbers

| Metric | Value | Implication |
|---|---|---|
| Validation MAE | 26.1622 Wh | ~27% sai số tuyệt đối trung bình so với Appliances mean = 97.7 Wh |
| Validation RMSE | 66.4297 Wh | Outliers/spikes đẩy RMSE lên cao |
| Validation R² | 0.481326 | Persistence giải thích ~48% phương sai |
| Validation samples | 2,960 | Full common L144-anchored Validation population (WINDOWPOP-v1) |

**Diễn giải:** Persistence baseline là một reference đơn giản nhưng đã giải thích được gần một nửa phương sai. Phase 15 (LSTM) và Phase 16 (Transformer) sẽ cần beat R² = 0.481 trên Validation trước khi được phép Test evaluation (qua Phase 47 FINAL_TEST gate).

### 20.3. Phases CÒN PHẢI LÀM (chưa triển khai)

Theo `architecture_rule.md` §7.19:

```
models/*, training/* và downstream attention/* chưa được triển khai trong current scope.
Sự tồn tại của namespace rỗng không được xem là Phase đã triển khai.
```

#### A. Modeling Phase (15+)

| Phase | Module cần tạo | Tên module dự kiến | Trạng thái |
|---|---|---|---|
| 15 | LSTM model + training + evaluation | `models/lstm.py`, `training/lstm_trainer.py` | **CHƯA CÓ** |
| 16 | Transformer Encoder model + training + evaluation | `models/transformer_encoder.py`, `training/transformer_trainer.py` | **CHƯA CÓ** |
| 17+ | Hyperparameter sweep, model selection, ensemble | `experiments/sweeps.py`, etc. | **CHƯA CÓ** |

#### B. Attention Phase (21–23) — đặc biệt chú ý

File đã tồn tại dưới `src/course_work/attention/` nhưng **TẤT CẢ 4 FILE ĐỀU RỖNG (0 bytes / 0 dòng)**:

| File path | Size | Lines | Trạng thái |
|---|---|---|---|
| `attention/__init__.py` | 0 | 0 | Rỗng |
| `attention/extraction.py` | 0 bytes | 0 dòng | Rỗng — namespace reserved |
| `attention/head_comparison.py` | 0 bytes | 0 dòng | Rỗng — namespace reserved |
| `attention/heatmaps.py` | 0 bytes | 0 dòng | Rỗng — namespace reserved |
| `attention/last_query.py` | 0 bytes | 0 dòng | Rỗng — namespace reserved |

Sự tồn tại của 4 file rỗng chỉ là namespace reservation. **Không được xem là Phase 21–23 đã triển khai.**

### 20.4. Source code ownership CHƯA CÓ

| Package / Module dự kiến | Owner cho Phase nào | Trạng thái |
|---|---|---|
| `models/lstm.py` | Phase 15 | CHƯA CÓ |
| `models/transformer_encoder.py` | Phase 16 | CHƯA CÓ |
| `models/__init__.py` | Phase 15-16 | CHƯA CÓ package |
| `training/lstm_trainer.py` | Phase 15 | CHƯA CÓ |
| `training/transformer_trainer.py` | Phase 16 | CHƯA CÓ |
| `training/__init__.py` | Phase 15-16 | CHƯA CÓ package |
| `attention/extraction.py` | Phase 21 | CÓ file rỗng |
| `attention/heatmaps.py` | Phase 22 | CÓ file rỗng |
| `attention/last_query.py` | Phase 23 | CÓ file rỗng |
| `attention/head_comparison.py` | Phase 23 | CÓ file rỗng |
| `attention/__init__.py` | Phase 21-23 | CÓ file rỗng |

### 20.5. Ràng buộc cho các Phase tiếp theo

Theo architecture rule §24 Change-control gate, **CẦN architecture amendment và Human approval** trước khi:

- Tạo package mới ngoài cây canonical
- Đổi canonical notebook
- Đổi owner module
- Đổi dependency direction
- Đổi data lifecycle
- Đổi artifact root
- Đổi config hierarchy
- Đổi Phase-to-module mapping
- Cho phép processing logic trong notebook
- Thêm Phase mới

### 20.6. Test firewall

Test prediction và Test metric **chưa được phép** cho đến khi Phase 47 mở FINAL_TEST gate. Phase 14 Persistence hiện chỉ compute Validation metrics. Phase 15–16 (LSTM/Transformer) cũng sẽ chỉ compute Validation metrics trong khi training.

---

## 21. Phụ lục: Danh sách file đã đọc

### 21.1. Notebook (59 cells)

| File | Lines | Cells |
|---|---|---|
| `/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb` | 6,014 | 59 |

#### Cấu trúc cells (tóm tắt)

| Cell # | Type | Phase | First line / nội dung chính |
|---|---|---|---|
| 1 | markdown | Header | `# MULTIVARIATE TIME-SERIES REGRESSION` |
| 2 | markdown | Header | `## Define Problem` (9 bullet points defining task) |
| 3 | code | Header | Imports all materialize_phase_N functions + sets PROJECT_ROOT |
| 4 | markdown | 1 | `## Phase 1 - Environment` |
| 5 | code | 1 | `phase_1_signoff = materialize_phase_1(...)` + render summary |
| 6 | markdown | 2 | `## Phase 2 - Data Acquisition` |
| 7 | code | 2 | `phase_2_signoff = materialize_phase_2(...)` |
| 8 | markdown | 3 | `## Phase 3 - Schema Audit` |
| 9 | code | 3 | `phase_3_signoff = materialize_phase_3(...)` |
| 10 | markdown | 4 | `## Phase 4 - Temporal Integrity Audit` |
| 11 | code | 4 | `phase_4_signoff = materialize_phase_4(...)` |
| 12 | markdown | 5 | `## Phase 5 - Exploratory Data Analysis` |
| 13 | code | 5 | Setup: sns theme, materialize_phase_5, load schema, load temporal view |
| 14 | markdown | 5.1 | `### 5.1 Data Overview` |
| 15 | code | 5.1 | Head(8) + describe() table |
| 16 | markdown | 5.2 | `### 5.2 Missing-Value Analysis` |
| 17 | code | 5.2 | Missing counts + heatmap |
| 18 | markdown | 5.3 | `### 5.3 Calendar Derivations and Sampling Coverage` |
| 19 | code | 5.3 | Hour/day/weekend counts + 4 figures |
| 20 | markdown | 5.4 | `### 5.4 Target Distribution` |
| 21 | code | 5.4 | Target summary + quantiles + 4 figures (hist/ECDF/boxplot/spikes) |
| 22 | markdown | 5.5 | `### 5.5 Target Timeline and Representative Week` |
| 23 | code | 5.5 | Daily average + first-week plot + rep week table |
| 24 | markdown | 5.6 | `### 5.6 Calendar Energy Patterns` |
| 25 | code | 5.6 | 3 profile tables + 4 figures |
| 26 | markdown | 5.7 | `### 5.7 Feature Distributions` |
| 27 | code | 5.7 | 3 distribution figures (temperature/humidity/lights) |
| 28 | markdown | 5.8 | `### 5.8 Numerical Relationships` |
| 29 | code | 5.8 | Top correlations + high_correlation_pairs + heatmap + scatter |
| 30 | markdown | 5.9 | `### 5.9 Cross-Correlation at Short Lags` |
| 31 | code | 5.9 | Segment-aware cross-correlation |
| 32 | markdown | 5.10 | `### 5.10 Categorical and Numerical Relationships` |
| 33 | code | 5.10 | Weekday/weekend describe + 2-panel boxplot |
| 34 | markdown | 5.11 | `### 5.11 Descriptive IQR Outlier Diagnostics` |
| 35 | code | 5.11 | IQR outlier summary table |
| 36 | markdown | 5.12 | `### 5.12 Isolated Outlier-Smoothing Demonstration` |
| 37 | code | 5.12 | Before/after smoothing on deep copy |
| 38 | markdown | 5.13 | `### 5.13 Time-Series Diagnostics` |
| 39 | code | 5.13 | 2 lag correlation tables + 3 figures (ACF, rolling mean, rolling std) |
| 40 | markdown | 5.14 | `### 5.14 Extreme Samples, Hypotheses and Sign-Off` |
| 41 | code | 5.14 | extreme_target_samples + eda_hypotheses |
| 42 | markdown | 6 | `## Phase 6 - Feature Engineering` |
| 43 | code | 6 | `phase_6_signoff = materialize_phase_6(...)` |
| 44 | markdown | 7 | `## Phase 7 - Feature-Set Variants` |
| 45 | code | 7 | `phase_7_signoff = materialize_phase_7(...)` |
| 46 | markdown | 8 | `## Phase 8 - Chronological Split` |
| 47 | code | 8 | `phase_8_signoff = materialize_phase_8(...)` |
| 48 | markdown | 9 | `## Phase 9 - Train-Only Scaling` |
| 49 | code | 9 | `phase_9_signoff = materialize_phase_9(...)` |
| 50 | markdown | 10 | `## Phase 10 - Window Builder` |
| 51 | code | 10 | `phase_10_signoff = materialize_phase_10(...)` |
| 52 | markdown | 11 | `## Phase 11 - DataLoaders` |
| 53 | code | 11 | `phase_11_signoff = materialize_phase_11(...)` |
| 54 | markdown | 12 | `## Phase 12 - Shared Metrics` |
| 55 | code | 12 | `phase_12_signoff = materialize_phase_12(...)` |
| 56 | markdown | 13 | `## Phase 13 - Experiment Registry` |
| 57 | code | 13 | `phase_13_signoff = materialize_phase_13(...)` |
| 58 | markdown | 14 | `## Phase 14 - Persistence Baseline` |
| 59 | code | 14 | `phase_14_signoff = materialize_phase_14(...)` |

### 21.2. Python source files (31 files)

#### Contracts

| File | Lines | Vai trò |
|---|---|---|
| `src/course_work/__init__.py` | 1 | Package init (empty) |
| `src/course_work/contracts/__init__.py` | 1 | Package init (empty) |
| `src/course_work/contracts/coursework.py` | 185 | Phase 0 owner |

#### Utils

| File | Lines | Vai trò |
|---|---|---|
| `src/course_work/utils/__init__.py` | 1 | Package init (empty) |
| `src/course_work/utils/artifacts.py` | 74 | Atomic write, SHA-256, JSON/CSV serialization |
| `src/course_work/utils/environment.py` | 317 | Phase 1 owner (env inventory, device, smoke) |
| `src/course_work/utils/reproducibility.py` | 76 | Seed management, deterministic config |

#### Data

| File | Lines | Vai trò |
|---|---|---|
| `src/course_work/data/__init__.py` | 1 | Package init (empty) |
| `src/course_work/data/acquisition.py` | 309 | Phase 2 owner |
| `src/course_work/data/schema.py` | 404 | Phase 3 owner |
| `src/course_work/data/temporal.py` | 464 | Phase 4 owner |
| `src/course_work/data/eda.py` | 443 | Phase 5 calculation owner |
| `src/course_work/data/features.py` | 438 | Phase 6 owner |
| `src/course_work/data/feature_sets.py` | 574 | Phase 7 owner |
| `src/course_work/data/splitting.py` | 503 | Phase 8 owner |
| `src/course_work/data/scaling.py` | 875 | Phase 9 owner |
| `src/course_work/data/windows.py` | 913 | Phase 10 owner |
| `src/course_work/data/datasets.py` | 1014 | Phase 11 owner |

#### Evaluation

| File | Lines | Vai trò |
|---|---|---|
| `src/course_work/evaluation/metrics.py` | 922 | Phase 12 owner (METRICS-v1) |

#### Experiments

| File | Lines | Vai trò |
|---|---|---|
| `src/course_work/experiments/registry.py` | 1735 | Phase 13 owner (EXPERIMENTS-v1) |

#### Baselines

| File | Lines | Vai trò |
|---|---|---|
| `src/course_work/baselines/persistence.py` | 1161 | Phase 14 owner (PERSISTENCE-v1) |

#### Reporting

| File | Lines | Vai trò |
|---|---|---|
| `src/course_work/reporting/__init__.py` | 1 | Package init (empty) |
| `src/course_work/reporting/eda.py` | 347 | Phase 5 figure rendering owner |
| `src/course_work/reporting/phase_summary.py` | 992 | Presentation layer cho Phase 0-14 |

#### Attention (rỗng — namespace reserved)

| File | Lines | Vai trò |
|---|---|---|
| `src/course_work/attention/__init__.py` | 0 | (empty — Phase 21-23 reserved) |
| `src/course_work/attention/extraction.py` | 0 | (empty — Phase 21) |
| `src/course_work/attention/head_comparison.py` | 0 | (empty — Phase 23) |
| `src/course_work/attention/heatmaps.py` | 0 | (empty — Phase 22) |
| `src/course_work/attention/last_query.py` | 0 | (empty — Phase 23) |

### 21.3. Tổng số source code (đã triển khai)

| Nhóm | Số file đã triển khai | Tổng dòng |
|---|---|---|
| Contracts | 1 | 185 |
| Utils | 3 | 467 |
| Data | 10 | 5,935 |
| Evaluation | 1 | 922 |
| Experiments | 1 | 1,735 |
| Baselines | 1 | 1,161 |
| Reporting | 2 | 1,339 |
| **Total đã triển khai** | **19 files** | **11,744 dòng** |
| Attention (rỗng) | 4 files | 0 dòng |
| **Tổng cộng** | **23 files** | **11,744 dòng** |

### 21.4. Notebook statistics

| Item | Value |
|---|---|
| Total cells | 59 |
| Markdown cells | 30 (bao gồm cell 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58) |
| Code cells | 29 (bao gồm cell 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59) |
| Phases có cell | 1, 2, 3, 4, 5 (với 14 sub-cells), 6, 7, 8, 9, 10, 11, 12, 13, 14 |
| Phase 0 cells | 0 (internal only) |

### 21.5. Architecture rule document đã đọc tham chiếu

| File | Lines | Vai trò |
|---|---|---|
| `COURSE_WORK/docs/RULE_BASE/architecture_rule.md` | 1411 | COURSE-WORK-ARCHITECTURE-v1, status ACTIVE_HUMAN_APPROVED |

---

## Tổng kết các phát hiện chính

1. **Project coursework đã hoàn thành Phase 0–14 với đầy đủ sign-off** (COURSEWORK-CONTRACT → PERSISTENCE-v1). Mọi Phase đều có canonical owner module, atomic-write artifacts, và đều pass phase gate.

2. **Persistence baseline (Phase 14) đã chạy thành công** với Validation metrics: MAE = 26.1622 Wh, RMSE = 66.4297 Wh, R² = 0.481326 trên 2,960 mẫu WINDOWPOP-v1 (L144-anchored, WB0). EXPERIMENTS-v1 hiện quản lý **một completed canonical PERSISTENCE_BASELINE run** duy nhất (RUN_PS_PS_0001_AFFD3E3F).

3. **Notebook có 59 cells** với 1 markdown title (`# MULTIVARIATE TIME-SERIES REGRESSION`), 1 problem-definition markdown, 1 setup code cell (imports + PROJECT_ROOT), 28 cells cho Phase 1–14 (markdown header + code `materialize_phase_N`), và 28 cells cho Phase 5 sub-sections (5.1 → 5.14).

4. **Data pipeline shape:** 19,735 rows × 29 columns raw → 37 columns sau Phase 6 (27 raw + 5 calendar features + 4 metadata) → 6 feature set variants (FS0/FS1/FS2 × TF0/TF1) → 70/15/15 split (Train 13,814 / Val 2,960 / Test 2,961) → WINDOWPOP-v1 common target population 19,591.

5. **Source code có 19 files Python đã triển khai (11,744 dòng)** + 4 attention files rỗng (0 dòng) là namespace reserved. Tổng cộng 23 file Python trong `src/course_work/`. Tất cả đều tuân theo architecture rule COURSE-WORK-ARCHITECTURE-v1 (status ACTIVE_HUMAN_APPROVED).

6. **Các gap cần làm tiếp:**
   - **Phase 15 (LSTM)** và **Phase 16 (Transformer Encoder)** — chưa có file source code nào trong `src/course_work/` (chưa có package `models/` hay `training/`).
   - **Phase 21–23 (Attention analysis)** — 4 file rỗng trong `attention/` cần được triển khai.
   - **Phase 47 FINAL_TEST gate** — Test prediction và Test metric vẫn bị lock cho đến khi mở gate.

7. **Test firewall nghiêm ngặt:** Persistence Validation đã chạy nhưng Test prediction/evaluation vẫn LOCKED_UNTIL_PHASE_47. Tất cả Phase 12+ đều reject development Test targets.

8. **Single source of truth cho architecture:** `docs/RULE_BASE/architecture_rule.md` (1411 dòng) quy định rõ Phase-to-module mapping, presentation allowlist, dependency direction, artifact lifecycle. Phase 0 là internal prerequisite không xuất hiện trong notebook.

---

*Tài liệu này được tổng hợp từ việc đọc chi tiết toàn bộ notebook (6,014 dòng JSON) và 23 file Python trong `src/course_work/` (tổng 11,744 dòng code đã triển khai), không thực thi bất kỳ code cell nào. Tất cả thông tin về outputs, sign-off status, feature counts, sample counts, scaler registries, run IDs đều được trích từ HTML summary outputs đã được lưu trong notebook execution.*
