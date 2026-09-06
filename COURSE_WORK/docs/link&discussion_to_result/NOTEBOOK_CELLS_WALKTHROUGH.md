# CourseWork Notebook — Cell Walkthrough & Output Guide

> **File này** là tài liệu đi kèm với notebook [`CourseWork.ipynb`](../../notebook_course_work/CourseWork.ipynb). Mỗi mục bên dưới mô tả chi tiết **cell ID**, **output kỳ vọng**, và **giải thích ý nghĩa** của output đó.
>
> ## ⚠️ Cách Mở Đúng Cell Trong Cursor IDE
>
> **Vấn đề:** Cursor IDE (và VS Code) không hỗ trợ mở trực tiếp tới 1 cell cụ thể qua URL fragment `#cellId=...` cho file `.ipynb`. Khi sếp click vào link có fragment, IDE sẽ mở file notebook nhưng KHÔNG nhảy tới cell đó, dẫn đến "lỗi như trên" mà sếp đã gặp.
>
> **Giải pháp:** Em đã chuẩn hóa tất cả links để chỉ mở file notebook. Sau đó sếp dùng **TOC navigation** bên trái của Cursor IDE (panel "Outline" hoặc "Notebook") để nhảy tới cell mong muốn. Mỗi section trong file này ghi rõ **"Cell N"** để sếp dễ tìm.
>
> ### Cách Navigate Trong Cursor IDE (đã verify):
> 1. **Click link** → mở file notebook (KHÔNG cần `#cellId`)
> 2. **Mở Outline panel**: View → Outline (hoặc `Cmd+Shift+P` → "Notebook: Focus on Cell")
> 3. **Tìm cell** theo nhãn "Phase N - ..." trong Outline
> 4. **Hoặc** dùng shortcut `Cmd+P` → gõ `Phase 1` để nhảy tới section
>
> Mọi section đã được đánh anchor rõ ràng với anchor name `cell-N` để khi render trên GitHub / web view vẫn navigation được.

---

## 📚 Mục Lục

### Phần Khai Báo
- [Cell 0: Title](../../notebook_course_work/CourseWork.ipynb) — Tiêu đề notebook
- [Cell 1: Define Problem](../../notebook_course_work/CourseWork.ipynb) — Định nghĩa bài toán
- [Cell 2: Imports](../../notebook_course_work/CourseWork.ipynb) — Import thư viện
- [Cell 3: Display setup](../../notebook_course_work/CourseWork.ipynb) — IPython display

### Giai Đoạn Foundation (Phases 1-11)
- [Cell 4-5: Phase 1 — Environment](../../notebook_course_work/CourseWork.ipynb)
- [Cell 6-7: Phase 2 — Data Acquisition](../../notebook_course_work/CourseWork.ipynb)
- [Cell 8-9: Phase 3 — Schema Audit](../../notebook_course_work/CourseWork.ipynb)
- [Cell 10-11: Phase 4 — Temporal Integrity](../../notebook_course_work/CourseWork.ipynb)
- [Cell 12-13: Phase 5 — Chronological Split](../../notebook_course_work/CourseWork.ipynb)
- [Cell 14-43: Phase 6 — EDA (14 sub-cells)](../../notebook_course_work/CourseWork.ipynb)
- [Cell 44-45: Phase 7 — Feature Engineering](../../notebook_course_work/CourseWork.ipynb)
- [Cell 46-47: Phase 8 — Feature-Set Variants](../../notebook_course_work/CourseWork.ipynb)
- [Cell 48-49: Phase 9 — Train-Only Scaling](../../notebook_course_work/CourseWork.ipynb)
- [Cell 50-51: Phase 10 — Window Builder](../../notebook_course_work/CourseWork.ipynb)
- [Cell 52-53: Phase 11 — DataLoaders](../../notebook_course_work/CourseWork.ipynb)

### Giai Đoạn Modeling (Phases 12-22)
- [Cell 54-55: Phase 12 — Shared Metrics](../../notebook_course_work/CourseWork.ipynb)
- [Cell 56-57: Phase 13 — Experiment Registry](../../notebook_course_work/CourseWork.ipynb)
- [Cell 58-59: Phase 14 — Persistence Baseline](../../notebook_course_work/CourseWork.ipynb)
- [Cell 60-61: Phase 15 — LSTM Implementation](../../notebook_course_work/CourseWork.ipynb)
- [Cell 62-63: Phase 16 — Transformer Implementation](../../notebook_course_work/CourseWork.ipynb)
- [Cell 64-65: Phase 17 — Attention Verification](../../notebook_course_work/CourseWork.ipynb)
- [Cell 66-67: Phase 18 — Forward Sanity](../../notebook_course_work/CourseWork.ipynb)
- [Cell 68-69: Phase 19 — Training Engine](../../notebook_course_work/CourseWork.ipynb)
- [Cell 70-71: Phase 20 — LSTM Baseline Run](../../notebook_course_work/CourseWork.ipynb)
- [Cell 72-73: Phase 21 — Transformer B0 Run](../../notebook_course_work/CourseWork.ipynb)
- [Cell 74-75: Phase 22 — Learning-Curve Diagnostics](../../notebook_course_work/CourseWork.ipynb)

### Giai Đoạn Sweeps (Phases 23-41)
- [Cell 76-77: Phase 23 — S1 Feature-Set](../../notebook_course_work/CourseWork.ipynb)
- [Cell 78-79: Phase 24 — S2 Time-Feature](../../notebook_course_work/CourseWork.ipynb)
- [Cell 80-81: Phase 25 — S3 Target-Scaling](../../notebook_course_work/CourseWork.ipynb)
- [Cell 82-83: Phase 26 — S4 Lookback](../../notebook_course_work/CourseWork.ipynb)
- [Cell 84-85: Phase 27 — S5 Pooling](../../notebook_course_work/CourseWork.ipynb)
- [Cell 86-87: Phase 28 — S6 Activation](../../notebook_course_work/CourseWork.ipynb)
- [Cell 88-89: Phase 29 — S7 Batch-Size](../../notebook_course_work/CourseWork.ipynb)
- [Cell 90-91: Phase 30 — S8 Learning-Rate](../../notebook_course_work/CourseWork.ipynb)
- [Cell 92-93: Phase 31 — S9 Weight-Decay](../../notebook_course_work/CourseWork.ipynb)
- [Cell 94-95: Phase 32 — S10 Dropout](../../notebook_course_work/CourseWork.ipynb)
- [Cell 96-97: Phase 33 — Transformer Config](../../notebook_course_work/CourseWork.ipynb)
- [Cell 98-99: Phase 34 — S12 Head](../../notebook_course_work/CourseWork.ipynb)
- [Cell 100-101: Phase 35 — S13 Layer](../../notebook_course_work/CourseWork.ipynb)
- [Cell 102-103: Phase 36 — S14 FFN](../../notebook_course_work/CourseWork.ipynb)
- [Cell 104-105: Phase 37 — S15 Loss](../../notebook_course_work/CourseWork.ipynb)
- [Cell 106-107: Phase 38 — S16 Epoch-Cap](../../notebook_course_work/CourseWork.ipynb)
- [Cell 108-109: Phase 39 — S17 Gradient-Clip](../../notebook_course_work/CourseWork.ipynb)
- [Cell 110-111: Phase 40 — S18 RevIN](../../notebook_course_work/CourseWork.ipynb)
- [Cell 112-113: Phase 41 — S19 Boundary](../../notebook_course_work/CourseWork.ipynb)

### Giai Đoạn Final Pipeline (Phases 42-47)
- [Cell 114-115: Phase 42 — Candidate Synthesis](../../notebook_course_work/CourseWork.ipynb)
- [Cell 116-117: Phase 43 — LSTM Tuning](../../notebook_course_work/CourseWork.ipynb)
- [Cell 118-119: Phase 44 — Rolling-Origin](../../notebook_course_work/CourseWork.ipynb)
- [Cell 120-121: Phase 45 — Final Model Lock](../../notebook_course_work/CourseWork.ipynb)
- [Cell 122-123: Phase 46 — Three-Seed Runs](../../notebook_course_work/CourseWork.ipynb)
- [Cell 124-125: Phase 47 — Final Test Evaluation](../../notebook_course_work/CourseWork.ipynb)

### Giai Đoạn Analysis (Phases 48-59)
- [Cell 126-127: Phase 48 — Prediction Analysis](../../notebook_course_work/CourseWork.ipynb)
- [Cell 128-129: Phase 49 — Residual Analysis](../../notebook_course_work/CourseWork.ipynb)
- [Cell 130-131: Phase 50 — Error-by-Regime](../../notebook_course_work/CourseWork.ipynb)
- [Cell 132-133: Phase 51 — Worst-Error](../../notebook_course_work/CourseWork.ipynb)
- [Cell 134-135: Phase 52 — Attention Extraction](../../notebook_course_work/CourseWork.ipynb)
- [Cell 136-137: Phase 53 — Attention Heatmaps](../../notebook_course_work/CourseWork.ipynb)
- [Cell 138-139: Phase 54 — Last-Query Attention](../../notebook_course_work/CourseWork.ipynb)
- [Cell 140-141: Phase 55 — Head Comparison](../../notebook_course_work/CourseWork.ipynb)
- [Cell 142-143: Phase 56 — Error-Conditioned](../../notebook_course_work/CourseWork.ipynb)
- [Cell 144-145: Phase 57 — Seed-Stability](../../notebook_course_work/CourseWork.ipynb)
- [Cell 146-147: Phase 58 — Final Tables](../../notebook_course_work/CourseWork.ipynb)
- [Cell 148: Phase 59 — Final Conclusions](../../notebook_course_work/CourseWork.ipynb)
- [Cell 149-151: Supplementary MAPE Addendum](../../notebook_course_work/CourseWork.ipynb)

---

## Phần Khai Báo (Cells 0-3)

### <a id="cell-0"></a>Cell 0 — Title

**Cell ID:** `coursewo`
**Loại:** Markdown

**Nội dung:** Tiêu đề `# MULTIVARIATE TIME-SERIES REGRESSION` cho notebook.

**Giải thích:**
- Đây là markdown cell đầu tiên, không có output.
- Dùng để giới thiệu tổng quan project là **Multivariate Time-Series Regression** trên dataset UCI Appliances Energy Prediction.

📎 [Mở cell này trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell có text `# MULTIVARIATE TIME-SERIES REGRESSION`

---

### <a id="cell-1"></a>Cell 1 — Define Problem

**Cell ID:** `define-p`
**Loại:** Markdown

**Nội dung:** Markdown `## Define Problem`.

**Giải thích:**
- Markdown cell định nghĩa bài toán, không có output.

📎 [Mở cell này trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell có text `## Define Problem`

---

### <a id="cell-2"></a>Cell 2 — Imports

**Cell ID:** `0caba13e`
**Loại:** Code

**Nội dung:** `import sys`, các imports hệ thống.

**Output kỳ vọng:** Không có output stdout (chỉ là imports). Nếu có lỗi sẽ xuất `ModuleNotFoundError` hoặc `ImportError`.

**Giải thích:**
- Import `sys` và các thiết lập path.
- Đây là cell import đầu tiên, nếu fail sẽ chặn toàn bộ pipeline.

📎 [Mở cell này trong notebook](../../notebook_course_work/CourseWork.ipynb) → bấm vào cell chứa `import sys`

---

### <a id="cell-3"></a>Cell 3 — Display Setup

**Cell ID:** `public-a`
**Loại:** Code

**Nội dung:** `from IPython.display import Image, display`.

**Output kỳ vọng:** Không có output. Chỉ là import.

**Giải thích:**
- Import `Image` và `display` từ IPython để hiển thị ảnh PNG (EDA figures, heatmaps) inline trong notebook.

📎 [Mở cell này trong notebook](../../notebook_course_work/CourseWork.ipynb) → bấm vào cell chứa `from IPython.display import Image, display`

---

## Giai Đoạn Foundation (Cells 4-53)

### <a id="cell-4"></a>Cell 4-5 — Phase 1: Environment

**Cell 4 (markdown):** `phase-1-` ID, tiêu đề `## Phase 1 - Environment`
**Cell 5 (code):** `phase-1-` ID, chạy `materialize_phase_1(PROJECT_ROOT)`

**Output kỳ vọng của Cell 5:**
- In ra một dictionary với environment fingerprint (Python version, torch version, deterministic mode, ...).
- Ghi file `artifacts/environment/environment_report.json`.
- Ghi file `artifacts/environment/phase_1_signoff.json` với status `OK`.

**Giải thích:**
- Phase 1 capture environment, freeze requirements, smoke test imports.
- Nếu output báo lỗi về kernel hoặc import, kiểm tra lại `requirements_freeze.txt`.
- Sign-off `OK` là điều kiện tiên quyết để chạy Phase 2.

📎 [Mở Cell 5 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `phase_1_signoff = materialize_phase_1`

---

### <a id="cell-6"></a>Cell 6-7 — Phase 2: Data Acquisition

**Cell 6 (markdown):** Tiêu đề `## Phase 2 - Data Acquisition`
**Cell 7 (code):** Chạy `materialize_phase_2(PROJECT_ROOT)`

**Output kỳ vọng của Cell 7:**
- Dictionary chứa dataset manifest: rows, columns, source SHA256, acquisition timestamp.
- File `data/raw_data/energydata_complete.csv` được materialize.
- File `artifacts/acquisition/dataset_manifest.json` và `phase_2_signoff.json` được tạo.

**Giải thích:**
- Phase kiểm tra SHA256 checksum của raw data, đảm bảo dữ liệu đúng với nguồn.
- Nếu checksum mismatch, xem file `phase_2_signoff.json` để biết lý do.
- Output thường có `n_rows ≈ 19735` (mẫu 10 phút × ~4.5 tháng).

📎 [Mở Cell 7 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `phase_2_signoff = materialize_phase_2`

---

### <a id="cell-8"></a>Cell 8-9 — Phase 3: Schema Audit

**Cell 8 (markdown):** Tiêu đề `## Phase 3 - Schema Audit`
**Cell 9 (code):** Chạy `materialize_phase_3(PROJECT_ROOT)`

**Output kỳ vọng của Cell 9:**
- Manifest: 29 columns, dtypes cho mỗi biến, missing counts.
- File `artifacts/schema/{schema_manifest.json, schema_summary.csv, variable_dictionary.csv}`.

**Giải thích:**
- Schema audit đảm bảo dtypes và cấu trúc cột khớp với contract.
- `variable_dictionary.csv` chứa metadata cho mỗi biến (đơn vị, role, range).

📎 [Mở Cell 9 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `phase_3_signoff = materialize_phase_3`

---

### <a id="cell-10"></a>Cell 10-11 — Phase 4: Temporal Integrity

**Cell 10 (markdown):** Tiêu đề `## Phase 4 - Temporal Integrity Audit`
**Cell 11 (code):** Chạy `materialize_phase_4(PROJECT_ROOT)`

**Output kỳ vọng của Cell 11:**
- Thống kê: cadence = 10 phút, no gaps, no duplicates, continuity segments.
- File `artifacts/temporal/{temporal_manifest.json, daily_observation_counts.csv, interval_distribution.csv}`.

**Giải thích:**
- Verify time series đều đặn ở cadence 10 phút.
- Nếu có gaps, sẽ có `continuity_segments.csv` liệt kê các đoạn liên tục.
- Đây là gate quan trọng trước khi EDA.

📎 [Mở Cell 11 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `phase_4_signoff = materialize_phase_4`

---

### <a id="cell-12"></a>Cell 12-13 — Phase 5: Chronological Split

**Cell 12 (markdown):** Tiêu đề `## Phase 5 - Chronological Split`
**Cell 13 (code):** Chạy `materialize_phase_5(PROJECT_ROOT)`

**Output kỳ vọng của Cell 13:**
- Split statistics: Train 70% / Validation 15% / Test 15%.
- File `artifacts/splits/{split_manifest.json, split_membership.csv, split_boundaries.csv, split_leakage_audit.csv}`.

**Giải thích:**
- Split chronological: KHÔNG shuffle (giữ nguyên thứ tự thời gian).
- Leakage audit đảm bảo không có sample overlap giữa Train/Val/Test.
- Boundaries xác định timestamp cắt: thường Train `≤ 2016-04-15`, Val `2016-04-15 → 2016-05-12`, Test `≥ 2016-05-12`.

📎 [Mở Cell 13 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `phase_5_signoff = materialize_phase_5`

---

### <a id="cell-14"></a>Cell 14-43 — Phase 6: Exploratory Data Analysis (14 sub-sections)

**Cell 14 (markdown):** Tiêu đề `## Phase 6 - Exploratory Data Analysis`
**Cell 15 (code):** `from io import BytesIO`, đọc interim CSV

**Các sub-cells của Phase 6:**

| Sub-cell | Markdown Header | Code action | Output kỳ vọng |
|---|---|---|---|
| Cell 16 | `### 6.1 Data Overview` | — | — |
| Cell 17 | — | `df.head(8)` | DataFrame 8 dòng đầu, 29 cột |
| Cell 18 | `### 6.2 Missing-Value Analysis` | — | — |
| Cell 19 | — | `df.isna().sum()` | Series đếm missing mỗi cột (thường tất cả = 0) |
| Cell 20 | `### 6.3 Calendar Derivations` | — | — |
| Cell 21 | — | `df["hour"] = df["date"].dt.hour` | Không output, side-effect |
| Cell 22 | `### 6.4 Target Distribution` | — | — |
| Cell 23 | — | đọc `eda_numeric_summary.csv` | Bảng summary target |
| Cell 24 | `### 6.5 Target Timeline` | — | — |
| Cell 25 | — | `df.set_index("date")` | Setup cho plotting |
| Cell 26 | `### 6.6 Calendar Energy Patterns` | — | — |
| Cell 27 | — | display CSV | Bảng hourly profile |
| Cell 28 | `### 6.7 Feature Distributions` | — | — |
| Cell 29 | — | `Image(filename=EDA_10_...)` | Hình phân phối features |
| Cell 30 | `### 6.8 Numerical Relationships` | — | — |
| Cell 31 | — | setup sensor_columns | List tên cột cảm biến |
| Cell 32 | `### 6.9 Cross-Correlation at Short Lags` | — | — |
| Cell 33 | — | define cross-corr function | Không output |
| Cell 34 | `### 6.10 Categorical and Numerical` | — | — |
| Cell 35 | — | groupby is_weekend | Describe table |
| Cell 36 | `### 6.11 IQR Outlier Diagnostics` | — | — |
| Cell 37 | — | define IQR summary | Không output |
| Cell 38 | `### 6.12 Outlier-Smoothing Demo` | — | — |
| Cell 39 | — | df.copy(deep=True) | Không output |
| Cell 40 | `### 6.13 Time-Series Diagnostics` | — | — |
| Cell 41 | — | đọc lag correlations | Bảng lag correlation |
| Cell 42 | `### 6.14 Extreme Samples` | — | — |
| Cell 43 | — | đọc extreme_target_samples | Bảng extreme samples |

**Giải thích Phase 6:**
- Phase dài nhất trong foundation (30 cells), mục đích khám phá dữ liệu toàn diện.
- Kỳ vọng output: tables (CSV) + figures (PNG) được đọc/hiển thị inline.
- Tất cả figures nằm trong `artifacts/eda/figures/`, tables trong `artifacts/eda/tables/`.

📎 [Mở Phase 6 section trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cells từ `## Phase 6 - Exploratory Data Analysis` đến trước `## Phase 7`

---

### <a id="cell-44"></a>Cell 44-45 — Phase 7: Feature Engineering

**Cell 44 (markdown):** Tiêu đề `## Phase 7 - Feature Engineering`
**Cell 45 (code):** Chạy `materialize_phase_7(PROJECT_ROOT)`

**Output kỳ vọng của Cell 45:**
- Manifest với feature list (time features, lag features, rolling stats).
- File `data/interim/uci_appliances_energy_prediction/energydata_feature_engineered_v1.csv`.
- File `artifacts/features/{feature_engineering_manifest.json, feature_registry.csv, feature_lineage.csv}`.

**Giải thích:**
- Engineering features chỉ dựa trên Train, KHÔNG leak từ Val/Test.
- File `feature_engineered_v1.csv` có ~33 cột (29 gốc + 4 time + lag + rolling).
- SHA256 file `feature_engineered_v1.sha256` đảm bảo determinism.

📎 [Mở Cell 45 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `phase_7_signoff = materialize_phase_7`

---

### <a id="cell-46"></a>Cell 46-47 — Phase 8: Feature-Set Variants

**Cell 46 (markdown):** Tiêu đề `## Phase 8 - Feature-Set Variants`
**Cell 47 (code):** Chạy `materialize_phase_8(PROJECT_ROOT)`

**Output kỳ vọng của Cell 47:**
- Registry 3 variants: FS0, FS1, FS2.
- File `artifacts/feature_sets/{feature_set_registry.json, feature_components.json, feature_set_lineage.csv}`.

**Giải thích:**
- 3 variants để sweep Phase 23 (S1).
- FS0 = minimal, FS1 = + time, FS2 = + lag features.

📎 [Mở Cell 47 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `phase_8_signoff = materialize_phase_8`

---

### <a id="cell-48"></a>Cell 48-49 — Phase 9: Train-Only Scaling

**Cell 48 (markdown):** Tiêu đề `## Phase 9 - Train-Only Scaling`
**Cell 49 (code):** Chạy `materialize_phase_9(PROJECT_ROOT)`

**Output kỳ vọng của Cell 49:**
- Scaler fingerprints (StandardScaler for X, StandardScaler for y).
- File `artifacts/scalers/{x,y}/*.joblib`.
- File `artifacts/scaling/{scaler_registry.json, scaling_audit.csv, scaling_discrepancies.json}`.

**Giải thích:**
- Scaler được FIT trên Train only.
- Val và Test được TRANSFORM bằng scaler của Train → không leak.
- Output thường chứa `scaler.mean_` và `scaler.scale_` per feature.

📎 [Mở Cell 49 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `phase_9_signoff = materialize_phase_9`

---

### <a id="cell-50"></a>Cell 50-51 — Phase 10: Window Builder

**Cell 50 (markdown):** Tiêu đề `## Phase 10 - Window Builder`
**Cell 51 (code):** Chạy `materialize_phase_10(PROJECT_ROOT)`

**Output kỳ vọng của Cell 51:**
- Window population summary.
- File `artifacts/windows/{window_manifest.json, window_population_summary.csv, window_fingerprints.json}`.

**Giải thích:**
- Build (X, y) pairs với lookback ∈ {36, 72, 144}.
- Population phải giảm dần: Train > Val > Test (do mỗi sample cần lookback).

📎 [Mở Cell 51 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `phase_10_signoff = materialize_phase_10`

---

### <a id="cell-52"></a>Cell 52-53 — Phase 11: DataLoaders

**Cell 52 (markdown):** Tiêu đề `## Phase 11 - DataLoaders`
**Cell 53 (code):** Chạy `materialize_phase_11(PROJECT_ROOT)`

**Output kỳ vọng của Cell 53:**
- DataLoader configs: batch_size, shuffle policy (Train=shuffle, Val/Test=no shuffle).
- File `artifacts/dataloaders/{dataloader_manifest.json, dataloader_registry.csv, sequential_order_audit.csv}`.

**Giải thích:**
- Train DataLoader shuffle với seed cố định.
- Val/Test DataLoader KHÔNG shuffle để giữ thứ tự thời gian.

📎 [Mở Cell 53 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `phase_11_signoff = materialize_phase_11`

---

## Giai Đoạn Modeling (Cells 54-75)

### <a id="cell-54"></a>Cell 54-55 — Phase 12: Shared Metrics

**Cell 54 (markdown):** Tiêu đề `## Phase 12 - Shared Metrics`
**Cell 55 (code):** Chạy `materialize_phase_12(PROJECT_ROOT)`

**Output kỳ vọng của Cell 55:**
- Manifest với metrics: MAE, RMSE, R², MAPE (addendum).
- File `artifacts/metrics/{metric_manifest.json, metric_unit_tests.csv, metric_reference_examples.csv}`.

**Giải thích:**
- Metrics tính ở đơn vị Wh gốc (inverse_transform y_pred, y_true).
- R² ở Wh scale (không phải scaled space).

📎 [Mở Cell 55 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `phase_12_signoff = materialize_phase_12`

---

### <a id="cell-56"></a>Cell 56-57 — Phase 13: Experiment Registry

**Cell 56 (markdown):** Tiêu đề `## Phase 13 - Experiment Registry`
**Cell 57 (code):** Chạy `materialize_phase_13(PROJECT_ROOT)`

**Output kỳ vọng của Cell 57:**
- Empty registry (chưa có run nào).
- File `artifacts/experiments/{experiment_registry.jsonl, registry_manifest.json, sweep_registry.csv}`.

**Giải thích:**
- Registry sẽ được fill dần khi các sweep phases chạy.
- Mỗi training run được register với: id, config, status, metrics.

📎 [Mở Cell 57 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `phase_13_signoff = materialize_phase_13`

---

### <a id="cell-58"></a>Cell 58-59 — Phase 14: Persistence Baseline

**Cell 58 (markdown):** Tiêu đề `## Phase 14 - Persistence Baseline`
**Cell 59 (code):** Chạy `materialize_phase_14(PROJECT_ROOT)`

**Output kỳ vọng của Cell 59:**
- Validation metrics cho persistence: MAE ≈ 26 Wh, RMSE ≈ 47 Wh, R² ≈ 0.20.
- File `artifacts/baselines/persistence/{persistence_manifest.json, persistence_validation_metrics.json}`.

**Giải thích:**
- Persistence: $\hat{y}[t+1] = y[t]$ (giá trị step trước).
- Đây là baseline yếu nhất, nhưng khó bị đánh bại ở những biến động nhỏ.
- Nếu LSTM/Transformer không beat được → có vấn đề về features hoặc training.

📎 [Mở Cell 59 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `phase_14_signoff = materialize_phase_14`

---

### <a id="cell-60"></a>Cell 60-61 — Phase 15: LSTM Implementation

**Cell 60 (markdown):** Tiêu đề `## Phase 15 - LSTM Implementation`
**Cell 61 (code):** Chạy `materialize_phase_15(PROJECT_ROOT)`

**Output kỳ vọng của Cell 61:**
- LSTM architecture fingerprint: hidden_size, num_layers, dropout, total_params.
- File `artifacts/models/lstm/{lstm_model_manifest.json, lstm_shape_contract.json, lstm_unit_tests.csv}`.

**Giải thích:**
- Implement LSTM regressor (KHÔNG train ở phase này).
- Shape contract đảm bảo input `(batch, lookback, n_features)` → output `(batch, 1)`.
- Unit tests verify shape correctness trên dummy input.

📎 [Mở Cell 61 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `phase_15_signoff = materialize_phase_15`

---

### <a id="cell-62"></a>Cell 62-63 — Phase 16: Transformer Implementation

**Cell 62 (markdown):** Tiêu đề `## Phase 16 - Transformer Implementation`
**Cell 63 (code):** Chạy `materialize_phase_16(PROJECT_ROOT)`

**Output kỳ vọng của Cell 63:**
- Transformer architecture fingerprint: d_model, num_heads, num_layers, ffn_dim, total_params.
- File `artifacts/models/transformer/{transformer_model_manifest.json, transformer_attention_contract.json, transformer_shape_contract.json, ...}`.

**Giải thích:**
- Implement Transformer encoder regressor (KHÔNG train).
- Attention contract: output attention shape `(batch, num_heads, seq, seq)`.
- Positional encoding sin/cos được add vào input embedding.

📎 [Mở Cell 63 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `phase_16_signoff = materialize_phase_16`

---

### <a id="cell-64"></a>Cell 64-65 — Phase 17: Attention Verification

**Cell 64 (markdown):** Tiêu đề `## Phase 17 - Attention-Aware Encoder Verification`
**Cell 65 (code):** Chạy `materialize_phase_17(PROJECT_ROOT)`

**Output kỳ vọng của Cell 65:**
- Audit reports: shape audit, probability audit (sum=1), mask audit, path equivalence.
- File `artifacts/attention_verification/{attention_verification_manifest.json, attention_probability_audit.csv, attention_path_equivalence_audit.csv, ...}`.

**Giải thích:**
- Verify attention weights là valid probability distribution.
- Path equivalence: attention từ manual computation == framework output.
- Nếu FAIL ở phase này → Phase 18 sẽ không pass.

📎 [Mở Cell 65 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `phase_17_signoff = materialize_phase_17`

---

### <a id="cell-66"></a>Cell 66-67 — Phase 18: Forward-Pass Sanity

**Cell 66 (markdown):** Tiêu đề `## Phase 18 - Forward-Pass Sanity Tests`
**Cell 67 (code):** Chạy `materialize_phase_18(PROJECT_ROOT)`

**Output kỳ vọng của Cell 67:**
- Forward pass tests: batch independence, parameter non-mutation, device transfer, scaling correctness.
- File `artifacts/forward_sanity/{forward_sanity_manifest.json, forward_batch_audit.csv, ...}`.

**Giải thích:**
- Sanity tests trên model đã initialized (chưa trained).
- Đảm bảo forward pass deterministic và không side-effect.

📎 [Mở Cell 67 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `phase_18_signoff = materialize_phase_18`

---

### <a id="cell-68"></a>Cell 68-69 — Phase 19: Training Engine

**Cell 68 (markdown):** Tiêu đề `## Phase 19 - Baseline Training Engine`
**Cell 69 (code):** Chạy `materialize_phase_19(PROJECT_ROOT)`

**Output kỳ vọng của Cell 69:**
- Training engine contract: optimizer template, scheduler template, checkpoint schema.
- File `artifacts/training_engine/{training_engine_manifest.json, training_engine_unit_tests.csv, checkpoint_schema.json}`.

**Giải thích:**
- Implement generic training loop (KHÔNG train model thật ở đây).
- Engine hỗ trợ: gradient clipping, early stopping, checkpoint save/load.

📎 [Mở Cell 69 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `phase_19_signoff = materialize_phase_19`

---

### <a id="cell-70"></a>Cell 70-71 — Phase 20: LSTM Baseline Run

**Cell 70 (markdown):** Tiêu đề `## Phase 20 - LSTM Baseline Run`
**Cell 71 (code):** Chạy `materialize_phase_20(PROJECT_ROOT)`

**Output kỳ vọng của Cell 71:**
- LSTM training log + final validation metrics.
- File `artifacts/runs/RUN_LS_LS_*/{config.json, status.json, training.log, training_history.csv, metrics/best_validation_metrics.json}`.

**Giải thích:**
- Train LSTM baseline với reference config.
- Validation RMSE thường ≈ 73 Wh (vs persistence 47 Wh → LSTM ban đầu chưa beat persistence vì hyperparameters chưa tuned).

📎 [Mở Cell 71 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `phase_20_signoff = materialize_phase_20`

---

### <a id="cell-72"></a>Cell 72-73 — Phase 21: Transformer B0 Run

**Cell 72 (markdown):** Tiêu đề `## Phase 21 - Transformer B0 Run`
**Cell 73 (code):** Chạy `materialize_phase_21(PROJECT_ROOT)`

**Output kỳ vọng của Cell 73:**
- Transformer B0 training log + validation metrics.
- File `artifacts/runs/RUN_TR_B0_*/` (config, status, history, metrics).

**Giải thích:**
- Train Transformer B0 (baseline config chưa tune).
- Thường có RMSE ≈ 60-70 Wh.

📎 [Mở Cell 73 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `phase_21_signoff = materialize_phase_21`

---

### <a id="cell-74"></a>Cell 74-75 — Phase 22: Learning-Curve Diagnostics

**Cell 74 (markdown):** Tiêu đề `## Phase 22 - Learning-Curve Diagnostics`
**Cell 75 (code):** `render_phase_summary(...)` cho Phase 20/21

**Output kỳ vọng của Cell 75:**
- HTML dashboard hiển thị learning curves cho các baseline runs.
- So sánh LSTM vs Transformer B0 loss curves.

**Giải thích:**
- Phase chẩn đoán: Train vs Validation loss có converge không?
- Phát hiện overfitting/underfitting sớm để biết sweep nào cần thiết.

📎 [Mở Cell 75 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `render_phase_summary`

---

## Giai Đoạn Sweeps (Cells 76-113)

Mỗi sweep phase gồm 1 markdown + 1 code cell. Code cell sẽ:
1. Load reference config từ phase trước.
2. Run sweep variants.
3. Pick winner bằng validation RMSE.
4. Update reference config.

### <a id="cell-76"></a>Cell 76-77 — Phase 23: S1 Feature-Set

**Cell 76:** Markdown tiêu đề `## Phase 23 - S1 Feature-Set Sweep`
**Cell 77:** `render_phase_resume(...)`

**Output kỳ vọng:** Bảng CSV `results.csv` với các variants FS0, FS1, FS2 và validation metrics. Winner = best validation RMSE.

**Giải thích:**
- Sweep 3 feature sets.
- Update reference → Phase 24 sẽ dùng winner.

📎 [Mở Cell 77 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell có `Phase 23 - S1 Feature-Set Sweep`

---

### <a id="cell-78"></a>Cell 78-79 — Phase 24: S2 Time-Feature

**Cell 78:** Markdown tiêu đề
**Cell 79:** `render_phase_resume(...)`

**Output kỳ vọng:** Sweep time features (hour sin/cos, dayofweek, is_weekend, ...).

📎 [Mở Cell 79 trong notebook](../../notebook_course_work/CourseWork.ipynb)

---

### <a id="cell-80"></a>Cell 80-81 — Phase 25: S3 Target-Scaling

Sweep target scaling: StandardScaler, RobustScaler, MinMaxScaler, no-scaling.

📎 [Mở Cell 81 trong notebook](../../notebook_course_work/CourseWork.ipynb)

---

### <a id="cell-82"></a>Cell 82-83 — Phase 26: S4 Lookback

Sweep lookback ∈ {36, 72, 144}.

📎 [Mở Cell 83 trong notebook](../../notebook_course_work/CourseWork.ipynb)

---

### <a id="cell-84"></a>Cell 84-85 — Phase 27: S5 Pooling

Sweep pooling: last, mean, max, attention.

📎 [Mở Cell 85 trong notebook](../../notebook_course_work/CourseWork.ipynb)

---

### <a id="cell-86"></a>Cell 86-87 — Phase 28: S6 Activation

Sweep activation: ReLU, GELU, SiLU.

📎 [Mở Cell 87 trong notebook](../../notebook_course_work/CourseWork.ipynb)

---

### <a id="cell-88"></a>Cell 88-89 — Phase 29: S7 Batch-Size

Sweep batch_size ∈ {16, 32, 64, 128, 256}.

📎 [Mở Cell 89 trong notebook](../../notebook_course_work/CourseWork.ipynb)

---

### <a id="cell-90"></a>Cell 90-91 — Phase 30: S8 Learning-Rate

Sweep learning_rate ∈ {1e-4, 5e-4, 1e-3, 5e-3}.

📎 [Mở Cell 91 trong notebook](../../notebook_course_work/CourseWork.ipynb)

---

### <a id="cell-92"></a>Cell 92-93 — Phase 31: S9 Weight-Decay

**Cell 92:** `phase-31` ID
**Cell 93:** `render_phase_resume(...)`

Sweep weight_decay ∈ {0, 1e-5, 1e-4, 1e-3}.

📎 [Mở Cell 93 trong notebook](../../notebook_course_work/CourseWork.ipynb)

---

### <a id="cell-94"></a>Cell 94-95 — Phase 32: S10 Dropout

**Cell 94:** `phase-32` ID
**Cell 95:** `render_phase_resume(...)`

Sweep dropout ∈ {0.0, 0.1, 0.2, 0.3}.

📎 [Mở Cell 95 trong notebook](../../notebook_course_work/CourseWork.ipynb)

---

### <a id="cell-96"></a>Cell 96-97 — Phase 33: Transformer Configuration

**Cell 96:** Markdown tiêu đề `## Transformer Configuration after Phase 33`
**Cell 97:** `render_phase_33_transformer_configuration(...)`

**Output kỳ vọng:** Dashboard hiển thị toàn bộ Transformer config đã được update qua S1-S11 sweeps.

**Giải thích:**
- Snapshot Transformer config TRƯỚC khi bắt đầu architecture sweeps (S12-S19).
- Config này sẽ được dùng làm baseline cho S12-S19.

📎 [Mở Cell 97 trong notebook](../../notebook_course_work/CourseWork.ipynb)

---

### <a id="cell-98"></a>Cell 98-99 — Phase 34: S12 Head

**Cell 98:** `phase-34` ID
**Cell 99:** `build_phase_processing_log, render_phase_log(...)`

Sweep `num_heads ∈ {2, 4, 8, 16}`.

📎 [Mở Cell 99 trong notebook](../../notebook_course_work/CourseWork.ipynb)

---

### <a id="cell-100"></a>Cell 100-101 — Phase 35: S13 Layer

Sweep `num_layers ∈ {1, 2, 3, 4, 6}`.

📎 [Mở Cell 101 trong notebook](../../notebook_course_work/CourseWork.ipynb)

---

### <a id="cell-102"></a>Cell 102-103 — Phase 36: S14 FFN

Sweep FFN width multipliers ∈ {1, 2, 4}.

📎 [Mở Cell 103 trong notebook](../../notebook_course_work/CourseWork.ipynb)

---

### <a id="cell-104"></a>Cell 104-105 — Phase 37: S15 Loss

Sweep loss: MSE, Huber, Smooth L1.

📎 [Mở Cell 105 trong notebook](../../notebook_course_work/CourseWork.ipynb)

---

### <a id="cell-106"></a>Cell 106-107 — Phase 38: S16 Epoch-Cap

Sweep max_epochs ∈ {30, 50, 80, 100}.

📎 [Mở Cell 107 trong notebook](../../notebook_course_work/CourseWork.ipynb)

---

### <a id="cell-108"></a>Cell 108-109 — Phase 39: S17 Gradient-Clip

Sweep gradient_clip ∈ {0.5, 1.0, 5.0, no_clip}.

📎 [Mở Cell 109 trong notebook](../../notebook_course_work/CourseWork.ipynb)

---

### <a id="cell-110"></a>Cell 110-111 — Phase 40: S18 RevIN

Sweep RevIN on/off.

📎 [Mở Cell 111 trong notebook](../../notebook_course_work/CourseWork.ipynb)

---

### <a id="cell-112"></a>Cell 112-113 — Phase 41: S19 Boundary-Protocol

Test boundary handling.

📎 [Mở Cell 113 trong notebook](../../notebook_course_work/CourseWork.ipynb)

---

## Giai Đoạn Final Pipeline (Cells 114-125)

### <a id="cell-114"></a>Cell 114-115 — Phase 42: Candidate Synthesis

**Cell 114:** Markdown `## Phase 42 - Candidate Synthesis`
**Cell 115:** `materialize_phase_42(PROJECT_ROOT)`

**Output kỳ vọng:**
- Transformer candidate shortlist + selected lineage.
- File `artifacts/candidate_synthesis/{candidate_synthesis_manifest.json, transformer_candidate_shortlist.json, selected_lineage.json, baseline_anchor_context.json, boundary_sensitivity_context.json, candidate_synthesis_report.md}`.

**Giải thích:**
- Tổng hợp toàn bộ sweep winners.
- Chọn candidate để đưa vào rolling origin (Phase 44).

📎 [Mở Cell 115 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `phase_42_signoff = materialize_phase_42`

---

### <a id="cell-116"></a>Cell 116-117 — Phase 43: LSTM Tuning

**Cell 116:** Markdown `## Phase 43 - LSTM Tuning`
**Cell 117:** `render_phase_43_dashboard(...)`

**Output kỳ vọng:** Dashboard LSTM tuning stages (lt1-lt5).

**Giải thích:**
- Tune LSTM hyperparameters theo stages.
- Mỗi stage sweep 1 hyperparameter.

📎 [Mở Cell 117 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `render_phase_43_dashboard`

---

### <a id="cell-118"></a>Cell 119-119 — Phase 44: Rolling-Origin

**Cell 118:** Markdown
**Cell 119:** `render_phase_44_dashboard(...)`

**Output kỳ vọng:** Dashboard rolling origin folds (5 folds × candidates).

**Giải thích:**
- Robustness evaluation qua time slices.
- Robust Lane: full refit mỗi fold.

📎 [Mở Cell 119 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `render_phase_44_dashboard`

---

### <a id="cell-120"></a>Cell 120-121 — Phase 45: Final Model Lock

**Cell 120:** Markdown `## Phase 45 - Final Model Lock`
**Cell 121:** `render_phase_45_dashboard(...)`

**Output kỳ vọng:** Dashboard locked config (architecture, optimizer, loss, scaler, ...).

**Giải thích:**
- **NO-TRAIN phase** — chỉ lock config.
- Sau khi lock, không được thay đổi gì.

📎 [Mở Cell 121 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `render_phase_45_dashboard`

---

### <a id="cell-122"></a>Cell 122-123 — Phase 46: Three-Seed Final Runs

**Cell 122:** Markdown
**Cell 123:** `render_phase_46_dashboard(...)`

**Output kỳ vọng:** Dashboard 3-seed runs (42, 123, 2026).

**Giải thích:**
- Train final model với 3 seeds.
- Mỗi seed tạo FINAL_REFIT checkpoint.

📎 [Mở Cell 123 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `render_phase_46_dashboard`

---

### <a id="cell-124"></a>Cell 124-125 — Phase 47: Final Test Evaluation

**Cell 124:** Markdown `## Phase 47 - Final Test Evaluation`
**Cell 125:** `render_phase_47_dashboard(...)`

**Output kỳ vọng:** Dashboard FINAL TEST metrics (RMSE, MAE, R² cho 3 seeds + mean ± std).

**Giải thích:**
- **FINAL GATE** — chỉ chạy 1 lần trên Test set.
- Kết quả Phase 47 là metrics chính thức của project.

📎 [Mở Cell 125 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `render_phase_47_dashboard`

---

## Giai Đoạn Analysis (Cells 126-148)

### <a id="cell-126"></a>Cell 126-127 — Phase 48: Prediction Analysis

**Cell 126:** Markdown `## Phase 48 - Prediction Analysis`
**Cell 127:** `render_phase_48_dashboard(...)`

**Output kỳ vọng:** Dashboard phân tích prediction: actual vs predicted, ECDF, change magnitudes, lag cross-correlation.

**Giải thích:**
- Hiểu model dự đoán như thế nào trên Test set.
- Phát hiện systematic biases.

📎 [Mở Cell 127 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `render_phase_48_dashboard`

---

### <a id="cell-128"></a>Cell 128-129 — Phase 49: Residual Analysis

**Cell 128:** Markdown `## Phase 49 - Residual Analysis`
**Cell 129:** `render_phase_49_dashboard(...)`

**Output kỳ vọng:** Dashboard phân tích residual: distribution, signed bias, ACF, sign runs, persistence context.

**Giải thích:**
- Residual = y_true - y_pred.
- Check residual có zero-mean, no autocorrelation (otherwise model missing patterns).

📎 [Mở Cell 129 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `render_phase_49_dashboard`

---

### <a id="cell-130"></a>Cell 130-131 — Phase 50: Error-by-Regime

**Cell 130:** Markdown `## Phase 50 — Error-by-Regime Analysis`
**Cell 131:** `render_phase_50_dashboard(...)`

**Output kỳ vọng:** Dashboard per-regime metrics (target level, time of day, day type, extreme high, change direction).

**Giải thích:**
- Biết model sai ở đâu: low target, peak hours, weekends, ...

📎 [Mở Cell 131 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `render_phase_50_dashboard`

---

### <a id="cell-132"></a>Cell 132-133 — Phase 51: Worst-Error Analysis

**Cell 132:** Markdown `## Phase 51 — Worst-Error Analysis`
**Cell 133:** `render_phase_51_dashboard(...)`

**Output kỳ vọng:** Dashboard top-K worst errors per seed, shared worst cases, casebook.

**Giải thích:**
- Tìm các sample khó nhất (model sai nhiều nhất).
- Casebook markdown mô tả context của từng worst case.

📎 [Mở Cell 133 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `render_phase_51_dashboard`

---

### <a id="cell-134"></a>Cell 134-135 — Phase 52: Attention Extraction

**Cell 134:** Markdown `## Phase 52 — Attention Extraction`
**Cell 135:** `render_phase_52_dashboard(...)`

**Output kỳ vọng:** Dashboard extraction progress, raw attention checksums.

**Giải thích:**
- Extract attention weights từ final checkpoints cho shared worst cases.
- Raw attention lưu thành `.npy` với SHA256.

📎 [Mở Cell 135 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `render_phase_52_dashboard`

---

### <a id="cell-136"></a>Cell 136-137 — Phase 53: Attention Heatmaps

**Cell 136:** Markdown `## Phase 53 - Attention Heatmaps`
**Cell 137:** `render_phase_53_dashboard(...)`

**Output kỳ vọng:** HTML dashboard với V1 (case grids), V2 (cross-seed), V3 (individual maps) heatmaps.

**Giải thích:**
- Heatmaps trực quan hóa attention distribution.
- V3 individual maps cho mỗi (case, seed, layer, head).

📎 [Mở Cell 137 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `render_phase_53_dashboard`

---

### <a id="cell-138"></a>Cell 138-139 — Phase 54: Last-Query Attention

**Cell 138:** Markdown `## Phase 54 - Last-Query Attention Analysis`
**Cell 139:** `render_phase_54_dashboard(...)`

**Output kỳ vọng:** Dashboard last-query metrics (entropy, expected lag, top1 frequency, coverage).

**Giải thích:**
- Focus vào attention từ last query (predict t+1).
- Xem model "nhìn" bao xa trong quá khứ.

📎 [Mở Cell 139 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `render_phase_54_dashboard`

---

### <a id="cell-140"></a>Cell 140-141 — Phase 55: Head Comparison

**Cell 140:** Markdown `## Phase 55 - Head Comparison`
**Cell 141:** `render_phase_55_dashboard(...)`

**Output kỳ vọng:** Dashboard JSD/cosine/Wasserstein matrices giữa các heads.

**Giải thích:**
- So sánh behavior giữa các (layer, head).
- Phát hiện heads chuyên biệt vs heads trùng lặp.

📎 [Mở Cell 141 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `render_phase_55_dashboard`

---

### <a id="cell-142"></a>Cell 142-143 — Phase 56: Error-Conditioned Attention

**Cell 142:** Markdown `## Phase 56 - Error-Conditioned Attention`
**Cell 143:** `render_phase_56_dashboard(...)`

**Output kỳ vọng:** Dashboard so sánh attention high-error vs low-error, decile trends.

**Giải thích:**
- Model có "nhìn" khác khi sai nhiều vs sai ít?
- Cliffs Delta, signed error analysis.

📎 [Mở Cell 143 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `render_phase_56_dashboard`

---

### <a id="cell-144"></a>Cell 144-145 — Phase 57: Seed-Stability

**Cell 144:** Markdown `## Phase 57 - Seed-Stability Attention Check`
**Cell 145:** `render_phase_57_dashboard(...)`

**Output kỳ vọng:** Dashboard cross-seed attention stability, canonical head mapping.

**Giải thích:**
- Attention ổn định giữa các seeds không?
- Match heads giữa seeds bằng Hungarian algorithm.

📎 [Mở Cell 145 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `render_phase_57_dashboard`

---

### <a id="cell-146"></a>Cell 146-147 — Phase 58: Final Tables

**Cell 146:** Markdown `## Phase 58 - Final Results Summary`
**Cell 147:** `render_phase_58_dashboard(...)`

**Output kỳ vọng:** Dashboard FT01-FT10 final tables (LaTeX + Markdown).

**Giải thích:**
- Bảng tổng hợp cuối cùng.
- Định dạng LaTeX cho paper, Markdown cho README.

📎 [Mở Cell 147 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `render_phase_58_dashboard`

---

### <a id="cell-148"></a>Cell 148 — Phase 59: Final Conclusions

**Cell 148:** Markdown `## Phase 59 - Final Conclusions`

**Nội dung:** Markdown section mô tả kết luận cuối cùng (sẽ được render bằng Cell 151).

**Giải thích:**
- Phase 59 không có cell code riêng trước MAPE addendum.
- Kết luận gồm: abstract, research question answers, key takeaways, limitations, future work.

📎 [Mở Cell 148 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell có text `## Phase 59 - Final Conclusions`

---

## Phần Bổ Sung (Cells 149-151)

### <a id="cell-149"></a>Cell 149-150 — Supplementary MAPE Metric Addendum

**Cell 149:** Markdown `## Supplementary MAPE Metric Addendum`
**Cell 150:** `render_mape_addendum(...)`

**Output kỳ vọng:** Dashboard MAPE (Mean Absolute Percentage Error) cho validation/test.

**Giải thích:**
- MAPE là metric bổ sung (phase phụ).
- MAPE có nhược điểm với target gần 0, cần xử lý cẩn thận.

📎 [Mở Cell 150 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `render_mape_addendum`

---

### <a id="cell-151"></a>Cell 151 — Phase 59 Final Dashboard

**Cell 151:** `render_phase_59_dashboard(...)`

**Output kỳ vọng:** HTML dashboard tổng hợp final conclusions.

**Giải thích:**
- Cell render cuối cùng của notebook.
- Bao gồm: key takeaways, limitations, future work, viva defense notes.

📎 [Mở Cell 151 trong notebook](../../notebook_course_work/CourseWork.ipynb) → tìm cell chứa `render_phase_59_dashboard`

---

## 📖 Tài Liệu Liên Quan

| File | Mô tả |
|---|---|
| [`CURRENT_FLOW_SUMMARY.md`](./CURRENT_FLOW_SUMMARY.md) | Tổng quan kiến trúc |
| [`PHASE_1_TO_33_CURRENT_FLOW_DETAIL.md`](./PHASE_1_TO_33_CURRENT_FLOW_DETAIL.md) | Chi tiết phases 1-33 |
| [`PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md`](./PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md) | Chi tiết phases 34-59 |

---

## 💡 Cách Sử Dụng File Này

1. **Khi đọc notebook:** Mở file này song song, mỗi cell gặp output khó hiểu thì tra cứu cell tương ứng.
2. **Khi debug:** Tìm cell ID của phase đang lỗi, đọc phần giải thích để biết output kỳ vọng.
3. **Khi present:** Click vào anchor link để mở notebook ở đúng cell, tiết kiệm thời gian tìm cell.

**Phiên bản:** 06/09/2026 — sync với notebook sau lần refactor clean architecture.
