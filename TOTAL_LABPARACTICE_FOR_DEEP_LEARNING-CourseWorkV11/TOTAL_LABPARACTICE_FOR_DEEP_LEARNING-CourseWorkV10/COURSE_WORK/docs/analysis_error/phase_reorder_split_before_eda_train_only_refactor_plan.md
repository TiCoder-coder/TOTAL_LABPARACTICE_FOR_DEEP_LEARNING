# Plan: Refactor Reorder Chronological Split (Phase 5) trước EDA (Phase 6) và chuyển EDA/FE/FS sang Train-Only Scope

**Document ID:** `COURSE-WORK-REFACTOR-PHASE-REORDER-001`
**Status:** `PRE_PROCESS_PLAN_PENDING_HUMAN_APPROVAL`
**Created at:** 2026-08-17
**Author:** Deep Learning Coursework architecture analysis
**Linked architecture rule:** `COURSE-WORK-ARCHITECTURE-v1` (Status: `ACTIVE_HUMAN_APPROVED`)
**Activation gate:** Plan này chỉ thành pre-process plan đang hoạt động sau khi Human đọc và xác nhận rõ. KHÔNG tự động thực thi.

---

## Mục lục

1. [Tổng quan vấn đề](#1-tổng-quan-vấn-đề)
2. [Phân tích vấn đề hiện tại](#2-phân-tích-vấn-đề-hiện-tại)
3. [Đề xuất refactor](#3-đề-xuất-refactor)
4. [Plan thay đổi chi tiết](#4-plan-thay-đổi-chi-tiết)
5. [Lộ trình thực thi (Execution Roadmap)](#5-lộ-trình-thực-thi-execution-roadmap)
6. [Risks & Mitigations](#6-risks--mitigations)
7. [Verification Checklist](#7-verification-checklist)
8. [Phụ lục: Impact Analysis](#8-phụ-lục-impact-analysis)
9. [Khuyến nghị cho Human approval](#9-khuyến-nghị-cho-human-approval)
10. [Câu hỏi cần Human quyết định](#10-câu-hỏi-cần-human-quyết-định)

---

## 1. Tổng quan vấn đề

### 1.1. Mô tả vấn đề hiện tại

Notebook `CourseWork.ipynb` hiện đang xử lý theo thứ tự Phase 5 → 8 là:

| Phase | Nội dung | Scope dữ liệu | Vị trí notebook |
|---|---|---|---|
| 5 | Exploratory Data Analysis (EDA) | **Full dataset (19,735 rows)** | Cells 12-41 (30 cells) |
| 6 | Feature Engineering | **Full dataset** | Cells 42-43 |
| 7 | Feature-Set Variants (validation) | **Full feature_view** | Cells 44-45 |
| 8 | Chronological Split | Toàn bộ dataset mới được phân chia 70/15/15 | Cells 46-47 |

Trong khi đó, các Phase 9-14 đã chính xác duy trì nguyên tắc **Train-only scientific decision** và **Test firewall**:

- Phase 9 Scaling → `fit_x_scaler_bundle` filter `split_values != {"TRAIN"}`.
- Phase 10 Windowing → Test target policy locked.
- Phase 11 DataLoaders → 4 target access modes (`TRAIN`, `VALIDATION`, `TEST_LOCKED`, `TEST_EVALUATION`).
- Phase 12 Metrics → selection split = Validation, Test metrics gated bởi Phase 47.
- Phase 13 Registry → reject development Test targets.
- Phase 14 Persistence → Validation-only evaluation, Test firewall.

**Tuy nhiên**, từ Phase 5 → 8 hiện tại vẫn đang dùng **full dataset** để:
- Tính distribution statistics (mean, std, quantiles) trên cả 19,735 rows.
- Generate hypotheses (H-EDA-001 → H-EDA-010) dựa trên patterns có thể chỉ xuất hiện ở Validation/Test.
- Validate 6 feature-set variants trên toàn bộ feature_view.
- Phân chia split sau khi EDA đã "nhìn thấy" tất cả dữ liệu.

### 1.2. Mục tiêu refactor

Mục tiêu của refactor này là **eliminate Train-set leakage in hypothesis generation và feature engineering decision-making** bằng cách:

1. **Di chuyển Phase 8 (Chronological Split) lên trước Phase 5 (EDA)** trong thứ tự notebook.
2. **Đánh lại số phase** cho phù hợp:
   - Phase 5 mới = Chronological Split (cũ Phase 8)
   - Phase 6 mới = Exploratory Data Analysis (cũ Phase 5)
   - Phase 7 mới = Feature Engineering (cũ Phase 6)
   - Phase 8 mới = Feature-Set Variants (cũ Phase 7)
3. **Giới hạn scope Train-only** cho EDA, Feature Engineering, Feature-Set Variants:
   - EDA phân tích chỉ trên TRAIN rows.
   - Feature Engineering chỉ tính toán trên TRAIN rows trong validation/invariant scope.
   - Feature-Set Variants validate trên TRAIN rows.
4. **Bảo toàn** logic Phase 9-14 (đã đúng) — chỉ cần update upstream signoff references.

### 1.3. Phạm vi thay đổi

#### 1.3.1. Source code Python (4 files)

- `src/course_work/data/eda.py` — Modify `prepare_eda_analysis` để filter TRAIN rows.
- `src/course_work/data/features.py` — Modify `verify_phase_6_inputs` và `build_feature_view` để filter TRAIN.
- `src/course_work/data/feature_sets.py` — Modify `verify_phase_7_inputs` và `validate_feature_variants` để filter TRAIN.
- `src/course_work/reporting/phase_summary.py` — Update `PHASE_NAMES`, `LOG_FILENAMES`, `SOURCE_SPECS`, `PRESENTATION_SPECS`, `_phase_content` keys 5/6/7/8.

#### 1.3.2. Source code KHÔNG đổi logic (4 files)

- `src/course_work/data/splitting.py` — Giữ logic, chỉ thay đổi `phase_id`/`phase_version` signoff.
- `src/course_work/data/scaling.py` — Chỉ update upstream signoff reference từ `phase_8_signoff.json` → `phase_5_signoff.json` (mới).
- `src/course_work/data/windows.py` — Tương tự scaling.
- `src/course_work/evaluation/metrics.py`, `experiments/registry.py`, `baselines/persistence.py` — Tương tự.

#### 1.3.3. Notebook (1 file)

- `notebook_course_work/CourseWork.ipynb` (59 cells) — Reorder cells, update markdown headings, add train-only notes.

#### 1.3.4. Architecture rule (1 file)

- `docs/RULE_BASE/architecture_rule.md` — Update §9, §13, §14.3, §7.9.1, §25, transition state.

#### 1.3.5. Logs (4 files)

- `docs/save_log_in_processing/phase_5_eda_log.json` → `phase_6_eda_log.json`
- `docs/save_log_in_processing/phase_6_feature_engineering_log.json` → `phase_7_feature_engineering_log.json`
- `docs/save_log_in_processing/phase_7_feature_set_variants_log.json` → `phase_8_feature_set_variants_log.json`
- `docs/save_log_in_processing/phase_8_chronological_split_log.json` → `phase_5_chronological_split_log.json`

---

## 2. Phân tích vấn đề hiện tại

### 2.1. Bảng Phase hiện tại

| Phase ID | Tên | Owner module | Notebook cells | Sign-off file | Scope hiện tại | Test firewall |
|---|---|---|---|---|---|---|
| 0 | Coursework Contract | `contracts/coursework.py` | (internal only) | `artifacts/contracts/phase_0_signoff.json` | N/A | N/A |
| 1 | Environment | `utils/environment.py` | 4-5 | `artifacts/environment/phase_1_signoff.json` | N/A | N/A |
| 2 | Data Acquisition | `data/acquisition.py` | 6-7 | `artifacts/acquisition/phase_2_signoff.json` | Raw data only | N/A |
| 3 | Schema Audit | `data/schema.py` | 8-9 | `artifacts/schema/phase_3_signoff.json` | Metadata | N/A |
| 4 | Temporal Integrity Audit | `data/temporal.py` | 10-11 | `artifacts/temporal/phase_4_signoff.json` | Lineage, no value mutation | N/A |
| **5** | **Exploratory Data Analysis** | `data/eda.py` + `reporting/eda.py` | **12-41 (30 cells)** | `artifacts/eda/phase_5_signoff.json` | **Full 19,735 rows** | ❌ |
| **6** | **Feature Engineering** | `data/features.py` | **42-43** | `artifacts/features/phase_6_signoff.json` | **Full feature_view** | ❌ |
| **7** | **Feature-Set Variants** | `data/feature_sets.py` | **44-45** | `artifacts/feature_sets/phase_7_signoff.json` | **Full feature_view** | ❌ |
| **8** | **Chronological Split** | `data/splitting.py` | **46-47** | `artifacts/splits/phase_8_signoff.json` | Deterministic 70/15/15 | ✅ |
| 9 | Train-Only Scaling | `data/scaling.py` | 48-49 | `artifacts/scaling/phase_9_signoff.json` | Train-only fit | ✅ |
| 10 | Window Builder | `data/windows.py` | 50-51 | `artifacts/windows/phase_10_signoff.json` | Test target locked | ✅ |
| 11 | DataLoaders | `data/datasets.py` | 52-53 | `artifacts/dataloaders/phase_11_signoff.json` | Test target access modes | ✅ |
| 12 | Shared Metrics | `evaluation/metrics.py` | 54-55 | `artifacts/metrics/phase_12_signoff.json` | Validation selection only | ✅ |
| 13 | Experiment Registry | `experiments/registry.py` | 56-57 | `artifacts/experiments/phase_13_signoff.json` | Reject dev Test | ✅ |
| 14 | Persistence Baseline | `baselines/persistence.py` | 58-59 | `artifacts/baselines/persistence/phase_14_signoff.json` | Validation-only | ✅ |

#### 2.1.1. Sub-sections của Phase 5 EDA hiện tại (30 cells)

```
5.1.  Numerator summary (5.1-5.2)
5.2.  Target quantiles (5.3)
5.3.  Hourly / weekday profile (5.4-5.5)
5.4.  Weekend vs weekday (5.6)
5.5.  Hour×weekday heatmap (5.7)
5.6.  Temperature / humidity distributions (5.8-5.9)
5.7.  Lights distribution (5.10)
5.8.  Correlation matrix (5.11)
5.9.  Selected-lag autocorrelations (5.12)
5.10. Cross-correlation features (5.13)
5.11. Segment-aware rolling statistics (5.14)
5.12. Representative windows (5.15)
5.13. Extreme target samples (5.16)
5.14. Hypothesis registry (5.17 - 10 hypotheses H-EDA-001 → H-EDA-010)
```

### 2.2. Phân tích leakage risk hiện tại

#### 2.2.1. EDA — Distribution statistics

**Vấn đề:** `numeric_summary`, `target_quantiles`, `hourly_profile`, `weekday_profile`, `weekend_profile`, `correlation_outputs`, `autocorrelation_profile` đều đang được tính trên **19,735 rows** = full dataset.

```python
# src/course_work/data/eda.py:382-443
def prepare_eda_analysis(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    phase_4 = materialize_phase_4(root)
    if phase_4.get("status") not in {"PASS", "PASS_WITH_WARNING"}:
        raise RuntimeError("TEMPORAL-v1 does not permit EDA")
    raw_path = root / "data/raw_data/energydata_complete.csv"
    raw_hash_before = sha256_file(raw_path)
    raw = load_raw_csv(raw_path)
    raw_dataframe_before = dataframe_fingerprint(raw)
    temporal_view = load_validated_temporal_view(root)
    eda_view = build_eda_view(temporal_view)  # <- full 19,735 rows
    # ... tính mọi stats trên full eda_view
```

**Risk cụ thể:**

| Statistic | Ảnh hưởng Validation/Test |
|---|---|
| `numeric_summary` mean/std của Appliances | 2,960 Validation rows + 2,961 Test rows đóng góp ~30% vào mean. |
| `target_quantiles` q95 (1,080 Wh) | Spike events ở Validation/Test cuối dataset có thể nâng q95. |
| `correlation_matrix` Pearson | Quantify correlations bị ảnh hưởng bởi all-rows. |
| `autocorrelation_profile` (lag 1, 6, 36, 72, 144, 1008) | Rolling correlation 144h gần cuối dataset có thể "thấy" Test slice. |
| `hourly_profile`, `weekday_profile` | Monday-Sunday phân bố bị ảnh hưởng nếu Validation/Test rơi nhiều vào weekday nhất định. |

#### 2.2.2. EDA — Hypothesis generation

`hypothesis_registry` sinh 10 hypotheses:

```python
# src/course_work/data/eda.py:297-321
rows = [
    ("H-EDA-001", "target_distribution", "Target is right-skewed..."),
    ("H-EDA-002", "target_spikes", "The global 95th percentile..."),
    ("H-EDA-008", "temporal_drift", "First and last 7-day rolling means..."),
    # ...
]
```

**Risk cụ thể:**

- **H-EDA-008 (temporal_drift):** `rolling_first` đến từ đầu dataset (TRAIN), `rolling_last` đến từ cuối dataset (TEST). So sánh này đã leak signal từ Test vào "hypothesis" — có thể khiến downstream quyết định áp dụng RevIN hoặc drift normalization.
- **H-EDA-002 (target_spikes):** `target_row['q95']` tính trên full dataset → ngưỡng spike global có thể bị ảnh hưởng bởi Test slice.
- **H-EDA-006 (daily_pattern):** `hourly['mean']` dùng toàn bộ 19,735 rows để characterize pattern. Downstream phases sử dụng pattern này để motivate TF0/TF1.
- **H-EDA-007 (weekly_pattern):** Weekday vs Weekend comparison dùng full dataset. Weekend pattern ở cuối dataset có thể khác đầu dataset.

#### 2.2.3. Feature Engineering — Safe nhưng có indirect leakage

**Phân tích `build_feature_view`:**

```python
# src/course_work/data/features.py:127-139
def build_feature_view(temporal_view: pd.DataFrame, schema_manifest: dict[str, Any]) -> pd.DataFrame:
    validate_feature_input(temporal_view, schema_manifest)
    ordered_raw_features = schema_manifest["regular_feature_columns"]
    random_controls = schema_manifest["random_control_columns"]
    output = pd.DataFrame({
        "raw_row_index": temporal_view["raw_row_index"].to_numpy(copy=True),
        "date": temporal_view["date"].astype(str).to_numpy(copy=True),
        "timestamp": temporal_view["timestamp_parsed"].to_numpy(copy=True),
        "continuity_segment_id": temporal_view["continuity_segment_id"].to_numpy(copy=True),
    })
    for column in [TARGET_COLUMN, *ordered_raw_features, *random_controls]:
        output[column] = temporal_view[column].to_numpy(copy=True)
    return add_time_features(output)
```

Calendar features (`hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `weekend`) đều là **deterministic** từ timestamp — không có leakage về giá trị feature.

**NHƯNG** vẫn có indirect leakage risk:

1. **Invariant validation scope:** `validate_feature_invariants` kiểm tra `row_count_preserved`, `target_preserved`, `raw_features_preserved` trên full dataset. Leakage không xảy ra ở đây.

2. **Feature selection reasoning:** Trong Phase 6 description (manifest note "Phase 7 owns final ordered feature-set membership"), không có explicit feature selection. Tuy nhiên, các downstream phase (FS0/FS1/FS2 selection trong Phase 7/13) sẽ đọc EDA hypothesis notes. Nếu EDA hypothesis nói "test spike threshold based on q95 = 1080 Wh", thì decision trong Phase 37 / 50 có thể bị ảnh hưởng.

3. **Documentation/docs/plan-doc/phase_6_feature_engineering_implementation_plan.md:** Hiện không thấy nhưng cần check có dùng hypothesis làm input cho downstream decision không.

#### 2.2.4. Feature-Set Variants — Validation scope mismatch

```python
# src/course_work/data/feature_sets.py:151-274
def validate_feature_variants(
    feature_view: pd.DataFrame,
    registries: dict[str, pd.DataFrame],
    variants: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    # ...
    for variant_id, features in variants.items():
        # ...
        missing_value_count = 0 if missing else int(feature_view[list(features)].isna().sum().sum())
        all_finite = False if missing else bool(np.isfinite(feature_view[list(features)].to_numpy(dtype=float)).all())
```

`missing_value_count` và `all_finite` hiện đang validate trên **full feature_view (29,735 rows)** — không khớp với actual training scope (TRAIN).

**Risk:** Nếu Validation/Test có missing values hoặc giá trị infinite mà TRAIN không có, `all_finite = True` ở scope full nhưng có thể false ở Train-only scope. Ngược lại, ensure-pass hiện tại trên full có thể "che giấu" issue ở Train-only.

#### 2.2.5. Splitting — Logically OK nhưng scope pattern đảo lộn

`splitting.py` hiện đang đúng về mặt logic:
- `build_chronological_membership` chỉ dựa trên `row index → timestamp` (deterministic).
- KHÔNG sử dụng target values, feature values, hay distribution statistics.
- `verify_phase_8_inputs` chỉ check upstream sign-off, không check value.

**Vấn đề duy nhất:** Vị trí Phase 8 trong pipeline là **sai** về mặt kiến trúc — phải là prerequisite cho EDA/FE/FS để:
1. EDA chỉ nhìn TRAIN.
2. FE chỉ validate invariants trên TRAIN.
3. FS validation chỉ apply trên TRAIN.

### 2.3. Điểm mạnh cần giữ

#### 2.3.1. EDA boundary §15 vẫn giữ nguyên nội dung

Phase 5 EDA vẫn được phép:
- Mô tả target và features.
- Tạo EDA-only temporal columns trong df_eda.
- Visualize temporal pattern.
- Tạo descriptive correlation.
- Tạo hypothesis cho Phase sau.
- Direct descriptive EDA trong notebook (Human-approved).

Phase 5 KHÔNG được:
- Chọn final feature set.
- Chọn final lookback.
- Chọn final loss.
- Bật RevIN mặc định.
- Xóa spike.
- Thay outlier bằng NaN.
- Interpolate sensor hoặc target.
- Tạo final regime từ full dataset.
- Tune bằng Test.
- Kết luận quan hệ nhân quả từ correlation.

**Train-only scope chỉ ADD constraint lên EXISTING 5.1-5.14 outputs, không thay đổi sematic EDA.**

#### 2.3.2. Determinism của calendar features

`add_time_features` (features.py:107-124) sinh `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `weekend` từ `timestamp` — **deterministic, không có leakage**. Việc tính trên 19,735 rows hay 13,814 TRAIN rows đều cho cùng giá trị cho từng row.

#### 2.3.3. Test firewall đã đúng ở Phase 9-14

Các phase sau split đều có:
- Phase 9: `fit_x_scaler_bundle` filter `split_values != {"TRAIN"}`.
- Phase 10: `test_target_access_policy` locked.
- Phase 11: 4 target access modes.
- Phase 12: Test metrics gated.
- Phase 13: Reject dev Test targets.
- Phase 14: Validation-only evaluation.

**Refactor này chỉ đưa Test firewall lên sớm hơn, không phải thêm logic firewall mới.**

#### 2.3.4. 6 feature set variants giữ nguyên

VARIANT_COMPONENTS, EXPECTED_FEATURE_COUNTS, build_feature_variants, build_registry_rows, build_variant_rows, build_lineage_rows — TẤT CẢ đều không phụ thuộc split. Refactor không cần thay đổi bất kỳ cấu trúc variant nào.

#### 2.3.5. Phase 14 Persistence baseline đã đúng

`baselines/persistence.py` đã có:
- `test_target_values_and_test_metrics_inaccessible: True`
- `test_access_authorized: True` chỉ trong `final_test_evaluation` mode.
- Validation-only evaluation cho Phase 14 final.

Refactor không cần thay đổi phase 14 logic.

#### 2.3.6. Phase 9-14 contract checksums đã verified

Phase 9-14 đều có `verify_existing_signoff` check `status="PASS"` + `output_checksums`. Sau khi refactor, các phase này sẽ verify signoff mới (`phase_5_signoff.json` thay vì `phase_8_signoff.json`).

---

## 3. Đề xuất refactor

### 3.1. Bảng mapping Phase cũ → Phase mới

| Phase cũ | Tên cũ | Module sở hữu | Phase mới | Tên mới | Thay đổi |
|---|---|---|---|---|---|
| 5 | Exploratory Data Analysis | `data/eda.py` | 6 | Exploratory Data Analysis (Train-Only) | Phase renumber, add train filter |
| 6 | Feature Engineering | `data/features.py` | 7 | Feature Engineering (Train-Only) | Phase renumber, add train filter |
| 7 | Feature-Set Variants | `data/feature_sets.py` | 8 | Feature-Set Variants (Train-Only) | Phase renumber, add train filter |
| 8 | Chronological Split | `data/splitting.py` | 5 | Chronological Split | Phase renumber, log update |
| 9 | Train-Only Scaling | `data/scaling.py` | 9 | Train-Only Scaling | (Không đổi số) Update signoff ref |
| 10 | Window Builder | `data/windows.py` | 10 | Window Builder | (Không đổi số) Update signoff ref |
| 11 | DataLoaders | `data/datasets.py` | 11 | DataLoaders | (Không đổi số) |
| 12 | Shared Metrics | `evaluation/metrics.py` | 12 | Shared Metrics | (Không đổi số) |
| 13 | Experiment Registry | `experiments/registry.py` | 13 | Experiment Registry | (Không đổi số) |
| 14 | Persistence Baseline | `baselines/persistence.py` | 14 | Persistence Baseline | (Không đổi số) |

### 3.2. Lý do Phase 5 = Split

#### 3.2.1. Split chỉ dựa trên temporal integrity

`build_chronological_membership` thực hiện:
1. Validate split input schema (raw_row_index, timestamp, continuity_segment_id).
2. Compute floor-based boundaries theo train_ratio, validation_ratio, test_ratio.
3. Assign split_id theo row position.

```python
# src/course_work/data/splitting.py:89-105
def build_chronological_membership(
    dataframe: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    validation_ratio: float = VALIDATION_RATIO,
    test_ratio: float = TEST_RATIO,
) -> pd.DataFrame:
    validate_split_input(dataframe)
    boundaries = compute_split_boundaries(len(dataframe), train_ratio, validation_ratio, test_ratio)
    output = dataframe[["raw_row_index", "timestamp", "continuity_segment_id"]].copy(deep=True).reset_index(drop=True)
    output["split_id"] = ""
    output["split_position"] = -1
    for split_id in SPLIT_IDS:
        start, end = boundaries[split_id]
        output.loc[start : end - 1, "split_id"] = split_id
        output.loc[start : end - 1, "split_position"] = np.arange(end - start, dtype=np.int64)
    return output
```

**KHÔNG** sử dụng:
- Target values (Appliances).
- Feature values (lights, T1, etc.).
- Distribution statistics.
- Any computed feature.

**Chỉ** dựa trên:
- Row index position (deterministic).
- Timestamp (temporal integrity đã verify ở Phase 4).
- `compute_split_boundaries` (deterministic floor-based).

#### 3.2.2. Split KHÔNG phụ thuộc giá trị target/features

`build_chronological_membership` không có branch nào sử dụng:
- `frame[TARGET_COLUMN]` (Appliances).
- `frame["lights"]`, `frame["T1"]`, etc.
- `frame.quantile()`, `frame.mean()`, `frame.std()`.

→ Hoàn toàn deterministic, không có leakage risk.

#### 3.2.3. Cho phép EDA chỉ thấy Train → loại bỏ look-ahead bias

Khi EDA chỉ thấy 13,814 TRAIN rows (thay vì 19,735 full):
- H-EDA-008 (temporal_drift): `rolling_first` và `rolling_last` đều từ TRAIN → drift diagnosed strictly within training period.
- H-EDA-002 (target_spikes): `q95` tính trên TRAIN → spike threshold là training-derived.
- H-EDA-006 (daily_pattern): `hourly_profile` không bị ảnh hưởng bởi weekday distribution cuối dataset.
- H-EDA-007 (weekly_pattern): Weekend comparison không thiên về weekday type.

→ Eliminates hypothesis stage leakage.

### 3.3. Lý do EDA/FE/FS phải Train-only

#### 3.3.1. Distribution analysis chỉ nên trên Train

Mọi predictive model sẽ được TRAIN trên Train data. Khi đó:
- Mean, std, quantiles của target trên full dataset ≠ giá trị thực tế model sẽ thấy.
- H-EDA-001 nói "right-skewed" dựa trên full dataset; nếu Train-only có skewness khác, hypothesis note sẽ misleading.

→ Train-only statistics align với actual training scope.

#### 3.3.2. Feature engineering outputs giống nhau vì deterministic

Như đã phân tích ở §2.2.3, `add_time_features` deterministic → row-level values identical giữa full vs Train-only computation. **Tuy nhiên** `validate_feature_invariants` (e.g., `no_new_nulls`, `time_feature_ranges_valid`) chỉ nên assert trên TRAIN rows where model sẽ fit.

→ Align invariant scope với training scope.

#### 3.3.3. Feature-set variants validation phải trên Train

`validate_feature_variants` check `missing_value_count` và `all_finite` trên full feature_view. Nếu sau này Training gặp missing/infinite ở Trainer rows, model sẽ fail trong khi Phase 7 sign-off đã PASS.

→ Train-only validation đảm bảo variants align với actual training data.

### 3.4. Architecture amendment cần thiết

#### 3.4.1. Phase-to-module mapping đảo vị trí

| Phase mới | Owner chính | Thay đổi so với cũ |
|---|---|---|
| 5 | `data/splitting.py` | (cũ Phase 8) |
| 6 | `data/eda.py` + `reporting/eda.py` | (cũ Phase 5) + thêm train filter |
| 7 | `data/features.py` | (cũ Phase 6) + thêm train filter |
| 8 | `data/feature_sets.py` | (cũ Phase 7) + thêm train filter |

#### 3.4.2. Notebook boundary §14.3 thay đổi cell order

Cũ:
```
Phase 5 → Phase 6 → Phase 7 → Phase 8
```

Mới:
```
Phase 5 (Chronological Split) → Phase 6 (EDA Train-Only) → Phase 7 (FE Train-Only) → Phase 8 (FS Train-Only)
```

#### 3.4.3. Transition state §25 update

Mỗi phase 5/6/7/8 mới cần một entry mô tả trong §25:
- Phase 5 mới: "Phase 5 = Chronological Split, structural only, deterministic 70/15/15 row membership."
- Phase 6 mới: "Phase 6 = EDA, train-only scope, hypotheses strictly train-derived."
- Phase 7 mới: "Phase 7 = Feature Engineering, train-only invariant scope."
- Phase 8 mới: "Phase 8 = Feature-Set Variants, train-only validation scope."

#### 3.4.4. Article §6.6 EDA boundary mở rộng

EDA boundary cho Phase 6 mới (cũ 5) cần thêm ngôn ngữ:
```
Direct EDA Train-only: Phân tích chỉ trên TRAIN rows đã được phân chia ở Phase 5.
EDA summary tables và figures phải dựa trên TRAIN-derived statistics.
Hypothesis chỉ motivate từ TRAIN patterns.
```

#### 3.4.5. Change-control gate §24

Theo §24, "Đổi Phase-to-module mapping" và "Đổi canonical notebook" CẦN architecture amendment + Human approval trước khi thực thi. **Plan này phải được Human approve trước khi bất kỳ source modification nào.**

---

## 4. Plan thay đổi chi tiết

### 4.1. Module Python changes

#### 4.1.1. `src/course_work/data/eda.py` (Train-Only EDA)

**Mục đích:** Filter EDA view theo TRAIN rows sau Phase 5 (Split) mới.

**Thay đổi chính:**

**a. Modify `prepare_eda_analysis(root)`:**

```python
# src/course_work/data/eda.py:382-443
def prepare_eda_analysis(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    phase_4 = materialize_phase_4(root)
    if phase_4.get("status") not in {"PASS", "PASS_WITH_WARNING"}:
        raise RuntimeError("TEMPORAL-v1 does not permit EDA")
    # NEW: Phase 5 (Split) must be PASS for EDA Train-only
    phase_5 = materialize_phase_5(root)  # NOTE: function name may stay as-is per §3.4.1
    if phase_5.get("artifact_version") != SPLIT_VERSION or phase_5.get("status") != "PASS":
        raise RuntimeError("SPLIT-v1 does not permit EDA")
    raw_path = root / "data/raw_data/energydata_complete.csv"
    raw_hash_before = sha256_file(raw_path)
    raw = load_raw_csv(raw_path)
    raw_dataframe_before = dataframe_fingerprint(raw)
    temporal_view = load_validated_temporal_view(root)
    
    # NEW: load split membership and filter TRAIN rows
    split_membership = load_validated_split_membership(root)
    train_indices = split_membership.index[split_membership["split_id"].eq("TRAIN")]
    eda_view = build_eda_view(temporal_view.loc[train_indices])  # NEW: train-only
    
    # ... rest of the function uses eda_view (now train-only)
    
    # NEW: audit train-only scope
    train_row_count_used = len(eda_view)
    train_only_eda_validated = (train_row_count_used == int(phase_5["train_rows"]))
    if not train_only_eda_validated:
        raise RuntimeError(f"Train-only EDA scope mismatch: {train_row_count_used} vs {phase_5['train_rows']}")
    
    analysis["train_only_eda_validated"] = True
    analysis["train_row_count_used"] = train_row_count_used
    analysis["train_only_target_fingerprint"] = hashlib.sha256(...).hexdigest()
    # ... rest unchanged
```

**b. `compare_temporal_splits` (existing helper, cập nhật name):**

```python
# Existing unused helper - keep for documentation/debugging
def compare_temporal_splits(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Returns 3-split comparison for documentation ONLY
    # Does NOT affect Phase 6 EDA scope
```

**c. Modify `reporting/eda.py::materialize_phase_5` (function name giữ, nhưng Phase 5 giờ là EDA, không phải Split):**

Lưu ý: `reporting/eda.py::materialize_phase_5` hiện đang ghi `phase_5_signoff.json` cho EDA. Sau refactor:
- `reporting/eda.py::materialize_phase_5` → đổi tên thành `materialize_phase_6` (vì Phase 6 mới = EDA).
- Signoff file: `phase_5_signoff.json` → `phase_6_signoff.json`.
- Manifest thêm fields:
  - `train_only_eda_scope: True`
  - `train_rows: int`
  - `train_only_eda_validated: True`
  - `phase_id: 6` (thay vì 5)
  - `phase_version: "PHASE-6-v1"` (thay vì "PHASE-5-v1")

**d. Import update:**

```python
# Trong reporting/eda.py
from course_work.data.splitting import load_validated_split_membership, materialize_phase_5 as materialize_phase_5_split
# Rename to avoid collision:
from course_work.data.splitting import (
    load_validated_split_membership,
    materialize_phase_5 as materialize_phase_5_chronological_split,
)
```

#### 4.1.2. `src/course_work/data/features.py` (Train-Only Feature Engineering)

**Mục đích:** Filter FE validation/invariant scope theo TRAIN rows.

**Thay đổi chính:**

**a. Modify `verify_phase_6_inputs`:**

```python
# src/course_work/data/features.py:51-83
def verify_phase_6_inputs(root: Path) -> dict[str, dict[str, Any]]:
    signoffs = {
        relative_path: verify_signed_artifact(root, relative_path, expected_version, allowed_statuses)
        for relative_path, (expected_version, allowed_statuses) in UPSTREAM_SIGNOFFS.items()
    }
    # ... existing schema_manifest, temporal_manifest checks ...
    
    # REPLACE: eda_manifest check với split_manifest check
    split_manifest = read_json(root / "artifacts/splits/split_manifest.json")
    split_signoff = read_json(root / "artifacts/splits/phase_5_signoff.json")  # NEW: phase 5 split
    if split_signoff.get("artifact_version") != SPLIT_VERSION or split_signoff.get("status") != "PASS":
        raise RuntimeError("SPLIT-v1 is not signed off")
    # ... preserve existing upstream checks ...
```

**b. Modify `UPSTREAM_SIGNOFFS`:**

```python
# src/course_work/data/features.py:27-34
UPSTREAM_SIGNOFFS = {
    "artifacts/contracts/phase_0_signoff.json": ("COURSEWORK-CONTRACT-v1", {"PASS"}),
    "artifacts/environment/phase_1_signoff.json": ("ENV-v1", {"PASS"}),
    "artifacts/acquisition/phase_2_signoff.json": ("DATA-v1", {"PASS"}),
    "artifacts/schema/phase_3_signoff.json": ("SCHEMA-v1", {"PASS", "PASS_WITH_WARNING"}),
    "artifacts/temporal/phase_4_signoff.json": ("TEMPORAL-v1", {"PASS", "PASS_WITH_WARNING"}),
    # REPLACE: "artifacts/eda/phase_5_signoff.json" → "artifacts/splits/phase_5_signoff.json"
    "artifacts/splits/phase_5_signoff.json": ("SPLIT-v1", {"PASS"}),  # NEW: Phase 5 = Split
}
```

**c. Modify `build_feature_view`:**

```python
# src/course_work/data/features.py:127-139
def build_feature_view(
    temporal_view: pd.DataFrame,
    schema_manifest: dict[str, Any],
    membership: pd.DataFrame | None = None,  # NEW: optional filter
) -> pd.DataFrame:
    validate_feature_input(temporal_view, schema_manifest)
    if membership is not None:
        train_indices = membership.index[membership["split_id"].eq("TRAIN")]
        if len(train_indices) != len(membership):
            raise ValueError("build_feature_view with membership must include only TRAIN rows")
        temporal_view = temporal_view.loc[train_indices].copy(deep=True)
    # ... rest unchanged ...
```

**Lưu ý:** `build_feature_view` sinh CSV derived `energydata_feature_engineered_v1.csv` cho Phase 7 validation. **Hai lựa chọn:**

1. **Derived CSV = full 19,735 rows** (giữ nguyên hiện tại), chỉ validate invariants trên TRAIN subset.
2. **Derived CSV = TRAIN 13,814 rows** (giảm scope).

**Recommend:** Giữ full derived CSV (vì FEATURES-v1 hợp đồng về full timeline), nhưng thêm audit field `feature_engineering_train_only_validated: True` vào manifest. Validate invariants trên TRAIN subset.

**d. Modify `materialize_phase_6`:**

```python
# src/course_work/data/features.py:305-425
def materialize_phase_6(project_root: Path | None = None) -> dict[str, Any]:
    # ... existing code ...
    upstream = verify_phase_6_inputs(root)
    # ... load temporal_view, schema_manifest ...
    
    # NEW: Load split membership
    split_membership = load_validated_split_membership(root)
    
    # Build feature_view (full timeline as before)
    feature_view = build_feature_view(temporal_view, schema_manifest)
    
    # NEW: Compute train-only feature_view for validation
    feature_view_train = build_feature_view(temporal_view, schema_manifest, membership=split_membership)
    
    # NEW: Validate invariants on TRAIN subset
    audit_rows = validate_feature_invariants_train_only(temporal_view, feature_view_train, schema_manifest)
    
    # ... existing lineage, registry, availability, leakage audits (full scope) ...
    
    manifest = {
        # ... existing fields ...
        "train_only_feature_engineering": True,  # NEW
        "train_rows": len(feature_view_train),   # NEW
        "feature_engineering_train_only_validated": int(audit_rows["status"] == "PASS"),  # NEW
        # ...
    }
    
    signoff = {
        # ... existing fields ...
        "phase_id": 7,  # REPLACE: 6 → 7
        "phase_version": "PHASE-7-v1",  # REPLACE: "PHASE-6-v1" → "PHASE-7-v1"
        # ...
    }
```

**e. Update `verify_existing_signoff`:**

```python
# src/course_work/data/features.py:289-302
def verify_existing_signoff(root: Path, signoff_path: Path) -> dict[str, Any]:
    signoff = read_json(signoff_path)
    if signoff.get("status") != "PASS" or signoff.get("artifact_version") != FEATURE_VERSION:
        raise RuntimeError("Existing Phase 7 sign-off is invalid")  # REPLACE: "Phase 6" → "Phase 7"
    # ... rest unchanged ...
```

**f. `signoff_path` update:**

```python
# src/course_work/data/features.py:307
signoff_path = root / FEATURE_ARTIFACT_ROOT / "phase_7_signoff.json"  # REPLACE: phase_6 → phase_7
```

#### 4.1.3. `src/course_work/data/feature_sets.py` (Train-Only Feature-Set Variants)

**Mục đích:** Filter FS validation scope theo TRAIN rows.

**Thay đổi chính:**

**a. Modify `verify_phase_7_inputs`:**

```python
# src/course_work/data/feature_sets.py:106-128
def verify_phase_7_inputs(root: Path) -> dict[str, Any]:
    phase_7 = materialize_phase_7(root)  # NOTE: function name kept as-is
    # NEW: Check that Phase 7 (FE-train-only) is signed off
    # NOTE: phase_7_signoff.json refers to FEATURES-v1
    if phase_7.get("artifact_version") != FEATURE_VERSION or phase_7.get("status") != "PASS":
        raise RuntimeError("FEATURES-v1 is not signed off")
    # ... rest unchanged ...
```

**NOTE:** Function name `verify_phase_7_inputs` đang ambiguous. Sau refactor:
- Phase 7 = Feature Engineering (new).
- `verify_phase_7_inputs` vẫn đặt tên theo Phase 7 (FE) sign-off.

**b. Modify `validate_feature_variants`:**

```python
# src/course_work/data/feature_sets.py:151-274
def validate_feature_variants(
    feature_view: pd.DataFrame,
    registries: dict[str, pd.DataFrame],
    variants: dict[str, tuple[str, ...]],
    membership: pd.DataFrame | None = None,  # NEW: optional TRAIN filter
) -> dict[str, Any]:
    # NEW: Filter to TRAIN if membership provided
    if membership is not None:
        train_indices = membership.index[membership["split_id"].eq("TRAIN")]
        feature_view = feature_view.loc[train_indices].copy(deep=True)
    
    # ... rest of validation logic (unchanged) ...
```

**c. Modify `materialize_phase_7`:**

```python
# src/course_work/data/feature_sets.py:429-562
def materialize_phase_7(project_root: Path | None = None) -> dict[str, Any]:
    # ... existing code ...
    phase_7 = verify_phase_7_inputs(root)
    feature_view = load_validated_feature_view(root)  # NEW: full feature_view
    split_membership = load_validated_split_membership(root)  # NEW: load membership
    
    # NEW: Validate variants on TRAIN subset only
    audit = validate_feature_variants(feature_view, registries, variants, membership=split_membership)
    
    # ... rest unchanged ...
    
    manifest = {
        # ... existing fields ...
        "train_only_feature_set_validation": True,  # NEW
        "train_rows": int(split_membership["split_id"].eq("TRAIN").sum()),  # NEW
        # ...
    }
    
    signoff = {
        # ... existing fields ...
        "phase_id": 8,  # REPLACE: 7 → 8
        "phase_version": "PHASE-8-v1",  # REPLACE: "PHASE-7-v1" → "PHASE-8-v1"
        # ...
    }
```

**d. Update `signoff_path` và `verify_existing_signoff`:**

```python
# src/course_work/data/feature_sets.py:431
signoff_path = root / FEATURE_SET_ARTIFACT_ROOT / "phase_8_signoff.json"  # REPLACE: phase_7 → phase_8

# src/course_work/data/feature_sets.py:414-426
def verify_existing_signoff(root: Path, signoff_path: Path) -> dict[str, Any]:
    signoff = read_json(signoff_path)
    if signoff.get("status") != "PASS" or signoff.get("artifact_version") != FEATURE_SET_VERSION:
        raise RuntimeError("Existing Phase 8 sign-off is invalid")  # REPLACE: "Phase 7" → "Phase 8"
    # ... rest unchanged ...
```

#### 4.1.4. `src/course_work/data/splitting.py` (Phase 5 mới = Chronological Split)

**Mục đích:** KHÔNG thay đổi logic splitting. Chỉ thay đổi signoff path và phase_id/phase_version.

**Thay đổi:**

**a. Function name `materialize_phase_8` KHÔNG rename:**

```python
# src/course_work/data/splitting.py:342 (existing)
def materialize_phase_8(project_root: Path | None = None) -> dict[str, Any]:
    # Function name kept as-is to avoid breaking ownership references
    # ...
```

**Lý do giữ tên:**
- Function name là API identifier, không phải architecture rule mapping.
- Rất nhiều module (scaling.py, windows.py, notebook) gọi `materialize_phase_8(...)`.
- Rename sẽ tạo nhiều cross-file changes không cần thiết.
- Architecture rule §9 mapping nói "Phase 8 = data/splitting.py" — function name vs phase number không cần synchronize.

**b. Update `signoff_path`:**

```python
# src/course_work/data/splitting.py:344
signoff_path = root / SPLIT_ARTIFACT_ROOT / "phase_5_signoff.json"  # REPLACE: phase_8 → phase_5
```

**c. Update `verify_existing_signoff`:**

```python
# src/course_work/data/splitting.py:325-339
def verify_existing_signoff(root: Path, signoff_path: Path) -> dict[str, Any]:
    signoff = read_json(signoff_path)
    if signoff.get("status") != "PASS" or signoff.get("artifact_version") != SPLIT_VERSION:
        raise RuntimeError("Existing Phase 5 sign-off is invalid")  # REPLACE: "Phase 8" → "Phase 5"
    # ... rest unchanged ...
```

**d. Update `signoff`:**

```python
# src/course_work/data/splitting.py:456-484
signoff = {
    "artifact_version": SPLIT_VERSION,
    "phase_id": 5,  # REPLACE: 8 → 5
    "phase_version": "PHASE-5-v1",  # REPLACE: "PHASE-8-v1" → "PHASE-5-v1"
    # ... rest unchanged ...
}
```

**e. Update `input_paths`:**

```python
# src/course_work/data/splitting.py:446-455
input_paths = [
    # REPLACE: phase_6 (FE) → phase_7 (FE mới), phase_7 (FS) → phase_8 (FS mới)
    "artifacts/feature_sets/phase_8_signoff.json",  # NEW: phase 8 = FS mới
    "artifacts/feature_sets/feature_set_registry.json",
    # REMOVE: artifacts/features/phase_6_signoff.json (FE), artifacts/features/feature_engineering_manifest.json
    # Phase 5 (Split) dependencies are now DATA, SCHEMA, TEMPORAL only
    "artifacts/temporal/phase_4_signoff.json",
    "artifacts/temporal/temporal_manifest.json",
    "configs/base/coursework_contract.json",
]
```

**Lưu ý:** Sau refactor, Phase 5 (Split) nhận input từ Phase 4 (Temporal) + Phase 0 (Contract) + Phase 1-3 (Environment/Acquisition/Schema). KHÔNG còn phụ thuộc FEATURES-v1.

**f. Update audit messages:**

```python
# src/course_work/data/splitting.py:218-247
def build_split_audit(...):
    # Existing checks for chronology, coverage, disjointness, Test firewall
    # Add: train_only_eda_split_basis: "by_target_timestamp"
    # ...
```

**g. Update `loading function name`:**

```python
# src/course_work/data/splitting.py:489-503
def load_validated_split_membership(project_root: Path | None = None) -> pd.DataFrame:
    # Function name kept as-is
    root = (project_root or get_project_root()).resolve()
    materialize_phase_8(root)  # Function name kept
    # ... rest unchanged ...
```

#### 4.1.5. `src/course_work/data/scaling.py` (Update upstream signoff reference)

**Mục đích:** KHÔNG thay đổi scaling logic. Chỉ update upstream signoff reference từ `phase_8_signoff.json` → `phase_5_signoff.json` (mới).

**Thay đổi:**

```python
# src/course_work/data/scaling.py:520-537
def verify_phase_9_inputs(root: Path) -> dict[str, Any]:
    phase_9 = materialize_phase_9(root)  # Function name kept
    # REPLACE: phase_8 references → phase_5 references
    phase_5 = read_json(root / "artifacts/splits/phase_5_signoff.json")  # NEW
    if phase_5.get("artifact_version") != SPLIT_VERSION or phase_5.get("status") != "PASS":
        raise RuntimeError("SPLIT-v1 is not signed off")
    # ... rest unchanged ...
```

```python
# src/course_work/data/scaling.py:792-803
input_paths = [
    # REPLACE: phase_6, phase_7, phase_8 → phase_7, phase_8, phase_5
    "artifacts/features/phase_7_signoff.json",  # NEW: Phase 7 = FE mới
    "artifacts/features/feature_engineering_manifest.json",
    "artifacts/feature_sets/phase_8_signoff.json",  # NEW: Phase 8 = FS mới
    "artifacts/feature_sets/feature_set_registry.json",
    "artifacts/splits/phase_5_signoff.json",  # NEW: Phase 5 = Split mới
    "artifacts/splits/split_manifest.json",
    "artifacts/splits/split_membership.csv",
    "artifacts/environment/environment_report.json",
    "configs/base/coursework_contract.json",
]
```

#### 4.1.6. `src/course_work/data/windows.py` (Update upstream signoff references)

**Thay đổi:**

```python
# src/course_work/data/windows.py:622-624
required_signoffs = {
    "artifacts/temporal/phase_4_signoff.json": ("TEMPORAL-v1", {"PASS", "PASS_WITH_WARNING"}),
    "artifacts/features/phase_7_signoff.json": (FEATURE_VERSION, {"PASS"}),  # NEW: Phase 7 = FE mới
    "artifacts/feature_sets/phase_8_signoff.json": (FEATURE_SET_VERSION, {"PASS"}),  # NEW: Phase 8 = FS mới
    "artifacts/splits/phase_5_signoff.json": (SPLIT_VERSION, {"PASS"}),  # NEW: Phase 5 = Split mới
}
```

```python
# src/course_work/data/windows.py:829-834
input_paths = [
    "artifacts/features/phase_7_signoff.json",  # NEW
    "artifacts/features/feature_engineering_manifest.json",
    "artifacts/feature_sets/phase_8_signoff.json",  # NEW
    "artifacts/feature_sets/feature_set_registry.json",
    "artifacts/splits/phase_5_signoff.json",  # NEW
    "artifacts/splits/split_manifest.json",
    "artifacts/splits/split_membership.csv",
    # ... rest unchanged ...
]
```

#### 4.1.7. `src/course_work/evaluation/metrics.py`, `experiments/registry.py`, `baselines/persistence.py`

**Mục đích:** Update upstream signoff references cho phase 5/6/7/8 nếu cần.

**Hiện tại (theo grep):**

- `metrics.py`: Không thấy reference trực tiếp `phase_5_signoff` / `phase_6_signoff` / `phase_7_signoff` / `phase_8_signoff` (verify qua `verify_existing_signoff` chain).
- `registry.py`: Tương tự.
- `persistence.py`: Tương tự.

**Refactor:** KHÔNG cần thay đổi logic. Chỉ cần ensure rằng upstream signoff references vẫn dùng đúng path:
- `phase_5_signoff.json` giờ là SPLIT-v1 (was phase_8_signoff.json).
- `phase_7_signoff.json` giờ là FEATURES-v1 (was phase_6_signoff.json).
- `phase_8_signoff.json` giờ là FEATURESETS-v1 (was phase_7_signoff.json).

Mọi chain từ Phase 9-14 theo `verify_existing_signoff` chain → KHÔNG cần hard-code paths.

**Lưu ý:** Cần check kỹ `experiments/registry.py` và `baselines/persistence.py` để confirm không có hard-coded `phase_8_signoff` reference.

#### 4.1.8. `src/course_work/reporting/phase_summary.py` (Presentation layer update)

**Mục đích:** Update PHASE_NAMES, LOG_FILENAMES, SOURCE_SPECS, PRESENTATION_SPECS, `_phase_content` keys 5/6/7/8.

**Thay đổi:**

**a. Update `PHASE_NAMES`:**

```python
# src/course_work/reporting/phase_summary.py:18-34
PHASE_NAMES = {
    0: "Coursework Contract",
    1: "Environment",
    2: "Data Acquisition",
    3: "Schema Audit",
    4: "Temporal Integrity Audit",
    5: "Chronological Split",  # REPLACE: "Exploratory Data Analysis" → "Chronological Split"
    6: "Exploratory Data Analysis (Train-Only)",  # REPLACE: "Feature Engineering" → "Exploratory Data Analysis (Train-Only)"
    7: "Feature Engineering (Train-Only)",  # REPLACE: "Feature-Set Variants" → "Feature Engineering (Train-Only)"
    8: "Feature-Set Variants (Train-Only)",  # REPLACE: "Chronological Split" → "Feature-Set Variants (Train-Only)"
    9: "Train-Only Scaling",
    # ... rest unchanged ...
}
```

**b. Update `LOG_FILENAMES`:**

```python
# src/course_work/reporting/phase_summary.py:35-51
LOG_FILENAMES = {
    0: "phase_0_coursework_contract_log.json",
    1: "phase_1_environment_log.json",
    2: "phase_2_data_acquisition_log.json",
    3: "phase_3_schema_audit_log.json",
    4: "phase_4_temporal_integrity_log.json",
    5: "phase_5_chronological_split_log.json",  # REPLACE: "phase_5_eda_log.json" → "phase_5_chronological_split_log.json"
    6: "phase_6_eda_log.json",  # REPLACE: "phase_6_feature_engineering_log.json" → "phase_6_eda_log.json"
    7: "phase_7_feature_engineering_log.json",  # REPLACE: "phase_7_feature_set_variants_log.json" → "phase_7_feature_engineering_log.json"
    8: "phase_8_feature_set_variants_log.json",  # REPLACE: "phase_8_chronological_split_log.json" → "phase_8_feature_set_variants_log.json"
    9: "phase_9_train_only_scaling_log.json",
    # ... rest unchanged ...
}
```

**c. Update `SOURCE_SPECS`:**

```python
# src/course_work/reporting/phase_summary.py:52-122
SOURCE_SPECS = {
    # ... 0-4 unchanged ...
    5: (
        ("splits", "artifacts/splits/split_manifest.json"),
        ("signoff", "artifacts/splits/phase_5_signoff.json"),  # NEW: Phase 5 = Split
    ),
    6: (  # REPLACE: 5
        ("eda", "artifacts/eda/eda_manifest.json"),
        ("anomalies", "artifacts/eda/eda_anomalies.json"),
        ("signoff", "artifacts/eda/phase_6_signoff.json"),  # NEW: phase 6 = EDA
    ),
    7: (  # REPLACE: 6
        ("features", "artifacts/features/feature_engineering_manifest.json"),
        ("signoff", "artifacts/features/phase_7_signoff.json"),  # NEW: phase 7 = FE
    ),
    8: (  # REPLACE: 7
        ("feature_sets", "artifacts/feature_sets/feature_set_manifest.json"),
        ("signoff", "artifacts/feature_sets/phase_8_signoff.json"),  # NEW: phase 8 = FS
    ),
    # ... 9-14 unchanged ...
    # REMOVE: old entry 8 for splits (moved to 5)
}
```

**d. Update `PRESENTATION_SPECS`:**

```python
# src/course_work/reporting/phase_summary.py:123-240
PRESENTATION_SPECS = {
    # ... 0-4 unchanged ...
    5: {  # REPLACE: old 5 (EDA)
        "summary_title": None,
        "summary_fields": (),
        "sections": ({"title": "Chronological membership"},),
        "split_allocation": True,
    },
    6: {  # REPLACE: old 6 (FE)
        "summary_title": None,
        "summary_fields": (),
        "sections": (),  # EDA: no summary in core phase_summary, output via sub-sections 5.1-5.14
    },
    7: {  # REPLACE: old 7 (FS)
        "summary_title": "Feature overview",
        "summary_fields": ("Rows", "Columns", "Raw features", "Engineered features", "Metadata columns"),
        "sections": ({"title": "Engineered feature registry"},),
    },
    8: {  # REPLACE: old 8 (Split)
        "summary_title": None,
        "summary_fields": (),
        "sections": (
            {
                "title": "Feature-set registry",
                "columns": ("Variant", "Feature count", "Baseline"),
            },
        ),
    },
    # ... 9-14 unchanged ...
}
```

**e. Update `_phase_content()` branches:**

```python
# src/course_work/reporting/phase_summary.py:395-471
# REPLACE: phase_id == 5 → split content, phase_id == 6 → EDA, phase_id == 7 → FE, phase_id == 8 → FS

# Phase 5 (mới = Split) - was at the bottom of _phase_content
if phase_id == 5:
    splits = sources["splits"]
    summary = {
        "Audit": splits["audit_status"],
        "Version": splits["split_version"],
        # ... rest of split content ...
    }
    # ... rest unchanged ...

# Phase 6 (mới = EDA) - was old 5
if phase_id == 6:
    eda = sources["eda"]
    # ... existing EDA content with train_only_eda_scope added ...

# Phase 7 (mới = FE) - was old 6
if phase_id == 7:
    features = sources["features"]
    # ... existing FE content with train_only_feature_engineering added ...

# Phase 8 (mới = FS) - was old 7
if phase_id == 8:
    feature_sets = sources["feature_sets"]
    # ... existing FS content with train_only_feature_set_validation added ...
```

**Lưu ý:** Cấu trúc `if phase_id == X ... return` cần re-order để 5/6/7/8 match trong đúng thứ tự mới.

### 4.2. Notebook changes

#### 4.2.1. `notebook_course_work/CourseWork.ipynb` (59 cells)

**Mục đích:** Reorder cells, update markdown headings, add train-only notes.

**a. Cell reorder plan:**

| Cell range cũ | Nội dung cũ | Vị trí mới | Phase mới |
|---|---|---|---|
| 4-5 | Phase 1 Environment | 4-5 | 1 (unchanged) |
| 6-7 | Phase 2 Acquisition | 6-7 | 2 (unchanged) |
| 8-9 | Phase 3 Schema Audit | 8-9 | 3 (unchanged) |
| 10-11 | Phase 4 Temporal Audit | 10-11 | 4 (unchanged) |
| **12-41** | **Phase 5 EDA cũ** | **26-55 (sau Phase 5 mới)** | **6 (Train-Only EDA)** |
| **42-43** | **Phase 6 FE cũ** | **56-57** | **7 (Train-Only FE)** |
| **44-45** | **Phase 7 FS cũ** | **58-59** | **8 (Train-Only FS)** |
| **46-47** | **Phase 8 Split cũ** | **12-13** | **5 (Chronological Split)** |
| 48-49 | Phase 9 Scaling | 60-61 | 9 (unchanged) |
| 50-51 | Phase 10 Windowing | 62-63 | 10 (unchanged) |
| 52-53 | Phase 11 DataLoaders | 64-65 | 11 (unchanged) |
| 54-55 | Phase 12 Metrics | 66-67 | 12 (unchanged) |
| 56-57 | Phase 13 Registry | 68-69 | 13 (unchanged) |
| 58-59 | Phase 14 Persistence | 70-71 | 14 (unchanged) |

**NOTE:** Số cell tổng giữ nguyên 59-72 cells (tùy hiện trạng). Phase 5 mới ở vị trí 12-13 (markdown + code), Phase 6 mới ở 14-43 (markdown + 30 EDA sub-sections).

**b. Markdown heading updates:**

| Cell cũ | Heading cũ | Heading mới |
|---|---|---|
| 12-13 (sau move) | `## Phase 5 - Exploratory Data Analysis` | `## Phase 6 - Exploratory Data Analysis (Train-Only)` |
| 42-43 (sau move) | `## Phase 6 - Feature Engineering` | `## Phase 7 - Feature Engineering (Train-Only)` |
| 44-45 (sau move) | `## Phase 7 - Feature-Set Variants` | `## Phase 8 - Feature-Set Variants (Train-Only)` |
| 46-47 (sau move) | `## Phase 8 - Chronological Split` | `## Phase 5 - Chronological Split` |

**c. Markdown explanation additions:**

**Phase 6 (EDA Train-Only) — added note:**

```markdown
## Phase 6 - Exploratory Data Analysis (Train-Only)

Phase 6 thực hiện Exploratory Data Analysis trên **TRAIN rows only** (sinh ra bởi Phase 5 Chronological Split).
Validation và Test rows hoàn toàn bị ẩn trong toàn bộ Phase 6.

Lý do:
- Phase 5 cung cấp chronological 70/15/15 row membership.
- EDA descriptive statistics (mean, std, quantiles, correlation) chỉ nên dựa trên training-data distribution.
- Hypothesis generation (H-EDA-001 → H-EDA-010) chỉ motivate từ TRAIN patterns.
- Downstream phases (FE, FS, Scaling, etc.) sẽ dùng TRAIN-derived statistics.

Phạm vi cho phép (không thay đổi so với Phase 5 cũ):
- Mô tả target và features.
- Tạo EDA-only temporal columns trong df_eda.
- Visualize temporal pattern.
- Tạo descriptive correlation.
- Tạo hypothesis cho Phase sau.

Phạm vi KHÔNG được phép:
- Chọn final feature set.
- Chọn final lookback.
- Chọn final loss.
- Bật RevIN mặc định.
- Xóa spike, thay outlier bằng NaN.
- Interpolate sensor hoặc target.
- Tạo final regime từ full dataset.
- Tune bằng Test.
- Kết luận quan hệ nhân quả từ correlation.
```

**Phase 7 (FE Train-Only) — added note:**

```markdown
## Phase 7 - Feature Engineering (Train-Only)

Phase 7 thực hiện Feature Engineering trên **TRAIN rows** (sinh ra bởi Phase 5 Chronological Split).

Lưu ý quan trọng:
- Derived CSV `energydata_feature_engineered_v1.csv` vẫn chứa full 19,735 rows (giữ contract Phase 7).
- Validate invariants (no_new_nulls, time_feature_ranges_valid, cyclical_norm_valid, ...) chỉ apply trên TRAIN subset.
- Manifest bổ sung field `train_only_feature_engineering: True`.
- Feature engineering values (hour_sin, hour_cos, dow_sin, dow_cos, weekend) deterministic từ timestamp → không leakage.

Determinism:
- Deterministic calendar features (Phase 7 engine) — values identical giữa full vs TRAIN-only computation.
- Boundary preservation: phase_5_signoff.json (SPLIT-v1) là upstream prerequisite.
```

**Phase 8 (FS Train-Only) — added note:**

```markdown
## Phase 8 - Feature-Set Variants (Train-Only)

Phase 8 validate 6 feature-set variants (FS0_TF0, FS0_TF1, FS1_TF0, FS1_TF1, FS2_TF0, FS2_TF1) trên **TRAIN rows only**.

Lưu ý:
- `validate_feature_variants` filter theo `membership["split_id"] == "TRAIN"` trước khi check `missing_value_count`, `all_finite`.
- Validation logic (pairwise contracts, metadata exclusion, ...) không thay đổi.
- Manifest bổ sung field `train_only_feature_set_validation: True`.
- 6 variants structure (FEATURE_COMPONENTS, VARIANT_COMPONENTS, EXPECTED_FEATURE_COUNTS) không thay đổi.
```

**Phase 5 (Split) — added note:**

```markdown
## Phase 5 - Chronological Split

Phase 5 thực hiện deterministic chronological 70/15/15 split trên toàn bộ dataset.

Lưu ý quan trọng:
- Phase 5 không phụ thuộc FEATURES-v1 hay EDA-v1.
- Phase 5 là prerequisite cho Phase 6 (EDA Train-Only), Phase 7 (FE Train-Only), Phase 8 (FS Train-Only).
- Row membership deterministic floor-based:
  - TRAIN: 0 → floor(0.70 * 19735) = 13,814 rows
  - VALIDATION: 13,814 → floor(0.85 * 19735) = 16,774 rows
  - TEST: 16,774 → 19,735 rows
- Test struct metadata only (no value disclosure).
- Test firewall được enforce từ Phase 5 trở đi.
```

**d. Code cell updates:**

| Cell call | Update |
|---|---|
| `materialize_phase_5(project_root)` (trong cell 12 cũ) | Giữ nguyên nhưng giờ gọi `materialize_phase_5` cho Phase 5 = Split (sau refactor, function name `materialize_phase_8` ở splitting.py được exposed as `materialize_phase_5` trong notebook cell). |
| `materialize_phase_5(project_root)` (trong cell 42 cũ) | Đổi thành `materialize_phase_6(project_root)` (Phase 6 = EDA). |
| `materialize_phase_6(project_root)` (trong cell 44 cũ) | Đổi thành `materialize_phase_7(project_root)` (Phase 7 = FE). |
| `materialize_phase_7(project_root)` (trong cell 46 cũ) | Đổi thành `materialize_phase_8(project_root)` (Phase 8 = FS). |
| `materialize_phase_8(project_root)` (trong cell 48 cũ) | REPLACE: `materialize_phase_8` → `materialize_phase_5(project_root)` (Phase 5 mới = Split, function name kept). |

**Chi tiết hơn:**

Trong notebook, syntax standard:
```python
# Phase 5 (mới = Split)
signoff = materialize_phase_8(project_root)  # Function name kept as-is
# NOTE: Function name keeps historical "phase_8" to avoid ownership refactoring
# Phase number 5 corresponds to chronological split in new architecture ordering.
```

Hoặc alternative cleaner approach:
```python
# Phase 5 (mới = Split) - function name kept for backward compat
from course_work.data.splitting import materialize_phase_8 as materialize_phase_5_split
signoff = materialize_phase_5_split(project_root)
```

**Recommend:** Sử dụng alias `materialize_phase_8 as materialize_phase_5_split` để rõ ràng trong notebook.

### 4.3. Architecture rule changes

#### 4.3.1. `docs/RULE_BASE/architecture_rule.md`

**Mục đích:** Update §9 Phase-to-module mapping, §13 Artifact ownership, §14.3 Notebook boundary, §7.9.1 Presentation allowlist, §25 Transition state.

**a. §9 Phase-to-module mapping table update:**

```markdown
## 9. Phase-to-module mapping  (UPDATED)

| Phase | Owner chính | Owner hỗ trợ | Notebook responsibility |
|---|---|---|---|
| 0 | `contracts/coursework.py` | ... | (unchanged) |
| 1 | `utils/environment.py` | ... | (unchanged) |
| 2 | `data/acquisition.py` | ... | (unchanged) |
| 3 | `data/schema.py` | ... | (unchanged) |
| 4 | `data/temporal.py` | ... | (unchanged) |
| **5** | **`data/splitting.py`** | `data/temporal.py`, `utils/artifacts.py`, `reporting/phase_summary.py` | Gọi `materialize_phase_8` API (function name kept) và hiển thị Chronological membership cùng Split allocation |
| **6** | **`data/eda.py`** | **`reporting/eda.py`**, **`reporting/phase_summary.py`**, `utils/artifacts.py`, Human-approved direct notebook EDA (Train-Only scope) | Gọi EDA workflow với TRAIN-only filter, hiển thị outputs |
| **7** | **`data/features.py`** | `utils/artifacts.py`, `reporting/phase_summary.py` | Gọi `materialize_phase_7` API (FE Train-Only validation) và hiển thị HTML summary |
| **8** | **`data/feature_sets.py`** | `data/features.py`, `utils/artifacts.py`, `reporting/phase_summary.py` | Gọi `materialize_phase_8` API (FS Train-Only validation) và hiển thị HTML summary |
| 9 | `data/scaling.py` | (existing) | (unchanged) |
| ... | ... | ... | ... |
```

**b. §13 Artifact ownership update:**

```markdown
## 13. Artifact ownership Phase 0-14  (UPDATED)

artifacts/contracts
-> Phase 0 contract và sign-off

artifacts/environment
-> Phase 1 inventory, smoke tests, dependency freeze và sign-off

artifacts/acquisition
-> Phase 2 acquisition log và sign-off

artifacts/schema
-> Phase 3 schema outputs và sign-off

artifacts/temporal
-> Phase 4 temporal outputs và sign-off

artifacts/splits
-> Phase 5 chronological membership, boundaries, fingerprints, structural/leakage audits, Train/Validation-only diagnostics, timestamp-only figure, manifest, discrepancies và sign-off  (MOVED UP)

artifacts/eda
-> Phase 6 tables, figures, manifest, anomalies (Train-Only scope) và sign-off  (MOVED DOWN)

artifacts/features
-> Phase 7 registry, lineage, availability, leakage audit, engineering audit, manifest, derived checksum, discrepancies (Train-Only invariant scope) và sign-off  (RENUMBERED)

artifacts/feature_sets
-> Phase 8 components, ordered variant registries, fingerprints, lineage, semantic/leakage/order audits (Train-Only validation scope), manifest, discrepancies, human-readable specification và sign-off  (RENUMBERED)

# Phases 9-14 unchanged
```

**c. §14.3 Notebook boundary docstring update:**

```markdown
### 14.3. Cell order  (UPDATED)

Notebook phải có section tuần tự:

```text
Phase 1
Phase 2
Phase 3
Phase 4
Phase 5  (Chronological Split - prerequisite cho EDA/FE/FS)
Phase 6  (Exploratory Data Analysis - Train-Only)
Phase 7  (Feature Engineering - Train-Only invariant scope)
Phase 8  (Feature-Set Variants - Train-Only validation scope)
Phase 9
Phase 10
Phase 11
Phase 12
Phase 13
Phase 14
Phase 1-14 Boundary
```

Phase 0 vẫn là canonical prerequisite nội bộ nhưng không có heading, orchestration cell, import hoặc output trong notebook.
```

**d. §7.9.1 Presentation allowlist update:**

```markdown
### 7.9.1. (continued)  (UPDATED)

Notebook presentation bắt buộc tuân thủ:

```text
Header chỉ gồm Phase, artifact version và status
Phase 1 hiển thị Environment overview và Core package versions
Phase 2 hiển thị Dataset overview
Phase 3 hiển thị Schema overview và Critical schema checks
Phase 4 hiển thị Temporal coverage và Critical temporal checks
Phase 5 chỉ hiển thị Chronological membership và Split allocation  (UPDATED)
Phase 6 chỉ hiển thị header cùng các output EDA 5.1-5.14 đã được duyệt (Train-Only scope)  (UPDATED)
Phase 7 hiển thị Feature overview và Engineered feature registry (Train-Only invariant scope)  (UPDATED)
Phase 8 chỉ hiển thị Feature-set registry và không hiển thị fingerprint (Train-Only validation scope)  (UPDATED)
Phase 9 chỉ hiển thị X scaler bundles và Target scaling options
Phase 10 chỉ hiển thị Window contract và Common target population
Phase 11 chỉ hiển thị Dataset population và Loader policy
Phase 12 chỉ hiển thị Metric registry và Evaluation policy
Phase 13 chỉ hiển thị Registry state và Core safeguards
Phase 14 chỉ hiển thị Validation performance và Baseline contract
```
```

**e. §15 EDA boundary extension:**

```markdown
## 15. EDA boundary  (UPDATED)

Phase 6 (renumbered from Phase 5) được phép:

```text
Mô tả target và features trên TRAIN rows only.
Tạo EDA-only temporal columns trong df_eda trên TRAIN rows only.
Visualize temporal pattern trên TRAIN rows only.
Tạo descriptive correlation trên TRAIN rows only.
Tạo hypothesis cho Phase sau (H-EDA-001 → H-EDA-010) chỉ motivate từ TRAIN patterns.
Thực hiện direct descriptive EDA trong CourseWork.ipynb theo plan CW-PHASE-6-EDA-DIRECT-001-TRAIN-ONLY (renumbered).
```

Phase 6 KHÔNG được:

```text
Chọn final feature set.
Chọn final lookback.
Chọn final loss.
Bật RevIN mặc định.
Xóa spike.
Thay outlier bằng NaN.
Interpolate sensor hoặc target.
Tạo final regime từ full dataset.
Tune bằng Test.
Kết luận quan hệ nhân quả từ correlation.
Sử dụng Validation hoặc Test rows trong bất kỳ descriptive statistic nào.
```

**NOTE:** Plan reference đổi từ `CW-PHASE-5-EDA-DIRECT-001` → `CW-PHASE-6-EDA-DIRECT-001-TRAIN-ONLY` (plan name change).
```

**f. §25 Transition state update:**

```markdown
## 25. Transition state  (UPDATED)

Trạng thái hiện hành:

```text
Phase 0-14 có canonical source owner và signed artifact.
CourseWork.ipynb giữ direct descriptive Phase 6 EDA exception đã được Human duyệt, Train-Only scope.  (UPDATED)
Phase 0 giữ vai trò contract nội bộ và không xuất hiện trong CourseWork.ipynb.
Phase 1-14 dùng presentation allowlist tối giản trên notebook.

Phase 5 trong notebook chỉ gọi public API và hiển thị outputs.  (UPDATED)
SPLIT-v1 khóa chronological 70/15/15 row membership, WB0 primary metadata và Test firewall.  (MOVED UP)

Phase 6 trong notebook chỉ gọi public API và hiển thị outputs.  (UPDATED)
EDA-v1 giữ raw lineage, descriptive scope chỉ áp dụng trên TRAIN subset, 10 hypothesis H-EDA-001 → H-EDA-010 motivate từ TRAIN patterns.  (UPDATED)

Phase 7 trong notebook chỉ gọi public API và hiển thị outputs.  (RENUMBERED)
FEATURES-v1 giữ raw lineage, thêm 5 deterministic calendar features, invariant validation scope giới hạn TRAIN rows.  (UPDATED)

Phase 8 trong notebook chỉ gọi public API và hiển thị outputs.  (RENUMBERED)
FEATURESETS-v1 khóa sáu ordered variants và fingerprints, validation scope giới hạn TRAIN rows, không sao chép data.  (UPDATED)

Phase 9 trong notebook chỉ gọi public API và hiển thị outputs.
SCALING-v1 khóa sáu X scaler bundles, YS0 identity, một YS1 scaler và Train-only fit protocol.

# ... (Phases 9-14 unchanged)
```

**g. §26 Architecture validation checklist update:**

```markdown
## 26. Architecture validation checklist  (UPDATED)

```text
[x] Canonical root rõ ràng.
[x] Canonical notebook rõ ràng.
[x] Source ownership rõ ràng.
[x] Phase 0-14 mapping đầy đủ.  (UPDATED Phase 5/6/7/8 ordering)
[x] Dependency direction rõ ràng.
[x] Raw data contract rõ ràng.
[x] Derived-view contract rõ ràng.
[x] Configuration ownership rõ ràng.
[x] Artifact lifecycle rõ ràng.
[x] Notebook boundary rõ ràng.  (UPDATED cell order Phase 5/6/7/8)
[x] EDA boundary rõ ràng.  (UPDATED Phase 6 Train-Only)
[x] Test ownership rõ ràng.
[x] Phase gate rõ ràng.
[x] Sign-off contract rõ ràng.
[x] Git tracking rule rõ ràng.
[x] Change-control gate rõ ràng.
[x] Phase 5 có canonical owner, artifacts, tests và notebook orchestration boundary.  (UPDATED)
[x] Phase 6 có canonical owner, artifacts, tests và notebook orchestration boundary (Train-Only).  (UPDATED)
[x] Phase 7 có canonical owner, artifacts, tests và notebook orchestration boundary (Train-Only invariant).  (UPDATED)
[x] Phase 8 có canonical owner, artifacts, tests và notebook orchestration boundary (Train-Only validation).  (UPDATED)
# ... (rest unchanged)
```
```

### 4.4. Sign-off & manifest updates

#### 4.4.1. Phase 5 signoff (mới = Chronological Split)

**File rename:**
- `artifacts/splits/phase_8_signoff.json` → `artifacts/splits/phase_5_signoff.json`

**Content update:**

```json
{
  "artifact_version": "SPLIT-v1",
  "phase_id": 5,
  "phase_version": "PHASE-5-v1",
  "created_at": "<existing created_at>",
  "environment_id": "ENV-v1",
  "dataset_revision": "DATA-v1",
  "feature_version": "FEATURES-v1",  // deprecated, kept for back-compat
  "feature_set_version": "FEATURESETS-v1",  // deprecated, kept for back-compat
  "input_paths": [
    "artifacts/temporal/phase_4_signoff.json",
    "artifacts/temporal/temporal_manifest.json",
    "configs/base/coursework_contract.json"
  ],
  "input_checksums": { /* recomputed */ },
  "output_paths": [ /* same as before */ ],
  "output_checksums": { /* recomputed */ },
  "config_fingerprint": "<from phase 0>",
  "global_split_fingerprint": "<existing>",
  "test_locked": true,
  "status": "PASS",
  "tests": [...],
  "warnings": [],
  "discrepancies": []
}
```

#### 4.4.2. Phase 6 signoff (mới = EDA)

**File rename:**
- `artifacts/eda/phase_5_signoff.json` → `artifacts/eda/phase_6_signoff.json`

**Content update:**

```json
{
  "artifact_version": "EDA-v1",
  "phase_id": 6,
  "phase_version": "PHASE-6-v1",
  "created_at": "<existing>",
  "train_only_eda_scope": true,  // NEW
  "train_rows": 13814,  // NEW
  "train_only_eda_validated": true,  // NEW
  "input_paths": [
    "artifacts/acquisition/phase_2_signoff.json",
    "artifacts/schema/phase_3_signoff.json",
    "artifacts/temporal/phase_4_signoff.json",
    "artifacts/splits/phase_5_signoff.json",  // NEW
    "data/raw_data/energydata_complete.csv"
  ],
  "input_checksums": { /* recomputed */ },
  "output_paths": [ /* same as before */ ],
  "output_checksums": { /* recomputed */ },
  "status": "PASS",
  "tests": [...],
  "warnings": [],
  "discrepancies": []
}
```

#### 4.4.3. Phase 7 signoff (mới = Feature Engineering)

**File rename:**
- `artifacts/features/phase_6_signoff.json` → `artifacts/features/phase_7_signoff.json`

**Content update:**

```json
{
  "artifact_version": "FEATURES-v1",
  "phase_id": 7,
  "phase_version": "PHASE-7-v1",
  "created_at": "<existing>",
  "train_only_feature_engineering": true,  // NEW
  "train_rows": 13814,  // NEW
  "feature_engineering_train_only_validated": true,  // NEW
  "input_paths": [
    "artifacts/contracts/phase_0_signoff.json",
    "artifacts/environment/phase_1_signoff.json",
    "artifacts/acquisition/phase_2_signoff.json",
    "artifacts/schema/phase_3_signoff.json",
    "artifacts/temporal/phase_4_signoff.json",
    "artifacts/splits/phase_5_signoff.json",  // NEW: SPLIT-v1 dependency
    "data/raw_data/energydata_complete.csv",
    "configs/base/coursework_contract.json"
  ],
  "input_checksums": { /* recomputed */ },
  "output_paths": [ /* same as before */ ],
  "output_checksums": { /* recomputed */ },
  "status": "PASS",
  "tests": [...],
  "warnings": [],
  "discrepancies": []
}
```

#### 4.4.4. Phase 8 signoff (mới = Feature-Set Variants)

**File rename:**
- `artifacts/feature_sets/phase_7_signoff.json` → `artifacts/feature_sets/phase_8_signoff.json`

**Content update:**

```json
{
  "artifact_version": "FEATURESETS-v1",
  "phase_id": 8,
  "phase_version": "PHASE-8-v1",
  "created_at": "<existing>",
  "train_only_feature_set_validation": true,  // NEW
  "train_rows": 13814,  // NEW
  "input_paths": [
    "artifacts/features/phase_7_signoff.json",  // NEW: FE dependency
    "artifacts/features/feature_engineering_manifest.json",
    "artifacts/features/feature_registry.csv",
    "artifacts/features/feature_lineage.csv",
    "artifacts/features/feature_availability.csv",
    "artifacts/features/feature_leakage_audit.csv",
    "data/interim/uci_appliances_energy_prediction/energydata_feature_engineered_v1.csv",
    "configs/base/coursework_contract.json"
  ],
  "input_checksums": { /* recomputed */ },
  "output_paths": [ /* same as before */ ],
  "output_checksums": { /* recomputed */ },
  "status": "PASS",
  "tests": [...],
  "warnings": [],
  "discrepancies": []
}
```

#### 4.4.5. Phase 9-14 signoff updates

**Phase 9 (Scaling):**
- Update `input_paths` để reference `phase_5_signoff.json` (SPLIT-v1 mới) thay vì `phase_8_signoff.json` (cũ).
- Update `input_paths` cho phases 6/7/8 nếu downstream phases có hard-coded path.

**Phase 10 (Windows):**
- Tương tự Phase 9.

**Phase 11-14:** Nếu hard-code upstream paths → update.

### 4.5. Tests cần update

#### 4.5.1. `tests/contracts/`

**Phase 5/6/7/8 contract tests** — update phase number expectations:

```python
# tests/contracts/test_phase_5_chronological_split.py (renamed from test_phase_8_chronological_split.py)
def test_phase_5_signoff_phase_id():
    signoff = read_json(PROJECT_ROOT / "artifacts/splits/phase_5_signoff.json")
    assert signoff["phase_id"] == 5
    assert signoff["phase_version"] == "PHASE-5-v1"
```

```python
# tests/contracts/test_phase_6_eda.py (renamed from test_phase_5_eda.py)
def test_phase_6_signoff_phase_id():
    signoff = read_json(PROJECT_ROOT / "artifacts/eda/phase_6_signoff.json")
    assert signoff["phase_id"] == 6
    assert signoff["phase_version"] == "PHASE-6-v1"
    assert signoff["train_only_eda_scope"] is True
```

```python
# tests/contracts/test_phase_7_features.py (renamed from test_phase_6_features.py)
def test_phase_7_signoff_phase_id():
    signoff = read_json(PROJECT_ROOT / "artifacts/features/phase_7_signoff.json")
    assert signoff["phase_id"] == 7
    assert signoff["phase_version"] == "PHASE-7-v1"
    assert signoff["train_only_feature_engineering"] is True
```

```python
# tests/contracts/test_phase_8_feature_sets.py (renamed from test_phase_7_feature_sets.py)
def test_phase_8_signoff_phase_id():
    signoff = read_json(PROJECT_ROOT / "artifacts/feature_sets/phase_8_signoff.json")
    assert signoff["phase_id"] == 8
    assert signoff["phase_version"] == "PHASE-8-v1"
    assert signoff["train_only_feature_set_validation"] is True
```

#### 4.5.2. `tests/integration/`

**Integration tests Phase 5 → 6 (Split → EDA):**

```python
# tests/integration/test_phase_5_to_6.py (renamed from test_phase_7_to_8.py)
def test_phase_5_to_6_correct_order():
    # Phase 5 must complete before Phase 6
    phase_5_signoff = read_json(PROJECT_ROOT / "artifacts/splits/phase_5_signoff.json")
    phase_6_signoff = read_json(PROJECT_ROOT / "artifacts/eda/phase_6_signoff.json")
    assert phase_5_signoff["status"] == "PASS"
    assert phase_6_signoff["status"] == "PASS"
    # Phase 6 input must include Phase 5 signoff
    assert "artifacts/splits/phase_5_signoff.json" in phase_6_signoff["input_paths"]
```

**Integration tests Phase 6 → 7 (EDA → FE)**, **Phase 7 → 8 (FE → FS)**: Tương tự.

#### 4.5.3. `tests/unit/`

**Unit tests cho `eda.py` với new train-only scope:**

```python
# tests/unit/test_eda_train_only.py
def test_prepare_eda_analysis_filters_train():
    analysis = prepare_eda_analysis(PROJECT_ROOT)
    assert "train_only_eda_validated" in analysis
    assert analysis["train_only_eda_validated"] is True
    assert analysis["train_row_count_used"] == 13814  # TRAIN rows
    # Verify eda_view is train-only
    assert len(analysis["eda_view"]) == 13814
```

**Unit tests cho `features.py` với new train-only invariant scope:**

```python
# tests/unit/test_features_train_only.py
def test_validate_feature_invariants_train_only():
    # ...
    assert manifest["train_only_feature_engineering"] is True
    assert manifest["feature_engineering_train_only_validated"] is True
```

**Unit tests cho `feature_sets.py` với new train-only validation scope:**

```python
# tests/unit/test_feature_sets_train_only.py
def test_validate_feature_variants_train_only():
    # ...
    assert manifest["train_only_feature_set_validation"] is True
```

---

## 5. Lộ trình thực thi (Execution Roadmap)

### Phase R1: Architecture amendment (preparation)

**Mục đích:** Xin Human approval cho architecture change.

**Bước thực hiện:**

1. Tạo issue/pre-process plan file này: `docs/plan-doc/analysis_error/phase_reorder_split_before_eda_train_only_refactor_plan.md`.
2. Xin Human approval cho architecture change (qua §24 change-control gate).
3. Sau khi approved, update `docs/RULE_BASE/architecture_rule.md`:
   - §9 Phase-to-module mapping
   - §13 Artifact ownership
   - §14.3 Notebook boundary docstring
   - §7.9.1 Presentation allowlist
   - §15 EDA boundary extension
   - §25 Transition state
   - §26 Architecture validation checklist

**Deliverable:** Architecture rule amended.

**Duration estimate:** 1-2 days (Human review time).

### Phase R2: Source code changes (Python)

**Mục đích:** Update 4 module Python.

**Bước thực hiện:**

1. **Update `eda.py`:**
   - Add `load_validated_split_membership` import.
   - Modify `prepare_eda_analysis` để filter TRAIN rows.
   - Add train-only audit fields.

2. **Update `features.py`:**
   - Modify `verify_phase_6_inputs` để check `phase_5_signoff.json` (SPLIT-v1) thay vì `phase_5_signoff.json` (EDA-v1).
   - Update `UPSTREAM_SIGNOFFS`.
   - Modify `build_feature_view` signature thêm `membership` param.
   - Modify `materialize_phase_6` signature → thêm train-only validation.
   - Update `verify_existing_signoff` và `signoff_path`.

3. **Update `feature_sets.py`:**
   - Modify `verify_phase_7_inputs` để check `phase_7_signoff.json` (FEATURES-v1 mới) thay vì `phase_6_signoff.json` (cũ).
   - Modify `validate_feature_variants` signature thêm `membership` param.
   - Modify `materialize_phase_7` signature → thêm train-only validation.
   - Update `verify_existing_signoff` và `signoff_path`.

4. **Update `splitting.py`:**
   - Update `signoff_path` từ `phase_8_signoff.json` → `phase_5_signoff.json`.
   - Update `signoff` content: `phase_id: 5`, `phase_version: "PHASE-5-v1"`.
   - Update `input_paths` để remove FE/FS dependencies.
   - Update audit messages.

5. **Update `scaling.py`:**
   - Update `verify_phase_9_inputs` để reference `phase_5_signoff.json` (SPLIT-v1 mới) thay vì `phase_8_signoff.json` (cũ).
   - Update `input_paths` cho phase 7/8 (FE/FS mới).

6. **Update `windows.py`:**
   - Update `required_signoffs` keys.
   - Update `input_paths` cho phase 7/8/5 (mới).

7. **Update `reporting/phase_summary.py`:**
   - Update `PHASE_NAMES` keys 5/6/7/8.
   - Update `LOG_FILENAMES` keys 5/6/7/8.
   - Update `SOURCE_SPECS` keys 5/6/7/8.
   - Update `PRESENTATION_SPECS` keys 5/6/7/8.
   - Update `_phase_content()` branches theo phase mới.

**Deliverable:** Source code updated.

**Duration estimate:** 2-3 days.

### Phase R3: Notebook reordering

**Mục đích:** Reorder cells, update markdown headings, add train-only notes.

**Bước thực hiện:**

1. **Backup notebook hiện tại:**
   ```bash
   cp notebook_course_work/CourseWork.ipynb notebook_course_work/CourseWork.ipynb.bak
   ```

2. **Di chuyển cells Phase 5 EDA xuống vị trí Phase 6 mới:**
   - Cells 12-41 (30 cells) → vị trí mới 26-55.

3. **Di chuyển cells Phase 8 Split lên vị trí Phase 5 mới:**
   - Cells 46-47 → vị trí mới 12-13.

4. **Di chuyển cells Phase 6 FE xuống vị trí Phase 7 mới:**
   - Cells 42-43 → vị trí mới 56-57.

5. **Di chuyển cells Phase 7 FS xuống vị trí Phase 8 mới:**
   - Cells 44-45 → vị trí mới 58-59.

6. **Đánh số lại markdown headings:**
   - `## Phase 5 - Exploratory Data Analysis` → `## Phase 6 - Exploratory Data Analysis (Train-Only)`
   - `## Phase 6 - Feature Engineering` → `## Phase 7 - Feature Engineering (Train-Only)`
   - `## Phase 7 - Feature-Set Variants` → `## Phase 8 - Feature-Set Variants (Train-Only)`
   - `## Phase 8 - Chronological Split` → `## Phase 5 - Chronological Split`

7. **Update code cell `materialize_phase_X(PROJECT_ROOT)` calls:**
   - Cell 12 (Split, Phase 5): `materialize_phase_8(project_root)` (function name kept, alias as `materialize_phase_5_split`).
   - Cell 14 (EDA, Phase 6): `materialize_phase_6(project_root)` (NA → keep alias).
   - Cell 56 (FE, Phase 7): `materialize_phase_7(project_root)`.
   - Cell 58 (FS, Phase 8): `materialize_phase_8(project_root)`.

8. **Thêm train-only note vào EDA/FE/FS markdown** (xem §4.2.1c).

9. **Add Phase 5 (Split) markdown note** (xem §4.2.1c).

**Deliverable:** Notebook reordered.

**Duration estimate:** 1-2 days (notebook JSON manipulation).

### Phase R4: Re-materialization

**Mục đích:** Xóa signoff artifacts cũ và re-run notebook từ đầu.

**Bước thực hiện:**

1. **Delete các signoff artifacts cũ:**
   ```bash
   rm artifacts/eda/phase_5_signoff.json
   rm artifacts/eda/phase_5_signoff.json.sha256  # if exists
   rm artifacts/features/phase_6_signoff.json
   rm artifacts/feature_sets/phase_7_signoff.json
   rm artifacts/splits/phase_8_signoff.json
   ```

2. **Re-run notebook từ đầu:**
   ```bash
   jupyter nbconvert --to notebook --execute notebook_course_work/CourseWork.ipynb --output CourseWork.ipynb
   ```

3. **Verify all phase signoffs PASS:**
   - Phase 5: `artifacts/splits/phase_5_signoff.json` status=PASS
   - Phase 6: `artifacts/eda/phase_6_signoff.json` status=PASS, train_only_eda_validated=true
   - Phase 7: `artifacts/features/phase_7_signoff.json` status=PASS, train_only_feature_engineering=true
   - Phase 8: `artifacts/feature_sets/phase_8_signoff.json` status=PASS, train_only_feature_set_validation=true
   - Phase 9-14: status=PASS, upstream signoff references correct

**Deliverable:** Fresh artifacts generated.

**Duration estimate:** 1 day.

### Phase R5: Validation

**Mục đích:** Verify tất cả tests pass và outputs correct.

**Bước thực hiện:**

1. **Run all unit tests:**
   ```bash
   pytest tests/unit/ -v
   ```

2. **Run all integration tests:**
   ```bash
   pytest tests/integration/ -v
   ```

3. **Run all contract tests:**
   ```bash
   pytest tests/contracts/ -v
   ```

4. **Verify output artifacts:**
   - All 5 signoff files (Phase 5-8 new) exist.
   - All checksums match.
   - All manifest fields correctly populated.

5. **Visual sanity check:**
   - EDA figures dựa trên TRAIN subset (figures should look different from before).
   - Feature statistics match Train-only scope.
   - Train/Validation distribution summary in Phase 5 manifest.
   - Phase 9-14 fingerprints consistent.

**Deliverable:** Validation report.

**Duration estimate:** 1-2 days.

---

## 6. Risks & Mitigations

### Risk 1: Architecture rule violation (đổi Phase mapping không có Human approval)

**Risk level:** HIGH
**Description:** Nếu refactor được thực hiện mà không có Human approval trước, vi phạm §24 change-control gate.

**Mitigation:**
- Plan này yêu cầu Human approval trước khi bất kỳ source modification nào.
- Sau approval, mới tiến hành Phase R2 (source code changes).

### Risk 2: Notebook boundary violation

**Risk level:** MEDIUM
**Description:** Reorder cells có thể vô tình thêm logic hoặc duplicate cells.

**Mitigation:**
- Backup notebook trước khi reorder.
- Chỉ reorder cells, KHÔNG thêm logic.
- Sau reorder, verify cell count = 59 (or whatever the current count is).
- Spot-check execution: chạy cell 12 chỉ thực thi Split logic, không phải EDA.

### Risk 3: EDA output khác với trước (vì chỉ dùng Train)

**Risk level:** MEDIUM (expected behavior)
**Description:** Sau refactor, EDA statistics (mean, std, quantiles, correlation) sẽ khác với trước. Đây là expected behavior vì Train-only scope là intentional.

**Mitigation:**
- Document change trong `CURRENT_FLOW_SUMMARY.md`.
- Save old EDA outputs (Pre-refactor) ở `docs/plan-doc/pre_refactor_artifacts/` cho comparison.
- Add note trong CURSOR-RULES hoặc README: "EDA statistics are Train-only derived; Validation/Test slice is hidden."

### Risk 4: Old signoff artifacts không tương thích

**Risk level:** MEDIUM
**Description:** Signoff files cũ (Phase 5 EDA, Phase 6 FE, etc.) có `phase_id` cũ. Nếu giữ lại, sẽ conflict với new signoffs.

**Mitigation:**
- Phase R4 step 1: Delete all old signoff files trước khi re-materialize.
- Verify file paths không có old signoff còn sót lại.

### Risk 5: Tests fail vì cũ reference phase number

**Risk level:** MEDIUM
**Description:** Tests hiện tại expect `phase_id` = 5 (EDA cũ), 6 (FE cũ), etc. Sau refactor, expect `phase_id` = 5 (Split), 6 (EDA), 7 (FE), 8 (FS).

**Mitigation:**
- Phase R2 parallel: Update tests sync với source code.
- Run tests song song với source modifications.
- Update CI pipeline (nếu có) để match new phase numbering.

### Risk 6: Phase 0/1/2/3/4 quyết định từ full dataset

**Risk level:** LOW (intended)
**Description:** Phase 0-4 (env, acquisition, schema, temporal) KHÔNG filter theo split vì chúng chỉ verify raw data integrity, KHÔNG đụng giá trị target/features.

**Mitigation:**
- Document rõ trong plan: Phase 0-4 giữ nguyên full dataset scope.
- Phase 5 (Split) là phase đầu tiên introduce split concept.

### Risk 7: Phase 14 persistence baseline input_paths references

**Risk level:** LOW
**Description:** `baselines/persistence.py` có thể hard-code `phase_5_signoff.json` (EDA-v1 cũ) trong input_paths verification.

**Mitigation:**
- Phase R2 step 6: Verify `baselines/persistence.py` không hard-code `phase_5_signoff.json`.
- Nếu có, update thành `phase_5_signoff.json` (SPLIT-v1 mới) — may require additional context check.

### Risk 8: Function name `materialize_phase_8` vs phase_id 5 confusion

**Risk level:** LOW
**Description:** Function name `materialize_phase_8` trong `splitting.py` không khớp với phase_id mới (=5). Có thể gây confusion cho developers.

**Mitigation:**
- Add docstring note: "Function name `materialize_phase_8` is kept for backward compatibility. Phase 5 = Chronological Split."
- Sử dụng alias `materialize_phase_8 as materialize_phase_5_split` trong notebook.
- Update architecture rule để document decision.

### Risk 9: Tests contract references phase number inconsistency

**Risk level:** MEDIUM
**Description:** `tests/contracts/` check phase_id hard-coded values. Sau refactor, các contracts check 5/6/7/8 changed.

**Mitigation:**
- Phase R2 parallel: Update contract tests.
- Rename test files: `test_phase_5_eda.py` → `test_phase_6_eda.py`, etc.
- Update test classes/functions.

### Risk 10: Data lineage preservation

**Risk level:** LOW
**Description:** Sau refactor, `derived_file_path` của FEATURES-v1 (`data/interim/uci_appliances_energy_prediction/energydata_feature_engineered_v1.csv`) vẫn chứa full 19,735 rows. Đây là intentional nhưng cần document.

**Mitigation:**
- Document rõ: "FEATURES-v1 derived CSV = full timeline (timestamps + features), but invariants/validation scoped to TRAIN."
- Update README_FEATURE_SETS.md để reflect.

### Risk 11: EDA figures based on TRAIN subset

**Risk level:** LOW
**Description:** Sau refactor, EDA figures (histograms, scatter plots, timelines) dựa trên TRAIN subset. Figures sẽ cover only first 13,814 rows.

**Mitigation:**
- Document trong `EDA-04 target timeline` figure description: "Timeline covers TRAIN rows only (first 13,814 of 19,735)."
- Add split labels in figures.

### Risk 12: Hypothesis H-EDA-008 (temporal_drift) sẽ thay đổi content

**Risk level:** LOW (intended)
**Description:** `rolling_first` và `rolling_last` sẽ đều từ TRAIN subset. Số liệu sẽ khác.

**Mitigation:**
- Expected: hypothesis content sẽ thay đổi sau refactor.
- Document trong CURRENT_FLOW_SUMMARY.md.

---

## 7. Verification Checklist

Sau khi refactor hoàn tất, các tiêu chí sau phải pass:

- [ ] **Architecture rule approved by Human.**
- [ ] §9 Phase-to-module mapping updated (Phase 5 = Split, 6 = EDA, 7 = FE, 8 = FS).
- [ ] §13 Artifact ownership updated (Phase 5 = splits, 6 = eda, 7 = features, 8 = feature_sets).
- [ ] §14.3 Notebook boundary docstring updated.
- [ ] §7.9.1 Presentation allowlist updated.
- [ ] §15 EDA boundary extension (TRAIN-only scope).
- [ ] §25 Transition state updated.
- [ ] §26 Architecture validation checklist updated.

### 7.2. Source code (4 files updated)

- [ ] `src/course_work/data/eda.py` — `prepare_eda_analysis` filters TRAIN rows.
- [ ] `src/course_work/data/features.py` — `verify_phase_6_inputs` checks `phase_5_signoff.json` (SPLIT-v1), `build_feature_view` accepts membership.
- [ ] `src/course_work/data/feature_sets.py` — `validate_feature_variants` filters TRAIN.
- [ ] `src/course_work/reporting/phase_summary.py` — `PHASE_NAMES`, `LOG_FILENAMES`, `SOURCE_SPECS`, `PRESENTATION_SPECS`, `_phase_content` keys 5/6/7/8 updated.

### 7.3. Source code (4 files NOT updated)

- [ ] `src/course_work/data/splitting.py` — Logic unchanged, only signoff path/content updated.
- [ ] `src/course_work/data/scaling.py` — Only upstream signoff reference updated.
- [ ] `src/course_work/data/windows.py` — Only upstream signoff references updated.
- [ ] `src/course_work/evaluation/metrics.py`, `experiments/registry.py`, `baselines/persistence.py` — If hard-coded upstream paths, updated.

### 7.4. Notebook reordered

- [ ] Cells 12-13 (Phase 5 = Split) moved from old 46-47.
- [ ] Cells 26-55 (Phase 6 = EDA Train-Only) moved from old 12-41.
- [ ] Cells 56-57 (Phase 7 = FE Train-Only) moved from old 42-43.
- [ ] Cells 58-59 (Phase 8 = FS Train-Only) moved from old 44-45.
- [ ] Markdown headings updated.
- [ ] Code cell `materialize_phase_X` calls updated.
- [ ] Train-only notes added to EDA/FE/FS markdown.
- [ ] Phase 5 (Split) markdown note added.

### 7.5. Re-materialization succeeds

- [ ] Old signoff files deleted (Phase 5 EDA cũ, Phase 6 FE cũ, Phase 7 FS cũ, Phase 8 Split cũ).
- [ ] Fresh signoff files generated:
  - `artifacts/splits/phase_5_signoff.json` (status=PASS, phase_id=5)
  - `artifacts/eda/phase_6_signoff.json` (status=PASS, phase_id=6, train_only_eda_validated=true)
  - `artifacts/features/phase_7_signoff.json` (status=PASS, phase_id=7, train_only_feature_engineering=true)
  - `artifacts/feature_sets/phase_8_signoff.json` (status=PASS, phase_id=8, train_only_feature_set_validation=true)
- [ ] Phase 9-14 signoff files regenerated with new upstream references.

### 7.6. Tests pass

- [ ] Unit tests pass.
- [ ] Integration tests pass.
- [ ] Contract tests pass.
- [ ] Notebook runs to completion without errors.

### 7.7. Output artifacts integrity

- [ ] All 5 (Phase 5-8 new) signoff files have `status: "PASS"`.
- [ ] All checksums match.
- [ ] All manifest fields correctly populated.
- [ ] Train-only fields present where expected.

### 7.8. Validation set và Test set scope enforcement

- [ ] Validation set KHÔNG xuất hiện trong Phase 6 EDA tables/figures.
- [ ] Test set KHÔNG xuất hiện trong Phase 6 EDA tables/figures.
- [ ] Validation set KHÔNG xuất hiện trong Phase 7 FE invariants.
- [ ] Test set KHÔNG xuất hiện trong Phase 7 FE invariants.
- [ ] Validation set KHÔNG xuất hiện trong Phase 8 FS validation.
- [ ] Test set KHÔNG xuất hiện trong Phase 8 FS validation.
- [ ] Validation set CHỈ xuất hiện ở Phase 9-14 (scaling, windowing, training, evaluation).
- [ ] Test set CHỈ xuất hiện ở Phase 14 final evaluation (Phase 47 gate).
- [ ] `feature_engineering_train_only_validated: True` in Phase 7 manifest.
- [ ] `train_only_feature_set_validation: True` in Phase 8 manifest.
- [ ] `train_only_eda_validated: True` in Phase 6 manifest.

### 7.9. Documentation

- [ ] `docs/plan-doc/analysis_error/phase_reorder_split_before_eda_train_only_refactor_plan.md` (this file) archived.
- [ ] `docs/plan-doc/plan_before_process/phase_reorder_split_before_eda_train_only_pre_process_plan.md` (post-approval) created.
- [ ] `docs/current_flow/CURRENT_FLOW_SUMMARY.md` updated.
- [ ] EDA documentation (`README_EDA.md` nếu có) updated.

---

## 8. Phụ lục: Impact Analysis

### 8.1. Files cần thay đổi

#### 8.1.1. Source code (4 files)

| File | Thay đổi chính | Loại thay đổi |
|---|---|---|
| `src/course_work/data/eda.py` | Add train filter trong `prepare_eda_analysis` | Add train-only scope |
| `src/course_work/data/features.py` | Add membership param, update FE references | Update phase references + train-only |
| `src/course_work/data/feature_sets.py` | Add train filter trong `validate_feature_variants` | Update phase references + train-only |
| `src/course_work/reporting/phase_summary.py` | Update 5 data structures cho keys 5/6/7/8 | Update phase numbering |

#### 8.1.2. Source code (4 files logic unchanged, only references)

| File | Thay đổi | Loại thay đổi |
|---|---|---|
| `src/course_work/data/splitting.py` | Update signoff path + phase_id | Renumber only |
| `src/course_work/data/scaling.py` | Update upstream signoff references | Update references |
| `src/course_work/data/windows.py` | Update upstream signoff references | Update references |
| `src/course_work/evaluation/metrics.py`, `experiments/registry.py`, `baselines/persistence.py` | (If needed) Update upstream signoff references | Update references |

#### 8.1.3. Notebook (1 file)

| File | Thay đổi | Loại thay đổi |
|---|---|---|
| `notebook_course_work/CourseWork.ipynb` | Reorder cells, update markdown headings, add train-only notes | Reorder + update |

#### 8.1.4. Architecture rule (1 file)

| File | Thay đổi | Loại thay đổi |
|---|---|---|
| `docs/RULE_BASE/architecture_rule.md` | Update §9, §13, §14.3, §7.9.1, §15, §25, §26 | Architecture amendment |

#### 8.1.5. Docs logs (4 files)

| File cũ | File mới | Loại |
|---|---|---|
| `docs/save_log_in_processing/phase_5_eda_log.json` | `docs/save_log_in_processing/phase_6_eda_log.json` | Rename |
| `docs/save_log_in_processing/phase_6_feature_engineering_log.json` | `docs/save_log_in_processing/phase_7_feature_engineering_log.json` | Rename |
| `docs/save_log_in_processing/phase_7_feature_set_variants_log.json` | `docs/save_log_in_processing/phase_8_feature_set_variants_log.json` | Rename |
| `docs/save_log_in_processing/phase_8_chronological_split_log.json` | `docs/save_log_in_processing/phase_5_chronological_split_log.json` | Rename |

#### 8.1.6. Tests (3 directories)

| Directory | Loại thay đổi |
|---|---|
| `tests/contracts/` | Update phase number expectations |
| `tests/integration/` | Update phase transition tests |
| `tests/unit/` | Update unit tests for train-only scope |

### 8.2. Files KHÔNG cần thay đổi

- `src/course_work/data/acquisition.py` (Phase 2)
- `src/course_work/data/schema.py` (Phase 3)
- `src/course_work/data/temporal.py` (Phase 4)
- `src/course_work/utils/environment.py` (Phase 1)
- `src/course_work/utils/reproducibility.py`
- `src/course_work/utils/artifacts.py`
- `src/course_work/contracts/coursework.py` (Phase 0)
- `src/course_work/attention/*` (Phase 15+)
- `src/course_work/__init__.py` (package init)
- `src/course_work/data/__init__.py`
- `src/course_work/reporting/__init__.py`

### 8.3. Tổng kết impact

**Tổng files affected:** 4 source code + 4 references + 1 notebook + 1 architecture rule + 4 logs + 3 test dirs = **17 files/dirs**.

**Tổng lines of code modified (estimated):** ~500-700 lines.

**Effort estimate:** 5-7 days (1-2 days approval + 2-3 days code + 1-2 days notebook + 1-2 days validation).

---

## 9. Khuyến nghị cho Human approval

### 9.1. Refactor này thay đổi Phase 5-8 order

Refactor này thay đổi:
- Phase-to-module mapping (§9) — Phase 5 ↔ 8 swap.
- Notebook boundary cell order (§14.3).
- EDA boundary scope (§15) — added Train-only constraint.
- Transition state (§25) — Phase 5/6/7/8 narrative updates.
- Function semantics — `prepare_eda_analysis` thêm train filter.

→ **Cần architecture amendment + Human approval theo §24 change-control gate.**

### 9.2. Plan này là pre-process plan

Plan này KHÔNG tự động thực thi. Sau Human approval, mới tiến hành execute theo roadmap R1-R5.

### 9.3. Activation gate

Sau khi Human approve plan này:
- Set `ARCHITECTURE_RULE_APPROVED=true` (already true).
- Set `SOURCE_REFACTOR_ALLOWED=true` for this specific refactor scope.
- Begin Phase R2 (Source code changes).

Sau khi refactor complete và pass verification checklist:
- Update `CURRENT_FLOW_SUMMARY.md` to reflect new order.
- Archive this plan to `docs/plan-doc/plan_before_process/phase_reorder_split_before_eda_train_only_pre_process_plan.md`.

---

## 10. Câu hỏi cần Human quyết định

Trước khi phê duyệt plan, vui lòng trả lời các câu hỏi sau:

### Câu hỏi 1: Function name `materialize_phase_8` trong `splitting.py`

**Vấn đề:** Function `materialize_phase_8` đang ở `splitting.py` (Phase 8). Sau refactor, Phase 5 mới = Split, nhưng giữ function name.

**Option A: Giữ tên function cũ** `materialize_phase_8` (Recommend).
- Ưu điểm: Không phải đổi ownership reference ở nhiều file.
- Nhược điểm: Tên function không khớp phase_id mới.

**Option B: Rename function thành `materialize_phase_5_chronological_split`**.
- Ưu điểm: Rõ ràng cho developers.
- Nhược điểm: Phải update tất cả callsite (splitting.py, scaling.py, windows.py, notebook, etc.).

**Recommendation:** Option A (giữ tên function cũ).

---

### Câu hỏi 2: Có nên xóa hẳn các signoff artifacts cũ?

**Vấn đề:** Sau refactor, signoff files cũ (Phase 5 EDA = `phase_5_signoff.json`, Phase 6 FE = `phase_6_signoff.json`, Phase 7 FS = `phase_7_signoff.json`, Phase 8 Split = `phase_8_signoff.json`) có phase_id cũ. Nếu giữ lại, conflict với new signoffs.

**Option A: Delete + re-materialize (Recommend).**
- Ưu điểm: Clean state, no confusion.
- Nhược điểm: Mất lịch sử Phase 5 EDA cũ (nhưng có logs nếu cần).

**Option B: Archive + delete.**
- Ưu điểm: Giữ lịch sử.
- Nhược điểm: Phức tạp hơn.

**Recommendation:** Option A.

---

### Câu hỏi 3: Có nên giữ lại EDA summary trên Validation để debug drift?

**Vấn đề:** Hiện tại EDA không có Validation summary. Nếu thêm, có thể debug drift nhưng vi phạm firewall.

**Option A: KHÔNG thêm Validation summary (Recommend).**
- Ưu điểm: Strict Train-only scope.
- Nhược điểm: Khó debug drift.

**Option B: Thêm Validation summary ở Phase 9+ (Scaling).**
- Ưu điểm: Có diagnostics.
- Nhược điểm: Cần thêm logic ở Phase 9.

**Recommendation:** Option A. Drift diagnostic đã có ở Phase 9 (`validation_shift_diagnostic`).

---

### Câu hỏi 4: Có nên update tests song song hay để sau?

**Vấn đề:** Tests reference phase numbers cũ. Nếu update source code trước, tests sẽ fail.

**Option A: Update tests song song (Recommend).**
- Ưu điểm: Verify ngay từng step.
- Nhược điểm: Effort song song.

**Option B: Update tests sau.**
- Ưu điểm: Tách biệt concerns.
- Nhược điểm: Risk source code không match tests.

**Recommendation:** Option A.

---

### Câu hỏi 5: Có nên giữ `feature_engineered_v1.csv` ở full 19,735 rows?

**Vấn đề:** Sau refactor, validate feature invariants chỉ trên TRAIN. Nhưng derived CSV vẫn chứa full rows (để giữ contract với downstream).

**Option A: Giữ full 19,735 rows (Recommend).**
- Ưu điểm: Downstream phases (FS, scaling, windows) vẫn dùng full timeline.
- Nhược điểm: Possible confusion "Why is full CSV but invariant on TRAIN?".

**Option B: Filter CSV ở TRAIN subset.**
- Ưu điểm: Cleaner.
- Nhược điểm: Phase 8 (FS) train-only validation OK, nhưng scaling/windowing cần full timeline.

**Recommendation:** Option A. Document rõ "FEATURES-v1 CSV = full timeline, invariant/validation scope = TRAIN".

---

### Câu hỏi 6: Có nên backup current plan `phase_14_persistence_baseline_pre_process_plan.md` không?

**Vấn đề:** Sau refactor, các plan files đã duyệt cho Phase 5-14 cũ cần archive.

**Option A: Archive old plans + create new plans (Recommend).**
- Ưu điểm: Trace history.
- Nhược điểm: Effort.

**Option B: Reuse existing plans + note version.**
- Ưu điểm: Less effort.
- Nhược điểm: Plans có thể cũ.

**Recommendation:** Option A sau khi Human approve.

---

### Câu hỏi 7: Phase 5 (Split) dependencies có thay đổi không?

**Vấn đề:** Hiện tại Phase 8 (Split) dependencies: Phase 6 (FE), Phase 7 (FS). Sau refactor, Phase 5 (Split) dependencies: chỉ cần Phase 4 (Temporal) + Phase 0 (Contract).

**Option A: Phase 5 chỉ depend on Phase 4 + Phase 0 (Recommend).**
- Ưu điểm: Cleaner dependency graph.
- Nhược điểm: Phải update `verify_phase_8_inputs` thành `verify_phase_5_inputs` (or keep function name).

**Option B: Phase 5 vẫn depend on Phase 6 (FE) + Phase 7 (FS) — circular?**
- Nhược điểm: Circular dependency.

**Recommendation:** Option A.

---

### Câu hỏi 8: Có nên chuyển DRY check cho `materialize_phase_X` functions?

**Vấn đề:** Hiện tại mỗi phase có `materialize_phase_X` riêng. Có thể refactor DRY hơn (parameterize phase_id).

**Option A: Giữ separate functions (Recommend).**
- Ưu điểm: Clear ownership, dễ debug.
- Nhược điểm: Code duplication small.

**Option B: Generic factory.**
- Ưu điểm: DRY.
- Nhược điểm: Phức tạp hơn.

**Recommendation:** Option A. Refactor này chỉ reorder, KHÔNG thay đổi architecture pattern.

---

### Câu hỏi 9: Notebook cell naming convention?

**Vấn đề:** Cells có thể có metadata `name` hoặc không.

**Option A: KHÔNG thêm metadata name (Recommend).**
- Ưu điểm: Notebook hiện tại có thể không dùng metadata name.

**Option B: Thêm metadata name để dễ debug.**
- Ưu điểm: Trace được cell theo name.

**Recommendation:** Option A. Tương thích với notebook hiện tại.

---

### Câu hỏi 10: Sau refactor, có cần reset Phase 14 PERSISTENCE_BASELINE run không?

**Vấn đề:** Phase 13 đã register 1 PERSISTENCE_BASELINE run. Sau refactor, config_fingerprint có thể thay đổi (vì upstream signoff changes).

**Option A: Đăng ký run mới (Recommend).**
- Ưu điểm: Clean state.

**Option B: Reuse run cũ.**
- Nhược điểm: Config fingerprint mismatch.

**Recommendation:** Option A. Re-run Phase 14 sau refactor để có PERSISTENCE_BASELINE run với new upstream references.

---

## Phụ lục A: Glossary

| Term | Definition |
|---|---|
| EDA | Exploratory Data Analysis |
| FE | Feature Engineering |
| FS | Feature-Set Variants |
| Split | Chronological Split (deterministic 70/15/15) |
| Train-only | Scope restricted to TRAIN rows only |
| Test firewall | Rule preventing Test data leakage to scientific decisions |
| Phase gate | Dependency rule: Phase N chỉ chạy khi Phase N-1 PASS |
| Signoff | Phase completion evidence artifact |
| Architecture amendment | Change to architecture rule requiring Human approval |
| Pre-process plan | Plan describing proposed changes before execution |

---

## Phụ lục B: Cross-references

- Phase 0 contract: `configs/base/coursework_contract.json`
- Phase 1-4 (unchanged): raw data pipeline
- Phase 5 (mới = Split): `src/course_work/data/splitting.py::materialize_phase_8` (function name kept)
- Phase 6 (mới = EDA Train-Only): `src/course_work/data/eda.py::prepare_eda_analysis` + `src/course_work/reporting/eda.py::materialize_phase_5`
- Phase 7 (mới = FE Train-Only): `src/course_work/data/features.py::materialize_phase_6` (function name needs update)
- Phase 8 (mới = FS Train-Only): `src/course_work/data/feature_sets.py::materialize_phase_7` (function name needs update)
- Phase 9-14 (unchanged): existing pipeline

---

## Phụ lục C: Decision log template

Sau khi Human approve, ghi vào `docs/plan-doc/decision_log.md`:

```markdown
## 2026-08-17: Phase reorder refactor (Split before EDA, Train-only scope)

**Decision:** Approve refactor moving Phase 8 (Chronological Split) to Phase 5, and restricting Phase 6/7/8 (EDA/FE/FS) to Train-only scope.

**Approved by:** Human (name, date)

**Effective:** YYYY-MM-DD (after Phase R1)

**Consequences:**
- Phase 5/6/7/8 numbering changes
- EDA Train-only scope eliminates hypothesis stage leakage
- Notebook boundary cell order updated
- Architecture rule §9, §13, §14.3, §7.9.1, §15, §25, §26 amended
```

---

**End of plan. Awaiting Human approval.**
