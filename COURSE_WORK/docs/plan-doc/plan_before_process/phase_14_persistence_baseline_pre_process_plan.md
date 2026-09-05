# Pre-process Plan Phase 14 - Persistence Baseline

## 1. Trạng thái kế hoạch

| Thuộc tính | Giá trị |
|---|---|
| Plan ID | PHASE-14-PERSISTENCE-PREPROCESS-v1 |
| Ngày lập | 2026-08-16 |
| Trạng thái | IMPLEMENTED_AND_VERIFIED |
| Phase mục tiêu | Phase 14 - Persistence Baseline |
| Baseline version | PERSISTENCE-v1 |
| Model ID | PERSISTENCE_LAST_VALUE |
| Execution type | EVALUATION |
| Experiment family | PERSISTENCE_BASELINE |
| Split được phép | VALIDATION |
| Test access | FORBIDDEN_UNTIL_PHASE_47 |

Kế hoạch này chỉ mô tả cách triển khai. Chưa được tạo source code, sửa notebook, sửa test, sửa architecture rule, tạo run hoặc sinh artifact Phase 14 trước khi Human duyệt.

## 2. Understanding đã khóa

### 2.1 What

Xây dựng một Persistence baseline cho bài toán multivariate time-series regression, trong đó dự báo điện năng ở bước kế tiếp bằng giá trị `Appliances` quan sát gần nhất tại cuối cửa sổ đầu vào:

`y_hat(t+1) = y(t)`

Baseline chỉ được đánh giá trên tập Validation chung đã khóa bởi `WINDOWPOP-v1`, dùng horizon một bước tương đương 10 phút và dùng đúng metric contract `METRICS-v1`.

### 2.2 Who

- Người thực thi: source module Phase 14 và các owner upstream đã tồn tại.
- Người sử dụng: notebook reader, người đánh giá coursework và các phase model về sau.
- Người phê duyệt: Human sở hữu coursework.

### 2.3 Goal

Tạo một mốc tham chiếu đơn giản, xác định, không học tham số và có thể audit để trả lời liệu LSTM hoặc Transformer về sau có cải thiện so với việc lặp lại giá trị điện năng gần nhất hay không.

## 3. Phạm vi đã khóa

### 3.1 Trong phạm vi

- Triển khai đúng một production module mới cho Persistence baseline.
- Đọc metadata cửa sổ từ `WINDOWS-v1` và population từ `WINDOWPOP-v1`.
- Chỉ lấy các mẫu `VALIDATION`, lookback neo population là L144, protocol là WB0 và horizon H1.
- Lấy `y_pred_wh` từ `Appliances` tại `input_end_raw_row_index`.
- Lấy `y_true_wh` từ `Appliances` tại `target_raw_row_index`.
- Kiểm tra quan hệ source-target đúng một bước và đúng 10 phút.
- Dùng `METRICS-v1` để tính MAE, RMSE, R² và residual theo đơn vị Wh.
- Đăng ký run trong `EXPERIMENTS-v1` trước khi thực hiện đánh giá Validation.
- Sinh artifact, checksum, audit, sign-off và processing log theo kiến trúc hiện hành.
- Notebook chỉ import và gọi public API; không chứa logic xử lý Phase 14.
- Bổ sung test unit, test registry, test presentation và test integration liên quan.
- Cập nhật architecture rule để phạm vi chính thức chuyển từ Phase 0–13 sang Phase 0–14.

### 3.2 Ngoài phạm vi

- Không train model.
- Không dùng optimizer, loss, scheduler, early stopping, checkpoint hoặc gradient clipping.
- Không dùng GPU, MPS hoặc CUDA cho baseline evaluator.
- Không dùng DataLoader để tính prediction.
- Không dùng feature variant FS0, FS1, FS2, TF0 hoặc TF1 để tạo prediction.
- Không fit, load hoặc áp dụng X scaler hay Y scaler.
- Không dùng seed hoặc tạo multi-seed run giả.
- Không đánh giá Train để lựa chọn baseline.
- Không đọc, tính, hiển thị hoặc đăng ký metric/prediction của Test.
- Không thêm moving-average, seasonal naive hoặc baseline khác.
- Không refactor các phase trước ngoài các owner boundary được liệt kê trong kế hoạch.
- Không thêm dependency mới.
- Không đổi formula Persistence theo kết quả Validation.

## 4. Input contract

| Input | Vai trò | Điều kiện bắt buộc trước khi dùng |
|---|---|---|
| `data/raw_data/energydata_complete.csv` | Nguồn `Appliances` gốc | SHA-256 phải là `2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d` |
| `artifacts/windows/phase_10_signoff.json` | Sign-off cửa sổ | `status=PASS`, `WINDOWS-v1` |
| `artifacts/windows/window_manifest.json` | Horizon, cadence, protocol và population contract | H1, 10 phút, WB0, Test locked |
| `artifacts/windows/window_index.csv` | Source/target indices và timestamps | Schema đúng, chronology đúng, fingerprint đúng |
| `artifacts/windows/common_target_population.csv` | Population chung | `WINDOWPOP-v1`, không duplicate, thứ tự thời gian tăng dần |
| `artifacts/dataloaders/phase_11_signoff.json` | Upstream lineage | `status=PASS`, `DATALOADERS-v1` |
| `artifacts/metrics/phase_12_signoff.json` | Metric contract | `status=PASS`, `METRICS-v1` |
| `artifacts/metrics/metric_manifest.json` | Metric names, unit và Test firewall | MAE/RMSE/R², Wh, Validation primary |
| `artifacts/experiments/phase_13_signoff.json` | Registry contract | `status=PASS`, `EXPERIMENTS-v1` |
| `artifacts/experiments/experiment_registry.jsonl` | Production run registry | Trước Phase 14 đang rỗng và hợp lệ |

## 5. Output contract

### 5.1 Production source

Tạo mới:

- `src/course_work/baselines/persistence.py`

Module này là owner duy nhất của toàn bộ logic Phase 14, bao gồm contract, input validation, Persistence prediction, H1 audit, population alignment, metric integration, registry lifecycle, artifact materialization, idempotent verification và `materialize_phase_14`.

Không tách thêm `validation.py` trong Phase 14 để tuân thủ yêu cầu chỉ tạo một file `.py` riêng cho phase này. Các validation function vẫn phải được phân tách rõ theo trách nhiệm bên trong `persistence.py`.

### 5.2 Canonical artifacts

Tạo dưới `artifacts/baselines/persistence/`:

| Artifact | Nội dung bắt buộc |
|---|---|
| `persistence_manifest.json` | Version, formula, lineage, fingerprints, run ID, artifact paths, checksums và status |
| `persistence_baseline_summary.json` | Baseline contract và kết quả Validation tối thiểu |
| `persistence_validation_predictions.csv` | Prediction-level evidence theo schema đã khóa |
| `persistence_validation_metrics.json` | MAE, RMSE, R², R² status, sample count, unit và metric fingerprint |
| `persistence_audit.csv` | Audit công thức, horizon, chronology, population, unit, invariance và firewall |
| `persistence_unit_tests.csv` | Kết quả các kiểm thử synthetic và contract |
| `persistence_discrepancies.json` | Warnings và discrepancies có cấu trúc |
| `README_PERSISTENCE.md` | Mô tả artifact, formula, phạm vi và cách tái kiểm tra |
| `phase_14_signoff.json` | Sign-off cuối cùng sau khi run đã hoàn tất |

Registry được phép tạo `artifacts/runs/<run_id>/config.json` và `artifacts/runs/<run_id>/status.json` theo owner Phase 13. Không nhân bản prediction hoặc metric sang run directory nếu registry có thể đăng ký trực tiếp canonical artifact bằng path và checksum.

### 5.3 Prediction CSV schema

Giữ đúng thứ tự cột:

1. `run_id`
2. `sample_idx`
3. `window_id`
4. `target_timestamp`
5. `source_timestamp`
6. `source_raw_row_index`
7. `target_raw_row_index`
8. `y_true_wh`
9. `y_pred_wh`
10. `residual_wh`
11. `absolute_error_wh`
12. `squared_error_wh`

Không làm tròn số trước khi ghi artifact. `residual_wh` phải tuân thủ `y_true_wh - y_pred_wh`.

### 5.4 Processing log

Tạo:

- `docs/save_log_in_processing/phase_14_persistence_baseline_log.json`

Log lưu đầy đủ lineage, checksums, run metadata, metric, audit, discrepancies và technical details. Notebook không hiển thị toàn bộ log này.

## 6. Kết quả kiểm tra nền trước khi lập plan

Kiểm tra được thực hiện ở chế độ không triển khai Phase 14.

| Hạng mục | Kết quả hiện tại |
|---|---|
| Phase 10 sign-off | PASS |
| Phase 11 sign-off | PASS |
| Phase 12 sign-off | PASS |
| Phase 13 sign-off | PASS |
| Window version | WINDOWS-v1 |
| Population version | WINDOWPOP-v1 |
| Common population fingerprint | `a40ded8802e90008535d268720bad9e9dcca5eee1436ddb359deea3ad39a1987` |
| Primary lookback anchor | 144 bước |
| Forecast horizon | 1 bước, 10 phút |
| Boundary protocol | WB0_CONTEXT_CARRY_OVER |
| Validation population | 2.960 mẫu |
| Test access policy | LOCKED_UNTIL_PHASE_47 |
| Test target values exported by Phase 10 | False |
| Metric version | METRICS-v1 |
| Required metrics | MAE, RMSE, R² |
| Metric target unit | Wh |
| Metric contract fingerprint | `4509825a7be2ee75220f88bedf31da6d062e6fee5168af45b0d54ec1db137198` |
| Experiment version | EXPERIMENTS-v1 |
| Registry contract fingerprint | `124a2edf1dc8a30845f7ffa7e0c46bf8d05a8c7ebe916b539f711e8d9dff36df` |
| Production registry run count | 0 |
| Existing test suite | 152 passed |
| Source/artifact Phase 14 | Chưa tồn tại |

## 7. Phân tích khoảng trống hiện tại

### 7.1 Chưa có owner Phase 14

`src/course_work/baselines/persistence.py` chưa tồn tại. Chưa có public API `materialize_phase_14`, chưa có baseline artifact và chưa có test Phase 14.

### 7.2 Registry hiện chưa biểu diễn đúng task-level Persistence

`build_reference_run_config` hiện gắn Persistence với baseline neural defaults như `FS1_TF1`, `YS1`, feature fingerprint, scaler bundle, dataloader fingerprint và seed. `validate_run_config` cũng yêu cầu các trường này có giá trị.

Điều đó xung đột trực tiếp với Phase 14 contract:

- `feature_variant_id = null`
- `baseline_scope = TASK_LEVEL`
- `scaler_bundle_id = null`
- `target_scaling_option = null` ở run config
- `dataloader_config_id = null`
- `seed = null`
- `trainable_parameters = 0`
- `device_type = cpu`

Không được giải quyết bằng cách khai giả Persistence là `FS1_TF1` hoặc `YS1`. Cần mở rộng tối thiểu đúng owner `experiments/registry.py` với nhánh validation riêng cho model family `PERSISTENCE`, đồng thời giữ nguyên hành vi LSTM và Transformer.

### 7.3 Required artifacts của EVALUATION chưa đủ chặt cho Persistence

Registry hiện yêu cầu `METRICS` nhưng chưa bắt buộc `PREDICTIONS` đối với execution type `EVALUATION`. Phase 14 yêu cầu cả prediction và metric phải được đăng ký trước khi complete run.

Thay đổi phải giới hạn theo record/model family `PERSISTENCE`; không được âm thầm đổi contract của các evaluation family khác.

### 7.4 Metric API và run config có semantic khác nhau

`PredictionBundle` của `METRICS-v1` yêu cầu `target_scaling_option` là `YS0` hoặc `YS1`. Persistence không dùng scaler, nhưng prediction đã ở raw Wh.

Quy ước cần khóa:

- Run config của Persistence: `target_scaling_option = null` vì scaler không áp dụng.
- Lời gọi metric API: dùng `YS0` chỉ để biểu diễn identity path cho dữ liệu đã ở Wh.
- Không load hoặc áp dụng bất kỳ scaler artifact nào.
- Test phải chứng minh không có scaler dependency.

### 7.5 Test firewall cần chặt từ tầng đọc dữ liệu

Không chỉ cấm ghi Test metric. Phase 14 phải tránh đọc Test target values vào computation path.

Kế hoạch đọc dữ liệu:

- Đầu tiên lọc metadata `window_index` xuống đúng L144, WB0, common population và `VALIDATION`.
- Xác định raw-row range tối đa cần thiết từ Validation subset.
- Chỉ đọc hai cột `date` và `Appliances` tới raw row cuối cùng cần cho Validation.
- Không đọc các raw row thuộc Test target period.
- Hard fail nếu bất kỳ selected row nào mang `target_split_id=TEST` hoặc vượt Validation boundary.

### 7.6 Notebook và presentation chưa biết Phase 14

Notebook hiện kết thúc ở Phase 13. `phase_summary.py`, notebook boundary tests và architecture rule cũng mới khóa Phase 0–13.

## 8. Thiết kế source API

Production module mới phải cung cấp các trách nhiệm sau, tên cụ thể được khóa để test và notebook dùng ổn định:

| API/khối trách nhiệm | Vai trò |
|---|---|
| `PersistenceConfig` | Khóa version, formula, horizon, lag, split, scope và Test policy |
| `predict_persistence` | Pure function trả prediction từ source indices và raw Appliances values |
| `validate_persistence_inputs` | Kiểm tra schema, finite values, uniqueness và bounds |
| `validate_h1_alignment` | Kiểm tra source là target trừ một bước, delta 10 phút và cùng continuity segment |
| `validate_population` | Kiểm tra đúng toàn bộ Validation population, đúng order và fingerprint |
| `validate_persistence_predictions` | Kiểm tra prediction bằng chính raw source value và đúng Wh |
| `build_persistence_prediction_bundle` | Tạo dữ liệu tương thích `METRICS-v1` với identity mode YS0 |
| `verify_phase_14_inputs` | Xác minh sign-off, version, checksum và Test lock upstream |
| `verify_existing_signoff` | Bảo đảm rerun idempotent và artifact không bị sửa |
| `materialize_phase_14` | Public orchestration API duy nhất cho notebook |

Không kế thừa `nn.Module`, không khai báo model parameters và không tạo checkpoint.

## 9. Data flow đã khóa

1. Xác minh checksum và sign-off Phase 10–13.
2. Đọc `WINDOWPOP-v1` và L144/WB0 window metadata.
3. Lọc `target_split_id=VALIDATION` trước khi lấy target values.
4. Giữ nguyên index gốc của `window_index.csv` làm `sample_idx`; không reset index.
5. Cross-check `sample_idx` với `expected_sample_indices("VALIDATION", lookback_steps=144)`.
6. Cross-check `target_sample_id` với common Validation population.
7. Đọc bounded raw prefix chỉ tới Validation row cuối cần thiết.
8. Lấy `y_pred_wh` tại `input_end_raw_row_index`.
9. Lấy `y_true_wh` tại `target_raw_row_index`.
10. Kiểm tra `timeline_input_end = timeline_target - 1`.
11. Kiểm tra `source_raw_row_index = target_raw_row_index - 1`.
12. Kiểm tra `target_timestamp - source_timestamp = 10 phút`.
13. Kiểm tra source và target thuộc cùng continuity segment.
14. Tạo prediction artifact và residual arrays.
15. Gọi `compute_regression_metrics` đúng một lần trên toàn bộ Validation split.
16. Đăng ký artifacts và ba metric rows.
17. Complete run chỉ sau khi tất cả required evidence đã PASS.
18. Ghi manifest và sign-off cuối cùng.

## 10. Registry lifecycle

### 10.1 Run config Persistence

Run config phải giữ project-wide lineage version/fingerprint nhưng loại bỏ dependency giả ở model-specific fields.

| Nhóm | Giá trị bắt buộc |
|---|---|
| Model | `PERSISTENCE`, `PERSISTENCE_LAST_VALUE`, `PERSISTENCE-v1` |
| Scope | `TASK_LEVEL` |
| Population anchor | L144, `WINDOWPOP-v1` |
| Horizon | 1 bước, 10 phút |
| Source lag | 1 bước, 10 phút |
| Split | VALIDATION |
| Feature variant | null |
| Feature count/fingerprint | null ở model-specific lineage |
| Target scaling | null trong run config |
| Scaler bundle/checksum | null |
| DataLoader fingerprint/config | null |
| Training enabled | false |
| Training fields | null |
| Seed fields | null |
| Deterministic | true |
| Device | cpu |
| Trainable parameters | 0 |
| Test access authorized | false |

### 10.2 Thứ tự lifecycle

1. Xây dựng và validate canonical config.
2. `register_run` với family `PERSISTENCE_BASELINE`, type `EVALUATION` và Test authorization false.
3. `start_run`.
4. Thực hiện Validation prediction và metric.
5. Ghi canonical artifacts bằng atomic write.
6. Đăng ký `PREDICTIONS` và `METRICS` là required artifacts.
7. Đăng ký audit, tables và supporting artifacts theo đúng artifact type.
8. Đăng ký đúng ba metric rows `mae_wh`, `rmse_wh`, `r2` cho `VALIDATION`.
9. Gọi `complete_run` với Validation RMSE đã đăng ký.
10. Ghi manifest và Phase 14 sign-off sau khi registry record là `COMPLETED`.

### 10.3 Failure lifecycle

- Nếu lỗi xảy ra sau `start_run`, phải gọi `fail_run` và giữ lại traceability.
- Lỗi horizon, alignment, population/window mapping dùng `WINDOW_CONTRACT_ERROR`.
- Lỗi metric hoặc metric population dùng `METRIC_ERROR`.
- Lỗi truy cập Test dùng `TEST_FIREWALL_VIOLATION`.
- Lỗi không thuộc taxonomy trên dùng `OTHER` và phải có failure stage rõ ràng.
- Không tự xóa failed run.
- Không tự tạo duplicate rerun trong cùng lượt.
- Sau failure phải dừng, lập issue analysis và fix plan; lần chạy lại sau code fix dùng canonical rerun reason `CODE_FIX`.

### 10.4 Idempotency

- Nếu `phase_14_signoff.json` đã tồn tại, `materialize_phase_14` chỉ verify version, input checksums, output checksums, run status, metric fingerprint và registry references rồi trả sign-off.
- Không tạo run mới khi canonical sign-off hiện hữu và hợp lệ.
- Nếu artifact hiện hữu khác bytes mong đợi, hard fail; không overwrite âm thầm.

## 11. Notebook và presentation contract

### 11.1 Notebook boundary

`CourseWork.ipynb` chỉ được thay đổi ở các boundary sau:

- Import `materialize_phase_14` trong import cell hiện hữu.
- Thêm markdown title Phase 14.
- Thêm đúng một orchestration cell gọi `materialize_phase_14(PROJECT_ROOT)` và `render_phase_summary(14, PROJECT_ROOT)`.
- Đổi boundary marker từ Phase 1–13 thành Phase 1–14.

Toàn bộ 22 logical operations được nêu trong Phase 14 detail phải nằm trong public source API. Không chuyển 22 operations thành 22 code cells vì điều đó vi phạm architecture rule và yêu cầu notebook chỉ call lại.

### 11.2 Visible output allowlist

Notebook chỉ hiển thị phase header và hai section:

| Section | Field được phép hiển thị |
|---|---|
| Validation performance | MAE (Wh), RMSE (Wh), R², Validation samples |
| Baseline contract | Formula, horizon, task-level scope, population version, source lag, Test status |

Không hiển thị raw JSON, checksums, full run config, technical details, warnings/discrepancies table hoặc full prediction rows. Những nội dung này chỉ lưu trong processing log và artifact.

## 12. File impact matrix

### 12.1 File tạo mới khi được duyệt

| File | Thay đổi dự kiến | Lý do |
|---|---|---|
| `src/course_work/baselines/persistence.py` | Toàn bộ implementation Phase 14 | Giữ notebook orchestration-only và đúng source owner |
| `tests/unit/test_persistence.py` | Unit/contract tests Phase 14 | Khóa formula, alignment, population, firewall và invariance |
| `artifacts/baselines/persistence/*` | Chín canonical artifacts | Evidence và reproducibility |
| `artifacts/runs/<run_id>/*` | Config/status do registry quản lý | Traceability của run |
| `docs/save_log_in_processing/phase_14_persistence_baseline_log.json` | Full processing log | Tách technical output khỏi notebook |

### 12.2 File sửa khi được duyệt

| File | Thay đổi tối thiểu | Không được thay đổi |
|---|---|---|
| `src/course_work/experiments/registry.py` | Thêm canonical Persistence config/validation và required predictions cho Persistence evaluation | Neural config validation, status model, Test firewall và run ID policy |
| `src/course_work/reporting/phase_summary.py` | Thêm Phase 14 sources, log name, content và visible allowlist | Style/CSS và output contract của Phase 1–13 |
| `tests/unit/test_experiment_registry.py` | Test null/non-applicable Persistence fields và completion guard | Existing LSTM/Transformer tests |
| `tests/unit/test_phase_summary.py` | Test Phase 14 processing log và visible output | Existing presentation allowlists |
| `tests/integration/test_phase_0_to_5_chain.py` | Mở chain đến Phase 14 và kiểm tra artifacts | Existing Phase 0–13 assertions |
| `tests/integration/test_notebook_boundary.py` | Mở notebook boundary đến Phase 14 | Phase 5 EDA exception và Phase 1–13 boundaries |
| `notebook_course_work/CourseWork.ipynb` | Import, title, một orchestration cell, boundary marker | Nội dung Phase 1–13 và EDA cells |
| `docs/RULE_BASE/architecture_rule.md` | Thêm Phase 14 owner, artifacts, tests, notebook và presentation contract | Các rule hiện hành và mapping Phase 0–13 |

### 12.3 File không được sửa

- `data/raw_data/energydata_complete.csv`
- Toàn bộ signed artifacts Phase 0–13, trừ registry files được Phase 13 owner cập nhật hợp lệ khi đăng ký run Phase 14.
- Các source module Phase 0–12.
- `requirements.txt` và `pyproject.toml`.
- Các phase detail documents.
- Các config sweep, adaptation và final.
- Các notebook cell ngoài boundary Phase 14 được nêu trên.

## 13. Kế hoạch thực thi tuần tự sau khi được duyệt

Mỗi bước phải hoàn thành và verify ngay trước khi sang bước tiếp theo. Nếu một check fail, dừng flow, tạo issue analysis và fix plan; không tiếp tục bằng workaround.

### Bước 0 - Chụp baseline bất biến

Thực hiện:

- Ghi nhận `git status` hiện tại để bảo toàn user changes.
- Ghi nhận checksum raw data, Phase 10–13 sign-offs, window population và notebook.
- Xác nhận production registry đang hợp lệ.
- Chạy test suite nền ở chế độ không tạo pytest cache.

Verify:

- Raw SHA giữ nguyên.
- Phase 10–13 PASS.
- 152 test hiện tại vẫn pass trước implementation.
- Không có Phase 14 artifact/run từ trước.

### Bước 1 - Mở rộng registry đúng Persistence contract

Thực hiện:

- Thêm Persistence-specific branch trong reference config builder.
- Cho phép null ở các field không áp dụng chỉ khi model family là `PERSISTENCE`.
- Hard fail nếu Persistence mang feature/scaler/seed/training dependency.
- Hard fail nếu Persistence không dùng CPU, H1, Validation hoặc task-level scope.
- Yêu cầu `PREDICTIONS` và `METRICS` trước khi complete Persistence evaluation.

Verify ngay:

- Existing registry unit tests pass.
- Persistence canonical config pass.
- Persistence với feature/scaler/seed giả fail.
- LSTM/Transformer với invalid fields vẫn fail như cũ.
- Persistence run thiếu predictions không complete được.
- Test registration vẫn bị firewall chặn.

### Bước 2 - Xây dựng pure Persistence core

Thực hiện:

- Tạo `persistence.py` với config và pure prediction/validation functions.
- Dùng NumPy/Pandas đã có; không thêm dependency.
- Không thêm comments hoặc icons.
- Không dùng torch model, scaler hoặc dataloader.

Verify ngay:

- Constant series trả đúng constant prediction.
- Increasing, decreasing và varying series đều lấy đúng previous value.
- Wrong lag, out-of-range, duplicate, NaN/Inf và wrong length hard fail.
- Output là vector một chiều, finite và giữ full precision.

### Bước 3 - Tích hợp population và H1 metadata

Thực hiện:

- Load validated window index và common population bằng owner Phase 10.
- Lọc Validation trước khi truy cập raw target.
- Bảo toàn original DataFrame index làm `sample_idx`.
- Đọc raw prefix bị giới hạn ở Validation boundary.
- Xây prediction rows theo schema đã khóa.

Verify ngay:

- Có đúng 2.960 mẫu.
- Sample IDs unique và đúng order.
- Population fingerprint trùng `WINDOWPOP-v1`.
- Source index bằng target index trừ một.
- Source-target timestamp delta đúng 10 phút.
- Cùng continuity segment.
- Không có Test row hoặc Test target value trong computation/output.

### Bước 4 - Đăng ký và bắt đầu run

Thực hiện:

- Build canonical Persistence config.
- Register run trước evaluation.
- Start run.

Verify ngay:

- Family là `PERSISTENCE_BASELINE`.
- Execution type là `EVALUATION`.
- Status chuyển `REGISTERED` sang `RUNNING` hợp lệ.
- Config fingerprint ổn định.
- Config và status checksums đã được registry quản lý.
- Test authorization là false.

### Bước 5 - Tính Validation metric và audit

Thực hiện:

- Tạo PredictionBundle bằng identity path `YS0` chỉ tại metric API boundary.
- Gọi `compute_regression_metrics` trên full Validation population.
- Tính residual arrays bằng owner `METRICS-v1`.
- Chạy audit và synthetic contract checks.

Verify ngay:

- MAE, RMSE và R² khớp independent reference calculation.
- MAE/RMSE finite và không âm.
- R² giữ nguyên giá trị âm nếu có; không ép finite fallback.
- Unit là Wh.
- Residual convention đúng.
- Không có sample weighting hoặc batch-mean aggregation.

### Bước 6 - Ghi và đăng ký artifacts

Thực hiện:

- Ghi canonical outputs bằng atomic write hoặc write-once-or-verify.
- Tính baseline config fingerprint, prediction fingerprint và metric result fingerprint.
- Register required predictions, metrics và supporting artifacts.
- Register đúng ba Validation metric rows.

Verify ngay:

- CSV schema và row count đúng.
- Không duplicate metric/artifact registrations.
- Checksum trong registry trùng file thật.
- Không có Test record.
- Không có scaler/model/checkpoint artifact.

### Bước 7 - Complete run và sign-off

Thực hiện:

- Complete run sau khi required artifacts và metrics đủ.
- Ghi manifest, summary, README, discrepancies và sign-off cuối.

Verify ngay:

- Run status là `COMPLETED`.
- Exactly one completed canonical Persistence run.
- `best_validation_rmse_wh` bằng metric row đã đăng ký.
- Manifest tham chiếu đúng run ID và artifact checksums.
- `phase_14_signoff.json` là artifact cuối cùng và `PASS`.
- Gọi lại `materialize_phase_14` không tạo run hoặc thay đổi bytes.

### Bước 8 - Presentation và processing log

Thực hiện:

- Thêm Phase 14 source spec, presentation spec và log file mapping.
- Render đúng visible allowlist.
- Lưu full processing log.

Verify ngay:

- Notebook-visible HTML chỉ có hai section đã duyệt.
- Không có raw JSON, technical details hoặc full prediction rows.
- Log chứa đủ provenance và checksum.
- Phase 1–13 render output không đổi.

### Bước 9 - Notebook orchestration boundary

Thực hiện:

- Thêm import, title, một orchestration cell và boundary marker Phase 1–14.

Verify ngay:

- Notebook parse được bằng nbformat.
- Phase 14 code cell chỉ gọi public APIs.
- Không có data processing, metric calculation, registry mutation logic hoặc artifact write logic trực tiếp trong notebook.
- Không xuất hiện `materialize_phase_0` trở lại notebook.
- Existing Phase 1–13 cells giữ nguyên ngoài boundary update được duyệt.

### Bước 10 - Architecture rule

Thực hiện:

- Bổ sung owner map cho baseline module.
- Bổ sung artifact directory và registry lifecycle Phase 14.
- Bổ sung notebook/presentation allowlist và test ownership.
- Chuyển current phase boundary thành Phase 0–14.

Verify ngay:

- Documentation mapping khớp filesystem thực tế.
- Không tuyên bố file/artifact chưa tồn tại sau implementation.
- Không làm sai hoặc xóa rule Phase 0–13.

### Bước 11 - Full regression và acceptance audit

Thực hiện tuần tự:

- Chạy targeted Phase 14 tests.
- Chạy registry/metrics/window/presentation tests.
- Chạy integration chain Phase 0–14.
- Chạy notebook boundary tests.
- Chạy toàn bộ test suite.
- Chạy source syntax/import validation.
- Kiểm tra artifact checksums và registry consistency.
- Kiểm tra diff whitespace và phạm vi file thay đổi.

Verify cuối:

- Không test failure.
- Không notebook error output.
- Không warning do implementation Phase 14.
- Không comments hoặc icons mới trong source/notebook Phase 14.
- Raw data và signed upstream artifacts bất biến, ngoại trừ production registry mutation hợp lệ.
- Không file ngoài impact matrix bị thay đổi.

## 14. Test matrix bắt buộc

### 14.1 Pure behavior

- Constant series.
- Increasing series.
- Decreasing series.
- Varying series.
- Single-step H1 source.
- Wrong lag rejection.
- Gap/continuity rejection.
- Target-row leakage rejection.
- Non-finite input rejection.
- Deterministic repeated call.

### 14.2 Population and alignment

- Exactly 2.960 Validation samples.
- Sample uniqueness.
- Chronological order.
- Exact L144 common population membership.
- Population mismatch rejection.
- Duplicate sample rejection.
- Source/target raw index alignment.
- Source/target timestamp alignment.
- Window ID and target sample ID consistency.

### 14.3 Invariance

- Lookback invariance trên cùng target population.
- Feature-variant invariance.
- Time-feature invariance.
- Target-scaling invariance.
- Batch-size invariance.
- Device invariance ở mức formula.
- Seed invariance và no-seed contract.

### 14.4 Metric integration

- MAE independent cross-check.
- RMSE independent cross-check.
- R² independent cross-check.
- Raw Wh unit enforcement.
- Residual sign convention.
- Metric population fingerprint.
- Metric API identity path YS0 không load scaler.

### 14.5 Registry

- Canonical null fields accepted only for Persistence.
- Neural invalid null fields remain rejected.
- Register-before-evaluate ordering.
- Valid status transitions.
- Predictions and metrics required before completion.
- Duplicate config requires canonical rerun reason.
- Completed config immutability.
- Failure traceability.
- Test metric registration rejection.

### 14.6 Notebook and presentation

- One Phase 14 orchestration cell.
- No inline processing.
- Two visible sections only.
- Full log persisted separately.
- Phase boundary becomes 1–14.

## 15. Requirements traceability

| Requirement Phase 14 | Owner implementation | Evidence |
|---|---|---|
| `y_hat(t+1)=y(t)` | `persistence.py` pure predictor | Unit tests, prediction CSV |
| H1 = 10 phút | H1 validator | Audit CSV, unit tests |
| Raw Appliances Wh | Bounded raw loader | Prediction CSV, metric JSON |
| Common L144 population | Window/population integration | Population audit, fingerprint |
| Validation only | Materializer filter và metric context | Prediction rows, metric split |
| Test locked | Data-read boundary, metrics firewall, registry firewall | Firewall tests và audit |
| No feature/scaler dependency | Persistence registry branch và source imports | Config artifact, dependency tests |
| No training/seed/checkpoint | Persistence config validation | Config artifact, registry tests |
| METRICS-v1 | Metric owner API | Metric JSON và metric registry |
| EXPERIMENTS-v1 | Registry lifecycle | JSONL/CSV registry và run status |
| Full traceability | Manifest/sign-off/checksums | Canonical artifacts |
| Notebook call-only | Notebook orchestration cell | Notebook boundary tests |
| Minimal visible output | `phase_summary.py` allowlist | Presentation tests |

## 16. Rủi ro và kiểm soát

| Rủi ro | Hậu quả | Kiểm soát |
|---|---|---|
| Reset DataFrame index | Sample population metric bị lệch | Giữ original window-index index và cross-check metric owner |
| Đọc toàn bộ raw target bao gồm Test | Vi phạm firewall | Bounded raw prefix sau khi lọc Validation metadata |
| Dùng target row làm source | Leakage | Exact source-target index và timestamp audit |
| Gắn Persistence với FS1/YS1 | Sai task-level semantics | Null registry fields và hard-fail validator |
| Dùng scaler vì metric API | Dependency giả | YS0 chỉ ở identity boundary, test no-scaler |
| Complete run thiếu predictions | Thiếu evidence | Persistence-specific required artifact guard |
| Partial artifact sau failure | Rerun không nhất quán | Atomic write, fail_run, stop-and-plan, write-once verify |
| Duplicate canonical run | Registry nhiễu | Existing sign-off verification và no auto-rerun |
| Thay đổi Phase 1–13 presentation | Regression notebook | Snapshot/allowlist tests hiện hữu |
| Dirty worktree bị ghi đè | Mất user changes | Chụp status, chỉ sửa impact matrix, diff review từng bước |

## 17. Alternatives đã đánh giá

### Alternative A - Persistence-specific compatibility trong registry

Chọn.

Lý do:

- Đúng owner và schema lifecycle hiện có.
- Biểu diễn chính xác các field không áp dụng.
- Không bypass Experiment Registry.
- Giữ nguyên contract neural.

### Alternative B - Gắn giả Persistence với FS1_TF1, YS1 và seed 42

Không chọn.

Lý do:

- Sai Phase 14 contract.
- Tạo dependency scaler/feature/seed không tồn tại.
- Làm sai interpretability của baseline.

### Alternative C - Không dùng EXPERIMENTS-v1

Không chọn.

Lý do:

- Mất run lifecycle, checksum registry và failure traceability.
- Vi phạm Phase 14 requirement.

### Alternative D - Tách `persistence.py` và `validation.py`

Không chọn trong Phase 14.

Lý do:

- Phase detail chỉ khuyến nghị hai module.
- Human yêu cầu một file `.py` riêng cho phase.
- Một module vẫn có thể giữ các function boundary rõ ràng mà không làm notebook chứa logic.

## 18. Không cần dependency hoặc migration

- Không thêm package.
- Không sửa `requirements.txt`.
- Không sửa `pyproject.toml`.
- Không migration raw/interim/processed data.
- Không bump `PERSISTENCE-v1` vì đây là implementation đầu tiên.
- Không bump `EXPERIMENTS-v1` nếu thay đổi chỉ hoàn thiện nhánh Persistence đã được family contract Phase 13 dự kiến; phải chứng minh registry contract fingerprint và neural behavior vẫn ổn định.

## 19. Definition of Done

Phase 14 chỉ được xem là hoàn tất khi tất cả điều kiện sau đúng:

- Có đúng một canonical `PERSISTENCE_LAST_VALUE` implementation.
- Formula đúng previous raw Appliances value.
- Validation population đủ 2.960 mẫu và đúng fingerprint.
- Horizon/source lag đúng một bước, 10 phút.
- Không có Test target access hoặc Test metric.
- Không feature/scaler/dataloader/training/seed/checkpoint dependency cho computation.
- MAE, RMSE, R² được tính bởi `METRICS-v1` trên Wh.
- Prediction và metrics được register trước completion.
- Canonical run là `COMPLETED` và idempotent.
- Chín baseline artifacts tồn tại, đúng schema và checksum.
- Processing log đầy đủ; notebook chỉ hiển thị hai section tối thiểu.
- Notebook chỉ call public API Phase 14.
- Architecture rule phản ánh đúng filesystem và flow Phase 0–14.
- Toàn bộ targeted và regression tests pass.
- Không có source comment hoặc icon mới.
- Không có file ngoài scope bị sửa.

## 20. Approval gate

Trạng thái hiện tại: `APPROVED_AND_COMPLETED`.

Human đã duyệt kế hoạch và cho phép thực thi tuần tự. Phase 14 đã hoàn tất toàn bộ Bước 0–11, canonical run đạt `COMPLETED`, sign-off đạt `PASS`, notebook chạy đủ 29 code cell không có stderr và full regression đạt 173 test cùng 4 subtest.
