# Kế hoạch trực quan hóa cấu hình Transformer sau Phase 33 trong notebook

## 1. Danh tính kế hoạch

```text
Plan ID: PHASE-33-TRANSFORMER-CONFIGURATION-VISUALIZATION-PLAN-v1
Related Phase: Phase 33
Related Phase name: S11 d_model Sweep
Change type: Reporting and notebook orchestration
Execution mode: Strictly sequential
Implementation status: WAITING_FOR_USER_APPROVAL
```

## 2. Mục tiêu

Tạo một HTML visualization trong `CourseWork.ipynb` để người dùng nhìn thấy toàn bộ cấu hình Transformer canonical sau Phase 33, current reference run và Validation RMSE mà không phải mở trực tiếp các artifact JSON.

Notebook chỉ được gọi hàm reporting. Mọi thao tác đọc artifact, kiểm tra lineage, xây dựng dữ liệu trình bày và sinh HTML phải thuộc `src/course_work/reporting/phase_summary.py`.

## 3. Current canonical state đã xác minh

Nguồn canonical hiện tại xác nhận:

```text
Phase 33 status: PASS
Approved for Phase 34: true
Current reference run: RUN_TR_S09_0016_AE0FB819
Selected d_model: 64
Validation RMSE: 58.08190056355405 Wh
Feature variant: FS2_TF1
Feature count: 33
Target scaling: YS1
Lookback: L36, 36 steps, 6 hours
Pooling: LAST_STEP
Activation: GELU
Batch size: 32
Learning rate: 0.0003
Weight decay: 0.001
Dropout: 0.1
Heads: 4
Layers: 2
FFN width: 128
Loss: MSE
Max epochs: 50
Early-stopping patience: 10
Gradient clipping: 1.0
RevIN: disabled
Boundary protocol: WB0_CONTEXT_CARRY_OVER
```

Nguồn đã đối chiếu:

```text
artifacts/sweeps/S11_d_model/phase_33_signoff.json
artifacts/sweeps/S11_d_model/s11_d_model_winner.json
artifacts/sweeps/S11_d_model/s11_reference_update.json
artifacts/runs/RUN_TR_S09_0016_AE0FB819/config.json
artifacts/feature_sets/feature_set_registry.json
```

## 4. Phạm vi thay đổi

### 4.1. File dự kiến thay đổi

```text
COURSE_WORK/src/course_work/reporting/phase_summary.py
COURSE_WORK/notebook_course_work/CourseWork.ipynb
COURSE_WORK/tests/unit/test_phase_summary.py
COURSE_WORK/tests/contracts/test_selective_execution_policy.py
```

### 4.2. File chỉ đọc

```text
COURSE_WORK/working_rule.md
COURSE_WORK/docs/RULE_BASE/rule_code.md
COURSE_WORK/docs/RULE_BASE/architecture_rule.md
COURSE_WORK/docs/plan-doc/plan_detail_for_each_phase/Phase_23_S1_Feature-set_sweep.md
COURSE_WORK/docs/plan-doc/plan_detail_for_each_phase/Phase_24_S2_Time-feature_sweep.md
COURSE_WORK/docs/plan-doc/plan_detail_for_each_phase/Phase_25_S3_Target-scaling_sweep.md
COURSE_WORK/docs/plan-doc/plan_detail_for_each_phase/Phase_26_S4_Lookback_sweep.md
COURSE_WORK/docs/plan-doc/plan_detail_for_each_phase/Phase_27_S5_Pooling_sweep.md
COURSE_WORK/docs/plan-doc/plan_detail_for_each_phase/Phase_28_S6_Activation_sweep.md
COURSE_WORK/docs/plan-doc/plan_detail_for_each_phase/Phase_29_S7_Batch_sweep.md
COURSE_WORK/docs/plan-doc/plan_detail_for_each_phase/Phase_30_S8_Learning-rate_sweep.md
COURSE_WORK/docs/plan-doc/plan_detail_for_each_phase/Phase_31_S9_Weight-decay_sweep.md
COURSE_WORK/docs/plan-doc/plan_detail_for_each_phase/Phase_32_S10_Dropout_sweep.md
COURSE_WORK/docs/plan-doc/plan_detail_for_each_phase/Phase_33_S11_d_model_sweep.md
COURSE_WORK/artifacts/sweeps/S11_d_model/phase_33_signoff.json
COURSE_WORK/artifacts/sweeps/S11_d_model/s11_d_model_winner.json
COURSE_WORK/artifacts/sweeps/S11_d_model/s11_reference_update.json
COURSE_WORK/artifacts/runs/RUN_TR_S09_0016_AE0FB819/config.json
COURSE_WORK/artifacts/feature_sets/feature_set_registry.json
```

### 4.3. File tuyệt đối không thay đổi

```text
Mọi artifact canonical của Phase 0-33
Mọi processing log hiện có
Mọi checkpoint và training history
Mọi raw, interim, processed và split data
Mọi model, training, sweep và evaluation implementation
Mọi Phase detail
Mọi output cell notebook hiện có ngoài cell mới thuộc visualization này
```

## 5. Non-goals

Kế hoạch này không:

```text
chạy lại training
chạy lại sweep
thay current reference
thay hyperparameter
thay Validation metric
đọc hoặc mở Test
ghi đè artifact Phase 33
sửa scientific behavior
đưa processing logic vào notebook
xóa output notebook hiện có
thêm widget hoặc JavaScript runtime
```

## 6. Architecture constraints

```text
Reporting logic thuộc src/course_work/reporting/phase_summary.py.
Notebook chỉ import và gọi một public rendering function.
Notebook không tự đọc JSON, không tự ghép bảng và không chứa CSS/HTML construction.
Visualization phải dùng static IPython HTML để không phụ thuộc widget state.
Artifact canonical là source of truth; notebook không hard-code metric hoặc configuration.
Mọi giá trị phải được HTML-escaped trước khi render.
Không thêm comment, docstring, icon, emoji hoặc decorative symbol vào source code.
Không tạo artifact scientific mới cho thay đổi presentation này.
Không thay đổi output đã lưu của các cell không thuộc scope.
```

## 7. Thiết kế output

HTML visualization dự kiến có ba phần.

### 7.1. Header và current reference

Hiển thị:

```text
Transformer Configuration after Phase 33
Status: PASS
Current reference run: RUN_TR_S09_0016_AE0FB819
Validation RMSE: 58.08190056355405 Wh
Approved next phase: Phase 34
```

### 7.2. Current configuration table

Hiển thị đủ các cột:

```text
Component
Current value
Decision source
State
```

Các hàng bắt buộc:

```text
Feature set
Target scaling
Lookback
Pooling
Activation
Batch size
Learning rate
AdamW weight decay
Dropout
d_model
Heads
Layers
FFN width
Loss
Max epochs
Early-stopping patience
Gradient clipping
RevIN
Boundary protocol
```

`Decision source` thể hiện Phase đã chọn giá trị hoặc trạng thái frozen/chưa sweep tại Phase 33. `State` phân biệt giá trị đã được sweep với giá trị đang frozen hoặc chưa được kiểm tra.

### 7.3. Lineage summary

Hiển thị ngắn gọn:

```text
Phase 33 artifact version
Winner condition
Selection metric
Current reference run
Validation RMSE
Validation MAE
Validation R²
Trainable parameter count
Test access
```

Không hiển thị checksum, fingerprint dài, raw JSON hoặc technical dump.

## 8. Quy tắc lấy dữ liệu

Public builder phải thực hiện tuần tự:

1. Đọc `phase_33_signoff.json`.
2. Xác nhận `status == PASS` và `approved_for_phase34 == true`.
3. Đọc `s11_d_model_winner.json` và `s11_reference_update.json`.
4. Xác nhận winner run, reference run, selected `d_model` và Validation RMSE đồng nhất giữa ba artifact.
5. Resolve run config từ `current_reference_run_id`, không hard-code tên thư mục run.
6. Xác nhận run ID bên trong config trùng current reference.
7. Đọc data, model và training config từ run config.
8. Đối chiếu feature count của selected feature variant với `feature_set_registry.json`.
9. Chỉ sau khi toàn bộ consistency gate pass mới xây dựng view model cho renderer.
10. Nếu thiếu hoặc lệch artifact, fail rõ ràng thay vì hiển thị cấu hình có thể sai.

## 9. Trình tự implementation và verify

### Step 1. Khóa preservation baseline

Thực hiện:

```text
Ghi nhận git diff hiện tại của các file target.
Ghi nhận cell ID, source và output hash của toàn bộ notebook.
Ghi nhận checksum các artifact Phase 33 và current reference config.
```

Verify ngay:

```text
Không có file bị thay đổi ở Step 1.
Dirty worktree hiện có được phân biệt với thay đổi của task này.
Artifact canonical có thể đọc và các trường bắt buộc tồn tại.
```

Stop condition:

```text
Artifact Phase 33 thiếu, status không PASS hoặc lineage không đồng nhất.
```

### Step 2. Xây dựng reporting data contract

Thực hiện trong `phase_summary.py`:

```text
Thêm builder đọc và validate canonical Phase 33 state.
Tạo view model chỉ gồm dữ liệu decision-facing.
Không ghi file và không thay đổi artifact.
```

Verify ngay:

```text
Builder trả đúng current reference run.
Feature count bằng 33.
RMSE, MAE và R² khớp winner artifact.
Model/training/data configuration khớp run config.
Mismatch synthetic bị reject.
```

Regression link:

```text
Builder không gọi training, sweep finalization hoặc Test evaluation.
Existing Phase 0-33 reporting APIs giữ nguyên signature và behavior.
```

### Step 3. Xây dựng static HTML renderer

Thực hiện trong `phase_summary.py`:

```text
Thêm public renderer cho cấu hình Transformer sau Phase 33.
Tái sử dụng visual language và table-layout contract hiện tại.
Giữ cột Component và Current value gần nhau, dễ đọc và responsive.
Không dùng widget, JavaScript hoặc external asset.
```

Verify ngay:

```text
HTML có header, status, metric cards, configuration table và lineage summary.
Tất cả giá trị bắt buộc xuất hiện đúng một lần trong khu vực tương ứng.
Không có raw JSON, checksum hoặc technical dump.
Không có script, widget MIME, icon hoặc comment mới.
Giá trị đầu vào được escape.
```

Regression link:

```text
CSS selector được namespace riêng.
Existing phase tables không đổi layout.
Renderer không mutate dữ liệu đã load.
```

### Step 4. Tích hợp notebook ở orchestration layer

Thực hiện trong `CourseWork.ipynb`:

```text
Chèn một Markdown heading sau Phase 32 và trước All Phase Logs.
Chèn một code cell chỉ import và gọi public renderer.
Giữ nguyên toàn bộ cell và output hiện có.
Không thêm JSON loading, table construction, CSS hoặc HTML logic vào notebook.
```

Verify ngay:

```text
Notebook vẫn là JSON hợp lệ.
Cell mới nằm đúng thứ tự.
Code cell chỉ có import và function call.
Không có duplicate cell ID.
Không có widget MIME output.
Không có output cũ bị xóa hoặc thay đổi.
```

Regression link:

```text
Phase 1-32 cells giữ nguyên source và output.
All Phase Logs cell giữ nguyên source và output.
Notebook không có processing hoặc training logic mới.
```

### Step 5. Bổ sung automated tests

Thực hiện:

```text
Unit test canonical builder và HTML renderer.
Test artifact consistency gates.
Contract test notebook cell source, vị trí và static presentation boundary.
Preservation test cho non-target notebook outputs.
```

Verify ngay:

```text
Targeted unit tests pass.
Notebook boundary tests pass.
Selective execution reporting tests pass.
```

### Step 6. Final regression

Thực hiện tuần tự:

```text
Compile source reporting module.
Run targeted reporting tests.
Run notebook architecture and boundary tests.
Run full relevant test suite nếu targeted tests pass.
Review final diff.
Recompute artifact checksums để chứng minh canonical artifacts không đổi.
Audit comment, icon, widget MIME và notebook output preservation.
```

## 10. Validation strategy

### 10.1. Data correctness

```text
Current reference lấy từ s11_reference_update.json.
Metric lấy từ s11_d_model_winner.json và đối chiếu sign-off.
Config lấy từ config.json của current reference run.
Feature count được đối chiếu với feature_set_registry.json.
```

### 10.2. Presentation correctness

```text
Static HTML render được trong Jupyter và VS Code notebook.
Không phụ thuộc kernel widget state.
Không hiển thị raw JSON.
Không kéo giãn cột Component và Current value bất hợp lý.
Không expose fingerprint và checksum không cần thiết.
```

### 10.3. Architecture correctness

```text
Source owns artifact loading and rendering.
Notebook owns orchestration only.
Artifact layer không bị sửa.
Training và evaluation layer không bị gọi.
```

## 11. Acceptance criteria

Task chỉ hoàn thành khi:

```text
Visualization xuất hiện sau Phase 32 và trước All Phase Logs.
Toàn bộ 19 thành phần cấu hình được hiển thị.
Current reference run đúng RUN_TR_S09_0016_AE0FB819.
Validation RMSE đúng 58.08190056355405 Wh.
Validation MAE đúng 27.595002038670813 Wh.
Validation R² đúng 0.6034923842647237.
Feature count đúng 33.
Phase/state của từng cấu hình được trình bày rõ.
Notebook code cell chỉ gọi reporting function.
Không có artifact canonical nào thay đổi.
Không có output notebook cũ nào bị xóa.
Không có comment, icon, widget hoặc JavaScript mới.
Targeted tests và regression tests liên quan đều pass.
```

## 12. Rollback và stop conditions

Phải dừng, không tự mở rộng scope nếu:

```text
Phát hiện Phase 33 artifact không đồng nhất.
Current reference config thiếu trường bắt buộc.
Feature registry không khớp run config.
Việc chèn cell làm thay đổi output cell khác.
Existing dirty changes chồng lấn và không thể bảo toàn.
Targeted test phát hiện scientific hoặc notebook regression.
Cần sửa producer artifact, training code hoặc sweep code.
```

Rollback của task chỉ được hoàn tác đúng các dòng và cell do task này tạo, không được reset hoặc ghi đè thay đổi hiện có của người dùng.

## 13. User approval gate

```text
PLAN_CREATED=true
IMPLEMENTATION_STARTED=false
WAITING_FOR_USER_APPROVAL=true
```

Sau khi người dùng kiểm tra và duyệt plan này, implementation mới được bắt đầu theo đúng Step 1 đến Step 6.

## 14. Điều chỉnh trong quá trình thực thi

Notebook boundary regression tại Step 6 phát hiện metadata execution của các cell cuối đang theo thứ tự:

```text
Phase 30: 46
Phase 31: 2
Phase 32: 3
Phase 33 configuration: null
All Phase Logs: 5
```

Trạng thái này vi phạm contract yêu cầu notebook đã execute phải có execution count dạng integer, tăng dần và không trùng.

Điều chỉnh được giới hạn trong cùng file notebook và cùng presentation scope:

```text
Phase 31: 47
Phase 32: 48
Phase 33 configuration: 49
All Phase Logs: 50
```

Static HTML của riêng Phase 33 configuration được materialize từ public renderer vào output của cell mới. Không chạy lại Phase, không chạy lại notebook, không thay source hoặc output content của cell cũ và không thay artifact canonical.
