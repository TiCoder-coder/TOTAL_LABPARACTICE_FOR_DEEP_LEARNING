# Remaining Pipeline Compliance Issues Refactor Plan

## 1. Trạng thái plan

Plan này tổng hợp các vấn đề còn lại sau khi bổ sung `POSTHOC-MAPE-v1` và chạy `ML-PIPELINE-COMPLIANCE-v1`.

Plan chỉ mô tả trình tự kiểm tra, quyết định và refactor. Chưa có thay đổi code, artifact, log hoặc notebook nào được phép thực thi từ plan này cho đến khi Human duyệt bước triển khai.

## 2. Mục tiêu

- Khôi phục tính nhất quán provenance của Phase 9 mà không sửa mù quáng artifact đã ký.
- Đưa toàn bộ direct EDA trong notebook về đúng Train-only decision boundary.
- Kiểm tra khả năng phục hồi frozen Test prediction bundles trước khi quyết định trạng thái Test MAPE.
- Phân biệt rõ lỗi có thể sửa, blocker do thiếu source và limitation khoa học đã được khóa.
- Cập nhật compliance audit và HTML dashboard để người dùng hiểu chính xác trạng thái của từng vấn đề.
- Giữ nguyên model selection, final model lock, Test access history và scientific conclusions Phase 47–59.

## 3. Phân loại vấn đề

### 3.1. Nhóm A: Corrective bắt buộc

#### A0. Phase 1 sign-off checksum provenance corruption

Stage 0 phát hiện ba checksum trong canonical `artifacts/environment/phase_1_signoff.json` không khớp ba output Phase 1 hiện tại. Git provenance xác nhận commit `ff6f963ac1009456e34664523a845792ea8e2fee` chỉ thay checksum trong sign-off mà không thay output tương ứng.

Corrective này được tách thành:

- `docs/plan-doc/analysis_error/phase_1_signoff_checksum_provenance_corruption_issue.md`.
- `docs/plan-doc/plan_before_process/phase_1_signoff_checksum_provenance_corrective_plan.md`.

Stage 1 chưa được phép bắt đầu cho đến khi corrective A0 được Human duyệt, thực thi và verify PASS.

#### A1. Phase 9 README checksum mismatch

Expected SHA-256 trong `artifacts/scaling/phase_9_signoff.json`:

`52edfe46135533583eae041a77bd5b21bbf016e71e7ce8b85b08ec9e9754970c`

Actual SHA-256 của `artifacts/scaling/README_SCALING.md`:

`79ddc8f5a357e5a9ae233c0e52823d82c565a7a2e872c714c17dacbcb2c0dab7`

Ảnh hưởng hiện tại:

- Strict Phase 9 verification từ chối scaler bundle.
- `load_validated_target_scaler` bị chặn.
- Compatibility test của shared metrics còn một lỗi.
- Chưa có bằng chứng cho thấy scaler scientific payload bị sai.

#### A2. Direct EDA notebook boundary

Canonical `EDA-v1` sử dụng 13,814 Train rows và có `train_only_scope=true`. Tuy nhiên direct EDA trong `CourseWork_1.ipynb` đang gọi `load_validated_temporal_view(PROJECT_ROOT)` và tạo DataFrame từ full temporal view 19,735 rows.

Ảnh hưởng hiện tại:

- Canonical EDA artifacts vẫn đúng Train-only.
- Direct notebook EDA không hoàn toàn nhất quán với nguyên tắc split trước, thống kê sau.
- Nếu các quan sát full-data chỉ dùng để mô tả thì chưa tạo model leakage trực tiếp, nhưng boundary vẫn không đủ chặt để dùng làm coursework evidence chuẩn.

### 3.2. Nhóm B: Conditional source-recovery blocker

#### B1. Test MAPE source unavailable

`artifacts/final_test/prediction_checksums.json` khai báo bốn frozen prediction bundles:

- Persistence.
- Transformer seed 42.
- Transformer seed 123.
- Transformer seed 2026.

Bốn CSV không tồn tại ở canonical paths. Vì MAPE không thể suy ra từ MAE, RMSE và R2 nên Test MAPE đang có trạng thái `BLOCKED_SOURCE_UNAVAILABLE`.

Đây chỉ có thể được giải quyết nếu tìm lại đúng frozen prediction CSV và tất cả checksum, row count, schema, population identity đều hợp lệ.

### 3.3. Nhóm C: Declared scientific limitations

#### C1. LSTM final Test comparison

LSTM đã được đánh giá và tuning trên Validation nhưng không được đánh giá trên `FINAL_TEST_POP-v1` do lookback L36 không khớp final Test protocol L72. Trạng thái hiện tại là `NOT_EVALUATED`.

Đây không phải lỗi code. Việc chạy LSTM Test mới sau Phase 59 sẽ thay đổi Test access history và mở rộng frozen protocol.

#### C2. Inferential statistics

Pipeline báo cáo deterministic metrics và descriptive variability qua ba seed. Không có formal hypothesis test hoặc confidence interval. Trạng thái `NOT_APPLICABLE` phản ánh giới hạn nghiên cứu đã công bố.

Không được tự động thêm statistical test hậu kiểm sau Phase 59 nếu chưa có protocol amendment và Human approval.

## 4. Nguyên tắc kiến trúc bắt buộc

### 4.1. Data và EDA layer

- Train-only EDA population được xác định và kiểm tra trong `src/course_work/data/`.
- Notebook không tự lọc split bằng index, timestamp hoặc tỷ lệ hard-code.
- Nếu đã có public Train-only EDA accessor thì phải tái sử dụng.
- Nếu chưa có accessor phù hợp thì chỉ bổ sung một public accessor có trách nhiệm hẹp trong data layer.
- EDA renderer chỉ đọc DataFrame hoặc artifact đã được data layer xác nhận.

### 4.2. Scaling layer

- `src/course_work/data/scaling.py` sở hữu Phase 9 scaler validation và materialization.
- Provenance audit không được thay checksum hoặc bỏ strict verification.
- Reporting layer không được sửa, fit hoặc deserialize scaler để che lỗi.
- Notebook không được xử lý corrective Phase 9 bằng code nội tuyến.

### 4.3. Metric addendum layer

- `src/course_work/evaluation/metrics.py` chỉ sở hữu công thức và numerical guards.
- `src/course_work/metric_addendum/` sở hữu source verification và derived MAPE artifacts.
- Không đưa Test inference, checkpoint loading hoặc scaler fitting vào metric addendum.

### 4.4. Reporting layer

- `src/course_work/reporting/` chỉ đọc artifact và sinh HTML tĩnh.
- Không tính metric, không xác minh scientific source thay data/evaluation layer và không ghi artifact.
- Không dùng JavaScript hoặc Jupyter widget.

### 4.5. Notebook layer

- Notebook chỉ gọi public data/reporting interfaces.
- Không thêm công thức MAPE, checksum logic, split logic, scaler fit, model forward hoặc file serialization vào notebook.
- Không xóa hoặc chạy lại các phase không liên quan.

## 5. Trình tự thực thi

## Stage 0. Preservation gate

### Mục tiêu

Khóa trạng thái trước corrective để mọi thay đổi đều có thể đối chiếu.

### Thực hiện

1. Ghi nhận `git status` và các thay đổi đang tồn tại của Human.
2. Chụp SHA-256 cho:
   - Toàn bộ Phase 9 outputs.
   - Phase 9 signoff và processing log.
   - Phase 10–59 canonical signoffs.
   - Phase 47 final Test contract và checksum registry.
   - Phase 58 final tables signoff.
   - Phase 59 final conclusions signoff.
   - `CourseWork_1.ipynb`.
3. Ghi nhận số cell, cell IDs, execution counts và số output hiện có trong notebook.
4. Chạy targeted baseline tests trước khi sửa:
   - `test_scaling.py`.
   - `test_metrics.py`.
   - MAPE unit/integration/reporting tests.
   - EDA tests hiện hành.

### Gate

- Không sửa file nếu phát hiện thêm scientific artifact ngoài README bị checksum mismatch.
- Không tiếp tục nếu notebook có thay đổi mới chưa xác định thuộc Human hay agent.
- Nếu Phase 1 strict validation fail, dừng Stage 0, lập corrective plan riêng và chờ Human approval trước khi tiếp tục.

## Stage 1. Phase 9 provenance audit

### Mục tiêu

Xác định source-of-truth của `README_SCALING.md` trước khi chọn corrective mode.

### Thực hiện

1. Đọc generator hoặc template tạo README trong scaling layer.
2. So sánh README hiện tại với:
   - Git history.
   - Phase 9 processing log.
   - Phase 9 signoff.
   - Mọi archive hoặc recovery artifact còn tồn tại.
3. Xác minh riêng từng scientific output của Phase 9:
   - X scaler joblib bundles.
   - Y scaler joblib bundle.
   - Scaler registry.
   - Scaling manifest.
   - X/Y scaler statistics.
   - Leakage audit.
   - Round-trip evidence.
4. Kiểm tra Train-only fit population, feature order và split fingerprint mà không refit.
5. Ghi root-cause finding trước khi sửa.

### Corrective modes

#### Mode P9-A: Exact byte recovery

Chỉ dùng khi tìm được bản README canonical có đúng expected checksum và provenance đầy đủ.

- Khôi phục đúng byte README.
- Không sửa Phase 9 signoff.
- Không sửa processing log.
- Không refit scaler.

#### Mode P9-B: Documentation-only amendment

Chỉ dùng khi README hiện tại được chứng minh là canonical mới và thay đổi không tác động scientific contract.

- Không rewrite lịch sử Phase 9 signoff tại chỗ.
- Tạo amendment/version mới theo architecture rule.
- Ghi old checksum, new checksum, lý do thay đổi và compatibility decision.
- Cập nhật strict loader theo amendment đã duyệt, không vô hiệu hóa checksum validation.

#### Mode P9-C: Scientific corrective

Dùng khi bất kỳ scaler/statistics/leakage artifact nào sai hoặc không còn xác minh được.

- Dừng plan tổng hợp.
- Lập corrective plan Phase 9 riêng.
- Không refit hoặc phát hành artifact mới trước Human approval.

### Verification sau Stage 1

1. `load_validated_target_scaler` phải hoạt động.
2. `test_scaling.py` phải PASS.
3. `test_metrics.py` phải PASS.
4. Scaler numerical statistics phải không đổi nếu Mode P9-A hoặc P9-B được chọn.
5. Phase 10–59 frozen checksums phải không đổi.

### Gate

Chỉ được chuyển sang Stage 2 khi Mode P9-A hoặc Mode P9-B hoàn thành và toàn bộ verification PASS. Mode P9-C yêu cầu dừng để lập plan mới.

## Stage 2. Train-only direct EDA refactor

### Mục tiêu

Đảm bảo canonical artifacts và mọi direct EDA output trong notebook cùng dùng Phase 5 Train population.

### Thực hiện tại data layer

1. Kiểm tra public accessor hiện có để lấy Train-only temporal/EDA view.
2. Nếu accessor phù hợp đã tồn tại, tái sử dụng và không tạo API trùng.
3. Nếu chưa có, bổ sung accessor hẹp với các kiểm tra:
   - Đọc canonical split membership.
   - Chỉ lấy `TRAIN` rows.
   - Bảo toàn chronological order.
   - Bảo toàn schema column order.
   - Xác minh đúng 13,814 rows.
   - Xác minh train fingerprint hoặc membership identity.
   - Không ghi hoặc biến đổi canonical dataset.
4. Không hard-code row boundary trong notebook.

### Thực hiện tại reporting layer

1. Tái sử dụng renderer HTML hiện có nếu renderer nhận Train-only DataFrame.
2. Chỉ sửa reporting code nếu cần cập nhật displayed population label hoặc metadata.
3. Các bảng và hình phải ghi rõ `Train-only EDA` và `13,814 rows`.

### Thực hiện tại notebook layer

1. Thay full temporal view bằng public Train-only accessor.
2. Không thay đổi thứ tự EDA sections.
3. Không xóa các bước `info`, descriptive summary, histogram, KDE, skewness, tail, outlier, correlation, lag, autocorrelation và rolling analysis.
4. Không thay đổi canonical feature selection hoặc model configuration từ EDA.
5. Chỉ chạy lại nhóm cell Phase 6 EDA.
6. Không chạy lại Phase 1–5 hoặc Phase 7–59.
7. Giữ nguyên output của các phase ngoài Phase 6.

### Verification sau Stage 2

1. Tất cả direct EDA DataFrame có 13,814 rows.
2. Timestamp đầu/cuối khớp Train membership.
3. Không có Validation/Test row trong EDA population.
4. Canonical EDA signoff vẫn PASS.
5. Raw data và canonical feature artifacts không đổi checksum.
6. EDA test suite PASS.
7. Notebook JSON hợp lệ.
8. Chỉ Phase 6 execution/output metadata được thay đổi.

### Gate

Không chuyển Stage 3 nếu bất kỳ EDA output nào còn hiển thị 19,735 rows hoặc lấy dữ liệu ngoài Train.

## Stage 3. Frozen Test prediction source recovery audit

### Mục tiêu

Xác định Test MAPE có thể được tính từ evidence cũ hay phải tiếp tục blocked.

### Read-only search scope

1. Canonical `artifacts/final_test/`.
2. Các thư mục archive hiện có trong project.
3. Git history của đúng prediction paths.
4. Backup path do Human cung cấp nếu có.

Không tìm hoặc sử dụng prediction từ notebook output, hình ảnh, metric summary hoặc dữ liệu gõ tay.

### Source acceptance gate

Một prediction bundle chỉ hợp lệ khi đồng thời thỏa mãn:

- Path hoặc source identity khớp frozen registry.
- SHA-256 khớp tuyệt đối.
- Row count bằng 2,961.
- Có `y_true_wh` và `y_pred_wh` ở original Wh.
- Không có NaN hoặc Inf.
- Sample IDs không trùng.
- Population fingerprint khớp `FINAL_TEST_POP-v1`.
- Residual identity nhất quán nếu residual column tồn tại.

### Branch B1-A: Đủ bốn bundle hợp lệ

1. Khôi phục bundle về canonical paths mà không thay đổi byte.
2. Chạy source audit và checksum verification.
3. Tính MAPE một lần trên toàn bộ population của từng bundle.
4. Báo cáo Persistence và ba Transformer seed riêng biệt.
5. Chỉ tính mean và sample SD của ba Transformer seed sau khi cả ba seed đều hợp lệ.
6. Gắn nhãn `POSTHOC_SUPPLEMENTARY_TEST_METRIC`.
7. Không thay đổi Phase 47–59 artifacts hoặc conclusions.

### Branch B1-B: Thiếu hoặc sai bất kỳ bundle nào

1. Không khôi phục một phần để tạo aggregate không đầy đủ.
2. Giữ `BLOCKED_SOURCE_UNAVAILABLE`.
3. Cập nhật source audit với path, expected checksum, actual checksum và lý do từ chối.
4. Không chạy Test inference.

### Verification sau Stage 3

- Không có model import, checkpoint load, forward pass hoặc inference call.
- Không có training hoặc scaler fitting.
- MAPE được đối chiếu độc lập với công thức chuẩn.
- Zero-target policy được kiểm tra trên từng population.
- Phase 47–59 checksum không đổi.

## Stage 4. LSTM final Test limitation decision

### Mục tiêu

Ngăn trạng thái `PARTIAL` bị hiểu nhầm thành lỗi triển khai có thể tự động sửa.

### Thực hiện

1. Xác minh lại LSTM final Test eligibility source và lookback mismatch L36/L72.
2. Không chạy LSTM Test inference trong plan này.
3. Cập nhật compliance classification thành một trạng thái rõ nghĩa, ví dụ `DECLARED_LIMITATION`, nếu architecture contract cho phép.
4. Dashboard phải giải thích:
   - LSTM Validation comparison đã tồn tại.
   - LSTM final Test không được thực thi theo frozen eligibility decision.
   - Không thể so sánh final Test Transformer và LSTM trên cùng protocol hiện tại.

### Escalation condition

Nếu rubric bắt buộc LSTM phải có final Test metric, dừng và yêu cầu Human cùng giảng viên quyết định một protocol amendment mới. Không mở Test hoặc thay đổi Phase 59 dưới plan này.

## Stage 5. Inferential statistics classification

### Mục tiêu

Giữ đúng scope khoa học và tránh coi `NOT_APPLICABLE` là execution failure.

### Thực hiện

1. Giữ nguyên không có formal significance test hậu kiểm.
2. Phân loại thành `DECLARED_LIMITATION` hoặc giữ `NOT_APPLICABLE`.
3. Dashboard giải thích mean và sample SD ba seed là descriptive statistics, không phải confidence interval hoặc hypothesis test.
4. Không thêm p-value, confidence interval hoặc block bootstrap sau Phase 59 trong plan này.

## Stage 6. Compliance audit and dashboard refactor

### Mục tiêu

Hiển thị trạng thái đúng nghĩa thay vì gom tất cả vào `Needs attention`.

### Audit categories đề xuất

- `PASS`.
- `ACTION_REQUIRED`.
- `BLOCKED_SOURCE_UNAVAILABLE`.
- `DECLARED_LIMITATION`.
- `NOT_APPLICABLE` nếu cần giữ tương thích.
- `FAIL` chỉ dùng khi invariant hoặc verification thực sự sai.

### Thực hiện tại audit layer

1. Re-materialize compliance audit từ canonical evidence.
2. Phase 9 chỉ chuyển PASS sau khi strict loader và tests PASS.
3. EDA boundary chỉ chuyển PASS sau khi mọi direct EDA view là Train-only.
4. Test MAPE trạng thái phụ thuộc duy nhất vào Stage 3 source gate.
5. LSTM và inferential statistics được phân loại là limitation, không phải lỗi corrective.

### Thực hiện tại reporting layer

1. Dashboard chia riêng:
   - Corrective actions.
   - Source blockers.
   - Declared limitations.
2. Không hiển thị limitation như execution failure.
3. Giữ HTML tĩnh.
4. Không thêm widget, script, icon hoặc nội dung tính toán.

### Thực hiện tại notebook layer

1. Cell MAPE addendum tiếp tục chỉ gọi public renderer.
2. Chạy lại duy nhất cell `mape-addendum-dashboard`.
3. Không xóa output cũ của các phase khác.

## Stage 7. Final verification

### Code and architecture verification

- Không có comment, docstring hoặc icon mới trong code.
- Không trộn data, evaluation, materialization và reporting responsibilities.
- Không có notebook processing logic mới.
- Import graph của metric addendum không chạm training, Phase 47 inference hoặc checkpoint loader.

### Test verification

- Phase 9 scaling tests PASS.
- Shared metrics tests PASS.
- MAPE unit tests PASS.
- MAPE integration tests PASS.
- MAPE reporting tests PASS.
- EDA tests PASS.
- Notebook structural validation PASS.

### Artifact verification

- Processing log checksum khớp output artifact.
- Không có MAPE aggregate giữa các hyperparameter configuration khác nhau.
- Test MAPE chỉ tồn tại khi source gate PASS.
- Phase 10–59 frozen artifacts không thay đổi ngoài read-only presentation artifact được cho phép.

### Notebook verification

- Notebook vẫn có đầy đủ cell và output ngoài scope.
- Chỉ Phase 6 cells và MAPE dashboard cell được phép có execution metadata mới.
- Không có `application/vnd.jupyter.widget-view+json` trong output mới.
- HTML hiển thị đúng population, trạng thái và limitation.

## 6. File scope dự kiến

### Có thể được đọc

- `src/course_work/data/`.
- `src/course_work/evaluation/`.
- `src/course_work/metric_addendum/`.
- `src/course_work/reporting/`.
- `artifacts/scaling/`.
- `artifacts/eda/`.
- `artifacts/final_test/`.
- `artifacts/metric_addendum/mape/`.
- `docs/save_log_in_processing/`.
- `notebook_course_work/CourseWork_1.ipynb`.
- Các tests liên quan.

### Chỉ được sửa sau khi mode tương ứng được duyệt

- Phase 9 README hoặc amendment path được architecture rule cho phép.
- Public Train-only EDA accessor nếu thực sự còn thiếu.
- EDA reporting helper nếu population label cần sửa.
- Direct EDA cells trong `CourseWork_1.ipynb`.
- `src/course_work/metric_addendum/audit.py`.
- `src/course_work/reporting/mape_addendum.py`.
- Derived MAPE addendum artifacts và processing logs.
- Targeted tests.

### Không được sửa

- Phase 47 official Test metrics.
- Phase 45 final model lock.
- Phase 46 final checkpoints.
- Phase 48–57 diagnostic scientific artifacts.
- Phase 58 final tables.
- Phase 59 conclusions.
- Historical prediction values.
- Model winner hoặc seed decision.

## 7. Điều cấm tuyệt đối

- Không thay checksum chỉ để test PASS.
- Không tắt strict verification.
- Không fit lại scaler khi chưa có scientific corrective approval.
- Không dùng Validation/Test để fit preprocessing statistics.
- Không chạy lại toàn notebook.
- Không chạy training hoặc fine-tuning.
- Không chạy Test inference mới.
- Không load checkpoint để tái tạo frozen Test prediction.
- Không suy ngược MAPE từ MAE, RMSE hoặc R2.
- Không đọc số liệu từ ảnh.
- Không gõ tay scientific metric.
- Không chọn best seed.
- Không tạo ensemble.
- Không sửa kết luận frozen để chèn MAPE như metric đã được định trước.
- Không thêm comment, docstring hoặc icon vào code.

## 8. Thứ tự ưu tiên

1. Stage 0 preservation.
2. Stage 1 Phase 9 provenance corrective.
3. Stage 2 Train-only direct EDA refactor.
4. Stage 3 Test prediction source recovery audit.
5. Stage 4 LSTM limitation classification.
6. Stage 5 inferential-statistics classification.
7. Stage 6 audit/dashboard refresh.
8. Stage 7 final verification.

Không đảo thứ tự Stage 1 và Stage 2 vì Phase 9 compatibility phải được đóng trước khi chạy regression liên kết toàn pipeline.

## 9. Điều kiện hoàn thành

Plan được coi là hoàn thành khi:

- Phase 9 strict verification và related tests PASS, hoặc Phase 9 được dừng ở Mode P9-C với corrective plan riêng đã được Human xác nhận.
- Direct notebook EDA hoàn toàn Train-only.
- Test MAPE có kết quả từ đủ source hợp lệ hoặc giữ blocked với source audit đầy đủ.
- LSTM và inferential statistics được hiển thị đúng là declared limitations.
- Compliance audit không còn dùng `PARTIAL` cho các limitation không thể sửa trong frozen protocol.
- Dashboard cuối notebook đã chạy riêng và hiển thị đúng trạng thái mới.
- Không có frozen scientific artifact nào bị thay đổi trái phép.

## 10. Approval gate

Sau khi Human duyệt plan này, quá trình thực thi phải diễn ra tuần tự từ Stage 0. Sau mỗi stage phải chạy verification tương ứng và báo kết quả trước khi chuyển stage tiếp theo.

Stage 1 phải dừng thêm một lần sau provenance audit để Human duyệt `P9-A`, `P9-B` hoặc `P9-C` trước khi sửa bất kỳ Phase 9 artifact nào.
