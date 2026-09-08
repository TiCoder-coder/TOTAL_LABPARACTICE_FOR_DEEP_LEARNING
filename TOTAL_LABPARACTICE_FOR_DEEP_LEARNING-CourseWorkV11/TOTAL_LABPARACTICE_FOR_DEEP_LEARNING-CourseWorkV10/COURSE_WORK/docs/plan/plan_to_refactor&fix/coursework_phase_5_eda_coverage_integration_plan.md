# COURSEWORK PHASE 5 EDA COVERAGE INTEGRATION PLAN

## Plan metadata

```text
Plan ID: CW-PHASE-5-EDA-COVERAGE-001
Scope: Integrate non-duplicate EDA coverage from EDA.ipynb into CourseWork.ipynb
Execution mode: STRICTLY_SEQUENTIAL
Status: SUPERSEDED_BY_CW-PHASE-5-EDA-DIRECT-001
Canonical notebook: COURSE_WORK/notebook_course_work/CourseWork.ipynb
Reference notebook: COURSE_WORK/notebook_course_work/EDA.ipynb
```

## 1. Objective

Mở rộng phần Phase 5 trong `CourseWork.ipynb` để trình bày đầy đủ các nhóm EDA có giá trị trong `EDA.ipynb`, đồng thời:

```text
Không lặp lại analysis đã có trong EDA-v1.
Không thay đổi thứ tự Phase 0 đến Phase 5.
Không đưa processing logic vào notebook.
Không tải lại dataset từ UCI trong Phase 5.
Không thay đổi raw data hoặc validated temporal view.
Không xóa outlier, thay outlier bằng NaN hoặc interpolation.
Không khởi động Phase 6.
Không ghi đè sign-off EDA-v1 hiện có.
```

## 2. Baseline cần bảo toàn

```text
CourseWork.ipynb SHA-256: 01cb6b38c8aacca3adc579036a1b6ce0144d89a2fba7623c2d265add6999e1f9
EDA.ipynb SHA-256: 407c32445ee0ee2a5d7e07af96b037c280f3408db38e881e04d26ae20f658780
Phase 5 sign-off SHA-256: c08c53bf97e9f9950f2b04ad85d850ebe8216b7a227c1625d03af97299455327
Raw CSV SHA-256: 2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d
EDA-v1 figures: 16
EDA-v1 tables: 16
Phase 6 artifacts: absent
```

Các hash trên là baseline trước implementation. Hash của `CourseWork.ipynb` được phép thay đổi do refactor presentation. Raw CSV và sign-off Phase 0 đến Phase 5 không được thay đổi.

## 3. Nguyên tắc tích hợp

### 3.1. Giữ notebook orchestration-only

`CourseWork.ipynb` chỉ được:

```text
Gọi public materialization API.
Đọc signed manifest và table artifact.
Hiển thị DataFrame đã được materialize.
Hiển thị figure artifact bằng IPython.display.Image.
Hiển thị sign-off và boundary statement.
```

Notebook không được chứa:

```text
fetch_ucirepo
pd.read_csv
pd.to_datetime
DataFrame mutation
Function hoặc class definition
Loop xử lý dữ liệu
corr hoặc rolling calculation
IQR calculation
interpolate, fillna hoặc drop
matplotlib hoặc seaborn plotting logic
sys.path hoặc PYTHONPATH bootstrap
```

### 3.2. Không tạo file Python owner mới

Calculation bổ sung thuộc owner hiện có:

```text
src/course_work/data/eda.py
```

Figure rendering và artifact materialization bổ sung thuộc owner hiện có:

```text
src/course_work/reporting/eda.py
```

### 3.3. Bảo toàn EDA-v1

Không sửa hoặc xóa các artifact đã ký trong:

```text
artifacts/eda/tables
artifacts/eda/figures
artifacts/eda/eda_manifest.json
artifacts/eda/phase_5_signoff.json
```

Phần bổ sung được materialize độc lập dưới:

```text
artifacts/eda/supplement/tables
artifacts/eda/supplement/figures
artifacts/eda/supplement/eda_presentation_manifest.json
artifacts/eda/supplement/phase_5_presentation_signoff.json
```

Supplement là phần mở rộng presentation của Phase 5, không phải Phase mới và không thay thế `EDA-v1`.

## 4. Ma trận đối chiếu EDA

| Nhóm trong EDA.ipynb | Trạng thái trong workflow hiện tại | Quyết định |
|---|---|---|
| Import pandas, NumPy, plotting libraries | Đã thuộc source owners | Không đưa vào notebook |
| `fetch_ucirepo(id=374)` | Phase 2 đã có acquisition và provenance | Bỏ phần trùng |
| Ghép `X` và `y` | Phase 2–4 đã tạo validated data flow | Bỏ phần trùng |
| `df.info()` và dtype overview | Phase 3 đã có schema manifest và variable dictionary | Bổ sung presentation từ artifact Phase 3 |
| `df.describe()` | Đã có `eda_numeric_summary.csv` với nhiều quantile hơn | Dùng artifact hiện có |
| Missing count và percentage | Phase 3 đã audit null count | Bổ sung bảng presentation, không tính lại trong notebook |
| Missingness heatmap | Chưa có figure tương đương | Bổ sung figure supplement |
| Parse timestamp | Phase 4 đã parse và audit | Bỏ phần trùng |
| Hour, weekday, day name, weekend | `build_eda_view()` đã có | Dùng derived EDA view hiện có |
| Weekday/weekend sample counts | Profile hiện có chứa count nhưng chưa có count figure riêng | Bổ sung sampling-count figure |
| Day-of-week sample counts | Profile hiện có chứa count | Bổ sung vào sampling-count figure |
| Hour sample counts | Profile hiện có chứa count | Bổ sung vào sampling-count figure |
| Target skewness | Đã có trong numeric summary | Dùng artifact hiện có |
| Target histogram | Đã có `EDA_01` | Bỏ phần trùng |
| Daily target trend | `EDA_04` và rolling figures đã có | Bỏ phần trùng |
| First representative week | Representative window đã được chọn continuity-safe | Bổ sung display từ canonical view nếu chưa được notebook hiển thị |
| Correlation matrix | Đã có `EDA_13` | Bỏ phần trùng |
| Top-five target correlations | Đã có `target_correlations.csv` | Bổ sung table display trong notebook |
| Top-two correlation scatterplots | Chưa có | Bổ sung figure supplement |
| Cross-correlation ±24 bước | Hiện có selected positive lags, chưa có profile ±24 | Bổ sung segment-aware bidirectional profile và figure |
| Target theo weekday/weekend | Đã có aggregate profile nhưng chưa có conditional boxplot | Bổ sung figure supplement |
| Target theo hour | Đã có aggregate profile nhưng chưa có conditional boxplot | Bổ sung figure supplement |
| Sensor IQR outlier counts | Chưa có table riêng | Bổ sung descriptive table và figure |
| Thay outlier bằng NaN | Vi phạm Phase 5 boundary | Không tích hợp |
| Linear interpolation | Vi phạm Phase 5 boundary và có nguy cơ leakage | Không tích hợp |
| Before/after cleaned boxplot | Phụ thuộc preprocessing bị cấm | Thay bằng raw diagnostic với IQR bounds, không sửa dữ liệu |

## 5. Output bổ sung dự kiến

### 5.1. Tables

```text
eda_missingness_summary.csv
top_feature_cross_correlation_profile.csv
sensor_iqr_outlier_summary.csv
```

Không tạo table mới cho hourly, weekday, weekend, correlation ranking hoặc numeric summary vì các table tương ứng đã tồn tại trong EDA-v1.

### 5.2. Figures

```text
EDA_17_missingness_overview.png
EDA_18_sampling_counts.png
EDA_19_top_target_feature_scatter.png
EDA_20_target_conditional_boxplots.png
EDA_21_top_feature_cross_correlation.png
EDA_22_sensor_iqr_outlier_counts.png
EDA_23_sensor_outlier_diagnostic.png
```

`EDA_23` chỉ hiển thị raw distribution, raw boxplot và IQR bounds của feature có nhiều IQR flags nhất. Figure này không hiển thị dữ liệu đã nội suy và không tạo cleaned DataFrame.

## 6. Files dự kiến bị ảnh hưởng

```text
COURSE_WORK/src/course_work/data/eda.py
COURSE_WORK/src/course_work/reporting/eda.py
COURSE_WORK/notebook_course_work/CourseWork.ipynb
COURSE_WORK/tests/unit/test_eda.py
COURSE_WORK/tests/integration/test_phase_0_to_5_chain.py
COURSE_WORK/tests/integration/test_notebook_boundary.py
COURSE_WORK/docs/plan-doc/plan_detail_for_each_phase/Phase_5_EDA.md
```

Artifact mới dự kiến:

```text
COURSE_WORK/artifacts/eda/supplement
```

Không sửa:

```text
COURSE_WORK/data/raw_data/energydata_complete.csv
COURSE_WORK/artifacts/contracts
COURSE_WORK/artifacts/environment
COURSE_WORK/artifacts/acquisition
COURSE_WORK/artifacts/schema
COURSE_WORK/artifacts/temporal
COURSE_WORK/artifacts/eda/phase_5_signoff.json
COURSE_WORK/src/course_work/data/features.py
```

## 7. Implementation tuần tự

### Step 1 - Revalidate baseline

```text
Đối chiếu raw CSV SHA-256.
Đối chiếu sign-off Phase 0 đến Phase 5.
Đối chiếu notebook boundary tests.
Xác nhận Phase 6 chưa bắt đầu.
Xác nhận EDA.ipynb chỉ là reference và không trở thành dependency runtime.
```

Checkpoint:

```text
Không implementation nếu baseline đã drift ngoài thay đổi hiện có của Human.
```

### Step 2 - Bổ sung calculation API trong owner hiện có

Thêm API có trách nhiệm đơn nhất cho:

```text
Missingness summary.
Segment-aware cross-correlation profile từ lag -24 đến +24.
Sensor IQR descriptive diagnostics.
Supplement analysis assembly.
```

Yêu cầu:

```text
Không mutation input DataFrame.
Không pair dữ liệu qua continuity-segment boundary.
Không dùng IQR flags để xóa hoặc sửa giá trị.
Không tạo modeling decision.
```

Checkpoint:

```text
Chạy unit tests Phase 3–5.
So sánh fingerprint raw DataFrame trước và sau.
Kiểm tra cross-correlation bằng fixture có temporal gap.
Kiểm tra IQR output deterministic.
```

### Step 3 - Bổ sung reporting và supplement materializer

Mở rộng owner hiện có để:

```text
Render đúng bảy figure supplement.
Write đúng ba table supplement.
Tạo manifest ghi input checksum, output checksum và policy.
Tạo presentation sign-off chỉ sau khi mọi output hợp lệ.
Publish artifact atomically sau khi staging hoàn tất.
Nếu supplement sign-off đã tồn tại thì chỉ verify, không overwrite.
```

Checkpoint:

```text
Figure set đúng tuyệt đối, không thiếu và không thừa.
Table schema đúng contract.
Mọi checksum trong supplement sign-off khớp file thực tế.
EDA-v1 sign-off checksum không đổi.
```

### Step 4 - Refactor Phase 5 presentation trong CourseWork.ipynb

Giữ nguyên cell gọi:

```text
materialize_phase_5(PROJECT_ROOT)
```

Thêm lời gọi supplement materializer và chia presentation thành các subsection:

```text
5.1 Data overview and missingness
5.2 Target distribution
5.3 Target timeline and representative windows
5.4 Sampling coverage and calendar patterns
5.5 Feature distributions
5.6 Correlation ranking and top-feature relationships
5.7 Segment-aware lag relationships
5.8 Conditional target distributions
5.9 Descriptive IQR outlier diagnostics
5.10 Hypotheses, anomalies and sign-offs
```

Notebook chỉ đọc table và hiển thị image artifact. Không copy source code từ `EDA.ipynb`.

Checkpoint:

```text
Notebook JSON hợp lệ.
Phase heading order không đổi.
Không có function, loop hoặc scientific calculation trong cell.
Không có comment hoặc icon trong code cell.
Không có machine-specific path.
```

### Step 5 - Cập nhật test contracts

Bổ sung kiểm thử cho:

```text
Missingness schema và totals.
Bidirectional lag range -24 đến +24.
Cross-correlation không vượt continuity boundary.
IQR diagnostic không mutation dữ liệu.
Supplement figure/table inventory.
Supplement checksum verification.
Notebook hiển thị đủ subsection và artifact mới.
Notebook vẫn orchestration-only.
```

Checkpoint:

```text
Chạy unit tests liên quan ngay sau thay đổi.
Chạy integration Phase 0–5 ngay sau unit tests.
Không đi tiếp nếu test fail.
```

### Step 6 - Clean notebook execution

```text
Clear mọi execution count và output cũ.
Restart kernel sạch từ repository venv.
Chạy CourseWork.ipynb từ đầu đến cuối, không PYTHONPATH injection.
Chỉ ghi notebook canonical sau khi toàn bộ cell thành công.
```

Checkpoint:

```text
Execution counts liên tục và không trùng.
Không error output.
Không warning stream.
Tất cả table và figure hiển thị được.
```

### Step 7 - Final linked verification

```text
Chạy full automated test suite.
Chạy pip check.
Kiểm tra git diff --check.
Đối chiếu raw CSV hash.
Đối chiếu sign-off Phase 0 đến Phase 5.
Kiểm tra EDA-v1 artifacts không đổi.
Kiểm tra supplement sign-off và manifest.
Kiểm tra Phase 6 artifacts vẫn absent.
```

Nếu một checkpoint fail, dừng step hiện tại, ghi issue riêng, lập recovery plan và chỉ tiếp tục sau khi lỗi đã được xử lý.

## 8. Risks và controls

| Risk | Control |
|---|---|
| Lặp EDA hiện có | Ma trận coverage là allowlist; chỉ bổ sung mục `ADD` |
| Notebook trở lại processing-heavy | AST boundary test tiếp tục cấm calculation và plotting logic |
| Tải lại UCI gây phụ thuộc network | Chỉ dùng validated Phase 4 view |
| Cross-correlation ghép qua gap | Tính theo continuity segment và test fixture có gap |
| IQR bị hiểu là cleaning rule | Ghi rõ `DESCRIPTIVE_ONLY`; không tạo cleaned data |
| Leakage từ interpolation | Không tích hợp interpolation |
| Phá signed EDA-v1 | Supplement dùng namespace và sign-off riêng |
| Output matplotlib không deterministic | Giữ canonical style, fixed ordering, fixed dimensions và metadata policy |
| Notebook quá dài | Chia subsection theo nhóm nhưng chỉ display table/figure |

## 9. Alternatives rejected

```text
Copy nguyên 43 cell từ EDA.ipynb: tạo duplicate logic, hidden state và vi phạm notebook boundary.
Chạy fetch_ucirepo trong Phase 5: phá provenance của Phase 2.
Đưa plotting trực tiếp vào notebook: phá source-owned processing.
Interpolate outlier trong Phase 5: biến EDA thành preprocessing và có nguy cơ leakage.
Ghi đè EDA-v1: phá immutable sign-off đã xác minh.
Tạo package EDA mới: không cần thiết vì owner hiện có đã đúng kiến trúc.
```

## 10. Acceptance criteria

```text
Mọi nhóm EDA hợp lệ và không trùng từ EDA.ipynb xuất hiện trong CourseWork.ipynb.
Các nhóm đã có dùng lại EDA-v1 artifacts thay vì tính lại.
CourseWork.ipynb vẫn chỉ orchestration và presentation.
Không fetch dữ liệu, cleaning, interpolation hoặc feature selection trong Phase 5.
Không có scientific logic trong notebook code cells.
Không có comment hoặc icon trong notebook code cells.
Raw CSV và Phase 0-5 sign-off checksums không đổi.
Supplement tables, figures, manifest và sign-off đầy đủ.
Cross-correlation mới continuity-segment-aware.
IQR analysis chỉ descriptive và không mutation dữ liệu.
Notebook chạy sạch từ đầu đến cuối.
Toàn bộ automated tests pass.
Phase 6 vẫn chưa được bắt đầu.
```

## 11. Approval gate

Plan này đã được thay thế sau khi Human yêu cầu rõ ràng rằng EDA processing code phải được viết trực tiếp trong `CourseWork.ipynb`.
