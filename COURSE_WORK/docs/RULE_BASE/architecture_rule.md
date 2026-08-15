# QUY TẮC KIẾN TRÚC COURSE_WORK

## 1. Danh tính tài liệu

```text
Document ID: COURSE-WORK-ARCHITECTURE-v1
Repository scope: COURSE_WORK
Architecture style: Source-owned processing with orchestration-only notebook
Scientific scope: Multivariate time-series regression
Current implementation scope: Phase 0 through Phase 5
Status: ACTIVE_HUMAN_APPROVED
```

Tài liệu này quy định vị trí lưu trữ, quyền sở hữu logic, hướng dependency, vòng đời dữ liệu, vòng đời artifact, ranh giới notebook và trách nhiệm kiểm thử của `COURSE_WORK`.

Tài liệu không thay thế:

```text
COURSE_WORK/working_rule.md
COURSE_WORK/docs/RULE_BASE/rule_code.md
Phase detail tương ứng
Pre-process plan đã được Human duyệt
```

Khi có xung đột chưa được giải quyết rõ bằng rule cấp cao hơn, phải dừng thực thi và xin Human quyết định.

## 2. Mục tiêu kiến trúc

Kiến trúc phải bảo đảm:

```text
Một nơi sở hữu duy nhất cho mỗi loại logic.
Notebook không phải source of truth cho xử lý dữ liệu.
Mọi Phase có input, output và sign-off truy vết được.
Raw data bất biến.
Không có leakage giữa Train, Validation và Test.
Mọi kết quả có thể tái tạo từ config, source và artifact.
Mọi Phase chỉ chạy sau khi upstream Phase hợp lệ.
EDA không bị trộn với preprocessing hoặc model selection.
Code, artifact và report có trách nhiệm tách biệt.
```

## 3. Những điều kiến trúc không cho phép

```text
Đặt processing logic trong notebook.
Duplicate cùng một logic ở nhiều module.
Ghi đè raw data.
Tạo file hoặc thư mục không có owner.
Tạo artifact không có schema hoặc provenance.
Để Phase sau đoán lại contract của Phase trước.
Để source import notebook.
Để scientific decision phụ thuộc Test trước final gate.
Đưa interpolation hoặc data repair vào EDA một cách âm thầm.
Tạo Phase mới mà không có Phase detail và approval.
```

## 4. Canonical project root

Project root duy nhất:

```text
COURSE_WORK/
```

Python source root duy nhất:

```text
COURSE_WORK/src/course_work/
```

Notebook chính duy nhất cho coursework:

```text
COURSE_WORK/notebook_course_work/CourseWork.ipynb
```

Không tạo package hoặc notebook song song chỉ khác chữ hoa, chữ thường hoặc cách viết.

Các tên sau không canonical:

```text
COURSE_WORK/SRC/
COURSE_WORK/src/coursework/
COURSE_WORK/course_work/
COURSE_WORK/notebook_course_work/course_work.ipynb
COURSE_WORK/notebook_course_work/practice_3.ipynb
```

## 5. Cây kiến trúc canonical

```text
COURSE_WORK/
├── README.md
├── requirements.txt
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
│   ├── data_after_processing/
│   └── data_after_split/
├── artifacts/
│   ├── contracts/
│   ├── environment/
│   ├── acquisition/
│   ├── schema/
│   ├── temporal/
│   └── eda/
│       ├── tables/
│       └── figures/
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
│       │   ├── splitting.py
│       │   ├── scaling.py
│       │   ├── windows.py
│       │   └── datasets.py
│       ├── reporting/
│       │   ├── __init__.py
│       │   └── eda.py
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── artifacts.py
│       │   ├── environment.py
│       │   └── reproducibility.py
│       └── attention/
│           ├── extraction.py
│           ├── heatmaps.py
│           ├── last_query.py
│           └── head_comparison.py
├── tests/
│   ├── contracts/
│   ├── unit/
│   └── integration/
├── notebook_course_work/
│   └── CourseWork.ipynb
└── docs/
    ├── RULE_BASE/
    └── plan-doc/
```

Các package và artifact ngoài Phase 0-5 chỉ là namespace được bảo lưu. Không được triển khai logic Phase 6 trở đi trong scope refactor Phase 0-5.

## 6. Trách nhiệm thư mục cấp cao

### 6.1. `configs`

Lưu cấu hình khai báo đã được duyệt.

Quy tắc:

```text
Không chứa kết quả runtime.
Không chứa absolute path theo máy.
Không chứa secret.
Không chứa giá trị được chọn bằng Test.
Mỗi config phải có schema hoặc validator.
```

### 6.2. `data`

Lưu dữ liệu nguồn và dữ liệu dẫn xuất theo vòng đời đã duyệt.

`data` không chứa source code.

### 6.3. `artifacts`

Lưu machine-readable output của các Phase.

Artifact là bằng chứng thực thi, không phải source code và không phải cấu hình đầu vào tùy ý.

### 6.4. `src/course_work`

Lưu toàn bộ reusable processing logic.

Mọi logic được dùng bởi notebook hoặc nhiều Phase phải có owner tại đây.

### 6.5. `tests`

Lưu contract, unit và integration tests.

Notebook output không thay thế automated tests.

### 6.6. `notebook_course_work`

Lưu presentation và orchestration notebook.

Notebook không sở hữu scientific implementation.

### 6.7. `docs`

Lưu rules, issue synthesis, pre-process plans, Phase details, execution records và report đã được xác minh.

## 7. Source package ownership

### 7.1. `contracts/coursework.py`

Sở hữu Phase 0:

```text
Coursework contract schema
Contract validation
Option-registry validation
Research-question validation
Contract fingerprint
Phase 0 materialization
Phase 0 sign-off eligibility
```

Không đọc dataset và không chạy computation phụ thuộc dữ liệu.

### 7.2. `utils/environment.py`

Sở hữu phần environment của Phase 1:

```text
Interpreter inventory
Package-version inventory
Platform inventory
Working-directory validation
CUDA, MPS và CPU detection
Device selection
Tensor, autograd, module và optimizer smoke tests
Environment report
```

### 7.3. `utils/reproducibility.py`

Sở hữu:

```text
Python seed
NumPy seed
PyTorch seed
CUDA seed khi áp dụng
Deterministic policy
Randomness smoke tests
```

Không tuyên bố reproducibility tuyệt đối giữa mọi platform và package version.

### 7.4. `utils/artifacts.py`

Sở hữu:

```text
Canonical path resolution
Directory creation trong approved scope
Atomic write
JSON và CSV serialization
SHA-256 helper
Artifact fingerprint
Artifact reload verification
```

Không sở hữu schema khoa học, EDA calculation hoặc Phase decision.

### 7.5. `data/acquisition.py`

Sở hữu Phase 2:

```text
Official source identity
Acquisition method
Archive integrity
Safe extraction
Archive và CSV hashing
Minimal CSV smoke test
Dataset manifest
Source metadata
Acquisition log
DATA-v1 sign-off eligibility
```

Không preprocessing, split, scale, drop column hoặc overwrite raw CSV.

### 7.6. `data/schema.py`

Sở hữu Phase 3:

```text
Expected raw schema
Expected-versus-actual comparison
Raw dtype audit
Semantic roles
Feature groups
Units và descriptions
Null, all-null và constant audit
Numeric coercion
Non-finite audit
Timestamp parse probe
Schema fingerprint
SCHEMA-v1 sign-off eligibility
```

Không convert timestamp chính thức và không mutate raw DataFrame.

### 7.7. `data/temporal.py`

Sở hữu Phase 4:

```text
Strict timestamp parsing
Original-order audit
Delta classification
Duplicate timestamp classification
Grid alignment
Missing timestamps
Gap events
Coverage metrics
Continuity segmentation
Window-safety contract
TEMPORAL-v1 sign-off eligibility
```

Không interpolation, resampling, split hoặc window construction.

### 7.8. `data/eda.py`

Sở hữu calculation của Phase 5:

```text
Derived EDA view
Numeric summary
Target quantiles
Hourly và weekday profiles
Feature summaries
Correlation tables
High-correlation pairs
Selected-lag autocorrelation
Segment-aware cross-correlation
Segment-aware rolling statistics
Extreme-target samples
Hypothesis registry
Anomaly registry
EDA-v1 sign-off eligibility
```

Không sở hữu plotting backend và không thực hiện preprocessing.

### 7.9. `reporting/eda.py`

Sở hữu figure rendering của Phase 5:

```text
Canonical EDA figure names
Plot style
Axis labels
Figure export
Figure checksum handoff
```

Chỉ nhận validated tables hoặc validated derived views từ `data/eda.py`.

Không tự tính lại scientific tables theo logic khác.

### 7.10. Các module Phase 6 trở đi

Các file sau không được sửa trong refactor Phase 0-5:

```text
data/features.py
data/splitting.py
data/scaling.py
data/windows.py
data/datasets.py
attention/*
```

Sự tồn tại của file rỗng không được xem là Phase đã triển khai.

## 8. Hướng dependency

### 8.1. Dependency được phép

| Caller | Dependency được phép |
|---|---|
| Notebook | Public API của contracts, data, reporting và utils |
| Reporting | Validated EDA result và artifact utility |
| Data EDA | Temporal contract, schema contract và artifact utility |
| Temporal | Schema contract và artifact utility |
| Schema | Acquisition contract và artifact utility |
| Acquisition | Artifact utility và approved external libraries |
| Contracts | Standard library và schema validator được duyệt |
| Utils | Standard library và dependency tối thiểu được duyệt |
| Tests | Public API và test fixtures |

### 8.2. Dependency bị cấm

```text
src -> notebook
contracts -> data
utils -> scientific Phase modules
schema -> temporal
temporal -> EDA
reporting -> notebook
data -> notebook
Phase trước -> Phase sau
raw data -> generated artifact
Test result -> Phase 0-45 selection decision
```

Không circular import.

Nếu hai module cần import lẫn nhau, phải dừng và sửa ownership qua architecture plan.

## 9. Phase-to-module mapping

| Phase | Owner chính | Owner hỗ trợ | Notebook responsibility |
|---|---|---|---|
| 0 | `contracts/coursework.py` | `utils/artifacts.py` | Hiển thị contract và sign-off |
| 1 | `utils/environment.py` | `utils/reproducibility.py`, `utils/artifacts.py` | Gọi audit và hiển thị report |
| 2 | `data/acquisition.py` | `utils/artifacts.py` | Gọi acquisition verification và hiển thị provenance |
| 3 | `data/schema.py` | `utils/artifacts.py` | Gọi schema audit và hiển thị summary |
| 4 | `data/temporal.py` | `utils/artifacts.py` | Gọi temporal audit và hiển thị summary |
| 5 | `data/eda.py` | `reporting/eda.py`, `utils/artifacts.py` | Gọi EDA workflow và hiển thị outputs |

Không Phase nào được triển khai trong notebook cell.

## 10. Data lifecycle

### 10.1. Raw data

Canonical raw CSV:

```text
COURSE_WORK/data/raw_data/energydata_complete.csv
```

Raw contract:

```text
Immutable by convention
No overwrite
No rename columns
No row sort
No date conversion on disk
No missing-value repair
No outlier removal
No split
No scaling
No feature removal
```

Protected Phase 0-5 baseline SHA-256:

```text
2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d
```

Mọi Phase đọc data phải xác minh lại hash.

### 10.2. Derived audit views

Phase 3 chỉ tạo parse probe.

Phase 4 tạo deep-copy temporal view.

Phase 5 tạo `df_eda` từ validated temporal view.

Derived view phải giữ:

```text
raw_row_index
timestamp lineage
continuity_segment_id khi có
dataset revision
upstream artifact fingerprints
```

### 10.3. Processed và split data

`data_after_processing` và `data_after_split` không được dùng trong Phase 0-5.

Phase 6-10 mới được quyền ghi các output tương ứng sau approval riêng.

## 11. Configuration ownership

Phase 0 contract canonical:

```text
COURSE_WORK/configs/base/coursework_contract.json
```

Quy tắc:

```text
Human-readable và machine-readable values phải nhất quán.
Contract phải được validate trước khi materialize artifact.
Artifact copy phải có checksum.
Không sửa contract sau experiment mà không có protocol amendment.
Không lưu runtime result trong config.
```

## 12. Artifact lifecycle

### 12.1. Trạng thái artifact

```text
DRAFT
VALIDATED
SIGNED_OFF
SUPERSEDED
FAILED
```

Chỉ artifact `SIGNED_OFF` mới được Phase sau sử dụng.

### 12.2. Field provenance tối thiểu

Mỗi manifest hoặc sign-off phải có khi phù hợp:

```text
artifact_version
phase_id
created_at
environment_id
dataset_revision
input_paths
input_checksums
output_paths
output_checksums
config_fingerprint
status
warnings
discrepancies
```

Field không áp dụng phải là `null` hoặc được loại theo schema. Không điền giá trị giả.

### 12.3. Atomicity

Machine-readable artifact phải được ghi atomically và reload để verify trước sign-off.

Không overwrite artifact đã sign-off.

## 13. Artifact ownership Phase 0-5

```text
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

artifacts/eda
-> Phase 5 tables, figures, manifest, anomalies và sign-off
```

Không lưu checkpoint, model hoặc prediction trong các Phase 0-5 roots.

## 14. Notebook boundary

### 14.1. Nội dung được phép

```text
Markdown title
Markdown scientific explanation
Import public APIs
Resolve an approved config name
Call a Phase public API
Display returned summary
Display saved figure
Display sign-off
```

### 14.2. Nội dung bị cấm

```text
def
class
pd.read_csv
fetch_ucirepo
pd.to_datetime
DataFrame mutation
Feature derivation
Hash implementation
Schema comparison implementation
Temporal delta implementation
Gap implementation
Autocorrelation implementation
Rolling implementation
Interpolation
Outlier cleaning
Plot-construction implementation
Artifact serialization
Path-discovery logic
Exception-repair logic
```

### 14.3. Cell order

Notebook phải có section tuần tự:

```text
Phase 0
Phase 1
Phase 2
Phase 3
Phase 4
Phase 5
```

Notebook không được dựa vào hidden kernel state.

Mỗi code cell phải chạy được sau restart kernel theo đúng thứ tự.

## 15. EDA boundary

Phase 5 được phép:

```text
Mô tả target và features.
Tạo EDA-only temporal columns trong df_eda.
Visualize temporal pattern.
Tạo descriptive correlation.
Tạo hypothesis cho Phase sau.
```

Phase 5 không được:

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
```

Lag và rolling calculation phải continuity-segment-aware khi temporal audit phát hiện gap.

## 16. Test architecture

### 16.1. Contract tests

Lưu tại:

```text
COURSE_WORK/tests/contracts/
```

Kiểm tra:

```text
Phase contract schema
Artifact schema
Sign-off schema
Phase-gate dependency
Notebook boundary
```

### 16.2. Unit tests

Lưu tại:

```text
COURSE_WORK/tests/unit/
```

Mỗi public calculation phải có normal, boundary và failure tests.

### 16.3. Integration tests

Lưu tại:

```text
COURSE_WORK/tests/integration/
```

Kiểm tra:

```text
Phase 0 -> Phase 1
Phase 1 -> Phase 2
Phase 2 -> Phase 3
Phase 3 -> Phase 4
Phase 4 -> Phase 5
Raw checksum preservation
Artifact reload
Notebook orchestration boundary
```

## 17. Phase gate

Phase N chỉ được chạy khi:

```text
Phase N-1 sign-off cho phép tiếp tục.
Input artifact tồn tại.
Input checksum khớp.
Input schema hợp lệ.
Không có unresolved critical discrepancy.
Pre-process plan đã được duyệt.
```

Nếu một điều kiện không đạt:

```text
STOP
không tạo sign-off PASS
không chạy Phase tiếp theo
ghi issue và correction plan
```

## 18. Sign-off rules

Sign-off phải được tạo từ validation result, không được hard-code `PASS` trước khi chạy checks.

Sign-off tối thiểu chứa:

```text
phase_id
phase_version
status
input_artifacts
input_checksums
output_artifacts
output_checksums
tests
warnings
discrepancies
created_at
```

Allowed status phụ thuộc Phase detail:

```text
PASS
PASS_WITH_WARNING khi Phase cho phép
FAIL
BLOCKED
```

## 19. Code style

```text
Module và function dùng snake_case.
Class dùng PascalCase.
Constant dùng UPPER_SNAKE_CASE.
Path dùng pathlib.Path.
Public API có type annotations.
Function có responsibility đơn nhất.
Không dùng inplace mutation cho canonical inputs.
Không hard-code absolute path.
Không dùng bare except.
Không swallow exception.
Không fabricate result.
Không thêm comment hoặc icon vào code.
Không thêm dependency ngoài approved scope.
```

Scientific table phải giữ full precision. Làm tròn chỉ thuộc presentation.

## 20. Naming conventions

### 20.1. Python

```text
module_name.py
test_<module>_<behavior>.py
```

### 20.2. Artifacts

Tên artifact phải theo Phase detail nếu đã được quy định.

Không dùng:

```text
temp
new
final_final
result2
test1
```

### 20.3. Versions

```text
COURSEWORK-CONTRACT-v1
ENV-v1
DATA-v1
SCHEMA-v1
TEMPORAL-v1
EDA-v1
```

Không tái sử dụng version ID cho nội dung khác.

## 21. Reproducibility rules

```text
Environment phải được ghi trước data-dependent computation.
Representative EDA windows phải deterministic.
Random operation phải dùng approved seed.
Notebook phải chạy sạch từ đầu đến cuối.
Execution counts phải tăng tuần tự.
Không có unexecuted dependency cell.
Không dùng output cũ từ runtime khác làm bằng chứng.
```

## 22. Security and integrity

```text
Không ghi secret vào source, config hoặc artifact.
Không log toàn bộ sensitive environment variables.
Không extract archive bằng path không an toàn.
Không overwrite file ngoài approved root.
Không dùng unresolved user input làm file path.
Không chạy downloaded code từ dataset archive.
```

## 23. Git and ignore policy

Source dưới:

```text
COURSE_WORK/src/course_work/data/
```

phải trackable.

Pattern ignore cho data assets không được vô tình ignore Python source chỉ vì thư mục có tên `data`.

Git không theo dõi thư mục rỗng. Chỉ tạo package khi có `__init__.py` hoặc implementation thực sự.

Không dùng `git add .` cho workflow Phase.

## 24. Change-control gate

Cần architecture amendment và Human approval trước khi:

```text
Tạo package mới ngoài cây canonical.
Đổi canonical notebook.
Đổi owner module.
Đổi dependency direction.
Đổi data lifecycle.
Đổi artifact root.
Đổi config hierarchy.
Đổi Phase-to-module mapping.
Cho phép processing logic trong notebook.
Thêm Phase mới.
```

Không sửa architecture rule trong im lặng để hợp thức hóa code đã viết sai.

## 25. Transition state

Tại thời điểm draft này:

```text
CourseWork.ipynb còn processing logic và stale execution state.
Phase 0-5 chưa có sign-off.
requirements.txt rỗng.
Source modules hiện tại là scaffold rỗng.
artifacts chưa tồn tại.
.gitignore đang ignore nhầm src/course_work/data.
Phase 6 chưa triển khai.
```

Transition phải theo plan:

```text
CW-REFACTOR-0005-001
```

Không được bỏ qua Phase gate trong quá trình chuyển đổi.

## 26. Architecture validation checklist

```text
[x] Canonical root rõ ràng.
[x] Canonical notebook rõ ràng.
[x] Source ownership rõ ràng.
[x] Phase 0-5 mapping đầy đủ.
[x] Dependency direction rõ ràng.
[x] Raw data contract rõ ràng.
[x] Derived-view contract rõ ràng.
[x] Configuration ownership rõ ràng.
[x] Artifact lifecycle rõ ràng.
[x] Notebook boundary rõ ràng.
[x] EDA boundary rõ ràng.
[x] Test ownership rõ ràng.
[x] Phase gate rõ ràng.
[x] Sign-off contract rõ ràng.
[x] Git tracking rule rõ ràng.
[x] Change-control gate rõ ràng.
[x] Không triển khai Phase 6.
```

## 27. Activation gate

Tài liệu này chỉ trở thành architecture contract đang hoạt động sau khi Human đọc và xác nhận rõ.

Trạng thái trước approval:

```text
ARCHITECTURE_RULE_CREATED=true
ARCHITECTURE_RULE_VALIDATED=false
ARCHITECTURE_RULE_APPROVED=false
SOURCE_REFACTOR_ALLOWED=false
PHASE_0_IMPLEMENTATION_ALLOWED=false
```

Sau approval, Step A2 và Phase 0 mới được phép bắt đầu theo plan `CW-REFACTOR-0005-001`.

Trạng thái hiện hành sau Human approval:

```text
ARCHITECTURE_RULE_CREATED=true
ARCHITECTURE_RULE_VALIDATED=true
ARCHITECTURE_RULE_APPROVED=true
SOURCE_REFACTOR_ALLOWED=true
PHASE_0_IMPLEMENTATION_ALLOWED=true
```
