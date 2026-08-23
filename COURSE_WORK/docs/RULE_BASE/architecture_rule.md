# QUY TẮC KIẾN TRÚC COURSE_WORK

## 1. Danh tính tài liệu

```text
Document ID: COURSE-WORK-ARCHITECTURE-v1
Repository scope: COURSE_WORK
Architecture style: Source-owned processing with Human-approved direct Phase 5 EDA exception
Scientific scope: Multivariate time-series regression
Current implementation scope: Phase 0 through Phase 34
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
Notebook không phải source of truth cho canonical data processing ngoài direct exploratory Phase 5 exception.
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
Đặt processing logic trong notebook ngoài direct exploratory Phase 5 exception đã được Human duyệt.
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
│       ├── tables/
│       └── figures/
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
│       ├── models/
│       │   ├── __init__.py
│       │   ├── lstm_regressor.py
│       │   ├── transformer_regressor.py
│       │   ├── transformer_encoder_layer.py
│       │   └── positional_encoding.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── engine.py
│       │   └── engine_materialize.py
│       ├── sanity/
│       │   ├── __init__.py
│       │   └── forward_sanity.py
│       ├── diagnostics/
│       │   ├── __init__.py
│       │   ├── learning_curves.py
│       │   └── learning_diagnostics.py
│       ├── sweeps/
│       │   ├── d_model.py
│       │   ├── dropout.py
│       │   ├── heads.py
│       │   ├── sweep_results.py
│       │   └── weight_decay.py
│       ├── evaluation/
│       │   └── metrics.py
│       ├── experiments/
│       │   ├── registry.py
│       │   ├── phase_execution.py
│       │   └── sweep_recovery.py
│       ├── baselines/
│       │   ├── persistence.py
│       │   ├── lstm_baseline.py
│       │   └── transformer_b0.py
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── artifacts.py
│       │   ├── environment.py
│       │   └── reproducibility.py
│       └── attention/
│           ├── extraction.py
│           ├── heatmaps.py
│           ├── last_query.py
│           ├── head_comparison.py
│           ├── mapping.py
│           └── verification.py
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

Các package và artifact ngoài Phase 0-34 chỉ là namespace được bảo lưu. Không được triển khai logic Phase 35 trở đi nếu chưa có Human approval riêng.

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

Notebook không sở hữu canonical scientific implementation.

Ngoại lệ được Human duyệt cho Phase 6 EDA:

```text
CourseWork.ipynb được chứa direct descriptive EDA calculation và plotting.
Direct EDA phải dùng validated Phase 4 view.
Direct EDA không được fetch dataset hoặc ghi artifact.
Direct EDA không được thay đổi raw data hoặc signed outputs.
DataFrame smoothing chỉ được tồn tại trên deep copy để minh họa.
Smoothed demonstration không được dùng bởi Phase 6 hoặc modeling.
Ngoại lệ không áp dụng cho split, scaling, windowing, training hoặc evaluation.
```

### 6.7. `docs`

Lưu rules, issue synthesis, pre-process plans, Phase details, execution records và report đã được xác minh.

### 6.8. `pyproject.toml`

Sở hữu cấu hình packaging của canonical Python source root:

```text
COURSE_WORK/src/course_work
```

Quy tắc:

```text
Notebook và test import package qua environment đã cài project.
Không chèn sys.path hoặc PYTHONPATH bootstrap vào notebook.
Editable install không thay đổi scientific dependency contract.
Không khai báo dependency trùng với requirements.txt.
Không package namespace bảo lưu chưa có __init__.py.
```

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

### 7.9.1. `reporting/phase_summary.py`

Sở hữu presentation layer dùng chung cho Phase 0-33:

```text
Đọc canonical machine-readable artifacts đã được materialize
Tạo processing log đầy đủ độc lập với notebook presentation
Áp dụng presentation allowlist riêng cho từng Phase
Chỉ render bảng hoặc visualization trực tiếp phục vụ quyết định của Phase
Ghi một processing log JSON atomically cho mỗi Phase dưới docs/save_log_in_processing
Render HTML/CSS cục bộ, không dùng JavaScript hoặc external resource
Escape mọi artifact value trước khi đưa vào HTML
Không render warnings, discrepancies, checksum, fingerprint, source artifacts hoặc technical lineage
```

Không tính lại scientific result, không thay đổi canonical artifact và không thay thế sign-off.

Notebook presentation bắt buộc tuân thủ:

```text
Header chỉ gồm Phase, artifact version và status
Phase 1 hiển thị Environment overview và Core package versions
Phase 2 hiển thị Dataset overview
Phase 3 hiển thị Schema overview và Critical schema checks
Phase 4 hiển thị Temporal coverage và Critical temporal checks
Phase 5 chỉ hiển thị Chronological membership và Split allocation
Phase 6 chỉ hiển thị header cùng các output EDA 5.1-5.14 đã được duyệt với TRAIN-only scope
Phase 7 hiển thị Feature overview và Engineered feature registry với TRAIN-only scope
Phase 8 chỉ hiển thị Feature-set registry và không hiển thị fingerprint với TRAIN-only scope
Phase 9 chỉ hiển thị X scaler bundles và Target scaling options
Phase 10 chỉ hiển thị Window contract và Common target population
Phase 11 chỉ hiển thị Dataset population và Loader policy
Phase 12 chỉ hiển thị Metric registry và Evaluation policy
Phase 13 chỉ hiển thị Registry state và Core safeguards
Phase 14 chỉ hiển thị Validation performance và Baseline contract
```

Phase 15 hiển thị model contract và tensor-interface verification.
Phase 16 hiển thị Transformer contract và tensor-interface verification.
Phase 17 hiển thị attention verification contract và kiểm tra attention weights.
Phase 18 hiển thị forward-sanity matrix và kết quả kiểm tra hữu hạn.
Phase 19 hiển thị training-engine contract và smoke-test result.
Phase 20 hiển thị LSTM Validation metrics và learning curves.
Phase 21 hiển thị Transformer B0 Validation metrics và learning curves.
Phase 22 hiển thị diagnostic findings và learning-curve figures.
Phase 23-33 hiển thị phase state, resolved action, prerequisite state, expected conditions, verified conditions, missing conditions và bảng Validation comparison đã xác minh.

Phase mới phải khai báo presentation allowlist trước khi triển khai. Training phase chỉ hiển thị learning curves và metrics chính. Evaluation phase chỉ hiển thị model comparison và error plots cần thiết. Attention phase chỉ hiển thị heatmaps và diễn giải trực tiếp liên quan. Mọi technical detail vẫn phải được giữ đầy đủ trong processing log JSON.

### 7.10. `data/features.py`

Sở hữu toàn bộ canonical Phase 6 feature engineering:

```text
Verify signed DATA-v1, SCHEMA-v1, TEMPORAL-v1 và EDA-v1 inputs
Build deterministic FEATURES-v1 view
Create hour_sin, hour_cos, dow_sin, dow_cos và weekend
Preserve raw values, target, timestamps, row lineage và continuity segments
Create feature registry, lineage, availability và leakage audits
Write the derived CSV, manifest, checksum, discrepancy log và Phase 6 sign-off
```

Không sở hữu:

```text
Final feature-set selection
Chronological split
Scaling hoặc imputation
Manual lag hoặc rolling model features
Window construction
Training
```

### 7.11. `data/feature_sets.py`

Sở hữu toàn bộ canonical Phase 7 feature-set variants:

```text
Verify signed FEATURES-v1 inputs và checksums
Define immutable ordered feature components
Build FS0/FS1/FS2 kết hợp TF0/TF1
Validate metadata, historical-target, random-control và time-feature isolation
Validate numeric compatibility, missingness và finite values
Compute deterministic ordered-feature fingerprints
Write FEATURESETS-v1 registry, lineage, audits, manifest và sign-off
Expose defensive-copy feature-list lookup
```

Không sở hữu:

```text
Feature value transformation
Feature-set winner selection
Chronological split
Scaling hoặc imputation
Window construction
Training
```

### 7.12. `data/splitting.py`

Sở hữu toàn bộ canonical Phase 8 chronological split:

```text
Verify signed FEATURES-v1, FEATURESETS-v1 và TEMPORAL-v1 inputs
Compute deterministic floor-based 70/15/15 boundaries
Assign TRAIN, VALIDATION và TEST row membership
Preserve one full master timeline for WB0 context carry-over
Prepare WB1 strict-isolation metadata for downstream comparison
Audit chronology, coverage, disjointness và Test firewall
Compute per-split và global membership fingerprints
Create Train/Validation-only descriptive diagnostics
Create timestamp-only split timeline
Write SPLIT-v1 manifest, artifacts và sign-off
```

Không sở hữu:

```text
Scaling hoặc imputation
Feature-set winner selection
Window construction
DataLoader construction
Training hoặc evaluation metrics
Detailed Test distribution analysis trước Phase 47
```

### 7.13. `data/scaling.py`

Sở hữu toàn bộ canonical Phase 9 Train-only scaling:

```text
Verify signed FEATURES-v1, FEATURESETS-v1 và SPLIT-v1 inputs
Fit sáu variant-specific X StandardScaler bundles chỉ trên TRAIN rows
Scale continuous channels và giữ cyclical/binary channels pass-through
Fit một YS1 target scaler chỉ trên TRAIN target period
Expose YS0 identity cùng target inverse-transform utility
Bind scaler với feature order, feature fingerprint và split fingerprint
Audit Train standardization, Validation transform và structural-only Test transform
Serialize trusted local scaler artifacts, statistics, checksums, manifest và sign-off
```

Không sở hữu:

```text
Window construction
DataLoader construction
Feature-set hoặc target-scaling winner selection
Model training hoặc metrics
Detailed Test distribution inspection
```

### 7.14. `data/windows.py`

Sở hữu toàn bộ canonical Phase 10 window construction:

```text
Verify signed TEMPORAL-v1, FEATURES-v1, FEATURESETS-v1, SPLIT-v1 và SCALING-v1 inputs
Build deterministic L36, L72 và L144 native window indices cho H1
Validate timestamp cadence, continuity segments, target exclusion và sequence direction
Assign sample split bằng target timestamp
Register WB0 context carry-over và WB1 strict-isolation eligibility
Lock WINDOWPOP-v1 common target population cho controlled comparisons
Transform frozen SCALING-v1 feature timelines và materialize deterministic probes lazily
Preserve Test target firewall và không export target values
Write window index, population, audits, fingerprints, manifest, README và Phase 10 sign-off
```

Không sở hữu:

```text
PyTorch Dataset hoặc DataLoader
Model training hoặc evaluation metrics
Lookback winner selection
Full 3D tensor persistence
Test target outcome analysis trước Phase 47
```

### 7.15. `data/datasets.py`

Sở hữu toàn bộ canonical Phase 11 Dataset và DataLoader contract:

```text
Verify signed FEATURESETS-v1, SPLIT-v1, SCALING-v1, WINDOWS-v1 và WINDOWPOP-v1 inputs
Build map-style SequenceWindowDataset bằng lazy slicing từ read-only float32 feature timeline
Bind Dataset với variant, lookback, target option, window fingerprint và population fingerprint
Expose TRAIN, VALIDATION, TEST_LOCKED và explicit Phase 47 TEST_EVALUATION target access modes
Return batch-first CPU tensors với stable int64 sample_idx
Build split-specific DataLoaders cho B32 và B64
Shuffle TRAIN reproducibly và giữ VALIDATION/TEST chronological
Use drop_last false, separate split generators và canonical worker seed utility
Apply CUDA-only pin-memory policy và keep device transfer outside Dataset
Audit batch shapes, dtypes, coverage, ordering, reproducibility và Test firewall
Write registries, audits, device policy, manifest, README và Phase 11 sign-off
```

Không sở hữu:

```text
Window construction hoặc scaler fitting
Sample population selection
Model training hoặc metric calculation
Batch-size winner selection
Serialized Dataset, DataLoader hoặc full 3D tensors
Test target evaluation trước explicit Phase 47 gate
```

### 7.16. `evaluation/metrics.py`

Sở hữu toàn bộ canonical Phase 12 shared metric contract:

```text
Verify signed SCALING-v1, WINDOWS-v1, WINDOWPOP-v1 và DATALOADERS-v1 inputs
Normalize single-output arrays có shape N hoặc N x 1 về NumPy float64
Convert YS0 identity và inverse-transform YS1 bằng frozen Train-only target scaler
Compute MAE Wh, RMSE Wh và R² trên full aligned split population một lần
Preserve negative R² và explicit undefined states cho constant target hoặc N nhỏ hơn 2
Enforce residual bằng actual trừ prediction
Enforce sample index uniqueness, population coverage và chronological canonical order
Aggregate epoch loss theo sample count
Compare baseline và model chỉ khi contract, split, population, unit, horizon và sample count khớp
Enforce FINAL_TEST mode cùng model lock id trước mọi Test metric
Write metric contract, registry, schemas, audits, reference examples, manifest, README và Phase 12 sign-off
```

Không sở hữu:

```text
Model training hoặc model selection run
Prediction generation
Persistence baseline implementation
Test prediction hoặc Test target materialization trong Phase 12
Batch-mean RMSE hoặc batch-mean R² aggregation
Prediction clipping hoặc rounding trước metric
```

### 7.17. `experiments/registry.py`

Sở hữu toàn bộ canonical Phase 13 experiment registry contract:

```text
Verify signed contracts từ ENV-v1 đến METRICS-v1
Canonicalize nested run config và tạo deterministic SHA-256 config fingerprint
Validate data, model, training, reproducibility, runtime và upstream lineage fields
Allocate unique run ID theo model family, experiment family, sequence và config hash
Require canonical rerun reason cho duplicate config
Enforce parent-child lineage và final-model-lock references
Control REGISTERED, RUNNING, COMPLETED, FAILED, CANCELLED, INVALIDATED và ARCHIVED transitions
Preserve completed config immutability và failed-run evidence
Register project-relative artifacts với checksum và file size
Register METRICS-v1 rows với unit, sample count và population guards
Reject development Test targets và Test metrics
Allow final Test chỉ với FINAL_TEST family, execution type, lock id và authorization
Validate one-factor sweep consistency và run comparison compatibility
Write canonical JSONL, derived CSV views, families, audits, manifest, index, README và Phase 13 sign-off
Initialize Phase 13 production registry without fabricated runs or results
Allow Phase 14+ owners to register real runs through public lifecycle APIs
Require predictions and metrics before completing Persistence evaluation
Allow task-level Persistence config với null feature, scaler, DataLoader và seed fields
```

Không sở hữu:

```text
Model training hoặc evaluation execution
Persistence, LSTM hoặc Transformer implementation
Winner selection
Checkpoint, prediction hoặc actual metric generation
Test authorization trước Phase 47
Manual registry edits hoặc Excel source of truth
```

### 7.18. `baselines/persistence.py`

Sở hữu toàn bộ canonical Phase 14 Persistence baseline:

```text
Verify signed WINDOWS-v1, WINDOWPOP-v1, DATALOADERS-v1, METRICS-v1 và EXPERIMENTS-v1 inputs
Lock PERSISTENCE-v1 formula y_hat(t+1) = y(t)
Use the full L144-anchored common Validation population under WB0
Read raw Appliances values only through the final required Validation row
Keep Test target values and Test metrics inaccessible
Validate source-target H1 alignment, ten-minute cadence, continuity and population order
Generate raw-Wh predictions directly from input-end target history
Compute MAE Wh, RMSE Wh, R² and residuals through METRICS-v1
Register and start one PERSISTENCE_BASELINE evaluation run before metric computation
Register prediction, metric, audit and supporting artifacts before completion
Complete the run without training, scaler, DataLoader, seed, optimizer or checkpoint
Write manifest, summary, predictions, metrics, audits, discrepancies, README và Phase 14 sign-off
Expose materialize_phase_14 as the only notebook orchestration API
```

Không sở hữu:

```text
Learned model implementation
Feature-set hoặc target-scaling selection
Training, optimizer, checkpoint hoặc hyperparameter sweep
Test evaluation trước Phase 47
Moving-average hoặc seasonal baseline extension
```

### 7.19. `models/lstm_regressor.py`

Sở hữu Phase 15 LSTM implementation contract, model construction, interface audit và signed implementation artifact.

### 7.20. `models/transformer_regressor.py`

Sở hữu Phase 16 Transformer implementation contract, positional encoding integration, encoder construction, regression interface audit và signed implementation artifact.

### 7.21. `attention/verification.py`

Sở hữu Phase 17 attention-aware encoder verification, attention tensor contract, masking verification và signed verification artifact.

### 7.22. `sanity/forward_sanity.py`

Sở hữu Phase 18 forward-pass sanity matrix cho LSTM và Transformer, shape, dtype, finite-value, backward compatibility và signed sanity artifact.

### 7.23. `training/engine.py` và `training/engine_materialize.py`

Sở hữu Phase 19 training engine, epoch aggregation, Validation-only monitoring, checkpoint contract, early stopping, reproducibility handoff và signed engine artifact.

### 7.24. `baselines/lstm_baseline.py`

Sở hữu Phase 20 canonical LSTM baseline run, run-registry lifecycle, Validation metrics, checkpoint lineage, learning history và signed baseline artifact.

### 7.25. `baselines/transformer_b0.py`

Sở hữu Phase 21 canonical Transformer B0 run, run-registry lifecycle, Validation metrics, checkpoint lineage, learning history và signed baseline artifact.

### 7.26. `diagnostics/learning_diagnostics.py` và `diagnostics/learning_curves.py`

Sở hữu Phase 22 learning-curve diagnostics, validated training-history comparison, diagnostic findings, saved figures và signed diagnostics artifact.

### 7.27. `sweeps/sweep_results.py`

Sở hữu validation contract dùng chung cho Phase 23-34 sweep results, condition completeness, Validation-only winner selection, manifest, winner handoff, reference update và signed sweep artifacts.

Không được tái sử dụng sign-off nếu output path thiếu, checksum sai, condition thiếu hoặc run configuration không khớp condition.

### 7.28. `experiments/phase_execution.py`

Sở hữu selective phase inspection và resume orchestration:

```text
Phase dependency registry
Expected condition registry
Canonical evidence inspection
State classification
Smallest-safe-action resolution
Missing-only dispatch
Derived processing-log refresh
Notebook-ready persistent presentation payload
```

Module này không sở hữu training logic, scientific metric calculation hoặc HTML rendering. Module không được gọi materializer của Phase trước chỉ để kiểm tra Phase được chọn.

### 7.29. `sweeps/weight_decay.py`

Sở hữu Phase 31 S9 weight-decay contract, Phase 30 handoff preflight, WD0/WD1/WD2 mapping, frozen AdamW parameter-group policy boundary, WD1 exact-reference reuse gate, Phase 31 condition preparation và Test firewall.

Module không được hard-code S8 winner, không được xem processing log là canonical evidence, không được thêm explicit L2 loss, không được đổi optimizer group topology và không được chạy condition khi Phase 30 gate không hợp lệ.

### 7.30. `experiments/sweep_recovery.py`

Sở hữu read-only recovery audit từ Phase 22 tới target Phase, current-runtime identity inspection, earliest-invalid-dependency resolution, minimal execution-set calculation và dependency-aware recovery proposal.

Module không sở hữu training, winner selection, artifact finalization, HTML rendering hoặc scientific metric calculation. Audit mode không được ghi scientific artifact, thay đổi notebook output hoặc xem processing log là scientific evidence.

### 7.31. `sweeps/dropout.py`

Sở hữu Phase 32 S10 dropout contract, Phase 31 handoff preflight, DR01/DR02/DR03 mapping, frozen dropout-site scope, DR01 exact-reference reuse gate, DR02/DR03 fresh-run preparation, train/eval dropout semantics và Test firewall.

Module không được hard-code S9 winner, không được xem processing log là canonical evidence, không được thêm dropout site mới, không được thay đổi capacity hoặc optimizer contract và không được chạy condition khi Phase 31 gate không hợp lệ.

### 7.32. `sweeps/d_model.py`

Sở hữu Phase 33 S11 d_model contract, Phase 32 handoff preflight, D32/D64 mapping, H4/N2/F128 frozen-capacity boundary, D64 exact-reference reuse gate, D32 fresh-run preparation, model-geometry audit, capacity-efficiency context và Test firewall.

Module không được hard-code S10 winner, không được xem processing log là canonical evidence, không được bù thay đổi d_model bằng cách thay đổi head count, layer count hoặc FFN dimension và không được chạy condition khi Phase 32 gate không hợp lệ.

### 7.33. `sweeps/heads.py`

Sở hữu Phase 34 S12 head-count contract, Phase 33 handoff preflight, H2/H4 mapping, dynamic selected `d_model`, head-dimension và divisibility audit, parameter-schema equality, H4 exact-reference reuse gate, H2 fresh-run preparation, attention API sanity và Test firewall.

Module không được hard-code S11 winner, không được xem processing log là canonical evidence, không được thay `d_model`, layer count hoặc FFN dimension để bù head geometry, không được warm-start H2 từ H4 và không được chạy condition khi Phase 33 gate không hợp lệ.

### 7.34. Các module Phase 35 trở đi

Phase 35 trở đi nằm ngoài current implementation scope. Không được triển khai nếu chưa có Phase detail, pre-process plan và Human approval riêng.

## 8. Hướng dependency

### 8.1. Dependency được phép

| Caller | Dependency được phép |
|---|---|
| Notebook | Public API của contracts, data, evaluation, experiments, reporting và utils |
| Selective phase execution | Approved Phase detail, experiment registry, canonical run artifacts, phase manifests, sign-offs, checksums và artifact utility |
| Sweep recovery audit | Signed environment evidence, current runtime identity, Phase 22 canonical evidence, selective phase inspection và approved Phase 23-34 registries |
| Sweep results | Approved condition registry, experiment registry, Validation metrics, upstream reference update và artifact utility |
| Learning diagnostics | Phase 20-21 training histories, Validation metrics và artifact utility |
| LSTM và Transformer baselines | Phase 15-19 contracts, Dataset/DataLoader, shared metrics, experiment registry, training engine và artifact utility |
| Training engine | Dataset/DataLoader public API, model interfaces, shared metrics, reproducibility và artifact utility |
| Forward sanity | Model public interfaces, attention verification và artifact utility |
| Attention verification | Transformer encoder layer, attention mapping và artifact utility |
| Model implementations | Frozen feature dimensions, model contract config và artifact utility |
| Reporting | Validated Phase 0-33 artifacts, validated phase-state payload và artifact utility |
| Weight-decay sweep | Validated Phase 30 winner, reference update, sign-off, experiment registry, frozen training contract, selective execution gate và artifact utility |
| Dropout sweep | Validated Phase 31 winner, reference update, sign-off, experiment registry, frozen Transformer and training contracts, selective execution gate và artifact utility |
| d_model sweep | Validated Phase 32 winner, reference update, sign-off, experiment registry, frozen Transformer and training contracts, selective execution gate và artifact utility |
| Persistence baseline | Raw Appliances prefix, WINDOWS-v1, WINDOWPOP-v1, METRICS-v1, EXPERIMENTS-v1 và artifact utility |
| Experiment registry | Validated Phase 0-12 artifacts, shared metrics contract và artifact utility |
| Shared metrics | Frozen target scaler, canonical window population, artifact utility và approved metric libraries |
| Datasets | Feature view, feature-set registry, frozen scaler bundles, WINDOWPOP-v1 index và reproducibility utility |
| Windows | Feature view, feature-set registry, split membership, frozen scaler bundles và artifact utility |
| Scaling | Feature view, feature-set registry, split membership và artifact utility |
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
Phase trước -> Phase sau, ngoại trừ explicit validation dependency được architecture phê duyệt
raw data -> generated artifact
Test result -> Phase 0-45 selection decision
Selective phase execution -> prerequisite materializer
Reporting -> training engine
Processing log -> canonical scientific decision
```

Không circular import.

Nếu hai module cần import lẫn nhau, phải dừng và sửa ownership qua architecture plan.

## 9. Phase-to-module mapping

| Phase | Owner chính | Owner hỗ trợ | Notebook responsibility |
|---|---|---|---|
| 0 | `contracts/coursework.py` | `utils/artifacts.py`, `reporting/phase_summary.py` | Contract nội bộ, không hiển thị trong notebook |
| 1 | `utils/environment.py` | `utils/reproducibility.py`, `utils/artifacts.py`, `reporting/phase_summary.py` | Gọi audit và hiển thị HTML summary |
| 2 | `data/acquisition.py` | `utils/artifacts.py`, `reporting/phase_summary.py` | Gọi acquisition verification và hiển thị HTML summary |
| 3 | `data/schema.py` | `utils/artifacts.py`, `reporting/phase_summary.py` | Gọi schema audit và hiển thị HTML summary |
| 4 | `data/temporal.py` | `utils/artifacts.py`, `reporting/phase_summary.py` | Gọi temporal audit và hiển thị HTML summary |
| 5 | `data/splitting.py` | `data/features.py`, `data/feature_sets.py`, `utils/artifacts.py`, `reporting/phase_summary.py` | Gọi Phase 5 public API và hiển thị HTML summary |
| 6 | `data/eda.py` | `reporting/eda.py`, `reporting/phase_summary.py`, `utils/artifacts.py`, Human-approved direct notebook EDA | Gọi EDA workflow với TRAIN-only scope, thực hiện descriptive supplement và hiển thị outputs |
| 7 | `data/features.py` | `utils/artifacts.py`, `reporting/phase_summary.py` | Gọi Phase 7 public API với TRAIN-only scope và hiển thị HTML summary |
| 8 | `data/feature_sets.py` | `data/features.py`, `utils/artifacts.py`, `reporting/phase_summary.py` | Gọi Phase 8 public API với TRAIN-only scope và hiển thị HTML summary |
| 9 | `data/scaling.py` | `data/features.py`, `data/feature_sets.py`, `data/splitting.py`, `utils/artifacts.py`, `reporting/phase_summary.py` | Gọi Phase 9 public API và hiển thị HTML summary |
| 10 | `data/windows.py` | `data/features.py`, `data/feature_sets.py`, `data/splitting.py`, `data/scaling.py`, `utils/artifacts.py`, `reporting/phase_summary.py` | Gọi Phase 10 public API và hiển thị minimal HTML summary |
| 11 | `data/datasets.py` | `data/features.py`, `data/feature_sets.py`, `data/scaling.py`, `data/windows.py`, `utils/reproducibility.py`, `utils/artifacts.py`, `reporting/phase_summary.py` | Gọi Phase 11 public API và chỉ hiển thị Dataset population cùng Loader policy |
| 12 | `evaluation/metrics.py` | `data/scaling.py`, `data/windows.py`, `data/datasets.py`, `utils/artifacts.py`, `reporting/phase_summary.py` | Gọi Phase 12 public API và chỉ hiển thị Metric registry cùng Evaluation policy |
| 13 | `experiments/registry.py` | `evaluation/metrics.py`, Phase 0-12 artifacts, `utils/artifacts.py`, `reporting/phase_summary.py` | Gọi Phase 13 public API và chỉ hiển thị Registry state cùng Core safeguards |
| 14 | `baselines/persistence.py` | `data/windows.py`, `evaluation/metrics.py`, `experiments/registry.py`, `utils/artifacts.py`, `reporting/phase_summary.py` | Gọi Phase 14 public API và chỉ hiển thị Validation performance cùng Baseline contract |
| 15 | `models/lstm_regressor.py` | `data/datasets.py`, `utils/artifacts.py`, `reporting/phase_summary.py` | Gọi Phase 15 public API và hiển thị implementation contract |
| 16 | `models/transformer_regressor.py` | `models/positional_encoding.py`, `models/transformer_encoder_layer.py`, `utils/artifacts.py`, `reporting/phase_summary.py` | Gọi Phase 16 public API và hiển thị implementation contract |
| 17 | `attention/verification.py` | `attention/mapping.py`, `models/transformer_encoder_layer.py`, `utils/artifacts.py`, `reporting/phase_summary.py` | Gọi Phase 17 public API và hiển thị attention verification |
| 18 | `sanity/forward_sanity.py` | Phase 15-17 public model contracts, `utils/artifacts.py`, `reporting/phase_summary.py` | Gọi Phase 18 public API và hiển thị forward sanity matrix |
| 19 | `training/engine.py`, `training/engine_materialize.py` | `data/datasets.py`, `evaluation/metrics.py`, `experiments/registry.py`, `utils/reproducibility.py`, `utils/artifacts.py`, `reporting/phase_summary.py` | Gọi Phase 19 public API và hiển thị engine contract |
| 20 | `baselines/lstm_baseline.py` | Phase 15, 18, 19 contracts, `experiments/registry.py`, `reporting/phase_summary.py` | Gọi Phase 20 public API và hiển thị Validation metrics cùng learning curves |
| 21 | `baselines/transformer_b0.py` | Phase 16-19 contracts, `experiments/registry.py`, `reporting/phase_summary.py` | Gọi Phase 21 public API và hiển thị Validation metrics cùng learning curves |
| 22 | `diagnostics/learning_diagnostics.py` | `diagnostics/learning_curves.py`, Phase 20-21 artifacts, `reporting/phase_summary.py` | Gọi Phase 22 public API và hiển thị diagnostic findings cùng saved figures |
| 23 | `sweeps/sweep_results.py` | `experiments/phase_execution.py`, `experiments/registry.py`, `reporting/phase_summary.py` | Gọi selective Phase 23 public API và hiển thị verified S1 comparison |
| 24 | `sweeps/sweep_results.py` | `experiments/phase_execution.py`, Phase 23 reference, `experiments/registry.py`, `reporting/phase_summary.py` | Gọi selective Phase 24 public API và hiển thị verified S2 comparison |
| 25 | `sweeps/sweep_results.py` | `experiments/phase_execution.py`, Phase 24 reference, `experiments/registry.py`, `reporting/phase_summary.py` | Gọi selective Phase 25 public API và hiển thị verified S3 comparison |
| 26 | `sweeps/sweep_results.py` | `experiments/phase_execution.py`, Phase 25 reference, `experiments/registry.py`, `reporting/phase_summary.py` | Gọi selective Phase 26 public API và hiển thị verified S4 comparison |
| 27 | `sweeps/sweep_results.py` | `experiments/phase_execution.py`, Phase 26 reference, `experiments/registry.py`, `reporting/phase_summary.py` | Gọi selective Phase 27 public API và hiển thị verified S5 comparison |
| 28 | `sweeps/sweep_results.py` | `experiments/phase_execution.py`, Phase 27 reference, `experiments/registry.py`, `reporting/phase_summary.py` | Gọi selective Phase 28 public API và hiển thị verified S6 comparison |
| 29 | `sweeps/sweep_results.py` | `experiments/phase_execution.py`, Phase 28 reference, `experiments/registry.py`, `reporting/phase_summary.py` | Gọi selective Phase 29 public API và hiển thị verified S7 comparison |
| 30 | `sweeps/sweep_results.py` | `experiments/phase_execution.py`, Phase 29 reference, `experiments/registry.py`, `reporting/phase_summary.py` | Gọi một selective Phase 30 public API độc lập và hiển thị verified S8 comparison hoặc block reason |
| 31 | `sweeps/weight_decay.py` | `experiments/phase_execution.py`, Phase 30 winner/reference/sign-off, `experiments/registry.py`, frozen Training Engine, `reporting/phase_summary.py` | Gọi một selective Phase 31 public API độc lập và hiển thị verified WD0/WD1/WD2 comparison hoặc canonical block reason |
| 32 | `sweeps/dropout.py` | `experiments/phase_execution.py`, Phase 31 winner/reference/sign-off, `experiments/registry.py`, frozen Transformer và Training Engine, `reporting/phase_summary.py` | Gọi một selective Phase 32 public API độc lập và hiển thị verified DR01/DR02/DR03 comparison hoặc canonical block reason |
| 33 | `sweeps/d_model.py` | `experiments/phase_execution.py`, Phase 32 winner/reference/sign-off, `experiments/registry.py`, frozen Transformer và Training Engine, `reporting/phase_summary.py` | Gọi một selective Phase 33 public API độc lập và hiển thị verified D32/D64 comparison hoặc canonical block reason |
| 34 | `sweeps/heads.py` | `experiments/phase_execution.py`, Phase 33 winner/reference/sign-off, `experiments/registry.py`, frozen Transformer và Training Engine, `reporting/phase_summary.py` | Gọi một selective Phase 34 public API độc lập và hiển thị verified H2/H4 comparison hoặc canonical block reason |

Phases 6, 7, 8 thực hiện calculation, validation và feature engineering trên TRAIN rows only. Validation và Test rows được firewall triệt để cho đến Phase 9.

Không Phase nào ngoài direct descriptive Phase 6 exception được triển khai processing logic trong notebook cell.

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

`data_after_processing` và `data_after_split` không được dùng làm nguồn khoa học ngoài contract đã được Phase 0-33 phê duyệt.

Phase 6 chỉ được ghi derived master table đã duyệt tại `data/interim/uci_appliances_energy_prediction/energydata_feature_engineered_v1.csv`.

Phase 7 không ghi data artifact mới. Nó chỉ tham chiếu một `FEATURES-v1` master table và ghi ordered registries dưới `artifacts/feature_sets`.

Phase 8 không tạo ba bản sao feature data. Nó chỉ ghi row membership, boundaries và structural split artifacts dưới `artifacts/splits`.

Phase 9 chỉ ghi scaler/statistics artifacts và không ghi full scaled dataset.

Phase 10 chỉ ghi window indices, common target population, audits, fingerprints và sign-off dưới `artifacts/windows`. Không ghi full 3D tensors hoặc target values.

Phase 11 chỉ ghi Dataset/DataLoader configuration, registries, audits, fingerprints, device policy và sign-off dưới `artifacts/dataloaders`. Không ghi Dataset object, DataLoader object, full 3D tensors hoặc Test target values.

Phase 12 chỉ ghi metric contract, registry, schemas, synthetic reference examples, audits, discrepancies, README, manifest và sign-off dưới `artifacts/metrics`. Không ghi model prediction, model checkpoint, Test target hoặc Test metric.

Phase 13 không ghi data artifact. Nó chỉ ghi experiment registries, family definitions, lifecycle/guard audits, discrepancies, index, README, manifest và sign-off dưới `artifacts/experiments`. Synthetic run records chỉ tồn tại trong temporary directory và không được persist vào production registry.

Phase 14 không ghi derived data dưới `data`. Nó chỉ đọc bounded raw Appliances prefix, dùng Phase 10 window metadata và ghi prediction/evaluation evidence dưới `artifacts/baselines/persistence`. Production run config/status và registry views được owner Phase 13 quản lý dưới `artifacts/runs` và `artifacts/experiments`.

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

## 13. Artifact ownership Phase 0-33

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

artifacts/splits
-> Phase 5 membership, boundaries, fingerprints, structural/leakage audits, Train-only distribution summary, timestamp-only figure, manifest, discrepancies và sign-off

artifacts/eda
-> Phase 6 tables, figures, manifest, anomalies, Train-only scope audit và sign-off

artifacts/features
-> Phase 7 registry, lineage, availability, leakage audit, engineering audit, manifest, derived checksum, discrepancies, Train-only scope audit và sign-off

artifacts/feature_sets
-> Phase 8 components, ordered variant registries, fingerprints, lineage, semantic/leakage/order audits, manifest, discrepancies, human-readable specification, Train-only validation audit và sign-off

artifacts/scaling
-> Phase 9 scaler registry, statistics, scaling/leakage audits, Validation-only shift diagnostic, checksums, manifest, discrepancies, README và sign-off

artifacts/scalers
-> Phase 9 six X scaler bundles và one YS1 target scaler serialized bằng trusted local joblib

artifacts/windows
-> Phase 10 window index, common target population, population summary, rejected candidates, boundary/leakage/materialization audits, fingerprints, discrepancies, README, manifest và sign-off

artifacts/dataloaders
-> Phase 11 Dataset registry, DataLoader registry, sample coverage, batch, shuffle, chronology, worker, Test-firewall audits, device policy, discrepancies, README, manifest và sign-off

artifacts/metrics
-> Phase 12 metric contract, registry, prediction/comparison schemas, unit/reference/implementation/Test-firewall audits, discrepancies, README, manifest và sign-off

artifacts/experiments
-> Phase 13 canonical JSONL registry, flattened run/family/sweep/artifact/metric/failure/comparison CSV views, validation audit, discrepancies, index, README, manifest và sign-off; Phase 14 cập nhật qua public registry APIs

artifacts/baselines/persistence
-> Phase 14 manifest, baseline summary, Validation predictions, Validation metrics, audit, unit-test evidence, discrepancies, README và sign-off

artifacts/runs
-> Phase 14+ canonical run config và status do EXPERIMENTS-v1 quản lý; Phase 14 hiện có một completed PERSISTENCE_BASELINE run

artifacts/models/lstm
-> Phase 15 LSTM implementation contract, audits, manifest và sign-off

artifacts/models/transformer
-> Phase 16 Transformer implementation contract, audits, manifest và sign-off

artifacts/attention_verification
-> Phase 17 attention tensor verification, audits, manifest và sign-off

artifacts/forward_sanity
-> Phase 18 forward-pass sanity matrix, audits, manifest và sign-off

artifacts/training_engine
-> Phase 19 training-engine contract, smoke tests, manifest và sign-off

artifacts/lstm_baseline
-> Phase 20 LSTM baseline Validation run summary, history, metrics, checkpoint lineage và sign-off

artifacts/transformer_b0
-> Phase 21 Transformer B0 Validation run summary, history, metrics, checkpoint lineage và sign-off

artifacts/learning_diagnostics
-> Phase 22 diagnostic summary, findings, saved figures, manifest và sign-off

artifacts/sweeps
-> Phase 23-34 condition results, condition lineage, manifests, winners, reference updates, revisions và sign-offs; Phase 31 S9 outputs nằm dưới artifacts/sweeps/S9_weight_decay, Phase 32 S10 outputs nằm dưới artifacts/sweeps/S10_dropout, Phase 33 S11 outputs nằm dưới artifacts/sweeps/S11_d_model và Phase 34 S12 outputs nằm dưới artifacts/sweeps/S12_heads
```

Không lưu checkpoint hoặc learned model trong các Phase 0-14 roots. Checkpoint Phase 20 trở đi phải nằm trong owner run root và được EXPERIMENTS-v1 index bằng checksum. Actual Persistence Validation predictions chỉ nằm trong owner root `artifacts/baselines/persistence`; `artifacts/experiments` chỉ index và link artifact do Phase consumer sở hữu.

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
Call a selective phase execution public API after kernel restart
Display a persistent widget-free HTML phase state
```

Chỉ được display nội dung nằm trong presentation allowlist của Phase. Không được dump JSON, warning table, discrepancy table, source artifact, checksum, fingerprint hoặc technical lineage ra notebook.

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
Transient widget state
Direct sweep CSV loading
Condition completeness logic
Training-process dispatch logic
```

### 14.3. Cell order

Notebook phải có section tuần tự:

```text
Phase 1
Phase 2
Phase 3
Phase 4
Phase 5
Phase 6
Phase 7
Phase 8
Phase 9
Phase 10
Phase 11
Phase 12
Phase 13
Phase 14
Phase 15
Phase 16
Phase 17
Phase 18
Phase 19
Phase 20
Phase 21
Phase 22
Phase 23
Phase 24
Phase 25
Phase 26
Phase 27
Phase 28
Phase 29
Phase 30
Phase 31
Phase 32
Phase 1-32 Boundary
```

Trong đó Phase 5 là Chronological Split, Phase 6 là Exploratory Data Analysis với TRAIN-only scope, Phase 7 là Feature Engineering với TRAIN-only scope, Phase 8 là Feature-Set Variants với TRAIN-only scope. Validation và Test rows chỉ xuất hiện trong các phép biến đổi và đánh giá từ Phase 9 trở đi.

Phase 0 vẫn là canonical prerequisite nội bộ nhưng không có heading, orchestration cell, import hoặc output trong notebook.

Notebook không được dựa vào hidden kernel state.

Mỗi code cell phải chạy được sau restart kernel theo đúng thứ tự. Phase 23-33 orchestration cell phải chạy độc lập sau restart kernel bằng một public API call, không phụ thuộc hidden state từ cell trước và không gọi materializer của Phase trước.

### 14.4. Selective phase state

Selective execution phải phân loại Phase thành đúng một trạng thái:

```text
VALID_REUSABLE
LOG_MISSING
LOG_STALE
DERIVED_ARTIFACT_MISSING
CONDITION_INCOMPLETE
SIGNOFF_INVALID
UPSTREAM_INVALID
ENVIRONMENT_INVALID
RUNNING
FAILED
```

Trạng thái được xác định từ canonical evidence, không được xác định chỉ từ processing log hoặc notebook output.

### 14.5. Selective phase action

Action được phép:

```text
RENDER_ONLY
REBUILD_LOG_ONLY
REBUILD_DERIVED_ONLY
EXECUTE_MISSING_ONLY
WAIT_FOR_RUNNING_PROCESS
BLOCK
```

Dispatcher phải chọn action nhỏ nhất đủ an toàn. `EXECUTE_MISSING_ONLY` chỉ được phép khi upstream, environment, condition registry và run lineage đều hợp lệ.

### 14.6. Evidence authority

Thứ tự authority:

```text
Approved Phase detail và architecture
Approved config
Experiment registry run config và status
Run artifacts và checksums
Phase manifest và sign-off
Processing log và source checksums
Notebook stored output
```

Processing log dưới `docs/save_log_in_processing` là derived presentation record. Processing log không được tự mình chứng minh scientific completion.

### 14.7. Signed artifact revision

Artifact đã sign-off không được overwrite để che giấu invalid historical state. Correction phải tạo revision mới và active-revision pointer. Processing log có thể được tái tạo từ active validated revision.

### 14.8. Notebook output preservation

Trước khi sửa output notebook phải ghi checksum, cell ID, execution count, output count và MIME type. Không được global clear output. Chỉ output thuộc cell được refactor hoặc cell được người dùng chủ động chạy lại mới được thay đổi.

## 15. EDA boundary

Phase 5 được phép:

```text
Mô tả target và features.
Tạo EDA-only temporal columns trong df_eda.
Visualize temporal pattern.
Tạo descriptive correlation.
Tạo hypothesis cho Phase sau.
Thực hiện direct descriptive EDA trong CourseWork.ipynb theo plan CW-PHASE-5-EDA-DIRECT-001.
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

Direct notebook smoothing demonstration không phải canonical Phase 5 processing. Nó chỉ được chạy trên deep copy, không được ghi file và không được handoff sang Phase sau.

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
Phase 5 -> Phase 6
Phase 6 -> Phase 7
Phase 7 -> Phase 8
Phase 8 -> Phase 9
Phase 9 -> Phase 10
Phase 10 -> Phase 11
Phase 11 -> Phase 12
Phase 12 -> Phase 13
Phase 13 -> Phase 14
Phase 14 -> Phase 15
Phase 15 -> Phase 16
Phase 16 -> Phase 17
Phase 17 -> Phase 18
Phase 18 -> Phase 19
Phase 19 -> Phase 20
Phase 20 -> Phase 21
Phase 21 -> Phase 22
Phase 22 -> Phase 23
Phase 23 -> Phase 24
Phase 24 -> Phase 25
Phase 25 -> Phase 26
Phase 26 -> Phase 27
Phase 27 -> Phase 28
Phase 28 -> Phase 29
Phase 29 -> Phase 30
Raw checksum preservation
Artifact reload
Notebook orchestration boundary
Selective phase state resolution
Missing-only condition resolution
Persistent widget-free notebook output
Notebook output preservation
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
Expected condition coverage và run lineage hợp lệ khi Phase là sweep.
Signed environment hợp lệ trước mọi scientific execution.
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

Human đã duyệt direct Phase 6 EDA exception qua yêu cầu dẫn đến plan `CW-PHASE-6-EDA-DIRECT-001`. Mọi mở rộng ngoại lệ sang Phase khác vẫn phải quay lại change-control gate.

## 25. Transition state

Trạng thái hiện hành:

```text
Phase 0-33 có canonical source owner.
CourseWork.ipynb giữ direct descriptive Phase 6 EDA exception đã được Human duyệt.
Phase 0 giữ vai trò contract nội bộ và không xuất hiện trong CourseWork.ipynb.
Phase 1-33 dùng presentation allowlist tối giản trên notebook.
Phase 0-33 vẫn lưu processing log JSON đầy đủ, độc lập với notebook presentation.
Phase 5 Chronological Split hiện chạy trước EDA, FE và FS, khoá chronological 70/15/15 row membership, WB0 primary metadata và Test firewall.
Phase 6 EDA trong notebook chỉ gọi public API với TRAIN-only scope và hiển thị outputs.
FEATURES-v1 giữ raw lineage và chỉ thêm năm deterministic calendar features trên TRAIN rows.
Phase 7 Feature Engineering trong notebook chỉ gọi public API với TRAIN-only scope và hiển thị outputs.
FEATURESETS-v1 khóa sáu ordered variants và fingerprints mà không sao chép data trên TRAIN rows.
Phase 8 Feature-Set Variants trong notebook chỉ gọi public API với TRAIN-only scope và hiển thị outputs.
Phase 9 trong notebook chỉ gọi public API và hiển thị outputs.
SCALING-v1 khóa sáu X scaler bundles, YS0 identity, một YS1 scaler và Train-only fit protocol.
Phase 10 trong notebook chỉ gọi public API và hiển thị outputs.
WINDOWS-v1 khóa L36/L72/L144 H1 geometry, WB0/WB1 metadata và lazy materialization contract.
WINDOWPOP-v1 khóa common target population cho mọi controlled comparison.
Phase 11 trong notebook chỉ gọi public API và hiển thị Dataset population cùng Loader policy.
DATALOADERS-v1 khóa lazy map-style Dataset, B32/B64, Train-only shuffle, full sample coverage, split-specific RNG và Test target firewall.
Phase 12 trong notebook chỉ gọi public API và hiển thị Metric registry cùng Evaluation policy.
METRICS-v1 khóa MAE Wh, RMSE Wh, R², full-split aggregation, original-Wh reporting, residual convention, sample-weighted epoch loss và final-Test firewall.
Phase 13 trong notebook chỉ gọi public API và hiển thị Registry state cùng Core safeguards.
EXPERIMENTS-v1 khóa run identity, canonical config fingerprint, complete lineage, lifecycle, artifact/metric links, sweep consistency, completed-config immutability và final-Test firewall.
Phase 13 khởi tạo production registry với zero fabricated runs; synthetic tests không được persist.
Phase 14 trong notebook chỉ gọi public API và hiển thị Validation performance cùng Baseline contract.
PERSISTENCE-v1 khóa task-level last-value formula, L144-anchored Validation population, raw-Wh evaluation và Test firewall.
EXPERIMENTS-v1 hiện quản lý một completed canonical PERSISTENCE_BASELINE run với null feature/scaler/seed fields và zero trainable parameters.
Phase 15-19 khóa model, attention, forward-sanity và training-engine contracts.
Phase 20-21 sở hữu canonical learned-model baseline runs và Validation-only histories.
Phase 22 sở hữu learning-curve diagnostics.
Phase 23-34 sở hữu tuần tự các controlled sweeps S1-S12.
Selective phase execution kiểm tra canonical evidence và không tự chạy upstream Phase.
Processing log là derived record và không đủ để xác nhận completion.
Phase 29, Phase 30, Phase 31, Phase 32, Phase 33 và Phase 34 hiện phải qua selective validation trước khi được phép reuse hoặc scientific recovery.
Phase 31 sở hữu S9 weight-decay sweep, dùng WD0=0, WD1=1e-4, WD2=1e-3 và chỉ được scientific execution sau khi Phase 30 canonical handoff hợp lệ.
Phase 32 sở hữu S10 dropout sweep, dùng DR01=0.1, DR02=0.2, DR03=0.3 và chỉ được scientific execution sau khi Phase 31 canonical handoff hợp lệ.
Phase 33 sở hữu S11 d_model sweep, dùng D32=32, D64=64, giữ H4/N2/F128 cố định, tái sử dụng D64 reference và chỉ chạy mới D32 sau khi Phase 32 canonical handoff hợp lệ.
Phase 34 sở hữu S12 head sweep, dùng H2=2 và H4=4, giữ selected d_model/N2/F128 cố định, tái sử dụng H4 exact reference và chỉ chạy mới H2 sau khi Phase 33 canonical handoff hợp lệ.
Phase 30 recovery phải bắt đầu bằng read-only audit, xác định earliest invalid Phase, thực thi tuần tự từ dependency đó và dừng ngay khi một Phase không đạt canonical verification.
Môi trường lịch sử phải được bảo toàn; runtime identity mới không được âm thầm ghi đè signed environment evidence.
Canonical finalizer Phase 23-34 chỉ được đọc verified registry evidence, áp dụng Validation-only selection và tạo kết quả khi toàn bộ registered conditions hợp lệ.
Dependency-aware terminal runner không được chạy lại notebook, không được tái huấn luyện condition có complete exact-match evidence và không được vượt qua Test firewall.
`--audit-only` và `--dry-run` phải giữ nguyên scientific artifacts; `--recover-environment` chỉ được tạo revision mới sau khi kernel, CUDA/MPS, smoke test và dependency freeze đều hợp lệ.
Notebook output phải dùng persistent HTML và không phụ thuộc widget model state.
```

Phase 9-34 được triển khai theo các Phase detail tương ứng:

```text
Phase_9_Train_only_scaling.md
Phase_10_Window_builder.md
Phase_11_DataLoaders.md
Phase_12_Shared_metrics.md
Phase_13_Experiment_registry.md
Phase_14_Persistence_baseline.md
Phase_15_LSTM_implementation.md
Phase_16_Transformer_implementation.md
Phase_17_Attention-aware_encoder_verification.md
Phase_18_Forward-pass_sanity_tests.md
Phase_19_Baseline_training_engine.md
Phase_20_LSTM_baseline_run.md
Phase_21_Transformer_B0_run.md
Phase_22_Learning-curve_diagnostics.md
Phase_23_S1_Feature-set_sweep.md
Phase_24_S2_Time-feature_sweep.md
Phase_25_S3_Target-scaling_sweep.md
Phase_26_S4_Lookback_sweep.md
Phase_27_S5_Pooling_sweep.md
Phase_28_S6_Activation_sweep.md
Phase_29_S7_Batch_sweep.md
Phase_30_S8_Learning-rate_sweep.md
Phase_31_S9_Weight-decay_sweep.md
```

Không được bỏ qua Phase gate trong quá trình chuyển đổi.

## 26. Architecture validation checklist

```text
[x] Canonical root rõ ràng.
[x] Canonical notebook rõ ràng.
[x] Source ownership rõ ràng.
[x] Phase 0-33 mapping đầy đủ.
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
[x] Phase 5 Chronological Split có canonical owner, artifacts, tests và notebook orchestration boundary.
[x] Phase 6 EDA có canonical owner, artifacts, tests, notebook orchestration boundary và TRAIN-only scope gate.
[x] Phase 7 Feature Engineering có canonical owner, artifacts, tests, notebook orchestration boundary và TRAIN-only scope gate.
[x] Phase 8 Feature-Set Variants có canonical owner, artifacts, tests, notebook orchestration boundary và TRAIN-only scope gate.
[x] Phase 9 có canonical owner, artifacts, tests và notebook orchestration boundary.
[x] Phase 10 có canonical owner, artifacts, tests và notebook orchestration boundary.
[x] Phase 11 có canonical owner, artifacts, tests và notebook orchestration boundary.
[x] Phase 12 có canonical owner, artifacts, tests và notebook orchestration boundary.
[x] Phase 13 có canonical owner, artifacts, tests và notebook orchestration boundary.
[x] Phase 14 có canonical owner, artifacts, tests và notebook orchestration boundary.
[x] Phase 14 Persistence run đã complete với Validation-only metrics và Test firewall.
[x] Phase 15-22 có canonical source owner và artifact root.
[x] Phase 23-34 có sweep owner, prerequisite chain và Validation-only selection boundary.
[x] Selective phase state và action model đã được quy định.
[x] Processing log không được dùng làm canonical scientific evidence.
[x] Notebook output preservation và widget-free presentation đã được quy định.
[x] Phase 31 có S9-specific owner, selective gate, notebook boundary và Validation-only selection contract.
[x] Phase 32 có S10-specific owner, selective gate, notebook boundary và Validation-only selection contract.
[x] Phase 33 có S11-specific owner, selective gate, terminal-owned execution, presentation boundary và Validation-only selection contract.
[x] Phase 34 có S12-specific owner, selective gate, terminal-owned execution, presentation boundary và Validation-only selection contract.
[x] Không triển khai Phase 35.
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
PHASE_15_TO_30_ARCHITECTURE_APPROVED=true
SELECTIVE_PHASE_REFACTOR_ALLOWED=true
PHASE_31_ARCHITECTURE_APPROVED=true
PHASE_30_DEPENDENCY_RECOVERY_APPROVED=true
PHASE_32_ARCHITECTURE_APPROVED=true
PHASE_33_ARCHITECTURE_APPROVED=true
```

## 28. Amendment log

```text
v1.0 (2026-08-17) - Initial activation
v1.1 (2026-08-17) - Phase reorder amendment
  - Phase 5 = Chronological Split (was EDA)
  - Phase 6 = EDA với TRAIN-only scope (was FE)
  - Phase 7 = Feature Engineering với TRAIN-only scope (was FS)
  - Phase 8 = Feature-Set Variants với TRAIN-only scope (was Split)
  - Authoritative plan: docs/plan-doc/analysis_error/phase_reorder_split_before_eda_train_only_refactor_plan.md
  - Human approval: granted 2026-08-17
v1.2 (2026-08-22) - Phase 15-30 and selective resume amendment
  - Extended canonical ownership and notebook boundary through Phase 30
  - Added selective state, action, evidence authority and artifact revision contracts
  - Added persistent widget-free presentation and output-preservation contracts
  - Authoritative plan: docs/plan-doc/plan_before_process/refactor_notebook_selective_phase_resume_plan.md
  - Human approval: granted 2026-08-22
v1.3 (2026-08-22) - Phase 31 S9 weight-decay amendment
  - Extended canonical ownership and notebook boundary through Phase 31
  - Added S9-specific owner, Phase 30 handoff gate and WD0/WD1/WD2 contract
  - Kept processing logs derived and scientific execution conditional on canonical Phase 30 evidence
  - Authoritative plan: docs/plan-doc/plan_before_process/phase_31_s9_weight_decay_selective_execution_plan.md
  - Human approval: granted 2026-08-22
v1.4 (2026-08-22) - Phase 30 dependency recovery amendment
  - Added read-only Phase 22-30 recovery inspection ownership
  - Added environment reconciliation boundary and minimal execution-set contract
  - Added verified canonical finalization and dependency-aware terminal execution rules
  - Preserved notebook outputs, historical environment evidence and Test firewall
  - Authoritative plan: docs/plan-doc/plan_before_process/refactor_and_rerun_phase_30_dependency_recovery_plan.md
  - Human approval: granted 2026-08-22
v1.5 (2026-08-22) - Phase 32 S10 dropout amendment
  - Extended canonical ownership and notebook boundary through Phase 32
  - Added S10 dropout owner, Phase 31 handoff gate and DR01/DR02/DR03 contract
  - Preserved terminal-owned execution, derived processing logs, static HTML and Test firewall
  - Authoritative plan: docs/plan-doc/plan_before_process/phase_32_s10_dropout_selective_execution_plan.md
  - Human approval: granted 2026-08-22
v1.6 (2026-08-23) - Phase 33 S11 d_model amendment
  - Extended canonical ownership and selective presentation boundary through Phase 33
  - Added S11 d_model owner, Phase 32 handoff gate and D32/D64 contract
  - Froze H4/N2/F128, reused D64 exact reference and limited fresh training to D32
  - Preserved terminal-owned execution, derived processing logs, static HTML and Test firewall
  - Authoritative plan: docs/plan-doc/plan_before_process/phase_33_s11_d_model_selective_execution_plan.md
  - Human approval: granted 2026-08-23
v1.7 (2026-08-23) - Phase 34 S12 head amendment
  - Extended canonical ownership and selective presentation boundary through Phase 34
  - Added S12 head owner, Phase 33 handoff gate and H2/H4 contract
  - Froze selected d_model/N2/F128, reused H4 exact reference and limited fresh training to H2
  - Preserved terminal-owned execution, derived processing logs, static HTML and Test firewall
  - Authoritative plan: docs/plan-doc/plan_before_process/phase_34_s12_head_sweep_selective_terminal_execution_plan.md
  - Human approval: granted 2026-08-23
```
