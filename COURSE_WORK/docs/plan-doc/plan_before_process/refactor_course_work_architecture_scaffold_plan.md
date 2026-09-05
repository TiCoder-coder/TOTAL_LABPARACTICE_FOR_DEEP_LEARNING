# REFACTOR PLAN — COURSE_WORK ARCHITECTURE SCAFFOLD

## 1. Plan metadata

```text
Task ID:
ARCH-SCAFFOLD-001

Task type:
Additive architecture scaffold refactor

Execution scope:
Create only approved files and directories

Code implementation:
Forbidden

Move / rename / delete:
Forbidden

Existing-file modification:
Forbidden

Current state:
WAITING_FOR_USER_APPROVAL
```

---

## 2. Objective

Refactor nhẹ cấu trúc `COURSE_WORK` bằng cách bổ sung các layer còn thiếu từ kiến trúc mẫu, đồng thời giữ nguyên toàn bộ file/folder hiện hữu có vai trò tương đương.

Kết quả của task này chỉ là filesystem scaffold:

```text
Tạo folder cần thiết
Tạo file placeholder được phê duyệt
Không thêm code
Không thêm config values
Không tạo artifact giả
Không di chuyển dữ liệu
Không đổi scientific protocol
```

---

## 3. Sources reviewed

Các nguồn đã được kiểm tra cho plan này:

```text
COURSE_WORK/working_rule.md
COURSE_WORK/docs/RULE_BASE/rule_code.md
COURSE_WORK/docs/plan-doc/plan_overview/Main_plan.md
COURSE_WORK/docs/plan-doc/plan_overview/Requirement_analyst_&_plan_to_make.md
COURSE_WORK/docs/plan-doc/plan_overview/Prevent_overfitting.md
COURSE_WORK/docs/plan-doc/plan_detail_for_each_phase/
COURSE_WORK tree hiện tại trên filesystem
Kiến trúc mẫu trong pasted-text.txt
Root .gitignore
```

`COURSE_WORK/working_rule.md` đã được đối chiếu hash và giống byte-for-byte với `working_rule.md` ở repository root.

Các file hiện đang rỗng:

```text
COURSE_WORK/README.md
COURSE_WORK/requirements.txt
COURSE_WORK/docs/code_base_audit.md
COURSE_WORK/notebook_course_work/practice_3.ipynb
```

Không bổ sung nội dung vào các file trên trong task này.

---

## 4. Blocking governance finding

`rule_code.md` quy định bắt buộc phải đọc:

```text
architecture_rule.md
```

nhưng file này hiện chưa tồn tại trong `COURSE_WORK`.

Kiến trúc mẫu có đề xuất:

```text
docs/rules/architecture_rule.md
```

Trong cấu trúc hiện tại, folder tương đương đã tồn tại là:

```text
COURSE_WORK/docs/RULE_BASE/
```

Quyết định additive-only được đề xuất:

```text
Tạo placeholder:
COURSE_WORK/docs/RULE_BASE/architecture_rule.md

Không tạo folder docs/rules/ song song.
Không di chuyển working_rule.md.
Không tự viết nội dung architecture rule trong task scaffold.
```

Hệ quả:

```text
Placeholder rỗng không được xem là architecture contract đã hoàn tất.
Mọi code/Phase implementation tiếp tục bị block cho đến khi
architecture_rule.md được viết, review và Human phê duyệt trong task riêng.
```

---

## 5. Current paths that must be preserved

Các path hiện hữu sau có vai trò tương đương với kiến trúc mẫu và phải giữ nguyên:

| Current path | Sample equivalent | Decision |
|---|---|---|
| `COURSE_WORK/working_rule.md` | `docs/rules/working_rule.md` | Giữ current path, không duplicate |
| `COURSE_WORK/docs/RULE_BASE/` | `docs/rules/` | Giữ current path và tên folder |
| `COURSE_WORK/docs/code_base_audit.md` | `docs/audits/code_base_audit.md` | Giữ current path, không duplicate |
| `COURSE_WORK/docs/plan-doc/plan_to_refactor&fix/` | `plan_to_refactor_fix/` | Giữ current path, không tạo folder song song |
| `COURSE_WORK/notebook_course_work/practice_3.ipynb` | `coursework.ipynb` | Giữ current filename, không tạo notebook thứ hai |
| Repository-root `.gitignore` | `COURSE_WORK/.gitignore` | Giữ root file, không tạo nested duplicate |
| `data/raw_data/energydata_complete.csv` | Cùng raw CSV | Giữ nguyên, không copy/move |
| `docs/plan-doc/plan_overview/*.md` | `plan.md`, `plan_v2.md`, ... | Giữ các plan hiện có, không tạo plan giả |
| `docs/save_process_proceduce_own_phase_refactor&fix/` | Không có trong sample | Giữ nguyên, không xóa |

---

## 6. Architecture assessment

### 6.1. Điểm phù hợp của kiến trúc mẫu

Kiến trúc mẫu bổ sung đúng các separation-of-concerns còn thiếu:

```text
configs
data lifecycle directories
reusable src package
thin phase entry points
tests by responsibility
maintenance scripts
runtime artifacts
documentation logs
```

### 6.2. Điểm không được sao chép nguyên xi

Kiến trúc mẫu chưa phản ánh đầy đủ current project contracts:

```text
Main_plan và Phase details có artifact folders cho environment,
acquisition, schema, temporal, EDA, features, feature sets, splits,
scaling, windows, dataloaders, metrics, runs và sweeps.

Kiến trúc mẫu chưa có module/config/artifact rõ ràng cho:
self-supervised pretraining,
LoRA,
partial fine-tuning.

Một số source filename trong Phase details khác sample:
dataset.py so với datasets.py,
loaders.py không xuất hiện trong sample,
lstm_regressor.py so với lstm.py,
transformer_regressor.py so với transformer.py.
```

Do `architecture_rule.md` chưa tồn tại, task này không được tự chọn một phía rồi khẳng định đó là final module contract.

Plan sử dụng filename của kiến trúc mẫu cho scaffold chính và chỉ bổ sung các path LoRA tối thiểu. Các alias/file khác trong Phase detail không được tạo thêm để tránh duplicate ownership cho đến khi architecture rule được chốt.

### 6.3. Generated files không được scaffold rỗng

Không tạo trước các file sau:

```text
data/manifests/*.json
artifacts/**/*.json
artifacts/**/*.csv
artifacts/**/*.jsonl
checkpoint files
prediction files
figure files
result reports
```

Lý do:

```text
JSON rỗng không hợp lệ.
Artifact placeholder có thể bị hiểu nhầm là output Phase đã tồn tại.
Artifact phải được producer Phase tạo với schema, lineage và checksum thật.
```

---

## 7. Proposed additive scaffold

### 7.1. Project metadata

Tạo mới:

```text
COURSE_WORK/pyproject.toml
```

File chỉ là zero-byte placeholder trong task này. Nó chưa được xem là package configuration hợp lệ cho tới khi có task riêng để điền và duyệt nội dung.

Không tạo `COURSE_WORK/.gitignore` vì repository-root `.gitignore` đã tồn tại và đang áp dụng cho toàn workspace.

### 7.2. Config hierarchy

Tạo directories:

```text
COURSE_WORK/configs/
COURSE_WORK/configs/base/
COURSE_WORK/configs/sweeps/
COURSE_WORK/configs/final/
COURSE_WORK/configs/adaptation/
```

Tạo zero-byte config placeholders:

```text
configs/base/data.yaml
configs/base/model.yaml
configs/base/training.yaml

configs/sweeps/feature_set.yaml
configs/sweeps/time_feature.yaml
configs/sweeps/target_scaling.yaml
configs/sweeps/lookback.yaml
configs/sweeps/pooling.yaml
configs/sweeps/activation.yaml
configs/sweeps/batch_size.yaml
configs/sweeps/learning_rate.yaml
configs/sweeps/weight_decay.yaml
configs/sweeps/dropout.yaml
configs/sweeps/d_model.yaml
configs/sweeps/heads.yaml
configs/sweeps/layers.yaml
configs/sweeps/ffn.yaml
configs/sweeps/loss.yaml
configs/sweeps/epoch_cap.yaml
configs/sweeps/gradient_clipping.yaml
configs/sweeps/revin.yaml

configs/final/final_model.yaml

configs/adaptation/pretraining.yaml
configs/adaptation/lora.yaml
configs/adaptation/partial_fine_tuning.yaml
```

Các placeholder này không được load trong execution cho đến khi có nội dung được phê duyệt.

### 7.3. Data lifecycle directories

Giữ nguyên:

```text
data/raw_data/
data/raw_data/energydata_complete.csv
data/data_after_split/
```

Tạo thêm:

```text
data/interim/

data/data_after_split/train/
data/data_after_split/validation/
data/data_after_split/test/

data/processed/
data/processed/features/
data/processed/windows/
data/processed/scaled/

data/manifests/
```

Không tạo manifest/data output file trong task này.

### 7.4. Reusable source package

Tạo package root:

```text
src/course_work/
src/course_work/__init__.py
```

Tạo package directories và zero-byte `__init__.py`:

```text
src/course_work/contracts/
src/course_work/data/
src/course_work/models/
src/course_work/models/components/
src/course_work/training/
src/course_work/evaluation/
src/course_work/experiments/
src/course_work/attention/
src/course_work/reporting/
src/course_work/utils/
```

Tạo zero-byte source placeholders theo kiến trúc mẫu:

```text
src/course_work/contracts/data_contracts.py
src/course_work/contracts/model_contracts.py
src/course_work/contracts/training_contracts.py
src/course_work/contracts/metric_contracts.py
src/course_work/contracts/attention_contracts.py
src/course_work/contracts/artifact_contracts.py
src/course_work/contracts/phase_contracts.py

src/course_work/data/acquisition.py
src/course_work/data/schema.py
src/course_work/data/temporal.py
src/course_work/data/eda.py
src/course_work/data/features.py
src/course_work/data/feature_sets.py
src/course_work/data/splitting.py
src/course_work/data/scaling.py
src/course_work/data/windows.py
src/course_work/data/datasets.py

src/course_work/models/lstm.py
src/course_work/models/transformer.py
src/course_work/models/components/attention.py
src/course_work/models/components/encoder_layer.py
src/course_work/models/components/positional_encoding.py
src/course_work/models/components/revin.py
src/course_work/models/components/lora.py

src/course_work/training/engine.py
src/course_work/training/checkpointing.py
src/course_work/training/early_stopping.py
src/course_work/training/optimization.py
src/course_work/training/reproducibility.py
src/course_work/training/pretraining.py
src/course_work/training/fine_tuning.py

src/course_work/evaluation/metrics.py
src/course_work/evaluation/inference.py
src/course_work/evaluation/predictions.py
src/course_work/evaluation/residuals.py
src/course_work/evaluation/regimes.py
src/course_work/evaluation/worst_cases.py

src/course_work/experiments/registry.py
src/course_work/experiments/sweeps.py
src/course_work/experiments/candidates.py
src/course_work/experiments/rolling_origin.py
src/course_work/experiments/final_lock.py
src/course_work/experiments/adaptation.py

src/course_work/attention/extraction.py
src/course_work/attention/heatmaps.py
src/course_work/attention/last_query.py
src/course_work/attention/head_comparison.py
src/course_work/attention/error_conditioned.py
src/course_work/attention/seed_stability.py

src/course_work/reporting/notebook_view.py
src/course_work/reporting/tables.py
src/course_work/reporting/conclusions.py
src/course_work/reporting/provenance.py

src/course_work/utils/io.py
src/course_work/utils/hashing.py
src/course_work/utils/paths.py
src/course_work/utils/validation.py
src/course_work/utils/device.py
```

Mọi `.py` mới phải có kích thước 0 byte sau scaffold. Không có code, comment, docstring hoặc import.

### 7.5. Phase entry-point scaffold

Giữ nguyên folder:

```text
COURSE_WORK/processing_own_phase/
```

Tạo zero-byte files:

```text
processing_own_phase/__init__.py
processing_own_phase/phase_00_coursework_contract.py
processing_own_phase/phase_01_environment.py
processing_own_phase/phase_02_data_acquisition.py
processing_own_phase/phase_03_schema_audit.py
processing_own_phase/phase_04_temporal_integrity_audit.py
processing_own_phase/phase_05_eda.py
processing_own_phase/phase_06_feature_engineering.py
processing_own_phase/phase_07_feature_set_variants.py
processing_own_phase/phase_08_chronological_split.py
processing_own_phase/phase_09_train_only_scaling.py
processing_own_phase/phase_10_window_builder.py
processing_own_phase/phase_11_dataloaders.py
processing_own_phase/phase_12_shared_metrics.py
processing_own_phase/phase_13_experiment_registry.py
processing_own_phase/phase_14_persistence_baseline.py
processing_own_phase/phase_15_lstm_implementation.py
processing_own_phase/phase_16_transformer_implementation.py
processing_own_phase/phase_17_attention_encoder_verification.py
processing_own_phase/phase_18_forward_pass_sanity_tests.py
processing_own_phase/phase_19_baseline_training_engine.py
processing_own_phase/phase_20_lstm_baseline_run.py
processing_own_phase/phase_21_transformer_b0_run.py
processing_own_phase/phase_22_learning_curve_diagnostics.py
processing_own_phase/phase_23_s1_feature_set_sweep.py
processing_own_phase/phase_24_s2_time_feature_sweep.py
processing_own_phase/phase_25_s3_target_scaling_sweep.py
processing_own_phase/phase_26_s4_lookback_sweep.py
processing_own_phase/phase_27_s5_pooling_sweep.py
processing_own_phase/phase_28_s6_activation_sweep.py
processing_own_phase/phase_29_s7_batch_sweep.py
processing_own_phase/phase_30_s8_learning_rate_sweep.py
processing_own_phase/phase_31_s9_weight_decay_sweep.py
processing_own_phase/phase_32_s10_dropout_sweep.py
processing_own_phase/phase_33_s11_d_model_sweep.py
processing_own_phase/phase_34_s12_head_sweep.py
processing_own_phase/phase_35_s13_layer_sweep.py
processing_own_phase/phase_36_s14_ffn_sweep.py
processing_own_phase/phase_37_s15_loss_sweep.py
processing_own_phase/phase_38_s16_epoch_cap_sweep.py
processing_own_phase/phase_39_s17_gradient_clipping_sweep.py
processing_own_phase/phase_40_s18_revin_sweep.py
processing_own_phase/phase_41_s19_boundary_protocol_check.py
processing_own_phase/phase_42_candidate_synthesis.py
processing_own_phase/phase_43_lstm_tuning.py
processing_own_phase/phase_44_rolling_origin_robustness.py
processing_own_phase/phase_45_final_model_lock.py
processing_own_phase/phase_46_three_seed_final_runs.py
processing_own_phase/phase_47_final_test_evaluation.py
processing_own_phase/phase_48_prediction_analysis.py
processing_own_phase/phase_49_residual_analysis.py
processing_own_phase/phase_50_error_by_regime_analysis.py
processing_own_phase/phase_51_worst_error_analysis.py
processing_own_phase/phase_52_attention_extraction.py
processing_own_phase/phase_53_attention_heatmaps.py
processing_own_phase/phase_54_last_query_attention.py
processing_own_phase/phase_55_head_comparison.py
processing_own_phase/phase_56_error_conditioned_attention.py
processing_own_phase/phase_57_seed_stability_attention_check.py
processing_own_phase/phase_58_final_tables.py
processing_own_phase/phase_59_final_conclusions.py
```

Không tạo Phase 60+ trong task này vì chưa có Phase detail tương ứng được phê duyệt cho pretraining/LoRA/partial fine-tuning.

### 7.6. Test hierarchy

Giữ nguyên `COURSE_WORK/tests/` và tạo:

```text
tests/unit/
tests/unit/contracts/
tests/unit/data/
tests/unit/models/
tests/unit/training/
tests/unit/evaluation/
tests/unit/experiments/
tests/unit/attention/
tests/unit/reporting/

tests/integration/
tests/integration/data_pipeline/
tests/integration/training_pipeline/
tests/integration/evaluation_pipeline/
tests/integration/attention_pipeline/

tests/contracts/
tests/regression/
tests/regression/phase_regression/
tests/fixtures/
```

Không tạo test implementation trong task này.

### 7.7. Maintenance scripts

Tạo:

```text
scripts/
scripts/maintenance/
scripts/maintenance/verify_project_state.py
scripts/maintenance/verify_phase_dependencies.py
scripts/maintenance/verify_artifacts.py
scripts/maintenance/verify_notebook_contract.py
scripts/maintenance/audit_architecture.py
```

Các `.py` trên là zero-byte placeholders, chưa phải executable scripts.

### 7.8. Artifact hierarchy

Tạo directories cho Phase/runtime outputs, không tạo output files:

```text
artifacts/
artifacts/manifests/
artifacts/registry/
artifacts/checkpoints/
artifacts/experiments/
artifacts/runs/
artifacts/sweeps/

artifacts/environment/
artifacts/acquisition/
artifacts/schema/
artifacts/temporal/
artifacts/eda/
artifacts/features/
artifacts/feature_sets/
artifacts/splits/
artifacts/scaling/
artifacts/windows/
artifacts/dataloaders/
artifacts/metrics/

artifacts/pretraining/
artifacts/adaptation/
artifacts/adaptation/lora/
artifacts/adaptation/partial_fine_tuning/

artifacts/rolling_origin/
artifacts/final_runs/
artifacts/final_test/
artifacts/prediction_analysis/
artifacts/residual_analysis/
artifacts/error_by_regime/
artifacts/worst_error_analysis/
artifacts/attention_extraction/
artifacts/attention_heatmaps/
artifacts/last_query_attention/
artifacts/head_comparison/
artifacts/error_conditioned_attention/
artifacts/seed_stability_attention/
artifacts/final_tables/
artifacts/final_conclusions/
```

Không thêm `.gitkeep` trong task này. Vì vậy các empty directories sẽ tồn tại trên filesystem nhưng Git sẽ không lưu chúng nếu không có file bên trong.

### 7.9. Documentation support directories

Giữ các docs hiện tại và tạo thêm:

```text
docs/ai_interaction_logs/
docs/ai_interaction_logs/phases/
docs/ai_interaction_logs/issues/
docs/ai_interaction_logs/refactors/
```

Trong `docs/result/`, tạo:

```text
phase_00/
phase_01/
...
phase_59/
```

Không tạo `docs/audits/` vì `docs/code_base_audit.md` đã tồn tại ở current path.

Không tạo `docs/rules/` vì `docs/RULE_BASE/` đã tồn tại.

Không tạo `plan_to_refactor_fix/` vì `plan_to_refactor&fix/` đã tồn tại.

---

## 8. Files and paths that must not be changed

```text
COURSE_WORK/data/raw_data/energydata_complete.csv
COURSE_WORK/README.md
COURSE_WORK/requirements.txt
COURSE_WORK/working_rule.md
COURSE_WORK/docs/RULE_BASE/rule_code.md
COURSE_WORK/docs/code_base_audit.md
COURSE_WORK/docs/plan-doc/plan_detail_for_each_phase/*
COURSE_WORK/docs/plan-doc/plan_overview/*
COURSE_WORK/notebook_course_work/practice_3.ipynb
Repository-root .gitignore
Mọi file/folder hiện hữu khác
```

Không được:

```text
rename
move
delete
overwrite
format
populate content
git add
git commit
```

---

## 9. Sequential execution plan

### Step 0 — Precondition check

```text
Confirm no other active execution task.
Capture current git status.
Capture current file and directory inventory.
Capture hashes of all existing non-generated files.
Confirm raw CSV path and hash.
```

Stop nếu current state thay đổi ngoài dự kiến trước khi scaffold bắt đầu.

### Step 1 — Governance placeholder

```text
Create docs/RULE_BASE/architecture_rule.md as zero-byte placeholder.
Verify no existing RULE_BASE file changed.
Record that architecture governance remains incomplete.
```

Không coi Step 1 là hoàn tất `ARCHITECTURE_RULE_READ=true` cho code execution.

### Step 2 — Project and config scaffold

```text
Create pyproject.toml placeholder.
Create configs hierarchy.
Create approved zero-byte YAML placeholders.
Verify every new config file is zero bytes.
```

### Step 3 — Data lifecycle directories

```text
Create only approved data directories.
Do not touch raw CSV.
Do not create manifest or processed data files.
Verify raw CSV hash unchanged.
```

### Step 4 — Source package scaffold

```text
Create package directories.
Create approved zero-byte .py files.
Verify no source file contains code/comment/docstring/icon.
```

### Step 5 — Phase entry-point scaffold

```text
Create Phase 00–59 zero-byte entry-point files sequentially.
Verify one-to-one filename coverage against existing Phase detail docs.
Do not create unapproved Phase 60+ files.
```

### Step 6 — Test hierarchy

```text
Create approved test directories only.
Do not create test logic.
```

### Step 7 — Maintenance scaffold

```text
Create maintenance directories and zero-byte script placeholders.
Do not execute the placeholders.
```

### Step 8 — Artifact directories

```text
Create approved artifact directories sequentially.
Do not create generated artifact files.
Do not create checkpoints or reports.
```

### Step 9 — Documentation support directories

```text
Create AI interaction log directories.
Create docs/result/phase_00 through phase_59 directories.
Do not duplicate current docs paths.
```

### Step 10 — Final validation

```text
Compare before/after inventories.
Confirm all changes are additive.
Confirm every new placeholder file is zero bytes.
Confirm all existing-file hashes are unchanged.
Confirm raw CSV hash is unchanged.
Confirm no file was moved, renamed or deleted.
Confirm no duplicate equivalent paths were created.
Review git status without staging.
Produce refactor report.
```

Mỗi Step phải hoàn thành và được verify trước khi sang Step tiếp theo.

---

## 10. Validation strategy

### 10.1. Filesystem validation

```text
All approved directories exist.
All approved placeholder files exist.
No unapproved path exists.
No existing path disappeared.
```

### 10.2. Content validation

```text
All newly created .py files are 0 bytes.
All newly created .yaml files are 0 bytes.
pyproject.toml is 0 bytes.
architecture_rule.md is 0 bytes.
No JSON/CSV/JSONL artifact placeholder was created.
```

### 10.3. Preservation validation

```text
Existing-file hash manifest before = after.
Raw CSV hash before = after.
Existing notebook remains untouched.
Existing documentation remains untouched.
```

### 10.4. Scope validation

```text
No code implementation.
No configuration values.
No tests executed against empty modules.
No package installation.
No dependency change.
No Git staging or commit.
```

---

## 11. Risks and limitations

### Risk 1 — Empty architecture rule

`architecture_rule.md` rỗng chỉ là placeholder. Nó chưa giải quyết governance blocker.

### Risk 2 — Empty configs

YAML placeholders rỗng không phải operational configs. Không được load chúng trước Phase/config implementation.

### Risk 3 — Empty pyproject

`pyproject.toml` rỗng chưa cài đặt package layout và chưa xác định dependencies/tooling.

### Risk 4 — Empty Python modules

Các module tồn tại nhưng chưa có contract hoặc implementation. File existence không đồng nghĩa Phase đã hoàn thành.

### Risk 5 — Empty directories and Git

Git không track empty directories. Cấu trúc artifact/test/result rỗng sẽ không được preserve sau clone nếu không thêm `.gitkeep` hoặc file thật trong task sau.

### Risk 6 — Root `.gitignore`

Pattern `data/` trong root `.gitignore` có thể ignore toàn bộ `COURSE_WORK/data/`, bao gồm future manifest files. Không sửa `.gitignore` trong scope này; vấn đề cần được review riêng trước khi quyết định provenance policy.

### Risk 7 — Existing zero-byte notebook

`practice_3.ipynb` hiện có kích thước 0 byte và chưa phải notebook JSON hợp lệ. Task scaffold không sửa file này.

### Risk 8 — Phase plan and LoRA mismatch

Requirement plan đã có pretraining, LoRA và partial fine-tuning, nhưng Phase detail hiện chỉ có Phase 0–59 của protocol cũ. Vì vậy không tạo thêm phase entry points cho LoRA cho đến khi Phase plan được cập nhật và phê duyệt.

### Risk 9 — Source naming conflict

Một số Phase detail nhắc filename khác kiến trúc mẫu. `architecture_rule.md` phải chốt canonical names trước implementation để tránh duplicate module ownership.

---

## 12. Alternatives considered

### Alternative A — Clone kiến trúc mẫu nguyên xi

Không chọn vì tạo duplicate paths, đổi tên current structure và bỏ sót LoRA/current Phase artifact contracts.

### Alternative B — Move/rename current files cho giống sample

Không chọn vì trái yêu cầu giữ nguyên cấu trúc hiện tại và tạo architecture-impacting migration.

### Alternative C — Chỉ tạo top-level directories

Ít rủi ro hơn nhưng không đạt mục tiêu scaffold file/module theo sample.

### Alternative D — Populate tất cả configs/modules ngay

Không chọn vì user chỉ cho phép tạo file/folder và cấm code trong task này.

### Selected approach

```text
Additive-only scaffold
+
preserve current equivalents
+
zero-byte source/config placeholders
+
directories-only for generated artifacts
+
explicit LoRA-aware additions
```

---

## 13. Acceptance criteria

Task scaffold chỉ được PASS khi:

```text
User đã approve plan này.
Execution diễn ra tuần tự.
Chỉ approved paths được tạo.
Không file hiện hữu nào bị sửa.
Không file/folder nào bị move/rename/delete.
Không code/config content được thêm.
Không generated artifact giả được tạo.
Không duplicate current-equivalent path được tạo.
Raw CSV và docs hiện tại giữ nguyên hash.
Git status được report nhưng không stage.
Remaining blockers được báo cáo trung thực.
```

---

## 14. Stop conditions

Execution phải dừng nếu:

```text
Một target path đã tồn tại nhưng không rỗng.
Phát hiện concurrent user change tại target path.
Cần sửa hoặc move current file để tiếp tục.
Cần thêm content vào placeholder.
Cần tạo Phase 60+ chưa có Phase detail.
Cần thay .gitignore.
Cần tạo artifact data giả.
Một step tạo path ngoài approved list.
User thay đổi scope.
```

---

## 15. Explicit non-goals

```text
Không implement data pipeline.
Không implement model.
Không implement LoRA.
Không implement training/evaluation.
Không viết tests.
Không sửa notebook.
Không viết README.
Không điền requirements.
Không cấu hình pyproject.
Không điền YAML.
Không tạo manifest content.
Không chạy Phase.
Không tải dependency.
Không commit Git.
```

---

## 16. User approval gate

```text
PLAN_CREATED=true
USER_APPROVED_PREPROCESS_PLAN=false
EXECUTION_ALLOWED=false
```

Sau khi Human verify và approve plan, task scaffold mới được phép bắt đầu từ Step 0 và thực hiện tuần tự.
