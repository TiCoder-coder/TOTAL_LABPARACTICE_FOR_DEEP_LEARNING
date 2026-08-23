# Phase 34 S12 Head Sweep Selective Terminal Execution Plan

## 1. Danh tính plan

```text
Plan ID: CW-PHASE-34-S12-HEAD-SWEEP-SELECTIVE-TERMINAL-v1
Phase ID: 34
Phase name: S12 Head Sweep
Phase detail: Phase_34_S12_Head_sweep.md
Plan status: IMPLEMENTED_AWAITING_H2_TERMINAL_EXECUTION
Implementation started: true
Scientific execution started: false
Implementation completed: true
Terminal handoff ready: true
Execution mode: STRICTLY_SEQUENTIAL_TERMINAL_ONLY
Notebook role: STATIC_PRESENTATION_ONLY
```

## 2. Mục tiêu

Triển khai Phase 34 thành một controlled sweep chỉ thay đổi `num_heads`, so sánh:

```text
H2 = 2 attention heads
H4 = 4 attention heads
```

Phase 34 phải:

```text
đọc canonical handoff của Phase 33
khóa toàn bộ winner từ Phase 23 đến Phase 33
tái sử dụng chính xác H4 reference từ Phase 33
chỉ train mới H2 bằng terminal
không chạy lại notebook hoặc các Phase trước
không retrain H4
không dùng processing log làm scientific source of truth
chọn winner bằng full-precision Validation RMSE Wh
giữ Test bị khóa
tạo canonical handoff cho Phase 35
trực quan hóa kết quả bằng static HTML trong notebook
```

## 3. Trạng thái project tại thời điểm lập plan

### 3.1 Canonical Phase 33

```text
Phase 33 status: PASS
approved_for_phase34: true
S11 winner: D64
Selected d_model: 64
Current num_heads: 4
Current head_dim: 16
Current reference run: RUN_TR_S09_0016_AE0FB819
Validation RMSE: 58.08190056355405 Wh
Validation MAE: 27.595002038670813 Wh
Validation R2: 0.6034923842647237
Population fingerprint: a40ded8802e90008535d268720bad9e9dcca5eee1436ddb359deea3ad39a1987
Test status: FORBIDDEN
```

Canonical upstream artifacts hiện có:

```text
artifacts/sweeps/S11_d_model/phase_33_signoff.json
artifacts/sweeps/S11_d_model/s11_d_model_winner.json
artifacts/sweeps/S11_d_model/s11_reference_update.json
artifacts/sweeps/S11_d_model/s11_d_model_metrics.csv
artifacts/sweeps/S11_d_model/s11_d_model_sweep_manifest.json
```

Derived processing log hiện có:

```text
docs/save_log_in_processing/phase_33_s11_d_model_log.json
```

Processing log chỉ được dùng để khôi phục trạng thái hiển thị và quyết định resume. Canonical artifacts, registry, run config, checkpoint, history và metrics mới là scientific evidence.

### 3.2 Phase 34 hiện tại

```text
Phase 34 source owner: absent
Phase 34 selective spec: absent
Phase 34 terminal dispatch: absent
Phase 34 canonical finalizer: absent
Phase 34 processing log: absent
Phase 34 notebook section: absent
artifacts/sweeps/S12_heads: absent
H2 scientific run: absent
H4 reusable reference: available from Phase 33, pending Phase 34 exact-match audit
```

### 3.3 Trạng thái regression hiện tại

Full repository regression gate sau corrective execution:

```text
274 passed
4 subtests passed
0 failed
0 errors
```

Các nonconformance đã được ghi tại:

```text
docs/plan-doc/analysis_error/full_test_suite_historical_state_nonconformance_after_phase_33_visualization_issue.md
```

Corrective plan đã được thực thi và verify:

```text
docs/plan-doc/plan_before_process/full_test_suite_historical_state_reconciliation_plan.md
```

Phase 34 source implementation được phép tiếp tục. Expensive H2 training vẫn chỉ được phép sau Phase 34 preflight.

### 3.4 Implementation checkpoint

```text
Architecture mapping: COMPLETE
Scientific owner: COMPLETE
Selective execution state: COMPLETE
Terminal dispatch: COMPLETE
Canonical finalizer: COMPLETE
Static HTML reporting: COMPLETE
Notebook orchestration cell: COMPLETE
Phase 34 derived processing log: COMPLETE
Phase 34 state: CONDITION_INCOMPLETE
Missing condition: H2
Reusable condition: H4
Execution readiness: true
Test access: FORBIDDEN
Focused verification: 70 passed
Full repository regression: 288 passed, 4 subtests passed
Preservation mismatches: 0
Scientific artifact root before terminal handoff: absent
```

## 4. Upstream dependencies

Phase 34 cần toàn bộ điều kiện sau:

```text
Phase 33 sign-off PASS hoặc PASS_WITH_WARNING không có critical discrepancy
approved_for_phase34 = true
s11_d_model_winner.json hợp lệ
s11_reference_update.json hợp lệ
Phase 33 output checksums khớp
current reference run tồn tại trong experiment registry
reference config, BEST checkpoint, history và Validation metrics đầy đủ
environment revision hợp lệ cho terminal training
MPS hoặc CUDA sẵn sàng theo environment contract
Test firewall được xác nhận
full repository regression gate được giải quyết
```

Không được suy ra handoff chỉ từ `phase_33_s11_d_model_log.json`.

## 5. Downstream dependencies

Phase 35 chỉ được chạy sau khi Phase 34 tạo và verify:

```text
artifacts/sweeps/S12_heads/s12_head_winner.json
artifacts/sweeps/S12_heads/s12_reference_update.json
artifacts/sweeps/S12_heads/phase_34_signoff.json
approved_for_phase35 = true
```

Phase 35 phải nhận selected head count, selected head dimension, selected run, D64, N2, F128 và toàn bộ frozen lineage từ Phase 34.

## 6. Scientific contract

```text
Sweep ID: S12_HEADS
Sweep version: SWEEP_S12_HEADS-v1
Swept field: model.num_heads
H2: num_heads = 2
H4: num_heads = 4
Reference condition: H4
New condition: H2
Reference execution: REUSE_EXACT_PHASE_33_REFERENCE
New execution: ONE_FRESH_SEED_42_RUN
Primary metric: Validation RMSE Wh
Selection direction: MIN
Exact tie rule: H2
Test access: FORBIDDEN
```

Frozen configuration hiện tại:

| Thành phần | Giá trị khóa | Nguồn |
|---|---:|---|
| Feature set | FS2_TF1 | Phase 23-24 |
| Feature count | 33 | Phase 23-24 |
| Target scaling | YS1 Train-only StandardScaler | Phase 25 |
| Lookback | L36, 36 steps, 6 hours | Phase 26 |
| Pooling | LAST_STEP | Phase 27 |
| Activation | GELU | Phase 28 |
| Batch size | 32 | Phase 29 |
| Learning rate | 0.0003 | Phase 30 |
| AdamW weight decay | 0.001 | Phase 31 |
| Dropout | 0.1 | Phase 32 |
| d_model | 64 | Phase 33 |
| Layers | 2 | Frozen contract |
| FFN width | 128 | Frozen contract |
| Loss | MSE | Frozen contract |
| Max epochs | 50 | Frozen contract |
| Early-stopping patience | 10 | Training contract |
| Early-stopping min_delta | 0 | Training contract |
| Gradient clipping | 1.0 | Frozen contract |
| Scheduler | None | Frozen contract |
| Warmup | None | Frozen contract |
| Gradient accumulation | 1 | Frozen contract |
| Seed | 42 | Reproducibility contract |
| RevIN | Disabled | Current reference |
| Boundary | WB0 context carry-over | Current reference |

Derived geometry hiện tại:

```text
H2 head_dim = 64 / 2 = 32
H4 head_dim = 64 / 4 = 16
64 % 2 = 0
64 % 4 = 0
```

Expected parameter policy:

```text
same parameter names
same parameter tensor shapes
same state_dict keys
same total parameter count
same trainable parameter count
different head partitioning
different attention head-axis size
standard 1/sqrt(head_dim) scaling remains unchanged by custom compensation
```

## 7. Những hành vi bị cấm

```text
không chạy lại toàn bộ notebook
không chạy lại Phase 1-33 để tạo Phase 34
không retrain H4
không dùng H4 BEST làm warm-start cho H2
không reuse optimizer state
không merge hoặc remap attention heads
không thay d_model để bù head count
không thay FFN width hoặc layer count
không thêm attention temperature
không thay dropout, learning rate, weight decay hoặc batch size
không thêm H1, H8, H16 hoặc candidate khác
không dùng attention aesthetics để chọn winner
không dùng runtime để override RMSE
không tạo rerun theo kết quả score
không dùng Test loader, Test prediction hoặc Test metric
không xóa output đang lưu trong notebook
không tạo widget MIME output
không đặt processing, training hoặc metric logic trong notebook
không thêm comment, docstring, icon hoặc emoji vào code mới
```

## 8. Ownership theo kiến trúc

### 8.1 `src/course_work/sweeps/heads.py`

Owner của logic khoa học riêng Phase 34:

```text
Phase 33 handoff validation
H2/H4 condition registry
dynamic selected d_model resolution
frozen configuration resolution
head dimension and divisibility audit
MHA geometry audit
architecture-role audit
parameter-schema and parameter-count audit
config-delta whitelist audit
H4 exact-reference reuse gate
H2 fresh-run preparation
forward, backward and attention API sanity
initialization, sample-order and dropout-scope audit inputs
Test firewall
Phase 34 preflight payload
```

Module này không sở hữu generic training loop, experiment registry persistence, generic artifact IO, HTML rendering hoặc notebook orchestration.

### 8.2 `src/course_work/experiments/phase_execution.py`

```text
đăng ký Phase 34 selective spec
đăng ký H2 và H4
đăng ký Phase 33 prerequisites
phân loại current Phase 34 state
xác định smallest safe action
chỉ authorize H2 khi H4 reusable và upstream hợp lệ
không authorize condition đã verified hoặc đang running
```

### 8.3 `src/course_work/experiments/sweep_recovery.py`

```text
audit direct Phase 34 recovery
không materialize hoặc rerun Phase 1-33
phân biệt canonical reusable state với derived log state
trả đúng earliest invalid dependency
```

### 8.4 `scripts/run_single_condition.py`

```text
dispatch S12_HEADS
prepare H2 config từ exact Phase 33 winner config
chỉ đổi model.num_heads từ 4 thành 2
giữ H4 ở REUSE_REFERENCE
đăng ký H2 trước training
chạy H2 qua TRAINING_ENGINE-v1
ghi canonical run artifacts và live result
```

### 8.5 `scripts/run_all_pending.py`

```text
audit Phase 34
dry-run Phase 34
execute only missing H2
finalize sau khi H2 và H4 evidence đầy đủ
refresh Phase 34 derived log
verify final state VALID_REUSABLE
```

### 8.6 `scripts/run_phase_background.py`

```text
cho phép target Phase 34
khởi chạy terminal process có status và log riêng
không mở hoặc execute notebook
```

### 8.7 `scripts/sweep_results_to_csv.py`

```text
đăng ký S12_heads
chuyển verified live result evidence thành s12_head_metrics.csv
không tạo giả H4 training record
```

### 8.8 `src/course_work/sweeps/sweep_results.py`

```text
verify đủ H2 và H4 evidence
verify H2 new run và H4 exact reuse
rank full-precision Validation RMSE
chọn H2 chỉ khi exact tie hoặc RMSE thấp hơn
tạo winner, Phase 35 reference và sign-off
điều phối materialization các required Phase 34 artifacts
không truy cập Test
```

### 8.9 `src/course_work/reporting/phase_summary.py`

```text
tạo phase_34_s12_head_log.json từ canonical artifacts
render compact persistent static HTML
hiển thị frozen configuration, H2/H4 geometry, evidence, Validation metrics, effect và winner
hiển thị exact block reason khi chưa đủ điều kiện
không hiển thị raw JSON dump, checksum dump hoặc technical detail thừa
```

### 8.10 Notebook

Notebook chỉ có một markdown cell và một code cell cho Phase 34.

Code cell dự kiến:

```python
from course_work.reporting.phase_summary import render_phase_resume
render_phase_resume(34)
```

Cell không được:

```text
parse JSON trực tiếp
construct model
construct loader
train
evaluate
select winner
write artifact
start terminal subprocess
```

## 9. Files phải đọc trước implementation

```text
working_rule.md
docs/RULE_BASE/architecture_rule.md
docs/RULE_BASE/rule_code.md
docs/plan-doc/plan_detail_for_each_phase/Phase_33_S11_d_model_sweep.md
docs/plan-doc/plan_detail_for_each_phase/Phase_34_S12_Head_sweep.md
docs/plan-doc/plan_detail_for_each_phase/Phase_35_S13_Layer_sweep.md
src/course_work/models/transformer_regressor.py
src/course_work/models/transformer_encoder_layer.py
src/course_work/attention/verification.py
src/course_work/training/engine.py
src/course_work/experiments/registry.py
src/course_work/experiments/phase_execution.py
src/course_work/experiments/sweep_recovery.py
src/course_work/sweeps/d_model.py
src/course_work/sweeps/dropout.py
src/course_work/sweeps/sweep_results.py
src/course_work/reporting/phase_summary.py
scripts/run_single_condition.py
scripts/run_all_pending.py
scripts/run_phase_background.py
scripts/sweep_results_to_csv.py
notebook_course_work/CourseWork.ipynb
tests/contracts/test_selective_execution_policy.py
tests/unit/test_condition_runner_config.py
tests/unit/test_d_model.py
tests/unit/test_phase_execution.py
tests/unit/test_sweep_finalization.py
tests/unit/test_sweep_recovery.py
tests/unit/test_selective_phase_reporting.py
tests/integration/test_notebook_boundary.py
current Phase 33 canonical artifacts
current experiment registry and reference run artifacts
```

## 10. Files dự kiến thay đổi sau khi được duyệt

```text
docs/RULE_BASE/architecture_rule.md
notebook_course_work/CourseWork.ipynb
src/course_work/sweeps/heads.py
src/course_work/experiments/phase_execution.py
src/course_work/experiments/sweep_recovery.py
src/course_work/sweeps/sweep_results.py
src/course_work/reporting/phase_summary.py
scripts/run_single_condition.py
scripts/run_all_pending.py
scripts/run_phase_background.py
scripts/sweep_results_to_csv.py
tests/contracts/test_selective_execution_policy.py
tests/unit/test_heads.py
tests/unit/test_condition_runner_config.py
tests/unit/test_phase_execution.py
tests/unit/test_sweep_finalization.py
tests/unit/test_sweep_recovery.py
tests/unit/test_selective_phase_reporting.py
tests/integration/test_notebook_boundary.py
```

Danh sách phải được thu hẹp trong implementation nếu public API hiện tại đã đáp ứng yêu cầu.

Generated files chỉ được tạo bởi canonical execution path:

```text
docs/save_log_in_processing/phase_34_s12_head_log.json
artifacts/sweeps/S12_heads/**
artifacts/runs/<new_H2_run_id>/**
```

## 11. Files không được thay đổi

```text
data/raw_data/**
data/interim/**
data/data_after_split/**
artifacts/contracts/**
valid Phase 1-33 scientific artifacts
RUN_TR_S09_0016_AE0FB819 checkpoint, config, history, predictions and metrics
Phase 33 winner, reference and sign-off
existing Test artifacts
stored notebook outputs ngoài Phase 34 cells
unrelated user changes trong dirty worktree
```

Không được xóa hoặc ghi đè evidence lịch sử. Nếu canonical output Phase 34 stale cần thay, phải archive theo revisioned history trước.

## 12. Required Phase 34 canonical outputs

Required root:

```text
artifacts/sweeps/S12_heads/
```

Required machine-readable and report outputs:

```text
s12_head_sweep_manifest.json
s12_head_sweep_contract.json
s12_head_preflight_audit.csv
s12_run_matrix.csv
s12_head_definition_audit.csv
s12_mha_geometry_audit.csv
s12_architecture_role_audit.csv
s12_parameter_schema_audit.csv
s12_parameter_count_audit.csv
s12_config_delta_audit.csv
s12_head_unit_tests.csv
s12_common_data_audit.csv
s12_head_training_audit.csv
s12_initialization_audit.csv
s12_sample_order_audit.csv
s12_dropout_scope_audit.csv
s12_attention_api_audit.csv
s12_optimizer_budget_audit.csv
s12_head_run_provenance.csv
s12_head_metrics.csv
s12_head_effect.csv
s12_head_efficiency_context.csv
s12_optimization_diagnostics.csv
s12_convergence_diagnostics.csv
s12_runtime_diagnostics.csv
s12_hypothesis_outcomes.csv
s12_head_findings.csv
s12_head_winner.json
s12_reference_update.json
s12_head_sweep_tests.csv
s12_head_discrepancies.json
s12_head_sweep_summary.json
s12_head_sweep_report.md
README_S12_HEAD_SWEEP.md
phase_34_signoff.json
```

Required figures:

```text
figures/S12_01_validation_rmse_by_epoch.png
figures/S12_02_validation_mae_by_epoch.png
figures/S12_03_train_loss_by_epoch.png
figures/S12_04_gradient_clipping_fraction.png
figures/S12_05_best_validation_metrics.png
figures/S12_06_runtime_vs_rmse.png
figures/S12_07_convergence_summary.png
```

Optional outputs may only be omitted with explicit recorded reason:

```text
s12_generalization_diagnostics.csv
figures/S12_08_generalization_gap_optional.png
peak-memory fields
```

## 13. Sequential implementation and execution plan

### Step 1. Resolve repository regression gate

Thực thi và verify corrective plan đã tồn tại trước khi mở Phase 34 source scope.

Verification:

```text
full suite runs in the signed environment
no unresolved failure or error
no Phase 33 canonical evidence changes
no Test access
```

Nếu gate không xanh, dừng Phase 34.

### Step 2. Re-audit Phase 33 canonical handoff

Read-only verify:

```text
sign-off identity and status
approved_for_phase34
input and output checksums
winner and reference identity
registry membership
run config fingerprint
BEST checkpoint presence
training history presence
Validation metrics presence
Test firewall
```

Verify ngay sau Step 2. Không dùng processing log để thay canonical evidence.

### Step 3. Capture preservation baseline

Ghi preservation manifest cho:

```text
CourseWork.ipynb cell IDs and output hashes
Phase 33 artifacts and checksums
experiment registry checksum
reference run artifact checksums
current Phase 34 path presence
```

Verify baseline reload và không có scientific write ngoài preservation manifest.

### Step 4. Extend approved architecture scope through Phase 34

Cập nhật architecture scope và Phase-to-module mapping cho `sweeps/heads.py` cùng Phase 34 terminal/reporting ownership.

Verify:

```text
single owner per logic
no source-to-notebook dependency
no notebook scientific ownership
Phase 35 remains reserved
```

### Step 5. Implement Phase 34 scientific owner

Tạo `src/course_work/sweeps/heads.py` với H2/H4 registry, Phase 33 handoff, dynamic D resolution, frozen contract, geometry, schema, parameter, config delta, reference reuse, sanity and Test-firewall gates.

Verify focused `test_heads.py` trước khi sửa dispatch.

### Step 6. Extend selective inspection

Đăng ký Phase 34 vào `phase_execution.py` và `sweep_recovery.py`.

Expected state transitions:

```text
invalid Phase 33 -> UPSTREAM_INVALID and BLOCK
valid Phase 33 plus missing H2 -> CONDITION_INCOMPLETE and EXECUTE_MISSING_ONLY
valid H2 plus reusable H4 but missing derived outputs -> DERIVED_ARTIFACT_MISSING and REBUILD_DERIVED_ONLY
complete canonical Phase 34 plus stale log -> LOG_STALE and REBUILD_LOG_ONLY
complete canonical Phase 34 plus valid log -> VALID_REUSABLE and RENDER_ONLY
```

Verify state tests immediately.

### Step 7. Extend terminal condition dispatch

Đăng ký `S12_HEADS` và Phase 34 trong terminal scripts.

Verify:

```text
H4 returns REUSE_REFERENCE
H2 is the only TRAIN_NEW condition
H2 config differs from H4 only in num_heads and derived head metadata
dry-run and audit-only produce no scientific artifacts
no notebook process starts
```

### Step 8. Implement fail-fast Phase 34 preflight

Chạy toàn bộ non-training gates theo thứ tự:

```text
Phase 33 sign-off
S11 winner identity
all frozen fields
H2/H4 registry
divisibility
derived head dimensions
same population
MHA geometry
parameter schema equality
parameter count equality
architecture-role invariance
config-delta whitelist
dropout scope
training config
H2 forward and backward
attention API
H4 reuse eligibility
Test firewall
registry readiness
```

Verify mọi gate PASS trước H2 run. Bất kỳ failure nào tạo issue và corrective plan riêng rồi dừng.

### Step 9. Execute only fresh H2 on terminal

Sau khi user duyệt plan và mọi gate xanh, terminal runner phải:

```text
register H2 run before training
seed 42
create fresh loaders
create fresh H2 model
create fresh AdamW optimizer
use standard forward without attention in training
train through TRAINING_ENGINE-v1
write run config, checkpoints, history, Validation predictions and metrics
append one H2 live result
never load H4 weights or optimizer state
```

Verify H2 ngay sau execution:

```text
run status COMPLETE
config num_heads = 2
head_dim = 32
BEST checkpoint metadata matches H2
predictions finite
Validation metrics finite and in Wh
full ordered Validation population matches H4
no Test artifact
```

Không chạy bước tiếp theo nếu verify thất bại.

### Step 10. Reuse and verify H4

Bind H4 evidence trực tiếp tới:

```text
RUN_TR_S09_0016_AE0FB819
```

Verify exact frozen configuration, population, checkpoint, history, Validation metrics and config fingerprint. Không tạo H4 run mới.

### Step 11. Finalize canonical Phase 34 outputs

Tạo required artifacts từ verified H2 run và reused H4 evidence.

Winner logic:

```text
if RMSE_H2 < RMSE_H4: H2 wins
if RMSE_H4 < RMSE_H2: H4 wins
if full-precision RMSE_H2 == RMSE_H4: H2 wins
```

Verify từng artifact reload, schema, row count, lineage and checksum ngay sau write.

### Step 12. Verify Phase 35 handoff

Verify:

```text
selected_num_heads
selected_head_dim
selected_d_model = 64
winner_run_id
winner_config_fingerprint
current_num_layers = 2
ffn_dim = 128
population_fingerprint
metric_version
approved_for_phase35 = true
Test status remains FORBIDDEN
```

### Step 13. Build derived processing log and static HTML

Tạo:

```text
docs/save_log_in_processing/phase_34_s12_head_log.json
```

Notebook presentation chỉ hiển thị phần cần thiết:

```text
Phase status
Frozen configuration
H2/H4 geometry
Reference reuse and new-run provenance
Validation RMSE, MAE and R2 comparison
RMSE effect and winner
Phase 35 handoff
block reason only when applicable
```

Không hiển thị raw JSON, checksum dump, widget hoặc technical table dài.

### Step 14. Add Phase 34 notebook presentation cells

Thêm Phase 34 sau Phase 33 và trước All Phase Logs.

Verify:

```text
stable cell IDs
one reporting API call only
no JSON parsing
no processing or training
no output deletion outside new Phase 34 cells
stored output is static text/html
no application/vnd.jupyter.widget-view+json
```

### Step 15. Sequential regression verification

Thứ tự test:

```text
test_heads.py
test_condition_runner_config.py
test_phase_execution.py
test_sweep_recovery.py
test_sweep_finalization.py
test_selective_phase_reporting.py
test_selective_execution_policy.py
test_notebook_boundary.py
all Phase 31-34 targeted tests
full repository test suite
```

Sau mỗi nhóm, dừng nếu có lỗi, tạo issue và corrective plan, fix sau approval rồi chạy lại từ nhóm bị lỗi và liên kết lại toàn bộ nhóm trước.

### Step 16. Final verification and execution report

Chỉ kết luận Phase 34 hoàn thành khi:

```text
inspect_phase_state(34) = VALID_REUSABLE
resolve action = RENDER_ONLY
all required canonical outputs reload and checksum-valid
processing log source hashes match canonical outputs
notebook Phase 34 static HTML persists
All Phase Logs counts Phase 34 exactly once
Phase 34 status displays PASS, not VALID_REUSABLE
full regression suite passes
Test remains untouched
```

Sau đó tạo execution report tại `docs/plan-doc/plan_to_refactor&fix/`.

## 14. Terminal command contract sau khi implementation được duyệt và verify

Các command chính thức sẽ chỉ được khóa sau khi Step 7 tests pass. Expected interface:

```text
python scripts/run_phase_background.py sweep --target-phase 34 --audit-only --foreground
python scripts/run_phase_background.py sweep --target-phase 34 --dry-run
python scripts/run_phase_background.py sweep --target-phase 34
python scripts/run_phase_background.py --status
```

Runner phải selective-resume theo canonical state:

```text
H2 missing -> run H2 only
H2 complete and derived artifacts missing -> finalize only
canonical artifacts complete and log missing -> rebuild log only
all valid -> render only
```

Không được đưa lệnh scientific execution cho người dùng trước khi source implementation và preflight tests hoàn tất.

## 15. Validation matrix

| Gate | PASS condition | Failure action |
|---|---|---|
| Phase 33 | canonical sign-off, winner and reference valid | stop |
| Architecture | Phase 34 owner mapping approved | stop |
| Registry | H2/H4 registered, H4 reusable | stop |
| Geometry | D divisible, head dimensions correct | stop |
| Parameters | names, shapes and counts equal | stop |
| Config delta | only head fields differ | stop |
| Population | Train and Validation identities match | stop |
| Attention API | H2 `[B,2,L,L]`, H4 contract `[B,4,L,L]` | stop |
| Training | H2 fresh run complete | stop |
| BEST | checkpoint metadata and Validation verification match | stop |
| Selection | minimum full-precision Validation RMSE | stop |
| Test firewall | no Test access | stop |
| Handoff | Phase 35 fields complete | stop |
| Reporting | static HTML from derived JSON | stop |
| Regression | targeted and full suite pass | stop |

## 16. Acceptance criteria

Phase 34 chỉ PASS khi:

```text
Phase 33 is valid and approved for Phase 34
exactly H2 and H4 are represented
D64 is fixed dynamically from Phase 33 rather than hard-coded as an assumption
H2 head_dim is 32 and H4 head_dim is 16
only num_heads and derived head geometry differ
parameter schema and count equality pass
same Train and Validation population is used
H4 exact Phase 33 reference is reused
one fresh seed-42 H2 run is complete
no H4 retraining or H4-to-H2 warm-start occurs
H2 forward, backward and attention API pass
H2 BEST verification passes
winner is selected by full-precision Validation RMSE
exact tie selects H2
runtime and attention diagnostics remain secondary
all required artifacts and figures are generated or optional omission is documented
Phase 35 reference is valid and approved
processing log is derived and checksum-linked
notebook only renders static HTML
All Phase Logs includes Phase 34 once with scientific PASS status
all targeted tests pass
full repository suite passes
Test remains forbidden
```

## 17. Rollback and preservation

Nếu implementation hoặc execution thất bại:

```text
không sửa hoặc xóa Phase 33 evidence
không xóa H2 failed-run evidence
mark failed run through registry lifecycle
archive stale Phase 34 derived outputs before replacement
restore notebook only from captured Phase 34 preservation baseline
record issue and corrective plan
resume from smallest invalid Phase 34 action
```

Không dùng destructive Git command hoặc global notebook output clearing.

## 18. Stop conditions

Phải dừng ngay khi:

```text
Phase 33 checksum or lineage invalid
full repository regression gate unresolved
environment is not training-ready
H4 exact reference cannot be proven
selected d_model is not divisible by both head counts
parameter schema or count differs unexpectedly
config delta contains a non-head scientific field
population or sample order cannot be reconciled
H2 sanity fails
attention API contract fails
Test access is detected
scientific write occurs during audit-only or dry-run
notebook requires processing logic
user changes overlap the exact implementation scope and cannot be preserved
```

## 19. Approval gate

```text
PLAN_CREATED=true
PLAN_VERIFIED_AGAINST_CURRENT_PHASE_33_STATE=true
IMPLEMENTATION_STARTED=true
SCIENTIFIC_EXECUTION_STARTED=false
WAITING_FOR_USER_APPROVAL=false
USER_APPROVAL_GRANTED=2026-08-23
BLOCKED_BY_UNRESOLVED_FULL_REGRESSION=false
```

Sau khi người dùng duyệt plan này, thứ tự bắt buộc vẫn là:

```text
resolve and verify repository regression gate
implement Phase 34 sequentially
verify source and preflight
provide exact terminal command
wait for terminal H2 completion when required
finalize and verify artifacts
add static notebook presentation
run full regression
report completion
```

Không code, không tạo Phase 34 artifact và không chạy H2 trước approval.
