# PHASE34_41_RECONSTRUCTION_PRE_PROCESS_PLAN

> **Status: SUPERSEDED_NOT_EXECUTED.** Kế hoạch reconstruction lineage mới này
> không được thực thi. Historical Phase 34–41 sau đó đã được phục hồi/xác minh
> bằng exact-signoff/registry-bound compatibility và hiện là
> `VALID_REUSABLE/RENDER_ONLY`; không có reconstruction training run. Xem
> [FINAL_CONSISTENCY_AUDIT_20260913.md](../../current_flow/FINAL_CONSISTENCY_AUDIT_20260913.md).

## 1. Plan identity

```text
Plan ID: PHASE34_41_RECONSTRUCTION_PRE_PROCESS_PLAN
Plan version: PHASE34_41_RECONSTRUCTION-v1
Plan status: PREPARED_FOR_IMPLEMENTATION_REVIEW
Scientific execution status: NOT_STARTED
Training authorized: NO
Test access: FORBIDDEN
Lineage class for every new output: RECONSTRUCTED_LINEAGE
Artifact namespace: artifacts/reconstruction/phase34_41_v1/
Canonical upstream handoff: Phase 33
```

Tài liệu này định nghĩa một lineage reconstruction mới cho Phase 34–41. Đây không phải quy trình phục hồi byte-exact các artifact lịch sử. Các artifact/signoff Phase 34–41 hiện hữu phải được giữ nguyên, không được sửa checksum, ghi đè hoặc dùng file thay thế để làm validator lịch sử PASS.

## 2. Objective and scope

Mục tiêu là chạy lại chuỗi controlled sweep từ Phase 34, bắt đầu từ handoff Phase 33 đang hợp lệ, đồng thời giữ toàn bộ kết quả mới trong namespace riêng và bảo toàn provenance.

Trong phạm vi:

- khóa upstream Phase 33 trước Phase 34;
- chuẩn bị implementation, preflight, focused tests và Human training gate;
- sau Human approval, thực thi tuần tự Phase 34 → Phase 41;
- tạo đầy đủ config, metrics, history, predictions, checkpoint, audit, manifest, signoff và checksum cho lineage mới;
- hỗ trợ resume an toàn, không retrain run đã hoàn thành hợp lệ.

Ngoài phạm vi:

- byte-exact recovery Phase 34–41 lịch sử;
- sửa hoặc overwrite `artifacts/sweeps/S12_heads/` đến `artifacts/sweeps/S19_boundary_protocol/`;
- sửa historical run records hoặc historical signoffs;
- sửa Phase 43+;
- dùng Test cho training, selection, preflight hoặc audit;
- training trong task chuẩn bị plan này;
- `git add`, `commit` hoặc `push`.

## 3. Authoritative sources

Thứ tự ưu tiên:

1. Phase 33 canonical machine-readable evidence hiện tại.
2. Phase-detail plan cho Phase 34–41.
3. Source code và focused tests hiện hữu để hiểu scientific behavior.
4. Artifact Phase 34–41 lịch sử chỉ để audit/đối chiếu; không được dùng làm canonical reconstruction result.

Các source chính:

```text
artifacts/sweeps/S11_d_model/phase_33_signoff.json
artifacts/sweeps/S11_d_model/s11_d_model_winner.json
artifacts/sweeps/S11_d_model/s11_reference_update.json
artifacts/sweeps/S11_d_model/s11_d_model_sweep_manifest.json
artifacts/sweeps/S11_d_model/s11_d_model_metrics.csv
artifacts/runs/RUN_TR_S09_0016_AE0FB819/config.json
artifacts/runs/RUN_TR_S09_0016_AE0FB819/status.json
artifacts/runs/RUN_TR_S09_0016_AE0FB819/metrics/best_validation_metrics.json
artifacts/scaling/scaler_registry.json
artifacts/scaling/scaling_manifest.json
artifacts/splits/split_manifest.json
artifacts/windows/window_manifest.json
docs/plan/plan_detail_for_each_phase/Phase_34_S12_Head_sweep.md
docs/plan/plan_detail_for_each_phase/Phase_35_S13_Layer_sweep.md
docs/plan/plan_detail_for_each_phase/Phase_36_S14_FFN_sweep.md
docs/plan/plan_detail_for_each_phase/Phase_37_S15_Loss_sweep.md
docs/plan/plan_detail_for_each_phase/Phase_38_S16_Epoch-cap_sweep.md
docs/plan/plan_detail_for_each_phase/Phase_39_S17_Gradient-clipping_sweep.md
docs/plan/plan_detail_for_each_phase/Phase_40_S18_RevIN_sweep.md
docs/plan/plan_detail_for_each_phase/Phase_41_S19_Boundary-protocol_check.md
```

## 4. Frozen Phase 33 upstream handoff

### 4.1 Canonical identity

| Field | Locked value |
|---|---|
| Phase 33 status | `PASS` |
| Approved for Phase 34 | `true` |
| Winner condition | `D64` |
| Winner/source run | `RUN_TR_S09_0016_AE0FB819` |
| Winner config fingerprint | `ae0fb819cd0ecca797892a5f6e2f9758c7809c01f5ab92daaeca21a880d5a6ba` |
| Population fingerprint | `a40ded8802e90008535d268720bad9e9dcca5eee1436ddb359deea3ad39a1987` |
| Validation sample count | `2960` |
| Seed | `42` |
| Test status | `FORBIDDEN` |

### 4.2 Current source checksums

Các checksum này khóa đúng file hiện tại trước implementation. Preflight phải tính lại và so sánh, không được tự cập nhật expected value khi mismatch.

| Source | SHA-256 |
|---|---|
| `artifacts/sweeps/S11_d_model/phase_33_signoff.json` | `a99ffff2a15ffab5e97f30db093ad31a91c0f7626dc2d19a50c464ae1a93ea7a` |
| `artifacts/sweeps/S11_d_model/s11_d_model_winner.json` | `08b49539a0428d84a8cbbe86f1d3674b3e80c8511ba893dfaf2951f4eb00ee88` |
| `artifacts/sweeps/S11_d_model/s11_reference_update.json` | `944043e5fe8bbb508d6b85de161b05d12943518f4150a424ff153359f176b933` |
| `artifacts/sweeps/S11_d_model/s11_d_model_sweep_manifest.json` | `90031af5cf34c3ace4b5708bd94baaea26eea7165a3a531d8f7568a0ff0c3a08` |
| `artifacts/sweeps/S11_d_model/s11_d_model_metrics.csv` | `aa2332774eb56ad547fd8bcc4c265fdea50e531396c8c2fb0a54dad564bdbefb` |
| `artifacts/runs/RUN_TR_S09_0016_AE0FB819/config.json` | `d8bcd4ae977a15a6658c5ff969f8ca1eb46c84b03cbaeb02f14a03d000921cd4` |
| `artifacts/runs/RUN_TR_S09_0016_AE0FB819/status.json` | `1b51f349ed680a8de00fcf931322e273b6d905088fb5db062790b17abd68c376` |
| `artifacts/runs/RUN_TR_S09_0016_AE0FB819/metrics/best_validation_metrics.json` | `2120573c95273770bfa64bf0d365f4f337fe7b478fc8fd67d2eb3f19845cd95d` |

### 4.3 Locked scientific configuration

| Group | Locked value |
|---|---|
| Data | `FS2_TF1`, 33 features, `L36`, horizon 1, `WB0_CONTEXT_CARRY_OVER` |
| Target | `Appliances`, `YS1` |
| Model | Transformer Encoder, `d_model=64`, `num_heads=4`, `num_layers=2`, `ffn_dim=128` |
| Model details | `GELU`, `LAST_STEP`, sinusoidal PE, `norm_first=false`, dropout `0.1` |
| Training | AdamW, LR `3e-4`, weight decay `1e-3`, MSE |
| Budget | batch `32`, max epochs `50`, patience `10`, `min_delta=0` |
| Gradient | global clip enabled, max norm `1.0` |
| Scheduler/warmup | `OFF` / `OFF` |
| Reproducibility | seed `42`, dataloader seed `42`, deterministic algorithms enabled |
| Selection | lowest full-precision Validation RMSE in Wh |

### 4.4 Locked lineage fingerprints

| Field | Locked value |
|---|---|
| Dataset fingerprint | `2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d` |
| Global split fingerprint | `4d0115b92a3c81406b62deb4d36dd8ceac847f948294cdc2e4dc49338ebeb821` |
| Train fingerprint | `886b4be193059e3002f444dbe9adf6e579c4b7e1e264357d119b1d8d6ecf42e4` |
| Validation fingerprint | `4ac9af3728fee4f26805984d4a415ca4a3957da7361d5296bd19c072ccd4f243` |
| Feature fingerprint | `fc9c428964285d1ad74e97f3ae2c18e7efeb3e61d61f7acd0b54039bc2ca6dee` |
| Window fingerprint `L036_H01_WB0` | `d4fc515a7f75f108a484e0d10cdf5259370af6181b6b5b88cdb1cdf501fa6bca` |
| Dataloader fingerprint | `48dff0f10ac2165443f3e2ab60f93d3d1ce9149b70c985fbbe31f4059eb4cf7e` |
| Metric contract fingerprint | `4509825a7be2ee75220f88bedf31da6d062e6fee5168af45b0d54ec1db137198` |
| Optimizer config fingerprint | `388ec31e6959b60d5d13667a4ef30d6f9f66efe96819b7b3ff91c227e0630800` |

### 4.5 Locked scaler lineage

| Scaler | Fit population | Artifact SHA-256 | Statistics fingerprint |
|---|---|---|---|
| `XSCALER__FS2_TF1` | Train only, 13,814 rows | `4e7c96d5accc2917855ed669b33d2666b3aa07f6b83b40097a9690c6a5f5e334` | `d16491699be80721d75f5710299cfb010e0b47999ee8491938ee85a04096f906` |
| `YSCALER__YS1` | Train only, 13,814 rows | `b3326a79da81b092460ef8d4a140a101b21f2433ff30c36971d626d2a2491697` | `e934dbdfe7ab196ea70e43106854a5a77159c7d350d6e7120ec02b6684da2524` |

Không refit scaler trong Phase 34 reconstruction. Mọi condition phải dùng đúng hai frozen scaler trên. Validation và Test không được tham gia scaler fitting.

## 5. Train/Validation-only data firewall

Reconstruction chỉ được dùng Train và Validation. Test rows, Test feature values, Test targets, Test predictions, Test metrics và Test-derived thresholds đều bị cấm.

Locked structural boundary từ `split_manifest.json`:

```text
Train rows: 13,814
Validation rows: 2,960
Pre-Test row count: 16,774
Validation end: 2016-05-07 04:30:00
Test start: 2016-05-07 04:40:00
```

Implementation bắt buộc:

- load feature CSV với `usecols` chính xác và giới hạn `nrows=16774`, hoặc dùng nguồn pre-Test riêng đã được xác minh;
- không load full-data CSV rồi filter trong memory;
- không mở mixed-population `window_index.csv` hoặc `common_target_population.csv` để materialize Test records;
- assert mọi loaded timestamp `< 2016-05-07 04:40:00`;
- assert mọi target split thuộc `{TRAIN, VALIDATION}`;
- assert `test_rows_read=0`, `test_target_ids_seen=0`;
- Validation chỉ dùng cho metric, early stopping và selection;
- Phase 41 chỉ xây WB0/WB1 cho Validation sensitivity; không xây Test windows.

Metadata cấu trúc trong signed manifest có thể được dùng để khóa boundary, nhưng không được dùng Test metric/target value.

## 6. Reconstruction lineage and namespace contract

Mọi file mới phải nằm dưới:

```text
artifacts/reconstruction/phase34_41_v1/
```

Không được ghi vào:

```text
artifacts/sweeps/S12_heads/
artifacts/sweeps/S13_layers/
artifacts/sweeps/S14_ffn/
artifacts/sweeps/S15_loss/
artifacts/sweeps/S16_epoch_cap/
artifacts/sweeps/S17_gradient_clipping/
artifacts/sweeps/S18_revin/
artifacts/sweeps/S19_boundary_protocol/
artifacts/runs/
artifacts/final_model_lock/
artifacts/final_test/
```

Mọi JSON/CSV mới phải chứa hoặc kế thừa các field:

```text
lineage_class = RECONSTRUCTED_LINEAGE
reconstruction_version = PHASE34_41_RECONSTRUCTION-v1
namespace = artifacts/reconstruction/phase34_41_v1/
historical_artifact_replacement = false
test_status = FORBIDDEN
```

Run ID format dự kiến:

```text
RUN_RECON_P<PHASE>_<SWEEP>_<CONDITION>_<SEQUENCE>_<CONFIG8>
```

Registry reconstruction phải riêng biệt:

```text
artifacts/reconstruction/phase34_41_v1/registry/reconstruction_registry.jsonl
```

Không đăng ký run reconstruction vào historical V1 registry nếu việc đó làm lẫn namespace hoặc lineage.

## 7. Common scientific execution contract

Cho mọi run mới:

- seed phải được set trước khi tạo model và DataLoader;
- model khởi tạo fresh;
- optimizer AdamW khởi tạo fresh từ model parameters của đúng run;
- optimizer state và scheduler state không được reuse;
- không load historical checkpoint để warm-start;
- data order, seed policy, loss và training budget phải giống condition control trừ primary factor của phase;
- một phase chỉ thay đúng một primary factor;
- checkpoint selection dùng earliest strict minimum Validation RMSE Wh;
- metric được tính trên inverse-transformed prediction/target ở Wh;
- full-precision metric là source of truth;
- failed/interrupted run phải được giữ lại, không xóa hoặc ghi đè;
- completed run chỉ được reuse sau khi config, status, checkpoint và checksum đều PASS.

## 8. Sequential phase matrix

| Phase | Sweep | Primary factor | Conditions | Reference rule | New training under normal flow |
|---:|---|---|---|---|---:|
| 34 | `S12_HEADS` | `model.num_heads` | `H2=2`, `H4=4` | H4 = exact Phase 33 upstream reference; H2 fresh | 1 |
| 35 | `S13_LAYERS` | `model.num_layers` | `N1=1`, `N2=2` | exact inherited reference reused; alternate fresh | 1 |
| 36 | `S14_FFN` | `model.ffn_dim` | `F64`, `F128`, `F256` | exact inherited width reused; two alternates fresh | 2 |
| 37 | `S15_LOSS` | `training.loss_name` | `L0=MSE`, `L1=Huber(delta=1.0 model-space)` | exact inherited loss reused; alternate fresh | 1 |
| 38 | `S16_EPOCH_CAP` | `training.max_epochs` | `E50=50`, `E100=100` | exact inherited cap reused; alternate fresh | 1 |
| 39 | `S17_GRADIENT_CLIPPING` | clipping policy | `GC0=OFF`, `GC1=global norm 1.0` | exact inherited policy reused; alternate fresh | 1 |
| 40 | `S18_REVIN` | `model.revin_enabled` | `RN0=OFF`, `RN1=ON` | RN1 only if exactly one historical `Appliances` input exists | 0 or 1 |
| 41 | `S19_BOUNDARY_PROTOCOL` | `window_boundary_protocol` | `WB0`, `WB1` | Validation-only sensitivity, not ordinary tuning | contract-dependent |

Mỗi phase phải resolve reference từ reconstructed handoff của phase ngay trước, không hard-code winner tương lai. Historical Phase 34–41 metrics không được quyết định reconstructed winner.

## 9. Phase progression gates

Chuỗi bắt buộc:

```text
Phase 33 canonical PASS
→ Phase 34 reconstructed PASS
→ Phase 35 implementation/preflight gate
→ Phase 35 reconstructed PASS
→ ...
→ Phase 41 reconstructed signoff
```

Phase 35 tuyệt đối không được mở nếu Phase 34 reconstruction chưa có:

```text
phase_34_signoff.json: status=PASS or PASS_WITH_WARNING
approved_for_phase35=true
winner/reference/config checksum PASS
population/scaler/leakage/Test firewall PASS
checksums.sha256 complete
```

Tương tự, Phase `N+1` chỉ được mở từ reconstructed signoff của Phase `N`. Historical signoff không được bypass reconstructed gate.

## 10. Complete output contract

### 10.1 Per-run outputs

Mỗi run directory:

```text
artifacts/reconstruction/phase34_41_v1/runs/<RUN_ID>/
├── config.json
├── status.json
├── training.log
├── training_history.csv
├── checkpoints/
│   ├── best_checkpoint.pt
│   └── checkpoint_manifest.json
├── metrics/
│   └── best_validation_metrics.json
├── predictions/
│   └── best_validation_predictions.csv
└── audit/
    ├── data_population_audit.json
    ├── scaler_lineage_audit.json
    ├── initialization_audit.json
    └── test_firewall_audit.json
```

`training_history.csv` tối thiểu phải giữ:

```text
epoch
train_loss
validation_rmse_wh
validation_mae_wh
validation_r2
learning_rate
gradient_norm_pre_clip
gradient_clip_applied
epoch_runtime_seconds
```

`best_validation_predictions.csv` tối thiểu phải giữ ordered target identity, timestamp, `y_true_wh`, `y_pred_wh`, residual và run/config fingerprint. Không ghi Test row.

### 10.2 Per-phase outputs

```text
artifacts/reconstruction/phase34_41_v1/phase_<NN>/
├── phase_<NN>_contract.json
├── phase_<NN>_preflight.json
├── phase_<NN>_run_matrix.csv
├── phase_<NN>_metrics.csv
├── phase_<NN>_winner.json
├── phase_<NN>_reference_update.json
├── phase_<NN>_manifest.json
├── phase_<NN>_provenance_audit.csv
├── phase_<NN>_population_audit.csv
├── phase_<NN>_scaler_audit.csv
├── phase_<NN>_leakage_audit.json
├── phase_<NN>_discrepancies.json
├── phase_<NN>_signoff.json
└── checksums.sha256
```

### 10.3 Checksum/signoff rules

- SHA-256 được tính sau khi file đã đóng và flush;
- manifest liệt kê input/output path, checksum, schema version và row count;
- checkpoint manifest giữ SHA-256, parameter count, model/config fingerprint, best epoch và parent run identity;
- signoff không tự chứa checksum của chính nó;
- signoff chỉ PASS khi mọi required output tồn tại và checksum khớp;
- warning phải có code, scope, scientific impact và blocking status;
- missing metrics/history/predictions/checkpoint là hard failure, không được hạ xuống warning;
- không được sửa expected checksum để hợp thức hóa file thay thế.

## 11. Phase 34 reconstruction contract

### 11.1 Scientific question

So sánh `H2` và `H4` khi chỉ thay `model.num_heads`, giữ nguyên Phase 33 winner ở mọi field khác.

```text
H4 = 4 heads, canonical Phase 33 upstream reference, read-only
H2 = 2 heads, one fresh reconstruction run
d_model = 64
head_dim(H4) = 16
head_dim(H2) = 32
selection metric = Validation RMSE Wh
exact tie = H2
Test = FORBIDDEN
```

H4 source phải được reuse trực tiếp bằng immutable reference đến `RUN_TR_S09_0016_AE0FB819`; không copy checkpoint thành run reconstruction và không relabel historical run. Phase 34 reconstructed summary outputs vẫn mang `RECONSTRUCTED_LINEAGE` và ghi rõ `reference_lineage=PHASE33_CANONICAL_SOURCE`.

### 11.2 One-primary-change contract

Allowed delta cho H2:

```text
model.num_heads: 4 -> 2
```

Derived field được phép:

```text
head_dim: 16 -> 32
```

Mọi delta khác ở data, feature order, lookback, target scaling, model width/depth/FFN, optimizer, LR, weight decay, dropout, loss, batch size, epoch/early stopping, clipping, scheduler, seed, population hoặc scaler phải BLOCK.

### 11.3 Phase 34 config fingerprints

```text
Base Phase 33 config fingerprint:
ae0fb819cd0ecca797892a5f6e2f9758c7809c01f5ab92daaeca21a880d5a6ba

H4 source config fingerprint:
ae0fb819cd0ecca797892a5f6e2f9758c7809c01f5ab92daaeca21a880d5a6ba

H2 reconstruction config fingerprint:
HUMAN_TO_LOCK_AFTER_IMPLEMENTATION_PREFLIGHT
```

H2 fingerprint phải được tạo từ canonical serialization của complete config, bao gồm reconstruction namespace/lineage, rồi ghi vào Phase 34 config snapshot trước Human training approval. Không được dùng historical H2 fingerprint vì đây là lineage mới.

## 12. Phase 34 preflight design

Preflight dự kiến:

```text
python -m course_work.reconstruction.phase34_41.runner \
  --phase 34 \
  --mode preflight \
  --seed 42
```

Preflight phải là no-training và không tạo scientific run ID. Output tối thiểu:

```text
plan_id=PHASE34_41_RECONSTRUCTION_PRE_PROCESS_PLAN
phase=34
lineage_class=RECONSTRUCTED_LINEAGE
namespace=artifacts/reconstruction/phase34_41_v1/
upstream_phase33_status=PASS
upstream_run_id=RUN_TR_S09_0016_AE0FB819
base_config_fingerprint=ae0fb819cd0ecca797892a5f6e2f9758c7809c01f5ab92daaeca21a880d5a6ba
h2_config_fingerprint=<LOCKED_SHA256>
population_fingerprint=a40ded8802e90008535d268720bad9e9dcca5eee1436ddb359deea3ad39a1987
x_scaler_sha256=4e7c96d5accc2917855ed669b33d2666b3aa07f6b83b40097a9690c6a5f5e334
y_scaler_sha256=b3326a79da81b092460ef8d4a140a101b21f2433ff30c36971d626d2a2491697
train_rows_loaded=13814
validation_rows_loaded=2960
test_rows_read=0
test_target_ids_seen=0
h4_reuse_status=PASS
h2_fresh_model_status=PASS
h2_fresh_optimizer_status=PASS
one_primary_change=PASS
training_executed=false
status=PASS
```

Preflight hard stops:

- Phase 33 source checksum mismatch;
- H4 config/status/metrics/checkpoint checksum mismatch;
- H2 changes any field beyond `num_heads` and derived `head_dim`;
- feature order/count mismatch;
- scaler SHA/statistics mismatch;
- population/ordered target ID mismatch;
- any loaded Test row/ID;
- any path escapes reconstruction namespace;
- official mode lacks explicit `--authorize-training`;
- H2 config fingerprint not locked.

## 13. Focused test plan for Phase 34

Planned test file:

```text
tests/reconstruction/test_phase34_41_phase34_preflight.py
```

Required cases:

1. Phase 33 signoff/winner/reference/config/checksum gate PASS.
2. Any upstream checksum mismatch FAIL.
3. Exact FS2_TF1 feature order/count 33 PASS.
4. H2/H4 ordered Train and Validation target IDs identical.
5. Population, split, feature, window, metric and dataloader fingerprints match locked values.
6. X/Y scaler artifact SHA and statistics fingerprint match locked values.
7. Loader reads only first 16,774 chronological rows with `usecols`; Test rows read `0`.
8. Timestamp at/after Test boundary fails immediately.
9. H2 changes only `model.num_heads`; unrelated delta fails.
10. Seed is applied before model/DataLoader construction.
11. H2 uses fresh model and fresh optimizer; no checkpoint/optimizer state reuse.
12. H4 is read-only Phase 33 reference and is not registered/retrained/copied.
13. All prospective writes remain under `artifacts/reconstruction/phase34_41_v1/`.
14. Per-run/per-phase output schemas require history, metrics, predictions, checkpoint, audits, manifest, signoff and checksums.
15. Preflight allocates no scientific run ID and calls neither `TrainingEngine.train()` nor inference/evaluation.
16. Official mode without `--authorize-training` refuses before registration/training.
17. Phase 35 gate refuses until reconstructed Phase 34 signoff PASS and `approved_for_phase35=true`.
18. Arbitrary reconstruction namespace or lineage class is rejected.

Baseline unit tests may verify reusable scientific components, but they do not replace the reconstruction-specific tests above.

## 14. Recovery and no-retrain contract

Registry lifecycle:

```text
REGISTERED -> RUNNING -> COMPLETED
REGISTERED/RUNNING -> FAILED or INTERRUPTED
```

Rules:

- `COMPLETED` chỉ reusable khi config fingerprint, checkpoint SHA, metrics, predictions, history và status đều valid;
- `FAILED`, `INTERRUPTED` hoặc stale `RUNNING` không được coi là completed evidence;
- failed/interrupted records phải giữ nguyên;
- resume không tạo duplicate cho completed canonical condition;
- stale run có checkpoint chỉ được finalization-only khi output contract chứng minh training hoàn tất; nếu không, tạo run mới cho đúng missing condition sau Human authorization;
- Phase finalization chỉ chạy khi canonical matrix đầy đủ;
- finalization không train và không chạy inference lại nếu prediction artifact đã đủ;
- mọi resume tiếp tục giữ Test firewall.

## 15. Implementation preparation sequence

Sau khi Human duyệt plan, implementation riêng phải:

1. Tạo package/runner reconstruction với explicit namespace allow-list.
2. Tạo Train/Validation-only adapter không load Test rows vào memory.
3. Tạo Phase 33 lock/config snapshot cho reconstruction.
4. Tạo Phase 34 H2 config snapshot và khóa fingerprint.
5. Tích hợp fresh seed/model/optimizer guard.
6. Tích hợp isolated registry và run-ID formatter.
7. Tích hợp complete output/checksum/signoff writer.
8. Thêm focused tests ở Mục 13.
9. Chạy static syntax, focused tests và preflight.
10. Kiểm tra `git diff --check` trên các file implementation mới/sửa, tách riêng pre-existing worktree debt.
11. STOP và xin Human training approval.

Không được đồng thời sửa V1/historical execution semantics chỉ để hỗ trợ reconstruction.

## 16. Proposed Phase 34 Human training command

Chỉ dùng sau khi implementation, focused tests và preflight đều PASS, H2 fingerprint đã khóa và Human cấp quyền training riêng:

```bash
cd "/Users/vientu/Deep Learning/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK"

caffeinate -dimsu env \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src \
MPLCONFIGDIR=/private/tmp/course-work-phase34-41-reconstruction-mpl \
./.venv/bin/python -m course_work.reconstruction.phase34_41.runner \
  --phase 34 \
  --mode official \
  --seed 42 \
  --authorize-training
```

Command trên là proposed interface của implementation kế tiếp; ở trạng thái plan-only hiện tại module chưa được tạo, do đó command chưa được phép chạy.

Expected Phase 34 training:

```text
H2: 1 new fresh run
H4: 0 new run; Phase 33 reference reused read-only
Test: FORBIDDEN
```

## 17. Risks and hard stops

| Risk | Required response |
|---|---|
| Historical/reconstruction namespace collision | STOP |
| Recomputing historical checksum ledger | STOP |
| Test row/target/prediction access | STOP |
| Full-data load followed by in-memory Test filtering | STOP |
| Scaler mismatch or refit on Validation/Test | STOP |
| Population/ordered target mismatch | STOP |
| Wrong Phase 33 reference or checkpoint | STOP |
| More than one primary change | STOP |
| Seed set after model/DataLoader construction | STOP |
| Optimizer/checkpoint warm-start | STOP |
| Missing history/prediction/checkpoint in completed run | FAIL, preserve evidence |
| Phase 35 opened before Phase 34 reconstructed PASS | STOP |
| Any Phase 43+ modification | STOP |

## 18. Definition of ready for Phase 34 Human training gate

Ready chỉ khi tất cả điều kiện sau PASS:

```text
Plan approved by Human
Implementation confined to reconstruction namespace
Phase 33 locked evidence verified
H2 config fingerprint locked
Seed/population/scalers/config locked
Train/Validation-only loader verified
test_rows_read=0
test_target_ids_seen=0
H4 immutable reference verified
H2 one-primary-change verified
Fresh model/optimizer verified
Complete output contract verified
Recovery/no-retrain contract verified
Focused tests PASS
Relevant regression tests PASS
Phase 34 preflight PASS
Training executed=false
Explicit Human authorization pending
```

Current state at plan creation:

```text
READY_FOR_IMPLEMENTATION = YES
READY_FOR_PHASE34_HUMAN_TRAINING = NO
TRAINING_AUTHORIZED = NO
TEST_ACCESS = FORBIDDEN
```
