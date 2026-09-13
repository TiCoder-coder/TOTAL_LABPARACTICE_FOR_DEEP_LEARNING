# Final consistency audit — 2026-09-13

## Phạm vi và nguyên tắc

Audit này là read-only đối với scientific evidence. Không training, không
inference, không đọc Test source và không thay đổi checkpoint, scaler, metric,
prediction hay historical signoff. Các artifact được restore chỉ được chép vào
canonical path còn thiếu sau khi SHA-256 của source khớp checksum đã khóa.

## Trạng thái Phase 1–59

| Phase | State | Scientific core | Signoff/artifacts | Processing log / presentation | Classification |
|---:|---|---|---|---|---|
| 1 | `PASS` | Hợp lệ | Hợp lệ | Renderable | `STABLE` |
| 2 | `PASS` | Hợp lệ sau exact-SHA restore | 6/6 artifact đã restore đúng checksum | Renderable | `STABLE` |
| 3 | `PASS_WITH_WARNING` | Hợp lệ | Hợp lệ | Renderable | `STABLE_WITH_REPORTING_DEBT` |
| 4–6 | `PASS` | Hợp lệ | Hợp lệ | Renderable | `STABLE` |
| 7 | `PASS` | Hợp lệ sau exact-SHA restore | Feature-engineered CSV đã restore đúng checksum | Renderable | `STABLE` |
| 8–10 | `PASS` | Hợp lệ | Hợp lệ | Renderable | `STABLE` |
| 11 | `PASS` | Hợp lệ sau exact-SHA restore | 3/3 registry/audit CSV đã restore đúng checksum | Renderable | `STABLE` |
| 12–13 | `PASS` | Hợp lệ | Hợp lệ | Renderable | `STABLE` |
| 14 | `PASS` | Hợp lệ sau exact-SHA restore | Validation prediction CSV đã restore đúng checksum | Renderable | `STABLE` |
| 15–22 | `PASS` | Hợp lệ | Hợp lệ | Renderable | `STABLE` |
| 23–30 | `VALID_REUSABLE` | Hợp lệ | Required artifact count đầy đủ | Live action `RENDER_ONLY`; saved log còn nhãn historical `BLOCKED` | `STABLE_WITH_REPORTING_DEBT` |
| 31–33 | `VALID_REUSABLE` | Hợp lệ | Required artifact count đầy đủ | `RENDER_ONLY` | `STABLE` |
| 34–36 | `VALID_REUSABLE` | Canonical run/output và recovered signoff đúng checksum | Hợp lệ | `RENDER_ONLY`; log hợp lệ | `STABLE` |
| 37 | `VALID_REUSABLE` | Canonical `RUN_TR_S15_0024_9420CDD7` verified | Exact legacy signoff verified | `RENDER_ONLY` | `STABLE_WITH_REPORTING_DEBT` |
| 38 | `VALID_REUSABLE` | Canonical `RUN_TR_S16_0025_49060872` verified | Exact legacy signoff verified | `RENDER_ONLY` | `STABLE_WITH_REPORTING_DEBT` |
| 39 | `VALID_REUSABLE` | Canonical `RUN_TR_S17_0029_082F7FF5` verified | Exact legacy signoff verified | `RENDER_ONLY` | `STABLE_WITH_REPORTING_DEBT` |
| 40 | `VALID_REUSABLE` | Canonical `RUN_TR_S18_0031_A711A9B8` verified | Exact legacy signoff verified | `RENDER_ONLY` | `STABLE_WITH_REPORTING_DEBT` |
| 41 | `VALID_REUSABLE` | Canonical `RUN_TR_S19_0034_CF8C1FE8` verified | Exact legacy signoff verified | `RENDER_ONLY` | `STABLE_WITH_REPORTING_DEBT` |
| 42 | `PASS` | Hợp lệ | Hợp lệ | Renderable | `STABLE` |
| 43–44 | Verified result renderable | Signoff hiện tại đủ cho notebook | Core signoff hợp lệ | Generic processing-log rebuild thiếu reporting CSV | `STABLE_WITH_REPORTING_DEBT` |
| 45–50 | Verified result renderable | Hợp lệ | Current machine-readable source tồn tại | 3-table read-only renderer | `STABLE` |
| 51 | Verified result renderable | Hợp lệ | Summary/signoff/figures hiện có | 2 historical reporting CSV còn thiếu | `STABLE_WITH_REPORTING_DEBT` |
| 52–53 | Verified result renderable | Hợp lệ | Current machine-readable source tồn tại | 3-table read-only renderer | `STABLE` |
| 54–57 | Verified result renderable | Hợp lệ | Summary/signoff hiện có; historical dashboard snapshot checksum-verified | Một số intermediate CSV vẫn là debt | `STABLE_WITH_REPORTING_DEBT` |
| 58–59 | Verified result renderable | Hợp lệ | Current machine-readable source tồn tại | 3-table read-only renderer | `STABLE` |

## Exact-checksum artifact restoration

Manifest đầy đủ: [final_consistency_exact_artifact_restore_20260913T061808Z.json](../../artifacts/_reconstruction_history/final_consistency_exact_artifact_restore_20260913T061808Z.json).

| Phase | Artifact group | Count | Checksum result |
|---:|---|---:|---|
| 2 | Raw-data source metadata, manifest, checksum ledger, ZIP và variable metadata | 6 | `EXACT_SHA_MATCH` |
| 7 | `energydata_feature_engineered_v1.csv` | 1 | `EXACT_SHA_MATCH` |
| 11 | Dataset/DataLoader registries và shuffle audit | 3 | `EXACT_SHA_MATCH` |
| 14 | `persistence_validation_predictions.csv` | 1 | `EXACT_SHA_MATCH` |

Mọi destination đều absent trước copy. Không canonical file nào bị overwrite.

## Phase 36–41 notebook refresh audit

Saved HTML output của từng Phase 36–41 bằng byte với output tạo in-memory bởi
`render_frozen_phase_evidence()` ở current tree. Sáu output đều chứa
`VALID_REUSABLE` và `RENDER_ONLY`, không chứa `BLOCKED`. Vì nội dung đã current,
notebook không bị sửa chỉ để thay `execution_count`.

## Reporting debt còn giữ nguyên

- Phase 23–30: saved processing logs mang historical label `BLOCKED`, trong khi
  current fail-closed inspection là `VALID_REUSABLE/RENDER_ONLY`. Notebook dùng
  live read-only audit nên không hiển thị sai.
- Phase 37: thiếu `s15_loss_sweep_manifest.json`, `s15_loss_metrics.csv` và
  canonical `training_history.csv`.
- Phase 38: thiếu `s16_run_matrix.csv`, `s16_epoch_cap_metrics.csv`,
  `s16_epoch_cap_effect.csv` và canonical `training_history.csv`.
- Phase 39: thiếu các CSV audit/reporting S17 và canonical
  `training_history.csv` được khai báo trong exact legacy contract.
- Phase 40: thiếu `s18_revin_metrics.csv` và canonical `training_history.csv`.
- Phase 41: thiếu `s19_boundary_metrics.csv`, `s19_boundary_winner.json` và
  canonical `training_history.csv`.
- Phase 43: generic processing-log rebuild thiếu `lstm_stage_lineage.csv`.
- Phase 44: generic processing-log rebuild thiếu `rolling_origin_results.csv`.
- Phase 51: `phase51_attention_handoff_cases.csv` và
  `phase51_tests_summary.csv` có expected SHA trong manifest nhưng chưa tìm thấy
  source byte-verifiable; chúng không được fabricate.

Các file trên là `NON_BLOCKING` chỉ tại nơi exact legacy/current contract đã
chứng minh scientific core độc lập. Nếu core artifact hoặc SHA lệch, validator
vẫn fail-closed.

## Validation cuối

| Validation | Result |
|---|---|
| Focused reporting/recovery/Step16/Step17/selective-execution suite | `163 passed, 1 skipped` |
| Phase 1–22 processing-log build | `PASS`/`PASS_WITH_WARNING`, không exception |
| Phase 23–41 inspection | `19/19 VALID_REUSABLE`, action `RENDER_ONLY` |
| Phase 43–59 verified renderer | `17/17`, mỗi phase 3 bảng |
| Notebook | 153 cells; 77/77 code cells có `execution_count`; 0 saved Python error |
| Relative Markdown links | 389 checked; 0 broken relative links |
| Scoped `git diff --check` cho file sửa trong audit này | `PASS` |
| Full `git diff --check` | `FAIL`: trailing whitespace/CRLF trong exact recovered historical JSON; không normalize vì sẽ đổi SHA |

## Kết luận

- `PROJECT_SCIENTIFIC_STATUS = STABLE_WITH_REPORTING_DEBT`
- `PHASE_1_59_STATUS = PASS_WITH_NON_BLOCKING_REPORTING_DEBT`
- `PHASE_34_41_STATUS = VALID_REUSABLE`
- `TRAINING_REQUIRED = NO`
- `TRAINING_EXECUTED = NO`
- `INFERENCE_EXECUTED = NO`
- `TEST_ACCESSED_DURING_AUDIT = NO`
- `SCIENTIFIC_EVIDENCE_OVERWRITTEN = NO`
