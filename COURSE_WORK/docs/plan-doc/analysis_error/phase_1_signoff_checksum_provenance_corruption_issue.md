# Phase 1 Sign-off Checksum Provenance Corruption Issue

## 1. Trạng thái

```text
Issue ID: PHASE-1-SIGNOFF-CHECKSUM-PROVENANCE-CORRUPTION-v1
Status: CONFIRMED
Severity: BLOCKING_STAGE_0
Detected by: Remaining Pipeline Compliance Stage 0
Implementation status: NOT_STARTED
```

## 2. Hiện tượng

`tests/unit/test_eda.py` có bốn test pass và một test fail. Failure xuất hiện trước logic EDA tại strict Phase 1 validation:

```text
RuntimeError: Phase 1 signed output is invalid: artifacts/environment/environment_report.json
```

Canonical `phase_1_signoff.json` khai báo:

```text
environment_report.json expected: ee9f78b9ca72e4293aa1d739f189e494ddce35e2581d8e7016ec213fca3b042a
requirements_freeze.txt expected: ae3256326e1f492450365a3df073ff253996d4b8b515bc073287d07f16fe062c
smoke_test_report.json expected: e00f1049f49a87f9370536e6a3e78e6202ef5985f01675a4d181c7f8decbeb03
```

Checksum thực tế:

```text
environment_report.json actual: ebc3af1a9555f1f5543bca94db38037ec9c852e1691b74acd1849cf07c7fedf5
requirements_freeze.txt actual: 395381b424655ace22a41f89880eb9384df9d6665f91fbca51d0d147802db2c5
smoke_test_report.json actual: 90ddf8824fc4c093abdcafd9a9c328d956c42af9327d896b15740259341d63e6
```

## 3. Phân tích provenance

Canonical output files hiện tại khớp byte với revision Git `1f648662980c49e0ee42de9f2cf43639e6966b64`.

Sign-off tại revision đó khai báo đúng ba checksum thực tế:

```text
environment_report.json: ebc3af1a9555f1f5543bca94db38037ec9c852e1691b74acd1849cf07c7fedf5
requirements_freeze.txt: 395381b424655ace22a41f89880eb9384df9d6665f91fbca51d0d147802db2c5
smoke_test_report.json: 90ddf8824fc4c093abdcafd9a9c328d956c42af9327d896b15740259341d63e6
```

Commit `ff6f963ac1009456e34664523a845792ea8e2fee` thay đúng ba checksum trong:

- Canonical `artifacts/environment/phase_1_signoff.json`.
- Hai bản `phase_1_signoff.json` dưới `_history`.

Commit này không thay ba canonical output tương ứng. Không có revision của ba output trong Git history hoặc `_history` khớp ba checksum được chèn bởi commit đó.

## 4. Root cause

Root cause là sign-off provenance bị thay đổi tách rời khỏi output artifacts. Đây không phải lỗi công thức EDA, không phải lỗi `load_validated_environment_signoff`, không phải runtime drift hiện tại và không phải lỗi scientific payload của ba Phase 1 outputs.

Strict loader từ chối chuỗi upstream là hành vi đúng.

## 5. Phạm vi ảnh hưởng

- Direct EDA preparation bị chặn khi gọi chuỗi Phase 1 đến Phase 4.
- Các consumer gọi `materialize_phase_1` hoặc `load_validated_environment_signoff` có thể bị chặn.
- Shared metrics mismatch Phase 9 là lỗi độc lập.
- MAPE unit, integration và reporting tests không bị ảnh hưởng.
- Không có bằng chứng cho thấy model, split, scaler, checkpoint hoặc Test artifacts bị thay đổi bởi lỗi Phase 1 này.

## 6. Hướng sửa hợp lệ

Khôi phục exact canonical Phase 1 sign-off bytes từ revision ngay trước corruption, sau đó xác minh:

- Canonical sign-off khớp exact source blob được chọn.
- Ba declared output checksum khớp ba current output files.
- Không regenerate environment report.
- Không chạy environment recovery.
- Không sửa runtime path hoặc hardware fields.
- Không sửa historical sign-offs trong corrective tối thiểu này.

## 7. Hướng sửa bị cấm

- Không thay ba canonical output để cố khớp checksum không có source.
- Không tắt strict checksum validation.
- Không gọi `recover_environment_revision`.
- Không regenerate dependency freeze hoặc smoke test.
- Không sửa EDA để bỏ qua Phase 1.
- Không sửa các downstream sign-off.
- Không chạy lại toàn notebook.

## 8. Stop gate

Issue này nằm ngoài giả định ban đầu của approved master plan. Không được chuyển sang Phase 9 provenance audit hoặc EDA refactor cho đến khi corrective plan riêng được Human duyệt và Phase 1 strict validation trở lại PASS.
