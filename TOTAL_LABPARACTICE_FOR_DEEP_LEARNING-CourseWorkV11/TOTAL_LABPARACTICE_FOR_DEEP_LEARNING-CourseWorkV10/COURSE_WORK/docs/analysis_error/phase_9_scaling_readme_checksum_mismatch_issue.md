# Phase 9 Scaling README Checksum Mismatch Issue

## Phát hiện

Regression test `SharedMetricsTest.test_target_scale_conversion_supports_ys0_and_frozen_ys1` fail khi `load_validated_target_scaler` xác minh Phase 9.

Expected SHA-256 trong `artifacts/scaling/phase_9_signoff.json`:

```text
52edfe46135533583eae041a77bd5b21bbf016e71e7ce8b85b08ec9e9754970c
```

Actual SHA-256 của `artifacts/scaling/README_SCALING.md`:

```text
79ddc8f5a357e5a9ae233c0e52823d82c565a7a2e872c714c17dacbcb2c0dab7
```

README không có uncommitted diff. Lỗi tồn tại trong trạng thái repository hiện hành và không do thay đổi MAPE tạo ra.

## Phạm vi ảnh hưởng

YS1 scaler loading bị chặn bởi provenance verification. MAPE implementation không phụ thuộc scaler khi nhận prediction đã ở Wh, nhưng regression suite chung không thể đạt toàn bộ PASS.

## Corrective plan

Không sửa README hoặc Phase 9 signoff trong MAPE addendum. Ghi nhận lỗi upstream riêng, tiếp tục MAPE bằng frozen prediction CSV đã chứa `y_true_wh` và `y_pred_wh`, đồng thời giữ verification trạng thái `UPSTREAM_WARNING`.

Một Phase 9 corrective run riêng phải xác định source-of-truth của README, khôi phục đúng byte hoặc phát hành signoff version mới. Corrective run đó cần approval riêng vì có thể thay đổi frozen Phase 9 artifact.
