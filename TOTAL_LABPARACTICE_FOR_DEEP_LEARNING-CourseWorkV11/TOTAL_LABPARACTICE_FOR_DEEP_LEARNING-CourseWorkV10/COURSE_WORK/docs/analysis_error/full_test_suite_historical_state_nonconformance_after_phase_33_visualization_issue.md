# Phân tích lỗi full test suite sau khi tích hợp visualization Phase 33

## 1. Phạm vi phát hiện

Lỗi được phát hiện ở final regression của task trực quan hóa cấu hình Transformer sau Phase 33.

Kết quả full suite:

```text
260 passed
6 failed
6 errors
4 subtests passed
```

Targeted scope của visualization vẫn pass:

```text
Reporting unit tests: 31 passed
Notebook contract và boundary tests: 24 passed
```

Không failure/error nào có traceback đi qua builder hoặc renderer mới.

## 2. Nhóm lỗi đã xác nhận

### 2.1. Test registry sử dụng expectation của Phase 13 ban đầu

Hai assertion vẫn yêu cầu:

```text
registry_manifest.run_count == 1
len(experiment_registry.csv) == 1
```

Current canonical registry sau Phase 33 có:

```text
run_count == 19
```

Đây là expected state sau các baseline run và sweep Phase 20-33, không phải lỗi do visualization.

### 2.2. Phase 2 sign-off checksum không còn khớp current files

Mismatch được xác nhận:

```text
data/raw_data/README_SOURCE.md
data/raw_data/dataset_manifest.json
```

`phase_2_signoff.json` vẫn giữ checksum của revision cũ trong khi file hiện tại đã thay đổi.

### 2.3. Processing log Phase 15-22 giữ source checksum cũ

Các processing log Phase 15-22 đang tham chiếu checksum cũ của một hoặc nhiều source artifacts hiện tại.

Affected phases:

```text
15
16
17
18
19
20
21
22
```

Không được sửa tay checksum trong log. Producer và canonical artifact state phải được audit trước khi quyết định regenerate log.

### 2.4. GPU enforcement chặn test materialization trên test process hiện tại

Các test EDA và splitting gọi lại chuỗi materialization từ Phase 1. `materialize_phase_1` yêu cầu CUDA hoặc MPS, nhưng process chạy test không phát hiện GPU nên raise RuntimeError.

Affected test groups:

```text
tests/integration/test_phase_0_to_5_chain.py
tests/unit/test_eda.py
tests/unit/test_splitting.py
```

Không được bỏ GPU gate hoặc làm yếu test khi chưa xác định đúng execution contract cho test-only materialization.

## 3. Bằng chứng không liên quan tới visualization Phase 33

Năm source canonical của visualization giữ nguyên checksum trước và sau implementation:

```text
phase_33_signoff.json: f8caf290991f8a26fe1f899a0490a8727d2d8096172dc612b55571741fa7e76e
s11_d_model_winner.json: 0882ffd94a9867b11979559addc4ef20e4d485cc97eae724b88ea1c66608b780
s11_reference_update.json: f8bd5da635a68529641ffde503047b277907358639b08c3156eff04a7f1e6dd4
current reference config.json: d8bcd4ae977a15a6658c5ff969f8ca1eb46c84b03cbaeb02f14a03d000921cd4
feature_set_registry.json: 03885a997d33c9b9270dcc82e396c7bc02c860bcffc50a41f1514e0c59aac067
```

Current reference builder và static renderer pass toàn bộ targeted tests.

## 4. Root causes

```text
Integration expectations chưa được revision theo lifecycle Phase 13-33.
Một số sign-off và processing log không được regenerate sau khi source artifact được revision.
Environment materialization policy đang trộn runtime training GPU requirement với test execution requirement.
```

## 5. Không được thực hiện trong issue này

```text
Không sửa checksum bằng tay.
Không giảm assertion run_count về kiểm tra lỏng.
Không tắt GPU enforcement tùy tiện.
Không regenerate toàn bộ Phase tự động.
Không xóa hoặc phục hồi artifact bằng git command phá hủy.
Không thay current reference Phase 33.
```

## 6. Mức ảnh hưởng

```text
Visualization Phase 33: không bị ảnh hưởng
Canonical Phase 33 source: không bị ảnh hưởng
Full repository regression status: chưa sạch
Historical artifact integrity: cần audit riêng
Test portability: cần thiết kế lại có kiểm soát
```

