# BÁO CÁO TỔNG KẾT PHASE V2.3 — DATA LOADER, TRANSFORMS, LOSS & TRAINING CONTRACT

Báo cáo này tổng hợp kiến trúc Data Pipeline (Loader, Transform, Loss Config) chuẩn bị cho việc huấn luyện mô hình (Training Contract) của Practice 2.2 V2. Các module được đảm bảo không rò rỉ dữ liệu, xử lý chính xác và sửa các lỗi tồn đọng từ V1.

---

## A. Dataset Builders
Đã xây dựng module `v2_dataset.py` quản lý Class `CosmeticsDatasetV2` với các factory functions:
- `build_train_dataset()`
- `build_validation_dataset()`
- `build_test_dataset()`
Mỗi hàm đều trỏ đúng tới `split` tương ứng trong file manifest của V2.

## B. Sample Counts & Exact Match
- **Train:** 2027 ảnh (original, không generated, không quarantine).
- **Validation:** 433 ảnh.
- **Test:** 434 ảnh.
Tất cả đã vượt qua các Assertions so khớp chính xác từng path từ manifest. Không có bất kỳ hiện tượng load thừa hay thiếu sample.

## C. Train Transforms
Sử dụng hàm `get_train_transforms()` áp dụng baseline Augmentation mức độ vừa:
- `RandomResizedCrop(224, scale=(0.8, 1.0))`
- `RandomHorizontalFlip(p=0.5)`
- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)`
- `ToTensor()`
- `RandomErasing(p=0.1, scale=(0.02, 0.1))`
- `Normalize(ImageNet)`

## D. Validation Transforms
Sử dụng hàm `get_val_test_transforms()` **hoàn toàn deterministic**:
- `Resize(256)`
- `CenterCrop(224)`
- `ToTensor()`
- `Normalize(ImageNet)`

## E. Test Transforms
Hoàn toàn giống Validation Transforms. (Sử dụng chung hàm `get_val_test_transforms()`).

## F. Normalization
Duy trì Normalize bằng `ImageNet` standard để tối ưu hóa Pre-trained weights của ResNet (hoặc các mạng khác):
- **Mean:** `[0.485, 0.456, 0.406]`
- **Std:** `[0.229, 0.224, 0.225]`

## G. Loss Policy
Do độ lệch nhãn (Imbalance Severity) ở mức SLIGHT, chúng ta sử dụng hàm mất mát chuẩn:
- **Criterion:** `nn.CrossEntropyLoss()`
- **Label Smoothing:** Không áp dụng trong Baseline đầu tiên.
- **Class Weights:** Tạm thời không sử dụng.

## H. Sampler Policy
- **Train DataLoader:** `shuffle=True` (Không dùng `WeightedRandomSampler`).
- **Validation DataLoader:** `shuffle=False`.
- **Test DataLoader:** `shuffle=False`.

## I. Epoch-Loss Bug Fix
Lỗi chia Epoch Loss sai của V1 (nhân với `batch_size` thay vì trọng số của batch) đã được khắc phục hoàn toàn thông qua helper `EpochLossAggregator` và `get_loss_normalizer()` trong `v2_loss_utils.py`.
- Hàm này tự động quét `criterion.weight` để biết cách cộng dồn mẫu số (normalizer) một cách chính xác nhất bất kể có dùng Class Weights hay không.

## J. Mathematical Verification
Các hàm test toán học (`test_epoch_loss_unweighted_math` và `test_epoch_loss_weighted_math`) đã chạy thành công.
- Đối chiếu việc chia batch (qua Loss Helper) bằng tổng loss tuyệt đối chia normalizer với việc chạy hàm CE Loss thẳng trên ma trận 100 sample.
- Kết quả trùng khớp hoàn hảo (sai số `1e-6`), chứng minh toán học của Loss Aggregator là hoàn toàn chính xác.

## K. Test-Access Contract
Các lớp Dataset cho V2 có lưu flag `is_test`. Test suite `test_training_pipeline_does_not_load_test` đã xác minh rằng hai builder Train và Validation tuyệt đối không vô tình mang cờ này, thiết lập rào cản ngăn chặn model tự động gọi tới Test data trong giai đoạn train.

## L. Config Snapshot
Đã sinh file `v2_baseline_config.json` khóa cứng toàn bộ thiết lập từ `split_fingerprint`, `dataset_fingerprint`, `manifest_sha256`, seed, transforms pipeline đến kiến trúc cấu hình chuẩn. File này nằm tại:
`total_practice/practice_2_2/data/manifests/v2_stratified_group_s42/v2_baseline_config.json`

## M. Automated Tests
Đã chạy toàn bộ **16 Unit Tests**, đạt tỉ lệ PASS 100%. Các test bao gồm:
- Dữ liệu rác/generated không được load.
- Không trùng chéo Data (Disjoint paths).
- Variance của Train Transform được xác thực chặt chẽ bằng ma trận pixel thay vì random seed đơn giản.
- Validation Transform không thay đổi pixel giữa các lần gọi.
- Loss Aggregation khớp về mặt toán học.
- V1 Manifest và V2 Manifest SHA256 không bị sửa đổi.

## N. Remaining Limitations
- Hiện tại, Batch Size được fix cứng bằng 32, tuy nhiên, cấu hình phần cứng MacOS thực tế sẽ định đoạt `num_workers`. 
- Cần thận trọng khi bắt đầu huấn luyện Phase tới về Memory Limit của thiết bị.

==================================================
# FINAL VERDICT
==================================================

**V2_DATA_PIPELINE_READY**

*(Tất cả Dataset Builders, Transform configs và Loss aggregation bug-fix đều đã kiểm thử. System sẵn sàng để tích hợp vào Epoch Loop để Train Baseline).*
