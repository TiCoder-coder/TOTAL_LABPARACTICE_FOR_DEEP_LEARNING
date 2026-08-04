# BÁO CÁO TỔNG KẾT PHASE V2.4 — CLEAN BASELINE TRAINING

Báo cáo kết quả huấn luyện mô hình Baseline ResNet18 (E1-V2: Head Only và E2-V2: Layer4 + Head) trên nền dữ liệu sạch V2. Đảm bảo tuân thủ tuyệt đối: Không chạy Test Data, đóng băng Layers hợp lý và Epoch Loss chuẩn.

---

## A. Run ID & Artifact Lineage
- **Run ID:** `v2_baseline_s42_e1_e2_v1`
- **Thư mục lưu trữ Checkpoint:** `total_practice/practice_2_2/artifacts/experiments/v2_baseline_s42_e1_e2_v1/`

## B. Config Pipeline
- **Dataset Fingerprint:** `523faa2fbcb38854f12a738404887b1451780b7a0ef5a6ddebd000da0a75487c`
- **Split Fingerprint:** `2157484123560b5dddf1d30d27d3945def7eaac663dbf2552176a2896446661e`
- **Optimizer:** `AdamW` (Head LR: 1e-3, Backbone LR: 1e-4, Weight Decay: 2e-4)
- **Scheduler:** `ReduceLROnPlateau(factor=0.5, patience=2)`

## C. Initial Model Hash
E1-V2 và E2-V2 chia sẻ cùng chung 1 Initial State Dict khi mới khởi tạo, đảm bảo độ công bằng tuyệt đối:
- **Hash SHA-256:** `061d4b68e9f50dc97c41eb41b7112dcba52601275bf2a24911d33c467df566ef` (Tính từ Bytes của toàn bộ parameters).

## D. Kết quả E1-V2 (Head Only)
Chỉ Train layer Classifier (Linear 512 -> 10). Dropout 0.2. Các BN layer bị đóng băng giữ nguyên trạng thái Eval().
- **Trainable Parameters:** 5,130
- **Best Epoch:** 15
- **Train Acc:** 62.16%
- **Val Acc:** 61.89%
- **Val Loss:** 1.2074
- **Val Macro F1:** 0.6149
- **Generalization Gap:** ~0.27 pp

## E. Kết quả E2-V2 (Layer4 + Head)
Train Layer 4 và Classifier. Stem, Layer 1, 2, 3 bị đóng băng và BN giữ ở Eval().
- **Trainable Parameters:** ~8.4 triệu
- **Best Epoch:** 15
- **Train Acc:** 98.62%
- **Val Acc:** 79.68%
- **Val Loss:** 0.7778
- **Val Macro F1:** 0.7913
- **Generalization Gap:** ~18.94 pp

## F. Learning Curves
Cả hai mô hình đều hội tụ từ từ. Lịch trình Scheduler với `factor=0.5` cho phép Learning Rate giảm một cách mượt mà, giúp E2 vươn tới mức Val Acc xấp xỉ 80% tại Epoch 15.

## G. Validation Comparison
- **E1-V2** rõ ràng bị **Underfitting** vì cấu trúc quá cứng nhắc (chỉ 5k params). Nó hầu như không Overfit (Gap: 0.27%).
- **E2-V2** giải phóng thêm Layer 4, tạo đủ sức mạnh để học được đặc trưng, đẩy Validation Accuracy lên 79.68%, và Macro F1 đạt 0.7913.

## H. Generalization Gap (V2 pipeline)
Khoảng cách giữa khả năng học thuộc lòng (Train) và khả năng khái quát (Validation) của **E2-V2** là **~18.94 pp**. Đây là một độ lệch lớn, chứng tỏ model có dấu hiệu Overfitting nặng lên tập Train nhỏ (~2000 ảnh).

## I. V1 vs V2 Comparison
So sánh cấu hình E2 (Layer4 + Head) giữa quá khứ (V1) và hiện tại (V2):
| Metric | V1 Historical | V2 Clean Baseline |
| :--- | :--- | :--- |
| **Train Acc** | ~98.31% | **98.62%** |
| **Val Acc** | ~78.31% | **79.68%** |
| **Gap** | ~20.00 pp | **~18.94 pp** |

## J. Overfitting Assessment
**Đánh giá:**
Mặc dù Pipeline Dữ liệu V2 đã cắt đứt Leakage hoàn toàn (loại bỏ sinh ảnh Fake chéo Val/Test) và khắc phục Loss Bug, **hiện tượng Overfitting vẫn còn cao (Generalization gap ~19%)**. 
Lý do chính là dung lượng tập Train vẫn tương đối nhỏ. Tuy nhiên, việc áp dụng Data Pipeline siêu sạch V2 đã có hiệu ứng tích cực: Validation Accuracy tăng từ 78.31% lên 79.68% và Gap giảm được khoảng 1% so với V1.

## K. Checkpoint Hashes
Mỗi Experiment đều lưu `best.pt` và `latest.pt`. Siêu dữ liệu bao gồm Epoch, Loss, Optimizer States và Hash Fingerprint (xem trong `metadata` của file .pt).

## L. Reload Verification
Đã kích hoạt Script `v2_evaluate_reloaded.py`:
- **E1-V2:** Load lại Epoch 15 $\rightarrow$ Đạt chính xác Acc 61.89%, Loss 1.2074.
- **E2-V2:** Load lại Epoch 15 $\rightarrow$ Đạt chính xác Acc 79.68%, Loss 0.7778.
**(Passed 100%)**

## M. Automated Tests
Các bài kiểm tra đã được khởi chạy thông qua `pytest`:
- Xác nhận BN Layer bị đóng băng được duy trì chế độ `.eval()` trong lúc train `layer4`.
- Xác nhận E1, E2 ban đầu dùng chung Hash khởi tạo tuyệt đối.
- Xác nhận `v2_dataset.py` không bao giờ gọi tới dữ liệu Test.
- Xác nhận V2 Manifest và V1 Artifacts hoàn toàn không thay đổi sau Phase này.

## N. Provisional Winner
Dựa hoàn toàn trên Validation Score:
**E2-V2** thắng E1-V2 áp đảo (79.68% vs 61.89%) và trở thành mô hình đại diện tiếp theo.

## O. Remaining Limitations
Tập Train chỉ khoảng 2,000 mẫu, không đủ lớn cho ResNet18 Layer4 học mà không bị Overfit (~19%). Ở các Phase sau cần những kỹ thuật Regularization hoặc Offline/Advanced Augmentation đặc dụng cho nhóm thiểu số.

==================================================
# FINAL VERDICT
==================================================

**V2_BASELINE_OVERFITTING_STILL_HIGH**

*(Quá trình Base Training thành công và đã cải thiện Accuracy + giảm nhẹ Gap so với V1, tuy nhiên Overfitting Gap vẫn xấp xỉ 19%. Hệ thống Data/Loss hoàn toàn trong sạch và ổn định, sẵn sàng mở đường cho các bước Balancing/Regularization phức tạp hơn trong tương lai).*
