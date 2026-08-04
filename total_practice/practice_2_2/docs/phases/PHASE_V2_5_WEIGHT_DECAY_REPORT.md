# BÁO CÁO TỔNG KẾT PHASE V2.5 — CONTROLLED REGULARIZATION EXPERIMENT

Báo cáo kết quả thử nghiệm kiểm soát (Controlled Experiment) trên việc điều chỉnh Weight Decay, nhằm đánh giá khả năng giảm Overfitting đối với Baseline E2-V2.

---

## 1. Thiết lập Thử nghiệm (Experiment Setup)

- **Mục tiêu:** Giảm độ hổng tổng quát (Generalization Gap) do Overfitting gây ra trên tập Train nhỏ (~2,000 ảnh) của E2-V2.
- **Biến duy nhất thay đổi (Independent Variable):** `weight_decay` của AdamW (tăng từ `2e-4` lên `5e-4`).
- **Các biến kiểm soát (Controlled Variables):**
  - **Architecture / Freeze Policy:** Y hệt E2 (Backbone Layer4 + Classifier unfreezed, còn lại đóng băng, BatchNorm giữ ở `.eval()`).
  - **Dữ liệu:** Không đổi (V2 Split - Train: 2027, Val: 433).
  - **Initial Hash:** `4e7500733de1a2f3262489d172646dcc5c730ec8ffe0e82f00a9533a1b9368fd` (Chia sẻ cùng một xuất phát điểm chính xác từng Byte với E2).
  - **Pipeline Config:** LR (1e-4 / 1e-3), Dropout (0.2), Scheduler (ReduceLROnPlateau), Không dùng Test set.

## 2. Kết quả E3-V2

- **Best Epoch:** 11 (Early stopping sau Epoch 14)
- **Train Accuracy:** 98.72%
- **Validation Accuracy:** 77.83%
- **Validation Loss:** 0.8215
- **Validation Macro F1:** 0.7740
- **Generalization Gap:** ~20.89 pp

### 2.1 Reload Verification
Chạy tool `v2_evaluate_reloaded.py --strategy E3` trên checkpoint `best.pt`:
- Thực tế ghi nhận: Acc 77.83%, Loss 0.8215, Macro F1 0.7740.
- Khớp 100% với Metadata Metadata trong file Checkpoint. **(Passed)**

### 2.2 Checkpoint SHA-256 (E3 Best)
- `3ce3f05dec7e67232aa611efdeb98312bb0d0f245ff179a7e73e7be3a42da105`

## 3. So Sánh E2-V2 và E3-V2 (Comparison)

| Experiment | Weight Decay | Train Acc | Val Acc | Val Loss | Macro F1 | Gap |
|---|---:|---:|---:|---:|---:|---:|
| **E2-V2** | `2e-4` | 98.62% | **79.68%** | **0.7778** | **0.7913** | **18.94** |
| **E3-V2** | `5e-4` | 98.72% | 77.83% | 0.8215 | 0.7740 | 20.89 |

### 3.1 Phân Tích
- Trái với kỳ vọng, việc đẩy mạnh `weight_decay` từ `2e-4` lên `5e-4` không hề giúp mô hình "bớt học vẹt" dữ liệu huấn luyện (Train Acc vẫn đạt xấp xỉ 99%).
- Ngược lại, mức Weight Decay mạnh hơn đã làm thui chột khả năng nắm bắt miền tri thức của model (Domain Adaptation Capacity), khiến Validation Acc sụt giảm gần 2% (từ 79.68% xuống 77.83%), đồng thời đẩy Val Loss cao lên.
- Khoảng cách hụt (Generalization Gap) cũng giãn rộng ra (20.89% so với 18.94%).

## 4. Automated Tests
Tất cả các bài kiểm thử Automated Assertion (V2) đều Pass thành công 100%:
- Freeze policy của E2 và E3 giống hệt nhau.
- Initial model hash trùng nhau hoàn toàn.
- `EpochLossAggregator` chuẩn xác.
- Tập Test vẫn chưa hề bị load ở bất cứ đâu.
- File config snapshot chỉ cho phép `weight_decay` và `experiment_id` là khác nhau, mọi biến khác đều bị khoá chặn sửa đổi.

==================================================
# FINAL VERDICT
==================================================

**V2_REGULARIZATION_NO_IMPROVEMENT**

E2-V2 (Weight Decay = 2e-4) vẫn bảo toàn được vị trí **Provisional Winner** của Phase này. Thử nghiệm trên E3-V2 chứng tỏ rằng chúng ta không thể "ép" một mô hình đã pre-trained ImageNet chống Overfit trên 2000 ảnh chỉ bằng cách bẻ cong Weight Decay một cách đơn điệu. 

*Khuyến nghị cho Phase tiếp theo:* Áp dụng các chiến lược Augmented Data tinh vi hơn (Offline Augmentation), Weighted Sampler, hoặc Label Smoothing thay vì dựa vào Weight Decay/L2 Regularization cơ bản. Đã kết thúc thử nghiệm và đóng băng mọi cấu hình tại điểm này.
