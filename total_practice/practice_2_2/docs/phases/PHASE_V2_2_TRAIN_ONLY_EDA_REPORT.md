# BÁO CÁO TỔNG KẾT PHASE V2.2 — TRAIN-ONLY EDA & BALANCING AUDIT

Báo cáo này trình bày kết quả khám phá dữ liệu (EDA) và kiểm toán cân bằng (Balancing Audit) cho tập Train của Practice 2.2 phiên bản V2 (Lineage `v2_stratified_group_s42`). Mọi phân tích được đảm bảo **chỉ sử dụng tập Train**, các tập Validation và Test hoàn toàn bị cách ly.

---

## A. Train Distribution
Tổng số lượng mẫu trong tập Train (chỉ tính original và use_for_model=True): **2,027 ảnh**.
- **Min count:** 159 (`moisturizer`)
- **Max count:** 225 (`sunscreen`)
- **Mean count:** ~202.7
- **Standard Deviation:** ~18.25
- **Coefficient of Variation (CV):** 0.09
- **Max/Min ratio:** 1.41
- **Đánh giá:** **SLIGHT IMBALANCE**. Mức độ mất cân bằng khá nhẹ, tỷ lệ class lớn nhất gấp ~1.4 lần class nhỏ nhất. 

| Class | Train Samples | Percentage |
| :--- | :--- | :--- |
| **sunscreen** | 225 | 11.10% |
| **facial_cleanser** | 224 | 11.05% |
| **body_wash** | 215 | 10.60% |
| **face_mask** | 210 | 10.36% |
| **serum** | 208 | 10.26% |
| **perfume** | 202 | 9.96% |
| **lipstick** | 200 | 9.86% |
| **toner** | 193 | 9.52% |
| **shampoo** | 191 | 9.42% |
| **moisturizer** | 159 | 7.84% |

## B. Effective Diversity (Train Only)
Phân tích tính đa dạng thực sự (Effective Diversity) dựa trên các `effective_group_id`.

| Class | Images | Effective Groups | Images/Group | Largest Group | Diversity Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **sunscreen** | 225 | 212 | 1.06 | 3 | 94.22% |
| **facial_cleanser** | 224 | 196 | 1.14 | 3 | 87.50% |
| **body_wash** | 215 | 182 | 1.18 | 2 | 84.65% |
| **face_mask** | 210 | 191 | 1.09 | 4 | 90.95% |
| **serum** | 208 | 194 | 1.07 | 2 | 93.26% |
| **perfume** | 202 | 182 | 1.10 | 4 | 90.09% |
| **lipstick** | 200 | 180 | 1.11 | 2 | 90.00% |
| **toner** | 193 | 176 | 1.09 | 2 | 91.19% |
| **shampoo** | 191 | 164 | 1.16 | 3 | 85.86% |
| **moisturizer** | 159 | 140 | 1.13 | 3 | 88.05% |

> *Đánh giá: Diversity Ratio dao động từ 84.6% - 94.2%. Các lớp như `body_wash` và `shampoo` có độ trùng lặp nội bộ (images/group) nhỉnh hơn một chút so với các lớp khác nhưng vẫn ở mức khá đa dạng.*

## C. Label / Content Quality (Heuristic Audit)
Áp dụng các Heuristics (độ sáng bất thường, kích thước bất thường, độ tương phản cực thấp) trên tập Train để phát hiện các file lỗi.
- Đã xuất **729** candidates có nguy cơ chất lượng kém ra file `TRAIN_REVIEW_REQUIRED.csv`.
- Đa phần các file bị flag vì độ sáng khá cao (sản phẩm chụp trên nền trắng lóa sáng).
- **Lưu ý:** Không tự động sửa/xóa bất kỳ file nào. Các label và file vẫn được giữ nguyên cho việc Training.

## D. Image Statistics (Train Only)
- **Mean Width:** 224.0 pixels
- **Mean Height:** 224.0 pixels
- **Mean Brightness:** 0.76 (khá sáng, thiên trắng)
- **Mean Contrast (Std):** 0.20 (độ tương phản trung bình)

## E. Train RGB Statistics
Tính toán pixel statistics trực tiếp trên toàn bộ ảnh của tập Train:
- **Train Mean RGB:** `[0.780, 0.761, 0.750]` (Rất cao, phản ánh nền trắng phổ biến của bộ dữ liệu mỹ phẩm)
- **Train Std RGB:** `[0.274, 0.267, 0.287]`

## F. Normalization Decision
So sánh với ImageNet:
- ImageNet Mean: `[0.485, 0.456, 0.406]`
- ImageNet Std: `[0.229, 0.224, 0.225]`
- **Quyết định:** Mặc dù Train dataset sáng hơn rõ rệt so với ImageNet, hiện tại khuyến nghị **KEEP_IMAGENET_NORMALIZATION**. Mô hình Pretrained ResNet18 (hoặc các mạng khác trên ImageNet) vẫn phản ứng tốt nhất với Input được normalize bằng ImageNet stats ban đầu của nó, để không phá vỡ weight mapping của các layer đầu. Không đổi Normalization mặc định ở giai đoạn này.

## G. Balancing Assessment
- Imbalance chỉ ở mức **SLIGHT** (1.41 max/min ratio).
- **Kết luận:** Không bắt buộc phải sử dụng Weighted Loss hay Weighted Random Sampler phức tạp. Việc áp dụng Data Augmentation tiêu chuẩn là đủ để giảm thiểu hiện tượng Overfitting nhẹ lên các Major Classes.

## H. Candidate Class Weights
Trọng số ứng viên tính toán (Dùng trong trường hợp vẫn muốn áp dụng Weighted Loss sau này, nhưng không tự động áp dụng):
*Dựa trên công thức Normalized Inverse Frequency từ Train*
- `sunscreen`: 0.892
- `facial_cleanser`: 0.896
- `body_wash`: 0.934
- `face_mask`: 0.956
- `serum`: 0.965
- `perfume`: 0.994
- `lipstick`: 1.004
- `toner`: 1.040
- `shampoo`: 1.051
- `moisturizer`: 1.263 (Lớp thiểu số, trọng số cao nhất)

## I. Augmentation Recommendations
Dựa vào loại sản phẩm mỹ phẩm và đặc trưng nền trắng:
- **RandomResizedCrop:** OPTIONAL (Nhẹ, tránh cắt mất chi tiết sản phẩm).
- **HorizontalFlip:** KEEP (Sản phẩm thường đối xứng, giúp tăng đa dạng).
- **ColorJitter:** KEEP (Giúp model chống chịu độ sáng chênh lệch).
- **RandomRotation:** KEEP (Nhẹ, ~15-30 độ).
- **RandomErasing:** OPTIONAL (Rất nhẹ).

## J. Train-only Leakage Contract
Quy ước Data Preprocessing cho V2:
1. **TRAIN:** Dùng để tính toán mọi statistics, augmentation đa dạng, sinh class weights nếu cần.
2. **VALIDATION:** Deterministic transforms, không dùng thống kê riêng. Dùng cho Early Stopping.
3. **TEST:** Deterministic transforms, dùng để đánh giá cuối cùng.
> *(Tất cả script EDA đã tuân thủ 100% việc không đọc bất kỳ thông tin nào của Val/Test).*

## K. Overfitting Root-Cause Ranking
Đánh giá lại tại sao V1 E2 bị Overfitting mạnh (Train acc ~98%, Val acc ~78%):
1. **Dataset Size & Effective Diversity:** Tổng mẫu 2,027 ảnh Train là khá ít cho một mạng lớn.
2. **Leakage từ Augmentation V1:** V1 trộn 212 ảnh Generated vào Train và rò rỉ cả sang Val/Test, khiến model dễ "học thuộc" artifact thay vì feature.
3. **Mất cân bằng tự nhiên:** (Minority class ít mẫu hơn).
4. **Weighted Loss Bug (V1):** Trong V1, Loss aggregated bị sai do nhân nhầm với `batch_size` (Xem phần N), khiến việc quan sát Epoch Loss không phản ánh đúng gradient step.

## L. Recommended V2 Training-Data Policy
- **Loss:** `CrossEntropyLoss()` tiêu chuẩn (Có thể áp dụng `class_weight` nhẹ nhưng KHÔNG khuyến nghị).
- **Sampler:** None (Dùng default shuffle).
- **Augmentation:** Compose(RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomRotation).
- **Normalization:** ImageNet.

## M. Automated Tests
Đã chạy toàn bộ test suite (`test_v2_eda.py`):
- `test_no_validation_paths_loaded`: PASS
- `test_no_test_paths_loaded`: PASS
- `test_statistics_derived_from_train_only`: PASS
- `test_rgb_stats_derived_from_train_only`: PASS
- `test_review_candidates_are_not_deleted`: PASS
- `test_no_images_are_modified`: PASS
- `test_eda_is_deterministic`: PASS
- `test_manifest_hash_unchanged_after_eda`: PASS
- `test_v1_artifacts_unchanged`: PASS
- `test_v2_split_fingerprint_unchanged`: PASS

## N. V1 Weighted Loss Bug Document
Đã xác nhận lỗi báo cáo Epoch Aggregation của implementation V1:
V1 thực hiện: `loss.item() * inputs.size(0)`. Do hàm `CrossEntropyLoss` (với class weights) mặc định lấy mean có trọng số (weighted mean), việc nhân ngược lại bằng unweighted `batch_size` gây ra Epoch Loss sai lệch hoàn toàn (scaling by mismatched denominator). Lỗi này sẽ được sửa khi implement Training cho V2.

## O. Remaining Limitations
- Dataset Train nhỏ, có một lượng khá lớn (729 ảnh) lọt vào danh sách review do nền trắng chói sáng hoặc kích thước ảnh thấp (Heuristics dựa trên độ sáng và kích thước báo cờ cao). Cần augmentation thật sự tốt để bù đắp việc này.

## P. Decision Table cho Phase V2.3 (Training)
Dựa vào các kết quả EDA, dưới đây là quyết định cấu hình sơ bộ cho quá trình Training:

| Thành phần | Đánh giá / Decision | Rationale |
| :--- | :--- | :--- |
| **Imbalance Severity** | `SLIGHT` | Tỷ lệ class lớn nhất / nhỏ nhất = 1.41, chênh lệch không đáng kể. |
| **Class-weighted CE Loss** | `NO` | Không cần thiết do mức độ mất cân bằng thấp. Sẽ dùng chuẩn `CrossEntropyLoss`. |
| **WeightedRandomSampler** | `NO` | Không dùng để tránh bỏ sót các ảnh thiểu số quý giá hoặc oversample lặp lại quá nhiều gây overfit trên tập nhỏ. |
| **Minority Augmentation** | `NO` (Offline) / `YES` (On-the-fly) | Tăng cường Augmentation chung cho toàn dataset trên RAM để đa dạng hóa, không sinh thêm ảnh offline cho thiểu số. |
| **Normalization** | `ImageNet` | Sử dụng mean `[0.485, 0.456, 0.406]` & std `[0.229, 0.224, 0.225]` để tối ưu pre-trained weights. |
| **Label/Content Review** | `LATER` | Đã lưu 729 candidates nhưng hiện tại giữ nguyên cho Phase huấn luyện (Không can thiệp Label tự động). |
| **Dataset Readiness** | `READY` | Tập Train an toàn, không có Leakage hay Data Generated. |

==================================================
# FINAL VERDICT
==================================================

**V2_TRAIN_DATA_READY**

*(EDA chỉ đọc dữ liệu Train và xác nhận chính xác các đường dẫn Train bằng `loaded_train_paths == expected_train_paths_from_v2_manifest`. Dataset V2 hoàn toàn sạch sẽ, cách ly chặt chẽ khỏi Val/Test và sẵn sàng cho việc chuẩn bị DataLoaders).*
