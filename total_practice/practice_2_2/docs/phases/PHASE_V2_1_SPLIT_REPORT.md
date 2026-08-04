# BÁO CÁO TỔNG KẾT PHASE V2.1 — STRATIFIED GROUP-AWARE SPLIT

Dựa trên kết quả triển khai bằng thuật toán **Greedy Group-Aware Stratified Split**, dưới đây là báo cáo toàn diện cho tập dữ liệu Practice 2.2 phiên bản V2 (Lineage `v2_stratified_group_s42`).

---

## A. Input Original Dataset
- **Tổng dataset hiện tại (bao gồm generated):** 3,202 ảnh
- **Total usable originals (đầu vào của thuật toán V2):** 2,894 ảnh

## B. Excluded Generated/Quarantine Data
Tất cả các ảnh không phải ảnh gốc hợp lệ đã bị loại trừ hoàn toàn khỏi Split V2.
- **Excluded Generated Images:** 306 ảnh
- **Excluded Quarantined Images:** 2 ảnh
- **Tổng số ảnh không dùng cho mô hình bị loại (is_model=False):** 308 ảnh

## C. Group Definition
- Thuật toán nội suy ra trường `effective_group_id` bằng mô hình đồ thị có hướng (connected components) qua quan hệ giữa `duplicate_cluster_id` và `source_group`.
- Bất kỳ ảnh nào nằm chung cây phả hệ (source) hoặc thuộc chung cụm trùng lặp đều được coi là một khối thống nhất (atomic group) và bị khóa cứng không thể xé lẻ vào các split khác nhau.

## D. Split Algorithm
- **Thuật toán:** Greedy joint optimization. Sắp xếp các `effective_group_id` theo kích thước giảm dần. Khi gán mỗi nhóm vào một split, hàm phạt (penalty) được tính dựa trên độ lệch dung lượng (target capacity) và độ lệch phân phối lớp (class stratification deviation). Nhóm sẽ được đẩy vào Split có Penalty thấp nhất.

## E. Random Seed
- **Seed:** 42

## F. Target Ratio
- **Mục tiêu:** 70% (Train) / 15% (Validation) / 15% (Test)

## G. Actual Ratio (Tỷ lệ Đạt Được)
- **Train:** 2,027 ảnh (**70.04%**)
- **Validation:** 433 ảnh (**14.96%**)
- **Test:** 434 ảnh (**15.00%**)
> *Đánh giá: Tỷ lệ gần như hoàn hảo.*

## H. Per-class Distribution

| Class | Available | Train | Validation | Test |
| :--- | :--- | :--- | :--- | :--- |
| **body_wash** | 303 | 215 | 44 | 44 |
| **face_mask** | 297 | 210 | 43 | 44 |
| **facial_cleanser** | 321 | 224 | 46 | 51 |
| **lipstick** | 286 | 200 | 44 | 42 |
| **moisturizer** | 228 | 159 | 34 | 35 |
| **perfume** | 287 | 202 | 42 | 43 |
| **serum** | 300 | 208 | 47 | 45 |
| **shampoo** | 274 | 191 | 42 | 41 |
| **sunscreen** | 318 | 225 | 46 | 47 |
| **toner** | 280 | 193 | 45 | 42 |

## I. Train Class Balance Statistics
Trong tập Train:
- **Min class count:** 159 (moisturizer)
- **Max class count:** 225 (sunscreen)
- **Max/Min ratio:** ~1.41
> *Mặc dù vẫn còn lệch do bản chất các khối (group) bị mất cân bằng tự nhiên (moisturizer ít ảnh độc lập hơn sunscreen), thuật toán đã nỗ lực phân bố tỷ lệ tối ưu nhất để bù đắp mà không phá vỡ tính cô lập cụm.*

## J. Validation Class Balance Statistics
- Chênh lệch nhỏ nhất 34, cao nhất 47. Tỷ lệ được phản chiếu khá đều.

## K. Test Class Balance Statistics
- Chênh lệch nhỏ nhất 35, cao nhất 51. Tỷ lệ phản chiếu hợp lý.

## L. Distribution Deviation
- **Max Class Distribution Deviation:** **1.07%** (xảy ra ở lớp `toner`)
> *Đánh giá: Stratification Quality cực kỳ tốt. Phân phối lớp ở cả Train/Val/Test gần như giống hệt phân phối trong kho ảnh gốc, sai số không vượt quá 1.1%.*

## M. Leakage Verification
- **Exact hash cross split:** 0 (PASS)
- **Duplicate cluster cross split:** 0 (PASS)
- **Source group cross split:** 0 (PASS)
- **Effective group cross split:** 0 (PASS)

## N. Generated-data Verification
- **Generated in Train:** 0
- **Generated in Validation:** 0
- **Generated in Test:** 0
> *Tất cả các ảnh generated (306 file) đã bị loại trừ khỏi Split. Lỗi nghiêm trọng của V1 đã được vá triệt để.*

## O. Reproducibility Verification
- Kiểm tra độc lập trên test suite tự động (`pytest`):
  - Chạy `seed=42` nhiều lần sinh ra cùng một dataset assignment. (PASS)
  - Chạy bằng command line và chạy trong script đưa ra mã băm (`split_fingerprint`) đồng nhất. (PASS)

## P. V1 vs V2 Comparison

| Metric | Canonical V1 | V2 Stratified Group (Seed 42) | Đánh giá |
| :--- | :--- | :--- | :--- |
| **Total usable originals** | 2,894 | 2,894 | = |
| **Train originals** | 2,016 | 2,027 | Đạt Target 70% chuẩn xác hơn |
| **Validation originals** | 438 | 433 | Đạt Target 15% |
| **Test originals** | 440 | 434 | Đạt Target 15% |
| **Generated Train** | 212 | 0 | Đã được làm sạch |
| **Generated Validation** | **58 (Lỗi)** | **0 (Chuẩn)** | Fixed |
| **Generated Test** | **36 (Lỗi)** | **0 (Chuẩn)** | Fixed |
| **Duplicate leakage** | 0 | 0 | Safe |
| **Source-group leakage** | 0 | 0 | Safe |
| **Split strategy** | Unknown/Random Group | Greedy Group Stratified | Tối ưu độ lệch < 1.1% |

## Q. V2 Fingerprints
- **Dataset Fingerprint SHA-256:** `523faa2fbcb38854f12a738404887b1451780b7a0ef5a6ddebd000da0a75487c`
- **Split Fingerprint SHA-256:** `2157484123560b5dddf1d30d27d3945def7eaac663dbf2552176a2896446661e`
- **Manifest SHA-256:** `2c2f6d300e5ce127001c107124e21333c7edaaa0e74d763c6f685a2cedec1ff6`

## R. Automated Test Results
- Tổng số Test đã viết trong `total_practice/practice_2_2/tests/test_v2_split.py`: **17 Tests**
- **Trạng thái:** Toàn bộ 17/17 (100%) đều PASS. 
- *Bao gồm tất cả các kiểm tra vô trùng (như test_no_generated_in_test), kiểm tra rò rỉ (test_group_isolation), kiểm tra seed (test_seed_42_deterministic) và giữ nguyên V1 (test_v1_manifest_unchanged).*

## S. Remaining Limitations
- **Natural Imbalance:** Do phải duy trì tính toàn vẹn (isolation) của các Duplicate Clusters lớn, lớp có ít cụm lớn (như `moisturizer`) vẫn còn thiếu hụt ảnh trong tập Train so với lớp khác. *Khuyến nghị: Áp dụng Augmentation On-the-fly (Transform) hoặc Offline Augmentation nhắm riêng cho các class thiểu số này TRONG KHI huấn luyện trên tập Train V2.*

==================================================
# FINAL VERDICT
==================================================

**V2_SPLIT_READY**

*(Tất cả 17 Assertions đều PASS, Leakage = 0, Không chứa Augmented Data ở Val/Test, Phân phối lớp (Stratification) cực tốt, Tỷ lệ sát 70/15/15 và V1 Baseline được bảo tồn 100%).*
