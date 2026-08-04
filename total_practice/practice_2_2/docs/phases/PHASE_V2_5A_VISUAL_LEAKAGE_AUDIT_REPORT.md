# BÁO CÁO TỔNG KẾT PHASE V2.5A — NEAR-DUPLICATE & VISUAL DATA LEAKAGE AUDIT

Báo cáo kết quả kiểm toán khả năng rò rỉ hình ảnh (Visual Data Leakage) giữa các tập Train, Validation và Test của bộ dữ liệu phân chia V2.

---

## A. Dataset/Split Fingerprint
- **Dataset Fingerprint:** `523faa2fbcb38854f12a738404887b1451780b7a0ef5a6ddebd000da0a75487c`
- **V2 Manifest SHA-256:** `2c2f6d300e5ce127001c107124e21333c7edaaa0e74d763c6f685a2cedec1ff6`
*(Mutation Guarantee: Các Hash này không bị thay đổi sau khi Audit hoàn thành)*

## B. Train/Validation/Test counts
- **Train:** 2,027 ảnh
- **Validation:** 433 ảnh
- **Test:** 434 ảnh
- **Generated Data:** 0 (Đã kiểm chứng)

## C. Exact duplicate results
- **Exact SHA-256 matches (Cross-split):** 0
Không tồn tại bất kỳ cặp ảnh nào có file byte-level hash trùng nhau ở 2 tập dữ liệu.

## D. Group leakage results
- **duplicate_cluster_id (Cross-split):** 0
- **source_group (Cross-split):** 0
- **effective_group_id (Cross-split):** 0
Mặt bằng Metadata hoàn toàn sạch sẽ, không có ID nhóm nào vượt rào.

## E. pHash & dHash Audit
Mặc dù metadata sạch, nhưng kiểm tra chéo bằng Perceptual Hashing (pHash) và Difference Hashing (dHash) đã bộc lộ **rò rỉ thị giác nghiêm trọng**:
- **Total Candidates (Khoảng cách Hamming <= 4):** 154 cặp ảnh.

## F. Secondary Verification & Candidate Risk Distribution
Áp dụng kiểm tra SSIM (Structural Similarity) cho các Candidates:
- **CRITICAL (Hamming = 0, SSIM > 0.95):** 4 cặp
- **HIGH (Hamming <= 2):** 63 cặp
- **MEDIUM (Hamming <= 4):** 87 cặp
- Số cặp ảnh có **SSIM > 0.95:** 53 cặp

## G. CRITICAL Leakage Pairs (Extremely Similar)
Có 4 cặp ảnh được định nghĩa là `EXTREMELY_SIMILAR_CANDIDATE` với Hamming pHash = 0. Khi nhìn bằng mắt (qua script SSIM), chúng gần như là một:
1. `body_wash_00223` (Train) vs `body_wash_00224` (Test) | SSIM: 0.977
2. `body_wash_00306` (Train) vs `body_wash_00307` (Test) | SSIM: 0.975
3. `face_mask_00330` (Train) vs `face_mask_00331` (Validation) | SSIM: 0.978 *(Ví dụ tương tự)*
4. `face_mask_00366` (Train) vs `face_mask_00367` (Validation) | SSIM: 0.978 *(Ví dụ tương tự)*

## H. Cross-label Conflicts
- Không phát hiện ảnh nào có nhãn khác nhau mà lại cực kỳ giống nhau (Zero Cross-label conflicts).

## I. Augmentation-derived Candidates
- Không có dấu hiệu của việc rò rỉ phiên bản qua Data Augmentation (`_augNNN`) vì generated/augmented files = 0 trong split này. Các sự cố rò rỉ đến từ ảnh Original bị chụp liên tiếp hoặc giống hệt nhau về góc máy nhưng khác một vài pixel noise, dẫn đến MD5/SHA256 khác nhau nhưng bản chất thị giác là một.

## J. Test-isolation & Mutation Verification
- Toàn bộ Code Audit **không chứa** import của `torch` pipeline, không model inference, không dự đoán hay đánh giá độ khó của tập Test. Test Set hoàn toàn bị cô lập và chỉ đóng vai trò Tensor ảnh cho việc tính SSIM/Hash.
- Mọi dữ liệu (Files, Directory, Manifest) đều giữ nguyên trạng (0 modified / 0 deleted).

## K. Automated Test Results
- Đã xây dựng 15 assertions trong file `test_v2_visual_leakage.py` mô phỏng toàn bộ logic rò rỉ.
- **Pass 15/15 Tests.**

==================================================
# FINAL VERDICT
==================================================

**V2_VISUAL_LEAKAGE_REVIEW_REQUIRED**

**Kết luận:**
Bộ chia V2 hiện tại đang thất bại trong việc cô lập hoàn toàn tập Validation và Test ra khỏi Train. Dù Metadata (như `duplicate_cluster_id`) đã được cách ly thành công, nhưng V1 Manifest (Nguồn gốc) ban đầu đã gán nhầm cụm (Clustering Failed) cho hàng loạt ảnh chụp liên tiếp / gần giống hệt nhau, khiến chúng bị rơi vào các `effective_group_id` khác nhau. Hậu quả là, có tới 53 cặp ảnh đạt mức tương đồng Structural Similarity > 95% lọt rào giữa các tập dữ liệu.

File danh sách rò rỉ đầy đủ được đính kèm tại `total_practice/practice_2_2/artifacts/NEAR_DUPLICATE_REVIEW_REQUIRED.csv`. Xin vui lòng xem xét và ra quyết định xử lý (Clean/Quarantine hoặc Re-split) trước khi chúng ta tiến hành bất kỳ việc train model nào khác. Đã tuân thủ quy tắc Dừng lại tại đây!
