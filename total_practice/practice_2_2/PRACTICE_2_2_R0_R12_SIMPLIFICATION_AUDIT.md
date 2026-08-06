# PRACTICE 2.2 R0-R12 SIMPLIFICATION AUDIT

**Audit Date**: 2026-08-06
**Type**: READ-ONLY AUDIT
**Verdict**: **READY_FOR_VISUAL_LEAKAGE_FIX**

---

## A. Executive Summary

Quá trình refactor R0–R12 đã thiết lập một hệ thống quản lý vô cùng chặt chẽ với các "fail-closed gates", tuy nhiên điều này tạo ra rào cản quá mức (over-engineering) so với mục tiêu thực tế của Practice 2.2. Vấn đề cốt lõi lớn nhất hiện tại vẫn là **Visual Leakage** (rò rỉ dữ liệu thị giác) khi vẫn còn 22 cặp ảnh `CONFIRMED_NEAR_DUPLICATE` nằm vắt chéo giữa các tập Train, Validation và Test.

Mục tiêu hiện tại cần đơn giản hóa luồng làm việc: Dừng ngay các thí nghiệm phức tạp (E3, E4, M0-M3 staged transfer, R10 robust selection), dọn dẹp các mã nguồn thừa, và tập trung 100% vào việc xử lý dứt điểm rò rỉ dữ liệu trước khi train lại baseline E1/E2.

## B. Current Repository State

- Pipeline chính: Đang bị **BLOCKED** hoàn toàn bởi luật Leakage và các cổng kiểm duyệt (Gates) chưa có người xác nhận.
- Model Performance: Các mô hình cũ (E2) cho thấy dấu hiệu overfit nặng (Gap > 29%).
- Codebase: Chứa rất nhiều file thừa, file cũ (V1, V2.4) và các kịch bản R0-R12 chưa hoàn thiện.

## C. Current Data Authority

- **Dataset**: `total_practice/practice_2_2/data/final/data_clean_balanced`
- **Current Manifest**: `data/manifests/v2_visual_group_stratified_s42/split_manifest.csv`
- **Identity Fields**: `visual_group_id`, `duplicate_cluster_id`, `source_group`.
- Mặc dù đã có các cơ chế group, việc phân tách split vẫn chưa triệt để dẫn đến rò rỉ.

## D. Remaining Visual Leakage

Dựa trên file `V2_REMAINING_VISUAL_PAIRS_REVIEW.csv`, tổng quan rò rỉ như sau:

- **Tổng số cặp (Total Pairs)**: 49
- **REVIEW_ONLY (Cần xem xét)**: 49 (đã được audit bằng rule)
- **CONFIRMED_NEAR_DUPLICATE**: 22
- **Train-Val**: 27 cặp
- **Train-Test**: 15 cặp
- **Val-Test**: 7 cặp
- **Same-label**: 45 cặp
- **Cross-label**: 4 cặp

**Rule xác nhận (Confirmation Rule)**: Những cặp có `same_label` (cùng nhãn dán) và `SSIM > 0.85` (vượt ngưỡng tương đồng thị giác) sẽ tự động chuyển từ `REVIEW_ONLY` sang `CONFIRMED_NEAR_DUPLICATE`.

*Hệ quả: Pipeline hiện tại chưa an toàn để Train/Test.*

## E. R0-R12 Classification

| Phase | Mục tiêu | Trạng thái / Phân loại |
|---|---|---|
| R0 (Authority Gate) | Kiểm kê và chặn mọi rủi ro | **REMOVE_CANDIDATE** (Quá phức tạp) |
| R1 (Ontology Contract) | Đặt luật gán nhãn | **SIMPLIFY** (Giữ luật, bỏ chặn) |
| R2 (Provenance Audit) | Quản lý nguồn gốc ảnh | **REMOVE_CANDIDATE** |
| R3 (Grouping) | Gom nhóm ảnh trùng lặp | **KEEP/SIMPLIFY** (Quan trọng để sửa Leakage) |
| R4 (Stratified Split) | Chia tỷ lệ dữ liệu an toàn | **KEEP/SIMPLIFY** |
| R5 (Train-only EDA) | Cách ly Test/Val khỏi EDA | **KEEP** |
| R6 (Transform Ablation) | So sánh an toàn Augmentation | **SIMPLIFY** (Chỉ dùng Safe Transform) |
| R7 (Training Math) | Sửa lỗi chia Loss | **KEEP** |
| R8 (Staged Transfer) | Luồng Fine-tuning phức tạp | **REMOVE_CANDIDATE** (Quay về V2.6) |
| R9 (Arch Ceiling) | So sánh Model Arch | **REMOVE_CANDIDATE** |
| R10 (Robust Selection) | Chọn Model đa hạt giống | **REMOVE_CANDIDATE** |
| R11 (One-Time Test) | Đánh giá Test 1 lần | **SIMPLIFY** |
| R12 (Report Builder) | Tự động hóa HTML/Notebook | **REMOVE_CANDIDATE** |

## F. V2.1-V2.6 Classification

| Phase | Phân loại |
|---|---|
| V2.1 (Ngẫu nhiên -> Stratified) | **HISTORICAL** |
| V2.2 (Train-only EDA) | **KEEP** |
| V2.3 (Contracts) | **KEEP** |
| V2.4 (Clean Baseline) | **HISTORICAL** |
| V2.5 (Weight Decay E3) | **HISTORICAL** |
| V2.5A/B (Visual Leakage Audit) | **BLOCKER_TO_FIX** |
| V2.6 (Visual-safe Baseline) | **KEEP** (Là pipeline chính hiện tại) |

## G. Duplicate/Overlapping Pipelines

- R7/R8 cố gắng định nghĩa lại Baseline Training nhưng quá phức tạp, trùng lặp với `train_v2_6_baseline.py` vốn đã đầy đủ và đơn giản.
- Việc chia Split ở R4 đang bị chồng chéo với kịch bản của V2.5B. Cần gom lại thành 1 kịch bản chia Split duy nhất.

## H. Active Code

- `src/practice_2_2/train_v2_6_baseline.py`
- `src/practice_2_2/v2_dataset.py`
- `src/practice_2_2/v2_transforms.py`
- `src/practice_2_2/v2_models.py`
- `src/practice_2_2/verify_reload_v2_6.py`

## I. Historical Code

- `src/practice_2_2/train_practice_2_2.py`
- `src/practice_2_2/train_v2_baseline.py`
- `src/practice_2_2/canonical_train_practice_2_2.py`

## J. Remove Candidates

- Toàn bộ các file R0-R12: `src/practice_2_2/r*.py`, `scripts/*_r*.py`, `configs/r*.json`.
- Các markdown file giải thích R0-R12 quá dài dòng.

## K. E3/E4 Cleanup Candidates

Current Decision: **KHÔNG TRAIN E3/E4**.
Các ứng viên cần dọn dẹp:
- `src/practice_2_2/canonical_e3_practice_2_2.py`
- `src/practice_2_2/canonical_e4_practice_2_2.py`
- Các điều kiện rẽ nhánh `elif strategy in ["E2", "E3"]:` bên trong `v2_models.py` và `train_v2_baseline.py`.

## L. Test Isolation

- **TEST_INFERENCE_COUNT**: 0 (Chưa từng đánh giá model hiện tại trên tập Test).
- **TEST_USED_FOR_TUNING**: False.
- **TEST_USED_FOR_EDA**: False.
- **TEST_USED_FOR_LEAKAGE_AUDIT**: True (Chỉ dùng Manifest Metadata để kiểm tra rò rỉ).

## M. Loss/Balancing State

Baseline chuẩn và Active (V2.6) đang áp dụng:
- **CrossEntropyLoss(weight=None)**: Chuẩn (Không dùng weight).
- **WeightedRandomSampler = False**: Chuẩn (Bỏ hoàn toàn).
- **Offline augmentation = False**: Chuẩn.
- **On-the-fly augmentation = True**: Chuẩn.
- Legacy epoch-loss aggregation: Đã được sửa để chia đúng số lượng sample/weight.

## N. V2.6 Metric Source-of-Truth

Lấy trực tiếp từ Artifacts: `v2_visualsafe_baseline_s42_v1`

**E1 (Head Only)**:
- Best Epoch: 15
- Train Accuracy: 65.13%
- Validation Accuracy: 57.60%
- Validation Loss: 1.375
- Macro F1: 0.5608
- Generalization Gap: 7.53% (Underfit)

**E2 (Layer4 + Head)**:
- Best Epoch: 8
- Train Accuracy: 98.27%
- Validation Accuracy: 69.12%
- Validation Loss: 1.200
- Macro F1: 0.6819
- Generalization Gap: 29.15% (Overfit nặng)

*Nguyên nhân bất đồng trước đây*: E1 (Train ≈ 62.72%) có thể thuộc một lineage hoặc split cũ trước khi Visual Leakage được fix. JSON artifact V2.6 là Source-of-Truth cuối cùng.

## O. Current Blockers

1. Dữ liệu: 22 cặp `CONFIRMED_NEAR_DUPLICATE` nằm chéo split.
2. Hệ thống: Các cơ chế R0-R12 đang làm khóa pipeline, không thể đưa vào vận hành trơn tru.

## P. Minimum Required Pipeline

Pipeline tối giản và khoa học:

`DATA AUTHORITY -> VISUAL LEAKAGE FIX -> VISUAL GROUP RESPLIT -> LEAKAGE AUDIT -> TRAIN-ONLY EDA -> SAFE TRANSFORMS -> E1 / E2 BASELINE -> VALIDATION COMPARISON -> OVERFITTING ASSESSMENT -> OPTIONAL REGULARIZATION -> FINAL TEST -> REPORT`

## Q. Files To Keep

- Pipeline V2.6: `train_v2_6_baseline.py`, `v2_dataset.py`, `v2_transforms.py`, `v2_models.py`.
- Tập metadata: `v2_visual_group_stratified_s42`.

## R. Files To Simplify

- `v2_models.py`: Bỏ logic kiểm tra E3/E4.

## S. Files To Archive Later

- Các kịch bản V1, V2.4, Canonical Train.
- Toàn bộ mã nguồn/config của R0-R12 và M0-M3.
- `canonical_e3_...`, `canonical_e4_...`.

## T. Files That Must Not Be Deleted

- Image Data: `data/final/data_clean_balanced/`.
- File audit rò rỉ: `V2_REMAINING_VISUAL_PAIRS_REVIEW.csv`.

## U. Recommended Next Phase

1. Tắt toàn bộ rào cản R0-R12.
2. Dùng file CSV audit để cập nhật `visual_group_id` cho 22 cặp ảnh trùng.
3. Chạy lệnh Resplit lại dữ liệu và kiểm tra lại toàn bộ rò rỉ (`exact_hash_cross_split == 0`, v.v).

---

## 11. Over-Engineering Review

| Thành phần | Câu hỏi (A/B/C/D) | Đánh giá |
|---|---|---|
| Ontology contract | D | Có thể bỏ chặn, chỉ dùng như guideline. |
| Provenance framework | C | Quá mức (Over-engineering). |
| Product identity system | B | Tốt nhưng quá phức tạp, gom nhóm bằng Visual Hash là đủ. |
| R0 authority gate | C | Quá mức. Gây kẹt pipeline. |
| R1/R2 manual review | D | Không cần thiết để block code run. |
| M0-M3 fine-tuning | C | Quá phức tạp. V2.6 baseline đủ tốt. |
| Architecture comparison | D | Chưa cần thiết ở giai đoạn Baseline. |
| Multi-seed model selection | B | Tốt, nhưng không bắt buộc cho Practice 2.2. |
| One-time Test machinery | B | Isolate Test là đủ. |
| Report builder | C | Jupyter Notebook tĩnh hoặc Markdown là đủ. |
| Fail-closed gates | C | Quá cứng nhắc, cản trở testing. |

---

## 12. Component Status Table

| Component | Current Status | Decision | Reason |
|---|---|---|---|
| R0 (Authority) | Blocked | REMOVE_CANDIDATE | Gây tắc nghẽn, không hợp cho lab practice. |
| R1 (Labels) | Pending | SIMPLIFY | Dùng làm Guideline, không làm Gate. |
| R2 (Provenance) | Blocked | REMOVE_CANDIDATE | Quá mức quản trị dữ liệu hiện tại. |
| R3 (Grouping) | Blocked | KEEP/SIMPLIFY | Cần thiết để fix Duplicate. |
| R4 (Split) | Blocked | KEEP/SIMPLIFY | Phục vụ Resplit. |
| R5 (EDA) | Blocked | KEEP | Giữ tính trung thực (Không EDA Test). |
| R6 (Transform) | Blocked | SIMPLIFY | Dùng luôn Safe Transforms V2.6. |
| R7 (Loss Math) | Pass (Synthetic) | KEEP | Đã tích hợp vào V2.6. |
| R8 (Transfer) | Blocked | REMOVE_CANDIDATE | Fallback về Baseline Train E1/E2. |
| R9 (Arch) | Blocked | REMOVE_CANDIDATE | Yêu cầu bài không bắt buộc đổi Arch. |
| R10 (Selection) | Blocked | REMOVE_CANDIDATE | 1 Seed 1 Split là đủ theo yêu cầu Lab. |
| R11 (Test) | Not Executed | SIMPLIFY | Tránh Inference Test. |
| R12 (Report) | Incomplete | REMOVE_CANDIDATE | Quá kỹ thuật. |

---

## 13. Action Priority Table

| Priority | Action |
|---|---|
| **P0** | Sửa Visual Leakage (Merge 22 cặp `CONFIRMED_NEAR_DUPLICATE` vào cùng 1 Visual Group). |
| **P1** | Chạy lại kịch bản Resplit (Stratified Component Split) và Xác minh Leakage = 0. |
| **P2** | Train lại E1 và E2 bằng Pipeline V2.6 sau khi Split được làm sạch. |
| **P3** | Dọn dẹp/Archive các mã nguồn và tệp cấu hình không cần thiết (R0-R12, E3, E4). |
