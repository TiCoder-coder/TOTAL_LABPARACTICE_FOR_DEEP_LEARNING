# Practice 2.2 Phase Reports

Thư mục này chứa các báo cáo (Reports) theo từng giai đoạn (Phases) của Practice 2.2 V2, được sắp xếp theo thứ tự thời gian và mức độ phát triển của pipeline:

1. **[PHASE V2.1 — Data Split](PHASE_V2_1_SPLIT_REPORT.md)**
   - Khởi tạo split `v2_stratified_group_s42` (70/15/15) đảm bảo group-level isolation.
2. **[PHASE V2.2 — Train-Only EDA](PHASE_V2_2_TRAIN_ONLY_EDA_REPORT.md)**
   - Phân tích EDA độc lập, cam kết Zero Data Leakage từ Validation và Test set.
3. **[PHASE V2.3 — Data Pipeline](PHASE_V2_3_DATA_PIPELINE_REPORT.md)**
   - Xây dựng Data Pipeline với Augmentation.
4. **[PHASE V2.4 — Baseline Training](PHASE_V2_4_BASELINE_TRAINING_REPORT.md)**
   - Train baseline model `E1` (Head only) và `E2` (Layer4 + Head) với ImageNet pretrained weights.
5. **[PHASE V2.5A — Visual Leakage Audit](PHASE_V2_5A_VISUAL_LEAKAGE_AUDIT_REPORT.md)**
   - Kiểm tra rò rỉ dữ liệu (Visual Data Leakage), phát hiện ảnh gần giống nhau (near-duplicates) bị lẫn giữa các split.
6. **[PHASE V2.5B — Visual Group Resplit](PHASE_V2_5B_VISUAL_GROUP_RESPLIT_REPORT.md)**
   - Khắc phục Data Leakage: gom nhóm lại (Visual Groups) và tạo split mới an toàn `v2_visual_group_stratified_s42`.
7. **[PHASE V2.5 — Weight Decay Regularization](PHASE_V2_5_WEIGHT_DECAY_REPORT.md)**
   - Áp dụng kỹ thuật Regularization trên model (experiment E3, trước khi khắc phục triệt để Visual Leakage).
8. **[PHASE V2.6 — Visual-Safe Baseline Retraining](PHASE_V2_6_VISUALSAFE_BASELINE_REPORT.md)**
   - Train lại mô hình Baseline (`E1` & `E2`) trên split đã được làm sạch rò rỉ. Đánh giá lại Generalization Gap.

> *Lưu ý: Quá trình Test evaluation hoàn toàn bị cách ly cho đến khi mọi experiments được chốt để đảm bảo tính minh bạch.*
