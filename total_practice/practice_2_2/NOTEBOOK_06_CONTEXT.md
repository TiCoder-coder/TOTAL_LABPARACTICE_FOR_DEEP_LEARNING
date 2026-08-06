# NOTEBOOK 06 CONTEXT & SUMMARY

**Tài liệu Context dành cho AI/Agent.** Đọc kỹ tài liệu này để nắm bắt toàn bộ ngữ cảnh, kết quả và những quy định khắt khe của `06_accuracy_refactor_report.ipynb` mà không cần đọc lại toàn bộ lịch sử project.

---

## 1. Mục tiêu của Notebook 6
- Trình bày báo cáo tổng kết (presentation report) cho experiment Accuracy Refactor (Notebook 5).
- Đánh giá năng lực của các mô hình, phân tích Overfitting/Underfitting thông qua **Generalization Gap**.
- Tuân thủ chặt chẽ **Validation-Only Policy** (chỉ đánh giá mô hình dựa trên tập Validation, khóa hoàn toàn tập Test).

## 2. Nguồn Data/Artifact
Toàn bộ dữ liệu của Notebook 6 được lấy trực tiếp từ:
`artifacts/new_work/accuracy_refactor_validation_only_v1/`

## 3. Dataset / Manifest
- **Manifest:** `split_manifest_v3_format.csv`
- **Train:** 2,016 samples (Active)
- **Validation:** 438 samples (Active)
- **Test:** 451 samples (LOCKED - bị khóa chặt để tránh data leakage)

## 4. Models / Architectures đã train
- `ResNet18`
- `EfficientNet-B0`

## 5. Seeds và Configs
- **Seeds:** 42, 123, 2026.
- **Label Smoothing:** 0.0, 0.05.
- Đã chạy tổng cộng **12 single-model runs**.

## 6. Metrics của Best Single-Model Run
Mô hình đơn lẻ tốt nhất được ghi nhận (trước khi Ensemble) là **ResNet18 (Label Smoothing 0.05, Seed 123)**:
- **Train Accuracy:** 97.17%
- **Validation Accuracy:** 70.32%

## 7. Generalization Gap & Overfitting
- **Gap nhỏ nhất:** 11.33 pp (EfficientNet-B0, Seed 42).
- **Gap lớn nhất:** 28.15 pp (ResNet18, Seed 2026).
- **Gap trung bình:** 18.59 pp.
- **Generalization Gap của Best Run:** 26.85 pp (97.17% - 70.32%).
- **Đánh giá (Assessment):** Mặc dù Validation Accuracy đạt yêu cầu, nhưng Generalization Gap nằm trong ngưỡng **High/Very High**, cho thấy model có dấu hiệu bị **Overfitting** lên tập Train. Đây là tín hiệu báo động (Diagnostic) để cân nhắc cho các experiment sau (tăng augmentation, dropout, v.v.).

## 8. Best Single Model
- **ResNet18 (Label Smoothing 0.05, Seed 123)** (Validation Acc: 70.32%).

## 9. Final Selected Strategy (Mô hình chiến thắng cuối cùng)
- Mặc dù ResNet18 là best single model, thuật toán Selection Policy đã quyết định chọn **Ensemble** vì nó cho Validation Acc cao hơn.
- **Members (2):** `ResNet18` + `EfficientNet-B0` (sử dụng TTA scale 1.0).

## 10. Final Performance Metrics (của Ensemble)
- **Validation Accuracy:** 73.97%
- **Validation Macro F1:** 0.7371

## 11. Vì sao Ensemble Train Accuracy = N/A
- Pipeline (Notebook 5) chỉ tổng hợp kết quả Ensemble trên tập Validation để chọn model tốt nhất.
- Pipeline **không chạy cụm Ensemble trên toàn bộ tập Train** vì việc tính toán lại inference của tất cả các mô hình lên hàng ngàn ảnh Train rất tốn kém (computationally expensive), trong khi con số đó lại không có ý nghĩa trong việc ra quyết định chọn model.
- Do đó, Notebook 6 ghi rõ Train Accuracy của Ensemble là **N/A** (Không có trong artifact).

## 12. Test Status
- **NOT EVALUATED** (Test Evaluated = False). Pipeline chưa chạm vào tập Test.

## 13. Artifacts Quan Trọng
- `repeated_seed_summary.json`: Chứa lịch sử config, metrics (bao gồm `train_metrics` ẩn bên trong từng epoch của `epoch_history`).
- `validation_inference_selection.json`: Chứa quyết định chọn Ensemble, final Validation Metrics và Confusion Matrix.

## 14. Metrics/Artifacts chưa được record
- **Training Curves (Epoch-by-Epoch):** Notebook 5 hiện không persist lịch sử metrics từng epoch.
- **Prediction Gallery:** Không lưu ảnh dự đoán cụ thể của từng run.
*(Cả 2 section này đều được Notebook 6 xử lý Graceful Degradation: báo NOT AVAILABLE).*

## 15. NHỮNG QUY TẮC TUYỆT ĐỐI KHÔNG ĐƯỢC HIỂU NHẦM
- **Notebook 4 chỉ là UI Template:** Nó mang cấu trúc, theme, cách vẽ bảng, cách gọi HTML. Đừng lấy data của nó!
- **Notebook 5 là Data Source:** Nơi chứa logic training, inference và sinh ra các file `.json`.
- **Notebook 6 là Presentation:** Nhiệm vụ của nó là nhúng Data của NB5 vào Giao diện của NB4.
- **KHÔNG ĐƯỢC lấy metrics cũ (Canonical) của Notebook 4 nhét vào Notebook 6.**
- **KHÔNG ĐƯỢC tự bịa (fabricate) metric:** Nếu Artifact không có, hiển thị `N/A`.
- **KHÔNG ĐƯỢC chạy Inference lại:** Mọi con số phải lấy trực tiếp từ Artifact có sẵn.

---

## CURRENT STATUS

- **Best Single Model:** ResNet18 (Label Smoothing 0.05, Seed 123)
- **Final Selected Strategy:** Ensemble (2 members: ResNet18, EfficientNet-B0)
- **Train Acc (Best Single):** 97.17%
- **Train Acc (Ensemble):** N/A
- **Validation Acc (Ensemble):** 73.97%
- **Macro F1 (Ensemble):** 0.7371
- **Test Acc:** NOT EVALUATED (N/A)
- **Overfitting status:** High Gap (Mean Gap 18.59 pp, Best Run Gap 26.85 pp) -> Có dấu hiệu Overfitting.
- **Test evaluated:** False
