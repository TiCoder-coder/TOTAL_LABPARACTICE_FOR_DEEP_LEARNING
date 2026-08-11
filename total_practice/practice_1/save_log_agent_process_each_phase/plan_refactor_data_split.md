# Plan Refactor Data Split - `practice_1.ipynb`

> **Trạng thái:** Superseded ngày 2026-07-25. Theo quyết định mới của chủ project,
> notebook không tạo validation split; toàn bộ 60,000 official training images được
> dùng cho EDA/preprocessing/training và 10,000 official test images chỉ dùng cho
> final evaluation. Nội dung bên dưới được giữ lại như lịch sử quyết định trước đó.

**Ngày thực hiện:** 2026-07-25  
**Phạm vi:** data loading, data split, EDA, preprocessing, DataLoaders và phase
numbering trong `practice_1.ipynb`.

## 1. Data protocol đã chốt

FashionMNIST cung cấp sẵn:

- Official training set: 60,000 images.
- Official test set: 10,000 images.

Project giữ nguyên official test set để kết quả cuối còn so sánh được với benchmark.
Official training set được split reproducibly:

| Split | Nguồn | Số mẫu | Vai trò |
|---|---|---:|---|
| Train | Official train | 54,000 | EDA, fit preprocessing, train model |
| Validation | Official train | 6,000 | Validate sau mỗi epoch, chọn model |
| Test | Official test | 10,000 | Final evaluation sau khi khóa cấu hình |

Tỷ lệ trong official training pool là 90/10. Không gộp official test vào training
pool để ép tỷ lệ tổng thể thành 80/10/10.

## 2. Phase mapping mới

| Phase | Tên |
|---:|---|
| 1 | Define Problem |
| 2 | Environment Setup |
| 3 | Data Loading |
| 4 | Data Split |
| 5 | Exploratory Data Analysis |
| 6 | Data Preprocessing |
| 7 | Model Building |
| 8 | Model Training |
| 9 | Model Evaluation |
| 10 | Save Model & Visualization |

Tất cả phase dùng số nguyên dương, liên tục và không có Phase 0.

## 3. Thiết kế implementation

1. Phase 3 tải official train/test với deterministic `ToTensor()` để phục vụ split
   và EDA.
2. Phase 4 tạo train/validation indices bằng
   `torch.utils.data.random_split()` và generator seed 42.
3. Assert kích thước, tính không overlap và coverage của train/validation indices.
4. Phase 5 chỉ thực hiện EDA trên train subset.
5. Phase 6 định nghĩa:
   - Train transform: augmentation, `ToTensor()`, normalization.
   - Evaluation transform: `ToTensor()`, normalization, không augmentation.
6. Tạo hai FashionMNIST instances cho official train:
   - Một instance mang train transform.
   - Một instance mang evaluation transform.
7. Dùng cùng indices để tạo augmented train subset và deterministic validation
   subset.
8. Tạo train/validation/test DataLoaders sau khi split và sau khi gắn transform.
9. Xóa split và validation DataLoader cũ khỏi Model Training.
10. Giữ test loader chỉ cho Model Evaluation và prediction visualization.

## 4. Acceptance criteria

- Train/validation/test có lần lượt 54,000/6,000/10,000 samples.
- Train/validation indices không overlap và cover đủ 60,000 official train samples.
- `len(train_loader.dataset) == 54_000`.
- `len(val_loader.dataset) == 6_000`.
- `len(test_loader.dataset) == 10_000`.
- Train transform có randomness; validation/test transform deterministic.
- EDA không truy cập validation hoặc test samples.
- Training loop chỉ dùng `train_loader`.
- Validation loop chỉ dùng `val_loader`.
- Final evaluation chỉ dùng `test_loader`.
- Batch image shape `[B, 1, 28, 28]`, dtype `float32`.
- Batch label shape `[B]`, dtype `int64`.
- Notebook không còn phase 0 hoặc split nằm trong Model Training.

## 5. Verification plan

1. Parse notebook và kiểm tra phase headings.
2. Compile toàn bộ Python code cells sau khi loại IPython magics.
3. Chạy data-only smoke test bằng dataset local.
4. Kiểm tra sizes, index intersection, index coverage và loader references.
5. Đọc cùng validation/test sample hai lần để kiểm tra determinism.
6. Kiểm tra transformed batch shape, dtype và range.
7. Kiểm tra diff và bảo đảm không sửa ngoài phạm vi.

## 6. Trạng thái

- [x] Chốt data protocol.
- [x] Thiết kế phase mapping.
- [x] Refactor notebook.
- [x] Chạy verification.
- [x] Ghi kết quả cuối.

## 7. Kết quả verification

| Kiểm tra | Kết quả |
|---|---|
| Phase headings | Phase 1 đến Phase 10, liên tục |
| Python code-cell compilation | 43/43 code cells pass |
| Train/validation/test sizes | 54,000 / 6,000 / 10,000 |
| Train/validation overlap | 0 |
| Train/validation coverage | 60,000 |
| Train/validation/test batches | 844 / 94 / 157 |
| Image batch shape/dtype | `[64,1,28,28]` / `float32` |
| Label batch shape/dtype | `[64]` / `int64` |
| Normalized train batch range | `[-1.0,1.0]` |
| Validation transform deterministic | Pass |
| Test transform deterministic | Pass |
| Train augmentation observable | Pass |
| Official test preserved | Pass |

## 8. Thay đổi đã thực hiện

1. Đổi Phase 0 thành Phase 1 và đánh lại toàn bộ phase đến Phase 10.
2. Thêm Phase 4 - Data Split ngay sau Data Loading.
3. Split official training set bằng seed 42 và lưu indices để tái sử dụng.
4. Chuyển EDA sang Phase 5 và giới hạn toàn bộ EDA ở train subset.
5. Tạo riêng augmented train source và deterministic validation source.
6. Tạo đủ ba DataLoaders sau split/preprocessing.
7. Xóa split và `val_loader` cũ khỏi Model Training.
8. Giữ validation trong training loop và official test trong final evaluation.
9. Xóa notebook outputs cũ vì chúng được tạo từ data protocol có leakage.

## 9. Phạm vi verification

Đã chạy syntax check và data-only runtime smoke test bằng FashionMNIST local.
Chưa chạy lại 10 epochs training vì environment hiện tại còn thiếu `scikit-learn`
và `tensorboard`; các metrics cũ đã được xóa để không trình bày kết quả không còn
hợp lệ.
