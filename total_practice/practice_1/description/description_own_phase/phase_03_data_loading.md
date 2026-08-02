# Phase 3 - Data Loading

[Phase 2](phase_02_environment_setup.md) | [Mục lục](README.md) | [Mở đúng Cell 6][cell-6] | [Notebook dự phòng](../../practice_1.ipynb) | [Phase 4](phase_04_exploratory_data_analysis.md)

## 1. Vị trí và phạm vi

- Notebook cells: `6` và `7`.
- Code cell đã chạy: `In [4]`.
- Input: đường dẫn `DATA_DIR`, TorchVision và seed từ các phase trước.
- Output: `train_dataset`, `test_dataset` và `class_names`.

| Nội dung | Đích chính xác trong notebook |
|---|---|
| Heading Phase 3 | [Cell 6][cell-6] |
| Seed, raw transform, tải dữ liệu, class names và output kích thước | [Cell 7, `In [4]` + output][cell-7] |

## 2. Seed ban đầu

Notebook định nghĩa:

```python
SEED = 42
```

và seed ba nguồn sinh số ngẫu nhiên:

- `random.seed(SEED)` cho Python;
- `np.random.seed(SEED)` cho NumPy;
- `torch.manual_seed(SEED)` cho PyTorch.

Data loading bản thân nó không cần random để đọc FashionMNIST, nhưng seed này
được dùng tiếp cho EDA sample selection, stratified split, DataLoader shuffle và
khởi tạo/training model. Khai báo sớm tạo một nguồn cấu hình thống nhất.

## 3. Raw transform

Ở phase này notebook chỉ dùng:

```python
raw_transform = transforms.ToTensor()
```

`ToTensor()` thực hiện bốn thay đổi cần thiết khi một sample được lấy qua
`dataset[index]`:

1. Chuyển PIL image hoặc NumPy image sang `torch.Tensor`.
2. Đưa channel lên trước, tạo shape `[1, 28, 28]`.
3. Chuyển dtype ảnh sang `torch.float32`.
4. Scale pixel `uint8` từ `[0, 255]` sang số thực trong `[0, 1]`.

Việc scale tạo miền số phù hợp cho neural network và gradient-based optimization.
Việc đổi sang channel-first tuân theo convention `[N, C, H, W]` mà PyTorch sử
dụng khi gom sample thành batch.

Phase này chưa normalize theo mean/std và chưa augmentation. Nhờ vậy EDA có thể
đọc raw storage qua `train_dataset.data` trong miền `[0, 255]`, đồng thời vẫn có
thể lấy transformed sample trong miền `[0, 1]` qua `train_dataset[index]`.

## 4. Tải official training pool

Notebook gọi `datasets.FashionMNIST` với:

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `root` | `DATA_DIR` | Thư mục lưu/read dataset |
| `train` | `True` | Chọn official training partition |
| `download` | `True` | Tải nếu dữ liệu chưa tồn tại |
| `transform` | `raw_transform` | Chỉ áp dụng `ToTensor()` khi truy xuất sample |

Object kết quả là `train_dataset`, có 60,000 ảnh gắn nhãn. Trong project này nó
được gọi là **official training pool** vì Phase 5 mới chia nó thành internal train
và validation.

## 5. Tải official test set

Lần gọi thứ hai chỉ thay `train=False`, tạo `test_dataset` gồm 10,000 ảnh chính
thức dành cho đánh giá cuối. Ở thời điểm này object được tải vào bộ nhớ chương
trình nhưng protocol quy định nội dung test không được dùng cho EDA, preprocessing
statistics, hyperparameter comparison hoặc model selection.

## 6. Lấy tên lớp

```python
class_names = train_dataset.classes
```

Danh sách này giữ đúng mapping label-to-name của TorchVision. Các phase sau dùng
nó để:

- in class distribution;
- gắn label lên biểu đồ EDA;
- tạo `classification_report`;
- gắn trục confusion matrix;
- hiển thị predicted và actual class trong sample grid;
- lưu metadata cùng checkpoint.

## 7. Output đang lưu

Code cell in:

```text
Official training pool: 60,000
Official test set: 10,000
```

Đây là kiểm tra nhanh rằng đúng hai partition chuẩn đã được tải. Phase 4 và Phase
5 bổ sung assertion chi tiết hơn cho shape, dtype, range, label và kích thước sau
split.

## 8. Hai cách nhìn cùng một dataset

TorchVision FashionMNIST cho phép notebook sử dụng hai representation:

| Cách truy cập | Representation | Được dùng ở đâu |
|---|---|---|
| `train_dataset.data` | Raw tensor `[60000, 28, 28]`, `torch.uint8`, `[0, 255]` | EDA chính xác trên dữ liệu gốc; tính train mean/std sau split |
| `train_dataset[index][0]` | Tensor `[1, 28, 28]`, `torch.float32`, `[0, 1]` | DataLoader EDA và kiểm tra transformed contract |

Phân biệt này giải thích vì sao schema EDA có raw dtype `uint8`, nhưng sample đi
qua transform lại là `float32`.

## 9. Flow của phase

```text
SEED = 42
    -> seed Python, NumPy, PyTorch
    -> tạo raw_transform = ToTensor()
    -> tải train=True thành train_dataset (60,000)
    -> tải train=False thành test_dataset (10,000)
    -> lấy class_names từ metadata
    -> in kích thước hai partition
```

## 10. Ranh giới trách nhiệm

Phase 3 chỉ làm **loading**. Những việc sau được cố ý để sang phase chuyên trách:

- EDA: Phase 4.
- Stratified train/validation split: Phase 5.
- Tính normalization từ internal train: Phase 5.
- Random augmentation: Phase 5, chỉ dành cho experiment E4.
- Batching và shuffle policy: Phase 5/7.
- Official test inference: Phase 8.

Tách các trách nhiệm này giúp tránh việc transform ngẫu nhiên làm biến dạng EDA
hoặc việc thống kê validation/test đi vào preprocessing.

## 11. Liên kết sang phase tiếp theo

[Phase 4](phase_04_exploratory_data_analysis.md) sử dụng riêng
`train_dataset.data`, `train_dataset.targets` và `class_names` để kiểm tra schema,
phân phối lớp, pixel, chất lượng dữ liệu, cấu trúc chiều và các mẫu đại diện.

[cell-6]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=6>
[cell-7]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=7>
