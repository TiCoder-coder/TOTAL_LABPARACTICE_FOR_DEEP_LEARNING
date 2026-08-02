# Phase 2 - Environment Setup

[Phase 1](phase_01_problem_definition.md) | [Mục lục](README.md) | [Mở đúng Cell 2][cell-2] | [Notebook dự phòng](../../practice_1.ipynb) | [Phase 3](phase_03_data_loading.md)

## 1. Vị trí và phạm vi

- Notebook cells: `2` đến `5`.
- Code cells đã chạy: `In [1]` đến `In [3]`.
- Trách nhiệm: tìm project root, khai báo thư mục dữ liệu/output/log, ghi nhận
  runtime và import dependency dùng cho toàn bộ pipeline.

| Nội dung | Đích chính xác trong notebook |
|---|---|
| Heading Phase 2 | [Cell 2][cell-2] |
| Project path và artifact directories | [Cell 3, `In [1]`][cell-3] |
| Inline plotting, runtime và accelerator output | [Cell 4, `In [2]` + output][cell-4] |
| Dependency imports | [Cell 5, `In [3]`][cell-5] |

## 2. Xác định đường dẫn project

Cell `In [1]` bắt đầu từ `Path.cwd().resolve()` rồi kiểm tra hai trường hợp chạy
notebook thường gặp:

1. Current working directory đã là thư mục `practice_1`.
2. Current working directory là project root và notebook nằm tại
   `total_practice/practice_1`.

Candidate đầu tiên chứa `practice_1.ipynb` được chọn làm `PROJECT_DIR`. Từ đó
notebook xây dựng ba đường dẫn trung tâm:

| Biến | Mục đích |
|---|---|
| `DATA_DIR` | Nơi TorchVision tải và đọc FashionMNIST |
| `OUTPUT_DIR` | Checkpoint, biểu đồ và các artifact kết quả |
| `RUNS_DIR` | TensorBoard event logs của từng experiment |

Cách resolve này giúp notebook ít phụ thuộc vào việc VS Code/Jupyter được mở từ
thư mục nào. Mọi phase sau dùng lại cùng các object `Path`, tránh hard-code đường
dẫn ở nhiều nơi.

## 3. Thiết lập Python import path

Notebook thêm `PROJECT_DIR.parent` vào `sys.path` nếu đường dẫn đó chưa có sẵn.
Mục đích là cho phép import package theo dạng:

```python
from practice_1.processing_own_phase.data import analyze_training_pool
```

Đây là dependency trực tiếp của Phase 4. Khi chạy notebook từ kernel sạch, các
module được import phải tồn tại dưới `practice_1/processing_own_phase` và phải
import được từ environment hiện tại.

## 4. Cấu hình hiển thị notebook

Cell `In [2]` chạy magic:

```python
%matplotlib inline
```

Magic này yêu cầu Matplotlib render figure ngay trong output cell. Nhờ vậy EDA,
training curves, confusion matrix và predicted-versus-actual grid được nhúng vào
notebook thay vì chỉ mở ở một cửa sổ bên ngoài.

## 5. Ghi nhận runtime

Notebook import `torch` và `torchvision`, sau đó in phiên bản Python, PyTorch,
TorchVision và khả năng dùng CUDA/MPS. Output đang lưu là:

| Kiểm tra | Giá trị hiện tại |
|---|---|
| Python | `3.10.11` |
| PyTorch | `2.13.0` |
| TorchVision | `0.28.0` |
| CUDA available | `False` |
| MPS available | `True` |

Ý nghĩa:

- CUDA không khả dụng trong environment/máy hiện tại.
- MPS khả dụng, nên Phase 7 chọn `torch.device('mps')` để dùng GPU Apple Silicon.
- Việc in version giúp truy vết khác biệt khi một lần chạy khác cho output hoặc
  hành vi thư viện khác.

## 6. Các thư viện được sử dụng

Cell `In [3]` tập trung import dependency cho các phase sau:

| Thư viện/API | Được dùng để làm gì |
|---|---|
| `random`, `numpy.random`, `torch` | Seed và reproducibility |
| `time` | Đo thời gian train và đặt tên run session |
| `torch.nn` | Layer, module và `CrossEntropyLoss` |
| `torch.optim` | Adam và SGD |
| `torchvision.datasets` | Tải FashionMNIST |
| `torchvision.transforms` | `ToTensor`, normalize và augmentation |
| `matplotlib.pyplot` | Vẽ EDA, learning curves, confusion matrix, sample grid |
| `numpy` | Xử lý array và thống kê/phép biến đổi cho EDA |
| `seaborn` | Histogram, heatmap và biểu đồ phân phối |
| `sklearn.metrics` | Classification report và confusion matrix |
| `collections.Counter` | Tiện ích đếm được import sẵn |
| `DataLoader`, `Subset` | Batch dữ liệu và ánh xạ train/validation indices |
| `SummaryWriter` | Ghi metric vào TensorBoard |
| `typing` | Type hint cho config, sequence và giá trị optional |

Một số import EDA chuyên biệt như `PCA`, `StandardScaler` và `TSNE` được đặt ngay
tại cell sử dụng trong Phase 4 thay vì ở import block chung.

## 7. Flow thực thi

```text
Path.cwd()
    -> tìm PROJECT_DIR
    -> tạo object Path cho data/outputs/runs
    -> mở đường import cho package practice_1
    -> bật inline plotting
    -> in version và accelerator availability
    -> import dependency dùng trong notebook
```

Thứ tự này quan trọng vì Phase 3 cần `DATA_DIR`, Phase 4 cần import path và
`OUTPUT_DIR`, Phase 7 cần `RUNS_DIR`, còn Phase 9 cần lại `OUTPUT_DIR` để lưu
checkpoint.

## 8. Điều phase này không thực hiện

- Chưa tải dữ liệu.
- Chưa chọn device train; Phase 7 mới gọi `select_device()`.
- Chưa tạo thư mục bằng `mkdir`; các API tải dữ liệu hoặc lưu artifact sẽ tạo khi
  cần.
- Chưa train model và chưa thay đổi random state ngoài các import.

## 9. Đầu ra và điều kiện hoàn thành

Phase 2 hoàn thành khi:

- `PROJECT_DIR`, `DATA_DIR`, `OUTPUT_DIR`, `RUNS_DIR` trỏ đúng vị trí;
- kernel import được PyTorch, TorchVision, scikit-learn và TensorBoard;
- thông tin version/accelerator được in thành công;
- Matplotlib có thể hiển thị inline.

Sau đó [Phase 3](phase_03_data_loading.md) dùng `DATA_DIR`, `datasets` và
`transforms` để tải hai partition chính thức của FashionMNIST.

[cell-2]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=2>
[cell-3]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=3>
[cell-4]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=4>
[cell-5]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=5>
