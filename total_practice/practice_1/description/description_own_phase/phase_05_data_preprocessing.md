# Phase 5 - Data Preprocessing

[Phase 4](phase_04_exploratory_data_analysis.md) | [Mục lục](README.md) | [Mở đúng Cell 44][cell-44] | [Notebook dự phòng](../../practice_1.ipynb) | [Phase 6](phase_06_model_building.md)

## 1. Vị trí và phạm vi

- Notebook cells: `44` đến `54`.
- Code cells đã chạy: `In [27]` đến `In [31]`.
- Input: official training pool 60,000 ảnh, official test set 10,000 ảnh,
  `SEED = 42` và kết luận EDA.
- Output: train/validation indices, train-only mean/std, ba transform policy,
  các dataset view, subset và DataLoader.

| Nhóm nội dung | Code và output chính xác trong notebook |
|---|---|
| Heading Phase 5 | [Cell 44][cell-44] |
| Split/transform strategy và sơ đồ | [Cell 45][cell-45]; [Cell 46, sơ đồ đã render][cell-46] |
| Split, train-only normalization và transforms | [Cell 47, `In [27]` + output][cell-47] |
| Dataset construction | [Cell 49, `In [28]`][cell-49] |
| DataLoaders | [Cell 51, `In [29]`][cell-51] |
| Split/loader sanity checks | [Cell 53, `In [30]` + output][cell-53] |
| Batch contract checks | [Cell 54, `In [31]` + output][cell-54] |

## 2. Mục tiêu thiết kế

Preprocessing phải đồng thời bảo đảm:

1. Train và validation không trùng sample.
2. Mỗi lớp giữ cùng tỷ lệ 90/10.
3. Mean/std chỉ được fit từ internal train.
4. Validation và test không có random augmentation.
5. Mỗi experiment có thể chọn baseline hoặc augmented train dataset mà không
   làm thay đổi validation.
6. Mọi batch tuân thủ shape/dtype contract của model.

[Mở strategy tại Cell 45][cell-45], [sơ đồ tại Cell 46][cell-46] và
[implementation/output tại Cell 47][cell-47].

## 3. Stratified split 90/10

Notebook đặt `VALIDATION_RATIO = 0.10` và tạo một `torch.Generator` có seed 42.
Split được thực hiện riêng cho từng class từ 0 đến 9:

1. Lấy toàn bộ index có label bằng class hiện tại.
2. Sinh permutation bằng generator cố định.
3. Lấy 10% đầu làm validation.
4. Lấy 90% còn lại làm train.
5. Ghép index của 10 class.
6. Shuffle lại danh sách train và validation bằng cùng generator.

Vì mỗi class ban đầu có 6,000 ảnh:

| Split | Mỗi class | Tổng |
|---|---:|---:|
| Internal train | 5,400 | 54,000 |
| Internal validation | 600 | 6,000 |

Cách chia này deterministic khi seed và dataset không đổi. Nó cũng bảo toàn cân
bằng lớp chính xác, thay vì chỉ kỳ vọng cân bằng xấp xỉ từ random split toàn cục.

## 4. Fit normalization chỉ trên train

Notebook chọn raw pixel ở `train_indices`, đổi sang `float32`, chia 255, rồi tính:

| Thống kê | Output hiện tại |
|---|---:|
| `TRAIN_MEAN` | `0.286139` |
| `TRAIN_STD` | `0.353084` |

Temporary tensor `training_pixels` được xóa sau khi tính để giải phóng bộ nhớ.

Điểm quan trọng là validation indices không tham gia phép tính này. Nếu mean/std
được fit trên cả 60,000 ảnh, thông tin phân phối validation sẽ đi vào transform
trước khi model được đánh giá, tạo preprocessing leakage.

## 5. Ba transform policy

### Baseline training transform

```text
ToTensor
    -> Normalize(TRAIN_MEAN, TRAIN_STD)
```

Transform này hoàn toàn deterministic. Nó được dùng cho E0, E1, E2 và E3 để mỗi
experiment chỉ thay đổi factor được công bố.

### Augmented training transform

```text
RandomHorizontalFlip(p=0.5)
    -> RandomRotation(degrees=10)
    -> ToTensor
    -> Normalize(TRAIN_MEAN, TRAIN_STD)
```

Transform này chỉ dùng cho E4. Flip và rotation tạo biến thể hình học nhỏ trong
lúc lấy sample, còn normalization vẫn dùng đúng thống kê internal train. Random
operation nằm trước `ToTensor()` nên chúng xử lý ảnh theo representation mà
TorchVision hỗ trợ.

### Evaluation transform

```text
ToTensor
    -> Normalize(TRAIN_MEAN, TRAIN_STD)
```

Validation và test dùng transform deterministic giống baseline về numerical
preprocessing. Không có random flip/rotation, vì cùng một sample phải tạo cùng
input trong mọi lần đánh giá.

## 6. Ý nghĩa của normalization

Với mỗi pixel `x` sau khi scale về `[0, 1]`, transform tính:

```text
z = (x - TRAIN_MEAN) / TRAIN_STD
```

Kết quả đưa dữ liệu về miền có mean gần 0 và scale gần 1 trên internal train.
Điều này thường giúp optimizer làm việc với input có scale ổn định hơn. Với
mean/std hiện tại, raw pixel 0 trở thành khoảng `-0.810`, còn raw pixel 1 trở
thành khoảng `2.022`, đúng với output sanity check.

## 7. Tạo các dataset view độc lập

[Mở Dataset Construction code tại đúng Cell 49][cell-49].

Notebook tạo lại nhiều `datasets.FashionMNIST` object cùng trỏ tới dữ liệu trên
disk nhưng mang transform khác nhau:

| Dataset view | Partition | Transform |
|---|---|---|
| `baseline_training_pool` | Official train | Baseline deterministic |
| `augmented_training_pool` | Official train | Random augmentation + normalize |
| `validation_pool` | Official train | Evaluation deterministic |
| `test_dataset` | Official test | Evaluation deterministic |

Sau đó `Subset` ánh xạ indices:

| Subset | Parent dataset | Indices | Kích thước |
|---|---|---|---:|
| `train_subset` | Baseline training pool | `train_indices` | 54,000 |
| `augmented_train_subset` | Augmented training pool | `train_indices` | 54,000 |
| `validation_subset` | Validation pool | `validation_indices` | 6,000 |

Hai train subset dùng cùng indices, vì vậy E4 chỉ thay đổi augmentation chứ không
đổi sample. Separate parent datasets ngăn validation vô tình thừa hưởng random
transform của train.

Sau model selection, Phase 7 dùng toàn bộ parent training pool 60,000 ảnh để final
training; không dùng `Subset` 54,000 ở bước cuối.

## 8. DataLoader policy

[Mở DataLoader code tại đúng Cell 51][cell-51].

Notebook đặt:

```text
BATCH_SIZE = 64
num_workers = 0
```

| Loader | Dataset | Shuffle | Generator |
|---|---|---|---|
| `train_loader` | `train_subset` | `True` | Seed 42 |
| `validation_loader` | `validation_subset` | `False` | Không cần |
| `test_loader` | `test_dataset` | `False` | Không cần |

Train được shuffle để thứ tự batch không cố định theo class/sample. Validation và
test giữ thứ tự ổn định vì thứ tự không ảnh hưởng metric và determinism giúp việc
đối chiếu prediction dễ hơn.

`num_workers=0` làm việc đọc dữ liệu diễn ra trong process chính. Cấu hình này ưu
tiên tính tương thích và reproducibility trong notebook; nó không phải tuyên bố
rằng đây luôn là giá trị throughput cao nhất.

Phase 7 tạo lại train loader qua factory cho từng experiment, để mọi run nhận một
shuffle generator mới cùng seed thay vì kế thừa state đã bị tiêu thụ.

## 9. Sanity checks cho split

[Mở split/loader sanity checks và output tại đúng Cell 53][cell-53].

Cell `In [30]` assert:

- train subset có 54,000 sample;
- augmented train subset có 54,000 sample;
- validation subset có 6,000 sample;
- official test có 10,000 sample;
- train và validation không có index giao nhau;
- hợp train/validation chứa đúng 60,000 unique indices;
- train có chính xác 5,400 sample mỗi class;
- validation có chính xác 600 sample mỗi class;
- truy xuất cùng validation sample hai lần cho image và label giống hệt nhau.

Số batch output:

| Loader | Số batch |
|---|---:|
| Train | `844` |
| Validation | `94` |
| Test | `157` |

Batch cuối có thể nhỏ hơn 64 vì code không đặt `drop_last=True`.

## 10. Sanity checks cho batch contract

[Mở batch contract checks và output tại đúng Cell 54][cell-54].

Cell `In [31]` lấy batch đầu từ ba loader và kiểm tra:

- ảnh có 4 chiều `[B, C, H, W]`;
- mỗi ảnh có shape `[1, 28, 28]`;
- ảnh có dtype `torch.float32`;
- label có shape `[B]` và dtype `torch.int64`;
- mọi pixel sau transform đều finite.

Output hiện tại của cả ba batch:

```text
images=(64, 1, 28, 28)
labels=(64,)
range=[-0.810, 2.022]
```

### Lưu ý bám sát implementation

Cell sanity check hiện tại có materialize batch đầu của `test_loader` để kiểm tra
shape, dtype, finiteness và transformed range. Batch test không được dùng để tính
metric, fit preprocessing, chọn hyperparameter hay chọn checkpoint. Vì vậy ranh
giới modeling vẫn được giữ, dù cụm từ “untouched until final evaluation” nếu hiểu
theo nghĩa tuyệt đối sẽ chặt hơn code hiện tại. Tài liệu ghi rõ hành vi này để
không mô tả khác notebook.

## 11. Flow hoàn chỉnh của phase

```text
60,000 official train labels
    -> per-class seeded permutation
    -> 54,000 train indices + 6,000 validation indices
    -> fit TRAIN_MEAN/TRAIN_STD trên 54,000 raw train images
    -> tạo baseline/augmented/evaluation transforms
    -> tạo parent dataset views độc lập
    -> áp indices bằng Subset
    -> tạo train/validation/test DataLoaders
    -> assert split, class balance, determinism, batch contract
```

## 12. Đầu ra cho phase sau

Phase 6 và Phase 7 nhận:

- `train_subset` và `augmented_train_subset`;
- `validation_subset` và `validation_loader`;
- `test_dataset` và `test_loader`;
- `TRAIN_MEAN`, `TRAIN_STD`;
- `BATCH_SIZE`;
- các batch đầu `train_images`, `train_labels` cho model sanity check.

[Phase 6](phase_06_model_building.md) dùng batch contract này để xây MLP nhận
`[B, 1, 28, 28]` và trả `[B, 10]` raw logits.

[cell-44]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=44>
[cell-45]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=45>
[cell-46]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=46>
[cell-47]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=47>
[cell-49]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=49>
[cell-51]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=51>
[cell-53]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=53>
[cell-54]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=54>
