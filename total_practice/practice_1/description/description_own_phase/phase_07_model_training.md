# Phase 7 - Model Training

[Phase 6](phase_06_model_building.md) | [Mục lục](README.md) | [Mở đúng Cell 60][cell-60] | [Notebook dự phòng](../../practice_1.ipynb) | [Phase 8](phase_08_model_evaluation.md)

## 1. Vị trí và phạm vi

- Notebook cells: `60` đến `78`.
- Code cells đã chạy: `In [35]` đến `In [47]`.
- Input: model factory, train/augmented-train datasets, validation loader và các
  experiment config.
- Output: lịch sử train/validation của năm experiment, best checkpoint của từng
  experiment, experiment được chọn và final model train trên 60,000 ảnh.

| Nhóm nội dung | Code và output chính xác trong notebook |
|---|---|
| Training protocol diagram | [Cell 61, sơ đồ đã render][cell-61] |
| Reproducibility và device | [Cell 62, `In [35]` + output][cell-62] |
| Train-loader và optimizer factories | [Cell 63, `In [36]`][cell-63] |
| Validation loop | [Cell 64, `In [37]`][cell-64] |
| Batch training loop và sơ đồ | [Cell 65, `In [38]`][cell-65]; [Cell 66, sơ đồ đã render][cell-66] |
| Experiment runner | [Cell 67, `In [39]`][cell-67] |
| Controlled experiment configs | [Cell 69, `In [40]`][cell-69] |
| Chạy năm experiments | [Cell 70, `In [41]` + output][cell-70] |
| Experiment summary | [Cell 72, `In [42]` + output][cell-72] |
| Learning curves và comparison | [Cell 73, `In [43]` + output][cell-73]; [Cell 74, `In [44]` + output][cell-74] |
| Final-training function | [Cell 76, `In [45]`][cell-76] |
| Final-training run | [Cell 77, `In [46]` + output][cell-77] |
| TensorBoard integration | [Cell 78, `In [47]` + output][cell-78] |

## 2. Training protocol

Mỗi controlled experiment tuân theo cùng một protocol:

1. Reset seed.
2. Dựng model mới từ config.
3. Dựng optimizer mới.
4. Dựng train DataLoader mới với shuffle generator cùng seed.
5. Train trên 54,000 internal training images.
6. Evaluate trên 6,000 validation images sau mỗi epoch.
7. Giữ checkpoint có validation accuracy cao nhất.
8. Nếu accuracy bằng nhau, chọn validation loss thấp hơn.
9. Không truyền `test_loader` vào bất kỳ training/model-selection function nào.

Loss theo epoch được tính bằng sample-weighted mean. Accuracy được giữ nội bộ ở
dạng fraction `[0, 1]` và chỉ format thành phần trăm khi in/vẽ.

Diagram ngay dưới heading mô tả epoch loop và checkpoint selection.

[Mở training protocol diagram tại đúng Cell 61][cell-61].

## 3. Reproducibility và chọn device

[Mở device-selection code và output tại đúng Cell 62][cell-62].

Cell `In [35]` định nghĩa `set_reproducibility(seed)` để seed:

- Python `random`;
- NumPy;
- PyTorch qua `torch.manual_seed`.

`select_device()` ưu tiên theo thứ tự:

```text
CUDA nếu torch.cuda.is_available()
    -> MPS nếu torch.backends.mps.is_available()
    -> CPU nếu không có accelerator trên
```

Output của lần chạy đang lưu:

```text
Selected device: mps
```

`mps` là backend GPU của PyTorch trên Apple Silicon thông qua Metal Performance
Shaders. Model và batch được chuyển sang device này trong training/evaluation
loop bằng `.to(device)`.

Code hiện tại cho phép fallback về CPU; nó không raise lỗi nếu không có GPU. Seed
giúp kiểm soát nhiều nguồn ngẫu nhiên nhưng không cấu hình deterministic
algorithms tuyệt đối, nên kết quả có thể chênh nhẹ giữa backend hoặc phiên bản
thư viện.

## 4. Tạo train loader mới cho từng run

[Mở train-loader factory tại đúng Cell 63][cell-63].

`make_train_loader(dataset, batch_size, seed)` tạo một
`torch.Generator().manual_seed(seed)` mới và trả DataLoader với:

- `shuffle=True`;
- `batch_size` lấy từ config;
- `num_workers=0`.

Việc tạo generator mới ngăn experiment sau nhận shuffle state đã bị experiment
trước tiêu thụ. Với cùng dataset và seed, các non-augmentation experiment bắt đầu
từ cùng thứ tự shuffle logic. E4 vẫn tạo ảnh ngẫu nhiên vì transform có flip và
rotation.

## 5. Optimizer factory

[Mở optimizer factory tại đúng Cell 63][cell-63].

`build_optimizer(model, config)` hỗ trợ hai nhánh:

| Optimizer | Tham số được đọc |
|---|---|
| Adam | `learning_rate`, `weight_decay` |
| SGD | `learning_rate`, `momentum`, `weight_decay` |

Tên optimizer được đổi sang lowercase trước khi so sánh. Một tên khác Adam/SGD
sẽ raise `ValueError`, giúp config lỗi không âm thầm dùng default ngoài dự kiến.

## 6. Validation loop: `evaluate_loader`

[Mở validation-loop implementation tại đúng Cell 64][cell-64].

Hàm validation thực hiện:

```text
model.eval()
    -> torch.inference_mode()
    -> chuyển images/labels sang device
    -> forward logits
    -> CrossEntropyLoss
    -> kiểm tra loss finite
    -> cộng loss * batch_size
    -> argmax logits và đếm prediction đúng
    -> chia tổng theo sample_count
```

`model.eval()` tắt hành vi train-time của Dropout. `torch.inference_mode()` tắt
gradient tracking để giảm overhead. Nhân batch loss với batch size trước khi cộng
giúp epoch loss là mean theo sample ngay cả khi batch cuối nhỏ hơn 64.

Hàm raise lỗi nếu loader rỗng và trả:

- `loss`;
- `accuracy`;
- `sample_count`.

## 7. Batch training loop: `train_one_epoch`

[Mở training-loop code tại Cell 65][cell-65] và
[batch-level diagram tại Cell 66][cell-66].

Diagram batch-level nằm ngay trước function trong notebook. Với mỗi batch, hàm
thực hiện đúng chuỗi training PyTorch:

```text
images, labels
    -> chuyển sang device
    -> optimizer.zero_grad(set_to_none=True)
    -> logits = model(images)
    -> loss = criterion(logits, labels)
    -> kiểm tra loss finite
    -> loss.backward()
    -> optimizer.step()
```

Sau đó hàm tích lũy:

- sample-weighted cross-entropy loss;
- số prediction đúng từ `argmax`;
- tổng sample;
- thời gian epoch bằng `time.perf_counter()`.

`set_to_none=True` tránh ghi zero vào mọi gradient tensor và là cách reset gradient
hiệu quả được PyTorch hỗ trợ. Loss được kiểm tra finite ở từng batch; nếu NaN/Inf
xuất hiện, training dừng ngay và báo index của batch.

## 8. Experiment runner: `run_training`

[Mở experiment-runner implementation tại đúng Cell 67][cell-67].

`run_training(...)` là orchestration layer của một experiment.

### Khởi tạo độc lập

Hàm gọi lại seed, model factory, criterion, optimizer và train loader. Vì vậy
không experiment nào fine-tune tiếp trọng số của experiment trước.

### Epoch history

Mỗi epoch tạo record gồm:

| Field | Ý nghĩa |
|---|---|
| `epoch` | Số epoch bắt đầu từ 1 |
| `train_loss` | Sample-weighted train CE loss |
| `train_accuracy` | Train accuracy dạng fraction |
| `validation_loss` | Sample-weighted validation CE loss |
| `validation_accuracy` | Validation accuracy dạng fraction |
| `learning_rate` | Learning rate hiện tại từ optimizer |
| `elapsed_seconds` | Thời gian riêng của train epoch |

### Best checkpoint

Khi validation tốt hơn, hàm clone mọi tensor trong `state_dict` về CPU. Clone này
tạo snapshot thực sự; nếu chỉ giữ reference, các optimizer step sau sẽ tiếp tục
thay đổi tensor của checkpoint.

Quy tắc so sánh:

```text
validation_accuracy cao hơn
    hoặc
accuracy bằng nhau và validation_loss thấp hơn
```

Sau epoch cuối, best state được nạp lại vào model và model được đặt ở eval mode.
Do đó object trả về không nhất thiết chứa trọng số epoch cuối; nó chứa trọng số
của best validation epoch.

### TensorBoard

Khi có `log_directory`, `SummaryWriter` ghi:

- `Loss/Train`;
- `Loss/Validation`;
- `Accuracy/Train`;
- `Accuracy/Validation`;
- `LearningRate`.

Writer luôn được đóng trong `finally`, kể cả khi training raise exception.

## 9. Controlled experiments

[Mở controlled-experiment configs tại đúng Cell 69][cell-69].

Mọi experiment dùng seed 42, batch size 64, 10 epoch, 10 classes, input 784 và
weight decay 0. Các yếu tố thay đổi:

| ID | Hidden dims | Dropout | Optimizer | Learning rate | Augmentation | Mục đích so sánh |
|---|---|---:|---|---:|---|---|
| E0_baseline | `(128,)` | `0.0` | Adam | `0.001` | Không | Mốc baseline |
| E1_deeper | `(256, 128)` | `0.0` | Adam | `0.001` | Không | Tác động của capacity/depth |
| E2_dropout | `(256, 128)` | `0.2` | Adam | `0.001` | Không | Tác động của dropout trên anchor E1 |
| E3_sgd | `(256, 128)` | `0.0` | SGD, momentum `0.9` | `0.01` | Không | Tác động của optimizer trên anchor E1 |
| E4_augmentation | `(256, 128)` | `0.0` | Adam | `0.001` | Có | Tác động của flip/rotation trên anchor E1 |

Thiết kế diễn ra theo hai tầng:

- E1 thay architecture của E0.
- E2/E3/E4 giữ architecture E1 và mỗi run thay đúng một yếu tố bổ sung.

E0 và E1 không chỉ khác độ sâu mà cả width hidden đầu (`128` thành `256`), nên
factor đang được kiểm tra chính xác là **architecture/capacity**, không chỉ số
layer.

## 10. Chạy experiment và lưu artifact

[Mở experiment execution và toàn bộ training output tại đúng Cell 70][cell-70].

Cell `In [41]` tạo:

```text
runs/<YYYYMMDD-HHMMSS>/<experiment_id>/
outputs/experiments/<experiment_id>.pth
```

Mỗi experiment checkpoint chứa config, best epoch, best validation metrics và
history. TensorBoard event file được tách theo experiment ID trong cùng run
session.

Sau khi cả năm run hoàn tất, `max(...)` chọn result theo tuple:

```text
(best_validation_accuracy, -best_validation_loss)
```

## 11. Kết quả năm experiment hiện tại

[Mở Experiment Results và output tại đúng Cell 72][cell-72].

| Experiment | Best val accuracy | Best val loss | Best epoch | Parameters | Runtime |
|---|---:|---:|---:|---:|---:|
| E0_baseline | `88.43%` | `0.3265` | 6 | 101,770 | 69.3 s |
| E1_deeper | `89.15%` | `0.3177` | 7 | 235,146 | 76.1 s |
| E2_dropout | `88.72%` | `0.3279` | 8 | 235,146 | 80.0 s |
| E3_sgd | `88.93%` | `0.3183` | 10 | 235,146 | 81.3 s |
| E4_augmentation | `87.82%` | `0.3366` | 10 | 235,146 | 104.2 s |

Theo primary metric, `E1_deeper` được chọn với:

- best epoch: `7`;
- best validation accuracy: `89.15%`;
- best validation loss: `0.3177`.

Dropout và augmentation ở đúng cấu hình đang thử không cải thiện accuracy so với
E1. SGD đạt gần E1 nhưng vẫn thấp hơn 0.22 percentage point trong lần chạy này.

## 12. Learning curves và experiment comparison

[Mở learning curves tại Cell 73][cell-73] và
[experiment comparison tại Cell 74][cell-74].

Notebook lấy history của experiment được chọn rồi vẽ hai panel:

- train/validation cross-entropy loss theo epoch;
- train/validation accuracy theo epoch.

Biểu đồ cho phép quan sát convergence và khoảng cách train-validation. Notebook
cũng vẽ bar chart best validation accuracy của năm experiment, ghi phần trăm trên
từng cột.

Các chart này được hiển thị inline bằng `plt.show()`; code hiện tại không gọi
`savefig` trong các cell này.

## 13. Final training sau model selection

[Mở final-training function tại Cell 76][cell-76] và
[final-training run/output tại Cell 77][cell-77].

Sau khi đã khóa config và số epoch, notebook dựng một model mới hoàn toàn và train
trên toàn bộ 60,000 official training images.

Quy trình:

1. Copy `selected_config`.
2. Đổi experiment ID thành `E1_deeper_final`.
3. Đặt số epoch bằng best epoch của E1, tức 7.
4. Chọn full `baseline_training_pool` vì E1 không dùng augmentation.
5. Khởi tạo fresh model/optimizer/loader.
6. Train đủ 7 epoch, không chạy validation và không chọn lại checkpoint.

Việc đưa 6,000 validation images trở lại training chỉ diễn ra **sau** khi
architecture, optimizer và epoch count đã được quyết định. Nhờ vậy final model có
thể học từ toàn bộ 60,000 labeled development images mà không dùng official test.

Output hiện tại:

| Epoch | Train loss | Train accuracy |
|---:|---:|---:|
| 1 | `0.4648` | `83.00%` |
| 2 | `0.3439` | `87.27%` |
| 3 | `0.3110` | `88.47%` |
| 4 | `0.2834` | `89.47%` |
| 5 | `0.2664` | `89.96%` |
| 6 | `0.2473` | `90.61%` |
| 7 | `0.2323` | `91.30%` |

Final training sample count được assert/in là `60,000`.

## 14. TensorBoard notebook integration

[Mở TensorBoard integration và output tại đúng Cell 78][cell-78].

Cell cuối Phase 7 chạy:

```text
%load_ext tensorboard
%tensorboard --logdir $TENSORBOARD_LOG_DIR
```

`TENSORBOARD_LOG_DIR` trỏ tới toàn bộ `RUNS_DIR`, cho phép xem và so sánh scalar
của năm experiment cùng final run ngay trong notebook.

## 15. Flow hoàn chỉnh của phase

```text
chọn mps
    -> định nghĩa loader/optimizer/train/evaluate functions
    -> khai báo E0-E4
    -> với từng experiment:
         fresh model + fresh optimizer + fresh loader
         train 54,000
         validate 6,000 mỗi epoch
         giữ best validation state
         ghi TensorBoard + checkpoint
    -> chọn E1_deeper theo validation accuracy
    -> lấy best epoch = 7
    -> fresh final E1 model
    -> train 7 epoch trên toàn bộ 60,000
    -> chuyển final model sang eval mode
```

## 16. Đầu ra cho phase sau

[Phase 8](phase_08_model_evaluation.md) nhận final `model`, `test_loader`,
`device` và class names. Đây là thời điểm model lần đầu được chấm bằng official
test metrics để báo cáo khả năng tổng quát hóa.

[cell-60]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=60>
[cell-61]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=61>
[cell-62]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=62>
[cell-63]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=63>
[cell-64]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=64>
[cell-65]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=65>
[cell-66]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=66>
[cell-67]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=67>
[cell-69]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=69>
[cell-70]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=70>
[cell-72]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=72>
[cell-73]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=73>
[cell-74]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=74>
[cell-76]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=76>
[cell-77]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=77>
[cell-78]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=78>
