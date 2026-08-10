# Phase 7 - Model Training

[Phase 6](phase_06_model_building.md) | [Mục lục](README.md) | [Mở đúng Cell 60][cell-60] | [Notebook dự phòng](../../practice_1.ipynb) | [Phase 8](phase_08_model_evaluation.md)

## 1. Vị trí và phạm vi

- Notebook cells: `60` đến `78`.
- Code cells đã chạy: `In [35]` đến `In [47]`.
- Input: model factory, train/augmented-train datasets, validation loader và các
  experiment config.
- Output: controlled-baseline results, 17 staged-search trials, multi-seed
  confirmation, selected hyperparameters và final model train trên 60,000 ảnh.

| Nhóm nội dung | Code và output chính xác trong notebook |
|---|---|
| Training protocol diagram | [Cell 61, sơ đồ đã render][cell-61] |
| Reproducibility và device | [Cell 62, `In [35]` + output][cell-62] |
| Train-loader và optimizer factories | [Cell 63, `In [36]`][cell-63] |
| Validation loop | [Cell 64, `In [37]`][cell-64] |
| Batch training loop và sơ đồ | [Cell 65, `In [38]`][cell-65]; [Cell 66, sơ đồ đã render][cell-66] |
| Experiment runner và live-monitor callback | [Cell 67, `In [39]`][cell-67] |
| Controlled experiment configs | [Cell 69, `In [40]`][cell-69] |
| Controlled runs + Stage A + search runner | [Cell 70][cell-70] |
| Controlled summary + Stage B | [Cell 72][cell-72] |
| Stage C và A/B/C comparison | [Cell 73][cell-73] |
| Stage D, final selection và exports | [Cell 74][cell-74] |
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

### Live training monitor

`run_training(...)` nhận thêm callback tùy chọn `monitor`. Sau khi epoch metrics,
best-checkpoint decision và TensorBoard logging hoàn tất, hàm gửi chính
`epoch_record` đó tới monitor. Thứ tự này bảo đảm dashboard chỉ hiển thị epoch đã
hoàn thành và dùng cùng nguồn số liệu với checkpoint/history, không tính lại
metric riêng.

Module
[`processing_own_phase/training_monitor.py`](../../processing_own_phase/training_monitor.py)
cung cấp `TrainingMonitor` với hai panel cập nhật theo epoch:

- cross-entropy loss;
- accuracy theo phần trăm.

Experiment monitor hiển thị cả train/validation và đánh dấu best epoch bằng
đường dọc nét đứt. Final-training monitor chỉ hiển thị train vì validation đã
được đưa lại vào full training pool. Monitor dùng một IPython `display_id` để cập
nhật cùng một dashboard thay vì tạo một output mới ở mỗi epoch; ngoài notebook,
nó fallback về Matplotlib `Agg`. Context manager luôn đóng figure khi run kết
thúc hoặc raise exception.

Đây là giám sát ở **epoch level**. Monitor không chạy trong batch loop, không giữ
model/optimizer/DataLoader và không thay đổi random state, nên không can thiệp
gradient updates hay model-selection protocol.

### Anti-fail recovery checkpoint

Module
[`processing_own_phase/training_checkpoint.py`](../../processing_own_phase/training_checkpoint.py)
quản lý recovery checkpoint độc lập với best-model artifact. Sau khi epoch đã
hoàn thành đủ train, validation, history/best-state update và TensorBoard flush,
training loop atomic-save trạng thái tại:

```text
outputs/recovery/<experiment_id>.resume.pth
```

Mỗi file chứa current model, optimizer, completed epoch, history, best state,
Python/NumPy/Torch CPU/CUDA/MPS RNG, DataLoader generator state, elapsed time và
log directory. Nếu crash giữa một epoch, file của epoch hoàn thành trước đó vẫn
nguyên vẹn; epoch đang dở được chạy lại từ đầu.

`TrainingCheckpointManager` ghi vào file tạm cùng directory, `fsync`, rồi
`os.replace`, và load bằng `weights_only=True`. Schema version cùng SHA-256
signature của config, split indices, normalization, device và library versions
ngăn resume nhầm checkpoint của một run không tương thích. Corrupted hoặc
mismatched checkpoint làm notebook dừng rõ ràng thay vì silently overwrite.

Khi resume, model/optimizer/RNG/DataLoader state được restore trước epoch kế
tiếp. Monitor render history cũ một lần; TensorBoard tiếp tục log directory cũ và
dùng `purge_step` để không tạo step trùng.

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
history, parameter/sample count, elapsed time và recovery provenance. Artifact
được atomic-save sau khi `run_training(...)` hoàn tất. TensorBoard event file
được tách theo experiment ID trong cùng run session.

`ENABLE_LIVE_TRAINING_PLOTS = True` bật dashboard cho từng experiment. Cell tạo
một `TrainingMonitor` mới bên trong vòng lặp, truyền nó vào `run_training(...)`
và quản lý bằng `with`, vì vậy history/display state không bị chia sẻ giữa E0-E4.
Có thể đặt flag thành `False` để giữ nguyên training và TensorBoard nhưng bỏ phần
render realtime.

`ENABLE_RECOVERY_CHECKPOINTS = True` bật checkpoint sau từng epoch;
`RESUME_IF_AVAILABLE = True` cho phép tự động tiếp tục từ file tương thích. Vì
recovery directory không chứa timestamp, một kernel/notebook session mới vẫn tìm
được đúng run đang dở. Checkpoint có trạng thái `completed` cho phép dựng lại
result và bỏ qua training đã hoàn tất.

Sau khi cả năm run hoàn tất, `max(...)` giữ lại baseline đối chiếu theo tuple:

```text
(best_validation_accuracy, -best_validation_loss)
```

### Staged search sau controlled baseline

Source hiện tại tiếp tục với bốn stage:

1. Stage A chạy bốn learning rate trên anchor `(256, 128)` trong 20 epoch.
2. Stage B chạy năm hidden architecture với learning rate thắng trong 25 epoch.
3. Stage C chạy hai geometric neighbors quanh coarse best trong 25 epoch; trial
   trùng Stage B bị loại bằng seed-independent `candidate_key`.
4. Stage D chạy top hai candidate qua seed `42`, `123`, `2026` trong 40 epoch.

Mỗi trial dùng `outputs/recovery/hyperparameter_search/<trial-id>.resume.pth`,
TensorBoard directory riêng, live PNG riêng và atomic best-model artifact riêng.
Official test loader không xuất hiện trong source Cell 69-74.

Confirmation chọn theo mean validation accuracy rồi mean loss, standard
deviation, parameter count và candidate key. Final epoch count dùng median best
epoch của candidate thắng; không lấy best seed đơn lẻ làm estimator cuối.

## 11. Kết quả năm experiment hiện tại

[Mở Experiment Results và output tại đúng Cell 72][cell-72].

| Experiment | Best val accuracy | Best val loss | Best epoch | Parameters | Runtime |
|---|---:|---:|---:|---:|---:|
| E0_baseline | `88.43%` | `0.3265` | 6 | 101,770 | 43.5 s |
| E1_deeper | `89.15%` | `0.3177` | 7 | 235,146 | 46.9 s |
| E2_dropout | `88.72%` | `0.3279` | 8 | 235,146 | 45.8 s |
| E3_sgd | `88.93%` | `0.3183` | 10 | 235,146 | 42.0 s |
| E4_augmentation | `87.82%` | `0.3366` | 10 | 235,146 | 59.3 s |

Theo primary metric, `E1_deeper` được chọn với:

- best epoch: `7`;
- best validation accuracy: `89.15%`;
- best validation loss: `0.3177`.

Dropout và augmentation ở đúng cấu hình đang thử không cải thiện accuracy so với
E1. SGD đạt gần E1 nhưng vẫn thấp hơn 0.22 percentage point trong lần chạy này.

Các runtime controlled trên thuộc fresh run `20260811-002044`. Trong staged
search run `20260811-010514`, E0-E4 được restore từ completed checkpoints rồi 17
search trials và final retraining được thực thi mới.

Kết quả Stage D:

| Candidate | Mean val accuracy | Std | Mean val loss | Median best epoch |
|---|---:|---:|---:|---:|
| `(512,256,128)`, lr `0.001` | `89.7833%` | `0.0816%` | `0.5520` | 32 |
| `(256,128)`, lr `0.001` | `89.6222%` | `0.0864%` | `0.4962` | 25 |

Candidate rộng hơn thắng primary metric với margin khoảng `0.1611` percentage
point. Loss tăng trong khi accuracy bão hòa ở các epoch muộn là dấu hiệu
overfitting cần được nêu cùng kết quả accuracy.

## 12. Staged-search curves và comparison

[Mở learning curves tại Cell 73][cell-73] và
[experiment comparison tại Cell 74][cell-74].

Cell 73 lưu comparison accuracy của Stage A/B/C. Cell 74 lưu confirmation mean ±
standard deviation và loss/accuracy history của representative run thuộc
candidate thắng. Ngoài aggregate charts, monitor của mỗi trial lưu PNG history
riêng với adaptive ticks dưới `outputs/hyperparameter_search/plots/`.

## 13. Final training sau model selection

[Mở final-training function tại Cell 76][cell-76] và
[final-training run/output tại Cell 77][cell-77].

Sau multi-seed confirmation khóa config và median best epoch, notebook dựng một
model mới hoàn toàn và train trên toàn bộ 60,000 official training images.

Quy trình:

1. Copy `selected_config` từ Stage D.
2. Thêm hậu tố `_final` vào selected candidate ID.
3. Đặt số epoch bằng confirmation median best epoch.
4. Chọn full `baseline_training_pool` vì selected config không dùng augmentation.
5. Khởi tạo fresh model/optimizer/loader.
6. Train đủ 32 epoch, không chạy validation và không chọn lại checkpoint.
7. Cập nhật dashboard train-only sau mỗi epoch rồi đóng figure khi hoàn tất.
8. Atomic-save train-only recovery state sau mỗi completed epoch.

Việc đưa 6,000 validation images trở lại training chỉ diễn ra **sau** khi
architecture, optimizer và epoch count đã được quyết định. Nhờ vậy final model có
thể học từ toàn bộ 60,000 labeled development images mà không dùng official test.

Stored output hiện tại của final selected model:

| Epoch | Train loss | Train accuracy |
|---:|---:|---:|
| 1 | `0.4703` | `82.83%` |
| 8 | `0.2227` | `91.55%` |
| 16 | `0.1488` | `94.25%` |
| 24 | `0.1090` | `95.86%` |
| 32 | `0.0819` | `96.82%` |

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
         flush TensorBoard
         atomic-save recovery checkpoint
         cập nhật live dashboard
    -> Stage A: learning-rate coarse search
    -> Stage B: hidden architecture search
    -> Stage C: local learning-rate refinement
    -> Stage D: top-2 x 3-seed confirmation
    -> chọn config bằng aggregate validation metrics
    -> lấy median best epoch
    -> fresh final selected model
    -> train trên toàn bộ 60,000
    -> atomic-save final recovery state theo epoch
    -> cập nhật live train-only dashboard theo epoch
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
