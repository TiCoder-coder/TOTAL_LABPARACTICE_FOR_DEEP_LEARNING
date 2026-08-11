# Phase 6 - Model Building

[Phase 5](phase_05_data_preprocessing.md) | [Mục lục](README.md) | [Mở đúng Cell 55][cell-55] | [Notebook dự phòng](../../practice_1.ipynb) | [Phase 7](phase_07_model_training.md)

## 1. Vị trí và phạm vi

- Notebook cells: `55` đến `59`.
- Code cells đã chạy: `In [32]` đến `In [34]`.
- Input: normalized batch `[B, 1, 28, 28]`, labels `[B]` và cấu hình model.
- Output: class `FashionMNISTModel`, model factory, baseline model, parameter count
  và kết quả fail-fast sanity check.

| Nội dung | Đích chính xác trong notebook |
|---|---|
| Heading Phase 6 | [Cell 55][cell-55] |
| Model-building diagram | [Cell 56, sơ đồ đã render][cell-56] |
| `FashionMNISTModel` implementation | [Cell 57, `In [32]`][cell-57] |
| Baseline config, factory, model và parameter-count output | [Cell 58, `In [33]` + output][cell-58] |
| Forward/backward/optimizer sanity check | [Cell 59, `In [34]` + output][cell-59] |

## 2. Lựa chọn baseline architecture

Notebook dùng configurable multi-layer perceptron (MLP). Luồng tensor tổng quát:

```text
[B, 1, 28, 28]
    -> Flatten
[B, 784]
    -> Linear/ReLU/(Dropout) cho từng hidden dimension
    -> Linear cuối
[B, 10] raw logits
```

MLP là baseline phù hợp cho bài tập vì nó minh họa rõ các khái niệm tensor,
module, forward pass, autograd và optimizer mà không che luồng học phía sau một
kiến trúc phức tạp. Class vẫn cho phép thay đổi số hidden layer, chiều layer và
dropout để chạy controlled experiments.

Diagram của model nằm ngay dưới heading Phase 6 trong notebook.

[Mở model-building diagram tại Cell 56][cell-56].

## 3. Constructor của `FashionMNISTModel`

[Mở class implementation tại đúng Cell 57][cell-57].

Class kế thừa `nn.Module` và nhận bốn tham số:

| Tham số | Mặc định | Ý nghĩa |
|---|---:|---|
| `hidden_dims` | `(128,)` nếu `None` | Kích thước tuần tự của hidden layers |
| `dropout` | `0.0` | Xác suất dropout sau mỗi ReLU |
| `num_classes` | `10` | Số logit đầu ra |
| `input_dim` | `784` | Số feature sau flatten |

`hidden_dims` được đổi sang tuple để config có representation ổn định. Với mỗi
hidden dimension, class nối:

```text
Linear(in_features, hidden_dim)
ReLU()
Dropout(dropout) chỉ khi dropout > 0
```

Cuối cùng class thêm `Linear(in_features, num_classes)` và gói toàn bộ layer trong
`nn.Sequential`.

Method `forward(inputs)` chỉ gọi `self.network(inputs)`, giữ data flow trực tiếp
và phù hợp với architecture tuần tự.

## 4. Constructor validation

Model chủ động raise `ValueError` khi:

- `input_dim <= 0`;
- `num_classes <= 1`;
- bất kỳ hidden dimension nào không dương;
- dropout nằm ngoài interval `[0, 1)`.

Những check này bắt lỗi config ngay lúc dựng model, trước khi training chạy lâu
rồi mới thất bại do shape hoặc layer không hợp lệ.

## 5. Vì sao output không có Softmax

Layer cuối trả raw logits. Notebook cố ý không thêm `Softmax` vì training dùng:

```python
nn.CrossEntropyLoss()
```

Cross-entropy của PyTorch nhận raw logits và kết hợp log-softmax với negative
log-likelihood theo cách ổn định số học. Thêm Softmax trước loss sẽ thay đổi đầu
vào mà loss mong đợi và có thể làm gradient kém ổn định.

Khi cần class prediction, Phase 7-9 chỉ lấy:

```text
argmax(logits, dim=1)
```

Argmax của logits và argmax của softmax probabilities giống nhau, nên không cần
tính softmax để đo accuracy.

## 6. Baseline configuration

[Mở config, factory và model output tại đúng Cell 58][cell-58].

`BASELINE_CONFIG` định nghĩa E0:

| Thuộc tính | Giá trị |
|---|---|
| Experiment | `E0_baseline` |
| Hidden dimensions | `(128,)` |
| Dropout | `0.0` |
| Input dimension | `784` |
| Number of classes | `10` |
| Optimizer | Adam |
| Learning rate | `0.001` |
| Weight decay | `0.0` |
| Batch size | `64` |
| Epochs | `10` |
| Augmentation | `False` |

Model config và training config được đặt chung trong dictionary để Phase 7 có thể
clone/ghi đè từng yếu tố cho controlled experiments.

## 7. Factory và parameter counting

`build_model(config)` chỉ lấy các field architecture rồi tạo một
`FashionMNISTModel` mới. Việc gọi factory cho mỗi experiment bảo đảm mỗi run bắt
đầu bằng model object và random initialization mới.

`count_trainable_parameters(model)` cộng `numel()` của mọi parameter có
`requires_grad=True`.

Baseline có:

```text
Linear 784 -> 128: 784 * 128 weights + 128 biases = 100,480
Linear 128 -> 10:  128 * 10 weights  + 10 biases  =   1,290
Total                                               = 101,770
```

Output notebook xác nhận model structure và `101,770` trainable parameters.

## 8. Fail-fast sanity check

[Mở sanity-check code và output tại đúng Cell 59][cell-59].

`sanity_check_model(...)` không train trực tiếp object baseline đang được giữ cho
experiment. Nó dựng một model mới có cùng architecture rồi dùng 8 sample đầu của
batch train để kiểm tra end-to-end một optimization step.

### Tensor contract

Hàm assert:

- images có shape `(8, 1, 28, 28)`;
- image dtype/device khớp parameter đầu tiên của model;
- labels có shape `(8,)` và dtype `torch.int64`.

Ở thời điểm Phase 6, device training chưa được chọn, nên sanity model và batch
đang ở CPU. Phase 7 mới chuyển model/batch sang CUDA, MPS hoặc CPU.

### Forward và loss

Hàm chạy forward và assert:

- logits có shape `(8, 10)`;
- mọi logit finite;
- `CrossEntropyLoss` finite.

### Backward và optimizer step

Hàm chụp snapshot parameter, gọi:

```text
zero_grad(set_to_none=True)
    -> loss.backward()
    -> kiểm tra mọi gradient tồn tại và finite
    -> kiểm tra có ít nhất một gradient khác 0
    -> SGD step với learning rate 0.01
    -> xác nhận ít nhất một parameter đã thay đổi
```

Đây là test nhỏ nhưng bao phủ đúng chuỗi forward, loss, autograd và optimize mà
bài tập yêu cầu.

## 9. Output sanity hiện tại

| Kiểm tra | Output |
|---|---|
| Logit shape | `(8, 10)` |
| Finite loss | `2.219638` |
| Gradients finite | `True` |
| `parameters_changed` | `True` |
| Baseline parameter count | `101,770` |

Loss sanity phụ thuộc random initialization và chỉ dùng để xác nhận finiteness;
nó không phải training result hoặc metric dùng chọn model.

## 10. Điều architecture giữ cố định và cho phép thay đổi

Giữ cố định:

- input là 784 normalized pixels;
- activation là ReLU;
- output là 10 logits;
- objective sau này là cross-entropy.

Cho phép Phase 7 thay đổi:

- controlled experiments: baseline `(128,)` và anchor `(256, 128)`;
- staged architecture search: `(256,)`, `(256, 128)`, `(256, 128, 64)`,
  `(256, 128, 64, 32)` và `(512, 256, 128)`;
- learning rate theo các mức coarse rồi refinement quanh mức tốt nhất;
- dropout `0.0` hoặc `0.2` trong controlled experiments;
- optimizer Adam hoặc SGD trong controlled experiments;
- có/không augmentation ở input pipeline.

Class không hard-code số hidden layer, nên cùng một factory phục vụ cả controlled
experiments lẫn hyperparameter search mà vẫn giữ nguyên hợp đồng input/output.

## 11. Giới hạn có chủ đích của baseline

Flatten làm mất cấu trúc không gian tường minh: pixel gần nhau và xa nhau đều trở
thành feature trong một vector. MLP vì vậy không tận dụng locality/weight sharing
như CNN. Notebook giữ thiết kế này để hoàn thành một deep-learning baseline rõ
ràng; các cải tiến CNN có thể là experiment sau, không phải hành vi hiện tại cần
được mô tả như đã có.

## 12. Flow hoàn chỉnh của phase

```text
định nghĩa configurable FashionMNISTModel
    -> validate architecture config
    -> build BASELINE_CONFIG model
    -> đếm 101,770 trainable parameters
    -> lấy 8 normalized train samples
    -> forward thành [8, 10] logits
    -> cross-entropy
    -> backward
    -> SGD sanity step
    -> xác nhận gradient finite và parameter thay đổi
```

## 13. Đầu ra cho phase sau

[Phase 7](phase_07_model_training.md) tái sử dụng:

- `FashionMNISTModel`;
- `build_model(config)`;
- `count_trainable_parameters(model)`;
- cấu trúc config của E0;
- hợp đồng input/output đã được sanity check.

Từ đó mỗi experiment có model/optimizer mới, train trên 54,000 sample và được chọn
chỉ bằng validation metrics.

[cell-55]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=55>
[cell-56]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=56>
[cell-57]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=57>
[cell-58]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=58>
[cell-59]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=59>
