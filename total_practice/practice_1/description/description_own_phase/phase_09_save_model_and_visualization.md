# Phase 9 - Save Model & Visualization

[Phase 8](phase_08_model_evaluation.md) | [Mục lục](README.md) | [Mở đúng Cell 88][cell-88] | [Notebook dự phòng](../../practice_1.ipynb)

## 1. Vị trí và phạm vi

- Notebook cells: `88` đến `91`.
- Code cells đã chạy: `In [55]` đến `In [57]`.
- Input: final trained model, selected/final configs, validation/test metrics,
  normalization statistics và class names.
- Output: portable checkpoint, reconstructed model, equality verification và
  predicted-versus-actual image grid.

| Nội dung | Đích chính xác trong notebook |
|---|---|
| Heading Phase 9 | [Cell 88][cell-88] |
| Tạo và lưu final checkpoint | [Cell 89, `In [55]` + output][cell-89] |
| Load, reconstruct và equivalence verification | [Cell 90, `In [56]` + output][cell-90] |
| Predicted-versus-actual grid | [Cell 91, `In [57]` + output][cell-91] |

## 2. Tạo checkpoint path

[Mở checkpoint-save code và output tại đúng Cell 89][cell-89].

Notebook bảo đảm `OUTPUT_DIR` tồn tại bằng:

```python
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
```

và đặt checkpoint tại:

```text
outputs/fashion_mnist_model.pth
```

Output hiện được lưu trong notebook in absolute path tương ứng dưới project
`total_practice/practice_1/outputs`.

## 3. Nội dung checkpoint

Checkpoint không chỉ lưu weights. Nó lưu đủ metadata để hiểu và reconstruct
pipeline:

| Key | Nội dung |
|---|---|
| `model_state_dict` | Weight và bias tensors của final model |
| `model_config` | `hidden_dims`, `dropout`, `num_classes`, `input_dim` |
| `training_config` | Toàn bộ final experiment config |
| `selected_experiment` | ID experiment thắng validation |
| `best_epoch` | Epoch được chọn từ internal validation |
| `best_validation_accuracy` | Primary selection metric |
| `best_validation_loss` | Tie-break validation metric |
| `test_accuracy` | Official test accuracy |
| `test_loss` | Official test cross-entropy |
| `train_mean` | Mean fit từ internal train |
| `train_std` | Standard deviation fit từ internal train |
| `class_names` | Mapping output index sang tên lớp |
| `final_training_history` | Final train metrics của mọi epoch |
| `final_training_total_seconds` | Runtime tích lũy qua recovery sessions |
| `final_training_resumed_from_epoch` | Epoch được khôi phục, hoặc 0 nếu fresh |
| `recovery_checkpoint_path` | Đường dẫn anti-fail checkpoint nguồn |

Mỗi state tensor được `detach()`, chuyển về CPU và `clone()` trước khi lưu. Điều
này tách checkpoint khỏi autograd graph, khỏi mutable parameter hiện tại và khỏi
device MPS, giúp file dễ nạp trên device khác hơn.

`atomic_torch_save(checkpoint, MODEL_PATH)` ghi file tạm hoàn chỉnh, `fsync`, rồi
`os.replace` sang `MODEL_PATH`. Vì vậy final checkpoint cũ không bị biến thành
file dở nếu kernel/process lỗi đúng lúc đang ghi.

## 4. Vì sao cần lưu config và preprocessing metadata

`state_dict` chỉ là tập tensor theo tên parameter. Để sử dụng lại, code cần biết:

- phải dựng bao nhiêu hidden layer và kích thước từng layer;
- output có bao nhiêu class;
- input phải flatten thành bao nhiêu feature;
- ảnh phải normalize bằng mean/std nào;
- output index tương ứng với class name nào.

Nếu thiếu các metadata này, weights có thể không load được vào architecture hoặc
inference dùng preprocessing khác training, dẫn đến prediction sai dù file weight
không hỏng.

## 5. Nạp checkpoint an toàn theo device

[Mở load/reconstruct code và output tại đúng Cell 90][cell-90].

Notebook nạp bằng:

```text
torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True,
)
```

- `map_location=device` ánh xạ tensor tới selected runtime device.
- `weights_only=True` giới hạn unpickling theo chế độ phù hợp cho checkpoint chứa
  tensor và primitive metadata.

Sau đó notebook:

1. Dựng `FashionMNISTModel(**loaded_checkpoint['model_config'])` mới.
2. Chuyển model mới sang `device`.
3. Nạp `model_state_dict`.
4. Chuyển model sang `eval()`.

Đây là kiểm tra thực tế rằng metadata architecture đủ để reconstruct model chứ
không chỉ chứng minh file được ghi thành công.

## 6. Kiểm chứng equivalence sau save/load

Notebook lấy batch đầu từ deterministic `test_loader`, chuyển images sang device
và chạy inference bằng cả:

- final model còn trong memory;
- `loaded_model` vừa reconstruct từ disk.

Hai mức kiểm tra được thực hiện:

| Kiểm tra | Cách tính |
|---|---|
| Prediction equality | So sánh `argmax` của hai logit tensors bằng `torch.equal` |
| Logit equality gần đúng | Dùng `torch.allclose(original_logits, loaded_logits)` |

Notebook còn tính maximum absolute logit difference. Output hiện tại:

```text
Predictions match: True
Maximum logit difference: 0.00000000
```

Hai assertion sẽ dừng cell nếu prediction khác hoặc logits không all-close. Sai
khác bằng 0 trong lần chạy hiện tại xác nhận save/load giữ nguyên hàm inference
trên verification batch.

## 7. Chuẩn bị ảnh để hiển thị

Batch trong `test_loader` đã normalize. Để hiển thị đúng cường độ grayscale,
notebook đảo normalization:

```text
display_images = verification_images * TRAIN_STD + TRAIN_MEAN
```

sau đó clamp về `[0, 1]`. Nếu hiển thị trực tiếp tensor normalized có range khoảng
`[-0.810, 2.022]`, colormap vẫn có thể rescale nhưng ảnh không còn thể hiện đúng
miền input trực quan ban đầu.

Prediction dùng chính `loaded_model`, vì mục tiêu của phase là chứng minh artifact
đã lưu có thể phục vụ inference, không phụ thuộc model object trước save.

## 8. Predicted-versus-actual grid

[Mở prediction-grid code và output tại đúng Cell 91][cell-91].

Notebook tạo grid `4 x 4`, hiển thị 16 ảnh đầu của verification batch. Mỗi subplot
có:

- grayscale image;
- tên class model dự đoán;
- tên class thật;
- title màu xanh khi đúng;
- title màu đỏ khi sai;
- trục được ẩn để tập trung vào ảnh và nhãn.

Figure title là `Predicted and Actual FashionMNIST Labels`. Grid đáp ứng phần
deliverable yêu cầu hiển thị predicted-versus-actual images và giúp quan sát lỗi ở
mức từng sample, bổ sung cho metric aggregate của Phase 8.

Cell hiện tại chỉ gọi `plt.show()`; nó không gọi `savefig` trong chính source của
Phase 9.

## 9. Phân biệt ba loại artifact

Pipeline tạo ba nhóm artifact khác nhau:

| Artifact | Nội dung | Mục đích |
|---|---|---|
| TensorBoard logs | Metric theo epoch | Theo dõi và so sánh training |
| Experiment checkpoints | Best validation model/history của E0-E4 | Audit model selection |
| Recovery checkpoints | Model/optimizer/RNG state của completed epoch | Resume sau interruption |
| Final checkpoint | Final weights + config + metrics + preprocessing | Reconstruct và inference |

Phase 9 tập trung vào final checkpoint, còn per-experiment checkpoint đã được lưu
trong Phase 7.

## 10. Quy trình inference từ checkpoint

Metadata trong file cho phép một inference pipeline về mặt logic như sau:

```text
đọc checkpoint
    -> dựng model từ model_config
    -> load model_state_dict
    -> model.eval()
    -> biến ảnh thành [1, 28, 28] float32 trong [0, 1]
    -> normalize bằng train_mean/train_std
    -> thêm batch dimension
    -> forward raw logits
    -> argmax thành class index
    -> map index qua class_names
```

Notebook hiện minh họa quy trình trên bằng test DataLoader thay vì xây API nhận
ảnh bên ngoài.

## 11. Flow hoàn chỉnh của phase

```text
final model + configs + metrics + normalization + class names
    -> clone weights về CPU
    -> atomic-save final checkpoint
    -> torch.load với map_location và weights_only
    -> reconstruct FashionMNISTModel
    -> load_state_dict + eval mode
    -> chạy original và loaded model trên cùng batch
    -> assert predictions/logits tương đương
    -> inverse-normalize 16 images
    -> hiển thị predicted vs actual grid
```

## 12. Điều kiện hoàn thành toàn project

Sau Phase 9, notebook đã bao phủ đầy đủ yêu cầu bài tập:

1. Load FashionMNIST và transforms.
2. EDA và preprocessing có ranh giới dữ liệu rõ ràng.
3. Build neural network bằng PyTorch.
4. Train bằng forward, loss, backward và optimizer step.
5. Validate và chạy controlled hyperparameter/model experiments.
6. Vẽ loss/accuracy và so sánh experiment.
7. Evaluate bằng accuracy, report và confusion matrix.
8. Save/load model cùng metadata.
9. Kiểm chứng loaded model.
10. Hiển thị predicted-versus-actual images.

Quay lại [mục lục tài liệu](README.md) để xem bản đồ toàn bộ pipeline hoặc
[mở đúng title Cell 0][cell-0]. Trình Markdown ngoài VS Code có thể dùng
[notebook dự phòng](../../practice_1.ipynb).

[cell-0]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=0>
[cell-88]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=88>
[cell-89]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=89>
[cell-90]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=90>
[cell-91]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=91>
