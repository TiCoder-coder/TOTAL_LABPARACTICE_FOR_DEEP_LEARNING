# Phase 8 - Model Evaluation

[Phase 7](phase_07_model_training.md) | [Mục lục](README.md) | [Mở đúng Cell 79][cell-79] | [Notebook dự phòng](../../practice_1.ipynb) | [Phase 9](phase_09_save_model_and_visualization.md)

## 1. Vị trí và phạm vi

- Notebook cells: `79` đến `87`.
- Cells 85-87 được thêm cho probability contract, ROC và Precision-Recall;
  output của ba cell này phải được tạo từ cùng lần chạy với evaluator mới.
- Input: final model train trên 60,000 ảnh, deterministic `test_loader`, selected
  device và 10 class names.
- Output: official test loss/accuracy, prediction/target/probability arrays,
  classification report, per-class one-vs-rest TN/TP/FN/FP/TPR/FPR, confusion
  matrix, ROC curves với AUC và Precision-Recall curves với Average Precision.

| Nội dung | Đích chính xác trong notebook |
|---|---|
| Heading Phase 8 | [Cell 79][cell-79] |
| `evaluate_classifier` implementation | [Cell 80, `In [48]`][cell-80] |
| Official test loss/accuracy | [Cell 81, `In [49]` + output][cell-81] |
| Per-class classification và one-vs-rest report | [Cell 82][cell-82] |
| Result Analysis | [Cell 83][cell-83] |
| Confusion matrix | [Cell 84, `In [51]` + output][cell-84] |
| One-vs-rest target/probability contract | [Cell 85][cell-85] |
| Per-class ROC curves và AUC | [Cell 86][cell-86] |
| Per-class Precision-Recall curves và AP | [Cell 87][cell-87] |

## 2. Ranh giới đánh giá

Phase này chạy sau khi:

- architecture đã được chọn;
- optimizer/hyperparameter đã được chọn;
- best epoch đã được chọn từ validation;
- final model đã train lại trên toàn bộ official training pool.

Không kết quả nào từ official test được đưa ngược trở lại để đổi config hoặc
training process. Vì vậy test metric đóng vai trò báo cáo cuối cùng, không phải
model-selection metric.

## 3. `evaluate_classifier`

[Mở evaluator implementation tại đúng Cell 80][cell-80].

Hàm bắt đầu bằng `model.eval()` và `torch.inference_mode()`, rồi lặp qua toàn bộ
`test_loader`.

Trong mỗi batch:

1. Chuyển image/label sang selected device.
2. Chạy forward để lấy raw logits.
3. Tính `CrossEntropyLoss` và kiểm tra finite.
4. Chuyển logits thành class probabilities bằng `softmax(dim=1)`.
5. Lấy prediction bằng `argmax(dim=1)`.
6. Cộng `loss.item() * batch_size` và đếm prediction đúng.
7. Chuyển predictions, targets và probabilities về CPU để lưu cho scikit-learn.

Sau loop, evaluator kiểm tra số sample, tensor shape, giá trị finite, miền xác
suất `[0, 1]`, tổng xác suất mỗi hàng bằng 1 và `argmax(probabilities)` khớp
prediction. Cross-entropy vẫn nhận raw logits; Softmax chỉ phục vụ metric curves.

Sau loop, hàm kiểm tra loader không rỗng rồi trả:

| Field | Ý nghĩa |
|---|---|
| `loss` | Sample-weighted mean cross-entropy trên test |
| `accuracy` | Tổng prediction đúng chia 10,000 |
| `sample_count` | Tổng sample đã evaluate |
| `predictions` | NumPy array 10,000 predicted labels |
| `targets` | NumPy array 10,000 ground-truth labels |
| `probabilities` | NumPy array `(10,000, 10)` chứa Softmax score theo class |

Toàn bộ prediction/target được lưu vì classification report và confusion matrix
cần kết quả theo từng sample, không chỉ aggregate accuracy.

## 4. Official test result

[Mở official-test metrics và output tại đúng Cell 81][cell-81].

Cell `In [49]` dùng `nn.CrossEntropyLoss`, gọi evaluator và assert số sample bằng
kích thước dataset.

Output hiện tại:

| Metric | Giá trị |
|---|---:|
| Test samples | `10,000` |
| Test loss | `0.5489` |
| Test accuracy | `89.42%` |

Accuracy tương đương 8,942 prediction đúng trong 10,000 ảnh ở lần chạy đang lưu.
Test loss phản ánh cả độ đúng và độ tự tin tương đối của logits, trong khi accuracy
chỉ quan tâm class có logit lớn nhất.

## 5. Classification report

[Mở classification report và output tại đúng Cell 82][cell-82].

Notebook gọi `sklearn.metrics.classification_report` với:

- targets thật;
- predictions của model;
- tên 10 class;
- `output_dict=True` để tái sử dụng metric trong bảng mở rộng;
- `zero_division=0` để định nghĩa hành vi an toàn nếu một class không có
  prediction.

Notebook tạo confusion matrix trong cùng cell và xem từng class theo chiến lược
one-vs-rest. Với class index `i`:

```text
TP = confusion[i, i]
FN = sum(confusion[i, :]) - TP
FP = sum(confusion[:, i]) - TP
TN = total_samples - TP - FN - FP
TPR = TP / (TP + FN)
FPR = FP / (FP + TN)
```

`TPR` bằng recall. `FPR` đo tỷ lệ sample thuộc chín class còn lại nhưng bị dự
đoán nhầm thành class đang xét; nó không bằng `1 - precision`.

Kết quả theo lớp:

Cell 82 hiển thị mỗi class với các cột:

| Nhóm | Cột |
|---|---|
| Classification | `precision`, `recall`, `f1-score`, `support` |
| Confusion counts | `tn`, `tp`, `fn`, `fp` |
| Rates | `tpr`, `fpr` |

Toàn bộ classification metric và confusion count được sinh từ cùng arrays
`targets` và `predictions`, nên bảng không ghép số liệu giữa các lần chạy.

Aggregate report:

| Aggregate | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Macro average | 0.8946 | 0.8942 | 0.8939 |
| Weighted average | 0.8946 | 0.8942 | 0.8939 |

Macro và weighted average gần như giống nhau vì test set cân bằng, mỗi class có
đúng 1,000 sample.

### Cách đọc precision, recall và F1

- Precision của một class: trong mọi ảnh model dự đoán là class đó, tỷ lệ dự
  đoán đúng.
- Recall của một class: trong mọi ảnh thật thuộc class đó, tỷ lệ model tìm đúng.
- F1-score: harmonic mean của precision và recall, cao khi cả hai cùng tốt.
- Support: số ground-truth sample của class.
- TP: ground-truth và prediction đều là class đang xét.
- FN: ground-truth là class đang xét nhưng prediction là class khác.
- FP: ground-truth là class khác nhưng prediction là class đang xét.
- TN: ground-truth và prediction đều không phải class đang xét.
- TPR: tỷ lệ positive ground-truth được phát hiện đúng, bằng recall.
- FPR: tỷ lệ negative ground-truth bị báo nhầm thành positive.

## 6. Result Analysis

[Mở Result Analysis tại đúng Cell 83][cell-83].

Markdown trong notebook chia class thành hai nhóm đáng chú ý.

### Nhóm mạnh

- Trouser: F1 `0.9848`.
- Bag: F1 `0.9688`.
- Sandal: F1 `0.9663`.
- Ankle boot cũng cao với F1 `0.9572`.

Các class này có silhouette khá riêng biệt ở độ phân giải thấp, nên MLP phân biệt
tốt hơn.

### Nhóm yếu

- Shirt: F1 `0.7193`.
- Coat: F1 `0.8265`.
- Pullover: F1 `0.8252`.

Shirt vừa có precision thấp (`0.7202`) vừa có recall thấp (`0.7000`): model bỏ sót
nhiều ảnh Shirt thật và đồng thời gán nhầm một số class khác thành Shirt.

Notebook liên hệ confusion của Shirt với T-shirt/top, Pullover và Coat. Đây đều là
upper-body garments có đường viền tương tự khi chỉ còn `28 x 28` pixel grayscale.
Kết quả này phù hợp với giả thuyết từ class mean images trong Phase 4.

Phần kết luận trong notebook diễn giải độ phân giải thấp là một bottleneck quan
trọng. Kết quả thực nghiệm trực tiếp chứng minh model hiện tại gặp khó ở nhóm
upper-body; nó không tự chứng minh rằng mọi architecture khác đều không thể cải
thiện nhóm này.

## 7. Confusion matrix

[Mở confusion-matrix code và output tại đúng Cell 84][cell-84].

Cell 84 tái sử dụng ma trận `10 x 10` đã được Cell 82 tạo từ
`confusion_matrix(targets, predictions)`:

- hàng là actual label;
- cột là predicted label;
- đường chéo là prediction đúng;
- ô ngoài đường chéo là từng hướng nhầm lẫn cụ thể.

Seaborn heatmap hiển thị count nguyên trong từng ô, dùng cùng `class_names` cho
hai trục. Nhờ đó có thể phân biệt hai loại lỗi khác nhau, chẳng hạn actual Shirt
thành Coat và actual Coat thành Shirt.

Figure hiện được hiển thị inline qua `plt.show()`; cell hiện tại không gọi
`savefig` để ghi confusion matrix ra file.

## 8. ROC curves theo one-vs-rest

[Cell 85][cell-85] chuyển targets thành ma trận nhị phân `(10,000, 10)` và xác
nhận probability array có cùng shape, hữu hạn, tổng hàng bằng 1 và giữ nguyên
argmax prediction. Mỗi cột biểu diễn một class là positive và chín class còn lại
là negative.

[Cell 86][cell-86] gọi `roc_curve` cho từng cặp binary target/probability column,
tính AUC bằng `auc(fpr, tpr)` và vẽ 10 subplot theo bố cục `2 x 5`. Mỗi subplot
có đường baseline ngẫu nhiên `TPR = FPR`, AUC trong legend và validation cho
range cùng endpoints `(0, 0)` đến `(1, 1)`.

ROC mô tả trade-off giữa TPR và FPR khi threshold thay đổi. Nó không được tính
từ hard predictions, vì hard predictions chỉ cung cấp một operating point.

## 9. Precision-Recall curves theo one-vs-rest

[Cell 87][cell-87] gọi `precision_recall_curve` cho từng class và dùng
`average_precision_score` làm scalar summary. Mỗi subplot hiển thị AP và baseline
bằng prevalence thật của class thay vì hardcode `0.1`.

PR curve tập trung trực tiếp vào precision/recall của positive class và bổ sung
góc nhìn khác ROC khi class distribution thay đổi. Với test set hiện tại mỗi
class có 1,000 positive trên 10,000 sample, nhưng code vẫn suy ra prevalence từ
ground truth.

## 10. Vì sao cần nhiều metric

Accuracy phù hợp để chọn model vì các class cân bằng, nhưng một con số tổng thể có
thể che class yếu. Report và confusion matrix bổ sung:

| Công cụ | Câu hỏi trả lời |
|---|---|
| Overall accuracy | Model đúng bao nhiêu phần trăm toàn test set? |
| Test loss | Logit distribution có phù hợp với nhãn đến mức nào? |
| Per-class precision | Prediction của từng class đáng tin đến đâu? |
| Per-class recall | Model tìm được bao nhiêu ảnh thật của từng class? |
| F1-score | Precision và recall cân bằng ra sao? |
| TN/TP/FN/FP | Ground-truth và prediction tạo loại kết quả nào cho từng class? |
| TPR | Bao nhiêu positive sample được tìm đúng? |
| FPR | Bao nhiêu negative sample bị báo nhầm? |
| Confusion matrix | Class nào bị nhầm sang class nào? |
| ROC/AUC | TPR/FPR thay đổi ra sao trên toàn bộ threshold? |
| Precision-Recall/AP | Precision/recall thay đổi ra sao theo threshold? |

## 11. Flow hoàn chỉnh của phase

```text
final selected `(512,256,128)` model + deterministic test_loader
    -> eval mode + inference mode
    -> forward toàn bộ 10,000 images
    -> weighted CE loss + overall accuracy
    -> concatenate predictions/targets/probabilities về CPU
    -> validate probability contract
    -> confusion matrix + one-vs-rest counts cho 10 class
    -> classification report + TPR/FPR
    -> phân tích class mạnh/yếu
    -> confusion-matrix heatmap dùng lại cùng matrix
    -> one-hot targets theo one-vs-rest
    -> 10 ROC curves + AUC
    -> 10 Precision-Recall curves + Average Precision
```

## 12. Đầu ra cho phase sau

[Phase 9](phase_09_save_model_and_visualization.md) đưa `test_accuracy`,
`test_loss`, model config, preprocessing statistics và model weights vào cùng một
checkpoint. Sau đó phase đó nạp lại checkpoint và kiểm chứng inference tương
đương trước khi hiển thị ảnh prediction.

[cell-79]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=79>
[cell-80]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=80>
[cell-81]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=81>
[cell-82]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=82>
[cell-83]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=83>
[cell-84]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=84>
[cell-85]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=85>
[cell-86]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=86>
[cell-87]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=87>
