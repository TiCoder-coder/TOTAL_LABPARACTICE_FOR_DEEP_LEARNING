# Phase 8 - Model Evaluation

[Phase 7](phase_07_model_training.md) | [Mục lục](README.md) | [Mở đúng Cell 79][cell-79] | [Notebook dự phòng](../../practice_1.ipynb) | [Phase 9](phase_09_save_model_and_visualization.md)

## 1. Vị trí và phạm vi

- Notebook cells: `79` đến `84`.
- Code cells đã chạy: `In [48]` đến `In [51]`.
- Input: final model train trên 60,000 ảnh, deterministic `test_loader`, selected
  device và 10 class names.
- Output: official test loss/accuracy, prediction/target arrays, classification
  report, phân tích kết quả và confusion matrix.

| Nội dung | Đích chính xác trong notebook |
|---|---|
| Heading Phase 8 | [Cell 79][cell-79] |
| `evaluate_classifier` implementation | [Cell 80, `In [48]`][cell-80] |
| Official test loss/accuracy | [Cell 81, `In [49]` + output][cell-81] |
| Classification report | [Cell 82, `In [50]` + output][cell-82] |
| Result Analysis | [Cell 83][cell-83] |
| Confusion matrix | [Cell 84, `In [51]` + output][cell-84] |

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
4. Lấy prediction bằng `argmax(dim=1)`.
5. Cộng `loss.item() * batch_size`.
6. Đếm số prediction đúng.
7. Chuyển predictions và targets về CPU để lưu cho scikit-learn.

Sau loop, hàm kiểm tra loader không rỗng rồi trả:

| Field | Ý nghĩa |
|---|---|
| `loss` | Sample-weighted mean cross-entropy trên test |
| `accuracy` | Tổng prediction đúng chia 10,000 |
| `sample_count` | Tổng sample đã evaluate |
| `predictions` | NumPy array 10,000 predicted labels |
| `targets` | NumPy array 10,000 ground-truth labels |

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
| Test loss | `0.3294` |
| Test accuracy | `88.79%` |

Accuracy tương đương 8,879 prediction đúng trong 10,000 ảnh ở lần chạy đang lưu.
Test loss phản ánh cả độ đúng và độ tự tin tương đối của logits, trong khi accuracy
chỉ quan tâm class có logit lớn nhất.

## 5. Classification report

[Mở classification report và output tại đúng Cell 82][cell-82].

Notebook gọi `sklearn.metrics.classification_report` với:

- targets thật;
- predictions của model;
- tên 10 class;
- bốn chữ số thập phân;
- `zero_division=0` để định nghĩa hành vi an toàn nếu một class không có
  prediction.

Kết quả theo lớp:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| T-shirt/top | 0.8247 | 0.8610 | 0.8425 | 1,000 |
| Trouser | 0.9928 | 0.9670 | 0.9797 | 1,000 |
| Pullover | 0.7874 | 0.8480 | 0.8166 | 1,000 |
| Dress | 0.8815 | 0.9150 | 0.8979 | 1,000 |
| Coat | 0.8474 | 0.7720 | 0.8080 | 1,000 |
| Sandal | 0.9763 | 0.9490 | 0.9625 | 1,000 |
| Shirt | 0.7202 | 0.7000 | 0.7099 | 1,000 |
| Sneaker | 0.9039 | 0.9780 | 0.9395 | 1,000 |
| Bag | 0.9805 | 0.9560 | 0.9681 | 1,000 |
| Ankle boot | 0.9770 | 0.9330 | 0.9545 | 1,000 |

Aggregate report:

| Aggregate | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Macro average | 0.8892 | 0.8879 | 0.8879 |
| Weighted average | 0.8892 | 0.8879 | 0.8879 |

Macro và weighted average gần như giống nhau vì test set cân bằng, mỗi class có
đúng 1,000 sample.

### Cách đọc precision, recall và F1

- Precision của một class: trong mọi ảnh model dự đoán là class đó, tỷ lệ dự
  đoán đúng.
- Recall của một class: trong mọi ảnh thật thuộc class đó, tỷ lệ model tìm đúng.
- F1-score: harmonic mean của precision và recall, cao khi cả hai cùng tốt.
- Support: số ground-truth sample của class.

## 6. Result Analysis

[Mở Result Analysis tại đúng Cell 83][cell-83].

Markdown trong notebook chia class thành hai nhóm đáng chú ý.

### Nhóm mạnh

- Trouser: F1 `0.9797`.
- Bag: F1 `0.9681`.
- Sandal: F1 `0.9625`.
- Ankle boot cũng cao với F1 `0.9545`.

Các class này có silhouette khá riêng biệt ở độ phân giải thấp, nên MLP phân biệt
tốt hơn.

### Nhóm yếu

- Shirt: F1 `0.7099`.
- Coat: F1 `0.8080`.
- Pullover: F1 `0.8166`.

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

Cell `In [51]` gọi `confusion_matrix(targets, predictions)` để tạo ma trận `10 x
10`:

- hàng là actual label;
- cột là predicted label;
- đường chéo là prediction đúng;
- ô ngoài đường chéo là từng hướng nhầm lẫn cụ thể.

Seaborn heatmap hiển thị count nguyên trong từng ô, dùng cùng `class_names` cho
hai trục. Nhờ đó có thể phân biệt hai loại lỗi khác nhau, chẳng hạn actual Shirt
thành Coat và actual Coat thành Shirt.

Figure hiện được hiển thị inline qua `plt.show()`; cell hiện tại không gọi
`savefig` để ghi confusion matrix ra file.

## 8. Vì sao cần nhiều metric

Accuracy phù hợp để chọn model vì các class cân bằng, nhưng một con số tổng thể có
thể che class yếu. Report và confusion matrix bổ sung:

| Công cụ | Câu hỏi trả lời |
|---|---|
| Overall accuracy | Model đúng bao nhiêu phần trăm toàn test set? |
| Test loss | Logit distribution có phù hợp với nhãn đến mức nào? |
| Per-class precision | Prediction của từng class đáng tin đến đâu? |
| Per-class recall | Model tìm được bao nhiêu ảnh thật của từng class? |
| F1-score | Precision và recall cân bằng ra sao? |
| Confusion matrix | Class nào bị nhầm sang class nào? |

## 9. Flow hoàn chỉnh của phase

```text
final E1_deeper model + deterministic test_loader
    -> eval mode + inference mode
    -> forward toàn bộ 10,000 images
    -> weighted CE loss + overall accuracy
    -> concatenate predictions/targets về CPU
    -> classification report cho 10 class
    -> phân tích class mạnh/yếu
    -> confusion matrix
```

## 10. Đầu ra cho phase sau

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
