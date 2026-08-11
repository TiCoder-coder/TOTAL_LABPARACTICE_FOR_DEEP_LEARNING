# Phase 8 - Model Evaluation Outputs

[Phase 7](phase_07_results.md) | [Mục lục](README.md) | [Phase 9](phase_09_results.md)

| Cell output | Mô tả ngắn | Liên kết |
|---|---|---|
| Cell 81, `In [49]` | Official test result: 10,000 samples, loss `0.5489`, accuracy `89.42%`. | [Mở output Cell 81][cell-81] |
| Cell 82, `In [50]` | Bảng per-class gồm precision, recall, F1-score, support, TN, TP, FN, FP, TPR và FPR. | [Mở output Cell 82][cell-82] |
| Cell 84, `In [51]` | Heatmap tái sử dụng confusion matrix `10 x 10` đã tính tại Cell 82. | [Mở output Cell 84][cell-84] |
| Cell 86, `In [53]` | Figure `2 x 5` gồm ROC curve và AUC cho từng class. | [Mở output Cell 86][cell-86] |
| Cell 87, `In [54]` | Figure `2 x 5` gồm Precision-Recall curve, AP và prevalence baseline cho từng class. | [Mở output Cell 87][cell-87] |

## Per-class curve metrics

Các giá trị dưới đây được tính lại từ probabilities của cùng checkpoint và
official test set tạo accuracy `0.894200`:

| Class | ROC AUC | Average Precision |
|---|---:|---:|
| T-shirt/top | 0.986166 | 0.908182 |
| Trouser | 0.998271 | 0.994828 |
| Pullover | 0.980237 | 0.881667 |
| Dress | 0.992550 | 0.947754 |
| Coat | 0.984214 | 0.880319 |
| Sandal | 0.998794 | 0.993775 |
| Shirt | 0.958145 | 0.786351 |
| Sneaker | 0.998750 | 0.989118 |
| Bag | 0.998590 | 0.991683 |
| Ankle boot | 0.998363 | 0.992136 |

[cell-81]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=81>
[cell-82]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=82>
[cell-84]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=84>
[cell-86]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=86>
[cell-87]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=87>
