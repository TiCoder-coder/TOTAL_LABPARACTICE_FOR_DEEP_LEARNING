# Tài liệu từng phase của `practice_1.ipynb`

Thư mục này giải thích tuần tự toàn bộ pipeline PyTorch FashionMNIST trong
`practice_1.ipynb`. Nội dung được viết theo đúng source code, markdown cell và
output hiện đang được lưu trong notebook.

## Cách sử dụng tài liệu

1. Đọc các file theo thứ tự từ Phase 1 đến Phase 9.
2. Bảo đảm workspace đang được Trust và cài helper extension đi kèm project một
   lần bằng lệnh:

   ```bash
   code --install-extension total_practice/practice_1/tools/vscode-notebook-links/practice1-notebook-links-0.1.0.vsix --force
   ```

3. Mở Markdown Preview trong VS Code rồi dùng liên kết **Mở đúng cell**. Helper
   nhận URI `vscode://`, mở notebook, chọn cell và đưa cell đó lên đầu viewport.
4. Với code cell đã có kết quả, cùng một liên kết sẽ mở cell chứa cả code và
   output. Với hình được nhúng bằng Markdown, tài liệu có liên kết riêng tới cell
   đang hiển thị hình.
5. Khi notebook được chạy lại, các số liệu thực nghiệm có thể thay đổi nhẹ theo
   thiết bị và phiên bản thư viện; thiết kế pipeline và ý nghĩa của từng bước vẫn
   giữ nguyên.

Helper được lưu tại
[tools/vscode-notebook-links](../../tools/vscode-notebook-links/README.md) và đã
được cài trong VS Code trên máy hiện tại. Liên kết
[Mở notebook dự phòng](../../practice_1.ipynb) chỉ mở file từ đầu và dành cho
máy chưa cài helper hoặc trình Markdown ngoài VS Code. Deep-link bám theo thứ tự
cell hiện tại; nếu chèn, xóa hoặc đổi thứ tự cell thì cần đồng bộ lại bản đồ liên
kết. Việc bổ sung deep-link không thay đổi source code, output, execution count
hay metadata của notebook.

## Bản đồ pipeline

| Phase | Phạm vi notebook | Vai trò chính | Tài liệu | Notebook |
|---|---:|---|---|---|
| 1 | Cell 1 | Định nghĩa bài toán và protocol đánh giá | [Phase 1](phase_01_problem_definition.md) | [Mở đúng Cell 1][cell-1] |
| 2 | Cell 2-5, `In [1]`-`In [3]` | Thiết lập đường dẫn, môi trường và thư viện | [Phase 2](phase_02_environment_setup.md) | [Mở đúng Cell 2][cell-2] |
| 3 | Cell 6-7, `In [4]` | Tải hai partition chính thức của FashionMNIST | [Phase 3](phase_03_data_loading.md) | [Mở đúng Cell 6][cell-6] |
| 4 | Cell 8-43, `In [5]`-`In [26]` | EDA trên official training pool | [Phase 4](phase_04_exploratory_data_analysis.md) | [Mở đúng Cell 8][cell-8] |
| 5 | Cell 44-54, `In [27]`-`In [31]` | Split, normalization, transform và DataLoader | [Phase 5](phase_05_data_preprocessing.md) | [Mở đúng Cell 44][cell-44] |
| 6 | Cell 55-59, `In [32]`-`In [34]` | Xây dựng và kiểm tra MLP | [Phase 6](phase_06_model_building.md) | [Mở đúng Cell 55][cell-55] |
| 7 | Cell 60-78, `In [35]`-`In [47]` | Train, validate, so sánh experiment và final training | [Phase 7](phase_07_model_training.md) | [Mở đúng Cell 60][cell-60] |
| 8 | Cell 79-87 | Official-test metrics, confusion matrix, ROC và Precision-Recall | [Phase 8](phase_08_model_evaluation.md) | [Mở đúng Cell 79][cell-79] |
| 9 | Cell 88-91, `In [55]`-`In [57]` | Save/load checkpoint và hiển thị dự đoán | [Phase 9](phase_09_save_model_and_visualization.md) | [Mở đúng Cell 88][cell-88] |

## Luồng dữ liệu tổng quát

```text
Định nghĩa bài toán
    -> kiểm tra môi trường
    -> tải 60,000 train-pool + 10,000 official-test
    -> EDA chỉ trên 60,000 train-pool
    -> stratified split 54,000 train + 6,000 validation
    -> tính normalization từ 54,000 train
    -> dựng MLP và sanity check
    -> train/validate 5 experiment độc lập
    -> staged search: learning rate -> architecture -> refinement
    -> multi-seed confirmation top 2
    -> chọn (512,256,128), Adam, lr=0.001
    -> train mới trên toàn bộ 60,000 ảnh trong 32 epoch
    -> đánh giá một lần trên 10,000 official-test
    -> lưu, nạp lại và kiểm chứng checkpoint
```

## Ranh giới dữ liệu quan trọng

| Tập dữ liệu | Kích thước | Được sử dụng ở đâu | Không được sử dụng để làm gì |
|---|---:|---|---|
| Official training pool | 60,000 | EDA; nguồn tạo train/validation; final training sau model selection | Không phải official test set |
| Internal training subset | 54,000 | Tính mean/std; tối ưu tham số; các experiment | Không dùng làm thước đo chọn model |
| Internal validation subset | 6,000 | Theo dõi mỗi epoch; chọn checkpoint và experiment | Không tham gia gradient update trong experiment |
| Official test set | 10,000 | Một lần đánh giá cuối cùng; report và confusion matrix | Không chọn hyperparameter, checkpoint hoặc epoch |

## Kết quả đang lưu trong notebook

- Thiết bị train: Apple Metal Performance Shaders (`mps`).
- Candidate được chọn: `(512,256,128)`, Adam, learning rate `0.001`.
- Multi-seed mean validation accuracy: `89.7833% ± 0.0816%`.
- Median best validation epoch: `32`.
- Final training: `60,000` ảnh trong `32` epoch.
- Official test loss: `0.5489`.
- Official test accuracy: `89.42%`.
- Save/load verification: prediction khớp và sai khác logit lớn nhất bằng
  `0.00000000`.

Các giá trị trên mô tả lần chạy hiện được nhúng trong notebook, không phải một
cam kết rằng mọi lần chạy lại sẽ cho kết quả giống tuyệt đối trên mọi máy.

[cell-1]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=1>
[cell-2]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=2>
[cell-6]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=6>
[cell-8]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=8>
[cell-44]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=44>
[cell-55]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=55>
[cell-60]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=60>
[cell-79]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=79>
[cell-88]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=88>
