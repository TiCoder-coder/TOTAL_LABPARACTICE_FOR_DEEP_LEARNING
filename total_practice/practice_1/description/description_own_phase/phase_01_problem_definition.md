# Phase 1 - Problem Definition

[Mục lục](README.md) | [Mở đúng Cell 1][cell-1] | [Notebook dự phòng](../../practice_1.ipynb) | [Phase 2](phase_02_environment_setup.md)

## 1. Vị trí và phạm vi

- Notebook cell: `1`.
- Loại cell: Markdown.
- Phase này chưa chạy code và chưa thay đổi dữ liệu.
- Mục đích là khóa chặt bài toán, đầu vào, đầu ra, metric và protocol đánh giá
  trước khi xây dựng pipeline.

## 2. Bài toán được định nghĩa

Notebook đặt mục tiêu xây dựng một bộ phân loại ảnh bằng PyTorch cho bộ dữ liệu
FashionMNIST. Mỗi ảnh chỉ có đúng một nhãn và nhãn thuộc một trong 10 lớp, do đó
đây là bài toán:

- học có giám sát, vì mỗi ảnh train đi kèm nhãn;
- phân loại đa lớp, vì có 10 lớp;
- single-label, vì một ảnh chỉ nhận một nhãn cuối cùng.

Mười lớp theo đúng thứ tự label của FashionMNIST là:

| Label | Tên lớp |
|---:|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

[Xem Objective tại đúng Cell 1][cell-1]

## 3. Hợp đồng tensor

Phase 1 xác định interface mà các phase sau phải tuân thủ:

| Thành phần | Hợp đồng |
|---|---|
| Input của model | Ảnh grayscale có shape `[1, 28, 28]` |
| Target | Một số nguyên trong đoạn `[0, 9]` |
| Output của model | Tensor gồm 10 raw logits, mỗi logit ứng với một lớp |
| Hàm loss | Cross-entropy loss |
| Metric chính | Classification accuracy |

Dimension đầu tiên của ảnh là channel. FashionMNIST là ảnh xám nên chỉ có một
channel. Model xuất raw logits thay vì xác suất vì `CrossEntropyLoss` của PyTorch
tự áp dụng phép biến đổi log-softmax phù hợp bên trong.

[Xem Task Formulation tại đúng Cell 1][cell-1]

## 4. Protocol chia và sử dụng dữ liệu

Notebook bắt đầu từ hai partition chính thức của FashionMNIST:

- official training pool: 60,000 ảnh;
- official test set: 10,000 ảnh.

Official training pool sau đó được chia đúng một lần bằng seed cố định và theo
từng lớp:

```text
60,000 official training images
    -> 54,000 internal training images (90%)
    ->  6,000 internal validation images (10%)

10,000 official test images
    -> giữ riêng cho lần đánh giá cuối cùng
```

Vai trò của từng tập:

| Tập | Vai trò |
|---|---|
| 54,000 train | Tính thống kê preprocessing, cập nhật trọng số và chạy experiment |
| 6,000 validation | Theo dõi mỗi epoch, chọn checkpoint và chọn experiment |
| 10,000 official test | Ước lượng khả năng tổng quát hóa sau khi mọi quyết định đã khóa |

EDA được phép đọc toàn bộ 60,000 ảnh và nhãn của official training pool vì EDA
chỉ mô tả dữ liệu, không fit tham số model. Tuy nhiên, mean/std dùng để normalize
phải được tính riêng từ 54,000 internal training images. Điều này giữ cho
validation không ảnh hưởng đến preprocessing đã học từ dữ liệu.

Validation và test chỉ dùng transform xác định. Random augmentation chỉ được áp
dụng lên train và chỉ trong experiment dành riêng cho augmentation.

[Xem Evaluation Protocol tại đúng Cell 1][cell-1]

## 5. Vì sao protocol này cần thiết

### Tách validation khỏi train

Nếu vừa tối ưu model vừa đánh giá trên cùng một tập, metric sẽ phản ánh khả năng
ghi nhớ training data nhiều hơn khả năng tổng quát hóa. Validation tạo một tín
hiệu độc lập tương đối để so sánh architecture và hyperparameter.

### Giữ official test đến cuối

Mỗi lần xem test result rồi điều chỉnh model đều làm thông tin test tham gia gián
tiếp vào model selection. Notebook quy định chỉ dùng test sau khi experiment,
checkpoint và số epoch đã được quyết định từ validation.

### Stratified split

FashionMNIST cân bằng sẵn, mỗi lớp có 6,000 ảnh trong official training pool.
Chia theo lớp bảo đảm internal train có chính xác 5,400 ảnh mỗi lớp và validation
có 600 ảnh mỗi lớp. Vì vậy khác biệt metric giữa các experiment không bị gây ra
bởi một split lệch phân phối lớp.

## 6. Tiêu chí hoàn thành bài tập

Phase 1 chuyển yêu cầu bài tập thành checklist kỹ thuật cho toàn notebook:

1. Tải và preprocess FashionMNIST.
2. Xây dựng neural network bằng PyTorch.
3. Train bằng forward pass, loss, backward pass và optimizer step.
4. So sánh các experiment bằng validation metrics.
5. Chỉ đánh giá model đã chọn trên official test set.
6. Trực quan hóa loss, accuracy và predicted-versus-actual samples.
7. Lưu model, nạp lại model và xác nhận kết quả không thay đổi.

[Xem Success Criteria tại đúng Cell 1][cell-1]

## 7. Đầu ra của phase

Phase này không sinh tensor hay artifact. Đầu ra là một specification dùng để
kiểm tra tất cả phase sau:

- Phase 3 phải tải đúng hai official partitions.
- Phase 4 không được đọc official test để ra quyết định.
- Phase 5 phải split đúng 54,000/6,000 và fit normalization trên train.
- Phase 7 chỉ được dùng validation để chọn model.
- Phase 8 chỉ chạy khi model selection đã kết thúc.
- Phase 9 phải chứng minh checkpoint có thể khôi phục model.

## 8. Liên kết sang phase tiếp theo

Sau khi bài toán và ranh giới dữ liệu đã rõ, [Phase 2](phase_02_environment_setup.md)
chuẩn hóa môi trường chạy, các thư mục artifact và toàn bộ dependency cần cho
pipeline.

[cell-1]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=1>
