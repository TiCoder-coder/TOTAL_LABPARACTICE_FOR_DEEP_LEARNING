# PyTorch FashionMNIST Classification

## 1. Tổng quan project

Project này xây dựng một pipeline Deep Learning hoàn chỉnh bằng PyTorch để phân loại ảnh FashionMNIST vào 10 nhóm trang phục. Notebook chính là `practice_1.ipynb`; thư mục `processing_own_phase/` cung cấp cùng workflow dưới dạng các Python module có thể tái sử dụng và chạy từ command line.

Đây là bài toán:

| Thuộc tính | Giá trị |
|---|---|
| Learning paradigm | Supervised learning |
| Task | Single-label multi-class classification |
| Input | Ảnh grayscale `28 x 28` |
| Tensor input | `[batch_size, 1, 28, 28]` |
| Target | Một class index trong `[0, 9]` |
| Model output | 10 raw logits |
| Loss | `CrossEntropyLoss` |
| Primary metric | Accuracy |
| Supporting metrics | Loss, precision, recall, F1-score, confusion matrix |

Mục tiêu của project không chỉ là đạt accuracy cao. Pipeline còn phải bảo đảm:

- Không để official test set tham gia preprocessing, training hoặc model selection.
- Có validation protocol rõ ràng để so sánh architecture và hyperparameters.
- Mỗi experiment bắt đầu từ model, optimizer và shuffle state mới.
- Metric được tính đúng trên toàn bộ số sample.
- Có thể tái lập workflow bằng notebook hoặc reusable package.
- Checkpoint chứa đủ thông tin để dựng lại model.
- Model sau khi load phải cho kết quả giống model trước khi save.
- Có loss graph, accuracy graph, confusion matrix và predicted-versus-actual images.

### 1.1. Đối chiếu với yêu cầu bài tập

| Yêu cầu bài tập | Nơi thực hiện |
|---|---|
| Học PyTorch tensors | Phase 3, Phase 4 và Phase 5 |
| Học Dataset và DataLoader | Phase 3 và Phase 5 |
| Học transforms | Phase 3 và Phase 5 |
| Build neural network | Phase 6 |
| Forward pass | Phase 6 và Phase 7 |
| Autograd và backward pass | Phase 6 sanity check và Phase 7 |
| Optimization | Phase 7 |
| Validation và model selection | Phase 7 |
| Evaluate model accuracy | Phase 8 |
| Precision, recall, F1 và confusion matrix | Phase 8 |
| Experiment architecture/hyperparameters | Phase 7 |
| Visualize loss | Phase 7 và artifact `loss_curve.png` |
| Display predicted versus actual images | Phase 9 |
| Save và load model | Phase 9 |
| Python code | `practice_1.ipynb` và `processing_own_phase/` |
| Brief report | Phase 8 analysis và README này |

## 2. Luồng xử lý end-to-end

Pipeline được thực hiện theo thứ tự:

```text
Define problem and metrics
        |
        v
Record environment and imports
        |
        v
Load official FashionMNIST partitions
        |
        v
EDA on the 60,000-image official training pool
        |
        v
Class-stratified split: 54,000 train + 6,000 validation
        |
        v
Compute normalization statistics from 54,000 train images only
        |
        v
Create deterministic and augmented dataset views
        |
        v
Build and sanity-check the MLP
        |
        v
Run five controlled experiments on train/validation
        |
        v
Select configuration and best epoch using validation metrics
        |
        v
Rebuild from fresh state and train on all 60,000 official training images
        |
        v
Evaluate once per completed run on the 10,000-image official test set
        |
        v
Save checkpoint, reload model, verify logits and display predictions
```

### 2.1. Vai trò của từng data partition

FashionMNIST có hai official partitions:

- Official training pool: 60,000 labeled images.
- Official test set: 10,000 labeled images.

Project chia official training pool thành train và validation trước khi fit model:

| Partition | Số sample | Số sample mỗi class | Vai trò |
|---|---:|---:|---|
| Training subset | 54,000 | 5,400 | Tính normalization statistics, optimization và controlled experiments |
| Validation subset | 6,000 | 600 | Theo dõi epoch, so sánh experiment và model selection |
| Official test set | 10,000 | 1,000 | Final generalization evaluation |

Sau khi validation đã chọn xong configuration và số epoch, model được khởi tạo lại rồi train trên toàn bộ 60,000 official training images. Việc này cho phép model cuối sử dụng lại 6,000 validation images để fit parameters sau khi mọi modeling decision đã được khóa.

### 2.2. Quy tắc chống data leakage

Pipeline áp dụng các ràng buộc sau:

1. Official test set không được dùng để tính mean hoặc standard deviation.
2. Official test set không được truyền vào training API.
3. Official test result không được dùng để chọn architecture, optimizer, learning rate, dropout, augmentation hoặc epoch.
4. Validation subset không được nhận random augmentation.
5. Train và validation indices phải disjoint.
6. Normalization statistics chỉ được tính từ 54,000 training indices.
7. Experiment selection chỉ dựa trên validation accuracy và validation loss.
8. Sau final test evaluation, model configuration không được thay đổi dựa trên test result.

EDA được phép quan sát official training pool trước split vì EDA không fit model parameters. Tuy nhiên, official test set vẫn bị loại khỏi EDA để giữ final evaluation độc lập.

## 3. Cấu trúc project

```text
practice_1/
  README.md
  practice_1.ipynb
  data/
  diagrams/
    01_preprocessing.png
    02_model_building.png
    03_training_epoch.png
    04_training_batch.png
    Pratice_diagram.pdf
    pratice1_diagram.drawio
  processing_own_phase/
    __init__.py
    config.py
    data.py
    evaluate.py
    experiment.py
    main.py
    model.py
    requirements.txt
    save_load.py
    train.py
    utils.py
    visualize.py
  outputs/
    experiments/
      E0_baseline.pth
      E1_deeper.pth
      E2_dropout.pth
      E3_sgd.pth
      E4_augmentation.pth
    fashion_mnist_model.pth
    summary.json
    summary_quick.json
    notebook_verification.json
    eda_class_distribution.png
    pixel_intensity_distribution.png
    image_brightness_contrast.png
    per_class_intensity_boxplot.png
    class_samples.png
    class_mean_images.png
    eda_outliers.png
    data_samples.png
    class_distribution.png
    loss_curve.png
    accuracy_curve.png
    experiment_comparison.png
    confusion_matrix.png
    predictions_grid.png
  runs/
    <run-session>/
      <experiment-id>/
  save_log_agent_process_each_phase/
```

### 3.1. Hai cách biểu diễn cùng pipeline

`practice_1.ipynb` là deliverable có tính trình bày. Nó sắp xếp code, Markdown, output và biểu đồ theo từng phase để người học có thể theo dõi toàn bộ quá trình.

`processing_own_phase/` là reusable implementation. Logic được tách theo trách nhiệm để có thể test, import và chạy lại mà không phụ thuộc notebook state.

| Module | Trách nhiệm |
|---|---|
| `config.py` | Paths, class names, baseline config và experiment configs |
| `data.py` | Loading, chunked EDA analysis, duplicate audit, stratified split, normalization, transforms, datasets và DataLoaders |
| `model.py` | Configurable MLP, model builder và fail-fast sanity check |
| `train.py` | Optimizer factory, one-epoch training, experiment training và final training |
| `experiment.py` | Chạy các controlled experiments, lưu checkpoint và chọn experiment |
| `evaluate.py` | Validation/test evaluation và per-class accuracy |
| `save_load.py` | Save checkpoint, load checkpoint, rebuild model và verify reload |
| `visualize.py` | Data samples, distributions, curves, comparison, predictions và confusion matrix |
| `utils.py` | Device selection, reproducibility, parameter count và environment summary |
| `main.py` | Orchestrate toàn bộ pipeline end-to-end |

### 3.2. Notebook-to-module mapping

| Notebook phase | Reusable module tương ứng |
|---|---|
| Phase 1 | Problem contract trong notebook và README |
| Phase 2 | `config.py`, `utils.py` |
| Phase 3 | `data.load_raw_datasets()` |
| Phase 4 | `data.analyze_training_pool()` và các hàm EDA plotting trong `visualize.py` |
| Phase 5 | `data.stratified_split_indices()`, `prepare_datasets()` và loader factories |
| Phase 6 | `model.FashionMNISTModel`, `build_model()` và `sanity_check_model()` |
| Phase 7 | `train.py` và `experiment.py` |
| Phase 8 | `evaluate.py` |
| Phase 9 | `save_load.py` và `visualize.py` |

## 4. Class mapping và configuration chung

FashionMNIST sử dụng class index theo thứ tự:

| Index | Class |
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

Baseline configuration:

| Hyperparameter | Giá trị |
|---|---|
| Seed | `42` |
| Validation ratio | `0.10` |
| Batch size | `64` |
| Number of workers | `0` |
| Input dimension | `784` |
| Hidden dimensions | `[128]` |
| Number of classes | `10` |
| Dropout | `0.0` |
| Optimizer | Adam |
| Learning rate | `0.001` |
| Weight decay | `0.0` |
| Epoch budget | `10` |
| Baseline augmentation | Không |

Seed `42` được dùng cho Python, NumPy, PyTorch, train/validation split và DataLoader shuffle generator. Mục tiêu là làm cho cùng một run trong cùng môi trường có khả năng tái lập. Seed không bảo đảm bitwise-identical result giữa CPU, CUDA và MPS vì mỗi backend có thể dùng numerical kernels khác nhau.

## 5. Phase 1 - Problem Definition

### 5.1. Phase này làm gì?

Phase 1 xác định chính xác bài toán trước khi viết model:

- Dữ liệu đầu vào là ảnh grayscale `28 x 28`.
- Mỗi ảnh chỉ thuộc một class.
- Có 10 classes.
- Model trả về một score cho mỗi class.
- Accuracy là primary metric.
- Cross-entropy là optimization objective.
- Train/validation/test protocol phải được khóa trước experiment.

### 5.2. Tại sao phải định nghĩa metric trước model?

Architecture chỉ có ý nghĩa khi biết tiêu chí đánh giá. FashionMNIST được cân bằng, mỗi class có cùng số sample, vì vậy overall accuracy là primary metric hợp lý:

```text
accuracy = number_of_correct_predictions / number_of_samples
```

Accuracy cho biết tỷ lệ dự đoán đúng tổng thể. Tuy nhiên, accuracy không giải thích model đang yếu ở class nào, nên Phase 8 bổ sung precision, recall, F1-score và confusion matrix.

Cross-entropy cho một sample có target class `y` được hiểu là:

```text
loss = -log(softmax(logits)[y])
```

Khi lấy trung bình trên dataset, loss phạt model nếu probability dành cho true class thấp. Đây là signal khả vi để Autograd tính gradient.

### 5.3. Input, target và output contract

| Thành phần | Contract |
|---|---|
| Một ảnh trước batch | `[1, 28, 28]`, `torch.float32` |
| Một batch ảnh | `[B, 1, 28, 28]`, `torch.float32` |
| Một batch label | `[B]`, `torch.int64` |
| Model output | `[B, 10]` raw logits |
| Prediction | `argmax(logits, dim=1)` |

Model không trả về class name trực tiếp. Nó trả về 10 logits; index của logit lớn nhất được ánh xạ sang `class_names`.

### 5.4. Success criteria

Một run được xem là hoàn chỉnh khi:

1. Load đúng 60,000 training images và 10,000 test images.
2. Tạo đúng train/validation split mà không overlap.
3. Build model nhận đúng tensor shape.
4. Forward, loss, backward và optimizer update đều hoạt động.
5. Controlled experiments chỉ dùng train và validation.
6. Selected configuration được train lại từ fresh state.
7. Final model được đánh giá trên đủ 10,000 test samples.
8. Có learning curves và error analysis.
9. Checkpoint load lại thành công.
10. Loaded model khớp model trước khi save.

## 6. Phase 2 - Environment Setup

### 6.1. Phase này làm gì?

Phase 2 chuẩn bị execution environment trước khi xử lý data:

- Resolve `PROJECT_DIR`, `DATA_DIR`, `OUTPUT_DIR` và `RUNS_DIR`.
- Thêm project parent vào `sys.path` khi cần.
- Kích hoạt inline plotting trong notebook.
- Import PyTorch, TorchVision, NumPy, Matplotlib, Seaborn, scikit-learn và TensorBoard.
- In Python, PyTorch và TorchVision versions.
- Kiểm tra CUDA và Apple MPS.

### 6.2. Tại sao phải ghi nhận environment?

Deep Learning result phụ thuộc không chỉ vào code và seed mà còn vào:

- Python version.
- PyTorch/TorchVision version.
- CPU, CUDA hoặc MPS backend.
- Numerical kernel của thiết bị.
- Dependency versions.

Ghi environment giúp phân biệt code regression với sai khác số học do backend.

Latest notebook run hiện ghi nhận:

| Thuộc tính | Giá trị |
|---|---|
| Python | `3.10.11` |
| PyTorch | `2.13.0` |
| TorchVision | `0.28.0` |
| CUDA available | `False` |
| MPS available | `True` |
| Selected notebook device | `mps` |

Reusable package chọn device theo thứ tự:

```text
CUDA -> MPS -> CPU
```

Điều này cho phép cùng source code chạy trên NVIDIA GPU, Apple Silicon GPU hoặc CPU mà không thay training logic.

### 6.3. Vai trò của các dependency chính

| Dependency | Mục đích |
|---|---|
| `torch` | Tensor, Autograd, model, loss và optimization |
| `torchvision` | FashionMNIST dataset và transforms |
| `numpy` | Array processing và evaluation output |
| `matplotlib` | Plot samples và learning curves |
| `seaborn` | Confusion matrix heatmap |
| `scikit-learn` | Classification report và confusion matrix trong notebook |
| `tensorboard` | Theo dõi train/validation metrics theo epoch |
| `notebook`, `jupyter` | Chạy interactive notebook |

Version constraints được lưu trong `processing_own_phase/requirements.txt`.

## 7. Phase 3 - Data Loading

### 7.1. Phase này làm gì?

FashionMNIST được load bằng `torchvision.datasets.FashionMNIST`:

```text
train=True  -> official training pool, 60,000 images
train=False -> official test set, 10,000 images
```

Ở phase này chỉ áp dụng `ToTensor()`.

### 7.2. `ToTensor()` xử lý gì?

`ToTensor()`:

- Chuyển PIL image hoặc NumPy image thành `torch.Tensor`.
- Đổi shape thành channel-first `[1, 28, 28]`.
- Đổi dtype thành `torch.float32`.
- Scale raw pixel từ `[0, 255]` sang `[0, 1]`.

Normalization chưa được áp dụng ở Phase 3 vì mean và standard deviation hợp lệ phải được tính sau khi có training indices ở Phase 5.

### 7.3. Tại sao chưa split ngay trong Data Loading?

Phase 3 chịu trách nhiệm load hai official partitions đúng theo TorchVision. Phase 4 cần quan sát cấu trúc toàn bộ official training pool cho EDA. Train/validation split được tạo ở Phase 5, trước mọi bước fit model hoặc tính preprocessing statistics.

Việc tách trách nhiệm như vậy giúp phân biệt:

- Official partition do dataset provider định nghĩa.
- Internal validation split do project định nghĩa.

### 7.4. Output của phase

| Object | Kích thước | Transform |
|---|---:|---|
| `train_dataset` | 60,000 | `ToTensor()` |
| `test_dataset` | 10,000 | `ToTensor()` |
| `class_names` | 10 | Class label mapping |

Official test set được load để chuẩn bị pipeline nhưng không được đọc trong EDA, preprocessing statistics, experiment hoặc model selection.

## 8. Phase 4 - Exploratory Data Analysis

### 8.1. Mục tiêu của EDA

EDA kiểm tra data contract và các assumption trước khi preprocessing hoặc xây model:

- Raw image shape, dtype, range và label domain có đúng contract không?
- Class distribution có cân bằng và có class nào bị thiếu không?
- Pixel intensity phân bố như thế nào trên cả background và foreground?
- Brightness và contrast thay đổi ra sao giữa từng ảnh và từng class?
- Có non-finite value, all-black, all-white hoặc constant image không?
- Có raw image trùng hoàn toàn hoặc duplicate mang conflicting label không?
- Sample, class-average image và statistical extreme có phù hợp với label không?

EDA chỉ đọc `train_dataset.data` và `train_dataset.targets` của 60,000-image official training pool. Official test images và labels không được đọc, trực quan hóa hoặc dùng để đưa ra modeling decision.

`analyze_training_pool()` xử lý ảnh theo chunk mặc định 2,048 samples. Cách này tính exact histogram, sums và image-level statistics mà không tạo thêm một floating-point copy của toàn bộ 60,000 ảnh. Duplicate audit dùng raw image bytes nên không bị sai lệch bởi normalization.

### 8.2. Raw data contract

Kết quả kiểm tra trên current dataset copy:

| Thuộc tính | Giá trị |
|---|---|
| EDA scope | Official training pool only |
| Samples | 60,000 |
| Raw shape | `[60000, 28, 28]` |
| Resolution | `28 x 28` |
| Channels | `1` |
| Raw dtype | `torch.uint8` |
| Raw range | `[0, 255]` |
| Label range | `[0, 9]` |
| Number of classes | `10` |
| Shape after `ToTensor()` | `[1, 28, 28]` |

Raw contract xác nhận mỗi model input có một channel và `28 * 28 = 784` features sau khi flatten. Assertions fail ngay nếu dataset size, dimensions, dtype, pixel range hoặc label domain lệch khỏi contract.

### 8.3. Class distribution

Class count được tính bằng `torch.bincount`; mỗi class có đúng 6,000 images, tương đương 10% official training pool:

| Class | Count |
|---|---:|
| T-shirt/top | 6,000 |
| Trouser | 6,000 |
| Pullover | 6,000 |
| Dress | 6,000 |
| Coat | 6,000 |
| Sandal | 6,000 |
| Shirt | 6,000 |
| Sneaker | 6,000 |
| Bag | 6,000 |
| Ankle boot | 6,000 |

Dataset cân bằng nên:

- Accuracy có thể dùng làm primary selection metric.
- Không cần class weighting trong baseline loss.
- Stratified split có thể giữ chính xác cùng tỷ lệ ở mỗi class.

`outputs/eda_class_distribution.png` dùng bar chart để count giữa các class có thể được so sánh trực tiếp.

### 8.4. Pixel intensity distribution

Exact 256-bin histogram kiểm kê toàn bộ `60,000 x 28 x 28 = 47,040,000` raw pixels. Statistics được scale sang `[0, 1]` để cùng interpretation với output của `ToTensor()`:

| Statistic | Giá trị |
|---|---:|
| Pixel count | `47,040,000` |
| Mean | `0.286041` |
| Standard deviation | `0.353024` |
| Median | `0.000000` |
| 5th percentile | `0.000000` |
| 95th percentile | `0.905882` |
| Zero-valued fraction | `50.21%` |
| Maximum-valued fraction | `0.81%` |

`outputs/pixel_intensity_distribution.png` gồm một full-range histogram và một non-zero view. Full view làm rõ background spike tại zero; non-zero view giúp quan sát foreground clothing pixels mà không bị spike này che khuất.

Các giá trị mean và standard deviation trên đây chỉ là descriptive EDA statistics. Phase 5 tính lại statistics từ 54,000 training subset; validation samples không tham gia fit normalization.

### 8.5. Image-level brightness và contrast

Với mỗi ảnh:

- Brightness được định nghĩa là mean của 784 pixel đã scale.
- Contrast được định nghĩa là population standard deviation của 784 pixel đã scale.

Mean brightness của toàn pool là `0.286041`; mean image contrast là `0.320249`. Per-class summaries cho thấy footwear và upper-body garments không chiếm cùng intensity range. Ví dụ, Sandal có mean brightness `0.1367`, trong khi Coat có mean brightness `0.3853`.

`outputs/image_brightness_contrast.png` biểu diễn overall distributions. `outputs/per_class_intensity_boxplot.png` dùng boxplot để so sánh spread, median và outlier giữa 10 classes.

Sự khác nhau giữa class không dẫn đến class-specific normalization, vì true class không tồn tại tại inference time. Project dùng một bộ training-only global statistics nhất quán cho mọi input.

### 8.6. Data quality và duplicate audit

Quality audit kiểm tra:

- Non-finite transformed values.
- All-black, all-white và constant images.
- Labels ngoài `[0, 9]` và missing classes.
- Exact duplicates dựa trên raw bytes.
- Duplicate groups có nhiều hơn một label.

Current dataset copy ghi nhận:

| Check | Count |
|---|---:|
| Non-finite images | `0` |
| All-black images | `0` |
| All-white images | `0` |
| Constant images | `0` |
| Invalid labels | `0` |
| Missing classes | `0` |
| Exact duplicate groups | `0` |
| Exact duplicate samples | `0` |
| Conflicting-label duplicate groups | `0` |

Audit chỉ report và assert các contract quan trọng; nó không tự động xóa dữ liệu. Một observation cực trị về brightness hoặc contrast chưa đủ để kết luận ảnh bị corrupt.

### 8.7. Class samples và class-average images

`outputs/class_samples.png` hiển thị hai samples cho mỗi class. Indices được chọn bằng local generator với seed `42`, nên kết quả reproducible và không thay đổi global RNG state dùng cho split hoặc training.

Grid này kiểm tra orientation, grayscale rendering, label mapping và intra-class variation. Việc lấy đúng số sample theo từng class tránh trường hợp một random batch vô tình thiếu class.

`outputs/class_mean_images.png` tính pixel-wise mean image cho từng class. Mean silhouettes làm rõ:

- Separation tương đối rõ giữa footwear, bag và upper-body garments.
- Visual overlap đáng kể giữa T-shirt/top, Pullover, Coat và Shirt.
- Confusion giữa các upper-body classes là hypothesis hợp lý để kiểm tra ở Phase 8, không phải kết luận model trước khi evaluate.

### 8.8. Statistical extremes

`outputs/eda_outliers.png` hiển thị darkest, brightest, lowest-contrast và highest-contrast images theo deterministic ranking. Ảnh cực trị vẫn là garment hợp lệ trong current inspection nên được giữ nguyên.

Mục đích của bước này là tạo review candidates có cơ sở định lượng. EDA không tự động gán “outlier” thành “corruption”, vì extreme-but-valid samples có thể đại diện cho variation thật mà model cần học.

### 8.9. Kết luận từ EDA

EDA cho phép chốt các quyết định:

1. MLP input phải có 784 features.
2. Output layer phải có 10 units.
3. Accuracy phù hợp làm primary metric vì class balance hoàn toàn.
4. Baseline không cần class weights.
5. Không có bằng chứng để tự động loại image trong current dataset copy.
6. Global normalization là phù hợp; statistics phải được fit lại trên training subset.
7. Official training pool đủ lớn để dành 10% làm validation.
8. Upper-body class overlap cần được kiểm tra bằng confusion matrix sau final evaluation.

## 9. Phase 5 - Data Preprocessing

### 9.1. Mục tiêu của phase

Phase 5 biến raw dataset thành các dataset views và DataLoaders có contract rõ ràng cho training, validation và final test.

Các bước được thực hiện:

1. Tạo class-stratified train/validation split.
2. Tính training-only mean và standard deviation.
3. Định nghĩa baseline, augmented và evaluation transforms.
4. Tạo các FashionMNIST parent datasets riêng.
5. Gắn disjoint indices bằng `Subset`.
6. Tạo DataLoaders.
7. Chạy assertions cho split, class balance, determinism, shape và dtype.

### 9.2. Class-stratified split

Với từng class từ `0` đến `9`:

1. Lấy toàn bộ indices thuộc class đó.
2. Shuffle indices bằng `torch.Generator().manual_seed(42)`.
3. Lấy 10% đầu tiên, tương đương 600 images, cho validation.
4. Lấy 90% còn lại, tương đương 5,400 images, cho training.
5. Ghép indices của tất cả classes.
6. Shuffle lại train indices và validation indices bằng cùng seeded generator.

Kết quả:

| Partition | Total | Mỗi class |
|---|---:|---:|
| Train | 54,000 | 5,400 |
| Validation | 6,000 | 600 |

### 9.3. Tại sao dùng stratified split?

Random split thông thường có thể tạo chênh lệch nhỏ về class count. Stratification bảo đảm:

- Train và validation giữ đúng class balance.
- Accuracy giữa experiment không bị ảnh hưởng bởi validation class composition khác nhau.
- Per-class comparison có cùng support.
- Split có thể tái tạo bằng seed.

### 9.4. Training-only normalization statistics

Raw pixels tại `train_indices` được chuyển sang `float32`, chia `255.0`, rồi tính:

```text
TRAIN_MEAN = 0.286139
TRAIN_STD  = 0.353084
```

Normalization áp dụng:

```text
normalized_pixel = (pixel - TRAIN_MEAN) / TRAIN_STD
```

Mục đích:

- Center input quanh zero.
- Đưa scale feature về phạm vi ổn định hơn.
- Hỗ trợ optimizer học hiệu quả hơn.
- Giữ cùng preprocessing contract cho train, validation, test và inference.

Chỉ 54,000 training images tham gia tính statistics. Validation và test chỉ tiêu thụ hai giá trị đã được tính từ train.

### 9.5. Transform strategy

| Dataset view | Transform pipeline | Tính chất |
|---|---|---|
| Baseline train | `ToTensor()` -> `Normalize()` | Deterministic |
| Augmented train | `RandomHorizontalFlip(0.5)` -> `RandomRotation(10)` -> `ToTensor()` -> `Normalize()` | Stochastic |
| Validation | `ToTensor()` -> `Normalize()` | Deterministic |
| Test | `ToTensor()` -> `Normalize()` | Deterministic |

Baseline transform không chứa augmentation để E0, E1, E2 và E3 so sánh trong điều kiện data giống nhau.

Augmentation chỉ xuất hiện ở E4. Thiết kế này giúp đo riêng tác động của augmentation thay vì trộn augmentation vào baseline.

Validation và test không được augment vì:

- Cùng một sample phải cho cùng tensor ở mỗi lần evaluation.
- Metric phải phản ánh dữ liệu gốc, không phản ánh một random view.
- Random evaluation transform làm metric dao động ngoài tác động của model.

### 9.6. Tại sao phải tạo các parent dataset riêng?

`Subset` chỉ lưu indices và gọi transform của parent dataset. Nếu train và validation dùng chung một parent có random augmentation, validation cũng có thể bị augment.

Project tạo riêng:

- `baseline_training_pool`
- `augmented_training_pool`
- `validation_pool`
- `test_dataset`

Sau đó:

- `train_subset` dùng baseline parent và train indices.
- `augmented_train_subset` dùng augmented parent nhưng vẫn chỉ dùng train indices.
- `validation_subset` dùng deterministic parent và validation indices.

Nhờ vậy, transform policy được tách khỏi index policy.

### 9.7. DataLoader strategy

| Loader | Dataset | Batch size | Shuffle | Số batch |
|---|---|---:|---|---:|
| Train | 54,000 train samples | 64 | Có | 844 |
| Validation | 6,000 validation samples | 64 | Không | 94 |
| Test | 10,000 official test samples | 64 | Không | 157 |

Training cần shuffle để batch order không bị cố định theo dataset order. Validation và test không shuffle để evaluation order deterministic và dễ đối chiếu predictions.

Trong Phase 7, mỗi experiment tạo train DataLoader mới với generator mới nhưng cùng seed. Điều này bảo đảm mỗi experiment nhận cùng initial shuffle protocol.

### 9.8. Sanity checks

Pipeline assert:

- `len(train_subset) == 54_000`
- `len(validation_subset) == 6_000`
- `len(test_dataset) == 10_000`
- Train và validation không overlap.
- Union của train và validation có đủ 60,000 unique indices.
- Train có 5,400 samples mỗi class.
- Validation có 600 samples mỗi class.
- Cùng một validation item được đọc hai lần cho tensor giống nhau.
- Images có shape `[B, 1, 28, 28]`.
- Images có dtype `torch.float32`.
- Labels có shape `[B]`.
- Labels có dtype `torch.int64`.
- Tất cả image values đều finite.

Batch đầu tiên trong current run:

```text
Train:      images=(64, 1, 28, 28), labels=(64,), range=[-0.810, 2.022]
Validation: images=(64, 1, 28, 28), labels=(64,), range=[-0.810, 2.022]
Test:       images=(64, 1, 28, 28), labels=(64,), range=[-0.810, 2.022]
```

Giá trị sau normalization không còn giới hạn trong `[0, 1]`; đây là expected behavior.

![Phase 5 preprocessing pipeline](diagrams/01_preprocessing.png)

## 10. Phase 6 - Model Building

### 10.1. Architecture

Model là configurable multi-layer perceptron:

```text
Input [B, 1, 28, 28]
        |
        v
Flatten -> [B, 784]
        |
        v
Linear hidden layer
        |
        v
ReLU
        |
        v
Optional Dropout
        |
        v
Additional hidden layers when configured
        |
        v
Linear output layer -> [B, 10] raw logits
```

Baseline:

```text
Flatten
Linear(784, 128)
ReLU
Linear(128, 10)
```

Deeper configuration:

```text
Flatten
Linear(784, 256)
ReLU
Linear(256, 128)
ReLU
Linear(128, 10)
```

### 10.2. Tại sao chọn MLP?

MLP là Deep Learning baseline phù hợp với mục tiêu học PyTorch cơ bản:

- Architecture đủ đơn giản để quan sát rõ forward, loss, backward và optimizer.
- Parameter count có thể tính và kiểm tra trực tiếp.
- Training nhanh hơn CNN trên môi trường không có GPU mạnh.
- Có thể kiểm soát riêng depth, dropout và optimizer.

Trade-off là `Flatten` làm mất explicit spatial structure. MLP không khai thác locality và translation patterns tốt như CNN, nên các class có hình dạng gần nhau vẫn khó phân biệt.

### 10.3. Configurable constructor

`FashionMNISTModel` nhận:

| Argument | Ý nghĩa |
|---|---|
| `hidden_dims` | Số unit của từng hidden layer |
| `dropout` | Dropout probability |
| `num_classes` | Số output logits |
| `input_dim` | Số flattened input features |

Constructor validate:

- `input_dim > 0`
- `num_classes > 1`
- Mọi hidden dimension phải dương.
- `0 <= dropout < 1`

Validation này làm configuration error fail sớm thay vì chỉ phát hiện sau khi training đã bắt đầu.

### 10.4. Tại sao output không có Softmax?

`CrossEntropyLoss` nhận raw logits và nội bộ kết hợp `LogSoftmax` với negative log-likelihood. Thêm Softmax vào model trước loss sẽ:

- Lặp transformation không cần thiết.
- Làm numerical stability kém hơn.
- Vi phạm input contract của `CrossEntropyLoss`.

Softmax chỉ được dùng ở evaluation hoặc visualization khi cần chuyển logits thành probabilities.

### 10.5. Parameter count

Baseline:

```text
Linear(784, 128): 784 * 128 + 128 = 100,480
Linear(128, 10):  128 * 10 + 10   =   1,290
Total                                  101,770
```

Deeper model:

```text
Linear(784, 256): 784 * 256 + 256 = 200,960
Linear(256, 128): 256 * 128 + 128 =  32,896
Linear(128, 10):  128 * 10 + 10   =   1,290
Total                                  235,146
```

Parameter count được report để accuracy gain có thể được so sánh với model complexity.

### 10.6. Fail-fast model sanity check

Trước training, pipeline tạo một model riêng cho sanity check và chạy 8 real samples:

1. Kiểm tra image shape `(8, 1, 28, 28)`.
2. Kiểm tra image dtype và device khớp model parameters.
3. Kiểm tra label shape `(8,)` và dtype `int64`.
4. Chạy forward.
5. Kiểm tra logits shape `(8, 10)`.
6. Kiểm tra logits finite.
7. Tính `CrossEntropyLoss`.
8. Kiểm tra loss finite.
9. Snapshot parameters.
10. Chạy `zero_grad`, `backward` và kiểm tra gradients.
11. Chạy một optimizer step.
12. Xác nhận ít nhất một parameter đã thay đổi.

Current sanity result:

| Check | Result |
|---|---|
| Output shape | `(8, 10)` |
| Finite loss | `2.312276` trong latest notebook run |
| Gradients present | Pass |
| Gradients finite | Pass |
| Non-zero gradient exists | Pass |
| Parameters changed | Pass |

Sanity check dùng model riêng để không làm thay đổi model sẽ được dùng cho controlled experiment.

![Phase 6 model building and sanity-check flow](diagrams/02_model_building.png)

## 11. Phase 7 - Model Training

### 11.1. Mục tiêu của phase

Phase 7 triển khai:

- Device selection.
- Reproducibility setup.
- Optimizer factory.
- One-epoch training loop.
- Validation loop.
- Best-checkpoint selection.
- TensorBoard logging.
- Controlled experiments.
- Final retraining trên toàn bộ official training pool.

### 11.2. Reproducibility

Trước mỗi run:

```text
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
```

Mỗi experiment đồng thời nhận:

- Fresh model.
- Fresh optimizer.
- Fresh DataLoader.
- Fresh seeded shuffle generator.
- Fresh history list.
- Fresh TensorBoard writer.

Nếu tái sử dụng model hoặc optimizer giữa các experiment, experiment sau sẽ hưởng weights hoặc momentum từ experiment trước và comparison sẽ không còn công bằng.

### 11.3. Device selection

Priority:

```text
torch.cuda.is_available()
torch.backends.mps.is_available()
fallback to cpu
```

Mọi image và label batch được chuyển sang selected device trước forward. Model cũng được chuyển sang cùng device.

### 11.4. Batch-level training flow

Với mỗi batch:

```text
images, labels -> device
optimizer.zero_grad(set_to_none=True)
logits = model(images)
loss = criterion(logits, labels)
validate loss is finite
loss.backward()
optimizer.step()
accumulate weighted loss and correct predictions
```

Ý nghĩa từng bước:

| Bước | Mục đích |
|---|---|
| `zero_grad` | Xóa gradients từ batch trước |
| Forward | Tính logits từ images |
| Loss | Đo sai lệch giữa logits và labels |
| `backward` | Autograd tính gradient theo từng parameter |
| `optimizer.step` | Cập nhật parameters theo gradient |
| Metric accumulation | Tính epoch loss và accuracy |

![Phase 7 batch-level training flow](diagrams/04_training_batch.png)

### 11.5. Sample-weighted epoch loss

`CrossEntropyLoss` mặc định trả mean loss của batch. Batch cuối có thể nhỏ hơn batch size chuẩn. Vì vậy epoch loss được tính:

```text
loss_sum += batch_loss * actual_batch_size
epoch_loss = loss_sum / total_sample_count
```

Nếu chỉ lấy trung bình đều giữa các batch, batch cuối nhỏ vẫn có trọng số bằng batch đầy và epoch loss sẽ bị lệch nhẹ.

Accuracy được tính:

```text
correct += (argmax(logits) == labels).sum()
epoch_accuracy = correct / total_sample_count
```

Không dùng `tensor.data`; predictions được lấy trực tiếp từ logits bằng `argmax`.

### 11.6. Validation flow

Sau mỗi training epoch:

- Chuyển model sang `eval()`.
- Chạy validation trong `torch.inference_mode()`.
- Không tạo gradient graph.
- Không update parameters.
- Tính sample-weighted validation loss.
- Tính validation accuracy.

`inference_mode()` giảm memory và computation vì evaluation không cần backward.

### 11.7. Best-checkpoint rule

Một epoch được xem là tốt hơn nếu:

1. Validation accuracy cao hơn best accuracy hiện tại.
2. Nếu accuracy bằng chính xác, validation loss thấp hơn được dùng làm tie-breaker.

Khi có best epoch mới, `state_dict` được clone về CPU. Sau khi experiment kết thúc, model load lại best state thay vì giữ weights của epoch cuối.

Điều này quan trọng vì train loss có thể tiếp tục giảm trong khi validation performance bắt đầu xấu đi.

![Phase 7 epoch-level training and checkpoint-selection flow](diagrams/03_training_epoch.png)

### 11.8. Optimizers

Optimizer factory hỗ trợ:

| Optimizer | Configuration |
|---|---|
| Adam | Learning rate và weight decay |
| SGD | Learning rate, momentum và weight decay |

Adam là baseline vì adaptive learning rate thường hội tụ nhanh trong bài toán nhỏ. SGD được giữ thành controlled experiment để quan sát trade-off, không được thay vào baseline mà không có validation evidence.

### 11.9. Controlled experiment design

Năm experiments:

| ID | Hidden dimensions | Dropout | Optimizer | Learning rate | Momentum | Augmentation |
|---|---|---:|---|---:|---:|---|
| E0_baseline | `[128]` | 0.0 | Adam | 0.001 | Không áp dụng | Không |
| E1_deeper | `[256, 128]` | 0.0 | Adam | 0.001 | Không áp dụng | Không |
| E2_dropout | `[256, 128]` | 0.2 | Adam | 0.001 | Không áp dụng | Không |
| E3_sgd | `[256, 128]` | 0.0 | SGD | 0.01 | 0.9 | Không |
| E4_augmentation | `[256, 128]` | 0.0 | Adam | 0.001 | Không áp dụng | Có |

Logic comparison:

- E0 thiết lập simple MLP baseline.
- E1 chỉ thay architecture depth và width so với E0.
- E2 giữ architecture E1, chỉ thêm dropout.
- E3 giữ architecture E1, chỉ thay optimizer và learning rate phù hợp với SGD.
- E4 giữ architecture/optimizer của E1, chỉ thêm augmentation.

Các yếu tố được giữ cố định:

- Cùng train/validation indices.
- Cùng seed.
- Cùng batch size.
- Cùng 10-epoch budget.
- Cùng normalization statistics.
- Cùng primary metric.
- Mỗi run bắt đầu từ fresh state.

### 11.10. TensorBoard logging

Mỗi experiment ghi:

- `Loss/Train`
- `Loss/Validation`
- `Accuracy/Train`
- `Accuracy/Validation`
- `LearningRate`

Logs được đặt trong:

```text
runs/<YYYYMMDD-HHMMSS>/<experiment-id>/
```

Session timestamp ngăn experiment mới ghi đè TensorBoard event của session trước. Writer luôn được close bằng `finally` để event buffer được flush ngay cả khi training gặp exception.

### 11.11. Latest notebook experiment results

Latest interactive notebook run ngày `2026-07-27` sử dụng MPS:

| Experiment | Best epoch | Validation loss | Validation accuracy | Parameters | Runtime |
|---|---:|---:|---:|---:|---:|
| E0_baseline | 6 | 0.3265 | 88.43% | 101,770 | 76.9 s |
| E1_deeper | 7 | 0.3177 | 89.15% | 235,146 | 81.0 s |
| E2_dropout | 8 | 0.3279 | 88.72% | 235,146 | 85.3 s |
| E3_sgd | 10 | 0.3183 | 88.93% | 235,146 | 65.6 s |
| E4_augmentation | 10 | 0.3366 | 87.82% | 235,146 | 92.0 s |

E1_deeper được chọn vì có validation accuracy cao nhất là `89.15%`.

Interpretation:

- Deeper MLP cải thiện baseline trong run này.
- Dropout `0.2` không cải thiện validation accuracy trong 10-epoch budget.
- SGD đạt gần E1 nhưng vẫn thấp hơn primary metric.
- Augmentation được chọn chưa phù hợp với MLP/budget hiện tại hoặc cần tuning riêng.
- Kết quả không hỗ trợ việc tự động kết luận model phức tạp hơn luôn tốt hơn; quyết định vẫn dựa trên validation evidence.

Các plots dưới đây được tạo bởi recorded reusable-package CPU run. Chúng minh họa cùng experiment protocol; số liệu chính xác của latest MPS notebook run nằm trong bảng phía trên.

![Recorded CPU experiment comparison](outputs/experiment_comparison.png)

![Recorded CPU loss curves](outputs/loss_curve.png)

![Recorded CPU accuracy curves](outputs/accuracy_curve.png)

### 11.12. Final retraining

Sau model selection:

1. Copy selected configuration.
2. Đổi experiment ID thành `<selected-experiment>_final`.
3. Đặt epoch count bằng selected best epoch.
4. Khởi tạo model mới từ fresh state.
5. Khởi tạo optimizer mới.
6. Train trên toàn bộ 60,000 official training images.
7. Không evaluate official test set trong quá trình final training.

Trong latest MPS notebook run:

```text
Selected configuration: E1_deeper
Final training samples: 60,000
Final training epochs: 7
Final epoch train loss: 0.2323
Final epoch train accuracy: 91.30%
```

Final training không còn validation loop vì validation đã hoàn thành vai trò model selection. Epoch count đã được cố định trước khi 6,000 validation samples được đưa trở lại training pool.

## 12. Phase 8 - Model Evaluation

### 12.1. Mục tiêu của phase

Phase 8 đo generalization của final model trên official test set. Test evaluation chỉ diễn ra sau khi:

- Preprocessing protocol đã cố định.
- Experiment comparison đã hoàn thành.
- Architecture và hyperparameters đã được chọn.
- Epoch count đã được chọn.
- Final model đã train xong.

Trong một completed pipeline run, test result không được dùng để quay lại sửa model.

### 12.2. Evaluation procedure

Evaluation:

1. Chuyển model sang `eval()`.
2. Chạy trong `torch.inference_mode()`.
3. Forward toàn bộ 10,000 test images.
4. Tính sample-weighted cross-entropy loss.
5. Lấy prediction bằng `argmax`.
6. Tích lũy predictions và targets trên CPU.
7. Kiểm tra đã evaluate đủ số sample.

### 12.3. Các metrics

Accuracy:

```text
accuracy = correct_predictions / all_predictions
```

Precision cho class `c`:

```text
precision_c = true_positive_c / predicted_positive_c
```

Recall cho class `c`:

```text
recall_c = true_positive_c / actual_positive_c
```

F1-score:

```text
F1_c = 2 * precision_c * recall_c / (precision_c + recall_c)
```

Vì test set có support 1,000 cho mỗi class, macro average và weighted average rất gần nhau.

### 12.4. Latest notebook test result

Latest MPS notebook run:

| Metric | Value |
|---|---:|
| Test samples | 10,000 |
| Test loss | 0.3294 |
| Test accuracy | 88.79% |
| Macro precision | 88.92% |
| Macro recall | 88.79% |
| Macro F1 | 88.79% |

Per-class report:

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

### 12.5. Error analysis

Trong latest notebook run:

- Trouser, Bag và Sandal có F1 cao nhất.
- Shirt có F1 thấp nhất.
- Pullover và Coat cũng yếu hơn các class có silhouette rõ.
- Shirt có cả precision và recall thấp, nghĩa là model vừa bỏ sót true Shirt images vừa dự đoán nhầm các class khác thành Shirt.
- Confusion matrix cho thấy nhóm upper-body garments như T-shirt/top, Pullover, Coat và Shirt dễ bị nhầm lẫn với nhau.

Nguyên nhân hợp lý từ dữ liệu:

- Ảnh chỉ có độ phân giải `28 x 28`.
- Ảnh là grayscale.
- Fine-grained details như cổ áo, tay áo và texture bị giảm mạnh.
- MLP flatten ảnh nên không tận dụng spatial locality như CNN.

Đây là phân tích từ observed metrics và confusion matrix của run hiện tại. Nó không được dùng để thay đổi final model sau khi đã xem test set.

### 12.6. Confusion matrix

Notebook hiển thị raw count confusion matrix:

- Row biểu diễn actual class.
- Column biểu diễn predicted class.
- Diagonal là số prediction đúng.
- Off-diagonal cells thể hiện cặp class bị nhầm.

Reusable package lưu normalized confusion matrix vào `outputs/confusion_matrix.png`, giúp so sánh recall pattern giữa các class theo tỷ lệ.

![Recorded CPU normalized confusion matrix](outputs/confusion_matrix.png)

## 13. Phase 9 - Save Model and Visualization

### 13.1. Mục tiêu của phase

Phase 9 bảo đảm final model không chỉ tồn tại trong notebook memory. Model phải:

- Được save thành checkpoint.
- Có đủ architecture và preprocessing metadata.
- Có thể load trên selected device hoặc CPU.
- Có thể dựng lại đúng architecture.
- Cho predictions và logits giống model trước khi save.
- Có visual examples để kiểm tra prediction.

### 13.2. Canonical checkpoint path

Checkpoint hiện tại:

```text
outputs/fashion_mnist_model.pth
```

File này có thể bị ghi đè khi chạy lại full notebook hoặc package. Vì vậy metrics đi kèm cần được đọc từ chính checkpoint hoặc summary có cùng run provenance.

### 13.3. Notebook checkpoint content

Notebook lưu:

| Key | Nội dung |
|---|---|
| `model_state_dict` | Final learned weights và biases |
| `model_config` | Hidden dimensions, dropout, classes và input dimension |
| `training_config` | Selected optimizer, learning rate, epochs và các config khác |
| `selected_experiment` | Experiment thắng validation |
| `best_epoch` | Epoch được validation chọn |
| `best_validation_accuracy` | Best validation accuracy |
| `best_validation_loss` | Best validation loss |
| `test_accuracy` | Final test accuracy |
| `test_loss` | Final test loss |
| `train_mean` | Training-only normalization mean |
| `train_std` | Training-only normalization standard deviation |
| `class_names` | Mapping class index sang label |

Reusable package lưu cùng nhóm thông tin theo schema:

```text
model_state_dict
model_config
training_config
metadata
```

Trong package, selection metrics, test metrics, normalization statistics và class names nằm trong `metadata`.

### 13.4. Tại sao không save toàn bộ model object?

Save `state_dict` cùng configuration có các lợi ích:

- Ít phụ thuộc vào Python object serialization.
- Architecture được tái tạo một cách explicit.
- Configuration có thể kiểm tra độc lập.
- Dễ load trên device khác bằng `map_location`.
- Phù hợp với PyTorch checkpoint practice.

### 13.5. Load procedure

Load flow:

1. `torch.load(..., map_location=device, weights_only=True)`.
2. Đọc `model_config`.
3. Khởi tạo `FashionMNISTModel(**model_config)`.
4. Load `model_state_dict`.
5. Chuyển model sang selected device.
6. Gọi `model.eval()`.

`weights_only=True` giới hạn loading vào dữ liệu checkpoint cần thiết thay vì arbitrary Python object.

### 13.6. Reload verification

Project lấy cùng một test batch và chạy:

- Original in-memory model.
- Freshly reconstructed loaded model.

Sau đó kiểm tra:

```text
argmax(original_logits) == argmax(loaded_logits)
torch.allclose(original_logits, loaded_logits)
maximum absolute logit difference
```

Latest notebook result:

| Check | Result |
|---|---|
| Predictions match | `True` |
| Logits match | `True` |
| Maximum logit difference | `0.00000000` |

Verification này mạnh hơn việc chỉ load không báo lỗi. Nó xác nhận architecture và weights thực sự tái tạo cùng function output trên input đã kiểm tra.

### 13.7. Predicted-versus-actual visualization

Notebook:

1. Chạy loaded model trên một test batch.
2. Lấy predicted class bằng `argmax`.
3. Unnormalize images:

```text
display_image = normalized_image * TRAIN_STD + TRAIN_MEAN
```

4. Clamp pixel về `[0, 1]`.
5. Hiển thị grid `4 x 4`.
6. Ghi predicted label và actual label.
7. Dùng màu xanh cho correct prediction và màu đỏ cho incorrect prediction.

Unnormalization chỉ phục vụ display. Model vẫn nhận normalized tensor.

Reusable package còn tính Softmax probabilities để hiển thị confidence của predicted class trong `outputs/predictions_grid.png`.

![Recorded CPU predicted-versus-actual examples](outputs/predictions_grid.png)

## 14. Artifacts và ý nghĩa

### 14.1. Experiment checkpoints

```text
outputs/experiments/E0_baseline.pth
outputs/experiments/E1_deeper.pth
outputs/experiments/E2_dropout.pth
outputs/experiments/E3_sgd.pth
outputs/experiments/E4_augmentation.pth
```

Mỗi checkpoint chứa best validation state và experiment history của một controlled run.

### 14.2. Final model và reports

| Artifact | Nội dung |
|---|---|
| `outputs/fashion_mnist_model.pth` | Final selected model checkpoint |
| `outputs/summary.json` | Full reusable-package run summary |
| `outputs/summary_quick.json` | Quick integration run summary |
| `outputs/notebook_verification.json` | Notebook verification evidence của recorded verification run |

### 14.3. Visual artifacts

| Artifact | Câu hỏi được trả lời |
|---|---|
| `eda_class_distribution.png` | Official training pool có cân bằng giữa 10 classes không? |
| `pixel_intensity_distribution.png` | Background và foreground pixel intensities phân bố như thế nào? |
| `image_brightness_contrast.png` | Brightness và contrast thay đổi ra sao giữa các ảnh? |
| `per_class_intensity_boxplot.png` | Intensity characteristics khác nhau như thế nào giữa classes? |
| `class_samples.png` | Mỗi class có sample và label mapping hợp lý không? |
| `class_mean_images.png` | Silhouette trung bình và visual overlap giữa classes là gì? |
| `eda_outliers.png` | Những ảnh cực trị theo brightness và contrast có hợp lệ không? |
| `data_samples.png` | Input images và labels có hợp lý không? |
| `class_distribution.png` | Train/validation/test có cân bằng không? |
| `loss_curve.png` | Train và validation loss thay đổi thế nào? |
| `accuracy_curve.png` | Train và validation accuracy thay đổi thế nào? |
| `experiment_comparison.png` | Experiment nào có best validation accuracy cao nhất? |
| `confusion_matrix.png` | Model nhầm class nào với class nào? |
| `predictions_grid.png` | Loaded model dự đoán đúng/sai trên sample cụ thể ra sao? |

### 14.4. Legacy artifacts

Một số root-level `.pt` files trong `outputs/` có tên như `E1_lr_low.pt`, `E2_lr_high.pt` hoặc `E5_adam.pt`. Đây là artifacts từ experiment protocol cũ, không thuộc controlled experiment set hiện tại.

Nguồn hiện tại cần ưu tiên:

- `outputs/experiments/*.pth` cho experiment checkpoints mới.
- `outputs/fashion_mnist_model.pth` cho final checkpoint.
- `outputs/summary.json` hoặc notebook outputs cho metrics, kèm đúng run provenance.

File `fashion_mnist_model.pth` ở project root cũng không phải canonical path của pipeline hiện tại.

## 15. Kết quả theo execution environment

### 15.1. Tại sao có hai bộ số liệu?

Project có recorded full runs trên CPU và MPS. Cùng seed và cùng protocol không bảo đảm bitwise-identical training giữa device backends vì floating-point operations và numerical kernels có thể khác.

Hai run đều:

- Chọn `E1_deeper`.
- Đạt validation accuracy xấp xỉ 89%.
- Đạt official test accuracy xấp xỉ 88.8%.
- Verify checkpoint reload với maximum logit difference bằng zero trong chính run đó.

### 15.2. So sánh recorded runs

| Run | Device | Selected experiment | Best epoch | Validation loss | Validation accuracy | Test loss | Test accuracy |
|---|---|---|---:|---:|---:|---:|---:|
| Latest notebook, 2026-07-27 | MPS | E1_deeper | 7 | 0.3177 | 89.15% | 0.3294 | 88.79% |
| Reusable package, 2026-07-25 | CPU | E1_deeper | 10 | 0.3308 | 89.30% | 0.3320 | 88.84% |

Không nên ghép best epoch của run này với checkpoint hoặc test metric của run kia. Mỗi result phải được đọc như một complete run:

```text
environment
split and seed
experiment history
selected epoch
final retraining
test metrics
checkpoint
```

### 15.3. CPU package experiment results

Recorded trong `outputs/summary.json`:

| Experiment | Best epoch | Validation loss | Validation accuracy | Parameters |
|---|---:|---:|---:|---:|
| E0_baseline | 10 | 0.3509 | 88.45% | 101,770 |
| E1_deeper | 10 | 0.3308 | 89.30% | 235,146 |
| E2_dropout | 7 | 0.3201 | 88.65% | 235,146 |
| E3_sgd | 7 | 0.3204 | 89.07% | 235,146 |
| E4_augmentation | 10 | 0.3440 | 87.67% | 235,146 |

CPU package final metrics:

| Metric | Value |
|---|---:|
| Test loss | 0.3320 |
| Test accuracy | 88.84% |
| Maximum reload logit difference | 0.00000000 |
| Full pipeline runtime | Khoảng 6 phút 18 giây |

## 16. Cách chạy project

### 16.1. Chạy notebook

Từ workspace root:

```bash
cd "/Users/ticoder-coder/Documents/DEEP_LEARNING/LAB&PRACTICE"
```

Chọn Python kernel từ project virtual environment:

```text
venv (3.10.11)
```

Mở:

```text
total_practice/practice_1/practice_1.ipynb
```

Sau đó:

1. Restart kernel để xóa state từ run trước.
2. Chạy notebook từ trên xuống dưới.
3. Không chạy Phase 8 trước khi Phase 7 hoàn thành.
4. Không dùng test result để sửa experiment config trong cùng evaluation protocol.

Notebook cần chạy theo thứ tự vì các phase sau sử dụng object được tạo ở phase trước.

### 16.2. Kiểm tra dependencies

```bash
./venv/bin/python -m pip check
```

Install hoặc đồng bộ dependencies khi cần:

```bash
./venv/bin/python -m pip install -r total_practice/practice_1/processing_own_phase/requirements.txt
```

### 16.3. Chạy reusable full pipeline

Từ workspace root:

```bash
./venv/bin/python -m total_practice.practice_1.processing_own_phase.main
```

Full pipeline:

- Chuẩn bị data.
- Chạy năm experiments.
- Chọn best validation result.
- Final train trên 60,000 images.
- Evaluate official test set.
- Save/reload checkpoint.
- Tạo figures.
- Ghi `outputs/summary.json`.

### 16.4. Chạy quick integration check

```bash
./venv/bin/python -m total_practice.practice_1.processing_own_phase.main --quick
```

Quick mode:

- Giới hạn training data còn tối đa 1,024 samples.
- Giới hạn validation và test còn tối đa 512 samples.
- Chạy mỗi experiment 1 epoch.

Quick result chỉ dùng để kiểm tra integration contract. Nó không phải benchmark và không được so sánh với full-run accuracy.

### 16.5. Mở TensorBoard

Trong notebook:

```python
%load_ext tensorboard
%tensorboard --logdir $TENSORBOARD_LOG_DIR
```

Hoặc từ terminal:

```bash
./venv/bin/tensorboard --logdir total_practice/practice_1/runs
```

## 17. Verification checklist

### 17.1. Data verification

- Official training pool có 60,000 samples.
- Official test set có 10,000 samples.
- Internal train subset có 54,000 samples.
- Internal validation subset có 6,000 samples.
- Train và validation không overlap.
- Train/validation union có 60,000 unique indices.
- Train có 5,400 samples mỗi class.
- Validation có 600 samples mỗi class.
- Mean/std chỉ được tính từ training indices.
- Validation transform deterministic.
- Test transform deterministic.

### 17.2. Tensor and loader verification

- Image batch có shape `[B, 1, 28, 28]`.
- Label batch có shape `[B]`.
- Image dtype là `float32`.
- Label dtype là `int64`.
- Image values đều finite.
- Train loader shuffle.
- Validation và test loaders không shuffle.

### 17.3. Model verification

- Baseline có 101,770 trainable parameters.
- Deeper model có 235,146 trainable parameters.
- Forward output có shape `[B, 10]`.
- Logits finite.
- Loss finite.
- Gradients tồn tại.
- Gradients finite.
- Ít nhất một gradient non-zero.
- Optimizer step làm parameters thay đổi.

### 17.4. Training verification

- Mỗi experiment dùng fresh model và optimizer.
- Mỗi experiment dùng fresh seeded shuffle generator.
- Epoch loss được sample-weighted.
- Validation chạy trong inference mode.
- Best state được chọn bằng validation accuracy.
- Validation loss chỉ là tie-breaker.
- Official test loader không được truyền vào Phase 7 training functions.
- TensorBoard writer được close.

### 17.5. Final evaluation và checkpoint verification

- Final training dùng đủ 60,000 official training images.
- Final epoch count được chọn trước test evaluation.
- Test evaluation dùng đủ 10,000 official test images.
- Test loss được sample-weighted.
- Checkpoint chứa weights, architecture, preprocessing và metrics.
- Loaded model được dựng từ checkpoint config.
- Loaded predictions khớp original predictions.
- Loaded logits khớp original logits.
- Prediction visualization dùng ảnh đã unnormalize.

## 18. Các design decision quan trọng

### 18.1. Accuracy là primary metric

Được chọn vì dataset cân bằng hoàn toàn. Precision, recall và F1 vẫn cần cho error analysis nhưng không được dùng ngầm để thay selection rule sau khi experiment bắt đầu.

### 18.2. Baseline-first

Simple MLP E0 được thiết lập trước deeper model. E1 chỉ được chấp nhận khi validation evidence cho thấy cải thiện.

### 18.3. Validation-driven selection

Nếu chọn model bằng test accuracy, test set sẽ trở thành validation set không chính thức và reported test performance sẽ lạc quan. Vì vậy model selection chỉ dùng validation.

### 18.4. Controlled experiments thay vì thay nhiều yếu tố cùng lúc

Mỗi experiment sau anchor E1 chỉ thay một nhóm quyết định chính. Điều này giúp giải thích nguyên nhân performance thay đổi.

### 18.5. Best checkpoint thay vì last checkpoint

Epoch cuối không nhất thiết có validation performance tốt nhất. Best state được giữ độc lập với last epoch.

### 18.6. Final retraining trên toàn bộ official training pool

Validation samples chỉ cần tách ra trong model selection. Sau khi selection hoàn thành, chúng trở thành labeled training data hợp lệ cho final parameter fitting.

### 18.7. Save configuration cùng weights

Weights không đủ để biết số layer, hidden dimensions, dropout, input dimension hoặc normalization statistics. Checkpoint phải chứa cả model/preprocessing contract.

## 19. Giới hạn hiện tại

### 19.1. Model limitation

MLP flatten ảnh nên không khai thác spatial structure. Đây là lý do hợp lý để một CNN có thể là bước phát triển tiếp theo, nhưng CNN chưa được tự động thay vào baseline vì project hiện tập trung vào PyTorch fundamentals và controlled comparison.

### 19.2. Experiment budget

Mỗi experiment chỉ có 10 epochs và một seed. Result đủ cho exercise nhưng chưa phải robust benchmark trên nhiều seeds.

### 19.3. Cross-device reproducibility

CPU và MPS có thể chọn best epoch khác nhau dù dùng cùng seed. README vì vậy tách result theo execution environment.

### 19.4. Test-set governance

Mỗi lần chạy full pipeline sẽ evaluate test set lại. Trong học tập điều này giúp verify implementation, nhưng trong benchmark nghiêm ngặt cần hạn chế số lần quan sát test result và giữ một evaluation record cố định.

### 19.5. Artifact overwrite

`outputs/fashion_mnist_model.pth` và các figure names là fixed paths. Full run mới có thể ghi đè artifacts cũ. TensorBoard logs tránh vấn đề này bằng timestamped session folders.

### 19.6. EDA memory use

EDA không stack 60,000 floating-point tensors. `analyze_training_pool()` đọc raw `uint8` images theo chunk 2,048 samples để cập nhật exact histogram, sums, per-image statistics và class sums. Cách này phù hợp với FashionMNIST và giảm peak memory, nhưng duplicate audit vẫn giữ một hash map của raw image bytes. Với dataset lớn hơn, duplicate detection nên chuyển sang batched cryptographic hashes hoặc một external indexing workflow.

## 20. Hướng cải thiện hợp lệ trong tương lai

Các hướng có thể nghiên cứu tiếp, nhưng phải tiếp tục giữ official test set ngoài model selection:

1. Thêm CNN baseline để tận dụng spatial locality.
2. Chạy nhiều seeds và report mean cùng standard deviation.
3. Tuning learning rate và weight decay trên validation.
4. Thêm learning-rate scheduler thành controlled experiment.
5. Thử augmentation phù hợp hơn với FashionMNIST.
6. Phân tích calibration và confidence.
7. Lưu run manifest liên kết environment, summary và checkpoint.
8. Thêm automated tests cho split, metrics và checkpoint contracts.

Mỗi improvement cần:

- Có hypothesis trước khi chạy.
- Giữ baseline để so sánh.
- Chỉ thay một factor chính khi có thể.
- Dùng validation metric cho quyết định.
- Không dùng official test set để tuning.

## 21. Tóm tắt project

Project hiện triển khai đầy đủ một PyTorch classification workflow:

1. Xác định classification task, loss và metrics.
2. Ghi execution environment.
3. Load đúng official FashionMNIST partitions.
4. Thực hiện EDA trên official training pool.
5. Tạo stratified train/validation split.
6. Tính training-only normalization statistics.
7. Xây deterministic và augmented data pipelines.
8. Build configurable MLP và chạy fail-fast sanity check.
9. Train năm controlled experiments.
10. Chọn model bằng validation accuracy.
11. Train lại selected configuration trên 60,000 images.
12. Evaluate trên 10,000-image official test set.
13. Phân tích per-class errors và confusion matrix.
14. Save checkpoint cùng configuration và preprocessing metadata.
15. Load model và verify predictions/logits.
16. Tạo learning curves, experiment comparison và prediction displays.

Kết luận chính từ các recorded runs là deeper MLP `E1_deeper` cải thiện simple baseline và được chọn nhất quán trên cả CPU lẫn MPS. Official test accuracy nằm quanh `88.8%`. Điểm yếu chính vẫn là phân biệt các upper-body garment classes có hình dạng gần nhau ở độ phân giải `28 x 28` grayscale.
