# Plan Verify Phase 1 to Phase 4 - `practice_1.ipynb`

**Ngày kiểm định:** 2026-07-25  
**Đối tượng:** `total_practice/practice_1/practice_1.ipynb`  
**Phạm vi:** Phase 1 - Environment Setup, Phase 2 - Data Loading, Phase 3 - EDA,
Phase 4 - Data Preprocessing  
**Ngoài phạm vi:** sửa code notebook, model building, training, evaluation và
save/load model. Các phase sau chỉ được đọc khi cần kiểm tra hợp đồng đầu ra của
Phase 1-4 và phát hiện data leakage.

## 1. Mục tiêu kiểm định

1. Xác minh Phase 1-4 chạy được từ một kernel sạch theo đúng thứ tự.
2. Xác minh dữ liệu FashionMNIST được tải đúng và transform đúng.
3. Xác minh EDA phản ánh đúng đặc điểm dữ liệu, không tạo kết luận sai.
4. Xác minh preprocessing, split và DataLoader tạo đầu vào hợp lệ cho training.
5. Đối chiếu Phase 1-4 với các phần tương ứng trong đề bài và PyTorch docs/tutorials.
6. Ghi rõ phần đã đạt, phần còn thiếu, lỗi, rủi ro và hướng cải thiện; không sửa
   notebook khi chưa có yêu cầu riêng.

## 2. Yêu cầu bài tập nằm trong phạm vi

| Yêu cầu | Phase dự kiến | Bằng chứng cần có |
|---|---:|---|
| Study Tensors | 1, 3, 4 | `Tensor`, shape, dtype, value range và batch shape |
| Study Datasets/Loaders | 2, 4 | `FashionMNIST`, `DataLoader`, split hợp lệ |
| Study Transforms | 2, 4 | transform train/eval đúng mục đích |
| Load FashionMNIST | 2 | đúng train/test split và số lượng mẫu |
| Apply transforms | 2, 4 | tensor hóa, normalize/augmentation hợp lệ |
| Chuẩn bị dữ liệu cho model | 4 | batch `[B, 1, 28, 28]`, label `[B]`, dtype đúng |
| Dùng PyTorch docs/tutorials | 1-4 | implementation phù hợp API và best practice hiện hành |

Model, autograd, optimization, accuracy, save/load, experiment và các deliverable
cuối cùng chưa được dùng để đánh giá mức hoàn thành của Phase 1-4.

## 3. Tiêu chí pass/fail

### Phase 1 - Environment Setup

- Import đầy đủ dependency thực sự cần cho Phase 1-4.
- In được Python, PyTorch, TorchVision và accelerator availability.
- Kernel/runtime của output lưu trong notebook phải nhất quán hoặc được ghi rõ.
- Path setup không phụ thuộc sai vào thư mục mở notebook.
- Seed và device policy phải được đặt trước khi tạo dữ liệu ngẫu nhiên nếu notebook
  muốn reproducible.

### Phase 2 - Data Loading

- `train=True` có 60,000 mẫu; `train=False` có 10,000 mẫu.
- Root path ổn định và không vô tình tải data vào nhiều vị trí.
- `ToTensor()` tạo `torch.float32`, shape `[1, 28, 28]`, range `[0, 1]`.
- Class names và labels đúng 10 lớp FashionMNIST.
- Không dùng official test set để tune hyperparameter.

### Phase 3 - EDA

- Kiểm tra size, shape, channels, dtype, pixel range và label range.
- Kiểm tra class distribution và sample visualization.
- Mean/std được tính đúng, có chú thích phạm vi dữ liệu và mục đích sử dụng.
- Data-quality checks không gắn nhãn sai cho ảnh hợp lệ.
- EDA không dùng cách làm tiêu tốn RAM/thời gian không cần thiết.
- Các cell có thể chạy lại từ kernel sạch và output không bị stale.

### Phase 4 - Data Preprocessing

- Train transform và evaluation transform được tách đúng.
- Augmentation chỉ áp dụng cho training subset.
- Validation và test transform phải deterministic.
- Train/validation split phải xảy ra trước khi tạo DataLoader.
- Train, validation, test không overlap.
- `shuffle=True` chỉ cho train; validation/test dùng `shuffle=False`.
- Batch image/label có shape, dtype và range đúng sau normalize.
- DataLoader không giữ reference tới dataset cũ sau khi biến `train_dataset` bị gán lại.

## 4. Kế hoạch thực thi

| Bước | Kiểm tra | Phương pháp | Trạng thái |
|---:|---|---|---|
| 1 | Xác định cell thuộc Phase 1-4 | Parse cấu trúc notebook và heading | Hoàn thành |
| 2 | Kiểm tra execution metadata/output | So execution count, output và runtime version | Hoàn thành |
| 3 | Static review từng cell | Review API, logic, thứ tự biến và side effect | Hoàn thành |
| 4 | Đối chiếu PyTorch docs | Dataset, transform, split, DataLoader best practice | Hoàn thành |
| 5 | Runtime smoke test | Chạy logic Phase 1-4 trong process sạch | Hoàn thành |
| 6 | Data integrity test | Size, class, shape, dtype, range, NaN/Inf | Hoàn thành |
| 7 | Split/leakage test | Kiểm tra indices, transforms và loader references | Hoàn thành |
| 8 | Đánh giá completeness | Trace từng yêu cầu bài tập tới bằng chứng | Hoàn thành |
| 9 | Tổng hợp findings | Severity, tác động, nguyên nhân, improvement | Hoàn thành |
| 10 | Final review tài liệu | Kiểm tra tính nhất quán và kết luận | Hoàn thành |

## 5. Thang mức độ findings

- **Critical:** làm sai kết quả đánh giá hoặc gây data leakage.
- **High:** pipeline chạy nhưng validation/test không đáng tin hoặc khó tái lập.
- **Medium:** thiếu kiểm tra quan trọng, hiệu năng kém, hoặc mô tả dễ gây hiểu sai.
- **Low:** vấn đề trình bày, maintainability hoặc bằng chứng chưa đầy đủ.

## 6. Nguyên tắc đề xuất improvement

1. Ưu tiên correctness và data integrity trước tối ưu tốc độ.
2. Giữ code học tập dễ đọc, không thêm abstraction không cần thiết.
3. Mỗi đề xuất phải nêu vị trí, tác động và tiêu chí xác nhận sau khi sửa.
4. Tách rõ phần bắt buộc để đạt đề bài và phần nâng cao.
5. Không sửa notebook trong lần kiểm định này.

## 7. Findings và kết luận

### 7.1 Kết luận điều hành

**Phase 1-4 chưa thể được xem là hoàn chỉnh hoặc đúng chuẩn end-to-end.**

- Phase 2 tải FashionMNIST đúng và **Pass**.
- Phase 3 có EDA cơ bản đúng nhưng còn thiếu sanity checks và chưa tối ưu, nên
  **Pass có điều kiện**.
- Phase 1 có output môi trường không khớp workspace hiện tại và import không chạy
  được với dependency đã khai báo, nên **Chưa pass**.
- Phase 4 tạo được transformed batch đúng shape/dtype/range, nhưng cách đặt split
  sau DataLoader gây data leakage và validation dùng random augmentation, nên
  **Không pass**.

Với riêng yêu cầu bài tập, Phase 1-4 đã bao phủ đúng hướng các mục Tensors,
Datasets/DataLoaders, Transforms, load FashionMNIST và apply transforms. Tuy nhiên,
chúng chưa tạo ra một data pipeline đáng tin cậy để chuyển sang model training và
hyperparameter experiments. Toàn bộ bài tập cũng chưa thể kết luận hoàn thành vì
model, autograd, optimization, evaluation, save/load và deliverables nằm ngoài
phạm vi kiểm định lần này.

### 7.2 Scorecard theo phase

| Phase | Kết quả | Phần đã đúng | Phần chặn hoàn thành |
|---|---|---|---|
| 1 - Environment | Chưa pass | In version và accelerator flags | Kernel/output stale; thiếu dependency; chưa seed |
| 2 - Data Loading | Pass | Đúng train/test split, `ToTensor`, classes | Root path còn phụ thuộc working directory |
| 3 - EDA | Pass có điều kiện | Size, shape, dtype, balance, mean/std, sample grid | Thiếu range/label checks; quality rule chưa chặt; tốn RAM |
| 4 - Preprocessing | Không pass | Train/test transforms tách riêng; batch đúng | Full validation leakage; stochastic validation; split sai thứ tự |

## 8. Bằng chứng kiểm định

### 8.1 Runtime và data integrity

Các phép kiểm tra được chạy bằng environment hiện có trong workspace và dataset
đã tải local, không tải lại dữ liệu.

| Thuộc tính | Kết quả | Đánh giá |
|---|---:|---|
| Official training set | 60,000 | Đúng |
| Official test set | 10,000 | Đúng |
| Số class | 10 | Đúng |
| Số ảnh mỗi class trong official train | 6,000 | Cân bằng |
| Sample shape | `[1, 28, 28]` | Đúng |
| Sample dtype sau `ToTensor()` | `torch.float32` | Đúng |
| Raw tensor range | `[0.0, 1.0]` | Đúng |
| Label range/dtype | `[0, 9]`, `torch.int64` | Đúng |
| Dataset mean/std | `0.2860406 / 0.3530242` | Khớp output notebook |
| Transformed batch shape | `[64, 1, 28, 28]` | Đúng |
| Label batch shape | `[64]` | Đúng |
| Batch range sau `Normalize(0.5, 0.5)` | `[-1.0, 1.0]` | Đúng |
| All-black/all-white images | `0 / 0` | Không phát hiện |

### 8.2 Bằng chứng leakage và validation instability

Notebook tạo `train_loader` ở cell 30 từ toàn bộ transformed training dataset,
sau đó mới gán lại `train_dataset` thành subset ở cell 39.

| Kiểm tra | Kết quả |
|---|---:|
| Kích thước train subset sau split | 48,000 |
| Kích thước validation subset | 12,000 |
| `len(train_loader.dataset)` sau split | 60,000 |
| Validation indices vẫn có trong `train_loader.dataset` | 12,000 / 12,000 |
| Train và validation share cùng augmented parent dataset | `True` |
| Cùng một validation sample có thể đổi qua nhiều lần đọc | `True` |
| Test sample có deterministic transform | `True` |

Điều này xác nhận validation set vừa bị model nhìn thấy trong training, vừa bị
random flip/rotation khi đánh giá.

### 8.3 Notebook state và dependency

- Notebook metadata ghi kernel `.venv (3.11.9)`.
- Output đã lưu ghi Python `3.11.9`, PyTorch `2.5.1+cu121`, CUDA `True`.
- Workspace `venv` hiện tại chạy Python `3.10.11`, PyTorch `2.13.0`,
  TorchVision `0.28.0`, CUDA `False`, MPS `False`.
- Environment log cũ lại ghi PyTorch `2.13.0`, MPS `True`.
- Cell import chung không chạy được trong `venv` hiện tại vì thiếu cả
  `scikit-learn` và `tensorboard`; hai package này cũng không có trong
  `processing_own_phase/requirements.txt`.
- Một số code cell Phase 1-4 có output nhưng `execution_count=null`, nên output
  lưu trong notebook không đủ làm bằng chứng cho một lần Run All sạch.

## 9. Findings chi tiết

### F01 - Critical - Validation leakage do tạo DataLoader trước split

**Vị trí:** cell 30, liên quan cell 38-41.

**Nguyên nhân:** `DataLoader` giữ reference tới object dataset được truyền khi khởi
tạo. Việc gán lại biến `train_dataset` ở cell 39 không thay dataset bên trong
`train_loader`.

**Tác động:** model train trên đủ 60,000 mẫu, bao gồm toàn bộ 12,000 validation
samples. Validation loss/accuracy và mọi lựa chọn hyperparameter dựa trên chúng
không còn đáng tin cậy.

**Improvement bắt buộc:**

1. Split official training indices trước.
2. Tạo train subset và validation subset có transform phù hợp.
3. Chỉ sau đó mới tạo `train_loader`, `val_loader`, `test_loader`.
4. Assert train/validation indices không giao nhau.
5. Assert `len(train_loader.dataset) == train_size`.

**Tiêu chí verify:** 48,000 train, 12,000 validation, intersection bằng 0, train
loader chỉ có 48,000 samples.

### F02 - High - Validation đang dùng random augmentation

**Vị trí:** cell 27, 29 và 39-41.

**Nguyên nhân:** `random_split()` trả về hai `Subset` cùng trỏ tới một
`FashionMNIST` parent có `RandomHorizontalFlip` và `RandomRotation`.

**Tác động:** cùng một validation image thay đổi giữa các epoch/lần evaluate;
validation metric có thêm nhiễu và không đo trên một distribution cố định.

**Improvement bắt buộc:** dùng cùng indices nhưng hai dataset instances:

- Train dataset: augmentation + tensor conversion + normalization.
- Validation dataset: tensor conversion + cùng normalization, không random
  augmentation.
- Test dataset: giống validation transform.

**Tiêu chí verify:** đọc cùng một validation index nhiều lần phải nhận tensor giống
nhau; train sample được phép khác nhau.

### F03 - High - Phase 1 chưa reproducible từ environment hiện tại

**Vị trí:** cell 3-5, notebook metadata và requirements.

**Tác động:** Run All từ environment của repository có thể dừng ngay ở cell import;
output hiện tại không phản ánh máy đang kiểm định.

**Improvement bắt buộc:**

1. Chốt một environment/kernel chính thức.
2. Khai báo đủ dependency dùng trong notebook, gồm `scikit-learn` và
   `tensorboard`, hoặc dời optional imports về phase cần dùng.
3. Chọn đúng kernel rồi Restart Kernel + Run All.
4. Lưu output mới và kiểm tra execution counts tăng tuần tự.
5. Ghi version thực tế vào report thay vì dùng output cũ.

**Tiêu chí verify:** Phase 1-4 chạy từ kernel sạch không lỗi import; version trong
notebook trùng version kiểm tra từ environment đã khai báo.

### F04 - Medium - Thiếu seed policy cho Phase 1-4

**Vị trí:** trước cell 24 và cell 30.

Notebook chỉ seed generator cho split ở phase sau. EDA sample order, DataLoader
shuffle và random augmentation chưa có generator/seed policy rõ ràng.

**Improvement:** đặt `SEED = 42`, seed `random`, NumPy và PyTorch ngay Phase 1; truyền
`torch.Generator().manual_seed(SEED)` cho split/DataLoader khi cần. Ghi chú rằng
kết quả vẫn có thể khác giữa device/version.

### F05 - Medium - Baseline preprocessing chưa được giải thích như controlled choice

**Vị trí:** cell 18-19 và cell 27-28.

EDA tính mean/std `0.2860/0.3530` nhưng preprocessing dùng `(0.5, 0.5)`.
`Normalize((0.5,), (0.5,))` không sai; nó ánh xạ `[0,1]` sang `[-1,1]`. Tuy nhiên,
notebook chưa giải thích vì sao không dùng thống kê vừa tính. Random augmentation
cũng được đưa thẳng vào baseline, làm khó tách ảnh hưởng của network/hyperparameter.

**Improvement đề xuất:**

- Baseline đơn giản: chỉ `ToTensor()` hoặc một normalization cố định có giải thích.
- Nếu standardize theo data: tính mean/std chỉ từ training subset.
- Đưa augmentation thành một experiment riêng, giữ các biến khác cố định.

### F06 - Medium - EDA/data-quality checks chưa đủ chặt và dùng RAM chưa tối ưu

**Vị trí:** cell 18 và cell 20-22.

- `torch.stack()` materialize toàn bộ 60,000 ảnh float32, riêng tensor khoảng
  `188,160,000` bytes, chưa tính list và tensor trung gian.
- Chưa kiểm tra pixel min/max toàn dataset, `Inf`, label dtype/range, shape đồng nhất.
- Ảnh all-black/all-white không mặc định là corrupted data; nên ghi là suspicious
  và review, không kết luận hỏng chỉ bằng intensity.
- Có tạo `black_images`/`white_images` nhưng output chỉ in tổng corrupted count.

**Improvement:** dùng `dataset.data`/`dataset.targets` theo cách vectorized hoặc
streaming statistics; thêm assertions cho shape, dtype, finite values, label range
và in từng category kèm sample indices cần review.

### F07 - Medium - Thiếu sanity check trực quan sau preprocessing

**Vị trí:** cell 31-32.

Notebook mới in batch shape, chưa xác minh transformed image có còn hợp lý sau
rotation/flip/normalize. Đây là bước dễ bắt lỗi transform order, range hoặc hiển thị
ảnh bị sai.

**Improvement:** in dtype/range của batch và vẽ một grid nhỏ sau khi de-normalize;
hiển thị cùng label. Không cần biến Phase 4 thành EDA lớn.

### F08 - Medium - Notebook và tài liệu hiện có chưa cùng một data protocol

- Notebook dùng split 80/20, tương ứng 48,000/12,000.
- README và modular pipeline dùng khoảng 90/10, tương ứng 54,001/5,999.
- Các file `phase_03_model_build.md` và `phase_04_baseline_train.md` là log của
  modular pipeline, không tương ứng với Phase 3 EDA và Phase 4 preprocessing trong
  notebook.

**Tác động:** không thể dùng log/README hiện tại làm bằng chứng trực tiếp cho
notebook; report có thể mô tả một protocol khác code deliverable.

**Improvement:** chọn một split protocol làm nguồn sự thật, cập nhật notebook,
report và phase log cho nhất quán. Tỷ lệ 80/20 hay 90/10 đều hợp lệ nếu dùng đúng;
vấn đề là tính nhất quán và không leakage.

### F09 - Low - Data path phụ thuộc working directory

**Vị trí:** cell 3 và cell 7/29.

`root="./data"` phụ thuộc nơi Jupyter server được mở. `CURRENT_DIR = Path.cwd()` cũng
không đảm bảo đó là thư mục chứa notebook.

**Improvement:** định nghĩa và in rõ `PROJECT_DIR`/`DATA_DIR`, kiểm tra thư mục mong
đợi tồn tại trước khi load. Tránh việc cùng dataset bị tải vào nhiều thư mục.

## 10. Mapping tới yêu cầu bài tập

| Yêu cầu trong scope | Hiện trạng | Kết luận |
|---|---|---|
| Load FashionMNIST | Đúng train/test split và số lượng | Đạt |
| Apply transforms | API và output tensor đúng | Đạt về API, chưa đạt về validation design |
| Tensors | Có shape, dtype, mean/std và batch tensor | Đạt cơ bản, nên bổ sung range/label checks |
| Datasets/Loaders | Tạo được Dataset và DataLoader | Chưa đạt end-to-end vì loader tạo trước split |
| Use PyTorch docs/tutorials | Cấu trúc bám đúng tutorial cơ bản | Đạt định hướng |
| Chuẩn bị cho experiment | Có augmentation và normalize | Chưa đạt vì leakage và stochastic validation |

## 11. Thứ tự improvement đề xuất

### P0 - Phải sửa trước khi train lại

1. Chốt environment và dependency để Run All được.
2. Chuyển split lên trước DataLoader.
3. Tách deterministic validation transform khỏi augmented train transform.
4. Tạo đủ ba loaders sau split và assert không overlap.

### P1 - Cần làm để bài chuẩn và dễ bảo vệ

1. Seed toàn bộ nguồn randomness.
2. Chọn 80/20 hoặc 90/10 và đồng bộ notebook/report/log.
3. Bổ sung transformed-data sanity checks.
4. Giải thích normalization và tách augmentation thành controlled experiment.

### P2 - Nâng chất lượng

1. Vectorize/stream EDA statistics.
2. Hoàn thiện data-quality assertions.
3. Ổn định project/data paths và dọn optional imports theo phase.

## 12. Data flow mục tiêu sau improvement

```text
Environment + seed + paths
        |
Load official train data without random augmentation
        |
Create reproducible, non-overlapping train/validation indices
        |
        +--> Train dataset + train transform (random allowed)
        |
        +--> Validation dataset + eval transform (deterministic)

Load official test dataset + eval transform (deterministic)
        |
Create train/validation/test DataLoaders
        |
Assert sizes, non-overlap, shape, dtype, range and repeatability
```

## 13. Definition of Done cho Phase 1-4

- Restart Kernel + Run All đến hết Phase 4 không lỗi.
- Version output trùng environment đã khai báo.
- Dependency file chứa đủ package được import hoặc optional imports được dời đi.
- Train/validation/test sizes đúng protocol đã chọn.
- Train và validation indices không overlap.
- Validation/test transform deterministic.
- Train loader chỉ chứa train subset.
- Batch image `[B,1,28,28]`, `float32`; labels `[B]`, `int64`.
- Pixel range trước/sau normalize được kiểm tra và giải thích.
- Sample grid trước và sau preprocessing hiển thị đúng.
- Notebook, README và phase logs dùng cùng split/transform protocol.

## 14. Tài liệu PyTorch dùng để đối chiếu

- [Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
- [Datasets & DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [Transforms](https://docs.pytorch.org/tutorials/beginner/basics/transforms_tutorial.html)
- [`torch.utils.data` API](https://docs.pytorch.org/docs/stable/data.html)
- [`Normalize`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Normalize.html)

## 15. Files thay đổi trong lần kiểm định

- Chỉ cập nhật
  `save_log_agent_process_each_phase/plan_verify_phase1_to_4.md`.
- Không sửa `practice_1.ipynb`.
- Không chạy model training và không đánh giá Phase 5 trở đi.

## 16. Follow-up sau data split refactor

**Ngày cập nhật:** 2026-07-25

Các findings về data split trong báo cáo này phản ánh notebook trước khi refactor.
Notebook hiện tại đã được sửa theo
`save_log_agent_process_each_phase/plan_refactor_data_split.md`:

- F01 đã resolved: DataLoaders được tạo sau split và train loader chỉ có 54,000 mẫu.
- F02 đã resolved: validation dùng deterministic evaluation transform.
- F04 đã được cải thiện: split, sample loader và train loader dùng seed 42.
- F07 đã được cải thiện: Phase 6 kiểm tra batch shape, dtype và range.
- Phase numbering hiện là số nguyên dương liên tục từ Phase 1 đến Phase 10.

Các findings về dependency/environment, EDA efficiency và path robustness vẫn còn
hiệu lực và nằm ngoài phạm vi data split refactor.

## 17. Follow-up sau khi chuyển về official train/test protocol

**Ngày cập nhật:** 2026-07-25

Theo quyết định mới của chủ project, validation split đã được gỡ bỏ hoàn toàn:

- Phase Data Split không còn tồn tại.
- Official train có đủ 60,000 ảnh và được dùng cho EDA, preprocessing, training.
- Official test có đủ 10,000 ảnh và chỉ được dùng trong final evaluation/prediction.
- Không còn `Subset`, `random_split`, `val_dataset`, `val_loader` hoặc validation
  metrics trong notebook.
- Training curves chỉ biểu diễn training loss và training accuracy.
- Phase numbering hiện liên tục từ Phase 1 đến Phase 9.

Data-only verification pass với 938 train batches, 157 test batches, deterministic
test transform và train augmentation hoạt động. Không được dùng official test set
để lựa chọn architecture hoặc hyperparameters trong protocol hai tập này.
