# Code Base Audit - `practice_1.ipynb`

## 1. Thông tin audit

| Thuộc tính | Giá trị |
|---|---|
| Đối tượng | `total_practice/practice_1/practice_1.ipynb` |
| Ngày đối chiếu | 2026-08-01 |
| Bài toán | Phân loại FashionMNIST bằng PyTorch |
| Phạm vi | Source code, Markdown, stored output, checkpoint và artifact được notebook tham chiếu |
| Phương pháp | Static review, kiểm tra cấu trúc notebook và đối chiếu artifact hiện có |
| Ngoài phạm vi | Không train lại model và không tự động refactor notebook |

[Mở notebook dự phòng](../../practice_1.ipynb) mở file từ đầu. Các liên kết
**Mở Cell** trong tài liệu này dùng helper `ticoder.practice1-notebook-links`
để mở đúng cell và đưa cell đó lên đầu viewport trong VS Code. Hướng dẫn helper
nằm tại [description_own_phase](../description_own_phase/README.md); chỉ mục
stored output nằm tại [description_result](../description_result/README.md).

## 2. Kết luận điều hành

**Kết luận tổng thể: Conditional Pass.** Notebook đã triển khai đủ pipeline
baseline mà bài tập yêu cầu: tải dữ liệu, transform, train/validation split,
xây model, autograd, optimization, controlled experiments, final training,
official-test evaluation, learning curves, classification report, confusion
matrix, checkpoint save/load và predicted-versus-actual display.

Thiết kế ranh giới dữ liệu là điểm mạnh nhất. Official test set không được đưa
vào preprocessing, gradient update hoặc model selection; validation được dùng
để chọn experiment và epoch; final model được train lại trên toàn bộ official
training pool sau khi quyết định modeling đã cố định. Luồng này bám đúng protocol
đã khai báo tại [Cell 1][cell-1].

Notebook chưa đạt trạng thái clean-run reproducible. Import tại
[Cell 10][cell-10] phụ thuộc vào package `practice_1.processing_own_phase`, nhưng
package này không tồn tại trong codebase tại thời điểm audit. Ngoài ra, stored
outputs tại [Cell 27][cell-27], [Cell 29][cell-29], [Cell 32][cell-32] và
[Cell 36][cell-36] chứa cảnh báo `divide by zero`, `overflow` và `invalid value`
trong phép nhân ma trận của scikit-learn. Hai nhóm vấn đề này phải được xử lý
trước khi coi notebook là bản bàn giao có thể tái tạo hoàn toàn.

Kết quả hợp lệ đang được lưu trong lần chạy hiện tại:

| Kết quả | Giá trị | Bằng chứng |
|---|---:|---|
| Thiết bị | Apple MPS | [Cell 62][cell-62] |
| Experiment được chọn | `E1_deeper` | [Cell 70][cell-70] |
| Best validation epoch | 7 | [Cell 70][cell-70] |
| Best validation accuracy | 89.15% | [Cell 70][cell-70] |
| Final-training samples | 60,000 | [Cell 77][cell-77] |
| Official-test loss | 0.3294 | [Cell 81][cell-81] |
| Official-test accuracy | 88.79% | [Cell 81][cell-81] |
| Save/load maximum logit difference | 0.00000000 | [Cell 87][cell-87] |

## 3. Health Scorecard

| Hạng mục | Trạng thái | Nhận định ngắn | Bằng chứng |
|---|---|---|---|
| Độ phủ yêu cầu bài tập | Pass | Đủ load, transform, model, train, evaluate, experiment, visualize và save/load | [Cell 7][cell-7], [Cell 57][cell-57], [Cell 70][cell-70], [Cell 87][cell-87] |
| Ranh giới train/validation/test | Pass | Split có seed, stratified và official test được giữ riêng | [Cell 47][cell-47], [Cell 53][cell-53] |
| Tensor và model contract | Pass | Shape, dtype, logits, loss, gradient và parameter update đều được assert | [Cell 54][cell-54], [Cell 59][cell-59] |
| Training và model selection | Pass | Mỗi run có model/optimizer mới; best checkpoint chọn bằng validation | [Cell 67][cell-67], [Cell 70][cell-70] |
| Official-test evaluation | Pass | Test chạy sau final training và không điều khiển modeling decision | [Cell 79][cell-79], [Cell 81][cell-81] |
| Checkpoint round trip | Pass | Checkpoint đủ inference metadata; logits và predictions khớp sau load | [Cell 86][cell-86], [Cell 87][cell-87] |
| Clean-kernel reproducibility | Fail | Thiếu hai module local mà Phase 4 import | [Cell 10][cell-10] |
| Numerical health của EDA | Needs Fix | Bốn cell PCA/t-SNE lưu runtime warning từ linear algebra backend | [Cell 27][cell-27], [Cell 36][cell-36] |
| Environment reproducibility | Needs Fix | Có version report nhưng không có dependency manifest/lock file | [Cell 4][cell-4] |
| Artifact synchronization | Needs Fix | Một số plot chỉ hiển thị, không được lưu lại bởi code hiện tại | [Cell 73][cell-73], [Cell 88][cell-88] |
| Automated regression tests | Needs Fix | Assert trong notebook tốt nhưng chưa có test độc lập cho pipeline | [Cell 53][cell-53], [Cell 59][cell-59] |

## 4. Ma trận yêu cầu bài tập

| Yêu cầu | Trạng thái | Cách notebook đáp ứng | Bằng chứng |
|---|---|---|---|
| PyTorch tensors | Complete | `ToTensor`, normalization, device transfer và tensor assertions | [Cell 7][cell-7], [Cell 47][cell-47], [Cell 54][cell-54] |
| Datasets/DataLoaders | Complete | FashionMNIST, Subset và loader riêng cho train/validation/test | [Cell 49][cell-49], [Cell 51][cell-51] |
| Transforms | Complete | Baseline, augmentation và deterministic evaluation transforms | [Cell 47][cell-47] |
| Model building | Complete | Configurable MLP trả về 10 raw logits | [Cell 57][cell-57], [Cell 58][cell-58] |
| Autograd | Complete | `zero_grad`, forward, loss, `backward`, optimizer step | [Cell 59][cell-59], [Cell 65][cell-65] |
| Optimization | Complete | Adam/SGD factory và five controlled experiments | [Cell 63][cell-63], [Cell 69][cell-69] |
| Training loop | Complete | Sample-weighted loss, accuracy, validation và best-state selection | [Cell 65][cell-65], [Cell 67][cell-67] |
| Accuracy evaluation | Complete | Overall accuracy, per-class metrics và confusion matrix | [Cell 81][cell-81], [Cell 82][cell-82], [Cell 84][cell-84] |
| Save/load model | Complete | Lưu state dict plus metadata, reconstruct và verify logits | [Cell 86][cell-86], [Cell 87][cell-87] |
| Hyperparameter experiments | Complete | So sánh architecture, dropout, optimizer và augmentation | [Cell 69][cell-69], [Cell 72][cell-72] |
| Loss visualization | Complete in notebook | Learning curves hiển thị train/validation loss và accuracy | [Cell 73][cell-73] |
| Predicted vs actual images | Complete in notebook | Grid 4 x 4 trên official-test batch | [Cell 88][cell-88] |
| Brief report | Complete | Problem statement, EDA conclusions và test-result analysis có trong notebook | [Cell 1][cell-1], [Cell 43][cell-43], [Cell 83][cell-83] |
| PyTorch docs/tutorial references | Not Evidenced | Notebook chưa có mục references hoặc citation tới tài liệu PyTorch đã sử dụng | [Cell 0][cell-0] |

## 5. Kiến trúc và luồng dữ liệu đã audit

```text
FashionMNIST official training pool: 60,000
    -> EDA descriptive only
    -> stratified split with seed 42
       -> internal train: 54,000
          -> train-only mean/std
          -> five experiments with gradient updates
       -> internal validation: 6,000
          -> epoch/checkpoint/experiment selection only
    -> selected configuration + selected epoch count
    -> fresh final model trained on all 60,000 images

FashionMNIST official test set: 10,000
    -> deterministic transform using frozen mean/std
    -> one final evaluation
    -> classification report, confusion matrix and prediction display
    -> checkpoint metadata only; never feeds back into training
```

Ranh giới trên được khai báo tại [Cell 1][cell-1], hiện thực bằng split tại
[Cell 47][cell-47], kiểm tra bằng assertions tại [Cell 53][cell-53], áp dụng vào
experiment tại [Cell 70][cell-70], final training tại [Cell 77][cell-77] và
official-test evaluation tại [Cell 81][cell-81]. Không phát hiện code path nào
đưa `test_loader` vào training hoặc model selection.

## 6. Audit theo từng phase

| Phase | Phạm vi | Điểm đạt | Điểm cần theo dõi | Trạng thái |
|---|---|---|---|---|
| 1 - Problem Definition | Cell 1 | Objective, tensor contract, metric và evaluation protocol rõ ràng | Không có blocker | Pass |
| 2 - Environment Setup | Cell 2-5 | Tìm project path, khai báo data/output/run directories và in versions | Thiếu dependency manifest; device/version report chưa đủ để cài lại môi trường | Pass with Finding |
| 3 - Data Loading | Cell 6-7 | Dùng đúng hai official partitions và transform raw tối thiểu | `download=True` phụ thuộc network khi data chưa có, đúng kỳ vọng của TorchVision | Pass |
| 4 - EDA | Cell 8-43 | Scope đúng, kiểm tra schema/class/quality và có nhiều visualization | Missing local modules, numerical warnings, duplicated computation và memory cost cao | Fail until fixed |
| 5 - Data Preprocessing | Cell 44-54 | Stratified split, train-only statistics, deterministic validation/test và strong assertions | `std()` convention nên được ghi rõ là sample hay population; ảnh hưởng số học rất nhỏ | Pass |
| 6 - Model Building | Cell 55-59 | MLP configurable, raw logits đúng với CrossEntropyLoss, fail-fast sanity test tốt | Type contract của config còn rộng (`Dict`) | Pass |
| 7 - Model Training | Cell 60-78 | Train/eval separation, best-state copy, five fresh runs, TensorBoard và final retraining đúng protocol | Determinism trên CUDA/MPS chưa được cam kết; training/final loops còn lặp code | Pass with Finding |
| 8 - Model Evaluation | Cell 79-84 | Sample-weighted loss, accuracy, report và confusion matrix đầy đủ | Kết luận nguyên nhân lỗi ở Cell 83 mạnh hơn bằng chứng hiện có | Pass with Finding |
| 9 - Save/Load & Visualization | Cell 85-88 | Inference checkpoint đầy đủ, load bằng `map_location`, verify logits và hiển thị predictions | Prediction plot không được `savefig` bởi cell hiện tại | Pass with Finding |

### Phase 1 - Problem Definition

[Cell 1][cell-1] định nghĩa đúng bài toán supervised single-label multi-class,
input `[1, 28, 28]`, target `[0, 9]`, output 10 logits, cross-entropy objective và
accuracy metric. Success criteria bao phủ đủ deliverables. Protocol cho phép EDA
trên official training pool nhưng chỉ cho phép tính normalization từ internal
train subset, phù hợp với cách Phase 5 triển khai.

### Phase 2 - Environment Setup

[Cell 3][cell-3] xử lý hai working-directory scenario và thêm parent directory
vào `sys.path`. [Cell 4][cell-4] ghi nhận Python, PyTorch, TorchVision, CUDA và
MPS. Cách này tốt cho chẩn đoán, nhưng không thay thế dependency manifest vì nó
chỉ cho biết môi trường sau khi đã cài thành công.

### Phase 3 - Data Loading

[Cell 7][cell-7] tải đúng `train=True` và `train=False`, thiết lập seed trước khi
phân tích và chỉ dùng `ToTensor()` để EDA nhìn dữ liệu chưa normalize. Output xác
nhận 60,000 official-training images và 10,000 official-test images.

### Phase 4 - Exploratory Data Analysis

Scope tại [Cell 8][cell-8] giữ official test set ngoài EDA. Schema, class balance,
pixel statistics và quality audit có assertions tại [Cell 11][cell-11],
[Cell 13][cell-13], [Cell 17][cell-17] và [Cell 23][cell-23]. Class-stratified
samples, mean images và statistical extremes cũng được tạo có chủ đích tại
[Cell 38][cell-38] và [Cell 41][cell-41].

Phase này hiện có hai vấn đề làm giảm độ tin cậy. Thứ nhất, [Cell 10][cell-10]
không thể import module local trong codebase hiện tại. Thứ hai, PCA/t-SNE output
có numerical warnings nhưng chưa có `np.isfinite` assertion cho intermediate
arrays. Các con số explained variance và projection chỉ nên được xem là stored
results của lần chạy cũ cho tới khi warnings được xử lý và notebook chạy sạch.

### Phase 5 - Data Preprocessing

[Cell 47][cell-47] thực hiện stratified 90/10 split theo từng class với
`torch.Generator` có seed, sau đó tính `TRAIN_MEAN` và `TRAIN_STD` chỉ từ 54,000
training indices. [Cell 49][cell-49] dùng parent dataset riêng để transform policy
không bị chia sẻ ngoài ý muốn. [Cell 53][cell-53] xác minh kích thước, disjointness,
coverage và exact class balance; [Cell 54][cell-54] xác minh shape, dtype,
finiteness và transformed range.

### Phase 6 - Model Building

[Cell 57][cell-57] xây MLP từ `hidden_dims`, validate constructor arguments và
không thêm Softmax trước `CrossEntropyLoss`. [Cell 58][cell-58] cung cấp baseline
config và parameter count. [Cell 59][cell-59] kiểm tra forward output, finite
loss, finite gradients và parameter update bằng một model độc lập, đây là sanity
check phù hợp trước khi bước vào training dài.

### Phase 7 - Model Training

[Cell 64][cell-64] đặt model ở evaluation mode và dùng `torch.inference_mode()`.
[Cell 65][cell-65] triển khai đúng thứ tự `zero_grad -> forward -> loss ->
backward -> step`. [Cell 67][cell-67] tạo model/optimizer mới cho mỗi experiment,
lưu deep copy của best validation state và đóng TensorBoard writer bằng `finally`.

[Cell 69][cell-69] định nghĩa năm experiment và [Cell 70][cell-70] chọn model chỉ
theo validation accuracy, dùng validation loss làm tie-breaker. [Cell 76][cell-76]
và [Cell 77][cell-77] rebuild model từ đầu, giữ selected epoch count rồi train
trên toàn bộ 60,000 official-training images. Official test loader không xuất
hiện trong bất kỳ training function signature nào.

### Phase 8 - Model Evaluation

[Cell 80][cell-80] tích lũy sample-weighted test loss, accuracy, predictions và
targets dưới inference mode. [Cell 81][cell-81] đánh giá đúng 10,000 samples;
[Cell 82][cell-82] tạo per-class report và [Cell 84][cell-84] tạo confusion matrix.
Kết quả cho thấy nhóm upper-body garments là điểm yếu chính của baseline MLP.

### Phase 9 - Save Model & Visualization

[Cell 86][cell-86] lưu CPU-cloned state dict cùng architecture, preprocessing,
selection và evaluation metadata. [Cell 87][cell-87] reconstruct model mới,
load checkpoint lên device hiện tại và kiểm chứng cả predictions lẫn logits.
[Cell 88][cell-88] inverse-normalize ảnh trước khi hiển thị predicted-versus-
actual labels, nhờ đó visualization phản ánh đúng ảnh gốc.

## 7. Findings theo mức độ ưu tiên

Mức độ được hiểu như sau: `Blocker` ngăn clean run; `High` có thể làm kết quả
không đáng tin; `Medium` ảnh hưởng reproducibility, deliverable hoặc diễn giải;
`Low` chủ yếu ảnh hưởng maintainability và độ chính xác của tài liệu.

### F-01 - Missing local EDA package

| Thuộc tính | Nội dung |
|---|---|
| Severity | Blocker |
| Evidence | [Cell 10][cell-10] |
| Hiện trạng | `practice_1.processing_own_phase.data` và `practice_1.processing_own_phase.visualize` không tồn tại trong project |
| Ảnh hưởng | Restart kernel rồi Run All sẽ dừng trước khi `EDA_ANALYSIS` được tạo; Phase 4-9 không thể tái tạo từ source hiện tại |
| Hướng xử lý | Khôi phục package với `__init__.py`, `data.py`, `visualize.py`, hoặc chuyển toàn bộ implementation cần thiết vào notebook/module đang được version-control |
| Tiêu chí đóng | Import smoke test pass và notebook Run All thành công từ kernel sạch mà không dựa vào object còn sót trong memory |

### F-02 - Numerical warnings trong PCA và t-SNE

| Thuộc tính | Nội dung |
|---|---|
| Severity | High |
| Evidence | [Cell 27][cell-27], [Cell 29][cell-29], [Cell 32][cell-32], [Cell 36][cell-36] |
| Hiện trạng | Stored stderr có `divide by zero`, `overflow` và `invalid value encountered in matmul` từ scikit-learn |
| Ảnh hưởng | PCA coordinates, explained variance và t-SNE initialization chưa có bằng chứng numerical-finite đầy đủ |
| Hướng xử lý | Sửa dependency stack trước; assert finiteness cho `X_flatten`, `X_scaled`, PCA outputs và variance arrays; dùng solver có chủ đích; pre-reduce và stratified-sample trước t-SNE |
| Tiêu chí đóng | Bốn cell chạy không có warning, mọi intermediate array finite và visualization không rỗng |

### F-03 - EDA trùng lặp và sử dụng bộ nhớ lớn

| Thuộc tính | Nội dung |
|---|---|
| Severity | Medium |
| Evidence | [Cell 9][cell-9], [Cell 12][cell-12], [Cell 15][cell-15], [Cell 27][cell-27], [Cell 29][cell-29], [Cell 36][cell-36] |
| Hiện trạng | Cell 9 mô tả chunked aggregation nhưng Cell 15 materialize toàn bộ 60,000 ảnh float; class distribution được vẽ lại bằng pie; PCA 3 components được fit hai lần; t-SNE chạy trên toàn bộ pool |
| Ảnh hưởng | Narrative không hoàn toàn khớp code, tăng peak memory và kéo dài runtime mà không thêm nhiều thông tin |
| Hướng xử lý | Dùng artifact từ `EDA_ANALYSIS`, giữ một class-distribution chart, reuse PCA fit và dùng stratified sample cho projection visualization |
| Tiêu chí đóng | Mỗi statistic/embedding chỉ được tính một lần, memory strategy khớp Markdown và kết quả vẫn đủ cho EDA conclusions |

### F-04 - Thiếu dependency manifest

| Thuộc tính | Nội dung |
|---|---|
| Severity | Medium |
| Evidence | [Cell 4][cell-4], [Cell 5][cell-5] |
| Hiện trạng | Repository không có `requirements.txt`, `pyproject.toml`, environment file hoặc lock file |
| Ảnh hưởng | Không thể dựng lại chính xác tổ hợp PyTorch, TorchVision, NumPy, SciPy và scikit-learn; rủi ro binary incompatibility cao |
| Hướng xử lý | Thêm dependency manifest có Python range và các phiên bản đã verify; ghi rõ cách tạo kernel |
| Tiêu chí đóng | Tạo venv mới từ manifest, import toàn bộ Cell 5 và chạy smoke test thành công |

### F-05 - Plot artifacts không đồng bộ với lần chạy notebook hiện tại

| Thuộc tính | Nội dung |
|---|---|
| Severity | Medium |
| Evidence | [Cell 73][cell-73], [Cell 74][cell-74], [Cell 84][cell-84], [Cell 88][cell-88] |
| Hiện trạng | Các cell hiển thị learning curves, experiment comparison, confusion matrix và prediction grid nhưng không gọi `savefig`; các file cùng loại trong `outputs/` có timestamp cũ hơn lần chạy notebook hiện tại |
| Ảnh hưởng | Người đọc có thể nhầm external PNG cũ là artifact của stored metrics mới |
| Hướng xử lý | Mỗi figure dùng `fig.savefig(...)` trước `plt.show()` với tên file canonical; ghi run metadata hoặc overwrite có chủ đích |
| Tiêu chí đóng | Regenerate bốn artifact trong cùng Run All và đối chiếu checksum/timestamp với run session hiện tại |

### F-06 - Kết luận nguyên nhân lỗi vượt quá bằng chứng

| Thuộc tính | Nội dung |
|---|---|
| Severity | Medium |
| Evidence | [Cell 83][cell-83] |
| Hiện trạng | Report quy khó khăn ở upper-body classes cho dataset bottleneck và loại trừ model architecture như một nguyên nhân |
| Ảnh hưởng | Confusion matrix chỉ cho biết pattern lỗi, không đủ để kết luận architecture không đóng góp; CNN thường khai thác spatial structure tốt hơn MLP |
| Hướng xử lý | Diễn đạt đây là kết quả của visual overlap kết hợp giới hạn baseline MLP; chỉ khẳng định nhân quả sau controlled architecture comparison |
| Tiêu chí đóng | Report phân biệt rõ observation, interpretation và hypothesis |

### F-07 - Reproducibility mới ở mức best effort

| Thuộc tính | Nội dung |
|---|---|
| Severity | Low |
| Evidence | [Cell 62][cell-62], [Cell 63][cell-63], [Cell 67][cell-67] |
| Hiện trạng | Python, NumPy, Torch và DataLoader generator được seed, nhưng deterministic algorithm policy và backend limitation chưa được ghi lại |
| Ảnh hưởng | CUDA/MPS hoặc library version khác có thể cho metric hơi khác dù dùng cùng seed |
| Hướng xử lý | Ghi rõ reproducibility contract; bật deterministic mode khi backend hỗ trợ hoặc ghi nhận sai số kỳ vọng |
| Tiêu chí đóng | Hai clean runs trên cùng environment có split/checkpoint selection giống nhau và metric nằm trong tolerance đã định nghĩa |

### F-08 - Training/evaluation code còn duplication và type contract rộng

| Thuộc tính | Nội dung |
|---|---|
| Severity | Low |
| Evidence | [Cell 58][cell-58], [Cell 64][cell-64], [Cell 67][cell-67], [Cell 76][cell-76], [Cell 80][cell-80] |
| Hiện trạng | Config dùng `Dict`; validation/test evaluation và experiment/final training có logic gần giống nhau |
| Ảnh hưởng | Dễ phát sinh drift khi sửa một path nhưng quên path còn lại |
| Hướng xử lý | Dùng typed config/dataclass và shared epoch/evaluation primitives, vẫn giữ notebook dễ học |
| Tiêu chí đóng | Một implementation cho shared behavior, test xác nhận metrics và checkpoint selection không đổi |

### F-09 - Chưa có automated test cho pipeline

| Thuộc tính | Nội dung |
|---|---|
| Severity | Low |
| Evidence | [Cell 53][cell-53], [Cell 59][cell-59], [Cell 87][cell-87] |
| Hiện trạng | Notebook có nhiều assertions tốt nhưng chỉ chạy khi người dùng thực thi thủ công đúng thứ tự |
| Ảnh hưởng | Missing module, artifact drift và numerical warning không được CI phát hiện |
| Hướng xử lý | Thêm smoke test cho import, split, one-batch train/eval, checkpoint round trip và notebook execution |
| Tiêu chí đóng | Test suite chạy từ command line và fail khi dependency/module/output contract bị phá vỡ |

## 8. Điểm mạnh cần giữ nguyên

- Evaluation protocol tại [Cell 1][cell-1] rõ ràng và nhất quán với code.
- Stratified split, disjointness và class balance được assert tại
  [Cell 53][cell-53].
- Normalization statistics chỉ lấy từ internal training subset tại
  [Cell 47][cell-47].
- Validation và test transforms không chứa random augmentation tại
  [Cell 47][cell-47].
- Model trả raw logits và sanity check xác minh cả backward pass tại
  [Cell 59][cell-59].
- Best state được clone về CPU, tránh giữ reference tới weights tiếp tục thay
  đổi tại [Cell 67][cell-67].
- Final model được rebuild từ fresh state và train trên toàn bộ 60,000 samples
  tại [Cell 76][cell-76] và [Cell 77][cell-77].
- Official test set chỉ được sử dụng sau khi modeling decisions đã cố định tại
  [Cell 79][cell-79].
- Checkpoint lưu preprocessing và class metadata, sau đó được round-trip verify
  tại [Cell 86][cell-86] và [Cell 87][cell-87].

## 9. Thứ tự cải thiện đề xuất

| Ưu tiên | Hành động | Findings được đóng | Kết quả mong đợi |
|---:|---|---|---|
| 1 | Khôi phục `processing_own_phase` và thêm import smoke test | F-01 | Notebook chạy được từ kernel sạch |
| 2 | Ổn định scientific Python stack và xử lý PCA/t-SNE warnings | F-02, F-04 | EDA numerically clean và có thể tái tạo |
| 3 | Refactor EDA để bỏ full-copy, duplicate plot/PCA và full-pool t-SNE | F-03 | Runtime, memory và narrative nhất quán |
| 4 | Lưu lại mọi required figure từ chính cell tạo figure | F-05 | Artifact bên ngoài đồng bộ stored run |
| 5 | Chỉnh Result Analysis theo đúng mức bằng chứng | F-06 | Report chính xác về mặt khoa học |
| 6 | Bổ sung reproducibility contract, typed config và smoke tests | F-07, F-08, F-09 | Pipeline dễ bảo trì và phát hiện regression sớm |
| 7 | Thêm mục PyTorch references | Requirement gap | Chứng minh yêu cầu sử dụng docs/tutorials |

Không nên tối ưu accuracy hoặc mở rộng experiment space trước khi F-01 và F-02
được đóng. Một pipeline không chạy lại được hoặc còn numerical warning không
phải nền tảng đáng tin để so sánh model mới.

## 10. Release Gate đề xuất

Notebook chỉ nên được đánh dấu `Reproducible Baseline: Pass` khi đồng thời thỏa
các điều kiện sau:

- Restart Kernel + Run All hoàn tất từ Cell 1 đến Cell 88.
- Không có error output hoặc stderr warning chưa được giải thích.
- Import local modules thành công trong venv mới.
- Split vẫn là 54,000/6,000, disjoint và class-stratified.
- `TRAIN_MEAN`/`TRAIN_STD` chỉ được tính từ internal training indices.
- Mỗi experiment bắt đầu từ fresh model và optimizer state.
- Official test set không tham gia preprocessing, model selection hoặc gradient
  update.
- Checkpoint round trip giữ predictions và logits trong tolerance.
- Learning curves, experiment comparison, confusion matrix và prediction grid
  được regenerate trong cùng run.
- Result report khớp chính xác stored outputs của run đó.
- Dependency manifest có thể dựng lại một kernel chạy được.

## 11. Dữ liệu xác minh tại thời điểm audit

| Kiểm tra | Kết quả |
|---|---|
| Tổng số notebook cells | 89 |
| Code cells | 54 |
| Markdown cells | 35 |
| Execution count | Liên tục từ `In [1]` đến `In [54]` |
| Stored error outputs | 0 |
| Cells có stored stderr warnings | 4: Cell 27, 29, 32, 36 |
| Markdown image references | 11 |
| Missing referenced images | 0 |
| Final checkpoint | Tồn tại và load được bằng `weights_only=True` |
| Checkpoint selected experiment | `E1_deeper` |
| Checkpoint best epoch | 7 |
| Checkpoint test accuracy | 0.8879 |
| Missing runtime package | `practice_1.processing_own_phase` |

Audit này phản ánh chính xác source, stored output và filesystem tại ngày đối
chiếu. Khi cell được thêm, xóa hoặc đổi thứ tự, các deep-link bên dưới phải được
đồng bộ lại.

[cell-0]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=0>
[cell-1]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=1>
[cell-3]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=3>
[cell-4]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=4>
[cell-5]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=5>
[cell-7]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=7>
[cell-8]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=8>
[cell-9]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=9>
[cell-10]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=10>
[cell-11]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=11>
[cell-12]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=12>
[cell-13]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=13>
[cell-15]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=15>
[cell-17]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=17>
[cell-23]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=23>
[cell-27]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=27>
[cell-29]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=29>
[cell-32]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=32>
[cell-36]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=36>
[cell-38]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=38>
[cell-41]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=41>
[cell-43]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=43>
[cell-47]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=47>
[cell-49]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=49>
[cell-51]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=51>
[cell-53]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=53>
[cell-54]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=54>
[cell-57]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=57>
[cell-58]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=58>
[cell-59]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=59>
[cell-62]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=62>
[cell-63]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=63>
[cell-64]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=64>
[cell-65]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=65>
[cell-67]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=67>
[cell-69]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=69>
[cell-70]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=70>
[cell-72]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=72>
[cell-73]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=73>
[cell-74]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=74>
[cell-76]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=76>
[cell-77]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=77>
[cell-79]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=79>
[cell-80]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=80>
[cell-81]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=81>
[cell-82]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=82>
[cell-83]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=83>
[cell-84]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=84>
[cell-86]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=86>
[cell-87]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=87>
[cell-88]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=88>
