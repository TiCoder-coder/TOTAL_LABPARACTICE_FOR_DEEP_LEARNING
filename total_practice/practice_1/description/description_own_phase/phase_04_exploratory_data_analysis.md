# Phase 4 - Exploratory Data Analysis

[Phase 3](phase_03_data_loading.md) | [Mục lục](README.md) | [Mở đúng Cell 8][cell-8] | [Notebook dự phòng](../../practice_1.ipynb) | [Phase 5](phase_05_data_preprocessing.md)

## 1. Vị trí và phạm vi

- Notebook cells: `8` đến `43`.
- Code cells đã chạy: `In [5]` đến `In [26]`.
- Tập được phân tích: toàn bộ 60,000 ảnh thuộc official training pool.
- Tập không tham gia EDA: 10,000 ảnh và nhãn thuộc official test set.
- Mục đích: hiểu schema, phân phối nhãn, phân phối pixel, chất lượng dữ liệu và
  cấu trúc thị giác trước khi quyết định preprocessing/modeling.

| Nhóm nội dung | Code và output chính xác trong notebook |
|---|---|
| EDA setup và raw-data contract | [Cell 10, `In [5]`][cell-10]; [Cell 11, `In [6]` + output][cell-11] |
| Class distribution | [Cell 13, `In [7]` + output][cell-13]; [Cell 14, hình đã render][cell-14]; [Cell 15, `In [8]` + output][cell-15] |
| Pixel distribution | [Cell 17, `In [9]` + output][cell-17]; [Cell 18, hình đã render][cell-18] |
| Brightness và contrast | [Cell 20, `In [10]` + output][cell-20]; [Cell 21, hình đã render][cell-21] |
| Quality và duplicates | [Cell 23, `In [11]` + output][cell-23] |
| Pixel correlation | [Cell 24, `In [12]` + output][cell-24]; [Cell 25, `In [13]` + output][cell-25] |
| PCA ba chiều | [Cell 27, `In [15]` + output][cell-27]; [Cell 28, `In [16]` + output][cell-28]; [Cell 29, `In [17]` + output][cell-29] |
| Standardized full PCA | [Cell 32, `In [20]` + output][cell-32]; [Cell 33, `In [21]` + output][cell-33]; [Cell 34, `In [22]` + output][cell-34] |
| t-SNE | [Cell 36, `In [24]` + output][cell-36] |
| Class samples và mean images | [Cell 38, `In [25]`][cell-38]; [Cell 39, hình đã render][cell-39] |
| Statistical extremes | [Cell 41, `In [26]`][cell-41]; [Cell 42, hình đã render][cell-42] |
| EDA conclusions | [Cell 43][cell-43] |

Tất cả thống kê tại đây mang tính mô tả. Mean/std của EDA không được tái sử dụng
để normalize model; Phase 5 tính lại mean/std chỉ từ 54,000 internal training
images sau khi split.

## 2. Input và representation

EDA nhận ba input chính từ Phase 3:

| Input | Nội dung |
|---|---|
| `train_dataset.data` | Raw images, shape `[60000, 28, 28]`, dtype `torch.uint8` |
| `train_dataset.targets` | 60,000 integer labels trong `[0, 9]` |
| `class_names` | Mapping 10 label sang tên lớp |

Notebook import hàm phân tích `analyze_training_pool` và các hàm vẽ từ:

```text
practice_1.processing_own_phase.data
practice_1.processing_own_phase.visualize
```

`analyze_training_pool(...)` trả về `EDA_ANALYSIS`. Trong đó:

- `EDA_ANALYSIS['report']`, được gán thành `EDA_SUMMARY`, chứa các thống kê có
  thể serialize và in báo cáo;
- các array như histogram, per-image mean/std, class mean và sample indices được
  giữ trong `EDA_ANALYSIS` để trực quan hóa;
- `scope` được assert bằng `official_training_pool_only`;
- kích thước `test_dataset` chỉ được assert là 10,000, không dùng nội dung test
  trong EDA.

Để chạy lại cell import từ kernel sạch, các module helper trên phải tồn tại và
import được trong project environment.

[Mở implementation EDA scope tại Cell 10][cell-10] và
[schema output tại Cell 11][cell-11].

## 3. Kiểm tra schema và raw data contract

Cell `In [6]` xác nhận dữ liệu có đúng format mà các phase sau mong đợi:

| Thuộc tính | Output hiện tại |
|---|---:|
| Số sample | `60,000` |
| Raw shape | `(60000, 28, 28)` |
| Shape sau `ToTensor()` | `(1, 28, 28)` |
| Raw dtype | `torch.uint8` |
| Raw range | `[0, 255]` |
| Label range | `[0, 9]` |
| Số lớp | `10` |

Assertions dừng notebook sớm nếu số sample, chiều ảnh, số channel, dtype, pixel
range hoặc label range sai. Đây là bước bảo vệ khỏi việc train trên dataset hỏng
hoặc một representation khác với thiết kế.

## 4. Class distribution

[Mở Class Distribution code/output tại Cell 13][cell-13].

### Phép tính chính xác

Helper dùng `torch.bincount` để đếm nhãn. Output xác nhận mỗi lớp có đúng:

- 6,000 ảnh;
- 10.0% official training pool.

Notebook assert mọi count đều bằng 6,000 và lưu bar chart tại:

```text
outputs/eda_class_distribution.png
```

Bar chart phù hợp cho kiểm tra chênh lệch count vì độ cao cột có thể so sánh trực
tiếp. Kết quả cân bằng hỗ trợ hai quyết định baseline:

- accuracy có thể dùng làm primary selection metric;
- chưa cần class weighting trong `CrossEntropyLoss`.

### Biểu đồ tổng hợp bổ sung

Cell `In [8]` tạo một DataLoader có batch size 60,000 để lấy toàn bộ transformed
training pool thành:

- `images`: `[60000, 1, 28, 28]`;
- `labels`: `[60000]`;
- `images_np`: `[60000, 28, 28]` sau `squeeze()`;
- `labels_np`: NumPy array dùng cho scikit-learn và Matplotlib.

Cell vẽ thêm pie chart class distribution và histogram của 1,000,000 pixel được
lấy mẫu không hoàn lại. Hai hình này cung cấp góc nhìn nhanh; thống kê histogram
chính xác trên toàn bộ pixel vẫn được thực hiện ở bước kế tiếp.

## 5. Pixel intensity distribution

[Mở Pixel Intensity Distribution code/output tại Cell 17][cell-17].

Helper xây histogram đủ 256 mức trên toàn bộ:

```text
60,000 x 28 x 28 = 47,040,000 pixel
```

Output hiện tại:

| Thống kê | Giá trị |
|---|---:|
| Mean | `0.286041` |
| Standard deviation | `0.353024` |
| Median | `0.000000` |
| Q05 | `0.000000` |
| Q95 | `0.905882` |
| Tỷ lệ pixel bằng 0 | `50.21%` |
| Tỷ lệ pixel bằng 1 | `0.81%` |

Notebook kiểm tra tổng histogram đúng bằng 47,040,000 và transformed range nằm
trong `[0, 1]`. Figure được lưu ở:

```text
outputs/pixel_intensity_distribution.png
```

Khoảng một nửa pixel bằng 0 phản ánh nền đen rộng quanh vật thể. Phân phối phần
foreground vẫn rộng, vì vậy Phase 5 áp dụng global normalization thay vì giữ
nguyên miền `[0, 1]`.

## 6. Brightness và contrast theo ảnh

[Mở Image-Level Brightness and Contrast code/output tại Cell 20][cell-20].

Đối với từng ảnh:

- brightness được định nghĩa là mean của 784 pixel;
- contrast được định nghĩa là standard deviation của 784 pixel.

Output toàn tập:

| Thống kê | Giá trị |
|---|---:|
| Mean image brightness | `0.286041` |
| Mean image contrast | `0.320249` |

Notebook còn tính mean brightness/contrast cho từng lớp. Ví dụ, `Sandal` có
brightness thấp nhất trong output (`0.1367`), còn `Coat` có brightness cao
(`0.3853`). Sự khác biệt cho thấy silhouette và lượng foreground khác nhau giữa
các nhóm quần áo.

Hai artifact được tạo:

```text
outputs/image_brightness_contrast.png
outputs/per_class_intensity_boxplot.png
```

Notebook không dùng class-specific normalization vì class thật chưa được biết ở
thời điểm inference. Mọi lớp dùng chung mean/std lấy từ internal train.

## 7. Data quality và duplicate audit

[Mở Data Quality and Duplicate Audit code/output tại Cell 23][cell-23].

Audit kiểm tra:

1. Ảnh transformed có giá trị không hữu hạn hay không.
2. Ảnh toàn đen, toàn trắng hoặc mọi pixel giống nhau.
3. Label ngoài `[0, 9]` và class bị thiếu.
4. Raw image trùng hoàn toàn.
5. Một raw image trùng nhưng được gắn label xung đột.

Output hiện tại đều bằng `0` cho:

- non-finite images;
- black/white/constant images;
- invalid labels và missing classes;
- duplicate groups/samples;
- conflicting-label duplicate groups.

Notebook assert các điều kiện nghiêm trọng nhất: không non-finite image, không
invalid label, không missing class và không conflicting-label duplicate. Nếu có
candidate bất thường, thiết kế hiện tại chỉ báo cáo để review chứ không tự xóa.

## 8. Correlation giữa các pixel

[Mở correlation heatmap tại Cell 24][cell-24] và
[random-pixel correlation tại Cell 25][cell-25].

Notebook flatten mỗi ảnh thành vector 784 chiều:

```text
X_flatten.shape = [60000, 784]
```

Hai correlation heatmap được vẽ:

1. Correlation của 100 pixel đầu tiên.
2. Correlation của 100 vị trí pixel được lấy ngẫu nhiên với seed 42.

100 pixel đầu tập trung nhiều ở các hàng trên cùng của ảnh và có thể chứa nhiều
nền. Random subset phân tán vị trí trên toàn khung ảnh, bổ sung góc nhìn ít lệch
vị trí hơn. Correlation cho thấy pixel lân cận/cùng vùng hình thể không độc lập,
là động cơ để quan sát tiếp cấu trúc thấp chiều bằng PCA và t-SNE.

Hai heatmap này chỉ hiển thị inline; code hiện tại không lưu chúng thành file.

## 9. PCA ba chiều và principal-component images

[Mở PCA 3D tại Cell 27][cell-27], [explained variance tại Cell 28][cell-28] và
[principal-component images tại Cell 29][cell-29].

Notebook import `PCA` từ scikit-learn rồi fit `PCA(n_components=3)` trên
`X_flatten` chưa standardize.

### 3D projection

60,000 vector 784 chiều được chiếu xuống ba principal components và vẽ scatter
3D, tô màu theo 10 label. Output explained variance:

| Component | Explained variance |
|---|---:|
| PC1 | `29.03%` |
| PC2 | `17.76%` |
| PC3 | `6.02%` |

Ba component đầu giữ khoảng 52.81% variance, cho thấy có cấu trúc thấp chiều rõ
rệt nhưng ba chiều chưa đủ mô tả toàn bộ khác biệt ảnh.

### Component heatmaps

Mỗi vector loading dài 784 của PC1, PC2, PC3 được reshape về `28 x 28` và vẽ
heatmap. Hình này giúp diễn giải vùng pixel nào đóng góp dương/âm vào từng hướng
biến thiên chính.

Các PCA figure này được hiển thị inline và không được ghi ra `OUTPUT_DIR` trong
code hiện tại.

## 10. Standardized PCA toàn phần

[Mở full PCA tại Cell 32][cell-32], [component thresholds tại Cell 33][cell-33]
và [scree plot tại Cell 34][cell-34].

Notebook dùng `StandardScaler` để standardize 784 feature theo cột, tạo
`X_scaled`, rồi fit `PCA()` không giới hạn số component.

Hai biểu đồ được tạo:

- cumulative explained variance theo số component;
- scree plot của explained variance ratio từng component.

Kết quả đang lưu:

- cần `137` components để đạt ít nhất 90% cumulative variance;
- cần `256` components để đạt ít nhất 95% cumulative variance.

Phân tích này mô tả độ phức tạp nội tại của dữ liệu. Notebook không dùng PCA làm
preprocessing cho MLP; model ở Phase 6 vẫn nhận đủ 784 pixel đã normalize.

## 11. t-SNE visualization

[Mở t-SNE code và hai output tại Cell 36][cell-36].

Notebook chạy:

```text
TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
```

trên toàn bộ `X_scaled` và vẽ scatter 2D theo class label. t-SNE ưu tiên cấu trúc
lân cận, giúp quan sát cluster cục bộ và vùng chồng lấn giữa các lớp. Nó chỉ dùng
cho trực quan hóa, không tạo feature cho training và không tham gia model
selection.

Do chạy trên 60,000 mẫu với 784 feature, đây là một trong các cell tốn thời gian
và bộ nhớ nhất của EDA.

## 12. Class-stratified samples và mean images

[Mở plotting code tại Cell 38][cell-38] và
[hai hình đã render tại Cell 39][cell-39].

Helper chọn chính xác hai sample cho mỗi lớp với seed 42:

- tổng cộng 20 indices;
- tất cả index là duy nhất;
- mọi class đều được đại diện.

Artifacts:

```text
outputs/class_samples.png
outputs/class_mean_images.png
```

Sample grid cho phép kiểm tra ảnh thực tế và nhãn. Mean image làm nổi bật
silhouette chung của từng lớp, đồng thời cho thấy sự giống nhau giữa
T-shirt/top, Pullover, Coat và Shirt. Quan sát này tạo giả thuyết về confusion
được kiểm tra lại ở Phase 8.

## 13. Statistical extremes

[Mở plotting code tại Cell 41][cell-41] và
[hình statistical extremes tại Cell 42][cell-42].

Notebook hiển thị các ảnh:

- tối nhất;
- sáng nhất;
- contrast thấp nhất;
- contrast cao nhất.

Artifact được lưu tại:

```text
outputs/eda_outliers.png
```

Các ảnh extreme được xem là review candidates, không mặc định là dữ liệu hỏng.
Kết luận EDA cho biết chúng vẫn là garment dễ nhận biết nên được giữ lại.

## 14. Kết luận được chuyển sang preprocessing/modeling

[Mở EDA Conclusions tại đúng Cell 43][cell-43].

| Phát hiện | Quyết định downstream |
|---|---|
| 10 lớp cân bằng tuyệt đối | Dùng accuracy; không bật class weighting baseline |
| Pixel background bằng 0 chiếm khoảng 50% | Dùng global normalization |
| Brightness/contrast khác nhau theo class | Không normalize theo class vì class chưa biết khi inference |
| Không phát hiện corruption/duplicate conflict | Không xóa sample tự động |
| Upper-body silhouettes chồng lấn | Dự kiến Shirt/Pullover/Coat/T-shirt khó phân loại |
| Extreme samples vẫn hợp lệ | Giữ toàn bộ official training pool |
| EDA mean/std đọc 60,000 ảnh | Không tái sử dụng để fit preprocessing |

## 15. Flow hoàn chỉnh của phase

```text
raw train data + targets
    -> scope/schema assertions
    -> class count và class charts
    -> exact pixel histogram
    -> per-image/per-class brightness và contrast
    -> quality + duplicate audit
    -> pixel correlation heatmaps
    -> raw PCA 3D + PC heatmaps
    -> standardized full PCA + scree/cumulative variance
    -> t-SNE 2D
    -> deterministic class samples + mean images
    -> statistical extremes
    -> EDA conclusions
```

## 16. Đầu ra của phase

Phase 4 cung cấp:

- `EDA_ANALYSIS` và `EDA_SUMMARY` trong kernel;
- các assertion xác nhận data contract;
- các figure inline;
- bảy EDA image artifacts trong `outputs`;
- các quyết định định hướng Phase 5 mà không fit model parameter.

[Phase 5](phase_05_data_preprocessing.md) tiếp tục bằng cách chia official
training pool theo lớp, tính normalization từ internal train và tạo transform/
DataLoader riêng cho train, validation và test.

[cell-8]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=8>
[cell-10]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=10>
[cell-11]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=11>
[cell-13]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=13>
[cell-14]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=14>
[cell-15]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=15>
[cell-17]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=17>
[cell-18]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=18>
[cell-20]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=20>
[cell-21]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=21>
[cell-23]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=23>
[cell-24]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=24>
[cell-25]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=25>
[cell-27]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=27>
[cell-28]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=28>
[cell-29]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=29>
[cell-32]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=32>
[cell-33]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=33>
[cell-34]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=34>
[cell-36]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=36>
[cell-38]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=38>
[cell-39]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=39>
[cell-41]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=41>
[cell-42]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=42>
[cell-43]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=43>
