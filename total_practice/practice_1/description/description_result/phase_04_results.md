# Phase 4 - Exploratory Data Analysis Outputs

[Phase 3](phase_03_results.md) | [Mục lục](README.md) | [Phase 5](phase_05_results.md)

| Cell output | Mô tả ngắn | Liên kết |
|---|---|---|
| Cell 11, `In [6]` | Schema của 60,000 ảnh: raw/transformed shape, dtype, pixel range, label range và 10 classes. | [Mở output Cell 11][cell-11] |
| Cell 13, `In [7]` | Class counts xác nhận mỗi lớp có 6,000 ảnh, tương đương 10% training pool. | [Mở output Cell 13][cell-13] |
| Cell 14 | Bar chart class distribution đã lưu trong `outputs/eda_class_distribution.png`. | [Mở output Cell 14][cell-14] |
| Cell 15, `In [8]` | Figure gồm class-distribution pie chart và sampled pixel-value histogram. | [Mở output Cell 15][cell-15] |
| Cell 17, `In [9]` | Thống kê 47,040,000 pixel: mean, std, median, quantiles và tỷ lệ pixel biên. | [Mở output Cell 17][cell-17] |
| Cell 18 | Pixel-intensity distribution đã lưu trong `outputs/pixel_intensity_distribution.png`. | [Mở output Cell 18][cell-18] |
| Cell 20, `In [10]` | Brightness/contrast toàn tập và giá trị trung bình riêng cho 10 classes. | [Mở output Cell 20][cell-20] |
| Cell 21 | Hai artifact về brightness/contrast tổng thể và per-class boxplots. | [Mở output Cell 21][cell-21] |
| Cell 23, `In [11]` | Data-quality audit; corruption, invalid label và duplicate-conflict counts đều bằng 0. | [Mở output Cell 23][cell-23] |
| Cell 24, `In [12]` | Correlation heatmap của 100 pixel đầu tiên. | [Mở output Cell 24][cell-24] |
| Cell 25, `In [13]` | Correlation heatmap của 100 pixel positions lấy mẫu với seed 42. | [Mở output Cell 25][cell-25] |
| Cell 27, `In [15]` | Runtime warnings đang lưu và 3D PCA projection theo class. | [Mở output Cell 27][cell-27] |
| Cell 28, `In [16]` | Explained variance của PC1, PC2 và PC3 lần lượt là 29.03%, 17.76% và 6.02%. | [Mở output Cell 28][cell-28] |
| Cell 29, `In [17]` | Runtime warnings đang lưu và heatmaps của ba principal-component images. | [Mở output Cell 29][cell-29] |
| Cell 32, `In [20]` | Runtime warnings đang lưu và cumulative explained-variance curve của full PCA. | [Mở output Cell 32][cell-32] |
| Cell 33, `In [21]` | Số components cần cho 90% và 95% variance lần lượt là 137 và 256. | [Mở output Cell 33][cell-33] |
| Cell 34, `In [22]` | Scree plot của explained variance ratio theo principal component. | [Mở output Cell 34][cell-34] |
| Cell 36, `In [24]` | Runtime warnings đang lưu và t-SNE scatter plot hai chiều theo class. | [Mở output Cell 36][cell-36] |
| Cell 39 | Hai artifact: class-stratified sample grid và class-average images. | [Mở output Cell 39][cell-39] |
| Cell 42 | Artifact hiển thị các ảnh cực trị theo brightness và contrast. | [Mở output Cell 42][cell-42] |

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
[cell-39]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=39>
[cell-42]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=42>
