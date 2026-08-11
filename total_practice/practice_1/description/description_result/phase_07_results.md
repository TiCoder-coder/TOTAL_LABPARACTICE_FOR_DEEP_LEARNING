# Phase 7 - Model Training Outputs

[Phase 6](phase_06_results.md) | [Mục lục](README.md) | [Phase 8](phase_08_results.md)

> Provenance: notebook hiện lưu clean-kernel staged-search run
> `20260811-010514`; execution count liên tục `1-57`, không có error output.

| Cell output | Mô tả ngắn | Liên kết |
|---|---|---|
| Cell 62, `In [35]` | Device được chọn cho training là Apple MPS. | [Mở output Cell 62][cell-62] |
| Cell 70, `In [41]` | E0-E4 completed-resume và Stage A; coarse best learning rate là `0.001`. | [Mở Cell 70][cell-70] |
| Cell 72, `In [42]` | Controlled summary và Stage B; best architecture là `(512, 256, 128)`. | [Mở Cell 72][cell-72] |
| Cell 73, `In [43]` | Stage C với hai refined rates và comparison A/B/C đã lưu PNG. | [Mở Cell 73][cell-73] |
| Cell 74, `In [44]` | Stage D: selected `(512, 256, 128)`, lr `0.001`, stored output mean val accuracy `89.78% ± 0.08%`, median epoch 32. | [Mở Cell 74][cell-74] |
| Cell 77, `In [46]` | Fresh final run 32 epoch trên 60,000 ảnh; final train accuracy `96.82%`. | [Mở output Cell 77][cell-77] |
| Cell 78, `In [47]` | TensorBoard HTML output được nhúng trong notebook. | [Mở output Cell 78][cell-78] |

[cell-62]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=62>
[cell-70]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=70>
[cell-72]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=72>
[cell-73]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=73>
[cell-74]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=74>
[cell-77]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=77>
[cell-78]: <vscode://ticoder.practice1-notebook-links/open-cell?notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=78>
