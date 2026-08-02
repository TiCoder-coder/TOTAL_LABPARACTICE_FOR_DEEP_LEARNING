# Practice 2.2 — Hướng dẫn chạy end-to-end

Pipeline tổng thể:

```
crawl → clean + dedupe (intra + cross-class) + quality check → resize → notebook (split + train + eval)
```

---

## 0. Môi trường

Yêu cầu: **Python 3.10+**, **Windows PowerShell** (hoặc CMD / Git Bash).

### 0.1. Tạo & kích hoạt virtualenv

```powershell
cd D:\Hoc_tap\TOTAL_LABPARACTICE_FOR_DEEP_LEARNING\total_practice\practice_2_2
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Nếu gặp lỗi *"running scripts is disabled on this system"*, mở PowerShell as Administrator và chạy:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

rồi activate lại.

**CMD:**

```cmd
.venv\Scripts\activate.bat
```

**Git Bash / WSL:**

```bash
source .venv/Scripts/activate
```

### 0.2. Cài dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Deactivate khi xong: `deactivate`

---

## 1. Crawl dữ liệu (tuỳ chọn — bỏ qua nếu đã có `data/`)

Folder `craw/` chứa 3 spider:

| Script | Nguồn | Ghi chú |
|---|---|---|
| `crawl_tiki.py` | Tiki.vn | Nhanh, ổn định, có cấu hình sẵn |
| `crawl_google.py` | Google Images | Cần proxy/cookie ổn định |
| `crawl_duckduckgo.py` | DuckDuckGo | Không cần API key |

```powershell
python -m craw.crawl_tiki
python -m craw.crawl_google
python -m craw.crawl_duckduckgo
```

Mỗi script tạo ra cây thư mục `data/<class_name>/*.jpg`. Mặc định target ~10 lớp:
`body_wash`, `face_mask`, `facial_cleanser`, `lipstick`, `moisturizer`,
`perfume`, `serum`, `shampoo`, `sunscreen`, `toner`.

---

## 2. Chạy pipeline xử lý dữ liệu

Pipeline tự động làm **5 bước** trên folder `data/` và xuất ra `data_clean/`:

| # | Bước | Mô tả |
|---|---|---|
| 1 | **Clean** | Xoá ảnh corrupt / kích thước quá nhỏ (<64px) |
| 2 | **Dedupe intra-class** | Xoá file trùng MD5 trong cùng class |
| 3 | **Dedupe cross-class** | Phát hiện ảnh trùng MD5 ở **nhiều class khác nhau** và xoá (giữ class nhiều ảnh hơn) |
| 4 | **Quality check** | Lọc ảnh mờ (Laplacian variance < 30) hoặc nội dung quá ít |
| 5 | **Resize** | Resize về `224×224` và copy sang `data_clean/` |

```powershell
# Chạy mặc định (src=data, dst=data_clean, size=224, mode=crop)
python -m data_processing.pipeline

# Tùy chỉnh
python -m data_processing.pipeline --src data --dst data_clean_224 --size 224 --mode crop

# Ưu tiên giữ toner khi có xung đột cross-class với facial_cleanser
python -m data_processing.pipeline --prefer toner
```

### 2.1. Output mẫu

```
============================================================
DATA PROCESSING PIPELINE
  src = D:\...\practice_2_2\data
  dst = D:\...\practice_2_2\data_clean_224
  target_size = (224, 224), mode = crop
============================================================

[1/5] Cleaning corrupt / too-small images...
  done in 12.3s — removed 14 files

[2/5] Removing duplicate images (intra-class)...
  done in 4.1s — removed 9 duplicates

[3/5] Detecting & removing cross-class duplicates (MD5)...
  ⚠ done in 3.7s — found 11 leak groups, removed 11 files
    hash b705a07990... present in: ['facial_cleanser', 'toner']
    hash 34780eb88d... present in: ['facial_cleanser', 'toner']
    ...

[4/5] Filtering blurry / low-content images...
  done in 21.4s — removed 38 files

[5/5] Resizing images to clean folder...
  done in 9.8s — 2857 resized, 0 failed

============================================================
PIPELINE COMPLETE
============================================================
```

### 2.2. Chạy riêng từng bước (debug)

```powershell
python -m data_processing.clean_data      --root data
python -m data_processing.deduplicate     --root data                  # intra-class
python -m data_processing.deduplicate     --root data --cross-class    # cross-class
python -m data_processing.quality_check   --root data
python -m data_processing.resize_data     --src data --dst data_clean
```

Thêm `--dry-run` để xem trước không xoá.

---

## 3. Chạy notebook preprocessing

Notebook `practice_2_2_preprocessing.ipynb` (đã tạo sẵn ở `total_practice/practice_2/notebooks/`) sẽ:

1. Setup environment + kiểm tra device (CUDA / MPS / CPU)
2. Verify MD5 integrity (đảm bảo 0 leak, 0 dup)
3. Chia stratified group-aware split 70 / 15 / 15
4. EDA: phân bố class, sample ảnh
5. Định nghĩa augmentation pipeline + tạo DataLoader
6. Export `preprocessing_split.json` cho notebook training

```powershell
# Mở Jupyter
jupyter notebook "D:\Hoc_tap\TOTAL_LABPARACTICE_FOR_DEEP_LEARNING\total_practice\practice_2\notebooks\practice_2_2_preprocessing.ipynb"
```

Hoặc mở trực tiếp trong VS Code / Cursor.

---

## 4. Train & evaluate

Mở file notebook training (xem thư mục `notebooks/`), chạy tuần tự các cell.
Safety gates trong notebook:

- `RUN_TRAINING = False` — không train cho đến khi bạn đổi thành `True`
- `RUN_FINAL_TEST = False` — không đánh giá test cho đến khi sẵn sàng
- Best checkpoint chọn trên **validation**, test chỉ chạy 1 lần cuối

Checkpoints & history được ghi vào:

```
runs/practice_2_2/<experiment_name>/best.pt
outputs/practice_2_2/controlled_training_history.json
reports/practice_2_2/*.png
```

---

## 5. Troubleshooting

| Lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| `No module named craw` / `data_processing` | Chạy ngoài folder `practice_2_2` | `cd` vào folder rồi chạy `python -m ...` |
| `Dataset directory not found: ...` | Path `data_clean_224` không tồn tại | Chạy pipeline trước (mục 2) |
| `AssertionError: cross-class leaks detected` | Có ảnh trùng MD5 ở 2 class | Đã được pipeline tự xử lý. Nếu vẫn còn, kiểm tra lại `--prefer` |
| `RuntimeError: CUDA out of memory` | Batch size quá lớn | Giảm `--batch-size` trong notebook (mặc định 32 → 16 hoặc 8) |
| Pipeline chạy chậm | Hash 2,857 ảnh ~3–5 giây | Bình thường. Dùng `--dry-run` để test |
| `Set-ExecutionPolicy` bị chặn | Policy hệ thống | Mở PowerShell as Admin → `Set-ExecutionPolicy RemoteSigned` |
| Notebook không thấy GPU | `torch.cuda.is_available() = False` | Cài `torch` bản CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118` |

---

## 6. Cấu trúc thư mục

```
practice_2_2/
├── craw/                       # spiders
│   ├── crawl_tiki.py
│   ├── crawl_google.py
│   ├── crawl_duckduckgo.py
│   └── config.py
├── data/                       # raw crawled (input)
│   ├── body_wash/
│   ├── facial_cleanser/
│   └── ...
├── data_clean/                 # default pipeline output
├── data_clean_224/             # current clean dataset (output)
├── data_processing/            # pipeline modules
│   ├── clean_data.py
│   ├── deduplicate.py          # ← đã thêm cross-class detection
│   ├── quality_check.py
│   ├── resize_data.py
│   └── pipeline.py             # ← đã thêm step 3
├── outputs/practice_2_2/       # JSON / CSV kết quả
├── reports/practice_2_2/       # biểu đồ
├── runs/practice_2_2/          # checkpoints
├── requirements.txt
├── QUICKSTART.md
└── README.md
```

Notebook liên quan (ở `total_practice/practice_2/notebooks/`):
- `practice_2_2_preprocessing.ipynb` — preprocessing & split (12 cells)
- `practice_2_2.ipynb` — notebook training gốc (transfer learning ResNet18)