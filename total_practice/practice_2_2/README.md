# Practice 2.2 — Custom Image Crawl & Data Pipeline

Pipeline độc lập cho **Practice 2**: crawl ảnh mỹ phẩm từ web → làm sạch → cân bằng → xuất dataset **sẵn sàng cho Transfer Learning**.

> **Dataset cuối**: `data_clean_balanced/` — 3,202 files, 10 classes, 224×224 RGB JPEG. File counts nearly balanced (nine classes có 320 files, `facial_cleanser` có 322). Tuy nhiên dataset thực chất chỉ có **2,896 source groups độc lập + 306 ảnh offline-augmented**, nên không nên mô tả là 3,202 ảnh độc lập hay cân bằng tuyệt đối ở cấp source.

---

## 📁 Cấu trúc dự án

```
practice_2_2/
├── craw/                          # GIAI ĐOẠN 1 — Thu thập ảnh thô
│   ├── __init__.py
│   ├── config.py                  # Cấu hình DuckDuckGo / Google
│   ├── config_tiki.py             # Cấu hình Tiki (10 lớp mỹ phẩm)
│   ├── crawl_duckduckgo.py        # Crawl DuckDuckGo (free, no API key)
│   ├── crawl_google.py            # Crawl Google Images (cần Chrome + Selenium)
│   ├── crawl_tiki.py              # ★ Crawl Tiki.vn (dùng cho project này)
│   └── README.md
│
├── data_processing/               # GIAI ĐOẠN 2 — Làm sạch & chuẩn hoá
│   ├── __init__.py
│   ├── clean_data.py              # Bước 1: Xoá ảnh lỗi / kích thước quá nhỏ
│   ├── deduplicate.py             # Bước 2: Loại ảnh trùng bằng MD5
│   ├── quality_check.py           # Bước 3: Lọc ảnh mờ / low-content
│   ├── resize_data.py             # Bước 4: Resize về 224×224 (crop/pad/stretch)
│   ├── pipeline.py                # Chạy tổng hợp 4 bước trên
│   └── README.md
│
├── data_clean/                    # Thư mục trung gian (chỉ giữ .gitkeep)
│   └── .gitkeep
│
├── data_clean_balanced/           # ★ DATASET CUỐI — dùng cho training
│   ├── body_wash/        (320 ảnh)
│   ├── face_mask/        (320 ảnh)
│   ├── facial_cleanser/  (322 ảnh)
│   ├── lipstick/         (320 ảnh)
│   ├── moisturizer/      (320 ảnh)
│   ├── perfume/          (320 ảnh)
│   ├── serum/            (320 ảnh)
│   ├── shampoo/          (320 ảnh)
│   ├── sunscreen/        (320 ảnh)
│   └── toner/            (320 ảnh)
│
├── requirements.txt
└── README.md
```

---

## 🔄 Chu trình Pipeline (đã chạy thực tế)

### Sơ đồ tổng quan

```
┌─────────────────┐    ┌──────────────────────┐    ┌──────────────────┐
│ 1. CRAWL (raw)  │ →  │ 2. PROCESS (clean)   │ →  │ 3. BALANCE       │
│ Tiki.vn API     │    │ 4 bước pipeline      │    │ Offline augment  │
│ 10 lớp, ~400/lớp│    │ data/  →  data_clean │    │ data_clean →     │
└─────────────────┘    └──────────────────────┘    │ data_clean_balanced│
   ↓                       ↓                          └──────────────────┘
 data/                  data_clean/                      ↓
 (raw, duplicate)       (sạch, imbalanced)         data_clean_balanced/
                                                    (320/lớp, ready)
```

### Chi tiết từng bước

#### **Giai đoạn 1 — Crawl** (`craw/crawl_tiki.py`)

Dùng Tiki internal API, search 10 query tiếng Việt → lấy `images[*].base_url` của từng sản phẩm.

```bash
python -m craw.crawl_tiki
```

**Output thực tế** (`data/`):

| Lớp | Query | Ảnh thô |
|---|---|---|
| facial_cleanser | "sữa rửa mặt" | 322 |
| sunscreen | "kem chống nắng" | 318 |
| body_wash | "sữa tắm" | 303 |
| serum | "serum" | 300 |
| face_mask | "mặt nạ" | 299 |
| toner | "nước hoa hồng" | 292 |
| perfume | "nước hoa" | 287 |
| lipstick | "son môi" | 286 |
| shampoo | "dầu gội đầu" | 274 |
| moisturizer | "kem dưỡng ẩm" | 241 |
| **Tổng** | | **~2,922** |

#### **Giai đoạn 2 — Data Processing Pipeline** (`data_processing/pipeline.py`)

4 bước chạy nối tiếp trong cùng folder `data/` → copy kết quả sạch sang `data_clean/`:

```bash
python -m data_processing.pipeline
```

| Bước | Script | Hành động | Tiêu chí |
|---|---|---|---|
| **1. Clean** | `clean_data.py` | Xoá ảnh corrupt, kích thước quá nhỏ | min 64×64 |
| **2. Dedupe** | `deduplicate.py` | Xoá ảnh trùng MD5 | exact match |
| **3. Quality** | `quality_check.py` | Xoá ảnh mờ / nội dung rỗng | Laplacian var ≥ 30, content ratio ≥ 10% |
| **4. Resize** | `resize_data.py` | Resize về 224×224 + copy sang `data_clean/` | crop center |

**Kết quả thực tế** (`data_clean/`):

- Tổng: **~2,896 ảnh** (sau khi loại bỏ 26 ảnh trùng MD5)
- Ảnh sắc nét (sharpness trung bình: 2,000–3,000 Laplacian variance)
- Kích thước chuẩn **224×224** RGB JPEG
- Phân bố vẫn lệch: `moisturizer` chỉ 241 ảnh, các lớp khác 274–322

#### **Giai đoạn 3 — Balance via Augmentation** (`quality_check_extra.py` + `dedup_and_balance.py`)

> **Lưu ý**: Hai script này (`dedup_and_balance.py`, `quality_check_extra.py`) đã được **xoá sau khi chạy xong** để giữ repo gọn. Logic của chúng vẫn được tài liệu hoá ở đây để có thể tái tạo.

**Bước 3a — Deep EDA** (`quality_check_extra.py`):

- MD5 scan: phát hiện **18 nhóm trùng lặp** (26 ảnh)
- Laplacian variance: đánh giá độ sắc nét
- Size + aspect ratio: phát hiện ảnh nhỏ bất thường
- Output: `quality_report.csv`, `quality_summary.csv`, `quality_heatmap.png`

**Bước 3b — Dedupe + Balance** (`dedup_and_balance.py`):

1. **Dedupe**: copy `data_clean/` → `data_clean_dedup/`, bỏ MD5 trùng (giữ 2,896 ảnh).
2. **Augment**: tăng mỗi lớp lên **320 ảnh** bằng augmentation nhẹ:
   - `RandomHorizontalFlip(p=0.5)`
   - `RandomRotation(15°)`
   - `ColorJitter(brightness/contrast/saturation=0.2, hue=0.05)`
   - `RandomResizedCrop(224, scale=(0.85, 1.0))`
3. Output: `data_clean_balanced/` (320 ảnh/lớp × 10 = **3,202 ảnh**, với `facial_cleanser` có 322 file)

**Số ảnh augment thêm vào mỗi lớp:**

| Lớp | Trước | Sau | Augment |
|---|---|---|---|
| moisturizer | 228 | 320 | **+92** ← yếu nhất |
| shampoo | 274 | 320 | +46 |
| toner | 281 | 320 | +39 |
| lipstick | 286 | 320 | +34 |
| perfume | 287 | 320 | +33 |
| face_mask | 297 | 320 | +23 |
| serum | 300 | 320 | +20 |
| body_wash | 303 | 320 | +17 |
| sunscreen | 318 | 320 | +2 |
| facial_cleanser | 322 | 322 | +0 (đã đủ) |

### Tổng kết pipeline

```
data/              2,922 ảnh thô từ Tiki
  ↓ pipeline.py (clean + dedupe + quality + resize)
data_clean/        2,896 ảnh sạch 224×224
  ↓ dedup MD5
data_clean_dedup/  2,896 ảnh (đã dedup)   ← folder trung gian
  ↓ augment về 320/lớp
**Δ phân bố trước → sau:** min=241, max=322 (Δ=81) → min=320, max=322 (Δ=2) — cân bằng **40× tốt hơn**.

### Data audit và leakage-safe use

Audit tái lập được viết tại `practice_2/processing_own_phase/audit_practice_2_2_data.py`. Kết quả đã verify nằm ở `practice_2/outputs/practice_2_2/data_audit.json`:

- 3,202 ảnh RGB readable, tất cả 224×224; không có file corrupt.
- 2,896 source groups và 306 file offline-generated.
- Không tìm thấy exact duplicate ở decoded pixel. Tuy nhiên, các group dựa trên filename không đáng tin làm product identity: audit thị giác tìm thấy 16 cặp cross-split gần giống pixel-hoàn-toàn ở MAE ≤ 1 (và 255 candidate review rộng hơn ở MAE ≤ 5). Vì vậy split và metrics cũ cần coi là provisional cho tới khi perceptual duplicate clusters được gán trước khi split và model được train lại.
- Split là group-aware: Train 2,028 groups, Validation 434 groups, Test 434 groups.
- Validation và Test chỉ chứa originals. Historical Train policy chứa 2,028 originals + 198 file offline-generated mà source thuộc Train.

**Quy tắc split:**
- Split phải được gán từ source groups **trước khi** attach transforms.
- Augmentation ngẫu nhiên chỉ được dùng ở Train; Validation và Test dùng deterministic transform.
- File offline-generated phải ở cùng split với ảnh source.
- Crawled files phải được cluster bằng perceptual-duplicate rule đã review; filename stems không đủ vì ảnh sản phẩm lặp lại có thể khác filename.

Original-only Train set với inverse-frequency sampling cũng được đánh giá làm controlled experiment (best Val Acc = 78.11%), document nhưng không chọn. Fix audit giữ group-safe Train set hiện tại và freeze running statistics chỉ cho BatchNorm layers thuộc backbone đã đóng băng.

> **Hạn chế reproducibility**: `dedup_and_balance.py` và `quality_check_extra.py` đã bị xoá sau khi sinh dataset. File cuối có thể audit, nhưng offline-generation step chính xác không reproduce được từ repo này. Version tương lai nên phục hồi script đó hoặc thay offline balancing bằng versioned online augmentation.

---

## 🚀 Cách chạy lại toàn bộ từ đầu

```bash
# 1. Cài thư viện
pip install -r requirements.txt

# 2. Crawl ảnh thô từ Tiki
python -m craw.crawl_tiki

# 3. Chạy pipeline xử lý 4 bước
python -m data_processing.pipeline

# 4. (Tái tạo) Phân tích chất lượng + dedup + balance
#    Cần đặt lại file dedup_and_balance.py nếu muốn chạy
#    hoặc viết script riêng theo logic trong README này.
```

---

## 🎯 Nguồn crawl thay thế

| Nguồn | File | Cần gì | Tốc độ | Chất lượng |
|---|---|---|---|---|
| **Tiki.vn** ★ | `crawl_tiki.py` | Chỉ `pip` | Vừa (rate-limit 2s) | Rất cao (sản phẩm chuẩn pose) |
| **DuckDuckGo** | `crawl_duckduckgo.py` | Chỉ `pip` | Nhanh | Trung bình |
| **Google Images** | `crawl_google.py` | Chrome + Selenium | Chậm | Cao |

```bash
python -m craw.crawl_duckduckgo    # Tổng quát
python -m craw.crawl_google        # Cần Chrome
python -m craw.crawl_tiki          # ★ Đã dùng cho project
```

---

## 📦 Đầu ra — Dùng cho Practice 2

Dataset `data_clean_balanced/` đã sẵn sàng:

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(
    root="total_practice/practice_2_2/data_clean_balanced",
    transform=transform,
)
print(f"Classes: {dataset.classes}")    # 10 lớp
print(f"Total:   {len(dataset)}")       # 3,202 ảnh
```

Trỏ `DATA_DIR` trong `practice_2/configs/core_config.py` đến `data_clean_balanced/` rồi train.

---

## ⚠️ Lưu ý ToS

- Crawl Tiki tuân thủ `REQUEST_DELAY_SEC=2.0` để tránh bị block.
- Ảnh chỉ dùng cho mục đích **học tập / nghiên cứu**, không thương mại hoá.
