# Data Processing Module

Làm sạch & chuẩn hoá ảnh đã crawl, xuất ra folder `data_clean/`.

## Cài đặt

```bash
pip install Pillow numpy tqdm
```

## Cấu trúc scripts

| File | Chức năng |
|---|---|
| `clean_data.py` | Xoá ảnh lỗi, không đọc được, kích thước quá nhỏ |
| `deduplicate.py` | Loại ảnh trùng (MD5 hash) |
| `quality_check.py` | Lọc ảnh mờ (Laplacian variance) / không có nội dung |
| `resize_data.py` | Resize về 224×224, hỗ trợ crop / pad / stretch |
| `pipeline.py` | Chạy tổng hợp 4 bước trên |

## Chạy từng bước

```bash
python -m data_processing.clean_data --root data
python -m data_processing.deduplicate --root data
python -m data_processing.quality_check --root data
python -m data_processing.resize_data --src data --dst data_clean --size 224 --mode crop
```

## Chạy tổng hợp

```bash
python -m data_processing.pipeline
# Tương đương:
python -m data_processing.pipeline --src data --dst data_clean --size 224 --mode crop
```

## Tham số

- `--min-width`, `--min-height`: ngưỡng kích thước tối thiểu (mặc định 64×64)
- `--blur-threshold`: ngưỡng Laplacian variance (mặc định 30.0)
- `--min-content-ratio`: tỉ lệ pixel "có nội dung" tối thiểu (mặc định 0.10)
- `--size`: kích thước đầu ra (mặc định 224)
- `--mode`: `crop` (cắt giữa), `pad` (thêm viền), `stretch` (méo ảnh)

## Output

```
data_clean/
├── airplane/
│   ├── airplane_00000.jpg
│   └── ...
├── car/
└── ...
```

Mỗi ảnh ~224×224 RGB JPEG, sẵn sàng cho `torchvision.datasets.ImageFolder`.
