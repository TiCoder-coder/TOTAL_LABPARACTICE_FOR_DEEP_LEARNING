# Crawl Module

Thu thập ảnh từ web theo từng lớp (mỗi lớp 1 folder).

## Cài đặt

```bash
pip install -r ../requirements.txt
```

## Cấu hình

| File | Dùng cho | Nội dung |
|---|---|---|
| `config.py` | DuckDuckGo / Google | `CLASSES`, `IMAGES_PER_CLASS`, `OUTPUT_DIR` |
| `config_tiki.py` | Tiki.vn | `CLASSES` 10 lớp mỹ phẩm, `REQUEST_DELAY_SEC`, headers |

## Chạy

```bash
# DuckDuckGo (mặc định, free, không cần key)
python -m craw.crawl_duckduckgo

# Google Images (cần Chrome + Selenium)
python -m craw.crawl_google

# Tiki.vn (chủ đề mỹ phẩm, dùng config_tiki.py)
python -m craw.crawl_tiki
```

## So sánh

| | DuckDuckGo | Google | Tiki |
|---|---|---|---|
| **Cần API key** | Không | Không | Không |
| **Cần Chrome** | Không | Có | Không |
| **Tốc độ** | Nhanh | Chậm | Vừa (rate-limit) |
| **Chất lượng** | Trung bình | Cao (lẫn nhiều rác) | Rất cao (ảnh sản phẩm) |
| **Phù hợp** | Đa chủ đề | Đa chủ đề | Hàng hoá VN |

## Output

Sau khi chạy, folder `data/` chứa các thư mục con, mỗi folder = 1 lớp:

```
data/
├── lipstick/
│   ├── lipstick_00000.jpg
│   ├── lipstick_00001.jpg
│   └── ...
├── sunscreen/
└── ...
```

Bước tiếp theo: chạy `data_processing/pipeline.py` để làm sạch & resize.

## Cảnh báo ToS

- **DuckDuckGo**: Không có public ToS cấm crawl, nhưng nên có delay.
- **Google**: ToS cho phép index nhưng không cho phép bulk download.
- **Tiki**: ToS nghiêm ngặt — chỉ dùng cho mục đích học tập, tôn trọng rate limit.