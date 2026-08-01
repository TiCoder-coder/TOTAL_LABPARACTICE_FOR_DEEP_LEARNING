"""Tiki crawl config — chủ đề mỹ phẩm.

Mỗi lớp = 1 query tìm kiếm trên Tiki. Folder sẽ được tạo theo key (tiếng Anh).
"""

# ─── 10 lớp mỹ phẩm ────────────────────────────────────────────────
# key -> search query (tiếng Việt để match tốt hơn với Tiki)
CLASSES = {
    "lipstick":     "son môi",
    "sunscreen":    "kem chống nắng",
    "facial_cleanser": "sữa rửa mặt",
    "perfume":      "nước hoa",
    "shampoo":      "dầu gội đầu",
    "body_wash":    "sữa tắm",
    "moisturizer":  "kem dưỡng ẩm",
    "face_mask":    "mặt nạ",
    "serum":        "serum",
    "toner":        "nước hoa hồng",
}

# Số ảnh tối đa mỗi lớp (sẽ lọc & dedupe sau).
IMAGES_PER_CLASS = 400

# Thư mục output cho ảnh thô.
OUTPUT_DIR = "data"

# ─── HTTP / Rate-limit ──────────────────────────────────────────────
BASE_URL = "https://tiki.vn"
PRODUCT_SEARCH_URL = "https://tiki.vn/api/v2/products"
PRODUCT_DETAIL_URL = "https://tiki.vn/api/v2/products/{product_id}"

# Số sản phẩm Tiki trả về / trang (max 40 cho API products).
PAGE_SIZE = 40

# Số trang tối đa / lớp (40 sp/trang × 8 trang = 320 sp tối đa).
# Nếu cần 400 ảnh, mỗi sản phẩm trung bình 1.5 ảnh → đủ.
MAX_PAGES_PER_CLASS = 10

# Delay giữa các request (giây) — giữ lịch sự, tránh bị block.
REQUEST_DELAY_SEC = 2.0

# Số lần retry khi request fail.
MAX_RETRIES = 3

# ─── Image filtering ────────────────────────────────────────────────
MIN_WIDTH = 128
MIN_HEIGHT = 128
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# User-Agent giả lập browser thật.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

# Headers chuẩn cho Tiki API.
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://tiki.vn/",
    "Origin": "https://tiki.vn",
}
