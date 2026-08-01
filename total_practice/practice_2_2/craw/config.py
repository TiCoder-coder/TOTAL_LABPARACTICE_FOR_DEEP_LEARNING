"""Crawl config: danh sách lớp và tham số crawl."""

# Danh sách lớp — bạn có thể đổi tuỳ chủ đề muốn crawl.
# Tên folder sẽ lấy theo key (tiếng Anh, snake_case).
CLASSES = {
    "airplane": "airplane",
    "car":      "car",
    "dog":      "dog",
    "cat":      "cat",
    "bicycle":  "bicycle",
    "cup":      "cup",
    "chair":    "chair",
    "laptop":   "laptop",
}

# Số ảnh tối đa mỗi lớp — DuckDuckGo thường giới hạn ~500/ lệnh.
IMAGES_PER_CLASS = 200

# Thư mục output (ảnh thô).
OUTPUT_DIR = "data"

# Nguồn: "duckduckgo" (free, mặc định) hoặc "google".
SOURCE = "duckduckgo"

# Lọc size tối thiểu (pixel) — ảnh quá nhỏ sẽ bị bỏ.
MIN_WIDTH = 128
MIN_HEIGHT = 128

# Lọc định dạng hợp lệ.
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
