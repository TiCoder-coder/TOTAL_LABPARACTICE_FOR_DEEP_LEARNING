"""Crawl ảnh từ Tiki.vn dùng internal API.

API reference (từ community reverse-engineering):
  - Search:  GET https://tiki.vn/api/v2/products?category=...&q=...&page=...
  - Detail:  GET https://tiki.vn/api/v2/products/{id}

Mỗi sản phẩm trả về list ảnh trong field `images` (mỗi item có `base_url`).

Cài:
    pip install requests Pillow tqdm

Lưu ý ToS:
  - Tuân thủ REQUEST_DELAY_SEC
  - Chỉ dùng cho mục đích học tập / nghiên cứu
  - Không thương mại hoá ảnh crawl được
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image
from tqdm import tqdm

from .config_tiki import (
    ALLOWED_EXTENSIONS,
    CLASSES,
    HEADERS,
    IMAGES_PER_CLASS,
    MAX_PAGES_PER_CLASS,
    MAX_RETRIES,
    MIN_HEIGHT,
    MIN_WIDTH,
    OUTPUT_DIR,
    PAGE_SIZE,
    PRODUCT_DETAIL_URL,
    PRODUCT_SEARCH_URL,
    REQUEST_DELAY_SEC,
)


# ─── HTTP helpers ───────────────────────────────────────────────────

def http_get(url: str, params: dict | None = None) -> dict | None:
    """GET request với retry + delay. Trả về JSON hoặc None."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                # Rate-limited — chờ lâu hơn
                wait = REQUEST_DELAY_SEC * (2 ** attempt)
                print(f"  [429] rate-limited, sleeping {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                time.sleep(REQUEST_DELAY_SEC * attempt)
                continue
            # Client error khác — bỏ qua
            return None
        except requests.RequestException as e:
            print(f"  [net err] {e!r} (attempt {attempt})")
            time.sleep(REQUEST_DELAY_SEC * attempt)
    return None


# ─── Tiki API wrappers ──────────────────────────────────────────────

def search_products(query: str, page: int) -> list[dict]:
    """Trả về list sản phẩm từ Tiki search API."""
    params = {
        "q": query,
        "page": page,
        "limit": PAGE_SIZE,
        "sort": "popular",
    }
    data = http_get(PRODUCT_SEARCH_URL, params=params)
    if not data:
        return []
    return data.get("data", []) or []


def get_product_detail(product_id: int) -> dict | None:
    """Lấy detail (bao gồm full list ảnh) của 1 sản phẩm."""
    url = PRODUCT_DETAIL_URL.format(product_id=product_id)
    return http_get(url)


def collect_image_urls(product: dict, detail: dict | None) -> list[str]:
    """Rút trường `images[*].base_url` từ product + detail."""
    urls: list[str] = []

    # 1. Field images ngay trong product search result
    for img in product.get("images", []) or []:
        if isinstance(img, dict):
            url = img.get("base_url") or img.get("large_url")
            if url:
                urls.append(url)
        elif isinstance(img, str):
            urls.append(img)

    # 2. Field thumbnail_url (fallback)
    if product.get("thumbnail_url"):
        urls.append(product["thumbnail_url"])

    # 3. Detail cho nhiều ảnh hơn
    if detail:
        for img in detail.get("images", []) or []:
            if isinstance(img, dict):
                url = img.get("base_url") or img.get("large_url")
                if url and url not in urls:
                    urls.append(url)

    return urls


# ─── Image download & validate ──────────────────────────────────────

def is_valid_image(content: bytes) -> bool:
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()
        img = Image.open(io.BytesIO(content))
        if img.width < MIN_WIDTH or img.height < MIN_HEIGHT:
            return False
        return True
    except Exception:
        return False


def download_image(url: str) -> bytes | None:
    try:
        resp = requests.get(url, timeout=15, headers=HEADERS)
        if resp.status_code != 200:
            return None
        if not is_valid_image(resp.content):
            return None
        return resp.content
    except Exception:
        return None


def get_extension(url: str) -> str:
    url = url.split("?")[0].split("#")[0].lower()
    for ext in ALLOWED_EXTENSIONS:
        if url.endswith(ext):
            return ext
    return ".jpg"


# ─── Crawl per class ────────────────────────────────────────────────

def crawl_class(
    class_name: str,
    query: str,
    target_dir: Path,
    max_images: int,
) -> int:
    """Crawl 1 lớp. Duyệt qua các trang search, lấy ảnh từ product + detail."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for old in target_dir.glob("*"):
        if old.is_file():
            old.unlink()

    saved = 0
    seen_urls: set[str] = set()
    product_id_seen: set[int] = set()

    print(f"  Searching Tiki for: '{query}'")
    for page in range(1, MAX_PAGES_PER_CLASS + 1):
        if saved >= max_images:
            break
        products = search_products(query, page)
        if not products:
            print(f"  [page {page}] no more products, stop")
            break

        for product in tqdm(products, desc=f"  {class_name} p{page}", leave=False):
            if saved >= max_images:
                break
            pid = product.get("id")
            if not pid or pid in product_id_seen:
                continue
            product_id_seen.add(pid)

            # Lấy detail (nhiều ảnh hơn)
            time.sleep(REQUEST_DELAY_SEC * 0.3)  # micro-delay
            detail = get_product_detail(pid)

            urls = collect_image_urls(product, detail)
            for url in urls:
                if saved >= max_images:
                    break
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                content = download_image(url)
                if content is None:
                    continue

                save_path = target_dir / f"{class_name}_{saved:05d}{get_extension(url)}"
                save_path.write_bytes(content)
                saved += 1

        # Delay giữa các page
        time.sleep(REQUEST_DELAY_SEC)

    return saved


def crawl_all() -> None:
    output_root = Path(OUTPUT_DIR)
    print(f"Tiki crawler — {len(CLASSES)} classes, target ~{IMAGES_PER_CLASS} imgs/class")
    print(f"Output: {output_root.resolve()}\n")

    summary = {}
    for cls_name, query in CLASSES.items():
        target_dir = output_root / cls_name
        start = time.time()
        try:
            n = crawl_class(cls_name, query, target_dir, IMAGES_PER_CLASS)
        except Exception as e:
            print(f"  [error] {cls_name}: {e!r}")
            n = 0
        elapsed = time.time() - start
        summary[cls_name] = (n, elapsed)
        print(f"  [{cls_name}] saved {n} images in {elapsed:.1f}s\n")

    print("=" * 60)
    print("TIKI CRAWL SUMMARY")
    print("=" * 60)
    total = 0
    for cls, (n, t) in summary.items():
        print(f"  {cls:<20} {n:>4} images  ({t:.1f}s)")
        total += n
    print(f"\nTotal saved: {total} images")


if __name__ == "__main__":
    crawl_all()
