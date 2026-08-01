"""Crawl ảnh từ DuckDuckGo (không cần API key).

Cài: pip install duckduckgo-search Pillow tqdm requests
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

from .config import (
    ALLOWED_EXTENSIONS,
    CLASSES,
    IMAGES_PER_CLASS,
    MIN_HEIGHT,
    MIN_WIDTH,
    OUTPUT_DIR,
)


def is_valid_image(content: bytes) -> bool:
    """Kiểm tra ảnh có đọc được và đạt kích thước tối thiểu không."""
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()
        img = Image.open(io.BytesIO(content))
        if img.width < MIN_WIDTH or img.height < MIN_HEIGHT:
            return False
        return True
    except Exception:
        return False


def download_image(url: str, timeout: float = 10.0) -> bytes | None:
    """Tải ảnh về RAM, trả về bytes nếu hợp lệ."""
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 practice_2_2-crawler"},
        )
        if resp.status_code != 200:
            return None
        if not is_valid_image(resp.content):
            return None
        return resp.content
    except Exception:
        return None


def get_extension(url: str) -> str:
    """Rút extension từ URL."""
    url = url.split("?")[0].split("#")[0].lower()
    for ext in ALLOWED_EXTENSIONS:
        if url.endswith(ext):
            return ext
    return ".jpg"


def crawl_class(
    class_name: str,
    query: str,
    target_dir: Path,
    max_images: int,
) -> int:
    """Crawl ảnh cho 1 lớp, lưu vào target_dir. Trả về số ảnh download được."""
    target_dir.mkdir(parents=True, exist_ok=True)
    # Xoá ảnh cũ nếu có để tránh trộn lẫn nhiều lần crawl
    for old in target_dir.glob("*"):
        if old.is_file():
            old.unlink()

    try:
        from duckduckgo_search import DDGS
    except ImportError:
        raise SystemExit(
            "Chưa cài duckduckgo-search. Chạy: pip install duckduckgo-search"
        )

    saved = 0
    with DDGS() as ddgs:
        iterator = ddgs.images(
            keywords=query,
            max_results=max_images * 3,  # request dư vì lọc nhiều
            size="Medium",
            type_image="photo",
        )
        for i, result in enumerate(tqdm(iterator, desc=f"  {class_name}")):
            if saved >= max_images:
                break
            url = result.get("image")
            if not url:
                continue
            content = download_image(url)
            if content is None:
                continue
            save_path = target_dir / f"{class_name}_{i:05d}{get_extension(url)}"
            save_path.write_bytes(content)
            saved += 1

    return saved


def crawl_all() -> None:
    """Crawl tất cả các lớp trong config.CLASSES."""
    output_root = Path(OUTPUT_DIR)
    print(f"Start crawling {len(CLASSES)} classes, target ~{IMAGES_PER_CLASS} imgs/class")
    print(f"Output: {output_root.resolve()}\n")

    summary = {}
    for cls_name, query in CLASSES.items():
        target_dir = output_root / cls_name
        start = time.time()
        n = crawl_class(cls_name, query, target_dir, IMAGES_PER_CLASS)
        elapsed = time.time() - start
        summary[cls_name] = (n, elapsed)
        print(f"  [{cls_name}] saved {n} images in {elapsed:.1f}s\n")

    print("=" * 50)
    print("CRAWL SUMMARY")
    print("=" * 50)
    for cls, (n, t) in summary.items():
        print(f"  {cls:<12} {n:>4} images  ({t:.1f}s)")
    print(f"\nTotal saved: {sum(n for n, _ in summary.values())} images")


if __name__ == "__main__":
    crawl_all()
