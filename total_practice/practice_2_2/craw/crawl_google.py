"""Crawl ảnh từ Google Images (Selenium).

Cài:
    pip install selenium webdriver-manager
    pip install Pillow tqdm requests

Lưu ý: Google Images thay đổi DOM thường xuyên, nên ưu tiên DuckDuckGo.
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

from .config import ALLOWED_EXTENSIONS, CLASSES, IMAGES_PER_CLASS, OUTPUT_DIR


def init_driver():
    """Khởi tạo Selenium Chrome driver."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
    except ImportError:
        raise SystemExit(
            "Chưa cài selenium / webdriver-manager. "
            "Chạy: pip install selenium webdriver-manager"
        )

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def get_image_urls(driver, query: str, max_results: int) -> list[str]:
    """Mở Google Images, scroll và lấy URL ảnh."""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys

    driver.get("https://www.google.com/imghp?hl=en")
    box = driver.find_element(By.NAME, "q")
    box.send_keys(query)
    box.send_keys(Keys.ENTER)
    time.sleep(2)

    urls = []
    last_count = 0
    stagnant_tries = 0
    while len(urls) < max_results and stagnant_tries < 5:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.5)
        imgs = driver.find_elements(By.CSS_SELECTOR, "img.rg_i, img.Q4LuWd")
        for img in imgs:
            try:
                src = img.get_attribute("src")
                if src and src.startswith("http") and src not in urls:
                    urls.append(src)
            except Exception:
                continue
        if len(urls) == last_count:
            stagnant_tries += 1
        else:
            stagnant_tries = 0
        last_count = len(urls)
    return urls[:max_results]


def download_image(url: str, timeout: float = 10.0) -> bytes | None:
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 practice_2_2-crawler"},
        )
        if resp.status_code != 200:
            return None
        img = Image.open(io.BytesIO(resp.content))
        img.verify()
        img = Image.open(io.BytesIO(resp.content))
        if img.width < 128 or img.height < 128:
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


def crawl_class(class_name: str, query: str, target_dir: Path, max_images: int) -> int:
    """Crawl 1 lớp bằng Google + Selenium."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for old in target_dir.glob("*"):
        if old.is_file():
            old.unlink()

    driver = init_driver()
    try:
        urls = get_image_urls(driver, query, max_images * 3)
    finally:
        driver.quit()

    saved = 0
    for i, url in enumerate(tqdm(urls, desc=f"  {class_name}")):
        if saved >= max_images:
            break
        content = download_image(url)
        if content is None:
            continue
        path = target_dir / f"{class_name}_{i:05d}{get_extension(url)}"
        path.write_bytes(content)
        saved += 1
    return saved


def crawl_all() -> None:
    output_root = Path(OUTPUT_DIR)
    print(f"Start crawl Google Images cho {len(CLASSES)} classes...")
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
