"""Lọc ảnh chất lượng kém (quá mờ, content ratio quá thấp).

Dùng Laplacian variance để ước lượng độ nét; ảnh có variance thấp = mờ.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .clean_data import iter_image_files


# Ngưỡng mặc định — dưới ngưỡng này sẽ bị loại.
DEFAULT_BLUR_THRESHOLD = 30.0
DEFAULT_CONTENT_RATIO = 0.10  # ảnh có <10% pixel "không phải nền đơn sắc" bị loại


def is_blurry(path: Path, threshold: float) -> tuple[bool, float]:
    """Trả về (is_blurry, laplacian_variance)."""
    try:
        with Image.open(path) as img:
            img = img.convert("L")  # grayscale
            arr = np.asarray(img, dtype=np.float32)
    except Exception:
        return True, 0.0

    # Laplacian kernel đơn giản
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    from numpy.lib.stride_tricks import sliding_window_view

    windows = sliding_window_view(arr, kernel.shape)
    conv = (windows * kernel).sum(axis=(-2, -1))
    var = float(conv.var())
    return var < threshold, var


def has_meaningful_content(path: Path, min_ratio: float) -> tuple[bool, float]:
    """Trả về (ok, ratio). Ratio = phần pixel không phải nền trắng/đen đơn sắc."""
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            arr = np.asarray(img)
    except Exception:
        return False, 0.0

    # Pixel "nền" thường có R=G=B và lệch mạnh về 0 hoặc 255
    maxc = arr.max(axis=-1)
    minc = arr.min(axis=-1)
    spread = (maxc - minc).astype(np.float32)
    ratio = float((spread > 20).mean())
    return ratio >= min_ratio, ratio


def quality_check_folder(
    folder: Path,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
    min_content_ratio: float = DEFAULT_CONTENT_RATIO,
    dry_run: bool = False,
) -> dict:
    """Kiểm tra & xoá ảnh mờ / không có nội dung."""
    removed = []
    for path in iter_image_files(folder):
        blurry, var = is_blurry(path, blur_threshold)
        meaningful, ratio = has_meaningful_content(path, min_content_ratio)
        if blurry or not meaningful:
            removed.append((path.name, var, ratio))
            if not dry_run:
                path.unlink()
    return {
        "folder": folder.name,
        "removed_count": len(removed),
        "removed": removed,
    }


def quality_check_root(
    root: Path,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
    min_content_ratio: float = DEFAULT_CONTENT_RATIO,
    dry_run: bool = False,
) -> list[dict]:
    reports = []
    for sub in sorted(root.iterdir()):
        if sub.is_dir():
            reports.append(
                quality_check_folder(sub, blur_threshold, min_content_ratio, dry_run)
            )
    return reports


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter low-quality images.")
    parser.add_argument("--root", default="data", type=str)
    parser.add_argument("--blur-threshold", default=DEFAULT_BLUR_THRESHOLD, type=float)
    parser.add_argument("--min-content-ratio", default=DEFAULT_CONTENT_RATIO, type=float)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    reports = quality_check_root(
        Path(args.root), args.blur_threshold, args.min_content_ratio, args.dry_run
    )
    for r in reports:
        print(f"  [{r['folder']}] removed {r['removed_count']} low-quality")
