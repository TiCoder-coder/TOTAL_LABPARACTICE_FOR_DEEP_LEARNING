"""Loại bỏ ảnh lỗi, không đọc được, hoặc kích thước quá nhỏ."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError


def iter_image_files(folder: Path) -> Iterable[Path]:
    """Lặp qua tất cả file ảnh trong folder (đệ quy 1 cấp)."""
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"
        }:
            yield path


def validate_image(path: Path, min_width: int = 64, min_height: int = 64) -> tuple[bool, str]:
    """Kiểm tra ảnh có hợp lệ không. Trả về (ok, reason)."""
    try:
        with Image.open(path) as img:
            img.verify()
    except (UnidentifiedImageError, OSError, ValueError) as e:
        return False, f"corrupt ({e.__class__.__name__})"

    try:
        with Image.open(path) as img:
            img.load()
            w, h = img.size
    except Exception as e:
        return False, f"unreadable ({e.__class__.__name__})"

    if w < min_width or h < min_height:
        return False, f"too_small ({w}x{h})"

    return True, "ok"


def clean_folder(
    folder: Path,
    min_width: int = 64,
    min_height: int = 64,
    dry_run: bool = False,
) -> dict:
    """Xoá ảnh lỗi / quá nhỏ trong 1 folder. Trả về báo cáo."""
    removed = {}
    for path in iter_image_files(folder):
        ok, reason = validate_image(path, min_width, min_height)
        if not ok:
            removed[str(path.name)] = reason
            if not dry_run:
                path.unlink()
    return {
        "folder": folder.name,
        "removed_count": len(removed),
        "removed": removed,
    }


def clean_root(
    root: Path,
    min_width: int = 64,
    min_height: int = 64,
    dry_run: bool = False,
) -> list[dict]:
    """Làm sạch tất cả folder con trong root."""
    reports = []
    for sub in sorted(root.iterdir()):
        if sub.is_dir():
            report = clean_folder(sub, min_width, min_height, dry_run)
            reports.append(report)
    return reports


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean a folder of crawled images.")
    parser.add_argument("--root", default="data", type=str)
    parser.add_argument("--min-width", default=64, type=int)
    parser.add_argument("--min-height", default=64, type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    reports = clean_root(Path(args.root), args.min_width, args.min_height, args.dry_run)
    for r in reports:
        print(f"  [{r['folder']}] removed {r['removed_count']} files")
