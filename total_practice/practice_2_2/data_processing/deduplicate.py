"""Loại bỏ ảnh trùng lặp trong từng folder (dựa trên MD5 hash)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .clean_data import iter_image_files


def file_hash(path: Path, chunk_size: int = 1 << 16) -> str:
    """Tính MD5 hash của file theo chunk."""
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def dedupe_folder(folder: Path, dry_run: bool = False) -> dict:
    """Xoá ảnh trùng hash trong 1 folder. Giữ lại file đầu tiên."""
    seen = set()
    removed = []
    for path in iter_image_files(folder):
        digest = file_hash(path)
        if digest in seen:
            removed.append(path.name)
            if not dry_run:
                path.unlink()
        else:
            seen.add(digest)
    return {
        "folder": folder.name,
        "removed_count": len(removed),
        "removed": removed,
    }


def dedupe_root(root: Path, dry_run: bool = False) -> list[dict]:
    """Dedupe tất cả folder con."""
    reports = []
    for sub in sorted(root.iterdir()):
        if sub.is_dir():
            reports.append(dedupe_folder(sub, dry_run))
    return reports


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dedupe images by MD5 hash.")
    parser.add_argument("--root", default="data", type=str)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    reports = dedupe_root(Path(args.root), args.dry_run)
    for r in reports:
        print(f"  [{r['folder']}] removed {r['removed_count']} duplicates")
