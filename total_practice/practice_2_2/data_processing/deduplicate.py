"""Loại bỏ ảnh trùng lặp trong từng folder và cross-class (dựa trên MD5 hash).

Bao gồm:
  - dedupe_folder: xoá trùng MD5 trong 1 folder (giữ file đầu tiên).
  - dedupe_root:   áp dụng cho tất cả sub-folder.
  - dedupe_cross_class: phát hiện & xoá file trùng MD5 giữa nhiều class.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Iterable

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
    seen: set[str] = set()
    removed: list[str] = []
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
    reports: list[dict] = []
    for sub in sorted(root.iterdir()):
        if sub.is_dir():
            reports.append(dedupe_folder(sub, dry_run))
    return reports


def dedupe_cross_class(
    root: Path,
    prefer_classes: list[str] | None = None,
    dry_run: bool = False,
) -> dict:
    """Phát hiện & xoá ảnh trùng MD5 giữa các class folders.

    Mặc định giữ lại bản ở class có **nhiều ảnh hơn** (lớn hơn = đáng tin cậy hơn).
    Nếu `prefer_classes` được cung cấp, ưu tiên giữ ảnh ở các class đó (thứ tự ưu tiên).

    Returns:
        {
            "hash_to_classes": {hash: [class1, class2, ...]},
            "removed": [{"hash", "path", "kept_in_class", "kept_in_path"} ...],
            "removed_count": int,
        }
    """
    if not root.exists():
        return {"hash_to_classes": {}, "removed": [], "removed_count": 0}

    hash_to_files: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    class_counts: dict[str, int] = {}

    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        cls = sub.name
        files = list(iter_image_files(sub))
        class_counts[cls] = len(files)
        for path in files:
            digest = file_hash(path)
            hash_to_files[digest].append((cls, path))

    cross_groups = {
        h: lst for h, lst in hash_to_files.items() if len(set(c for c, _ in lst)) > 1
    }

    removed: list[dict] = []

    for digest, lst in cross_groups.items():
        classes_present = set(c for c, _ in lst)

        # Xác định class "winner" (giữ lại)
        if prefer_classes:
            winners = [c for c in prefer_classes if c in classes_present]
            if winners:
                keep_class = winners[0]
            else:
                keep_class = max(classes_present, key=lambda c: class_counts.get(c, 0))
        else:
            keep_class = max(classes_present, key=lambda c: class_counts.get(c, 0))

        # Lấy file đầu tiên của class winner để giữ
        keep_path = next(p for c, p in lst if c == keep_class)

        # Xoá các bản ở class khác
        for cls, path in lst:
            if cls == keep_class and path == keep_path:
                continue
            if cls == keep_class:
                # cùng class winner nhưng trùng tên/cùng hash → cũng xoá (giữ 1)
                if not dry_run:
                    path.unlink()
                removed.append(
                    {
                        "hash": digest[:10],
                        "path": str(path),
                        "kept_in_class": keep_class,
                        "kept_in_path": str(keep_path),
                        "reason": "same-class-duplicate",
                    }
                )
            else:
                if not dry_run:
                    path.unlink()
                removed.append(
                    {
                        "hash": digest[:10],
                        "path": str(path),
                        "kept_in_class": keep_class,
                        "kept_in_path": str(keep_path),
                        "reason": f"cross-class-leak-prefer-{keep_class}",
                    }
                )

    return {
        "hash_to_classes": {h: sorted(set(c for c, _ in lst)) for h, lst in cross_groups.items()},
        "removed": removed,
        "removed_count": len(removed),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dedupe images by MD5 hash.")
    parser.add_argument("--root", default="data", type=str)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--cross-class",
        action="store_true",
        help="Also detect and remove cross-class duplicates.",
    )
    parser.add_argument(
        "--prefer",
        nargs="*",
        default=None,
        help="Preferred classes (in order) when resolving cross-class dups.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    print(f"[intra] dedupe per folder...")
    for r in dedupe_root(root, args.dry_run):
        print(f"  [{r['folder']}] removed {r['removed_count']} duplicates")

    if args.cross_class:
        print(f"[cross] dedupe across classes (prefer={args.prefer})...")
        report = dedupe_cross_class(root, args.prefer, args.dry_run)
        print(f"  found {len(report['hash_to_classes'])} cross-class duplicate groups")
        print(f"  removed {report['removed_count']} files")