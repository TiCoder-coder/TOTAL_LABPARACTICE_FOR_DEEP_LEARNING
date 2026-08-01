"""Pipeline tổng hợp: clean → dedupe → quality check → resize.

Chạy:
    python -m data_processing.pipeline
"""

from __future__ import annotations

import time
from pathlib import Path

from .clean_data import clean_root
from .deduplicate import dedupe_root
from .quality_check import quality_check_root
from .resize_data import TARGET_SIZE, RESIZE_MODE, resize_root


# Đường dẫn mặc định — sửa nếu bạn đặt folder khác.
DEFAULT_SRC = Path("data")
DEFAULT_DST = Path("data_clean")

# Ngưỡng mặc định
MIN_WIDTH = 64
MIN_HEIGHT = 64
BLUR_THRESHOLD = 30.0
MIN_CONTENT_RATIO = 0.10


def run_pipeline(
    src: Path = DEFAULT_SRC,
    dst: Path = DEFAULT_DST,
    min_width: int = MIN_WIDTH,
    min_height: int = MIN_HEIGHT,
    blur_threshold: float = BLUR_THRESHOLD,
    min_content_ratio: float = MIN_CONTENT_RATIO,
    target_size: tuple[int, int] = TARGET_SIZE,
    resize_mode: str = RESIZE_MODE,
) -> dict:
    """Chạy pipeline đầy đủ. Trả về summary."""
    if not src.exists():
        raise FileNotFoundError(f"Source folder not found: {src}")

    summary = {"steps": []}

    print("=" * 60)
    print(f"DATA PROCESSING PIPELINE")
    print(f"  src = {src.resolve()}")
    print(f"  dst = {dst.resolve()}")
    print(f"  target_size = {target_size}, mode = {resize_mode}")
    print("=" * 60)

    # Step 1: Clean (xoá ảnh lỗi / quá nhỏ)
    print("\n[1/4] Cleaning corrupt / too-small images...")
    t0 = time.time()
    clean_reports = clean_root(src, min_width, min_height)
    total_removed = sum(r["removed_count"] for r in clean_reports)
    print(f"  done in {time.time()-t0:.1f}s — removed {total_removed} files")
    summary["steps"].append({"step": "clean", "removed": total_removed})

    # Step 2: Dedupe
    print("\n[2/4] Removing duplicate images...")
    t0 = time.time()
    dedupe_reports = dedupe_root(src)
    total_dup = sum(r["removed_count"] for r in dedupe_reports)
    print(f"  done in {time.time()-t0:.1f}s — removed {total_dup} duplicates")
    summary["steps"].append({"step": "dedupe", "removed": total_dup})

    # Step 3: Quality check
    print("\n[3/4] Filtering blurry / low-content images...")
    t0 = time.time()
    qc_reports = quality_check_root(src, blur_threshold, min_content_ratio)
    total_qc = sum(r["removed_count"] for r in qc_reports)
    print(f"  done in {time.time()-t0:.1f}s — removed {total_qc} files")
    summary["steps"].append({"step": "quality", "removed": total_qc})

    # Step 4: Resize + copy to data_clean
    print("\n[4/4] Resizing images to clean folder...")
    t0 = time.time()
    resize_reports = resize_root(src, dst, target_size, resize_mode)
    total_ok = sum(r["ok"] for r in resize_reports)
    total_fail = sum(r["failed"] for r in resize_reports)
    print(f"  done in {time.time()-t0:.1f}s — {total_ok} resized, {total_fail} failed")
    summary["steps"].append({"step": "resize", "ok": total_ok, "failed": total_fail})

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Source: {src.resolve()}")
    print(f"  Output: {dst.resolve()}")
    print(f"  Total steps: {len(summary['steps'])}")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run full data processing pipeline.")
    parser.add_argument("--src", default="data", type=str)
    parser.add_argument("--dst", default="data_clean", type=str)
    parser.add_argument("--min-width", default=MIN_WIDTH, type=int)
    parser.add_argument("--min-height", default=MIN_HEIGHT, type=int)
    parser.add_argument("--blur-threshold", default=BLUR_THRESHOLD, type=float)
    parser.add_argument("--min-content-ratio", default=MIN_CONTENT_RATIO, type=float)
    parser.add_argument("--size", default=224, type=int)
    parser.add_argument("--mode", default=RESIZE_MODE, choices=["crop", "pad", "stretch"])
    args = parser.parse_args()

    run_pipeline(
        Path(args.src),
        Path(args.dst),
        args.min_width,
        args.min_height,
        args.blur_threshold,
        args.min_content_ratio,
        (args.size, args.size),
        args.mode,
    )
