"""Resize tất cả ảnh trong các folder con về cùng kích thước TARGET_SIZE.

Mặc định resize về 224x224 (ImageNet) để khớp model Pretrained của practice_2.
Tuỳ chọn: "pad" (thêm viền) hoặc "crop" (cắt giữa).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from tqdm import tqdm

from .clean_data import iter_image_files


TARGET_SIZE = (224, 224)
RESIZE_MODE = "crop"  # "crop" | "pad" | "stretch"


def resize_keep_aspect(img: Image.Image, target: int) -> Image.Image:
    """Resize ảnh giữ tỉ lệ sao cho cạnh ngắn = target."""
    w, h = img.size
    if w < h:
        new_w = target
        new_h = int(round(h * target / w))
    else:
        new_h = target
        new_w = int(round(w * target / h))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def crop_center(img: Image.Image, target: tuple[int, int]) -> Image.Image:
    """Resize giữ tỉ lệ rồi crop giữa về target."""
    img = resize_keep_aspect(img, target[0])
    w, h = img.size
    left = (w - target[0]) // 2
    top = (h - target[1]) // 2
    return img.crop((left, top, left + target[0], top + target[1]))


def pad_to_square(img: Image.Image, target: tuple[int, int], color=(0, 0, 0)) -> Image.Image:
    """Resize giữ tỉ lệ rồi pad về target (thêm viền)."""
    img = resize_keep_aspect(img, target[0])
    w, h = img.size
    new = Image.new("RGB", target, color)
    new.paste(img, ((target[0] - w) // 2, (target[1] - h) // 2))
    return new


def stretch(img: Image.Image, target: tuple[int, int]) -> Image.Image:
    """Resize trực tiếp về target (méo ảnh)."""
    return img.resize(target, Image.Resampling.LANCZOS)


def resize_image(img: Image.Image, target: tuple[int, int], mode: str) -> Image.Image:
    if mode == "crop":
        return crop_center(img, target)
    if mode == "pad":
        return pad_to_square(img, target)
    return stretch(img, target)


def resize_folder(
    src: Path,
    dst: Path,
    target: tuple[int, int] = TARGET_SIZE,
    mode: str = RESIZE_MODE,
) -> dict:
    """Resize tất cả ảnh từ src sang dst."""
    dst.mkdir(parents=True, exist_ok=True)
    ok, fail = 0, 0
    for path in tqdm(list(iter_image_files(src)), desc=f"  {src.name}"):
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                out = resize_image(img, target, mode)
            save_path = dst / (path.stem + ".jpg")
            out.save(save_path, "JPEG", quality=92)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"    [warn] {path.name}: {e}")
    return {"folder": src.name, "ok": ok, "failed": fail}


def resize_root(
    src_root: Path,
    dst_root: Path,
    target: tuple[int, int] = TARGET_SIZE,
    mode: str = RESIZE_MODE,
) -> list[dict]:
    """Resize toàn bộ cây thư mục."""
    dst_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for sub in sorted(src_root.iterdir()):
        if sub.is_dir():
            reports.append(resize_folder(sub, dst_root / sub.name, target, mode))
    return reports


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Resize images to a common size.")
    parser.add_argument("--src", default="data", type=str)
    parser.add_argument("--dst", default="data_clean", type=str)
    parser.add_argument("--size", default=224, type=int)
    parser.add_argument("--mode", default=RESIZE_MODE, choices=["crop", "pad", "stretch"])
    args = parser.parse_args()

    reports = resize_root(
        Path(args.src), Path(args.dst), (args.size, args.size), args.mode
    )
    for r in reports:
        print(f"  [{r['folder']}] resized {r['ok']} (failed {r['failed']})")
