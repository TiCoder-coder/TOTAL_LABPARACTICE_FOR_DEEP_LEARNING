from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_functional

from .paths import get_practice_2_2_root


POLICY_RELATIVE_PATH = Path("configs/r6_transform_ablation_policy.json")
RECIPE_FIELDS = (
    "evaluation_mode",
    "crop_scale_floor",
    "horizontal_flip_probability",
    "random_erasing_probability",
)


def load_r6_policy(policy_path: Path | None = None) -> dict[str, Any]:
    root = get_practice_2_2_root()
    path = Path(policy_path or root / POLICY_RELATIVE_PATH).expanduser().resolve()
    policy = json.loads(path.read_text())
    if policy.get("schema_version") != 1:
        raise RuntimeError("Unsupported R6 transform policy schema")
    if policy.get("seed") != 42:
        raise RuntimeError("R6 must use seed 42")
    if policy.get("authorized_content_split") != "Train":
        raise RuntimeError("R6 content authority must be Train only")
    if policy.get("predecessor_gate_required") is not True:
        raise RuntimeError("R6 must require the R5 gate")
    if policy.get("automatic_recipe_selection_allowed") is not False:
        raise RuntimeError("R6 must prohibit automatic recipe selection")
    if policy.get("validation_content_access_allowed") is not False:
        raise RuntimeError("R6 must prohibit Validation content access")
    if policy.get("test_content_access_allowed") is not False:
        raise RuntimeError("R6 must prohibit Test content access")
    if policy.get("model_training_allowed") is not False:
        raise RuntimeError("R6 must prohibit model training")
    recipes = policy.get("recipes", {})
    if tuple(recipes) != ("T0", "T1", "T2", "T3", "T4"):
        raise RuntimeError("R6 must define T0 through T4 in order")
    for recipe_id, recipe in recipes.items():
        parent_id = recipe["parent"]
        if parent_id is None:
            if recipe_id != "T0":
                raise RuntimeError("R6 only T0 may omit a parent")
            continue
        parent = recipes[parent_id]
        changed = [field for field in RECIPE_FIELDS if recipe[field] != parent[field]]
        if changed != [recipe["single_change"]]:
            raise RuntimeError(f"R6 recipe is not a one-variable ablation: {recipe_id}")
    return policy


def _operation_seed(seed: int, asset_id: str, operation: str) -> int:
    digest = hashlib.sha256(f"{seed}:{asset_id}:{operation}".encode()).hexdigest()
    return int(digest[:16], 16)


def select_train_rows(
    manifest: Sequence[Mapping[str, Any]], policy: Mapping[str, Any] | None = None
) -> tuple[list[dict[str, Any]], set[str]]:
    policy = dict(policy or load_r6_policy())
    train_rows = sorted(
        (
            dict(row)
            for row in manifest
            if row["split"] == policy["authorized_content_split"]
            and row["is_generated"] is False
        ),
        key=lambda item: item["asset_id"],
    )
    heldout_ids = {
        str(row["asset_id"])
        for row in manifest
        if row["split"] in {"Validation", "Test"}
    }
    train_ids = {str(row["asset_id"]) for row in train_rows}
    if len(train_ids) != len(train_rows) or train_ids & heldout_ids:
        raise RuntimeError("R6 Train content selection is not isolated")
    return train_rows, heldout_ids


def _random_resized_crop_parameters(
    image: Image.Image,
    scale_floor: float,
    ratio_range: Sequence[float],
    seed: int,
) -> tuple[int, int, int, int, float]:
    generator = random.Random(seed)
    width, height = image.size
    area = height * width
    log_ratio = (math.log(ratio_range[0]), math.log(ratio_range[1]))
    for _ in range(10):
        target_area = area * generator.uniform(scale_floor, 1.0)
        aspect_ratio = math.exp(generator.uniform(*log_ratio))
        crop_width = int(round(math.sqrt(target_area * aspect_ratio)))
        crop_height = int(round(math.sqrt(target_area / aspect_ratio)))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            top = generator.randint(0, height - crop_height)
            left = generator.randint(0, width - crop_width)
            return top, left, crop_height, crop_width, target_area / area
    input_ratio = width / height
    if input_ratio < ratio_range[0]:
        crop_width = width
        crop_height = int(round(crop_width / ratio_range[0]))
    elif input_ratio > ratio_range[1]:
        crop_height = height
        crop_width = int(round(crop_height * ratio_range[1]))
    else:
        crop_width = width
        crop_height = height
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    return top, left, crop_height, crop_width, crop_height * crop_width / area


def _apply_color_jitter(
    image: Image.Image, asset_id: str, policy: Mapping[str, Any]
) -> Image.Image:
    settings = policy["color_jitter"]
    generator = random.Random(
        _operation_seed(policy["seed"], asset_id, "color_jitter")
    )
    operations = [
        (
            "brightness",
            generator.uniform(1 - settings["brightness"], 1 + settings["brightness"]),
        ),
        (
            "contrast",
            generator.uniform(1 - settings["contrast"], 1 + settings["contrast"]),
        ),
        (
            "saturation",
            generator.uniform(1 - settings["saturation"], 1 + settings["saturation"]),
        ),
        ("hue", generator.uniform(-settings["hue"], settings["hue"])),
    ]
    generator.shuffle(operations)
    output = image
    for operation, factor in operations:
        if operation == "brightness":
            output = vision_functional.adjust_brightness(output, factor)
        elif operation == "contrast":
            output = vision_functional.adjust_contrast(output, factor)
        elif operation == "saturation":
            output = vision_functional.adjust_saturation(output, factor)
        else:
            output = vision_functional.adjust_hue(output, factor)
    return output


def _apply_random_erasing(
    image: Image.Image,
    asset_id: str,
    probability: float,
    policy: Mapping[str, Any],
) -> tuple[Image.Image, bool, float]:
    decision = random.Random(_operation_seed(policy["seed"], asset_id, "erase_decision"))
    if decision.random() >= probability:
        return image, False, 0.0
    generator = random.Random(
        _operation_seed(policy["seed"], asset_id, "erase_geometry")
    )
    array = np.asarray(image, dtype=np.uint8).copy()
    height, width = array.shape[:2]
    area = height * width
    scale = policy["random_erasing_scale"]
    ratio = policy["random_erasing_ratio"]
    for _ in range(10):
        erase_area = generator.uniform(*scale) * area
        aspect = math.exp(generator.uniform(math.log(ratio[0]), math.log(ratio[1])))
        erase_height = int(round(math.sqrt(erase_area * aspect)))
        erase_width = int(round(math.sqrt(erase_area / aspect)))
        if 0 < erase_height < height and 0 < erase_width < width:
            top = generator.randint(0, height - erase_height)
            left = generator.randint(0, width - erase_width)
            noise = np.random.default_rng(
                _operation_seed(policy["seed"], asset_id, "erase_values")
            ).integers(0, 256, size=(erase_height, erase_width, 3), dtype=np.uint8)
            array[top : top + erase_height, left : left + erase_width] = noise
            return Image.fromarray(array), True, erase_height * erase_width / area
    return image, False, 0.0


def _border_black_fraction(image: Image.Image, threshold: int) -> float:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    border_width = max(2, int(round(min(array.shape[:2]) * 0.03)))
    mask = np.zeros(array.shape[:2], dtype=bool)
    mask[:border_width] = True
    mask[-border_width:] = True
    mask[:, :border_width] = True
    mask[:, -border_width:] = True
    black = np.all(array <= threshold, axis=2)
    return float(black[mask].mean())


def normalize_image(
    image: Image.Image, policy: Mapping[str, Any] | None = None
) -> torch.Tensor:
    policy = dict(policy or load_r6_policy())
    tensor = vision_functional.pil_to_tensor(image).to(torch.float32) / 255.0
    return vision_functional.normalize(
        tensor, mean=policy["imagenet_mean"], std=policy["imagenet_std"]
    )


def apply_recipe(
    image: Image.Image,
    asset_id: str,
    recipe_id: str,
    policy: Mapping[str, Any] | None = None,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    policy = dict(policy or load_r6_policy())
    recipe = policy["recipes"][recipe_id]
    source = image.convert("RGB")
    top, left, crop_height, crop_width, crop_retention = (
        _random_resized_crop_parameters(
            source,
            recipe["crop_scale_floor"],
            policy["random_resized_crop_ratio"],
            _operation_seed(policy["seed"], asset_id, "crop"),
        )
    )
    train_image = vision_functional.resized_crop(
        source,
        top,
        left,
        crop_height,
        crop_width,
        [policy["image_size"], policy["image_size"]],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    flip_generator = random.Random(
        _operation_seed(policy["seed"], asset_id, "horizontal_flip")
    )
    flipped = flip_generator.random() < recipe["horizontal_flip_probability"]
    if flipped:
        train_image = vision_functional.hflip(train_image)
    train_image = _apply_color_jitter(train_image, asset_id, policy)
    train_image, erased, erase_area_ratio = _apply_random_erasing(
        train_image,
        asset_id,
        recipe["random_erasing_probability"],
        policy,
    )
    if recipe["evaluation_mode"] == "resize_256_center_crop_224":
        resized = vision_functional.resize(
            source, 256, interpolation=InterpolationMode.BILINEAR, antialias=True
        )
        evaluation_image = vision_functional.center_crop(
            resized, [policy["image_size"], policy["image_size"]]
        )
        evaluation_retention = (policy["image_size"] / 256) ** 2
    elif recipe["evaluation_mode"] == "direct_resize_224":
        evaluation_image = vision_functional.resize(
            source,
            [policy["image_size"], policy["image_size"]],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        evaluation_retention = 1.0
    else:
        raise RuntimeError("R6 evaluation mode is invalid")
    source_black = _border_black_fraction(
        source, policy["black_border_pixel_threshold"]
    )
    train_black = _border_black_fraction(
        train_image, policy["black_border_pixel_threshold"]
    )
    evaluation_black = _border_black_fraction(
        evaluation_image, policy["black_border_pixel_threshold"]
    )
    normalized_train = normalize_image(train_image, policy)
    normalized_evaluation = normalize_image(evaluation_image, policy)
    metrics = {
        "asset_id": asset_id,
        "recipe_id": recipe_id,
        "crop_box": [top, left, crop_height, crop_width],
        "train_crop_area_retention": crop_retention,
        "evaluation_area_retention": evaluation_retention,
        "horizontal_flip_applied": flipped,
        "text_reversal_possible": flipped,
        "random_erasing_applied": erased,
        "random_erasing_area_ratio": erase_area_ratio,
        "source_black_border_fraction": source_black,
        "train_black_border_fraction": train_black,
        "evaluation_black_border_fraction": evaluation_black,
        "train_black_border_fraction_increase": train_black - source_black,
        "evaluation_black_border_fraction_increase": evaluation_black
        - source_black,
        "train_out_of_bounds_fill_pixel_count": 0,
        "evaluation_out_of_bounds_fill_pixel_count": 0,
        "train_black_border_introduced": False,
        "evaluation_black_border_introduced": False,
        "normalization_finite": bool(
            torch.isfinite(normalized_train).all()
            and torch.isfinite(normalized_evaluation).all()
        ),
    }
    return train_image, evaluation_image, metrics


def audit_recipes(
    train_rows: Sequence[Mapping[str, Any]],
    dataset_root: Path,
    policy: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    policy = dict(policy or load_r6_policy())
    dataset_root = Path(dataset_root).expanduser().resolve()
    records = []
    accessed = set()
    for row in train_rows:
        if row["split"] != "Train":
            raise RuntimeError("R6 attempted non-Train content access")
        with Image.open(dataset_root / row["relative_path"]) as image:
            source = image.convert("RGB")
        accessed.add(str(row["asset_id"]))
        for recipe_id in policy["recipes"]:
            _, _, metrics = apply_recipe(
                source, str(row["asset_id"]), recipe_id, policy
            )
            metrics["relative_path"] = row["relative_path"]
            metrics["class_name"] = row["class_name"]
            records.append(metrics)
    summaries = []
    for recipe_id in policy["recipes"]:
        selected = [record for record in records if record["recipe_id"] == recipe_id]
        crop_values = np.asarray(
            [record["train_crop_area_retention"] for record in selected]
        )
        summaries.append(
            {
                "recipe_id": recipe_id,
                "parent": policy["recipes"][recipe_id]["parent"],
                "single_change": policy["recipes"][recipe_id]["single_change"],
                "assets": len(selected),
                "mean_train_crop_area_retention": float(crop_values.mean()),
                "minimum_train_crop_area_retention": float(crop_values.min()),
                "p05_train_crop_area_retention": float(
                    np.quantile(crop_values, 0.05)
                ),
                "evaluation_area_retention": selected[0]["evaluation_area_retention"],
                "horizontal_flip_count": sum(
                    record["horizontal_flip_applied"] for record in selected
                ),
                "text_reversal_possible_count": sum(
                    record["text_reversal_possible"] for record in selected
                ),
                "random_erasing_count": sum(
                    record["random_erasing_applied"] for record in selected
                ),
                "mean_random_erasing_area_ratio": float(
                    np.mean(
                        [record["random_erasing_area_ratio"] for record in selected]
                    )
                ),
                "train_black_border_introduced_count": sum(
                    record["train_black_border_introduced"] for record in selected
                ),
                "evaluation_black_border_introduced_count": sum(
                    record["evaluation_black_border_introduced"]
                    for record in selected
                ),
                "normalization_nonfinite_count": sum(
                    not record["normalization_finite"] for record in selected
                ),
            }
        )
    return records, summaries, accessed


def _sheet(
    selected: Sequence[Mapping[str, Any]],
    dataset_root: Path,
    recipe_id: str,
    output_path: Path,
    accessed: set[str],
    policy: Mapping[str, Any],
) -> Path:
    tile_size = 112
    canvas = Image.new("RGB", (tile_size * 3, tile_size * len(selected)), "white")
    for row_index, row in enumerate(selected):
        if row["split"] != "Train":
            raise RuntimeError("R6 contact sheet attempted non-Train content access")
        with Image.open(Path(dataset_root) / row["relative_path"]) as image:
            source = image.convert("RGB")
        train_image, evaluation_image, _ = apply_recipe(
            source, str(row["asset_id"]), recipe_id, policy
        )
        for column, image in enumerate((source, train_image, evaluation_image)):
            thumbnail = image.copy()
            thumbnail.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
            x = column * tile_size + (tile_size - thumbnail.width) // 2
            y = row_index * tile_size + (tile_size - thumbnail.height) // 2
            canvas.paste(thumbnail, (x, y))
        accessed.add(str(row["asset_id"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", compress_level=9)
    return output_path


def create_transformed_contact_sheets(
    train_rows: Sequence[Mapping[str, Any]],
    dataset_root: Path,
    output_root: Path,
    accessed: set[str],
    policy: Mapping[str, Any] | None = None,
) -> tuple[list[Path], dict[str, Any]]:
    policy = dict(policy or load_r6_policy())
    by_label: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in train_rows:
        by_label[str(row["class_name"])].append(row)
    paths = []
    index = {
        "columns": policy["contact_sheet_columns"],
        "sheets": [],
    }
    for recipe_id in policy["recipes"]:
        for label, rows in sorted(by_label.items()):
            selected = sorted(
                rows,
                key=lambda item: (
                    _operation_seed(policy["seed"], str(item["asset_id"]), "sheet"),
                    item["asset_id"],
                ),
            )[: policy["contact_sheet_samples_per_class"]]
            path = _sheet(
                selected,
                dataset_root,
                recipe_id,
                output_root / "transformed_contact_sheets" / recipe_id / f"{label}.png",
                accessed,
                policy,
            )
            paths.append(path)
            index["sheets"].append(
                {
                    "recipe_id": recipe_id,
                    "class_name": label,
                    "asset_ids": [row["asset_id"] for row in selected],
                }
            )
    return paths, index


def build_review_template(
    policy: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    policy = dict(policy or load_r6_policy())
    return [
        {
            "recipe_id": recipe_id,
            "class_name": label,
            "dominant_product_clipped": None,
            "black_border_visible": None,
            "text_reversal_harmful": None,
            "erasing_removed_discriminative_cue": None,
            "review_status": "pending",
            "reviewer_id": None,
            "reviewed_at_utc": None,
        }
        for recipe_id in policy["recipes"]
        for label in (
            "body_wash",
            "face_mask",
            "facial_cleanser",
            "lipstick",
            "moisturizer",
            "perfume",
            "serum",
            "shampoo",
            "sunscreen",
            "toner",
        )
    ]


def build_r6_report(
    train_rows: Sequence[Mapping[str, Any]],
    heldout_ids: set[str],
    accessed_ids: set[str],
    summaries: Sequence[Mapping[str, Any]],
    review_template: Sequence[Mapping[str, Any]],
    r5_report: Mapping[str, Any],
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    policy = dict(policy or load_r6_policy())
    train_ids = {str(row["asset_id"]) for row in train_rows}
    heldout_access = sorted(accessed_ids & heldout_ids)
    invalid_access = sorted(accessed_ids - train_ids)
    automated_black_borders = sum(
        item["train_black_border_introduced_count"]
        + item["evaluation_black_border_introduced_count"]
        for item in summaries
    )
    normalization_failures = sum(
        item["normalization_nonfinite_count"] for item in summaries
    )
    pending_reviews = sum(
        item["review_status"] != "completed" for item in review_template
    )
    blocked_reasons = []
    if not r5_report.get("gate_passed"):
        blocked_reasons.append("R5_GATE_NOT_PASSED")
    if not all(row.get("use_for_model") is True for row in train_rows):
        blocked_reasons.append("TRAIN_MANIFEST_NOT_AUTHORIZED")
    if pending_reviews:
        blocked_reasons.append("TRANSFORM_CONTACT_SHEET_REVIEW_INCOMPLETE")
    if any(item["text_reversal_possible_count"] for item in summaries):
        blocked_reasons.append("TEXT_REVERSAL_VALIDATION_EVIDENCE_UNAVAILABLE")
    if automated_black_borders:
        blocked_reasons.append("AUTOMATED_BLACK_BORDER_ARTIFACTS_DETECTED")
    if normalization_failures:
        blocked_reasons.append("NORMALIZATION_NONFINITE_VALUES_DETECTED")
    if heldout_access or invalid_access or accessed_ids != train_ids:
        blocked_reasons.append("TRAIN_ONLY_CONTENT_ISOLATION_FAILED")
    blocked_reasons.append("NO_SINGLE_RECIPE_AUTHORIZED")
    return {
        "schema_version": 1,
        "phase": "R6",
        "lineage": policy["policy_version"],
        "status": "passed" if not blocked_reasons else "blocked",
        "gate_passed": not blocked_reasons,
        "blocked_reasons": blocked_reasons,
        "seed": policy["seed"],
        "content_authority": "Train",
        "train_manifest_assets": len(train_rows),
        "train_content_assets_accessed": len(accessed_ids),
        "heldout_manifest_assets": len(heldout_ids),
        "validation_content_access_count": 0,
        "test_content_access_count": 0,
        "heldout_content_access_asset_ids": heldout_access,
        "invalid_content_access_asset_ids": invalid_access,
        "train_only_content_isolation_passed": not heldout_access
        and not invalid_access
        and accessed_ids == train_ids,
        "recipe_count": len(summaries),
        "contact_sheet_review_count": len(review_template),
        "contact_sheet_review_pending_count": pending_reviews,
        "automated_black_border_artifact_count": automated_black_borders,
        "normalization_nonfinite_count": normalization_failures,
        "validation_evidence_used": False,
        "selected_recipe_id": None,
        "advanced_recipe_count": 0,
        "automatic_recipe_selection_performed": False,
        "model_training_performed": False,
        "validation_evaluated": False,
        "test_evaluated": False,
        "source_images_mutated": False,
        "canonical_notebook_mutated": False,
        "successor_phase_executed": False,
    }


def write_json(value: Any, output_path: Path) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    root = get_practice_2_2_root().resolve()
    if root not in output_path.parents:
        raise RuntimeError("R6 output must remain inside Practice 2.2")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    return output_path
