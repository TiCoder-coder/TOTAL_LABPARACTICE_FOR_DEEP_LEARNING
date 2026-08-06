from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image

from .paths import get_practice_2_2_root


POLICY_RELATIVE_PATH = Path("configs/r5_data_readiness_policy.json")


def load_r5_policy(policy_path: Path | None = None) -> dict[str, Any]:
    root = get_practice_2_2_root()
    path = Path(policy_path or root / POLICY_RELATIVE_PATH).expanduser().resolve()
    policy = json.loads(path.read_text())
    if policy.get("schema_version") != 1:
        raise RuntimeError("Unsupported R5 data-readiness policy schema")
    if policy.get("seed") != 42:
        raise RuntimeError("R5 must use seed 42")
    if policy.get("authorized_content_split") != "Train":
        raise RuntimeError("R5 content authority must be Train only")
    if policy.get("predecessor_gate_required") is not True:
        raise RuntimeError("R5 must require the R4 gate")
    if policy.get("automatic_semantic_decisions_allowed") is not False:
        raise RuntimeError("R5 must prohibit automatic semantic decisions")
    if policy.get("automatic_quarantine_allowed") is not False:
        raise RuntimeError("R5 must prohibit automatic quarantine")
    if policy.get("validation_content_access_allowed") is not False:
        raise RuntimeError("R5 must prohibit Validation content access")
    if policy.get("test_content_access_allowed") is not False:
        raise RuntimeError("R5 must prohibit Test content access")
    if policy.get("model_training_allowed") is not False:
        raise RuntimeError("R5 must prohibit model training")
    return policy


def _seed_rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _cosine_vector(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return np.einsum("ij,j->i", matrix, vector, optimize=False)


def _cosine_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.einsum("ik,jk->ij", left, right, optimize=False)


def select_train_rows(
    manifest: Sequence[Mapping[str, Any]], policy: Mapping[str, Any] | None = None
) -> tuple[list[dict[str, Any]], set[str]]:
    policy = dict(policy or load_r5_policy())
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
    train_ids = [str(row["asset_id"]) for row in train_rows]
    if len(train_ids) != len(set(train_ids)) or set(train_ids) & heldout_ids:
        raise RuntimeError("R5 Train selection is not isolated")
    if any(row["split"] != "Train" for row in train_rows):
        raise RuntimeError("R5 selected non-Train content")
    return train_rows, heldout_ids


def _text_and_border_metrics(gray: np.ndarray) -> dict[str, float | bool]:
    height, width = gray.shape
    edges = cv2.Canny(gray, 100, 200) > 0
    border_width = max(2, int(round(min(height, width) * 0.06)))
    border_mask = np.zeros_like(edges, dtype=bool)
    border_mask[:border_width] = True
    border_mask[-border_width:] = True
    border_mask[:, :border_width] = True
    border_mask[:, -border_width:] = True
    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5)),
    )
    _, binary = cv2.threshold(
        blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    text_area = 0
    text_components = 0
    for index in range(1, count):
        x, y, component_width, component_height, area = stats[index]
        aspect = component_width / max(component_height, 1)
        if (
            6 <= area <= height * width * 0.08
            and 2 <= component_height <= height * 0.2
            and 0.5 <= aspect <= 20
            and x >= 0
            and y >= 0
        ):
            text_area += int(area)
            text_components += 1
    band = max(2, int(round(height * 0.18)))
    band_edge_density = float(
        np.concatenate((edges[:band].ravel(), edges[-band:].ravel())).mean()
    )
    center = edges[band:-band]
    center_edge_density = float(center.mean()) if center.size else float(edges.mean())
    return {
        "edge_density": float(edges.mean()),
        "border_edge_density": float(edges[border_mask].mean()),
        "text_like_area_ratio": text_area / float(height * width),
        "text_like_component_count": text_components,
        "band_edge_density": band_edge_density,
        "center_edge_density": center_edge_density,
    }


def audit_train_images(
    train_rows: Sequence[Mapping[str, Any]],
    inventory: Sequence[Mapping[str, Any]],
    dataset_root: Path,
    policy: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], set[str]]:
    policy = dict(policy or load_r5_policy())
    thresholds = policy["thresholds"]
    inventory_by_id = {str(record["asset_id"]): record for record in inventory}
    dataset_root = Path(dataset_root).expanduser().resolve()
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_square_sum = np.zeros(3, dtype=np.float64)
    total_pixels = 0
    accessed = set()
    audited = []
    for row in train_rows:
        if row["split"] != "Train":
            raise RuntimeError("R5 attempted non-Train image access")
        asset_id = str(row["asset_id"])
        path = dataset_root / row["relative_path"]
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        accessed.add(asset_id)
        normalized = rgb.astype(np.float64) / 255.0
        pixels = rgb.shape[0] * rgb.shape[1]
        channel_sum += normalized.sum(axis=(0, 1))
        channel_square_sum += np.square(normalized).sum(axis=(0, 1))
        total_pixels += pixels
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        visual = _text_and_border_metrics(gray)
        source = inventory_by_id[asset_id]
        reasons = list(source["quality_flags"])
        if visual["text_like_area_ratio"] >= thresholds[
            "text_like_area_review_ratio"
        ]:
            reasons.append("TEXT_LIKE_AREA_HIGH")
        if visual["border_edge_density"] >= thresholds[
            "border_edge_crop_review_ratio"
        ]:
            reasons.append("BORDER_CROP_RISK")
        multiplier = thresholds["banner_band_edge_multiplier"]
        banner_candidate = (
            visual["band_edge_density"]
            >= multiplier * max(visual["center_edge_density"], 1e-6)
            and visual["text_like_area_ratio"] > 0.02
        )
        if banner_candidate:
            reasons.append("BANNER_BAND_CANDIDATE")
        audited.append(
            {
                "asset_id": asset_id,
                "relative_path": row["relative_path"],
                "class_name": row["class_name"],
                "component_id": row["component_id"],
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
                "brightness": source["brightness"],
                "contrast": source["contrast"],
                "laplacian_variance": source["laplacian_variance"],
                "content_ratio": source["content_ratio"],
                **visual,
                "banner_band_candidate": banner_candidate,
                "automated_review_reasons": sorted(set(reasons)),
            }
        )
    mean = channel_sum / total_pixels
    variance = channel_square_sum / total_pixels - np.square(mean)
    pixel_statistics = {
        "source_split": "Train",
        "image_count": len(audited),
        "pixel_count": total_pixels,
        "rgb_mean": mean.tolist(),
        "rgb_std": np.sqrt(np.maximum(variance, 0.0)).tolist(),
        "minimum_width": min(item["width"] for item in audited),
        "maximum_width": max(item["width"] for item in audited),
        "minimum_height": min(item["height"] for item in audited),
        "maximum_height": max(item["height"] for item in audited),
        "mean_brightness": float(np.mean([item["brightness"] for item in audited])),
        "mean_contrast": float(np.mean([item["contrast"] for item in audited])),
        "validation_images_used": 0,
        "test_images_used": 0,
    }
    return audited, pixel_statistics, accessed


def deterministic_kmeans(
    embeddings: np.ndarray,
    asset_ids: Sequence[str],
    cluster_count: int,
    seed: int,
    maximum_iterations: int = 100,
) -> tuple[np.ndarray, np.ndarray, int]:
    if embeddings.shape[0] != len(asset_ids) or cluster_count >= len(asset_ids):
        raise RuntimeError("R5 embedding clustering input is invalid")
    generator = np.random.default_rng(seed)
    first_index = int(generator.integers(len(asset_ids)))
    center_indexes = [first_index]
    closest_distance = 1.0 - _cosine_vector(embeddings, embeddings[first_index])
    while len(center_indexes) < cluster_count:
        weights = np.square(np.maximum(closest_distance, 0.0))
        weights[center_indexes] = 0.0
        total = float(weights.sum())
        if total == 0:
            remaining = [index for index in range(len(asset_ids)) if index not in center_indexes]
            selected = min(remaining, key=lambda index: asset_ids[index])
        else:
            selected = int(generator.choice(len(asset_ids), p=weights / total))
        center_indexes.append(selected)
        closest_distance = np.minimum(
            closest_distance,
            1.0 - _cosine_vector(embeddings, embeddings[selected]),
        )
    centers = embeddings[center_indexes].copy()
    previous = None
    for iteration in range(1, maximum_iterations + 1):
        assignments = np.argmax(_cosine_matrix(embeddings, centers), axis=1)
        if previous is not None and np.array_equal(assignments, previous):
            return assignments, centers, iteration
        previous = assignments.copy()
        updated = []
        similarities = _cosine_matrix(embeddings, centers)
        for cluster_id in range(cluster_count):
            members = embeddings[assignments == cluster_id]
            if len(members) == 0:
                nearest = similarities.max(axis=1)
                selected = min(
                    range(len(asset_ids)),
                    key=lambda index: (nearest[index], asset_ids[index]),
                )
                center = embeddings[selected].copy()
            else:
                center = members.mean(axis=0)
            norm = float(np.linalg.norm(center))
            updated.append(center / max(norm, 1e-12))
        centers = np.asarray(updated, dtype=np.float32)
    return assignments, centers, maximum_iterations


def build_embedding_audits(
    train_rows: Sequence[Mapping[str, Any]],
    stored_asset_ids: Sequence[str],
    stored_embeddings: np.ndarray,
    policy: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:
    policy = dict(policy or load_r5_policy())
    row_by_id = {str(row["asset_id"]): row for row in train_rows}
    embedding_index = {
        str(asset_id): index for index, asset_id in enumerate(stored_asset_ids)
    }
    asset_ids = sorted(row_by_id)
    if any(asset_id not in embedding_index for asset_id in asset_ids):
        raise RuntimeError("R5 Train embeddings are incomplete")
    embeddings = np.asarray(
        [stored_embeddings[embedding_index[asset_id]] for asset_id in asset_ids]
    )
    if len(embeddings) != len(train_rows):
        raise RuntimeError("R5 Train embeddings are incomplete")
    assignments, centers, iterations = deterministic_kmeans(
        embeddings,
        asset_ids,
        policy["embedding_cluster_count"],
        policy["seed"],
    )
    clusters = []
    for cluster_id in range(policy["embedding_cluster_count"]):
        indexes = np.flatnonzero(assignments == cluster_id)
        labels = Counter(row_by_id[asset_ids[index]]["class_name"] for index in indexes)
        probabilities = np.asarray(list(labels.values()), dtype=np.float64) / len(indexes)
        entropy = float(-(probabilities * np.log2(probabilities)).sum())
        similarities = _cosine_vector(embeddings[indexes], centers[cluster_id])
        representative_order = sorted(
            range(len(indexes)),
            key=lambda position: (
                -float(similarities[position]),
                asset_ids[int(indexes[position])],
            ),
        )
        clusters.append(
            {
                "cluster_id": cluster_id,
                "size": len(indexes),
                "label_counts": dict(sorted(labels.items())),
                "dominant_label": labels.most_common(1)[0][0],
                "dominant_label_share": labels.most_common(1)[0][1] / len(indexes),
                "label_entropy_bits": entropy,
                "representative_asset_ids": [
                    asset_ids[int(indexes[position])]
                    for position in representative_order[
                        : policy["cluster_contact_sheet_samples"]
                    ]
                ],
            }
        )
    similarity = _cosine_matrix(embeddings, embeddings)
    np.fill_diagonal(similarity, -1.0)
    neighbor_count = policy["embedding_neighbor_count"]
    neighborhoods = []
    cross_label_pairs = {}
    confusion_counts = Counter()
    top1_cross_label = 0
    for left_index, asset_id in enumerate(asset_ids):
        candidate_indexes = np.argpartition(
            similarity[left_index], -neighbor_count
        )[-neighbor_count:]
        ordered = sorted(
            candidate_indexes,
            key=lambda right_index: (
                -float(similarity[left_index, right_index]),
                asset_ids[int(right_index)],
            ),
        )
        left_label = row_by_id[asset_id]["class_name"]
        neighbors = []
        for rank, right_index in enumerate(ordered, start=1):
            right_id = asset_ids[int(right_index)]
            right_label = row_by_id[right_id]["class_name"]
            score = float(similarity[left_index, right_index])
            cross_label = left_label != right_label
            neighbors.append(
                {
                    "rank": rank,
                    "asset_id": right_id,
                    "class_name": right_label,
                    "cosine_similarity": round(score, 8),
                    "cross_label": cross_label,
                }
            )
            if cross_label:
                label_pair = tuple(sorted((left_label, right_label)))
                confusion_counts[label_pair] += 1
                pair = tuple(sorted((asset_id, right_id)))
                cross_label_pairs[pair] = max(score, cross_label_pairs.get(pair, -1.0))
        if neighbors[0]["cross_label"]:
            top1_cross_label += 1
        neighborhoods.append(
            {
                "asset_id": asset_id,
                "class_name": left_label,
                "neighbors": neighbors,
            }
        )
    ordered_pairs = sorted(
        (
            {
                "left_asset_id": pair[0],
                "right_asset_id": pair[1],
                "left_class_name": row_by_id[pair[0]]["class_name"],
                "right_class_name": row_by_id[pair[1]]["class_name"],
                "cosine_similarity": round(score, 8),
            }
            for pair, score in cross_label_pairs.items()
        ),
        key=lambda item: (
            -item["cosine_similarity"],
            item["left_asset_id"],
            item["right_asset_id"],
        ),
    )
    cluster_report = {
        "source_split": "Train",
        "asset_count": len(asset_ids),
        "cluster_count": len(clusters),
        "iterations": iterations,
        "clusters": clusters,
        "assignments": [
            {"asset_id": asset_id, "cluster_id": int(assignments[index])}
            for index, asset_id in enumerate(asset_ids)
        ],
        "validation_embeddings_used": 0,
        "test_embeddings_used": 0,
    }
    neighbor_report = {
        "source_split": "Train",
        "asset_count": len(asset_ids),
        "neighbor_count": neighbor_count,
        "top1_cross_label_count": top1_cross_label,
        "top1_cross_label_ratio": top1_cross_label / len(asset_ids),
        "cross_label_pair_count": len(ordered_pairs),
        "cross_label_pairs": ordered_pairs,
        "confusion_pair_counts": {
            "|".join(pair): count for pair, count in sorted(confusion_counts.items())
        },
        "neighborhoods": neighborhoods,
        "validation_embeddings_used": 0,
        "test_embeddings_used": 0,
    }
    return cluster_report, neighbor_report, embeddings, centers


def _make_contact_sheet(
    rows: Sequence[Mapping[str, Any]],
    dataset_root: Path,
    output_path: Path,
    columns: int,
    accessed: set[str],
    tile_size: int = 112,
) -> Path:
    row_count = max(1, math.ceil(len(rows) / columns))
    canvas = Image.new("RGB", (columns * tile_size, row_count * tile_size), "white")
    for index, row in enumerate(rows):
        if row["split"] != "Train":
            raise RuntimeError("R5 contact sheet attempted non-Train content access")
        with Image.open(Path(dataset_root) / row["relative_path"]) as image:
            thumbnail = image.convert("RGB")
            thumbnail.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
        x = (index % columns) * tile_size + (tile_size - thumbnail.width) // 2
        y = (index // columns) * tile_size + (tile_size - thumbnail.height) // 2
        canvas.paste(thumbnail, (x, y))
        accessed.add(str(row["asset_id"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", compress_level=9)
    return output_path


def create_contact_sheets(
    train_rows: Sequence[Mapping[str, Any]],
    cluster_report: Mapping[str, Any],
    neighbor_report: Mapping[str, Any],
    dataset_root: Path,
    output_root: Path,
    accessed: set[str],
    policy: Mapping[str, Any] | None = None,
) -> tuple[list[Path], dict[str, Any]]:
    policy = dict(policy or load_r5_policy())
    rows_by_id = {str(row["asset_id"]): row for row in train_rows}
    by_label: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in train_rows:
        by_label[str(row["class_name"])].append(row)
    paths = []
    index = {"class_sheets": [], "cluster_sheets": [], "overlap_sheet": None}
    for label, rows in sorted(by_label.items()):
        selected = sorted(
            rows,
            key=lambda item: (
                _seed_rank(policy["seed"], str(item["asset_id"])),
                item["asset_id"],
            ),
        )[: policy["contact_sheet_samples_per_class"]]
        path = _make_contact_sheet(
            selected,
            dataset_root,
            output_root / "class_contact_sheets" / f"{label}.png",
            policy["contact_sheet_columns"],
            accessed,
        )
        paths.append(path)
        index["class_sheets"].append(
            {"class_name": label, "asset_ids": [row["asset_id"] for row in selected]}
        )
    for cluster in cluster_report["clusters"]:
        selected = [rows_by_id[asset_id] for asset_id in cluster["representative_asset_ids"]]
        path = _make_contact_sheet(
            selected,
            dataset_root,
            output_root / "cluster_contact_sheets" / f"cluster_{cluster['cluster_id']:02d}.png",
            4,
            accessed,
        )
        paths.append(path)
        index["cluster_sheets"].append(
            {
                "cluster_id": cluster["cluster_id"],
                "asset_ids": cluster["representative_asset_ids"],
            }
        )
    pair_count = policy["cross_label_pair_sheet_count"]
    pair_rows = []
    pair_ids = []
    for pair in neighbor_report["cross_label_pairs"][:pair_count]:
        pair_rows.extend(
            [rows_by_id[pair["left_asset_id"]], rows_by_id[pair["right_asset_id"]]]
        )
        pair_ids.append([pair["left_asset_id"], pair["right_asset_id"]])
    overlap_path = _make_contact_sheet(
        pair_rows,
        dataset_root,
        output_root / "cross_label_neighbor_pairs.png",
        2,
        accessed,
    )
    paths.append(overlap_path)
    index["overlap_sheet"] = {"pairs": pair_ids}
    return paths, index


def build_class_readiness(
    train_rows: Sequence[Mapping[str, Any]],
    audited: Sequence[Mapping[str, Any]],
    cluster_report: Mapping[str, Any],
    policy: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    policy = dict(policy or load_r5_policy())
    audit_by_id = {str(item["asset_id"]): item for item in audited}
    cluster_by_id = {
        item["asset_id"]: item["cluster_id"]
        for item in cluster_report["assignments"]
    }
    result = []
    for label in sorted({row["class_name"] for row in train_rows}):
        rows = [row for row in train_rows if row["class_name"] == label]
        component_counts = Counter(row["component_id"] for row in rows)
        source_counts = Counter(row["source_group"] for row in rows)
        cluster_counts = Counter(cluster_by_id[row["asset_id"]] for row in rows)
        flags = Counter(
            reason
            for row in rows
            for reason in audit_by_id[row["asset_id"]]["automated_review_reasons"]
        )
        result.append(
            {
                "class_name": label,
                "images": len(rows),
                "components": len(component_counts),
                "component_diversity_ratio": len(component_counts) / len(rows),
                "largest_component_size": max(component_counts.values()),
                "largest_component_share": max(component_counts.values()) / len(rows),
                "source_groups": len(source_counts),
                "largest_source_group_share": max(source_counts.values()) / len(rows),
                "unique_products": None,
                "vendor_count": None,
                "largest_embedding_cluster_share": max(cluster_counts.values())
                / len(rows),
                "automated_review_reason_counts": dict(sorted(flags.items())),
            }
        )
    return result


def build_train_review_queue(
    train_rows: Sequence[Mapping[str, Any]],
    audited: Sequence[Mapping[str, Any]],
    neighbor_report: Mapping[str, Any],
    policy: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    policy = dict(policy or load_r5_policy())
    audit_by_id = {str(item["asset_id"]): item for item in audited}
    neighborhood_by_id = {
        str(item["asset_id"]): item for item in neighbor_report["neighborhoods"]
    }
    threshold = policy["thresholds"]["cross_label_neighbor_review_cosine"]
    queue = []
    for row in train_rows:
        asset_id = str(row["asset_id"])
        audit = audit_by_id[asset_id]
        neighbors = neighborhood_by_id[asset_id]["neighbors"]
        nearest_cross_label = next(
            (item for item in neighbors if item["cross_label"]), None
        )
        reasons = set(audit["automated_review_reasons"])
        if nearest_cross_label and nearest_cross_label["cosine_similarity"] >= threshold:
            reasons.add("HIGH_SIMILARITY_CROSS_LABEL_NEIGHBOR")
        priority = "P1" if reasons else "P2"
        queue.append(
            {
                "asset_id": asset_id,
                "relative_path": row["relative_path"],
                "original_label": row["class_name"],
                "component_id": row["component_id"],
                "priority": priority,
                "automated_review_reasons": sorted(reasons),
                "automated_metrics": {
                    "brightness": audit["brightness"],
                    "contrast": audit["contrast"],
                    "laplacian_variance": audit["laplacian_variance"],
                    "text_like_area_ratio": audit["text_like_area_ratio"],
                    "border_edge_density": audit["border_edge_density"],
                    "banner_band_candidate": audit["banner_band_candidate"],
                    "nearest_cross_label_neighbor": nearest_cross_label,
                },
                "review_status": "pending",
                "semantic_decision": None,
                "proposed_label": None,
                "reason_code": None,
                "dominant_product_visible": None,
                "text_role": None,
                "crop_loss": None,
                "watermark_present": None,
                "banner_present": None,
                "bundle_present": None,
                "bundle_policy_compliant": None,
                "people_only": None,
                "reviewer_id": None,
                "reviewed_at_utc": None,
            }
        )
    return sorted(
        queue,
        key=lambda item: (item["priority"], item["original_label"], item["asset_id"]),
    )


def build_r5_report(
    train_rows: Sequence[Mapping[str, Any]],
    heldout_ids: set[str],
    accessed_ids: set[str],
    class_readiness: Sequence[Mapping[str, Any]],
    review_queue: Sequence[Mapping[str, Any]],
    r4_report: Mapping[str, Any],
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    policy = dict(policy or load_r5_policy())
    thresholds = policy["thresholds"]
    train_ids = {str(row["asset_id"]) for row in train_rows}
    invalid_access = sorted(accessed_ids - train_ids)
    heldout_access = sorted(accessed_ids & heldout_ids)
    priority_counts = Counter(item["priority"] for item in review_queue)
    pending = sum(item["review_status"] != "completed" for item in review_queue)
    dominant_source_template = any(
        item["largest_source_group_share"] > thresholds["maximum_source_group_share"]
        or item["largest_component_share"]
        > thresholds["maximum_duplicate_component_share"]
        for item in class_readiness
    )
    diversity_failed = any(
        item["component_diversity_ratio"]
        < thresholds["minimum_component_diversity_ratio"]
        for item in class_readiness
    )
    embedding_concentration = any(
        item["largest_embedding_cluster_share"]
        > thresholds["maximum_embedding_cluster_class_share"]
        for item in class_readiness
    )
    blocked_reasons = []
    if not r4_report.get("gate_passed"):
        blocked_reasons.append("R4_GATE_NOT_PASSED")
    if not all(row.get("use_for_model") is True for row in train_rows):
        blocked_reasons.append("TRAIN_MANIFEST_NOT_AUTHORIZED")
    if any(row.get("product_id") in {None, ""} for row in train_rows):
        blocked_reasons.append("PRODUCT_PROVENANCE_INCOMPLETE")
    if pending:
        blocked_reasons.append("TRAIN_SEMANTIC_REVIEW_INCOMPLETE")
    if priority_counts.get("P1", 0):
        blocked_reasons.append("CRITICAL_REVIEW_CANDIDATES_UNRESOLVED")
    if any(item["vendor_count"] is None for item in class_readiness):
        blocked_reasons.append("SOURCE_VENDOR_PROVENANCE_INCOMPLETE")
    if dominant_source_template:
        blocked_reasons.append("DOMINANT_SOURCE_TEMPLATE_DETECTED")
    if diversity_failed:
        blocked_reasons.append("COMPONENT_DIVERSITY_BELOW_THRESHOLD")
    if embedding_concentration:
        blocked_reasons.append("EMBEDDING_CLUSTER_CONCENTRATION_EXCEEDED")
    if invalid_access or heldout_access or accessed_ids != train_ids:
        blocked_reasons.append("TRAIN_ONLY_CONTENT_ISOLATION_FAILED")
    return {
        "schema_version": 1,
        "phase": "R5",
        "lineage": policy["policy_version"],
        "predecessor_lineage": policy["predecessor_lineage"],
        "status": "passed" if not blocked_reasons else "blocked",
        "gate_passed": not blocked_reasons,
        "data_ready": not blocked_reasons,
        "blocked_reasons": blocked_reasons,
        "seed": policy["seed"],
        "content_authority": "Train",
        "train_manifest_assets": len(train_rows),
        "train_unique_assets": len(train_ids),
        "train_content_assets_accessed": len(accessed_ids),
        "heldout_manifest_assets": len(heldout_ids),
        "validation_content_access_count": 0,
        "test_content_access_count": 0,
        "invalid_content_access_asset_ids": invalid_access,
        "heldout_content_access_asset_ids": heldout_access,
        "train_only_content_isolation_passed": not invalid_access
        and not heldout_access
        and accessed_ids == train_ids,
        "classes": len(class_readiness),
        "semantic_review_queue_count": len(review_queue),
        "semantic_review_pending_count": pending,
        "review_priority_counts": dict(sorted(priority_counts.items())),
        "product_provenance_coverage": sum(
            row.get("product_id") not in {None, ""} for row in train_rows
        )
        / len(train_rows),
        "dominant_source_template_detected": dominant_source_template,
        "component_diversity_threshold_passed": not diversity_failed,
        "embedding_cluster_concentration_passed": not embedding_concentration,
        "automatic_semantic_decisions_performed": False,
        "automatic_quarantine_performed": False,
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
        raise RuntimeError("R5 output must remain inside Practice 2.2")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    return output_path
