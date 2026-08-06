from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from .paths import get_practice_2_2_root
from .resources import file_sha256


POLICY_RELATIVE_PATH = Path("configs/r3_grouping_policy.json")
EXPECTED_WEIGHT_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"


def load_r3_policy(policy_path: Path | None = None) -> dict[str, Any]:
    root = get_practice_2_2_root()
    path = Path(policy_path or root / POLICY_RELATIVE_PATH).expanduser().resolve()
    policy = json.loads(path.read_text())
    if policy.get("schema_version") != 1:
        raise RuntimeError("Unsupported R3 grouping policy schema")
    if policy.get("seed") != 42:
        raise RuntimeError("R3 grouping must use seed 42")
    if policy.get("generated_assets_excluded") is not True:
        raise RuntimeError("R3 must exclude generated assets")
    if policy.get("split_creation_allowed") is not False:
        raise RuntimeError("R3 must not create data splits")
    if policy.get("automatic_deletion_allowed") is not False:
        raise RuntimeError("R3 must prohibit automatic deletion")
    if policy.get("automatic_quarantine_allowed") is not False:
        raise RuntimeError("R3 must prohibit automatic quarantine")
    if policy.get("cross_label_union_allowed") is not False:
        raise RuntimeError("R3 must prohibit cross-label unions")
    return policy


class DeterministicUnionFind:
    def __init__(self, asset_ids: Iterable[str]) -> None:
        values = sorted(set(asset_ids))
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self.parent[second] = first

    def components(self) -> list[list[str]]:
        values: dict[str, list[str]] = defaultdict(list)
        for asset_id in sorted(self.parent):
            values[self.find(asset_id)].append(asset_id)
        return sorted((sorted(members) for members in values.values()), key=lambda x: x[0])


def stable_component_id(asset_ids: Sequence[str], prefix: str = "r3g_") -> str:
    payload = "\n".join(sorted(asset_ids)).encode()
    return f"{prefix}{hashlib.sha256(payload).hexdigest()[:20]}"


def stable_edge_id(left_asset_id: str, right_asset_id: str) -> str:
    payload = "\n".join(sorted((left_asset_id, right_asset_id))).encode()
    return f"r3e_{hashlib.sha256(payload).hexdigest()[:20]}"


def _phash(path: Path, hash_size: int) -> int:
    with Image.open(path) as image:
        gray = np.asarray(
            image.convert("L").resize(
                (hash_size * 4, hash_size * 4), Image.Resampling.LANCZOS
            ),
            dtype=np.float32,
        )
    transformed = cv2.dct(gray)[:hash_size, :hash_size]
    values = transformed.ravel()[1:]
    threshold = float(np.median(values))
    value = 0
    for bit in transformed > threshold:
        for enabled in bit:
            value = (value << 1) | int(enabled)
    return value


def _load_gray(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(
            image.convert("L").resize((size, size), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )


def structural_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_mean = cv2.GaussianBlur(left, (11, 11), 1.5)
    right_mean = cv2.GaussianBlur(right, (11, 11), 1.5)
    left_square = left_mean * left_mean
    right_square = right_mean * right_mean
    product = left_mean * right_mean
    left_variance = cv2.GaussianBlur(left * left, (11, 11), 1.5) - left_square
    right_variance = cv2.GaussianBlur(right * right, (11, 11), 1.5) - right_square
    covariance = cv2.GaussianBlur(left * right, (11, 11), 1.5) - product
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    numerator = (2 * product + c1) * (2 * covariance + c2)
    denominator = (left_square + right_square + c1) * (
        left_variance + right_variance + c2
    )
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


class _EmbeddingDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        dataset_root: Path,
        transform: Any,
    ) -> None:
        self.records = list(records)
        self.dataset_root = dataset_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        record = self.records[index]
        with Image.open(self.dataset_root / record["relative_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, str(record["asset_id"])


def extract_pretrained_embeddings(
    records: Sequence[Mapping[str, Any]],
    dataset_root: Path,
    checkpoint_path: Path,
    batch_size: int = 64,
) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint_sha256 = file_sha256(checkpoint_path)
    if checkpoint_sha256 != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("R3 pretrained checkpoint SHA-256 mismatch")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.fc = nn.Identity()
    model.eval()
    ordered = sorted(records, key=lambda item: item["asset_id"])
    dataset = _EmbeddingDataset(ordered, Path(dataset_root), weights.transforms())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    asset_ids = []
    batches = []
    with torch.inference_mode():
        for images, batch_asset_ids in loader:
            features = torch.nn.functional.normalize(model(images), dim=1)
            batches.append(features.cpu().numpy().astype(np.float32))
            asset_ids.extend(batch_asset_ids)
    embeddings = np.concatenate(batches, axis=0)
    metadata = {
        "model": "resnet18",
        "weights": "IMAGENET1K_V1",
        "checkpoint_sha256": checkpoint_sha256,
        "embedding_dimension": int(embeddings.shape[1]),
        "asset_count": len(asset_ids),
        "normalized": True,
        "device": "cpu",
    }
    return asset_ids, embeddings, metadata


def embedding_neighbor_pairs(
    asset_ids: Sequence[str],
    embeddings: np.ndarray,
    neighbor_count: int,
    minimum_cosine: float,
) -> dict[tuple[str, str], float]:
    if embeddings.shape[0] != len(asset_ids):
        raise RuntimeError("R3 embedding row count does not match asset IDs")
    values = torch.from_numpy(embeddings)
    similarities = values @ values.T
    similarities.fill_diagonal_(-1.0)
    count = min(neighbor_count, max(0, len(asset_ids) - 1))
    scores, indexes = torch.topk(similarities, k=count, dim=1)
    pairs: dict[tuple[str, str], float] = {}
    for left_index, left_id in enumerate(asset_ids):
        for score, right_index in zip(scores[left_index], indexes[left_index]):
            value = float(score)
            if value < minimum_cosine:
                continue
            pair = tuple(sorted((left_id, asset_ids[int(right_index)])))
            pairs[pair] = max(value, pairs.get(pair, -1.0))
    return pairs


def _identity_edges(
    records: Sequence[Mapping[str, Any]],
    identity_keys: Sequence[str],
) -> list[dict[str, Any]]:
    edges = []
    for key in identity_keys:
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for record in records:
            value = record.get(key)
            if value not in {None, ""}:
                groups[str(value)].append(record)
        for value, members in sorted(groups.items()):
            ordered = sorted(members, key=lambda item: item["asset_id"])
            anchor = ordered[0]
            for right in ordered[1:]:
                edges.append(
                    {
                        "left_asset_id": anchor["asset_id"],
                        "right_asset_id": right["asset_id"],
                        "left_label": anchor["class_name"],
                        "right_label": right["class_name"],
                        "edge_type": "identity",
                        "identity_key": key,
                        "identity_value_sha256": hashlib.sha256(value.encode()).hexdigest(),
                    }
                )
    return edges


def build_visual_edges(
    records: Sequence[Mapping[str, Any]],
    dataset_root: Path,
    asset_ids: Sequence[str],
    embeddings: np.ndarray,
    policy: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    evidence = policy["visual_evidence"]
    records_by_id = {str(record["asset_id"]): record for record in records}
    phashes = {
        asset_id: _phash(
            Path(dataset_root) / records_by_id[asset_id]["relative_path"],
            evidence["phash_size"],
        )
        for asset_id in sorted(records_by_id)
    }
    candidates = embedding_neighbor_pairs(
        asset_ids,
        embeddings,
        evidence["embedding_neighbor_count"],
        evidence["embedding_candidate_cosine"],
    )
    ordered_ids = sorted(records_by_id)
    for left_index, left_id in enumerate(ordered_ids):
        left_record = records_by_id[left_id]
        left_dhash = int(left_record["dhash64"], 16)
        for right_id in ordered_ids[left_index + 1 :]:
            right_record = records_by_id[right_id]
            phash_distance = (phashes[left_id] ^ phashes[right_id]).bit_count()
            dhash_distance = (
                left_dhash ^ int(right_record["dhash64"], 16)
            ).bit_count()
            if (
                phash_distance <= evidence["phash_candidate_distance"]
                or dhash_distance <= evidence["dhash_candidate_distance"]
            ):
                candidates.setdefault((left_id, right_id), None)
    embedding_index = {asset_id: index for index, asset_id in enumerate(asset_ids)}
    gray_cache = {
        asset_id: _load_gray(
            Path(dataset_root) / records_by_id[asset_id]["relative_path"],
            evidence["ssim_image_size"],
        )
        for pair in candidates
        for asset_id in pair
    }
    confirmed = evidence["confirmed_same_label"]
    edges = []
    for left_id, right_id in sorted(candidates):
        left = records_by_id[left_id]
        right = records_by_id[right_id]
        phash_distance = (phashes[left_id] ^ phashes[right_id]).bit_count()
        dhash_distance = (
            int(left["dhash64"], 16) ^ int(right["dhash64"], 16)
        ).bit_count()
        embedding_cosine = float(
            embeddings[embedding_index[left_id]] @ embeddings[embedding_index[right_id]]
        )
        ssim = structural_similarity(gray_cache[left_id], gray_cache[right_id])
        same_label = left["class_name"] == right["class_name"]
        auto_confirmed = (
            same_label
            and phash_distance <= confirmed["maximum_phash_distance"]
            and dhash_distance <= confirmed["maximum_dhash_distance"]
            and ssim >= confirmed["minimum_ssim"]
            and embedding_cosine >= confirmed["minimum_embedding_cosine"]
        )
        edges.append(
            {
                "left_asset_id": left_id,
                "right_asset_id": right_id,
                "left_relative_path": left["relative_path"],
                "right_relative_path": right["relative_path"],
                "left_label": left["class_name"],
                "right_label": right["class_name"],
                "phash_distance": phash_distance,
                "dhash_distance": dhash_distance,
                "ssim": round(ssim, 8),
                "embedding_cosine": round(embedding_cosine, 8),
                "same_label": same_label,
                "auto_confirmed": auto_confirmed,
                "review_status": "not_required" if auto_confirmed else "pending",
                "review_decision": "confirm_visual_duplicate" if auto_confirmed else None,
                "reviewer_id": "r3_threshold_policy" if auto_confirmed else None,
            }
        )
    return edges, {asset_id: f"{value:016x}" for asset_id, value in phashes.items()}


def build_groups(
    records: Sequence[Mapping[str, Any]],
    identity_edges: Sequence[Mapping[str, Any]],
    visual_edges: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    records_by_id = {str(record["asset_id"]): record for record in records}
    union_find = DeterministicUnionFind(records_by_id)
    review_edges = []
    accepted_edges = []
    for edge in identity_edges:
        same_label = edge["left_label"] == edge["right_label"]
        completed = dict(edge)
        completed["edge_id"] = stable_edge_id(
            edge["left_asset_id"], edge["right_asset_id"]
        )
        completed["same_label"] = same_label
        completed["auto_confirmed"] = same_label
        completed["review_status"] = "not_required" if same_label else "pending"
        completed["review_decision"] = "confirm_same_product" if same_label else None
        completed["reviewer_id"] = "r3_identity_policy" if same_label else None
        completed["reviewed_at_utc"] = None
        completed["review_notes"] = None
        if same_label:
            union_find.union(edge["left_asset_id"], edge["right_asset_id"])
            accepted_edges.append(completed)
        else:
            review_edges.append(completed)
    for edge in visual_edges:
        completed = dict(edge)
        completed["edge_id"] = stable_edge_id(
            edge["left_asset_id"], edge["right_asset_id"]
        )
        completed["reviewed_at_utc"] = None
        completed["review_notes"] = None
        if edge["auto_confirmed"]:
            union_find.union(edge["left_asset_id"], edge["right_asset_id"])
            accepted_edges.append(completed)
        else:
            review_edges.append(completed)
    components = []
    for members in union_find.components():
        labels = sorted({records_by_id[asset_id]["class_name"] for asset_id in members})
        component_id = stable_component_id(members, policy["component_id_prefix"])
        components.append(
            {
                "component_id": component_id,
                "member_asset_ids": members,
                "member_relative_paths": [
                    records_by_id[asset_id]["relative_path"] for asset_id in members
                ],
                "labels": labels,
                "member_count": len(members),
            }
        )
    components.sort(key=lambda item: item["component_id"])
    review_edges.sort(
        key=lambda item: (
            item["left_label"] == item["right_label"],
            item.get("phash_distance", -1),
            item["left_asset_id"],
            item["right_asset_id"],
        )
    )
    return components, accepted_edges, review_edges


def build_group_report(
    records: Sequence[Mapping[str, Any]],
    components: Sequence[Mapping[str, Any]],
    accepted_edges: Sequence[Mapping[str, Any]],
    review_edges: Sequence[Mapping[str, Any]],
    embedding_metadata: Mapping[str, Any],
    predecessor_report: Mapping[str, Any],
    dataset_sha256: str,
) -> dict[str, Any]:
    original_count = len(records)
    assigned = [asset_id for component in components for asset_id in component["member_asset_ids"]]
    cross_label_components = [component for component in components if len(component["labels"]) > 1]
    product_coverage = sum(record.get("product_id") not in {None, ""} for record in records)
    pending_cross_label = sum(
        edge["left_label"] != edge["right_label"] and edge["review_status"] == "pending"
        for edge in review_edges
    )
    pending_same_label = sum(
        edge["left_label"] == edge["right_label"] and edge["review_status"] == "pending"
        for edge in review_edges
    )
    blocked_reasons = []
    if not predecessor_report.get("gate_passed"):
        blocked_reasons.append("R2_GATE_NOT_PASSED")
    if product_coverage != original_count:
        blocked_reasons.append("PRODUCT_PROVENANCE_INCOMPLETE")
    if pending_cross_label:
        blocked_reasons.append("CROSS_LABEL_VISUAL_REVIEW_INCOMPLETE")
    if pending_same_label:
        blocked_reasons.append("UNCERTAIN_SAME_LABEL_EDGE_REVIEW_INCOMPLETE")
    if len(assigned) != original_count or len(set(assigned)) != original_count:
        blocked_reasons.append("COMPONENT_ASSIGNMENT_INVALID")
    if cross_label_components:
        blocked_reasons.append("CROSS_LABEL_COMPONENT_PRESENT")
    class_component_counts = Counter(
        component["labels"][0] for component in components if len(component["labels"]) == 1
    )
    return {
        "schema_version": 1,
        "phase": "R3",
        "lineage": "r3_product_visual_grouping_v1",
        "status": "passed" if not blocked_reasons else "blocked",
        "gate_passed": not blocked_reasons,
        "blocked_reasons": blocked_reasons,
        "seed": 42,
        "dataset_directory_sha256": dataset_sha256,
        "eligible_original_assets": original_count,
        "generated_assets_excluded": True,
        "component_count": len(components),
        "singleton_component_count": sum(component["member_count"] == 1 for component in components),
        "largest_component_size": max(component["member_count"] for component in components),
        "component_assignment_count": len(assigned),
        "unique_component_assignment_count": len(set(assigned)),
        "class_component_counts": dict(sorted(class_component_counts.items())),
        "accepted_identity_or_visual_edge_count": len(accepted_edges),
        "pending_review_edge_count": len(review_edges),
        "pending_cross_label_edge_count": pending_cross_label,
        "pending_same_label_edge_count": pending_same_label,
        "confirmed_cross_label_conflict_count": 0,
        "automatic_quarantine_performed": False,
        "automatic_deletion_performed": False,
        "split_manifest_created": False,
        "test_loader_constructed": False,
        "test_evaluated": False,
        "product_provenance_coverage": product_coverage / original_count,
        "cross_label_component_count": len(cross_label_components),
        "embedding": dict(embedding_metadata),
    }


def write_json(value: Any, output_path: Path) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    root = get_practice_2_2_root().resolve()
    if root not in output_path.parents:
        raise RuntimeError("R3 output must remain inside Practice 2.2")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    return output_path
