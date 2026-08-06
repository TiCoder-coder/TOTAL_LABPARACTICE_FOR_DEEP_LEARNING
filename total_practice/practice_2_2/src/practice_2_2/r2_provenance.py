from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .paths import get_practice_2_2_root
from .resources import file_sha256


POLICY_RELATIVE_PATH = Path("configs/r2_provenance_policy.json")


def load_r2_policy(policy_path: Path | None = None) -> dict[str, Any]:
    root = get_practice_2_2_root()
    path = Path(policy_path or root / POLICY_RELATIVE_PATH).expanduser().resolve()
    policy = json.loads(path.read_text())
    if policy.get("schema_version") != 1:
        raise RuntimeError("Unsupported R2 provenance policy schema")
    if policy.get("raw_assets_immutable") is not True:
        raise RuntimeError("R2 raw assets must be immutable")
    if policy.get("automatic_deletion_allowed") is not False:
        raise RuntimeError("R2 must prohibit automatic deletion")
    if policy.get("detail_gallery_allowed") is not False:
        raise RuntimeError("R2 must reject detail-gallery collection")
    if set(policy.get("classes", {})) != {
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
    }:
        raise RuntimeError("R2 policy must define exactly ten classes")
    return policy


def evaluate_product_candidate(
    class_name: str,
    product_name: str,
    category_name: str,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    policy = dict(policy or load_r2_policy())
    if class_name not in policy["classes"]:
        raise RuntimeError(f"Unknown R2 class: {class_name}")
    rules = policy["classes"][class_name]
    text = f"{product_name} {category_name}".casefold()
    positive = sorted(term for term in rules["required_terms"] if term.casefold() in text)
    negative = sorted(term for term in rules["negative_terms"] if term.casefold() in text)
    return {
        "accepted": bool(positive) and not negative and bool(category_name.strip()),
        "positive_terms": positive,
        "negative_terms": negative,
        "category_identity_present": bool(category_name.strip()),
    }


def _image_url(item: Any) -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, Mapping):
        value = item.get("base_url") or item.get("large_url") or item.get("url")
        return str(value) if value else None
    return None


def select_packshot_candidates(
    product: Mapping[str, Any],
    policy: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    policy = dict(policy or load_r2_policy())
    candidates = []
    seen = set()
    thumbnail = product.get("thumbnail_url")
    if thumbnail:
        seen.add(str(thumbnail))
        candidates.append(
            {
                "image_url": str(thumbnail),
                "image_index": 0,
                "image_role": "primary_thumbnail",
            }
        )
    for item in product.get("images", []) or []:
        url = _image_url(item)
        if not url or url in seen:
            continue
        seen.add(url)
        candidates.append(
            {
                "image_url": url,
                "image_index": len(candidates),
                "image_role": "primary_gallery",
            }
        )
        if len(candidates) >= policy["maximum_images_per_product"]:
            break
    return candidates[: policy["maximum_images_per_product"]]


def build_provenance_record(
    *,
    class_name: str,
    product_id: str | int,
    listing_url: str,
    image_url: str,
    query: str,
    category_id: str | int,
    category_name: str,
    image_index: int,
    image_role: str,
    crawl_timestamp_utc: str,
    raw_sha256: str,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    policy = dict(policy or load_r2_policy())
    asset_id = hashlib.sha256(
        f"{policy['source_provider']}:{product_id}:{image_url}".encode()
    ).hexdigest()
    record = {
        "asset_id": asset_id,
        "source_provider": policy["source_provider"],
        "product_id": str(product_id),
        "listing_url": listing_url,
        "image_url": image_url,
        "query": query,
        "category_id": str(category_id),
        "category_name": category_name,
        "image_index": int(image_index),
        "image_role": image_role,
        "crawl_timestamp_utc": crawl_timestamp_utc,
        "raw_sha256": raw_sha256,
        "class_name": class_name,
    }
    missing = [
        field
        for field in policy["required_provenance_fields"]
        if field not in record or record[field] in {None, ""}
    ]
    if missing:
        raise RuntimeError(f"R2 provenance fields are missing: {missing}")
    if image_role not in policy["allowed_image_roles"]:
        raise RuntimeError("R2 image role is not allowed")
    if class_name not in policy["classes"]:
        raise RuntimeError("R2 provenance class is invalid")
    if len(raw_sha256) != 64:
        raise RuntimeError("R2 raw SHA-256 is invalid")
    datetime.fromisoformat(crawl_timestamp_utc.replace("Z", "+00:00"))
    return record


def write_immutable_asset(
    content: bytes,
    extension: str,
    record: Mapping[str, Any],
    raw_root: Path,
    metadata_root: Path,
) -> dict[str, Path]:
    raw_sha256 = hashlib.sha256(content).hexdigest()
    if raw_sha256 != record["raw_sha256"]:
        raise RuntimeError("R2 raw content hash does not match provenance")
    extension = extension.lower()
    if extension not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise RuntimeError("R2 raw image extension is invalid")
    raw_path = (
        Path(raw_root)
        / str(record["class_name"])
        / str(record["product_id"])
        / f"{record['asset_id']}{extension}"
    )
    metadata_path = Path(metadata_root) / f"{record['asset_id']}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists() and file_sha256(raw_path) != raw_sha256:
        raise RuntimeError("R2 immutable raw asset conflict")
    if not raw_path.exists():
        raw_path.write_bytes(content)
    payload = json.dumps(dict(record), sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    if metadata_path.exists() and metadata_path.read_text() != payload:
        raise RuntimeError("R2 immutable provenance sidecar conflict")
    if not metadata_path.exists():
        metadata_path.write_text(payload)
    return {"raw_path": raw_path, "metadata_path": metadata_path}
