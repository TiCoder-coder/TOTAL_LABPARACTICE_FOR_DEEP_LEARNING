from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image

from practice_2_2.r2_provenance import (
    build_provenance_record,
    evaluate_product_candidate,
    load_r2_policy,
    select_packshot_candidates,
    write_immutable_asset,
)


SEARCH_URL = "https://tiki.vn/api/v2/products"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://tiki.vn/",
}


def _get_json(url: str, params: dict, retries: int, delay: float) -> dict:
    for attempt in range(retries):
        response = requests.get(url, params=params, headers=HEADERS, timeout=20)
        if response.status_code == 200:
            return response.json()
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(delay * (attempt + 1))
            continue
        raise RuntimeError(f"Tiki request failed with status {response.status_code}")
    raise RuntimeError("Tiki request retry budget exhausted")


def _download_image(url: str, minimum_width: int, minimum_height: int) -> bytes:
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()
    content = response.content
    with Image.open(io.BytesIO(content)) as image:
        image.verify()
    with Image.open(io.BytesIO(content)) as image:
        if image.width < minimum_width or image.height < minimum_height:
            raise RuntimeError("R2 image is below minimum dimensions")
    return content


def _extension(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp"} else ".jpg"


def _listing_url(product: dict) -> str:
    if product.get("url_path"):
        return f"https://tiki.vn/{str(product['url_path']).lstrip('/')}"
    if product.get("url_key"):
        return f"https://tiki.vn/{product['url_key']}-p{product['id']}.html"
    return f"https://tiki.vn/p{product['id']}.html"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--category-id", required=True)
    parser.add_argument("--category-name", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--max-products", type=int, default=40)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    policy = load_r2_policy()
    if args.class_name not in policy["classes"]:
        raise RuntimeError("R2 collection class is invalid")
    rules = policy["classes"][args.class_name]
    accepted_products = 0
    rejected_products = 0
    saved_assets = 0
    seen_products = set()
    for page in range(1, args.pages + 1):
        payload = _get_json(
            SEARCH_URL,
            {
                "q": rules["query"],
                "category": args.category_id,
                "page": page,
                "limit": 40,
                "sort": "popular",
            },
            args.retries,
            args.delay,
        )
        for product in payload.get("data", []) or []:
            product_id = product.get("id")
            if not product_id or product_id in seen_products:
                continue
            seen_products.add(product_id)
            if len(seen_products) > args.max_products:
                break
            category_name = str(product.get("category_name") or args.category_name)
            decision = evaluate_product_candidate(
                args.class_name,
                str(product.get("name") or ""),
                category_name,
                policy,
            )
            if not decision["accepted"]:
                rejected_products += 1
                continue
            accepted_products += 1
            for candidate in select_packshot_candidates(product, policy):
                try:
                    content = _download_image(
                        candidate["image_url"],
                        policy["quality_thresholds"]["minimum_width"],
                        policy["quality_thresholds"]["minimum_height"],
                    )
                    raw_sha256 = hashlib.sha256(content).hexdigest()
                    record = build_provenance_record(
                        class_name=args.class_name,
                        product_id=product_id,
                        listing_url=_listing_url(product),
                        image_url=candidate["image_url"],
                        query=rules["query"],
                        category_id=args.category_id,
                        category_name=category_name,
                        image_index=candidate["image_index"],
                        image_role=candidate["image_role"],
                        crawl_timestamp_utc=datetime.now(timezone.utc).isoformat(),
                        raw_sha256=raw_sha256,
                        policy=policy,
                    )
                    write_immutable_asset(
                        content,
                        _extension(candidate["image_url"]),
                        record,
                        args.raw_root,
                        args.metadata_root,
                    )
                    saved_assets += 1
                except Exception:
                    continue
            time.sleep(args.delay)
        if len(seen_products) >= args.max_products:
            break
    print(
        json.dumps(
            {
                "class_name": args.class_name,
                "query": rules["query"],
                "category_id": args.category_id,
                "accepted_products": accepted_products,
                "rejected_products": rejected_products,
                "saved_assets": saved_assets,
                "detail_gallery_used": False,
                "automatic_deletion_performed": False,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
