from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from course_workutils.artifacts import get_project_root, sha256_file


PREDICTIONS_REL_DIR = Path("artifacts/final_test/predictions")
PREDICTION_CHECKSUMS_REL = Path("artifacts/final_test/prediction_checksums.json")
POPULATION_MANIFEST_REL = Path("artifacts/final_test/final_test_population_manifest.json")
PHASE47_SIGNOFF_REL = Path("artifacts/final_test/phase_47_signoff.json")


SEED_TO_BUNDLE: dict[int, str] = {
    42: "final_test_predictions_seed42.csv",
    123: "final_test_predictions_seed123.csv",
    2026: "final_test_predictions_seed2026.csv",
}


def _resolve(project_root: Path) -> Path:
    return project_root / PREDICTIONS_REL_DIR


def load_prediction_checksums(project_root: Path | None = None) -> dict[str, Any]:
    root = project_root if project_root is not None else get_project_root()
    return json.loads((root / PREDICTION_CHECKSUMS_REL).read_text(encoding="utf-8"))


def load_population_manifest(project_root: Path | None = None) -> dict[str, Any]:
    root = project_root if project_root is not None else get_project_root()
    return json.loads((root / POPULATION_MANIFEST_REL).read_text(encoding="utf-8"))


def load_phase47_signoff(project_root: Path | None = None) -> dict[str, Any]:
    root = project_root if project_root is not None else get_project_root()
    return json.loads((root / PHASE47_SIGNOFF_REL).read_text(encoding="utf-8"))


PHASE48_SIGNOFF_REL = "artifacts/prediction_analysis/phase_48_signoff.json"


def load_phase48_signoff(project_root: Path | None = None) -> dict[str, Any]:
    root = project_root if project_root is not None else get_project_root()
    return json.loads((root / PHASE48_SIGNOFF_REL).read_text(encoding="utf-8"))


def expected_seed_bundle_sha256(checksums: dict[str, Any], seed: int) -> str:
    bundle_name = SEED_TO_BUNDLE[seed]
    bundle_key = _bundle_to_checksum_key(bundle_name)
    entry = checksums["predictions"][bundle_key]
    return entry["sha256"]


def _bundle_to_checksum_key(bundle_name: str) -> str:
    if bundle_name == "final_test_predictions_seed42.csv":
        return "seed_42"
    if bundle_name == "final_test_predictions_seed123.csv":
        return "seed_123"
    if bundle_name == "final_test_predictions_seed2026.csv":
        return "seed_2026"
    raise KeyError(bundle_name)


def load_seed_bundle_rows(
    seed: int,
    project_root: Path | None = None,
) -> list[dict[str, str]]:
    root = project_root if project_root is not None else get_project_root()
    bundle_path = _resolve(root) / SEED_TO_BUNDLE[seed]
    with bundle_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader)


def compute_seed_bundle_sha256(
    seed: int,
    project_root: Path | None = None,
) -> str:
    root = project_root if project_root is not None else get_project_root()
    bundle_path = _resolve(root) / SEED_TO_BUNDLE[seed]
    return sha256_file(bundle_path)


class SourceVerificationFailure(Exception):
    pass


def verify_source_bundles(
    project_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    root = project_root if project_root is not None else get_project_root()
    checksums = load_prediction_checksums(root)
    population = load_population_manifest(root)
    expected_population_sha = population["target_ids_sha256"]

    out: dict[str, dict[str, Any]] = {}

    target_id_sets: dict[int, set[str]] = {}
    target_ts_orders: dict[int, list[str]] = {}
    y_true_vectors: dict[int, list[float]] = {}

    for seed in (42, 123, 2026):
        bundle_name = SEED_TO_BUNDLE[seed]
        bundle_path = _resolve(root) / bundle_name
        if not bundle_path.exists():
            raise SourceVerificationFailure(
                f"Phase 49 source bundle missing: {bundle_path}"
            )

        observed_sha = sha256_file(bundle_path)
        expected_sha = expected_seed_bundle_sha256(checksums, seed)
        sha_ok = observed_sha == expected_sha

        rows = load_seed_bundle_rows(seed, root)
        n_rows = len(rows)

        target_ids = [r["target_id"] for r in rows]
        n_unique = len(set(target_ids))
        unique_ok = n_unique == n_rows

        n_ok = n_rows == 2961

        y_true_values = [float(r["y_true_wh"]) for r in rows]
        y_pred_values = [float(r["y_pred_wh"]) for r in rows]
        all_finite = all(
            (y == y and y not in (float("inf"), float("-inf")))
            for y in y_true_values + y_pred_values
        )

        target_ts = [r["target_timestamp"] for r in rows]

        target_id_sets[seed] = set(target_ids)
        target_ts_orders[seed] = target_ts
        y_true_vectors[seed] = y_true_values

        out[str(seed)] = {
            "bundle_path": str(bundle_path),
            "bundle_name": bundle_name,
            "observed_sha256": observed_sha,
            "expected_sha256": expected_sha,
            "sha256_match": sha_ok,
            "n_rows": n_rows,
            "n_rows_expected": 2961,
            "n_rows_ok": n_ok,
            "n_unique_target_id": n_unique,
            "target_id_unique_ok": unique_ok,
            "all_y_finite": all_finite,
            "first_target_id": target_ids[0],
            "last_target_id": target_ids[-1],
            "first_timestamp": target_ts[0],
            "last_timestamp": target_ts[-1],
        }

    cross_seed_alignment = {
        "target_id_sets_identical": all(
            target_id_sets[42] == s for s in target_id_sets.values()
        ),
        "target_timestamp_order_identical": all(
            target_ts_orders[42] == target_ts_orders[s]
            for s in (123, 2026)
        ),
        "y_true_vector_identical": all(
            y_true_vectors[42] == y_true_vectors[s] for s in (123, 2026)
        ),
        "test_population_sha256_expected": expected_population_sha,
    }
    cross_seed_alignment["cross_seed_alignment_pass"] = (
        cross_seed_alignment["target_id_sets_identical"]
        and cross_seed_alignment["target_timestamp_order_identical"]
        and cross_seed_alignment["y_true_vector_identical"]
    )

    out["__cross_seed__"] = cross_seed_alignment

    overall_pass = all(
        v.get("sha256_match")
        and v.get("n_rows_ok")
        and v.get("target_id_unique_ok")
        and v.get("all_y_finite")
        for v in out.values()
        if isinstance(v, dict) and "sha256_match" in v
    ) and cross_seed_alignment["cross_seed_alignment_pass"]

    out["__overall_pass__"] = overall_pass

    if not overall_pass:
        raise SourceVerificationFailure(
            "Phase49-B source verification failed; see verification payload"
        )

    return out
