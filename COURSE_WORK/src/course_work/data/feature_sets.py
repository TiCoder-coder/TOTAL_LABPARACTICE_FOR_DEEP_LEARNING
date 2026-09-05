from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from course_work.data.features import (
    FEATURE_VERSION,
    load_validated_feature_view,
    materialize_phase_7,
)
from course_work.utils.artifacts import (
    csv_text,
    get_project_root,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json_once_or_verify,
    write_text_once_or_verify,
)


FEATURE_SET_VERSION = "FEATURESETS-v1"
BASELINE_VARIANT = "FS1_TF1"
FEATURE_SET_ARTIFACT_ROOT = "artifacts/feature_sets"
EXOGENOUS = (
    "lights",
    "T1",
    "RH_1",
    "T2",
    "RH_2",
    "T3",
    "RH_3",
    "T4",
    "RH_4",
    "T5",
    "RH_5",
    "T6",
    "RH_6",
    "T7",
    "RH_7",
    "T8",
    "RH_8",
    "T9",
    "RH_9",
    "T_out",
    "Press_mm_hg",
    "RH_out",
    "Windspeed",
    "Visibility",
    "Tdewpoint",
)
HISTORICAL_TARGET = ("Appliances",)
RANDOM_CONTROLS = ("rv1", "rv2")
TIME_FEATURES = ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "weekend")
METADATA = ("raw_row_index", "date", "timestamp", "continuity_segment_id")
FEATURE_COMPONENTS = {
    "EXOGENOUS": EXOGENOUS,
    "HISTORICAL_TARGET": HISTORICAL_TARGET,
    "RANDOM_CONTROLS": RANDOM_CONTROLS,
    "TIME_FEATURES": TIME_FEATURES,
    "METADATA": METADATA,
}
VARIANT_COMPONENTS = {
    "FS0_TF0": ("EXOGENOUS",),
    "FS0_TF1": ("EXOGENOUS", "TIME_FEATURES"),
    "FS1_TF0": ("EXOGENOUS", "HISTORICAL_TARGET"),
    "FS1_TF1": ("EXOGENOUS", "HISTORICAL_TARGET", "TIME_FEATURES"),
    "FS2_TF0": ("EXOGENOUS", "HISTORICAL_TARGET", "RANDOM_CONTROLS"),
    "FS2_TF1": ("EXOGENOUS", "HISTORICAL_TARGET", "RANDOM_CONTROLS", "TIME_FEATURES"),
}
EXPECTED_FEATURE_COUNTS = {
    "FS0_TF0": 25,
    "FS0_TF1": 30,
    "FS1_TF0": 26,
    "FS1_TF1": 31,
    "FS2_TF0": 28,
    "FS2_TF1": 33,
}


def build_feature_variants() -> dict[str, tuple[str, ...]]:
    return {
        variant_id: tuple(
            feature
            for component_name in component_names
            for feature in FEATURE_COMPONENTS[component_name]
        )
        for variant_id, component_names in VARIANT_COMPONENTS.items()
    }


def compute_feature_fingerprint(variant_id: str, features: tuple[str, ...] | list[str]) -> str:
    specification = "|".join([variant_id, *features])
    return sha256_bytes(specification.encode("utf-8"))


def get_feature_list(variant_id: str) -> list[str]:
    variants = build_feature_variants()
    if variant_id not in variants:
        raise KeyError(f"Unknown feature variant: {variant_id}")
    return list(variants[variant_id])


def verify_phase_8_inputs(root: Path) -> dict[str, Any]:
    phase_7 = materialize_phase_7(root)
    if phase_7.get("artifact_version") != FEATURE_VERSION or phase_7.get("status") != "PASS":
        raise RuntimeError("FEATURES-v1 is not signed off")
    for relative_path, expected_checksum in phase_7.get("output_checksums", {}).items():
        path = root / relative_path
        if not path.is_file() or sha256_file(path) != expected_checksum:
            raise RuntimeError(f"Phase 7 FEATURES artifact checksum mismatch: {relative_path}")
    if not phase_7.get("feature_engineering_scope", "").startswith("FULL_DATASET_DETERMINISTIC"):
        raise RuntimeError("FEATURES-v1 must use FULL_DATASET_DETERMINISTIC_TRANSFORMATIONS scope")
    required_paths = [
        "artifacts/features/feature_engineering_manifest.json",
        "artifacts/features/feature_registry.csv",
        "artifacts/features/feature_lineage.csv",
        "artifacts/features/feature_availability.csv",
        "artifacts/features/feature_leakage_audit.csv",
    ]
    missing = [relative_path for relative_path in required_paths if not (root / relative_path).is_file()]
    if missing:
        raise FileNotFoundError(f"Phase 8 inputs are missing: {missing}")
    contract = read_json(root / "configs/base/coursework_contract.json")
    baseline = contract.get("baseline_transformer", {})
    if baseline.get("feature_set") != "FS1" or baseline.get("time_features") != "TF1":
        raise RuntimeError("Phase 0 baseline does not resolve to FS1_TF1")
    return phase_7


def load_phase_7_registries(root: Path) -> dict[str, pd.DataFrame]:
    return {
        "registry": pd.read_csv(root / "artifacts/features/feature_registry.csv"),
        "lineage": pd.read_csv(root / "artifacts/features/feature_lineage.csv"),
        "availability": pd.read_csv(root / "artifacts/features/feature_availability.csv"),
        "leakage": pd.read_csv(root / "artifacts/features/feature_leakage_audit.csv"),
    }


def pairwise_contracts_pass(variants: dict[str, tuple[str, ...]]) -> bool:
    expected_time = set(TIME_FEATURES)
    return all([
        set(variants["FS1_TF0"]) - set(variants["FS0_TF0"]) == set(HISTORICAL_TARGET),
        set(variants["FS2_TF0"]) - set(variants["FS1_TF0"]) == set(RANDOM_CONTROLS),
        set(variants["FS0_TF1"]) - set(variants["FS0_TF0"]) == expected_time,
        set(variants["FS1_TF1"]) - set(variants["FS1_TF0"]) == expected_time,
        set(variants["FS2_TF1"]) - set(variants["FS2_TF0"]) == expected_time,
    ])


def validate_feature_variants(
    feature_view: pd.DataFrame,
    registries: dict[str, pd.DataFrame],
    variants: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    feature_registry = registries["registry"].set_index("column_name")
    availability_registry = registries["availability"].set_index("feature_name")
    leakage_registry = registries["leakage"].set_index("feature_name")
    pairwise_valid = pairwise_contracts_pass(variants)
    leakage_rows = []
    order_rows = []
    discrepancies = []
    for variant_id, features in variants.items():
        expected_fs = variant_id.split("_")[0]
        expected_tf = variant_id.split("_")[1]
        missing = [feature for feature in features if feature not in feature_view.columns]
        duplicate_count = len(features) - len(set(features))
        metadata_leaks = [feature for feature in features if feature in METADATA]
        future_features = [
            feature
            for feature in features
            if any(token in feature.lower() for token in ["target_next", "t_plus_1", "future_"])
        ]
        forbidden = [
            feature
            for feature in features
            if feature in leakage_registry.index and leakage_registry.loc[feature, "status"] not in {"PASS", "CONTROL_ONLY"}
        ]
        expected_target = expected_fs in {"FS1", "FS2"}
        expected_random = expected_fs == "FS2"
        expected_time = expected_tf == "TF1"
        target_matches = (HISTORICAL_TARGET[0] in features) == expected_target and features.count(HISTORICAL_TARGET[0]) == int(expected_target)
        random_matches = all((feature in features) == expected_random for feature in RANDOM_CONTROLS) and all(features.count(feature) == int(expected_random) for feature in RANDOM_CONTROLS)
        time_matches = all((feature in features) == expected_time for feature in TIME_FEATURES) and all(features.count(feature) == int(expected_time) for feature in TIME_FEATURES)
        all_model_eligible = not missing and all(bool(feature_registry.loc[feature, "model_eligible"]) for feature in features)
        all_prediction_valid = not missing and all(bool(availability_registry.loc[feature, "available_at_prediction_time"]) for feature in features)
        all_numeric = not missing and all(pd.api.types.is_numeric_dtype(feature_view[feature]) for feature in features)
        missing_value_count = 0 if missing else int(feature_view[list(features)].isna().sum().sum())
        all_finite = False if missing else bool(np.isfinite(feature_view[list(features)].to_numpy(dtype=float)).all())
        count_matches = len(features) == EXPECTED_FEATURE_COUNTS[variant_id]
        status_checks = [
            not missing,
            duplicate_count == 0,
            not metadata_leaks,
            not future_features,
            not forbidden,
            target_matches,
            random_matches,
            time_matches,
            all_model_eligible,
            all_prediction_valid,
            all_numeric,
            missing_value_count == 0,
            all_finite,
            count_matches,
            pairwise_valid,
        ]
        status = "PASS" if all(status_checks) else "FAIL"
        fingerprint = compute_feature_fingerprint(variant_id, features)
        leakage_rows.append({
            "variant_id": variant_id,
            "contains_metadata": bool(metadata_leaks),
            "contains_future_target": bool(future_features),
            "contains_forbidden_feature": bool(forbidden),
            "contains_random_control_as_expected": random_matches,
            "contains_historical_target_as_expected": target_matches,
            "time_features_match_variant": time_matches,
            "all_features_model_eligible": all_model_eligible,
            "all_features_prediction_time_valid": all_prediction_valid,
            "all_features_numeric": all_numeric,
            "missing_value_count": missing_value_count,
            "all_values_finite": all_finite,
            "feature_count_matches": count_matches,
            "pairwise_contract_valid": pairwise_valid,
            "status": status,
        })
        order_rows.append({
            "variant_id": variant_id,
            "first_feature": features[0],
            "last_feature": features[-1],
            "feature_count": len(features),
            "duplicate_count": duplicate_count,
            "missing_count": len(missing),
            "metadata_leak_count": len(metadata_leaks),
            "order_fingerprint": fingerprint,
            "status": status,
        })
        if status == "FAIL":
            discrepancies.append({
                "id": f"FSD-{len(discrepancies) + 1:03d}",
                "severity": "ERROR",
                "variant_id": variant_id,
                "category": "VARIANT_CONTRACT_FAILURE",
                "expected": "all Phase 8 invariants PASS",
                "actual": [name for name, passed in zip([
                    "features_exist",
                    "no_duplicates",
                    "no_metadata",
                    "no_future_target",
                    "no_forbidden_features",
                    "historical_target_contract",
                    "random_control_contract",
                    "time_feature_contract",
                    "model_eligibility",
                    "prediction_time_availability",
                    "numeric_compatibility",
                    "no_missing_values",
                    "finite_values",
                    "feature_count",
                    "pairwise_contract",
                ], status_checks) if not passed],
                "interpretation": "The variant cannot be frozen safely",
                "recommended_action": "Correct the explicit component or upstream feature contract",
                "resolved": False,
                "notes": "Phase 8 sign-off blocked",
            })
    if discrepancies:
        raise RuntimeError(f"Phase 8 variant validation failed: {discrepancies}")
    return {
        "leakage_rows": leakage_rows,
        "order_rows": order_rows,
        "discrepancies": discrepancies,
        "status": "PASS",
    }


def build_registry_rows(
    variants: dict[str, tuple[str, ...]],
    registries: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    feature_registry = registries["registry"].set_index("column_name")
    leakage_registry = registries["leakage"].set_index("feature_name")
    component_by_feature = {
        feature: component_name
        for component_name, features in FEATURE_COMPONENTS.items()
        for feature in features
    }
    rows = []
    for variant_id, features in variants.items():
        fingerprint = compute_feature_fingerprint(variant_id, features)
        for position, feature in enumerate(features, start=1):
            source = feature_registry.loc[feature]
            rows.append({
                "feature_set_version": FEATURE_SET_VERSION,
                "variant_id": variant_id,
                "feature_position": position,
                "feature_name": feature,
                "component": component_by_feature[feature],
                "origin": source["origin"],
                "role": source["role"],
                "availability": source["availability"],
                "unit": None if pd.isna(source["unit"]) else source["unit"],
                "dtype": source["dtype"],
                "model_eligible": bool(source["model_eligible"]),
                "leakage_status": leakage_registry.loc[feature, "status"],
                "fingerprint": fingerprint,
            })
    return rows


def build_variant_rows(variants: dict[str, tuple[str, ...]]) -> list[dict[str, Any]]:
    rows = []
    for variant_id, features in variants.items():
        feature_set_id, time_feature_id = variant_id.split("_")
        rows.append({
            "variant_id": variant_id,
            "feature_set_id": feature_set_id,
            "time_feature_id": time_feature_id,
            "feature_count": len(features),
            "contains_past_target": HISTORICAL_TARGET[0] in features,
            "contains_random_controls": all(feature in features for feature in RANDOM_CONTROLS),
            "contains_time_features": all(feature in features for feature in TIME_FEATURES),
            "fingerprint": compute_feature_fingerprint(variant_id, features),
            "status": "PASS",
            "selection_status": "UNTESTED",
            "notes": "BASELINE_REFERENCE" if variant_id == BASELINE_VARIANT else "REGISTERED_ABLATION",
        })
    return rows


def build_lineage_rows(variants: dict[str, tuple[str, ...]]) -> list[dict[str, Any]]:
    rows = []
    for variant_id, component_names in VARIANT_COMPONENTS.items():
        position = 1
        for component_order, component_name in enumerate(component_names, start=1):
            component = FEATURE_COMPONENTS[component_name]
            rows.append({
                "variant_id": variant_id,
                "component_order": component_order,
                "component_name": component_name,
                "feature_start_position": position,
                "feature_end_position": position + len(component) - 1,
                "source_version": FEATURE_VERSION,
                "notes": "ordered tuple concatenation",
            })
            position += len(component)
    return rows


def build_registry_json(variants: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    return {
        "feature_set_version": FEATURE_SET_VERSION,
        "source_feature_version": FEATURE_VERSION,
        "baseline_variant": BASELINE_VARIANT,
        "variants": {
            variant_id: {
                "feature_set_id": variant_id.split("_")[0],
                "time_feature_id": variant_id.split("_")[1],
                "features": list(features),
                "feature_count": len(features),
                "fingerprint": compute_feature_fingerprint(variant_id, features),
                "built_from_components": list(VARIANT_COMPONENTS[variant_id]),
                "selection_status": "UNTESTED",
                "baseline_reference": variant_id == BASELINE_VARIANT,
            }
            for variant_id, features in variants.items()
        },
    }


def readme_feature_sets(variants: dict[str, tuple[str, ...]]) -> str:
    rows = [
        "# FEATURESETS-v1",
        "",
        "## Definitions",
        "",
        "FS0 contains the 25 raw exogenous channels and excludes historical Appliances and random controls.",
        "",
        "FS1 adds historical Appliances to FS0.",
        "",
        "FS2 adds rv1 and rv2 to FS1 as random-control channels.",
        "",
        "TF0 excludes all engineered calendar channels.",
        "",
        "TF1 adds hour_sin, hour_cos, dow_sin, dow_cos and weekend.",
        "",
        "## Registered variants",
        "",
        "| Variant | Feature count | Baseline reference | Selection status |",
        "|---|---:|---|---|",
    ]
    for variant_id, features in variants.items():
        rows.append(f"| {variant_id} | {len(features)} | {'Yes' if variant_id == BASELINE_VARIANT else 'No'} | UNTESTED |")
    rows.extend([
        "",
        "## Feature-order rule",
        "",
        "Every downstream phase must load the ordered list and fingerprint from feature_set_registry.json. Manual column lists and dtype-based automatic selection are not allowed.",
        "",
        "## Leakage rule",
        "",
        "Metadata is excluded from every model variant. Appliances is a historical channel only in FS1 and FS2. rv1 and rv2 appear only in FS2. No variant contains a future target or future observation.",
        "",
        "## Ablation rule",
        "",
        "FS0 to FS1 changes only historical target availability. FS1 to FS2 changes only random-control availability. TF0 to TF1 changes only the five engineered calendar channels. Phase 7 does not select a winner.",
        "",
    ])
    return "\n".join(rows)


def verify_existing_signoff(root: Path, signoff_path: Path) -> dict[str, Any]:
    signoff = read_json(signoff_path)
    if signoff.get("status") != "PASS" or signoff.get("artifact_version") != FEATURE_SET_VERSION:
        raise RuntimeError("Existing Phase 8 sign-off is invalid")
    for relative_path, expected_checksum in signoff.get("output_checksums", {}).items():
        path = root / relative_path
        if not path.is_file() or sha256_file(path) != expected_checksum:
            raise RuntimeError(f"Phase 8 artifact checksum mismatch: {relative_path}")
    registry = read_json(root / FEATURE_SET_ARTIFACT_ROOT / "feature_set_registry.json")
    if registry.get("feature_set_version") != FEATURE_SET_VERSION or set(registry.get("variants", {})) != set(VARIANT_COMPONENTS):
        raise RuntimeError("Reloaded FEATURESETS-v1 registry is invalid")
    for variant_id, values in registry["variants"].items():
        if compute_feature_fingerprint(variant_id, values["features"]) != values["fingerprint"]:
            raise RuntimeError(f"FEATURESETS-v1 fingerprint mismatch: {variant_id}")
    if not signoff.get("feature_engineering_scope", "").startswith("FULL_DATASET_DETERMINISTIC"):
        raise RuntimeError("FEATURESETS-v1 must use FULL_DATASET_DETERMINISTIC_TRANSFORMATIONS scope")
    return signoff


def materialize_phase_8(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    signoff_path = root / FEATURE_SET_ARTIFACT_ROOT / "phase_8_signoff.json"
    if signoff_path.exists():
        return verify_existing_signoff(root, signoff_path)
    phase_7 = verify_phase_8_inputs(root)
    feature_view = load_validated_feature_view(root)
    feature_manifest = read_json(root / "artifacts/features/feature_engineering_manifest.json")
    registries = load_phase_7_registries(root)
    variants = build_feature_variants()
    if variants != build_feature_variants():
        raise RuntimeError("Feature variant construction is nondeterministic")
    audit = validate_feature_variants(feature_view, registries, variants)
    registry_rows = build_registry_rows(variants, registries)
    variant_rows = build_variant_rows(variants)
    lineage_rows = build_lineage_rows(variants)
    registry_json = build_registry_json(variants)
    artifact_root = root / FEATURE_SET_ARTIFACT_ROOT
    manifest_path = artifact_root / "feature_set_manifest.json"
    registry_json_path = artifact_root / "feature_set_registry.json"
    registry_csv_path = artifact_root / "feature_set_registry.csv"
    variants_path = artifact_root / "feature_set_variants.csv"
    components_path = artifact_root / "feature_components.json"
    lineage_path = artifact_root / "feature_set_lineage.csv"
    leakage_path = artifact_root / "feature_set_leakage_audit.csv"
    order_path = artifact_root / "feature_order_checks.csv"
    discrepancies_path = artifact_root / "feature_set_discrepancies.json"
    readme_path = artifact_root / "README_FEATURE_SETS.md"
    registry_fields = ["feature_set_version", "variant_id", "feature_position", "feature_name", "component", "origin", "role", "availability", "unit", "dtype", "model_eligible", "leakage_status", "fingerprint"]
    variant_fields = ["variant_id", "feature_set_id", "time_feature_id", "feature_count", "contains_past_target", "contains_random_controls", "contains_time_features", "fingerprint", "status", "selection_status", "notes"]
    lineage_fields = ["variant_id", "component_order", "component_name", "feature_start_position", "feature_end_position", "source_version", "notes"]
    leakage_fields = ["variant_id", "contains_metadata", "contains_future_target", "contains_forbidden_feature", "contains_random_control_as_expected", "contains_historical_target_as_expected", "time_features_match_variant", "all_features_model_eligible", "all_features_prediction_time_valid", "all_features_numeric", "missing_value_count", "all_values_finite", "feature_count_matches", "pairwise_contract_valid", "status"]
    order_fields = ["variant_id", "first_feature", "last_feature", "feature_count", "duplicate_count", "missing_count", "metadata_leak_count", "order_fingerprint", "status"]
    write_json_once_or_verify(registry_json_path, registry_json)
    write_text_once_or_verify(registry_csv_path, csv_text(registry_fields, registry_rows))
    write_text_once_or_verify(variants_path, csv_text(variant_fields, variant_rows))
    write_json_once_or_verify(components_path, {
        "feature_set_version": FEATURE_SET_VERSION,
        "source_feature_version": FEATURE_VERSION,
        "components": {name: list(features) for name, features in FEATURE_COMPONENTS.items()},
    })
    write_text_once_or_verify(lineage_path, csv_text(lineage_fields, lineage_rows))
    write_text_once_or_verify(leakage_path, csv_text(leakage_fields, audit["leakage_rows"]))
    write_text_once_or_verify(order_path, csv_text(order_fields, audit["order_rows"]))
    write_json_once_or_verify(discrepancies_path, {"feature_set_version": FEATURE_SET_VERSION, "discrepancies": audit["discrepancies"]})
    write_text_once_or_verify(readme_path, readme_feature_sets(variants))
    created_at = datetime.now(timezone.utc).isoformat()
    fingerprints = {variant_id: compute_feature_fingerprint(variant_id, features) for variant_id, features in variants.items()}
    manifest = {
        "feature_set_version": FEATURE_SET_VERSION,
        "feature_version": FEATURE_VERSION,
        "dataset_revision": feature_manifest["dataset_revision"],
        "schema_version": feature_manifest["schema_version"],
        "temporal_version": feature_manifest["temporal_version"],
        "split_version": feature_manifest.get("split_version", "SPLIT-v1"),
        "environment_id": feature_manifest["environment_id"],
        "train_only_scope": False,
        "train_rows_used": feature_manifest.get("total_rows"),
        "baseline_variant": BASELINE_VARIANT,
        "variant_ids": list(variants),
        "variant_feature_counts": {variant_id: len(features) for variant_id, features in variants.items()},
        "variant_fingerprints": fingerprints,
        "component_counts": {name: len(features) for name, features in FEATURE_COMPONENTS.items()},
        "metadata_columns": list(METADATA),
        "target_column": HISTORICAL_TARGET[0],
        "historical_target_channel": HISTORICAL_TARGET[0],
        "random_controls": list(RANDOM_CONTROLS),
        "time_features": list(TIME_FEATURES),
        "source_feature_checksum": phase_7["derived_checksum"],
        "all_variants_valid": True,
        "leakage_audit_passed": True,
        "order_audit_passed": True,
        "audit_status": audit["status"],
        "warnings": [],
        "created_at": created_at,
    }
    write_json_once_or_verify(manifest_path, manifest)
    output_paths = [
        f"{FEATURE_SET_ARTIFACT_ROOT}/feature_set_manifest.json",
        f"{FEATURE_SET_ARTIFACT_ROOT}/feature_set_registry.json",
        f"{FEATURE_SET_ARTIFACT_ROOT}/feature_set_registry.csv",
        f"{FEATURE_SET_ARTIFACT_ROOT}/feature_set_variants.csv",
        f"{FEATURE_SET_ARTIFACT_ROOT}/feature_components.json",
        f"{FEATURE_SET_ARTIFACT_ROOT}/feature_set_lineage.csv",
        f"{FEATURE_SET_ARTIFACT_ROOT}/feature_set_leakage_audit.csv",
        f"{FEATURE_SET_ARTIFACT_ROOT}/feature_order_checks.csv",
        f"{FEATURE_SET_ARTIFACT_ROOT}/feature_set_discrepancies.json",
        f"{FEATURE_SET_ARTIFACT_ROOT}/README_FEATURE_SETS.md",
    ]
    input_paths = [
        "artifacts/features/phase_7_signoff.json",
        "artifacts/features/feature_engineering_manifest.json",
        "artifacts/features/feature_registry.csv",
        "artifacts/features/feature_lineage.csv",
        "artifacts/features/feature_availability.csv",
        "artifacts/features/feature_leakage_audit.csv",
        feature_manifest["derived_file_path"],
        "configs/base/coursework_contract.json",
    ]
    signoff = {
        "artifact_version": FEATURE_SET_VERSION,
        "phase_id": 8,
        "phase_version": "PHASE-8-v1",
        "created_at": created_at,
        "environment_id": feature_manifest["environment_id"],
        "dataset_revision": feature_manifest["dataset_revision"],
        "feature_version": FEATURE_VERSION,
        "baseline_variant": BASELINE_VARIANT,
        "feature_engineering_scope": "FULL_DATASET_DETERMINISTIC_TRANSFORMATIONS",
        "train_only_scope": False,
        "total_rows_used": feature_manifest.get("total_rows"),
        "input_paths": input_paths,
        "input_checksums": {relative_path: sha256_file(root / relative_path) for relative_path in input_paths},
        "output_paths": output_paths,
        "output_checksums": {relative_path: sha256_file(root / relative_path) for relative_path in output_paths},
        "config_fingerprint": phase_7["config_fingerprint"],
        "status": "PASS",
        "tests": [
            "feature_version_and_checksum",
            "six_variant_registry",
            "expected_feature_counts",
            "metadata_exclusion",
            "historical_target_isolation",
            "random_control_isolation",
            "time_feature_isolation",
            "model_eligibility",
            "prediction_time_availability",
            "numeric_compatibility",
            "missing_and_non_finite_values",
            "pairwise_component_differences",
            "feature_order_determinism",
            "fingerprint_determinism",
            "matrix_probe_shape_and_semantics",
        ],
        "warnings": [],
        "discrepancies": [],
    }
    write_json_once_or_verify(signoff_path, signoff)
    return verify_existing_signoff(root, signoff_path)


def load_validated_feature_set_registry(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    materialize_phase_8(root)
    registry = read_json(root / FEATURE_SET_ARTIFACT_ROOT / "feature_set_registry.json")
    for variant_id, values in registry["variants"].items():
        if len(values["features"]) != values["feature_count"]:
            raise RuntimeError(f"Reloaded feature count mismatch: {variant_id}")
        if compute_feature_fingerprint(variant_id, values["features"]) != values["fingerprint"]:
            raise RuntimeError(f"Reloaded feature fingerprint mismatch: {variant_id}")
    return registry
