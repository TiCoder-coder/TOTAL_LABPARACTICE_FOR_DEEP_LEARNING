import csv
import hashlib
import os
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile

from course_work.utils.artifacts import (
    atomic_write_bytes,
    canonical_json_bytes,
    csv_text,
    get_project_root,
    read_json,
    sha256_file,
    write_json_once_or_verify,
    write_text_once_or_verify,
)
from course_work.utils.environment import materialize_phase_1


DATASET_METADATA = {
    "dataset_revision": "DATA-v1",
    "dataset_name": "Appliances Energy Prediction",
    "uci_id": 374,
    "doi": "10.24432/C5VC8G",
    "creator": "Luis Candanedo",
    "license": "CC BY 4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "task": "Regression",
    "characteristics": ["Multivariate", "Time-Series"],
    "feature_type": "Real",
    "reported_num_instances": 19735,
    "reported_num_features": 28,
    "reported_sampling_interval_minutes": 10,
    "reported_duration": "approximately 4.5 months",
    "target": "Appliances",
    "source_provider": "UCI Machine Learning Repository",
    "source_type": "official_repository",
    "dataset_page": "https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction",
    "source_url": "https://archive.ics.uci.edu/static/public/374/appliances%2Benergy%2Bprediction.zip",
    "download_method": "AQ0_direct_uci",
    "original_download_filename": "appliances+energy+prediction.zip",
    "expected_csv": "energydata_complete.csv",
}

VARIABLE_METADATA = [
    {"name": "date", "role": "Feature", "data_type": "Date", "description": "Timestamp in year-month-day hour:minute:second format", "units": "", "missing_values": "no"},
    {"name": "Appliances", "role": "Target", "data_type": "Integer", "description": "Appliances energy use", "units": "Wh", "missing_values": "no"},
    {"name": "lights", "role": "Feature", "data_type": "Integer", "description": "Light fixtures energy use", "units": "Wh", "missing_values": "no"},
    {"name": "T1", "role": "Feature", "data_type": "Continuous", "description": "Temperature in kitchen area", "units": "C", "missing_values": "no"},
    {"name": "RH_1", "role": "Feature", "data_type": "Continuous", "description": "Humidity in kitchen area", "units": "%", "missing_values": "no"},
    {"name": "T2", "role": "Feature", "data_type": "Continuous", "description": "Temperature in living room area", "units": "C", "missing_values": "no"},
    {"name": "RH_2", "role": "Feature", "data_type": "Continuous", "description": "Humidity in living room area", "units": "%", "missing_values": "no"},
    {"name": "T3", "role": "Feature", "data_type": "Continuous", "description": "Temperature in laundry room area", "units": "C", "missing_values": "no"},
    {"name": "RH_3", "role": "Feature", "data_type": "Continuous", "description": "Humidity in laundry room area", "units": "%", "missing_values": "no"},
    {"name": "T4", "role": "Feature", "data_type": "Continuous", "description": "Temperature in office room", "units": "C", "missing_values": "no"},
    {"name": "RH_4", "role": "Feature", "data_type": "Continuous", "description": "Humidity in office room", "units": "%", "missing_values": "no"},
    {"name": "T5", "role": "Feature", "data_type": "Continuous", "description": "Temperature in bathroom", "units": "C", "missing_values": "no"},
    {"name": "RH_5", "role": "Feature", "data_type": "Continuous", "description": "Humidity in bathroom", "units": "%", "missing_values": "no"},
    {"name": "T6", "role": "Feature", "data_type": "Continuous", "description": "Temperature outside the building north side", "units": "C", "missing_values": "no"},
    {"name": "RH_6", "role": "Feature", "data_type": "Continuous", "description": "Humidity outside the building north side", "units": "%", "missing_values": "no"},
    {"name": "T7", "role": "Feature", "data_type": "Continuous", "description": "Temperature in ironing room", "units": "C", "missing_values": "no"},
    {"name": "RH_7", "role": "Feature", "data_type": "Continuous", "description": "Humidity in ironing room", "units": "%", "missing_values": "no"},
    {"name": "T8", "role": "Feature", "data_type": "Continuous", "description": "Temperature in teenager room 2", "units": "C", "missing_values": "no"},
    {"name": "RH_8", "role": "Feature", "data_type": "Continuous", "description": "Humidity in teenager room 2", "units": "%", "missing_values": "no"},
    {"name": "T9", "role": "Feature", "data_type": "Continuous", "description": "Temperature in parents room", "units": "C", "missing_values": "no"},
    {"name": "RH_9", "role": "Feature", "data_type": "Continuous", "description": "Humidity in parents room", "units": "%", "missing_values": "no"},
    {"name": "T_out", "role": "Feature", "data_type": "Continuous", "description": "Temperature outside from Chievres weather station", "units": "C", "missing_values": "no"},
    {"name": "Press_mm_hg", "role": "Feature", "data_type": "Continuous", "description": "Pressure from Chievres weather station", "units": "mm Hg", "missing_values": "no"},
    {"name": "RH_out", "role": "Feature", "data_type": "Continuous", "description": "Humidity outside from Chievres weather station", "units": "%", "missing_values": "no"},
    {"name": "Windspeed", "role": "Feature", "data_type": "Continuous", "description": "Wind speed from Chievres weather station", "units": "m/s", "missing_values": "no"},
    {"name": "Visibility", "role": "Feature", "data_type": "Continuous", "description": "Visibility from Chievres weather station", "units": "km", "missing_values": "no"},
    {"name": "Tdewpoint", "role": "Feature", "data_type": "Continuous", "description": "Dew point temperature from Chievres weather station", "units": "C", "missing_values": "no"},
    {"name": "rv1", "role": "Feature", "data_type": "Continuous", "description": "Random variable 1", "units": "nondimensional", "missing_values": "no"},
    {"name": "rv2", "role": "Feature", "data_type": "Continuous", "description": "Random variable 2", "units": "nondimensional", "missing_values": "no"},
]


def validate_archive(archive_path: Path) -> dict[str, Any]:
    if not archive_path.is_file() or archive_path.stat().st_size == 0:
        raise FileNotFoundError(f"Archive is missing or empty: {archive_path}")
    try:
        with ZipFile(archive_path) as archive:
            bad_member = archive.testzip()
            members = archive.infolist()
    except BadZipFile as error:
        raise ValueError(f"Invalid ZIP archive: {archive_path}") from error
    if bad_member is not None:
        raise ValueError(f"ZIP integrity failure at member: {bad_member}")
    unsafe = [
        info.filename
        for info in members
        if PurePosixPath(info.filename).is_absolute() or ".." in PurePosixPath(info.filename).parts
    ]
    if unsafe:
        raise ValueError(f"Unsafe ZIP members: {unsafe}")
    matching = [info for info in members if PurePosixPath(info.filename).name == DATASET_METADATA["expected_csv"]]
    if len(matching) != 1:
        raise ValueError("ZIP must contain exactly one expected CSV member")
    return {
        "archive_integrity_ok": True,
        "paths_safe": True,
        "member_count": len(members),
        "expected_member": matching[0].filename,
        "expected_member_size_bytes": matching[0].file_size,
    }


def extract_expected_member(archive_path: Path, destination: Path) -> Path:
    validation = validate_archive(archive_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive_path) as archive, archive.open(validation["expected_member"]) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    return destination


def smoke_test_csv(csv_path: Path, preview_rows: int = 5) -> dict[str, Any]:
    if preview_rows < 1:
        raise ValueError("preview_rows must be positive")
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        rows = [row for _, row in zip(range(preview_rows), reader)]
    if not header or len(rows) != preview_rows:
        raise ValueError("CSV header or preview rows are incomplete")
    if any(len(row) != len(header) for row in rows):
        raise ValueError("CSV preview row width differs from header width")
    return {
        "csv_smoke_test_ok": True,
        "header_present": True,
        "preview_rows": len(rows),
        "preview_columns": len(header),
    }


def verify_existing_signoff(root: Path, signoff_path: Path) -> dict[str, Any]:
    signoff = read_json(signoff_path)
    if signoff.get("status") != "PASS" or signoff.get("dataset_revision") != "DATA-v1":
        raise RuntimeError("Existing Phase 2 sign-off is not valid")
    for relative_path, expected_hash in signoff.get("output_checksums", {}).items():
        path = root / relative_path
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Phase 2 artifact checksum mismatch: {relative_path}")
    return signoff


def _phase_2_readme_bytes(acquired_utc: datetime, archive_hash: str, csv_hash: str) -> bytes:
    value = (
        "# Appliances Energy Prediction Source\n\n"
        "Dataset: Appliances Energy Prediction\n\n"
        "UCI ID: 374\n\n"
        "DOI: 10.24432/C5VC8G\n\n"
        f"Source: {DATASET_METADATA['dataset_page']}\n\n"
        "License: CC BY 4.0\n\n"
        f"Acquired at UTC: {acquired_utc.isoformat()}\n\n"
        "Acquisition method: AQ0 direct UCI archive\n\n"
        "Dataset revision: DATA-v1\n\n"
        f"Archive SHA-256: {archive_hash}\n\n"
        f"CSV SHA-256: {csv_hash}\n\n"
        "Canonical raw files must not be modified in place.\n\n"
        "Citation: Candanedo, L. (2017). Appliances Energy Prediction [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5VC8G.\n"
    )
    return value.encode("utf-8")


def _phase_2_manifest_candidate(
    acquired_utc: datetime,
    acquired_local: datetime,
    acquisition_log: dict[str, Any],
) -> dict[str, Any]:
    return {
        **DATASET_METADATA,
        "local_archive_path": acquisition_log["archive_path"],
        "local_csv_path": acquisition_log["csv_path"],
        "archive_size_bytes": acquisition_log["archive_size_bytes"],
        "csv_size_bytes": acquisition_log["csv_size_bytes"],
        "archive_sha256": acquisition_log["archive_sha256"],
        "csv_sha256": acquisition_log["csv_sha256"],
        "acquired_at_utc": acquired_utc.isoformat(),
        "acquired_at_local": acquired_local.isoformat(),
        "timezone": str(acquired_local.tzinfo),
        "environment_id": "ENV-v1",
        "archive_integrity_ok": acquisition_log["archive_integrity_ok"],
        "archive_paths_safe": acquisition_log["archive_paths_safe"],
        "csv_smoke_test_ok": acquisition_log["csv_smoke_test_ok"],
        "ucimlrepo_crosscheck_status": acquisition_log["ucimlrepo_crosscheck_status"],
        "raw_csv_reused_after_checksum_match": True,
    }


def resolve_signed_phase_2_metadata(project_root: Path | None = None) -> dict[str, bytes]:
    root = (project_root or get_project_root()).resolve()
    signoff = read_json(root / "artifacts/acquisition/phase_2_signoff.json")
    acquisition_log = read_json(root / "artifacts/acquisition/acquisition_log.json")
    acquired_utc = datetime.fromisoformat(acquisition_log["start_time"])
    completed_utc = datetime.fromisoformat(acquisition_log["end_time"])
    duration_microseconds = int((completed_utc - acquired_utc).total_seconds() * 1_000_000)
    if duration_microseconds < 0 or duration_microseconds > 5_000_000:
        raise RuntimeError("Phase 2 acquisition timestamp interval is invalid")
    readme_bytes = _phase_2_readme_bytes(
        acquired_utc,
        acquisition_log["archive_sha256"],
        acquisition_log["csv_sha256"],
    )
    expected_readme = signoff["output_checksums"]["data/raw_data/README_SOURCE.md"]
    if hashlib.sha256(readme_bytes).hexdigest() != expected_readme:
        raise RuntimeError("Phase 2 README cannot be reconstructed from signed provenance")
    local_timezone = timezone(timedelta(hours=7), name="+07")
    expected_manifest = signoff["output_checksums"]["data/raw_data/dataset_manifest.json"]
    manifest_bytes = None
    for offset in range(duration_microseconds + 1):
        acquired_local = (acquired_utc + timedelta(microseconds=offset)).astimezone(local_timezone)
        candidate_bytes = canonical_json_bytes(
            _phase_2_manifest_candidate(acquired_utc, acquired_local, acquisition_log)
        )
        if hashlib.sha256(candidate_bytes).hexdigest() == expected_manifest:
            manifest_bytes = candidate_bytes
            break
    if manifest_bytes is None:
        raise RuntimeError("Phase 2 manifest cannot be reconstructed from signed provenance")
    return {
        "data/raw_data/README_SOURCE.md": readme_bytes,
        "data/raw_data/dataset_manifest.json": manifest_bytes,
    }


def recover_phase_2_metadata_revision(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    signoff_path = root / "artifacts/acquisition/phase_2_signoff.json"
    signoff = read_json(signoff_path)
    recovered = resolve_signed_phase_2_metadata(root)
    protected_csv = root / "data/raw_data/energydata_complete.csv"
    protected_archive = root / "data/raw_data/source/appliances_energy_prediction.zip"
    expected_csv = signoff["input_checksums"]["data/raw_data/energydata_complete.csv"]
    expected_archive = signoff["output_checksums"]["data/raw_data/source/appliances_energy_prediction.zip"]
    if sha256_file(protected_csv) != expected_csv or sha256_file(protected_archive) != expected_archive:
        raise RuntimeError("Phase 2 protected data checksum is invalid")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    history_root = root / "artifacts/acquisition/_history" / stamp
    history_root.mkdir(parents=True, exist_ok=False)
    before = {}
    for relative_path, value in recovered.items():
        path = root / relative_path
        before[relative_path] = sha256_file(path)
        shutil.copy2(path, history_root / path.name)
        atomic_write_bytes(path, value)
    recovery_manifest = {
        "artifact_version": "PHASE2-METADATA-RECOVERY-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "before_checksums": before,
        "after_checksums": {
            relative_path: sha256_file(root / relative_path)
            for relative_path in recovered
        },
        "signed_output_checksums": {
            relative_path: signoff["output_checksums"][relative_path]
            for relative_path in recovered
        },
        "protected_csv_checksum": sha256_file(protected_csv),
        "protected_archive_checksum": sha256_file(protected_archive),
        "status": "PASS",
    }
    atomic_write_bytes(history_root / "recovery_manifest.json", canonical_json_bytes(recovery_manifest))
    verify_existing_signoff(root, signoff_path)
    return recovery_manifest


def materialize_phase_2(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    phase_1 = materialize_phase_1(root)
    if phase_1.get("status") != "PASS":
        raise RuntimeError("Phase 1 sign-off is not PASS")
    raw_root = root / "data/raw_data"
    source_root = raw_root / "source"
    protected_csv = raw_root / "energydata_complete.csv"
    partial_archive = source_root / "appliances_energy_prediction.zip.part"
    canonical_archive = source_root / "appliances_energy_prediction.zip"
    artifact_root = root / "artifacts/acquisition"
    signoff_path = artifact_root / "phase_2_signoff.json"
    if signoff_path.exists():
        if partial_archive.exists():
            raise RuntimeError("Unexpected partial archive exists after Phase 2 sign-off")
        return verify_existing_signoff(root, signoff_path)
    if canonical_archive.exists() and partial_archive.exists():
        raise RuntimeError("Both partial and canonical archives exist before Phase 2 sign-off")
    candidate_archive = partial_archive if partial_archive.exists() else canonical_archive
    validation = validate_archive(candidate_archive)
    protected_hash_before = sha256_file(protected_csv)
    if protected_hash_before != "2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d":
        raise RuntimeError("Protected raw CSV does not match the approved baseline")
    with tempfile.TemporaryDirectory(prefix="course_work_uci_374_") as directory:
        candidate_csv = Path(directory) / "energydata_complete.csv"
        extract_expected_member(candidate_archive, candidate_csv)
        candidate_hash = sha256_file(candidate_csv)
        if candidate_hash != protected_hash_before:
            raise RuntimeError("Official extracted CSV differs from the protected raw CSV")
        candidate_smoke = smoke_test_csv(candidate_csv)
    protected_smoke = smoke_test_csv(protected_csv)
    if candidate_archive == partial_archive:
        canonical_archive.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial_archive, canonical_archive)
    archive_hash = sha256_file(canonical_archive)
    if sha256_file(protected_csv) != protected_hash_before:
        raise RuntimeError("Protected raw CSV changed during acquisition")
    acquired_utc = datetime.now(timezone.utc)
    acquired_local = datetime.now().astimezone()
    relative_archive = "data/raw_data/source/appliances_energy_prediction.zip"
    relative_csv = "data/raw_data/energydata_complete.csv"
    manifest = {
        **DATASET_METADATA,
        "local_archive_path": relative_archive,
        "local_csv_path": relative_csv,
        "archive_size_bytes": canonical_archive.stat().st_size,
        "csv_size_bytes": protected_csv.stat().st_size,
        "archive_sha256": archive_hash,
        "csv_sha256": protected_hash_before,
        "acquired_at_utc": acquired_utc.isoformat(),
        "acquired_at_local": acquired_local.isoformat(),
        "timezone": str(acquired_local.tzinfo),
        "environment_id": "ENV-v1",
        "archive_integrity_ok": validation["archive_integrity_ok"],
        "archive_paths_safe": validation["paths_safe"],
        "csv_smoke_test_ok": candidate_smoke["csv_smoke_test_ok"] and protected_smoke["csv_smoke_test_ok"],
        "ucimlrepo_crosscheck_status": "NOT_REQUIRED_OFFICIAL_AQ0_VERIFIED",
        "raw_csv_reused_after_checksum_match": True,
    }
    source_metadata = {
        "dataset_name": DATASET_METADATA["dataset_name"],
        "uci_id": DATASET_METADATA["uci_id"],
        "doi": DATASET_METADATA["doi"],
        "dataset_page": DATASET_METADATA["dataset_page"],
        "download_url": DATASET_METADATA["source_url"],
        "creator": DATASET_METADATA["creator"],
        "license": DATASET_METADATA["license"],
        "license_url": DATASET_METADATA["license_url"],
        "citation": "Candanedo, L. (2017). Appliances Energy Prediction [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5VC8G.",
        "metadata_source": "official_uci_dataset_page",
    }
    checksums = f"{archive_hash}  source/appliances_energy_prediction.zip\n{protected_hash_before}  energydata_complete.csv\n"
    variable_fields = ["name", "role", "data_type", "description", "units", "missing_values"]
    variables = csv_text(variable_fields, VARIABLE_METADATA)
    readme = (
        "# Appliances Energy Prediction Source\n\n"
        "Dataset: Appliances Energy Prediction\n\n"
        "UCI ID: 374\n\n"
        "DOI: 10.24432/C5VC8G\n\n"
        f"Source: {DATASET_METADATA['dataset_page']}\n\n"
        "License: CC BY 4.0\n\n"
        f"Acquired at UTC: {acquired_utc.isoformat()}\n\n"
        "Acquisition method: AQ0 direct UCI archive\n\n"
        "Dataset revision: DATA-v1\n\n"
        f"Archive SHA-256: {archive_hash}\n\n"
        f"CSV SHA-256: {protected_hash_before}\n\n"
        "Canonical raw files must not be modified in place.\n\n"
        "Citation: Candanedo, L. (2017). Appliances Energy Prediction [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5VC8G.\n"
    )
    manifest_path = raw_root / "dataset_manifest.json"
    source_metadata_path = raw_root / "source_metadata.json"
    variable_metadata_path = raw_root / "variable_metadata.csv"
    checksums_path = raw_root / "checksums.sha256"
    readme_path = raw_root / "README_SOURCE.md"
    acquisition_log_path = artifact_root / "acquisition_log.json"
    write_json_once_or_verify(manifest_path, manifest)
    write_json_once_or_verify(source_metadata_path, source_metadata)
    write_text_once_or_verify(variable_metadata_path, variables)
    write_text_once_or_verify(checksums_path, checksums)
    write_text_once_or_verify(readme_path, readme)
    acquisition_log = {
        "run_id": "AQ0-DATA-v1",
        "environment_id": "ENV-v1",
        "dataset_revision": "DATA-v1",
        "method": "AQ0_direct_uci",
        "start_time": acquired_utc.isoformat(),
        "end_time": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "source": DATASET_METADATA["source_url"],
        "archive_path": relative_archive,
        "csv_path": relative_csv,
        "archive_size_bytes": canonical_archive.stat().st_size,
        "csv_size_bytes": protected_csv.stat().st_size,
        "archive_sha256": archive_hash,
        "csv_sha256": protected_hash_before,
        "archive_integrity_ok": True,
        "archive_paths_safe": True,
        "csv_smoke_test_ok": True,
        "ucimlrepo_crosscheck_status": "NOT_REQUIRED_OFFICIAL_AQ0_VERIFIED",
        "notes": ["Existing canonical CSV reused after exact SHA-256 match with the official archive member"],
    }
    write_json_once_or_verify(acquisition_log_path, acquisition_log)
    output_paths = [
        relative_archive,
        "data/raw_data/checksums.sha256",
        "data/raw_data/dataset_manifest.json",
        "data/raw_data/source_metadata.json",
        "data/raw_data/variable_metadata.csv",
        "data/raw_data/README_SOURCE.md",
        "artifacts/acquisition/acquisition_log.json",
    ]
    output_checksums = {relative: sha256_file(root / relative) for relative in output_paths}
    signoff = {
        "artifact_version": "DATA-v1",
        "phase_id": 2,
        "phase_version": "PHASE-2-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment_id": "ENV-v1",
        "dataset_revision": "DATA-v1",
        "input_paths": ["artifacts/environment/phase_1_signoff.json", relative_csv],
        "input_checksums": {
            "artifacts/environment/phase_1_signoff.json": sha256_file(root / "artifacts/environment/phase_1_signoff.json"),
            relative_csv: protected_hash_before,
        },
        "output_paths": output_paths,
        "output_checksums": output_checksums,
        "config_fingerprint": phase_1["config_fingerprint"],
        "status": "PASS",
        "tests": [
            "official_source_identity",
            "zip_integrity",
            "safe_member_path",
            "expected_csv_member",
            "official_csv_checksum_match",
            "minimal_csv_parse",
            "raw_immutability",
        ],
        "warnings": [],
        "discrepancies": [],
    }
    write_json_once_or_verify(signoff_path, signoff)
    return signoff
