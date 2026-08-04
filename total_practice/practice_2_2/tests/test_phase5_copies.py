import hashlib
import json
from pathlib import Path

from practice_2_2.paths import get_practice_2_2_root
from practice_2_2.resources import (
    file_sha256,
    load_canonical_registry,
    resolve_registry_resource,
    verify_canonical_resources,
)


RUN_ID = "canonical_26dc4625_52aaf974_s42_v1"
E3_RUN_ID = "canonical_26dc4625_52aaf974_s42_phase25_e3_layer4_1_v1"
E4_RUN_ID = "canonical_26dc4625_52aaf974_s42_phase26_e4_augmentation_v1"


def _directory_digest(root):
    root = Path(root)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(
            f"{file_sha256(path)} {path.stat().st_size} {relative}\n".encode()
        )
    return len(files), sum(path.stat().st_size for path in files), digest.hexdigest()


def test_phase5_source_destination_trees_are_byte_identical():
    root = get_practice_2_2_root()
    pairs = [
        (
            root / "archive/phase_7/pre_migration_practice_2_2/data/data_clean_balanced",
            root / "data/final/data_clean_balanced",
        ),
        (
            root / "archive/phase_7/pre_migration_practice_2_2/outputs/practice_2_2/canonical_split",
            root / "data/manifests/canonical_split",
        ),
        (
            root / "archive/phase_7/pre_migration_practice_2_2/runs/practice_2_2" / RUN_ID,
            root / "artifacts/canonical" / RUN_ID / "checkpoints",
        ),
        (
            root / "archive/phase_7/pre_migration_practice_2_2/outputs/practice_2_2" / RUN_ID,
            root / "artifacts/canonical" / RUN_ID / "outputs",
        ),
        (
            root / "archive/phase_7/pre_migration_practice_2_2/runs/practice_2_2" / E3_RUN_ID,
            root / "artifacts/ablations/E3" / E3_RUN_ID / "checkpoints",
        ),
        (
            root / "archive/phase_7/pre_migration_practice_2_2/outputs/practice_2_2" / E3_RUN_ID,
            root / "artifacts/ablations/E3" / E3_RUN_ID / "outputs",
        ),
        (
            root / "archive/phase_7/pre_migration_practice_2_2/runs/practice_2_2" / E4_RUN_ID,
            root / "artifacts/ablations/E4" / E4_RUN_ID / "checkpoints",
        ),
        (
            root / "archive/phase_7/pre_migration_practice_2_2/outputs/practice_2_2" / E4_RUN_ID,
            root / "artifacts/ablations/E4" / E4_RUN_ID / "outputs",
        ),
    ]
    for source, destination in pairs:
        assert source.is_dir() and destination.is_dir()
        assert _directory_digest(source) == _directory_digest(destination)


def test_phase5_draft_is_retained_after_phase6_promotion():
    root = get_practice_2_2_root()
    active = load_canonical_registry()
    draft_path = root / "configs/canonical_registry_phase5_draft.json"
    draft = load_canonical_registry(draft_path)
    assert active["authority_status"] == "active"
    assert active["resources"]["dataset_root"] == "data/final/data_clean_balanced"
    assert draft["authority_status"] == "verified_copy_not_active"
    assert draft["resources"]["dataset_root"] == "data/final/data_clean_balanced"
    result = verify_canonical_resources(
        registry_path=draft_path, verify_dataset_contents=False
    )
    assert result["authority_status"] == "verified_copy_not_active"


def test_phase5_checkpoint_and_ablation_hashes():
    root = get_practice_2_2_root()
    draft = load_canonical_registry(
        root / "configs/canonical_registry_phase5_draft.json"
    )
    expected = draft["expected"]
    assert file_sha256(
        resolve_registry_resource(draft, "e2_best_checkpoint", require_file=True)
    ) == expected["checkpoint_sha256"]
    e3_best = resolve_registry_resource(draft, "e3_run", require_file=False)
    e3_best = e3_best / "E3_layer4_1_head/best.pt"
    e4_best = resolve_registry_resource(draft, "e4_run", require_file=False)
    e4_best = e4_best / "E4_moderate_augmentation/best.pt"
    assert file_sha256(e3_best) == expected["e3_best_checkpoint_sha256"]
    assert file_sha256(e4_best) == expected["e4_best_checkpoint_sha256"]


def test_phase5_copied_final_guard_and_lineage_are_unchanged():
    root = get_practice_2_2_root()
    source = (
        root
        / "archive/phase_7/pre_migration_practice_2_2/outputs/practice_2_2"
        / RUN_ID
        / "final_test"
    )
    copied = root / "artifacts/canonical" / RUN_ID / "outputs/final_test"
    for name in [
        "final_selection.json",
        "final_test_summary.json",
        "final_test_lineage.json",
        "FINAL_TEST_COMPLETED.json",
    ]:
        assert file_sha256(source / name) == file_sha256(copied / name)
    guard = json.loads((copied / "FINAL_TEST_COMPLETED.json").read_text())
    assert guard["FINAL_TEST_COMPLETED"] is True
    assert guard["final_test_evaluation_count"] == 1
    assert guard["repeat_evaluation_allowed"] is False
    assert guard["maintenance_override_used"] is False
