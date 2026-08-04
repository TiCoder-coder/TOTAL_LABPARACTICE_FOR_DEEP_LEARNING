import json
from pathlib import Path

from practice_2_2 import compatibility_status
from practice_2_2.resources import (
    load_canonical_registry,
    load_legacy_compat_registry,
    resolve_best_checkpoint,
    resolve_dataset_root,
    resolve_legacy_resource,
    resolve_resource,
    resolve_split_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def test_active_registry_is_new_authority():
    registry = load_canonical_registry()
    assert registry["authority_status"] == "active"
    assert resolve_dataset_root() == (ROOT / "data/final/data_clean_balanced").resolve()
    assert resolve_split_manifest() == (
        ROOT / "data/manifests/canonical_split/split_manifest.csv"
    ).resolve()
    assert resolve_best_checkpoint() == (
        ROOT
        / "artifacts/canonical/canonical_26dc4625_52aaf974_s42_v1"
        / "checkpoints/E2_partial_finetune/best.pt"
    ).resolve()
    assert resolve_resource("canonical_output_dir", require_file=False) == (
        ROOT
        / "artifacts/canonical/canonical_26dc4625_52aaf974_s42_v1/outputs"
    ).resolve()


def test_old_layout_requires_explicit_legacy_registry():
    legacy = load_legacy_compat_registry()
    assert legacy["authority_status"] == "legacy_compatibility_read_only"
    assert resolve_legacy_resource("dataset_root", require_file=False) == (
        ROOT / "archive/phase_7/pre_migration_practice_2_2/data/data_clean_balanced"
    ).resolve()
    assert resolve_dataset_root() != resolve_legacy_resource(
        "dataset_root", require_file=False
    )


def test_import_and_notebook_authority_are_switched_safely():
    status = compatibility_status()
    assert status["phase"] == 6
    assert status["canonical_entrypoint_switched"] is True
    notebook = json.loads((ROOT / "notebooks/04_canonical_report.ipynb").read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "load_canonical_registry" in source
    assert 'authority_status") == "active"' in source
    assert ' / "outputs" / "practice_2_2"' not in source
    assert "DataLoader(" not in source
    assert "optimizer.step(" not in source
    assert ".train(" not in source
