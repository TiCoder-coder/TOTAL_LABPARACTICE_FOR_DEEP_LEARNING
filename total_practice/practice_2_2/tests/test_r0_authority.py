import hashlib
from pathlib import Path

import pytest

from practice_2_2.r0_authority import (
    assert_r0_ready,
    build_r0_inventory,
    directory_digest,
    load_r0_policy,
    resolve_policy_path,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIRECTORY_SHA256 = "fd1cd4557352b28054b976feebd69932c34af9edda44eb1251e8f46b7e97e8a5"


def test_r0_policy_keeps_canonical_and_final_test_locked():
    policy = load_r0_policy()
    assert policy["implementation_authorized"] is False
    assert policy["canonical_notebook_mutation_authorized"] is False
    assert policy["final_test_re_evaluation_authorized"] is False
    assert "notebooks/04_canonical_report.ipynb" in policy["classifications"]["frozen_historical"]


def test_r0_policy_paths_are_relative_and_confined():
    policy = load_r0_policy()
    for authority in policy["authorities"]:
        path = Path(authority["path"])
        assert not path.is_absolute()
        assert ".." not in path.parts
        assert ROOT in resolve_policy_path(authority["path"]).parents


def test_restored_canonical_dataset_matches_phase5_record():
    result = directory_digest(ROOT / "data/final/data_clean_balanced")
    assert result == {
        "file_count": 3202,
        "total_bytes": 48502262,
        "directory_sha256": DATASET_DIRECTORY_SHA256,
    }


def test_r0_inventory_fails_closed_without_all_authorities():
    notebook = ROOT / "notebooks/04_canonical_report.ipynb"
    before = hashlib.sha256(notebook.read_bytes()).hexdigest()
    inventory = build_r0_inventory()
    after = hashlib.sha256(notebook.read_bytes()).hexdigest()
    assert inventory["status"] == "blocked"
    assert inventory["implementation_authorized"] is False
    assert inventory["test_loader_constructed"] is False
    assert inventory["test_evaluated"] is False
    assert before == after
    with pytest.raises(RuntimeError, match="R0 authority gate is blocked"):
        assert_r0_ready(inventory)


def test_r0_module_does_not_import_training_or_test_loading():
    source = (ROOT / "src/practice_2_2/r0_authority.py").read_text()
    assert "torch" not in source
    assert "DataLoader" not in source
    assert "create_test_dataset" not in source
    assert "optimizer" not in source
