import hashlib
from pathlib import Path

from practice_2_2.paths import (
    get_practice_2_2_root,
    get_practice_2_root,
    get_repo_root,
)
from practice_2_2.resources import (
    file_sha256,
    load_canonical_registry,
    load_immutable_artifact,
    resolve_best_checkpoint,
    resolve_dataset_root,
    resolve_final_test_dir,
    resolve_persisted_path,
    resolve_resource,
    resolve_split_manifest,
    verify_canonical_resources,
)


CHECKPOINT_SHA256 = (
    "4f65bec200d023c51f95493833d057029b83a92729c3b88dd9159158dd949ae3"
)


def test_project_roots_do_not_depend_on_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    repo = get_repo_root()
    assert (repo / "total_practice").is_dir()
    assert get_practice_2_root() == repo / "total_practice" / "practice_2"
    assert get_practice_2_2_root() == repo / "total_practice" / "practice_2_2"


def test_registry_contains_only_practice_2_2_relative_paths():
    registry = load_canonical_registry()
    for value in registry["resources"].values():
        path = Path(value)
        assert not path.is_absolute()
        assert ".." not in path.parts


def test_canonical_resource_resolution_and_fast_integrity():
    assert resolve_dataset_root().is_dir()
    assert resolve_split_manifest().is_file()
    assert resolve_final_test_dir().is_dir()
    assert file_sha256(resolve_best_checkpoint()) == CHECKPOINT_SHA256
    result = verify_canonical_resources(verify_dataset_contents=False)
    assert result["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert result["test_loader_constructed"] is False
    assert result["test_evaluated"] is False


def test_immutable_adapter_falls_back_without_mutating_artifact():
    selection_path = resolve_resource("final_selection", require_file=True)
    before = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    artifact = load_immutable_artifact("final_selection")
    historical_value = artifact["checkpoint_path"]
    in_memory = dict(artifact)
    in_memory["checkpoint_path"] = "/missing/historical/best.pt"
    resolved = resolve_persisted_path(
        in_memory,
        "checkpoint_path",
        "e2_best_checkpoint",
        expected_sha256=CHECKPOINT_SHA256,
    )
    assert resolved == resolve_best_checkpoint()
    assert artifact["checkpoint_path"] == historical_value
    assert hashlib.sha256(selection_path.read_bytes()).hexdigest() == before
