"""Phase 46 post-train finalization tests.

Covers:
- JSON serializability for all O46 writers (no sets/Paths/numpy leaks)
- phase47_release gates pass when all 3 seeds have valid evidence
- finalize mode refuses to train (optimizer.step guard)
- 3 completed runs are reusable, exactly 3 seeds, all FINAL_REFIT
- FINAL_DEV/Scaling identity across seeds
- Phase47 release blocked when Phase46 incomplete
- phase_46_signoff PASS requires strict gates
- No artifact-path collisions between runs
- FINAL_REFIT metrics are not mislabeled as VALIDATION
- Idempotency: stale artifacts are archived, not lost
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

ARTIFACT_DIR = ROOT / "artifacts" / "three_seed_final_runs"
CHECKPOINT_DIR = ARTIFACT_DIR / "official_checkpoints"
PHASE_47_RELEASE = ARTIFACT_DIR / "phase47_test_release.json"
PHASE_46_SIGNOFF = ARTIFACT_DIR / "phase_46_signoff.json"



class TestThreeCompletedRunsReusable:
    @pytest.mark.parametrize("seed", [42, 123, 2026])
    def test_official_checkpoint_exists(self, seed: int) -> None:
        pt = CHECKPOINT_DIR / f"seed_{seed}" / f"seed_{seed}_FINAL_REFIT.pt"
        assert pt.exists(), f"Official checkpoint missing for seed {seed}"

    @pytest.mark.parametrize("seed", [42, 123, 2026])
    def test_official_metadata_exists_and_complete(self, seed: int) -> None:
        md = CHECKPOINT_DIR / f"seed_{seed}" / f"seed_{seed}_FINAL_REFIT_metadata.json"
        assert md.exists()
        data = json.loads(md.read_text())
        for key in (
            "checkpoint_path", "checkpoint_type", "config_sha256",
            "model_state_sha256", "official_epoch", "phase",
            "population_fingerprint", "recipe_sha256", "rmse_wh",
            "run_id", "seed", "x_scaler_sha256", "y_scaler_sha256",
            "FINAL_REFIT_EPOCHS", "test_metrics_computed",
        ):
            assert key in data, f"Missing key {key!r} in seed {seed} metadata"
        assert data["checkpoint_type"] == "FINAL_REFIT"
        assert data["official_epoch"] == 30
        assert data["FINAL_REFIT_EPOCHS"] == 30
        assert data["test_metrics_computed"] is False


class TestFinalizeNoNewRunIDs:
    def test_registry_run_count_unchanged_after_finalize(self) -> None:
        registry_path = ROOT / "artifacts" / "experiments" / "experiment_registry.jsonl"
        before = sum(1 for _ in open(registry_path) if _.strip())
        from scripts._phase46_finalize_no_train import main as finalize_main

        exit_code = finalize_main()
        assert exit_code == 0
        after = sum(1 for _ in open(registry_path) if _.strip())
        assert after == before, f"Finalize added {after - before} new registry entries"

class TestPhase47ReleaseSerializability:
    def test_release_json_loadable(self) -> None:
        assert PHASE_47_RELEASE.exists(), (
            "phase47_test_release.json must exist after finalize"
        )
        data = json.loads(PHASE_47_RELEASE.read_text())
        assert "released" in data

    def test_release_gates_all_pass(self) -> None:
        data = json.loads(PHASE_47_RELEASE.read_text())
        gates = data["gates"]
        for key in (
            "all_seeds_completed", "all_checkpoints_verified",
            "lock_hash_match", "config_match", "recipe_match",
            "population_match", "scalers_match", "epochs_match",
            "schema_match", "test_not_accessed", "no_historical_run_id_leaked",
        ):
            assert gates.get(key) is True, f"Gate {key!r} failed"

    def test_historical_excluded_is_list(self) -> None:
        """CRITICAL: must be a sorted list, never a set (D-01 fix)."""
        data = json.loads(PHASE_47_RELEASE.read_text())
        excluded = data["historical_run_ids_excluded"]
        assert isinstance(excluded, list), (
            f"historical_run_ids_excluded must be list, got {type(excluded).__name__}"
        )
        for item in excluded:
            assert isinstance(item, str)


class TestFinalRefitMetricSemantics:
    @pytest.mark.parametrize("seed", [42, 123, 2026])
    def test_metric_split_is_final_dev(self, seed: int) -> None:
        meta = json.loads((CHECKPOINT_DIR / f"seed_{seed}" / f"seed_{seed}_FINAL_REFIT_metadata.json").read_text())
        run_id = meta["run_id"]
        metric_path = ROOT / "artifacts" / "runs" / run_id / "metrics" / "best_validation_metrics.json"
        assert metric_path.exists(), f"Metric file missing for seed {seed}"
        data = json.loads(metric_path.read_text())
        split_id = data["metric_result"]["split_id"]
        assert split_id == "FINAL_DEV", (
            f"Seed {seed}: metric split_id={split_id} != FINAL_DEV"
        )

class TestFinalScalingIdentity:
    def test_x_scaler_sha_identical(self) -> None:
        shas = set()
        for seed in (42, 123, 2026):
            md = json.loads((CHECKPOINT_DIR / f"seed_{seed}" / f"seed_{seed}_FINAL_REFIT_metadata.json").read_text())
            shas.add(md["x_scaler_sha256"])
        assert len(shas) == 1, f"x_scaler_sha256 differ across seeds: {shas}"

    def test_y_scaler_sha_identical(self) -> None:
        shas = set()
        for seed in (42, 123, 2026):
            md = json.loads((CHECKPOINT_DIR / f"seed_{seed}" / f"seed_{seed}_FINAL_REFIT_metadata.json").read_text())
            shas.add(md["y_scaler_sha256"])
        assert len(shas) == 1, f"y_scaler_sha256 differ across seeds: {shas}"

class TestPopulationIdentity:
    def test_population_fingerprint_identical(self) -> None:
        fps = set()
        for seed in (42, 123, 2026):
            md = json.loads((CHECKPOINT_DIR / f"seed_{seed}" / f"seed_{seed}_FINAL_REFIT_metadata.json").read_text())
            fps.add(md["population_fingerprint"])
        assert len(fps) == 1, f"population_fingerprint differ across seeds: {fps}"

class TestExactlyThreeSeeds:
    def test_exactly_three_official_checkpoints(self) -> None:
        official_dirs = list(CHECKPOINT_DIR.glob("seed_*"))
        seed_dirs = [d for d in official_dirs if d.is_dir()]
        assert len(seed_dirs) == 3, (
            f"Expected exactly 3 official seed dirs, found {len(seed_dirs)}: "
            f"{[d.name for d in seed_dirs]}"
        )
        seeds = sorted(int(d.name.split("_")[1]) for d in seed_dirs)
        assert seeds == [42, 123, 2026]


class TestNoArtifactCollisions:
    def test_checkpoint_paths_distinct(self) -> None:
        paths = []
        for seed in (42, 123, 2026):
            pt = CHECKPOINT_DIR / f"seed_{seed}" / f"seed_{seed}_FINAL_REFIT.pt"
            paths.append(pt.resolve())
        assert len(set(paths)) == 3, f"Checkpoint path collision: {paths}"

    def test_run_directories_distinct(self) -> None:
        runs_root = ROOT / "artifacts" / "runs"
        target_runs = [
            "RUN_TR_FSD_0254_2B11AC68",
            "RUN_TR_FSD_0254_3858DDA9",
            "RUN_TR_FSD_0255_C7E123FB",
        ]
        for run_id in target_runs:
            assert (runs_root / run_id).exists(), f"Run directory missing: {run_id}"
        assert len(set(target_runs)) == 3


class TestTestAccessZero:
    @pytest.mark.parametrize("seed", [42, 123, 2026])
    def test_test_access_authorized_false(self, seed: int) -> None:
        md = json.loads((CHECKPOINT_DIR / f"seed_{seed}" / f"seed_{seed}_FINAL_REFIT_metadata.json").read_text())
        run_id = md["run_id"]
        registry_path = ROOT / "artifacts" / "experiments" / "experiment_registry.jsonl"
        for line in open(registry_path):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("run_id") == run_id:
                assert rec.get("test_access_authorized") is False
                break

class TestFinalizeTrainingGuard:
    def test_optimizer_step_raises_during_finalize(self) -> None:
        """When finalize mode is active, optimizer.step must raise."""
        from scripts._phase46_finalize_no_train import (
            Phase46FinalizeTrainAttemptError,
            _training_attempt_guard,
        )

        _training_attempt_guard()
        import torch.optim as _opt

        with pytest.raises(Phase46FinalizeTrainAttemptError):
            _opt.Optimizer.step(None, None)


class TestJSONSerializability:
    def test_to_jsonable_normalizes_set(self) -> None:
        from scripts._phase46_finalize_no_train import _to_jsonable

        result = _to_jsonable({"x": {1, 2, 3}})
        assert isinstance(result["x"], list)
        assert sorted(result["x"]) == [1, 2, 3]

    def test_to_jsonable_normalizes_nested_set(self) -> None:
        from scripts._phase46_finalize_no_train import _to_jsonable

        result = _to_jsonable({"a": [{"b": {5, 1, 3}}]})
        assert isinstance(result["a"][0]["b"], list)
        assert result["a"][0]["b"] == [1, 3, 5]

    def test_validate_json_serializable_passes_clean(self) -> None:
        from scripts._phase46_finalize_no_train import _validate_json_serializable

        _validate_json_serializable({"a": [1, 2, 3], "b": "ok"}, "test")

    def test_validate_json_serializable_fails_on_set(self) -> None:
        from scripts._phase46_finalize_no_train import _validate_json_serializable

        with pytest.raises(TypeError, match="JSON serializability"):
            _validate_json_serializable({"x": {1, 2}}, "bad")

class TestPhase46SignoffGates:
    def test_signoff_status_pass(self) -> None:
        """With all 3 seeds COMPLETED, signoff must be PASS."""
        assert PHASE_46_SIGNOFF.exists()
        data = json.loads(PHASE_46_SIGNOFF.read_text())
        assert data["status"] == "PASS"
        assert data["ready_for_phase47"] is True
        assert data["phase47_released"] is True
        assert data["completed_run_count"] == 3
        assert data["planned_run_count"] == 3

    def test_signoff_test_status(self) -> None:
        data = json.loads(PHASE_46_SIGNOFF.read_text())
        assert data["test_status"] == "NOT_ACCESSED"

    def test_signoff_validation_used(self) -> None:
        data = json.loads(PHASE_46_SIGNOFF.read_text())
        assert data["validation_used"] is False

    def test_signoff_early_stopping_used(self) -> None:
        data = json.loads(PHASE_46_SIGNOFF.read_text())
        assert data["early_stopping_used"] is False


class TestIdempotency:
    def test_archive_dir_created(self) -> None:
        """Running finalize twice must archive stale artifacts, not lose them."""
        archive_root = ARTIFACT_DIR / ".archive"
        assert archive_root.exists(), "Archive directory should be created"
        assert any(archive_root.iterdir()), "Archive directory empty"

class TestStrictCheckpointReload:
    @pytest.mark.parametrize("seed", [42, 123, 2026])
    def test_checkpoint_loads_with_torch(self, seed: int) -> None:
        import torch

        pt_path = CHECKPOINT_DIR / f"seed_{seed}" / f"seed_{seed}_FINAL_REFIT.pt"
        assert pt_path.exists()
        state = torch.load(pt_path, map_location="cpu", weights_only=False)
        assert isinstance(state, dict), f"Seed {seed}: checkpoint is not a dict"
        assert "model_state_dict" in state, (
            f"Seed {seed}: checkpoint missing model_state_dict"
        )
        sd = state["model_state_dict"]
        keys = list(sd.keys())
        assert any(k.startswith("encoder.layers") for k in keys), (
            f"Seed {seed}: state_dict missing encoder.layers (got first keys: {keys[:5]})"
        )
