from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from course_work.model_improvement_v2 import step17_post_hoc_benchmark as step17


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def preflight():
    return step17.run_preflight(ROOT)


def test_preflight_locks_post_hoc_label_and_final_policy(preflight):
    assert preflight["status"] == "PASS"
    assert preflight["benchmark_label"] == "POST_HOC_V2_BENCHMARK"
    assert "NOT_UNBIASED_UNSEEN_TEST" in preflight["old_test_interpretation"]
    assert preflight["prediction_policy"] == {
        "policy_id": "MEAN_E14_M1_SEEDS_42_123_2026",
        "candidate_id": "TR_C2_ALT_LOOKBACK_E14_M1",
        "seeds": [42, 123, 2026],
        "weights": [1 / 3, 1 / 3, 1 / 3],
    }


def test_preflight_verifies_exact_lock_checkpoints_and_scalers(preflight):
    assert preflight["final_lock"]["status"] == "PASS"
    assert preflight["checkpoints"]["status"] == "PASS"
    assert preflight["checkpoints"]["count"] == 3
    assert [row["seed"] for row in preflight["checkpoints"]["records"]] == [42, 123, 2026]
    assert all(row["status"] == "PASS" for row in preflight["checkpoints"]["records"])
    assert preflight["scalers"]["status"] == "PASS"
    assert preflight["scalers"]["mode"] == "TRANSFORM_ONLY"


def test_preflight_does_not_open_test_or_load_checkpoint_payload(monkeypatch):
    monkeypatch.setattr(step17, "_open_feature_source", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Test source opened")))
    monkeypatch.setattr(step17.torch, "load", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("checkpoint payload loaded")))
    result = step17.run_preflight(ROOT)
    assert result["test_access_guard"] == {
        "status": "PASS",
        "test_files_opened": 0,
        "test_rows_read": 0,
        "test_target_ids_seen": 0,
    }
    assert result["checkpoints"]["payloads_loaded"] == 0
    assert result["training_executed"] is False
    assert result["inference_executed"] is False


def test_test_access_guard_is_strictly_one_shot():
    calls = []

    def loader(root, features, contract):
        calls.append((root, tuple(features), contract))
        return "loaded"

    guard = step17.TestAccessGuard(loader)
    assert guard.load_once(Path("/tmp/project"), ("f0",), {}) == "loaded"
    with pytest.raises(PermissionError, match="exactly once"):
        guard.load_once(Path("/tmp/project"), ("f0",), {})
    assert guard.access_count == 1
    assert len(calls) == 1


def test_population_fingerprint_is_locked_without_opening_test():
    target_ids = [f"TGT_{value:08d}" for value in range(16774, 19735)]
    assert len(target_ids) == 2961
    assert step17._population_fingerprint(target_ids) == step17.EXPECTED_TEST_POPULATION_FINGERPRINT


def test_equal_weight_ensemble_and_metric_shapes():
    y_true = np.asarray([1.0, 2.0, 4.0])
    predictions = {
        42: np.asarray([1.0, 2.0, 3.0]),
        123: np.asarray([2.0, 2.0, 4.0]),
        2026: np.asarray([0.0, 2.0, 5.0]),
    }
    ensemble = sum(weight * predictions[seed] for weight, seed in zip(step17.ENSEMBLE_WEIGHTS, step17.FINAL_SEEDS))
    assert np.allclose(ensemble, np.asarray([1.0, 2.0, 4.0]), rtol=0, atol=1e-15)
    metrics = step17._metrics(y_true, ensemble)
    assert metrics["sample_count"] == 3
    assert metrics["rmse_wh"] == pytest.approx(0.0, abs=1e-15)
    assert metrics["mae_wh"] == pytest.approx(0.0, abs=1e-15)
    assert metrics["r2"] == pytest.approx(1.0)


def test_benchmark_refuses_without_both_human_flags(monkeypatch):
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(step17, "run_benchmark", forbidden)
    base = ["--experiment", "STEP17", "--mode", "benchmark"]
    assert step17.main(base) == 3
    assert step17.main(base + ["--authorize-post-hoc-benchmark"]) == 3
    assert step17.main(base + ["--acknowledge-old-test-not-unseen"]) == 3
    assert called is False


def test_preflight_rejects_authorization_flags_without_access(monkeypatch):
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(step17, "run_preflight", forbidden)
    result = step17.main([
        "--experiment", "STEP17",
        "--mode", "preflight",
        "--authorize-post-hoc-benchmark",
    ])
    assert result == 2
    assert called is False
