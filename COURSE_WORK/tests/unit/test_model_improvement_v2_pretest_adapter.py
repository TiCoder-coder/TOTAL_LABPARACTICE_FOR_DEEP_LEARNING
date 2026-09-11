from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from course_work.model_improvement_v2.pretest_adapter import (
    build_e01_pretest_dataset,
)
from course_work.model_improvement_v2.runner import (
    build_e01_candidate,
    build_e01_run_context,
    LOCKED_E01_COMPLETED_RUN_IDS,
    load_e01_config_snapshot,
    main,
    run_resume_stage_c_e01,
)
from course_work.rolling_origin.real_run import (
    _prepare_stage_c_model_for_device,
    fit_fold_local_scaler,
)
from course_work.rolling_origin.stages import (
    load_fold_subset_loader_with_y_rescale,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _build():
    document = load_e01_config_snapshot(PROJECT_ROOT)
    dataset, evidence, audit = build_e01_pretest_dataset(
        PROJECT_ROOT,
        tuple(document["v2_orchestration"]["feature_order"]),
    )
    return document, dataset, evidence, audit


def test_adapter_loads_only_canonical_pretest_population():
    document, dataset, evidence, audit = _build()
    assert audit.rows_loaded == 13_814 + 2_960
    assert audit.test_rows_read == 0
    assert audit.test_target_ids_seen == 0
    assert audit.feature_order_status == "PASS"
    assert audit.population_status == "PASS"
    assert audit.fold_fingerprint_status == {
        "RO1": "PASS",
        "RO2": "PASS",
        "RO3": "PASS",
    }
    assert len(dataset) == 16_630
    assert dataset.window_records["target_id"].tolist() == list(
        evidence.train_ids + evidence.validation_ids
    )
    assert tuple(dataset[0]["x"].shape) == (72, 33)
    assert tuple(dataset[-1]["x"].shape) == (72, 33)
    assert document["candidate_id"] == "TR_C2_ALT_LOOKBACK"


def test_v2_loader_applies_fold_local_x_and_y_scaling():
    document, dataset, evidence, _audit = _build()
    candidate = build_e01_candidate(document)
    fold = evidence.folds[0]
    bundle, scaler_audit = fit_fold_local_scaler(
        candidate=candidate,
        fold=fold,
        fold_dataset=dataset,
        fit_stage="A",
    )
    assert scaler_audit.status == "PASS"
    assert scaler_audit.test_rows_used == 0
    assert len(bundle.feature_indices_passthrough) == 0

    target_id = fold.inner_val_ids[0]
    loader, indices = load_fold_subset_loader_with_y_rescale(
        base_dataset=dataset,
        target_ids=[target_id],
        batch_size=1,
        shuffle=False,
        fold_stage_target_scaler=bundle,
        apply_fold_x_scaling=True,
    )
    batch = next(iter(loader))
    raw = dataset[indices[0]]
    expected_x = bundle.transform_x(raw["x"].numpy())
    expected_y = bundle.transform_y(raw["y_raw_wh"].numpy())
    assert np.allclose(batch["x"].numpy()[0], expected_x)
    assert np.allclose(batch["y_model"].numpy()[0], expected_y)


def test_runner_context_injects_phase44_population_without_windowpop():
    document = load_e01_config_snapshot(PROJECT_ROOT)
    context = build_e01_run_context(PROJECT_ROOT, document)
    assert context.apply_fold_x_scaling is True
    assert len(context.robase_train_ids) == 13_670
    assert len(context.robase_val_ids) == 2_960
    assert context.dataset_factory is not None
    assert "model_improvement_v2" in str(context.artifact_dir)



def test_stage_c_device_fix_is_v2_scoped():
    class TrackingModel:
        def __init__(self):
            self.moves = []

        def to(self, device):
            self.moves.append(device)
            return self

    selected = torch.device("mps")
    v2_model = TrackingModel()
    assert _prepare_stage_c_model_for_device(
        model=v2_model, device=selected, execution_track="MODEL_IMPROVEMENT_V2_E01"
    ) is v2_model
    assert v2_model.moves == [selected]

    v1_model = TrackingModel()
    assert _prepare_stage_c_model_for_device(
        model=v1_model, device=selected, execution_track="V1"
    ) is v1_model
    assert v1_model.moves == []


def test_stage_c_resume_requires_human_authorization():
    assert main(["--experiment", "E01", "--mode", "resume-stage-c", "--seed", "42"]) == 3


def test_stage_c_resume_reuses_only_locked_runs(monkeypatch):
    captured = {}

    def fake_pipeline(context):
        captured["context"] = context
        return SimpleNamespace(exit_code=1, summary="STATIC_STOP", exception=None)

    monkeypatch.setattr(
        "course_work.rolling_origin.real_run.run_real_pipeline", fake_pipeline
    )
    assert run_resume_stage_c_e01(PROJECT_ROOT) == 1
    context = captured["context"]
    assert context.reuse_completed_runs is True
    assert context.reuse_completed_run_ids == LOCKED_E01_COMPLETED_RUN_IDS
    assert context.execution_track == "MODEL_IMPROVEMENT_V2_E01"
