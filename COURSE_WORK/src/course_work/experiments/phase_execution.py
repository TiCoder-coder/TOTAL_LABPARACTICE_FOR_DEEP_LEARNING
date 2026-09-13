from __future__ import annotations

import json
import math
import platform
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from course_work.utils.artifacts import get_project_root, read_json, sha256_file


# Exact, commit-bound compatibility contracts for legacy sweep signoffs written
# before the generic phase identity/checksum schema stabilized.
# These hashes are from historical commit
# 1f648662980c49e0ee42de9f2cf43639e6966b64.  The exception is deliberately
# fail-closed: it applies only when every retained scientific artifact below
# is byte-identical to its historical/registry-bound value.
LEGACY_SWEEP_CONTRACTS: dict[int, dict[str, Any]] = {
    37: {
        "signoff_sha256": "7553d6d5e44c9306db75419f51dc0ab95068c9dd2eff1a00950dd6aa424ffddc",
        "identity": {"phase": 37, "overall_status": "PASS"},
        "run_id": "RUN_TR_S15_0024_9420CDD7",
        "experiment_family": "S15_LOSS",
        "historical_files": {
            "artifacts/sweeps/S15_loss/s15_loss_winner.json": "55c913f59b126cc6c270c0be47cd8392b01bfa3264a97b89d5379b10c6a418e8",
            "artifacts/sweeps/S15_loss/s15_reference_update.json": "507575b39f2eb77edf03b8985194c74028fbe28447ce87ae4dfaff2e03c31334",
        },
        "core_artifacts": {
            "artifacts/runs/RUN_TR_S15_0024_9420CDD7/config.json": "58c232b1829733acfb62e3032a914b75afff03ee50d2b45acf66a8754f031646",
            "artifacts/runs/RUN_TR_S15_0024_9420CDD7/status.json": "cdcf705b6b4869f2fb36ea01bef4e4b254df25acc76d44c6b4b6758da0557ced",
            "artifacts/runs/RUN_TR_S15_0024_9420CDD7/training.log": "bfff17e357f30604e728eadb70a5a05e18735d4b35fa9dd3459923c60fcc0885",
            "artifacts/runs/RUN_TR_S15_0024_9420CDD7/checkpoints/best_checkpoint.pt": "a7e86fb884e9d060886762375c5a241bcca412ef48fe178110080c2e86a48ee3",
            "artifacts/runs/RUN_TR_S15_0024_9420CDD7/metrics/best_validation_metrics.json": "a4b13ff0378afdcff0ae02fbe8e2256876ac8bf3b067290bbf95caa2d7cebc18",
            "artifacts/runs/RUN_TR_S15_0024_9420CDD7/predictions/best_validation_predictions.csv": "7e948d5bfa7ab698f4cde5822d1fad56f85e235b8a473c2f4fdab19eadba418d",
        },
        "reporting_debt": (
            "artifacts/sweeps/S15_loss/s15_loss_sweep_manifest.json",
            "artifacts/sweeps/S15_loss/s15_loss_metrics.csv",
            "artifacts/runs/RUN_TR_S15_0024_9420CDD7/training_history.csv",
        ),
    },
    38: {
        "signoff_sha256": "d2a703576ef91c4953e66d22cca66c3c21b26ac77b132f64dcce9a06cc6a49ce",
        "identity": {"phase_id": 38, "phase_version": "PHASE-38-v1", "status": "PASS"},
        "run_id": "RUN_TR_S16_0025_49060872",
        "experiment_family": "S16_EPOCH_CAP",
        "historical_files": {
            "artifacts/sweeps/S16_epoch_cap/s16_epoch_cap_winner.json": "04779733b1aa521174cdc4b6fe25a9751926f2b93769f6189d020d90a70a9355",
            "artifacts/sweeps/S16_epoch_cap/s16_reference_update.json": "feeced256dffed14a4ad617d1adfc5066b7f85bd834e460399b9f47807ec6622",
            "artifacts/sweeps/S16_epoch_cap/s16_epoch_cap_sweep_manifest.json": "d91c3204e116b542a775c68f44757538a1fcf9bc677ea1bae39335da484d5962",
            "artifacts/sweeps/S16_epoch_cap/s16_epoch_cap_sweep_contract.json": "7bb974214ad2e94642d44ff640258e46e699e5c16711bbcebfd065051b202ca7",
            "artifacts/sweeps/S16_epoch_cap/README_S16_EPOCH_CAP_SWEEP.md": "c7e5cfa29e6f9baabc6962671f91e626135323f90ee99dc06c5b96e8d737e9e5",
        },
        "core_artifacts": {
            "artifacts/runs/RUN_TR_S16_0025_49060872/config.json": "078fb97ba06a1cfa7720b45a9a793c2bcc9df29234a65372a4da5b1330bbc155",
            "artifacts/runs/RUN_TR_S16_0025_49060872/status.json": "4f8b0a9efca48d56d250f4b4f309be8f1eb139bd3db714bb25f1491415ec75ed",
            "artifacts/runs/RUN_TR_S16_0025_49060872/training.log": "bfff17e357f30604e728eadb70a5a05e18735d4b35fa9dd3459923c60fcc0885",
            "artifacts/runs/RUN_TR_S16_0025_49060872/checkpoints/best_checkpoint.pt": "1704b901c2fc337e1853dc378afed3b1c84d616bb4509f7bad51c74def079a60",
            "artifacts/runs/RUN_TR_S16_0025_49060872/metrics/best_validation_metrics.json": "3d258d9bd535bceb04e86fdf60a6f7f1bc23ef79b2f0b74577695777af5e9cdd",
            "artifacts/runs/RUN_TR_S16_0025_49060872/predictions/best_validation_predictions.csv": "2977e15cfcc23bbfc243bc815342e88b9ccfab70b61c0fe20f26db5da094171e",
        },
        "reporting_debt": (
            "artifacts/sweeps/S16_epoch_cap/s16_run_matrix.csv",
            "artifacts/sweeps/S16_epoch_cap/s16_epoch_cap_metrics.csv",
            "artifacts/sweeps/S16_epoch_cap/s16_epoch_cap_effect.csv",
            "artifacts/runs/RUN_TR_S16_0025_49060872/training_history.csv",
        ),
    },
    39: {
        "signoff_sha256": "b6a2743b31d6b6752d5684522a7df78cbf2aeb8e92206c1baf6f4045bf863226",
        "identity": {"phase_id": 39, "phase_version": "PHASE-39-v1", "status": "PASS"},
        "run_id": "RUN_TR_S17_0029_082F7FF5",
        "experiment_family": "S17_GRADIENT_CLIPPING",
        "canonical_runs": (
            ("RUN_TR_S17_0029_082F7FF5", "S17_GRADIENT_CLIPPING"),
            ("RUN_TR_S14_0023_A711A9B8", "S14_FFN"),
        ),
        "condition_evidence": (
            ("GC0", "NONE", "RUN_TR_S17_0029_082F7FF5", False),
            ("GC1", "GLOBAL_L2_MAX_NORM_1.0", "RUN_TR_S14_0023_A711A9B8", True),
        ),
        "historical_files": {
            "artifacts/sweeps/S17_gradient_clipping/s17_gradient_clip_winner.json": "fc89147316e2d6ba15daef48906bc0a08fa3658dd8080d041d47759e8b1a3b38",
            "artifacts/sweeps/S17_gradient_clipping/s17_reference_update.json": "e17f9304ba0bf664716c8f295b0bf71d5fb0489fe3afac7ab7a09463a7d8fbed",
            "artifacts/sweeps/S17_gradient_clipping/s17_gradient_clip_sweep_manifest.json": "4e7020cb9d4ec8b603f33f26272c814e09960e9532e7f8809e67d823f2a9475f",
            "artifacts/sweeps/S17_gradient_clipping/s17_gradient_clip_sweep_contract.json": "bd9f6b91f90d1c140208a24a475f9903dc597230d029d6c2e934fc880c119604",
        },
        "core_artifacts": {
            "artifacts/runs/RUN_TR_S17_0029_082F7FF5/config.json": "d54d3d346b032e59627d6f629e7f5990685268294c45fc0f266c8ab0b20d5e51",
            "artifacts/runs/RUN_TR_S17_0029_082F7FF5/status.json": "5be6a7f2cc4bb06929d1690167509b536abde3348a727e3a34061dedaf7a665f",
            "artifacts/runs/RUN_TR_S17_0029_082F7FF5/training.log": "bfff17e357f30604e728eadb70a5a05e18735d4b35fa9dd3459923c60fcc0885",
            "artifacts/runs/RUN_TR_S17_0029_082F7FF5/checkpoints/best_checkpoint.pt": "d1bf18e60031b68b79f05d5b3dfae2e00ab3d9211d73ea7067fade677254b9b1",
            "artifacts/runs/RUN_TR_S17_0029_082F7FF5/metrics/best_validation_metrics.json": "120d20206d4462d4cbd51b665731b5da6fa24d70382276354c0ff903e17ecfca",
            "artifacts/runs/RUN_TR_S17_0029_082F7FF5/predictions/best_validation_predictions.csv": "7b31e55441063185e1769e9371913585e39bf2f46a415b6c35a42ea019d5a392",
            "artifacts/runs/RUN_TR_S14_0023_A711A9B8/config.json": "6cd6f15fd11f3acab2484081e18d7325edb7937c14ef657c379e176cf443b74d",
            "artifacts/runs/RUN_TR_S14_0023_A711A9B8/status.json": "59e8211c5caef8bac00c214ef96976cbccd81b5fd216a394cb20785e64d735ec",
            "artifacts/runs/RUN_TR_S14_0023_A711A9B8/training.log": "bfff17e357f30604e728eadb70a5a05e18735d4b35fa9dd3459923c60fcc0885",
            "artifacts/runs/RUN_TR_S14_0023_A711A9B8/checkpoints/best_checkpoint.pt": "2eccd2e1b8cf6ca49169720f97783520ab0c202b0b57f8d35c1cada6f8e725c6",
            "artifacts/runs/RUN_TR_S14_0023_A711A9B8/metrics/best_validation_metrics.json": "1c289d6e44a295ea4268601c0d9661ec5741b4e72ad6645f1f4f4509ccc9572c",
            "artifacts/runs/RUN_TR_S14_0023_A711A9B8/predictions/best_validation_predictions.csv": "ed64d7d2247f5ee0ce58c906a7b447bf07e7fa21a16ff151dea73d101d35ce89",
        },
        "reporting_debt": (
            "artifacts/sweeps/S17_gradient_clipping/s17_gradient_clip_metrics.csv",
            "artifacts/sweeps/S17_gradient_clipping/s17_run_matrix.csv",
            "artifacts/sweeps/S17_gradient_clipping/s17_gradient_clip_unit_tests.csv",
            "artifacts/sweeps/S17_gradient_clipping/s17_gradient_mutation_tests.csv",
            "artifacts/sweeps/S17_gradient_clipping/s17_nonfinite_guard_tests.csv",
            "artifacts/runs/RUN_TR_S17_0029_082F7FF5/training_history.csv",
        ),
    },
    40: {
        "signoff_sha256": "85c63506db443ae3fd4cc22b7c35b240f7eb3dbe16f6d4b0c9e60ecc06eb5744",
        "identity": {"phase_id": 40, "phase_version": "PHASE-40-v1", "phase_status": "PASS"},
        "run_id": "RUN_TR_S18_0031_A711A9B8",
        "experiment_family": "S18_REVIN",
        "canonical_runs": (
            ("RUN_TR_S14_0023_A711A9B8", "S14_FFN"),
            ("RUN_TR_S18_0031_A711A9B8", "S18_REVIN"),
        ),
        "condition_evidence": (
            ("RN0", False, "RUN_TR_S14_0023_A711A9B8", True),
            ("RN1", True, "RUN_TR_S18_0031_A711A9B8", False),
        ),
        "historical_files": {
            "artifacts/sweeps/S18_revin/s18_revin_winner.json": "fdfec1095c3f175716b2450f991b0eedaa45b26e83aa756320487a317ca9b6c7",
            "artifacts/sweeps/S18_revin/s18_reference_update.json": "d41159dcbb99a98c0a7505bce1a37865872a77640ac4d19522d330d2eb57b8ff",
            "artifacts/sweeps/S18_revin/s18_revin_sweep_manifest.json": "082dfa3e22dd9d74fc29dbd70b3212e9d131da60adc85f41026eede0b06a371a",
            "artifacts/sweeps/S18_revin/s18_revin_sweep_contract.json": "ad91f2e286665f4fed2301db2f574c74dce393088bf71ee210dd78635367d153",
            "artifacts/sweeps/S18_revin/rn1_external_strict_best_report.json": "b480d9d4aed2d2255e56114a1c8685cfaddc0cdc7503d4b8edf7ef6b37dfb5e6",
        },
        "core_artifacts": {
            "artifacts/runs/RUN_TR_S14_0023_A711A9B8/config.json": "6cd6f15fd11f3acab2484081e18d7325edb7937c14ef657c379e176cf443b74d",
            "artifacts/runs/RUN_TR_S14_0023_A711A9B8/status.json": "59e8211c5caef8bac00c214ef96976cbccd81b5fd216a394cb20785e64d735ec",
            "artifacts/runs/RUN_TR_S14_0023_A711A9B8/training.log": "bfff17e357f30604e728eadb70a5a05e18735d4b35fa9dd3459923c60fcc0885",
            "artifacts/runs/RUN_TR_S14_0023_A711A9B8/checkpoints/best_checkpoint.pt": "2eccd2e1b8cf6ca49169720f97783520ab0c202b0b57f8d35c1cada6f8e725c6",
            "artifacts/runs/RUN_TR_S14_0023_A711A9B8/metrics/best_validation_metrics.json": "1c289d6e44a295ea4268601c0d9661ec5741b4e72ad6645f1f4f4509ccc9572c",
            "artifacts/runs/RUN_TR_S14_0023_A711A9B8/predictions/best_validation_predictions.csv": "ed64d7d2247f5ee0ce58c906a7b447bf07e7fa21a16ff151dea73d101d35ce89",
            "artifacts/runs/RUN_TR_S18_0031_A711A9B8/config.json": "ae791d475bfed10fee3e9d6b1c671f51657604a7a332289d928752ba1b41e0a6",
            "artifacts/runs/RUN_TR_S18_0031_A711A9B8/status.json": "bbfdf886182a008edea5e5aa652c16bba6f21175f169b397a440514a8e9ec6f9",
            "artifacts/runs/RUN_TR_S18_0031_A711A9B8/training.log": "1ad36c078ceb20f16d9df9d846316ac7055bd8a0f101a6298bfaba3ce9cb6946",
            "artifacts/runs/RUN_TR_S18_0031_A711A9B8/checkpoints/best_checkpoint.pt": "b8f136d1f94acff1ae99cbb11369d3874273622ef74c3a3b8d1a3f20ca77785c",
            "artifacts/runs/RUN_TR_S18_0031_A711A9B8/metrics/best_validation_metrics.json": "0aaf9e1cf47c5e157a8c976141673020a8a3c52362495bfa8f83fa19480331ca",
            "artifacts/runs/RUN_TR_S18_0031_A711A9B8/predictions/best_validation_predictions.csv": "65a352bed42d0fa3d4a9a513fe05060ee1e833ecc70731f7d1e9504984fda7c5",
        },
        "reporting_debt": (
            "artifacts/sweeps/S18_revin/s18_revin_metrics.csv",
            "artifacts/runs/RUN_TR_S18_0031_A711A9B8/training_history.csv",
        ),
    },
    41: {
        "signoff_sha256": "02dabbf77bed4d6c7ae641f5e4936a4227044a27f40705de9dace580cc06acd5",
        "identity": {"phase_id": 41, "phase_version": "PHASE-41-v2", "phase_status": "COMPLETED"},
        "run_id": "RUN_TR_S19_0034_CF8C1FE8",
        "experiment_family": "S19_BOUNDARY_PROTOCOL",
        "canonical_runs": (
            ("RUN_TR_S14_0023_A711A9B8", "S14_FFN"),
            ("RUN_TR_S19_0034_CF8C1FE8", "S19_BOUNDARY_PROTOCOL"),
        ),
        "condition_evidence": (
            ("WB0", "WB0_CONTEXT_CARRY_OVER", "RUN_TR_S14_0023_A711A9B8", True),
            ("WB1", "WB1_STRICT_ISOLATION", "RUN_TR_S19_0034_CF8C1FE8", False),
        ),
        "historical_files": {
            "artifacts/sweeps/S19_boundary_protocol/s19_reference_update.json": "5b66dbefc36b830d896914fc3122ed0d83b9dc8c46e40fbefde659831ed2705a",
            "artifacts/sweeps/S19_boundary_protocol/s19_boundary_sweep_manifest.json": "a90bfdce55ee557ef1abb7c7dc3f4d6b8ccd55b103ac66d44c9c6b37392aca91",
            "artifacts/sweeps/S19_boundary_protocol/s19_boundary_sweep_contract.json": "65d38fd731cf4cbdf89c07c9939497ae7c66034d320e1b62a6af27a6b7d0af86",
            "artifacts/sweeps/S19_boundary_protocol/s19_boundary_discrepancies.json": "1b757cb8ef6cf4cc6bee286307d40904cedcc6f4bb9c95288893ea737b0e5e10",
        },
        "core_artifacts": {
            "artifacts/runs/RUN_TR_S14_0023_A711A9B8/config.json": "6cd6f15fd11f3acab2484081e18d7325edb7937c14ef657c379e176cf443b74d",
            "artifacts/runs/RUN_TR_S14_0023_A711A9B8/status.json": "59e8211c5caef8bac00c214ef96976cbccd81b5fd216a394cb20785e64d735ec",
            "artifacts/runs/RUN_TR_S14_0023_A711A9B8/training.log": "bfff17e357f30604e728eadb70a5a05e18735d4b35fa9dd3459923c60fcc0885",
            "artifacts/runs/RUN_TR_S14_0023_A711A9B8/checkpoints/best_checkpoint.pt": "2eccd2e1b8cf6ca49169720f97783520ab0c202b0b57f8d35c1cada6f8e725c6",
            "artifacts/runs/RUN_TR_S14_0023_A711A9B8/metrics/best_validation_metrics.json": "1c289d6e44a295ea4268601c0d9661ec5741b4e72ad6645f1f4f4509ccc9572c",
            "artifacts/runs/RUN_TR_S14_0023_A711A9B8/predictions/best_validation_predictions.csv": "ed64d7d2247f5ee0ce58c906a7b447bf07e7fa21a16ff151dea73d101d35ce89",
            "artifacts/runs/RUN_TR_S19_0034_CF8C1FE8/config.json": "04495890cf68c20d9c073dad0537d14b7967257d9ec4514fff1e55e6bec1cab3",
            "artifacts/runs/RUN_TR_S19_0034_CF8C1FE8/status.json": "0cb6eaf0bb866e7fe0b4a38734c2da25371f6e3e13dea53aa9e7f79f8fcdfb29",
            "artifacts/runs/RUN_TR_S19_0034_CF8C1FE8/training.log": "bfff17e357f30604e728eadb70a5a05e18735d4b35fa9dd3459923c60fcc0885",
            "artifacts/runs/RUN_TR_S19_0034_CF8C1FE8/checkpoints/best_checkpoint.pt": "5bb20249aeaa7cb4511502093d1d168ce105c9d4dc635aa72fc77494017961ec",
            "artifacts/runs/RUN_TR_S19_0034_CF8C1FE8/metrics/best_validation_metrics.json": "89d18a9a336b33e9b743025d4fd5c70ba984f65196d74635f5abe62c0cf9d8c1",
            "artifacts/runs/RUN_TR_S19_0034_CF8C1FE8/predictions/best_validation_predictions.csv": "d462610ee7b60295d469b6c7e24834d51752d9b118bf50656fe2542ba9d1772d",
        },
        "reporting_debt": (
            "artifacts/sweeps/S19_boundary_protocol/s19_boundary_metrics.csv",
            "artifacts/sweeps/S19_boundary_protocol/s19_boundary_winner.json",
            "artifacts/runs/RUN_TR_S19_0034_CF8C1FE8/training_history.csv",
        ),
    },
}


class PhaseState(str, Enum):
    VALID_REUSABLE = "VALID_REUSABLE"
    LOG_MISSING = "LOG_MISSING"
    LOG_STALE = "LOG_STALE"
    DERIVED_ARTIFACT_MISSING = "DERIVED_ARTIFACT_MISSING"
    CONDITION_INCOMPLETE = "CONDITION_INCOMPLETE"
    SIGNOFF_INVALID = "SIGNOFF_INVALID"
    UPSTREAM_INVALID = "UPSTREAM_INVALID"
    ENVIRONMENT_INVALID = "ENVIRONMENT_INVALID"
    RUNNING = "RUNNING"
    FAILED = "FAILED"


class PhaseAction(str, Enum):
    RENDER_ONLY = "RENDER_ONLY"
    REBUILD_LOG_ONLY = "REBUILD_LOG_ONLY"
    REBUILD_DERIVED_ONLY = "REBUILD_DERIVED_ONLY"
    EXECUTE_MISSING_ONLY = "EXECUTE_MISSING_ONLY"
    WAIT_FOR_RUNNING_PROCESS = "WAIT_FOR_RUNNING_PROCESS"
    BLOCK = "BLOCK"


@dataclass(frozen=True)
class SweepPhaseSpec:
    phase_id: int
    phase_name: str
    family_id: str
    sweep_code: str
    artifact_directory: str
    winner_filename: str
    reference_filename: str
    log_filename: str
    prerequisite_paths: tuple[str, ...]
    condition_path: tuple[str, ...]
    condition_values: tuple[tuple[str, Any], ...]
    reference_condition: str
    manifest_filename: str = "sweep_manifest.json"
    results_filename: str = "results.csv"

    @property
    def artifact_root(self) -> Path:
        return Path("artifacts/sweeps") / self.artifact_directory

    @property
    def manifest_path(self) -> Path:
        return self.artifact_root / self.manifest_filename

    @property
    def results_path(self) -> Path:
        return self.artifact_root / self.results_filename

    @property
    def signoff_path(self) -> Path:
        return self.artifact_root / f"phase_{self.phase_id}_signoff.json"

    @property
    def winner_path(self) -> Path:
        return self.artifact_root / self.winner_filename

    @property
    def reference_path(self) -> Path:
        return self.artifact_root / self.reference_filename

    @property
    def processing_log_path(self) -> Path:
        return Path("docs/save_log_in_processing") / self.log_filename

    @property
    def expected_conditions(self) -> tuple[str, ...]:
        return tuple(condition_id for condition_id, _ in self.condition_values)


SWEEP_PHASE_SPECS: dict[int, SweepPhaseSpec] = {
    23: SweepPhaseSpec(23, "S1 Feature-Set Sweep", "S1_FEATURE_SET", "S1", "s1_feature_set", "s1_feature_set_winner.json", "s1_reference_update.json", "phase_23_s1_feature_set_log.json", ("artifacts/learning_diagnostics/phase_22_signoff.json",), ("data", "feature_variant_id"), (("FS0_TF1", "FS0_TF1"), ("FS1_TF1", "FS1_TF1"), ("FS2_TF1", "FS2_TF1")), "FS1_TF1"),
    24: SweepPhaseSpec(24, "S2 Time-Feature Sweep", "S2_TIME_FEATURES", "S2", "s2_time_feature", "s2_time_feature_winner.json", "s2_reference_update.json", "phase_24_s2_time_feature_log.json", ("artifacts/sweeps/s1_feature_set/phase_23_signoff.json", "artifacts/sweeps/s1_feature_set/s1_feature_set_winner.json", "artifacts/sweeps/s1_feature_set/s1_reference_update.json"), ("data", "feature_variant_id"), (("TF0", "FS1_TF0"), ("TF1", "FS1_TF1")), "TF1"),
    25: SweepPhaseSpec(25, "S3 Target-Scaling Sweep", "S3_TARGET_SCALING", "S3", "s3_target_scaling", "s3_target_scaling_winner.json", "s3_reference_update.json", "phase_25_s3_target_scaling_log.json", ("artifacts/sweeps/s2_time_feature/phase_24_signoff.json", "artifacts/sweeps/s2_time_feature/s2_time_feature_winner.json", "artifacts/sweeps/s2_time_feature/s2_reference_update.json"), ("data", "target_scaling_option"), (("YS0", "YS0"), ("YS1", "YS1")), "YS1"),
    26: SweepPhaseSpec(26, "S4 Lookback Sweep", "S4_LOOKBACK", "S4", "s4_lookback", "s4_lookback_winner.json", "s4_reference_update.json", "phase_26_s4_lookback_log.json", ("artifacts/sweeps/s3_target_scaling/phase_25_signoff.json", "artifacts/sweeps/s3_target_scaling/s3_target_scaling_winner.json", "artifacts/sweeps/s3_target_scaling/s3_reference_update.json"), ("data", "lookback_steps"), (("L36", 36), ("L72", 72), ("L144", 144)), "L144"),
    27: SweepPhaseSpec(27, "S5 Pooling Sweep", "S5_POOLING", "S5", "s5_pooling", "s5_pooling_winner.json", "s5_reference_update.json", "phase_27_s5_pooling_log.json", ("artifacts/sweeps/s4_lookback/phase_26_signoff.json", "artifacts/sweeps/s4_lookback/s4_lookback_winner.json", "artifacts/sweeps/s4_lookback/s4_reference_update.json"), ("model", "pooling"), (("LAST_STEP", "LAST_STEP"), ("MEAN", "MEAN")), "LAST_STEP"),
    28: SweepPhaseSpec(28, "S6 Activation Sweep", "S6_ACTIVATION", "S6", "s6_activation", "s6_activation_winner.json", "s6_reference_update.json", "phase_28_s6_activation_log.json", ("artifacts/sweeps/s5_pooling/phase_27_signoff.json", "artifacts/sweeps/s5_pooling/s5_pooling_winner.json", "artifacts/sweeps/s5_pooling/s5_reference_update.json"), ("model", "activation"), (("RELU", "RELU"), ("GELU", "GELU")), "GELU"),
    29: SweepPhaseSpec(29, "S7 Batch-Size Sweep", "S7_BATCH_SIZE", "S7", "s7_batch_size", "s7_batch_winner.json", "s7_reference_update.json", "phase_29_s7_batch_size_log.json", ("artifacts/sweeps/s6_activation/phase_28_signoff.json", "artifacts/sweeps/s6_activation/s6_activation_winner.json", "artifacts/sweeps/s6_activation/s6_reference_update.json"), ("training", "batch_size"), (("B32", 32), ("B64", 64)), "B64"),
    30: SweepPhaseSpec(30, "S8 Learning-Rate Sweep", "S8_LEARNING_RATE", "S8", "s8_learning_rate", "s8_learning_rate_winner.json", "s8_reference_update.json", "phase_30_s8_learning_rate_log.json", ("artifacts/sweeps/s7_batch_size/phase_29_signoff.json", "artifacts/sweeps/s7_batch_size/s7_batch_winner.json", "artifacts/sweeps/s7_batch_size/s7_reference_update.json"), ("training", "learning_rate"), (("LR1", 0.0001), ("LR2", 0.0003), ("LR3", 0.001)), "LR2"),
    31: SweepPhaseSpec(31, "S9 Weight-Decay Sweep", "S9_WEIGHT_DECAY", "S9", "S9_weight_decay", "s9_weight_decay_winner.json", "s9_reference_update.json", "phase_31_s9_weight_decay_log.json", ("artifacts/sweeps/s8_learning_rate/phase_30_signoff.json", "artifacts/sweeps/s8_learning_rate/s8_learning_rate_winner.json", "artifacts/sweeps/s8_learning_rate/s8_reference_update.json"), ("training", "weight_decay"), (("WD0", 0.0), ("WD1", 0.0001), ("WD2", 0.001)), "WD1", "s9_weight_decay_sweep_manifest.json", "s9_weight_decay_metrics.csv"),
    32: SweepPhaseSpec(32, "S10 Dropout Sweep", "S10_DROPOUT", "S10", "S10_dropout", "s10_dropout_winner.json", "s10_reference_update.json", "phase_32_s10_dropout_log.json", ("artifacts/sweeps/S9_weight_decay/phase_31_signoff.json", "artifacts/sweeps/S9_weight_decay/s9_weight_decay_winner.json", "artifacts/sweeps/S9_weight_decay/s9_reference_update.json"), ("model", "dropout"), (("DR01", 0.1), ("DR02", 0.2), ("DR03", 0.3)), "DR01", "s10_dropout_sweep_manifest.json", "s10_dropout_metrics.csv"),
    33: SweepPhaseSpec(33, "S11 d_model Sweep", "S11_D_MODEL", "S11", "S11_d_model", "s11_d_model_winner.json", "s11_reference_update.json", "phase_33_s11_d_model_log.json", ("artifacts/sweeps/S10_dropout/phase_32_signoff.json", "artifacts/sweeps/S10_dropout/s10_dropout_winner.json", "artifacts/sweeps/S10_dropout/s10_reference_update.json"), ("model", "d_model"), (("D32", 32), ("D64", 64)), "D64", "s11_d_model_sweep_manifest.json", "s11_d_model_metrics.csv"),
    34: SweepPhaseSpec(34, "S12 Head Sweep", "S12_HEADS", "S12", "S12_heads", "s12_head_winner.json", "s12_reference_update.json", "phase_34_s12_head_log.json", ("artifacts/sweeps/S11_d_model/phase_33_signoff.json", "artifacts/sweeps/S11_d_model/s11_d_model_winner.json", "artifacts/sweeps/S11_d_model/s11_reference_update.json"), ("model", "num_heads"), (("H2", 2), ("H4", 4)), "H4", "s12_head_sweep_manifest.json", "s12_head_metrics.csv"),
    35: SweepPhaseSpec(35, "S13 Layer Sweep", "S13_LAYERS", "S13", "S13_layers", "s13_layer_winner.json", "s13_reference_update.json", "phase_35_s13_layer_log.json", ("artifacts/sweeps/S12_heads/phase_34_signoff.json", "artifacts/sweeps/S12_heads/s12_head_winner.json", "artifacts/sweeps/S12_heads/s12_reference_update.json"), ("model", "num_layers"), (("N1", 1), ("N2", 2)), "N2", "s13_layer_sweep_manifest.json", "s13_layer_metrics.csv"),
    36: SweepPhaseSpec(36, "S14 FFN Sweep", "S14_FFN", "S14", "S14_ffn", "s14_ffn_winner.json", "s14_reference_update.json", "phase_36_s14_ffn_log.json", ("artifacts/sweeps/S13_layers/phase_35_signoff.json", "artifacts/sweeps/S13_layers/s13_layer_winner.json", "artifacts/sweeps/S13_layers/s13_reference_update.json"), ("model", "ffn_dim"), (("F64", 64), ("F128", 128), ("F256", 256)), "F128", "s14_ffn_sweep_manifest.json", "s14_ffn_metrics.csv"),
    37: SweepPhaseSpec(37, "S15 Loss Sweep", "S15_LOSS", "S15", "S15_loss", "s15_loss_winner.json", "s15_reference_update.json", "phase_37_s15_loss_log.json", ("artifacts/sweeps/S14_ffn/phase_36_signoff.json", "artifacts/sweeps/S14_ffn/s14_ffn_winner.json", "artifacts/sweeps/S14_ffn/s14_reference_update.json"), ("training", "loss_name"), (("L0", "MSE"), ("L1", "HUBER")), "L0", "s15_loss_sweep_manifest.json", "s15_loss_metrics.csv"),
    38: SweepPhaseSpec(38, "S16 Epoch-Cap Sweep", "S16_EPOCH_CAP", "S16", "S16_epoch_cap", "s16_epoch_cap_winner.json", "s16_reference_update.json", "phase_38_s16_epoch_cap_log.json", ("artifacts/sweeps/S15_loss/phase_37_signoff.json", "artifacts/sweeps/S15_loss/s15_loss_winner.json", "artifacts/sweeps/S15_loss/s15_reference_update.json"), ("training", "max_epochs"), (("E50", 50), ("E100", 100)), "E50", "s16_epoch_cap_sweep_manifest.json", "s16_epoch_cap_metrics.csv"),
    39: SweepPhaseSpec(39, "S17 Gradient Clipping Sweep", "S17_GRADIENT_CLIPPING", "S17", "S17_gradient_clipping", "s17_gradient_clip_winner.json", "s17_reference_update.json", "phase_39_s17_gradient_clip_log.json", ("artifacts/sweeps/S16_epoch_cap/phase_38_signoff.json", "artifacts/sweeps/S16_epoch_cap/s16_epoch_cap_winner.json", "artifacts/sweeps/S16_epoch_cap/s16_reference_update.json"), ("training", "gradient_clipping_policy"), (("GC0", "NONE"), ("GC1", "GLOBAL_L2_MAX_NORM_1.0")), "GC1", "s17_gradient_clip_sweep_manifest.json", "s17_gradient_clip_metrics.csv"),
    40: SweepPhaseSpec(40, "S18 RevIN Sweep", "S18_REVIN", "S18", "S18_revin", "s18_revin_winner.json", "s18_reference_update.json", "phase_40_s18_revin_log.json", ("artifacts/sweeps/S17_gradient_clipping/phase_39_signoff.json", "artifacts/sweeps/S17_gradient_clipping/s17_gradient_clip_winner.json", "artifacts/sweeps/S17_gradient_clipping/s17_reference_update.json"), ("model", "revin"), (("RN0", False), ("RN1", True)), "RN0", "s18_revin_sweep_manifest.json", "s18_revin_metrics.csv"),
    41: SweepPhaseSpec(41, "S19 Boundary Protocol Check", "S19_BOUNDARY_PROTOCOL", "S19", "S19_boundary_protocol", "s19_boundary_winner.json", "s19_reference_update.json", "phase_41_s19_boundary_protocol_log.json", ("artifacts/sweeps/S18_revin/phase_40_signoff.json", "artifacts/sweeps/S18_revin/s18_revin_winner.json", "artifacts/sweeps/S18_revin/s18_reference_update.json"), ("data", "window_boundary_protocol"), (("WB0", "WARM_BOUNDS"), ("WB1", "WB1_STRICT_ISOLATION")), "WB0", "s19_boundary_sweep_manifest.json", "s19_boundary_metrics.csv"),
}


STATE_ACTIONS = {
    PhaseState.VALID_REUSABLE: PhaseAction.RENDER_ONLY,
    PhaseState.LOG_MISSING: PhaseAction.REBUILD_LOG_ONLY,
    PhaseState.LOG_STALE: PhaseAction.REBUILD_LOG_ONLY,
    PhaseState.DERIVED_ARTIFACT_MISSING: PhaseAction.REBUILD_DERIVED_ONLY,
    PhaseState.CONDITION_INCOMPLETE: PhaseAction.EXECUTE_MISSING_ONLY,
    PhaseState.SIGNOFF_INVALID: PhaseAction.BLOCK,
    PhaseState.UPSTREAM_INVALID: PhaseAction.BLOCK,
    PhaseState.ENVIRONMENT_INVALID: PhaseAction.BLOCK,
    PhaseState.RUNNING: PhaseAction.WAIT_FOR_RUNNING_PROCESS,
    PhaseState.FAILED: PhaseAction.EXECUTE_MISSING_ONLY,
}


def resolve_phase_action(state: PhaseState | str) -> PhaseAction:
    return STATE_ACTIONS[PhaseState(state)]


def get_sweep_phase_spec(phase_id: int) -> SweepPhaseSpec:
    if phase_id not in SWEEP_PHASE_SPECS:
        raise ValueError(f"Selective sweep execution supports Phase 23-37, received Phase {phase_id}")
    return SWEEP_PHASE_SPECS[phase_id]


def _project_path(root: Path, relative_path: str | Path) -> Path:
    candidate = (root / relative_path).resolve()
    candidate.relative_to(root)
    return candidate


def _json_status(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, "MISSING"
    try:
        value = read_json(path)
    except (OSError, ValueError, TypeError):
        return None, "INVALID_JSON"
    if not isinstance(value, dict):
        return None, "INVALID_OBJECT"
    return value, None


# Backward-compatible resolver for signoff pass status. Older signoffs in
# artifacts/sweeps/*/phase_*_signoff.json were written with three different
# field names across the project history:
#   * ``status``          — canonical, used by Phase 33+ and V2 signoffs
#   * ``overall_status``  — used by Phase 15, 18, 19, 33-41 dashboard signoffs
#   * ``phase_status``    — used by Phase 40, 41 boundary/revin signoffs
# The validator must accept all three schemas to avoid spurious
# SIGNOFF_NOT_PASS complaints when the upstream scientific evidence is intact.
def _signoff_pass_status(record: dict[str, Any]) -> str | None:
    """Return the effective PASS-status of a signoff record.

    Prefers the canonical ``status`` field; falls back to ``overall_status``,
    then ``phase_status``. Phase 40/41 signoffs use ``phase_status="PASS"``
    or ``"COMPLETED"``; both are returned as-is so the upstream gate can
    accept them. Returns ``None`` only when no field carries a non-empty string.
    """
    for field in ("status", "overall_status", "phase_status"):
        value = record.get(field)
        if isinstance(value, str) and value:
            return value
    return None


def _normalize_signoff_status(raw_status: str | None) -> str | None:
    """Normalize heterogeneous signoff status strings to a canonical form.

    Phase 40/41 boundary/revin signoffs write ``phase_status="COMPLETED"``
    to indicate a successful run; this is semantically equivalent to
    ``PASS_WITH_WARNING`` for upstream gate purposes. All other values are
    returned unchanged. ``None`` is returned when the input is missing.
    """
    if not isinstance(raw_status, str) or not raw_status:
        return None
    if raw_status == "COMPLETED":
        return "PASS_WITH_WARNING"
    return raw_status


def _validate_declared_artifacts(root: Path, record: dict[str, Any], path_field: str, checksum_field: str) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    paths = record.get(path_field, [])
    checksums = record.get(checksum_field, {})
    if not isinstance(paths, list) or not isinstance(checksums, dict):
        return [{"path": path_field, "reason": "INVALID_DECLARATION"}]
    for relative_path in paths:
        if not isinstance(relative_path, str):
            issues.append({"path": str(relative_path), "reason": "INVALID_PATH"})
            continue
        try:
            path = _project_path(root, relative_path)
        except (ValueError, OSError):
            issues.append({"path": relative_path, "reason": "PATH_OUTSIDE_PROJECT"})
            continue
        if not path.is_file():
            issues.append({"path": relative_path, "reason": "MISSING"})
            continue
        expected = checksums.get(relative_path)
        if not isinstance(expected, str):
            issues.append({"path": relative_path, "reason": "CHECKSUM_MISSING"})
            continue
        if sha256_file(path) != expected:
            issues.append({"path": relative_path, "reason": "CHECKSUM_MISMATCH"})
    return issues


def _inspect_prerequisites(root: Path, spec: SweepPhaseSpec) -> dict[str, Any]:
    records = []
    valid = True
    for relative_path in spec.prerequisite_paths:
        path = _project_path(root, relative_path)
        value, error = _json_status(path)
        item: dict[str, Any] = {"path": relative_path, "status": "PASS" if error is None else error}
        if error is None and relative_path.endswith("signoff.json"):
            status = _normalize_signoff_status(_signoff_pass_status(value))
            legacy_phase_id = value.get("phase_id", value.get("phase"))
            if legacy_phase_id in LEGACY_SWEEP_CONTRACTS:
                legacy = _validate_legacy_sweep_contract(root, int(legacy_phase_id), value)
                issues = legacy["issues"]
                item["legacy_contract"] = legacy
            else:
                issues = _validate_declared_artifacts(root, value, "output_paths", "output_checksums")
            if status not in {"PASS", "PASS_WITH_WARNING"}:
                item["status"] = "SIGNOFF_NOT_PASS"
            if issues:
                item["status"] = "SIGNOFF_ARTIFACT_INVALID"
                item["issues"] = issues
        if item["status"] != "PASS":
            valid = False
        records.append(item)
    return {"valid": valid, "records": records}


def _inspect_signoff(root: Path, spec: SweepPhaseSpec) -> dict[str, Any]:
    relative_path = str(spec.signoff_path)
    value, error = _json_status(_project_path(root, relative_path))
    if error is not None:
        return {"valid": False, "path": relative_path, "status": error, "issues": []}
    if spec.phase_id in LEGACY_SWEEP_CONTRACTS:
        legacy = _validate_legacy_sweep_contract(root, spec.phase_id, value)
        return {
            "valid": legacy["valid"],
            "path": relative_path,
            "status": _normalize_signoff_status(_signoff_pass_status(value)),
            "issues": legacy["issues"],
            "record": value,
            "legacy_contract": legacy,
            "reporting_debt": legacy["reporting_debt"],
        }
    issues = _validate_declared_artifacts(root, value, "output_paths", "output_checksums")
    status_valid = _normalize_signoff_status(_signoff_pass_status(value)) in {"PASS", "PASS_WITH_WARNING"}
    identity_valid = value.get("phase_id") == spec.phase_id and value.get("phase_version") == f"PHASE-{spec.phase_id}-v1"
    if not status_valid:
        issues.append({"path": relative_path, "reason": "SIGNOFF_NOT_PASS"})
    if not identity_valid:
        issues.append({"path": relative_path, "reason": "SIGNOFF_IDENTITY_MISMATCH"})
    return {"valid": not issues, "path": relative_path, "status": value.get("status"), "issues": issues, "record": value}


def _inspect_required_phase_artifacts(root: Path, spec: SweepPhaseSpec) -> dict[str, Any]:
    relative_paths = (spec.manifest_path, spec.results_path, spec.winner_path, spec.reference_path, spec.signoff_path)
    records = []
    legacy_valid = False
    reporting_debt: set[str] = set()
    if spec.phase_id in LEGACY_SWEEP_CONTRACTS:
        signoff, error = _json_status(_project_path(root, spec.signoff_path))
        if error is None:
            legacy = _validate_legacy_sweep_contract(root, spec.phase_id, signoff)
            legacy_valid = legacy["valid"]
            reporting_debt = set(legacy["reporting_debt"])
    for relative_path in relative_paths:
        path = _project_path(root, relative_path)
        relative_text = str(relative_path)
        if path.is_file():
            status = "PASS"
        elif legacy_valid and relative_text in reporting_debt:
            status = "HISTORICAL_REPORTING_DEBT"
        else:
            status = "MISSING"
        records.append({"path": relative_text, "status": status})
    return {
        "valid": all(item["status"] in {"PASS", "HISTORICAL_REPORTING_DEBT"} for item in records),
        "records": records,
        "reporting_debt": sorted(reporting_debt) if legacy_valid else [],
    }


def _inspect_processing_log(root: Path, spec: SweepPhaseSpec) -> dict[str, Any]:
    relative_path = str(spec.processing_log_path)
    value, error = _json_status(_project_path(root, relative_path))
    if error is not None:
        return {"valid": False, "path": relative_path, "status": error, "issues": []}
    issues = []
    for source in value.get("source_artifacts", []):
        source_path = source.get("path")
        expected = source.get("sha256")
        if not isinstance(source_path, str) or not isinstance(expected, str):
            issues.append({"path": str(source_path), "reason": "INVALID_SOURCE_DECLARATION"})
            continue
        try:
            path = _project_path(root, source_path)
        except (ValueError, OSError):
            issues.append({"path": source_path, "reason": "PATH_OUTSIDE_PROJECT"})
            continue
        if not path.is_file():
            issues.append({"path": source_path, "reason": "MISSING"})
        elif sha256_file(path) != expected:
            issues.append({"path": source_path, "reason": "CHECKSUM_MISMATCH"})
    identity_valid = value.get("phase_id") == spec.phase_id
    if not identity_valid:
        issues.append({"path": relative_path, "reason": "LOG_IDENTITY_MISMATCH"})
    return {"valid": not issues, "path": relative_path, "status": value.get("status"), "issues": issues, "record": value}


def _nested_value(value: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _condition_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
    return actual == expected


def _load_registry_records(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    path = root / "artifacts/experiments/experiment_registry.jsonl"
    if not path.is_file():
        return [], [{"path": str(path.relative_to(root)), "reason": "MISSING"}]
    records = []
    issues = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            issues.append({"path": str(path.relative_to(root)), "reason": f"INVALID_JSON_LINE_{line_number}"})
            continue
        if isinstance(value, dict):
            records.append(value)
        else:
            issues.append({"path": str(path.relative_to(root)), "reason": f"INVALID_RECORD_LINE_{line_number}"})
    return records, issues


def _validate_legacy_sweep_contract(
    root: Path,
    phase_id: int,
    signoff: dict[str, Any],
) -> dict[str, Any]:
    """Validate one explicitly allow-listed historical sweep contract.

    This is not a generic legacy-schema fallback.  Acceptance requires the
    exact historical signoff, exact commit-bound sweep files, the explicitly
    listed canonical COMPLETED registry records, and exact historical SHA-256
    values for every retained scientific run artifact (including Validation
    predictions).
    """
    contract = LEGACY_SWEEP_CONTRACTS.get(phase_id)
    if contract is None:
        return {
            "valid": False,
            "phase_id": phase_id,
            "issues": [{"path": f"phase_{phase_id}", "reason": "LEGACY_CONTRACT_NOT_ALLOW_LISTED"}],
            "reporting_debt": [],
        }

    issues: list[dict[str, str]] = []
    signoff_paths = {
        37: "artifacts/sweeps/S15_loss/phase_37_signoff.json",
        38: "artifacts/sweeps/S16_epoch_cap/phase_38_signoff.json",
        39: "artifacts/sweeps/S17_gradient_clipping/phase_39_signoff.json",
        40: "artifacts/sweeps/S18_revin/phase_40_signoff.json",
        41: "artifacts/sweeps/S19_boundary_protocol/phase_41_signoff.json",
    }
    signoff_path = root / signoff_paths[phase_id]
    signoff_relative = str(signoff_path.relative_to(root))
    if not signoff_path.is_file():
        issues.append({"path": signoff_relative, "reason": "MISSING"})
    elif sha256_file(signoff_path) != contract["signoff_sha256"]:
        issues.append({"path": signoff_relative, "reason": "LEGACY_SIGNOFF_CHECKSUM_MISMATCH"})

    for field, expected in contract["identity"].items():
        if signoff.get(field) != expected:
            issues.append({"path": signoff_relative, "reason": f"LEGACY_SIGNOFF_{field.upper()}_MISMATCH"})
    if phase_id == 38 and signoff.get("output_checksums") != {}:
        issues.append({"path": signoff_relative, "reason": "LEGACY_EMPTY_CHECKSUM_LEDGER_MISMATCH"})

    for relative_path, expected_sha256 in contract["historical_files"].items():
        path = _project_path(root, relative_path)
        if not path.is_file():
            issues.append({"path": relative_path, "reason": "MISSING"})
        elif sha256_file(path) != expected_sha256:
            issues.append({"path": relative_path, "reason": "HISTORICAL_CHECKSUM_MISMATCH"})

    records, registry_issues = _load_registry_records(root)
    issues.extend(registry_issues)
    run_id = contract["run_id"]
    canonical_runs = contract.get(
        "canonical_runs",
        ((run_id, contract["experiment_family"]),),
    )
    declared: dict[str, Any] = {}
    for canonical_run_id, experiment_family in canonical_runs:
        matches = [record for record in records if record.get("run_id") == canonical_run_id]
        if len(matches) != 1:
            issues.append(
                {"path": canonical_run_id, "reason": "CANONICAL_REGISTRY_RECORD_COUNT_MISMATCH"}
            )
            continue
        record = matches[0]
        if record.get("status") != "COMPLETED":
            issues.append({"path": canonical_run_id, "reason": "CANONICAL_RUN_NOT_COMPLETED"})
        if record.get("experiment_family") != experiment_family:
            issues.append(
                {"path": canonical_run_id, "reason": "CANONICAL_EXPERIMENT_FAMILY_MISMATCH"}
            )
        if record.get("test_access_authorized") is not False:
            issues.append({"path": canonical_run_id, "reason": "TEST_FIREWALL_NOT_VERIFIED"})
        declared.update(
            {
                item.get("artifact_path"): item.get("sha256")
                for item in record.get("artifacts", [])
                if isinstance(item, dict)
            }
        )

    for relative_path, expected_sha256 in contract["core_artifacts"].items():
        if declared.get(relative_path) != expected_sha256:
            issues.append({"path": relative_path, "reason": "REGISTRY_CHECKSUM_MISMATCH"})
            continue
        path = _project_path(root, relative_path)
        if not path.is_file():
            issues.append({"path": relative_path, "reason": "MISSING_CANONICAL_CORE_ARTIFACT"})
        elif sha256_file(path) != expected_sha256:
            issues.append({"path": relative_path, "reason": "CHECKSUM_MISMATCH"})

    reporting_debt = [
        relative_path
        for relative_path in contract["reporting_debt"]
        if not _project_path(root, relative_path).is_file()
    ]
    return {
        "valid": not issues,
        "phase_id": phase_id,
        "policy": f"PHASE_{phase_id}_COMMIT_BOUND_LEGACY-v1",
        "historical_commit": "1f648662980c49e0ee42de9f2cf43639e6966b64",
        "signoff_sha256": contract["signoff_sha256"],
        "canonical_run_id": run_id,
        "canonical_run_ids": [item[0] for item in canonical_runs],
        "issues": issues,
        "reporting_debt": reporting_debt,
        "test_status": "NOT_ACCESSED",
    }


def _validate_run_artifacts(
    root: Path,
    record: dict[str, Any],
    allowed_missing_paths: set[str] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    issues = []
    missing_artifacts = []
    allowed_missing_paths = allowed_missing_paths or set()
    required_artifacts = [item for item in record.get("artifacts", []) if item.get("required") is True]
    required_types = {item.get("artifact_type") for item in required_artifacts}
    for required_type in ("CONFIG", "STATUS", "METRICS"):
        if required_type not in required_types:
            issues.append({"path": str(record.get("run_id")), "reason": f"REQUIRED_{required_type}_MISSING"})
    for artifact in required_artifacts:
        relative_path = artifact.get("artifact_path")
        expected_checksum = artifact.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(expected_checksum, str):
            issues.append({"path": str(relative_path), "reason": "INVALID_ARTIFACT_DECLARATION"})
            continue
        try:
            path = _project_path(root, relative_path)
        except (ValueError, OSError):
            issues.append({"path": relative_path, "reason": "PATH_OUTSIDE_PROJECT"})
            continue
        if not path.is_file():
            if relative_path in allowed_missing_paths:
                missing_artifacts.append(
                    {
                        "artifact_path": relative_path,
                        "expected_sha256": expected_checksum,
                        "retention_status": "MISSING_UNRECOVERABLE",
                    }
                )
            else:
                issues.append({"path": relative_path, "reason": "MISSING"})
        elif True:
            try:
                actual = sha256_file(path)
            except PermissionError:
                actual = None
            if actual != expected_checksum:
                issues.append({"path": relative_path, "reason": "CHECKSUM_MISMATCH"})
    return issues, missing_artifacts


def _validation_metrics(record: dict[str, Any]) -> dict[str, float] | None:
    values = {}
    for metric in record.get("metrics", []):
        if metric.get("split_id") != "VALIDATION" or metric.get("status") != "PASS":
            continue
        name = metric.get("metric_name")
        value = metric.get("metric_value")
        if name in {"mae_wh", "rmse_wh", "r2"} and isinstance(value, (int, float)) and math.isfinite(float(value)):
            values[name] = float(value)
    if set(values) != {"mae_wh", "rmse_wh", "r2"}:
        return None
    return values


def _load_run_config(root: Path, run_id: str) -> tuple[dict[str, Any] | None, str | None]:
    payload, error = _json_status(root / "artifacts/runs" / run_id / "config.json")
    if error is not None:
        return None, error
    if payload.get("run_id") != run_id or not isinstance(payload.get("config"), dict):
        return None, "CONFIG_IDENTITY_MISMATCH"
    return payload, None


def _reference_run_id(root: Path, spec: SweepPhaseSpec) -> str | None:
    if spec.phase_id == 23:
        return None
    for relative_path in reversed(spec.prerequisite_paths):
        if not relative_path.endswith(".json") or relative_path.endswith("signoff.json"):
            continue
        value, error = _json_status(_project_path(root, relative_path))
        if error is not None:
            continue
        for key in ("winner_run_id", "reference_run_id", "current_reference_run_id", "source_winner_run_id"):
            run_id = value.get(key)
            if isinstance(run_id, str) and run_id:
                return run_id
    return None


def resolve_condition_values(phase_id: int, project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    spec = get_sweep_phase_spec(phase_id)
    values = dict(spec.condition_values)
    if phase_id != 24:
        return values
    reference_run_id = _reference_run_id(root, spec)
    if reference_run_id is None:
        return values
    payload, error = _load_run_config(root, reference_run_id)
    if error is not None:
        return values
    reference_variant = _nested_value(payload["config"], ("data", "feature_variant_id"))
    if not isinstance(reference_variant, str) or "_TF" not in reference_variant:
        return values
    feature_set = reference_variant.rsplit("_TF", 1)[0]
    return {"TF0": f"{feature_set}_TF0", "TF1": f"{feature_set}_TF1"}


def resolve_phase_conditions(phase_id: int, project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    spec = get_sweep_phase_spec(phase_id)
    contract = LEGACY_SWEEP_CONTRACTS.get(phase_id)
    if contract is not None and contract.get("condition_evidence"):
        signoff, signoff_error = _json_status(_project_path(root, spec.signoff_path))
        if signoff_error is None:
            legacy = _validate_legacy_sweep_contract(root, phase_id, signoff)
            if legacy["valid"]:
                records, registry_issues = _load_registry_records(root)
                records_by_id = {record.get("run_id"): record for record in records}
                verified = []
                for condition_id, factor_value, run_id, reused_reference in contract["condition_evidence"]:
                    metrics = _validation_metrics(records_by_id[run_id])
                    if metrics is None:
                        break
                    verified.append(
                        {
                            "condition_id": condition_id,
                            "factor_value": factor_value,
                            "run_id": run_id,
                            "rmse_wh": metrics["rmse_wh"],
                            "mae_wh": metrics["mae_wh"],
                            "r2": metrics["r2"],
                            "reused_reference": reused_reference,
                            "evidence_mode": "COMMIT_AND_REGISTRY_BOUND_LEGACY_EVIDENCE",
                            "evidence_status": "PASS_WITH_WARNING",
                            "warnings": ["HISTORICAL_REPORTING_DEBT"],
                            "missing_artifacts": [
                                {
                                    "artifact_path": path,
                                    "expected_sha256": None,
                                    "retention_status": "HISTORICAL_REPORTING_DEBT",
                                }
                                for path in legacy["reporting_debt"]
                            ],
                        }
                    )
                if len(verified) == len(spec.expected_conditions):
                    return {
                        "expected_conditions": list(spec.expected_conditions),
                        "verified_conditions": verified,
                        "missing_conditions": [],
                        "invalid_conditions": [],
                        "running_conditions": [],
                        "failed_conditions": [],
                        "reference_condition": spec.reference_condition,
                        "reference_run_id": next(
                            item["run_id"] for item in verified if item["reused_reference"]
                        ),
                        "registry_issues": registry_issues,
                        "complete": not registry_issues,
                        "legacy_contract": legacy,
                    }
    records, registry_issues = _load_registry_records(root)
    reference_run_id = _reference_run_id(root, spec)
    verified = []
    invalid = []
    running = []
    failed = []
    condition_values = resolve_condition_values(phase_id, root)
    for condition_id in spec.expected_conditions:
        expected_value = condition_values[condition_id]
        candidates = []
        condition_invalid = []
        condition_running = []
        condition_failed = []
        for record in records:
            run_id = record.get("run_id")
            if not isinstance(run_id, str):
                continue
            config_payload, config_error = _load_run_config(root, run_id)
            if config_error is not None:
                continue
            actual_value = _nested_value(config_payload["config"], spec.condition_path)
            if not _condition_matches(actual_value, expected_value):
                continue
            is_reference = condition_id == spec.reference_condition
            family_matches = record.get("experiment_family") == spec.family_id
            reference_matches = is_reference and reference_run_id is not None and run_id == reference_run_id
            phase_23_reference = phase_id == 23 and is_reference and record.get("experiment_family") == "TRANSFORMER_BASELINE"
            if not family_matches and not reference_matches and not phase_23_reference:
                continue
            candidates.append((record, config_payload))
        valid_candidates = []
        for record, config_payload in candidates:
            run_id = record["run_id"]
            status = record.get("status")
            historical_h4 = phase_id in {34, 35, 36} and condition_id == spec.reference_condition
            historical_paths = {
                f"artifacts/runs/{run_id}/checkpoints/best_checkpoint.pt",
                f"artifacts/runs/{run_id}/training.log",
                f"artifacts/runs/{run_id}/predictions/best_validation_predictions.csv",
            } if historical_h4 else set()
            evidence_issues, missing_artifacts = _validate_run_artifacts(
                root,
                record,
                historical_paths,
            )
            if historical_h4:
                history_path = f"artifacts/runs/{run_id}/training_history.csv"
                artifact_checksums = {
                    item.get("artifact_path"): item.get("sha256")
                    for item in record.get("artifacts", [])
                    if isinstance(item.get("artifact_path"), str)
                }
                expected_missing = {
                    **{path: artifact_checksums.get(path) for path in historical_paths},
                    history_path: "635e00dfcd032c3664c5e031b13f38d94caa3f73c4c6bf142eb0d6883d9eb43b",
                }
                known_missing = {item["artifact_path"] for item in missing_artifacts}
                for relative_path, expected_sha256 in expected_missing.items():
                    path = root / relative_path
                    if path.is_file():
                        if not isinstance(expected_sha256, str):
                            evidence_issues.append(
                                {"path": relative_path, "reason": "HISTORICAL_CHECKSUM_MISSING"}
                            )
                        elif sha256_file(path) != expected_sha256:
                            evidence_issues.append(
                                {"path": relative_path, "reason": "CHECKSUM_MISMATCH"}
                            )
                    elif relative_path not in known_missing:
                        missing_artifacts.append(
                            {
                                "artifact_path": relative_path,
                                "expected_sha256": expected_sha256,
                                "retention_status": "MISSING_UNRECOVERABLE",
                            }
                        )
            metrics = _validation_metrics(record)
            config_fingerprint_matches = config_payload.get("config_fingerprint") == record.get("config_fingerprint")
            if status == "RUNNING":
                condition_running.append({"condition_id": condition_id, "run_id": run_id})
                continue
            if status == "FAILED":
                condition_failed.append({"condition_id": condition_id, "run_id": run_id})
                continue
            if status != "COMPLETED":
                continue
            if evidence_issues:
                condition_invalid.append({"condition_id": condition_id, "run_id": run_id, "reasons": evidence_issues})
                continue
            if metrics is None:
                condition_invalid.append({"condition_id": condition_id, "run_id": run_id, "reasons": [{"path": run_id, "reason": "VALIDATION_METRICS_INCOMPLETE"}]})
                continue
            if not config_fingerprint_matches:
                condition_invalid.append({"condition_id": condition_id, "run_id": run_id, "reasons": [{"path": run_id, "reason": "CONFIG_FINGERPRINT_MISMATCH"}]})
                continue
            valid_candidates.append((record, metrics, expected_value, missing_artifacts))
        if len(valid_candidates) == 1:
            record, metrics, factor_value, missing_artifacts = valid_candidates[0]
            historical_h4 = phase_id in {34, 35, 36} and condition_id == spec.reference_condition
            verified.append(
                {
                    "condition_id": condition_id,
                    "factor_value": factor_value,
                    "run_id": record["run_id"],
                    "rmse_wh": metrics["rmse_wh"],
                    "mae_wh": metrics["mae_wh"],
                    "r2": metrics["r2"],
                    "reused_reference": condition_id == spec.reference_condition,
                    "evidence_mode": (
                        "HISTORICAL_REFERENCE_WITH_INCOMPLETE_ARTIFACT_RETENTION"
                        if historical_h4
                        else "COMPLETE_RUN_ARTIFACTS"
                    ),
                    "evidence_status": "PASS_WITH_WARNING" if historical_h4 else "PASS",
                    "warnings": ["H4_SOURCE_ARTIFACT_RETENTION_INCOMPLETE"] if historical_h4 else [],
                    "missing_artifacts": missing_artifacts,
                }
            )
        elif len(valid_candidates) > 1:
            invalid.append({"condition_id": condition_id, "run_id": None, "reasons": [{"path": condition_id, "reason": "MULTIPLE_VALID_RUNS"}]})
        else:
            invalid.extend(condition_invalid)
            running.extend(condition_running)
            failed.extend(condition_failed)
    verified_ids = {item["condition_id"] for item in verified}
    missing = [condition_id for condition_id in spec.expected_conditions if condition_id not in verified_ids]
    return {
        "expected_conditions": list(spec.expected_conditions),
        "verified_conditions": verified,
        "missing_conditions": missing,
        "invalid_conditions": invalid,
        "running_conditions": running,
        "failed_conditions": failed,
        "reference_condition": spec.reference_condition,
        "reference_run_id": reference_run_id,
        "registry_issues": registry_issues,
        "complete": not missing and not invalid and not registry_issues,
    }


def inspect_execution_readiness(phase_id: int, project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    spec = get_sweep_phase_spec(phase_id)
    prerequisites = _inspect_prerequisites(root, spec)
    signoff_path = root / "artifacts/environment/phase_1_signoff.json"
    report_path = root / "artifacts/environment/environment_report.json"
    signoff, signoff_error = _json_status(signoff_path)
    report, report_error = _json_status(report_path)
    environment_issues = []
    if signoff_error is not None:
        environment_issues.append({"path": str(signoff_path.relative_to(root)), "reason": signoff_error})
    elif signoff.get("status") != "PASS":
        environment_issues.append({"path": str(signoff_path.relative_to(root)), "reason": "SIGNOFF_NOT_PASS"})
    else:
        environment_issues.extend(_validate_declared_artifacts(root, signoff, "output_paths", "output_checksums"))
    if report_error is not None:
        environment_issues.append({"path": str(report_path.relative_to(root)), "reason": report_error})
    else:
        if report.get("python_version") != platform.python_version():
            environment_issues.append({"path": str(report_path.relative_to(root)), "reason": "PYTHON_VERSION_MISMATCH"})
        signed_executable = report.get("python_executable")
        if not isinstance(signed_executable, str) or Path(signed_executable).resolve() != Path(sys.executable).resolve():
            environment_issues.append({"path": str(report_path.relative_to(root)), "reason": "PYTHON_EXECUTABLE_MISMATCH"})
    if not prerequisites["valid"]:
        state = PhaseState.UPSTREAM_INVALID
    elif environment_issues:
        state = PhaseState.ENVIRONMENT_INVALID
    else:
        state = PhaseState.VALID_REUSABLE
    return {
        "ready": state is PhaseState.VALID_REUSABLE,
        "state": state.value,
        "prerequisites": prerequisites,
        "environment": {
            "valid": not environment_issues,
            "issues": environment_issues,
            "current_python_version": platform.python_version(),
            "current_python_executable": str(Path(sys.executable).resolve()),
        },
    }


def validate_condition_request(phase_id: int, condition_id: str, project_root: Path | None = None) -> dict[str, Any]:
    spec = get_sweep_phase_spec(phase_id)
    if condition_id not in spec.expected_conditions:
        raise ValueError(f"Condition {condition_id} is not registered for Phase {phase_id}")
    readiness = inspect_execution_readiness(phase_id, project_root)
    conditions = resolve_phase_conditions(phase_id, project_root)
    already_verified = condition_id in {item["condition_id"] for item in conditions["verified_conditions"]}
    already_running = condition_id in {item["condition_id"] for item in conditions["running_conditions"]}
    return {
        "phase_id": phase_id,
        "condition_id": condition_id,
        "allowed": readiness["ready"] and not already_verified and not already_running,
        "readiness": readiness,
        "already_verified": already_verified,
        "already_running": already_running,
    }


def _block_reasons(inspection: dict[str, Any], readiness: dict[str, Any]) -> list[str]:
    reasons = []
    for record in inspection["prerequisites"]["records"]:
        if record["status"] == "PASS":
            continue
        issues = record.get("issues", [])
        if issues:
            reasons.extend(f"{issue['path']}: {issue['reason']}" for issue in issues)
        else:
            reasons.append(f"{record['path']}: {record['status']}")
    for issue in inspection["signoff"].get("issues", []):
        reasons.append(f"{issue['path']}: {issue['reason']}")
    for issue in inspection["conditions"].get("registry_issues", []):
        reasons.append(f"{issue['path']}: {issue['reason']}")
    for issue in readiness["environment"].get("issues", []):
        reasons.append(f"{issue['path']}: {issue['reason']}")
    return list(dict.fromkeys(reasons))


def plan_phase_resume(phase_id: int, project_root: Path | None = None, allow_execution: bool = False) -> dict[str, Any]:
    inspection = inspect_phase_state(phase_id, project_root)
    resolved_action = PhaseAction(inspection["action"])
    readiness = inspect_execution_readiness(phase_id, project_root)
    effective_action = resolved_action
    reasons = _block_reasons(inspection, readiness) if resolved_action is PhaseAction.BLOCK else []
    if resolved_action is PhaseAction.EXECUTE_MISSING_ONLY:
        if not readiness["ready"]:
            effective_action = PhaseAction.BLOCK
            reasons.append(readiness["state"])
        elif not allow_execution:
            effective_action = PhaseAction.BLOCK
            reasons.append("SCIENTIFIC_EXECUTION_NOT_AUTHORIZED")
    return {
        "phase_id": phase_id,
        "state": inspection["state"],
        "resolved_action": resolved_action.value,
        "effective_action": effective_action.value,
        "execution_authorized": allow_execution,
        "reasons": reasons,
        "inspection": inspection,
        "readiness": readiness,
    }


def inspect_phase_state(phase_id: int, project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    spec = get_sweep_phase_spec(phase_id)
    prerequisites = _inspect_prerequisites(root, spec)
    signoff = _inspect_signoff(root, spec)
    artifacts = _inspect_required_phase_artifacts(root, spec)
    processing_log = _inspect_processing_log(root, spec)
    conditions = resolve_phase_conditions(phase_id, root)
    if not prerequisites["valid"]:
        state = PhaseState.UPSTREAM_INVALID
    elif conditions["running_conditions"]:
        state = PhaseState.RUNNING
    elif conditions["missing_conditions"] and conditions["failed_conditions"]:
        state = PhaseState.FAILED
    elif conditions["missing_conditions"]:
        state = PhaseState.CONDITION_INCOMPLETE
    elif signoff["status"] != "MISSING" and not signoff["valid"]:
        state = PhaseState.SIGNOFF_INVALID
    elif not signoff["valid"]:
        state = PhaseState.DERIVED_ARTIFACT_MISSING
    elif not artifacts["valid"]:
        state = PhaseState.DERIVED_ARTIFACT_MISSING
    elif processing_log["status"] == "MISSING":
        state = PhaseState.LOG_MISSING
    elif not processing_log["valid"]:
        state = PhaseState.LOG_STALE
    else:
        state = PhaseState.VALID_REUSABLE
    action = resolve_phase_action(state)
    return {
        "phase_id": phase_id,
        "phase_name": spec.phase_name,
        "state": state.value,
        "action": action.value,
        "artifact_root": str(spec.artifact_root),
        "prerequisites": prerequisites,
        "signoff": signoff,
        "artifacts": artifacts,
        "processing_log": processing_log,
        "conditions": conditions,
    }
