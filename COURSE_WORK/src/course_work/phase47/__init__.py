"""Phase 47 — Final Test Evaluation.

Evaluation-only infrastructure for Phase 47: first authorized Test evaluation of
the three FINAL_REFIT Transformer seeds, Persistence baseline, and optionally the
frozen LSTM_TUNED_DEV baseline.

Hard Rules (enforced throughout):
    - NO training
    - NO optimizer.step()
    - NO backward()
    - NO scaler fitting on Test
    - NO checkpoint writes
    - NO Phase46/45 artifact modification
    - Test access ONLY after Phase46 release verified

Pipeline:
    1. Verify Phase46 release gates.
    2. Freeze evaluation contract.
    3. Authorize Test access (record FIRST_ACCESS_EVENT).
    4. Materialize FINAL_TEST_POP-v1 (L72, WB0, H1, 2961 windows).
    5. Load 3 FINAL_REFIT checkpoints (strict load).
    6. Load FINAL_SCALING-v1 (transform-only, never fit).
    7. Per-seed Test inference (eval + inference_mode).
    8. Persistence baseline inference.
    9. LSTM_TUNED_DEV eligibility gate + inference if eligible.
    10. Per-seed metrics (MAE/RMSE/R²).
    11. Aggregate: mean ± sample SD across 3 seeds.
    12. Baseline deltas.
    13. Write all O47 artifacts.
    14. Sign off.
"""

from __future__ import annotations

from pathlib import Path

# Re-export version for consumers
PHASE_VERSION = "PHASE_47_FINAL_TEST_EVALUATION"
PHASE_NUM = 47
OUTPUT_VERSION = "FINAL_TEST_EVAL-v1"

# Hard-locked configuration (from Phase46 handoff)
LOCKED_CANDIDATE = "TR_C2_ALT_LOOKBACK"
LOCKED_LOOKBACK = 72
LOCKED_FEATURES = 33
LOCKED_HORIZON = 1
LOCKED_EPOCHS = 30
LOCKED_BOUNDARY_PROTOCOL = "WB0_CONTEXT_CARRY_OVER"
LOCKED_FEATURE_VARIANT = "FS2_TF1"
LOCKED_TARGET_SCALING = "YS1"
LOCKED_SEEDS = [42, 123, 2026]
LOCKED_CONFIG_FP = "585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24"

# Canonical Phase46 checkpoint metadata
OFFICIAL_RUNS = {
    42: {
        "run_id": "RUN_TR_FSD_0254_2B11AC68",
        "checkpoint_path": "artifacts/three_seed_final_runs/official_checkpoints/seed_42/seed_42_FINAL_REFIT.pt",
        "checkpoint_sha256": "c3cfad116aa91d47fb2f2950407498901b980a344a99462776e4b90828805b10",
        "seed": 42,
    },
    123: {
        "run_id": "RUN_TR_FSD_0254_3858DDA9",
        "checkpoint_path": "artifacts/three_seed_final_runs/official_checkpoints/seed_123/seed_123_FINAL_REFIT.pt",
        "checkpoint_sha256": "8a134fec517be0dfcd81ac6f961bf75fa69e8d34666205f731d893969bf01e22",
        "seed": 123,
    },
    2026: {
        "run_id": "RUN_TR_FSD_0255_C7E123FB",
        "checkpoint_path": "artifacts/three_seed_final_runs/official_checkpoints/seed_2026/seed_2026_FINAL_REFIT.pt",
        "checkpoint_sha256": "8753800539f7a617bacbaa541daafbbfaff338a1de38ae1a5146650b4593dc6f",
        "seed": 2026,
    },
}

# FINAL_SCALING-v1 checksums (from Phase46 handoff)
FINAL_SCALING = {
    "x_scaler_sha256": "7280c166232ac53ef5947fa1991a1b38e9f5ec75045711b7092ddba9c53a17fd",
    "y_scaler_sha256": "e8c8edb970591afa5c25faf619d2257b544b1b27c0b725b3be376a52a5946cca",
    "x_bundle_id": "XSCALER__FS2_TF1",
    "y_bundle_id": "YSCALER__YS1",
}

# FINAL_DEV_POP fingerprint (TRAIN+VAL, L72, WB0)
FINAL_DEV_POP_FP = "0a904beeec68f245fbb1214f7968679d76ebc6dfc010db5215f677ddc2d31f39"
FINAL_DEV_WINDOW_COUNT = 16630

# LSTM baseline metadata (Phase43 LSTM_TUNED_DEV)
LSTM_TUNED_DEV = {
    "model_id": "LSTM_TUNED_DEV",
    "source_phase": 43,
    "run_id": "RUN_LS_LST_0175_BCE5A2CD",
    "checkpoint_path": "artifacts/runs/RUN_LS_LST_0175_BCE5A2CD/checkpoints/best_checkpoint.pt",
    "config_fingerprint": "bce5a2cd6ba86435b7c02a1f1a9d25a6e214493dbd8d6da22886e328287f8593",
    "test_status": "NOT_ACCESSED",
    "lookback_steps": 36,  # LSTM used L36, NOT L72!
    "target_scaling": "YS1",
    "feature_variant": "FS2_TF1",
    # LSTM was trained with Phase9 scalers (SCALING-v1), NOT FINAL_SCALING-v1
    # This is a known fairness caveat per Phase47 plan §61, §143
}

# Phase47-specific authorized split
PHASE_47_SPLIT = "TEST"

# LSTM eligibility statuses
LSTM_ELIGIBILITY_STATUSES = {
    "ELIGIBLE_FROZEN_DEV_BASELINE": "Eligible — frozen Phase43 checkpoint with pre-Test provenance",
    "NOT_ELIGIBLE_CHECKPOINT_MISSING": "Ineligible — checkpoint not found",
    "NOT_ELIGIBLE_CONFIG_MISMATCH": "Ineligible — config/lookback/scaler mismatch with final lock",
    "NOT_ELIGIBLE_POPULATION_MISMATCH": "Ineligible — incompatible FINAL_TEST_POP-v1 population",
    "NOT_EVALUATED_BY_PROTOCOL": "Not evaluated — protocol does not require LSTM evaluation",
}
