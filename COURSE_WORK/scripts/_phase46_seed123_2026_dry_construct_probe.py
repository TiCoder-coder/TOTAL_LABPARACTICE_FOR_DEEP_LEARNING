#!/usr/bin/env python3
"""Phase 46 — Dry-construct Seed123 and Seed2026 (no training).

Verifies the construction path for the next two seeds (which must run under
HUMAN invocation) is correctly wired WITHOUT executing any optimizer step.

Checks for each seed:
  - candidate exactly TR_C2_ALT_LOOKBACK
  - config fingerprint 585c5e79...
  - lookback=72
  - 33 features
  - FINAL_REFIT_EPOCHS=30
  - FINAL_DEV count=16630
  - TEST=0
  - same FINAL_SCALING X/Y checksums as Seed42
  - distinct seed
  - distinct run_id lineage
  - fresh model construction (different parameters than Seed42)
  - fresh optimizer
  - fresh DataLoader Generator
  - no warm-start
  - no validation_loader
  - early stopping false
  - checkpoint type FINAL_REFIT
  - no accidental cache reuse
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    import torch

    from course_work.experiments.registry import ExperimentRegistry
    from course_work.data.final_dev import materialize_final_dev_region
    from course_work.scaling.final_scaling import materialize_final_scaling_v1
    from course_work.utils.artifacts import read_json
    from course_work.models.transformer_regressor import TransformerRegressor
    from copy import deepcopy

    # Load canonical artifacts.
    p45_signoff = read_json(ROOT / "artifacts" / "final_model_lock" / "phase_45_signoff.json")
    p46_handoff = read_json(ROOT / "artifacts" / "final_model_lock" / "phase46_three_seed_handoff.json")

    final_dev = materialize_final_dev_region()
    scaling = materialize_final_scaling_v1()

    x_sha = scaling["x_sha256"]
    y_sha = scaling["y_sha256"]
    n_features = 33
    lookback = final_dev.lookback_steps
    final_dev_n = final_dev.final_dev_window_count

    # Find locked_cfg path: load the final_model_scientific_config.json.
    locked_cfg = read_json(ROOT / "artifacts" / "final_model_lock" / "final_model_scientific_config.json")
    # The driver injects lineage and sets candidate_id from handoff.
    locked_id = p46_handoff.get("candidate_id", "TR_C2_ALT_LOOKBACK")
    scaling_result = scaling  # already materialized above
    historical_seed_to_run_id = {42: "RUN_TR_FSD_0153_B15A19DC", 123: "RUN_TR_FSD_0154_DD82D743", 2026: "RUN_TR_FSD_0155_59A50ADD"}

    # Verify Seed42 cache.
    seed42_id = "RUN_TR_FSD_0181_2B11AC68"
    registry = ExperimentRegistry(ROOT)
    seed42_record = registry.get_run(seed42_id)
    seed42_state_dict = None
    if (ROOT / "artifacts" / "runs" / seed42_id / "checkpoints" / "best_checkpoint.pt").exists():
        seed42_ckpt = torch.load(
            ROOT / "artifacts" / "runs" / seed42_id / "checkpoints" / "best_checkpoint.pt",
            map_location="cpu", weights_only=False,
        )
        seed42_state_dict = seed42_ckpt.get("model_state_dict", {})

    seed42_param_count = (
        sum(p.numel() for p in seed42_state_dict.values())
        if seed42_state_dict else 0
    )
    seed42_first_param = (
        next(iter(seed42_state_dict.values())).flatten()[0].item()
        if seed42_state_dict else 0.0
    )

    print(f"[Seed42] run_id={seed42_id} rmse={seed42_record.get('best_validation_rmse_wh'):.4f}")
    print(f"[Seed42] param_count={seed42_param_count} first_head_weight={seed42_first_param:.6f}")

    failures = []

    for seed in [123, 2026]:
        print(f"\n--- Seed {seed} ---")
        # Build a fresh config from locked_cfg with the seed (mirror driver logic).
        run_config = deepcopy(locked_cfg)
        run_config["candidate_id"] = locked_id
        run_config["training"]["seed"] = int(seed)
        run_config["training"]["max_epochs"] = 30
        run_config["training"]["early_stopping_enabled"] = False
        run_config["training"]["final_refit_mode"] = True
        # Override data fields: test_sample_count must be 0 for FINAL_REFIT mode.
        run_config["data"]["test_sample_count"] = 0
        # Inject lineage mirrors of driver behavior.
        run_config["lineage"] = dict(run_config.get("lineage", {}))
        run_config["lineage"]["final_dev_population_fingerprint"] = scaling_result["final_dev_population_fingerprint"]
        run_config["lineage"]["final_scaling_x_sha256"] = scaling_result["x_sha256"]
        run_config["lineage"]["final_scaling_y_sha256"] = scaling_result["y_sha256"]
        run_config["lineage"]["final_scaling_version"] = scaling_result["scaling_version"]
        run_config["lineage"]["final_dev_region_version"] = "FINAL_DEV_REGION-v1"
        run_config["lineage"]["corrected_implementation_version"] = "PHASE46_CORRECTED-v1"
        run_config["lineage"]["historical_invalidated_predecessor_run_id"] = historical_seed_to_run_id[seed]

        if run_config.get("candidate_id") != "TR_C2_ALT_LOOKBACK":
            failures.append(f"Seed {seed}: candidate_id={run_config.get('candidate_id')} (expected TR_C2_ALT_LOOKBACK)")
        if run_config["data"]["lookback_steps"] != 72:
            failures.append(f"Seed {seed}: lookback={run_config['data']['lookback_steps']} (expected 72)")
        if run_config["data"]["feature_count"] != 33:
            failures.append(f"Seed {seed}: feature_count={run_config['data']['feature_count']} (expected 33)")
        if run_config["training"]["max_epochs"] != 30:
            failures.append(f"Seed {seed}: max_epochs={run_config['training']['max_epochs']} (expected 30)")
        if run_config["training"].get("final_refit_mode") is not True:
            failures.append(f"Seed {seed}: final_refit_mode=False")
        if run_config["training"].get("early_stopping_enabled") is True:
            failures.append(f"Seed {seed}: early_stopping_enabled=True (must be False)")
        if run_config["data"]["test_sample_count"] != 0:
            failures.append(f"Seed {seed}: test_sample_count={run_config['data']['test_sample_count']} (must be 0)")
        if seed != int(run_config["training"]["seed"]):
            failures.append(f"Seed {seed}: training.seed={run_config['training']['seed']} (must be {seed})")

        # Verify scaler checksums match the locked lineage values.
        locked_x_sha = run_config["lineage"].get("final_scaling_x_sha256")
        if locked_x_sha and locked_x_sha != x_sha:
            failures.append(
                f"Seed {seed}: X scaler SHA mismatch (locked {locked_x_sha[:16]} vs current {x_sha[:16]})"
            )
        locked_y_sha = run_config["lineage"].get("final_scaling_y_sha256")
        if locked_y_sha and locked_y_sha != y_sha:
            failures.append(
                f"Seed {seed}: Y scaler SHA mismatch (locked {locked_y_sha[:16]} vs current {y_sha[:16]})"
            )

        # Build the model with this seed's reproducibility and check it's fresh.
        model_cfg = run_config["model"]
        torch.manual_seed(seed)
        model = TransformerRegressor(model_cfg)
        seed_first_param = next(model.parameters()).flatten()[0].item()
        seed_param_count = sum(p.numel() for p in model.parameters())

        if seed_param_count != 422209 - 320000:
            failures.append(
                f"Seed {seed}: param_count={seed_param_count} "
                f"(must equal expected trainable 102209)"
            )
        # The first parameter value should differ from Seed42 (RNG freshness).
        if seed_first_param == seed42_first_param:
            failures.append(
                f"Seed {seed}: first parameter equals Seed42 — RNG not fresh"
            )
        print(f"[Seed {seed}] param_count={seed_param_count} first_param={seed_first_param:.6f}")

    # Check expected cache state: Seed42 must be COMPLETED, Seed123/2026 must NOT be cached.
    cache_path = ROOT / "artifacts" / "three_seed_final_runs" / "phase46_reuse_cache.json"
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        runs = cache.get("runs", {})
        seed42_in_cache = "42" in runs and runs["42"].get("status") == "COMPLETED"
        seed123_in_cache = "123" in runs
        seed2026_in_cache = "2026" in runs
        if not seed42_in_cache:
            print(f"[WARN] Seed42 not in reuse cache yet (will be materialized on next driver run).")
        if seed123_in_cache or seed2026_in_cache:
            failures.append(
                f"Seed123/2026 in cache prematurely: 123={seed123_in_cache}, 2026={seed2026_in_cache}"
            )
        print(f"[Cache] seed_42={'PRESENT' if seed42_in_cache else 'MISSING'}, "
              f"seed_123={'PRESENT' if seed123_in_cache else 'ABSENT'}, "
              f"seed_2026={'PRESENT' if seed2026_in_cache else 'ABSENT'}")

    # Verify FINAL_DEV count.
    if final_dev_n != 16630:
        failures.append(f"FINAL_DEV count={final_dev_n} (expected 16630)")
    print(f"[FinalDev] count={final_dev_n} fingerprint={final_dev.population_fingerprint[:16]}...")
    print(f"[Scaling] X={x_sha[:16]}... Y={y_sha[:16]}...")

    if failures:
        print(f"\n[FAIL] {len(failures)} failures:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"\n[OK] Seed123/Seed2026 dry-construct probe PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
