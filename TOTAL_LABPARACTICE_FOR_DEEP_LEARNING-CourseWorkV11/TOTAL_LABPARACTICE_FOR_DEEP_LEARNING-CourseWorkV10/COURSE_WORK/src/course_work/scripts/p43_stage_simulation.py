"""Phase 43 full LT1 -> LT5 simulated execution harness."""
from __future__ import annotations

import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from course_work.experiments.registry import (
    ExperimentRegistry,
    build_reference_run_config,
    compute_config_fingerprint,
    load_upstream_context,
    validate_run_config,
)
from course_work.lstm_tuning import tuning_space as ts
from course_work.lstm_tuning.reference_resolution import resolve_lstm_t0_reference
from course_work.lstm_tuning.shared_data_contract import resolve_shared_data_contract
from course_work.lstm_tuning.stages import StageExecutor
from course_work.lstm_tuning.tuning_space import (
    FIXED_TRAINING_CONTRACT,
    MAX_FRESH_SCIENTIFIC_RUNS,
    REFERENCE_HYPERPARAMETERS,
    _set_path,
)
from course_work.lstm_tuning.winners import StageWinner, select_stage_winner
from scripts.phase43_lstm_tuning import _enforce_scientific_contract


def _stage_winner(candidates, stage: str, scores: dict[str, float]) -> tuple[StageWinner, list[dict]]:
    rngd = []
    for c in candidates:
        if c.source_type != "FRESH" or c.option not in scores:
            continue
        rngd.append({
            "run_id": f"RUN_SYNTH_{stage}_{c.option}",
            "option": c.option,
            "value": c.value,
            "validation_rmse_wh": scores[c.option],
            "validation_mae_wh": None,
            "validation_r2": None,
            "best_epoch": 25,
            "epochs_completed": 35,
            "stop_reason": "EARLY_STOPPED",
            "trainable_parameters": 1024 + (hash(c.option) % 4096),
            "metrics": {
                "rmse_wh": scores[c.option],
                "mae_wh": scores[c.option] * 0.6,
                "r2": 0.5,
            },
            "source_type": "FRESH",
            "config_fingerprint": "syn",
        })
    return select_stage_winner(stage, rngd), rngd


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    contract = resolve_shared_data_contract(project_root)
    registry = ExperimentRegistry(project_root)
    upstream = load_upstream_context(project_root)
    reference = resolve_lstm_t0_reference(registry, contract)
    base = None
    if reference.phase20_exact_match and reference.reference_run_id:
        for rec in registry._load_records():
            if rec.get("run_id") == reference.reference_run_id:
                base = deepcopy(rec["config"])
                break
    if base is None:
        base = build_reference_run_config(project_root=project_root, model_family="LSTM")
    for path, value in FIXED_TRAINING_CONTRACT.items():
        _set_path(base, path, value)
    for path, value in REFERENCE_HYPERPARAMETERS.items():
        _set_path(base, path, value)

    executor = StageExecutor(
        contract=contract,
        reference_config=base,
        reference_run_id=reference.reference_run_id if reference.phase20_exact_match else None,
    )

    SYNTH_SCORES = {
        "LT1": {"LH32": 70.0, "LH128": 65.0},
        "LT2": {"LN1": 67.0},
        "LT3": {"LD0": 66.5, "LD2": 67.5},
        "LT4": {"LLR1": 65.5, "LLR3": 64.8},
        "LT5": {"LWD0": 64.0, "LWD2": 64.6},
    }
    chosen_winners: dict[str, str] = {}
    fresh_total = 0
    trace = []
    for stage_name in ts.LT_STAGE_ORDER:
        planned = executor.plan_stage(stage_name)
        line = f"[{planned.stage}] applicable={planned.applicable}"
        if not planned.applicable:
            line += " -> SKIPPED (num_layers < 2)"
            trace.append(line)
            continue
        winner, fresh_records = _stage_winner(planned.candidates, planned.stage, SYNTH_SCORES[planned.stage])
        line += (
            f" candidates={[c.option for c in planned.candidates]}"
            f" fresh={[r['option'] for r in fresh_records]}"
            f" winner={winner.option} rmse={winner.validation_rmse_wh:.4f}"
        )
        trace.append(line)
        fresh_total += len(fresh_records)
        chosen_winners[planned.stage] = winner.option
        executor.commit_winner(planned.stage, winner, winner.run_id)

    final_cfg = executor.current_config
    final_cfg = _enforce_scientific_contract(final_cfg, contract, upstream)
    validated = validate_run_config(final_cfg, upstream)
    final_fp = compute_config_fingerprint(validated)
    trace.append(f"[END] fresh_total={fresh_total} <= MAX_FRESH={MAX_FRESH_SCIENTIFIC_RUNS}")
    trace.append(f"[END] chosen_winners={chosen_winners}")
    trace.append(
        f"[END] final_config fingerprint={final_fp[:16]}"
        f" hs={final_cfg['model']['hidden_size']}"
        f" nl={final_cfg['model']['num_layers']}"
        f" do={final_cfg['model']['dropout']}"
        f" lr={final_cfg['training']['learning_rate']}"
        f" wd={final_cfg['training']['weight_decay']}"
    )

    print("\n".join(trace))

    assert fresh_total <= MAX_FRESH_SCIENTIFIC_RUNS
    assert final_cfg["model"]["hidden_size"] == 128  # LT1 winner=LH128
    assert final_cfg["model"]["num_layers"] == 1     # LT2 winner=LN1
    assert final_cfg["model"]["dropout"] == 0.1      # LT3 skipped (num_layers<2); reference preserved
    assert abs(final_cfg["training"]["learning_rate"] - 1e-3) < 1e-6  # LT4 winner=LLR3
    assert final_cfg["training"]["weight_decay"] == 0.0                # LT5 winner=LWD0
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
