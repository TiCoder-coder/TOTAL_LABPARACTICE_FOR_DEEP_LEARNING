"""Consistency checks for Phase 43 LSTM tuning artifacts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from course_work.lstm_tuning import artifacts as ph43_artifacts
from course_work.lstm_tuning.shared_data_contract import SharedDataContract
from course_work.lstm_tuning.winners import StageWinner


@dataclass
class ConsistencyReport:
    ok: bool
    discrepancies: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "discrepancies": self.discrepancies}


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def verify_final_winner_matches_lineage(
    project_root: Path,
    contract: SharedDataContract,
    final_winner_run_id: str | None,
    final_winner_config: dict[str, Any],
    final_winner_rmse_wh: float,
) -> ConsistencyReport:
    """Verify chained LT1→LT5 winner config == lstm_tuned_winner.json == phase_43_signoff."""
    disc: list[dict[str, Any]] = []
    tuned_winner_path = project_root / "artifacts/lstm_tuning/lstm_tuned_winner.json"
    signoff_path = project_root / "artifacts/lstm_tuning/phase_43_signoff.json"
    phase44_path = project_root / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json"

    tuned = _load_json(tuned_winner_path) if tuned_winner_path.exists() else None
    signoff = _load_json(signoff_path) if signoff_path.exists() else None
    phase44 = _load_json(phase44_path) if phase44_path.exists() else None

    def _flat(d: dict[str, Any]) -> dict[str, Any]:
        return {
            "hidden_size": d.get("hidden_size"),
            "num_layers": d.get("num_layers"),
            "dropout_arg": d.get("dropout_arg"),
            "learning_rate": d.get("learning_rate"),
            "weight_decay": d.get("weight_decay"),
        }

    model = final_winner_config.get("model", {})
    training = final_winner_config.get("training", {})
    flat = {
        "hidden_size": model.get("hidden_size"),
        "num_layers": model.get("num_layers"),
        "dropout_arg": model.get("dropout"),
        "learning_rate": training.get("learning_rate"),
        "weight_decay": training.get("weight_decay"),
    }

    if tuned is not None:
        for k, v in flat.items():
            if tuned.get(k) != v:
                disc.append({"check": f"tuned_winner.{k}", "expected": v, "actual": tuned.get(k)})
        if tuned.get("lookback_id") and tuned["lookback_id"] != contract.lookback_id:
            disc.append({"check": "tuned_winner.lookback_id", "expected": contract.lookback_id, "actual": tuned["lookback_id"]})
        if tuned.get("batch_size") and tuned["batch_size"] != contract.batch_size:
            disc.append({"check": "tuned_winner.batch_size", "expected": contract.batch_size, "actual": tuned["batch_size"]})
        if tuned.get("winner_run_id") and final_winner_run_id and tuned["winner_run_id"] != final_winner_run_id:
            disc.append({"check": "tuned_winner.winner_run_id", "expected": final_winner_run_id, "actual": tuned["winner_run_id"]})

    if signoff is not None:
        winner = signoff.get("final_winner") or signoff.get("winner") or {}
        for k, v in flat.items():
            actual = winner.get(k) if isinstance(winner, dict) else None
            if actual is not None and actual != v:
                disc.append({"check": f"signoff.final_winner.{k}", "expected": v, "actual": actual})
        if signoff.get("shared_data_contract", {}).get("lookback_id") != contract.lookback_id:
            disc.append({"check": "signoff.shared_lookback", "expected": contract.lookback_id, "actual": signoff.get("shared_data_contract", {}).get("lookback_id")})

    if phase44 is not None:
        if phase44.get("shared_lookback") and phase44["shared_lookback"] != contract.lookback_steps:
            disc.append({"check": "phase44.shared_lookback", "expected": contract.lookback_steps, "actual": phase44["shared_lookback"]})

    return ConsistencyReport(ok=not disc, discrepancies=disc)


def verify_stage_lineage_chain(project_root: Path) -> ConsistencyReport:
    """Each LT* winner config must become the next reference."""
    disc: list[dict[str, Any]] = []
    stages = ["lt1_hidden_size_winner", "lt2_layers_winner", "lt3_dropout_winner", "lt4_learning_rate_winner", "lt5_weight_decay_winner"]
    prev = None
    for s in stages:
        path = project_root / "artifacts/lstm_tuning" / f"{s}.json"
        if not path.exists():
            continue
        d = _load_json(path)
        if s == "lt3_dropout_winner":
            if d.get("stage_status") == "SKIPPED_NOT_APPLICABLE":
                prev = prev  
                continue
        flat = {
            "hidden_size": d.get("winner_value") if s == "lt1_hidden_size_winner" else None,
            "num_layers": d.get("winner_value") if s == "lt2_layers_winner" else None,
            "dropout": d.get("winner_value") if s == "lt3_dropout_winner" else None,
            "learning_rate": d.get("winner_value") if s == "lt4_learning_rate_winner" else None,
            "weight_decay": d.get("winner_value") if s == "lt5_weight_decay_winner" else None,
        }
        if prev is not None:
            for k, v in prev.items():
                if v is None:
                    continue
                if d.get("reference_used_value") and d["reference_used_value"] != v:
                    disc.append({"check": f"{s}.reference_used_value", "expected": v, "actual": d.get("reference_used_value")})
                    break
        prev = flat

    return ConsistencyReport(ok=not disc, discrepancies=disc)
