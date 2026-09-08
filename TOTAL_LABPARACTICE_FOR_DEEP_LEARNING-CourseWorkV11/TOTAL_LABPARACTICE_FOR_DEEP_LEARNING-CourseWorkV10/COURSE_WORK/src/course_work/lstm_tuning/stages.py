"""Sequential one-factor LT1-LT5 stage execution helpers (non-training).

Provides a `StageExecutor` that prepares the candidate space for each stage,
validates one-factor delta vs the current reference, ensures the reference
candidate is included, and prepares the winner config for the next stage.

The actual training is intentionally NOT invoked by this module. The
corrected scientific execution entry point is `scripts/phase43_lstm_tuning.py`,
which must run as the human-only step after this package validates the
candidate plan.

The non-scientific `scripts/phase43_dry_run.py` validates the executor and
all candidates without touching the registry or training engine.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from course_work.lstm_tuning import tuning_space as ts
from course_work.lstm_tuning.shared_data_contract import SharedDataContract
from course_work.lstm_tuning.winners import StageWinner


@dataclass(frozen=True)
class PlannedCandidate:
    option: str
    value: Any
    config: dict[str, Any]
    source_type: str  
    reuse_run_id: str | None = None


@dataclass(frozen=True)
class PlannedStage:
    stage: str
    factor: str
    factor_values: tuple[Any, ...]
    candidates: list[PlannedCandidate]
    reference_option: str
    reference_value: Any
    reference_run_id: str | None
    reference_reused: bool
    applicable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "factor": self.factor,
            "factor_values": list(self.factor_values),
            "candidates": [
                {
                    "option": c.option,
                    "value": c.value,
                    "source_type": c.source_type,
                    "reuse_run_id": c.reuse_run_id,
                }
                for c in self.candidates
            ],
            "reference_option": self.reference_option,
            "reference_value": self.reference_value,
            "reference_run_id": self.reference_run_id,
            "reference_reused": self.reference_reused,
            "applicable": self.applicable,
        }


class StageExecutor:
    def __init__(
        self,
        contract: SharedDataContract,
        reference_config: dict[str, Any],
        reference_run_id: str | None = None,
    ) -> None:
        self._contract = contract
        self._current_config = deepcopy(reference_config)
        self._current_run_id = reference_run_id
        self._history: list[PlannedStage] = []
        self._planned_candidate_count = 0
        self._reference_run_id = reference_run_id

    @property
    def current_config(self) -> dict[str, Any]:
        return deepcopy(self._current_config)

    @property
    def current_run_id(self) -> str | None:
        return self._current_run_id

    @property
    def history(self) -> list[PlannedStage]:
        return list(self._history)

    @property
    def planned_candidate_count(self) -> int:
        return self._planned_candidate_count

    def plan_stage(self, stage: str) -> PlannedStage:
        factor = ts.stage_factor(stage)
        factor_values = ts.stage_values(stage)

        if stage == "LT3":
            num_layers = self._current_config.get("model", {}).get("num_layers")
            if not ts.lt3_applicable(int(num_layers or 0)):
                planned = PlannedStage(
                    stage="LT3",
                    factor=factor,
                    factor_values=tuple(),
                    candidates=[],
                    reference_option="LD0",
                    reference_value=0.0,
                    reference_run_id=self._current_run_id,
                    reference_reused=False,
                    applicable=False,
                )
                self._history.append(planned)
                return planned

        candidate_options = {}
        for value in factor_values:
            option = _candidate_option(stage, value)
            candidate_options[option] = value

        candidates: list[PlannedCandidate] = []
        for option, value in candidate_options.items():
            cfg = deepcopy(self._current_config)
            _set_path(cfg, factor, value)
            reference_value = ts.reference_value(stage)
            is_reference_value = value == reference_value
            candidates.append(
                PlannedCandidate(
                    option=option,
                    value=value,
                    config=cfg,
                    source_type="REUSED_REFERENCE" if is_reference_value else "FRESH",
                    reuse_run_id=self._current_run_id if is_reference_value else None,
                )
            )
        ts.assert_reference_included(stage, {c.option: c.config for c in candidates})

        for c in candidates:
            if c.source_type == "FRESH":
                ts.candidate_differs_only_one_factor(stage, c.config, self._current_config)

        planned = PlannedStage(
            stage=stage,
            factor=factor,
            factor_values=factor_values,
            candidates=candidates,
            reference_option=_candidate_option(stage, ts.reference_value(stage)),
            reference_value=ts.reference_value(stage),
            reference_run_id=self._current_run_id,
            reference_reused=any(c.source_type == "REUSED_REFERENCE" for c in candidates),
            applicable=True,
        )
        self._history.append(planned)
        fresh_count = sum(1 for c in candidates if c.source_type == "FRESH")
        self._planned_candidate_count += fresh_count
        return planned

    def commit_winner(self, stage: str, winner: StageWinner, winner_run_id: str | None) -> None:
        cfg = deepcopy(self._current_config)
        _set_path(cfg, ts.stage_factor(stage), winner.value)
        self._current_config = cfg
        self._current_run_id = winner_run_id

    def plan_full_sweep(self) -> list[PlannedStage]:
        return [self.plan_stage(stage) for stage in ts.LT_STAGE_ORDER]


def _set_path(d: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor = d
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _candidate_option(stage: str, value: Any) -> str:
    if stage == "LT1":
        return f"LH{int(value)}"
    if stage == "LT2":
        return f"LN{int(value)}"
    if stage == "LT3":
        return f"LD{int(round(float(value) * 10))}"
    if stage == "LT4":
        if abs(float(value) - 1e-4) < 1e-12:
            return "LLR1"
        if abs(float(value) - 3e-4) < 1e-12:
            return "LLR2"
        if abs(float(value) - 1e-3) < 1e-12:
            return "LLR3"
        raise ValueError(f"LT4 LR not in registered space: {value!r}")
    if stage == "LT5":
        if float(value) == 0.0:
            return "LWD0"
        if abs(float(value) - 1e-4) < 1e-12:
            return "LWD1"
        if abs(float(value) - 1e-3) < 1e-12:
            return "LWD2"
        raise ValueError(f"LT5 WD not in registered space: {value!r}")
    raise ValueError(f"Unknown stage {stage!r}")


def estimate_fresh_scientific_runs(reference_resolved: bool) -> int:
    if reference_resolved:
        return ts.MAX_FRESH_SCIENTIFIC_RUNS - 1
    return ts.MAX_FRESH_SCIENTIFIC_RUNS
