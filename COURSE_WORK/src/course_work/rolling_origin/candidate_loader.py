"""Phase 44 — Candidate loader.

Reads the frozen Phase 42 Transformer shortlist and the corrected Phase 43 LSTM
winner, returning a unified list of `CandidateSpec` records suitable for the
Stage A/B/C orchestration.

Each CandidateSpec carries:
  - candidate_id           (TR_C0_PRIMARY / TR_C1_ALT_WEIGHT_DECAY / TR_C2_ALT_LOOKBACK / LSTM_TUNED_WINNER)
  - model_family           (TRANSFORMER_ENCODER | LSTM)
  - shortlist_position     (0-based rank in Phase 42 shortlist, or 3 for LSTM)
  - config                 (the full {data, model, training, lineage, reproducibility, runtime} dict)
  - config_fingerprint
  - feature_variant_id
  - target_scaling_option
  - lookback_steps
  - boundary_protocol      (always WB0_CONTEXT_CARRY_OVER)
  - source_phase           ("PHASE_42" | "PHASE_43")
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from course_work.experiments.registry import compute_config_fingerprint


LSTM_CANDIDATE_ID = "LSTM_TUNED_WINNER"
PERSISTENCE_CANDIDATE_ID = "PERSISTENCE_LAST_VALUE"
PERSISTENCE_VERSION = "WB0_LAST_STEP_v1"


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    model_family: str
    shortlist_position: int
    config: dict
    config_fingerprint: str
    feature_variant_id: str
    target_scaling_option: str
    lookback_steps: int
    boundary_protocol: str
    source_phase: str
    candidate_role: str
    requires_training: bool = True
    notes: str = ""

    def as_row(self) -> dict[str, Any]:
        """Row used by O44.7 Candidate matrix."""
        d = self.config.get("data", {})
        t = self.config.get("training", {})
        return {
            "candidate_id": self.candidate_id,
            "model_family": self.model_family,
            "candidate_role": self.candidate_role,
            "config_fingerprint": self.config_fingerprint,
            "feature_variant": self.feature_variant_id,
            "target_scaling": self.target_scaling_option,
            "lookback": self.lookback_steps,
            "batch": t.get("batch_size", d.get("batch_size", 64)),
            "loss": t.get("loss_name", "MSE"),
            "max_epochs": t.get("max_epochs", 50),
            "patience": t.get("early_stopping_patience", 10),
            "clipping": t.get("gradient_clipping_enabled", False),
            "revin": t.get("revin_enabled", False),
            "requires_training": self.requires_training,
            "source_phase": self.source_phase,
            "status": "READY",
        }


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def load_candidates(
    *,
    project_root: Path,
    transformer_shortlist_path: Path,
    lstm_handoff_path: Path,
) -> list[CandidateSpec]:
    """Load all 4 candidates in canonical order: TR_C0, TR_C1, TR_C2, LSTM."""
    candidates: list[CandidateSpec] = []

    shortlist = _read_json(transformer_shortlist_path)
    raw = shortlist.get("candidates", [])
    for pos, c in enumerate(raw):
        cfg = c.get("config", {})
        d = cfg.get("data", {})
        fp = c.get(
            "candidate_config_fingerprint"
        ) or compute_config_fingerprint(cfg)
        candidates.append(
            CandidateSpec(
                candidate_id=c.get("candidate_id", f"TR_C{pos}"),
                model_family="TRANSFORMER_ENCODER",
                shortlist_position=pos,
                config=cfg,
                config_fingerprint=fp,
                feature_variant_id=d.get("feature_variant_id", "FS2_TF1"),
                target_scaling_option=d.get("target_scaling_option", "YS1"),
                lookback_steps=int(d.get("lookback_steps", 36)),
                boundary_protocol=d.get(
                    "boundary_protocol", "WB0_CONTEXT_CARRY_OVER"
                ),
                source_phase="PHASE_42",
                candidate_role=c.get("candidate_role", "PRIMARY"),
            )
        )

    handoff = _read_json(lstm_handoff_path)
    lstm_cfg = handoff.get("winner_config", {})
    if lstm_cfg:
        d = lstm_cfg.get("data", {})
        # Canonical Phase 43 -> 44 identity is the ExperimentRegistry
        # config_fingerprint of the complete winner_config.  Never accept an
        # unrelated artifact hash or a hash computed over a reduced/reshaped
        # payload as though it had the same semantics.
        recomputed_fp = compute_config_fingerprint(lstm_cfg)
        recorded_fp = str(handoff.get("winner_config_fingerprint") or "")
        if not recorded_fp:
            raise ValueError("Phase 43 handoff is missing winner_config_fingerprint")
        if recorded_fp != recomputed_fp:
            raise ValueError(
                "Phase 43 handoff fingerprint conflict: recorded "
                f"{recorded_fp}, recomputed canonical config_fingerprint "
                f"{recomputed_fp}"
            )
        fp = recorded_fp
        candidates.append(
            CandidateSpec(
                candidate_id=LSTM_CANDIDATE_ID,
                model_family="LSTM",
                shortlist_position=len(candidates),
                config=lstm_cfg,
                config_fingerprint=fp,
                feature_variant_id=d.get("feature_variant_id", "FS2_TF1"),
                target_scaling_option=d.get("target_scaling_option", "YS1"),
                lookback_steps=int(d.get("lookback_steps", 36)),
                boundary_protocol=d.get(
                    "boundary_protocol", "WB0_CONTEXT_CARRY_OVER"
                ),
                source_phase="PHASE_43",
                candidate_role="TUNED_WINNER",
            )
        )

    return candidates
