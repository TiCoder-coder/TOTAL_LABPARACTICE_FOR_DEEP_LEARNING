"""Unit tests for Phase 44 candidate loader and registration payloads."""
from __future__ import annotations

from pathlib import Path

import json

import pytest

from course_work.rolling_origin.candidate_loader import (
    CandidateSpec,
    LSTM_CANDIDATE_ID,
    PERSISTENCE_CANDIDATE_ID,
    load_candidates,
)
from course_work.rolling_origin.payloads import build_all_payloads
from course_work.rolling_origin.folds import build_rolling_folds


def _write_shortlist(tmp_path: Path) -> tuple[Path, Path]:
    shortlist = {
        "frozen": True,
        "candidates": [
            {
                "candidate_id": f"TR_C{i}_{kind}",
                "candidate_role": kind.upper(),
                "candidate_config_fingerprint": f"TRFP{i:064x}"[:64],
                "config": {
                    "data": {
                        "feature_variant_id": "FS2_TF1",
                        "target_scaling_option": "YS1",
                        "lookback_steps": 36 if i < 2 else 72,
                        "boundary_protocol": "WB0_CONTEXT_CARRY_OVER",
                        "test_sample_count": 2961,
                        "train_sample_count": 13670,
                        "validation_sample_count": 2960,
                    },
                    "training": {
                        "max_epochs": 50,
                        "early_stopping_patience": 10,
                        "early_stopping_enabled": True,
                        "learning_rate": 1e-3,
                        "weight_decay": 0.0,
                        "gradient_clipping_enabled": True,
                        "gradient_clip_max_norm": 1.0,
                        "loss_name": "MSE",
                    },
                    "lineage": {},
                    "model": {"model_family": "TRANSFORMER_ENCODER"},
                    "reproducibility": {"seed": 42},
                },
            }
            for i, kind in enumerate(["primary", "alt_wd", "alt_lookback"])
        ],
    }
    handoff = {
        "winner_config_fingerprint": "bce5a2cd6ba86435b7c02a1f1a9d25a6e214493dbd8d6da22886e328287f8593",
        "winner_config": {
            "data": {
                "feature_variant_id": "FS2_TF1",
                "target_scaling_option": "YS1",
                "lookback_steps": 36,
                "boundary_protocol": "WB0_CONTEXT_CARRY_OVER",
                "feature_count": 33,
            },
            "training": {
                "max_epochs": 50,
                "early_stopping_patience": 10,
                "early_stopping_enabled": True,
                "learning_rate": 3e-4,
                "weight_decay": 0.0,
                "gradient_clipping_enabled": True,
                "gradient_clip_max_norm": 1.0,
                "loss_name": "MSE",
            },
            "lineage": {},
            "model": {"model_family": "LSTM"},
            "reproducibility": {"seed": 42},
        },
    }
    shortlist_path = tmp_path / "shortlist.json"
    handoff_path = tmp_path / "lstm_handoff.json"
    shortlist_path.write_text(json.dumps(shortlist))
    handoff_path.write_text(json.dumps(handoff))
    return shortlist_path, handoff_path


class TestCandidateLoader:
    def test_loads_three_transformers_and_lstm(self, tmp_path: Path) -> None:
        shortlist_path, handoff_path = _write_shortlist(tmp_path)
        # Project root is irrelevant for this unit test since extract_robase_population isn't called
        candidates = load_candidates(
            project_root=tmp_path,
            transformer_shortlist_path=shortlist_path,
            lstm_handoff_path=handoff_path,
        )
        assert len(candidates) == 4
        # 3 Transformers + 1 LSTM
        families = [c.model_family for c in candidates]
        assert families.count("TRANSFORMER_ENCODER") == 3
        assert families.count("LSTM") == 1
        assert candidates[-1].candidate_id == LSTM_CANDIDATE_ID
        # Lookbacks: 36, 36, 72, 36
        assert candidates[0].lookback_steps == 36
        assert candidates[2].lookback_steps == 72
        assert candidates[3].lookback_steps == 36
        # Shortlist positions
        assert candidates[0].shortlist_position == 0
        assert candidates[3].shortlist_position == 3


class TestPayloads:
    def test_build_all_payloads_returns_24(self, tmp_path: Path) -> None:
        shortlist_path, handoff_path = _write_shortlist(tmp_path)
        candidates = load_candidates(
            project_root=tmp_path,
            transformer_shortlist_path=shortlist_path,
            lstm_handoff_path=handoff_path,
        )
        rtrn = [str(i) for i in range(100)]  # "0", "1", ..., "99"
        rval = [str(1000 + i) for i in range(30)]  # "1000", ..., "1029"
        folds = build_rolling_folds(rtrn, rval, k=3)
        # Synthesize 12 scaler audits
        scaler_audits = []
        for f in folds:
            for c in candidates:
                class A:
                    pass

                a = A()
                a.candidate_id = c.candidate_id
                a.fold_id = str(f.fold_id)
                a.bundle_checksum = "abc"
                scaler_audits.append(a)
        a_payloads, b_payloads = build_all_payloads(
            candidates=candidates,
            folds=folds,
            scaler_a_audits=scaler_audits,
            scaler_b_audits=scaler_audits,
        )
        assert len(a_payloads) == 12
        assert len(b_payloads) == 12
        # Stage B has parent_run_id pointing at Stage A
        for a, b in zip(a_payloads, b_payloads):
            assert a.stage == "A"
            assert b.stage == "B"
            assert a.parent_run_id is None
            assert b.parent_run_id is not None
            assert b.parent_run_id.startswith("RUN_RO_A_")
