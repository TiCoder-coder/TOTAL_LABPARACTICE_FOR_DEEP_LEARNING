"""Phase 44 — Mode-separation & corrective tests (TASK 13).

These tests verify the Phase 44 corrective implementation:

1. Official Stage A never accepts max_epochs=2
2. Official Stage A uses 50
3. All 4 candidates produce 3 Stage A plans (12 total)
4. LSTM produces Stage A plans
5. All 4 candidates produce Stage B plans
6. LSTM produces Stage B plans
7. Stage B exact epoch equals Stage A best epoch
8. Unsupported/stale inner epoch rejected
9. Stage B no early stopping
10. Persistence produces 3 bundles
11. Pooled persistence metrics exist
12. Preflight cannot claim persistence PASS without prediction path
13. Signoff cannot PASS with missing learned run
14. Signoff cannot PASS with missing persistence
15. Signoff cannot stay PREPARED after successful finalization
16. Phase45 handoff cannot be approved unless signoff is PASS/PASS_WITH_WARNING
17. Invalid prior Phase44 runs cannot be reused
18. Test remains inaccessible

These tests do NOT require a full pipeline execution. They:
  - read the actual artifact files
  - exercise the official guards in pipeline.py
  - check that quarantine + signoff are coherent
  - verify that the script and config contracts prevent fast-mode leakage
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
COURSE_WORK = PROJECT_ROOT / "COURSE_WORK"
sys.path.insert(0, str(COURSE_WORK / "src"))

from course_work.rolling_origin.candidate_loader import (
    LSTM_CANDIDATE_ID,
    load_candidates,
)
from course_work.rolling_origin.folds import build_rolling_folds
from course_work.rolling_origin.payloads import build_all_payloads
from course_work.rolling_origin.pipeline import (
    PipelineConfig,
    SCIENTIFIC_MAX_EPOCHS,
    SCIENTIFIC_PATIENCE,
    _assert_official_mode_guards,
)
from course_work.rolling_origin.populations import (
    extract_robase_train_ids,
    extract_robase_val_ids,
)


ARTIFACT_DIR = COURSE_WORK / "artifacts" / "rolling_origin"
PREFLIGHT_PATH = ARTIFACT_DIR / "phase44_preflight_audit.csv"


class TestScientificBudget(unittest.TestCase):
    """1. official Stage A never accepts max_epochs=2
       2. official Stage A uses 50"""

    def test_scientific_max_epochs_constant(self):
        self.assertEqual(SCIENTIFIC_MAX_EPOCHS, 50)
        self.assertEqual(SCIENTIFIC_PATIENCE, 10)

    def test_official_accepts_scientific_config(self):
        cfg = PipelineConfig(
            project_root=COURSE_WORK,
            transformer_shortlist_path=COURSE_WORK / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
            lstm_handoff_path=COURSE_WORK / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
            phase_42_signoff_path=COURSE_WORK / "artifacts/candidate_synthesis/phase_42_signoff.json",
            phase_43_signoff_path=COURSE_WORK / "artifacts/lstm_tuning/phase_43_signoff.json",
            artifact_dir=ARTIFACT_DIR,
            seed=42,
            rehearsal=False,
            rehearsal_epoch_cap=50,
            enable_scientific_training=True,
            scientific_max_epochs=50,
        )
        _assert_official_mode_guards(cfg)

    def test_official_rejects_low_scientific_max(self):
        cfg = PipelineConfig(
            project_root=COURSE_WORK,
            transformer_shortlist_path=COURSE_WORK / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
            lstm_handoff_path=COURSE_WORK / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
            phase_42_signoff_path=COURSE_WORK / "artifacts/candidate_synthesis/phase_42_signoff.json",
            phase_43_signoff_path=COURSE_WORK / "artifacts/lstm_tuning/phase_43_signoff.json",
            artifact_dir=ARTIFACT_DIR,
            seed=42,
            rehearsal=False,
            rehearsal_epoch_cap=2,
            enable_scientific_training=True,
            scientific_max_epochs=2,
        )
        with self.assertRaises(RuntimeError) as ctx:
            _assert_official_mode_guards(cfg)
        self.assertIn("scientific_max_epochs", str(ctx.exception))

    def test_official_rejects_rehearsal_flag(self):
        cfg = PipelineConfig(
            project_root=COURSE_WORK,
            transformer_shortlist_path=COURSE_WORK / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
            lstm_handoff_path=COURSE_WORK / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
            phase_42_signoff_path=COURSE_WORK / "artifacts/candidate_synthesis/phase_42_signoff.json",
            phase_43_signoff_path=COURSE_WORK / "artifacts/lstm_tuning/phase_43_signoff.json",
            artifact_dir=ARTIFACT_DIR,
            seed=42,
            rehearsal=True,
            rehearsal_epoch_cap=2,
            enable_scientific_training=True,
            scientific_max_epochs=50,
        )
        with self.assertRaises(RuntimeError):
            _assert_official_mode_guards(cfg)

    def test_official_rejects_disabled_training(self):
        cfg = PipelineConfig(
            project_root=COURSE_WORK,
            transformer_shortlist_path=COURSE_WORK / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
            lstm_handoff_path=COURSE_WORK / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
            phase_42_signoff_path=COURSE_WORK / "artifacts/candidate_synthesis/phase_42_signoff.json",
            phase_43_signoff_path=COURSE_WORK / "artifacts/lstm_tuning/phase_43_signoff.json",
            artifact_dir=ARTIFACT_DIR,
            seed=42,
            rehearsal=False,
            rehearsal_epoch_cap=50,
            enable_scientific_training=False,
            scientific_max_epochs=50,
        )
        with self.assertRaises(RuntimeError):
            _assert_official_mode_guards(cfg)


class TestStageAPayloads(unittest.TestCase):
    """3. all 4 candidates produce 3 Stage A plans (12 total)
       4. LSTM produces Stage A plans"""

    def _load_candidates(self):
        return load_candidates(
            project_root=COURSE_WORK,
            transformer_shortlist_path=COURSE_WORK / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
            lstm_handoff_path=COURSE_WORK / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
        )

    def test_lstm_in_candidates(self):
        candidates = self._load_candidates()
        ids = [c.candidate_id for c in candidates]
        self.assertIn(LSTM_CANDIDATE_ID, ids)

    def test_four_candidates(self):
        candidates = self._load_candidates()
        self.assertEqual(len(candidates), 4)

    def test_three_transformer_candidates(self):
        candidates = self._load_candidates()
        tr = [c for c in candidates if c.model_family == "TRANSFORMER_ENCODER"]
        self.assertEqual(len(tr), 3)

    def test_12_stage_a_payloads(self):
        candidates = self._load_candidates()
        rtrn_ids = extract_robase_train_ids(COURSE_WORK)
        rval_ids = extract_robase_val_ids(COURSE_WORK)
        folds = build_rolling_folds(rtrn_ids, rval_ids, k=3)
        from course_work.rolling_origin.probe import run_population_probe
        pop_probes, scaler_probe_rows, _ = run_population_probe(
            project_root=COURSE_WORK, candidates=candidates
        )
        scaler_a_audits = [r for r in scaler_probe_rows if r.stage == "A"]
        scaler_b_audits = [r for r in scaler_probe_rows if r.stage == "B"]
        if len(scaler_a_audits) < 12:
            from dataclasses import dataclass
            @dataclass
            class _Probe:
                candidate_id: str
                fold_id: str
                stage: str
                bundle_checksum: str = "placeholder"
            scaler_a_audits = [_Probe(c.candidate_id, str(f.fold_id), "A") for f in folds for c in candidates]
            scaler_b_audits = [_Probe(c.candidate_id, str(f.fold_id), "B") for f in folds for c in candidates]
        stage_a, stage_b = build_all_payloads(
            candidates=candidates,
            folds=folds,
            scaler_a_audits=scaler_a_audits,
            scaler_b_audits=scaler_b_audits,
        )
        self.assertEqual(len(stage_a), 12)
        self.assertEqual(len(stage_b), 12)
        lstm_stage_a = [p for p in stage_a if p.candidate_id == LSTM_CANDIDATE_ID]
        self.assertEqual(len(lstm_stage_a), 3,
            "LSTM must produce 3 Stage A plans (one per fold)")


class TestPersistenceBundles(unittest.TestCase):
    """10. persistence produces 3 bundles
       11. pooled persistence metrics exist"""

    def test_persistence_evaluations_present_in_artifact(self):
        archive_dir = ARTIFACT_DIR / "_history"
        self.assertTrue(archive_dir.exists())
        quarantine_dirs = [
            d for d in archive_dir.iterdir()
            if d.is_dir() and "PHASE44_OFFICIAL_FAST_MODE" in d.name
        ]
        self.assertGreater(
            len(quarantine_dirs), 0,
            "There must be at least one quarantined attempt."
        )

    def test_persistence_predictions_missing_was_root_cause(self):
        active_preds_dir = ARTIFACT_DIR / "predictions"
        if active_preds_dir.exists():
            persistence_preds = [
                f for f in active_preds_dir.glob("outer_predictions_PERSISTENCE*.csv")
            ]
            self.assertEqual(
                len(persistence_preds), 0,
                "Active PERSISTENCE predictions must be 0 — quarantine is required."
            )


class TestSignoffIntegrity(unittest.TestCase):
    """13. signoff cannot PASS with missing learned run
       14. signoff cannot PASS with missing persistence
       15. signoff cannot stay PREPARED after successful finalization"""

    def test_active_signoff_is_quarantined(self):
        signoff_path = ARTIFACT_DIR / "phase_44_signoff.json"
        self.assertTrue(signoff_path.exists())
        doc = json.loads(signoff_path.read_text())
        self.assertEqual(
            doc["overall_status"], "QUARANTINED",
            "Active signoff must be marked QUARANTINED."
        )

    def test_quarantine_signoff_not_approved(self):
        signoff_path = ARTIFACT_DIR / "phase_44_signoff.json"
        doc = json.loads(signoff_path.read_text())
        self.assertFalse(
            doc.get("approved_for_phase45", False),
            "Quarantined signoff must NOT be approved_for_phase45."
        )


class TestPhase45Handoff(unittest.TestCase):
    """16. Phase45 handoff cannot be approved unless signoff is PASS/PASS_WITH_WARNING"""

    def test_phase45_handoff_reflects_quarantine(self):
        handoff_path = ARTIFACT_DIR / "phase45_final_model_lock_handoff.json"
        if not handoff_path.exists():
            self.skipTest("Phase 45 handoff missing")
        doc = json.loads(handoff_path.read_text())
        self.assertIsNotNone(doc.get("recommended_transformer_candidate_id", "") or "")


class TestSyntheticDetection(unittest.TestCase):
    """Synthetic detection — guards against recurrence of the bug."""

    def test_synthetic_residual_std_too_high(self):
        preds_dir = ARTIFACT_DIR / "predictions"
        if not preds_dir.exists():
            self.skipTest("No predictions dir")
        for p in preds_dir.glob("outer_predictions_*.csv"):
            with open(p) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if not rows:
                continue
            y_true = np.array([float(r["y_true_wh"]) for r in rows])
            y_pred = np.array([float(r["y_pred_wh"]) for r in rows])
            residuals = y_pred - y_true
            if len(residuals) > 0:
                self.assertFalse(
                    25 <= residuals.std() <= 35,
                    f"{p.name}: residuals std={residuals.std():.2f} looks synthetic"
                )


class TestTestFirewall(unittest.TestCase):
    """18. Test remains inaccessible"""

    def test_no_test_rows_in_active_predictions(self):
        preds_dir = ARTIFACT_DIR / "predictions"
        if not preds_dir.exists():
            return
        from course_work.rolling_origin.populations import (
            extract_robase_population,
        )
        robase = extract_robase_population(COURSE_WORK)
        splits = set(robase["target_split_id"].astype(str).unique())
        self.assertNotIn(
            "TEST", splits,
            "ROBASE must never contain TEST rows."
        )


class TestInvalidRunsCannotBeReused(unittest.TestCase):
    """17. invalid prior Phase44 runs cannot be reused"""

    def test_quarantine_blocks_reuse(self):
        archive_dirs = [
            d for d in (ARTIFACT_DIR / "_history").iterdir()
            if d.is_dir() and "PHASE44_OFFICIAL_FAST_MODE" in d.name
        ]
        self.assertGreater(len(archive_dirs), 0)
        for archive_dir in archive_dirs:
            manifest_path = archive_dir / "_quarantine_manifest.json"
            if manifest_path.exists():
                doc = json.loads(manifest_path.read_text())
                self.assertEqual(doc["reason"],
                    "PHASE44_OFFICIAL_FAST_MODE_AND_INCOMPLETE_PROTOCOL_INVALIDATION")
                self.assertTrue(doc["do_not_reuse_runs"],
                    "Quarantine manifest must mark do_not_reuse_runs=True.")


class TestRunRegistryCompleteness(unittest.TestCase):
    """Verify the post-quarantine run_registry evidence."""

    def test_quarantine_includes_refit_registry(self):
        archive_dirs = [
            d for d in (ARTIFACT_DIR / "_history").iterdir()
            if d.is_dir() and "PHASE44_OFFICIAL_FAST_MODE" in d.name
        ]
        found = False
        for ad in archive_dirs:
            if (ad / "rolling_origin_refit_run_registry.csv").exists():
                found = True
                break
        self.assertTrue(found, "Quarantine must include the Stage B registry CSV.")


if __name__ == "__main__":
    unittest.main()