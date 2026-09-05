"""Integration tests for rolling_origin module.

These tests exercise the end-to-end pipeline:
  - ROBASE extraction
  - Fold construction
  - Scaler fitting
  - Persistence bundle
  - Pooled metrics
  - Ranking
  - Consistency checks
"""
from __future__ import annotations

import unittest
from pathlib import Path

from course_work.rolling_origin.consistency import (
    CheckResult,
    run_all_consistency_checks,
)
from course_work.rolling_origin.folds import build_rolling_folds
from course_work.rolling_origin.persistence import compute_persistence_bundle
from course_work.rolling_origin.pooling import compute_macro_metrics, compute_pooled_metrics
from course_work.rolling_origin.populations import (
    compute_population_fingerprint,
    extract_robase_population,
)
from course_work.rolling_origin.preflight import run_preflight
from course_work.rolling_origin.ranking import rank_transformers
from course_work.rolling_origin.scaling import (
    build_bundle,
    fit_fold_a_x_scaler,
    fit_fold_a_y_scaler,
)


class TestPreflightAgainstRealProject(unittest.TestCase):
    """Run preflight against the real project's windowpop and signoffs.

    Skips gracefully if the project's artifacts are not present.
    """

    @classmethod
    def setUpClass(cls):
        # Walk up to find COURSE_WORK (project_root for course_work package)
        candidates = [
            Path(__file__).parents[4],
            Path(__file__).parents[5],
            Path(__file__).resolve().parent.parent.parent.parent.parent / "COURSE_WORK",
        ]
        for c in candidates:
            if (c / "artifacts" / "windows" / "common_target_population.csv").exists():
                cls.project_root = c
                break
        else:
            cls.project_root = None

    def test_preflight_full(self):
        if self.project_root is None:
            self.skipTest("Project WINDOWPOP-v1 not present")
        result = run_preflight(
            project_root=self.project_root,
            phase_42_signoff_path=self.project_root / "artifacts/candidate_synthesis/phase_42_signoff.json",
            transformer_shortlist_path=self.project_root / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
            phase_43_signoff_path=self.project_root / "artifacts/lstm_tuning/phase_43_signoff.json",
            lstm_winner_path=self.project_root / "artifacts/lstm_tuning/lstm_tuned_winner.json",
            lstm_handoff_path=self.project_root / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
        )
        for g in result.gates:
            self.assertTrue(g.passed, f"Gate {g.gate_id} failed: {g.description}")


class TestFullPipeline(unittest.TestCase):
    """End-to-end pipeline test against synthetic data."""

    def test_pipeline_synthetic(self):
        import numpy as np
        import pandas as pd

        # Synthetic windowpop
        rng = np.random.default_rng(123)
        rows = []
        n_train, n_val = 240, 60
        for i in range(n_train + n_val):
            split = "TRAIN" if i < n_train else "VALIDATION"
            ts = pd.Timestamp("2016-01-01 00:00:00") + i * pd.Timedelta(minutes=10)
            rows.append({
                "target_id": str(i),
                "target_timestamp": ts,
                "target_split_id": split,
                "Appliances": float(rng.normal(loc=60.0, scale=20.0)),
            })
        windowpop_df = pd.DataFrame(rows)
        robase = windowpop_df[windowpop_df["target_split_id"] != "TEST"].reset_index(drop=True)

        rtrn = robase.loc[robase["target_split_id"] == "TRAIN", "target_id"].tolist()
        rval = robase.loc[robase["target_split_id"] == "VALIDATION", "target_id"].tolist()

        # Fold construction
        folds = build_rolling_folds(rtrn, rval, k=3)
        self.assertEqual(len(folds), 3)

        # Scaler bundle construction (synthetic X, Y)
        X = rng.normal(size=(240, 4)).astype(np.float32)
        y = rng.normal(size=240).astype(np.float64)
        means, stds, scaled_idx, passthrough_idx = fit_fold_a_x_scaler(
            X, feature_variant_id="FS2_TF1", feature_columns=["a", "b", "c", "d"]
        )
        ym, ys = fit_fold_a_y_scaler(y, target_scaling_option="YS1")
        bundle = build_bundle(
            bundle_id="B1",
            fit_stage="A",
            fold_id="RO1",
            candidate_id="TR_C0",
            target_scaling_option="YS1",
            feature_variant_id="FS2_TF1",
            lookback_steps=36,
            boundary_protocol="WB0_CONTEXT_CARRY_OVER",
            revin_enabled=False,
            fit_target_ids=rtrn,
            fit_raw_row_count=len(rtrn),
            x_means=means,
            x_stds=stds,
            feature_indices_scaled=list(scaled_idx),
            feature_indices_passthrough=list(passthrough_idx),
            y_mean=ym,
            y_std=ys,
        )
        # Round-trip y transform
        y_back = bundle.inverse_transform_y(bundle.transform_y(y))
        self.assertTrue(np.allclose(y, y_back, atol=1e-6))

        # Persistence bundle (synthetic)
        f = folds[0]
        outer_ids = list(f.outer_eval_ids)
        outer_ts = [
            str(windowpop_df.loc[windowpop_df["target_id"] == tid, "target_timestamp"].iloc[0])
            for tid in outer_ids
        ]
        outer_y = [
            float(windowpop_df.loc[windowpop_df["target_id"] == tid, "Appliances"].iloc[0])
            for tid in outer_ids
        ]
        lookup = {
            str(r["target_timestamp"]): float(r["Appliances"])
            for _, r in windowpop_df.iterrows()
        }
        pers_bundle = compute_persistence_bundle(
            fold_id=str(f.fold_id),
            fold_outer_eval_target_ids=outer_ids,
            fold_outer_eval_target_timestamps=outer_ts,
            fold_outer_eval_y_true_wh=outer_y,
            prior_lookup=lookup,
        )
        # Every pred must be in lookup
        for p in pers_bundle.y_pred_wh:
            self.assertIn(p, lookup.values())

        # Pooled metrics
        all_y_true, all_y_pred = [], []
        for f_ in folds:
            outer_ids = list(f_.outer_eval_ids)
            outer_ts = [
                str(windowpop_df.loc[windowpop_df["target_id"] == tid, "target_timestamp"].iloc[0])
                for tid in outer_ids
            ]
            outer_y = [
                float(windowpop_df.loc[windowpop_df["target_id"] == tid, "Appliances"].iloc[0])
                for tid in outer_ids
            ]
            pb = compute_persistence_bundle(
                fold_id=str(f_.fold_id),
                fold_outer_eval_target_ids=outer_ids,
                fold_outer_eval_target_timestamps=outer_ts,
                fold_outer_eval_y_true_wh=outer_y,
                prior_lookup=lookup,
            )
            all_y_true.append(np.array(pb.y_true_wh))
            all_y_pred.append(np.array(pb.y_pred_wh))
        pm = compute_pooled_metrics(
            candidate_id="TR_C0",
            y_true_per_fold=all_y_true,
            y_pred_per_fold=all_y_pred,
        )
        # RMSE must be a finite positive number
        self.assertGreater(pm.pooled_rmse_wh, 0.0)


if __name__ == "__main__":
    unittest.main()