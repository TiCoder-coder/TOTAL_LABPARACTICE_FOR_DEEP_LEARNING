"""Unit tests for rolling_origin module.

Run with:
    pytest COURSE_WORK/src/course_work/rolling_origin/tests/
or:
    PYTHONPATH=COURSE_WORK/src python3 -m unittest \
        COURSE_WORK.src.course_work.rolling_origin.tests.unit.test_folds \
        ...
"""
from __future__ import annotations

import unittest
from pathlib import Path

from course_work.rolling_origin.folds import (
    FoldId,
    _chronological_key,
    _disjoint,
    _strict_temporal_ordering,
    build_rolling_folds,
    validate_fold_temporal_ordering,
)
from course_work.rolling_origin.persistence import (
    PERSISTENCE_MODEL_ID,
    build_prior_history_lookup,
    compute_persistence_bundle,
)
from course_work.rolling_origin.pooling import (
    compute_macro_metrics,
    compute_pooled_metrics,
)
from course_work.rolling_origin.populations import (
    ROBASE_VERSION,
    WINDOWPOP_VERSION,
    compute_population_fingerprint,
)
from course_work.rolling_origin.ranking import rank_transformers
from course_work.rolling_origin.scaling import (
    build_bundle,
    fit_fold_a_x_scaler,
    fit_fold_a_y_scaler,
    fit_fold_b_x_scaler,
    fit_fold_b_y_scaler,
)


class TestChronologicalKey(unittest.TestCase):
    def test_int_keys(self):
        self.assertEqual(_chronological_key(5), 5)

    def test_str_keys(self):
        self.assertEqual(_chronological_key("TGT_00000144"), 144)
        self.assertEqual(_chronological_key("99"), 99)
        self.assertEqual(_chronological_key("TGT_00000099"), 99)

    def test_temporal_ordering_strings(self):
        # Lex would put "99" > "240"; chronological should put 99 < 240
        self.assertTrue(_strict_temporal_ordering(["99"], ["240"]))
        self.assertTrue(
            _strict_temporal_ordering(
                ["TGT_00000080"], ["TGT_00000081", "TGT_00000099"]
            )
        )

    def test_disjoint(self):
        self.assertTrue(_disjoint(["a", "b"], ["c", "d"]))
        self.assertFalse(_disjoint(["a", "b"], ["b", "c"]))


class TestFolds(unittest.TestCase):
    def test_K3_folds_built(self):
        rtrn = [str(i) for i in range(240)]
        rval = [str(i) for i in range(240, 300)]
        folds = build_rolling_folds(rtrn, rval, k=3)
        self.assertEqual(len(folds), 3)
        self.assertEqual([str(f.fold_id) for f in folds], ["RO1", "RO2", "RO3"])

    def test_K3_fold_population_sizes(self):
        rtrn = [str(i) for i in range(240)]
        rval = [str(i) for i in range(240, 300)]
        folds = build_rolling_folds(rtrn, rval, k=3)
        # RO1
        self.assertEqual(len(folds[0].inner_train_ids), 220)
        self.assertEqual(len(folds[0].inner_val_ids), 20)
        self.assertEqual(len(folds[0].outer_train_ids), 240)
        self.assertEqual(len(folds[0].outer_eval_ids), 20)
        # RO2
        self.assertEqual(len(folds[1].outer_eval_ids), 20)
        self.assertEqual(len(folds[1].outer_train_ids), 260)
        # RO3
        self.assertEqual(len(folds[2].outer_eval_ids), 20)
        self.assertEqual(len(folds[2].outer_train_ids), 280)

    def test_outer_eval_disjoint_and_covers_rval(self):
        rtrn = [str(i) for i in range(240)]
        rval = [str(i) for i in range(240, 300)]
        folds = build_rolling_folds(rtrn, rval, k=3)
        seen = set()
        for f in folds:
            for tid in f.outer_eval_ids:
                self.assertNotIn(tid, seen, f"Duplicate outer_eval {tid}")
                seen.add(tid)
        self.assertEqual(seen, set(rval))

    def test_temporal_ordering_rows(self):
        rtrn = [str(i) for i in range(240)]
        rval = [str(i) for i in range(240, 300)]
        folds = build_rolling_folds(rtrn, rval, k=3)
        rows = validate_fold_temporal_ordering(folds)
        self.assertEqual(len(rows), 3)
        for r in rows:
            self.assertTrue(r["temporal_ok"], f"Fold {r['fold_id']} temporal not OK")
            self.assertTrue(r["disjoint_ok"], f"Fold {r['fold_id']} disjoint not OK")

    def test_K_too_small_raises(self):
        rtrn = [str(i) for i in range(240)]
        rval = [str(i) for i in range(240, 300)]
        with self.assertRaises(ValueError):
            build_rolling_folds(rtrn, rval, k=1)

    def test_disjoint_rtrn_rval_raises(self):
        rtrn = [str(i) for i in range(10)]
        rval = [str(i) for i in range(5, 15)]
        with self.assertRaises(AssertionError):
            build_rolling_folds(rtrn, rval, k=3)


class TestPopulation(unittest.TestCase):
    def test_versions(self):
        self.assertEqual(ROBASE_VERSION, "ROBASE-v1")
        self.assertEqual(WINDOWPOP_VERSION, "WINDOWPOP-v1")

    def test_fingerprint_deterministic(self):
        ids = ["TGT_00000003", "TGT_00000001", "TGT_00000002"]
        fp1 = compute_population_fingerprint(ids)
        fp2 = compute_population_fingerprint(reversed(ids))
        self.assertEqual(fp1, fp2)
        self.assertEqual(len(fp1), 64)

    def test_fingerprint_changes_with_set(self):
        fp1 = compute_population_fingerprint(["1", "2", "3"])
        fp2 = compute_population_fingerprint(["1", "2", "4"])
        self.assertNotEqual(fp1, fp2)


class TestScalerBundle(unittest.TestCase):
    def test_x_scaler_fit_synthetic(self):
        X = __import__("numpy").zeros((10, 4), dtype="float32")
        means, stds, scaled, passthrough = fit_fold_a_x_scaler(
            X, feature_variant_id="FS0_TF0", feature_columns=["a", "b", "c", "d"]
        )
        self.assertEqual(len(means), 4)
        self.assertEqual(len(stds), 4)

    def test_y_scaler_YS0_is_identity(self):
        y = __import__("numpy").array([1.0, 2.0, 3.0])
        m, s = fit_fold_a_y_scaler(y, target_scaling_option="YS0")
        self.assertIsNone(m)
        self.assertIsNone(s)

    def test_y_scaler_YS1_standard(self):
        import numpy as np
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m, s = fit_fold_a_y_scaler(y, target_scaling_option="YS1")
        self.assertAlmostEqual(m, 3.0)
        self.assertGreater(s, 0.0)

    def test_bundle_checksum_changes_with_content(self):
        b1 = build_bundle(
            bundle_id="B1",
            fit_stage="A",
            fold_id="RO1",
            candidate_id="TR_C0",
            target_scaling_option="YS1",
            feature_variant_id="FS2_TF1",
            lookback_steps=36,
            boundary_protocol="WB0_CONTEXT_CARRY_OVER",
            revin_enabled=False,
            fit_target_ids=["TGT_00000001"],
            fit_raw_row_count=1,
            x_means=None,
            x_stds=None,
            feature_indices_scaled=[],
            feature_indices_passthrough=[],
            y_mean=None,
            y_std=None,
        )
        b2 = build_bundle(
            bundle_id="B1",
            fit_stage="A",
            fold_id="RO1",
            candidate_id="TR_C1",
            target_scaling_option="YS1",
            feature_variant_id="FS2_TF1",
            lookback_steps=36,
            boundary_protocol="WB0_CONTEXT_CARRY_OVER",
            revin_enabled=False,
            fit_target_ids=["TGT_00000001"],
            fit_raw_row_count=1,
            x_means=None,
            x_stds=None,
            feature_indices_scaled=[],
            feature_indices_passthrough=[],
            y_mean=None,
            y_std=None,
        )
        self.assertNotEqual(b1.checksum(), b2.checksum())


class TestPersistence(unittest.TestCase):
    def test_persistence_model_id(self):
        self.assertEqual(PERSISTENCE_MODEL_ID, "PERSISTENCE_LAST_VALUE")

    def test_compute_persistence_bundle_basic(self):
        # Outer block: t0=10:00, t1=10:10, t2=10:20 with y_true 5,7,3
        # Prior history: 9:50=1, 9:40=2, 9:30=4
        # Predictions:
        #   for t0=10:00 -> prior 9:50 -> 1
        #   for t1=10:10 -> prior 10:00 -> 5
        #   for t2=10:20 -> prior 10:10 -> 7
        lookup = {
            "2016-01-01 09:30:00": 4.0,
            "2016-01-01 09:40:00": 2.0,
            "2016-01-01 09:50:00": 1.0,
            "2016-01-01 10:00:00": 5.0,
            "2016-01-01 10:10:00": 7.0,
        }
        bundle = compute_persistence_bundle(
            fold_id="RO1",
            fold_outer_eval_target_ids=[100, 101, 102],
            fold_outer_eval_target_timestamps=[
                "2016-01-01 10:00:00",
                "2016-01-01 10:10:00",
                "2016-01-01 10:20:00",
            ],
            fold_outer_eval_y_true_wh=[5.0, 7.0, 3.0],
            prior_lookup=lookup,
        )
        self.assertEqual(bundle.y_pred_wh, (1.0, 5.0, 7.0))
        self.assertEqual(bundle.y_true_wh, (5.0, 7.0, 3.0))

    def test_build_prior_history_lookup_dedupes(self):
        import pandas as pd
        df = pd.DataFrame(
            {
                "target_timestamp": ["2016-01-01 10:00:00", "2016-01-01 10:00:00", "2016-01-01 10:10:00"],
                "Appliances": [1.0, 5.0, 7.0],  # last value wins
            }
        )
        lookup = build_prior_history_lookup(df, value_column="Appliances")
        # Last value for 10:00:00 is 5.0
        self.assertEqual(lookup["2016-01-01 10:00:00"], 5.0)


class TestPooling(unittest.TestCase):
    def test_pooled_metrics_concatenated_residuals(self):
        import numpy as np
        yt = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])]
        yp = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0])]
        pm = compute_pooled_metrics(
            candidate_id="TR_C0", y_true_per_fold=yt, y_pred_per_fold=yp
        )
        # residuals = [1, 2, 3, 4, 5], mae=3.0, rmse=sqrt((1+4+9+16+25)/5)=sqrt(11)
        self.assertAlmostEqual(pm.pooled_mae_wh, 3.0)
        self.assertAlmostEqual(pm.pooled_rmse_wh, (11.0) ** 0.5)

    def test_pooled_metrics_not_macro(self):
        # If we had used "mean of fold RMSEs" we'd get sqrt(14/3)=2.16
        # but pooled rmse from concatenated is sqrt(11)=3.32
        import numpy as np
        yt = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])]
        yp = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0])]
        pm = compute_pooled_metrics(
            candidate_id="X", y_true_per_fold=yt, y_pred_per_fold=yp
        )
        # sqrt(11) ≈ 3.32, NOT sqrt(14/3) ≈ 2.16
        self.assertNotAlmostEqual(pm.pooled_rmse_wh, ((1 + 4 + 9) / 3) ** 0.5)

    def test_macro_metrics(self):
        mm = compute_macro_metrics(
            candidate_id="X", fold_rmse_wh=[1.0, 2.0, 3.0], fold_mae_wh=[0.5, 1.5, 2.5]
        )
        self.assertAlmostEqual(mm.macro_mae_wh, (0.5 + 1.5 + 2.5) / 3)
        self.assertAlmostEqual(mm.worst_fold_rmse_wh, 3.0)
        self.assertAlmostEqual(mm.best_fold_rmse_wh, 1.0)


class TestRanking(unittest.TestCase):
    def test_rank_transformers_skips_persistence(self):
        from course_work.rolling_origin.pooling import (
            PooledMetrics,
            MacroRobustnessMetrics,
        )
        from course_work.rolling_origin.ranking import TransformerRankingEntry

        pooled = {
            "TR_C0": PooledMetrics(
                candidate_id="TR_C0", fold_count=3,
                pooled_mae_wh=1.0, pooled_rmse_wh=2.0, pooled_r2=0.5,
            ),
            "TR_C1": PooledMetrics(
                candidate_id="TR_C1", fold_count=3,
                pooled_mae_wh=1.5, pooled_rmse_wh=2.5, pooled_r2=0.4,
            ),
            "PERSISTENCE_LAST_VALUE": PooledMetrics(
                candidate_id="PERSISTENCE_LAST_VALUE", fold_count=3,
                pooled_mae_wh=2.0, pooled_rmse_wh=3.0, pooled_r2=0.0,
            ),
        }
        macro = {
            cid: MacroRobustnessMetrics(
                candidate_id=cid, fold_count=3,
                macro_mae_wh=1.0, macro_rmse_wh=2.0,
                worst_fold_rmse_wh=2.5, best_fold_rmse_wh=1.5, fold_rmse_sd_wh=0.5,
            )
            for cid in pooled
        }
        families = {
            "TR_C0": "TRANSFORMER_ENCODER",
            "TR_C1": "TRANSFORMER_ENCODER",
            "PERSISTENCE_LAST_VALUE": "PERSISTENCE",
        }
        ranking = rank_transformers(
            candidate_pooled=pooled,
            candidate_macro=macro,
            shortlist_position={"TR_C0": 0, "TR_C1": 1},
            candidate_families=families,
        )
        self.assertEqual(len(ranking), 2)
        self.assertEqual(ranking[0].candidate_id, "TR_C0")
        self.assertEqual(ranking[1].candidate_id, "TR_C1")
        for r in ranking:
            self.assertIsInstance(r, TransformerRankingEntry)

    def test_rank_transformers_tiebreak(self):
        from course_work.rolling_origin.pooling import (
            PooledMetrics,
            MacroRobustnessMetrics,
        )
        # Two TR candidates with IDENTICAL pooled_rmse; tie-break by shortlist_position
        pooled = {
            "TR_C0": PooledMetrics("TR_C0", 3, 1.0, 2.0, 0.5),
            "TR_C1": PooledMetrics("TR_C1", 3, 1.0, 2.0, 0.5),
        }
        macro = {
            "TR_C0": MacroRobustnessMetrics("TR_C0", 3, 1.0, 2.0, 2.5, 1.5, 0.5),
            "TR_C1": MacroRobustnessMetrics("TR_C1", 3, 1.0, 2.0, 2.5, 1.5, 0.5),
        }
        ranking = rank_transformers(
            candidate_pooled=pooled,
            candidate_macro=macro,
            shortlist_position={"TR_C0": 0, "TR_C1": 1},
        )
        # TR_C0 has earlier shortlist_position -> rank 1
        self.assertEqual(ranking[0].candidate_id, "TR_C0")


if __name__ == "__main__":
    unittest.main()