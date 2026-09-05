"""Phase 44 — Consistency checks that gate the signoff.

`run_all_consistency_checks` returns a list of CheckResult objects.
The signoff writer (O44.39) calls this and only emits `status: PASS`
if every check has `passed=True`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CheckResult:
    check_id: str
    description: str
    passed: bool
    details: dict[str, Any]


@dataclass
class Phase44Consistency:
    results: list[CheckResult]

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed]


def run_all_consistency_checks(
    *,
    folds,
    inner_selection_registry_rows: list[dict],
    refit_run_registry_rows: list[dict],
    pooled_predictions_df,
    candidate_pooled_metrics: dict,
    transformer_ranking_rows: list[dict],
    scaler_fit_audit_rows: list[dict],
    inner_best_epoch_rows: list[dict],
    lstm_winner_fingerprint: str,
    phase43_signoff_fingerprint: str,
    candidate_compatibility_rows: list[dict],
    recommended_transformer_id: str | None,
    o44_files_present: dict[str, bool],
    test_firewall_passed: bool,
) -> Phase44Consistency:
    results: list[CheckResult] = []

    # 1. Phase 43 fingerprint match
    results.append(
        CheckResult(
            check_id="C01_phase43_fingerprint",
            description="Corrected Phase 43 winner fingerprint matches LSTM handoff",
            passed=(lstm_winner_fingerprint == phase43_signoff_fingerprint and lstm_winner_fingerprint != ""),
            details={
                "lstm_winner_fingerprint": lstm_winner_fingerprint,
                "phase43_signoff_fingerprint": phase43_signoff_fingerprint,
            },
        )
    )

    # 2. K = 3 folds
    results.append(
        CheckResult(
            check_id="C02_K_folds",
            description="K = 3 fold construction",
            passed=(len(folds) == 3),
            details={"n_folds": len(folds)},
        )
    )

    # 3. Temporal ordering across all folds
    temporal_ok = True
    for f in folds:
        if not (f.inner_train_ids and f.inner_val_ids and f.outer_eval_ids):
            temporal_ok = False
            break
        if max(f.inner_train_ids) >= min(f.inner_val_ids):
            temporal_ok = False
            break
        if max(f.inner_val_ids) >= min(f.outer_eval_ids):
            temporal_ok = False
            break
    results.append(
        CheckResult(
            check_id="C03_temporal_ordering",
            description="max(inner_train) < min(inner_val) < min(outer_eval) for every fold",
            passed=temporal_ok,
            details={},
        )
    )

    # 4. Inner selection registry: every (candidate, fold) present once
    expected_inner_keys = set()
    for f in folds:
        for cid in candidate_pooled_metrics.keys():
            expected_inner_keys.add((cid, str(f.fold_id)))
    seen_inner_keys = set(
        (r["candidate_id"], r["fold_id"])
        for r in inner_selection_registry_rows
    )
    results.append(
        CheckResult(
            check_id="C04_inner_selection_completeness",
            description="All (candidate, fold) pairs have an inner-selection registry row",
            passed=(seen_inner_keys == expected_inner_keys),
            details={"expected": len(expected_inner_keys), "seen": len(seen_inner_keys)},
        )
    )

    # 5. Refit registry: every (candidate, fold) present once
    expected_refit_keys = set()
    for f in folds:
        for cid in candidate_pooled_metrics.keys():
            expected_refit_keys.add((cid, str(f.fold_id)))
    seen_refit_keys = set(
        (r["candidate_id"], r["fold_id"])
        for r in refit_run_registry_rows
    )
    results.append(
        CheckResult(
            check_id="C05_refit_completeness",
            description="All (candidate, fold) pairs have a refit registry row",
            passed=(seen_refit_keys == expected_refit_keys),
            details={"expected": len(expected_refit_keys), "seen": len(seen_refit_keys)},
        )
    )

    # 6. Pooled predictions unique target_ids per fold
    if pooled_predictions_df is None or pooled_predictions_df.empty:
        unique_ok = False
    else:
        unique_ok = True
        for fold_id in pooled_predictions_df["fold_id"].unique():
            sub = pooled_predictions_df[pooled_predictions_df["fold_id"] == fold_id]
            # Use model_id if present, otherwise candidate_id
            if "model_id" in sub.columns:
                id_col = sub["model_id"]
                cid_list = id_col.unique()
            else:
                cid_list = sub["candidate_id"].unique() if "candidate_id" in sub.columns else []
            for cid in cid_list:
                sub2 = sub[sub["model_id"] == cid] if "model_id" in sub.columns else sub[sub["candidate_id"] == cid]
                if sub2["target_id"].duplicated().any():
                    unique_ok = False
                    break
            if not unique_ok:
                break
    results.append(
        CheckResult(
            check_id="C06_pooled_predictions_unique",
            description="Per-fold per-candidate target_ids are unique",
            passed=unique_ok,
            details={},
        )
    )

    # 7. All Transformer candidates ranked
    expected_tr_count = sum(1 for cid in candidate_pooled_metrics if cid.startswith("TR_"))
    results.append(
        CheckResult(
            check_id="C07_transformer_ranking_count",
            description="All Transformer candidates appear in transformer ranking",
            passed=(len(transformer_ranking_rows) == expected_tr_count),
            details={"expected_tr_count": expected_tr_count, "seen": len(transformer_ranking_rows)},
        )
    )

    # 8. Scaler-fit audit: 0 outer-eval rows, 0 test rows, status PASS
    scaler_ok = all(
        int(r.get("outer_eval_rows_used", -1)) == 0
        and int(r.get("test_rows_used", -1)) == 0
        and r.get("status") == "PASS"
        for r in scaler_fit_audit_rows
    )
    results.append(
        CheckResult(
            check_id="C08_scaler_fit_audit",
            description="All scaler-fit audits have 0 outer-eval rows and 0 test rows",
            passed=scaler_ok,
            details={"audit_count": len(scaler_fit_audit_rows)},
        )
    )

    # 9. Inner best-epoch entries match refit epoch counts
    epoch_match_ok = True
    for inner_row in inner_best_epoch_rows:
        cid = inner_row["candidate_id"]
        fid = inner_row["fold_id"]
        match = [
            r for r in refit_run_registry_rows
            if r["candidate_id"] == cid and r["fold_id"] == fid
        ]
        if not match:
            epoch_match_ok = False
            break
        best_epoch = int(inner_row["best_epoch_inner"])
        official_epoch = int(match[0]["official_epoch"])
        if best_epoch != official_epoch:
            epoch_match_ok = False
            break
    results.append(
        CheckResult(
            check_id="C09_inner_refit_epoch_match",
            description="best_epoch_inner == official_epoch for every (candidate, fold)",
            passed=epoch_match_ok,
            details={},
        )
    )

    # 10. Candidate compatibility
    compat_ok = all(
        r.get("status") == "PASS" and r.get("all_ok") in (True, "True", "true")
        for r in candidate_compatibility_rows
    )
    results.append(
        CheckResult(
            check_id="C10_candidate_compatibility",
            description="All (candidate, fold) pairs are fold-compatible",
            passed=compat_ok,
            details={},
        )
    )

    # 11. Recommended transformer exists and is the top of ranking
    rec_ok = (
        recommended_transformer_id is not None
        and len(transformer_ranking_rows) > 0
        and transformer_ranking_rows[0].get("candidate_id") == recommended_transformer_id
    )
    results.append(
        CheckResult(
            check_id="C11_recommended_transformer_consistent",
            description="recommended_transformer matches top of transformer ranking",
            passed=rec_ok,
            details={"recommended": recommended_transformer_id, "top": transformer_ranking_rows[0].get("candidate_id") if transformer_ranking_rows else None},
        )
    )

    # 12. All O44 files present
    missing = [k for k, present in o44_files_present.items() if not present]
    results.append(
        CheckResult(
            check_id="C12_o44_files_present",
            description="All O44.1-O44.39 files exist on disk",
            passed=(not missing),
            details={"missing": missing, "checked": len(o44_files_present)},
        )
    )

    # 13. Test firewall
    results.append(
        CheckResult(
            check_id="C13_test_firewall",
            description="No Test rows used in any scaler or fold population",
            passed=test_firewall_passed,
            details={},
        )
    )

    return Phase44Consistency(results=results)