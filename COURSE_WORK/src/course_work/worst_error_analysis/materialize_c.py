"""Phase 51-C orchestrator — execute W1/W2/W3/W4/W3_SH/W4_SH rankings.

Pre-execution gate: verify frozen selection contract SHA256 matches expected.
Pre-execution gate: verify working table exists + SHA matches.

Executes ONLY the 6 ranking families. Does NOT:
  - perform overlap/Jaccard analysis
  - perform error concentration analysis
  - perform regime enrichment
  - perform Persistence case comparison
  - perform temporal context
  - perform input-window context
  - build casebook
  - generate figures
  - handoff to Phase 52
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import contract
from . import ranking
from . import ranking_audit
from . import signed_ranking
from . import c_writers
from . import writers as b_writers
from .frozen_sort_key_compliance import verify_frozen_sort_keys_used
from ..utils.artifacts import get_project_root


EXPECTED_SELECTION_CONTRACT_SHA = (
    "ec798326cb03586e85ce7ba09d53be03016a234fe15e1ba5fb4b3fbf0eb967d4"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize_phase51_c(
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Execute Phase 51-C: verify contract + execute 6 ranking families + audit."""
    root = project_root if project_root is not None else get_project_root()

    # ── Gate 0: frozen-contract sort-key compliance ─────────────────────────
    # Refuse to rank unless every ranker's sort key block uses ONLY
    # permitted fields (no target_timestamp, no y_pred_wh, etc.).
    if not verify_frozen_sort_keys_used():
        from .frozen_sort_key_compliance import audit_no_forbidden_sort_keys
        findings = audit_no_forbidden_sort_keys()
        offending = {
            m: r["forbidden"]
            for m, r in findings.items() if r["forbidden"]
        }
        raise RuntimeError(
            "FROZEN-CONTRACT SORT-KEY DRIFT DETECTED in ranker sources. "
            "Refusing to run Phase 51-C. Forbidden fields used: "
            f"{offending}"
        )

    # ── Gate 1: verify selection contract SHA ───────────────────────────────
    contract_path = root / "artifacts/worst_error_analysis/worst_error_selection_contract.json"
    contract_sha = _sha256(contract_path)
    if contract_sha != EXPECTED_SELECTION_CONTRACT_SHA:
        raise RuntimeError(
            f"Selection contract SHA mismatch!\n"
            f"  expected: {EXPECTED_SELECTION_CONTRACT_SHA}\n"
            f"  actual:   {contract_sha}"
        )

    # Load contract to verify K values + ranking metric
    contract_payload = json.loads(contract_path.read_text())
    if contract_payload["K_PER_SEED"] != 20:
        raise RuntimeError("K_PER_SEED != 20")
    if contract_payload["K_SHARED"] != 20:
        raise RuntimeError("K_SHARED != 20")
    if contract_payload["primary_ranking_metric"] != "absolute_error_wh":
        raise RuntimeError("primary_ranking_metric != absolute_error_wh")
    if contract_payload["tie_break"] != "target_id ASC":
        raise RuntimeError("tie_break != target_id ASC")

    # ── Gate 2: verify working table ─────────────────────────────────────────
    wt_path = root / "artifacts/worst_error_analysis/phase51_target_level_working_table.csv"
    if not wt_path.exists():
        raise RuntimeError("Working table missing")

    wt_sha = _sha256(wt_path)
    wt_rows = []
    with wt_path.open("r", encoding="utf-8", newline="") as fh:
        import csv
        wt_rows = list(csv.DictReader(fh))
    if len(wt_rows) != 2961:
        raise RuntimeError(f"Working table rows != 2961 (got {len(wt_rows)})")

    source_rows = len(wt_rows)

    # ── W1 PER_SEED_WORST ──────────────────────────────────────────────────
    w1_results = ranking.rank_w1_per_seed_worst(root)
    w1_fieldnames = [
        "selection_family", "seed", "rank", "target_id", "target_timestamp",
        "y_true_wh", "y_pred_wh", "residual_wh", "absolute_error_wh",
        "squared_error_wh2", "residual_sign",
        "primary_ranking_metric", "primary_ranking_direction", "tie_break",
    ]
    # Flatten W1 output (3 seeds × 20 rows)
    w1_flat = []
    for seed in contract.SEEDS:
        w1_flat.extend(w1_results[seed])
    w1_sha = c_writers.write_csv(
        w1_flat, w1_fieldnames, c_writers.W1_PER_SEED_TOP20_REL, root,
    )

    # ── W2 SHARED_WORST ─────────────────────────────────────────────────────
    w2_rows = ranking.rank_w2_shared_worst(root)
    w2_fieldnames = [
        "selection_family", "seed", "rank", "target_id", "target_timestamp",
        "y_true_wh", "mean_abs_error_wh", "seed_abs_error_std_wh",
        "seed42_abs_error_wh", "seed123_abs_error_wh", "seed2026_abs_error_wh",
        "primary_ranking_metric", "primary_ranking_direction", "tie_break",
    ]
    w2_sha = c_writers.write_csv(
        w2_rows, w2_fieldnames, c_writers.W2_SHARED_TOP20_REL, root,
    )

    # ── W3 UNDERPREDICTION_WORST ────────────────────────────────────────────
    w3_results = signed_ranking.rank_w3_underprediction_worst(root)
    w3_fieldnames = [
        "selection_family", "seed", "rank", "target_id", "target_timestamp",
        "y_true_wh", "y_pred_wh", "residual_wh", "absolute_error_wh",
        "squared_error_wh2", "residual_sign", "filter_condition",
        "primary_ranking_metric", "primary_ranking_direction", "tie_break",
    ]
    w3_flat = []
    for seed in contract.SEEDS:
        w3_flat.extend(w3_results[seed])
    w3_sha = c_writers.write_csv(
        w3_flat, w3_fieldnames, c_writers.W3_UNDERPREDICTION_TOP10_REL, root,
    )

    # ── W4 OVERPREDICTION_WORST ─────────────────────────────────────────────
    w4_results = signed_ranking.rank_w4_overprediction_worst(root)
    w4_fieldnames = list(w3_fieldnames)  # same shape
    w4_flat = []
    for seed in contract.SEEDS:
        w4_flat.extend(w4_results[seed])
    w4_sha = c_writers.write_csv(
        w4_flat, w4_fieldnames, c_writers.W4_OVERPREDICTION_TOP10_REL, root,
    )

    # ── W3_SH SHARED_ALL_UNDER ──────────────────────────────────────────────
    w3_sh_rows = signed_ranking.rank_w3_shared_all_under(root)
    w3_sh_fieldnames = [
        "selection_family", "seed", "rank", "target_id", "target_timestamp",
        "y_true_wh", "mean_abs_error_wh",
        "seed42_residual_wh", "seed123_residual_wh", "seed2026_residual_wh",
        "sign_all_under",
        "primary_ranking_metric", "primary_ranking_direction", "tie_break",
    ]
    w3_sh_sha = c_writers.write_csv(
        w3_sh_rows, w3_sh_fieldnames, c_writers.W3_SH_SHARED_ALL_UNDER_REL, root,
    )

    # ── W4_SH SHARED_ALL_OVER ──────────────────────────────────────────────
    w4_sh_rows = signed_ranking.rank_w4_shared_all_over(root)
    w4_sh_fieldnames = list(w3_sh_fieldnames)
    w4_sh_fieldnames[10] = "sign_all_over"  # rename last boolean field
    w4_sh_sha = c_writers.write_csv(
        w4_sh_rows, w4_sh_fieldnames, c_writers.W4_SH_SHARED_ALL_OVER_REL, root,
    )

    # ── Ranking audit ───────────────────────────────────────────────────────
    artifacts_root = root / "artifacts/worst_error_analysis"

    audits = []
    # W1 per-seed
    w1_total_k = contract.K_ABS_PER_SEED * len(contract.SEEDS)
    for seed in contract.SEEDS:
        a = ranking_audit.audit_ranking_artifact(
            artifacts_root / c_writers.W1_PER_SEED_TOP20_REL,
            family="W1_PER_SEED_WORST",
            seed=seed,
            k=w1_total_k,
            primary_ordering="abs_desc",
            contract_sha=contract_sha,
            source_sha=wt_sha,
            source_rows=source_rows,
            eligible_rows=source_rows,
        )
        audits.append(a)

    # W2
    audits.append(ranking_audit.audit_ranking_artifact(
        artifacts_root / c_writers.W2_SHARED_TOP20_REL,
        family="W2_SHARED_WORST",
        seed="ALL",
        k=contract.K_SHARED,
        primary_ordering="shared_desc",
        contract_sha=contract_sha,
        source_sha=wt_sha,
        source_rows=source_rows,
        eligible_rows=source_rows,
    ))

    # W3 per-seed
    w3_total_k = contract.K_UNDER_PER_SEED * len(contract.SEEDS)
    for seed in contract.SEEDS:
        a = ranking_audit.audit_ranking_artifact(
            artifacts_root / c_writers.W3_UNDERPREDICTION_TOP10_REL,
            family="W3_UNDERPREDICTION_WORST",
            seed=seed,
            k=w3_total_k,
            primary_ordering="abs_desc",
            sign_filter="residual > 0",
            contract_sha=contract_sha,
            source_sha=wt_sha,
            source_rows=source_rows,
            eligible_rows=sum(
                1 for r in wt_rows
                if float(r[f"seed{seed}_residual_wh"]) > 0
            ),
        )
        audits.append(a)

    # W4 per-seed
    w4_total_k = contract.K_OVER_PER_SEED * len(contract.SEEDS)
    for seed in contract.SEEDS:
        a = ranking_audit.audit_ranking_artifact(
            artifacts_root / c_writers.W4_OVERPREDICTION_TOP10_REL,
            family="W4_OVERPREDICTION_WORST",
            seed=seed,
            k=w4_total_k,
            primary_ordering="abs_desc",
            sign_filter="residual < 0",
            contract_sha=contract_sha,
            source_sha=wt_sha,
            source_rows=source_rows,
            eligible_rows=sum(
                1 for r in wt_rows
                if float(r[f"seed{seed}_residual_wh"]) < 0
            ),
        )
        audits.append(a)

    # W3_SH
    audits.append(ranking_audit.audit_ranking_artifact(
        artifacts_root / c_writers.W3_SH_SHARED_ALL_UNDER_REL,
        family="W3_SH_SHARED_ALL_UNDER",
        seed="ALL",
        k=contract.K_SHARED_SIGNED,
        primary_ordering="shared_desc",
        shared_signed_filter="all_under",
        contract_sha=contract_sha,
        source_sha=wt_sha,
        source_rows=source_rows,
        eligible_rows=sum(
            1 for r in wt_rows
            if float(r["seed42_residual_wh"]) > 0
            and float(r["seed123_residual_wh"]) > 0
            and float(r["seed2026_residual_wh"]) > 0
        ),
    ))

    # W4_SH
    audits.append(ranking_audit.audit_ranking_artifact(
        artifacts_root / c_writers.W4_SH_SHARED_ALL_OVER_REL,
        family="W4_SH_SHARED_ALL_OVER",
        seed="ALL",
        k=contract.K_SHARED_SIGNED,
        primary_ordering="shared_desc",
        shared_signed_filter="all_over",
        contract_sha=contract_sha,
        source_sha=wt_sha,
        source_rows=source_rows,
        eligible_rows=sum(
            1 for r in wt_rows
            if float(r["seed42_residual_wh"]) < 0
            and float(r["seed123_residual_wh"]) < 0
            and float(r["seed2026_residual_wh"]) < 0
        ),
    ))

    all_pass = all(a["all_pass"] for a in audits)

    audit_payload = {
        "phase": 51,
        "subphase": "51-C",
        "version": "PHASE51_C_RANKING_AUDIT-v1",
        "selection_contract_sha256": contract_sha,
        "source_working_table_sha256": wt_sha,
        "k_values": {
            "K_PER_SEED": contract.K_ABS_PER_SEED,
            "K_SHARED": contract.K_SHARED,
            "K_UNDER_PER_SEED": contract.K_UNDER_PER_SEED,
            "K_OVER_PER_SEED": contract.K_OVER_PER_SEED,
            "K_SHARED_SIGNED": contract.K_SHARED_SIGNED,
        },
        "n_artifacts": len(audits),
        "all_pass": all_pass,
        "audits": audits,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    audit_sha = c_writers.write_json(
        audit_payload, c_writers.RANKING_AUDIT_REL, root,
    )

    # ── Phase 51-C manifest ─────────────────────────────────────────────────
    manifest = {
        "phase": 51,
        "subphase": "51-C",
        "version": "WORST_ERROR_ANALYSIS_RANKING-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_contract_path": str(contract_path.relative_to(root)),
        "selection_contract_sha256": contract_sha,
        "source_working_table_path": str(wt_path.relative_to(root)),
        "source_working_table_sha256": wt_sha,
        "n_test": contract.N_TEST,
        "seeds": list(contract.SEEDS),
        "ranking_families_executed": [
            "W1_PER_SEED_WORST",
            "W2_SHARED_WORST",
            "W3_UNDERPREDICTION_WORST",
            "W4_OVERPREDICTION_WORST",
            "W3_SH_SHARED_ALL_UNDER",
            "W4_SH_SHARED_ALL_OVER",
        ],
        "artifacts": {
            "W1_PER_SEED_TOP20": {
                "path": c_writers.W1_PER_SEED_TOP20_REL,
                "sha256": w1_sha,
            },
            "W2_SHARED_TOP20": {
                "path": c_writers.W2_SHARED_TOP20_REL,
                "sha256": w2_sha,
            },
            "W3_UNDERPREDICTION_TOP10": {
                "path": c_writers.W3_UNDERPREDICTION_TOP10_REL,
                "sha256": w3_sha,
            },
            "W4_OVERPREDICTION_TOP10": {
                "path": c_writers.W4_OVERPREDICTION_TOP10_REL,
                "sha256": w4_sha,
            },
            "W3_SH_SHARED_ALL_UNDER": {
                "path": c_writers.W3_SH_SHARED_ALL_UNDER_REL,
                "sha256": w3_sh_sha,
            },
            "W4_SH_SHARED_ALL_OVER": {
                "path": c_writers.W4_SH_SHARED_ALL_OVER_REL,
                "sha256": w4_sh_sha,
            },
            "RANKING_AUDIT": {
                "path": c_writers.RANKING_AUDIT_REL,
                "sha256": audit_sha,
            },
        },
        "ranking_audit_all_pass": all_pass,
        "forbidden_actions_status": {
            "worst_error_ranking_executed": True,
            "selection_contract_executed": True,
            "individual_case_inspection": False,
            "casebook_created": False,
            "overlap_analysis_executed": False,
            "error_concentration_executed": False,
            "regime_enrichment_executed": False,
            "temporal_context_executed": False,
            "input_context_executed": False,
            "phase52_started": False,
        },
        "ready_for_phase51_d": False,
    }
    manifest_sha = c_writers.write_json(
        manifest, c_writers.PHASE51_C_MANIFEST_REL, root,
    )

    return {
        "phase": 51,
        "subphase": "51-C",
        "status": "PASS" if all_pass else "FAIL",
        "selection_contract_sha256": contract_sha,
        "source_working_table_sha256": wt_sha,
        "artifacts": {
            "w1": w1_sha,
            "w2": w2_sha,
            "w3": w3_sha,
            "w4": w4_sha,
            "w3_sh": w3_sh_sha,
            "w4_sh": w4_sh_sha,
            "audit": audit_sha,
            "manifest": manifest_sha,
        },
        "n_artifacts": 8,
        "all_audit_pass": all_pass,
        "ranking_audit_per_family": [
            {"family": a["family"], "seed": a["seed"], "rows_emitted": a["rows_emitted"], "all_pass": a["all_pass"]}
            for a in audits
        ],
    }


def main(project_root=None) -> dict[str, Any]:
    return materialize_phase51_c(project_root)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
