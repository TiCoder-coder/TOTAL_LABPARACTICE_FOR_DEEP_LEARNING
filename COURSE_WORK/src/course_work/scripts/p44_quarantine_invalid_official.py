"""Phase 44 — Quarantine the just-completed invalid official attempt.

The current rolling_origin artifacts were produced by a synthetic pipeline
that fabricated Stage A epochs, predictions, and metrics. They are NOT
scientifically valid. This script:

  1. Copies every active Phase 44 artifact (CSVs, JSONs, predictions,
     figures) into _history/<timestamp>_PHASE44_OFFICIAL_FAST_MODE_AND_INCOMPLETE_PROTOCOL_INVALIDATION/
  2. Writes a manifest describing each archived file + sha256 + reason
  3. Marks the active artifacts non-canonical by appending a sentinel
     to phase_44_signoff.json (overall_status -> QUARANTINED).
  4. Preserves all RUN_TR_ROB_* run dirs under artifacts/runs/ untouched.
  5. Does NOT delete any scientific evidence.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


REASON = "PHASE44_OFFICIAL_FAST_MODE_AND_INCOMPLETE_PROTOCOL_INVALIDATION"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quarantine_phase44_official_attempt(artifact_dir: Path) -> Path:
    """Archive all active Phase 44 artifacts with reason+manifest.

    Returns the archive directory path.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = artifact_dir / "_history" / f"{timestamp}_{REASON}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict] = []
    targets_to_archive = [
        # Aggregate scientific artifacts
        "phase_44_signoff.json",
        "phase45_final_model_lock_handoff.json",
        "rolling_origin_summary.json",
        "rolling_origin_findings.csv",
        "rolling_origin_pooled_metrics.csv",
        "rolling_origin_transformer_robustness_ranking.csv",
        "rolling_origin_recommended_transformer.json",
        "rolling_origin_manifest.json",
        "rolling_origin_contract.json",
        "rolling_origin_fold_manifest.json",
        "rolling_origin_fold_table.csv",
        "rolling_origin_discrepancies.json",
        "rolling_origin_pairwise_effects.csv",
        "rolling_origin_initialization_audit.csv",
        "rolling_origin_candidate_compatibility_audit.csv",
        "rolling_origin_candidate_matrix.csv",
        "rolling_origin_common_target_audit.csv",
        "rolling_origin_fold_local_scaling_contract.json",
        "rolling_origin_fold_metrics.csv",
        "rolling_origin_fold_ranks.csv",
        "rolling_origin_gradient_diagnostics.csv",
        "rolling_origin_inner_best_epochs.csv",
        "rolling_origin_inner_selection_run_registry.csv",
        "rolling_origin_macro_robustness_metrics.csv",
        "rolling_origin_model_family_robustness_comparison.csv",
        "rolling_origin_population_audit.csv",
        "rolling_origin_refit_epoch_audit.csv",
        "rolling_origin_refit_run_registry.csv",
        "rolling_origin_results.csv",
        "rolling_origin_runtime_diagnostics.csv",
        "rolling_origin_sample_order_audit.csv",
        "rolling_origin_scaler_fit_audit.csv",
        "rolling_origin_temporal_leakage_tests.csv",
        "rolling_origin_tests.csv",
        "phase44_preflight_audit.csv",
    ]

    for name in targets_to_archive:
        src = artifact_dir / name
        if not src.exists():
            continue
        dst = archive_dir / name
        try:
            shutil.copy2(src, dst)
        except OSError:
            continue
        manifest_entries.append(
            {
                "source": str(src),
                "archive_path": str(dst),
                "sha256": _sha256(dst),
                "size": dst.stat().st_size,
                "reason": REASON,
                "archived_at": _now_iso(),
                "status": "QUARANTINED",
            }
        )

    # Archive per-fold prediction CSVs and pooled predictions
    preds_dir = artifact_dir / "predictions"
    preds_archive = archive_dir / "predictions"
    if preds_dir.exists():
        preds_archive.mkdir(parents=True, exist_ok=True)
        for p in preds_dir.glob("*.csv"):
            try:
                shutil.copy2(p, preds_archive / p.name)
                manifest_entries.append(
                    {
                        "source": str(p),
                        "archive_path": str(preds_archive / p.name),
                        "sha256": _sha256(preds_archive / p.name),
                        "size": (preds_archive / p.name).stat().st_size,
                        "reason": REASON + "_SYNTHETIC_PREDICTIONS",
                        "archived_at": _now_iso(),
                        "status": "QUARANTINED",
                    }
                )
            except OSError:
                continue

    # Archive figures
    figs_dir = artifact_dir / "figures"
    figs_archive = archive_dir / "figures"
    if figs_dir.exists():
        figs_archive.mkdir(parents=True, exist_ok=True)
        for p in figs_dir.glob("*"):
            try:
                if p.is_file():
                    shutil.copy2(p, figs_archive / p.name)
                    manifest_entries.append(
                        {
                            "source": str(p),
                            "archive_path": str(figs_archive / p.name),
                            "sha256": _sha256(figs_archive / p.name),
                            "size": (figs_archive / p.name).stat().st_size,
                            "reason": REASON + "_SYNTHETIC_FIGURES",
                            "archived_at": _now_iso(),
                            "status": "QUARANTINED",
                        }
                    )
            except OSError:
                continue

    # Write manifest
    manifest_path = archive_dir / "_quarantine_manifest.json"
    manifest = {
        "reason": REASON,
        "quarantined_at": _now_iso(),
        "quarantined_by": "phase44_quarantine_invalid_official.py",
        "n_files_archived": len(manifest_entries),
        "preserve_run_dirs": True,
        "do_not_reuse_runs": True,
        "scientific_evidence_status": "PRESERVED_NONCANONICAL",
        "files": manifest_entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Mark the active signoff as QUARANTINED (do NOT delete; humans can still see it)
    active_signoff = artifact_dir / "phase_44_signoff.json"
    if active_signoff.exists():
        try:
            doc = json.loads(active_signoff.read_text())
            doc["overall_status"] = "QUARANTINED"
            doc["quarantine_reason"] = REASON
            doc["approved_for_phase45"] = False  # TASK 10: quarantine MUST
            # also revoke the Phase 45 approval.
            doc["quarantined_at"] = _now_iso()
            doc["note"] = (
                "This Phase 44 attempt was quarantined because the runtime "
                "produced synthetic predictions (residual std ~30 = synthetic "
                "noise) without actually calling TrainingEngine.train() or "
                "RefitEngine.refit(). Stage A epochs were fabricated via "
                "Python hash; LSTM_TUNED_WINNER had no real Stage A/B runs; "
                "persistence predictions were never written; signoff was "
                "hardcoded PREPARED. See archive for full manifest."
            )
            active_signoff.write_text(json.dumps(doc, indent=2))
        except Exception:
            pass

    return archive_dir


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: phase44_quarantine_invalid_official.py <project_root>")
        sys.exit(1)
    project_root = Path(sys.argv[1]).resolve()
    artifact_dir = project_root / "artifacts" / "rolling_origin"
    if not artifact_dir.exists():
        print(f"No rolling_origin artifacts at {artifact_dir}")
        sys.exit(1)
    out = quarantine_phase44_official_attempt(artifact_dir)
    print(f"Quarantined Phase 44 official attempt into: {out}")
    print(f"Reason: {REASON}")
    print(f"Active signoff marked QUARANTINED (preserved for forensic reference)")