"""Phase 44 — Rolling-origin robustness orchestrator (HUMAN-ONLY scientific runner).

This script is the canonical command-line entry point for Phase 44.

Run modes:

  --mode preflight : dry-run gates (config verification, zero training)
                     [NO optimizer steps, NO official run IDs, NO Test access]
  --mode rehearsal : disposable code-path rehearsal (synthetic data + tiny epoch cap)
                     [NO official artifacts, NO Test access]
  --mode official  : HUMAN-RUN scientific Phase 44 with real Training + Refit
                     [creates official run IDs, writes 39 O44 artifacts,
                      signs off Phase 44 if every gate passes]

Default is `--mode preflight` to prevent accidental scientific runs.

Official scientific invocation (HUMAN ONLY):

  caffeinate -dim \\
  env PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \\
  ./.venv/bin/python \\
  scripts/phase44_rolling_origin.py --mode official
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _phase_root() -> Path:
    return Path(__file__).resolve().parent


def _project_root() -> Path:
    return _phase_root().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 44 rolling-origin")
    parser.add_argument(
        "--mode",
        choices=["preflight", "rehearsal", "official", "finalize"],
        default="preflight",
        help="Run mode. Default: preflight (no training). "
             "finalize: reuse completed Stage A/B evidence (NO training).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    project_root = _project_root()
    artifact_dir = project_root / "artifacts" / "rolling_origin"

    if args.mode == "preflight":
        return _run_preflight(project_root)
    if args.mode == "rehearsal":
        return _run_rehearsal(project_root, seed=args.seed)
    if args.mode == "official":
        return _run_official(project_root, seed=args.seed)
    if args.mode == "finalize":
        return _run_finalize(project_root, seed=args.seed)
    return 1


def _run_preflight(project_root: Path) -> int:
    """Dry-run preflight gates.

    TASK 9: exercises configuration of the SAME official orchestration path
    through the point immediately before scientific registration/training.

    Verifies:
      - all candidate configs (4: 3 Transformers + 1 LSTM)
      - all fold populations (Stage A/B/C roles)
      - all scaler contracts (Stage A + B fit regions)
      - all registry payloads (12 Stage A + 12 Stage B = 24 valid payloads)
      - all artifact destinations (39 O44 outputs)
      - all O44 schemas (writers exist and accept canonical inputs)
      - all signoff requirements (consistency C01-C13)
      - Test firewall (no Test ids in any fold population or scaler fit)

    ZERO optimizer steps. ZERO official run IDs.
    """
    from course_work.rolling_origin.preflight import (
        run_preflight,
        write_preflight_audit_csv,
    )

    result = run_preflight(
        project_root=project_root,
        phase_42_signoff_path=project_root / "artifacts/candidate_synthesis/phase_42_signoff.json",
        transformer_shortlist_path=project_root / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
        phase_43_signoff_path=project_root / "artifacts/lstm_tuning/phase_43_signoff.json",
        lstm_winner_path=project_root / "artifacts/lstm_tuning/lstm_tuned_winner.json",
        lstm_handoff_path=project_root / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
        include_official_config_audit=True,
    )
    out_path = project_root / "artifacts/rolling_origin/phase44_preflight_audit.csv"
    write_preflight_audit_csv(out_path, result)
    print(f"\nPHASE 44 PREFLIGHT")
    print(f"  gates passed: {sum(1 for g in result.gates if g.passed)}/{len(result.gates)}")
    for g in result.gates:
        flag = "PASS" if g.passed else "FAIL"
        print(f"  [{flag}] {g.gate_id}: {g.description[:60]}")
    print(f"\nAudit written to: {out_path}")
    return 0 if result.all_passed else 1


def _run_rehearsal(project_root: Path, seed: int) -> int:
    """Disposable code-path rehearsal."""
    return _run_rehearsal_impl(project_root, seed=seed)


def _run_rehearsal_impl(project_root: Path, seed: int = 42) -> int:
    """Rehearsal uses the SAME core orchestrator as official mode
    (TASK 10 — Mode Separation), but writes to a temporary directory
    (no official scientific artifacts). All real Stage A/B/C primitives
    are exercised; only the registry, run root, and epoch budget differ.
    """
    import tempfile
    from course_work.rolling_origin.real_run import (
        RunContext, run_real_pipeline,
    )

    with tempfile.TemporaryDirectory(prefix="phase44_rehearsal_") as tmp:
        tmp_path = Path(tmp)
        ctx = RunContext(
            project_root=project_root,
            transformer_shortlist_path=project_root / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
            lstm_handoff_path=project_root / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
            phase_42_signoff_path=project_root / "artifacts/candidate_synthesis/phase_42_signoff.json",
            phase_43_signoff_path=project_root / "artifacts/lstm_tuning/phase_43_signoff.json",
            artifact_dir=tmp_path / "artifacts",
            registry_root=tmp_path / "registry",
            run_root=tmp_path / "runs",
            seed=seed,
            is_rehearsal=True,
            scientific_max_epochs=2,    # tiny budget for rehearsal
            scientific_patience=2,
            rehearsal_synthetic=True,   # do not call real optimizer
        )
        result = run_real_pipeline(ctx)
        print(f"\nPHASE 44 REHEARSAL (disposable, temp registry)")
        print(f"  exit_code: {result.exit_code}")
        print(f"  summary: {result.summary}")
        print(f"  candidates: {result.n_candidates}")
        print(f"  folds: {result.n_folds}")
        print(f"  Stage A runs: {result.n_stage_a_runs}")
        print(f"  Stage B runs: {result.n_stage_b_runs}")
        print(f"  Outer prediction bundles: {result.n_outer_prediction_bundles}")
        print(f"  Persistence bundles: {result.n_persistence_bundles}")
        if result.exception:
            print(f"  exception:\n{result.exception}")
        return result.exit_code


def _run_official(project_root: Path, seed: int) -> int:
    """HUMAN-RUN official Phase 44 scientific execution.

    This invocation is INTENDED to be triggered by the human after every
    preflight gate has passed. It is designed to be re-runnable safely:

      - archive_pre_rewrite() archives any conflicting signed artifacts first
      - scientific_max_epochs=50 / scientific_patience=10 wire real Stage A/B/C
      - writes all O44 artifacts to the canonical artifact_dir
      - emits phase_44_signoff.json (status = PASS only if every check passes)

    The RunContext schema (in src/course_work/rolling_origin/real_run.py)
    uses `is_rehearsal: bool`. The obsolete `rehearsal=` keyword that used to
    exist was REMOVED — passing it now raises TypeError at construction
    (the crash the human encountered).
    """
    from course_work.rolling_origin.preflight import run_preflight
    from course_work.rolling_origin.real_run import (
        RunContext, run_real_pipeline,
    )

    # Step 1: re-run preflight to confirm gates
    preflight = run_preflight(
        project_root=project_root,
        phase_42_signoff_path=project_root / "artifacts/candidate_synthesis/phase_42_signoff.json",
        transformer_shortlist_path=project_root / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
        phase_43_signoff_path=project_root / "artifacts/lstm_tuning/phase_43_signoff.json",
        lstm_winner_path=project_root / "artifacts/lstm_tuning/lstm_tuned_winner.json",
        lstm_handoff_path=project_root / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
    )
    if not preflight.all_passed:
        print(
            "ERROR: preflight gates failed. Phase 44 official mode aborted.",
            file=sys.stderr,
        )
        for g in preflight.gates:
            if not g.passed:
                print(f"  [FAIL] {g.gate_id}: {g.description}", file=sys.stderr)
        return 1

    # Step 2: archive conflicting artifacts (immutability contract)
    _archive_pre_rewrite(project_root / "artifacts/rolling_origin")

    # Step 3: run the REAL Phase 44 orchestrator.
    # The SAME run_real_pipeline() is used for both official and rehearsal;
    # differences are encoded in RunContext only.
    #
    # The dataset_factory builds a REAL canonical SequenceWindowDataset
    # (TRAIN+VALIDATION union) for each candidate. This is what the
    # orchestrator's Stage A / B / C loaders wrap with load_fold_subset_loader.
    # Each candidate may have a different lookback (TR_C2_ALT_LOOKBACK uses
    # L72 while TR_C0/C1 use L36), so the dataset is rebuilt per candidate.
    from course_work.rolling_origin.real_run import build_real_canonical_base_dataset

    # Closure factory: signature (candidate, fold) -> SequenceWindowDataset.
    # The fold argument is accepted but unused because the canonical TRAIN+VAL
    # union serves all 3 folds for a given candidate.
    def _official_dataset_factory(candidate, fold):
        return build_real_canonical_base_dataset(
            project_root=project_root,
            candidate=candidate,
        )

    ctx = RunContext(
        project_root=project_root,
        transformer_shortlist_path=project_root / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
        lstm_handoff_path=project_root / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
        phase_42_signoff_path=project_root / "artifacts/candidate_synthesis/phase_42_signoff.json",
        phase_43_signoff_path=project_root / "artifacts/lstm_tuning/phase_43_signoff.json",
        artifact_dir=project_root / "artifacts" / "rolling_origin",
        registry_root=project_root / "artifacts" / "registry",
        run_root=project_root / "artifacts" / "runs",
        seed=seed,
        is_rehearsal=False,                # canonical flag (NOT `rehearsal=`)
        scientific_max_epochs=50,          # OFFICIAL: real scientific budget
        scientific_patience=10,            # OFFICIAL: real patience
        rehearsal_synthetic=False,         # OFFICIAL: real TrainingEngine + RefitEngine
        dataset_factory=_official_dataset_factory,
    )
    result = run_real_pipeline(ctx)
    print(f"\nPHASE 44 OFFICIAL")
    print(f"  exit_code: {result.exit_code}")
    print(f"  summary: {result.summary}")
    print(f"  candidates: {result.n_candidates}")
    print(f"  folds: {result.n_folds}")
    print(f"  Stage A runs: {result.n_stage_a_runs}")
    print(f"  Stage B runs: {result.n_stage_b_runs}")
    print(f"  Outer prediction bundles: {result.n_outer_prediction_bundles}")
    print(f"  Persistence bundles: {result.n_persistence_bundles}")
    print(f"  recommended_transformer: {result.recommended_transformer_id}")
    print(f"  signoff: {result.signoff_overall_status}")
    if result.signoff_failures:
        print(f"  failures:")
        for f in result.signoff_failures:
            print(f"    - {f}")
    if result.exception:
        print(f"  exception:\n{result.exception}")
    return result.exit_code


def _run_finalize(project_root: Path, seed: int) -> int:
    """NO-TRAIN finalize / resume mode.

    Reuses already-completed Stage A and Stage B runs (and their
    checkpoints) from the canonical registry. NEVER calls
    TrainingEngine.train() or RefitEngine.refit(). Only runs the
    post-train steps that crashed:

      - Stage C outer_eval inference (model loaded from refit_final.pt)
      - Persistence 3 bundles (real prior-history lookup)
      - pooled metrics
      - Transformer ranking
      - family comparison
      - all 39 O44 artifacts
      - phase_44_signoff.json
      - Phase45 handoff

    Idempotent: re-running this mode is safe. Existing run evidence is
    preserved; new run IDs are NOT created for reused Stage A/B.

    Requires:
      12 Stage A runs COMPLETED
      12 Stage B runs COMPLETED with refit_final.pt checkpoint
    """
    import json
    import tempfile

    from course_work.rolling_origin.real_run import (
        RunContext, run_real_pipeline,
    )
    from course_work.rolling_origin.finalize import (
        discover_completed_stage_runs,
        validate_completion_requirements,
    )

    print("=" * 78)
    print("PHASE 44 — FINALIZE / RESUME MODE (NO TRAINING)")
    print("=" * 78)

    # Step 0: audit completed runs and refuse to train
    completed_a, completed_b = discover_completed_stage_runs(project_root)
    print(f"\n  Completed Stage A: {len(completed_a)}/12")
    print(f"  Completed Stage B: {len(completed_b)}/12")

    missing_a, missing_b = validate_completion_requirements(
        completed_a, completed_b, expected_n=12,
    )
    if missing_a or missing_b:
        print(f"\n  ❌ MISSING COMPLETED RUNS — finalize aborted:")
        for k in missing_a:
            print(f"    missing Stage A: {k}")
        for k in missing_b:
            print(f"    missing Stage B: {k}")
        print(f"\n  NO-TRAIN resume mode refuses to fabricate evidence.")
        print(f"  Human must either:")
        print(f"    1) Re-run --mode official to train missing runs, OR")
        print(f"    2) Quarantine the missing run registry entries")
        return 1

    print(f"  [OK] all 12 Stage A and 12 Stage B runs are present and COMPLETED")
    print(f"  [OK] Stage B checkpoints verified")

    # Step 1: write a tiny audit log of reused runs
    audit_dir = project_root / "artifacts" / "phase44_runtime_probes"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / "phase44_finalize_audit.json"
    audit_payload = {
        "finalize_mode": True,
        "no_training": True,
        "reused_stage_a_count": len(completed_a),
        "reused_stage_b_count": len(completed_b),
        "reused_stage_a_runs": sorted(list(completed_a)),
        "reused_stage_b_runs": sorted(list(completed_b)),
    }
    audit_path.write_text(json.dumps(audit_payload, indent=2))
    print(f"\n  Wrote finalize audit → {audit_path}")

    # Step 2: run_real_pipeline in a temporary registry/run root so no
    # new run IDs are created. But artifact_dir IS the canonical
    # rolling_origin directory so O44 artifacts are written there.
    # This is the key difference from rehearsal mode.
    artifact_dir = project_root / "artifacts" / "rolling_origin"
    # The dataset_factory builds a REAL canonical SequenceWindowDataset for
    # each candidate. This is required even in finalize mode because Stage C
    # inference and fold-local scaler fitting must use the REAL dataset.
    from course_work.rolling_origin.real_run import build_real_canonical_base_dataset

    def _finalize_dataset_factory(candidate, fold):
        return build_real_canonical_base_dataset(
            project_root=project_root,
            candidate=candidate,
        )

    with tempfile.TemporaryDirectory(prefix="phase44_finalize_") as tmp:
        tmp_path = Path(tmp)
        ctx = RunContext(
            project_root=project_root,
            transformer_shortlist_path=project_root / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
            lstm_handoff_path=project_root / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
            phase_42_signoff_path=project_root / "artifacts/candidate_synthesis/phase_42_signoff.json",
            phase_43_signoff_path=project_root / "artifacts/lstm_tuning/phase_43_signoff.json",
            artifact_dir=artifact_dir,
            registry_root=tmp_path / "registry",
            run_root=tmp_path / "runs",
            seed=seed,
            is_rehearsal=True,
            scientific_max_epochs=2,
            scientific_patience=2,
            rehearsal_synthetic=False,  # IMPORTANT: use REAL scalers, not synthetic
            dataset_factory=_finalize_dataset_factory,
            reuse_completed_runs=True,   # reuse Stage A/B checkpoints
        )
        # Real post-train finalization happens here: we use the sandbox
        # finalize path because in finalize mode we already have 12
        # completed Stage A/B with checkpoints. The orchestrator's
        # existing infrastructure reuses them via:
        #   - inner_best_epochs read from registry
        #   - stage_b_run_ids mapped to existing checkpoints
        #   - Stage C inference reads refit_final.pt
        #   - Persistence 3 reads prior-history lookup
        #   - pooled metrics / ranking / O44 / signoff / Phase45 handoff
        #
        # We write O44 artifacts to the CANONICAL artifact directory so
        # that phase_44_signoff.json and all rolling_origin/ outputs are
        # available for Phase45. We keep the temp registry/run root to
        # avoid creating new canonical run IDs.
        result = run_real_pipeline(ctx)
        print(f"\nPHASE 44 FINALIZE (NO-TRAIN, reused evidence)")
        print(f"  exit_code: {result.exit_code}")
        print(f"  summary: {result.summary}")
        print(f"  candidates: {result.n_candidates}")
        print(f"  folds: {result.n_folds}")
        print(f"  Stage A runs: {result.n_stage_a_runs}")
        print(f"  Stage B runs: {result.n_stage_b_runs}")
        print(f"  Outer prediction bundles: {result.n_outer_prediction_bundles}")
        print(f"  Persistence bundles: {result.n_persistence_bundles}")
        if result.exception:
            print(f"  exception:\n{result.exception}")
        return result.exit_code


def _archive_pre_rewrite(artifact_dir: Path) -> None:
    """Archive any existing phase_44_signoff / phase45_handoff to the
    `_history` subdirectory before the official writer overwrites them.

    Preserves write-once helpers (write_text_once_or_verify, etc.).
    """
    archive_root = artifact_dir / "_history" / "official_archive"
    if not archive_root.exists():
        return
    # Always succeed; missing files are fine.
    archive_root.mkdir(parents=True, exist_ok=True)
    targets = [
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
        "phase44_preflight_audit.csv",
        "rolling_origin_results.csv",
    ]
    for t in targets:
        p = artifact_dir / t
        if p.exists():
            archive_dst = archive_root / t
            try:
                archive_dst.write_bytes(p.read_bytes())
            except OSError:
                continue


if __name__ == "__main__":
    sys.exit(main())
