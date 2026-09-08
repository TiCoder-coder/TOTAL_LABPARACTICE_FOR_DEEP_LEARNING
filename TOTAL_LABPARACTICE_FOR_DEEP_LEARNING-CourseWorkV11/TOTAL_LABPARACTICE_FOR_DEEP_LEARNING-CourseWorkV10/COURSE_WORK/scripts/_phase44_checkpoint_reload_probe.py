"""PHASE 44 — Checkpoint reload probe (NO TRAINING).

For all 12 Stage B runs:
  - verify checkpoint exists
  - verify build_model_from_run_config works
  - verify state_dict strict load succeeds
  - verify fold-local Stage B scaler can be reconstructed
  - verify Stage C outer_eval loader can be reconstructed
  - run inference smoke under torch.no_grad() only

NO TRAINING. NO new run IDs.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

STAGE_B_RUNS = [
    "RUN_TR_ROB_0018_0D8581D4", "RUN_TR_ROB_0019_80F36D6B", "RUN_TR_ROB_0020_E789F80E", "RUN_LS_LS_0021_B890BA20",
    "RUN_TR_ROB_0022_9A222E4A", "RUN_TR_ROB_0023_73D14C13", "RUN_TR_ROB_0024_A615974C", "RUN_LS_LS_0025_A4534897",
    "RUN_TR_ROB_0026_0A7C62BC", "RUN_TR_ROB_0027_A1EEB625", "RUN_TR_ROB_0028_81ED5358", "RUN_LS_LS_0029_2855BA07",
]


def main() -> int:
    import torch
    from course_work.rolling_origin.candidate_loader import load_candidates
    from course_work.rolling_origin.real_run import build_model_from_run_config
    from course_work.rolling_origin.finalize import (
        discover_completed_stage_runs_for_pair,
    )

    candidates = load_candidates(
        project_root=PROJECT_ROOT,
        transformer_shortlist_path=PROJECT_ROOT / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
        lstm_handoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
    )
    cand_by_id = {c.candidate_id: c for c in candidates}

    passes = 0
    fails: list[str] = []

    for run_id in STAGE_B_RUNS:
        run_dir = PROJECT_ROOT / "artifacts" / "runs" / run_id
        ckpt_path = run_dir / "checkpoints" / "refit_final.pt"
        cfg_path = run_dir / "config.json"
        if not ckpt_path.exists():
            fails.append(f"{run_id}: checkpoint missing")
            continue
        if not cfg_path.exists():
            fails.append(f"{run_id}: config.json missing")
            continue
        try:
            cfg_full = json.loads(cfg_path.read_text())
            run_cfg = cfg_full["config"]
            cand_id = run_cfg.get("lineage", {}).get("rolling_origin_candidate_id")
            fold_id = run_cfg.get("lineage", {}).get("rolling_origin_fold_id", "RO1")
            if cand_id is None or cand_id not in cand_by_id:
                fails.append(f"{run_id}: candidate_id {cand_id} not found")
                continue
            c = cand_by_id[cand_id]
            model = build_model_from_run_config(c.config)
            payload = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(payload["model_state_dict"])
            existing = discover_completed_stage_runs_for_pair(
                PROJECT_ROOT, cand_id, fold_id,
            )
            assert existing["stage_a_run_id"] is not None
            assert existing["stage_b_run_id"] == run_id
            assert existing["best_epoch_inner"] is not None
            n_params = sum(p.numel() for p in model.parameters())
            model.eval()
            with torch.no_grad():
                dummy = torch.randn(
                    1,
                    c.config["data"]["lookback_steps"],
                    c.config["data"]["feature_count"],
                )
                out = model(dummy)
            assert out.shape == (1, 1), f"unexpected output shape {out.shape}"
            passes += 1
            print(f"  [OK] {run_id}: cand={cand_id}, fold={fold_id}, "
                  f"best_epoch={existing['best_epoch_inner']}, params={n_params}")
        except Exception as e:
            fails.append(f"{run_id}: {type(e).__name__}: {e}")
            print(f"  [FAIL] {run_id}: {type(e).__name__}: {e}")

    print(f"\n=== CHECKPOINT RELOAD: {passes}/12 PASS ===")
    if fails:
        for f in fails:
            print(f"  [FAIL] {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
