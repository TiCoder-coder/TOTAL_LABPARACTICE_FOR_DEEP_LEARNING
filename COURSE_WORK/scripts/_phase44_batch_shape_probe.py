"""TASK 2 — Capture the exact real batch shape.

Reconstructs the EXACT first Stage A setup the human's official attempt
hit (TR_C0_PRIMARY, RO1, L36), builds the inner_train_loader exactly as
real_run.py does, reads ONE batch, and prints the observed shape.

NO TrainingEngine.train call.
NO optimizer step.
NO canonical mutation.
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> int:
    print("=" * 78)
    print("PHASE 44 — CAPTURE EXACT REAL BATCH SHAPE (TASK 2)")
    print("=" * 78)

    canonical_artifact_dir = PROJECT_ROOT / "artifacts" / "rolling_origin"
    canonical_before = (
        sorted(p.name for p in canonical_artifact_dir.iterdir())
        if canonical_artifact_dir.exists() else []
    )

    with tempfile.TemporaryDirectory(prefix="phase44_batch_") as tmp:
        tmp_path = Path(tmp)

        # Step 1: load candidates
        from course_work.rolling_origin.candidate_loader import load_candidates
        candidates = load_candidates(
            project_root=PROJECT_ROOT,
            transformer_shortlist_path=PROJECT_ROOT / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
            lstm_handoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
        )
        tr_c0 = next(c for c in candidates if c.candidate_id == "TR_C0_PRIMARY")
        print(f"\n[FIRST STAGE A SETUP]")
        print(f"  candidate:        {tr_c0.candidate_id}")
        print(f"  model_family:     {tr_c0.model_family}")
        print(f"  lookback:         {tr_c0.lookback_steps}")
        print(f"  feature_count:    {tr_c0.config.get('data', {}).get('feature_count', '?')}")
        print(f"  feature_variant:  {tr_c0.feature_variant_id}")
        print(f"  target_scaling:   {tr_c0.target_scaling_option}")

        # Step 2: build folds
        from course_work.rolling_origin.populations import (
            extract_robase_train_ids, extract_robase_val_ids,
        )
        from course_work.rolling_origin.folds import build_rolling_folds
        rtrn_ids = extract_robase_train_ids(PROJECT_ROOT)
        rval_ids = extract_robase_val_ids(PROJECT_ROOT)
        folds = build_rolling_folds(rtrn_ids, rval_ids, k=3)
        first_fold = folds[0]
        print(f"\n[FOLD]")
        print(f"  fold_id:          {first_fold.fold_id}")
        print(f"  inner_train_ids:  {len(first_fold.inner_train_ids)}")
        print(f"  inner_val_ids:    {len(first_fold.inner_val_ids)}")
        print(f"  outer_train_ids:  {len(first_fold.outer_train_ids)}")
        print(f"  outer_eval_ids:   {len(first_fold.outer_eval_ids)}")

        # Step 3: replicate real_run.py's _ROREFOLD_dataset path
        # exactly as the human's official attempt would have.
        # (This is the BROKEN path that produces [B,F] instead of [B,L,F].)
        from course_work.rolling_origin.real_run import (
            _build_robase_dataset,  # the BROKEN adapter
        )
        from course_work.rolling_origin.appliance_lookup import (
            build_windowpop_with_appliances,
        )

        windowpop_df = build_windowpop_with_appliances(PROJECT_ROOT)
        all_target_ids = set(str(t) for t in rtrn_ids) | set(str(t) for t in rval_ids)
        wb = windowpop_df.copy()
        wb["target_id"] = wb["target_id"].astype(str)
        wb_subset = wb[wb["target_id"].isin(all_target_ids)].copy()
        drop_cols = {
            "target_id", "target_timestamp", "target_split_id",
            "Appliances", "raw_row_index", "continuity_segment_id",
            "valid_L36", "valid_L72", "valid_L144",
            "included_common_population", "target_sample_id",
        }
        feat_cols = [c for c in wb_subset.columns if c not in drop_cols]
        print(f"\n[FEATURE COLUMNS DISCOVERED]")
        print(f"  feat_cols: {feat_cols}")
        print(f"  feature_count from column count: {len(feat_cols)}")

        fold_dataset_broken = _build_robase_dataset(wb_subset, feat_cols)
        print(f"\n[BROKEN DATASET]")
        print(f"  _X.shape: {fold_dataset_broken._X.shape}")
        print(f"  type: {type(fold_dataset_broken).__name__}")
        # Sample one item
        sample = fold_dataset_broken[0]
        print(f"  sample['x']: shape={sample['x'].shape if hasattr(sample['x'], 'shape') else type(sample['x'])}")
        print(f"  sample['x']: type={type(sample['x']).__name__}")
        if hasattr(sample['x'], 'shape'):
            print(f"  sample['x'].ndim: {sample['x'].ndim}")
            print(f"  sample['x'].dtype: {sample['x'].dtype}")

        # Step 4: build the loader exactly as real_run.py does
        from course_work.rolling_origin.stages import load_fold_subset_loader
        loader, indices = load_fold_subset_loader(
            base_dataset=fold_dataset_broken,
            target_ids=list(first_fold.inner_train_ids)[:64],  # small subset
            batch_size=32,
            shuffle=False,
            seed=42,
        )
        print(f"\n[LOADER]")
        print(f"  indices returned: {len(indices)}")
        print(f"  dataset size after subset: {len(loader.dataset)}")

        # Read ONE batch
        batch = next(iter(loader))
        print(f"\n[OBSERVED REAL BATCH]")
        print(f"  type(batch):           {type(batch).__name__}")
        if isinstance(batch, dict):
            print(f"  batch keys:            {sorted(batch.keys())}")
        else:
            print(f"  batch:                 {batch}")

        if isinstance(batch, dict):
            for k, v in batch.items():
                if hasattr(v, 'shape'):
                    print(f"  batch['{k}']: shape={tuple(v.shape)}, dtype={v.dtype}, ndim={v.ndim}")
                elif isinstance(v, (list, tuple)):
                    print(f"  batch['{k}']: list len={len(v)}, first type={type(v[0]).__name__ if len(v) > 0 else 'NA'}")
                else:
                    print(f"  batch['{k}']: type={type(v).__name__}, value={v!r:.60}")

        # Step 5: try the same loader with the REAL canonical SequenceWindowDataset
        print(f"\n[CANONICAL DATASET COMPARISON]")
        print(f"  Building real SequenceWindowDataset for TR_C0_PRIMARY L36...")
        from course_work.data.datasets import build_dataset_suite

        suite, window_fp = build_dataset_suite(
            project_root=PROJECT_ROOT,
            variant_id=tr_c0.feature_variant_id,
            lookback=tr_c0.lookback_steps,
            target_option=tr_c0.target_scaling_option,
            boundary_protocol=tr_c0.boundary_protocol,
        )
        train_ds = suite["TRAIN"]
        val_ds = suite["VALIDATION"]
        sample_train = train_ds[0]
        sample_val = val_ds[0]
        print(f"  train_ds keys:         {sorted(sample_train.keys())}")
        print(f"  sample['x'].shape:     {tuple(sample_train['x'].shape)}")
        print(f"  sample['x'].ndim:      {sample_train['x'].ndim}")
        print(f"  sample['x'].dtype:     {sample_train['x'].dtype}")
        print(f"  train_ds len:          {len(train_ds)}")
        print(f"  val_ds len:            {len(val_ds)}")
        print(f"  window_fingerprint:    {window_fp[:24]}...")

        # Now build a fold-specific loader from the canonical dataset
        from torch.utils.data import DataLoader, Subset

        # Use the actual inner_train_ids (which are timeline target indices,
        # but canonical SequenceWindowDataset is indexed by position).
        # The canonical train_ds's position N corresponds to row N of TRAIN.
        # For our probe, we don't need exact mapping; we just want to see
        # the SHAPE from a fold-specific loader.
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=0)
        train_batch = next(iter(train_loader))
        print(f"\n[CANONICAL TRAIN BATCH]")
        if isinstance(train_batch, dict):
            print(f"  batch keys:            {sorted(train_batch.keys())}")
            for k, v in train_batch.items():
                if hasattr(v, 'shape'):
                    print(f"  batch['{k}'].shape:    {tuple(v.shape)}")
                    print(f"  batch['{k}'].ndim:     {v.ndim}")
                    print(f"  batch['{k}'].dtype:    {v.dtype}")

    canonical_after = (
        sorted(p.name for p in canonical_artifact_dir.iterdir())
        if canonical_artifact_dir.exists() else []
    )
    assert canonical_after == canonical_before, (
        "canonical artifact_dir was mutated by batch probe"
    )
    print(f"\n[OK] canonical artifact_dir unchanged ({len(canonical_after)} files)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(99)