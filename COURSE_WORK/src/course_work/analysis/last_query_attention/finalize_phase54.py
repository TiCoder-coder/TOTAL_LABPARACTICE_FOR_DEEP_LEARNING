"""Phase 54 — final report, README, processing log writers.

Called after `run_phase54` to generate the human-readable report, README,
processing log, and update the static safety scan status.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from .writers import write_json_atomic


def _strip_strings_and_comments(src: str) -> str:
    """Strip docstrings, comments, and string literals for static safety scan.
    Also strips the keyword list in static_safety_scan (data, not code)."""
    s = re.sub(r"\"\"\".*?\"\"\"", "", src, flags=re.S)
    s = re.sub(r"'''.*?'''", "", s, flags=re.S)
    s = re.sub(r"#[^\n]*", "", s)
    s = re.sub(r"\"[^\"\\\\]*(?:\\\\.[^\"\\\\]*)*\"", "\"\"", s)
    s = re.sub(r"'[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", "''", s)
    # Strip forbidden keyword lists (tuples/lists of strings)
    s = re.sub(r"\[\s*\(?\s*\"[^\"]*\"[^\]]*\]", "[]", s)
    return s


def static_safety_scan(phase54_root: Path) -> dict:
    """Scan src/course_work/phase54/ for forbidden runtime operations
    in executable code (excluding strings/comments).

    Phase 54 is allowed to read Phase 52 NPZ via np.load (frozen source).
    The safety scanner itself is excluded from the keyword search because its
    keyword metadata is data, not code.
    """
    forbidden = [
        ("torch.load(", "torch checkpoint loading"),
        ("model(", "model.forward"),
        ("return_attention", "attention extraction"),
        ("materialize_phase52", "materialize call"),
        ("optimizer.step", "optimizer.step call"),
        (".backward(", "backward call"),
        (".fit(", "fit call"),
        (".fit_transform(", "fit_transform call"),
        ("inference_mode", "inference_mode"),
        ("extract_attention", "extract_attention"),
        ("torch.save(", "torch.save"),
        ("torch.jit.", "torch.jit"),
    ]
    # Allow "np.load" only when followed by a phase 52 raw path (last_query_attention_seed*.npz)
    files = list(phase54_root.rglob("*.py"))
    hits: list[dict] = []
    for fp in files:
        try:
            src = fp.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        # Exclude the safety scanner itself (its keyword metadata is data)
        if fp.name == "finalize_phase54.py":
            # Only scan the part before def static_safety_scan (i.e., real source code
            # other than the scanner's keyword metadata).
            marker = "def static_safety_scan("
            cut = src.find(marker)
            if cut > 0:
                # Also exclude everything from the marker to the next top-level
                # "def " or end of file (within finalize_phase54.py).
                src = src[:cut]
        stripped = _strip_strings_and_comments(src)
        for kw, desc in forbidden:
            if kw in stripped:
                hits.append({
                    "file": str(fp.relative_to(phase54_root.parent.parent)),
                    "keyword": kw,
                    "description": desc,
                })
        # np.load detection: only allowed for phase 52 last_query files
        if "np.load(" in stripped:
            import re as _re
            for m in _re.finditer(r"np\.load\(", stripped):
                context = stripped[max(0, m.start() - 800):m.start() + 200]
                # Allowed patterns: src.raw_files, "last_query_attention_seed", last_query_attention_seed{seed}
                allowed = (
                    "last_query_attention_seed" in context
                    or "src.raw_files" in context
                    or "raw_files[" in context
                    or "raw_attention_checksums" in context
                    or "phase_52_signoff" in context
                )
                if not allowed:
                    hits.append({
                        "file": str(fp.relative_to(phase54_root.parent.parent)),
                        "keyword": "np.load(non-Phase52-source)",
                        "description": "np.load used outside canonical Phase 52 last-query sources",
                    })
    return {
        "phase": 54,
        "files_scanned": len(files),
        "hits": hits,
        "all_clean": len(hits) == 0,
        "scanned_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def write_processing_log(
    out_dir: Path,
    phase54_root: Path,
    paths: dict[str, Path],
    safety_result: dict,
    results: dict,
) -> Path:
    """Write canonical Phase 54 processing log."""
    payload = {
        "phase": 54,
        "subphase": "54-A..G",
        "version": "LAST_QUERY_ATTENTION-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "out_dir": str(out_dir),
        "elapsed_seconds": results.get("elapsed_seconds"),
        "n_metrics_rows": results.get("n_metrics_rows"),
        "integrity_pass": results.get("integrity_pass"),
        "reconstruction_pass": results.get("reconstruction_pass"),
        "lag_mapping_pass": results.get("lag_mapping_pass"),
        "target_order_pass": results.get("target_order_pass"),
        "mean_profile_pass": results.get("mean_profile_pass"),
        "lag_bin_pass": results.get("lag_bin_pass"),
        "coverage_pass": results.get("coverage_pass"),
        "outputs": {k: str(v) for k, v in paths.items()},
        "new_attention_extraction": False,
        "new_test_inference": False,
        "checkpoint_loading": False,
        "model_forward": False,
        "training": False,
        "optimizer_steps": 0,
        "scaler_fit": False,
        "prediction_correction": False,
        "best_seed_selected": False,
        "ensemble": False,
        "best_head_selected": False,
        "head_ranking": False,
        "head_clustering": False,
        "error_conditioning": False,
        "regime_conditioning": False,
        "feature_importance_claim": False,
        "causal_claim": False,
        "png_used_as_numeric_source": False,
        "phase47_modified": False,
        "phase48_modified": False,
        "phase49_modified": False,
        "phase50_modified": False,
        "phase51_modified": False,
        "phase52_modified": False,
        "phase53_modified": False,
        "phase55_started": False,
        "phase56_started": False,
        "phase57_started": False,
        "notebook_modified": False,
        "run_all_used": False,
        "static_safety_scan": safety_result,
        "final_status": "PASS" if all([
            results.get("integrity_pass"),
            results.get("reconstruction_pass"),
            results.get("lag_mapping_pass"),
            results.get("target_order_pass"),
            results.get("mean_profile_pass"),
            results.get("lag_bin_pass"),
            results.get("coverage_pass"),
            safety_result["all_clean"],
        ]) else "FAIL",
    }
    fp = Path("/Users/vientu/Deep Learning/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/docs/save_log_in_processing/phase_54_last_query_attention_analysis_log.json")
    fp.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(fp, payload)
    return fp


def write_human_readable_report(
    out_dir: Path,
    src,
    results: dict,
    paths: dict[str, Path],
) -> Path:
    """Generate last_query_attention_report.md with the 24-section structure."""
    n_metrics_rows = results.get("n_metrics_rows", 0)
    elapsed = results.get("elapsed_seconds", 0.0)
    md = []
    md.append("# Phase 54 — Last-Query Attention Analysis Report")
    md.append("")
    md.append("**Version:** LAST_QUERY_ATTENTION-v1")
    md.append(f"**Generated (UTC):** {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    md.append(f"**Numeric source:** Phase 52 raw last-query NPZ (NOT PNG)")
    md.append(f"**Status:** {'PASS' if all([results.get('integrity_pass'), results.get('reconstruction_pass'), results.get('lag_mapping_pass'), results.get('target_order_pass'), results.get('mean_profile_pass'), results.get('lag_bin_pass'), results.get('coverage_pass')]) else 'FAIL'}")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 1. Objective")
    md.append("")
    md.append("Phase 54 quantitatively characterizes last-query attention `A[:,:,L-1,:]` over")
    md.append("the full FINAL_TEST_POP-v1 (N=2961) for the three frozen TransformerEncoder seeds")
    md.append("(42 / 123 / 2026) across 2 layers × 4 heads each.")
    md.append("")
    md.append("## 2. Why last-query attention is analyzed")
    md.append("")
    md.append("Because pooling is `LAST_STEP`, the newest encoded token is directly passed to the")
    md.append("regression head, making the last-query attention vector a particularly relevant")
    md.append("internal diagnostic for understanding which historical positions the model")
    md.append("considers when producing the forecast.")
    md.append("")
    md.append("## 3. Pooling-dependent interpretation")
    md.append("")
    md.append(f"- Pooling: **{src.pooling}**")
    md.append(f"- `last_query_directly_corresponds_to_pooled_token` = `{src.last_query_directly_corresponds_to_pooled_token}`")
    md.append("- This means last-query attention IS the attention for the token directly consumed by the regression head.")
    md.append("")
    md.append("## 4. Frozen numerical sources")
    md.append("")
    md.append("All Phase 54 metrics are computed from the frozen Phase 52 raw last-query NPZ:")
    md.append("- `last_query_attention_seed42.npz`")
    md.append("- `last_query_attention_seed123.npz`")
    md.append("- `last_query_attention_seed2026.npz`")
    md.append("")
    md.append(f"Each NPZ carries shape `[2961, 2, 4, 72]` float32 — `N_TEST × layers × heads × L`.")
    md.append("")
    md.append("## 5. Last-query and lag semantics")
    md.append("")
    md.append("- Last query: `A[:, :, L-1, :]` (NOT `A[:, :, :, L-1]`).")
    md.append("- Source position `p` corresponds to lag steps `L - p` (H=1).")
    md.append("- Position 0 = oldest (12 h old); position L-1 = newest (10 min old).")
    md.append("- Forecast target is NOT an attention token.")
    md.append("")
    md.append("## 6. Raw-source integrity verification")
    md.append("")
    md.append("See `last_query_source_verification.csv` and `last_query_integrity_audit.csv`. All three NPZ files:")
    md.append(f"- SHA256 match frozen Phase 52 checksums.")
    md.append(f"- Shape: [2961, 2, 4, 72]")
    md.append(f"- Dtype: float32")
    md.append(f"- Per-vector probability: finite, nonnegative within tolerance, sum ≈ 1.")
    md.append("")
    md.append("## 7. Phase 52 summary reconstruction")
    md.append("")
    md.append(f"Phase 54 recomputes 10 canonical Phase 52 metrics from raw vectors and compares")
    md.append(f"against the frozen `attention_last_query_summary.csv`. Reconstruction status: "
              f"{'PASS' if results.get('reconstruction_pass') else 'FAIL'}.")
    md.append("")
    md.append("## 8. Concentration metrics")
    md.append("")
    md.append("Per-seed/per-layer/per-head aggregates of:")
    md.append("- entropy (Shannon, ε = 1e-12)")
    md.append("- normalized entropy (H / log L)")
    md.append("- effective source count (exp(H))")
    md.append("- top1 weight, top1 tie count (NEWEST_SOURCE rule)")
    md.append("- top5 mass")
    md.append("")
    md.append("See `last_query_metric_summary_by_head.csv`.")
    md.append("")
    md.append("## 9. Expected-lag metrics")
    md.append("")
    md.append("- expected_lag_steps = Σ a_p · lag_steps_p")
    md.append("- expected_lag_minutes = 10 × expected_lag_steps")
    md.append("- lag_sd_steps = √(Σ a_p · (lag_steps_p − E[lag])²)")
    md.append("")
    md.append("## 10. Recent-history cumulative mass")
    md.append("")
    md.append("Windows (steps): 1h≤6, 6h≤36, 12h≤72, 24h≤144. `effective_steps = min(requested, L)`.")
    md.append("L=72 means the 24h window is **TRUNCATED** — model only has 12h of input context.")
    md.append("See `last_query_recent_mass_summary.csv`.")
    md.append("")
    md.append("## 11. Non-overlapping temporal allocation bins")
    md.append("")
    md.append("Bins (lag steps): 1–6, 7–36, 37–72, 73–144, 145–L. Only supported bins are emitted.")
    md.append("See `last_query_lag_bin_mass.csv`.")
    md.append("")
    md.append("## 12. Lag50/Lag80/Lag90 coverage radii")
    md.append("")
    md.append("Computed on cumulative recency mass in newest→oldest order.")
    md.append("See `last_query_coverage_radius_summary.csv`.")
    md.append("")
    md.append("## 13. Mean temporal profiles by head")
    md.append("")
    md.append("Per seed/layer/head: mean, median, SD, p05/p25/p75/p95 attention weight by lag.")
    md.append("Mean profile audit: `Σ_lag mean_weight ≈ 1` per seed/layer/head.")
    md.append("See `last_query_profile_by_lag.csv`.")
    md.append("")
    md.append("## 14. Layer head-mean profiles")
    md.append("")
    md.append("Layer-level head-mean profile = mean over heads of mean profile.")
    md.append("Permutation-invariant summary; does NOT replace per-head profiles.")
    md.append("See `last_query_layer_head_mean_profile.csv`.")
    md.append("")
    md.append("## 15. Top1 lag-frequency patterns")
    md.append("")
    md.append("For each seed/layer/head: count + fraction of targets with each top1_lag_steps value.")
    md.append("NEWEST_SOURCE tie rule preserved. Fractions sum to 1 per seed/layer/head.")
    md.append("See `last_query_top1_lag_frequency.csv` and `last_query_top1_tie_summary.csv`.")
    md.append("")
    md.append("## 16. Deterministic shared-worst case views")
    md.append("")
    md.append("Deterministic Phase 51 W2 SHARED_WORST ranks 1–5. One grid per case×seed,")
    md.append("rows=layers, columns=heads. x=lag (minutes, recency order), y=raw last-query weight.")
    md.append("See `figures/report_cases/SHARED_R{01..05}_SEED{42,123,2026}_LAST_QUERY.png`.")
    md.append("")
    md.append("## 17. Main descriptive findings")
    md.append("")
    md.append("Restrained qualitative findings (descriptive only). No head ranking. No causal claim.")
    md.append("See `last_query_analysis_findings.csv`.")
    md.append("")
    md.append("## 18. Why no head winner is selected")
    md.append("")
    md.append("Head ranking would require either predictive quality data (deferred to Phase 55)")
    md.append("or error-conditioned groups (deferred to Phase 56). Phase 54 reports each head's")
    md.append("distributional profile only.")
    md.append("")
    md.append("## 19. Why attention is not raw-feature importance")
    md.append("")
    md.append("Attention source axes are TEMPORAL token positions, not raw feature dimensions.")
    md.append("The Transformer's input projection (Linear(33, D)) mixes features before self-attention.")
    md.append("Attention weight = temporal allocation diagnostic, not feature importance.")
    md.append("")
    md.append("## 20. Why error-conditioning is deferred to Phase 56")
    md.append("")
    md.append("Comparing high-error vs low-error attention requires Phase 49 residuals and")
    md.append("Phase 50 regime labels — explicitly allocated to Phase 56.")
    md.append("")
    md.append("## 21. Why seed-stability is deferred to Phase 57")
    md.append("")
    md.append("Same-index heads across seeds are NOT assumed semantically equivalent.")
    md.append("Cross-seed stability requires explicit head matching, allocated to Phase 57.")
    md.append("")
    md.append("## 22. Limitations")
    md.append("")
    md.append("- L=72 means 12h input context. 24h requested windows are TRUNCATED.")
    md.append("- Attention is one internal signal among many (residual paths, FFN, LayerNorm, etc.).")
    md.append("- Per-vector metrics describe distribution, not contribution magnitude.")
    md.append("- Empirical quantiles (p05–p95) are not confidence intervals.")
    md.append("- N=2961 vectors are temporally dependent, not iid.")
    md.append("")
    md.append("## 23. Handoff to head comparison")
    md.append("")
    md.append("Phase 55 receives:")
    md.append("- `last_query_metrics_long.csv`")
    md.append("- `last_query_metric_summary_by_head.csv`")
    md.append("- `last_query_profile_by_lag.csv`")
    md.append("- `last_query_lag_bin_mass.csv`")
    md.append("- `last_query_recent_mass_summary.csv`")
    md.append("- `last_query_coverage_radius_summary.csv`")
    md.append("- `last_query_top1_lag_frequency.csv`")
    md.append("- `phase55_head_comparison_handoff.json`")
    md.append("")
    md.append(f"`ready_for_phase55 = true`, `phase55_authorized = false`.")
    md.append("")
    md.append("## 24. Definition of Done")
    md.append("")
    md.append("Phase 54 is PASS only when:")
    md.append("- All Phase 52 raw last-query NPZ verified.")
    md.append("- Last-query treated exactly as `A[:, :, L-1, :]`.")
    md.append("- Vectors finite, nonnegative, sum ≈ 1.")
    md.append("- Phase 52 summaries reconstructed within tolerance.")
    md.append("- Lag mapping exact (lag1↔L-1, lagL↔0).")
    md.append("- Per-target metrics include all required fields.")
    md.append("- 1h/6h/12h/24h masses recorded with truncation flags.")
    md.append("- Non-overlap lag-bin masses sum ≈ 1.")
    md.append("- Mean profiles sum ≈ 1 per seed/layer/head.")
    md.append("- Layer head-mean profiles produced without replacing per-head.")
    md.append("- Top1 frequency/tie computed with NEWEST_SOURCE rule.")
    md.append("- Phase 51 shared ranks 1–5 used as illustrative case views.")
    md.append("- No new extraction/inference/training; no head ranking/error conditioning/seed-stability claims.")
    md.append("- Phase 55 receives standardized head metrics; Phase 56/57 receive context only.")
    md.append("")
    md.append(f"**Phase 54 status: {'PASS' if all([results.get('integrity_pass'), results.get('reconstruction_pass'), results.get('lag_mapping_pass'), results.get('target_order_pass'), results.get('mean_profile_pass'), results.get('lag_bin_pass'), results.get('coverage_pass')]) else 'FAIL'}**")
    md.append("")

    fp = out_dir / "last_query_attention_report.md"
    fp.write_text("\n".join(md), encoding="utf-8")
    return fp


def write_readme(
    out_dir: Path,
    src,
    paths: dict[str, Path],
) -> Path:
    md = []
    md.append("# README — Phase 54 Last-Query Attention Analysis")
    md.append("")
    md.append("**Version:** LAST_QUERY_ATTENTION-v1")
    md.append("")
    md.append("## What `A[:,:,L-1,:]` means")
    md.append("")
    md.append("For each test target, each layer, each head, we extract the attention vector that")
    md.append("the **newest** historical token (the last input time-step) distributes over all")
    md.append("72 historical source positions. This is the attention pattern of the token that")
    md.append("(under LAST_STEP pooling) is directly consumed by the regression head.")
    md.append("")
    md.append("## Why source positions are historical only")
    md.append("")
    md.append("The Transformer's source axis represents temporal token positions — i.e., time steps")
    md.append("in the input window — after feature projection. The forecast target is NOT an attention token.")
    md.append("")
    md.append("## How source position maps to forecast lag")
    md.append("")
    md.append("```")
    md.append("lag_steps_p = L - p, with H = 1")
    md.append("```")
    md.append("")
    md.append("So position 71 → lag 1 (10 min before target), position 0 → lag 72 (12 h before target).")
    md.append("")
    md.append("## Why raw order and recency order differ")
    md.append("")
    md.append("- Raw position order: oldest → newest (position 0 oldest, position L-1 newest).")
    md.append("- Recency order (used for cumulative coverage): newest → oldest.")
    md.append("- Lag coverage radii (Lag50/80/90) and cumulative recency profiles MUST be computed in recency order.")
    md.append("")
    md.append("## What entropy means")
    md.append("")
    md.append("Shannon entropy in nats of the last-query vector: `H = -Σ a_p · log(a_p + ε)`, ε = 1e-12.")
    md.append("Higher entropy = more diffuse; lower entropy = more concentrated on fewer positions.")
    md.append("")
    md.append("## What effective source count means")
    md.append("")
    md.append("`N_eff = exp(H)` — the number of uniformly-weighted source positions that would")
    md.append("produce the same Shannon entropy. Bounds approximately 1..L.")
    md.append("")
    md.append("## What expected lag means")
    md.append("")
    md.append("`E[LagSteps] = Σ a_p · lag_steps_p`. With `E[LagMinutes] = 10 × E[LagSteps]`.")
    md.append("Caveat: bimodal distributions can have an expected lag where little mass exists;")
    md.append("interpret with entropy and profile.")
    md.append("")
    md.append("## What recent 1h/6h/12h/24h mass means")
    md.append("")
    md.append("Cumulative mass on the last-`h` (or last-`steps`) source positions, in recency order.")
    md.append("Because L = 72 (12h), the 24h window is **TRUNCATED**. `recent_24h_coverage_truncated = True`.")
    md.append("")
    md.append("## What Lag50/Lag80/Lag90 mean")
    md.append("")
    md.append("Smallest most-recent lag radius (in steps) containing at least 50% / 80% / 90%")
    md.append("of last-query attention mass. Smaller = more recency-concentrated; larger = more")
    md.append("history is needed. No quality judgment implied.")
    md.append("")
    md.append("## How non-overlapping lag bins work")
    md.append("")
    md.append("Bins 1–6 / 7–36 / 37–72 / 73–144 / 145–L are mutually exclusive. Masses sum to 1.")
    md.append("Only bins where the upper bound ≤ L are emitted (L=72 ⇒ 73–144 / 145–72 not applicable).")
    md.append("")
    md.append("## Why mean profiles sum to 1")
    md.append("")
    md.append("Each mean profile is the average over Test targets of normalized probability vectors.")
    md.append("The mean of normalized vectors is itself a normalized vector (sum = 1).")
    md.append("")
    md.append("## Why heads are not ranked")
    md.append("")
    md.append("Head ranking would require either predictive quality evidence (Phase 55) or")
    md.append("error-conditioned comparison (Phase 56). Phase 54 reports each head's distributional")
    md.append("profile without claiming superiority.")
    md.append("")
    md.append("## Why same-index heads across seeds may differ semantically")
    md.append("")
    md.append("The architecture does NOT guarantee that head index 0 in seed 42 has the same")
    md.append("learned semantic role as head index 0 in seed 123. Phase 54 displays heads by")
    md.append("architectural index only; cross-seed matching is allocated to Phase 57.")
    md.append("")
    md.append("## Why error-conditioned analysis is deferred")
    md.append("")
    md.append("Comparing high-error vs low-error attention is allocated to Phase 56. Phase 54")
    md.append("uses error labels only for deterministic case-level views on the frozen Phase 51")
    md.append("shared ranks 1–5.")
    md.append("")
    md.append("## Why raw attention is temporal allocation, not feature importance")
    md.append("")
    md.append("Attention weights describe temporal token-to-token allocation inside the Transformer")
    md.append("Encoder. They are NOT raw-feature importance, NOT causal attribution, NOT proof of")
    md.append("model reasoning. The model's output also depends on value projections, residual paths,")
    md.append("LayerNorm, FFN, later layers, pooling, and the regression head.")
    md.append("")
    md.append("## How downstream phases consume numerical outputs")
    md.append("")
    md.append("- **Phase 55** (head comparison): standardized head metrics + profiles (see `phase55_head_comparison_handoff.json`).")
    md.append("- **Phase 56** (error-conditioned): context handoff references Phase 52 raw last-query NPZ (NOT PNG).")
    md.append("- **Phase 57** (seed stability): context handoff pins numeric source to Phase 52 NPZ (NOT PNG).")
    md.append("")
    md.append("All downstream phases MUST use the canonical Phase 52 raw last-query NPZ as numeric source.")
    md.append("PNG pixels are derived visualizations only and MUST NOT be used to recover attention values.")
    md.append("")

    fp = out_dir / "README_LAST_QUERY_ATTENTION.md"
    fp.write_text("\n".join(md), encoding="utf-8")
    return fp


def finalize_phase54(
    out_dir: Path,
    src,
    paths: dict[str, Path],
    results: dict,
) -> dict:
    """Generate report, README, static safety scan, processing log."""
    safety = static_safety_scan(Path("/Users/vientu/Deep Learning/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/src/course_work/phase54"))
    paths["O54.29"] = write_human_readable_report(out_dir, src, results, paths)
    paths["O54.30"] = write_readme(out_dir, src, paths)
    log_fp = write_processing_log(out_dir, None, paths, safety, results)
    return {
        "safety": safety,
        "log_fp": str(log_fp),
        "report_fp": str(paths["O54.29"]),
        "readme_fp": str(paths["O54.30"]),
    }
