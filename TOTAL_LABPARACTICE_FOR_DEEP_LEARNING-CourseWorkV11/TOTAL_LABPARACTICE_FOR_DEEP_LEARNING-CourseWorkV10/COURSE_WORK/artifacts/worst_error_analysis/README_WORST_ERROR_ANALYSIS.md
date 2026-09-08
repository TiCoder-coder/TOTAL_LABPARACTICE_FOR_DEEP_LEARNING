# Phase 51 — Worst-Error Analysis (README)

## Purpose

Phase 51 performs descriptive, post-hoc worst-case error analysis on the final Transformer (FS2_TF1) Test predictions across three seeds (42 / 123 / 2026). It produces rankings, regime context, persistence context, local temporal context, exact 72×33 input-window reconstruction, casebook, findings, figures, and a Phase 52 handoff. It does NOT modify predictions, train a model, load checkpoints, or perform attention analysis.

## Residual Convention

```
residual = y_true - y_pred
positive = UNDERPREDICTION
negative = OVERPREDICTION
zero     = EXACT_ZERO
```

## Ranking Contract

- W1 = Per-seed Top-20 absolute error.
- W2 = Shared across 3 seeds.
- W3 = Top-10 UNDERPREDICTION (residual > 0).
- W4 = Top-10 OVERPREDICTION (residual < 0).
- Tie-break: `target_id ASC` (frozen).

## Casebook Layout

Case identifiers are deterministic:

```
CASE_{selection_family_short}_{seed}_rank{NNN}_{target_id}
```

The casebook index, membership bridge, and unique master are stored in:

- `casebook_index.csv` — per (family, seed, rank) row.
- `casebook_membership_bridge.csv` — case_id ↔ target_id mapping.
- `casebook_unique_case_master.csv` — unique target_id.

## Context Semantics

| Artifact | Coordinate | Source |
|---|---|---|
| `local_temporal_context.csv` | ±6 rows in time | Persistence Test predictions |
| `input_window_manifest.csv` | Contract only (lookback=72, FC=33, FS2_TF1) | frozen contracts |
| `exact_input_window_reconstruction.csv` | **RAW** and **MODEL_VISIBLE** | FEATURES-v1 + WINDOWPOP-v1 + FINAL_SCALING-v1 (transform_only) |
| `exact_input_windows_per_feature_summary.csv` | **RAW** + **MODEL_VISIBLE** per feature | reconstructed windows |
| `target_history_context.csv` | Historical Appliances, last 72 steps | FEATURES-v1 |

RAW and MODEL_VISIBLE coordinates are **separate**. Do not mix them.

## Frozen Artifacts

All Phase 51-B/C/D/E/F artifacts are sealed at chmod 0444. Their SHAs are recorded in `phase51_summary.json`. SHA drift in any frozen artifact blocks Phase 51-G signoff.

## Phase 52 Handoff

`phase52_attention_extraction_handoff.json` is the canonical handoff from Phase 51 to Phase 52. It is read-only and contains:

- final candidate lineage
- selection contract SHA256
- ranking / casebook / exact-input reconstruction SHAs
- FS2_TF1 feature order reference
- lookback=72, feature_count=33, WB0 protocol
- Phase 50 regime context reference
- residual convention
- LSTM canonical eligibility & verbatim reason
- safety statement: `phase52_authorized = false`

Phase 52 MUST NOT execute attention analysis without a separate human approval gate. The handoff merely provides the deterministic Phase 51 case universe Phase 52 may inspect after authorization.

## What Phase 51 Explicitly Did NOT Do

- It did NOT rerank or modify predictions.
- It did NOT retrain the model.
- It did NOT compute attention weights or heatmaps.
- It did NOT perform SHAP or other feature-importance analysis.
- It did NOT prescribe corrections / retraining fixes.
- It did NOT select a best seed.
- It did NOT ensemble predictions.
- It did NOT apply prediction correction to Test outputs.
- It did NOT modify Phase 47-50 artifacts.
- It did NOT modify the notebook (Phase 51-H is reserved for that).

## Provenance Map (subset)

| Output | Source frozen artifact |
|---|---|
| W1/W2 rankings | `worst_per_seed_top20.csv`, `worst_shared_top20.csv` |
| W3/W4 signed | `worst_underprediction_top10.csv`, `worst_overprediction_top10.csv` |
| Cross-seed overlap | `seed_overlap_table.csv` |
| Error concentration | `error_concentration_table.csv` |
| Hardness | `hardness_vs_seed_disagreement.csv`, `hardness_group_summary.csv` |
| Regime context | `regime_overrepresentation.csv` (frozen Phase 50 labels) |
| Baseline context | `baseline_context.csv` (Phase 47 Persistence test predictions) |
| LSTM context | `lstm_eligibility_context.json` (verbatim from canonical eligibility) |
| Casebook | `casebook_index.csv`, `casebook_unique_case_master.csv` |
| Exact input | `exact_input_window_reconstruction.csv` (reconstructed 72×33 values) |
| Figures | `figures/*.png` (deterministic names) |
| Findings | `phase51_findings.json` (descriptive only) |
| Handoff | `phase52_attention_extraction_handoff.json` |
| Signoff | `phase51_signoff.json` |
