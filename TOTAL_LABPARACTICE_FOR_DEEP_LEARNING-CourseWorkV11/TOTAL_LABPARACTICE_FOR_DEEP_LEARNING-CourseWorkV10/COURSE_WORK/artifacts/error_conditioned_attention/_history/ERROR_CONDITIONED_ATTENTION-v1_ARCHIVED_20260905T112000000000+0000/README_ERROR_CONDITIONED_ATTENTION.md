# Phase 56 — Error-Conditioned Attention (ERROR_CONDITIONED_ATTENTION-v1)

## What this phase does

Phase 56 analyzes temporal attention of the Final Transformer **conditioned on realized Test forecast error**. It is strictly diagnostic.

## Why absolute error (AE) is the primary conditioning variable

- `AE = |residual| = |y_true - y_pred|`. Sign-neutral, interpretable in Wh, same basis as MAE, directly aligned with Phase 51 worst-case logic.

## Why signed residual is analyzed separately

- Signed residual `e = y_true - y_pred` separates UNDERPREDICTION (e > 0) from OVERPREDICTION (e < 0). EXACT_ZERO (e == 0) is retained but excluded from two-group contrasts.

## Why Phase 56 Test error cohorts are diagnostic rather than deployment regimes

- HIGH_ERROR can only be known after the forecast has been produced. It is a diagnostic cohort, not an operating regime.

## How rank-based 20/60/20 cohorts are created

- For each seed, sort all Test targets by `absolute_error_wh` ASC, then `timestamp` ASC, then `target_id` ASC.
- `n_edge = max(1, floor(0.20 * N))`.
- `LOW_ERROR = first n_edge`, `HIGH_ERROR = last n_edge`, `MID_ERROR = all remaining`.

## How exact rank deciles are assigned

- For zero-based rank index `r = 0..N-1`: `decile = 1 + floor(10*r/N)`, capped at 10.

## What shared hardness means

- `SharedHardness_t = (|e_42|+|e_123|+|e_2026|)/3` per target_id. Seed-invariant target-level difficulty diagnostic. NOT ensemble error.

## Why cohorts are frozen before attention joins

- Cohort assignments are derived purely from realized error. Attention cannot influence the cohort definition. An assignment SHA256 is computed and frozen BEFORE the join.

## Why Spearman is used

- Error and attention metrics may be nonlinear, skewed, heavy-tailed. Spearman captures monotonic association and is less dominated by extremes.
- No naive iid p-value is used as headline.

## What Cliff's delta means

- `delta = P(X_high > X_low) - P(X_high < X_low)`. Range `[-1, 1]`. No canned magnitude thresholds.

## How high/low temporal profiles are compared

- Cohort mean profile is computed as the mean attention vector over cohort members.
- Comparison: JSD (natural log), L1 distance, cosine similarity, Wasserstein distance in minutes.
- `D_HL(k) = P_HIGH(k) - P_LOW(k)` summed over k is verified to be ~0.

## Why Wasserstein is in minutes

- Lag support is in minutes (10 min cadence x 72 steps = 720 min = 12h).

## Why layer head-mean is useful

- Permutation-invariant to head order within a layer. Useful for cross-seed descriptive comparison WITHOUT semantic head alignment.

## Why per-head results are NOT averaged across seeds

- Same head index across seeds does NOT imply semantic alignment. Per-head cross-seed averaging is reserved for Phase 57 (after head matching).

## Why no p-value fishing is used

- Test observations are temporally dependent. Naive iid p-values are not headline evidence.

## Why no best head/retraining is allowed

- Phase 56 is diagnostic. No model is retrained. No prediction is corrected. No attention is re-extracted. No best head is selected.

## Why attention-error association is not causal

- Attention and error are co-observed outcomes of the same forward pass. Conditioning on realized error does not create causal identification.

## Output artifacts (O56.1-O56.38)

See `error_conditioned_attention_manifest.json` and `phase_56_signoff.json`.
