# Phase 47 — Final Test Evaluation

## Why Test First Opens Here

Phase 47 is the **first phase in which Test target values are authorized for access**.
All prior phases (1–46) kept Test targets locked. Phase 47 marks the transition from
development to the final held-out evaluation.

## Why All Three Seeds Are Evaluated

The three seeds (42, 123, 2026) were declared before any Test access.
All three FINAL_REFIT checkpoints are materializations of the same locked scientific
configuration. Each seed represents a different stochastic realization of the same
model. All three must be evaluated.

## Why No Best Seed Is Chosen

Selecting the best seed based on Test performance would be Test-driven model selection.
This violates the scientific contract. Reporting all three seeds plus their mean ± SD
shows stochastic sensitivity without introducing selection bias.

## Why Mean ± SD Is Reported

The three seeds are repeated materializations of the same model. The arithmetic mean
and sample standard deviation (ddof=1) summarize the sensitivity to random seed initialization.
This is descriptive variability, not a new ensemble model.

## Why No Seed Ensemble

An ensemble was not locked before Test access. Creating an ensemble post-hoc based on
Test scores would be Test-driven. The primary reporting uses per-seed metrics + mean ± SD.

## Why Scalers Are Not Refit

FINAL_SCALING-v1 was fitted on TRAIN+VALIDATION (FINAL_DEV) in Phase 46. It is reused
exactly in Phase 47. Refitting scalers on Test would leak Test statistics into preprocessing.

## WB0 Actual-History Semantics

WB0 allows the input lookback to include historical observed values from before the Test period,
as long as all input timestamps precede the target. This is not future leakage — it is
context carry-over from observed history.

## Persistence Semantics

Persistence uses the actual previous observed Appliances value (y_t) as the prediction for y_{t+1}.
This is the naive autoregressive baseline for the 10-minute-ahead forecasting task.

## LSTM Eligibility / Fairness Caveat

The LSTM_TUNED_DEV checkpoint from Phase 43 uses L36 lookback (not L72) and was tuned
on the development split. It is not a FINAL_REFIT symmetric with the Transformer.
Its comparison has an asymmetric training protocol caveat.

## Why Detailed Analyses Are Deferred

Detailed residual analysis, worst-error analysis, and attention extraction are handled in
Phases 48–52. These analyses use the frozen prediction bundles and do not require re-running
Test inference.

## Why Test Cannot Be Reused for Tuning

After Test metrics exist, the scientific configuration is permanently locked. Any post-Test
model or hyperparameter change invalidates Test as an untouched holdout.
