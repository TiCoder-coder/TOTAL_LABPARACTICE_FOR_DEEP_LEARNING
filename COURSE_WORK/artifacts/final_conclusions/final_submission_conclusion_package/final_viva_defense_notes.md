# Viva / Defense Notes

## Why Transformer?
A Transformer Encoder was required by the coursework specification. It provides a
sequence-to-One multivariate regression model with multi-head self-attention,
which offers an interpretable temporal diagnostic via attention analysis.

## Why LSTM baseline?
The coursework required a comparison with an LSTM baseline. The LSTM was tuned
via sequential hyperparameter sweeps in Phase 43 and served as the learned
baseline against which the Transformer was compared.

## Why H=1?
The primary forecasting horizon is H=1 (10-minute one-step-ahead). Multi-step
forecasting would require a different experimental protocol and is listed as
future work.

## Why WB0?
The WB0 (walking-bar-0) boundary means that for one-step-ahead forecasting,
the previously observed target values are assumed available. This is standard
for H=1 regression and matches the UCI dataset's design.

## Why three seeds?
Three seeds (42, 123, 2026) were declared before final runs. They provide a
limited stochastic robustness check. Only three seeds were used because the
final model lock was applied after the development phase.

## Why no ensemble?
An ensemble would average multiple seed predictions into a single forecast.
The three-seed summary reports mean ± SD of independent runs. Averaging
predictions would be a new model decision not authorized in the protocol.

## What does attention mean?
Attention describes temporal token-to-token allocation in the encoder's last
layer. It is NOT raw-feature importance and does NOT establish causal
contribution. It provides a view of which historical time steps the model
weighted most heavily for each prediction.

## Why head matching?
Different heads within the same layer can learn different temporal profiles.
JSD-based exhaustive permutation matching (within each layer) was used to
identify corresponding heads across seeds. This does not prove functional
equivalence — value and output projections may differ.

## What is the strongest limitation?
The single-household dataset limits external validity. Results cannot be
generalized to other households, buildings, or climates without additional
evaluation.

## What would you do next?
Evaluate on additional households (external validity); extend to multi-step
forecasting; apply head ablation or feature attribution; use more seeds;
evaluate prospective deployment.
