# Limitations

## L1A_DATASET_SINGLE_HOUSE

**Category:** L1_DATASET

Single-house UCI Appliances dataset; no multi-house, multi-climate, or multi-building validation.

Why it matters: External validity cannot be established from a single dwelling.

Restricted claims: Performance / attention findings cannot be extrapolated to other households.

---

## L2A_FORECAST_H1

**Category:** L2_FORECASTING_DESIGN

H=1 one-step-ahead only (10 min); no recursive multi-step evaluation.

Why it matters: Long-horizon forecasting requires a different protocol.

Restricted claims: Performance does not characterize multi-step forecasts.

---

## L2B_WB0

**Category:** L2_FORECASTING_DESIGN

Final primary boundary is WB0: one-step forecasting assumes previously observed target values become available for subsequent predictions.

Why it matters: Open-loop multi-step deployment is not what the current model targets.

Restricted claims: Cannot imply fully open-loop multi-step forecasting.

---

## L3A_SEQUENTIAL_TUNING

**Category:** L3_MODEL_SELECTION_EVALUATION

Sequential one-factor tuning rather than full Cartesian / global hyperparameter search; selected hyperparameters do not guarantee the global optimum.

Why it matters: Alternative tuning strategies could yield different selected hyperparameters.

Restricted claims: Cannot claim optimal hyperparameters.

---

## L3B_THREE_SEEDS

**Category:** L3_MODEL_SELECTION_EVALUATION

Only three final seeds (42, 123, 2026); SD is descriptive only and does not characterize the full distribution over random initializations.

Why it matters: Three-seed SD is a limited stochastic robustness check.

Restricted claims: Stability findings are limited to the three predefined seeds.

---

## L4A_TIME_SERIES_DEPENDENCE

**Category:** L4_STATISTICAL

Time-series dependence; naive iid significance claims are not warranted. Diagnostics are descriptive.

Why it matters: Cannot use 'statistically significant' without a formal test.

Restricted claims: Effects reported as descriptive only.

---

## L5A_ATTENTION_TEMPORAL_ALLOCATION

**Category:** L5_ATTENTION_INTERPRETABILITY

Attention describes temporal token allocation only — NOT raw-feature importance and NOT causal explanation.

Why it matters: Misreading attention as feature importance or as causal would overclaim.

Restricted claims: Cannot be used for feature importance / causal claims.

---

## L5B_HEAD_MATCHING_NOT_FUNCTIONAL

**Category:** L5_ATTENTION_INTERPRETABILITY

Head-profile similarity does NOT prove functional equivalence; value / output projections can differ; matching ambiguity and partial cycle consistency are surfaced.

Why it matters: Head pruning / ablation implication would overclaim.

Restricted claims: No head-pruning recommendation.

---

## L6A_DEPLOYMENT_GENERALIZATION

**Category:** L6_DEPLOYMENT_GENERALIZATION

No prospective deployment, no online adaptation, no multi-house validation, no computational latency benchmark.

Why it matters: Deployment claims require separate evidence.

Restricted claims: Cannot be marked as deployment-ready.

---

## L6B_EXTERNAL_GENERALIZATION

**Category:** L6_DEPLOYMENT_GENERALIZATION

Results cannot establish generalization to other households, buildings, climates, or energy systems without external validation.

Why it matters: External generalization is out of scope of the current project.

Restricted claims: Cannot be generalized externally.

---

