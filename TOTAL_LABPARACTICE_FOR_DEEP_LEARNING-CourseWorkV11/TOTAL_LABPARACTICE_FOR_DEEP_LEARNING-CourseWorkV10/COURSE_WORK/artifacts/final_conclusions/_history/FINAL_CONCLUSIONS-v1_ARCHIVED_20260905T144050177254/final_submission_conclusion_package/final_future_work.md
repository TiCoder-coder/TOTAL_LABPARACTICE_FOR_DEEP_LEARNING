# Future Work

## FW1_MULTI_STEP

Direct multi-horizon, recursive, seq2seq / decoder, or probabilistic multi-step forecasting for H>1.

Linked limitation: L2A_FORECAST_H1

Priority: P2 | Not performed: TRUE

---

## FW2_EXTERNAL_VALIDATION

Evaluate on additional households / buildings, different seasons / climates, and out-of-distribution periods.

Linked limitation: L1A_DATASET_SINGLE_HOUSE

Priority: P1 | Not performed: TRUE

---

## FW3_ADDITIONAL_BASELINES

Additional model baselines (e.g. CNN, N-BEATS, classical statistical baselines).

Linked limitation: L3A_SEQUENTIAL_TUNING

Priority: P5 | Not performed: TRUE

---

## FW4_MORE_SEEDS

More seeds and stronger uncertainty characterization (e.g. tens of seeds, bootstrap CI on rolling-origin folds).

Linked limitation: L3B_THREE_SEEDS

Priority: P4 | Not performed: TRUE

---

## FW5_HEAD_ABLATION_ATTRIBUTION

Head ablation, Integrated Gradients or SHAP-style feature attribution, attention rollout, counterfactual temporal masking, and value-path analysis.

Linked limitation: L5A_ATTENTION_TEMPORAL_ALLOCATION

Priority: P3 | Not performed: TRUE

---

## FW6_FEATURE_ATTRIBUTION

Feature-level attribution methods on the input side.

Linked limitation: L5A_ATTENTION_TEMPORAL_ALLOCATION

Priority: P3 | Not performed: TRUE

---

## FW7_BLOCK_AWARE_INFERENCE

Temporal block-aware statistical inference (e.g. block-bootstrap, Newey-West covariance) to address time-series dependence.

Linked limitation: L4A_TIME_SERIES_DEPENDENCE

Priority: P4 | Not performed: TRUE

---

## FW8_ONLINE_DEPLOYMENT

Online / rolling deployment evaluation with prospective data.

Linked limitation: L6A_DEPLOYMENT_GENERALIZATION

Priority: P2 | Not performed: TRUE

---

## FW9_UNCERTAINTY_FORECASTING

Uncertainty-aware forecasting (quantile / probabilistic outputs).

Linked limitation: L4B_NO_FORMAL_UNCERTAINTY

Priority: P3 | Not performed: TRUE

---

## FW10_EFFICIENCY_LATENCY

Efficiency / latency analysis on representative hardware.

Linked limitation: L6A_DEPLOYMENT_GENERALIZATION

Priority: P5 | Not performed: TRUE

---

