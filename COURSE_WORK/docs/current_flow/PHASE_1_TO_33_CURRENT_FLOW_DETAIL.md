# PHASE 1 TO 33 — Current Flow Detail

> Tài liệu chi tiết dòng chảy của từng phase từ 1 đến 33, dựa trên kiến trúc source code đã được refactor clean architecture.

---

## Phase 1 — Environment

**Module:** `src/course_work/utils/environment.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | `requirements.txt`, `pyproject.toml` |
| **Output** | `artifacts/environment/environment_report.json`, `phase_1_signoff.json`, `requirements_freeze.txt` |
| **Invariants** | Deterministic mode D0, MPS visibility drift controlled |

**Nhiệm vụ:**
- Capture Python, torch, sklearn, numpy versions
- Freeze requirements
- Smoke test imports
- Set deterministic seed

---

## Phase 2 — Data Acquisition

**Module:** `src/course_work/data/acquisition.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | `data/raw_data/source/appliances_energy_prediction.zip` |
| **Output** | `artifacts/acquisition/dataset_manifest.json`, `phase_2_signoff.json` |
| **Output (data)** | `data/raw_data/energydata_complete.csv` |

**Nhiệm vụ:**
- Verify SHA256 checksums
- Materialize raw CSV
- Generate dataset manifest

---

## Phase 3 — Schema Audit

**Module:** `src/course_work/data/schema.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | `data/raw_data/energydata_complete.csv` |
| **Output** | `artifacts/schema/schema_manifest.json`, `schema_summary.csv`, `variable_dictionary.csv`, `phase_3_signoff.json` |

**Nhiệm vụ:**
- Validate dtypes
- Detect schema drift
- Generate variable dictionary

---

## Phase 4 — Temporal Integrity Audit

**Module:** `src/course_work/data/temporal.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | `artifacts/schema/` |
| **Output** | `artifacts/temporal/temporal_manifest.json`, `daily_observation_counts.csv`, `interval_distribution.csv`, `phase_4_signoff.json` |

**Nhiệm vụ:**
- Verify cadence = 10 minutes
- Detect gaps, duplicates
- Continuity segments

---

## Phase 5 — EDA

**Module:** `src/course_work/data/eda.py` + `src/course_work/reporting/eda.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | `artifacts/temporal/` |
| **Output** | `artifacts/eda/{figures,tables,eda_manifest.json,eda_anomalies.json}`, `phase_5_signoff.json` |

**Nhiệm vụ:**
- Target distribution, ECDF, timeline
- Hourly, weekday profiles
- Correlation heatmap, autocorrelation
- Hypotheses generation

---

## Phase 6 — Feature Engineering

**Module:** `src/course_work/data/features.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | `artifacts/eda/` |
| **Output (data)** | `data/interim/uci_appliances_energy_prediction/energydata_feature_engineered_v1.csv` |
| **Output (artifacts)** | `artifacts/features/feature_engineering_manifest.json`, `feature_registry.csv`, `phase_6_signoff.json` |

**Nhiệm vụ:**
- Time features (hour, dayofweek, sin/cos)
- Lag features (selected lags only)
- Rolling statistics
- Train-only feature computation

---

## Phase 7 — Feature Set Variants

**Module:** `src/course_work/data/feature_sets.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | `artifacts/features/` |
| **Output** | `artifacts/feature_sets/feature_set_registry.json`, `phase_7_signoff.json` |

**Nhiệm vụ:**
- Define feature set variants (FS0, FS1, FS2)
- Track lineage
- Leakage audit

---

## Phase 8 — Chronological Split

**Module:** `src/course_work/data/splitting.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | `artifacts/feature_sets/` |
| **Output** | `artifacts/splits/{split_manifest.json, split_membership.csv, split_boundaries.csv}`, `phase_8_signoff.json` |

**Nhiệm vụ:**
- 70/15/15 chronological split
- Leakage audit
- Boundary neighborhood check

---

## Phase 9 — Train-Only Scaling

**Module:** `src/course_work/data/scaling.py` (also re-exported at `course_work.scaling`)

| Thành phần | Chi tiết |
|---|---|
| **Input** | `artifacts/splits/` |
| **Output (scalers)** | `artifacts/scalers/{x,y}/*.joblib` |
| **Output (artifacts)** | `artifacts/scaling/{scaling_manifest.json, scaler_registry.json}`, `phase_9_signoff.json` |

**Nhiệm vụ:**
- Fit scalers on Train only
- Scale Validation and Test using Train statistics
- SHA256 checksums

---

## Phase 10 — Window Builder

**Module:** `src/course_work/data/windows.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | `artifacts/scaling/` |
| **Output** | `artifacts/windows/window_manifest.json`, `window_population_summary.csv`, `phase_10_signoff.json` |

**Nhiệm vụ:**
- Build lookback windows
- Compute population summaries
- Leakage audit

---

## Phase 11 — DataLoaders

**Module:** `src/course_work/data/datasets.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | `artifacts/windows/` |
| **Output** | `artifacts/dataloaders/{dataloader_manifest.json, dataloader_registry.csv}`, `phase_11_signoff.json` |

**Nhiệm vụ:**
- Train / Validation / Test DataLoaders
- Sequential ordering preserved
- Shuffle reproducibility audit

---

## Phase 12 — Shared Metrics

**Module:** `src/course_work/evaluation/metrics.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | None (pure functions) |
| **Output** | `artifacts/metrics/{metric_manifest.json, metric_unit_tests.csv}`, `phase_12_signoff.json` |

**Nhiệm vụ:**
- MAE, RMSE, R² in original Wh units
- Reference examples
- Test firewall audit

---

## Phase 13 — Experiment Registry

**Module:** `src/course_work/experiments/registry.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | All upstream artifacts |
| **Output** | `artifacts/experiments/{experiment_registry.jsonl, run_artifact_registry.csv}`, `phase_13_signoff.json` |

**Nhiệm vụ:**
- Register every training run
- Track configs, metrics, status
- Family taxonomy

---

## Phase 14 — Persistence Baseline

**Module:** `src/course_work/baselines/persistence.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | `artifacts/dataloaders/`, `artifacts/scaling/` |
| **Output** | `artifacts/baselines/persistence/{persistence_manifest.json, persistence_validation_metrics.json}`, `phase_14_signoff.json` |

**Nhiệm vụ:**
- Persistence forecast: $\hat{y}[t+1] = y[t]$
- Compare with LSTM and Transformer baselines later

---

## Phase 15 — LSTM Implementation

**Module:** `src/course_work/models/lstm_regressor.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | None (architecture code) |
| **Output** | `artifacts/models/lstm/{lstm_model_manifest.json, lstm_shape_contract.json, lstm_unit_tests.csv}`, `phase_15_signoff.json` |

**Nhiệm vụ:**
- LSTM regressor (input → hidden → output)
- Shape contracts
- Parameter audit
- Unit tests

---

## Phase 16 — Transformer Implementation

**Module:** `src/course_work/models/transformer_regressor.py` + `transformer_encoder_layer.py` + `positional_encoding.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | None (architecture code) |
| **Output** | `artifacts/models/transformer/{transformer_model_manifest.json, transformer_shape_contract.json, ...}`, `phase_16_signoff.json` |

**Nhiệm vụ:**
- Transformer encoder regressor
- Positional encoding (sin/cos)
- Multi-head self-attention
- Shape + attention contracts

---

## Phase 17 — Attention Verification

**Module:** `src/course_work/attention/verification.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | Phase 16 outputs |
| **Output** | `artifacts/attention_verification/{attention_verification_manifest.json, attention_path_equivalence_audit.csv, attention_probability_audit.csv, ...}`, `phase_17_signoff.json` |

**Nhiệm vụ:**
- Verify attention weights sum to 1 per query
- Probability/mask audit
- Aggregation audit
- Path equivalence (manual vs framework)

---

## Phase 18 — Forward Sanity Tests

**Module:** `src/course_work/sanity/forward_sanity.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | Phase 16 + 17 |
| **Output** | `artifacts/forward_sanity/{forward_sanity_manifest.json, forward_batch_audit.csv, ...}`, `phase_18_signoff.json` |

**Nhiệm vụ:**
- Test forward pass correctness
- Batch independence
- Parameter mutation audit
- Device transfer

---

## Phase 19 — Training Engine

**Module:** `src/course_work/training/engine.py` + `losses.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | Phase 11, 15, 16 |
| **Output** | `artifacts/training_engine/{training_engine_manifest.json, training_engine_unit_tests.csv, ...}`, `phase_19_signoff.json` |

**Nhiệm vụ:**
- Generic training loop
- Checkpoint save/load (best + last)
- Early stopping
- Gradient clipping
- Resume support

---

## Phase 20 — LSTM Baseline Run

**Module:** `src/course_work/baselines/lstm_baseline.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | Phase 19 (training engine) |
| **Output** | `artifacts/baselines/lstm_baseline/{lstm_baseline_run_contract.json, lstm_baseline_discrepancies.json, ...}`, `phase_20_signoff.json` |
| **Output (runs)** | `artifacts/runs/RUN_LS_LS_*` |

**Nhiệm vụ:**
- Train reference LSTM
- Validation metrics vs persistence
- Compare to Transformer_B0

---

## Phase 21 — Transformer B0 Run

**Module:** `src/course_work/baselines/transformer_b0.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | Phase 19 |
| **Output** | `artifacts/transformer_b0/{transformer_b0_run_contract.json, ...}`, `phase_21_signoff.json` |
| **Output (runs)** | `artifacts/runs/RUN_TR_B0_*` |

**Nhiệm vụ:**
- Train reference Transformer B0
- Population audit (same Train/Val/Test as LSTM baseline)
- Validation metrics comparison

---

## Phase 22 — Learning Diagnostics

**Module:** `src/course_work/diagnostics/learning_diagnostics.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | Phase 20, 21 runs |
| **Output** | `artifacts/learning_diagnostics/{learning_diagnostics_manifest.json, learning_diagnostics_summary.csv}`, `phase_22_signoff.json` |

**Nhiệm vụ:**
- Train vs Validation loss curves
- Overfit detection
- Gradient norms

---

## Phase 23 — S1: Feature Set Sweep

**Module:** `src/course_work/experiments/phase_execution.py`

| Thành phần | Chi tiết |
|---|---|
| **Input** | Phase 7, 11, 19 |
| **Output** | `artifacts/sweeps/s1_feature_set/{sweep_manifest.json, results.csv, s1_feature_set_winner.json, s1_reference_update.json}`, `phase_23_signoff.json` |

**Nhiệm vụ:**
- Sweep over FS0, FS1, FS2
- Pick winner by validation RMSE
- Update reference config

---

## Phase 24 — S2: Time Feature Sweep

**Module:** `src/course_work/sweeps/` (via `experiments/phase_execution.py`)

| Thành phần | Chi tiết |
|---|---|
| **Output** | `artifacts/sweeps/s2_time_feature/{..., s2_time_feature_winner.json, s2_reference_update.json}`, `phase_24_signoff.json` |

**Nhiệm vụ:**
- Sweep time feature variants

---

## Phase 25 — S3: Target Scaling Sweep

| Thành phần | Chi tiết |
|---|---|
| **Output** | `artifacts/sweeps/s3_target_scaling/`, `phase_25_signoff.json` |

**Nhiệm vụ:**
- Sweep target scaling options

---

## Phase 26 — S4: Lookback Sweep

| Thành phần | Chi tiết |
|---|---|
| **Output** | `artifacts/sweeps/s4_lookback/`, `phase_26_signoff.json` |

**Nhiệm vụ:**
- Sweep `lookback ∈ {36, 72, 144}`

---

## Phase 27 — S5: Pooling Sweep

| Thành phần | Chi tiết |
|---|---|
| **Output** | `artifacts/sweeps/s5_pooling/`, `phase_27_signoff.json` |

**Nhiệm vụ:**
- Sweep pooling strategies (last, mean, max, attention)

---

## Phase 28 — S6: Activation Sweep

| Thành phần | Chi tiết |
|---|---|
| **Output** | `artifacts/sweeps/s6_activation/`, `phase_28_signoff.json` |

**Nhiệm vụ:**
- Sweep activation functions (ReLU, GELU, SiLU, ...)

---

## Phase 29 — S7: Batch Size Sweep

| Thành phần | Chi tiết |
|---|---|
| **Output** | `artifacts/sweeps/s7_batch_size/`, `phase_29_signoff.json` |

**Nhiệm vụ:**
- Sweep batch sizes

---

## Phase 30 — S8: Learning Rate Sweep

| Thành phần | Chi tiết |
|---|---|
| **Output** | `artifacts/sweeps/s8_learning_rate/`, `phase_30_signoff.json` |

**Nhiệm vụ:**
- Sweep learning rate values

---

## Phase 31 — S9: Weight Decay Sweep

**Module:** `src/course_work/sweeps/weight_decay.py`

| Thành phần | Chi tiết |
|---|---|
| **Output** | `artifacts/sweeps/S9_weight_decay/{sweep_manifest.json, s9_weight_decay_metrics.csv, s9_weight_decay_winner.json, s9_reference_update.json}`, `phase_31_signoff.json` |

**Nhiệm vụ:**
- Sweep weight decay values

---

## Phase 32 — S10: Dropout Sweep

**Module:** `src/course_work/sweeps/dropout.py`

| Thành phần | Chi tiết |
|---|---|
| **Output** | `artifacts/sweeps/S10_dropout/`, `phase_32_signoff.json` |

**Nhiệm vụ:**
- Sweep dropout rates

---

## Phase 33 — S11: d_model Sweep

**Module:** `src/course_work/sweeps/d_model.py`

| Thành phần | Chi tiết |
|---|---|
| **Output** | `artifacts/sweeps/S11_d_model/`, `phase_33_signoff.json` |

**Nhiệm vụ:**
- Sweep d_model (transformer hidden dimension)

---

## Tổng Kết Phases 1-33

| Nhóm | Số phases | Đặc điểm |
|---|---|---|
| **Foundation** (1-10) | 10 | Data pipeline: acquisition → features → splits → scaling → windows |
| **Modeling** (11-22) | 12 | DataLoaders, metrics, models, baselines, training engine, diagnostics |
| **Sweeps 1** (23-33) | 11 | S1-S11 hyperparameter sweeps, mỗi phase update reference config |

Tất cả các sweeps đều follow cùng pattern:
1. Read previous winner from reference
2. Run sweep variants
3. Pick winner by validation RMSE
4. Update reference for next sweep

---

**Tiếp theo:** Xem `PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md` để biết chi tiết phases 34–59.
