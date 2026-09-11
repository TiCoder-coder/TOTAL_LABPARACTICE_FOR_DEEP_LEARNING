# Plan_improve_model.md

## Kế hoạch audit, refactor và cải thiện mô hình Deep Learning Coursework

**Project:** UCI Appliances Energy Prediction — Multivariate Time-Series Regression  
**Current primary model:** Transformer Encoder Regression (`TRANSFORMER-v1`)  
**Improvement track:** `MODEL_IMPROVEMENT-v2`  
**Nguyên tắc:** Không phá vỡ hoặc ghi đè evidence của V1; mọi cải tiến V2 phải có lineage, config, artifact và evaluation protocol riêng.

---

# 0. Mục tiêu của kế hoạch

Kế hoạch này không nhằm “thử thêm vài learning rate” một cách rời rạc. Mục tiêu là xây dựng một **Model Improvement V2** có quy trình khoa học rõ ràng để:

1. xác định đúng bottleneck hiện tại của Transformer;
2. sửa các inconsistency/provenance issue trước khi tuning;
3. cải thiện đồng thời **RMSE, MAE và R²** thay vì chỉ tối ưu RMSE;
4. giảm hiện tượng **prediction smoothing / range compression / one-step lag**;
5. cải thiện khả năng dự báo **high-consumption, rapid-change và upward-spike regimes**;
6. tận dụng ưu thế rất mạnh của Persistence mà vẫn giữ khả năng học nonlinear correction của Transformer;
7. mở rộng search space và chuyển từ one-factor-at-a-time tuning sang controlled multi-factor search;
8. refactor training engine để hỗ trợ scheduler, warm-up, staged fine-tuning và differential learning rates;
9. thử các architecture time-series phù hợp hơn nếu Transformer-v2 vẫn bị bottleneck;
10. giữ nguyên scientific integrity của coursework: **không dùng Test đã nhìn thấy để tiếp tục tune rồi gọi đó là unbiased held-out evaluation**.

---

# 1. Current-state audit — những gì mô hình hiện tại đang làm được và chưa làm được

## 1.1. Final Transformer V1 hiện tại

Cấu hình final đã khóa:

| Thành phần | Giá trị hiện tại |
|---|---|
| Feature set | `FS2_TF1` |
| Input features | 33 |
| Lookback | 72 steps = 12 hours |
| Horizon | 1 step = 10 minutes |
| Boundary | `WB0_CONTEXT_CARRY_OVER` |
| Target scaling | `YS1` standardisation |
| d_model | 64 |
| Attention heads | 4 |
| Encoder layers | 2 |
| FFN dimension | 256 |
| Activation | GELU |
| Dropout | 0.1 |
| Pooling | LAST_STEP |
| Loss | MSE |
| Optimizer | AdamW |
| Learning rate | `3e-4` |
| Weight decay | `1e-3` |
| Gradient clipping | global L2 max norm 1.0 |
| Scheduler | None |
| Warm-up | None |
| Final epochs | 30 fixed epochs |
| Final seeds | 42, 123, 2026 |

Current Transformer architecture is approximately **102k trainable parameters** for the selected F256 configuration. It is therefore not a large Transformer; however, several capacity sweeps ended at the largest value tested, so current evidence does **not** establish that 102k parameters are sufficient.

---

## 1.2. Final held-out performance của V1

### Transformer — 3 final seeds

| Seed | MAE (Wh) | RMSE (Wh) | R² |
|---:|---:|---:|---:|
| 42 | 29.529 | 64.943 | 0.489 |
| 123 | 27.115 | 61.986 | 0.535 |
| 2026 | 28.942 | 64.560 | 0.495 |
| **Mean** | **28.529** | **63.830** | **0.506** |

### Persistence baseline

| MAE (Wh) | RMSE (Wh) | R² |
|---:|---:|---:|
| **26.738** | **66.837** | **0.459** |

### Interpretation

Transformer V1:

- giảm RMSE khoảng **3.01 Wh (~4.5%)** so với Persistence;
- tăng R² khoảng **+0.047**;
- nhưng MAE lại **cao hơn 1.79 Wh (~6.7%)**.

Điều này rất quan trọng: mô hình hiện tại **không uniformly outperform Persistence**. Nó có khả năng giảm một số large errors đủ để cải thiện RMSE/R², nhưng lại trả giá bằng sai số trung bình lớn hơn ở rất nhiều mẫu bình thường/flat.

**Implication cho V2:** Không nên chỉ tối ưu RMSE. V2 cần giữ lợi ích của Transformer ở spikes/dynamics nhưng không được làm mất ưu thế của Persistence ở vùng ổn định.

---

# 2. Root-cause analysis — nguyên nhân chính khiến chỉ số hiện tại còn hạn chế

## 2.1. Target có tính autoregressive rất mạnh

Autocorrelation / one-step relationship của `Appliances` rất mạnh:

- TRAIN lag-1: ~0.759
- VALIDATION lag-1: ~0.741
- TEST lag-1: ~0.729

Persistence vì vậy là một baseline rất mạnh cho H=1.

V1 Transformer đã nhận historical `Appliances`, nhưng kiến trúc hiện tại vẫn dự báo trực tiếp:

```text
historical multivariate window
        ↓
Transformer
        ↓
LAST_STEP representation
        ↓
Linear(d_model → 1)
        ↓
y_hat(t+1)
```

Nó **không có explicit persistence skip connection**:

```text
y_hat(t+1) = y(t) + correction
```

Đây là một missing inductive bias quan trọng đối với bài toán 10-minute one-step forecasting.

### Hypothesis

Mô hình đang phải học lại từ đầu một quy luật rất đơn giản mà Persistence đã biết: “nếu không có change signal đủ mạnh, dự báo gần với giá trị hiện tại”.

**Priority:** Rất cao.

---

## 2.2. Prediction bị range compression và smoothing

Diagnostics cho thấy:

| Seed | Pred std / True std | Pred range / True range |
|---:|---:|---:|
| 42 | 0.740 | 0.721 |
| 123 | 0.754 | 0.675 |
| 2026 | 0.779 | 0.744 |

Tức là mô hình chỉ tạo ra khoảng **74–78% độ biến thiên chuẩn** và khoảng **68–74% range** của target thật.

Change diagnostics:

- actual mean absolute change: **26.75 Wh**;
- Transformer: khoảng **19.53–21.31 Wh**;
- actual p95 absolute change: **140 Wh**;
- Transformer p95: khoảng **91.5–118.8 Wh**.

### Interpretation

Model đang **quá smooth / quá conservative**. Nó dự báo khá hợp lý ở vùng bình thường nhưng không phản ứng đủ mạnh khi consumption thay đổi nhanh.

### Root causes có thể gồm

1. direct MSE regression kéo prediction về conditional mean;
2. single linear head quá đơn giản;
3. dominant flat/normal samples khiến spike samples ít ảnh hưởng;
4. no explicit delta/change objective;
5. no residual-to-persistence mechanism;
6. constant LR có thể dẫn đến suboptimal convergence;
7. input representation có thể không làm nổi bật local derivative/change signals.

**Priority:** Rất cao.

---

## 2.3. Model có dấu hiệu “lagging” target một bước

Lag diagnostics trên Test cho thấy correlation giữa prediction và ground truth cao nhất ở **lag -1**:

- seed 42: corr lag -1 ≈ 0.829; lag 0 ≈ 0.703
- seed 123: corr lag -1 ≈ 0.867; lag 0 ≈ 0.734
- seed 2026: corr lag -1 ≈ 0.853; lag 0 ≈ 0.708

Điều này phù hợp với observation rằng model rất gần persistence-like dynamics nhưng chưa học correction đủ tốt tại change points.

### Consequence

V2 phải tập trung vào:

- **delta prediction**;
- **direction-of-change**;
- **rapid-change detection**;
- hoặc explicit blending với Persistence.

---

## 2.4. Sai số tập trung vào spikes và rapid changes

Mean RMSE across final seeds:

| Regime | Transformer RMSE (Wh) |
|---|---:|
| TL_LOW | 20.73 |
| TL_MID | 23.50 |
| TL_HIGH | 116.26 |
| EXTREME_HIGH | 192.02 |
| CHANGE_NORMAL | 30.58 |
| CHANGE_RAPID | 201.17 |
| DIR_FLAT | 17.29 |
| DIR_UP | 92.37 |
| DIR_DOWN | 54.50 |

Worst-error analysis còn cho thấy:

- Top-20 cases chiếm khoảng **32–34% total SSE**;
- 17/20 worst cases được chia sẻ giữa cả ba final seeds;
- worst cases overrepresented mạnh ở `EXTREME_HIGH`, `CHANGE_RAPID`, `DIR_UP`.

### Interpretation

Đây không chỉ là seed noise. Failure pattern mang tính **structural**.

### Consequence

Model V2 cần một mechanism dành riêng cho dynamics, thay vì chỉ tăng `d_model`.

---

## 2.5. Transformer tốt hơn Persistence ở một số dynamic regimes nhưng tệ hơn ở flat/up regimes

Persistence regime RMSE:

- `CHANGE_RAPID`: ~229.89 Wh → Transformer ~201.17 (Transformer tốt hơn)
- `EXTREME_HIGH`: ~196.55 → Transformer ~192.02 (Transformer tốt hơn nhẹ)
- `TL_HIGH`: ~123.28 → Transformer ~116.26
- `DIR_FLAT`: Persistence = **0**, Transformer ~17.29
- `DIR_UP`: Persistence ~87.27, Transformer ~92.37 (Transformer tệ hơn)

### Core insight

Hai model có **complementary strengths**:

- Persistence cực mạnh khi series flat/stable;
- Transformer có lợi thế khi dynamics phức tạp/large changes.

Đây là lý do kiến trúc **Persistence-aware residual/gated model** được xếp Priority 1.

---

# 3. Diagnostic experiment rất quan trọng: Persistence + Transformer có tính bổ sung

> **CẢNH BÁO:** Các phép tính dưới đây được thực hiện sau khi Test đã được mở, vì vậy chỉ được dùng để hình thành hypothesis. Không được dùng như unbiased final evidence cho Model V2.

Nếu average prediction của 3 seed V1:

- MAE ≈ **27.14**
- RMSE ≈ **61.74**
- R² ≈ **0.538**

Chỉ riêng seed ensemble đã cải thiện đáng kể so với mean single-seed result.

Một diagnostic equal blend:

```text
prediction = 0.5 * Transformer_3seed_mean
           + 0.5 * Persistence
```

cho khoảng:

- MAE ≈ **25.20 Wh**
- RMSE ≈ **60.09 Wh**
- R² ≈ **0.563**

Một số post-hoc weights quanh 0.6–0.7 Transformer cho RMSE ~59.7, nhưng **không được dùng** vì weight đã được quan sát trên Test.

### Kết luận từ diagnostic này

Không phải “Transformer cần mạnh hơn ở mọi mẫu”. Khả năng cao V2 cần học:

> **khi nào nên giữ Persistence và khi nào cần correction.**

Đây là hypothesis quan trọng nhất của toàn improvement plan.

---

# 4. Feature audit — vấn đề FS2_TF1 và random controls

## 4.1. `rv1` và `rv2` không phải predictive variables bình thường

UCI dataset documentation ghi rõ hai biến random được đưa vào để **test regression models và filter non-predictive attributes**.

Trong actual data của coursework:

```text
rv1 == rv2  → True cho toàn dataset
corr(rv1, rv2) ≈ 1.0
```

Next-step correlation với `Appliances(t+1)`:

| Split | rv1 corr | rv2 corr |
|---|---:|---:|
| TRAIN | -0.014 | -0.014 |
| VALIDATION | -0.009 | -0.009 |
| TEST | -0.014 | -0.014 |

Tuy vậy S1 đã chọn `FS2_TF1` vì một single-seed Validation run:

- `FS1_TF1`: RMSE 61.03
- `FS2_TF1`: RMSE 59.43

### Risk

Đây có thể là:

- random optimization variance;
- chance validation fit;
- duplicated random-noise regularization effect;
- hoặc spurious selection do single validation segment.

Không có đủ evidence để xem `rv1/rv2` là real predictive features.

### V2 action

`rv1` và `rv2` phải trở lại đúng vai trò **negative controls**.

Không xóa ngay dựa trên cảm tính. Thực hiện robust ablation:

```text
FS1_TF1  vs  FS2_TF1
× rolling-origin folds
× multiple seeds
```

Nếu lợi ích của FS2 không consistent, V2 final feature set phải loại bỏ cả `rv1` và `rv2`.

**Priority:** Critical.

---

# 5. Temporal distribution shift audit

Dataset thay đổi đáng kể theo thời gian.

Ví dụ:

| Variable | TRAIN mean | VALIDATION mean | TEST mean |
|---|---:|---:|---:|
| T_out | 5.72 | 8.22 | 14.51 |
| RH_out | 82.58 | 72.39 | 73.90 |
| lights | 4.55 | 2.12 | 1.98 |
| Appliances std | 106.86 | 92.25 | 90.89 |

Đây là một genuine chronological distribution shift, phần lớn phản ánh thời điểm trong năm và thay đổi environmental conditions.

### Consequences

1. single static validation segment không đủ để chọn hyperparameter ổn định;
2. rolling-origin phải trở thành **primary development protocol**, không chỉ final robustness check;
3. context-relative/delta features có thể generalise tốt hơn absolute levels;
4. chỉ tăng model capacity có nguy cơ overfit chronology của TRAIN.

### Recommended response

- 5-fold hoặc ít nhất 3-fold expanding rolling-origin cho V2 tuning;
- fold-local scaling;
- evaluate worst fold, not only mean;
- consider robust/delta features before complex normalization;
- không ưu tiên lại RevIN hiện tại vì S18 đã làm RMSE tệ hơn ~4.59 Wh.

---

# 6. Sweep audit — tại sao S1–S19 chưa đủ để kết luận “đã tune xong”

Current workflow chủ yếu là **sequential one-factor-at-a-time (OFAT)**.

Điều này có hai vấn đề:

1. hyperparameters interact;
2. mỗi winner trở thành reference cho sweep tiếp theo, nên một early local choice có thể khóa search path.

## 6.1. Search space có nhiều winner nằm ở upper boundary

- `d_model`: 64 thắng 32; **64 là max đã thử**;
- heads: 4 thắng 2; **4 là max**;
- layers: 2 thắng 1; **2 là max**;
- FFN: 256 thắng 64/128; **256 là max**.

Đặc biệt:

- layers 2 cải thiện RMSE khoảng 1.70 Wh so với 1 layer;
- FFN 256 cải thiện tiếp so với 128;
- capacity trend chưa plateau rõ.

### Implication

Cần mở rộng moderate capacity:

```text
d_model: 64, 96, 128
layers:   2, 3, 4
heads:    4, 8 (khi divisible)
FFN:      256, 384, 512
```

nhưng phải làm **sau khi** residual/persistence head và training scheduler được refactor, nếu không ta chỉ scale một formulation đang sai bias.

---

## 6.2. MAE và RMSE ranking liên tục diverge

Ví dụ:

### Learning rate

- `3e-4`: RMSE 58.08, MAE 27.59
- `1e-3`: RMSE 58.41, MAE **25.73**

### d_model

- D64: RMSE 58.08, MAE 27.60
- D32: RMSE 58.46, MAE **25.70**

### dropout

- 0.1: RMSE 58.08, MAE 27.60
- 0.3: RMSE 58.36, MAE **26.18**

### Interpretation

Optimizing RMSE-only đã tạo ra model ưu tiên giảm large errors nhưng không tối ưu typical sample error.

### V2 action

Use a **Pareto / constrained selection rule**:

1. primary objective: pooled rolling-origin RMSE;
2. guardrail: MAE không được degrade đáng kể;
3. secondary: worst-fold RMSE;
4. diagnostic guardrail: rapid-change/high-level RMSE.

Không tạo một weighted composite score tùy ý ngay từ đầu. Dùng Pareto front và lexicographic constraints dễ giải thích hơn.

---

## 6.3. Weight decay sweep gần như không có meaningful difference

S9:

- WD0: 58.0847
- WD1: 58.0839
- WD2: 58.0819

Chênh lệch chỉ vài phần nghìn Wh.

### Conclusion

`weight_decay=1e-3` không nên được xem là một discovery mạnh. Đây gần như metric noise ở mức single run.

### V2 action

- weight decay là low-priority tuning factor;
- chỉ tune joint với LR/dropout;
- test `1e-5 – 1e-2` log-scale nhưng không dành nhiều trials.

---

## 6.4. Epoch-cap sweep không phải bottleneck

E50 và E100 có cùng best epoch = 12 vì early stopping.

=> tăng max epochs đơn thuần không giúp.

Vấn đề thật là:

- scheduler = None;
- warm-up = None;
- patience/min_delta chưa tune cùng scheduler.

---

## 6.5. Gradient clipping là hữu ích và nên giữ

S17:

- OFF: RMSE ~58.88
- clip norm 1.0: RMSE ~57.70

### V2

Giữ clipping mặc định; chỉ tune nhẹ:

```text
0.5, 1.0, 2.0
```

Không quay lại `OFF` trừ ablation.

---

## 6.6. RevIN hiện tại không nên ưu tiên lại

S18:

- RevIN OFF: 57.70
- RevIN ON: 62.29

Chênh lệch ~4.59 Wh.

### V2

Không tái sử dụng nguyên implementation RN1 hiện tại.

Nếu sau này cần distribution-shift normalization, phải coi đó là **new design** (ví dụ context-delta features, robust scaler, alternative local normalization), không phải rerun S18.

---

# 7. Provenance / artifact inconsistencies cần sửa trước V2

## 7.1. LSTM Phase 43 lineage inconsistency

Đã phát hiện các artifact hiện tại không hoàn toàn tự nhất quán:

- `lstm_shared_data_contract.json`: lookback 144;
- `phase_43_signoff.json`: shared lookback 36;
- tuned LSTM config: `FS2_TF1`, feature_count 33;
- nhưng lineage trong một số tuned LSTM artifacts trỏ `XSCALER__FS1_TF1`.

Ngoài ra một số LSTM winner artifacts có field values khác nhau giữa historical/corrective outputs.

### Action — P0 blocker

Trước khi dùng LSTM làm benchmark V2:

1. reconstruct exact Phase43 handoff;
2. resolve intended lookback from Phase42 handoff;
3. verify feature variant and actual scaler file dimensions;
4. strict-load winner checkpoint;
5. rerun inference on its exact registered Validation population;
6. recompute MAE/RMSE/R²;
7. regenerate one canonical `LSTM_TUNING-v2_AUDITED` handoff;
8. add contract tests ensuring:
   - feature count matches scaler count;
   - feature set ID matches scaler bundle ID;
   - shared lookback matches signoff;
   - model input size matches transformed data.

**Do not tune new LSTM architecture before this audit passes.**

---

## 7.2. Separate development config and final-refit config more explicitly

`final_model_scientific_config.json` still contains development training fields such as early stopping/max_epochs, while `final_training_recipe.json` correctly specifies final refit = 30 fixed epochs, no validation, no early stopping.

V2 nên tách rõ:

```text
ModelConfig
DevelopmentTrainingConfig
FinalRefitConfig
EvaluationConfig
```

để tránh report/code hiểu nhầm.

---

# 8. Scientific rule quan trọng nhất cho Model V2: Test V1 đã được nhìn thấy

Current Test đã được sử dụng trong:

- final metrics;
- prediction analysis;
- residual analysis;
- regime analysis;
- worst-error analysis;
- attention diagnostics;
- và chính plan V2 này dùng các diagnostics đó để hình thành hypotheses.

Do đó:

> **Test cũ không còn là unbiased unseen test đối với V2.**

### Required versioning

Freeze toàn bộ V1:

```text
TRANSFORMER-v1
artifacts/final_model_lock/
artifacts/three_seed_final_runs/
artifacts/final_test/
artifacts/prediction_analysis/
...
```

Không overwrite.

Tạo V2 riêng:

```text
configs/improvement_v2/
artifacts/improvement_v2/
src/course_work/models/transformer_v2.py
...
```

### V2 evidence policy

Primary V2 development evidence:

- TRAIN + VALIDATION chronology;
- nested / expanding rolling-origin;
- fold-local scaling;
- no use of old Test target for selection.

Old Test:

- có thể dùng cuối cùng như **post-hoc historical benchmark**;
- phải ghi rõ không còn independent vì V2 hypotheses đã chịu ảnh hưởng từ Test diagnostics.

Muốn có unbiased final evaluation thực sự cho V2 cần:

1. new future observations; hoặc
2. external household/dataset; hoặc
3. một truly untouched data source not previously inspected.

---

# 9. Target success criteria cho V2

Các criterion chính phải được đánh giá **trên rolling-origin development**, không dựa vào old Test.

## 9.1. Minimum acceptance criteria

V2 candidate chỉ được promote nếu:

1. pooled RO RMSE cải thiện **>= 2%** so với V1 equivalent;
2. RO MAE không tệ hơn V1;
3. worst-fold RMSE không tăng;
4. improvement xuất hiện ở >= 2/3 folds;
5. improvement không chỉ đến từ một seed;
6. no leakage / provenance audit PASS.

## 9.2. Preferred criteria

- pooled RMSE gain >= 3–5%;
- MAE đồng thời giảm;
- rapid-change RMSE giảm >= 5–10%;
- high-level RMSE giảm;
- prediction std ratio tiến gần 0.9–1.0;
- change amplitude ratio cải thiện;
- seed SD không tăng đáng kể.

## 9.3. Stretch target — chỉ để định hướng, không dùng tune trên Test

Dựa trên diagnostic synergy hiện tại, một V2 tốt có thể hướng tới vùng:

- RMSE < ~60 Wh;
- MAE < Persistence (~26.7 Wh);
- R² > ~0.55.

Đây là **aspirational benchmark**, không phải criterion được phép optimize trực tiếp trên old Test.

---

# 10. Improvement Roadmap — thứ tự triển khai đề xuất

---

# Phase I0 — Freeze V1 và repair audit/provenance

## Goal

Bảo đảm baseline reference chính xác trước khi tạo V2.

## Tasks

### I0.1 Freeze V1

Tạo manifest:

```text
artifacts/improvement_v2/v1_baseline_snapshot.json
```

Lưu checksum của:

- final config;
- final checkpoints;
- final metrics;
- test population;
- final scalers;
- all current conclusions.

### I0.2 Audit LSTM Phase43

Theo Section 7.1.

### I0.3 Run full test suite

Trước refactor:

```bash
pytest -q
```

Lưu output summary.

### I0.4 Add V2 configuration contract

Create:

```text
configs/improvement_v2/model_improvement_contract.json
```

Contract phải định nghĩa:

- allowed data regions;
- old Test policy;
- feature variants;
- architecture options;
- scheduler options;
- loss options;
- search budget;
- seed policy;
- promotion criteria.

### Exit criteria

- V1 immutable snapshot PASS;
- LSTM lineage either repaired or formally marked ineligible;
- full tests PASS;
- V2 contract signed.

---

# Phase I1 — Refactor evaluation diagnostics trước khi train model mới

## Goal

Không chỉ nhìn MAE/RMSE/R² sau mỗi run.

## Add development-only metrics

For every Validation / RO fold record:

1. MAE;
2. RMSE;
3. R²;
4. bias / mean residual;
5. prediction std ratio;
6. range ratio;
7. mean abs change ratio;
8. direction agreement;
9. high-target RMSE;
10. rapid-change RMSE;
11. flat-regime RMSE;
12. worst 1% SSE share.

**Regime thresholds must be derived from TRAIN/fold-train only.**

## Files

Create:

```text
src/course_work/evaluation/improvement_metrics.py
src/course_work/evaluation/regime_metrics.py
```

Tests:

```text
tests/unit/test_improvement_metrics.py
tests/unit/test_regime_threshold_train_only.py
```

### Why

Nếu không có các diagnostics này, optimizer có thể tiếp tục giảm RMSE bằng cách sacrifice MAE/flat regions mà ta không phát hiện đến final Test.

---

# Phase I2 — Clean feature baseline và feature engineering V2

## I2.1 Remove random controls from primary predictive candidate

Create canonical V2 clean baseline:

```text
FS_CLEAN = FS1_TF1
```

Keep `rv1/rv2` only as negative-control experiment.

### Experiment

```text
FS1_TF1 vs FS2_TF1
× RO folds
× seeds 42, 123, 2026 (for finalists)
```

Promotion rule:

`rv1/rv2` chỉ được giữ nếu gain stable across folds/seeds. Nếu không, remove.

---

## I2.2 Add explicit change features

High priority features:

```text
appliances_delta_1 = y_t - y_{t-1}
appliances_abs_delta_1
appliances_delta_2
```

Rolling context computed with **past/current only**:

```text
appliances_roll_mean_3
appliances_roll_mean_6
appliances_roll_mean_12
appliances_roll_std_6
appliances_roll_std_12
appliances_roll_max_12
appliances_roll_min_12
```

Why:

Current model sees raw sequence, nhưng explicit derivatives giúp model nhận ra change state mà không phải học phép trừ thông qua attention + projection.

---

## I2.3 Add daily seasonal lag

ACF at 24h (144 steps) remains positive across splits:

- TRAIN ~0.208
- VALIDATION ~0.225
- TEST ~0.251

Final lookback L72 chỉ nhìn 12h, nên không trực tiếp thấy same-time previous day.

Add:

```text
Appliances_lag_144
```

Optional later:

```text
Appliances_lag_1008   # 7 days
```

Weekly lag chỉ nên thử sau daily lag vì:

- giảm eligible early samples nhiều hơn;
- dataset chỉ ~4.5 months;
- weekly pattern weaker/less stable.

### Fairness requirement

Feature with long lag làm thay đổi eligible population. Tất cả comparisons phải dùng **common target population**.

---

## I2.4 Add context-relative environmental features

Low/medium priority:

```text
T_out_delta_1
RH_out_delta_1
lights_delta_1
selected room temperature/humidity deltas
```

Không tạo hàng chục redundant rolling features ngay từ đầu.

Use staged ablation.

---

## I2.5 Proposed feature variants

```text
FV2_A = FS1_TF1                                  # clean baseline
FV2_B = FV2_A + target delta features
FV2_C = FV2_B + rolling target stats
FV2_D = FV2_C + lag_144
FV2_E = FV2_D + selected exogenous deltas
```

Evaluate incrementally using same V2 model reference.

---

# Phase I3 — Refactor training engine V2

Current training engine uses constant AdamW LR.

This is one of the clearest optimization gaps.

## I3.1 Add scheduler abstraction

Create:

```text
src/course_work/training/schedulers.py
```

Interface example:

```text
SchedulerConfig:
    name
    warmup_epochs / warmup_steps
    min_lr
    factor
    patience
    T_max
```

Support initially:

### A. Constant LR

Keep as control.

### B. Warm-up + Cosine Decay — primary Transformer experiment

Recommended grid:

```text
max_lr:       3e-4, 5e-4, 7e-4, 1e-3
warmup_ratio: 0.00, 0.05, 0.10
min_lr:       1e-5
```

Do not exhaustively Cartesian-test all 12 immediately. Use random/TPE or staged shortlist.

### C. ReduceLROnPlateau

```text
initial_lr: 3e-4, 5e-4, 1e-3
factor:     0.5
patience:   3–5
min_lr:     1e-5
```

### Optional later

OneCycleLR.

---

## I3.2 Expand learning-rate search

Existing LR search only tested:

```text
1e-4, 3e-4, 1e-3
```

V2 pilot:

```text
1e-4
2e-4
3e-4
5e-4
7e-4
1e-3
```

Important: LR must be tuned jointly with scheduler and batch size, because current evidence already shows RMSE/MAE tradeoff.

---

## I3.3 Early stopping with scheduler

Current patience = 10, min_delta = 0.

For scheduler experiments:

```text
patience: 15–20
min_delta_rmse: 0.05–0.10 Wh
```

Reason:

- scheduler cần thời gian giảm LR trước khi early-stop;
- min_delta 0 có thể react to negligible noise.

Do not increase epoch cap alone.

Recommended max cap:

```text
60–100 epochs
```

but actual stop controlled by scheduler + early stopping.

---

## I3.4 Keep gradient clipping and tune only locally

Grid:

```text
0.5, 1.0, 2.0
```

Do not prioritize OFF because existing evidence clearly disfavors it.

---

## I3.5 Batch-size interaction

Current tested only 32 vs 64.

V2:

```text
16, 32, 64
```

Test jointly with LR.

Do not assume smaller batch always better.

---

## I3.6 Add epoch-level LR logging

`training_history.csv` must include:

```text
epoch
train_loss
val_rmse_wh
val_mae_wh
learning_rate
grad_norm_preclip
fraction_batches_clipped
```

This is mandatory to diagnose whether scheduler really helps.

---

# Phase I4 — Primary architecture improvement: Residual-to-Persistence Transformer

**Priority: Highest.**

## I4.1 Base residual formulation

Instead of direct forecasting:

```text
y_hat = Model(X)
```

use:

```text
delta_hat = Model(X)
y_hat = y_t + delta_hat
```

where `y_t` is the most recent observed `Appliances` value.

### Why

For H=1, Persistence is strong. Residual formulation gives model a simple default:

```text
no predicted change → persistence
```

Model capacity can focus on predicting **correction/change**, especially spikes.

---

## I4.2 Correct scaling implementation

Do **not** blindly use the X-scaled `Appliances` channel as persistence base unless scaler equivalence is formally verified.

Recommended dataset output:

```text
x_scaled
y_target_scaled
last_appliances_raw
delta_target_raw
sample_id
```

Then transform persistence base with **Y scaler**:

```text
y_t_yspace = YScaler.transform(last_appliances_raw)
delta_yspace = (y_{t+1} - y_t) / YScaler.scale
```

This ensures output and residual live in the same target model space.

---

## I4.3 Files

Create:

```text
src/course_work/models/transformer_v2.py
src/course_work/models/residual_forecasting.py
```

Possible class:

```text
ResidualTransformerRegressor
```

Config:

```text
prediction_mode = DIRECT | RESIDUAL
```

Tests:

- zero delta => exact Persistence;
- target row excluded;
- raw-to-Y-space roundtrip;
- no future leakage;
- shape `[B,1]`.

---

# Phase I5 — Persistence-gated Transformer

After residual model, test a learnable gate.

## I5.1 Gated blend

```text
neural_prediction = y_t + delta_hat

g = sigmoid(gate_head(h))

y_hat = (1-g) * y_t + g * neural_prediction
```

Equivalent:

```text
y_hat = y_t + g * delta_hat
```

### Interpretation

- `g ≈ 0` → trust Persistence;
- `g ≈ 1` → apply full neural correction.

This architecture maps directly to observed complementary behavior.

## I5.2 Gate inputs

Start simple:

- pooled Transformer state only.

Optional later:

- recent `|delta|`;
- rolling std;
- last consumption level.

Do not hand-code Test-derived regime thresholds into gate.

---

# Phase I6 — Upgrade regression head

Current head:

```python
nn.Linear(d_model, 1)
```

Test V2 head:

```text
LayerNorm
→ Linear(d_model, 2*d_model)
→ GELU
→ Dropout
→ Linear(2*d_model, 1)
```

or smaller:

```text
Linear(d_model, d_model)
→ GELU
→ Dropout
→ Linear(d_model, 1)
```

### Why

Current Transformer encoder may learn useful representation but a single linear map can be insufficient to translate it into nonlinear correction magnitude.

### Ablation

```text
HEAD_LINEAR
HEAD_MLP
HEAD_RESIDUAL_LINEAR
HEAD_RESIDUAL_MLP
HEAD_GATED_RESIDUAL
```

Run this **before** large capacity expansion.

---

# Phase I7 — Capacity expansion and normalization architecture

Only after I4–I6 produce a stable formulation.

## I7.1 Expand capacity moderately

Recommended search bounds:

```text
d_model:     64, 96, 128
num_layers:  2, 3, 4
num_heads:   4, 8 when divisible
ffn_dim:     256, 384, 512
dropout:     0.05, 0.10, 0.20
```

Avoid immediately creating multi-million parameter model because training dataset is only ~14k rows/windows per canonical development split.

Set a temporary cap such as:

```text
trainable_parameters < ~1 million
```

unless evidence supports going larger.

---

## I7.2 Add Pre-LN option

Current encoder forcibly rejects `norm_first=True` and is Post-LN.

For deeper V2 configurations, add:

```text
norm_order = POST_NORM | PRE_NORM
```

Why:

Pre-LN often produces more stable gradient flow for deeper Transformers; Post-LN commonly benefits more from warm-up.

### Experiment order

1. current 2-layer Post-LN control;
2. 3-layer Post-LN + warm-up;
3. 3-layer Pre-LN;
4. 4-layer Pre-LN only if 3-layer helps.

Do not conflate capacity and norm-order in one uncontrolled change.

---

## I7.3 Pooling

Current implementation only supports:

```text
LAST_STEP
MEAN
```

LAST_STEP already won and is appropriate for H=1.

Therefore pooling is **not a first-priority bottleneck**.

Only if residual/gated model plateaus, test:

- learned gated pooling;
- query pooling / attention pooling;
- concat(last, mean) followed by MLP.

---

# Phase I8 — Loss redesign: improve spikes without destroying MAE

## I8.1 Do not simply replace MSE with current Huber

Current Huber delta=1 standardized (~106.85 Wh equivalent) was not tuned and slightly worsened RMSE.

Therefore do not rerun the same setting expecting a different outcome.

---

## I8.2 Hybrid level + delta objective — recommended

For residual model:

```text
L_level = MSE(y_hat, y)
L_delta = SmoothL1(delta_hat, delta_true)

L_total = L_level + λ_delta * L_delta
```

Grid:

```text
λ_delta = 0.10, 0.25, 0.50
```

This directly attacks the smoothing/change problem.

---

## I8.3 Optional MSE + MAE/SmoothL1 hybrid

```text
L = α*MSE + (1-α)*SmoothL1
```

Test:

```text
α = 0.5, 0.75, 0.9
```

Purpose:

- retain RMSE sensitivity to spikes;
- reduce typical absolute error.

---

## I8.4 Change-aware sample weighting — medium priority

Weights must be derived from **fold-train only**.

Example:

```text
normal change: weight 1.0
above train q90 |delta|: weight 1.25
above train q95 |delta|: weight 1.5
```

or continuous clipped weight:

```text
w = 1 + alpha * clip(|delta| / q95_train, 0, 1)
```

with:

```text
alpha = 0.25, 0.5
```

Do not use large weights initially because that can worsen flat/normal MAE.

---

# Phase I9 — Multi-task change prediction

If residual model still under-reacts to rises/spikes, add auxiliary tasks.

## I9.1 Delta regression head

Already covered by I8; can have a dedicated auxiliary head if main head predicts level.

## I9.2 Direction head

Classes:

```text
DOWN / FLAT / UP
```

Targets must be defined from training data only and consistently across folds.

Possible auxiliary loss:

```text
L = L_level + λ_delta*L_delta + λ_dir*CrossEntropy(direction)
```

Start:

```text
λ_dir = 0.05, 0.10
```

Do not make direction classification primary objective; it is an auxiliary representation constraint.

---

# Phase I10 — Fine-tuning / freezing / unfreezing plan

## Important rule

Current Transformer is trained **from scratch**. Therefore freezing random encoder layers at the beginning is not appropriate.

Layer freezing only makes sense as a **warm-start experiment** after loading a trained V1/V2 checkpoint.

## Recommended staged fine-tuning experiment

Use only after new residual/MLP head is introduced.

### Stage FT-A — Head adaptation

Load V1 encoder weights.

Freeze:

```text
input_projection
positional_encoding (no trainable params currently)
encoder layers
```

Train only new residual/MLP/gate head:

```text
3–5 epochs
head LR = 1e-3
```

Purpose: initialize new head without disturbing encoder.

### Stage FT-B — Unfreeze top encoder block

Unfreeze last encoder layer.

Differential LR:

```text
head LR       = 3e-4
last block LR = 1e-4
```

Train ~5–10 epochs with scheduler.

### Stage FT-C — Full fine-tuning

Unfreeze all layers.

Possible discriminative LR:

```text
head:          1e-4
last encoder:  5e-5
lower encoder: 3e-5
input proj:    3e-5
```

Cosine decay to low LR.

### Mandatory comparison

Compare staged warm-start V2 against:

```text
same V2 architecture trained from scratch
```

If warm-start only converges faster but not better, prefer scratch for scientific simplicity.

---

# Phase I11 — Replace OFAT with controlled multi-factor search

## 11.1 Search strategy

Do not run a massive Cartesian grid.

Use staged search:

### Stage A — Cheap hypothesis screening

- one development seed (42);
- 1–2 representative inner folds;
- 10–20 trials;
- reject clearly weak options.

### Stage B — Robust shortlist

Top 5 configurations:

- all rolling-origin folds;
- same fixed seed;
- complete diagnostics.

### Stage C — Seed robustness

Top 2–3 configs:

```text
seeds 42, 123, 2026
× all RO folds
```

### Stage D — Final V2 lock

Select by:

1. pooled RMSE;
2. MAE guardrail;
3. worst-fold RMSE;
4. parameter/runtime context;
5. regime behavior.

---

## 11.2 Suggested search space

After residual architecture is established:

```text
prediction_mode:
  RESIDUAL
  GATED_RESIDUAL

head:
  LINEAR
  MLP

feature_variant:
  FV2_A ... FV2_D

lr:
  loguniform 1e-4 to 1e-3

scheduler:
  WARMUP_COSINE
  REDUCE_ON_PLATEAU

batch:
  16, 32, 64

d_model:
  64, 96, 128

layers:
  2, 3

ffn_dim:
  256, 384, 512

dropout:
  0.05, 0.10, 0.20

weight_decay:
  1e-5 to 1e-2 log-scale

grad_clip:
  0.5, 1.0, 2.0

loss:
  MSE
  MSE_PLUS_DELTA
  HYBRID
```

Only add layer=4 after evidence supports deeper model.

---

# Phase I12 — Improve validation protocol

## Current issue

Hyperparameter decisions often come from a single Validation segment and one seed before rolling-origin robustness is performed later.

## V2 protocol

Rolling-origin becomes central during search.

Suggested 5-fold expanding pattern if computational budget permits:

```text
Fold 1: early train → next validation
Fold 2: expanded train → next validation
...
Fold 5: latest pre-Test train → latest pre-Test validation
```

If runtime is limited, keep 3 folds but increase multi-seed confirmation.

### Fold-local requirements

Each fold must fit separately:

- X scaler;
- Y scaler;
- train-derived regime thresholds;
- any robust scaler/normalizer statistics.

No global future statistics.

---

# Phase I13 — Stronger baselines before adopting a much more complex Transformer

V1 rolling-origin evidence showed tuned LSTM is already highly competitive with Transformer candidates.

Therefore add/calibrate simple models.

## 13.1 Persistence-correction linear model — very high value

```text
y_hat = y_t + Linear(engineered_change_features)
```

This tells us how much benefit comes simply from explicit residual formulation.

## 13.2 Ridge / ElasticNet with lag features

Use V2 lag/delta features under same chronology.

## 13.3 DLinear / linear time-series baseline

Relevant because time-series literature has shown that simple linear models can be unexpectedly competitive with Transformer forecasters on some datasets.

## 13.4 TCN — recommended deep baseline

Temporal Convolutional Network:

- causal/dilated convolutions;
- strong local temporal inductive bias;
- highly relevant to 10-minute one-step dynamics;
- good candidate when current Transformer is too smooth/lagged.

**Priority among new architectures: TCN > large exotic Transformer.**

---

# Phase I14 — Alternative time-series architectures if V2 Transformer plateaus

Only proceed after I4–I13.

## 14.1 TSMixer / PatchTSMixer

Why relevant:

- efficient time/feature mixing;
- lower complexity;
- potentially suitable for a relatively small dataset.

## 14.2 iTransformer

Current Transformer treats each timestamp as a token and linearly fuses 33 heterogeneous variates into one temporal token.

iTransformer reverses this perspective and models variates as tokens, which may better capture cross-variable relationships.

Why potentially useful here:

- temperature/humidity/weather variables have distinct physical semantics;
- current single projection can mix them too early.

## 14.3 PatchTST

PatchTST can compress temporal sequences into patches and allow larger receptive fields.

Potential value:

- investigate L144 or longer histories without attention cost exploding;
- exploit daily context more efficiently.

But current task is H=1 and dataset is small, so PatchTST is **Tier-3**, not first choice.

## 14.4 TimeMixer

Optional if multiscale dynamics remain a bottleneck.

Do not implement multiple sophisticated models simultaneously. Select one after strong V2 baselines.

---

# Phase I15 — Ensemble strategy

Current 3-seed prediction average shows a clear diagnostic improvement.

V2 should formalize ensemble as a legitimate candidate **if coursework rules allow it**.

## 15.1 Seed ensemble

```text
prediction = mean(pred_seed42, pred_seed123, pred_seed2026)
```

Evaluate on development RO first.

## 15.2 Persistence + neural ensemble

Pre-register simple weights on development:

```text
alpha ∈ {0.25, 0.5, 0.75}

y_hat = alpha * neural + (1-alpha) * persistence
```

Do not tune alpha on old Test.

## 15.3 Learned gate preferred

If gated residual architecture works, prefer it over hand-selected constant blend because gate can adapt by sample.

---

# Phase I16 — Model selection must become multi-metric but still scientifically simple

## Recommended promotion logic

### Step 1

Sort by pooled rolling-origin RMSE.

### Step 2

Reject candidate if:

- MAE materially worse than V1;
- worst-fold RMSE worse;
- rapid-change RMSE deteriorates severely;
- artifacts/provenance fail.

### Step 3

Among remaining candidates, use:

- RMSE;
- MAE;
- worst fold;
- seed variance;
- parameter count/runtime.

### Avoid

- manually inventing one composite score after seeing results;
- choosing best seed;
- choosing candidate from one unusually good fold.

---

# 17. Exact refactor map — files nên sửa/tạo

## 17.1. Configuration

Create:

```text
configs/improvement_v2/model_improvement_contract.json
```

Optional:

```text
configs/improvement_v2/search_space.json
configs/improvement_v2/promotion_policy.json
```

---

## 17.2. Data / features

Refactor/create:

```text
src/course_work/data/features_v2.py
src/course_work/data/feature_sets_v2.py
src/course_work/data/datasets_v2.py
```

Responsibilities:

- delta features;
- rolling features;
- lag144;
- negative-control exclusion;
- return persistence base / delta target;
- train-only threshold fitting.

Do **not** silently modify `FEATURES-v1` artifacts.

---

## 17.3. Models

Create:

```text
src/course_work/models/transformer_v2.py
src/course_work/models/residual_forecasting.py
src/course_work/models/regression_heads.py
```

Options:

```text
DIRECT
RESIDUAL
GATED_RESIDUAL
```

Add:

```text
PRE_NORM / POST_NORM
LINEAR / MLP head
```

---

## 17.4. Training

Create/refactor:

```text
src/course_work/training/schedulers.py
src/course_work/training/objectives.py
src/course_work/training/parameter_groups.py
```

Modify `engine.py` to support:

- scheduler step per batch/epoch;
- ReduceLROnPlateau after validation;
- warm-up;
- multiple parameter groups;
- layer-wise LR;
- freeze/unfreeze stage transitions;
- learning-rate logging;
- optional sample weights;
- auxiliary losses.

Keep V1 engine behavior backward-compatible.

---

## 17.5. Evaluation

Create:

```text
src/course_work/evaluation/improvement_metrics.py
src/course_work/evaluation/model_selection_v2.py
```

---

## 17.6. Search / experiments

Create:

```text
src/course_work/experiments/improvement_v2.py
src/course_work/experiments/search_v2.py
src/course_work/experiments/v2_registry.py   # or reuse existing registry with new family
```

Prefer reusing current registry if it can isolate experiment family cleanly.

Experiment IDs:

```text
RUN_V2_...
```

Artifacts:

```text
artifacts/improvement_v2/runs/
artifacts/improvement_v2/sweeps/
artifacts/improvement_v2/rolling_origin/
```

---

## 17.7. Verification/tests

Add at minimum:

```text
tests/unit/test_v2_feature_no_future_leakage.py
tests/unit/test_random_controls_excluded.py
tests/unit/test_lag144_population.py
tests/unit/test_residual_prediction_identity.py
tests/unit/test_gated_residual.py
tests/unit/test_scheduler.py
tests/unit/test_lr_parameter_groups.py
tests/unit/test_freeze_unfreeze.py
tests/unit/test_multitask_loss.py
tests/integration/test_v2_training_chain.py
tests/integration/test_v2_rolling_origin.py
tests/contracts/test_v2_test_policy.py
```

---

# 18. Notebook strategy

Do **not** immediately rewrite Phases 0–59, vì đây là frozen scientific history của V1.

Recommended development notebook:

```text
notebook_course_work/Model_Improvement_v2.ipynb
```

Suggested sections:

```text
I0  V1 snapshot & audit
I1  V2 contract
I2  clean feature ablation
I3  new temporal features
I4  scheduler/LR experiments
I5  residual model
I6  gated residual model
I7  head/capacity experiments
I8  loss / multi-task experiments
I9  joint search
I10 rolling-origin shortlist
I11 seed robustness
I12 ensemble comparison
I13 V2 lock
I14 evaluation & report handoff
```

Sau khi V2 được chấp nhận, mới quyết định cách integrate vào final coursework report/codebase.

---

# 19. Experiment priority matrix

| Priority | Experiment | Expected value | Cost | Recommendation |
|---|---|---:|---:|---|
| P0 | Fix Phase43 LSTM provenance | Scientific integrity | Low-Med | **Do first** |
| P0 | Freeze V1 / create V2 contract | Scientific integrity | Low | **Do first** |
| P1 | Remove/ablate rv1-rv2 | High | Low | **Immediate** |
| P1 | Residual-to-Persistence head | Very high | Low-Med | **Primary architecture** |
| P1 | Gated residual blend | Very high | Medium | **Primary architecture** |
| P1 | Warmup + cosine / LR refactor | High | Medium | **Immediate** |
| P1 | Multi-fold validation during tuning | Very high | Compute | **Required** |
| P1 | MLP regression head | Medium-High | Low | Test with residual |
| P2 | Delta / rolling features | High | Medium | After clean baseline |
| P2 | Lag144 feature | Medium-High | Medium | Strong periodic hypothesis |
| P2 | Hybrid level+delta loss | High | Medium | After residual model |
| P2 | Joint LR/batch/dropout search | High | Compute | Replace OFAT |
| P2 | d_model 96/128, layer 3 | Medium-High | Compute | After formulation fix |
| P2 | Pre-LN | Medium | Medium | Especially layer>=3 |
| P2 | Seed ensemble | Medium-High | Low | Validate on development |
| P3 | Weighted spike loss | Medium | Medium | Carefully guard MAE |
| P3 | Direction auxiliary head | Medium | Medium | If change issue remains |
| P3 | TCN | High potential | Medium | Strong alternative baseline |
| P3 | iTransformer | Medium-High | High | If cross-variate representation issue |
| P4 | PatchTST / TSMixer / TimeMixer | Unknown | High | Only after simpler methods |
| P4 | RevIN current design | Low | Medium | **Do not prioritize** |
| P4 | Pure epoch-cap increase | Low | Low | **Not useful alone** |

---

# 20. Recommended first experiment batch — không làm quá nhiều cùng lúc

Sau khi P0 audit pass, chạy đúng thứ tự sau:

## Batch A — establish clean baseline

1. current V1 architecture + `FS1_TF1` clean;
2. current V1 architecture + `FS2_TF1` control;
3. same training settings;
4. rolling-origin evaluation.

Question:

> random controls có thực sự giúp robustly không?

---

## Batch B — residual hypothesis

Using winner clean feature set:

1. DIRECT + linear head;
2. RESIDUAL + linear head;
3. RESIDUAL + MLP head;
4. GATED_RESIDUAL + MLP head.

Same optimizer initially.

Question:

> persistence-aware inductive bias có cải thiện cả RMSE và MAE không?

---

## Batch C — training optimization

Take best architecture from Batch B.

Test:

```text
Constant 3e-4
Cosine warmup maxLR 3e-4
Cosine warmup maxLR 5e-4
Cosine warmup maxLR 1e-3
ReducePlateau initial 1e-3
```

Do not mix capacity changes yet.

---

## Batch D — features

1. clean baseline;
2. + delta;
3. + rolling;
4. + lag144.

---

## Batch E — capacity + joint search

Only after A–D identify a good V2 formulation.

---

# 21. Suggested multi-factor finalists

Instead of hundreds of combinations, start with structured candidates.

## Candidate V2-C0 — Minimal residual

```text
FS1_TF1
d_model=64
layers=2
heads=4
ffn=256
Residual head
MSE
AdamW
warmup+cosine
```

## V2-C1 — Residual MLP

```text
same + MLP head
```

## V2-C2 — Gated residual

```text
same + learned persistence gate
```

## V2-C3 — Change-aware

```text
V2-C2
+ delta features
+ level+delta loss
```

## V2-C4 — Daily-context

```text
V2-C3
+ lag144
```

## V2-C5 — Moderate capacity

```text
V2-C3 or C4
d_model=96/128
layers=3
ffn=384/512
Pre-LN
```

Then robustly compare them rather than continuously inheriting one sequential winner.

---

# 22. Staged fine-tuning decision tree

```text
Did architecture change only at head?
    |
    +-- YES --> warm-start experiment allowed
    |            |
    |            +-- freeze encoder → train head
    |            +-- unfreeze last block
    |            +-- full low-LR fine-tune
    |
    +-- NO --> train from scratch first

Did warm-start beat scratch on RO folds?
    |
    +-- YES --> retain as candidate
    +-- NO  --> discard freezing strategy
```

Không biến freezing thành một bước bắt buộc chỉ vì nó phổ biến trong pretrained NLP/CV.

---

# 23. What NOT to do

1. **Không tune tiếp trên old Test.**
2. Không chọn best seed.
3. Không tăng layers/d_model trước khi sửa persistence/change formulation.
4. Không giữ `rv1/rv2` chỉ vì một single Validation sweep thắng.
5. Không chạy lại RevIN RN1 unchanged.
6. Không chỉ tăng epoch cap.
7. Không chỉ chọn hyperparameter theo RMSE single seed.
8. Không overwrite V1 artifacts.
9. Không gọi old Test “untouched Test” cho V2.
10. Không implement 4 architecture mới cùng lúc.
11. Không freeze random encoder from scratch.
12. Không dùng Test-derived regime threshold để weight training samples.
13. Không thêm future-derived rolling/lag feature.
14. Không compare models trên different target populations mà không common-population audit.

---

# 24. Expected mechanism of improvement

Nếu plan hoạt động đúng, improvement nên đến theo chain sau:

```text
Clean features
    ↓
less spurious validation fitting
    ↓
Persistence-aware residual prediction
    ↓
better flat-region MAE + less one-step lag
    ↓
Delta/change objective
    ↓
stronger reaction to upward/rapid changes
    ↓
Scheduler + richer optimization
    ↓
better convergence / less local optimum
    ↓
Moderate capacity expansion
    ↓
more representational power only where justified
    ↓
Multi-fold + multi-seed selection
    ↓
more reliable temporal generalisation
```

---

# 25. Expected deliverables của V2

Khi hoàn thành, cần có ít nhất:

```text
Plan_improve_model.md
configs/improvement_v2/model_improvement_contract.json
artifacts/improvement_v2/v1_baseline_snapshot.json
artifacts/improvement_v2/feature_ablation/
artifacts/improvement_v2/training_optimization/
artifacts/improvement_v2/residual_models/
artifacts/improvement_v2/rolling_origin/
artifacts/improvement_v2/seed_robustness/
artifacts/improvement_v2/model_lock/
artifacts/improvement_v2/evaluation/
```

Bảng final V2 development comparison phải bao gồm:

- pooled RMSE;
- pooled MAE;
- R²;
- worst fold;
- fold SD;
- rapid-change RMSE;
- high-target RMSE;
- prediction range/std ratio;
- parameter count;
- runtime;
- seed stability.

---

# 26. Proposed execution sequence — thứ tự thực hiện cụ thể

## Step 1

Freeze V1 and repair Phase43 provenance.

## Step 2

Create V2 contract, artifact namespace and tests.

## Step 3

Run `FS1_TF1` vs `FS2_TF1` robust feature ablation.

## Step 4

Implement `ResidualTransformerRegressor`.

## Step 5

Implement gated residual + MLP head.

## Step 6

Add warm-up/cosine and ReduceLROnPlateau support.

## Step 7

Tune LR/scheduler/batch on residual architecture.

## Step 8

Add target delta/rolling features.

## Step 9

Add lag144 and test on common population.

## Step 10

Add level+delta auxiliary loss.

## Step 11

Run structured multi-factor search.

## Step 12

Expand d_model/layers/FFN and add Pre-LN if necessary.

## Step 13

Run 3-seed full rolling-origin finalists.

## Step 14

Evaluate seed ensemble and fixed persistence-neural blends on development folds.

## Step 15

If V2 Transformer still plateaus, implement TCN first; then consider iTransformer/TSMixer/PatchTST.

## Step 16

Lock V2 candidate before any benchmark evaluation.

## Step 17

Use old Test only with correct `POST_HOC_V2_BENCHMARK` label unless truly new unseen evaluation data is obtained.

---

# 27. Most likely high-impact combination

Based on current evidence, the combination tôi ưu tiên nhất để thử đầu tiên là:

```text
Feature set:
    FS1_TF1 (remove rv1/rv2)
    + Appliances delta features
    + rolling target statistics
    + optional lag144

Architecture:
    Transformer Encoder
    d_model 64 initially
    layers 2 initially
    heads 4
    FFN 256
    LAST_STEP
    MLP residual head
    learned persistence gate

Prediction:
    y_hat = y_t + g * delta_hat

Loss:
    MSE(level)
    + 0.25 * SmoothL1(delta)

Optimizer:
    AdamW

Training:
    batch 32
    gradient clip 1.0
    warmup 5–10%
    cosine decay
    max LR search 3e-4 / 5e-4 / 1e-3
    early stopping after scheduler has opportunity to decay

Selection:
    rolling-origin pooled RMSE
    MAE guardrail
    worst-fold guardrail
    multi-seed confirmation
```

Lý do: cấu hình này trực tiếp attack ba failure modes có evidence mạnh nhất:

1. persistence-like target dynamics;
2. under-reaction to changes/spikes;
3. single-seed / OFAT model-selection instability.

---

# 28. Confidence ranking của các hypotheses

| Hypothesis | Confidence | Evidence |
|---|---|---|
| Persistence-aware residual/gate sẽ hữu ích | **Very High** | Persistence MAE + complementary regime behavior + diagnostic blend |
| Current predictions are too smooth | **Very High** | std/range/change compression across all 3 seeds |
| Rapid/high regimes are structural failure | **Very High** | regime RMSE + 17/20 cross-seed worst overlap |
| `rv1/rv2` should not be primary predictive features | **Very High** | UCI negative-control purpose + exact duplicate + near-zero correlations |
| Scheduler/LR refactor can help | **High** | no scheduler/warmup + LR tradeoff + early stop behavior |
| Current capacity search is too narrow | **High** | multiple upper-bound winners |
| Delta/rolling features can help | **High** | strong autoregression + low direction agreement |
| Lag144 can help | **Medium-High** | stable ~0.2–0.25 daily ACF |
| Pre-LN can help deeper model | **Medium** | relevant mainly after layers >2 |
| Weighted spike loss will help | **Medium** | targeted failure but may hurt MAE |
| Direction auxiliary task will help | **Medium** | low current direction agreement, but extra objective risk |
| RevIN redesign will help | **Low-Medium** | current RevIN clearly failed |
| Much larger vanilla Transformer alone will solve problem | **Low-Medium** | capacity trend positive but formulation issue dominates |

---

# 29. External research anchors used to motivate optional V2 directions

The following research should be treated as **design inspiration**, not evidence that a method will automatically improve this dataset:

1. **Vaswani et al. (2017), Attention Is All You Need.**  
   Transformer training introduced a learning-rate schedule with warm-up, reinforcing that optimizer schedule is an architectural training consideration rather than a cosmetic parameter.

2. **Xiong et al. (2020), On Layer Normalization in the Transformer Architecture.**  
   Analyses differences between Post-LN and Pre-LN training stability; relevant if V2 increases Transformer depth.

3. **Zeng et al. (2023), Are Transformers Effective for Time Series Forecasting?**  
   Demonstrates the importance of strong simple linear forecasting baselines and cautions against assuming complex Transformers are automatically superior.

4. **Bai, Kolter & Koltun (2018), An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.**  
   Supports TCN as a serious sequence-model baseline.

5. **Nie et al. (2023), A Time Series is Worth 64 Words: Long-term Forecasting with Transformers / PatchTST.**  
   Patching and channel-independent processing are potential alternatives if longer contexts become important.

6. **Liu et al. (2024), iTransformer: Inverted Transformers Are Effective for Time Series Forecasting.**  
   Motivates variate-centric tokenisation when conventional timestamp tokens fuse heterogeneous variables too early.

7. **Chen et al. (2023), TSMixer.**  
   Motivates lightweight time/feature mixing as an alternative to attention-heavy models.

8. **UCI Appliances Energy Prediction documentation / Candanedo et al. (2017).**  
   Confirms `rv1`/`rv2` are random variables provided to test/filter non-predictive attributes and provides original dataset context.

---

# 30. Final recommendation

Không nên bắt đầu V2 bằng việc đổi Transformer thành model rất lớn hoặc chạy một hyperparameter grid khổng lồ.

**Recommended order:**

```text
1. Repair scientific lineage
2. Freeze V1
3. Clean random-control features
4. Residual-to-Persistence model
5. Gated residual model
6. MLP head
7. LR scheduler + warm-up
8. Delta / rolling / daily-lag features
9. Level + delta loss
10. Multi-factor rolling-origin search
11. Moderate capacity expansion + Pre-LN
12. Multi-seed confirmation
13. Ensemble
14. TCN / iTransformer / TSMixer only if needed
```

The single most important architectural hypothesis is:

> **Do not force the neural network to replace Persistence everywhere. Let Persistence handle stable periods, and train the deep model to learn when and how much to correct it.**

Điều này phù hợp trực tiếp với evidence hiện tại: Persistence đang tốt ở flat/typical samples, Transformer đang giúp ở difficult dynamics, trong khi V1 hiện bị smooth, lagged và compressed. Một persistence-aware residual/gated V2 là hướng có cơ sở nhất để cố gắng cải thiện đồng thời **MAE, RMSE và R²** mà không đánh đổi một metric để lấy metric khác.
