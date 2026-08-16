# Practice 3 v2.1 – Validation Loss Overfitting & Confidence Audit

**Date:** 2026-08-15  
**Protocol:** practice_3_v2.1  
**Audit type:** READ-ONLY checkpoint analysis  
**Validation split:** locked v2.1, 960 samples (480 POS / 480 NEG)  
**Checkpoints audited:** 6 (3 epochs × 2 runs)  
**Holdout accessed:** FALSE | **Official test accessed:** FALSE | **Registry modified:** FALSE | **p3v21_lr_5e-5 run:** FALSE  

---

## Observed Pattern

Across both runs, the following trajectory is clearly visible:

| Metric | Direction (Ep1 → Ep3) |
|---|---|
| Train Loss | ↓ Drops sharply (0.44 → 0.14) |
| Val Loss | ↑ Increases significantly (+55–84%) |
| Val Accuracy | Peaks at Ep2, drops at Ep3 |
| Val F1 | Peaks at Ep2, drops at Ep3 |
| Mean Confidence | ↑↑ Surges dramatically (0.85 → 0.97) |
| Wrong-prediction confidence | ↑↑ Surges dramatically (0.73 → 0.92) |
| Prediction Entropy | ↓↓ Collapses (0.36 → 0.08) |
| ECE | ↑↑ Worsens severely (0.02 → 0.11) |
| Logit magnitude | ↑↑ Grows 2.5–2.7× |

**Summary:** The model is becoming increasingly overconfident on wrong predictions at the same time that its generalization deteriorates. This is the textbook signature of **confidence overfitting**.

---

## Checkpoint Comparison

### Run: p3v21_lr_2e-5≠

| Epoch | Checkpoint | Train Loss | Val Loss | Val Acc | Val Prec | Val Rec | Val F1 |
|---|---|---|---|---|---|---|---|
| 1 | checkpoint-480 | 0.4405 | 0.3403 | 0.8615 | 0.9141 | 0.7979 | 0.8521 |
| 2 | checkpoint-960 | 0.2500 | 0.3503 | **0.8792** | 0.8939 | 0.8604 | **0.8769** |
| 3 | checkpoint-1440 | 0.1437 | 0.5285 | 0.8635 | 0.9030 | 0.8146 | 0.8565 |

### Run: p3v21_lr_3e-5

| Epoch | Checkpoint | Train Loss | Val Loss | Val Acc | Val Prec | Val Rec | Val F1 |
|---|---|---|---|---|---|---|---|
| 1 | checkpoint-480 | 0.4384 | 0.3266 | 0.8698 | 0.9043 | 0.8271 | 0.8640 |
| 2 | checkpoint-960 | 0.2264 | 0.3978 | **0.8771** | 0.8868 | 0.8646 | **0.8755** |
| 3 | checkpoint-1440 | 0.1193 | 0.5985 | 0.8615 | 0.8952 | 0.8188 | 0.8553 |

**Key observation:** In both runs, accuracy/F1 peak at Epoch 2 then regress at Epoch 3, while train loss continues to drop. Val Loss diverges sharply from Epoch 2 onward. This confirms **accuracy–loss divergence**.

---

## Confidence Analysis

### Run: p3v21_lr_2e-5

| Epoch | Mean Conf | Conf (Correct) | Conf (Wrong) | N Wrong | Wrong ≥ 0.90 | Wrong ≥ 0.95 | Mean Entropy | Mean |Logit| | Max |Logit| |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.8493 | 0.8678 | 0.7346 | 133 | 17 | 5 | 0.3632 | 1.045 | 2.195 |
| 2 | 0.9318 | 0.9440 | 0.8431 | 116 | **61** | **43** | 0.1877 | 1.735 | 2.879 |
| 3 | **0.9711** | 0.9803 | **0.9127** | 131 | **92** | **81** | **0.0819** | **2.569** | **3.620** |

### Run: p3v21_lr_3e-5

| Epoch | Mean Conf | Conf (Correct) | Conf (Wrong) | N Wrong | Wrong ≥ 0.90 | Wrong ≥ 0.95 | Mean Entropy | Mean |Logit| | Max |Logit| |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.8498 | 0.8677 | 0.7297 | 125 | 13 | 4 | 0.3610 | 1.055 | 2.189 |
| 2 | 0.9492 | 0.9593 | 0.8773 | 118 | **71** | **59** | 0.1416 | 2.022 | 3.097 |
| 3 | **0.9734** | 0.9814 | **0.9238** | 133 | **102** | **91** | **0.0698** | **2.816** | **3.797** |

### Critical Finding: Wrong-prediction confidence explosion

- **LR 2e-5:** WrongConf: 0.73 → 0.84 → **0.91**
  - Wrong predictions with conf ≥ 0.90: 17 → 61 → **92** (5.4× increase Ep1→Ep3)
  - Wrong predictions with conf ≥ 0.95: 5 → 43 → **81** (16.2× increase)

- **LR 3e-5:** WrongConf: 0.73 → 0.88 → **0.92**
  - Wrong predictions with conf ≥ 0.90: 13 → 71 → **102** (7.8× increase)
  - Wrong predictions with conf ≥ 0.95: 4 → 59 → **91** (22.8× increase)

At Epoch 3, the majority of wrong predictions are made with confidence ≥ 0.90. The model is **strongly overconfident** and getting worse in a high-confidence manner.

---

## Calibration Analysis

### Run: p3v21_lr_2e-5

| Epoch | NLL | Brier Score | ECE |
|---|---|---|---|
| 1 | 0.3403 | 0.2088 | **0.0182** |
| 2 | 0.3503 | 0.1973 | 0.0597 |
| 3 | 0.5285 | 0.2390 | **0.1076** |

### Run: p3v21_lr_3e-5

| Epoch | NLL | Brier Score | ECE |
|---|---|---|---|
| 1 | 0.3266 | 0.1988 | **0.0305** |
| 2 | 0.3978 | 0.2096 | 0.0747 |
| 3 | 0.5985 | 0.2490 | **0.1152** |

**Observations:**

- **NLL** = Val Loss (cross-entropy per sample) — increases consistently, confirming loss growth is real.
- **Brier Score** dips slightly at Ep2 (model is more accurate), then rises sharply at Ep3 as high-confidence wrong predictions dominate.
- **ECE** at Ep1 is near-perfect (0.018–0.031). By Ep3 it reaches 0.108–0.115 — the model's expressed confidence systematically exceeds its actual accuracy. **Calibration worsening is confirmed and severe.**

---

## Overfitting Analysis

### Train Loss vs Val Loss gap

| Run | Ep1 gap (Val−Train) | Ep2 gap | Ep3 gap |
|---|---|---|---|
| LR 2e-5 | −0.100 (val < train) | +0.100 | **+0.385** |
| LR 3e-5 | −0.112 (val < train) | +0.171 | **+0.479** |

- Ep1: validation loss is *below* train loss — model is still learning broadly.
- Ep2: gap opens, model slightly overfit.
- Ep3: gap is massive. **Classical overfitting confirmed.**

### Accuracy vs Val Loss divergence

- Accuracy improves Ep1→Ep2 (+1.8pp) while Val Loss also increases (+3%). The hypothesis that accuracy can improve while loss worsens is confirmed at Ep1→Ep2.
- At Ep3, both accuracy and Val Loss worsen together. The primary mechanism is **confidence inflation on wrong predictions**, not accuracy alone.

---

## Training Configuration Analysis

| Parameter | Value | Risk |
|---|---|---|
| Model | DistilBERT full fine-tuning (67M params) | **HIGH** — 67M params / 7,676 samples = ~114 samples/param inverse ratio |
| warmup_steps | 0 | **MEDIUM** — maximum LR from step 1 risks early over-adaptation |
| weight_decay | 0.01 | **LOW-MEDIUM** — provides L2 but insufficient to prevent logit growth |
| max_grad_norm | 1.0 | **LOW** — prevents explosions but does not limit confidence growth |
| label_smoothing | not configured (0.0) | **HIGH** — hard targets incentivize unbounded logit magnitude growth |
| scheduler | linear (no warmup) | **MEDIUM** — LR highest when model is most uncertain |
| max_epochs | 10 | **MEDIUM** — safe with patience=2 but aggressive if patience is increased |
| early_stopping_patience | 2 | **EFFECTIVE** — stopped correctly at Ep3 |

---

## Root Cause / Most Likely Explanation

**Primary root cause: Confidence overfitting driven by label smoothing absence + full fine-tuning capacity.**

**Mechanism:**

1. Epoch 1: Model learns discriminative features, moderate confidence (0.85). Val Loss slightly below train loss — healthy learning.

2. Epoch 2: Model genuinely improves accuracy (+1.8pp). However, AdamW pushes logit magnitudes higher (1.05 → 1.74) because cross-entropy with hard targets has no floor on logit growth. Confidence surges (0.85 → 0.93). ECE begins worsening.

3. Epoch 3: Model memorizes training-specific features. Train loss continues to drop (0.14) but on validation data, memorized patterns apply with high confidence where they do not generalize. Wrong predictions at Ep3 carry confidence ≥ 0.90 in 70% of cases. Cross-entropy penalizes high-confidence wrong predictions severely, causing the Val Loss spike (+55/+83%).

**Proof of mechanism:**

- Number of wrong predictions barely changes (133 → 116 → 131 for LR 2e-5) while Val Loss increases 55%.
- The gap is explained only by confidence inflation: wrong predictions at Ep3 have confidence 0.91 vs 0.73 at Ep1. Cross-entropy loss of a wrong prediction with confidence 0.91 is −log(0.09) = 2.41. At confidence 0.73 it is −log(0.27) = 1.31. That alone accounts for a near-doubling of the per-wrong-sample loss contribution.

**This is confidence overfitting, not classical prediction overfitting.**

---

## Recommended v2.2 Changes

> These are proposals only. v2.2 has NOT been implemented. Nothing has been modified.

| Parameter | v2.1 | Proposed v2.2 | Rationale |
|---|---|---|---|
| **label_smoothing** | 0.0 | **0.1** | Highest priority. Creates soft targets (0.9/0.1), bounding logit growth, directly addressing root cause |
| **warmup_steps** | 0 | **~480 steps (~10% of total)** | Stabilizes model before large updates; protects pre-trained representations |
| **fine-tuning strategy** | Full (all 67M) | **Staged** (freeze bottom 3 layers Ep1–2, unfreeze later) | Reduces effective capacity during early training, prevents early memorization |
| **weight_decay** | 0.01 | **0.05** | Stronger L2 regularization to suppress weight/logit magnitude growth |
| **classifier dropout** | 0.1 (default) | **0.2** | Additional regularization in the task-specific head |
| **max_epochs** | 10 | **6** | Conservative; sufficient given Ep1–3 trajectory |
| **early_stopping_patience** | 2 | **2 (keep)** | Working correctly — no change needed |
| **learning_rates** | 2e-5, 3e-5 | Keep 2e-5, consider adding 1e-5 | LR 2e-5 showed better calibration; 1e-5 with warmup may be more stable |

**Priority order:** Label smoothing > Warmup > Staged fine-tuning > Weight decay > Dropout

---

## Safety Verification

| Check | Status |
|---|---|
| p3v21_lr_5e-5 executed | FALSE ✅ |
| Holdout accessed | FALSE ✅ |
| Official Test accessed | FALSE ✅ |
| Registry modified | FALSE ✅ |
| practice_3_v2.1 modified | FALSE ✅ |
| Checkpoints deleted or overwritten | FALSE ✅ |
| Winner selected | FALSE ✅ |
| New official training run | FALSE ✅ |
| Validation split used | Locked v2.1 (960 samples, fingerprint 4d22ccf1...) ✅ |
| holdout_content_in_manifest | FALSE (verified at runtime) ✅ |
| official_test_loaded | FALSE (verified at runtime) ✅ |

