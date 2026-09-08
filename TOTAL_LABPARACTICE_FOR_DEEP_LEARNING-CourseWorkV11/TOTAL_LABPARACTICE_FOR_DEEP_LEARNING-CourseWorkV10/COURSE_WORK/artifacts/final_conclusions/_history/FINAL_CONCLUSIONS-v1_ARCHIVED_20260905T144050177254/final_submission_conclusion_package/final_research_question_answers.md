## RQ1

**Question:** Final Transformer dự báo Appliances energy tốt đến mức nào trên held-out chronological Test?

**Evidence:** FT02 (58)

**Status:** SUPPORTED | **Claim Level:** LEVEL_3_FINAL_HELD_OUT_RESULT

**Conclusion:** On the frozen chronological Held-Out Test segment (FINAL_TEST_POP-v1), the final selected Transformer achieved three-seed mean MAE 28.53 Wh, RMSE 63.83 Wh, and R² 0.506 across seeds 42 / 123 / 2026.

**Caveat:** mean ± sample SD (ddof=1), NOT an ensemble; scoped to held-out chronological segment

---

## RQ2

**Question:** Final Transformer so với Persistence và tuned LSTM như thế nào?

**Evidence:** FT02 (58)

**Status:** MIXED | **Claim Level:** LEVEL_3_FINAL_HELD_OUT_RESULT

**Conclusion:** Persistence baseline MAE 26.74 Wh, RMSE 66.84 Wh, R² 0.459. Tuned LSTM baseline was not evaluated on FINAL_TEST_POP-v1 due to lookback mismatch (L36 vs L72). The final Transformer RMSE is 63.83 Wh vs Persistence 66.84 Wh.

**Caveat:** mixed evidence preserved if applicable; Persistence and LSTM both reported

---

## RQ3

**Question:** Performance có robust theo temporal rolling-origin evaluation không?

**Evidence:** FT03 (58)

**Status:** PARTIALLY_SUPPORTED | **Claim Level:** LEVEL_2_ROBUST_DESCRIPTIVE_PATTERN

**Conclusion:** Rolling-origin pooled RMSE and fold variability are reported in FT03 as DEVELOPMENT_EVIDENCE only.

**Caveat:** rolling-origin is DEVELOPMENT_EVIDENCE only; pooled RMSE primary, mean fold RMSE secondary

---

## RQ4

**Question:** Forecast error tập trung ở những kiểu tình huống/regime nào?

**Evidence:** FT05 (58)

**Status:** SUPPORTED | **Claim Level:** LEVEL_1_OBSERVED_ASSOCIATION

**Conclusion:** FT05 reports error by regime and worst-error concentration with frozen Phase 50/51 definitions; worst-error cases remain valid frozen Test observations.

**Caveat:** residual sign convention: y_true - y_pred; positive = UNDERPREDICTION

---

## RQ5

**Question:** Last-query attention tập trung vào các temporal lags nào?

**Evidence:** FT06 (58)

**Status:** PARTIALLY_SUPPORTED | **Claim Level:** LEVEL_2_ROBUST_DESCRIPTIVE_PATTERN

**Conclusion:** FT06 reports last-query temporal attention summary (recent-1h mass, recent-6h mass, expected lag, normalized entropy) per seed, per layer. Lookback = 72 steps = 12 h, so the most recent 24-h mass is truncated.

**Caveat:** 24-h mass truncated because lookback = 72 = 12 h; last-query is one token

---

## RQ6

**Question:** Các attention heads có học temporal allocation behavior khác nhau không?

**Evidence:** FT07 (58)

**Status:** SUPPORTED | **Claim Level:** LEVEL_2_ROBUST_DESCRIPTIVE_PATTERN

**Conclusion:** FT07 reports within-seed head-comparison descriptive metrics (pairwise JSD, Wasserstein minutes, cosine, top1 TVD) per seed, per layer. Higher values indicate higher pairwise temporal-profile diversity, NOT functional redundancy.

**Caveat:** similarity ≠ functional redundancy; value/output projections can differ

---

## RQ7

**Question:** Attention behavior có co-vary với realized forecast error không?

**Evidence:** FT08 (58)

**Status:** PARTIALLY_SUPPORTED | **Claim Level:** LEVEL_1_OBSERVED_ASSOCIATION

**Conclusion:** FT08 reports Phase-56 error-conditioned Spearman ρ and HIGH_ERROR vs LOW_ERROR differences. HIGH/LOW are Test-relative diagnostic cohorts, NOT deployment regimes.

**Caveat:** associations are descriptive; weak/mixed outcomes preserved

---

## RQ8

**Question:** Các attention findings có ổn định qua ba final seeds không?

**Evidence:** FT09 (58)

**Status:** PARTIALLY_SUPPORTED | **Claim Level:** LEVEL_2_ROBUST_DESCRIPTIVE_PATTERN

**Conclusion:** FT09 reports Phase-57 seed-stability evidence. Layer head-mean (permutation-invariant) is the primary view; permutation-aware matched-head and matching sensitivity are secondary. Same numeric head indices across seeds are NOT assumed semantically equivalent.

**Caveat:** layer head-mean prioritized; matching ambiguity / cycle inconsistency surfaced

---

## RQ9

**Question:** Những điều gì có thể và không thể kết luận từ attention analysis?

**Evidence:** FT10 (58)

**Status:** SUPPORTED | **Claim Level:** LEVEL_0_DESCRIPTIVE_ONLY

**Conclusion:** Attention describes temporal token allocation only — NOT raw-feature importance, NOT causal contribution. Similarity in attention profiles does NOT prove functional redundancy. Matching ambiguity and cycle consistency (Layer 0 = 1/4; Layer 1 = 4/4) are reported when partial.

**Caveat:** interpretive boundary is permanent; partial / ambiguous matching acknowledged

---

## RQ10

**Question:** Những limitation nào giới hạn external/general scientific interpretation?

**Evidence:** FA12 (58)

**Status:** SUPPORTED | **Claim Level:** LEVEL_0_DESCRIPTIVE_ONLY

**Conclusion:** FA12 propagates upstream caveats; limitations are grouped into L1 (dataset), L2 (forecast scope), L3 (model selection), L4 (statistical), L5 (attention interpretability), L6 (deployment/generalization).

**Caveat:** single-house dataset; three seeds; sequential tuning; attention diagnostic only

---

