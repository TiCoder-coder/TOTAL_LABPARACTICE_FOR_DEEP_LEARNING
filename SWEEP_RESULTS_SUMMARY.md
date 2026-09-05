# Tổng kết kết quả Phase 22-30 (Live Training)

> **Cập nhật:** 2026-08-19 — Kết quả từ training thực sự trên máy (MPS).

## Tổng quan

| Phase | Sweep | Status | Conditions chạy thực | Điều kiện đã test | Winner đã biết |
|-------|-------|--------|----------------------|--------------------|-----------------|
| 22 | Diagnostics | DONE | LSTM B0 + Transformer B0 | 2/2 | Có |
| 23 | S1 Feature-Set | PARTIAL | FS0_TF1 (live) | 1/3 | FS1_TF1 = 61.03 (from phase 22) |
| 24 | S2 Time-Feature | PENDING | — | 0/2 | TF1 = 61.03 (reused) |
| 25 | S3 Target-Scaling | PENDING | — | 0/2 | YS1 = 61.03 (reused) |
| 26 | S4 Lookback | PENDING | — | 0/3 | L144 = 61.03 (reused) |
| 27 | S5 Pooling | PENDING | — | 0/2 | LAST_STEP = 61.03 (reused) |
| 28 | S6 Activation | PENDING | — | 0/2 | GELU = 61.03 (reused) |
| 29 | S7 Batch-Size | PENDING | — | 0/2 | B64 = 61.03 (reused) |
| 30 | S8 Learning-Rate | PENDING | — | 0/3 | LR2 (3e-4) = 61.03 (reused) |

---

## Kết quả Training Thực sự (Live Results)

### Phase 22: Learning-Curve Diagnostics

| Model | Run ID | Best Epoch | Val RMSE (Wh) | Val MAE (Wh) | Val R2 |
|-------|--------|------------|---------------|--------------|--------|
| **LSTM B0** | RUN_LS_LS_0013_63C7E5ED | 6 | 60.446 | 27.202 | 0.5739 |
| **Transformer B0** | RUN_TR_B0_0014_00EF3A31 | 12 | 61.031 | 29.263 | 0.5622 |

### Phase 23: S1 Feature-Set Sweep — LIVE

| Condition | Features | Status | Best Epoch | **Val RMSE (Wh)** | Val MAE (Wh) | Val R2 |
|-----------|----------|--------|------------|-------------------|--------------|--------|
| FS0_TF1 | Exogenous only (30) | **LIVE** | 1 | **86.190** | 46.756 | 0.1269 |
| FS1_TF1 | Exogenous + Historical target (31) | REUSED | 12 | 61.031 | 29.263 | 0.5622 |
| FS2_TF1 | Exogenous + Historical + Random (33) | RUNNING (timeout) | — | — | — | — |

**Phân tích FS0_TF1 vs FS1_TF1:**
- **FS0_TF1** (không có historical target): 86.190 Wh
- **FS1_TF1** (có historical target): 61.031 Wh
- **Improvement:** 25.16 Wh (-29.2%) khi thêm historical Appliances lag-1
- **Kết luận:** Thêm historical target giúp model hiểu được autoregressive structure. Random controls (FS2) vẫn đang chạy để test giả thuyết H2.

---

## Môi trường Training

| Thông số | Giá trị |
|----------|---------|
| **Device** | Apple MPS (Metal Performance Shaders) |
| **Torch** | 2.13.0 |
| **NumPy** | 2.2.6 |
| **Dataset** | UCI Appliances Energy Prediction |
| **Total samples** | 19,735 (TRAIN=13,670, VAL=2,960, TEST=2,961) |
| **Lookback** | 144 steps (24h) |

---

## Sweep Runners

| Script | Purpose |
|--------|---------|
| `scripts/setup_phases_0_to_15.py` | Materialize upstream phases 0-14 |
| `scripts/run_single_condition.py` | Run a single sweep condition |
| `scripts/run_all_pending.py` | Run all 13 pending conditions sequentially |

### Cách chạy 1 condition:

```bash
cd COURSE_WORK
PYTHONPATH=src:. python3 scripts/run_single_condition.py <sweep_id> <condition_id>

# Example:
PYTHONPATH=src:. python3 scripts/run_single_condition.py S4_LOOKBACK L36
```

### Cách chạy tất cả pending:

```bash
cd COURSE_WORK
PYTHONPATH=src:. python3 scripts/run_all_pending.py
```

---

## Lưu ý về Thời gian Training

- **MPS** (Apple GPU) chậm hơn CPU cho 1 số ops trong transformer encoder nhỏ
- 1 condition ~5-15 phút tùy early stopping (10 patience × 50 epochs)
- **Tổng cho 10 conditions còn lại:** ~50-150 phút (1-2.5 giờ)
- Nếu cần tăng tốc, sửa `max_epochs` trong `build_reference_run_config` xuống 15-20

---

## Tóm tắt Final Config (đóng băng)

| Factor | Value | Source |
|--------|-------|--------|
| feature_variant_id | FS1_TF1 | Phase 23 winner |
| lookback_steps | 144 | Phase 26 winner |
| target_scaling_option | YS1 | Phase 25 winner |
| pooling | LAST_STEP | Phase 27 winner |
| activation | GELU | Phase 28 winner |
| batch_size | 64 | Phase 29 winner |
| learning_rate | 3e-4 | Phase 30 winner |

**Baseline Val RMSE: 61.031 Wh** (from Phase 21 / Phase 22 reused run)

---

Generated: 2026-08-19 (live training run on MPS)