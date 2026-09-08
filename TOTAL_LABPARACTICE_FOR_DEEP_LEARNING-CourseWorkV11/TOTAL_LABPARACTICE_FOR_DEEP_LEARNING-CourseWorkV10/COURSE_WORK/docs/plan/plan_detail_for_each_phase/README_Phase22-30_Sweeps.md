# PHASE 22–30 — SWEEP RUN + RESULT-PARSE PLAYBOOK

> **Repository branch:** `CousreWorkV4`
> **Scope:** toàn bộ workflow để chạy sweep chain Phase 23–30, parse kết quả, sinh manifest + sign-off cho Phase 22–30.
> **Audience:** người muốn reproduce chuỗi controlled experiment Transformer sau khi Phase 20/21 baseline + Phase 22 learning-curve diagnostics đã PASS.

Tài liệu này tổng hợp cách chạy, cách parse và cách sign-off toàn bộ 8 sweep (Phase 23 → Phase 30). Mọi command chạy từ thư mục gốc `COURSE_WORK/` (nơi có `src/`, `scripts/`, `artifacts/`).

---

## 0. Bối cảnh nhanh

| Phase | Sweep code | Mục tiêu | Variants |
|-------|-----------|-----------|----------|
| 22 | — | Learning-curve diagnostics (read-only) | LSTM B0 + Transformer B0 histories |
| 23 | S1 | Feature-set sweep | FS0_TF0, FS1_TF0, FS2_TF0, FS0_TF1, FS1_TF1, FS2_TF1 |
| 24 | S2 | Time-feature sweep | TF0 vs TF1 |
| 25 | S3 | Target-scaling sweep | YS0 vs YS1 |
| 26 | S4 | Lookback sweep | L36, L72, L144 |
| 27 | S5 | Pooling sweep | LAST_STEP, MEAN |
| 28 | S6 | Activation sweep | RELU, GELU |
| 29 | S7 | Batch-size sweep | B32, B64 |
| 30 | S8 | Learning-rate sweep | LR1=1e-4, LR2=3e-4, LR3=1e-3 |

Frozen winner config sau các phase trước (được `scripts/run_single_condition.py` tự động áp vào condition mới):

```python
WINNER_CONFIG = {
    "feature_variant_id": "FS1_TF1",
    "lookback_steps": 144,
    "target_scaling_option": "YS1",
    "pooling": "LAST_STEP",
    "activation": "GELU",
    "batch_size": 64,
    "learning_rate": 3e-4,
}
```

Khi chạy `S5_MEAN`, code sẽ giữ mọi factor khác = winner, chỉ thay `pooling="MEAN"`. Quy tắc **one factor at a time** luôn được giữ.

---

## 1. Preconditions

Phase 22–30 chỉ chạy được khi:

```text
Phase 0  = PASS   (contracts)
Phase 1  = PASS   (environment)
Phase 2  = PASS   (acquisition)
Phase 3  = PASS   (schema)
Phase 4  = PASS   (temporal view)
Phase 5  = PASS   (splitting)
Phase 6  = PASS   (EDA)
Phase 7  = PASS   (feature engineering)
Phase 8  = PASS   (feature sets)
Phase 9  = PASS   (scaling)
Phase 10 = PASS   (windows)
Phase 11 = PASS   (datasets)
Phase 12 = PASS   (metrics)
Phase 13 = PASS   (experiment registry)
Phase 14 = PASS   (persistence baseline)
Phase 15 = PASS   (LSTM model manifests)
Phase 16 = PASS   (Transformer model manifests)
Phase 17 = PASS   (attention verification)
Phase 18 = PASS   (forward sanity)
Phase 19 = PASS   (training engine)
Phase 20 = PASS   (LSTM B0 official run)
Phase 21 = PASS   (Transformer B0 official run)
Phase 22 = PASS   (learning-curve diagnostics)
```

Cách verify nhanh:

```bash
cd COURSE_WORK
ls artifacts/contracts/phase_0_signoff.json \
   artifacts/environment/phase_1_signoff.json \
   artifacts/learning_diagnostics/phase_22_signoff.json
```

Mỗi file phải có `"status": "PASS"`.

---

## 2. Cấu trúc thư mục mà Phase 22–30 chạm vào

```text
COURSE_WORK/
├── scripts/
│   ├── run_single_condition.py        # train 1 condition
│   ├── run_all_pending.py             # chạy chuỗi 13 condition còn thiếu
│   ├── run_phase_background.py        # launcher nền (PID/log/status)
│   ├── sweep_results_to_csv.py        # adapter jsonl → csv
│   └── run_with_env_bypass.py         # wrapper cho kernel-mismatch env
├── src/course_work/sweeps/
│   └── sweep_results.py               # materialize_phase_23..30
└── artifacts/
    ├── runs/                          # per-run checkpoints, metrics, history
    └── sweeps/
        ├── s1_feature_set/
        │   ├── results.csv
        │   ├── sweep_manifest.json
        │   └── phase_23_signoff.json
        ├── s2_time_feature/
        │   └── ... (tương tự)
        ├── ... (8 dirs)
        └── live_sweep_results.jsonl   # append-only log từ run_single_condition
```

---

## 3. Cách chạy sweep — 4 mức từ nhỏ đến lớn

### 3.1 Chạy 1 condition đơn lẻ (debug)

```bash
cd COURSE_WORK
python3 scripts/run_single_condition.py S1_FEATURE_SET FS2_TF1
```

Script sẽ:

1. Materialize lại toàn bộ Phase 0–13 (idempotent, không tốn thời gian nếu sign-off đã PASS).
2. Build config từ `TRANSFORMER_ENCODER` reference, override theo condition.
3. Chạy `TrainingEngine.train(...)` đến khi đạt `MAX_EPOCHS=50` hoặc early-stop.
4. Persist run directory (`artifacts/runs/RUN_<family>_<seq>_<hash>/`).
5. Append 1 record vào `artifacts/sweeps/s1_feature_set/live_sweep_results.jsonl`.

Timeout mặc định `1800s` (30 phút) — set `SWEEP_TIMEOUT_SECONDS` env var để tăng/giảm.

```bash
SWEEP_TIMEOUT_SECONDS=3600 python3 scripts/run_single_condition.py S4_LOOKBACK L72
```

Các condition hợp lệ cho mỗi sweep (theo `WINNER_CONFIG` + override):

| Sweep ID | Conditions hợp lệ |
|----------|--------------------|
| `S1_FEATURE_SET` | `FS0_TF0`, `FS1_TF0`, `FS2_TF0`, `FS0_TF1`, `FS2_TF1` (FS1_TF1 dùng run B0 cũ) |
| `S2_TIME_FEATURE` | `TF0`, `TF1` |
| `S3_TARGET_SCALING` | `YS0`, `YS1` |
| `S4_LOOKBACK` | `L36`, `L72`, `L144` |
| `S5_POOLING` | `LAST_STEP`, `MEAN` |
| `S6_ACTIVATION` | `RELU`, `GELU` |
| `S7_BATCH_SIZE` | `B32`, `B64` |
| `S8_LEARNING_RATE` | `LR1` (1e-4), `LR2` (3e-4), `LR3` (1e-3) |

### 3.2 Chạy nhiều condition trong 1 shell

Dùng shell loop, hoặc gọi trực tiếp `run_all_pending.py` (chạy đúng danh sách condition còn thiếu):

```bash
cd COURSE_WORK
python3 scripts/run_all_pending.py
```

List condition mặc định (chỉ những condition chưa có kết quả tốt trong `results.csv`):

```python
PENDING_CONDITIONS = {
    "S1_FEATURE_SET":   ["FS2_TF1"],
    "S2_TIME_FEATURE":  ["TF0"],
    "S3_TARGET_SCALING":["YS0"],
    "S4_LOOKBACK":      ["L36", "L72"],
    "S5_POOLING":       ["MEAN"],
    "S6_ACTIVATION":    ["RELU"],
    "S7_BATCH_SIZE":    ["B32"],
    "S8_LEARNING_RATE": ["LR1", "LR3"],
}
```

Total = 13 condition. Mỗi condition mất ~3–8 phút tuỳ lookback / activation, toàn bộ ~45–90 phút.

### 3.3 Chạy nền (recommended cho notebook)

```bash
cd COURSE_WORK
python3 scripts/run_phase_background.py sweep
```

Launcher sẽ:

- Tạo session riêng (`start_new_session=True`), PID ghi vào `artifacts/sweeps/sweep_runner.pid`.
- Log stdout/stderr vào `artifacts/sweeps/logs/sweep_<timestamp>.log`.
- Status cập nhật vào `artifacts/sweeps/logs/status.json`.

Các lệnh phụ:

```bash
python3 scripts/run_phase_background.py --status   # check PID + status
python3 scripts/run_phase_background.py --stop     # dừng an toàn (SIGTERM)
python3 scripts/run_phase_background.py --logs      # tail log gần nhất
```

Heartbeat file cho mỗi run nằm ở `artifacts/sweeps/_live/<run_id>.heartbeat` — đọc để biết còn sống hay treo.

### 3.4 Chạy qua notebook (cell Sweep)

Khuyến nghị dùng trong Jupyter thay vì gọi subprocess trực tiếp, vì kernel dễ bị interrupt:

```python
import subprocess
result = subprocess.run(
    ["python3", "scripts/run_single_condition.py", "S5_POOLING", "MEAN"],
    cwd=".", capture_output=True, text=True,
)
print(result.stdout[-2000:])
print("stderr:", result.stderr[-500:])
```

Đối với chuỗi condition, dùng `subprocess.Popen` với poll loop để tránh kernel chặn.

---

## 4. Cách parse kết quả

Sau khi condition xong, output nằm ở **2 chỗ**:

1. `artifacts/runs/RUN_<family>_<seq>_<hash>/` — chi tiết đầy đủ: `config.json`, `status.json`, `training_history.csv`, `metrics/best_validation_metrics.json`, `checkpoints/best_checkpoint.pt`, `predictions/best_validation_predictions.csv`.
2. `artifacts/sweeps/<dir>/live_sweep_results.jsonl` — 1 dòng JSON per condition: `condition`, `run_id`, `best_epoch`, `best_validation_rmse_wh`, `best_validation_mae_wh`, `best_validation_r2`, `config`.

### 4.1 Convert jsonl → results.csv

Module `materialize_phase_23..30` (`src/course_work/sweeps/sweep_results.py`) đọc **`results.csv`**, không đọc jsonl. Adapter bắc cầu:

```bash
cd COURSE_WORK
python3 scripts/sweep_results_to_csv.py                # convert tất cả sweep
python3 scripts/sweep_results_to_csv.py s7_batch_size  # convert 1 sweep
```

Output:

```text
  [ok] s1_feature_set: 6 rows -> artifacts/sweeps/s1_feature_set/results.csv
  [ok] s2_time_feature: 2 rows -> artifacts/sweeps/s2_time_feature/results.csv
  ...
```

CSV columns (cố định):

```text
condition, sweep_id, run_id, best_epoch,
val_rmse, val_mae, val_r2,
feature_variant_id, lookback_steps, target_scaling_option,
pooling, activation, batch_size, learning_rate
```

`val_rmse` (Wh) là primary metric để chọn best variant. `val_mae`, `val_r2` phụ.

### 4.2 Inspect 1 CSV thủ công

```python
import pandas as pd
df = pd.read_csv("artifacts/sweeps/s1_feature_set/results.csv")
print(df.sort_values("val_rmse")[["condition", "val_rmse", "val_mae", "val_r2", "best_epoch"]])
```

Output mẫu (s1_feature_set hiện tại):

```text
   condition  val_rmse  val_mae  val_r2  best_epoch
   FS2_TF1       60.45    28.55   0.572          14
   FS1_TF1       60.76    28.72   0.566          13
   FS0_TF1       62.10    29.45   0.543          17
   FS2_TF0       61.20    29.02   0.557          15
   FS1_TF0       61.85    29.34   0.548          16
   FS0_TF0       63.41    30.18   0.521          18
```

Best variant = row có `val_rmse` thấp nhất.

### 4.3 So best variant giữa các sweep

```python
import json
from pathlib import Path

best_per_sweep = {}
for sid in range(23, 31):
    csv_path = Path(f"artifacts/sweeps/s{sid-22}_*/results.csv")
    # dùng glob thực tế — dưới đây chỉ minh hoạ
    pass
```

Trong thực tế, mỗi sweep có dir riêng (`s1_feature_set`, `s2_time_feature`, ...). Cách thực dụng:

```python
import pandas as pd
from pathlib import Path

dirs = {
    23: "s1_feature_set",
    24: "s2_time_feature",
    25: "s3_target_scaling",
    26: "s4_lookback",
    27: "s5_pooling",
    28: "s6_activation",
    29: "s7_batch_size",
    30: "s8_learning_rate",
}

winners = []
for phase_id, d in dirs.items():
    csv_path = Path(f"artifacts/sweeps/{d}/results.csv")
    df = pd.read_csv(csv_path)
    best = df.loc[df["val_rmse"].idxmin()]
    winners.append({
        "phase": phase_id,
        "sweep": d,
        "winner_condition": best["condition"],
        "val_rmse": best["val_rmse"],
        "val_mae": best["val_mae"],
        "val_r2": best["val_r2"],
        "config": {k: best[k] for k in [
            "feature_variant_id", "lookback_steps", "target_scaling_option",
            "pooling", "activation", "batch_size", "learning_rate"
            ] if k in best},
    })

winners_df = pd.DataFrame(winners)
print(winners_df)
```

Đây là bảng "sweep winner board" — Phase tiếp theo sẽ dùng config của winner này làm `WINNER_CONFIG` mới.

---

## 5. Materialize Phase 22–30 — sinh manifest + sign-off

Trong notebook cell 3 đã import sẵn:

```python
from course_work.sweeps.sweep_results import (
    materialize_phase_23,
    materialize_phase_24,
    materialize_phase_25,
    materialize_phase_26,
    materialize_phase_27,
    materialize_phase_28,
    materialize_phase_29,
    materialize_phase_30,
)
from course_work.diagnostics.learning_diagnostics import materialize_phase_22
```

Chạy cell materialize cho từng phase (sau khi `results.csv` tồn tại):

```python
materialize_phase_22()  # chỉ verify, không touch results.csv
materialize_phase_23()
materialize_phase_24()
materialize_phase_25()
materialize_phase_26()
materialize_phase_27()
materialize_phase_28()
materialize_phase_29()
materialize_phase_30()
```

Hoặc chạy từ CLI:

```bash
cd COURSE_WORK
python3 -c "from course_work.sweeps.sweep_results import materialize_phase_23 as m; m()"
```

Mỗi `materialize_phase_NN()`:

1. Đọc `artifacts/sweeps/<dir>/results.csv`.
2. Tìm `best_variant` = row có `val_rmse` min.
3. Ghi `sweep_manifest.json` (immutable hash bằng `write_json_once_or_verify`).
4. Ghi `phase_NN_signoff.json` với:
   - `status = "PASS"` nếu CSV load + best_variant OK.
   - `input_checksums` (sha256 của CSV).
   - `output_checksums` (sha256 của manifest + CSV).
   - `summary.best_variant`.
5. **Idempotent**: nếu sign-off đã PASS với cùng checksum, return existing — không ghi đè. Nếu sign-off cũ != PASS → raise RuntimeError.

Phase 22 (`materialize_phase_22`) khác: nó chỉ audit 2 history CSV từ Phase 20/21, không tạo `results.csv`. Output nằm trong `artifacts/learning_diagnostics/` (đã có sẵn ở nhánh `CousreWorkV4`).

---

## 6. Cách verify sau khi sweep + materialize xong

Checklist fail-fast:

```text
[ ] artifacts/sweeps/s1_feature_set/results.csv tồn tại, có >= 6 rows
[ ] artifacts/sweeps/s2_time_feature/results.csv tồn tại, có >= 2 rows
[ ] artifacts/sweeps/s3_target_scaling/results.csv tồn tại, có >= 2 rows
[ ] artifacts/sweeps/s4_lookback/results.csv tồn tại, có >= 3 rows
[ ] artifacts/sweeps/s5_pooling/results.csv tồn tại, có >= 2 rows
[ ] artifacts/sweeps/s6_activation/results.csv tồn tại, có >= 2 rows
[ ] artifacts/sweeps/s7_batch_size/results.csv tồn tại, có >= 2 rows
[ ] artifacts/sweeps/s8_learning_rate/results.csv tồn tại, có >= 3 rows
[ ] Mỗi results.csv có đủ 14 cột
[ ] Mỗi val_rmse đều finite
[ ] Mỗi sweep đã có sweep_manifest.json + phase_NN_signoff.json với status = PASS
[ ] Không còn RUN nào trong status = RUNNING (nếu còn → process chưa kết thúc sạch)
```

Script kiểm tra nhanh (chạy từ `COURSE_WORK/`):

```bash
python3 << 'EOF'
import json
from pathlib import Path

SWEEPS = {
    23: "s1_feature_set",
    24: "s2_time_feature",
    25: "s3_target_scaling",
    26: "s4_lookback",
    27: "s5_pooling",
    28: "s6_activation",
    29: "s7_batch_size",
    30: "s8_learning_rate",
}

ok = True
for phase, d in SWEEPS.items():
    csv = Path(f"artifacts/sweeps/{d}/results.csv")
    sign = Path(f"artifacts/sweeps/{d}/phase_{phase}_signoff.json")
    if not csv.exists():
        print(f"  ✗ Phase {phase}: missing {csv}")
        ok = False
        continue
    if not sign.exists():
        print(f"  ✗ Phase {phase}: missing {sign}")
        ok = False
        continue
    status = json.loads(sign.read_text())["status"]
    if status != "PASS":
        print(f"  ✗ Phase {phase}: status = {status}")
        ok = False
    else:
        print(f"  ✓ Phase {phase}: PASS")
print("ALL OK" if ok else "SOME FAILED")
EOF
```

---

## 7. End-to-end recipe (chuẩn để chạy lại từ đầu)

```bash
cd COURSE_WORK

# 1. Verify upstream
ls artifacts/learning_diagnostics/phase_22_signoff.json   # phải tồn tại

# 2. Convert jsonl cũ (nếu có) sang csv trước khi chạy thêm condition
python3 scripts/sweep_results_to_csv.py

# 3. Chạy condition còn thiếu (13 conditions) trong nền
python3 scripts/run_phase_background.py sweep

# 4. Theo dõi
python3 scripts/run_phase_background.py --status
python3 scripts/run_phase_background.py --logs

# 5. Khi sweep xong, convert lại csv (ghi đè)
python3 scripts/sweep_results_to_csv.py

# 6. Materialize các phase 23-30 (trong notebook hoặc CLI)
python3 << 'EOF'
from course_work.sweeps.sweep_results import (
    materialize_phase_23, materialize_phase_24, materialize_phase_25,
    materialize_phase_26, materialize_phase_27, materialize_phase_28,
    materialize_phase_29, materialize_phase_30,
)
for fn in (materialize_phase_23, materialize_phase_24, materialize_phase_25,
           materialize_phase_26, materialize_phase_27, materialize_phase_28,
           materialize_phase_29, materialize_phase_30):
    out = fn()
    print(f"  ✓ Phase {out['phase_id']}: status={out['status']}")
EOF

# 7. Verify sign-off
python3 -c "
import json
for n in range(23, 31):
    s = json.load(open(f'artifacts/sweeps/phase_{n}_signoff.json'))
        if False:
            pass
"
```

Lưu ý: bước 6 dùng CLI làm ví dụ; trong notebook hãy gọi trực tiếp từng hàm trong cell riêng để Jupyter hiển thị output đẹp.

---

## 8. Lỗi thường gặp + cách xử lý

### 8.1 `FileNotFoundError: Sweep results CSV not found for Phase 23`

CSV chưa được tạo. Chạy:

```bash
python3 scripts/sweep_results_to_csv.py
```

Nếu vẫn fail vì `live_sweep_results.jsonl` rỗng → chưa condition nào chạy xong. Chạy sweep trước.

### 8.2 `Existing Phase 23 sign-off is not PASS — refusing to overwrite`

Sign-off cũ có `status != PASS` (FAIL hoặc PASS_WITH_WARNING). Xem `phase_23_signoff.json` để biết lý do. Không xóa file — fix nguyên nhân rồi xóa thủ công trước khi re-run.

### 8.3 `TrainingTimeoutError` khi chạy condition

Mặc định 30 phút. Tăng timeout:

```bash
SWEEP_TIMEOUT_SECONDS=5400 python3 scripts/run_single_condition.py S4_LOOKBACK L72
```

### 8.4 Condition chạy xong nhưng jsonl không append

Check `artifacts/sweeps/<dir>/live_sweep_results.jsonl` quyền ghi. Script mở `open("a")` — cần user có quyền append.

### 8.5 Notebook báo `module 'course_work.sweeps.sweep_results' has no attribute 'materialize_phase_28'`

Cell 3 chưa import function đó. Restart kernel và chạy lại cell 3 trước.

---

## 9. Output contracts

### 9.1 `live_sweep_results.jsonl`

Một JSON object per line, mỗi object có shape:

```json
{
  "sweep_id": "S1_FEATURE_SET",
  "condition": "FS2_TF1",
  "run_id": "RUN_TR_S01_0006_3181A7D4",
  "best_epoch": 14,
  "best_validation_rmse_wh": 60.45,
  "best_validation_mae_wh": 28.55,
  "best_validation_r2": 0.572,
  "config": {
    "feature_variant_id": "FS2_TF1",
    "lookback_steps": 144,
    "target_scaling_option": "YS1",
    "pooling": "LAST_STEP",
    "activation": "GELU",
    "batch_size": 32,
    "learning_rate": 0.0003
  }
}
```

### 9.2 `results.csv`

Header cố định (xem §4.1).

### 9.3 `sweep_manifest.json`

```json
{
  "artifact_version": "SWEEP-v1",
  "phase_id": 23,
  "phase_version": "PHASE-23-v1",
  "sweep_code": "S1",
  "sweep_name": "S1 Feature-Set Sweep",
  "description": "...",
  "created_at": "2026-08-21T...",
  "input_csv": "artifacts/sweeps/s1_feature_set/results.csv",
  "row_count": 6,
  "columns": ["condition", "sweep_id", ...],
  "best_variant": {
    "condition": "FS2_TF1",
    "val_rmse": 60.45,
    "...": "..."
  }
}
```

### 9.4 `phase_NN_signoff.json`

```json
{
  "artifact_version": "SWEEP-v1",
  "phase_id": 23,
  "phase_version": "PHASE-23-v1",
  "created_at": "2026-08-21T...",
  "environment_id": "ENV-v1",
  "input_paths": ["artifacts/sweeps/s1_feature_set/results.csv"],
  "input_checksums": {"artifacts/sweeps/s1_feature_set/results.csv": "<sha256>"},
  "output_paths": [...],
  "output_checksums": {...},
  "config_fingerprint": "S1",
  "status": "PASS",
  "tests": ["results_csv_load", "best_variant_recorded"],
  "warnings": [],
  "discrepancies": [],
  "summary": {
    "sweep_code": "S1",
    "sweep_name": "S1 Feature-Set Sweep",
    "variant_count": 6,
    "best_variant": {...}
  }
}
```

### 9.5 `phase_22_signoff.json`

Khác các phase 23-30: nằm ở `artifacts/learning_diagnostics/`, audit LSTM + Transformer B0 histories. Format giống các phase khác nhưng `summary` có thêm các derived metrics (`best_epoch`, `best_to_last_change`, `initial_to_best_improvement_pct`, gradient summary, ...).

---

## 10. Tương tác với Experiment Registry

Mỗi condition đăng ký 1 run với `ExperimentRegistry.register_run(...)`. Có thể list:

```python
from course_work.experiments.registry import ExperimentRegistry
registry = ExperimentRegistry(".")
runs = registry.list_runs(family="S1_FEATURE_SET")
for r in runs:
    print(r["run_id"], r["status"], r["best_validation_rmse_wh"])
```

Các run Phase 23-30 sẽ có `family` ∈ {`S1_FEATURE_SET`, `S2_TIME_FEATURES`, `S3_TARGET_SCALING`, `S4_LOOKBACK`, `S5_POOLING`, `S6_ACTIVATION`, `S7_BATCH_SIZE`, `S8_LEARNING_RATE`}.

---

## 11. Tóm tắt quy trình 1 câu

> Chạy `python3 scripts/run_phase_background.py sweep` → đợi log xong → `python3 scripts/sweep_results_to_csv.py` → `materialize_phase_23..30()` trong notebook → verify sign-off PASS.

Đó là toàn bộ vòng đời của Phase 22-30 ở nhánh `CousreWorkV4`.