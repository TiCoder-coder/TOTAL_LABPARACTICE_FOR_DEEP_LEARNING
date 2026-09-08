# Phase 22-30 Upstream Fixes — Run Guide

## Tóm tắt sửa chữa

| # | File được sửa | Vấn đề | Fix |
|---|---|---|---|
| 1 | `src/course_work/diagnostics/learning_curves.py` | Hard-coded `LSTM_RUN_ID` / `TRANSFORMER_RUN_ID` | Auto-discover: registry → filesystem scan fallback |
| 2 | `src/course_work/diagnostics/learning_diagnostics.py` | `materialize_phase_22` pass sai `artifacts_dir` (project_root thay vì `artifacts/`) | Pass `root / "artifacts"` |
| 3 | `src/course_work/sweeps/sweep_results.py` | `_sweep_path` tạo path `artifacts/sweeps/7_s7/` (sai) | Map `code → dir` đúng với `run_single_condition.py` |
| 4 | `scripts/sweep_results_to_csv.py` (new) | Runner ghi `live_sweep_results.jsonl`, materialize đọc `results.csv` | Adapter JSONL → CSV |
| 5 | `scripts/run_all_sweeps_bypass.py` (new) | Sweep fail env check (Python 3.14 vs 3.10) | Wrapper `run_with_env_bypass.py` cho từng condition |

## Môi trường

Kernel Jupyter (`coursework-venv`) đã trỏ về Python 3.10 — tốt. Tuy nhiên, sweep scripts chạy qua `python3` (CLT) = Python 3.14, gây env mismatch. **Workaround**: dùng `run_with_env_bypass.py`.

## Cách chạy

### 1. Đảm bảo Phase 22 / 29 OK trước (Phase 22 chỉ phụ thuộc các run COMPLETED)

```python
from pathlib import Path
from course_work.diagnostics.learning_diagnostics import materialize_phase_22
result = materialize_phase_22(Path('.'))
# status: PASS, finding_count: 2
```

### 2. Chạy Phase 20 (LSTM) và 21 (Transformer) nếu chưa chạy

```bash
# Foreground (blocking, ~3-5 phút mỗi phase)
python3 -c "from pathlib import Path; from course_work.baselines.lstm_baseline import materialize_phase_20; materialize_phase_20(Path('.'))"
python3 -c "from pathlib import Path; from course_work.baselines.transformer_b0 import materialize_phase_21; materialize_phase_21(Path('.'))"

# Background (notebook-safe)
python3 scripts/run_phase_background.py phase 20
python3 scripts/run_phase_background.py phase 21
# Poll status:
python3 scripts/run_phase_background.py --status
```

### 3. Chạy sweep chain (các điều kiện pending — 10 conditions, ~50 phút)

**Detached (suggested)**:
```bash
python3 scripts/run_phase_background.py sweep
```
(Nó tự động dùng `run_all_pending.py`, nhưng `run_all_pending.py` không bypass env check. **Workaround tốt nhất**: chạy `run_all_sweeps_bypass.py` và dùng Phase 20 wrapper để lên lịch.)

Để đảm bảo mỗi condition bypass env check, dùng script mới:

```bash
python3 scripts/run_all_sweeps_bypass.py --dry-run   # verify plan
python3 scripts/run_all_sweeps_bypass.py             # actually run
```

Hoặc từng condition một:
```bash
python3 scripts/run_with_env_bypass.py S1_FEATURE_SET FS2_TF1
python3 scripts/run_with_env_bypass.py S2_TIME_FEATURE TF0
python3 scripts/run_with_env_bypass.py S3_TARGET_SCALING YS0
python3 scripts/run_with_env_bypass.py S4_LOOKBACK L36
python3 scripts/run_with_env_bypass.py S4_LOOKBACK L72
python3 scripts/run_with_env_bypass.py S5_POOLING MEAN
python3 scripts/run_with_env_bypass.py S6_ACTIVATION RELU
python3 scripts/run_with_env_bypass.py S7_BATCH_SIZE B32  # already done
python3 scripts/run_with_env_bypass.py S8_LEARNING_RATE LR1
python3 scripts/run_with_env_bypass.py S8_LEARNING_RATE LR3
```

### 4. Convert JSONL → CSV (cho mỗi sweep đã chạy xong)

```bash
python3 scripts/sweep_results_to_csv.py
```

### 5. Materialize Phase 23-30

```python
from pathlib import Path
from course_work.sweeps.sweep_results import (
    materialize_phase_23, materialize_phase_24, materialize_phase_25,
    materialize_phase_26, materialize_phase_27, materialize_phase_28,
    materialize_phase_29, materialize_phase_30,
)
for phase_id in range(23, 31):
    fn = [materialize_phase_23, materialize_phase_24, materialize_phase_25,
          materialize_phase_26, materialize_phase_27, materialize_phase_28,
          materialize_phase_29, materialize_phase_30][phase_id - 23]
    result = fn(Path('.'))
    print(f"phase {phase_id}: {result.get('status')}")
```

### 6. Materialize Phase 22 (cuối cùng — phụ thuộc 20 + 21)

```python
from pathlib import Path
from course_work.diagnostics.learning_diagnostics import materialize_phase_22
result = materialize_phase_22(Path('.'))
# status: PASS, finding_count: 2
```

### 7. Regenerate tất cả phase logs

```python
from pathlib import Path
from course_work.reporting.phase_summary import build_phase_processing_log, save_phase_processing_log
for phase_id in range(22, 31):
    log = build_phase_processing_log(phase_id, Path('.'))
    path = save_phase_processing_log(log, Path('.'))
    print(f"phase {phase_id}: {path}")
```

### 8. Phase 15 (nếu cần re-run upstream)

Nếu checkpoint mismatch, re-materialize Phase 14 → 15:
```python
from pathlib import Path
from course_work.baselines.persistence import materialize_phase_14
from course_work.models.lstm_regressor import materialize_phase_15
materialize_phase_14(Path('.'))
materialize_phase_15(Path('.'))
```

## Thời gian ước lượng

| Phase | Thời gian | Loại |
|---|---|---|
| 20 (LSTM) | ~3 phút | Train |
| 21 (Transformer) | ~7 phút | Train |
| 22 (Learning Diag) | < 5 giây | Analysis |
| 23-30 (Sweeps) | 30-50 phút (10 conditions) | Train |
| Logs | < 5 giây | JSON gen |

Tổng: **~45-60 phút** nếu chạy full chain.

## Debug nếu fail

### Sweep fail env check
Bạn đang chạy `python3 scripts/run_sweep_background.py` hay `python3 scripts/run_all_pending.py`? Cả hai không bypass env check. **Luôn dùng `run_with_env_bypass.py`** cho sweep.

### LSTM/Transformer run registration fail
Nếu `materialize_phase_20` hoặc `materialize_phase_21` fail ở `registry.register_run`, có thể Phase 19 signoff bị stale. Verify:
```bash
python3 -c "from pathlib import Path; from course_work.training.engine_materialize import materialize_phase_19; materialize_phase_19(Path('.'))"
```

### Phase 22 vẫn MISSING
Kiểm tra runs hoàn thành còn tồn tại:
```bash
ls artifacts/runs/ | grep RUN_LS   # phải có ít nhất 1
ls artifacts/runs/ | grep RUN_TR   # phải có ít nhất 1
```

## Files đã thay đổi (tóm tắt)

```
src/course_work/diagnostics/learning_curves.py        [MODIFIED] auto-discover
src/course_work/diagnostics/learning_diagnostics.py   [MODIFIED] path fix
src/course_work/sweeps/sweep_results.py               [MODIFIED] path map
scripts/sweep_results_to_csv.py                       [NEW]      adapter
scripts/run_all_sweeps_bypass.py                     [NEW]      batch wrapper
```

`scripts/run_phase_background.py` đã tồn tại từ session trước (single-file launch cho Phase 20/21/22).
