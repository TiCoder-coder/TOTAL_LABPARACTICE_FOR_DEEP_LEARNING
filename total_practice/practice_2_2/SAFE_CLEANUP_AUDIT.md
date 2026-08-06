# SAFE CLEANUP AUDIT

**Audit Date**: 2026-08-06
**Type**: Dependency & Safe-Cleanup Audit (Read-Only)

Báo cáo này phân loại toàn bộ cấu trúc mã nguồn của `practice_2_2` dựa trên sơ đồ phụ thuộc (Dependency Map) thực tế. KHÔNG có file nào được phép xóa, di chuyển hoặc archive cho đến khi rò rỉ hình ảnh (Visual Leakage) được sửa hoàn toàn và E1/E2 được train lại thành công.

---

## 1. Kết luận về các file DO_NOT_MOVE_YET

Nghiêm cấm archive các file core legacy (`evaluate.py`, `train.py`, `model.py`, `save_load.py`, `utils.py`) ở thời điểm hiện tại. Dependency Map chứng minh chúng đang có **active imports** từ các thành phần khác trong codebase:

- `model.py` bị import bởi: `canonical_e3_practice_2_2.py`, `canonical_train_practice_2_2.py`, `test_canonical_e3_practice_2_2.py`, `test_canonical_e4_practice_2_2.py`.
- `train.py` bị import bởi: `canonical_e3_practice_2_2.py`, `canonical_train_practice_2_2.py`, `test_regularization_practice_2_2.py`, `test_canonical_e3_practice_2_2.py`, `scripts/validate_active_authority.py`.
- `save_load.py` bị import bởi: `canonical_e3_practice_2_2.py`, `canonical_train_practice_2_2.py`, `final_test_practice_2_2.py`, `scripts/validate_active_authority.py`.
- `utils.py` bị import bởi: `canonical_e3_practice_2_2.py`, `canonical_train_practice_2_2.py`.

Rủi ro: Việc tự ý archive các file này sẽ gây lỗi vỡ dependencies dây chuyền lên hàng loạt bài test và canonical reports.

---

## 2. Kết luận về Visual Leakage & Split V2 Cũ

- **Các file Visual Leakage**: Đã được đưa vào danh sách **KEEP / ACTIVE** (`audit_visual_leakage.py`, `find_all_visual_edges.py`, `build_v2_visual_group_split.py`, `recompute_v2_5a.py`). Chức năng của chúng là bắt buộc ở giai đoạn tiếp theo để làm sạch dữ liệu.
- **`split_v2_pipeline.py`**: Qua quá trình audit, file này được import bởi `tests/test_v2_split.py`. Đây là file đại diện cho split ngẫu nhiên theo V2.1 cũ (đã chứng minh không an toàn). Nó được xếp vào **ARCHIVE_CANDIDATE** nhưng bị **BLOCKED** bởi dependency từ file test.

---

## 3. Bảng phân loại từng File

| File | Imported By | Runtime Use | Status | Proposed Action | Risk |
|---|---|---|---|---|---|
| **ACTIVE (Visual & V2.6 Pipeline)** | | | | | |
| `train_v2_6_baseline.py` | `verify_reload_v2_6`, `test_v2_6_baseline` | Pipeline V2.6 | ACTIVE | Giữ nguyên | Thấp |
| `verify_reload_v2_6.py` | (None) | Validation V2.6 | ACTIVE | Giữ nguyên | Thấp |
| `v2_dataset.py` | `train_v2_baseline`, `test_v2_training`, `test_v2_data_pipeline` | Core Dataloader V2 | ACTIVE | Giữ nguyên | Thấp |
| `v2_transforms.py` | `train_v2_baseline`, `test_v2_data_pipeline` | Core Transform V2 | ACTIVE | Giữ nguyên | Thấp |
| `v2_models.py` | `train_v2_baseline`, `test_v2_training` | Core Architecture V2 | ACTIVE | Giữ nguyên | Thấp |
| `v2_loss_utils.py` | `train_v2_baseline`, `test_v2_data_pipeline` | Core Loss V2 | ACTIVE | Giữ nguyên | Thấp |
| `v2_evaluate_reloaded.py` | (None) | Inference V2 | ACTIVE | Giữ nguyên | Thấp |
| `audit_visual_leakage.py` | (None) | Visual Leakage | ACTIVE | **KEEP** | Cao (Core) |
| `find_all_visual_edges.py` | (None) | Visual Leakage | ACTIVE | **KEEP** | Cao (Core) |
| `build_v2_visual_group_split.py`| (None) | Visual Leakage | ACTIVE | **KEEP** | Cao (Core) |
| `recompute_v2_5a.py` | (None) | Visual Leakage | ACTIVE | **KEEP** | Cao (Core) |
| **SHARED (Common Libs)** | | | | | |
| `paths.py` | R-series, `test_phase5_copies`, `test_paths_resources` | Path resolution | SHARED | Giữ nguyên | Rất Cao |
| `resources.py` | R-series, nhiều Test files | Resource config | SHARED | Giữ nguyên | Rất Cao |
| **DO_NOT_MOVE_YET (Legacy Core)** | | | | | |
| `model.py` | `canonical_e3/4`, `test_canonical_e3/4` | Mạng V1/Canonical | DO_NOT_MOVE_YET| Chờ dọn dẹp Canonical | Cao (Gãy Test) |
| `train.py` | `canonical_train/e3`, `test_reg`, `scripts` | Train V1/Canonical | DO_NOT_MOVE_YET| Chờ dọn dẹp Canonical | Cao (Gãy Test) |
| `save_load.py` | `canonical_train/e3`, `final_test`, `scripts`| I/O V1/Canonical | DO_NOT_MOVE_YET| Chờ dọn dẹp Canonical | Cao (Gãy Test) |
| `utils.py` | `canonical_train/e3` | Tools V1 | DO_NOT_MOVE_YET| Chờ dọn dẹp Canonical | Cao (Gãy Test) |
| `evaluate.py` | (None) | Eval V1 | DO_NOT_MOVE_YET| Chờ dọn dẹp Canonical | Trung bình |
| `data_practice_2_2.py` | `train_practice`, `canonical_*`, `final_test`, `tests` | Dataloader V1 | DO_NOT_MOVE_YET| Chờ dọn dẹp Canonical | Cao (Gãy Test) |
| `analyze_practice_2_2.py` | `test_analyze_practice_2_2` | Phân tích V1 | DO_NOT_MOVE_YET| Chờ dọn dẹp Test | Cao (Gãy Test) |
| `audit_practice_2_2_data.py`| `test_audit_practice_2_2_data` | Audit dữ liệu V1 | DO_NOT_MOVE_YET| Chờ dọn dẹp Test | Cao (Gãy Test) |
| `final_test_practice_2_2.py`| `test_final_test_practice_2_2` | R11 của V1 | DO_NOT_MOVE_YET| Chờ dọn dẹp Test | Cao (Gãy Test) |
| `regularization_practice_2_2.py`| `test_regularization_practice_2_2`| EDA E3 của V1 | DO_NOT_MOVE_YET| Chờ dọn dẹp Test | Cao (Gãy Test) |
| **HISTORICAL (V2.1-V2.5)** | | | | | |
| `train_v2_baseline.py` | `test_v2_training` | Baseline V2.4 | HISTORICAL | Move sau khi Fix Leakage| Trung bình |
| `eda_v2_pipeline.py` | (None) | EDA V2.2 | HISTORICAL | Tái chế Train-only EDA | Thấp |
| `split_v2_pipeline.py` | `test_v2_split` | Split V2.1 | HISTORICAL | Move sau khi Fix Leakage| Cao (Gãy Test) |
| **REDUNDANT** | | | | | |
| `train_practice_2_2.py` | `test_train_practice_2_2` | Huấn luyện rác (V1) | REDUNDANT | Move sau khi Fix Leakage| Thấp |
| **ARCHIVE_CANDIDATE (R0-R12)** | | | | | |
| `r0_...` to `r12_...py` | Scripts `run_r*`, Tests `test_r*`| Protocol over-engineering| ARCHIVE_CANDIDATE| Chờ qua Visual Fix | Thấp (Nếu Archive cùng Test)|
| `configs/r*.json` | R-series Python modules | Configs của R0-R12 | ARCHIVE_CANDIDATE| Chờ qua Visual Fix | Thấp |
| `scripts/run_r*.py` | (None) | Entrypoints R0-R12 | ARCHIVE_CANDIDATE| Chờ qua Visual Fix | Thấp |
| `scripts/verify_r*.py` | (None) | Entrypoints R0-R12 | ARCHIVE_CANDIDATE| Chờ qua Visual Fix | Thấp |
| `tests/test_r*.py` | (None) | Tests R0-R12 | ARCHIVE_CANDIDATE| Chờ qua Visual Fix | Thấp |
| `canonical_e3...py` | `test_canonical_e3` | Baseline E3 cũ | ARCHIVE_CANDIDATE| Chờ qua Visual Fix | Cao (Gãy Test) |
| `canonical_e4...py` | `test_canonical_e4` | Baseline E4 cũ | ARCHIVE_CANDIDATE| Chờ qua Visual Fix | Cao (Gãy Test) |
| `canonical_train_...py`| `canonical_e3/4`, `final_test`, `test` | Baseline V1 | ARCHIVE_CANDIDATE| Chờ dẹp e3/e4 | Dây chuyền gãy |

---

## 4. Reports Migration

Đối với các báo cáo `PHASE_V2_*.md`:
Đề xuất chuyển vào `docs/archive/` vì đây chỉ là bằng chứng lịch sử trước khi Visual Leakage được phát hiện. Việc dọn dẹp các Markdown này không gây vỡ code, nhưng tuân thủ nguyên tắc **CHƯA MOVE TRONG PHASE NÀY** để phục vụ tham chiếu nguyên trạng.

---

## 5. Tổng kết sau Audit

- **Files chắc chắn ACTIVE**: `train_v2_6_baseline.py`, các thư viện `v2_*.py`, các file Visual Leakage (`audit_visual_leakage.py`, `build_v2_visual_group_split.py`, v.v.).
- **Files SHARED**: `paths.py`, `resources.py`.
- **Files HISTORICAL**: `eda_v2_pipeline.py`, `split_v2_pipeline.py`, `train_v2_baseline.py`.
- **Files có thể archive sau**: Toàn bộ R0-R12 (Configs, Code, Scripts, Tests), và cụm Canonical E3/E4.
- **Files nguy hiểm nếu move**: `train.py`, `model.py`, `save_load.py`, `utils.py`, `evaluate.py`, `data_practice_2_2.py`, `canonical_train_practice_2_2.py` (Vỡ depend dây chuyền lên hàng chục Tests và Scripts).
- **Dependency blockers**: Không thể archive R-series/Canonical nếu không archive KÈM toàn bộ Tests và Scripts của chúng trong cùng 1 atomic commit.
- **Proposed Final Tree**:
  - `src/practice_2_2/`: Chỉ chứa pipeline V2.6, V2 Libs, Visual Leakage scripts, và SHARED (`paths.py`, `resources.py`).
  - `tests/`: Chỉ chứa các test cho pipeline V2.6 và Visual Leakage.
  - `docs/`: Chỉ chứa báo cáo Active.
  - `archive/`: Chứa toàn bộ Legacy Core, Canonical, R0-R12 và các file DO_NOT_MOVE_YET sau khi gỡ bỏ chặn từ Tests.

*Tài liệu này không làm thay đổi hay di chuyển bất kỳ tệp tin nào trên hệ thống thực.*
