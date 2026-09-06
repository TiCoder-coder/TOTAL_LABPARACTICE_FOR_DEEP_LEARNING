# Phase 2 Checksum Mismatch - Downstream Blocker

## 1. Trạng thái

```
Issue ID: PHASE-2-CHECKSUM-MISMATCH-BLOCKER-v1
Status: ROOT_CAUSE_IDENTIFIED
Severity: BLOCKING_EDA_CHAIN
Detected by: Phase 1 Corrective Step 5 verification
```

## 2. Hiện tượng

Sau khi Phase 1 sign-off được khôi phục thành công, `test_eda.py` vẫn fail với:

```
RuntimeError: Phase 2 artifact checksum mismatch: artifacts/acquisition/acquisition_log.json
```

Stack trace:
```
src/course_work/data/schema.py:263: materialize_phase_3
src/course_work/data/acquisition.py:286: materialize_phase_2
src/course_work/data/acquisition.py:146: verify_existing_signoff
RuntimeError: Phase 2 artifact checksum mismatch: artifacts/acquisition/acquisition_log.json
```

## 3. Phân tích chi tiết

### 3.1 Output checksum mismatch

| Artifact | Declared in sign-off | Actual checksum | Status |
|----------|---------------------|-----------------|--------|
| `artifacts/acquisition/acquisition_log.json` | `f0202f155d2e253ef27c1673cb49d6bb372c87a74fc95832691b8f87626126ad` | `010695e4487fbd39ea466cdd73bafc2bbd0c6b964f2f688d361076154c5186e7` | MISMATCH |

### 3.2 Input checksum reference outdated

| Input | Referenced in Phase 2 | Current after Phase 1 recovery | Status |
|-------|----------------------|-------------------------------|--------|
| `artifacts/environment/phase_1_signoff.json` | `f909c62d3b280c7d8f3c14479fa0366bb87cc97d61c9843e069c76085282546f` | `db146d0a576aacefcb816667a84abef8e11f727c3ff165cdb20d50d2028fbfbd` | MISMATCH |

### 3.3 Git history analysis

| Commit | acquisition_log.json | phase_2_signoff.json |
|--------|---------------------|---------------------|
| b7cf41c4 | `00345f790e...` | checksum: `00345f790e...` |
| 92ab9df1 | `010695e4487fbd39ea466cdd73bafc2bbd0c6b964f2f688d361076154c5186e7` | checksum: `010695e4487fbd39ea466cdd73bafc2bbd0c6b964f2f688d361076154c5186e7` |
| ff6f963a | unchanged | checksum: `f0202f155d...` (CORRUPTED) |

### 3.4 Root cause

Commit `ff6f963a` thay đổi output_checksums trong phase_2_signoff.json nhưng không thay đổi acquisition_log.json. Đây là corruption pattern giống Phase 1.

## 4. Corrective action required

Phase 2 sign-off cần cập nhật:

1. **Output checksum:** `acquisition_log.json` từ `f0202f155d...` → `010695e4487fbd39ea466cdd73bafc2bbd0c6b964f2f688d361076154c5186e7`

2. **Input checksum:** `phase_1_signoff.json` từ `f909c62d3b...` → `db146d0a576aacefcb816667a84abef8e11f727c3ff165cdb20d50d2028fbfbd`

## 5. Hành động tiếp theo

Lập corrective plan chi tiết cho Phase 2.
