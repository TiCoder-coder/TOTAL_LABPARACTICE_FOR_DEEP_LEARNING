# CourseWork Notebook — Cell Walkthrough & Output Guide

> Tài liệu này đi kèm với notebook [`CourseWork.ipynb`](../../notebook_course_work/CourseWork.ipynb). Mọi cell index, cell ID và output locator bên dưới được đồng bộ từ notebook hiện tại gồm 151 cell.
>
> ## Cách mở đúng cell output trong Cursor
>
> Liên kết mang nhãn `Cell ...` sử dụng URI nội bộ `vscode-notebook-cell:` của Cursor/VS Code và trỏ thẳng tới cell đích. Với mỗi phase, đích mặc định là code cell đang sở hữu output, không phải markdown heading.
>
> Liên kết `Giải thích` chỉ điều hướng tới phần diễn giải tương ứng trong tài liệu này.
>
> Cell URI dùng cell handle của notebook. Sau khi notebook bị thêm, xóa hoặc di chuyển cell, tài liệu phải được đồng bộ lại. Nếu notebook đang mở từ trước một thay đổi cấu trúc, hãy đóng tab notebook rồi mở lại trước khi kiểm tra deep-link.

---

## Mục Lục

### Phần Khai Báo

- [Cell 0: Title](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#W0sZmlsZQ%3D%3D) — đích Cell 0, ID `coursework-title` · [Giải thích](#cell-0)
- [Cell 1: Define Problem](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#W1sZmlsZQ%3D%3D) — đích Cell 1, ID `define-problem` · [Giải thích](#cell-1)
- [Cell 2: Imports](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#W2sZmlsZQ%3D%3D) — đích Cell 2, ID `0caba13e` · [Giải thích](#cell-2)
- [Cell 3: Display setup](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#W3sZmlsZQ%3D%3D) — đích Cell 3, ID `public-api-imports` · [Giải thích](#cell-3)

### Giai Đoạn Foundation (Phases 1–11)

- [Cell 4–5: Phase 1 — Environment](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#W5sZmlsZQ%3D%3D) — đích Cell 5, ID `phase-1-orchestration` · [Giải thích](#cell-4)
- [Cell 6–7: Phase 2 — Data Acquisition](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X10sZmlsZQ%3D%3D) — đích Cell 7, ID `phase-2-orchestration` · [Giải thích](#cell-6)
- [Cell 8–9: Phase 3 — Schema Audit](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X12sZmlsZQ%3D%3D) — đích Cell 9, ID `phase-3-orchestration` · [Giải thích](#cell-8)
- [Cell 10–11: Phase 4 — Temporal Integrity](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X14sZmlsZQ%3D%3D) — đích Cell 11, ID `phase-4-orchestration` · [Giải thích](#cell-10)
- [Cell 12–13: Phase 5 — Chronological Split](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X16sZmlsZQ%3D%3D) — đích Cell 13, ID `phase-5-orchestration` · [Giải thích](#cell-12)
- [Cell 14–43: Phase 6 — EDA](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X23sZmlsZQ%3D%3D) — đích Cell 17, ID `phase-6-data-overview` · [Giải thích](#cell-14)
- [Cell 44–45: Phase 7 — Feature Engineering](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X63sZmlsZQ%3D%3D) — đích Cell 45, ID `phase-7-orchestration` · [Giải thích](#cell-44)
- [Cell 46–47: Phase 8 — Feature-Set Variants](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X65sZmlsZQ%3D%3D) — đích Cell 47, ID `phase-8-orchestration` · [Giải thích](#cell-46)
- [Cell 48–49: Phase 9 — Train-Only Scaling](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y100sZmlsZQ%3D%3D) — đích Cell 49, ID `phase-9-orchestration` · [Giải thích](#cell-48)
- [Cell 50–51: Phase 10 — Window Builder](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y102sZmlsZQ%3D%3D) — đích Cell 51, ID `phase-10-orchestration` · [Giải thích](#cell-50)
- [Cell 52–53: Phase 11 — DataLoaders](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y104sZmlsZQ%3D%3D) — đích Cell 53, ID `phase-11-orchestration` · [Giải thích](#cell-52)

### Giai Đoạn Modeling (Phases 12–22)

- [Cell 54–55: Phase 12 — Shared Metrics](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y106sZmlsZQ%3D%3D) — đích Cell 55, ID `phase-12-orchestration` · [Giải thích](#cell-54)
- [Cell 56–57: Phase 13 — Experiment Registry](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y111sZmlsZQ%3D%3D) — đích Cell 57, ID `phase-13-orchestration` · [Giải thích](#cell-56)
- [Cell 58–59: Phase 14 — Persistence Baseline](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y113sZmlsZQ%3D%3D) — đích Cell 59, ID `phase-14-orchestration` · [Giải thích](#cell-58)
- [Cell 60–61: Phase 15 — LSTM Implementation](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y115sZmlsZQ%3D%3D) — đích Cell 61, ID `499bc911` · [Giải thích](#cell-60)
- [Cell 62–63: Phase 16 — Transformer Implementation](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y120sZmlsZQ%3D%3D) — đích Cell 63, ID `518fca61` · [Giải thích](#cell-62)
- [Cell 64–65: Phase 17 — Attention Verification](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y122sZmlsZQ%3D%3D) — đích Cell 65, ID `efc7f487` · [Giải thích](#cell-64)
- [Cell 66–67: Phase 18 — Forward Sanity](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y124sZmlsZQ%3D%3D) — đích Cell 67, ID `d60e1951` · [Giải thích](#cell-66)
- [Cell 68–69: Phase 19 — Training Engine](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y126sZmlsZQ%3D%3D) — đích Cell 69, ID `2acc38f9` · [Giải thích](#cell-68)
- [Cell 70–71: Phase 20 — LSTM Baseline Run](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y131sZmlsZQ%3D%3D) — đích Cell 71, ID `2eb13c9c` · [Giải thích](#cell-70)
- [Cell 72–73: Phase 21 — Transformer B0 Run](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y133sZmlsZQ%3D%3D) — đích Cell 73, ID `03db9e36` · [Giải thích](#cell-72)
- [Cell 74–75: Phase 22 — Learning-Curve Diagnostics](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y135sZmlsZQ%3D%3D) — đích Cell 75, ID `cd4716af` · [Giải thích](#cell-74)

### Giai Đoạn Sweeps (Phases 23–41)

- [Cell 76–77: Phase 23 — S1 Feature-Set](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y140sZmlsZQ%3D%3D) — đích Cell 77, ID `0881cfe3` · [Giải thích](#cell-76)
- [Cell 78–79: Phase 24 — S2 Time-Feature](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y142sZmlsZQ%3D%3D) — đích Cell 79, ID `9b2d6e88` · [Giải thích](#cell-78)
- [Cell 80–81: Phase 25 — S3 Target-Scaling](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y144sZmlsZQ%3D%3D) — đích Cell 81, ID `51bc5965` · [Giải thích](#cell-80)
- [Cell 82–83: Phase 26 — S4 Lookback](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y146sZmlsZQ%3D%3D) — đích Cell 83, ID `585b390d` · [Giải thích](#cell-82)
- [Cell 84–85: Phase 27 — S5 Pooling](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y151sZmlsZQ%3D%3D) — đích Cell 85, ID `86f4ac0c` · [Giải thích](#cell-84)
- [Cell 86–87: Phase 28 — S6 Activation](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y153sZmlsZQ%3D%3D) — đích Cell 87, ID `b70c9707` · [Giải thích](#cell-86)
- [Cell 88–89: Phase 29 — S7 Batch-Size](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y155sZmlsZQ%3D%3D) — đích Cell 89, ID `eb4f1b80` · [Giải thích](#cell-88)
- [Cell 90–91: Phase 30 — S8 Learning-Rate](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y160sZmlsZQ%3D%3D) — đích Cell 91, ID `3fe4478c` · [Giải thích](#cell-90)
- [Cell 92–93: Phase 31 — S9 Weight-Decay](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y162sZmlsZQ%3D%3D) — đích Cell 93, ID `phase-31-resume` · [Giải thích](#cell-92)
- [Cell 94–95: Phase 32 — S10 Dropout](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y164sZmlsZQ%3D%3D) — đích Cell 95, ID `phase-32-resume` · [Giải thích](#cell-94)
- [Cell 96–97: Phase 33 — Transformer Config](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y166sZmlsZQ%3D%3D) — đích Cell 97, ID `phase-33-config-display` · [Giải thích](#cell-96)
- [Cell 98–99: Phase 34 — S12 Head](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y201sZmlsZQ%3D%3D) — đích Cell 99, ID `phase-34-resume` · [Giải thích](#cell-98)
- [Cell 100–101: Phase 35 — S13 Layer](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y203sZmlsZQ%3D%3D) — đích Cell 101, ID `phase-35-resume` · [Giải thích](#cell-100)
- [Cell 102–103: Phase 36 — S14 FFN](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y205sZmlsZQ%3D%3D) — đích Cell 103, ID `phase-36-resume` · [Giải thích](#cell-102)
- [Cell 104–105: Phase 37 — S15 Loss](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y210sZmlsZQ%3D%3D) — đích Cell 105, ID `phase-37-resume` · [Giải thích](#cell-104)
- [Cell 106–107: Phase 38 — S16 Epoch-Cap](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y212sZmlsZQ%3D%3D) — đích Cell 107, ID `17dc5a66` · [Giải thích](#cell-106)
- [Cell 108–109: Phase 39 — S17 Gradient-Clip](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y214sZmlsZQ%3D%3D) — đích Cell 109, ID `f0fcdce7` · [Giải thích](#cell-108)
- [Cell 110–111: Phase 40 — S18 RevIN](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y216sZmlsZQ%3D%3D) — đích Cell 111, ID `d407f95c` · [Giải thích](#cell-110)
- [Cell 112–113: Phase 41 — S19 Boundary](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y221sZmlsZQ%3D%3D) — đích Cell 113, ID `7e86663d` · [Giải thích](#cell-112)

### Giai Đoạn Final Pipeline (Phases 42–47)

- [Cell 114–115: Phase 42 — Candidate Synthesis](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y223sZmlsZQ%3D%3D) — đích Cell 115, ID `994bbdb9` · [Giải thích](#cell-114)
- [Cell 116–117: Phase 43 — LSTM Tuning](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y225sZmlsZQ%3D%3D) — đích Cell 117, ID `6a082d01` · [Giải thích](#cell-116)
- [Cell 118–119: Phase 44 — Rolling-Origin](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y230sZmlsZQ%3D%3D) — đích Cell 119, ID `2f0130ea` · [Giải thích](#cell-118)
- [Cell 120–121: Phase 45 — Final Model Lock](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y232sZmlsZQ%3D%3D) — đích Cell 121, ID `4e6b273d` · [Giải thích](#cell-120)
- [Cell 122–123: Phase 46 — Three-Seed Runs](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y234sZmlsZQ%3D%3D) — đích Cell 123, ID `753ab0f1` · [Giải thích](#cell-122)
- [Cell 124–125: Phase 47 — Final Test Evaluation](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y236sZmlsZQ%3D%3D) — đích Cell 125, ID `4dea0c14` · [Giải thích](#cell-124)

### Phần Bổ Sung Metric

- [Cell 126: Supplementary MAPE Addendum](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y240sZmlsZQ%3D%3D) — đích Cell 126, ID `ac4fe7dd` · [Giải thích](#cell-126)

### Giai Đoạn Analysis (Phases 48–59)

- [Cell 127–128: Phase 48 — Prediction Analysis](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y242sZmlsZQ%3D%3D) — đích Cell 128, ID `1451ccf0` · [Giải thích](#cell-127)
- [Cell 129–130: Phase 49 — Residual Analysis](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y244sZmlsZQ%3D%3D) — đích Cell 130, ID `bbcf08f2` · [Giải thích](#cell-129)
- [Cell 131–132: Phase 50 — Error-by-Regime](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y246sZmlsZQ%3D%3D) — đích Cell 132, ID `1d5b5327` · [Giải thích](#cell-131)
- [Cell 133–134: Phase 51 — Worst-Error](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y251sZmlsZQ%3D%3D) — đích Cell 134, ID `c9df346a` · [Giải thích](#cell-133)
- [Cell 135–136: Phase 52 — Attention Extraction](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y253sZmlsZQ%3D%3D) — đích Cell 136, ID `7b48186f` · [Giải thích](#cell-135)
- [Cell 137–138: Phase 53 — Attention Heatmaps](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y255sZmlsZQ%3D%3D) — đích Cell 138, ID `1c098fa6` · [Giải thích](#cell-137)
- [Cell 139–140: Phase 54 — Last-Query Attention](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y260sZmlsZQ%3D%3D) — đích Cell 140, ID `ec7b2ea5` · [Giải thích](#cell-139)
- [Cell 141–142: Phase 55 — Head Comparison](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y262sZmlsZQ%3D%3D) — đích Cell 142, ID `05ec07a2` · [Giải thích](#cell-141)
- [Cell 143–144: Phase 56 — Error-Conditioned](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y264sZmlsZQ%3D%3D) — đích Cell 144, ID `5e9eec01` · [Giải thích](#cell-143)
- [Cell 145–146: Phase 57 — Seed-Stability](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y266sZmlsZQ%3D%3D) — đích Cell 146, ID `4293c1aa` · [Giải thích](#cell-145)
- [Cell 147–148: Phase 58 — Final Tables](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y301sZmlsZQ%3D%3D) — đích Cell 148, ID `9f864b91` · [Giải thích](#cell-147)
- [Cell 149–150: Phase 59 — Final Conclusions](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y303sZmlsZQ%3D%3D) — đích Cell 150, ID `76de7c69` · [Giải thích](#cell-149)

---

## Phần Khai Báo (Cells 0-3)

### <a id="cell-0"></a>Cell 0 — Title

**Cell ID:** `coursework-title`
**Loại:** Markdown

**Nội dung:** Tiêu đề `# MULTIVARIATE TIME-SERIES REGRESSION` cho notebook.

**Giải thích:**
- Đây là markdown cell đầu tiên, không có output.
- Dùng để giới thiệu tổng quan project là **Multivariate Time-Series Regression** trên dataset UCI Appliances Energy Prediction.

[Mở trực tiếp Cell 0 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#W0sZmlsZQ%3D%3D) → tìm cell có text `# MULTIVARIATE TIME-SERIES REGRESSION`

---

### <a id="cell-1"></a>Cell 1 — Define Problem

**Cell ID:** `define-problem`
**Loại:** Markdown

**Nội dung:** Markdown `## Define Problem`.

**Giải thích:**
- Markdown cell định nghĩa bài toán, không có output.

[Mở trực tiếp Cell 1 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#W1sZmlsZQ%3D%3D) → tìm cell có text `## Define Problem`

---

### <a id="cell-2"></a>Cell 2 — Imports

**Cell ID:** `0caba13e`
**Loại:** Code

**Nội dung:** `import sys`, các imports hệ thống.

**Output kỳ vọng:** Không có output stdout (chỉ là imports). Nếu có lỗi sẽ xuất `ModuleNotFoundError` hoặc `ImportError`.

**Giải thích:**
- Import `sys` và các thiết lập path.
- Đây là cell import đầu tiên, nếu fail sẽ chặn toàn bộ pipeline.

[Mở trực tiếp Cell 2 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#W2sZmlsZQ%3D%3D) → bấm vào cell chứa `import sys`

---

### <a id="cell-3"></a>Cell 3 — Display Setup

**Cell ID:** `public-api-imports`
**Loại:** Code

**Nội dung:** `from IPython.display import Image, display`.

**Output kỳ vọng:** Không có output. Chỉ là import.

**Giải thích:**
- Import `Image` và `display` từ IPython để hiển thị ảnh PNG (EDA figures, heatmaps) inline trong notebook.

[Mở trực tiếp Cell 3 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#W3sZmlsZQ%3D%3D) → bấm vào cell chứa `from IPython.display import Image, display`

---

## Giai Đoạn Foundation (Cells 4-53)

### <a id="cell-4"></a>Cell 4-5 — Phase 1: Environment

**Cell 4 (markdown):** `phase-1-heading` ID, tiêu đề `## Phase 1 - Environment`
**Cell 5 (code):** `phase-1-orchestration` ID, chạy `materialize_phase_1(PROJECT_ROOT)`

**Output kỳ vọng của Cell 5:**
- In ra một dictionary với environment fingerprint (Python version, torch version, deterministic mode, ...).
- Ghi file `artifacts/environment/environment_report.json`.
- Ghi file `artifacts/environment/phase_1_signoff.json` với status `PASS`.

**Giải thích:**
- Phase 1 capture environment, freeze requirements, smoke test imports.
- Nếu output báo lỗi về kernel hoặc import, kiểm tra lại `requirements_freeze.txt`.
- Sign-off `PASS` là điều kiện tiên quyết để chạy Phase 2.

[Mở trực tiếp Cell 5 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#W5sZmlsZQ%3D%3D) → tìm cell chứa `phase_1_signoff = materialize_phase_1`

---

### <a id="cell-6"></a>Cell 6-7 — Phase 2: Data Acquisition

**Cell 6 (markdown):** Tiêu đề `## Phase 2 - Data Acquisition`
**Cell 7 (code):** Chạy `materialize_phase_2(PROJECT_ROOT)`

**Output kỳ vọng của Cell 7:**
- Dictionary chứa dataset manifest: rows, columns, source SHA256, acquisition timestamp.
- File `data/raw_data/energydata_complete.csv` được materialize.
- File `artifacts/acquisition/dataset_manifest.json` và `phase_2_signoff.json` được tạo.

**Giải thích:**
- Phase kiểm tra SHA256 checksum của raw data, đảm bảo dữ liệu đúng với nguồn.
- Nếu checksum mismatch, xem file `phase_2_signoff.json` để biết lý do.
- Output thường có `n_rows ≈ 19735` (mẫu 10 phút × ~4.5 tháng).

[Mở trực tiếp Cell 7 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X10sZmlsZQ%3D%3D) → tìm cell chứa `phase_2_signoff = materialize_phase_2`

---

### <a id="cell-8"></a>Cell 8-9 — Phase 3: Schema Audit

**Cell 8 (markdown):** Tiêu đề `## Phase 3 - Schema Audit`
**Cell 9 (code):** Chạy `materialize_phase_3(PROJECT_ROOT)`

**Output kỳ vọng của Cell 9:**
- Manifest: 29 columns, dtypes cho mỗi biến, missing counts.
- File `artifacts/schema/{schema_manifest.json, schema_summary.csv, variable_dictionary.csv}`.

**Giải thích:**
- Schema audit đảm bảo dtypes và cấu trúc cột khớp với contract.
- `variable_dictionary.csv` chứa metadata cho mỗi biến (đơn vị, role, range).

[Mở trực tiếp Cell 9 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X12sZmlsZQ%3D%3D) → tìm cell chứa `phase_3_signoff = materialize_phase_3`

---

### <a id="cell-10"></a>Cell 10-11 — Phase 4: Temporal Integrity

**Cell 10 (markdown):** Tiêu đề `## Phase 4 - Temporal Integrity Audit`
**Cell 11 (code):** Chạy `materialize_phase_4(PROJECT_ROOT)`

**Output kỳ vọng của Cell 11:**
- Thống kê: cadence = 10 phút, no gaps, no duplicates, continuity segments.
- File `artifacts/temporal/{temporal_manifest.json, daily_observation_counts.csv, interval_distribution.csv}`.

**Giải thích:**
- Verify time series đều đặn ở cadence 10 phút.
- Nếu có gaps, sẽ có `continuity_segments.csv` liệt kê các đoạn liên tục.
- Đây là gate quan trọng trước khi EDA.

[Mở trực tiếp Cell 11 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X14sZmlsZQ%3D%3D) → tìm cell chứa `phase_4_signoff = materialize_phase_4`

---

### <a id="cell-12"></a>Cell 12-13 — Phase 5: Chronological Split

**Cell 12 (markdown):** Tiêu đề `## Phase 5 - Chronological Split`
**Cell 13 (code):** Chạy `materialize_phase_5(PROJECT_ROOT)`

**Output kỳ vọng của Cell 13:**
- Split statistics: Train 70% / Validation 15% / Test 15%.
- File `artifacts/splits/{split_manifest.json, split_membership.csv, split_boundaries.csv, split_leakage_audit.csv}`.

**Giải thích:**
- Split chronological: KHÔNG shuffle (giữ nguyên thứ tự thời gian).
- Leakage audit đảm bảo không có sample overlap giữa Train/Val/Test.
- Boundaries xác định timestamp cắt: thường Train `≤ 2016-04-15`, Val `2016-04-15 → 2016-05-12`, Test `≥ 2016-05-12`.

[Mở trực tiếp Cell 13 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X16sZmlsZQ%3D%3D) → tìm cell chứa `phase_5_signoff = materialize_phase_5`

---

### <a id="cell-14"></a>Cell 14-43 — Phase 6: Exploratory Data Analysis (14 sub-sections)

**Cell 14 (markdown):** Tiêu đề `## Phase 6 - Exploratory Data Analysis`
**Cell 15 (code):** `from io import BytesIO`, đọc interim CSV

**Các sub-cells của Phase 6:**

| Sub-cell | Markdown Header | Code action | Output kỳ vọng |
|---|---|---|---|
| [Cell 16](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X22sZmlsZQ%3D%3D) | `### 6.1 Data Overview` | — | — |
| [Cell 17](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X23sZmlsZQ%3D%3D) | — | `df.head(8)` | DataFrame 8 dòng đầu, 29 cột |
| [Cell 18](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X24sZmlsZQ%3D%3D) | `### 6.2 Missing-Value Analysis` | — | — |
| [Cell 19](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X25sZmlsZQ%3D%3D) | — | `df.isna().sum()` | Series đếm missing mỗi cột (thường tất cả = 0) |
| [Cell 20](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X26sZmlsZQ%3D%3D) | `### 6.3 Calendar Derivations` | — | — |
| [Cell 21](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X30sZmlsZQ%3D%3D) | — | `df["hour"] = df["date"].dt.hour` | Không output, side-effect |
| [Cell 22](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X31sZmlsZQ%3D%3D) | `### 6.4 Target Distribution` | — | — |
| [Cell 23](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X32sZmlsZQ%3D%3D) | — | đọc `eda_numeric_summary.csv` | Bảng summary target |
| [Cell 24](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X33sZmlsZQ%3D%3D) | `### 6.5 Target Timeline` | — | — |
| [Cell 25](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X34sZmlsZQ%3D%3D) | — | `df.set_index("date")` | Setup cho plotting |
| [Cell 26](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X35sZmlsZQ%3D%3D) | `### 6.6 Calendar Energy Patterns` | — | — |
| [Cell 27](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X36sZmlsZQ%3D%3D) | — | display CSV | Bảng hourly profile |
| [Cell 28](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X40sZmlsZQ%3D%3D) | `### 6.7 Feature Distributions` | — | — |
| [Cell 29](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X41sZmlsZQ%3D%3D) | — | `Image(filename=EDA_10_...)` | Hình phân phối features |
| [Cell 30](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X42sZmlsZQ%3D%3D) | `### 6.8 Numerical Relationships` | — | — |
| [Cell 31](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X43sZmlsZQ%3D%3D) | — | setup sensor_columns | List tên cột cảm biến |
| [Cell 32](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X44sZmlsZQ%3D%3D) | `### 6.9 Cross-Correlation at Short Lags` | — | — |
| [Cell 33](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X45sZmlsZQ%3D%3D) | — | define cross-corr function | Không output |
| [Cell 34](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X46sZmlsZQ%3D%3D) | `### 6.10 Categorical and Numerical` | — | — |
| [Cell 35](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X50sZmlsZQ%3D%3D) | — | groupby is_weekend | Describe table |
| [Cell 36](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X51sZmlsZQ%3D%3D) | `### 6.11 IQR Outlier Diagnostics` | — | — |
| [Cell 37](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X52sZmlsZQ%3D%3D) | — | define IQR summary | Không output |
| [Cell 38](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X53sZmlsZQ%3D%3D) | `### 6.12 Outlier-Smoothing Demo` | — | — |
| [Cell 39](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X54sZmlsZQ%3D%3D) | — | df.copy(deep=True) | Không output |
| [Cell 40](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X55sZmlsZQ%3D%3D) | `### 6.13 Time-Series Diagnostics` | — | — |
| [Cell 41](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X56sZmlsZQ%3D%3D) | — | đọc lag correlations | Bảng lag correlation |
| [Cell 42](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X60sZmlsZQ%3D%3D) | `### 6.14 Extreme Samples` | — | — |
| [Cell 43](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X61sZmlsZQ%3D%3D) | — | đọc extreme_target_samples | Bảng extreme samples |

**Giải thích Phase 6:**
- Phase dài nhất trong foundation (30 cells), mục đích khám phá dữ liệu toàn diện.
- Kỳ vọng output: tables (CSV) + figures (PNG) được đọc/hiển thị inline.
- Tất cả figures nằm trong `artifacts/eda/figures/`, tables trong `artifacts/eda/tables/`.

[Mở trực tiếp output đầu tiên của Phase 6 tại Cell 17](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X23sZmlsZQ%3D%3D) → tìm cells từ `## Phase 6 - Exploratory Data Analysis` đến trước `## Phase 7`

---

### <a id="cell-44"></a>Cell 44-45 — Phase 7: Feature Engineering

**Cell 44 (markdown):** Tiêu đề `## Phase 7 - Feature Engineering`
**Cell 45 (code):** Chạy `materialize_phase_7(PROJECT_ROOT)`

**Output kỳ vọng của Cell 45:**
- Manifest với feature list (time features, lag features, rolling stats).
- File `data/interim/uci_appliances_energy_prediction/energydata_feature_engineered_v1.csv`.
- File `artifacts/features/{feature_engineering_manifest.json, feature_registry.csv, feature_lineage.csv}`.

**Giải thích:**
- Engineering features chỉ dựa trên Train, KHÔNG leak từ Val/Test.
- File `feature_engineered_v1.csv` có ~33 cột (29 gốc + 4 time + lag + rolling).
- SHA256 file `feature_engineered_v1.sha256` đảm bảo determinism.

[Mở trực tiếp Cell 45 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X63sZmlsZQ%3D%3D) → tìm cell chứa `phase_7_signoff = materialize_phase_7`

---

### <a id="cell-46"></a>Cell 46-47 — Phase 8: Feature-Set Variants

**Cell 46 (markdown):** Tiêu đề `## Phase 8 - Feature-Set Variants`
**Cell 47 (code):** Chạy `materialize_phase_8(PROJECT_ROOT)`

**Output kỳ vọng của Cell 47:**
- Registry 3 variants: FS0, FS1, FS2.
- File `artifacts/feature_sets/{feature_set_registry.json, feature_components.json, feature_set_lineage.csv}`.

**Giải thích:**
- 3 variants để sweep Phase 23 (S1).
- FS0 = minimal, FS1 = + time, FS2 = + lag features.

[Mở trực tiếp Cell 47 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#X65sZmlsZQ%3D%3D) → tìm cell chứa `phase_8_signoff = materialize_phase_8`

---

### <a id="cell-48"></a>Cell 48-49 — Phase 9: Train-Only Scaling

**Cell 48 (markdown):** Tiêu đề `## Phase 9 - Train-Only Scaling`
**Cell 49 (code):** Chạy `materialize_phase_9(PROJECT_ROOT)`

**Output kỳ vọng của Cell 49:**
- Scaler fingerprints (StandardScaler for X, StandardScaler for y).
- File `artifacts/scalers/{x,y}/*.joblib`.
- File `artifacts/scaling/{scaler_registry.json, scaling_audit.csv, scaling_discrepancies.json}`.

**Giải thích:**
- Scaler được FIT trên Train only.
- Val và Test được TRANSFORM bằng scaler của Train → không leak.
- Output thường chứa `scaler.mean_` và `scaler.scale_` per feature.

[Mở trực tiếp Cell 49 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y100sZmlsZQ%3D%3D) → tìm cell chứa `phase_9_signoff = materialize_phase_9`

---

### <a id="cell-50"></a>Cell 50-51 — Phase 10: Window Builder

**Cell 50 (markdown):** Tiêu đề `## Phase 10 - Window Builder`
**Cell 51 (code):** Chạy `materialize_phase_10(PROJECT_ROOT)`

**Output kỳ vọng của Cell 51:**
- Window population summary.
- File `artifacts/windows/{window_manifest.json, window_population_summary.csv, window_fingerprints.json}`.

**Giải thích:**
- Build (X, y) pairs với lookback ∈ {36, 72, 144}.
- Population phải giảm dần: Train > Val > Test (do mỗi sample cần lookback).

[Mở trực tiếp Cell 51 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y102sZmlsZQ%3D%3D) → tìm cell chứa `phase_10_signoff = materialize_phase_10`

---

### <a id="cell-52"></a>Cell 52-53 — Phase 11: DataLoaders

**Cell 52 (markdown):** Tiêu đề `## Phase 11 - DataLoaders`
**Cell 53 (code):** Chạy `materialize_phase_11(PROJECT_ROOT)`

**Output kỳ vọng của Cell 53:**
- DataLoader configs: batch_size, shuffle policy (Train=shuffle, Val/Test=no shuffle).
- File `artifacts/dataloaders/{dataloader_manifest.json, dataloader_registry.csv, sequential_order_audit.csv}`.

**Giải thích:**
- Train DataLoader shuffle với seed cố định.
- Val/Test DataLoader KHÔNG shuffle để giữ thứ tự thời gian.

[Mở trực tiếp Cell 53 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y104sZmlsZQ%3D%3D) → tìm cell chứa `phase_11_signoff = materialize_phase_11`

---

## Giai Đoạn Modeling (Cells 54-75)

### <a id="cell-54"></a>Cell 54-55 — Phase 12: Shared Metrics

**Cell 54 (markdown):** Tiêu đề `## Phase 12 - Shared Metrics`
**Cell 55 (code):** Chạy `materialize_phase_12(PROJECT_ROOT)`

**Output kỳ vọng của Cell 55:**
- Manifest với metrics: MAE, RMSE, R², MAPE (addendum).
- File `artifacts/metrics/{metric_manifest.json, metric_unit_tests.csv, metric_reference_examples.csv}`.

**Giải thích:**
- Metrics tính ở đơn vị Wh gốc (inverse_transform y_pred, y_true).
- R² ở Wh scale (không phải scaled space).

[Mở trực tiếp Cell 55 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y106sZmlsZQ%3D%3D) → tìm cell chứa `phase_12_signoff = materialize_phase_12`

---

### <a id="cell-56"></a>Cell 56-57 — Phase 13: Experiment Registry

**Cell 56 (markdown):** Tiêu đề `## Phase 13 - Experiment Registry`
**Cell 57 (code):** Chạy `materialize_phase_13(PROJECT_ROOT)`

**Output kỳ vọng của Cell 57:**
- Empty registry (chưa có run nào).
- File `artifacts/experiments/{experiment_registry.jsonl, registry_manifest.json, sweep_registry.csv}`.

**Giải thích:**
- Registry sẽ được fill dần khi các sweep phases chạy.
- Mỗi training run được register với: id, config, status, metrics.

[Mở trực tiếp Cell 57 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y111sZmlsZQ%3D%3D) → tìm cell chứa `phase_13_signoff = materialize_phase_13`

---

### <a id="cell-58"></a>Cell 58-59 — Phase 14: Persistence Baseline

**Cell 58 (markdown):** Tiêu đề `## Phase 14 - Persistence Baseline`
**Cell 59 (code):** Chạy `materialize_phase_14(PROJECT_ROOT)`

**Output kỳ vọng của Cell 59:**
- Validation metrics cho persistence: MAE ≈ 26 Wh, RMSE ≈ 47 Wh, R² ≈ 0.20.
- File `artifacts/baselines/persistence/{persistence_manifest.json, persistence_validation_metrics.json}`.

**Giải thích:**
- Persistence: $\hat{y}[t+1] = y[t]$ (giá trị step trước).
- Đây là baseline yếu nhất, nhưng khó bị đánh bại ở những biến động nhỏ.
- Nếu LSTM/Transformer không beat được → có vấn đề về features hoặc training.

[Mở trực tiếp Cell 59 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y113sZmlsZQ%3D%3D) → tìm cell chứa `phase_14_signoff = materialize_phase_14`

---

### <a id="cell-60"></a>Cell 60-61 — Phase 15: LSTM Implementation

**Cell 60 (markdown):** Tiêu đề `## Phase 15 - LSTM Implementation`
**Cell 61 (code):** Chạy `materialize_phase_15(PROJECT_ROOT)`

**Output kỳ vọng của Cell 61:**
- LSTM architecture fingerprint: hidden_size, num_layers, dropout, total_params.
- File `artifacts/models/lstm/{lstm_model_manifest.json, lstm_shape_contract.json, lstm_unit_tests.csv}`.

**Giải thích:**
- Implement LSTM regressor (KHÔNG train ở phase này).
- Shape contract đảm bảo input `(batch, lookback, n_features)` → output `(batch, 1)`.
- Unit tests verify shape correctness trên dummy input.

[Mở trực tiếp Cell 61 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y115sZmlsZQ%3D%3D) → tìm cell chứa `phase_15_signoff = materialize_phase_15`

---

### <a id="cell-62"></a>Cell 62-63 — Phase 16: Transformer Implementation

**Cell 62 (markdown):** Tiêu đề `## Phase 16 - Transformer Implementation`
**Cell 63 (code):** Chạy `materialize_phase_16(PROJECT_ROOT)`

**Output kỳ vọng của Cell 63:**
- Transformer architecture fingerprint: d_model, num_heads, num_layers, ffn_dim, total_params.
- File `artifacts/models/transformer/{transformer_model_manifest.json, transformer_attention_contract.json, transformer_shape_contract.json, ...}`.

**Giải thích:**
- Implement Transformer encoder regressor (KHÔNG train).
- Attention contract: output attention shape `(batch, num_heads, seq, seq)`.
- Positional encoding sin/cos được add vào input embedding.

[Mở trực tiếp Cell 63 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y120sZmlsZQ%3D%3D) → tìm cell chứa `phase_16_signoff = materialize_phase_16`

---

### <a id="cell-64"></a>Cell 64-65 — Phase 17: Attention Verification

**Cell 64 (markdown):** Tiêu đề `## Phase 17 - Attention-Aware Encoder Verification`
**Cell 65 (code):** Chạy `materialize_phase_17(PROJECT_ROOT)`

**Output kỳ vọng của Cell 65:**
- Audit reports: shape audit, probability audit (sum=1), mask audit, path equivalence.
- File `artifacts/attention_verification/{attention_verification_manifest.json, attention_probability_audit.csv, attention_path_equivalence_audit.csv, ...}`.

**Giải thích:**
- Verify attention weights là valid probability distribution.
- Path equivalence: attention từ manual computation == framework output.
- Nếu FAIL ở phase này → Phase 18 sẽ không pass.

[Mở trực tiếp Cell 65 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y122sZmlsZQ%3D%3D) → tìm cell chứa `phase_17_signoff = materialize_phase_17`

---

### <a id="cell-66"></a>Cell 66-67 — Phase 18: Forward-Pass Sanity

**Cell 66 (markdown):** Tiêu đề `## Phase 18 - Forward-Pass Sanity Tests`
**Cell 67 (code):** Chạy `materialize_phase_18(PROJECT_ROOT)`

**Output kỳ vọng của Cell 67:**
- Forward pass tests: batch independence, parameter non-mutation, device transfer, scaling correctness.
- File `artifacts/forward_sanity/{forward_sanity_manifest.json, forward_batch_audit.csv, ...}`.

**Giải thích:**
- Sanity tests trên model đã initialized (chưa trained).
- Đảm bảo forward pass deterministic và không side-effect.

[Mở trực tiếp Cell 67 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y124sZmlsZQ%3D%3D) → tìm cell chứa `phase_18_signoff = materialize_phase_18`

---

### <a id="cell-68"></a>Cell 68-69 — Phase 19: Training Engine

**Cell 68 (markdown):** Tiêu đề `## Phase 19 - Baseline Training Engine`
**Cell 69 (code):** Chạy `materialize_phase_19(PROJECT_ROOT)`

**Output kỳ vọng của Cell 69:**
- Training engine contract: optimizer template, scheduler template, checkpoint schema.
- File `artifacts/training_engine/{training_engine_manifest.json, training_engine_unit_tests.csv, checkpoint_schema.json}`.

**Giải thích:**
- Implement generic training loop (KHÔNG train model thật ở đây).
- Engine hỗ trợ: gradient clipping, early stopping, checkpoint save/load.

[Mở trực tiếp Cell 69 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y126sZmlsZQ%3D%3D) → tìm cell chứa `phase_19_signoff = materialize_phase_19`

---

### <a id="cell-70"></a>Cell 70-71 — Phase 20: LSTM Baseline Run

**Cell 70 (markdown):** Tiêu đề `## Phase 20 - LSTM Baseline Run`
**Cell 71 (code):** Chạy `materialize_phase_20(PROJECT_ROOT)`

**Output kỳ vọng của Cell 71:**
- LSTM training log + final validation metrics.
- File `artifacts/runs/RUN_LS_LS_*/{config.json, status.json, training.log, training_history.csv, metrics/best_validation_metrics.json}`.

**Giải thích:**
- Train LSTM baseline với reference config.
- Validation RMSE thường ≈ 73 Wh (vs persistence 47 Wh → LSTM ban đầu chưa beat persistence vì hyperparameters chưa tuned).

[Mở trực tiếp Cell 71 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y131sZmlsZQ%3D%3D) → tìm cell chứa `phase_20_signoff = materialize_phase_20`

---

### <a id="cell-72"></a>Cell 72-73 — Phase 21: Transformer B0 Run

**Cell 72 (markdown):** Tiêu đề `## Phase 21 - Transformer B0 Run`
**Cell 73 (code):** Chạy `materialize_phase_21(PROJECT_ROOT)`

**Output kỳ vọng của Cell 73:**
- Transformer B0 training log + validation metrics.
- File `artifacts/runs/RUN_TR_B0_*/` (config, status, history, metrics).

**Giải thích:**
- Train Transformer B0 (baseline config chưa tune).
- Thường có RMSE ≈ 60-70 Wh.

[Mở trực tiếp Cell 73 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y133sZmlsZQ%3D%3D) → tìm cell chứa `phase_21_signoff = materialize_phase_21`

---

### <a id="cell-74"></a>Cell 74-75 — Phase 22: Learning-Curve Diagnostics

**Cell 74 (markdown):** Tiêu đề `## Phase 22 - Learning-Curve Diagnostics`
**Cell 75 (code):** `render_phase_summary(...)` cho Phase 20/21

**Output kỳ vọng của Cell 75:**
- HTML dashboard hiển thị learning curves cho các baseline runs.
- So sánh LSTM vs Transformer B0 loss curves.

**Giải thích:**
- Phase chẩn đoán: Train vs Validation loss có converge không?
- Phát hiện overfitting/underfitting sớm để biết sweep nào cần thiết.

[Mở trực tiếp Cell 75 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y135sZmlsZQ%3D%3D) → tìm cell chứa `render_phase_summary`

---

## Giai Đoạn Sweeps (Cells 76-113)

Mỗi sweep phase gồm 1 markdown + 1 code cell. Code cell sẽ:
1. Load reference config từ phase trước.
2. Run sweep variants.
3. Pick winner bằng validation RMSE.
4. Update reference config.

### <a id="cell-76"></a>Cell 76-77 — Phase 23: S1 Feature-Set

**Cell 76:** Markdown tiêu đề `## Phase 23 - S1 Feature-Set Sweep`
**Cell 77:** `render_phase_resume(...)`

**Output kỳ vọng:** Bảng CSV `results.csv` với các variants FS0, FS1, FS2 và validation metrics. Winner = best validation RMSE.

**Giải thích:**
- Sweep 3 feature sets.
- Update reference → Phase 24 sẽ dùng winner.

[Mở trực tiếp Cell 77 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y140sZmlsZQ%3D%3D) → tìm cell có `Phase 23 - S1 Feature-Set Sweep`

---

### <a id="cell-78"></a>Cell 78-79 — Phase 24: S2 Time-Feature

**Cell 78:** Markdown tiêu đề
**Cell 79:** `render_phase_resume(...)`

**Output kỳ vọng:** Sweep time features (hour sin/cos, dayofweek, is_weekend, ...).

[Mở trực tiếp Cell 79 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y142sZmlsZQ%3D%3D)

---

### <a id="cell-80"></a>Cell 80-81 — Phase 25: S3 Target-Scaling

Sweep target scaling: StandardScaler, RobustScaler, MinMaxScaler, no-scaling.

[Mở trực tiếp Cell 81 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y144sZmlsZQ%3D%3D)

---

### <a id="cell-82"></a>Cell 82-83 — Phase 26: S4 Lookback

Sweep lookback ∈ {36, 72, 144}.

[Mở trực tiếp Cell 83 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y146sZmlsZQ%3D%3D)

---

### <a id="cell-84"></a>Cell 84-85 — Phase 27: S5 Pooling

Sweep pooling: last, mean, max, attention.

[Mở trực tiếp Cell 85 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y151sZmlsZQ%3D%3D)

---

### <a id="cell-86"></a>Cell 86-87 — Phase 28: S6 Activation

Sweep activation: ReLU, GELU, SiLU.

[Mở trực tiếp Cell 87 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y153sZmlsZQ%3D%3D)

---

### <a id="cell-88"></a>Cell 88-89 — Phase 29: S7 Batch-Size

Sweep batch_size ∈ {16, 32, 64, 128, 256}.

[Mở trực tiếp Cell 89 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y155sZmlsZQ%3D%3D)

---

### <a id="cell-90"></a>Cell 90-91 — Phase 30: S8 Learning-Rate

Sweep learning_rate ∈ {1e-4, 5e-4, 1e-3, 5e-3}.

[Mở trực tiếp Cell 91 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y160sZmlsZQ%3D%3D)

---

### <a id="cell-92"></a>Cell 92-93 — Phase 31: S9 Weight-Decay

**Cell 92:** `phase-31` ID
**Cell 93:** `render_phase_resume(...)`

Sweep weight_decay ∈ {0, 1e-5, 1e-4, 1e-3}.

[Mở trực tiếp Cell 93 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y162sZmlsZQ%3D%3D)

---

### <a id="cell-94"></a>Cell 94-95 — Phase 32: S10 Dropout

**Cell 94:** `phase-32` ID
**Cell 95:** `render_phase_resume(...)`

Sweep dropout ∈ {0.0, 0.1, 0.2, 0.3}.

[Mở trực tiếp Cell 95 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y164sZmlsZQ%3D%3D)

---

### <a id="cell-96"></a>Cell 96-97 — Phase 33: Transformer Configuration

**Cell 96:** Markdown tiêu đề `## Transformer Configuration after Phase 33`
**Cell 97:** `render_phase_33_transformer_configuration(...)`

**Output kỳ vọng:** Dashboard hiển thị toàn bộ Transformer config đã được update qua S1-S11 sweeps.

**Giải thích:**
- Snapshot Transformer config TRƯỚC khi bắt đầu architecture sweeps (S12-S19).
- Config này sẽ được dùng làm baseline cho S12-S19.

[Mở trực tiếp Cell 97 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y166sZmlsZQ%3D%3D)

---

### <a id="cell-98"></a>Cell 98-99 — Phase 34: S12 Head

**Cell 98:** `phase-34` ID
**Cell 99:** `build_phase_processing_log, render_phase_log(...)`

Sweep `num_heads ∈ {2, 4, 8, 16}`.

[Mở trực tiếp Cell 99 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y201sZmlsZQ%3D%3D)

---

### <a id="cell-100"></a>Cell 100-101 — Phase 35: S13 Layer

Sweep `num_layers ∈ {1, 2, 3, 4, 6}`.

[Mở trực tiếp Cell 101 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y203sZmlsZQ%3D%3D)

---

### <a id="cell-102"></a>Cell 102-103 — Phase 36: S14 FFN

Sweep FFN width multipliers ∈ {1, 2, 4}.

[Mở trực tiếp Cell 103 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y205sZmlsZQ%3D%3D)

---

### <a id="cell-104"></a>Cell 104-105 — Phase 37: S15 Loss

Sweep loss: MSE, Huber, Smooth L1.

[Mở trực tiếp Cell 105 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y210sZmlsZQ%3D%3D)

---

### <a id="cell-106"></a>Cell 106-107 — Phase 38: S16 Epoch-Cap

Sweep max_epochs ∈ {30, 50, 80, 100}.

[Mở trực tiếp Cell 107 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y212sZmlsZQ%3D%3D)

---

### <a id="cell-108"></a>Cell 108-109 — Phase 39: S17 Gradient-Clip

Sweep gradient_clip ∈ {0.5, 1.0, 5.0, no_clip}.

[Mở trực tiếp Cell 109 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y214sZmlsZQ%3D%3D)

---

### <a id="cell-110"></a>Cell 110-111 — Phase 40: S18 RevIN

Sweep RevIN on/off.

[Mở trực tiếp Cell 111 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y216sZmlsZQ%3D%3D)

---

### <a id="cell-112"></a>Cell 112-113 — Phase 41: S19 Boundary-Protocol

Test boundary handling.

[Mở trực tiếp Cell 113 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y221sZmlsZQ%3D%3D)

---

## Giai Đoạn Final Pipeline (Cells 114-125)

### <a id="cell-114"></a>Cell 114-115 — Phase 42: Candidate Synthesis

**Cell 114:** Markdown `## Phase 42 - Candidate Synthesis`
**Cell 115:** `materialize_phase_42(PROJECT_ROOT)`

**Output kỳ vọng:**
- Transformer candidate shortlist + selected lineage.
- File `artifacts/candidate_synthesis/{candidate_synthesis_manifest.json, transformer_candidate_shortlist.json, selected_lineage.json, baseline_anchor_context.json, boundary_sensitivity_context.json, candidate_synthesis_report.md}`.

**Giải thích:**
- Tổng hợp toàn bộ sweep winners.
- Chọn candidate để đưa vào rolling origin (Phase 44).

[Mở trực tiếp Cell 115 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y223sZmlsZQ%3D%3D) → tìm cell chứa `phase_42_signoff = materialize_phase_42`

---

### <a id="cell-116"></a>Cell 116-117 — Phase 43: LSTM Tuning

**Cell 116:** Markdown `## Phase 43 - LSTM Tuning`
**Cell 117:** `render_phase_43_dashboard(...)`

**Output kỳ vọng:** Dashboard LSTM tuning stages (lt1-lt5).

**Giải thích:**
- Tune LSTM hyperparameters theo stages.
- Mỗi stage sweep 1 hyperparameter.

[Mở trực tiếp Cell 117 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y225sZmlsZQ%3D%3D) → tìm cell chứa `render_phase_43_dashboard`

---

### <a id="cell-118"></a>Cell 118–119 — Phase 44: Rolling-Origin

**Cell 118:** Markdown
**Cell 119:** `render_phase_44_dashboard(...)`

**Output kỳ vọng:** Dashboard rolling origin folds (5 folds × candidates).

**Giải thích:**
- Robustness evaluation qua time slices.
- Robust Lane: full refit mỗi fold.

[Mở trực tiếp Cell 119 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y230sZmlsZQ%3D%3D) → tìm cell chứa `render_phase_44_dashboard`

---

### <a id="cell-120"></a>Cell 120-121 — Phase 45: Final Model Lock

**Cell 120:** Markdown `## Phase 45 - Final Model Lock`
**Cell 121:** `render_phase_45_dashboard(...)`

**Output kỳ vọng:** Dashboard locked config (architecture, optimizer, loss, scaler, ...).

**Giải thích:**
- **NO-TRAIN phase** — chỉ lock config.
- Sau khi lock, không được thay đổi gì.

[Mở trực tiếp Cell 121 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y232sZmlsZQ%3D%3D) → tìm cell chứa `render_phase_45_dashboard`

---

### <a id="cell-122"></a>Cell 122-123 — Phase 46: Three-Seed Final Runs

**Cell 122:** Markdown
**Cell 123:** `render_phase_46_dashboard(...)`

**Output kỳ vọng:** Dashboard 3-seed runs (42, 123, 2026).

**Giải thích:**
- Train final model với 3 seeds.
- Mỗi seed tạo FINAL_REFIT checkpoint.

[Mở trực tiếp Cell 123 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y234sZmlsZQ%3D%3D) → tìm cell chứa `render_phase_46_dashboard`

---

### <a id="cell-124"></a>Cell 124-125 — Phase 47: Final Test Evaluation

**Cell 124:** Markdown `## Phase 47 - Final Test Evaluation`
**Cell 125:** `render_phase_47_dashboard(...)`

**Output kỳ vọng:** Dashboard FINAL TEST metrics (RMSE, MAE, R² cho 3 seeds + mean ± std).

**Giải thích:**
- **FINAL GATE** — chỉ chạy 1 lần trên Test set.
- Kết quả Phase 47 là metrics chính thức của project.

[Mở trực tiếp Cell 125 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y236sZmlsZQ%3D%3D) → tìm cell chứa `render_phase_47_dashboard`

---

## Phần Bổ Sung Metric (Cell 126)

### <a id="cell-126"></a>Cell 126 — Supplementary MAPE Metric Addendum

**Cell 126 (code) ID:** `ac4fe7dd`
**Source token:** `render_mape_addendum`
**Execution count:** `1`
**Output count:** `1`

**Output:** Dashboard MAPE cho các kết quả validation và test đã được lưu.

**Ý nghĩa:**
- Đây là code cell bổ sung metric, không có markdown heading riêng trong notebook hiện tại.
- MAPE không thay thế MAE, RMSE hoặc R².
- Khi target nhỏ, MAPE có thể tăng mạnh nên phải đọc cùng các metric tuyệt đối.

[Mở trực tiếp Cell 126 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y240sZmlsZQ%3D%3D)

---

## Giai Đoạn Analysis (Cells 127–150)

### <a id="cell-127"></a>Cell 127–128 — Phase 48: Prediction Analysis

**Cell 127 (markdown) ID:** `b496ac6f`
**Cell 128 (code) ID:** `1451ccf0`
**Source token:** `render_phase_48_dashboard`
**Output count:** `1`

**Output:** Phân tích actual và predicted, ECDF, change magnitudes và lag cross-correlation.

**Ý nghĩa:** Phát hiện systematic bias và các vùng dự đoán khó.

[Mở trực tiếp Cell 128 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y242sZmlsZQ%3D%3D)

---

### <a id="cell-129"></a>Cell 129–130 — Phase 49: Residual Analysis

**Cell 129 (markdown) ID:** `7cc29214`
**Cell 130 (code) ID:** `bbcf08f2`
**Source token:** `render_phase_49_dashboard`
**Output count:** `1`

**Output:** Phân tích residual distribution, signed bias, ACF, sign runs và persistence context.

**Ý nghĩa:** Kiểm tra bias và autocorrelation còn lại trong sai số.

[Mở trực tiếp Cell 130 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y244sZmlsZQ%3D%3D)

---

### <a id="cell-131"></a>Cell 131–132 — Phase 50: Error-by-Regime Analysis

**Cell 131 (markdown) ID:** `9a2f1b89`
**Cell 132 (code) ID:** `1d5b5327`
**Source token:** `render_phase_50_dashboard`
**Output count:** `1`

**Output:** Tổng hợp metric theo target level, time of day, day type, extreme-high regime và change direction.

**Ý nghĩa:** Tách hiệu năng tổng thể khỏi hiệu năng theo từng điều kiện dữ liệu.

[Mở trực tiếp Cell 132 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y246sZmlsZQ%3D%3D)

---

### <a id="cell-133"></a>Cell 133–134 — Phase 51: Worst-Error Analysis

**Cell 133 (markdown) ID:** `2e8f083e`
**Cell 134 (code) ID:** `c9df346a`
**Source token:** `render_phase_51_dashboard`
**Output count:** `1`

**Output:** Hiển thị top-K worst errors theo seed, shared worst cases và casebook.

**Ý nghĩa:** Giữ context của các sample khó nhất để phân tích định tính.

[Mở trực tiếp Cell 134 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y251sZmlsZQ%3D%3D)

---

### <a id="cell-135"></a>Cell 135–136 — Phase 52: Attention Extraction

**Cell 135 (markdown) ID:** `870efbe4`
**Cell 136 (code) ID:** `7b48186f`
**Source token:** `render_phase_52_dashboard`
**Output count:** `1`

**Output:** Hiển thị extraction state và checksum của raw attention artifacts.

**Ý nghĩa:** Raw attention được materialize để các phase sau chỉ đọc lại.

[Mở trực tiếp Cell 136 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y253sZmlsZQ%3D%3D)

---

### <a id="cell-137"></a>Cell 137–138 — Phase 53: Attention Heatmaps

**Cell 137 (markdown) ID:** `afeb48a2`
**Cell 138 (code) ID:** `1c098fa6`
**Source token:** `render_phase_53_dashboard`
**Output count:** `1`

**Output:** Hiển thị heatmap theo case, seed, layer và head.

**Ý nghĩa:** Cho phép so sánh attention distribution giữa các trường hợp và checkpoint.

[Mở trực tiếp Cell 138 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y255sZmlsZQ%3D%3D)

---

### <a id="cell-139"></a>Cell 139–140 — Phase 54: Last-Query Attention Analysis

**Cell 139 (markdown) ID:** `3c915fcc`
**Cell 140 (code) ID:** `ec7b2ea5`
**Source token:** `render_phase_54_dashboard`
**Output count:** `1`

**Output:** Hiển thị entropy, expected lag, top-one frequency và coverage.

**Ý nghĩa:** Định lượng khoảng quá khứ mà query cuối ưu tiên.

[Mở trực tiếp Cell 140 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y260sZmlsZQ%3D%3D)

---

### <a id="cell-141"></a>Cell 141–142 — Phase 55: Head Comparison

**Cell 141 (markdown) ID:** `644dbb41`
**Cell 142 (code) ID:** `05ec07a2`
**Source token:** `render_phase_55_dashboard`
**Output count:** `1`

**Output:** So sánh JSD, cosine và Wasserstein giữa attention heads.

**Ý nghĩa:** Xác định head chuyên biệt hoặc dư thừa.

[Mở trực tiếp Cell 142 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y262sZmlsZQ%3D%3D)

---

### <a id="cell-143"></a>Cell 143–144 — Phase 56: Error-Conditioned Attention

**Cell 143 (markdown) ID:** `de13c9d0`
**Cell 144 (code) ID:** `5e9eec01`
**Source token:** `render_phase_56_dashboard`
**Output count:** `1`

**Output:** So sánh attention giữa high-error và low-error samples cùng decile trends.

**Ý nghĩa:** Liên kết interpretability với model error mà không suy diễn causal claim.

[Mở trực tiếp Cell 144 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y264sZmlsZQ%3D%3D)

---

### <a id="cell-145"></a>Cell 145–146 — Phase 57: Seed-Stability Attention Check

**Cell 145 (markdown) ID:** `5627f983`
**Cell 146 (code) ID:** `4293c1aa`
**Source token:** `render_phase_57_dashboard`
**Output count:** `1`

**Output:** Hiển thị cross-seed attention stability và canonical head mapping.

**Ý nghĩa:** Đánh giá attention có ổn định giữa các seed hay không.

[Mở trực tiếp Cell 146 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y266sZmlsZQ%3D%3D)

---

### <a id="cell-147"></a>Cell 147–148 — Phase 58: Final Results Summary

**Cell 147 (markdown) ID:** `332dd9c8`
**Cell 148 (code) ID:** `9f864b91`
**Source token:** `render_phase_58_dashboard`
**Output count:** `1`

**Output:** Tổng hợp final tables và analysis tables đã materialize.

**Ý nghĩa:** Cung cấp representation phục vụ báo cáo và kiểm tra provenance.

[Mở trực tiếp Cell 148 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y301sZmlsZQ%3D%3D)

---

### <a id="cell-149"></a>Cell 149–150 — Phase 59: Final Conclusions

**Cell 149 (markdown) ID:** `251da856`
**Cell 150 (code) ID:** `76de7c69`
**Source token:** `render_phase_59_dashboard`
**Output count:** `1`

**Output:** Tổng hợp kết luận, research-question answers, limitations và future work.

**Ý nghĩa:** Cell 150 là output-producing code cell cuối cùng của notebook hiện tại.

[Mở trực tiếp Cell 150 trong notebook](vscode-notebook-cell:/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb#Y303sZmlsZQ%3D%3D)

---

## Tài Liệu Liên Quan

| File | Mô tả |
|---|---|
| [`CURRENT_FLOW_SUMMARY.md`](../current_flow/CURRENT_FLOW_SUMMARY.md) | Tổng quan kiến trúc |
| [`PHASE_1_TO_33_CURRENT_FLOW_DETAIL.md`](../current_flow/PHASE_1_TO_33_CURRENT_FLOW_DETAIL.md) | Chi tiết phases 1-33 |
| [`PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md`](../current_flow/PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md) | Chi tiết phases 34-59 |

---

## Cách Sử Dụng File Này

1. **Khi đọc notebook:** Mở file này song song, mỗi cell gặp output khó hiểu thì tra cứu cell tương ứng.
2. **Khi debug:** Tìm cell ID của phase đang lỗi, đọc phần giải thích để biết output kỳ vọng.
3. **Khi present:** Click vào liên kết `Cell ...` để mở đúng notebook cell; dùng liên kết `Giải thích` để quay lại walkthrough.

**Phiên bản:** 06/09/2026 — sync với notebook sau lần refactor clean architecture.
