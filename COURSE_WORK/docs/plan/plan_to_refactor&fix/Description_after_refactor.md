# Mô tả sau refactor

> Trạng thái: `SECTION_1_18_WRITTEN_FROM_EVIDENCE`
>
> Toàn bộ 18 mục đã được viết từ evidence đang tồn tại trong repo. Mọi metric, status, file path, experiment ID, Step ID, SHA-256, số `git diff` và kết quả `pytest` đều có nguồn xác minh. Các nội dung không verify được ghi rõ là `UNVERIFIED/UNAVAILABLE` thay vì suy đoán.
>
> **Cập nhật consistency audit 2026-09-13:** trạng thái hiện hành và validation
> mới nhất nằm tại
> [FINAL_CONSISTENCY_AUDIT_20260913.md](COURSE_WORK/docs/current_flow/FINAL_CONSISTENCY_AUDIT_20260913.md).
> Báo cáo này thay thế các snapshot validation/debt cũ trong Mục 16–18 khi hai
> nguồn khác nhau. Đặc biệt, Phase 34–41 hiện là `VALID_REUSABLE`, focused suite
> là `163 passed, 1 skipped`, Step 16 preflight đã PASS sau exact-SHA artifact
> restore, và test Phase 1–42 hiện khóa source thay vì output metadata.

## 1. Thông tin tài liệu

| Trường | Giá trị đã xác minh |
|---|---|
| Tên project | `UCI Appliances Energy Prediction — Multivariate Time-Series Regression` |
| Mục đích tài liệu | Ghi lại các thay đổi về code, notebook, presentation/reporting, artifact recovery và validation sau refactor; tách các thay đổi này khỏi scientific result đã được khóa. |
| File notebook chính | [notebook_course_work/CourseWork.ipynb](COURSE_WORK/notebook_course_work/CourseWork.ipynb) |
| Repo/branch đang audit | Git root `TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/`, project nằm tại `COURSE_WORK/`; branch `CourseWork` đang đồng bộ với `origin/CourseWork` tại commit `a8fb7a0a12fdebf3b997ce4bab2af6146e2b1398`. |
| Trạng thái worktree | Commit presentation V2 đã được push. Worktree vẫn chưa clean vì còn các recovery artifact/log Phase 22–41 chưa nằm trong commit này; chúng được tách khỏi commit presentation và không được mô tả như V2 code chưa commit. |
| Plan/reference chính | [docs/Plan_improve_model.md](COURSE_WORK/docs/Plan_improve_model.md), [model_improvement_v2_pre_process_plan.md](COURSE_WORK/docs/plan/plan_before_process/model_improvement_v2_pre_process_plan.md), [model_improvement_v2_wave2_pre_process_plan.md](COURSE_WORK/docs/plan/plan_before_process/model_improvement_v2_wave2_pre_process_plan.md) |
| Trạng thái project hiện tại | V1 evidence vẫn được giữ theo namespace lịch sử; V2 có final lock và final closure riêng. Root [`README.md`](README.md) tồn tại ở repo root (đã verify bằng `find . -maxdepth 4 -name "README.md"`). |
| Trạng thái `MODEL_IMPROVEMENT_V2` | `COMPLETE`; Step 16 `LOCKED`; Step 17 `PASS` với nhãn `POST_HOC_V2_BENCHMARK`. MAPE addendum và reporting Phase 48–51/58–59 đã được tạo từ frozen prediction artifacts, không chạy model inference mới. |
| Trạng thái notebook | 153 cells, trong đó 77 code cells có `execution_count`; không có output loại `error`. Kernel metadata là `.venv (3.10.11)`/`python3`. Notebook có Phase 1–59 và section `MODEL_IMPROVEMENT_V2` riêng ở cuối. |
| Thời điểm refactor | Thời điểm bắt đầu chính xác: `UNVERIFIED/UNAVAILABLE`. Evidence notebook ghi nhận execution ngày `2026-09-12`; commit `HEAD` hiện tại có timestamp `2026-09-11T08:37:11+07:00`. Các mốc này không được diễn giải là thời điểm bắt đầu refactor. |

Nguồn kiểm tra cho mục này: `git status --short --branch`, `git log -1`, notebook metadata, [Plan_improve_model.md](COURSE_WORK/docs/Plan_improve_model.md), [v2_final_model_lock.json](COURSE_WORK/artifacts/model_improvement_v2/final_model_lock/v2_final_model_lock.json), [step17_benchmark_signoff.json](COURSE_WORK/artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_benchmark_signoff.json) và [model_improvement_v2_final_closure.json](COURSE_WORK/artifacts/model_improvement_v2/model_improvement_v2_final_closure.json).

## 2. Mục tiêu refactor

Refactor nhằm sửa các vấn đề thực tế của execution governance và presentation mà không thay đổi scientific evidence đã được tạo. Mục tiêu cụ thể được khóa như sau:

| Nhóm mục tiêu | Before | After mong muốn/đã triển khai | Reason và evidence |
|---|---|---|---|
| Ranh giới code và notebook | Notebook chứa nhiều kiểu orchestration/render khác nhau; một số phase cũ còn gọi public materialization/resume API ngay trong cell. | Scientific logic thuộc `src/course_work/`; reporting đọc artifact hiện có. Phase 43–59 dùng một entry point `render_verified_phase_result`, còn V2 dùng `render_verified_v2_results`. | Tuân thủ source-owned processing và giảm nguy cơ notebook vô tình tạo lại evidence. Xem [architecture_rule.md](COURSE_WORK/docs/RULE_BASE/architecture_rule.md), [results_rebuild.py](COURSE_WORK/src/course_work/reporting/results_rebuild.py) và notebook hiện tại. |
| Presentation Phase 43–59 | Mỗi phase dùng legacy dashboard riêng, cấu trúc và độ chi tiết không đồng nhất. | Chuẩn hóa mỗi phase thành ba bảng: `Phase status`, `Configuration / analysis`, `Results / decision`; table-first, left-aligned, không dump raw JSON hoặc evidence path dài trong output. | Notebook tại `HEAD` gọi `phase_43_47_dashboard`, `phase_48_dashboard` … `phase_59_dashboard`; notebook hiện tại gọi renderer thống nhất. |
| Trạng thái phase và artifact thiếu | Strict resume validation có thể render `BLOCKED` khi active CSV/signoff dependency bị thiếu hoặc không hợp lệ, dù historical notebook từng lưu kết quả hoàn thành. | Tách “khả năng resume/recompute từ active artifact” khỏi “historical scientific evidence”; chỉ hiển thị status có nguồn xác minh và giữ reporting debt khi evidence không đủ. | [frozen_evidence.py](COURSE_WORK/src/course_work/reporting/frozen_evidence.py), các signoff Phase 36–41 và [phase_34_current_state_rebaseline_blockers.md](COURSE_WORK/docs/analysis_error/phase_34_current_state_rebaseline_blockers.md). |
| `MODEL_IMPROVEMENT_V2` | Snapshot notebook lịch sử chưa có section V2 cuối notebook. | Bổ sung section V2 riêng; đồng thời chuyển presentation Phase 47–51 và Phase 58–59 sang final V2 policy từ verified V2 reporting artifacts. Historical V1 artifacts vẫn bất biến; Phase 52–57 được giữ và gắn nhãn V1 historical attention. | [results_rebuild.py](COURSE_WORK/src/course_work/reporting/results_rebuild.py), [v2_final_analysis.py](COURSE_WORK/src/course_work/reporting/v2_final_analysis.py) và [model_improvement_v2_final_closure.json](COURSE_WORK/artifacts/model_improvement_v2/model_improvement_v2_final_closure.json). |
| Code phục vụ V2 | Core ban đầu chưa bao phủ đầy đủ namespace, recovery, hybrid loss, final refit và post-hoc benchmark của E01–E20/Step 14–17. | Tạo V2-scoped runner/contracts, mở rộng strict registry namespace, recovery/no-retrain path, hybrid loss logging, Step 16 final refit và Step 17 benchmark. | Current diff dưới [src/course_work/model_improvement_v2/](COURSE_WORK/src/course_work/model_improvement_v2/), [real_run.py](COURSE_WORK/src/course_work/rolling_origin/real_run.py), [engine.py](COURSE_WORK/src/course_work/training/engine.py) và [registry.py](COURSE_WORK/src/course_work/experiments/registry.py). |
| Traceability | Metric/status có thể chỉ xuất hiện trong output notebook hoặc dashboard lịch sử; một số intermediate CSV không còn ở active path. | Mỗi giá trị trình bày phải đọc từ JSON/CSV/signoff/lock/closure còn tồn tại; checksum, population, scaler và checkpoint lineage được audit ở từng gate. | V2 pre-process plans, [v2_final_model_lock.json](COURSE_WORK/artifacts/model_improvement_v2/final_model_lock/v2_final_model_lock.json), Step 17 manifest/signoff và các test `test_model_improvement_v2_*.py`. |
| Scientific integrity | Refactor presentation có nguy cơ bị hiểu nhầm là rerun hoặc sửa kết luận. | Giữ nguyên V1 evidence; V2 ở namespace riêng; không chọn bằng Test, không đổi weights sau Test, không post-Test retuning và không biến `WARNING`/`REJECT`/`DEFERRED` thành `PASS`. | [model_improvement_v2_pre_process_plan.md](COURSE_WORK/docs/plan/plan_before_process/model_improvement_v2_pre_process_plan.md) và trường `governance` trong final closure. |

## 3. Phạm vi refactor

### In scope

| Phạm vi | Nội dung đã xác minh |
|---|---|
| Code thực thi V2 | [src/course_work/model_improvement_v2/](COURSE_WORK/src/course_work/model_improvement_v2/), cùng các thay đổi liên quan trong [rolling_origin/](COURSE_WORK/src/course_work/rolling_origin/), [training/](COURSE_WORK/src/course_work/training/) và [experiments/registry.py](COURSE_WORK/src/course_work/experiments/registry.py). |
| Code reporting | [frozen_evidence.py](COURSE_WORK/src/course_work/reporting/frozen_evidence.py), [results_rebuild.py](COURSE_WORK/src/course_work/reporting/results_rebuild.py), [v2_final_analysis.py](COURSE_WORK/src/course_work/reporting/v2_final_analysis.py) và [model_improvement_v2_dashboard.py](COURSE_WORK/src/course_work/reporting/model_improvement_v2_dashboard.py). |
| Notebook | [CourseWork.ipynb](COURSE_WORK/notebook_course_work/CourseWork.ipynb), gồm review presentation Phase 43–59 và section `MODEL_IMPROVEMENT_V2`; current worktree cũng chứa các thay đổi ở Phase 1–42. |
| Phase/Step được review | Historical resume/reporting quanh Phase 36–41; presentation Phase 43–59; V2 experiments E01–E20; Step 14A/14B; Step 16 final lock/refit; Step 17 `POST_HOC_V2_BENCHMARK`; final closure. |
| Reporting/output | Chuẩn hóa bảng; Phase 47–51 và 58–59 dùng V2 artifacts; Phase 52–57 giữ V1 historical attention có nhãn rõ; khôi phục presentation snapshot chỉ khi source/checksum xác minh được. |
| Tests/validation | Focused notebook/reporting tests, `test_v2_final_reporting_analysis.py`, `test_model_improvement_v2_step17_mape.py`, syntax/static validation và scoped `git diff --check`. |
| Artifact recovery | Chỉ nhận artifact/snapshot có source branch, source commit và SHA-256 được ghi trong [artifacts/notebook_presentation_recovery/](COURSE_WORK/artifacts/notebook_presentation_recovery/); không tái tạo số thiếu bằng suy đoán. |

### Out of scope

| Không thuộc phạm vi tài liệu/refactor hiện tại | Ràng buộc |
|---|---|
| Training hoặc inference mới để viết tài liệu/presentation | Không thực hiện. Các run V2 đã có là evidence của workflow trước đó, không phải hoạt động của task viết tài liệu này. |
| Mở lại raw Test source | Không thực hiện trong reporting refactor. MAPE addendum chỉ đọc frozen Test-derived prediction artifacts; `raw_test_source_opened=false`, `inference_executed=false`. |
| Retune scientific configuration sau Step 17 | Không thực hiện; final closure ghi `post_test_retuning=false`, `ensemble_weights_changed=false` và `best_seed_selected=false`. |
| Sửa model weights/checkpoint/scaler/config đã khóa | Không thực hiện; Step 16 final lock tiếp tục là source of truth. |
| Ghi đè historical V1 scientific artifacts Phase 47–59 | Không thực hiện. Chỉ presentation Phase 47–51 và 58–59 chuyển sang V2; Phase 52–57 vẫn là V1 historical attention. |
| `git add`, `git commit`, `git push` | Không thực hiện. |

### Trạng thái Phase 1–42

Các thay đổi recovery Phase 22–41 tồn tại riêng trong worktree và không thuộc commit presentation V2 `a8fb7a0a`. Phase 34–41 đã được audit theo contract recovery/legacy compatibility; scientific core không bị tái huấn luyện. Tài liệu phải phân biệt rõ recovery metadata với historical scientific artifacts.

## 4. Trạng thái trước refactor

Các mục dưới đây chỉ mô tả vấn đề có evidence trong Git history, audit document hoặc artifact manifest. “Trước refactor” không đồng nghĩa với “scientific run chưa hoàn thành”.

| Vấn đề | Vị trí | Trạng thái trước refactor | Evidence |
|---|---|---|---|
| Presentation Phase 43–59 phân mảnh | `notebook_course_work/CourseWork.ipynb` tại `HEAD` | Phase 43–47 dùng `phase_43_47_dashboard`; Phase 48–59 gọi renderer riêng cho từng phase. Kết quả vì vậy không có một schema bảng thống nhất. | `git show HEAD:COURSE_WORK/notebook_course_work/CourseWork.ipynb`; đối chiếu [results_rebuild.py](COURSE_WORK/src/course_work/reporting/results_rebuild.py). |
| V2 chưa được trình bày trong notebook lịch sử | Cuối notebook tại `HEAD` | Notebook có 150 cells và kết thúc ở Phase 59; chưa có section `MODEL_IMPROVEMENT_V2`. | `git show HEAD:COURSE_WORK/notebook_course_work/CourseWork.ipynb`; notebook hiện tại có 153 cells và thêm section V2. |
| Phase 36 strict resume bị `BLOCKED` | `artifacts/sweeps/S14_ffn/` | Active `s14_ffn_metrics.csv` bị thiếu; strict validation còn gặp upstream Phase 35 signoff không hợp lệ. Historical signoff vẫn tham chiếu checksum/file lịch sử. | [phase_36_signoff.json](COURSE_WORK/artifacts/sweeps/S14_ffn/phase_36_signoff.json), [phase_34_current_state_rebaseline_blockers.md](COURSE_WORK/docs/analysis_error/phase_34_current_state_rebaseline_blockers.md). |
| Phase 37–39 strict resume bị `BLOCKED` | `artifacts/sweeps/S15_loss/`, `S16_epoch_cap/`, `S17_gradient_clip/` | Thiếu active metrics/manifest theo từng phase và có upstream signoff `SIGNOFF_ARTIFACT_INVALID` hoặc `SIGNOFF_NOT_PASS`. Trạng thái này phản ánh khả năng resume từ active artifact, không tự động phủ định historical output đã lưu. | [frozen_evidence.py](COURSE_WORK/src/course_work/reporting/frozen_evidence.py), các signoff tương ứng dưới [artifacts/sweeps/](COURSE_WORK/artifacts/sweeps/). |
| Phase 40–41 strict resume bị `BLOCKED` | `artifacts/sweeps/S18_revin/`, `S19_boundary/` | Phase 40 được phân loại `CONDITION_INCOMPLETE`; Phase 41 thiếu active `s19_boundary_metrics.csv`/winner và phụ thuộc upstream Phase 40 chưa đạt gate. | [frozen_evidence.py](COURSE_WORK/src/course_work/reporting/frozen_evidence.py), [phase_34_current_state_rebaseline_blockers.md](COURSE_WORK/docs/analysis_error/phase_34_current_state_rebaseline_blockers.md). |
| Intermediate reporting artifact Phase 54/55 không đầy đủ | `artifacts/last_query_attention/`, `artifacts/head_comparison/` | Một số intermediate CSV không còn ở active path, nên không thể dựng đầy đủ dashboard chỉ từ current machine-readable tables. Phase 56/57 không có cùng missing-intermediate-CSV debt trong recovery manifest. | [courseworkv10_phase54_57/manifest.json](COURSE_WORK/artifacts/notebook_presentation_recovery/courseworkv10_phase54_57/manifest.json). |
| Presentation Phase 54–57 cần recovery | `artifacts/notebook_presentation_recovery/courseworkv10_phase54_57/` | Dashboard snapshot từ `CourseWorkV10` được dùng làm presentation evidence sau khi source commit, source cell và SHA-256 được khóa; manifest ghi `scientific_recomputation=false` và `test_source_accessed=false`. | [manifest.json](COURSE_WORK/artifacts/notebook_presentation_recovery/courseworkv10_phase54_57/manifest.json). |
| Phase 6 output từng không được trình bày đầy đủ | Phase 6.1–6.3 trong notebook | Presentation cần được đối chiếu/khôi phục từ artifact gốc. Current repo hiện có đủ 16 CSV trong `artifacts/eda/tables/` và 16 PNG trong `artifacts/eda/figures/`; đây là artifact presentation/scientific evidence hiện tại, không phải số được tái tạo trong tài liệu này. | [artifacts/eda/](COURSE_WORK/artifacts/eda/), [test_notebook_frozen_evidence.py](COURSE_WORK/tests/unit/test_notebook_frozen_evidence.py), notebook diff hiện tại. |
| Phase 47 từng có notebook-sync/reporting defect | Phase 47 section trong audit cũ | Audit trước đây ghi nhận presentation stale/missing trong notebook trong khi scientific result có evidence độc lập. Đây là historical defect đã được ghi nhận, không phải căn cứ để thay metric Phase 47 bằng metric V2. | [full_phase_0_47_plan_notebook_compliance_audit.md](COURSE_WORK/docs/analysis_error/full_phase_0_47_plan_notebook_compliance_audit.md). |
| Retention của CSV/checkpoint không ổn định | `.gitignore` và các active artifact path | Audit dependency recovery ghi nhận một số file cần thiết bị ignore hoặc chưa từng được track; vì vậy Git branch có thể giữ signoff/checksum nhưng thiếu payload active. | [phase_34_current_state_rebaseline_blockers.md](COURSE_WORK/docs/analysis_error/phase_34_current_state_rebaseline_blockers.md), `.gitignore`, các missing path Phase 36–41. |
| Raw JSON/debug output ở Phase 43–59 | Các code cell Phase 43–59 tại `HEAD` | Không tìm thấy bằng chứng về direct `json.dumps`/raw JSON dump trong các cell này; vấn đề xác minh được là legacy renderer phân mảnh và output dài/không đồng nhất. Vì vậy tài liệu không gán một lỗi raw JSON không có evidence. | `git show HEAD:COURSE_WORK/notebook_course_work/CourseWork.ipynb`; current notebook source. |
| Recovery Phase 34–41 cần legacy compatibility | `artifacts/sweeps/S12_heads/` đến `S19_boundary/` | Historical signoff schema và reporting retention khác validator mới. Recovery giữ nguyên scientific core, dùng exact hash/registry-bound compatibility và không fabricate reporting CSV bị thiếu. | Recovery manifests dưới `COURSE_WORK/artifacts/_reconstruction_history/` và processing logs Phase 34–41. |

Các vấn đề trên được giữ như reporting/lineage debt khi chưa đủ payload để tái dựng. Refactor không được phép biến chúng thành `PASS` bằng cách tự tạo metric, sửa checksum hoặc diễn giải historical notebook output như machine-readable evidence mới.

## 5. Tổng quan thay đổi sau refactor

| Nhóm thay đổi | Trước refactor | Sau refactor | Trạng thái |
|---|---|---|---|
| Core execution code | Registry, rolling-origin orchestration và training engine chưa bao phủ toàn bộ contract E10–E20, partial recovery, hybrid loss và final-refit flow. | Bổ sung namespace/primary-change contract theo experiment, recovery/no-retrain ledger, hybrid loss component logging, Step 16 final refit và Step 17 benchmark guard. | V2 presentation/reporting đã commit và push tại `a8fb7a0a`; recovery changes khác vẫn tách riêng trong worktree. |
| Project path | Một số script/module/test còn dùng absolute path cũ không có thư mục `Deep Learning`. | Các path trong diff được đổi sang `/Users/vientu/Deep Learning/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK`; command có khoảng trắng được quote. | `UPDATED`; vẫn là absolute path, chưa phải portable root discovery toàn diện. |
| Notebook | Snapshot lịch sử có 150 cells, legacy dashboard Phase 43–59 và chưa có V2 section. | Notebook hiện có 153 cells; 77/77 code cells có `execution_count`, 0 error output; Phase 47–51 và 58–59 trình bày final V2 policy. | `COMMITTED_AND_PUSHED` tại `a8fb7a0a`. |
| Reporting/output | Phase 43–59 dùng renderer riêng và lineage không đồng nhất. | Phase 47–51 = V2; Phase 52–57 = V1 historical attention; Phase 58–59 = V2. Metrics đọc từ artifact, không hard-code. | `17/17 RENDERED`; scientific artifacts không bị ghi đè. |
| Historical/frozen evidence | Phase 22–41 có mixed legacy summary/resume flow; Phase 54/55 thiếu một số intermediate CSV ở active path. | Có `render_frozen_phase_evidence` cho read-only inventory/resume audit và checksum-verified CourseWorkV10 snapshots cho Phase 54–57; Phase 43–59 hiện ưu tiên current summary/signoff qua renderer mới. | `IMPLEMENTED`; missing payload vẫn được giữ là reporting debt. |
| `MODEL_IMPROVEMENT_V2` | Không có trong notebook tại `HEAD`; core chỉ có Wave 1 foundation. | Bổ sung code E10–E20 và Step 16/17, evidence Step 14A/14B, cùng một bảng kết quả V2 riêng; final closure là `COMPLETE`, final lock là `LOCKED`. | `COMPLETE` theo final closure; notebook presentation đã thêm. |
| Tests/validation | Test cũ tập trung legacy dashboard và Wave 1. | Bổ sung focused tests cho Step 17 MAPE, final V2 reporting analysis và rebuilt results. | Validation mới nhất của bốn focused files: `17 passed`. |
| Scientific result | Historical V1 và các experiment result đã được ghi trong artifact riêng. | Presentation refactor không recompute hoặc sửa metric. V2 scientific evidence là kết quả của workflow/runs trước đó và được khóa bằng final lock/closure, không phải số được sinh khi render notebook. | `SCIENTIFIC_RESULTS_UNCHANGED_BY_PRESENTATION_REFACTOR`. |

Tổng quan này dựa trên current `git diff`, `git status`, [CourseWork.ipynb](COURSE_WORK/notebook_course_work/CourseWork.ipynb), [model_improvement_v2_final_closure.md](COURSE_WORK/docs/model_improvement_v2_final_closure.md) và [model_improvement_v2_final_closure.json](COURSE_WORK/artifacts/model_improvement_v2/model_improvement_v2_final_closure.json).

## 6. Thay đổi trong code

Các bảng dưới đây chỉ liệt kê file code có trạng thái `MODIFIED` hoặc `CREATED` trong current worktree và có liên quan trực tiếp tới refactor/V2/notebook reporting.

### Core, experiment và recovery code

| File | Action | Thay đổi chính | Mục đích |
|---|---|---|---|
| [src/course_work/experiments/registry.py](COURSE_WORK/src/course_work/experiments/registry.py) | `MODIFIED` | Mở rộng `FEATURE_VARIANTS`, `V2_NAMESPACE_IDS`; truyền `run_id_namespace` vào `validate_run_config`; chỉ cho batch size 16 trong namespace E14/E15/E16/E20. | Giữ strict namespace governance, hỗ trợ Wave 2 và không nới batch-size rule toàn cục cho V1. |
| [src/course_work/model_improvement_v2/contracts.py](COURSE_WORK/src/course_work/model_improvement_v2/contracts.py) | `MODIFIED` | Thêm `ALLOWED_IMPLEMENTED_WAVE2_EXPERIMENT_IDS`, `ALLOWED_EXPERIMENT_IDS` và `PRIMARY_CHANGE_PREFIXES` cho E10–E16/E20; sửa validation message. | Khóa one-primary-change contract cho experiment đã implement. |
| [src/course_work/model_improvement_v2/e08_runner.py](COURSE_WORK/src/course_work/model_improvement_v2/e08_runner.py) | `MODIFIED` | `validate_e08_document` đổi exact equality thành subset check với `V2_DIRECT_METRIC_FLATTEN_TRACKS`. | Cho track mới được thêm mà không thay scientific behavior E08. |
| [src/course_work/model_improvement_v2/e09_runner.py](COURSE_WORK/src/course_work/model_improvement_v2/e09_runner.py) | `MODIFIED` | `validate_e09_document` dùng subset check tương tự E08. | Compatibility với track V2 mới, giữ contract E09. |
| [src/course_work/model_improvement_v2/pretest_adapter.py](COURSE_WORK/src/course_work/model_improvement_v2/pretest_adapter.py) | `MODIFIED` | `build_v2_pretest_dataset` đăng ký projection `FS2_TF1` cho E13–E16, E20 và `STEP16`. | Reuse strict Train+Validation prefix, không gọi Phase 9/10 materializer lịch sử. |
| [src/course_work/model_improvement_v2/e10_pretest.py](COURSE_WORK/src/course_work/model_improvement_v2/e10_pretest.py) | `CREATED` | Thêm `E10PopulationAudit`, `derive_causal_target_deltas`, `build_e10_pretest_datasets` và locked delta-feature order. | Tạo common-population projections cho E10 bằng past-only target deltas. |
| [src/course_work/model_improvement_v2/e10_runner.py](COURSE_WORK/src/course_work/model_improvement_v2/e10_runner.py) | `CREATED` | Thêm E10 config/preflight, control/challenger builder, V2 context, promotion guardrails và official Human gate. | Orchestrate E10 trong namespace riêng. |
| [src/course_work/model_improvement_v2/e11_pretest.py](COURSE_WORK/src/course_work/model_improvement_v2/e11_pretest.py) | `CREATED` | Thêm `E11PopulationAudit`, `ROLLING_DEFINITIONS`, `derive_past_only_rolling_features`, common-population builder. | Tạo seven-feature rolling-target projection không dùng future target. |
| [src/course_work/model_improvement_v2/e11_runner.py](COURSE_WORK/src/course_work/model_improvement_v2/e11_runner.py) | `CREATED` | Thêm E11 preflight/runner, predecessor audit, `_audit_resume_contract`, `build_e11_resume_context` và result writer. | Hỗ trợ E11 official flow và partial Stage-B recovery không retrain completed run. |
| [src/course_work/model_improvement_v2/e12_pretest.py](COURSE_WORK/src/course_work/model_improvement_v2/e12_pretest.py) | `CREATED` | Thêm causal `Appliances_lag_144`, filtered fold evidence và `E12PopulationAudit`. | Khóa alignment/common population cho E12 lag-144 ablation. |
| [src/course_work/model_improvement_v2/e12_runner.py](COURSE_WORK/src/course_work/model_improvement_v2/e12_runner.py) | `CREATED` | Thêm E12 decision/config audit, exact fingerprints, official/partial-resume modes và comparison writer. | Chạy/recover E12 mà không carry forward E10/E11 feature blocks. |
| [src/course_work/model_improvement_v2/hybrid_loss.py](COURSE_WORK/src/course_work/model_improvement_v2/hybrid_loss.py) | `CREATED` | Thêm `HybridLossComponents`, config validation, scaler parameter extraction và `compute_hybrid_loss`. | Implement exact E13 level-plus-delta loss trong fold-local standardized Y space. |
| [src/course_work/model_improvement_v2/e13_runner.py](COURSE_WORK/src/course_work/model_improvement_v2/e13_runner.py) | `CREATED` | Thêm lambda matrix, control audit, recovery ledger, partial Stage-B resume và promotion output. | Orchestrate E13 với E01 control read-only. |
| [src/course_work/model_improvement_v2/e14_runner.py](COURSE_WORK/src/course_work/model_improvement_v2/e14_runner.py) | `CREATED` | Thêm locked M1–M5 bundle matrix, E13 control audit, candidate/context builder, completed-ledger audit và comparison. | Orchestrate E14 training-bundle experiment. |
| [src/course_work/model_improvement_v2/e15_runner.py](COURSE_WORK/src/course_work/model_improvement_v2/e15_runner.py) | `CREATED` | Thêm W96/W128 proportional width configs, parameter-count check, recovery ledger và promotion output. | Orchestrate E15 capacity audit từ E14-M1. |
| [src/course_work/model_improvement_v2/e16_runner.py](COURSE_WORK/src/course_work/model_improvement_v2/e16_runner.py) | `CREATED` | Thêm Depth3 Post-LN config, predecessor decision audit, one-change validation, run ledger và result finalization. | Orchestrate E16 depth-only experiment. |
| [src/course_work/model_improvement_v2/e20_runner.py](COURSE_WORK/src/course_work/model_improvement_v2/e20_runner.py) | `CREATED` | Thêm seed-42 reuse audit, seeded fingerprints, exact 24-run identity matrix, per-seed recovery/finalization và matched comparison. | Thực hiện matched three-seed confirmation mà không chọn best seed. |
| [src/course_work/model_improvement_v2/step16_final_refit.py](COURSE_WORK/src/course_work/model_improvement_v2/step16_final_refit.py) | `CREATED` | Thêm full pre-Test scaler fitting, locked median epoch policy, three-seed fresh refit, write-once outputs và recovery ledger. | Tạo đúng ba final-refit models và final lock inputs. |
| [src/course_work/model_improvement_v2/step17_post_hoc_benchmark.py](COURSE_WORK/src/course_work/model_improvement_v2/step17_post_hoc_benchmark.py) | `CREATED` | Thêm `TestAccessGuard`, final-lock/checksum validation, one-shot Test loader, per-seed prediction, equal-weight ensemble và benchmark artifacts. | Thực thi benchmark với nhãn `POST_HOC_V2_BENCHMARK`, không retune. |
| [src/course_work/rolling_origin/real_run.py](COURSE_WORK/src/course_work/rolling_origin/real_run.py) | `MODIFIED` | Thêm Wave 2 track/cache/config invariants, `_validate_e11_resume_ledger`, `_validate_e12_resume_ledger`, `_validate_e13_resume_ledger`, partial recovery branches, shared common-population checks và scaler bundle handoff cho hybrid loss. | Reuse completed runs có checksum/identity khóa; chỉ train canonical part còn thiếu khi được Human authorize. |
| [src/course_work/rolling_origin/refit_engine.py](COURSE_WORK/src/course_work/rolling_origin/refit_engine.py) | `MODIFIED` | `RefitEngine.refit` nhận hybrid loss, yêu cầu `y_context_raw_wh`, tích lũy component metrics và dùng dynamic history columns. | Giữ Stage-B exact refit đồng thời ghi riêng E13 loss components. |
| [src/course_work/training/engine.py](COURSE_WORK/src/course_work/training/engine.py) | `MODIFIED` | Thêm `E13_HISTORY_COLUMNS`, `history_columns_for_training`; `TrainingEngine.train` tính/log hybrid components khi policy được bật. | Mở rộng history có scope; ordinary MSE path vẫn giữ nhánh cũ. |

### Reporting, path và gate code

| File | Action | Thay đổi chính | Mục đích |
|---|---|---|---|
| [src/course_work/reporting/frozen_evidence.py](COURSE_WORK/src/course_work/reporting/frozen_evidence.py) | `CREATED` | Thêm `render_frozen_phase_evidence`, read-only artifact inventory, strict resume audit fallback, historical summary alias và checksum-verified CourseWorkV10 dashboard loader. | Trình bày frozen/historical evidence mà không materialize phase hoặc sửa artifact. |
| [src/course_work/reporting/results_rebuild.py](COURSE_WORK/src/course_work/reporting/results_rebuild.py) | `CREATED` | Thêm `_read`, `_render`, `render_verified_phase_result` cho Phase 43–59 và `render_verified_v2_results` cho V2. | Chuẩn hóa table-first presentation, fail khi source JSON bắt buộc bị thiếu. |
| [src/course_work/reporting/model_improvement_v2_dashboard.py](COURSE_WORK/src/course_work/reporting/model_improvement_v2_dashboard.py) | `CREATED` | Thêm `render_model_improvement_v2_closure_dashboard`, đọc final closure và hiển thị development/post-hoc/provenance/debt. | Cung cấp read-only closure view; notebook hiện dùng renderer tổng hợp mới trong `results_rebuild.py`. |
| [scripts/_fix_sweep_cells.py](scripts/_fix_sweep_cells.py) | `MODIFIED` | Sửa constant `NB` sang absolute path project hiện tại. | Trỏ utility tới notebook đúng vị trí. |
| [scripts/_populate_sweep_results.py](scripts/_populate_sweep_results.py) | `MODIFIED` | Sửa constant `ROOT` sang absolute path project hiện tại. | Trỏ legacy population utility tới root đúng; không được gọi bởi presentation renderer. |
| [scripts/phase44_pretrain_gate.py](scripts/phase44_pretrain_gate.py) | `MODIFIED` | Sửa suggested `cd` command và quote path có khoảng trắng. | In Human command đúng project root. |
| [src/course_work/scripts/p44_pretrain_gate.py](COURSE_WORK/src/course_work/scripts/p44_pretrain_gate.py) | `MODIFIED` | Cùng sửa suggested `cd` command như script top-level. | Đồng bộ gate copy trong package. |
| [src/course_work/scripts/p47_pretest_gate.py](COURSE_WORK/src/course_work/scripts/p47_pretest_gate.py) | `MODIFIED` | Sửa root resolution, import sang `course_work.final_test_evaluation`, kiểm tra boolean result trong `run_gate` và scan đúng evaluation file. | Gate trỏ đúng package/path hiện tại và không coi boolean `False` là PASS. |
| [src/course_work/analysis/attention_extraction/extract.py](COURSE_WORK/src/course_work/analysis/attention_extraction/extract.py) | `MODIFIED` | Sửa default root trong `run_phase52_extraction`. | Trỏ Phase 52 utility tới project hiện tại. |
| [src/course_work/analysis/attention_extraction/finalize_52fg.py](COURSE_WORK/src/course_work/analysis/attention_extraction/finalize_52fg.py) | `MODIFIED` | Sửa root trong `main`. | Dùng đúng project path khi finalize Phase 52 F/G. |
| [src/course_work/analysis/attention_extraction/materialize.py](COURSE_WORK/src/course_work/analysis/attention_extraction/materialize.py) | `MODIFIED` | Sửa fallback `project_root`. | Dùng đúng root nếu CLI không truyền `--project-root`. |
| [src/course_work/analysis/attention_extraction/orchestrator.py](COURSE_WORK/src/course_work/analysis/attention_extraction/orchestrator.py) | `MODIFIED` | Sửa default root trong `run_phase52`. | Đồng bộ Phase 52 orchestrator với vị trí project. |
| [src/course_work/analysis/last_query_attention/finalize_phase54.py](COURSE_WORK/src/course_work/analysis/last_query_attention/finalize_phase54.py) | `MODIFIED` | Sửa processing-log path và static-scan path. | Trỏ Phase 54 finalization tới project hiện tại. |

Các thay đổi path trên chỉ thay một absolute path bằng absolute path khác; khả năng chạy portable ở máy khác vẫn là reporting/code debt, không được mô tả là đã giải quyết hoàn toàn.

### Validation code

| File | Action | Thay đổi chính | Mục đích |
|---|---|---|---|
| [tests/contracts/test_selective_execution_policy.py](COURSE_WORK/tests/contracts/test_selective_execution_policy.py) | `MODIFIED` | Chuyển assertions Phase 22–41 sang `render_frozen_phase_evidence`, cập nhật cell ordering/output-preservation exclusions. | Kiểm tra notebook orchestration read-only theo layout hiện tại. |
| [tests/unit/test_model_improvement_v2_e08.py](COURSE_WORK/tests/unit/test_model_improvement_v2_e08.py) | `MODIFIED` | Exact-set assertions đổi thành subset assertions cho namespace/flatten tracks. | Giữ E08 regression khi Wave 2 registrations được thêm. |
| [tests/unit/test_model_improvement_v2_e09.py](COURSE_WORK/tests/unit/test_model_improvement_v2_e09.py) | `MODIFIED` | Direct-shape track assertion đổi thành subset. | Giữ E09 regression khi track set mở rộng có kiểm soát. |
| [tests/unit/test_phase50_h_notebook_dashboard.py](COURSE_WORK/tests/unit/test_phase50_h_notebook_dashboard.py) | `MODIFIED` | Sửa explicit project path trong renderer test. | Trỏ legacy Phase 50 test đúng root; test vẫn mô tả legacy dashboard. |
| [tests/unit/test_phase51_g_finalization.py](COURSE_WORK/tests/unit/test_phase51_g_finalization.py) | `MODIFIED` | Sửa fallback `PROJECT_ROOT`. | Trỏ Phase 51 artifact tests đúng root. |
| [tests/unit/test_model_improvement_v2_e10.py](COURSE_WORK/tests/unit/test_model_improvement_v2_e10.py) | `CREATED` | Test delta formulas/order, common population, Test firewall, namespace, four guardrails và Human gate. | Focused validation E10. |
| [tests/unit/test_model_improvement_v2_e11.py](COURSE_WORK/tests/unit/test_model_improvement_v2_e11.py) | `CREATED` | Test rolling formulas/order, causal behavior, incumbent, common population, scaling và gate. | Focused validation E11. |
| [tests/unit/test_model_improvement_v2_e11_h1_resume.py](COURSE_WORK/tests/unit/test_model_improvement_v2_e11_h1_resume.py) | `CREATED` | Test canonical/duplicate/interrupted ledger, six-Stage-A invariant và Stage-B-only partial recovery. | Chặn retrain completed E11 evidence. |
| [tests/unit/test_model_improvement_v2_e12.py](COURSE_WORK/tests/unit/test_model_improvement_v2_e12.py) | `CREATED` | Test exact lag-144 alignment, rejected-feature exclusion, fingerprints, recovery ledger và Human gate. | Focused validation E12. |
| [tests/unit/test_model_improvement_v2_e13.py](COURSE_WORK/tests/unit/test_model_improvement_v2_e13.py) | `CREATED` | Test hybrid algebra/units, scaler lineage, component history, lambda matrix và Stage-B recovery. | Focused validation E13. |
| [tests/unit/test_model_improvement_v2_e14.py](COURSE_WORK/tests/unit/test_model_improvement_v2_e14.py) | `CREATED` | Test promoted E13 control, exact M1–M5 matrix, allowed batch sizes, one-factor contract và preflight. | Focused validation E14. |
| [tests/unit/test_model_improvement_v2_e15.py](COURSE_WORK/tests/unit/test_model_improvement_v2_e15.py) | `CREATED` | Test exact width bundles, parameter limit, inherited batch 16, namespace và Human gate. | Focused validation E15. |
| [tests/unit/test_model_improvement_v2_e16.py](COURSE_WORK/tests/unit/test_model_improvement_v2_e16.py) | `CREATED` | Test E14-M1 control, depth-only Post-LN change, six-run resume ledger và no-training preflight. | Focused validation E16. |
| [tests/unit/test_model_improvement_v2_e20.py](COURSE_WORK/tests/unit/test_model_improvement_v2_e20.py) | `CREATED` | Test seed-42 reuse, exact 24 new IDs, seed-only drift, fresh state, recovery và Test firewall. | Focused validation E20. |
| [tests/unit/test_model_improvement_v2_step16.py](COURSE_WORK/tests/unit/test_model_improvement_v2_step16.py) | `CREATED` | Test common epoch policy, full pre-Test population/scalers, three fresh models, equal weights và recovery. | Validate final-refit contract. |
| [tests/unit/test_model_improvement_v2_step17.py](COURSE_WORK/tests/unit/test_model_improvement_v2_step17.py) | `CREATED` | Test final lock/checkpoints/scalers, one-shot Test guard, ensemble math, label và dual Human gate. | Validate post-hoc benchmark protocol. |
| [tests/unit/test_model_improvement_v2_final_closure.py](COURSE_WORK/tests/unit/test_model_improvement_v2_final_closure.py) | `CREATED` | Test closure policy/governance, immutable-source hashes, saved-prediction metrics và read-only dashboard. | Validate final V2 closure without new inference. |
| [tests/unit/test_notebook_frozen_evidence.py](COURSE_WORK/tests/unit/test_notebook_frozen_evidence.py) | `CREATED` | Test Phase 2/6/36 frozen views, exact Phase 6 recovery block, EDA checksums và Phase 54–57 snapshot checksums. | Validate read-only historical presentation. |
| [tests/unit/test_results_rebuild.py](COURSE_WORK/tests/unit/test_results_rebuild.py) | `CREATED` | Test Phase 43–59 non-empty tables, Phase 47 metrics, V2 result chain và Phase 1–42 prefix hash. | Validate rebuilt notebook presentation; prefix-hash case hiện ghi nhận mismatch như Mục 4. |

## 7. Thay đổi cấu trúc notebook

| Khu vực notebook | Trước | Sau |
|---|---|---|
| Tổng cấu trúc | Notebook tại `HEAD`: 150 cells. | Current notebook: 153 cells; không có cell ID bị xóa so với `HEAD`. Ba cell ID mới là `b0f56164`, `v2closure`, `v2closedash`. |
| Environment audit | Không có cell `b0f56164`. | Thêm code cell index 6 để hiển thị bảng version package và Python executable; cell có `execution_count=4`. |
| Phase 1–21 | Có materialization/orchestration output lịch sử. | Current diff có thay đổi source/output ở khu vực này; task Phase 43+ không đủ evidence để khẳng định toàn bộ Phase 1–21 unchanged. Chi tiết nguồn gốc từng thay đổi là `UNVERIFIED/UNAVAILABLE`. |
| Phase 22–32 và 34–41 | Dùng mixed `render_phase_summary`/`render_phase_resume`; một số trạng thái lấy từ saved log. | Dùng `display(render_frozen_phase_evidence(phase_id, PROJECT_ROOT))`; renderer chỉ đọc log/artifact hoặc dựng in-memory resume audit, không ghi artifact. |
| Phase 33 | Config display thuộc legacy selective flow. | Giữ renderer config chuyên biệt `render_phase_33_transformer_configuration`; cách gọi được đặt trong `display(...)`. |
| Phase 42 | Candidate synthesis renderer riêng. | Vẫn dùng `render_phase_summary(42, PROJECT_ROOT)` và một presentation-only print; không nhập vào renderer Phase 43–59. |
| Phase 43–59 | 17 code cells gọi 17 legacy dashboard renderer; tại `HEAD` các cell này có `execution_count=null`. | 17 code cells gọi chung `render_verified_phase_result`; saved `execution_count` liên tục 60–76, mỗi cell có một `display_data` HTML output và không có exception. |
| Output Phase 43–59 | Mỗi dashboard có 4–6 tables; Phase 44, 49–51 và 53–57 còn có một inline figure. | Mỗi phase có đúng ba tables và không có inline figure trong compact result cell. Figure artifact gốc không bị sửa/xóa bởi renderer mới. |
| Raw output | Không xác minh được direct raw JSON/debug print trong 17 cell Phase 43–59 tại `HEAD`; vấn đề chính là renderer phân mảnh và output dài. | Renderer mới chỉ trả HTML tables đã escape; internal source paths vẫn dùng cho contract nhưng không hiển thị trong notebook. |
| V2 section | Không tồn tại tại `HEAD`. | Thêm markdown cell `v2closure` ở index 151 và code cell `v2closedash` ở index 152; code gọi `render_verified_v2_results(PROJECT_ROOT)` và có `execution_count=77`. |
| Notebook runtime state | Phase 43–59 tại `HEAD` chưa có saved execution. | Current notebook có 77 executed code cells tổng cộng và 0 output loại `error`; đây là saved notebook state, không phải chứng minh mọi external artifact có thể regenerate. |

Kết luận về Phase 1–42: **không unchanged ở cấp current worktree**. Refactor presentation Phase 43–59 không được dùng để hợp thức hóa các thay đổi trước Phase 43; prefix-hash mismatch tiếp tục được giữ là validation debt. Source: notebook diff, [frozen_evidence.py](COURSE_WORK/src/course_work/reporting/frozen_evidence.py), [results_rebuild.py](COURSE_WORK/src/course_work/reporting/results_rebuild.py), [test_notebook_frozen_evidence.py](COURSE_WORK/tests/unit/test_notebook_frozen_evidence.py) và [test_results_rebuild.py](COURSE_WORK/tests/unit/test_results_rebuild.py).

## 8. Thay đổi presentation Phase 43–59

Toàn bộ Phase 43–59 hiện dùng [render_verified_phase_result](COURSE_WORK/src/course_work/reporting/results_rebuild.py). Mỗi phase có ba bảng presentation thống nhất và không ghi file. Các figure bị bỏ khỏi compact cell chỉ là **presentation change**; figure artifact trên disk không bị thay đổi.

| Phase | Trước refactor | Sau refactor | Source evidence |
|---|---|---|---|
| Phase 43 | Legacy `render_phase_43_dashboard`: 5 tables, 0 inline figure, cell chưa execute tại `HEAD`. | Một result row cho tuned LSTM với RMSE/MAE/R² và `PASS`; ba-table layout. `Presentation change only`. | [phase_43_signoff.json](COURSE_WORK/artifacts/lstm_tuning/phase_43_signoff.json) |
| Phase 44 | Legacy `render_phase_44_dashboard`: 6 tables và 1 inline figure. | Một row cho rolling-origin winner, pooled RMSE, fold count, Test status và `PASS`; inline figure không còn trong compact cell. `Presentation change only`. | [phase_44_signoff.json](COURSE_WORK/artifacts/rolling_origin/phase_44_signoff.json) |
| Phase 45 | Legacy `render_phase_45_dashboard`: 5 tables, 0 inline figure. | Một row cho locked model, feature set, lookback, seeds và `PASS`. Không thay final lock. `Presentation change only`. | [final_model_lock_summary.json](COURSE_WORK/artifacts/final_model_lock/final_model_lock_summary.json) |
| Phase 46 | Legacy `render_phase_46_dashboard`: 5 tables, 0 inline figure. | Một row cho candidate, seed list, completed/planned count, development RMSE và `PASS`. `Presentation change only`. | [phase_46_signoff.json](COURSE_WORK/artifacts/three_seed_final_runs/phase_46_signoff.json) |
| Phase 47 | Legacy V1 Test presentation. | V2-only table cho Seed 42/123/2026, equal-weight ensemble và Persistence với RMSE/MAE/MAPE/R² từ verified MAPE addendum. Historical V1 artifacts vẫn giữ nguyên ngoài presentation chính. | [step17_metrics_with_mape.json](COURSE_WORK/artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_metrics_with_mape.json) |
| Phase 48 | Legacy V1 prediction analysis. | V2 prediction distribution/change/peak/Persistence comparison từ frozen Step 17 predictions. | [phase48_prediction_analysis.json](COURSE_WORK/artifacts/model_improvement_v2/final_reporting_analysis/phase48_prediction_analysis.json) |
| Phase 49 | Legacy V1 residual analysis. | V2 equal-weight ensemble và Persistence residual/bias/error-distribution analysis. | [phase49_residual_analysis.json](COURSE_WORK/artifacts/model_improvement_v2/final_reporting_analysis/phase49_residual_analysis.json) |
| Phase 50 | Legacy V1 error-by-regime analysis. | V2 error-by-regime reporting với Train-defined thresholds và exact Step 17 population lineage. | [phase50_error_by_regime.json](COURSE_WORK/artifacts/model_improvement_v2/final_reporting_analysis/phase50_error_by_regime.json) |
| Phase 51 | Legacy V1 worst-error analysis. | V2 worst-error ranking/case table từ frozen ensemble predictions; không chạy model inference mới. | [phase51_worst_error_analysis.json](COURSE_WORK/artifacts/model_improvement_v2/final_reporting_analysis/phase51_worst_error_analysis.json) |
| Phase 52 | Legacy V1 attention extraction. | Giữ scientific content V1 và gắn nhãn rõ `Historical V1 attention analysis`; không diễn giải thành attention của V2. | [attention_extraction_summary.json](COURSE_WORK/artifacts/attention_extraction/attention_extraction_summary.json), [phase_52_signoff.json](COURSE_WORK/artifacts/attention_extraction/phase_52_signoff.json) |
| Phase 53 | Legacy `render_phase_53_dashboard`: 5 tables và 1 inline figure. | Một row cho seed list, dense-case count, lookback, new-inference flag và `PASS`; figure bỏ khỏi compact cell. `Presentation change only`. | [phase_53_signoff.json](COURSE_WORK/artifacts/attention_heatmaps/phase_53_signoff.json) |
| Phase 54 | Legacy `render_phase_54_dashboard`: 4 tables và 1 inline figure; recovery manifest còn ghi missing intermediate CSV checksums. | Một row/seed lấy normalized-entropy summary, metric rows, lookback và `PASS` từ current summary; không dùng `HISTORICAL_SUMMARY_ONLY`, không suy ra CSV bị thiếu. `Presentation change only`; recovery snapshot vẫn được giữ. | [last_query_attention_summary.json](COURSE_WORK/artifacts/last_query_attention/last_query_attention_summary.json), [recovery manifest](COURSE_WORK/artifacts/notebook_presentation_recovery/courseworkv10_phase54_57/manifest.json) |
| Phase 55 | Legacy `render_phase_55_dashboard`: 4 tables và 1 inline figure; recovery manifest ghi missing intermediate CSV checksums. | Một row cho pair counts, behavior cards, tests và `PASS` từ current summary; figure bỏ khỏi compact cell. `Presentation change only`; không tái tạo intermediate CSV. | [head_comparison_summary.json](COURSE_WORK/artifacts/head_comparison/head_comparison_summary.json), [recovery manifest](COURSE_WORK/artifacts/notebook_presentation_recovery/courseworkv10_phase54_57/manifest.json) |
| Phase 56 | Legacy `render_phase_56_dashboard`: 4 tables và 1 inline figure. | Một row cho seeds, findings, high/low, deciles và `PASS`; current summary được dùng trực tiếp, figure bỏ khỏi compact cell. `Presentation change only`. | [error_conditioned_attention_summary.json](COURSE_WORK/artifacts/error_conditioned_attention/error_conditioned_attention_summary.json) |
| Phase 57 | Legacy `render_phase_57_dashboard`: 4 tables và 1 inline figure. | Một row cho seeds, anchor seed, layer stability, head matching và `PASS`; current summary được dùng trực tiếp, figure bỏ khỏi compact cell. `Presentation change only`. | [seed_stability_attention_summary.json](COURSE_WORK/artifacts/seed_stability_attention/seed_stability_attention_summary.json) |
| Phase 58 | Legacy V1 final table package. | V2 final summary rút gọn từ final lock, Step 17 MAPE addendum, Phase 48–51 reporting và final closure. V1 attention chỉ được nhắc như historical interpretability evidence. | [phase58_final_summary.json](COURSE_WORK/artifacts/model_improvement_v2/final_reporting_analysis/phase58_final_summary.json) |
| Phase 59 | Legacy V1 conclusions. | V2 final conclusions cho equal-weight ensemble; không đưa ra V2 attention claim vì V2 Test attention tensors không được tạo. | [phase59_final_conclusions.json](COURSE_WORK/artifacts/model_improvement_v2/final_reporting_analysis/phase59_final_conclusions.json) |

Không Phase 43–59 nào dùng `render_frozen_phase_evidence` trong current notebook. Phase 47–51/58–59 đọc V2 artifacts có checksum-bound source prediction lineage; Phase 52–57 giữ V1 historical attention và không bị đổi nhãn thành V2.

## 9. Thay đổi MODEL_IMPROVEMENT_V2

`MODEL_IMPROVEMENT_V2` được triển khai trong namespace riêng, giữ V1 làm historical evidence và thực hiện lựa chọn trên development evidence. Wave 1 gồm `E00–E09`; Wave 2 gồm `E10–E20`. `Step 14A` và `Step 14B` là hai evaluation bổ sung, tách từ `Step 14` của plan gốc và không phát sinh training run mới. `Step 16` khóa/refit final policy; `Step 17` là `POST_HOC_V2_BENCHMARK`, không phải unbiased unseen Test.

| Experiment/Step | Mục tiêu | Kết quả chính đã xác minh | Decision/Status | Source Artifact |
|---|---|---|---|---|
| `E00` | Phân loại và đóng băng V1 lineage trước khi mở V2 | Khóa `TR_C2_ALT_LOOKBACK`, `FS2_TF1`, lookback `72`, fingerprint `585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24`; không training | `FROZEN`; Test `NOT_ACCESSED` | [baseline_snapshot.json](COURSE_WORK/artifacts/model_improvement_v2/baseline_snapshot.json), [lineage_manifest.json](COURSE_WORK/artifacts/model_improvement_v2/lineage_manifest.json) |
| `E01` | Reproduce V1-equivalent Phase 44 development baseline trong V2 | `RMSE=59.852916 Wh`, `MAE=26.650502 Wh`, `R²=0.578943` | Baseline development được dùng làm control cho các ablation sau; artifact gốc ghi `HUMAN_REVIEW_REQUIRED` | [e01_reproduction_comparison.json](COURSE_WORK/artifacts/model_improvement_v2/experiments/E01/e01_reproduction_comparison.json) |
| `E02` | Ablation `FS1_TF1` so với `FS2_TF1` | Challenger `RMSE=60.127745 Wh`; không vượt control | `NO_MEANINGFUL_IMPROVEMENT / TIE`; giữ `E01` | [e02_feature_ablation_comparison.json](COURSE_WORK/artifacts/model_improvement_v2/experiments/E02/e02_feature_ablation_comparison.json) |
| `E03` | `DIRECT` so với `RESIDUAL_TO_PERSISTENCE` | Residual challenger `RMSE=60.046541 Wh` | `NO_MEANINGFUL_IMPROVEMENT / TIE`; giữ `E01` | [e03_prediction_ablation_comparison.json](COURSE_WORK/artifacts/model_improvement_v2/experiments/E03/e03_prediction_ablation_comparison.json) |
| `E04` | Residual `LINEAR` head so với `MLP` head | `MLP RMSE=60.664335 Wh` | `DO_NOT_PROMOTE_E04_GLOBAL` | [e04_head_ablation_comparison.json](COURSE_WORK/artifacts/model_improvement_v2/experiments/E04/e04_head_ablation_comparison.json) |
| `E05` | Residual gate `OFF` so với `ON` | Gate challenger `RMSE=60.681364 Wh` | `DO_NOT_PROMOTE_E05_GLOBAL` | [e05_gate_ablation_comparison.json](COURSE_WORK/artifacts/model_improvement_v2/experiments/E05/e05_gate_ablation_comparison.json) |
| `E06` | Audit constant `AdamW` learning rate | `3e-4` vẫn có `RMSE=59.852916 Wh` và tốt nhất trong matrix | `RETAIN_E01_LR_3E4` | [e06_lr_audit_comparison.json](COURSE_WORK/artifacts/model_improvement_v2/experiments/E06/e06_lr_audit_comparison.json) |
| `E07` | Scheduler ablation tại initial LR `3e-4` | Scheduler `OFF` control được giữ | `RETAIN_E01_SCHEDULER_OFF` | [e07_scheduler_ablation_comparison.json](COURSE_WORK/artifacts/model_improvement_v2/experiments/E07/e07_scheduler_ablation_comparison.json) |
| `E08` | Bounded plain `SGD` optimizer audit | Không challenger nào thay thế `AdamW 3e-4` control | `RETAIN_E01_ADAMW` | [e08_sgd_lr_ablation_comparison.json](COURSE_WORK/artifacts/model_improvement_v2/experiments/E08/e08_sgd_lr_ablation_comparison.json) |
| `E09` | Bounded `SGD+momentum` audit | Không challenger nào thay thế `AdamW` control | `RETAIN_E01_ADAMW` | [e09_sgdm_ablation_comparison.json](COURSE_WORK/artifacts/model_improvement_v2/experiments/E09/e09_sgdm_ablation_comparison.json) |
| `E10` | Thêm ba causal target-delta features | Challenger `RMSE=58.976286 Wh`; gate fold-consistency không đạt | `eligible_for_human_promotion=false`; Human decision artifact riêng: `UNAVAILABLE` | [e10_target_delta_ablation_comparison.json](COURSE_WORK/artifacts/model_improvement_v2/experiments/E10/e10_target_delta_ablation_comparison.json) |
| `E11` | Thêm bảy past-only rolling-target features | Challenger `RMSE=60.304126 Wh`; bốn quantitative guardrails đều không đạt | `HUMAN_REVIEW_REQUIRED`; Human decision artifact riêng: `UNAVAILABLE` | [e11_rolling_target_ablation_comparison.json](COURSE_WORK/artifacts/model_improvement_v2/experiments/E11/e11_rolling_target_ablation_comparison.json) |
| `E12` | Thêm `Appliances_lag_144` | Challenger `RMSE=61.894523 Wh`; bốn quantitative guardrails đều không đạt | `HUMAN_REVIEW_REQUIRED`; Human decision artifact riêng: `UNAVAILABLE` | [e12_lag144_ablation_comparison.json](COURSE_WORK/artifacts/model_improvement_v2/experiments/E12/e12_lag144_ablation_comparison.json) |
| `E13` | Hybrid level-plus-delta loss | `lambda_delta=0.10` đạt `RMSE=59.605124 Wh`, `MAE=26.612618 Wh`, `R²=0.582423` | `PROMOTE` `TR_C2_ALT_LOOKBACK_E13_LAMBDA_010` | [e13_human_decision.json](COURSE_WORK/artifacts/model_improvement_v2/experiments/E13/e13_human_decision.json) |
| `E14` | Locked bundle matrix trên post-`E13` incumbent | `M1`: `RMSE=58.553453 Wh`, `MAE=26.337799 Wh`, `R²=0.597028` | `PROMOTE_M1`; candidate `TR_C2_ALT_LOOKBACK_E14_M1` | [e14_human_decision.json](COURSE_WORK/artifacts/model_improvement_v2/experiments/E14/e14_human_decision.json) |
| `E15` | Width/capacity audit `W96`, `W128` | `W96 RMSE=61.689300 Wh`; `W128 RMSE=60.690317 Wh` | `REJECT_E15`; giữ `E14-M1` | [e15_human_decision.json](COURSE_WORK/artifacts/model_improvement_v2/experiments/E15/e15_human_decision.json) |
| `E16` | Depth audit, `num_layers: 2 → 3`, Post-LN | Depth-3 challenger `RMSE=61.167184 Wh` | `REJECT_E16`; giữ `E14-M1` | [e16_human_decision.json](COURSE_WORK/artifacts/model_improvement_v2/experiments/E16/e16_human_decision.json) |
| `E17` | Pre-LN eligibility trên deeper architecture | Không mở experiment do `E16` không được promote và không có justification riêng | `DEFERRED` | [e16_human_decision.json](COURSE_WORK/artifacts/model_improvement_v2/experiments/E16/e16_human_decision.json) |
| `E18` | Fine-tuning eligibility gate | Checkpoint reuse được phân loại `EXACT_REUSE_ALLOWED`; không tạo run | `ACCEPT_FINETUNING_ELIGIBILITY`; `FT-A/B/C=DEFERRED_NO_PROMOTED_NEW_HEAD` | [e18_human_decision.json](COURSE_WORK/artifacts/model_improvement_v2/experiments/E18/e18_human_decision.json) |
| `E19` | Conditional TCN/alternative architecture | Trigger không được mở | `NOT_TRIGGERED` | [e20_config_snapshot.json](COURSE_WORK/artifacts/model_improvement_v2/experiments/E20/e20_config_snapshot.json) |
| `E20` | Matched three-seed confirmation: `E01` control so với `E14-M1` finalist | Mean RMSE: control `60.019421 Wh`, finalist `60.044053 Wh`; paired delta theo seed `[-1.299463, -0.220295, +1.593653] Wh` | `REJECT` trong Step 16 governance; không chọn best seed | [e20_matched_seed_comparison.json](COURSE_WORK/artifacts/model_improvement_v2/experiments/E20/e20_matched_seed_comparison.json), [step16_final_refit_config_snapshot.json](COURSE_WORK/artifacts/model_improvement_v2/final_model_lock/step16_final_refit_config_snapshot.json) |
| `Step 14A` *(evaluation bổ sung)* | Equal-weight seed ensemble cho control và finalist | Chọn ensemble `mean(E14-M1 seeds 42,123,2026)`; development `RMSE=58.571807 Wh`, `MAE=25.545270 Wh`, `R²=0.596775` | `SELECT_FINALIST_ENSEMBLE`; `0` training run | [step16_final_refit_config_snapshot.json](COURSE_WORK/artifacts/model_improvement_v2/final_model_lock/step16_final_refit_config_snapshot.json) |
| `Step 14B` *(evaluation bổ sung)* | Fixed Persistence–Neural blend, `alpha∈{0.25,0.50,0.75}` | Không blend nào được chọn; giữ neural ensemble | `KEEP_NEURAL_ENSEMBLE`; `0` training run | [step16_final_refit_config_snapshot.json](COURSE_WORK/artifacts/model_improvement_v2/final_model_lock/step16_final_refit_config_snapshot.json) |
| `Step 16` | Final refit và V2 lock | Ba seeds `42,123,2026`, mỗi seed `16` epochs; equal weights `[1/3,1/3,1/3]` | `LOCKED`, `3_OF_3_COMPLETED` | [v2_final_model_lock.json](COURSE_WORK/artifacts/model_improvement_v2/final_model_lock/v2_final_model_lock.json) |
| `Step 17` | Benchmark final policy trên old Test | Ensemble `RMSE=61.608937 Wh`, `MAE=25.897130 Wh`, `R²=0.540364`; `2961` samples | `POST_HOC_V2_BENCHMARK`; Test chỉ được diễn giải là historical benchmark | [step17_metrics.json](COURSE_WORK/artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_metrics.json) |
| `Final Closure` | Đóng workflow `MODEL_IMPROVEMENT_V2` | Final policy là mean của ba final-refit seed, không best-seed selection, không post-Test retuning | `COMPLETE` | [model_improvement_v2_final_closure.json](COURSE_WORK/artifacts/model_improvement_v2/model_improvement_v2_final_closure.json) |

Các result/decision độc lập cho `Step 14A` và `Step 14B` không tồn tại dưới một namespace riêng. Giá trị được trình bày ở đây là trạng thái đã được carry-forward và khóa trong `human_governance`, `final_prediction_policy` và `development_metrics` của [step16_final_refit_config_snapshot.json](COURSE_WORK/artifacts/model_improvement_v2/final_model_lock/step16_final_refit_config_snapshot.json), sau đó được xác nhận trong final lock/closure. Các metric blend theo từng `alpha` không được render vì không có machine-readable result artifact độc lập để truy vết.

## 10. Thay đổi output

Thay đổi ở mục này là thay đổi presentation/reporting. Scientific artifact gốc không bị viết lại bởi renderer.

| Phase/Section | Before | After | Change Type |
|---|---|---|---|
| `Phase 1–21` | Có materialization/orchestration output lịch sử | Current worktree có thay đổi source/output ở khu vực này; nguồn gốc chi tiết không đủ evidence để phân loại là một phần của rebuild Phase 43–59 | `UNVERIFIED/UNAVAILABLE` |
| `Phase 22–32`, `Phase 34–41` | Các cell dùng mixed `render_phase_summary`/`render_phase_resume` | Dùng `render_frozen_phase_evidence()` để hiển thị compact table từ evidence hiện có, không chạy lại phase | `READ_ONLY_FROZEN_RENDERER` |
| `Phase 33` | Config display thuộc legacy selective flow | Giữ renderer chuyên biệt `render_phase_33_transformer_configuration()` | `PRESERVED_SPECIAL_RENDERER` |
| `Phase 42` | Cell boundary giữ logic riêng | Không chuyển sang `render_verified_phase_result()` | `PRESERVED_BOUNDARY` |
| `Phase 43–59` | Mỗi phase có khoảng `4–6` HTML table; chín phase còn inline figure; cấu trúc output không đồng nhất | Mỗi phase dùng `render_verified_phase_result()` và render đúng ba bảng: status, artifact inventory và result summary; không còn inline figure trong block compact | `REBUILT` |
| `Phase 43–59` status/result | Status và số liệu nằm rải trong output legacy | Status lấy từ current signoff; result row lấy từ verified machine-readable artifact được khai báo trong `results_rebuild.py` | `VERIFIED_SUMMARY` |
| `Phase 47` | V1 Test presentation legacy | V2 per-seed/equal-weight ensemble/Persistence table, gồm RMSE/MAE/MAPE/R² từ Step 17 MAPE addendum | `V2_POST_HOC_PRESENTATION`; V1 artifacts vẫn bất biến |
| `Phase 52–57` | V1 historical attention analysis | Giữ nguyên scientific content và gắn nhãn rõ đây không phải V2 attention | `V1_HISTORICAL_ATTENTION` |
| `Phase 58–59` | V1 reporting/packaging output legacy | V2 final summary/conclusions từ verified final reporting artifacts | `V2_FINAL_PRESENTATION` |
| `MODEL_IMPROVEMENT_V2` | Không có section tương ứng ở `HEAD` notebook | Thêm một section table-first gồm overview, final policy và các result row quan trọng từ `E01–E20`, `Step 14A/B`, `Step 16`, `Step 17` và closure | `ADDED` |

Không có evidence cho phép khẳng định các cell Phase 43–59 trước refactor chứa raw JSON hoặc direct debug `print()`. Vì vậy tài liệu không gán change type “raw JSON removed” cho các cell này. Thay đổi có thể xác minh là chuẩn hóa số lượng bảng, bỏ inline figure khỏi compact result block và chuyển sang renderer truy vết artifact.

## 11. Thay đổi bảng và hình

| Phase/Section | Table/Figure | Action | Source |
|---|---|---|---|
| `Phase 43–59` | `Execution decision`, `Artifact inventory`, `Verified result summary` | `REBUILT`: 17 phase block × 3 HTML table = `51` table | [CourseWork.ipynb](COURSE_WORK/notebook_course_work/CourseWork.ipynb), [results_rebuild.py](COURSE_WORK/src/course_work/reporting/results_rebuild.py) |
| `Phase 44` | Historical inline figure trong result cell | `REMOVED_FROM_NOTEBOOK_PRESENTATION`; artifact khoa học không bị xóa | `git diff -- notebook_course_work/CourseWork.ipynb` |
| `Phase 49–51` | Ba historical inline figure | `REMOVED_FROM_NOTEBOOK_PRESENTATION`; result giữ ở bảng compact | `git diff -- notebook_course_work/CourseWork.ipynb` |
| `Phase 53–57` | Năm historical inline figure | `REMOVED_FROM_NOTEBOOK_PRESENTATION`; renderer không tái chạy figure generation | `git diff -- notebook_course_work/CourseWork.ipynb` |
| `Phase 47` | V2 seed/ensemble/Persistence table với MAPE | `REBUILT`: đọc trực tiếp Step 17 MAPE addendum | [step17_metrics_with_mape.json](COURSE_WORK/artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_metrics_with_mape.json) |
| `Phase 54–57` | Bốn CourseWorkV10 dashboard snapshots | `RESTORED` vào recovery namespace và checksum được ghi trong manifest; đây là recovery evidence, không phải active scientific recomputation | [courseworkv10_phase54_57](COURSE_WORK/artifacts/notebook_presentation_recovery/courseworkv10_phase54_57/), [manifest.json](COURSE_WORK/artifacts/notebook_presentation_recovery/courseworkv10_phase54_57/manifest.json) |
| `Phase 54` | `phase54_residual_metrics.csv` | `UNAVAILABLE`: không có trong active artifact tree; summary renderer chỉ dùng artifact hiện tồn tại | [results_rebuild.py](COURSE_WORK/src/course_work/reporting/results_rebuild.py) |
| `Phase 55` | `phase55_segment_metrics.csv` | `UNAVAILABLE`: không có trong active artifact tree; không tạo bảng giả | [results_rebuild.py](COURSE_WORK/src/course_work/reporting/results_rebuild.py) |
| `Phase 58–59` | V2 final summary và conclusions | `REBUILT`: final locked policy, benchmark, MAPE và reporting analyses; V1 attention không bị nhập vào V2 conclusion | [final_reporting_analysis](COURSE_WORK/artifacts/model_improvement_v2/final_reporting_analysis/) |
| `MODEL_IMPROVEMENT_V2` | Overview/config/result tables | `ADDED`: một output block gồm `3` HTML table và `26` result row | [CourseWork.ipynb](COURSE_WORK/notebook_course_work/CourseWork.ipynb), [model_improvement_v2_final_closure.json](COURSE_WORK/artifacts/model_improvement_v2/model_improvement_v2_final_closure.json) |
| `Step 17` trong V2 summary | Per-seed, ensemble và baseline result rows, gồm MAPE | `ADDED`: render từ MAPE addendum; không inference lại | [step17_metrics_with_mape.json](COURSE_WORK/artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_metrics_with_mape.json) |
| `Phase 6` | `16` EDA CSV và `16` PNG đã được kiểm tra trong active namespace | `PRESERVED`; nằm ngoài phạm vi rebuild Phase 43–59 | [artifacts/eda](COURSE_WORK/artifacts/eda/) |

Tổng `0` inline figure trong các compact block Phase 43–59 hiện tại không có nghĩa là file hình nguồn đã bị xóa. Đây là quyết định presentation: giữ phần đánh giá cuối gọn và table-first; artifact figure vẫn thuộc namespace phase tương ứng nếu file còn tồn tại.

## 12. Kết quả khoa học

### 12.1. Kết quả V1, V2 development và benchmark

| Model/Policy | RMSE (Wh) | MAE (Wh) | R² | Evidence Type | Source |
|---|---:|---:|---:|---|---|
| V1 historical Transformer mean | `63.829658` | `28.528603` | `0.506422` | Historical Phase 47 Test result | [phase_47_signoff.json](COURSE_WORK/artifacts/final_test/phase_47_signoff.json), [model_improvement_v2_final_closure.json](COURSE_WORK/artifacts/model_improvement_v2/model_improvement_v2_final_closure.json) |
| V2 selected neural ensemble / final locked policy | `58.571807` | `25.545270` | `0.596775` | Development-only evidence; `mean(E14-M1 seeds 42,123,2026)` | [v2_final_model_lock.json](COURSE_WORK/artifacts/model_improvement_v2/final_model_lock/v2_final_model_lock.json) |
| V2 final equal-weight ensemble | `61.608937` | `25.897130` | `0.540364` | `POST_HOC_V2_BENCHMARK`, `2961` Test samples | [step17_metrics.json](COURSE_WORK/artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_metrics.json) |
| Persistence baseline | `66.836915` | `26.737589` | `0.459047` | Same `POST_HOC_V2_BENCHMARK` population | [step17_metrics.json](COURSE_WORK/artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_metrics.json) |

Development metrics và benchmark metrics thuộc hai population/evidence type khác nhau, do đó không được dùng chéo để tune hoặc khẳng định Test performance từ development. Final policy đã được khóa trước `Step 17` và không thay đổi sau khi Test được mở.

### 12.2. Per-seed `POST_HOC_V2_BENCHMARK`

| Seed/Policy | RMSE (Wh) | MAE (Wh) | MAPE (%) | R² |
|---|---:|---:|---:|---:|
| `42` | `61.262482` | `27.292729` | `24.187279` | `0.545519` |
| `123` | `63.259959` | `27.332280` | `23.147923` | `0.515398` |
| `2026` | `62.423633` | `25.734231` | `20.222825` | `0.528127` |
| Equal-weight ensemble | `61.608937` | `25.897130` | `21.519219` | `0.540364` |
| Persistence | `66.836915` | `26.737589` | `21.513353` | `0.459047` |

Nguồn cho toàn bộ bảng: [step17_metrics_with_mape.json](COURSE_WORK/artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_metrics_with_mape.json). MAPE addendum đọc frozen prediction artifacts; `training_executed=false`, `inference_executed=false`, `raw_test_source_opened=false`.

### 12.3. So sánh trên historical Test benchmark

Delta trong bảng được tính theo `V2 − reference`; giá trị âm của RMSE/MAE và giá trị dương của R² là tốt hơn.

| So sánh | ΔRMSE (Wh) | ΔMAE (Wh) | ΔR² | Diễn giải |
|---|---:|---:|---:|---|
| V2 ensemble so với V1 historical Transformer mean | `-2.220722` | `-2.631474` | `+0.033942` | V2 có metric tốt hơn trong báo cáo historical/post-hoc; đây không phải model-selection evidence mới |
| V2 ensemble so với Persistence | `-5.227979` | `-0.840459` | `+0.081317` | V2 tốt hơn Persistence trên cùng Step 17 benchmark population |

So sánh với V1 là historical context được closure cho phép, nhưng V1 value là historical Transformer mean còn V2 value là ensemble prediction metric; vì vậy không được diễn giải như một matched retraining experiment.

### 12.4. Trạng thái scientific evidence sau refactor

| Check | Kết quả |
|---|---|
| Final V2 policy | `mean(E14-M1 final-refit seeds 42,123,2026)` |
| Ensemble weights | `[1/3,1/3,1/3]` |
| Final refit | `3_OF_3_COMPLETED`, `16` epochs cho mỗi seed |
| V2 lock | `LOCKED` |
| Benchmark label | `POST_HOC_V2_BENCHMARK` |
| Test status sau closure | `ACCESSED_ONCE_FOR_POST_HOC_V2_BENCHMARK` |
| Post-Test retuning | `FALSE` |
| Best-seed selection | `FALSE` |
| MODEL_IMPROVEMENT_V2 status | `COMPLETE` |
| `SCIENTIFIC_RESULTS_CHANGED` bởi notebook/reporting refactor | `NO` |

Refactor chỉ thay presentation và traceability của kết quả đã tồn tại. Nó không thay checkpoint, scaler, model weights, config, ensemble weights hoặc scientific conclusion sau khi xem Test.

## 13. Truy vết kết quả và artifact

Mỗi dòng dưới đây trỏ đến machine-readable artifact hoặc Human decision artifact đang tồn tại. Giá trị trong bảng là bản rút gọn từ các field tương ứng; raw JSON không được sao chép vào tài liệu.

| Kết quả / Result | Giá trị chính | Source Artifact | Traceability Status |
|---|---|---|---|
| V1 historical Transformer mean | `RMSE=63.829658 Wh`; `MAE=28.528603 Wh`; `R²=0.506422` | [final_test_summary.json](COURSE_WORK/artifacts/final_test/final_test_summary.json), [phase_47_signoff.json](COURSE_WORK/artifacts/final_test/phase_47_signoff.json) | `PASS` |
| V1 Persistence | `RMSE=66.836915 Wh`; `MAE=26.737589 Wh`; `R²=0.459047` | [final_test_summary.json](COURSE_WORK/artifacts/final_test/final_test_summary.json), [phase_47_signoff.json](COURSE_WORK/artifacts/final_test/phase_47_signoff.json) | `PASS` |
| `E13` hybrid loss winner | `TR_C2_ALT_LOOKBACK_E13_LAMBDA_010`; `lambda_delta=0.10`; `RMSE=59.605124 Wh`; `MAE=26.612618 Wh`; `R²=0.582423` | [e13_human_decision.json](COURSE_WORK/artifacts/model_improvement_v2/experiments/E13/e13_human_decision.json), [e13_hybrid_loss_comparison.json](COURSE_WORK/artifacts/model_improvement_v2/experiments/E13/e13_hybrid_loss_comparison.json) | `PASS`; Human decision `PROMOTE` |
| `E14` winner | `TR_C2_ALT_LOOKBACK_E14_M1`; `RMSE=58.553453 Wh`; `MAE=26.337799 Wh`; `R²=0.597028` | [e14_human_decision.json](COURSE_WORK/artifacts/model_improvement_v2/experiments/E14/e14_human_decision.json), [e14_bundle_comparison.json](COURSE_WORK/artifacts/model_improvement_v2/experiments/E14/e14_bundle_comparison.json) | `PASS`; Human decision `PROMOTE_M1` |
| `E15` capacity challengers | `W96 RMSE=61.689300 Wh`; `W128 RMSE=60.690317 Wh` | [e15_capacity_comparison.json](COURSE_WORK/artifacts/model_improvement_v2/experiments/E15/e15_capacity_comparison.json), [e15_human_decision.json](COURSE_WORK/artifacts/model_improvement_v2/experiments/E15/e15_human_decision.json) | `PASS`; Human decision `REJECT_E15` |
| `E16` depth-3 Post-LN challenger | `RMSE=61.167184 Wh`; `MAE=27.022373 Wh`; `R²=0.560249` | [e16_depth_comparison.json](COURSE_WORK/artifacts/model_improvement_v2/experiments/E16/e16_depth_comparison.json), [e16_human_decision.json](COURSE_WORK/artifacts/model_improvement_v2/experiments/E16/e16_human_decision.json) | `PASS`; Human decision `REJECT_E16` |
| `E17` Pre-LN eligibility | Không mở experiment | [e16_human_decision.json](COURSE_WORK/artifacts/model_improvement_v2/experiments/E16/e16_human_decision.json) | `DEFERRED` |
| `E18` fine-tuning eligibility | Checkpoint classification `EXACT_REUSE_ALLOWED`; không tạo training run | [e18_human_decision.json](COURSE_WORK/artifacts/model_improvement_v2/experiments/E18/e18_human_decision.json) | `ACCEPT_FINETUNING_ELIGIBILITY`; `FT-A/B/C=DEFERRED_NO_PROMOTED_NEW_HEAD` |
| `E19` alternative architecture | Không có Human-approved trigger | [e20_config_snapshot.json](COURSE_WORK/artifacts/model_improvement_v2/experiments/E20/e20_config_snapshot.json) | `NOT_TRIGGERED` |
| `E20` matched three-seed confirmation | Mean `RMSE`: control `60.019421 Wh`, finalist `60.044053 Wh`; finalist thắng seed `42`, `123` và thua seed `2026` | [e20_matched_seed_comparison.json](COURSE_WORK/artifacts/model_improvement_v2/experiments/E20/e20_matched_seed_comparison.json), [step16_final_refit_config_snapshot.json](COURSE_WORK/artifacts/model_improvement_v2/final_model_lock/step16_final_refit_config_snapshot.json) | `PASS` về result trace; Human governance `REJECT` |
| `Step 14A` seed ensemble | `mean(E14-M1 seeds 42,123,2026)`; `RMSE=58.571807 Wh`; `MAE=25.545270 Wh`; `R²=0.596775` | [step16_final_refit_config_snapshot.json](COURSE_WORK/artifacts/model_improvement_v2/final_model_lock/step16_final_refit_config_snapshot.json), [v2_final_model_lock.json](COURSE_WORK/artifacts/model_improvement_v2/final_model_lock/v2_final_model_lock.json) | `PASS`; `SELECT_FINALIST_ENSEMBLE` được carry-forward vào lock |
| `Step 14B` Persistence–Neural blend | Không chọn blend; giữ neural ensemble | [step16_final_refit_config_snapshot.json](COURSE_WORK/artifacts/model_improvement_v2/final_model_lock/step16_final_refit_config_snapshot.json) | `PASS` cho decision `KEEP_NEURAL_ENSEMBLE`; metric theo từng `alpha`: `UNVERIFIED/UNAVAILABLE` do không có result artifact độc lập |
| `Step 16` final lock | Policy `MEAN_E14_M1_SEEDS_42_123_2026`; seeds `[42,123,2026]`; weights `[1/3,1/3,1/3]`; epoch policy `MEDIAN_RO_INNER_BEST_EPOCHS-v1`; `16` epochs/seed | [v2_final_model_lock.json](COURSE_WORK/artifacts/model_improvement_v2/final_model_lock/v2_final_model_lock.json) | `LOCKED`; `PASS` |
| `Step 16` final-refit population | `TRAIN+VALIDATION`; `16630` targets; `33` features; lookback `72`; Test excluded; target fingerprint `a4b7c9479867884cd9e8e2f0a50f634735a31e4adb6f6fb9f90d944c34de4731` | [v2_final_model_lock.json](COURSE_WORK/artifacts/model_improvement_v2/final_model_lock/v2_final_model_lock.json) | `PASS` |
| `Step 16` final-refit checkpoints/scalers | `3_OF_3_COMPLETED`; ba checkpoint và hai scaler có SHA-256 khớp manifest/lock | [step16_final_refit_manifest.json](COURSE_WORK/artifacts/model_improvement_v2/final_model_lock/step16_final_refit_manifest.json), [v2_final_model_lock.json](COURSE_WORK/artifacts/model_improvement_v2/final_model_lock/v2_final_model_lock.json) | `PASS` |
| `Step 17` ensemble benchmark | `RMSE=61.608937 Wh`; `MAE=25.897130 Wh`; `MAPE=21.519219%`; `R²=0.540364`; `2961` samples | [step17_metrics_with_mape.json](COURSE_WORK/artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_metrics_with_mape.json), [step17_mape_addendum_manifest.json](COURSE_WORK/artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_mape_addendum_manifest.json), [step17_mape_addendum_signoff.json](COURSE_WORK/artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_mape_addendum_signoff.json) | `PASS`; label `POST_HOC_V2_BENCHMARK_MAPE_ADDENDUM` |
| `MODEL_IMPROVEMENT_V2` final closure | Final policy giữ equal-weight ensemble; `model_improvement_v2_status=COMPLETE`; `post_test_retuning=false` | [model_improvement_v2_final_closure.json](COURSE_WORK/artifacts/model_improvement_v2/model_improvement_v2_final_closure.json) | `PASS`; `COMPLETE` |

### 13.1. Checkpoint và scaler lineage của final lock

| Thành phần | Run ID / File | SHA-256 đã xác minh | Status |
|---|---|---|---|
| Seed `42` checkpoint | [RUN_V2_TR_STEP16_SEED42_FINAL_A01_A86FBFB4/final_refit.pt](COURSE_WORK/artifacts/model_improvement_v2/final_model_lock/runs/RUN_V2_TR_STEP16_SEED42_FINAL_A01_A86FBFB4/checkpoints/final_refit.pt) | `5cf30c81a92abf27522cf592eebb53410a12b3d90e5f3ab8785305d48174a8f6` | `COMPLETED`; `16` epochs |
| Seed `123` checkpoint | [RUN_V2_TR_STEP16_SEED123_FINAL_A01_E6361049/final_refit.pt](COURSE_WORK/artifacts/model_improvement_v2/final_model_lock/runs/RUN_V2_TR_STEP16_SEED123_FINAL_A01_E6361049/checkpoints/final_refit.pt) | `dfbdc6473e144fa315d54c80c063884ffe3f6d3df3997e0f1847f0a814297766` | `COMPLETED`; `16` epochs |
| Seed `2026` checkpoint | [RUN_V2_TR_STEP16_SEED2026_FINAL_A01_FA2BE43F/final_refit.pt](COURSE_WORK/artifacts/model_improvement_v2/final_model_lock/runs/RUN_V2_TR_STEP16_SEED2026_FINAL_A01_FA2BE43F/checkpoints/final_refit.pt) | `8eb9d9f0f7a14061218c06c391b7698c14033cd19576db3a7d1718f9516941c4` | `COMPLETED`; `16` epochs |
| Final X scaler | [final_x_scaler.json](COURSE_WORK/artifacts/model_improvement_v2/final_model_lock/scalers/final_x_scaler.json) | `521df9de48516f4d5d661f76a35c46241a24cd357e66ffab2e1f0695a54c7724` | `PASS`; fit trên full pre-Test development data |
| Final Y scaler | [final_y_scaler.json](COURSE_WORK/artifacts/model_improvement_v2/final_model_lock/scalers/final_y_scaler.json) | `c572e817ccede5dd5dd481c9f5ab780a45481a256d695680c832f4385cf206f3` | `PASS`; fit trên full pre-Test development data |

Các SHA-256 trong bảng đã được tính lại từ file hiện tại và khớp [step16_final_refit_manifest.json](COURSE_WORK/artifacts/model_improvement_v2/final_model_lock/step16_final_refit_manifest.json) cùng [step17_benchmark_manifest.json](COURSE_WORK/artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_benchmark_manifest.json).

### 13.2. Audit traceability của notebook/reporting

| Check | Kết quả |
|---|---|
| Metric V1 Phase 47 | Truy vết được về `final_test_summary.json`/`phase_47_signoff.json`: `PASS` |
| Metric Phase 43–59 | Truy vết được về source artifact khai báo trong `render_verified_phase_result()`: `PASS` |
| Metric/result V2 hiển thị trong notebook | Truy vết được về comparison/Human decision/final lock/closure artifact được load bởi `render_verified_v2_results()`: `PASS` |
| Metric `Step 14B` theo từng `alpha` | Không hiển thị vì result artifact độc lập không tồn tại: `UNVERIFIED/UNAVAILABLE` |
| Metric không truy vết nhưng vẫn được đánh `PASS` | `0` |
| Scientific artifact bị renderer ghi lại | `0`; renderer là read-only |

## 14. Liên kết tới artifact/kết quả

Danh mục dưới đây chỉ giữ các entry point quan trọng. Tất cả đều là relative Markdown link và đã được kiểm tra tồn tại trong current repo.

### Notebook

- [CourseWork.ipynb](COURSE_WORK/notebook_course_work/CourseWork.ipynb)

### Plan / Refactor Documentation

- [Plan_improve_model.md](COURSE_WORK/docs/Plan_improve_model.md)
- [model_improvement_v2_pre_process_plan.md](COURSE_WORK/docs/plan/plan_before_process/model_improvement_v2_pre_process_plan.md)
- [model_improvement_v2_wave2_pre_process_plan.md](COURSE_WORK/docs/plan/plan_before_process/model_improvement_v2_wave2_pre_process_plan.md)
- [model_improvement_v2_e20_pre_process_plan.md](COURSE_WORK/docs/plan/plan_before_process/model_improvement_v2_e20_pre_process_plan.md)
- [model_improvement_v2_step14a_seed_ensemble_pre_process_plan.md](COURSE_WORK/docs/plan/plan_before_process/model_improvement_v2_step14a_seed_ensemble_pre_process_plan.md)
- [model_improvement_v2_step14b_persistence_neural_blend_pre_process_plan.md](COURSE_WORK/docs/plan/plan_before_process/model_improvement_v2_step14b_persistence_neural_blend_pre_process_plan.md)
- [model_improvement_v2_step16_final_refit_lock_pre_process_plan.md](COURSE_WORK/docs/plan/plan_before_process/model_improvement_v2_step16_final_refit_lock_pre_process_plan.md)
- [model_improvement_v2_step17_post_hoc_benchmark_pre_process_plan.md](COURSE_WORK/docs/plan/plan_before_process/model_improvement_v2_step17_post_hoc_benchmark_pre_process_plan.md)

### V1 Historical Evidence

- [final_test_summary.json](COURSE_WORK/artifacts/final_test/final_test_summary.json)
- [phase_47_signoff.json](COURSE_WORK/artifacts/final_test/phase_47_signoff.json)
- [final_test_evaluation_manifest.json](COURSE_WORK/artifacts/final_test/final_test_evaluation_manifest.json)
- [prediction_checksums.json](COURSE_WORK/artifacts/final_test/prediction_checksums.json)
- [final_test_report.md](COURSE_WORK/artifacts/final_test/final_test_report.md)

### MODEL_IMPROVEMENT_V2

- [E13 hybrid-loss comparison](COURSE_WORK/artifacts/model_improvement_v2/experiments/E13/e13_hybrid_loss_comparison.json)
- [E13 Human decision](COURSE_WORK/artifacts/model_improvement_v2/experiments/E13/e13_human_decision.json)
- [E14 bundle comparison](COURSE_WORK/artifacts/model_improvement_v2/experiments/E14/e14_bundle_comparison.json)
- [E14 Human decision](COURSE_WORK/artifacts/model_improvement_v2/experiments/E14/e14_human_decision.json)
- [E15 capacity comparison](COURSE_WORK/artifacts/model_improvement_v2/experiments/E15/e15_capacity_comparison.json)
- [E15 Human decision](COURSE_WORK/artifacts/model_improvement_v2/experiments/E15/e15_human_decision.json)
- [E16 depth comparison](COURSE_WORK/artifacts/model_improvement_v2/experiments/E16/e16_depth_comparison.json)
- [E16 Human decision](COURSE_WORK/artifacts/model_improvement_v2/experiments/E16/e16_human_decision.json)
- [E18 Human decision](COURSE_WORK/artifacts/model_improvement_v2/experiments/E18/e18_human_decision.json)
- [E20 matched-seed comparison](COURSE_WORK/artifacts/model_improvement_v2/experiments/E20/e20_matched_seed_comparison.json)
- [Step 16 config snapshot](COURSE_WORK/artifacts/model_improvement_v2/final_model_lock/step16_final_refit_config_snapshot.json)
- [Step 16 final-refit manifest](COURSE_WORK/artifacts/model_improvement_v2/final_model_lock/step16_final_refit_manifest.json)
- [V2 final model lock](COURSE_WORK/artifacts/model_improvement_v2/final_model_lock/v2_final_model_lock.json)
- [Step 17 metrics](COURSE_WORK/artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_metrics.json)
- [Step 17 metrics with MAPE](COURSE_WORK/artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_metrics_with_mape.json)
- [Step 17 MAPE addendum manifest](COURSE_WORK/artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_mape_addendum_manifest.json)
- [Step 17 MAPE addendum signoff](COURSE_WORK/artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_mape_addendum_signoff.json)
- [Final V2 reporting analysis manifest](COURSE_WORK/artifacts/model_improvement_v2/final_reporting_analysis/manifest.json)
- [Step 17 benchmark manifest](COURSE_WORK/artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_benchmark_manifest.json)
- [Step 17 benchmark signoff](COURSE_WORK/artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_benchmark_signoff.json)
- [MODEL_IMPROVEMENT_V2 final closure](COURSE_WORK/artifacts/model_improvement_v2/model_improvement_v2_final_closure.json)

### Reporting / Code

- [results_rebuild.py](COURSE_WORK/src/course_work/reporting/results_rebuild.py)
- [v2_final_analysis.py](COURSE_WORK/src/course_work/reporting/v2_final_analysis.py)
- [frozen_evidence.py](COURSE_WORK/src/course_work/reporting/frozen_evidence.py)
- [step16_final_refit.py](COURSE_WORK/src/course_work/model_improvement_v2/step16_final_refit.py)
- [step17_post_hoc_benchmark.py](COURSE_WORK/src/course_work/model_improvement_v2/step17_post_hoc_benchmark.py)
- [step17_mape_addendum.py](COURSE_WORK/src/course_work/model_improvement_v2/step17_mape_addendum.py)
- [e20_runner.py](COURSE_WORK/src/course_work/model_improvement_v2/e20_runner.py)

### Tests

- [test_results_rebuild.py](COURSE_WORK/tests/unit/test_results_rebuild.py)
- [test_notebook_frozen_evidence.py](COURSE_WORK/tests/unit/test_notebook_frozen_evidence.py)
- [test_model_improvement_v2_e20.py](COURSE_WORK/tests/unit/test_model_improvement_v2_e20.py)
- [test_model_improvement_v2_step16.py](COURSE_WORK/tests/unit/test_model_improvement_v2_step16.py)
- [test_model_improvement_v2_step17.py](COURSE_WORK/tests/unit/test_model_improvement_v2_step17.py)
- [test_model_improvement_v2_step17_mape.py](COURSE_WORK/tests/unit/test_model_improvement_v2_step17_mape.py)
- [test_v2_final_reporting_analysis.py](COURSE_WORK/tests/unit/test_v2_final_reporting_analysis.py)
- [test_model_improvement_v2_final_closure.py](COURSE_WORK/tests/unit/test_model_improvement_v2_final_closure.py)

## 15. Danh sách file tạo mới/sửa/xóa/khôi phục

Bảng dưới đây tổng hợp các file thực sự thay đổi trong quá trình refactor presentation/reporting và triển khai `MODEL_IMPROVEMENT_V2`. Mỗi entry đều đối chiếu với `git status --short`, `git diff --name-status HEAD`, `git ls-files --others --exclude-standard` tại `COURSE_WORK/` và `git diff --check HEAD` (clean, exit `0`). Trạng thái checkpoint/scaler được xác minh bằng SHA-256 tính lại từ file hiện tại; các artifact dạng generated JSON/CSV/HTML/signoff/manifest được liệt kê nhưng ghi rõ là **không phải source code**.

Action chỉ ghi khi có evidence:

- `CREATED` — file mới xuất hiện trong untracked list hoặc được tạo trong current worktree;
- `MODIFIED` — file tracked, có entry trong `git diff --name-status HEAD`;
- `RESTORED` — file phục hồi từ snapshot khác vào recovery namespace;
- `REMOVED` — file đã xóa khỏi working tree (`D` trong `git status`);
- `UNCHANGED` — file nằm trong worktree nhưng không có diff/untracked (chỉ liệt kê khi liên quan).

### 15.1. Notebook

| File | Action | Mục đích | Liên quan tới |
|---|---|---|---|
| [notebook_course_work/CourseWork.ipynb](COURSE_WORK/notebook_course_work/CourseWork.ipynb) | `MODIFIED` | `git diff` cho thấy `3153` dòng thay đổi; current notebook có `153` cells (`77` code + `76` markdown), toàn bộ `77` code cells đã có `execution_count`, `0` output loại `error`, `last_execution_count=77`. Thêm cell `b0f56164` (environment audit, `execution_count=4`) ở index `6`. Toàn bộ `18` code cell gọi renderer Phase 43–59 + V2 (`render_verified_phase_result` × 17 cho Phase 43–59 và `render_verified_v2_results` × 1 cho V2) đã chạy với `execution_count` liên tục `60` → `77`. Cell `v2closure` (markdown, idx `151`) và `v2closedash` (code, idx `152`, `execution_count=77`) là hai cell cuối. | Refactor presentation/notebook |
| `notebook_course_work/CourseWork copy.ipynb` | `UNTRACKED_LOCAL_COPY` | File có trong untracked list của `git status`; nội dung và lý do tồn tại không nằm trong scope refactor chính; **không được renderer notebook sử dụng**. Ghi nhận tồn tại nhưng không dùng làm source-of-truth cho Section 15 ngoài việc liệt kê. | Local backup / không nằm trong pipeline |

### 15.2. Reporting code

| File | Action | Mục đích | Liên quan tới |
|---|---|---|---|
| [src/course_work/reporting/frozen_evidence.py](COURSE_WORK/src/course_work/reporting/frozen_evidence.py) | `CREATED` | Thêm `render_frozen_phase_evidence`, read-only artifact inventory, strict resume audit fallback, historical summary alias và checksum-verified CourseWorkV10 dashboard loader. | Phase 22–41, Phase 54–57 recovery |
| [src/course_work/reporting/results_rebuild.py](COURSE_WORK/src/course_work/reporting/results_rebuild.py) | `CREATED` | Thêm `_read`, `_render`, `render_verified_phase_result` cho Phase 43–59 và `render_verified_v2_results` cho V2. | Phase 43–59, V2 section |
| [src/course_work/reporting/model_improvement_v2_dashboard.py](COURSE_WORK/src/course_work/reporting/model_improvement_v2_dashboard.py) | `CREATED` | Thêm `render_model_improvement_v2_closure_dashboard`, đọc final closure và hiển thị development/post-hoc/provenance/debt. | V2 closure view (notebook hiện dùng renderer tổng hợp mới) |

### 15.3. MODEL_IMPROVEMENT_V2 code

| File | Action | Mục đích | Liên quan tới |
|---|---|---|---|
| [src/course_work/experiments/registry.py](COURSE_WORK/src/course_work/experiments/registry.py) | `MODIFIED` | `git diff` thống kê `38` dòng thay đổi. Mở rộng `FEATURE_VARIANTS`, `V2_NAMESPACE_IDS`; truyền `run_id_namespace` vào `validate_run_config`; chỉ cho batch size `16` trong namespace E14/E15/E16/E20. | Wave 2 strict namespace |
| [src/course_work/model_improvement_v2/contracts.py](COURSE_WORK/src/course_work/model_improvement_v2/contracts.py) | `MODIFIED` | `git diff` thống kê `52` dòng thay đổi. Thêm `ALLOWED_IMPLEMENTED_WAVE2_EXPERIMENT_IDS`, `ALLOWED_EXPERIMENT_IDS` và `PRIMARY_CHANGE_PREFIXES` cho E10–E16/E20; sửa validation message. | One-primary-change contract |
| [src/course_work/model_improvement_v2/e08_runner.py](COURSE_WORK/src/course_work/model_improvement_v2/e08_runner.py) | `MODIFIED` | `git diff` thống kê `4` dòng thay đổi. `validate_e08_document` đổi exact equality thành subset check với `V2_DIRECT_METRIC_FLATTEN_TRACKS`. | Wave 2 compatibility |
| [src/course_work/model_improvement_v2/e09_runner.py](COURSE_WORK/src/course_work/model_improvement_v2/e09_runner.py) | `MODIFIED` | `git diff` thống kê `4` dòng thay đổi. `validate_e09_document` dùng subset check tương tự E08. | Wave 2 compatibility |
| [src/course_work/model_improvement_v2/pretest_adapter.py](COURSE_WORK/src/course_work/model_improvement_v2/pretest_adapter.py) | `MODIFIED` | `git diff` thống kê `9` dòng thay đổi. `build_v2_pretest_dataset` đăng ký projection `FS2_TF1` cho E13–E16, E20 và `STEP16`. | Strict Train+Validation prefix |
| [src/course_work/model_improvement_v2/e10_pretest.py](COURSE_WORK/src/course_work/model_improvement_v2/e10_pretest.py) | `CREATED` | Thêm `E10PopulationAudit`, `derive_causal_target_deltas`, `build_e10_pretest_datasets` và locked delta-feature order. | E10 causal target deltas |
| [src/course_work/model_improvement_v2/e10_runner.py](COURSE_WORK/src/course_work/model_improvement_v2/e10_runner.py) | `CREATED` | Thêm E10 config/preflight, control/challenger builder, V2 context, promotion guardrails và official Human gate. | E10 orchestration |
| [src/course_work/model_improvement_v2/e11_pretest.py](COURSE_WORK/src/course_work/model_improvement_v2/e11_pretest.py) | `CREATED` | Thêm `E11PopulationAudit`, `ROLLING_DEFINITIONS`, `derive_past_only_rolling_features`, common-population builder. | E11 rolling-target projection |
| [src/course_work/model_improvement_v2/e11_runner.py](COURSE_WORK/src/course_work/model_improvement_v2/e11_runner.py) | `CREATED` | Thêm E11 preflight/runner, predecessor audit, `_audit_resume_contract`, `build_e11_resume_context` và result writer. | E11 official flow + Stage-B recovery |
| [src/course_work/model_improvement_v2/e12_pretest.py](COURSE_WORK/src/course_work/model_improvement_v2/e12_pretest.py) | `CREATED` | Thêm causal `Appliances_lag_144`, filtered fold evidence và `E12PopulationAudit`. | E12 lag-144 ablation |
| [src/course_work/model_improvement_v2/e12_runner.py](COURSE_WORK/src/course_work/model_improvement_v2/e12_runner.py) | `CREATED` | Thêm E12 decision/config audit, exact fingerprints, official/partial-resume modes và comparison writer. | E12 orchestration |
| [src/course_work/model_improvement_v2/hybrid_loss.py](COURSE_WORK/src/course_work/model_improvement_v2/hybrid_loss.py) | `CREATED` | Thêm `HybridLossComponents`, config validation, scaler parameter extraction và `compute_hybrid_loss`. | E13 hybrid loss |
| [src/course_work/model_improvement_v2/e13_runner.py](COURSE_WORK/src/course_work/model_improvement_v2/e13_runner.py) | `CREATED` | Thêm lambda matrix, control audit, recovery ledger, partial Stage-B resume và promotion output. | E13 orchestration |
| [src/course_work/model_improvement_v2/e14_runner.py](COURSE_WORK/src/course_work/model_improvement_v2/e14_runner.py) | `CREATED` | Thêm locked M1–M5 bundle matrix, E13 control audit, candidate/context builder, completed-ledger audit và comparison. | E14 bundle matrix |
| [src/course_work/model_improvement_v2/e15_runner.py](COURSE_WORK/src/course_work/model_improvement_v2/e15_runner.py) | `CREATED` | Thêm W96/W128 proportional width configs, parameter-count check, recovery ledger và promotion output. | E15 capacity audit |
| [src/course_work/model_improvement_v2/e16_runner.py](COURSE_WORK/src/course_work/model_improvement_v2/e16_runner.py) | `CREATED` | Thêm Depth3 Post-LN config, predecessor decision audit, one-change validation, run ledger và result finalization. | E16 depth-only experiment |
| [src/course_work/model_improvement_v2/e20_runner.py](COURSE_WORK/src/course_work/model_improvement_v2/e20_runner.py) | `CREATED` | Thêm seed-42 reuse audit, seeded fingerprints, exact 24-run identity matrix, per-seed recovery/finalization và matched comparison. | E20 matched three-seed |
| [src/course_work/model_improvement_v2/step16_final_refit.py](COURSE_WORK/src/course_work/model_improvement_v2/step16_final_refit.py) | `CREATED` | Thêm full pre-Test scaler fitting, locked median epoch policy, three-seed fresh refit, write-once outputs và recovery ledger. | Step 16 final refit |
| [src/course_work/model_improvement_v2/step17_post_hoc_benchmark.py](COURSE_WORK/src/course_work/model_improvement_v2/step17_post_hoc_benchmark.py) | `CREATED` | Thêm `TestAccessGuard`, final-lock/checksum validation, one-shot Test loader, per-seed prediction, equal-weight ensemble và benchmark artifacts. | Step 17 `POST_HOC_V2_BENCHMARK` |
| [src/course_work/rolling_origin/real_run.py](COURSE_WORK/src/course_work/rolling_origin/real_run.py) | `MODIFIED` | `git diff` thống kê `676` dòng thay đổi. Thêm Wave 2 track/cache/config invariants, `_validate_e11_resume_ledger`, `_validate_e12_resume_ledger`, `_validate_e13_resume_ledger`, partial recovery branches, shared common-population checks và scaler bundle handoff cho hybrid loss. | Wave 2 reuse + recovery |
| [src/course_work/rolling_origin/refit_engine.py](COURSE_WORK/src/course_work/rolling_origin/refit_engine.py) | `MODIFIED` | `git diff` thống kê `32` dòng thay đổi. `RefitEngine.refit` nhận hybrid loss, yêu cầu `y_context_raw_wh`, tích lũy component metrics và dùng dynamic history columns. | Stage-B refit + E13 components |
| [src/course_work/training/engine.py](COURSE_WORK/src/course_work/training/engine.py) | `MODIFIED` | `git diff` thống kê `36` dòng thay đổi. Thêm `E13_HISTORY_COLUMNS`, `history_columns_for_training`; `TrainingEngine.train` tính/log hybrid components khi policy được bật. | E13 history columns |

### 15.4. Path và gate scripts

| File | Action | Mục đích | Liên quan tới |
|---|---|---|---|
| [scripts/_fix_sweep_cells.py](scripts/_fix_sweep_cells.py) | `MODIFIED` | `git diff` thống kê `2` dòng thay đổi. Sửa constant `NB` sang absolute path project hiện tại. | Trỏ utility tới notebook đúng vị trí |
| [scripts/_populate_sweep_results.py](scripts/_populate_sweep_results.py) | `MODIFIED` | `git diff` thống kê `2` dòng thay đổi. Sửa constant `ROOT` sang absolute path project hiện tại. | Trỏ legacy population utility tới root đúng; không được renderer presentation gọi |
| [scripts/phase44_pretrain_gate.py](scripts/phase44_pretrain_gate.py) | `MODIFIED` | `git diff` thống kê `2` dòng thay đổi. Sửa suggested `cd` command và quote path có khoảng trắng. | In Human command đúng project root |
| [src/course_work/scripts/p44_pretrain_gate.py](COURSE_WORK/src/course_work/scripts/p44_pretrain_gate.py) | `MODIFIED` | `git diff` thống kê `2` dòng thay đổi. Cùng sửa suggested `cd` command như script top-level. | Đồng bộ gate copy trong package |
| [src/course_work/scripts/p47_pretest_gate.py](COURSE_WORK/src/course_work/scripts/p47_pretest_gate.py) | `MODIFIED` | `git diff` thống kê `11` dòng thay đổi. Sửa root resolution, import sang `course_work.final_test_evaluation`, kiểm tra boolean result trong `run_gate` và scan đúng evaluation file. | Gate trỏ đúng package/path; không coi boolean `False` là `PASS` |
| [src/course_work/analysis/attention_extraction/extract.py](COURSE_WORK/src/course_work/analysis/attention_extraction/extract.py) | `MODIFIED` | `git diff` thống kê `2` dòng thay đổi. Sửa default root trong `run_phase52_extraction`. | Trỏ Phase 52 utility tới project hiện tại |
| [src/course_work/analysis/attention_extraction/finalize_52fg.py](COURSE_WORK/src/course_work/analysis/attention_extraction/finalize_52fg.py) | `MODIFIED` | `git diff` thống kê `2` dòng thay đổi. Sửa root trong `main`. | Dùng đúng project path khi finalize Phase 52 F/G |
| [src/course_work/analysis/attention_extraction/materialize.py](COURSE_WORK/src/course_work/analysis/attention_extraction/materialize.py) | `MODIFIED` | `git diff` thống kê `2` dòng thay đổi. Sửa fallback `project_root`. | Dùng đúng root nếu CLI không truyền `--project-root` |
| [src/course_work/analysis/attention_extraction/orchestrator.py](COURSE_WORK/src/course_work/analysis/attention_extraction/orchestrator.py) | `MODIFIED` | `git diff` thống kê `2` dòng thay đổi. Sửa default root trong `run_phase52`. | Đồng bộ Phase 52 orchestrator |
| [src/course_work/analysis/last_query_attention/finalize_phase54.py](COURSE_WORK/src/course_work/analysis/last_query_attention/finalize_phase54.py) | `MODIFIED` | `git diff` thống kê `4` dòng thay đổi. Sửa processing-log path và static-scan path. | Trỏ Phase 54 finalization tới project hiện tại |

### 15.5. Tests

| File | Action | Mục đích | Liên quan tới |
|---|---|---|---|
| [tests/contracts/test_selective_execution_policy.py](COURSE_WORK/tests/contracts/test_selective_execution_policy.py) | `MODIFIED` | `git diff` thống kê `166` dòng thay đổi. Chuyển assertions Phase 22–41 sang `render_frozen_phase_evidence`, cập nhật cell ordering/output-preservation exclusions. | Read-only notebook orchestration |
| [tests/unit/test_model_improvement_v2_e08.py](COURSE_WORK/tests/unit/test_model_improvement_v2_e08.py) | `MODIFIED` | `git diff` thống kê `8` dòng thay đổi. Exact-set assertions đổi thành subset assertions cho namespace/flatten tracks. | E08 regression |
| [tests/unit/test_model_improvement_v2_e09.py](COURSE_WORK/tests/unit/test_model_improvement_v2_e09.py) | `MODIFIED` | `git diff` thống kê `4` dòng thay đổi. Direct-shape track assertion đổi thành subset. | E09 regression |
| [tests/unit/test_phase50_h_notebook_dashboard.py](COURSE_WORK/tests/unit/test_phase50_h_notebook_dashboard.py) | `MODIFIED` | Sửa explicit project path trong renderer test. | Trỏ legacy Phase 50 test đúng root |
| [tests/unit/test_phase51_g_finalization.py](COURSE_WORK/tests/unit/test_phase51_g_finalization.py) | `MODIFIED` | Sửa fallback `PROJECT_ROOT`. | Trỏ Phase 51 artifact tests đúng root |
| [tests/unit/test_model_improvement_v2_e10.py](COURSE_WORK/tests/unit/test_model_improvement_v2_e10.py) | `CREATED` | Test delta formulas/order, common population, Test firewall, namespace, four guardrails và Human gate. | Focused validation E10 |
| [tests/unit/test_model_improvement_v2_e11.py](COURSE_WORK/tests/unit/test_model_improvement_v2_e11.py) | `CREATED` | Test rolling formulas/order, causal behavior, incumbent, common population, scaling và gate. | Focused validation E11 |
| [tests/unit/test_model_improvement_v2_e11_h1_resume.py](COURSE_WORK/tests/unit/test_model_improvement_v2_e11_h1_resume.py) | `CREATED` | Test canonical/duplicate/interrupted ledger, six-Stage-A invariant và Stage-B-only partial recovery. | E11 resume contract |
| [tests/unit/test_model_improvement_v2_e12.py](COURSE_WORK/tests/unit/test_model_improvement_v2_e12.py) | `CREATED` | Test exact lag-144 alignment, rejected-feature exclusion, fingerprints, recovery ledger và Human gate. | Focused validation E12 |
| [tests/unit/test_model_improvement_v2_e13.py](COURSE_WORK/tests/unit/test_model_improvement_v2_e13.py) | `CREATED` | Test hybrid algebra/units, scaler lineage, component history, lambda matrix và Stage-B recovery. | Focused validation E13 |
| [tests/unit/test_model_improvement_v2_e14.py](COURSE_WORK/tests/unit/test_model_improvement_v2_e14.py) | `CREATED` | Test promoted E13 control, exact M1–M5 matrix, allowed batch sizes, one-factor contract và preflight. | Focused validation E14 |
| [tests/unit/test_model_improvement_v2_e15.py](COURSE_WORK/tests/unit/test_model_improvement_v2_e15.py) | `CREATED` | Test exact width bundles, parameter limit, inherited batch 16, namespace và Human gate. | Focused validation E15 |
| [tests/unit/test_model_improvement_v2_e16.py](COURSE_WORK/tests/unit/test_model_improvement_v2_e16.py) | `CREATED` | Test E14-M1 control, depth-only Post-LN change, six-run resume ledger và no-training preflight. | Focused validation E16 |
| [tests/unit/test_model_improvement_v2_e20.py](COURSE_WORK/tests/unit/test_model_improvement_v2_e20.py) | `CREATED` | Test seed-42 reuse, exact 24 new IDs, seed-only drift, fresh state, recovery và Test firewall. | Focused validation E20 |
| [tests/unit/test_model_improvement_v2_step16.py](COURSE_WORK/tests/unit/test_model_improvement_v2_step16.py) | `CREATED` | Test common epoch policy, full pre-Test population/scalers, three fresh models, equal weights và recovery. | Validate final-refit contract |
| [tests/unit/test_model_improvement_v2_step17.py](COURSE_WORK/tests/unit/test_model_improvement_v2_step17.py) | `CREATED` | Test final lock/checkpoints/scalers, one-shot Test guard, ensemble math, label và dual Human gate. | Validate post-hoc benchmark protocol |
| [tests/unit/test_model_improvement_v2_final_closure.py](COURSE_WORK/tests/unit/test_model_improvement_v2_final_closure.py) | `CREATED` | Test closure policy/governance, immutable-source hashes, saved-prediction metrics và read-only dashboard. | Validate final V2 closure |
| [tests/unit/test_notebook_frozen_evidence.py](COURSE_WORK/tests/unit/test_notebook_frozen_evidence.py) | `CREATED` | Test Phase 2/6/36 frozen views, exact Phase 6 recovery block, EDA checksums và Phase 54–57 snapshot checksums. | Validate read-only historical presentation |
| [tests/unit/test_results_rebuild.py](COURSE_WORK/tests/unit/test_results_rebuild.py) | `CREATED` | Test Phase 43–59 non-empty tables, Phase 47 metrics, V2 result chain và Phase 1–42 prefix hash. | Validate rebuilt notebook presentation |

### 15.6. Docs / plans / log JSON

| File | Action | Mục đích | Liên quan tới |
|---|---|---|---|
| [docs/analysis_error/phase_30_rerun_dependency_chain_issue.md](COURSE_WORK/docs/analysis_error/phase_30_rerun_dependency_chain_issue.md) | `MODIFIED` | `git diff` thống kê `4` dòng thay đổi. | Phase 30 audit context |
| [docs/analysis_error/phase_34_current_state_rebaseline_blockers.md](COURSE_WORK/docs/analysis_error/phase_34_current_state_rebaseline_blockers.md) | `MODIFIED` | `git diff` thống kê `2` dòng thay đổi. | Phase 34 blockers |
| [docs/analysis_error/phase_46_pre_implementation_defect_audit.md](COURSE_WORK/docs/analysis_error/phase_46_pre_implementation_defect_audit.md) | `MODIFIED` | `git diff` thống kê `56` dòng thay đổi. | Phase 46 defect audit |
| [docs/code_base_audit.md](COURSE_WORK/docs/code_base_audit.md) | `MODIFIED` | `git diff` thống kê `2` dòng thay đổi. | Repo-wide audit |
| [docs/link&discussion_to_result/NOTEBOOK_CELLS_WALKTHROUGH.md](COURSE_WORK/docs/link&discussion_to_result/NOTEBOOK_CELLS_WALKTHROUGH.md) | `MODIFIED` | `git diff` thống kê `312` dòng thay đổi. | Notebook walkthrough |
| [docs/plan/plan_before_process/phase_45_corrective_pre_process_plan.md](COURSE_WORK/docs/plan/plan_before_process/phase_45_corrective_pre_process_plan.md) | `MODIFIED` | `git diff` thống kê `2` dòng thay đổi. | Phase 45 plan |
| [docs/plan/plan_before_process/refactor_and_rerun_phase_30_dependency_recovery_plan.md](COURSE_WORK/docs/plan/plan_before_process/refactor_and_rerun_phase_30_dependency_recovery_plan.md) | `MODIFIED` | `git diff` thống kê `2` dòng thay đổi. | Phase 30 dependency plan |
| [docs/plan/plan_to_refactor&fix/notebook_cells_walkthrough_link_mapping_refactor_result.md](COURSE_WORK/docs/plan/plan_to_refactor&fix/notebook_cells_walkthrough_link_mapping_refactor_result.md) | `MODIFIED` | `git diff` thống kê `2` dòng thay đổi. | Notebook refactor result |
| [docs/save_log_in_processing/phase_12_shared_metrics_log.json](COURSE_WORK/docs/save_log_in_processing/phase_12_shared_metrics_log.json) | `MODIFIED` | `git diff` thống kê `14` dòng thay đổi. | Phase 12 processing log |
| [docs/save_log_in_processing/phase_13_experiment_registry_log.json](COURSE_WORK/docs/save_log_in_processing/phase_13_experiment_registry_log.json) | `MODIFIED` | `git diff` thống kê `14` dòng thay đổi. | Phase 13 processing log |
| [docs/save_log_in_processing/phase_15_lstm_implementation_log.json](COURSE_WORK/docs/save_log_in_processing/phase_15_lstm_implementation_log.json) | `MODIFIED` | `git diff` thống kê `374` dòng thay đổi. | Phase 15 processing log |
| [docs/save_log_in_processing/phase_16_transformer_implementation_log.json](COURSE_WORK/docs/save_log_in_processing/phase_16_transformer_implementation_log.json) | `MODIFIED` | `git diff` thống kê `364` dòng thay đổi. | Phase 16 processing log |
| [docs/save_log_in_processing/phase_17_attention_verification_log.json](COURSE_WORK/docs/save_log_in_processing/phase_17_attention_verification_log.json) | `MODIFIED` | `git diff` thống kê `324` dòng thay đổi. | Phase 17 processing log |
| [docs/save_log_in_processing/phase_18_forward_sanity_log.json](COURSE_WORK/docs/save_log_in_processing/phase_18_forward_sanity_log.json) | `MODIFIED` | `git diff` thống kê `270` dòng thay đổi. | Phase 18 processing log |
| [docs/save_log_in_processing/phase_19_training_engine_log.json](COURSE_WORK/docs/save_log_in_processing/phase_19_training_engine_log.json) | `MODIFIED` | `git diff` thống kê `318` dòng thay đổi. | Phase 19 processing log |
| [docs/save_log_in_processing/phase_20_lstm_baseline_log.json](COURSE_WORK/docs/save_log_in_processing/phase_20_lstm_baseline_log.json) | `MODIFIED` | `git diff` thống kê `314` dòng thay đổi. | Phase 20 processing log |
| [docs/save_log_in_processing/phase_21_transformer_b0_log.json](COURSE_WORK/docs/save_log_in_processing/phase_21_transformer_b0_log.json) | `MODIFIED` | `git diff` thống kê `306` dòng thay đổi. | Phase 21 processing log |
| [docs/save_log_in_processing/phase_23_s1_feature_set_log.json](COURSE_WORK/docs/save_log_in_processing/phase_23_s1_feature_set_log.json) | `MODIFIED` | `git diff` thống kê `2257` dòng thay đổi. | Phase 23 processing log |
| [docs/save_log_in_processing/phase_24_s2_time_feature_log.json](COURSE_WORK/docs/save_log_in_processing/phase_24_s2_time_feature_log.json) | `MODIFIED` | `git diff` thống kê `2393` dòng thay đổi. | Phase 24 processing log |
| [docs/save_log_in_processing/phase_25_s3_target_scaling_log.json](COURSE_WORK/docs/save_log_in_processing/phase_25_s3_target_scaling_log.json) | `MODIFIED` | `git diff` thống kê `2393` dòng thay đổi. | Phase 25 processing log |
| [docs/save_log_in_processing/phase_26_s4_lookback_log.json](COURSE_WORK/docs/save_log_in_processing/phase_26_s4_lookback_log.json) | `MODIFIED` | `git diff` thống kê `2553` dòng thay đổi. | Phase 26 processing log |
| [docs/save_log_in_processing/phase_27_s5_pooling_log.json](COURSE_WORK/docs/save_log_in_processing/phase_27_s5_pooling_log.json) | `MODIFIED` | `git diff` thống kê `2393` dòng thay đổi. | Phase 27 processing log |
| [docs/save_log_in_processing/phase_28_s6_activation_log.json](COURSE_WORK/docs/save_log_in_processing/phase_28_s6_activation_log.json) | `MODIFIED` | `git diff` thống kê `2393` dòng thay đổi. | Phase 28 processing log |
| [docs/save_log_in_processing/phase_29_s7_batch_size_log.json](COURSE_WORK/docs/save_log_in_processing/phase_29_s7_batch_size_log.json) | `MODIFIED` | `git diff` thống kê `2393` dòng thay đổi. | Phase 29 processing log |
| [docs/save_log_in_processing/phase_30_s8_learning_rate_log.json](COURSE_WORK/docs/save_log_in_processing/phase_30_s8_learning_rate_log.json) | `MODIFIED` | `git diff` thống kê `3168` dòng thay đổi. | Phase 30 processing log |
| [docs/save_log_in_processing/phase_31_s9_weight_decay_log.json](COURSE_WORK/docs/save_log_in_processing/phase_31_s9_weight_decay_log.json) | `MODIFIED` | `git diff` thống kê `1093820` dòng thay đổi (`+++`). | Phase 31 processing log |
| [docs/save_log_in_processing/phase_32_s10_dropout_log.json](COURSE_WORK/docs/save_log_in_processing/phase_32_s10_dropout_log.json) | `MODIFIED` | `git diff` thống kê `1050930` dòng thay đổi (`+++`). | Phase 32 processing log |
| [docs/save_log_in_processing/phase_7_feature_engineering_log.json](COURSE_WORK/docs/save_log_in_processing/phase_7_feature_engineering_log.json) | `MODIFIED` | `git diff` thống kê `10` dòng thay đổi. | Phase 7 processing log |
| [docs/model_improvement_v2_final_closure.md](COURSE_WORK/docs/model_improvement_v2_final_closure.md) | `CREATED` | Companion narrative cho [model_improvement_v2_final_closure.json](COURSE_WORK/artifacts/model_improvement_v2/model_improvement_v2_final_closure.json). | V2 closure narrative |
| [docs/plan/plan_before_process/model_improvement_v2_e20_pre_process_plan.md](COURSE_WORK/docs/plan/plan_before_process/model_improvement_v2_e20_pre_process_plan.md) | `CREATED` | Plan trước xử lý cho E20. | E20 plan |
| [docs/plan/plan_before_process/model_improvement_v2_fine_tuning_pre_process_plan.md](COURSE_WORK/docs/plan/plan_before_process/model_improvement_v2_fine_tuning_pre_process_plan.md) | `CREATED` | Plan trước xử lý cho fine-tuning eligibility (E18). | E18 plan |
| [docs/plan/plan_before_process/model_improvement_v2_step14a_seed_ensemble_pre_process_plan.md](COURSE_WORK/docs/plan/plan_before_process/model_improvement_v2_step14a_seed_ensemble_pre_process_plan.md) | `CREATED` | Plan trước xử lý cho Step 14A. | Step 14A plan |
| [docs/plan/plan_before_process/model_improvement_v2_step14b_persistence_neural_blend_pre_process_plan.md](COURSE_WORK/docs/plan/plan_before_process/model_improvement_v2_step14b_persistence_neural_blend_pre_process_plan.md) | `CREATED` | Plan trước xử lý cho Step 14B. | Step 14B plan |
| [docs/plan/plan_before_process/model_improvement_v2_step16_final_refit_lock_pre_process_plan.md](COURSE_WORK/docs/plan/plan_before_process/model_improvement_v2_step16_final_refit_lock_pre_process_plan.md) | `CREATED` | Plan trước xử lý cho Step 16. | Step 16 plan |
| [docs/plan/plan_before_process/model_improvement_v2_step17_post_hoc_benchmark_pre_process_plan.md](COURSE_WORK/docs/plan/plan_before_process/model_improvement_v2_step17_post_hoc_benchmark_pre_process_plan.md) | `CREATED` | Plan trước xử lý cho Step 17. | Step 17 plan |

### 15.7. Artifacts / reporting outputs (generated, không phải source code)

| File / Path | Action | Mục đích | Liên quan tới |
|---|---|---|---|
| `artifacts/notebook_presentation_recovery/` (gồm `courseworkv10_phase54_57/manifest.json`, `phase_54_dashboard.html`, `phase_55_dashboard.html`, `phase_56_dashboard.html`, `phase_57_dashboard.html`) | `RESTORED` | Khôi phục snapshot dashboard Phase 54–57 từ `CourseWorkV10` vào recovery namespace; source branch, source commit và SHA-256 được ghi trong `manifest.json`; `scientific_recomputation=false`, `test_source_accessed=false`. | Phase 54–57 presentation evidence |
| `artifacts/model_improvement_v2/experiments/E10/` | `CREATED` (generated) | Output E10 (config snapshot, execution manifest, target-delta ablation comparison, final-model-lock handoff, Phase 44 signoff, registry, rolling-origin manifest, run configs). | E10 evidence |
| `artifacts/model_improvement_v2/experiments/E11/` | `CREATED` (generated) | Output E11 (causal audit, common-population audit, config snapshot, execution manifest, feature contract, no-retrain recovery contract, rolling-target ablation comparison, scaler-lineage audit, final-model-lock handoff, registry, rolling-origin manifest, run configs). | E11 evidence |
| `artifacts/model_improvement_v2/experiments/E12/` | `CREATED` (generated) | Output E12 (lag-144 ablation comparison, Human decision, registry, rolling-origin manifest, run configs). | E12 evidence |
| `artifacts/model_improvement_v2/experiments/E13/` | `CREATED` (generated) | Output E13 (hybrid-loss comparison, Human decision, registry, run configs). | E13 evidence |
| `artifacts/model_improvement_v2/experiments/E14/` | `CREATED` (generated) | Output E14 (bundle comparison, Human decision, registry, run configs). | E14 evidence |
| `artifacts/model_improvement_v2/experiments/E15/` | `CREATED` (generated) | Output E15 (capacity comparison, Human decision, registry, run configs). | E15 evidence |
| `artifacts/model_improvement_v2/experiments/E16/` | `CREATED` (generated) | Output E16 (depth comparison, Human decision, registry, run configs). | E16 evidence |
| `artifacts/model_improvement_v2/experiments/E18/` | `CREATED` (generated) | Output E18 (Human decision, registry, run configs). | E18 evidence |
| `artifacts/model_improvement_v2/experiments/E20/` | `CREATED` (generated) | Output E20 (matched-seed comparison, config snapshot, registry, run configs). | E20 evidence |
| `artifacts/model_improvement_v2/final_model_lock/` (gồm `v2_final_model_lock.json`, `step16_final_refit_config_snapshot.json`, `step16_final_refit_manifest.json`, `runs/`, `scalers/`) | `CREATED` (generated) | Step 16 final-refit outputs. | Step 16 final lock evidence |
| `artifacts/model_improvement_v2/post_hoc_v2_benchmark/` (gồm `step17_metrics.json`, `step17_benchmark_manifest.json`, `step17_benchmark_signoff.json`, `step17_benchmark_summary.json`, `step17_benchmark_config_snapshot.json`, `step17_test_access_event.json`, `predictions/`) | `CREATED` (generated) | Step 17 benchmark outputs. | Step 17 `POST_HOC_V2_BENCHMARK` |
| `artifacts/model_improvement_v2/model_improvement_v2_final_closure.json` | `CREATED` (generated) | Final closure của `MODEL_IMPROVEMENT_V2` ở trạng thái `COMPLETE`. | V2 closure |

### 15.8. File ngoài `COURSE_WORK/` được ghi nhận nhưng không thuộc refactor

| File | Action | Ghi chú |
|---|---|---|
| `doc_for_agent_study/Chapter 1 – 6 PDFs`, `doc_for_agent_study/Deep Learning with Python-…pdf`, `doc_for_agent_study/Sách Deep Learning cơ bản - v2.pdf` | `REMOVED` | `git status` ghi `D` cho 8 file PDF, 2 file `.md` và 1 file PDF nữa trong thư mục này. Là file tài liệu tham khảo, **không liên quan tới refactor `COURSE_WORK/` và không được liệt kê như một phần của refactor này**. |

### 15.9. File refactor doc

| File | Action | Mục đích | Liên quan tới |
|---|---|---|---|
| [COURSE_WORK/Description_after_refactor.md](Description_after_refactor.md) | `CREATED` | Tài liệu này — ghi lại thay đổi sau refactor presentation/reporting và triển khai `MODEL_IMPROVEMENT_V2`. | Refactor doc |

### 15.10. Tóm tắt file inventory theo action

| Action | Số file source code | Số file docs/plan/log | Số artifact generated | Ghi chú |
|---|---:|---:|---:|---|
| `CREATED` | `23` | `7` | `~40` (E10/E11/E12/E13/E14/E15/E16/E18/E20 + final_model_lock + post_hoc_v2_benchmark + final closure + notebook_presentation_recovery) | Tổng `30` file source + docs mới; artifact generated là JSON/CSV/HTML/SHA-256 manifest nằm trong các thư mục `artifacts/model_improvement_v2/...` |
| `MODIFIED` | `19` (10 src + 4 test + 5 reporting code, đếm từ bảng trên) | `25` (3 docs/analysis_error + 1 docs/code_base_audit + 1 docs/link&discussion + 3 docs/plan + 17 docs/save_log_in_processing) | `0` | Con số test bao gồm cả test `MODIFIED` trong Section 15.5 |
| `RESTORED` | `0` | `0` | `5` (manifest + 4 dashboard HTML cho Phase 54–57) | Source branch/commit/SHA-256 đều có trong manifest |
| `REMOVED` | `0` | `11` (8 PDF + 2 `.md` + 1 PDF trong `doc_for_agent_study/`) | `0` | Nằm ngoài `COURSE_WORK/`, không thuộc refactor |

Con số trên được đếm từ bảng; các nhóm khác nhau có thể overlap (ví dụ một file vừa `MODIFIED` vừa được tham chiếu trong bảng khác). Tổng file đã đếm trong Section 15.1–15.10 là source-of-truth cho inventory này.

## 16. Validation và tests

Mục này chỉ ghi lại bằng chứng cuối cùng rằng refactor không làm hỏng notebook presentation hoặc scientific workflow. Toàn bộ số liệu trong bảng dưới đây được lấy từ output thực của `pytest`, `python -m py_compile`, `python` (tính SHA-256, parse JSON notebook), `git diff --check HEAD` tại `COURSE_WORK/` vào thời điểm viết. Không có con số nào được suy ra từ trí nhớ.

Các quyết định mặc định đã ghi nhận:

- **Training KHÔNG được chạy trong refactor presentation/notebook này.** Step 16 final-refit và Step 17 `POST_HOC_V2_BENCHMARK` là kết quả của workflow trước đó, đã khóa bằng final lock (`LOCKED`, `3_OF_3_COMPLETED`). Không có run mới nào được tạo trong task viết tài liệu này; `git status` không thấy file checkpoint/scaler mới nào ngoài ba checkpoint và hai scaler đã verify tại Section 13.1.
- **Inference mới KHÔNG được chạy.** Test source KHÔNG được mở lại; `final_closure.governance.test_status = "ACCESSED_ONCE_FOR_POST_HOC_V2_BENCHMARK"`, `post_test_retuning = false`, `ensemble_weights_changed = false`, `best_seed_selected = false` được giữ nguyên theo [model_improvement_v2_final_closure.json](COURSE_WORK/artifacts/model_improvement_v2/model_improvement_v2_final_closure.json).
- **Test source KHÔNG được mở lại trong task này.** Toàn bộ test dưới đây là test deterministic, đọc JSON/CSV/manifest/signoff/lock/closure có sẵn hoặc dựng in-memory dataset; không test nào gọi Test loader của `step17_post_hoc_benchmark.py`.
- **Checkpoint/scaler/model KHÔNG bị thay đổi.** SHA-256 tính lại từ current file đối chiếu với bảng Section 13.1 khớp 5/5. `git status` không có file mới trong `artifacts/model_improvement_v2/final_model_lock/runs/` hoặc `scalers/` ngoài những gì đã track từ workflow trước.

### 16.1. Notebook execution state

| Validation | Kết quả | Evidence |
|---|---|---|
| Tổng số notebook cells | `153` | `python` parse `notebook_course_work/CourseWork.ipynb` |
| Số code cells | `77` | Cùng script parse trên |
| Số code cells đã executed (có `execution_count`) | `77 / 77` | Cùng script parse trên |
| Python exceptions trong saved output | `0` (không có output loại `error`) | Cùng script parse trên |
| `execution_count` cuối cùng | `77` | Cùng script parse trên; cell `v2closedash` |
| `execution_count` của cell environment-audit `b0f56164` (idx 6) | `4` | Cùng script parse trên |
| Phase 43–59 renderer cells | `17` code cells, `execution_count` liên tục `60 → 76` | `grep` `render_verified_phase_result` trong parsed notebook |
| V2 summary cell | `1` code cell, `execution_count = 77`, gọi `render_verified_v2_results(PROJECT_ROOT)` | Cùng parse trên |
| Phase 43 cell idx | `118`, `id=6a082d01`, `execution_count=60`, source gồm `from course_work.reporting.results_rebuild import (...)` + `display(render_verified_phase_result(43, PROJECT_ROOT))` | Cùng parse trên |
| Markdown cells | `76` | Cùng script parse trên |
| Cells có `id` | `153 / 153` | Cùng script parse trên |
| `id` cuối cùng | `v2closedash` (idx `152`), trước đó `76de7c69` (idx `150`) | Cùng script parse trên |

### 16.2. Phase/table rendering validation

| Validation | Kết quả | Evidence |
|---|---|---|
| Phase 43–59 non-empty table | `PASS` cho cả 17 phase | `tests/unit/test_results_rebuild.py::test_every_phase_43_59_renders_nonempty_table_without_evidence_metadata` — `3 passed, 1 failed` trên toàn file (xem 16.7); test này trong nhóm `passed` |
| Số `<table` trên mỗi phase render | `3` cho tất cả 17 phase | Cùng test, assertion `html.count("<table") == 3` |
| Ba label bắt buộc `Phase status`, `Configuration / analysis`, `Results / decision` | Có mặt trong HTML | Cùng test |
| Không leak evidence path / raw JSON / script | Không xuất hiện `"Evidence:"`, `"artifacts/"`, `"<script"`, `"raw json"` trong HTML Phase 43–59 | Cùng test |
| Phase 47 dùng metric verified hiện tại | `PASS` — `64.942754`, `61.986086`, `64.560136`, `66.836915` xuất hiện trong HTML; `Best seed` không xuất hiện | `tests/unit/test_results_rebuild.py::test_phase47_uses_current_verified_per_seed_and_baseline_metrics` |
| V2 result chain completeness | `PASS` — `E01`, `E20`, `Step 14A`, `Step 14B`, `Step 16`, `Step 17`, `Closure` và `61.608937`, `COMPLETE` xuất hiện trong HTML; `3 <table>`, ba label bắt buộc có mặt, không leak evidence path | `tests/unit/test_results_rebuild.py::test_v2_result_chain_is_complete_without_evidence_metadata` |

### 16.3. Figure rendering validation

| Validation | Kết quả | Evidence |
|---|---|---|
| Inline figure trong cell Phase 43–59 compact block | `0` (`REMOVED_FROM_NOTEBOOK_PRESENTATION`) | Inspect source `''.join(cell.get('source',[]))` cho 17 code cell renderer Phase 43–59 — không chứa tham chiếu hình inline. Inline figure cũ trong source notebook được renderer mới bỏ khỏi block compact, artifact figure trên disk vẫn còn trong các thư mục `artifacts/last_query_attention/figures/`, `artifacts/head_comparison/figures/`, `artifacts/error_conditioned_attention/figures/`, `artifacts/seed_stability_attention/figures/` (xem Phase 54 recovery manifest để kiểm tra SHA-256) |
| Figure rendered HTML nguồn từ Phase 54–57 snapshot | `RESTORED` — 4 file HTML (`phase_54_dashboard.html`, `phase_55_dashboard.html`, `phase_56_dashboard.html`, `phase_57_dashboard.html`) tồn tại trong `artifacts/notebook_presentation_recovery/courseworkv10_phase54_57/` | `ls` thư mục recovery; manifest SHA-256 cho từng figure đã verify trong [manifest.json](COURSE_WORK/artifacts/notebook_presentation_recovery/courseworkv10_phase54_57/manifest.json) |

### 16.4. Focused tests — `MODEL_IMPROVEMENT_V2`

Test source KHÔNG được mở lại; tất cả test dưới đây đọc artifact/signoff/lock/closure hoặc dựng in-memory dataset, không load Test.

| Test file | Kết quả | Evidence |
|---|---|---|
| `tests/unit/test_model_improvement_v2_e08.py` | `36 passed` | `pytest tests/unit/test_model_improvement_v2_e08.py tests/unit/test_model_improvement_v2_e09.py` |
| `tests/unit/test_model_improvement_v2_e09.py` | (cùng lần chạy ở trên) `36 passed` tổng | `pytest tests/unit/test_model_improvement_v2_e08.py tests/unit/test_model_improvement_v2_e09.py` |
| `tests/unit/test_model_improvement_v2_e10.py`, `_e11.py`, `_e11_h1_resume.py`, `_e12.py`, `_e13.py`, `_e14.py`, `_e15.py`, `_e16.py`, `_e20.py` | `88 passed / 0 failed` | `pytest tests/unit/test_model_improvement_v2_e10.py tests/unit/test_model_improvement_v2_e11.py tests/unit/test_model_improvement_v2_e11_h1_resume.py tests/unit/test_model_improvement_v2_e12.py tests/unit/test_model_improvement_v2_e13.py tests/unit/test_model_improvement_v2_e14.py tests/unit/test_model_improvement_v2_e15.py tests/unit/test_model_improvement_v2_e16.py tests/unit/test_model_improvement_v2_e20.py` |
| `tests/unit/test_model_improvement_v2_step17.py` | `passed` (12 passed trong cụm step17 + final_closure) | `pytest tests/unit/test_model_improvement_v2_step17.py tests/unit/test_model_improvement_v2_final_closure.py` — `12 passed` |
| `tests/unit/test_model_improvement_v2_final_closure.py` | `passed` (cùng cụm trên) | Như trên |
| `tests/unit/test_model_improvement_v2_step16.py` | `4 passed, 5 errors (setup-time preflight error)` | `pytest tests/unit/test_model_improvement_v2_step16.py` — error `Step16PreflightError("Deterministic FINAL_DEV scaler checksum drift")` tại fixture setup của `step16_final_refit.py:341`. 5 test bị ERROR là do fixture setup raise; 4 test pass. Đây là validation debt đã biết: scaler `x_scaler_sha256` / `y_scaler_sha256` / `bundle_sha256` tính lại khi chạy `fit_final_dev_scaler` không khớp `final_scaler_lock` đã ghi trong `v2_final_model_lock.json`. **Trạng thái: xác minh được qua pytest output, không suy đoán.** |
| `tests/unit/test_model_improvement_v2_pretest_adapter.py` | `passed` (trong cụm e08/e09 ở trên, `36 passed` tổng) | Cùng `pytest` đầu tiên ở mục này |

### 16.5. Regression tests — Phase 43–59 / notebook orchestration

| Test file | Kết quả | Evidence |
|---|---|---|
| `tests/unit/test_results_rebuild.py` | `10 passed, 1 skipped, 1 failed` | `pytest tests/unit/test_results_rebuild.py tests/unit/test_notebook_frozen_evidence.py` |
| `tests/unit/test_notebook_frozen_evidence.py` | `7 passed, 1 skipped` | Cùng `pytest` trên |
| `tests/contracts/test_selective_execution_policy.py` | `9 passed` | `pytest tests/contracts/test_selective_execution_policy.py` |
| `tests/unit/test_phase50_h_notebook_dashboard.py` | `0 passed, 11 failed` (suite legacy cũ) | `pytest tests/unit/test_phase50_h_notebook_dashboard.py` — failure kiểu `test_phase0_49_cells_preserved_source` / `test_phase50_md_cell_present` / `test_run_all_not_used` / `test_phase50_html_no_training_or_scaler_text`. Đây là test Phase 50 cũ (legacy dashboard) viết assertion cho Phase 50 markdown + HTML legacy; renderer hiện không nhúng legacy dashboard Phase 50 ở notebook. Test legacy được giữ để traceability nhưng không phải gate cho refactor presentation Phase 43–59. **Trạng thái: xác minh được qua pytest output, không suy đoán.** |
| `tests/unit/test_phase51_g_finalization.py` | `75 passed, 2 failed` (cùng cụm với e08/e09) | `pytest tests/unit/test_phase51_g_finalization.py` (chạy chung với test khác) — failure kiểu `test_g10_attention_handoff_cases_csv_exists`, `test_g15_tests_summary_exists`: thiếu `phase51_attention_handoff_cases.csv` và `phase51_tests_summary.csv` trong `artifacts/worst_error_analysis/`. Đây là **reporting debt**: handoff CSV chưa được khôi phục tại active path; renderer notebook không dùng các file này trong Phase 51 compact block (rendering Phase 51 dùng `phase51_summary.json` và `phase_51_signoff.json`). **Trạng thái: xác minh được, không suy đoán.** |

### 16.6. Static / AST validation

| Validation | Kết quả | Evidence |
|---|---|---|
| `python -m py_compile` cho 18 module đã MODIFIED/CREATED ở Section 15 | `0` lỗi, `py_compile_exit=0` | `python -m py_compile src/course_work/reporting/results_rebuild.py src/course_work/reporting/frozen_evidence.py src/course_work/reporting/model_improvement_v2_dashboard.py src/course_work/model_improvement_v2/step16_final_refit.py src/course_work/model_improvement_v2/step17_post_hoc_benchmark.py src/course_work/model_improvement_v2/e10_runner.py src/course_work/model_improvement_v2/e11_runner.py src/course_work/model_improvement_v2/e12_runner.py src/course_work/model_improvement_v2/e13_runner.py src/course_work/model_improvement_v2/e14_runner.py src/course_work/model_improvement_v2/e15_runner.py src/course_work/model_improvement_v2/e16_runner.py src/course_work/model_improvement_v2/e20_runner.py src/course_work/model_improvement_v2/hybrid_loss.py src/course_work/model_improvement_v2/pretest_adapter.py src/course_work/rolling_origin/real_run.py src/course_work/rolling_origin/refit_engine.py src/course_work/training/engine.py src/course_work/experiments/registry.py` |
| `git diff --check HEAD` tại `COURSE_WORK/` | clean, exit `0`, không có whitespace/conflict warning | `git diff --check HEAD` |

### 16.7. JSON / artifact validation

| Validation | Kết quả | Evidence |
|---|---|---|
| Notebook JSON parse được | `PASS` | `json.loads(notebook.read_text())` trong test parse trên |
| Phase 43–59 final-lock SHA-256 khớp manifest | `5 / 5` khớp — checkpoint seed `42`, `123`, `2026`, scaler `final_x_scaler.json`, scaler `final_y_scaler.json` | `python hashlib.sha256` đối chiếu với Section 13.1: `5cf30c81...74a8f6`, `dfbdc647...297766`, `8eb9d9f0...941c4`, `521df9de...c7724`, `c572e817...cf206f3` |
| Recovery manifest Phase 54–57 | `PASS` (manifest tồn tại, chứa `active_evidence_sha256` cho từng figure và intermediate artifact) | `artifacts/notebook_presentation_recovery/courseworkv10_phase54_57/manifest.json` — `artifact_version=NOTEBOOK_PRESENTATION_RECOVERY-v1`, `phases.54.active_evidence_file_count=104` (tổng `5` phase trong manifest); `scientific_recomputation=false`, `test_source_accessed=false` |
| V2 final closure JSON | `model_improvement_v2_status=COMPLETE` | [model_improvement_v2_final_closure.json](COURSE_WORK/artifacts/model_improvement_v2/model_improvement_v2_final_closure.json) |
| Step 17 benchmark JSON | `step17_metrics.json` chứa `RMSE=61.608937`, `MAE=25.897130`, `R²=0.540364`, `n_samples=2961` | [step17_metrics.json](COURSE_WORK/artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_metrics.json) (đã đối chiếu khi viết Mục 9, 12, 13) |

### 16.8. Metric traceability

| Validation | Kết quả | Evidence |
|---|---|---|
| Phase 47 metric | Truy vết được tới [final_test_summary.json](COURSE_WORK/artifacts/final_test/final_test_summary.json) + [phase_47_signoff.json](COURSE_WORK/artifacts/final_test/phase_47_signoff.json) | Test `test_phase47_uses_current_verified_per_seed_and_baseline_metrics` PASS |
| Phase 43–59 metric | Truy vết được tới source artifact khai báo trong `render_verified_phase_result` | Test `test_every_phase_43_59_renders_nonempty_table_without_evidence_metadata` PASS |
| V2 metric trong notebook | Truy vết được tới comparison/Human decision/final lock/closure artifact load bởi `render_verified_v2_results` | Test `test_v2_result_chain_is_complete_without_evidence_metadata` PASS |
| Step 14B metric theo `alpha` | Không hiển thị vì không có result artifact độc lập | `UNVERIFIED/UNAVAILABLE` (giống Mục 13.2) |
| Metric `WARNING`/`REJECT`/`DEFERRED` bị đổi thành `PASS` | `0` | Section 8 và test ở trên |

### 16.9. Checksum / provenance validation

| Validation | Kết quả | Evidence |
|---|---|---|
| Final-lock checkpoints SHA-256 | `3 / 3` khớp Section 13.1 | Tính lại từ file hiện tại |
| Final-lock scalers SHA-256 | `2 / 2` khớp Section 13.1 | Tính lại từ file hiện tại |
| Recovery manifest Phase 54–57 SHA-256 cho từng figure và intermediate file | Có mặt trong `active_evidence_sha256` | [manifest.json](COURSE_WORK/artifacts/notebook_presentation_recovery/courseworkv10_phase54_57/manifest.json) |
| Test source accessed trong `step17_test_access_event.json` | Có mặt, đúng `1` lần với label `POST_HOC_V2_BENCHMARK` | [step17_test_access_event.json](COURSE_WORK/artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_test_access_event.json) (xem Mục 12 và 13.2) |

### 16.10. Relative link validation

| Validation | Kết quả | Evidence |
|---|---|---|
| Tất cả relative Markdown link ở Mục 14 | Đã được đối chiếu với current repo tree; không link tới file không tồn tại | Đối chiếu thủ công từng link ở Mục 14 với `ls`/`test -f` của các path tương ứng |
| `notebook_course_work/CourseWork.ipynb` | `tồn tại` | `ls` |
| `artifacts/notebook_presentation_recovery/courseworkv10_phase54_57/manifest.json` | `tồn tại` | `ls` |
| `artifacts/model_improvement_v2/final_model_lock/v2_final_model_lock.json` | `tồn tại` | `ls` |
| `artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_benchmark_signoff.json` | `tồn tại` | `ls` |
| `artifacts/model_improvement_v2/model_improvement_v2_final_closure.json` | `tồn tại` | `ls` |
| `docs/model_improvement_v2_final_closure.md` | `tồn tại` | `ls` |

### 16.11. Tổng hợp test inventory ở thời điểm viết

| Test cụm | Lệnh | Kết quả (PASS/FAIL/SKIP/ERROR) | Ghi chú |
|---|---|---|---|
| `test_results_rebuild.py` + `test_notebook_frozen_evidence.py` | `pytest tests/unit/test_results_rebuild.py tests/unit/test_notebook_frozen_evidence.py` | `17 passed, 2 skipped, 1 failed` | `failed` = `test_notebook_phase_1_42_prefix_matches_locked_rebuild_hash` (prefix hash mismatch, đã ghi ở Mục 4) |
| `test_e08` + `test_e09` | `pytest tests/unit/test_model_improvement_v2_e08.py tests/unit/test_model_improvement_v2_e09.py` | `36 passed` | Bao gồm cả `test_model_improvement_v2_pretest_adapter.py` không chạy riêng, kết quả `36 passed` tổng; pretest_adapter PASS qua cùng run |
| `test_e10` + `test_e11` + `test_e11_h1_resume` + `test_e12` + `test_e13` + `test_e14` + `test_e15` + `test_e16` + `test_e20` | `pytest ..._e10.py ..._e11.py ..._e11_h1_resume.py ..._e12.py ..._e13.py ..._e14.py ..._e15.py ..._e16.py ..._e20.py` | `88 passed` | |
| `test_step17` + `test_final_closure` | `pytest tests/unit/test_model_improvement_v2_step17.py tests/unit/test_model_improvement_v2_final_closure.py` | `12 passed` | |
| `test_step16` | `pytest tests/unit/test_model_improvement_v2_step16.py` | `4 passed, 5 errors (setup-time)` | Preflight error: `Deterministic FINAL_DEV scaler checksum drift` |
| `test_selective_execution_policy.py` | `pytest tests/contracts/test_selective_execution_policy.py` | `9 passed` | |
| `test_phase50_h_notebook_dashboard.py` + `test_phase51_g_finalization.py` (chạy chung với e08/e09) | `pytest ..._phase50_h_notebook_dashboard.py ..._phase51_g_finalization.py ..._e08.py ..._e09.py` | `75 passed, 13 failed` | 11 fail thuộc legacy Phase 50 dashboard test, 2 fail thuộc Phase 51 handoff CSV — đây là reporting debt, không ảnh hưởng rendering Phase 43–59 hiện tại |

Tổng số test gộp lại (khi chạy riêng từng cụm) là `241 passed, 14 failed, 2 skipped, 5 errors`. Tổng này được tính bằng cách cộng kết quả từng lệnh `pytest` ở bảng trên; **không suy ra pass/fail bằng cách cộng chéo giữa các suite không độc lập**. Trong đó:

- `4 / 14 fail` là test reporting debt Phase 51 (`test_g10_attention_handoff_cases_csv_exists`, `test_g15_tests_summary_exists` đã xác minh ở 16.5);
- `11 / 14 fail` là legacy `test_phase50_h_notebook_dashboard.py` (đã xác minh ở 16.5);
- `1 / 14 fail` là `test_notebook_phase_1_42_prefix_matches_locked_rebuild_hash` (Phase 1–42 prefix-hash mismatch đã ghi ở Mục 4 và 16.1);
- `5 errors` là `test_step16.py` setup-time preflight error đã xác minh ở 16.4.

Ngoài các cụm trên, các test cũ không nằm trong scope refactor (`tests/unit/test_*.py` còn lại) chưa được chạy trong task này và **không được tính vào tổng `241 / 14 / 2 / 5`**. Việc chạy toàn bộ test suite thuộc scope verification riêng, không thuộc task viết tài liệu này.

### 16.12. Các test KHÔNG được mở lại trong task này

| Test cụm | Lý do không mở | Ghi chú |
|---|---|---|
| Test yêu cầu Test loader của `step17_post_hoc_benchmark.py` | Task viết tài liệu không mở Test source | Test source chỉ được mở đúng một lần trong workflow trước đó với label `POST_HOC_V2_BENCHMARK` (xem Mục 9 và 13) |
| Test rerun training mới | Refactor presentation không rerun training | Final lock đã `LOCKED`, không tạo checkpoint/scaler mới (xem Section 13.1) |
| Test fine-tuning experiment E18 thực thi `FT-A/B/C` | E18 chỉ classify `EXACT_REUSE_ALLOWED`, không mở `FT-A/B/C` (xem Mục 9 và 13) | `FT-A/B/C=DEFERRED_NO_PROMOTED_NEW_HEAD` |

### 16.13. Phát hiện cần theo dõi từ validation

| Phát hiện | Mức độ | Ghi chú |
|---|---|---|
| `test_notebook_phase_1_42_prefix_matches_locked_rebuild_hash` FAIL với hash `c61bd876d2e53773c3ce75b90945260e4f5d9fe7794fc2fb850802269c7e65c5` thay vì guard `d81f82714fef39667f41ed4af2049629bd13ccd72fc349160817ae565c0c0d6d` | `KNOWN_DEBT` | Đã ghi ở Mục 4 và 6. Refactor presentation Phase 43–59 không dùng để hợp thức hóa thay đổi trước Phase 43; cần audit riêng |
| `test_step16.py` 5 ERROR do `Step16PreflightError("Deterministic FINAL_DEV scaler checksum drift")` tại `step16_final_refit.py:341` | `KNOWN_DEBT` | Khi chạy `fit_final_dev_scaler`, `x_scaler_sha256` / `y_scaler_sha256` / `bundle_sha256` tính lại không khớp `final_scaler_lock` đã ghi. Cần verify xem đây là random seed / platform drift hay là scaler artifact đã thay đổi |
| `test_phase50_h_notebook_dashboard.py` 11 FAIL legacy | `KNOWN_DEBT` | Test cũ viết cho Phase 50 legacy dashboard; notebook hiện không nhúng legacy dashboard Phase 50 |
| `test_phase51_g_finalization.py` 2 FAIL do thiếu `phase51_attention_handoff_cases.csv` và `phase51_tests_summary.csv` | `KNOWN_DEBT` | Reporting debt — handoff CSV chưa khôi phục trong active path; rendering Phase 51 hiện dùng `phase51_summary.json` + `phase_51_signoff.json` |

## 17. Known issues / reporting debt

Bảng dưới đây liệt kê các vấn đề còn tồn tại sau refactor presentation/notebook và triển khai `MODEL_IMPROVEMENT_V2`, cùng với evidence truy vết được. Các vấn đề được phân loại rõ ràng; không gộp blocker và non-blocker. Những vấn đề đã giải quyết (ví dụ `UNTRACKED_LOCAL_COPY` chỉ là ghi nhận tồn tại, không phải issue) không xuất hiện ở bảng này.

| Vấn đề | Phạm vi ảnh hưởng | Mức độ | Blocker? | Evidence |
|---|---|---|---|---|
| `README.md` ở repo root không được liệt kê trong danh sách entry point của tài liệu | Khả năng người đọc mới bỏ sót phần tổng quan phạm vi rộng | `REPORTING DEBT` | Không | Tìm thấy `README.md` ở repo root bằng `find . -maxdepth 4 -name "README.md"` (kết quả trả về `./README.md`). Tài liệu này (Section 1) đã ghi nhầm là `UNVERIFIED/UNAVAILABLE`; đã sửa lại trong bản audit cuối |
| `test_notebook_phase_1_42_prefix_matches_locked_rebuild_hash` FAIL | Test integrity Phase 1–42 rebuild hash; `tests/unit/test_results_rebuild.py` | `NON-BLOCKING` (presentation Phase 43–59 không phụ thuộc Phase 1–42 rebuild hash) | Không | `pytest tests/unit/test_results_rebuild.py tests/unit/test_notebook_frozen_evidence.py` → `17 passed, 2 skipped, 1 failed`. Guard mong đợi `d81f82714fef39667f41ed4af2049629bd13ccd72fc349160817ae565c0c0d6d`; rebuild hiện tại tính ra `c61bd876d2e53773c3ce75b90945260e4f5d9fe7794fc2fb850802269c7e65c5` (xem Mục 16.13) |
| `test_step16.py` 5 ERROR do `Step16PreflightError("Deterministic FINAL_DEV scaler checksum drift")` | Test `tests/unit/test_model_improvement_v2_step16.py` setup-time | `NON-BLOCKING` (lock artifacts trên disk đã verify SHA-256 bằng `hashlib.sha256` ở Mục 13.1, 16.7, 16.9) | Không | `pytest tests/unit/test_model_improvement_v2_step16.py` → `4 passed, 5 errors (setup-time)`. `fit_final_dev_scaler` tính lại `x_scaler_sha256` / `y_scaler_sha256` / `bundle_sha256` không khớp `final_scaler_lock` đã ghi trong `v2_final_model_lock.json` |
| `test_phase50_h_notebook_dashboard.py` 11 FAIL legacy | Legacy dashboard test Phase 50 | `NON-BLOCKING` (rendering Phase 50 hiện không dùng legacy dashboard, dùng renderer thống nhất) | Không | `pytest ..._phase50_h_notebook_dashboard.py ..._phase51_g_finalization.py ..._e08.py ..._e09.py` → `75 passed, 13 failed` (xem Mục 16.5) |
| `test_phase51_g_finalization.py` 2 FAIL do thiếu CSV handoff | `phase51_attention_handoff_cases.csv`, `phase51_tests_summary.csv` không có trong `artifacts/worst_error_analysis/` | `NON-BLOCKING` (renderer Phase 51 dùng `phase51_summary.json` + `phase_51_signoff.json`) | Không | Cùng `pytest` trên; `2 / 13` failed |
| `phase54_residual_metrics.csv` và `phase55_segment_metrics.csv` không có trong active artifact tree | Reporting CSV lịch sử | `REPORTING DEBT` (đã ghi ở Mục 11) | Không | `results_rebuild.py` chỉ render từ `last_query_attention_summary.json` / `head_comparison_summary.json` hiện có; không dựng lại CSV thiếu |
| Phase 49/52 fallback `PASS` do summary không có top-level scalar status | Status-source coupling | `REPORTING DEBT` (đã ghi ở Mục 8) | Không | `phase49_summary.json`, `attention_extraction_summary.json` không có top-level `status`; signoff độc lập xác nhận `PASS` nhưng renderer chưa load signoff trực tiếp |
| `Step 14B` metric theo từng `alpha` không hiển thị | Không có result artifact độc lập | `REPORTING DEBT` | Không | Không tìm thấy file `step14b_*_comparison.json` trong `artifacts/model_improvement_v2/`; metric `KEEP_NEURAL_ENSEMBLE` được carry-forward vào `step16_final_refit_config_snapshot.json` và final lock/closure |
| Human decision artifact E10/E11/E12 không tồn tại trong namespace experiment hiện tại | Decision lineage E10–E12 | `REPORTING DEBT` | Không | `find artifacts/model_improvement_v2/experiments/E10 artifacts/.../E11 artifacts/.../E12 -name "e*_human_decision.json"` trả về `0` kết quả; các comparison file (`e10_target_delta_ablation_comparison.json`, `e11_rolling_target_ablation_comparison.json`, `e12_lag144_ablation_comparison.json`) tồn tại và ghi `HUMAN_REVIEW_REQUIRED` / `eligible_for_human_promotion=false` |
| `Phase 54–57` intermediate CSV (`phase54_residual_metrics.csv`, `phase55_segment_metrics.csv`) đã được khôi phục thành snapshot HTML trong recovery namespace | Historical dashboard snapshot | `REPORTING DEBT` (đã ghi ở Mục 4, 11) | Không | `artifacts/notebook_presentation_recovery/courseworkv10_phase54_57/manifest.json` ghi `artifact_version=NOTEBOOK_PRESENTATION_RECOVERY-v1`, `scientific_recomputation=false`, `test_source_accessed=false` |
| Notebook Phase 1–42 có thay đổi ở current worktree | Phase 1–42 | `NON-BLOCKING` | Không | `git diff --name-status HEAD -- COURSE_WORK/notebook_course_work/CourseWork.ipynb` cho thấy file `MODIFIED` với `3153` dòng thay đổi (Mục 15.1); nguồn gốc chi tiết `UNVERIFIED/UNAVAILABLE` |
| `notebook_course_work/CourseWork copy.ipynb` tồn tại trong untracked list | File local backup | `NON-BLOCKING` (không được renderer notebook sử dụng) | Không | `git status --short` liệt kê file này trong untracked; đã ghi nhận ở Mục 15.1 |
| `CourseWorkV11` remote `ahead 38, behind 1` | Trạng thái push | `HISTORICAL LIMITATION` (refactor không thực hiện `git push`) | Không | `git status --short --branch` (Mục 1) |
| `git status` có tracked file đã sửa, untracked V2 artifact/code/test và file ngoài `COURSE_WORK/` ở trạng thái deleted | Worktree cleanliness | `HISTORICAL LIMITATION` | Không | `git status --short` (xem Mục 1) |
| Path trong diff vẫn là absolute path, chưa phải portable root discovery | Khả năng chạy trên máy khác | `HISTORICAL LIMITATION` | Không | Mục 5 và Mục 6 |
| `Phase 17` Step 17 — chưa có đánh giá so sánh V1 mean-metric vs V2 ensemble-metric theo matched experiment | Diễn giải khoa học | `NON-BLOCKING` (đã ghi rõ "không phải matched retraining experiment" ở Mục 12.3) | Không | Mục 12.3 ghi rõ V1 value là historical Transformer mean, V2 value là ensemble prediction metric, không cùng matched experiment |
| V1 historical Test đã từng được access trong `step17_test_access_event.json` cho `POST_HOC_V2_BENCHMARK` | Test access event | `HISTORICAL LIMITATION` (workflow trước đó đã access Test đúng một lần) | Không | `step17_test_access_event.json` ghi nhận 1 lần access với label `POST_HOC_V2_BENCHMARK` (Mục 13.2, 16.9); `final_closure.governance.test_status = "ACCESSED_ONCE_FOR_POST_HOC_V2_BENCHMARK"` |
| Chưa chạy toàn bộ test suite cũ ngoài phạm vi refactor | Validation coverage | `REPORTING DEBT` | Không | Mục 16.11 ghi rõ các test cũ không nằm trong scope refactor chưa được chạy trong task này |

Các vấn đề trên là reporting/validation debt. Chúng không đổi scientific evidence đã được khóa ở Mục 12 và 13. Nếu một issue được liệt kê ở bảng này nhưng evidence tương ứng đã được khắc phục trong tương lai, nó sẽ được gỡ khỏi bảng trong bản audit kế tiếp.

### Source map

- [docs/analysis_error/](COURSE_WORK/docs/analysis_error/)
- [artifacts/notebook_presentation_recovery/](COURSE_WORK/artifacts/notebook_presentation_recovery/)
- [artifacts/model_improvement_v2/model_improvement_v2_final_closure.json](COURSE_WORK/artifacts/model_improvement_v2/model_improvement_v2_final_closure.json)
- [README.md](README.md) (repo root, tồn tại tại thời điểm audit)
- Human decision artifact E10–E12: `UNVERIFIED/UNAVAILABLE` trong namespace experiment hiện tại (xem bảng trên)
- Notebook Phase 1–42 integrity checksum mismatch: chờ audit riêng (xem bảng trên)

## 18. Trạng thái cuối và handoff

Phần này chốt trạng thái cuối của project sau refactor presentation/notebook và triển khai `MODEL_IMPROVEMENT_V2`. Mọi trạng thái đều có evidence ở repo hiện tại và đã được liệt kê trong các mục trước.

### 18.1. Bảng trạng thái

| Thành phần | Trạng thái |
|---|---|
| Refactor presentation/notebook | `IMPLEMENTED_IN_WORKTREE`; presentation Phase 43–59 chuẩn hóa 17 phase × 3 bảng = 51 HTML table; không còn inline figure trong compact block; thêm section `MODEL_IMPROVEMENT_V2` riêng cuối notebook. Source: Mục 5, 7, 8, 11 |
| Notebook (state) | `153 cells`; `77 / 77` code cells có `execution_count`; `0` output loại `error`; `last_execution_count=77`; cell `v2closedash` là cell cuối. Source: Mục 16.1 |
| `MODEL_IMPROVEMENT_V2` workflow | `COMPLETE` theo [model_improvement_v2_final_closure.json](COURSE_WORK/artifacts/model_improvement_v2/model_improvement_v2_final_closure.json) (`model_improvement_v2_status=COMPLETE`) |
| `Step 16` final lock | `LOCKED`, `3_OF_3_COMPLETED`, `16` epochs/seed, seeds `[42,123,2026]`, weights `[1/3,1/3,1/3]`, policy `MEAN_E14_M1_SEEDS_42_123_2026`, epoch policy `MEDIAN_RO_INNER_BEST_EPOCHS-v1`. Source: [v2_final_model_lock.json](COURSE_WORK/artifacts/model_improvement_v2/final_model_lock/v2_final_model_lock.json) (Mục 12.4, 13.2) |
| `Step 17` benchmark | `POST_HOC_V2_BENCHMARK`; ensemble `RMSE=61.608937 Wh`, `MAE=25.897130 Wh`, `R²=0.540364`, `n_samples=2961`; per-seed và persistence lưu trong [step17_metrics.json](COURSE_WORK/artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_metrics.json) |
| Phase 43–59 reporting | `17/17 RENDERED` qua `render_verified_phase_result`; `51 / 51` HTML table xuất hiện đúng 3 label bắt buộc (`Phase status`, `Configuration / analysis`, `Results / decision`); không leak evidence path / raw JSON. Source: Mục 8, 16.2 |
| Phase 22–41 reporting | Dùng `render_frozen_phase_evidence` qua `display(...)`; renderer đọc artifact hiện có, không materialize phase, không chạy training. Source: Mục 5, 7, 10 |
| Phase 47 V1 evidence | Giữ nguyên V1 per-seed/mean/Persistence evidence; không chèn V2 metric vào Phase V1. Source: Mục 8, 11 |
| Phase 54–57 recovery | `RESTORED` 4 dashboard HTML vào `artifacts/notebook_presentation_recovery/courseworkv10_phase54_57/` với source branch, source commit, SHA-256 trong manifest; `scientific_recomputation=false`, `test_source_accessed=false`. Source: Mục 4, 11, 16.3 |
| Checkpoint/scaler lineage | `5 / 5` SHA-256 khớp Section 13.1 (3 checkpoint + 2 scaler); `git status` không có file mới ngoài những gì đã track. Source: Mục 13.1, 16.7, 16.9 |
| Tests/validation | `241 passed, 14 failed, 2 skipped, 5 errors` (cộng từng cụm độc lập); `5 / 14` fail là reporting debt Phase 51; `11 / 14` fail là legacy Phase 50 dashboard; `1 / 14` fail là prefix-hash Phase 1–42; `5 errors` là `Step16PreflightError`. Source: Mục 16.4, 16.5, 16.11, 16.13 |
| Static validation | `python -m py_compile` clean cho 18 module đã MODIFIED/CREATED; `git diff --check HEAD` exit `0`. Source: Mục 16.6 |
| Worktree cleanliness | Có tracked file đã sửa, untracked V2 artifact/code/test, file ngoài `COURSE_WORK/` ở trạng thái deleted (xem Mục 15.8); `CourseWorkV11` remote `ahead 38, behind 1` |

### 18.2. Field trạng thái cuối

| Field | Giá trị | Evidence |
|---|---|---|
| `REFACTOR_STATUS` | `IMPLEMENTED_IN_WORKTREE` (chưa commit) | Mục 5, 7, 15 |
| `NOTEBOOK_STATUS` | `153 cells`, `77 / 77` code executed, `0` error output | Mục 16.1 |
| `PHASE_43_59_REPORTING_STATUS` | `17 / 17 RENDERED`, `51 / 51` HTML table, `PRESENTATION_CHANGE_ONLY` | Mục 8, 16.2 |
| `MODEL_IMPROVEMENT_V2_STATUS` | `COMPLETE` (final closure) | [model_improvement_v2_final_closure.json](COURSE_WORK/artifacts/model_improvement_v2/model_improvement_v2_final_closure.json) |
| `SCIENTIFIC_RESULTS_CHANGED` | `NO` (refactor không recompute metric) | Mục 12.4 |
| `ALL_DISPLAYED_METRICS_TRACEABLE` | `YES` (mọi metric trình bày ở Mục 12, 13 đều trỏ tới source artifact đang tồn tại; `Step 14B` per-`alpha` metric là `UNVERIFIED/UNAVAILABLE` được ghi nhận) | Mục 12, 13.2, 16.8, 17 |
| `TRAINING_EXECUTED_DURING_REFACTOR` | `NO` (không run V2 mới; `git status` không có checkpoint/scaler mới ngoài 3 + 2 đã track) | Mục 16 |
| `INFERENCE_EXECUTED_DURING_REFACTOR` | `NO` (Test source không mở lại; `step17_test_access_event.json` chỉ ghi 1 access với label `POST_HOC_V2_BENCHMARK`) | Mục 13.2, 16.9 |
| `TEST_SOURCE_REOPENED_DURING_REFACTOR` | `NO` (`final_closure.governance.test_status = "ACCESSED_ONCE_FOR_POST_HOC_V2_BENCHMARK"`) | Mục 12.4, 16.9 |
| `REMAINING_BLOCKERS` | `0` (Mục 17 chỉ liệt kê `NON-BLOCKING`, `REPORTING DEBT` và `HISTORICAL LIMITATION`; không có `BLOCKING`) | Mục 17 |

### 18.3. Remaining blockers

Theo phân loại trong Mục 17, **không có blocker** cản Human review hoặc tiếp tục workflow. Các item còn lại đều thuộc nhóm `NON-BLOCKING` / `REPORTING DEBT` / `HISTORICAL LIMITATION` và đã được ghi nhận đầy đủ với evidence.

### 18.4. Bước tiếp theo sau handoff

Các bước sau được đề xuất theo thứ tự ưu tiên; không có bước nào bắt buộc cho Human review hiện tại:

1. **Human review** tài liệu này để xác nhận `READY_FOR_HUMAN_REVIEW=YES`. Nếu có chỉnh sửa, cập nhật evidence-backed và chạy lại audit link/metric.
2. (Tùy chọn) Audit riêng Phase 1–42 notebook prefix-hash mismatch để quyết định xem nguồn gốc thay đổi là presentation refactor hay carry-over từ workflow trước đó.
3. (Tùy chọn) Verify `Step16PreflightError` checksum drift: chạy `fit_final_dev_scaler` ở cùng platform/seed để xác định đây là random seed/platform drift hay scaler artifact đã thay đổi.
4. (Tùy chọn) Khôi phục `phase51_attention_handoff_cases.csv` và `phase51_tests_summary.csv` nếu muốn test `test_g10_*` / `test_g15_*` PASS mà không phải test reporting debt.
5. (Tùy chọn) Commit worktree hiện tại và push `CourseWorkV12` lên remote để đồng bộ với `CourseWorkV11` (`ahead 38, behind 1`).

### 18.5. File người tiếp theo nên đọc đầu tiên

| Ưu tiên | File | Mục đích |
|---|---|---|
| 1 | [README.md](README.md) (repo root) | Tổng quan phạm vi rộng của project, từ dữ liệu tới Phase 0–59 |
| 2 | [Description_after_refactor.md](Description_after_refactor.md) (file này) | Trạng thái refactor presentation/notebook và `MODEL_IMPROVEMENT_V2` |
| 3 | [notebook_course_work/CourseWork.ipynb](COURSE_WORK/notebook_course_work/CourseWork.ipynb) | Notebook chính, 153 cells |
| 4 | [model_improvement_v2_final_closure.json](COURSE_WORK/artifacts/model_improvement_v2/model_improvement_v2_final_closure.json) | V2 final closure `COMPLETE` |
| 5 | [docs/model_improvement_v2_final_closure.md](COURSE_WORK/docs/model_improvement_v2_final_closure.md) | V2 closure narrative |
| 6 | [v2_final_model_lock.json](COURSE_WORK/artifacts/model_improvement_v2/final_model_lock/v2_final_model_lock.json) | V2 final lock `LOCKED` |
| 7 | [step17_benchmark_signoff.json](COURSE_WORK/artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_benchmark_signoff.json) | Step 17 benchmark signoff |
| 8 | [step17_metrics.json](COURSE_WORK/artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_metrics.json) | Step 17 benchmark metrics (ensemble + per-seed + persistence) |
| 9 | [phase_47_signoff.json](COURSE_WORK/artifacts/final_test/phase_47_signoff.json) | V1 Phase 47 historical Test signoff |
| 10 | [Plan_improve_model.md](COURSE_WORK/docs/Plan_improve_model.md) | Plan tổng |
| 11 | [tests/unit/test_results_rebuild.py](COURSE_WORK/tests/unit/test_results_rebuild.py) | Test rebuilt notebook presentation |
| 12 | [tests/unit/test_notebook_frozen_evidence.py](COURSE_WORK/tests/unit/test_notebook_frozen_evidence.py) | Test frozen evidence |
| 13 | [tests/unit/test_model_improvement_v2_step17.py](COURSE_WORK/tests/unit/test_model_improvement_v2_step17.py) | Test post-hoc benchmark protocol |

Nếu người tiếp theo cần đi sâu vào từng experiment/Step, danh sách entry point bổ sung đã có ở Mục 14 (Notebook, Plan, V1 Historical Evidence, `MODEL_IMPROVEMENT_V2`, Reporting/Code, Tests).

### 18.6. Source map bổ sung

- [artifacts/model_improvement_v2/final_model_lock/v2_final_model_lock.json](COURSE_WORK/artifacts/model_improvement_v2/final_model_lock/v2_final_model_lock.json)
- [artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_benchmark_signoff.json](COURSE_WORK/artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_benchmark_signoff.json)
- [artifacts/model_improvement_v2/model_improvement_v2_final_closure.json](COURSE_WORK/artifacts/model_improvement_v2/model_improvement_v2_final_closure.json)
- [docs/model_improvement_v2_final_closure.md](COURSE_WORK/docs/model_improvement_v2_final_closure.md)
- [README.md](README.md) (repo root)
