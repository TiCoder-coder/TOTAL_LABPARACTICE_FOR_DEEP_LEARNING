# Pre-Process Plan — Notebook Phase 33 / 46 / 47 Presentation Sync

**Plan ID:** `NB-P47-PRES-SYNC-001`
**Plan timestamp:** 2026-09-03 (UTC+7)
**Plan author role:** Senior ML Systems Engineer + Notebook Architecture Reviewer
**Target file:** `notebook_course_work/CourseWork.ipynb`
**Status:** `DRAFT — WAITING FOR HUMAN APPROVAL`

---

## 1. Plan objective

Synchronize the canonical notebook `CourseWork.ipynb` with the post-Phase-47 official repository state by correcting four presentation defects (D-01, D-02, D-03, D-04) identified in the Phase 0→47 compliance audit. The plan is **strictly presentation-only**. No scientific artifacts may be modified. No Phase 45/46/47 evidence may be altered. No Test re-evaluation may be performed.

Hard scope limits:

- `notebook_course_work/CourseWork.ipynb` cells 2, 96, 97, 121–124
- New cells to be inserted at end of notebook (Phase 47 section)
- No other source files in `src/course_work/` may be modified
- No scripts under `scripts/` may be invoked from notebook cells
- No checkpoints may be loaded; no model inference may be triggered

---

## 2. Upstream dependencies (verified)

| Item | Status | Source |
|---|---|---|
| `architecture_rule.md` v1.9 read | ✓ | `docs/RULE_BASE/architecture_rule.md` §6.6, §6.8, §7.9.1, §13, §14 |
| `rule_code.md` read | ✓ | `docs/RULE_BASE/rule_code.md` §15, §58, §112, §119, §126 |
| Phase 33 detail | ✓ | `docs/plan-doc/plan_detail_for_each_phase/Phase_33_S11_d_model_sweep.md` |
| Phase 46 detail | ✓ | `docs/plan-doc/plan_detail_for_each_phase/Phase_46_Three-seed_final_runs.md` |
| Phase 47 detail | ✓ | `docs/plan-doc/plan_detail_for_each_phase/Phase_47_Final_test_evaluation.md` |
| Compliance audit | ✓ | `docs/plan-doc/analysis_error/full_phase_0_47_plan_notebook_compliance_audit.md` |
| Audit JSON matrix | ✓ | `docs/save_log_in_processing/full_phase_0_47_compliance_audit.json` |
| Phase 47 signoff | ✓ PASS | `artifacts/final_test/phase_47_signoff.json` |
| Phase 47 processing log | ✓ 24 keys | `docs/save_log_in_processing/phase_47_final_test_evaluation_log.json` |
| Phase 47 metrics | ✓ 4 CSVs | `artifacts/final_test/final_test_metrics_by_seed.csv`, `transformer_seed_aggregate_metrics.csv`, `final_test_baseline_metrics.csv`, `final_test_model_comparison.csv` |
| Phase 47 signoff independently verified | ✓ tolerance 0.00e+00 | (independent recomputation in audit) |

---

## 3. Defect-by-defect correction mapping

### 3.1 D-04 (MINOR) — Phase 33 heading normalization

| Field | Value |
|---|---|
| **Requirement** | Heading `## Transformer Configuration after Phase 33` must be canonical `## Phase 33 - S11 d_model Sweep` per architecture_rule §14.3 |
| **Source of truth** | `docs/plan-doc/plan_detail_for_each_phase/Phase_33_S11_d_model_sweep.md`; existing cell 98 (`## Phase 34 - S12 Head Sweep`) as naming template |
| **Notebook cell affected** | Cell 96 (markdown) — heading only |
| **Exact proposed correction** | Replace heading text `## Transformer Configuration after Phase 33` with `## Phase 33 - S11 d_model Sweep` (single-line, no scientific content change). Code cell 97 (`render_phase_33_transformer_configuration()`) is unchanged |
| **src/reporting module used** | None (heading only) |
| **Processing JSON loaded** | None |
| **Artifact loaded** | None |
| **Verification method** | `grep "## Phase 33" notebook_course_work/CourseWork.ipynb` returns ≥1 line; manual cell inspection |
| **Expected notebook output** | Notebook reads sequentially: Phase 32 → Phase 33 → Phase 34 |
| **Risk** | None (text-only, no scientific impact) |
| **Rollback** | Restore previous heading text |

### 3.2 D-02 (CRITICAL) — Phase 46 stale `BLOCKED` presentation

| Field | Value |
|---|---|
| **Requirement** | Cell 123 must NOT print stale "DERIVED FROM HISTORICAL PHASE 45 — BLOCKED". Must show current canonical Phase 46 PASS state with `ready_for_phase47=true`, `phase47_released=true`, `test_status=NOT_ACCESSED` |
| **Source of truth** | `artifacts/three_seed_final_runs/phase_46_signoff.json` (canonical: status=PASS, `ready_for_phase47=true`, `phase47_released=true`, `test_status=NOT_ACCESSED`) |
| **Notebook cell affected** | Cell 122 (markdown) — replace stale "STATUS: DERIVED FROM HISTORICAL PHASE 45 / BLOCKED" note; cell 123 (code) — replace print block |
| **Exact proposed correction** | (a) Cell 122 markdown: replace the entire `> **STATUS: ...` block with the canonical Phase 46 status derived from the signoff file. (b) Cell 123 code: REPLACE `materialize_phase_46(PROJECT_ROOT)` call + stale print block with: load `artifacts/three_seed_final_runs/phase_46_signoff.json`, read the canonical signoff fields, display PASS state and Test-release readiness. Also call `display(render_phase_summary(46, PROJECT_ROOT))` to use canonical rendering. **`materialize_phase_46` must NOT be invoked** because it triggers `scripts/phase46_three_seed_runs.py` which **trains** the final Transformer (would create optimizer steps) |
| **src/reporting module used** | `course_work.reporting.phase_summary.render_phase_summary` (read-only, loads signoff and signoff-related artifacts) |
| **Processing JSON loaded** | `docs/save_log_in_processing/phase_46_three_seed_final_runs_log.json` (read-only) |
| **Artifact loaded** | `artifacts/three_seed_final_runs/phase_46_signoff.json` (read-only) |
| **Verification method** | (a) Notebook cell 122 contains no "BLOCKED" text. (b) Cell 123 prints `overall_status=PASS` and `ready_for_phase47=True`. (c) Recompute: 3/3 checkpoint SHAs match; 0 new training runs created; 0 optimizer steps |
| **Expected notebook output** | Phase 46 row shows: `overall_status: PASS`, `ready_for_phase47: true`, `phase47_released: true`, `test_status: NOT_ACCESSED`, `candidate: TR_C2_ALT_LOOKBACK`, `seed42/123/2026 run IDs and checkpoint SHAs`, `final_refit_epochs: 30` |
| **Risk** | If `materialize_phase_46` is inadvertently invoked, **it triggers training**. Mitigation: explicitly remove the call; replace with read-only JSON load. Add inline comment-as-prose or docstring noting "DO NOT call materialize_phase_46 — it retrains" |
| **Rollback** | Restore cell 122/123 to pre-change content |

### 3.3 D-03 (MAJOR) — `sys.path` hack in cell 2

| Field | Value |
|---|---|
| **Requirement** | Notebook must not contain `sys.path.append` or `PYTHONPATH` bootstrap per architecture_rule §6.8 |
| **Source of truth** | `architecture_rule.md` §6.8: *"Không chèn sys.path hoặc PYTHONPATH bootstrap vào notebook. Notebook và test import package qua environment đã cài project."* |
| **Notebook cell affected** | Cell 2 (code) |
| **Exact proposed correction** | Replace the 3-line content with a single pip-install command in a Markdown cell (e.g. `## Environment setup` + bash code cell running `pip install -e . --quiet`). The notebook already relies on the editable install being present. **Alternative**: a markdown note instructing the user to run `pip install -e .` outside the notebook before opening. No path hack in any form is allowed. After removal, cell 3 (which imports from `course_work.*`) continues to work because the package is installed editable |
| **src/reporting module used** | None (this is environment setup) |
| **Processing JSON loaded** | None |
| **Artifact loaded** | None |
| **Verification method** | `grep -n "sys.path\|PYTHONPATH" notebook_course_work/CourseWork.ipynb` returns 0 lines in any code cell; `python -c "from course_work.baselines.persistence import materialize_phase_14"` succeeds (package importable without notebook bootstrap) |
| **Expected notebook output** | Notebook opens cleanly with the package available via installed editable mode |
| **Risk** | If user opens notebook without `pip install -e .`, imports in cell 3 fail. Mitigation: add a top-of-notebook markdown cell clearly stating "Run `pip install -e .` from `COURSE_WORK/` before opening this notebook." |
| **Rollback** | Restore original cell 2 |

### 3.4 D-01 (CRITICAL) — Phase 47 missing section + cell 124 stale "PENDING RESEAL"

| Field | Value |
|---|---|
| **Requirement** | Add `## Phase 47 - Final Test Evaluation` section that loads canonical Phase 47 artifacts (signoff + CSVs) and presents the Test results. Remove or replace the stale Cell 124 "Summary — Corrective Pipeline State" text that says "Phase 47 (final Test release): PENDING RESEAL." |
| **Source of truth** | `artifacts/final_test/phase_47_signoff.json` (status=PASS); `artifacts/final_test/final_test_metrics_by_seed.csv`; `artifacts/final_test/transformer_seed_aggregate_metrics.csv`; `artifacts/final_test/final_test_baseline_metrics.csv`; `artifacts/final_test/final_test_model_comparison.csv`; `artifacts/final_test/final_test_report.md` (human-readable summary) |
| **Notebook cells to insert** | After cell 123 (Phase 46), insert 4 new cells: (a) markdown `## Phase 47 - Final Test Evaluation`; (b) code: load Phase 47 signoff + CSVs and display PASS state + candidate/seeds/checkpoints/N/population fingerprint; (c) code: display `final_test_metrics_by_seed.csv` + `transformer_seed_aggregate_metrics.csv`; (d) code: display `final_test_baseline_metrics.csv` + `final_test_model_comparison.csv`; (e) markdown conclusion block describing scientific interpretation (Transformer wins RMSE/R², Persistence wins MAE) |
| **Notebook cell to replace** | Cell 124 (markdown) — replace "Phase 47 (final Test release): PENDING RESEAL." with "Phase 47 (final Test evaluation): PASS — canonical results presented above." Optionally retain the Phase 43/44/45/46 history block but with current PASS status for 45/46/47 |
| **src/reporting module used** | NEW `course_work.reporting.phase_summary.render_phase_summary(47, project_root)` (already exists for any phase_id, reads `phase_47_signoff.json` and the processing log). The notebook only **reads** the existing files via `pd.read_csv` and `json.loads` — no training, no inference, no Test access |
| **Processing JSON loaded** | `docs/save_log_in_processing/phase_47_final_test_evaluation_log.json` (read-only) |
| **Artifacts loaded** | `phase_47_signoff.json`, 4 CSV metrics, `final_test_population_manifest.json`, `final_test_access_event.json` — all read-only |
| **Verification method** | (a) Notebook contains heading `## Phase 47 - Final Test Evaluation`. (b) Cell code reads only canonical artifacts (no `materialize_phase_47` invocation, no `scripts/phase47_final_test_evaluation.py` execution). (c) Displayed metrics match stored values exactly: MAE/RMSE/R² for seed42/123/2026, aggregate mean ± SD, Persistence MAE/RMSE/R². (d) No new training run directories created. (e) `grep "PENDING RESEAL" notebook` returns 0 |
| **Expected notebook output** | Phase 47 section shows: status=PASS, candidate=TR_C2_ALT_LOOKBACK, config fingerprint, 3 seeds with exact checkpoint SHAs, N=2961, per-seed metrics, mean ± SD, persistence baseline, conclusion that Transformer wins RMSE (63.83 vs 66.84) and R² (0.506 vs 0.459), Persistence wins MAE (26.74 vs 28.53). All values match `phase_47_signoff.json` exactly |
| **Risk** | If `scripts/phase47_final_test_evaluation.py` is invoked inadvertently, it would **re-run Test inference** — violation. Mitigation: notebook cell code only does `json.loads` + `pd.read_csv` of already-finalized files. No `run_path`, no `subprocess`. Add docstring prose noting "READ-ONLY: loads canonical Phase 47 artifacts only; never re-runs Test evaluation" |
| **Rollback** | Remove new cells; restore cell 124 |

---

## 4. Notebook architecture compliance (per architecture_rule §6.6 / §7.9.1)

After corrections:

- Notebook contains **only** presentation/orchestration code that imports from `course_work.*` (no `sys.path` hack)
- No reusable processing logic added to notebook
- Phase 6 EDA exception (cells 14–43) is **preserved unchanged**
- Phase 47 cells use only `pd.read_csv` of already-validated metric CSVs (read-only)
- No `def`/`class` introduced in new cells (only display logic)
- No `pd.to_datetime` on raw data
- No DataFrame mutation
- No hash implementation
- No schema/temporal/gap/autocorrelation/rolling implementation
- No plot construction (only loading of saved figures referenced by `final_test_figures/` if needed — actually skip figure embedding; existing Phase 47 `figures/` directory has 4 PNGs but they are not required to be embedded in notebook per §14.1 "Display saved figure" allowlist)

---

## 5. Notebook execution strategy (top-to-bottom without scientific retraining)

After corrections:

- Cell 2 = environment setup markdown (no sys.path hack)
- Cell 3 = imports + project root resolution
- Cells 5–119 unchanged — each calls `materialize_phase_N(PROJECT_ROOT)` for N ∈ {1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21}. These materializers either re-run small idempotent steps or load existing artifacts (per Phase 0–22 patterns). Risk: `materialize_phase_46` re-trains — see §3.2 fix
- Cell 121 (`materialize_phase_45`) — also triggers training; needs the same read-only replacement as cell 123. Plan therefore also revises cell 121 to **read-only** (NOT a regression — was previously broken too)
- New Phase 47 cells use only `json.loads` + `pd.read_csv` of existing canonical artifacts
- Cell 124 updated to current status

**Net effect**: notebook executes top-to-bottom without any retraining and without any Phase 47 Test re-evaluation.

**Side-effect fix included**: Cell 121 currently calls `materialize_phase_45` which re-runs `scripts/phase45_final_model_lock.py` — also a training step. The plan therefore extends D-02 fix to cover cell 121 as well (read-only replacement). This is an **additional presentation fix discovered during planning** — flagged as D-10 (MINOR, presentation-only).

---

## 6. Files to modify

| File | Modification |
|---|---|
| `notebook_course_work/CourseWork.ipynb` | Cell 2 text replacement; cell 96 heading text replacement; cell 121 (Phase 45) → read-only replacement; cell 122 (Phase 46 markdown) stale-note replacement; cell 123 (Phase 46 code) read-only replacement; new Phase 47 section (5 cells); cell 124 (summary markdown) stale-text replacement |

## 7. Files NOT to modify

- Any file under `src/course_work/` — including `src/course_work/phase47/*` (Phase 47 source already complete)
- Any file under `scripts/` — including `scripts/phase47_final_test_evaluation.py`
- Any file under `artifacts/three_seed_final_runs/`, `artifacts/final_test/`, `artifacts/final_model_lock/`
- Any file under `docs/save_log_in_processing/` (log correction not required)
- Any file under `docs/RULE_BASE/`
- Any file under `tests/`
- Any file under `configs/`
- Any file under `data/`

---

## 8. Verification checklist (post-implementation)

After all edits, the implementer must verify:

- [ ] Cell 2 contains no `sys.path` or `PYTHONPATH` (D-03)
- [ ] Cell 96 heading is `## Phase 33 - S11 d_model Sweep` (D-04)
- [ ] Cell 121 (Phase 45) does NOT call `materialize_phase_45` (D-10 — bonus fix)
- [ ] Cell 122 (Phase 46 markdown) does NOT contain "BLOCKED" (D-02)
- [ ] Cell 123 (Phase 46 code) does NOT call `materialize_phase_46`; prints canonical PASS state (D-02)
- [ ] Notebook contains heading `## Phase 47 - Final Test Evaluation` (D-01)
- [ ] Phase 47 section reads only canonical artifacts (no script invocation)
- [ ] Phase 47 displayed values match `phase_47_signoff.json` to 0.00e+00 tolerance
- [ ] Cell 124 does NOT contain "PENDING RESEAL"
- [ ] No absolute `/Users/vientu/...` paths anywhere in notebook
- [ ] `grep -c "sys.path" notebook_course_work/CourseWork.ipynb` returns 0
- [ ] `grep -c "BLOCKED\|PENDING RESEAL" notebook_course_work/CourseWork.ipynb` returns 0
- [ ] `find artifacts/three_seed_final_runs -newer docs/save_log_in_processing/full_phase_0_47_compliance_audit.json -type f` returns 0 (no Phase 46 evidence modified)
- [ ] `find artifacts/final_test -newer docs/save_log_in_processing/full_phase_0_47_compliance_audit.json -type f` returns 0 (no Phase 47 evidence modified)
- [ ] `find artifacts/final_model_lock -newer docs/save_log_in_processing/full_phase_0_47_compliance_audit.json -type f` returns 0 (no Phase 45 evidence modified)
- [ ] Notebook phase order is 1–47 (Phase 33 and Phase 47 now present and ordered correctly)
- [ ] Notebook top-to-bottom execution does not require scientific retraining
- [ ] All notebook-only compliance checks PASS

---

## 9. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| Notebook cell inadvertently invokes `materialize_phase_46` and re-trains | HIGH | Explicit text replacement; code review; regression test grep |
| Notebook cell inadvertently invokes `materialize_phase_45` and re-trains | HIGH | Same — extend D-02 fix to cell 121 |
| Notebook cell inadvertently invokes Phase 47 script and re-evaluates Test | HIGH | Notebook Phase 47 section reads only CSVs/JSON; no script invocation |
| User opens notebook without `pip install -e .` and gets ImportError | LOW | Add markdown prerequisite cell |
| `render_phase_summary(47, PROJECT_ROOT)` missing template for Phase 47 | LOW | Pre-test the call offline before notebook implementation; if missing, fall back to manual display of canonical JSON/CSV content |

---

## 10. Rollback plan

If any defect introduction is detected after the corrections:

1. Restore notebook from `notebook_course_work/CourseWork.ipynb.bak_refactor_20260817` (full historical backup)
2. Re-apply only the verified-safe corrections individually
3. Re-run verification checklist

Since **no scientific artifacts are touched**, rollback is limited to notebook file only and has zero scientific risk.

---

## 11. Acceptance criteria (DoD)

The implementation is complete when:

- [ ] All four primary defects (D-01, D-02, D-03, D-04) corrected
- [ ] D-10 (cell 121 read-only) corrected as bonus
- [ ] Verification checklist §8 fully passed
- [ ] Notebook opens, all cells execute top-to-bottom without ImportError
- [ ] Phase 47 values match canonical signoff exactly
- [ ] No scientific evidence modified
- [ ] User has reviewed the corrected notebook cells and approved

---

## 12. User approval gate (per rule_code §15)

Per `rule_code.md` §15, §58, §119, this pre-process plan **must be approved by the Human** before any notebook implementation begins.

After Human approval, execution follows the canonical workflow:

1. Read rules (already done)
2. Sequential implementation of each correction in a separate step
3. Verify each step
4. Run verification checklist
5. Sign-off report

Per rule_code §16: **CREATE PLAN ≠ EXECUTE PLAN**. This plan is the deliverable for the current task. **Implementation will only begin after explicit Human approval.**
