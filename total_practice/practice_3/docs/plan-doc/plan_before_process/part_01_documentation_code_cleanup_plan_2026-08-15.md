# Part 01 — Documentation and Code Cleanup Plan

## Objective

Document the implemented Practice 3 codebase and make its Source → Artifact → Notebook → Result provenance traceable without changing methodology, model weights, checkpoint selection, data splits, metrics, or historical evidence.

## Sources to audit

- `PRACTICE3_WORKFLOW_HANDOFF_RULE.md`
- `docs/plan-doc/plan_overview/plan.md`
- `notebook_practice_3/practice_3.ipynb`
- every module in `processing_own_phase/`
- Phase 0–15 plans, process logs, and result artifacts
- `data/` and `image_diagram/`

## Expected changes

- Complete `docs/code_base_audit.md` from current source evidence.
- Add an artifact-derived result-document generator and generated Markdown provenance pages for phases with material outputs.
- Remove the Phase 10 metric-value duplication while preserving artifact validation and behavior.
- Add this plan, a focused refactor plan, and the Part 1 process log.

## Intentionally unchanged

- Practice 1, Practice 2, `COURSE_WORK/`, and `practice_3 copy.ipynb`.
- Phase 9 training artifacts/checkpoints, Phase 11 Test artifacts/count, Phase 14 package, model weights, data splits, metrics, and methodology.
- `docs/current_flow/`; final flow documentation is deferred until Protocol v2 is complete.
- Notebook outputs unless a broken reference is discovered.

## Risks

- Importing orchestration modules can trigger heavy library initialization.
- Running the notebook could touch guarded artifacts or external caches.
- Rewriting historical JSON/figures could change timestamps or hashes.
- Over-generalizing duplicated guard helpers could introduce regressions.

## Validation strategy

1. Parse and compile all Python and notebook code cells without executing training/evaluation.
2. Import only the lightweight cleanup/documentation modules.
3. Generate result Markdown solely by reading existing JSON artifacts.
4. Verify every documented source/artifact/notebook reference exists.
5. Compare authoritative checkpoint/package hashes and `test_evaluation_count` before and after.
6. Confirm Git changes remain under `total_practice/practice_3/` and exclude historical binary artifacts.

## Definition of done

- Phase 0–15 mapping reflects actual source and artifacts.
- Important phase results have consistent, artifact-derived provenance pages.
- No artifact metric is hard-coded in presentation/analysis source.
- Comments, docstrings, constants, paths, and duplication are audited with evidence.
- Static/selective validation passes with no training, Test access, checkpoint change, or unrelated edits.

