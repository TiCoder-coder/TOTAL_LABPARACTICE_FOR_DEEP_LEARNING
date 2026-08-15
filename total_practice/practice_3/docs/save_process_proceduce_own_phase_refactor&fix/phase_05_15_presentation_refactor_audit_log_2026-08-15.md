# Practice 3 presentation refactor audit log

Date: 2026-08-15

## Request

Audit and refactor Practice 3 to retain its methodology while improving controlled experiments, visualizations, explanation, and lecturer-facing presentation.

## Sources audited

- workflow and overview plan;
- notebook source and stored outputs;
- Phase 5–15 modules;
- EDA, preprocessing, training, learning-curve, final-evaluation, error-analysis, inference, save/reload, and final-summary artifacts.

## Key findings

- The core architecture and Train/Validation/Test isolation are strong.
- Existing EDA already verifies class balance, missing/empty text, duplicates, cross-split overlap, word/character/token lengths, representative/extreme samples, and `max_length=80` evidence.
- Existing visual coverage lacks split-size chart, class token-length boxplot, explicit percentile table coverage, Validation-versus-Test chart, FP/FN chart, and confidence-distribution charts.
- Phase 10 already presents loss and Validation metric curves and marks epoch 2/checkpoint-1068.
- Phase 12 already derives TN/FP/FN/TP and confidence/length/lexical summaries from 1,066 frozen predictions.
- Phase 13 currently has four custom examples, fewer than the requested six semantic categories.
- There is only one authoritative LR run (`2e-5`).
- Test has already been evaluated once; a new authoritative LR selection now conflicts with the locked one-time-Test ordering.

## Actions in this audit turn

- Created the methodology-conflict analysis.
- Created the presentation/refactor plan.
- Did not modify source, notebook, artifacts, model, checkpoint, or Test result.
- Did not train or evaluate.

## Status

Safe artifact-driven presentation refactor: IMPLEMENTED AND VERIFIED.

Authoritative controlled LR experiment: NOT RUN by explicit option-1 decision.

## Implemented presentation changes

- Phase 5: split sizes, max-length token histogram, class token-length boxplot, percentile evidence and interpretation.
- Phase 6: compact preprocessing flow and two real examples.
- Phase 7: compact DistilBERT architecture figure.
- Phase 11: Validation-versus-Test figure and generalization gaps.
- Phase 12: FP/FN and correct-versus-wrong confidence figures.
- Phase 13: probability visualization for the existing custom authored examples.
- Phase 15: final Test metric chart.
- Notebook commentary follows Observe → Explain → Decision/Conclusion where a chart drives a decision.

## Verification

- Repository `.venv` Run All: 16/16 code cells executed, zero error outputs, 19 PNG outputs and no `application/vnd.jupyter.widget-view+json` output.
- Phase 9 guard: `loaded_verified_artifacts_no_retraining`.
- Phase 11 guard: loaded verified artifacts, no Test reevaluation.
- Phase 14 guard: loaded verified package, no resave.
- Test evaluation count: 1.
- Authoritative and packaged weight SHA-256: `22fcd24d3db7bfebc8ea1dd4f6c0665be5176e34de005e168a747ee83acfa660`.
- Unrelated `COURSE_WORK` changes and `practice_3 copy.ipynb` were not modified.
