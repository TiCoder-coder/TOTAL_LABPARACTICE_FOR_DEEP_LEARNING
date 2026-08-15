# Practice 3 audit — post-Test controlled-experiment conflict

Date: 2026-08-15

## Context

Practice 3 has completed Phase 0–15. The authoritative model is `checkpoint-1068` (epoch 2), and Phase 11 records one final Test evaluation. A new refactor request asks for a three-learning-rate controlled experiment and ranking before the final evaluation while preserving the existing Train/Validation/Test methodology and `test_evaluation_count == 1`.

## Evidence

- `docs/result/phase_11_evaluation/phase_11_evaluation_manifest.json` records the final Test evaluation and a count of one.
- Phase 9 contains only the baseline learning rate `2e-5`; there are no comparable completed runs for `3e-5` and `5e-5`.
- The requested LR sweep therefore requires new training and Validation-based ranking.
- The Test result is already visible before this proposed model-selection activity.

## Methodology conflict

Running and ranking new candidates after observing Test creates post-Test model-development activity. If a new winner replaces `checkpoint-1068`, the stored Test result is not an evaluation of that new winner. Evaluating the new winner on Test would make the count greater than one. Keeping the old Test result while claiming the new winner as final would be incorrect.

## Safe work that remains possible

The following refactors can use frozen artifacts without changing the model or accessing Test again:

- Phase 5 EDA tables and additional figures;
- Phase 6 preprocessing presentation;
- Phase 7 architecture diagram;
- Phase 10 learning-curve presentation;
- Phase 11 compact Validation-versus-Test presentation;
- Phase 12 confidence/error visualizations and evidence-based summaries;
- Phase 13 richer custom-sentence inference, provided it remains outside Test;
- Phase 14 equivalence table;
- Phase 15 final presentation.

## Required decision for controlled experiments

Choose one protocol before implementing the LR sweep:

1. Keep the current completed experiment authoritative. Add no LR sweep; controlled-experiment status remains NOT VERIFIED/FAIL.
2. Start a new experiment protocol with a new untouched holdout or nested split, invalidate the current final-Test claim for that new protocol, and only evaluate the new locked winner on the new untouched holdout. This changes the current methodology/artifacts and requires explicit approval.
3. Run a clearly labelled retrospective Validation-only study that cannot replace the authoritative model and cannot be presented as the experiment that selected the already-tested model. This supports sensitivity analysis but does not satisfy the requested causal ordering.

## Status

RESOLVED BY SCOPE DECISION for this refactor: the user selected option 1 on 2026-08-15. The authoritative `checkpoint-1068` and one-time Test result are retained, and no LR sweep is added. Controlled LR experiments remain intentionally NOT RUN rather than being retroactively represented as model selection. No training, checkpoint change, or Test access was performed by the presentation refactor.
