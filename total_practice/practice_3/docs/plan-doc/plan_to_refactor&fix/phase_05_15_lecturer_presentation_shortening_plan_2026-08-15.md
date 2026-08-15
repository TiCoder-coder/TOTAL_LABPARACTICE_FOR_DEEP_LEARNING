# Practice 3 lecturer-facing notebook shortening plan

Date: 2026-08-15

## Objective

Shorten the Practice 3 notebook presentation without changing methodology, authoritative results, artifact provenance, checkpoint integrity, or the one-time Test contract.

## Scope

- Refactor notebook display code for Phase 5–15.
- Reuse existing processing modules and presentation helpers.
- Keep complete technical evidence in JSON/log artifacts.
- Do not change Train/Validation/Test behavior, checkpoint selection, metrics, or model weights.

## Presentation contract

Each phase should communicate WHAT → IMPORTANT RESULT → WHY → CONCLUSION. Technical hashes, paths, guard internals, debug dictionaries, package inventories, and repeated verification booleans remain in artifacts rather than the main notebook output.

## Planned changes

- Phase 5: retain split/class/length/max-length/class-boxplot evidence; remove repeated length tables and representative/extreme sample dumps.
- Phase 6: retain two concise examples and dynamic-padding conclusion; suppress raw verification dictionary dumps.
- Phase 7: retain architecture figure, compact model/parameter/shape table and PASS result.
- Phase 8: retain metric definitions, compact effective configuration and Validation-only selection statement.
- Phase 9: join epoch Train/Validation metrics into one table; retain only selected epoch/checkpoint and compact artifact status.
- Phase 10: retain two curves and short epoch 1→2 / 2→3 interpretation; remove repeated engineering tables.
- Phase 11: present exactly Selected Model, Verification, and Final Metrics with one comparison chart and Accuracy/F1 gaps.
- Phase 12: retain confusion matrix, counts, FP/FN, confidence, and 6 deterministic representative errors; remove verbose derived/check tables.
- Phase 13: show custom sentence, prediction and confidence plus probability chart and three isolation checks.
- Phase 14: show five compact save/reload checks and at most three before/after examples.
- Phase 15: show one final summary table, one final metric chart, four conclusions and four final status items.

## Files affected

- `notebook_practice_3/practice_3.ipynb`
- process/audit documentation only if validation changes evidence

No unrelated file or `COURSE_WORK` file will be modified.

## Validation

- Run All with repository `.venv`;
- Phase 9 guard loads artifacts without retraining;
- Phase 11 guard does not reevaluate Test;
- `test_evaluation_count == 1`;
- Phase 12 reads frozen predictions only;
- Phase 14 package weights/hash unchanged and package not resaved;
- no stale widget output;
- figures render and final metrics originate from artifacts.

## Completion criteria

- notebook outputs are materially shorter and non-duplicative;
- lecturer-facing evidence remains sufficient;
- 16/16 code cells execute without error;
- methodology and authoritative values remain unchanged.
