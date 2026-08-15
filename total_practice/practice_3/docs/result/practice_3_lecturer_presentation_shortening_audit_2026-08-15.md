# Practice 3 lecturer-facing presentation shortening audit

Date: 2026-08-15

## Audit table

| Phase | Before | After | Removed / collapsed | Kept | Status |
|---|---|---|---|---|---|
| 0–4 | Existing concise flow | Unchanged except clean widget behavior | No methodology content removed | Problem, environment, pretrained demo, tokenization, dataset | PASS |
| 5 | 27 outputs; repeated quality/length/sample tables | 12 outputs | Representative/extreme dumps, repeated length and artifact-status tables | Split size, class balance, word/token length, max=80 line, class boxplot, P50–P99 | PASS |
| 6 | Raw sample/check dictionaries and long arrays | 4 outputs | Raw verification dictionaries and save paths | Two examples with 12-item previews, token count, truncation, dynamic-padding conclusion | PASS |
| 7 | Three wide technical tables and artifact path | 5 outputs | Full mappings/flags/debug fields | Architecture figure, model, labels, parameters, shapes, forward PASS | PASS |
| 8 | Full TrainingArguments and checkpoint-policy fixtures | 3 outputs | Internal strategies, booleans and synthetic winner records | Compact configuration, four metric definitions, Validation-only decision | PASS |
| 9 | 7 outputs including debug/cache/checkpoint internals | 3 outputs | Debug fixture, guard details, duplicate checkpoint records | One epoch table and selected epoch/checkpoint | PASS |
| 10 | 9 outputs with repeated epoch/delta/engineering tables | 3 outputs | Repeated Phase 9 table and internal flags | Two learning curves and short overfitting interpretation | PASS |
| 11 | 10 technical outputs including hash/path fields | 9 compact outputs | SHA-256/path/guard/prediction-artifact details | Selected model, four verification checks, metric table, one chart, Accuracy/F1 gaps | PASS |
| 12 | 15 outputs with derived/check/group tables | 8 outputs | Metric verification, class/lexical/internal tables | Confusion heatmap, counts, FP/FN, confidence, six deterministic errors, conclusion | PASS |
| 13 | 9 outputs with fingerprint/batch/boolean details | 5 outputs | Fingerprint, guard and verification dictionaries | Six custom categories, prediction/confidence table, probability chart, three isolation checks | PASS |
| 14 | 12 outputs with paths, hashes, inventory and full comparisons | 3 outputs | Package inventory, hashes and full tensor/tokenizer checks | Five save/reload checks, three examples and equivalence conclusion | PASS |
| 15 | 27 outputs repeating curves, confusion, inference and package evidence | 5 outputs | All duplicated Phase 10–14 charts/tables and audit booleans | One final table, one metric chart, four conclusions, four final statuses | PASS |

## Verification evidence

- Repository .venv Run All: 16/16 code cells executed.
- Runtime errors: 0.
- Stale Jupyter widget outputs: 0.
- Phase 9 artifacts remained authoritative; no training was started.
- Phase 11 manifest remains FINAL_TEST_COMPLETE.
- test_evaluation_count == 1.
- Test was not used for model selection and the checkpoint was not changed after Test.
- Phase 12 continued to read frozen Phase 11 predictions only.
- Phase 13 presentation contains six custom-authored categories and reports no Test access.
- Phase 14 package was not resaved.
- Checkpoint and saved-package SHA-256 remain identical: 22fcd24d3db7bfebc8ea1dd4f6c0665be5176e34de005e168a747ee83acfa660.
- Notebook size reduced from approximately 1.53 MB to approximately 0.98 MB.

## Final result

- Methodology preserved: PASS
- Notebook presentation shortened: PASS
- Visualization preserved: PASS
- Duplicate outputs removed: PASS
- Run All: PASS
- Final Test count = 1: PASS

Technical evidence remains available in the artifact layer; the notebook now prioritizes lecturer-facing results and decisions.
