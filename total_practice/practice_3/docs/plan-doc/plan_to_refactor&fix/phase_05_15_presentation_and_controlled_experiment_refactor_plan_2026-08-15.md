# Practice 3 Phase 5–15 presentation and experiment refactor plan

Date: 2026-08-15

## Objective

Raise Practice 3 presentation quality to Practice 2 level while preserving module/notebook separation, artifact provenance, checkpoint integrity, and the existing Train/Validation/Test boundary.

## Inputs

- Existing Phase 0–15 source modules and notebook;
- frozen Phase 4–15 artifacts;
- authoritative `checkpoint-1068` and one-time Phase 11 Test result;
- workflow and overview plan;
- methodology-conflict analysis dated 2026-08-15.

## Workstream A — safe artifact-driven presentation refactor

### Phase 5

- Extend EDA artifact generation with P50/P75/P90/P95/P99 token percentiles, split-size chart, token-length-by-class boxplot, and the `max_length=80` reference line.
- Preserve the existing evidence: Train P95=47, P99=56, maximum=78 and zero samples over 80.
- Present each figure using Observe → Explain → Decision.

### Phase 6

- Add a compact raw text → tokens → IDs → attention mask → dynamic padding presentation.
- Show one or two real examples without dumping long arrays.

### Phase 7

- Add a compact model-flow diagram and retain only parameter and forward-pass sanity evidence.

### Phase 10

- Read only Phase 9 artifacts.
- Present train/Validation loss together, Validation Accuracy/F1, selected epoch/checkpoint, overfitting-onset interpretation, and learning rate by step if available in the frozen Trainer history.

### Phase 11

- Do not evaluate Test again.
- Add a Validation-versus-Test metric chart and compute Accuracy/F1 gaps from frozen artifacts.
- Keep hashes and lock details in a compact verification block rather than the main narrative.

### Phase 12

- Read only frozen Phase 11 predictions.
- Add FP/FN and confidence-distribution figures.
- Add deterministic high-confidence errors and cautious evidence-based lexical/length group summaries; do not assert causes unsupported by text.

### Phase 13

- Expand custom authored inputs to six categories: clear positive, clear negative, negation, contrast, mixed sentiment, ambiguous/implicit sentiment.
- This changes the Phase 13 artifact and therefore requires Phase 14 equivalence verification to use the same updated inputs; it must not resave or change model weights.

### Phase 14–15

- Present original-versus-reloaded equivalence compactly.
- Build the final table and metric chart from artifacts only.

## Workstream B — controlled LR experiment

Status: BLOCKED pending an explicit protocol decision.

The sweep cannot retrospectively become the model-selection process for a model whose Test result was already observed. No implementation or training begins until the decision in the analysis document is resolved.

If a new untouched evaluation protocol is explicitly approved, create a separate plan before implementation. The minimum controlled design will fix seed, dataset/splits, tokenizer, maximum length, model, batches, epochs, optimizer family, scheduler, and weight decay; vary only LR (`2e-5`, `3e-5`, `5e-5`); rank by minimum Validation loss, then Validation F1, then Validation Accuracy, then earlier epoch. The notebook must load valid cached artifacts and never retrain automatically.

## Files expected to change for Workstream A

- `processing_own_phase/phase_05_dataset_eda.py`
- `processing_own_phase/phase_06_preprocessing.py`
- `processing_own_phase/phase_07_model_construction.py` or a diagram artifact generated from verified configuration
- `processing_own_phase/phase_10_learning_curves.py`
- `processing_own_phase/phase_11_final_evaluation.py` presentation helpers only
- `processing_own_phase/phase_12_error_analysis.py`
- `processing_own_phase/phase_13_new_sentence_inference.py`
- `processing_own_phase/phase_14_save_reload.py` only if needed to verify the new Phase 13 input contract
- `processing_own_phase/phase_15_final_summary.py`
- `notebook_practice_3/practice_3.ipynb`
- new/updated artifacts in `docs/result/`

## Validation

- Unit/synthetic checks for new artifact transformations;
- no training for Workstream A;
- Phase 9 guard loads existing artifacts;
- Phase 11 guard loads existing artifacts and Test count remains one;
- Phase 12 never loads a model/dataset/Test provider;
- Phase 14 package weights and hashes remain unchanged;
- notebook Phase 0–15 executes with all charts rendered;
- final metrics are read from artifacts.

## Completion criteria

- Presentation quality PASS when every requested safe chart and interpretation exists and renders;
- methodology PASS only if no post-Test selection claim is introduced;
- controlled experiments remain BLOCKED/NOT VERIFIED until an approved valid protocol exists;
- unrelated `COURSE_WORK` files and `practice_3 copy.ipynb` remain untouched.

## Explicit out of scope until approval

- new training runs;
- changing the authoritative checkpoint;
- reevaluating Test;
- representing retrospective experiments as pre-Test selection;
- modifying Practice 1, Practice 2, or `COURSE_WORK`.
