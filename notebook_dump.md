--- CELL 0 (markdown) ---
# Practice 3: Sentiment Analysis with Hugging Face Transformers

This notebook presents an end-to-end binary sentiment-classification workflow on the Rotten Tomatoes dataset. Core processing stays in `processing_own_phase/*.py`; the notebook calls those modules and presents verified results.
--- CELL 1 (markdown) ---
## Phase 0 - Problem Definition

### Objective

Build and verify a DistilBERT classifier that maps an English movie review to `NEGATIVE` or `POSITIVE`.

| Item | Definition |
|---|---|
| Input | English movie-review text |
| Output | `0 = NEGATIVE`, `1 = POSITIVE` |
| Dataset | Rotten Tomatoes: 8,530 Train / 1,066 Validation / 1,066 Test |
| Model | `distilbert/distilbert-base-uncased` |
| Metrics | Loss, Accuracy, Precision, Recall and F1 |
| Selection rule | Lowest Validation loss; Test is used once after checkpoint selection |

### Success Criteria

- Dataset, labels, tokenization and preprocessing pass sanity checks.
- Fine-tuning uses Train; checkpoint selection uses Validation only.
- The locked checkpoint is evaluated once on Test.
- Saved model and tokenizer reload with equivalent predictions.
--- CELL 2 (markdown) ---
### Workflow

![Practice 3 workflow](../docs/result/phase_00_pipeline_uml.png)

The workflow keeps model selection on Validation and uses Test only for the locked model's final evaluation.
--- CELL 3 (markdown) ---
## Phase 1 Environment & Reproducibility
--- CELL 4 (code) ---
import sys
from pathlib import Path

project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"Project root added: {project_root}") 
--- CELL 5 (code) ---
# Phase 1: Environment & Reproducibility
from processing_own_phase.phase_01_environment import (
    set_seed,
    print_environment_info,
    save_environment_report,
)

# Set seed (must be done before running anything else)
set_seed(42)

# Print full report and save JSON file
info = print_environment_info()
filepath = save_environment_report(info)

# Quick summary
print(f"\nReport saved: {filepath}")
print(f"Device: {info['device']}")
print(f"Seed: {info['seed']}")
--- CELL 6 (markdown) ---
# Phase 2: Pretrained Sentiment Inference
--- CELL 7 (code) ---
from transformers.utils import logging as transformers_logging
transformers_logging.disable_progress_bar()

from processing_own_phase.phase_02_pretrained_inference import (
    get_sentiment_pipeline,
    run_inference,
    inspect_tokenizer
)

# Load pipeline (first time downloads ~260MB model, requires internet)
pipe = get_sentiment_pipeline()

# 3 sample sentences
test_sentences = [
    "I absolutely loved this movie! The performances were outstanding.",
    "This film was a complete waste of time. Terrible acting.",
    "It was okay, nothing special."
]

# Run inference
results = run_inference(pipe, test_sentences)
for sentence, result in zip(test_sentences, results):
    print(f"Sentence: {sentence[:60]}...")
    print(f"  Label: {result['label']}, Score: {result['score']:.4f}\n")

# Inspect tokenizer
token_info = inspect_tokenizer(test_sentences[0])
print("Tokenizer vocab size:", token_info['vocab_size'])
print("Tokens (first 20):", token_info['tokens'][:20])
--- CELL 8 (markdown) ---
# Phase 3: Tokenization Investigation
--- CELL 9 (code) ---
# Phase 3: Tokenization Investigation
from processing_own_phase.phase_03_tokenization import (
    tokenize_sentence_to_table,
    print_tokenized_table,
    decode_sanity_check,
    compare_tokenizers,
)

sentence = "I absolutely loved this movie! The performances were outstanding."
print(f"Raw sentence: {sentence}\n")

# Display the implementation's readable token table (tokens + token IDs).
token_table = tokenize_sentence_to_table(sentence)
print_tokenized_table(token_table)

# Display the attention mask produced by the same tokenizer comparison.
comparison = compare_tokenizers(sentence)
print(f"\nToken IDs: {comparison['finetuned']['input_ids']}")
print(f"Attention mask: {comparison['finetuned']['attention_mask']}")

# Verify that decoding preserves the sentence meaning.
decode_result = decode_sanity_check(sentence)
print(f"\nDecoded text: {decode_result['decoded_sentence']}")
print(f"Word overlap ratio: {decode_result['word_overlap_ratio']}")
print(f"Decode sanity check: {'PASS' i...
--- CELL 10 (markdown) ---
# Phase 4: Dataset Loading
--- CELL 11 (code) ---
# Phase 4: Dataset Loading
from processing_own_phase.phase_04_dataset_loading import (
    load_rotten_tomatoes,
    verify_dataset_contract,
    get_dataset_stats,
    check_null_values,
    print_sample_rows,
    save_dataset_summary,
)

# Load dataset
dataset = load_rotten_tomatoes()
print(f"Dataset loaded! Splits: {list(dataset.keys())}")

# Verify all required splits, sizes, schema and every label.
dataset_contract = verify_dataset_contract(dataset)
assert dataset_contract['all_pass'], "Phase 4 dataset contract failed"
print("\nPhase 4 verification:")
print(f"  Splits exist: {dataset_contract['splits_exist']}")
print(f"  Split sizes: {dataset_contract['split_sizes']}")
print(f"  Schema: {dataset_contract['schema']}")
print(f"  Label sets: {dataset_contract['label_sets']}")

# Get statistics
stats = get_dataset_stats(dataset)
print("\nDataset Statistics:")
for split_name, split_stats in stats.items():
    print(f"\n{split_name.capitalize()}:")
    print(f"  Total: {split_stats['tot...
--- CELL 12 (markdown) ---
# Phase 5: EDA and Sanity Checks
--- CELL 13 (code) ---
# Phase 5: EDA and Sanity Checks
import pandas as pd
from IPython.display import Image, Markdown, display
from processing_own_phase.phase_05_dataset_eda import (
    build_eda_summary,
    create_eda_figures,
    save_eda_summary,
    save_token_length_statistics,
)
from processing_own_phase.phase_06_preprocessing import get_tokenizer
from processing_own_phase.config import MAX_TOKEN_LENGTH
from processing_own_phase.presentation_quality import phase_5_presentation

tokenizer = get_tokenizer()
eda_summary = build_eda_summary(dataset, tokenizer, MAX_TOKEN_LENGTH)
assert eda_summary['all_pass'], "Phase 5 EDA sanity checks failed"
recommendation = eda_summary['token_length_recommendation']
assert MAX_TOKEN_LENGTH >= recommendation['max']
eda_summary['selected_max_length'] = MAX_TOKEN_LENGTH
eda_summary['max_length_justification'] = (
    f"P95={recommendation['p95']:.0f}, P99={recommendation['p99']:.0f}, "
    f"observed max={recommendation['max']:.0f}; selected {MAX_TOKEN_LENGTH} "
    "i...
--- CELL 14 (markdown) ---
# Phase 6: Tokenizer and Preprocessing
--- CELL 15 (code) ---
# Phase 6: Tokenizer and Preprocessing
from processing_own_phase.phase_06_preprocessing import (
    get_tokenizer,
    tokenize_dataset,
    check_tokenized_sample,
    decode_sanity_check,
    verify_preprocessing,
    verify_dynamic_padding,
    save_preprocessing_summary,
)
from processing_own_phase.presentation_quality import phase_6_examples

tokenizer = get_tokenizer()
tokenized_dataset = tokenize_dataset(dataset, tokenizer)
sample_check = check_tokenized_sample(tokenized_dataset, "train", 0)
decode_checks = decode_sanity_check(tokenized_dataset, tokenizer, dataset, "train", 3)
assert sample_check['all_pass']
assert all(check['pass'] for check in decode_checks)
preprocessing_summary = verify_preprocessing(dataset, tokenized_dataset, tokenizer)
dynamic_padding = verify_dynamic_padding(tokenized_dataset, tokenizer)
assert preprocessing_summary['all_pass']
assert dynamic_padding['all_pass']
preprocessing_summary['dynamic_padding'] = dynamic_padding
preprocessing_summary_path = save...
--- CELL 16 (markdown) ---
## Phase 7 — Model Construction and Sanity Check

Construct the generic pretrained DistilBERT binary classifier and verify its Phase 6 input contract using one deterministic **Train-only** batch. This phase performs no training, validation, Test access, optimizer/scheduler creation, or model-checkpoint saving.
--- CELL 17 (code) ---
# Phase 7: Model Construction + Train-only Forward Sanity Check
from processing_own_phase.phase_06_preprocessing import get_data_collator
from processing_own_phase.phase_07_model_construction import (
    build_model, verify_model_construction, save_model_verification,
)
from processing_own_phase.presentation_quality import phase_7_architecture_figure

model = build_model()
phase_07_verification = verify_model_construction(
    model, tokenizer, tokenized_dataset, get_data_collator(tokenizer)
)
assert phase_07_verification['status'] == 'PASS', phase_07_verification
phase_07_artifact_path = save_model_verification(phase_07_verification)
phase_07_architecture_path = phase_7_architecture_figure()
display(Image(filename=phase_07_architecture_path))
display(Markdown('The pretrained DistilBERT backbone supplies contextual language representations; the classification head maps them to two sentiment logits. Fine-tuning adapts both components to Rotten Tomatoes.'))

display(pd.DataFrame([{
    ...
--- CELL 18 (markdown) ---
## Phase 8 — Metrics and Training Configuration

Verify the binary Accuracy, Precision, Recall and F1 contract; construct the approved baseline `TrainingArguments`; and verify the Validation-only checkpoint ranking policy. This phase creates no `Trainer`, performs no training, and does not access Test.
--- CELL 19 (code) ---
# Phase 8: Metrics + Training Configuration (configuration only)
from processing_own_phase.phase_08_metrics_training_configuration import (
    create_training_arguments,
    verify_phase_08,
    save_phase_08_verification,
)

training_args = create_training_arguments()
phase_08_verification = verify_phase_08(training_args)
assert phase_08_verification['status'] == 'PASS', phase_08_verification
phase_08_artifact_path = save_phase_08_verification(phase_08_verification)

effective = phase_08_verification['effective_training_arguments']
display(pd.DataFrame([
    {'Parameter': 'Model', 'Value': 'DistilBERT'},
    {'Parameter': 'Max Length', 'Value': 80},
    {'Parameter': 'Epochs', 'Value': effective['num_train_epochs']},
    {'Parameter': 'Learning Rate', 'Value': effective['learning_rate']},
    {'Parameter': 'Train / Eval Batch', 'Value': f"{effective['per_device_train_batch_size']} / {effective['per_device_eval_batch_size']}"},
    {'Parameter': 'Optimizer', 'Value': str(training_args...
--- CELL 20 (markdown) ---
## Phase 9 — Fine-Tuning

Run the mandatory debug gate and fresh three-epoch Train/Validation baseline once. On later Run All executions, the guard validates and reloads the selected checkpoint from complete Phase 9 artifacts without retraining. Test remains isolated.
--- CELL 21 (code) ---
# Phase 9: guarded fine-tuning call + artifact presentation
from processing_own_phase.phase_09_fine_tuning import (
    run_or_load_phase_09, load_phase_09_report,
)
from processing_own_phase.phase_10_learning_curves import load_phase_09_artifacts, build_epoch_records

model, phase_09_manifest = run_or_load_phase_09(
    debug_model=model,
    train_dataset=tokenized_dataset['train'],
    validation_dataset=tokenized_dataset['validation'],
    data_collator=get_data_collator(tokenizer),
    tokenizer=tokenizer,
)
phase_09_report = load_phase_09_report()
assert phase_09_report['manifest']['status'] == 'PASS'
assert phase_09_report['debug']['status'] == 'PASS'
assert phase_09_report['manifest']['test_accessed'] is False

phase_09_epochs = build_epoch_records(load_phase_09_artifacts())
display(pd.DataFrame(phase_09_epochs).rename(columns={'epoch': 'Epoch', 'train_loss': 'Train Loss', 'validation_loss': 'Val Loss', 'validation_accuracy': 'Val Accuracy', 'validation_f1': 'Val F1'}).drop(col...
--- CELL 22 (markdown) ---
## Phase 10 — Learning Curves and Training Analysis

Read the verified Phase 9 artifacts to visualize the three-epoch training trajectory and analyze generalization. Phase 10 does not load/evaluate a model or dataset, retrain, access Test, or change the selected checkpoint.
--- CELL 23 (code) ---
# Phase 10: artifact-only learning curves and analysis
from processing_own_phase.phase_10_learning_curves import analyze_learning_curves

phase_10_analysis = analyze_learning_curves()
assert phase_10_analysis['status'] == 'PASS', phase_10_analysis
assert phase_10_analysis['test_accessed'] is False

display(Image(filename=phase_10_analysis['figures']['loss_curves']['path']))
display(Image(filename=phase_10_analysis['figures']['validation_metrics']['path']))
display(Markdown('**Epoch 1 → 2:** Validation loss decreases and Validation F1 improves.  \n**Epoch 2 → 3:** Train loss continues decreasing while Validation loss increases; generalization begins to worsen.  \n**Conclusion:** epoch 2 provides the best observed generalization and remains the selected checkpoint.'))
--- CELL 24 (markdown) ---
## Phase 11 — Validation Verification and Final Test Evaluation

Reload the fixed Phase 9 epoch-2 checkpoint, reproduce its Validation metrics within the approved tolerances, and only then perform the single final Test evaluation. Later Run All executions load the frozen Phase 11 artifacts and do not evaluate Test again.
--- CELL 25 (code) ---
# Phase 11: guarded Validation verification + one-time final Test evaluation
from processing_own_phase.phase_11_final_evaluation import (
    run_or_load_phase_11, load_phase_11_report,
)
from processing_own_phase.presentation_quality import artifact_presentation

phase_11_run = run_or_load_phase_11(
    validation_dataset=tokenized_dataset['validation'],
    data_collator=get_data_collator(tokenizer),
    tokenizer=tokenizer,
    test_data_provider=lambda: (tokenized_dataset['test'], dataset['test']),
)
phase_11_report = load_phase_11_report()
presentation_report = artifact_presentation()
assert phase_11_report['checkpoint']['status'] == 'PASS'
assert phase_11_report['validation']['status'] == 'PASS'
assert phase_11_report['test']['status'] == 'PASS'
assert phase_11_report['manifest']['status'] == 'FINAL_TEST_COMPLETE'
assert phase_11_report['manifest']['test_evaluation_count'] == 1
assert phase_11_report['manifest']['test_used_for_selection'] is False
assert phase_11_report['manifest...
--- CELL 26 (markdown) ---
## Phase 12 — Confusion Matrix and Error Analysis

Analyze only the frozen Phase 11 prediction artifact. This section derives the confusion matrix, class-level behavior and deterministic representative errors without loading a model/dataset, reevaluating Test, changing the checkpoint or starting Phase 13.
--- CELL 27 (code) ---
# Phase 12: frozen-artifact confusion matrix + deterministic error analysis
from processing_own_phase.phase_12_error_analysis import analyze_phase_12

phase_12_result = analyze_phase_12()
phase_12_confusion = phase_12_result['confusion']
phase_12_errors = phase_12_result['error_samples']
phase_12_analysis = phase_12_result['analysis']
assert phase_12_result['status'] == 'PASS'
assert phase_12_analysis['test_evaluation_count'] == 1
assert phase_12_analysis['model_loaded'] is False
assert phase_12_analysis['dataset_loaded'] is False
assert phase_12_analysis['test_provider_called'] is False
assert phase_12_analysis['test_evaluated'] is False
assert phase_12_analysis['training_performed'] is False
assert phase_12_analysis['checkpoint_changed'] is False
assert phase_12_analysis['phase_13_started'] is False

display(Markdown('### Frozen Final-Test confusion matrix'))
display(Image(filename=phase_12_confusion['figure']['path']))
display(pd.DataFrame([{'TN': phase_12_confusion['counts']['TN'],...
--- CELL 28 (markdown) ---
## Phase 13 — New-Sentence Inference

Run guarded inference only on newly authored custom sentences using the fixed epoch-2 `checkpoint-1068`. The module verifies the Phase 11 SHA-256 fingerprint, reuses the locked Phase 6 tokenizer contract (`max_length=80`, truncation and dynamic padding), and never loads or reevaluates Test.
--- CELL 29 (code) ---
# Phase 13: guarded custom-text inference using the authoritative checkpoint
from processing_own_phase.phase_13_new_sentence_inference import (
    DEFAULT_CUSTOM_SENTENCES, run_or_load_phase_13,
)
from processing_own_phase.presentation_quality import phase_13_presentation_examples, phase_13_probability_figure

phase_13_report = run_or_load_phase_13(DEFAULT_CUSTOM_SENTENCES)
assert phase_13_report['status'] == 'PASS'
assert phase_13_report['checkpoint']['checkpoint_name'] == 'checkpoint-1068'
assert phase_13_report['checkpoint']['epoch'] == 2.0
assert phase_13_report['checkpoint']['step'] == 1068
assert phase_13_report['logits_shape'] == [len(DEFAULT_CUSTOM_SENTENCES), 2]
assert all(phase_13_report['verification']['checks'].values())
assert phase_13_report['training_performed'] is False
assert phase_13_report['test_accessed'] is False
assert phase_13_report['test_evaluated'] is False
assert phase_13_report['test_evaluation_count'] == 1
assert phase_13_report['checkpoint_changed'] is Fa...
--- CELL 30 (markdown) ---
## Phase 14 — Save & Reload

Verify the authoritative epoch-2 `checkpoint-1068`, save a reusable Hugging Face model/tokenizer package once, and prove that its locally reloaded state and predictions are equivalent. Later Run All executions verify/load the package without resaving, training or Test reevaluation.
--- CELL 31 (code) ---
# Phase 14: guarded package integrity + save/reload equivalence
from processing_own_phase.phase_14_save_reload import run_or_load_phase_14

phase_14_report = run_or_load_phase_14()
assert phase_14_report['status'] == 'PASS'
assert phase_14_report['source_checkpoint']['checkpoint_name'] == 'checkpoint-1068'
assert phase_14_report['source_checkpoint']['epoch'] == 2.0
assert phase_14_report['source_checkpoint']['step'] == 1068
assert phase_14_report['state_dict_equivalence']['status'] == 'PASS'
assert phase_14_report['state_dict_equivalence']['all_tensors_exact_equal'] is True
assert phase_14_report['prediction_equivalence']['status'] == 'PASS'
assert all(phase_14_report['completion_checks'].values())
assert phase_14_report['training_performed'] is False
assert phase_14_report['test_accessed'] is False
assert phase_14_report['test_evaluated'] is False
assert phase_14_report['test_evaluation_count'] == 1
assert phase_14_report['checkpoint_selected_or_changed'] is False
assert phase_14_repo...
--- CELL 32 (markdown) ---
## Phase 15 — Final Summary

Consolidate the verified Phase 0–14 artifacts into the final Practice 3 conclusion. This section is artifact-only: it does not load a model/dataset, train, infer, evaluate Test, change a checkpoint or resave the Phase 14 package.
--- CELL 33 (code) ---
# Phase 15: artifact-only final project summary
from processing_own_phase.phase_15_final_summary import run_or_load_phase_15

phase_15_report = run_or_load_phase_15()
assert phase_15_report['status'] == 'PASS'
assert all(status == 'PASS' for status in phase_15_report['phase_status'].values())
assert all(phase_15_report['verification_checks'].values())
assert phase_15_report['final_model']['selected_checkpoint'] == 'checkpoint-1068'
assert phase_15_report['final_model']['epoch'] == 2.0
assert phase_15_report['final_model']['step'] == 1068
assert phase_15_report['dataset_contract']['split_counts'] == {'train': 8530, 'validation': 1066, 'test': 1066}
assert phase_15_report['test_evaluation_count'] == 1
assert phase_15_report['model_loaded'] is False
assert phase_15_report['dataset_loaded'] is False
assert phase_15_report['training_performed'] is False
assert phase_15_report['inference_performed'] is False
assert phase_15_report['test_accessed'] is False
assert phase_15_report['test_evalua...
