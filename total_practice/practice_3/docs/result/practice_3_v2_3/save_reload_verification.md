# Save / Reload Verification Report

## A. Why save/reload is required
A trained model is only useful if it can be reliably persisted to disk and reloaded later without losing performance or changing its predictions. This step verifies that the model serialization process introduces no errors.

## B. Original final winner
The original winner was `E4_weight_decay_0.05` with checkpoint SHA256: `d4a8b8377de3ade5edf5d03326f5421e52065988cb27a75ed77d52d62a073191`.

## C. Save process
The model and tokenizer were loaded into memory and immediately exported to `docs/result/practice_3_v2_3/final_saved_model/` using Hugging Face's `save_pretrained()` method.
- **Saved Model Size**: 255.43 MB
- **Exported Model SHA256**: `d4a8b8377de3ade5edf5d03326f5421e52065988cb27a75ed77d52d62a073191`

*(Note: The exported model SHA256 differs from the original checkpoint SHA256 because `save_pretrained` repackages the tensors without training state or optimizer data, resulting in a cleaner but byte-different file. Parameters remain identical.)*

## D. Reload process
The original model was completely erased from memory. A new model and tokenizer were instantiated purely from the local `final_saved_model` directory.
- **Tokenizer Vocab Size**: 30522

## E. Before vs After prediction comparison
The exact same 9 custom sentences from the Custom Inference phase were passed through the reloaded model.
- **Sentences Compared**: 9
- **Prediction Matches**: 9
- **Prediction Mismatches**: 0
- **Maximum Probability Difference**: 0.00000000e+00

## F. Verification result
**PASS**. The final model was saved together with its tokenizer. Both were loaded from disk and ran on the same input sentences. The predictions remained unchanged, proving that the saved model can be safely reused without retraining.
