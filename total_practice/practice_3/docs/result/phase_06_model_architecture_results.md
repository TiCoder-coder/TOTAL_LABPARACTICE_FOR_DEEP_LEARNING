# Phase 06 – Model Architecture

## Notebook Reference
Notebook:
[`practice_3.ipynb`](../../notebook_practice_3/practice_3.ipynb)

Cell:
[`Cell 14` (Code)](../../notebook_practice_3/practice_3.ipynb)

Cell output:
Styled DataFrame table titled `Model Architecture Configuration` displaying structural hyperparameters loaded from the saved model configuration.

## Processing Source
- [`pretraining_integration_v2.py`](../../processing_own_phase/pretraining_integration_v2.py)
- Hugging Face `DistilBertForSequenceClassification`

## Artifact Sources
- [`final_saved_model/config.json`](./practice_3_v2_3/final_saved_model/config.json)
- [`final_saved_model_manifest.json`](./practice_3_v2_3/final_saved_model_manifest.json)

## Results
| Attribute | Value | Description |
|---|---|---|
| **Architecture** | `DistilBertForSequenceClassification` | Pretrained DistilBERT with a linear classification head |
| **Model Type** | `distilbert` | 6-layer distilled bidirectional transformer |
| **Hidden Dimension ($d_{model}$)** | 768 | Dimensionality of encoder layers and pooler |
| **Feed-Forward Dim ($d_{ffn}$)** | 3,072 | Intermediate dimension in feed-forward sub-layers |
| **Number of Layers ($n_{layers}$)** | 6 | Transformer encoder blocks |
| **Attention Heads ($n_{heads}$)** | 12 | Multi-head self-attention mechanisms (64 dim/head) |
| **Vocab Size** | 30,522 | WordPiece token embedding matrix size |
| **Max Position Embeddings** | 512 | Maximum absolute positional encoding span |
| **Activation** | `gelu` | Gaussian Error Linear Unit |
| **Output Classes** | 2 | Binary sentiment logits (`0: NEGATIVE`, `1: POSITIVE`) |
| **Total Parameters** | 66,955,010 | ~66.95M parameters (100% fine-tuned) |

## Evidence
- `Model Architecture Configuration` table rendered in Notebook [`Cell 14`](../../notebook_practice_3/practice_3.ipynb).
- Checkpoint configuration verified in [`final_saved_model/config.json`](./practice_3_v2_3/final_saved_model/config.json).

## Summary
The model employs `DistilBertForSequenceClassification`, combining a 6-layer pretrained DistilBERT backbone (768 hidden dim, 12 attention heads) with a 2-class linear projection head for binary sentiment prediction.
