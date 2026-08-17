# Phase 05 – Tokenization

## Notebook Reference
Notebook:
[`practice_3.ipynb`](../../notebook_practice_3/practice_3.ipynb)

Cell:
[`Cell 09` (Code)](../../notebook_practice_3/practice_3.ipynb)

Cell output:
Raw review sample text followed by a styled DataFrame table titled `Tokenization Example` mapping Token IDs to decoded subwords.

## Processing Source
- [`pretraining_integration_v2.py`](../../processing_own_phase/pretraining_integration_v2.py)
- Transformers `AutoTokenizer.from_pretrained("distilbert-base-uncased")`

## Artifact Sources
- [`tokenization_demo.json`](./practice_3_v2_3/tokenization_demo.json)
- [`tokenizer_validation.json`](./practice_3_v2_3/tokenizer_validation.json)

## Results
| Tokenizer Parameter | Value |
|---|---|
| **Tokenizer Model** | `distilbert-base-uncased` |
| **Algorithm** | WordPiece |
| **Vocabulary Size** | 30,522 |
| **Max Sequence Length** | 80 tokens (padded/truncated) |
| **Special Tokens** | `[CLS]` (101), `[SEP]` (102), `[PAD]` (0), `[UNK]` (100), `[MASK]` (103) |
| **Padding Strategy** | Dynamic batch padding via `DataCollatorWithPadding` |

### Sample Tokenization Mapping
- **Input Text**: *"the rock is destined to be the 21st century's new \" conan \" and that he's going to make a splash..."*
- **First 10 Encoded Tokens**:
  - `101` $\rightarrow$ `[CLS]`
  - `1996` $\rightarrow$ `the`
  - `2600` $\rightarrow$ `rock`
  - `2003` $\rightarrow$ `is`
  - `16035` $\rightarrow$ `destined`
  - `2000` $\rightarrow$ `to`
  - `2022` $\rightarrow$ `be`
  - `1996` $\rightarrow$ `the`
  - `7398` $\rightarrow$ `21st`
  - `2301` $\rightarrow$ `century`

## Evidence
- Tokenization demo table rendered in Notebook [`Cell 09`](../../notebook_practice_3/practice_3.ipynb).
- Tokenizer validation schema in [`tokenization_demo.json`](./practice_3_v2_3/tokenization_demo.json).

## Summary
The `distilbert-base-uncased` tokenizer converts text into numeric token IDs with WordPiece subword splitting. Inputs are bounded at max length 80 and prepended with `[CLS]` and appended with `[SEP]` for transformer classification.
