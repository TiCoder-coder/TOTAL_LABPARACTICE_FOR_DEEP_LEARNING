# Phase 12 – Custom Inference

## Notebook Reference
Notebook:
[`practice_3.ipynb`](../../notebook_practice_3/practice_3.ipynb)

Cell:
[`Cell 28` (Code)](../../notebook_practice_3/practice_3.ipynb)

Cell output:
Styled DataFrame table titled `Custom Inference Results` showing 9 handcrafted test sentences evaluated with predicted labels, softmax confidences, expected targets, and match indicators.

## Processing Source
- [`custom_inference_v2_3.py`](../../processing_own_phase/custom_inference_v2_3.py)
- Function: `run_custom_inference()`

## Artifact Sources
- [`custom_inference_results.csv`](./practice_3_v2_3/custom_inference_results.csv)
- [`custom_inference_results.json`](./practice_3_v2_3/custom_inference_results.json)
- [`custom_inference_inputs.json`](./practice_3_v2_3/custom_inference_inputs.json)

## Results
| Category | Test Sentence | Predicted Label | Confidence | Expected | Match |
|---|---|:---:|:---:|:---:|:---:|
| **STANDARD_POSITIVE** | *"A truly masterpiece that captures the essence of human emotion and resilience."* | Positive | 0.9760 | Positive | True |
| **STANDARD_POSITIVE** | *"Incredible performances and brilliant direction make this an unforgettable experience."* | Positive | 0.9782 | Positive | True |
| **STANDARD_NEGATIVE** | *"A completely uninspired waste of time with flat characters and terrible pacing."* | Negative | 0.9744 | Negative | True |
| **STANDARD_NEGATIVE** | *"Painfully boring and predictable from the first minute to the last."* | Negative | 0.9311 | Negative | True |
| **SUBTLE_POSITIVE** | *"It is not perfect, but I would gladly watch it again on a rainy afternoon."* | Negative | 0.5462 | Positive | False |
| **SUBTLE_NEGATIVE** | *"The movie looks beautiful, although there is little else to recommend about it."* | Positive | 0.5888 | Negative | False |
| **AMBIGUOUS** | *"I really expected more from this film given the cast."* | Negative | 0.6623 | None | — |
| **CONTRAST** | *"The beginning was excruciatingly slow, but the ending made the whole movie completely worthwhile."* | Positive | 0.8839 | Positive | True |
| **NEGATION** | *"Not exactly what I was hoping for."* | Negative | 0.9050 | Negative | True |

## Evidence
- `Custom Inference Results` table rendered in Notebook [`Cell 28`](../../notebook_practice_3/practice_3.ipynb).
- Inference JSON dump in [`custom_inference_results.json`](./practice_3_v2_3/custom_inference_results.json).

## Summary
The model demonstrates strong performance on standard sentiment (97%+ confidence), explicit negation ("Not exactly..."), and strong contrastive conjunctions. Subtle mixed reviews exhibit expected uncertainty with confidences hovering near the 0.50–0.58 decision threshold.
