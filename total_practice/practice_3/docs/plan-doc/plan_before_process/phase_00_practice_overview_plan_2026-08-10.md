# Phase 0 Practice Overview Plan

**Date:** 2026-08-10  
**Phase:** 0/15  
**File path:** `docs/plan-doc/plan_before_process/phase_00_practice_overview_plan_2026-08-10.md`

---

## 1. Knowledge Overview

### 1.1 Objective

Clearly define the entire Practice 3 before starting any implementation.

Specifically:

- Clarify the problem to be solved (Binary Sentiment Classification).
- Distinguish between the two layers of knowledge:
  - **Pretraining** → Language knowledge
  - **Downstream Fine-tuning** → Sentiment classification knowledge
- Define the scope and boundary between Exercise 1 and Exercise 2.
- Establish a solid foundation for the entire Stage 1 (Phase 0 → Phase 6).
- Ensure compliance with the clean Transfer Learning principle.

### 1.2 Scientific and Theoretical Foundation

**(a) Why Choose Rotten Tomatoes**

| Criteria                                          | Rotten Tomatoes | IMDb                       |
| ------------------------------------------------- | --------------- | -------------------------- |
| Labeled samples                                   | 10,662          | 50,000                     |
| Train split                                       | 8,530           | 25,000                     |
| Validation split                                  | 1,066           | No separate official split |
| Test split                                        | 1,066           | 25,000                     |
| Binary sentiment                                  | Yes             | Yes                        |
| Typical text length                               | Short           | Longer                     |
| Fine-tuning cost                                  | Lower           | Higher                     |
| Suitability for a compact practice notebook       | Very high       | High                       |
| Alignment with the official Hugging Face tutorial | Medium          | Very high                  |

**Reason for choosing the dataset:** a binary classification task, a moderate computational workload for a practice notebook, three existing train/validation/test splits, relatively short reviews, no need to create an additional validation split, and support for a clean experimental workflow on a personal computer.

Dataset:

```
Train: 8,530
Validation: 1,066
Test: 1,066
Total: 10,662

Label 0: NEGATIVE
Label 1: POSITIVE
```

**(b) Why Choose DistilBERT**

```
flowchart LR
    A[Tokenized Text] --> B[DistilBERT Backbone]
    B --> C[Contextual Representation]
    C --> D[Classification Head]
    D --> E[Two Logits]
    E --> F[NEGATIVE or POSITIVE]
```

| Model        | Main Advantages                | Main Limitations                                        |
| ------------ | ------------------------------ | ------------------------------------------------------- |
| DistilBERT   | Lightweight, practical         | Lower capacity than BERT-base                           |
| BERT-base    | Classic Transformer baseline   | Higher computational cost                               |
| RoBERTa-base | Strong language representation | Higher computational cost                               |
| MiniLM       | Very lightweight               | Less aligned with the traditional introductory workflow |
| ALBERT       | Parameter-efficient            | Different architectural characteristics                 |

Primary implementation choice: **DistilBERT** because it is a pretrained Transformer, is lighter than BERT-base, directly supports sequence classification, is suitable for English sentiment analysis, reduces training cost while preserving the nature of the transfer-learning workflow, and is suitable for a student lab environment.

**(c) Conceptual Nature of Transfer Learning**

```
flowchart TD
    A[Pretraining] --> B[General Language Knowledge]
    B --> C[Downstream Fine-Tuning]
    C --> D[Binary Sentiment Knowledge]
    D --> E[Inference]
    E --> F[Positive or Negative Prediction]
```

```
Exercise 1
Already fine-tuned model
        |
        v
Inference

Exercise 2
Generic pretrained model
        |
        v
Task-specific fine-tuning
        |
        v
Binary classifier

```

**(d) How is Fine-tuning Different from Feature Extraction**

```
Feature extraction
Freeze Transformer backbone
        |
        v
Train only classifier

Fine-tuning
Update Transformer backbone
        +
Update classification head
```

**(e) Things That Must Never Be Done Throughout Practice 3**

```
Do not:
Fine-tune on the test set

Do not:
Use test performance to choose epochs or hyperparameters

Do not:
Aggressively remove stopwords, punctuation, or linguistic structure without justification

Do not:
Train DistilBERT from scratch for this practice

Do not:
Use an already sentiment-fine-tuned checkpoint in Exercise 2
and describe the process as generic pretrained-model fine-tuning
without explicitly documenting the checkpoint's prior task-specific training

Do not:
Fabricate loss curves, metrics, confusion matrices, or benchmark results
```

### 1.3 Overall End-to-End Pipeline Map

```
flowchart TD
    S[START] --> A[Environment and Seeds]

    A --> B[Exercise 1]
    B --> C[Load Fine-Tuned Sentiment Model]
    C --> D[Inspect Sentence Tokens]
    D --> E[Run Sentiment Inference]

    E --> F[Exercise 2]
    F --> G[Load Rotten Tomatoes Dataset]
    G --> H[EDA and Data Quality Checks]
    H --> I[Load DistilBERT Tokenizer]
    I --> J[Token-Length Analysis]
    J --> K[Tokenize Dataset]
    K --> L[Dynamic Padding]
    L --> M[Load Pretrained DistilBERT Classifier]
    M --> N[Model Sanity Check]
    N --> O[Define Metrics]
    O --> P[Configure TrainingArguments]
    P --> Q[Create Trainer]
    Q --> R[Debug Subset Run]
    R --> T{Pipeline Valid?}
    T -- No --> U[Fix and Re-run]
    U --> R
    T -- Yes --> V[Full Fine-Tuning]
    V --> W[Validation Monitoring]
    W --> X[Load Best Checkpoint]
    X --> Y[Final Test Evaluation]
    Y --> Z[Confusion Matrix]
    Z --> AA[Error Analysis]
    AA --> AB[New-Sentence Inference]
    AB --> AC[Save Model and Tokenizer]
    AC --> AD[Reload Sanity Test]
    AD --> AE[FINAL SUMMARY]
```

### 1.4 16-Phase Diagram According to Notebook Architecture

```
flowchart TD
    P0[Phase 0<br/>Practice Overview] --> P1[Phase 1<br/>Environment and Reproducibility]
    P1 --> P2[Phase 2<br/>Pretrained Sentiment Inference]
    P2 --> P3[Phase 3<br/>Tokenization Investigation]
    P3 --> P4[Phase 4<br/>Dataset Loading]
    P4 --> P5[Phase 5<br/>EDA and Sanity Checks]
    P5 --> P6[Phase 6<br/>Tokenizer and Preprocessing]
    P6 --> P7[Phase 7<br/>Model Construction]
    P7 --> P8[Phase 8<br/>Metrics and Training Configuration]
    P8 --> P9[Phase 9<br/>Fine-Tuning]
    P9 --> P10[Phase 10<br/>Learning Curves]
    P10 --> P11[Phase 11<br/>Validation and Test Evaluation]
    P11 --> P12[Phase 12<br/>Error Analysis]
    P12 --> P13[Phase 13<br/>New-Sentence Inference]
    P13 --> P14[Phase 14<br/>Save and Reload]
    P14 --> P15[Phase 15<br/>Final Summary]
```

### 1.5 Final Conceptual Summary: What Must the Notebook Demonstrate?

```
flowchart LR
    A[General Language Pretraining] --> B[Pretrained DistilBERT]
    B --> C[Binary Sentiment Fine-Tuning]
    C --> D[Task-Specific Classifier]
    D --> E[Validation]
    E --> F[Independent Test Evaluation]
    F --> G[Reusable Sentiment Model]
```

```
Exercise 1
Pretrained model reuse
        +
Tokenizer understanding

Exercise 2
Transfer learning
        +
Fine-tuning
        +
Scientific evaluation
        +
Reusable model artifacts
```

---

## 2. Input

| Source               | Content                                                                                            |
| -------------------- | -------------------------------------------------------------------------------------------------- |
| Workflow Rule        | `PRACTICE3_WORKFLOW_HANDOFF_RULE.md`                                                               |
| Overall Plan         | `docs/plan-doc/plan_overview/plan.md`                                                              |
| Technical Constraint | Exercise 2 must start from `distilbert-base-uncased` (do not use a sentiment-finetuned checkpoint) |
| Dataset              | Rotten Tomatoes                                                                                    |

---

## 3. Problem Analysis

### 3.1. Problem Definition

| Thành phần    | Giá trị                                                  |
| ------------- | -------------------------------------------------------- |
| **Task type** | Binary Text Classification (Sentiment Analysis)          |
| **Input**     | English text sentence (movie review)                     |
| **Output**    | Label `Positive` (1) / `Negative` (0) + confidence score |
| **Domain**    | Movie review sentiment                                   |
| **Dataset**   | Rotten Tomatoes (Hugging Face)                           |

### 3.2. Academic Architecture to Master

     ```

PRETRAINING

Language knowledge
(distilbert-base-uncased)
↓
DOWNSTREAM FINE-TUNING
Sentiment classification knowledge
↓
INFERENCE
Positive / Negative prediction

````

### 3.3. Stage 1 Scope Division

| Phase | Phase Name                      | Main Objective                                 |
| ----- | ------------------------------ | ---------------------------------------------- |
| 0     | Practice Overview              | Problem definition + academic architecture      |
| 1     | Environment & Reproducibility  | Set up environment, seed, device                 |
| 2     | Pretrained Sentiment Inference | Exercise 1 – Inference with a fine-tuned model |
| 3     | Tokenization Investigation     | Tokenizer analysis                            |
| 4     | Dataset Loading                | Load Rotten Tomatoes                           |
| 5     | Dataset EDA & Sanity Checks    | Data analysis + quality checks        |
| 6     | Tokenizer & Preprocessing      | Prepare data for fine-tuning                  |

---

## 4. Key Technical Decisions (Decision Log)

### Decision 1: Model for Exercise 1

* **Decision:** Use `distilbert-base-uncased-finetuned-sst-2-english`
* **Reason:** This model has already been fine-tuned on SST-2 and is suitable for demonstrating inference.
* **Status:** Approved

### Decision 2: Model for Exercise 2

* **Decision:** Start from `distilbert-base-uncased` (generic pretrained)
* **Reason:** Demonstrate clean Transfer Learning (language knowledge → task knowledge). A sentiment-finetuned checkpoint must not be used.
* **Trade-off:** Fine-tuning will take longer and have lower initial performance than continuing to fine-tune from a sentiment checkpoint, but it matches the academic objective.
* **Status:** Approved

### Decision 3: Dataset

- **Decision:** Rotten Tomatoes
- **Reason:** A standard binary sentiment dataset, available on Hugging Face `datasets`, suitable for the Practice objective.
- **Status:** Approved

### Decision 4: Hardware

- **Decision:** Run on CPU (`torch==2.13.0+cpu`)
- **Reason:** Duyên's current environment does not have CUDA.
- **Trade-off:** Training (Phase 9) will be slow → batch size and the number of epochs need to be designed appropriately in later phases.
- **Status:** Approved

---

## 5. Expected Output của Phase 0

After completing Phase 0, there must be:

1. This plan file saved at the correct path and with the correct naming.
2. The core technical decisions finalized and clearly recorded in the Decision Log.
3. The Overview content clear enough to be placed in the first cell of the notebook.
4. A solid foundation for writing the Phase 1 plan without going off track.

---

## 6. Files Affected

| File                                                                              | Action                               |
| --------------------------------------------------------------------------------- | --------------------------------------- |
| `docs/plan-doc/plan_before_process/phase_00_practice_overview_plan_2026-08-10.md` | Create new                                 |
| `notebook_practice_3/practice_3.ipynb`                                            | An Overview cell will be added during implementation |

---

## 7. Validation / Sanity Checks

- [ ] Exercise 1 (inference) and Exercise 2 (fine-tuning from a generic checkpoint) are clearly distinguished
- [ ] The Pretraining → Downstream Fine-tuning → Inference architecture is clearly stated
- [ ] The dataset is identified as Rotten Tomatoes
- [ ] There is no remaining ambiguity about the model checkpoint used for each exercise
- [ ] All technical constraints have been fully documented
- [ ] The Decision Log has been clearly written
- [ ] All Markdown tables display correctly when Run/Preview is used
- [ ] Mermaid renders correctly as diagrams in VSCode/Jupyter
- [ ] There are no code cells in the notebook in Phase 0
- [ ] The cell order in the notebook follows the presentation order in this plan file

---

## 8. Completion Criteria

Phase 0 is considered complete when:

1. The plan file is saved in the correct directory and follows the correct naming convention.
2. The core technical decisions have been finalized.
3. This plan can be used to write the Phase 1 plan without needing to ask again about the overall direction.

---

## 9. Risks & Notes

| Risk                                | Severity     | Notes                                                |
| ----------------------------------- | ---------- | ------------------------------------------------------ |
| Confusion between the two model checkpoints  | High        | Exercise 1 and Exercise 2 must always be clearly distinguished        |
| Misunderstanding the objective of Transfer Learning | Medium | The principle of “starting from generic pretrained” must be maintained |
| Incorrect file naming convention   | Low       | The date-based format has been standardized                 |

---

## 10. Next Step

After Phase 0 is approved → proceed to write the Phase 1 plan:

```text
docs/plan-doc/plan_before_process/phase_01_environment_plan_2026-08-10.md
````
