# Phase 0 Practice Overview Plan

**Ngày:** 2026-08-10  
**Người thực hiện:** Duyên  
**Phase:** 0/15  
**File lưu:** `docs/plan-doc/plan_before_process/phase_00_practice_overview_plan_2026-08-10.md`

---

## 1. Tổng quan kiến thức

### 1.1 Mục tiêu

Định nghĩa rõ ràng toàn bộ Practice 3 trước khi bắt đầu bất kỳ implementation nào.

Cụ thể:

- Làm rõ bài toán cần giải quyết (Binary Sentiment Classification).
- Phân biệt hai tầng kiến thức:
  - **Pretraining** → Language knowledge
  - **Downstream Fine-tuning** → Sentiment classification knowledge
- Xác định phạm vi và ranh giới giữa Exercise 1 và Exercise 2.
- Thiết lập foundation vững chắc cho toàn bộ Stage 1 (Phase 0 → Phase 6).
- Đảm bảo tuân thủ nguyên tắc Transfer Learning sạch.

### 1.2 Nền tảng khoa học và lý thuyết

**(a) Vì sao chọn Rotten Tomatoes**

| Tiêu chí                                 | Rotten Tomatoes | IMDb                            |
| ---------------------------------------- | --------------- | ------------------------------- |
| Labeled samples                          | 10,662          | 50,000                          |
| Train split                              | 8,530           | 25,000                          |
| Validation split                         | 1,066           | Không có split chính thức riêng |
| Test split                               | 1,066           | 25,000                          |
| Binary sentiment                         | Có              | Có                              |
| Độ dài văn bản điển hình                 | Ngắn            | Dài hơn                         |
| Chi phí fine-tuning                      | Thấp hơn        | Cao hơn                         |
| Phù hợp cho notebook thực hành gọn       | Rất cao         | Cao                             |
| Bám sát tutorial chính thức Hugging Face | Trung bình      | Rất cao                         |

**Lý do chọn bộ dataset:** bài toán phân loại nhị phân, khối lượng tính toán vừa phải cho notebook thực hành, đã có sẵn 3 split train/validation/test, review tương đối ngắn, tránh phải tự tạo thêm validation split, hỗ trợ một quy trình thực nghiệm sạch trên máy cá nhân.

Dataset:

```
Train: 8,530
Validation: 1,066
Test: 1,066
Total: 10,662

Label 0: NEGATIVE
Label 1: POSITIVE
```

**(b) Vì sao chọn DistilBERT**

```
flowchart LR
    A[Tokenized Text] --> B[DistilBERT Backbone]
    B --> C[Contextual Representation]
    C --> D[Classification Head]
    D --> E[Two Logits]
    E --> F[NEGATIVE or POSITIVE]
```

| Model        | Ưu điểm chính                      | Hạn chế chính                                |
| ------------ | ---------------------------------- | -------------------------------------------- |
| DistilBERT   | Nhẹ, thực tế                       | Capacity thấp hơn BERT-base                  |
| BERT-base    | Baseline kinh điển của Transformer | Chi phí tính toán cao hơn                    |
| RoBERTa-base | Biểu diễn ngôn ngữ mạnh            | Chi phí tính toán cao hơn                    |
| MiniLM       | Rất nhẹ                            | Ít bám sát quy trình giới thiệu truyền thống |
| ALBERT       | Hiệu quả về tham số                | Đặc tính kiến trúc khác biệt                 |

Lựa chọn triển khai chính: **DistilBERT** vì là Transformer đã pretrained, nhẹ hơn BERT-base, hỗ trợ trực tiếp sequence classification, phù hợp sentiment analysis tiếng Anh, giảm chi phí training nhưng vẫn giữ nguyên bản chất transfer-learning workflow, phù hợp môi trường lab của sinh viên.

**(c) Bản chất khái niệm Transfer Learning**

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

**(d) Fine-tuning khác Feature Extraction như thế nào**

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

**(e) Những điều tuyệt đối không được làm trong toàn bộ Practice 3**

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

### 1.3 Bản đồ tổng thể Pipeline End-to-End

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

### 1.4 Sơ đồ 16 Phase theo notebook architecture

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

### 1.5 Final Conceptual Summary Notebook phải chứng minh được điều gì?

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

| Nguồn              | Nội dung                                                                                                |
| ------------------ | ------------------------------------------------------------------------------------------------------- |
| Workflow Rule      | `PRACTICE3_WORKFLOW_HANDOFF_RULE.md`                                                                    |
| Plan tổng          | `docs/plan-doc/plan_overview/plan.md`                                                                   |
| Ràng buộc kỹ thuật | Exercise 2 bắt buộc bắt đầu từ `distilbert-base-uncased` (không dùng checkpoint đã sentiment-finetuned) |
| Dataset            | Rotten Tomatoes                                                                                         |

---

## 3. Phân tích bài toán

### 3.1. Problem Definition

| Thành phần    | Giá trị                                                 |
| ------------- | ------------------------------------------------------- |
| **Task type** | Binary Text Classification (Sentiment Analysis)         |
| **Input**     | Câu văn bản tiếng Anh (movie review)                    |
| **Output**    | Nhãn `Positive` (1) / `Negative` (0) + confidence score |
| **Domain**    | Movie review sentiment                                  |
| **Dataset**   | Rotten Tomatoes (Hugging Face)                          |

### 3.2. Kiến trúc học thuật cần nắm vững

     PRETRAINING

Language knowledge
(distilbert-base-uncased)
↓
DOWNSTREAM FINE-TUNING
Sentiment classification knowledge
↓
INFERENCE
Positive / Negative prediction

- **Exercise 1** bắt đầu ở tầng cuối: already fine-tuned model → Inference.
- **Exercise 2** thực hiện bước giữa: generic pretrained model → Fine-tuning → Classifier.

### 3.3. Phân chia phạm vi Stage 1

| Phase | Tên Phase                      | Mục tiêu chính                                 |
| ----- | ------------------------------ | ---------------------------------------------- |
| 0     | Practice Overview              | Định nghĩa bài toán + kiến trúc học thuật      |
| 1     | Environment & Reproducibility  | Setup môi trường, seed, device                 |
| 2     | Pretrained Sentiment Inference | Exercise 1 – Inference với model đã fine-tuned |
| 3     | Tokenization Investigation     | Phân tích tokenizer                            |
| 4     | Dataset Loading                | Load Rotten Tomatoes                           |
| 5     | Dataset EDA & Sanity Checks    | Phân tích dữ liệu + kiểm tra chất lượng        |
| 6     | Tokenizer & Preprocessing      | Chuẩn bị data cho fine-tuning                  |

---

## 4. Quyết định kỹ thuật quan trọng (Decision Log)

### Decision 1 Model cho Exercise 1

- **Quyết định:** Dùng `distilbert-base-uncased-finetuned-sst-2-english`
- **Lý do:** Đây là model đã được fine-tuned sẵn trên SST-2, phù hợp để minh họa inference.
- **Status:** Approved

### Decision 2 Model cho Exercise 2

- **Quyết định:** Bắt đầu từ `distilbert-base-uncased` (generic pretrained)
- **Lý do:** Minh họa Transfer Learning sạch (language knowledge → task knowledge). Không được dùng checkpoint đã sentiment-finetuned.
- **Trade-off:** Fine-tuning sẽ lâu hơn và performance ban đầu thấp hơn so với tiếp tục fine-tune từ sentiment checkpoint, nhưng đúng mục tiêu học thuật.
- **Status:** Approved

### Decision 3 Dataset

- **Quyết định:** Rotten Tomatoes
- **Lý do:** Dataset binary sentiment chuẩn, có sẵn trên Hugging Face `datasets`, phù hợp với mục tiêu Practice.
- **Status:** Approved

### Decision 4 Hardware

- **Quyết định:** Chạy trên CPU (`torch==2.13.0+cpu`)
- **Lý do:** Môi trường hiện tại của Duyên không có CUDA.
- **Trade-off:** Training (Phase 9) sẽ chậm → cần thiết kế batch size và số epoch hợp lý ở các phase sau.
- **Status:** Approved

---

## 5. Expected Output của Phase 0

Sau khi hoàn thành Phase 0 phải có:

1. File plan này được lưu đúng đường dẫn và naming.
2. Các quyết định kỹ thuật cốt lõi đã được chốt và ghi rõ trong Decision Log.
3. Nội dung Overview đủ rõ để đưa vào cell đầu tiên của notebook.
4. Foundation vững để viết plan Phase 1 mà không bị lệch hướng.

---

## 6. Files Affected

| File                                                                              | Hành động                               |
| --------------------------------------------------------------------------------- | --------------------------------------- |
| `docs/plan-doc/plan_before_process/phase_00_practice_overview_plan_2026-08-10.md` | Tạo mới                                 |
| `docs/plan-doc/plan_overview/plan.md`                                             |                                         |
| `notebook_practice_3/practice_3.ipynb`                                            | Sẽ thêm cell Overview ở giai triển khai |

---

## 7. Validation / Sanity Checks

- [ ] Đã phân biệt rõ Exercise 1 (inference) và Exercise 2 (fine-tuning từ generic checkpoint)
- [ ] Đã nêu rõ kiến trúc Pretraining → Downstream Fine-tuning → Inference
- [ ] Dataset được xác định là Rotten Tomatoes
- [ ] Không còn ambiguity về model checkpoint dùng cho mỗi exercise
- [ ] Các ràng buộc kỹ thuật đã được ghi nhận đầy đủ
- [ ] Decision Log đã được viết rõ ràng
- [ ] Tất cả bảng markdown hiển thị đúng định dạng khi Run/Preview

---

## 8. Completion Criteria

Phase 0 được xem là hoàn thành khi:

1. File plan được lưu đúng thư mục và đúng naming convention.
2. Các quyết định kỹ thuật cốt lõi đã được chốt.
3. Có thể dựa vào plan này để viết plan Phase 1 mà không cần hỏi lại về hướng đi tổng thể.

---

## 9. Risks & Notes

| Risk                                | Mức độ     | Ghi chú                                                |
| ----------------------------------- | ---------- | ------------------------------------------------------ |
| Nhầm lẫn giữa hai model checkpoint  | Cao        | Phải luôn phân biệt rõ Exercise 1 và Exercise 2        |
| Hiểu sai mục tiêu Transfer Learning | Trung bình | Cần giữ đúng tinh thần “bắt đầu từ generic pretrained” |
| Naming file không đúng convention   | Thấp       | Đã thống nhất format có ngày tháng năm                 |

---

## 10. Next Step

Sau khi Phase 0 được duyệt → chuyển sang viết plan Phase 1:

```text
docs/plan-doc/plan_before_process/phase_01_environment_plan_2026-08-10.md
```
