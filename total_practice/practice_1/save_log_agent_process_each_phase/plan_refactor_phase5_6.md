# Review and Implementation Plan - Phase 6 Model Building & Phase 7 Model Training

**Ngay review:** 2026-07-25  
**Notebook:** `total_practice/practice_1/practice_1.ipynb`  
**Pham vi thuc te:** Phase 6 - Model Building va Phase 7 - Model Training  
**Trang thai:** Implemented and verified on 2026-07-25  

> Ten file `plan_refactor_phase5_6.md` duoc giu nguyen theo yeu cau. Trong notebook
> hien tai, noi dung can review nam o Phase 6 va Phase 7, khong phai Phase 5 va 6.

## 1. Nguyen tac va pham vi

Lan review nay:

1. Doc day du cell 37-49 cua notebook.
2. Trace input contract tu Phase 5 va output contract sang Phase 8-9.
3. Doi chieu voi:
   - `processing_own_phase/model.py`
   - `processing_own_phase/train.py`
   - `processing_own_phase/evaluate.py`
   - `processing_own_phase/config.py`
   - `processing_own_phase/experiment.py`
   - `README.md`
   - cac phase log cu.
4. Chay static check va smoke test trong process rieng.
5. Chi ghi findings va implementation plan vao file nay.

Ngoai pham vi:

- Khong sua code/cell cua `practice_1.ipynb`.
- Khong train lai full model.
- Khong sua model checkpoint, TensorBoard event hay output cu.
- Khong tu y khoi phuc validation split da bi loai bo.
- Khong danh gia lai official test set.

## 2. Source of truth hien tai

Theo quyet dinh moi nhat trong notebook:

- Official training set: 60,000 samples.
- Official test set: 10,000 samples.
- Toan bo 60,000 samples duoc dung cho EDA, preprocessing va training.
- Official test set chi duoc dung tai Phase 8.
- Hien tai khong co validation set.

Review nay lay protocol tren lam source of truth. README, source modules va log cu van
mo ta train/validation split; do la tai lieu stale va khong duoc xem la bang chung
cho notebook hien tai.

## 3. Hien trang Phase 6 - Model Building

### 3.1 Architecture hien tai

Default model:

```text
Input [B, 1, 28, 28]
  -> Flatten
  -> Linear(784, 128)
  -> ReLU
  -> Linear(128, 10)
  -> Raw logits [B, 10]
```

Default configuration:

| Thuoc tinh | Gia tri |
|---|---:|
| Class | `FashionMNISTModel` |
| Input dimension | 784 |
| Hidden dimensions | `[128]` |
| Dropout | `0.0` |
| Number of classes | 10 |
| Trainable parameters | 101,770 |
| Final activation | Khong co Softmax |

Khong them Softmax trong model la dung. `nn.CrossEntropyLoss` nhan raw logits va
thuc hien phan tinh toan can thiet ben trong.

### 3.2 Phan da dung

- Dung `nn.Module` va khai bao `forward`.
- Dung `nn.Sequential` de mo ta MLP ro rang.
- Flatten tu image tensor thanh 784 features.
- Cho phep thay doi `hidden_dims` va them optional dropout.
- Output mac dinh co shape `[batch_size, 10]`.
- Default model co dung 101,770 trainable parameters.
- Forward, loss, backward va optimizer step deu chay duoc tren synthetic batch.
- Forward, loss, backward va optimizer step deu chay duoc tren mot FashionMNIST
  batch that.
- Loss ban dau finite, gradients duoc tao va parameters thay doi sau optimizer step.

### 3.3 Runtime evidence

```text
Default output shape:       (8, 10)
Default parameter count:    101,770
Finite loss:                True
Gradients present:          True
Parameters changed:         True

Real batch shape:           (64, 1, 28, 28)
Real labels shape:          (64,)
Real logits shape:          (64, 10)
Real initial loss:          2.343283
Real optimizer update:      True
```

## 4. Hien trang Phase 7 - Model Training

Current training configuration:

| Thuoc tinh | Gia tri |
|---|---:|
| Dataset | 60,000 augmented training samples |
| Batch size | 64 |
| Number of batches | 938 |
| Epochs | 10 |
| Loss | `nn.CrossEntropyLoss()` |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Validation | Khong co |
| TensorBoard run | `runs/fashion_mnist_experiment_1` |

Core loop dang dung thu tu:

```text
model.train()
  -> move batch to device
  -> optimizer.zero_grad()
  -> forward
  -> compute loss
  -> backward
  -> optimizer.step()
  -> accumulate train metrics
```

Official `test_loader` khong bi truy cap trong Phase 7. Day la diem dung va can
duoc giu nguyen.

Stored training output:

| Epoch | Train loss | Train accuracy |
|---:|---:|---:|
| 1 | 0.5879 | 78.51% |
| 5 | 0.3835 | 85.86% |
| 10 | 0.3435 | 87.27% |

Loss giam va accuracy tang deu trong output hien tai. Khong co dau hieu crash,
NaN/Inf hoac divergence trong lan run da luu.

## 5. Ket luan review

### 5.1 Executive conclusion

Phase 6 va Phase 7 **co mot baseline chay duoc**, bao phu dung cac buoc model
building, autograd va optimization co ban cua de bai. Tuy nhien, hai phase nay
**chua nen duoc xem la hoan chinh hoac san sang cho controlled experiments**.

Danh gia:

| Phase | Ket qua | Ly do |
|---|---|---|
| Phase 6 - Model Building | Pass co dieu kien | Default model dung, nhung API checks mau thuan va sanity check chua fail-fast |
| Phase 7 - Model Training | Pass co dieu kien | Core loop dung, nhung metric aggregation, reproducibility va rerun state chua an toan |
| Hyperparameter experiments | Chua pass | Notebook moi train mot configuration va khong co protocol chon model |
| Test isolation trong Phase 7 | Pass | `test_loader` khong duoc su dung |

Khong phat hien bug lam model mac dinh khong the train. Cac van de chinh nam o
correctness cua API, do tin cay cua metric, reproducibility, hidden notebook state
va su mau thuan giua notebook voi tai lieu/module cu.

## 6. Findings chi tiet

### F01 - High - Chua co protocol hop le de experiment va chon hyperparameters

**Vi tri:** Phase 7, README, `config.py`, `train.py`, `experiment.py`.

Notebook hien tai khong co validation set, nhung de bai yeu cau experiment voi
network/hyperparameters. README va source modules lai tuyen bo chon best model theo
validation accuracy.

Neu khong co validation:

- Train metrics khong du de uoc luong generalization.
- Khong the ket luan configuration nao "best" mot cach dang tin cay.
- Neu test tung configuration roi chon model theo test accuracy, official test set
  se bi dung de tuning va final metric khong con la unbiased estimate.
- Ket qua E0-E5 va "Best Validation Accuracy" trong README khong dai dien cho
  notebook hien tai.

**Can owner quyet dinh truoc implementation:**

- **Option A - Giu protocol hien tai:** train mot configuration duoc pre-commit;
  test dung mot lan. Cac experiment chi duoc mo ta la training-behavior experiments,
  khong duoc claim best generalization.
- **Option B - Recommended cho controlled experiments:** tach mot validation subset
  co dinh tu official 60,000 training samples; tune tren validation; giu 10,000
  official test samples untouched; co the retrain selected configuration tren toan
  bo 60,000 samples truoc final test.

Plan nay khong tu dong chon Option B vi no thay doi data protocol ma owner vua chot.

### F02 - High - Rerun Phase 7 tiep tuc train hidden state cu

**Vi tri:** cells 39, 43-47.

Model, optimizer va history duoc tao o cac cell tach roi. Neu chi rerun cell training:

- model tiep tuc tu weights sau epoch 10;
- optimizer tiep tuc voi Adam moments cu;
- epoch label lai bat dau tu 1;
- history co the append them vao list cu;
- TensorBoard ghi them vao cung run directory.

Ket qua moi khi do khong con la mot fresh 10-epoch experiment, du giao dien notebook
van hien thi nhu vay.

**Improvement bat buoc:** mot training run phai tao fresh model, criterion,
optimizer, history va writer tu mot configuration ro rang. Rerun cung config tren
fresh state phai co semantics nhat quan.

### F03 - High - Notebook, README va source modules dang mo ta ba pipeline khac nhau

**Vi tri:** Phase 6-7 va cac file doi chieu.

| Noi dung | Notebook hien tai | README/source modules |
|---|---|---|
| Data range | Normalize ve gan `[-1, 1]` | README noi baseline `[0, 1]` |
| Split | 60,000 train, no validation | 54,001/5,999 train/validation |
| Baseline optimizer | Adam, lr=0.001 | SGD, lr=0.01, momentum=0.9 |
| Default architecture | `[128]`, no dropout | Best model `[256,128]`, dropout=0.2 |
| Experiments | Chua co trong notebook | Tuyen bo E0-E5 da hoan thanh |
| Model class | `FashionMNISTModel` | `FashionMLP` |

**Tac dong:** nguoi doc khong xac dinh duoc implementation nao la source of truth;
report co the cong bo metric khong tai lap duoc tu notebook.

**Improvement bat buoc:** sau khi owner chot protocol, dong bo notebook, README,
config, source modules va phase logs. Khong duoc giu cac claim validation/best model
neu notebook khong con tao validation set.

### F04 - Medium - Sanity check nuot exception va van cho notebook chay tiep

**Vi tri:** cell 40.

Cell dung `try/except Exception` roi chi print `Failed`. Neu model bi sai, notebook
van co the tiep tuc sang Phase 7 va tao loi kho truy vet hon.

Sanity check hien tai chi kiem tra forward output. No chua kiem tra:

- output shape bang chinh xac `[B, num_classes]`;
- logits va loss finite;
- label/loss compatibility;
- gradient co duoc populate va finite;
- optimizer step co thay doi parameters;
- input that tu `train_loader`.

**Improvement:** dung assertions/fail-fast va sanity model rieng, de sanity optimizer
step khong lam thay doi model se duoc train.

### F05 - Medium - `num_classes` configurable nhung output check hard-code 10

**Vi tri:** `FashionMNISTModel.forward`.

Constructor nhan `num_classes`, luu `self.num_classes`, va tao final layer theo gia
tri nay. Tuy nhien `forward` van kiem tra:

```python
if out.shape[1] != 10:
```

Runtime test voi `FashionMNISTModel(num_classes=5)` tao output hop le `[B,5]` nhung
lai raise `ValueError`.

**Improvement:** khong hard-code output contract trong `forward`; neu can validate,
dung `self.num_classes`. Constructor can validate `num_classes > 1`.

### F06 - Medium - Dtype validation noi `float64` hop le nhung model khong chay duoc

**Vi tri:** `FashionMNISTModel.forward`.

Current check chap nhan `torch.float32` va `torch.float64`. Model parameters mac
dinh la `float32`, do do float64 input gay:

```text
RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float
```

Check nay cung tu choi float16/bfloat16 ngay ca khi model da duoc chuyen sang dtype
tuong ung.

**Improvement:** dat dtype/shape validation o boundary sanity check, dua tren dtype
va device cua model parameters. `forward` chi nen thuc hien model computation va
khong tuyen bo support mot dtype ma model khong support.

### F07 - Medium - Epoch loss dang average theo batch, khong theo sample

**Vi tri:** cell 47.

`CrossEntropyLoss` mac dinh tra mean loss cua tung batch. Notebook cong
`loss.item()` roi chia cho `len(train_loader)`. Cach nay cho moi batch cung trong
so, trong khi batch cuoi chi co 32 samples va cac batch khac co 64 samples.

Voi 60,000 samples va batch size 64:

```text
937 full batches x 64 samples + 1 final batch x 32 samples = 938 batches
```

Sai lech nho trong case nay nhung metric definition van chua chinh xac.

**Improvement:**

```python
running_loss += loss.item() * batch_size
epoch_loss = running_loss / total_samples
```

Pattern nay da ton tai trong `processing_own_phase/train.py` va co the tai su dung.

### F08 - Medium - Dung `outputs.data` de tinh prediction

**Vi tri:** cell 47; pattern lap lai o Phase 8.

`.data` bypass mot phan autograd safety va khong can thiet. Training metric khong
can gradient.

**Improvement:**

```python
predictions = logits.argmax(dim=1)
```

Neu luu tensor ngoai training step thi dung `detach()` hoac mot no-grad/inference
context phu hop.

### F09 - Medium - Device policy bo qua MPS

**Vi tri:** cell 43.

Current selection:

```python
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Neu CUDA khong co nhung MPS co, notebook van chay CPU. Environment hien tai cho
ket qua `mps_built=True`, `mps_available=False`, nen chua gay sai runtime hom nay;
tuy nhien code khong dung duoc accelerator khi MPS available tren Apple Silicon.

**Improvement:** device order `CUDA -> MPS -> CPU`, in ro device da chon va giu
device selection o mot noi duy nhat.

PyTorch MPS documentation:
https://docs.pytorch.org/docs/stable/notes/mps.html

### F10 - Medium - Seed dat qua xa model construction

**Vi tri:** cell 7 so voi cells 36-39 va random transforms.

Global seed duoc dat truoc EDA, sample loading va augmented batch sanity check.
Nhung thao tac random o giua co the tieu thu RNG state truoc khi initialize weights.
Rerun cell theo thu tu khac co the tao initial weights khac.

**Improvement:**

- reset seed ngay truoc moi fresh training run;
- seed Python, NumPy va PyTorch tai mot helper;
- gan generator rieng cho DataLoader;
- neu tang `num_workers`, them worker seeding;
- khong promise bitwise-identical result qua khac device/PyTorch version.

PyTorch reproducibility notes:
https://docs.pytorch.org/docs/stable/notes/randomness.html

### F11 - Medium - TensorBoard run lifecycle co the tron du lieu cu

**Vi tri:** cells 45, 47-48.

Run directory bi hard-code la `runs/fashion_mnist_experiment_1`. Rerun co the ghi
them event vao cung experiment. Neu training raise exception, `writer.close()` co
the khong duoc goi.

**Improvement:**

- run name phai duoc tao tu experiment id/config va khong ambiguous;
- dat log path theo project root thay vi phu thuoc current working directory;
- close writer bang `try/finally` hoac context manager;
- log epoch theo 1-based step, learning rate va elapsed time;
- TensorBoard log khong duoc dung lam source duy nhat cua history.

### F12 - Medium - Training loop thieu fail-fast numerical checks

**Vi tri:** cell 47.

Loop khong kiem tra loss/logits/gradients co NaN/Inf. Output hien tai finite, nhung
learning-rate experiment hoac architecture moi co the divergence va van chay den
cuoi.

**Improvement:** it nhat assert finite loss moi batch va report epoch/batch khi
fail. Gradient clipping khong can them cho baseline nay neu chua co bang chung
exploding gradients.

### F13 - Medium - Hyperparameter experiment task chua duoc implement trong notebook

**Vi tri:** Phase 6-7.

Model class co kha nang thay `hidden_dims` va `dropout`, nhung notebook chi tao:

```text
hidden_dims=[128], dropout=0.0, Adam, lr=0.001, epochs=10
```

Khong co controlled comparison, experiment table, fresh-run guarantee hay selection
rule. Vi vay muc "Experiment with network/hyperparameters" cua de bai chua dat
trong notebook hien tai.

### F14 - Low - Training configuration va model evidence chua duoc trinh bay du

**Vi tri:** Phase 6-7 markdown.

Con thieu:

- architecture table va parameter count;
- ly do chon baseline;
- input/output/logit contract;
- ly do khong them Softmax;
- optimizer va learning-rate rationale;
- metric definitions;
- ghi chu training accuracy duoc do tren augmented samples;
- hardware/device va training time.

### F15 - Low - Epoch axis va TensorBoard step khong dong bo voi printed epoch

**Vi tri:** cells 47 va 49.

Console hien epoch 1-10 nhung TensorBoard/plot dung index 0-9. Khong lam sai model,
nhung gay lech khi doi chieu log.

**Improvement:** luu ro `epochs = range(1, num_epochs + 1)` va dung cung index cho
history, plot va TensorBoard.

### F16 - Low - Relative artifact paths phu thuoc thu muc mo notebook

**Vi tri:** `SummaryWriter("runs/...")`; Phase 9 cung co van de tuong tu.

Notebook da tao event tai `total_practice/practice_1/runs/...`, nhung neu kernel
duoc mo tu workspace root thi path co the chuyen sang noi khac.

**Improvement:** xac dinh `PROJECT_DIR` mot lan va derive `RUNS_DIR`,
`OUTPUT_DIR`, checkpoint paths tu do.

### F17 - Low - Baseline hien tai da kem random augmentation

**Vi tri dependency:** Phase 5 transforms; anh huong Phase 7.

Random flip va rotation khong sai, nhung dua augmentation vao baseline ngay tu dau:

- training metrics duoc do tren stochastic augmented images;
- kho tach improvement do model/hyperparameter hay do augmentation;
- rerun co them variance.

**Improvement de xuat:** giu deterministic preprocessing cho baseline, sau do xem
augmentation la mot controlled experiment rieng. Neu owner muon giu augmentation,
markdown phai mo ta no la mot thanh phan co dinh cua moi experiment.

## 7. Implementation plan de xuat

Implementation chi bat dau sau khi owner approve plan va chot Decision D1.

### Stage 0 - Chot hai architecture decisions

#### D1 - Evaluation protocol

Owner chon mot trong hai option tai F01.

Recommended neu can hoan thanh controlled hyperparameter experiments:

```text
Official train pool 60,000
  -> train subset
  -> validation subset

Official test 10,000
  -> untouched until one final evaluation
```

Neu owner giu no-validation protocol, implementation se khong them validation va
se xoa/doi wording "best model", "best validation accuracy" khoi artifacts lien
quan.

#### D2 - Code ownership

Recommended:

- Notebook van hien thi day du code cot loi de dap ung deliverable hoc tap.
- `processing_own_phase/model.py` va `train.py` la reusable counterpart.
- Hai noi dung phai dung cung API/metric definitions.
- Khong import blind source modules hien tai vi chung van phu thuoc validation.

### Stage 1 - Chot contracts va baseline config

Them mot config duy nhat cho Phase 6-7:

```text
seed
input_dim
hidden_dims
dropout
num_classes
batch_size
epochs
optimizer_name
learning_rate
weight_decay
run_id
```

Chot contracts:

- Images: `float32`, `[B,1,28,28]`.
- Labels: `int64`, `[B]`, values 0-9.
- Logits: `float32`, `[B,10]`.
- Loss: sample-weighted epoch mean.
- Accuracy: fraction `[0,1]` trong data structure; format `%` chi khi display.

### Stage 2 - Refactor model definition

Refactor `FashionMNISTModel` theo huong:

1. Validate constructor arguments:
   - `input_dim > 0`
   - `num_classes > 1`
   - moi hidden dimension > 0
   - `0 <= dropout < 1`
2. Luu architecture metadata tren model.
3. Giu `Flatten -> Linear/ReLU/(Dropout) -> Linear`.
4. Giu raw logits, khong them Softmax.
5. Loai hard-coded `10` khoi output check.
6. Loai dtype claim sai khoi `forward`.
7. Them type hints va docstring input/output.
8. Giu `forward` nho, deterministic va tap trung vao computation.

### Stage 3 - Refactor device va reproducibility setup

1. Chon device theo `CUDA -> MPS -> CPU`.
2. In selected device mot lan.
3. Tao `set_seed(seed)` cho Python, NumPy va PyTorch.
4. Goi `set_seed` ngay truoc model construction cua moi run.
5. Khoi tao fresh DataLoader generator cho moi controlled run neu can fair ordering.
6. Ghi chu gioi han reproducibility cross-device/cross-version.

### Stage 4 - Nang cap model sanity check

Dung separate sanity model hoac snapshot/restore state:

1. Lay mot batch that tu `train_loader`.
2. Assert input shape/dtype va label contract.
3. Forward va assert logits shape.
4. Assert logits finite.
5. Tinh `CrossEntropyLoss` va assert finite.
6. Backward va assert trainable parameters co finite gradients.
7. Optimizer step va assert it nhat mot parameter thay doi.
8. Raise loi that khi fail; khong `except Exception` chi de print.
9. In architecture, parameter count va ket qua check.
10. Khong lam thay doi training model chinh.

### Stage 5 - Tach `train_one_epoch`

Tao function voi contract ro:

```text
inputs:
  model, train_loader, criterion, optimizer, device

returns:
  loss, accuracy, sample_count, elapsed_seconds
```

Implementation:

1. `model.train()`.
2. Move data/labels sang device.
3. `optimizer.zero_grad(set_to_none=True)`.
4. Forward va finite-loss guard.
5. Backward va optimizer step.
6. Dung `logits.argmax(dim=1)`, khong `.data`.
7. Aggregate `loss * batch_size`.
8. Chia loss/accuracy theo tong sample.
9. Khong tham chieu `test_loader`.

### Stage 6 - Tao fresh-run training orchestrator

Tao `run_training(config, train_loader, device)` hoac API tuong duong:

1. Reset seed.
2. Build fresh model.
3. Build fresh criterion.
4. Build fresh optimizer.
5. Build empty history.
6. Tao writer cho run id rieng.
7. Train dung so epoch.
8. Luu epoch 1-based, loss, accuracy, lr va elapsed time.
9. Dam bao writer duoc close neu co exception.
10. Return model, optimizer-independent history va run metadata.

Neu D1 chon validation protocol, orchestrator se them deterministic evaluation moi
epoch va model-selection rule. Neu D1 giu no-validation protocol, orchestrator
khong duoc cham `test_loader` va khong claim "best epoch".

### Stage 7 - Refactor visualization va TensorBoard

1. Plot x-axis 1 den `num_epochs`.
2. Plot sample-weighted train loss.
3. Plot train accuracy voi unit nhat quan.
4. Khong bat buoc log scale; chi dung neu markdown giai thich.
5. TensorBoard tags va static plots dung cung history.
6. Run directory derive tu project root va unique run id.
7. Khong ghi de/tron event cua run khac.

### Stage 8 - Them controlled experiments theo D1

Experiment matrix toi thieu nen thay mot bien moi lan:

| Experiment | Architecture | Dropout | Optimizer | LR | Muc dich |
|---|---|---:|---|---:|---|
| E0 | `[128]` | 0.0 | Adam | 0.001 | Baseline hien tai |
| E1 | `[64]` | 0.0 | Adam | 0.001 | Capacity |
| E2 | `[256,128]` | 0.0 | Adam | 0.001 | Depth/width |
| E3 | `[128]` | 0.2 | Adam | 0.001 | Dropout |
| E4 | `[128]` | 0.0 | SGD | 0.01 | Optimizer |

Bang tren chi la proposal. Owner co the phe duyet so experiment nho hon de tiet
kiem compute.

Fair-comparison requirements:

- fresh model/optimizer cho moi experiment;
- cung data protocol;
- cung epoch budget;
- reset seed va loader order;
- test set khong duoc dung de rank;
- report parameter count va runtime;
- khong claim statistical superiority tu mot seed duy nhat.

### Stage 9 - Dong bo downstream contracts

Sau khi Phase 6-7 pass:

1. Phase 8 nhan dung returned model/history.
2. Phase 8 thay `.data` bang `argmax`.
3. Test loss cung aggregate theo sample.
4. Phase 9 checkpoint luu architecture config cung `state_dict`.
5. Reload model tu saved config, khong hard-code default architecture.
6. Verify logits/predictions cua model truoc va sau reload giong nhau.

Stage nay la downstream compatibility work; chi implementation neu owner phe duyet
pham vi mo rong.

### Stage 10 - Dong bo documentation va source modules

Cap nhat sau cung, khi code va metrics da duoc verify:

- `README.md`
- `processing_own_phase/model.py`
- `processing_own_phase/train.py`
- `processing_own_phase/config.py`
- `processing_own_phase/experiment.py`
- phase logs lien quan.

Khong giu metric/claim E0-E5 cu neu khong the tai lap bang final notebook.

## 8. Expected cell structure sau refactor

Proposal cho Phase 6:

```text
Markdown: model objective and input/output contract
Code: training/model config
Code: FashionMNISTModel
Code: build fresh baseline model + parameter count
Code: fail-fast sanity check
Markdown: sanity-check interpretation
```

Proposal cho Phase 7:

```text
Markdown: training objective and test-isolation rule
Code: device + run seed
Code: criterion/optimizer builders
Code: train_one_epoch
Code: run_training
Code: execute approved baseline/experiments
Code: TensorBoard/static plots
Markdown: evidence-based observations and limitations
```

## 9. Verification plan sau implementation

### 9.1 Static verification

- Notebook van la valid JSON.
- Tat ca non-magic code cells compile.
- Khong con `outputs.data`.
- Khong con output check hard-code 10.
- Khong co reference toi `test_loader` trong Phase 7.
- Markdown/config/code cung noi mot protocol.

### 9.2 Model unit smoke tests

- Default output shape `[64,10]`.
- Flattened input `[64,784]` neu duoc tuy chon support.
- Parameter count default = 101,770.
- Invalid hidden dimension/dropout/num_classes fail voi message ro.
- Float32 real batch forward thanh cong.
- Logits/loss finite.
- Backward tao finite gradients.
- Optimizer step thay doi parameters.
- Sanity check khong mutate final training model.

### 9.3 Training smoke tests

- Chay 2-3 batches tren subset.
- Tong sample count dung.
- Sample-weighted loss khop independent calculation.
- Accuracy nam trong `[0,1]`.
- History co dung keys va dung epoch count.
- TensorBoard writer dong thanh cong.
- Test loader khong bi iterate.

### 9.4 Full baseline verification

- Restart kernel va Run All den het Phase 7.
- Training 10 epochs khong NaN/Inf.
- Loss/accuracy trend duoc report tu output moi, khong copy metric cu.
- Plot va TensorBoard khop history.
- Rerun tu fresh kernel voi cung config cho ket qua trong tolerance hop ly tren
  cung environment/device.

### 9.5 Protocol verification

Neu no-validation:

- Khong co best-model claim.
- Test chi chay dung final evaluation.
- Experiment conclusions bi gioi han o optimization/training behavior.

Neu co validation:

- Train/validation indices disjoint.
- Validation transform deterministic.
- Model selection chi dung validation.
- Official test set untouched cho den final evaluation.

## 10. Acceptance criteria

Refactor chi duoc xem la pass khi:

1. Phase 6 fail-fast neu model contract sai.
2. Default model co 101,770 parameters va output `[B,10]`.
3. Model API khong con mau thuan `num_classes`/dtype.
4. Phase 7 moi lan run tao fresh state.
5. Epoch loss la sample-weighted mean.
6. Khong dung `.data`.
7. Device policy support CUDA, MPS va CPU.
8. Seed duoc reset o dung run boundary.
9. TensorBoard runs khong bi tron.
10. Official test loader khong duoc truy cap trong training.
11. Hyperparameter claims phu hop protocol duoc owner chon.
12. Notebook, README va reusable modules khong con noi nhung pipeline mau thuan.
13. Clean-kernel execution va smoke/full checks pass.

## 11. Rủi ro khi implementation

| Rui ro | Tac dong | Mitigation |
|---|---|---|
| Doi evaluation protocol | Metric cu khong con comparable | Chot D1 truoc khi code |
| Fresh-run semantics | Output moi khac output stored | Xem output moi la source of truth |
| MPS/CPU khac nhau | Metric/time khac nhe | Ghi device/version va dung tolerance |
| Dong bo nhieu source files | De tiep tuc drift | Chot D2 va mot contract duy nhat |
| Full experiment matrix ton compute | Review cham | Smoke test truoc, gioi han experiments |
| Checkpoint cu khac architecture | Load failure | Luu model config kem state dict |

## 12. Tai lieu ky thuat doi chieu

- PyTorch MPS backend:
  https://docs.pytorch.org/docs/stable/notes/mps.html
- PyTorch reproducibility:
  https://docs.pytorch.org/docs/stable/notes/randomness.html
- `CrossEntropyLoss` reduction semantics:
  https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- Autograd and inference modes:
  https://docs.pytorch.org/docs/stable/notes/autograd

## 13. Approval checklist cho owner

Truoc khi cho phep implementation, owner verify:

- [x] Dong y findings va severity.
- [x] Chon D1: class-stratified train/validation protocol.
- [x] Chon D2: dong bo notebook, reusable modules va README.
- [x] Phe duyet baseline config.
- [x] Phe duyet five controlled experiments.
- [x] Mo rong compatibility changes sang Phase 8-9.
- [x] Hoan thanh implementation va clean-process verification.

## 14. Implementation resolution

| Finding | Resolution | Verification |
|---|---|---|
| F01 | 54,000/6,000 class-stratified train/validation split; official test isolated | Split, overlap and class-count assertions pass |
| F02 | Moi run tao fresh model, optimizer, loader generator, history va writer | Repeated-run metrics va weights match |
| F03 | Notebook, package va README dung cung protocol/configs | Experiment-config parity check pass |
| F04 | Sanity check fail-fast, khong nuot exception | Forward/loss/gradient/update checks pass |
| F05 | Output contract dung `num_classes`, khong hard-code 10 | Custom five-class model returns `[4,5]` |
| F06 | Bo dtype claim sai khoi `forward`; boundary check theo parameter dtype | Float32 real-batch contract pass |
| F07 | Epoch loss weighted theo batch sample count | Match independent summed-loss calculation |
| F08 | Thay `.data` bang `argmax(dim=1)` | Static scan pass |
| F09 | Device priority CUDA, MPS, CPU | CPU full run pass; MPS branch present |
| F10 | Reset Python, NumPy va PyTorch seed tai run boundary | Clean runs reproduce exact metrics |
| F11 | Unique run session va writer `finally` lifecycle | TensorBoard event file cho moi run |
| F12 | Fail-fast non-finite loss checks trong train/evaluation | Full run khong NaN/Inf |
| F13 | Five controlled experiments voi validation selection | E0-E4 full results recorded |
| F14 | Markdown va README mo ta contracts/configuration/metrics | Documentation review pass |
| F15 | Epoch history, plots va TensorBoard dung 1-based index | History/plot smoke tests pass |
| F16 | Data, output va run paths derive tu project root | Root and notebook-directory execution pass |
| F17 | Baseline deterministic; augmentation la experiment rieng | E4 isolated augmentation config pass |

## 15. Final verified result

```text
Selected experiment: E1_deeper
Architecture: [256, 128]
Dropout: 0.0
Optimizer: Adam
Learning rate: 0.001
Best epoch: 10
Best validation accuracy: 89.30%
Final test loss: 0.3320
Final test accuracy: 88.84%
Checkpoint predictions match: True
Maximum logit difference: 0.00000000
```

**Implementation status: COMPLETED AND VERIFIED.**
