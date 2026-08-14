# QUY TAC QUY TRINH VA KIEN TRUC COURSE_WORK

## 1. Muc dich

File nay la tai lieu quy dinh cach su dung kien truc hien tai cua `COURSE_WORK`.

Moi nguoi dung va agent lam viec trong project phai dua vao tai lieu nay de xac dinh:

- moi thu muc dung de luu gi;
- moi file Python chiu trach nhiem cho logic nao;
- tung Phase 0-59 phai trien khai tai dau;
- dependency nao duoc phep va dependency nao bi cam;
- notebook duoc phep thuc thi nhu the nao;
- du lieu, model, checkpoint, metric, hinh anh va bao cao phai duoc quan ly ra sao;
- cach kiem tra mot thay doi truoc khi chuyen sang Phase tiep theo.

Tai lieu nay mo ta kien truc dang ton tai. No khong tu dong cho phep tao them package, module, Phase hoac artifact path chua co trong project.

Tai thoi diem audit de viet rule nay:

- 33 file Python trong `src/course_work` deu la scaffold rong;
- `course_work.ipynb` la file rong;
- `README.md` la file rong;
- `requirements.txt` la file rong;
- chua co implementation nao duoc xem la hoan tat chi dua tren su ton tai cua ten file.

Moi mo ta trach nhiem trong tai lieu nay la ownership contract cho implementation tuong lai, khong phai bang chung rang chuc nang da ton tai hoac da duoc kiem thu.

## 2. Pham vi ap dung

Quy tac ap dung cho toan bo:

```text
COURSE_WORK/
```

Quy tac ap dung cho:

- code trong `src/course_work`;
- notebook trong `notebook_course_work`;
- du lieu trong `data`;
- kiem thu trong `tests`;
- tai lieu trong `docs`;
- hinh anh trong `image_diagram`;
- dependencies trong `requirements.txt`;
- cac Phase 0-59;
- cac workflow pretraining, LoRA va fine-tuning duoc Human phe duyet sau nay.

## 3. Thu tu uu tien cua tai lieu

Khi thuc thi cong viec, phai doc va ap dung theo thu tu:

1. `COURSE_WORK/working_rule.md`.
2. `COURSE_WORK/docs/RULE_BASE/rule_code.md`.
3. `COURSE_WORK/docs/RULE_BASE/rule_process_with_architecture.md`.
4. Phase detail tuong ung trong `docs/plan-doc/plan_detail_for_each_phase`.
5. Pre-process plan da duoc Human phe duyet.
6. Code, tests, artifacts va notebook lien quan.

Neu co mau thuan chua the giai quyet bang thu tu tren, phai dung thuc thi, ghi lai bang chung va xin Human quyet dinh.

## 4. Kien truc canonical hien tai

```text
COURSE_WORK/
├── README.md
├── requirements.txt
├── working_rule.md
├── data/
│   ├── raw_data/
│   │   └── energydata_complete.csv
│   └── data_after_split/
├── docs/
│   ├── RULE_BASE/
│   │   ├── rule_code.md
│   │   └── rule_process_with_architecture.md
│   ├── current_flow/
│   ├── plan-doc/
│   │   ├── analysis_error/
│   │   ├── plan_before_process/
│   │   ├── plan_detail_for_each_phase/
│   │   ├── plan_overview/
│   │   └── plan_to_refactor&fix/
│   ├── result/
│   ├── save_process_proceduce_own_phase_refactor&fix/
│   └── code_base_audit.md
├── image_diagram/
├── notebook_course_work/
│   └── course_work.ipynb
├── src/
│   └── course_work/
│       ├── attention/
│       ├── data/
│       ├── evaluation/
│       ├── experiments/
│       ├── models/
│       │   └── components/
│       └── training/
└── tests/
```

Ten package canonical la:

```text
course_work
```

Duong dan source canonical la:

```text
COURSE_WORK/src/course_work/
```

Khong duoc tao package song song nhu:

```text
COURSE_WORK/SRC/
COURSE_WORK/src/coursework/
COURSE_WORK/course_work/
COURSE_WORK/source/
```

## 5. Nguyen tac kien truc tong quat

1. `src/course_work` la noi duy nhat chua logic xu ly co the tai su dung.
2. `course_work.ipynb` chi dieu phoi va goi logic tu `src/course_work`.
3. Notebook khong duoc la source of truth cho bat ky thuat toan, bien doi du lieu, model, training loop, metric hoac attention analysis nao.
4. Moi module chi so huu mot nhom trach nhiem ro rang.
5. Khong duplicate cung mot logic o hai file.
6. Khong ghi du lieu sinh ra vao `raw_data`.
7. Khong su dung Test de chon feature, hyperparameter, checkpoint, adaptation method hoac architecture.
8. Moi Phase chi duoc doc artifact upstream da duoc verify.
9. Moi output phai truy vet duoc Phase, config, seed, input va code tao ra no.
10. Khong dua logic nghiep vu vao ten file tam, notebook cell hoac script ngoai kien truc.
11. Khong tao file rong chi de lam cay thu muc co ve day du.
12. Khong them dependency, module, package hoac output root ma khong co plan duoc phe duyet.

## 6. Trach nhiem cua cac file va thu muc cap cao

### 6.1. `README.md`

Luu huong dan khoi dong project o muc tong quan:

- muc tieu coursework;
- dataset;
- bai toan regression;
- cach cai dat moi truong;
- cach dat `src` tren Python path hoac cai package;
- cach chay notebook;
- cach chay tests;
- thu tu Phase;
- vi tri output da duoc phe duyet.

`README.md` khong thay the Phase plan va khong chua implementation logic.

### 6.2. `requirements.txt`

Luu dependency runtime va development da duoc phe duyet.

Moi dependency phai:

- can thiet cho mot module hoac test cu the;
- co version constraint phu hop;
- duoc kiem tra conflict;
- khong trung vai tro voi dependency hien co ma khong co ly do.

Khong them thu vien chi de thay the vai dong code don gian da co trong standard library.

### 6.3. `working_rule.md`

Luu quy tac hop tac, phan tich, phe duyet, thuc thi va bao cao chung.

Khong luu architecture implementation trong file nay neu noi dung da thuoc tai lieu hien tai.

### 6.4. `.DS_Store`

`.DS_Store` la metadata cua he dieu hanh, khong phai thanh phan kien truc va khong duoc dung lam input, output hoac bang chung thuc thi.

## 7. Kien truc du lieu

### 7.1. `data/raw_data`

Chi luu du lieu nguon bat bien.

File hien tai:

```text
data/raw_data/energydata_complete.csv
```

Quy tac:

- khong sua noi dung raw data;
- khong overwrite;
- khong them cot;
- khong scale;
- khong split truc tiep vao cung file;
- phai kiem tra checksum, schema, timestamp va so dong truoc xu ly;
- moi du lieu dan xuat phai duoc ghi sang location downstream.

### 7.2. `data/data_after_split`

Luu cac tap du lieu sau chronological split da duoc Phase 8 phe duyet.

Moi output split phai xac dinh:

- Train;
- Validation;
- Test;
- moc thoi gian dau va cuoi;
- so dong;
- quy tac boundary;
- checksum hoac fingerprint;
- version cua split contract.

Khong random split. Khong refit scaler bang Validation hoac Test. Khong thay doi split sau khi da xem ket qua Test.

### 7.3. Luong du lieu bat buoc

```text
raw_data
-> acquisition validation
-> schema audit
-> temporal integrity audit
-> feature engineering
-> chronological split
-> Train-only scaling
-> sequence windows
-> dataset and DataLoader
-> model
-> prediction bundle
-> evaluation
-> analysis and reporting
```

Khong duoc dao nguoc thu tu scaling va split. Scaler chi duoc fit tren Train.

## 8. Package `src/course_work/data`

### 8.1. `acquisition.py`

So huu logic Phase 2:

- xac minh file dataset;
- kiem tra nguon va metadata;
- tinh checksum;
- kiem tra kha nang doc;
- tao acquisition manifest khi output path duoc phe duyet.

Khong so huu feature engineering, split hoac scaling.

### 8.2. `schema.py`

So huu logic Phase 3:

- ten cot;
- dtype;
- missing values;
- duplicate rows;
- target column;
- timestamp column;
- schema contract;
- validation error cho schema khong hop le.

Khong tu sua du lieu bi loi neu chua co repair plan.

### 8.3. `temporal.py`

So huu logic Phase 4:

- parse timestamp;
- sort theo thoi gian;
- duplicate timestamp;
- khoang cach sampling;
- gap detection;
- temporal boundary validation;
- leakage checks lien quan den thoi gian.

Khong so huu model hay metric.

### 8.4. `features.py`

So huu logic Phase 5-7:

- feature engineering;
- cyclic time features;
- feature-set definitions;
- loai `rv1` va `rv2` khoi feature set chinh;
- cac feature ablation da duoc phe duyet;
- bao toan target va timestamp alignment.

Khong fit scaler va khong truy cap Test metric de chon feature.

### 8.5. `splitting.py`

So huu logic Phase 8:

- chronological Train, Validation va Test split;
- boundary policy;
- split manifest;
- overlap checks;
- timestamp continuity checks;
- split fingerprint.

Khong randomize thu tu thoi gian.

### 8.6. `scaling.py`

So huu logic Phase 9 va target-scaling experiment cua Phase 25:

- fit feature scaler tren Train;
- transform Train, Validation va Test bang cung fitted state;
- target scaler neu config yeu cau;
- inverse transform prediction;
- scaler serialization contract;
- scaler provenance.

Khong fit lai tren Validation hoac Test.

### 8.7. `windows.py`

So huu logic Phase 10:

- tao lookback windows;
- xac dinh forecast horizon;
- can chinh `X`, `y` va timestamp;
- kiem tra boundary protocol;
- dam bao khong dua future observation vao input.

Output contract toi thieu:

```text
X: [N, L, F]
y: [N, 1]
timestamp: [N]
```

### 8.8. `datasets.py`

So huu logic Phase 11:

- PyTorch Dataset;
- DataLoader construction;
- batch shape validation;
- shuffle policy;
- deterministic worker seeding;
- target access contract.

Train loader co the shuffle theo config. Validation va Test loader khong duoc shuffle.

## 9. Package `src/course_work/models`

### 9.1. `lstm.py`

So huu LSTM baseline cua Phase 15 va model duoc tune o Phase 43:

- `LSTMRegressor`;
- LSTM configuration contract;
- forward pass;
- last-sequence-output hoac representation da duoc Phase phe duyet;
- linear regression head;
- output shape `[B, 1]`;
- parameter counting.

File nay khong chua training loop, DataLoader hoac metric implementation.

### 9.2. `transformer.py`

So huu Transformer Encoder regression model cua Phase 16:

- input projection;
- positional encoding integration;
- encoder stack;
- pooling strategy;
- regression head;
- `return_attention` contract;
- output shape `[B, 1]`;
- model configuration;
- parameter counting.

Model dung de phan tich attention phai la cung architecture, cung state va cung checkpoint voi model dung de prediction.

### 9.3. `models/components/attention.py`

So huu primitive self-attention can expose attention weights:

- attention tensor contract;
- mask handling;
- head dimension;
- return-weight behavior;
- shape validation.

Khong chua heatmap rendering hoac scientific interpretation.

### 9.4. `models/components/encoder_layer.py`

So huu attention-aware Transformer encoder layer:

- self-attention block;
- residual connection;
- normalization;
- feed-forward network;
- dropout;
- attention return path.

Khong duoc co mot encoder rieng cho training va mot encoder khac cho attention extraction.

### 9.5. `models/components/positional_encoding.py`

So huu positional encoding:

- sinusoidal encoding hoac method da duoc phe duyet;
- maximum sequence length validation;
- device va dtype alignment;
- output shape preservation.

### 9.6. `models/components/revin.py`

So huu RevIN cua Phase 40:

- normalize theo contract;
- denormalize;
- affine parameters neu duoc phe duyet;
- numerical stability;
- shape preservation.

RevIN khong duoc ngam thay the Train-only scaler.

### 9.7. `models/components/lora.py`

So huu LoRA components:

- low-rank adapter layers;
- target-module selection;
- rank, alpha va dropout contract;
- freeze and unfreeze validation;
- trainable-parameter accounting;
- merge policy neu duoc phe duyet.

Khong tu dong gan LoRA vao moi linear layer. Target modules phai duoc config va experiment protocol xac dinh.

LoRA khong nam trong Phase 0-59 hien tai. Khong Phase 0-59 nao duoc tu nhan la Phase LoRA neu chua co Phase extension va pre-process plan duoc phe duyet.

## 10. Package `src/course_work/training`

### 10.1. `engine.py`

So huu training engine cua Phase 19:

- train epoch;
- validation epoch;
- loss aggregation;
- optimizer step;
- gradient clipping;
- history recording;
- early-stopping integration;
- checkpoint integration;
- deterministic execution hooks.

Khong so huu split, scaler, model definition hoac final scientific conclusion.

### 10.2. `checkpointing.py`

So huu:

- checkpoint save;
- atomic write contract;
- checkpoint load;
- model state;
- optimizer state khi can resume;
- scheduler state neu co;
- epoch va best metric;
- config fingerprint;
- scaler references;
- code and data lineage.

Checkpoint khong duoc overwrite neu no da duoc lock cho final evaluation.

### 10.3. `early_stopping.py`

So huu:

- monitored Validation metric;
- patience;
- minimum improvement;
- best-state tracking;
- stop decision.

Early stopping khong duoc doc Test metric.

### 10.4. `reproducibility.py`

So huu:

- Python seed;
- NumPy seed;
- PyTorch seed;
- CUDA seed neu ap dung;
- deterministic flags;
- worker seed;
- runtime environment record.

### 10.5. `pretraining.py`

So huu workflow pretraining da duoc phe duyet:

- pretraining objective;
- pretraining dataset boundary;
- encoder state export;
- checkpoint lineage;
- prevention of Validation and Test leakage;
- compatibility validation voi downstream Transformer.

Pretraining khong duoc truy cap Held-Out Test. File nay khong tu tao Phase moi.

### 10.6. `fine_tuning.py`

So huu:

- linear probing neu duoc phe duyet;
- LoRA adaptation;
- partial fine-tuning;
- freeze and unfreeze policy;
- optimizer parameter groups;
- trainable-parameter report;
- adaptation checkpoint lineage.

Full fine-tuning khong phai workflow mac dinh. Moi so sanh LoRA va partial fine-tuning phai dung cung data boundary, seed policy, evaluation metric va budget contract.

## 11. Package `src/course_work/evaluation`

### 11.1. `metrics.py`

So huu metric dung chung cua Phase 12:

- MAE;
- RMSE;
- R-squared neu duoc Phase contract yeu cau;
- metric input validation;
- inverse-scale evaluation;
- aggregate schema.

Metric phai duoc tinh trong original target unit khi bao cao ket qua chinh.

### 11.2. `inference.py`

So huu:

- deterministic inference;
- prediction bundle generation;
- timestamp and target alignment;
- inverse transform;
- seed-specific prediction export;
- Test-access guard;
- batched inference khong thay doi prediction semantics.

File nay khong chon checkpoint dua tren Test.

### 11.3. `residuals.py`

So huu Phase 49 va phan residual cua Phase 48:

- residual definition;
- residual summary;
- residual distribution;
- residual over time;
- heteroscedasticity diagnostics duoc phe duyet;
- alignment voi frozen prediction bundle.

### 11.4. `regimes.py`

So huu Phase 50-51:

- Train-only regime threshold;
- regime assignment;
- error by regime;
- worst-error case selection;
- cohort metadata;
- threshold provenance.

Khong duoc tao threshold tu Test distribution.

## 12. Package `src/course_work/experiments`

### 12.1. `registry.py`

So huu experiment registry cua Phase 13:

- run ID;
- run family;
- config fingerprint;
- data and split fingerprint;
- seed;
- model identity;
- checkpoint path;
- metric path;
- status;
- failure record;
- parent-child lineage.

Registry la source of truth cho lich su run, khong phai notebook output cell.

### 12.2. `sweeps.py`

So huu dieu phoi cac sweep Phase 23-41:

- one-factor screening;
- feature-set sweep;
- time-feature sweep;
- target-scaling sweep;
- lookback sweep;
- pooling, activation va batch sweep;
- learning-rate, weight-decay va dropout sweep;
- model-size sweep;
- loss, epoch-cap va gradient-clipping sweep;
- RevIN sweep;
- boundary-protocol check.

File nay goi module data, model, training va evaluation. No khong duplicate implementation cua cac module do.

### 12.3. `adaptation.py`

So huu experiment orchestration cho:

- pretrained encoder baseline;
- LoRA learning;
- partial fine-tuning;
- frozen encoder hoac linear probing neu duoc phe duyet;
- fair-comparison budget;
- trainable-parameter comparison;
- runtime and memory comparison;
- adaptation result registration.

File nay khong chua LoRA layer implementation hoac training-loop implementation.

### 12.4. `final_lock.py`

So huu Phase 42 va Phase 45-47 o muc khoa va xac minh:

- candidate specification;
- final configuration fingerprint;
- checkpoint identity;
- seed list;
- scaler checksums;
- data boundary checksums;
- immutable final lock;
- release gate cho three-seed run;
- release gate cho Held-Out Test.

Khong sua final lock sau khi da truy cap Test.

## 13. Package `src/course_work/attention`

### 13.1. `extraction.py`

So huu Phase 52:

- load dung final checkpoint;
- bat `return_attention` tren cung model;
- extract raw attention;
- luu layer and head dimensions;
- position mapping;
- last-query slice source;
- checksum va provenance.

Raw attention khong duoc head-average, layer-average, smooth hoac normalize lai truoc khi dong bang source artifact.

### 13.2. `heatmaps.py`

So huu Phase 53:

- render heatmap tu frozen Phase 52 artifact;
- scale va color policy;
- axis and position labels;
- figure export;
- image checksum.

Khong suy dien ket luan khoa hoc chi tu hinh anh.

### 13.3. `last_query.py`

So huu Phase 54:

- newest historical query selection;
- temporal attention profile;
- lag mapping;
- quantitative summaries;
- entropy hoac concentration metric da duoc Phase phe duyet.

### 13.4. `head_comparison.py`

So huu Phase 55-57:

- per-head comparison;
- pairwise profile comparison;
- error-conditioned attention comparison;
- seed-stability analysis;
- head-order provenance;
- caveat ve head semantic alignment giua seed.

Khong khang dinh attention weight la causal contribution hoac feature importance.

## 14. Thu muc `tests`

`tests` la noi duy nhat chua automated tests cua coursework.

Moi source module phai co test tuong ung truoc khi duoc dung trong official run.

Nhom test toi thieu:

- schema and temporal integrity;
- chronological split and no-overlap;
- Train-only scaling;
- window alignment and no-future leakage;
- Dataset and DataLoader shapes;
- metric reference examples;
- LSTM forward shape;
- Transformer forward and attention shapes;
- RevIN round trip;
- LoRA freeze and trainable-parameter contract;
- checkpoint round trip;
- reproducibility;
- inference alignment;
- registry integrity;
- Test firewall;
- attention extraction consistency.

Quy tac dat ten:

```text
test_<module>_<behavior>.py
```

Test khong duoc ghi de len raw data, frozen checkpoint hoac official artifact.

## 15. Thu muc `docs`

### 15.1. `docs/RULE_BASE`

Luu quy tac bat buoc cua project.

- `rule_code.md`: quy trinh implementation, fix, validation va approval.
- `rule_process_with_architecture.md`: ownership, dependency, phase mapping va notebook boundary.

### 15.2. `docs/current_flow`

Luu trang thai flow hien tai:

- Phase dang chuan bi;
- Phase da duoc phe duyet;
- Phase dang blocked;
- upstream dependencies;
- output da verify;
- buoc tiep theo.

Khong ghi ket qua gia dinh.

### 15.3. `docs/plan-doc/analysis_error`

Luu issue synthesis da xac minh:

- symptom;
- evidence;
- impacted scope;
- root cause hoac hypothesis status;
- upstream and downstream impact;
- unresolved questions.

### 15.4. `docs/plan-doc/plan_before_process`

Luu pre-process plan phai duoc Human duyet truoc implementation.

Plan phai co objective, scope, impacted files, dependency, risks, sequential steps, validation, regression, outputs, stop conditions va approval gate.

### 15.5. `docs/plan-doc/plan_detail_for_each_phase`

Luu contract chi tiet cho Phase 0-59. Moi Phase file la acceptance baseline bat buoc cua Phase do.

Khong sua Phase detail de hop thuc hoa implementation da lam sai.

### 15.6. `docs/plan-doc/plan_overview`

Luu requirement analysis, main plan va overfitting strategy o muc tong quan.

Overview khong override Phase detail cu the.

### 15.7. `docs/plan-doc/plan_to_refactor&fix`

Luu cac plan refactor hoac fix da duoc phan loai theo workflow cua project neu quy trinh hien hanh chi dinh location nay.

Khong thuc thi plan khi chua co Human approval.

### 15.8. `docs/result`

Luu ban tong hop ket qua da duoc tao tu machine-readable artifacts va da verify.

Khong dung `docs/result` lam noi luu checkpoint, scaler, raw prediction hoac tensor attention.

### 15.9. `docs/save_process_proceduce_own_phase_refactor&fix`

Luu execution record cua tung Phase, refactor va fix:

- plan ID;
- cac buoc da chay;
- validation sau moi buoc;
- input and output paths;
- discrepancy;
- final status;
- next gate.

### 15.10. `docs/code_base_audit.md`

Luu audit tong quan cua codebase tai mot moc da xac dinh. Audit phai ghi thoi diem, scope va evidence, khong duoc coi la trang thai real-time vinh vien.

## 16. Thu muc `image_diagram`

Luu hinh anh va diagram dung cho report hoac giai thich architecture da duoc tao tu output hop le.

Moi hinh phai co:

- source artifact;
- Phase tao ra;
- config rendering;
- caption hoac mapping den report;
- khong che lap machine-readable artifact.

Heatmap canonical duoc sinh boi `attention/heatmaps.py`, khong duoc ve thu cong trong notebook.

## 17. Notebook orchestration-only

File canonical:

```text
COURSE_WORK/notebook_course_work/course_work.ipynb
```

Notebook chi duoc phep:

- import public functions va classes tu `course_work`;
- load mot config hoac contract da duoc phe duyet thong qua function trong source;
- goi mot Phase runner hoac mot ham orchestration cap cao;
- nhan ve result reference hoac artifact reference;
- hien thi bang, hinh va thong tin da duoc source module tao san;
- ghi narrative Markdown cua coursework.

Notebook khong duoc chua:

- ham xu ly du lieu;
- class Dataset;
- feature engineering;
- split logic;
- scaling logic;
- window-building logic;
- model class;
- custom layer;
- loss implementation;
- optimizer construction logic;
- training loop;
- evaluation metric formula;
- inference loop;
- residual calculation;
- regime threshold calculation;
- attention extraction;
- heatmap construction;
- hyperparameter sweep loop;
- checkpoint serialization;
- path-discovery logic;
- exception-repair logic;
- duplicate implementation tu `src/course_work`.

Mot code cell hop le nen chi the hien orchestration cap cao, vi du ve mat cau truc:

```python
from course_work.experiments.registry import run_registered_phase

result = run_registered_phase(phase_id=20, config_name="lstm_baseline")
result.display()
```

Ten API tren chi minh hoa boundary, khong tu dong tuyen bo API da ton tai. Khi implementation, phai import dung public API that su da duoc test.

Neu notebook can hon mot chuoi call ngan de thuc thi mot Phase, logic do phai duoc chuyen vao source module phu hop.

Notebook khong duoc dung execution order cua cell lam dependency ngam. Moi Phase call phai validate upstream state tu artifact va registry.

## 18. Dependency direction

Dependency duoc phep:

```text
notebook
-> experiments and phase-level public APIs
-> data, models, training, evaluation, attention
-> models/components
```

Dependency chi tiet:

- `data` chi phu thuoc utility co ban va dependency du lieu da phe duyet.
- `models` khong phu thuoc `training`, `evaluation`, `experiments` hoac notebook.
- `models/components` khong phu thuoc model cap cao.
- `training` co the phu thuoc model contracts, nhung khong phu thuoc notebook.
- `evaluation` co the nhan prediction bundle va model output, nhung khong so huu training.
- `experiments` co the orchestration data, models, training, evaluation va attention.
- `attention` co the doc model attention contract va frozen artifacts, nhung khong thay doi model state.
- notebook chi phu thuoc public API cua package.

Dependency bi cam:

```text
src -> notebook
models -> training
models -> experiments
data -> models
evaluation -> notebook
attention -> notebook
raw_data -> generated output
Test results -> model selection
```

Khong circular import. Neu hai module can import lan nhau, phai tach contract hoac dieu chinh ownership qua plan kien truc.

## 19. Phase-to-code mapping

| Phase | Trach nhiem | Vi tri code chinh |
|---|---|---|
| 0 | Coursework contract | Tai lieu Phase; source chi doc contract qua public API duoc phe duyet |
| 1 | Environment | `requirements.txt`, `training/reproducibility.py` |
| 2 | Data acquisition | `data/acquisition.py` |
| 3 | Schema audit | `data/schema.py` |
| 4 | Temporal integrity audit | `data/temporal.py` |
| 5 | EDA | Chua co module EDA owner; phai them qua plan, khong viet logic vao notebook |
| 6 | Feature engineering | `data/features.py` |
| 7 | Feature-set variants | `data/features.py` |
| 8 | Chronological split | `data/splitting.py` |
| 9 | Train-only scaling | `data/scaling.py` |
| 10 | Window builder | `data/windows.py` |
| 11 | DataLoaders | `data/datasets.py` |
| 12 | Shared metrics | `evaluation/metrics.py` |
| 13 | Experiment registry | `experiments/registry.py` |
| 14 | Persistence baseline | Chua co module owner trong kien truc hien tai; phai them module qua plan, khong viet vao notebook |
| 15 | LSTM implementation | `models/lstm.py` |
| 16 | Transformer implementation | `models/transformer.py`, `models/components/*` |
| 17 | Attention-aware verification | `models/components/attention.py`, `models/components/encoder_layer.py`, tests |
| 18 | Forward-pass sanity tests | `tests`, `evaluation/inference.py` |
| 19 | Baseline training engine | `training/engine.py`, `checkpointing.py`, `early_stopping.py`, `reproducibility.py` |
| 20 | LSTM baseline run | `experiments/registry.py` orchestration voi `models/lstm.py` va `training/engine.py` |
| 21 | Transformer B0 run | `experiments/registry.py` orchestration voi `models/transformer.py` va `training/engine.py` |
| 22 | Learning-curve diagnostics | Chua co module diagnostics rieng; phai them qua plan, khong viet vao notebook |
| 23 | S1 feature-set sweep | `experiments/sweeps.py`, `data/features.py` |
| 24 | S2 time-feature sweep | `experiments/sweeps.py`, `data/features.py`, `data/temporal.py` |
| 25 | S3 target-scaling sweep | `experiments/sweeps.py`, `data/scaling.py` |
| 26 | S4 lookback sweep | `experiments/sweeps.py`, `data/windows.py` |
| 27 | S5 pooling sweep | `experiments/sweeps.py`, `models/transformer.py` |
| 28 | S6 activation sweep | `experiments/sweeps.py`, `models/transformer.py`, `models/components/encoder_layer.py` |
| 29 | S7 batch sweep | `experiments/sweeps.py`, `training/engine.py`, `data/datasets.py` |
| 30 | S8 learning-rate sweep | `experiments/sweeps.py`, `training/engine.py` |
| 31 | S9 weight-decay sweep | `experiments/sweeps.py`, `training/engine.py` |
| 32 | S10 dropout sweep | `experiments/sweeps.py`, `models/transformer.py`, `models/components/encoder_layer.py` |
| 33 | S11 d-model sweep | `experiments/sweeps.py`, `models/transformer.py` |
| 34 | S12 head sweep | `experiments/sweeps.py`, `models/transformer.py`, `models/components/attention.py` |
| 35 | S13 layer sweep | `experiments/sweeps.py`, `models/transformer.py` |
| 36 | S14 FFN sweep | `experiments/sweeps.py`, `models/components/encoder_layer.py` |
| 37 | S15 loss sweep | `experiments/sweeps.py`, `training/engine.py` |
| 38 | S16 epoch-cap sweep | `experiments/sweeps.py`, `training/engine.py`, `training/early_stopping.py` |
| 39 | S17 gradient-clipping sweep | `experiments/sweeps.py`, `training/engine.py` |
| 40 | S18 RevIN sweep | `experiments/sweeps.py`, `models/components/revin.py`, `models/transformer.py` |
| 41 | S19 boundary-protocol check | `experiments/sweeps.py`, `data/splitting.py`, `data/windows.py` |
| 42 | Candidate synthesis | `experiments/final_lock.py` |
| 43 | LSTM tuning | `experiments/sweeps.py`, `models/lstm.py`, `training/engine.py` |
| 44 | Rolling-origin robustness | `experiments/sweeps.py`, `evaluation/metrics.py` |
| 45 | Final model lock | `experiments/final_lock.py` |
| 46 | Three-seed final runs | `experiments/final_lock.py`, `training/engine.py`, `training/reproducibility.py` |
| 47 | Final Test evaluation | `evaluation/inference.py`, `evaluation/metrics.py`, `experiments/final_lock.py` |
| 48 | Prediction analysis | `evaluation/inference.py`, `evaluation/residuals.py` |
| 49 | Residual analysis | `evaluation/residuals.py` |
| 50 | Error by regime | `evaluation/regimes.py` |
| 51 | Worst-error analysis | `evaluation/regimes.py`, `evaluation/residuals.py` |
| 52 | Attention extraction | `attention/extraction.py` |
| 53 | Attention heatmaps | `attention/heatmaps.py` |
| 54 | Last-query attention | `attention/last_query.py` |
| 55 | Head comparison | `attention/head_comparison.py` |
| 56 | Error-conditioned attention | `attention/head_comparison.py`, `evaluation/residuals.py`, `evaluation/regimes.py` |
| 57 | Seed-stability attention | `attention/head_comparison.py` |
| 58 | Final tables | Chua co reporting module trong kien truc hien tai; phai them qua plan, khong viet logic vao notebook |
| 59 | Final conclusions | Tai lieu ket luan chi duoc tao tu artifacts da verify; neu can generator phai them reporting module qua plan |

## 20. Quy tac implement tung Phase

Moi Phase phai theo flow:

```text
Doc rules
-> doc Phase detail
-> verify upstream Phase
-> audit files lien quan
-> tao pre-process plan
-> Human approval
-> implement trong owner module
-> viet hoac cap nhat tests
-> chay unit validation
-> chay upstream regression
-> chay current-Phase validation
-> chay downstream compatibility checks
-> ghi artifacts va registry
-> cap nhat execution record
-> chi sau do them notebook call
```

Sau moi buoc implementation:

1. Verify output cua buoc vua thuc thi.
2. Verify input contract tu buoc truoc.
3. Verify khong thay doi raw data.
4. Verify khong access Test trai gate.
5. Verify khong tao dependency ngoai plan.
6. Verify tests lien quan.
7. Dung ngay neu co discrepancy chua giai thich.

Notebook call luon la buoc cuoi, sau khi source logic va tests da hoan tat.

## 21. Quy tac code

Moi code trong `src/course_work` phai:

- co mot responsibility ro rang;
- dung ten phan anh nghiep vu;
- co input and output contract;
- validate shape, dtype, device va temporal alignment khi lien quan;
- deterministic khi protocol yeu cau;
- khong hard-code absolute path;
- khong log du lieu nhay cam hoac toan bo dataset;
- khong ghi file ngoai output path da phe duyet;
- khong truy cap Test ngoai Phase cho phep;
- khong duplicate logic;
- khong them comment hoac icon;
- khong che giau exception;
- khong fabricate result;
- khong tu dong repair du lieu.

Public API phai on dinh trong mot Phase chain. Neu doi signature, phai kiem tra tat ca caller, tests va notebook call.

## 22. Configuration va parameter ownership

Kien truc hien tai chua co `configs` directory va chua co config module rieng.

Do do:

- khong duoc luu hyperparameter canonical trong notebook;
- khong duoc duplicate config giua cac module;
- moi config tam thoi phai thuoc pre-process plan va Phase contract;
- truoc official run, phai co plan bo sung config ownership ro rang;
- checkpoint va registry phai luu config fingerprint;
- moi sweep chi thay doi bien duoc Phase cho phep.

Khong tu tao `configs`, YAML, JSON hoac Python config file khi chua co plan kien truc duoc Human phe duyet.

## 23. Artifact va output ownership

Kien truc hien tai chua co `artifacts` directory canonical.

Vi vay khong duoc tu do ghi checkpoint, scaler, history, predictions hoac attention tensor vao mot vi tri tuy y.

Truoc Phase tao runtime output, pre-process plan phai chot:

- output root;
- file naming;
- format;
- schema;
- atomic-write rule;
- checksum;
- immutability;
- retention;
- registry reference;
- overwrite policy.

Phan loai output:

- raw dataset: `data/raw_data`;
- approved split data: `data/data_after_split`;
- human-readable verified result: `docs/result`;
- report diagram and exported image: `image_diagram`;
- process record: `docs/save_process_proceduce_own_phase_refactor&fix`;
- runtime artifacts chua co canonical root: blocked cho den khi co plan duoc phe duyet.

Khong luu checkpoint, tensor, prediction bundle hoac scaler trong notebook.

## 24. Pretraining, LoRA va partial fine-tuning

Workflow adaptation phai tach ba lop:

```text
models/components/lora.py
-> LoRA primitive

training/pretraining.py
-> pretraining objective and checkpoint

training/fine_tuning.py
-> freeze policy and optimization

experiments/adaptation.py
-> comparison orchestration and registry
```

So sanh LoRA va partial fine-tuning phai cung:

- pretrained source checkpoint;
- Train and Validation boundary;
- feature set;
- target definition;
- lookback and horizon;
- seed list;
- evaluation metrics;
- early-stopping rule;
- compute-budget policy;
- Test firewall.

Phai bao cao:

- total parameters;
- trainable parameters;
- trainable percentage;
- training time;
- peak memory neu do duoc dang tin cay;
- Validation metrics;
- final Test metrics sau lock;
- convergence behavior;
- failure modes.

Khong ket luan LoRA tot hon chi vi it parameter hon. Ket luan phai dua tren metric, stability, resource cost va generalization.

## 25. Cac khoang trong kien truc hien tai

Cac responsibility sau chua co owner module rieng:

- persistence baseline cua Phase 14;
- EDA implementation cua Phase 5;
- learning-curve diagnostics cua Phase 22;
- final table generation cua Phase 58;
- final conclusion generation cua Phase 59;
- canonical configuration storage;
- canonical runtime artifact root;
- package initialization and public API files neu Python packaging yeu cau.

Khong duoc dua cac logic nay vao notebook de lap khoang trong.

Khi den Phase lien quan, phai:

1. ghi nhan missing-owner issue;
2. de xuat module va dependency;
3. tao pre-process architecture plan;
4. xin Human approval;
5. tao module;
6. test module;
7. sau cung moi them notebook call.

## 26. Naming conventions

Python package, module, function va variable:

```text
snake_case
```

Class:

```text
PascalCase
```

Constant:

```text
UPPER_SNAKE_CASE
```

Test file:

```text
test_<module>_<behavior>.py
```

Artifact file phai chua toi thieu Phase hoac run identity neu schema cua Phase khong quy dinh ten chinh xac.

Khong dung ten mo ho nhu:

```text
temp.py
new.py
final_final.py
utils2.py
test1.py
result_new.csv
```

## 27. Change-control gate

Can architecture plan va Human approval truoc khi:

- tao package moi;
- doi ten file;
- di chuyen logic giua module;
- thay dependency direction;
- tao config root;
- tao artifact root;
- thay data lifecycle;
- thay notebook boundary;
- them Phase;
- thay Phase owner;
- thay public API;
- thay checkpoint contract;
- thay Test-access policy.

Refactor architecture khong duoc thay scientific protocol mot cach ngam dinh.

## 28. Acceptance gate cho mot Phase

Mot Phase chi duoc danh dau complete khi:

- upstream contracts hop le;
- implementation nam dung owner module;
- notebook khong chua processing logic;
- tests bat buoc pass;
- input and output schema pass;
- temporal alignment pass;
- leakage checks pass;
- expected artifact ton tai tai location da phe duyet;
- registry da ghi run hoac Phase status;
- discrepancy da duoc resolve hoac cong khai;
- execution record da duoc cap nhat;
- downstream handoff da duoc verify.

Neu mot dieu kien khong dat, Phase giu trang thai blocked hoac incomplete.

## 29. Quy tac cho nguoi dung va agent tiep theo

Truoc khi sua mot file:

1. Xac dinh Phase va objective.
2. Tim owner module trong tai lieu nay.
3. Doc Phase detail.
4. Doc upstream, target va downstream files.
5. Kiem tra data and artifact state.
6. Lap plan va xin phe duyet theo `rule_code.md`.
7. Chi sua file trong approved scope.
8. Verify ngay sau moi buoc.
9. Khong chuyen buoc khi con loi.
10. Bao cao file da thay doi, validation, risk va next gate.

Khong duoc:

- dua logic vao notebook cho nhanh;
- tao file moi de tranh sua owner module;
- thay source of truth bang notebook output;
- suy dien ket qua chua chay;
- bo qua persistence baseline;
- bo qua LSTM baseline;
- tune tren Test;
- thay Transformer Encoder bang architecture khac ma khong co amendment;
- giai thich attention nhu quan he nhan qua;
- goi Phase complete khi output hoac tests chua dat.

## 30. Trang thai ap dung

Tai lieu nay mo ta ownership va execution boundary cua kien truc `COURSE_WORK` tai thoi diem no duoc Human chap nhan.

Moi thay doi sau nay phai duoc version hoa, neu ro ly do, pham vi anh huong va migration path. Khong sua rule trong im lang de hop thuc hoa code da ton tai.
