# Implementation Plan — PyTorch FashionMNIST Classification

> **Ngày tạo:** 2026-07-24  
> **Agent:** Main Agent (Cursor)  
> **Workspace:** `/Users/ticoder-coder/Documents/DEEP_LEARNING/LAB&PRACTICE`

---

## Mục Lục

1. [Tổng Quan Kiến Trúc](#1-tổng-quan-kiến-trúc)
2. [Cấu Trúc Thư Mục](#2-cấu-trúc-thư-mục)
3. [Phân Tích Đề Bài](#3-phân-tích-đề-bài)
4. [Thiết Kế Chi Tiết](#4-thiết-kế-chi-tiết)
5. [Code Implementation](#5-code-implementation)
6. [Kế Hoạch Thực Nghiệm](#6-kế-hoạch-thực-nghiệm)
7. [Visualization](#7-visualization)
8. [Deliverables Checklist](#8-deliverables-checklist)
9. [Rủi Ro và Xử Lý](#9-rủi-ro-và-xử-lý)
10. [Definition of Done](#10-definition-of-done)

---

## 1. Tổng Quan Kiến Trúc

### 1.1 Mục Tiêu
- Hoàn thành bài tập PyTorch FashionMNIST Classification
- Thực hành toàn bộ Deep Learning pipeline: Dataset → Transform → Model → Autograd → Optimization → Evaluation → Saving/Loading
- Không dùng notebook để code chính; code Python trong `processing_own_phase/`
- Log quá trình trong `save_log_agent_process_each_phase/`

### 1.2 Kiến Trúc Tổng Quan

```
FashionMNIST Dataset (60k train + 10k test)
        │
        ├── 54.000 train
        ├──  6.000 validation
        └── 10.000 test (CHỈ DÙNG CUỐI)
                │
                ▼
        DataLoader + Transform
                │
                ▼
        FashionMLP (subclass nn.Module)
                │
                ▼
        Training Loop: forward → loss → backward → optimize
                │
                ▼
        Validation Loop (chọn best model)
                │
                ▼
        Experiment Tracking
                │
                ▼
        Final Test Evaluation (1 LẦN DUY NHẤT)
                │
                ▼
        Save Model + Visualization
```

---

## 2. Cấu Trúc Thư Mục

```
practice_1/
│
├── practice_1.ipynb                    # File placeholder (không code ở đây)
│
├── processing_own_phase/               # Tất cả code Python
│   ├── __init__.py
│   ├── config.py                      # Cấu hình hyperparameters
│   ├── data.py                       # Dataset, Transforms, DataLoaders
│   ├── model.py                      # FashionMLP class
│   ├── train.py                      # Training functions
│   ├── evaluate.py                    # Evaluation functions
│   ├── experiment.py                  # Experiment runner
│   ├── visualize.py                  # Visualization functions
│   ├── save_load.py                  # Model saving/loading
│   ├── main.py                       # Entry point
│   └── utils.py                     # Helper functions
│
├── save_log_agent_process_each_phase/ # Log quá trình làm việc
│   ├── first_plan.md                 # Plan này
│   ├── phase_01_environment.md       # Log setup environment
│   ├── phase_02_data_loading.md     # Log data loading
│   ├── phase_03_model_build.md       # Log model building
│   ├── phase_04_baseline_train.md    # Log baseline training
│   ├── phase_05_experiments.md       # Log experiments
│   ├── phase_06_final_test.md       # Log final test
│   ├── phase_07_visualization.md     # Log visualization
│   └── experiment_results.csv        # Bảng kết quả experiments
│
├── planuml/                          # PlanUML diagrams (nếu có)
│   └── architecture.puml
│
└── outputs/                         # Output files (tạo khi chạy)
    ├── loss_curve.png
    ├── accuracy_curve.png
    ├── experiment_comparison.png
    ├── predictions_grid.png
    ├── confusion_matrix.png
    └── best_model.pth
```

---

## 3. Phân Tích Đề Bài

### 3.1 Problem Definition

| Thành phần | Giá trị |
|------------|----------|
| **Input** | Ảnh grayscale 28 × 28 pixels |
| **Output** | 10 classes (0-9) |
| **Task** | Supervised multi-class classification |
| **Model** | Artificial Neural Network (MLP) |
| **Metric** | Accuracy, Cross-Entropy Loss |

### 3.2 FashionMNIST Classes

| Index | Tên |
|-------|-----|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

### 3.3 Deliverables

| Deliverable | Định dạng | Vị trí |
|------------|-----------|---------|
| Python code | `.py` files | `processing_own_phase/` |
| Brief report | Markdown cells + summary | `phase_XX_final.md` |
| Loss graphs | `.png` | `outputs/` |
| Predicted vs actual images | `.png` | `outputs/` |
| Saved model | `.pth` | `outputs/` |
| Experiment results | `.csv` + `.md` | `save_log_agent_process_each_phase/` |

---

## 4. Thiết Kế Chi Tiết

### 4.1 Config (`config.py`)

```python
CONFIG = {
    # Seed cho reproducibility
    "seed": 42,
    
    # Data
    "batch_size": 64,
    "train_split_ratio": 0.9,  # 54k train, 6k val
    
    # Model
    "hidden_dims": [128],  # Baseline: [128]
    "dropout": 0.0,
    
    # Training
    "epochs": 10,
    "learning_rate": 0.01,
    "optimizer": "SGD",  # hoặc "Adam"
    
    # Paths
    "data_dir": "./data",
    "output_dir": "./outputs",
    "model_save_path": "./outputs/best_model.pth",
}
```

### 4.2 Data Pipeline (`data.py`)

#### Transforms
```python
transform_train = transforms.Compose([
    transforms.ToImage(),           # PIL → tensor
    transforms.ToDtype(torch.float32, scale=True),  # [0, 255] → [0, 1]
])

transform_test = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
])
```

#### Data Split
```
60,000 training images
        │
        ├── 54,000 train (90%)
        └──  6,000 validation (10%)

10,000 test images (GIỮ NGUYÊN, KHÔNG CHẠM)
```

### 4.3 Model (`model.py`)

```python
class FashionMLP(nn.Module):
    def __init__(self, hidden_dims: list, dropout: float = 0.0):
        """
        Args:
            hidden_dims: list of hidden layer sizes, e.g., [128] or [256, 128]
            dropout: dropout probability
        """
        super().__init__()
        layers = []
        in_features = 784
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_dim
        
        layers.append(nn.Linear(in_features, 10))  # 10 classes
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten: [B, 1, 28, 28] → [B, 784]
        return self.network(x)  # Logits, KHÔNG softmax
```

### 4.4 Training Loop (`train.py`)

```python
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = total_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc
```

### 4.5 Evaluation (`evaluate.py`)

```python
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            probabilities = F.softmax(logits, dim=1)
            _, predicted = logits.max(1)
            
            total_loss += loss.item() * images.size(0)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "predictions": np.array(all_predictions),
        "labels": np.array(all_labels),
        "probabilities": np.array(all_probabilities),
    }
```

### 4.6 Experiment Runner (`experiment.py`)

```python
class ExperimentRunner:
    def __init__(self, config):
        self.config = config
        self.experiments = []
    
    def run_experiment(self, exp_id, model_config, train_loader, val_loader, device):
        """
        Run single experiment and return results
        """
        # Create model
        model = FashionMLP(**model_config).to(device)
        
        # Train
        history = []
        best_val_acc = 0.0
        best_epoch = 0
        
        for epoch in range(self.config["epochs"]):
            train_loss, train_acc = train_one_epoch(...)
            val_loss, val_acc = evaluate(...)
            
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            })
            
            # Save best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                self.save_checkpoint(model, ...)
        
        return {
            "exp_id": exp_id,
            "model_config": model_config,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "history": history,
        }
```

---

## 5. Code Implementation

### 5.1 File Structure

```
processing_own_phase/
├── __init__.py
├── config.py           # CONFIG dictionary
├── data.py             # Dataset, Transforms, DataLoaders
├── model.py            # FashionMLP class
├── train.py            # train_one_epoch, fit
├── evaluate.py         # evaluate function
├── experiment.py       # ExperimentRunner class
├── visualize.py        # Plotting functions
├── save_load.py        # Model save/load utilities
├── utils.py            # Helper functions
├── main.py             # Entry point
└── run_experiments.py  # Script to run all experiments
```

### 5.2 Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.7.0
numpy>=1.24.0
pandas>=2.0.0
seaborn>=0.12.0  # Cho confusion matrix
tqdm>=4.65.0      # Progress bar
```

### 5.3 Entry Point (`main.py`)

```python
def main():
    # 1. Setup
    setup_reproducibility(CONFIG["seed"])
    device = get_device()
    create_directories()
    
    # 2. Load data
    train_dataset, test_dataset = load_datasets()
    train_subset, val_subset = split_train_val(train_dataset)
    train_loader, val_loader, test_loader = create_dataloaders(...)
    
    # 3. EDA
    visualize_data_samples(train_loader)
    plot_class_distribution(train_subset, val_subset, test_subset)
    
    # 4. Model sanity check
    model = FashionMLP(CONFIG["hidden_dims"], CONFIG["dropout"])
    sanity_check(model, train_loader, device)
    
    # 5. Baseline training
    exp_runner = ExperimentRunner(CONFIG)
    baseline_result = exp_runner.run_experiment(
        "E0_baseline",
        {"hidden_dims": CONFIG["hidden_dims"], "dropout": CONFIG["dropout"]},
        train_loader, val_loader, device
    )
    
    # 6. Experiments
    # ... (xem section 6)
    
    # 7. Final test
    best_model = load_best_model(CONFIG["model_save_path"])
    final_result = evaluate(best_model, test_loader, criterion, device)
    
    # 8. Visualization
    plot_loss_curves(results)
    plot_predictions(best_model, test_loader, device)
    plot_confusion_matrix(final_result)
    
    print(f"Final Test Accuracy: {final_result['accuracy']:.4f}")
```

---

## 6. Kế Hoạch Thực Nghiệm

### 6.1 Phase A — Baseline

| ID | Architecture | Optimizer | LR | Batch | Dropout |
|----|-------------|-----------|-----|-------|---------|
| E0 | [128] | SGD | 0.01 | 64 | 0.0 |

**Mục tiêu:**
- Verify pipeline hoạt động đúng
- Tạo baseline để so sánh
- Generate loss curve đầu tiên

### 6.2 Phase B — Learning Rate Experiment

| ID | Architecture | LR | Mục tiêu |
|----|-------------|-----|----------|
| E1 | [128] | 0.001 | LR thấp hơn |
| E0 | [128] | 0.01 | Baseline |
| E2 | [128] | 0.1 | LR cao hơn |

**So sánh:**
- Convergence speed
- Final validation loss
- Stability của loss curve

### 6.3 Phase C — Architecture Experiment

| ID | Architecture | Mục tiêu |
|----|-------------|----------|
| E-best-LR | [128] | Baseline |
| E3 | [256, 128] | Thêm capacity |

**Câu hỏi:** Thêm hidden layer có cải thiện đủ rõ không?

### 6.4 Phase D — Regularization

| ID | Architecture | Dropout |
|----|-------------|---------|
| E3 | [256, 128] | 0.0 |
| E4 | [256, 128] | 0.2 |

**Câu hỏi:** Dropout có giảm overfitting không?

### 6.5 Phase E — Optimizer Comparison

| ID | Optimizer | LR |
|----|-----------|-----|
| E4 | SGD | best LR |
| E5 | Adam | 0.001 |

### 6.6 Experiment Results Table

| Experiment | Architecture | Optimizer | LR | Dropout | Best Val Acc | Best Epoch |
|------------|-------------|-----------|-----|---------|--------------|------------|
| E0 | [128] | SGD | 0.01 | 0.0 | ... | ... |
| E1 | [128] | SGD | 0.001 | 0.0 | ... | ... |
| ... | ... | ... | ... | ... | ... | ... |

---

## 7. Visualization

### 7.1 Required Plots

| Plot | File | Description |
|------|------|-------------|
| Loss Curve | `loss_curve.png` | Train vs Val loss theo epoch |
| Accuracy Curve | `accuracy_curve.png` | Train vs Val accuracy theo epoch |
| Data Samples | `data_samples.png` | 20 ảnh ngẫu nhiên |
| Class Distribution | `class_distribution.png` | Bar chart 10 classes |
| Predictions Grid | `predictions_grid.png` | 16 ảnh với actual vs predicted |
| Error Analysis | `error_analysis.png` | Chỉ các prediction sai |
| Confusion Matrix | `confusion_matrix.png` | Heatmap 10x10 |
| Experiment Comparison | `experiment_comparison.png` | Bar chart so sánh experiments |

### 7.2 Color Conventions

- Prediction đúng: Tiêu đề màu xanh lá
- Prediction sai: Tiêu đề màu đỏ
- Confidence cao nhưng sai: Highlight đặc biệt

---

## 8. Deliverables Checklist

### 8.1 Code Files

- [ ] `config.py` — Hyperparameters
- [ ] `data.py` — Dataset loading và transforms
- [ ] `model.py` — FashionMLP class
- [ ] `train.py` — Training functions
- [ ] `evaluate.py` — Evaluation functions
- [ ] `experiment.py` — Experiment runner
- [ ] `visualize.py` — Plotting functions
- [ ] `save_load.py` — Model persistence
- [ ] `utils.py` — Helpers
- [ ] `main.py` — Entry point
- [ ] `run_experiments.py` — Full experiment runner
- [ ] `requirements.txt` — Dependencies

### 8.2 Log Files

- [ ] `first_plan.md` — Plan này
- [ ] `phase_01_environment.md` — Setup log
- [ ] `phase_02_data_loading.md` — Data loading log
- [ ] `phase_03_model_build.md` — Model building log
- [ ] `phase_04_baseline_train.md` — Baseline training log
- [ ] `phase_05_experiments.md` — Experiments log
- [ ] `phase_06_final_test.md` — Final test log
- [ ] `experiment_results.csv` — Results table

### 8.3 Output Files

- [ ] `outputs/best_model.pth` — Trained model
- [ ] `outputs/loss_curve.png`
- [ ] `outputs/accuracy_curve.png`
- [ ] `outputs/predictions_grid.png`
- [ ] `outputs/confusion_matrix.png`
- [ ] `outputs/experiment_comparison.png`

---

## 9. Rủi Ro và Xử Lý

| Rủi ro | Xử lý |
|---------|--------|
| Test leakage | Test loader không xuất hiện trong training/experiment |
| Shape mismatch | Sanity check với batch nhỏ |
| Gradient tích lũy | `optimizer.zero_grad()` mỗi batch |
| Quên eval mode | `model.train()` và `model.eval()` đúng chỗ |
| Không deterministic | Set seed + `torch.use_deterministic_algorithms()` |
| Save nhầm last model | Save best validation checkpoint |
| Load model sai | Verify predictions trước và sau load |

---

## 10. Definition of Done

### 10.1 Code Requirements

- [ ] Code chạy từ đầu đến cuối không lỗi
- [ ] Tải FashionMNIST tự động
- [ ] In đúng dataset sizes và tensor shapes
- [ ] Có train/validation/test separation
- [ ] FashionMLP subclass từ `nn.Module`
- [ ] Có forward, loss, backward, optimizer step
- [ ] Có train loss và validation loss theo epoch
- [ ] Có baseline và ít nhất 3 experiments có kiểm soát
- [ ] Có bảng so sánh experiments
- [ ] Có loss curve và accuracy curve
- [ ] Có predicted vs actual images
- [ ] Có error analysis images
- [ ] Có final test accuracy
- [ ] Có file `.pth`
- [ ] Load vào model object mới thành công
- [ ] Predictions trước và sau load nhất quán

### 10.2 Log Requirements

- [ ] Mỗi phase có log riêng trong `save_log_agent_process_each_phase/`
- [ ] Log chứa code đã chạy (copy paste)
- [ ] Log chứa output thực tế
- [ ] Log chứa observations và issues (nếu có)
- [ ] Bảng experiment results được cập nhật

### 10.3 Report Requirements

- [ ] Mô tả dataset
- [ ] Mô tả model architecture
- [ ] Mô tả training setup
- [ ] Bảng experiments với giả thuyết
- [ ] Kết quả thực tế (không gian lận)
- [ ] Error analysis
- [ ] Conclusion

---

## 11. Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: SETUP                               │
│  ├── Create directories                                        │
│  ├── Install dependencies                                      │
│  ├── Setup reproducibility (seed)                              │
│  └── Get device (CUDA/MPS/CPU)                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 2: DATA                                 │
│  ├── Load FashionMNIST                                         │
│  ├── Apply transforms                                          │
│  ├── Split train/validation                                    │
│  └── Create DataLoaders                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 3: MODEL                                │
│  ├── Define FashionMLP class                                   │
│  ├── Sanity checks (forward, loss, backward)                    │
│  └── Small overfit test                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 4: BASELINE                             │
│  ├── Train E0 baseline                                         │
│  ├── Plot loss/accuracy curves                                 │
│  └── Save baseline results                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 5: EXPERIMENTS                          │
│  ├── Phase B: Learning rate experiments                        │
│  ├── Phase C: Architecture experiments                        │
│  ├── Phase D: Regularization experiments                      │
│  ├── Phase E: Optimizer experiments                            │
│  └── Compare all experiments                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 6: SELECTION                           │
│  ├── Select best model by validation accuracy                   │
│  ├── Consider validation loss and train-val gap                │
│  └── Lock configuration                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 7: FINAL TEST                          │
│  ├── Load best checkpoint                                      │
│  ├── Evaluate on test set (ONCE ONLY)                        │
│  ├── Generate confusion matrix                                 │
│  └── Generate prediction images                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 8: SAVE & REPORT                       │
│  ├── Save best model                                          │
│  ├── Verify loading                                           │
│  ├── Create visualizations                                     │
│  └── Write final report                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Agent Acknowledgement

**Tôi xác nhận đã đọc và hiểu toàn bộ plan này.**

**Cam kết:**
1. Chỉ code trong `processing_own_phase/`, không code trong notebook
2. Log quá trình trong `save_log_agent_process_each_phase/`
3. Không tự ý thêm cấu trúc thư mục khi chưa được phép
4. Tuân thủ train/val/test split đúng cách
5. Không dùng test set để tuning
6. Verify model loading trước khi báo cáo kết quả

**Signature:** `Main Agent (Cursor) — 2026-07-24`
