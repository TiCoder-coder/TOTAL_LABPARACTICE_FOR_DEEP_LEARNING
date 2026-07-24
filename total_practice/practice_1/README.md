<div align="center">

<img
src="https://capsule-render.vercel.app/api?type=waving&height=220&color=0:111827,45:4F46E5,100:EC4899&text=PyTorch%20FashionMNIST%20Classification&fontColor=FFFFFF&fontSize=38&fontAlignY=38&desc=Deep%20Learning%20Lab%20Practice%20%E2%80%A2%20MLP%20%E2%80%A2%20PyTorch%20%E2%80%A2%20Apple%20Silicon&descAlignY=60&animation=fadeIn"
width="100%"
alt="FashionMNIST Project Banner"
/>

<img
src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=22&duration=2600&pause=900&color=8B5CF6&center=true&vCenter=true&width=900&lines=Learn+%E2%86%92+Build+%E2%86%92+Train+%E2%86%92+Evaluate;Six+Controlled+Deep+Learning+Experiments;Validation-Driven+Model+Selection;Reproducible+PyTorch+Workflow"
alt="Animated typing introduction"
/>

<br/>



<br/>

<img
src="https://skillicons.dev/icons?i=python,pytorch,git,github,vscode"
alt="Technology icons"
/>

<br/><br/>

<a href="#-project-overview">
  <img src="https://img.shields.io/badge/Overview-4F46E5?style=flat-square" alt="Overview"/>
</a>
<a href="#-dataset">
  <img src="https://img.shields.io/badge/Dataset-0EA5E9?style=flat-square" alt="Dataset"/>
</a>
<a href="#-model-architecture">
  <img src="https://img.shields.io/badge/Model-8B5CF6?style=flat-square" alt="Model"/>
</a>
<a href="#-training-pipeline">
  <img src="https://img.shields.io/badge/Pipeline-F59E0B?style=flat-square" alt="Pipeline"/>
</a>
<a href="#-experiments--results">
  <img src="https://img.shields.io/badge/Experiments-EC4899?style=flat-square" alt="Experiments"/>
</a>
<a href="#-how-to-run">
  <img src="https://img.shields.io/badge/Run%20Project-22C55E?style=flat-square" alt="Run project"/>
</a>

</div>

📌 PyTorch FashionMNIST Classification — Practice 1

Deep Learning Lab Practice using FashionMNIST, a reusable Multi-Layer Perceptron, controlled hyperparameter experiments, model evaluation, visualization, checkpoint saving, and reload verification.

🧭 Table of Contents

Section

Description

🚀 Project Overview

Problem, objective, framework, device, and best result

🗂️ Repository Structure

Main notebook, source package, outputs, and logs

📦 Dataset

FashionMNIST classes, split strategy, and transforms

🧠 Model Architecture

FashionMLP, hidden layers, dropout, and sanity checks

🔁 Training Pipeline

Complete eight-phase workflow

🧪 Experiments & Results

Six controlled experiments and comparison

🏆 Best Model

Final validation, test metrics, and class performance

🧩 Source Modules

Responsibilities of each Python module

▶️ How to Run

Notebook, command-line, and Python import methods

🖼️ Output Gallery

Loss, accuracy, predictions, confusion matrix, and artifacts

🛠️ Troubleshooting

Common runtime and import errors

🛣️ Development Roadmap

CNN, augmentation, tracking, deployment, and extensions

👨‍🎓 Student Information

Contact, university, subject, and group

🚀 Project Overview

<div align="center">

<table>
  <tr>
    <td align="center" width="20%">
      <h2>70,000</h2>
      <b>🖼️ Images</b>
    </td>
    <td align="center" width="20%">
      <h2>10</h2>
      <b>🏷️ Classes</b>
    </td>
    <td align="center" width="20%">
      <h2>6</h2>
      <b>🧪 Experiments</b>
    </td>
    <td align="center" width="20%">
      <h2>88.17%</h2>
      <b>🎯 Test Accuracy</b>
    </td>
    <td align="center" width="20%">
      <h2>235,146</h2>
      <b>🧠 Parameters</b>
    </td>
  </tr>
</table>

</div>

Information

Details

Problem

Classify 28 × 28 grayscale fashion images into 10 classes

Dataset

FashionMNIST: 60,000 training images and 10,000 test images

Framework

PyTorch 2.13.0

Device

MPS on Apple Silicon

Backbone

FashionMLP

Optimizers

SGD and Adam

Best Experiment

E5_adam

Best Validation Accuracy

88.58%

Final Test Accuracy

88.17%

Best Model Size

235,146 trainable parameters

🎯 Learning Objectives

Build a complete PyTorch workflow from data loading to final evaluation.

Practice tensors, datasets, data loaders, transforms, models, autograd, and optimization.

Compare neural-network architectures and hyperparameters using controlled experiments.

Select the best model using the validation set rather than the test set.

Visualize training loss, validation performance, confusion matrix, and predictions.

Save the best checkpoint, reload it into a fresh model, and verify identical behavior.

🗂️ Repository Structure

practice_1/
├── README.md
├── practice_1.ipynb
│
├── processing_own_phase/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize.py
│   ├── utils.py
│   ├── experiment.py
│   ├── save_load.py
│   ├── main.py
│   └── requirements.txt
│
├── outputs/
│   ├── best_model.pth
│   ├── E0_baseline.pt
│   ├── E1_lr_low.pt
│   ├── E2_lr_high.pt
│   ├── E3_deeper.pt
│   ├── E4_dropout.pt
│   ├── E5_adam.pt
│   ├── loss_curve.png
│   ├── accuracy_curve.png
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   ├── experiment_comparison.png
│   ├── data_samples.png
│   ├── predictions_grid.png
│   └── summary.json
│
├── data/
│   └── FashionMNIST/
│       └── raw/
│
└── save_log_agent_process_each_phase/
    ├── experiment_results.csv
    ├── plan_from_gpt.md
    └── phase_*.md

The source code is divided into reusable modules so that data processing, modeling, training, evaluation, visualization, experimentation, and model persistence remain independent and maintainable.

📦 Dataset

FashionMNIST Classes

FashionMNIST contains 28 × 28 grayscale images belonging to ten fashion categories.

Label

Class

Vietnamese Description

0

T-shirt/top

Áo phông

1

Trouser

Quần dài

2

Pullover

Áo len

3

Dress

Váy

4

Coat

Áo khoác

5

Sandal

Dép

6

Shirt

Áo sơ mi

7

Sneaker

Giày thể thao

8

Bag

Túi xách

9

Ankle boot

Ủng

Data Split Strategy

%%{init: {"theme":"base","themeVariables":{"primaryColor":"#312E81","primaryTextColor":"#FFFFFF","primaryBorderColor":"#818CF8","lineColor":"#8B5CF6","secondaryColor":"#FDF2F8","tertiaryColor":"#EFF6FF"}}}%%
flowchart TB
    A["📦 FashionMNIST<br/>70,000 images"]:::root

    A --> B["🧪 Official Training Set<br/>60,000 images"]:::train
    A --> C["🔒 Official Test Set<br/>10,000 images"]:::test

    B --> D["🔥 Train Subset<br/>54,001 images<br/>Model optimization"]:::train
    B --> E["🧭 Validation Subset<br/>5,999 images<br/>Model selection"]:::val

    C --> F["✅ Final Evaluation<br/>Used once after configuration lock"]:::test

    classDef root fill:#111827,stroke:#6D28D9,stroke-width:3px,color:#FFFFFF
    classDef train fill:#1D4ED8,stroke:#60A5FA,stroke-width:2px,color:#FFFFFF
    classDef val fill:#7C3AED,stroke:#C4B5FD,stroke-width:2px,color:#FFFFFF
    classDef test fill:#BE185D,stroke:#F9A8D4,stroke-width:2px,color:#FFFFFF

Why Three Data Partitions?

Partition

Purpose

Train set

Learn model parameters through backpropagation

Validation set

Select architecture, optimizer, learning rate, dropout, and best epoch

Test set

Estimate final generalization performance after all decisions are locked

[!IMPORTANT]The test set is evaluated only after the best validation configuration has been selected. Repeatedly checking test performance would contaminate the final evaluation.

Data Transform

PIL Image
   ↓
ToTensor()
   ↓
Float32 tensor
   ↓
Pixel range [0, 1]
   ↓
Shape [1, 28, 28]

ToTensor() converts each image into a torch.float32 tensor.

Baseline experiments use raw normalized pixel values in the range [0, 1].

Dataset mean and standard deviation can be computed through compute_mean_std() for future normalization experiments.

Commonly Confused Classes

Shirt     ↔ T-shirt/top
Pullover  ↔ Coat
Shirt     ↔ Pullover

These classes share similar shapes in low-resolution grayscale images, making them more difficult for a fully connected MLP to distinguish.

🧠 Model Architecture

FashionMLP

The model receives an image tensor and converts it into ten raw class logits.

%%{init: {"theme":"base","flowchart":{"curve":"basis","nodeSpacing":35,"rankSpacing":45},"themeVariables":{"lineColor":"#8B5CF6"}}}%%
flowchart LR
    A["🖼️ Input Image<br/>1 × 28 × 28"]:::input
    B["📐 Flatten<br/>784 features"]:::process
    C["🧠 Linear Layer<br/>784 → 256"]:::hidden
    D["⚡ ReLU"]:::activation
    E["🛡️ Dropout<br/>p = 0.2"]:::regularization
    F["🧠 Linear Layer<br/>256 → 128"]:::hidden
    G["⚡ ReLU"]:::activation
    H["🛡️ Dropout<br/>p = 0.2"]:::regularization
    I["🎯 Output Layer<br/>128 → 10"]:::output
    J["📊 Raw Logits<br/>No Softmax in Model"]:::logits

    A --> B --> C --> D --> E --> F --> G --> H --> I --> J

    classDef input fill:#0F172A,stroke:#64748B,stroke-width:2px,color:#FFFFFF
    classDef process fill:#0369A1,stroke:#7DD3FC,stroke-width:2px,color:#FFFFFF
    classDef hidden fill:#4338CA,stroke:#A5B4FC,stroke-width:2px,color:#FFFFFF
    classDef activation fill:#B45309,stroke:#FCD34D,stroke-width:2px,color:#FFFFFF
    classDef regularization fill:#9D174D,stroke:#F9A8D4,stroke-width:2px,color:#FFFFFF
    classDef output fill:#047857,stroke:#6EE7B7,stroke-width:2px,color:#FFFFFF
    classDef logits fill:#111827,stroke:#E5E7EB,stroke-width:2px,color:#FFFFFF

General Architecture

Input: [B, 1, 28, 28]
        ↓
Flatten
        ↓
Linear(784 → hidden_dims[0])
        ↓
ReLU
        ↓
Optional Dropout
        ↓
Additional hidden layers
        ↓
Linear(last_hidden → 10)
        ↓
Output: [B, 10] raw logits

Configurations Tested

Configuration

Hidden Dimensions

Dropout

Optimizer

Learning Rate

Parameters

Baseline

[128]

0.0

SGD

0.01

101,770

Two Layers

[256, 128]

0.0

SGD

0.01

235,146

Dropout

[256, 128]

0.2

SGD

0.01

235,146

Adam

[256, 128]

0.2

Adam

0.001

235,146

Forward Pass

def forward(self, x):
    # Supported input shapes:
    # [B, 1, 28, 28] or [B, 784]
    if x.dim() == 4:
        x = x.view(x.size(0), -1)
    elif x.dim() == 3:
        x = x.view(x.size(0), -1)

    return self.network(x)  # [B, 10] raw logits

nn.CrossEntropyLoss() expects raw logits, so Softmax is intentionally excluded from the final model layer. Probabilities are computed only when visualization or confidence reporting is required.

Model Sanity Checks

Before full training, sanity_check_model() verifies:

Forward shape: output must have shape [batch_size, 10].

Finite loss: cross-entropy loss must not contain NaN or Inf.

Gradient flow: model parameters must receive gradients after loss.backward().

Parameter update: at least one trainable parameter must change after an optimizer step.

🔁 Training Pipeline

The full workflow is orchestrated by:

run_full_pipeline()

Complete Eight-Phase Pipeline

%%{init: {"theme":"base","flowchart":{"curve":"basis","nodeSpacing":25,"rankSpacing":55},"themeVariables":{"lineColor":"#64748B"}}}%%
flowchart LR

    subgraph DATA["📦 DATA PIPELINE"]
        direction TB
        A1["1️⃣ Environment Setup<br/>Seed • Device • Versions"]:::data
        A2["2️⃣ Load FashionMNIST<br/>Transforms • Download"]:::data
        A3["3️⃣ Split Train / Validation<br/>54,001 / 5,999"]:::data
        A4["4️⃣ Build DataLoaders<br/>Train • Val • Test"]:::data
        A1 --> A2 --> A3 --> A4
    end

    subgraph MODEL["🧠 MODEL PIPELINE"]
        direction TB
        B1["5️⃣ Build FashionMLP<br/>Architecture from config"]:::model
        B2["6️⃣ Sanity Checks<br/>Forward • Loss • Gradient"]:::model
        B1 --> B2
    end

    subgraph EXP["🧪 EXPERIMENT PIPELINE"]
        direction TB
        C1["7️⃣ Run E0 → E5<br/>Controlled experiments"]:::experiment
        C2["📈 Validate Every Epoch<br/>Save best checkpoint"]:::experiment
        C3["🏆 Select Best Model<br/>Highest validation accuracy"]:::experiment
        C1 --> C2 --> C3
    end

    subgraph FINAL["✅ FINAL PIPELINE"]
        direction TB
        D1["8️⃣ Final Test<br/>Evaluate once"]:::final
        D2["📊 Visualize Results<br/>Curves • Matrix • Predictions"]:::final
        D3["💾 Save & Reload<br/>Fresh model instance"]:::final
        D4["🔍 Verify Model<br/>Identical predictions"]:::final
        D1 --> D2 --> D3 --> D4
    end

    A4 --> B1
    B2 --> C1
    C3 --> D1

    classDef data fill:#0369A1,stroke:#7DD3FC,stroke-width:2px,color:#FFFFFF
    classDef model fill:#4338CA,stroke:#A5B4FC,stroke-width:2px,color:#FFFFFF
    classDef experiment fill:#9D174D,stroke:#F9A8D4,stroke-width:2px,color:#FFFFFF
    classDef final fill:#047857,stroke:#6EE7B7,stroke-width:2px,color:#FFFFFF

Training Loop

%%{init: {"theme":"base","themeVariables":{"lineColor":"#8B5CF6"}}}%%
flowchart TD
    A["Start Epoch"]:::start
    B["model.train()"]:::train
    C["Load Batch<br/>images, labels"]:::train
    D["optimizer.zero_grad()"]:::train
    E["Forward Pass<br/>logits = model(images)"]:::train
    F["Compute Loss<br/>CrossEntropyLoss"]:::train
    G["Backward Pass<br/>loss.backward()"]:::train
    H["Update Weights<br/>optimizer.step()"]:::train
    I{"More Batches?"}:::decision
    J["Compute Train Metrics"]:::metric
    K["model.eval()<br/>torch.no_grad()"]:::eval
    L["Compute Validation Metrics"]:::eval
    M{"Validation Accuracy Improved?"}:::decision
    N["Save Best Checkpoint"]:::save
    O["Store Epoch History"]:::metric
    P{"More Epochs?"}:::decision
    Q["Finish Training"]:::finish

    A --> B --> C --> D --> E --> F --> G --> H --> I
    I -- Yes --> C
    I -- No --> J --> K --> L --> M
    M -- Yes --> N --> O
    M -- No --> O
    O --> P
    P -- Yes --> A
    P -- No --> Q

    classDef start fill:#111827,stroke:#A78BFA,stroke-width:3px,color:#FFFFFF
    classDef train fill:#1D4ED8,stroke:#93C5FD,stroke-width:2px,color:#FFFFFF
    classDef eval fill:#7C3AED,stroke:#C4B5FD,stroke-width:2px,color:#FFFFFF
    classDef decision fill:#B45309,stroke:#FCD34D,stroke-width:2px,color:#FFFFFF
    classDef metric fill:#0F766E,stroke:#5EEAD4,stroke-width:2px,color:#FFFFFF
    classDef save fill:#BE185D,stroke:#F9A8D4,stroke-width:2px,color:#FFFFFF
    classDef finish fill:#047857,stroke:#6EE7B7,stroke-width:3px,color:#FFFFFF

🧪 Experiments & Results

<div align="center">



</div>

Controlled Experiment Table

ID

Description

Hidden Dimensions

Dropout

Optimizer

LR

Best Val Acc

Parameters

Time

E0_baseline

One hidden layer baseline

[128]

0.0

SGD

0.01

88.13%

101,770

43.7s

E1_lr_low

Lower learning rate

[128]

0.0

SGD

0.001

84.31%

101,770

39.3s

E2_lr_high

Higher learning rate

[128]

0.0

SGD

0.1

85.85%

101,770

41.7s

E3_deeper

Two hidden layers

[256, 128]

0.0

SGD

0.01

88.45%

235,146

54.9s

E4_dropout

Add dropout regularization

[256, 128]

0.2

SGD

0.01

88.01%

235,146

53.6s

E5_adam

Adam optimizer

[256, 128]

0.2

Adam

0.001

88.58%

235,146

67.1s

Validation Accuracy Comparison

E0_baseline  ████████████████████████████████████████  88.13%
E1_lr_low    ██████████████████████████████████████    84.31%
E2_lr_high   ███████████████████████████████████████   85.85%
E3_deeper    ████████████████████████████████████████  88.45%
E4_dropout   ████████████████████████████████████████  88.01%
E5_adam      █████████████████████████████████████████ 88.58%

Experiment Interpretation

<table>
  <tr>
    <td width="20%" align="center"><b>🧪 Experiment</b></td>
    <td><b>🔍 Main Observation</b></td>
  </tr>
  <tr>
    <td align="center"><code>E1_lr_low</code></td>
    <td>The learning rate was too small for the fixed ten-epoch budget, resulting in slow convergence and underfitting.</td>
  </tr>
  <tr>
    <td align="center"><code>E2_lr_high</code></td>
    <td>The larger learning rate accelerated updates but produced less stable convergence and lower validation accuracy.</td>
  </tr>
  <tr>
    <td align="center"><code>E3_deeper</code></td>
    <td>Adding a second hidden layer improved validation accuracy by approximately 0.32 percentage points over the baseline.</td>
  </tr>
  <tr>
    <td align="center"><code>E4_dropout</code></td>
    <td>Dropout reduced effective capacity, but the model did not show enough overfitting for dropout alone to improve validation accuracy.</td>
  </tr>
  <tr>
    <td align="center"><code>E5_adam</code></td>
    <td>Adam with a learning rate of 0.001 achieved the strongest validation performance among the six experiments.</td>
  </tr>
</table>

[!NOTE]The improvement between E3_deeper and E5_adam is small. A stronger scientific comparison would repeat the best configurations across multiple random seeds and report mean ± standard deviation.

🏆 Best Model

<div align="center">

<img
src="https://img.shields.io/badge/Best%20Experiment-E5__adam-22C55E?style=for-the-badge&logo=pytorch&logoColor=white"
alt="Best experiment"
/>

</div>

Metric

Value

Architecture

FashionMLP [256, 128]

Dropout

0.2

Optimizer

Adam

Learning Rate

0.001

Best Epoch

10 / 10

Best Validation Accuracy

88.58%

Final Test Accuracy

88.17%

Final Test Loss

0.3286

Trainable Parameters

235,146

Per-Class Accuracy

Class

Accuracy

Assessment

Trouser

97.6%

✅ Very strong

Sandal

97.2%

✅ Very strong

Bag

97.1%

✅ Very strong

Sneaker

95.4%

✅ Strong

Ankle boot

94.3%

✅ Strong

Dress

91.6%

✅ Strong

Coat

80.4%

⚠️ Moderate

T-shirt/top

79.6%

⚠️ Moderate

Shirt

74.7%

❌ Difficult

Pullover

73.8%

❌ Most difficult

Save and Reload Verification

Same predictions after reload: True
Maximum logit difference:      0.0

This confirms that the checkpoint was saved and restored correctly without changing the model output.

🧩 Source Modules

<details>
<summary><b>⚙️ <code>config.py</code> — Global configuration</b></summary>

Defines PROJECT_ROOT, DATA_DIR, and OUTPUT_DIR.

Stores all ten FashionMNIST class names.

Defines global configuration values such as batch size, epochs, and seed.

Contains the six experiment definitions from E0 to E5.

</details>

<details>
<summary><b>📦 <code>data.py</code> — Dataset and DataLoader utilities</b></summary>

Function

Responsibility

get_transforms()

Returns train and test transforms

compute_mean_std()

Computes dataset mean and standard deviation

load_datasets()

Downloads and loads FashionMNIST

split_train_val()

Splits the official training set

get_class_distribution()

Counts samples in each class

make_dataloaders()

Builds train, validation, and test loaders

sanity_check_sample()

Verifies sample shape, type, and range

</details>

<details>
<summary><b>🧠 <code>model.py</code> — FashionMLP</b></summary>

Component

Responsibility

FashionMLP

Reusable fully connected neural network

build_model()

Creates a model from an experiment configuration

sanity_check_model()

Verifies forward pass, loss, gradients, and updates

</details>

<details>
<summary><b>🔥 <code>train.py</code> — Training process</b></summary>

Function

Responsibility

train_one_epoch()

Trains the model for one complete epoch

fit()

Trains for multiple epochs and evaluates validation performance

Best-checkpoint callback

Saves the model when validation accuracy improves

</details>

<details>
<summary><b>📊 <code>evaluate.py</code> — Evaluation</b></summary>

Function

Responsibility

evaluate()

Returns loss, accuracy, predictions, targets, and probabilities

per_class_accuracy()

Calculates accuracy for every class

</details>

<details>
<summary><b>🎨 <code>visualize.py</code> — Visualization</b></summary>

Function

Output

plot_data_samples()

data_samples.png

plot_class_distribution()

class_distribution.png

plot_loss_curves()

loss_curve.png

plot_accuracy_curves()

accuracy_curve.png

plot_experiment_comparison()

experiment_comparison.png

plot_confusion_matrix()

confusion_matrix.png

plot_predictions_grid()

predictions_grid.png

</details>

<details>
<summary><b>🧰 <code>utils.py</code> — Utility functions</b></summary>

get_device() chooses CUDA, MPS, or CPU.

setup_reproducibility() configures random seeds.

count_parameters() reports trainable parameter count.

format_time() formats elapsed time.

print_environment_summary() displays software and hardware information.

ensure_dir() creates missing directories.

</details>

<details>
<summary><b>🧪 <code>experiment.py</code> — ExperimentRunner</b></summary>

class ExperimentRunner:
    def run(
        exp_id,
        description,
        train_loader,
        val_loader,
        hidden_dims,
        dropout,
        optimizer_name,
        learning_rate,
        batch_size,
        epochs,
        ...
    ):
        # Build model
        # Build optimizer
        # Train and validate
        # Save best checkpoint
        # Append results to CSV
        # Return structured result

</details>

<details>
<summary><b>💾 <code>save_load.py</code> — Model persistence</b></summary>

Function

Responsibility

save_checkpoint()

Saves weights, architecture, configuration, and metrics

load_checkpoint()

Reads a checkpoint file

load_model_from_checkpoint()

Rebuilds and restores the model

verify_loaded_model()

Compares outputs before and after reload

</details>

<details>
<summary><b>🚀 <code>main.py</code> — Project entry point</b></summary>

Runs the complete eight-phase workflow.

Supports full and quick-test modes.

Generates visualizations.

Saves summary.json.

Returns the final experiment summary.

</details>

▶️ How to Run

Option 1 — Jupyter Notebook

cd "/Users/ticoder-coder/Documents/DEEP_LEARNING/LAB&PRACTICE/total_practice/practice_1"
jupyter notebook practice_1.ipynb

Run the notebook cells in order:

Check environment and PyTorch device support.

Run run_full_pipeline().

Display all generated figures.

Display the experiment comparison table.

from processing_own_phase.main import run_full_pipeline

# Full experiments: 10 epochs per configuration
summary = run_full_pipeline(quick_test=False)

For a faster pipeline check:

summary = run_full_pipeline(quick_test=True)

Option 2 — Run from Terminal

cd "/Users/ticoder-coder/Documents/DEEP_LEARNING/LAB&PRACTICE/total_practice/practice_1"

Quick Test

python -c "
import sys
sys.path.insert(0, '.')
from processing_own_phase.main import run_full_pipeline
run_full_pipeline(quick_test=True)
"

Full Pipeline

python -c "
import sys
sys.path.insert(0, '.')
from processing_own_phase.main import run_full_pipeline
run_full_pipeline(quick_test=False)
"

Option 3 — Import from Another Script

import sys

PROJECT_DIR = (
    "/Users/ticoder-coder/Documents/"
    "DEEP_LEARNING/LAB&PRACTICE/total_practice/practice_1"
)

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from processing_own_phase.main import run_full_pipeline

summary = run_full_pipeline(quick_test=False)
print(summary["test_accuracy"])

🖼️ Output Gallery

The following image paths use relative links. They will display directly on GitHub after the generated files are committed to the repository.

<table>
  <tr>
    <td align="center" width="50%">
      <b>📉 Training and Validation Loss</b><br/><br/>
      <img src="./outputs/loss_curve.png" alt="Loss curve" width="100%"/>
    </td>
    <td align="center" width="50%">
      <b>📈 Training and Validation Accuracy</b><br/><br/>
      <img src="./outputs/accuracy_curve.png" alt="Accuracy curve" width="100%"/>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <b>🧪 Experiment Comparison</b><br/><br/>
      <img src="./outputs/experiment_comparison.png" alt="Experiment comparison" width="100%"/>
    </td>
    <td align="center" width="50%">
      <b>🧩 Confusion Matrix</b><br/><br/>
      <img src="./outputs/confusion_matrix.png" alt="Confusion matrix" width="100%"/>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <b>👕 FashionMNIST Samples</b><br/><br/>
      <img src="./outputs/data_samples.png" alt="FashionMNIST samples" width="100%"/>
    </td>
    <td align="center" width="50%">
      <b>🔍 Predicted vs Actual</b><br/><br/>
      <img src="./outputs/predictions_grid.png" alt="Prediction grid" width="100%"/>
    </td>
  </tr>
</table>

Generated Artifacts

File

Description

best_model.pth

Final checkpoint of the selected model

E0_baseline.pt → E5_adam.pt

Checkpoints for all experiments

data_samples.png

Random FashionMNIST samples

class_distribution.png

Distribution of samples across classes

loss_curve.png

Training and validation loss

accuracy_curve.png

Training and validation accuracy

experiment_comparison.png

Validation comparison across experiments

confusion_matrix.png

Normalized confusion matrix

predictions_grid.png

Correct and incorrect predictions

summary.json

Final structured experiment summary

experiment_results.csv

Tabular results of all six experiments

🛠️ Troubleshooting

<details>
<summary><b>❌ ModuleNotFoundError: processing_own_phase</b></summary>

Cause: The project root is not available in sys.path.

import os
import sys

PROJECT_ROOT = (
    "/Users/ticoder-coder/Documents/"
    "DEEP_LEARNING/LAB&PRACTICE/total_practice/practice_1"
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.chdir(PROJECT_ROOT)

</details>

<details>
<summary><b>❌ ModuleNotFoundError: seaborn</b></summary>

python -m pip install seaborn

</details>

<details>
<summary><b>❌ ModuleNotFoundError: matplotlib</b></summary>

python -m pip install matplotlib numpy pandas seaborn

</details>

<details>
<summary><b>⚠️ MPS is not available</b></summary>

The project automatically falls back to CPU when MPS is unavailable.

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

</details>

<details>
<summary><b>⚠️ torch.inference_mode() is unavailable</b></summary>

For older PyTorch versions, replace it with:

with torch.no_grad():
    ...

</details>

🛣️ Development Roadmap

Tier 1 — Model Improvement

Replace the MLP with a CNN backbone.

Add controlled image augmentation.

Add Batch Normalization.

Evaluate learning-rate schedulers.

Test weight decay and stronger regularization.

Repeat top configurations across multiple seeds.

Tier 2 — Experiment Tracking

Integrate Weights & Biases.

Add TensorBoard logging.

Add MLflow experiment tracking.

Store configuration and metrics in a versioned experiment registry.

Tier 3 — Deployment

Export the model to ONNX.

Create an inference API using FastAPI.

Create an interactive Gradio interface.

Measure model latency, throughput, and memory.

Explore model quantization.

Tier 4 — Extended Problems

Apply the pipeline to CIFAR-10.

Use transfer learning with ResNet or EfficientNet.

Explore object detection for fashion products.

Explore generative models for fashion-image synthesis.

⚙️ Technical Environment

<div align="center">

Component

Version / Configuration

Python

3.10.11

PyTorch

2.13.0

TorchVision

0.28.0

Matplotlib

3.x

Seaborn

0.12+

NumPy

1.24+

Pandas

2.0+

Device

MPS — Apple Silicon

Epochs

10 per experiment

Batch Size

64

Best Model Parameters

235,146

</div>

👨‍🎓 Student Information

<div align="center">

<img
src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=20&duration=2800&pause=1000&color=EC4899&center=true&vCenter=true&width=780&lines=Deep+Learning+Lab+Practice;Ho+Chi+Minh+City+University+of+Transport;Group+2+%E2%80%A2+Build+with+PyTorch"
alt="Student information animation"
/>

<br/>

<table>
  <tr>
    <td align="center"><b>📧 Contact Email</b></td>
    <td align="center">
      <a href="mailto:voanhnhat1612@gmail.com">
        <b>voanhnhat1612@gmail.com</b>
      </a>
    </td>
  </tr>
  <tr>
    <td align="center"><b>🏛️ University</b></td>
    <td align="center"><b>Hồ Chí Minh City University of Transport</b></td>
  </tr>
  <tr>
    <td align="center"><b>📚 Subject</b></td>
    <td align="center"><b>Deep Learning</b></td>
  </tr>
  <tr>
    <td align="center"><b>👥 Group</b></td>
    <td align="center"><b>2</b></td>
  </tr>
</table>

<br/>

<a href="mailto:voanhnhat1612@gmail.com">
  <img
    src="https://img.shields.io/badge/Gmail-Contact%20Me-EA4335?style=for-the-badge&logo=gmail&logoColor=white"
    alt="Contact by Gmail"
  />
</a>

<a href="#">
  <img
    src="https://img.shields.io/badge/Subject-Deep%20Learning-6D28D9?style=for-the-badge&logo=pytorch&logoColor=white"
    alt="Deep Learning subject"
  />
</a>

<a href="#">
  <img
    src="https://img.shields.io/badge/Group-2-0EA5E9?style=for-the-badge&logo=github&logoColor=white"
    alt="Group 2"
  />
</a>

<br/><br/>

🧠 Learn → 🛠️ Build → 🔥 Train → 📊 Evaluate → 🚀 Improve

<br/>

<img
src="https://capsule-render.vercel.app/api?type=waving&height=140&section=footer&color=0:EC4899,55:7C3AED,100:111827&animation=fadeIn"
width="100%"
alt="Footer wave"
/>

<sub>
  <b>Document updated: 2026-07-24</b><br/>
  PyTorch FashionMNIST Classification — Deep Learning Lab Practice
</sub>

</div>