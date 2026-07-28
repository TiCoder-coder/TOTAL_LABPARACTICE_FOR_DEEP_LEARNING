# Changelog

All notable changes to this project will be documented in this file.

## [Phase 10] - Documentation, Portfolio & Final Release
- Fully rewrote `README.md` with a professional overview, Markdown formatting, and Mermaid architecture diagrams.
- Created `CHANGELOG.md` to document the 10-phase evolution of the project.
- Added `CONTRIBUTING.md` guidelines for open source collaboration.
- Extracted and displayed actual results in the README.
- Final validation ensuring zero technical debt and clean code.

## [Phase 9] - Repository Architecture & Project Engineering
- Standardized directory layout (`notebooks/`, `docs/`, `checkpoints/`, `configs/`, etc.).
- Refactored monolithic config into a modular `configs/` package (`core_config`, `training_config`, `experiment_config`).
- Fixed import paths across all pipeline modules and tests.
- Cleaned up scratch scripts and byte-compiled pycache files.
- Added MIT `LICENSE` and a comprehensive `.gitignore`.

## [Phase 8] - Research Notebook & Presentation
- Developed a professional `notebook.ipynb` tailored for presentation rather than logic storage.
- Integrated all external logic via standard imports, completely decoupling core logic from the notebook.
- Recreated plots in the notebook referencing saved output artifacts.

## [Phase 7] - Visualization & Reporting
- Enhanced the `visualize.py` module to plot learning curves (loss, accuracy).
- Added Confusion Matrix visualization post-evaluation.
- Created EDA routines to plot and save raw CIFAR-10 data samples.

## [Phase 6] - TensorBoard & Logging System
- Upgraded the manual print logging system to Python's built-in `logging` module.
- Integrated `torch.utils.tensorboard.SummaryWriter` into the `train.py` training loop.
- Captured training/validation scalars and exported the PyTorch computation graph to TensorBoard.

## [Phase 5] - Experiment Management & Hyperparameter Tuning
- Built an `experiment.py` runner capable of executing predefined configurations from `config.py`.
- Automated the loading of distinct hyperparameter sets, looping through model initialization, training, and evaluation for each.
- Added CSV exports for experiment tracking (`experiment_results.csv`).

## [Phase 4] - Evaluation & Model Analysis
- Developed `evaluate.py` to calculate comprehensive test metrics.
- Introduced Precision, Recall, and Macro-F1 score on top of base Accuracy.
- Ensured test and validation sets are processed with `torch.no_grad()` to strictly prevent memory leaks.

## [Phase 3] - Training Pipeline
- Solidified the `train.py` loop incorporating standard epoch iterations.
- Integrated `EarlyStopping` to prevent model overfitting.
- Added Learning Rate Scheduling (`ReduceLROnPlateau`).
- Enabled model checkpointing, saving the `best_model.pth` based on validation loss.

## [Phase 2] - Model Architecture
- Constructed the generic `PretrainedClassifier` wrapper inside `model.py`.
- Integrated `ResNet18`, `VGG16`, and `DenseNet121` from `torchvision.models`.
- Added support for multiple Transfer Learning strategies (`head_only`, `partial_finetune`, `full_finetune`).

## [Phase 1] - Data Pipeline
- Defined `data.py` loading the `CIFAR-10` dataset via `torchvision.datasets`.
- Correctly split Data into Train/Validation/Test, strictly avoiding data leakage.
- Implemented standard ImageNet Normalization and Data Augmentation transforms.
