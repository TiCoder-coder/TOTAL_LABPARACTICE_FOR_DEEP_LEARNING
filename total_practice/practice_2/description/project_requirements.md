# Practice 2 -- Hands-on Practice with Pre-trained Neural Network Architectures

## Objective

Build an image classification model using a pre-trained neural network
in PyTorch. Learn how to load, customize, train, evaluate, and monitor a
pre-trained model.

## Requirements

### 1. Environment Setup

-   Install PyTorch and Torchvision.
-   Verify the environment is ready for training.

### 2. Load a Pre-trained Model

-   Use a model from `torchvision.models` (e.g. ResNet, VGG, DenseNet).
-   Load ImageNet pre-trained weights.

### 3. Explore the Model

-   Print the model architecture.
-   Identify important layers and trainable parameters.

### 4. Adapt the Model

-   Replace the final classification layer to match the number of
    classes.
-   Choose one of the following:
    -   Transfer Learning (freeze feature extraction layers).
    -   Fine-tuning (unfreeze selected layers).

### 5. Prepare the Dataset

-   Load the dataset.
-   Apply preprocessing using `torchvision.transforms`.
-   Create `DataLoader` objects for train and test datasets.

### 6. Train the Model

-   Define:
    -   Loss Function
    -   Optimizer
-   Implement the training loop.
-   Adjust hyperparameters such as:
    -   Learning Rate
    -   Batch Size
    -   Number of Epochs

### 7. Monitor Training

-   Integrate TensorBoard.
-   Track:
    -   Training Loss
    -   Validation Loss
    -   Accuracy

### 8. Evaluate the Model

-   Test the trained model on the test dataset.
-   Report final evaluation metrics.

## Expected Deliverables

-   Source code.
-   Configured pre-trained model.
-   Training and evaluation pipeline.
-   TensorBoard logs.
-   Final evaluation results.
