"""Model architecture for Practice 2 (Pre-trained Neural Networks)."""

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torchvision import models

from configs import NUM_CLASSES


class PretrainedClassifier(nn.Module):
    """A wrapper for pre-trained models from torchvision."""
    
    def __init__(
        self,
        model_name: str = "resnet18",
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.0,
        hidden_layers: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.model_name = model_name.lower()
        self.num_classes = num_classes
        self.dropout = float(dropout)
        self.hidden_layers = tuple(int(size) for size in (hidden_layers or ()))
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if any(size <= 0 for size in self.hidden_layers):
            raise ValueError("hidden layer sizes must be positive")
        
        self.network, self.classifier_name = self._build_model()
        self._initialize_classifier()
        
    def _build_model(self) -> Tuple[nn.Module, str]:
        """Build the requested model and replace its classifier head."""
        if self.model_name == "resnet18":
            network = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            in_features = network.fc.in_features
            network.fc = self._make_classifier(in_features)
            classifier_name = "fc"
            
        elif self.model_name == "vgg16":
            network = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            in_features = network.classifier[6].in_features
            network.classifier[6] = self._make_classifier(in_features)
            classifier_name = "classifier.6"
            
        elif self.model_name == "densenet121":
            network = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            in_features = network.classifier.in_features
            network.classifier = self._make_classifier(in_features)
            classifier_name = "classifier"
            
        elif self.model_name == "mobilenet_v3_small":
            network = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            in_features = network.classifier[3].in_features
            network.classifier[3] = self._make_classifier(in_features)
            classifier_name = "classifier.3"
            
        else:
            raise ValueError(f"Model {self.model_name} not supported.")
            
        return network, classifier_name

    def _make_classifier(self, in_features: int) -> nn.Module:
        if not self.hidden_layers:
            linear = nn.Linear(in_features, self.num_classes)
            if self.dropout > 0:
                return nn.Sequential(nn.Dropout(self.dropout), linear)
            return linear

        layers = []
        previous = in_features
        for width in self.hidden_layers:
            layers.extend([nn.Linear(previous, width), nn.ReLU(inplace=True)])
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            previous = width
        layers.append(nn.Linear(previous, self.num_classes))
        return nn.Sequential(*layers)

    @staticmethod
    def _classifier_linear(module: nn.Module) -> nn.Linear:
        if isinstance(module, nn.Linear):
            return module
        linear_layers = [layer for layer in module.modules() if isinstance(layer, nn.Linear)]
        if not linear_layers:
            raise RuntimeError("Classifier head contains no Linear layer")
        return linear_layers[-1]
        
    def _initialize_classifier(self):
        """Initialize the newly replaced classifier head properly."""
        if self.model_name == "resnet18":
            layer = self.network.fc
        elif self.model_name == "vgg16":
            layer = self.network.classifier[6]
        elif self.model_name == "densenet121":
            layer = self.network.classifier
        elif self.model_name == "mobilenet_v3_small":
            layer = self.network.classifier[3]
        else:
            return

        for linear in (item for item in layer.modules() if isinstance(item, nn.Linear)):
            nn.init.xavier_normal_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

    def set_training_mode(self, mode: str = "head_only"):
        """Configure which parameters require gradients based on the training mode.
        
        Modes:
            - head_only: Freeze everything except the final classification layer.
            - partial_finetune: Open the last blocks and the classifier.
            - last_block_finetune: Open ResNet18 layer4.1 and the classifier.
            - full_finetune: Train the entire network.
        """
        # First freeze all parameters
        for param in self.network.parameters():
            param.requires_grad = False
            
        if mode == "head_only":
            self._unfreeze_classifier()
            
        elif mode == "partial_finetune":
            self._unfreeze_classifier()
            if self.model_name == "resnet18":
                for param in self.network.layer4.parameters():
                    param.requires_grad = True
            elif self.model_name == "vgg16":
                for i in range(24, len(self.network.features)):
                    for param in self.network.features[i].parameters():
                        param.requires_grad = True
            elif self.model_name == "densenet121":
                for param in self.network.features.denseblock4.parameters():
                    param.requires_grad = True
            elif self.model_name == "mobilenet_v3_small":
                # Unfreeze from the last bottleneck block onwards (features[12:])
                for i in range(12, len(self.network.features)):
                    for param in self.network.features[i].parameters():
                        param.requires_grad = True

        elif mode == "last_block_finetune":
            if self.model_name != "resnet18":
                raise ValueError(
                    "last_block_finetune is defined only for ResNet18"
                )
            self._unfreeze_classifier()
            for param in self.network.layer4[1].parameters():
                param.requires_grad = True

        elif mode == "full_finetune":
            for param in self.network.parameters():
                param.requires_grad = True
                
        else:
            raise ValueError(f"Unknown training mode: {mode}")

        self.training_mode = mode

    def _unfreeze_classifier(self):
        """Helper to unfreeze just the final classification layer."""
        if self.model_name == "resnet18":
            for param in self.network.fc.parameters():
                param.requires_grad = True
        elif self.model_name == "vgg16":
            for param in self.network.classifier.parameters():
                param.requires_grad = True
        elif self.model_name == "densenet121":
            for param in self.network.classifier.parameters():
                param.requires_grad = True
        elif self.model_name == "mobilenet_v3_small":
            for param in self.network.classifier.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)
        
    def summary(self, input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224)) -> Dict:
        """Provide a summary of the model, including input/output shapes and parameter counts."""
        device = next(self.parameters()).device
        dummy_input = torch.randn(*input_shape, device=device)
        
        with torch.no_grad():
            output = self(dummy_input)
            
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            "model_name": self.model_name,
            "input_shape": input_shape,
            "output_shape": tuple(output.shape),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params
        }


def build_model(
    model_name: str,
    training_mode: str = "head_only",
    num_classes: int = NUM_CLASSES,
    dropout: float = 0.0,
    hidden_layers: Optional[Sequence[int]] = None,
) -> PretrainedClassifier:
    """Factory function to build a pre-trained classifier with a specific mode."""
    model = PretrainedClassifier(
        model_name=model_name,
        num_classes=num_classes,
        dropout=dropout,
        hidden_layers=hidden_layers,
    )
    model.set_training_mode(training_mode)
    return model


def sanity_check_model(model: PretrainedClassifier, device: torch.device, batch_size: int = 4) -> Dict:
    """Run quick sanity checks on the model.

    Verifies:
        1. Forward pass shape [B, num_classes].
        2. Backward pass populates gradients only for requires_grad=True parameters.
    """
    model = model.to(device)
    model.train()
    
    # Use standard ImageNet size
    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
    dummy_labels = torch.randint(0, model.num_classes, (batch_size,), device=device)

    logits = model(dummy_input)
    assert logits.shape == (batch_size, model.num_classes), f"Expected shape {(batch_size, model.num_classes)}, got {logits.shape}"

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    loss = criterion(logits, dummy_labels)
    assert torch.isfinite(loss).item(), "Loss is not finite"

    optimizer.zero_grad()
    loss.backward()
    
    # Check that at least one parameter has gradients
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    optimizer.step()
    
    summary_info = model.summary((batch_size, 3, 224, 224))

    return {
        "forward_shape": tuple(logits.shape),
        "loss_value": float(loss.item()),
        "gradients_populated": has_grad,
        "summary": summary_info
    }
