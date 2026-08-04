import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
import hashlib

def _hash_state_dict(state_dict):
    """Tính hash SHA256 deterministic của toàn bộ tham số model."""
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        tensor = state_dict[k]
        h.update(k.encode('utf-8'))
        h.update(tensor.cpu().numpy().tobytes())
    return h.hexdigest()

def get_or_create_initial_state_dict(save_dir="total_practice/practice_2_2/artifacts/experiments/v2_baseline_s42_e1_e2_v1") -> str:
    """Khởi tạo một state_dict duy nhất của ResNet18 + Head và lưu lại."""
    save_path = Path(save_dir) / "initial_state_dict.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if save_path.exists():
        return str(save_path)
        
    torch.manual_seed(42) # Deterministic for creation
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Thay thế head
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs, 10)
    )
    
    state_dict = model.state_dict()
    torch.save(state_dict, save_path)
    return str(save_path)

def build_v2_model(strategy="E1", state_dict_path=None):
    """
    Tạo model, load state_dict khởi tạo chung, và áp dụng freeze policy.
    E1: Freeze toàn bộ backbone, chỉ train classifier Linear.
    E2: Freeze stem->layer3. Train layer4 và classifier Linear.
    """
    model = models.resnet18(weights=None) # Start from scratch
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs, 10)
    )
    
    # Load chung 1 state dict
    if state_dict_path is None:
        state_dict_path = get_or_create_initial_state_dict()
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    
    initial_hash = _hash_state_dict(model.state_dict())
    
    # Freeze logic
    if strategy == "E1":
        # Freeze all
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze classifier Linear
        for param in model.fc[1].parameters():
            param.requires_grad = True
            
    elif strategy in ["E2", "E3"]:
        # Freeze all
        for param in model.parameters():
            param.requires_grad = False
            
        # Unfreeze layer4
        for param in model.layer4.parameters():
            param.requires_grad = True
            
        # Unfreeze classifier Linear
        for param in model.fc[1].parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
        
    return model, initial_hash

def get_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_bn_eval_for_frozen_layers(model, strategy):
    """Đảm bảo các BN layer bị đóng băng không thay đổi running stats trong lúc train."""
    if strategy == "E1":
        # Toàn bộ backbone frozen, nên tất cả BN trừ head (head ko có BN) phải eval mode
        def set_bn_eval(m):
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
        model.apply(set_bn_eval)
    elif strategy in ["E2", "E3"]:
        # Freeze stem, layer1, layer2, layer3. Layer4 train bình thường.
        modules_to_eval = [model.bn1, model.layer1, model.layer2, model.layer3]
        for mod in modules_to_eval:
            for m in mod.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
