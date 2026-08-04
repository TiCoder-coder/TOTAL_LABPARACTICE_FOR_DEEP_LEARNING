import os
import torch
import torchvision.models as models
import torch.nn as nn
import hashlib
import json

def get_deterministic_tensor_hash(state_dict):
    """
    Computes a deterministic hash over all tensors in a state_dict.
    Sorts keys to ensure order independence.
    """
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        tensor = state_dict[k]
        # Hash the key name
        h.update(k.encode('utf-8'))
        # Hash the tensor data (converting to contiguous numpy array first)
        tensor_np = tensor.cpu().numpy().copy(order='C')
        h.update(tensor_np.tobytes())
    return h.hexdigest()

def main():
    torch.manual_seed(42)
    
    # Load ImageNet pretrained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Replace the classification head for 10 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs, 10)
    )
    
    # Create artifacts directory if not exists
    out_dir = "total_practice/practice_2_2/artifacts/experiments/v2_visualsafe_baseline_s42_v1"
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, "initial_state.pt")
    state_dict = model.state_dict()
    
    torch.save(state_dict, out_path)
    
    # Compute tensor deterministic hash
    tensor_hash = get_deterministic_tensor_hash(state_dict)
    
    # Compute file hash
    file_hash = hashlib.sha256()
    with open(out_path, 'rb') as f:
        file_hash.update(f.read())
    file_hash = file_hash.hexdigest()
    
    metadata = {
        "seed": 42,
        "model": "ResNet18",
        "tensor_hash": tensor_hash,
        "file_hash": file_hash
    }
    
    with open(os.path.join(out_dir, "initial_state_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("Initial state created and saved.")
    print(f"Tensor hash: {tensor_hash}")
    print(f"File hash: {file_hash}")

if __name__ == "__main__":
    main()
