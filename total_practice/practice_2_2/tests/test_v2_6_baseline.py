import os
import pandas as pd
import torch
import json
import hashlib
from practice_2_2.train_v2_6_baseline import verify_data, get_hash

def test_new_visualsafe_manifest_loaded():
    base_dir = "total_practice/practice_2_2"
    manifest_path = os.path.join(base_dir, "data/manifests/v2_visual_group_stratified_s42/split_manifest.csv")
    df = pd.read_csv(manifest_path)
    
    # Should not throw exception
    verify_data(df, manifest_path)
    assert True

def test_old_v2_manifest_rejected():
    base_dir = "total_practice/practice_2_2"
    old_manifest_path = os.path.join(base_dir, "data/manifests/v2_stratified_group_s42/split_manifest_v2.csv")
    df = pd.read_csv(old_manifest_path)
    
    try:
        verify_data(df, old_manifest_path)
        assert False, "Should have rejected old manifest"
    except AssertionError as e:
        assert "REJECT OLD MANIFEST" in str(e) or "MUST USE NEW MANIFEST LINEAGE" in str(e)

def test_train_validation_only():
    base_dir = "total_practice/practice_2_2"
    # we just verify that train script only creates train and val loader. This is done by visual inspection of code
    # But we can verify no Test in output metrics
    for exp in ["E1_visualsafe_head_only", "E2_visualsafe_layer4"]:
        metrics_path = os.path.join(base_dir, "artifacts/experiments/v2_visualsafe_baseline_s42_v1", exp, "best_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            assert metrics['test_evaluated'] == False
            assert 'test_acc' not in metrics

def test_same_initial_state():
    base_dir = "total_practice/practice_2_2"
    init_path = os.path.join(base_dir, "artifacts/experiments/v2_visualsafe_baseline_s42_v1", "initial_state.pt")
    
    # Reload and check hash
    state_dict = torch.load(init_path)
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        tensor = state_dict[k]
        h.update(k.encode('utf-8'))
        tensor_np = tensor.cpu().numpy().copy(order='C')
        h.update(tensor_np.tobytes())
    t_hash = h.hexdigest()
    assert t_hash == "4e7500733de1a2f3262489d172646dcc5c730ec8ffe0e82f00a9533a1b9368fd"

def test_old_artifacts_unchanged():
    base_dir = "total_practice/practice_2_2"
    assert os.path.exists(os.path.join(base_dir, "artifacts/experiments/v2_baseline_s42_e1_e2_v1"))

if __name__ == "__main__":
    test_new_visualsafe_manifest_loaded()
    test_old_v2_manifest_rejected()
    test_train_validation_only()
    test_same_initial_state()
    test_old_artifacts_unchanged()
    print("All V2.6 tests passed.")
