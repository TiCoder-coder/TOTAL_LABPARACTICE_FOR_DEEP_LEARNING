import nbformat

nb_path = 'total_practice/practice_2_2/notebooks/04_canonical_report.ipynb'
nb = nbformat.read(nb_path, as_version=4)

cell_curves = '''
display(Markdown("### 📈 Training Curves: Validation Accuracy & Loss"))
display(Markdown("Biểu đồ so sánh quá trình huấn luyện (Learning Curves) giữa các thử nghiệm E1 và E2."))

histories_path = final_root.parent / "training_histories.json"
if histories_path.exists():
    import json
    with open(histories_path, 'r') as f:
        histories = json.load(f)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    colors = {'E1_head_only': vu.BLUE, 'E2_partial_finetune': vu.RED}
    
    for exp_id, hist in histories.items():
        epochs = hist['epoch']
        color = colors.get(exp_id, vu.GRAY)
        
        # Validation Accuracy
        ax1.plot(epochs, hist['val_acc'], label=f"{exp_id} (Val Acc)", color=color, linewidth=2, marker='o')
        
        # Training and Validation Loss
        ax2.plot(epochs, hist['train_loss'], label=f"{exp_id} (Train Loss)", color=color, linestyle='--', alpha=0.6)
        ax2.plot(epochs, hist['val_loss'], label=f"{exp_id} (Val Loss)", color=color, linewidth=2, marker='o')
        
        # Highlight best epoch
        best_ep = hist.get('best_epoch')
        if best_ep and best_ep in epochs:
            best_idx = epochs.index(best_ep)
            ax1.scatter([best_ep], [hist['val_acc'][best_idx]], s=150, color=vu.GREEN, zorder=5, edgecolors='white', linewidths=2)
            
    ax1.set_title("Validation Accuracy Over Epochs", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Epoch", fontweight='bold')
    ax1.set_ylabel("Accuracy", fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()
    
    ax2.set_title("Training & Validation Loss Over Epochs", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Epoch", fontweight='bold')
    ax2.set_ylabel("Loss", fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
else:
    display(Markdown("*Không tìm thấy file training_histories.json*"))
'''

has_curves = any("Training Curves: Validation Accuracy" in cell.source for cell in nb.cells)

if not has_curves:
    # Insert after the experiment comparison table
    idx_curves = -1
    for i, cell in enumerate(nb.cells):
        if 'selection["selected_experiment_id"]' in cell.source:
            idx_curves = i + 1
            # keep searching to find the last cell that mentions this, or just the first
            break
    if idx_curves != -1:
        nb.cells.insert(idx_curves, nbformat.v4.new_code_cell(cell_curves.strip()))
        print("Added Training Curves cell.")
        nbformat.write(nb, nb_path)
        print("Notebook updated.")
else:
    print("Curves cell already exists.")
