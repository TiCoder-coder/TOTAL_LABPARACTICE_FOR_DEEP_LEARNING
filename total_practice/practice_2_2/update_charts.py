import nbformat

nb_path = 'total_practice/practice_2_2/notebooks/04_canonical_report.ipynb'
nb = nbformat.read(nb_path, as_version=4)

cell1_source = '''
import matplotlib.image as mpimg

display(Markdown("### 🖼️ Dataset Examples per Class"))
display(Markdown("Hiển thị 5 ảnh ngẫu nhiên cho mỗi nhãn (class) từ tập Train, tương tự cách trình bày của Practice 2."))

classes = sorted(model_use['class_name'].unique())
n_classes = len(classes)
n_examples = 5

fig, axes = plt.subplots(n_classes, n_examples, figsize=(10, 1.8 * n_classes))

for i, cls in enumerate(classes):
    samples = model_use[(model_use['split'] == 'Train') & (model_use['class_name'] == cls)].sample(n_examples, random_state=42)
    
    for j, (idx, row) in enumerate(samples.iterrows()):
        ax = axes[i, j]
        img_path = data_root / row['relative_path']
        try:
            img = mpimg.imread(str(img_path))
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, 'Error', ha='center', va='center')
        ax.axis('off')
        
        if j == 0:
            ax.text(-0.15, 0.5, cls, ha='right', va='center', transform=ax.transAxes, fontsize=12)

plt.subplots_adjust(wspace=0.05, hspace=0.1)
fig.suptitle("Dataset Examples per Class", fontsize=16, y=0.92)
plt.show()
'''

cell2_source = '''
display(Markdown("### 🔍 Highest-confidence predictions"))
display(Markdown("Hiển thị trực quan các dự đoán đúng và sai có độ tự tin cao nhất của mô hình, giống với bố cục của Practice 2."))

predictions_path = final_root / "final_test_predictions.csv"
if predictions_path.exists():
    preds = pd.read_csv(predictions_path)
    
    # 1. Highest-confidence correct predictions
    # Filter where correct is True (string 'True' or bool True)
    correct_mask = preds['correct'] == True
    if correct_mask.dtype != bool:
        correct_mask = preds['correct'].astype(str).str.lower() == 'true'
        
    correct_preds = preds[correct_mask].nlargest(12, 'confidence')
    
    fig, axes = plt.subplots(2, 6, figsize=(15, 5))
    fig.suptitle("Top Confident Correct Predictions", fontsize=14)
    
    for ax, (_, row) in zip(axes.flatten(), correct_preds.iterrows()):
        img_path = data_root / row['relative_path']
        try:
            img = mpimg.imread(str(img_path))
            ax.imshow(img)
        except Exception:
            pass
        ax.axis('off')
        ax.set_title(f"True: {row['true_class']}\\nPred: {row['predicted_class']}\\nConf: {row['confidence']*100:.1f}%", 
                     color='#2ca02c', fontsize=9)
    plt.tight_layout()
    plt.show()
    
    # 2. Highest-confidence incorrect predictions
    incorrect_mask = ~correct_mask
    incorrect_preds = preds[incorrect_mask].nlargest(12, 'confidence')
    
    fig, axes = plt.subplots(2, 6, figsize=(15, 5))
    fig.suptitle("Top Confident Misclassified Predictions", fontsize=14)
    
    for ax, (_, row) in zip(axes.flatten(), incorrect_preds.iterrows()):
        img_path = data_root / row['relative_path']
        try:
            img = mpimg.imread(str(img_path))
            ax.imshow(img)
        except Exception:
            pass
        ax.axis('off')
        ax.set_title(f"True: {row['true_class']}\\nPred: {row['predicted_class']}\\nConf: {row['confidence']*100:.1f}%", 
                     color='#d62728', fontsize=9)
    plt.tight_layout()
    plt.show()
else:
    display(Markdown("*Không tìm thấy file final_test_predictions.csv*"))
'''

for cell in nb.cells:
    if "Representative Dataset Samples" in cell.source or "Dataset Examples per Class" in cell.source:
        cell.source = cell1_source.strip()
        print("Updated Cell 1")
    if "Visualizing Top Confusion Pairs" in cell.source or "Highest-confidence predictions" in cell.source:
        cell.source = cell2_source.strip()
        print("Updated Cell 2")

nbformat.write(nb, nb_path)
print("Saved.")
