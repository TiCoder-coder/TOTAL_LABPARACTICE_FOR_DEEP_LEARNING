import nbformat
nb = nbformat.read('total_practice/practice_2_2/notebooks/04_canonical_report.ipynb', as_version=4)

new_code = '''
import matplotlib.image as mpimg

display(Markdown("### 🔍 Visualizing Top Confusion Pairs"))
display(Markdown("Hiển thị trực quan các hình ảnh bị nhầm lẫn nhiều nhất (Top 3 cặp), ưu tiên những dự đoán mô hình có độ tự tin cao nhất (High Confidence Mistakes)."))

misclassified_path = final_root / "misclassified_samples.csv"
if misclassified_path.exists():
    misclassified = pd.read_csv(misclassified_path)
    top_pairs = final_summary.get("top_confusion_pairs", [])[:3]

    if not top_pairs:
        display(Markdown("*Không có cặp nhầm lẫn nào.*"))
    else:
        for pair in top_pairs:
            true_c = pair['true_class']
            pred_c = pair['predicted_class']
            count = pair['count']
            
            display(Markdown(f"#### Thực tế: `{true_c}` ➔ Mô hình đoán: `{pred_c}` ({count} mẫu)"))
            
            # Lấy các mẫu cho cặp này
            samples = misclassified[(misclassified['true_class'] == true_c) & (misclassified['predicted_class'] == pred_c)]
            
            # Chọn tối đa 5 mẫu có confidence sai cao nhất
            samples = samples.sort_values(by='confidence', ascending=False).head(5)
            
            n_samples = len(samples)
            if n_samples == 0:
                continue
                
            fig, axes = plt.subplots(1, max(n_samples, 2), figsize=(3 * max(n_samples, 2), 3))
            
            # Đảm bảo axes luôn là list/array
            if not isinstance(axes, (list, tuple, type(plt.np.ndarray([])))):
                axes = [axes]
                
            for i, ax in enumerate(axes):
                if i < n_samples:
                    row = samples.iloc[i]
                    img_path = data_root / row['relative_path']
                    try:
                        img = mpimg.imread(str(img_path))
                        ax.imshow(img)
                        ax.set_title(f"Conf: {row['confidence']:.2f}", fontsize=10)
                    except Exception as e:
                        ax.text(0.5, 0.5, 'Image Error', ha='center', va='center')
                ax.axis('off')
                
            plt.tight_layout()
            plt.show()
else:
    display(Markdown("*File misclassified_samples.csv không tồn tại.*"))
'''

# Check if cell already exists to avoid duplicates
if "Visualizing Top Confusion Pairs" not in nb.cells[-1].source:
    new_cell = nbformat.v4.new_code_cell(new_code.strip())
    nb.cells.append(new_cell)
    nbformat.write(nb, 'total_practice/practice_2_2/notebooks/04_canonical_report.ipynb')
    print("Added image visualization cell.")
else:
    print("Image cell already exists.")
