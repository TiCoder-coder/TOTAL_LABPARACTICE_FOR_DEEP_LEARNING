import json
from pathlib import Path

import numpy as np
import pytest

from processing_own_phase.eda import analyze_cifar10_training_subset, render_cifar10_eda


CLASS_NAMES = tuple(f"class_{index}" for index in range(10))


def _pool() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    training = rng.integers(0, 128, size=(30, 32, 32, 3), dtype=np.uint8)
    held_out = np.full((10, 32, 32, 3), 255, dtype=np.uint8)
    images = np.concatenate([training, held_out])
    labels = np.tile(np.arange(10, dtype=np.int64), 4)
    return images, labels, np.arange(30)


def test_analysis_reads_only_explicit_training_indices() -> None:
    images, labels, indices = _pool()
    analysis = analyze_cifar10_training_subset(
        images, labels, indices, CLASS_NAMES,
        samples_per_class=1, projection_samples_per_class=1,
    )
    assert analysis["report"]["scope"] == "training_subset_only"
    assert analysis["report"]["schema"]["sample_count"] == 30
    assert analysis["images"].max() < 255
    assert analysis["report"]["channel_statistics"]["Red"]["mean"] < .6
    assert analysis["report"]["quality"]["invalid_label_count"] == 0
    assert analysis["report"]["quality"]["missing_class_count"] == 0


def test_analysis_is_deterministic() -> None:
    images, labels, indices = _pool()
    first = analyze_cifar10_training_subset(images, labels, indices, CLASS_NAMES, samples_per_class=1, projection_samples_per_class=1)
    second = analyze_cifar10_training_subset(images, labels, indices, CLASS_NAMES, samples_per_class=1, projection_samples_per_class=1)
    assert first["sample_indices"] == second["sample_indices"]
    assert first["projection_indices"] == second["projection_indices"]


def test_invalid_contract_fails_closed() -> None:
    images, labels, indices = _pool()
    with pytest.raises(ValueError, match="unique"):
        analyze_cifar10_training_subset(images, labels, [0, 0], CLASS_NAMES)
    with pytest.raises(ValueError, match="uint8"):
        analyze_cifar10_training_subset(images.astype(np.float32), labels, indices, CLASS_NAMES)


def test_renderer_writes_only_requested_eda_figures(tmp_path: Path) -> None:
    images, labels, indices = _pool()
    analysis = analyze_cifar10_training_subset(images, labels, indices, CLASS_NAMES, samples_per_class=1, projection_samples_per_class=1)
    paths = render_cifar10_eda(analysis, CLASS_NAMES, tmp_path, include_projection=False)
    assert set(paths) == {"rgb_brightness_contrast", "class_means", "brightness_extremes"}
    assert all(path.is_file() and path.parent == tmp_path for path in paths.values())


def test_notebook_uses_eda_helper_without_training() -> None:
    notebook = Path(__file__).parents[1] / "notebooks" / "practice_2_presentation.ipynb"
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"] if cell["cell_type"] == "code")
    assert "analyze_cifar10_training_subset" in source
    assert "render_practice1_style_cifar10_eda" in source
    assert "EDA_SUMMARY[\"scope\"] == \"training_subset_only\"" in source
    assert "trainer.train(" not in source


def test_main_notebook_eda_headings_are_named_without_step_numbers() -> None:
    notebook = Path(__file__).parents[1] / "notebooks" / "practice_2_presentation.ipynb"
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    phase4_headings = []
    in_phase4 = False
    for cell in payload["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        source = "".join(cell.get("source", []))
        if source.startswith("## Phase 4"):
            in_phase4 = True
            continue
        if source.startswith("## Phase 5"):
            break
        if in_phase4 and source.startswith("### "):
            phase4_headings.append(source.splitlines()[0].removeprefix("### "))

    assert phase4_headings == [
        "Dataset Boundary and EDA Contract",
        "Exact Class Distribution",
        "RGB Pixel-Intensity Distribution and Quantiles",
        "Image Brightness and Contrast by Class",
        "Data Quality and Exact-Duplicate Audit",
        "RGB Spatial-Feature Correlation",
        "Raw PCA in Three Dimensions",
        "PCA Explained Variance",
        "RGB Principal-Component Images",
        "Standardized PCA and t-SNE",
        "Representative Samples and Class Means",
        "Statistical Extremes",
        "EDA Conclusions",
    ]
