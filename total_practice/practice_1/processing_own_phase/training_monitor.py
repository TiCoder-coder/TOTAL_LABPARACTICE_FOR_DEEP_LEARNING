"""Live epoch-level training curves for notebook workflows."""

from collections.abc import Mapping, Sequence
import math
from numbers import Real
import os
from pathlib import Path
import sys
from typing import Any, Optional, Union

import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

try:
    from IPython.display import display as ipython_display
except ImportError:  # pragma: no cover - IPython is optional outside notebooks.
    ipython_display = None


MetricValue = Union[int, float]


class TrainingMonitor:
    """Update loss and accuracy curves after each completed epoch."""

    def __init__(
        self,
        run_name: str,
        total_epochs: int,
        include_validation: bool,
        enabled: bool = True,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        if not isinstance(run_name, str) or not run_name.strip():
            raise ValueError("run_name must be a non-empty string")
        if isinstance(total_epochs, bool) or not isinstance(total_epochs, int):
            raise TypeError("total_epochs must be an integer")
        if total_epochs <= 0:
            raise ValueError("total_epochs must be positive")
        if not isinstance(include_validation, bool):
            raise TypeError("include_validation must be a boolean")
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a boolean")
        if save_path is not None and not isinstance(save_path, (str, Path)):
            raise TypeError("save_path must be a string, Path or None")
        resolved_save_path = Path(save_path) if save_path is not None else None
        if resolved_save_path is not None and (
            not resolved_save_path.name
            or resolved_save_path.suffix.lower() != ".png"
        ):
            raise ValueError("save_path must point to a .png file")

        self.run_name = run_name.strip()
        self.total_epochs = total_epochs
        self.include_validation = include_validation
        self.enabled = enabled
        self.save_path = resolved_save_path
        self._records: list[dict[str, MetricValue]] = []
        self._best_epoch: Optional[int] = None
        self._figure: Optional[Figure] = None
        self._loss_axis: Optional[Axes] = None
        self._accuracy_axis: Optional[Axes] = None
        self._display_handle: Any = None
        self._closed = False
        self._live_display = bool(
            enabled
            and ipython_display is not None
            and "ipykernel" in sys.modules
        )

    @property
    def history(self) -> tuple[dict[str, MetricValue], ...]:
        """Return defensive copies of the accepted epoch records."""
        return tuple(dict(record) for record in self._records)

    @property
    def figure(self) -> Optional[Figure]:
        """Return the monitor figure when rendering has started."""
        return self._figure

    @property
    def axes(self) -> tuple[Axes, ...]:
        """Return the loss and accuracy axes when available."""
        if self._loss_axis is None or self._accuracy_axis is None:
            return ()
        return (self._loss_axis, self._accuracy_axis)

    @property
    def closed(self) -> bool:
        """Report whether the monitor lifecycle has ended."""
        return self._closed

    def __enter__(self) -> "TrainingMonitor":
        if self._closed:
            raise RuntimeError("A closed TrainingMonitor cannot be reused")
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def update(
        self,
        epoch_record: Mapping[str, Any],
        best_epoch: Optional[int] = None,
    ) -> None:
        """Validate one epoch record and refresh the dashboard."""
        if not self.enabled:
            return
        if self._closed:
            raise RuntimeError("Cannot update a closed TrainingMonitor")
        if not isinstance(epoch_record, Mapping):
            raise TypeError("epoch_record must be a mapping")

        normalized_record = self._normalize_record(epoch_record)
        epoch = int(normalized_record["epoch"])
        expected_epoch = len(self._records) + 1

        if epoch != expected_epoch:
            raise ValueError(
                f"Expected epoch {expected_epoch}, received epoch {epoch}"
            )
        if epoch > self.total_epochs:
            raise ValueError("Epoch exceeds the configured total_epochs")

        if best_epoch is not None:
            if isinstance(best_epoch, bool) or not isinstance(best_epoch, int):
                raise TypeError("best_epoch must be an integer or None")
            if not 1 <= best_epoch <= epoch:
                raise ValueError("best_epoch must be within completed epochs")

        self._records.append(normalized_record)
        self._best_epoch = best_epoch
        self._render()

    def restore(
        self,
        history: Sequence[Mapping[str, Any]],
        best_epoch: Optional[int] = None,
    ) -> None:
        """Restore completed epoch history and render it exactly once."""
        if not self.enabled:
            return
        if self._closed:
            raise RuntimeError("Cannot restore a closed TrainingMonitor")
        if self._records:
            raise RuntimeError("TrainingMonitor history is already initialized")
        if not isinstance(history, Sequence) or isinstance(
            history,
            (str, bytes, bytearray),
        ):
            raise TypeError("history must be a sequence of metric mappings")

        normalized_records = []
        for expected_epoch, epoch_record in enumerate(history, start=1):
            if not isinstance(epoch_record, Mapping):
                raise TypeError("Every history record must be a mapping")
            normalized_record = self._normalize_record(epoch_record)
            epoch = int(normalized_record["epoch"])
            if epoch != expected_epoch:
                raise ValueError(
                    f"Expected epoch {expected_epoch}, received epoch {epoch}"
                )
            if epoch > self.total_epochs:
                raise ValueError("Epoch exceeds the configured total_epochs")
            normalized_records.append(normalized_record)

        completed_epochs = len(normalized_records)
        if best_epoch is not None:
            if isinstance(best_epoch, bool) or not isinstance(best_epoch, int):
                raise TypeError("best_epoch must be an integer or None")
            if not 1 <= best_epoch <= completed_epochs:
                raise ValueError("best_epoch must be within completed epochs")

        self._records = normalized_records
        self._best_epoch = best_epoch
        if self._records:
            self._render()

    def close(self) -> None:
        """Publish the final dashboard state and release figure resources."""
        if self._closed:
            return
        if self.enabled and self._figure is not None:
            if self._live_display and self._display_handle is not None:
                self._display_handle.update(self._figure)
            if self.save_path is not None:
                self._save_figure()
            plt.close(self._figure)
        self._closed = True

    def _normalize_record(
        self,
        epoch_record: Mapping[str, Any],
    ) -> dict[str, MetricValue]:
        required_fields = ["epoch", "train_loss", "train_accuracy"]
        if self.include_validation:
            required_fields.extend(("validation_loss", "validation_accuracy"))

        missing_fields = [
            field for field in required_fields if field not in epoch_record
        ]
        if missing_fields:
            raise KeyError(
                "Missing monitor metrics: " + ", ".join(missing_fields)
            )

        epoch = epoch_record["epoch"]
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise TypeError("epoch must be an integer")

        normalized: dict[str, MetricValue] = {"epoch": epoch}
        metric_fields = required_fields[1:]
        optional_fields = ("learning_rate", "elapsed_seconds")

        for field in metric_fields:
            normalized[field] = self._finite_number(field, epoch_record[field])
        for field in optional_fields:
            if field in epoch_record:
                normalized[field] = self._finite_number(
                    field,
                    epoch_record[field],
                )

        for field in ("train_loss", "validation_loss"):
            if field in normalized and normalized[field] < 0:
                raise ValueError(f"{field} must be non-negative")
        for field in ("train_accuracy", "validation_accuracy"):
            if field in normalized and not 0.0 <= normalized[field] <= 1.0:
                raise ValueError(f"{field} must be in the [0, 1] range")
        for field in ("learning_rate", "elapsed_seconds"):
            if field in normalized and normalized[field] < 0:
                raise ValueError(f"{field} must be non-negative")

        return normalized

    @staticmethod
    def _finite_number(name: str, value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise FloatingPointError(f"{name} must be finite")
        return numeric_value

    def _render(self) -> None:
        if self._figure is None:
            figure, axes = plt.subplots(1, 2, figsize=(14, 5))
            self._figure = figure
            self._loss_axis = axes[0]
            self._accuracy_axis = axes[1]

        if self._loss_axis is None or self._accuracy_axis is None:
            raise RuntimeError("Training monitor axes were not initialized")

        epochs = [int(record["epoch"]) for record in self._records]
        train_losses = [float(record["train_loss"]) for record in self._records]
        train_accuracies = [
            float(record["train_accuracy"]) * 100
            for record in self._records
        ]

        self._loss_axis.clear()
        self._accuracy_axis.clear()
        self._loss_axis.plot(
            epochs,
            train_losses,
            marker="o",
            linewidth=2,
            label="Train",
        )
        self._accuracy_axis.plot(
            epochs,
            train_accuracies,
            marker="o",
            linewidth=2,
            label="Train",
        )

        if self.include_validation:
            validation_losses = [
                float(record["validation_loss"])
                for record in self._records
            ]
            validation_accuracies = [
                float(record["validation_accuracy"]) * 100
                for record in self._records
            ]
            self._loss_axis.plot(
                epochs,
                validation_losses,
                marker="o",
                linewidth=2,
                label="Validation",
            )
            self._accuracy_axis.plot(
                epochs,
                validation_accuracies,
                marker="o",
                linewidth=2,
                label="Validation",
            )

        if self._best_epoch is not None:
            for axis in self.axes:
                axis.axvline(
                    self._best_epoch,
                    color="gray",
                    linestyle="--",
                    linewidth=1,
                    label=f"Best epoch: {self._best_epoch}",
                )

        self._loss_axis.set_title("Loss by Epoch")
        self._loss_axis.set_xlabel("Epoch")
        self._loss_axis.set_ylabel("Cross-Entropy Loss")
        self._loss_axis.xaxis.set_major_locator(
            MaxNLocator(integer=True, nbins=min(self.total_epochs, 10))
        )
        self._loss_axis.set_xlim(1, max(2, self.total_epochs))
        self._loss_axis.grid(True, alpha=0.3)
        self._loss_axis.legend()

        self._accuracy_axis.set_title("Accuracy by Epoch")
        self._accuracy_axis.set_xlabel("Epoch")
        self._accuracy_axis.set_ylabel("Accuracy (%)")
        self._accuracy_axis.xaxis.set_major_locator(
            MaxNLocator(integer=True, nbins=min(self.total_epochs, 10))
        )
        self._accuracy_axis.set_xlim(1, max(2, self.total_epochs))
        self._accuracy_axis.set_ylim(0.0, 100.0)
        self._accuracy_axis.grid(True, alpha=0.3)
        self._accuracy_axis.legend()

        current_epoch = epochs[-1]
        best_epoch_text = (
            f" | Best epoch: {self._best_epoch}"
            if self._best_epoch is not None
            else ""
        )
        self._figure.suptitle(
            f"{self.run_name} | Epoch {current_epoch}/{self.total_epochs}"
            f"{best_epoch_text}",
            fontsize=14,
        )
        self._figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

        if self._live_display:
            if self._display_handle is None:
                self._display_handle = ipython_display(
                    self._figure,
                    display_id=True,
                )
            else:
                self._display_handle.update(self._figure)
        else:
            self._figure.canvas.draw()

    def _save_figure(self) -> None:
        if self._figure is None or self.save_path is None:
            return
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.save_path.with_suffix(".png.tmp")
        try:
            self._figure.savefig(
                temporary_path,
                format="png",
                dpi=150,
                bbox_inches="tight",
            )
            os.replace(temporary_path, self.save_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
