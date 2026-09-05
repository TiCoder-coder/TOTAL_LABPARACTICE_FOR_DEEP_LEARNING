"""Phase 44 — Exact-epoch refit wrapper for Stage B.

This wrapper implements Stage B from scratch using the canonical contracts:

  - fresh model (built from candidate config, no warm-start from Stage A)
  - fresh optimizer (AdamW with candidate-specific LR / WD)
  - fresh fold-local scaler bundle (already fit upstream; passed in)
  - train on OUTER_TRAIN only (one DataLoader)
  - exactly `best_epoch_inner` epochs
  - NO validation-based model selection
  - NO early stopping
  - NO FINAL_DEV semantics (no eval split override)
  - REFIT_FINAL = final-epoch weights

The wrapper deliberately avoids `TrainingEngine.train(final_refit_mode=True)`
because:

  - that flag also forces `eval_split_id = "FINAL_DEV"` for diagnostic eval,
    which is Phase 46 territory and would contaminate Phase 44.
  - `TrainingEngine.train` returns `best_validation_rmse_wh` and
    `best_sample_idx`/`best_y_true_wh`/`best_y_pred_wh` from validation, which
    do not exist for Stage B (no validation).
  - `TrainingEngine.persist_run_artifacts` always saves
    `best_validation_predictions.csv`, which is meaningless for Stage B.

The wrapper implements a minimal training loop and saves:
  - refit_final.pt (final-epoch state_dict)
  - training_history.csv (epoch, train_loss, epoch_seconds, is_best=False)
  - refit_run_status.json (official_epoch, refit_seed, best_epoch_inner)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from course_work.experiments.registry import ArtifactType, ExperimentRegistry
from course_work.training.engine import (
    HISTORY_COLUMNS,
    TrainingResult,
    build_model_from_run_config,
)
from course_work.training.losses import (
    build_training_criterion,
    validate_criterion_inputs,
)
from course_work.utils.artifacts import atomic_write_bytes, csv_text


@dataclass
class StageBResult:
    run_id: str
    refit_epoch: int
    train_loss_per_epoch: list[float]
    history: pd.DataFrame
    trainable_parameters: int
    refit_final_checkpoint_path: Path
    refit_status_path: Path
    refit_seed: int = 42
    parent_run_id: str | None = None
    fold_id: str = "RO?"
    candidate_id: str = "?"
    notes: str = ""


class RefitEngine:
    """Exact-epoch, no-validation, no-early-stopping refit wrapper.

    Does NOT use `TrainingEngine.final_refit_mode` (avoids FINAL_DEV
    contamination). Implements Phase 44 Stage B contract from scratch.
    """

    def __init__(self, registry: ExperimentRegistry) -> None:
        self.registry = registry

    def refit(
        self,
        *,
        run_id: str,
        train_loader: DataLoader,
        model: torch.nn.Module,
        device: torch.device,
        config: dict,
        best_epoch_inner: int,
        fold_id: str,
        candidate_id: str,
        refit_seed: int = 42,
        parent_run_id: str | None = None,
        notes: str = "",
        heartbeat_path: Path | None = None,
        rehearsal_synthetic: bool = False,
    ) -> StageBResult:
        """Train exactly `best_epoch_inner` epochs. Return REFIT_FINAL.

        No validation. No warm-start. No early stopping.

        When `rehearsal_synthetic=True`, this returns a deterministic
        synthetic StageBResult WITHOUT training. Blocked in official mode
        by `assert_context_invariants`.
        """
        if rehearsal_synthetic:
            from course_work.experiments.registry import RunStatus as _RS
            try:
                self.registry._transition(run_id, _RS.RUNNING.value)
            except Exception:
                pass
            try:
                self.registry._transition(
                    run_id, _RS.COMPLETED.value,
                    update={"synthetic": True,
                            "exact_match": True,
                            "epochs_completed": int(best_epoch_inner)},
                )
            except Exception:
                pass
            return StageBResult(
                run_id=run_id,
                refit_epoch=int(best_epoch_inner),
                train_loss_per_epoch=[0.5] * int(best_epoch_inner),
                history=pd.DataFrame({"epoch": list(range(1, int(best_epoch_inner)+1)),
                                       "train_rmse_wh": [80.0] * int(best_epoch_inner)}),
                trainable_parameters=1000,
                refit_final_checkpoint_path=Path("/tmp/synthetic.pt"),
                refit_status_path=Path("/tmp/synthetic_status.json"),
                refit_seed=refit_seed,
                parent_run_id=parent_run_id,
                fold_id=fold_id,
                candidate_id=candidate_id,
                notes=f"synthetic:{notes}",
            )
        record = self.registry.get_run(run_id)
        training_cfg = config["training"]
        max_epochs = int(best_epoch_inner)
        if max_epochs < 1:
            raise ValueError(
                f"best_epoch_inner must be >= 1 (got {max_epochs} for {run_id})"
            )
        learning_rate = float(training_cfg["learning_rate"])
        weight_decay = float(training_cfg["weight_decay"])
        clip_enabled = bool(training_cfg["gradient_clipping_enabled"])
        clip_norm = (
            float(training_cfg["gradient_clip_max_norm"])
            if training_cfg["gradient_clip_max_norm"] is not None
            else 0.0
        )

        # ---- HARD CONTRACT ASSERTIONS ----
        assert (
            training_cfg.get("early_stopping_enabled") is False
        ), f"Stage B config early_stopping_enabled must be False ({run_id})"

        torch.manual_seed(refit_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(refit_seed)

        model = model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        loss_fn = build_training_criterion(training_cfg)

        # Heartbeat for watchdog (best-effort, optional)
        hb = heartbeat_path or Path(
            __import__("os").environ.get("SWEEP_HEARTBEAT_PATH", "/tmp/sweep_heartbeat.txt")
        )

        def _hb(epoch: int, stage: str) -> None:
            try:
                hb.write_text(
                    f"epoch={epoch}\nstage={stage}\ntimestamp={time.time()}\nrun_id={run_id}\n",
                    encoding="utf-8",
                )
            except OSError:
                pass

        history_rows: list[dict] = []
        train_losses: list[float] = []

        _hb(0, "refit_started")
        print(
            f"[REFIT] Starting {max_epochs} epochs (run_id={run_id}, "
            f"fold={fold_id}, candidate={candidate_id})",
            flush=True,
        )

        for epoch in range(1, max_epochs + 1):
            model.train()
            epoch_loss = 0.0
            sample_count = 0
            batch_count = 0
            total_batches = len(train_loader)
            epoch_start = time.time()

            _hb(epoch, "epoch_started")
            for batch in train_loader:
                x = batch["x"].to(device)
                y = batch["y_model"].to(device)
                optimizer.zero_grad(set_to_none=True)
                predictions = model(x)
                validate_criterion_inputs(predictions, y)
                loss = loss_fn(predictions, y)
                loss.backward()

                # Non-finite gradient guard (mirrors TrainingEngine)
                has_nonfinite = False
                for p in model.parameters():
                    if p.grad is not None and not p.grad.isfinite().all():
                        has_nonfinite = True
                        break
                if has_nonfinite:
                    raise ValueError(
                        f"Non-finite gradient at refit epoch {epoch}, batch {batch_count} "
                        f"(run_id={run_id})"
                    )

                if clip_enabled:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        clip_norm,
                        error_if_nonfinite=False,
                    )

                optimizer.step()
                bsz = x.shape[0]
                epoch_loss += float(loss.item()) * bsz
                sample_count += bsz
                batch_count += 1

            train_loss = epoch_loss / max(sample_count, 1)
            train_losses.append(train_loss)
            epoch_seconds = time.time() - epoch_start
            _hb(epoch, "epoch_completed")
            history_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    # Stage B has no validation: placeholders for schema compat.
                    "train_rmse_wh": float("nan"),
                    "validation_rmse_wh": float("nan"),
                    "validation_mae_wh": float("nan"),
                    "validation_r2": float("nan"),
                    "learning_rate": learning_rate,
                    "epoch_seconds": epoch_seconds,
                    "is_best": False,
                }
            )
            print(
                f"  [REFIT] epoch {epoch}/{max_epochs} train_loss={train_loss:.4f}",
                flush=True,
            )

        # After exactly `max_epochs` epochs, snapshot REFIT_FINAL.
        refit_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history_df = pd.DataFrame(history_rows, columns=HISTORY_COLUMNS)

        # ---- Persist artifacts ----
        run_dir = self.registry.run_root / run_id
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        refit_final_path = ckpt_dir / "refit_final.pt"
        payload = {
            "model_state_dict": refit_state,
            "checkpoint_metadata": model.checkpoint_metadata()
            if hasattr(model, "checkpoint_metadata")
            else {},
            "official_epoch": max_epochs,
            "best_epoch_inner": max_epochs,
            "refit_seed": refit_seed,
            "fold_id": fold_id,
            "candidate_id": candidate_id,
            "parent_run_id": parent_run_id,
            "stage": "B",
        }
        torch.save(payload, refit_final_path)

        history_path = run_dir / "training_history.csv"
        atomic_write_bytes(
            history_path,
            csv_text(HISTORY_COLUMNS, history_df.to_dict(orient="records")).encode("utf-8"),
        )

        # Training log (human-readable)
        log_path = run_dir / "training.log"
        log_path.write_text(
            f"official_epoch={max_epochs}\n"
            f"best_epoch_inner={max_epochs}\n"
            f"stopped_reason=EXACT_EPOCHS\n"
            f"refit_seed={refit_seed}\n"
            f"fold_id={fold_id}\n"
            f"candidate_id={candidate_id}\n",
            encoding="utf-8",
        )

        refit_status_path = run_dir / "refit_status.json"
        refit_status = {
            "run_id": run_id,
            "stage": "B",
            "official_epoch": max_epochs,
            "best_epoch_inner": max_epochs,
            "stopped_reason": "EXACT_EPOCHS",
            "refit_seed": refit_seed,
            "fold_id": fold_id,
            "candidate_id": candidate_id,
            "parent_run_id": parent_run_id,
            "train_loss_final": float(train_losses[-1]) if train_losses else None,
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "early_stopping_enabled": False,
            "validation_used": False,
            "warm_start": False,
            "final_dev_semantics": False,
            "notes": notes,
        }
        atomic_write_bytes(refit_status_path, json.dumps(refit_status, indent=2).encode("utf-8"))

        # Register artifact in registry
        for art_type, art_path in (
            (ArtifactType.TRAIN_LOG.value, log_path),
            (ArtifactType.METRICS.value, refit_status_path),
        ):
            try:
                self.registry.register_artifact(run_id, art_type, art_path, required=True)
            except Exception:
                # Non-fatal: persistence may already have logged these.
                pass

        return StageBResult(
            run_id=run_id,
            refit_epoch=max_epochs,
            train_loss_per_epoch=train_losses,
            history=history_df,
            trainable_parameters=refit_status["trainable_parameters"],
            refit_final_checkpoint_path=refit_final_path,
            refit_status_path=refit_status_path,
            refit_seed=refit_seed,
            parent_run_id=parent_run_id,
            fold_id=fold_id,
            candidate_id=candidate_id,
            notes=notes,
        )