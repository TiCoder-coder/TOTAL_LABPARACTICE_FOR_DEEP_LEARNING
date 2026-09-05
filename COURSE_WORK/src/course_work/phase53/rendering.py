"""Phase 53 — pure rendering module.

This module renders per-head heatmap panels and grids from pre-loaded
Phase 52 raw dense attention arrays. It does NOT load model checkpoints,
does NOT extract new attention, and does NOT perform Test inference.

Rendering contract (frozen in `render_config.py`):

* matrix_orientation = QUERY_ROWS_SOURCE_COLUMNS
* origin = ROW0_TOP
* no interpolation (nearest)
* sequential perceptually uniform colormap (viridis)
* colorbar label = "Attention weight"
* Mode A: vmin=0, vmax=1
* Mode B: vmin=0, vmax=case_wide_max
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")  # no display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .sources import (
    ASPECT,
    CADENCE_MINUTES,
    COLORMAP,
    COLORBAR_LABEL,
    DPI,
    LOOKBACK,
    MODE_A_NAME,
    MODE_A_VMAX,
    MODE_A_VMIN,
    MODE_B_NAME,
    MODE_B_VMIN,
    NUM_HEADS,
    NUM_LAYERS,
    SEEDS,
    TRANSPOSE,
)


def _format_lag_label(pos: int) -> str:
    lag_min = (LOOKBACK - pos) * CADENCE_MINUTES
    if lag_min == 0:
        return f"t{lag_min}m"
    return f"t-{lag_min}m"


def _overlay_xticks(ax, positions: Iterable[int]) -> None:
    """Overlay phase53 lag-aware x-tick labels on heatmap axes."""
    pos_list = list(positions)
    if not pos_list:
        return
    ax.set_xticks(pos_list)
    ax.set_xticklabels([_format_lag_label(p) for p in pos_list], fontsize=8.5, rotation=0)
    ax.tick_params(axis="x", pad=4, labelsize=8.5)


def _overlay_yticks(ax, positions: Iterable[int]) -> None:
    pos_list = list(positions)
    if not pos_list:
        return
    ax.set_yticks(pos_list)
    ax.set_yticklabels([_format_lag_label(p) for p in pos_list], fontsize=8.5)
    ax.tick_params(axis="y", pad=4, labelsize=8.5)


def _apply_axis_labels(ax) -> None:
    ax.set_xlabel("Source / key historical position", fontsize=10, labelpad=8)
    ax.set_ylabel("Query historical position", fontsize=10, labelpad=8)


def render_mode_b_panel(
    M: np.ndarray,
    vmax: float,
    ax,
    title: str = "",
) -> None:
    """Render a single heatmap panel with Mode B case-shared scale.

    M: [L, L] raw attention matrix (M[q,s], NO transpose)
    vmax: case-shared vmax
    """
    assert M.shape == (LOOKBACK, LOOKBACK), f"expected [L,L] got {M.shape}"
    img = ax.imshow(
        M,
        cmap=COLORMAP,
        vmin=MODE_B_VMIN,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
        origin="upper",
    )
    if title:
        ax.set_title(title, fontsize=9)
    _apply_axis_labels(ax)
    # Lag-aware ticks every 12 positions (6 ticks per axis: 0,12,24,36,48,60)
    positions = list(range(0, LOOKBACK, 12))
    _overlay_xticks(ax, positions)
    _overlay_yticks(ax, positions)


def render_mode_a_panel(
    M: np.ndarray,
    ax,
    title: str = "",
) -> None:
    """Render a single heatmap panel with Mode A FIXED_PROBABILITY [0,1] scale."""
    assert M.shape == (LOOKBACK, LOOKBACK)
    img = ax.imshow(
        M,
        cmap=COLORMAP,
        vmin=MODE_A_VMIN,
        vmax=MODE_A_VMAX,
        interpolation="nearest",
        aspect="equal",
        origin="upper",
    )
    if title:
        ax.set_title(title, fontsize=9)
    _apply_axis_labels(ax)
    positions = list(range(0, LOOKBACK, 12))
    _overlay_xticks(ax, positions)
    _overlay_yticks(ax, positions)


def render_v1_grid(
    case_attention: np.ndarray,
    case_vmax: float,
    target_id: str,
    target_timestamp: str,
    seed: int,
    output_fp: Path,
) -> dict:
    """Render V1 grid: rows=layers, columns=heads, single shared colorbar.

    case_attention: [L_layers, H_heads, L, L]
    case_vmax: case-shared vmax
    Returns dict metadata.
    """
    L_layers, H_heads, _, _ = case_attention.shape
    fig, axes = plt.subplots(
        nrows=L_layers,
        ncols=H_heads,
        figsize=(4.0 * H_heads, 4.0 * L_layers),
        dpi=DPI,
    )

    for il in range(L_layers):
        for ih in range(H_heads):
            ax = axes[il, ih] if L_layers > 1 and H_heads > 1 else (
                axes[il] if H_heads == 1 else axes[ih]
            )
            M = case_attention[il, ih, :, :]
            render_mode_b_panel(M, case_vmax, ax, title=f"L{il+1}-H{ih+1}")

    # Shared colorbar
    img0 = axes[0, 0].images[0] if L_layers > 1 else (
        axes[0].images[0] if H_heads == 1 else axes[0].images[0]
    )
    cbar = fig.colorbar(img0, ax=axes.ravel().tolist(), shrink=0.85, label=COLORBAR_LABEL)

    fig.suptitle(
        f"V1 — Seed {seed} | {target_id} | {target_timestamp} | vmax={case_vmax:.4f}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_fp, dpi=DPI)
    plt.close(fig)

    return {
        "panel_count": L_layers * H_heads,
        "rows": L_layers,
        "columns": H_heads,
        "file_bytes": output_fp.stat().st_size,
        "vmin": MODE_B_VMIN,
        "vmax": case_vmax,
        "mode": MODE_B_NAME,
    }


def render_v2_grid(
    case_attention_per_seed: dict,
    case_vmax: float,
    target_id: str,
    target_timestamp: str,
    layer_idx: int,
    output_fp: Path,
) -> dict:
    """Render V2 grid: rows=seeds, columns=heads.

    case_attention_per_seed: {seed: np.ndarray [L_layers, H_heads, L, L]}
    """
    n_seeds = len(case_attention_per_seed)
    fig, axes = plt.subplots(
        nrows=n_seeds,
        ncols=NUM_HEADS,
        figsize=(3.5 * NUM_HEADS, 3.5 * n_seeds),
        dpi=DPI,
    )
    seed_list = sorted(case_attention_per_seed.keys())

    for ir, seed in enumerate(seed_list):
        for ih in range(NUM_HEADS):
            ax = axes[ir, ih]
            M = case_attention_per_seed[seed][layer_idx, ih, :, :]
            render_mode_b_panel(M, case_vmax, ax, title=f"S{seed}-L{layer_idx+1}-H{ih+1}")

    img0 = axes[0, 0].images[0]
    cbar = fig.colorbar(img0, ax=axes.ravel().tolist(), shrink=0.85, label=COLORBAR_LABEL)

    fig.suptitle(
        f"V2 — layer L{layer_idx+1} | {target_id} | {target_timestamp} | case vmax={case_vmax:.4f}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_fp, dpi=DPI)
    plt.close(fig)

    return {
        "panel_count": n_seeds * NUM_HEADS,
        "rows": n_seeds,
        "columns": NUM_HEADS,
        "file_bytes": output_fp.stat().st_size,
        "vmin": MODE_B_VMIN,
        "vmax": case_vmax,
        "mode": MODE_B_NAME,
    }


def render_fixed_probability_grid(
    case_attention_per_seed: dict,
    target_id: str,
    target_timestamp: str,
    layer_idx: int,
    output_fp: Path,
) -> dict:
    """Render Mode A FIXED_PROBABILITY [0,1] grid for a (case, layer)."""
    n_seeds = len(case_attention_per_seed)
    fig, axes = plt.subplots(
        nrows=n_seeds,
        ncols=NUM_HEADS,
        figsize=(4.0 * NUM_HEADS, 4.0 * n_seeds),
        dpi=DPI,
    )
    seed_list = sorted(case_attention_per_seed.keys())

    for ir, seed in enumerate(seed_list):
        for ih in range(NUM_HEADS):
            ax = axes[ir, ih]
            M = case_attention_per_seed[seed][layer_idx, ih, :, :]
            render_mode_a_panel(M, ax, title=f"S{seed}-L{layer_idx+1}-H{ih+1}")

    img0 = axes[0, 0].images[0]
    cbar = fig.colorbar(img0, ax=axes.ravel().tolist(), shrink=0.85, label=COLORBAR_LABEL)

    fig.suptitle(
        f"Mode A (FIXED [0,1]) — layer L{layer_idx+1} | {target_id} | {target_timestamp}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_fp, dpi=DPI)
    plt.close(fig)

    return {
        "panel_count": n_seeds * NUM_HEADS,
        "rows": n_seeds,
        "columns": NUM_HEADS,
        "file_bytes": output_fp.stat().st_size,
        "vmin": MODE_A_VMIN,
        "vmax": MODE_A_VMAX,
        "mode": MODE_A_NAME,
    }


def render_individual_map(
    M: np.ndarray,
    vmax: float,
    target_id: str,
    target_timestamp: str,
    seed: int,
    layer_display: int,
    head_display: int,
    mode: str,
    output_fp: Path,
) -> dict:
    """Render one Mode-B (or Mode A) per-head image, single panel."""
    if mode == MODE_A_NAME:
        vmin, vmax_used = MODE_A_VMIN, MODE_A_VMAX
    else:
        vmin, vmax_used = MODE_B_VMIN, vmax
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=DPI)
    img = ax.imshow(
        M,
        cmap=COLORMAP,
        vmin=vmin,
        vmax=vmax_used,
        interpolation="nearest",
        aspect="equal",
        origin="upper",
    )
    fig.colorbar(img, ax=ax, label=COLORBAR_LABEL)
    _apply_axis_labels(ax)
    positions = list(range(0, LOOKBACK, 12))
    _overlay_xticks(ax, positions)
    _overlay_yticks(ax, positions)
    ax.set_title(f"{target_id}\n{target_timestamp}\nseed {seed} | L{layer_display}-H{head_display} | {mode}", fontsize=9)
    fig.tight_layout()
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_fp, dpi=DPI)
    plt.close(fig)

    return {
        "file_bytes": output_fp.stat().st_size,
        "vmin": vmin,
        "vmax": vmax_used,
        "mode": mode,
    }


# ────────────────────────────── Phase 53-Helper: png-to-bytes for QA ─────────
def png_bytes(path: Path) -> bytes:
    """Read PNG bytes deterministically (used by image checksum step)."""
    return path.read_bytes()
