"""
Visualization utilities for trajectory prediction.

Plots predicted trajectories, ground truth, lane topology, and
attention weights for qualitative analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from typing import Any


def plot_trajectories(
    trajectories: np.ndarray,        # [K, T, 2] — predicted modes
    confidences: np.ndarray | None,  # [K] — confidence scores
    gt_future: np.ndarray | None,    # [T, 2] — ground truth
    history: np.ndarray | None = None,      # [T_hist, 2] — past trajectory
    lane_polylines: np.ndarray | None = None,  # [L, P, 2] — lane centerlines
    title: str = "Trajectory Prediction",
    save_path: str | None = None,
    figsize: tuple[int, int] = (10, 10),
) -> plt.Figure:
    """
    Plot multi-modal trajectory predictions with context.

    Args:
        trajectories: [K, T, 2] in agent-centric frame
        confidences: [K] softmax probabilities (optional)
        gt_future: [T, 2] ground truth future (optional)
        history: [T_hist, 2] past trajectory (optional)
        lane_polylines: [L, P, 2] lane centerline points (optional)
        title: plot title
        save_path: if provided, save figure to this path
        figsize: figure size in inches

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    K = trajectories.shape[0]
    colors = plt.cm.tab10(np.linspace(0, 1, max(K, 10)))

    # ── Lane polylines (background) ──
    if lane_polylines is not None:
        for lane in lane_polylines:
            valid = ~np.all(lane == 0, axis=1)
            pts = lane[valid]
            if len(pts) > 1:
                ax.plot(pts[:, 0], pts[:, 1], color="gray", linewidth=0.5, alpha=0.4)

    # ── History ──
    if history is not None:
        ax.plot(history[:, 0], history[:, 1], "k-", linewidth=1.5, label="History")
        ax.scatter(history[-1, 0], history[-1, 1], c="black", s=50, zorder=5, marker="s")

    # ── Predicted modes ──
    if confidences is not None:
        probs = np.exp(confidences) / np.sum(np.exp(confidences))
    else:
        probs = np.ones(K) / K

    for k in range(K):
        alpha = 0.3 + 0.7 * probs[k]
        label = f"Mode {k + 1} (p={probs[k]:.2f})" if confidences is not None else f"Mode {k + 1}"
        ax.plot(
            trajectories[k, :, 0], trajectories[k, :, 1],
            color=colors[k], linewidth=1.5 * (0.5 + probs[k]),
            alpha=alpha, label=label,
        )
        ax.scatter(
            trajectories[k, -1, 0], trajectories[k, -1, 1],
            color=colors[k], s=40 * (0.5 + probs[k]),
            alpha=alpha, marker="o",
        )

    # ── Ground truth ──
    if gt_future is not None:
        ax.plot(gt_future[:, 0], gt_future[:, 1], "r--", linewidth=2, label="GT Future")
        ax.scatter(gt_future[-1, 0], gt_future[-1, 1], c="red", s=60, zorder=5, marker="*")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_attention_weights(
    attention_weights: np.ndarray,    # [H, q_len, kv_len] or [q_len, kv_len]
    labels_q: list[str] | None = None,
    labels_kv: list[str] | None = None,
    title: str = "Attention Weights",
    save_path: str | None = None,
    figsize: tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Plot attention weight heatmap.

    Args:
        attention_weights: [H, q_len, kv_len] → averaged over heads, or [q_len, kv_len]
    """
    if attention_weights.ndim == 3:
        attn = attention_weights.mean(axis=0)
    else:
        attn = attention_weights

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(attn, cmap="viridis", aspect="auto")

    if labels_q:
        ax.set_yticks(range(len(labels_q)))
        ax.set_yticklabels(labels_q, fontsize=8)
    if labels_kv:
        ax.set_xticks(range(len(labels_kv)))
        ax.set_xticklabels(labels_kv, fontsize=8, rotation=45, ha="right")

    ax.set_xlabel("Key/Value")
    ax.set_ylabel("Query")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Attention Weight")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_metric_curves(
    metrics_history: dict[str, list[float]],
    title: str = "Training Curves",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot training/validation metric curves over epochs.

    Args:
        metrics_history: e.g. {"train_minADE": [...], "val_minADE": [...]}
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    metric_names = ["minADE", "minFDE", "MR", "loss"]
    for i, name in enumerate(metric_names):
        ax = axes[i]
        train_key = f"train_{name}"
        val_key = f"val_{name}"

        if train_key in metrics_history:
            ax.plot(metrics_history[train_key], label="Train", color="blue")
        if val_key in metrics_history:
            ax.plot(metrics_history[val_key], label="Val", color="red")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
