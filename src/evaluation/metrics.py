"""
Evaluation metrics for trajectory prediction.

Implements standard Argoverse metrics:
  - minADE: minimum Average Displacement Error over K modes
  - minFDE: minimum Final Displacement Error over K modes
  - MR: Miss Rate (fraction of samples where minFDE > threshold)
  - DAC: Drivable Area Compliance (fraction of predictions on road)
  - brier-minFDE: minFDE + (1 - confidence_of_best_mode)
"""

import torch
import torch.nn.functional as F
from typing import Any


# ---------------------------------------------------------------------------
# Core Metrics
# ---------------------------------------------------------------------------

def minADE(
    trajectories: torch.Tensor,   # [B, K, T, 2]
    gt_future: torch.Tensor,      # [B, T, 2]
    gt_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Minimum Average Displacement Error.

    For each sample, selects the mode with lowest ADE and averages
    across the batch.
    """
    B, K, T, _ = trajectories.shape

    if gt_mask is not None:
        mask = gt_mask.unsqueeze(1).unsqueeze(-1)
        diff = (trajectories - gt_future.unsqueeze(1)) * mask
        err = diff.norm(dim=-1)
        ade_per_mode = err.sum(dim=-1) / (gt_mask.sum(dim=-1, keepdim=True) + 1e-8)
    else:
        diff = trajectories - gt_future.unsqueeze(1)
        ade_per_mode = diff.norm(dim=-1).mean(dim=-1)

    return ade_per_mode.min(dim=1).values.mean()


def minFDE(
    trajectories: torch.Tensor,
    gt_future: torch.Tensor,
    gt_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Minimum Final Displacement Error.

    For each sample, selects the mode with lowest FDE (error at final
    timestep) and averages across the batch.
    """
    B, K, T, _ = trajectories.shape

    if gt_mask is not None:
        last_valid = gt_mask.float().sum(dim=1).long() - 1
        last_valid = last_valid.clamp(min=0)
        batch_idx = torch.arange(B, device=trajectories.device)
        gt_end = gt_future[batch_idx, last_valid]
        pred_end = trajectories[batch_idx, :, last_valid]
    else:
        gt_end = gt_future[:, -1, :]
        pred_end = trajectories[:, :, -1, :]

    fde_per_mode = torch.norm(pred_end - gt_end.unsqueeze(1), dim=-1)
    return fde_per_mode.min(dim=1).values.mean()


def miss_rate(
    trajectories: torch.Tensor,
    gt_future: torch.Tensor,
    gt_mask: torch.Tensor | None = None,
    threshold: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Miss Rate: fraction of predictions where ALL modes miss by > threshold.

    Uses FDE (error at final timestep) as the distance metric.

    Returns:
        mr: scalar [0, 1]
        min_fde_per_sample: [B]
    """
    B, K, T, _ = trajectories.shape

    if gt_mask is not None:
        last_valid = gt_mask.float().sum(dim=1).long() - 1
        last_valid = last_valid.clamp(min=0)
        batch_idx = torch.arange(B, device=trajectories.device)
        gt_end = gt_future[batch_idx, last_valid]
        pred_end = trajectories[batch_idx, :, last_valid]
    else:
        gt_end = gt_future[:, -1, :]
        pred_end = trajectories[:, :, -1, :]

    fde_per_mode = torch.norm(pred_end - gt_end.unsqueeze(1), dim=-1)
    min_fde = fde_per_mode.min(dim=1).values
    mr = (min_fde > threshold).float().mean()

    return mr, min_fde


def brier_minFDE(
    trajectories: torch.Tensor,
    confidences: torch.Tensor,
    gt_future: torch.Tensor,
    gt_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Brier-minFDE: minFDE ​+ (1 − p_best)².

    Where p_best is the softmax probability of the best (lowest FDE) mode.
    Penalizes both poor localization and poor confidence calibration.
    """
    B, K, T, _ = trajectories.shape

    if gt_mask is not None:
        last_valid = gt_mask.float().sum(dim=1).long() - 1
        last_valid = last_valid.clamp(min=0)
        batch_idx = torch.arange(B, device=trajectories.device)
        gt_end = gt_future[batch_idx, last_valid]
        pred_end = trajectories[batch_idx, :, last_valid]
    else:
        gt_end = gt_future[:, -1, :]
        pred_end = trajectories[:, :, -1, :]

    fde_per_mode = torch.norm(pred_end - gt_end.unsqueeze(1), dim=-1)
    min_fde, winner_idx = fde_per_mode.min(dim=1)

    probs = F.softmax(confidences, dim=-1)
    p_best = probs[torch.arange(B, device=trajectories.device), winner_idx]

    return (min_fde + (1 - p_best) ** 2).mean()


# ---------------------------------------------------------------------------
# Compute all metrics at once
# ---------------------------------------------------------------------------

def compute_metrics(
    trajectories: torch.Tensor,
    confidences: torch.Tensor,
    gt_future: torch.Tensor,
    gt_mask: torch.Tensor | None = None,
    mr_threshold: float = 2.0,
) -> dict[str, float]:
    """
    Compute all standard trajectory prediction metrics.

    Args:
        trajectories: [B, K, T, 2]
        confidences:  [B, K]
        gt_future:    [B, T, 2]
        gt_mask:      [B, T]
        mr_threshold: miss rate threshold in meters

    Returns:
        dict with minADE, minFDE, MR, brier_minFDE
    """
    with torch.no_grad():
        ade = minADE(trajectories, gt_future, gt_mask)
        fde = minFDE(trajectories, gt_future, gt_mask)
        mr, _ = miss_rate(trajectories, gt_future, gt_mask, mr_threshold)
        brier = brier_minFDE(trajectories, confidences, gt_future, gt_mask)

    return {
        "minADE": round(ade.item(), 4),
        "minFDE": round(fde.item(), 4),
        "MR": round(mr.item(), 4),
        "brier_minFDE": round(brier.item(), 4),
    }


def compute_per_mode_metrics(
    trajectories: torch.Tensor,
    gt_future: torch.Tensor,
    gt_mask: torch.Tensor | None = None,
) -> dict[str, list[float]]:
    """
    Compute ADE/FDE for each mode separately (not just the best).
    Useful for analyzing mode coverage.
    """
    B, K, T, _ = trajectories.shape

    if gt_mask is not None:
        mask = gt_mask.unsqueeze(1).unsqueeze(-1)
        diff = (trajectories - gt_future.unsqueeze(1)) * mask
        err = diff.norm(dim=-1)
        ade_per_mode = err.sum(dim=-1) / (gt_mask.sum(dim=-1, keepdim=True) + 1e-8)

        last_valid = gt_mask.float().sum(dim=1).long() - 1
        last_valid = last_valid.clamp(min=0)
        batch_idx = torch.arange(B, device=trajectories.device)
        gt_end = gt_future[batch_idx, last_valid]
        pred_end = trajectories[batch_idx, :, last_valid]
        fde_per_mode = torch.norm(pred_end - gt_end.unsqueeze(1), dim=-1)
    else:
        diff = trajectories - gt_future.unsqueeze(1)
        ade_per_mode = diff.norm(dim=-1).mean(dim=-1)
        fde_per_mode = torch.norm(
            trajectories[:, :, -1, :] - gt_future[:, -1, :].unsqueeze(1), dim=-1
        )

    return {
        "ade_per_mode": [round(ade_per_mode[:, k].mean().item(), 4) for k in range(K)],
        "fde_per_mode": [round(fde_per_mode[:, k].mean().item(), 4) for k in range(K)],
    }
