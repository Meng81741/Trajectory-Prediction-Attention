"""
Loss functions for multi-modal trajectory prediction.

Key losses:
  - Winner-Takes-All (WTA) regression loss: only the best mode gets gradient
  - Confidence loss: cross-entropy on mode selection
  - Diversity loss: encourages mode spread
  - Off-road penalty: discourage predictions far from lane graph
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Winner-Takes-All Regression Loss
# ---------------------------------------------------------------------------

def winner_takes_all_loss(
    trajectories: torch.Tensor,     # [B, K, T, 2]
    gt_future: torch.Tensor,        # [B, T, 2]
    gt_mask: torch.Tensor | None = None,  # [B, T]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute minADE over K modes using the winner-takes-all strategy.

    For each sample, selects the mode with minimum average displacement
    error and computes the loss only for that mode.

    Returns:
        loss: scalar WTA regression loss
        winner_idx: [B] index of winning mode
    """
    B, K, T, _ = trajectories.shape

    # Compute per-mode error on valid timesteps
    if gt_mask is not None:
        # Apply mask — only count valid timesteps
        mask = gt_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, T, 1]
        diff = (trajectories - gt_future.unsqueeze(1)) * mask  # [B, K, T, 2]
        err = diff.norm(dim=-1)  # [B, K, T]
        valid_counts = mask.sum(dim=(2, 3)) + 1e-8  # [B, 1, 1]
        ade = err.sum(dim=-1) / gt_mask.sum(dim=-1, keepdim=True).unsqueeze(1)  # [B, K]
    else:
        diff = trajectories - gt_future.unsqueeze(1)  # [B, K, T, 2]
        ade = diff.norm(dim=-1).mean(dim=-1)  # [B, K]

    # Winner = mode with minimum ADE
    min_ade, winner_idx = ade.min(dim=1)  # [B], [B]

    # Gather ADE only for the winning mode
    batch_indices = torch.arange(B, device=trajectories.device)
    winner_ade = ade[batch_indices, winner_idx].mean()

    return winner_ade, winner_idx


# ---------------------------------------------------------------------------
# Final Displacement Error (FDE) Loss
# ---------------------------------------------------------------------------

def final_displacement_error(
    trajectories: torch.Tensor,     # [B, K, T, 2]
    gt_future: torch.Tensor,        # [B, T, 2]
    gt_mask: torch.Tensor | None = None,
    mode: str = "min",
) -> torch.Tensor:
    """
    Compute FDE over K modes.

    Args:
        mode: "min" → minFDE (best mode), "avg" → average FDE

    Returns:
        fde: scalar
    """
    B, K, T, _ = trajectories.shape

    if gt_mask is not None:
        # Find last valid timestep per sample
        last_valid = gt_mask.float().sum(dim=1).long() - 1  # [B]
        last_valid = last_valid.clamp(min=0)

        batch_idx = torch.arange(B, device=trajectories.device)
        gt_end = gt_future[batch_idx, last_valid]  # [B, 2]
        pred_end = trajectories[batch_idx, :, last_valid]  # [B, K, 2]
    else:
        gt_end = gt_future[:, -1, :]  # [B, 2]
        pred_end = trajectories[:, :, -1, :]  # [B, K, 2]

    fde_per_mode = torch.norm(pred_end - gt_end.unsqueeze(1), dim=-1)  # [B, K]

    if mode == "min":
        return fde_per_mode.min(dim=1).values.mean()
    else:
        return fde_per_mode.mean()


# ---------------------------------------------------------------------------
# Average Displacement Error (ADE)
# ---------------------------------------------------------------------------

def average_displacement_error(
    trajectories: torch.Tensor,     # [B, K, T, 2]
    gt_future: torch.Tensor,        # [B, T, 2]
    gt_mask: torch.Tensor | None = None,
    mode: str = "min",
) -> torch.Tensor:
    """
    Compute ADE over K modes.

    Args:
        mode: "min" → minADE, "avg" → average ADE

    Returns:
        ade: scalar
    """
    if gt_mask is not None:
        mask = gt_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, T, 1]
        diff = (trajectories - gt_future.unsqueeze(1)) * mask
        err = diff.norm(dim=-1)  # [B, K, T]
        ade_per_mode = err.sum(dim=-1) / (gt_mask.sum(dim=-1, keepdim=True) + 1e-8)  # [B, K]
    else:
        diff = trajectories - gt_future.unsqueeze(1)
        ade_per_mode = diff.norm(dim=-1).mean(dim=-1)

    if mode == "min":
        return ade_per_mode.min(dim=1).values.mean()
    else:
        return ade_per_mode.mean()


# ---------------------------------------------------------------------------
# Miss Rate
# ---------------------------------------------------------------------------

def miss_rate(
    trajectories: torch.Tensor,     # [B, K, T, 2]
    gt_future: torch.Tensor,        # [B, T, 2]
    gt_mask: torch.Tensor | None = None,
    threshold: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Miss Rate: fraction of samples where ALL K modes have
    FDE > threshold.

    Returns:
        mr: scalar miss rate [0, 1]
        min_fde: [B] min FDE per sample
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

    fde_per_mode = torch.norm(pred_end - gt_end.unsqueeze(1), dim=-1)  # [B, K]
    min_fde = fde_per_mode.min(dim=1).values  # [B]
    mr = (min_fde > threshold).float().mean()

    return mr, min_fde


# ---------------------------------------------------------------------------
# Combined Multi-Modal Loss
# ---------------------------------------------------------------------------

class MultiModalLoss(nn.Module):
    """
    Combined loss for multi-modal trajectory prediction.

    loss = w_reg * WTA_regression_loss
         + w_conf * confidence_cross_entropy
         + w_div * diversity_loss
    """

    def __init__(
        self,
        regression_weight: float = 1.0,
        confidence_weight: float = 1.0,
        diversity_weight: float = 0.1,
    ):
        super().__init__()
        self.regression_weight = regression_weight
        self.confidence_weight = confidence_weight
        self.diversity_weight = diversity_weight

    def forward(
        self,
        trajectories: torch.Tensor,  # [B, K, T, 2]
        confidences: torch.Tensor,   # [B, K]
        gt_future: torch.Tensor,     # [B, T, 2]
        gt_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict with reg_loss, conf_loss, div_loss, total_loss, and metrics
        """
        B, K, T, _ = trajectories.shape

        # ── Regression loss (WTA) ──
        reg_loss, winner_idx = winner_takes_all_loss(
            trajectories, gt_future, gt_mask
        )

        # ── Confidence loss ──
        conf_loss = F.cross_entropy(confidences, winner_idx)

        # ── Diversity loss ──
        from ..model.decoder import compute_diversity_loss
        div_loss = compute_diversity_loss(trajectories)

        # ── Total ──
        total_loss = (
            self.regression_weight * reg_loss
            + self.confidence_weight * conf_loss
            + self.diversity_weight * div_loss
        )

        # ── Metrics (no grad) ──
        with torch.no_grad():
            min_ade = average_displacement_error(trajectories, gt_future, gt_mask, "min")
            min_fde = final_displacement_error(trajectories, gt_future, gt_mask, "min")
            mr, _ = miss_rate(trajectories, gt_future, gt_mask, threshold=2.0)

        return {
            "reg_loss": reg_loss,
            "conf_loss": conf_loss,
            "div_loss": div_loss,
            "total_loss": total_loss,
            "minADE": min_ade,
            "minFDE": min_fde,
            "MR": mr,
        }


# ---------------------------------------------------------------------------
# Convenience aliases
# ---------------------------------------------------------------------------

def minADE(
    trajectories: torch.Tensor,
    gt_future: torch.Tensor,
    gt_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return average_displacement_error(trajectories, gt_future, gt_mask, "min")


def minFDE(
    trajectories: torch.Tensor,
    gt_future: torch.Tensor,
    gt_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return final_displacement_error(trajectories, gt_future, gt_mask, "min")
