"""
MGFNet: Multi-Level Attention Trajectory Prediction Model.

End-to-end model for Argoverse 1 Motion Forecasting.

Architecture:
    Input
    ├── Target Agent History (20 steps × 6 features)
    ├── Neighbor Agent Histories (N × 20 × 6)
    └── Lane Topology Polylines (L lanes × P points × 5 features)
         │
    AgentMapEncoder
    ├── AgentHistoryEncoder → MultiLevelAttention (multi-scale temporal)
    ├── LaneEncoder → Polyline self-attention + pooling
    ├── CrossModalAttention → agent ↔ map interaction
    ├── NeighborAttention → social interaction
    └── Fusion → agent_token [B, H]
         │
    MultiModalDecoder
    ├── K=6 learnable mode queries
    ├── ModeQueryDecoder → self-attn + cross-attn to agent + map
    └── TrajectoryPredictionHead → K × 30 × (Δx, Δy) + confidence
         │
    Output
    ├── trajectories: [B, 6, 30, 2]
    └── confidences: [B, 6]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

from .encoder import AgentMapEncoder
from .decoder import MultiModalDecoder, compute_diversity_loss


class MGFNet(nn.Module):
    """
    MGFNet: Multi-Level Graph Fourier Network for trajectory prediction.

    Combines:
      - Multi-level temporal attention (multi-scale dilation + Fourier differencing)
      - Agent-map cross-modal attention
      - Neighbor social interaction
      - Multi-modal trajectory decoding (6 modes)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Args:
            config: model hyperparameters dict. If None, uses defaults.
        """
        super().__init__()

        cfg = config or {}
        model_cfg = cfg.get("model", {})

        # Hyperparameters
        self.agent_input_dim = model_cfg.get("agent_input_dim", 6)
        self.lane_input_dim = model_cfg.get("lane_input_dim", 5)
        self.hidden_dim = model_cfg.get("hidden_dim", 128)
        self.num_heads = model_cfg.get("num_attention_heads", 8)
        self.num_encoder_layers = model_cfg.get("num_encoder_layers", 4)
        self.num_decoder_layers = model_cfg.get("num_decoder_layers", 4)
        self.dilations = model_cfg.get("temporal_dilations", [1, 2, 4])
        self.num_map_layers = model_cfg.get("num_map_layers", 3)
        self.num_neighbor_layers = model_cfg.get("num_neighbor_layers", 3)
        self.num_modes = model_cfg.get("num_modes", 6)
        self.future_steps = model_cfg.get("future_steps", 30)
        self.mode_embedding_dim = model_cfg.get("mode_embedding_dim", 64)
        self.dropout = model_cfg.get("dropout", 0.1)

        # ── Encoder ──
        self.encoder = AgentMapEncoder(
            agent_input_dim=self.agent_input_dim,
            lane_input_dim=self.lane_input_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_encoder_layers=self.num_encoder_layers,
            num_map_layers=self.num_map_layers,
            num_neighbor_layers=self.num_neighbor_layers,
            dilations=self.dilations,
            dropout=self.dropout,
        )

        # ── Decoder ──
        self.decoder = MultiModalDecoder(
            hidden_dim=self.hidden_dim,
            num_modes=self.num_modes,
            future_steps=self.future_steps,
            num_heads=self.num_heads,
            num_decoder_layers=self.num_decoder_layers,
            mode_embedding_dim=self.mode_embedding_dim,
            dropout=self.dropout,
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize linear layers with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        target_traj: torch.Tensor,
        neighbor_trajs: torch.Tensor,
        lane_polylines: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
        lane_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            target_traj:    [B, 20, 6] — target agent history
            neighbor_trajs: [B, N, 20, 6] — neighbor histories
            lane_polylines: [B, L, P, 5] — lane polylines
            target_mask:    [B, 20]
            neighbor_mask:  [B, N]
            lane_mask:      [B, L, P]

        Returns:
            trajectories: [B, 6, 30, 2] — (Δx, Δy) per mode per step
            confidences:  [B, 6]         — mode logits
        """
        # Encode
        agent_token, map_tokens, _ = self.encoder(
            target_traj=target_traj,
            neighbor_trajs=neighbor_trajs,
            lane_polylines=lane_polylines,
            target_mask=target_mask,
            neighbor_mask=neighbor_mask,
            lane_mask=lane_mask,
        )

        # Create lane-level validity mask
        if lane_mask is not None:
            map_valid = lane_mask.any(dim=-1)  # [B, L]
        else:
            map_valid = None

        # Decode multi-modal trajectories
        trajectories, confidences = self.decoder(
            agent_token=agent_token,
            map_tokens=map_tokens,
            map_mask=map_valid,
        )

        return trajectories, confidences

    def compute_loss(
        self,
        trajectories: torch.Tensor,
        confidences: torch.Tensor,
        gt_future: torch.Tensor,
        gt_mask: torch.Tensor | None = None,
        diversity_weight: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        """
        Compute multi-modal loss.

        Uses Winner-Takes-All: only the mode closest to GT receives
        regression gradient. Confidence is trained via cross-entropy
        against the "winning" mode.

        Args:
            trajectories: [B, K, T, 2] — predicted displacements
            confidences:  [B, K]        — mode logits
            gt_future:    [B, T, 2]     — ground-truth displacements
            gt_mask:      [B, T]        — valid GT timesteps
            diversity_weight: weight for mode diversity regularization

        Returns:
            dict with "reg_loss", "conf_loss", "div_loss", "total_loss"
        """
        B, K, T, _ = trajectories.shape

        # Compute L2 distance between each mode and GT at final endpoint
        with torch.no_grad():
            if gt_mask is not None:
                gt_endpoint = gt_future[gt_mask].view(B, -1, 2)[:, -1, :]  # [B, 2]
                pred_endpoints = trajectories[gt_mask.unsqueeze(1).expand(-1, K, -1, -1)]
                pred_endpoints = pred_endpoints.view(B, K, -1, 2)[:, :, -1, :]  # [B, K, 2]
            else:
                gt_endpoint = gt_future[:, -1, :]  # [B, 2]
                pred_endpoints = trajectories[:, :, -1, :]  # [B, K, 2]

            dists = torch.norm(pred_endpoints - gt_endpoint.unsqueeze(1), dim=-1)  # [B, K]
            winner_idx = dists.argmin(dim=1)  # [B]

        # ── Regression loss (only winning mode) ──
        reg_loss = torch.tensor(0.0, device=trajectories.device)
        valid_count = 0
        for b in range(B):
            k = winner_idx[b]
            pred = trajectories[b, k]  # [T, 2]
            gt = gt_future[b]  # [T, 2]
            if gt_mask is not None:
                mask_b = gt_mask[b]  # [T]
                diff = (pred[mask_b] - gt[mask_b]).norm(dim=-1)
                reg_loss = reg_loss + diff.mean()
            else:
                reg_loss = reg_loss + (pred - gt).norm(dim=-1).mean()
            valid_count += 1

        reg_loss = reg_loss / max(valid_count, 1)

        # ── Confidence loss (cross-entropy) ──
        conf_loss = F.cross_entropy(confidences, winner_idx)

        # ── Diversity loss ──
        div_loss = compute_diversity_loss(trajectories)

        # ── Total ──
        total_loss = reg_loss + conf_loss + diversity_weight * div_loss

        return {
            "reg_loss": reg_loss,
            "conf_loss": conf_loss,
            "div_loss": div_loss,
            "total_loss": total_loss,
        }

