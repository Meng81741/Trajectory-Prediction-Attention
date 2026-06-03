"""
Multi-Modal Trajectory Decoder for MGFNet.

Outputs K=6 distinct future trajectory modes, each consisting of 30 (Δx, Δy)
displacement predictions over a 3-second horizon at 10 Hz.

Architecture:
  - K learnable mode query embeddings
  - Each query cross-attends to the agent token and map tokens
  - Mode-specific MLP heads predict trajectory + confidence
  - Diversity regularization via inter-mode distance penalty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CrossAttentionLayer, MultiHeadAttention


class ModeQueryDecoder(nn.Module):
    """
    Decodes a single mode from a learnable query interacting with context.

    One instance handles all K modes in parallel.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Self-attention among mode queries (modes should be diverse)
        self.mode_self_attn = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(hidden_dim),
                "attn": MultiHeadAttention(hidden_dim, num_heads, dropout),
            })
            for _ in range(num_layers)
        ])

        # Cross-attention: mode queries attend to agent context
        self.agent_cross_attn = nn.ModuleList([
            CrossAttentionLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Cross-attention: mode queries attend to map context
        self.map_cross_attn = nn.ModuleList([
            CrossAttentionLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        mode_queries: torch.Tensor,
        agent_token: torch.Tensor,
        map_tokens: torch.Tensor,
        map_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            mode_queries: [batch, K, hidden_dim]
            agent_token:  [batch, 1, hidden_dim]
            map_tokens:   [batch, L, hidden_dim]
            map_mask:     [batch, L]
        Returns:
            [batch, K, hidden_dim] — enriched mode queries
        """
        h = mode_queries

        for i in range(len(self.agent_cross_attn)):
            # Self-attention among modes
            residual = h
            h_norm = self.mode_self_attn[i]["norm"](h)
            h = residual + self.dropout(
                self.mode_self_attn[i]["attn"](h_norm, h_norm, h_norm)
            )

            # Cross-attend to agent context
            h = self.agent_cross_attn[i](h, agent_token, None)

            # Cross-attend to map context
            h = self.map_cross_attn[i](h, map_tokens, map_mask)

        return h


class TrajectoryPredictionHead(nn.Module):
    """
    Predicts trajectory displacements + confidence score for one mode.

    Output:
        traj:   [batch, future_steps, 2] — (Δx, Δy) at each step
        conf:   [batch, 1] — mode probability logit
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        future_steps: int = 30,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.future_steps = future_steps

        self.traj_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, future_steps * 2),
        )

        self.conf_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, mode_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mode_feat: [batch, hidden_dim] — mode-specific feature
        Returns:
            traj: [batch, future_steps, 2]
            conf: [batch, 1]
        """
        batch = mode_feat.size(0)
        traj_flat = self.traj_decoder(mode_feat)  # [B, future_steps*2]
        traj = traj_flat.view(batch, self.future_steps, 2)  # [B, T_f, 2]
        conf = self.conf_head(mode_feat)  # [B, 1]
        return traj, conf


# ---------------------------------------------------------------------------
# Full Multi-Modal Decoder
# ---------------------------------------------------------------------------

class MultiModalDecoder(nn.Module):
    """
    Complete multi-modal decoder producing K=6 trajectory modes.

    Each mode:
      - A learnable query embedding interacts with agent + map context
      - Produces 30 (Δx, Δy) displacements and a confidence logit
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_modes: int = 6,
        future_steps: int = 30,
        num_heads: int = 8,
        num_decoder_layers: int = 4,
        mode_embedding_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.hidden_dim = hidden_dim

        # Learnable mode query embeddings
        self.mode_embeddings = nn.Parameter(
            torch.randn(num_modes, mode_embedding_dim) * 0.02
        )
        self.mode_proj = nn.Linear(mode_embedding_dim, hidden_dim)

        # Mode query decoder
        self.query_decoder = ModeQueryDecoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            dropout=dropout,
        )

        # Shared trajectory prediction head (applied per-mode)
        self.pred_head = TrajectoryPredictionHead(
            hidden_dim=hidden_dim,
            future_steps=future_steps,
            dropout=dropout,
        )

    def forward(
        self,
        agent_token: torch.Tensor,
        map_tokens: torch.Tensor,
        map_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            agent_token: [batch, hidden_dim] — aggregated agent representation
            map_tokens:  [batch, max_lanes, hidden_dim] — lane tokens
            map_mask:    [batch, max_lanes]
        Returns:
            trajectories: [batch, K, future_steps, 2] — (Δx, Δy) for each mode
            confidences:  [batch, K] — logits (softmax applied during loss)
        """
        B = agent_token.size(0)

        # Expand mode embeddings to batch
        mode_emb = self.mode_embeddings.unsqueeze(0).expand(B, -1, -1)  # [B, K, E]
        mode_queries = self.mode_proj(mode_emb)  # [B, K, H]

        # Agent token as context
        agent_ctx = agent_token.unsqueeze(1)  # [B, 1, H]

        # Decode mode queries
        mode_feats = self.query_decoder(
            mode_queries, agent_ctx, map_tokens, map_mask
        )  # [B, K, H]

        # Predict trajectory + confidence per mode
        all_trajs = []
        all_confs = []

        for k in range(self.num_modes):
            feat_k = mode_feats[:, k, :]  # [B, H]
            traj_k, conf_k = self.pred_head(feat_k)
            all_trajs.append(traj_k.unsqueeze(1))  # [B, 1, T, 2]
            all_confs.append(conf_k)  # [B, 1]

        trajectories = torch.cat(all_trajs, dim=1)  # [B, K, T, 2]
        confidences = torch.cat(all_confs, dim=1)  # [B, K]

        return trajectories, confidences


# ---------------------------------------------------------------------------
# Helper: compute diversity loss between modes
# ---------------------------------------------------------------------------

def compute_diversity_loss(
    trajectories: torch.Tensor,  # [B, K, T, 2]
    margin: float = 1.0,
) -> torch.Tensor:
    """
    Encourages mode diversity by penalizing mode pairs that are too similar.

    Uses a hinge loss on pairwise L2 distance between mode endpoints.
    """
    B, K, T, _ = trajectories.shape
    endpoints = trajectories[:, :, -1, :]  # [B, K, 2] — final position per mode

    # Pairwise distances
    dists = torch.cdist(endpoints, endpoints, p=2)  # [B, K, K]

    # Penalize pairs closer than margin
    loss = F.relu(margin - dists).sum(dim=(1, 2)) / (K * (K - 1) + 1e-8)
    return loss.mean()
