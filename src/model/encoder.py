"""
Agent-Map Interaction Encoder for MGFNet.

Encodes multi-modal inputs:
  - Target agent historical trajectory (20 steps)
  - Neighbor agent trajectories
  - Lane topology (polylines from HD map)

Architecture:
  1. Agent Temporal Encoder: MultiLevelAttention over each agent's history
  2. Lane Polyline Encoder: point-level + segment-level encoding
  3. Agent↔Map Cross-Modal Attention: bidirectional interaction
  4. Neighbor Interaction: target attends to neighbors
  5. Global Pooling: aggregate to fixed-size representation
"""

import torch
import torch.nn as nn

from .attention import (
    MultiLevelAttention,
    CrossModalAttention,
    NeighborAttention,
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
    TransformerEncoderLayer,
)


# ---------------------------------------------------------------------------
# Agent History Encoder
# ---------------------------------------------------------------------------

class AgentHistoryEncoder(nn.Module):
    """
    Encodes a single agent's 20-step history into a fixed-dimension token.

    Uses MultiLevelAttention (multi-scale dilation + Fourier differencing)
    followed by a learnable [CLS]-style query token attention pooling.
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        num_heads: int = 8,
        dilations: list[int] = [1, 2, 4],
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project raw input to hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Multi-level temporal attention
        self.temporal_attn = MultiLevelAttention(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dilations=dilations,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Learnable query token for attention pooling
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Cross-attention: query token attends to temporal features
        from .attention import CrossAttentionLayer
        self.pool_attn = CrossAttentionLayer(hidden_dim, num_heads, dropout)

    def forward(
        self,
        traj: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            traj: [batch, seq_len, input_dim] — agent trajectory
            mask: [batch, seq_len] — valid timesteps
        Returns:
            [batch, hidden_dim] — aggregated agent token
        """
        B = traj.size(0)
        h = self.input_proj(traj)  # [B, T, H]
        h = self.temporal_attn(h, mask)  # [B, T, H]

        # Attention pooling with learnable query
        query = self.query_token.expand(B, -1, -1)  # [B, 1, H]
        pooled = self.pool_attn(query, h, mask)  # [B, 1, H]
        return pooled.squeeze(1)  # [B, H]


# ---------------------------------------------------------------------------
# Lane Polyline Encoder
# ---------------------------------------------------------------------------

class LaneEncoder(nn.Module):
    """
    Encodes lane polylines into fixed-size per-lane tokens.

    Each lane is a polyline of P points, each with features (x, y, dx, dy, type).
    Encoding pipeline:
        1. PointMLP: per-point projection
        2. Point-level self-attention (within each lane)
        3. Segment-level aggregation (learnable query pooling)
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Point-level projection
        self.point_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Positional encoding for polyline points
        self.pos_encoding = LearnedPositionalEncoding(hidden_dim, max_len=100)

        # Point-level self-attention per lane
        self.point_encoders = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Segment query for pooling
        self.segment_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        from .attention import CrossAttentionLayer
        self.pool_attn = CrossAttentionLayer(hidden_dim, num_heads, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        lanes: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            lanes: [batch, num_lanes, num_points, input_dim]
            mask:  [batch, num_lanes, num_points] — valid points per lane
        Returns:
            [batch, num_lanes, hidden_dim] — per-lane tokens
        """
        B, L, P, D = lanes.shape

        # Flatten batch and lanes for point-level encoding
        lanes_flat = lanes.view(B * L, P, D)  # [B*L, P, D]
        h = self.point_mlp(lanes_flat)  # [B*L, P, H]
        h = self.pos_encoding(h)

        # Point-level self-attention (shared across all lanes)
        if mask is not None:
            mask_flat = mask.view(B * L, P)
        else:
            mask_flat = None

        for layer in self.point_encoders:
            if mask_flat is not None:
                attn_mask = mask_flat.unsqueeze(1).unsqueeze(2) & mask_flat.unsqueeze(1).unsqueeze(3)
                attn_mask = attn_mask[:, 0, 0, :].unsqueeze(1).unsqueeze(2)  # [B*L, 1, 1, P]
                h = layer(h, attn_mask)
            else:
                h = layer(h)

        # Pool each lane to a single token
        query = self.segment_query.expand(B * L, -1, -1)  # [B*L, 1, H]
        pooled = self.pool_attn(query, h, mask_flat)  # [B*L, 1, H]
        pooled = pooled.view(B, L, self.hidden_dim)  # [B, L, H]

        return pooled


# ---------------------------------------------------------------------------
# Full Agent-Map Encoder
# ---------------------------------------------------------------------------

class AgentMapEncoder(nn.Module):
    """
    Complete agent-map interaction encoder.

    Inputs:
        target_traj:   [B, 20, 6] — target agent history
        neighbor_trajs:[B, N, 20, 6] — neighbor histories
        lane_polylines:[B, L, P, 5] — lane topology
        masks for all of the above

    Processing:
        1. Encode target history via AgentHistoryEncoder → target_token [B, H]
        2. Encode each neighbor via AgentHistoryEncoder → neighbor_tokens [B, N, H]
        3. Encode lanes via LaneEncoder → map_tokens [B, L, H]
        4. Agent↔Map Cross-Modal Attention → enriched tokens
        5. Target attends to neighbors → socially-aware target token
        6. Concatenate and project to final agent token [B, H]
    """

    def __init__(
        self,
        agent_input_dim: int = 6,
        lane_input_dim: int = 5,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_encoder_layers: int = 4,
        num_map_layers: int = 3,
        num_neighbor_layers: int = 3,
        dilations: list[int] = [1, 2, 4],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Agent history encoder (shared across target + neighbors)
        self.agent_encoder = AgentHistoryEncoder(
            input_dim=agent_input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dilations=dilations,
            num_layers=num_encoder_layers,
            dropout=dropout,
        )

        # Lane encoder
        self.lane_encoder = LaneEncoder(
            input_dim=lane_input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=2,
            dropout=dropout,
        )

        # Agent ↔ Map cross-modal attention
        self.cross_modal = CrossModalAttention(
            d_model=hidden_dim,
            num_heads=num_heads,
            num_layers=num_map_layers,
            dropout=dropout,
        )

        # Neighbor interaction
        self.neighbor_attn = NeighborAttention(
            d_model=hidden_dim,
            num_heads=num_heads,
            num_layers=num_neighbor_layers,
            dropout=dropout,
        )

        # Map feature aggregation (pool all lane tokens → single map context)
        self.map_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        from .attention import CrossAttentionLayer
        self.map_pool = CrossAttentionLayer(hidden_dim, num_heads, dropout)

        # Final fusion: target + map_context + neighbor_context → agent_token
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        target_traj: torch.Tensor,
        neighbor_trajs: torch.Tensor,
        lane_polylines: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
        lane_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            target_traj:    [B, 20, 6]
            neighbor_trajs: [B, max_neighbors, 20, 6]
            lane_polylines: [B, max_lanes, max_points, 5]
            target_mask:    [B, 20]
            neighbor_mask:  [B, max_neighbors]
            lane_mask:      [B, max_lanes, max_points]

        Returns:
            agent_token:    [B, hidden_dim] — final agent representation
            map_tokens:     [B, max_lanes, hidden_dim] — lane tokens
            neighbor_tokens:[B, max_neighbors, hidden_dim] — neighbor tokens
        """
        B = target_traj.size(0)

        # ── 1. Encode target agent ──
        target_token = self.agent_encoder(target_traj, target_mask)  # [B, H]

        # ── 2. Encode neighbors ──
        N = neighbor_trajs.size(1)
        nbr_flat = neighbor_trajs.view(B * N, *neighbor_trajs.shape[2:])  # [B*N, 20, 6]
        if neighbor_mask is not None:
            nbr_tmask = neighbor_mask.unsqueeze(-1).expand(-1, -1, 20).reshape(B * N, 20)
        else:
            nbr_tmask = None
        nbr_tokens = self.agent_encoder(nbr_flat, nbr_tmask)  # [B*N, H]
        nbr_tokens = nbr_tokens.view(B, N, self.hidden_dim)  # [B, N, H]

        # ── 3. Encode lanes ──
        map_tokens = self.lane_encoder(lane_polylines, lane_mask)  # [B, L, H]

        # ── 4. Agent ↔ Map cross-modal attention ──
        # Stack target + neighbors as "agent tokens" for map interaction
        agent_tokens = torch.cat([
            target_token.unsqueeze(1),  # [B, 1, H]
            nbr_tokens,                 # [B, N, H]
        ], dim=1)  # [B, 1+N, H]

        if neighbor_mask is not None:
            full_agent_mask = torch.cat([
                torch.ones(B, 1, device=neighbor_mask.device, dtype=torch.bool),
                neighbor_mask,
            ], dim=1)
        else:
            full_agent_mask = None

        # Create lane-level mask (is any point in lane valid?)
        if lane_mask is not None:
            lane_valid = lane_mask.any(dim=-1)  # [B, L]
        else:
            lane_valid = None

        agent_tokens, map_tokens = self.cross_modal(
            agent_tokens, map_tokens, full_agent_mask, lane_valid
        )

        # ── 5. Neighbor interaction ──
        target_token = agent_tokens[:, 0:1, :]  # [B, 1, H]
        neighbor_tokens = agent_tokens[:, 1:, :]  # [B, N, H]

        target_token = self.neighbor_attn(
            target_token, neighbor_tokens, neighbor_mask
        )  # [B, 1, H]
        target_token = target_token.squeeze(1)  # [B, H]

        # ── 6. Aggregate map context ──
        map_query = self.map_query.expand(B, -1, -1)  # [B, 1, H]
        map_context = self.map_pool(map_query, map_tokens, lane_valid)  # [B, 1, H]
        map_context = map_context.squeeze(1)  # [B, H]

        # ── 7. Aggregate neighbor context ──
        if neighbor_mask is not None:
            nbr_context = (neighbor_tokens * neighbor_mask.unsqueeze(-1).float()).sum(dim=1) / \
                          (neighbor_mask.sum(dim=1, keepdim=True).float() + 1e-8)
        else:
            nbr_context = neighbor_tokens.mean(dim=1)

        # ── 8. Final fusion ──
        fused = torch.cat([target_token, map_context, nbr_context], dim=-1)  # [B, 3H]
        agent_token = self.fusion(self.dropout(fused))  # [B, H]

        return agent_token, map_tokens, neighbor_tokens
