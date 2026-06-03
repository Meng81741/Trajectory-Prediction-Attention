"""
Multi-Level Attention Modules for MGFNet.

Implements:
- MultiLevelTemporalAttention: hierarchical attention over historical trajectory
  with multi-scale dilation rates (inspired by MGFNet's multi-scale differencing).
- CrossModalAttention: agent↔map interaction via cross-attention.
- FourierDiffEncoder: multi-scale Fourier differencing to capture motion dynamics
  at different temporal resolutions.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            x with positional encoding added: [batch, seq_len, d_model]
        """
        return x + self.pe[: x.size(1)].unsqueeze(0)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embedding for map/lane tokens (spatial)."""

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# Multi-Head Attention (base building block)
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Standard multi-head scaled dot-product attention."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [batch, q_len, d_model]
            key:   [batch, kv_len, d_model]
            value: [batch, kv_len, d_model]
            mask:  [batch, q_len, kv_len] or [batch, 1, 1, kv_len] — True = attend
        Returns:
            [batch, q_len, d_model]
        """
        B = query.size(0)

        Q = self.w_q(query).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, q, kv]

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)  # [B, H, q, d_k]
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.w_o(out)


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Position-wise feed-forward with GELU activation."""

    def __init__(self, d_model: int, d_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Transformer Encoder Layer
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    """Self-attention + FFN with pre-norm residual."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Self-attention
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, mask)
        x = residual + self.dropout(x)
        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        return residual + x


# ---------------------------------------------------------------------------
# Cross-Attention Layer
# ---------------------------------------------------------------------------

class CrossAttentionLayer(nn.Module):
    """Cross-attention from source to context, with pre-norm residual."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm_src = nn.LayerNorm(d_model)
        self.norm_ctx = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm_out = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        ctx: torch.Tensor,
        ctx_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            src:     [batch, src_len, d_model] — query
            ctx:     [batch, ctx_len, d_model] — key/value
            ctx_mask:[batch, ctx_len] or [batch, 1, ctx_len]
        Returns:
            [batch, src_len, d_model]
        """
        residual = src
        src_normed = self.norm_src(src)
        ctx_normed = self.norm_ctx(ctx)
        out = self.cross_attn(src_normed, ctx_normed, ctx_normed, ctx_mask)
        out = residual + self.dropout(out)

        residual = out
        out = self.norm_out(out)
        out = self.ffn(out)
        return residual + out


# ---------------------------------------------------------------------------
# Fourier Multi-Scale Difference Encoder (MGFNet core)
# ---------------------------------------------------------------------------

class FourierDiffEncoder(nn.Module):
    """
    Multi-scale Fourier differencing encoder.

    Extracts motion patterns at multiple temporal resolutions by computing
    differences at various dilation rates and projecting them with a Fourier basis.
    This captures both fine-grained and coarse motion dynamics.

    Args:
        input_dim: dimension of per-timestep features (e.g., 6 for x,y,vx,vy,h,Δt)
        hidden_dim: output embedding dimension
        dilations: list of dilation rates for multi-scale differencing
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dilations: list[int] = [1, 2, 4],
        num_fourier_components: int = 32,
    ):
        super().__init__()
        self.dilations = dilations
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_fourier = num_fourier_components

        # Fourier basis frequencies (learnable)
        self.frequencies = nn.Parameter(
            torch.randn(len(dilations), num_fourier_components) * 0.1
        )

        # Project concatenated differences + Fourier features → hidden_dim
        total_diff_dim = input_dim * len(dilations)
        self.proj = nn.Sequential(
            nn.Linear(total_diff_dim + 2 * num_fourier_components * len(dilations), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim] — agent trajectory history
        Returns:
            [batch, seq_len, hidden_dim] — multi-scale difference features
        """
        B, T, D = x.shape
        diff_features = []

        for i, d in enumerate(self.dilations):
            # Compute difference at dilation rate d
            if d < T:
                diff = x[:, d:] - x[:, :-d]  # [B, T-d, D]
                # Pad to original length
                pad = torch.zeros(B, d, D, device=x.device)
                diff = torch.cat([pad, diff], dim=1)  # [B, T, D]
            else:
                diff = torch.zeros(B, T, D, device=x.device)
            diff_features.append(diff)

        # Concatenate all scale differences
        all_diffs = torch.cat(diff_features, dim=-1)  # [B, T, D * num_dilations]

        # Fourier features at each dilation scale
        fourier_parts = []
        t = torch.arange(T, device=x.device, dtype=torch.float32).unsqueeze(1)  # [T, 1]

        for i in range(len(self.dilations)):
            freqs = self.frequencies[i]  # [num_fourier]
            angles = t * freqs.unsqueeze(0)  # [T, num_fourier]
            fourier_parts.append(torch.sin(angles))
            fourier_parts.append(torch.cos(angles))

        fourier_feat = torch.cat(fourier_parts, dim=-1)  # [T, 2 * num_fourier * num_dilations]
        fourier_feat = fourier_feat.unsqueeze(0).expand(B, -1, -1)  # [B, T, ...]

        combined = torch.cat([all_diffs, fourier_feat], dim=-1)
        return self.proj(combined)  # [B, T, hidden_dim]


# ---------------------------------------------------------------------------
# Multi-Level Temporal Attention (MGFNet signature)
# ---------------------------------------------------------------------------

class MultiLevelAttention(nn.Module):
    """
    Hierarchical multi-level temporal attention.

    Applies self-attention at multiple temporal resolutions (dilation rates)
    and fuses the results. This captures motion patterns at different time scales
    — from sub-second maneuvers to multi-second intentions.

    Architecture:
        1. FourierDiffEncoder extracts multi-scale difference features.
        2. Parallel TransformerEncoderLayers at each dilation rate attend
           over temporally-stratified views of the trajectory.
        3. Outputs are concatenated and projected to hidden_dim.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dilations: list[int] = [1, 2, 4],
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dilations = dilations
        self.hidden_dim = hidden_dim

        # Multi-scale Fourier differencing
        self.fourier_encoder = FourierDiffEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dilations=dilations,
        )

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim)

        # Parallel transformer blocks — one per dilation scale
        self.scale_encoders = nn.ModuleList([
            nn.ModuleList([
                TransformerEncoderLayer(hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ])
            for _ in dilations
        ])

        # Cross-scale fusion: concatenate all scales → hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * len(dilations), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:    [batch, seq_len, input_dim] — raw trajectory features
            mask: [batch, seq_len] — True = valid timestep
        Returns:
            [batch, seq_len, hidden_dim] — temporally enriched features
        """
        # Multi-scale Fourier encoding
        h = self.fourier_encoder(x)  # [B, T, hidden_dim]
        h = self.pos_encoding(h)

        scale_outputs = []
        for scale_idx, encoder_layers in enumerate(self.scale_encoders):
            h_scale = h.clone()
            for layer in encoder_layers:
                h_scale = layer(h_scale, mask)
            scale_outputs.append(h_scale)

        # Fuse across scales
        fused = torch.cat(scale_outputs, dim=-1)  # [B, T, H * num_scales]
        return self.fusion(self.dropout(fused))


# ---------------------------------------------------------------------------
# Cross-Modal Attention (Agent ↔ Map)
# ---------------------------------------------------------------------------

class CrossModalAttention(nn.Module):
    """
    Agent↔Map cross-modal interaction encoder.

    Alternating cross-attention layers allow the agent state to attend to the
    lane topology (map) and vice versa, building a joint agent-map representation.

    Two directions:
        Agent → Map: agent queries attend to lane key/values
        Map → Agent: lane queries attend to agent key/values
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Agent → Map cross-attention layers
        self.agent_to_map = nn.ModuleList([
            CrossAttentionLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Map → Agent cross-attention layers
        self.map_to_agent = nn.ModuleList([
            CrossAttentionLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        agent_feat: torch.Tensor,
        map_feat: torch.Tensor,
        agent_mask: torch.Tensor | None = None,
        map_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            agent_feat: [batch, num_agents, d_model]
            map_feat:   [batch, num_lanes, d_model]
            agent_mask: [batch, num_agents]
            map_mask:   [batch, num_lanes]
        Returns:
            (updated_agent_feat, updated_map_feat) — both [B, N, d_model]
        """
        # Agent attends to map → enriches agent with lane context
        agent_out = agent_feat
        for layer in self.agent_to_map:
            agent_out = layer(agent_out, map_feat, map_mask)

        # Map attends to (updated) agent → enriches map with agent context
        map_out = map_feat
        for layer in self.map_to_agent:
            map_out = layer(map_out, agent_out, agent_mask)

        return agent_out, map_out


# ---------------------------------------------------------------------------
# Neighbor Interaction Attention
# ---------------------------------------------------------------------------

class NeighborAttention(nn.Module):
    """
    Attention between the target agent and surrounding neighbor agents.

    Target agent attends to neighbors to capture social interactions
    (following, overtaking, collision avoidance, etc.).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cross_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        target_feat: torch.Tensor,
        neighbor_feat: torch.Tensor,
        neighbor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            target_feat:   [batch, 1, d_model] — target agent token
            neighbor_feat: [batch, max_neighbors, d_model] — neighbor tokens
            neighbor_mask: [batch, max_neighbors]
        Returns:
            [batch, 1, d_model] — target agent enriched with social context
        """
        out = target_feat
        for layer in self.cross_layers:
            out = layer(out, neighbor_feat, neighbor_mask)
        return out
