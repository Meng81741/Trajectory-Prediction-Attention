"""
Feature extraction utilities for Argoverse 1 Motion Forecasting.

Extracts vectorized features from raw Argoverse data:
  - Agent features: position, velocity, heading, timestamp
  - Lane features: polyline points with direction and type
  - Coordinate normalization: agent-centric frame
"""

import numpy as np
from typing import Any


# ---------------------------------------------------------------------------
# Agent Feature Extraction
# ---------------------------------------------------------------------------

def extract_agent_features(
    track: np.ndarray,
    timestamps: np.ndarray,
    obs_len: int = 20,
) -> np.ndarray:
    """
    Extract per-timestep features for a single agent.

    Args:
        track: [T, 2] — (x, y) positions in world frame
        timestamps: [T] — timestamps for each step
        obs_len: number of observation steps to use

    Returns:
        features: [obs_len, 6] — (x, y, vx, vy, heading, Δt)
    """
    T = min(len(track), obs_len)
    features = np.zeros((obs_len, 6), dtype=np.float32)

    # Pad at the beginning if track is too short
    offset = obs_len - T

    for t in range(T):
        idx = offset + t
        features[idx, 0] = track[t, 0]  # x
        features[idx, 1] = track[t, 1]  # y

        if t > 0:
            dt = (timestamps[t] - timestamps[t - 1]) / 1e9  # ns → seconds
            features[idx, 2] = (track[t, 0] - track[t - 1, 0]) / max(dt, 0.01)  # vx
            features[idx, 3] = (track[t, 1] - track[t - 1, 1]) / max(dt, 0.01)  # vy
            features[idx, 4] = np.arctan2(features[idx, 3], features[idx, 2])  # heading
            features[idx, 5] = dt  # Δt
        elif t == 0 and T > 1:
            # Estimate from next step
            dt_next = (timestamps[1] - timestamps[0]) / 1e9
            features[idx, 2] = (track[1, 0] - track[0, 0]) / max(dt_next, 0.01)
            features[idx, 3] = (track[1, 1] - track[0, 1]) / max(dt_next, 0.01)
            features[idx, 4] = np.arctan2(features[idx, 3], features[idx, 2])
            features[idx, 5] = 0.1  # default 10Hz
        else:
            features[idx, 4] = 0.0
            features[idx, 5] = 0.1

        # Carry forward heading for padded timesteps
        if t == 0 and offset > 0:
            for p in range(offset):
                features[p, 2:6] = features[offset, 2:6]

    return features


# ---------------------------------------------------------------------------
# Lane Feature Extraction
# ---------------------------------------------------------------------------

def extract_lane_features(
    lane_centerline: np.ndarray,
    lane_type: int = 0,
    resolution: float = 5.0,
    max_points: int = 20,
) -> np.ndarray:
    """
    Extract polyline features for a single lane segment.

    Args:
        lane_centerline: [N, 2] — (x, y) points along lane centerline
        lane_type: integer encoding lane type (0=unknown, 1=double-yellow, etc.)
        resolution: spacing between sampled points (meters)
        max_points: maximum number of points in output polyline

    Returns:
        features: [max_points, 5] — (x, y, dx, dy, lane_type) per point
    """
    points = resample_polyline(lane_centerline, resolution)
    P = min(len(points), max_points)
    features = np.zeros((max_points, 5), dtype=np.float32)

    for i in range(P):
        features[i, 0] = points[i, 0]
        features[i, 1] = points[i, 1]

        if i < P - 1:
            dx = points[i + 1, 0] - points[i, 0]
            dy = points[i + 1, 1] - points[i, 1]
            norm = np.sqrt(dx**2 + dy**2) + 1e-8
            features[i, 2] = dx / norm
            features[i, 3] = dy / norm
        else:
            # Last point: use same direction as previous segment
            features[i, 2] = features[i - 1, 2] if i > 0 else 0.0
            features[i, 3] = features[i - 1, 3] if i > 0 else 0.0

        features[i, 4] = float(lane_type)

    return features


def resample_polyline(
    polyline: np.ndarray,
    resolution: float = 5.0,
) -> np.ndarray:
    """
    Resample a polyline to uniform spacing.

    Args:
        polyline: [N, 2]
        resolution: desired spacing between consecutive points

    Returns:
        resampled: [M, 2] with uniform spacing
    """
    if len(polyline) < 2:
        return polyline

    # Compute cumulative arc length
    diffs = np.diff(polyline, axis=0)
    seg_lengths = np.sqrt((diffs**2).sum(axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])

    total_length = cumulative[-1]
    if total_length < resolution:
        return polyline

    num_samples = int(np.ceil(total_length / resolution)) + 1
    sample_dists = np.linspace(0, total_length, num_samples)

    # Interpolate
    resampled = np.zeros((num_samples, 2), dtype=np.float32)
    for d in range(2):
        resampled[:, d] = np.interp(sample_dists, cumulative, polyline[:, d])

    return resampled


# ---------------------------------------------------------------------------
# Coordinate Normalization
# ---------------------------------------------------------------------------

def compute_agent_reference_frame(
    track: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Compute agent-centric reference frame.

    The frame is centered at the agent's position at the LAST observation
    timestep, with the x-axis aligned to the agent's heading.

    Args:
        track: [T, 2] — agent positions in world frame

    Returns:
        origin: [2] — reference frame origin (last observed position)
        heading: float — reference frame heading (radians)
        rotation: [2, 2] — rotation matrix from world to agent frame
    """
    origin = track[-1].copy()
    if len(track) >= 2:
        heading_vec = track[-1] - track[-2]
        heading = np.arctan2(heading_vec[1], heading_vec[0])
    else:
        heading = 0.0

    cos_h = np.cos(-heading)
    sin_h = np.sin(-heading)
    rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float32)

    return origin, heading, rotation


def normalize_trajectories(
    trajectories: np.ndarray,
    origin: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    """
    Transform trajectories from world frame to agent-centric frame.

    Args:
        trajectories: [..., 2] — positions in world frame (x, y last dim)
        origin: [2] — reference origin
        rotation: [2, 2] — rotation matrix

    Returns:
        [..., 2] — positions in agent-centric frame
    """
    translated = trajectories - origin
    return translated @ rotation.T


def normalize_lanes(
    lanes: np.ndarray,
    origin: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    """
    Transform lane polylines from world frame to agent-centric frame.

    Args:
        lanes: [L, P, 5] — lane features (x, y, dx, dy, type) in world frame
        origin: [2]
        rotation: [2, 2]

    Returns:
        [L, P, 5] — lane features in agent-centric frame (directions also rotated)
    """
    normalized = lanes.copy()
    # Rotate and translate positions
    pos = normalized[..., :2]  # [L, P, 2]
    pos = (pos - origin) @ rotation.T
    normalized[..., :2] = pos
    # Rotate direction vectors
    dirs = normalized[..., 2:4]  # [L, P, 2]
    dirs = dirs @ rotation.T
    normalized[..., 2:4] = dirs
    return normalized
