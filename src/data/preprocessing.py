"""
Preprocessing utilities for Argoverse 1 data.

Handles:
  - Vectorizing raw Argoverse scenes into model-ready tensors
  - Coordinate normalization (agent-centric frame)
  - Lane graph extraction from HD maps
  - Neighbor selection and padding
"""

import numpy as np
from typing import Any

from .features import (
    extract_agent_features,
    extract_lane_features,
    compute_agent_reference_frame,
    normalize_trajectories,
    normalize_lanes,
)


# ---------------------------------------------------------------------------
# Scene Vectorization
# ---------------------------------------------------------------------------

def vectorize_lanes(
    city_map: Any,
    query_xy: np.ndarray,
    radius: float = 100.0,
    max_lanes: int = 50,
    max_points: int = 20,
    resolution: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract lane polylines near a query point from the HD map.

    Uses Argoverse API's lane graph to find nearby lane segments.

    Args:
        city_map: Argoverse map object with lane graph
        query_xy: [2] — query point in world coordinates
        radius: search radius (meters)
        max_lanes: max number of lane segments to return
        max_points: max points per polyline
        resolution: sampling resolution (meters)

    Returns:
        lanes: [max_lanes, max_points, 5] — padded lane features
        mask:  [max_lanes, max_points] — valid point mask
    """
    lanes = np.zeros((max_lanes, max_points, 5), dtype=np.float32)
    mask = np.zeros((max_lanes, max_points), dtype=bool)

    try:
        # Query nearby lane segments via Argoverse API
        lane_ids = city_map.get_lane_ids_in_xy_bbox(
            query_xy[0], query_xy[1], city_name="MIA", query_search_range_manhattan=radius
        )
    except Exception:
        # Fallback: empty lanes if map API unavailable
        return lanes, mask

    lane_count = 0
    for lid in lane_ids[:max_lanes]:
        try:
            centerline = city_map.get_lane_segment_centerline(lid, city_name="MIA")
            if centerline is None or len(centerline) < 2:
                continue

            lane_type = 0  # Could query lane type from map metadata
            features = extract_lane_features(
                centerline, lane_type=lane_type,
                resolution=resolution, max_points=max_points,
            )
            P = min(len(features), max_points)
            lanes[lane_count, :P] = features[:P]
            mask[lane_count, :P] = True
            lane_count += 1
        except Exception:
            continue

    return lanes, mask


# ---------------------------------------------------------------------------
# Agent Feature Extraction from Scene
# ---------------------------------------------------------------------------

def extract_target_and_neighbors(
    scene_data: dict[str, Any],
    obs_len: int = 20,
    max_neighbors: int = 20,
    neighbor_radius: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract target agent and neighbor agent features from an Argoverse scene.

    Args:
        scene_data: dict with keys:
            "target_track": [T_obs + T_future, 2] world positions
            "target_timestamps": [T_obs + T_future] timestamps
            "neighbor_tracks": list of (track, timestamps) tuples
            "obs_len": observation horizon length
        obs_len: number of observation timesteps
        max_neighbors: max number of neighbors
        neighbor_radius: radius to include neighbors (meters)

    Returns:
        target_features:  [obs_len, 6]
        target_gt:        [future_len, 2] ground-truth future
        target_mask:      [obs_len] valid history timesteps
        neighbor_features:[max_neighbors, obs_len, 6]
        neighbor_mask:    [max_neighbors]
        future_gt_mask:   [future_len] valid future timesteps
    """
    target_track = np.array(scene_data["target_track"], dtype=np.float32)
    target_ts = np.array(scene_data["target_timestamps"], dtype=np.float32)

    T_total = len(target_track)
    T_obs = obs_len
    T_future = T_total - T_obs

    # Split into observation and future
    obs_track = target_track[:T_obs]
    future_track = target_track[T_obs:]

    # Reference frame from last observation point
    origin, heading, rotation = compute_agent_reference_frame(obs_track)

    # Normalize observation and future
    obs_normalized = normalize_trajectories(obs_track, origin, rotation)
    future_normalized = normalize_trajectories(future_track, origin, rotation)

    # Extract features for observation
    target_features = extract_agent_features(
        obs_normalized, target_ts[:T_obs], obs_len
    )
    target_mask = np.ones(obs_len, dtype=bool)
    # Mark padded steps as invalid
    if T_obs < obs_len:
        target_mask[:obs_len - T_obs] = False

    # Ground truth future (displacements from origin)
    target_gt = future_normalized  # [T_future, 2]
    future_len = target_gt.shape[0]
    future_gt_mask = np.ones(future_len, dtype=bool)

    # ── Neighbors ──
    neighbor_tracks_list = scene_data.get("neighbor_tracks", [])
    neighbor_features = np.zeros((max_neighbors, obs_len, 6), dtype=np.float32)
    neighbor_mask = np.zeros(max_neighbors, dtype=bool)

    nbr_count = 0
    target_final_pos = obs_track[-1]

    for nbr_track, nbr_ts in neighbor_tracks_list:
        if nbr_count >= max_neighbors:
            break

        nbr_track = np.array(nbr_track, dtype=np.float32)
        nbr_ts = np.array(nbr_ts, dtype=np.float32)

        # Check distance at last observation
        if len(nbr_track) > T_obs:
            dist = np.linalg.norm(nbr_track[T_obs - 1] - target_final_pos)
        elif len(nbr_track) > 0:
            dist = np.linalg.norm(nbr_track[-1] - target_final_pos)
        else:
            continue

        if dist > neighbor_radius:
            continue

        # Normalize neighbor trajectory
        nbr_obs = nbr_track[:min(len(nbr_track), T_obs)]
        nbr_obs_norm = normalize_trajectories(nbr_obs, origin, rotation)

        feat = extract_agent_features(nbr_obs_norm, nbr_ts[:len(nbr_obs)], obs_len)
        neighbor_features[nbr_count] = feat
        neighbor_mask[nbr_count] = True
        nbr_count += 1

    return (
        target_features,
        target_gt,
        target_mask,
        neighbor_features,
        neighbor_mask,
        future_gt_mask,
    )


# ---------------------------------------------------------------------------
# Full Scene Preprocessing
# ---------------------------------------------------------------------------

def preprocess_scene(
    scene_data: dict[str, Any],
    city_map: Any = None,
    config: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """
    Preprocess a single Argoverse scene into model inputs.

    Args:
        scene_data: raw scene dict from Argoverse dataset
        city_map: Argoverse map object (optional)
        config: configuration dict

    Returns:
        dict with keys:
            target_traj, target_mask, gt_future, gt_mask,
            neighbor_trajs, neighbor_mask, lane_polylines, lane_mask
    """
    cfg = config or {}
    data_cfg = cfg.get("data", {})

    obs_len = data_cfg.get("history_steps", 20)
    max_neighbors = data_cfg.get("max_neighbors", 20)
    max_lanes = data_cfg.get("max_lane_segments", 50)
    lane_resolution = data_cfg.get("lane_resolution", 5.0)
    lane_radius = data_cfg.get("lane_radius", 100.0)
    neighbor_radius = data_cfg.get("neighbor_radius", 50.0)

    # Extract agents
    (
        target_features, target_gt, target_mask,
        neighbor_features, neighbor_mask, future_gt_mask,
    ) = extract_target_and_neighbors(
        scene_data, obs_len=obs_len, max_neighbors=max_neighbors,
        neighbor_radius=neighbor_radius,
    )

    # Pad future GT to 30 steps
    future_steps = data_cfg.get("future_steps", 30)
    gt_future = np.zeros((future_steps, 2), dtype=np.float32)
    gt_mask = np.zeros(future_steps, dtype=bool)
    T_future = min(len(target_gt), future_steps)
    gt_future[:T_future] = target_gt[:T_future]
    gt_mask[:T_future] = True

    # Extract lanes
    if city_map is not None:
        # Use the LAST observed position as query point (world frame)
        # We stored it in scene_data
        query_point = scene_data.get("target_track", np.zeros((1, 2)))[obs_len - 1]
        lane_features, lane_mask = vectorize_lanes(
            city_map, query_point,
            radius=lane_radius,
            max_lanes=max_lanes,
            resolution=lane_resolution,
        )

        # Normalize lanes to agent frame
        obs_track = np.array(scene_data["target_track"][:obs_len], dtype=np.float32)
        origin, heading, rotation = compute_agent_reference_frame(obs_track)
        lane_features = normalize_lanes(lane_features, origin, rotation)
    else:
        lane_features = np.zeros((max_lanes, 20, 5), dtype=np.float32)
        lane_mask = np.zeros((max_lanes, 20), dtype=bool)

    return {
        "target_traj": target_features.astype(np.float32),    # [20, 6]
        "target_mask": target_mask,                             # [20]
        "gt_future": gt_future.astype(np.float32),             # [30, 2]
        "gt_mask": gt_mask,                                     # [30]
        "neighbor_trajs": neighbor_features.astype(np.float32),# [N, 20, 6]
        "neighbor_mask": neighbor_mask,                         # [N]
        "lane_polylines": lane_features.astype(np.float32),    # [L, P, 5]
        "lane_mask": lane_mask,                                 # [L, P]
    }
