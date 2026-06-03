"""
Argoverse 1 Motion Forecasting Dataset.

PyTorch Dataset that loads Argoverse 1 data, preprocesses scenes into
vectorized agent/lane features in agent-centric coordinates, and returns
model-ready tensors.

Supports:
  - Training split (with ground truth)
  - Validation split
  - On-the-fly preprocessing (with optional caching)
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any

from .preprocessing import preprocess_scene


class ArgoverseDataset(Dataset):
    """
    Argoverse 1 Motion Forecasting Dataset.

    Expected data directory structure:
        data/argoverse/
        ├── train/
        │   ├── data/
        │   │   └── *.pkl    # Per-scene pickle files
        │   └── map/         # HD map data
        └── val/
            ├── data/
            └── map/

    Each .pkl file should contain a dict with at minimum:
        {
            "target_track": np.ndarray [T_total, 2],
            "target_timestamps": np.ndarray [T_total],
            "neighbor_tracks": list of (track [T,2], timestamps [T]),
            "scene_id": str,
        }
    """

    def __init__(
        self,
        data_dir: str = "data/argoverse",
        split: str = "train",
        config: dict[str, Any] | None = None,
        cache_preprocessed: bool = False,
    ):
        """
        Args:
            data_dir: root data directory
            split: "train" or "val"
            config: model/data config dict
            cache_preprocessed: if True, cache preprocessed scenes in memory
        """
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.config = config or {}
        self.cache_preprocessed = cache_preprocessed

        data_cfg = self.config.get("data", {})
        self.data_subdir = os.path.join(data_dir, split, "data")
        self.map_dir = os.path.join(data_dir, split, "map")

        # Collect scene files
        self.scene_files = sorted([
            f for f in os.listdir(self.data_subdir)
            if f.endswith(".pkl") or f.endswith(".pkl")
        ])
        if len(self.scene_files) == 0:
            # Try flat structure
            alt_dir = os.path.join(data_dir, split)
            if os.path.isdir(alt_dir):
                self.scene_files = sorted([
                    f for f in os.listdir(alt_dir)
                    if f.endswith(".pkl")
                ])
                self.data_subdir = alt_dir

        self.cache: dict[int, dict[str, np.ndarray]] = {}

        # Lazy-load map
        self._city_map = None

    def __len__(self) -> int:
        return len(self.scene_files)

    def _load_scene(self, idx: int) -> dict[str, Any]:
        """Load raw scene data from disk."""
        filepath = os.path.join(self.data_subdir, self.scene_files[idx])
        with open(filepath, "rb") as f:
            scene_data = pickle.load(f)
        return scene_data

    def _get_map(self) -> Any:
        """Lazy-load Argoverse city map."""
        if self._city_map is None:
            try:
                from argoverse.map_representation.map_api import ArgoverseMap
                self._city_map = ArgoverseMap()
            except ImportError:
                print("Warning: argoverse map API not available. Lane features disabled.")
                self._city_map = None
        return self._city_map

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict with keys:
                target_traj:    [20, 6] float
                target_mask:    [20] bool
                gt_future:      [30, 2] float
                gt_mask:        [30] bool
                neighbor_trajs: [N, 20, 6] float
                neighbor_mask:  [N] bool
                lane_polylines: [L, P, 5] float
                lane_mask:      [L, P] bool
                scene_id:       str
        """
        # Check cache
        if self.cache_preprocessed and idx in self.cache:
            return self._to_tensors(self.cache[idx])

        # Load and preprocess
        scene_data = self._load_scene(idx)
        city_map = self._get_map()

        preprocessed = preprocess_scene(scene_data, city_map, self.config)
        preprocessed["scene_id"] = scene_data.get("scene_id", self.scene_files[idx])

        if self.cache_preprocessed:
            self.cache[idx] = preprocessed

        return self._to_tensors(preprocessed)

    @staticmethod
    def _to_tensors(data: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        """Convert numpy arrays to torch tensors."""
        result = {}
        for key, val in data.items():
            if isinstance(val, np.ndarray):
                if val.dtype == bool:
                    result[key] = torch.from_numpy(val)
                else:
                    result[key] = torch.from_numpy(val)
            else:
                result[key] = val
        return result

    @staticmethod
    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """
        Custom collate function that stacks tensors across batch dim.
        Handles variable-length neighbor/lane counts via padding.
        """
        collated = {}

        for key in batch[0].keys():
            if key == "scene_id":
                collated[key] = [item[key] for item in batch]
            else:
                values = [item[key] for item in batch]
                # Stack along batch dimension
                if isinstance(values[0], torch.Tensor):
                    collated[key] = torch.stack(values, dim=0)
                else:
                    collated[key] = values

        return collated


# ---------------------------------------------------------------------------
# Synthetic Dataset for testing / prototyping without Argoverse data
# ---------------------------------------------------------------------------

class SyntheticTrajectoryDataset(Dataset):
    """
    Generates synthetic trajectory data for model development and testing.

    Produces plausible vehicle trajectories with lane-following behavior
    and random lane topology. Useful when Argoverse data is not available.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        config: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.config = config or {}
        data_cfg = self.config.get("data", {})

        self.history_steps = data_cfg.get("history_steps", 20)
        self.future_steps = data_cfg.get("future_steps", 30)
        self.max_neighbors = data_cfg.get("max_neighbors", 20)
        self.max_lanes = data_cfg.get("max_lane_segments", 50)

        # Pre-generate all data
        self.data: list[dict[str, np.ndarray]] = []
        for _ in range(num_samples):
            self.data.append(self._generate_scene())

    def _generate_scene(self) -> dict[str, np.ndarray]:
        """Generate one synthetic scene."""
        rng = np.random.RandomState()

        # Target agent: constant velocity with slight curvature
        v = rng.uniform(5.0, 15.0)  # m/s
        heading = rng.uniform(-0.5, 0.5)  # initial heading
        turn_rate = rng.uniform(-0.05, 0.05)  # radians per step

        dt = 0.1  # 10 Hz
        T_total = self.history_steps + self.future_steps

        positions = np.zeros((T_total, 2), dtype=np.float32)
        current_pos = np.array([0.0, 0.0], dtype=np.float32)
        current_heading = heading

        for t in range(T_total):
            # Add noise
            noise = rng.normal(0, 0.1, 2).astype(np.float32)
            current_pos[0] += v * np.cos(current_heading) * dt + noise[0] * 0.1
            current_pos[1] += v * np.sin(current_heading) * dt + noise[1] * 0.1
            current_heading += turn_rate * dt
            positions[t] = current_pos.copy()

        # Target features
        target_traj = np.zeros((self.history_steps, 6), dtype=np.float32)
        target_mask = np.ones(self.history_steps, dtype=bool)
        for t in range(self.history_steps):
            target_traj[t, 0] = positions[t, 0]
            target_traj[t, 1] = positions[t, 1]
            if t > 0:
                vx = (positions[t, 0] - positions[t - 1, 0]) / dt
                vy = (positions[t, 1] - positions[t - 1, 1]) / dt
                target_traj[t, 2] = vx
                target_traj[t, 3] = vy
                target_traj[t, 4] = np.arctan2(vy, vx)
            target_traj[t, 5] = dt

        # Ground truth future
        gt_future = np.zeros((self.future_steps, 2), dtype=np.float32)
        gt_mask = np.ones(self.future_steps, dtype=bool)
        for t in range(self.future_steps):
            gt_future[t] = positions[self.history_steps + t]

        # Neighbors (few random agents)
        n_neighbors = rng.randint(1, 5)
        neighbor_trajs = np.zeros((self.max_neighbors, self.history_steps, 6), dtype=np.float32)
        neighbor_mask = np.zeros(self.max_neighbors, dtype=bool)

        for n in range(min(n_neighbors, self.max_neighbors)):
            offset = rng.uniform(-20, 20, 2).astype(np.float32)
            nbr_v = rng.uniform(3.0, 18.0)
            nbr_h = rng.uniform(-np.pi, np.pi)
            for t in range(self.history_steps):
                neighbor_trajs[n, t, 0] = offset[0] + nbr_v * np.cos(nbr_h) * t * dt
                neighbor_trajs[n, t, 1] = offset[1] + nbr_v * np.sin(nbr_h) * t * dt
                neighbor_trajs[n, t, 2] = nbr_v * np.cos(nbr_h)
                neighbor_trajs[n, t, 3] = nbr_v * np.sin(nbr_h)
                neighbor_trajs[n, t, 4] = nbr_h
                neighbor_trajs[n, t, 5] = dt
            neighbor_mask[n] = True

        # Synthetic lanes
        n_lanes = rng.randint(5, 15)
        lane_polylines = np.zeros((self.max_lanes, 20, 5), dtype=np.float32)
        lane_mask = np.zeros((self.max_lanes, 20), dtype=bool)

        for l in range(min(n_lanes, self.max_lanes)):
            lane_offset = rng.uniform(-30, 30)
            lane_h = rng.uniform(-0.3, 0.3)
            n_pts = rng.randint(5, 20)
            for p in range(n_pts):
                x = p * 5.0 - 30.0
                y = lane_offset + np.sin(lane_h) * x
                lane_polylines[l, p, 0] = x
                lane_polylines[l, p, 1] = y
                lane_polylines[l, p, 2] = 1.0  # dx
                lane_polylines[l, p, 3] = 0.0  # dy
                lane_polylines[l, p, 4] = 0.0  # lane type
                lane_mask[l, p] = True

        return {
            "target_traj": target_traj,
            "target_mask": target_mask,
            "gt_future": gt_future,
            "gt_mask": gt_mask,
            "neighbor_trajs": neighbor_trajs,
            "neighbor_mask": neighbor_mask,
            "lane_polylines": lane_polylines,
            "lane_mask": lane_mask,
            "scene_id": f"synth_{rng.randint(0, 1_000_000)}",
        }

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = self.data[idx]
        return {
            k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
            for k, v in data.items()
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        collated = {}
        for key in batch[0].keys():
            if key == "scene_id":
                collated[key] = [item[key] for item in batch]
            else:
                collated[key] = torch.stack([item[key] for item in batch], dim=0)
        return collated
