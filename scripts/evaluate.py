#!/usr/bin/env python3
"""
Evaluation script for MGFNet trajectory prediction model.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --data_dir data/argoverse
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --synthetic
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.mgfnet import MGFNet
from src.data.dataset import ArgoverseDataset, SyntheticTrajectoryDataset
from src.evaluation.metrics import compute_metrics, compute_per_mode_metrics
from src.evaluation.visualize import plot_trajectories


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Evaluate MGFNet model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/argoverse",
                        help="Path to Argoverse 1 data directory")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_viz", type=int, default=5,
                        help="Number of samples to visualize")
    parser.add_argument("--viz_dir", type=str, default="visualizations",
                        help="Directory to save visualizations")
    parser.add_argument("--mr_threshold", type=float, default=2.0,
                        help="Miss rate threshold (meters)")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Config
    config_path = os.path.join(PROJECT_ROOT, args.config)
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # ── Load checkpoint ──
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", config)

    model = MGFNet(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── Dataset ──
    if args.synthetic or not os.path.isdir(args.data_dir):
        print("Using synthetic dataset")
        dataset = SyntheticTrajectoryDataset(num_samples=1000, config=config)
    else:
        print(f"Loading Argoverse dataset from: {args.data_dir}")
        dataset = ArgoverseDataset(data_dir=args.data_dir, split="val", config=config)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    print(f"Evaluation samples: {len(dataset)}")

    # ── Evaluate ──
    all_trajectories = []
    all_confidences = []
    all_gt_future = []
    all_gt_mask = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            trajectories, confidences = model(
                target_traj=batch["target_traj"],
                neighbor_trajs=batch["neighbor_trajs"],
                lane_polylines=batch["lane_polylines"],
                target_mask=batch.get("target_mask"),
                neighbor_mask=batch.get("neighbor_mask"),
                lane_mask=batch.get("lane_mask"),
            )

            all_trajectories.append(trajectories.cpu())
            all_confidences.append(confidences.cpu())
            all_gt_future.append(batch["gt_future"].cpu())
            if "gt_mask" in batch:
                all_gt_mask.append(batch["gt_mask"].cpu())

    trajectories = torch.cat(all_trajectories, dim=0)
    confidences = torch.cat(all_confidences, dim=0)
    gt_future = torch.cat(all_gt_future, dim=0)
    gt_mask = torch.cat(all_gt_mask, dim=0) if all_gt_mask else None

    # ── Compute metrics ──
    metrics = compute_metrics(
        trajectories, confidences, gt_future, gt_mask,
        mr_threshold=args.mr_threshold,
    )
    per_mode = compute_per_mode_metrics(trajectories, gt_future, gt_mask)

    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"  minADE:       {metrics['minADE']:.4f} m")
    print(f"  minFDE:       {metrics['minFDE']:.4f} m")
    print(f"  Miss Rate:    {metrics['MR']:.4f} (threshold={args.mr_threshold}m)")
    print(f"  brier-minFDE: {metrics['brier_minFDE']:.4f}")
    print(f"\n  Per-mode ADE: {per_mode['ade_per_mode']}")
    print(f"  Per-mode FDE: {per_mode['fde_per_mode']}")
    print("=" * 50)

    # ── Visualize ──
    os.makedirs(args.viz_dir, exist_ok=True)
    viz_indices = np.random.choice(len(dataset), min(args.num_viz, len(dataset)), replace=False)

    for i, idx in enumerate(viz_indices):
        traj_np = trajectories[idx].numpy()
        conf_np = confidences[idx].numpy()
        gt_np = gt_future[idx].numpy()

        fig = plot_trajectories(
            trajectories=traj_np,
            confidences=conf_np,
            gt_future=gt_np,
            title=f"Sample {idx} — minADE: {metrics['minADE']:.3f}m",
            save_path=os.path.join(args.viz_dir, f"sample_{idx:04d}.png"),
        )
        plt.close(fig)

    print(f"\nSaved {len(viz_indices)} visualizations to {args.viz_dir}/")


if __name__ == "__main__":
    main()
