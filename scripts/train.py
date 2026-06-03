#!/usr/bin/env python3
"""
Training script for MGFNet trajectory prediction model.

Usage:
    python scripts/train.py --config configs/default.yaml --data_dir data/argoverse
    python scripts/train.py --config configs/default.yaml --synthetic
"""

import argparse
import os
import sys

# Ensure the project root is on the Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import yaml
import torch
from torch.utils.data import DataLoader

from src.model.mgfnet import MGFNet
from src.training.trainer import Trainer
from src.data.dataset import ArgoverseDataset, SyntheticTrajectoryDataset


def main():
    parser = argparse.ArgumentParser(description="Train MGFNet trajectory prediction model")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--data_dir", type=str, default="data/argoverse",
                        help="Path to Argoverse 1 data directory")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (no Argoverse dataset required)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cuda', 'cpu', or 'auto'")
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(PROJECT_ROOT, args.config)
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 64)
    num_workers = train_cfg.get("num_workers", 4)
    epochs = train_cfg.get("epochs", 100)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    print(f"Config: {config_path}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")

    # ── Dataset ──
    if args.synthetic or not os.path.isdir(args.data_dir):
        print("Using synthetic dataset (--synthetic flag or data_dir not found)")
        train_dataset = SyntheticTrajectoryDataset(
            num_samples=5000,
            config=config,
        )
        val_dataset = SyntheticTrajectoryDataset(
            num_samples=500,
            config=config,
        )
    else:
        print(f"Loading Argoverse dataset from: {args.data_dir}")
        train_dataset = ArgoverseDataset(
            data_dir=args.data_dir,
            split="train",
            config=config,
        )
        val_dataset = ArgoverseDataset(
            data_dir=args.data_dir,
            split="val",
            config=config,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # ── Model ──
    model = MGFNet(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── Resume ──
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        start_epoch = 0

    # ── Trainer ──
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )
    trainer.current_epoch = start_epoch

    # ── Train ──
    trainer.train()


if __name__ == "__main__":
    main()
