"""
Training loop for MGFNet trajectory prediction model.

Features:
  - Gradient clipping
  - Learning rate warmup + cosine annealing
  - TensorBoard logging
  - Periodic validation
  - Checkpoint saving (best + periodic)
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.model.mgfnet import MGFNet
from src.training.losses import MultiModalLoss


class Trainer:
    """MGFNet training loop."""

    def __init__(
        self,
        config: dict[str, Any],
        model: MGFNet,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        device: torch.device | None = None,
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_cfg = config.get("training", {})

        self.epochs = train_cfg.get("epochs", 100)
        self.warmup_epochs = train_cfg.get("warmup_epochs", 10)
        self.gradient_clip = train_cfg.get("gradient_clip", 1.0)

        # Optimizer
        lr = train_cfg.get("learning_rate", 0.001)
        wd = train_cfg.get("weight_decay", 0.0001)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd
        )

        # Scheduler: linear warmup + cosine annealing
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs - self.warmup_epochs
        )

        # Loss
        self.criterion = MultiModalLoss(
            regression_weight=train_cfg.get("regression_weight", 1.0),
            confidence_weight=1.0,
            diversity_weight=train_cfg.get("diversity_weight", 0.1),
        )

        # Logging
        self.log_dir = train_cfg.get("log_dir", "logs/mgfnet")
        self.writer = SummaryWriter(self.log_dir)
        self.checkpoint_dir = train_cfg.get("checkpoint_dir", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # State
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.global_step = 0

        self.model.to(self.device)

    def train(self) -> None:
        """Full training loop."""
        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self._train_epoch()

            # Validate
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self._validate_epoch()

            # Log
            self._log_epoch(epoch, train_metrics, val_metrics)

            # Save checkpoint
            self._save_checkpoint(epoch, val_metrics)

            # Adjust LR
            if epoch >= self.warmup_epochs:
                self.scheduler.step()

        self.writer.close()
        print(f"Training complete. Best val loss: {self.best_val_loss:.4f}")

    def _train_epoch(self) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_reg = 0.0
        total_conf = 0.0
        total_div = 0.0
        total_minADE = 0.0
        total_minFDE = 0.0
        total_MR = 0.0  # miss rate
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.epochs} [Train]")
        for batch in pbar:
            batch = self._to_device(batch)

            # Warmup LR
            if self.current_epoch < self.warmup_epochs:
                lr_scale = (self.current_epoch + 1) / max(self.warmup_epochs, 1)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = pg["lr"] * lr_scale

            # Forward
            trajectories, confidences = self.model(
                target_traj=batch["target_traj"],
                neighbor_trajs=batch["neighbor_trajs"],
                lane_polylines=batch["lane_polylines"],
                target_mask=batch.get("target_mask"),
                neighbor_mask=batch.get("neighbor_mask"),
                lane_mask=batch.get("lane_mask"),
            )

            # Loss
            loss_dict = self.criterion(
                trajectories, confidences,
                batch["gt_future"], batch.get("gt_mask"),
            )

            # Backward
            self.optimizer.zero_grad()
            loss_dict["total_loss"].backward()

            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )

            self.optimizer.step()

            # Accumulate
            total_loss += loss_dict["total_loss"].item()
            total_reg += loss_dict["reg_loss"].item()
            total_conf += loss_dict["conf_loss"].item()
            total_div += loss_dict["div_loss"].item()
            total_minADE += loss_dict["minADE"].item()
            total_minFDE += loss_dict["minFDE"].item()
            total_MR += loss_dict["MR"].item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss_dict['total_loss'].item():.3f}",
                "minADE": f"{loss_dict['minADE'].item():.3f}",
            })

            # Periodic logging
            if self.global_step % 100 == 0:
                self.writer.add_scalar("train/loss_step", loss_dict["total_loss"].item(), self.global_step)
                self.writer.add_scalar("train/minADE_step", loss_dict["minADE"].item(), self.global_step)

        return {
            "loss": total_loss / max(num_batches, 1),
            "reg_loss": total_reg / max(num_batches, 1),
            "conf_loss": total_conf / max(num_batches, 1),
            "div_loss": total_div / max(num_batches, 1),
            "minADE": total_minADE / max(num_batches, 1),
            "minFDE": total_minFDE / max(num_batches, 1),
            "MR": total_MR / max(num_batches, 1),
        }

    @torch.no_grad()
    def _validate_epoch(self) -> dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        total_minADE = 0.0
        total_minFDE = 0.0
        total_MR = 0.0
        num_batches = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1}/{self.epochs} [Val]")
        for batch in pbar:
            batch = self._to_device(batch)

            trajectories, confidences = self.model(
                target_traj=batch["target_traj"],
                neighbor_trajs=batch["neighbor_trajs"],
                lane_polylines=batch["lane_polylines"],
                target_mask=batch.get("target_mask"),
                neighbor_mask=batch.get("neighbor_mask"),
                lane_mask=batch.get("lane_mask"),
            )

            loss_dict = self.criterion(
                trajectories, confidences,
                batch["gt_future"], batch.get("gt_mask"),
            )

            total_loss += loss_dict["total_loss"].item()
            total_minADE += loss_dict["minADE"].item()
            total_minFDE += loss_dict["minFDE"].item()
            total_MR += loss_dict["MR"].item()
            num_batches += 1

            pbar.set_postfix({
                "val_loss": f"{loss_dict['total_loss'].item():.3f}",
                "minADE": f"{loss_dict['minADE'].item():.3f}",
            })

        return {
            "loss": total_loss / max(num_batches, 1),
            "minADE": total_minADE / max(num_batches, 1),
            "minFDE": total_minFDE / max(num_batches, 1),
            "MR": total_MR / max(num_batches, 1),
        }

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> None:
        """Log epoch metrics to TensorBoard and console."""
        # Console
        lr = self.optimizer.param_groups[0]["lr"]
        print(f"\n--- Epoch {epoch + 1}/{self.epochs} (lr={lr:.6f}) ---")
        print(f"  Train: loss={train_metrics['loss']:.4f}, "
              f"minADE={train_metrics['minADE']:.4f}, "
              f"minFDE={train_metrics['minFDE']:.4f}, "
              f"MR={train_metrics['MR']:.4f}")
        if val_metrics:
            print(f"  Val:   loss={val_metrics['loss']:.4f}, "
                  f"minADE={val_metrics['minADE']:.4f}, "
                  f"minFDE={val_metrics['minFDE']:.4f}, "
                  f"MR={val_metrics['MR']:.4f}")

        # TensorBoard
        for k, v in train_metrics.items():
            self.writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_metrics.items():
            self.writer.add_scalar(f"val/{k}", v, epoch)
        self.writer.add_scalar("lr", lr, epoch)

    def _save_checkpoint(
        self, epoch: int, val_metrics: dict[str, float]
    ) -> None:
        """Save model checkpoint."""
        val_loss = val_metrics.get("loss", float("inf"))

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "val_loss": val_loss,
        }

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save(checkpoint, path)

        # Best checkpoint
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"  → New best model saved (val_loss={val_loss:.4f})")

        # Latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, "latest.pt")
        torch.save(checkpoint, latest_path)

    def _to_device(self, batch: dict) -> dict:
        """Move batch tensors to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    @classmethod
    def from_config(
        cls,
        config_path: str,
        data_dir: str = "data/argoverse",
        device: torch.device | None = None,
    ) -> "Trainer":
        """Create Trainer from a YAML config file."""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        model = MGFNet(config)

        # Determine dataset class
        from src.data.dataset import ArgoverseDataset, SyntheticTrajectoryDataset

        if os.path.isdir(data_dir):
            DatasetClass = ArgoverseDataset
        else:
            print(f"Data dir '{data_dir}' not found. Using synthetic data.")
            DatasetClass = SyntheticTrajectoryDataset

        train_cfg = config.get("training", {})
        batch_size = train_cfg.get("batch_size", 64)
        num_workers = train_cfg.get("num_workers", 4)

        train_dataset = DatasetClass(
            data_dir=data_dir,
            split="train",
            config=config,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=train_dataset.collate_fn,
        )

        val_dataset = DatasetClass(
            data_dir=data_dir,
            split="val",
            config=config,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=val_dataset.collate_fn,
        )

        return cls(config, model, train_loader, val_loader, device)
