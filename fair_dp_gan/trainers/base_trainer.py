"""
Base Trainer Class

Provides common functionality for all Fair-DP-GAN trainers.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


class BaseTrainer:
    """
    Base class for all GAN trainers.

    Provides common functionality:
    - Model management
    - Optimization setup
    - Training loop structure
    - Logging and checkpointing

    Args:
        generator: Generator model
        critic: Critic model
        device: Device to train on
        lr_g: Learning rate for generator (default: 0.0002)
        lr_c: Learning rate for critic (default: 0.0002)
        beta1: Adam beta1 parameter (default: 0.5)
        beta2: Adam beta2 parameter (default: 0.999)
        n_critic: Number of critic updates per generator update (default: 5)
    """

    def __init__(
        self,
        generator: nn.Module,
        critic: nn.Module,
        device: str = 'cuda',
        lr_g: float = 0.0002,
        lr_c: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
        n_critic: int = 5
    ):
        self.generator = generator.to(device)
        self.critic = critic.to(device)
        self.device = device
        self.n_critic = n_critic

        # Optimizers
        self.optimizer_g = torch.optim.Adam(
            generator.parameters(),
            lr=lr_g,
            betas=(beta1, beta2)
        )
        self.optimizer_c = torch.optim.Adam(
            critic.parameters(),
            lr=lr_c,
            betas=(beta1, beta2)
        )

        # Training statistics
        self.losses_g = []
        self.losses_c = []
        self.metrics_history = []

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Should be implemented by subclasses.

        Returns:
            Dict of metric_name -> value
        """
        raise NotImplementedError

    def train(
        self,
        dataloader: DataLoader,
        n_epochs: int,
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        Full training loop.

        Args:
            dataloader: Training data loader
            n_epochs: Number of epochs to train
            verbose: Whether to print progress

        Returns:
            Training history
        """
        for epoch in range(n_epochs):
            metrics = self.train_epoch(dataloader, epoch)

            if verbose:
                metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                print(f"Epoch {epoch+1}/{n_epochs} - {metric_str}")

            self.metrics_history.append(metrics)

        return {
            'losses_g': self.losses_g,
            'losses_c': self.losses_c,
            'metrics': self.metrics_history
        }

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'generator': self.generator.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_c': self.optimizer_c.state_dict(),
            'losses_g': self.losses_g,
            'losses_c': self.losses_c,
            'metrics_history': self.metrics_history
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        self.optimizer_c.load_state_dict(checkpoint['optimizer_c'])
        self.losses_g = checkpoint['losses_g']
        self.losses_c = checkpoint['losses_c']
        self.metrics_history = checkpoint['metrics_history']
