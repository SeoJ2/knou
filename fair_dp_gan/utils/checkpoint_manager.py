"""
Checkpoint Manager

Handles saving and loading model checkpoints, configurations, and results.
"""

import os
import json
import torch
from typing import Dict, Any, Optional
from datetime import datetime


class CheckpointManager:
    """
    Manages model checkpoints and experiment results.

    Provides functionality for:
    - Saving/loading model weights
    - Saving/loading configurations
    - Saving experiment results
    - Managing multiple checkpoints

    Args:
        checkpoint_dir: Directory to store checkpoints (default: 'checkpoints')
        max_checkpoints: Maximum number of checkpoints to keep (default: 5)
    """

    def __init__(
        self,
        checkpoint_dir: str = 'checkpoints',
        max_checkpoints: int = 5
    ):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints

        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(
        self,
        model_name: str,
        generator: torch.nn.Module,
        critic: torch.nn.Module,
        optimizer_g: torch.optim.Optimizer,
        optimizer_c: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a complete checkpoint.

        Args:
            model_name: Name of the model
            generator: Generator model
            critic: Critic model
            optimizer_g: Generator optimizer
            optimizer_c: Critic optimizer
            epoch: Current epoch
            metrics: Training metrics
            config: Model configuration

        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{model_name}_epoch{epoch}_{timestamp}.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

        checkpoint = {
            'model_name': model_name,
            'epoch': epoch,
            'timestamp': timestamp,
            'generator_state_dict': generator.state_dict(),
            'critic_state_dict': critic.state_dict(),
            'optimizer_g_state_dict': optimizer_g.state_dict(),
            'optimizer_c_state_dict': optimizer_c.state_dict(),
            'metrics': metrics,
            'config': config or {}
        }

        torch.save(checkpoint, checkpoint_path)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints(model_name)

        print(f"Saved checkpoint to {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: str,
        generator: torch.nn.Module,
        critic: torch.nn.Module,
        optimizer_g: Optional[torch.optim.Optimizer] = None,
        optimizer_c: Optional[torch.optim.Optimizer] = None,
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            generator: Generator model to load weights into
            critic: Critic model to load weights into
            optimizer_g: Generator optimizer (optional)
            optimizer_c: Critic optimizer (optional)
            device: Device to load to

        Returns:
            Checkpoint metadata
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        generator.load_state_dict(checkpoint['generator_state_dict'])
        critic.load_state_dict(checkpoint['critic_state_dict'])

        if optimizer_g is not None and 'optimizer_g_state_dict' in checkpoint:
            optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])

        if optimizer_c is not None and 'optimizer_c_state_dict' in checkpoint:
            optimizer_c.load_state_dict(checkpoint['optimizer_c_state_dict'])

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Model: {checkpoint.get('model_name', 'Unknown')}")
        print(f"  Epoch: {checkpoint.get('epoch', 'Unknown')}")

        return checkpoint

    def save_config(
        self,
        model_name: str,
        config: Dict[str, Any]
    ):
        """
        Save model configuration.

        Args:
            model_name: Name of the model
            config: Configuration dict
        """
        config_path = os.path.join(self.checkpoint_dir, f"{model_name}_config.json")

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)

        print(f"Saved config to {config_path}")

    def load_config(
        self,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Load model configuration.

        Args:
            model_name: Name of the model

        Returns:
            Configuration dict
        """
        config_path = os.path.join(self.checkpoint_dir, f"{model_name}_config.json")

        with open(config_path, 'r') as f:
            config = json.load(f)

        return config

    def save_results(
        self,
        experiment_name: str,
        results: Dict[str, Any]
    ):
        """
        Save experiment results.

        Args:
            experiment_name: Name of the experiment
            results: Results dict
        """
        results_path = os.path.join(self.checkpoint_dir, f"{experiment_name}_results.json")

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"Saved results to {results_path}")

    def load_results(
        self,
        experiment_name: str
    ) -> Dict[str, Any]:
        """
        Load experiment results.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Results dict
        """
        results_path = os.path.join(self.checkpoint_dir, f"{experiment_name}_results.json")

        with open(results_path, 'r') as f:
            results = json.load(f)

        return results

    def get_latest_checkpoint(
        self,
        model_name: str
    ) -> Optional[str]:
        """
        Get path to most recent checkpoint for a model.

        Args:
            model_name: Name of the model

        Returns:
            Path to latest checkpoint, or None if not found
        """
        checkpoints = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith(model_name) and f.endswith('.pt')
        ]

        if not checkpoints:
            return None

        # Sort by modification time
        checkpoints.sort(
            key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x)),
            reverse=True
        )

        return os.path.join(self.checkpoint_dir, checkpoints[0])

    def _cleanup_old_checkpoints(self, model_name: str):
        """
        Remove old checkpoints, keeping only the most recent max_checkpoints.

        Args:
            model_name: Name of the model
        """
        checkpoints = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith(model_name) and f.endswith('.pt')
        ]

        if len(checkpoints) <= self.max_checkpoints:
            return

        # Sort by modification time
        checkpoints.sort(
            key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x))
        )

        # Remove oldest checkpoints
        for checkpoint in checkpoints[:-self.max_checkpoints]:
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint)
            os.remove(checkpoint_path)
            print(f"Removed old checkpoint: {checkpoint_path}")

    def list_checkpoints(self, model_name: Optional[str] = None) -> list:
        """
        List all checkpoints.

        Args:
            model_name: If provided, only list checkpoints for this model

        Returns:
            List of checkpoint filenames
        """
        checkpoints = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.endswith('.pt')
        ]

        if model_name:
            checkpoints = [c for c in checkpoints if c.startswith(model_name)]

        return sorted(checkpoints)

    def __repr__(self) -> str:
        return f"CheckpointManager(dir={self.checkpoint_dir}, max={self.max_checkpoints})"
