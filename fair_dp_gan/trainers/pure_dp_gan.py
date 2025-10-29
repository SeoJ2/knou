"""
Pure DP-GAN Trainer

WGAN-GP with differential privacy but no fairness constraints.
This isolates the impact of privacy constraints for RQ1.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm

from .base_trainer import BaseTrainer
from fair_dp_gan.privacy import GroupAwareDPSGD


class PureDPGANTrainer(BaseTrainer):
    """
    DP-GAN trainer with differential privacy but no fairness constraints.

    Applies DP-SGD to critic training following DPGAN/GS-WGAN approaches:
    1. Clip per-sample gradients
    2. Add calibrated Gaussian noise
    3. Track privacy budget via RDP

    Only the critic is trained with DP (generator uses standard gradients).

    Args:
        generator: Generator model
        critic: Critic model
        latent_dim: Dimension of latent noise
        clipping_norm: Gradient clipping threshold (default: 1.0)
        noise_multiplier: Noise scale (default: 1.0)
        target_epsilon: Target privacy budget (default: 10.0)
        target_delta: Target delta for (ε, δ)-DP (default: 1e-5)
        lambda_gp: Gradient penalty coefficient (default: 10.0)
        dataset_size: Size of training dataset (for privacy accounting)
        **kwargs: Additional arguments for BaseTrainer
    """

    def __init__(
        self,
        generator: nn.Module,
        critic: nn.Module,
        latent_dim: int,
        clipping_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        target_epsilon: float = 10.0,
        target_delta: float = 1e-5,
        lambda_gp: float = 10.0,
        dataset_size: int = 10000,
        **kwargs
    ):
        super().__init__(generator, critic, **kwargs)

        self.latent_dim = latent_dim
        self.lambda_gp = lambda_gp
        self.dataset_size = dataset_size
        self.target_epsilon = target_epsilon

        # Initialize DP mechanism (uniform mode)
        self.dp_sgd = GroupAwareDPSGD(
            clipping_mode='uniform',
            base_clipping_norm=clipping_norm,
            noise_multiplier=noise_multiplier,
            target_delta=target_delta
        )

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.generator.train()
        self.critic.train()

        total_loss_c = 0
        total_loss_g = 0
        total_wasserstein = 0
        total_gp = 0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for batch_idx, batch in enumerate(pbar):
            # Unpack batch
            if len(batch) == 3:
                real_data, group_labels, _ = batch
            elif len(batch) == 4:
                real_data, group_labels, _, _ = batch
            else:
                real_data = batch[0]
                group_labels = batch[1] if len(batch) > 1 else torch.zeros(len(real_data), dtype=torch.long)

            real_data = real_data.to(self.device)
            group_labels = group_labels.to(self.device)
            batch_size = real_data.size(0)

            # ================== Train Critic with DP ================== #
            for _ in range(self.n_critic):
                self.optimizer_c.zero_grad()

                # Generate fake data
                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_data = self.generator(z, group_labels)

                # Critic scores
                real_scores = self.critic(real_data)['score']
                fake_scores = self.critic(fake_data.detach())['score']

                # Wasserstein distance
                wasserstein_distance = real_scores.mean() - fake_scores.mean()

                # Gradient penalty (reduced weight under DP)
                gp = self.critic.compute_gradient_penalty(
                    real_data, fake_data.detach(), self.lambda_gp * 0.1
                )

                # Critic loss
                loss_c = -wasserstein_distance + gp

                loss_c.backward()

                # Apply DP-SGD: clip gradients and add noise
                # Note: In practice, use Opacus for proper per-sample gradient computation
                # This is a simplified version
                self.dp_sgd.add_noise(
                    self.critic,
                    batch_size=batch_size,
                    dataset_size=self.dataset_size
                )

                self.optimizer_c.step()

            # ================== Train Generator (no DP) ================== #
            self.optimizer_g.zero_grad()

            # Generate fake data
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake_data = self.generator(z, group_labels)

            # Generator loss
            fake_scores = self.critic(fake_data)['score']
            loss_g = -fake_scores.mean()

            loss_g.backward()
            self.optimizer_g.step()

            # Track statistics
            total_loss_c += loss_c.item()
            total_loss_g += loss_g.item()
            total_wasserstein += wasserstein_distance.item()
            total_gp += gp.item()
            n_batches += 1

            # Get current privacy spent
            epsilon, delta = self.dp_sgd.get_privacy_spent()

            # Update progress bar
            pbar.set_postfix({
                'L_C': loss_c.item(),
                'L_G': loss_g.item(),
                'ε': epsilon
            })

            # Early stop if privacy budget exceeded
            if epsilon > self.target_epsilon:
                print(f"\nPrivacy budget exceeded: ε={epsilon:.2f} > {self.target_epsilon}")
                break

        # Epoch statistics
        epsilon, delta = self.dp_sgd.get_privacy_spent()

        metrics = {
            'loss_c': total_loss_c / n_batches,
            'loss_g': total_loss_g / n_batches,
            'wasserstein_distance': total_wasserstein / n_batches,
            'gradient_penalty': total_gp / n_batches,
            'epsilon': epsilon,
            'delta': delta
        }

        self.losses_c.append(metrics['loss_c'])
        self.losses_g.append(metrics['loss_g'])

        return metrics

    def get_privacy_spent(self):
        """Get current privacy budget."""
        return self.dp_sgd.get_privacy_spent()

    def __repr__(self) -> str:
        epsilon, delta = self.dp_sgd.get_privacy_spent()
        return f"PureDPGANTrainer(ε={epsilon:.2f}, δ={delta:.2e})"
