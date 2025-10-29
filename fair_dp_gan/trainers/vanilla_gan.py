"""
Vanilla GAN Trainer (Baseline)

Standard WGAN-GP training without fairness or privacy constraints.
This serves as the utility baseline for RQ1.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict
from tqdm import tqdm

from .base_trainer import BaseTrainer


class VanillaGANTrainer(BaseTrainer):
    """
    Vanilla WGAN-GP trainer without constraints.

    This is the unconstrained baseline that achieves maximum utility
    but provides no fairness or privacy guarantees.

    Uses standard WGAN-GP objective:
        L_C = E[C(x_fake)] - E[C(x_real)] + λ_GP * GP
        L_G = -E[C(x_fake)]

    Args:
        generator: Generator model
        critic: Critic model
        latent_dim: Dimension of latent noise
        lambda_gp: Gradient penalty coefficient (default: 10.0)
        **kwargs: Additional arguments for BaseTrainer
    """

    def __init__(
        self,
        generator: nn.Module,
        critic: nn.Module,
        latent_dim: int,
        lambda_gp: float = 10.0,
        **kwargs
    ):
        super().__init__(generator, critic, **kwargs)

        self.latent_dim = latent_dim
        self.lambda_gp = lambda_gp

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
            # Unpack batch (handle both image and tabular data)
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

            # ================== Train Critic ================== #
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

                # Gradient penalty
                gp = self.critic.compute_gradient_penalty(
                    real_data, fake_data.detach(), self.lambda_gp
                )

                # Critic loss
                loss_c = -wasserstein_distance + gp

                loss_c.backward()
                self.optimizer_c.step()

            # ================== Train Generator ================== #
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

            # Update progress bar
            pbar.set_postfix({
                'L_C': loss_c.item(),
                'L_G': loss_g.item(),
                'W': wasserstein_distance.item()
            })

        # Epoch statistics
        metrics = {
            'loss_c': total_loss_c / n_batches,
            'loss_g': total_loss_g / n_batches,
            'wasserstein_distance': total_wasserstein / n_batches,
            'gradient_penalty': total_gp / n_batches
        }

        self.losses_c.append(metrics['loss_c'])
        self.losses_g.append(metrics['loss_g'])

        return metrics

    def __repr__(self) -> str:
        return f"VanillaGANTrainer(λ_GP={self.lambda_gp})"
