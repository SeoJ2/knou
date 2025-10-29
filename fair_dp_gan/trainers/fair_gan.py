"""
Fair GAN Trainer

WGAN-GP with fairness regularization but no differential privacy.
This isolates the impact of fairness constraints for RQ1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict
from tqdm import tqdm

from .base_trainer import BaseTrainer


class FairGANTrainer(BaseTrainer):
    """
    Fair GAN trainer with group-based fairness regularization.

    Adds fairness regularization to encourage statistical parity:
        L_fair = |P(ŷ=1|G=0) - P(ŷ=1|G=1)|

    Uses adversarial fairness: critic has auxiliary head to predict group,
    generator tries to fool both main and group heads.

    Total Generator Loss:
        L_G = -E[C(x_fake)] + λ_fair * L_fair

    Args:
        generator: Generator model
        critic: Critic model (must have group_head)
        latent_dim: Dimension of latent noise
        lambda_gp: Gradient penalty coefficient (default: 10.0)
        lambda_fair: Fairness regularization weight (default: 1.0)
        **kwargs: Additional arguments for BaseTrainer
    """

    def __init__(
        self,
        generator: nn.Module,
        critic: nn.Module,
        latent_dim: int,
        lambda_gp: float = 10.0,
        lambda_fair: float = 1.0,
        **kwargs
    ):
        super().__init__(generator, critic, **kwargs)

        self.latent_dim = latent_dim
        self.lambda_gp = lambda_gp
        self.lambda_fair = lambda_fair

        # Check that critic has group head
        if not hasattr(critic, 'use_group_head') or not critic.use_group_head:
            print("Warning: Critic should have group_head for fairness regularization")

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
        total_fairness_loss = 0
        total_group_accuracy = 0
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

            # ================== Train Critic ================== #
            for _ in range(self.n_critic):
                self.optimizer_c.zero_grad()

                # Generate fake data
                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_data = self.generator(z, group_labels)

                # Critic outputs
                real_outputs = self.critic(real_data)
                fake_outputs = self.critic(fake_data.detach())

                real_scores = real_outputs['score']
                fake_scores = fake_outputs['score']

                # Wasserstein distance
                wasserstein_distance = real_scores.mean() - fake_scores.mean()

                # Gradient penalty
                gp = self.critic.compute_gradient_penalty(
                    real_data, fake_data.detach(), self.lambda_gp
                )

                # Group classification loss (auxiliary task)
                loss_group = 0
                if 'group_logits' in real_outputs:
                    group_logits_real = real_outputs['group_logits']
                    loss_group = F.cross_entropy(group_logits_real, group_labels)

                # Total critic loss
                loss_c = -wasserstein_distance + gp + loss_group

                loss_c.backward()
                self.optimizer_c.step()

            # ================== Train Generator ================== #
            self.optimizer_g.zero_grad()

            # Generate fake data
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake_data = self.generator(z, group_labels)

            # Critic outputs for fake data
            fake_outputs = self.critic(fake_data)
            fake_scores = fake_outputs['score']

            # Standard generator loss
            loss_g_adv = -fake_scores.mean()

            # Fairness loss: encourage group confusion
            # Generator tries to make group unidentifiable
            fairness_loss = 0
            if 'group_logits' in fake_outputs:
                group_logits_fake = fake_outputs['group_logits']

                # Adversarial fairness: maximize entropy of group predictions
                # This makes generated samples group-agnostic
                group_probs = F.softmax(group_logits_fake, dim=1)
                fairness_loss = -torch.mean(
                    -torch.sum(group_probs * torch.log(group_probs + 1e-8), dim=1)
                )

            # Total generator loss
            loss_g = loss_g_adv + self.lambda_fair * fairness_loss

            loss_g.backward()
            self.optimizer_g.step()

            # Track statistics
            total_loss_c += loss_c.item()
            total_loss_g += loss_g.item()
            total_wasserstein += wasserstein_distance.item()
            total_gp += gp.item()
            total_fairness_loss += fairness_loss.item() if isinstance(fairness_loss, torch.Tensor) else 0

            if 'group_logits' in real_outputs:
                group_pred = real_outputs['group_logits'].argmax(dim=1)
                group_acc = (group_pred == group_labels).float().mean().item()
                total_group_accuracy += group_acc

            n_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'L_C': loss_c.item(),
                'L_G': loss_g.item(),
                'Fair': fairness_loss.item() if isinstance(fairness_loss, torch.Tensor) else 0
            })

        # Epoch statistics
        metrics = {
            'loss_c': total_loss_c / n_batches,
            'loss_g': total_loss_g / n_batches,
            'wasserstein_distance': total_wasserstein / n_batches,
            'gradient_penalty': total_gp / n_batches,
            'fairness_loss': total_fairness_loss / n_batches,
            'group_accuracy': total_group_accuracy / n_batches
        }

        self.losses_c.append(metrics['loss_c'])
        self.losses_g.append(metrics['loss_g'])

        return metrics

    def __repr__(self) -> str:
        return f"FairGANTrainer(λ_GP={self.lambda_gp}, λ_fair={self.lambda_fair})"
