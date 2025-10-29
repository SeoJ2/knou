"""
Sequential Trainer (Baseline for RQ3)

Two-stage optimization approach:
1. First optimize for privacy + utility (DP-GAN)
2. Then fine-tune for fairness

This serves as baseline to show that simultaneous optimization
(Adaptive Fair-DP-GAN) outperforms sequential approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict
from tqdm import tqdm

from .base_trainer import BaseTrainer
from fair_dp_gan.privacy import GroupAwareDPSGD


class SequentialTrainer(BaseTrainer):
    """
    Sequential two-stage optimization for Fair-DP-GAN.

    **Stage 1** (epochs 0 to switch_epoch):
        Train with DP for privacy + utility (no fairness)

    **Stage 2** (epochs switch_epoch to end):
        Fine-tune with fairness regularization (maintain DP)

    This baseline tests RQ3: Is simultaneous optimization better than sequential?

    Args:
        generator: Generator model
        critic: Critic model (with group_head)
        latent_dim: Dimension of latent noise
        switch_epoch: Epoch to switch from stage 1 to stage 2 (default: 50)
        clipping_norm: Gradient clipping threshold (default: 1.0)
        noise_multiplier: Noise scale (default: 1.0)
        target_epsilon: Target privacy budget (default: 10.0)
        target_delta: Target delta for (ε, δ)-DP (default: 1e-5)
        lambda_gp: Gradient penalty coefficient (default: 10.0)
        lambda_fair: Fairness regularization weight for stage 2 (default: 1.0)
        dataset_size: Size of training dataset
        **kwargs: Additional arguments for BaseTrainer
    """

    def __init__(
        self,
        generator: nn.Module,
        critic: nn.Module,
        latent_dim: int,
        switch_epoch: int = 50,
        clipping_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        target_epsilon: float = 10.0,
        target_delta: float = 1e-5,
        lambda_gp: float = 10.0,
        lambda_fair: float = 1.0,
        dataset_size: int = 10000,
        **kwargs
    ):
        super().__init__(generator, critic, **kwargs)

        self.latent_dim = latent_dim
        self.switch_epoch = switch_epoch
        self.lambda_gp = lambda_gp
        self.lambda_fair = lambda_fair
        self.dataset_size = dataset_size
        self.target_epsilon = target_epsilon

        # Current training stage
        self.current_stage = 1

        # Initialize DP mechanism
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

        # Check if we should switch stages
        if epoch >= self.switch_epoch and self.current_stage == 1:
            print(f"\n{'='*60}")
            print(f"SWITCHING TO STAGE 2: Fairness Fine-tuning")
            print(f"{'='*60}\n")
            self.current_stage = 2

        # Route to appropriate training function
        if self.current_stage == 1:
            return self._train_epoch_stage1(dataloader, epoch)
        else:
            return self._train_epoch_stage2(dataloader, epoch)

    def _train_epoch_stage1(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Stage 1: Privacy + Utility (no fairness).
        Standard DP-GAN training.
        """
        self.generator.train()
        self.critic.train()

        total_loss_c = 0
        total_loss_g = 0
        total_wasserstein = 0
        total_gp = 0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [STAGE 1: DP]")

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

                # Critic scores (ignore group head in stage 1)
                real_scores = self.critic(real_data)['score']
                fake_scores = self.critic(fake_data.detach())['score']

                # Wasserstein distance
                wasserstein_distance = real_scores.mean() - fake_scores.mean()

                # Gradient penalty
                gp = self.critic.compute_gradient_penalty(
                    real_data, fake_data.detach(), self.lambda_gp * 0.1
                )

                # Critic loss (no fairness in stage 1)
                loss_c = -wasserstein_distance + gp

                loss_c.backward()

                # Apply DP-SGD
                self.dp_sgd.add_noise(
                    self.critic,
                    batch_size=batch_size,
                    dataset_size=self.dataset_size
                )

                self.optimizer_c.step()

            # ================== Train Generator ================== #
            self.optimizer_g.zero_grad()

            # Generate fake data
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake_data = self.generator(z, group_labels)

            # Generator loss (no fairness in stage 1)
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
                'Stage': 1,
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
            'stage': 1,
            'loss_c': total_loss_c / n_batches,
            'loss_g': total_loss_g / n_batches,
            'wasserstein_distance': total_wasserstein / n_batches,
            'gradient_penalty': total_gp / n_batches,
            'fairness_loss': 0,  # No fairness in stage 1
            'epsilon': epsilon,
            'delta': delta
        }

        self.losses_c.append(metrics['loss_c'])
        self.losses_g.append(metrics['loss_g'])

        return metrics

    def _train_epoch_stage2(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Stage 2: Fairness Fine-tuning (maintain DP).
        Add fairness regularization while maintaining privacy.
        """
        self.generator.train()
        self.critic.train()

        total_loss_c = 0
        total_loss_g = 0
        total_wasserstein = 0
        total_gp = 0
        total_fairness_loss = 0
        total_group_accuracy = 0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [STAGE 2: Fairness]")

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

            # ================== Train Critic with DP + Fairness ================== #
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
                    real_data, fake_data.detach(), self.lambda_gp * 0.1
                )

                # Group classification loss (added in stage 2)
                loss_group = 0
                if 'group_logits' in real_outputs:
                    group_logits_real = real_outputs['group_logits']
                    loss_group = F.cross_entropy(group_logits_real, group_labels)

                # Total critic loss
                loss_c = -wasserstein_distance + gp + loss_group

                loss_c.backward()

                # Apply DP-SGD
                self.dp_sgd.add_noise(
                    self.critic,
                    batch_size=batch_size,
                    dataset_size=self.dataset_size
                )

                self.optimizer_c.step()

            # ================== Train Generator with Fairness ================== #
            self.optimizer_g.zero_grad()

            # Generate fake data
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake_data = self.generator(z, group_labels)

            # Critic outputs for fake data
            fake_outputs = self.critic(fake_data)
            fake_scores = fake_outputs['score']

            # Standard generator loss
            loss_g_adv = -fake_scores.mean()

            # Fairness loss (key addition in stage 2)
            fairness_loss = 0
            if 'group_logits' in fake_outputs:
                group_logits_fake = fake_outputs['group_logits']
                group_probs = F.softmax(group_logits_fake, dim=1)
                fairness_loss = -torch.mean(
                    -torch.sum(group_probs * torch.log(group_probs + 1e-8), dim=1)
                )

            # Total generator loss (with fairness)
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

            # Get current privacy spent
            epsilon, delta = self.dp_sgd.get_privacy_spent()

            # Update progress bar
            pbar.set_postfix({
                'Stage': 2,
                'L_C': loss_c.item(),
                'L_G': loss_g.item(),
                'ε': epsilon,
                'Fair': fairness_loss.item() if isinstance(fairness_loss, torch.Tensor) else 0
            })

            # Early stop if privacy budget exceeded
            if epsilon > self.target_epsilon:
                print(f"\nPrivacy budget exceeded: ε={epsilon:.2f} > {self.target_epsilon}")
                break

        # Epoch statistics
        epsilon, delta = self.dp_sgd.get_privacy_spent()

        metrics = {
            'stage': 2,
            'loss_c': total_loss_c / n_batches,
            'loss_g': total_loss_g / n_batches,
            'wasserstein_distance': total_wasserstein / n_batches,
            'gradient_penalty': total_gp / n_batches,
            'fairness_loss': total_fairness_loss / n_batches,
            'group_accuracy': total_group_accuracy / n_batches,
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
        return (
            f"SequentialTrainer("
            f"stage={self.current_stage}, "
            f"switch_epoch={self.switch_epoch}, "
            f"ε={epsilon:.2f})"
        )
