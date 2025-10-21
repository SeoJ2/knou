"""
Sequential Trainer - Two-phase training approach
Phase 1: DP learning (50 epochs)
Phase 2: Rebalancing + Fairness fine-tuning (20 epochs)
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
from ..optimizers.group_aware_dpsgd import SimpleGroupAwareDPSGD


class SequentialTrainer:
    """
    Sequential training approach:
    1. Phase 1: Train with DP (privacy-preserving learning)
    2. Phase 2: Rebalance data + Fine-tune with fairness constraints
    """

    def __init__(self, generator, discriminator, config):
        """
        Args:
            generator: Generator model
            discriminator: FairDiscriminator model (for phase 2)
            config: Configuration object
        """
        self.generator = generator
        self.discriminator = discriminator
        self.config = config
        self.device = config.DEVICE

        # Move models to device
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        # Training history
        self.history = {
            'phase1': {
                'g_loss': [],
                'd_loss': [],
                'epsilon': []
            },
            'phase2': {
                'g_loss': [],
                'd_loss': [],
                'fairness_loss': []
            }
        }

        self.fairness_lambda = config.FAIRNESS_LAMBDA

    def _create_rebalanced_loader(self, train_loader):
        """
        Create a rebalanced data loader for Phase 2
        Ensures equal representation of sensitive groups

        Args:
            train_loader: Original training data loader

        Returns:
            Rebalanced data loader
        """
        print("\n" + "="*60)
        print("Creating Rebalanced Dataset for Phase 2")
        print("="*60)

        # Collect all data
        all_images = []
        all_attributes = []

        for images, attributes in train_loader:
            all_images.append(images)
            all_attributes.append(attributes)

        all_images = torch.cat(all_images, dim=0)
        all_attributes = torch.cat(all_attributes, dim=0)

        # Get sensitive attribute (Male = 20)
        sensitive_idx = 20
        sensitive_attrs = all_attributes[:, sensitive_idx]

        # Separate by groups
        group_0_mask = sensitive_attrs == 0
        group_1_mask = sensitive_attrs == 1

        group_0_images = all_images[group_0_mask]
        group_0_attrs = all_attributes[group_0_mask]

        group_1_images = all_images[group_1_mask]
        group_1_attrs = all_attributes[group_1_mask]

        print(f"Group 0 (non-male): {len(group_0_images)} samples")
        print(f"Group 1 (male): {len(group_1_images)} samples")

        # Balance: take equal samples from each group
        min_samples = min(len(group_0_images), len(group_1_images))

        # Randomly sample
        group_0_indices = torch.randperm(len(group_0_images))[:min_samples]
        group_1_indices = torch.randperm(len(group_1_images))[:min_samples]

        balanced_images = torch.cat([
            group_0_images[group_0_indices],
            group_1_images[group_1_indices]
        ], dim=0)

        balanced_attrs = torch.cat([
            group_0_attrs[group_0_indices],
            group_1_attrs[group_1_indices]
        ], dim=0)

        print(f"Rebalanced dataset: {len(balanced_images)} samples")
        print(f"Each group: {min_samples} samples")
        print("="*60 + "\n")

        # Create new data loader
        from torch.utils.data import TensorDataset, DataLoader
        balanced_dataset = TensorDataset(balanced_images, balanced_attrs)
        balanced_loader = DataLoader(
            balanced_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )

        return balanced_loader

    def phase1_train_step(self, real_images, d_optimizer):
        """
        Phase 1 training step: DP learning only

        Args:
            real_images: Batch of real images
            d_optimizer: DP-SGD optimizer for discriminator

        Returns:
            (g_loss, d_loss)
        """
        batch_size = real_images.size(0)
        real_images = real_images.to(self.device)

        # Labels
        real_labels = torch.ones(batch_size, device=self.device)
        fake_labels = torch.zeros(batch_size, device=self.device)

        criterion = nn.BCELoss()

        # =====================
        # Train Discriminator with DP
        # =====================
        d_optimizer.zero_grad()

        # Real images
        d_real_output = self.discriminator(real_images, return_fairness=False)
        d_real_loss = criterion(d_real_output, real_labels)

        # Fake images
        z = torch.randn(batch_size, self.config.LATENT_DIM, 1, 1, device=self.device)
        with torch.no_grad():
            fake_images = self.generator(z)

        d_fake_output = self.discriminator(fake_images, return_fairness=False)
        d_fake_loss = criterion(d_fake_output, fake_labels)

        # Total loss
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()

        # Clip and add noise (DP-SGD)
        d_optimizer.clip_gradients(self.discriminator)
        d_optimizer.add_noise(self.discriminator)
        d_optimizer.step()

        # ==================
        # Train Generator
        # ==================
        self.g_optimizer.zero_grad()

        z = torch.randn(batch_size, self.config.LATENT_DIM, 1, 1, device=self.device)
        fake_images = self.generator(z)

        d_fake_output = self.discriminator(fake_images, return_fairness=False)
        g_loss = criterion(d_fake_output, real_labels)

        g_loss.backward()
        self.g_optimizer.step()

        return g_loss.item(), d_loss.item()

    def phase2_train_step(self, real_images, sensitive_attrs, d_optimizer):
        """
        Phase 2 training step: Fairness fine-tuning

        Args:
            real_images: Batch of real images
            sensitive_attrs: Sensitive attribute labels
            d_optimizer: Standard optimizer (no DP in phase 2)

        Returns:
            (g_loss, d_loss, fairness_loss)
        """
        batch_size = real_images.size(0)
        real_images = real_images.to(self.device)
        sensitive_attrs = sensitive_attrs.to(self.device).float()

        real_labels = torch.ones(batch_size, device=self.device)
        fake_labels = torch.zeros(batch_size, device=self.device)

        criterion = nn.BCELoss()

        # =====================
        # Train Discriminator with Fairness
        # =====================
        d_optimizer.zero_grad()

        d_real_output, sensitive_real_pred = self.discriminator(real_images, return_fairness=True)
        d_real_loss = criterion(d_real_output, real_labels)
        fairness_d_loss = criterion(sensitive_real_pred, sensitive_attrs)

        z = torch.randn(batch_size, self.config.LATENT_DIM, 1, 1, device=self.device)
        with torch.no_grad():
            fake_images = self.generator(z)

        d_fake_output = self.discriminator(fake_images, return_fairness=False)
        d_fake_loss = criterion(d_fake_output, fake_labels)

        d_loss = d_real_loss + d_fake_loss + self.fairness_lambda * fairness_d_loss
        d_loss.backward()
        d_optimizer.step()

        # ==================
        # Train Generator with Fairness
        # ==================
        self.g_optimizer.zero_grad()

        z = torch.randn(batch_size, self.config.LATENT_DIM, 1, 1, device=self.device)
        fake_images = self.generator(z)

        d_fake_output, sensitive_fake_pred = self.discriminator(fake_images, return_fairness=True)
        g_loss_adv = criterion(d_fake_output, real_labels)

        random_sensitive = torch.randint_like(sensitive_attrs, 0, 2).float()
        fairness_g_loss = criterion(sensitive_fake_pred, random_sensitive)

        g_loss = g_loss_adv + self.fairness_lambda * fairness_g_loss
        g_loss.backward()
        self.g_optimizer.step()

        return g_loss.item(), d_loss.item(), fairness_g_loss.item()

    def train(self, train_loader):
        """
        Two-phase sequential training

        Args:
            train_loader: Training data loader

        Returns:
            Training history
        """
        print(f"\n{'='*60}")
        print(f"Sequential Training Approach")
        print(f"{'='*60}")
        print(f"Phase 1: DP Learning ({self.config.PHASE1_EPOCHS} epochs)")
        print(f"Phase 2: Rebalancing + Fairness ({self.config.PHASE2_EPOCHS} epochs)")
        print(f"{'='*60}\n")

        # ==================
        # PHASE 1: DP Learning
        # ==================
        print("\n" + "="*60)
        print("PHASE 1: DP Learning")
        print("="*60)

        # Create optimizers for phase 1
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config.LEARNING_RATE,
            betas=(self.config.BETA1, self.config.BETA2)
        )

        base_d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.LEARNING_RATE,
            betas=(self.config.BETA1, self.config.BETA2)
        )

        d_optimizer_phase1 = SimpleGroupAwareDPSGD(
            base_d_optimizer,
            max_grad_norm=self.config.MAX_GRAD_NORM,
            noise_multiplier=self._compute_noise_multiplier(),
            adaptive_clipping=False
        )

        self.generator.train()
        self.discriminator.train()

        for epoch in range(self.config.PHASE1_EPOCHS):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Phase 1 - Epoch {epoch+1}/{self.config.PHASE1_EPOCHS}")
            for real_images, _ in pbar:
                g_loss, d_loss = self.phase1_train_step(real_images, d_optimizer_phase1)

                epoch_g_loss += g_loss
                epoch_d_loss += d_loss

                pbar.set_postfix({
                    'G_loss': f'{g_loss:.4f}',
                    'D_loss': f'{d_loss:.4f}',
                })

            num_batches = len(train_loader)
            avg_g_loss = epoch_g_loss / num_batches
            avg_d_loss = epoch_d_loss / num_batches

            epsilon_est = self._estimate_epsilon(epoch + 1, self.config.PHASE1_EPOCHS)

            self.history['phase1']['g_loss'].append(avg_g_loss)
            self.history['phase1']['d_loss'].append(avg_d_loss)
            self.history['phase1']['epsilon'].append(epsilon_est)

            print(f"Phase 1 - Epoch {epoch+1}/{self.config.PHASE1_EPOCHS} - "
                  f"G: {avg_g_loss:.4f}, D: {avg_d_loss:.4f}, Eps: {epsilon_est:.2f}")

        print("="*60)
        print("Phase 1 Complete")
        print("="*60 + "\n")

        # ==================
        # PHASE 2: Rebalancing + Fairness
        # ==================
        print("\n" + "="*60)
        print("PHASE 2: Rebalancing + Fairness Fine-tuning")
        print("="*60)

        # Create rebalanced data loader
        balanced_loader = self._create_rebalanced_loader(train_loader)

        # Create new optimizers for phase 2 (no DP, just standard)
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config.LEARNING_RATE * 0.1,  # Lower LR for fine-tuning
            betas=(self.config.BETA1, self.config.BETA2)
        )

        d_optimizer_phase2 = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.LEARNING_RATE * 0.1,
            betas=(self.config.BETA1, self.config.BETA2)
        )

        for epoch in range(self.config.PHASE2_EPOCHS):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_fairness_loss = 0.0

            pbar = tqdm(balanced_loader, desc=f"Phase 2 - Epoch {epoch+1}/{self.config.PHASE2_EPOCHS}")
            for real_images, attributes in pbar:
                sensitive_idx = 20
                sensitive_attrs = attributes[:, sensitive_idx]

                g_loss, d_loss, fairness_loss = self.phase2_train_step(
                    real_images, sensitive_attrs, d_optimizer_phase2
                )

                epoch_g_loss += g_loss
                epoch_d_loss += d_loss
                epoch_fairness_loss += fairness_loss

                pbar.set_postfix({
                    'G_loss': f'{g_loss:.4f}',
                    'D_loss': f'{d_loss:.4f}',
                    'Fair': f'{fairness_loss:.4f}',
                })

            num_batches = len(balanced_loader)
            avg_g_loss = epoch_g_loss / num_batches
            avg_d_loss = epoch_d_loss / num_batches
            avg_fairness_loss = epoch_fairness_loss / num_batches

            self.history['phase2']['g_loss'].append(avg_g_loss)
            self.history['phase2']['d_loss'].append(avg_d_loss)
            self.history['phase2']['fairness_loss'].append(avg_fairness_loss)

            print(f"Phase 2 - Epoch {epoch+1}/{self.config.PHASE2_EPOCHS} - "
                  f"G: {avg_g_loss:.4f}, D: {avg_d_loss:.4f}, Fair: {avg_fairness_loss:.4f}")

        print("="*60)
        print("Phase 2 Complete")
        print("="*60)

        print(f"\n{'='*60}")
        print(f"Sequential Training Complete")
        print(f"{'='*60}\n")

        return self.history

    def _compute_noise_multiplier(self):
        """Compute noise multiplier for target epsilon"""
        import math
        total_steps = self.config.PHASE1_EPOCHS * (10000 // self.config.BATCH_SIZE)
        sigma = math.sqrt(2 * total_steps * math.log(1 / self.config.DELTA)) / self.config.EPSILON_TARGET
        sigma = max(0.5, min(sigma, 10.0))
        return sigma

    def _estimate_epsilon(self, current_epoch, total_epochs):
        """Estimate current epsilon"""
        progress = current_epoch / total_epochs
        return self.config.EPSILON_TARGET * progress

    def save_models(self, path):
        """Save models"""
        os.makedirs(path, exist_ok=True)
        torch.save(self.generator.state_dict(), os.path.join(path, 'generator.pth'))
        torch.save(self.discriminator.state_dict(), os.path.join(path, 'discriminator.pth'))

    def load_models(self, path):
        """Load models"""
        self.generator.load_state_dict(torch.load(os.path.join(path, 'generator.pth')))
        self.discriminator.load_state_dict(torch.load(os.path.join(path, 'discriminator.pth')))
