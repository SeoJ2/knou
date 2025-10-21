"""
Adaptive FAIR-DP-GAN trainer (PROPOSED METHOD)
Combines fairness + adaptive group-aware DP
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import os
from ..optimizers.group_aware_dpsgd import SimpleGroupAwareDPSGD


class AdaptiveFairDPGANTrainer:
    """
    GAN with fairness and adaptive group-aware differential privacy
    KEY DIFFERENCE: Uses adaptive per-group gradient clipping
    """

    def __init__(self, generator, discriminator, config):
        """
        Args:
            generator: Generator model
            discriminator: FairDiscriminator model
            config: Configuration object
        """
        self.generator = generator
        self.discriminator = discriminator
        self.config = config
        self.device = config.DEVICE

        # Move models to device
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        # Standard optimizer for generator
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=config.LEARNING_RATE,
            betas=(config.BETA1, config.BETA2)
        )

        # DP optimizer for discriminator (ADAPTIVE clipping)
        base_d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.LEARNING_RATE,
            betas=(config.BETA1, config.BETA2)
        )

        self.d_optimizer = SimpleGroupAwareDPSGD(
            base_d_optimizer,
            max_grad_norm=config.MAX_GRAD_NORM,
            noise_multiplier=self._compute_noise_multiplier(),
            adaptive_clipping=True  # ADAPTIVE (group-aware)
        )

        # Loss function
        self.criterion = nn.BCELoss()

        # Training history
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'fairness_loss': [],
            'd_real_acc': [],
            'd_fake_acc': [],
            'epsilon': []
        }

        self.fairness_lambda = config.FAIRNESS_LAMBDA
        self.steps = 0

    def _compute_noise_multiplier(self):
        """Compute noise multiplier for target epsilon"""
        import math
        total_steps = self.config.EPOCHS * (10000 // self.config.BATCH_SIZE)
        sigma = math.sqrt(2 * total_steps * math.log(1 / self.config.DELTA)) / self.config.EPSILON_TARGET
        sigma = max(0.5, min(sigma, 10.0))
        print(f"Computed noise multiplier: {sigma:.4f}")
        return sigma

    def train_step(self, real_images, sensitive_attrs):
        """
        Single training step with adaptive DP and fairness

        Args:
            real_images: Batch of real images
            sensitive_attrs: Sensitive attribute labels

        Returns:
            (g_loss, d_loss, fairness_loss, d_real_acc, d_fake_acc)
        """
        batch_size = real_images.size(0)
        real_images = real_images.to(self.device)
        sensitive_attrs = sensitive_attrs.to(self.device).float()

        # Labels
        real_labels = torch.ones(batch_size, device=self.device)
        fake_labels = torch.zeros(batch_size, device=self.device)

        # =====================
        # Train Discriminator with ADAPTIVE DP-SGD + Fairness
        # =====================
        self.d_optimizer.zero_grad()

        # Real images
        d_real_output, sensitive_real_pred = self.discriminator(real_images, return_fairness=True)
        d_real_loss = self.criterion(d_real_output, real_labels)
        fairness_d_loss = self.criterion(sensitive_real_pred, sensitive_attrs)

        # Fake images
        z = torch.randn(batch_size, self.config.LATENT_DIM, 1, 1, device=self.device)
        with torch.no_grad():
            fake_images = self.generator(z)

        d_fake_output = self.discriminator(fake_images, return_fairness=False)
        d_fake_loss = self.criterion(d_fake_output, fake_labels)

        # Total discriminator loss
        d_loss = d_real_loss + d_fake_loss + self.fairness_lambda * fairness_d_loss
        d_loss.backward()

        # KEY: Clip gradients with ADAPTIVE group-aware clipping
        # Pass group IDs (sensitive attributes) for adaptive clipping
        group_ids = sensitive_attrs.long()
        self.d_optimizer.clip_gradients(self.discriminator, group_ids=group_ids)

        # Add DP noise
        self.d_optimizer.add_noise(self.discriminator)

        # Optimizer step
        self.d_optimizer.step()
        self.steps += 1

        # Accuracy
        with torch.no_grad():
            d_real_acc = ((d_real_output > 0.5).float() == real_labels).float().mean().item()
            d_fake_acc = ((d_fake_output < 0.5).float() == (fake_labels == 0)).float().mean().item()

        # ==================
        # Train Generator
        # ==================
        self.g_optimizer.zero_grad()

        z = torch.randn(batch_size, self.config.LATENT_DIM, 1, 1, device=self.device)
        fake_images = self.generator(z)

        d_fake_output, sensitive_fake_pred = self.discriminator(fake_images, return_fairness=True)
        g_loss_adv = self.criterion(d_fake_output, real_labels)

        # Fairness loss for generator
        random_sensitive = torch.randint_like(sensitive_attrs, 0, 2).float()
        fairness_g_loss = self.criterion(sensitive_fake_pred, random_sensitive)

        g_loss = g_loss_adv + self.fairness_lambda * fairness_g_loss
        g_loss.backward()
        self.g_optimizer.step()

        return g_loss.item(), d_loss.item(), fairness_g_loss.item(), d_real_acc, d_fake_acc

    def train(self, train_loader, epochs=None):
        """Train Adaptive FAIR-DP-GAN"""
        if epochs is None:
            epochs = self.config.EPOCHS

        print(f"\n{'='*60}")
        print(f"Training Adaptive FAIR-DP-GAN (PROPOSED)")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}")
        print(f"Target Epsilon: {self.config.EPSILON_TARGET}")
        print(f"Fairness Lambda: {self.fairness_lambda}")
        print(f"Clipping: ADAPTIVE (group-aware)")
        print(f"{'='*60}\n")

        self.generator.train()
        self.discriminator.train()

        for epoch in range(epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_fairness_loss = 0.0
            epoch_d_real_acc = 0.0
            epoch_d_fake_acc = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_idx, (real_images, attributes) in enumerate(pbar):
                sensitive_attr_idx = 20  # Male attribute
                sensitive_attrs = attributes[:, sensitive_attr_idx]

                g_loss, d_loss, fairness_loss, d_real_acc, d_fake_acc = self.train_step(
                    real_images, sensitive_attrs
                )

                epoch_g_loss += g_loss
                epoch_d_loss += d_loss
                epoch_fairness_loss += fairness_loss
                epoch_d_real_acc += d_real_acc
                epoch_d_fake_acc += d_fake_acc

                pbar.set_postfix({
                    'G_loss': f'{g_loss:.4f}',
                    'D_loss': f'{d_loss:.4f}',
                    'Fair': f'{fairness_loss:.4f}',
                })

            num_batches = len(train_loader)
            avg_g_loss = epoch_g_loss / num_batches
            avg_d_loss = epoch_d_loss / num_batches
            avg_fairness_loss = epoch_fairness_loss / num_batches
            avg_d_real_acc = epoch_d_real_acc / num_batches
            avg_d_fake_acc = epoch_d_fake_acc / num_batches

            epsilon_est = self._estimate_epsilon(epoch + 1, epochs)

            self.history['g_loss'].append(avg_g_loss)
            self.history['d_loss'].append(avg_d_loss)
            self.history['fairness_loss'].append(avg_fairness_loss)
            self.history['d_real_acc'].append(avg_d_real_acc)
            self.history['d_fake_acc'].append(avg_d_fake_acc)
            self.history['epsilon'].append(epsilon_est)

            print(f"Epoch {epoch+1}/{epochs} - "
                  f"G: {avg_g_loss:.4f}, D: {avg_d_loss:.4f}, "
                  f"Fair: {avg_fairness_loss:.4f}, Eps: {epsilon_est:.2f}")

        print(f"\n{'='*60}")
        print(f"Adaptive FAIR-DP-GAN Training Complete")
        print(f"{'='*60}\n")

        return self.history

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
