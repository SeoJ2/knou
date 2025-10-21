"""
DP-GAN trainer - Differential Privacy constraint only
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import os
from ..optimizers.group_aware_dpsgd import SimpleGroupAwareDPSGD


class DPGANTrainer:
    """
    GAN with differential privacy constraint
    Uses DP-SGD for discriminator training
    """

    def __init__(self, generator, discriminator, config):
        """
        Args:
            generator: Generator model
            discriminator: Discriminator model
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

        # DP optimizer for discriminator
        base_d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.LEARNING_RATE,
            betas=(config.BETA1, config.BETA2)
        )

        # Wrap with DP-SGD (uniform clipping, no group-awareness)
        self.d_optimizer = SimpleGroupAwareDPSGD(
            base_d_optimizer,
            max_grad_norm=config.MAX_GRAD_NORM,
            noise_multiplier=self._compute_noise_multiplier(),
            adaptive_clipping=False  # Uniform DP
        )

        # Loss function
        self.criterion = nn.BCELoss()

        # Training history
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'd_real_acc': [],
            'd_fake_acc': [],
            'epsilon': []
        }

        # Privacy accounting
        self.steps = 0

    def _compute_noise_multiplier(self):
        """
        Compute noise multiplier for target epsilon

        Returns:
            Noise multiplier (sigma)
        """
        # Simplified: sigma = sqrt(2 * T * ln(1/delta)) / epsilon
        # where T is the number of steps
        # For better accuracy, use opacus RDP accountant
        import math

        total_steps = self.config.EPOCHS * (10000 // self.config.BATCH_SIZE)  # Approximate
        sigma = math.sqrt(2 * total_steps * math.log(1 / self.config.DELTA)) / self.config.EPSILON_TARGET

        # Ensure reasonable value
        sigma = max(0.5, min(sigma, 10.0))

        print(f"Computed noise multiplier: {sigma:.4f}")
        return sigma

    def train_step(self, real_images):
        """
        Single training step with DP

        Args:
            real_images: Batch of real images

        Returns:
            (g_loss, d_loss, d_real_acc, d_fake_acc)
        """
        batch_size = real_images.size(0)
        real_images = real_images.to(self.device)

        # Labels
        real_labels = torch.ones(batch_size, device=self.device)
        fake_labels = torch.zeros(batch_size, device=self.device)

        # =====================
        # Train Discriminator with DP-SGD
        # =====================
        self.d_optimizer.zero_grad()

        # Real images
        d_real_output = self.discriminator(real_images)
        d_real_loss = self.criterion(d_real_output, real_labels)

        # Fake images
        z = torch.randn(batch_size, self.config.LATENT_DIM, 1, 1, device=self.device)
        with torch.no_grad():
            fake_images = self.generator(z)

        d_fake_output = self.discriminator(fake_images)
        d_fake_loss = self.criterion(d_fake_output, fake_labels)

        # Total discriminator loss
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()

        # Clip gradients (DP-SGD)
        self.d_optimizer.clip_gradients(self.discriminator)

        # Add noise (DP-SGD)
        self.d_optimizer.add_noise(self.discriminator)

        # Optimizer step
        self.d_optimizer.step()

        self.steps += 1

        # Accuracy
        with torch.no_grad():
            d_real_acc = ((d_real_output > 0.5).float() == real_labels).float().mean().item()
            d_fake_acc = ((d_fake_output < 0.5).float() == (fake_labels == 0)).float().mean().item()

        # ==================
        # Train Generator (no DP)
        # ==================
        self.g_optimizer.zero_grad()

        # Generate fake images
        z = torch.randn(batch_size, self.config.LATENT_DIM, 1, 1, device=self.device)
        fake_images = self.generator(z)

        # Generator wants discriminator to think fake images are real
        d_fake_output = self.discriminator(fake_images)
        g_loss = self.criterion(d_fake_output, real_labels)

        g_loss.backward()
        self.g_optimizer.step()

        return g_loss.item(), d_loss.item(), d_real_acc, d_fake_acc

    def train(self, train_loader, epochs=None):
        """
        Train DP-GAN

        Args:
            train_loader: DataLoader for training data
            epochs: Number of epochs

        Returns:
            Training history
        """
        if epochs is None:
            epochs = self.config.EPOCHS

        print(f"\n{'='*60}")
        print(f"Training DP-GAN")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}")
        print(f"Target Epsilon: {self.config.EPSILON_TARGET}")
        print(f"Max Grad Norm: {self.config.MAX_GRAD_NORM}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        self.generator.train()
        self.discriminator.train()

        for epoch in range(epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_d_real_acc = 0.0
            epoch_d_fake_acc = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_idx, (real_images, _) in enumerate(pbar):
                g_loss, d_loss, d_real_acc, d_fake_acc = self.train_step(real_images)

                epoch_g_loss += g_loss
                epoch_d_loss += d_loss
                epoch_d_real_acc += d_real_acc
                epoch_d_fake_acc += d_fake_acc

                pbar.set_postfix({
                    'G_loss': f'{g_loss:.4f}',
                    'D_loss': f'{d_loss:.4f}',
                })

            # Average losses
            num_batches = len(train_loader)
            avg_g_loss = epoch_g_loss / num_batches
            avg_d_loss = epoch_d_loss / num_batches
            avg_d_real_acc = epoch_d_real_acc / num_batches
            avg_d_fake_acc = epoch_d_fake_acc / num_batches

            # Estimate epsilon (simplified)
            epsilon_est = self._estimate_epsilon(epoch + 1, epochs)

            # Save history
            self.history['g_loss'].append(avg_g_loss)
            self.history['d_loss'].append(avg_d_loss)
            self.history['d_real_acc'].append(avg_d_real_acc)
            self.history['d_fake_acc'].append(avg_d_fake_acc)
            self.history['epsilon'].append(epsilon_est)

            print(f"Epoch {epoch+1}/{epochs} - "
                  f"G_loss: {avg_g_loss:.4f}, D_loss: {avg_d_loss:.4f}, "
                  f"Epsilon (est): {epsilon_est:.2f}")

        print(f"\n{'='*60}")
        print(f"DP-GAN Training Complete")
        print(f"Final Epsilon (estimated): {self.history['epsilon'][-1]:.2f}")
        print(f"{'='*60}\n")

        return self.history

    def _estimate_epsilon(self, current_epoch, total_epochs):
        """
        Estimate current epsilon (linear scaling for simplicity)

        Args:
            current_epoch: Current epoch number
            total_epochs: Total number of epochs

        Returns:
            Estimated epsilon
        """
        # Linear scaling (very rough approximation)
        progress = current_epoch / total_epochs
        epsilon = self.config.EPSILON_TARGET * progress
        return epsilon

    def save_models(self, path):
        """Save models"""
        os.makedirs(path, exist_ok=True)
        torch.save(self.generator.state_dict(), os.path.join(path, 'generator.pth'))
        torch.save(self.discriminator.state_dict(), os.path.join(path, 'discriminator.pth'))
        print(f"Models saved to {path}")

    def load_models(self, path):
        """Load models"""
        self.generator.load_state_dict(torch.load(os.path.join(path, 'generator.pth')))
        self.discriminator.load_state_dict(torch.load(os.path.join(path, 'discriminator.pth')))
        print(f"Models loaded from {path}")
