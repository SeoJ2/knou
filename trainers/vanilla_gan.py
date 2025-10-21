"""
Vanilla GAN trainer - No constraints (utility upper bound)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os


class VanillaGANTrainer:
    """
    Standard GAN training without any privacy or fairness constraints
    Serves as the utility upper bound for comparison
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

        # Optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config.LEARNING_RATE,
            betas=(config.BETA1, config.BETA2)
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config.LEARNING_RATE,
            betas=(config.BETA1, config.BETA2)
        )

        # Loss function
        self.criterion = nn.BCELoss()

        # Training history
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'd_real_acc': [],
            'd_fake_acc': []
        }

    def train_step(self, real_images):
        """
        Single training step

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
        # Train Discriminator
        # =====================
        self.d_optimizer.zero_grad()

        # Real images
        d_real_output = self.discriminator(real_images)
        d_real_loss = self.criterion(d_real_output, real_labels)

        # Fake images
        z = torch.randn(batch_size, self.config.LATENT_DIM, 1, 1, device=self.device)
        fake_images = self.generator(z)
        d_fake_output = self.discriminator(fake_images.detach())
        d_fake_loss = self.criterion(d_fake_output, fake_labels)

        # Total discriminator loss
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.d_optimizer.step()

        # Accuracy
        d_real_acc = ((d_real_output > 0.5).float() == real_labels).float().mean().item()
        d_fake_acc = ((d_fake_output < 0.5).float() == (fake_labels == 0)).float().mean().item()

        # ==================
        # Train Generator
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
        Train the GAN

        Args:
            train_loader: DataLoader for training data
            epochs: Number of epochs (uses config if None)

        Returns:
            Training history
        """
        if epochs is None:
            epochs = self.config.EPOCHS

        print(f"\n{'='*60}")
        print(f"Training Vanilla GAN")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}")
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

                # Update progress bar
                pbar.set_postfix({
                    'G_loss': f'{g_loss:.4f}',
                    'D_loss': f'{d_loss:.4f}',
                    'D_real_acc': f'{d_real_acc:.2f}',
                    'D_fake_acc': f'{d_fake_acc:.2f}'
                })

            # Average losses
            num_batches = len(train_loader)
            avg_g_loss = epoch_g_loss / num_batches
            avg_d_loss = epoch_d_loss / num_batches
            avg_d_real_acc = epoch_d_real_acc / num_batches
            avg_d_fake_acc = epoch_d_fake_acc / num_batches

            # Save history
            self.history['g_loss'].append(avg_g_loss)
            self.history['d_loss'].append(avg_d_loss)
            self.history['d_real_acc'].append(avg_d_real_acc)
            self.history['d_fake_acc'].append(avg_d_fake_acc)

            print(f"Epoch {epoch+1}/{epochs} - "
                  f"G_loss: {avg_g_loss:.4f}, D_loss: {avg_d_loss:.4f}, "
                  f"D_real_acc: {avg_d_real_acc:.2f}, D_fake_acc: {avg_d_fake_acc:.2f}")

        print(f"\n{'='*60}")
        print(f"Vanilla GAN Training Complete")
        print(f"{'='*60}\n")

        return self.history

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
