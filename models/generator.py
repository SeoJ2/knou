"""
Generator network for FAIR-DP-GAN
"""
import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    DCGAN-style Generator for image generation
    """

    def __init__(self, config):
        """
        Args:
            config: Configuration object
        """
        super(Generator, self).__init__()
        self.config = config
        self.latent_dim = config.LATENT_DIM
        self.gen_features = config.GEN_FEATURES
        self.num_channels = config.NUM_CHANNELS
        self.image_size = config.IMAGE_SIZE

        # Calculate the initial size
        self.init_size = config.IMAGE_SIZE // 16  # 4 upsampling layers

        # Main generator network
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(
                self.latent_dim,
                self.gen_features * 8,
                kernel_size=self.init_size,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(self.gen_features * 8),
            nn.ReLU(True),
            # State: (gen_features*8) x init_size x init_size

            nn.ConvTranspose2d(
                self.gen_features * 8,
                self.gen_features * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(self.gen_features * 4),
            nn.ReLU(True),
            # State: (gen_features*4) x (init_size*2) x (init_size*2)

            nn.ConvTranspose2d(
                self.gen_features * 4,
                self.gen_features * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(self.gen_features * 2),
            nn.ReLU(True),
            # State: (gen_features*2) x (init_size*4) x (init_size*4)

            nn.ConvTranspose2d(
                self.gen_features * 2,
                self.gen_features,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(self.gen_features),
            nn.ReLU(True),
            # State: gen_features x (init_size*8) x (init_size*8)

            nn.ConvTranspose2d(
                self.gen_features,
                self.num_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.Tanh()
            # Output: num_channels x image_size x image_size
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, z):
        """
        Forward pass

        Args:
            z: Latent noise tensor of shape (batch_size, latent_dim, 1, 1)
               or (batch_size, latent_dim)

        Returns:
            Generated images of shape (batch_size, num_channels, image_size, image_size)
        """
        # Reshape if needed
        if z.dim() == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)

        return self.main(z)

    def generate(self, num_samples, device=None):
        """
        Generate random samples

        Args:
            num_samples: Number of samples to generate
            device: Device to generate on

        Returns:
            Generated images
        """
        if device is None:
            device = next(self.parameters()).device

        z = torch.randn(num_samples, self.latent_dim, 1, 1, device=device)
        with torch.no_grad():
            fake_images = self.forward(z)
        return fake_images


def weights_init_normal(m):
    """Custom weights initialization for Generator"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
