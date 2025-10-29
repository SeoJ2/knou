"""
Generator Network for Fair-DP-GAN

This module implements a group-conditional generator based on DCGAN architecture.
The generator takes as input:
1. Random noise z ~ N(0, I)
2. Group label g ∈ {0, 1, ..., K-1}

And produces synthetic samples conditioned on the group label.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class Generator(nn.Module):
    """
    Group-conditional generator for Fair-DP-GAN.

    The generator uses a DCGAN-style architecture with transposed convolutions
    for image data, or fully-connected layers for tabular data.

    Args:
        latent_dim: Dimension of latent noise vector z
        n_groups: Number of demographic groups
        output_shape: Shape of generated samples
            - For images: (C, H, W) e.g., (3, 64, 64)
            - For tabular: (feature_dim,) e.g., (100,)
        embedding_dim: Dimension of group embedding (default: 50)
        base_channels: Base number of channels for conv layers (default: 64)
        data_type: 'image' or 'tabular'
    """

    def __init__(
        self,
        latent_dim: int,
        n_groups: int,
        output_shape: Tuple,
        embedding_dim: int = 50,
        base_channels: int = 64,
        data_type: str = 'image'
    ):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.n_groups = n_groups
        self.output_shape = output_shape
        self.embedding_dim = embedding_dim
        self.data_type = data_type

        # Group embedding
        self.group_embedding = nn.Embedding(n_groups, embedding_dim)

        # Combined input dimension
        self.input_dim = latent_dim + embedding_dim

        if data_type == 'image':
            self._build_image_generator(base_channels)
        else:
            self._build_tabular_generator()

    def _build_image_generator(self, base_channels: int):
        """
        Build DCGAN-style generator for image data.

        Architecture:
            Input: z (latent_dim) + embedding (embedding_dim)
            → FC + Reshape to (base_channels*8, 4, 4)
            → TransConv + BN + ReLU (base_channels*4, 8, 8)
            → TransConv + BN + ReLU (base_channels*2, 16, 16)
            → TransConv + BN + ReLU (base_channels, 32, 32)
            → TransConv + Tanh (3, 64, 64)
        """
        C, H, W = self.output_shape

        # Initial projection
        init_size = 4
        self.init_channels = base_channels * 8

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.init_channels * init_size * init_size),
            nn.BatchNorm1d(self.init_channels * init_size * init_size),
            nn.ReLU(inplace=True)
        )

        self.init_size = init_size

        # Transposed convolution layers
        self.conv_blocks = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(
                self.init_channels, base_channels * 4,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(
                base_channels * 4, base_channels * 2,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(
                base_channels * 2, base_channels,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(
                base_channels, C,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.Tanh()
        )

    def _build_tabular_generator(self):
        """
        Build fully-connected generator for tabular data.

        Architecture:
            Input: z (latent_dim) + embedding (embedding_dim)
            → FC + BN + LeakyReLU (256)
            → FC + BN + LeakyReLU (512)
            → FC + BN + LeakyReLU (512)
            → FC (output_dim)
        """
        output_dim = self.output_shape[0] if isinstance(self.output_shape, tuple) else self.output_shape

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, output_dim),
            # No final activation for tabular data (will be post-processed)
        )

    def forward(
        self,
        z: torch.Tensor,
        group_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate samples conditioned on group labels.

        Args:
            z: Latent noise, shape (batch_size, latent_dim)
            group_labels: Group labels, shape (batch_size,)

        Returns:
            Generated samples, shape (batch_size, *output_shape)
        """
        # Embed group labels
        group_embed = self.group_embedding(group_labels)  # (batch_size, embedding_dim)

        # Concatenate noise and group embedding
        gen_input = torch.cat([z, group_embed], dim=1)  # (batch_size, input_dim)

        if self.data_type == 'image':
            # Project and reshape
            out = self.fc(gen_input)
            out = out.view(out.size(0), self.init_channels, self.init_size, self.init_size)

            # Generate image
            img = self.conv_blocks(out)
            return img
        else:
            # Generate tabular data
            out = self.model(gen_input)
            return out

    def sample(
        self,
        n_samples: int,
        group_label: Optional[int] = None,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Sample from the generator.

        Args:
            n_samples: Number of samples to generate
            group_label: If provided, generate all samples from this group.
                        If None, sample uniformly from all groups.
            device: Device to generate samples on

        Returns:
            Generated samples
        """
        self.eval()
        with torch.no_grad():
            # Sample noise
            z = torch.randn(n_samples, self.latent_dim, device=device)

            # Sample or assign group labels
            if group_label is not None:
                group_labels = torch.full(
                    (n_samples,), group_label,
                    dtype=torch.long, device=device
                )
            else:
                group_labels = torch.randint(
                    0, self.n_groups, (n_samples,),
                    dtype=torch.long, device=device
                )

            # Generate
            samples = self.forward(z, group_labels)

        return samples

    def get_n_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initialize_weights(self):
        """
        Initialize weights using DCGAN initialization scheme.
        Conv layers: N(0, 0.02)
        BatchNorm: weight=1, bias=0
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def __repr__(self) -> str:
        return (
            f"Generator(\n"
            f"  data_type={self.data_type},\n"
            f"  latent_dim={self.latent_dim},\n"
            f"  n_groups={self.n_groups},\n"
            f"  output_shape={self.output_shape},\n"
            f"  n_params={self.get_n_params():,}\n"
            f")"
        )
