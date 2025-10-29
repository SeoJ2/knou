"""
Critic (Discriminator) Network for Fair-DP-GAN

This module implements a multi-head critic based on WGAN-GP architecture.
The critic provides:
1. Real/fake discrimination (main task)
2. Optional: Group classification (for fairness regularization)
3. Wasserstein distance estimation
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict


class Critic(nn.Module):
    """
    Multi-head critic for Fair-DP-GAN.

    Uses WGAN-GP style critic (no sigmoid at output) with optional auxiliary
    heads for group classification and fairness enforcement.

    Args:
        input_shape: Shape of input samples
            - For images: (C, H, W) e.g., (3, 64, 64)
            - For tabular: (feature_dim,) e.g., (100,)
        n_groups: Number of demographic groups
        base_channels: Base number of channels for conv layers (default: 64)
        data_type: 'image' or 'tabular'
        use_group_head: Whether to include group classification head
        use_spectral_norm: Whether to use spectral normalization (alternative to GP)
    """

    def __init__(
        self,
        input_shape: Tuple,
        n_groups: int,
        base_channels: int = 64,
        data_type: str = 'image',
        use_group_head: bool = True,
        use_spectral_norm: bool = False
    ):
        super(Critic, self).__init__()

        self.input_shape = input_shape
        self.n_groups = n_groups
        self.data_type = data_type
        self.use_group_head = use_group_head
        self.use_spectral_norm = use_spectral_norm

        if data_type == 'image':
            self._build_image_critic(base_channels)
        else:
            self._build_tabular_critic()

    def _build_image_critic(self, base_channels: int):
        """
        Build DCGAN-style critic for image data.

        Architecture:
            Input: (3, 64, 64)
            → Conv + LeakyReLU (base_channels, 32, 32)
            → Conv + BN + LeakyReLU (base_channels*2, 16, 16)
            → Conv + BN + LeakyReLU (base_channels*4, 8, 8)
            → Conv + BN + LeakyReLU (base_channels*8, 4, 4)
            → Flatten
            → FC (feature_dim=512)

        Then split into heads:
            - Main head: FC (1) for real/fake score
            - Group head: FC (n_groups) for group classification
        """
        C, H, W = self.input_shape

        def conv_block(in_channels, out_channels, normalize=True):
            layers = [
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)
            ]
            if normalize:
                if self.use_spectral_norm:
                    layers[0] = nn.utils.spectral_norm(layers[0])
                else:
                    layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Feature extraction
        self.feature_extractor = nn.Sequential(
            # 64x64 -> 32x32
            *conv_block(C, base_channels, normalize=False),

            # 32x32 -> 16x16
            *conv_block(base_channels, base_channels * 2),

            # 16x16 -> 8x8
            *conv_block(base_channels * 2, base_channels * 4),

            # 8x8 -> 4x4
            *conv_block(base_channels * 4, base_channels * 8),

            nn.Flatten()
        )

        # Compute feature dimension
        feature_dim = base_channels * 8 * 4 * 4

        # Shared feature layer
        self.shared_fc = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Main head: real/fake discrimination
        self.main_head = nn.Linear(512, 1)

        # Group classification head (auxiliary task)
        if self.use_group_head:
            self.group_head = nn.Sequential(
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, self.n_groups)
            )

    def _build_tabular_critic(self):
        """
        Build fully-connected critic for tabular data.

        Architecture:
            Input: (feature_dim,)
            → FC + LeakyReLU (512)
            → FC + LeakyReLU (512)
            → FC + LeakyReLU (256)
            → Split into heads
        """
        input_dim = self.input_shape[0] if isinstance(self.input_shape, tuple) else self.input_shape

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )

        # Main head: real/fake discrimination
        self.main_head = nn.Linear(256, 1)

        # Group classification head
        if self.use_group_head:
            self.group_head = nn.Sequential(
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(128, self.n_groups)
            )

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through critic.

        Args:
            x: Input samples
            return_features: If True, also return intermediate features

        Returns:
            Dict with keys:
                - 'score': Real/fake scores (no sigmoid)
                - 'group_logits': Group classification logits (if use_group_head)
                - 'features': Intermediate features (if return_features)
        """
        # Extract features
        features = self.feature_extractor(x)

        if self.data_type == 'image':
            shared_features = self.shared_fc(features)
        else:
            shared_features = features

        # Main head: real/fake score
        score = self.main_head(shared_features)

        outputs = {'score': score}

        # Group classification head
        if self.use_group_head:
            group_logits = self.group_head(shared_features)
            outputs['group_logits'] = group_logits

        # Return features if requested
        if return_features:
            outputs['features'] = shared_features

        return outputs

    def compute_gradient_penalty(
        self,
        real_samples: torch.Tensor,
        fake_samples: torch.Tensor,
        lambda_gp: float = 10.0
    ) -> torch.Tensor:
        """
        Compute gradient penalty for WGAN-GP.

        GP = λ * E[(||∇_x D(x)||_2 - 1)^2]

        where x is sampled uniformly along lines between real and fake samples.

        Args:
            real_samples: Real data samples
            fake_samples: Generated samples
            lambda_gp: Gradient penalty coefficient

        Returns:
            Gradient penalty loss
        """
        batch_size = real_samples.size(0)
        device = real_samples.device

        # Random weight for interpolation
        alpha = torch.rand(batch_size, 1, device=device)

        # Expand alpha to match sample dimensions
        if self.data_type == 'image':
            alpha = alpha.view(batch_size, 1, 1, 1)

        # Interpolated samples
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

        # Critic scores for interpolated samples
        d_interpolates = self.forward(interpolates)['score']

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Flatten gradients
        gradients = gradients.view(batch_size, -1)

        # Compute gradient penalty
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()

        return gradient_penalty

    def get_n_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initialize_weights(self):
        """
        Initialize weights using DCGAN initialization scheme.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def __repr__(self) -> str:
        return (
            f"Critic(\n"
            f"  data_type={self.data_type},\n"
            f"  input_shape={self.input_shape},\n"
            f"  n_groups={self.n_groups},\n"
            f"  use_group_head={self.use_group_head},\n"
            f"  n_params={self.get_n_params():,}\n"
            f")"
        )
