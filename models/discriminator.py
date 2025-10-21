"""
Discriminator network for FAIR-DP-GAN
"""
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    DCGAN-style Discriminator for distinguishing real vs fake images
    """

    def __init__(self, config):
        """
        Args:
            config: Configuration object
        """
        super(Discriminator, self).__init__()
        self.config = config
        self.disc_features = config.DISC_FEATURES
        self.num_channels = config.NUM_CHANNELS
        self.image_size = config.IMAGE_SIZE

        # Main discriminator network
        self.main = nn.Sequential(
            # Input: num_channels x image_size x image_size
            nn.Conv2d(
                self.num_channels,
                self.disc_features,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # State: disc_features x (image_size/2) x (image_size/2)

            nn.Conv2d(
                self.disc_features,
                self.disc_features * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(self.disc_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (disc_features*2) x (image_size/4) x (image_size/4)

            nn.Conv2d(
                self.disc_features * 2,
                self.disc_features * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(self.disc_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (disc_features*4) x (image_size/8) x (image_size/8)

            nn.Conv2d(
                self.disc_features * 4,
                self.disc_features * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(self.disc_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (disc_features*8) x (image_size/16) x (image_size/16)

            nn.Conv2d(
                self.disc_features * 8,
                1,
                kernel_size=self.image_size // 16,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.Sigmoid()
            # Output: 1 x 1 x 1
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input images of shape (batch_size, num_channels, image_size, image_size)

        Returns:
            Discriminator output of shape (batch_size, 1)
        """
        output = self.main(x)
        return output.view(-1, 1).squeeze(1)


class FairDiscriminator(nn.Module):
    """
    Discriminator with fairness constraint
    Outputs both real/fake prediction and sensitive attribute prediction
    """

    def __init__(self, config):
        """
        Args:
            config: Configuration object
        """
        super(FairDiscriminator, self).__init__()
        self.config = config
        self.disc_features = config.DISC_FEATURES
        self.num_channels = config.NUM_CHANNELS
        self.image_size = config.IMAGE_SIZE

        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(self.num_channels, self.disc_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.disc_features, self.disc_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.disc_features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.disc_features * 2, self.disc_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.disc_features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.disc_features * 4, self.disc_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.disc_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Real/Fake classifier head
        self.classifier = nn.Sequential(
            nn.Conv2d(self.disc_features * 8, 1, self.image_size // 16, 1, 0, bias=False),
            nn.Sigmoid()
        )

        # Fairness head (predicts sensitive attribute)
        self.fairness_head = nn.Sequential(
            nn.Conv2d(self.disc_features * 8, 1, self.image_size // 16, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x, return_fairness=False):
        """
        Forward pass

        Args:
            x: Input images
            return_fairness: If True, also return sensitive attribute prediction

        Returns:
            If return_fairness=False: real/fake predictions
            If return_fairness=True: (real/fake predictions, sensitive attribute predictions)
        """
        features = self.features(x)
        real_fake_pred = self.classifier(features).view(-1, 1).squeeze(1)

        if return_fairness:
            fairness_pred = self.fairness_head(features).view(-1, 1).squeeze(1)
            return real_fake_pred, fairness_pred
        else:
            return real_fake_pred
