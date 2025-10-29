"""
Comprehensive Evaluation Metrics

Implements metrics for assessing:
1. Utility: FID, IS, Feature Distance
2. Fairness: SPD, EOD, Group-wise metrics
3. Privacy: Empirical privacy (via MIA)
4. TSTR: Train on Synthetic, Test on Real
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from scipy import linalg
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.models import inception_v3
import warnings

warnings.filterwarnings('ignore')


class Evaluator:
    """
    Comprehensive evaluator for Fair-DP-GAN.

    Provides metrics for:
    - **Utility**: FID, Inception Score
    - **Fairness**: Statistical Parity Difference, Equalized Odds Difference
    - **TSTR**: Train classifier on synthetic, test on real data

    Args:
        device: Device to run evaluation on
        data_type: 'image' or 'tabular'
        n_groups: Number of demographic groups
    """

    def __init__(
        self,
        device: str = 'cuda',
        data_type: str = 'image',
        n_groups: int = 2
    ):
        self.device = device
        self.data_type = data_type
        self.n_groups = n_groups

        # For image evaluation: load Inception model
        if data_type == 'image':
            self.inception_model = inception_v3(
                pretrained=True,
                transform_input=False
            ).to(device)
            self.inception_model.eval()
        else:
            self.inception_model = None

    def evaluate_all(
        self,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        real_groups: torch.Tensor,
        fake_groups: torch.Tensor,
        real_labels: Optional[torch.Tensor] = None,
        fake_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.

        Args:
            real_data: Real data samples
            fake_data: Generated synthetic samples
            real_groups: Group labels for real data
            fake_groups: Group labels for fake data
            real_labels: Target labels for TSTR (optional)
            fake_labels: Generated target labels for TSTR (optional)

        Returns:
            Dict with all metrics
        """
        metrics = {}

        # Utility metrics
        if self.data_type == 'image':
            metrics['fid'] = self.compute_fid(real_data, fake_data)
            metrics['inception_score'] = self.compute_inception_score(fake_data)
        else:
            metrics['feature_distance'] = self.compute_feature_distance(
                real_data, fake_data
            )

        # Fairness metrics
        fairness_metrics = self.compute_fairness_metrics(
            fake_data, fake_groups, real_groups
        )
        metrics.update(fairness_metrics)

        # Group-wise utility
        group_metrics = self.compute_group_wise_metrics(
            real_data, fake_data, real_groups, fake_groups
        )
        metrics.update(group_metrics)

        # TSTR (if labels available)
        if real_labels is not None and fake_labels is not None:
            tstr_metrics = self.compute_tstr(
                real_data, fake_data, real_labels, fake_labels
            )
            metrics.update(tstr_metrics)

        # Privacy-Utility-Fairness (PUF) composite score
        metrics['puf_score'] = self.compute_puf_score(metrics)

        return metrics

    def compute_fid(
        self,
        real_data: torch.Tensor,
        fake_data: torch.Tensor
    ) -> float:
        """
        Compute Fréchet Inception Distance.

        FID = ||μ_r - μ_f||² + Tr(Σ_r + Σ_f - 2√(Σ_r Σ_f))

        Lower is better (0 = perfect match).

        Args:
            real_data: Real images, shape (N, C, H, W)
            fake_data: Fake images, shape (N, C, H, W)

        Returns:
            FID score
        """
        if self.inception_model is None:
            return 0.0

        # Get Inception features
        real_features = self._get_inception_features(real_data)
        fake_features = self._get_inception_features(fake_data)

        # Compute statistics
        mu_real = np.mean(real_features, axis=0)
        mu_fake = np.mean(fake_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        sigma_fake = np.cov(fake_features, rowvar=False)

        # Compute FID
        diff = mu_real - mu_fake
        covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)

        return float(fid)

    def _get_inception_features(self, images: torch.Tensor) -> np.ndarray:
        """Extract features from Inception network."""
        features = []

        with torch.no_grad():
            for i in range(0, len(images), 32):
                batch = images[i:i+32].to(self.device)

                # Resize to 299x299 for Inception
                if batch.size(2) != 299 or batch.size(3) != 299:
                    batch = torch.nn.functional.interpolate(
                        batch, size=(299, 299),
                        mode='bilinear', align_corners=False
                    )

                # Get features
                pred = self.inception_model(batch)

                # Pool to fixed size
                if pred.dim() == 4:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                    pred = pred.squeeze(-1).squeeze(-1)

                features.append(pred.cpu().numpy())

        return np.concatenate(features, axis=0)

    def compute_inception_score(
        self,
        fake_data: torch.Tensor,
        splits: int = 10
    ) -> float:
        """
        Compute Inception Score.

        IS = exp(E[KL(p(y|x) || p(y))])

        Higher is better (more diverse and recognizable).

        Args:
            fake_data: Generated images
            splits: Number of splits for std computation

        Returns:
            Mean Inception Score
        """
        if self.inception_model is None:
            return 0.0

        # Get predictions
        preds = []

        with torch.no_grad():
            for i in range(0, len(fake_data), 32):
                batch = fake_data[i:i+32].to(self.device)

                # Resize to 299x299
                if batch.size(2) != 299 or batch.size(3) != 299:
                    batch = torch.nn.functional.interpolate(
                        batch, size=(299, 299),
                        mode='bilinear', align_corners=False
                    )

                # Get predictions
                pred = self.inception_model(batch)
                pred = torch.nn.functional.softmax(pred, dim=1)
                preds.append(pred.cpu().numpy())

        preds = np.concatenate(preds, axis=0)

        # Compute IS for each split
        split_scores = []
        n_samples = len(preds)
        split_size = n_samples // splits

        for k in range(splits):
            part = preds[k * split_size:(k + 1) * split_size]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(len(part)):
                pyx = part[i]
                scores.append(np.sum(pyx * np.log(pyx / py + 1e-10)))
            split_scores.append(np.exp(np.mean(scores)))

        return float(np.mean(split_scores))

    def compute_feature_distance(
        self,
        real_data: torch.Tensor,
        fake_data: torch.Tensor
    ) -> float:
        """
        Compute L2 distance between real and fake feature distributions.
        For tabular data.

        Args:
            real_data: Real samples
            fake_data: Fake samples

        Returns:
            Mean L2 distance
        """
        real_mean = real_data.mean(dim=0)
        fake_mean = fake_data.mean(dim=0)

        distance = torch.norm(real_mean - fake_mean, p=2).item()

        return distance

    def compute_fairness_metrics(
        self,
        fake_data: torch.Tensor,
        fake_groups: torch.Tensor,
        real_groups: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute fairness metrics.

        SPD (Statistical Parity Difference):
            |P(Ŷ=1|G=0) - P(Ŷ=1|G=1)|
            Measures if generator produces balanced output across groups.

        For GANs, we measure: |P(G=0) - P(G=1)| in generated data
        Compared to real distribution.

        Args:
            fake_data: Generated samples
            fake_groups: Group labels for generated samples
            real_groups: Group labels for real data (for comparison)

        Returns:
            Dict with SPD and other fairness metrics
        """
        # Count group distribution
        real_counts = torch.bincount(real_groups, minlength=self.n_groups).float()
        fake_counts = torch.bincount(fake_groups, minlength=self.n_groups).float()

        real_dist = real_counts / real_counts.sum()
        fake_dist = fake_counts / fake_counts.sum()

        # Statistical Parity Difference
        spd = torch.abs(real_dist - fake_dist).max().item()

        # Demographic Parity (max difference from uniform)
        uniform_dist = torch.ones(self.n_groups) / self.n_groups
        dp = torch.abs(fake_dist - uniform_dist).max().item()

        return {
            'spd': spd,
            'demographic_parity': dp,
            'group_dist_real': real_dist.tolist(),
            'group_dist_fake': fake_dist.tolist()
        }

    def compute_group_wise_metrics(
        self,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        real_groups: torch.Tensor,
        fake_groups: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute quality metrics per group.

        Returns:
            Dict with per-group quality scores
        """
        metrics = {}

        for g in range(self.n_groups):
            # Select samples from this group
            real_mask = (real_groups == g)
            fake_mask = (fake_groups == g)

            if real_mask.sum() == 0 or fake_mask.sum() == 0:
                continue

            real_group_data = real_data[real_mask]
            fake_group_data = fake_data[fake_mask]

            # Compute distance for this group
            if len(real_group_data) > 0 and len(fake_group_data) > 0:
                real_mean = real_group_data.mean(dim=0)
                fake_mean = fake_group_data.mean(dim=0)
                distance = torch.norm(real_mean - fake_mean, p=2).item()

                metrics[f'group_{g}_distance'] = distance

        return metrics

    def compute_tstr(
        self,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        real_labels: torch.Tensor,
        fake_labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Train on Synthetic, Test on Real (TSTR).

        Measures practical utility: can we train a classifier on synthetic
        data and get good performance on real test data?

        Args:
            real_data: Real test data
            fake_data: Synthetic training data
            real_labels: Real test labels
            fake_labels: Synthetic training labels

        Returns:
            Dict with TSTR accuracy
        """
        # Convert to numpy
        X_train = fake_data.cpu().numpy()
        y_train = fake_labels.cpu().numpy()
        X_test = real_data.cpu().numpy()
        y_test = real_labels.cpu().numpy()

        # Flatten if image data
        if len(X_train.shape) > 2:
            X_train = X_train.reshape(len(X_train), -1)
            X_test = X_test.reshape(len(X_test), -1)

        # Train classifier on synthetic data
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)

        # Test on real data
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return {
            'tstr_accuracy': accuracy
        }

    def compute_puf_score(self, metrics: Dict[str, float]) -> float:
        """
        Compute Privacy-Utility-Fairness (PUF) composite score.

        PUF = α * Utility + β * Fairness + γ * Privacy

        Normalized to [0, 1] where 1 is best.

        Args:
            metrics: Dict containing individual metrics

        Returns:
            PUF score
        """
        # Utility component (lower FID or distance is better)
        if 'fid' in metrics:
            utility = 1 / (1 + metrics['fid'] / 100)  # Normalize
        elif 'feature_distance' in metrics:
            utility = 1 / (1 + metrics['feature_distance'])
        else:
            utility = 0.5

        # Fairness component (lower SPD is better)
        fairness = 1 - metrics.get('spd', 0.5)

        # Privacy component (would come from MIA or epsilon)
        # For now, use placeholder
        privacy = 0.5

        # Weighted combination
        alpha, beta, gamma = 0.4, 0.4, 0.2
        puf = alpha * utility + beta * fairness + gamma * privacy

        return puf

    def get_summary(self, metrics: Dict[str, float]) -> str:
        """
        Get formatted summary of evaluation metrics.

        Args:
            metrics: Dict of metrics

        Returns:
            Formatted string
        """
        summary = "=== Evaluation Summary ===\n\n"

        # Utility
        summary += "UTILITY:\n"
        if 'fid' in metrics:
            summary += f"  FID: {metrics['fid']:.2f}\n"
        if 'inception_score' in metrics:
            summary += f"  Inception Score: {metrics['inception_score']:.2f}\n"
        if 'feature_distance' in metrics:
            summary += f"  Feature Distance: {metrics['feature_distance']:.4f}\n"

        # Fairness
        summary += "\nFAIRNESS:\n"
        summary += f"  SPD: {metrics.get('spd', 0):.4f}\n"
        summary += f"  Demographic Parity: {metrics.get('demographic_parity', 0):.4f}\n"

        # TSTR
        if 'tstr_accuracy' in metrics:
            summary += "\nTSTR:\n"
            summary += f"  Accuracy: {metrics['tstr_accuracy']:.4f}\n"

        # PUF Score
        summary += f"\nPUF Score: {metrics.get('puf_score', 0):.4f}\n"

        return summary
