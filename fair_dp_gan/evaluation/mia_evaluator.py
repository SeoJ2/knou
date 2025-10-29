"""
Membership Inference Attack (MIA) Evaluator

Evaluates empirical privacy by attempting to infer whether a sample
was in the training dataset. Higher attack accuracy = worse privacy.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


class MIAEvaluator:
    """
    Membership Inference Attack evaluator.

    Implements threshold-based and ML-based attacks to test privacy:
    1. **Threshold Attack**: Use discriminator score to infer membership
    2. **ML Attack**: Train classifier on features to predict membership

    High attack success rate indicates privacy leakage.

    Args:
        critic: Trained critic/discriminator model
        device: Device to run evaluation on
    """

    def __init__(
        self,
        critic: nn.Module,
        device: str = 'cuda'
    ):
        self.critic = critic
        self.device = device
        self.critic.eval()

    def evaluate_privacy(
        self,
        train_data: torch.Tensor,
        test_data: torch.Tensor,
        n_samples: int = 1000
    ) -> Dict[str, float]:
        """
        Evaluate privacy via membership inference attacks.

        Args:
            train_data: Training data (members)
            test_data: Test data (non-members)
            n_samples: Number of samples to use for evaluation

        Returns:
            Dict with attack metrics
        """
        # Sample equal number from train and test
        n_per_class = min(n_samples // 2, len(train_data), len(test_data))

        train_indices = np.random.choice(len(train_data), n_per_class, replace=False)
        test_indices = np.random.choice(len(test_data), n_per_class, replace=False)

        members = train_data[train_indices].to(self.device)
        non_members = test_data[test_indices].to(self.device)

        # Labels: 1 = member, 0 = non-member
        member_labels = np.ones(len(members))
        non_member_labels = np.zeros(len(non_members))

        # Combine
        all_data = torch.cat([members, non_members], dim=0)
        all_labels = np.concatenate([member_labels, non_member_labels])

        # Extract features from critic
        features, scores = self._extract_features(all_data)

        # Threshold-based attack
        threshold_metrics = self._threshold_attack(scores, all_labels)

        # ML-based attack
        ml_metrics = self._ml_attack(features, all_labels)

        # Combine metrics
        metrics = {
            'mia_threshold_accuracy': threshold_metrics['accuracy'],
            'mia_threshold_auc': threshold_metrics['auc'],
            'mia_ml_accuracy': ml_metrics['accuracy'],
            'mia_ml_auc': ml_metrics['auc'],
            'privacy_risk': (threshold_metrics['auc'] + ml_metrics['auc']) / 2
        }

        return metrics

    def _extract_features(
        self,
        data: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and scores from critic.

        Args:
            data: Input samples

        Returns:
            features: Intermediate features
            scores: Discriminator scores
        """
        features_list = []
        scores_list = []

        with torch.no_grad():
            for i in range(0, len(data), 32):
                batch = data[i:i+32]

                # Get critic outputs with features
                outputs = self.critic(batch, return_features=True)

                scores_list.append(outputs['score'].cpu().numpy())

                if 'features' in outputs:
                    features_list.append(outputs['features'].cpu().numpy())

        scores = np.concatenate(scores_list, axis=0)

        if features_list:
            features = np.concatenate(features_list, axis=0)
        else:
            # Use scores as features if no intermediate features
            features = scores

        return features, scores

    def _threshold_attack(
        self,
        scores: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Threshold-based MIA.

        Attack strategy: Samples with higher discriminator scores
        are more likely to be members.

        Args:
            scores: Discriminator scores
            labels: True membership labels (1=member, 0=non-member)

        Returns:
            Attack metrics
        """
        # Try different thresholds
        thresholds = np.percentile(scores, np.linspace(0, 100, 100))
        best_accuracy = 0

        for threshold in thresholds:
            predictions = (scores > threshold).astype(int)
            accuracy = accuracy_score(labels, predictions)
            best_accuracy = max(best_accuracy, accuracy)

        # Compute AUC
        # Higher scores should indicate membership
        auc = roc_auc_score(labels, scores.flatten())

        return {
            'accuracy': best_accuracy,
            'auc': auc
        }

    def _ml_attack(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        ML-based MIA using Random Forest.

        Train a classifier on critic features to predict membership.

        Args:
            features: Extracted features
            labels: True membership labels

        Returns:
            Attack metrics
        """
        # Split into train/test for attack model
        n_samples = len(features)
        n_train = int(0.7 * n_samples)

        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        X_train = features[train_idx]
        y_train = labels[train_idx]
        X_test = features[test_idx]
        y_test = labels[test_idx]

        # Flatten features if needed
        if len(X_train.shape) > 2:
            X_train = X_train.reshape(len(X_train), -1)
            X_test = X_test.reshape(len(X_test), -1)

        # Train attack model
        attack_model = RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        )
        attack_model.fit(X_train, y_train)

        # Evaluate
        y_pred = attack_model.predict(X_test)
        y_pred_proba = attack_model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        return {
            'accuracy': accuracy,
            'auc': auc
        }

    def get_privacy_score(self, metrics: Dict[str, float]) -> float:
        """
        Convert MIA results to privacy score.

        Privacy score ∈ [0, 1] where:
        - 1.0 = perfect privacy (attack at random chance)
        - 0.0 = no privacy (perfect attack)

        Args:
            metrics: Dict containing MIA metrics

        Returns:
            Privacy score
        """
        # AUC of 0.5 = random guessing = perfect privacy
        # AUC of 1.0 = perfect attack = no privacy

        avg_auc = metrics.get('privacy_risk', 0.5)

        # Convert: AUC 0.5 -> score 1.0, AUC 1.0 -> score 0.0
        privacy_score = max(0, 2 * (1 - avg_auc))

        return privacy_score

    def __repr__(self) -> str:
        return "MIAEvaluator(threshold + ML attacks)"
