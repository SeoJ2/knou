"""
Adult Income Dataset Loader with Group-aware Splitting

This module handles loading and preprocessing of the Adult Income dataset,
which is used for tabular data generation tasks. It supports group-aware data splitting
based on sensitive attributes (e.g., gender, race) to enable fairness-aware training.
"""

import os
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


class AdultDataset(Dataset):
    """
    Adult Income dataset with group labels.

    Args:
        data: Preprocessed feature data
        groups: Group labels
        labels: Target labels (optional, for TSTR evaluation)
    """

    def __init__(
        self,
        data: np.ndarray,
        groups: np.ndarray,
        labels: Optional[np.ndarray] = None
    ):
        self.data = torch.FloatTensor(data)
        self.groups = torch.LongTensor(groups)
        self.labels = torch.LongTensor(labels) if labels is not None else None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns:
            features: Feature vector
            group: Group label
            label: Target label (if available)
            idx: Original index
        """
        if self.labels is not None:
            return self.data[idx], self.groups[idx], self.labels[idx], idx
        return self.data[idx], self.groups[idx], idx


class AdultDataLoader:
    """
    Data loader for Adult Income dataset with group-aware splitting.

    This loader processes the Adult Income dataset, encodes categorical variables,
    and provides group-aware splitting for fairness-aware training.

    The Adult dataset contains 14 attributes including age, workclass, education,
    occupation, etc., with the task of predicting whether income exceeds $50K.

    Args:
        data_path: Path to Adult dataset CSV file (if None, downloads from UCI)
        sensitive_attr: Attribute to use for group division (default: 'sex')
        batch_size: Batch size for training
        test_split: Fraction of data to use for testing
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        sensitive_attr: str = 'sex',
        batch_size: int = 256,
        test_split: float = 0.2
    ):
        self.data_path = data_path
        self.sensitive_attr = sensitive_attr
        self.batch_size = batch_size
        self.test_split = test_split

        # Column names for Adult dataset
        self.column_names = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
            'income'
        ]

        # Load and preprocess data
        self._load_and_preprocess()

        # Get group statistics
        self.group_info = self._compute_group_info()

    def _load_and_preprocess(self):
        """Load and preprocess the Adult dataset."""
        # Load data
        if self.data_path and os.path.exists(self.data_path):
            df = pd.read_csv(self.data_path, names=self.column_names,
                           skipinitialspace=True, na_values='?')
        else:
            # Create synthetic data for testing if file doesn't exist
            print("Warning: Data file not found. Creating synthetic data.")
            df = self._create_synthetic_data()

        # Remove missing values
        df = df.dropna()

        # Extract target variable
        self.target_encoder = LabelEncoder()
        labels = self.target_encoder.fit_transform(df['income'])

        # Extract sensitive attribute for grouping
        if self.sensitive_attr in df.columns:
            if df[self.sensitive_attr].dtype == 'object':
                # For categorical sensitive attributes
                sensitive_encoder = LabelEncoder()
                groups = sensitive_encoder.fit_transform(df[self.sensitive_attr])
            else:
                # For numerical sensitive attributes (e.g., age > 30)
                groups = (df[self.sensitive_attr] > df[self.sensitive_attr].median()).astype(int).values
        else:
            # Default: random groups
            groups = np.random.randint(0, 2, size=len(df))

        # Remove target and prepare features
        df = df.drop(['income'], axis=1)

        # Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        self.label_encoders = {}

        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col])

        # Normalize numerical features
        self.scaler = StandardScaler()
        data = self.scaler.fit_transform(df.values)

        # Split into train/test
        n_samples = len(data)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        split_idx = int(n_samples * (1 - self.test_split))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        # Create train/test datasets
        self.train_dataset = AdultDataset(
            data[train_indices],
            groups[train_indices],
            labels[train_indices]
        )

        self.test_dataset = AdultDataset(
            data[test_indices],
            groups[test_indices],
            labels[test_indices]
        )

        self.feature_dim = data.shape[1]
        self.n_groups = len(np.unique(groups))

    def _create_synthetic_data(self) -> pd.DataFrame:
        """Create synthetic Adult-like dataset for testing."""
        n_samples = 10000

        data = {
            'age': np.random.randint(18, 80, n_samples),
            'workclass': np.random.choice(['Private', 'Self-emp', 'Gov'], n_samples),
            'fnlwgt': np.random.randint(10000, 500000, n_samples),
            'education': np.random.choice(['HS-grad', 'Bachelors', 'Masters', 'Doctorate'], n_samples),
            'education-num': np.random.randint(9, 16, n_samples),
            'marital-status': np.random.choice(['Married', 'Single', 'Divorced'], n_samples),
            'occupation': np.random.choice(['Tech-support', 'Sales', 'Other'], n_samples),
            'relationship': np.random.choice(['Husband', 'Wife', 'Own-child'], n_samples),
            'race': np.random.choice(['White', 'Black', 'Asian'], n_samples),
            'sex': np.random.choice(['Male', 'Female'], n_samples),
            'capital-gain': np.random.randint(0, 100000, n_samples),
            'capital-loss': np.random.randint(0, 5000, n_samples),
            'hours-per-week': np.random.randint(20, 80, n_samples),
            'native-country': np.random.choice(['United-States', 'Mexico', 'Other'], n_samples),
            'income': np.random.choice(['<=50K', '>50K'], n_samples)
        }

        return pd.DataFrame(data)

    def _compute_group_info(self) -> Dict:
        """Compute statistics about groups in the dataset."""
        groups = self.train_dataset.groups.numpy()
        unique_groups, counts = np.unique(groups, return_counts=True)

        group_counts = {int(g): int(c) for g, c in zip(unique_groups, counts)}
        total = sum(group_counts.values())
        group_ratios = {g: count / total for g, count in group_counts.items()}

        return {
            'counts': group_counts,
            'ratios': group_ratios,
            'total': total,
            'n_groups': self.n_groups
        }

    def get_train_loader(self, shuffle: bool = True) -> DataLoader:
        """Get training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True
        )

    def get_test_loader(self) -> DataLoader:
        """Get test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

    def get_group_loader(self, group: int, split: str = 'train') -> DataLoader:
        """
        Get data loader for a specific group.

        Args:
            group: Group index
            split: 'train' or 'test'
        """
        dataset = self.train_dataset if split == 'train' else self.test_dataset

        # Filter indices for this group
        group_mask = (dataset.groups == group).numpy()
        group_indices = np.where(group_mask)[0]

        group_dataset = Subset(dataset, group_indices)

        return DataLoader(
            group_dataset,
            batch_size=self.batch_size,
            shuffle=(split == 'train'),
            num_workers=2,
            pin_memory=True
        )

    def get_balanced_loader(self) -> DataLoader:
        """
        Get a balanced data loader with equal samples from each group.
        """
        groups = self.train_dataset.groups.numpy()

        # Get indices for each group
        group_indices = {}
        for g in range(self.n_groups):
            group_indices[g] = np.where(groups == g)[0]

        # Sample equally from each group
        min_count = min(len(indices) for indices in group_indices.values())
        balanced_indices = []
        for indices in group_indices.values():
            sampled = np.random.choice(indices, min_count, replace=False)
            balanced_indices.extend(sampled)

        np.random.shuffle(balanced_indices)
        balanced_dataset = Subset(self.train_dataset, balanced_indices)

        return DataLoader(
            balanced_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

    def get_feature_dim(self) -> int:
        """Get the dimension of feature vectors."""
        return self.feature_dim

    def get_group_info_summary(self) -> str:
        """Get a formatted summary of group statistics."""
        info = self.group_info
        summary = f"Adult Dataset Summary:\n"
        summary += f"  Sensitive Attribute: {self.sensitive_attr}\n"
        summary += f"  Feature Dimension: {self.feature_dim}\n"
        summary += f"  Total Training Samples: {info['total']}\n"
        summary += f"  Number of Groups: {info['n_groups']}\n"
        for group, count in info['counts'].items():
            ratio = info['ratios'][group]
            summary += f"    Group {group}: {count} samples ({ratio:.2%})\n"
        return summary

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized data back to original scale.
        Useful for interpreting generated synthetic data.
        """
        return self.scaler.inverse_transform(data)
