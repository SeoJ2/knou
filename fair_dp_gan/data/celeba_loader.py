"""
CelebA Dataset Loader with Group-aware Splitting

This module handles loading and preprocessing of the CelebA face dataset,
which is used for image generation tasks. It supports group-aware data splitting
based on sensitive attributes (e.g., gender, age) to enable fairness-aware training.
"""

import os
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd


class CelebADataset(Dataset):
    """
    CelebA dataset with group labels.

    Args:
        root_dir: Root directory containing CelebA images and attributes
        transform: Image transformations to apply
        sensitive_attr: Sensitive attribute for grouping (e.g., 'Male', 'Young')
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        sensitive_attr: str = 'Male'
    ):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'img_align_celeba')
        self.attr_file = os.path.join(root_dir, 'list_attr_celeba.txt')
        self.transform = transform or self._default_transform()
        self.sensitive_attr = sensitive_attr

        # Load attributes
        self.attributes = self._load_attributes()
        self.image_names = list(self.attributes.index)

    def _default_transform(self) -> transforms.Compose:
        """Default image preprocessing pipeline."""
        return transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def _load_attributes(self) -> pd.DataFrame:
        """Load CelebA attributes from text file."""
        try:
            # Read attributes file (skip first line which contains count)
            df = pd.read_csv(
                self.attr_file,
                delim_whitespace=True,
                skiprows=1
            )
            # Convert -1/1 to 0/1
            df = (df + 1) // 2
            return df
        except FileNotFoundError:
            # If attribute file doesn't exist, create dummy data for testing
            print(f"Warning: {self.attr_file} not found. Creating dummy data.")
            return pd.DataFrame()

    def __len__(self) -> int:
        return len(self.image_names) if len(self.attributes) > 0 else 0

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        Returns:
            image: Preprocessed image tensor
            group: Group label (0 or 1)
            idx: Original index
        """
        if len(self.attributes) == 0:
            # Return dummy data for testing
            img = torch.randn(3, 64, 64)
            group = np.random.randint(0, 2)
            return img, group, idx

        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # Load and transform image
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except:
            # If image doesn't exist, return random tensor
            image = torch.randn(3, 64, 64)

        # Get group label
        group = int(self.attributes.loc[img_name, self.sensitive_attr])

        return image, group, idx


class CelebADataLoader:
    """
    Data loader for CelebA dataset with group-aware splitting.

    This loader divides the dataset into groups based on sensitive attributes
    and provides stratified sampling capabilities for fairness-aware training.

    Args:
        root_dir: Root directory of CelebA dataset
        sensitive_attr: Attribute to use for group division (default: 'Male')
        batch_size: Batch size for training
        image_size: Size to resize images to
        test_split: Fraction of data to use for testing
    """

    def __init__(
        self,
        root_dir: str,
        sensitive_attr: str = 'Male',
        batch_size: int = 128,
        image_size: int = 64,
        test_split: float = 0.2
    ):
        self.root_dir = root_dir
        self.sensitive_attr = sensitive_attr
        self.batch_size = batch_size
        self.image_size = image_size
        self.test_split = test_split

        # Create transforms
        self.transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Load full dataset
        self.full_dataset = CelebADataset(
            root_dir=root_dir,
            transform=self.transform,
            sensitive_attr=sensitive_attr
        )

        # Split into train/test
        self._split_dataset()

        # Get group statistics
        self.group_info = self._compute_group_info()

    def _split_dataset(self):
        """Split dataset into train and test sets."""
        n_samples = len(self.full_dataset)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        split_idx = int(n_samples * (1 - self.test_split))
        self.train_indices = indices[:split_idx]
        self.test_indices = indices[split_idx:]

        self.train_dataset = Subset(self.full_dataset, self.train_indices)
        self.test_dataset = Subset(self.full_dataset, self.test_indices)

    def _compute_group_info(self) -> Dict:
        """Compute statistics about groups in the dataset."""
        group_counts = {0: 0, 1: 0}

        for idx in self.train_indices:
            _, group, _ = self.full_dataset[idx]
            group_counts[group] += 1

        total = sum(group_counts.values())
        group_ratios = {g: count / total for g, count in group_counts.items()}

        return {
            'counts': group_counts,
            'ratios': group_ratios,
            'total': total,
            'n_groups': len(group_counts)
        }

    def get_train_loader(self, shuffle: bool = True) -> DataLoader:
        """Get training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True
        )

    def get_test_loader(self) -> DataLoader:
        """Get test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def get_group_loader(self, group: int, split: str = 'train') -> DataLoader:
        """
        Get data loader for a specific group.

        Args:
            group: Group index (0 or 1)
            split: 'train' or 'test'
        """
        indices = self.train_indices if split == 'train' else self.test_indices

        # Filter indices for this group
        group_indices = [
            idx for idx in indices
            if self.full_dataset[idx][1] == group
        ]

        group_dataset = Subset(self.full_dataset, group_indices)

        return DataLoader(
            group_dataset,
            batch_size=self.batch_size,
            shuffle=(split == 'train'),
            num_workers=4,
            pin_memory=True
        )

    def get_balanced_loader(self) -> DataLoader:
        """
        Get a balanced data loader with equal samples from each group.
        This is useful for fairness-aware training.
        """
        # Get indices for each group
        group_indices = {0: [], 1: []}
        for idx in self.train_indices:
            _, group, _ = self.full_dataset[idx]
            group_indices[group].append(idx)

        # Sample equally from each group
        min_count = min(len(indices) for indices in group_indices.values())
        balanced_indices = []
        for indices in group_indices.values():
            sampled = np.random.choice(indices, min_count, replace=False)
            balanced_indices.extend(sampled)

        np.random.shuffle(balanced_indices)
        balanced_dataset = Subset(self.full_dataset, balanced_indices)

        return DataLoader(
            balanced_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    def get_data_shape(self) -> Tuple[int, int, int]:
        """Get the shape of data samples (C, H, W)."""
        return (3, self.image_size, self.image_size)

    def get_group_info_summary(self) -> str:
        """Get a formatted summary of group statistics."""
        info = self.group_info
        summary = f"CelebA Dataset Summary:\n"
        summary += f"  Sensitive Attribute: {self.sensitive_attr}\n"
        summary += f"  Total Training Samples: {info['total']}\n"
        summary += f"  Number of Groups: {info['n_groups']}\n"
        for group, count in info['counts'].items():
            ratio = info['ratios'][group]
            summary += f"    Group {group}: {count} samples ({ratio:.2%})\n"
        return summary
