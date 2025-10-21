"""
CelebA dataset loader with attribute labels for fairness evaluation
"""
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import os


class CelebADataset:
    """CelebA dataset loader with fairness-aware sampling"""

    def __init__(self, config, root_dir=None):
        """
        Args:
            config: Configuration object
            root_dir: Root directory for CelebA dataset
        """
        self.config = config
        self.root_dir = root_dir or config.DATA_DIR

        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.dataset = None
        self.attr_names = None

    def load_dataset(self, split='train', download=True):
        """
        Load CelebA dataset

        Args:
            split: 'train', 'valid', or 'test'
            download: Whether to download the dataset if not available
        """
        try:
            self.dataset = datasets.CelebA(
                root=self.root_dir,
                split=split,
                target_type='attr',
                transform=self.transform,
                download=download
            )
            self.attr_names = self.dataset.attr_names
            print(f"Loaded CelebA {split} set: {len(self.dataset)} images")
            print(f"Attributes: {self.attr_names}")

        except Exception as e:
            print(f"Error loading CelebA: {e}")
            print("Creating synthetic dataset for demonstration...")
            self._create_synthetic_dataset()

        return self.dataset

    def _create_synthetic_dataset(self):
        """Create a synthetic dataset for testing when CelebA is not available"""
        class SyntheticCelebA(Dataset):
            def __init__(self, num_samples, image_size, num_channels):
                self.num_samples = num_samples
                self.image_size = image_size
                self.num_channels = num_channels
                # CelebA has 40 attributes
                self.attr_names = [
                    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
                    'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
                    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
                    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
                    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
                    'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
                    'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
                ]
                # Generate random attributes
                np.random.seed(42)
                self.attributes = torch.from_numpy(
                    np.random.binomial(1, 0.5, (num_samples, 40))
                ).float()

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                # Generate random image
                image = torch.randn(self.num_channels, self.image_size, self.image_size)
                attrs = self.attributes[idx]
                return image, attrs

        self.dataset = SyntheticCelebA(
            self.config.NUM_SAMPLES,
            self.config.IMAGE_SIZE,
            self.config.NUM_CHANNELS
        )
        self.attr_names = self.dataset.attr_names

    def get_subset(self, num_samples=None):
        """Get a subset of the dataset"""
        if num_samples is None:
            num_samples = self.config.NUM_SAMPLES

        num_samples = min(num_samples, len(self.dataset))
        indices = np.random.choice(len(self.dataset), num_samples, replace=False)
        return Subset(self.dataset, indices)

    def create_dataloader(self, subset=None, batch_size=None, shuffle=True, num_workers=4):
        """Create DataLoader"""
        if batch_size is None:
            batch_size = self.config.BATCH_SIZE

        dataset = subset if subset is not None else self.dataset

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )

    def get_sensitive_attribute_index(self, attr_name=None):
        """Get the index of the sensitive attribute"""
        if attr_name is None:
            attr_name = self.config.SENSITIVE_ATTR
        return self.attr_names.index(attr_name)

    def get_target_attribute_index(self, attr_name=None):
        """Get the index of the target attribute"""
        if attr_name is None:
            attr_name = self.config.TARGET_ATTR
        return self.attr_names.index(attr_name)

    def get_group_statistics(self, subset=None):
        """
        Calculate group statistics for fairness evaluation

        Returns:
            dict: Statistics for each group defined by sensitive attribute
        """
        dataset = subset if subset is not None else self.dataset
        sensitive_idx = self.get_sensitive_attribute_index()
        target_idx = self.get_target_attribute_index()

        stats = {
            'group_0': {'count': 0, 'positive': 0},
            'group_1': {'count': 0, 'positive': 0}
        }

        for i in range(len(dataset)):
            _, attrs = dataset[i]
            sensitive = int(attrs[sensitive_idx].item())
            target = int(attrs[target_idx].item())

            group_key = f'group_{sensitive}'
            stats[group_key]['count'] += 1
            if target == 1:
                stats[group_key]['positive'] += 1

        # Calculate positive rates
        for group_key in stats:
            if stats[group_key]['count'] > 0:
                stats[group_key]['positive_rate'] = (
                    stats[group_key]['positive'] / stats[group_key]['count']
                )
            else:
                stats[group_key]['positive_rate'] = 0.0

        return stats


def get_celeba_dataloaders(config, num_samples=None):
    """
    Convenience function to get train and test dataloaders

    Args:
        config: Configuration object
        num_samples: Number of samples to use (None for all)

    Returns:
        train_loader, test_loader, dataset_info
    """
    # Load training data
    celeba_train = CelebADataset(config)
    celeba_train.load_dataset(split='train', download=True)
    train_subset = celeba_train.get_subset(num_samples)
    train_loader = celeba_train.create_dataloader(train_subset, shuffle=True)

    # Load test data
    celeba_test = CelebADataset(config)
    celeba_test.load_dataset(split='test', download=True)
    test_subset = celeba_test.get_subset(num_samples // 10 if num_samples else 1000)
    test_loader = celeba_test.create_dataloader(test_subset, shuffle=False)

    # Get dataset statistics
    train_stats = celeba_train.get_group_statistics(train_subset)
    test_stats = celeba_test.get_group_statistics(test_subset)

    dataset_info = {
        'attr_names': celeba_train.attr_names,
        'sensitive_attr': config.SENSITIVE_ATTR,
        'target_attr': config.TARGET_ATTR,
        'train_stats': train_stats,
        'test_stats': test_stats,
        'train_size': len(train_subset),
        'test_size': len(test_subset)
    }

    print(f"\nDataset Info:")
    print(f"Training set: {dataset_info['train_size']} samples")
    print(f"Test set: {dataset_info['test_size']} samples")
    print(f"Sensitive attribute: {config.SENSITIVE_ATTR}")
    print(f"Target attribute: {config.TARGET_ATTR}")
    print(f"Train group statistics: {train_stats}")

    return train_loader, test_loader, dataset_info
