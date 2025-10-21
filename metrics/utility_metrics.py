"""
Utility metrics: FID, IS, TSTR (Train on Synthetic, Test on Real)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg
from tqdm import tqdm
from torchvision.models import inception_v3
from torch.utils.data import TensorDataset, DataLoader


class InceptionV3FeatureExtractor(nn.Module):
    """InceptionV3 for feature extraction (for FID and IS)"""

    def __init__(self):
        super().__init__()
        self.inception = inception_v3(pretrained=True, transform_input=False)
        self.inception.fc = nn.Identity()  # Remove final FC layer
        self.inception.eval()

    def forward(self, x):
        """
        Args:
            x: Images in range [-1, 1]
        Returns:
            Features from InceptionV3
        """
        # Resize to 299x299 for InceptionV3
        if x.shape[2:] != (299, 299):
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        # Normalize from [-1, 1] to [0, 1]
        x = (x + 1) / 2

        # InceptionV3 expects specific normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        with torch.no_grad():
            features = self.inception(x)

        return features


def calculate_fid(real_images, fake_images, batch_size=50, device='cuda'):
    """
    Calculate Frechet Inception Distance (FID)

    Args:
        real_images: Real images tensor
        fake_images: Generated images tensor
        batch_size: Batch size for processing
        device: Device to use

    Returns:
        FID score (lower is better)
    """
    # Feature extractor
    feature_extractor = InceptionV3FeatureExtractor().to(device)
    feature_extractor.eval()

    def get_features(images):
        """Extract features from images"""
        features_list = []
        num_batches = (len(images) + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches), desc="Extracting features"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(images))
            batch = images[start_idx:end_idx].to(device)

            with torch.no_grad():
                features = feature_extractor(batch)
            features_list.append(features.cpu())

        return torch.cat(features_list, dim=0)

    # Extract features
    print("Extracting features from real images...")
    real_features = get_features(real_images)

    print("Extracting features from fake images...")
    fake_features = get_features(fake_images)

    # Calculate statistics
    mu_real = real_features.mean(dim=0).numpy()
    sigma_real = np.cov(real_features.numpy(), rowvar=False)

    mu_fake = fake_features.mean(dim=0).numpy()
    sigma_fake = np.cov(fake_features.numpy(), rowvar=False)

    # Calculate FID
    diff = mu_real - mu_fake
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)

    # Handle numerical errors
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)

    return float(fid)


def calculate_inception_score(images, batch_size=50, splits=10, device='cuda'):
    """
    Calculate Inception Score (IS)

    Args:
        images: Generated images tensor
        batch_size: Batch size for processing
        splits: Number of splits for calculating IS
        device: Device to use

    Returns:
        (mean_IS, std_IS) tuple
    """
    # Use full InceptionV3 with classifier
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()

    def get_predictions(images):
        """Get class predictions from InceptionV3"""
        preds_list = []
        num_batches = (len(images) + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches), desc="Computing IS"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(images))
            batch = images[start_idx:end_idx].to(device)

            # Resize to 299x299
            if batch.shape[2:] != (299, 299):
                batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)

            # Normalize
            batch = (batch + 1) / 2
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            batch = (batch - mean) / std

            with torch.no_grad():
                pred = F.softmax(inception(batch), dim=1)
            preds_list.append(pred.cpu())

        return torch.cat(preds_list, dim=0)

    # Get predictions
    preds = get_predictions(images).numpy()

    # Calculate IS
    split_scores = []
    n_samples = preds.shape[0]
    split_size = n_samples // splits

    for i in range(splits):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < splits - 1 else n_samples
        split_preds = preds[start_idx:end_idx]

        # KL divergence
        kl_divs = split_preds * (np.log(split_preds + 1e-10) - np.log(np.mean(split_preds, axis=0, keepdims=True) + 1e-10))
        kl_div = np.mean(np.sum(kl_divs, axis=1))
        split_scores.append(np.exp(kl_div))

    return np.mean(split_scores), np.std(split_scores)


def train_synthetic_test_real(
    synthetic_images,
    synthetic_labels,
    real_test_images,
    real_test_labels,
    config,
    epochs=10,
    device='cuda'
):
    """
    Train on Synthetic, Test on Real (TSTR)
    Trains a classifier on synthetic data and tests on real data

    Args:
        synthetic_images: Synthetic training images
        synthetic_labels: Labels for synthetic images
        real_test_images: Real test images
        real_test_labels: Real test labels
        config: Configuration object
        epochs: Number of training epochs
        device: Device to use

    Returns:
        Test accuracy on real data
    """
    from ..models.attribute_classifier import AttributeClassifier

    # Create a simple classifier
    classifier = AttributeClassifier(config, num_attributes=1).to(device)

    # Create dataloaders
    synthetic_dataset = TensorDataset(synthetic_images, synthetic_labels)
    synthetic_loader = DataLoader(synthetic_dataset, batch_size=32, shuffle=True)

    real_test_dataset = TensorDataset(real_test_images, real_test_labels)
    real_test_loader = DataLoader(real_test_dataset, batch_size=32, shuffle=False)

    # Train
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    classifier.train()
    for epoch in range(epochs):
        for images, labels in synthetic_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Test on real data
    classifier.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in real_test_loader:
            images = images.to(device)
            labels = labels.to(device).float()

            outputs = classifier(images).squeeze()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    return accuracy


def evaluate_utility(
    generator,
    real_images,
    config,
    num_samples=1000,
    device=None
):
    """
    Evaluate utility metrics for generated images

    Args:
        generator: Trained generator
        real_images: Real images for comparison
        config: Configuration object
        num_samples: Number of samples to generate
        device: Device to use

    Returns:
        Dictionary with utility metrics
    """
    if device is None:
        device = config.DEVICE

    generator.eval()

    # Generate samples
    print(f"Generating {num_samples} samples for evaluation...")
    with torch.no_grad():
        fake_images = generator.generate(num_samples, device=device)

    # Ensure real_images has the same number of samples
    if len(real_images) > num_samples:
        indices = torch.randperm(len(real_images))[:num_samples]
        real_images = real_images[indices]

    print("\nCalculating FID...")
    try:
        fid_score = calculate_fid(real_images, fake_images, device=device)
    except Exception as e:
        print(f"Error calculating FID: {e}")
        fid_score = -1

    print("\nCalculating Inception Score...")
    try:
        is_mean, is_std = calculate_inception_score(fake_images, device=device)
    except Exception as e:
        print(f"Error calculating IS: {e}")
        is_mean, is_std = -1, -1

    results = {
        'fid': fid_score,
        'is_mean': is_mean,
        'is_std': is_std,
    }

    print(f"\nUtility Metrics:")
    print(f"FID: {fid_score:.2f}")
    print(f"IS: {is_mean:.2f} ± {is_std:.2f}")

    return results
