"""
Privacy metrics: Epsilon calculation and Membership Inference Attack (MIA)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm


class MembershipInferenceAttacker(nn.Module):
    """
    Membership Inference Attack model
    Tries to determine if a sample was in the training set
    """

    def __init__(self, input_dim=2048):
        """
        Args:
            input_dim: Dimension of input features
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).squeeze()


def extract_discriminator_features(discriminator, images, device):
    """
    Extract features from discriminator for MIA

    Args:
        discriminator: Trained discriminator
        images: Input images
        device: Device to use

    Returns:
        Features tensor
    """
    discriminator.eval()
    features_list = []

    with torch.no_grad():
        # If discriminator has a features attribute, use it
        if hasattr(discriminator, 'features'):
            for i in range(0, len(images), 32):
                batch = images[i:i+32].to(device)
                feats = discriminator.features(batch)
                feats = feats.view(feats.size(0), -1)
                features_list.append(feats.cpu())
        else:
            # Otherwise, use discriminator output as feature
            for i in range(0, len(images), 32):
                batch = images[i:i+32].to(device)
                output = discriminator(batch)
                if output.dim() > 1:
                    output = output.view(output.size(0), -1)
                else:
                    output = output.unsqueeze(1)
                features_list.append(output.cpu())

    features = torch.cat(features_list, dim=0)
    return features


def membership_inference_attack(
    discriminator,
    train_images,
    test_images,
    config,
    device=None,
    epochs=20
):
    """
    Perform Membership Inference Attack

    Args:
        discriminator: Trained discriminator from GAN
        train_images: Images used for training the GAN (members)
        test_images: Images not used for training (non-members)
        config: Configuration object
        device: Device to use
        epochs: Epochs to train the attack model

    Returns:
        Attack accuracy (higher means more privacy leakage)
    """
    if device is None:
        device = config.DEVICE

    print("\n" + "="*60)
    print("Performing Membership Inference Attack")
    print("="*60)

    # Extract features
    print("Extracting features from member samples...")
    member_features = extract_discriminator_features(discriminator, train_images, device)

    print("Extracting features from non-member samples...")
    non_member_features = extract_discriminator_features(discriminator, test_images, device)

    # Create labels (1 for members, 0 for non-members)
    member_labels = torch.ones(len(member_features))
    non_member_labels = torch.zeros(len(non_member_features))

    # Combine and shuffle
    all_features = torch.cat([member_features, non_member_features], dim=0)
    all_labels = torch.cat([member_labels, non_member_labels], dim=0)

    # Shuffle
    indices = torch.randperm(len(all_features))
    all_features = all_features[indices]
    all_labels = all_labels[indices]

    # Split into train and test for attack model
    split_idx = int(0.8 * len(all_features))
    train_features = all_features[:split_idx]
    train_labels = all_labels[:split_idx]
    test_features = all_features[split_idx:]
    test_labels = all_labels[split_idx:]

    # Create dataloaders
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = TensorDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Train attack model
    input_dim = train_features.size(1)
    attacker = MembershipInferenceAttacker(input_dim=input_dim).to(device)
    optimizer = optim.Adam(attacker.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    print(f"\nTraining attack model with {input_dim}-dim features...")
    attacker.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = attacker(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    # Test attack model
    attacker.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = attacker(features)
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    attack_accuracy = 100.0 * correct / total

    print(f"\nMIA Results:")
    print(f"Attack Accuracy: {attack_accuracy:.2f}%")
    print(f"Privacy Score (100 - attack_acc): {100 - attack_accuracy:.2f}%")
    print("="*60 + "\n")

    return attack_accuracy


def calculate_epsilon_from_accountant(accountant, delta=1e-5):
    """
    Calculate epsilon from privacy accountant

    Args:
        accountant: Privacy accountant (e.g., from opacus)
        delta: Target delta

    Returns:
        Epsilon value
    """
    try:
        epsilon = accountant.get_epsilon(delta=delta)
        return epsilon
    except Exception as e:
        print(f"Error calculating epsilon: {e}")
        return -1


def calculate_epsilon_naive(
    noise_multiplier,
    num_steps,
    batch_size,
    dataset_size,
    delta=1e-5
):
    """
    Naive epsilon calculation using basic composition
    This is a very rough approximation

    Args:
        noise_multiplier: Noise multiplier (sigma)
        num_steps: Number of training steps
        batch_size: Batch size
        dataset_size: Total dataset size
        delta: Target delta

    Returns:
        Approximate epsilon
    """
    # Sampling rate
    q = batch_size / dataset_size

    # Basic composition (very loose bound)
    # This is just for demonstration - use proper accounting in practice
    if noise_multiplier == 0:
        return float('inf')

    # Rough approximation: epsilon ≈ sqrt(2 * T * ln(1/delta)) / sigma
    # where T is the number of steps
    epsilon = np.sqrt(2 * num_steps * np.log(1 / delta)) / noise_multiplier

    return epsilon


def evaluate_privacy(
    discriminator,
    train_images,
    test_images,
    config,
    epsilon_theoretical=None,
    device=None
):
    """
    Evaluate privacy of the model

    Args:
        discriminator: Trained discriminator
        train_images: Training images (members)
        test_images: Test images (non-members)
        config: Configuration object
        epsilon_theoretical: Theoretical epsilon (if available)
        device: Device to use

    Returns:
        Dictionary with privacy metrics
    """
    if device is None:
        device = config.DEVICE

    # Perform MIA
    mia_accuracy = membership_inference_attack(
        discriminator,
        train_images,
        test_images,
        config,
        device=device
    )

    results = {
        'epsilon_theoretical': epsilon_theoretical if epsilon_theoretical is not None else config.EPSILON_TARGET,
        'mia_accuracy': mia_accuracy,
        'privacy_score': 100 - mia_accuracy,  # Higher is better
    }

    return results
