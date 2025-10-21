"""
Attribute classifier for generating downstream labels
This is a pre-trained classifier used to predict attributes from generated images
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class AttributeClassifier(nn.Module):
    """
    Multi-label attribute classifier for CelebA
    Predicts multiple binary attributes from face images
    """

    def __init__(self, config, num_attributes=40):
        """
        Args:
            config: Configuration object
            num_attributes: Number of binary attributes to predict
        """
        super(AttributeClassifier, self).__init__()
        self.config = config
        self.num_attributes = num_attributes

        # Feature extractor (similar to discriminator)
        self.features = nn.Sequential(
            nn.Conv2d(config.NUM_CHANNELS, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Classifier head
        feature_size = config.IMAGE_SIZE // 16
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * feature_size * feature_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_attributes),
            nn.Sigmoid()  # Multi-label classification
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input images of shape (batch_size, num_channels, image_size, image_size)

        Returns:
            Attribute predictions of shape (batch_size, num_attributes)
            Each value is in [0, 1] representing probability of attribute being present
        """
        features = self.features(x)
        predictions = self.classifier(features)
        return predictions

    def predict_attributes(self, images, threshold=0.5):
        """
        Predict binary attributes from images

        Args:
            images: Input images
            threshold: Threshold for binary classification

        Returns:
            Binary predictions (0 or 1)
        """
        with torch.no_grad():
            probs = self.forward(images)
            predictions = (probs > threshold).float()
        return predictions


def train_attribute_classifier(classifier, train_loader, config, epochs=10, device=None):
    """
    Train the attribute classifier on real data
    IMPORTANT: This classifier is trained on REAL downstream labels, not sensitive attributes

    Args:
        classifier: AttributeClassifier model
        train_loader: DataLoader for training data
        config: Configuration object
        epochs: Number of training epochs
        device: Device to train on

    Returns:
        Trained classifier
    """
    if device is None:
        device = config.DEVICE

    classifier = classifier.to(device)
    classifier.train()

    # Binary cross-entropy loss for multi-label classification
    criterion = nn.BCELoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))

    print(f"\n{'='*60}")
    print(f"Training Attribute Classifier")
    print(f"{'='*60}")

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, attributes in pbar:
            images = images.to(device)
            attributes = attributes.to(device)

            # Convert attributes from {-1, 1} to {0, 1} if necessary
            if attributes.min() < 0:
                attributes = (attributes + 1) / 2

            # Forward pass
            optimizer.zero_grad()
            predictions = classifier(images)

            # Compute loss
            loss = criterion(predictions, attributes)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            total_loss += loss.item()
            predicted_binary = (predictions > 0.5).float()
            correct += (predicted_binary == attributes).sum().item()
            total += attributes.numel()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    print(f"{'='*60}")
    print(f"Attribute Classifier Training Complete")
    print(f"{'='*60}\n")

    classifier.eval()
    return classifier


def evaluate_attribute_classifier(classifier, test_loader, config, device=None):
    """
    Evaluate the attribute classifier

    Args:
        classifier: Trained AttributeClassifier model
        test_loader: DataLoader for test data
        config: Configuration object
        device: Device to evaluate on

    Returns:
        Dictionary with evaluation metrics
    """
    if device is None:
        device = config.DEVICE

    classifier = classifier.to(device)
    classifier.eval()

    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, attributes in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            attributes = attributes.to(device)

            # Convert attributes from {-1, 1} to {0, 1} if necessary
            if attributes.min() < 0:
                attributes = (attributes + 1) / 2

            # Predict
            predictions = classifier(images)
            predicted_binary = (predictions > 0.5).float()

            # Collect results
            all_predictions.append(predicted_binary.cpu())
            all_labels.append(attributes.cpu())

            # Accuracy
            correct += (predicted_binary == attributes).sum().item()
            total += attributes.numel()

    accuracy = 100.0 * correct / total

    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Per-attribute accuracy
    per_attr_acc = []
    for i in range(all_labels.size(1)):
        attr_correct = (all_predictions[:, i] == all_labels[:, i]).sum().item()
        attr_total = all_labels.size(0)
        per_attr_acc.append(100.0 * attr_correct / attr_total)

    results = {
        'overall_accuracy': accuracy,
        'per_attribute_accuracy': per_attr_acc,
        'mean_per_attribute_accuracy': sum(per_attr_acc) / len(per_attr_acc)
    }

    print(f"\nAttribute Classifier Evaluation:")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Mean Per-Attribute Accuracy: {results['mean_per_attribute_accuracy']:.2f}%")

    return results
