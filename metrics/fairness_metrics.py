"""
Fairness metrics: SPD, DI, EOD
"""
import torch
import numpy as np


def statistical_parity_difference(predictions, sensitive_attr):
    """
    Statistical Parity Difference (SPD)
    SPD = P(Y=1|S=1) - P(Y=1|S=0)

    Args:
        predictions: Binary predictions (0 or 1)
        sensitive_attr: Sensitive attribute values (0 or 1)

    Returns:
        SPD value (closer to 0 is more fair)
    """
    predictions = predictions.float()
    sensitive_attr = sensitive_attr.float()

    # Group 0
    mask_0 = (sensitive_attr == 0)
    if mask_0.sum() > 0:
        prob_y1_s0 = predictions[mask_0].mean().item()
    else:
        prob_y1_s0 = 0.0

    # Group 1
    mask_1 = (sensitive_attr == 1)
    if mask_1.sum() > 0:
        prob_y1_s1 = predictions[mask_1].mean().item()
    else:
        prob_y1_s1 = 0.0

    spd = abs(prob_y1_s1 - prob_y1_s0)
    return spd


def disparate_impact(predictions, sensitive_attr):
    """
    Disparate Impact (DI)
    DI = P(Y=1|S=0) / P(Y=1|S=1)

    Args:
        predictions: Binary predictions (0 or 1)
        sensitive_attr: Sensitive attribute values (0 or 1)

    Returns:
        DI value (closer to 1 is more fair)
    """
    predictions = predictions.float()
    sensitive_attr = sensitive_attr.float()

    # Group 0
    mask_0 = (sensitive_attr == 0)
    if mask_0.sum() > 0:
        prob_y1_s0 = predictions[mask_0].mean().item()
    else:
        prob_y1_s0 = 1e-8

    # Group 1
    mask_1 = (sensitive_attr == 1)
    if mask_1.sum() > 0:
        prob_y1_s1 = predictions[mask_1].mean().item()
    else:
        prob_y1_s1 = 1e-8

    # Avoid division by zero
    if prob_y1_s1 < 1e-8:
        prob_y1_s1 = 1e-8

    di = prob_y1_s0 / prob_y1_s1

    # Return absolute deviation from 1
    return abs(1.0 - di)


def equalized_odds_difference(predictions, sensitive_attr, true_labels):
    """
    Equalized Odds Difference (EOD)
    EOD = |P(Y_hat=1|Y=1,S=0) - P(Y_hat=1|Y=1,S=1)| +
          |P(Y_hat=1|Y=0,S=0) - P(Y_hat=1|Y=0,S=1)|

    Args:
        predictions: Binary predictions (0 or 1)
        sensitive_attr: Sensitive attribute values (0 or 1)
        true_labels: Ground truth labels (0 or 1)

    Returns:
        EOD value (closer to 0 is more fair)
    """
    predictions = predictions.float()
    sensitive_attr = sensitive_attr.float()
    true_labels = true_labels.float()

    # TPR difference (Y=1)
    mask_y1_s0 = (true_labels == 1) & (sensitive_attr == 0)
    mask_y1_s1 = (true_labels == 1) & (sensitive_attr == 1)

    if mask_y1_s0.sum() > 0:
        tpr_s0 = predictions[mask_y1_s0].mean().item()
    else:
        tpr_s0 = 0.0

    if mask_y1_s1.sum() > 0:
        tpr_s1 = predictions[mask_y1_s1].mean().item()
    else:
        tpr_s1 = 0.0

    tpr_diff = abs(tpr_s0 - tpr_s1)

    # FPR difference (Y=0)
    mask_y0_s0 = (true_labels == 0) & (sensitive_attr == 0)
    mask_y0_s1 = (true_labels == 0) & (sensitive_attr == 1)

    if mask_y0_s0.sum() > 0:
        fpr_s0 = predictions[mask_y0_s0].mean().item()
    else:
        fpr_s0 = 0.0

    if mask_y0_s1.sum() > 0:
        fpr_s1 = predictions[mask_y0_s1].mean().item()
    else:
        fpr_s1 = 0.0

    fpr_diff = abs(fpr_s0 - fpr_s1)

    eod = tpr_diff + fpr_diff
    return eod


def evaluate_fairness(generator, attribute_classifier, config, num_samples=1000, device=None):
    """
    Evaluate fairness of generated samples

    Args:
        generator: Trained generator model
        attribute_classifier: Trained attribute classifier
        config: Configuration object
        num_samples: Number of samples to generate for evaluation
        device: Device to use

    Returns:
        Dictionary with fairness metrics
    """
    if device is None:
        device = config.DEVICE

    generator.eval()
    attribute_classifier.eval()

    # Generate samples
    with torch.no_grad():
        fake_images = generator.generate(num_samples, device=device)

        # Predict attributes
        fake_attrs = attribute_classifier.predict_attributes(fake_images)

    # Get sensitive and target attribute indices
    sensitive_idx = config.SENSITIVE_ATTR
    target_idx = config.TARGET_ATTR

    # Handle attribute names vs indices
    if isinstance(sensitive_idx, str):
        # Need to get index from attribute_classifier
        # For now, assume Male=20, Attractive=2 (standard CelebA indices)
        sensitive_idx = 20 if sensitive_idx == "Male" else 0
    if isinstance(target_idx, str):
        target_idx = 2 if target_idx == "Attractive" else 0

    sensitive_attr = fake_attrs[:, sensitive_idx]
    target_pred = fake_attrs[:, target_idx]

    # Calculate fairness metrics
    spd = statistical_parity_difference(target_pred, sensitive_attr)
    di = disparate_impact(target_pred, sensitive_attr)

    # For EOD, we need true labels (use predictions as proxy)
    eod = equalized_odds_difference(target_pred, sensitive_attr, target_pred)

    results = {
        'spd': spd,
        'di': di,
        'eod': eod,
        'group_0_positive_rate': target_pred[sensitive_attr == 0].mean().item() if (sensitive_attr == 0).sum() > 0 else 0.0,
        'group_1_positive_rate': target_pred[sensitive_attr == 1].mean().item() if (sensitive_attr == 1).sum() > 0 else 0.0,
    }

    return results


def calculate_fairness_loss(fake_images, discriminator, sensitive_attr):
    """
    Calculate fairness loss for FairGAN training
    Encourages the discriminator to not be able to predict sensitive attributes

    Args:
        fake_images: Generated images
        discriminator: Discriminator model (must be FairDiscriminator)
        sensitive_attr: Ground truth sensitive attributes

    Returns:
        Fairness loss
    """
    # Get discriminator predictions with fairness head
    _, sensitive_pred = discriminator(fake_images, return_fairness=True)

    # BCE loss - we want the discriminator to fail at predicting sensitive attributes
    # So we use random labels or adversarial labels
    random_labels = torch.randint_like(sensitive_attr, 0, 2).float()
    fairness_loss = torch.nn.functional.binary_cross_entropy(
        sensitive_pred,
        random_labels
    )

    return fairness_loss
