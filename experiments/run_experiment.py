"""
Main experiment script to train and evaluate all 6 models
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
from datetime import datetime

from utils.config import Config
from data.celeba_loader import get_celeba_dataloaders
from models import Generator, Discriminator, FairDiscriminator
from models import AttributeClassifier, train_attribute_classifier, evaluate_attribute_classifier
from trainers import (
    VanillaGANTrainer,
    FairGANTrainer,
    DPGANTrainer,
    UniformFairDPGANTrainer,
    AdaptiveFairDPGANTrainer,
    SequentialTrainer
)
from metrics import evaluate_fairness, evaluate_utility, evaluate_privacy


def collect_real_images(data_loader, num_samples=10000):
    """Collect real images from data loader"""
    images = []
    for batch_images, _ in data_loader:
        images.append(batch_images)
        if len(torch.cat(images, dim=0)) >= num_samples:
            break
    all_images = torch.cat(images, dim=0)[:num_samples]
    return all_images


def evaluate_model(
    model_name,
    generator,
    discriminator,
    attribute_classifier,
    real_train_images,
    real_test_images,
    config
):
    """
    Evaluate a single model on all metrics

    Args:
        model_name: Name of the model
        generator: Trained generator
        discriminator: Trained discriminator
        attribute_classifier: Pre-trained attribute classifier
        real_train_images: Real training images (for MIA)
        real_test_images: Real test images (for MIA)
        config: Configuration

    Returns:
        Dictionary with all evaluation metrics
    """
    print(f"\n{'='*70}")
    print(f"Evaluating {model_name}")
    print(f"{'='*70}\n")

    results = {'model_name': model_name}

    # 1. Fairness metrics
    print(f"\n[{model_name}] Evaluating Fairness...")
    fairness_results = evaluate_fairness(
        generator,
        attribute_classifier,
        config,
        num_samples=config.NUM_GENERATED_SAMPLES,
        device=config.DEVICE
    )
    results['fairness'] = fairness_results
    print(f"SPD: {fairness_results['spd']:.4f}")
    print(f"DI: {fairness_results['di']:.4f}")
    print(f"EOD: {fairness_results['eod']:.4f}")

    # 2. Utility metrics
    print(f"\n[{model_name}] Evaluating Utility...")
    utility_results = evaluate_utility(
        generator,
        real_test_images,
        config,
        num_samples=min(1000, len(real_test_images)),
        device=config.DEVICE
    )
    results['utility'] = utility_results
    print(f"FID: {utility_results['fid']:.2f}")
    print(f"IS: {utility_results['is_mean']:.2f} ± {utility_results['is_std']:.2f}")

    # 3. Privacy metrics
    print(f"\n[{model_name}] Evaluating Privacy...")
    privacy_results = evaluate_privacy(
        discriminator,
        real_train_images,
        real_test_images,
        config,
        epsilon_theoretical=config.EPSILON_TARGET,
        device=config.DEVICE
    )
    results['privacy'] = privacy_results
    print(f"MIA Accuracy: {privacy_results['mia_accuracy']:.2f}%")
    print(f"Privacy Score: {privacy_results['privacy_score']:.2f}%")

    print(f"\n{'='*70}")
    print(f"{model_name} Evaluation Complete")
    print(f"{'='*70}\n")

    return results


def train_and_evaluate_all_models(config):
    """
    Train and evaluate all 6 models

    Args:
        config: Configuration object

    Returns:
        Dictionary with all results
    """
    print("\n" + "="*70)
    print("FAIR-DP-GAN: Complete Experimental Pipeline")
    print("="*70)
    print(f"Dataset: {config.DATASET}")
    print(f"Samples: {config.NUM_SAMPLES}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Target Epsilon: {config.EPSILON_TARGET}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Device: {config.DEVICE}")
    print("="*70 + "\n")

    # =============================
    # 1. Load Data
    # =============================
    print("\n[STEP 1] Loading CelebA Dataset...")
    train_loader, test_loader, dataset_info = get_celeba_dataloaders(
        config,
        num_samples=config.NUM_SAMPLES
    )

    # Collect real images for evaluation
    print("\n[STEP 2] Collecting Real Images for Evaluation...")
    real_train_images = collect_real_images(train_loader, config.NUM_SAMPLES)
    real_test_images = collect_real_images(test_loader, config.NUM_SAMPLES // 10)

    # =============================
    # 2. Train Attribute Classifier
    # =============================
    print("\n[STEP 3] Training Attribute Classifier...")
    attribute_classifier = AttributeClassifier(config, num_attributes=40)
    attribute_classifier = train_attribute_classifier(
        attribute_classifier,
        train_loader,
        config,
        epochs=10,
        device=config.DEVICE
    )

    # Evaluate classifier
    print("\nEvaluating Attribute Classifier...")
    evaluate_attribute_classifier(attribute_classifier, test_loader, config)

    # Save classifier
    classifier_path = os.path.join(config.CHECKPOINT_DIR, 'attribute_classifier')
    os.makedirs(classifier_path, exist_ok=True)
    torch.save(attribute_classifier.state_dict(), os.path.join(classifier_path, 'classifier.pth'))
    print(f"Attribute classifier saved to {classifier_path}")

    # =============================
    # 3. Train All Models
    # =============================
    all_results = {}

    models_to_train = [
        ('Vanilla-GAN', 'vanilla', False),
        ('FairGAN', 'fair', True),
        ('DP-GAN', 'dp', False),
        ('Uniform-FAIR-DP-GAN', 'uniform_fair_dp', True),
        ('Adaptive-FAIR-DP-GAN', 'adaptive_fair_dp', True),
        ('Sequential-Approach', 'sequential', True)
    ]

    for model_name, model_type, use_fair_disc in models_to_train:
        print(f"\n{'='*70}")
        print(f"[STEP 4.{models_to_train.index((model_name, model_type, use_fair_disc)) + 1}] Training {model_name}")
        print(f"{'='*70}\n")

        # Create models
        generator = Generator(config)
        if use_fair_disc:
            discriminator = FairDiscriminator(config)
        else:
            discriminator = Discriminator(config)

        # Select trainer
        if model_type == 'vanilla':
            trainer = VanillaGANTrainer(generator, discriminator, config)
            trainer.train(train_loader)

        elif model_type == 'fair':
            trainer = FairGANTrainer(generator, discriminator, config)
            trainer.train(train_loader)

        elif model_type == 'dp':
            trainer = DPGANTrainer(generator, discriminator, config)
            trainer.train(train_loader)

        elif model_type == 'uniform_fair_dp':
            trainer = UniformFairDPGANTrainer(generator, discriminator, config)
            trainer.train(train_loader)

        elif model_type == 'adaptive_fair_dp':
            trainer = AdaptiveFairDPGANTrainer(generator, discriminator, config)
            trainer.train(train_loader)

        elif model_type == 'sequential':
            trainer = SequentialTrainer(generator, discriminator, config)
            trainer.train(train_loader)

        # Save models
        model_save_path = os.path.join(config.CHECKPOINT_DIR, model_type)
        trainer.save_models(model_save_path)

        # Evaluate
        results = evaluate_model(
            model_name,
            generator,
            discriminator,
            attribute_classifier,
            real_train_images,
            real_test_images,
            config
        )

        all_results[model_name] = results

    # =============================
    # 4. Save Results
    # =============================
    print(f"\n{'='*70}")
    print("Saving Results")
    print(f"{'='*70}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(config.OUTPUT_DIR, f'results_{timestamp}.json')
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to: {results_path}")

    # =============================
    # 5. Print Summary
    # =============================
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}\n")

    print(f"{'Model':<30} {'FID':<10} {'SPD':<10} {'MIA Acc':<10} {'Privacy':<10}")
    print("-" * 70)

    for model_name, results in all_results.items():
        fid = results['utility']['fid']
        spd = results['fairness']['spd']
        mia = results['privacy']['mia_accuracy']
        privacy = results['privacy']['privacy_score']

        print(f"{model_name:<30} {fid:<10.2f} {spd:<10.4f} {mia:<10.2f} {privacy:<10.2f}")

    print(f"\n{'='*70}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*70}\n")

    return all_results


def main():
    """Main function"""
    # Create configuration
    config = Config()

    # Create directories
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # Run experiments
    results = train_and_evaluate_all_models(config)

    return results


if __name__ == "__main__":
    main()
