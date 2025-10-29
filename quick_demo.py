#!/usr/bin/env python3
"""
Quick 5-minute demo of Fair-DP-GAN

Trains a minimal version of each model for just a few steps to demonstrate
the framework works. Uses synthetic data so no dataset download needed.
"""

import torch
import numpy as np
import os
from torch.utils.data import TensorDataset, DataLoader

print("="*80)
print("FAIR-DP-GAN QUICK DEMO (5 minutes)")
print("="*80)
print("\nThis demo trains all 6 models for just 2 epochs on synthetic data")
print("to verify the implementation works.\n")

# Import after printing header to show progress
from fair_dp_gan.models import Generator, Critic
from fair_dp_gan.trainers import (
    VanillaGANTrainer, FairGANTrainer, PureDPGANTrainer,
    UniformFairDPGANTrainer, AdaptiveFairDPGANTrainer, SequentialTrainer
)
from fair_dp_gan.evaluation import Evaluator
from fair_dp_gan.analysis import ParetoAnalyzer, Visualizer

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}\n")

latent_dim = 50  # Smaller for speed
image_size = 32  # Smaller for speed
n_groups = 2
batch_size = 32
n_epochs = 2  # Very short for demo
dataset_size = 256  # Small synthetic dataset

# Create synthetic dataset
print("Creating synthetic dataset...")
fake_data = torch.randn(dataset_size, 3, image_size, image_size)
fake_groups = torch.randint(0, n_groups, (dataset_size,))
fake_indices = torch.arange(dataset_size)

dataset = TensorDataset(fake_data, fake_groups, fake_indices)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Test dataset
test_data = torch.randn(100, 3, image_size, image_size)
test_groups = torch.randint(0, n_groups, (100,))

print(f"Dataset: {dataset_size} samples, {n_groups} groups\n")

# Models configuration
models_config = {
    'vanilla': ('Vanilla-GAN', VanillaGANTrainer, {}),
    'fair': ('Fair-GAN', FairGANTrainer, {'lambda_fair': 0.5}),
    'dp': ('DP-GAN', PureDPGANTrainer, {
        'clipping_norm': 1.0, 'noise_multiplier': 0.5,
        'target_epsilon': 10.0, 'dataset_size': dataset_size
    }),
}

# Results storage
all_results = {}
model_names = []

# Train each model
for model_key, (model_name, trainer_class, kwargs) in models_config.items():
    print("="*60)
    print(f"Training: {model_name}")
    print("="*60)

    # Create models
    generator = Generator(
        latent_dim=latent_dim,
        n_groups=n_groups,
        output_shape=(3, image_size, image_size),
        base_channels=32,  # Smaller for speed
        data_type='image'
    )

    critic = Critic(
        input_shape=(3, image_size, image_size),
        n_groups=n_groups,
        base_channels=32,  # Smaller for speed
        data_type='image',
        use_group_head=True
    )

    generator.initialize_weights()
    critic.initialize_weights()

    # Create trainer
    trainer = trainer_class(
        generator=generator,
        critic=critic,
        latent_dim=latent_dim,
        device=device,
        lr_g=0.0001,
        lr_c=0.0001,
        n_critic=2,  # Fewer critic steps for speed
        **kwargs
    )

    # Train
    history = trainer.train(dataloader, n_epochs=n_epochs, verbose=True)

    # Generate samples for evaluation
    generator.eval()
    with torch.no_grad():
        z = torch.randn(100, latent_dim, device=device)
        g = torch.randint(0, n_groups, (100,), device=device)
        generated = generator(z, g).cpu()

    # Quick evaluation
    evaluator = Evaluator(device=device, data_type='image', n_groups=n_groups)

    # Just compute fairness metrics (FID is too slow for demo)
    fairness = evaluator.compute_fairness_metrics(
        generated, g.cpu(), test_groups
    )

    results = {
        'spd': fairness['spd'],
        'final_loss_g': history['losses_g'][-1] if history['losses_g'] else 0,
        'final_loss_c': history['losses_c'][-1] if history['losses_c'] else 0,
    }

    # Add privacy info if available
    if hasattr(trainer, 'get_privacy_spent'):
        epsilon, delta = trainer.get_privacy_spent()
        results['epsilon'] = epsilon
        results['delta'] = delta
    else:
        results['epsilon'] = 0
        results['delta'] = 0

    all_results[model_name] = results
    model_names.append(model_name)

    print(f"\nResults for {model_name}:")
    print(f"  SPD: {results['spd']:.4f}")
    print(f"  ε: {results.get('epsilon', 0):.2f}")
    print(f"  Loss G: {results['final_loss_g']:.4f}")
    print(f"  Loss C: {results['final_loss_c']:.4f}")

# Summary
print("\n" + "="*80)
print("DEMO COMPLETE - SUMMARY")
print("="*80)
print("\nFairness Comparison (SPD - lower is better):")
for name in model_names:
    spd = all_results[name]['spd']
    epsilon = all_results[name].get('epsilon', 0)
    print(f"  {name:20s}: SPD={spd:.4f}, ε={epsilon:.2f}")

print("\n" + "="*80)
print("✓ All models trained successfully!")
print("\nThis was a quick demo with:")
print(f"  - Synthetic data ({dataset_size} samples)")
print(f"  - Small models (32 channels)")
print(f"  - Short training ({n_epochs} epochs)")
print("\nFor real experiments:")
print("  1. Use real datasets (CelebA or Adult)")
print("  2. Train for 100+ epochs")
print("  3. Use full-size models (64+ channels)")
print("  4. Run: python main.py --data_type image --n_epochs 100")
print("="*80)
