#!/usr/bin/env python3
"""
Main script to run Fair-DP-GAN experiments

Usage:
    python main.py --data_type image --n_epochs 100
    python main.py --data_type tabular --target_epsilon 5.0
"""

import argparse
from fair_dp_gan.experiments import run_complete_experiment, ExperimentConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Fair-DP-GAN experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data settings
    parser.add_argument(
        '--data_type', type=str, default='image',
        choices=['image', 'tabular'],
        help='Type of data to use'
    )
    parser.add_argument(
        '--data_path', type=str, default='./data/celeba',
        help='Path to dataset'
    )
    parser.add_argument(
        '--sensitive_attr', type=str, default='Male',
        help='Sensitive attribute for group division'
    )
    parser.add_argument(
        '--batch_size', type=int, default=128,
        help='Batch size for training'
    )

    # Model settings
    parser.add_argument(
        '--latent_dim', type=int, default=100,
        help='Dimension of latent noise vector'
    )
    parser.add_argument(
        '--n_groups', type=int, default=2,
        help='Number of demographic groups'
    )
    parser.add_argument(
        '--image_size', type=int, default=64,
        help='Size of images (for image data)'
    )

    # Training settings
    parser.add_argument(
        '--n_epochs', type=int, default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr_g', type=float, default=0.0002,
        help='Learning rate for generator'
    )
    parser.add_argument(
        '--lr_c', type=float, default=0.0002,
        help='Learning rate for critic'
    )

    # Privacy settings
    parser.add_argument(
        '--target_epsilon', type=float, default=10.0,
        help='Target privacy budget (epsilon)'
    )
    parser.add_argument(
        '--target_delta', type=float, default=1e-5,
        help='Target delta for (ε, δ)-DP'
    )
    parser.add_argument(
        '--clipping_norm', type=float, default=1.0,
        help='Gradient clipping threshold'
    )
    parser.add_argument(
        '--noise_multiplier', type=float, default=1.0,
        help='Noise scale multiplier'
    )

    # Fairness settings
    parser.add_argument(
        '--lambda_fair', type=float, default=1.0,
        help='Fairness regularization weight'
    )

    # Experiment settings
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./experiments/results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--run_name', type=str, default='fair_dp_gan_experiment',
        help='Name of this experimental run'
    )
    parser.add_argument(
        '--models', type=str, nargs='+',
        default=['vanilla', 'fair', 'dp', 'uniform_fair_dp', 'adaptive_fair_dp', 'sequential'],
        help='Which models to train'
    )

    # Device
    parser.add_argument(
        '--device', type=str, default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for training'
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Create configuration
    config = ExperimentConfig(
        data_type=args.data_type,
        data_path=args.data_path,
        sensitive_attr=args.sensitive_attr,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        n_groups=args.n_groups,
        image_size=args.image_size,
        n_epochs=args.n_epochs,
        lr_g=args.lr_g,
        lr_c=args.lr_c,
        target_epsilon=args.target_epsilon,
        target_delta=args.target_delta,
        clipping_norm=args.clipping_norm,
        noise_multiplier=args.noise_multiplier,
        lambda_fair=args.lambda_fair,
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir,
        run_name=args.run_name,
        models_to_train=args.models
    )

    # Run experiment
    results = run_complete_experiment(config)

    print("\n" + "="*80)
    print("EXPERIMENT FINISHED SUCCESSFULLY!")
    print("="*80)
    print(f"\nCheck results in: {args.output_dir}")
    print("\nFor paper writing:")
    print(f"  - Quantitative results: {args.output_dir}/experiment_summary.json")
    print(f"  - Pareto frontier plot: {args.output_dir}/visualizations/pareto_frontier.png")
    print(f"  - Training curves: {args.output_dir}/visualizations/training_curves.png")
    print(f"  - Fairness comparison: {args.output_dir}/visualizations/fairness_comparison.png")

    return results


if __name__ == '__main__':
    main()
