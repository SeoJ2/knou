"""
Complete Experiment Orchestration

Runs all Fair-DP-GAN experiments and generates paper-ready results.

This script:
1. Trains all 6 model variants
2. Evaluates each on utility, fairness, privacy
3. Performs Pareto analysis
4. Generates visualizations
5. Saves all results for paper writing
"""

import torch
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import json
import os
from tqdm import tqdm

from fair_dp_gan.data import CelebADataLoader, AdultDataLoader
from fair_dp_gan.models import Generator, Critic
from fair_dp_gan.trainers import (
    VanillaGANTrainer,
    FairGANTrainer,
    PureDPGANTrainer,
    UniformFairDPGANTrainer,
    AdaptiveFairDPGANTrainer,
    SequentialTrainer
)
from fair_dp_gan.evaluation import Evaluator, MIAEvaluator
from fair_dp_gan.analysis import ParetoAnalyzer, Visualizer
from fair_dp_gan.utils import CheckpointManager


@dataclass
class ExperimentConfig:
    """
    Configuration for Fair-DP-GAN experiments.

    All hyperparameters and settings for reproducible experiments.
    """
    # Data settings
    data_type: str = 'image'  # 'image' or 'tabular'
    data_path: str = './data/celeba'
    sensitive_attr: str = 'Male'
    batch_size: int = 128
    test_split: float = 0.2

    # Model architecture
    latent_dim: int = 100
    n_groups: int = 2
    image_size: int = 64
    base_channels: int = 64

    # Training
    n_epochs: int = 100
    lr_g: float = 0.0002
    lr_c: float = 0.0002
    n_critic: int = 5

    # Privacy
    target_epsilon: float = 10.0
    target_delta: float = 1e-5
    clipping_norm: float = 1.0
    noise_multiplier: float = 1.0

    # Fairness
    lambda_fair: float = 1.0
    lambda_gp: float = 10.0

    # Sequential trainer
    switch_epoch: int = 50

    # Experiment settings
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42
    output_dir: str = './experiments/results'
    run_name: str = 'fair_dp_gan_experiment'

    # Which models to train
    models_to_train: List[str] = None

    def __post_init__(self):
        if self.models_to_train is None:
            self.models_to_train = [
                'vanilla',
                'fair',
                'dp',
                'uniform_fair_dp',
                'adaptive_fair_dp',
                'sequential'
            ]


def run_complete_experiment(config: ExperimentConfig) -> Dict:
    """
    Run complete Fair-DP-GAN experiment.

    This function orchestrates all experiments, answering the 4 research questions:
    - RQ1: Quantify the trilemma (Vanilla vs Fair vs DP)
    - RQ2: Adaptive vs Uniform Fair-DP-GAN
    - RQ3: Simultaneous vs Sequential optimization
    - RQ4: TSTR practical utility

    Args:
        config: Experiment configuration

    Returns:
        Dict containing all results
    """
    print("="*80)
    print("FAIR-DP-GAN: COMPREHENSIVE EXPERIMENT")
    print("="*80)
    print(f"\nConfiguration:")
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")
    print()

    # Set random seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)

    # Initialize managers
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=os.path.join(config.output_dir, 'checkpoints')
    )
    visualizer = Visualizer(
        output_dir=os.path.join(config.output_dir, 'visualizations')
    )
    pareto_analyzer = ParetoAnalyzer()

    # Load data
    print("\n" + "="*80)
    print("STEP 1: Loading Data")
    print("="*80)

    if config.data_type == 'image':
        dataloader_class = CelebADataLoader
        output_shape = (3, config.image_size, config.image_size)
    else:
        dataloader_class = AdultDataLoader
        output_shape = (100,)  # Will be updated after loading

    data_loader = dataloader_class(
        root_dir=config.data_path,
        sensitive_attr=config.sensitive_attr,
        batch_size=config.batch_size,
        test_split=config.test_split
    )

    print(data_loader.get_group_info_summary())

    if config.data_type == 'tabular':
        output_shape = (data_loader.get_feature_dim(),)

    train_loader = data_loader.get_train_loader()
    test_loader = data_loader.get_test_loader()

    # Initialize evaluator
    evaluator = Evaluator(
        device=config.device,
        data_type=config.data_type,
        n_groups=config.n_groups
    )

    # Storage for results
    all_results = {}
    all_histories = {}
    model_names = []

    # Train each model
    print("\n" + "="*80)
    print("STEP 2: Training Models")
    print("="*80)

    models_config = {
        'vanilla': {
            'name': 'Vanilla-GAN',
            'trainer_class': VanillaGANTrainer,
            'kwargs': {
                'latent_dim': config.latent_dim,
                'lambda_gp': config.lambda_gp
            }
        },
        'fair': {
            'name': 'Fair-GAN',
            'trainer_class': FairGANTrainer,
            'kwargs': {
                'latent_dim': config.latent_dim,
                'lambda_gp': config.lambda_gp,
                'lambda_fair': config.lambda_fair
            }
        },
        'dp': {
            'name': 'DP-GAN',
            'trainer_class': PureDPGANTrainer,
            'kwargs': {
                'latent_dim': config.latent_dim,
                'clipping_norm': config.clipping_norm,
                'noise_multiplier': config.noise_multiplier,
                'target_epsilon': config.target_epsilon,
                'target_delta': config.target_delta,
                'lambda_gp': config.lambda_gp,
                'dataset_size': data_loader.group_info['total']
            }
        },
        'uniform_fair_dp': {
            'name': 'Uniform-FAIR-DP-GAN',
            'trainer_class': UniformFairDPGANTrainer,
            'kwargs': {
                'latent_dim': config.latent_dim,
                'clipping_norm': config.clipping_norm,
                'noise_multiplier': config.noise_multiplier,
                'target_epsilon': config.target_epsilon,
                'target_delta': config.target_delta,
                'lambda_gp': config.lambda_gp,
                'lambda_fair': config.lambda_fair,
                'dataset_size': data_loader.group_info['total']
            }
        },
        'adaptive_fair_dp': {
            'name': 'Adaptive-FAIR-DP-GAN',
            'trainer_class': AdaptiveFairDPGANTrainer,
            'kwargs': {
                'latent_dim': config.latent_dim,
                'base_clipping_norm': config.clipping_norm,
                'noise_multiplier': config.noise_multiplier,
                'target_epsilon': config.target_epsilon,
                'target_delta': config.target_delta,
                'lambda_gp': config.lambda_gp,
                'lambda_fair': config.lambda_fair,
                'dataset_size': data_loader.group_info['total']
            }
        },
        'sequential': {
            'name': 'Sequential-Approach',
            'trainer_class': SequentialTrainer,
            'kwargs': {
                'latent_dim': config.latent_dim,
                'switch_epoch': config.switch_epoch,
                'clipping_norm': config.clipping_norm,
                'noise_multiplier': config.noise_multiplier,
                'target_epsilon': config.target_epsilon,
                'target_delta': config.target_delta,
                'lambda_gp': config.lambda_gp,
                'lambda_fair': config.lambda_fair,
                'dataset_size': data_loader.group_info['total']
            }
        }
    }

    for model_key, model_config in models_config.items():
        if model_key not in config.models_to_train:
            continue

        model_name = model_config['name']
        model_names.append(model_name)

        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}")

        # Create models
        generator = Generator(
            latent_dim=config.latent_dim,
            n_groups=config.n_groups,
            output_shape=output_shape,
            base_channels=config.base_channels,
            data_type=config.data_type
        )

        critic = Critic(
            input_shape=output_shape,
            n_groups=config.n_groups,
            base_channels=config.base_channels,
            data_type=config.data_type,
            use_group_head=True
        )

        generator.initialize_weights()
        critic.initialize_weights()

        # Create trainer
        trainer = model_config['trainer_class'](
            generator=generator,
            critic=critic,
            device=config.device,
            lr_g=config.lr_g,
            lr_c=config.lr_c,
            n_critic=config.n_critic,
            **model_config['kwargs']
        )

        # Set adaptive weights if applicable
        if model_key == 'adaptive_fair_dp':
            trainer.set_group_weights_from_data(train_loader)

        # Train
        history = trainer.train(
            dataloader=train_loader,
            n_epochs=config.n_epochs,
            verbose=True
        )

        all_histories[model_name] = history

        # Evaluate
        print(f"\nEvaluating {model_name}...")
        results = evaluate_model(
            generator, critic, trainer,
            data_loader, evaluator,
            config
        )

        all_results[model_name] = results

        # Save checkpoint
        checkpoint_manager.save_checkpoint(
            model_name=model_key,
            generator=generator,
            critic=critic,
            optimizer_g=trainer.optimizer_g,
            optimizer_c=trainer.optimizer_c,
            epoch=config.n_epochs,
            metrics=results,
            config=asdict(config)
        )

        print(f"\nResults for {model_name}:")
        print(evaluator.get_summary(results))

    # Analyze results
    print("\n" + "="*80)
    print("STEP 3: Pareto Analysis")
    print("="*80)

    results_list = list(all_results.values())
    pareto_indices, pareto_results = pareto_analyzer.find_pareto_frontier(
        results_list, model_names
    )

    print(pareto_analyzer.get_summary(results_list, model_names))

    # Generate visualizations
    print("\n" + "="*80)
    print("STEP 4: Generating Visualizations")
    print("="*80)

    visualizer.plot_pareto_frontier(
        results_list, model_names, pareto_indices
    )

    visualizer.plot_training_curves(all_histories)

    visualizer.plot_fairness_comparison(results_list, model_names)

    visualizer.plot_metrics_heatmap(results_list, model_names)

    # Save summary
    print("\n" + "="*80)
    print("STEP 5: Saving Results")
    print("="*80)

    summary = {
        'config': asdict(config),
        'results': {name: results for name, results in zip(model_names, results_list)},
        'pareto_optimal': [model_names[i] for i in pareto_indices],
        'hypervolume': pareto_analyzer.compute_hypervolume(results_list),
        'rankings': pareto_analyzer.rank_solutions(results_list)
    }

    summary_path = os.path.join(config.output_dir, 'experiment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Saved complete summary to {summary_path}")

    # Print research question answers
    print("\n" + "="*80)
    print("RESEARCH QUESTIONS SUMMARY")
    print("="*80)

    print_research_question_answers(all_results, pareto_indices, model_names)

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {config.output_dir}")
    print(f"Visualizations saved to: {os.path.join(config.output_dir, 'visualizations')}")
    print(f"Checkpoints saved to: {os.path.join(config.output_dir, 'checkpoints')}")

    return summary


def evaluate_model(
    generator, critic, trainer,
    data_loader, evaluator, config
):
    """Evaluate a trained model."""

    generator.eval()
    critic.eval()

    # Generate samples
    n_samples = 1000
    z = torch.randn(n_samples, config.latent_dim, device=config.device)
    group_labels = torch.randint(0, config.n_groups, (n_samples,), device=config.device)

    with torch.no_grad():
        fake_data = generator(z, group_labels)

    # Get real data
    real_data_list = []
    real_groups_list = []

    for batch in data_loader.get_test_loader():
        if len(batch) >= 2:
            real_data_list.append(batch[0])
            real_groups_list.append(batch[1])

        if sum(len(b) for b in real_data_list) >= n_samples:
            break

    real_data = torch.cat(real_data_list, dim=0)[:n_samples]
    real_groups = torch.cat(real_groups_list, dim=0)[:n_samples]

    # Evaluate
    metrics = evaluator.evaluate_all(
        real_data=real_data,
        fake_data=fake_data,
        real_groups=real_groups,
        fake_groups=group_labels
    )

    # Add privacy info if available
    if hasattr(trainer, 'get_privacy_spent'):
        epsilon, delta = trainer.get_privacy_spent()
        metrics['epsilon'] = epsilon
        metrics['delta'] = delta
    else:
        metrics['epsilon'] = 0
        metrics['delta'] = 0

    # MIA evaluation for privacy
    mia_evaluator = MIAEvaluator(critic, config.device)

    # Get train data
    train_data_list = []
    for batch in data_loader.get_train_loader():
        train_data_list.append(batch[0])
        if sum(len(b) for b in train_data_list) >= n_samples:
            break

    train_data = torch.cat(train_data_list, dim=0)[:n_samples]
    test_data = real_data[:n_samples]

    mia_metrics = mia_evaluator.evaluate_privacy(train_data, test_data, n_samples=500)
    metrics.update(mia_metrics)

    return metrics


def print_research_question_answers(all_results, pareto_indices, model_names):
    """Print answers to the four research questions."""

    print("\n** RQ1: Quantifying the Trilemma **")
    print("How do privacy and fairness constraints individually impact utility?")

    vanilla_fid = all_results.get('Vanilla-GAN', {}).get('fid', 0)
    fair_fid = all_results.get('Fair-GAN', {}).get('fid', 0)
    dp_fid = all_results.get('DP-GAN', {}).get('fid', 0)

    print(f"  Vanilla-GAN FID: {vanilla_fid:.2f} (baseline)")
    print(f"  Fair-GAN FID: {fair_fid:.2f} (fairness cost: {fair_fid - vanilla_fid:.2f})")
    print(f"  DP-GAN FID: {dp_fid:.2f} (privacy cost: {dp_fid - vanilla_fid:.2f})")

    print("\n** RQ2: Adaptive Mechanism **")
    print("Does adaptive group-aware clipping improve fairness-utility tradeoff?")

    uniform_spd = all_results.get('Uniform-FAIR-DP-GAN', {}).get('spd', 0)
    adaptive_spd = all_results.get('Adaptive-FAIR-DP-GAN', {}).get('spd', 0)

    print(f"  Uniform SPD: {uniform_spd:.4f}")
    print(f"  Adaptive SPD: {adaptive_spd:.4f} (improvement: {uniform_spd - adaptive_spd:.4f})")

    print("\n** RQ3: Optimization Strategy **")
    print("Is simultaneous optimization better than sequential?")

    adaptive_puf = all_results.get('Adaptive-FAIR-DP-GAN', {}).get('puf_score', 0)
    sequential_puf = all_results.get('Sequential-Approach', {}).get('puf_score', 0)

    print(f"  Adaptive PUF: {adaptive_puf:.4f}")
    print(f"  Sequential PUF: {sequential_puf:.4f}")

    if 'Adaptive-FAIR-DP-GAN' in [model_names[i] for i in pareto_indices]:
        print("  ✓ Adaptive approach is Pareto-optimal")

    print("\n** RQ4: Practical Utility (TSTR) **")
    print("Can synthetic data train useful classifiers?")

    for name in ['Adaptive-FAIR-DP-GAN', 'Uniform-FAIR-DP-GAN']:
        if name in all_results:
            tstr = all_results[name].get('tstr_accuracy', 0)
            print(f"  {name} TSTR: {tstr:.4f}")
