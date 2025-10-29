"""
Fair-DP-GAN: Fair Differentially Private Generative Adversarial Network

This package implements the Fair-DP-GAN framework for generating synthetic data
that balances three competing objectives:
1. Utility (data quality)
2. Privacy (differential privacy)
3. Fairness (demographic parity)

Core Components:
- Data processing: CelebADataLoader, AdultDataLoader
- Privacy mechanisms: RDPAccountant, GroupAwareDPSGD
- Model architectures: Generator, Critic
- Training strategies: 6 different trainers
- Evaluation: Comprehensive metrics and MIA
- Analysis: Pareto frontier analysis and visualization
"""

__version__ = "1.0.0"
__author__ = "Fair-DP-GAN Research Team"

from fair_dp_gan.data import CelebADataLoader, AdultDataLoader
from fair_dp_gan.privacy import RDPAccountant, GroupAwareDPSGD
from fair_dp_gan.models import Generator, Critic
from fair_dp_gan.evaluation import Evaluator, MIAEvaluator
from fair_dp_gan.analysis import ParetoAnalyzer, Visualizer
from fair_dp_gan.utils import CheckpointManager

__all__ = [
    'CelebADataLoader',
    'AdultDataLoader',
    'RDPAccountant',
    'GroupAwareDPSGD',
    'Generator',
    'Critic',
    'Evaluator',
    'MIAEvaluator',
    'ParetoAnalyzer',
    'Visualizer',
    'CheckpointManager',
]
