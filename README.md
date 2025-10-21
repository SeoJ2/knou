# FAIR-DP-GAN: Fairness and Differential Privacy in Generative Adversarial Networks

A comprehensive implementation of fairness-aware and privacy-preserving generative models.

## Overview

This repository implements and compares 6 different GAN approaches that balance three critical objectives:

1. **Privacy** - Differential Privacy (DP)
2. **Fairness** - Demographic parity and equal opportunity
3. **Utility** - Image quality and downstream task performance

## Models Implemented

| Model | Privacy | Fairness | Description |
|-------|---------|----------|-------------|
| **Vanilla-GAN** | ✗ | ✗ | Baseline (utility upper bound) |
| **FairGAN** | ✗ | ✓ | Fairness constraint only |
| **DP-GAN** | ✓ | ✗ | Differential privacy only |
| **Uniform-FAIR-DP-GAN** | ✓ | ✓ | Uniform DP + Fairness (baseline) |
| **Adaptive-FAIR-DP-GAN** | ✓ | ✓ | **Adaptive group-aware DP** (proposed) |
| **Sequential-Approach** | ✓ | ✓ | Two-phase: DP → Rebalancing + Fairness |

## Key Features

### 🔐 Privacy
- **Differential Privacy (DP-SGD)** with configurable epsilon budget
- **Group-Aware Adaptive Clipping** for fair privacy allocation
- **Membership Inference Attack (MIA)** evaluation

### ⚖️ Fairness
- **Statistical Parity Difference (SPD)**
- **Disparate Impact (DI)**
- **Equalized Odds Difference (EOD)**
- Pre-trained attribute classifier for downstream label prediction

### 📊 Utility
- **Fréchet Inception Distance (FID)**
- **Inception Score (IS)**
- **Train on Synthetic, Test on Real (TSTR)**

## Project Structure

```
knou/
├── data/                          # Data loading and preprocessing
│   ├── celeba_loader.py          # CelebA dataset with fairness annotations
│   └── __init__.py
├── models/                        # Neural network architectures
│   ├── generator.py              # DCGAN-style generator
│   ├── discriminator.py          # Standard & Fair discriminators
│   ├── attribute_classifier.py   # Multi-label classifier for attributes
│   └── __init__.py
├── optimizers/                    # Custom optimizers
│   ├── group_aware_dpsgd.py     # Adaptive DP-SGD with group awareness
│   └── __init__.py
├── trainers/                      # Training algorithms
│   ├── vanilla_gan.py            # Standard GAN
│   ├── fair_gan.py               # Fairness-constrained GAN
│   ├── dp_gan.py                 # DP-GAN
│   ├── uniform_fair_dp_gan.py    # Uniform DP + Fairness
│   ├── adaptive_fair_dp_gan.py   # Adaptive DP + Fairness (proposed)
│   ├── sequential_trainer.py     # Two-phase training
│   └── __init__.py
├── metrics/                       # Evaluation metrics
│   ├── privacy_metrics.py        # Epsilon, MIA
│   ├── fairness_metrics.py       # SPD, DI, EOD
│   ├── utility_metrics.py        # FID, IS, TSTR
│   └── __init__.py
├── utils/                         # Utilities
│   ├── config.py                 # Configuration settings
│   └── __init__.py
├── experiments/                   # Experiment scripts
│   └── run_experiment.py         # Main training and evaluation script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

```bash
# Clone repository
git clone https://github.com/SeoJ2/knou.git
cd knou

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

Run all experiments with default settings:

```bash
python experiments/run_experiment.py
```

### Configuration

Edit `utils/config.py` to customize:

```python
class Config:
    # Dataset
    DATASET = "CELEBA"
    NUM_SAMPLES = 10000
    IMAGE_SIZE = 64

    # Training
    EPOCHS = 100
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0002

    # Privacy
    EPSILON_TARGET = 8.0
    DELTA = 1e-5
    MAX_GRAD_NORM = 1.0

    # Sequential Training
    PHASE1_EPOCHS = 50   # DP learning
    PHASE2_EPOCHS = 20   # Rebalancing + Fairness

    # Fairness
    SENSITIVE_ATTR = "Male"
    TARGET_ATTR = "Attractive"
    FAIRNESS_LAMBDA = 0.1
```

### Training Individual Models

```python
from utils.config import Config
from data.celeba_loader import get_celeba_dataloaders
from models import Generator, FairDiscriminator
from trainers import AdaptiveFairDPGANTrainer

# Setup
config = Config()
train_loader, test_loader, _ = get_celeba_dataloaders(config)

# Create models
generator = Generator(config)
discriminator = FairDiscriminator(config)

# Train
trainer = AdaptiveFairDPGANTrainer(generator, discriminator, config)
trainer.train(train_loader)

# Save
trainer.save_models('./checkpoints/adaptive_fair_dp_gan')
```

## Experimental Setup

- **Dataset**: CelebA (10,000 samples)
- **Image Resolution**: 64×64
- **Epochs**: 100 (Phase 1: 50, Phase 2: 20 for Sequential)
- **Batch Size**: 16
- **Target Epsilon**: 8.0
- **Sensitive Attribute**: Male/Female
- **Target Attribute**: Attractive

## Key Implementation Details

### 1. Adaptive Group-Aware DP-SGD

The proposed method uses adaptive per-group gradient clipping:

```python
# Adaptive clipping norm for group g:
C_g = C_base * (1 + λ * (ratio_g - 1/K))

# Where:
# - C_base: base clipping norm
# - ratio_g: proportion of group g in batch
# - K: number of groups
# - λ: adaptation strength (default: 0.5)
```

### 2. Sequential Training (Phase 2)

Phase 2 implements **rebalancing** and **fairness fine-tuning**:

1. **Data Rebalancing**: Equal samples from each sensitive group
2. **Fairness Loss**: Adversarial fairness head
3. **Lower Learning Rate**: 0.1× of Phase 1 for stable fine-tuning

### 3. Attribute Classifier

Pre-trained on **real data** to generate accurate downstream labels:
- 40 CelebA attributes
- BCE loss for multi-label classification
- Used for TSTR and fairness evaluation

## Evaluation Metrics

### Privacy
- **Epsilon (ε)**: Privacy budget (lower is better)
- **MIA Accuracy**: Membership inference attack success rate (lower is better)

### Fairness
- **SPD** (Statistical Parity Difference): |P(Y=1|S=0) - P(Y=1|S=1)| → 0
- **DI** (Disparate Impact): |1 - P(Y=1|S=0)/P(Y=1|S=1)| → 0
- **EOD** (Equalized Odds Difference): TPR + FPR differences → 0

### Utility
- **FID** (Fréchet Inception Distance): Lower is better
- **IS** (Inception Score): Higher is better
- **TSTR**: Downstream classifier accuracy

## Results Format

Results are saved in JSON format:

```json
{
  "Adaptive-FAIR-DP-GAN": {
    "fairness": {
      "spd": 0.0234,
      "di": 0.0156,
      "eod": 0.0421
    },
    "utility": {
      "fid": 45.23,
      "is_mean": 2.14,
      "is_std": 0.12
    },
    "privacy": {
      "epsilon_theoretical": 8.0,
      "mia_accuracy": 52.3,
      "privacy_score": 47.7
    }
  }
}
```

## Expected Outcomes

Based on the experimental design, expected performance:

| Model | FID ↓ | SPD ↓ | MIA ↓ | Notes |
|-------|-------|-------|-------|-------|
| Vanilla-GAN | **Best** | Worst | Worst | Utility upper bound |
| FairGAN | Good | **Best** | Worst | No privacy |
| DP-GAN | Moderate | Worst | **Best** | No fairness |
| Uniform-FAIR-DP-GAN | Moderate | Good | Good | Baseline |
| **Adaptive-FAIR-DP-GAN** | **Good** | **Good** | **Good** | **Proposed** |
| Sequential | Good | Good | Good | Alternative |

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{fair-dp-gan-2025,
  title={FAIR-DP-GAN: Balancing Fairness, Privacy, and Utility in Generative Models},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/SeoJ2/knou}
}
```

## License

MIT License

## Acknowledgments

- CelebA dataset: [Liu et al., 2015]
- Opacus library for DP-SGD
- PyTorch and torchvision

## Contact

For questions or issues, please open an issue on GitHub.

---

**Status**: ✅ Complete Implementation
- All 6 models implemented
- All evaluation metrics implemented
- Sequential Phase 2 with rebalancing
- Attribute classifier for accurate downstream labels
- Group-aware adaptive DP-SGD
