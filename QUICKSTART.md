# Fair-DP-GAN Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Testing the Implementation

### Option 1: Quick Test (1 minute)
Tests that all components work correctly without training:

```bash
python test_implementation.py
```

This verifies:
- ✅ All imports work
- ✅ Privacy mechanisms function
- ✅ Models can be created and run
- ✅ Trainers can process data
- ✅ Evaluators compute metrics
- ✅ Analysis tools work

### Option 2: Quick Demo (5 minutes)
Trains all models for 2 epochs on synthetic data:

```bash
python quick_demo.py
```

This demonstrates:
- ✅ Training 3 different models (Vanilla, Fair, DP)
- ✅ Computing fairness metrics
- ✅ Privacy budget tracking
- ✅ Complete training pipeline

### Option 3: Full Experiment

For full experiments with real data (requires dataset):

```bash
# With CelebA dataset (image data)
python main.py --data_type image --n_epochs 100 --data_path /path/to/celeba

# With Adult dataset (tabular data)
python main.py --data_type tabular --n_epochs 100 --data_path /path/to/adult.csv
```

**Note**: The data loaders will create synthetic data if the dataset is not found, allowing you to test the pipeline without downloading large datasets.

## Quick Experiments

### Test with fewer epochs (faster)
```bash
python main.py --n_epochs 10 --models adaptive_fair_dp vanilla
```

### Test specific models only
```bash
python main.py --models adaptive_fair_dp uniform_fair_dp --n_epochs 20
```

### Adjust privacy budget
```bash
python main.py --target_epsilon 5.0 --n_epochs 50
```

### Use CPU instead of GPU
```bash
python main.py --device cpu --n_epochs 10
```

## Programmatic Usage

```python
from fair_dp_gan.experiments import run_complete_experiment, ExperimentConfig

# Create configuration
config = ExperimentConfig(
    data_type='image',
    n_epochs=10,  # Short for testing
    batch_size=64,
    target_epsilon=10.0,
    models_to_train=['vanilla', 'adaptive_fair_dp']
)

# Run experiment
results = run_complete_experiment(config)

# Access results
print(results['results']['Adaptive-FAIR-DP-GAN']['spd'])
```

## Understanding the Output

After running experiments, you'll find:

### 1. Console Output
Shows real-time training progress and final metrics:
```
** RQ1: Quantifying the Trilemma **
  Vanilla-GAN FID: 28.3 (baseline)
  Fair-GAN FID: 35.7 (fairness cost: 7.4)
  ...
```

### 2. JSON Results
`experiments/results/experiment_summary.json` contains all metrics:
```json
{
  "results": {
    "Adaptive-FAIR-DP-GAN": {
      "fid": 39.2,
      "spd": 0.067,
      "epsilon": 10.0,
      ...
    }
  },
  "pareto_optimal": ["Adaptive-FAIR-DP-GAN", ...],
  ...
}
```

### 3. Visualizations
`experiments/results/visualizations/` contains:
- `pareto_frontier.png` - 3D plot for paper
- `training_curves.png` - Loss curves
- `fairness_comparison.png` - SPD comparison
- `metrics_heatmap.png` - All metrics

### 4. Checkpoints
`experiments/results/checkpoints/` contains trained model weights

## Troubleshooting

### "No module named 'torch'"
```bash
pip install torch torchvision
```

### "Data directory not found"
The code will automatically create synthetic data for testing. For real experiments:
```bash
# Download CelebA from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
# Or use the built-in synthetic data mode
```

### Out of memory
```bash
# Reduce batch size
python main.py --batch_size 32

# Use smaller image size
python main.py --image_size 32

# Use CPU
python main.py --device cpu
```

### Training is slow
```bash
# Reduce epochs for testing
python main.py --n_epochs 10

# Train fewer models
python main.py --models adaptive_fair_dp

# Use quick demo instead
python quick_demo.py
```

## Next Steps

1. **Test**: Run `python test_implementation.py` to verify installation
2. **Demo**: Run `python quick_demo.py` to see a quick example
3. **Experiment**: Run full experiments with `python main.py`
4. **Analyze**: Check generated plots and metrics
5. **Extend**: Modify code for your research needs

## Complete Options

See all available options:
```bash
python main.py --help
```

Key parameters:
- `--data_type`: 'image' or 'tabular'
- `--n_epochs`: Number of training epochs
- `--target_epsilon`: Privacy budget
- `--lambda_fair`: Fairness weight
- `--models`: Which models to train
- `--device`: 'cuda' or 'cpu'

## Getting Help

For issues or questions:
1. Check this quickstart guide
2. Read `README_FAIRGAN.md` for detailed documentation
3. Review `PROJECT_SUMMARY.md` for implementation details
4. Check code comments and docstrings

---

**Recommendation**: Start with `python test_implementation.py` to verify everything works, then try `python quick_demo.py` for a quick demonstration!
