"""
Quick test to verify the installation and imports
"""
import sys
import torch
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\nTesting imports...")

try:
    from utils.config import Config
    print("✓ Config imported")

    from data.celeba_loader import CelebADataset
    print("✓ CelebA loader imported")

    from models import Generator, Discriminator, FairDiscriminator, AttributeClassifier
    print("✓ Models imported")

    from optimizers import GroupAwareDPSGD, SimpleGroupAwareDPSGD
    print("✓ Optimizers imported")

    from trainers import (
        VanillaGANTrainer,
        FairGANTrainer,
        DPGANTrainer,
        UniformFairDPGANTrainer,
        AdaptiveFairDPGANTrainer,
        SequentialTrainer
    )
    print("✓ All trainers imported")

    from metrics import (
        evaluate_fairness,
        evaluate_utility,
        evaluate_privacy
    )
    print("✓ Metrics imported")

    print("\n✅ All imports successful!")

    # Test model creation
    print("\nTesting model creation...")
    config = Config()
    generator = Generator(config)
    discriminator = Discriminator(config)
    fair_discriminator = FairDiscriminator(config)
    classifier = AttributeClassifier(config)

    print(f"✓ Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"✓ Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    print(f"✓ Fair Discriminator parameters: {sum(p.numel() for p in fair_discriminator.parameters()):,}")
    print(f"✓ Classifier parameters: {sum(p.numel() for p in classifier.parameters()):,}")

    print("\n✅ All tests passed! Ready to run experiments.")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
