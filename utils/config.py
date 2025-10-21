"""
Configuration file for FAIR-DP-GAN experiments
"""
import torch

class Config:
    """Experimental configuration"""

    # Dataset settings
    DATASET = "CELEBA"
    NUM_SAMPLES = 10000
    IMAGE_SIZE = 64
    NUM_CHANNELS = 3

    # Training settings
    EPOCHS = 100
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0002
    BETA1 = 0.5
    BETA2 = 0.999

    # Privacy settings
    EPSILON_TARGET = 8.0
    DELTA = 1e-5
    MAX_GRAD_NORM = 1.0

    # Sequential trainer settings
    PHASE1_EPOCHS = 50  # DP learning
    PHASE2_EPOCHS = 20  # Rebalancing + fairness

    # Model architecture
    LATENT_DIM = 100
    GEN_FEATURES = 64
    DISC_FEATURES = 64

    # Fairness settings
    SENSITIVE_ATTR = "Male"  # Binary sensitive attribute
    TARGET_ATTR = "Attractive"  # Target attribute for fairness
    FAIRNESS_LAMBDA = 0.1  # Fairness loss weight

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    DATA_DIR = "./datasets/celeba"
    OUTPUT_DIR = "./outputs"
    CHECKPOINT_DIR = "./checkpoints"

    # Evaluation settings
    FID_BATCH_SIZE = 50
    NUM_GENERATED_SAMPLES = 10000
    MIA_SHADOW_MODELS = 5

    # Group-aware DPSGD
    USE_ADAPTIVE_CLIPPING = True
    MIN_GROUP_SIZE = 100  # Minimum group size for adaptive clipping

    def __repr__(self):
        return f"Config(dataset={self.DATASET}, samples={self.NUM_SAMPLES}, " \
               f"epochs={self.EPOCHS}, epsilon={self.EPSILON_TARGET})"
