# -*- coding: utf-8 -*-
"""
Neural architecture search configuration.
Holds all hyperparameter defaults.
"""
import os
import random


class Config:
    """Configuration container for all hyperparameters."""

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    # ==================== Evolution parameters ====================
    POPULATION_SIZE = 5          # Aging Evolution queue size
    MAX_GEN = 10                 # Total individuals evaluated in search
    TOURNAMENT_SIZE = 3            # Tournament sample size
    TOURNAMENT_WINNERS = 2         # Tournament winners (parent count)

    # ==================== Screening/training pipeline ====================
    HISTORY_TOP_N1 = 1            # Stage 1: Top N1 by NTK
    SHORT_TRAIN_EPOCHS = 5        # Stage 1 short training epochs
    HISTORY_TOP_N2 = 1             # Stage 2: Top N2 by validation accuracy
    FULL_TRAIN_EPOCHS = 5        # Final training epochs

    # ==================== Crossover/mutation ====================
    PROB_CROSSOVER = 0.5           # Crossover probability
    PROB_MUTATION = 0.5            # Mutation probability

    # ==================== Search space ====================
    MIN_UNIT_NUM = 2               # Min unit count
    MAX_UNIT_NUM = 4               # Max unit count
    MIN_BLOCK_NUM = 3              # Min blocks per unit
    MAX_BLOCK_NUM = 4              # Max blocks per unit

    CHANNEL_OPTIONS = [64, 128, 256]
    GROUP_OPTIONS = [4, 8, 16]
    POOL_TYPE_OPTIONS = [0, 1]
    POOL_STRIDE_OPTIONS = [1, 2]
    SENET_OPTIONS = [0, 1]

    # Activation types: 0=ReLU, 1=SiLU
    ACTIVATION_OPTIONS = [0, 1]
    # Dropout options
    DROPOUT_OPTIONS = [0.0, 0.1]
    # Skip connection types: 0=add, 1=concat, 2=none
    SKIP_TYPE_OPTIONS = [0, 1, 2]
    # Convolution kernel size options
    KERNEL_SIZE_OPTIONS = [3, 5]
    # Block expansion options (out_channels = mid_channels * expansion)
    EXPANSION_OPTIONS = [1, 2]

    # Initial stem convolution
    INIT_CONV_OUT_CHANNELS = 64
    INIT_CONV_KERNEL_SIZE = 3
    INIT_CONV_STRIDE = 1
    INIT_CONV_PADDING = 1

    # ==================== Mutation operator rates ====================
    PROB_SWAP_BLOCKS = 0.8
    PROB_SWAP_UNITS = 0.8
    PROB_ADD_UNIT = 0.4
    PROB_ADD_BLOCK = 0.6
    PROB_DELETE_UNIT = 0.4
    PROB_DELETE_BLOCK = 0.6
    PROB_MODIFY_BLOCK = 0.8

    # ==================== NTK evaluation ====================
    NTK_BATCH_SIZE = 64
    NTK_INPUT_SIZE = (3, 32, 32)
    NTK_NUM_CLASSES = 10
    NTK_PARAM_THRESHOLD = 10000000  # Skip overly large models

    # ==================== Training ====================
    DEVICE = "cuda"
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    ADAMW_BETAS = (0.9, 0.999)
    ADAMW_EPS = 1e-8

    # ==================== Early stopping ====================
    EARLY_STOPPING_ENABLED = True   # Enable early stopping
    EARLY_STOPPING_PATIENCE = 50    # Epochs without improvement before stop
    EARLY_STOPPING_MIN_DELTA = 0.01 # Minimum improvement (%) to reset patience

    # ==================== ImageNet ====================
    IMAGENET_ROOT = os.path.join(DATA_DIR, "imagenet")  # Dataset root
    IMAGENET_BATCH_SIZE = 64          # Batch size (memory-aware)
    IMAGENET_INPUT_SIZE = 224         # Input resolution
    IMAGENET_NUM_CLASSES = 1000       # Class count

    # ==================== Final evaluation ====================
    FINAL_DATASET = "cifar10"

    # ==================== SENet ====================
    SENET_REDUCTION = 16

    # ==================== Logging ====================
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    LOG_LEVEL = "INFO"
    SAVE_CHECKPOINT = True
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

    # ==================== TensorBoard ====================
    USE_TENSORBOARD = True
    TENSORBOARD_DIR = os.path.join(BASE_DIR, "runs")

    # ==================== Debugging ====================
    SAVE_FAILED_INDIVIDUALS = True
    FAILED_INDIVIDUALS_DIR = os.path.join(BASE_DIR, "failed_individuals")

    # ==================== Architecture constraints ====================
    MIN_FEATURE_SIZE = 1
    INPUT_IMAGE_SIZE = 32
    MAX_CHANNELS = 1024  # Prevent channel blow-up in concat mode

    # ==================== Misc ====================
    RANDOM_SEED = random.randint(0, 2**32 - 1)
    NUM_WORKERS = 8

    def get_search_space_summary(self) -> str:
        """Return a short search-space summary string."""
        # Simplified for now, can be expanded
        return "Search Space Summary"


# Global config instance
config = Config()
