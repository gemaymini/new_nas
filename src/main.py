# -*- coding: utf-8 -*-
"""
Entry point for the evolutionary NAS search.
"""
import argparse
import torch
import sys
import random
import numpy as np
from pathlib import Path

# Add project root to python path so `src` is importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from configuration.config import config
from search.evolution import AgingEvolutionNAS
from utils.logger import logger, tb_logger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Evolutionary NAS")
    parser.add_argument(
        "--dataset",
        type=str,
        default=config.FINAL_DATASET,
        choices=["cifar10", "cifar100", "imagenet"],
        help="Dataset for training and evaluation (default: cifar10)",
    )
    parser.add_argument(
        "--imagenet_root",
        type=str,
        default=config.IMAGENET_ROOT,
        help="Path to ImageNet dataset root directory",
    )

    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_final_eval", action="store_true")

    return parser.parse_args()


def main():
    logger.setup_file_logging()
    tb_logger.setup()

    args = parse_args()
    set_seed(args.seed)
    config.FINAL_DATASET = args.dataset

    if args.dataset == "imagenet":
        config.IMAGENET_ROOT = args.imagenet_root
        config.NTK_NUM_CLASSES = config.IMAGENET_NUM_CLASSES
        config.NTK_INPUT_SIZE = (3, config.IMAGENET_INPUT_SIZE, config.IMAGENET_INPUT_SIZE)
        config.BATCH_SIZE = config.IMAGENET_BATCH_SIZE
        config.INPUT_IMAGE_SIZE = config.IMAGENET_INPUT_SIZE
        config.INIT_CONV_KERNEL_SIZE = 7
        config.INIT_CONV_STRIDE = 2
        config.INIT_CONV_PADDING = 3
    elif args.dataset == "cifar100":
        config.NTK_NUM_CLASSES = 100
    else:
        config.NTK_NUM_CLASSES = 10

    logger.info(f"Dataset: {config.FINAL_DATASET}, Num Classes: {config.NTK_NUM_CLASSES}")
    logger.info(f"Input Size: {config.NTK_INPUT_SIZE}")

    from engine.evaluator import fitness_evaluator
    fitness_evaluator.reset()

    if config.DEVICE == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        config.DEVICE = "cpu"

    nas = AgingEvolutionNAS()
    if args.resume:
        try:
            nas.load_checkpoint(args.resume)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            sys.exit(1)

    try:
        nas.run_search()
        nas.run_screening_and_training()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        nas.save_checkpoint()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Evolution failed: {e}")
        nas.save_checkpoint()
        raise


if __name__ == "__main__":
    print(PROJECT_ROOT)
    main()
