# -*- coding: utf-8 -*-
"""
神经网络架构搜索算法 - 主程序入口
"""
import argparse
import torch
import sys
import os
import random
import numpy as np

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from configuration.config import config
from search.evolution import AgingEvolutionNAS
from core.encoding import Encoder
from models.network import NetworkBuilder
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
    parser = argparse.ArgumentParser(description='Evolutionary NAS')
    
    # Evolution params
    parser.add_argument('--population_size', type=int, default=config.POPULATION_SIZE)
    parser.add_argument('--max_gen', type=int, default=config.MAX_GEN)
    
    # Dataset params
    parser.add_argument('--dataset', type=str, default=config.FINAL_DATASET,
                        choices=['cifar10', 'cifar100'],
                        help='Dataset for training and evaluation (default: cifar10)')
    
    # Other params
    parser.add_argument('--seed', type=int, default=config.RANDOM_SEED)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--no_final_eval', action='store_true')
    
    return parser.parse_args()

def main():
    # Setup logging
    logger.setup_file_logging()
    tb_logger.setup()

    args = parse_args()
    set_seed(args.seed)
    config.POPULATION_SIZE = args.population_size
    config.MAX_GEN = args.max_gen
    config.FINAL_DATASET = args.dataset
    
    # Update NTK_NUM_CLASSES based on dataset
    if args.dataset == 'cifar100':
        config.NTK_NUM_CLASSES = 100
    else:
        config.NTK_NUM_CLASSES = 10
    
    logger.info(f"Dataset: {config.FINAL_DATASET}, Num Classes: {config.NTK_NUM_CLASSES}")
    
    if config.DEVICE == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        config.DEVICE = 'cpu'

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

if __name__ == '__main__':
    main()
