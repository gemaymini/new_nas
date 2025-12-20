# -*- coding: utf-8 -*-
"""
神经网络架构搜索算法 - 主程序入口
"""
import argparse
import torch
import sys
import random
import numpy as np
from utils.config import config
from search.evolution import AgingEvolutionNAS
from core.encoding import Encoder
from model.network import NetworkBuilder
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
    
    # Other params
    parser.add_argument('--seed', type=int, default=config.RANDOM_SEED)
    parser.add_argument('--device', type=str, default=config.DEVICE)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--no_final_eval', action='store_true')
    
    return parser.parse_args()

def test_network_building():
    logger.info("Testing network building...")
    encoding = Encoder.create_random_encoding()
    logger.info(f"Random encoding: {encoding}")
    is_valid = Encoder.validate_encoding(encoding)
    logger.info(f"Valid: {is_valid}")
    if is_valid:
        success = NetworkBuilder.test_forward(encoding)
        logger.info(f"Forward test: {'PASSED' if success else 'FAILED'}")
        if success:
            param_count = NetworkBuilder.calculate_param_count(encoding)
            logger.info(f"Params: {param_count}")
            Encoder.print_architecture(encoding)

def main():
    # Setup logging
    logger.setup_file_logging()
    tb_logger.setup()

    args = parse_args()
    set_seed(args.seed)
    
    # Update config
    config.POPULATION_SIZE = args.population_size
    config.MAX_GEN = args.max_gen
    config.DEVICE = args.device
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        config.DEVICE = 'cpu'
        
    if args.test:
        logger.info("=== TEST MODE ===")
        test_network_building()
        # Test short run
        config.POPULATION_SIZE = 10
        config.MAX_GEN = 20
        nas = AgingEvolutionNAS()
        nas.run_search()
        return

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
