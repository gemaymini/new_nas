# -*- coding: utf-8 -*-
"""
从Checkpoint加载种群并训练Top-K模型
"""
import argparse
import pickle
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from configuration.config import config
from engine.evaluator import FinalEvaluator
from utils.logger import logger, tb_logger

def train_from_checkpoint(checkpoint_path: str, top_k: int = None, epochs: int = None):
    """
    加载Checkpoint并对其中表现最好的K个个体进行最终训练
    """
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return

    # 1. Setup logging (ensure we can see output)
    logger.setup_file_logging()
    tb_logger.setup()

    print(f"Loading checkpoint from {checkpoint_path}...")
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    if 'population' not in checkpoint:
        print("Error: Invalid checkpoint format. 'population' key missing.")
        return

    population = checkpoint['population']
    current_gen = checkpoint.get('current_gen', 'Unknown')
    print(f"Loaded population size: {len(population)}, Generation: {current_gen}")

    # Use config values if not provided
    if top_k is None:
        top_k = config.HISTORY_TOP_N2
    if epochs is None:
        epochs = config.FULL_TRAIN_EPOCHS

    print(f"Configuration:")
    print(f"- Top K: {top_k}")
    print(f"- Train Epochs: {epochs}")
    print(f"- Dataset: {config.FINAL_DATASET}")
    
    # 2. Initialize FinalEvaluator
    try:
        evaluator = FinalEvaluator(dataset=config.FINAL_DATASET)
    except Exception as e:
        print(f"Error initializing evaluator: {e}")
        return

    # 3. Evaluate Top Individuals
    print("Starting evaluation...")
    best_ind, results = evaluator.evaluate_top_individuals(
        population, top_k=top_k, epochs=epochs
    )

    print("\n" + "="*50)
    print("Training Completed")
    print("="*50)
    if best_ind and results:
        # 从 results 中获取最佳准确率
        best_result = max(results, key=lambda x: x.get('best_accuracy', 0))
        best_accuracy = best_result.get('best_accuracy', 0)
        print(f"Best Individual ID: {best_ind.id}")
        print(f"Best Accuracy: {best_accuracy:.2f}%")
        print(f"Model saved in: checkpoints/final_models/")
    else:
        print("No individuals were successfully trained.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Top-K models from a checkpoint')
    parser.add_argument('checkpoint_path', type=str, help='Path to the .pkl checkpoint file')
    parser.add_argument('--top_k', type=int, default=None, help=f'Number of top individuals to train (default: {config.HISTORY_TOP_N2})')
    parser.add_argument('--epochs', type=int, default=None, help=f'Number of epochs to train (default: {config.FULL_TRAIN_EPOCHS})')
    
    args = parser.parse_args()
    
    train_from_checkpoint(args.checkpoint_path, args.top_k, args.epochs)
