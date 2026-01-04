# -*- coding: utf-8 -*-
"""
Continue training a model from a .pth checkpoint.
"""
import argparse
import os
import sys
import torch

# Add src to path (apply is now under src/apply/)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configuration.config import config
from models.network import NetworkBuilder
from core.encoding import Individual
from engine.trainer import NetworkTrainer
from data.dataset import DatasetLoader
from utils.logger import logger

def continue_training(model_path: str, epochs: int, lr: float = None):
    """
    Load a model from .pth and continue training.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return

    print(f"Loading model from {model_path}...")
    try:
        checkpoint = torch.load(model_path, map_location=config.DEVICE)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # 1. Restore Architecture
    if 'encoding' not in checkpoint:
        print("Error: Checkpoint does not contain 'encoding'. Cannot reconstruct network.")
        return
    
    encoding = checkpoint['encoding']
    print(f"Model Encoding: {encoding}")
    
    # Create dummy individual for NetworkBuilder
    ind = Individual(encoding)
    
    # Determine classes based on config or inference? 
    # Usually config.FINAL_DATASET determines the dataset we train on.
    if config.FINAL_DATASET == 'cifar10':
        num_classes = 10
        trainloader, testloader = DatasetLoader.get_cifar10()
    elif config.FINAL_DATASET == 'cifar100':
        num_classes = 100
        trainloader, testloader = DatasetLoader.get_cifar100()
    else:
        raise ValueError(f"Unknown dataset: {config.FINAL_DATASET}")

    # Build Network
    network = NetworkBuilder.build_from_individual(
        ind, input_channels=3, num_classes=num_classes
    )
    
    # 2. Load Weights
    if 'state_dict' in checkpoint:
        network.load_state_dict(checkpoint['state_dict'])
        print("Weights loaded successfully.")
    else:
        print("Warning: Checkpoint does not contain 'state_dict'. Training from scratch (with restored architecture).")

    # 3. Setup Trainer
    trainer = NetworkTrainer(config.DEVICE)
    
    # Optional: override LR if provided
    # Note: NetworkTrainer.train_network takes lr argument.
    
    print(f"Starting continuation training for {epochs} epochs...")
    print(f"Dataset: {config.FINAL_DATASET}")
    print(f"Using hyperparameters from config (unless overridden):")
    print(f"  LR: {lr if lr is not None else config.LEARNING_RATE}")
    print(f"  Momentum: {config.MOMENTUM}")
    print(f"  Weight Decay: {config.WEIGHT_DECAY}")
    
    # 4. Train
    best_acc, history = trainer.train_network(
        network, trainloader, testloader, epochs=epochs, lr=lr
    )
    
    # 5. Save New Checkpoint
    save_dir = os.path.dirname(model_path)
    base_name = os.path.basename(model_path)
    name_no_ext = os.path.splitext(base_name)[0]
    
    # Append _continued
    new_save_path = os.path.join(save_dir, f"{name_no_ext}_continued_acc{best_acc:.2f}.pth")
    
    # Merge history if exists
    old_history = checkpoint.get('history', [])
    full_history = old_history + history
    
    save_dict = {
        'state_dict': network.state_dict(),
        'encoding': encoding,
        'accuracy': best_acc,
        'param_count': checkpoint.get('param_count', network.get_param_count()),
        'fitness': checkpoint.get('fitness', None),
        'ntk_score': checkpoint.get('ntk_score', None),
        'history': full_history
    }
    
    torch.save(save_dict, new_save_path)
    print(f"\nTraining completed.")
    print(f"Best Accuracy: {best_acc:.2f}%")
    print(f"New model saved to: {new_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Continue training a model from .pth file')
    parser.add_argument('model_path', type=str, help='Path to the .pth model file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of additional epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (default: use config value)')
    
    args = parser.parse_args()
    
    continue_training(args.model_path, args.epochs, args.lr)
