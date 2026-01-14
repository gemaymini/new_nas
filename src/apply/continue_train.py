# -*- coding: utf-8 -*-
"""
Continue training from a saved checkpoint.
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
        print(f"ERROR: model file not found: {model_path}")
        return

    print(f"INFO: loading_model={model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=config.DEVICE)
    except Exception as e:
        print(f"ERROR: failed to load checkpoint: {e}")
        return

    # 1. Restore Architecture
    if 'encoding' not in checkpoint:
        print("ERROR: encoding missing; cannot reconstruct network")
        return
    
    encoding = checkpoint['encoding']
    print(f"INFO: encoding={encoding}")
    
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
        print("INFO: weights loaded")
    else:
        print("WARN: no state_dict; training from scratch")

    # 3. Setup Trainer
    trainer = NetworkTrainer(config.DEVICE)
    
    # Optional: override LR if provided
    # Note: NetworkTrainer.train_network takes lr argument.
    
    print(f"INFO: training epochs={epochs} dataset={config.FINAL_DATASET}")
    optimizer_name = config.OPTIMIZER.lower()
    opt_defaults = config.get_optimizer_params(optimizer_name)
    lr_display = lr if lr is not None else opt_defaults["lr"]
    hyper_parts = [
        f"optimizer={optimizer_name}",
        f"lr={lr_display}",
        f"weight_decay={opt_defaults['weight_decay']}",
    ]
    if optimizer_name in ("adamw", "adam", "radam"):
        hyper_parts.append(f"betas={opt_defaults.get('betas')}")
        hyper_parts.append(f"eps={opt_defaults.get('eps', config.ADAMW_EPS)}")
    elif optimizer_name == "sgd":
        hyper_parts.append(f"momentum={opt_defaults.get('momentum')}")
        hyper_parts.append(f"nesterov={opt_defaults.get('nesterov')}")
    elif optimizer_name == "rmsprop":
        hyper_parts.append(f"alpha={opt_defaults.get('alpha')}")
        hyper_parts.append(f"momentum={opt_defaults.get('momentum')}")
        hyper_parts.append(f"eps={opt_defaults.get('eps', config.ADAMW_EPS)}")
    hyper_parts.append(f"warmup_epochs={opt_defaults.get('warmup_epochs', 0)}")
    print("INFO: hyperparams " + " ".join(hyper_parts))
    
    # 4. Train
    best_acc, history = trainer.train_network(
        network,
        trainloader,
        testloader,
        epochs=epochs,
        lr=lr,
        optimizer_name=optimizer_name,
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
        'history': full_history
    }
    
    torch.save(save_dict, new_save_path)
    print("INFO: training complete")
    print(f"INFO: best_acc={best_acc:.2f}%")
    print(f"INFO: saved={new_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Continue training a model from .pth file')
    parser.add_argument('model_path', type=str, help='Path to the .pth model file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of additional epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (default: optimizer preset)')
    parser.add_argument(
        '--optimizer',
        type=str,
        default=None,
        choices=config.OPTIMIZER_OPTIONS,
        help='Optimizer to use (default: config value)'
    )
    
    args = parser.parse_args()
    
    if args.optimizer is not None:
        config.OPTIMIZER = args.optimizer
    continue_training(args.model_path, args.epochs, args.lr)
