# -*- coding: utf-8 -*-
"""
Test NTK condition number calculation on standard models to verify logic.
"""
import torch
import torch.nn as nn
import torchvision.models as models
import sys
import os
import traceback

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configuration.config import config
from engine.evaluator import NTKEvaluator

def get_cifar_compatible_resnet18():
    # ResNet18 adapted for CIFAR-10 (small input size)
    # Standard ResNet18 has 7x7 conv stride 2, then maxpool stride 2.
    # For CIFAR (32x32), we usually replace the first conv with 3x3 stride 1 and remove maxpool.
    model = models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

def get_simple_cnn():
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64 * 32 * 32, 10)
    )

def get_cifar_compatible_mobilenet_v2():
    # MobileNetV2 usually works fine with varying input sizes due to Global Avg Pooling
    # But we can change the first stride to 1 to preserve more spatial info for 32x32
    model = models.mobilenet_v2(num_classes=10)
    # First layer is features[0][0] which is Conv2dNormActivation -> Conv2d
    model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    return model

def get_cifar_compatible_densenet121():
    model = models.densenet121(num_classes=10)
    # Adapt first conv and remove maxpool
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.features.pool0 = nn.Identity()
    return model

def get_cifar_compatible_resnet50():
    model = models.resnet50(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

def main():
    print("Initializing NTKEvaluator with num_batch=1...")
    # Explicitly requesting num_batch=1 to save memory
    try:
        evaluator = NTKEvaluator(dataset="cifar10", num_batch=1)
    except Exception as e:
        print(f"Failed to initialize evaluator: {e}")
        return

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    models_to_test = {
        "SimpleCNN": get_simple_cnn(),
        "ResNet18 (CIFAR adapted)": get_cifar_compatible_resnet18(),
        "MobileNetV2 (CIFAR adapted)": get_cifar_compatible_mobilenet_v2(),
        "DenseNet121 (CIFAR adapted)": get_cifar_compatible_densenet121(),
        "ResNet50 (CIFAR adapted)": get_cifar_compatible_resnet50(),
    }

    print("\nStarting NTK Condition Number Tests...\n")
    print(f"{'Model Name':<30} | {'Params':<15} | {'NTK Score ':<25}")
    print("-" * 80)

    for name, model in models_to_test.items():
        try:
            model = model.to(device)
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Compute NTK
            # compute_ntk_score averages multiple runs
            score = evaluator.compute_ntk_score(model, num_runs=1)
            
            print(f"{name:<30} | {num_params:<15} | {score:<25}")
            
        except Exception as e:
            print(f"{name:<30} | {'ERROR':<15} | {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
