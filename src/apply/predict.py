# -*- coding: utf-8 -*-
"""
Run inference from a saved model checkpoint.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.encoding import Individual
from models.network import NetworkBuilder
from utils.checkpoint import load_checkpoint, extract_encoding_from_checkpoint, extract_state_dict

from utils.common import CIFAR10_CLASSES, CIFAR100_CLASSES

import argparse

def predict_image(model_path: str, image_path: str, encoding_str: str = None, device: str = 'cpu'):
    if not os.path.exists(image_path):
        print(f"ERROR: image file not found: {image_path}")
        return

    print(f"INFO: loading_model={model_path}")
    try:
        checkpoint = load_checkpoint(model_path, device=device)
    except Exception as e:
        print(f"ERROR: failed to load checkpoint: {e}")
        return

    try:
        encoding = extract_encoding_from_checkpoint(checkpoint, encoding_str)
        if encoding_str and encoding:
             print(f"INFO: using_encoding={encoding}")
    except ValueError as e:
        print(f"ERROR: {e}")
        print("INFO: usage_example=python src/apply/predict.py model.pth image.jpg --encoding \"[3, 2, 2, 2, ...]\"")
        return

    try:
        state_dict = extract_state_dict(checkpoint)
    except ValueError:
        print("WARN: checkpoint might be raw state dict or unrecognized format")
        state_dict = checkpoint

    if isinstance(checkpoint, dict):
        acc = checkpoint.get('accuracy', 'N/A')
        print(f"INFO: checkpoint_accuracy={acc}")
        
    num_classes = 10 # Default
    # ... heuristics to determine num_classes from state_dict keys ...
    if 'fc.weight' in state_dict:
        num_classes = state_dict['fc.weight'].shape[0]
        print(f"INFO: num_classes_detected={num_classes} source=fc")
    elif 'classifier.weight' in state_dict:
        num_classes = state_dict['classifier.weight'].shape[0]
        print(f"INFO: num_classes_detected={num_classes} source=classifier")
    
    classes = None
    if num_classes == 10:
        classes = CIFAR10_CLASSES
        print("INFO: class_labels=cifar10")
    elif num_classes == 100:
        classes = CIFAR100_CLASSES
        print("INFO: class_labels=cifar100")
    else:
        print(f"WARN: unknown class count {num_classes}; using numeric labels")
        classes = [str(i) for i in range(num_classes)]

    try:
        ind = Individual(encoding)
        input_channels = 3
        
        network = NetworkBuilder.build_from_individual(
            ind, input_channels=input_channels, num_classes=num_classes
        )
        network.load_state_dict(state_dict)
        network.to(device)
        network.eval()
        print("INFO: network_ready")
    except Exception as e:
        print(f"ERROR: failed to build network: {e}")
        return

    try:
        transform = transforms.Compose([
            transforms.Resize((32, 32)), 
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"ERROR: failed to process image: {e}")
        return

    with torch.no_grad():
        outputs = network(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted_idx = torch.max(outputs, 1)
        
        idx = predicted_idx.item()
        if idx < len(classes):
            predicted_class = classes[idx]
        else:
            predicted_class = f"Class {idx}"
            
        confidence = probabilities[0][idx].item()

    print(
        f"INFO: prediction image={image_path} class={predicted_class} "
        f"confidence={confidence:.2%}"
    )
    print("INFO: top_predictions")
    top_k = min(3, num_classes)
    top_prob, top_idx = torch.topk(probabilities, top_k)
    for i in range(top_k):
        idx_val = top_idx[0][i].item()
        if idx_val < len(classes):
            cls = classes[idx_val]
        else:
            cls = f"Class {idx_val}"
        prob = top_prob[0][i].item()
        print(f"INFO: top{i+1} class={cls} prob={prob:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NAS Model Prediction')
    parser.add_argument('model_path', type=str, help='Path to model .pth file')
    parser.add_argument('image_path', type=str, help='Path to image file')
    parser.add_argument('--encoding', type=str, default=None, help='Architecture encoding list (required for legacy models)')
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    predict_image(args.model_path, args.image_path, args.encoding, device)
